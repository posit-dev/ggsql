//! Generic ODBC data source implementation
//!
//! Provides a reader for any ODBC-compatible database using runtime-loaded
//! ODBC bindings via `libloading`. The ODBC driver manager (`libodbc`) is
//! loaded on first use.

#[allow(dead_code)]
pub(crate) mod ffi;
mod snowflake;
mod wrapper;

use crate::reader::Reader;
use crate::{DataFrame, GgsqlError, Result};
use arrow::array::*;
#[cfg(test)]
use arrow::datatypes::DataType;
use ffi::*;
use std::sync::Arc;
use wrapper::{Connection, Statement};

/// Detect the backend SQL dialect from the DBMS name and connection string.
///
/// Delegates to the shared matcher in [`crate::reader::dialects`]; the
/// `Driver=` value from the ODBC connection string serves as the driver hint.
fn detect_dialect(dbms_name: Option<&str>, conn_str: &str) -> Result<Box<dyn super::SqlDialect>> {
    let driver = super::connection::extract_odbc_value(conn_str, "driver");
    super::registry::detect_or_err(dbms_name, driver.as_deref())
        .map(|d| d as Box<dyn super::SqlDialect>)
}

/// Pull ggsql-owned keys (`Dialect=<name>`) out of an ODBC connection
/// string before connecting. `Dialect=` is ggsql's escape hatch for unknown
/// backends, not a driver option — drivers reject unknown keys.
fn take_dialect_override(conn_str: &str) -> (String, Option<String>) {
    let (conn_str, ggsql) = super::connection::take_odbc_ggsql_params(conn_str);
    (conn_str, ggsql.dialect)
}

/// Generic ODBC reader implementing the `Reader` trait.
pub struct OdbcReader {
    connection: Connection,
    dialect: Box<dyn super::SqlDialect>,
    registered_tables: crate::reader::RegisteredTables,
    batch_size: usize,
}

// Safety: ODBC connections are safe to use from one thread at a time.
// The Reader trait requires &self (immutable) for execute_sql.
unsafe impl Send for OdbcReader {}

impl OdbcReader {
    /// Create a new ODBC reader from a `odbc://` connection URI.
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_str = uri
            .strip_prefix("odbc://")
            .ok_or_else(|| GgsqlError::ReaderError("ODBC URI must start with odbc://".into()))?;
        Self::from_odbc_conn_str(conn_str, None)
    }

    /// Create a new ODBC reader from a bare ODBC connection string.
    ///
    /// When `dialect` is `None`, it is detected from the DBMS name reported
    /// by the connection, falling back to the `Driver=` value in the
    /// connection string, then ANSI. Pass `Some(...)` to pin the dialect
    /// (e.g. from a backend-specific ggsql URI scheme).
    pub fn from_odbc_conn_str(
        conn_str: &str,
        dialect: Option<Box<dyn super::SqlDialect>>,
    ) -> Result<Self> {
        ffi::try_load()
            .map_err(|e| GgsqlError::ReaderError(format!("ODBC is not available: {}", e)))?;

        let mut conn_str = conn_str.to_string();

        // ggsql's `Dialect=` escape hatch: resolved here, stripped from the
        // connection string before the driver ever sees it.
        let (stripped, dialect_override) = take_dialect_override(&conn_str);
        conn_str = stripped;
        let dialect = match (dialect, dialect_override) {
            (d @ Some(_), _) => d,
            (None, Some(name)) => Some(
                crate::reader::registry::dialect_override(&name)
                    .map(|d| d as Box<dyn super::SqlDialect>)
                    .ok_or_else(|| crate::reader::registry::unknown_dialect_error(&name))?,
            ),
            (None, None) => None,
        };

        if snowflake::is_snowflake(&conn_str) {
            if let Some(resolved) = snowflake::resolve_connection_name(&conn_str) {
                conn_str = resolved;
            }
        }

        if snowflake::is_snowflake(&conn_str) && !snowflake::has_token(&conn_str) {
            if let Some(token) = snowflake::detect_workbench_token() {
                conn_str = snowflake::inject_snowflake_token(&conn_str, &token);
            }
        }

        let connection = match Connection::connect(wrapper::odbc_env()?, &conn_str) {
            Ok(c) => c,
            Err(e) => {
                // unixODBC IM005 ("Driver's SQLAllocHandle on
                // SQL_HANDLE_DBC failed"): some drivers (MonetDB) refuse
                // DBC allocation under an ODBC 3.80 environment. Retry once
                // under 3.0 semantics before giving up.
                if e.to_string().contains("IM005") {
                    Connection::connect(wrapper::odbc_env_legacy()?, &conn_str)?
                } else {
                    return Err(e);
                }
            }
        };

        let dbms_name = connection.dbms_name();

        let dialect = match dialect {
            Some(d) => d,
            None => detect_dialect(dbms_name.as_deref(), &conn_str)?,
        };

        // Oracle ODBC rejects block cursors (SQL_ATTR_ROW_ARRAY_SIZE > 1)
        // with HY090 at SQLFetch time, and the failed fetch leaves the
        // cursor unusable — fetch row-by-row instead.
        let batch_size = match &dbms_name {
            Some(name) if name.to_lowercase().contains("oracle") => 1,
            _ => BATCH_SIZE,
        };

        Ok(Self {
            connection,
            dialect,
            registered_tables: crate::reader::RegisteredTables::new(),
            batch_size,
        })
    }
}

impl Reader for OdbcReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        let cursor = self.connection.execute(sql)?;

        let Some(cursor) = cursor else {
            return Ok(DataFrame::empty());
        };

        cursor_to_dataframe(cursor, self.batch_size).map_err(|e| {
            let snippet: String = sql.chars().take(200).collect();
            GgsqlError::ReaderError(format!("{e} [statement: {snippet}]"))
        })
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        super::validate_table_name(name)?;

        if replace {
            let drop_sql = self.dialect.drop_table_sql(name);
            let _ = self.connection.execute(&drop_sql);
        }

        let schema = df.schema();
        let col_defs: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| {
                let ty = super::register_column_type(&*self.dialect, field.data_type())
                    .unwrap_or_else(|_| "TEXT".to_string());
                format!("{} {}", self.dialect.quote_ident(field.name()), ty)
            })
            .collect();
        for create_sql in self.dialect.sql_create_empty_temp_table(name, &col_defs) {
            self.connection.execute(&create_sql).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create temp table '{}': {}", name, e))
            })?;
        }

        let num_rows = df.height();
        if num_rows > 0 {
            let num_cols = df.width();
            let placeholders: Vec<&str> = vec!["?"; num_cols];
            let insert_sql = format!(
                "INSERT INTO {} VALUES ({})",
                self.dialect.quote_ident(name),
                placeholders.join(", ")
            );

            let columns = df.get_columns();
            let string_columns: Vec<Vec<Option<String>>> = columns
                .iter()
                .map(|col| {
                    (0..num_rows)
                        .map(|row| {
                            if col.is_null(row) {
                                None
                            } else {
                                Some(crate::array_util::value_to_string(col, row))
                            }
                        })
                        .collect()
                })
                .collect();

            let prepared = self.connection.prepare(&insert_sql).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to prepare INSERT for '{}': {}", name, e))
            })?;

            for row_idx in 0..num_rows {
                let row_values: Vec<Option<&[u8]>> = string_columns
                    .iter()
                    .map(|col| col[row_idx].as_ref().map(|s| s.as_bytes()))
                    .collect();

                let mut indicators: Vec<SqlLen> = row_values
                    .iter()
                    .map(|v| match v {
                        Some(bytes) => bytes.len() as SqlLen,
                        None => SQL_NULL_DATA,
                    })
                    .collect();

                for (col_idx, value) in row_values.iter().enumerate() {
                    let (ptr, len) = match value {
                        Some(bytes) => (bytes.as_ptr(), bytes.len() as SqlLen),
                        None => (std::ptr::null(), 0),
                    };
                    unsafe {
                        prepared.bind_text_parameter(
                            (col_idx + 1) as u16,
                            ptr,
                            len,
                            &mut indicators[col_idx],
                        )?;
                    }
                }

                prepared.execute().map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to insert row {} into '{}': {}",
                        row_idx, name, e
                    ))
                })?;

                prepared.reset_params()?;
            }
        }

        self.registered_tables.note_registered(name);
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        if !self.registered_tables.is_registered(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        let sql = self.dialect.drop_table_sql(name);
        self.connection.execute(&sql).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        self.registered_tables.note_unregistered(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<super::Spec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &*self.dialect
    }

    fn list_catalogs(&self) -> Result<Vec<String>> {
        // ODBC spec: CatalogName="%", SchemaName="", TableName=""
        let stmt = wrapper::sql_tables(&self.connection, Some("%"), Some(""), Some(""), None)?;
        let df = cursor_to_dataframe(stmt, self.batch_size)?;
        let mut catalogs = extract_string_column_ci(&df, "TABLE_CAT")?;
        catalogs.sort();
        catalogs.dedup();
        Ok(catalogs)
    }

    fn list_schemas(&self, _catalog: &str) -> Result<Vec<String>> {
        // ODBC spec: CatalogName="", SchemaName="%", TableName=""
        let stmt = wrapper::sql_tables(&self.connection, Some(""), Some("%"), Some(""), None)?;
        let df = cursor_to_dataframe(stmt, self.batch_size)?;
        let mut schemas = extract_string_column_ci(&df, "TABLE_SCHEM")?;
        schemas.sort();
        schemas.dedup();
        Ok(schemas)
    }

    fn list_tables(&self, catalog: &str, schema: &str) -> Result<Vec<super::TableInfo>> {
        let cat = if catalog.is_empty() {
            None
        } else {
            Some(catalog)
        };
        let sch = if schema.is_empty() {
            None
        } else {
            Some(schema)
        };
        let stmt = wrapper::sql_tables(&self.connection, cat, sch, Some("%"), Some("TABLE,VIEW"))?;
        let df = cursor_to_dataframe(stmt, self.batch_size)?;
        extract_table_infos_ci(&df)
    }

    fn list_columns(
        &self,
        catalog: &str,
        schema: &str,
        table: &str,
    ) -> Result<Vec<super::ColumnInfo>> {
        let cat = if catalog.is_empty() {
            None
        } else {
            Some(catalog)
        };
        let sch = if schema.is_empty() {
            None
        } else {
            Some(schema)
        };
        let stmt = wrapper::sql_columns(&self.connection, cat, sch, Some(table), None)?;
        let df = cursor_to_dataframe(stmt, self.batch_size)?;
        extract_column_infos_ci(&df)
    }
}

/// Find a column in a DataFrame by name (case-insensitive).
fn find_column_ci<'a>(df: &'a DataFrame, name: &str) -> Option<&'a ArrayRef> {
    let lower = name.to_lowercase();
    let schema = df.schema();
    for (i, field) in schema.fields().iter().enumerate() {
        if field.name().to_lowercase() == lower {
            return Some(&df.get_columns()[i]);
        }
    }
    None
}

fn extract_string_column_ci(df: &DataFrame, col_name: &str) -> Result<Vec<String>> {
    let col = find_column_ci(df, col_name).ok_or_else(|| {
        GgsqlError::ReaderError(format!("Column '{}' not found in ODBC result", col_name))
    })?;
    let mut results = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        if !col.is_null(i) {
            results.push(crate::array_util::value_to_string(col, i));
        }
    }
    Ok(results)
}

fn extract_table_infos_ci(df: &DataFrame) -> Result<Vec<super::TableInfo>> {
    let name_col = find_column_ci(df, "TABLE_NAME").ok_or_else(|| {
        GgsqlError::ReaderError("Column 'TABLE_NAME' not found in ODBC result".into())
    })?;
    let type_col = find_column_ci(df, "TABLE_TYPE").ok_or_else(|| {
        GgsqlError::ReaderError("Column 'TABLE_TYPE' not found in ODBC result".into())
    })?;
    let mut results = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        if !name_col.is_null(i) {
            results.push(super::TableInfo {
                name: crate::array_util::value_to_string(name_col, i),
                table_type: crate::array_util::value_to_string(type_col, i),
            });
        }
    }
    Ok(results)
}

fn extract_column_infos_ci(df: &DataFrame) -> Result<Vec<super::ColumnInfo>> {
    let name_col = find_column_ci(df, "COLUMN_NAME").ok_or_else(|| {
        GgsqlError::ReaderError("Column 'COLUMN_NAME' not found in ODBC result".into())
    })?;
    let type_col = find_column_ci(df, "TYPE_NAME").ok_or_else(|| {
        GgsqlError::ReaderError("Column 'TYPE_NAME' not found in ODBC result".into())
    })?;
    let mut results = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        if !name_col.is_null(i) {
            results.push(super::ColumnInfo {
                name: crate::array_util::value_to_string(name_col, i),
                data_type: crate::array_util::value_to_string(type_col, i),
            });
        }
    }
    Ok(results)
}

// ============================================================================
// Column builder (accumulates typed values across batches)
// ============================================================================

enum ColumnBuilder {
    Int8(Vec<Option<i8>>),
    Int16(Vec<Option<i16>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    Date(Vec<Option<i32>>),
    Time(Vec<Option<i64>>),
    Timestamp(Vec<Option<i64>>),
    Text(Vec<Option<String>>),
}

impl ColumnBuilder {
    fn from_sql_type(
        sql_type: SqlSmallInt,
        _col_size: SqlULen,
        _decimal_digits: SqlSmallInt,
    ) -> Self {
        match sql_type {
            SQL_TINYINT => Self::Int8(Vec::new()),
            SQL_SMALLINT => Self::Int16(Vec::new()),
            SQL_INTEGER => Self::Int32(Vec::new()),
            SQL_BIGINT => Self::Int64(Vec::new()),
            SQL_REAL => Self::Float32(Vec::new()),
            SQL_DOUBLE | SQL_FLOAT => Self::Float64(Vec::new()),
            // Bind as double rather than a sized integer: Oracle ODBC fails
            // with HY090 when converting SQL_DECIMAL to SQL_C_SLONG/SBIGINT,
            // and numeric→double conversion is supported by every driver.
            // Values are bound for plot coordinates (f64) anyway.
            SQL_NUMERIC | SQL_DECIMAL => Self::Float64(Vec::new()),
            SQL_BIT => Self::Boolean(Vec::new()),
            SQL_TYPE_DATE => Self::Date(Vec::new()),
            SQL_TYPE_TIME => Self::Time(Vec::new()),
            SQL_TYPE_TIMESTAMP => Self::Timestamp(Vec::new()),
            _ => Self::Text(Vec::new()),
        }
    }

    fn c_type(&self) -> SqlSmallInt {
        match self {
            Self::Int8(_) => SQL_C_STINYINT,
            Self::Int16(_) => SQL_C_SSHORT,
            Self::Int32(_) => SQL_C_SLONG,
            Self::Int64(_) => SQL_C_SBIGINT,
            Self::Float32(_) => SQL_C_FLOAT,
            Self::Float64(_) => SQL_C_DOUBLE,
            Self::Boolean(_) => SQL_C_BIT,
            Self::Date(_) => SQL_C_TYPE_DATE,
            Self::Time(_) => SQL_C_TYPE_TIME,
            Self::Timestamp(_) => SQL_C_TYPE_TIMESTAMP,
            Self::Text(_) => SQL_C_CHAR,
        }
    }

    fn element_size(&self) -> usize {
        match self {
            Self::Int8(_) => std::mem::size_of::<i8>(),
            Self::Int16(_) => std::mem::size_of::<i16>(),
            Self::Int32(_) => std::mem::size_of::<i32>(),
            Self::Int64(_) => std::mem::size_of::<i64>(),
            Self::Float32(_) => std::mem::size_of::<f32>(),
            Self::Float64(_) => std::mem::size_of::<f64>(),
            Self::Boolean(_) => 1,
            Self::Date(_) => std::mem::size_of::<SqlDateStruct>(),
            Self::Time(_) => std::mem::size_of::<SqlTimeStruct>(),
            Self::Timestamp(_) => std::mem::size_of::<SqlTimestampStruct>(),
            Self::Text(_) => 0, // text uses a separate buffer
        }
    }

    fn into_named_array(self, name: &str) -> (String, ArrayRef) {
        let array: ArrayRef = match self {
            Self::Int8(v) => Arc::new(Int8Array::from(v)),
            Self::Int16(v) => Arc::new(Int16Array::from(v)),
            Self::Int32(v) => Arc::new(Int32Array::from(v)),
            Self::Int64(v) => Arc::new(Int64Array::from(v)),
            Self::Float32(v) => Arc::new(Float32Array::from(v)),
            Self::Float64(v) => Arc::new(Float64Array::from(v)),
            Self::Boolean(v) => Arc::new(BooleanArray::from(v)),
            Self::Date(v) => Arc::new(Date32Array::from(v)),
            Self::Time(v) => Arc::new(Time64NanosecondArray::from(v)),
            Self::Timestamp(v) => Arc::new(TimestampMicrosecondArray::from(v)),
            Self::Text(v) => {
                let refs: Vec<Option<&str>> = v.iter().map(|s| s.as_deref()).collect();
                Arc::new(StringArray::from(refs))
            }
        };
        (name.to_string(), array)
    }
}

// ============================================================================
// Date/time conversion helpers
// ============================================================================

fn odbc_date_to_days(d: &SqlDateStruct) -> Option<i32> {
    chrono::NaiveDate::from_ymd_opt(d.year as i32, d.month as u32, d.day as u32).map(|date| {
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        (date - epoch).num_days() as i32
    })
}

fn odbc_time_to_nanos(t: &SqlTimeStruct) -> i64 {
    let h = t.hour as i64;
    let m = t.minute as i64;
    let s = t.second as i64;
    (h * 3600 + m * 60 + s) * 1_000_000_000
}

fn odbc_timestamp_to_micros(ts: &SqlTimestampStruct) -> Option<i64> {
    chrono::NaiveDate::from_ymd_opt(ts.year as i32, ts.month as u32, ts.day as u32)
        .and_then(|date| {
            date.and_hms_nano_opt(
                ts.hour as u32,
                ts.minute as u32,
                ts.second as u32,
                ts.fraction,
            )
        })
        .map(|dt| dt.and_utc().timestamp_micros())
}

// ============================================================================
// Cursor → DataFrame conversion
// ============================================================================

const BATCH_SIZE: usize = 1000;
/// Per-cell text fetch buffer. Values longer than this are an explicit
/// error (see the `ColumnBuilder::Text` arm of `extract_batch`) rather than
/// silently truncated; 16 KiB keeps a 1000-row batch of one text column at
/// 16 MiB.
const DEFAULT_TEXT_BUF_SIZE: usize = 16384;

struct ColumnBuffer {
    data: Vec<u8>,
    indicators: Vec<SqlLen>,
    text_buf_size: usize,
}

fn cursor_to_dataframe(stmt: Statement, batch_size: usize) -> Result<DataFrame> {
    let col_count = stmt.num_result_cols()?;
    if col_count == 0 {
        return Ok(DataFrame::empty());
    }

    // Describe all columns
    let mut col_names = Vec::with_capacity(col_count);
    let mut builders = Vec::with_capacity(col_count);

    for i in 1..=col_count as u16 {
        let (name, data_type, col_size, decimal_digits, _nullable) = stmt.describe_col(i)?;
        col_names.push(name);
        builders.push(ColumnBuilder::from_sql_type(
            data_type,
            col_size,
            decimal_digits,
        ));
    }

    // Set up batch fetching
    stmt.setup_batch_fetch(batch_size)?;
    let mut rows_fetched: SqlULen = 0;
    unsafe { stmt.set_rows_fetched_ptr(&mut rows_fetched)? };

    // Allocate and bind buffers
    let mut buffers: Vec<ColumnBuffer> = builders
        .iter()
        .enumerate()
        .map(|(i, builder)| {
            let (elem_size, text_buf_size) = if matches!(builder, ColumnBuilder::Text(_)) {
                // +2: null terminator, and keep the bind length even — some
                // Unicode drivers (Oracle) reject odd buffer lengths with HY090
                (DEFAULT_TEXT_BUF_SIZE + 2, DEFAULT_TEXT_BUF_SIZE)
            } else {
                (builder.element_size(), 0)
            };

            let data = vec![0u8; elem_size * batch_size];
            let indicators = vec![0isize; batch_size];

            let col_num = (i + 1) as u16;
            let c_type = builder.c_type();
            // We'll bind after creating the buffer
            let _ = (col_num, c_type);

            ColumnBuffer {
                data,
                indicators,
                text_buf_size,
            }
        })
        .collect();

    // Bind columns
    for (i, (builder, buf)) in builders.iter().zip(buffers.iter_mut()).enumerate() {
        let col_num = (i + 1) as u16;
        let c_type = builder.c_type();
        let elem_size = if matches!(builder, ColumnBuilder::Text(_)) {
            (buf.text_buf_size + 2) as SqlLen
        } else {
            builder.element_size() as SqlLen
        };

        stmt.bind_col(
            col_num,
            c_type,
            buf.data.as_mut_ptr() as SqlPointer,
            elem_size,
            buf.indicators.as_mut_ptr(),
        )?;
    }

    // Fetch loop
    loop {
        rows_fetched = 0;
        let rc = stmt.fetch_raw();

        match rc {
            SQL_NO_DATA => break,
            SQL_SUCCESS | SQL_SUCCESS_WITH_INFO => {}
            _ => {
                let diag = wrapper::extract_diagnostic(SQL_HANDLE_STMT, stmt.handle() as SqlHandle);
                return Err(GgsqlError::ReaderError(format!(
                    "Failed to fetch batch: {diag}"
                )));
            }
        }

        let n = rows_fetched as usize;
        if n == 0 {
            break;
        }

        // Extract data from buffers into builders
        for (col_idx, (builder, buf)) in builders.iter_mut().zip(buffers.iter()).enumerate() {
            extract_batch(builder, buf, n, &col_names[col_idx])?;
        }
    }

    // Convert builders to named arrays
    let named_arrays: Vec<(String, ArrayRef)> = col_names
        .iter()
        .zip(builders)
        .map(|(name, builder)| builder.into_named_array(name))
        .collect();

    DataFrame::new(named_arrays)
}

fn extract_batch(
    builder: &mut ColumnBuilder,
    buf: &ColumnBuffer,
    num_rows: usize,
    col_name: &str,
) -> Result<()> {
    for row in 0..num_rows {
        let indicator = buf.indicators[row];
        let is_null = indicator == SQL_NULL_DATA;

        match builder {
            ColumnBuilder::Int8(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let val = buf.data[row * std::mem::size_of::<i8>()] as i8;
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Int16(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let offset = row * std::mem::size_of::<i16>();
                    let val = i16::from_ne_bytes(buf.data[offset..offset + 2].try_into().unwrap());
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Int32(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let offset = row * std::mem::size_of::<i32>();
                    let val = i32::from_ne_bytes(buf.data[offset..offset + 4].try_into().unwrap());
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Int64(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let offset = row * std::mem::size_of::<i64>();
                    let val = i64::from_ne_bytes(buf.data[offset..offset + 8].try_into().unwrap());
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Float32(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let offset = row * std::mem::size_of::<f32>();
                    let val = f32::from_ne_bytes(buf.data[offset..offset + 4].try_into().unwrap());
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Float64(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let offset = row * std::mem::size_of::<f64>();
                    let val = f64::from_ne_bytes(buf.data[offset..offset + 8].try_into().unwrap());
                    v.push(Some(val));
                }
            }
            ColumnBuilder::Boolean(v) => {
                if is_null {
                    v.push(None);
                } else {
                    v.push(Some(buf.data[row] != 0));
                }
            }
            ColumnBuilder::Date(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let size = std::mem::size_of::<SqlDateStruct>();
                    let offset = row * size;
                    let d: SqlDateStruct = unsafe {
                        std::ptr::read_unaligned(buf.data[offset..].as_ptr() as *const _)
                    };
                    v.push(odbc_date_to_days(&d));
                }
            }
            ColumnBuilder::Time(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let size = std::mem::size_of::<SqlTimeStruct>();
                    let offset = row * size;
                    let t: SqlTimeStruct = unsafe {
                        std::ptr::read_unaligned(buf.data[offset..].as_ptr() as *const _)
                    };
                    v.push(Some(odbc_time_to_nanos(&t)));
                }
            }
            ColumnBuilder::Timestamp(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let size = std::mem::size_of::<SqlTimestampStruct>();
                    let offset = row * size;
                    let ts: SqlTimestampStruct = unsafe {
                        std::ptr::read_unaligned(buf.data[offset..].as_ptr() as *const _)
                    };
                    v.push(odbc_timestamp_to_micros(&ts));
                }
            }
            ColumnBuilder::Text(v) => {
                if is_null {
                    v.push(None);
                } else {
                    let elem_size = buf.text_buf_size + 2;
                    let offset = row * elem_size;
                    // indicator is the actual byte length, but may be
                    // SQL_NO_TOTAL (-4) if the driver can't determine length.
                    // In that case, scan for null terminator in the buffer.
                    let actual_len = if indicator >= 0 {
                        if indicator as usize > buf.text_buf_size {
                            return Err(GgsqlError::ReaderError(format!(
                                "ODBC text value in column '{}' is {} bytes, exceeding the \
                                 {}-byte fetch buffer; refusing to return a truncated value",
                                col_name, indicator, buf.text_buf_size,
                            )));
                        }
                        indicator as usize
                    } else {
                        let slice = &buf.data[offset..offset + buf.text_buf_size];
                        match slice.iter().position(|&b| b == 0) {
                            Some(pos) => pos,
                            // No terminator: the value filled the buffer and
                            // the driver can't tell us its true length —
                            // almost certainly truncated.
                            None => {
                                return Err(GgsqlError::ReaderError(format!(
                                    "ODBC text value in column '{}' exceeds the {}-byte fetch \
                                     buffer (driver reports SQL_NO_TOTAL); refusing to return \
                                     a truncated value",
                                    col_name, buf.text_buf_size,
                                )))
                            }
                        }
                    };
                    let bytes = &buf.data[offset..offset + actual_len];
                    let s = String::from_utf8_lossy(bytes).into_owned();
                    v.push(Some(s));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_dialect_from_dbms_name() {
        let d = detect_dialect(Some("Snowflake"), "anything").unwrap();
        assert!(!d.sql_greatest(&["a", "b"]).is_empty());

        let d = detect_dialect(None, "Driver=Snowflake;Server=foo").unwrap();
        assert!(!d.sql_greatest(&["a", "b"]).is_empty());

        let err = detect_dialect(None, "Driver=SomeOther;Server=localhost")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("dialect=ansi"), "got: {err}");
    }

    #[test]
    fn test_register_column_type_via_shared_mapping() {
        use crate::reader::AnsiDialect;
        let ty = |d: &DataType| crate::reader::register_column_type(&AnsiDialect, d).unwrap();
        assert_eq!(ty(&DataType::Int64), "BIGINT");
        assert_eq!(ty(&DataType::Float64), "DOUBLE PRECISION");
        assert_eq!(ty(&DataType::Boolean), "BOOLEAN");
        assert_eq!(ty(&DataType::Date32), "DATE");
        assert_eq!(ty(&DataType::Utf8), "VARCHAR");
    }

    #[test]
    fn test_column_builder_from_sql_type() {
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_TINYINT, 0, 0),
            ColumnBuilder::Int8(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_SMALLINT, 0, 0),
            ColumnBuilder::Int16(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_INTEGER, 0, 0),
            ColumnBuilder::Int32(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_BIGINT, 0, 0),
            ColumnBuilder::Int64(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_REAL, 0, 0),
            ColumnBuilder::Float32(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_DOUBLE, 0, 0),
            ColumnBuilder::Float64(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_FLOAT, 0, 0),
            ColumnBuilder::Float64(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_BIT, 0, 0),
            ColumnBuilder::Boolean(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_TYPE_DATE, 0, 0),
            ColumnBuilder::Date(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_TYPE_TIME, 0, 0),
            ColumnBuilder::Time(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_TYPE_TIMESTAMP, 0, 0),
            ColumnBuilder::Timestamp(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_VARCHAR, 0, 0),
            ColumnBuilder::Text(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_WVARCHAR, 0, 0),
            ColumnBuilder::Text(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_LONGVARCHAR, 0, 0),
            ColumnBuilder::Text(_)
        ));
        // Decimals bind as double regardless of precision/scale: Oracle ODBC
        // rejects SQL_DECIMAL → SQL_C_SBIGINT conversion with HY090
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_NUMERIC, 5, 0),
            ColumnBuilder::Float64(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_NUMERIC, 15, 0),
            ColumnBuilder::Float64(_)
        ));
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_NUMERIC, 25, 0),
            ColumnBuilder::Float64(_)
        ));
        // Decimal with scale>0 maps to Float64
        assert!(matches!(
            ColumnBuilder::from_sql_type(SQL_DECIMAL, 10, 2),
            ColumnBuilder::Float64(_)
        ));
    }

    #[test]
    fn test_column_builder_c_types() {
        assert_eq!(ColumnBuilder::Int8(vec![]).c_type(), SQL_C_STINYINT);
        assert_eq!(ColumnBuilder::Int16(vec![]).c_type(), SQL_C_SSHORT);
        assert_eq!(ColumnBuilder::Int32(vec![]).c_type(), SQL_C_SLONG);
        assert_eq!(ColumnBuilder::Int64(vec![]).c_type(), SQL_C_SBIGINT);
        assert_eq!(ColumnBuilder::Float32(vec![]).c_type(), SQL_C_FLOAT);
        assert_eq!(ColumnBuilder::Float64(vec![]).c_type(), SQL_C_DOUBLE);
        assert_eq!(ColumnBuilder::Boolean(vec![]).c_type(), SQL_C_BIT);
        assert_eq!(ColumnBuilder::Date(vec![]).c_type(), SQL_C_TYPE_DATE);
        assert_eq!(ColumnBuilder::Time(vec![]).c_type(), SQL_C_TYPE_TIME);
        assert_eq!(
            ColumnBuilder::Timestamp(vec![]).c_type(),
            SQL_C_TYPE_TIMESTAMP
        );
        assert_eq!(ColumnBuilder::Text(vec![]).c_type(), SQL_C_CHAR);
    }

    #[test]
    fn test_column_builder_element_sizes() {
        assert_eq!(ColumnBuilder::Int8(vec![]).element_size(), 1);
        assert_eq!(ColumnBuilder::Int16(vec![]).element_size(), 2);
        assert_eq!(ColumnBuilder::Int32(vec![]).element_size(), 4);
        assert_eq!(ColumnBuilder::Int64(vec![]).element_size(), 8);
        assert_eq!(ColumnBuilder::Float32(vec![]).element_size(), 4);
        assert_eq!(ColumnBuilder::Float64(vec![]).element_size(), 8);
        assert_eq!(ColumnBuilder::Boolean(vec![]).element_size(), 1);
        assert!(ColumnBuilder::Date(vec![]).element_size() >= 6);
        assert!(ColumnBuilder::Time(vec![]).element_size() >= 6);
        assert!(ColumnBuilder::Timestamp(vec![]).element_size() >= 14);
        assert_eq!(ColumnBuilder::Text(vec![]).element_size(), 0);
    }

    #[test]
    fn test_column_builder_into_named_array() {
        let builder = ColumnBuilder::Int64(vec![Some(1), None, Some(3)]);
        let (name, array) = builder.into_named_array("col");
        assert_eq!(name, "col");
        assert_eq!(array.len(), 3);
        assert!(!array.is_null(0));
        assert!(array.is_null(1));
        assert!(!array.is_null(2));

        let builder = ColumnBuilder::Text(vec![Some("hello".into()), None]);
        let (_, array) = builder.into_named_array("t");
        assert_eq!(array.len(), 2);
        assert!(!array.is_null(0));
        assert!(array.is_null(1));

        let builder = ColumnBuilder::Boolean(vec![Some(true), Some(false)]);
        let (_, array) = builder.into_named_array("b");
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_odbc_date_to_days() {
        let d = SqlDateStruct {
            year: 1970,
            month: 1,
            day: 1,
        };
        assert_eq!(odbc_date_to_days(&d), Some(0));

        let d = SqlDateStruct {
            year: 2000,
            month: 1,
            day: 1,
        };
        assert_eq!(odbc_date_to_days(&d), Some(10957));

        let d = SqlDateStruct {
            year: 2024,
            month: 2,
            day: 29,
        };
        assert!(odbc_date_to_days(&d).is_some());

        let d = SqlDateStruct {
            year: 2024,
            month: 13,
            day: 1,
        };
        assert_eq!(odbc_date_to_days(&d), None);
    }

    #[test]
    fn test_odbc_time_to_nanos() {
        let t = SqlTimeStruct {
            hour: 0,
            minute: 0,
            second: 0,
        };
        assert_eq!(odbc_time_to_nanos(&t), 0);

        let t = SqlTimeStruct {
            hour: 1,
            minute: 30,
            second: 45,
        };
        assert_eq!(odbc_time_to_nanos(&t), (3600 + 1800 + 45) * 1_000_000_000);

        let t = SqlTimeStruct {
            hour: 23,
            minute: 59,
            second: 59,
        };
        assert_eq!(odbc_time_to_nanos(&t), (86399) * 1_000_000_000);
    }

    #[test]
    fn test_odbc_timestamp_to_micros() {
        let ts = SqlTimestampStruct {
            year: 1970,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            fraction: 0,
        };
        assert_eq!(odbc_timestamp_to_micros(&ts), Some(0));

        let ts = SqlTimestampStruct {
            year: 2024,
            month: 6,
            day: 15,
            hour: 12,
            minute: 30,
            second: 45,
            fraction: 0,
        };
        assert!(odbc_timestamp_to_micros(&ts).unwrap() > 0);

        let ts = SqlTimestampStruct {
            year: 2024,
            month: 13,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            fraction: 0,
        };
        assert_eq!(odbc_timestamp_to_micros(&ts), None);
    }

    #[test]
    fn test_succeeded() {
        assert!(ffi::succeeded(SQL_SUCCESS));
        assert!(ffi::succeeded(SQL_SUCCESS_WITH_INFO));
        assert!(!ffi::succeeded(SQL_ERROR));
        assert!(!ffi::succeeded(SQL_NO_DATA));
        assert!(!ffi::succeeded(SQL_INVALID_HANDLE));
    }

    #[test]
    fn test_try_load_error_without_odbc() {
        // This tests the error path — if ODBC *is* installed it returns Ok,
        // if not it returns a descriptive error. Either way it shouldn't panic.
        let result = ffi::try_load();
        match result {
            Ok(()) => {} // ODBC is available on this machine
            Err(e) => {
                assert!(
                    e.contains("ODBC driver manager not found") || e.contains("GGSQL_ODBC_LIBRARY")
                );
            }
        }
    }

    #[test]
    fn test_connect_missing_prefix() {
        let result = OdbcReader::from_connection_string("DSN=foo");
        match result {
            Err(e) => assert!(e.to_string().contains("odbc://"), "Got: {}", e),
            Ok(_) => panic!("Should have failed without odbc:// prefix"),
        }
    }

    // ========================================================================
    // ODBC integration tests
    //
    // These require ODBC DSNs to be configured on the machine. Run with:
    //   cargo test --package ggsql -- odbc::tests --include-ignored --nocapture
    // ========================================================================

    fn try_connect(dsn: &str) -> Option<OdbcReader> {
        OdbcReader::from_connection_string(&format!("odbc://DSN={}", dsn)).ok()
    }

    // --- PostgreSQL via ODBC -------------------------------------------------

    const PG_DSN: &str = "ggsql-pg-test";

    #[test]
    #[ignore]
    fn pg_connect_and_detect_dialect() {
        let reader = try_connect(PG_DSN).expect("Cannot connect to PostgreSQL ODBC DSN");
        assert_eq!(reader.connection.dbms_name().as_deref(), Some("PostgreSQL"));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_integer() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader.execute_sql("SELECT 42 AS value").unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 1);
        let col = df.column("value").unwrap();
        assert_eq!(crate::array_util::value_to_string(col, 0), "42");
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_float() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT 4.28::double precision AS foo")
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("foo").unwrap();
        let val: f64 = crate::array_util::value_to_string(col, 0).parse().unwrap();
        assert!((val - 4.28).abs() < 0.001);
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_boolean() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader.execute_sql("SELECT true AS t, false AS f").unwrap();
        assert_eq!(df.height(), 1);
        let t = df.column("t").unwrap();
        let f = df.column("f").unwrap();
        assert!(!t.is_null(0));
        assert!(!f.is_null(0));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_date() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader.execute_sql("SELECT DATE '2024-06-15' AS d").unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("d").unwrap();
        assert!(!col.is_null(0));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_timestamp() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT TIMESTAMP '2024-06-15 12:30:45' AS ts")
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("ts").unwrap();
        assert!(!col.is_null(0));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_multiple_types() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT 1 AS i, 2.5::double precision AS f, 'hello' AS s, true AS b")
            .unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 4);
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_multiple_rows() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT generate_series AS n FROM generate_series(1, 100)")
            .unwrap();
        assert_eq!(df.height(), 100);
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_large_batch() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT generate_series AS n FROM generate_series(1, 5000)")
            .unwrap();
        assert_eq!(df.height(), 5000);
        let col = df.column("n").unwrap();
        assert!(!col.is_null(0));
        assert!(!col.is_null(4999));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_nulls() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT NULL::integer AS x UNION ALL SELECT 1")
            .unwrap();
        assert_eq!(df.height(), 2);
        let col = df.column("x").unwrap();
        assert!(col.is_null(0));
        assert!(!col.is_null(1));
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_null_text() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT NULL::text AS s UNION ALL SELECT 'hello'")
            .unwrap();
        assert_eq!(df.height(), 2);
        let col = df.column("s").unwrap();
        assert!(col.is_null(0));
        assert_eq!(crate::array_util::value_to_string(col, 1), "hello");
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_empty_result() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader.execute_sql("SELECT 1 AS x WHERE false").unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 1);
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_ddl_returns_empty() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("CREATE TEMPORARY TABLE __ggsql_test_ddl (x int)")
            .unwrap();
        assert_eq!(df.height(), 0);
        let _ = reader.execute_sql("DROP TABLE IF EXISTS __ggsql_test_ddl");
    }

    #[test]
    #[ignore]
    fn pg_execute_sql_unicode() {
        let reader = try_connect(PG_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT 'héllo wörld 日本語' AS s")
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("s").unwrap();
        let val = crate::array_util::value_to_string(col, 0);
        assert!(val.contains("héllo"));
        assert!(val.contains("日本語"));
    }

    #[test]
    #[ignore]
    fn pg_list_catalogs() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        let catalogs = reader.list_catalogs().unwrap();
        assert!(!catalogs.is_empty(), "Should have at least one catalog");
    }

    #[test]
    #[ignore]
    fn pg_list_schemas() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        let schemas = reader.list_schemas("").unwrap();
        assert!(
            schemas.iter().any(|s| s == "public"),
            "Should contain 'public' schema, got: {:?}",
            schemas
        );
    }

    #[test]
    #[ignore]
    fn pg_list_tables() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        let tables = reader.list_tables("", "public").unwrap();
        for t in &tables {
            assert!(!t.name.is_empty());
            assert!(!t.table_type.is_empty());
        }
    }

    #[test]
    #[ignore]
    fn pg_list_columns() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        // Create a temp table so we have something to list columns for
        let _ = reader.execute_sql(
            "CREATE TEMPORARY TABLE __ggsql_test_cols (id int, name text, score double precision)",
        );
        let cols = reader.list_columns("", "pg_temp_3", "__ggsql_test_cols");
        // May fail if schema name differs, just check it doesn't crash
        if let Ok(cols) = cols {
            if !cols.is_empty() {
                assert!(!cols[0].name.is_empty());
                assert!(!cols[0].data_type.is_empty());
            }
        }
        let _ = reader.execute_sql("DROP TABLE IF EXISTS __ggsql_test_cols");
    }

    #[test]
    #[ignore]
    fn pg_register_and_query() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();

        let df = crate::df!(
            "name" => vec!["alice", "bob", "carol"],
            "score" => vec![85i64, 92, 78]
        )
        .unwrap();

        reader.register("__ggsql_test_reg", df, true).unwrap();

        let result = reader
            .execute_sql("SELECT name, score FROM __ggsql_test_reg ORDER BY name")
            .unwrap();
        assert_eq!(result.height(), 3);

        let name_col = result.column("name").unwrap();
        assert_eq!(crate::array_util::value_to_string(name_col, 0), "alice");

        reader.unregister("__ggsql_test_reg").unwrap();
    }

    #[test]
    #[ignore]
    fn pg_register_with_nulls() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();

        let df = crate::df!(
            "x" => vec![Some(1i64), None, Some(3)],
            "s" => vec![Some("a"), None, Some("c")]
        )
        .unwrap();

        reader.register("__ggsql_test_null", df, true).unwrap();

        let result = reader
            .execute_sql("SELECT x, s FROM __ggsql_test_null ORDER BY x NULLS FIRST")
            .unwrap();
        assert_eq!(result.height(), 3);
        let x = result.column("x").unwrap();
        assert!(x.is_null(0));
        assert!(!x.is_null(1));

        reader.unregister("__ggsql_test_null").unwrap();
    }

    #[test]
    #[ignore]
    fn pg_unregister_nonexistent_errors() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        assert!(reader.unregister("__does_not_exist__").is_err());
    }

    // --- PostgreSQL PostGIS spatial tests ------------------------------------

    #[cfg(feature = "spatial")]
    #[test]
    #[ignore]
    fn pg_spatial_create_temp_table() {
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        reader
            .execute_sql("DROP TABLE IF EXISTS __ggsql_spatial_test")
            .unwrap();
        reader
            .execute_sql(
                "CREATE TEMP TABLE __ggsql_spatial_test (name TEXT, geom GEOMETRY(Polygon, 4326))",
            )
            .unwrap();
        reader
            .execute_sql(
                "INSERT INTO __ggsql_spatial_test VALUES \
                 ('a', ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 4326))",
            )
            .unwrap();
        let df = reader
            .execute_sql("SELECT name, ST_AsBinary(geom) AS geom FROM __ggsql_spatial_test")
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[cfg(feature = "spatial")]
    #[test]
    #[ignore]
    fn pg_spatial_st_transform_proj_string() {
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        let df = reader
            .execute_sql(
                "SELECT ST_AsText(ST_Transform(\
                    ST_SetSRID(ST_Point(10, 50), 4326), \
                    '+proj=laea +lon_0=10 +lat_0=52'\
                 )) AS wkt",
            )
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("wkt").unwrap();
        let wkt = crate::array_util::value_to_string(col, 0);
        assert!(wkt.contains("POINT"), "Expected POINT, got: {}", wkt);
    }

    #[cfg(feature = "spatial")]
    #[test]
    #[ignore]
    fn pg_spatial_st_transform_srid() {
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        let df = reader
            .execute_sql(
                "SELECT ST_SRID(ST_Transform(ST_SetSRID(ST_Point(10, 50), 4326), 3857)) AS srid",
            )
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("srid").unwrap();
        assert_eq!(crate::array_util::value_to_string(col, 0), "3857");
    }

    #[cfg(all(feature = "spatial", feature = "vegalite"))]
    #[test]
    #[ignore]
    fn pg_spatial_end_to_end_projection() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        reader
            .execute_sql("DROP TABLE IF EXISTS __ggsql_countries")
            .unwrap();
        reader
            .execute_sql(
                "CREATE TABLE __ggsql_countries (\
                    name TEXT, geom GEOMETRY(Polygon, 4326)\
                 )",
            )
            .unwrap();
        reader
            .execute_sql(
                "INSERT INTO __ggsql_countries VALUES \
                 ('France', ST_GeomFromText(\
                    'POLYGON((2.5 51.1, -4.8 48.4, -1.7 43.3, 3.0 42.4, 7.7 48.9, 2.5 51.1))', 4326)),\
                 ('Germany', ST_GeomFromText(\
                    'POLYGON((6.0 54.8, 14.7 54.0, 15.0 51.0, 12.1 47.7, 5.9 47.6, 6.0 54.8))', 4326))",
            )
            .unwrap();

        let spec = reader
            .execute(
                "SELECT name, geom FROM __ggsql_countries \
                 VISUALISE name AS fill \
                 DRAW spatial MAPPING geom AS geometry \
                 PROJECT TO lambert",
            )
            .unwrap();

        assert_eq!(spec.plot.layers.len(), 1);
        assert!(spec.layer_data(0).unwrap().height() > 0);

        reader.execute_sql("DROP TABLE __ggsql_countries").unwrap();
    }

    #[cfg(all(feature = "spatial", feature = "vegalite"))]
    #[test]
    #[ignore]
    fn pg_spatial_orthographic_clip() {
        use crate::reader::Reader;
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        reader
            .execute_sql("DROP TABLE IF EXISTS __ggsql_clip_test")
            .unwrap();
        reader
            .execute_sql(
                "CREATE TABLE __ggsql_clip_test (\
                    name TEXT, geom GEOMETRY(Polygon, 4326)\
                 )",
            )
            .unwrap();
        reader
            .execute_sql(
                "INSERT INTO __ggsql_clip_test VALUES \
                 ('visible', ST_GeomFromText(\
                    'POLYGON((5 45, 15 45, 15 55, 5 55, 5 45))', 4326)),\
                 ('hidden', ST_GeomFromText(\
                    'POLYGON((170 -40, 180 -40, 180 -30, 170 -30, 170 -40))', 4326))",
            )
            .unwrap();

        let spec = reader
            .execute(
                "SELECT name, geom FROM __ggsql_clip_test \
                 VISUALISE name AS fill \
                 DRAW spatial MAPPING geom AS geometry \
                 PROJECT TO orthographic SETTING origin => (10, 50)",
            )
            .unwrap();

        // Only the visible polygon should survive clipping
        assert_eq!(spec.layer_data(0).unwrap().height(), 1);

        reader.execute_sql("DROP TABLE __ggsql_clip_test").unwrap();
    }

    #[cfg(all(feature = "spatial", feature = "vegalite"))]
    #[test]
    #[ignore]
    fn pg_spatial_point_layer_projection() {
        use crate::reader::Reader;
        use crate::writer::Writer;
        let reader = try_connect(PG_DSN).unwrap();
        reader
            .execute_sql("CREATE EXTENSION IF NOT EXISTS postgis")
            .unwrap();
        reader
            .execute_sql("DROP TABLE IF EXISTS __ggsql_cities")
            .unwrap();
        reader
            .execute_sql(
                "CREATE TABLE __ggsql_cities (\
                    name TEXT, lon DOUBLE PRECISION, lat DOUBLE PRECISION\
                 )",
            )
            .unwrap();
        reader
            .execute_sql(
                "INSERT INTO __ggsql_cities VALUES \
                 ('Amsterdam', 4.90, 52.37),\
                 ('Paris', 2.35, 48.86),\
                 ('Berlin', 13.40, 52.52)",
            )
            .unwrap();

        let spec = reader
            .execute(
                "SELECT name, lon, lat FROM __ggsql_cities \
                 VISUALISE lon AS lon, lat AS lat, name AS label \
                 DRAW point \
                 PROJECT TO lambert SETTING origin => (10, 50)",
            )
            .unwrap();

        assert_eq!(spec.plot.layers.len(), 1);
        let df = spec.layer_data(0).unwrap();
        assert_eq!(df.height(), 3);

        let writer = crate::writer::vegalite::VegaLiteWriter::new();
        let json_str = writer.write(&spec.plot, &spec.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Coordinates should be projected (not raw lon/lat)
        let data = vl_spec["data"]["values"].as_array().unwrap();
        let layer_key = spec.plot.layers[0].data_key.as_ref().unwrap();
        let rows: Vec<_> = data
            .iter()
            .filter(|r| r[crate::naming::SOURCE_COLUMN] == layer_key.as_str())
            .collect();
        assert_eq!(rows.len(), 3);
        for row in &rows {
            let lon = row[crate::naming::aesthetic_column("pos1")]
                .as_f64()
                .expect("pos1 should be numeric");
            let lat = row[crate::naming::aesthetic_column("pos2")]
                .as_f64()
                .expect("pos2 should be numeric");
            // LAEA centered on (10, 50) produces meter-scale values, not degrees
            assert!(
                lon.abs() > 100.0 || lat.abs() > 100.0,
                "Expected projected coordinates (meters), got ({lon}, {lat})"
            );
        }

        reader.execute_sql("DROP TABLE __ggsql_cities").unwrap();
    }

    // --- SQLite via ODBC -----------------------------------------------------

    const SQLITE_DSN: &str = "ggsql-sqlite-test";

    #[test]
    #[ignore]
    fn sqlite_connect_and_detect_dialect() {
        let reader = try_connect(SQLITE_DSN).expect("Cannot connect to SQLite ODBC DSN");
        let dbms = reader.connection.dbms_name().unwrap_or_default();
        assert!(
            dbms.to_lowercase().contains("sqlite"),
            "Expected SQLite, got: {}",
            dbms
        );
    }

    #[test]
    #[ignore]
    fn sqlite_execute_sql_integer() {
        let reader = try_connect(SQLITE_DSN).unwrap();
        let df = reader.execute_sql("SELECT 42 AS value").unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("value").unwrap();
        assert_eq!(crate::array_util::value_to_string(col, 0), "42");
    }

    #[test]
    #[ignore]
    fn sqlite_execute_sql_text() {
        let reader = try_connect(SQLITE_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT 'hello world' AS greeting")
            .unwrap();
        assert_eq!(df.height(), 1);
        let col = df.column("greeting").unwrap();
        assert_eq!(crate::array_util::value_to_string(col, 0), "hello world");
    }

    #[test]
    #[ignore]
    fn sqlite_execute_sql_multiple_rows() {
        let reader = try_connect(SQLITE_DSN).unwrap();
        // SQLite doesn't have generate_series by default, use VALUES
        let df = reader
            .execute_sql(
                "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 50) SELECT x FROM cnt",
            )
            .unwrap();
        assert_eq!(df.height(), 50);
    }

    #[test]
    #[ignore]
    fn sqlite_execute_sql_nulls() {
        let reader = try_connect(SQLITE_DSN).unwrap();
        let df = reader
            .execute_sql("SELECT NULL AS x UNION ALL SELECT 1")
            .unwrap();
        assert_eq!(df.height(), 2);
        let col = df.column("x").unwrap();
        assert!(col.is_null(0));
    }

    #[test]
    #[ignore]
    fn sqlite_list_catalogs_empty() {
        use crate::reader::Reader;
        let reader = try_connect(SQLITE_DSN).unwrap();
        let catalogs = reader.list_catalogs().unwrap();
        // SQLite ODBC driver typically returns no catalogs
        let _ = catalogs;
    }

    #[test]
    #[ignore]
    fn sqlite_list_schemas() {
        use crate::reader::Reader;
        let reader = try_connect(SQLITE_DSN).unwrap();
        let schemas = reader.list_schemas("").unwrap();
        let _ = schemas;
    }

    #[test]
    #[ignore]
    fn sqlite_list_tables() {
        use crate::reader::Reader;
        let reader = try_connect(SQLITE_DSN).unwrap();
        let tables = reader.list_tables("", "").unwrap();
        let _ = tables;
    }

    #[test]
    #[ignore]
    fn sqlite_list_columns() {
        use crate::reader::Reader;
        let reader = try_connect(SQLITE_DSN).unwrap();
        // Create a table to list columns for
        let _ = reader
            .execute_sql("CREATE TABLE IF NOT EXISTS __ggsql_test_cols (id INTEGER, name TEXT)");
        let cols = reader.list_columns("", "", "__ggsql_test_cols").unwrap();
        if !cols.is_empty() {
            assert!(!cols[0].name.is_empty());
        }
        let _ = reader.execute_sql("DROP TABLE IF EXISTS __ggsql_test_cols");
    }

    #[test]
    #[ignore]
    fn sqlite_register_and_query() {
        use crate::reader::Reader;
        let reader = try_connect(SQLITE_DSN).unwrap();

        let df = crate::df!(
            "x" => vec![1i64, 2, 3],
            "y" => vec![10i64, 20, 30]
        )
        .unwrap();

        reader.register("__ggsql_test_reg", df, true).unwrap();

        let result = reader
            .execute_sql("SELECT x, y FROM __ggsql_test_reg ORDER BY x")
            .unwrap();
        assert_eq!(result.height(), 3);

        reader.unregister("__ggsql_test_reg").unwrap();
    }
}
