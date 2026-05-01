//! SQLite data source implementation
//!
//! Provides a reader for SQLite databases with Arrow DataFrame integration.
//! Works on both native targets and wasm32-unknown-unknown (via sqlite-wasm-rs).

use crate::reader::Reader;
use crate::{naming, DataFrame, GgsqlError, Result};
use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit};
use chrono::Datelike;
use rusqlite::Connection;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

/// SQLite SQL dialect.
///
/// Overrides type name methods for SQLite's limited type system
/// (TEXT for dates, REAL for numbers, INTEGER for booleans).
pub struct SqliteDialect;

impl super::SqlDialect for SqliteDialect {
    fn string_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("REAL")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("INTEGER")
    }

    fn boolean_type_name(&self) -> Option<&str> {
        Some("INTEGER")
    }

    fn date_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn time_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("date('1970-01-01', '+{} days')", days_since_epoch)
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let seconds = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("datetime('1970-01-01 00:00:00', '+{} seconds')", seconds)
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let seconds = nanoseconds_since_midnight as f64 / 1_000_000_000.0;
        format!("time('00:00:00', '+{} seconds')", seconds)
    }

    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }

    /// SQLite does not support `CREATE OR REPLACE`, so emit a drop-then-create
    /// pair. Column aliases are preserved portably via the default CTE wrapper.
    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = naming::quote_ident(name);
        let body = super::wrap_with_column_aliases(body_sql, column_aliases);
        vec![
            format!("DROP TABLE IF EXISTS {}", qname),
            format!("CREATE TEMP TABLE {} AS {}", qname, body),
        ]
    }
}

/// SQLite database reader
///
/// Executes SQL queries against SQLite databases (in-memory or file-based)
/// and returns results as DataFrames.
pub struct SqliteReader {
    conn: Connection,
    registered_tables: RefCell<HashSet<String>>,
}

impl SqliteReader {
    /// Create a new in-memory SQLite reader
    pub fn new() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to open in-memory SQLite: {}", e))
        })?;
        Ok(Self {
            conn,
            registered_tables: RefCell::new(HashSet::new()),
        })
    }

    /// Create a SQLite reader from a connection string
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_info = super::connection::parse_connection_string(uri)?;

        let conn = match conn_info {
            super::connection::ConnectionInfo::SQLite(path) => {
                Connection::open(&path).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to open SQLite file '{}': {}", path, e))
                })?
            }
            _ => {
                return Err(GgsqlError::ReaderError(format!(
                    "Connection string '{}' is not supported by SqliteReader",
                    uri
                )))
            }
        };

        Ok(Self {
            conn,
            registered_tables: RefCell::new(HashSet::new()),
        })
    }

    /// Get a reference to the underlying SQLite connection
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// List table names known to this reader.
    ///
    /// When `internal` is false, filters out internal tables (prefixed with `__ggsql_`).
    pub fn list_tables(&self, internal: bool) -> Vec<String> {
        self.registered_tables
            .borrow()
            .iter()
            .filter(|name| internal || !name.starts_with("__ggsql_"))
            .cloned()
            .collect()
    }

    /// Check if a table is registered
    fn table_exists(&self, name: &str) -> bool {
        let sql = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1";
        self.conn
            .prepare(sql)
            .and_then(|mut stmt| stmt.exists([name]))
            .unwrap_or(false)
    }
}

impl Default for SqliteReader {
    fn default() -> Self {
        Self::new().expect("Failed to create default SqliteReader")
    }
}

/// Validate a table name
fn validate_table_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(GgsqlError::ReaderError("Table name cannot be empty".into()));
    }

    let forbidden = ['\0', '\n', '\r'];
    for ch in forbidden {
        if name.contains(ch) {
            return Err(GgsqlError::ReaderError(format!(
                "Table name '{}' contains invalid character '{}'",
                name,
                ch.escape_default()
            )));
        }
    }

    Ok(())
}

/// Map an Arrow DataType to a SQLite column type string
fn arrow_type_to_sqlite(dtype: &DataType) -> &'static str {
    match dtype {
        DataType::Float32 | DataType::Float64 => "REAL",
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => "INTEGER",
        DataType::Boolean => "INTEGER",
        DataType::Date32 => "TEXT",
        DataType::Timestamp(_, _) => "TEXT",
        DataType::Time64(_) => "TEXT",
        _ => "TEXT",
    }
}

/// Convert an Arrow array value at a given row index to a rusqlite Value for parameter binding.
fn array_value_to_sqlite(array: &ArrayRef, row_idx: usize) -> rusqlite::types::Value {
    use crate::array_util;
    use rusqlite::types::Value;

    if array.is_null(row_idx) {
        return Value::Null;
    }

    match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::Int8 => {
            let arr = array_util::as_i8(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::Int16 => {
            let arr = array_util::as_i16(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::Int32 => {
            let arr = array_util::as_i32(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::Int64 => {
            let arr = array_util::as_i64(array).unwrap();
            Value::Integer(arr.value(row_idx))
        }
        DataType::UInt8 => {
            let arr = array_util::as_u8(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::UInt16 => {
            let arr = array_util::as_u16(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::UInt32 => {
            let arr = array_util::as_u32(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::UInt64 => {
            let arr = array_util::as_u64(array).unwrap();
            Value::Integer(arr.value(row_idx) as i64)
        }
        DataType::Float32 => {
            let arr = array_util::as_f32(array).unwrap();
            Value::Real(arr.value(row_idx) as f64)
        }
        DataType::Float64 => {
            let arr = array_util::as_f64(array).unwrap();
            Value::Real(arr.value(row_idx))
        }
        DataType::Utf8 => {
            let arr = array_util::as_str(array).unwrap();
            Value::Text(arr.value(row_idx).to_string())
        }
        DataType::Date32 => {
            let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
            let days = arr.value(row_idx);
            chrono::NaiveDate::from_num_days_from_ce_opt(days + 719_163)
                .and_then(|d| to_sql_value(&d))
                .unwrap_or(Value::Null)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let us = arr.value(row_idx);
            chrono::DateTime::from_timestamp_micros(us)
                .map(|d| d.naive_utc())
                .and_then(|d| to_sql_value(&d))
                .unwrap_or(Value::Null)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            let ms = arr.value(row_idx);
            chrono::DateTime::from_timestamp_millis(ms)
                .map(|d| d.naive_utc())
                .and_then(|d| to_sql_value(&d))
                .unwrap_or(Value::Null)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let arr = array
                .as_any()
                .downcast_ref::<Time64NanosecondArray>()
                .unwrap();
            let ns = arr.value(row_idx);
            let secs = (ns / 1_000_000_000) as u32;
            let nanos = (ns % 1_000_000_000) as u32;
            chrono::NaiveTime::from_num_seconds_from_midnight_opt(secs, nanos)
                .and_then(|t| to_sql_value(&t))
                .unwrap_or(Value::Null)
        }
        _ => {
            // Fallback: use array_util::value_to_string
            Value::Text(crate::array_util::value_to_string(array, row_idx))
        }
    }
}

/// Use rusqlite's `ToSql` to convert a value into a `rusqlite::types::Value`.
fn to_sql_value(v: &dyn rusqlite::types::ToSql) -> Option<rusqlite::types::Value> {
    use rusqlite::types::ToSqlOutput;
    match v.to_sql().ok()? {
        ToSqlOutput::Borrowed(vref) => Some(vref.into()),
        ToSqlOutput::Owned(val) => Some(val),
        _ => None,
    }
}

impl Reader for SqliteReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Handle ggsql:name namespaced identifiers (builtin datasets)
        #[cfg(all(feature = "builtin-data", feature = "parquet"))]
        {
            let dataset_names = super::data::extract_builtin_dataset_names(sql)?;
            for name in &dataset_names {
                let table_name = naming::builtin_data_table(name);
                if !self.table_exists(&table_name) {
                    let df = super::data::load_builtin_dataframe(name)?;
                    self.register(&table_name, df, true)?;
                }
            }
        }

        // Rewrite ggsql:name → __ggsql_data_name__ in SQL
        let sql = super::data::rewrite_namespaced_sql(sql)?;

        if !super::returns_rows(&sql) {
            self.conn
                .execute_batch(&sql)
                .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute SQL: {}", e)))?;
            return Ok(DataFrame::empty());
        }

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to prepare SQL: {}", e)))?;

        let column_count = stmt.column_count();
        if column_count == 0 {
            return Err(GgsqlError::ReaderError(
                "Query returned no columns".to_string(),
            ));
        }

        let column_names: Vec<String> = stmt
            .column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        // Collect all rows, inferring types from actual values
        let mut col_values: Vec<Vec<rusqlite::types::Value>> = vec![Vec::new(); column_count];

        let mut rows = stmt.raw_query();
        while let Some(row) = rows
            .next()
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to fetch row: {}", e)))?
        {
            for (col_idx, col_vec) in col_values.iter_mut().enumerate().take(column_count) {
                let value: rusqlite::types::Value = row.get(col_idx).map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to get value at column {}: {}",
                        col_idx, e
                    ))
                })?;
                col_vec.push(value);
            }
        }

        let named_arrays: Vec<(String, ArrayRef)> = col_values
            .into_iter()
            .enumerate()
            .map(|(col_idx, values)| {
                let name = column_names[col_idx].clone();
                let array = sqlite_values_to_array(&name, values)?;
                Ok((name, array))
            })
            .collect::<Result<Vec<_>>>()?;

        DataFrame::new(named_arrays)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        validate_table_name(name)?;

        if self.table_exists(name) {
            if replace {
                let sql = format!("DROP TABLE IF EXISTS {}", naming::quote_ident(name));
                self.conn.execute(&sql, []).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to drop table '{}': {}", name, e))
                })?;
                self.registered_tables.borrow_mut().remove(name);
            } else {
                return Err(GgsqlError::ReaderError(format!(
                    "Table '{}' already exists",
                    name
                )));
            }
        }

        // Build CREATE TABLE statement
        let col_names = df.get_column_names();
        let schema = df.schema();
        let col_defs: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| {
                let col_type = arrow_type_to_sqlite(field.data_type());
                format!("{} {}", naming::quote_ident(field.name()), col_type)
            })
            .collect();

        let create_sql = format!(
            "CREATE TABLE {} ({})",
            naming::quote_ident(name),
            col_defs.join(", ")
        );
        self.conn.execute(&create_sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to create table '{}': {}", name, e))
        })?;

        // Insert data row by row, wrapped in a transaction
        if df.height() > 0 {
            let placeholders: Vec<&str> = vec!["?"; df.width()];
            let insert_sql = format!(
                "INSERT INTO {} VALUES ({})",
                naming::quote_ident(name),
                placeholders.join(", ")
            );

            let columns = df.get_columns();
            let _ = &col_names; // keep col_names alive

            self.conn.execute_batch("BEGIN").map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to begin transaction: {}", e))
            })?;

            let result = (|| -> Result<()> {
                let mut stmt = self.conn.prepare(&insert_sql).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to prepare INSERT: {}", e))
                })?;

                for row_idx in 0..df.height() {
                    let values: Vec<rusqlite::types::Value> = columns
                        .iter()
                        .map(|col| array_value_to_sqlite(col, row_idx))
                        .collect();

                    stmt.execute(rusqlite::params_from_iter(values))
                        .map_err(|e| {
                            GgsqlError::ReaderError(format!(
                                "Failed to insert row {} into '{}': {}",
                                row_idx, name, e
                            ))
                        })?;
                }
                Ok(())
            })();

            match result {
                Ok(()) => {
                    self.conn.execute_batch("COMMIT").map_err(|e| {
                        GgsqlError::ReaderError(format!("Failed to commit transaction: {}", e))
                    })?;
                }
                Err(e) => {
                    let _ = self.conn.execute_batch("ROLLBACK");
                    return Err(e);
                }
            }
        }

        self.registered_tables.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        if !self.registered_tables.borrow().contains(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        let sql = format!("DROP TABLE IF EXISTS {}", naming::quote_ident(name));
        self.conn.execute(&sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        self.registered_tables.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<super::Spec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &SqliteDialect
    }

    fn list_catalogs(&self) -> Result<Vec<String>> {
        let df = self.execute_sql("SELECT name FROM pragma_database_list ORDER BY name")?;
        let col = df.column("name")?;
        let mut results = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            if !col.is_null(i) {
                results.push(crate::array_util::value_to_string(col, i));
            }
        }
        Ok(results)
    }

    fn list_schemas(&self, _catalog: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn list_tables(&self, catalog: &str, _schema: &str) -> Result<Vec<super::TableInfo>> {
        let df = self.execute_sql(&format!(
            "SELECT name, type FROM {}.sqlite_master \
             WHERE type IN ('table', 'view') ORDER BY name",
            naming::quote_ident(catalog)
        ))?;
        let name_col = df.column("name")?;
        let type_col = df.column("type")?;
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

    fn list_columns(
        &self,
        _catalog: &str,
        _schema: &str,
        table: &str,
    ) -> Result<Vec<super::ColumnInfo>> {
        let df = self.execute_sql(&format!(
            "SELECT name, type FROM pragma_table_info('{}') ORDER BY cid",
            table.replace('\'', "''")
        ))?;
        let name_col = df.column("name")?;
        let type_col = df.column("type")?;
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
}

/// Try to parse all non-null TEXT values as ISO-8601 dates (YYYY-MM-DD).
/// Returns a Date32 array if all non-null values parse, None otherwise.
fn try_parse_as_date(values: &[rusqlite::types::Value]) -> Option<ArrayRef> {
    use rusqlite::types::{FromSql, Value, ValueRef};

    // Days between 0001-01-01 (CE day 1) and 1970-01-01 (Unix epoch)
    const EPOCH_DAYS_FROM_CE: i32 = 719_163;

    let mut parsed: Vec<Option<i32>> = Vec::with_capacity(values.len());

    for v in values {
        match v {
            Value::Null => parsed.push(None),
            Value::Text(s) => {
                let vref = ValueRef::Text(s.as_bytes());
                let date: chrono::NaiveDate = FromSql::column_result(vref).ok()?;
                parsed.push(Some(date.num_days_from_ce() - EPOCH_DAYS_FROM_CE));
            }
            _ => return None,
        }
    }

    Some(Arc::new(Date32Array::from(parsed)) as ArrayRef)
}

/// Try to parse all non-null TEXT values as ISO-8601 datetimes.
/// Supports both "T" and space separators (e.g. "2024-01-15T10:30:00" or "2024-01-15 10:30:00").
/// Returns a TimestampMillisecond array if all non-null values parse, None otherwise.
fn try_parse_as_datetime(values: &[rusqlite::types::Value]) -> Option<ArrayRef> {
    use rusqlite::types::{FromSql, Value, ValueRef};

    let mut parsed: Vec<Option<i64>> = Vec::with_capacity(values.len());

    for v in values {
        match v {
            Value::Null => parsed.push(None),
            Value::Text(s) => {
                // Must contain a time separator to distinguish from plain dates
                if !s.contains('T') && !s.contains(' ') {
                    return None;
                }
                let vref = ValueRef::Text(s.as_bytes());
                let dt: chrono::NaiveDateTime = FromSql::column_result(vref).ok()?;
                parsed.push(Some(dt.and_utc().timestamp_millis()));
            }
            _ => return None,
        }
    }

    Some(Arc::new(TimestampMillisecondArray::from(parsed)) as ArrayRef)
}

/// Infer the best Arrow type from a column of SQLite values and build an ArrayRef.
///
/// SQLite uses dynamic typing, so we infer the column type from all values:
/// - All Integer -> Int64
/// - All Integer/Real -> Float64
/// - All Text -> String (with temporal detection)
/// - Mixed -> String fallback
fn sqlite_values_to_array(name: &str, values: Vec<rusqlite::types::Value>) -> Result<ArrayRef> {
    use rusqlite::types::Value;

    let _ = name; // name is unused now but kept for consistency

    if values.is_empty() {
        // Default to String for empty columns
        return Ok(Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as ArrayRef);
    }

    // Determine the dominant type
    let mut has_int = false;
    let mut has_real = false;
    let mut has_text = false;
    let mut has_blob = false;

    for v in &values {
        match v {
            Value::Null => {}
            Value::Integer(_) => has_int = true,
            Value::Real(_) => has_real = true,
            Value::Text(_) => has_text = true,
            Value::Blob(_) => has_blob = true,
        }
    }

    // If we have text, try temporal detection before falling back to String
    if has_text && !has_blob {
        if let Some(array) = try_parse_as_date(&values) {
            return Ok(array);
        }
        if let Some(array) = try_parse_as_datetime(&values) {
            return Ok(array);
        }
    }

    if has_text || has_blob {
        let vals: Vec<Option<String>> = values
            .into_iter()
            .map(|v| match v {
                Value::Null => None,
                Value::Integer(i) => Some(i.to_string()),
                Value::Real(f) => Some(f.to_string()),
                Value::Text(s) => Some(s),
                Value::Blob(b) => Some(format!("{:?}", b)),
            })
            .collect();
        let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
        return Ok(Arc::new(StringArray::from(refs)) as ArrayRef);
    }

    // If we have any reals, use f64
    if has_real {
        let vals: Vec<Option<f64>> = values
            .into_iter()
            .map(|v| match v {
                Value::Null => None,
                Value::Integer(i) => Some(i as f64),
                Value::Real(f) => Some(f),
                _ => None,
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(vals)) as ArrayRef);
    }

    // Pure integers
    if has_int {
        let vals: Vec<Option<i64>> = values
            .into_iter()
            .map(|v| match v {
                Value::Null => None,
                Value::Integer(i) => Some(i),
                _ => None,
            })
            .collect();
        return Ok(Arc::new(Int64Array::from(vals)) as ArrayRef);
    }

    // All nulls — default to String
    let vals: Vec<Option<&str>> = values.iter().map(|_| None).collect();
    Ok(Arc::new(StringArray::from(vals)) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_util::as_i64;
    use crate::df;

    #[test]
    fn test_create_in_memory() {
        let reader = SqliteReader::new();
        assert!(reader.is_ok());
    }

    #[test]
    fn test_simple_query() {
        let reader = SqliteReader::new().unwrap();
        let df = reader.execute_sql("SELECT 1 as x, 2 as y").unwrap();

        assert_eq!(df.shape(), (1, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_subquery_preserves_integer_types() {
        let reader = SqliteReader::new().unwrap();
        let df = reader
            .execute_sql("SELECT x, y FROM (SELECT 1 AS x, 1 AS y)")
            .unwrap();

        assert_eq!(df.shape(), (1, 2));
        assert_eq!(df.column_dtype("x").unwrap(), DataType::Int64);
        assert_eq!(df.column_dtype("y").unwrap(), DataType::Int64);
    }

    #[test]
    fn test_subquery_vegalite_quantitative() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();
        let spec = reader
            .execute("SELECT x, y FROM (SELECT 1 AS x, 1 AS y) VISUALISE x AS x, y AS y DRAW point")
            .unwrap();

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();

        // x and y should be quantitative, not nominal
        assert!(
            json.contains("\"quantitative\""),
            "Expected quantitative type in output: {}",
            json
        );
        assert!(
            !json.contains("\"nominal\""),
            "Did not expect nominal type in output: {}",
            json
        );
    }

    #[test]
    fn test_table_creation_and_query() {
        let reader = SqliteReader::new().unwrap();

        reader
            .connection()
            .execute("CREATE TABLE test(x INTEGER, y INTEGER)", [])
            .unwrap();

        reader
            .connection()
            .execute("INSERT INTO test VALUES (1, 2), (3, 4)", [])
            .unwrap();

        let df = reader.execute_sql("SELECT * FROM test").unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_invalid_sql() {
        let reader = SqliteReader::new().unwrap();
        let result = reader.execute_sql("INVALID SQL SYNTAX");
        assert!(result.is_err());
    }

    #[test]
    fn test_register_and_query() {
        let reader = SqliteReader::new().unwrap();

        let df = df! {
            "x" => vec![1i32, 2, 3],
            "y" => vec![10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_table", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT * FROM my_table ORDER BY x")
            .unwrap();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(
            result.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_register_duplicate_name_errors() {
        let reader = SqliteReader::new().unwrap();

        let df1 = df! { "a" => vec![1i32] }.unwrap();
        let df2 = df! { "b" => vec![2i32] }.unwrap();

        reader.register("dup_table", df1, false).unwrap();

        let result = reader.register("dup_table", df2, false);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn test_register_invalid_table_names() {
        let reader = SqliteReader::new().unwrap();
        let df = df! { "a" => vec![1i32] }.unwrap();

        let result = reader.register("", df.clone(), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Name with double quote should succeed (quote_ident escapes it)
        let result = reader.register("bad\"name", df.clone(), false);
        assert!(result.is_ok());
        reader.unregister("bad\"name").unwrap();

        let result = reader.register("bad\0name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));
    }

    #[test]
    fn test_register_empty_dataframe() {
        let reader = SqliteReader::new().unwrap();

        // Create an empty DataFrame with schema by slicing a 1-row df to 0 rows
        let df = df! {
            "x" => vec![0i32],
            "y" => vec!["placeholder"],
        }
        .unwrap()
        .slice(0, 0);

        reader.register("empty_table", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM empty_table").unwrap();
        assert_eq!(result.shape(), (0, 2));
        assert_eq!(
            result.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_unregister() {
        let reader = SqliteReader::new().unwrap();
        let df = df! { "x" => vec![1i32, 2, 3] }.unwrap();

        reader.register("test_data", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM test_data").unwrap();
        assert_eq!(result.height(), 3);

        reader.unregister("test_data").unwrap();

        let result = reader.execute_sql("SELECT * FROM test_data");
        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_not_registered() {
        let reader = SqliteReader::new().unwrap();

        reader
            .connection()
            .execute("CREATE TABLE user_table (x INTEGER)", [])
            .unwrap();

        let result = reader.unregister("user_table");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("was not registered via this reader"));
    }

    #[test]
    fn test_reregister_after_unregister() {
        let reader = SqliteReader::new().unwrap();
        let df = df! { "x" => vec![1i32, 2, 3] }.unwrap();

        reader.register("data", df.clone(), false).unwrap();
        reader.unregister("data").unwrap();

        reader.register("data", df, false).unwrap();
        let result = reader.execute_sql("SELECT * FROM data").unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_register_large_dataframe() {
        let reader = SqliteReader::new().unwrap();

        let n = 3000;
        let ids: Vec<i32> = (0..n).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let names: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();

        let df = df! {
            "id" => ids,
            "value" => values,
            "name" => names,
        }
        .unwrap();

        reader.register("large_table", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT COUNT(*) as cnt FROM large_table")
            .unwrap();
        let count = as_i64(result.column("cnt").unwrap()).unwrap().value(0);
        assert_eq!(count, n as i64);
    }

    #[test]
    fn test_query_with_aggregation() {
        let reader = SqliteReader::new().unwrap();

        reader
            .connection()
            .execute("CREATE TABLE sales(region TEXT, revenue REAL)", [])
            .unwrap();

        reader
            .connection()
            .execute(
                "INSERT INTO sales VALUES ('US', 100), ('US', 200), ('EU', 150)",
                [],
            )
            .unwrap();

        let df = reader
            .execute_sql("SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
            .unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["region".to_string(), "total".to_string()]
        );
    }

    #[test]
    fn test_register_with_replace() {
        let reader = SqliteReader::new().unwrap();

        let df1 = df! { "x" => vec![1i32] }.unwrap();
        let df2 = df! { "x" => vec![2i32, 3] }.unwrap();

        reader.register("data", df1, false).unwrap();
        reader.register("data", df2, true).unwrap();

        let result = reader.execute_sql("SELECT * FROM data").unwrap();
        assert_eq!(result.height(), 2);
    }

    #[test]
    fn test_ddl_execution() {
        let reader = SqliteReader::new().unwrap();

        // DDL should succeed and return empty DataFrame
        let result = reader
            .execute_sql("CREATE TABLE test (x INTEGER, y TEXT)")
            .unwrap();
        assert_eq!(result.height(), 0);

        // Insert data
        reader
            .execute_sql("INSERT INTO test VALUES (1, 'hello')")
            .unwrap();

        // Query it back
        let df = reader.execute_sql("SELECT * FROM test").unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_boolean_roundtrip() {
        let reader = SqliteReader::new().unwrap();

        let df = df! { "flag" => vec![true, false, true] }.unwrap();

        reader.register("bool_data", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM bool_data").unwrap();
        // Booleans are stored as INTEGER in SQLite, come back as i64
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_mixed_types_in_column() {
        let reader = SqliteReader::new().unwrap();

        // SQLite allows mixed types in a column
        reader
            .connection()
            .execute("CREATE TABLE mixed (val)", [])
            .unwrap();
        reader
            .connection()
            .execute("INSERT INTO mixed VALUES (1), (2.5), ('hello')", [])
            .unwrap();

        let df = reader.execute_sql("SELECT * FROM mixed").unwrap();
        assert_eq!(df.height(), 3);
        // Should fall back to String since we have mixed types
    }

    #[test]
    fn test_date_column_roundtrip() {
        let reader = SqliteReader::new().unwrap();

        // Register a DataFrame with a Date column (Date32 in Arrow)
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19000i32, 19001, 19002]));
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let df = DataFrame::new(vec![("d", dates), ("v", values)]).unwrap();

        reader.register("date_data", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM date_data").unwrap();
        assert_eq!(result.height(), 3);
        assert_eq!(result.column_dtype("d").unwrap(), DataType::Date32);
        assert_eq!(result.column_dtype("v").unwrap(), DataType::Int64);
    }

    #[test]
    fn test_datetime_column_roundtrip() {
        let reader = SqliteReader::new().unwrap();

        // Store datetime strings directly via SQL
        reader
            .execute_sql("CREATE TABLE dt_data (ts TEXT, v INTEGER)")
            .unwrap();
        reader
            .execute_sql(
                "INSERT INTO dt_data VALUES ('2024-01-15T10:30:00', 1), ('2024-01-16T11:45:00', 2)",
            )
            .unwrap();

        let result = reader.execute_sql("SELECT * FROM dt_data").unwrap();
        assert_eq!(result.height(), 2);
        assert!(
            matches!(
                result.column_dtype("ts").unwrap(),
                DataType::Timestamp(_, _)
            ),
            "Expected Timestamp, got {:?}",
            result.column_dtype("ts").unwrap()
        );
    }

    #[test]
    fn test_non_date_strings_stay_string() {
        let reader = SqliteReader::new().unwrap();

        reader
            .execute_sql("CREATE TABLE str_data (name TEXT)")
            .unwrap();
        reader
            .execute_sql("INSERT INTO str_data VALUES ('hello'), ('world')")
            .unwrap();

        let result = reader.execute_sql("SELECT * FROM str_data").unwrap();
        assert_eq!(result.column_dtype("name").unwrap(), DataType::Utf8);
    }

    #[test]
    fn test_date_vegalite_temporal() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();

        // Register a table with a date column (Date32 in Arrow)
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19000i32, 19001, 19002]));
        let values: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let df = DataFrame::new(vec![("date", dates), ("value", values)]).unwrap();
        reader.register("ts_data", df, false).unwrap();

        let spec = reader
            .execute("SELECT * FROM ts_data VISUALISE date AS x, value AS y DRAW line")
            .unwrap();

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();

        assert!(
            json.contains("\"temporal\""),
            "Expected temporal type in Vega-Lite output: {}",
            json
        );
    }

    // =========================================================================
    // Stat Transform Geom Tests
    // =========================================================================

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_bar_count_stat() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();
        reader
            .execute_sql("CREATE TABLE bar_data (category TEXT)")
            .unwrap();
        reader
            .execute_sql("INSERT INTO bar_data VALUES ('A'), ('B'), ('A'), ('C'), ('A'), ('B')")
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM bar_data VISUALISE DRAW bar MAPPING category AS x")
            .unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert!(spec.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"bar\""),
            "Expected bar mark in output: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_histogram() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();
        reader
            .execute_sql("CREATE TABLE hist_data (value REAL)")
            .unwrap();
        let values: Vec<String> = (0..50).map(|i| format!("({})", i as f64 * 2.0)).collect();
        reader
            .execute_sql(&format!(
                "INSERT INTO hist_data VALUES {}",
                values.join(", ")
            ))
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM hist_data VISUALISE DRAW histogram MAPPING value AS x")
            .unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        let layer_df = spec.layer_data(0).unwrap();
        assert!(
            layer_df.height() < 50,
            "Histogram should bin data: got {} rows",
            layer_df.height()
        );

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"bar\""),
            "Histogram should render as bar mark: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_density() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();
        reader
            .execute_sql("CREATE TABLE density_data (value REAL)")
            .unwrap();
        let values: Vec<String> = (0..50).map(|i| format!("({})", i as f64 * 0.5)).collect();
        reader
            .execute_sql(&format!(
                "INSERT INTO density_data VALUES {}",
                values.join(", ")
            ))
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM density_data VISUALISE DRAW density MAPPING value AS x")
            .unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert!(spec.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"area\""),
            "Density should render as area mark: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_boxplot() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();
        reader
            .execute_sql("CREATE TABLE box_data (grp TEXT, value REAL)")
            .unwrap();
        let mut values = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            values.push(format!("('A', {})", v));
        }
        for v in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0] {
            values.push(format!("('B', {})", v));
        }
        reader
            .execute_sql(&format!(
                "INSERT INTO box_data VALUES {}",
                values.join(", ")
            ))
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM box_data VISUALISE DRAW boxplot MAPPING grp AS x, value AS y")
            .unwrap();

        assert!(spec.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(!json.is_empty(), "Boxplot should render successfully");
    }
}

#[cfg(all(feature = "builtin-data", feature = "parquet"))]
#[cfg(test)]
mod builtin_data_tests {
    use super::*;

    #[test]
    fn test_builtin_penguins_auto_loads() {
        let reader = SqliteReader::new().unwrap();

        let result = reader
            .execute_sql("SELECT * FROM ggsql:penguins LIMIT 5")
            .unwrap();
        assert_eq!(result.height(), 5);
        assert!(result.width() > 0);
    }

    #[test]
    fn test_builtin_airquality_auto_loads() {
        let reader = SqliteReader::new().unwrap();

        let result = reader
            .execute_sql("SELECT * FROM ggsql:airquality LIMIT 5")
            .unwrap();
        assert_eq!(result.height(), 5);
        assert!(result.width() > 0);
    }

    #[test]
    fn test_builtin_airquality_date_is_temporal() {
        let reader = SqliteReader::new().unwrap();

        let result = reader
            .execute_sql("SELECT Date FROM ggsql:airquality LIMIT 5")
            .unwrap();
        assert_eq!(
            result.column_dtype("Date").unwrap(),
            DataType::Date32,
            "airquality Date column should be detected as Date32, not String"
        );
    }
}
