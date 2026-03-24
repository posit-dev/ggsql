//! SQLite data source implementation
//!
//! Provides a reader for SQLite databases with Polars DataFrame integration.
//! Works on both native targets and wasm32-unknown-unknown (via sqlite-wasm-rs).

use crate::reader::Reader;
use crate::{DataFrame, GgsqlError, Result};
use chrono::Datelike;
use polars::prelude::*;
use rusqlite::Connection;
use std::cell::RefCell;
use std::collections::HashSet;

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
}

/// SQLite database reader
///
/// Executes SQL queries against SQLite databases (in-memory or file-based)
/// and returns results as Polars DataFrames.
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

    let forbidden = ['"', '\0', '\n', '\r'];
    for ch in forbidden {
        if name.contains(ch) {
            return Err(GgsqlError::ReaderError(format!(
                "Table name '{}' contains invalid character '{}'",
                name,
                ch.escape_default()
            )));
        }
    }

    if name.len() > 128 {
        return Err(GgsqlError::ReaderError(format!(
            "Table name '{}' exceeds maximum length of 128 characters",
            name
        )));
    }

    Ok(())
}

/// Map a Polars DataType to a SQLite column type string
fn polars_type_to_sqlite(dtype: &DataType) -> &'static str {
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
        DataType::Date => "TEXT",
        DataType::Datetime(_, _) => "TEXT",
        DataType::Time => "TEXT",
        _ => "TEXT",
    }
}

/// Convert a Polars AnyValue to a rusqlite Value for parameter binding
fn anyvalue_to_sqlite(value: AnyValue, _dtype: &DataType) -> rusqlite::types::Value {
    use rusqlite::types::Value;

    match value {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Integer(b as i64),
        AnyValue::Int8(v) => Value::Integer(v as i64),
        AnyValue::Int16(v) => Value::Integer(v as i64),
        AnyValue::Int32(v) => Value::Integer(v as i64),
        AnyValue::Int64(v) => Value::Integer(v),
        AnyValue::UInt8(v) => Value::Integer(v as i64),
        AnyValue::UInt16(v) => Value::Integer(v as i64),
        AnyValue::UInt32(v) => Value::Integer(v as i64),
        AnyValue::UInt64(v) => Value::Integer(v as i64),
        AnyValue::Float32(v) => Value::Real(v as f64),
        AnyValue::Float64(v) => Value::Real(v),
        AnyValue::String(s) => Value::Text(s.to_string()),
        AnyValue::StringOwned(s) => Value::Text(s.to_string()),
        AnyValue::Date(days) => chrono::NaiveDate::from_num_days_from_ce_opt(days + 719_163)
            .and_then(|d| to_sql_value(&d))
            .unwrap_or(Value::Null),
        AnyValue::Datetime(us, _, _) => chrono::DateTime::from_timestamp_micros(us)
            .map(|d| d.naive_utc())
            .and_then(|d| to_sql_value(&d))
            .unwrap_or(Value::Null),
        AnyValue::Time(ns) => {
            let secs = (ns / 1_000_000_000) as u32;
            let nanos = (ns % 1_000_000_000) as u32;
            chrono::NaiveTime::from_num_seconds_from_midnight_opt(secs, nanos)
                .and_then(|t| to_sql_value(&t))
                .unwrap_or(Value::Null)
        }
        _ => Value::Text(format!("{}", value)),
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
        #[cfg(feature = "parquet")]
        {
            let dataset_names = super::data::extract_builtin_dataset_names(sql)?;
            for name in &dataset_names {
                let table_name = crate::naming::builtin_data_table(name);
                if !self.table_exists(&table_name) {
                    let df = super::data::load_builtin_dataframe(name)?;
                    self.register(&table_name, df, true)?;
                }
            }
        }

        // Rewrite ggsql:name → __ggsql_data_name__ in SQL
        let sql = super::data::rewrite_namespaced_sql(sql)?;

        // Check if this is a DDL statement
        let trimmed = sql.trim().to_uppercase();
        let is_ddl = trimmed.starts_with("CREATE ")
            || trimmed.starts_with("DROP ")
            || trimmed.starts_with("INSERT ")
            || trimmed.starts_with("UPDATE ")
            || trimmed.starts_with("DELETE ")
            || trimmed.starts_with("ALTER ");

        if is_ddl {
            self.conn
                .execute_batch(&sql)
                .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute DDL: {}", e)))?;
            return DataFrame::new(Vec::<Column>::new()).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create empty DataFrame: {}", e))
            });
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

        // Build Series from collected values
        let mut columns = Vec::with_capacity(column_count);
        for (col_idx, values) in col_values.into_iter().enumerate() {
            let name = &column_names[col_idx];
            let series = sqlite_values_to_series(name, values)?;
            columns.push(series.into());
        }

        DataFrame::new(columns)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to create DataFrame: {}", e)))
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        validate_table_name(name)?;

        if self.table_exists(name) {
            if replace {
                let sql = format!("DROP TABLE IF EXISTS \"{}\"", name);
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
        let col_defs: Vec<String> = df
            .get_columns()
            .iter()
            .map(|col| {
                let col_name = col.name().to_string();
                let col_type = polars_type_to_sqlite(col.dtype());
                format!("\"{}\" {}", col_name, col_type)
            })
            .collect();

        let create_sql = format!("CREATE TABLE \"{}\" ({})", name, col_defs.join(", "));
        self.conn.execute(&create_sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to create table '{}': {}", name, e))
        })?;

        // Insert data using params_from_iter, wrapped in a transaction
        if df.height() > 0 {
            let placeholders: Vec<&str> = vec!["?"; df.width()];
            let insert_sql = format!(
                "INSERT INTO \"{}\" VALUES ({})",
                name,
                placeholders.join(", ")
            );

            let dtypes: Vec<DataType> =
                df.get_columns().iter().map(|c| c.dtype().clone()).collect();

            self.conn.execute_batch("BEGIN").map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to begin transaction: {}", e))
            })?;

            let result = (|| -> Result<()> {
                let mut stmt = self.conn.prepare(&insert_sql).map_err(|e| {
                    GgsqlError::ReaderError(format!("Failed to prepare INSERT: {}", e))
                })?;

                for row_idx in 0..df.height() {
                    let values: Vec<rusqlite::types::Value> = df
                        .get_columns()
                        .iter()
                        .enumerate()
                        .map(|(col_idx, col)| {
                            let value = col.get(row_idx).map_err(|e| {
                                GgsqlError::ReaderError(format!(
                                    "Failed to get value at row {}, col {}: {}",
                                    row_idx, col_idx, e
                                ))
                            })?;
                            Ok(anyvalue_to_sqlite(value, &dtypes[col_idx]))
                        })
                        .collect::<Result<Vec<_>>>()?;

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

        let sql = format!("DROP TABLE IF EXISTS \"{}\"", name);
        self.conn.execute(&sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        self.registered_tables.borrow_mut().remove(name);
        Ok(())
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &SqliteDialect
    }
}

/// Try to parse all non-null TEXT values as ISO-8601 dates (YYYY-MM-DD).
/// Returns a Date series if all non-null values parse, None otherwise.
fn try_parse_as_date(name: &str, values: &[rusqlite::types::Value]) -> Option<Series> {
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

    let series = Series::new(name.into(), parsed);
    series.cast(&DataType::Date).ok()
}

/// Try to parse all non-null TEXT values as ISO-8601 datetimes.
/// Supports both "T" and space separators (e.g. "2024-01-15T10:30:00" or "2024-01-15 10:30:00").
/// Returns a Datetime series if all non-null values parse, None otherwise.
fn try_parse_as_datetime(name: &str, values: &[rusqlite::types::Value]) -> Option<Series> {
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

    let series = Series::new(name.into(), parsed);
    series
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .ok()
}

/// Infer the best Polars type from a column of SQLite values and build a Series.
///
/// SQLite uses dynamic typing, so we infer the column type from all values:
/// - All Integer → Int64
/// - All Integer/Real → Float64
/// - All Text → String
/// - Mixed → String fallback
fn sqlite_values_to_series(name: &str, values: Vec<rusqlite::types::Value>) -> Result<Series> {
    use rusqlite::types::Value;

    if values.is_empty() {
        // Default to String for empty columns
        return Ok(Series::new(name.into(), Vec::<Option<String>>::new()));
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
        if let Some(series) = try_parse_as_date(name, &values) {
            return Ok(series);
        }
        if let Some(series) = try_parse_as_datetime(name, &values) {
            return Ok(series);
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
        return Ok(Series::new(name.into(), vals));
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
        return Ok(Series::new(name.into(), vals));
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
        return Ok(Series::new(name.into(), vals));
    }

    // All nulls — default to String
    let vals: Vec<Option<String>> = values.into_iter().map(|_| None).collect();
    Ok(Series::new(name.into(), vals))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(df.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_subquery_preserves_integer_types() {
        let reader = SqliteReader::new().unwrap();
        let df = reader
            .execute_sql("SELECT x, y FROM (SELECT 1 AS x, 1 AS y)")
            .unwrap();

        assert_eq!(df.shape(), (1, 2));
        assert_eq!(df.column("x").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("y").unwrap().dtype(), &DataType::Int64);
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
        assert_eq!(df.get_column_names(), vec!["x", "y"]);
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

        let df = DataFrame::new(vec![
            Column::new("x".into(), vec![1i32, 2, 3]),
            Column::new("y".into(), vec![10i32, 20, 30]),
        ])
        .unwrap();

        reader.register("my_table", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT * FROM my_table ORDER BY x")
            .unwrap();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_register_duplicate_name_errors() {
        let reader = SqliteReader::new().unwrap();

        let df1 = DataFrame::new(vec![Column::new("a".into(), vec![1i32])]).unwrap();
        let df2 = DataFrame::new(vec![Column::new("b".into(), vec![2i32])]).unwrap();

        reader.register("dup_table", df1, false).unwrap();

        let result = reader.register("dup_table", df2, false);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn test_register_invalid_table_names() {
        let reader = SqliteReader::new().unwrap();
        let df = DataFrame::new(vec![Column::new("a".into(), vec![1i32])]).unwrap();

        let result = reader.register("", df.clone(), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        let result = reader.register("bad\"name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        let result = reader.register("bad\0name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        let long_name = "a".repeat(200);
        let result = reader.register(&long_name, df, false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum length"));
    }

    #[test]
    fn test_register_empty_dataframe() {
        let reader = SqliteReader::new().unwrap();

        let df = DataFrame::new(vec![
            Column::new("x".into(), Vec::<i32>::new()),
            Column::new("y".into(), Vec::<String>::new()),
        ])
        .unwrap();

        reader.register("empty_table", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM empty_table").unwrap();
        assert_eq!(result.shape(), (0, 2));
        assert_eq!(result.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_unregister() {
        let reader = SqliteReader::new().unwrap();
        let df = DataFrame::new(vec![Column::new("x".into(), vec![1i32, 2, 3])]).unwrap();

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
        let df = DataFrame::new(vec![Column::new("x".into(), vec![1i32, 2, 3])]).unwrap();

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

        let df = DataFrame::new(vec![
            Column::new("id".into(), ids),
            Column::new("value".into(), values),
            Column::new("name".into(), names),
        ])
        .unwrap();

        reader.register("large_table", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT COUNT(*) as cnt FROM large_table")
            .unwrap();
        let count = result.column("cnt").unwrap().i64().unwrap().get(0).unwrap();
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
        assert_eq!(df.get_column_names(), vec!["region", "total"]);
    }

    #[test]
    fn test_register_with_replace() {
        let reader = SqliteReader::new().unwrap();

        let df1 = DataFrame::new(vec![Column::new("x".into(), vec![1i32])]).unwrap();
        let df2 = DataFrame::new(vec![Column::new("x".into(), vec![2i32, 3])]).unwrap();

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

        let df = DataFrame::new(vec![Column::new("flag".into(), vec![true, false, true])]).unwrap();

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

        // Register a DataFrame with a Date column
        let dates = Series::new("d".into(), vec![19000i32, 19001, 19002]);
        let dates = dates.cast(&DataType::Date).unwrap();
        let df = DataFrame::new(vec![
            dates.into_column(),
            Column::new("v".into(), vec![1, 2, 3]),
        ])
        .unwrap();

        reader.register("date_data", df, false).unwrap();

        let result = reader.execute_sql("SELECT * FROM date_data").unwrap();
        assert_eq!(result.height(), 3);
        assert_eq!(result.column("d").unwrap().dtype(), &DataType::Date);
        assert_eq!(result.column("v").unwrap().dtype(), &DataType::Int64);
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
                result.column("ts").unwrap().dtype(),
                DataType::Datetime(_, _)
            ),
            "Expected Datetime, got {:?}",
            result.column("ts").unwrap().dtype()
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
        assert_eq!(result.column("name").unwrap().dtype(), &DataType::String);
    }

    #[test]
    fn test_date_vegalite_temporal() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = SqliteReader::new().unwrap();

        // Register a table with a date column
        let dates = Series::new("date".into(), vec![19000i32, 19001, 19002]);
        let dates = dates.cast(&DataType::Date).unwrap();
        let df = DataFrame::new(vec![
            dates.into_column(),
            Column::new("value".into(), vec![10, 20, 30]),
        ])
        .unwrap();
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

#[cfg(feature = "parquet")]
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
            result.column("Date").unwrap().dtype(),
            &DataType::Date,
            "airquality Date column should be detected as Date, not String"
        );
    }
}
