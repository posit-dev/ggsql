//! DuckDB data source implementation
//!
//! Provides a reader for DuckDB databases with direct Polars DataFrame integration.

use crate::reader::{connection::ConnectionInfo, Reader};
use crate::{DataFrame, GgsqlError, Result};
use arrow::ipc::reader::FileReader;
use duckdb::vtab::arrow::{arrow_recordbatch_to_query_params, ArrowVTab};
use duckdb::{params, Connection};
use polars::io::SerWriter;
use polars::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::io::Cursor;

/// DuckDB database reader
///
/// Executes SQL queries against DuckDB databases (in-memory or file-based)
/// and returns results as Polars DataFrames.
///
/// # Examples
///
/// ```rust,ignore
/// use ggsql::reader::{Reader, DuckDBReader};
///
/// // In-memory database
/// let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
/// let df = reader.execute_sql("SELECT 1 as x, 2 as y")?;
///
/// // File-based database
/// let reader = DuckDBReader::from_connection_string("duckdb://data.db")?;
/// let df = reader.execute_sql("SELECT * FROM sales")?;
/// ```
pub struct DuckDBReader {
    conn: Connection,
    registered_tables: RefCell<HashSet<String>>,
}

impl DuckDBReader {
    /// Create a new DuckDB reader from a connection string
    ///
    /// # Arguments
    ///
    /// * `uri` - Connection string (e.g., "duckdb://memory" or "duckdb://file.db")
    ///
    /// # Returns
    ///
    /// A configured DuckDB reader
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The connection string format is invalid
    /// - The database file cannot be opened
    /// - DuckDB initialization fails
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_info = super::connection::parse_connection_string(uri)?;

        let conn = match conn_info {
            ConnectionInfo::DuckDBMemory => Connection::open_in_memory().map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to open in-memory DuckDB: {}", e))
            })?,
            ConnectionInfo::DuckDBFile(path) => Connection::open(&path).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to open DuckDB file '{}': {}", path, e))
            })?,
            _ => {
                return Err(GgsqlError::ReaderError(format!(
                    "Connection string '{}' is not supported by DuckDBReader",
                    uri
                )))
            }
        };

        // Register Arrow virtual table function for DataFrame registration
        conn.register_table_function::<ArrowVTab>("arrow")
            .map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register arrow function: {}", e))
            })?;

        Ok(Self {
            conn,
            registered_tables: RefCell::new(HashSet::new()),
        })
    }

    /// Get a reference to the underlying DuckDB connection
    ///
    /// Useful for executing setup queries (CREATE TABLE, INSERT, etc.)
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Check if a table exists in the database
    fn table_exists(&self, name: &str) -> Result<bool> {
        let sql = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?";
        let count: i64 = self
            .conn
            .query_row(sql, [name], |row| row.get(0))
            .unwrap_or(0);
        Ok(count > 0)
    }
}

/// Validate a table name
fn validate_table_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(GgsqlError::ReaderError("Table name cannot be empty".into()));
    }

    // Reject characters that could break double-quoted identifiers or cause issues
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

    // Reasonable length limit
    if name.len() > 128 {
        return Err(GgsqlError::ReaderError(format!(
            "Table name '{}' exceeds maximum length of 128 characters",
            name
        )));
    }

    Ok(())
}

/// Convert a Polars DataFrame to DuckDB Arrow query parameters via IPC serialization
fn dataframe_to_arrow_params(df: DataFrame) -> Result<[usize; 2]> {
    // Serialize DataFrame to IPC format
    let mut buffer = Vec::new();
    {
        let mut writer = IpcWriter::new(&mut buffer);
        writer.finish(&mut df.clone()).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to serialize DataFrame: {}", e))
        })?;
    }

    // Read IPC into arrow crate's RecordBatch
    let cursor = Cursor::new(buffer);
    let reader = FileReader::try_new(cursor, None)
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to read IPC: {}", e)))?;

    // Collect all batches and concatenate if needed
    let batches: Vec<_> = reader.filter_map(|r| r.ok()).collect();

    if batches.is_empty() {
        return Err(GgsqlError::ReaderError(
            "DataFrame produced no Arrow batches".into(),
        ));
    }

    // For single batch, use directly; for multiple, concatenate
    let rb = if batches.len() == 1 {
        batches.into_iter().next().unwrap()
    } else {
        arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to concat batches: {}", e)))?
    };

    Ok(arrow_recordbatch_to_query_params(rb))
}

/// Helper struct for building typed columns from rows
enum ColumnBuilder {
    TinyInt(Vec<Option<i8>>),
    SmallInt(Vec<Option<i16>>),
    Int(Vec<Option<i32>>),
    BigInt(Vec<Option<i64>>),
    UTinyInt(Vec<Option<i16>>),  // Cast to i16
    USmallInt(Vec<Option<i32>>), // Cast to i32
    UInt(Vec<Option<i64>>),      // Cast to i64
    UBigInt(Vec<Option<u64>>),   // Keep as u64, check overflow
    Float(Vec<Option<f32>>),
    Double(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
    Date32(Vec<Option<i32>>),
    Timestamp(Vec<Option<i64>>),
    Time64(Vec<Option<i64>>),
    Decimal(Vec<Option<f64>>),     // Convert to Float64
    HugeInt(Vec<Option<i128>>),    // Will check overflow
    Blob(Vec<Option<String>>),     // Convert to String
    Fallback(Vec<Option<String>>), // Fallback for unsupported types
}

impl ColumnBuilder {
    fn new(duckdb_type: duckdb::types::Type) -> Self {
        use duckdb::types::Type;
        match duckdb_type {
            Type::TinyInt => ColumnBuilder::TinyInt(Vec::new()),
            Type::SmallInt => ColumnBuilder::SmallInt(Vec::new()),
            Type::Int => ColumnBuilder::Int(Vec::new()),
            Type::BigInt => ColumnBuilder::BigInt(Vec::new()),
            Type::UTinyInt => ColumnBuilder::UTinyInt(Vec::new()),
            Type::USmallInt => ColumnBuilder::USmallInt(Vec::new()),
            Type::UInt => ColumnBuilder::UInt(Vec::new()),
            Type::UBigInt => ColumnBuilder::UBigInt(Vec::new()),
            Type::Float => ColumnBuilder::Float(Vec::new()),
            Type::Double => ColumnBuilder::Double(Vec::new()),
            Type::Boolean => ColumnBuilder::Boolean(Vec::new()),
            Type::Text => ColumnBuilder::Text(Vec::new()),
            Type::Date32 => ColumnBuilder::Date32(Vec::new()),
            Type::Timestamp => ColumnBuilder::Timestamp(Vec::new()),
            Type::Time64 => ColumnBuilder::Time64(Vec::new()),
            Type::Decimal => ColumnBuilder::Decimal(Vec::new()),
            Type::HugeInt => ColumnBuilder::HugeInt(Vec::new()),
            Type::Blob => ColumnBuilder::Blob(Vec::new()),
            _ => ColumnBuilder::Fallback(Vec::new()),
        }
    }

    fn add_value(&mut self, row: &duckdb::Row, col_idx: usize) -> Result<()> {
        use ColumnBuilder::*;
        match self {
            TinyInt(ref mut values) => values.push(row.get(col_idx).ok()),
            SmallInt(ref mut values) => values.push(row.get(col_idx).ok()),
            Int(ref mut values) => values.push(row.get(col_idx).ok()),
            BigInt(ref mut values) => values.push(row.get(col_idx).ok()),
            UTinyInt(ref mut values) => {
                let val: Option<u8> = row.get(col_idx).ok();
                values.push(val.map(|v| v as i16));
            }
            USmallInt(ref mut values) => {
                let val: Option<u16> = row.get(col_idx).ok();
                values.push(val.map(|v| v as i32));
            }
            UInt(ref mut values) => {
                let val: Option<u32> = row.get(col_idx).ok();
                values.push(val.map(|v| v as i64));
            }
            UBigInt(ref mut values) => values.push(row.get(col_idx).ok()),
            Float(ref mut values) => values.push(row.get(col_idx).ok()),
            Double(ref mut values) => values.push(row.get(col_idx).ok()),
            Boolean(ref mut values) => values.push(row.get(col_idx).ok()),
            Text(ref mut values) => values.push(row.get(col_idx).ok()),
            Date32(ref mut values) => values.push(row.get(col_idx).ok()),
            Timestamp(ref mut values) => values.push(row.get(col_idx).ok()),
            Time64(ref mut values) => values.push(row.get(col_idx).ok()),
            Decimal(ref mut values) => {
                use duckdb::types::ValueRef;
                let val = match row.get_ref(col_idx) {
                    Ok(ValueRef::Decimal(d)) => {
                        // Convert Decimal to string, then parse as f64
                        let decimal_str = d.to_string();
                        decimal_str.parse::<f64>().ok()
                    }
                    Ok(ValueRef::Null) => None,
                    Ok(ValueRef::TinyInt(i)) => Some(i as f64),
                    Ok(ValueRef::SmallInt(i)) => Some(i as f64),
                    Ok(ValueRef::Int(i)) => Some(i as f64),
                    Ok(ValueRef::BigInt(i)) => Some(i as f64),
                    Ok(ValueRef::HugeInt(i)) => Some(i as f64),
                    Ok(ValueRef::UTinyInt(i)) => Some(i as f64),
                    Ok(ValueRef::USmallInt(i)) => Some(i as f64),
                    Ok(ValueRef::UInt(i)) => Some(i as f64),
                    Ok(ValueRef::UBigInt(i)) => Some(i as f64),
                    Ok(ValueRef::Float(f)) => Some(f as f64),
                    Ok(ValueRef::Double(f)) => Some(f),
                    _ => None,
                };
                values.push(val);
            }
            HugeInt(ref mut values) => values.push(row.get(col_idx).ok()),
            Blob(ref mut values) => {
                // Blob: try to get as String, or use empty string
                let val: Option<String> = row.get(col_idx).ok();
                values.push(val.or(Some(String::new())));
            }
            Fallback(ref mut values) => {
                // Fallback: try to get as String, or use empty string
                let val: Option<String> = row.get(col_idx).ok();
                values.push(val.or(Some(String::new())));
            }
        }
        Ok(())
    }

    fn build(self, column_name: &str) -> Result<polars::prelude::Series> {
        use polars::prelude::*;
        use ColumnBuilder::*;

        Ok(match self {
            TinyInt(values) => Series::new(column_name.into(), values),
            SmallInt(values) => Series::new(column_name.into(), values),
            Int(values) => Series::new(column_name.into(), values),
            BigInt(values) => Series::new(column_name.into(), values),
            UTinyInt(values) => Series::new(column_name.into(), values),
            USmallInt(values) => Series::new(column_name.into(), values),
            UInt(values) => Series::new(column_name.into(), values),
            UBigInt(values) => {
                // Check if all values fit in i64
                let all_fit = values
                    .iter()
                    .all(|opt_val| opt_val.map(|val| val <= i64::MAX as u64).unwrap_or(true));

                if all_fit {
                    let i64_values: Vec<Option<i64>> = values
                        .into_iter()
                        .map(|opt_val| opt_val.map(|val| val as i64))
                        .collect();
                    Series::new(column_name.into(), i64_values)
                } else {
                    eprintln!(
                        "Warning: UBigInt overflow in column '{}', converting to string",
                        column_name
                    );
                    let string_values: Vec<Option<String>> = values
                        .into_iter()
                        .map(|opt_val| opt_val.map(|val| val.to_string()))
                        .collect();
                    Series::new(column_name.into(), string_values)
                }
            }
            Float(values) => Series::new(column_name.into(), values),
            Double(values) => Series::new(column_name.into(), values),
            Boolean(values) => Series::new(column_name.into(), values),
            Text(values) => Series::new(column_name.into(), values),
            Date32(values) => {
                let series = Series::new(column_name.into(), values);
                series
                    .cast(&DataType::Date)
                    .map_err(|e| GgsqlError::ReaderError(format!("Date cast failed: {}", e)))?
            }
            Timestamp(values) => {
                let series = Series::new(column_name.into(), values);
                series
                    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .map_err(|e| GgsqlError::ReaderError(format!("Timestamp cast failed: {}", e)))?
            }
            Time64(values) => {
                let series = Series::new(column_name.into(), values);
                series
                    .cast(&DataType::Time)
                    .map_err(|e| GgsqlError::ReaderError(format!("Time cast failed: {}", e)))?
            }
            Decimal(values) => Series::new(column_name.into(), values),
            HugeInt(values) => {
                // Check if all values fit in i64
                let all_fit = values.iter().all(|opt_val| {
                    opt_val
                        .map(|val| val >= i64::MIN as i128 && val <= i64::MAX as i128)
                        .unwrap_or(true)
                });

                if all_fit {
                    let i64_values: Vec<Option<i64>> = values
                        .into_iter()
                        .map(|opt_val| opt_val.map(|val| val as i64))
                        .collect();
                    Series::new(column_name.into(), i64_values)
                } else {
                    eprintln!(
                        "Warning: HugeInt overflow in column '{}', converting to string",
                        column_name
                    );
                    let string_values: Vec<Option<String>> = values
                        .into_iter()
                        .map(|opt_val| opt_val.map(|val| val.to_string()))
                        .collect();
                    Series::new(column_name.into(), string_values)
                }
            }
            Blob(values) => {
                eprintln!(
                    "Warning: Converting Blob column '{}' to string (debug format)",
                    column_name
                );
                Series::new(column_name.into(), values)
            }
            Fallback(values) => {
                eprintln!(
                    "Warning: Using fallback string conversion for column '{}'",
                    column_name
                );
                Series::new(column_name.into(), values)
            }
        })
    }
}

impl Reader for DuckDBReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        use polars::prelude::*;

        // Register builtin datasets if referenced
        #[cfg(feature = "builtin-data")]
        super::data::register_builtin_datasets_duckdb(sql, &self.conn)?;

        // Rewrite ggsql:name → __ggsql_data_name__ in SQL
        let sql = super::data::rewrite_namespaced_sql(sql)?;

        // Check if this is a DDL statement (CREATE, DROP, INSERT, UPDATE, DELETE, ALTER)
        // DDL statements don't return rows, so we handle them specially
        let trimmed = sql.trim().to_uppercase();
        let is_ddl = trimmed.starts_with("CREATE ")
            || trimmed.starts_with("DROP ")
            || trimmed.starts_with("INSERT ")
            || trimmed.starts_with("UPDATE ")
            || trimmed.starts_with("DELETE ")
            || trimmed.starts_with("ALTER ");

        if is_ddl {
            // For DDL, just execute and return an empty DataFrame
            self.conn
                .execute(&sql, params![])
                .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute DDL: {}", e)))?;

            // Return empty DataFrame for DDL statements
            return DataFrame::new(Vec::<polars::prelude::Column>::new()).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create empty DataFrame: {}", e))
            });
        }

        // Prepare and execute statement to get schema
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to prepare SQL: {}", e)))?;

        // Execute to populate schema info
        stmt.execute(params![])
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute SQL: {}", e)))?;

        // Get column metadata BEFORE creating iterator
        let column_count = stmt.column_count();
        if column_count == 0 {
            return Err(GgsqlError::ReaderError(
                "Query returned no columns".to_string(),
            ));
        }

        let mut column_names = Vec::new();
        let mut column_types = Vec::new();
        for i in 0..column_count {
            column_names.push(
                stmt.column_name(i)
                    .map_err(|e| {
                        GgsqlError::ReaderError(format!("Failed to get column name: {}", e))
                    })?
                    .to_string(),
            );
            let data_type = stmt.column_type(i);
            let duckdb_type = duckdb::types::Type::from(&data_type);
            column_types.push(duckdb_type);
        }

        // Initialize storage for each column
        let column_builders: Vec<ColumnBuilder> = column_types
            .iter()
            .map(|t| ColumnBuilder::new(t.clone()))
            .collect();

        // Collect all values using query_map (which borrows stmt mutably during iteration)
        let builders_cell = std::cell::RefCell::new(column_builders);
        let error_cell = std::cell::RefCell::new(None);

        let _ = stmt
            .query_map(params![], |row| {
                // Handle errors by storing them in error_cell
                if error_cell.borrow().is_some() {
                    return Ok(());
                }

                let mut builders = builders_cell.borrow_mut();
                for col_idx in 0..column_count {
                    if let Err(e) = builders[col_idx].add_value(row, col_idx) {
                        *error_cell.borrow_mut() = Some(e);
                        return Ok(());
                    }
                }
                Ok(())
            })
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to iterate rows: {}", e)))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to process rows: {}", e)))?;

        // Check if there was an error during processing
        if let Some(err) = error_cell.into_inner() {
            return Err(err);
        }

        // Build Series from column builders (may be empty if query returned 0 rows)
        let column_builders = builders_cell.into_inner();
        let mut columns = Vec::new();
        for (col_idx, builder) in column_builders.into_iter().enumerate() {
            let series = builder.build(&column_names[col_idx])?;
            columns.push(series.into());
        }

        // Create DataFrame from typed columns
        let df = DataFrame::new(columns)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to create DataFrame: {}", e)))?;

        Ok(df)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        // Validate table name
        validate_table_name(name)?;

        // Check for duplicates
        if !replace && self.table_exists(name)? {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' already exists",
                name
            )));
        }

        // Workaround for a duckdb-rs limitation (not a DuckDB limitation).
        //
        // duckdb-rs's `ArrowVTab` writes each RecordBatch into a single DuckDB
        // `DataChunk`, which has a fixed capacity of `STANDARD_VECTOR_SIZE`.
        // That constant is defined in DuckDB's C++ source at
        // `src/include/duckdb/common/constants.hpp` and is currently 2048.
        // When a RecordBatch exceeds this, `FlatVector::copy` panics with
        // `assertion failed: data.len() <= self.capacity()`.
        //
        // We chunk large DataFrames to stay within this limit. The first chunk
        // creates the table (letting DuckDB infer the schema from Arrow), and
        // subsequent chunks INSERT into it.
        const MAX_ARROW_BATCH_ROWS: usize = 2048;
        let total_rows = df.height();
        let create_or_replace = if replace {
            "CREATE OR REPLACE"
        } else {
            "CREATE"
        };

        if total_rows <= MAX_ARROW_BATCH_ROWS {
            // Small DataFrame: register in a single batch
            let params = dataframe_to_arrow_params(df)?;
            let sql = format!(
                "{} TEMP TABLE \"{}\" AS SELECT * FROM arrow(?, ?)",
                create_or_replace, name
            );
            self.conn.execute(&sql, params).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register table '{}': {}", name, e))
            })?;
        } else {
            // Large DataFrame: create table from first chunk, then insert remaining chunks
            let first_chunk = df.slice(0, MAX_ARROW_BATCH_ROWS);
            let params = dataframe_to_arrow_params(first_chunk)?;
            let create_sql = format!(
                "{} TEMP TABLE \"{}\" AS SELECT * FROM arrow(?, ?)",
                create_or_replace, name
            );
            self.conn.execute(&create_sql, params).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register table '{}': {}", name, e))
            })?;

            let mut offset = MAX_ARROW_BATCH_ROWS;
            while offset < total_rows {
                let chunk_size = std::cmp::min(MAX_ARROW_BATCH_ROWS, total_rows - offset);
                let chunk = df.slice(offset as i64, chunk_size);
                let params = dataframe_to_arrow_params(chunk)?;
                let insert_sql = format!("INSERT INTO \"{}\" SELECT * FROM arrow(?, ?)", name);
                self.conn.execute(&insert_sql, params).map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to insert chunk into table '{}': {}",
                        name, e
                    ))
                })?;
                offset += chunk_size;
            }
        }

        // Track the table so we can unregister it later
        self.registered_tables.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        // Only allow unregistering tables we created via register()
        if !self.registered_tables.borrow().contains(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        // Drop the temp table
        let sql = format!("DROP TABLE IF EXISTS \"{}\"", name);
        self.conn.execute(&sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        // Remove from tracking
        self.registered_tables.borrow_mut().remove(name);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_in_memory() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory");
        assert!(reader.is_ok());
    }

    #[test]
    fn test_simple_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql("SELECT 1 as x, 2 as y").unwrap();

        assert_eq!(df.shape(), (1, 2));
        assert_eq!(df.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_table_creation_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create table
        reader
            .connection()
            .execute("CREATE TABLE test(x INT, y INT)", params![])
            .unwrap();

        // Insert data
        reader
            .connection()
            .execute("INSERT INTO test VALUES (1, 2), (3, 4)", params![])
            .unwrap();

        // Query data
        let df = reader.execute_sql("SELECT * FROM test").unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_invalid_sql() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let result = reader.execute_sql("INVALID SQL SYNTAX");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_with_aggregation() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute("CREATE TABLE sales(region TEXT, revenue REAL)", params![])
            .unwrap();

        reader
            .connection()
            .execute(
                "INSERT INTO sales VALUES ('US', 100), ('US', 200), ('EU', 150)",
                params![],
            )
            .unwrap();

        let df = reader
            .execute_sql("SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
            .unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.get_column_names(), vec!["region", "total"]);
    }

    #[test]
    fn test_register_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create a DataFrame
        let df = DataFrame::new(vec![
            Column::new("x".into(), vec![1i32, 2, 3]),
            Column::new("y".into(), vec![10i32, 20, 30]),
        ])
        .unwrap();

        // Register the DataFrame
        reader.register("my_table", df, false).unwrap();

        // Query the registered table
        let result = reader
            .execute_sql("SELECT * FROM my_table ORDER BY x")
            .unwrap();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_register_duplicate_name_errors() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df1 = DataFrame::new(vec![Column::new("a".into(), vec![1i32])]).unwrap();
        let df2 = DataFrame::new(vec![Column::new("b".into(), vec![2i32])]).unwrap();

        // First registration should succeed
        reader.register("dup_table", df1, false).unwrap();

        // Second registration with same name should fail (when replace=false)
        let result = reader.register("dup_table", df2, false);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn test_register_invalid_table_names() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = DataFrame::new(vec![Column::new("a".into(), vec![1i32])]).unwrap();

        // Empty name
        let result = reader.register("", df.clone(), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Name with double quote
        let result = reader.register("bad\"name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        // Name with null byte
        let result = reader.register("bad\0name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        // Name too long
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
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create an empty DataFrame with schema
        let df = DataFrame::new(vec![
            Column::new("x".into(), Vec::<i32>::new()),
            Column::new("y".into(), Vec::<String>::new()),
        ])
        .unwrap();

        reader.register("empty_table", df, false).unwrap();

        // Query should return empty result with correct schema
        let result = reader.execute_sql("SELECT * FROM empty_table").unwrap();
        assert_eq!(result.shape(), (0, 2));
        assert_eq!(result.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_unregister() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = DataFrame::new(vec![Column::new("x".into(), vec![1i32, 2, 3])]).unwrap();

        reader.register("test_data", df, false).unwrap();

        // Should be queryable
        let result = reader.execute_sql("SELECT * FROM test_data").unwrap();
        assert_eq!(result.height(), 3);

        // Unregister
        reader.unregister("test_data").unwrap();

        // Should no longer exist
        let result = reader.execute_sql("SELECT * FROM test_data");
        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_not_registered() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create a table directly (not via register)
        reader
            .connection()
            .execute("CREATE TABLE user_table (x INT)", params![])
            .unwrap();

        // Should fail - we didn't register this via register()
        let result = reader.unregister("user_table");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("was not registered via this reader"));
    }

    #[test]
    fn test_reregister_after_unregister() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = DataFrame::new(vec![Column::new("x".into(), vec![1i32, 2, 3])]).unwrap();

        reader.register("data", df.clone(), false).unwrap();
        reader.unregister("data").unwrap();

        // Should be able to register again
        reader.register("data", df, false).unwrap();
        let result = reader.execute_sql("SELECT * FROM data").unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_register_large_dataframe() {
        // duckdb-rs Arrow vtab has a vector capacity of 2048 rows. DataFrames
        // larger than this must be chunked to avoid a panic.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

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

        // Verify row count
        let result = reader
            .execute_sql("SELECT COUNT(*) as cnt FROM large_table")
            .unwrap();
        let count = result.column("cnt").unwrap().i64().unwrap().get(0).unwrap();
        assert_eq!(count, n as i64);

        // Verify first and last rows survived chunking intact
        let result = reader
            .execute_sql("SELECT id, name FROM large_table ORDER BY id LIMIT 1")
            .unwrap();
        assert_eq!(
            result.column("id").unwrap().i32().unwrap().get(0).unwrap(),
            0
        );
        assert_eq!(
            result
                .column("name")
                .unwrap()
                .str()
                .unwrap()
                .get(0)
                .unwrap(),
            "item_0"
        );

        let result = reader
            .execute_sql("SELECT id, name FROM large_table ORDER BY id DESC LIMIT 1")
            .unwrap();
        assert_eq!(
            result.column("id").unwrap().i32().unwrap().get(0).unwrap(),
            (n - 1) as i32
        );
        assert_eq!(
            result
                .column("name")
                .unwrap()
                .str()
                .unwrap()
                .get(0)
                .unwrap(),
            format!("item_{}", n - 1)
        );
    }
}
