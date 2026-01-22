//! DuckDB data source implementation
//!
//! Provides a reader for DuckDB databases with direct Polars DataFrame integration.

use crate::reader::data::init_builtin_data;
use crate::reader::{connection::ConnectionInfo, Reader};
use crate::{DataFrame, GgsqlError, Result};
use duckdb::{params, Connection};

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
/// let df = reader.execute("SELECT 1 as x, 2 as y")?;
///
/// // File-based database
/// let reader = DuckDBReader::from_connection_string("duckdb://data.db")?;
/// let df = reader.execute("SELECT * FROM sales")?;
/// ```
pub struct DuckDBReader {
    conn: Connection,
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

        Ok(Self { conn })
    }

    /// Get a reference to the underlying DuckDB connection
    ///
    /// Useful for executing setup queries (CREATE TABLE, INSERT, etc.)
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
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
    fn execute(&self, sql: &str) -> Result<DataFrame> {
        use polars::prelude::*;

        // Check if this is a DDL statement (CREATE, DROP, INSERT, UPDATE, DELETE, ALTER)
        // DDL statements don't return rows, so we handle them specially
        let trimmed = sql.trim().to_uppercase();
        let is_ddl = trimmed.starts_with("CREATE ")
            || trimmed.starts_with("DROP ")
            || trimmed.starts_with("INSERT ")
            || trimmed.starts_with("UPDATE ")
            || trimmed.starts_with("DELETE ")
            || trimmed.starts_with("ALTER ");

        // Initialise built-in datasets
        let inits = init_builtin_data(sql)?;
        for init in inits {
            if let Err(e) = self.conn.execute(&init, params![]) {
                return Err(GgsqlError::ReaderError(format!(
                    "Failed to initialise built-in dataset: {}",
                    e
                )));
            }
        }

        if is_ddl {
            // For DDL, just execute and return an empty DataFrame
            self.conn
                .execute(sql, params![])
                .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute DDL: {}", e)))?;

            // Return empty DataFrame for DDL statements
            return DataFrame::new(Vec::<polars::prelude::Column>::new()).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create empty DataFrame: {}", e))
            });
        }

        // Prepare and execute statement to get schema
        let mut stmt = self
            .conn
            .prepare(sql)
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

    fn validate_columns(&self, sql: &str, columns: &[String]) -> Result<()> {
        // Execute the query to get the schema
        let df = self.execute(sql)?;

        // Get column names from the DataFrame
        let schema_columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Check if all required columns exist
        for col in columns {
            if !schema_columns.contains(col) {
                return Err(GgsqlError::ValidationError(format!(
                    "Column '{}' not found in query result. Available columns: {}",
                    col,
                    schema_columns.join(", ")
                )));
            }
        }

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
        let df = reader.execute("SELECT 1 as x, 2 as y").unwrap();

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
        let df = reader.execute("SELECT * FROM test").unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_validate_columns_success() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let sql = "SELECT 1 as x, 2 as y";

        let result = reader.validate_columns(sql, &["x".to_string(), "y".to_string()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_columns_missing() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let sql = "SELECT 1 as x, 2 as y";

        let result = reader.validate_columns(sql, &["z".to_string()]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Column 'z' not found"));
    }

    #[test]
    fn test_invalid_sql() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let result = reader.execute("INVALID SQL SYNTAX");
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
            .execute("SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
            .unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.get_column_names(), vec!["region", "total"]);
    }
}
