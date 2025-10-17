//! DuckDB data source implementation
//!
//! Provides a reader for DuckDB databases with direct Polars DataFrame integration.

use crate::reader::{connection::ConnectionInfo, Reader};
use crate::{DataFrame, Result, VizqlError};
use duckdb::{params, Connection};

/// DuckDB database reader
///
/// Executes SQL queries against DuckDB databases (in-memory or file-based)
/// and returns results as Polars DataFrames.
///
/// # Examples
///
/// ```rust,ignore
/// use vizql::reader::{Reader, DuckDBReader};
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
    /// Helper function to convert a DuckDB value to a string
    fn value_to_string(row: &duckdb::Row, idx: usize) -> String {
        use duckdb::types::ValueRef;

        // Use get_ref to avoid panics on type mismatches
        match row.get_ref(idx) {
            Ok(ValueRef::Null) => String::new(),
            Ok(ValueRef::Boolean(b)) => b.to_string(),
            Ok(ValueRef::TinyInt(i)) => i.to_string(),
            Ok(ValueRef::SmallInt(i)) => i.to_string(),
            Ok(ValueRef::Int(i)) => i.to_string(),
            Ok(ValueRef::BigInt(i)) => i.to_string(),
            Ok(ValueRef::HugeInt(i)) => i.to_string(),
            Ok(ValueRef::UTinyInt(i)) => i.to_string(),
            Ok(ValueRef::USmallInt(i)) => i.to_string(),
            Ok(ValueRef::UInt(i)) => i.to_string(),
            Ok(ValueRef::UBigInt(i)) => i.to_string(),
            Ok(ValueRef::Float(f)) => f.to_string(),
            Ok(ValueRef::Double(f)) => f.to_string(),
            Ok(ValueRef::Decimal(d)) => d.to_string(),
            Ok(ValueRef::Text(s)) => String::from_utf8_lossy(s).to_string(),
            Ok(ValueRef::Blob(b)) => format!("{:?}", b), // Debug format for binary data
            Ok(ValueRef::Date32(d)) => d.to_string(),
            Ok(ValueRef::Time64(_, t)) => t.to_string(),
            Ok(ValueRef::Timestamp(_, ts)) => ts.to_string(),
            // Fallback for any other types or errors
            _ => String::new(),
        }
    }

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
                VizqlError::ReaderError(format!("Failed to open in-memory DuckDB: {}", e))
            })?,
            ConnectionInfo::DuckDBFile(path) => Connection::open(&path).map_err(|e| {
                VizqlError::ReaderError(format!("Failed to open DuckDB file '{}': {}", path, e))
            })?,
            _ => {
                return Err(VizqlError::ReaderError(format!(
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

impl Reader for DuckDBReader {
    fn execute(&self, sql: &str) -> Result<DataFrame> {
        // Execute query using DuckDB's rows API and manually build DataFrame
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| VizqlError::ReaderError(format!("Failed to prepare SQL: {}", e)))?;

        // Execute the query
        let mut rows = stmt
            .query(params![])
            .map_err(|e| VizqlError::ReaderError(format!("Failed to execute query: {}", e)))?;

        // Collect all rows into vectors and extract column names from first row
        let mut row_data: Vec<Vec<String>> = Vec::new();
        let mut column_names: Vec<String> = Vec::new();
        let column_count = 0;

        // Process first row to get column information
        if let Some(first_row) = rows.next().map_err(|e| {
            VizqlError::ReaderError(format!("Failed to fetch first row: {}", e))
        })? {
            // Get column count from the row
            let column_count = first_row.as_ref().column_count();

            if column_count == 0 {
                return Err(VizqlError::ReaderError(
                    "Query returned no columns".to_string(),
                ));
            }

            // Get column names from the row
            for i in 0..column_count {
                column_names.push(
                    first_row.as_ref().column_name(i)
                        .map_err(|e| VizqlError::ReaderError(format!("Failed to get column name: {}", e)))?
                        .to_string(),
                );
            }

            // Extract data from first row
            let mut first_row_vec = Vec::new();
            for i in 0..column_count {
                let val = Self::value_to_string(&first_row, i);
                first_row_vec.push(val);
            }
            row_data.push(first_row_vec);
        } else {
            return Err(VizqlError::ReaderError(
                "Query returned no rows".to_string(),
            ));
        }

        // Collect remaining rows
        while let Some(row) = rows.next().map_err(|e| {
            VizqlError::ReaderError(format!("Failed to fetch row: {}", e))
        })? {
            let mut row_vec = Vec::new();
            for i in 0..column_count {
                let val = Self::value_to_string(&row, i);
                row_vec.push(val);
            }
            row_data.push(row_vec);
        }

        // Convert to Polars DataFrame
        use polars::prelude::*;

        let mut series_vec = Vec::new();
        for (col_idx, col_name) in column_names.iter().enumerate() {
            let col_data: Vec<String> = row_data.iter().map(|row| row[col_idx].clone()).collect();
            let series = Series::new(col_name.into(), col_data);
            series_vec.push(series);
        }

        DataFrame::new(series_vec).map_err(|e| {
            VizqlError::ReaderError(format!("Failed to create DataFrame: {}", e))
        })
    }

    fn validate_columns(&self, sql: &str, columns: &[String]) -> Result<()> {
        // Execute the query to get the schema
        let df = self.execute(sql)?;

        // Get column names from the DataFrame
        let schema_columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

        // Check if all required columns exist
        for col in columns {
            if !schema_columns.contains(col) {
                return Err(VizqlError::ValidationError(format!(
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
            .execute(
                "CREATE TABLE sales(region TEXT, revenue REAL)",
                params![],
            )
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
