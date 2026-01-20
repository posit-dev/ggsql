//! Data source abstraction layer for ggsql
//!
//! The reader module provides a pluggable interface for executing SQL queries
//! against various data sources and returning Polars DataFrames for visualization.
//!
//! # Architecture
//!
//! All readers implement the `Reader` trait, which provides:
//! - SQL query execution → DataFrame conversion
//! - Column validation for query introspection
//! - Connection management and error handling
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::reader::{Reader, DuckDBReader};
//!
//! let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! let df = reader.execute("SELECT * FROM table")?;
//! ```

use crate::{DataFrame, Result};

#[cfg(feature = "duckdb")]
pub mod duckdb;

pub mod connection;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

/// Trait for data source readers
///
/// Readers execute SQL queries and return Polars DataFrames.
/// They provide a uniform interface for different database backends.
pub trait Reader {
    /// Execute a SQL query and return the result as a DataFrame
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL query to execute
    ///
    /// # Returns
    ///
    /// A Polars DataFrame containing the query results
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::ReaderError` if:
    /// - The SQL is invalid
    /// - The connection fails
    /// - The table or columns don't exist
    fn execute(&self, sql: &str) -> Result<DataFrame>;

    /// Validate that specified columns exist in a query result
    ///
    /// This is useful for checking column names before visualization
    /// to provide better error messages.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL query to introspect
    /// * `columns` - Column names to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if all columns exist, otherwise an error
    fn validate_columns(&self, sql: &str, columns: &[String]) -> Result<()>;
}
