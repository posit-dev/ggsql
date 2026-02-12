//! Data source abstraction layer for ggsql
//!
//! The reader module provides a pluggable interface for executing SQL queries
//! against various data sources and returning Polars DataFrames for visualization.
//!
//! # Architecture
//!
//! All readers implement the `Reader` trait, which provides:
//! - SQL query execution → DataFrame conversion
//! - Visualization query execution → Spec
//! - Optional DataFrame registration for queryable tables
//! - Connection management and error handling
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::reader::{Reader, DuckDBReader};
//! use ggsql::writer::{Writer, VegaLiteWriter};
//!
//! // Execute a ggsql query
//! let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
//!
//! // Render to Vega-Lite JSON
//! let writer = VegaLiteWriter::new();
//! let json = writer.render(&spec)?;
//!
//! // With DataFrame registration
//! let mut reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! reader.register("my_table", some_dataframe)?;
//! let spec = reader.execute("SELECT * FROM my_table VISUALISE x, y DRAW point")?;
//! ```

use std::collections::HashMap;

use crate::execute::prepare_data_with_reader;
use crate::plot::{Plot, SqlTypeNames};
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, GgsqlError, Result};

#[cfg(feature = "duckdb")]
pub mod duckdb;

pub mod connection;
pub mod data;
mod spec;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

// ============================================================================
// Spec - Result of reader.execute()
// ============================================================================

/// Result of executing a ggsql query, ready for rendering.
pub struct Spec {
    /// Single resolved plot specification
    pub(crate) plot: Plot,
    /// Internal data map (global + layer-specific DataFrames)
    pub(crate) data: HashMap<String, DataFrame>,
    /// Cached metadata about the prepared visualization
    pub(crate) metadata: Metadata,
    /// The main SQL query that was executed
    pub(crate) sql: String,
    /// The raw VISUALISE portion text
    pub(crate) visual: String,
    /// Per-layer filter/source queries (None = uses global data directly)
    pub(crate) layer_sql: Vec<Option<String>>,
    /// Per-layer stat transform queries (None = no stat transform)
    pub(crate) stat_sql: Vec<Option<String>>,
    /// Validation warnings from preparation
    pub(crate) warnings: Vec<ValidationWarning>,
}

/// Metadata about the prepared visualization.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub rows: usize,
    pub columns: Vec<String>,
    pub layer_count: usize,
}

// ============================================================================
// Reader Trait
// ============================================================================

/// Trait for data source readers
///
/// Readers execute SQL queries and return Polars DataFrames.
/// They provide a uniform interface for different database backends.
///
/// # DataFrame Registration
///
/// Some readers support registering DataFrames as queryable tables using
/// the [`register`](Reader::register) method. This allows you to query
/// in-memory DataFrames with SQL, join them with other tables, etc.
///
/// ```rust,ignore
/// // Register a DataFrame (takes ownership)
/// reader.register("sales", sales_df)?;
///
/// // Now you can query it
/// let result = reader.execute_sql("SELECT * FROM sales WHERE amount > 100")?;
/// ```
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
    fn execute_sql(&self, sql: &str) -> Result<DataFrame>;

    /// Register a DataFrame as a queryable table (takes ownership)
    ///
    /// After registration, the DataFrame can be queried by name in SQL:
    /// ```sql
    /// SELECT * FROM <name> WHERE ...
    /// ```
    ///
    /// # Arguments
    ///
    /// * `name` - The table name to register under
    /// * `df` - The DataFrame to register (ownership is transferred)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if registration fails or isn't supported.
    ///
    /// # Default Implementation
    ///
    /// Returns an error by default. Override for readers that support registration.
    fn register(&mut self, name: &str, _df: DataFrame) -> Result<()> {
        Err(GgsqlError::ReaderError(format!(
            "This reader does not support DataFrame registration for table '{}'",
            name
        )))
    }

    /// Unregister a previously registered table
    ///
    /// # Arguments
    ///
    /// * `name` - The table name to unregister
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Default Implementation
    ///
    /// Returns an error by default. Override for readers that support registration.
    fn unregister(&mut self, name: &str) -> Result<()> {
        Err(GgsqlError::ReaderError(format!(
            "This reader does not support unregistering table '{}'",
            name
        )))
    }

    /// Check if this reader supports DataFrame registration
    ///
    /// # Returns
    ///
    /// `true` if [`register`](Reader::register) is implemented, `false` otherwise.
    fn supports_register(&self) -> bool {
        false
    }

    /// Execute a ggsql query and return the visualization specification.
    ///
    /// This is the main entry point for creating visualizations. It parses the query,
    /// executes the SQL portion, and returns a `Spec` ready for rendering.
    ///
    /// # Arguments
    ///
    /// * `query` - The ggsql query (SQL + VISUALISE clause)
    ///
    /// # Returns
    ///
    /// A `Spec` containing the resolved visualization specification and data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query syntax is invalid
    /// - The query has no VISUALISE clause
    /// - The SQL execution fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ggsql::reader::{Reader, DuckDBReader};
    /// use ggsql::writer::{Writer, VegaLiteWriter};
    ///
    /// let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
    /// let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
    ///
    /// let writer = VegaLiteWriter::new();
    /// let json = writer.render(&spec)?;
    /// ```
    #[cfg(feature = "duckdb")]
    fn execute(&self, query: &str) -> Result<Spec> {
        // Run validation first to capture warnings
        let validated = validate(query)?;
        let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

        // Prepare data with type names for this reader
        let prepared_data = prepare_data_with_reader(query, self)?;

        // Get the first (and typically only) spec
        let plot = prepared_data.specs.into_iter().next().ok_or_else(|| {
            GgsqlError::ValidationError("No visualization spec found".to_string())
        })?;

        // For now, layer_sql and stat_sql are not tracked in PreparedData
        // (they were part of main's version but not HEAD's)
        let layer_sql = vec![None; plot.layers.len()];
        let stat_sql = vec![None; plot.layers.len()];

        Ok(Spec::new(
            plot,
            prepared_data.data,
            prepared_data.sql,
            prepared_data.visual,
            layer_sql,
            stat_sql,
            warnings,
        ))
    }

    // =========================================================================
    // SQL Type Names for Casting
    // =========================================================================

    /// SQL type name for numeric columns (e.g., "DOUBLE", "FLOAT", "NUMERIC")
    ///
    /// Used for casting string columns to numbers for binning.
    /// Returns None if the database doesn't support this cast.
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    /// SQL type name for DATE columns (e.g., "DATE", "date")
    ///
    /// Used for casting string columns to dates for temporal binning.
    /// Returns None if the database doesn't support native date types.
    fn date_type_name(&self) -> Option<&str> {
        Some("DATE")
    }

    /// SQL type name for DATETIME/TIMESTAMP columns
    ///
    /// Used for casting string columns to timestamps for temporal binning.
    /// Returns None if the database doesn't support this type.
    fn datetime_type_name(&self) -> Option<&str> {
        Some("TIMESTAMP")
    }

    /// SQL type name for TIME columns
    ///
    /// Used for casting string columns to time values for temporal binning.
    /// Returns None if the database doesn't support this type.
    fn time_type_name(&self) -> Option<&str> {
        Some("TIME")
    }

    /// SQL type name for VARCHAR/TEXT columns
    ///
    /// Used for casting columns to string type.
    /// Returns None if the database doesn't support this cast.
    fn string_type_name(&self) -> Option<&str> {
        Some("VARCHAR")
    }

    /// SQL type name for BOOLEAN columns
    ///
    /// Used for casting columns to boolean type.
    /// Returns None if the database doesn't support this cast.
    fn boolean_type_name(&self) -> Option<&str> {
        Some("BOOLEAN")
    }

    /// SQL type name for INTEGER columns (e.g., "BIGINT", "INTEGER")
    ///
    /// Used for casting columns to integer type.
    /// Returns None if the database doesn't support this cast.
    fn integer_type_name(&self) -> Option<&str> {
        Some("BIGINT")
    }

    /// Get SQL type names for this reader.
    ///
    /// Returns a SqlTypeNames struct populated from the individual type name methods.
    /// This is useful for passing to functions that need all type names at once.
    fn sql_type_names(&self) -> SqlTypeNames {
        SqlTypeNames {
            number: self.number_type_name().map(String::from),
            integer: self.integer_type_name().map(String::from),
            date: self.date_type_name().map(String::from),
            datetime: self.datetime_type_name().map(String::from),
            time: self.time_type_name().map(String::from),
            string: self.string_type_name().map(String::from),
            boolean: self.boolean_type_name().map(String::from),
        }
    }
}

#[cfg(test)]
#[cfg(all(feature = "duckdb", feature = "vegalite"))]
mod tests {
    use super::*;
    use crate::writer::{VegaLiteWriter, Writer};

    #[test]
    fn test_execute_and_render() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")
            .unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert_eq!(spec.metadata().layer_count, 1);
        assert!(spec.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    #[test]
    fn test_execute_metadata() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute(
                "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y) VISUALISE x, y DRAW point",
            )
            .unwrap();

        let metadata = spec.metadata();
        assert_eq!(metadata.rows, 3);
        assert_eq!(metadata.columns.len(), 2);
        assert!(metadata.columns.contains(&"x".to_string()));
        assert!(metadata.columns.contains(&"y".to_string()));
        assert_eq!(metadata.layer_count, 1);
    }

    #[test]
    fn test_execute_with_cte() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            WITH data AS (
                SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
            )
            SELECT * FROM data
            VISUALISE x, y DRAW point
        "#;

        let spec = reader.execute(query).unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert!(spec.layer_data(0).is_some());
        let df = spec.layer_data(0).unwrap();
        assert_eq!(df.height(), 2);
    }

    #[test]
    fn test_render_multi_layer() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW line MAPPING x AS x, y AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        assert!(result.contains("layer"));
    }

    #[test]
    fn test_register_and_query() {
        use polars::prelude::*;

        let mut reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df = df! {
            "x" => [1i32, 2, 3],
            "y" => [10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_data", df).unwrap();

        let query = "SELECT * FROM my_data VISUALISE x, y DRAW point";
        let spec = reader.execute(query).unwrap();

        assert_eq!(spec.metadata().rows, 3);
        assert!(spec.metadata().columns.contains(&"x".to_string()));

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    #[test]
    fn test_register_and_join() {
        use polars::prelude::*;

        let mut reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sales = df! {
            "id" => [1i32, 2, 3],
            "amount" => [100i32, 200, 300],
            "product_id" => [1i32, 1, 2],
        }
        .unwrap();

        let products = df! {
            "id" => [1i32, 2],
            "name" => ["Widget", "Gadget"],
        }
        .unwrap();

        reader.register("sales", sales).unwrap();
        reader.register("products", products).unwrap();

        let query = r#"
            SELECT s.id, s.amount, p.name
            FROM sales s
            JOIN products p ON s.product_id = p.id
            VISUALISE id AS x, amount AS y
            DRAW bar
        "#;

        let spec = reader.execute(query).unwrap();
        assert_eq!(spec.metadata().rows, 3);
    }

    #[test]
    fn test_execute_no_viz_fails() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y";

        let result = reader.execute(query);
        assert!(result.is_err());
    }
}
