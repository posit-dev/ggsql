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
//! reader.register("my_table", some_dataframe, false)?;
//! let spec = reader.execute("SELECT * FROM my_table VISUALISE x, y DRAW point")?;
//! ```

use std::collections::HashMap;

use crate::execute::prepare_data_with_reader;
use crate::plot::{Plot, SqlTypeNames};
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, GgsqlError, Result};

#[cfg(feature = "duckdb")]
pub mod duckdb;

#[cfg(feature = "polars-sql")]
pub mod polars_sql;

pub mod connection;
pub mod data;
mod spec;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

#[cfg(feature = "polars-sql")]
pub use polars_sql::PolarsReader;

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
/// Readers support registering DataFrames as queryable tables using
/// the [`register`](Reader::register) method. This allows you to query
/// in-memory DataFrames with SQL, join them with other tables, etc.
///
/// ```rust,ignore
/// // Register a DataFrame (takes ownership)
/// reader.register("sales", sales_df, false)?;
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
    /// * `replace` - If true, replace any existing table with the same name.
    ///   If false, return an error if the table already exists.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if registration fails.
    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()>;

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
    /// Returns an error by default. Override for readers that support unregistration.
    fn unregister(&self, name: &str) -> Result<()> {
        Err(GgsqlError::ReaderError(format!(
            "This reader does not support unregistering table '{}'",
            name
        )))
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
    /// let mut reader = DuckDBReader::from_connection_string("duckdb://memory")?;
    /// let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
    ///
    /// let writer = VegaLiteWriter::new();
    /// let json = writer.render(&spec)?;
    /// ```
    fn execute(&self, query: &str) -> Result<Spec>
    where
        Self: Sized,
    {
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
        // Columns now includes both user mappings (x, y) and resolved defaults (size, stroke, fill, opacity, shape, linewidth)
        assert_eq!(metadata.columns.len(), 8);
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

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df = df! {
            "x" => [1i32, 2, 3],
            "y" => [10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_data", df, false).unwrap();

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

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

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

        reader.register("sales", sales, false).unwrap();
        reader.register("products", products, false).unwrap();

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

    #[test]
    fn test_binned_fill_legend_renders_threshold_scale() {
        // End-to-end test for binned fill scale rendering to Vega-Lite
        // Verifies that binned non-positional aesthetics use threshold scale type
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with values that span the binned range
        // Binned scales use FROM [min, max] for range and SETTING breaks => [...] for explicit breaks
        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 15.0),
                (2, 20, 35.0),
                (3, 30, 55.0),
                (4, 40, 85.0)
            ) AS t(x, y, value)
            VISUALISE
            DRAW tile MAPPING x AS x, y AS y, value AS fill
            SCALE BINNED fill FROM [0, 100] TO viridis SETTING breaks => [0, 25, 50, 75, 100]
        "#;

        let spec = reader.execute(query).unwrap();

        // Verify spec structure
        assert_eq!(spec.plot().layers.len(), 1);
        // Note: scales may include auto-generated x/y scales plus the explicit fill scale
        assert!(
            spec.plot().find_scale("fill").is_some(),
            "Should have a fill scale"
        );

        // Render to Vega-Lite
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

        // Verify threshold scale type for fill
        let fill_scale = &vl["layer"][0]["encoding"]["fill"]["scale"];
        assert_eq!(
            fill_scale["type"],
            "threshold",
            "Binned fill should use threshold scale type. Got: {}",
            serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]).unwrap()
        );

        // Verify internal breaks as domain (excludes first and last terminals)
        // breaks = [0, 25, 50, 75, 100] → domain = [25, 50, 75]
        let domain = fill_scale["domain"].as_array().unwrap();
        assert_eq!(
            domain.len(),
            3,
            "Threshold domain should have internal breaks only. Got: {:?}",
            domain
        );
        assert_eq!(domain[0], 25.0);
        assert_eq!(domain[1], 50.0);
        assert_eq!(domain[2], 75.0);

        // Verify color output - viridis palette gets expanded to an explicit range array
        // for threshold scales (Vega-Lite needs explicit colors for threshold domain)
        assert!(
            fill_scale["range"].is_array() || fill_scale["scheme"] == "viridis",
            "Should have color range or scheme. Got scale: {}",
            serde_json::to_string_pretty(fill_scale).unwrap()
        );

        // Verify legend values
        // For `fill` alone (single binned legend scale), uses gradient legend with all 5 break values
        // For symbol legends (multiple binned scales or non-gradient aesthetics), would have N-1 values
        let legend_values = &vl["layer"][0]["encoding"]["fill"]["legend"]["values"];
        assert!(
            legend_values.is_array(),
            "Legend should have values array. Got: {}",
            serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]["legend"]).unwrap()
        );
        let values = legend_values.as_array().unwrap();
        assert_eq!(
            values.len(),
            5,
            "Gradient legend should have all 5 break values. Got: {:?}",
            values
        );
    }

    #[test]
    fn test_binned_color_legend_with_label_mapping() {
        // Test binned color scale with custom labels renders correctly
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 20.0),
                (2, 20, 60.0),
                (3, 30, 90.0)
            ) AS t(x, y, score)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, score AS color
            SCALE BINNED color FROM [0, 100] TO ['blue', 'yellow', 'red'] SETTING breaks => [0, 50, 100]
                RENAMING 0 => 'Low', 50 => 'High'
        "#;

        let spec = reader.execute(query).unwrap();

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

        // Verify threshold scale
        // Note: "color" aesthetic is mapped to "stroke" for point geom (not fill)
        let encoding = if vl["layer"].is_array() {
            &vl["layer"][0]["encoding"]
        } else {
            &vl["encoding"]
        };
        // Find the stroke or fill encoding (color maps to one of these)
        let color_encoding = if encoding["stroke"].is_object() {
            &encoding["stroke"]
        } else {
            &encoding["fill"]
        };
        assert_eq!(
            color_encoding["scale"]["type"],
            "threshold",
            "Binned color should use threshold scale. Got encoding: {}",
            serde_json::to_string_pretty(color_encoding).unwrap()
        );

        // Verify labelExpr exists for custom labels
        let legend = &color_encoding["legend"];
        assert!(
            legend["labelExpr"].is_string(),
            "Legend should have labelExpr for custom labels. Got legend: {}",
            serde_json::to_string_pretty(legend).unwrap()
        );

        let label_expr = legend["labelExpr"].as_str().unwrap_or("");
        // For symbol legends, VL generates range-style labels like "0 – 50"
        // Our labelExpr should map these to custom range formats
        assert!(
            label_expr.contains("Low") || label_expr.contains("High"),
            "labelExpr should contain custom labels, got: {}",
            label_expr
        );
    }
}
