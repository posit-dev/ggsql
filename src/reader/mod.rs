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
use crate::plot::{CastTargetType, Plot};
use crate::validate::{validate, ValidationWarning};
use crate::{naming, DataFrame, GgsqlError, Result};

// =============================================================================
// SQL Dialect
// =============================================================================

/// SQL type names and functionality in the syntax supported by that backend.
///
/// Default implementations produce portable ANSI SQL.
pub trait SqlDialect {
    /// SQL type name for numeric columns (e.g., "DOUBLE PRECISION")
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE PRECISION")
    }

    /// SQL type name for integer columns (e.g., "BIGINT")
    fn integer_type_name(&self) -> Option<&str> {
        Some("BIGINT")
    }

    /// SQL type name for DATE columns (e.g., "DATE")
    fn date_type_name(&self) -> Option<&str> {
        Some("DATE")
    }

    /// SQL type name for DATETIME/TIMESTAMP columns
    fn datetime_type_name(&self) -> Option<&str> {
        Some("TIMESTAMP")
    }

    /// SQL type name for TIME columns
    fn time_type_name(&self) -> Option<&str> {
        Some("TIME")
    }

    /// SQL type name for STRING/VARCHAR columns
    fn string_type_name(&self) -> Option<&str> {
        Some("VARCHAR")
    }

    /// SQL type name for BOOLEAN columns
    fn boolean_type_name(&self) -> Option<&str> {
        Some("BOOLEAN")
    }

    /// Get the SQL type name for a cast target type.
    fn type_name_for(&self, target: CastTargetType) -> Option<&str> {
        match target {
            CastTargetType::Number => self.number_type_name(),
            CastTargetType::Integer => self.integer_type_name(),
            CastTargetType::Date => self.date_type_name(),
            CastTargetType::DateTime => self.datetime_type_name(),
            CastTargetType::Time => self.time_type_name(),
            CastTargetType::String => self.string_type_name(),
            CastTargetType::Boolean => self.boolean_type_name(),
        }
    }

    // =========================================================================
    // Schema introspection queries (for Connections pane)
    // =========================================================================

    /// SQL to list catalog names. Returns rows with column `catalog_name`.
    fn sql_list_catalogs(&self) -> String {
        "SELECT DISTINCT catalog_name FROM information_schema.schemata ORDER BY catalog_name".into()
    }

    /// SQL to list schema names within a catalog. Returns rows with column `schema_name`.
    fn sql_list_schemas(&self, catalog: &str) -> String {
        format!(
            "SELECT DISTINCT schema_name FROM information_schema.schemata \
             WHERE catalog_name = '{}' ORDER BY schema_name",
            catalog.replace('\'', "''")
        )
    }

    /// SQL to list tables/views within a catalog and schema.
    /// Returns rows with columns `table_name` and `table_type`.
    fn sql_list_tables(&self, catalog: &str, schema: &str) -> String {
        format!(
            "SELECT DISTINCT table_name, table_type FROM information_schema.tables \
             WHERE table_catalog = '{}' AND table_schema = '{}' ORDER BY table_name",
            catalog.replace('\'', "''"),
            schema.replace('\'', "''")
        )
    }

    /// SQL to list columns in a table.
    /// Returns rows with columns `column_name` and `data_type`.
    fn sql_list_columns(&self, catalog: &str, schema: &str, table: &str) -> String {
        format!(
            "SELECT column_name, data_type FROM information_schema.columns \
             WHERE table_catalog = '{}' AND table_schema = '{}' AND table_name = '{}' \
             ORDER BY ordinal_position",
            catalog.replace('\'', "''"),
            schema.replace('\'', "''"),
            table.replace('\'', "''")
        )
    }

    /// Scalar MAX across any number of SQL expressions.
    fn sql_greatest(&self, exprs: &[&str]) -> String {
        let mut result = exprs[0].to_string();
        for expr in &exprs[1..] {
            result =
                format!("(CASE WHEN ({result}) >= ({expr}) THEN ({result}) ELSE ({expr}) END)");
        }
        result
    }

    /// Scalar MIN across any number of SQL expressions.
    fn sql_least(&self, exprs: &[&str]) -> String {
        let mut result = exprs[0].to_string();
        for expr in &exprs[1..] {
            result =
                format!("(CASE WHEN ({result}) <= ({expr}) THEN ({result}) ELSE ({expr}) END)");
        }
        result
    }

    /// Generate a series of integers 0..n-1 as a CTE fragment.
    ///
    /// Returns CTE fragment(s) producing table `__ggsql_seq__` with column `n`.
    fn sql_generate_series(&self, n: usize) -> String {
        // Uses a cube-root decomposition to avoid deep recursion: only recurses
        // ~cbrt(n) times, then cross-joins three copies to cover the full range.
        let base_size = (n as f64).cbrt().ceil() as usize;
        let base_sq = base_size * base_size;
        let base_max = base_size - 1;
        format!(
            "\"__ggsql_base__\"(n) AS (\
               SELECT 0 UNION ALL SELECT n + 1 FROM \"__ggsql_base__\" WHERE n < {base_max}\
             ),\
             \"__ggsql_seq__\"(n) AS (\
               SELECT CAST(a.n * {base_sq} + b.n * {base_size} + c.n AS REAL) AS n \
               FROM \"__ggsql_base__\" a, \"__ggsql_base__\" b, \"__ggsql_base__\" c \
               WHERE a.n * {base_sq} + b.n * {base_size} + c.n < {n}\
             )"
        )
    }

    /// Compute a percentile of a column
    ///
    /// Returns a scalar subquery expression that computes the specified percentile
    /// of a column within an optional grouping context.
    fn sql_percentile(&self, column: &str, fraction: f64, from: &str, groups: &[String]) -> String {
        // Uses NTILE(4) to divide data into quartiles, then interpolates between boundaries.
        let group_filter = groups
            .iter()
            .map(|g| {
                let q = naming::quote_ident(g);
                format!(
                    "AND {pct}.{q} IS NOT DISTINCT FROM {qt}.{q}",
                    pct = naming::quote_ident("__ggsql_pct__"),
                    qt = naming::quote_ident("__ggsql_qt__")
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        let lo_tile = (fraction * 4.0).ceil() as usize;
        let hi_tile = lo_tile + 1;
        let quoted_column = naming::quote_ident(column);

        format!(
            "(SELECT (\
              MAX(CASE WHEN __tile = {lo_tile} THEN __val END) + \
              MIN(CASE WHEN __tile = {hi_tile} THEN __val END)\
            ) / 2.0 \
            FROM (\
              SELECT {column} AS __val, \
                     NTILE(4) OVER (ORDER BY {column}) AS __tile \
              FROM ({from}) AS \"__ggsql_pct__\" \
              WHERE {column} IS NOT NULL {group_filter}\
            ))",
            column = quoted_column
        )
    }

    /// SQL literal for a date value (days since Unix epoch).
    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!(
            "CAST(DATE '1970-01-01' + INTERVAL {} DAY AS DATE)",
            days_since_epoch
        )
    }

    /// SQL literal for a datetime value (microseconds since Unix epoch).
    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!(
            "TIMESTAMP '1970-01-01 00:00:00' + INTERVAL {} MICROSECOND",
            microseconds_since_epoch
        )
    }

    /// SQL literal for a time value (nanoseconds since midnight).
    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let seconds = nanoseconds_since_midnight / 1_000_000_000;
        let nanos = nanoseconds_since_midnight % 1_000_000_000;
        format!(
            "TIME '00:00:00' + INTERVAL {} SECOND + INTERVAL {} NANOSECOND",
            seconds, nanos
        )
    }

    /// SQL literal for a boolean value.
    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        }
    }

    /// Build the DDL statement(s) needed to (re)create a temporary table
    /// that holds the result of `body_sql`.
    ///
    /// Column aliases from `WITH t(a, b) AS (...)` are preserved portably by
    /// wrapping the body in a named CTE with a column alias list, so the
    /// backend never needs to support `CREATE TABLE t(a, b) AS ...` syntax.
    ///
    /// Returned statements must be executed in order via `Reader::execute_sql`.
    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let body = wrap_with_column_aliases(body_sql, column_aliases);
        vec![format!(
            "CREATE OR REPLACE TEMP TABLE {} AS {}",
            naming::quote_ident(name),
            body
        )]
    }
}

/// Wrap a body SQL in a CTE with a column alias list when aliases are present.
/// This is a portable way to rename the body's output columns without relying
/// on `CREATE TABLE t(a, b) AS ...` (which SQLite does not support).
pub(crate) fn wrap_with_column_aliases(body_sql: &str, column_aliases: &[String]) -> String {
    if column_aliases.is_empty() {
        return body_sql.to_string();
    }
    let cols = column_aliases
        .iter()
        .map(|c| naming::quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "WITH __ggsql_aliased__({}) AS ({}) SELECT * FROM __ggsql_aliased__",
        cols, body_sql
    )
}

pub struct AnsiDialect;
impl SqlDialect for AnsiDialect {}

#[cfg(feature = "duckdb")]
pub mod duckdb;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "odbc")]
pub mod odbc;

#[cfg(feature = "odbc")]
pub mod snowflake;

pub mod connection;
pub mod data;
mod spec;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteReader;

#[cfg(feature = "odbc")]
pub use odbc::OdbcReader;

// ============================================================================
// Shared utilities
// ============================================================================

/// Validate a table name for use in SQL statements.
///
/// Rejects empty names and names containing null bytes or newlines.
pub(crate) fn validate_table_name(name: &str) -> Result<()> {
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

/// Does the SQL statement return rows?
///
/// Looks at the first keyword to decide: `SELECT`, `WITH`, `FROM`,
/// `DESCRIBE`, `SHOW` and `EXPLAIN` produce result sets; everything else
/// (DDL, DML) does not.
pub(crate) fn returns_rows(sql: &str) -> bool {
    let first_word = sql.split_whitespace().next().unwrap_or("");
    matches!(
        first_word.to_ascii_uppercase().as_str(),
        "SELECT" | "WITH" | "DESCRIBE" | "SHOW" | "EXPLAIN" | "FROM"
    )
}

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
    fn execute(&self, query: &str) -> Result<Spec>;

    /// Get the SQL dialect for this reader.
    ///
    /// Database-specific SQL type names and SQL generation methods
    fn dialect(&self) -> &dyn SqlDialect {
        &AnsiDialect
    }
}

/// Execute a ggsql query using any reader
///
/// This is the shared implementation behind `Reader::execute()`. Concrete
/// readers delegate to this so the trait stays object-safe (no `Self: Sized`
/// bound on `execute`).
pub fn execute_with_reader(reader: &dyn Reader, query: &str) -> Result<Spec> {
    let validated = validate(query)?;
    let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

    let prepared_data = prepare_data_with_reader(query, reader)?;

    let plot =
        prepared_data.specs.into_iter().next().ok_or_else(|| {
            GgsqlError::ValidationError("No visualization spec found".to_string())
        })?;

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

#[cfg(test)]
#[cfg(all(feature = "duckdb", feature = "vegalite"))]
mod tests {
    use super::*;
    use crate::df;
    use crate::writer::{VegaLiteWriter, Writer};

    fn data_layer(json: &serde_json::Value, index: usize) -> &serde_json::Value {
        json["layer"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|l| {
                !matches!(
                    l.get("description").and_then(|d| d.as_str()),
                    Some("background" | "foreground")
                )
            })
            .nth(index)
            .expect("data layer not found at index")
    }

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
        // Columns now includes both user mappings (pos1, pos2) and resolved defaults (size, stroke, fill, opacity, shape, linewidth)
        // Aesthetics are transformed to internal names (x -> pos1, y -> pos2)
        assert!(metadata.columns.contains(&"pos1".to_string()));
        assert!(metadata.columns.contains(&"pos2".to_string()));
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
    fn test_polar_project_with_start() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING start => 90
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Parse the JSON to verify the theta scale range is set correctly
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();

        // The encoding should have a theta channel with a scale range offset by 90 degrees
        // 90 degrees = π/2 radians
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        assert!(theta.is_object(), "theta encoding should exist");

        // Check that the scale has a range with the start offset
        let scale = &theta["scale"];
        let range = scale["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);

        // π/2 ≈ 1.5707963
        let start = range[0].as_f64().unwrap();
        assert!(
            (start - std::f64::consts::FRAC_PI_2).abs() < 0.001,
            "start should be π/2 (90 degrees), got {}",
            start
        );

        // π/2 + 2π ≈ 7.8539816
        let end = range[1].as_f64().unwrap();
        let expected_end = std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI;
        assert!(
            (end - expected_end).abs() < 0.001,
            "end should be π/2 + 2π, got {}",
            end
        );
    }

    #[test]
    fn test_polar_project_default_start() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Parse the JSON
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();

        // The theta encoding should NOT have a scale with range when start is 0 (default)
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        assert!(theta.is_object(), "theta encoding should exist");

        // Either no scale, or no range in scale (since default is 0)
        if let Some(scale) = theta.get("scale") {
            assert!(
                scale.get("range").is_none(),
                "theta scale should not have range when start is 0"
            );
        }
    }

    #[test]
    fn test_polar_project_with_end() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING start => -90, end => 90
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        let range = theta["scale"]["range"].as_array().unwrap();

        // -90° = -π/2 ≈ -1.5708, 90° = π/2 ≈ 1.5708
        let start = range[0].as_f64().unwrap();
        let end = range[1].as_f64().unwrap();
        assert!(
            (start - (-std::f64::consts::FRAC_PI_2)).abs() < 0.001,
            "start should be -π/2 (-90 degrees), got {}",
            start
        );
        assert!(
            (end - std::f64::consts::FRAC_PI_2).abs() < 0.001,
            "end should be π/2 (90 degrees), got {}",
            end
        );
    }

    #[test]
    fn test_polar_project_with_end_only() {
        // Test using end without explicit start (start defaults to 0)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING end => 180
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        let range = theta["scale"]["range"].as_array().unwrap();

        // start=0 (default), end=180° = π
        let start = range[0].as_f64().unwrap();
        let end = range[1].as_f64().unwrap();
        assert!(
            start.abs() < 0.001,
            "start should be 0 (default), got {}",
            start
        );
        assert!(
            (end - std::f64::consts::PI).abs() < 0.001,
            "end should be π (180 degrees), got {}",
            end
        );
    }

    #[test]
    fn test_polar_encoding_keys_independent_of_user_names() {
        // This test verifies that polar projections always produce theta/radius encoding keys
        // in Vega-Lite output, regardless of what position names the user specified in PROJECT.
        // This is critical because Vega-Lite expects specific channel names for polar marks.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Helper to check encoding keys
        fn check_encoding_keys(json: &serde_json::Value, test_name: &str) {
            let layer = data_layer(json, 0);
            assert!(
                layer["encoding"].get("theta").is_some(),
                "{} should produce theta encoding, got keys: {:?}",
                test_name,
                layer["encoding"]
                    .as_object()
                    .map(|o| o.keys().collect::<Vec<_>>())
            );
            // Also verify no x or y keys exist (they should be mapped to theta/radius)
            assert!(
                layer["encoding"].get("x").is_none(),
                "{} should NOT have x encoding in polar mode",
                test_name
            );
            assert!(
                layer["encoding"].get("y").is_none(),
                "{} should NOT have y encoding in polar mode",
                test_name
            );
        }

        // Test case 1: PROJECT y, x TO polar (y as pos1→radius, x as pos2→theta)
        let query1 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar
        "#;
        let spec1 = reader.execute(query1).unwrap();
        let writer = VegaLiteWriter::new();
        let result1 = writer.render(&spec1).unwrap();
        let json1: serde_json::Value = serde_json::from_str(&result1).unwrap();
        check_encoding_keys(&json1, "PROJECT y, x TO polar");

        // Test case 2: PROJECT x, y TO polar (x as pos1→radius, y as pos2→theta)
        let query2 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS x, category AS fill
            DRAW bar
            PROJECT x, y TO polar
        "#;
        let spec2 = reader.execute(query2).unwrap();
        let result2 = writer.render(&spec2).unwrap();
        let json2: serde_json::Value = serde_json::from_str(&result2).unwrap();
        check_encoding_keys(&json2, "PROJECT x, y TO polar");

        // Test case 3: PROJECT TO polar (default radius/angle names)
        let query3 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS angle, category AS fill
            DRAW bar
            PROJECT TO polar
        "#;
        let spec3 = reader.execute(query3).unwrap();
        let result3 = writer.render(&spec3).unwrap();
        let json3: serde_json::Value = serde_json::from_str(&result3).unwrap();
        check_encoding_keys(&json3, "PROJECT TO polar");

        // Test case 4: PROJECT a, b TO polar (custom aesthetic names)
        let query4 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS a, category AS fill
            DRAW bar
            PROJECT a, b TO polar
        "#;
        let spec4 = reader.execute(query4).unwrap();
        let result4 = writer.render(&spec4).unwrap();
        let json4: serde_json::Value = serde_json::from_str(&result4).unwrap();
        check_encoding_keys(&json4, "PROJECT a, b TO polar (custom names)");
    }

    #[test]
    fn test_cartesian_encoding_keys_with_custom_names() {
        // This test verifies that cartesian projections produce x/y encoding keys
        // even when custom position names are used in PROJECT.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        fn check_cartesian_keys(json: &serde_json::Value, test_name: &str) {
            let layer = data_layer(json, 0);
            assert!(
                layer["encoding"].get("x").is_some(),
                "{} should produce x encoding, got keys: {:?}",
                test_name,
                layer["encoding"]
                    .as_object()
                    .map(|o| o.keys().collect::<Vec<_>>())
            );
            // Verify no theta/radius keys exist
            assert!(
                layer["encoding"].get("theta").is_none(),
                "{} should NOT have theta encoding in cartesian mode",
                test_name
            );
        }

        // Test case: PROJECT a, b TO cartesian (custom aesthetic names)
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE category AS a, value AS b
            DRAW bar
            PROJECT a, b TO cartesian
        "#;
        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        check_cartesian_keys(&json, "PROJECT a, b TO cartesian (custom names)");
    }

    #[test]
    fn test_register_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df = df! {
            "x" => vec![1i32, 2, 3],
            "y" => vec![10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_data", df, false).unwrap();

        let query = "SELECT * FROM my_data VISUALISE x, y DRAW point";
        let spec = reader.execute(query).unwrap();

        assert_eq!(spec.metadata().rows, 3);
        // Aesthetics are transformed to internal names (x -> pos1)
        assert!(spec.metadata().columns.contains(&"pos1".to_string()));

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    #[test]
    fn test_register_and_join() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sales = df! {
            "id" => vec![1i32, 2, 3],
            "amount" => vec![100i32, 200, 300],
            "product_id" => vec![1i32, 1, 2],
        }
        .unwrap();

        let products = df! {
            "id" => vec![1i32, 2],
            "name" => vec!["Widget", "Gadget"],
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
        // Verifies that binned material aesthetics use threshold scale type
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
            DRAW point MAPPING x AS x, y AS y, value AS fill
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

    #[test]
    fn test_polar_project_with_inner() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING inner => 0.5
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Check radius scale has range with expressions
        let radius = &layer["encoding"]["radius"];
        assert!(radius["scale"]["range"].is_array());
        let range = radius["scale"]["range"].as_array().unwrap();

        // First element should be inner proportion expression
        assert!(
            range[0]["expr"].as_str().unwrap().contains("0.5"),
            "Inner radius expression should contain 0.5, got: {:?}",
            range[0]
        );

        // Second element should be the outer radius expression
        assert!(
            range[1]["expr"]
                .as_str()
                .unwrap()
                .contains("min(width, height) / 2"),
            "Outer radius expression should contain min(width, height) / 2, got: {:?}",
            range[1]
        );
    }

    #[test]
    fn test_stacked_bar_chart() {
        // Test stacked bar chart via position => 'stack'
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 'X', 10),
                ('A', 'Y', 20),
                ('B', 'X', 15),
                ('B', 'Y', 25)
            ) AS t(cat, grp, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y, grp AS fill
            SETTING position => 'stack'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
        let encoding = &layer["encoding"];
        assert!(encoding["y"].is_object(), "Should have y encoding");
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars"
        );

        // Verify Vega-Lite stacking is disabled (we handle it ourselves)
        assert!(
            encoding["y"]["stack"].is_null(),
            "y encoding should have stack: null to disable VL stacking. Got: {}",
            serde_json::to_string_pretty(&encoding["y"]).unwrap()
        );
    }

    #[test]
    fn test_stacked_bar_chart_dummy_x() {
        // Test stacked bar chart with no x mapping (dummy x column)
        // This is the case where only fill is mapped: all bars at same x position should stack
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW bar MAPPING species AS fill
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
        let encoding = &layer["encoding"];
        assert!(encoding["y"].is_object(), "Should have y encoding");
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars with dummy x. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );

        // Verify Vega-Lite stacking is disabled (we handle it ourselves)
        assert!(
            encoding["y"]["stack"].is_null(),
            "y encoding should have stack: null to disable VL stacking. Got: {}",
            serde_json::to_string_pretty(&encoding["y"]).unwrap()
        );
    }

    #[test]
    fn test_bar_chart_with_expand_setting() {
        // Test bar chart with SCALE y SETTING expand - should work even when y is stat-derived
        // This tests that:
        // 1. Scale type inference works for stat-generated count columns
        // 2. Stacking still works (y2 encoding exists) when SCALE y is specified
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW bar MAPPING species AS fill
            SCALE y SETTING expand => [0.05, 0.05]
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Should succeed without "discrete scale does not support SETTING 'expand'" error
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify stacking works (y2 encoding exists for stacked bars)
        let encoding = &layer["encoding"];
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[test]
    fn test_dodged_bar_chart() {
        // Test dodged bar chart via position => 'dodge'
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 'X', 10),
                ('A', 'Y', 20),
                ('B', 'X', 15),
                ('B', 'Y', 25)
            ) AS t(cat, grp, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y, grp AS fill
            SETTING position => 'dodge'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify xOffset encoding exists (dodged bars use xOffset for displacement)
        let encoding = &layer["encoding"];
        assert!(
            encoding["xOffset"].is_object(),
            "Should have xOffset encoding for dodged bars. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );

        // Verify bar width uses bandwidth expression with adjusted_width for dodged bars
        // For 2 groups with default width 0.9: adjusted_width = 0.9 / 2 = 0.45
        let mark = &layer["mark"];
        let width_expr = mark["width"]["expr"].as_str();
        assert!(
            width_expr.is_some(),
            "Dodged bars should have expression-based width. Mark: {}",
            serde_json::to_string_pretty(mark).unwrap()
        );
        let expr = width_expr.unwrap();
        assert!(
            expr.contains("bandwidth('x')") && expr.contains("0.45"),
            "Width expression should use bandwidth('x') * adjusted_width, got: {}",
            expr
        );
    }

    #[test]
    fn test_position_identity_default() {
        // Test that identity position (default) doesn't modify data
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 10),
                ('B', 20)
            ) AS t(cat, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify no xOffset encoding (identity position)
        let encoding = &layer["encoding"];
        assert!(
            encoding.get("xOffset").is_none(),
            "Identity position should not have xOffset encoding"
        );
    }

    #[test]
    fn test_label_with_flipped_project() {
        // End-to-end test: LABEL x/y with PROJECT y, x TO cartesian
        // Labels should be correctly applied to the flipped axes
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
            VISUALISE
            DRAW bar MAPPING x AS y, y AS x
            PROJECT y, x TO cartesian
            LABEL x => 'Value', y => 'Category'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];

        // With PROJECT y, x TO cartesian:
        // - y is pos1 (first position), renders to VL x-axis in cartesian
        // - x is pos2 (second position), renders to VL y-axis in cartesian
        // So LABEL y => 'Category' should appear on VL x-axis, LABEL x => 'Value' on VL y-axis
        let x_title = encoding["x"]["title"].as_str();
        let y_title = encoding["y"]["title"].as_str();

        assert_eq!(
            x_title,
            Some("Category"),
            "x-axis should have 'Category' title (from LABEL y). Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
        assert_eq!(
            y_title,
            Some("Value"),
            "y-axis should have 'Value' title (from LABEL x). Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[test]
    fn test_label_with_polar_project() {
        // End-to-end test: LABEL angle/radius with PROJECT TO polar
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS angle, category AS fill
            DRAW bar
            PROJECT TO polar
            LABEL angle => 'Angle', radius => 'Distance'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];

        // Verify theta encoding has the label
        let theta_title = encoding["theta"]["title"].as_str();
        assert_eq!(
            theta_title,
            Some("Angle"),
            "theta encoding should have 'Angle' title. Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }
}
