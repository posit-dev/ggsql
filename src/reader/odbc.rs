//! Generic ODBC data source implementation
//!
//! Provides a reader for any ODBC-compatible database (Snowflake, PostgreSQL,
//! SQL Server, etc.) using the `odbc-api` crate.

use crate::reader::Reader;
use crate::{DataFrame, GgsqlError, Result};
use odbc_api::{buffers::TextRowSet, ConnectionOptions, Cursor, Environment};
use polars::prelude::*;
use std::sync::OnceLock;

/// Global ODBC environment (must be a singleton per process).
fn odbc_env() -> &'static Environment {
    static ENV: OnceLock<Environment> = OnceLock::new();
    ENV.get_or_init(|| {
        Environment::new().expect("Failed to create ODBC environment")
    })
}

/// ODBC SQL dialect.
///
/// Uses ANSI SQL by default. The `variant` field can be used to detect
/// specific backends for dialect customization.
pub struct OdbcDialect {
    #[allow(dead_code)]
    variant: OdbcVariant,
}

/// Detected ODBC backend variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdbcVariant {
    Generic,
    Snowflake,
    PostgreSQL,
    SqlServer,
}

impl super::SqlDialect for OdbcDialect {}

/// Generic ODBC reader implementing the `Reader` trait.
pub struct OdbcReader {
    connection: odbc_api::Connection<'static>,
    dialect: OdbcDialect,
}

// Safety: odbc_api::Connection is Send when we ensure single-threaded access.
// The Reader trait requires &self (immutable) for execute_sql, and ODBC
// connections are safe to use from one thread at a time.
unsafe impl Send for OdbcReader {}

impl OdbcReader {
    /// Create a new ODBC reader from a `odbc://` connection URI.
    ///
    /// The URI format is `odbc://` followed by the raw ODBC connection string.
    /// For Snowflake with Posit Workbench credentials, the reader will
    /// automatically detect and inject OAuth tokens.
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_str = uri
            .strip_prefix("odbc://")
            .ok_or_else(|| GgsqlError::ReaderError("ODBC URI must start with odbc://".into()))?;

        let mut conn_str = conn_str.to_string();

        // Snowflake Workbench credential detection
        if is_snowflake(&conn_str) && !has_token(&conn_str) {
            if let Some(token) = detect_workbench_token() {
                conn_str = inject_snowflake_token(&conn_str, &token);
            }
        }

        // Detect variant from connection string
        let variant = detect_variant(&conn_str);

        let env = odbc_env();
        let connection = env
            .connect_with_connection_string(&conn_str, ConnectionOptions::default())
            .map_err(|e| GgsqlError::ReaderError(format!("ODBC connection failed: {}", e)))?;

        Ok(Self {
            connection,
            dialect: OdbcDialect { variant },
        })
    }
}

impl Reader for OdbcReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Execute the query (3rd arg = query timeout, None = no timeout)
        let cursor = self
            .connection
            .execute(sql, (), None)
            .map_err(|e| GgsqlError::ReaderError(format!("ODBC execute failed: {}", e)))?;

        let Some(cursor) = cursor else {
            // DDL or non-query statement — return empty DataFrame
            return DataFrame::new(Vec::<Column>::new())
                .map_err(|e| GgsqlError::ReaderError(format!("Empty DataFrame error: {}", e)));
        };

        cursor_to_dataframe(cursor)
    }

    fn register(&self, name: &str, _df: DataFrame, _replace: bool) -> Result<()> {
        Err(GgsqlError::ReaderError(format!(
            "ODBC reader does not support registering in-memory tables (attempted: '{}')",
            name
        )))
    }

    fn execute(&self, query: &str) -> Result<super::Spec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &self.dialect
    }
}

/// Convert an ODBC cursor to a Polars DataFrame by fetching all rows as text.
fn cursor_to_dataframe(mut cursor: impl Cursor) -> Result<DataFrame> {
    let col_count = cursor.num_result_cols()
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to get column count: {}", e)))?
        as usize;

    if col_count == 0 {
        return DataFrame::new(Vec::<Column>::new())
            .map_err(|e| GgsqlError::ReaderError(e.to_string()));
    }

    // Collect column names
    let mut col_names = Vec::with_capacity(col_count);
    for i in 1..=col_count as u16 {
        let name = cursor
            .col_name(i)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to get column {} name: {}", i, e)))?;
        col_names.push(name);
    }

    // Fetch all rows as text into column-oriented vectors
    let batch_size = 1000;
    let max_str_len = 4096;
    let mut columns: Vec<Vec<Option<String>>> = vec![Vec::new(); col_count];

    let mut row_set = TextRowSet::for_cursor(batch_size, &mut cursor, Some(max_str_len))
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to create row set: {}", e)))?;

    let mut block_cursor = cursor.bind_buffer(&mut row_set)
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to bind buffer: {}", e)))?;

    while let Some(batch) = block_cursor.fetch()
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to fetch batch: {}", e)))?
    {
        let num_rows = batch.num_rows();
        for col_idx in 0..col_count {
            for row_idx in 0..num_rows {
                let value = batch
                    .at_as_str(col_idx, row_idx)
                    .ok()
                    .flatten()
                    .map(|s| s.to_string());
                columns[col_idx].push(value);
            }
        }
    }

    // Build Polars Series from the text data, attempting type inference
    let series: Vec<Column> = col_names
        .iter()
        .zip(columns.iter())
        .map(|(name, values)| {
            // Try to parse as numeric first, then fall back to string
            let series = if let Some(int_series) = try_parse_integers(name, values) {
                int_series
            } else if let Some(float_series) = try_parse_floats(name, values) {
                float_series
            } else {
                // Fall back to string
                Series::new(
                    name.into(),
                    values
                        .iter()
                        .map(|v| v.as_deref())
                        .collect::<Vec<Option<&str>>>(),
                )
            };
            Column::from(series)
        })
        .collect();

    DataFrame::new(series).map_err(|e| GgsqlError::ReaderError(e.to_string()))
}

/// Try to parse all non-null values as i64.
fn try_parse_integers(name: &str, values: &[Option<String>]) -> Option<Series> {
    let parsed: Vec<Option<i64>> = values
        .iter()
        .map(|v| match v {
            None => Some(None),
            Some(s) => s.parse::<i64>().ok().map(Some),
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Series::new(name.into(), parsed))
}

/// Try to parse all non-null values as f64.
fn try_parse_floats(name: &str, values: &[Option<String>]) -> Option<Series> {
    let parsed: Vec<Option<f64>> = values
        .iter()
        .map(|v| match v {
            None => Some(None),
            Some(s) => s.parse::<f64>().ok().map(Some),
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Series::new(name.into(), parsed))
}

// ============================================================================
// Snowflake Workbench credential detection
// ============================================================================

fn is_snowflake(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("driver=snowflake")
}

fn has_token(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("token=")
}

fn detect_variant(conn_str: &str) -> OdbcVariant {
    let lower = conn_str.to_lowercase();
    if lower.contains("driver=snowflake") {
        OdbcVariant::Snowflake
    } else if lower.contains("driver={postgresql}") || lower.contains("driver=postgresql") {
        OdbcVariant::PostgreSQL
    } else if lower.contains("driver={odbc driver") || lower.contains("driver={sql server") {
        OdbcVariant::SqlServer
    } else {
        OdbcVariant::Generic
    }
}

/// Detect Posit Workbench Snowflake OAuth token.
///
/// Checks `SNOWFLAKE_HOME` for a Workbench-managed `connections.toml` file
/// containing OAuth credentials.
fn detect_workbench_token() -> Option<String> {
    let snowflake_home = std::env::var("SNOWFLAKE_HOME").ok()?;

    // Only use Workbench credentials if the path indicates Workbench management
    if !snowflake_home.contains("posit-workbench") {
        return None;
    }

    let toml_path = std::path::Path::new(&snowflake_home).join("connections.toml");
    let content = std::fs::read_to_string(&toml_path).ok()?;

    let doc = content.parse::<toml_edit::DocumentMut>().ok()?;
    let token = doc
        .get("workbench")?
        .get("token")?
        .as_str()?
        .to_string();

    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// Inject OAuth token into a Snowflake ODBC connection string.
fn inject_snowflake_token(conn_str: &str, token: &str) -> String {
    // Append authenticator and token parameters
    let mut result = conn_str.trim_end_matches(';').to_string();
    result.push_str(";Authenticator=oauth;Token=");
    result.push_str(token);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_snowflake() {
        assert!(is_snowflake("Driver=Snowflake;Server=foo.snowflakecomputing.com"));
        assert!(!is_snowflake("Driver={PostgreSQL};Server=localhost"));
    }

    #[test]
    fn test_has_token() {
        assert!(has_token("Driver=Snowflake;Token=abc123"));
        assert!(!has_token("Driver=Snowflake;Server=foo"));
    }

    #[test]
    fn test_detect_variant() {
        assert_eq!(
            detect_variant("Driver=Snowflake;Server=foo"),
            OdbcVariant::Snowflake
        );
        assert_eq!(
            detect_variant("Driver={PostgreSQL};Server=localhost"),
            OdbcVariant::PostgreSQL
        );
        assert_eq!(
            detect_variant("Driver=SomeOther;Server=localhost"),
            OdbcVariant::Generic
        );
    }

    #[test]
    fn test_inject_snowflake_token() {
        let result =
            inject_snowflake_token("Driver=Snowflake;Server=foo.snowflakecomputing.com", "mytoken");
        assert!(result.contains("Authenticator=oauth"));
        assert!(result.contains("Token=mytoken"));
    }
}
