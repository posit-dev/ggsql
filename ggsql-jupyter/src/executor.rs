//! Query execution module for ggsql Jupyter kernel
//!
//! This module handles the execution of ggsql queries using the existing
//! ggsql library components (parser and reader). Formatting the result — and
//! rendering a plot — is `display.rs`'s, since the format depends on where the
//! output is going. Supports dynamic reader switching via `-- @connect:`
//! meta-commands.

use anyhow::Result;
use ggsql::{
    reader::{
        connection::{extract_odbc_value, parse_connection_string},
        DuckDBReader, Reader, Spec,
    },
    validate::validate,
    DataFrame,
};

/// A resolved plot has to reach a render thread, so the design rests on this.
const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<Spec>;
};

/// Result of executing a ggsql query
pub enum ExecutionResult {
    /// Pure SQL query with no visualization
    DataFrame(DataFrame),
    /// A query carrying a `VISUALISE` clause, as the resolved plot rather than
    /// as rendered output.
    ///
    /// Not pre-rendered: the format depends on where the output is going, and
    /// once a plot comm is open it is asked again on every resize. Boxed because
    /// a `Spec` carries the post-stat DataFrames and dwarfs the other variants.
    Visualization(Box<Spec>),
    /// Connection changed via meta-command
    ConnectionChanged { uri: String, display_name: String },
}

// `Spec` is neither `Debug` nor `Clone`, so this summarises rather than
// deriving. What a log wants from a result is its shape and size anyway.
impl std::fmt::Debug for ExecutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataFrame(df) => f
                .debug_struct("DataFrame")
                .field("rows", &df.height())
                .field("columns", &df.width())
                .finish(),
            Self::Visualization(spec) => {
                let metadata = spec.metadata();
                f.debug_struct("Visualization")
                    .field("rows", &metadata.rows)
                    .field("layers", &metadata.layer_count)
                    .finish()
            }
            Self::ConnectionChanged { uri, display_name } => f
                .debug_struct("ConnectionChanged")
                .field("uri", uri)
                .field("display_name", display_name)
                .finish(),
        }
    }
}

/// Create a reader from a connection URI string.
///
/// Supported schemes:
/// - `duckdb://memory` or `duckdb://<path>` (always available)
/// - `sqlite://<path>` (requires `sqlite` feature)
/// - `odbc://...` (requires `odbc` feature)
pub fn create_reader(uri: &str) -> Result<Box<dyn Reader + Send>> {
    use ggsql::reader::connection::ConnectionInfo;

    let info = parse_connection_string(uri)?;
    match info {
        ConnectionInfo::DuckDBMemory => {
            let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
            Ok(Box::new(reader))
        }
        ConnectionInfo::DuckDBFile(path) => {
            let reader = DuckDBReader::from_connection_string(&format!("duckdb://{}", path))?;
            Ok(Box::new(reader))
        }
        #[cfg(feature = "odbc")]
        ConnectionInfo::ODBC(conn_str) => {
            let reader =
                ggsql::reader::OdbcReader::from_connection_string(&format!("odbc://{}", conn_str))?;
            Ok(Box::new(reader))
        }
        #[cfg(feature = "sqlite")]
        ConnectionInfo::SQLite(path) => {
            let reader =
                ggsql::reader::SqliteReader::from_connection_string(&format!("sqlite://{}", path))?;
            Ok(Box::new(reader))
        }
        _ => anyhow::bail!("Unsupported reader type for connection string: {}", uri),
    }
}

/// Generate a human-readable display name for a connection URI.
pub fn display_name_for_uri(uri: &str) -> String {
    if uri == "duckdb://memory" {
        return "DuckDB (memory)".to_string();
    }
    if let Some(path) = uri.strip_prefix("duckdb://") {
        return format!("DuckDB ({})", path);
    }
    if let Some(path) = uri.strip_prefix("sqlite://") {
        if path == ":memory:" || path.is_empty() {
            return "SQLite (memory)".to_string();
        }
        return format!("SQLite ({})", path);
    }
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        if let Some(dsn) = extract_odbc_value(odbc, "dsn") {
            return format!("{} (ODBC)", dsn);
        }
        if let Some(driver) = extract_odbc_value(odbc, "driver") {
            return format!("{} (ODBC)", driver);
        }
        return "ODBC".to_string();
    }
    uri.to_string()
}

/// Detect the database type name from a connection URI (e.g. "DuckDB", "Snowflake").
pub fn type_name_for_uri(uri: &str) -> String {
    if uri.starts_with("duckdb://") {
        return "DuckDB".to_string();
    }
    if uri.starts_with("sqlite://") {
        return "SQLite".to_string();
    }
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        if let Some(driver) = extract_odbc_value(odbc, "driver") {
            let lower = driver.to_lowercase();
            if lower.contains("snowflake") {
                return "Snowflake".to_string();
            }
            if lower.contains("postgresql") {
                return "PostgreSQL".to_string();
            }
        }
        return "ODBC".to_string();
    }
    "Unknown".to_string()
}

/// Extract the host portion from a connection URI.
pub fn host_for_uri(uri: &str) -> String {
    if uri == "duckdb://memory" {
        return "memory".to_string();
    }
    if let Some(path) = uri.strip_prefix("duckdb://") {
        return path.to_string();
    }
    if let Some(path) = uri.strip_prefix("sqlite://") {
        if path.is_empty() {
            return "memory".to_string();
        }
        return path.to_string();
    }
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        if let Some(server) = extract_odbc_value(odbc, "server") {
            return server;
        }
    }
    uri.to_string()
}

/// The `-- @connect:` meta-command prefix.
const META_CONNECT_PREFIX: &str = "-- @connect:";

/// Parse a `-- @connect: <uri>` meta-command, returning the URI if present.
pub fn parse_meta_command(code: &str) -> Option<String> {
    let trimmed = code.trim();
    trimmed
        .strip_prefix(META_CONNECT_PREFIX)
        .map(|rest| rest.trim().to_string())
}

/// Query executor maintaining persistent database connection
pub struct QueryExecutor {
    reader: Box<dyn Reader + Send>,
    reader_uri: String,
}

impl QueryExecutor {
    /// Create a new query executor with a given connection URI
    pub fn new_with_uri(uri: &str) -> Result<Self> {
        tracing::info!("Initializing query executor with reader: {}", uri);
        let reader = create_reader(uri)?;

        Ok(Self {
            reader,
            reader_uri: uri.to_string(),
        })
    }

    /// Create a new query executor with the default in-memory DuckDB database
    #[cfg(test)]
    pub fn new() -> Result<Self> {
        Self::new_with_uri("duckdb://memory")
    }

    /// Get the current reader URI
    pub fn reader_uri(&self) -> &str {
        &self.reader_uri
    }

    /// Get a reference to the current reader (for schema introspection)
    pub fn reader(&self) -> &dyn Reader {
        &*self.reader
    }

    /// Swap the reader to a new connection, returning the old URI
    pub fn swap_reader(&mut self, uri: &str) -> Result<String> {
        let new_reader = create_reader(uri)?;
        self.reader = new_reader;
        let old_uri = std::mem::replace(&mut self.reader_uri, uri.to_string());
        Ok(old_uri)
    }

    /// Execute a ggsql query or meta-command
    ///
    /// This handles:
    /// - `-- @connect: <uri>` meta-commands for switching readers
    /// - Pure SQL queries (no VISUALISE)
    /// - ggsql queries with VISUALISE clauses
    pub fn execute(&mut self, code: &str) -> Result<ExecutionResult> {
        tracing::debug!("Executing query: {} chars", code.len());

        // Check for meta-commands first
        if let Some(uri) = parse_meta_command(code) {
            tracing::info!("Meta-command: switching reader to {}", uri);
            self.swap_reader(&uri)?;
            let display_name = display_name_for_uri(&uri);
            return Ok(ExecutionResult::ConnectionChanged { uri, display_name });
        }

        // 1. Validate to check if there's a visualization
        let validated = validate(code)?;

        // 2. Check if there's a visualization
        if !validated.has_visual() {
            // Pure SQL query - execute directly and return DataFrame
            let df = self.reader.execute_sql(code)?;
            tracing::info!(
                "Pure SQL executed: {} rows, {} cols",
                df.height(),
                df.width()
            );
            return Ok(ExecutionResult::DataFrame(df));
        }

        // 3. Execute ggsql query using reader
        let spec = self.reader.execute(code)?;

        tracing::info!(
            "Query executed: {} rows, {} layers",
            spec.metadata().rows,
            spec.metadata().layer_count
        );

        // 4. Hand back the resolved plot. Choosing a format is the display
        //    layer's job, because only it knows where the output is going.
        Ok(ExecutionResult::Visualization(Box::new(spec)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_visualization() {
        let mut executor = QueryExecutor::new().unwrap();
        let code = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";
        let result = executor.execute(code).unwrap();

        assert!(matches!(result, ExecutionResult::Visualization(_)));
    }

    #[test]
    fn test_pure_sql() {
        let mut executor = QueryExecutor::new().unwrap();
        let code = "SELECT 1 as x, 2 as y";
        let result = executor.execute(code).unwrap();

        assert!(matches!(result, ExecutionResult::DataFrame(_)));
    }

    #[test]
    fn test_error_handling() {
        let mut executor = QueryExecutor::new().unwrap();
        let code = "SELECT * FROM nonexistent_table";
        let result = executor.execute(code);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_meta_command() {
        assert_eq!(
            parse_meta_command("-- @connect: duckdb://memory"),
            Some("duckdb://memory".to_string())
        );
        assert_eq!(
            parse_meta_command("  -- @connect:  duckdb://my.db  "),
            Some("duckdb://my.db".to_string())
        );
        assert_eq!(parse_meta_command("SELECT 1"), None);
    }

    #[test]
    fn test_meta_command_switches_reader() {
        let mut executor = QueryExecutor::new().unwrap();
        assert_eq!(executor.reader_uri(), "duckdb://memory");

        let result = executor.execute("-- @connect: duckdb://memory").unwrap();
        assert!(matches!(result, ExecutionResult::ConnectionChanged { .. }));
    }

    #[test]
    fn test_display_name_for_uri() {
        assert_eq!(display_name_for_uri("duckdb://memory"), "DuckDB (memory)");
        assert_eq!(display_name_for_uri("duckdb://my.db"), "DuckDB (my.db)");
        assert_eq!(display_name_for_uri("sqlite://:memory:"), "SQLite (memory)");
        assert_eq!(display_name_for_uri("sqlite://data.db"), "SQLite (data.db)");
        assert_eq!(
            display_name_for_uri("odbc://DSN=my-postgres"),
            "my-postgres (ODBC)"
        );
        assert_eq!(
            display_name_for_uri("odbc://Driver=Snowflake;Server=foo"),
            "Snowflake (ODBC)"
        );
        assert_eq!(
            display_name_for_uri("odbc://Driver={PostgreSQL};DSN=pg-test"),
            "pg-test (ODBC)"
        );
        assert_eq!(display_name_for_uri("odbc://"), "ODBC");
    }
}
