//! Query execution module for ggsql Jupyter kernel
//!
//! This module handles the execution of ggsql queries using the existing
//! ggsql library components (parser, DuckDB reader, Vega-Lite writer).
//! It supports dynamic reader switching via `-- @connect:` meta-commands.

use anyhow::Result;
use ggsql::{
    reader::{connection::parse_connection_string, DuckDBReader, Reader},
    validate::validate,
    writer::{VegaLiteWriter, Writer},
};
use polars::frame::DataFrame;

/// Result of executing a ggsql query
#[derive(Debug)]
pub enum ExecutionResult {
    /// Pure SQL query with no visualization
    DataFrame(DataFrame),
    /// Query with visualization specification
    Visualization {
        spec: String, // Vega-Lite JSON
    },
    /// Connection changed via meta-command
    ConnectionChanged {
        uri: String,
        display_name: String,
    },
}

/// Create a reader from a connection URI string.
///
/// Supported schemes:
/// - `duckdb://memory` or `duckdb://<path>` (always available)
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
            let reader =
                DuckDBReader::from_connection_string(&format!("duckdb://{}", path))?;
            Ok(Box::new(reader))
        }
        #[cfg(feature = "odbc")]
        ConnectionInfo::ODBC(conn_str) => {
            let reader = ggsql::reader::OdbcReader::from_connection_string(
                &format!("odbc://{}", conn_str),
            )?;
            Ok(Box::new(reader))
        }
        _ => anyhow::bail!(
            "Unsupported reader type for connection string: {}. \
             Only DuckDB connections are currently supported in ggsql-jupyter.",
            uri
        ),
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
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        // Try to extract driver name from ODBC string
        if let Some(driver_start) = odbc.to_lowercase().find("driver=") {
            let rest = &odbc[driver_start + 7..];
            let driver = rest
                .split(';')
                .next()
                .unwrap_or("ODBC")
                .trim_matches(|c| c == '{' || c == '}');
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
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        if odbc.to_lowercase().contains("driver=snowflake") {
            return "Snowflake".to_string();
        }
        if odbc.to_lowercase().contains("driver={postgresql}") || odbc.to_lowercase().contains("driver=postgresql") {
            return "PostgreSQL".to_string();
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
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        // Try to extract server
        if let Some(server_start) = odbc.to_lowercase().find("server=") {
            let rest = &odbc[server_start + 7..];
            if let Some(host) = rest.split(';').next() {
                return host.to_string();
            }
        }
    }
    uri.to_string()
}

/// The `-- @connect:` meta-command prefix.
const META_CONNECT_PREFIX: &str = "-- @connect:";

/// Parse a `-- @connect: <uri>` meta-command, returning the URI if present.
pub fn parse_meta_command(code: &str) -> Option<String> {
    let trimmed = code.trim();
    if let Some(rest) = trimmed.strip_prefix(META_CONNECT_PREFIX) {
        Some(rest.trim().to_string())
    } else {
        None
    }
}

/// Query executor maintaining persistent database connection
pub struct QueryExecutor {
    reader: Box<dyn Reader + Send>,
    writer: VegaLiteWriter,
    reader_uri: String,
}

impl QueryExecutor {
    /// Create a new query executor with a given connection URI
    pub fn new_with_uri(uri: &str) -> Result<Self> {
        tracing::info!("Initializing query executor with reader: {}", uri);
        let reader = create_reader(uri)?;
        let writer = VegaLiteWriter::new();

        Ok(Self {
            reader,
            writer,
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
            return Ok(ExecutionResult::ConnectionChanged {
                uri,
                display_name,
            });
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

        // 4. Render to output format
        let vega_json = self.writer.render(&spec)?;

        tracing::debug!("Generated Vega-Lite spec: {} chars", vega_json.len());

        // 5. Return result
        Ok(ExecutionResult::Visualization { spec: vega_json })
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

        assert!(matches!(result, ExecutionResult::Visualization { .. }));
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
    }
}
