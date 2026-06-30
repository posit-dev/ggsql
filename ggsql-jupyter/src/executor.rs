//! Query execution module for ggsql Jupyter kernel
//!
//! This module handles the execution of ggsql queries using the existing
//! ggsql library components (parser, DuckDB reader, Vega-Lite writer).
//! It supports leading `--` meta-command lines. Each occupies its own comment
//! line, so a cell may stack them above a query that then runs as normal.

use anyhow::Result;
use ggsql::{
    reader::{
        connection::{extract_odbc_value, reader_from_uri},
        Reader,
    },
    validate::validate,
    writer::{VegaLiteWriter, Writer},
    DataFrame,
};

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
    ConnectionChanged { display_name: String },
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
/// The `-- @uncache` meta-command prefix.
const META_UNCACHE_PREFIX: &str = "-- @uncache";

/// A leading cell directive expressed as a `--` line comment.
#[derive(Debug, PartialEq, Eq)]
pub enum MetaCommand {
    /// Switch the active reader to the given connection URI.
    Connect(String),
    /// Clear the active reader's cache.
    Uncache,
}

/// Split `code` into its first line and the remainder.
/// Handles `\n`, `\r\n`, and a lone `\r`.
fn split_first_line(code: &str) -> (&str, &str) {
    match code.find(['\n', '\r']) {
        None => (code, ""),
        Some(i) => {
            let line = &code[..i];
            let rest = &code[i..];
            let rest = rest
                .strip_prefix("\r\n")
                .or_else(|| rest.strip_prefix('\n'))
                .or_else(|| rest.strip_prefix('\r'))
                .unwrap_or(rest);
            (line, rest)
        }
    }
}

/// Peel a single leading meta-command from `code`, returning it together with
/// the rest of the cell to process next.
pub fn take_leading_meta(code: &str) -> Option<(MetaCommand, &str)> {
    let trimmed = code.trim_start();
    let (line, rest) = split_first_line(trimmed);
    let line = line.trim();
    if let Some(uri) = line.strip_prefix(META_CONNECT_PREFIX) {
        return Some((MetaCommand::Connect(uri.trim().to_string()), rest));
    }
    if line == META_UNCACHE_PREFIX {
        return Some((MetaCommand::Uncache, rest));
    }
    None
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
        let reader = reader_from_uri(uri)?;
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
        let new_reader = reader_from_uri(uri)?;
        self.reader = new_reader;
        let old_uri = std::mem::replace(&mut self.reader_uri, uri.to_string());
        Ok(old_uri)
    }

    /// Execute a ggsql query or meta-command
    ///
    /// This handles:
    /// - `-- @` meta-commands
    /// - Pure SQL queries (no VISUALISE)
    /// - ggsql queries with VISUALISE clauses
    pub fn execute(&mut self, code: &str) -> Result<ExecutionResult> {
        tracing::debug!("Executing query: {} chars", code.len());

        // Apply any leading meta-command lines, then run whatever SQL remains.
        let mut code = code;
        let mut last_connect: Option<String> = None;
        while let Some((cmd, rest)) = take_leading_meta(code) {
            match cmd {
                MetaCommand::Connect(uri) => {
                    tracing::info!("Meta-command: switching reader to {}", uri);
                    self.swap_reader(&uri)?;
                    last_connect = Some(uri);
                }
                MetaCommand::Uncache => {
                    tracing::info!("Meta-command: clearing cache");
                    self.reader.clear_cache()?;
                }
            }
            code = rest;
        }

        // A cell that was nothing but meta-commands.
        if code.trim().is_empty() {
            if let Some(uri) = last_connect {
                let display_name = display_name_for_uri(&uri);
                return Ok(ExecutionResult::ConnectionChanged { display_name });
            }
            // An empty DataFrame renders no cell output.
            return Ok(ExecutionResult::DataFrame(DataFrame::empty()));
        }

        // 1. Validate to check if there's a visualization
        let validated = validate(code)?;

        // 2. Check if there's a visualization
        if !validated.has_visual() {
            // Pure SQL query - execute directly and return DataFrame.
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
    fn test_take_leading_meta_connect() {
        // `-- @connect:` takes the rest of its line as the URI; the next line is
        // the remainder.
        assert_eq!(
            take_leading_meta("-- @connect: duckdb://memory"),
            Some((MetaCommand::Connect("duckdb://memory".to_string()), ""))
        );
        assert_eq!(
            take_leading_meta("  -- @connect:  duckdb://my.db  \nSELECT 1"),
            Some((
                MetaCommand::Connect("duckdb://my.db".to_string()),
                "SELECT 1"
            ))
        );
    }

    #[test]
    fn test_take_leading_meta_uncache() {
        assert_eq!(
            take_leading_meta("-- @uncache"),
            Some((MetaCommand::Uncache, ""))
        );
        assert_eq!(
            take_leading_meta("-- @uncache\nSELECT 1"),
            Some((MetaCommand::Uncache, "SELECT 1"))
        );
        assert_eq!(
            take_leading_meta("-- @uncache  \r\nSELECT 1"),
            Some((MetaCommand::Uncache, "SELECT 1"))
        );
        // `-- @uncache foo` on one line is an ordinary SQL comment, not the directive.
        assert_eq!(take_leading_meta("-- @uncache foo"), None);
    }

    #[test]
    fn test_take_leading_meta_non_directive() {
        assert_eq!(take_leading_meta("SELECT 1"), None);
        assert_eq!(take_leading_meta("-- a normal comment\nSELECT 1"), None);
    }

    #[test]
    fn test_meta_command_switches_reader() {
        let mut executor = QueryExecutor::new().unwrap();
        assert_eq!(executor.reader_uri(), "duckdb://memory");

        let result = executor.execute("-- @connect: duckdb://memory").unwrap();
        assert!(matches!(result, ExecutionResult::ConnectionChanged { .. }));
    }

    #[test]
    fn test_connect_then_runs_remaining_query() {
        // A leading `-- @connect:` switches the reader and still runs the query
        // below it in the same cell.
        let mut executor = QueryExecutor::new().unwrap();
        let result = executor
            .execute(
                "-- @connect: duckdb://memory\nSELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point",
            )
            .unwrap();
        assert_eq!(executor.reader_uri(), "duckdb://memory");
        assert!(matches!(result, ExecutionResult::Visualization { .. }));
    }

    #[test]
    fn test_uncache_meta_command_clears_cache() {
        // On the default reader (no cache) `clear_cache` is a no-op; this proves
        // the dispatch arm is wired and yields an empty DataFrame.
        let mut executor = QueryExecutor::new().unwrap();
        let result = executor.execute("-- @uncache").unwrap();
        match result {
            ExecutionResult::DataFrame(df) => assert_eq!(df.width(), 0),
            other => panic!("expected empty DataFrame, got {other:?}"),
        }
    }

    #[test]
    fn test_uncache_then_runs_remaining_query() {
        // A leading `-- @uncache` clears the cache and still runs the query below.
        let mut executor = QueryExecutor::new().unwrap();
        let result = executor
            .execute("-- @uncache\nSELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap();
        assert!(matches!(result, ExecutionResult::Visualization { .. }));
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
