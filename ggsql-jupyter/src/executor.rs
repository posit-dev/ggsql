//! Query execution module for ggsql Jupyter kernel
//!
//! This module handles the execution of ggsql queries using the existing
//! ggsql library components (parser and reader). Formatting the result — and
//! rendering a plot — is `display.rs`'s, since the format depends on where the
//! output is going.
//!
//! Supports leading `--` meta-command lines. Each occupies its own comment
//! line, so a cell may stack them above a query that then runs as normal.

use anyhow::Result;
use ggsql::{
    reader::{
        connection::{extract_odbc_value, reader_from_uri},
        Reader, Spec,
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
    ConnectionChanged { display_name: String },
    /// Open a retained query result in the data explorer, by key. The
    /// executor never sees retained results, so opening it is the kernel's.
    ViewResult { key: String },
    /// A query run under `-- @view`, whose table opens in the data explorer as
    /// well as showing in the cell. `title` is the one the directive gave, if
    /// any.
    View {
        df: DataFrame,
        title: Option<String>,
    },
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
            Self::ConnectionChanged { display_name } => f
                .debug_struct("ConnectionChanged")
                .field("display_name", display_name)
                .finish(),
            Self::ViewResult { key } => f.debug_struct("ViewResult").field("key", key).finish(),
            Self::View { df, title } => f
                .debug_struct("View")
                .field("rows", &df.height())
                .field("columns", &df.width())
                .field("title", title)
                .finish(),
        }
    }
}

/// The host portion of a `user:pass@host:port/db` URI body.
fn host_of(body: &str) -> &str {
    let after_auth = body.rsplit_once('@').map(|(_, h)| h).unwrap_or(body);
    let hostport = after_auth.split('/').next().unwrap_or("");
    hostport.split(':').next().unwrap_or("")
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
    // Backend schemes: registry display name plus the parsed host.
    if let Ok(conn) = ggsql::reader::connection::ConnUri::parse(uri) {
        if let Some(entry) = ggsql::reader::registry::by_scheme(&conn.scheme) {
            let host = host_of(&conn.body);
            if !host.is_empty() {
                return format!("{} ({})", entry.display_name, host);
            }
            return entry.display_name.to_string();
        }
    }
    uri.to_string()
}

/// Detect the database type name from a connection URI (e.g. "DuckDB", "Snowflake").
pub fn type_name_for_uri(uri: &str) -> String {
    if let Some(odbc) = uri.strip_prefix("odbc://") {
        let driver = extract_odbc_value(odbc, "driver");
        if let Some(entry) = ggsql::reader::registry::detect(None, driver.as_deref()) {
            return entry.display_name.to_string();
        }
        return "ODBC".to_string();
    }
    if let Ok(conn) = ggsql::reader::connection::ConnUri::parse(uri) {
        if let Some(entry) = ggsql::reader::registry::by_scheme(&conn.scheme) {
            return entry.display_name.to_string();
        }
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
    if let Ok(conn) = ggsql::reader::connection::ConnUri::parse(uri) {
        let host = host_of(&conn.body);
        if !host.is_empty() {
            return host.to_string();
        }
    }
    uri.to_string()
}

/// The `-- @connect:` meta-command prefix.
const META_CONNECT_PREFIX: &str = "-- @connect:";
/// The `-- @uncache` meta-command prefix.
const META_UNCACHE_PREFIX: &str = "-- @uncache";
/// The `-- @view` meta-command prefix.
const META_VIEW_PREFIX: &str = "-- @view";
/// The `-- @view-result` meta-command prefix.
const META_VIEW_RESULT_PREFIX: &str = "-- @view-result";

/// A leading cell directive expressed as a `--` line comment.
#[derive(Debug, PartialEq, Eq)]
pub enum MetaCommand {
    /// Switch the active reader to the given connection URI.
    Connect(String),
    /// Clear the active reader's cache.
    Uncache,
    /// Show the rest of the cell's result in the data explorer, under the
    /// given title if there is one.
    View(Option<String>),
    /// Open a retained query result in the data explorer. Sent by the ggsql
    /// extension when the link under a console result is clicked.
    ViewResult(String),
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
    if let Some(key) = line.strip_prefix(META_VIEW_RESULT_PREFIX) {
        // A space is required, so `-- @view-results` is not taken for it.
        if let Some(key) = key.strip_prefix(char::is_whitespace) {
            return Some((MetaCommand::ViewResult(key.trim().to_string()), rest));
        }
    }
    if line == META_VIEW_PREFIX {
        return Some((MetaCommand::View(None), rest));
    }
    if let Some(title) = line
        .strip_prefix(META_VIEW_PREFIX)
        .and_then(|t| t.strip_prefix(':'))
    {
        let title = title.trim();
        let title = (!title.is_empty()).then(|| title.to_string());
        return Some((MetaCommand::View(title), rest));
    }
    None
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
        let reader = reader_from_uri(uri)?;

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
        let mut view: Option<Option<String>> = None;
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
                MetaCommand::View(title) => view = Some(title),
                MetaCommand::ViewResult(key) => {
                    if !rest.trim().is_empty() {
                        anyhow::bail!("`-- @view-result` must be the only line in its cell");
                    }
                    return Ok(ExecutionResult::ViewResult { key });
                }
            }
            code = rest;
        }

        if let Some(title) = view {
            return self.execute_view(code, title);
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

        // 4. Hand back the resolved plot. Choosing a format is the display
        //    layer's job, because only it knows where the output is going.
        Ok(ExecutionResult::Visualization(Box::new(spec)))
    }

    /// Run the query under a `-- @view` directive for the data explorer.
    ///
    /// Checked before running, so a mistake costs nothing: the explorer shows
    /// a table, so a plot query and a cell with nothing to show are refused.
    fn execute_view(&mut self, code: &str, title: Option<String>) -> Result<ExecutionResult> {
        if code.trim().is_empty() {
            anyhow::bail!("`-- @view` needs a query after it to show in the Data Explorer");
        }
        if validate(code)?.has_visual() {
            anyhow::bail!(
                "`-- @view` shows a query's table, not a plot. Remove the VISUALISE clause to view the data."
            );
        }
        let df = self.reader.execute_sql(code)?;
        if df.width() == 0 {
            anyhow::bail!("`-- @view` needs a query that returns a table");
        }
        Ok(ExecutionResult::View { df, title })
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
    fn test_take_leading_meta_view_result() {
        assert_eq!(
            take_leading_meta("-- @view-result abc123"),
            Some((MetaCommand::ViewResult("abc123".to_string()), ""))
        );
        assert_eq!(take_leading_meta("-- @view-results abc123"), None);
    }

    #[test]
    fn test_view_result_must_stand_alone() {
        let mut executor = QueryExecutor::new().unwrap();
        match executor.execute("-- @view-result abc123").unwrap() {
            ExecutionResult::ViewResult { key } => assert_eq!(key, "abc123"),
            other => panic!("expected ViewResult, got {other:?}"),
        }
        assert!(executor
            .execute("-- @view-result abc123\nSELECT 1")
            .is_err());
    }

    #[test]
    fn test_take_leading_meta_view() {
        assert_eq!(
            take_leading_meta("-- @view\nSELECT 1"),
            Some((MetaCommand::View(None), "SELECT 1"))
        );
        assert_eq!(
            take_leading_meta("-- @view: Sales by region \nSELECT 1"),
            Some((
                MetaCommand::View(Some("Sales by region".to_string())),
                "SELECT 1"
            ))
        );
        assert_eq!(
            take_leading_meta("-- @view:\nSELECT 1"),
            Some((MetaCommand::View(None), "SELECT 1"))
        );
        // Anything else after the name is an ordinary comment.
        assert_eq!(take_leading_meta("-- @viewer\nSELECT 1"), None);
        assert_eq!(take_leading_meta("-- @view abc\nSELECT 1"), None);
    }

    #[test]
    fn test_view_runs_the_query_for_the_explorer() {
        let mut executor = QueryExecutor::new().unwrap();
        match executor
            .execute("-- @view: Numbers\nSELECT * FROM range(3) t(n)")
            .unwrap()
        {
            ExecutionResult::View { df, title } => {
                assert_eq!(df.height(), 3);
                assert_eq!(title.as_deref(), Some("Numbers"));
            }
            other => panic!("expected View, got {other:?}"),
        }
    }

    #[test]
    fn test_view_combines_with_other_directives() {
        let mut executor = QueryExecutor::new().unwrap();
        let result = executor
            .execute("-- @uncache\n-- @view\nSELECT 1 AS x")
            .unwrap();
        assert!(matches!(result, ExecutionResult::View { title: None, .. }));
    }

    #[test]
    fn test_view_refuses_what_has_no_table() {
        let mut executor = QueryExecutor::new().unwrap();
        let mut err = |code: &str| executor.execute(code).unwrap_err().to_string();
        assert!(err("-- @view").contains("needs a query"));
        assert!(err("-- @view\nCREATE TABLE t (x INT)").contains("returns a table"));
        assert!(
            err("-- @view\nSELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point").contains("not a plot")
        );
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
        // Registry-backed schemes get display name + parsed host.
        assert_eq!(
            display_name_for_uri("postgres://user:pw@db.example.com:5432/sales"),
            "PostgreSQL (db.example.com)"
        );
        assert_eq!(
            display_name_for_uri("snowflake://acct.snowflakecomputing.com/wh"),
            "Snowflake (acct.snowflakecomputing.com)"
        );
    }
}
