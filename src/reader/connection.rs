//! Connection string parsing for data sources
//!
//! Parses URI-style connection strings to determine database type and connection parameters.

use crate::{GgsqlError, Result};

/// Parsed connection information
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionInfo {
    /// DuckDB in-memory database
    DuckDBMemory,
    /// DuckDB file-based database
    DuckDBFile(String),
    /// PostgreSQL connection
    #[allow(dead_code)]
    PostgreSQL(String),
    /// SQLite file-based database
    #[allow(dead_code)]
    SQLite(String),
    /// Generic ODBC connection (raw connection string after `odbc://` prefix)
    #[allow(dead_code)]
    ODBC(String),
}

/// Parse a DuckDB/SQLite URI body into a filesystem path, following
/// SQLAlchemy conventions.
///
/// After the `scheme://` prefix has been stripped, `body` is interpreted as:
/// - `memory` -> handled by the caller (only valid for `duckdb://memory`)
/// - `<relative/path>` -> relative path, returned verbatim
/// - `/<absolute/path>` -> absolute path, returned with the leading `/`
/// - `//...` or empty -> error (ambiguous / missing path)
fn parse_uri_path(scheme: &str, body: &str) -> Result<String> {
    if body.is_empty() {
        return Err(GgsqlError::ReaderError(format!(
            "{} file path cannot be empty",
            scheme
        )));
    }

    if body.starts_with("//") {
        return Err(GgsqlError::ReaderError(format!(
            "Ambiguous {scheme} URI '{scheme}://{body}': use '{scheme}://relative/path' \
             for a relative path or '{scheme}:///absolute/path' for an absolute path",
            scheme = scheme,
            body = body,
        )));
    }

    Ok(body.to_string())
}

/// Parse a connection string into connection information
///
/// # Supported Formats
///
/// - `duckdb://memory` - DuckDB in-memory database
/// - `duckdb:///absolute/path/file.db` - DuckDB file (absolute path, SQLAlchemy convention)
/// - `duckdb://relative/file.db` - DuckDB file (relative path)
/// - `postgres://...` - PostgreSQL connection string
/// - `sqlite://...` - SQLite file path (same `//` vs `///` rules as DuckDB)
///
/// # Examples
///
/// ```
/// use ggsql::reader::connection::{parse_connection_string, ConnectionInfo};
///
/// let info = parse_connection_string("duckdb://memory").unwrap();
/// assert_eq!(info, ConnectionInfo::DuckDBMemory);
///
/// let info = parse_connection_string("duckdb://data.db").unwrap();
/// assert_eq!(info, ConnectionInfo::DuckDBFile("data.db".to_string()));
///
/// let info = parse_connection_string("duckdb:///tmp/data.db").unwrap();
/// assert_eq!(info, ConnectionInfo::DuckDBFile("/tmp/data.db".to_string()));
/// ```
pub fn parse_connection_string(uri: &str) -> Result<ConnectionInfo> {
    if uri == "duckdb://memory" {
        return Ok(ConnectionInfo::DuckDBMemory);
    }

    if let Some(body) = uri.strip_prefix("duckdb://") {
        let path = parse_uri_path("duckdb", body)?;
        return Ok(ConnectionInfo::DuckDBFile(path));
    }

    if uri.starts_with("postgres://") || uri.starts_with("postgresql://") {
        return Ok(ConnectionInfo::PostgreSQL(uri.to_string()));
    }

    if let Some(body) = uri.strip_prefix("sqlite://") {
        let path = parse_uri_path("sqlite", body)?;
        return Ok(ConnectionInfo::SQLite(path));
    }

    if let Some(conn_str) = uri.strip_prefix("odbc://") {
        if conn_str.is_empty() {
            return Err(GgsqlError::ReaderError(
                "ODBC connection string cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::ODBC(conn_str.to_string()));
    }

    Err(GgsqlError::ReaderError(format!(
        "Unsupported connection string format: {}. Supported: duckdb://, postgres://, sqlite://, odbc://",
        uri
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duckdb_memory() {
        let info = parse_connection_string("duckdb://memory").unwrap();
        assert_eq!(info, ConnectionInfo::DuckDBMemory);
    }

    #[test]
    fn test_duckdb_file_relative() {
        let info = parse_connection_string("duckdb://data.db").unwrap();
        assert_eq!(info, ConnectionInfo::DuckDBFile("data.db".to_string()));
    }

    #[test]
    fn test_duckdb_file_absolute() {
        // Three slashes -> absolute path (SQLAlchemy convention). The leading
        // `/` must be preserved so DuckDB opens the intended file rather than
        // silently creating a relative phantom DB. See issue #345.
        let info = parse_connection_string("duckdb:///tmp/data.db").unwrap();
        assert_eq!(
            info,
            ConnectionInfo::DuckDBFile("/tmp/data.db".to_string())
        );
    }

    #[test]
    fn test_duckdb_file_nested() {
        let info = parse_connection_string("duckdb://path/to/data.db").unwrap();
        assert_eq!(
            info,
            ConnectionInfo::DuckDBFile("path/to/data.db".to_string())
        );
    }

    #[test]
    fn test_duckdb_file_four_slashes_rejected() {
        // Four slashes is ambiguous — reject with a clear message instead of
        // silently interpreting as an absolute path.
        let result = parse_connection_string("duckdb:////tmp/data.db");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Ambiguous"), "unexpected error: {}", err);
    }

    #[test]
    fn test_sqlite_absolute() {
        let info = parse_connection_string("sqlite:///tmp/data.db").unwrap();
        assert_eq!(info, ConnectionInfo::SQLite("/tmp/data.db".to_string()));
    }

    #[test]
    fn test_sqlite_four_slashes_rejected() {
        let result = parse_connection_string("sqlite:////tmp/data.db");
        assert!(result.is_err());
    }

    #[test]
    fn test_postgres() {
        let uri = "postgres://user:pass@localhost/db";
        let info = parse_connection_string(uri).unwrap();
        assert_eq!(info, ConnectionInfo::PostgreSQL(uri.to_string()));
    }

    #[test]
    fn test_postgresql_alias() {
        let uri = "postgresql://user:pass@localhost/db";
        let info = parse_connection_string(uri).unwrap();
        assert_eq!(info, ConnectionInfo::PostgreSQL(uri.to_string()));
    }

    #[test]
    fn test_sqlite() {
        let info = parse_connection_string("sqlite://data.db").unwrap();
        assert_eq!(info, ConnectionInfo::SQLite("data.db".to_string()));
    }

    #[test]
    fn test_empty_duckdb_path() {
        let result = parse_connection_string("duckdb://");
        assert!(result.is_err());
    }

    #[test]
    fn test_odbc() {
        let info = parse_connection_string(
            "odbc://Driver=Snowflake;Server=myaccount.snowflakecomputing.com",
        )
        .unwrap();
        assert_eq!(
            info,
            ConnectionInfo::ODBC(
                "Driver=Snowflake;Server=myaccount.snowflakecomputing.com".to_string()
            )
        );
    }

    #[test]
    fn test_odbc_empty() {
        let result = parse_connection_string("odbc://");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_scheme() {
        let result = parse_connection_string("mysql://localhost/db");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported connection string"));
    }
}
