//! Connection string parsing for data sources
//!
//! Parses URI-style connection strings to determine database type and connection parameters.

use crate::{ggsqlError, Result};

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
}

/// Parse a connection string into connection information
///
/// # Supported Formats
///
/// - `duckdb://memory` - DuckDB in-memory database
/// - `duckdb:///absolute/path/file.db` - DuckDB file (absolute path)
/// - `duckdb://relative/file.db` - DuckDB file (relative path)
/// - `postgres://...` - PostgreSQL connection string
/// - `sqlite://...` - SQLite file path
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
/// ```
pub fn parse_connection_string(uri: &str) -> Result<ConnectionInfo> {
    if uri == "duckdb://memory" {
        return Ok(ConnectionInfo::DuckDBMemory);
    }

    if let Some(path) = uri.strip_prefix("duckdb://") {
        // Remove leading slashes for file paths
        let cleaned_path = path.trim_start_matches('/');
        if cleaned_path.is_empty() {
            return Err(ggsqlError::ReaderError(
                "DuckDB file path cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::DuckDBFile(cleaned_path.to_string()));
    }

    if uri.starts_with("postgres://") || uri.starts_with("postgresql://") {
        return Ok(ConnectionInfo::PostgreSQL(uri.to_string()));
    }

    if let Some(path) = uri.strip_prefix("sqlite://") {
        let cleaned_path = path.trim_start_matches('/');
        if cleaned_path.is_empty() {
            return Err(ggsqlError::ReaderError(
                "SQLite file path cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::SQLite(cleaned_path.to_string()));
    }

    Err(ggsqlError::ReaderError(format!(
        "Unsupported connection string format: {}. Supported: duckdb://, postgres://, sqlite://",
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
        let info = parse_connection_string("duckdb:///tmp/data.db").unwrap();
        assert_eq!(info, ConnectionInfo::DuckDBFile("tmp/data.db".to_string()));
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
    fn test_unsupported_scheme() {
        let result = parse_connection_string("mysql://localhost/db");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported connection string"));
    }
}
