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
    /// Google BigQuery native connection.
    #[allow(dead_code)]
    BigQuery(BigQueryConnectionInfo),
}

/// Parsed BigQuery connection information.
#[derive(Debug, Clone, PartialEq)]
pub struct BigQueryConnectionInfo {
    /// Explicit project ID, or `None` to resolve from ADC / environment.
    pub project_id: Option<String>,
    pub default_dataset: Option<String>,
    pub location: Option<String>,
}

/// Parse a connection string into connection information
///
/// # Supported Formats
///
/// - `duckdb://memory` - DuckDB in-memory database
/// - `duckdb://...` - DuckDB path
/// - `postgres://...` - PostgreSQL connection string
/// - `sqlite://...` - SQLite file path
/// - `bigquery://[project[/dataset]][?location=US]` - Google BigQuery (project defaults to ADC)
pub fn parse_connection_string(uri: &str) -> Result<ConnectionInfo> {
    if uri == "duckdb://memory" {
        return Ok(ConnectionInfo::DuckDBMemory);
    }

    if let Some(path) = uri.strip_prefix("duckdb://") {
        if path.is_empty() {
            return Err(GgsqlError::ReaderError(
                "DuckDB file path cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::DuckDBFile(path.to_string()));
    }

    if uri.starts_with("postgres://") || uri.starts_with("postgresql://") {
        return Ok(ConnectionInfo::PostgreSQL(uri.to_string()));
    }

    if let Some(path) = uri.strip_prefix("sqlite://") {
        if path.is_empty() {
            return Err(GgsqlError::ReaderError(
                "SQLite file path cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::SQLite(path.to_string()));
    }

    if let Some(conn_str) = uri.strip_prefix("odbc://") {
        if conn_str.is_empty() {
            return Err(GgsqlError::ReaderError(
                "ODBC connection string cannot be empty".to_string(),
            ));
        }
        return Ok(ConnectionInfo::ODBC(conn_str.to_string()));
    }

    if let Some(rest) = uri.strip_prefix("bigquery://") {
        let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
        let mut parts = path.split('/').filter(|s| !s.is_empty());
        let project_id = parts.next().map(|s| s.to_string());
        let default_dataset = parts.next().map(|s| s.to_string());

        if parts.next().is_some() {
            return Err(GgsqlError::ReaderError(
                "BigQuery URI must be bigquery://[project[/dataset]][?location=...]".to_string(),
            ));
        }

        let location = query.split('&').find_map(|part| {
            let (key, value) = part.split_once('=')?;
            (key == "location" && !value.is_empty()).then(|| value.to_string())
        });

        return Ok(ConnectionInfo::BigQuery(BigQueryConnectionInfo {
            project_id,
            default_dataset,
            location,
        }));
    }

    Err(GgsqlError::ReaderError(format!(
        "Unsupported connection string format: {}. Supported: duckdb://, postgres://, sqlite://, odbc://, bigquery://",
        uri
    )))
}

/// Extract a value from an ODBC connection string by key, stripping braces.
pub fn extract_odbc_value(conn_str: &str, key: &str) -> Option<String> {
    let lower = conn_str.to_lowercase();
    let prefix = format!("{}=", key);
    let start = lower.find(&prefix)?;
    let rest = &conn_str[start + prefix.len()..];
    let value = rest.split(';').next().unwrap_or("");
    let value = value.trim().trim_matches(|c| c == '{' || c == '}');
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
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
        assert_eq!(info, ConnectionInfo::DuckDBFile("/tmp/data.db".to_string()));
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
    fn test_sqlite_absolute() {
        let info = parse_connection_string("sqlite:///tmp/data.db").unwrap();
        assert_eq!(info, ConnectionInfo::SQLite("/tmp/data.db".to_string()));
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

    #[test]
    fn test_bigquery_bare() {
        let info = parse_connection_string("bigquery://").unwrap();
        assert_eq!(
            info,
            ConnectionInfo::BigQuery(BigQueryConnectionInfo {
                project_id: None,
                default_dataset: None,
                location: None,
            })
        );
    }

    #[test]
    fn test_bigquery_project() {
        let info = parse_connection_string("bigquery://my-project").unwrap();
        assert_eq!(
            info,
            ConnectionInfo::BigQuery(BigQueryConnectionInfo {
                project_id: Some("my-project".to_string()),
                default_dataset: None,
                location: None,
            })
        );
    }

    #[test]
    fn test_bigquery_project_dataset_location() {
        let info = parse_connection_string("bigquery://my-project/analytics?location=asia-northeast1")
            .unwrap();
        assert_eq!(
            info,
            ConnectionInfo::BigQuery(BigQueryConnectionInfo {
                project_id: Some("my-project".to_string()),
                default_dataset: Some("analytics".to_string()),
                location: Some("asia-northeast1".to_string()),
            })
        );
    }
}
