//! Connection string handling for data sources.
//!
//! Maps URI-style connection strings (`duckdb://…`, `sqlite://…`, `odbc://…`) and
//! the composite caching form (`<primary>+<cache>://…`) to readers.

use crate::reader::Reader;
use crate::{GgsqlError, Result};

/// Split a composite cache URI `<primary>+<cache>://<rest>` into the primary
/// connection URI and the cache backend scheme.
///
/// Returns `None` when there is no `+<cache>` before `://` (a plain URI).
///
/// # Example
/// ```
/// use ggsql::reader::connection::split_cache_uri;
/// assert_eq!(
///     split_cache_uri("odbc+duckdb://DSN=foo"),
///     Some(("odbc://DSN=foo".to_string(), "duckdb".to_string()))
/// );
/// assert_eq!(split_cache_uri("duckdb://memory"), None);
/// ```
pub fn split_cache_uri(uri: &str) -> Option<(String, String)> {
    let (scheme, rest) = uri.split_once("://")?;
    let (primary, cache) = scheme.split_once('+')?;
    if primary.is_empty() || cache.is_empty() || cache.contains('+') {
        return None;
    }
    Some((format!("{}://{}", primary, rest), cache.to_string()))
}

/// Map a cache-backend scheme to its in-memory connection URI.
#[cfg(any(feature = "duckdb", feature = "sqlite"))]
fn cache_uri(scheme: &str) -> Result<&'static str> {
    match scheme {
        "duckdb" => Ok("duckdb://memory"),
        "sqlite" => Ok("sqlite://memory"),
        _ => Err(GgsqlError::ReaderError(format!(
            "Unsupported cache backend '{}'. Supported: duckdb, sqlite",
            scheme
        ))),
    }
}

/// Build a reader from a non-composite connection URI
pub fn build_reader(uri: &str) -> Result<Box<dyn Reader + Send>> {
    if uri.starts_with("duckdb://") {
        #[cfg(feature = "duckdb")]
        {
            return Ok(Box::new(
                crate::reader::DuckDBReader::from_connection_string(uri)?,
            ));
        }
        #[cfg(not(feature = "duckdb"))]
        {
            return Err(GgsqlError::ReaderError(
                "DuckDB reader not compiled in. Rebuild with --features duckdb".to_string(),
            ));
        }
    }
    if uri.starts_with("sqlite://") {
        #[cfg(feature = "sqlite")]
        {
            return Ok(Box::new(
                crate::reader::SqliteReader::from_connection_string(uri)?,
            ));
        }
        #[cfg(not(feature = "sqlite"))]
        {
            return Err(GgsqlError::ReaderError(
                "SQLite reader not compiled in. Rebuild with --features sqlite".to_string(),
            ));
        }
    }
    if uri.starts_with("odbc://") {
        #[cfg(feature = "odbc")]
        {
            return Ok(Box::new(crate::reader::OdbcReader::from_connection_string(
                uri,
            )?));
        }
        #[cfg(not(feature = "odbc"))]
        {
            return Err(GgsqlError::ReaderError(
                "ODBC reader not compiled in. Rebuild with --features odbc".to_string(),
            ));
        }
    }
    if uri.starts_with("postgres://") || uri.starts_with("postgresql://") {
        return Err(GgsqlError::ReaderError(
            "PostgreSQL reader is not yet implemented".to_string(),
        ));
    }
    Err(GgsqlError::ReaderError(format!(
        "Unsupported connection string: {}. Supported: duckdb://, sqlite://, odbc://",
        uri
    )))
}

/// Construct a reader from a connection URI, wrapping it in a [`CachingReader`]
/// when the URI uses the composite `<primary>+<cache>://` form.
///
/// [`CachingReader`]: crate::reader::CachingReader
pub fn reader_from_uri(uri: &str) -> Result<Box<dyn Reader + Send>> {
    if let Some((primary_uri, cache_scheme)) = split_cache_uri(uri) {
        #[cfg(any(feature = "duckdb", feature = "sqlite"))]
        {
            let primary = build_reader(&primary_uri)?;
            let cache = build_reader(cache_uri(&cache_scheme)?)?;
            return Ok(Box::new(crate::reader::CachingReader::new(primary, cache)));
        }
        #[cfg(not(any(feature = "duckdb", feature = "sqlite")))]
        {
            let _ = (&primary_uri, &cache_scheme);
            return Err(GgsqlError::ReaderError(
                "Caching layer requires the duckdb or sqlite feature".to_string(),
            ));
        }
    }
    build_reader(uri)
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
    fn test_build_reader_unsupported_scheme() {
        let err = build_reader("mysql://localhost/db")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("Unsupported connection string"), "got: {err}");
    }

    #[test]
    fn test_build_reader_postgres_not_implemented() {
        let err = build_reader("postgres://user@localhost/db")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not yet implemented"), "got: {err}");
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_build_reader_duckdb_memory_and_empty() {
        assert!(build_reader("duckdb://memory").is_ok());
        assert!(build_reader("duckdb://").is_err());
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_build_reader_sqlite_memory() {
        assert!(build_reader("sqlite://memory").is_ok());
        assert!(build_reader("sqlite://:memory:").is_ok());
    }

    #[cfg(all(feature = "duckdb", feature = "sqlite"))]
    #[test]
    fn test_reader_from_uri_composite_builds() {
        assert!(reader_from_uri("duckdb+sqlite://memory").is_ok());
        assert!(reader_from_uri("sqlite+duckdb://memory").is_ok());
    }

    #[test]
    fn test_split_cache_uri_odbc_duckdb() {
        assert_eq!(
            split_cache_uri("odbc+duckdb://Driver=Snowflake;Server=x"),
            Some((
                "odbc://Driver=Snowflake;Server=x".to_string(),
                "duckdb".to_string()
            ))
        );
    }

    #[test]
    fn test_split_cache_uri_duckdb_sqlite_memory() {
        assert_eq!(
            split_cache_uri("duckdb+sqlite://memory"),
            Some(("duckdb://memory".to_string(), "sqlite".to_string()))
        );
    }

    #[test]
    fn test_split_cache_uri_plain_is_none() {
        assert_eq!(split_cache_uri("duckdb://memory"), None);
        assert_eq!(split_cache_uri("odbc://DSN=x"), None);
    }

    #[test]
    fn test_split_cache_uri_rejects_multiple_plus() {
        assert_eq!(split_cache_uri("a+b+c://x"), None);
    }

    #[test]
    fn test_split_cache_uri_rejects_empty_parts() {
        assert_eq!(split_cache_uri("+duckdb://x"), None);
        assert_eq!(split_cache_uri("odbc+://x"), None);
    }
}
