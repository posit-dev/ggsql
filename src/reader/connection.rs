//! Connection string handling for data sources.
//!
//! Maps URI-style connection strings (`duckdb://…`, `sqlite://…`, `odbc://…`) and
//! the composite caching form (`<cache>+<primary>://…`) to readers.

use crate::reader::Reader;
use crate::{GgsqlError, Result};

/// Split a composite cache URI `<cache>+<primary>://<rest>` into the primary
/// connection URI and the cache backend scheme.
///
/// Naming the cache first keeps the primary URI, and any query parameters it
/// carries, contiguous: `duckdb+odbc://DSN=foo` wraps `odbc://DSN=foo`.
///
/// Returns `None` when there is no `<cache>+` before `://` (a plain URI).
///
/// # Example
/// ```
/// use ggsql::reader::connection::split_cache_uri;
/// assert_eq!(
///     split_cache_uri("duckdb+odbc://DSN=foo"),
///     Some(("odbc://DSN=foo".to_string(), "duckdb".to_string()))
/// );
/// assert_eq!(split_cache_uri("duckdb://memory"), None);
/// ```
pub fn split_cache_uri(uri: &str) -> Option<(String, String)> {
    let (scheme, rest) = uri.split_once("://")?;
    let (cache, primary) = scheme.split_once('+')?;
    // `split_once` leaves any further `+` in the second half.
    if cache.is_empty() || primary.is_empty() || primary.contains('+') {
        return None;
    }
    Some((format!("{}://{}", primary, rest), cache.to_string()))
}

/// Cache-config keys recognised in a connection URI's trailing `?` query string.
#[cfg(any(feature = "duckdb", feature = "sqlite"))]
const KNOWN_CACHE_PARAMS: &[&str] = &["cache_ttl", "cache_max_bytes", "cache_disabled"];

/// Pull cache-config keys out of a connection URI's trailing `?key=value&…`
/// query string, returning the URI with those keys removed plus the overrides.
#[cfg(any(feature = "duckdb", feature = "sqlite"))]
fn strip_cache_params(uri: &str) -> (String, crate::reader::cache::CacheConfigOverride) {
    use crate::reader::cache::{parse_human_bytes, CacheConfigOverride};

    let mut over = CacheConfigOverride::default();
    let Some((body, query)) = uri.split_once('?') else {
        return (uri.to_string(), over);
    };

    let mut kept: Vec<&str> = Vec::new();
    for segment in query.split('&') {
        match segment.split_once('=') {
            Some((key, value)) if KNOWN_CACHE_PARAMS.contains(&key) => match key {
                "cache_ttl" => over.ttl_secs = value.trim().parse::<u64>().ok(),
                "cache_max_bytes" => over.max_bytes = parse_human_bytes(value),
                "cache_disabled" => {
                    let v = value.trim().to_ascii_lowercase();
                    over.enabled = Some(!matches!(v.as_str(), "1" | "true" | "yes"));
                }
                _ => unreachable!("validated against KNOWN_CACHE_PARAMS"),
            },
            // Not a cache key: keep it on the URI for the primary reader.
            _ => kept.push(segment),
        }
    }

    if kept.is_empty() {
        (body.to_string(), over)
    } else {
        (format!("{}?{}", body, kept.join("&")), over)
    }
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
    // Backend-specific schemes (postgres://, mysql://, snowflake://, …).
    // Selection: ADBC is tried first when a driver library is available;
    // otherwise we fall back to ODBC automatically. A `reader=odbc` query
    // param forces the ODBC path.
    if let Some((scheme, rest)) = uri.split_once("://") {
        let scheme = scheme.to_ascii_lowercase();
        let known = crate::reader::dialects::dialect_for_scheme(&scheme).is_some()
            || scheme == "adbc"
            || scheme == "flightsql";
        if known {
            return build_backend_reader(&scheme, rest, uri);
        }
    }
    Err(GgsqlError::ReaderError(format!(
        "Unsupported connection string: {}. Supported: duckdb://, sqlite://, odbc://, \
         adbc://, postgres://, mysql://, snowflake://, mssql://, bigquery://, \
         databricks://, clickhouse://, trino://, redshift://, oracle://, \
         exasol://, monetdb://, druid://, drill://, datafusion://, flightsql://",
        uri
    )))
}

/// True when the URI query string carries `reader=odbc`.
#[cfg(feature = "adbc")]
fn uri_forces_odbc(rest: &str) -> bool {
    rest.split_once('?')
        .map(|(_, q)| q.split('&').any(|seg| seg.to_lowercase() == "reader=odbc"))
        .unwrap_or(false)
}

/// Build a reader for a backend-specific scheme, preferring ADBC when a
/// driver library is loadable and falling back to ODBC automatically.
#[cfg_attr(not(all(feature = "adbc", feature = "odbc")), allow(unused_variables))]
fn build_backend_reader(scheme: &str, rest: &str, uri: &str) -> Result<Box<dyn Reader + Send>> {
    #[cfg(feature = "adbc")]
    if !uri_forces_odbc(rest)
        && (crate::reader::adbc::adbc_driver_available(if scheme == "adbc" {
            rest.split('?').next().unwrap_or(rest)
        } else {
            scheme
        }) || scheme == "adbc")
    {
        // The driver library loaded (probe); a failure here is a genuine
        // connection/config error, surfaced with an ODBC hint rather
        // than silently masking it.
        return crate::reader::adbc::AdbcReader::from_connection_string(uri)
            .map(|r| Box::new(r) as Box<dyn Reader + Send>)
            .map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "{}. The ADBC driver was found but connecting failed. \
                     To use ODBC instead, add ?reader=odbc to the URI.",
                    e
                ))
            });
    }

    #[cfg(feature = "odbc")]
    {
        if let Some(conn_str) = synthesize_odbc_conn_str(scheme, rest) {
            let dialect = crate::reader::dialects::dialect_for_scheme(scheme)
                .map(|d| d as Box<dyn crate::reader::SqlDialect>);
            let reader = crate::reader::OdbcReader::from_odbc_conn_str(&conn_str, dialect)?;
            return Ok(Box::new(reader));
        }
    }

    Err(GgsqlError::ReaderError(format!(
        "Could not connect for scheme '{}://'. No loadable ADBC driver was found \
         (searched ${}, ADBC driver paths, and system library paths), and no ODBC \
         fallback was available: provide DSN= or Driver= in the URI query \
         ({}://host/db?DSN=mydsn) or set {} to the name of an installed ODBC driver.",
        scheme,
        adbc_driver_env_var(scheme),
        scheme,
        odbc_driver_env_var(scheme),
    )))
}

/// Name of the env var overriding the ADBC driver for a scheme
/// (mirrors `adbc::driver_env_var`, duplicated here so error messages work
/// in builds without the `adbc` feature).
fn adbc_driver_env_var(scheme: &str) -> String {
    format!("GGSQL_{}_ADBC_DRIVER", env_var_scheme(scheme))
}

/// Name of the env var specifying the ODBC driver for a scheme.
fn odbc_driver_env_var(scheme: &str) -> String {
    format!("GGSQL_{}_ODBC_DRIVER", env_var_scheme(scheme))
}

/// Uppercased, identifier-safe form of a scheme for env var names.
fn env_var_scheme(scheme: &str) -> String {
    scheme
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Parsed form of `user:pass@host:port/db?params` backend URIs.
#[cfg(feature = "odbc")]
struct BackendUri<'a> {
    user: Option<&'a str>,
    password: Option<&'a str>,
    host: Option<&'a str>,
    port: Option<&'a str>,
    database: Option<&'a str>,
    params: Vec<&'a str>,
}

#[cfg(feature = "odbc")]
fn parse_backend_uri(rest: &str) -> BackendUri<'_> {
    let (body, query) = rest.split_once('?').unwrap_or((rest, ""));
    let params: Vec<&str> = query.split('&').filter(|s| !s.is_empty()).collect();

    let (auth, hostpath) = match body.split_once('@') {
        Some((a, h)) => (Some(a), h),
        None => (None, body),
    };
    let (user, password) = match auth.and_then(|a| a.split_once(':')) {
        Some((u, p)) => (Some(u), Some(p)),
        None => (auth.filter(|a| !a.is_empty()), None),
    };
    let (hostport, database) = match hostpath.split_once('/') {
        Some((h, d)) => (h, Some(d).filter(|d| !d.is_empty())),
        None => (hostpath, None),
    };
    let (host, port) = match hostport.split_once(':') {
        Some((h, p)) => (Some(h).filter(|h| !h.is_empty()), Some(p)),
        None => (Some(hostport).filter(|h| !h.is_empty()), None),
    };
    BackendUri {
        user,
        password,
        host,
        port,
        database,
        params,
    }
}

/// Try to assemble an ODBC connection string for a backend URI.
///
/// Returns `Some` when the URI query carries `DSN=` or `Driver=`, or when
/// `GGSQL_<SCHEME>_ODBC_DRIVER` names an installed driver; otherwise `None`
/// so the caller can report that no fallback was configured.
#[cfg(feature = "odbc")]
fn synthesize_odbc_conn_str(scheme: &str, rest: &str) -> Option<String> {
    let parsed = parse_backend_uri(rest);

    // Case 1: query params are already a (partial) ODBC connection string.
    let dsn = parsed
        .params
        .iter()
        .find_map(|p| p.strip_prefix("DSN=").or_else(|| p.strip_prefix("dsn=")));
    let driver = parsed
        .params
        .iter()
        .find_map(|p| {
            p.strip_prefix("Driver=")
                .or_else(|| p.strip_prefix("driver="))
        })
        .map(|d| d.trim_matches(|c| c == '{' || c == '}'));
    let env_driver = std::env::var(odbc_driver_env_var(scheme)).ok();

    let (mut parts, mut skip_keys): (Vec<String>, Vec<&str>) = (Vec::new(), Vec::new());
    if let Some(dsn) = dsn {
        parts.push(format!("DSN={}", dsn));
        skip_keys.push("dsn");
    } else {
        let driver = driver.or(env_driver.as_deref())?;
        parts.push(format!("Driver={{{}}}", driver));
        skip_keys.push("driver");
        if let Some(host) = parsed.host {
            parts.push(format!("Server={}", host));
        }
        if let Some(port) = parsed.port {
            parts.push(format!("Port={}", port));
        }
        if let Some(db) = parsed.database {
            parts.push(format!("Database={}", db));
        }
    }
    if let Some(user) = parsed.user {
        parts.push(format!("UID={}", user));
    }
    if let Some(password) = parsed.password {
        parts.push(format!("PWD={}", password));
    }
    // Pass through remaining query params verbatim.
    for p in &parsed.params {
        let key = p.split('=').next().unwrap_or("").to_ascii_lowercase();
        if skip_keys.contains(&key.as_str()) || key == "reader" {
            continue;
        }
        parts.push(p.to_string());
    }
    Some(parts.join(";"))
}

/// Construct a reader from a connection URI, wrapping it in a [`CachingReader`]
/// when the URI uses the composite `<cache>+<primary>://` form.
///
/// [`CachingReader`]: crate::reader::CachingReader
pub fn reader_from_uri(uri: &str) -> Result<Box<dyn Reader + Send>> {
    if let Some((primary_uri, cache_scheme)) = split_cache_uri(uri) {
        #[cfg(any(feature = "duckdb", feature = "sqlite"))]
        {
            use crate::reader::cache::CacheConfig;

            let (primary_uri, over) = strip_cache_params(&primary_uri);
            let config = CacheConfig::from_env().merge(over);
            let primary = build_reader(&primary_uri)?;
            let cache = build_reader(cache_uri(&cache_scheme)?)?;
            return Ok(Box::new(crate::reader::CachingReader::with_config(
                primary,
                cache,
                primary_uri,
                cache_scheme,
                config,
            )));
        }
        #[cfg(not(any(feature = "duckdb", feature = "sqlite")))]
        {
            let _ = (&primary_uri, &cache_scheme);
            return Err(GgsqlError::ReaderError(
                "Caching layer requires the duckdb or sqlite feature".to_string(),
            ));
        }
    }
    let reader = build_reader(uri)?;
    auto_cache_if_needed(reader, uri)
}

/// True when the URI query string carries `cache=off`, opting out of the
/// automatic caching layer.
fn uri_disables_cache(uri: &str) -> bool {
    uri.split_once('?')
        .map(|(_, q)| q.split('&').any(|seg| seg.to_lowercase() == "cache=off"))
        .unwrap_or(false)
}

/// Probe whether a freshly connected reader can create the temporary tables
/// ggsql stages internal results in, using the dialect's own temp-table DDL
/// (the same mechanism the executor relies on). Best effort: the probe table
/// is dropped afterwards, and any failure means "cannot".
fn probe_temp_tables(reader: &dyn Reader) -> bool {
    let probe = format!("__ggsql_probe_{}__", crate::naming::session_id());
    let dialect = reader.dialect();
    let stmts = dialect.create_or_replace_temp_table_sql(&probe, &[], "SELECT 1 AS x");
    let ok = stmts.iter().all(|s| reader.execute_sql(s).is_ok());
    let _ = reader.execute_sql(&format!(
        "DROP TABLE IF EXISTS {}",
        dialect.quote_ident(&probe)
    ));
    ok
}

/// Wrap `reader` in an in-memory [`CachingReader`] when the backend cannot
/// host ggsql's internal tables itself: either the dialect requires it
/// outright (Trino, Druid, Drill) or a one-time temp-table probe
/// fails (e.g. a read-only account). Explicit cache selection (`<cache>+…`
/// or `--cache`) has already been handled by the caller and wins; `cache=off`
/// in the URI opts out.
///
/// `duckdb://memory` is the cache backend when the `duckdb` feature is
/// compiled in, `sqlite://:memory:` otherwise.
///
/// [`CachingReader`]: crate::reader::CachingReader
fn auto_cache_if_needed(
    reader: Box<dyn Reader + Send>,
    uri: &str,
) -> Result<Box<dyn Reader + Send>> {
    if uri_disables_cache(uri) {
        return Ok(reader);
    }
    let scheme = uri
        .split_once("://")
        .map(|(s, _)| s.to_ascii_lowercase())
        .unwrap_or_default();
    // The cache backends themselves never need a cache.
    if scheme == "duckdb" || scheme == "sqlite" {
        return Ok(reader);
    }
    let needed = reader.dialect().requires_cache() || !probe_temp_tables(&*reader);
    if !needed {
        return Ok(reader);
    }

    #[cfg(any(feature = "duckdb", feature = "sqlite"))]
    {
        use crate::reader::cache::CacheConfig;

        let (cache_uri, cache_scheme) = if cfg!(feature = "duckdb") {
            ("duckdb://memory", "duckdb")
        } else {
            ("sqlite://:memory:", "sqlite")
        };
        let cache = build_reader(cache_uri)?;
        Ok(Box::new(crate::reader::CachingReader::with_config(
            reader,
            cache,
            uri.to_string(),
            cache_scheme.to_string(),
            CacheConfig::from_env(),
        )))
    }
    #[cfg(not(any(feature = "duckdb", feature = "sqlite")))]
    {
        let _ = reader;
        Err(GgsqlError::ReaderError(format!(
            "Connection '{uri}' needs an in-memory cache to stage intermediate tables, \
             but this build has neither the duckdb nor the sqlite feature. \
             Add ?cache=off to the URI to proceed without one."
        )))
    }
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
    fn test_uri_disables_cache() {
        assert!(uri_disables_cache("postgres://u@h/db?cache=off"));
        assert!(uri_disables_cache("postgres://u@h/db?DSN=pg&CACHE=OFF"));
        assert!(!uri_disables_cache("postgres://u@h/db?DSN=pg"));
        assert!(!uri_disables_cache("postgres://u@h/db"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_auto_cache_wraps_when_probe_fails() {
        use crate::reader::duckdb::DuckDBReader;
        use crate::reader::test_support::ReadOnlyReader;

        // A read-only primary (temp-table probe fails) gets wrapped.
        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = auto_cache_if_needed(
            Box::new(ReadOnlyReader::new(Box::new(primary))),
            "postgres://u@h/db",
        )
        .unwrap();
        assert!(reader.caches_sources(), "expected a caching reader");

        // The wrapped reader runs the full pipeline: temp tables and stat
        // transforms land in the cache.
        let spec = reader
            .execute("SELECT 1.0 AS x, 2.0 AS y VISUALISE x, y DRAW point")
            .unwrap();
        assert_eq!(spec.metadata().rows, 1);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_auto_cache_skips_writable_and_opted_out() {
        use crate::reader::duckdb::DuckDBReader;
        use crate::reader::test_support::ReadOnlyReader;

        // A writable primary passes the probe and is used directly.
        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = auto_cache_if_needed(Box::new(primary), "postgres://u@h/db").unwrap();
        assert!(!reader.caches_sources(), "no cache expected");

        // cache=off wins even when the probe would fail.
        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = auto_cache_if_needed(
            Box::new(ReadOnlyReader::new(Box::new(primary))),
            "postgres://u@h/db?cache=off",
        )
        .unwrap();
        assert!(!reader.caches_sources(), "cache=off must be honored");

        // The cache backends themselves are never wrapped.
        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = auto_cache_if_needed(Box::new(primary), "duckdb://memory").unwrap();
        assert!(!reader.caches_sources(), "duckdb needs no cache");
    }

    #[test]
    fn test_requires_cache_dialects() {
        for scheme in ["trino", "druid", "drill"] {
            let d = crate::reader::dialects::dialect_for_scheme(scheme).unwrap();
            assert!(d.requires_cache(), "scheme {scheme} should require a cache");
        }
        for scheme in ["postgres", "duckdb", "sqlite", "clickhouse", "datafusion"] {
            let d = crate::reader::dialects::dialect_for_scheme(scheme).unwrap();
            assert!(!d.requires_cache(), "scheme {scheme} should be probed");
        }
    }

    #[test]
    fn test_build_reader_unsupported_scheme() {
        let err = build_reader("couchdb://localhost/db")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("Unsupported connection string"), "got: {err}");
    }

    #[test]
    fn test_build_reader_postgres_errors_informatively_without_drivers() {
        // With no ADBC driver installed and no ODBC hints, the error must
        // explain both paths rather than saying "not yet implemented".
        let err = build_reader("postgres://user@localhost:5432/db")
            .err()
            .unwrap()
            .to_string();
        assert!(!err.contains("not yet implemented"), "got: {err}");
        assert!(err.contains("ADBC") || err.contains("odbc"), "got: {err}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_from_dsn() {
        let conn = synthesize_odbc_conn_str(
            "postgres",
            "user:pw@dbhost:5432/sales?DSN=pg&sslmode=require",
        )
        .unwrap();
        assert!(conn.contains("DSN=pg"), "got: {conn}");
        assert!(conn.contains("UID=user"), "got: {conn}");
        assert!(conn.contains("PWD=pw"), "got: {conn}");
        assert!(conn.contains("sslmode=require"), "got: {conn}");
        // DSN form does not inject Server/Database (the DSN defines them).
        assert!(!conn.contains("Server="), "got: {conn}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_from_driver() {
        let conn = synthesize_odbc_conn_str("mysql", "u@dbhost:3306/shop?Driver={MySQL ODBC 9.0}")
            .unwrap();
        assert!(conn.contains("Driver={MySQL ODBC 9.0}"), "got: {conn}");
        assert!(conn.contains("Server=dbhost"), "got: {conn}");
        assert!(conn.contains("Port=3306"), "got: {conn}");
        assert!(conn.contains("Database=shop"), "got: {conn}");
        assert!(conn.contains("UID=u"), "got: {conn}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_none_without_hints() {
        assert!(synthesize_odbc_conn_str("postgres", "u@h/db").is_none());
    }

    #[test]
    fn test_split_cache_uri_rejects_reader_suffix_lookalike() {
        // `+odbc` is not a reader suffix — the `+` form is reserved for
        // cache composition, so `postgres+odbc` must not parse.
        assert_eq!(split_cache_uri("duckdb+postgres+odbc://u@h/db"), None);
        assert_eq!(split_cache_uri("a+b+c://x"), None);
    }

    #[cfg(feature = "adbc")]
    #[test]
    fn test_uri_forces_odbc() {
        assert!(uri_forces_odbc("u@h/db?reader=odbc"));
        assert!(uri_forces_odbc("u@h/db?DSN=pg&reader=odbc"));
        assert!(uri_forces_odbc("u@h/db?READER=ODBC"));
        assert!(uri_forces_odbc("u@h/db?Reader=Odbc"));
        assert!(!uri_forces_odbc("u@h/db?DSN=pg"));
        assert!(!uri_forces_odbc("u@h/db"));
    }

    #[test]
    fn test_unknown_scheme_still_rejected() {
        let err = build_reader("nosuchdb://localhost/db")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("Unsupported connection string"), "got: {err}");
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
    fn test_split_cache_uri_duckdb_cache_over_odbc() {
        assert_eq!(
            split_cache_uri("duckdb+odbc://Driver=Snowflake;Server=x"),
            Some((
                "odbc://Driver=Snowflake;Server=x".to_string(),
                "duckdb".to_string()
            ))
        );
    }

    #[test]
    fn test_split_cache_uri_duckdb_cache_over_sqlite_memory() {
        assert_eq!(
            split_cache_uri("duckdb+sqlite://memory"),
            Some(("sqlite://memory".to_string(), "duckdb".to_string()))
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
        assert_eq!(split_cache_uri("duckdb+a+b://x"), None);
    }

    #[test]
    fn test_split_cache_uri_rejects_empty_parts() {
        assert_eq!(split_cache_uri("+duckdb://x"), None);
        assert_eq!(split_cache_uri("odbc+://x"), None);
    }

    #[cfg(any(feature = "duckdb", feature = "sqlite"))]
    #[test]
    fn test_strip_cache_params_parses_known_keys() {
        let (uri, over) = strip_cache_params("duckdb://memory?cache_ttl=600");
        assert_eq!(uri, "duckdb://memory");
        assert_eq!(over.ttl_secs, Some(600));
        assert_eq!(over.max_bytes, None);
        assert_eq!(over.enabled, None);

        let (uri, over) =
            strip_cache_params("duckdb://memory?cache_max_bytes=256mb&cache_disabled=true");
        assert_eq!(uri, "duckdb://memory");
        assert_eq!(over.max_bytes, Some(256 * 1024 * 1024));
        assert_eq!(over.enabled, Some(false));
    }

    #[cfg(any(feature = "duckdb", feature = "sqlite"))]
    #[test]
    fn test_strip_cache_params_keeps_non_cache_segments() {
        // A non-cache `?key=` tail contributes no overrides and is left in place.
        let (uri, over) = strip_cache_params("odbc://DSN=foo?warehouse=PROD");
        assert_eq!(uri, "odbc://DSN=foo?warehouse=PROD");
        assert_eq!(over.ttl_secs, None);

        // ODBC body with `=`/`;` and no `?` is returned verbatim.
        let (uri, over) = strip_cache_params("odbc://Driver=Snowflake;Server=x");
        assert_eq!(uri, "odbc://Driver=Snowflake;Server=x");
        assert_eq!(over.enabled, None);

        // Cache keys are extracted; other params (e.g. ODBC settings) are kept.
        let (uri, over) =
            strip_cache_params("odbc://DSN=foo?ttl=99&cache_ttl=10&cache_max_bytes=8mb");
        assert_eq!(uri, "odbc://DSN=foo?ttl=99");
        assert_eq!(over.ttl_secs, Some(10));
        assert_eq!(over.max_bytes, Some(8 * 1024 * 1024));

        // When every param is a cache key, the `?` is dropped entirely.
        let (uri, over) = strip_cache_params("duckdb://memory?cache_ttl=10&cache_disabled=1");
        assert_eq!(uri, "duckdb://memory");
        assert_eq!(over.ttl_secs, Some(10));
        assert_eq!(over.enabled, Some(false));

        // Plain URI, no query string.
        let (uri, over) = strip_cache_params("duckdb://memory");
        assert_eq!(uri, "duckdb://memory");
        assert_eq!(over.ttl_secs, None);
    }

    #[cfg(all(feature = "duckdb", feature = "sqlite"))]
    #[test]
    fn test_reader_from_uri_applies_uri_cache_params() {
        // A composite URI with a cache-param tail builds a CachingReader
        assert!(
            reader_from_uri("duckdb+sqlite://memory?cache_ttl=600&cache_max_bytes=64mb").is_ok()
        );
    }
}
