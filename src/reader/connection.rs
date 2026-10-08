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

/// ggsql's own query-string parameters from a connection URI, lifted
/// out of a [`ConnUri`].
///
/// These are consumed during dispatch and must never reach a driver's own
/// URI parsing or option map — drivers reject unknown keys.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConnectParams {
    /// `cache=off`: opt out of the automatic caching layer entirely —
    /// different from `cache_disabled`, which keeps the caching reader but
    /// turns its result memo off.
    pub cache_off: bool,
    /// `reader=native|adbc|odbc`: force one reader kind over the default
    /// preference order (native → ADBC → ODBC).
    pub reader: Option<String>,
    /// Explicit SQL-dialect pin (`dialect=<scheme|ansi>`), bypassing backend
    /// detection — the escape hatch for backends ggsql doesn't recognise.
    pub dialect: Option<String>,
    /// Maximum age of a cache entry in seconds before it counts as a miss
    /// (`cache_ttl=<secs>`). Raw string; parsed by
    /// [`ConnUri::cache_config_override`].
    pub cache_ttl: Option<String>,
    /// Maximum total bytes of cached results before least-recently-used
    /// entries are evicted (`cache_max_bytes=<n|512mb|1gb|…>`). Raw string;
    /// parsed by [`ConnUri::cache_config_override`].
    pub cache_max_bytes: Option<String>,
    /// Whether the cache is disabled through the cache config
    /// (`cache_disabled=1|true|yes`) — the cache reader still wraps the
    /// connection but never serves from memory, unlike `cache_off`.
    pub cache_disabled: Option<bool>,
    /// Driver *statement* options from `stmt.<key>=<value>` params (prefix
    /// stripped), applied to every statement the reader creates.
    pub stmt_options: Vec<(String, String)>,
}

/// A connection URI parsed once: scheme, body, and query parameters with
/// ggsql's own keys lifted out into [`ConnectParams`].
///
/// Everything dispatched from a connection string — cache wrapping, reader
/// selection, ADBC driver setup, ODBC connection-string synthesis — reads
/// from this struct rather than re-splitting the raw URI.
#[derive(Debug, Clone)]
pub struct ConnUri {
    /// Lowercased scheme (`postgres`).
    pub scheme: String,
    /// Everything between `://` and `?` (`user:pass@host:5432/db`).
    pub body: String,
    /// Driver-bound query params in original order and case; the value is
    /// `None` for a bare `&flag&` segment.
    pub params: Vec<(String, Option<String>)>,
    /// ggsql's own parameters.
    pub ggsql: ConnectParams,
}

impl ConnUri {
    /// Parse `<scheme>://<body>[?k=v&…]`.
    pub fn parse(uri: &str) -> Result<Self> {
        let (scheme, rest) = uri
            .split_once("://")
            .ok_or_else(|| GgsqlError::ReaderError(format!("Invalid connection URI: {}", uri)))?;
        let (body, query) = rest.split_once('?').unwrap_or((rest, ""));

        let mut params = Vec::new();
        let mut ggsql = ConnectParams::default();
        for segment in query.split('&') {
            if segment.is_empty() {
                continue;
            }
            let (key, value) = match segment.split_once('=') {
                Some((k, v)) => (k, Some(v)),
                None => (segment, None),
            };
            match key.to_ascii_lowercase().as_str() {
                "cache" => {
                    ggsql.cache_off = value.is_some_and(|v| v.trim().eq_ignore_ascii_case("off"))
                }
                "cache_ttl" => ggsql.cache_ttl = value.map(|v| v.to_string()),
                "cache_max_bytes" => ggsql.cache_max_bytes = value.map(|v| v.to_string()),
                "cache_disabled" => {
                    ggsql.cache_disabled = value.map(|v| {
                        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes")
                    })
                }
                "reader" => ggsql.reader = value.map(|v| v.to_string()),
                "dialect" => ggsql.dialect = value.map(|v| v.to_string()),
                _ if key.starts_with("stmt.") => {
                    if let Some(v) = value {
                        ggsql
                            .stmt_options
                            .push((key["stmt.".len()..].to_string(), v.to_string()));
                    }
                }
                _ => params.push((key.to_string(), value.map(|v| v.to_string()))),
            }
        }

        Ok(Self {
            scheme: scheme.to_ascii_lowercase(),
            body: body.to_string(),
            params,
            ggsql,
        })
    }

    /// The driver-bound query string (`k=v&k2=v2`), ggsql keys excluded.
    pub fn query_string(&self) -> String {
        self.params
            .iter()
            .map(|(k, v)| match v {
                Some(v) => format!("{k}={v}"),
                None => k.clone(),
            })
            .collect::<Vec<_>>()
            .join("&")
    }

    /// The URI with ggsql-owned keys stripped — safe to hand to drivers and
    /// to use as a cache key.
    pub fn to_uri(&self) -> String {
        let query = self.query_string();
        if query.is_empty() {
            format!("{}://{}", self.scheme, self.body)
        } else {
            format!("{}://{}?{}", self.scheme, self.body, query)
        }
    }

    /// True when the URI opts out of the automatic caching layer
    /// (`cache=off`). This is one of two distinct "disable cache" knobs:
    /// `cache=off` never wraps the reader at all, while `cache_disabled=1`
    /// keeps the caching reader but disables its result memo
    /// ([`ConnectParams::cache_disabled`]).
    pub fn cache_disabled_off(&self) -> bool {
        self.ggsql.cache_off
    }

    /// True when the URI forces the ODBC reader path (`reader=odbc`).
    pub fn forces_odbc(&self) -> bool {
        self.ggsql
            .reader
            .as_deref()
            .is_some_and(|r| r.eq_ignore_ascii_case("odbc"))
    }

    /// Cache-config overrides from the URI's `cache_*` keys.
    pub fn cache_config_override(&self) -> crate::reader::cache::CacheConfigOverride {
        use crate::reader::cache::{parse_human_bytes, CacheConfigOverride};

        CacheConfigOverride {
            ttl_secs: self
                .ggsql
                .cache_ttl
                .as_deref()
                .and_then(|v| v.trim().parse::<u64>().ok()),
            max_bytes: self
                .ggsql
                .cache_max_bytes
                .as_deref()
                .and_then(parse_human_bytes),
            enabled: self.ggsql.cache_disabled.map(|disabled| !disabled),
        }
    }
}

/// Map a cache-backend scheme to its in-memory connection URI.
fn cache_uri(scheme: &str) -> Result<&'static str> {
    match scheme {
        "duckdb" => Ok("duckdb://memory"),
        "sqlite" => Ok("sqlite://:memory:"),
        "datafusion" => Ok("datafusion://"),
        _ => Err(GgsqlError::ReaderError(format!(
            "Unsupported cache backend '{}'. Supported: duckdb, sqlite, datafusion",
            scheme
        ))),
    }
}

/// Build a reader from a non-composite connection URI
pub fn build_reader(uri: &str) -> Result<Box<dyn Reader + Send>> {
    let conn = ConnUri::parse(uri)?;
    build_reader_parsed(&conn, uri)
}

/// Build a reader from a parsed URI.
///
/// `odbc://` and `adbc://` are *transports*, not databases — a raw ODBC
/// connection string or a named ADBC driver — so they are dispatched
/// directly. Everything else resolves through the [registry] into one
/// uniform preference order (see [`build_backend_reader`]).
///
/// [registry]: crate::reader::registry
fn build_reader_parsed(conn: &ConnUri, uri: &str) -> Result<Box<dyn Reader + Send>> {
    if conn.scheme == "odbc" {
        #[cfg(feature = "odbc")]
        {
            // Hand over the URI with ggsql's own keys stripped — a raw
            // `odbc://DSN=x?cache=off` would otherwise leak `?cache=off`
            // into the driver's connection string.
            return Ok(Box::new(crate::reader::OdbcReader::from_connection_string(
                &conn.to_uri(),
            )?));
        }
        #[cfg(not(feature = "odbc"))]
        {
            return Err(GgsqlError::ReaderError(
                "ODBC reader not compiled in. Rebuild with --features odbc".to_string(),
            ));
        }
    }
    if conn.scheme == "adbc" {
        return build_backend_reader(None, conn, uri);
    }
    if let Some(entry) = crate::reader::registry::by_scheme(&conn.scheme) {
        return build_backend_reader(Some(entry), conn, uri);
    }
    Err(GgsqlError::ReaderError(format!(
        "Unsupported connection string: {}. Supported: {}",
        uri,
        crate::reader::registry::supported_schemes(),
    )))
}

/// Build a reader for a registered database backend, trying each reader
/// kind in one preference order:
///
/// 1. the in-process native reader, when the backend has one and its cargo
///    feature is compiled in (duckdb, sqlite);
/// 2. ADBC, when a driver library is loadable;
/// 3. ODBC, when the URI or environment provides enough to synthesize a
///    connection string.
///
/// A `reader=native|adbc|odbc` query param forces one kind. Native readers
/// come first because they need no external driver; ADBC precedes ODBC
/// because its drivers are the better-supported path for most backends.
#[cfg_attr(not(all(feature = "adbc", feature = "odbc")), allow(unused_variables))]
fn build_backend_reader(
    entry: Option<&crate::reader::registry::DatabaseEntry>,
    conn: &ConnUri,
    uri: &str,
) -> Result<Box<dyn Reader + Send>> {
    use crate::reader::registry::NativeReader;

    let scheme = conn.scheme.as_str();
    let forced = conn.ggsql.reader.as_deref().map(str::to_ascii_lowercase);
    match forced.as_deref() {
        None | Some("native") | Some("adbc") | Some("odbc") => {}
        Some(other) => {
            return Err(GgsqlError::ReaderError(format!(
                "Unknown reader '{other}' in connection URI. Supported: native, adbc, odbc."
            )))
        }
    }
    let wants_native = forced.is_none() || forced.as_deref() == Some("native");
    let wants_adbc = forced.is_none() || forced.as_deref() == Some("adbc");

    // 1. Native in-process reader. These readers' grammar is just
    // `scheme://path` — anything past the body (ggsql params like
    // `cache=off`, driver-bound params) would be misread as part of the
    // file path, so only the body is passed on.
    if wants_native {
        let native_uri = || format!("{}://{}", conn.scheme, conn.body);
        match entry.and_then(|e| e.native_reader) {
            Some(NativeReader::DuckDb) => {
                #[cfg(feature = "duckdb")]
                {
                    return Ok(Box::new(
                        crate::reader::DuckDBReader::from_connection_string(&native_uri())?,
                    ));
                }
            }
            Some(NativeReader::Sqlite) => {
                #[cfg(feature = "sqlite")]
                {
                    return Ok(Box::new(
                        crate::reader::SqliteReader::from_connection_string(&native_uri())?,
                    ));
                }
            }
            None => {}
        }
    }
    // A forced native reader must not silently fall through to ADBC/ODBC.
    if forced.as_deref() == Some("native") {
        let note = match entry.and_then(|e| e.native_reader) {
            Some(_) => {
                "its native reader is not compiled in (rebuild with the matching cargo feature)"
            }
            None => "no native in-process reader exists for it (only duckdb and sqlite have one)",
        };
        return Err(GgsqlError::ReaderError(format!(
            "reader=native was requested for '{scheme}://' but {note}."
        )));
    }

    #[cfg(feature = "adbc")]
    if wants_adbc && (scheme == "adbc" || crate::reader::adbc::adbc_driver_available(scheme)) {
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
    if forced.as_deref() == Some("adbc") {
        return Err(GgsqlError::ReaderError(format!(
            "reader=adbc was requested but no loadable ADBC driver was found for '{scheme}://' \
             (searched ${}, ADBC driver paths, and system library paths).",
            crate::reader::registry::adbc_driver_env_var(scheme),
        )));
    }

    #[cfg(feature = "odbc")]
    if let Some(entry) = entry {
        if let Some(conn_str) = synthesize_odbc_conn_str(entry, conn) {
            // An explicit `dialect=` override wins; otherwise the scheme's
            // dialect. Flight SQL's registry dialect is generic ANSI, and
            // detection from the connected DBMS is strictly better, so
            // leave it unset there.
            let dialect = if let Some(name) = &conn.ggsql.dialect {
                Some(
                    crate::reader::registry::dialect_override(name)
                        .ok_or_else(|| crate::reader::registry::unknown_dialect_error(name))?,
                )
            } else if entry.scheme == "flightsql" {
                None
            } else {
                Some(entry.dialect())
            };
            let reader = crate::reader::OdbcReader::from_odbc_conn_str(&conn_str, dialect)?;
            return Ok(Box::new(reader));
        }
    }

    let native_hint = match entry.and_then(|e| e.native_reader) {
        Some(NativeReader::DuckDb) if !cfg!(feature = "duckdb") => {
            " The DuckDB reader is not compiled in (rebuild with --features duckdb), and"
        }
        Some(NativeReader::Sqlite) if !cfg!(feature = "sqlite") => {
            " The SQLite reader is not compiled in (rebuild with --features sqlite), and"
        }
        _ => "",
    };
    Err(GgsqlError::ReaderError(format!(
        "Could not connect for scheme '{}://'.{} no loadable ADBC driver was found \
         (searched ${}, ADBC driver paths, and system library paths), and no ODBC \
         fallback was available: provide DSN= or Driver= in the URI query \
         ({}://host/db?DSN=mydsn) or set {} to the name of an installed ODBC driver.",
        scheme,
        native_hint,
        crate::reader::registry::adbc_driver_env_var(scheme),
        scheme,
        crate::reader::registry::odbc_driver_env_var(scheme),
    )))
}

/// Parsed form of the `user:pass@host:port/db` authority in backend URIs.
#[cfg(feature = "odbc")]
struct BackendUri<'a> {
    user: Option<&'a str>,
    password: Option<&'a str>,
    host: Option<&'a str>,
    port: Option<&'a str>,
    database: Option<&'a str>,
}

#[cfg(feature = "odbc")]
fn parse_backend_uri(body: &str) -> BackendUri<'_> {
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
    }
}

/// Try to assemble an ODBC connection string for a backend URI.
///
/// Returns `Some` when the URI query carries `DSN=` or `Driver=`, or when
/// `GGSQL_<SCHEME>_ODBC_DRIVER` names an installed driver; otherwise `None`
/// so the caller can report that no fallback was configured.
#[cfg(feature = "odbc")]
fn synthesize_odbc_conn_str(
    entry: &crate::reader::registry::DatabaseEntry,
    conn: &ConnUri,
) -> Option<String> {
    use crate::reader::registry::odbc_driver_env_var;

    let parsed = parse_backend_uri(&conn.body);
    let param = |key: &str| {
        conn.params
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .and_then(|(_, v)| v.as_deref())
    };

    let dsn = param("dsn");
    let driver = param("driver").map(|d| d.trim_matches(|c| c == '{' || c == '}'));
    let env_driver = std::env::var(odbc_driver_env_var(entry.scheme)).ok();

    // On DBQ-style backends (Oracle/DB2) a DBQ= param fully specifies where
    // to connect, so Server/Port/Database synthesis must be suppressed —
    // Oracle ODBC rejects a connection string that mixes the two vocabularies.
    let has_dbq = entry.odbc_dbq_style && param("dbq").is_some();

    let mut parts: Vec<String> = Vec::new();
    let mut skip_keys: Vec<&str> = Vec::new();
    if let Some(dsn) = dsn {
        parts.push(format!("DSN={}", dsn));
        skip_keys.push("dsn");
    } else {
        let driver = driver.or(env_driver.as_deref())?;
        parts.push(format!("Driver={{{}}}", driver));
        skip_keys.push("driver");
        if !has_dbq {
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
    }
    if let Some(user) = parsed.user {
        parts.push(format!("UID={}", user));
    }
    if let Some(password) = parsed.password {
        parts.push(format!("PWD={}", password));
    }
    for (key, value) in &conn.params {
        if skip_keys.contains(&key.to_ascii_lowercase().as_str()) {
            continue;
        }
        match value {
            Some(v) => parts.push(format!("{key}={v}")),
            None => parts.push(key.clone()),
        }
    }
    Some(parts.join(";"))
}

/// Construct a reader from a connection URI, wrapping it in a [`CachingReader`]
/// when the URI uses the composite `<cache>+<primary>://` form.
///
/// [`CachingReader`]: crate::reader::CachingReader
pub fn reader_from_uri(uri: &str) -> Result<Box<dyn Reader + Send>> {
    if let Some((primary_uri, cache_scheme)) = split_cache_uri(uri) {
        // DuckDB/SQLite caches need their cargo feature; the datafusion cache
        // comes through ADBC and works without either.
        #[cfg(not(any(feature = "duckdb", feature = "sqlite")))]
        if cache_scheme != "datafusion" {
            let _ = &primary_uri;
            return Err(GgsqlError::ReaderError(
                "Caching layer requires the duckdb or sqlite feature".to_string(),
            ));
        }

        use crate::reader::cache::CacheConfig;

        let conn = ConnUri::parse(&primary_uri)?;
        let config = CacheConfig::from_env().merge(conn.cache_config_override());
        let primary_uri = conn.to_uri();
        let primary = build_reader_parsed(&conn, &primary_uri)?;
        let cache = build_reader(cache_uri(&cache_scheme)?)?;
        return Ok(Box::new(crate::reader::CachingReader::with_config(
            primary,
            cache,
            primary_uri,
            cache_scheme,
            config,
        )));
    }
    let conn = ConnUri::parse(uri)?;
    let reader = build_reader_parsed(&conn, uri)?;
    auto_cache_if_needed(reader, &conn)
}

/// Probe whether a freshly connected reader can create the temporary tables
/// ggsql stages internal results in, using the dialect's own temp-table DDL
/// (the same mechanism the executor relies on). Best effort: the probe table
/// is dropped afterwards, and any failure means "cannot".
///
/// Note this runs DDL (CREATE + DROP of a `__ggsql_probe_<sid>__` table)
/// against the user's database on every connection; on backends without
/// temp tables (e.g. Oracle) that is a permanent table for the duration of
/// the probe. The drop goes through the dialect so backends without
/// `DROP TABLE IF EXISTS` (Oracle) get their guarded form.
fn probe_temp_tables(reader: &dyn Reader) -> bool {
    let probe = format!("__ggsql_probe_{}__", crate::naming::session_id());
    let dialect = reader.dialect();
    let stmts = dialect.create_or_replace_temp_table_sql(&probe, &[], "SELECT 1 AS x");
    let ok = stmts.iter().all(|s| reader.execute_sql(s).is_ok());
    let _ = reader.execute_sql(&dialect.drop_table_sql(&probe));
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
    conn: &ConnUri,
) -> Result<Box<dyn Reader + Send>> {
    if conn.cache_disabled_off() {
        return Ok(reader);
    }
    if conn.scheme == "duckdb" || conn.scheme == "sqlite" {
        return Ok(reader);
    }
    let needed = reader.dialect().requires_cache() || !probe_temp_tables(&*reader);
    if !needed {
        return Ok(reader);
    }

    #[cfg(any(feature = "duckdb", feature = "sqlite"))]
    {
        use crate::reader::cache::CacheConfig;

        let cache_scheme = if cfg!(feature = "duckdb") {
            "duckdb"
        } else {
            "sqlite"
        };
        let cache = build_reader(cache_uri(cache_scheme)?)?;
        Ok(Box::new(crate::reader::CachingReader::with_config(
            reader,
            cache,
            conn.to_uri(),
            cache_scheme.to_string(),
            CacheConfig::from_env(),
        )))
    }
    #[cfg(not(any(feature = "duckdb", feature = "sqlite")))]
    {
        let _ = reader;
        Err(GgsqlError::ReaderError(format!(
            "Connection '{}' needs an in-memory cache to stage intermediate tables, \
             but this build has neither the duckdb nor the sqlite feature. \
             Add ?cache=off to the URI to proceed without one.",
            conn.to_uri()
        )))
    }
}

/// Lift ggsql-owned keys out of an ODBC connection string, returning the
/// remaining string and the overrides.
///
/// This is the `;`-separated sibling of [`ConnUri::parse`]: ODBC conn
/// strings (`odbc://Driver=X;DSN=foo`) carry `k=v` pairs in their body
/// rather than a `?` query, so ggsql's own keys need a separate extraction
/// — but the knowledge of *which* keys are ggsql's lives here, alongside
/// `ConnUri`, not in the ODBC reader. Only keys meaningful inside a conn
/// string are recognized; URI-level keys (`cache`, `reader`) are parsed
/// from the `odbc://…?` query by `ConnUri` as usual.
pub(crate) fn take_odbc_ggsql_params(conn_str: &str) -> (String, ConnectParams) {
    let mut kept = Vec::new();
    let mut ggsql = ConnectParams::default();
    for segment in conn_str.split(';') {
        match segment.split_once('=') {
            Some((key, value)) if key.trim().eq_ignore_ascii_case("dialect") => {
                ggsql.dialect = Some(value.trim().to_string());
            }
            _ => kept.push(segment),
        }
    }
    (kept.join(";"), ggsql)
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
    fn test_take_odbc_ggsql_params() {
        let (conn, ggsql) = take_odbc_ggsql_params("Driver=X;Server=h;Dialect=ansi;UID=u");
        assert_eq!(conn, "Driver=X;Server=h;UID=u");
        assert_eq!(ggsql.dialect.as_deref(), Some("ansi"));

        let (conn, ggsql) = take_odbc_ggsql_params("dialect=postgres;DSN=x");
        assert_eq!(conn, "DSN=x");
        assert_eq!(ggsql.dialect.as_deref(), Some("postgres"));

        let (conn, ggsql) = take_odbc_ggsql_params("Driver=X;Server=h");
        assert_eq!(conn, "Driver=X;Server=h");
        assert_eq!(ggsql.dialect, None);
    }

    #[test]
    fn test_conn_uri_cache_off() {
        assert!(ConnUri::parse("postgres://u@h/db?cache=off")
            .unwrap()
            .cache_disabled_off());
        assert!(ConnUri::parse("postgres://u@h/db?DSN=pg&CACHE=OFF")
            .unwrap()
            .cache_disabled_off());
        assert!(!ConnUri::parse("postgres://u@h/db?DSN=pg")
            .unwrap()
            .cache_disabled_off());
        assert!(!ConnUri::parse("postgres://u@h/db")
            .unwrap()
            .cache_disabled_off());
    }

    #[test]
    fn test_conn_uri_strips_ggsql_params() {
        // ggsql-owned keys never reach a driver; everything else is kept
        // in order and original case.
        let conn = ConnUri::parse(
            "postgres://u@h/db?cache=off&user=x&cache_ttl=60&reader=odbc&cache_max_bytes=1MB",
        )
        .unwrap();
        assert_eq!(conn.query_string(), "user=x");
        assert!(conn.cache_disabled_off());
        assert!(conn.forces_odbc());
        assert_eq!(conn.ggsql.cache_ttl.as_deref(), Some("60"));

        let conn = ConnUri::parse("postgres://h/db?CACHE=OFF&Driver={PostgreSQL Unicode}&cache_disabled=1&dialect=ansi&stmt.x=y")
            .unwrap();
        assert_eq!(conn.query_string(), "Driver={PostgreSQL Unicode}");
        assert_eq!(conn.ggsql.cache_disabled, Some(true));
        assert_eq!(conn.ggsql.dialect.as_deref(), Some("ansi"));
        assert_eq!(
            conn.ggsql.stmt_options,
            vec![("x".to_string(), "y".to_string())]
        );

        let conn = ConnUri::parse("postgres://h/db?user=x&password=y&flag").unwrap();
        assert_eq!(conn.query_string(), "user=x&password=y&flag");
        assert_eq!(conn.to_uri(), "postgres://h/db?user=x&password=y&flag");

        let conn = ConnUri::parse("duckdb://memory").unwrap();
        assert_eq!(conn.to_uri(), "duckdb://memory");
        assert_eq!(
            ConnUri::parse("postgres://h/db?cache=off")
                .unwrap()
                .to_uri(),
            "postgres://h/db"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_auto_cache_wraps_when_probe_fails() {
        use crate::reader::duckdb::DuckDBReader;
        use crate::reader::test_support::ReadOnlyReader;

        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let conn = ConnUri::parse("postgres://u@h/db").unwrap();
        let reader =
            auto_cache_if_needed(Box::new(ReadOnlyReader::new(Box::new(primary))), &conn).unwrap();
        assert!(reader.caches_sources(), "expected a caching reader");

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

        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let conn = ConnUri::parse("postgres://u@h/db").unwrap();
        let reader = auto_cache_if_needed(Box::new(primary), &conn).unwrap();
        assert!(!reader.caches_sources(), "no cache expected");

        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let conn = ConnUri::parse("postgres://u@h/db?cache=off").unwrap();
        let reader =
            auto_cache_if_needed(Box::new(ReadOnlyReader::new(Box::new(primary))), &conn).unwrap();
        assert!(!reader.caches_sources(), "cache=off must be honored");

        let primary = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let conn = ConnUri::parse("duckdb://memory").unwrap();
        let reader = auto_cache_if_needed(Box::new(primary), &conn).unwrap();
        assert!(!reader.caches_sources(), "duckdb needs no cache");
    }

    #[test]
    fn test_requires_cache_dialects() {
        let dialect_for = |scheme: &str| {
            crate::reader::registry::by_scheme(scheme)
                .map(|e| e.dialect())
                .unwrap()
        };
        for scheme in ["trino", "druid", "drill"] {
            assert!(
                dialect_for(scheme).requires_cache(),
                "scheme {scheme} should require a cache"
            );
        }
        for scheme in ["postgres", "duckdb", "sqlite", "clickhouse", "datafusion"] {
            assert!(
                !dialect_for(scheme).requires_cache(),
                "scheme {scheme} should be probed"
            );
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
    fn test_unknown_reader_param_errors() {
        let err = build_reader("postgres://u@h/db?reader=jdbc")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("Unknown reader 'jdbc'"), "got: {err}");
    }

    #[cfg(feature = "adbc")]
    #[test]
    fn test_forced_adbc_errors_when_driver_unavailable() {
        // With reader=adbc, an unloadable driver is an explicit error rather
        // than a silent fallthrough to the ODBC/native paths. (Relies on no
        // MonetDB ADBC driver existing — the registry has none.)
        let err = build_reader("monetdb://u@h/db?reader=adbc")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("reader=adbc was requested"), "got: {err}");
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_native_reader_still_preferred_for_duckdb() {
        assert!(build_reader("duckdb://memory").is_ok());
        assert!(build_reader("duckdb://memory?reader=native").is_ok());
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_native_reader_drops_query_params_from_path() {
        // Regression: the native readers treat everything after `://` as a
        // file path, so a URI carrying query params must be reduced to
        // `scheme://body` before dispatch — otherwise DuckDB opens an
        // on-disk database literally named e.g. `memory?cache=off`.
        let leaked = "memory?cache=off&reader=native";
        let _ = std::fs::remove_file(leaked); // clean slate; ignore absence
        assert!(build_reader("duckdb://memory?cache=off&reader=native").is_ok());
        assert!(
            !std::path::Path::new(leaked).exists(),
            "query params leaked into the DuckDB file path"
        );
    }

    #[test]
    fn test_build_reader_postgres_errors_informatively_without_drivers() {
        let err = build_reader("postgres://user@localhost:5432/db")
            .err()
            .unwrap()
            .to_string();
        assert!(!err.contains("not yet implemented"), "got: {err}");
        assert!(err.contains("ADBC") || err.contains("odbc"), "got: {err}");
    }

    #[cfg(feature = "odbc")]
    fn synthesize(scheme: &str, uri_body_query: &str) -> Option<String> {
        let entry = crate::reader::registry::by_scheme(scheme).unwrap();
        let conn = ConnUri::parse(&format!("{scheme}://{uri_body_query}")).unwrap();
        synthesize_odbc_conn_str(entry, &conn)
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_from_dsn() {
        let conn = synthesize(
            "postgres",
            "user:pw@dbhost:5432/sales?DSN=pg&sslmode=require",
        )
        .unwrap();
        assert!(conn.contains("DSN=pg"), "got: {conn}");
        assert!(conn.contains("UID=user"), "got: {conn}");
        assert!(conn.contains("PWD=pw"), "got: {conn}");
        assert!(conn.contains("sslmode=require"), "got: {conn}");
        assert!(!conn.contains("Server="), "got: {conn}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_from_driver() {
        let conn = synthesize("mysql", "u@dbhost:3306/shop?Driver={MySQL ODBC 9.0}").unwrap();
        assert!(conn.contains("Driver={MySQL ODBC 9.0}"), "got: {conn}");
        assert!(conn.contains("Server=dbhost"), "got: {conn}");
        assert!(conn.contains("Port=3306"), "got: {conn}");
        assert!(conn.contains("Database=shop"), "got: {conn}");
        assert!(conn.contains("UID=u"), "got: {conn}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_dbq_suppresses_server_synthesis() {
        let conn = synthesize(
            "oracle",
            "ggsql:pw@localhost:1521/XEPDB1?Driver={Oracle}&DBQ=(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=XEPDB1)))",
        )
        .unwrap();
        assert!(conn.contains("Driver={Oracle}"), "got: {conn}");
        assert!(conn.contains("DBQ=(DESCRIPTION="), "got: {conn}");
        assert!(conn.contains("UID=ggsql"), "got: {conn}");
        assert!(!conn.contains("Server="), "got: {conn}");
        assert!(!conn.contains("Port="), "got: {conn}");
        assert!(!conn.contains("Database="), "got: {conn}");
    }

    #[cfg(feature = "odbc")]
    #[test]
    fn test_synthesize_odbc_conn_str_none_without_hints() {
        assert!(synthesize("postgres", "u@h/db").is_none());
    }

    #[test]
    fn test_split_cache_uri_rejects_reader_suffix_lookalike() {
        // `+odbc` is not a reader suffix — the `+` form is reserved for
        // cache composition, so `postgres+odbc` must not parse.
        assert_eq!(split_cache_uri("duckdb+postgres+odbc://u@h/db"), None);
        assert_eq!(split_cache_uri("a+b+c://x"), None);
    }

    #[test]
    fn test_conn_uri_forces_odbc() {
        let forces = |uri: &str| ConnUri::parse(uri).unwrap().forces_odbc();
        assert!(forces("postgres://u@h/db?reader=odbc"));
        assert!(forces("postgres://u@h/db?DSN=pg&reader=odbc"));
        assert!(forces("postgres://u@h/db?READER=ODBC"));
        assert!(forces("postgres://u@h/db?Reader=Odbc"));
        assert!(!forces("postgres://u@h/db?DSN=pg"));
        assert!(!forces("postgres://u@h/db"));
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
    fn test_conn_uri_cache_config_override() {
        let conn = ConnUri::parse("duckdb://memory?cache_ttl=600").unwrap();
        let over = conn.cache_config_override();
        assert_eq!(conn.to_uri(), "duckdb://memory");
        assert_eq!(over.ttl_secs, Some(600));
        assert_eq!(over.max_bytes, None);
        assert_eq!(over.enabled, None);

        let conn =
            ConnUri::parse("duckdb://memory?cache_max_bytes=256mb&cache_disabled=true").unwrap();
        let over = conn.cache_config_override();
        assert_eq!(conn.to_uri(), "duckdb://memory");
        assert_eq!(over.max_bytes, Some(256 * 1024 * 1024));
        assert_eq!(over.enabled, Some(false));

        let conn =
            ConnUri::parse("odbc://DSN=foo?ttl=99&cache_ttl=10&cache_max_bytes=8mb").unwrap();
        let over = conn.cache_config_override();
        assert_eq!(conn.to_uri(), "odbc://DSN=foo?ttl=99");
        assert_eq!(over.ttl_secs, Some(10));
        assert_eq!(over.max_bytes, Some(8 * 1024 * 1024));
    }

    #[test]
    fn test_forced_native_reader_errors_when_unavailable() {
        // postgres has no native in-process reader; reader=native must error
        // explicitly rather than silently falling through to ADBC/ODBC.
        let err = build_reader("postgres://u:p@localhost/db?reader=native")
            .err()
            .expect("reader=native without a native reader must fail");
        assert!(
            err.to_string().contains("reader=native"),
            "unexpected error: {err}"
        );
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
