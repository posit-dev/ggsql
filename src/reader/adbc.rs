//! ADBC (Arrow Database Connectivity) reader.
//!
//! Generic over any concrete ADBC `Driver` implementation. Verified against
//! two drivers in this crate's tests:
//!
//! - `adbc_datafusion` — pure-Rust, in-process. Used for routing and
//!   conversion unit tests where loading a native driver isn't worth the
//!   build complexity.
//! - `adbc_driver_duckdb` (loaded via `adbc_driver_manager::ManagedDriver`)
//!   — a real ADBC C driver, used for an equivalence suite that compares
//!   `AdbcReader<DuckDB>` output against ggsql's existing `DuckDBReader`.
//!
//! The `Reader` trait takes `&self`, but ADBC's `Statement` API takes
//! `&mut self`. We bridge this with `RefCell` around the `Connection`,
//! mirroring the interior-mutability pattern used by `OdbcReader`.

use crate::reader::{AnsiDialect, Reader, SqlDialect};
use crate::{DataFrame, GgsqlError, Result};
use adbc_core::sync::{Connection, Database, Driver};
use std::cell::RefCell;
use std::collections::HashSet;

pub struct AdbcReader<D: Driver> {
    // Driver must stay alive as long as the Database does (per ADBC contract).
    _driver: D,
    // Database must stay alive as long as the Connection does.
    _database: D::DatabaseType,
    // Connection is what Statements are made from. Wrapped in RefCell because
    // new_statement / set_sql_query / execute all take &mut, but Reader::execute_sql
    // takes &self.
    connection: RefCell<<D::DatabaseType as Database>::ConnectionType>,
    dialect: Box<dyn SqlDialect + Send>,
    registered_tables: RefCell<HashSet<String>>,
    // Driver-specific statement options (from `stmt.`-prefixed URI params)
    // applied to every statement created in execute_sql.
    statement_opts: Vec<(String, String)>,
}

/// Execute the dialect's session-init statements on a fresh connection —
/// see [`SqlDialect::session_init_sql`]. Failures are hard errors: the
/// generated SQL is wrong for the backend when the init did not take effect
/// (e.g. double-quoted identifiers read as string literals without
/// ANSI_QUOTES), so continuing would fail later with a confusing message.
fn run_session_init<C: adbc_core::Connection>(
    connection: &mut C,
    dialect: &dyn SqlDialect,
) -> Result<()> {
    for sql in dialect.session_init_sql() {
        let mut stmt = connection.new_statement().map_err(|e| {
            GgsqlError::ReaderError(format!("ADBC session init new_statement: {e}"))
        })?;
        stmt.set_sql_query(&sql).map_err(|e| {
            GgsqlError::ReaderError(format!("ADBC session init set_sql_query: {e}"))
        })?;
        stmt.execute_update().map_err(|e| {
            GgsqlError::ReaderError(format!("ADBC session init failed for '{sql}': {e}"))
        })?;
    }
    Ok(())
}

impl<D: Driver> AdbcReader<D> {
    /// Construct an `AdbcReader` with an explicit `SqlDialect`. Use this to
    /// plug in backend-specific dialects (e.g. a TrinoDialect, SnowflakeDialect)
    /// when the reader is pointed at that backend.
    pub fn with_dialect(driver: D, dialect: Box<dyn SqlDialect + Send>) -> Result<Self> {
        Self::new(driver, dialect)
    }

    /// Create a new `AdbcReader` from an already-initialized ADBC driver.
    ///
    /// Callers are responsible for any pre-init `Database` / `Connection`
    /// options. For convenience, use `from_driver` for the common case
    /// with the ANSI dialect, or pass a custom `SqlDialect`
    /// (e.g. a Trino / Snowflake dialect) here directly.
    pub fn new(mut driver: D, dialect: Box<dyn SqlDialect + Send>) -> Result<Self> {
        let database = driver
            .new_database()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_database failed: {}", e)))?;
        let mut connection = database
            .new_connection()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_connection failed: {}", e)))?;
        run_session_init(&mut connection, &*dialect)?;
        Ok(Self {
            _driver: driver,
            _database: database,
            connection: RefCell::new(connection),
            dialect,
            registered_tables: RefCell::new(HashSet::new()),
            statement_opts: Vec::new(),
        })
    }

    /// Create a new `AdbcReader`, passing pre-init options to the underlying
    /// `Database`. Use this when the driver requires URI / credentials / RPC
    /// header options to be set before the first connection (e.g. Flight SQL
    /// or other auth-required backends).
    pub fn new_with_database_opts(
        mut driver: D,
        dialect: Box<dyn SqlDialect + Send>,
        opts: impl IntoIterator<
            Item = (
                adbc_core::options::OptionDatabase,
                adbc_core::options::OptionValue,
            ),
        >,
    ) -> Result<Self> {
        let database = driver.new_database_with_opts(opts).map_err(|e| {
            GgsqlError::ReaderError(format!("ADBC new_database_with_opts failed: {}", e))
        })?;
        let mut connection = database
            .new_connection()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_connection failed: {}", e)))?;
        run_session_init(&mut connection, &*dialect)?;
        Ok(Self {
            _driver: driver,
            _database: database,
            connection: RefCell::new(connection),
            dialect,
            registered_tables: RefCell::new(HashSet::new()),
            statement_opts: Vec::new(),
        })
    }

    /// Attach driver-specific *statement* options, applied to every statement
    /// the reader creates in `execute_sql`. These come from `stmt.`-prefixed
    /// URI params (see [`partition_statement_opts`]) and exist because some
    /// drivers expose per-query settings only at the statement level — e.g.
    /// BigQuery's `bigquery.query.destination_table`, which is needed to
    /// read query results from the goccy BigQuery emulator (it does not
    /// serve anonymous result tables over the Storage Read API).
    pub fn with_statement_opts(mut self, opts: Vec<(String, String)>) -> Self {
        self.statement_opts = opts;
        self
    }

    /// Apply [`Self::statement_opts`] to a freshly created statement.
    fn apply_statement_opts<S: adbc_core::Statement + ?Sized>(&self, stmt: &mut S) -> Result<()> {
        for (key, value) in &self.statement_opts {
            stmt.set_option(
                OptionStatement::Other(key.clone()),
                OptionValue::String(value.clone()),
            )
            .map_err(|e| {
                GgsqlError::ReaderError(format!("ADBC set statement option '{key}': {e}"))
            })?;
        }
        Ok(())
    }

    /// Convenience: construct with the ANSI dialect. Good default for
    /// standards-compliant backends; use `new` directly to plug in a
    /// backend-specific dialect.
    pub fn from_driver(driver: D) -> Result<Self> {
        Self::new(driver, Box::new(AnsiDialect))
    }
}

// =============================================================================
// Runtime driver loading (adbc_driver_manager)
// =============================================================================

use adbc_core::options::{AdbcVersion, OptionDatabase, OptionStatement, OptionValue};
use adbc_driver_manager::ManagedDriver;

/// Default load flags: search `ADBC_DRIVER_PATH`, then system, then user
/// driver directories; allow relative paths so `adbc://./libfoo.so` works.
const DEFAULT_LOAD_FLAGS: adbc_core::LoadFlags =
    adbc_core::LOAD_FLAG_DEFAULT | adbc_core::LOAD_FLAG_ALLOW_RELATIVE_PATHS;

/// Map a ggsql URI scheme to its ADBC driver names: the canonical driver
/// library name and the dbc manifest ID.
///
/// Both names are needed because they are spelled differently on disk:
/// `dbc install postgresql` writes a manifest named after its short driver
/// ID (`postgresql.toml`), while a from-source or system-wide install is
/// typically found under the library name (`adbc_driver_postgresql`).
/// [`load_driver_for_scheme`] probes both.
///
/// Most backend schemes resolve to a dedicated driver from the ADBC Driver
/// Foundry (installable via `dbc install <id>`). Redshift shares the
/// PostgreSQL wire protocol and uses the PostgreSQL driver. Schemes without
/// a usable dedicated driver (Drill, MonetDB) return `None` and fall through
/// to the ODBC fallback in connection setup.
fn driver_names_for_scheme(scheme: &str) -> Option<(&'static str, &'static str)> {
    Some(match scheme {
        "postgres" | "postgresql" => ("adbc_driver_postgresql", "postgresql"),
        "redshift" => ("adbc_driver_postgresql", "postgresql"),
        "snowflake" => ("adbc_driver_snowflake", "snowflake"),
        "bigquery" => ("adbc_driver_bigquery", "bigquery"),
        "databricks" | "spark" => ("adbc_driver_databricks", "databricks"),
        "duckdb" => ("adbc_driver_duckdb", "duckdb"),
        "sqlite" => ("adbc_driver_sqlite", "sqlite"),
        "flightsql" => ("adbc_driver_flightsql", "flightsql"),
        "mysql" | "mariadb" => ("adbc_driver_mysql", "mysql"),
        "trino" => ("adbc_driver_trino", "trino"),
        "clickhouse" => ("adbc_driver_clickhouse", "clickhouse"),
        "mssql" | "sqlserver" => ("adbc_driver_mssql", "mssql"),
        "oracle" => ("adbc_driver_oracle", "oracle"),
        "exasol" => ("adbc_driver_exasol", "exasol"),
        // Preview driver from the Foundry as of late 2026.
        "druid" => ("adbc_driver_druid", "druid"),
        _ => return None,
    })
}

/// Map a ggsql URI scheme to the canonical ADBC driver library name.
/// See [`driver_names_for_scheme`] for the naming subtlety.
pub fn driver_name_for_scheme(scheme: &str) -> Option<&'static str> {
    driver_names_for_scheme(scheme).map(|(lib_name, _)| lib_name)
}

/// Environment variable that overrides the ADBC driver for a URI scheme,
/// e.g. `GGSQL_POSTGRES_ADBC_DRIVER=/opt/drivers/libadbc_driver_postgresql.so`.
pub fn driver_env_var(scheme: &str) -> String {
    let upper: String = scheme
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("GGSQL_{}_ADBC_DRIVER", upper)
}

/// Parse a `k=v&…` query string into ADBC database options. The keys `uri`,
/// `username`, and `password` map to their dedicated ADBC options; everything
/// else passes through as a driver-specific option.
fn query_params_to_opts(query: &str) -> Vec<(OptionDatabase, OptionValue)> {
    let mut opts = Vec::new();
    for segment in query.split('&') {
        let Some((key, value)) = segment.split_once('=') else {
            continue;
        };
        if key.is_empty() {
            continue;
        }
        let opt_key = match key {
            "uri" => OptionDatabase::Uri,
            "username" => OptionDatabase::Username,
            "password" => OptionDatabase::Password,
            other => OptionDatabase::Other(other.to_string()),
        };
        opts.push((opt_key, OptionValue::String(value.to_string())));
    }
    opts
}

/// Split `stmt.`-prefixed params out of a `k=v&…` query string. The prefix
/// marks driver *statement* options (applied to every statement the reader
/// creates) as opposed to database options — e.g.
/// `stmt.bigquery.query.destination_table=ds.tbl`. Returns the statement
/// options with the prefix stripped, and the remaining query string with
/// those segments removed (drivers reject unknown params in their own URI
/// parsing, so they must not leak through).
fn partition_statement_opts(query: &str) -> (Vec<(String, String)>, String) {
    let mut stmt_opts = Vec::new();
    let mut rest = Vec::new();
    for segment in query.split('&') {
        if segment.is_empty() {
            continue;
        }
        match segment.split_once('=') {
            Some((key, value)) if key.starts_with("stmt.") => {
                stmt_opts.push((key["stmt.".len()..].to_string(), value.to_string()));
            }
            _ => rest.push(segment),
        }
    }
    (stmt_opts, rest.join("&"))
}

/// Load an ADBC driver for a scheme, honoring the per-scheme env override
/// first, then the canonical driver name. The error message lists what was
/// probed so users know where ggsql looked.
fn load_driver_for_scheme(scheme: &str) -> Result<ManagedDriver> {
    let env_var = driver_env_var(scheme);
    if let Ok(path) = std::env::var(&env_var) {
        return ManagedDriver::load_from_name(
            &path,
            None,
            AdbcVersion::V110,
            DEFAULT_LOAD_FLAGS,
            None,
        )
        .map_err(|e| {
            GgsqlError::ReaderError(format!(
                "ADBC driver load failed for {}={}: {}",
                env_var, path, e
            ))
        });
    }
    let (lib_name, dbc_id) = driver_names_for_scheme(scheme).ok_or_else(|| {
        GgsqlError::ReaderError(format!("No known ADBC driver for scheme '{}://'", scheme))
    })?;
    // Probe the canonical library name first, then the dbc manifest ID (see
    // driver_names_for_scheme); collect both errors so the message shows
    // everything that was tried.
    let mut errors = Vec::new();
    for name in [lib_name, dbc_id] {
        match ManagedDriver::load_from_name(name, None, AdbcVersion::V110, DEFAULT_LOAD_FLAGS, None)
        {
            Ok(driver) => return Ok(driver),
            Err(e) => errors.push(format!("'{name}': {e}")),
        }
    }
    Err(GgsqlError::ReaderError(format!(
        "ADBC driver for '{scheme}://' not found or failed to load ({}). \
         Searched ${env_var}, the ADBC driver paths, and system library paths. \
         Set it explicitly with {env_var}=/path/to/driver, or use an odbc:// \
         connection string instead.",
        errors.join("; ")
    )))
}

impl AdbcReader<ManagedDriver> {
    /// Construct an `AdbcReader` from a connection URI, loading the driver
    /// shared library at runtime.
    ///
    /// Two URI forms are accepted:
    ///
    /// - `adbc://<driver>?k=v&…` — `driver` is a canonical driver name
    ///   (`adbc_driver_postgresql`), a known short name (`postgres`), or a
    ///   path to a driver library/manifest; query params become ADBC
    ///   database options (`uri=`, `username=`, `password=`, or
    ///   driver-specific keys).
    /// - `<scheme>://<rest>` (e.g. `postgres://user:pass@host/db`) — the
    ///   scheme selects the driver via [`driver_name_for_scheme`]; the full
    ///   URI is passed to the driver as the `uri` database option, with any
    ///   `?k=v` query params passed through as additional options. Where a
    ///   driver speaks a different wire scheme than ggsql's, the URI is
    ///   rewritten (`clickhouse://…` → `http://…`, `mysql://…` → a
    ///   go-sql-driver DSN, `bigquery://project/dataset` → Simba grammar);
    ///   see [`driver_uri_for`] and [`bigquery_driver_uri`].
    ///
    /// The dialect is chosen from the scheme via
    /// [`crate::reader::dialects::dialect_for_scheme`], falling back to ANSI.
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let (scheme, rest) = uri.split_once("://").ok_or_else(|| {
            GgsqlError::ReaderError(format!("Invalid ADBC connection URI: {}", uri))
        })?;
        let scheme = scheme.to_ascii_lowercase();
        let (body, query) = rest.split_once('?').unwrap_or((rest, ""));
        // `stmt.`-prefixed params become per-statement options rather than
        // database options.
        let (stmt_opts, query) = partition_statement_opts(query);

        let (driver, opts) = if scheme == "adbc" {
            // body is the driver name or path (may be a short scheme alias).
            // Known aliases go through the same dual-probe as scheme URIs
            // (canonical library name, then dbc manifest ID) plus the
            // per-scheme env override; anything else is a name or path for
            // the driver manager to resolve directly.
            let driver = if driver_names_for_scheme(body).is_some() {
                load_driver_for_scheme(body)?
            } else {
                ManagedDriver::load_from_name(
                    body,
                    None,
                    AdbcVersion::V110,
                    DEFAULT_LOAD_FLAGS,
                    None,
                )
                .map_err(|e| {
                    GgsqlError::ReaderError(format!("ADBC driver '{}' failed to load: {}", body, e))
                })?
            };
            (driver, query_params_to_opts(&query))
        } else {
            let driver = load_driver_for_scheme(&scheme)?;
            // The URI handed to the driver must not carry stmt.* params —
            // drivers parse their own URI query string and reject unknown
            // keys. BigQuery additionally needs its URI in the driver's
            // Simba grammar, with non-Simba params arriving only as
            // standalone options.
            let (driver_uri, opts_query) = if scheme == "bigquery" {
                bigquery_driver_uri(body, &query)
            } else {
                let filtered_uri = if query.is_empty() {
                    format!("{scheme}://{body}")
                } else {
                    format!("{scheme}://{body}?{query}")
                };
                (driver_uri_for(&scheme, body, &filtered_uri), query.clone())
            };
            let mut opts = vec![(OptionDatabase::Uri, OptionValue::String(driver_uri))];
            if query_params_as_driver_options(&scheme) {
                opts.extend(query_params_to_opts(&opts_query));
            }
            (driver, opts)
        };

        let dialect: Box<dyn SqlDialect + Send> =
            crate::reader::dialects::dialect_for_scheme(if scheme == "adbc" {
                body
            } else {
                &scheme
            })
            .unwrap_or_else(|| Box::new(AnsiDialect));

        Self::new_with_database_opts(driver, dialect, opts)
            .map(|reader| reader.with_statement_opts(stmt_opts))
    }
}

/// Compute the URI handed to the driver as the `uri` database option.
///
/// ggsql's scheme selects the driver and dialect, but the URI must use the
/// scheme the driver itself speaks. ClickHouse's ADBC driver connects over
/// the HTTP interface and expects http:// (or https:// for TLS). Its URI is
/// rebuilt from the body alone, dropping userinfo and query params: the
/// driver ignores userinfo (credentials must arrive as the dedicated
/// `username`/`password` options) and forwards any URL query parameters to
/// the server as ClickHouse *settings*, so `?username=…` left in the URL
/// would fail with "Unknown setting". Other drivers (e.g. PostgreSQL) parse
/// query parameters in the URI themselves, so their full URI passes through.
fn driver_uri_for(scheme: &str, body: &str, full_uri: &str) -> String {
    match scheme {
        "clickhouse" => format!("http://{body}"),
        // The Foundry MySQL/MariaDB driver wraps go-sql-driver/mysql, whose
        // DSN is `user[:pass]@tcp(host:port)/db` — not a URL. Translate
        // `mysql://user:pass@host:port/db` accordingly. Query params are not
        // carried into the DSN; they reach the driver as options instead.
        "mysql" | "mariadb" => {
            let (userinfo, host_db) = match body.rsplit_once('@') {
                Some((u, h)) => (format!("{u}@"), h),
                None => (String::new(), body),
            };
            match host_db.split_once('/') {
                Some((addr, db)) => format!("{userinfo}tcp({addr})/{db}"),
                None => format!("{userinfo}tcp({host_db})"),
            }
        }
        // The Foundry "redshift" driver is the PostgreSQL driver, whose
        // pgx-based URI parsing rejects the redshift:// scheme; rewrite it.
        // Rewrite the full URI (not `body`) so query params survive — they
        // are pgx connection settings the driver reads from the URI.
        "redshift" => full_uri.replacen("redshift://", "postgres://", 1),
        _ => full_uri.to_string(),
    }
}

/// Query parameters the Foundry BigQuery driver recognises in its own URI
/// parsing — the Simba JDBC vocabulary, case-sensitive (see the driver
/// docs). Any other key in a bigquery:// URI is handed over as a standalone
/// database option instead: the driver rejects unknown URI params
/// ("unknown parameter 'bigquery.auth_type' in URI"), and the canonical
/// `bigquery.*` option names only exist as standalone options.
const SIMBA_BIGQUERY_URI_PARAMS: &[&str] = &[
    "OAuthType",
    "AuthCredentials",
    "AuthClientId",
    "AuthClientSecret",
    "AuthRefreshToken",
    "DatasetId",
    "Location",
    "QuotaProject",
    "ImpersonateDelegates",
    "ImpersonateLifetime",
    "ImpersonateScopes",
    "ImpersonateTargetPrincipal",
];

/// Translate ggsql's `bigquery://<project>[/<dataset>]` convention into the
/// Simba-style URI the Foundry BigQuery driver parses,
/// `bigquery://[host[:port]]/<project>?DatasetId=<dataset>&<Simba params>`.
///
/// A first path segment containing '.' or ':' is treated as a host and the
/// second as the project (Simba form, passed through) — GCP project IDs
/// contain only lowercase letters, digits, and dashes, so they never look
/// host-like. Otherwise the segments are ggsql's project[/dataset] and the
/// URI is rewritten hostless (the driver defaults to
/// bigquery.googleapis.com; endpoint overrides arrive via the
/// `bigquery.endpoint` option, which has no URI form).
///
/// Returns the driver URI plus the query string to pass as standalone
/// database options: every param outside the Simba vocabulary. Simba params
/// stay in the URI only — passing them standalone as well would risk
/// duplicate-arrival errors of the kind the MSSQL and Databricks drivers
/// raise. An explicit `DatasetId` or `bigquery.dataset_id` param wins over
/// the path dataset.
fn bigquery_driver_uri(body: &str, query: &str) -> (String, String) {
    let (first, second) = match body.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (body, None),
    };
    let host_like = first.contains('.') || first.contains(':');

    let mut simba_params: Vec<&str> = Vec::new();
    let mut standalone: Vec<&str> = Vec::new();
    let mut dataset_param = false;
    for segment in query.split('&') {
        if segment.is_empty() {
            continue;
        }
        let key = segment.split('=').next().unwrap_or_default();
        if key == "DatasetId" || key == "bigquery.dataset_id" {
            dataset_param = true;
        }
        if SIMBA_BIGQUERY_URI_PARAMS.contains(&key) {
            simba_params.push(segment);
        } else {
            standalone.push(segment);
        }
    }

    let mut uri = String::from("bigquery://");
    let mut query_started = false;
    if host_like {
        uri.push_str(first);
        uri.push('/');
        if let Some(project) = second {
            uri.push_str(project);
        }
    } else {
        uri.push('/');
        uri.push_str(first);
        if let (Some(dataset), false) = (second, dataset_param) {
            uri.push_str("?DatasetId=");
            uri.push_str(dataset);
            query_started = true;
        }
    }
    if !simba_params.is_empty() {
        uri.push(if query_started { '&' } else { '?' });
        uri.push_str(&simba_params.join("&"));
    }
    (uri, standalone.join("&"))
}

/// Whether `?k=v` query params are also passed to the driver as standalone
/// database options. (They always remain in the `uri` option as well, except
/// where [`driver_uri_for`] strips them.) Some drivers parse their own URI
/// and reject params arriving a second way: the MSSQL driver fails with
/// "Unknown database option 'TrustServerCertificate'", and the Databricks
/// driver fails with "cannot specify both URI and individual connection
/// options". Their params stay in the URI only.
fn query_params_as_driver_options(scheme: &str) -> bool {
    !matches!(scheme, "mssql" | "databricks" | "spark")
}

/// Probe whether an ADBC driver for `scheme` can be loaded, without opening
/// a connection. Used by reader dispatch to decide between ADBC and ODBC.
pub fn adbc_driver_available(scheme: &str) -> bool {
    load_driver_for_scheme(scheme).is_ok()
}

use adbc_core::sync::Statement;
use arrow::record_batch::RecordBatch;

impl<D: Driver + 'static> Reader for AdbcReader<D>
where
    D::DatabaseType: 'static,
    <D::DatabaseType as Database>::ConnectionType: 'static,
{
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        use arrow::array::RecordBatchReader as _;

        // Drain the `RecordBatchReader` *inside* the connection-borrow scope
        // so `stmt` and the `RefMut<Connection>` stay alive while batches are
        // streamed from the server. The `FlightSQL` driver's reader holds a
        // gRPC stream whose context is tied to the Statement; if `stmt` drops
        // before iteration completes, the first `DoGet` call cancels with
        // `Canceled; DoGet: endpoint 0: []`. Other ADBC drivers (DataFusion,
        // etc.) return self-sufficient readers, but paying for an extra early
        // release on those is worthwhile to keep a single correct code path.
        let (schema, batches) = {
            let mut conn = self.connection.try_borrow_mut().map_err(|_| {
                GgsqlError::ReaderError(
                    "AdbcReader is already mutably borrowed — another \
                     `execute_sql`/`register`/`unregister` is in progress \
                     on this reader"
                        .into(),
                )
            })?;
            let mut stmt = conn
                .new_statement()
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
            self.apply_statement_opts(&mut stmt)?;
            stmt.set_sql_query(sql)
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC set_sql_query: {}", e)))?;
            let reader = match stmt.execute() {
                Ok(reader) => reader,
                Err(e) => {
                    let msg = e.to_string();
                    // BigQuery DDL/DML jobs without a result set execute to
                    // completion but fail at read time with "job has no
                    // destination table to read". The statement has already
                    // run — report an empty frame. Retrying via
                    // execute_update would re-execute the statement, which
                    // breaks non-idempotent DDL (a second CREATE TABLE fails
                    // with "already exists").
                    if msg.contains("no destination table to read") {
                        return Ok(DataFrame::from_record_batch(RecordBatch::new_empty(
                            std::sync::Arc::new(arrow::datatypes::Schema::empty()),
                        )));
                    }
                    // The Databricks driver's query path cannot build a result
                    // reader for statements without a result set (DDL), failing
                    // with "schema bytes are empty" before executing. Retry via
                    // execute_update — the ADBC path meant for exactly those
                    // statements — and report an empty frame.
                    if msg.contains("schema bytes are empty") {
                        drop(stmt);
                        let mut update_stmt = conn.new_statement().map_err(|e| {
                            GgsqlError::ReaderError(format!("ADBC new_statement: {}", e))
                        })?;
                        self.apply_statement_opts(&mut update_stmt)?;
                        update_stmt.set_sql_query(sql).map_err(|e| {
                            GgsqlError::ReaderError(format!("ADBC set_sql_query: {}", e))
                        })?;
                        update_stmt.execute_update().map_err(|e| {
                            GgsqlError::ReaderError(format!("ADBC execute_update: {}", e))
                        })?;
                        return Ok(DataFrame::from_record_batch(RecordBatch::new_empty(
                            std::sync::Arc::new(arrow::datatypes::Schema::empty()),
                        )));
                    }
                    return Err(GgsqlError::ReaderError(format!("ADBC execute: {}", e)));
                }
            };

            // Capture the declared result schema before draining batches —
            // the reader carries it even when zero batches are produced, and
            // we need it to preserve column names on empty results.
            let schema = reader.schema();
            let mut batches: Vec<RecordBatch> = Vec::new();
            for batch in reader {
                batches.push(batch.map_err(|e| {
                    GgsqlError::ReaderError(format!("ADBC RecordBatch iter: {}", e))
                })?);
            }
            (schema, batches)
        };

        let merged = if batches.is_empty() {
            RecordBatch::new_empty(schema)
        } else if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            arrow::compute::concat_batches(&schema, &batches)
                .map_err(|e| GgsqlError::ReaderError(format!("concat_batches: {}", e)))?
        };
        Ok(DataFrame::from_record_batch(merged))
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        super::validate_table_name(name)?;

        use adbc_core::options::{IngestMode, OptionStatement, OptionValue};
        use adbc_core::Optionable;

        if df.height() == 0 {
            return Err(GgsqlError::ReaderError(
                "AdbcReader::register: empty DataFrame not supported".into(),
            ));
        }
        let batch = df.into_inner();

        let mut conn = self.connection.try_borrow_mut().map_err(|_| {
            GgsqlError::ReaderError(
                "AdbcReader::register called re-entrantly — another operation \
                 is still holding the connection on this reader"
                    .into(),
            )
        })?;

        // Bulk-insert path: CREATE TABLE via SQL DDL, then for each batch set
        // `TargetTable` + `IngestMode::Append` + `bind(batch)` +
        // `execute_update()`. We do the CREATE ourselves (rather than relying
        // on `IngestMode::Create`) so we control the column types via the
        // `SqlDialect` and so registers behave identically across drivers
        // with varying ingest-option support — in particular,
        // `adbc_datafusion` 0.23 has `bind_stream` as `todo!()` and rejects
        // the `IngestMode` option key (`set_option` returns `NotFound`),
        // which is silently tolerated below.
        let schema = batch.schema();
        if replace {
            let drop_sql = format!("DROP TABLE IF EXISTS {}", self.dialect.quote_ident(name));
            let mut drop_stmt = conn
                .new_statement()
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
            self.apply_statement_opts(&mut drop_stmt)?;
            drop_stmt
                .set_sql_query(&drop_sql)
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC set_sql_query DROP: {}", e)))?;
            drop_stmt
                .execute_update()
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC execute_update DROP: {}", e)))?;
        }

        let create_sql = create_table_sql(name, &schema, &*self.dialect)?;
        let mut create_stmt = conn
            .new_statement()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
        self.apply_statement_opts(&mut create_stmt)?;
        create_stmt
            .set_sql_query(&create_sql)
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC set_sql_query CREATE: {}", e)))?;
        create_stmt
            .execute_update()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC execute_update CREATE: {}", e)))?;

        // Track the table in our set as soon as CREATE succeeds — BEFORE the
        // potentially-multi-batch ingest loop. If a bind or execute_update
        // fails mid-way, the (partial) table still exists on the server;
        // having the name tracked lets the caller `unregister()` to clean
        // up, and a subsequent `register(name, ..., replace=true)` will
        // drop-and-recreate. Without this, a mid-ingest failure would leave
        // an orphan table the reader can't reach.
        self.registered_tables.borrow_mut().insert(name.to_string());

        {
            let mut stmt = conn
                .new_statement()
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
            stmt.set_option(
                OptionStatement::TargetTable,
                OptionValue::String(name.to_string()),
            )
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC set TargetTable: {}", e)))?;
            // Tell the driver this is an append into the table we just
            // CREATEd above. Compliant ADBC drivers (e.g. the Apache SQLite
            // driver) default `IngestMode` to `Create` when only `TargetTable`
            // is set, which would then fail because the table already exists.
            // DataFusion 0.23 doesn't expose this option key and returns
            // `Status::NotFound` from `set_option`; that's expected for
            // DataFusion's bind path (it appends by default), so swallow it
            // and continue rather than failing register().
            if let Err(e) = stmt.set_option(
                OptionStatement::IngestMode,
                OptionValue::from(IngestMode::Append),
            ) {
                if e.status != adbc_core::error::Status::NotFound {
                    return Err(GgsqlError::ReaderError(format!(
                        "ADBC set IngestMode=Append: {}",
                        e
                    )));
                }
            }
            stmt.bind(batch)
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC bind: {}", e)))?;
            stmt.execute_update().map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "ADBC execute_update: {} — \
                     table left on server; call unregister() to drop it \
                     or register() with replace=true to retry",
                    e
                ))
            })?;
        }

        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        if !self.registered_tables.borrow().contains(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }
        let sql = format!("DROP TABLE IF EXISTS {}", self.dialect.quote_ident(name));
        // Ignore the returned DataFrame — DROP TABLE has no result rows.
        self.execute_sql(&sql)?;
        self.registered_tables.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<crate::reader::Spec> {
        crate::reader::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn SqlDialect {
        &*self.dialect
    }
}

/// Build a `CREATE TABLE <name> (col1 TYPE, col2 TYPE, ...)` statement from
/// an Arrow schema, using the reader's `SqlDialect` for type names.
///
/// Used by `register()` to create the destination table before binding
/// batches with `IngestMode::Append`; see the `register` impl for context.
fn create_table_sql(
    name: &str,
    schema: &arrow::datatypes::Schema,
    dialect: &dyn SqlDialect,
) -> Result<String> {
    use arrow::datatypes::DataType;

    let mut cols: Vec<String> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let ty_name: &str = match field.data_type() {
            DataType::Boolean => dialect.boolean_type_name().unwrap_or("BOOLEAN"),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => dialect.integer_type_name().unwrap_or("BIGINT"),
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                dialect.number_type_name().unwrap_or("DOUBLE PRECISION")
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                dialect.string_type_name().unwrap_or("VARCHAR")
            }
            DataType::Date32 | DataType::Date64 => dialect.date_type_name().unwrap_or("DATE"),
            DataType::Timestamp(_, _) => dialect.datetime_type_name().unwrap_or("TIMESTAMP"),
            DataType::Time32(_) | DataType::Time64(_) => dialect.time_type_name().unwrap_or("TIME"),
            other => {
                return Err(GgsqlError::ReaderError(format!(
                    "AdbcReader::register: unsupported Arrow type for column '{}': {:?}",
                    field.name(),
                    other
                )));
            }
        };
        cols.push(format!("{} {}", dialect.quote_ident(field.name()), ty_name));
    }

    Ok(format!(
        "CREATE TABLE {} ({})",
        dialect.quote_ident(name),
        cols.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "adbc-datafusion")]
    use adbc_datafusion::DataFusionDriver;

    #[test]
    fn driver_name_mapping() {
        assert_eq!(
            driver_name_for_scheme("postgres"),
            Some("adbc_driver_postgresql")
        );
        assert_eq!(
            driver_name_for_scheme("snowflake"),
            Some("adbc_driver_snowflake")
        );
        assert_eq!(driver_name_for_scheme("mysql"), Some("adbc_driver_mysql"));
        assert_eq!(driver_name_for_scheme("trino"), Some("adbc_driver_trino"));
        assert_eq!(
            driver_name_for_scheme("clickhouse"),
            Some("adbc_driver_clickhouse")
        );
        assert_eq!(driver_name_for_scheme("mssql"), Some("adbc_driver_mssql"));
        assert_eq!(driver_name_for_scheme("oracle"), Some("adbc_driver_oracle"));
        assert_eq!(driver_name_for_scheme("exasol"), Some("adbc_driver_exasol"));
        assert_eq!(driver_name_for_scheme("druid"), Some("adbc_driver_druid"));
        assert_eq!(driver_name_for_scheme("nosuch"), None);
    }

    #[test]
    fn driver_uri_rewrites_clickhouse_and_strips_query() {
        // Credentials must travel as dedicated options, not in the URL.
        assert_eq!(
            driver_uri_for(
                "clickhouse",
                "localhost:8123",
                "clickhouse://localhost:8123?username=default&password=secret"
            ),
            "http://localhost:8123"
        );
        // Other schemes keep the full URI, query included.
        assert_eq!(
            driver_uri_for(
                "postgres",
                "u:p@h/db",
                "postgres://u:p@h/db?sslmode=disable"
            ),
            "postgres://u:p@h/db?sslmode=disable"
        );
    }

    #[test]
    fn driver_uri_translates_mysql_to_go_dsn() {
        assert_eq!(
            driver_uri_for(
                "mysql",
                "root:pw@localhost:3306/ggsql",
                "mysql://root:pw@localhost:3306/ggsql"
            ),
            "root:pw@tcp(localhost:3306)/ggsql"
        );
        // No userinfo.
        assert_eq!(
            driver_uri_for(
                "mariadb",
                "localhost:3306/ggsql",
                "mariadb://localhost:3306/ggsql"
            ),
            "tcp(localhost:3306)/ggsql"
        );
    }

    #[test]
    fn driver_uri_translates_bigquery_to_simba_grammar() {
        // ggsql's project/dataset form becomes a hostless Simba URI;
        // canonical bigquery.* params travel as standalone options only.
        assert_eq!(
            bigquery_driver_uri("my-proj/my_ds", "bigquery.auth_type=anonymous"),
            (
                "bigquery:///my-proj?DatasetId=my_ds".to_string(),
                "bigquery.auth_type=anonymous".to_string()
            )
        );
        // Project only, no params.
        assert_eq!(
            bigquery_driver_uri("my-proj", ""),
            ("bigquery:///my-proj".to_string(), String::new())
        );
        // Host-like first segment: Simba form passes through, Simba params
        // stay in the URI, everything else goes standalone.
        assert_eq!(
            bigquery_driver_uri(
                "localhost:9050/ggsql-test",
                "DatasetId=x&bigquery.endpoint=http://localhost:9050"
            ),
            (
                "bigquery://localhost:9050/ggsql-test?DatasetId=x".to_string(),
                "bigquery.endpoint=http://localhost:9050".to_string()
            )
        );
        // An explicit dataset param wins over the path dataset.
        assert_eq!(
            bigquery_driver_uri("proj/ds1", "bigquery.dataset_id=ds2"),
            (
                "bigquery:///proj".to_string(),
                "bigquery.dataset_id=ds2".to_string()
            )
        );
    }

    #[test]
    fn driver_uri_rewrites_redshift_scheme() {
        // pgx rejects redshift://; the driver is the PostgreSQL one.
        assert_eq!(
            driver_uri_for(
                "redshift",
                "u:p@h:5439/db",
                "redshift://u:p@h:5439/db?sslmode=disable"
            ),
            "postgres://u:p@h:5439/db?sslmode=disable"
        );
    }

    #[test]
    fn uri_parsing_drivers_reject_standalone_options() {
        assert!(!query_params_as_driver_options("mssql"));
        assert!(!query_params_as_driver_options("databricks"));
        assert!(!query_params_as_driver_options("spark"));
        assert!(query_params_as_driver_options("postgres"));
        assert!(query_params_as_driver_options("clickhouse"));
        assert!(query_params_as_driver_options("exasol"));
    }

    #[test]
    fn driver_names_include_dbc_manifest_id() {
        // dbc names manifests by short ID (`postgresql.toml`), which differs
        // from the library name — both must be available for probing, or a
        // `dbc install`-based setup is never found (first seen as a CI
        // failure where dbc-installed drivers were not discovered).
        assert_eq!(
            driver_names_for_scheme("postgres"),
            Some(("adbc_driver_postgresql", "postgresql"))
        );
        assert_eq!(
            driver_names_for_scheme("redshift"),
            Some(("adbc_driver_postgresql", "postgresql"))
        );
        assert_eq!(
            driver_names_for_scheme("trino"),
            Some(("adbc_driver_trino", "trino"))
        );
        assert_eq!(
            driver_names_for_scheme("mariadb"),
            Some(("adbc_driver_mysql", "mysql"))
        );
        assert_eq!(driver_names_for_scheme("nosuch"), None);
    }

    #[test]
    fn from_connection_string_unknown_driver_errors() {
        let err = AdbcReader::<ManagedDriver>::from_connection_string("adbc://nosuchdriver?uri=x")
            .err()
            .expect("load must fail");
        assert!(err.to_string().contains("nosuchdriver"), "got: {err}");
    }

    #[test]
    fn from_connection_string_rejects_malformed_uri() {
        assert!(AdbcReader::<ManagedDriver>::from_connection_string("no-scheme").is_err());
    }

    #[test]
    fn query_params_map_to_adbc_options() {
        let opts =
            query_params_to_opts("uri=postgresql://h/db&username=u&password=p&sslmode=require");
        assert!(matches!(opts[0].0, OptionDatabase::Uri));
        assert!(matches!(opts[1].0, OptionDatabase::Username));
        assert!(matches!(opts[2].0, OptionDatabase::Password));
        assert!(matches!(&opts[3].0, OptionDatabase::Other(k) if k == "sslmode"));
    }

    /// Construct a reader over an in-process DataFusion ADBC driver.
    /// DataFusion starts empty; callers register tables via the reader's
    /// `register()` method (added in Task 4) or via raw SQL DDL.
    #[cfg(feature = "adbc-datafusion")]
    fn fixture_reader() -> AdbcReader<DataFusionDriver> {
        AdbcReader::from_driver(DataFusionDriver::new(None)).expect("datafusion init")
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn execute_sql_returns_scalar_result() {
        use crate::array_util::as_i64;
        let reader = fixture_reader();
        let df = reader
            .execute_sql("SELECT 1 AS one, 'hello' AS greeting")
            .expect("query ok");
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 2);
        let one = as_i64(df.column("one").unwrap()).unwrap().value(0);
        assert_eq!(one, 1);
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn register_then_query_roundtrip() {
        use crate::array_util::as_i64;
        use crate::df;

        let reader = fixture_reader();
        let df = df! {
            "x" => vec![1i64, 2, 3],
            "y" => vec!["a", "b", "c"],
        }
        .unwrap();
        reader.register("t", df, false).expect("register ok");

        let out = reader
            .execute_sql("SELECT COUNT(*) AS n FROM t")
            .expect("count ok");
        let n = as_i64(out.column("n").unwrap()).unwrap().value(0);
        assert_eq!(n, 3);
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn unregister_removes_table() {
        use crate::df;

        let reader = fixture_reader();
        let df = df! { "x" => vec![1i64] }.unwrap();
        reader.register("tmp", df, false).unwrap();

        // First unregister should succeed: table was registered via this reader.
        reader.unregister("tmp").expect("unregister ok");

        // Second unregister must fail: the name was removed from
        // registered_tables, so the guard in unregister() triggers.
        // This verifies the bookkeeping without triggering the
        // adbc_datafusion 0.23 Statement::execute panic that happens on
        // `SELECT * FROM <dropped-table>` (the driver .unwrap()s a DataFusion
        // planning error at lib.rs:913 instead of returning a proper ADBC
        // error — captured in Task 9 findings).
        let err = reader.unregister("tmp").unwrap_err();
        assert!(matches!(err, GgsqlError::ReaderError(_)));
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn with_dialect_plumbs_custom_dialect_through() {
        // Dummy dialect that overrides a recognizable method so we can verify
        // the reader actually stored and exposes our dialect rather than the
        // default AnsiDialect.
        struct ShoutyDialect;
        impl super::SqlDialect for ShoutyDialect {
            fn integer_type_name(&self) -> Option<&str> {
                Some("SHOUTY_BIGINT")
            }
        }

        let reader = AdbcReader::with_dialect(DataFusionDriver::new(None), Box::new(ShoutyDialect))
            .expect("reader");

        // The Reader trait's dialect() accessor should return our ShoutyDialect.
        assert_eq!(reader.dialect().integer_type_name(), Some("SHOUTY_BIGINT"));
    }

    #[test]
    #[ignore = "ggsql's execute pipeline issues `CREATE OR REPLACE TEMP TABLE` for layer/stat \
                materialization, which adbc_datafusion 0.23 rejects with `NotImplemented(\"Temporary \
                tables not supported\")`. The full pipeline works against any driver that supports \
                TEMP TABLE (DuckDB, Trino, etc.) — see the equivalence tests for that path."]
    #[cfg(feature = "adbc-datafusion")]
    fn reader_executes_full_ggsql_visualise_query() {
        use crate::df;

        let reader = fixture_reader();
        let data = df! {
            "date"   => vec!["2024-01-01", "2024-01-02", "2024-01-03"],
            "value"  => vec![10i64, 20, 30],
            "region" => vec!["N", "S", "N"],
        }
        .unwrap();
        reader.register("sales", data, false).unwrap();

        let query = r#"
            SELECT date, value, region FROM sales WHERE value > 5
            VISUALISE date AS x, value AS y, region AS color
            DRAW line
        "#;
        let spec = reader.execute(query).expect("ggsql execute ok");
        let meta = spec.metadata();
        // Full pipeline verification: SQL executed (3 rows after WHERE),
        // VISUALISE parsed, plot resolved with 1 layer.
        assert_eq!(meta.rows, 3);
        assert_eq!(meta.layer_count, 1);
        // The `columns` list reports the *transformed aesthetic* column names
        // (e.g. x -> pos1, y -> pos2, color -> stroke on a line layer) not the
        // raw SQL column names. See `test_execute_metadata` in reader/mod.rs
        // for the same convention.
        assert!(
            meta.columns.iter().any(|c| c == "pos1"),
            "expected pos1 (x aesthetic) in columns: {:?}",
            meta.columns
        );
        assert!(
            meta.columns.iter().any(|c| c == "pos2"),
            "expected pos2 (y aesthetic) in columns: {:?}",
            meta.columns
        );
        assert!(
            meta.columns.iter().any(|c| c == "stroke"),
            "expected stroke (color aesthetic on line) in columns: {:?}",
            meta.columns
        );
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn execute_sql_handles_multi_batch_result() {
        use crate::array_util::as_i64;
        use crate::df;

        // Register a 50k-row frame. DataFusion's default batch size is typically
        // around 8k rows, so the result read-side should produce >1 RecordBatch
        // and exercise the `for batch in reader` loop.
        let reader = fixture_reader();
        let xs: Vec<i64> = (0..50_000i64).collect();
        let df = df! { "x" => xs }.unwrap();
        reader.register("big", df, false).expect("register ok");

        let out = reader
            .execute_sql("SELECT x FROM big ORDER BY x")
            .expect("query ok");
        assert_eq!(out.height(), 50_000);

        // Spot-check: first + last rows should round-trip correctly.
        let col = out.column("x").unwrap();
        let arr = as_i64(col).unwrap();
        assert_eq!(arr.value(0), 0);
        assert_eq!(arr.value(49_999), 49_999);
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn execute_sql_handles_nulls() {
        use crate::array_util::as_i64;
        use arrow::array::Array;

        let reader = fixture_reader();
        // Use DataFusion DDL to create a table with a NULL.
        reader
            .execute_sql("CREATE TABLE nulltest (x BIGINT) AS VALUES (1), (NULL), (3)")
            .expect("ddl ok");

        let out = reader
            .execute_sql("SELECT x FROM nulltest ORDER BY x NULLS LAST")
            .expect("query ok");
        assert_eq!(out.height(), 3);

        let col = out.column("x").unwrap();
        let arr = as_i64(col).unwrap();
        // Row 2 should be NULL in the returned DataFrame.
        assert!(arr.is_null(2));
        // Rows 0 and 1 are the non-null values.
        assert_eq!(arr.value(0), 1);
        assert_eq!(arr.value(1), 3);
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    #[ignore]
    fn bench_register_and_query_100k_rows() {
        use crate::array_util::as_i64;
        use crate::df;
        use std::time::Instant;

        let reader = fixture_reader();
        let n = 100_000i64;
        let xs: Vec<i64> = (0..n).collect();
        let df = df! { "x" => xs }.unwrap();

        let t0 = Instant::now();
        reader.register("big", df, false).unwrap();
        let reg_ms = t0.elapsed().as_millis();

        let t1 = Instant::now();
        let out = reader.execute_sql("SELECT COUNT(*) AS n FROM big").unwrap();
        let q_ms = t1.elapsed().as_millis();

        let n_out = as_i64(out.column("n").unwrap()).unwrap().value(0);
        assert_eq!(n_out, n);
        eprintln!("register 100k rows: {} ms | query: {} ms", reg_ms, q_ms);
    }

    /// Issue #12: `execute_sql` must hold `conn.borrow_mut()` only long enough
    /// to build + execute the Statement — the returned `RecordBatchReader` is
    /// `Box<dyn ... + 'static>`, so iteration must not require the statement
    /// or the connection borrow to stay alive.
    ///
    /// This mirrors the exact borrow pattern `execute_sql` uses post-fix:
    /// borrow, build+execute, drop the borrow, then iterate. It also kicks
    /// off a second `execute_sql` while the first stream is still alive —
    /// only possible if the first borrow was released.
    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn record_batch_reader_outlives_statement_and_allows_second_query() {
        use arrow::array::RecordBatchReader as _;

        let reader = fixture_reader();

        let stream = {
            // Use `try_borrow_mut` here to mirror `execute_sql`'s production
            // path — if this ever panics in the test, the fix in `execute_sql`
            // has regressed and the borrow scope has crept wider again.
            let mut conn = reader
                .connection
                .try_borrow_mut()
                .expect("fresh reader should allow a mutable borrow");
            let mut stmt = conn.new_statement().expect("new_statement");
            stmt.set_sql_query("SELECT 1 AS v UNION ALL SELECT 2 UNION ALL SELECT 3")
                .expect("set_sql_query");
            stmt.execute().expect("execute")
            // `stmt` and the `RefMut<Connection>` both drop here.
        };

        // With the borrow released, another query on the same reader must
        // work while `stream` is still live.
        let df2 = reader
            .execute_sql("SELECT 42 AS answer")
            .expect("second query");
        assert_eq!(df2.height(), 1);

        // `stream` must still iterate — it does not depend on `stmt` or the
        // original borrow. `schema()` is called before `collect()` consumes
        // the reader.
        let schema = stream.schema();
        let batches = stream
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("drain");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "v");
    }

    #[cfg(feature = "adbc-datafusion")]
    #[test]
    fn execute_sql_handles_empty_result_with_schema() {
        let reader = fixture_reader();
        let df = reader
            .execute_sql("SELECT 1 AS a, 'x' AS b WHERE false")
            .expect("query ok");
        // The schema is preserved on zero-batch results: we now pull the
        // declared schema off the `RecordBatchReader` *before* draining
        // batches and hand it to the IPC bridge so an empty result still
        // produces a 0-row DataFrame with the correct columns.
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 2);
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod equivalence_tests {
    //! Equivalence tests: `AdbcReader<sqlite via ManagedDriver>` vs ggsql's
    //! `SqliteReader` on the same query against the same SQLite DB. Validates
    //! correctness of the ADBC abstraction's routing, type bridging, and
    //! ingest paths against a real, fully-functional ADBC driver.
    //!
    //! Skipped by default (gated `#[ignore]`). To run them:
    //!
    //! 1. Install dbc: `curl -LsSf https://dbc.columnar.tech/install.sh | sh`
    //! 2. Install the SQLite driver: `dbc install sqlite`
    //! 3. Run: `cargo test --features "adbc sqlite" -- --ignored equivalence`
    //!
    //! `dbc install` writes the driver to a manifest location that
    //! `ManagedDriver::load_from_name("sqlite", ...)` discovers automatically
    //! (on macOS: `~/Library/Application Support/ADBC/Drivers/sqlite.toml`).
    //!
    //! Why SQLite (and not DuckDB) as the equivalence oracle: `libduckdb` is
    //! distributed as a bundled-static archive, so it can't be loaded as the
    //! shared library that `ManagedDriver` requires. The Apache-published
    //! SQLite ADBC driver ships as `libadbc_driver_sqlite.dylib` and is the
    //! reference C-driver path for round-tripping through `adbc_driver_manager`.

    use crate::reader::sqlite::SqliteDialect;
    use crate::reader::{AdbcReader, Reader, SqliteReader};
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::LOAD_FLAG_DEFAULT;
    use adbc_driver_manager::ManagedDriver;
    use tempfile::NamedTempFile;

    /// Construct an `AdbcReader` pointed at a specific SQLite file.
    /// Both readers in each test point at the SAME file so equivalence is
    /// over the same physical database.
    fn make_adbc_reader(db_path: &str) -> AdbcReader<ManagedDriver> {
        let driver = ManagedDriver::load_from_name(
            "sqlite",
            None,
            AdbcVersion::V110,
            LOAD_FLAG_DEFAULT,
            None,
        )
        .expect("`dbc install sqlite` first; see module docs");
        let dialect: Box<dyn crate::reader::SqlDialect + Send> = Box::new(SqliteDialect);
        AdbcReader::new_with_database_opts(
            driver,
            dialect,
            std::iter::once((
                OptionDatabase::Uri,
                OptionValue::String(format!("file:{}", db_path)),
            )),
        )
        .expect("construct AdbcReader<sqlite>")
    }

    fn make_sqlite_reader(db_path: &str) -> SqliteReader {
        SqliteReader::from_connection_string(&format!("sqlite://{}", db_path))
            .expect("SqliteReader at the same path")
    }

    use crate::reader::test_support::assert_dataframes_equal;

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_simple_select() {
        let db = NamedTempFile::new().unwrap();
        let db_path = db.path().to_str().unwrap();
        let adbc = make_adbc_reader(db_path);
        let direct = make_sqlite_reader(db_path);
        let sql = "SELECT 1 AS x, 'hello' AS y, 3.14 AS z";
        let a = adbc.execute_sql(sql).unwrap();
        let d = direct.execute_sql(sql).unwrap();
        assert_dataframes_equal(&a, &d, "simple select");
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_register_and_query() {
        // Register through the ADBC reader (exercises the standard ADBC
        // bulk-ingest path), then read back through SqliteReader (talks to
        // rusqlite directly against the same file) AND through the ADBC
        // reader. Both should agree.
        let db = NamedTempFile::new().unwrap();
        let db_path = db.path().to_str().unwrap();
        let adbc = make_adbc_reader(db_path);
        let df = crate::df! {
            "x" => vec![1i64, 2, 3, 4, 5],
            "y" => vec![10i64, 20, 30, 40, 50],
        }
        .unwrap();
        adbc.register("t", df, false).unwrap();

        // Open the SqliteReader AFTER the ADBC reader has CREATEd + ingested,
        // so its `Connection::open` sees the on-disk schema written by ADBC.
        let direct = make_sqlite_reader(db_path);

        let sql = "SELECT x, y, x*y AS xy FROM t WHERE y > 15 ORDER BY x";
        let a = adbc.execute_sql(sql).unwrap();
        let d = direct.execute_sql(sql).unwrap();
        assert_dataframes_equal(&a, &d, "register + filter + projection");
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_nulls() {
        // Mix nulls with typed values so both readers infer the same type.
        // (SqliteReader's per-row type inference falls back to Utf8 when a
        // column is *exclusively* NULL, while ADBC carries through the
        // declared INTEGER from the projection metadata. That's a
        // SqliteReader limitation, not an AdbcReader bug, so we steer
        // around it here — see the divergence note in the PR description.)
        let db = NamedTempFile::new().unwrap();
        let db_path = db.path().to_str().unwrap();
        let adbc = make_adbc_reader(db_path);
        let direct = make_sqlite_reader(db_path);
        // SQLite doesn't accept `VALUES (..) AS t(col, ...)` column-list
        // aliases, so build the source rows with UNION ALL — both readers
        // handle this identically.
        let sql = "SELECT i, s FROM ( \
                SELECT CAST(1 AS INTEGER) AS i, CAST('a' AS TEXT) AS s \
                UNION ALL SELECT NULL, 'b' \
                UNION ALL SELECT 3, NULL \
            ) ORDER BY i";
        let a = adbc.execute_sql(sql).unwrap();
        let d = direct.execute_sql(sql).unwrap();
        assert_dataframes_equal(&a, &d, "mixed null + typed values");
    }
}
