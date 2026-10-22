//! ADBC (Arrow Database Connectivity) reader.
//!
//! Generic over any concrete ADBC `Driver` implementation. Verified against:
//!
//! - `adbc_driver_sqlite` (loaded via `adbc_driver_manager::ManagedDriver`)
//!   — a real ADBC C driver, used for an equivalence suite that compares
//!   `AdbcReader<SQLite>` output against ggsql's existing `SqliteReader`.
//! - the Foundry `datafusion` driver (also via `ManagedDriver`) — exercised
//!   end-to-end by the `live_datafusion` case in `tests/dialect_live.rs`.
//!
//! The `Reader` trait takes `&self`, but ADBC's `Statement` API takes
//! `&mut self`. We bridge this with `RefCell` around the `Connection`,
//! mirroring the interior-mutability pattern used by `OdbcReader`.

use crate::reader::{AnsiDialect, Reader, SqlDialect};
use crate::{DataFrame, GgsqlError, Result};
use adbc_core::sync::{Connection, Database, Driver};
use std::cell::RefCell;

pub struct AdbcReader<D: Driver> {
    // Driver must stay alive as long as the Database does (per ADBC contract).
    _driver: D,
    // Database must stay alive as long as the Connection does.
    _database: D::DatabaseType,
    // Connection is what Statements are made from. Wrapped in RefCell because
    // new_statement / set_sql_query / execute all take &mut, but Reader::execute_sql
    // takes &self.
    connection: RefCell<<D::DatabaseType as Database>::ConnectionType>,
    dialect: crate::reader::DialectRef,
    /// The registry entry for this backend, when the reader was built from
    /// a connection string for a registered backend — source of the
    /// driver quirks (error-text conventions) honored in `execute_sql` and
    /// `register`. `None` for ad-hoc drivers (`new`, `with_dialect`).
    entry: Option<&'static crate::reader::registry::DatabaseEntry>,
    registered_tables: crate::reader::RegisteredTables,
    // Driver-specific statement options (from `stmt.`-prefixed URI params)
    // applied to every statement created in execute_sql.
    statement_opts: Vec<(String, String)>,
}

impl<D: Driver> AdbcReader<D> {
    /// Construct an `AdbcReader` with an explicit `SqlDialect`. Use this to
    /// plug in backend-specific dialects (e.g. a TrinoDialect, SnowflakeDialect)
    /// when the reader is pointed at that backend.
    pub fn with_dialect(driver: D, dialect: crate::reader::DialectRef) -> Result<Self> {
        Self::new(driver, dialect)
    }

    /// Create a new `AdbcReader` from an already-initialized ADBC driver.
    ///
    /// Callers are responsible for any pre-init `Database` / `Connection`
    /// options. For convenience, use `from_driver` for the common case
    /// with the ANSI dialect, or pass a custom `SqlDialect`
    /// (e.g. a Trino / Snowflake dialect) here directly.
    pub fn new(mut driver: D, dialect: crate::reader::DialectRef) -> Result<Self> {
        let database = driver
            .new_database()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_database failed: {}", e)))?;
        let connection = database
            .new_connection()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_connection failed: {}", e)))?;
        Ok(Self {
            _driver: driver,
            _database: database,
            connection: RefCell::new(connection),
            dialect,
            entry: None,
            registered_tables: crate::reader::RegisteredTables::new(),
            statement_opts: Vec::new(),
        })
    }

    /// Create a new `AdbcReader`, passing pre-init options to the underlying
    /// `Database`. Use this when the driver requires URI / credentials / RPC
    /// header options to be set before the first connection (e.g. Flight SQL
    /// or other auth-required backends).
    pub fn new_with_database_opts(
        mut driver: D,
        dialect: crate::reader::DialectRef,
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
        let connection = database
            .new_connection()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_connection failed: {}", e)))?;
        Ok(Self {
            _driver: driver,
            _database: database,
            connection: RefCell::new(connection),
            dialect,
            entry: None,
            registered_tables: crate::reader::RegisteredTables::new(),
            statement_opts: Vec::new(),
        })
    }

    /// Attach driver-specific *statement* options, applied to every statement
    /// the reader creates in `execute_sql`. These come from `stmt.`-prefixed
    /// URI params (lifted out by [`crate::reader::connection::ConnUri`]) and
    /// exist because some
    /// drivers expose per-query settings only at the statement level — e.g.
    /// BigQuery's `bigquery.query.destination_table`, which is needed to
    /// read query results from the goccy BigQuery emulator (it does not
    /// serve anonymous result tables over the Storage Read API).
    pub fn with_statement_opts(mut self, opts: Vec<(String, String)>) -> Self {
        self.statement_opts = opts;
        self
    }

    /// Attach the registry entry for this backend, so its driver quirks
    /// (DDL-without-result-set signaling, ingest schema mismatches) are
    /// honored where the driver requires them.
    fn with_entry(mut self, entry: &'static crate::reader::registry::DatabaseEntry) -> Self {
        self.entry = Some(entry);
        self
    }

    /// Create a statement for `sql` with [`Self::statement_opts`] applied —
    /// the shared constructor behind every `set_sql_query` call site.
    fn new_query_statement(
        &self,
        conn: &mut <<D as Driver>::DatabaseType as Database>::ConnectionType,
        sql: &str,
    ) -> Result<
        <<<D as Driver>::DatabaseType as Database>::ConnectionType as Connection>::StatementType,
    > {
        let mut stmt = conn
            .new_statement()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
        self.apply_statement_opts(&mut stmt)?;
        stmt.set_sql_query(sql)
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC set_sql_query: {}", e)))?;
        Ok(stmt)
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
        Self::new(driver, &AnsiDialect)
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

/// ADBC driver details for a URI scheme, from the
/// [registry](crate::reader::registry): the canonical driver library name
/// and the dbc manifest ID.
///
/// Both names are needed because they are spelled differently on disk:
/// `dbc install postgresql` writes a manifest named after its short driver
/// ID (`postgresql.toml`), while a from-source or system-wide install is
/// typically found under the library name (`adbc_driver_postgresql`).
/// [`load_driver_for_scheme`] probes both.
fn adbc_info_for_scheme(scheme: &str) -> Option<crate::reader::registry::AdbcInfo> {
    crate::reader::registry::by_scheme(scheme).and_then(|e| e.adbc)
}

/// Map a ggsql URI scheme to the canonical ADBC driver library name.
pub fn driver_name_for_scheme(scheme: &str) -> Option<&'static str> {
    adbc_info_for_scheme(scheme).map(|info| info.lib_name)
}

/// Environment variable that overrides the ADBC driver for a URI scheme,
/// e.g. `GGSQL_POSTGRES_ADBC_DRIVER=/opt/drivers/libadbc_driver_postgresql.so`.
pub fn driver_env_var(scheme: &str) -> String {
    crate::reader::registry::adbc_driver_env_var(scheme)
}

/// Convert the parsed driver-bound params of a
/// [`ConnUri`](crate::reader::connection::ConnUri) into ADBC database
/// options, without re-splitting the query string. The keys `uri`,
/// `username`, and `password` map to their dedicated ADBC options;
/// everything else passes through as a driver-specific option. Bare flags
/// (no value) have no option equivalent and are skipped.
fn conn_params_to_opts(params: &[(String, Option<String>)]) -> Vec<(OptionDatabase, OptionValue)> {
    params
        .iter()
        .filter_map(|(key, value)| {
            let value = value.as_ref()?;
            let opt_key = match key.as_str() {
                "uri" => OptionDatabase::Uri,
                "username" => OptionDatabase::Username,
                "password" => OptionDatabase::Password,
                other => OptionDatabase::Other(other.to_string()),
            };
            Some((opt_key, OptionValue::String(value.clone())))
        })
        .collect()
}

/// Parse a `k=v&…` query string into ADBC database options. Used for
/// driver-rewritten query strings (BigQuery's Simba grammar) that no longer
/// correspond to the parsed [`ConnUri`](crate::reader::connection::ConnUri);
/// the parsed-params form is [`conn_params_to_opts`].
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
    let info = adbc_info_for_scheme(scheme).ok_or_else(|| {
        GgsqlError::ReaderError(format!("No known ADBC driver for scheme '{}://'", scheme))
    })?;
    let (lib_name, dbc_id) = (info.lib_name, info.dbc_id);
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
    /// The dialect is chosen from the scheme through the registry (see
    /// [`crate::reader::registry::resolve_dialect`]); unknown backends are
    /// an error, never a silent ANSI fallback.
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn = crate::reader::connection::ConnUri::parse(uri)?;
        let scheme = conn.scheme.as_str();
        let body = conn.body.as_str();
        // ggsql's own keys (`cache`, `reader`, `dialect`, cache tuning) and
        // `stmt.`-prefixed keys were lifted out by the parse, so what remains
        // is safe to hand to the driver's own URI/option parsing.
        let query = conn.query_string();
        let query = query.as_str();

        let (driver, opts) = if scheme == "adbc" {
            let driver = if adbc_info_for_scheme(body).is_some() {
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
            (driver, conn_params_to_opts(&conn.params))
        } else {
            let entry = crate::reader::registry::by_scheme(scheme).ok_or_else(|| {
                GgsqlError::ReaderError(format!("No known ADBC driver for scheme '{}://'", scheme))
            })?;
            // Check ADBC support before loading: the env-var override in
            // load_driver_for_scheme can otherwise succeed for schemes with
            // no registry ADBC info (drill, monetdb) and panic here.
            let info = entry.adbc.ok_or_else(|| {
                GgsqlError::ReaderError(format!(
                    "No known ADBC driver for scheme '{scheme}://'. \
                     Use an odbc:// connection string instead."
                ))
            })?;
            let driver = load_driver_for_scheme(scheme)?;
            // The URI handed to the driver must not carry stmt.* or ggsql
            // params — drivers parse their own URI query string and reject
            // unknown keys. BigQuery additionally needs its URI in the
            // driver's Simba grammar, with non-Simba params arriving only as
            // standalone options.
            let (driver_uri, opts_query) =
                if info.driver_uri == crate::reader::registry::DriverUri::BigQuerySimba {
                    bigquery_driver_uri(body, query)
                } else {
                    (
                        driver_uri_for(info.driver_uri, body, &conn.to_uri()),
                        query.to_string(),
                    )
                };
            let mut opts = vec![(OptionDatabase::Uri, OptionValue::String(driver_uri))];
            if info.params_as_options {
                opts.extend(query_params_to_opts(&opts_query));
            }
            (driver, opts)
        };

        let dialect = crate::reader::registry::resolve_dialect(&conn)?;

        let reader = Self::new_with_database_opts(driver, dialect, opts)?
            .with_statement_opts(conn.ggsql.stmt_options.clone());
        // adbc://<driver> with a recognized short name also resolves an entry.
        Ok(
            match crate::reader::registry::by_scheme(if scheme == "adbc" { body } else { scheme }) {
                Some(entry) => reader.with_entry(entry),
                None => reader,
            },
        )
    }
}

/// Compute the URI handed to the driver as the `uri` database option,
/// applying the registry's rewrite rule for the backend.
fn driver_uri_for(kind: crate::reader::registry::DriverUri, body: &str, full_uri: &str) -> String {
    use crate::reader::registry::DriverUri;
    match kind {
        DriverUri::Passthrough => full_uri.to_string(),
        DriverUri::ClickHouseHttp => format!("http://{body}"),
        DriverUri::MySqlGoDsn => {
            let (userinfo, host_db) = match body.rsplit_once('@') {
                Some((u, h)) => (format!("{u}@"), h),
                None => (String::new(), body),
            };
            match host_db.split_once('/') {
                Some((addr, db)) => format!("{userinfo}tcp({addr})/{db}"),
                None => format!("{userinfo}tcp({host_db})"),
            }
        }
        DriverUri::RedshiftAsPostgres => full_uri.replacen("redshift://", "postgres://", 1),
        DriverUri::BigQuerySimba => unreachable!("handled by bigquery_driver_uri"),
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

/// Probe whether an ADBC driver for `scheme` can be loaded, without opening
/// a connection. Used by reader dispatch to decide between ADBC and ODBC.
///
/// Note this loads (dlopens) the driver library and drops it;
/// `from_connection_string` loads it again when the probe succeeds. That
/// double-load is accepted connect-time cost: threading the preloaded
/// driver through dispatch would complicate the probe-then-build flow for
/// no per-query benefit.
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
            let mut stmt = self.new_query_statement(&mut conn, sql)?;
            let reader = match stmt.execute() {
                Ok(reader) => reader,
                Err(e) => {
                    let msg = e.to_string();
                    // Driver-declared quirks (registry `AdbcInfo`): some
                    // drivers signal "DDL without a result set" only through
                    // error text.
                    // - `ddl_empty_result_error` (BigQuery): the statement
                    //   already ran — report an empty frame. Retrying via
                    //   execute_update would re-execute it, which breaks
                    //   non-idempotent DDL ("already exists").
                    // - `ddl_retry_error` (Databricks): the query path cannot
                    //   run resultless statements at all; retry via
                    //   execute_update, the path meant for them.
                    let info = self.entry.and_then(|e| e.adbc);
                    if info
                        .and_then(|i| i.ddl_empty_result_error)
                        .is_some_and(|needle| msg.contains(needle))
                    {
                        return Ok(DataFrame::from_record_batch(RecordBatch::new_empty(
                            std::sync::Arc::new(arrow::datatypes::Schema::empty()),
                        )));
                    }
                    if info
                        .and_then(|i| i.ddl_retry_error)
                        .is_some_and(|needle| msg.contains(needle))
                    {
                        drop(stmt);
                        let mut update_stmt = self.new_query_statement(&mut conn, sql)?;
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
        let mut batch = crate::reader::normalize_result_batch(merged)?;
        if self.dialect.sniff_temporal_strings() {
            batch = crate::reader::sniff_temporal_strings_in_batch(batch)?;
        }
        Ok(DataFrame::from_record_batch(batch))
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        super::validate_table_name(name)?;

        use adbc_core::options::{IngestMode, OptionStatement, OptionValue};
        use adbc_core::Optionable;

        // Zero-row frames are fine: the CREATE below still runs, leaving an
        // empty table — the caching reader relies on this to memoize empty
        // query results.
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
        // with varying ingest-option support — drivers that reject the
        // `IngestMode` option key (`set_option` returns `NotFound`) are
        // silently tolerated below.
        let schema = batch.schema();
        if replace {
            let drop_sql = self.dialect.drop_table_sql(name);
            self.new_query_statement(&mut conn, &drop_sql)?
                .execute_update()
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC execute_update DROP: {}", e)))?;
        }

        let create_sql = crate::reader::create_table_sql(name, &schema, self.dialect)?;
        self.new_query_statement(&mut conn, &create_sql)?
            .execute_update()
            .map_err(|e| GgsqlError::ReaderError(format!("ADBC execute_update CREATE: {}", e)))?;

        // Track the table in our set as soon as CREATE succeeds — BEFORE the
        // potentially-multi-batch ingest loop. If a bind or execute_update
        // fails mid-way, the (partial) table still exists on the server;
        // having the name tracked lets the caller `unregister()` to clean
        // up, and a subsequent `register(name, ..., replace=true)` will
        // drop-and-recreate. Without this, a mid-ingest failure would leave
        // an orphan table the reader can't reach.
        self.registered_tables.note_registered(name);

        if batch.num_rows() > 0 {
            // Ingest, with one schema-alignment retry: drivers that validate
            // the batch against the table schema (e.g. the Foundry datafusion
            // driver >=0.27) reject mismatches — DataFusion surfaces VARCHAR
            // as Utf8View while our batches are Utf8. On that specific
            // failure, align the batch to the driver-reported table schema
            // and retry. Drivers without ingest-time validation succeed on
            // the first attempt and never touch the schema path.
            let mut aligned: Option<arrow::record_batch::RecordBatch> = None;
            let mut attempts = 0;
            loop {
                let attempt_batch = aligned.as_ref().unwrap_or(&batch).clone();
                let mut stmt = conn
                    .new_statement()
                    .map_err(|e| GgsqlError::ReaderError(format!("ADBC new_statement: {}", e)))?;
                stmt.set_option(
                    OptionStatement::TargetTable,
                    OptionValue::String(name.to_string()),
                )
                .map_err(|e| GgsqlError::ReaderError(format!("ADBC set TargetTable: {}", e)))?;
                // Tell the driver this is an append into the table we just
                // CREATEd above. Compliant ADBC drivers (e.g. the Apache
                // SQLite driver) default `IngestMode` to `Create` when only
                // `TargetTable` is set, which would then fail because the
                // table already exists. DataFusion drivers don't expose this
                // option key and return `Status::NotFound` from `set_option`;
                // that's expected for DataFusion's bind path (it appends by
                // default), so swallow it and continue rather than failing
                // register().
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
                stmt.bind(attempt_batch)
                    .map_err(|e| GgsqlError::ReaderError(format!("ADBC bind: {}", e)))?;
                match stmt.execute_update() {
                    Ok(_) => break,
                    Err(e) => {
                        let schema_mismatch = self
                            .entry
                            .and_then(|e| e.adbc)
                            .and_then(|i| i.ingest_schema_align_error)
                            .is_some_and(|needle| e.to_string().contains(needle));
                        if attempts == 0 && schema_mismatch {
                            attempts += 1;
                            match conn.get_table_schema(None, None, name) {
                                Ok(target) => {
                                    aligned = Some(align_batch_to_schema(&batch, &target)?);
                                    continue;
                                }
                                Err(schema_err) => {
                                    return Err(GgsqlError::ReaderError(format!(
                                        "ADBC execute_update: {e} — and the \
                                         schema-alignment fallback failed: {schema_err}"
                                    )));
                                }
                            }
                        }
                        return Err(GgsqlError::ReaderError(format!(
                            "ADBC execute_update: {} — \
                             table left on server; call unregister() to drop it \
                             or register() with replace=true to retry",
                            e
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        if !self.registered_tables.is_registered(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }
        let sql = self.dialect.drop_table_sql(name);
        // Ignore the returned DataFrame — DROP TABLE has no result rows.
        self.execute_sql(&sql)?;
        self.registered_tables.note_unregistered(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<crate::reader::Spec> {
        crate::reader::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn SqlDialect {
        self.dialect
    }
}

/// Cast `batch` columns to `target`'s field types (matched by position) so
/// drivers that validate the ingest schema accept the append — e.g. the
/// Foundry datafusion driver creates VARCHAR as Utf8View while our batches
/// are Utf8. Columns whose types already agree pass through untouched; a
/// column-count mismatch returns the batch unchanged and lets the driver's
/// own validation report the problem.
fn align_batch_to_schema(
    batch: &arrow::record_batch::RecordBatch,
    target: &arrow::datatypes::Schema,
) -> Result<arrow::record_batch::RecordBatch> {
    use std::sync::Arc;

    if target.fields().len() != batch.num_columns() {
        return Ok(batch.clone());
    }
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in target.fields().iter().enumerate() {
        let col = batch.column(i);
        if col.data_type() == field.data_type() {
            columns.push(col.clone());
        } else {
            columns.push(arrow::compute::cast(col, field.data_type()).map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "AdbcReader::register: cannot align column '{}' ({:?}) to \
                     the table's {:?}: {}",
                    field.name(),
                    col.data_type(),
                    field.data_type(),
                    e
                ))
            })?);
        }
    }
    arrow::record_batch::RecordBatch::try_new(Arc::new(target.clone()), columns).map_err(|e| {
        GgsqlError::ReaderError(format!(
            "AdbcReader::register: schema alignment failed: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_without_adbc_info_errors_instead_of_panicking() {
        // MonetDB/Drill have no registry ADBC info; even an env override
        // must produce an error, not a panic on the missing entry.
        for uri in ["monetdb://localhost:50000/db", "drill://localhost:8047"] {
            let err = AdbcReader::from_connection_string(uri)
                .err()
                .expect("schemes without ADBC info must fail");
            assert!(
                err.to_string().contains("No known ADBC driver"),
                "{uri}: unexpected error: {err}"
            );
        }
    }

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
        use crate::reader::registry::DriverUri;
        // Credentials must travel as dedicated options, not in the URL.
        assert_eq!(
            driver_uri_for(
                DriverUri::ClickHouseHttp,
                "localhost:8123",
                "clickhouse://localhost:8123?username=default&password=secret"
            ),
            "http://localhost:8123"
        );
        assert_eq!(
            driver_uri_for(
                DriverUri::Passthrough,
                "u:p@h/db",
                "postgres://u:p@h/db?sslmode=disable"
            ),
            "postgres://u:p@h/db?sslmode=disable"
        );
    }

    #[test]
    fn driver_uri_translates_mysql_to_go_dsn() {
        use crate::reader::registry::DriverUri;
        assert_eq!(
            driver_uri_for(
                DriverUri::MySqlGoDsn,
                "root:pw@localhost:3306/ggsql",
                "mysql://root:pw@localhost:3306/ggsql"
            ),
            "root:pw@tcp(localhost:3306)/ggsql"
        );
        assert_eq!(
            driver_uri_for(
                DriverUri::MySqlGoDsn,
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
        assert_eq!(
            bigquery_driver_uri("my-proj", ""),
            ("bigquery:///my-proj".to_string(), String::new())
        );
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
                crate::reader::registry::DriverUri::RedshiftAsPostgres,
                "u:p@h:5439/db",
                "redshift://u:p@h:5439/db?sslmode=disable"
            ),
            "postgres://u:p@h:5439/db?sslmode=disable"
        );
    }

    #[test]
    fn uri_parsing_drivers_reject_standalone_options() {
        let params_as_options = |scheme: &str| {
            crate::reader::registry::by_scheme(scheme)
                .and_then(|e| e.adbc)
                .map(|i| i.params_as_options)
        };
        assert_eq!(params_as_options("mssql"), Some(false));
        assert_eq!(params_as_options("databricks"), Some(false));
        assert_eq!(params_as_options("spark"), Some(false));
        assert_eq!(params_as_options("druid"), Some(false));
        assert_eq!(params_as_options("postgres"), Some(true));
        assert_eq!(params_as_options("clickhouse"), Some(true));
        assert_eq!(params_as_options("exasol"), Some(true));
    }

    #[test]
    fn driver_names_include_dbc_manifest_id() {
        // dbc names manifests by short ID (`postgresql.toml`), which differs
        // from the library name — both must be available for probing, or a
        // `dbc install`-based setup is never found (first seen as a CI
        // failure where dbc-installed drivers were not discovered).
        let names = |scheme: &str| adbc_info_for_scheme(scheme).map(|i| (i.lib_name, i.dbc_id));
        assert_eq!(
            names("postgres"),
            Some(("adbc_driver_postgresql", "postgresql"))
        );
        assert_eq!(
            names("redshift"),
            Some(("adbc_driver_postgresql", "postgresql"))
        );
        assert_eq!(names("trino"), Some(("adbc_driver_trino", "trino")));
        assert_eq!(names("mariadb"), Some(("adbc_driver_mysql", "mysql")));
        assert_eq!(names("nosuch"), None);
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
        AdbcReader::new_with_database_opts(
            driver,
            &SqliteDialect,
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
