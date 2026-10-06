//! The supported-database registry: ggsql's single source of per-backend
//! connection knowledge.
//!
//! Every backend ggsql knows how to connect to has one [`DatabaseEntry`]
//! here carrying its URI schemes, display name, DBMS/driver detection
//! patterns, dialect constructor, ADBC driver details, and ODBC synthesis
//! hints. Dispatch (`connection.rs`), the ADBC reader, and the Jupyter
//! kernel all consult this table rather than keeping their own lookups.

use super::dialects::*;
use super::AnsiDialect;

/// How the URI handed to an ADBC driver is derived from the ggsql URI.
///
/// ggsql's scheme selects the driver, but the URI must use the scheme and
/// grammar the driver itself speaks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriverUri {
    /// The full ggsql URI passes through unchanged.
    Passthrough,
    /// Rebuilt as `http://<body>`, dropping userinfo and query params: the
    /// ClickHouse driver connects over the HTTP interface, ignores userinfo
    /// (credentials arrive as dedicated options), and forwards URL query
    /// parameters to the server as ClickHouse *settings*.
    ClickHouseHttp,
    /// Translated to a go-sql-driver DSN, `user[:pass]@tcp(host:port)/db`;
    /// query params reach the driver as options, not in the DSN.
    MySqlGoDsn,
    /// `redshift://` rewritten to `postgres://`: the Redshift driver is the
    /// PostgreSQL driver, whose pgx-based URI parsing rejects redshift://.
    /// The full URI is rewritten so query params (pgx settings) survive.
    RedshiftAsPostgres,
    /// ggsql's `bigquery://<project>[/<dataset>]` translated to the Simba
    /// grammar the Foundry BigQuery driver parses. Complex enough to live
    /// in `adbc.rs` (`bigquery_driver_uri`).
    BigQuerySimba,
}

/// ADBC driver details for a backend.
#[derive(Clone, Copy)]
pub struct AdbcInfo {
    /// Canonical driver library name (`adbc_driver_postgresql`).
    pub lib_name: &'static str,
    /// dbc manifest ID (`postgresql`): `dbc install <id>` writes a manifest
    /// named after this short ID, while a from-source install is typically
    /// found under `lib_name`. Loading probes both.
    pub dbc_id: &'static str,
    /// How to rewrite the ggsql URI for the driver.
    pub driver_uri: DriverUri,
    /// Whether `?k=v` query params are *also* passed as standalone database
    /// options. (They always remain in the `uri` option except where the
    /// rewrite strips them.) Some drivers reject params arriving a second
    /// way: MSSQL fails with "Unknown database option
    /// 'TrustServerCertificate'", Databricks with "cannot specify both URI
    /// and individual connection options", and Druid with "Unsupported
    /// option: Other(\"tls\")" — their params stay in the URI only.
    pub params_as_options: bool,
    /// Error substring meaning "a DDL/DML statement without a result set
    /// ran fine but the read failed" — report an empty frame instead of an
    /// error (BigQuery: "job has no destination table to read"). Matching
    /// on driver error text is fragile, so it is declared here per driver
    /// rather than hard-coded in the reader.
    pub ddl_empty_result_error: Option<&'static str>,
    /// Error substring meaning the driver's query path cannot execute
    /// statements without a result set; retry those via `execute_update`
    /// (Databricks: "schema bytes are empty").
    pub ddl_retry_error: Option<&'static str>,
    /// Error substring on ingest meaning the target table's schema differs
    /// from the batch; retry once with the batch aligned to the target
    /// schema (DataFusion: "different schema").
    pub ingest_schema_align_error: Option<&'static str>,
}

impl AdbcInfo {
    /// No driver-specific error-text quirks; the `adbc!` macro spreads this.
    const NO_QUIRKS: Self = Self {
        lib_name: "",
        dbc_id: "",
        driver_uri: DriverUri::Passthrough,
        params_as_options: false,
        ddl_empty_result_error: None,
        ddl_retry_error: None,
        ingest_schema_align_error: None,
    };
}

/// A DBMS-name/driver-string detection pattern.
pub enum DetectPattern {
    /// Matches when the lowercased text contains this substring.
    Contains(&'static str),
    /// Matches when the lowercased text contains *all* these substrings.
    All(&'static [&'static str]),
}

impl DetectPattern {
    fn matches(&self, lower: &str) -> bool {
        match self {
            DetectPattern::Contains(s) => lower.contains(s),
            // Each needle must appear at a word boundary (start of the
            // string or right after a non-alphanumeric character), so
            // "ora" matches "Oracle ODBC Driver" but the "ora" in
            // "Teradata Corporation ODBC Driver" does not.
            DetectPattern::All(ss) => ss.iter().all(|s| contains_word_prefix(lower, s)),
        }
    }
}

/// `haystack` contains `needle` starting at a word boundary.
fn contains_word_prefix(haystack: &str, needle: &str) -> bool {
    haystack
        .match_indices(needle)
        .any(|(i, _)| i == 0 || !haystack.as_bytes()[i - 1].is_ascii_alphanumeric())
}

/// An in-process reader ggsql ships for a backend, preferred over external
/// drivers when its cargo feature is compiled in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NativeReader {
    DuckDb,
    Sqlite,
}

/// One supported database backend.
pub struct DatabaseEntry {
    /// Canonical URI scheme (`postgres`).
    pub scheme: &'static str,
    /// Additional accepted schemes (`postgresql`).
    pub aliases: &'static [&'static str],
    /// Human-readable name for UIs (`PostgreSQL`).
    pub display_name: &'static str,
    /// DBMS/driver detection patterns. Entries are scanned in registry
    /// order, most specific first (SQL Server before anything containing
    /// "sql", Redshift before Postgres).
    pub detect: &'static [DetectPattern],
    /// The backend's SQL dialect. Dialects are stateless unit structs, so
    /// the registry shares one static instance rather than boxing per
    /// lookup.
    dialect: crate::reader::DialectRef,
    /// ADBC driver details; `None` when no usable dedicated driver exists
    /// (Drill, MonetDB) — dispatch falls through to ODBC.
    pub adbc: Option<AdbcInfo>,
    /// ODBC synthesis: a `DBQ=` query param (Oracle/DB2-style server
    /// address) fully specifies where to connect, suppressing
    /// Server/Port/Database synthesis — Oracle ODBC rejects a connection
    /// string that mixes the two vocabularies.
    pub odbc_dbq_style: bool,
    /// ODBC fetch batch size override. Oracle ODBC rejects block cursors
    /// (SQL_ATTR_ROW_ARRAY_SIZE > 1) with HY090 at SQLFetch time, and the
    /// failed fetch leaves the cursor unusable — fetch row-by-row (1).
    pub odbc_row_array_size: Option<usize>,
    /// Bind NUMERIC/DECIMAL as double even when scale is 0 and the value
    /// fits an integer: Oracle ODBC fails with HY090 converting SQL_DECIMAL
    /// to SQL_C_SLONG/SBIGINT. Elsewhere integer-scale numerics keep their
    /// Int64 type (and precision past 2^53).
    pub odbc_numeric_as_double: bool,
    /// In-process reader to prefer when its cargo feature is compiled in
    /// (duckdb, sqlite). `None` for backends reached only through external
    /// ADBC/ODBC drivers.
    pub native_reader: Option<NativeReader>,
}

impl DatabaseEntry {
    /// The backend's SQL dialect (shared static instance).
    pub fn dialect(&self) -> crate::reader::DialectRef {
        self.dialect
    }

    /// All URI schemes naming this backend, canonical first.
    pub fn schemes(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.scheme).chain(self.aliases.iter().copied())
    }
}

macro_rules! entry {
    ($scheme:literal, $aliases:expr, $display:literal, $detect:expr, $dialect:expr, $adbc:expr) => {
        DatabaseEntry {
            scheme: $scheme,
            aliases: $aliases,
            display_name: $display,
            detect: $detect,
            dialect: $dialect,
            adbc: $adbc,
            odbc_dbq_style: false,
            odbc_row_array_size: None,
            odbc_numeric_as_double: false,
            native_reader: None,
        }
    };
}

macro_rules! adbc {
    ($lib:literal, $id:literal, $uri:expr, $opts:expr) => {
        Some(AdbcInfo {
            lib_name: $lib,
            dbc_id: $id,
            driver_uri: $uri,
            params_as_options: $opts,
            ..AdbcInfo::NO_QUIRKS
        })
    };
    ($lib:literal, $id:literal, $uri:expr, $opts:expr, $($k:ident: $v:expr),+ $(,)?) => {
        Some(AdbcInfo {
            lib_name: $lib,
            dbc_id: $id,
            driver_uri: $uri,
            params_as_options: $opts,
            $($k: $v),+,
            ..AdbcInfo::NO_QUIRKS
        })
    };
}

use DetectPattern::Contains as Has;
use DriverUri::*;

/// All supported backends. **Order matters for detection**: entries are
/// scanned top to bottom, so more specific patterns must precede generic
/// ones (`microsoft sql server` before anything matching `sql`, Redshift
/// before Postgres).
pub static REGISTRY: &[DatabaseEntry] = &[
    entry!(
        "mssql",
        &["sqlserver"],
        "SQL Server",
        &[
            Has("microsoft sql server"),
            Has("msodbcsql"),
            Has("sql server"),
            Has("sqlserver"),
            Has("mssql"),
        ],
        &MssqlDialect,
        adbc!("adbc_driver_mssql", "mssql", Passthrough, false)
    ),
    entry!(
        "redshift",
        &[],
        "Redshift",
        &[Has("redshift")],
        &RedshiftDialect,
        adbc!(
            "adbc_driver_postgresql",
            "postgresql",
            RedshiftAsPostgres,
            true
        )
    ),
    entry!(
        "postgres",
        &["postgresql"],
        "PostgreSQL",
        &[Has("postgres"), Has("psql")],
        &PostgresDialect,
        adbc!("adbc_driver_postgresql", "postgresql", Passthrough, true)
    ),
    entry!(
        "mysql",
        &["mariadb"],
        "MySQL",
        &[Has("mariadb"), Has("mysql")],
        &MySqlDialect,
        adbc!("adbc_driver_mysql", "mysql", MySqlGoDsn, true)
    ),
    entry!(
        "snowflake",
        &[],
        "Snowflake",
        &[Has("snowflake")],
        &SnowflakeDialect,
        adbc!("adbc_driver_snowflake", "snowflake", Passthrough, true)
    ),
    entry!(
        "bigquery",
        &[],
        "BigQuery",
        &[Has("bigquery")],
        &BigQueryDialect,
        adbc!(
            "adbc_driver_bigquery",
            "bigquery",
            BigQuerySimba,
            true,
            ddl_empty_result_error: Some("no destination table to read")
        )
    ),
    entry!(
        "databricks",
        &["spark"],
        "Databricks",
        &[Has("databricks"), Has("spark")],
        &DatabricksDialect,
        adbc!(
            "adbc_driver_databricks",
            "databricks",
            Passthrough,
            false,
            ddl_retry_error: Some("schema bytes are empty")
        )
    ),
    entry!(
        "clickhouse",
        &[],
        "ClickHouse",
        &[Has("clickhouse")],
        &ClickHouseDialect,
        adbc!("adbc_driver_clickhouse", "clickhouse", ClickHouseHttp, true)
    ),
    DatabaseEntry {
        odbc_dbq_style: true,
        odbc_row_array_size: Some(1),
        odbc_numeric_as_double: true,
        ..entry!(
            "oracle",
            &[],
            "Oracle",
            &[Has("oracle"), DetectPattern::All(&["ora", "driver"])],
            &OracleDialect,
            adbc!("adbc_driver_oracle", "oracle", Passthrough, true)
        )
    },
    entry!(
        "trino",
        &[],
        "Trino",
        &[Has("trino")],
        &TrinoDialect,
        adbc!("adbc_driver_trino", "trino", Passthrough, true)
    ),
    entry!(
        "exasol",
        &[],
        "Exasol",
        &[Has("exasol"), Has("exaodbc")],
        &ExasolDialect,
        adbc!("adbc_driver_exasol", "exasol", Passthrough, true)
    ),
    entry!(
        "monetdb",
        &[],
        "MonetDB",
        &[Has("monet")],
        &MonetDbDialect,
        None
    ),
    entry!(
        "druid",
        &[],
        "Apache Druid",
        &[Has("druid")],
        &DruidDialect,
        adbc!("adbc_driver_druid", "druid", Passthrough, false)
    ),
    entry!(
        "drill",
        &[],
        "Apache Drill",
        &[Has("drill")],
        &DrillDialect,
        None
    ),
    entry!(
        "datafusion",
        &[],
        "DataFusion",
        &[Has("datafusion")],
        &DataFusionDialect,
        adbc!(
            "adbc_driver_datafusion",
            "datafusion",
            Passthrough,
            true,
            ingest_schema_align_error: Some("different schema")
        )
    ),
    DatabaseEntry {
        native_reader: Some(NativeReader::DuckDb),
        ..entry!(
            "duckdb",
            &[],
            "DuckDB",
            &[Has("duckdb")],
            &DuckDbDialect,
            adbc!("adbc_driver_duckdb", "duckdb", Passthrough, true)
        )
    },
    DatabaseEntry {
        native_reader: Some(NativeReader::Sqlite),
        ..entry!(
            "sqlite",
            &[],
            "SQLite",
            &[Has("sqlite")],
            &SqliteDialect,
            adbc!("adbc_driver_sqlite", "sqlite", Passthrough, true)
        )
    },
    // Flight SQL is a wire protocol, not a database: any dialect choice is
    // a guess, so the ANSI dialect is assigned explicitly here (the one
    // place ANSI is the right answer rather than a silent fallback).
    entry!(
        "flightsql",
        &[],
        "Flight SQL",
        &[Has("flightsql"), DetectPattern::All(&["flight", "sql"])],
        &AnsiDialect,
        adbc!("adbc_driver_flightsql", "flightsql", Passthrough, true)
    ),
];

/// Look up a backend by URI scheme (canonical or alias), case-insensitively.
pub fn by_scheme(scheme: &str) -> Option<&'static DatabaseEntry> {
    let lower = scheme.to_ascii_lowercase();
    REGISTRY.iter().find(|e| e.schemes().any(|s| s == lower))
}

/// Detect the backend from a DBMS name and/or driver hint (ODBC driver
/// name, ADBC driver name). The DBMS name is checked first; matching is
/// case-insensitive substring matching in registry order.
pub fn detect(
    dbms_name: Option<&str>,
    driver_hint: Option<&str>,
) -> Option<&'static DatabaseEntry> {
    for text in [dbms_name, driver_hint].into_iter().flatten() {
        let lower = text.to_lowercase();
        if let Some(e) = REGISTRY
            .iter()
            .find(|e| e.detect.iter().any(|p| p.matches(&lower)))
        {
            return Some(e);
        }
    }
    None
}

/// `scheme://` list for error messages: every supported URI scheme
/// (canonical names; registry aliases omitted).
pub fn supported_schemes() -> String {
    let mut schemes: Vec<String> = vec!["odbc".into(), "adbc".into()];
    schemes.extend(REGISTRY.iter().map(|e| e.scheme.into()));
    schemes
        .into_iter()
        .map(|s| format!("{s}://"))
        .collect::<Vec<_>>()
        .join(", ")
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

/// Environment variable overriding the ADBC driver for a URI scheme, e.g.
/// `GGSQL_POSTGRES_ADBC_DRIVER=/opt/drivers/libadbc_driver_postgresql.so`.
pub fn adbc_driver_env_var(scheme: &str) -> String {
    format!("GGSQL_{}_ADBC_DRIVER", env_var_scheme(scheme))
}

/// Environment variable specifying the ODBC driver for a URI scheme.
pub fn odbc_driver_env_var(scheme: &str) -> String {
    format!("GGSQL_{}_ODBC_DRIVER", env_var_scheme(scheme))
}

/// Resolve a `dialect=` override value (`ansi` or any registry scheme) to a
/// dialect. This is the explicit escape hatch for backends ggsql doesn't
/// know: unknown backends are an error unless the user pins a dialect.
pub fn dialect_override(name: &str) -> Option<crate::reader::DialectRef> {
    if name.eq_ignore_ascii_case("ansi") {
        return Some(&AnsiDialect);
    }
    by_scheme(name).map(|e| e.dialect())
}

/// Error for an unrecognized `dialect=` override value, naming the escape
/// hatches. Shared so every dispatch path reports the same message.
pub fn unknown_dialect_error(name: &str) -> crate::GgsqlError {
    crate::GgsqlError::ReaderError(format!(
        "Unknown dialect '{name}'. Use dialect=ansi or any supported scheme \
         (postgres, mysql, …)."
    ))
}

/// Resolve the SQL dialect for a parsed connection URI: an explicit
/// `dialect=` override wins; otherwise the scheme (or, for
/// `adbc://<driver>`, the driver name) resolves through the registry.
/// Unknown backends are an error pointing at the override — never a silent
/// ANSI fallback.
pub fn resolve_dialect(
    conn: &crate::reader::connection::ConnUri,
) -> crate::Result<crate::reader::DialectRef> {
    if let Some(name) = &conn.ggsql.dialect {
        return dialect_override(name).ok_or_else(|| unknown_dialect_error(name));
    }
    if conn.scheme != "adbc" {
        return by_scheme(&conn.scheme).map(|e| e.dialect()).ok_or_else(|| {
            crate::GgsqlError::ReaderError(format!(
                "Unsupported connection scheme '{}://'. Supported: {}",
                conn.scheme,
                supported_schemes()
            ))
        });
    }
    let body = conn.body.as_str();
    if let Some(entry) = by_scheme(body) {
        return Ok(entry.dialect());
    }
    detect_or_err(None, Some(body))
}

/// Detect a dialect from an ODBC DBMS name and/or driver string, or error
/// naming what was seen. Unknown backends are an **error** — silently
/// falling back to ANSI produced broken SQL too often. The escape hatch is
/// a `dialect=ansi` (or `dialect=<scheme>`) parameter; see
/// [`dialect_override`].
pub fn detect_or_err(
    dbms_name: Option<&str>,
    driver_hint: Option<&str>,
) -> crate::Result<crate::reader::DialectRef> {
    detect(dbms_name, driver_hint)
        .map(|e| e.dialect())
        .ok_or_else(|| {
            crate::GgsqlError::ReaderError(format!(
                "Unrecognized database backend (DBMS name: {}, driver: {}). \
                 ggsql does not know which SQL dialect to use. If the backend \
                 is close to a supported one, pin the dialect explicitly with \
                 a `dialect=<scheme>` parameter (e.g. dialect=postgres), or use \
                 dialect=ansi for generic ANSI SQL.",
                dbms_name.unwrap_or("<none>"),
                driver_hint.unwrap_or("<none>"),
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_resolves_both_ways() {
        for e in REGISTRY {
            assert!(by_scheme(e.scheme).is_some(), "scheme {}", e.scheme);
            for alias in e.aliases {
                let found = by_scheme(alias).unwrap();
                assert!(
                    std::ptr::eq(found, e),
                    "alias {alias} should resolve to {}",
                    e.scheme
                );
            }
        }
        assert!(by_scheme("PostgreSQL").is_some(), "case-insensitive");
        assert!(by_scheme("nosuchdb").is_none());
    }

    #[test]
    fn detect_ordering_is_most_specific_first() {
        assert_eq!(
            detect(Some("Microsoft SQL Server"), None).unwrap().scheme,
            "mssql"
        );
        assert_eq!(detect(None, Some("msodbcsql18")).unwrap().scheme, "mssql");
        assert_eq!(
            detect(Some("Amazon Redshift"), None).unwrap().scheme,
            "redshift"
        );
        assert_eq!(detect(Some("PostgreSQL"), None).unwrap().scheme, "postgres");
        assert_eq!(
            detect(Some("Unknown DBMS"), Some("PostgreSQL Unicode"))
                .unwrap()
                .scheme,
            "postgres"
        );
        assert!(detect(Some("mystery-db"), None).is_none());
    }

    #[test]
    fn dialect_override_accepts_ansi_and_schemes() {
        assert!(dialect_override("ansi").is_some());
        assert!(dialect_override("ANSI").is_some());
        assert!(dialect_override("postgres").is_some());
        assert!(dialect_override("nosuch").is_none());
    }
}
