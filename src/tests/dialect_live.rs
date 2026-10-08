//! Live-backend dialect conformance tests (Tier 2).
//!
//! Each remote backend is enabled by setting an environment variable holding
//! a ggsql connection URI; without it the test prints a skip note and passes,
//! so the suite is safe to run unconditionally:
//!
//! - `GGSQL_TEST_URI_POSTGRES`   e.g. `postgres://postgres:postgres@localhost:5432/ggsql`
//! - `GGSQL_TEST_URI_TRINO`      e.g. `trino://localhost:8080/memory/default?username=test`
//! - `GGSQL_TEST_URI_CLICKHOUSE` e.g. `clickhouse://localhost:8123?username=default&password=x`
//! - `GGSQL_TEST_URI_MYSQL`      e.g. `mysql://root:x@localhost:3306/ggsql`
//! - `GGSQL_TEST_URI_MARIADB`    e.g. `mariadb://root:x@localhost:3306/ggsql`
//! - `GGSQL_TEST_URI_MSSQL`      e.g. `mssql://sa:x@localhost:1433/master?TrustServerCertificate=true`
//! - `GGSQL_TEST_URI_EXASOL`     e.g. `exasol://sys:exasol@localhost:8563/?tls=true&validateservercertificate=0`
//! - `GGSQL_TEST_URI_SQLITE`     e.g. `sqlite://:memory:` (embedded reader, no server)
//! - `GGSQL_TEST_URI_DUCKDB`     e.g. `duckdb://memory` (embedded reader, no server)
//! - `GGSQL_TEST_URI_SNOWFLAKE`  e.g. `snowflake://user:pass@account/db/schema?warehouse=x`
//! - `GGSQL_TEST_URI_BIGQUERY`   e.g. `bigquery://<project-id>/<dataset>?
//!   bigquery.auth_type=json_credential_file&
//!   bigquery.auth.credentials=/path/key.json&
//!   stmt.bigquery.query.default_dataset_id=<dataset>` — the driver never
//!   sends the connection's dataset as job defaultDataset, so the statement
//!   option is required. Validated against the BigQuery sandbox, which
//!   rejects DML, so this leg populates via register() load jobs.
//! - `GGSQL_TEST_URI_DATABRICKS` e.g. `databricks://token:x@host/sql/1.0/warehouses/id?catalog=x&schema=y`
//!
//! The snowflake/bigquery/databricks cases are Tier 3: they need real cloud
//! credentials and run only in the nightly .github/workflows/dialect-live-cloud.yml
//! workflow, whose repository secrets carry the URIs.
//! - `GGSQL_TEST_URI_REDSHIFT`   e.g. `redshift://u:p@localhost:5439/db` (CI runs it
//!   against a PostgreSQL container: the Foundry "redshift" driver is the
//!   PostgreSQL driver, so this exercises RedshiftDialect end-to-end over a
//!   wire-compatible server)
//! - `GGSQL_TEST_URI_ODBC`       e.g. `postgres://u:p@localhost:5432/db?reader=odbc&Driver={PostgreSQL Unicode}`
//!   (CI runs it against a PostgreSQL container over psqlODBC: the
//!   `?reader=odbc` query forces the generic ODBC fallback, exercising
//!   OdbcReader and the connection-string synthesis instead of the ADBC
//!   driver)
//! - `GGSQL_TEST_URI_MONETDB`    e.g. `monetdb://u:p@localhost:50000/db?reader=odbc&DSN=ggsql-monetdb`
//!   (CI runs it against a MonetDB container over the MonetDB ODBC driver:
//!   no usable ADBC driver exists for MonetDB, so the `?reader=odbc` query
//!   forces the ODBC fallback with the DSN registered by the workflow step)
//! - `GGSQL_TEST_URI_ORACLE`     e.g. `oracle://u:p@localhost:1521/XEPDB1?reader=odbc&Driver={Oracle}&DBQ=<tns-descriptor>`
//!   (the Foundry oracle ADBC driver is Columnar-commercial, so the leg
//!   runs against a gvenzl/oracle-xe container over Instant Client ODBC,
//!   forced by `?reader=odbc`; Driver + DBQ go in the URI because unixODBC's
//!   DSN attribute mapping never delivers DBQ to the Oracle driver)
//! - `GGSQL_TEST_URI_DRUID`      e.g. `druid://localhost:8082?tls=false`
//!   (tls=false is required against a plaintext broker: the Foundry driver
//!   defaults to https)
//!   (CI runs it against the vendor-supported docker-compose cluster
//!   (zookeeper + metadata postgres + one container per Druid service;
//!   the 37.x image is distroless, so the classic all-in-one quickstart
//!   cannot run) through the Foundry `druid` ADBC driver (prerelease).
//!   Druid has no DDL, so the start script creates and populates the
//!   datasource via an MSQ INSERT job seeded from the shared fixture CSV,
//!   and DruidDialect's `requires_cache` wraps the reader in a sqlite
//!   cache that hosts the battery's derived tables)
//!
//! The DataFusion case runs in-process via the Foundry ADBC driver and
//! needs no container, so one non-DuckDB engine always runs in CI.
//!
//! The battery itself — case list, per-case assertions, and the dataset —
//! lives in `src/tests/battery/mod.rs`, shared with the Tier 1 golden
//! tests (`src/execute/golden.rs`) so the two tiers cannot drift. This
//! file only adds the live-specific plumbing: per-scheme DDL templates,
//! connection setup, and value-level assertions on real results.

#[path = "battery/mod.rs"]
mod battery;

use ggsql::reader::connection::reader_from_uri;
use ggsql::reader::{ColumnInfo, Reader, Spec, SqlDialect, TableInfo};
use ggsql::{DataFrame, Result};

const TABLE: &str = battery::TABLE;

/// A `Reader` wrapper that records every statement the pipeline issues, so
/// a failing battery case can dump the exact SQL and re-run individual
/// statements for diagnosis.
struct SqlSpy<'a> {
    inner: &'a dyn Reader,
    log: std::cell::RefCell<Vec<String>>,
}

impl<'a> SqlSpy<'a> {
    fn new(inner: &'a dyn Reader) -> Self {
        Self {
            inner,
            log: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn take_log(&self) -> Vec<String> {
        self.log.borrow().clone()
    }
}

impl Reader for SqlSpy<'_> {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        self.log.borrow_mut().push(sql.to_string());
        self.inner.execute_sql(sql)
    }
    // Every defaulted method must delegate: routing e.g. execute_sql_cached
    // through the default (which calls execute_sql) would bypass a wrapping
    // CachingReader and send cache-dialect SQL to the remote server.
    fn execute_sql_cached(&self, sql: &str) -> Result<DataFrame> {
        // Log the logical statement, but delegate execution so a wrapping
        // CachingReader still routes it through the cache.
        self.log.borrow_mut().push(sql.to_string());
        self.inner.execute_sql_cached(sql)
    }
    fn materialize_table(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Result<()> {
        self.inner.materialize_table(name, column_aliases, body_sql)
    }
    fn caches_sources(&self) -> bool {
        self.inner.caches_sources()
    }
    fn clear_cache(&self) -> Result<()> {
        self.inner.clear_cache()
    }
    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.inner.register(name, df, replace)
    }
    fn unregister(&self, name: &str) -> Result<()> {
        self.inner.unregister(name)
    }
    fn execute(&self, query: &str) -> Result<Spec> {
        ggsql::reader::execute_with_reader(self, query)
    }
    fn dialect(&self) -> &dyn SqlDialect {
        self.inner.dialect()
    }
    fn list_catalogs(&self) -> Result<Vec<String>> {
        self.inner.list_catalogs()
    }
    fn list_schemas(&self, c: &str) -> Result<Vec<String>> {
        self.inner.list_schemas(c)
    }
    fn list_tables(&self, c: &str, s: &str) -> Result<Vec<TableInfo>> {
        self.inner.list_tables(c, s)
    }
    fn list_columns(&self, c: &str, s: &str, t: &str) -> Result<Vec<ColumnInfo>> {
        self.inner.list_columns(c, s, t)
    }
}

// The dataset (ids, values, groups, days) is battery::ROWS; the functions
// here only spell it per backend. Beyond id/val/grp the table carries:
// - `day` (a real date column, except on SQLite where it stays TEXT):
//   temporal literal/cast paths, exercised by the binned-scale cases,
// - `mixed Case` (float): a column whose name requires quoting, exercising
//   the dialects' quote_ident handling through stats that pass raw column
//   names (quantiles) and PARTITION BY.
fn insert_sql(scheme: &str, table: &str) -> String {
    let row = |r: &battery::Row| {
        format!(
            "({}, {}, '{}', {}, {})",
            r.id,
            r.val,
            r.grp,
            date_literal(scheme, r.day_epoch),
            r.mixed
        )
    };
    let rows: Vec<String> = battery::ROWS.iter().map(row).collect();
    if scheme == "oracle" {
        // Oracle has no multi-row VALUES; INSERT ALL ... SELECT * FROM dual
        // is the single-statement equivalent.
        let into: Vec<String> = rows
            .iter()
            .map(|r| format!("INTO {table} VALUES {r}"))
            .collect();
        return format!("INSERT ALL {} SELECT * FROM dual", into.join(" "));
    }
    format!("INSERT INTO {table} VALUES {}", rows.join(", "))
}

/// A date literal for a fixture day in the spelling the backend accepts.
/// Most take the ANSI `DATE 'YYYY-MM-DD'`; T-SQL wants the unambiguous
/// `YYYYMMDD` string form; ClickHouse and SQLite (TEXT column) take the
/// ISO string.
fn date_literal(scheme: &str, day_epoch: i32) -> String {
    let iso = battery::day_iso(day_epoch);
    match scheme {
        "mssql" => format!("'{}'", iso.replace('-', "")),
        "clickhouse" | "sqlite" => format!("'{iso}'"),
        _ => format!("DATE '{iso}'"),
    }
}

/// Identifier quoting for DDL: backtick for the MySQL-family and
/// standard-SQL-on-backtick engines, double quote elsewhere. The pipeline
/// itself re-quotes via the dialect; this is only for the raw DDL here.
fn ddl_quote(scheme: &str) -> char {
    match scheme {
        "mysql" | "mariadb" | "clickhouse" | "bigquery" | "databricks" => '`',
        _ => '"',
    }
}

fn create_table_sql(scheme: &str, table: &str) -> String {
    let q = ddl_quote(scheme);
    let ddl = |id_ty: &str, val_ty: &str, grp_ty: &str, day_ty: &str| {
        // `day` is a reserved word on several backends (Exasol, MonetDB,
        // Druid's Calcite parser), so it is quoted like `mixed Case`.
        format!(
            "CREATE TABLE {table} (id {id_ty}, val {val_ty}, grp {grp_ty}, \
             {q}day{q} {day_ty}, {q}mixed Case{q} {val_ty})"
        )
    };
    match scheme {
        "postgres" => ddl("INT", "DOUBLE PRECISION", "VARCHAR(16)", "DATE"),
        "trino" => ddl("INTEGER", "DOUBLE", "VARCHAR(16)", "DATE"),
        // Date32 rather than Date: the ADBC driver surfaces the 2-byte Date
        // as its native UInt16 (not Arrow Date32), so temporal columns are
        // not recognised as dates downstream.
        "clickhouse" => format!(
            "{ddl} ENGINE = Memory",
            ddl = ddl("Int32", "Float64", "String", "Date32")
        ),
        "mysql" | "mariadb" => ddl("INT", "DOUBLE", "VARCHAR(16)", "DATE"),
        // T-SQL has no DOUBLE; FLOAT is the 64-bit type.
        "mssql" => ddl("INT", "FLOAT", "VARCHAR(16)", "DATE"),
        "exasol" => ddl("INT", "DOUBLE PRECISION", "VARCHAR(16)", "DATE"),
        // Pseudo-leg: CI points this at a PostgreSQL container (see header).
        "redshift" => ddl("INT", "DOUBLE PRECISION", "VARCHAR(16)", "DATE"),
        // Generic-ODBC leg: CI points this at a PostgreSQL container over
        // psqlODBC (see header), so the DDL uses PostgreSQL types.
        "odbc" => ddl("INT", "DOUBLE PRECISION", "VARCHAR(16)", "DATE"),
        // MonetDB-over-ODBC leg: MonetDB's 64-bit float type is DOUBLE.
        "monetdb" => ddl("INT", "DOUBLE", "VARCHAR(16)", "DATE"),
        // Oracle-over-ODBC leg: NUMBER for ints, BINARY_DOUBLE is the native
        // 64-bit float, and the VARCHAR2 spelling is required (VARCHAR is
        // reserved for future standard-conforming semantics).
        "oracle" => ddl("NUMBER(10)", "BINARY_DOUBLE", "VARCHAR2(16)", "DATE"),
        // SQLite has no date type; day stays TEXT (ISO strings compare
        // lexicographically, so range filters still behave).
        "sqlite" => ddl("INTEGER", "REAL", "TEXT", "TEXT"),
        "duckdb" => ddl("INTEGER", "DOUBLE", "VARCHAR", "DATE"),
        // Snowflake FLOAT is 64-bit.
        "snowflake" => ddl("INT", "FLOAT", "VARCHAR(16)", "DATE"),
        // BigQuery resolves the unqualified name against the connection's
        // default dataset (stmt.bigquery.query.default_dataset_id in the
        // URI — the driver only sends defaultDataset as a statement option).
        "bigquery" => ddl("INT64", "FLOAT64", "STRING", "DATE"),
        "databricks" => ddl("INT", "DOUBLE", "STRING", "DATE"),
        other => panic!("no DDL template for scheme '{other}'"),
    }
}

/// Statements to run once after connecting, before the test table is set up.
fn setup_sql(scheme: &str) -> Vec<String> {
    match scheme {
        // Exasol rejects unqualified DDL/DML until a schema exists and is
        // opened ("no schema specified or opened"). Fresh containers have
        // no schemas, so create and open one for the battery.
        "exasol" => vec![
            "CREATE SCHEMA IF NOT EXISTS ggsql".to_string(),
            "OPEN SCHEMA ggsql".to_string(),
        ],
        _ => Vec::new(),
    }
}

/// The canonical battery, run against any live reader. Every case runs
/// through the SqlSpy so a failure can dump the exact pipeline SQL.
fn run_battery(reader: &dyn Reader, ctx: &str, table: &str) {
    let spy = SqlSpy::new(reader);
    for case in battery::cases() {
        if !case.runs_live_on(ctx) {
            continue;
        }
        let query = case.query.replace("{table}", table);
        // Catch panics too, not just Err: a Rust-side panic (e.g. a chrono
        // overflow in scale training) otherwise surfaces with no case name
        // and no pipeline SQL, which is undiagnosable from CI output.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| spy.execute(&query)));
        let spec = match result {
            Ok(Ok(spec)) => spec,
            Ok(Err(e)) => panic!(
                "{ctx}: case '{}' failed: {e}\npipeline SQL:\n{}",
                case.name,
                spy.take_log().join("\n")
            ),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                panic!(
                    "{ctx}: case '{}' panicked: {msg}\npipeline SQL:\n{}",
                    case.name,
                    spy.take_log().join("\n")
                );
            }
        };
        check_expect(&spec, &case, ctx, &spy);
    }
}

/// Apply a case's live assertion to its spec.
fn check_expect(spec: &Spec, case: &battery::Case, ctx: &str, spy: &SqlSpy) {
    let layer = || {
        spec.layer_data(0)
            .unwrap_or_else(|| panic!("{ctx}: case '{}' produced no layer data", case.name))
    };
    match case.expect {
        battery::Expect::Runs => {}
        battery::Expect::MinRows(n) => {
            let height = layer().height();
            if height < n {
                fail_empty(case, ctx, spy, height);
            }
        }
        battery::Expect::ExactRows(n) => {
            assert_eq!(layer().height(), n, "{ctx}: case '{}' row count", case.name);
        }
        battery::Expect::F64In(col, lo, hi) => {
            let y = first_f64(layer(), col, ctx, case.name);
            assert!(
                (lo..=hi).contains(&y),
                "{ctx}: case '{}' value should be in [{lo}, {hi}], got {y}",
                case.name
            );
        }
    }
}

/// A row-count assertion failed. For the binned-temporal case — the one
/// whose failure mode is a units/type mismatch between the trained breaks
/// and the dialect's temporal-to-number conversion — re-run the pipeline's
/// extent and CASE statements so CI output shows what the driver actually
/// returned at each stage; otherwise dump the pipeline SQL.
fn fail_empty(case: &battery::Case, ctx: &str, spy: &SqlSpy, height: usize) -> ! {
    let log = spy.take_log();
    if case.name == "binned_temporal" {
        let extent = log
            .iter()
            .find(|s| s.contains("MIN("))
            .map(|sql| spy.inner.execute_sql(sql));
        let case_stmt = log
            .iter()
            .rev()
            .find(|s| s.contains("CASE"))
            .map(|sql| spy.inner.execute_sql(sql));
        panic!(
            "{ctx}: binned temporal returned {height} rows\n\
             pipeline SQL:\n{}\nextent result: {extent:?}\ncase result: {case_stmt:?}",
            log.join("\n")
        );
    }
    panic!(
        "{ctx}: case '{}' returned {height} rows\npipeline SQL:\n{}",
        case.name,
        log.join("\n")
    );
}

/// Read the single f64 at row 0 of `col`, with a schema dump on any
/// surprise (wrong name, wrong type, NULL — the NTILE(4) fallback produces
/// NULL for p90, so the message names that explicitly).
fn first_f64(df: &ggsql::DataFrame, col: &str, ctx: &str, case: &str) -> f64 {
    use arrow::array::Array;
    let arr = df.column(col).unwrap_or_else(|_| {
        panic!(
            "{ctx}: {case}: column '{col}' missing; schema: {:?}",
            df.schema()
        )
    });
    if arr.is_null(0) {
        panic!("{ctx}: {case}: value is NULL (NTILE(4) fallback returns NULL off-quartiles)");
    }
    match arr.data_type() {
        arrow::datatypes::DataType::Float64 => arr
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(0),
        other => panic!("{ctx}: {case}: expected Float64, got {other:?}"),
    }
}

/// When `GGSQL_TEST_REQUIRE=1`, a missing backend URI is a failure rather
/// than a skip. CI sets this so a misconfigured leg cannot pass silently;
/// local runs keep the skip-and-note behavior.
fn require_backend(env_var: &str) {
    assert!(
        std::env::var("GGSQL_TEST_REQUIRE").is_err(),
        "{env_var} is not set and GGSQL_TEST_REQUIRE=1: \
         refusing to pass a live leg without its backend"
    );
}

/// Connect to the backend named by `scheme` if its env var is set, create
/// and populate the test table, run the battery, and clean up.
fn live_backend(scheme: &str) {
    let env_var = format!("GGSQL_TEST_URI_{}", scheme.to_uppercase());
    let Ok(uri) = std::env::var(&env_var) else {
        require_backend(&env_var);
        eprintln!("skipping {scheme}: {env_var} is not set");
        return;
    };
    let reader =
        reader_from_uri(&uri).unwrap_or_else(|e| panic!("{scheme}: connection failed: {e}"));
    let table = TABLE;

    for sql in setup_sql(scheme) {
        reader
            .execute_sql(&sql)
            .unwrap_or_else(|e| panic!("{scheme}: setup failed ({sql}): {e}"));
    }

    if scheme == "bigquery" {
        // The BigQuery sandbox (billing-less free tier) rejects DML with
        // billingNotEnabled, so the table cannot be populated by INSERT.
        // register() instead CREATEs the table from the Arrow schema and
        // appends via ADBC bulk ingest — batch load jobs, which the sandbox
        // does allow. This also exercises the driver's ingest path.
        use arrow::array::{Date32Array, Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("val", DataType::Float64, false),
                Field::new("grp", DataType::Utf8, false),
                Field::new("day", DataType::Date32, false),
                Field::new("mixed Case", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(
                    battery::ROWS
                        .iter()
                        .map(|r| r.id as i64)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    battery::ROWS.iter().map(|r| r.val).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    battery::ROWS.iter().map(|r| r.grp).collect::<Vec<_>>(),
                )),
                Arc::new(Date32Array::from(
                    battery::ROWS
                        .iter()
                        .map(|r| r.day_epoch)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    battery::ROWS.iter().map(|r| r.mixed).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("build bigquery test batch");
        reader
            .register(table, ggsql::DataFrame::from_record_batch(batch), true)
            .unwrap_or_else(|e| panic!("{scheme}: register failed: {e}"));
    } else if scheme == "druid" {
        // Druid has no DDL/DML; the start script creates and populates the
        // datasource via an MSQ INSERT job before the tests run.
    } else {
        let _ = reader.execute_sql(&format!("DROP TABLE IF EXISTS {table}"));
        reader
            .execute_sql(&create_table_sql(scheme, table))
            .unwrap_or_else(|e| panic!("{scheme}: create table failed: {e}"));
        reader
            .execute_sql(&insert_sql(scheme, table))
            .unwrap_or_else(|e| panic!("{scheme}: insert failed: {e}"));
    }

    run_battery(&*reader, scheme, table);

    // register() round-trip on the ODBC legs: the ODBC register path
    // builds temp-table DDL per dialect (Oracle: GLOBAL TEMPORARY) with
    // dialect type names — a plain CREATE TEMPORARY TABLE is invalid on
    // Oracle and MSSQL.
    if matches!(scheme, "odbc" | "monetdb" | "oracle") {
        let df = ggsql::df! {
            "id" => vec![1i32, 2, 3, 4],
            "val" => vec![1.5f64, 2.5, 3.5, 4.5],
        }
        .expect("register df");
        reader
            .register("ggsql_live_register", df, true)
            .unwrap_or_else(|e| panic!("{scheme}: register failed: {e}"));
        let spec = reader
            .execute("VISUALISE DRAW point MAPPING id AS x, val AS y FROM ggsql_live_register")
            .unwrap_or_else(|e| panic!("{scheme}: plot from registered table failed: {e}"));
        assert_eq!(
            spec.layer_data(0).map(|l| l.height()),
            Some(4),
            "{scheme}: registered table row count"
        );
        let _ = reader.unregister("ggsql_live_register");
    }

    let _ = reader.execute_sql(&format!("DROP TABLE IF EXISTS {table}"));
}

/// ggsql-owned URI params (`cache=off`, cache tuning) are consumed during
/// dispatch and must not leak into the ADBC driver's own URI — drivers
/// reject unknown keys. Regression test for the param-strip fix.
#[test]
fn live_postgres_ggsql_params() {
    let env_var = "GGSQL_TEST_URI_POSTGRES";
    let Ok(uri) = std::env::var(env_var) else {
        require_backend(env_var);
        eprintln!("skipping postgres params: {env_var} is not set");
        return;
    };
    let sep = if uri.contains('?') { '&' } else { '?' };
    let uri = format!("{uri}{sep}cache=off&cache_ttl=60&cache_max_bytes=1MB");
    let reader =
        reader_from_uri(&uri).unwrap_or_else(|e| panic!("postgres+params: connect failed: {e}"));
    let spec = reader
        .execute("SELECT 1 AS x, 2 AS y VISUALISE DRAW point")
        .unwrap_or_else(|e| panic!("postgres+params: query failed: {e}"));
    assert!(spec.layer_data(0).map(|l| l.height() > 0).unwrap_or(false));
}

#[test]
fn live_postgres() {
    live_backend("postgres");
}

#[test]
fn live_trino() {
    live_backend("trino");
}

#[test]
fn live_clickhouse() {
    live_backend("clickhouse");
}

#[test]
fn live_mysql() {
    live_backend("mysql");
}

#[test]
fn live_mariadb() {
    live_backend("mariadb");
}

#[test]
fn live_mssql() {
    live_backend("mssql");
}

#[test]
fn live_exasol() {
    live_backend("exasol");
}

#[test]
fn live_redshift() {
    live_backend("redshift");
}

#[test]
fn live_odbc() {
    live_backend("odbc");
}
#[test]
fn live_monetdb() {
    live_backend("monetdb");
}
#[test]
fn live_oracle() {
    live_backend("oracle");
}
#[test]
fn live_druid() {
    live_backend("druid");
}

#[test]
fn live_sqlite() {
    live_backend("sqlite");
}

#[test]
fn live_duckdb() {
    live_backend("duckdb");
}

#[test]
fn live_snowflake() {
    live_backend("snowflake");
}

#[test]
fn live_bigquery() {
    live_backend("bigquery");
}

#[test]
fn live_databricks() {
    live_backend("databricks");
}

/// DataFusion runs in-process through its ADBC driver and needs no
/// infrastructure, but the test is gated behind `GGSQL_TEST_DATAFUSION=1`:
/// the original in-tree Rust crate (`adbc_datafusion`, stalled at 0.23)
/// stores ingested batches without validating them against the table schema,
/// and the ggsql register flow (SQL CREATE with dialect types + Arrow
/// append) then panics in MIN/MAX aggregate execution. The driver has since
/// moved to the ADBC Driver Foundry (`dbc install datafusion`), which this
/// test exercises through the driver manager — no cargo feature needed, so
/// the heavy DataFusion/prost graph stays out of the test build.
#[cfg(feature = "adbc")]
#[test]
fn live_datafusion() {
    if std::env::var("GGSQL_TEST_DATAFUSION").is_err() {
        require_backend("GGSQL_TEST_DATAFUSION");
        eprintln!(
            "skipping datafusion: GGSQL_TEST_DATAFUSION is not set \
             (was gated on an adbc_datafusion 0.23 MIN/MAX conversion bug)"
        );
        return;
    }
    use ggsql::reader::adbc::AdbcReader;
    use std::sync::Arc;

    let reader = AdbcReader::from_connection_string("datafusion://").expect("datafusion init");

    let df = ggsql::DataFrame::new(vec![
        (
            "id",
            Arc::new(arrow::array::Int32Array::from(
                battery::ROWS.iter().map(|r| r.id).collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef,
        ),
        (
            "val",
            Arc::new(arrow::array::Float64Array::from(
                battery::ROWS.iter().map(|r| r.val).collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef,
        ),
        (
            "grp",
            Arc::new(arrow::array::StringArray::from(
                battery::ROWS.iter().map(|r| r.grp).collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef,
        ),
        (
            "day",
            Arc::new(arrow::array::Date32Array::from(
                battery::ROWS
                    .iter()
                    .map(|r| r.day_epoch)
                    .collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef,
        ),
        (
            "mixed Case",
            Arc::new(arrow::array::Float64Array::from(
                battery::ROWS.iter().map(|r| r.mixed).collect::<Vec<_>>(),
            )) as arrow::array::ArrayRef,
        ),
    ])
    .expect("test dataframe");
    reader.register(TABLE, df, true).expect("register table");

    run_battery(&reader, "datafusion", TABLE);
}

/// DataFusion as a cache backend: a sqlite primary wrapped in a datafusion
/// cache (`datafusion+sqlite://`). Runs the same plot twice so the second
/// pass exercises the memo-hit path (meta-table upsert via DELETE+INSERT
/// and the UPDATE last-accessed touch) against the Foundry driver.
#[cfg(all(feature = "adbc", feature = "sqlite"))]
#[test]
fn live_datafusion_as_cache() {
    if std::env::var("GGSQL_TEST_DATAFUSION").is_err() {
        require_backend("GGSQL_TEST_DATAFUSION");
        eprintln!("skipping datafusion cache: GGSQL_TEST_DATAFUSION is not set");
        return;
    }

    let reader =
        reader_from_uri("datafusion+sqlite://:memory:").expect("datafusion-cached sqlite reader");
    reader
        .execute_sql("CREATE TABLE t (id INT, val DOUBLE)")
        .expect("create");
    reader
        .execute_sql("INSERT INTO t VALUES (1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5)")
        .expect("insert");

    let query = "VISUALISE DRAW point MAPPING id AS x, val AS y FROM t";
    let first = reader.execute(query).expect("first run (cache fill)");
    let second = reader.execute(query).expect("second run (cache hit)");
    assert!(first.layer_data(0).map(|l| l.height() > 0).unwrap_or(false));
    assert!(second
        .layer_data(0)
        .map(|l| l.height() > 0)
        .unwrap_or(false));

    // Empty results must round-trip through the cache too: the fill pass
    // registers a zero-row frame, exercising the readers' empty-batch path.
    let empty_query = "VISUALISE DRAW point MAPPING id AS x, val AS y FROM t FILTER id > 100";
    let first = reader
        .execute(empty_query)
        .expect("empty first run (cache fill)");
    let second = reader
        .execute(empty_query)
        .expect("empty second run (cache hit)");
    assert_eq!(first.layer_data(0).map(|l| l.height()), Some(0));
    assert_eq!(second.layer_data(0).map(|l| l.height()), Some(0));
}
