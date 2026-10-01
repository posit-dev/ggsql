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
//! - `GGSQL_TEST_URI_DRUID`      e.g. `druid://localhost:8082`
//!   (CI runs it against a nano-quickstart Druid container through the
//!   Foundry `druid` ADBC driver (prerelease). Druid has no DDL, so the
//!   start script creates and populates the datasource via an MSQ INSERT
//!   job, and DruidDialect's `requires_cache` wraps the reader in a sqlite
//!   cache that hosts the battery's derived tables)
//!
//! The DataFusion case runs in-process via the Foundry ADBC driver and
//! needs no container, so one non-DuckDB engine always runs in CI.
//!
//! All cases run the same battery through the public reader pipeline: a
//! grouped scatter (quoting, projections, discrete + continuous channels),
//! a histogram (two-stage binning), a grouped boxplot (quantiles, qualified
//! projections), a grouped density (cross-group grid join), and the complex
//! derived-table geoms: smooth (aggregate CTE), ribbon and segment
//! (window-function densify), tile (2-D binning), area, and violin
//! (per-group density). The battery is validated implicitly because
//! DuckDB-backed unit tests exercise the same code paths.

use ggsql::reader::connection::reader_from_uri;
use ggsql::reader::Reader;

const TABLE: &str = "ggsql_live_test";

// Eight rows, four per group: density/violin compute a Silverman bandwidth
// from NTILE(4) tiles, which is degenerate (NULL) with fewer than four
// values per group — their grids would come back empty and the battery
// would pass vacuously.
fn insert_sql(table: &str) -> String {
    format!(
        "INSERT INTO {table} VALUES \
         (1, 1.5, 'a'), (2, 2.5, 'b'), (3, 3.5, 'a'), (4, 4.5, 'b'), \
         (5, 5.5, 'a'), (6, 6.5, 'b'), (7, 7.5, 'a'), (8, 8.5, 'b')"
    )
}

fn create_table_sql(scheme: &str, table: &str) -> String {
    match scheme {
        "postgres" => {
            format!("CREATE TABLE {table} (id INT, val DOUBLE PRECISION, grp VARCHAR(16))")
        }
        "trino" => format!("CREATE TABLE {table} (id INTEGER, val DOUBLE, grp VARCHAR(16))"),
        "clickhouse" => {
            format!("CREATE TABLE {table} (id Int32, val Float64, grp String) ENGINE = Memory")
        }
        "mysql" | "mariadb" => {
            format!("CREATE TABLE {table} (id INT, val DOUBLE, grp VARCHAR(16))")
        }
        // T-SQL has no DOUBLE; FLOAT is the 64-bit type.
        "mssql" => format!("CREATE TABLE {table} (id INT, val FLOAT, grp VARCHAR(16))"),
        "exasol" => {
            format!("CREATE TABLE {table} (id INT, val DOUBLE PRECISION, grp VARCHAR(16))")
        }
        // Pseudo-leg: CI points this at a PostgreSQL container (see header).
        "redshift" => {
            format!("CREATE TABLE {table} (id INT, val DOUBLE PRECISION, grp VARCHAR(16))")
        }
        // Generic-ODBC leg: CI points this at a PostgreSQL container over
        // psqlODBC (see header), so the DDL uses PostgreSQL types.
        "odbc" => {
            format!("CREATE TABLE {table} (id INT, val DOUBLE PRECISION, grp VARCHAR(16))")
        }
        // MonetDB-over-ODBC leg: MonetDB's 64-bit float type is DOUBLE.
        "monetdb" => format!("CREATE TABLE {table} (id INT, val DOUBLE, grp VARCHAR(16))"),
        "sqlite" => format!("CREATE TABLE {table} (id INTEGER, val REAL, grp TEXT)"),
        "duckdb" => format!("CREATE TABLE {table} (id INTEGER, val DOUBLE, grp VARCHAR)"),
        // Snowflake FLOAT is 64-bit.
        "snowflake" => format!("CREATE TABLE {table} (id INT, val FLOAT, grp VARCHAR(16))"),
        // BigQuery resolves the unqualified name against the connection's
        // default dataset (stmt.bigquery.query.default_dataset_id in the
        // URI — the driver only sends defaultDataset as a statement option).
        "bigquery" => format!("CREATE TABLE {table} (id INT64, val FLOAT64, grp STRING)"),
        "databricks" => format!("CREATE TABLE {table} (id INT, val DOUBLE, grp STRING)"),
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

/// The canonical battery, run against any live reader.
fn run_battery(reader: &dyn Reader, ctx: &str, table: &str) {
    // Grouped scatter: identifier quoting, qualified projections, and both
    // discrete (color) and continuous (x/y) channels.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW point MAPPING id AS x, val AS y, grp AS color FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: scatter pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: scatter produced no layer data"));
    assert_eq!(layer.height(), 8, "{ctx}: scatter row count");

    // Histogram: two-stage binning with GROUP BY-safe derived columns.
    // Row assertion: a broken binning stage could return zero rows and the
    // pipeline would pass vacuously without it.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW histogram MAPPING val AS x FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: histogram pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: histogram produced no layer data"));
    assert!(layer.height() > 0, "{ctx}: histogram returned zero rows");

    // Grouped boxplot: dialect quantile overrides and qualified
    // projections (`raw."g" AS "g"`). Row assertion against vacuous passes —
    // two groups always yield whisker/box/median rows.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW boxplot MAPPING grp AS x, val AS y FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: boxplot pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: boxplot produced no layer data"));
    assert!(layer.height() > 0, "{ctx}: boxplot returned zero rows");

    // Grouped density: cross-group grid join with qualified projections.
    // Row assertion: a degenerate (NULL) bandwidth would silently produce an
    // empty result, and the pipeline would pass without it.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW density MAPPING val AS x, grp AS color FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: density pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: density produced no layer data"));
    assert!(layer.height() > 0, "{ctx}: density returned zero rows");

    // Smooth (OLS): aggregate coefficient CTE with a derived table.
    reader
        .execute(&format!(
            "VISUALISE DRAW smooth MAPPING id AS x, val AS y FROM {table} SETTING method => 'ols'"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: smooth pipeline failed: {e}"));

    // Ribbon: window-function densify into a closed polygon outline.
    reader
        .execute(&format!(
            "VISUALISE DRAW ribbon MAPPING id AS x, id AS ymin, val AS ymax FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: ribbon pipeline failed: {e}"));

    // Segment: ROW_NUMBER densify plus a cross join against a vertex table.
    reader
        .execute(&format!(
            "VISUALISE DRAW segment MAPPING id AS x, val AS y, id AS xend, val AS yend FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: segment pipeline failed: {e}"));

    // Tile: two-dimensional binning with post-aggregation. Row assertion
    // against vacuous passes, as with the other cases.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW tile MAPPING val AS x, id AS y FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: tile pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: tile produced no layer data"));
    assert!(layer.height() > 0, "{ctx}: tile returned zero rows");

    // Area: ribbon variant with a synthesized zero baseline.
    reader
        .execute(&format!(
            "VISUALISE DRAW area MAPPING id AS x, val AS y FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: area pipeline failed: {e}"));

    // Violin: per-group density with mirrored outline. Same row assertion
    // as density — an empty grid would otherwise pass silently.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW violin MAPPING grp AS x, val AS y FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: violin pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: violin produced no layer data"));
    assert!(layer.height() > 0, "{ctx}: violin returned zero rows");
}

/// Connect to the backend named by `scheme` if its env var is set, create
/// and populate the test table, run the battery, and clean up.
fn live_backend(scheme: &str) {
    let env_var = format!("GGSQL_TEST_URI_{}", scheme.to_uppercase());
    let Ok(uri) = std::env::var(&env_var) else {
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
        use arrow::array::{Float64Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("val", DataType::Float64, false),
                Field::new("grp", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
                Arc::new(Float64Array::from(vec![
                    1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                ])),
                Arc::new(StringArray::from(vec![
                    "a", "b", "a", "b", "a", "b", "a", "b",
                ])),
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
            .execute_sql(&insert_sql(table))
            .unwrap_or_else(|e| panic!("{scheme}: insert failed: {e}"));
    }

    run_battery(&*reader, scheme, table);

    let _ = reader.execute_sql(&format!("DROP TABLE IF EXISTS {table}"));
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
        eprintln!(
            "skipping datafusion: GGSQL_TEST_DATAFUSION is not set \
             (was gated on an adbc_datafusion 0.23 MIN/MAX conversion bug)"
        );
        return;
    }
    use ggsql::reader::adbc::AdbcReader;

    let reader = AdbcReader::from_connection_string("datafusion://").expect("datafusion init");

    let df = ggsql::df! {
        "id" => vec![1i32, 2, 3, 4, 5, 6, 7, 8],
        "val" => vec![1.5f64, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "grp" => vec!["a", "b", "a", "b", "a", "b", "a", "b"],
    }
    .expect("test dataframe");
    reader.register(TABLE, df, true).expect("register table");

    run_battery(&reader, "datafusion", TABLE);
}
