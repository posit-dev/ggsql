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
//
// Beyond id/val/grp the table carries:
// - `day` (a real date column): temporal literal/cast paths, exercised by
//   the binned-scale and temporal-filter cases,
// - `mixed Case` (float): a column whose name requires quoting, exercising
//   the dialects' quote_ident handling through stats that pass raw column
//   names (quantiles) and PARTITION BY.
fn insert_sql(scheme: &str, table: &str) -> String {
    let row = |i: i32| {
        let day = date_literal(scheme, i);
        format!(
            "({i}, {v}, '{g}', {day}, {m})",
            v = i as f64 + 0.5,
            g = if i % 2 == 1 { 'a' } else { 'b' },
            m = i as f64 + 9.5,
        )
    };
    let rows: Vec<String> = (1..=8).map(row).collect();
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

/// A date literal for 2022-01-0`i` in the spelling the backend accepts.
/// Most take the ANSI `DATE 'YYYY-MM-DD'`; T-SQL wants the unambiguous
/// `YYYYMMDD` string form; ClickHouse and SQLite (TEXT column) take the
/// ISO string.
fn date_literal(scheme: &str, i: i32) -> String {
    match scheme {
        "mssql" => format!("'2022010{i}'"),
        "clickhouse" | "sqlite" => format!("'2022-01-0{i}'"),
        _ => format!("DATE '2022-01-0{i}'"),
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

    // ------------------------------------------------------------------
    // Dialect conformance regression cases (2026-10 review)
    // ------------------------------------------------------------------

    // Grouped variants of the derived-table geoms: the group columns join
    // the partition columns into stat queries, exercising grouped window
    // and derived-table paths the ungrouped cases miss (unaliased derived
    // tables break MySQL/MariaDB).
    reader
        .execute(&format!(
            "VISUALISE DRAW bar MAPPING grp AS x, grp AS fill FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: grouped bar pipeline failed: {e}"));

    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW histogram MAPPING val AS x, grp AS fill FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: grouped histogram pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: grouped histogram produced no layer data"));
    assert!(
        layer.height() > 0,
        "{ctx}: grouped histogram returned zero rows"
    );

    reader
        .execute(&format!(
            "VISUALISE DRAW smooth MAPPING id AS x, val AS y, grp AS color FROM {table} SETTING method => 'ols'"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: grouped smooth pipeline failed: {e}"));

    // A column whose name needs quoting, through the quantile path:
    // stat_aggregate passes the raw name to sql_quantile_inline /
    // sql_percentile, so dialects must quote it themselves. A name with a
    // space and mixed case breaks any dialect that interpolates it raw.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW boxplot MAPPING grp AS x, \"mixed Case\" AS y FROM {table}"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: quoted-column boxplot pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: quoted-column boxplot produced no layer data"));
    assert!(
        layer.height() > 0,
        "{ctx}: quoted-column boxplot returned zero rows"
    );

    // Percentile aggregates off the quartiles, with real value assertions:
    // the generic sql_percentile fallback uses NTILE(4), which is exact
    // only for p25/p50/p75. Group a is {1.5, 3.5, 5.5, 7.5}, so the correct
    // p10 is ~2.1 (the fallback returns the tile-boundary average 2.5) and
    // the correct p90 is ~6.9 (the fallback returns NULL: hi_tile = 5 does
    // not exist). Bounds are wide enough for approximate-quantile engines.
    for (func, lo, hi) in [("p10", 1.5, 2.35), ("p90", 6.5, 7.5)] {
        let spec = reader
            .execute(&format!(
                "VISUALISE DRAW point MAPPING grp AS x, val AS y FROM {table} \
                 SETTING aggregate => 'y:{func}' FILTER grp = 'a'"
            ))
            .unwrap_or_else(|e| panic!("{ctx}: {func} aggregate pipeline failed: {e}"));
        let layer = spec
            .layer_data(0)
            .unwrap_or_else(|| panic!("{ctx}: {func} aggregate produced no layer data"));
        let y = first_f64(layer, "__ggsql_aes_pos2__", ctx, func);
        assert!(
            (lo..=hi).contains(&y),
            "{ctx}: {func} of group a (vals 1.5, 3.5, 5.5, 7.5) should be in \
             [{lo}, {hi}], got {y}"
        );
    }

    // Binned scale over a temporal column with a non-temporal transform:
    // the numeric-CASE fallback (build_case_expression_numeric) must quote
    // the column with the active dialect, not hard-coded ANSI.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW point MAPPING day AS x, val AS y FROM {table} SCALE BINNED x VIA identity"
        ))
        .unwrap_or_else(|e| panic!("{ctx}: binned temporal pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: binned temporal produced no layer data"));
    assert!(
        layer.height() > 0,
        "{ctx}: binned temporal returned zero rows"
    );

    // PARTITION BY follows the same identifier rules as MAPPING: the name
    // is stored unquoted and re-quoted via the dialect. With a float column
    // every row is its own group, so the mean is the value itself.
    let spec = reader
        .execute(&format!(
            "VISUALISE DRAW point MAPPING id AS x, \"mixed Case\" AS y FROM {table} \
             SETTING aggregate => 'y:mean' PARTITION BY \"mixed Case\""
        ))
        .unwrap_or_else(|e| panic!("{ctx}: quoted PARTITION BY pipeline failed: {e}"));
    let layer = spec
        .layer_data(0)
        .unwrap_or_else(|| panic!("{ctx}: quoted PARTITION BY produced no layer data"));
    assert_eq!(
        layer.height(),
        8,
        "{ctx}: quoted PARTITION BY should yield one group per row"
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
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
                Arc::new(Float64Array::from(vec![
                    1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                ])),
                Arc::new(StringArray::from(vec![
                    "a", "b", "a", "b", "a", "b", "a", "b",
                ])),
                // 2022-01-01 .. 2022-01-08 as days since the epoch
                Arc::new(Date32Array::from(vec![
                    18993, 18994, 18995, 18996, 18997, 18998, 18999, 19000,
                ])),
                Arc::new(Float64Array::from(vec![
                    10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
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
            Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]))
                as arrow::array::ArrayRef,
        ),
        (
            "val",
            Arc::new(arrow::array::Float64Array::from(vec![
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            ])) as arrow::array::ArrayRef,
        ),
        (
            "grp",
            Arc::new(arrow::array::StringArray::from(vec![
                "a", "b", "a", "b", "a", "b", "a", "b",
            ])) as arrow::array::ArrayRef,
        ),
        (
            "day",
            Arc::new(arrow::array::Date32Array::from(vec![
                18993, 18994, 18995, 18996, 18997, 18998, 18999, 19000,
            ])) as arrow::array::ArrayRef,
        ),
        (
            "mixed Case",
            Arc::new(arrow::array::Float64Array::from(vec![
                10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
            ])) as arrow::array::ArrayRef,
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
    // registers a zero-row frame, which ADBC readers used to reject.
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
