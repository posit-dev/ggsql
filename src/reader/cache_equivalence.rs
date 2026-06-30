//! Equivalence and read-only-safety tests for the caching layer.
//!
//! These exercise the whole mechanism (materialization, routing, the result
//! cache, builtin routing, layer-source staging, dialect selection).

use super::CachingReader;
use crate::reader::test_support::{ReadOnlyReader, SpyReader};
use crate::reader::{CacheBackend, DuckDBReader, Reader, SqliteReader};
use crate::DataFrame;

/// One corpus entry. `builtin_only` queries read only `ggsql:` datasets (which
/// route to the cache), so they can be compared exactly to plain DuckDB.
struct Case {
    query: &'static str,
    builtin_only: bool,
}

/// A stat-heavy corpus — these generate the most cache-dialect SQL
/// (`sql_percentile`, `sql_greatest`/`sql_least`, `sql_generate_series`,
/// casts), where caching is most likely to diverge.
const CORPUS: &[Case] = &[
    // boxplot: quantiles / IQR
    Case {
        query: "VISUALISE species AS x, bill_len AS y FROM ggsql:penguins DRAW boxplot",
        builtin_only: true,
    },
    // histogram: binning + casts (global SELECT over a builtin)
    Case {
        query: "SELECT Temp FROM ggsql:airquality VISUALISE Temp AS x DRAW histogram",
        builtin_only: true,
    },
    // density: percentile + generate_series + stddev
    Case {
        query: "VISUALISE bill_len AS x, species AS colour FROM ggsql:penguins DRAW density",
        builtin_only: true,
    },
    // grouped aggregation + facet + discrete scale
    Case {
        query: "SELECT species, bill_len, island FROM ggsql:penguins \
                VISUALISE species AS x, bill_len AS y \
                DRAW bar SETTING aggregate => 'mean' FACET island",
        builtin_only: true,
    },
    // WITH CTE + multi-layer + FILTER (CTE materialization + global + per-layer routing)
    Case {
        query: "WITH hot AS (SELECT Date, Temp FROM ggsql:airquality WHERE Temp > 70) \
                SELECT Date, Temp FROM hot \
                VISUALISE Date AS x, Temp AS y \
                DRAW line DRAW point FILTER Temp > 80 SCALE x VIA date",
        builtin_only: true,
    },
    // explicit per-layer source from a seeded table → forces layer-source staging
    // (`caches_sources`) and the read-from-primary + stat-on-cache path.
    Case {
        query: "VISUALISE species AS x, bill_len AS y \
                DRAW boxplot MAPPING species AS x, bill_len AS y FROM cache_eq_tbl",
        builtin_only: false,
    },
];

/// Seed the table referenced by the non-builtin corpus entry.
fn seed(reader: &dyn Reader) {
    let df = crate::df! {
        "species" => vec!["A", "A", "B", "B", "B", "C"],
        "bill_len" => vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    .unwrap();
    reader.register("cache_eq_tbl", df, true).unwrap();
}

/// Stringify each row using the given (canonical, sorted) column order, so two
/// DataFrames can be compared as row multisets — ignoring both physical row order
/// (a query without `ORDER BY` may return rows in a different order on each
/// materialization path) and column order (aesthetic columns are emitted in
/// HashMap order; they bind to encoding channels by name, not position).
fn row_multiset(df: &DataFrame, names: &[String]) -> Vec<String> {
    (0..df.height())
        .map(|i| {
            names
                .iter()
                .map(|n| crate::array_util::value_to_string(df.column(n).unwrap(), i))
                .collect::<Vec<_>>()
                .join("\u{1f}")
        })
        .collect()
}

/// Assert two layer DataFrames have the same columns (by name + type) and the
/// same set of rows — insensitive to row and column ordering.
fn assert_data_equivalent(a: &DataFrame, b: &DataFrame, ctx: &str) {
    let mut na: Vec<String> = a
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let mut nb: Vec<String> = b
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    na.sort();
    nb.sort();
    assert_eq!(na, nb, "{ctx}: column-name set");
    for name in &na {
        assert_eq!(
            a.column(name).unwrap().data_type(),
            b.column(name).unwrap().data_type(),
            "{ctx}: column '{name}' type",
        );
    }
    assert_eq!(a.height(), b.height(), "{ctx}: row count");
    let (mut ra, mut rb) = (row_multiset(a, &na), row_multiset(b, &na));
    ra.sort();
    rb.sort();
    assert_eq!(ra, rb, "{ctx}: row multiset mismatch");
}

/// Assert that a query produces equivalent output through `plain` and `cached`.
/// A query a backend can't run must fail the same way with or without the cache.
fn assert_equivalent(plain: &dyn Reader, cached: &dyn Reader, query: &str) {
    let a = plain.execute(query);
    let b = cached.execute(query);
    assert_eq!(
        a.is_ok(),
        b.is_ok(),
        "ok-mismatch for `{query}`: plain={:?} cached={:?}",
        a.as_ref().err(),
        b.as_ref().err(),
    );
    let (Ok(sa), Ok(sb)) = (a, b) else { return };

    assert_eq!(
        sa.layer_count(),
        sb.layer_count(),
        "layer count for `{query}`"
    );
    for i in 0..sa.layer_count() {
        match (sa.layer_data(i), sb.layer_data(i)) {
            (Some(da), Some(db)) => assert_data_equivalent(da, db, &format!("`{query}` layer {i}")),
            (None, None) => {}
            _ => panic!("layer {i} data-presence mismatch for `{query}`"),
        }
    }
}

/// The builtin corpus through `{ ReadOnlyReader(SQLite), DuckDB }` matches plain
/// DuckDB exactly. The DuckDB cache does all reading and computing for `ggsql:`
/// sources; the SQLite primary stays idle and unwritten.
#[test]
fn mode1_builtin_equivalence_matches_plain_duckdb() {
    if !cfg!(feature = "builtin-data") {
        return; // builtin corpus needs the embedded datasets
    }
    for case in CORPUS.iter().filter(|c| c.builtin_only) {
        let plain = DuckDBReader::new_in_memory().unwrap();
        let primary = ReadOnlyReader::new(Box::new(SqliteReader::new().unwrap()));
        let cache = DuckDBReader::new_in_memory().unwrap();
        let cached = CachingReader::new(Box::new(primary), Box::new(cache), "test://primary");
        assert_equivalent(&plain, &cached, case.query);
    }
}

/// Read-only safety: the full corpus succeeds through the caching reader,
/// proving a read-only/remote primary is sufficient.
#[test]
fn mode1_read_only_primary_is_sufficient() {
    for case in CORPUS {
        if case.builtin_only && !cfg!(feature = "builtin-data") {
            continue;
        }
        let sqlite = SqliteReader::new().unwrap();
        seed(&sqlite);
        let primary = ReadOnlyReader::new(Box::new(sqlite));
        let cache = DuckDBReader::new_in_memory().unwrap();
        let cached = CachingReader::new(Box::new(primary), Box::new(cache), "test://primary");
        let r = cached.execute(case.query);
        assert!(
            r.is_ok(),
            "read-only primary should suffice with caching for `{}`: {:?}",
            case.query,
            r.err(),
        );
    }
}

/// Cross-call memoization: a second identical execute does not re-read the
/// primary, because the base read is served from the cache memo.
#[test]
fn cross_call_memoization_avoids_second_primary_read() {
    let sqlite = SqliteReader::new().unwrap();
    seed(&sqlite);
    let (primary, log) = SpyReader::wrap(Box::new(sqlite));
    let cache = DuckDBReader::new_in_memory().unwrap();
    let cached = CachingReader::new(primary, Box::new(cache), "test://primary");

    let query = "SELECT bill_len FROM cache_eq_tbl VISUALISE bill_len AS x DRAW histogram";
    cached.execute(query).unwrap();
    let after_first = log.lock().unwrap().len();
    cached.execute(query).unwrap();
    let after_second = log.lock().unwrap().len();

    assert!(after_first >= 1, "first execute should read the primary");
    assert_eq!(
        after_first,
        after_second,
        "second execute must not re-hit the primary; log: {:?}",
        *log.lock().unwrap(),
    );
}

/// The library factory builds working caching readers from composite URIs.
#[test]
fn factory_builds_caching_readers() {
    use crate::reader::connection::reader_from_uri;
    for uri in ["duckdb+sqlite://memory", "sqlite+duckdb://memory"] {
        let r = reader_from_uri(uri).unwrap_or_else(|e| panic!("build `{uri}`: {e}"));
        if cfg!(feature = "builtin-data") {
            let spec =
                r.execute("VISUALISE species AS x, bill_len AS y FROM ggsql:penguins DRAW boxplot");
            assert!(
                spec.is_ok(),
                "factory reader failed for `{uri}`: {:?}",
                spec.err()
            );
        }
    }
}

/// Map projections run entirely on the cache.
#[cfg(all(feature = "spatial", feature = "builtin-data"))]
#[test]
fn map_projection_runs_on_cache_not_primary() {
    let (primary, log) = SpyReader::wrap(Box::new(DuckDBReader::new_in_memory().unwrap()));
    let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
    let reader = CachingReader::new(primary, cache, "test://primary");

    let spec = reader.execute("VISUALISE FROM ggsql:world DRAW spatial PROJECT TO orthographic");
    assert!(
        spec.is_ok(),
        "map projection via cache failed: {:?}",
        spec.err()
    );

    // No dialect-specific spatial SQL, temp-table DDL, or `__ggsql_*` reference
    // ever reached the primary.
    let log = log.lock().unwrap();
    for stmt in log.iter() {
        let upper = stmt.to_uppercase();
        assert!(
            !upper.contains("ST_") && !upper.contains("TEMP TABLE") && !stmt.contains("__ggsql_"),
            "derived spatial SQL leaked to the primary: {stmt}"
        );
    }
}

/// A real external ADBC SQLite primary + DuckDB cache, compared against a bare ADBC reader.
/// `#[ignore]` — requires `dbc install sqlite`.
#[cfg(feature = "adbc")]
mod adbc_mode {
    use super::*;
    use crate::reader::sqlite::SqliteDialect;
    use crate::reader::test_support::assert_dataframes_equal;
    use crate::reader::{AdbcReader, Spec, SqlDialect};
    use crate::{DataFrame, Result};
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::LOAD_FLAG_DEFAULT;
    use adbc_driver_manager::ManagedDriver;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn make_adbc_reader(db_path: &str) -> AdbcReader<ManagedDriver> {
        let driver = ManagedDriver::load_from_name(
            "sqlite",
            None,
            AdbcVersion::V110,
            LOAD_FLAG_DEFAULT,
            None,
        )
        .expect("`dbc install sqlite` first; see adbc.rs::equivalence_tests docs");
        let dialect: Box<dyn SqlDialect + Send> = Box::new(SqliteDialect);
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

    fn seed_adbc(path: &str) {
        let bare = make_adbc_reader(path);
        let df = crate::df! {
            "x" => vec![1_i64, 2, 3, 4, 5],
            "y" => vec![10_i64, 20, 30, 40, 50],
        }
        .unwrap();
        bare.register("t", df, false).unwrap();
    }

    /// Counts `execute_sql` calls reaching the ADBC primary.
    struct CountingAdbcReader {
        inner: AdbcReader<ManagedDriver>,
        calls: Arc<AtomicUsize>,
    }

    impl Reader for CountingAdbcReader {
        fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.execute_sql(sql)
        }
        fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
            self.inner.register(name, df, replace)
        }
        fn unregister(&self, name: &str) -> Result<()> {
            self.inner.unregister(name)
        }
        fn execute(&self, query: &str) -> Result<Spec> {
            crate::reader::execute_with_reader(self, query)
        }
        fn dialect(&self) -> &dyn SqlDialect {
            self.inner.dialect()
        }
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`"]
    fn mode2_adbc_primary_duckdb_cache_equiv_and_memo() {
        let db = NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap();
        seed_adbc(path);

        let sql = "SELECT x, y, x*y AS xy FROM t WHERE y > 15 ORDER BY x";
        let golden = make_adbc_reader(path).execute_sql(sql).unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let primary = CountingAdbcReader {
            inner: make_adbc_reader(path),
            calls: calls.clone(),
        };
        let cache = DuckDBReader::new_in_memory().unwrap();
        let cached = CachingReader::new(Box::new(primary), Box::new(cache), "test://primary");

        // Base reads go through the source surface; the cache memoizes them.
        let miss = cached.execute_sql(sql).unwrap();
        assert_dataframes_equal(&golden, &miss, "adbc cache miss");
        let after_miss = calls.load(Ordering::SeqCst);
        assert!(after_miss >= 1, "miss should reach the ADBC primary");

        let hit = cached.execute_sql(sql).unwrap();
        assert_dataframes_equal(&golden, &hit, "adbc cache hit");
        let after_hit = calls.load(Ordering::SeqCst);
        assert_eq!(
            after_miss, after_hit,
            "cache hit must not round-trip to the ADBC primary"
        );
    }
}
