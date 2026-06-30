//! Caching reader: any primary [`Reader`] + an in-memory writeable cache.
//!
//! [`CachingReader`] wraps a `primary` reader and an in-memory `cache` backend,
//! splitting work across two surfaces:
//!
//! - **Source** ([`Reader::execute_sql`]) reads the primary: base reads of the
//!   user's data, plus user setup/DML.
//! - **Compute** ([`Reader::execute_sql_cached`]) runs on the cache: all derived,
//!   dialect-generated SQL operates on the `__ggsql_*` tables.
//! - [`Reader::materialize_table`] reads a body via the source surface and
//!   `register`s the result into the cache.
//! - [`Reader::dialect`] returns the cache dialect.

use crate::reader::{execute_with_reader, ColumnInfo, Reader, Spec, SqlDialect, TableInfo};
use crate::{naming, DataFrame, Result};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::Hasher;

pub struct CachingReader {
    /// Primary backend — the real data source.
    primary: Box<dyn Reader + Send>,
    /// In-memory writeable cache: derived tables, registered data, memoized reads.
    cache: Box<dyn Reader + Send>,
    /// Connection URI of the primary.
    primary_uri: String,
    /// Whether the metadata table has been created in the cache backend.
    meta_ready: Cell<bool>,
    /// Names registered into the cache. A source read that references one
    /// is routed to the cache rather than the primary.
    resident: RefCell<HashSet<String>>,
}

impl CachingReader {
    /// Construct a `CachingReader` from a primary reader, an in-memory cache
    /// backend, and the primary's connection URI. The cache is owned by the
    /// `CachingReader` and dropped with it.
    pub fn new(
        primary: Box<dyn Reader + Send>,
        cache: Box<dyn Reader + Send>,
        primary_uri: impl Into<String>,
    ) -> Self {
        Self {
            primary,
            cache,
            primary_uri: primary_uri.into(),
            meta_ready: Cell::new(false),
            resident: RefCell::new(HashSet::new()),
        }
    }

    /// Derive a stable cache key from the primary URI and the SQL text.
    fn cache_key(&self, sql: &str) -> String {
        let mut hasher = DefaultHasher::new();
        hasher.write(self.primary_uri.as_bytes());
        hasher.write(b"\n");
        hasher.write(sql.as_bytes());
        format!("{:016x}", hasher.finish())
    }

    /// Create the metadata table in the cache backend if it doesn't exist yet.
    fn ensure_meta_table(&self) -> Result<()> {
        if self.meta_ready.get() {
            return Ok(());
        }
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} \
             (cache_key VARCHAR PRIMARY KEY, sql VARCHAR NOT NULL, table_name VARCHAR NOT NULL)",
            naming::quote_ident(naming::CACHE_META_TABLE)
        );
        self.cache.execute_sql(&sql)?;
        self.meta_ready.set(true);
        Ok(())
    }

    /// Look up the cache table holding the memoized result for `key`.
    fn lookup_memo(&self, key: &str) -> Result<Option<String>> {
        let sql = format!(
            "SELECT table_name FROM {} WHERE cache_key = {}",
            naming::quote_ident(naming::CACHE_META_TABLE),
            naming::quote_literal(key),
        );
        let df = self.cache.execute_sql(&sql)?;
        if df.height() == 0 {
            return Ok(None);
        }
        let table = crate::array_util::as_str(df.column("table_name")?)?
            .value(0)
            .to_string();
        Ok(Some(table))
    }

    /// Record a memoized read in the metadata table.
    fn insert_memo(&self, key: &str, sql: &str, table: &str) -> Result<()> {
        let stmt = format!(
            "INSERT OR REPLACE INTO {} (cache_key, sql, table_name) VALUES ({}, {}, {})",
            naming::quote_ident(naming::CACHE_META_TABLE),
            naming::quote_literal(key),
            naming::quote_literal(sql),
            naming::quote_literal(table),
        );
        self.cache.execute_sql(&stmt)?;
        Ok(())
    }

    /// The cache tables holding all currently-memoized results.
    fn memo_tables(&self) -> Result<Vec<String>> {
        let sql = format!(
            "SELECT table_name FROM {}",
            naming::quote_ident(naming::CACHE_META_TABLE)
        );
        let df = self.cache.execute_sql(&sql)?;
        let n = df.height();
        if n == 0 {
            return Ok(Vec::new());
        }
        let col = crate::array_util::as_str(df.column("table_name")?)?;
        Ok((0..n).map(|i| col.value(i).to_string()).collect())
    }

    /// Whether `sql` references a cache-resident table by exact name.
    ///
    /// Parses the query's `table_ref` targets and tests them against `resident`.
    /// On a parse failure we conservatively return `false`.
    fn references_resident(&self, sql: &str) -> bool {
        let Ok(refs) = super::data::extract_table_refs(sql) else {
            return false;
        };
        let resident = self.resident.borrow();
        refs.iter().any(|t| resident.contains(t))
    }

    /// Drop every memoized result and empty the metadata table.
    fn clear_memo(&self) -> Result<()> {
        self.ensure_meta_table()?;
        for table in self.memo_tables()? {
            let _ = self.cache.unregister(&table);
        }
        let del = format!(
            "DELETE FROM {}",
            naming::quote_ident(naming::CACHE_META_TABLE)
        );
        self.cache.execute_sql(&del)?;
        Ok(())
    }
}

impl Reader for CachingReader {
    /// Source surface: base reads of the user's data (plus user setup/DML).
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Route to the cache when the read targets cache-resident objects, the
        // metadata table, or a builtin dataset.
        if sql.contains("ggsql:")
            || sql.contains(naming::CACHE_META_TABLE)
            || self.references_resident(sql)
        {
            return self.cache.execute_sql(sql);
        }

        self.ensure_meta_table()?;
        let key = self.cache_key(sql);

        if let Some(table) = self.lookup_memo(&key)? {
            return self
                .cache
                .execute_sql(&format!("SELECT * FROM {}", naming::quote_ident(&table)));
        }

        let df = self.primary.execute_sql(sql)?;

        if super::returns_rows(sql) {
            let table = naming::cache_result_table(&key);
            self.cache.register(&table, df.clone(), true)?;
            self.insert_memo(&key, sql, &table)?;
        } else {
            // A source-side write or DDL on the primary can change the data
            // behind a memoized read; drop the memo.
            self.clear_memo()?;
        }

        Ok(df)
    }

    /// Compute surface: derived/dialect-generated SQL runs on the cache.
    fn execute_sql_cached(&self, sql: &str) -> Result<DataFrame> {
        self.cache.execute_sql(sql)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.cache.register(name, df, replace)?;
        self.resident.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        self.cache.unregister(name)?;
        self.resident.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<Spec> {
        execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn SqlDialect {
        // All executor-generated SQL targets cache-resident tables.
        self.cache.dialect()
    }

    fn materialize_table(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Result<()> {
        // Read the body via the source surface, then register the result
        // into the cache.
        let body = super::wrap_with_column_aliases(body_sql, column_aliases);
        let df = self.execute_sql(&body)?;
        self.register(name, df, true)
    }

    fn caches_sources(&self) -> bool {
        true
    }

    // Schema introspection describes the real data source, so delegate to the
    // primary; the cache only holds synthetic `__ggsql_*` tables.
    fn list_catalogs(&self) -> Result<Vec<String>> {
        self.primary.list_catalogs()
    }

    fn list_schemas(&self, catalog: &str) -> Result<Vec<String>> {
        self.primary.list_schemas(catalog)
    }

    fn list_tables(&self, catalog: &str, schema: &str) -> Result<Vec<TableInfo>> {
        self.primary.list_tables(catalog, schema)
    }

    fn list_columns(&self, catalog: &str, schema: &str, table: &str) -> Result<Vec<ColumnInfo>> {
        self.primary.list_columns(catalog, schema, table)
    }
}

#[cfg(all(test, feature = "duckdb"))]
mod behavior_tests {
    use super::*;
    use crate::array_util::as_i64;
    use crate::df;
    use crate::reader::test_support::{ReadOnlyReader, SpyReader};
    use crate::reader::{CacheBackend, DuckDBReader};

    #[test]
    fn test_register_writes_to_cache_and_query_routes_there() {
        let (primary, log) = SpyReader::wrap(Box::new(DuckDBReader::new_in_memory().unwrap()));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        reader
            .register("t", df! { "x" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        // register writes to the cache; the compute surface reads it back.
        let out = reader
            .execute_sql_cached("SELECT COUNT(*) AS n FROM t")
            .unwrap();

        assert_eq!(as_i64(out.column("n").unwrap()).unwrap().value(0), 3);
        // The primary was never touched.
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn test_source_read_hits_primary_and_memoizes() {
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register("base", df! { "y" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let (primary, log) = SpyReader::wrap(Box::new(inner));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let q = "SELECT y FROM base ORDER BY y";
        let d1 = reader.execute_sql(q).unwrap();
        let d2 = reader.execute_sql(q).unwrap();
        assert_eq!(d1.height(), 3);
        assert_eq!(d2.height(), 3);

        // The primary executed the base read exactly once; the repeat was served
        // from the cache memo.
        let hits = log
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.as_str() == q)
            .count();
        assert_eq!(hits, 1);
    }

    #[test]
    fn test_full_execute_keeps_computation_off_primary() {
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register(
                "sales",
                df! { "x" => vec![1_i64, 2, 3, 4], "y" => vec![10_i64, 20, 30, 40] }.unwrap(),
                true,
            )
            .unwrap();
        let (primary, log) = SpyReader::wrap(Box::new(inner));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        reader
            .execute("SELECT x, y FROM sales VISUALISE x, y DRAW point")
            .unwrap();

        let log = log.lock().unwrap();
        // The primary is only ever read from: no temp-table DDL, no derived
        // `__ggsql_*` tables ever reach it.
        assert!(
            log.iter().all(|s| !s.to_uppercase().contains("TEMP TABLE")),
            "primary must not be written to: {:?}",
            *log
        );
        assert!(
            log.iter().all(|s| !s.contains("__ggsql_")),
            "primary must not see derived tables: {:?}",
            *log
        );
        // It did see the base read.
        assert!(log.iter().any(|s| s.contains("sales")));
    }

    #[test]
    fn test_caching_makes_read_only_primary_usable() {
        // The read-only-primary value proposition, exercised in every DuckDB
        // build (including duckdb-only CI): a primary that refuses all writes is
        // unusable on its own — materializing the global temp table fails — but
        // works once wrapped in a caching layer, because every write goes to the
        // cache and the primary is only read.
        let query = "SELECT v FROM t VISUALISE v AS x DRAW histogram";

        // Bare read-only primary: materialization must fail.
        let bare_primary = DuckDBReader::new_in_memory().unwrap();
        bare_primary
            .register(
                "t",
                df! { "v" => vec![1.0_f64, 2.0, 3.0, 4.0] }.unwrap(),
                true,
            )
            .unwrap();
        let bare = ReadOnlyReader::new(Box::new(bare_primary));
        assert!(
            bare.execute(query).is_err(),
            "a read-only primary with no cache must fail to materialize"
        );

        // Same read-only primary behind a cache: must succeed.
        let primary = DuckDBReader::new_in_memory().unwrap();
        primary
            .register(
                "t",
                df! { "v" => vec![1.0_f64, 2.0, 3.0, 4.0] }.unwrap(),
                true,
            )
            .unwrap();
        let cached = CachingReader::new(
            Box::new(ReadOnlyReader::new(Box::new(primary))),
            Box::new(DuckDBReader::new_in_memory().unwrap()),
            "test://primary",
        );
        assert!(
            cached.execute(query).is_ok(),
            "caching should make a read-only primary usable"
        );
    }

    #[test]
    fn test_no_cache_path_materializes_on_the_reader() {
        // A plain reader (no CachingReader) must keep today's behavior:
        // derived tables are materialized on the reader itself.
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register(
                "sales",
                df! { "x" => vec![1_i64, 2, 3], "y" => vec![10_i64, 20, 30] }.unwrap(),
                true,
            )
            .unwrap();
        let (reader, log) = SpyReader::wrap(Box::new(inner));

        reader
            .execute("SELECT x, y FROM sales VISUALISE x, y DRAW point")
            .unwrap();

        assert!(
            log.lock()
                .unwrap()
                .iter()
                .any(|s| s.to_uppercase().contains("TEMP TABLE")),
            "default path must materialize on the reader"
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_dialect_returns_cache_dialect() {
        use crate::reader::SqliteReader;
        // SQLite primary, DuckDB cache: dialect() must return DuckDB's (native
        // GREATEST), not SQLite's (CASE fallback).
        let primary = Box::new(SqliteReader::new().unwrap());
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");
        assert_eq!(reader.dialect().sql_greatest(&["a", "b"]), "GREATEST(a, b)");
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_explicit_layer_source_with_stat_heterogeneous() {
        use crate::reader::SqliteReader;
        // SQLite primary holds the table; DuckDB is the cache. A layer draws a
        // histogram from the primary table, which generates DuckDB-dialect stat
        // SQL. This only works because the layer source is materialized into the
        // cache; otherwise DuckDB SQL would run against the SQLite primary.
        let primary = SqliteReader::new().unwrap();
        primary
            .register(
                "tbl",
                df! { "val" => vec![1.0_f64, 2.0, 2.0, 3.0, 3.0, 3.0, 9.0] }.unwrap(),
                true,
            )
            .unwrap();
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(Box::new(primary), cache, "test://primary");

        let spec = reader.execute("VISUALISE x DRAW histogram MAPPING val AS x FROM tbl");
        assert!(
            spec.is_ok(),
            "explicit-source histogram failed: {:?}",
            spec.err()
        );
    }

    #[test]
    fn test_aliased_cte_reading_primary_routes_to_primary() {
        // A column-aliased CTE whose body reads a primary-only table must run on
        // the primary. The `__ggsql_aliased__` column-alias wrapper must not
        // misroute the read to the (empty) cache.
        let base = DuckDBReader::new_in_memory().unwrap();
        base.register("base", df! { "v" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let primary = Box::new(ReadOnlyReader::new(Box::new(base)));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let spec = reader.execute(
            "WITH t(a) AS (SELECT v FROM base) SELECT a FROM t VISUALISE a AS x DRAW point",
        );
        assert!(
            spec.is_ok(),
            "aliased CTE over a primary table should succeed: {:?}",
            spec.err()
        );
    }

    #[test]
    fn test_aliased_cte_referencing_prior_cte_routes_to_cache() {
        // A column-aliased CTE that references a *prior* CTE reads a table that
        // lives in the cache, so its body must route to the cache, while the
        // first CTE still reads the primary.
        let base = DuckDBReader::new_in_memory().unwrap();
        base.register("base", df! { "v" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let primary = Box::new(ReadOnlyReader::new(Box::new(base)));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let spec = reader.execute(
            "WITH a(p) AS (SELECT v FROM base), b(q) AS (SELECT p FROM a) \
             SELECT q FROM b VISUALISE q AS x DRAW point",
        );
        assert!(
            spec.is_ok(),
            "dependent aliased CTE should succeed: {:?}",
            spec.err()
        );
    }

    #[test]
    fn test_meta_table_records_and_serves_memo() {
        // A memoized read is recorded in the metadata table and served back from
        // the cache on repeat, without touching the primary again.
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register("base", df! { "y" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let (primary, log) = SpyReader::wrap(Box::new(inner));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let q = "SELECT y FROM base ORDER BY y";
        reader.execute_sql(q).unwrap();

        // The metadata table now has exactly one row for this read.
        let meta = reader
            .execute_sql(&format!("SELECT sql FROM {}", naming::CACHE_META_TABLE))
            .unwrap();
        assert_eq!(meta.height(), 1);
        assert_eq!(
            crate::array_util::as_str(meta.column("sql").unwrap())
                .unwrap()
                .value(0),
            q
        );

        // The repeat read is served from the cache, not the primary.
        reader.execute_sql(q).unwrap();
        let hits = log
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.as_str() == q)
            .count();
        assert_eq!(hits, 1);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_sqlite_cache_backend_memoizes() {
        use crate::reader::SqliteReader;
        // A SQLite cache backend must support the metadata table DDL/DML
        // (CREATE TABLE IF NOT EXISTS, INSERT OR REPLACE) and serve memoized reads.
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register("base", df! { "y" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let (primary, log) = SpyReader::wrap(Box::new(inner));
        let cache = Box::new(SqliteReader::new().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let q = "SELECT y FROM base ORDER BY y";
        let d1 = reader.execute_sql(q).unwrap();
        let d2 = reader.execute_sql(q).unwrap();
        assert_eq!(d1.height(), 3);
        assert_eq!(d2.height(), 3);
        let hits = log
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.as_str() == q)
            .count();
        assert_eq!(
            hits, 1,
            "SQLite cache should serve the repeat from the memo"
        );
    }

    #[test]
    fn test_resident_substring_not_false_matched() {
        // A primary-only table whose name *contains* a cache-resident table name
        // as a substring must still route to the primary. Exact-identifier
        // matching distinguishes `orders` (resident) from `orders_archive`
        // (primary-only).
        let primary = DuckDBReader::new_in_memory().unwrap();
        primary
            .register(
                "orders_archive",
                df! { "v" => vec![1_i64, 2, 3] }.unwrap(),
                true,
            )
            .unwrap();
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(Box::new(primary), cache, "test://primary");

        // `orders` lives only in the cache.
        reader
            .register("orders", df! { "v" => vec![9_i64] }.unwrap(), true)
            .unwrap();

        // Reading the primary-only `orders_archive` must hit the primary (3 rows),
        // not the resident `orders` (1 row).
        let df = reader.execute_sql("SELECT v FROM orders_archive").unwrap();
        assert_eq!(df.height(), 3);
    }

    #[test]
    fn test_source_write_invalidates_memo() {
        // Memoize a base read, mutate the primary, then re-read: the memo must be
        // invalidated by the non-row-returning statement so the second read is fresh.
        let primary = DuckDBReader::new_in_memory().unwrap();
        primary
            .register("t", df! { "v" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(Box::new(primary), cache, "test://primary");

        let q = "SELECT v FROM t";
        let d1 = reader.execute_sql(q).unwrap();
        assert_eq!(d1.height(), 3);

        reader.execute_sql("INSERT INTO t VALUES (4)").unwrap();

        let d2 = reader.execute_sql(q).unwrap();
        assert_eq!(
            d2.height(),
            4,
            "re-read must see the inserted row, not the memo"
        );
    }

    #[test]
    fn test_pure_sql_reads_primary_not_cache() {
        // The pure-SQL display path uses `execute_sql` (source), which reads the
        // primary; `execute_sql_cached` (compute) would hit the empty cache and fail.
        let primary = DuckDBReader::new_in_memory().unwrap();
        primary
            .register("t", df! { "v" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(Box::new(primary), cache, "test://primary");

        let df = reader.execute_sql("SELECT v FROM t").unwrap();
        assert_eq!(df.height(), 3);
        assert!(
            reader.execute_sql_cached("SELECT v FROM t").is_err(),
            "compute surface should not find the primary-only table"
        );
    }

    #[test]
    fn test_cache_resident_table_as_layer_source() {
        // A table registered directly on the CachingReader lives only in the
        // cache.
        let primary = Box::new(DuckDBReader::new_in_memory().unwrap());
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");
        reader
            .register(
                "only_in_cache",
                df! { "val" => vec![1.0_f64, 2.0, 2.0, 3.0, 3.0, 3.0, 9.0] }.unwrap(),
                true,
            )
            .unwrap();

        let spec = reader.execute("VISUALISE x DRAW histogram MAPPING val AS x FROM only_in_cache");
        assert!(
            spec.is_ok(),
            "cache-resident layer source should succeed: {:?}",
            spec.err()
        );
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_file_layer_source_staged_via_cache() {
        use crate::reader::SqliteReader;
        // A file source must be staged on the cache surface.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ggsql_cache_file_test_{}.csv", std::process::id()));
        std::fs::write(&path, "val\n1.0\n2.0\n2.0\n3.0\n3.0\n3.0\n9.0\n").unwrap();
        let path_str = path.to_str().unwrap().to_string();

        let primary = Box::new(SqliteReader::new().unwrap());
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache, "test://primary");

        let spec = reader.execute(&format!(
            "VISUALISE x DRAW histogram MAPPING val AS x FROM '{}'",
            path_str
        ));
        let _ = std::fs::remove_file(&path);
        assert!(
            spec.is_ok(),
            "file layer source via cache should succeed: {:?}",
            spec.err()
        );
    }
}
