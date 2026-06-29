//! Caching reader: any primary [`Reader`] + an in-memory writeable cache.
//!
//! [`CachingReader`] wraps a `primary` reader and an in-memory `cache` backend,
//! splitting work across two surfaces:
//!
//! - **Compute** ([`Reader::execute_sql`]) runs on the cache: all derived,
//!   dialect-generated SQL operates on the `__ggsql_*` tables.
//! - **Source** ([`Reader::execute_sql_primary`]) reads the primary: base reads of
//!   the user's data, plus user setup/DML.
//! - [`Reader::materialize_table`] reads a body via the source surface and
//!   `register`s the result into the cache.
//! - [`Reader::dialect`] returns the cache dialect.

use crate::reader::{execute_with_reader, ColumnInfo, Reader, Spec, SqlDialect, TableInfo};
use crate::{naming, DataFrame, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub struct CachingReader {
    /// Primary backend — the real data source.
    primary: Box<dyn Reader + Send>,
    /// In-memory writeable cache: derived tables, registered data, memoized reads.
    cache: Box<dyn Reader + Send>,
    /// Memo of base primary reads: source SQL text -> cache table holding the result.
    result_cache: RefCell<HashMap<String, String>>,
    /// Monotonic counter for unique memo table names.
    next_id: Cell<u64>,
}

impl CachingReader {
    /// Construct a `CachingReader` from a primary reader and an in-memory cache
    /// backend. The cache is owned by the `CachingReader` and dropped with it.
    pub fn new(primary: Box<dyn Reader + Send>, cache: Box<dyn Reader + Send>) -> Self {
        Self {
            primary,
            cache,
            result_cache: RefCell::new(HashMap::new()),
            next_id: Cell::new(0),
        }
    }
}

impl Reader for CachingReader {
    /// Compute surface: all derived/dialect-generated SQL runs on the cache.
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        self.cache.execute_sql(sql)
    }

    /// Source surface: base reads of the user's data (plus user setup/DML).
    fn execute_sql_primary(&self, sql: &str) -> Result<DataFrame> {
        if sql.contains("ggsql:") || sql.contains("__ggsql_") {
            return self.cache.execute_sql(sql);
        }

        if let Some(table) = self.result_cache.borrow().get(sql) {
            return self
                .cache
                .execute_sql(&format!("SELECT * FROM {}", naming::quote_ident(table)));
        }

        let df = self.primary.execute_sql(sql)?;

        if super::returns_rows(sql) {
            let id = self.next_id.get();
            self.next_id.set(id + 1);
            let table = naming::cache_result_table(id);
            self.cache.register(&table, df.clone(), true)?;
            self.result_cache
                .borrow_mut()
                .insert(sql.to_string(), table);
        }

        Ok(df)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.cache.register(name, df, replace)
    }

    fn unregister(&self, name: &str) -> Result<()> {
        self.cache.unregister(name)
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
        let df = self.execute_sql_primary(&body)?;
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
        let reader = CachingReader::new(primary, cache);

        reader
            .register("t", df! { "x" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        // register writes to the cache; the compute surface reads it back.
        let out = reader.execute_sql("SELECT COUNT(*) AS n FROM t").unwrap();

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
        let reader = CachingReader::new(primary, cache);

        let q = "SELECT y FROM base ORDER BY y";
        let d1 = reader.execute_sql_primary(q).unwrap();
        let d2 = reader.execute_sql_primary(q).unwrap();
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
        let reader = CachingReader::new(primary, cache);

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
        let reader = CachingReader::new(primary, cache);
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
        let reader = CachingReader::new(Box::new(primary), cache);

        let spec = reader.execute("VISUALISE x DRAW histogram MAPPING val AS x FROM tbl");
        assert!(
            spec.is_ok(),
            "explicit-source histogram failed: {:?}",
            spec.err()
        );
    }
}
