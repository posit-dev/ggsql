//! Caching reader: any primary [`Reader`] + an in-memory writable cache.
//!
//! [`CachingReader`] wraps a `primary` reader and a `cache` backend (an
//! in-memory [`CacheBackend`]). It sits between the primary reader and the
//! rest of ggsql:
//!
//! - [`Reader::register`] writes to the cache and records the name.
//! - [`Reader::materialize_table`] runs the body and `register`s the resulting
//!   DataFrame into the cache, so the primary is only ever read from — base
//!   reads happen via passthrough SQL, derived tables live in the cache.
//! - [`Reader::execute_sql`] routes each statement: queries referencing a
//!   cache-resident name (or a `ggsql:` builtin) run on the cache; base reads
//!   run on the primary and are memoized into the cache.
//! - [`Reader::dialect`] returns the **cache** dialect: every dialect-specific
//!   query the executor builds targets a cache-resident table.

use crate::reader::{execute_with_reader, ColumnInfo, Reader, Spec, SqlDialect, TableInfo};
use crate::{naming, DataFrame, Result};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

pub struct CachingReader {
    primary: Box<dyn Reader + Send>,
    cache: Box<dyn Reader + Send>,
    /// Names that currently live in the cache (registered tables / memo results).
    cached_names: RefCell<HashSet<String>>,
    /// Memo of base primary reads: SQL text -> cache table name holding the result.
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
            cached_names: RefCell::new(HashSet::new()),
            result_cache: RefCell::new(HashMap::new()),
            next_id: Cell::new(0),
        }
    }

    /// Whether `sql` is one of the cache dialect's spatial setup statements.
    ///
    /// ggsql-generated spatial setup (`INSTALL`/`LOAD spatial`, …) must run on
    /// the cache, where the spatial SQL executes — not on the primary.
    fn is_cache_spatial_setup(&self, sql: &str) -> bool {
        self.cache
            .dialect()
            .sql_spatial_setup()
            .iter()
            .any(|stmt| stmt == sql)
    }
}

impl Reader for CachingReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Cache-resident names and `ggsql:` builtins are served by the cache.
        if references_cached_name(sql, &self.cached_names.borrow()) || sql.contains("ggsql:") {
            return self.cache.execute_sql(sql);
        }

        // ggsql-generated spatial setup targets the cache (where spatial SQL runs).
        if self.is_cache_spatial_setup(sql) {
            return self.cache.execute_sql(sql);
        }

        // Base read against the primary, with result memoization.
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
            self.cached_names.borrow_mut().insert(table.clone());
            self.result_cache
                .borrow_mut()
                .insert(sql.to_string(), table);
        }

        Ok(df)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.cache.register(name, df, replace)?;
        self.cached_names.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        self.cache.unregister(name)?;
        self.cached_names.borrow_mut().remove(name);
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

/// Check whether `sql` references any name in `cached_names` as a SQL
/// identifier (not part of a longer identifier, not inside a single-quoted
/// string literal).
///
/// Matches a bare reference (`FROM orders`), a double-quoted identifier
/// (`FROM "orders"`), or a qualified reference (`FROM catalog.schema.orders`).
/// Does not match substrings of longer identifiers (`orders_detail`) or
/// string-literal contents (`'orders of magnitude'`). This is a lightweight
/// scanner, not a full SQL parser: a cached name appearing only inside a
/// comment is misrouted to the cache and fails with a clear error rather than
/// silently hitting the primary.
fn references_cached_name(sql: &str, cached_names: &HashSet<String>) -> bool {
    cached_names
        .iter()
        .any(|name| sql_references_identifier(sql, name))
}

fn sql_references_identifier(sql: &str, name: &str) -> bool {
    let bytes = sql.as_bytes();
    let name_bytes = name.as_bytes();
    let n = name_bytes.len();
    if n == 0 {
        return false;
    }
    let mut i = 0;
    while i + n <= bytes.len() {
        if &bytes[i..i + n] == name_bytes {
            let before_ok = i == 0 || !is_identifier_byte(bytes[i - 1]);
            let after_ok = i + n == bytes.len() || !is_identifier_byte(bytes[i + n]);
            if before_ok && after_ok && !is_inside_string_literal(bytes, i) {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Walk from start to `pos` tracking whether we're inside a single-quoted
/// string literal. SQL-standard doubled-single-quote (`''`) is an escape that
/// keeps us inside the literal.
fn is_inside_string_literal(bytes: &[u8], pos: usize) -> bool {
    let mut inside = false;
    let mut i = 0;
    while i < pos && i < bytes.len() {
        if bytes[i] == b'\'' {
            if inside && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                i += 2;
                continue;
            }
            inside = !inside;
        }
        i += 1;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_references_cached_name_empty_set() {
        let set = HashSet::new();
        assert!(!references_cached_name("SELECT * FROM foo", &set));
    }

    #[test]
    fn test_references_cached_name_match() {
        let mut set = HashSet::new();
        set.insert("__ggsql_global_abc123__".to_string());
        assert!(references_cached_name(
            "SELECT * FROM __ggsql_global_abc123__ WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_cached_name_rejects_longer_identifier() {
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_cached_name(
            "SELECT * FROM orders_detail WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_cached_name_rejects_prefix_of_longer_identifier() {
        let mut set = HashSet::new();
        set.insert("col".to_string());
        assert!(!references_cached_name("SELECT col_id FROM users", &set));
    }

    #[test]
    fn test_references_cached_name_rejects_inside_string_literal() {
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_cached_name(
            "SELECT 'orders of magnitude' AS label",
            &set
        ));
    }

    #[test]
    fn test_references_cached_name_matches_quoted_identifier() {
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_cached_name(r#"SELECT * FROM "orders""#, &set));
    }

    #[test]
    fn test_references_cached_name_matches_qualified_reference() {
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_cached_name(
            "SELECT * FROM catalog.schema.orders WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_cached_name_handles_escaped_quotes_in_literal() {
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_cached_name(
            "SELECT 'it''s fine' FROM orders",
            &set
        ));
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
        let out = reader.execute_sql("SELECT COUNT(*) AS n FROM t").unwrap();

        assert_eq!(as_i64(out.column("n").unwrap()).unwrap().value(0), 3);
        // The query routed to the cache; the primary was never touched.
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn test_base_read_routes_to_primary_and_memoizes() {
        let inner = DuckDBReader::new_in_memory().unwrap();
        inner
            .register("base", df! { "y" => vec![1_i64, 2, 3] }.unwrap(), true)
            .unwrap();
        let (primary, log) = SpyReader::wrap(Box::new(inner));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache);

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

    #[cfg(feature = "spatial")]
    #[test]
    fn test_spatial_setup_routes_to_cache() {
        let (primary, log) = SpyReader::wrap(Box::new(DuckDBReader::new_in_memory().unwrap()));
        let cache = Box::new(DuckDBReader::new_in_memory().unwrap());
        let reader = CachingReader::new(primary, cache);

        // ggsql-generated spatial setup must run on the cache (where the spatial
        // SQL executes), never on the primary.
        let setup = reader.dialect().sql_spatial_setup();
        assert!(!setup.is_empty(), "DuckDB cache should emit spatial setup");
        for stmt in &setup {
            // Ignore execution result (may require the extension); we only assert routing.
            let _ = reader.execute_sql(stmt);
        }
        let log = log.lock().unwrap();
        for stmt in &setup {
            assert!(
                !log.contains(stmt),
                "spatial setup must route to the cache, not the primary: {}",
                stmt
            );
        }
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
