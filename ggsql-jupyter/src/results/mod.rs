//! Query results kept so a frontend can explore them after the fact.
//!
//! A frontend-agnostic API over recent results, one page at a time: open a
//! result, filter and sort it, read pages, profile columns, export a selection,
//! and get SQL reproducing the view. [`methods`] exposes it as named JSON
//! calls. Nothing here knows about any particular frontend.
//!
//! A result is held as its Arrow data until it is first opened, and is then
//! copied into a dedicated in-memory DuckDB ([`CacheTables`], the same
//! machinery as ggsql's caching reader) where every filter, sort, page and
//! profile is SQL. The cache is the results API's own: the session's reader is
//! never wrapped, so nothing changes about the dialect the user's queries run
//! in.
//!
//! **One byte budget covers everything kept here**: unopened results, the
//! cache's tables (including those pinned by an open view) and every view's
//! order table. Making room drops closed results first, then the oldest
//! unopened ones; an open view is never pulled from under its frontend, so
//! when open views alone fill the budget, a new result is not kept and a sort
//! or filter that needs room fails with an error saying so.
//!
//! The budget counts what is kept, not the working memory DuckDB needs to
//! sort or profile it, so the cache's DuckDB is also given a hard memory limit
//! of [`memory_limit_for`] the budget. A query that would pass it fails with
//! DuckDB's out-of-memory error, which reaches the frontend as an error.

mod export;
pub mod filter;
pub mod methods;
pub mod view;

use anyhow::{anyhow, bail, Result};
use filter::{Filter, SortKey};
use ggsql::naming::quote_ident;
use ggsql::reader::cache::{estimate_bytes, CacheTables};
use ggsql::DataFrame;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use view::ResultView;

/// The column holding each row's position in the query's output, added when a
/// result is copied into the cache. Ties in a sort break on it, so paging is
/// stable and clearing the sort restores the query's own order.
pub const ROW: &str = "__ggsql_row__";

/// The default byte budget for everything the results API keeps.
pub const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024;

/// The least memory the cache's DuckDB is given, however small the budget.
const MIN_MEMORY_LIMIT: u64 = 64 * 1024 * 1024;

/// The most memory limit set on the cache's DuckDB; a budget that would pass
/// it leaves DuckDB's own default, which is most of the machine's memory.
const MAX_MEMORY_LIMIT: u64 = 1 << 50;

/// The memory the cache's DuckDB may use, kept tables and working memory
/// together, for a budget of `max_bytes`: twice the budget, since DuckDB holds
/// a table less compactly than Arrow's estimate of it and needs room on top to
/// sort and profile. `None` for no limit of ours.
pub fn memory_limit_for(max_bytes: u64) -> Option<u64> {
    max_bytes
        .checked_mul(2)
        .map(|limit| limit.max(MIN_MEMORY_LIMIT))
        .filter(|&limit| limit <= MAX_MEMORY_LIMIT)
}

/// A result's column.
#[derive(Debug, Clone, Serialize)]
pub struct Column {
    pub name: String,
    /// The cache backend's type name, e.g. `BIGINT`.
    pub type_name: String,
    /// A frontend-neutral kind: `integer`, `floating`, `boolean`, `string`,
    /// `date`, `datetime`, `time` or `unknown`.
    pub kind: String,
    /// Whether the query returned this name more than once, so it was renamed
    /// here and SQL over the query cannot refer to it.
    #[serde(skip)]
    pub repeated: bool,
}

/// A result not yet opened, held as its Arrow data.
struct Pending {
    key: String,
    title: String,
    query: String,
    df: DataFrame,
    bytes: u64,
}

/// A result copied into the cache.
struct Opened {
    title: String,
    query: String,
    table: String,
    columns: Vec<Column>,
    num_rows: usize,
}

/// What `open` reports about a view.
#[derive(Debug, Serialize)]
pub struct OpenedView {
    pub view: String,
    pub key: String,
    pub title: String,
    pub query: String,
    pub columns: Vec<Column>,
    pub num_rows: usize,
}

/// Recent query results and the open views of them.
pub struct ResultStore {
    /// The results cache: opened results, pinned while a view is open.
    cache: CacheTables,
    /// Results not opened yet, oldest first.
    pending: VecDeque<Pending>,
    /// The budget for everything kept.
    max_bytes: u64,
    opened: HashMap<String, Opened>,
    views: HashMap<String, ResultView>,
}

impl ResultStore {
    /// A store keeping at most `max_bytes`, by Arrow's estimate of the data,
    /// whose DuckDB uses at most [`memory_limit_for`] it.
    pub fn new(max_bytes: u64) -> Result<Self> {
        let backend = ggsql::reader::connection::reader_from_uri("duckdb://memory")?;
        // An in-memory DuckDB may otherwise spill to temporary files under
        // memory pressure; results must never be written to disk.
        backend.execute_sql("SET temp_directory = ''")?;
        if let Some(limit) = memory_limit_for(max_bytes) {
            backend.execute_sql(&format!("SET memory_limit = '{}KiB'", limit / 1024))?;
        }
        // The store evicts to its own budget, which spans more than the cache.
        let cache = CacheTables::new(backend, u64::MAX);
        Ok(Self {
            cache,
            pending: VecDeque::new(),
            max_bytes,
            opened: HashMap::new(),
            views: HashMap::new(),
        })
    }

    /// Bytes kept now.
    fn used(&self) -> Result<u64> {
        let pending: u64 = self.pending.iter().map(|p| p.bytes).sum();
        let orders: u64 = self.views.values().map(ResultView::order_bytes).sum();
        Ok(pending + orders + self.cache.total_bytes()?)
    }

    /// Drop closed results, then the oldest unopened ones, until `extra` more
    /// bytes fit in the budget. Returns whether they do. When they cannot fit
    /// even with everything droppable gone, nothing is dropped.
    fn make_room(&mut self, extra: u64) -> Result<bool> {
        let orders: u64 = self.views.values().map(ResultView::order_bytes).sum();
        // Open views' results and order tables stay whatever is dropped.
        let kept = self.cache.pinned_bytes()? + orders;
        if kept.saturating_add(extra) > self.max_bytes {
            return Ok(false);
        }
        let limit = self.max_bytes - extra;
        let outside_cache: u64 = self.pending.iter().map(|p| p.bytes).sum::<u64>() + orders;
        self.cache.evict_to(limit.saturating_sub(outside_cache))?;
        self.forget_evicted()?;
        while self.used()? > limit && self.pending.pop_front().is_some() {}
        Ok(self.used()? <= limit)
    }

    /// Forget the results the cache has evicted.
    fn forget_evicted(&mut self) -> Result<()> {
        let mut gone = Vec::new();
        for key in self.opened.keys() {
            if self.cache.lookup(key)?.is_none() {
                gone.push(key.clone());
            }
        }
        for key in gone {
            self.opened.remove(&key);
        }
        Ok(())
    }

    /// Keep `df`, produced by `query`, under a fresh key, which is returned.
    /// `None` when it cannot fit in the budget, even once everything that can
    /// go has gone.
    pub fn insert(
        &mut self,
        df: DataFrame,
        title: String,
        query: String,
    ) -> Result<Option<String>> {
        let bytes = estimate_bytes(&df).max(0) as u64;
        if bytes > self.max_bytes {
            return Ok(None);
        }
        // Unguessable, so a stale link from a previous kernel process can never
        // name a result of this one.
        let key = uuid::Uuid::new_v4().simple().to_string();
        let fits = self.make_room(bytes)?;
        if !fits {
            return Ok(None);
        }
        self.pending.push_back(Pending {
            key: key.clone(),
            title,
            query,
            df,
            bytes,
        });
        Ok(Some(key))
    }

    fn table_name(key: &str) -> String {
        format!("__ggsql_result_{key}__")
    }

    /// Open a new view of the result `key`, copying it into the cache the
    /// first time. The result stays pinned while any view of it is open.
    pub fn open(&mut self, key: &str) -> Result<OpenedView> {
        if let Some(i) = self.pending.iter().position(|p| p.key == key) {
            let pending = self.pending.remove(i).expect("position is in range");
            match self.copy_into_cache(&pending) {
                Ok(opened) => {
                    self.opened.insert(pending.key, opened);
                }
                Err(e) => {
                    // Kept, so the error says what went wrong each time
                    // rather than that the result is gone.
                    self.pending.insert(i, pending);
                    return Err(e);
                }
            }
        } else if self.opened.contains_key(key) {
            if self.cache.lookup(key)?.is_none() {
                self.opened.remove(key);
                return Err(gone());
            }
            self.cache.set_pinned(key, true)?;
        } else {
            return Err(gone());
        }

        let opened = &self.opened[key];
        let id = uuid::Uuid::new_v4().simple().to_string();
        let view = ResultView::new(
            key,
            &id,
            &opened.table,
            opened.columns.clone(),
            opened.num_rows,
        );
        self.views.insert(id.clone(), view);
        Ok(OpenedView {
            view: id,
            key: key.to_string(),
            title: opened.title.clone(),
            query: opened.query.clone(),
            columns: opened.columns.clone(),
            num_rows: opened.num_rows,
        })
    }

    fn copy_into_cache(&self, pending: &Pending) -> Result<Opened> {
        let num_rows = pending.df.height();
        let (df, repeated) = unique_names(pending.df.clone())?;
        let rows = arrow::array::UInt64Array::from_iter_values(0..num_rows as u64);
        let df = df.with_column(ROW, Arc::new(rows))?;
        let table = Self::table_name(&pending.key);
        // Pinned from the start, so the eviction its own insertion triggers
        // can't take it. The budget already counts it, as a pending result.
        self.cache
            .put(&pending.key, &pending.query, &table, df, true)?;

        let described = self
            .cache
            .backend()
            .execute_sql(&format!("DESCRIBE {}", quote_ident(&table)))?;
        let names = ggsql::array_util::as_str(described.column("column_name")?)?;
        let types = ggsql::array_util::as_str(described.column("column_type")?)?;
        let columns = (0..described.height())
            .filter(|&i| names.value(i) != ROW)
            .map(|i| Column {
                name: names.value(i).to_string(),
                type_name: types.value(i).to_string(),
                kind: kind_of(types.value(i)).to_string(),
                repeated: repeated.contains(names.value(i)),
            })
            .collect();
        Ok(Opened {
            title: pending.title.clone(),
            query: pending.query.clone(),
            table,
            columns,
            num_rows,
        })
    }

    /// Close a view, dropping what it keeps in the cache. Its result is
    /// unpinned once no view of it is open, and is then evicted when the
    /// budget needs the room.
    pub fn close(&mut self, view: &str) -> Result<()> {
        let mut v = self
            .views
            .remove(view)
            .ok_or_else(|| anyhow!("There is no view {view:?}"))?;
        v.close(self.cache.backend());
        if !self.views.values().any(|other| other.key == v.key) {
            self.cache.set_pinned(&v.key, false)?;
        }
        Ok(())
    }

    /// Close every view, for when the frontend holding them has gone.
    pub fn close_all(&mut self) {
        let ids: Vec<String> = self.views.keys().cloned().collect();
        for id in ids {
            if let Err(e) = self.close(&id) {
                tracing::warn!("Closing view {} failed: {}", id, e);
            }
        }
    }

    fn view(&self, view: &str) -> Result<&ResultView> {
        self.views
            .get(view)
            .ok_or_else(|| anyhow!("There is no view {view:?}"))
    }

    /// A view and the cache it reads from.
    pub fn view_mut(&mut self, view: &str) -> Result<(&mut ResultView, &CacheTables)> {
        let v = self
            .views
            .get_mut(view)
            .ok_or_else(|| anyhow!("There is no view {view:?}"))?;
        Ok((v, &self.cache))
    }

    /// Make room for a new order table of `view`, which replaces its current
    /// one once built.
    fn reserve_order(&mut self, view: &str) -> Result<()> {
        let v = self.view(view)?;
        let need = v.max_order_bytes();
        if !self.make_room(need)? {
            bail!(
                "Sorting or filtering this result needs more memory than the data explorer's \
                 budget has left ({} bytes). Close another Data Explorer tab and try again.",
                self.max_bytes
            );
        }
        Ok(())
    }

    /// Set a view's sort keys, within the budget.
    pub fn set_sort(&mut self, view: &str, keys: Vec<SortKey>) -> Result<(), String> {
        if !keys.is_empty() {
            self.reserve_order(view).map_err(|e| e.to_string())?;
        }
        let (v, cache) = self.view_mut(view).map_err(|e| e.to_string())?;
        v.set_sort(keys, cache.backend())
    }

    /// Set a view's row filters, within the budget.
    pub fn set_filters(
        &mut self,
        view: &str,
        filters: Vec<Filter>,
    ) -> Result<Vec<Option<String>>, String> {
        if !filters.is_empty() {
            self.reserve_order(view).map_err(|e| e.to_string())?;
        }
        let (v, cache) = self.view_mut(view).map_err(|e| e.to_string())?;
        v.set_filters(filters, cache.backend())
    }

    /// The query a view's result came from.
    pub fn query_of(&self, view: &str) -> Result<&str> {
        let v = self.view(view)?;
        Ok(&self.opened[&v.key].query)
    }
}

/// `df` with each repeated column name made unique, as DuckDB does: the second
/// `a` becomes `a_1`, the third `a_2`, skipping names already taken. A query
/// can return repeated names, but a table cannot hold them. Also returns the
/// new names.
fn unique_names(df: DataFrame) -> Result<(DataFrame, HashSet<String>)> {
    let names = df.get_column_names();
    let mut taken: HashSet<String> = names.iter().cloned().collect();
    let mut seen = HashSet::new();
    let mut renamed = HashSet::new();
    let mut columns = Vec::with_capacity(names.len());
    for (name, array) in names.iter().zip(df.get_columns()) {
        let mut unique = name.clone();
        if !seen.insert(name.clone()) {
            let mut n = 1;
            while taken.contains(&format!("{name}_{n}")) {
                n += 1;
            }
            unique = format!("{name}_{n}");
            taken.insert(unique.clone());
            renamed.insert(unique.clone());
        }
        columns.push((unique, array.clone()));
    }
    if renamed.is_empty() {
        return Ok((df, renamed));
    }
    // The first of a repeated name can't be named in SQL over the query either.
    for name in &names {
        if names.iter().filter(|n| *n == name).count() > 1 {
            renamed.insert(name.clone());
        }
    }
    Ok((DataFrame::new(columns)?, renamed))
}

fn gone() -> anyhow::Error {
    anyhow!("This query result is no longer available. Re-run the query to explore it.")
}

/// A DuckDB type name's neutral kind.
pub fn kind_of(type_name: &str) -> &'static str {
    let upper = type_name.trim().to_uppercase();
    // Nested types (`INTEGER[]`, `STRUCT(a INTEGER)`) name a scalar type
    // inside them, but are not one.
    if upper.ends_with(']') {
        return "unknown";
    }
    // The type itself, without parameters: `DECIMAL(18,3)` is `DECIMAL`,
    // `ENUM('a', 'b')` is `ENUM`.
    let base = upper.split('(').next().unwrap_or("").trim();
    match base {
        "TINYINT" | "SMALLINT" | "INTEGER" | "BIGINT" | "HUGEINT" | "UTINYINT" | "USMALLINT"
        | "UINTEGER" | "UBIGINT" | "UHUGEINT" | "INT" | "INT1" | "INT2" | "INT4" | "INT8" => {
            "integer"
        }
        "FLOAT" | "REAL" | "DOUBLE" | "DECIMAL" | "NUMERIC" => "floating",
        "BOOLEAN" | "BOOL" => "boolean",
        "DATE" => "date",
        "TIMESTAMP"
        | "TIMESTAMP WITH TIME ZONE"
        | "TIMESTAMPTZ"
        | "TIMESTAMP_S"
        | "TIMESTAMP_MS"
        | "TIMESTAMP_NS"
        | "DATETIME" => "datetime",
        "TIME" | "TIME WITH TIME ZONE" | "TIMETZ" => "time",
        "VARCHAR" | "CHAR" | "BPCHAR" | "TEXT" | "STRING" | "UUID" | "ENUM" => "string",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::filter::{Filter, SortKey};
    use crate::results::view::Rows;
    use serde_json::json;

    fn frame(sql: &str) -> DataFrame {
        ggsql::reader::connection::reader_from_uri("duckdb://memory")
            .unwrap()
            .execute_sql(sql)
            .unwrap()
    }

    fn keep(store: &mut ResultStore, df: DataFrame, title: String, query: String) -> String {
        store
            .insert(df, title, query)
            .unwrap()
            .expect("the result fits")
    }

    fn numbers(n: usize) -> DataFrame {
        frame(&format!(
            "SELECT i AS n, i % 3 AS g, 'item ' || i AS s FROM range({n}) t(i)"
        ))
    }

    fn page(
        store: &mut ResultStore,
        view: &str,
        first: usize,
        last: usize,
        col: usize,
    ) -> Vec<String> {
        let (v, cache) = store.view_mut(view).unwrap();
        v.page(cache.backend(), &Rows::Range { first, last }, &[col])
            .unwrap()
            .remove(0)
            .into_iter()
            .map(|v| v.as_str().unwrap_or("NULL").to_string())
            .collect()
    }

    #[test]
    fn test_open_describes_the_result_without_the_row_column() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            numbers(5),
            "Query [1]".into(),
            "SELECT 1".into(),
        );
        let opened = store.open(&key).unwrap();
        assert_eq!(opened.num_rows, 5);
        let names: Vec<_> = opened.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["n", "g", "s"]);
        assert_eq!(opened.columns[0].kind, "integer");
        assert_eq!(opened.columns[2].kind, "string");
        assert_eq!(page(&mut store, &opened.view, 1, 3, 0), ["1", "2", "3"]);
    }

    #[test]
    fn test_sort_is_stable_and_clearing_it_restores_the_order() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(9), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let (v, cache) = store.view_mut(&view).unwrap();
        v.set_sort(
            vec![SortKey {
                column: 1,
                ascending: false,
            }],
            cache.backend(),
        )
        .unwrap();
        // g = 2 first, ties in the query's own order.
        assert_eq!(page(&mut store, &view, 0, 2, 0), ["2", "5", "8"]);
        assert_eq!(page(&mut store, &view, 3, 5, 0), ["1", "4", "7"]);

        let (v, cache) = store.view_mut(&view).unwrap();
        v.set_sort(vec![], cache.backend()).unwrap();
        assert_eq!(page(&mut store, &view, 0, 2, 0), ["0", "1", "2"]);
    }

    #[test]
    fn test_filters_select_rows_and_report_bad_ones() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(10), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "compare", "column": 0, "op": ">=", "value": "4"},
            {"type": "compare", "column": 0, "op": "=", "value": "four"},
            {"type": "text", "column": 2, "match": "ends_with", "term": "7"}
        ]))
        .unwrap();
        let (v, cache) = store.view_mut(&view).unwrap();
        let errors = v.set_filters(filters, cache.backend()).unwrap();
        assert!(errors[0].is_none());
        assert!(errors[1].as_deref().unwrap().contains("not a number"));
        assert!(errors[2].is_none());
        assert_eq!(v.selected_rows, 1);
        assert_eq!(page(&mut store, &view, 0, 0, 2), ["item 7"]);
    }

    #[test]
    fn test_dates_page_and_profile_as_dates() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            frame("SELECT DATE '2024-01-01' + i::INTEGER AS d FROM range(3) t(i)"),
            "q".into(),
            String::new(),
        );
        let view = store.open(&key).unwrap().view;
        assert_eq!(
            page(&mut store, &view, 0, 1, 0),
            ["2024-01-01", "2024-01-02"]
        );
        let (v, cache) = store.view_mut(&view).unwrap();
        let req: view::ProfileRequest = serde_json::from_value(json!({"summary": true})).unwrap();
        let p = v.profile(cache.backend(), 0, &req).unwrap();
        assert_eq!(p["summary"]["min"], "2024-01-01");
        assert_eq!(p["summary"]["max"], "2024-01-03");
    }

    fn summary_of(sql: &str, column: usize) -> serde_json::Value {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, frame(sql), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let (v, cache) = store.view_mut(&view).unwrap();
        let req: view::ProfileRequest =
            serde_json::from_value(json!({"null_count": true, "summary": true})).unwrap();
        v.profile(cache.backend(), column, &req).unwrap()
    }

    #[test]
    fn test_stdev_keeps_its_precision_around_large_values() {
        let p = summary_of("SELECT 1000000000 + i AS v FROM range(3) t(i)", 0);
        let sd: f64 = p["summary"]["stdev"].as_str().unwrap().parse().unwrap();
        assert!((sd - 1.0).abs() < 1e-9, "{sd}");
    }

    #[test]
    fn test_stdev_is_left_out_with_a_nan_or_infinity() {
        for special in ["'nan'::DOUBLE", "'inf'::DOUBLE"] {
            let p = summary_of(
                &format!("SELECT * FROM (VALUES (1.0), (2.0), ({special})) t(v)"),
                0,
            );
            assert!(p["summary"].get("stdev").is_none(), "{special}: {p}");
        }
    }

    #[test]
    fn test_nested_columns_are_not_numbers_and_still_profile() {
        assert_eq!(kind_of("INTEGER[]"), "unknown");
        assert_eq!(kind_of("STRUCT(a INTEGER)"), "unknown");
        assert_eq!(kind_of("MAP(VARCHAR, INTEGER)"), "unknown");
        let p = summary_of("SELECT [1, 2] AS l, {'a': 1} AS s", 0);
        assert_eq!(p["null_count"], 0);
        assert_eq!(p["summary"], json!({}));
    }

    #[test]
    fn test_repeated_column_names_are_made_unique() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            frame("SELECT 1 AS a, 2 AS a, 3 AS a_1"),
            "q".into(),
            String::new(),
        );
        let opened = store.open(&key).unwrap();
        let names: Vec<_> = opened.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "a_2", "a_1"]);
        assert_eq!(page(&mut store, &opened.view, 0, 0, 1), ["2"]);
    }

    #[test]
    fn test_a_failed_open_keeps_the_result() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            frame("SELECT 1 AS x"),
            "q".into(),
            String::new(),
        );
        // Break the cache's bookkeeping, so copying the result in fails.
        store.cache.lookup("anything").unwrap();
        store
            .cache
            .backend()
            .execute_sql(&format!(
                "DROP TABLE {}",
                quote_ident(ggsql::naming::CACHE_META_TABLE)
            ))
            .unwrap();
        for _ in 0..2 {
            let err = store.open(&key).unwrap_err().to_string();
            // The real reason each time, not that the result is gone.
            assert!(!err.contains("no longer available"), "{err}");
        }
    }

    #[test]
    fn test_sql_leaves_out_a_filter_the_cache_rejected() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            frame("SELECT 'a' AS s FROM range(3)"),
            "q".into(),
            "SELECT 1".into(),
        );
        let view = store.open(&key).unwrap().view;
        // A regex the cache refuses to compile.
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "text", "column": 0, "match": "regex", "term": "(", "case_sensitive": true},
            {"type": "is_null", "column": 0, "negate": true}
        ]))
        .unwrap();
        let (v, cache) = store.view_mut(&view).unwrap();
        let errors = v.set_filters(filters, cache.backend()).unwrap();
        assert!(errors[0].is_some() && errors[1].is_none(), "{errors:?}");
        let sql = v.to_sql("SELECT 1", &ggsql::reader::AnsiDialect);
        assert!(sql[0].starts_with("-- Filter left out"), "{sql:?}");
        assert!(!sql.iter().any(|l| l.contains("regexp_matches")), "{sql:?}");
        assert!(sql.iter().any(|l| l.contains("IS NOT NULL")), "{sql:?}");
    }

    #[test]
    fn test_paging_past_the_end_returns_what_there_is() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(3), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        assert_eq!(page(&mut store, &view, 2, 10, 0), ["2"]);
    }

    #[test]
    fn test_profiles() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut store,
            frame("SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'a'), (NULL, NULL)) t(n, s)"),
            "q".into(),
            String::new(),
        );
        let view = store.open(&key).unwrap().view;
        let (v, cache) = store.view_mut(&view).unwrap();
        let req: view::ProfileRequest = serde_json::from_value(json!({
            "null_count": true, "summary": true,
            "histogram": {"bins": 10, "quantiles": [0.5]}
        }))
        .unwrap();
        let p = v.profile(cache.backend(), 0, &req).unwrap();
        assert_eq!(p["null_count"], 1);
        assert_eq!(p["summary"]["min"], "1");
        assert_eq!(p["summary"]["max"], "3");
        assert_eq!(p["summary"]["median"], "2");
        let counts: i64 = p["histogram"]["bin_counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_i64().unwrap())
            .sum();
        assert_eq!(counts, 3);

        let req: view::ProfileRequest =
            serde_json::from_value(json!({"frequency": {"limit": 1}})).unwrap();
        let p = v.profile(cache.backend(), 1, &req).unwrap();
        assert_eq!(p["frequency"]["values"], json!(["a"]));
        assert_eq!(p["frequency"]["counts"], json!([2]));
        assert_eq!(p["frequency"]["other_count"], 1);
    }

    #[test]
    fn test_export_follows_the_view() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(4), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let (v, cache) = store.view_mut(&view).unwrap();
        v.set_sort(
            vec![SortKey {
                column: 0,
                ascending: false,
            }],
            cache.backend(),
        )
        .unwrap();
        let csv = v
            .export(
                cache.backend(),
                Some(&Rows::Range { first: 0, last: 1 }),
                &[0, 2],
                "csv",
            )
            .unwrap();
        assert_eq!(csv, "n,s\n3,item 3\n2,item 2");
    }

    fn bytes_of(df: &DataFrame) -> u64 {
        estimate_bytes(df) as u64
    }

    #[test]
    fn test_an_open_result_survives_the_budget_and_a_closed_one_does_not() {
        // Room for two results and an order table, not three results.
        let one = bytes_of(&numbers(100));
        let mut store = ResultStore::new(2 * one + 100 * 16).unwrap();
        let a = keep(&mut store, numbers(100), "a".into(), String::new());
        let b = keep(&mut store, numbers(100), "b".into(), String::new());
        let va = store.open(&a).unwrap().view;
        let vb = store.open(&b).unwrap().view;

        // Both open: a third result does not fit, and nothing is dropped.
        assert!(store
            .insert(numbers(100), "c".into(), String::new())
            .unwrap()
            .is_none());
        assert_eq!(page(&mut store, &va, 0, 0, 0), ["0"]);
        assert_eq!(page(&mut store, &vb, 0, 0, 0), ["0"]);

        // Closed, `a` gives way to a new result.
        store.close(&va).unwrap();
        let c = store
            .insert(numbers(100), "c".into(), String::new())
            .unwrap();
        assert!(c.is_some());
        let err = store.open(&a).unwrap_err().to_string();
        assert!(err.contains("no longer available"), "{err}");
        assert!(store.open(&b).is_ok());
    }

    #[test]
    fn test_a_result_that_cannot_fit_beside_open_views_drops_nothing() {
        let one = bytes_of(&numbers(100));
        let mut store = ResultStore::new(4 * one).unwrap();
        let open = keep(&mut store, numbers(100), "open".into(), String::new());
        let closed = keep(&mut store, numbers(100), "closed".into(), String::new());
        let pending = keep(&mut store, numbers(100), "pending".into(), String::new());
        store.open(&open).unwrap();
        let v = store.open(&closed).unwrap().view;
        store.close(&v).unwrap();

        // Under the budget, but not beside the open result.
        let big = numbers(350);
        assert!(bytes_of(&big) <= 4 * one);
        assert!(store
            .insert(big, "big".into(), String::new())
            .unwrap()
            .is_none());
        assert!(store.open(&closed).is_ok());
        assert!(store.open(&pending).is_ok());
    }

    #[test]
    fn test_the_oldest_unopened_result_is_dropped_to_make_room() {
        let one = bytes_of(&numbers(10));
        let mut store = ResultStore::new(2 * one).unwrap();
        let first = keep(&mut store, numbers(10), "a".into(), String::new());
        let second = keep(&mut store, numbers(10), "b".into(), String::new());
        keep(&mut store, numbers(10), "c".into(), String::new());
        assert!(store.open(&first).is_err());
        assert!(store.open(&second).is_ok());
    }

    #[test]
    fn test_a_result_bigger_than_the_budget_is_not_kept() {
        let mut store = ResultStore::new(1).unwrap();
        assert!(store
            .insert(numbers(10), "a".into(), String::new())
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_order_tables_count_against_the_budget() {
        let one = bytes_of(&numbers(1000));
        // Room for the result and less than one order table.
        let mut store = ResultStore::new(one + 1000).unwrap();
        let key = keep(&mut store, numbers(1000), "a".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let err = store
            .set_sort(
                &view,
                vec![SortKey {
                    column: 0,
                    ascending: false,
                }],
            )
            .unwrap_err();
        assert!(err.contains("memory"), "{err}");
        // The view is untouched.
        assert_eq!(page(&mut store, &view, 0, 1, 0), ["0", "1"]);
    }

    #[test]
    fn test_close_all_unpins_every_result() {
        let one = bytes_of(&numbers(100));
        let mut store = ResultStore::new(one).unwrap();
        let a = keep(&mut store, numbers(100), "a".into(), String::new());
        store.open(&a).unwrap();
        store.open(&a).unwrap();
        store.close_all();
        assert!(store.views.is_empty());
        // Unpinned, it gives way to the next result.
        assert!(store
            .insert(numbers(100), "b".into(), String::new())
            .unwrap()
            .is_some());
        assert!(store.open(&a).is_err());
        assert!(store.opened.is_empty() || !store.opened.contains_key(&a));
    }

    #[test]
    fn test_a_failed_sort_leaves_the_view_as_it_was() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(9), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        store
            .set_sort(
                &view,
                vec![SortKey {
                    column: 1,
                    ascending: false,
                }],
            )
            .unwrap();
        let before = page(&mut store, &view, 0, 8, 0);
        // A key the database cannot sort on.
        let (v, cache) = store.view_mut(&view).unwrap();
        v.columns[0].name = "no such column".into();
        assert!(v
            .set_sort(
                vec![SortKey {
                    column: 0,
                    ascending: true
                }],
                cache.backend()
            )
            .is_err());
        assert_eq!(
            v.sort,
            vec![SortKey {
                column: 1,
                ascending: false
            }]
        );
        v.columns[0].name = "n".into();
        assert_eq!(page(&mut store, &view, 0, 8, 0), before);
    }

    #[test]
    fn test_the_memory_limit_follows_the_budget() {
        assert_eq!(
            memory_limit_for(1024 * 1024 * 1024),
            Some(2048 * 1024 * 1024)
        );
        // However small the budget, DuckDB gets room to run.
        assert_eq!(memory_limit_for(1), Some(MIN_MEMORY_LIMIT));
        // A budget too large to double, or to set, leaves DuckDB's default.
        assert_eq!(memory_limit_for(u64::MAX), None);
        assert_eq!(memory_limit_for(MAX_MEMORY_LIMIT), None);

        let store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let set = store
            .cache
            .backend()
            .execute_sql("SELECT current_setting('memory_limit') AS m")
            .unwrap();
        let set = ggsql::array_util::value_to_string(set.column("m").unwrap(), 0);
        assert_eq!(set, "2.0 GiB");
    }

    #[test]
    fn test_a_failed_filter_leaves_the_view_as_it_was() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(10), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let gt = |v: &str| -> Vec<Filter> {
            serde_json::from_value(json!([{"type": "compare", "column": 0, "op": ">", "value": v}]))
                .unwrap()
        };
        store.set_filters(&view, gt("6")).unwrap();
        // Make the next rebuild fail: its order table's name is taken by a view.
        store
            .cache
            .backend()
            .execute_sql(&format!(
                "CREATE TEMP VIEW {} AS SELECT 1 AS x",
                quote_ident(&format!("__ggsql_view_{view}_1__"))
            ))
            .unwrap();
        assert!(store.set_filters(&view, gt("1")).is_err());
        let (v, _) = store.view_mut(&view).unwrap();
        assert_eq!(v.selected_rows, 3);
        assert_eq!(v.filters, gt("6"));
        assert_eq!(page(&mut store, &view, 0, 9, 0), ["7", "8", "9"]);
    }

    #[test]
    fn test_pages_are_bounded_and_never_shift_rows() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(20), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let (v, cache) = store.view_mut(&view).unwrap();
        let get = |rows: Rows| v.page(cache.backend(), &rows, &[0]);

        // Scattered rows come back in the order asked, without the rows between.
        let cells = get(Rows::Indices {
            indices: vec![15, 2, 15],
        })
        .unwrap();
        assert_eq!(cells[0], vec![json!("15"), json!("2"), json!("15")]);
        // A row outside the view is an error, not a short column.
        assert!(get(Rows::Indices {
            indices: vec![3, 20]
        })
        .unwrap_err()
        .contains("past the end"));
        // A huge range is refused, not materialized.
        assert!(get(Rows::Range {
            first: 0,
            last: usize::MAX
        })
        .unwrap()
        .first()
        .is_some_and(|c| c.len() == 20));
        let mut big = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(
            &mut big,
            numbers(view::MAX_PAGE_ROWS + 1),
            "q".into(),
            String::new(),
        );
        let bv = big.open(&key).unwrap().view;
        let (v, cache) = big.view_mut(&bv).unwrap();
        assert!(v
            .page(
                cache.backend(),
                &Rows::Range {
                    first: 0,
                    last: usize::MAX
                },
                &[0]
            )
            .unwrap_err()
            .contains("at most"));
    }

    #[test]
    fn test_running_out_of_memory_is_an_error_not_a_spill() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let setting = store
            .cache
            .backend()
            .execute_sql("SELECT current_setting('temp_directory') AS d")
            .unwrap();
        assert_eq!(
            ggsql::array_util::as_str(setting.column("d").unwrap())
                .unwrap()
                .value(0),
            ""
        );

        let key = keep(
            &mut store,
            frame("SELECT i AS n, md5(i::VARCHAR) AS s FROM range(200000) t(i)"),
            "q".into(),
            String::new(),
        );
        let view = store.open(&key).unwrap().view;
        // One thread, so the test leaves the machine to the others.
        store
            .cache
            .backend()
            .execute_sql("SET threads = 1")
            .unwrap();
        // Just above what the copy uses, so the sort is what runs out.
        let used = store
            .cache
            .backend()
            .execute_sql("SELECT CAST(SUM(memory_usage_bytes) AS BIGINT) AS n FROM duckdb_memory()")
            .unwrap();
        let used = ggsql::array_util::as_i64(used.column("n").unwrap())
            .unwrap()
            .value(0);
        store
            .cache
            .backend()
            .execute_sql(&format!("SET memory_limit = '{}B'", used + 4_000_000))
            .unwrap();
        let (v, cache) = store.view_mut(&view).unwrap();
        let err = v
            .set_sort(
                vec![SortKey {
                    column: 1,
                    ascending: true,
                }],
                cache.backend(),
            )
            .unwrap_err();
        assert!(err.to_lowercase().contains("memory"), "{err}");
        // The view is as it was, and still pages.
        assert!(v.sort.is_empty());
        assert_eq!(page(&mut store, &view, 0, 1, 0), ["0", "1"]);
    }

    #[test]
    fn test_reopening_reuses_the_cached_copy() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let key = keep(&mut store, numbers(3), "q".into(), String::new());
        let first = store.open(&key).unwrap().view;
        store.close(&first).unwrap();
        let second = store.open(&key).unwrap();
        assert_ne!(second.view, first);
        assert_eq!(second.num_rows, 3);
    }

    #[test]
    fn test_an_unknown_key_says_the_result_is_gone() {
        let mut store = ResultStore::new(DEFAULT_MAX_BYTES).unwrap();
        let err = store.open("nope").err().unwrap().to_string();
        assert!(err.contains("no longer available"), "{err}");
    }

    #[test]
    fn test_kind_of() {
        assert_eq!(kind_of("BIGINT"), "integer");
        assert_eq!(kind_of("DECIMAL(18,3)"), "floating");
        assert_eq!(kind_of("TIMESTAMP WITH TIME ZONE"), "datetime");
        assert_eq!(kind_of("DATE"), "date");
        assert_eq!(kind_of("VARCHAR"), "string");
        assert_eq!(kind_of("INTERVAL"), "unknown");
        // Names that merely contain a type's name are not that type.
        assert_eq!(kind_of("POINT"), "unknown");
        assert_eq!(kind_of("GEOMETRY"), "unknown");
        assert_eq!(kind_of("ENUM('PRINT', 'POINT')"), "string");
        assert_eq!(kind_of("TIMESTAMP_NS"), "datetime");
        assert_eq!(kind_of("TIME WITH TIME ZONE"), "time");
        assert_eq!(kind_of("BIGINT[]"), "unknown");
    }
}
