//! Query results kept so a frontend can explore them after the fact.
//!
//! A frontend-agnostic API over recent results, one page at a time: open a
//! result, filter and sort it, read pages, profile columns, export a selection,
//! and get SQL reproducing the view. [`methods`] exposes it as named JSON
//! calls. Nothing here knows about any particular frontend.
//!
//! A result is held as its Arrow data until it is first opened, and is then
//! copied into a dedicated in-memory cache ([`CacheTables`], the same
//! machinery as ggsql's caching reader) where every filter, sort, page and
//! profile is SQL. The cache is the results API's own: the session's reader is
//! never wrapped, so nothing changes about the dialect the user's queries run
//! in.

mod export;
pub mod filter;
pub mod methods;
pub mod view;

use anyhow::{anyhow, Result};
use ggsql::naming::quote_ident;
use ggsql::reader::cache::{parse_human_bytes, CacheConfig, CacheTables};
use ggsql::DataFrame;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use view::ResultView;

/// The column holding each row's position in the query's output, added when a
/// result is copied into the cache. Ties in a sort break on it, so paging is
/// stable and clearing the sort restores the query's own order.
pub const ROW: &str = "__ggsql_row__";

/// The results cache's default byte budget.
const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024;

/// A result's column.
#[derive(Debug, Clone, Serialize)]
pub struct Column {
    pub name: String,
    /// The cache backend's type name, e.g. `BIGINT`.
    pub type_name: String,
    /// A frontend-neutral kind: `integer`, `floating`, `boolean`, `string`,
    /// `date`, `datetime`, `time` or `unknown`.
    pub kind: String,
}

/// A result not yet opened, held as its Arrow data.
struct Pending {
    key: String,
    title: String,
    query: String,
    df: DataFrame,
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
    /// Results not opened yet, oldest first, bounded by `max_pending`.
    pending: VecDeque<Pending>,
    max_pending: usize,
    opened: HashMap<String, Opened>,
    views: HashMap<String, ResultView>,
}

impl ResultStore {
    /// A store keeping up to `max_pending` unopened results. Opened results
    /// are bounded by the cache's byte budget, `GGSQL_RESULTS_MAX_BYTES`
    /// (default 1 GB).
    pub fn new(max_pending: usize) -> Result<Self> {
        let max_bytes = std::env::var("GGSQL_RESULTS_MAX_BYTES")
            .ok()
            .and_then(|v| parse_human_bytes(&v))
            .unwrap_or(DEFAULT_MAX_BYTES);
        Self::with_max_bytes(max_pending, max_bytes)
    }

    /// A store with an explicit byte budget for opened results.
    pub fn with_max_bytes(max_pending: usize, max_bytes: u64) -> Result<Self> {
        let config = CacheConfig {
            enabled: true,
            ttl_secs: u64::MAX,
            max_bytes,
        };
        let cache = CacheTables::in_memory(config)?;
        // An in-memory DuckDB may otherwise spill to temporary files under
        // memory pressure; results must never be written to disk.
        cache.backend().execute_sql("SET temp_directory = ''")?;
        Ok(Self {
            cache,
            pending: VecDeque::new(),
            max_pending: max_pending.max(1),
            opened: HashMap::new(),
            views: HashMap::new(),
        })
    }

    /// Keep `df`, produced by `query`, under a fresh key, which is returned.
    pub fn insert(&mut self, df: DataFrame, title: String, query: String) -> String {
        // Unguessable, so a stale link from a previous kernel process can never
        // name a result of this one.
        let key = uuid::Uuid::new_v4().simple().to_string();
        self.pending.push_back(Pending {
            key: key.clone(),
            title,
            query,
            df,
        });
        while self.pending.len() > self.max_pending {
            self.pending.pop_front();
        }
        key
    }

    fn table_name(key: &str) -> String {
        format!("__ggsql_result_{key}__")
    }

    /// Open a new view of the result `key`, copying it into the cache the
    /// first time. The result stays pinned while any view of it is open.
    pub fn open(&mut self, key: &str) -> Result<OpenedView> {
        if let Some(i) = self.pending.iter().position(|p| p.key == key) {
            let Pending {
                key,
                title,
                query,
                df,
            } = self.pending.remove(i).expect("position is in range");
            match self.copy_into_cache(&key, title.clone(), query.clone(), df.clone()) {
                Ok(opened) => {
                    self.opened.insert(key, opened);
                }
                Err(e) => {
                    // Kept, so the error says what went wrong each time
                    // rather than that the result is gone.
                    self.pending.insert(
                        i,
                        Pending {
                            key,
                            title,
                            query,
                            df,
                        },
                    );
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

    fn copy_into_cache(
        &self,
        key: &str,
        title: String,
        query: String,
        df: DataFrame,
    ) -> Result<Opened> {
        let num_rows = df.height();
        let df = unique_names(df)?;
        let rows = arrow::array::UInt64Array::from_iter_values(0..num_rows as u64);
        let df = df.with_column(ROW, Arc::new(rows))?;
        let table = Self::table_name(key);
        // Pinned from the start, so the eviction its own insertion triggers
        // can't take it.
        self.cache.put(key, &query, &table, df, true)?;

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
            })
            .collect();
        Ok(Opened {
            title,
            query,
            table,
            columns,
            num_rows,
        })
    }

    /// Close a view, dropping what it keeps in the cache. Its result is
    /// unpinned once no view of it is open, and is then evicted when the
    /// cache needs the room.
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

    /// A view and the cache it reads from.
    pub fn view_mut(&mut self, view: &str) -> Result<(&mut ResultView, &CacheTables)> {
        let v = self
            .views
            .get_mut(view)
            .ok_or_else(|| anyhow!("There is no view {view:?}"))?;
        Ok((v, &self.cache))
    }

    /// The query a view's result came from.
    pub fn query_of(&self, view: &str) -> Result<&str> {
        let v = self
            .views
            .get(view)
            .ok_or_else(|| anyhow!("There is no view {view:?}"))?;
        Ok(&self.opened[&v.key].query)
    }
}

/// `df` with each repeated column name made unique, as DuckDB does: the second
/// `a` becomes `a_1`, the third `a_2`, skipping names already taken. A query
/// can return repeated names, but a table cannot hold them.
fn unique_names(df: DataFrame) -> Result<DataFrame> {
    let names = df.get_column_names();
    let mut taken: std::collections::HashSet<String> = names.iter().cloned().collect();
    let mut seen = std::collections::HashSet::new();
    let mut changed = false;
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
            changed = true;
        }
        columns.push((unique, array.clone()));
    }
    if !changed {
        return Ok(df);
    }
    Ok(DataFrame::new(columns)?)
}

fn gone() -> anyhow::Error {
    anyhow!("This query result is no longer available. Re-run the query to explore it.")
}

/// A cache type name's neutral kind.
pub fn kind_of(type_name: &str) -> &'static str {
    let upper = type_name.to_uppercase();
    let upper = upper.as_str();
    // Nested types first: `INTEGER[]` and `STRUCT(a INTEGER)` name a scalar
    // type inside them, but are not numbers.
    let nested = upper.ends_with(']')
        || ["STRUCT", "MAP", "UNION", "LIST"]
            .iter()
            .any(|t| upper.starts_with(t));
    if nested || upper.contains("INTERVAL") {
        "unknown"
    } else if upper.contains("INT") {
        "integer"
    } else if ["FLOAT", "DOUBLE", "REAL", "NUMERIC", "DECIMAL"]
        .iter()
        .any(|t| upper.contains(t))
    {
        "floating"
    } else if upper.contains("BOOL") {
        "boolean"
    } else if upper.contains("TIMESTAMP") || upper.contains("DATETIME") {
        "datetime"
    } else if upper.contains("DATE") {
        "date"
    } else if upper.contains("TIME") {
        "time"
    } else if ["CHAR", "TEXT", "STRING", "UUID"]
        .iter()
        .any(|t| upper.contains(t))
    {
        "string"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::filter::{Capabilities, Filter, SortKey};
    use crate::results::view::Rows;
    use serde_json::json;

    const CAPS: Capabilities = Capabilities { regex: true };

    fn frame(sql: &str) -> DataFrame {
        ggsql::reader::connection::reader_from_uri("duckdb://memory")
            .unwrap()
            .execute_sql(sql)
            .unwrap()
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(5), "Query [1]".into(), "SELECT 1".into());
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(9), "q".into(), String::new());
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(10), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "compare", "column": 0, "op": ">=", "value": "4"},
            {"type": "compare", "column": 0, "op": "=", "value": "four"},
            {"type": "text", "column": 2, "match": "ends_with", "term": "7"}
        ]))
        .unwrap();
        let (v, cache) = store.view_mut(&view).unwrap();
        let errors = v.set_filters(filters, cache.backend(), CAPS).unwrap();
        assert!(errors[0].is_none());
        assert!(errors[1].as_deref().unwrap().contains("not a number"));
        assert!(errors[2].is_none());
        assert_eq!(v.selected_rows, 1);
        assert_eq!(page(&mut store, &view, 0, 0, 2), ["item 7"]);
    }

    #[test]
    fn test_dates_page_and_profile_as_dates() {
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(frame(sql), "q".into(), String::new());
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(frame("SELECT 1 AS x"), "q".into(), String::new());
        // Break the cache's bookkeeping, so copying the result in fails.
        store.cache.ensure_meta_table().unwrap();
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(
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
        let errors = v.set_filters(filters, cache.backend(), CAPS).unwrap();
        assert!(errors[0].is_some() && errors[1].is_none(), "{errors:?}");
        let sql = v.to_sql("SELECT 1", &ggsql::reader::AnsiDialect, CAPS);
        assert!(sql[0].starts_with("-- Filter left out"), "{sql:?}");
        assert!(!sql.iter().any(|l| l.contains("regexp_matches")), "{sql:?}");
        assert!(sql.iter().any(|l| l.contains("IS NOT NULL")), "{sql:?}");
    }

    #[test]
    fn test_paging_past_the_end_returns_what_there_is() {
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(3), "q".into(), String::new());
        let view = store.open(&key).unwrap().view;
        assert_eq!(page(&mut store, &view, 2, 10, 0), ["2"]);
    }

    #[test]
    fn test_profiles() {
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(
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
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(4), "q".into(), String::new());
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

    #[test]
    fn test_an_open_result_survives_the_budget_and_a_closed_one_does_not() {
        let mut store = ResultStore::with_max_bytes(4, 1).unwrap();
        let a = store.insert(numbers(3), "a".into(), String::new());
        let b = store.insert(numbers(3), "b".into(), String::new());
        let va = store.open(&a).unwrap().view;
        let vb = store.open(&b).unwrap().view;
        // Both open: both pinned, whatever the budget.
        assert_eq!(page(&mut store, &va, 0, 0, 0), ["0"]);
        assert_eq!(page(&mut store, &vb, 0, 0, 0), ["0"]);

        store.close(&va).unwrap();
        let err = store.open(&a).unwrap_err().to_string();
        assert!(err.contains("no longer available"), "{err}");
        assert!(store.open(&b).is_ok());
    }

    #[test]
    fn test_running_out_of_memory_is_an_error_not_a_spill() {
        let mut store = ResultStore::new(4).unwrap();
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

        let key = store.insert(
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
    }

    #[test]
    fn test_reopening_reuses_the_cached_copy() {
        let mut store = ResultStore::new(4).unwrap();
        let key = store.insert(numbers(3), "q".into(), String::new());
        let first = store.open(&key).unwrap().view;
        store.close(&first).unwrap();
        let second = store.open(&key).unwrap();
        assert_ne!(second.view, first);
        assert_eq!(second.num_rows, 3);
    }

    #[test]
    fn test_the_oldest_unopened_result_is_dropped_past_the_limit() {
        let mut store = ResultStore::new(2).unwrap();
        let first = store.insert(numbers(1), "a".into(), String::new());
        let second = store.insert(numbers(1), "b".into(), String::new());
        store.insert(numbers(1), "c".into(), String::new());
        assert!(store.open(&first).is_err());
        assert!(store.open(&second).is_ok());
    }

    #[test]
    fn test_an_unknown_key_says_the_result_is_gone() {
        let mut store = ResultStore::new(2).unwrap();
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
    }
}
