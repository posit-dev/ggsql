//! Query results retained so the data explorer can open them after the fact.
//!
//! Each result is copied into a private in-memory DuckDB rather than queried
//! where it came from: the explorer pages, sorts and profiles in one dialect
//! whatever the source database, a remote query is never re-run per page, and
//! the user's session never sees the tables.

use anyhow::{anyhow, Result};
use ggsql::reader::{connection::reader_from_uri, Reader};
use ggsql::DataFrame;
use std::collections::VecDeque;

/// The catalog and schema a result table lives in. `register` creates a
/// temporary table, and DuckDB keeps those in its `temp` catalog.
pub const CATALOG: &str = "temp";
pub const SCHEMA: &str = "main";

/// One retained result.
pub struct StoredResult {
    pub key: String,
    /// The table holding the rows in the private database.
    pub table: String,
    /// Shown as the data explorer's tab title.
    pub title: String,
    /// The SQL that produced it, which Convert to Code wraps.
    pub query: String,
}

/// Retained query results, oldest first, bounded by `max`.
///
/// Only results no data explorer has open count against the bound and are
/// evicted, so an open explorer never loses its table.
pub struct ResultStore {
    reader: Box<dyn Reader + Send>,
    entries: VecDeque<StoredResult>,
    max: usize,
}

impl ResultStore {
    pub fn new(max: usize) -> Result<Self> {
        Ok(Self {
            reader: reader_from_uri("duckdb://memory")?,
            entries: VecDeque::new(),
            max: max.max(1),
        })
    }

    /// The private database holding every retained result.
    pub fn reader(&self) -> &dyn Reader {
        &*self.reader
    }

    /// Retain `df`, produced by `query`, under a fresh key, which is returned.
    ///
    /// `in_use` says whether an explorer has a result open; the oldest of the
    /// rest are evicted to stay within the limit.
    pub fn insert(
        &mut self,
        df: DataFrame,
        title: String,
        query: String,
        in_use: impl Fn(&str) -> bool,
    ) -> Result<String> {
        // Unguessable, so a stale link from a previous kernel process can never
        // name a result of this one.
        let key = uuid::Uuid::new_v4().simple().to_string();
        let table = format!("__ggsql_result_{}__", key);
        self.reader.register(&table, df, false)?;
        self.entries.push_back(StoredResult {
            key: key.clone(),
            table,
            title,
            query,
        });

        let mut idle = self.entries.iter().filter(|r| !in_use(&r.key)).count();
        let mut i = 0;
        while idle > self.max && i < self.entries.len() {
            if in_use(&self.entries[i].key) {
                i += 1;
                continue;
            }
            if let Some(old) = self.entries.remove(i) {
                if let Err(e) = self.reader.unregister(&old.table) {
                    tracing::warn!("Failed to drop retained result {}: {}", old.key, e);
                }
            }
            idle -= 1;
        }
        Ok(key)
    }

    pub fn get(&self, key: &str) -> Result<&StoredResult> {
        self.entries.iter().find(|r| r.key == key).ok_or_else(|| {
            anyhow!("This query result is no longer available. Re-run the query to explore it.")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(n: i64) -> DataFrame {
        ggsql::reader::connection::reader_from_uri("duckdb://memory")
            .unwrap()
            .execute_sql(&format!("SELECT {} AS x", n))
            .unwrap()
    }

    #[test]
    fn test_a_retained_result_is_queryable() {
        let mut store = ResultStore::new(4).unwrap();
        let key = store
            .insert(frame(7), "Query [1]".into(), String::new(), |_| false)
            .unwrap();

        let table = &store.get(&key).unwrap().table;
        let df = store
            .reader()
            .execute_sql(&format!(
                "SELECT x FROM {}",
                ggsql::naming::quote_ident(table)
            ))
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_the_oldest_result_is_evicted_past_the_limit() {
        let mut store = ResultStore::new(2).unwrap();
        let first = store
            .insert(frame(1), "a".into(), String::new(), |_| false)
            .unwrap();
        let second = store
            .insert(frame(2), "b".into(), String::new(), |_| false)
            .unwrap();
        store
            .insert(frame(3), "c".into(), String::new(), |_| false)
            .unwrap();

        assert!(store.get(&first).is_err());
        assert!(store.get(&second).is_ok());
    }

    #[test]
    fn test_a_result_in_use_is_never_evicted() {
        let mut store = ResultStore::new(1).unwrap();
        let open = store
            .insert(frame(1), "a".into(), String::new(), |_| false)
            .unwrap();
        let pinned = |k: &str| k == open;
        let second = store
            .insert(frame(2), "b".into(), String::new(), pinned)
            .unwrap();
        let third = store
            .insert(frame(3), "c".into(), String::new(), pinned)
            .unwrap();

        assert!(store.get(&open).is_ok());
        assert!(store.get(&second).is_err());
        assert!(store.get(&third).is_ok());
    }

    #[test]
    fn test_an_unknown_key_says_the_result_is_gone() {
        let store = ResultStore::new(2).unwrap();
        let err = store.get("nope").err().unwrap().to_string();
        assert!(err.contains("no longer available"), "{err}");
    }
}
