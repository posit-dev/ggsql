//! Hybrid reader: data source + in-process DuckDB staging.
//!
//! Wraps a primary [`Reader`] (e.g. a remote analytic backend) and a staging
//! [`DuckDBReader`]. [`Reader::register`] writes to staging; [`Reader::execute_sql`]
//! routes queries that reference registered names to staging, everything else to
//! the primary data source.
//!
//! Designed for backends where `register()` is unavailable (read-only Flight SQL
//! servers, anonymous Trino, etc.) or where round-tripping during visualization
//! iteration is wasteful. Staging in a local DuckDB sidesteps both: the primary
//! query runs against the remote source; subsequent `register()`-based machinery
//! (stat transforms, layer filters, temp-table DDL) runs against in-process DuckDB.
//!
//! # Known limitations
//!
//! A single SQL statement cannot reference both staged names and primary-data
//! tables. Queries are dispatched whole to either staging or the primary backend
//! based on whether they mention a staged name, so cross-backend joins are not
//! supported; materialize one side into staging first if you need to combine them.
//!
//! Staged data lives in the in-process DuckDB instance and is released when the
//! `HybridReader` is dropped. There is no spill-to-disk and no shared cache across
//! readers — plan staging volume against available RAM.
//!
//! All internally-generated SQL (stat transforms, layer filters, temp-table DDL)
//! is emitted in DuckDB dialect, which is why [`Reader::dialect`] on `HybridReader`
//! returns staging's dialect. That is the correct choice for queries over staged
//! data; when you need SQL targeted at the remote source (e.g. schema introspection
//! of the remote catalog), use [`HybridReader::data_dialect`] instead.

use crate::reader::{DuckDBReader, Reader, Spec, SqlDialect};
use crate::{DataFrame, Result};
use std::cell::RefCell;
use std::collections::HashSet;

pub struct HybridReader {
    data: Box<dyn Reader + Send>,
    staging: DuckDBReader,
    staged_names: RefCell<HashSet<String>>,
}

impl HybridReader {
    /// Construct a `HybridReader` from a primary data reader and a staging
    /// DuckDB instance. The staging instance is owned by the `HybridReader`
    /// and dropped with it; staged tables do not persist across constructions.
    pub fn new(data: Box<dyn Reader + Send>, staging: DuckDBReader) -> Self {
        Self {
            data,
            staging,
            staged_names: RefCell::new(HashSet::new()),
        }
    }

    /// Dialect of the primary data backend. Useful for SQL targeted at the
    /// remote source rather than the staging DuckDB (e.g. schema introspection
    /// of the remote catalog).
    pub fn data_dialect(&self) -> &dyn SqlDialect {
        self.data.dialect()
    }
}

impl Reader for HybridReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        if references_staged_name(sql, &self.staged_names.borrow()) {
            self.staging.execute_sql(sql)
        } else {
            self.data.execute_sql(sql)
        }
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.staging.register(name, df, replace)?;
        self.staged_names.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        self.staging.unregister(name)?;
        self.staged_names.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<Spec> {
        crate::reader::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn SqlDialect {
        // All generated SQL (stats, layer filters, temp-table DDL) targets
        // the staging backend, so return the staging dialect. Callers that
        // need the primary-data dialect (e.g. schema introspection of the
        // remote catalog) can access it via `HybridReader::data_dialect()`.
        self.staging.dialect()
    }
}

/// Check whether `sql` references any name in `staged_names` as a SQL
/// identifier (not as part of a longer identifier, and not inside a
/// single-quoted string literal).
///
/// Matches when the name appears bare (`SELECT * FROM orders`), as a
/// double-quoted identifier (`FROM "orders"`), or adjacent to a qualified
/// prefix (`FROM catalog.schema.orders`). Does **not** match substrings of
/// longer identifiers (`orders_detail`) or string-literal contents
/// (`'orders of magnitude'`).
///
/// This is deliberately a lightweight scanner — it doesn't fully parse SQL.
/// False-positive cases we accept:
/// - Backslash-escaped quotes inside string literals (SQL standard escapes
///   a single quote as `''`, which we do handle).
/// - Comments containing what looks like an identifier: a primary-data
///   query whose only mention of a staged name is inside a SQL comment
///   will be misrouted to staging and fail with a clear error rather than
///   succeeding against the primary backend.
fn references_staged_name(sql: &str, staged_names: &HashSet<String>) -> bool {
    staged_names
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
/// string literal. SQL-standard doubled-single-quote (`''`) is an escape
/// that keeps us inside the literal.
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
    fn test_references_staged_name_empty_set() {
        let set = HashSet::new();
        assert!(!references_staged_name("SELECT * FROM foo", &set));
    }

    #[test]
    fn test_references_staged_name_no_match() {
        let mut set = HashSet::new();
        set.insert("__ggsql_global_abc123__".to_string());
        assert!(!references_staged_name(
            "SELECT * FROM iceberg.dse.foo",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_match() {
        let mut set = HashSet::new();
        set.insert("__ggsql_global_abc123__".to_string());
        assert!(references_staged_name(
            "SELECT * FROM __ggsql_global_abc123__ WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_rejects_longer_identifier() {
        // The query references `orders_detail`, NOT `orders`. Must not route.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_staged_name(
            "SELECT * FROM orders_detail WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_rejects_prefix_of_longer_identifier() {
        // The name is `col`; query uses `col_id`. Must not route.
        let mut set = HashSet::new();
        set.insert("col".to_string());
        assert!(!references_staged_name("SELECT col_id FROM users", &set));
    }

    #[test]
    fn test_references_staged_name_rejects_inside_string_literal() {
        // `orders` appears only inside a string literal. Must not route.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_staged_name(
            "SELECT 'orders of magnitude' AS label",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_matches_quoted_identifier() {
        // Double-quoted identifier — our boundary check lets this through
        // because `"` is not an identifier char.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(r#"SELECT * FROM "orders""#, &set));
    }

    #[test]
    fn test_references_staged_name_matches_qualified_reference() {
        // `catalog.schema.orders` — the dot is a non-identifier byte, so
        // `orders` at the end still matches.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(
            "SELECT * FROM catalog.schema.orders WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_handles_escaped_quotes_in_literal() {
        // SQL-standard '' is an escaped quote inside a string literal, so
        // the staged name appearing after should still be detected as
        // outside any literal.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(
            "SELECT 'it''s fine' FROM orders",
            &set
        ));
    }

    #[test]
    fn test_register_delegates_to_staging_and_tracks_name() {
        use crate::df;
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        let df = df! { "x" => vec![1_i64, 2, 3] }.unwrap();
        reader.register("my_table", df, true).unwrap();

        // The name is tracked so subsequent queries route correctly.
        assert!(reader.staged_names.borrow().contains("my_table"));
    }

    #[test]
    fn test_execute_sql_routes_staged_queries_to_staging() {
        use crate::array_util::as_i64;
        use crate::df;
        // Make the data reader a DuckDB that does NOT have the table.
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        let df = df! { "x" => vec![1_i64, 2, 3] }.unwrap();
        reader.register("my_table", df, true).unwrap();

        // Query referencing the registered name routes to staging (which has it)
        let result = reader
            .execute_sql("SELECT COUNT(*) AS n FROM my_table")
            .unwrap();
        let n = as_i64(result.column("n").unwrap()).unwrap().value(0);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_execute_sql_routes_unstaged_queries_to_data() {
        use crate::array_util::as_i64;
        use crate::df;
        // Data reader has a distinctive table; staging is empty.
        let data_reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_reader
            .register("data_table", df! { "y" => vec![42_i64] }.unwrap(), true)
            .unwrap();

        let data = Box::new(data_reader) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        // Nothing registered in staging; query for `data_table` must hit data reader
        let result = reader.execute_sql("SELECT y FROM data_table").unwrap();
        let y = as_i64(result.column("y").unwrap()).unwrap().value(0);
        assert_eq!(y, 42);
    }

    #[test]
    fn test_unregister_delegates_to_staging_and_untracks() {
        use crate::df;
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        reader
            .register("tmp", df! { "x" => vec![1_i64] }.unwrap(), true)
            .unwrap();
        assert!(reader.staged_names.borrow().contains("tmp"));

        reader.unregister("tmp").unwrap();
        assert!(!reader.staged_names.borrow().contains("tmp"));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_dialect_returns_staging_not_data() {
        use crate::reader::SqliteReader;
        // Use a SqliteReader on the data side so the data dialect (SQLite,
        // CASE-WHEN fallback for sql_greatest) differs from the staging
        // dialect (DuckDB, native GREATEST). This way the test would fail if
        // the impl returned the data dialect by mistake.
        let data = Box::new(SqliteReader::new().unwrap()) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        // dialect() returns the staging (DuckDB) dialect.
        let greatest = reader.dialect().sql_greatest(&["a", "b"]);
        assert_eq!(greatest, "GREATEST(a, b)");

        // data_dialect() returns the data-side (SQLite) dialect, whose
        // sql_greatest falls back to a portable CASE form.
        let data_greatest = reader.data_dialect().sql_greatest(&["a", "b"]);
        assert_ne!(data_greatest, "GREATEST(a, b)");
        assert!(
            data_greatest.contains("CASE"),
            "expected SQLite's CASE fallback, got: {data_greatest}"
        );
    }

    #[test]
    fn test_query_referencing_both_staged_and_remote_routes_to_staging() {
        use crate::df;
        // Primary has `remote_only` and ALSO `staged_only` (with different
        // values from the staging copy). Staging only has `staged_only`. If
        // the router incorrectly sent the query to the data side, the join
        // would succeed against the primary's two tables. Since routing must
        // pick staging on `staged_only`, the query fails because staging
        // lacks `remote_only`.
        let data_reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_reader
            .register(
                "remote_only",
                df! { "y" => vec![10_i64, 20] }.unwrap(),
                true,
            )
            .unwrap();
        data_reader
            .register(
                "staged_only",
                df! { "x" => vec![999_i64, 999] }.unwrap(),
                true,
            )
            .unwrap();
        let data = Box::new(data_reader) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        reader
            .register("staged_only", df! { "x" => vec![1_i64, 2] }.unwrap(), true)
            .unwrap();

        // Query references BOTH names. Routing matches on `staged_only`, so the
        // whole query goes to staging — which doesn't have `remote_only`. The
        // wrong-route case (data side) would silently succeed because the
        // primary has both tables. So `is_err()` plus a staging-side error
        // message mentioning `remote_only` confirms correct routing.
        let result = reader.execute_sql("SELECT s.x, r.y FROM staged_only s, remote_only r");
        assert!(
            result.is_err(),
            "cross-side query must error when staging lacks the remote table"
        );
        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("remote_only"),
            "expected staging-side error mentioning the missing `remote_only` table; got: {err_msg}"
        );
    }
}
