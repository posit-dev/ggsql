//! SQL Server (T-SQL) dialect.
//!
//! Main deviations from ANSI: `TOP`-style limiting (expressed as an outer
//! `SELECT TOP n *`), `BIT` booleans with 1/0 literals, `DATEADD` literals,
//! and `SELECT ... INTO` instead of `CREATE TABLE AS`.
//!
//! Spatial is disabled: SQL Server's geometry API is method-based
//! (`geom.STAsBinary()`) rather than the PostGIS-style function calls the
//! ANSI defaults emit, so we fail fast rather than produce broken SQL.

use crate::reader::dialects::split_cte_prefix;
use crate::reader::{wrap_with_column_aliases, SqlDialect};

/// SQL Server dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct MssqlDialect;

impl SqlDialect for MssqlDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("FLOAT")
    }

    fn string_type_name(&self) -> Option<&str> {
        Some("NVARCHAR(MAX)")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("DATETIME2")
    }

    fn boolean_type_name(&self) -> Option<&str> {
        Some("BIT")
    }

    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }

    fn sql_ceil(&self, expr: &str) -> String {
        // T-SQL has no CEIL function.
        format!("CEILING({expr})")
    }

    fn sql_with_recursive(&self) -> &'static str {
        // T-SQL CTEs are recursive by self-reference alone; the RECURSIVE
        // keyword is a syntax error.
        "WITH"
    }

    fn sql_derived_order_by(&self, ordering: &str) -> String {
        // T-SQL rejects ORDER BY in derived tables, subqueries, and CTEs
        // unless TOP, OFFSET, or FOR XML is present (error 1033); OFFSET 0
        // ROWS legitimizes the clause without changing the ordering.
        format!("ORDER BY {ordering} OFFSET 0 ROWS")
    }

    fn sql_limit(&self, query: &str, n: usize) -> String {
        // T-SQL forbids CTEs inside a derived table ("Incorrect syntax near
        // the keyword 'WITH'"), so hoist any leading WITH clause out of the
        // parenthesised wrapper.
        let __ggsql_lim__ = self.quote_ident("__ggsql_lim__");
        match split_cte_prefix(query) {
            Some((cte, body)) => {
                format!("{cte} SELECT TOP {n} * FROM ({body}) AS {__ggsql_lim__}")
            }
            None => format!("SELECT TOP {n} * FROM ({query}) AS {__ggsql_lim__}"),
        }
    }

    fn wrap_as_subquery(&self, query: &str, alias: &str) -> String {
        match split_cte_prefix(query) {
            Some((cte, body)) => format!("{cte} SELECT * FROM ({body}) AS {alias}"),
            None => format!("SELECT * FROM ({query}) AS {alias}"),
        }
    }

    fn select_from_subquery(&self, select_list: &str, query: &str, alias: &str) -> String {
        match split_cte_prefix(query) {
            Some((cte, body)) => format!("{cte} SELECT {select_list} FROM ({body}) AS {alias}"),
            None => format!("SELECT {select_list} FROM ({query}) AS {alias}"),
        }
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("DATEADD(day, {days_since_epoch}, CAST('1970-01-01' AS DATE))")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        // DATEADD's return type caps int arithmetic at ~68 years in
        // microseconds, so split into seconds + remainder microseconds.
        let secs = microseconds_since_epoch / 1_000_000;
        let micros = microseconds_since_epoch % 1_000_000;
        format!(
            "DATEADD(microsecond, {micros}, \
             DATEADD(second, {secs}, CAST('1970-01-01T00:00:00' AS DATETIME2)))"
        )
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let secs = nanoseconds_since_midnight / 1_000_000_000;
        let nanos = nanoseconds_since_midnight % 1_000_000_000;
        format!(
            "DATEADD(nanosecond, {nanos}, \
             DATEADD(second, {secs}, CAST('00:00:00' AS TIME)))"
        )
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        // SQL Server has no CREATE TABLE AS; SELECT INTO is the idiom.
        // Session-local `#temp` tables would be ideal, but ggsql references
        // materialized tables by their plain (quoted) name in later
        // statements, which `#name` would break — so we use regular tables
        // and rely on DROP for cleanup.
        let qname = self.quote_ident(name);
        let body =
            wrap_with_column_aliases(&|c: &str| self.quote_ident(c), body_sql, column_aliases);
        let __ggsql_src__ = self.quote_ident("__ggsql_src__");
        vec![
            format!("DROP TABLE IF EXISTS {}", qname),
            format!("SELECT * INTO {} FROM ({}) AS {__ggsql_src__}", qname, body),
        ]
    }

    fn supports_spatial(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_uses_top() {
        assert_eq!(
            MssqlDialect.sql_limit("SELECT a FROM t", 10),
            "SELECT TOP 10 * FROM (SELECT a FROM t) AS \"__ggsql_lim__\""
        );
    }

    #[test]
    fn limit_hoists_cte_out_of_derived_table() {
        assert_eq!(
            MssqlDialect.sql_limit("WITH c AS (SELECT 1 AS a) SELECT a FROM c", 10),
            "WITH c AS (SELECT 1 AS a) SELECT TOP 10 * FROM (SELECT a FROM c) AS \"__ggsql_lim__\""
        );
    }

    #[test]
    fn subquery_wrap_hoists_cte() {
        assert_eq!(
            MssqlDialect.wrap_as_subquery("WITH c AS (SELECT 1 AS a) SELECT a FROM c", "s"),
            "WITH c AS (SELECT 1 AS a) SELECT * FROM (SELECT a FROM c) AS s"
        );
    }

    #[test]
    fn ceil_is_ceiling() {
        assert_eq!(MssqlDialect.sql_ceil("x / 2.0"), "CEILING(x / 2.0)");
    }

    #[test]
    fn booleans_are_bit_literals() {
        assert_eq!(MssqlDialect.boolean_type_name(), Some("BIT"));
        assert_eq!(MssqlDialect.sql_boolean_literal(true), "1");
    }

    #[test]
    fn temp_table_uses_select_into() {
        let stmts = MssqlDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1 AS a");
        assert_eq!(
            stmts[1],
            "SELECT * INTO \"t\" FROM (SELECT 1 AS a) AS \"__ggsql_src__\""
        );
    }

    #[test]
    fn spatial_disabled() {
        assert!(!MssqlDialect.supports_spatial());
    }
}
