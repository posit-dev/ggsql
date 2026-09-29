//! SQL Server (T-SQL) dialect.
//!
//! Main deviations from ANSI: `TOP`-style limiting (expressed as an outer
//! `SELECT TOP n *`), `BIT` booleans with 1/0 literals, `DATEADD` literals,
//! and `SELECT ... INTO` instead of `CREATE TABLE AS`.
//!
//! Spatial is disabled: SQL Server's geometry API is method-based
//! (`geom.STAsBinary()`) rather than the PostGIS-style function calls the
//! ANSI defaults emit, so we fail fast rather than produce broken SQL.

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

    fn sql_limit(&self, query: &str, n: usize) -> String {
        format!("SELECT TOP {n} * FROM ({query}) AS \"__ggsql_lim__\"")
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
        let body = wrap_with_column_aliases(body_sql, column_aliases);
        vec![
            format!("DROP TABLE IF EXISTS {}", qname),
            format!(
                "SELECT * INTO {} FROM ({}) AS \"__ggsql_src__\"",
                qname, body
            ),
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
