//! Databricks (Spark SQL) dialect.
//!
//! Main deviations from ANSI: backtick quoting, Spark type names, `SEQUENCE`
//! series, `percentile_approx`, `DATE_ADD` literals, and temp views instead
//! of temp tables (Spark has no `CREATE TEMP TABLE AS`).

use crate::reader::{wrap_with_column_aliases, SqlDialect};

/// Databricks / Spark SQL dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct DatabricksDialect;

impl SqlDialect for DatabricksDialect {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    // Note: Spark treats double quotes as string literals, and SQL warehouses
    // reject `SET spark.sql.ansi.doubleQuotedIdentifiers` ("Configuration ...
    // is not available"), so the MySQL-style session-init fix was never
    // available here — ggsql-internal identifiers reach this dialect through
    // `quote_ident` at every emission site instead.

    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn string_type_name(&self) -> Option<&str> {
        Some("STRING")
    }

    fn time_type_name(&self) -> Option<&str> {
        // Spark SQL has no TIME type.
        None
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("GREATEST({})", exprs.join(", "))
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("LEAST({})", exprs.join(", "))
    }

    fn sql_generate_series(&self, n: usize) -> String {
        // Spark rejects a generator nested inside another expression
        // ("generator is not supported: nested in expressions"), so the
        // explode must stand alone in an inner SELECT and the CAST moves
        // outside it.
        format!(
            "`__ggsql_seq__`(n) AS (\
               SELECT CAST(n AS DOUBLE) AS n FROM (\
                 SELECT explode(sequence(0, {n} - 1)) AS n\
               )\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "percentile_approx({column}, {fraction})",
            column = self.quote_ident(column)
        ))
    }

    fn sql_percentile(
        &self,
        column: &str,
        fraction: f64,
        _from: &str,
        _groups: &[String],
    ) -> String {
        // Spark forbids correlated scalar subqueries in the SELECT list of a
        // GROUP BY query ("is neither present in GROUP BY, nor in an
        // aggregate function"), so the ANSI correlated-subquery fallback
        // fails outright. Return a plain aggregate instead: it computes the
        // percentile within the caller's own grouping context, which is the
        // same semantics the correlated form encodes (same trick as
        // ClickHouse's quantileExactInclusive override). Approximate, which
        // is acceptable for boxplot/density statistics.
        format!(
            "percentile_approx({column}, {fraction})",
            column = self.quote_ident(column)
        )
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("DATE_ADD(DATE '1970-01-01', {days_since_epoch})")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!(
            "TIMESTAMP '1970-01-01 00:00:00' + INTERVAL {microseconds_since_epoch} MICROSECONDS"
        )
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = self.quote_ident(name);
        let body =
            wrap_with_column_aliases(&|c: &str| self.quote_ident(c), body_sql, column_aliases);
        vec![format!("CREATE OR REPLACE TEMP VIEW {} AS {}", qname, body)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_uses_sequence_explode() {
        let sql = DatabricksDialect.sql_generate_series(8);
        assert!(sql.contains("sequence(0, 8 - 1)"), "got: {sql}");
    }

    #[test]
    fn no_time_type() {
        assert_eq!(DatabricksDialect.time_type_name(), None);
    }

    #[test]
    fn temp_table_is_temp_view() {
        let stmts = DatabricksDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1");
        assert_eq!(stmts, vec!["CREATE OR REPLACE TEMP VIEW `t` AS SELECT 1"]);
    }
}
