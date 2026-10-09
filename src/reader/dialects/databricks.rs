//! Databricks (Spark SQL) dialect.
//!
//! Main deviations from ANSI: backtick quoting, Spark type names, `SEQUENCE`
//! series, `percentile_approx`, `DATE_ADD` literals, and temp views instead
//! of temp tables (Spark has no `CREATE TEMP TABLE AS`).

use crate::reader::SqlDialect;

/// Databricks / Spark SQL dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct DatabricksDialect;

impl SqlDialect for DatabricksDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            string: Some("STRING"),
            time: None,
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn quote_ident(&self, name: &str) -> String {
        super::backtick_quote_ident(name)
    }

    // Spark treats double quotes as string literals, and SQL warehouses
    // reject `SET spark.sql.ansi.doubleQuotedIdentifiers` ("Configuration ...
    // is not available"), so ggsql-internal identifiers reach this dialect
    // through `quote_ident` at every emission site.

    fn sql_generate_series(&self, n: usize) -> String {
        // Spark rejects a generator nested inside another expression
        // ("generator is not supported: nested in expressions"), so the
        // explode must stand alone in an inner SELECT and the CAST moves
        // outside it.
        let seq = self.quote_ident("__ggsql_seq__");
        format!(
            "{seq}(n) AS (\
               SELECT CAST(n AS DOUBLE) AS n FROM (\
                 SELECT explode(sequence(0, {n} - 1)) AS n\
               )\
             )"
        )
    }

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        // Spark forbids correlated scalar subqueries in the SELECT list of a
        // GROUP BY query ("is neither present in GROUP BY, nor in an
        // aggregate function"), so the correlated-subquery default fails
        // outright. The plain aggregate computes the percentile within the
        // caller's own grouping context, which is the same semantics the
        // correlated form encodes — `from` and `groups` are unused.
        // Approximate, which is acceptable for boxplot/density statistics.
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

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::CreateOrReplaceTempView
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
        assert_eq!(DatabricksDialect.type_names().time, None);
    }

    #[test]
    fn temp_table_is_temp_view() {
        let stmts = DatabricksDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1");
        assert_eq!(stmts, vec!["CREATE OR REPLACE TEMP VIEW `t` AS SELECT 1"]);
    }
}
