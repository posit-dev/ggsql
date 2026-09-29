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
        format!(
            "`__ggsql_seq__`(n) AS (\
               SELECT CAST(explode(sequence(0, {n} - 1)) AS DOUBLE) AS n\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("percentile_approx({column}, {fraction})"))
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
        let body = wrap_with_column_aliases(body_sql, column_aliases);
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
