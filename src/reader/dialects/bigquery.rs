//! Google BigQuery dialect.
//!
//! Main deviations from ANSI: backtick quoting, `SAFE_CAST`, BigQuery type
//! names (`INT64`, `FLOAT64`, `STRING`), `GENERATE_ARRAY` series,
//! `APPROX_QUANTILES`, and `DATE_ADD`/`TIMESTAMP_ADD` literals.
//!
//! Spatial note: BigQuery `GEOGRAPHY` is WGS84-only and cannot reproject.
//! Basic ingestion (`ST_AsBinary`) and envelopes work, but any CRS
//! transformation will fail server-side; there is no way to express that in
//! the `SqlDialect` surface today, so it is documented rather than guarded.

use crate::reader::SqlDialect;

/// BigQuery dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct BigQueryDialect;

impl SqlDialect for BigQueryDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("FLOAT64"),
            integer: Some("INT64"),
            string: Some("STRING"),
            boolean: Some("BOOL"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn sql_cast(&self, expr: &str, type_name: &str) -> String {
        format!("SAFE_CAST({} AS {})", expr, type_name)
    }

    fn sql_generate_series(&self, n: usize) -> String {
        format!(
            "`__ggsql_seq__`(n) AS (\
               SELECT CAST(g AS FLOAT64) AS n \
               FROM UNNEST(GENERATE_ARRAY(0, {n} - 1)) AS g\
             )"
        )
    }

    fn sql_quantile(&self, column: &str, fraction: f64, _from: &str, _groups: &[String]) -> String {
        // APPROX_QUANTILES(x, 100) returns an array of 101 boundaries.
        // BigQuery rejects the correlated scalar subquery of the default; the
        // native aggregate computes within the caller's GROUP BY, so `from`
        // and `groups` are unused.
        let offset = (fraction * 100.0).round() as i64;
        format!(
            "APPROX_QUANTILES({column}, 100)[SAFE_OFFSET({offset})]",
            column = self.quote_ident(column)
        )
    }

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::CreateOrReplaceTemp
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("DATE_ADD(DATE '1970-01-01', INTERVAL {days_since_epoch} DAY)")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!(
            "TIMESTAMP_ADD(TIMESTAMP '1970-01-01 00:00:00', INTERVAL {microseconds_since_epoch} MICROSECOND)"
        )
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let micros = nanoseconds_since_midnight / 1_000;
        format!("TIME_ADD(TIME '00:00:00', INTERVAL {micros} MICROSECOND)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_use_backticks() {
        assert_eq!(BigQueryDialect.quote_ident("col"), "`col`");
    }

    #[test]
    fn casts_are_safe() {
        assert_eq!(
            BigQueryDialect.sql_cast("`x`", "FLOAT64"),
            "SAFE_CAST(`x` AS FLOAT64)"
        );
    }

    #[test]
    fn series_uses_generate_array() {
        let sql = BigQueryDialect.sql_generate_series(10);
        assert!(sql.contains("GENERATE_ARRAY(0, 10 - 1)"), "got: {sql}");
    }

    #[test]
    fn quantile_uses_approx_quantiles() {
        let sql = BigQueryDialect.sql_quantile("v", 0.25, "t", &[]);
        assert_eq!(sql, "APPROX_QUANTILES(`v`, 100)[SAFE_OFFSET(25)]");
    }

    #[test]
    fn quantile_is_not_correlated() {
        // sql_quantile must not emit a correlated subquery — BigQuery
        // rejects those. It uses the APPROX_QUANTILES aggregate, valid
        // inside the GROUP BY queries that boxplot and density build.
        let sql = BigQueryDialect.sql_quantile("v", 0.75, "SELECT * FROM t", &["g".to_string()]);
        assert_eq!(sql, "APPROX_QUANTILES(`v`, 100)[SAFE_OFFSET(75)]");
        assert!(!sql.contains("SELECT"), "must not be a subquery: {sql}");
    }

    #[test]
    fn temp_table_sql_uses_create_or_replace() {
        let stmts = BigQueryDialect.create_or_replace_temp_table_sql(
            "__ggsql_data",
            &["x".to_string()],
            "SELECT 1",
        );
        assert_eq!(
            stmts,
            vec![
                "CREATE OR REPLACE TEMP TABLE `__ggsql_data` AS WITH `__ggsql_aliased__`(`x`) AS (SELECT 1) SELECT * FROM `__ggsql_aliased__`"
                    .to_string()
            ]
        );
    }
}
