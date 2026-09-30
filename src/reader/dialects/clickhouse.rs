//! ClickHouse dialect.
//!
//! Main deviations from ANSI: backtick quoting, `Nullable(...)` cast targets
//! (ClickHouse cannot cast `NULL` to a non-nullable type), `numbers()` series,
//! `quantileExactInclusive(f)(col)` aggregates (matching the linear-
//! interpolation semantics of `QUANTILE_CONT`), and `stddevPop`/`varPop`
//! naming. `greatest`/`least` arguments are cast to Float64 because
//! ClickHouse has no common supertype for UInt64 (the type of `count()` and
//! `numbers()`) and Float64.
//!
//! Spatial is disabled: ClickHouse's geo functions cover only trivial point
//! distances, not the OGC `ST_` surface the defaults assume.
//!
//! Assumes ClickHouse 26.8 or newer (correlated subqueries, `IS NOT DISTINCT
//! FROM` in any clause). Override SQL verified against a live server in
//! PR #535.

use crate::reader::{default_sql_aggregate, wrap_with_column_aliases, SqlDialect};

/// ClickHouse dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClickHouseDialect;

/// Comma-separated argument list with every expression cast to Float64.
fn float_args(exprs: &[&str]) -> String {
    exprs
        .iter()
        .map(|e| format!("toFloat64({e})"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl SqlDialect for ClickHouseDialect {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("Nullable(Float64)")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("Nullable(Int64)")
    }

    fn string_type_name(&self) -> Option<&str> {
        Some("Nullable(String)")
    }

    fn date_type_name(&self) -> Option<&str> {
        Some("Nullable(Date32)")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("Nullable(DateTime64(6))")
    }

    /// ClickHouse has no portable time-of-day type; time columns are left as is.
    fn time_type_name(&self) -> Option<&str> {
        None
    }

    fn boolean_type_name(&self) -> Option<&str> {
        Some("Nullable(Bool)")
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("greatest({})", float_args(exprs))
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("least({})", float_args(exprs))
    }

    fn sql_select_replace(
        &self,
        expr: &str,
        col: &str,
        from: &str,
        _all_columns: &[String],
    ) -> String {
        if expr == col {
            return format!("SELECT * FROM ({from})");
        }
        format!("SELECT * REPLACE ({expr} AS {col}) FROM ({from})")
    }

    fn sql_generate_series(&self, n: usize) -> String {
        format!(
            "`__ggsql_seq__`(n) AS (\
               SELECT toFloat64(number) AS n FROM numbers({n})\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("quantileExactInclusive({fraction})({column})"))
    }

    /// Every caller embeds this in a `GROUP BY {groups}` query over `from`, so
    /// the native aggregate is equivalent to the correlated scalar subquery
    /// other dialects produce, and far cheaper.
    fn sql_percentile(
        &self,
        column: &str,
        fraction: f64,
        _from: &str,
        _groups: &[String],
    ) -> String {
        format!("quantileExactInclusive({fraction})({column})")
    }

    fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
        match name {
            "sdev" => Some(format!("stddevPop({})", qcol)),
            "var" => Some(format!("varPop({})", qcol)),
            "se" => Some(format!("(stddevPop({c}) / sqrt(count({c})))", c = qcol)),
            _ => default_sql_aggregate(&|c: &str| self.quote_ident(c), name, qcol),
        }
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("toDate32({days_since_epoch})")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!("fromUnixTimestamp64Micro({microseconds_since_epoch})")
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
        vec![
            format!("DROP TEMPORARY TABLE IF EXISTS {}", qname),
            format!("CREATE TEMPORARY TABLE {} AS {}", qname, body),
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
    fn type_names_are_nullable() {
        let d = ClickHouseDialect;
        assert_eq!(d.number_type_name(), Some("Nullable(Float64)"));
        assert_eq!(d.integer_type_name(), Some("Nullable(Int64)"));
        assert_eq!(d.string_type_name(), Some("Nullable(String)"));
        assert_eq!(d.datetime_type_name(), Some("Nullable(DateTime64(6))"));
    }

    #[test]
    fn series_uses_numbers() {
        let sql = ClickHouseDialect.sql_generate_series(1000);
        assert!(sql.contains("numbers(1000)"), "got: {sql}");
    }

    #[test]
    fn aggregates_use_clickhouse_names() {
        assert_eq!(
            ClickHouseDialect.sql_aggregate("sdev", "`v`").as_deref(),
            Some("stddevPop(`v`)")
        );
        assert_eq!(
            ClickHouseDialect.sql_aggregate("mean", "`v`").as_deref(),
            Some("AVG(`v`)")
        );
    }

    #[test]
    fn quantile_uses_exact_inclusive() {
        assert_eq!(
            ClickHouseDialect.sql_quantile_inline("`v`", 0.9).as_deref(),
            Some("quantileExactInclusive(0.9)(`v`)")
        );
        assert_eq!(
            ClickHouseDialect.sql_percentile("`v`", 0.5, "SELECT * FROM t", &[]),
            "quantileExactInclusive(0.5)(`v`)"
        );
    }

    #[test]
    fn greatest_least_cast_to_float64() {
        let sql = ClickHouseDialect.sql_greatest(&["a", "b"]);
        assert_eq!(sql, "greatest(toFloat64(a), toFloat64(b))");
    }

    #[test]
    fn temp_tables_use_temporary_keyword() {
        let stmts =
            ClickHouseDialect.create_or_replace_temp_table_sql("__ggsql_data", &[], "SELECT 1");
        assert_eq!(stmts[0], "DROP TEMPORARY TABLE IF EXISTS `__ggsql_data`");
        assert_eq!(
            stmts[1],
            "CREATE TEMPORARY TABLE `__ggsql_data` AS SELECT 1"
        );
    }

    #[test]
    fn date_and_datetime_literals() {
        assert_eq!(ClickHouseDialect.sql_date_literal(30), "toDate32(30)");
        assert_eq!(
            ClickHouseDialect.sql_datetime_literal(1_500_000),
            "fromUnixTimestamp64Micro(1500000)"
        );
    }
}
