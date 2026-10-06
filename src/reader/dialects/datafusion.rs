//! DataFusion dialect.
//!
//! Used with the ADBC Driver Foundry's in-process datafusion driver.
//! DataFusion has no TIME type and no recursive CTEs (series come from the
//! `generate_series` table function). It rejects the TEMP keyword but
//! supports in-memory CTAS on its per-connection catalog, which provides
//! the session scoping ggsql's staged tables need — so no cache wrap.

use crate::reader::SqlDialect;

/// DataFusion dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct DataFusionDialect;

impl SqlDialect for DataFusionDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            time: None,
            ..crate::reader::TypeNames::ANSI
        }
    }

    /// DataFusion rejects the TEMP keyword ("Temporary tables not
    /// supported") but supports in-memory CTAS, and its per-connection
    /// catalog already provides the session scoping TEMP would give — so
    /// stage internal tables as plain CREATE TABLE. The connect-time probe
    /// re-checks this and falls back to the cache wrap if a future driver
    /// build regresses it.
    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::DropThenCreate
    }

    fn sql_generate_series(&self, n: usize) -> String {
        // DataFusion's generate_series is a table function whose single
        // output column is named `value`.
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT CAST(value AS DOUBLE) AS n \
               FROM generate_series(0, {n} - 1)\
             )"
        )
    }

    /// DataFusion's physical planner rejects a correlated scalar subquery in
    /// projection outright ("Physical plan does not support logical
    /// expression ScalarSubquery"), so the default construction fails. Every
    /// caller embeds this in a `GROUP BY {groups}` query over `from`, so the
    /// native approximate aggregate is equivalent — and far cheaper.
    fn sql_quantile(&self, column: &str, fraction: f64, _from: &str, _groups: &[String]) -> String {
        format!(
            "approx_percentile_cont({}, {fraction})",
            self.quote_ident(column)
        )
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // DataFusion rejects temporal -> numeric casts ("Unsupported CAST
        // from Date32 to Float64"); EXTRACT EPOCH yields seconds for both
        // dates and timestamps, so scale to ggsql's epoch units.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("(EXTRACT(EPOCH FROM {expr}) / 86400)"),
            C::DateTime => format!("(EXTRACT(EPOCH FROM {expr}) * 1000000)"),
            _ => format!("CAST({expr} AS DOUBLE)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_uses_table_function() {
        let sql = DataFusionDialect.sql_generate_series(5);
        assert!(sql.contains("FROM generate_series(0, 5 - 1)"), "got: {sql}");
    }
}
