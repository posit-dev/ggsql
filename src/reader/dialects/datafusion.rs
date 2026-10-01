//! DataFusion dialect.
//!
//! Used for the in-process `adbc_datafusion` driver. DataFusion has no TIME
//! type, no recursive CTEs (series come from the `generate_series` table
//! function), and no temp tables — the full ggsql pipeline therefore needs
//! the caching reader (`duckdb+datafusion://...`); see the existing
//! `AdbcReader` test notes.

use crate::reader::SqlDialect;

/// DataFusion dialect (requested in #341).
#[derive(Debug, Default, Clone, Copy)]
pub struct DataFusionDialect;

impl SqlDialect for DataFusionDialect {
    /// DataFusion has no temp tables; connections are always wrapped in a
    /// caching reader rather than probed.
    fn requires_cache(&self) -> bool {
        true
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn time_type_name(&self) -> Option<&str> {
        None
    }

    fn sql_generate_series(&self, n: usize) -> String {
        // DataFusion's generate_series is a table function whose single
        // output column is named `value` in current releases (it was
        // `generate_series` in the DataFusion bundled with the 0.23 crate).
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT CAST(value AS DOUBLE) AS n \
               FROM generate_series(0, {n} - 1)\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("approx_percentile_cont({column}, {fraction})"))
    }

    /// DataFusion supports neither form of the default percentile
    /// construction: its `scalar_subquery_to_join` optimizer rule cannot
    /// decorrelate the NTILE(4) window variant, and its physical planner
    /// rejects a correlated scalar subquery in projection outright
    /// ("Physical plan does not support logical expression ScalarSubquery").
    /// Every caller embeds this in a `GROUP BY {groups}` query over `from`,
    /// so the native approximate aggregate is equivalent (same shape as the
    /// ClickHouse override) — and far cheaper.
    fn sql_percentile(
        &self,
        column: &str,
        fraction: f64,
        _from: &str,
        _groups: &[String],
    ) -> String {
        format!(
            "approx_percentile_cont({}, {fraction})",
            self.quote_ident(column)
        )
    }

    fn supports_spatial(&self) -> bool {
        false
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
