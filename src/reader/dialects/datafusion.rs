//! DataFusion dialect.
//!
//! Used with the ADBC Driver Foundry's in-process datafusion driver.
//! DataFusion has no TIME type and no recursive CTEs (series come from the
//! `generate_series` table function). It rejects the TEMP keyword but
//! supports in-memory CTAS on its per-connection catalog, which provides
//! the session scoping ggsql's staged tables need — so no cache wrap.

use crate::reader::SqlDialect;

/// DataFusion dialect (requested in #341).
#[derive(Debug, Default, Clone, Copy)]
pub struct DataFusionDialect;

impl SqlDialect for DataFusionDialect {
    /// DataFusion rejects the TEMP keyword ("Temporary tables not
    /// supported") but supports in-memory CTAS, and its per-connection
    /// catalog already provides the session scoping TEMP would give — so
    /// stage internal tables as plain CREATE TABLE. Verified against the
    /// Foundry 0.27 driver; the connect-time probe re-checks this and falls
    /// back to the cache wrap if a future build regresses it.
    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = self.quote_ident(name);
        let body = crate::reader::wrap_with_column_aliases(
            &|c: &str| self.quote_ident(c),
            body_sql,
            column_aliases,
        );
        vec![
            format!("DROP TABLE IF EXISTS {}", qname),
            format!("CREATE TABLE {} AS {}", qname, body),
        ]
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
        Some(format!(
            "approx_percentile_cont({column}, {fraction})",
            column = self.quote_ident(column)
        ))
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
