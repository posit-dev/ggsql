//! PostgreSQL SQL dialect.
//!
//! PostgreSQL is close to the ANSI defaults; overrides add native
//! `GREATEST`/`LEAST`, `GENERATE_SERIES`, ordered-set quantiles, and PostGIS
//! spatial setup.

use crate::reader::SqlDialect;

/// PostgreSQL dialect (PostGIS for spatial).
#[derive(Debug, Default, Clone, Copy)]
pub struct PostgresDialect;

impl SqlDialect for PostgresDialect {
    fn supports_spatial(&self) -> bool {
        true
    }

    fn sql_generate_series(&self, n: usize) -> String {
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT CAST(g AS DOUBLE PRECISION) AS n \
               FROM GENERATE_SERIES(0, {n} - 1) AS g\
             )"
        )
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // Postgres rejects temporal -> numeric casts.
        super::epoch_via_subtract_extract(self, expr, kind)
    }

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        format!(
            "PERCENTILE_CONT({fraction}) WITHIN GROUP (ORDER BY {column})",
            column = self.quote_ident(column)
        )
    }

    fn sql_spatial_setup(&self) -> Vec<String> {
        // Only executed when a spatial feature is actually used, so a
        // permission failure here never breaks non-spatial queries.
        vec!["CREATE EXTENSION IF NOT EXISTS postgis".into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_uses_generate_series() {
        let sql = PostgresDialect.sql_generate_series(100);
        assert!(sql.contains("GENERATE_SERIES(0, 100 - 1)"), "got: {sql}");
        assert!(sql.contains("__ggsql_seq__"), "got: {sql}");
    }

    #[test]
    fn quantile_uses_ordered_set() {
        let sql = PostgresDialect.sql_quantile("v", 0.5, crate::sql::FromItem::Table("t"), &[]);
        assert_eq!(sql, "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY \"v\")");
    }

    #[test]
    fn greatest_is_native() {
        assert_eq!(PostgresDialect.sql_greatest(&["a", "b"]), "GREATEST(a, b)");
    }
}
