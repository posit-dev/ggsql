//! Amazon Redshift dialect.
//!
//! Redshift is Postgres-derived, so this mirrors [`PostgresDialect`] where
//! Redshift kept the feature, and falls back to ANSI elsewhere: no
//! `GENERATE_SERIES` (recursive-CTE default applies), approximate quantiles
//! via `APPROXIMATE PERCENTILE_DISC`, and native (PostGIS-subset) spatial.

use crate::reader::SqlDialect;

/// Amazon Redshift dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct RedshiftDialect;

impl SqlDialect for RedshiftDialect {
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

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "APPROXIMATE PERCENTILE_DISC({fraction}) WITHIN GROUP (ORDER BY {column})"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_uses_approximate_percentile_disc() {
        let sql = RedshiftDialect.sql_quantile_inline("\"v\"", 0.5).unwrap();
        assert!(
            sql.contains("APPROXIMATE PERCENTILE_DISC(0.5)"),
            "got: {sql}"
        );
    }
}
