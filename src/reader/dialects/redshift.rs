//! Amazon Redshift dialect.
//!
//! Redshift is Postgres-derived, so this mirrors [`PostgresDialect`] where
//! Redshift kept the feature, and falls back to ANSI elsewhere: no
//! `GENERATE_SERIES` (recursive-CTE default applies), and native
//! (PostGIS-subset) spatial. Quantiles use the portable window-function
//! default rather than `APPROXIMATE PERCENTILE_DISC` — exact rather than
//! approximate, and the live CI leg runs the redshift driver against a
//! PostgreSQL container, which lacks the Redshift-only spelling.

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

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // Same date arithmetic as Postgres.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("({expr} - DATE '1970-01-01')"),
            C::DateTime => format!("(EXTRACT(EPOCH FROM {expr}) * 1000000)"),
            _ => format!("CAST({expr} AS DOUBLE PRECISION)"),
        }
    }
}
