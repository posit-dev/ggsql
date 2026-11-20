//! Amazon Redshift dialect.
//!
//! Redshift is Postgres-derived, so this mirrors [`PostgresDialect`] where
//! Redshift kept the feature, and falls back to ANSI elsewhere: no
//! `GENERATE_SERIES` (recursive-CTE default applies). Redshift has a
//! PostGIS-subset spatial surface, but spatial is not enabled here —
//! `supports_spatial()` keeps the `false` default. Quantiles use the
//! portable window-function
//! default rather than `APPROXIMATE PERCENTILE_DISC` — exact rather than
//! approximate, and the live CI leg runs the redshift driver against a
//! PostgreSQL container, which lacks the Redshift-only spelling.

use crate::reader::SqlDialect;

/// Amazon Redshift dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct RedshiftDialect;

impl SqlDialect for RedshiftDialect {
    // Unanchored, with the case flag as a parameter.
    fn sql_regex_match(&self, expr: &str, pattern: &str, case_sensitive: bool) -> Option<String> {
        let flags = if case_sensitive { "c" } else { "i" };
        Some(format!("REGEXP_COUNT({expr}, {pattern}, 1, '{flags}') > 0"))
    }

    fn sql_string_literal(&self, value: &str) -> String {
        super::backslash_quote_literal(value)
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        super::epoch_via_subtract_extract(self, expr, kind)
    }
}
