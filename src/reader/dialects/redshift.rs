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
    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        super::epoch_via_subtract_extract(self, expr, kind)
    }

    fn sql_number_as_temporal(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        use crate::plot::types::CastTargetType as C;
        match kind {
            // DATEADD's count is an int, so split whole seconds from the
            // microsecond remainder; epoch microseconds overflow int on
            // their own.
            C::DateTime => format!(
                "DATEADD(microsecond, CAST({expr} AS BIGINT) % 1000000, \
                 DATEADD(second, CAST(FLOOR({expr} / 1000000.0) AS BIGINT), \
                 TIMESTAMP '1970-01-01 00:00:00'))"
            ),
            C::Date => format!("DATEADD(day, CAST({expr} AS INT), DATE '1970-01-01')"),
            _ => {
                let ty = self.type_names().datetime.unwrap_or("TIMESTAMP");
                self.sql_cast(expr, ty)
            }
        }
    }
}
