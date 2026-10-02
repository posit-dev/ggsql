//! MonetDB dialect.
//!
//! MonetDB is strongly SQL-standard (like its CWI sibling DuckDB), so the
//! ANSI defaults mostly apply. Overrides: native `quantile` aggregate and
//! DOUBLE type naming. Spatial goes through the optional `geom` module,
//! which exposes a PostGIS-like `ST_` surface compatible with the defaults.

use crate::reader::SqlDialect;

/// MonetDB dialect (requested in #509).
#[derive(Debug, Default, Clone, Copy)]
pub struct MonetDbDialect;

impl SqlDialect for MonetDbDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "QUANTILE({column}, {fraction})",
            column = self.quote_ident(column)
        ))
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // MonetDB has no direct temporal -> double cast ("types date and
        // double are not equal"); EXTRACT EPOCH yields seconds.
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
    fn temporal_as_number_uses_epoch_extract() {
        use crate::plot::types::CastTargetType as C;
        assert_eq!(
            MonetDbDialect.sql_temporal_as_number("\"d\"", C::Date),
            "(EXTRACT(EPOCH FROM \"d\") / 86400)"
        );
    }

    #[test]
    fn quantile_is_native() {
        assert_eq!(
            MonetDbDialect.sql_quantile_inline("v", 0.75).as_deref(),
            Some("QUANTILE(\"v\", 0.75)")
        );
    }
}
