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
        // DataFusion's generate_series is a table function.
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT CAST(generate_series AS DOUBLE) AS n \
               FROM generate_series(0, {n} - 1)\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("approx_percentile_cont({column}, {fraction})"))
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
