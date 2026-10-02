//! DuckDB dialect.
//!
//! DuckDB is also reachable through ODBC and ADBC, so the dialect lives here
//! rather than behind the `duckdb` feature (which only gates the bundled
//! reader). Overrides use DuckDB-native functions (`GREATEST`, `LEAST`,
//! `GENERATE_SERIES`, `QUANTILE_CONT`) and the spatial extension's
//! `ST_*` surface.

use crate::reader::SqlDialect;

/// DuckDB SQL dialect with native function support.
#[derive(Debug, Default, Clone, Copy)]
pub struct DuckDbDialect;

impl SqlDialect for DuckDbDialect {
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

    fn sql_st_transform(&self, column: &str, source_crs: &str, target_crs: &str) -> String {
        format!(
            "ST_Transform({}, '{}', '{}', always_xy := true)",
            column,
            source_crs.replace('\'', "''"),
            target_crs.replace('\'', "''")
        )
    }

    /// WORKAROUND(duckdb-rs#714): geometry columns arrive as WKB BLOB via Arrow.
    fn sql_ensure_geometry(&self, column: &str) -> String {
        format!("ST_GeomFromWKB(CAST({column} AS BLOB))")
    }

    fn sql_select_replace(
        &self,
        expr: &str,
        col: &str,
        from: &str,
        _all_columns: &[String],
    ) -> String {
        format!("SELECT * REPLACE ({expr} AS {col}) FROM ({from})")
    }

    fn sql_geometry_to_wkb(&self, column: &str) -> String {
        format!("ST_AsWKB({column})")
    }

    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        format!(
            "SELECT ST_XMin(ext) AS xmin, ST_YMin(ext) AS ymin, \
                    ST_XMax(ext) AS xmax, ST_YMax(ext) AS ymax \
             FROM (SELECT ST_Extent_Agg({column}) AS ext FROM {from})"
        )
    }

    fn sql_spatial_setup(&self) -> Vec<String> {
        vec!["LOAD spatial".into()]
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let body = crate::reader::wrap_with_column_aliases(
            &|c: &str| self.quote_ident(c),
            body_sql,
            column_aliases,
        );
        vec![format!(
            "CREATE OR REPLACE TEMP TABLE {} AS {}",
            self.quote_ident(name),
            body
        )]
    }

    fn sql_generate_series(&self, n: usize) -> String {
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        // The subtraction happens in SQL: computing `n - 1` in Rust would
        // underflow for n = 0, while GENERATE_SERIES(0, -1) correctly
        // yields no rows.
        format!("{__ggsql_seq__}(n) AS (SELECT generate_series FROM GENERATE_SERIES(0, {n} - 1))")
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "QUANTILE_CONT({}, {})",
            self.quote_ident(column),
            fraction
        ))
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // DuckDB rejects temporal -> numeric casts; date subtraction
        // yields integer days, and EPOCH_US covers datetimes.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("({expr} - DATE '1970-01-01')"),
            C::DateTime => format!("EPOCH_US({expr})"),
            _ => {
                let ty = self.number_type_name().unwrap_or("DOUBLE");
                self.sql_cast(expr, ty)
            }
        }
    }

    fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
        match name {
            "first" => Some(format!("FIRST({})", qcol)),
            "last" => Some(format!("LAST({})", qcol)),
            "diff" => Some(format!("(LAST({c}) - FIRST({c}))", c = qcol)),
            _ => crate::reader::default_sql_aggregate(&|c: &str| self.quote_ident(c), name, qcol),
        }
    }

    fn sql_percentile(&self, column: &str, fraction: f64, from: &str, groups: &[String]) -> String {
        let __ggsql_pct__ = self.quote_ident("__ggsql_pct__");
        let __ggsql_qt__ = self.quote_ident("__ggsql_qt__");
        let group_filter = groups
            .iter()
            .map(|g| {
                let q = self.quote_ident(g);
                let cond = self.sql_null_safe_eq(
                    &format!("{__ggsql_pct__}.{q}"),
                    &format!("{__ggsql_qt__}.{q}"),
                );
                format!("AND {cond}")
            })
            .collect::<Vec<_>>()
            .join(" ");

        let quoted_column = self.quote_ident(column);
        format!(
            "(SELECT QUANTILE_CONT({column}, {fraction}) \
            FROM ({from}) AS {__ggsql_pct__} \
            WHERE {column} IS NOT NULL {group_filter})",
            column = quoted_column
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_handles_zero() {
        // Regression: `n - 1` computed in Rust underflowed usize for n = 0.
        let sql = DuckDbDialect.sql_generate_series(0);
        assert!(sql.contains("GENERATE_SERIES(0, 0 - 1)"), "got: {sql}");
    }

    #[test]
    fn series_sql_shape() {
        assert_eq!(
            DuckDbDialect.sql_generate_series(5),
            "\"__ggsql_seq__\"(n) AS (SELECT generate_series FROM GENERATE_SERIES(0, 5 - 1))"
        );
    }
}
