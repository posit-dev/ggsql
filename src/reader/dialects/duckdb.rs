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
    fn supports_spatial(&self) -> bool {
        true
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
        crate::sql::Select::new(self)
            .select(format!("* REPLACE ({expr} AS {col})"))
            .from_aliased(crate::sql::FromItem::Query(from), "__ggsql_sr__")
            .build()
    }

    fn sql_geometry_to_wkb(&self, column: &str) -> String {
        format!("ST_AsWKB({column})")
    }

    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        let inner = crate::sql::Select::new(self)
            .select(format!("ST_Extent_Agg({column}) AS ext"))
            .from(crate::sql::FromItem::Fragment(from))
            .build();
        crate::sql::Select::new(self)
            .select(
                "ST_XMin(ext) AS xmin, ST_YMin(ext) AS ymin, \
                 ST_XMax(ext) AS xmax, ST_YMax(ext) AS ymax",
            )
            .from_aliased(crate::sql::FromItem::Query(&inner), "__ggsql_ext__")
            .build()
    }

    fn sql_spatial_setup(&self) -> Vec<String> {
        vec!["LOAD spatial".into()]
    }

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::CreateOrReplaceTemp
    }

    fn sql_generate_series(&self, n: usize) -> String {
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        // The subtraction happens in SQL: computing `n - 1` in Rust would
        // underflow for n = 0, while GENERATE_SERIES(0, -1) correctly
        // yields no rows.
        format!("{__ggsql_seq__}(n) AS (SELECT generate_series FROM GENERATE_SERIES(0, {n} - 1))")
    }

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        format!("QUANTILE_CONT({}, {})", self.quote_ident(column), fraction)
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

    #[test]
    fn quantile_is_native_and_quotes_raw_names() {
        // stat_aggregate passes the raw column name; the dialect must quote.
        assert_eq!(
            DuckDbDialect.sql_quantile("mixed Case", 0.25, crate::sql::FromItem::Table("src"), &[]),
            "QUANTILE_CONT(\"mixed Case\", 0.25)"
        );
    }

    #[test]
    fn temporal_as_number_uses_subtraction_and_epoch() {
        use crate::plot::types::CastTargetType as C;
        assert_eq!(
            DuckDbDialect.sql_temporal_as_number("\"d\"", C::Date),
            "(\"d\" - DATE '1970-01-01')"
        );
        assert_eq!(
            DuckDbDialect.sql_temporal_as_number("\"d\"", C::DateTime),
            "EPOCH_US(\"d\")"
        );
    }

    #[test]
    fn temp_table_is_create_or_replace() {
        let stmts = DuckDbDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1");
        assert_eq!(
            stmts,
            vec!["CREATE OR REPLACE TEMP TABLE \"t\" AS SELECT 1".to_string()]
        );
    }

    #[test]
    fn spatial_setup_loads_extension() {
        assert!(DuckDbDialect.supports_spatial());
        assert_eq!(DuckDbDialect.sql_spatial_setup(), vec!["LOAD spatial"]);
    }

    #[test]
    fn st_transform_pins_always_xy_and_escapes_quotes() {
        assert_eq!(
            DuckDbDialect.sql_st_transform("geom", "EPSG:4326", "+proj=merc +lon_0=0"),
            "ST_Transform(geom, 'EPSG:4326', '+proj=merc +lon_0=0', always_xy := true)"
        );
        assert_eq!(
            DuckDbDialect.sql_st_transform("geom", "x'y", "z"),
            "ST_Transform(geom, 'x''y', 'z', always_xy := true)"
        );
    }

    #[test]
    fn geometry_arrives_as_wkb_blob() {
        // WORKAROUND(duckdb-rs#714): geometry columns arrive as WKB BLOB.
        assert_eq!(
            DuckDbDialect.sql_ensure_geometry("g"),
            "ST_GeomFromWKB(CAST(g AS BLOB))"
        );
    }

    #[test]
    fn first_last_diff_aggregates() {
        assert_eq!(
            DuckDbDialect.sql_aggregate("first", "\"v\""),
            Some("FIRST(\"v\")".to_string())
        );
        assert_eq!(
            DuckDbDialect.sql_aggregate("diff", "\"v\""),
            Some("(LAST(\"v\") - FIRST(\"v\"))".to_string())
        );
        assert_eq!(
            DuckDbDialect.sql_aggregate("mean", "\"v\""),
            Some("AVG(\"v\")".to_string())
        );
    }
}
