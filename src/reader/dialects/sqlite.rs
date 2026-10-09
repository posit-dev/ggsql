//! SQLite dialect.
//!
//! SQLite is also reachable through ODBC and ADBC, so the dialect lives here
//! rather than behind the `sqlite` feature (which only gates the bundled
//! reader). Overrides cover SQLite's limited type system (TEXT for dates and
//! times, REAL for numbers, INTEGER for booleans), the SpatiaLite `ST_*`
//! surface, and portable-arithmetic stand-ins for the missing variance
//! aggregates.

use crate::reader::SqlDialect;

/// SQLite SQL dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct SqliteDialect;

impl SqlDialect for SqliteDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("REAL"),
            integer: Some("INTEGER"),
            date: Some("TEXT"),
            datetime: Some("TEXT"),
            time: Some("TEXT"),
            string: Some("TEXT"),
            boolean: Some("INTEGER"),
        }
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        super::case_greatest(exprs)
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        super::case_least(exprs)
    }

    fn supports_spatial(&self) -> bool {
        true
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("date('1970-01-01', '+{} days')", days_since_epoch)
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // Temporal values are ISO text here; julianday converts them.
        // julianday('1970-01-01') = 2440587.5.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("(JULIANDAY({expr}) - 2440587.5)"),
            C::DateTime => format!("((JULIANDAY({expr}) - 2440587.5) * 86400000000.0)"),
            _ => format!("CAST({expr} AS REAL)"),
        }
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let seconds = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("datetime('1970-01-01 00:00:00', '+{} seconds')", seconds)
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let seconds = nanoseconds_since_midnight as f64 / 1_000_000_000.0;
        format!("time('00:00:00', '+{} seconds')", seconds)
    }

    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }

    fn sql_spatial_setup(&self) -> Vec<String> {
        vec![
            "SELECT load_extension('mod_spatialite')".into(),
            "SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM sqlite_master WHERE name='spatial_ref_sys') \
             THEN InitSpatialMetaData(1) END"
                .into(),
        ]
    }

    fn sql_st_transform(&self, column: &str, source_crs: &str, target_crs: &str) -> String {
        let source_srid = crate::reader::extract_epsg_srid(source_crs);
        let target_srid = crate::reader::extract_epsg_srid(target_crs);
        match (source_srid, target_srid) {
            (Some(src), Some(tgt)) => {
                format!("ST_Transform(SetSRID({}, {}), {})", column, src, tgt)
            }
            _ => {
                let source_proj = source_crs.replace('\'', "''");
                let target_proj = target_crs.replace('\'', "''");
                let input = match source_srid {
                    Some(srid) => format!("SetSRID({}, {})", column, srid),
                    None => column.to_string(),
                };
                format!(
                    "ST_Transform({}, 0, NULL, '{}', '{}')",
                    input, source_proj, target_proj
                )
            }
        }
    }

    fn sql_make_envelope(&self, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> String {
        format!("BuildMbr({xmin}, {ymin}, {xmax}, {ymax})")
    }

    fn sql_ensure_geometry(&self, column: &str) -> String {
        format!("COALESCE(GeomFromWKB({column}, 4326), {column})")
    }

    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        format!(
            "SELECT MIN(MbrMinX({column})) AS xmin, MIN(MbrMinY({column})) AS ymin, \
                    MAX(MbrMaxX({column})) AS xmax, MAX(MbrMaxY({column})) AS ymax \
             FROM {from}"
        )
    }

    /// Stock SQLite has no `STDDEV_POP` / `VAR_POP`, so express variance,
    /// standard deviation, and standard error in portable arithmetic. Every
    /// other aggregate falls through to the shared default.
    fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
        // Population variance with a `MAX(0, …)` floor against tiny negative
        // floats from catastrophic cancellation. Both `MAX(a, b)` and `SQRT`
        // are scalar functions in modern bundled SQLite (math-functions build).
        let var_pop = || format!("MAX(0.0, AVG({c} * {c}) - AVG({c}) * AVG({c}))", c = qcol);
        let s = match name {
            "var" => var_pop(),
            "sdev" => format!("SQRT({})", var_pop()),
            "se" => format!("(SQRT({}) / SQRT(COUNT({c})))", var_pop(), c = qcol),
            _ => {
                return crate::reader::default_sql_aggregate(
                    &|c: &str| self.quote_ident(c),
                    name,
                    qcol,
                )
            }
        };
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_map_to_storage_classes() {
        let d = SqliteDialect;
        assert_eq!(d.number_type_name(), Some("REAL"));
        assert_eq!(d.date_type_name(), Some("TEXT"));
        assert_eq!(d.datetime_type_name(), Some("TEXT"));
        assert_eq!(d.boolean_type_name(), Some("INTEGER"));
    }

    #[test]
    fn greatest_least_are_case_expressions() {
        assert_eq!(
            SqliteDialect.sql_greatest(&["a", "b"]),
            "(CASE WHEN (a) >= (b) THEN (a) ELSE (b) END)"
        );
        assert_eq!(
            SqliteDialect.sql_least(&["a", "b", "c"]),
            "(CASE WHEN ((CASE WHEN (a) <= (b) THEN (a) ELSE (b) END)) <= (c) THEN ((CASE WHEN (a) <= (b) THEN (a) ELSE (b) END)) ELSE (c) END)"
        );
    }

    #[test]
    fn temporal_literals_are_sqlite_functions() {
        assert_eq!(
            SqliteDialect.sql_date_literal(1),
            "date('1970-01-01', '+1 days')"
        );
        assert_eq!(
            SqliteDialect.sql_datetime_literal(1_500_000),
            "datetime('1970-01-01 00:00:00', '+1.5 seconds')"
        );
        assert_eq!(
            SqliteDialect.sql_time_literal(2_000_000_000),
            "time('00:00:00', '+2 seconds')"
        );
        assert_eq!(SqliteDialect.sql_boolean_literal(true), "1");
        assert_eq!(SqliteDialect.sql_boolean_literal(false), "0");
    }

    #[test]
    fn temporal_as_number_uses_julianday() {
        use crate::plot::types::CastTargetType as C;
        assert_eq!(
            SqliteDialect.sql_temporal_as_number("\"d\"", C::Date),
            "(JULIANDAY(\"d\") - 2440587.5)"
        );
        assert_eq!(
            SqliteDialect.sql_temporal_as_number("\"d\"", C::DateTime),
            "((JULIANDAY(\"d\") - 2440587.5) * 86400000000.0)"
        );
    }

    #[test]
    fn variance_aggregates_are_portable_arithmetic() {
        assert_eq!(
            SqliteDialect.sql_aggregate("var", "\"v\""),
            Some("MAX(0.0, AVG(\"v\" * \"v\") - AVG(\"v\") * AVG(\"v\"))".to_string())
        );
        assert_eq!(
            SqliteDialect.sql_aggregate("mean", "\"v\""),
            Some("AVG(\"v\")".to_string())
        );
    }

    #[test]
    fn spatial_setup_loads_spatialite() {
        assert!(SqliteDialect.supports_spatial());
        let setup = SqliteDialect.sql_spatial_setup();
        assert_eq!(setup.len(), 2);
        assert!(setup[0].contains("mod_spatialite"));
        assert!(setup[1].contains("InitSpatialMetaData"));
    }

    #[test]
    fn st_transform_uses_srid_when_extractable() {
        assert_eq!(
            SqliteDialect.sql_st_transform("geom", "EPSG:4326", "EPSG:3857"),
            "ST_Transform(SetSRID(geom, 4326), 3857)"
        );
        assert_eq!(
            SqliteDialect.sql_make_envelope(0.0, 1.0, 2.0, 3.0),
            "BuildMbr(0, 1, 2, 3)"
        );
    }
}
