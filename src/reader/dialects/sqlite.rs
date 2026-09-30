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
    fn string_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("REAL")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("INTEGER")
    }

    fn boolean_type_name(&self) -> Option<&str> {
        Some("INTEGER")
    }

    fn date_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn time_type_name(&self) -> Option<&str> {
        Some("TEXT")
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("date('1970-01-01', '+{} days')", days_since_epoch)
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
