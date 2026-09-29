//! Snowflake SQL dialect.
//!
//! Main deviations from ANSI: `TRY_CAST`, `GENERATOR`-based series, no
//! `INTERVAL` arithmetic on date/timestamp literals (uses `DATEADD`),
//! approximate quantiles, and a per-row-envelope bounding-box aggregate.

use crate::reader::SqlDialect;

/// Snowflake dialect (native GEOMETRY/GEOGRAPHY for spatial).
#[derive(Debug, Default, Clone, Copy)]
pub struct SnowflakeDialect;

impl SqlDialect for SnowflakeDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("NUMBER")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("TIMESTAMP_NTZ")
    }

    fn sql_cast(&self, expr: &str, type_name: &str) -> String {
        format!("TRY_CAST({} AS {})", expr, type_name)
    }

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

    fn sql_generate_series(&self, n: usize) -> String {
        format!(
            "\"__ggsql_seq__\"(n) AS (\
               SELECT SEQ4()::FLOAT AS n FROM TABLE(GENERATOR(ROWCOUNT => {n}))\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("APPROX_PERCENTILE({column}, {fraction})"))
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("DATEADD(day, {days_since_epoch}, DATE '1970-01-01')")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!("DATEADD(microsecond, {microseconds_since_epoch}, TIMESTAMP '1970-01-01 00:00:00')")
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        format!("DATEADD(nanosecond, {nanoseconds_since_midnight}, TIME '00:00:00')")
    }

    fn sql_geometry_to_wkb(&self, column: &str) -> String {
        format!("ST_ASWKB({column})")
    }

    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        // Snowflake has no ST_Extent aggregate; aggregate per-row envelopes.
        format!(
            "SELECT MIN(ST_XMIN(g)) AS xmin, MIN(ST_YMIN(g)) AS ymin, \
                    MAX(ST_XMAX(g)) AS xmax, MAX(ST_YMAX(g)) AS ymax \
             FROM (SELECT ST_ENVELOPE({column}) AS g FROM {from})"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn casts_are_non_throwing() {
        assert_eq!(
            SnowflakeDialect.sql_cast("\"x\"", "DOUBLE"),
            "TRY_CAST(\"x\" AS DOUBLE)"
        );
    }

    #[test]
    fn series_uses_generator() {
        let sql = SnowflakeDialect.sql_generate_series(50);
        assert!(sql.contains("GENERATOR(ROWCOUNT => 50)"), "got: {sql}");
    }

    #[test]
    fn date_literal_uses_dateadd() {
        assert_eq!(
            SnowflakeDialect.sql_date_literal(3),
            "DATEADD(day, 3, DATE '1970-01-01')"
        );
    }
}
