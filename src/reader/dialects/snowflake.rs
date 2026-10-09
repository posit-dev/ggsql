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
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            integer: Some("NUMBER"),
            datetime: Some("TIMESTAMP_NTZ"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn supports_spatial(&self) -> bool {
        true
    }

    fn sql_cast(&self, expr: &str, type_name: &str) -> String {
        format!("TRY_CAST({} AS {})", expr, type_name)
    }

    fn sql_generate_series(&self, n: usize) -> String {
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT SEQ4()::FLOAT AS n FROM TABLE(GENERATOR(ROWCOUNT => {n}))\
             )"
        )
    }

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        format!(
            "APPROX_PERCENTILE({column}, {fraction})",
            column = self.quote_ident(column)
        )
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
        let inner = crate::sql::Select::new(self)
            .select(format!("ST_ENVELOPE({column}) AS g"))
            .from(crate::sql::FromItem::Fragment(from))
            .build();
        crate::sql::Select::new(self)
            .select(
                "MIN(ST_XMIN(g)) AS xmin, MIN(ST_YMIN(g)) AS ymin, \
                 MAX(ST_XMAX(g)) AS xmax, MAX(ST_YMAX(g)) AS ymax",
            )
            .from_aliased(crate::sql::FromItem::Query(&inner), "__ggsql_env__")
            .build()
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
