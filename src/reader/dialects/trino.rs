//! Trino dialect.
//!
//! Main deviations from ANSI: `TRY_CAST`, `UNNEST(SEQUENCE(...))` series,
//! `approx_percentile`, quoted interval literals, and per-row-envelope
//! bounding boxes (Trino has no `ST_Extent`).
//!
//! Temp tables: most Trino connectors don't support `CREATE TEMP TABLE`;
//! for real workloads pair the Trino reader with the caching reader
//! (e.g. `duckdb+trino://...`) so internal tables land in the cache.

use crate::reader::SqlDialect;

/// Trino dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct TrinoDialect;

impl SqlDialect for TrinoDialect {
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
               SELECT CAST(t AS DOUBLE) AS n \
               FROM UNNEST(SEQUENCE(0, {n} - 1)) AS u(t)\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!("approx_percentile({column}, {fraction})"))
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        // Trino interval literals require a quoted value.
        format!("DATE '1970-01-01' + INTERVAL '{days_since_epoch}' DAY")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        // from_unixtime accepts fractional seconds.
        let secs = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("FROM_UNIXTIME({secs})")
    }

    /// Most Trino connectors don't support `CREATE TEMP TABLE`; connections
    /// are always wrapped in a caching reader rather than probed.
    fn requires_cache(&self) -> bool {
        true
    }

    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        format!(
            "SELECT MIN(ST_XMin(g)) AS xmin, MIN(ST_YMin(g)) AS ymin, \
                    MAX(ST_XMax(g)) AS xmax, MAX(ST_YMax(g)) AS ymax \
             FROM (SELECT ST_Envelope({column}) AS g FROM {from})"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_uses_unnest_sequence() {
        let sql = TrinoDialect.sql_generate_series(12);
        assert!(sql.contains("UNNEST(SEQUENCE(0, 12 - 1))"), "got: {sql}");
    }

    #[test]
    fn interval_is_quoted() {
        assert_eq!(
            TrinoDialect.sql_date_literal(2),
            "DATE '1970-01-01' + INTERVAL '2' DAY"
        );
    }

    #[test]
    fn bbox_uses_per_row_envelope() {
        let sql = TrinoDialect.sql_geometry_bbox("geom", "t");
        assert!(sql.contains("ST_Envelope"), "got: {sql}");
        assert!(!sql.contains("ST_Extent"), "got: {sql}");
    }
}
