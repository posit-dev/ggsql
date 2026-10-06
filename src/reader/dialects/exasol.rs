//! Exasol dialect.
//!
//! Exasol is close to ANSI (it implements the OGC `ST_` spatial surface
//! natively). Main deviations: `VARCHAR` requires a length, there is no SQL
//! `TIME` type (time values are stored as `VARCHAR(32)`, mirroring the
//! `SqliteDialect` precedent), date/timestamp literals use `ADD_DAYS` /
//! `ADD_SECONDS` rather than `INTERVAL`, and Exasol has no recursive CTEs —
//! the default `sql_generate_series` (recursive) will fail; small series
//! should instead come through the caching reader.
//!
//! Introspection goes through the `SYS.EXA_*` system tables: Exasol has no
//! `information_schema` views, and treats schemas as the top tier (so
//! `sql_list_catalogs` surfaces every schema as a catalog and
//! `sql_list_schemas` ignores its catalog argument).
//!
//! Known caveat: Exasol's `TIMESTAMP` truncates to millisecond precision;
//! sub-millisecond fractional input passed via `sql_datetime_literal` is
//! silently zeroed by the database and cannot be worked around at the
//! dialect layer.
//!
//! Override SQL verified against `exasol/docker-db:2025.2.0`.

use crate::reader::SqlDialect;

/// Exasol dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExasolDialect;

impl SqlDialect for ExasolDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            integer: Some("DECIMAL(19,0)"),
            time: Some("VARCHAR(32)"),
            string: Some("VARCHAR(2000000)"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("ADD_DAYS(DATE '1970-01-01', {days_since_epoch})")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        // TIMESTAMP truncates to millisecond precision (see module docs).
        let seconds_with_fraction = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("ADD_SECONDS(TIMESTAMP '1970-01-01 00:00:00', {seconds_with_fraction})")
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        // ISO-8601 string matching the VARCHAR(32) storage contract.
        let secs = nanoseconds_since_midnight / 1_000_000_000;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        let micros = (nanoseconds_since_midnight % 1_000_000_000) / 1_000;
        format!("'{h:02}:{m:02}:{s:02}.{micros:06}'")
    }

    fn sql_list_catalogs(&self) -> String {
        "SELECT SCHEMA_NAME AS catalog_name FROM SYS.EXA_SCHEMAS ORDER BY SCHEMA_NAME".to_string()
    }

    fn sql_list_schemas(&self, _catalog: &str) -> String {
        "SELECT SCHEMA_NAME AS schema_name FROM SYS.EXA_SCHEMAS ORDER BY SCHEMA_NAME".to_string()
    }

    fn sql_list_tables(&self, _catalog: &str, schema: &str) -> String {
        format!(
            "SELECT TABLE_NAME AS table_name, \
                CASE WHEN TABLE_IS_VIRTUAL THEN 'VIEW' ELSE 'BASE TABLE' END AS table_type \
             FROM SYS.EXA_ALL_TABLES \
             WHERE TABLE_SCHEMA = '{}' \
             ORDER BY TABLE_NAME",
            schema.replace('\'', "''")
        )
    }

    fn sql_list_columns(&self, _catalog: &str, schema: &str, table: &str) -> String {
        format!(
            "SELECT COLUMN_NAME AS column_name, COLUMN_TYPE AS data_type \
             FROM SYS.EXA_ALL_COLUMNS \
             WHERE COLUMN_SCHEMA = '{}' AND COLUMN_TABLE = '{}' \
             ORDER BY COLUMN_ORDINAL_POSITION",
            schema.replace('\'', "''"),
            table.replace('\'', "''")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varchar_has_length() {
        assert_eq!(ExasolDialect.string_type_name(), Some("VARCHAR(2000000)"));
    }

    #[test]
    fn date_and_datetime_literals_use_add_functions() {
        assert_eq!(
            ExasolDialect.sql_date_literal(30),
            "ADD_DAYS(DATE '1970-01-01', 30)"
        );
        let dt = ExasolDialect.sql_datetime_literal(1_500_000);
        assert!(
            dt.contains("ADD_SECONDS(TIMESTAMP '1970-01-01 00:00:00'"),
            "got: {dt}"
        );
        assert!(dt.contains("1.5"), "got: {dt}");
        assert!(!dt.to_uppercase().contains("MICROSECOND"), "got: {dt}");
    }

    #[test]
    fn time_is_stored_as_varchar() {
        assert_eq!(ExasolDialect.time_type_name(), Some("VARCHAR(32)"));
        let ns = 3723 * 1_000_000_000_i64 + 456_789_000;
        assert_eq!(ExasolDialect.sql_time_literal(ns), "'01:02:03.456789'");
    }

    #[test]
    fn greatest_least_are_native() {
        assert_eq!(ExasolDialect.sql_greatest(&["a", "b"]), "GREATEST(a, b)");
        assert_eq!(ExasolDialect.sql_least(&["a", "b"]), "LEAST(a, b)");
    }

    #[test]
    fn introspection_uses_sys_tables() {
        let d = ExasolDialect;
        let cats = d.sql_list_catalogs();
        assert!(cats.contains("SYS.EXA_SCHEMAS"), "got: {cats}");
        assert!(!cats.to_lowercase().contains("information_schema"));

        let tables = d.sql_list_tables("ignored", "O'Brien");
        assert!(tables.contains("SYS.EXA_ALL_TABLES"), "got: {tables}");
        assert!(
            tables.contains("TABLE_SCHEMA = 'O''Brien'"),
            "got: {tables}"
        );

        let cols = d.sql_list_columns("ignored", "S", "T'bl");
        assert!(cols.contains("SYS.EXA_ALL_COLUMNS"), "got: {cols}");
        assert!(cols.contains("COLUMN_TABLE = 'T''bl'"), "got: {cols}");
    }
}
