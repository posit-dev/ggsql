/* Exasol-specific SQL dialect.
 *
 * Override fragments verified against `exasol/docker-db:2025.2.0-arm64dev.0`
 * by the sibling `probe-exasol-dialect-behaviors` plan (Runs 1, 2, 3).
 *
 * Known behavior caveats baked into this dialect:
 *
 * - TIMESTAMP precision is millisecond-only on Exasol. Sub-millisecond
 *   fractional input passed via `sql_datetime_literal` is silently truncated
 *   by the database (e.g. `...123456` rounds to `...123000`). This cannot be
 *   worked around at the dialect layer; document and accept.
 *
 * - TIME maps to `VARCHAR(32)`. Exasol has no SQL TIME type (`TIME '01:02:03'`
 *   raises *Feature not supported: SQL-Type TIME*). This mirrors the
 *   `SqliteDialect` precedent at `src/reader/sqlite.rs:45-47`. Polars-side
 *   reparse is required for Vega-Lite temporal axes.
 *
 * - A pre-existing `OdbcReader` Int32 buffer bug affects DECIMAL-returning
 *   function results (e.g. `GREATEST` / `LEAST` over decimals with precision
 *   < 10). The dialect's emitted SQL is correct; the bug is in the
 *   result-binding layer in `src/reader/odbc.rs`. Tracked in
 *   `specs/_plans/probe-exasol-dialect-behaviors/upstream-issue-draft.md`
 *   and will be filed upstream post-PR.
 */

pub struct ExasolDialect;

impl super::SqlDialect for ExasolDialect {
    fn string_type_name(&self) -> Option<&str> {
        Some("VARCHAR(2000000)")
    }

    fn time_type_name(&self) -> Option<&str> {
        Some("VARCHAR(32)")
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        format!("GREATEST({})", exprs.join(", "))
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        format!("LEAST({})", exprs.join(", "))
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("ADD_DAYS(DATE '1970-01-01', {})", days_since_epoch)
    }

    // Note: Exasol TIMESTAMP truncates to millisecond precision; sub-millisecond
    // fractional input is silently zeroed by the database.
    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let seconds_with_fraction = microseconds_since_epoch as f64 / 1_000_000.0;
        format!(
            "ADD_SECONDS(TIMESTAMP '1970-01-01 00:00:00', {})",
            seconds_with_fraction
        )
    }

    // Time stored as VARCHAR(32) per time_type_name. Emit ISO-8601 string.
    // Polars-side reparse is needed for Vega-Lite temporal axis (mirrors SqliteDialect).
    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let secs = nanoseconds_since_midnight / 1_000_000_000;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        let nanos = nanoseconds_since_midnight % 1_000_000_000;
        let micros = nanos / 1_000; // VARCHAR carries µs; nanosecond truncation acceptable
        format!("'{:02}:{:02}:{:02}.{:06}'", h, m, s, micros)
    }

    fn sql_list_catalogs(&self) -> String {
        // Exasol has no catalog layer above schemas; surface every schema as a top-level catalog
        "SELECT SCHEMA_NAME AS catalog_name FROM SYS.EXA_SCHEMAS ORDER BY SCHEMA_NAME".into()
    }

    fn sql_list_schemas(&self, _catalog: &str) -> String {
        // Catalog argument is ignored: Exasol treats schema as the top tier
        "SELECT SCHEMA_NAME AS schema_name FROM SYS.EXA_SCHEMAS ORDER BY SCHEMA_NAME".into()
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
    use crate::reader::SqlDialect;

    #[test]
    fn test_sql_greatest_uses_native() {
        let d = ExasolDialect;
        assert_eq!(d.sql_greatest(&["a", "b", "c"]), "GREATEST(a, b, c)");
    }

    #[test]
    fn test_sql_least_uses_native() {
        let d = ExasolDialect;
        assert_eq!(d.sql_least(&["a", "b", "c"]), "LEAST(a, b, c)");
    }

    #[test]
    fn test_sql_datetime_literal_uses_add_seconds() {
        let d = ExasolDialect;
        // 1_500_000 microseconds = 1.5 seconds since epoch
        let sql = d.sql_datetime_literal(1_500_000);
        assert!(
            sql.contains("ADD_SECONDS(TIMESTAMP '1970-01-01 00:00:00'"),
            "expected ADD_SECONDS form, got: {}",
            sql
        );
        assert!(
            sql.contains("1.5"),
            "expected fractional seconds 1.5 in: {}",
            sql
        );
        // Must not use the broken default INTERVAL N MICROSECOND form
        assert!(
            !sql.to_uppercase().contains("MICROSECOND"),
            "must not emit MICROSECOND interval (unsupported on Exasol): {}",
            sql
        );
    }

    #[test]
    fn test_sql_time_literal_emits_varchar_string() {
        let d = ExasolDialect;
        // 01:02:03.456789 → 3723 sec + 456_789_000 ns
        let ns = 3723 * 1_000_000_000_i64 + 456_789_000;
        let sql = d.sql_time_literal(ns);
        assert_eq!(sql, "'01:02:03.456789'");
        // Must not use the broken default TIME literal / NANOSECOND interval forms
        assert!(
            !sql.to_uppercase().contains("TIME "),
            "must not emit TIME literal: {}",
            sql
        );
        assert!(
            !sql.to_uppercase().contains("NANOSECOND"),
            "must not emit NANOSECOND interval: {}",
            sql
        );
    }

    #[test]
    fn test_sql_date_literal_uses_add_days() {
        let d = ExasolDialect;
        let sql = d.sql_date_literal(20000);
        assert_eq!(sql, "ADD_DAYS(DATE '1970-01-01', 20000)");
    }

    #[test]
    fn test_string_type_name_has_length() {
        let d = ExasolDialect;
        assert_eq!(d.string_type_name(), Some("VARCHAR(2000000)"));
    }

    #[test]
    fn test_time_type_name_is_varchar() {
        let d = ExasolDialect;
        assert_eq!(d.time_type_name(), Some("VARCHAR(32)"));
    }

    #[test]
    fn test_sql_list_catalogs_uses_sys_exa_schemas() {
        let d = ExasolDialect;
        let sql = d.sql_list_catalogs();
        assert!(
            sql.contains("SYS.EXA_SCHEMAS"),
            "expected SYS.EXA_SCHEMAS in: {}",
            sql
        );
        assert!(
            !sql.to_lowercase().contains("information_schema"),
            "must not query information_schema (absent on Exasol): {}",
            sql
        );
    }

    #[test]
    fn test_sql_list_schemas_uses_sys_exa_schemas() {
        let d = ExasolDialect;
        let sql = d.sql_list_schemas("ignored_catalog");
        assert!(
            sql.contains("SYS.EXA_SCHEMAS"),
            "expected SYS.EXA_SCHEMAS in: {}",
            sql
        );
        assert!(
            !sql.to_lowercase().contains("information_schema"),
            "must not query information_schema (absent on Exasol): {}",
            sql
        );
    }

    #[test]
    fn test_sql_list_tables_uses_sys_exa_all_tables() {
        let d = ExasolDialect;
        let sql = d.sql_list_tables("ignored_catalog", "MY_SCHEMA");
        assert!(
            sql.contains("SYS.EXA_ALL_TABLES"),
            "expected SYS.EXA_ALL_TABLES in: {}",
            sql
        );
        assert!(
            sql.contains("CASE WHEN TABLE_IS_VIRTUAL THEN 'VIEW' ELSE 'BASE TABLE' END"),
            "expected synthesized table_type CASE expression in: {}",
            sql
        );
        assert!(
            sql.contains("TABLE_SCHEMA = 'MY_SCHEMA'"),
            "expected schema filter in: {}",
            sql
        );

        // Schema-string escape: O'Brien → O''Brien
        let sql_escaped = d.sql_list_tables("ignored", "O'Brien");
        assert!(
            sql_escaped.contains("TABLE_SCHEMA = 'O''Brien'"),
            "expected single-quote-escaped schema in: {}",
            sql_escaped
        );
    }

    #[test]
    fn test_sql_list_columns_uses_sys_exa_all_columns() {
        let d = ExasolDialect;
        let sql = d.sql_list_columns("ignored_catalog", "MY_SCHEMA", "MY_TABLE");
        assert!(
            sql.contains("SYS.EXA_ALL_COLUMNS"),
            "expected SYS.EXA_ALL_COLUMNS in: {}",
            sql
        );
        assert!(
            sql.contains("ORDER BY COLUMN_ORDINAL_POSITION"),
            "expected ordinal-position ordering in: {}",
            sql
        );
        assert!(
            sql.contains("COLUMN_SCHEMA = 'MY_SCHEMA'"),
            "expected schema filter in: {}",
            sql
        );
        assert!(
            sql.contains("COLUMN_TABLE = 'MY_TABLE'"),
            "expected table filter in: {}",
            sql
        );

        // Schema-string and table-string escape: O'Brien → O''Brien
        let sql_escaped = d.sql_list_columns("ignored", "O'Brien", "T'bl");
        assert!(
            sql_escaped.contains("COLUMN_SCHEMA = 'O''Brien'"),
            "expected single-quote-escaped schema in: {}",
            sql_escaped
        );
        assert!(
            sql_escaped.contains("COLUMN_TABLE = 'T''bl'"),
            "expected single-quote-escaped table in: {}",
            sql_escaped
        );
    }
}
