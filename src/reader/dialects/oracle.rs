//! Oracle dialect.
//!
//! Main deviations from ANSI: Oracle type names (`BINARY_DOUBLE`,
//! `NUMBER(19)`, `VARCHAR2`), no `LIMIT` (uses a `ROWNUM` wrapper), no
//! pre-23c boolean type, `CONNECT BY` series, `PERCENTILE_CONT` quantiles,
//! and PL/SQL-guarded drops (Oracle has no `DROP TABLE IF EXISTS`).
//!
//! Spatial is disabled: Oracle Spatial uses the `SDO_*` API rather than the
//! OGC `ST_` function surface the ANSI defaults emit.

use crate::reader::{wrap_with_column_aliases, SqlDialect};

/// Oracle dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct OracleDialect;

impl SqlDialect for OracleDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("BINARY_DOUBLE")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("NUMBER(19)")
    }

    fn string_type_name(&self) -> Option<&str> {
        Some("VARCHAR2(4000)")
    }

    fn time_type_name(&self) -> Option<&str> {
        None
    }

    fn boolean_type_name(&self) -> Option<&str> {
        // No boolean SQL type before 23c; NUMBER(1) is the convention.
        Some("NUMBER(1)")
    }

    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }

    fn sql_limit(&self, query: &str, n: usize) -> String {
        format!("SELECT * FROM ({query}) WHERE ROWNUM <= {n}")
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
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        format!(
            "{__ggsql_seq__}(n) AS (\
               SELECT CAST(LEVEL - 1 AS BINARY_DOUBLE) AS n \
               FROM DUAL CONNECT BY LEVEL <= {n}\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "PERCENTILE_CONT({fraction}) WITHIN GROUP (ORDER BY {column})",
            column = self.quote_ident(column)
        ))
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        // Oracle DATE + integer adds days.
        format!("DATE '1970-01-01' + {days_since_epoch}")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let secs = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("TIMESTAMP '1970-01-01 00:00:00' + NUMTODSINTERVAL({secs}, 'SECOND')")
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        // Oracle has no DROP TABLE IF EXISTS; guard the drop with PL/SQL.
        let qname = self.quote_ident(name);
        let body =
            wrap_with_column_aliases(&|c: &str| self.quote_ident(c), body_sql, column_aliases);
        vec![
            // The drop must spell the name exactly like the CREATE below:
            // an unquoted name would be folded to uppercase by Oracle and
            // never match the quoted (case-preserved) table, silently
            // leaving stale tables behind under the WHEN OTHERS guard.
            format!(
                "BEGIN EXECUTE IMMEDIATE 'DROP TABLE {}'; EXCEPTION WHEN OTHERS THEN NULL; END;",
                qname.replace('\'', "''")
            ),
            format!("CREATE TABLE {} AS {}", qname, body),
        ]
    }

    fn sql_with_recursive(&self) -> &'static str {
        // Oracle has no RECURSIVE keyword; recursion is implied by
        // self-reference.
        "WITH"
    }

    fn sql_table_alias(&self, alias: &str) -> String {
        // Oracle rejects AS before table aliases (ORA-00933).
        self.quote_ident(alias)
    }

    fn sql_select_star(&self, table_alias: &str) -> String {
        // Oracle rejects an unqualified * alongside other select items.
        format!("{}.*", self.quote_ident(table_alias))
    }

    fn sql_create_empty_temp_table(&self, name: &str, column_defs: &[String]) -> Vec<String> {
        // Oracle temp tables are GLOBAL TEMPORARY with session-private rows.
        vec![format!(
            "CREATE GLOBAL TEMPORARY TABLE {} ({}) ON COMMIT PRESERVE ROWS",
            self.quote_ident(name),
            column_defs.join(", ")
        )]
    }

    fn supports_spatial(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_uses_rownum() {
        assert_eq!(
            OracleDialect.sql_limit("SELECT a FROM t", 5),
            "SELECT * FROM (SELECT a FROM t) WHERE ROWNUM <= 5"
        );
    }

    #[test]
    fn series_uses_connect_by() {
        let sql = OracleDialect.sql_generate_series(10);
        assert!(sql.contains("CONNECT BY LEVEL <= 10"), "got: {sql}");
    }

    #[test]
    fn with_recursive_is_plain_with() {
        assert_eq!(OracleDialect.sql_with_recursive(), "WITH");
    }

    #[test]
    fn drop_is_plsql_guarded() {
        let stmts = OracleDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1 FROM dual");
        assert!(
            stmts[0].contains("EXECUTE IMMEDIATE 'DROP TABLE \"t\"'"),
            "got: {}",
            stmts[0]
        );
        assert_eq!(stmts[1], "CREATE TABLE \"t\" AS SELECT 1 FROM dual");
    }

    #[test]
    fn drop_quotes_like_create() {
        // A name needing quoting must be quoted identically in DROP and
        // CREATE, or the drop never matches (Oracle folds unquoted names
        // to uppercase).
        let stmts =
            OracleDialect.create_or_replace_temp_table_sql("my Temp", &[], "SELECT 1 FROM dual");
        assert!(
            stmts[0].contains("DROP TABLE \"my Temp\""),
            "got: {}",
            stmts[0]
        );
        assert!(
            stmts[1].starts_with("CREATE TABLE \"my Temp\" AS"),
            "got: {}",
            stmts[1]
        );
    }
}
