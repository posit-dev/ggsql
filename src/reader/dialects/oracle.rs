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
        format!(
            "\"__ggsql_seq__\"(n) AS (\
               SELECT CAST(LEVEL - 1 AS BINARY_DOUBLE) AS n \
               FROM DUAL CONNECT BY LEVEL <= {n}\
             )"
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        Some(format!(
            "PERCENTILE_CONT({fraction}) WITHIN GROUP (ORDER BY {column})"
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
        let body = wrap_with_column_aliases(body_sql, column_aliases);
        vec![
            format!(
                "BEGIN EXECUTE IMMEDIATE 'DROP TABLE {}'; EXCEPTION WHEN OTHERS THEN NULL; END;",
                name.replace('\'', "''")
            ),
            format!("CREATE TABLE {} AS {}", qname, body),
        ]
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
    fn drop_is_plsql_guarded() {
        let stmts = OracleDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1 FROM dual");
        assert!(
            stmts[0].contains("EXECUTE IMMEDIATE 'DROP TABLE t'"),
            "got: {}",
            stmts[0]
        );
        assert_eq!(stmts[1], "CREATE TABLE \"t\" AS SELECT 1 FROM dual");
    }
}
