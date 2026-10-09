//! Oracle dialect.
//!
//! Main deviations from ANSI: Oracle type names (`BINARY_DOUBLE`,
//! `NUMBER(19)`, `VARCHAR2`), no `LIMIT` (uses a `ROWNUM` wrapper), no
//! pre-23c boolean type, `CONNECT BY` series, `PERCENTILE_CONT` quantiles,
//! and PL/SQL-guarded drops (Oracle has no `DROP TABLE IF EXISTS`).
//!
//! Spatial is disabled: Oracle Spatial uses the `SDO_*` API rather than the
//! OGC `ST_` function surface the ANSI defaults emit.

use crate::reader::SqlDialect;

/// Oracle dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct OracleDialect;

impl SqlDialect for OracleDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("BINARY_DOUBLE"),
            integer: Some("NUMBER(19)"),
            time: None,
            string: Some("VARCHAR2(4000)"),
            boolean: Some("NUMBER(1)"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn sql_boolean_literal(&self, value: bool) -> String {
        super::one_zero_boolean_literal(value)
    }

    fn sql_limit(&self, query: &str, n: usize) -> String {
        crate::sql::Select::new(self)
            .select_star()
            .from_aliased(crate::sql::FromItem::Query(query), "__ggsql_lim__")
            .and_where(format!("ROWNUM <= {n}"))
            .build()
    }

    fn sql_limit_wraps_query(&self) -> bool {
        true
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

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        format!(
            "PERCENTILE_CONT({fraction}) WITHIN GROUP (ORDER BY {column})",
            column = self.quote_ident(column)
        )
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        // Oracle DATE + integer adds days.
        format!("DATE '1970-01-01' + {days_since_epoch}")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let secs = microseconds_since_epoch as f64 / 1_000_000.0;
        format!("TIMESTAMP '1970-01-01 00:00:00' + NUMTODSINTERVAL({secs}, 'SECOND')")
    }

    fn drop_table_sql(&self, name: &str) -> String {
        // Oracle has no DROP TABLE IF EXISTS. The drop must spell the name
        // exactly like the CREATE: an unquoted name would be folded to
        // uppercase and never match the quoted (case-preserved) table,
        // silently leaving stale tables behind under the WHEN OTHERS guard.
        let qname = self.quote_ident(name);
        format!(
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE {}'; EXCEPTION WHEN OTHERS THEN NULL; END;",
            qname.replace('\'', "''")
        )
    }

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        // Oracle has no DROP TABLE IF EXISTS; guard the drop with PL/SQL.
        crate::reader::TempTableStyle::GuardedDropThenCreate
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_uses_rownum() {
        assert_eq!(
            OracleDialect.sql_limit("SELECT a FROM t", 5),
            "SELECT * FROM (SELECT a FROM t) \"__ggsql_lim__\" WHERE ROWNUM <= 5"
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
