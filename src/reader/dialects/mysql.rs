//! MySQL / MariaDB SQL dialect.
//!
//! Main deviations from ANSI: backtick identifier quoting, no `DATE 'literal'`
//! syntax, `CREATE TEMPORARY TABLE`, and no nanosecond intervals.

use crate::reader::{wrap_with_column_aliases, SqlDialect};

/// MySQL dialect (also used for MariaDB).
#[derive(Debug, Default, Clone, Copy)]
pub struct MySqlDialect;

impl SqlDialect for MySqlDialect {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn string_type_name(&self) -> Option<&str> {
        // VARCHAR requires a length in MySQL; TEXT does not.
        Some("TEXT")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("DATETIME")
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

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        // MySQL has no `DATE '...'` literal syntax.
        format!("CAST('1970-01-01' + INTERVAL {days_since_epoch} DAY AS DATE)")
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!("'1970-01-01 00:00:00' + INTERVAL {microseconds_since_epoch} MICROSECOND")
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        // MySQL intervals top out at microseconds.
        let micros = nanoseconds_since_midnight / 1_000;
        format!("CAST('00:00:00' + INTERVAL {micros} MICROSECOND AS TIME)")
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = self.quote_ident(name);
        let body = wrap_with_column_aliases(body_sql, column_aliases);
        vec![
            format!("DROP TEMPORARY TABLE IF EXISTS {}", qname),
            format!("CREATE TEMPORARY TABLE {} AS {}", qname, body),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_use_backticks() {
        assert_eq!(MySqlDialect.quote_ident("col"), "`col`");
        assert_eq!(MySqlDialect.quote_ident("a`b"), "`a``b`");
    }

    #[test]
    fn date_literal_has_no_date_keyword_syntax() {
        let sql = MySqlDialect.sql_date_literal(7);
        assert_eq!(sql, "CAST('1970-01-01' + INTERVAL 7 DAY AS DATE)");
    }

    #[test]
    fn temp_table_uses_temporary() {
        let stmts = MySqlDialect.create_or_replace_temp_table_sql("t", &[], "SELECT 1");
        assert_eq!(stmts.len(), 2);
        assert_eq!(stmts[0], "DROP TEMPORARY TABLE IF EXISTS `t`");
        assert_eq!(stmts[1], "CREATE TEMPORARY TABLE `t` AS SELECT 1");
    }
}
