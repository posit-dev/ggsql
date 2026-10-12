//! MySQL / MariaDB SQL dialect.
//!
//! Main deviations from ANSI: backtick identifier quoting, no `DATE 'literal'`
//! syntax, `CREATE TEMPORARY TABLE`, and no nanosecond intervals.

use crate::reader::SqlDialect;

/// MySQL dialect (also used for MariaDB).
#[derive(Debug, Default, Clone, Copy)]
pub struct MySqlDialect;

impl SqlDialect for MySqlDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            datetime: Some("DATETIME"),
            string: Some("TEXT"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn quote_ident(&self, name: &str) -> String {
        super::backtick_quote_ident(name)
    }

    fn sql_null_safe_eq(&self, left: &str, right: &str) -> String {
        // MySQL/MariaDB lack IS NOT DISTINCT FROM; `<=>` is their null-safe
        // equality operator (true when both sides are NULL).
        format!("{left} <=> {right}")
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // MySQL CAST has no DOUBLE target before 8.0.17 and no
        // date -> number cast at all; TO_DAYS/TIMESTAMPDIFF are the
        // portable epoch conversions. TO_DAYS('1970-01-01') = 719528.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("(TO_DAYS({expr}) - 719528)"),
            C::DateTime => {
                format!("TIMESTAMPDIFF(MICROSECOND, '1970-01-01 00:00:00', {expr})")
            }
            _ => format!("CAST({expr} AS DOUBLE)"),
        }
    }

    fn sql_generate_series(&self, n: usize) -> String {
        // MariaDB's CAST has no REAL target; DOUBLE works on both MySQL
        // (8.0.17+) and MariaDB (10.4.5+).
        crate::reader::recursive_series_cte(self, n, "DOUBLE")
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

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::DropTemporaryThenCreateTemp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_safe_eq_uses_spaceship() {
        assert_eq!(
            MySqlDialect.sql_null_safe_eq("a.`g`", "b.`g`"),
            "a.`g` <=> b.`g`"
        );
    }

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
