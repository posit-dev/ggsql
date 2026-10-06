//! MonetDB dialect.
//!
//! MonetDB is strongly SQL-standard (like its CWI sibling DuckDB), so the
//! ANSI defaults mostly apply. Overrides: native `quantile` aggregate and
//! DOUBLE type naming. MonetDB's optional `geom` module exposes a
//! PostGIS-like `ST_` surface, but spatial is not enabled here —
//! `supports_spatial()` keeps the `false` default.

use crate::reader::SqlDialect;

/// MonetDB dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct MonetDbDialect;

impl SqlDialect for MonetDbDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        super::case_greatest(exprs)
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        super::case_least(exprs)
    }

    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        _from: crate::sql::FromItem<'_>,
        _groups: &[String],
    ) -> String {
        format!(
            "QUANTILE({column}, {fraction})",
            column = self.quote_ident(column)
        )
    }

    fn sql_create_empty_temp_table(&self, name: &str, column_defs: &[String]) -> Vec<String> {
        // MonetDB temp tables default to ON COMMIT DELETE ROWS; with ODBC
        // autocommit the register INSERT's commit would wipe the staged rows.
        vec![format!(
            "CREATE TEMPORARY TABLE {} ({}) ON COMMIT PRESERVE ROWS",
            self.quote_ident(name),
            column_defs.join(", ")
        )]
    }

    fn temp_table_style(&self) -> crate::reader::TempTableStyle {
        crate::reader::TempTableStyle::DropThenCreateTempPreserveRows
    }

    fn sql_temporal_as_number(
        &self,
        expr: &str,
        kind: crate::plot::types::CastTargetType,
    ) -> String {
        // MonetDB has no direct temporal -> double cast ("types date and
        // double are not equal"); EXTRACT EPOCH yields seconds.
        use crate::plot::types::CastTargetType as C;
        match kind {
            C::Date => format!("(EXTRACT(EPOCH FROM {expr}) / 86400)"),
            C::DateTime => format!("(EXTRACT(EPOCH FROM {expr}) * 1000000)"),
            _ => format!("CAST({expr} AS DOUBLE)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_table_preserves_rows_on_commit() {
        let ddl = MonetDbDialect.sql_create_empty_temp_table("t", &["\"a\" INT".to_string()]);
        assert_eq!(ddl.len(), 1);
        assert!(
            ddl[0].contains("ON COMMIT PRESERVE ROWS"),
            "got: {}",
            ddl[0]
        );
    }

    #[test]
    fn temporal_as_number_uses_epoch_extract() {
        use crate::plot::types::CastTargetType as C;
        assert_eq!(
            MonetDbDialect.sql_temporal_as_number("\"d\"", C::Date),
            "(EXTRACT(EPOCH FROM \"d\") / 86400)"
        );
    }

    #[test]
    fn quantile_is_native() {
        assert_eq!(
            MonetDbDialect.sql_quantile("v", 0.75, crate::sql::FromItem::Table("t"), &[]),
            "QUANTILE(\"v\", 0.75)"
        );
    }
}
