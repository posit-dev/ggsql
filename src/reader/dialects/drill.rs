//! Apache Drill dialect.
//!
//! Drill is close to ANSI for types but uses backtick quoting and has no
//! recursive CTEs (so the default `sql_generate_series` will fail) and no
//! spatial support. With no DDL, connections through Drill are always
//! wrapped in a caching reader (`requires_cache`), which hosts the staged
//! tables.

use crate::reader::SqlDialect;

/// Apache Drill dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct DrillDialect;

impl SqlDialect for DrillDialect {
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

    /// Drill has no temp tables; connections are always wrapped in a caching
    /// reader rather than probed.
    fn requires_cache(&self) -> bool {
        true
    }

    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_use_backticks() {
        assert_eq!(DrillDialect.quote_ident("x"), "`x`");
    }
}
