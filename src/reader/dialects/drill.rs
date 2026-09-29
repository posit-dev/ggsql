//! Apache Drill dialect.
//!
//! Drill is close to ANSI for types but uses backtick quoting and has no
//! recursive CTEs (so the default `sql_generate_series` will fail) and no
//! spatial support. As with Druid, the caching reader is the practical
//! execution mode.

use crate::reader::SqlDialect;

/// Apache Drill dialect (requested in #341).
#[derive(Debug, Default, Clone, Copy)]
pub struct DrillDialect;

impl SqlDialect for DrillDialect {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn supports_spatial(&self) -> bool {
        false
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
