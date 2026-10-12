//! Apache Druid dialect.
//!
//! Druid SQL (Calcite-based) is restrictive: no correlated scalar subqueries
//! (so the default `sql_quantile` fallback will fail), no DATE/TIME types
//! (time is always a TIMESTAMP), no temp tables, and no spatial. With no
//! DDL at all, connections through Druid are always wrapped in a caching
//! reader (`requires_cache`), so internal materialization happens
//! off-Druid.

use crate::reader::SqlDialect;

/// Apache Druid dialect.
#[derive(Debug, Default, Clone, Copy)]
pub struct DruidDialect;

impl SqlDialect for DruidDialect {
    fn type_names(&self) -> crate::reader::TypeNames {
        crate::reader::TypeNames {
            number: Some("DOUBLE"),
            date: None,
            time: None,
            boolean: None,
            ..crate::reader::TypeNames::ANSI
        }
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        super::case_greatest(exprs)
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        super::case_least(exprs)
    }

    /// Druid has no DDL at all; connections are always wrapped in a caching
    /// reader rather than probed.
    fn requires_cache(&self) -> bool {
        true
    }

    /// The prerelease Foundry ADBC driver loses Druid's temporal types:
    /// timestamps surface as epoch-millisecond integers and dates as plain
    /// strings. The integers are indistinguishable from real LONGs, but an
    /// ISO-8601 string column can be recovered — sniff it back to a
    /// temporal type at the reader boundary.
    fn sniff_temporal_strings(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_types_are_none() {
        assert_eq!(DruidDialect.type_names().date, None);
        assert_eq!(DruidDialect.type_names().time, None);
        assert_eq!(DruidDialect.type_names().boolean, None);
        assert!(!DruidDialect.supports_spatial());
    }
}
