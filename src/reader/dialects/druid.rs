//! Apache Druid dialect.
//!
//! Druid SQL (Calcite-based) is restrictive: no correlated scalar subqueries
//! (so the default `sql_percentile` fallback will fail), no DATE/TIME types
//! (time is always a TIMESTAMP), no temp tables, and no spatial. For real
//! use, pair with the caching reader (`duckdb+...`) so internal
//! materialization happens off-Druid.

use crate::reader::SqlDialect;

/// Apache Druid dialect (requested in #341).
#[derive(Debug, Default, Clone, Copy)]
pub struct DruidDialect;

impl SqlDialect for DruidDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE")
    }

    fn date_type_name(&self) -> Option<&str> {
        // Druid has no DATE type; time columns are TIMESTAMP.
        None
    }

    fn time_type_name(&self) -> Option<&str> {
        None
    }

    fn boolean_type_name(&self) -> Option<&str> {
        None
    }

    fn supports_spatial(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_types_are_none() {
        assert_eq!(DruidDialect.date_type_name(), None);
        assert_eq!(DruidDialect.time_type_name(), None);
        assert_eq!(DruidDialect.boolean_type_name(), None);
        assert!(!DruidDialect.supports_spatial());
    }
}
