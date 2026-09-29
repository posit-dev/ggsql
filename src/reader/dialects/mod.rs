//! Backend-specific SQL dialects and dialect detection.
//!
//! One unit struct per backend implementing [`SqlDialect`], plus:
//!
//! - [`detect_dialect`]: pick a dialect from an ODBC DBMS name or driver
//!   string (substring matching, most specific patterns first).
//! - [`dialect_for_scheme`]: pick a dialect from a ggsql URI scheme
//!   (`postgres://`, `mysql://`, …), used by ADBC/ODBC reader dispatch.
//!
//! Both return boxed trait objects so readers can store them uniformly.

pub mod bigquery;
pub mod clickhouse;
pub mod databricks;
pub mod datafusion;
pub mod drill;
pub mod druid;
pub mod duckdb;
pub mod exasol;
pub mod monetdb;
pub mod mssql;
pub mod mysql;
pub mod oracle;
pub mod postgres;
pub mod redshift;
pub mod snowflake;
pub mod sqlite;
pub mod trino;

pub use bigquery::BigQueryDialect;
pub use clickhouse::ClickHouseDialect;
pub use databricks::DatabricksDialect;
pub use datafusion::DataFusionDialect;
pub use drill::DrillDialect;
pub use druid::DruidDialect;
pub use duckdb::DuckDbDialect;
pub use exasol::ExasolDialect;
pub use monetdb::MonetDbDialect;
pub use mssql::MssqlDialect;
pub use mysql::MySqlDialect;
pub use oracle::OracleDialect;
pub use postgres::PostgresDialect;
pub use redshift::RedshiftDialect;
pub use snowflake::SnowflakeDialect;
pub use sqlite::SqliteDialect;
pub use trino::TrinoDialect;

use crate::reader::{AnsiDialect, SqlDialect};

/// Detect the backend SQL dialect from a DBMS name and/or a driver hint
/// (ODBC driver name, ADBC driver name, or URI scheme).
///
/// The DBMS name is checked first; the driver hint is a fallback. Matching
/// is case-insensitive substring matching with the more specific patterns
/// ordered before generic ones (e.g. `microsoft sql server` before `sql`).
pub fn detect_dialect(
    dbms_name: Option<&str>,
    driver_hint: Option<&str>,
) -> Box<dyn SqlDialect + Send> {
    for text in [dbms_name, driver_hint].into_iter().flatten() {
        if let Some(d) = match_from_substring(&text.to_lowercase()) {
            return d;
        }
    }
    Box::new(AnsiDialect)
}

/// Pick a dialect from a ggsql URI scheme (`postgres`, `mysql`, …).
///
/// Returns `None` for unknown schemes so dispatch can report the scheme
/// itself as unsupported.
pub fn dialect_for_scheme(scheme: &str) -> Option<Box<dyn SqlDialect + Send>> {
    let d: Box<dyn SqlDialect + Send> = match scheme.to_ascii_lowercase().as_str() {
        "postgres" | "postgresql" => Box::new(PostgresDialect),
        "redshift" => Box::new(RedshiftDialect),
        "mysql" | "mariadb" => Box::new(MySqlDialect),
        "snowflake" => Box::new(SnowflakeDialect),
        "mssql" | "sqlserver" => Box::new(MssqlDialect),
        "bigquery" => Box::new(BigQueryDialect),
        "databricks" | "spark" => Box::new(DatabricksDialect),
        "clickhouse" => Box::new(ClickHouseDialect),
        "oracle" => Box::new(OracleDialect),
        "trino" => Box::new(TrinoDialect),
        "exasol" => Box::new(ExasolDialect),
        "monetdb" => Box::new(MonetDbDialect),
        "druid" => Box::new(DruidDialect),
        "drill" => Box::new(DrillDialect),
        "datafusion" => Box::new(DataFusionDialect),
        "duckdb" => Box::new(DuckDbDialect),
        "sqlite" => Box::new(SqliteDialect),
        _ => return None,
    };
    Some(d)
}

/// Substring matcher shared by both entry points. Returns `None` when
/// nothing matches so callers can fall through to the next hint, then ANSI.
fn match_from_substring(lower: &str) -> Option<Box<dyn SqlDialect + Send>> {
    // Most specific first. `microsoft sql server`/`msodbcsql` before anything
    // containing "sql"; `redshift` before `postgres`.
    if lower.contains("microsoft sql server")
        || lower.contains("msodbcsql")
        || lower.contains("sql server")
        || lower.contains("sqlserver")
        || lower.contains("mssql")
    {
        return Some(Box::new(MssqlDialect));
    }
    if lower.contains("redshift") {
        return Some(Box::new(RedshiftDialect));
    }
    if lower.contains("postgres") || lower.contains("psql") {
        return Some(Box::new(PostgresDialect));
    }
    if lower.contains("mariadb") || lower.contains("mysql") {
        return Some(Box::new(MySqlDialect));
    }
    if lower.contains("snowflake") {
        return Some(Box::new(SnowflakeDialect));
    }
    if lower.contains("bigquery") {
        return Some(Box::new(BigQueryDialect));
    }
    if lower.contains("databricks") || lower.contains("spark") {
        return Some(Box::new(DatabricksDialect));
    }
    if lower.contains("clickhouse") {
        return Some(Box::new(ClickHouseDialect));
    }
    if lower.contains("oracle") || lower.contains("ora") && lower.contains("driver") {
        return Some(Box::new(OracleDialect));
    }
    if lower.contains("trino") {
        return Some(Box::new(TrinoDialect));
    }
    if lower.contains("exasol") || lower.contains("exa") && lower.contains("odbc") {
        return Some(Box::new(ExasolDialect));
    }
    if lower.contains("monet") {
        return Some(Box::new(MonetDbDialect));
    }
    if lower.contains("druid") {
        return Some(Box::new(DruidDialect));
    }
    if lower.contains("drill") {
        return Some(Box::new(DrillDialect));
    }
    if lower.contains("datafusion") {
        return Some(Box::new(DataFusionDialect));
    }
    if lower.contains("duckdb") {
        return Some(Box::new(DuckDbDialect));
    }
    if lower.contains("sqlite") {
        return Some(Box::new(SqliteDialect));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_type_name(d: &dyn SqlDialect, expected: Option<&str>) {
        assert_eq!(d.number_type_name(), expected);
    }

    #[test]
    fn detects_from_dbms_name() {
        // Probe with number_type_name, which differs across these dialects.
        assert_type_name(
            &*detect_dialect(Some("PostgreSQL"), None),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(&*detect_dialect(Some("MySQL"), None), Some("DOUBLE"));
        assert_type_name(
            &*detect_dialect(Some("Microsoft SQL Server"), None),
            Some("FLOAT"),
        );
        assert_type_name(&*detect_dialect(Some("Snowflake"), None), Some("DOUBLE"));
        assert_type_name(
            &*detect_dialect(Some("Oracle"), None),
            Some("BINARY_DOUBLE"),
        );
        assert_type_name(
            &*detect_dialect(Some("ClickHouse"), None),
            Some("Nullable(Float64)"),
        );
        assert_type_name(
            &*detect_dialect(Some("Amazon Redshift"), None),
            Some("DOUBLE PRECISION"),
        );
    }

    #[test]
    fn falls_through_to_driver_hint() {
        assert_type_name(
            &*detect_dialect(Some("Unknown DBMS"), Some("PostgreSQL Unicode")),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(&*detect_dialect(None, Some("msodbcsql18")), Some("FLOAT"));
    }

    #[test]
    fn unknown_falls_back_to_ansi() {
        let d = detect_dialect(Some("mystery-db"), None);
        assert_type_name(&*d, Some("DOUBLE PRECISION"));
        assert_eq!(d.quote_ident("x"), "\"x\"");
    }

    #[test]
    fn mssql_wins_over_generic_sql() {
        // "SQL" appears in many names; make sure MSSQL patterns win.
        let d = detect_dialect(Some("Microsoft SQL Server"), None);
        assert_eq!(d.boolean_type_name(), Some("BIT"));
    }

    #[test]
    fn schemes_map_to_dialects() {
        for (scheme, probe) in [
            ("postgres", Some("DOUBLE PRECISION")),
            ("mysql", Some("DOUBLE")),
            ("mariadb", Some("DOUBLE")),
            ("snowflake", Some("DOUBLE")),
            ("mssql", Some("FLOAT")),
            ("bigquery", Some("FLOAT64")),
            ("databricks", Some("DOUBLE")),
            ("clickhouse", Some("Nullable(Float64)")),
            ("oracle", Some("BINARY_DOUBLE")),
        ] {
            let d = dialect_for_scheme(scheme).unwrap_or_else(|| panic!("scheme {scheme}"));
            assert_eq!(d.number_type_name(), probe, "scheme {scheme}");
        }
        assert!(dialect_for_scheme("nosuchdb").is_none());
    }

    #[test]
    fn per_dialect_quoting() {
        assert_eq!(dialect_for_scheme("mysql").unwrap().quote_ident("c"), "`c`");
        assert_eq!(
            dialect_for_scheme("bigquery").unwrap().quote_ident("c"),
            "`c`"
        );
        assert_eq!(
            dialect_for_scheme("postgres").unwrap().quote_ident("c"),
            "\"c\""
        );
    }
}
