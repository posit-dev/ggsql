//! Backend-specific SQL dialects.
//!
//! One unit struct per backend implementing [`SqlDialect`. Dialect lookup
//! — by URI scheme, by `dialect=` override, or by ODBC DBMS name/driver
//! detection — lives in the [`registry`](crate::reader::registry).

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

use crate::reader::SqlDialect;

/// Generic ANSI SQL dialect: the fallback for backends ggsql doesn't
/// recognise, selected explicitly with `dialect=ansi`. Every trait method
/// keeps its portable default.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnsiDialect;

impl SqlDialect for AnsiDialect {}

/// Backtick identifier quoting (MySQL/MariaDB, ClickHouse, BigQuery,
/// Databricks, Drill); embedded backticks are doubled.
pub(crate) fn backtick_quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// `1`/`0` boolean literal for backends without a real BOOLEAN literal
/// (SQL Server, Oracle, SQLite).
pub(crate) fn one_zero_boolean_literal(value: bool) -> String {
    if value { "1" } else { "0" }.to_string()
}

/// Postgres-style epoch conversion for backends that reject temporal →
/// numeric casts: date subtraction yields integer days, EXTRACT EPOCH
/// yields seconds scaled to microseconds. Shared by the Postgres and
/// Redshift dialects.
pub(crate) fn epoch_via_subtract_extract<D: SqlDialect + ?Sized>(
    dialect: &D,
    expr: &str,
    kind: crate::plot::types::CastTargetType,
) -> String {
    use crate::plot::types::CastTargetType as C;
    match kind {
        C::Date => format!("({expr} - DATE '1970-01-01')"),
        C::DateTime => format!("(EXTRACT(EPOCH FROM {expr}) * 1000000)"),
        _ => {
            let ty = dialect.number_type_name().unwrap_or("DOUBLE PRECISION");
            dialect.sql_cast(expr, ty)
        }
    }
}

/// Epoch conversion for backends where EXTRACT EPOCH yields seconds for
/// both dates and timestamps (DataFusion, MonetDB): dates scale to days,
/// datetimes to microseconds; anything else falls through to a numeric
/// cast in the dialect's number type.
pub(crate) fn epoch_via_extract_seconds<D: SqlDialect + ?Sized>(
    dialect: &D,
    expr: &str,
    kind: crate::plot::types::CastTargetType,
) -> String {
    use crate::plot::types::CastTargetType as C;
    match kind {
        C::Date => format!("(EXTRACT(EPOCH FROM {expr}) / 86400)"),
        C::DateTime => format!("(EXTRACT(EPOCH FROM {expr}) * 1000000)"),
        _ => {
            let ty = dialect.number_type_name().unwrap_or("DOUBLE");
            dialect.sql_cast(expr, ty)
        }
    }
}

/// CASE-based scalar greatest/least for backends without `GREATEST`/`LEAST`
/// (SQL Server, SQLite, Druid, Drill, MonetDB). Builds a left-folded chain of
/// two-way comparisons.
pub(crate) fn case_greatest(exprs: &[&str]) -> String {
    let Some((first, rest)) = exprs.split_first() else {
        debug_assert!(false, "case_greatest called with no expressions");
        return String::new();
    };
    let mut result = first.to_string();
    for expr in rest {
        result = format!("(CASE WHEN ({result}) >= ({expr}) THEN ({result}) ELSE ({expr}) END)");
    }
    result
}

/// See [`case_greatest`].
pub(crate) fn case_least(exprs: &[&str]) -> String {
    let Some((first, rest)) = exprs.split_first() else {
        debug_assert!(false, "case_least called with no expressions");
        return String::new();
    };
    let mut result = first.to_string();
    for expr in rest {
        result = format!("(CASE WHEN ({result}) <= ({expr}) THEN ({result}) ELSE ({expr}) END)");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::registry::{by_scheme, detect_or_err as detect_dialect};

    fn dialect_for_scheme(scheme: &str) -> Option<crate::reader::DialectRef> {
        by_scheme(scheme).map(|e| e.dialect())
    }

    fn assert_type_name(d: &dyn SqlDialect, expected: Option<&str>) {
        assert_eq!(d.number_type_name(), expected);
    }

    #[test]
    fn detects_from_dbms_name() {
        assert_type_name(
            &*detect_dialect(Some("PostgreSQL"), None).unwrap(),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(
            &*detect_dialect(Some("MySQL"), None).unwrap(),
            Some("DOUBLE"),
        );
        assert_type_name(
            &*detect_dialect(Some("Microsoft SQL Server"), None).unwrap(),
            Some("FLOAT"),
        );
        assert_type_name(
            &*detect_dialect(Some("Snowflake"), None).unwrap(),
            Some("DOUBLE"),
        );
        assert_type_name(
            &*detect_dialect(Some("Oracle"), None).unwrap(),
            Some("BINARY_DOUBLE"),
        );
        assert_type_name(
            &*detect_dialect(Some("ClickHouse"), None).unwrap(),
            Some("Nullable(Float64)"),
        );
        assert_type_name(
            &*detect_dialect(Some("Amazon Redshift"), None).unwrap(),
            Some("DOUBLE PRECISION"),
        );
    }

    #[test]
    fn falls_through_to_driver_hint() {
        assert_type_name(
            &*detect_dialect(Some("Unknown DBMS"), Some("PostgreSQL Unicode")).unwrap(),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(
            &*detect_dialect(None, Some("msodbcsql18")).unwrap(),
            Some("FLOAT"),
        );
    }

    #[test]
    fn unknown_backend_errors_with_escape_hatch_hint() {
        let err = detect_dialect(Some("mystery-db"), None)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("mystery-db"), "got: {err}");
        assert!(err.contains("dialect=ansi"), "got: {err}");
    }

    #[test]
    fn mssql_wins_over_generic_sql() {
        let d = detect_dialect(Some("Microsoft SQL Server"), None).unwrap();
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

    /// Contract for the quantile hook: callers pass the raw (unquoted)
    /// column name and the dialect quotes it. A name needing quoting must
    /// appear quoted — interpolating it raw breaks on any real column whose
    /// name is not a bare lowercase identifier.
    #[test]
    fn quantile_hook_quotes_raw_column_names() {
        // Sweeps the registry directly so new backends are covered
        // automatically; canonical schemes and aliases alike.
        for entry in crate::reader::registry::REGISTRY {
            for scheme in entry.schemes() {
                let d = dialect_for_scheme(scheme).unwrap();
                let quoted = d.quote_ident("mixed Case");
                let sql = d.sql_quantile("mixed Case", 0.5, crate::sql::FromItem::Table("t"), &[]);
                assert!(
                    sql.contains(&quoted),
                    "{scheme}: sql_quantile does not quote its column: {sql}"
                );
            }
        }
    }

    /// The ANSI defaults render temporal literals in ISO form; arithmetic
    /// on `INTERVAL n DAY/MICROSECOND/NANOSECOND` is not portable (and not
    /// valid ANSI), so backends without an override get real literals.
    #[test]
    fn ansi_temporal_literals_are_iso() {
        let d = crate::reader::AnsiDialect;
        assert_eq!(d.sql_date_literal(18993), "DATE '2022-01-01'");
        assert_eq!(d.sql_date_literal(0), "DATE '1970-01-01'");
        assert_eq!(d.sql_datetime_literal(0), "TIMESTAMP '1970-01-01 00:00:00'");
        assert_eq!(
            d.sql_datetime_literal(1_500_000),
            "TIMESTAMP '1970-01-01 00:00:01.500000'"
        );
        assert_eq!(d.sql_time_literal(0), "TIME '00:00:00'");
        assert_eq!(
            d.sql_time_literal(3_723_000_000_001),
            "TIME '01:02:03.000000001'"
        );
    }
}
