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

#[cfg(test)]
use crate::reader::SqlDialect;

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

/// Split a query into its leading `WITH` clause and the remaining main
/// query. Returns `None` when the query does not start with `WITH`.
///
/// Used by dialects that forbid CTEs inside derived tables (SQL Server) to
/// hoist the CTE definitions out of a subquery wrap. The scanner is aware
/// of string literals and quoted identifiers, and handles optional CTE
/// column lists (`cte(c1, c2) AS (...)`) and `WITH RECURSIVE`.
pub fn split_cte_prefix(query: &str) -> Option<(&str, &str)> {
    let s = query.trim_start();
    let bytes = s.as_bytes();
    if bytes.len() < 5 || !s[..4].eq_ignore_ascii_case("with") || !bytes[4].is_ascii_whitespace() {
        return None;
    }
    let mut i = 4;
    skip_ws(bytes, &mut i);
    if s.len() - i >= 9 && s[i..i + 9].eq_ignore_ascii_case("recursive") {
        i += 9;
    }
    loop {
        skip_ws(bytes, &mut i);
        if i >= bytes.len() {
            return None;
        }
        let name_start = i;
        match bytes[i] {
            q @ (b'"' | b'`') => skip_quoted(bytes, &mut i, q),
            _ => {
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'$'))
                {
                    i += 1;
                }
            }
        }
        if i == name_start {
            return None;
        }
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b'(' {
            skip_balanced_parens(bytes, &mut i)?;
            skip_ws(bytes, &mut i);
        }
        if s.len() - i < 2 || !s[i..i + 2].eq_ignore_ascii_case("as") {
            return None;
        }
        i += 2;
        skip_ws(bytes, &mut i);
        if i >= bytes.len() || bytes[i] != b'(' {
            return None;
        }
        skip_balanced_parens(bytes, &mut i)?;
        let cte_end = i;
        let mut j = i;
        skip_ws(bytes, &mut j);
        if j < bytes.len() && bytes[j] == b',' {
            i = j + 1;
            continue;
        }
        return Some((&s[..cte_end], s[j..].trim_start()));
    }
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

/// Skip past a quoted region; `*i` is at the opening quote. A doubled quote
/// is treated as an escape (SQL string/identifier convention).
fn skip_quoted(bytes: &[u8], i: &mut usize, quote: u8) {
    *i += 1;
    while *i < bytes.len() {
        if bytes[*i] == quote {
            if *i + 1 < bytes.len() && bytes[*i + 1] == quote {
                *i += 2;
                continue;
            }
            *i += 1;
            return;
        }
        *i += 1;
    }
}

/// Skip a balanced parenthesised region; `*i` is at the opening `(`.
/// Returns `None` when the parens never balance.
fn skip_balanced_parens(bytes: &[u8], i: &mut usize) -> Option<()> {
    let mut depth = 0usize;
    while *i < bytes.len() {
        match bytes[*i] {
            b'(' => {
                depth += 1;
                *i += 1;
            }
            b')' => {
                depth -= 1;
                *i += 1;
                if depth == 0 {
                    return Some(());
                }
            }
            q @ (b'\'' | b'"' | b'`') => skip_quoted(bytes, i, q),
            _ => *i += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::registry::{by_scheme, detect_or_err as detect_dialect};

    fn dialect_for_scheme(scheme: &str) -> Option<Box<dyn SqlDialect + Send>> {
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
    fn splits_cte_prefix() {
        let (cte, body) = split_cte_prefix(
            "WITH a AS (SELECT 1 AS x), b(n) AS (SELECT 2) SELECT * FROM a JOIN b ON a.x = b.n",
        )
        .unwrap();
        assert_eq!(cte, "WITH a AS (SELECT 1 AS x), b(n) AS (SELECT 2)");
        assert_eq!(body, "SELECT * FROM a JOIN b ON a.x = b.n");
    }

    #[test]
    fn splits_cte_with_parens_and_strings() {
        let (cte, body) = split_cte_prefix(
            "WITH RECURSIVE \"__ggsql_t__\" AS (SELECT '(' AS s, f(1, (2)) AS v) SELECT v FROM \"__ggsql_t__\"",
        )
        .unwrap();
        assert_eq!(
            cte,
            "WITH RECURSIVE \"__ggsql_t__\" AS (SELECT '(' AS s, f(1, (2)) AS v)"
        );
        assert_eq!(body, "SELECT v FROM \"__ggsql_t__\"");
    }

    #[test]
    fn no_cte_returns_none() {
        assert!(split_cte_prefix("SELECT 1").is_none());
        assert!(split_cte_prefix("WITHHELD AS x").is_none());
        assert!(split_cte_prefix("WITH a AS (SELECT 1").is_none());
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

    /// All schemes the registry knows, for conformance sweeps.
    const ALL_SCHEMES: &[&str] = &[
        "postgres",
        "redshift",
        "mysql",
        "mariadb",
        "snowflake",
        "mssql",
        "bigquery",
        "databricks",
        "clickhouse",
        "oracle",
        "trino",
        "exasol",
        "monetdb",
        "druid",
        "drill",
        "datafusion",
        "duckdb",
        "sqlite",
    ];

    /// Contract for the quantile hook: callers pass the raw (unquoted)
    /// column name and the dialect quotes it. A name needing quoting must
    /// appear quoted — interpolating it raw breaks on any real column whose
    /// name is not a bare lowercase identifier.
    #[test]
    fn quantile_hook_quotes_raw_column_names() {
        for scheme in ALL_SCHEMES {
            let d = dialect_for_scheme(scheme).unwrap();
            let quoted = d.quote_ident("mixed Case");
            let sql = d.sql_quantile("mixed Case", 0.5, "t", &[]);
            assert!(
                sql.contains(&quoted),
                "{scheme}: sql_quantile does not quote its column: {sql}"
            );
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
