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

/// String literal for backends that read a backslash inside a string literal
/// as an escape (MySQL/MariaDB, ClickHouse, BigQuery, Databricks, Snowflake,
/// Redshift): backslashes and single quotes are both backslash-escaped.
/// BigQuery and Databricks have no `''` escape at all.
pub(crate) fn backslash_quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// [`SqlDialect::sql_like`] with `ESCAPE` naming the backslash, as a string
/// literal of the dialect's own, so the clause is valid wherever its literals
/// are.
pub(crate) fn like_with_escape_clause<D: SqlDialect + ?Sized>(
    dialect: &D,
    expr: &str,
    pattern: &str,
    case_sensitive: bool,
) -> String {
    format!(
        "{} ESCAPE {}",
        like_without_escape_clause(dialect, expr, pattern, case_sensitive),
        dialect.sql_string_literal("\\")
    )
}

/// [`SqlDialect::sql_like`] for backends whose `LIKE` has no `ESCAPE` clause
/// and always escapes with a backslash (ClickHouse, BigQuery).
pub(crate) fn like_without_escape_clause<D: SqlDialect + ?Sized>(
    dialect: &D,
    expr: &str,
    pattern: &str,
    case_sensitive: bool,
) -> String {
    let pattern = dialect.sql_string_literal(pattern);
    if case_sensitive {
        format!("{expr} LIKE {pattern}")
    } else {
        format!("LOWER({expr}) LIKE LOWER({pattern})")
    }
}

/// [`SqlDialect::sql_order_term`] for backends without `NULLS FIRST` /
/// `NULLS LAST` (MySQL/MariaDB, SQL Server): a leading key that sorts the
/// NULLs to the chosen end, then the expression itself.
pub(crate) fn nulls_order_by_case(expr: &str, ascending: bool, nulls_last: bool) -> String {
    let (null, other) = if nulls_last { (1, 0) } else { (0, 1) };
    let direction = if ascending { "ASC" } else { "DESC" };
    format!("CASE WHEN {expr} IS NULL THEN {null} ELSE {other} END, {expr} {direction}")
}

/// The SQL string literal `pattern` with an inline `(?i)` or `(?-i)` flag in
/// front, for backends whose regular expressions take case sensitivity only
/// that way (RE2, Java, ICU, PCRE).
pub(crate) fn inline_regex_flag<D: SqlDialect + ?Sized>(
    dialect: &D,
    pattern: &str,
    case_sensitive: bool,
) -> String {
    let flag = if case_sensitive { "(?-i)" } else { "(?i)" };
    format!("CONCAT({}, {pattern})", dialect.sql_string_literal(flag))
}

/// A [`SqlDialect::sql_like`] pattern as a `GLOB` pattern: `%` and `_`
/// become `*` and `?`, a backslash-escaped character is literal, and `GLOB`'s
/// own `*`, `?` and `[` are bracketed so they match themselves.
pub(crate) fn like_to_glob(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        let literal = match ch {
            '\\' => match chars.next() {
                Some(next) => next,
                None => break,
            },
            '%' => {
                out.push('*');
                continue;
            }
            '_' => {
                out.push('?');
                continue;
            }
            other => other,
        };
        match literal {
            '*' | '?' | '[' => {
                out.push('[');
                out.push(literal);
                out.push(']');
            }
            other => out.push(other),
        }
    }
    out
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
            let ty = dialect.type_names().number.unwrap_or("DOUBLE PRECISION");
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
            let ty = dialect.type_names().number.unwrap_or("DOUBLE");
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
    use crate::reader::registry::{by_scheme, detect};

    fn detect_dialect(
        dbms_name: Option<&str>,
        driver_hint: Option<&str>,
    ) -> crate::Result<crate::reader::DialectRef> {
        detect(dbms_name, driver_hint).map(|e| e.dialect())
    }

    fn dialect_for_scheme(scheme: &str) -> Option<crate::reader::DialectRef> {
        by_scheme(scheme).map(|e| e.dialect())
    }

    fn assert_type_name(d: &dyn SqlDialect, expected: Option<&str>) {
        assert_eq!(d.type_names().number, expected);
    }

    #[test]
    fn string_literals_follow_the_backends_escapes() {
        let value = r"it's a\b";
        for scheme in ["mysql", "bigquery", "clickhouse", "databricks", "snowflake"] {
            let d = dialect_for_scheme(scheme).unwrap();
            assert_eq!(d.sql_string_literal(value), r"'it\'s a\\b'", "{scheme}");
        }
        for scheme in ["duckdb", "postgres", "sqlite", "mssql", "trino"] {
            let d = dialect_for_scheme(scheme).unwrap();
            assert_eq!(d.sql_string_literal(value), r"'it''s a\b'", "{scheme}");
        }
    }

    #[test]
    fn like_names_its_escape_where_the_backend_has_the_clause() {
        let like = |scheme: &str| {
            dialect_for_scheme(scheme)
                .unwrap()
                .sql_like("c", r"a\%", true)
        };
        assert_eq!(like("duckdb"), r"c LIKE 'a\%' ESCAPE '\'");
        assert_eq!(
            like("mysql"),
            r"CAST(c AS BINARY) LIKE CAST('a\\%' AS BINARY) ESCAPE '\\'"
        );
        assert_eq!(like("bigquery"), r"c LIKE 'a\\%'");
        assert_eq!(like("clickhouse"), r"c LIKE 'a\\%'");
        assert_eq!(
            dialect_for_scheme("mssql")
                .unwrap()
                .sql_like("c", "[x]%", false),
            r"LOWER(c) LIKE LOWER('\[x]%') ESCAPE '\'"
        );
    }

    #[test]
    fn case_sensitive_like_does_not_follow_a_case_folding_collation() {
        let like = |scheme: &str| {
            dialect_for_scheme(scheme)
                .unwrap()
                .sql_like("c", "a%", true)
        };
        assert_eq!(like("sqlite"), "c GLOB 'a*'");
        assert_eq!(
            like("mssql"),
            r"c COLLATE Latin1_General_CS_AS LIKE 'a%' ESCAPE '\'"
        );
        assert_eq!(like_to_glob(r"%a\%b\_*?[x]_"), "*a%b_[*][?][[]x]?");
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn case_sensitive_like_runs_in_sqlite() {
        use crate::reader::Reader;
        let reader =
            crate::reader::SqliteReader::from_connection_string("sqlite://memory").unwrap();
        let p = SqliteDialect.sql_like("s", r"%A\_[%", true);
        let df = reader
            .execute_sql(&format!(
                "SELECT s FROM (SELECT 'xA_[y' AS s UNION ALL SELECT 'xa_[y' UNION ALL SELECT 'xAb[y') WHERE {p}"
            ))
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn an_empty_string_is_null_on_oracle() {
        let oracle = dialect_for_scheme("oracle").unwrap();
        assert_eq!(oracle.sql_is_empty_string("c", false), "c IS NULL");
        assert_eq!(oracle.sql_is_empty_string("c", true), "c IS NOT NULL");
        assert_eq!(AnsiDialect.sql_is_empty_string("c", false), "c = ''");
    }

    /// The default `LIKE` escape runs, with the backslash as written.
    #[cfg(feature = "duckdb")]
    #[test]
    fn like_escape_runs_in_duckdb() {
        use crate::reader::Reader;
        let reader =
            crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let p = DuckDbDialect.sql_like("s", r"%a\_b\\%", true);
        let df = reader
            .execute_sql(&format!(
                r"SELECT s FROM (VALUES ('xa_b\y'), ('xaxb\y'), ('xa_by')) t(s) WHERE {p}"
            ))
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn regex_search_is_unanchored_and_takes_case_where_the_backend_has_it() {
        let regex = |scheme: &str, cs: bool| {
            dialect_for_scheme(scheme)
                .unwrap()
                .sql_regex_match("c", "'^a'", cs)
        };
        assert_eq!(
            regex("bigquery", false).unwrap(),
            r"REGEXP_CONTAINS(c, CONCAT('(?i)', '^a'))"
        );
        assert_eq!(
            regex("mysql", true).unwrap(),
            r"c REGEXP CONCAT('(?-i)', '^a')"
        );
        assert_eq!(
            regex("snowflake", false).unwrap(),
            r"REGEXP_INSTR(c, '^a', 1, 1, 0, 'i') > 0"
        );
        assert_eq!(regex("oracle", true).unwrap(), r"REGEXP_LIKE(c, '^a', 'c')");
        assert_eq!(
            regex("duckdb", false).unwrap(),
            "regexp_matches(c, '^a', 'i')"
        );
        assert_eq!(regex("postgres", true).unwrap(), "c ~ '^a'");
        for scheme in ["clickhouse", "trino", "databricks", "redshift"] {
            assert!(regex(scheme, false).is_some(), "{scheme}");
        }
        for scheme in ["sqlite", "mssql"] {
            assert!(regex(scheme, false).is_none(), "{scheme}");
        }
    }

    /// Evaluate the CASE emulations against DuckDB's native
    /// `GREATEST`/`LEAST` so the left-folded chain is checked as executed
    /// SQL, not just as a string.
    #[cfg(feature = "duckdb")]
    #[test]
    fn case_greatest_least_evaluate_like_native() {
        use crate::array_util::as_f64;
        use crate::reader::Reader;

        let reader =
            crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let cases: &[&[&str]] = &[
            &["1.0", "2.0", "3.0"],
            &["3.5", "-2.0", "0.25"],
            &["10.0", "10.0", "10.0"],
            &["-1.5", "-20.0", "-3.25"],
            &["0.0", "-0.0"],
        ];
        for exprs in cases {
            let list = exprs.join(", ");
            for (emulated, native) in [
                (case_greatest(exprs), format!("GREATEST({list})")),
                (case_least(exprs), format!("LEAST({list})")),
            ] {
                let df = reader
                    .execute_sql(&format!("SELECT {emulated} AS emu, {native} AS nat"))
                    .unwrap();
                let emu = as_f64(df.column("emu").unwrap()).unwrap().value(0);
                let nat = as_f64(df.column("nat").unwrap()).unwrap().value(0);
                assert_eq!(emu, nat, "emulated {emulated} != native {native}");
            }
        }
    }

    #[test]
    fn detects_from_dbms_name() {
        assert_type_name(
            detect_dialect(Some("PostgreSQL"), None).unwrap(),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(detect_dialect(Some("MySQL"), None).unwrap(), Some("DOUBLE"));
        assert_type_name(
            detect_dialect(Some("Microsoft SQL Server"), None).unwrap(),
            Some("FLOAT"),
        );
        assert_type_name(
            detect_dialect(Some("Snowflake"), None).unwrap(),
            Some("DOUBLE"),
        );
        assert_type_name(
            detect_dialect(Some("Oracle"), None).unwrap(),
            Some("BINARY_DOUBLE"),
        );
        assert_type_name(
            detect_dialect(Some("ClickHouse"), None).unwrap(),
            Some("Nullable(Float64)"),
        );
        assert_type_name(
            detect_dialect(Some("Amazon Redshift"), None).unwrap(),
            Some("DOUBLE PRECISION"),
        );
    }

    #[test]
    fn falls_through_to_driver_hint() {
        assert_type_name(
            detect_dialect(Some("Unknown DBMS"), Some("PostgreSQL Unicode")).unwrap(),
            Some("DOUBLE PRECISION"),
        );
        assert_type_name(
            detect_dialect(None, Some("msodbcsql18")).unwrap(),
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
        assert_eq!(d.type_names().boolean, Some("BIT"));
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
            assert_eq!(d.type_names().number, probe, "scheme {scheme}");
        }
        assert!(dialect_for_scheme("nosuchdb").is_none());
    }

    /// Persistent internal tables (regular CREATE TABLE materialization)
    /// need explicit cleanup; genuine temp tables and in-process engines do
    /// not.
    #[test]
    fn internal_tables_need_cleanup_only_for_persistent_dialects() {
        for scheme in ["mysql", "mariadb", "mssql", "oracle"] {
            assert!(
                dialect_for_scheme(scheme)
                    .unwrap()
                    .internal_tables_need_cleanup(),
                "scheme {scheme} should need cleanup"
            );
        }
        for scheme in [
            "postgres",
            "duckdb",
            "sqlite",
            "datafusion",
            "clickhouse",
            "bigquery",
        ] {
            assert!(
                !dialect_for_scheme(scheme)
                    .unwrap()
                    .internal_tables_need_cleanup(),
                "scheme {scheme} should not need cleanup"
            );
        }
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
