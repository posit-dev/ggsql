//! Row filters and sort keys over a result's columns, and the SQL for them.
//!
//! Everything here builds from the request, the result's schema and a
//! [`SqlDialect`], with no database, so it is tested as plain string assembly.
//! The same code writes SQL for the results cache, to filter and sort a view,
//! and for the session's own database, for `result_sql`: identifiers,
//! string and boolean literals, `LIKE` and regular expressions all come from
//! the dialect it is given.

use super::Column;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use ggsql::parser::SourceTree;
use ggsql::reader::SqlDialect;
use serde::{Deserialize, Serialize};

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    #[serde(rename = "=")]
    Eq,
    #[serde(rename = "!=")]
    NotEq,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    LtEq,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    GtEq,
}

/// How a text filter matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextMatch {
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    Regex,
}

/// One row filter. `column` is an index into the result's columns, and every
/// value is a string, checked against the column's type before it reaches SQL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Filter {
    Compare {
        column: usize,
        op: CompareOp,
        value: String,
    },
    Between {
        column: usize,
        low: String,
        high: String,
        #[serde(default)]
        negate: bool,
    },
    IsNull {
        column: usize,
        #[serde(default)]
        negate: bool,
    },
    IsEmpty {
        column: usize,
        #[serde(default)]
        negate: bool,
    },
    IsTrue {
        column: usize,
    },
    IsFalse {
        column: usize,
    },
    Text {
        column: usize,
        #[serde(rename = "match")]
        how: TextMatch,
        term: String,
        #[serde(default)]
        case_sensitive: bool,
    },
    In {
        column: usize,
        values: Vec<String>,
        #[serde(default)]
        negate: bool,
    },
}

impl Filter {
    fn column(&self) -> usize {
        match self {
            Filter::Compare { column, .. }
            | Filter::Between { column, .. }
            | Filter::IsNull { column, .. }
            | Filter::IsEmpty { column, .. }
            | Filter::IsTrue { column }
            | Filter::IsFalse { column }
            | Filter::Text { column, .. }
            | Filter::In { column, .. } => *column,
        }
    }
}

/// One sort key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortKey {
    pub column: usize,
    #[serde(default = "ascending")]
    pub ascending: bool,
}

fn ascending() -> bool {
    true
}

fn column_of(columns: &[Column], index: usize) -> Result<&Column, String> {
    columns
        .get(index)
        .ok_or_else(|| format!("There is no column {index}"))
}

/// The SQL predicate for one row filter.
pub fn predicate(
    filter: &Filter,
    columns: &[Column],
    dialect: &dyn SqlDialect,
) -> Result<String, String> {
    let column = column_of(columns, filter.column())?;
    let c = dialect.quote_ident(&column.name);
    let lit = |v: &str| literal(v, &column.kind, dialect);

    match filter {
        Filter::Compare { op, value, .. } => {
            let op = match op {
                CompareOp::Eq => "=",
                CompareOp::NotEq => "<>",
                CompareOp::Lt => "<",
                CompareOp::LtEq => "<=",
                CompareOp::Gt => ">",
                CompareOp::GtEq => ">=",
            };
            Ok(format!("{c} {op} {}", lit(value)?))
        }
        Filter::Between {
            low, high, negate, ..
        } => {
            let between = format!("{c} BETWEEN {} AND {}", lit(low)?, lit(high)?);
            Ok(if *negate {
                format!("NOT ({between})")
            } else {
                between
            })
        }
        Filter::IsNull { negate: false, .. } => Ok(format!("{c} IS NULL")),
        Filter::IsNull { negate: true, .. } => Ok(format!("{c} IS NOT NULL")),
        Filter::IsEmpty { negate, .. } => Ok(dialect.sql_is_empty_string(&c, *negate)),
        Filter::IsTrue { .. } => Ok(format!("{c} = {}", dialect.sql_boolean_literal(true))),
        Filter::IsFalse { .. } => Ok(format!("{c} = {}", dialect.sql_boolean_literal(false))),
        Filter::Text {
            how,
            term,
            case_sensitive,
            ..
        } => text(&c, *how, term, *case_sensitive, dialect),
        Filter::In { values, negate, .. } => {
            if values.is_empty() {
                // Nothing selected: keep nothing, or exclude nothing.
                return Ok(if *negate { "1 = 1" } else { "1 = 0" }.to_string());
            }
            let list = values
                .iter()
                .map(|v| lit(v))
                .collect::<Result<Vec<_>, _>>()?;
            let not = if *negate { "NOT " } else { "" };
            Ok(format!("{c} {not}IN ({})", list.join(", ")))
        }
    }
}

/// A filter value as a SQL literal of its column's kind.
fn literal(value: &str, kind: &str, dialect: &dyn SqlDialect) -> Result<String, String> {
    match kind {
        "integer" | "floating" => {
            let v = value.trim();
            // Only a finite number passes, so nothing else reaches the SQL
            // unquoted.
            match v.parse::<f64>() {
                Ok(n) if n.is_finite() => Ok(v.to_string()),
                _ => Err(format!("{value:?} is not a number")),
            }
        }
        "boolean" => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(dialect.sql_boolean_literal(true)),
            "false" | "0" => Ok(dialect.sql_boolean_literal(false)),
            _ => Err(format!("{value:?} is not true or false")),
        },
        // A typed literal: not every backend casts a string to compare it with
        // a temporal column (Trino refuses, Oracle reads it by a session
        // setting).
        "date" => parse_date(value.trim())
            .map(|d| dialect.sql_date_literal((d - epoch_date()).num_days() as i32))
            .ok_or_else(|| format!("{value:?} is not a date (YYYY-MM-DD)")),
        "datetime" => parse_datetime(value.trim())
            .map(|dt| dialect.sql_datetime_literal(dt.and_utc().timestamp_micros()))
            .ok_or_else(|| format!("{value:?} is not a date and time (YYYY-MM-DD HH:MM:SS)")),
        "time" => parse_time(value.trim())
            .map(|t| {
                let since_midnight = t - NaiveTime::MIN;
                dialect.sql_time_literal(since_midnight.num_nanoseconds().unwrap_or(0))
            })
            .ok_or_else(|| format!("{value:?} is not a time (HH:MM:SS)")),
        _ => Ok(dialect.sql_string_literal(value)),
    }
}

fn epoch_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).expect("the epoch is a date")
}

fn parse_date(v: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(v, "%Y-%m-%d").ok()
}

/// An ISO date and time, with `T` or a space between them and optional
/// fractional seconds; a date alone is its midnight. No offset: a literal has
/// no time zone, so a database reads it in the session's own.
fn parse_datetime(v: &str) -> Option<NaiveDateTime> {
    [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]
    .iter()
    .find_map(|f| NaiveDateTime::parse_from_str(v, f).ok())
    .or_else(|| parse_date(v).map(|d| d.and_time(NaiveTime::MIN)))
}

fn parse_time(v: &str) -> Option<NaiveTime> {
    ["%H:%M:%S%.f", "%H:%M"]
        .iter()
        .find_map(|f| NaiveTime::parse_from_str(v, f).ok())
}

/// The predicate for a text filter.
fn text(
    c: &str,
    how: TextMatch,
    term: &str,
    case_sensitive: bool,
    dialect: &dyn SqlDialect,
) -> Result<String, String> {
    if how == TextMatch::Regex {
        return dialect
            .sql_regex_match(c, &dialect.sql_string_literal(term), case_sensitive)
            .ok_or_else(|| {
                "ggsql cannot write a regular-expression search for this database".into()
            });
    }

    let escaped = escape_like(term);
    let (pattern, negate) = match how {
        TextMatch::Contains => (format!("%{escaped}%"), false),
        TextMatch::NotContains => (format!("%{escaped}%"), true),
        TextMatch::StartsWith => (format!("{escaped}%"), false),
        TextMatch::EndsWith => (format!("%{escaped}"), false),
        TextMatch::Regex => unreachable!("handled above"),
    };
    let like = dialect.sql_like(c, &pattern, case_sensitive);
    Ok(if negate {
        format!("NOT ({like})")
    } else {
        like
    })
}

/// Escape `LIKE`'s wildcards, and the backslash that escapes them, so a
/// search term matches literally under [`SqlDialect::sql_like`].
fn escape_like(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// The sort keys as `ORDER BY` terms, without the keyword, with NULLs last
/// in either direction whatever the backend's default.
pub fn order_terms(
    keys: &[SortKey],
    columns: &[Column],
    dialect: &dyn SqlDialect,
) -> Result<Vec<String>, String> {
    keys.iter()
        .map(|key| {
            let column = column_of(columns, key.column)?;
            // NULLs last either way, as the results cache's DuckDB sorts
            // them, so SQL written for another database orders rows as the
            // view does.
            Ok(dialect.sql_order_term(&dialect.quote_ident(&column.name), key.ascending, true))
        })
        .collect()
}

/// Why `column` cannot be named in SQL over the query, if it cannot.
fn unnameable(columns: &[Column], index: usize) -> Option<String> {
    let column = columns.get(index)?;
    column.repeated.then(|| {
        format!(
            "column {:?} repeats a name in the query, which SQL cannot refer to",
            column.name
        )
    })
}

/// SQL for `dialect` reproducing a view of the result of `query`: its last
/// statement, filtered and sorted, after the statements that come before it.
///
/// A filter or sort key that cannot be expressed in `dialect` is left out with
/// a comment saying so, rather than failing the whole conversion.
pub fn to_sql(
    filters: &[Filter],
    sort: &[SortKey],
    query: &str,
    columns: &[Column],
    dialect: &dyn SqlDialect,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut predicates = Vec::new();
    for filter in filters {
        let built = match unnameable(columns, filter.column()) {
            Some(why) => Err(why),
            None => predicate(filter, columns, dialect),
        };
        match built {
            Ok(p) => predicates.push(p),
            Err(e) => lines.push(format!("-- Filter left out: {e}")),
        }
    }
    let mut keys = Vec::new();
    for key in sort {
        match unnameable(columns, key.column) {
            Some(why) => lines.push(format!("-- Sort key left out: {why}")),
            None => keys.push(key.clone()),
        }
    }

    let Statements {
        setup,
        last,
        last_is_query,
    } = split_statements(query);
    for stmt in setup {
        lines.extend(format!("{stmt};").lines().map(str::to_string));
    }
    if !last_is_query {
        // Only a query can be wrapped, so the rest is said rather than written.
        if !predicates.is_empty() || !keys.is_empty() {
            lines.push(
                "-- Filters and sort left out: the last statement is not a query \
                 that can be filtered or sorted"
                    .to_string(),
            );
        }
        lines.extend(format!("{last};").lines().map(str::to_string));
        return lines;
    }
    lines.push("SELECT *".to_string());
    lines.extend(
        format!("FROM {}", subquery(&last, dialect))
            .lines()
            .map(str::to_string),
    );
    for (i, p) in predicates.iter().enumerate() {
        let keyword = if i == 0 { "WHERE" } else { "  AND" };
        lines.push(format!("{keyword} {p}"));
    }
    match order_terms(&keys, columns, dialect) {
        Ok(terms) if !terms.is_empty() => lines.push(format!("ORDER BY {}", terms.join(", "))),
        Ok(_) => {}
        Err(e) => lines.push(format!("-- Sort left out: {e}")),
    }
    lines
}

/// A cell's last statement, whose result was kept, and those before it.
struct Statements {
    setup: Vec<String>,
    last: String,
    /// Whether the last statement is a query that can be a derived table.
    /// `DESCRIBE`, `SHOW`, `PRAGMA` and the like are not.
    last_is_query: bool,
}

/// A cell's statements. A query that does not parse cleanly is one statement,
/// taken to be a query.
fn split_statements(query: &str) -> Statements {
    let whole = || Statements {
        setup: Vec::new(),
        last: query.trim().trim_end_matches(';').to_string(),
        last_is_query: true,
    };
    let Ok(tree) = SourceTree::new(query) else {
        return whole();
    };
    let root = tree.root();
    // Tree-sitter recovers from what it cannot parse with ERROR nodes, and
    // statement boundaries found around one are guesses, as in
    // `SELECT $$x;y$$`.
    if root.has_error() {
        return whole();
    }
    let mut statements: Vec<(String, bool)> = tree
        .find_nodes(&root, "(sql_statement) @s")
        .into_iter()
        // The outermost statements: a subquery can be a statement of its own.
        .filter(|node| {
            let mut up = node.parent();
            while let Some(p) = up {
                if p.kind() == "sql_statement" {
                    return false;
                }
                up = p.parent();
            }
            true
        })
        .map(|node| {
            let is_query = node.child(0).is_some_and(|c| {
                matches!(
                    c.kind(),
                    "select_statement" | "with_statement" | "from_statement"
                )
            });
            (tree.get_text(&node).trim().to_string(), is_query)
        })
        .filter(|(text, _)| !text.is_empty())
        .collect();
    match statements.pop() {
        Some((last, last_is_query)) => Statements {
            setup: statements.into_iter().map(|(text, _)| text).collect(),
            last,
            last_is_query,
        },
        None => whole(),
    }
}

/// A query as a derived table, for wrapping in a view's filters.
fn subquery(query: &str, dialect: &dyn SqlDialect) -> String {
    let query = query.trim().trim_end_matches(';').trim_end();
    let body: Vec<String> = query.lines().map(|l| format!("    {l}")).collect();
    format!(
        "(\n{}\n) {}",
        body.join("\n"),
        dialect.sql_table_alias("result")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggsql::reader::dialects::DuckDbDialect;
    use ggsql::reader::registry::dialect_override as dialect_for_scheme;
    use ggsql::reader::AnsiDialect;
    use serde_json::{json, Value};

    fn columns() -> Vec<Column> {
        [("n", "integer"), ("name", "string"), ("ok", "boolean")]
            .iter()
            .map(|(name, kind)| Column {
                name: name.to_string(),
                type_name: String::new(),
                kind: kind.to_string(),
                repeated: false,
            })
            .collect()
    }

    fn sql(f: Value) -> Result<String, String> {
        let filter: Filter = serde_json::from_value(f).unwrap();
        predicate(&filter, &columns(), &AnsiDialect)
    }

    #[test]
    fn test_compare_maps_not_equal_and_checks_numbers() {
        assert_eq!(
            sql(json!({"type": "compare", "column": 0, "op": "!=", "value": "3"})).unwrap(),
            "\"n\" <> 3"
        );
        assert!(
            sql(json!({"type": "compare", "column": 0, "op": "=", "value": "1; DROP"})).is_err()
        );
        assert!(sql(json!({"type": "compare", "column": 0, "op": "=", "value": "inf"})).is_err());
    }

    #[test]
    fn test_string_values_are_quoted() {
        assert_eq!(
            sql(json!({"type": "compare", "column": 1, "op": "=", "value": "O'Brien"})).unwrap(),
            "\"name\" = 'O''Brien'"
        );
    }

    #[test]
    fn test_negated_between() {
        assert_eq!(
            sql(json!({"type": "between", "column": 0, "low": "1", "high": "5", "negate": true}))
                .unwrap(),
            "NOT (\"n\" BETWEEN 1 AND 5)"
        );
    }

    #[test]
    fn test_text_escapes_wildcards_and_folds_case() {
        assert_eq!(
            sql(json!({"type": "text", "column": 1, "match": "contains", "term": "50%_\\"}))
                .unwrap(),
            r#"LOWER("name") LIKE LOWER('%50\%\_\\%') ESCAPE '\'"#
        );
        assert_eq!(
            sql(
                json!({"type": "text", "column": 1, "match": "starts_with", "term": "a",
                       "case_sensitive": true})
            )
            .unwrap(),
            r#""name" LIKE 'a%' ESCAPE '\'"#
        );
        assert_eq!(
            sql(json!({"type": "text", "column": 1, "match": "not_contains", "term": "a"}))
                .unwrap(),
            r#"NOT (LOWER("name") LIKE LOWER('%a%') ESCAPE '\')"#
        );
    }

    #[test]
    fn test_the_like_escape_matches_literally_in_duckdb() {
        let reader = ggsql::reader::connection::reader_from_uri("duckdb://memory").unwrap();
        let filter: Filter = serde_json::from_value(
            json!({"type": "text", "column": 1, "match": "contains", "term": "5%\\_"}),
        )
        .unwrap();
        let p = predicate(&filter, &columns(), &DuckDbDialect).unwrap();
        let df = reader
            .execute_sql(&format!(
                r"SELECT name FROM (VALUES ('a5%\_b'), ('a5x\_b'), ('a5%\xb')) t(name) WHERE {p}"
            ))
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_regex_comes_from_the_dialect() {
        let f: Filter = serde_json::from_value(
            json!({"type": "text", "column": 1, "match": "regex", "term": "^a"}),
        )
        .unwrap();
        assert_eq!(
            predicate(&f, &columns(), &DuckDbDialect).unwrap(),
            "regexp_matches(\"name\", '^a', 'i')"
        );
        assert!(predicate(&f, &columns(), &AnsiDialect).is_err());
    }

    fn temporal_columns() -> Vec<Column> {
        [("d", "date"), ("ts", "datetime"), ("t", "time")]
            .iter()
            .map(|(name, kind)| Column {
                name: name.to_string(),
                type_name: String::new(),
                kind: kind.to_string(),
                repeated: false,
            })
            .collect()
    }

    #[test]
    fn test_temporal_values_are_typed_literals_of_the_dialect() {
        let f = |column: usize, value: &str, dialect: &dyn SqlDialect| {
            let filter: Filter = serde_json::from_value(
                json!({"type": "compare", "column": column, "op": ">=", "value": value}),
            )
            .unwrap();
            predicate(&filter, &temporal_columns(), dialect)
        };
        assert_eq!(
            f(0, "2024-03-05", &AnsiDialect).unwrap(),
            "\"d\" >= DATE '2024-03-05'"
        );
        assert_eq!(
            f(1, "2024-03-05T06:07:08.5", &AnsiDialect).unwrap(),
            "\"ts\" >= TIMESTAMP '2024-03-05 06:07:08.500000'"
        );
        assert_eq!(
            f(1, "2024-03-05", &AnsiDialect).unwrap(),
            "\"ts\" >= TIMESTAMP '2024-03-05 00:00:00'"
        );
        assert_eq!(
            f(2, "06:07", &AnsiDialect).unwrap(),
            "\"t\" >= TIME '06:07:00'"
        );
        // An override, not a string the backend may refuse to compare.
        let trino = dialect_for_scheme("trino").unwrap();
        let mssql = dialect_for_scheme("mssql").unwrap();
        assert_eq!(
            f(0, "2024-03-05", trino).unwrap(),
            format!("\"d\" >= {}", trino.sql_date_literal(19787))
        );
        assert_ne!(
            f(0, "2024-03-05", mssql).unwrap(),
            f(0, "2024-03-05", &AnsiDialect).unwrap()
        );
        for (column, bad) in [
            (0, "March 5"),
            (1, "2024-03-05 06:07:08+02:00"),
            (2, "noon"),
        ] {
            assert!(f(column, bad, &AnsiDialect).is_err(), "{bad}");
        }
    }

    #[test]
    fn test_temporal_filters_run_in_duckdb() {
        let reader = ggsql::reader::connection::reader_from_uri("duckdb://memory").unwrap();
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "between", "column": 0, "low": "2024-01-01", "high": "2024-01-31"},
            {"type": "compare", "column": 1, "op": ">", "value": "2024-01-10 12:00"},
            {"type": "in", "column": 2, "values": ["09:30:00"]}
        ]))
        .unwrap();
        let where_ = filters
            .iter()
            .map(|f| predicate(f, &temporal_columns(), &DuckDbDialect).unwrap())
            .collect::<Vec<_>>()
            .join(" AND ");
        let df = reader
            .execute_sql(&format!(
                "SELECT * FROM (VALUES \
                 (DATE '2024-01-15', TIMESTAMP '2024-01-15 00:00:00', TIME '09:30:00'), \
                 (DATE '2024-01-15', TIMESTAMP '2024-01-10 11:00:00', TIME '09:30:00'), \
                 (DATE '2024-02-15', TIMESTAMP '2024-02-15 00:00:00', TIME '09:30:00')\
                 ) t(d, ts, t) WHERE {where_}"
            ))
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_in() {
        assert_eq!(
            sql(json!({"type": "in", "column": 1, "values": ["a", "b"], "negate": true})).unwrap(),
            "\"name\" NOT IN ('a', 'b')"
        );
        assert_eq!(
            sql(json!({"type": "in", "column": 1, "values": []})).unwrap(),
            "1 = 0"
        );
    }

    #[test]
    fn test_booleans_use_the_dialect() {
        assert_eq!(
            sql(json!({"type": "is_true", "column": 2})).unwrap(),
            format!("\"ok\" = {}", AnsiDialect.sql_boolean_literal(true))
        );
        assert!(sql(json!({"type": "compare", "column": 2, "op": "=", "value": "maybe"})).is_err());
    }

    #[test]
    fn test_sort_puts_nulls_last_in_every_dialect() {
        let desc = [SortKey {
            column: 0,
            ascending: false,
        }];
        let mysql = dialect_for_scheme("mysql").unwrap();
        assert_eq!(
            order_terms(&desc, &columns(), mysql).unwrap(),
            ["CASE WHEN `n` IS NULL THEN 1 ELSE 0 END, `n` DESC"]
        );
        assert_eq!(
            order_terms(&desc, &columns(), &AnsiDialect).unwrap(),
            ["\"n\" DESC NULLS LAST"]
        );
        // And both shapes run, with the NULL last either way.
        for uri in ["duckdb://memory", "sqlite://memory"] {
            let reader = ggsql::reader::connection::reader_from_uri(uri).unwrap();
            let dialect = reader.dialect();
            for ascending in [true, false] {
                let key = [SortKey {
                    column: 0,
                    ascending,
                }];
                let terms = order_terms(&key, &columns(), dialect).unwrap().join(", ");
                let both = [terms, nulls_order_by(dialect, ascending)];
                for order in both {
                    let df = reader
                        .execute_sql(&format!(
                            "SELECT n FROM (SELECT NULL AS n UNION ALL SELECT 1 UNION ALL SELECT 2) t \
                             ORDER BY {order}"
                        ))
                        .unwrap();
                    assert!(df.column("n").unwrap().is_null(2), "{uri} {order}");
                }
            }
        }
    }

    /// The `CASE` form MySQL and SQL Server get, for running elsewhere.
    fn nulls_order_by(dialect: &dyn SqlDialect, ascending: bool) -> String {
        let mysql = dialect_for_scheme("mysql").unwrap();
        mysql
            .sql_order_term("n", ascending, true)
            .replace('`', &dialect.quote_ident("")[..1])
    }

    #[test]
    fn test_an_unknown_column_is_an_error() {
        assert!(sql(json!({"type": "is_null", "column": 9})).is_err());
        assert!(order_terms(
            &[SortKey {
                column: 9,
                ascending: true
            }],
            &columns(),
            &AnsiDialect
        )
        .is_err());
    }

    fn filters(f: Value) -> Vec<Filter> {
        serde_json::from_value(f).unwrap()
    }

    const BY_N: [SortKey; 1] = [SortKey {
        column: 0,
        ascending: true,
    }];

    #[test]
    fn test_to_sql_wraps_the_query() {
        let filters = filters(json!([
            {"type": "compare", "column": 0, "op": ">", "value": "2"},
            {"type": "is_null", "column": 1}
        ]));
        assert_eq!(
            to_sql(
                &filters,
                &BY_N,
                "SELECT *\nFROM t;",
                &columns(),
                &AnsiDialect
            ),
            vec![
                "SELECT *",
                "FROM (",
                "    SELECT *",
                "    FROM t",
                ") AS \"result\"",
                "WHERE \"n\" > 2",
                "  AND \"name\" IS NULL",
                "ORDER BY \"n\" ASC NULLS LAST",
            ]
        );
    }

    #[test]
    fn test_to_sql_writes_the_session_dialect() {
        let both = filters(json!([
            {"type": "text", "column": 1, "match": "contains", "term": "it's"},
            {"type": "compare", "column": 1, "op": "=", "value": "a\\b"}
        ]));
        let cases = [
            (
                "mysql",
                r"LOWER(`name`) LIKE LOWER('%it\'s%') ESCAPE '\\'",
                r"`name` = 'a\\b'",
            ),
            (
                "databricks",
                r"LOWER(`name`) LIKE LOWER('%it\'s%') ESCAPE '\\'",
                r"`name` = 'a\\b'",
            ),
            (
                "bigquery",
                r"LOWER(`name`) LIKE LOWER('%it\'s%')",
                r"`name` = 'a\\b'",
            ),
            (
                "clickhouse",
                r"LOWER(`name`) LIKE LOWER('%it\'s%')",
                r"`name` = 'a\\b'",
            ),
            (
                "snowflake",
                r#"LOWER("name") LIKE LOWER('%it\'s%') ESCAPE '\\'"#,
                r#""name" = 'a\\b'"#,
            ),
            (
                "postgres",
                r#"LOWER("name") LIKE LOWER('%it''s%') ESCAPE '\'"#,
                r#""name" = 'a\b'"#,
            ),
        ];
        for (scheme, like, compare) in cases {
            let dialect = dialect_for_scheme(scheme).unwrap();
            let sql = to_sql(&both, &BY_N, "SELECT 1", &columns(), dialect).join("\n");
            assert!(sql.contains(&format!("WHERE {like}\n")), "{scheme}: {sql}");
            assert!(sql.contains(&format!("AND {compare}\n")), "{scheme}: {sql}");
        }
        // The wildcard tail of a backslash-escaping dialect's pattern still
        // ends the literal: `'%\\%'`, not `'%\%'`.
        let backslash = filters(json!([
            {"type": "text", "column": 1, "match": "ends_with", "term": "\\", "case_sensitive": true}
        ]));
        let mysql = dialect_for_scheme("mysql").unwrap();
        let sql = to_sql(&backslash, &[], "SELECT 1", &columns(), mysql).join("\n");
        assert!(
            sql.contains(r"CAST(`name` AS BINARY) LIKE CAST('%\\\\' AS BINARY) ESCAPE '\\'"),
            "{sql}"
        );
    }

    #[test]
    fn test_to_sql_escapes_a_tsql_character_class() {
        let filters = filters(json!([
            {"type": "text", "column": 1, "match": "starts_with", "term": "[a]", "case_sensitive": true}
        ]));
        let mssql = dialect_for_scheme("mssql").unwrap();
        let sql = to_sql(&filters, &[], "SELECT 1", &columns(), mssql).join("\n");
        assert!(
            sql.contains(r#""name" COLLATE Latin1_General_CS_AS LIKE '\[a]%' ESCAPE '\'"#),
            "{sql}"
        );
    }

    #[test]
    fn test_to_sql_keeps_the_statements_before_the_query() {
        let query = "CREATE TABLE t AS SELECT 1 AS n;\nINSERT INTO t VALUES (2);\nSELECT * FROM t";
        assert_eq!(
            to_sql(&[], &BY_N, query, &columns(), &AnsiDialect),
            vec![
                "CREATE TABLE t AS SELECT 1 AS n;",
                "INSERT INTO t VALUES (2);",
                "SELECT *",
                "FROM (",
                "    SELECT * FROM t",
                ") AS \"result\"",
                "ORDER BY \"n\" ASC NULLS LAST",
            ]
        );
        // And what it writes runs.
        let reader = ggsql::reader::connection::reader_from_uri("duckdb://memory").unwrap();
        let lines = to_sql(&[], &BY_N, query, &columns(), &DuckDbDialect).join("\n");
        let mut last = None;
        for stmt in lines.split(';') {
            last = Some(reader.execute_sql(stmt).unwrap());
        }
        assert_eq!(last.unwrap().height(), 2);
    }

    #[test]
    fn test_a_query_that_does_not_parse_cleanly_is_one_statement() {
        // Parsed as `SELECT 1`, an ERROR node, then `x;y$$ AS s` as a
        // statement of its own.
        let query = "SELECT 1; SELECT $$x;y$$ AS s";
        let split = split_statements(query);
        assert!(split.setup.is_empty());
        assert_eq!(split.last, query);
        assert!(split.last_is_query);
    }

    #[test]
    fn test_to_sql_does_not_wrap_a_statement_that_is_not_a_query() {
        let lines = to_sql(
            &[],
            &BY_N,
            "CREATE TABLE t AS SELECT 1 AS n;\nDESCRIBE t",
            &columns(),
            &AnsiDialect,
        );
        assert_eq!(
            lines,
            [
                "CREATE TABLE t AS SELECT 1 AS n;",
                "-- Filters and sort left out: the last statement is not a query that can be \
                 filtered or sorted",
                "DESCRIBE t;",
            ]
        );
        // Queries are wrapped, however they start.
        for query in ["WITH a AS (SELECT 1 AS n) SELECT * FROM a", "FROM t"] {
            let lines = to_sql(&[], &BY_N, query, &columns(), &AnsiDialect);
            assert_eq!(lines[0], "SELECT *", "{query}: {lines:?}");
        }
    }

    #[test]
    fn test_to_sql_leaves_out_a_repeated_column() {
        let mut cols = columns();
        cols[1].repeated = true;
        let filters = filters(json!([{"type": "is_null", "column": 1}]));
        let sort = [SortKey {
            column: 1,
            ascending: true,
        }];
        let lines = to_sql(&filters, &sort, "SELECT 1", &cols, &AnsiDialect);
        assert!(
            lines[0].starts_with("-- Filter left out: column \"name\" repeats"),
            "{lines:?}"
        );
        assert!(lines[1].starts_with("-- Sort key left out"), "{lines:?}");
        assert!(!lines
            .iter()
            .any(|l| l.starts_with("WHERE") || l.starts_with("ORDER")));
    }

    #[test]
    fn test_to_sql_leaves_out_what_it_cannot_express() {
        let filters = filters(json!([
            {"type": "text", "column": 1, "match": "regex", "term": "x", "case_sensitive": true}
        ]));
        let lines = to_sql(&filters, &[], "SELECT 1", &columns(), &AnsiDialect);
        assert!(lines[0].starts_with("-- Filter left out"), "{lines:?}");
        assert_eq!(
            &lines[1..],
            ["SELECT *", "FROM (", "    SELECT 1", ") AS \"result\""]
        );
    }
}
