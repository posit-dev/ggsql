//! Row filters and sort keys over a result's columns, and the SQL for them.
//!
//! Everything here builds from the request, the result's schema and a
//! [`SqlDialect`], with no database, so it is tested as plain string assembly.
//! The same code writes SQL for the results cache, to filter and sort a view,
//! and for the session's own database, for `result_sql`: identifiers,
//! booleans and regular expressions all come from the dialect it is given.

use super::Column;
use ggsql::naming::quote_literal;
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
        Filter::IsEmpty { negate: false, .. } => Ok(format!("{c} = ''")),
        Filter::IsEmpty { negate: true, .. } => Ok(format!("{c} <> ''")),
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
        // Dates and times too: every backend compares a string literal against
        // a temporal column by casting it.
        _ => Ok(quote_literal(value)),
    }
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
            .sql_regex_match(c, &quote_literal(term), case_sensitive)
            .ok_or_else(|| "This database does not support regular-expression search".into());
    }

    let escaped = escape_like(term);
    let (pattern, negate) = match how {
        TextMatch::Contains => (format!("%{escaped}%"), false),
        TextMatch::NotContains => (format!("%{escaped}%"), true),
        TextMatch::StartsWith => (format!("{escaped}%"), false),
        TextMatch::EndsWith => (format!("%{escaped}"), false),
        TextMatch::Regex => unreachable!("handled above"),
    };
    let (lhs, rhs) = if case_sensitive {
        (c.to_string(), quote_literal(&pattern))
    } else {
        (
            format!("LOWER({c})"),
            format!("LOWER({})", quote_literal(&pattern)),
        )
    };
    let not = if negate { "NOT " } else { "" };
    Ok(format!("{lhs} {not}LIKE {rhs} ESCAPE '{LIKE_ESCAPE}'"))
}

/// `LIKE`'s escape character. Not a backslash, which MySQL and Snowflake
/// read as an escape inside the string literal itself, so `'\'` would never
/// end there.
const LIKE_ESCAPE: char = '!';

/// Escape `LIKE`'s wildcards, so a search term matches literally.
fn escape_like(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        if matches!(ch, '%' | '_') || ch == LIKE_ESCAPE {
            out.push(LIKE_ESCAPE);
        }
        out.push(ch);
    }
    out
}

/// The sort keys as `ORDER BY` terms, without the keyword.
pub fn order_terms(
    keys: &[SortKey],
    columns: &[Column],
    dialect: &dyn SqlDialect,
) -> Result<Vec<String>, String> {
    keys.iter()
        .map(|key| {
            let column = column_of(columns, key.column)?;
            let direction = if key.ascending { "ASC" } else { "DESC" };
            Ok(format!("{} {direction}", dialect.quote_ident(&column.name)))
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

    let (setup, last) = split_statements(query);
    for stmt in setup {
        lines.extend(format!("{stmt};").lines().map(str::to_string));
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

/// A cell's statements: those before the last, and the last, which is the one
/// whose result was kept. A query that does not parse is one statement.
fn split_statements(query: &str) -> (Vec<String>, String) {
    let whole = || (Vec::new(), query.trim().trim_end_matches(';').to_string());
    let Ok(tree) = SourceTree::new(query) else {
        return whole();
    };
    let root = tree.root();
    let mut statements: Vec<String> = tree
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
        .map(|node| tree.get_text(&node).trim().to_string())
        .filter(|text| !text.is_empty())
        .collect();
    match statements.pop() {
        Some(last) => (statements, last),
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
            sql(json!({"type": "text", "column": 1, "match": "contains", "term": "50%_!"}))
                .unwrap(),
            "LOWER(\"name\") LIKE LOWER('%50!%!_!!%') ESCAPE '!'"
        );
        assert_eq!(
            sql(
                json!({"type": "text", "column": 1, "match": "starts_with", "term": "a",
                       "case_sensitive": true})
            )
            .unwrap(),
            "\"name\" LIKE 'a%' ESCAPE '!'"
        );
    }

    #[test]
    fn test_the_like_escape_matches_literally_in_duckdb() {
        let reader = ggsql::reader::connection::reader_from_uri("duckdb://memory").unwrap();
        let filter: Filter = serde_json::from_value(
            json!({"type": "text", "column": 1, "match": "contains", "term": "5%!_"}),
        )
        .unwrap();
        let p = predicate(&filter, &columns(), &DuckDbDialect).unwrap();
        let df = reader
            .execute_sql(&format!(
                "SELECT name FROM (VALUES ('a5%!_b'), ('a5x!_b'), ('a5%!xb')) t(name) WHERE {p}"
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
                "ORDER BY \"n\" ASC",
            ]
        );
    }

    #[test]
    fn test_to_sql_quotes_for_the_session_dialect() {
        let filters = filters(json!([
            {"type": "text", "column": 1, "match": "contains", "term": "x"}
        ]));
        for scheme in ["mysql", "bigquery", "clickhouse", "databricks"] {
            let dialect = dialect_for_scheme(scheme).unwrap();
            let sql = to_sql(&filters, &BY_N, "SELECT 1", &columns(), dialect).join("\n");
            assert!(
                sql.contains("LOWER(`name`) LIKE LOWER('%x%') ESCAPE '!'"),
                "{scheme}: {sql}"
            );
            assert!(sql.contains("ORDER BY `n` ASC"), "{scheme}: {sql}");
            assert!(!sql.contains("\"name\""), "{scheme}: {sql}");
        }
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
                "ORDER BY \"n\" ASC",
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
