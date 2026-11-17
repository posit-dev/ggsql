//! Row filters and sort keys over a result's columns, and the SQL for them.
//!
//! Everything here builds from the request and the result's schema, with no
//! database, so it is tested as plain string assembly.

use super::Column;
use ggsql::naming::{quote_ident, quote_literal};
use ggsql::reader::SqlDialect;
use serde::{Deserialize, Serialize};

/// What a database can do beyond the SQL every backend shares.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// `regexp_matches`, for regular-expression text filters. DuckDB only.
    pub regex: bool,
}

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
    caps: Capabilities,
) -> Result<String, String> {
    let column = column_of(columns, filter.column())?;
    let c = quote_ident(&column.name);
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
        } => text(&c, *how, term, *case_sensitive, caps),
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
    caps: Capabilities,
) -> Result<String, String> {
    if how == TextMatch::Regex {
        if !caps.regex {
            return Err("This database does not support regular-expression search".into());
        }
        let flags = if case_sensitive { "" } else { ", 'i'" };
        return Ok(format!(
            "regexp_matches({c}, {}{flags})",
            quote_literal(term)
        ));
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
    Ok(format!("{lhs} {not}LIKE {rhs} ESCAPE '\\'"))
}

/// Escape `LIKE`'s wildcards, so a search term matches literally.
fn escape_like(term: &str) -> String {
    term.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// The sort keys as `ORDER BY` terms, without the keyword.
pub fn order_terms(keys: &[SortKey], columns: &[Column]) -> Result<Vec<String>, String> {
    keys.iter()
        .map(|key| {
            let column = column_of(columns, key.column)?;
            let direction = if key.ascending { "ASC" } else { "DESC" };
            Ok(format!("{} {direction}", quote_ident(&column.name)))
        })
        .collect()
}

/// SQL reproducing a view of a result: `from` filtered and sorted.
///
/// A filter that cannot be expressed in `dialect` is left out with a comment
/// saying so, rather than failing the whole conversion.
pub fn to_sql(
    filters: &[Filter],
    sort: &[SortKey],
    from: &str,
    columns: &[Column],
    dialect: &dyn SqlDialect,
    caps: Capabilities,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut predicates = Vec::new();
    for filter in filters {
        match predicate(filter, columns, dialect, caps) {
            Ok(p) => predicates.push(p),
            Err(e) => lines.push(format!("-- Filter left out: {e}")),
        }
    }

    lines.push("SELECT *".to_string());
    lines.extend(format!("FROM {from}").lines().map(str::to_string));
    for (i, p) in predicates.iter().enumerate() {
        let keyword = if i == 0 { "WHERE" } else { "  AND" };
        lines.push(format!("{keyword} {p}"));
    }
    match order_terms(sort, columns) {
        Ok(terms) if !terms.is_empty() => lines.push(format!("ORDER BY {}", terms.join(", "))),
        Ok(_) => {}
        Err(e) => lines.push(format!("-- Sort left out: {e}")),
    }
    lines
}

/// A query as a derived table, for wrapping in a view's filters.
pub fn subquery(query: &str) -> String {
    let query = query.trim().trim_end_matches(';').trim_end();
    let body: Vec<String> = query.lines().map(|l| format!("    {l}")).collect();
    format!("(\n{}\n) AS {}", body.join("\n"), quote_ident("result"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggsql::reader::AnsiDialect;
    use serde_json::json;

    const CAPS: Capabilities = Capabilities { regex: true };

    fn columns() -> Vec<Column> {
        [("n", "integer"), ("name", "string"), ("ok", "boolean")]
            .iter()
            .map(|(name, kind)| Column {
                name: name.to_string(),
                type_name: String::new(),
                kind: kind.to_string(),
            })
            .collect()
    }

    fn sql(f: Value) -> Result<String, String> {
        let filter: Filter = serde_json::from_value(f).unwrap();
        predicate(&filter, &columns(), &AnsiDialect, CAPS)
    }

    use serde_json::Value;

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
            sql(json!({"type": "text", "column": 1, "match": "contains", "term": "50%_"})).unwrap(),
            "LOWER(\"name\") LIKE LOWER('%50\\%\\_%') ESCAPE '\\'"
        );
        assert_eq!(
            sql(
                json!({"type": "text", "column": 1, "match": "starts_with", "term": "a",
                       "case_sensitive": true})
            )
            .unwrap(),
            "\"name\" LIKE 'a%' ESCAPE '\\'"
        );
    }

    #[test]
    fn test_regex_needs_the_capability() {
        let f: Filter = serde_json::from_value(
            json!({"type": "text", "column": 1, "match": "regex", "term": "^a"}),
        )
        .unwrap();
        assert_eq!(
            predicate(&f, &columns(), &AnsiDialect, CAPS).unwrap(),
            "regexp_matches(\"name\", '^a', 'i')"
        );
        assert!(predicate(&f, &columns(), &AnsiDialect, Capabilities { regex: false }).is_err());
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
            &columns()
        )
        .is_err());
    }

    #[test]
    fn test_to_sql_wraps_the_query() {
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "compare", "column": 0, "op": ">", "value": "2"},
            {"type": "is_null", "column": 1}
        ]))
        .unwrap();
        let sort = [SortKey {
            column: 0,
            ascending: true,
        }];
        let from = subquery("SELECT *\nFROM t;");
        assert_eq!(
            to_sql(&filters, &sort, &from, &columns(), &AnsiDialect, CAPS),
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
    fn test_to_sql_leaves_out_what_it_cannot_express() {
        let filters: Vec<Filter> = serde_json::from_value(json!([
            {"type": "text", "column": 1, "match": "regex", "term": "x", "case_sensitive": true}
        ]))
        .unwrap();
        let lines = to_sql(
            &filters,
            &[],
            "\"t\"",
            &columns(),
            &AnsiDialect,
            Capabilities { regex: false },
        );
        assert!(lines[0].starts_with("-- Filter left out"), "{lines:?}");
        assert_eq!(&lines[1..], ["SELECT *", "FROM \"t\""]);
    }
}
