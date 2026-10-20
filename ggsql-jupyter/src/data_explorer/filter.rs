//! Row filters, sort keys, schema search and Convert to Code, as SQL.
//!
//! Everything here builds from the protocol's JSON and the cached schema, with
//! no database, so it is tested as plain string assembly.

use super::ColumnInfo;
use ggsql::naming::{quote_ident, quote_literal};
use ggsql::reader::SqlDialect;
use serde_json::Value;

/// What a database can do beyond the SQL every backend shares.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// `regexp_matches`, for the `regex_match` search type. DuckDB only.
    pub regex: bool,
}

/// The SQL predicate for one row filter.
///
/// Every value arrives as a string, so each is checked against its column's
/// type before it reaches the SQL: a number must parse as one, and anything
/// else becomes a quoted literal.
pub fn predicate(
    filter: &Value,
    columns: &[ColumnInfo],
    dialect: &dyn SqlDialect,
    caps: Capabilities,
) -> Result<String, String> {
    let index = filter["column_schema"]["column_index"]
        .as_u64()
        .ok_or("The filter names no column")? as usize;
    let column = columns
        .get(index)
        .ok_or_else(|| format!("There is no column {index}"))?;
    let c = quote_ident(&column.name);
    let params = &filter["params"];
    let lit = |v: &Value| -> Result<String, String> {
        let value = v.as_str().ok_or("The filter is missing a value")?;
        literal(value, &column.type_display, dialect)
    };

    match filter["filter_type"].as_str().unwrap_or("") {
        "compare" => {
            let op = match params["op"].as_str().unwrap_or("") {
                "=" => "=",
                "!=" => "<>",
                "<" => "<",
                "<=" => "<=",
                ">" => ">",
                ">=" => ">=",
                other => return Err(format!("Unknown comparison {other:?}")),
            };
            Ok(format!("{c} {op} {}", lit(&params["value"])?))
        }
        kind @ ("between" | "not_between") => {
            let between = format!(
                "{c} BETWEEN {} AND {}",
                lit(&params["left_value"])?,
                lit(&params["right_value"])?
            );
            Ok(if kind == "between" {
                between
            } else {
                format!("NOT ({between})")
            })
        }
        "is_null" => Ok(format!("{c} IS NULL")),
        "not_null" => Ok(format!("{c} IS NOT NULL")),
        "is_empty" => Ok(format!("{c} = ''")),
        "not_empty" => Ok(format!("{c} <> ''")),
        "is_true" => Ok(format!("{c} = {}", dialect.sql_boolean_literal(true))),
        "is_false" => Ok(format!("{c} = {}", dialect.sql_boolean_literal(false))),
        "search" => search(&c, params, caps),
        "set_membership" => {
            let inclusive = params["inclusive"].as_bool().unwrap_or(true);
            let values = params["values"].as_array().cloned().unwrap_or_default();
            if values.is_empty() {
                // Nothing selected: keep nothing, or exclude nothing.
                return Ok(if inclusive { "1 = 0" } else { "1 = 1" }.to_string());
            }
            let list = values.iter().map(lit).collect::<Result<Vec<_>, _>>()?;
            let not = if inclusive { "" } else { "NOT " };
            Ok(format!("{c} {not}IN ({})", list.join(", ")))
        }
        other => Err(format!("Unsupported filter type {other:?}")),
    }
}

/// A filter value as a SQL literal of its column's type.
fn literal(value: &str, display: &str, dialect: &dyn SqlDialect) -> Result<String, String> {
    match display {
        "integer" | "floating" | "decimal" => {
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

/// The predicate for a text search.
fn search(c: &str, params: &Value, caps: Capabilities) -> Result<String, String> {
    let term = params["term"].as_str().ok_or("The search has no term")?;
    let case_sensitive = params["case_sensitive"].as_bool().unwrap_or(false);
    let kind = params["search_type"].as_str().unwrap_or("");

    if kind == "regex_match" {
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
    let (pattern, negate) = match kind {
        "contains" => (format!("%{escaped}%"), false),
        "not_contains" => (format!("%{escaped}%"), true),
        "starts_with" => (format!("{escaped}%"), false),
        "ends_with" => (format!("%{escaped}"), false),
        other => return Err(format!("Unknown search type {other:?}")),
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

/// The `ORDER BY` clause for the sort keys, with a leading space, or nothing
/// when unsorted.
///
/// `tiebreak` orders rows the keys leave equal, so paging through a sorted
/// view is stable; without it, rows with equal keys may move between pages.
pub fn order_by(
    sort_keys: &[Value],
    columns: &[ColumnInfo],
    tiebreak: Option<&str>,
) -> Result<String, String> {
    let mut parts = Vec::new();
    for key in sort_keys {
        let index = key["column_index"]
            .as_u64()
            .ok_or("A sort key names no column")? as usize;
        let column = columns
            .get(index)
            .ok_or_else(|| format!("There is no column {index}"))?;
        let direction = if key["ascending"].as_bool().unwrap_or(true) {
            "ASC"
        } else {
            "DESC"
        };
        parts.push(format!("{} {direction}", quote_ident(&column.name)));
    }
    if parts.is_empty() {
        return Ok(String::new());
    }
    parts.extend(tiebreak.map(str::to_string));
    Ok(format!(" ORDER BY {}", parts.join(", ")))
}

/// The indices of the columns matching every column filter, in `sort_order`.
pub fn search_schema(params: &Value, columns: &[ColumnInfo]) -> Vec<usize> {
    let filters = params["filters"].as_array().cloned().unwrap_or_default();
    let mut matches: Vec<usize> = (0..columns.len())
        .filter(|&i| filters.iter().all(|f| column_matches(f, &columns[i])))
        .collect();

    match params["sort_order"].as_str().unwrap_or("original") {
        "ascending_name" => matches.sort_by_key(|&i| columns[i].name.to_lowercase()),
        "descending_name" => {
            matches.sort_by_key(|&i| std::cmp::Reverse(columns[i].name.to_lowercase()))
        }
        "ascending_type" => matches.sort_by_key(|&i| columns[i].type_display.clone()),
        "descending_type" => {
            matches.sort_by_key(|&i| std::cmp::Reverse(columns[i].type_display.clone()))
        }
        _ => {}
    }
    matches
}

/// Whether a column passes one column filter.
fn column_matches(filter: &Value, column: &ColumnInfo) -> bool {
    let params = &filter["params"];
    match filter["filter_type"].as_str().unwrap_or("") {
        "text_search" => {
            let term = params["term"].as_str().unwrap_or("");
            let case_sensitive = params["case_sensitive"].as_bool().unwrap_or(false);
            let (name, term) = if case_sensitive {
                (column.name.clone(), term.to_string())
            } else {
                (column.name.to_lowercase(), term.to_lowercase())
            };
            match params["search_type"].as_str().unwrap_or("contains") {
                "contains" => name.contains(&term),
                "not_contains" => !name.contains(&term),
                "starts_with" => name.starts_with(&term),
                "ends_with" => name.ends_with(&term),
                "regex_match" => regex::RegexBuilder::new(&term)
                    .case_insensitive(!case_sensitive)
                    .build()
                    .is_ok_and(|re| re.is_match(&column.name)),
                _ => true,
            }
        }
        "match_data_types" => params["display_types"]
            .as_array()
            .is_some_and(|types| types.iter().any(|t| t == column.type_display.as_str())),
        _ => true,
    }
}

/// SQL reproducing the explorer's view: `from` filtered and sorted.
///
/// A filter that cannot be expressed is left out with a comment saying so,
/// rather than failing the whole conversion.
pub fn convert_to_code(
    params: &Value,
    from: &str,
    columns: &[ColumnInfo],
    dialect: &dyn SqlDialect,
    caps: Capabilities,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut predicates = Vec::new();
    for filter in params["row_filters"].as_array().into_iter().flatten() {
        if filter["is_valid"] == Value::Bool(false) {
            continue;
        }
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
    let sort_keys = params["sort_keys"].as_array().cloned().unwrap_or_default();
    match order_by(&sort_keys, columns, None) {
        Ok(order) if !order.is_empty() => lines.push(order.trim_start().to_string()),
        Ok(_) => {}
        Err(e) => lines.push(format!("-- Sort left out: {e}")),
    }
    lines
}

/// A query as a derived table, for wrapping in the explorer's filters.
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

    fn columns() -> Vec<ColumnInfo> {
        [("n", "integer"), ("name", "string"), ("ok", "boolean")]
            .iter()
            .map(|(name, display)| ColumnInfo {
                name: name.to_string(),
                type_name: String::new(),
                type_display: display.to_string(),
            })
            .collect()
    }

    fn filter(column_index: usize, filter_type: &str, params: Value) -> Value {
        json!({
            "filter_id": "f",
            "filter_type": filter_type,
            "column_schema": {"column_index": column_index},
            "condition": "and",
            "params": params
        })
    }

    fn sql(f: &Value) -> Result<String, String> {
        predicate(f, &columns(), &AnsiDialect, CAPS)
    }

    #[test]
    fn test_compare_maps_not_equal_and_checks_numbers() {
        assert_eq!(
            sql(&filter(0, "compare", json!({"op": "!=", "value": "3"}))).unwrap(),
            "\"n\" <> 3"
        );
        assert!(sql(&filter(
            0,
            "compare",
            json!({"op": "=", "value": "1; DROP"})
        ))
        .is_err());
        assert!(sql(&filter(0, "compare", json!({"op": "=", "value": "inf"}))).is_err());
    }

    #[test]
    fn test_string_values_are_quoted() {
        assert_eq!(
            sql(&filter(
                1,
                "compare",
                json!({"op": "=", "value": "O'Brien"})
            ))
            .unwrap(),
            "\"name\" = 'O''Brien'"
        );
    }

    #[test]
    fn test_not_between_negates_between() {
        assert_eq!(
            sql(&filter(
                0,
                "not_between",
                json!({"left_value": "1", "right_value": "5"})
            ))
            .unwrap(),
            "NOT (\"n\" BETWEEN 1 AND 5)"
        );
    }

    #[test]
    fn test_search_escapes_wildcards_and_folds_case() {
        assert_eq!(
            sql(&filter(
                1,
                "search",
                json!({"search_type": "contains", "term": "50%_", "case_sensitive": false})
            ))
            .unwrap(),
            "LOWER(\"name\") LIKE LOWER('%50\\%\\_%') ESCAPE '\\'"
        );
        assert_eq!(
            sql(&filter(
                1,
                "search",
                json!({"search_type": "starts_with", "term": "a", "case_sensitive": true})
            ))
            .unwrap(),
            "\"name\" LIKE 'a%' ESCAPE '\\'"
        );
    }

    #[test]
    fn test_regex_needs_the_capability() {
        let f = filter(
            1,
            "search",
            json!({"search_type": "regex_match", "term": "^a", "case_sensitive": false}),
        );
        assert_eq!(sql(&f).unwrap(), "regexp_matches(\"name\", '^a', 'i')");
        assert!(predicate(&f, &columns(), &AnsiDialect, Capabilities { regex: false }).is_err());
    }

    #[test]
    fn test_set_membership() {
        assert_eq!(
            sql(&filter(
                1,
                "set_membership",
                json!({"values": ["a", "b"], "inclusive": false})
            ))
            .unwrap(),
            "\"name\" NOT IN ('a', 'b')"
        );
        assert_eq!(
            sql(&filter(
                1,
                "set_membership",
                json!({"values": [], "inclusive": true})
            ))
            .unwrap(),
            "1 = 0"
        );
    }

    #[test]
    fn test_booleans_use_the_dialect() {
        assert_eq!(
            sql(&filter(2, "is_true", json!(null))).unwrap(),
            format!("\"ok\" = {}", AnsiDialect.sql_boolean_literal(true))
        );
    }

    #[test]
    fn test_order_by_appends_the_tiebreak_only_when_sorted() {
        let keys = [json!({"column_index": 1, "ascending": false})];
        assert_eq!(
            order_by(&keys, &columns(), Some("rowid")).unwrap(),
            " ORDER BY \"name\" DESC, rowid"
        );
        assert_eq!(order_by(&[], &columns(), Some("rowid")).unwrap(), "");
    }

    #[test]
    fn test_search_schema_filters_and_sorts() {
        let params = json!({
            "filters": [{"filter_type": "text_search",
                         "params": {"search_type": "contains", "term": "N", "case_sensitive": false}}],
            "sort_order": "descending_name"
        });
        assert_eq!(search_schema(&params, &columns()), vec![1, 0]);

        let params = json!({
            "filters": [{"filter_type": "match_data_types",
                         "params": {"display_types": ["boolean"]}}],
            "sort_order": "original"
        });
        assert_eq!(search_schema(&params, &columns()), vec![2]);
    }

    #[test]
    fn test_convert_to_code_wraps_the_query() {
        let params = json!({
            "row_filters": [
                filter(0, "compare", json!({"op": ">", "value": "2"})),
                filter(1, "is_null", json!(null)),
            ],
            "sort_keys": [{"column_index": 0, "ascending": true}],
            "column_filters": [],
            "code_syntax_name": {"code_syntax_name": "SQL"}
        });
        let from = subquery("SELECT *\nFROM t;");
        assert_eq!(
            convert_to_code(&params, &from, &columns(), &AnsiDialect, CAPS),
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
    fn test_convert_to_code_leaves_out_what_it_cannot_express() {
        let params = json!({
            "row_filters": [filter(
                1,
                "search",
                json!({"search_type": "regex_match", "term": "x", "case_sensitive": true})
            )],
            "sort_keys": []
        });
        let lines = convert_to_code(
            &params,
            "\"t\"",
            &columns(),
            &AnsiDialect,
            Capabilities { regex: false },
        );
        assert!(lines[0].starts_with("-- Filter left out"), "{lines:?}");
        assert_eq!(&lines[1..], ["SELECT *", "FROM \"t\""]);
    }
}
