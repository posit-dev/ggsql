//! CTE (Common Table Expression) extraction, transformation, and materialization.
//!
//! This module handles extracting CTE definitions from SQL using tree-sitter,
//! materializing them as temporary tables, and transforming CTE references
//! in SQL queries.

use crate::{naming, parser::SourceTree, DataFrame, GgsqlError, Result};
use std::collections::HashSet;
use tree_sitter::Node;

/// Extracted CTE (Common Table Expression) definition
#[derive(Debug, Clone)]
pub struct CteDefinition {
    /// Name of the CTE
    pub name: String,
    /// Full SQL text of the CTE body (including the SELECT statement inside)
    pub body: String,
}

/// Extract CTE definitions from the source tree
///
/// Extracts all CTE definitions from WITH clauses using the existing parse tree.
/// Returns CTEs in declaration order (important for dependency resolution).
pub fn extract_ctes(source_tree: &SourceTree) -> Vec<CteDefinition> {
    let root = source_tree.root();

    // Use declarative tree-sitter query to find all CTE definitions
    source_tree
        .find_nodes(&root, "(cte_definition) @cte")
        .into_iter()
        .filter_map(|node| parse_cte_definition(&node, source_tree.source))
        .collect()
}

/// Parse a single CTE definition node into a CteDefinition
fn parse_cte_definition(node: &Node, source: &str) -> Option<CteDefinition> {
    let mut name: Option<String> = None;
    let mut body_start: Option<usize> = None;
    let mut body_end: Option<usize> = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                name = Some(get_node_text(&child, source).to_string());
            }
            "select_statement" => {
                // The SELECT inside the CTE
                body_start = Some(child.start_byte());
                body_end = Some(child.end_byte());
            }
            _ => {}
        }
    }

    match (name, body_start, body_end) {
        (Some(n), Some(start), Some(end)) => {
            let body = source[start..end].to_string();
            Some(CteDefinition { name: n, body })
        }
        _ => None,
    }
}

/// Get text content of a node
pub(crate) fn get_node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

/// Transform CTE references in SQL to use temp table names
///
/// Replaces references to CTEs (e.g., `FROM sales`, `JOIN sales`) with
/// the corresponding temp table names (e.g., `FROM __ggsql_cte_sales__`).
///
/// This handles table references after FROM and JOIN keywords, being careful
/// to only replace whole word matches (not substrings).
pub fn transform_cte_references(sql: &str, cte_names: &HashSet<String>) -> String {
    if cte_names.is_empty() {
        return sql.to_string();
    }

    let mut result = sql.to_string();

    for cte_name in cte_names {
        let temp_table_name = naming::cte_table(cte_name);

        // Replace table references: FROM cte_name, JOIN cte_name, cte_name.column
        // Use word boundary matching to avoid replacing substrings
        // Pattern: (FROM|JOIN)\s+<cte_name>(\s|,|)|$)
        let patterns = [
            // FROM cte_name (case insensitive)
            (
                format!(r"(?i)(\bFROM\s+){}(\s|,|\)|$)", regex::escape(cte_name)),
                format!("${{1}}{}${{2}}", temp_table_name),
            ),
            // JOIN cte_name (case insensitive) - handles LEFT JOIN, RIGHT JOIN, etc.
            (
                format!(r"(?i)(\bJOIN\s+){}(\s|,|\)|$)", regex::escape(cte_name)),
                format!("${{1}}{}${{2}}", temp_table_name),
            ),
            // Qualified column references: cte_name.column (case insensitive)
            (
                format!(
                    r"(?i)\b{}(\.[a-zA-Z_][a-zA-Z0-9_]*)",
                    regex::escape(cte_name)
                ),
                format!("{}${{1}}", temp_table_name),
            ),
        ];

        for (pattern, replacement) in patterns {
            if let Ok(re) = regex::Regex::new(&pattern) {
                result = re.replace_all(&result, replacement.as_str()).to_string();
            }
        }
    }

    result
}

/// Materialize CTEs as temporary tables in the database
///
/// Creates a temp table for each CTE in declaration order. When a CTE
/// references an earlier CTE, the reference is transformed to use the
/// temp table name.
///
/// Returns the set of CTE names that were materialized.
pub fn materialize_ctes<F>(ctes: &[CteDefinition], execute_sql: &F) -> Result<HashSet<String>>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    let mut materialized = HashSet::new();

    for cte in ctes {
        // Transform the CTE body to replace references to earlier CTEs
        let transformed_body = transform_cte_references(&cte.body, &materialized);

        let temp_table_name = naming::cte_table(&cte.name);
        let create_sql = format!(
            "CREATE OR REPLACE TEMP TABLE {} AS {}",
            temp_table_name, transformed_body
        );

        execute_sql(&create_sql).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to materialize CTE '{}': {}", cte.name, e))
        })?;

        materialized.insert(cte.name.clone());
    }

    Ok(materialized)
}

/// Extract the trailing SELECT statement from a WITH clause using existing tree
///
/// Given SQL like `WITH a AS (...), b AS (...) SELECT * FROM a`, extracts
/// just the `SELECT * FROM a` part. Returns None if there's no trailing SELECT.
pub fn extract_trailing_select(source_tree: &SourceTree) -> Option<String> {
    let root = source_tree.root();

    // Try to find WITH statement first
    if let Some(with_node) = source_tree.find_node(&root, "(with_statement) @with") {
        // Look for the trailing SELECT that comes AFTER cte_definition nodes
        let mut cursor = with_node.walk();
        let mut seen_cte = false;
        for child in with_node.children(&mut cursor) {
            if child.kind() == "cte_definition" {
                seen_cte = true;
            } else if child.kind() == "select_statement" && seen_cte {
                // This is the trailing SELECT after CTEs
                return Some(source_tree.get_text(&child));
            }
        }
    }

    // Otherwise, look for direct SELECT statement (no WITH clause)
    source_tree.find_text(&root, "(sql_statement (select_statement) @select)")
}

/// Transform global SQL for execution with temp tables
///
/// If the SQL has a WITH clause followed by SELECT, extracts just the SELECT
/// portion and transforms CTE references to temp table names.
/// For SQL without WITH clause, just transforms any CTE references.
pub fn transform_global_sql(
    source_tree: &SourceTree,
    materialized_ctes: &HashSet<String>,
) -> Option<String> {
    // Try to extract SELECT (handles both WITH...SELECT and direct SELECT)
    if let Some(select_sql) = extract_trailing_select(source_tree) {
        // Transform CTE references in the SELECT
        Some(transform_cte_references(&select_sql, materialized_ctes))
    } else if has_executable_sql(source_tree) {
        // Non-SELECT executable SQL (CREATE, INSERT, UPDATE, DELETE)
        // OR VISUALISE FROM (which injects SELECT * FROM <source>)
        // Extract SQL (with injection if VISUALISE FROM) and transform CTE references
        let sql = source_tree.extract_sql()?;
        Some(transform_cte_references(&sql, materialized_ctes))
    } else {
        // No executable SQL (just CTEs)
        None
    }
}

/// Check if SQL contains executable statements (SELECT, INSERT, UPDATE, DELETE, CREATE)
///
/// Returns false if the SQL is just CTE definitions without a trailing statement.
/// This handles cases like `WITH a AS (...), b AS (...) VISUALISE` where the WITH
/// clause has no trailing SELECT - these CTEs are still extracted for layer use
/// but shouldn't be executed as global data.
pub fn has_executable_sql(source_tree: &SourceTree) -> bool {
    let root = source_tree.root();

    // Check for direct executable statements (SELECT, CREATE, INSERT, UPDATE, DELETE)
    let direct_statements = r#"
        (sql_statement
          [(select_statement)
           (create_statement)
           (insert_statement)
           (update_statement)
           (delete_statement)] @stmt)
    "#;

    if source_tree.find_node(&root, direct_statements).is_some() {
        return true;
    }

    // Check for WITH statements that have trailing SELECT
    let with_statements = source_tree.find_nodes(&root, "(with_statement) @with");
    for with_node in with_statements {
        if with_has_trailing_select(&with_node) {
            return true;
        }
    }

    // Check for VISUALISE FROM (which injects SELECT * FROM <source>)
    let visualise_from = r#"
        (visualise_statement
          (from_clause) @from)
    "#;
    if source_tree.find_node(&root, visualise_from).is_some() {
        return true;
    }

    false
}

/// Check if a with_statement node has a trailing SELECT (after CTEs)
fn with_has_trailing_select(with_node: &Node) -> bool {
    let mut cursor = with_node.walk();
    let mut seen_cte = false;

    for child in with_node.children(&mut cursor) {
        if child.kind() == "cte_definition" {
            seen_cte = true;
        } else if child.kind() == "select_statement" && seen_cte {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ctes_single() {
        let sql = "WITH sales AS (SELECT * FROM raw_sales) SELECT * FROM sales";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].name, "sales");
        assert!(ctes[0].body.contains("SELECT * FROM raw_sales"));
    }

    #[test]
    fn test_extract_ctes_multiple() {
        let sql = "WITH
            sales AS (SELECT * FROM raw_sales),
            targets AS (SELECT * FROM goals)
        SELECT * FROM sales";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 2);
        // Verify order is preserved
        assert_eq!(ctes[0].name, "sales");
        assert_eq!(ctes[1].name, "targets");
    }

    #[test]
    fn test_extract_ctes_none() {
        let sql = "SELECT * FROM sales WHERE year = 2024";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert!(ctes.is_empty());
    }

    #[test]
    fn test_transform_cte_references() {
        // Test cases: (sql, cte_names, expected_contains, exact_match)
        let test_cases: Vec<(
            &str,
            Vec<&str>,
            Vec<&str>,    // strings that should be in result
            Option<&str>, // exact match (if result should equal this)
        )> = vec![
            // Single CTE reference
            (
                "SELECT * FROM sales WHERE year = 2024",
                vec!["sales"],
                vec!["FROM __ggsql_cte_sales_", "__ WHERE year = 2024"],
                None,
            ),
            // Multiple CTE references with qualified columns
            (
                "SELECT sales.date, targets.revenue FROM sales JOIN targets ON sales.id = targets.id",
                vec!["sales", "targets"],
                vec![
                    "FROM __ggsql_cte_sales_",
                    "JOIN __ggsql_cte_targets_",
                    "__ggsql_cte_sales_",  // qualified reference sales.date
                    "__ggsql_cte_targets_", // qualified reference targets.revenue
                ],
                None,
            ),
            // Qualified column references only (no FROM/JOIN transformation needed)
            (
                "WHERE sales.date > '2024-01-01' AND sales.revenue > 100",
                vec!["sales"],
                vec!["__ggsql_cte_sales_"],
                None,
            ),
            // No matching CTE (unchanged)
            (
                "SELECT * FROM other_table",
                vec!["sales"],
                vec![],
                Some("SELECT * FROM other_table"),
            ),
            // Empty CTE names (unchanged)
            (
                "SELECT * FROM sales",
                vec![],
                vec![],
                Some("SELECT * FROM sales"),
            ),
            // No false positives on substrings (wholesale should not match 'sales')
            (
                "SELECT wholesale.date FROM wholesale",
                vec!["sales"],
                vec![],
                Some("SELECT wholesale.date FROM wholesale"),
            ),
        ];

        for (sql, cte_names_vec, expected_contains, exact_match) in test_cases {
            let cte_names: HashSet<String> = cte_names_vec.iter().map(|s| s.to_string()).collect();
            let result = transform_cte_references(sql, &cte_names);

            if let Some(expected) = exact_match {
                assert_eq!(result, expected, "SQL '{}' should remain unchanged", sql);
            } else {
                for expected in &expected_contains {
                    assert!(
                        result.contains(expected),
                        "Result '{}' should contain '{}' for SQL '{}'",
                        result,
                        expected,
                        sql
                    );
                }
                // When CTEs are transformed, result should contain session UUID
                if !cte_names_vec.is_empty() {
                    assert!(
                        result.contains(naming::session_id()),
                        "Result should contain session UUID"
                    );
                }
            }
        }
    }
}
