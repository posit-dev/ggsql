//! CTE (Common Table Expression) extraction, transformation, and materialization.
//!
//! This module handles extracting CTE definitions from SQL using tree-sitter,
//! materializing them as temporary tables, and transforming CTE references
//! in SQL queries.

use crate::{naming, DataFrame, GgsqlError, Result};
use std::collections::HashSet;
use tree_sitter::{Node, Parser};

/// Extracted CTE (Common Table Expression) definition
#[derive(Debug, Clone)]
pub struct CteDefinition {
    /// Name of the CTE
    pub name: String,
    /// Full SQL text of the CTE body (including the SELECT statement inside)
    pub body: String,
}

/// Extract CTE definitions from SQL using tree-sitter
///
/// Parses the SQL and extracts all CTE definitions from WITH clauses.
/// Returns CTEs in declaration order (important for dependency resolution).
pub fn extract_ctes(sql: &str) -> Vec<CteDefinition> {
    let mut ctes = Vec::new();

    // Parse with tree-sitter
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        return ctes;
    }

    let tree = match parser.parse(sql, None) {
        Some(t) => t,
        None => return ctes,
    };

    let root = tree.root_node();

    // Walk the tree looking for WITH statements
    extract_ctes_from_node(&root, sql, &mut ctes);

    ctes
}

/// Recursively extract CTEs from a node and its children
fn extract_ctes_from_node(node: &Node, source: &str, ctes: &mut Vec<CteDefinition>) {
    // Check if this is a with_statement
    if node.kind() == "with_statement" {
        // Find all cte_definition children (in declaration order)
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "cte_definition" {
                if let Some(cte) = parse_cte_definition(&child, source) {
                    ctes.push(cte);
                }
            }
        }
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_ctes_from_node(&child, source, ctes);
    }
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

/// Extract the trailing SELECT statement from a WITH clause
///
/// Given SQL like `WITH a AS (...), b AS (...) SELECT * FROM a`, extracts
/// just the `SELECT * FROM a` part. Returns None if there's no trailing SELECT.
pub fn extract_trailing_select(sql: &str) -> Option<String> {
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        return None;
    }

    let tree = parser.parse(sql, None)?;
    let root = tree.root_node();

    // Find sql_portion → sql_statement → with_statement → select_statement
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "sql_portion" {
            let mut sql_cursor = child.walk();
            for sql_child in child.children(&mut sql_cursor) {
                if sql_child.kind() == "sql_statement" {
                    let mut stmt_cursor = sql_child.walk();
                    for stmt_child in sql_child.children(&mut stmt_cursor) {
                        if stmt_child.kind() == "with_statement" {
                            // Find trailing select_statement in with_statement
                            let mut with_cursor = stmt_child.walk();
                            let mut seen_cte = false;
                            for with_child in stmt_child.children(&mut with_cursor) {
                                if with_child.kind() == "cte_definition" {
                                    seen_cte = true;
                                } else if with_child.kind() == "select_statement" && seen_cte {
                                    // This is the trailing SELECT
                                    return Some(get_node_text(&with_child, sql).to_string());
                                }
                            }
                        } else if stmt_child.kind() == "select_statement" {
                            // Direct SELECT (no WITH clause)
                            return Some(get_node_text(&stmt_child, sql).to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Transform global SQL for execution with temp tables
///
/// If the SQL has a WITH clause followed by SELECT, extracts just the SELECT
/// portion and transforms CTE references to temp table names.
/// For SQL without WITH clause, just transforms any CTE references.
pub fn transform_global_sql(sql: &str, materialized_ctes: &HashSet<String>) -> Option<String> {
    // Try to extract trailing SELECT from WITH clause
    if let Some(trailing_select) = extract_trailing_select(sql) {
        // Transform CTE references in the SELECT
        Some(transform_cte_references(
            &trailing_select,
            materialized_ctes,
        ))
    } else if has_executable_sql(sql) {
        // No WITH clause but has executable SQL - just transform references
        Some(transform_cte_references(sql, materialized_ctes))
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
pub fn has_executable_sql(sql: &str) -> bool {
    // Parse with tree-sitter to check for executable statements
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        // If we can't parse, assume it's executable (fail safely)
        return true;
    }

    let tree = match parser.parse(sql, None) {
        Some(t) => t,
        None => return true, // Assume executable if parse fails
    };

    let root = tree.root_node();

    // Look for sql_portion which should contain actual SQL statements
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "sql_portion" {
            // Check if sql_portion contains actual statement nodes
            let mut sql_cursor = child.walk();
            for sql_child in child.children(&mut sql_cursor) {
                if sql_child.kind() == "sql_statement" {
                    // Check if this is a WITH-only statement (no trailing SELECT)
                    let mut stmt_cursor = sql_child.walk();
                    for stmt_child in sql_child.children(&mut stmt_cursor) {
                        match stmt_child.kind() {
                            "select_statement" | "create_statement" | "insert_statement"
                            | "update_statement" | "delete_statement" => return true,
                            "with_statement" => {
                                // Check if WITH has trailing SELECT
                                if with_has_trailing_select(&stmt_child) {
                                    return true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
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
        let ctes = extract_ctes(sql);

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
        let ctes = extract_ctes(sql);

        assert_eq!(ctes.len(), 2);
        // Verify order is preserved
        assert_eq!(ctes[0].name, "sales");
        assert_eq!(ctes[1].name, "targets");
    }

    #[test]
    fn test_extract_ctes_none() {
        let sql = "SELECT * FROM sales WHERE year = 2024";
        let ctes = extract_ctes(sql);

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
