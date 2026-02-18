//! CTE (Common Table Expression) extraction, transformation, and materialization.
//!
//! This module handles extracting CTE definitions from SQL using tree-sitter,
//! materializing them as temporary tables, and transforming CTE references
//! in SQL queries.

use crate::reader::Reader;
use crate::{naming, parser::SourceTree, GgsqlError, Result};
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
pub fn materialize_ctes(ctes: &[CteDefinition], reader: &dyn Reader) -> Result<HashSet<String>> {
    let mut materialized = HashSet::new();

    for cte in ctes {
        // Transform the CTE body to replace references to earlier CTEs
        let transformed_body = transform_cte_references(&cte.body, &materialized);

        let temp_table_name = naming::cte_table(&cte.name);

        // Execute the CTE body SQL to get a DataFrame, then register it
        let df = reader.execute_sql(&transformed_body).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to materialize CTE '{}': {}", cte.name, e))
        })?;
        reader.register(&temp_table_name, df, true).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to register CTE '{}': {}", cte.name, e))
        })?;

        materialized.insert(cte.name.clone());
    }

    Ok(materialized)
}

/// Split a WITH...SELECT query into its CTE prefix and trailing SELECT.
///
/// Given SQL like `WITH a AS (...), b AS (...) SELECT * FROM a`, returns:
/// - CTE prefix: `"WITH a AS (...), b AS (...)"`
/// - Trailing SELECT: `"SELECT * FROM a"`
///
/// Returns `None` if the query is not a WITH statement, has no trailing SELECT,
/// or parsing fails.
pub fn split_with_query(source_tree: &SourceTree) -> Option<(String, String)> {
    let root = source_tree.root();
    let with_node = source_tree.find_node(&root, "(with_statement) @with")?;

    let mut cursor = with_node.walk();
    let mut last_cte_end: Option<usize> = None;
    let mut select_node = None;
    let mut seen_cte = false;

    for child in with_node.children(&mut cursor) {
        match child.kind() {
            "cte_definition" => {
                seen_cte = true;
                last_cte_end = Some(child.end_byte());
            }
            "select_statement" if seen_cte => {
                select_node = Some(child);
                break;
            }
            _ => {}
        }
    }

    let cte_prefix = source_tree.source[with_node.start_byte()..last_cte_end?].to_string();
    let trailing_select = source_tree.get_text(&select_node?);
    Some((cte_prefix, trailing_select))
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
    // Try to extract trailing SELECT (WITH...SELECT or direct SELECT)
    let select_sql = split_with_query(source_tree)
        .map(|(_, select)| select)
        .or_else(|| {
            // Fallback: direct SELECT statement (no WITH clause)
            let root = source_tree.root();
            source_tree.find_text(&root, "(sql_statement (select_statement) @select)")
        });

    if let Some(select_sql) = select_sql {
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
    if split_with_query(source_tree).is_some() {
        return true;
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

    #[test]
    fn test_split_with_query_basic() {
        let sql = "WITH cte AS (SELECT * FROM x) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT * FROM x)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_multiple_ctes() {
        let sql = "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a JOIN b";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH a AS (SELECT 1), b AS (SELECT 2)");
        assert_eq!(select, "SELECT * FROM a JOIN b");
    }

    #[test]
    fn test_split_with_query_nested_subquery() {
        let sql = "WITH cte AS (SELECT * FROM (SELECT 1)) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT * FROM (SELECT 1))");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_string_with_select_keyword() {
        let sql = "WITH cte AS (SELECT 'SELECT' AS col) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT 'SELECT' AS col)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_string_with_parens() {
        let sql = "WITH cte AS (SELECT '()' AS col) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT '()' AS col)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_not_a_with() {
        let sql = "SELECT * FROM x";
        let source_tree = SourceTree::new(sql).unwrap();
        assert!(split_with_query(&source_tree).is_none());
    }

    #[test]
    fn test_split_with_query_no_trailing_select() {
        let sql = "WITH cte AS (SELECT 1) VISUALISE DRAW point";
        let source_tree = SourceTree::new(sql).unwrap();
        assert!(split_with_query(&source_tree).is_none());
    }

    #[test]
    fn test_split_with_query_stat_transform_output() {
        // Realistic stat transform output (histogram pattern)
        let sql = "WITH __stat_src__ AS (SELECT x FROM data), \
                   __binned__ AS (SELECT x, COUNT(*) AS count FROM __stat_src__ GROUP BY x) \
                   SELECT *, count * 1.0 / SUM(count) OVER () AS density FROM __binned__";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert!(prefix.starts_with("WITH __stat_src__"));
        assert!(prefix.contains("__binned__"));
        assert!(prefix.ends_with(")"));
        assert!(select.starts_with("SELECT *"));
        assert!(select.contains("density"));
    }
}
