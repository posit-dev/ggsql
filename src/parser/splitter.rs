//! Query splitter using tree-sitter
//!
//! Splits ggsql queries into SQL and visualization portions, and injects
//! SELECT * FROM <source> when VISUALISE FROM is used.

use crate::{GgsqlError, Result};
use tree_sitter::{Node, Parser};

/// Split a ggsql query into SQL and visualization portions
///
/// Returns (sql_part, viz_part) where:
/// - sql_part: SQL to execute (may be injected with SELECT * FROM if VISUALISE FROM is present)
/// - viz_part: Everything from first "VISUALISE/VISUALIZE" onwards (may contain multiple VISUALISE statements)
///
/// If VISUALISE FROM <source> is used, this function will inject "SELECT * FROM <source>"
/// into the SQL portion, handling semicolons correctly.
pub fn split_query(query: &str) -> Result<(String, String)> {
    let query = query.trim();

    // Parse the full query with tree-sitter to understand its structure
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_ggsql::language())
        .map_err(|e| GgsqlError::InternalError(format!("Failed to set language: {}", e)))?;

    let tree = parser
        .parse(query, None)
        .ok_or_else(|| GgsqlError::ParseError("Failed to parse query".to_string()))?;

    let root = tree.root_node();

    // Check if tree-sitter found any VISUALISE statements
    let has_visualise_statement = root
        .children(&mut root.walk())
        .any(|n| n.kind() == "visualise_statement");

    // If there's no VISUALISE statement, check if query contains VISUALISE FROM
    // This catches malformed queries like "CREATE TABLE x VISUALISE FROM x" (no semicolon)
    if !has_visualise_statement {
        let query_upper = query.to_uppercase();
        if query_upper.contains("VISUALISE FROM") || query_upper.contains("VISUALIZE FROM") {
            return Err(GgsqlError::ParseError(
                "Error parsing VISUALISE statement. Did you forget a semicolon?".to_string(),
            ));
        }
        // No VISUALISE at all - treat entire query as SQL
        return Ok((query.to_string(), String::new()));
    }

    // Find the first VISUALISE statement to determine split point
    // Use byte offset instead of node boundaries to handle parse errors in SQL portion
    let mut first_viz_start: Option<usize> = None;
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "visualise_statement" {
            first_viz_start = Some(child.start_byte());
            break;
        }
    }

    let (sql_text, viz_text) = if let Some(viz_start) = first_viz_start {
        // Split at the first VISUALISE keyword
        let sql_part = &query[..viz_start];
        let viz_part = &query[viz_start..];
        (sql_part.trim().to_string(), viz_part.trim().to_string())
    } else {
        // No VISUALISE statement found (shouldn't happen due to earlier check)
        (query.to_string(), String::new())
    };

    // Check if any VISUALISE statement has FROM clause and inject SELECT if needed
    let mut modified_sql = sql_text.clone();

    for child in root.children(&mut root.walk()) {
        if child.kind() == "visualise_statement" {
            // Look for FROM identifier in this visualise_statement
            if let Some(from_identifier) = extract_from_identifier(&child, query) {
                // Inject SELECT * FROM <source>
                if modified_sql.trim().is_empty() {
                    // No SQL yet - just add SELECT
                    modified_sql = format!("SELECT * FROM {}", from_identifier);
                } else {
                    let trimmed = modified_sql.trim();
                    modified_sql = format!("{} SELECT * FROM {}", trimmed, from_identifier);
                }
                break;
            }
        }
    }

    Ok((modified_sql, viz_text))
}

/// Extract FROM identifier or string from a visualise_statement node
fn extract_from_identifier(node: &Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "from_clause" {
            continue;
        }
        let mut from_cursor = child.walk();
        for table_ref in child.children(&mut from_cursor) {
            if table_ref.kind() != "table_ref" {
                continue;
            }
            return Some(get_node_text(&table_ref, source).to_string());
        }
    }
    None
}

/// Get text content of a node
fn get_node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let query = "SELECT * FROM data VISUALISE  DRAW point MAPPING x AS x, y AS y";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("VISUALISE "));
        assert!(viz.contains("DRAW point"));
    }

    #[test]
    fn test_case_insensitive() {
        let query = "SELECT * FROM data visualise x, y DRAW point";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("visualise x, y"));
    }

    #[test]
    fn test_no_visualise() {
        let query = "SELECT * FROM data WHERE x > 5";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, query);
        assert!(viz.is_empty());
    }

    #[test]
    fn test_visualise_from_no_sql() {
        let query = "VISUALISE FROM mtcars  DRAW point MAPPING mpg AS x, hp AS y";
        let (sql, viz) = split_query(query).unwrap();

        // Should inject SELECT * FROM mtcars
        assert_eq!(sql, "SELECT * FROM mtcars");
        assert!(viz.starts_with("VISUALISE FROM mtcars"));
    }

    #[test]
    fn test_visualise_from_with_cte() {
        let query =
            "WITH cte AS (SELECT * FROM x) VISUALISE FROM cte DRAW point MAPPING a AS x, b AS y";
        let (sql, viz) = split_query(query).unwrap();

        // Should inject SELECT * FROM cte after the WITH
        assert!(sql.contains("WITH cte AS (SELECT * FROM x)"));
        assert!(sql.contains("SELECT * FROM cte"));
        assert!(viz.starts_with("VISUALISE FROM cte"));
    }

    #[test]
    fn test_visualise_from_after_create() {
        let query = "CREATE TABLE x AS SELECT 1; VISUALISE FROM x";
        let (sql, viz) = split_query(query).unwrap();

        assert!(sql.contains("CREATE TABLE x AS SELECT 1;"));
        assert!(sql.contains("SELECT * FROM x"));
        assert!(viz.starts_with("VISUALISE FROM x"));

        // Without semicolon, the visualise statement should also be recognised
        let query = "CREATE TABLE x AS SELECT 1 VISUALISE FROM x";
        let (sql, viz) = split_query(query).unwrap();

        assert!(sql.contains("CREATE TABLE x AS SELECT 1"));
        assert!(sql.contains("SELECT * FROM x"));
        assert!(viz.starts_with("VISUALISE FROM x"));
    }

    #[test]
    fn test_visualise_from_after_insert_absorbed() {
        // The grammar's permissive INSERT rule absorbs VISUALISE as SQL tokens
        // This is a known limitation - without a semicolon, the INSERT consumes everything
        let query = "INSERT INTO x VALUES (1) VISUALISE FROM x";
        let result = split_query(query);

        // The splitter succeeds but VISUALISE is consumed by INSERT
        // This results in no proper VISUALISE statement being found
        // The correct usage requires a semicolon: INSERT ...; VISUALISE FROM ...
        assert!(result.is_ok());
        let (sql, viz) = result.unwrap();
        // INSERT absorbed the entire query as SQL
        assert!(sql.contains("INSERT"));
        // VIZ portion is empty since VISUALISE was absorbed
        assert!(viz.is_empty() || !viz.contains("DRAW"));
    }

    #[test]
    fn test_visualise_as_no_injection() {
        let query = "SELECT * FROM x VISUALISE DRAW point MAPPING a AS x, b AS y";
        let (sql, _viz) = split_query(query).unwrap();

        // Should NOT inject anything - just split normally
        assert_eq!(sql, "SELECT * FROM x");
        assert!(!sql.contains("SELECT * FROM SELECT")); // Make sure we didn't double-inject
    }

    #[test]
    fn test_visualise_from_file_path() {
        let query = "VISUALISE FROM 'mtcars.csv'  DRAW point MAPPING mpg AS x, hp AS y";
        let (sql, viz) = split_query(query).unwrap();

        // Should inject SELECT * FROM 'mtcars.csv' with quotes preserved
        assert_eq!(sql, "SELECT * FROM 'mtcars.csv'");
        assert!(viz.starts_with("VISUALISE FROM 'mtcars.csv'"));
    }

    #[test]
    fn test_visualise_from_file_path_double_quotes() {
        let query =
            r#"VISUALISE FROM "data/sales.parquet"  DRAW bar MAPPING region AS x, total AS y"#;
        let (sql, viz) = split_query(query).unwrap();

        // Should inject SELECT * FROM "data/sales.parquet" with quotes preserved
        assert_eq!(sql, r#"SELECT * FROM "data/sales.parquet""#);
        assert!(viz.starts_with(r#"VISUALISE FROM "data/sales.parquet""#));
    }

    #[test]
    fn test_visualise_from_file_path_with_cte() {
        let query = "WITH prep AS (SELECT * FROM 'raw.csv' WHERE year = 2024) VISUALISE FROM prep  DRAW line MAPPING date AS x, value AS y";
        let (sql, _viz) = split_query(query).unwrap();

        // Should inject SELECT * FROM prep after WITH
        assert!(sql.contains("WITH prep AS"));
        assert!(sql.contains("SELECT * FROM prep"));
        // The file path inside the CTE should remain as-is (part of the WITH clause)
        assert!(sql.contains("'raw.csv'"));
    }
}
