//! Query splitter using tree-sitter
//!
//! Splits ggsql queries into SQL and visualization portions, and injects
//! SELECT * FROM <source> when VISUALISE FROM is used.

use crate::{GgsqlError, Result};

use super::SourceTree;

/// Split a ggsql query into SQL and visualization portions using an existing SourceTree
///
/// Returns (sql_part, viz_part) where:
/// - sql_part: SQL to execute (may be injected with SELECT * FROM if VISUALISE FROM is present)
/// - viz_part: Everything from first "VISUALISE/VISUALIZE" onwards (may contain multiple VISUALISE statements)
///
/// If VISUALISE FROM <source> is used, this function will inject "SELECT * FROM <source>"
/// into the SQL portion, handling semicolons correctly.
pub fn split_from_tree(source_tree: &SourceTree) -> Result<(String, String)> {
    let root = source_tree.root();

    // Check if tree-sitter found any VISUALISE statements
    let has_visualise_statement = source_tree
        .find_node(&root, "(visualise_statement) @viz")
        .is_some();

    // If there's no VISUALISE statement, check if query contains VISUALISE FROM
    // This catches malformed queries like "CREATE TABLE x VISUALISE FROM x" (no semicolon)
    if !has_visualise_statement {
        let query_upper = source_tree.source.to_uppercase();
        if query_upper.contains("VISUALISE FROM") || query_upper.contains("VISUALIZE FROM") {
            return Err(GgsqlError::ParseError(
                "Error parsing VISUALISE statement. Did you forget a semicolon?".to_string(),
            ));
        }
        // No VISUALISE at all - treat entire query as SQL
        return Ok((source_tree.source.to_string(), String::new()));
    }

    // Find the first VISUALISE statement to determine split point
    // Use byte offset instead of node boundaries to handle parse errors in SQL portion
    let first_viz_start = source_tree
        .find_node(&root, "(visualise_statement) @viz")
        .map(|node| node.start_byte());

    let (sql_text, viz_text) = if let Some(viz_start) = first_viz_start {
        // Split at the first VISUALISE keyword
        let sql_part = &source_tree.source[..viz_start];
        let viz_part = &source_tree.source[viz_start..];
        (sql_part.trim().to_string(), viz_part.trim().to_string())
    } else {
        // No VISUALISE statement found (shouldn't happen due to earlier check)
        (source_tree.source.to_string(), String::new())
    };

    // Check if any VISUALISE statement has FROM clause and inject SELECT if needed
    let from_query = r#"
        (visualise_statement
          (from_clause
            (table_ref) @table))
    "#;

    let modified_sql = if let Some(from_identifier) = source_tree.find_text(&root, from_query) {
        // Inject SELECT * FROM <source>
        if sql_text.trim().is_empty() {
            // No SQL yet - just add SELECT
            format!("SELECT * FROM {}", from_identifier)
        } else {
            let trimmed = sql_text.trim();
            format!("{} SELECT * FROM {}", trimmed, from_identifier)
        }
    } else {
        sql_text
    };

    Ok((modified_sql, viz_text))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let query = "SELECT * FROM data VISUALISE  DRAW point MAPPING x AS x, y AS y";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("VISUALISE "));
        assert!(viz.contains("DRAW point"));
    }

    #[test]
    fn test_case_insensitive() {
        let query = "SELECT * FROM data visualise x, y DRAW point";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("visualise x, y"));
    }

    #[test]
    fn test_no_visualise() {
        let query = "SELECT * FROM data WHERE x > 5";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        assert_eq!(sql, query);
        assert!(viz.is_empty());
    }

    #[test]
    fn test_visualise_from_no_sql() {
        let query = "VISUALISE FROM mtcars  DRAW point MAPPING mpg AS x, hp AS y";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should inject SELECT * FROM mtcars
        assert_eq!(sql, "SELECT * FROM mtcars");
        assert!(viz.starts_with("VISUALISE FROM mtcars"));
    }

    #[test]
    fn test_visualise_from_with_cte() {
        let query =
            "WITH cte AS (SELECT * FROM x) VISUALISE FROM cte DRAW point MAPPING a AS x, b AS y";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should inject SELECT * FROM cte after the WITH
        assert!(sql.contains("WITH cte AS (SELECT * FROM x)"));
        assert!(sql.contains("SELECT * FROM cte"));
        assert!(viz.starts_with("VISUALISE FROM cte"));
    }

    #[test]
    fn test_visualise_from_after_create() {
        let query = "CREATE TABLE x AS SELECT 1; VISUALISE FROM x";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        assert!(sql.contains("CREATE TABLE x AS SELECT 1;"));
        assert!(sql.contains("SELECT * FROM x"));
        assert!(viz.starts_with("VISUALISE FROM x"));

        // Without semicolon, the visualise statement should also be recognised
        let query = "CREATE TABLE x AS SELECT 1 VISUALISE FROM x";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        assert!(sql.contains("CREATE TABLE x AS SELECT 1"));
        assert!(sql.contains("SELECT * FROM x"));
        assert!(viz.starts_with("VISUALISE FROM x"));
    }

    #[test]
    fn test_visualise_from_after_insert_absorbed() {
        // The grammar's permissive INSERT rule absorbs VISUALISE as SQL tokens
        // This is a known limitation - without a semicolon, the INSERT consumes everything
        let query = "INSERT INTO x VALUES (1) VISUALISE FROM x DRAW";
        let result = split_from_tree(&SourceTree::new(query).unwrap());

        // The SQL used to absorb visualise. We don't want this to happen again.
        assert!(result.is_ok());
        let (sql, viz) = result.unwrap();
        assert!(sql.contains("INSERT"));
        assert!(viz.contains("DRAW"));
    }

    #[test]
    fn test_visualise_as_no_injection() {
        let query = "SELECT * FROM x VISUALISE DRAW point MAPPING a AS x, b AS y";
        let (sql, _viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should NOT inject anything - just split normally
        assert_eq!(sql, "SELECT * FROM x");
        assert!(!sql.contains("SELECT * FROM SELECT")); // Make sure we didn't double-inject
    }

    #[test]
    fn test_visualise_from_file_path() {
        let query = "VISUALISE FROM 'mtcars.csv'  DRAW point MAPPING mpg AS x, hp AS y";
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should inject SELECT * FROM 'mtcars.csv' with quotes preserved
        assert_eq!(sql, "SELECT * FROM 'mtcars.csv'");
        assert!(viz.starts_with("VISUALISE FROM 'mtcars.csv'"));
    }

    #[test]
    fn test_visualise_from_file_path_double_quotes() {
        let query =
            r#"VISUALISE FROM "data/sales.parquet"  DRAW bar MAPPING region AS x, total AS y"#;
        let (sql, viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should inject SELECT * FROM "data/sales.parquet" with quotes preserved
        assert_eq!(sql, r#"SELECT * FROM "data/sales.parquet""#);
        assert!(viz.starts_with(r#"VISUALISE FROM "data/sales.parquet""#));
    }

    #[test]
    fn test_visualise_from_file_path_with_cte() {
        let query = "WITH prep AS (SELECT * FROM 'raw.csv' WHERE year = 2024) VISUALISE FROM prep  DRAW line MAPPING date AS x, value AS y";
        let (sql, _viz) = split_from_tree(&SourceTree::new(query).unwrap()).unwrap();

        // Should inject SELECT * FROM prep after WITH
        assert!(sql.contains("WITH prep AS"));
        assert!(sql.contains("SELECT * FROM prep"));
        // The file path inside the CTE should remain as-is (part of the WITH clause)
        assert!(sql.contains("'raw.csv'"));
    }
}
