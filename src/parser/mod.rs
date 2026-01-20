/*!
ggsql Parser Module

Handles splitting ggsql queries into SQL and visualization portions, then parsing
the visualization specification into a typed AST.

## Architecture

1. **Query Splitting**: Use tree-sitter with external scanner to reliably split
   SQL from VISUALISE portions, handling edge cases like strings and comments.

2. **AST Building**: Convert tree-sitter concrete syntax tree (CST) into a
   typed abstract syntax tree (AST) representing the visualization specification.

3. **Validation**: Perform syntactic validation during parsing, with semantic
   validation deferred to execution time when data is available.

## Example Usage

```rust
# use ggsql::parser::parse_query;
# use ggsql::Geom;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let query = r#"
    SELECT date, revenue, region FROM sales WHERE year = 2024
    VISUALISE date AS x, revenue AS y, region AS color
    DRAW line
    LABEL
        title => 'Sales by Region'
"#;

let specs = parse_query(query)?;
assert_eq!(specs.len(), 1);
assert_eq!(specs[0].layers.len(), 1);
assert_eq!(specs[0].layers[0].geom, Geom::Line);
# Ok(())
# }
```
*/

use crate::{GgsqlError, Result};
use tree_sitter::Tree;

pub mod ast;
pub mod builder;
pub mod error;
pub mod splitter;

// Re-export key types
pub use ast::*;
pub use error::ParseError;
pub use splitter::split_query;

/// Main entry point for parsing ggsql queries
///
/// Takes a complete ggsql query (SQL + VISUALISE) and returns a vector of
/// parsed specifications (one per VISUALISE statement).
pub fn parse_query(query: &str) -> Result<Vec<VizSpec>> {
    // Parse the full query using tree-sitter (includes SQL + VISUALISE portions)
    let tree = parse_full_query(query)?;

    // Build AST from the tree-sitter parse tree
    let specs = builder::build_ast(&tree, query)?;

    Ok(specs)
}

/// Parse the full ggsql query (SQL + VISUALISE) using tree-sitter
fn parse_full_query(query: &str) -> Result<Tree> {
    let mut parser = tree_sitter::Parser::new();

    // Set the tree-sitter-ggsql language
    parser
        .set_language(&tree_sitter_ggsql::language())
        .map_err(|e| GgsqlError::ParseError(format!("Failed to set language: {}", e)))?;

    // Parse the full query (SQL + VISUALISE portions together)
    let tree = parser
        .parse(query, None)
        .ok_or_else(|| GgsqlError::ParseError("Failed to parse query".to_string()))?;

    // Check for parse errors
    if tree.root_node().has_error() {
        return Err(GgsqlError::ParseError(
            "Parse tree contains errors".to_string(),
        ));
    }

    Ok(tree)
}

/// Extract just the SQL portion from a ggsql query
pub fn extract_sql(query: &str) -> Result<String> {
    let (sql_part, _) = splitter::split_query(query)?;
    Ok(sql_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query_parsing() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE x, y
            DRAW point
        "#;

        let result = parse_query(query);
        assert!(result.is_ok(), "Failed to parse simple query: {:?}", result);

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);
        assert_eq!(specs[0].layers[0].geom, Geom::Point);
    }

    #[test]
    fn test_sql_extraction() {
        let query = r#"
            SELECT date, revenue FROM sales WHERE year = 2024
            VISUALISE date AS x, revenue AS y
            DRAW line
        "#;

        let sql = extract_sql(query).unwrap();
        assert!(sql.contains("SELECT date, revenue FROM sales"));
        assert!(sql.contains("WHERE year = 2024"));
        assert!(!sql.contains("VISUALISE"));
    }

    #[test]
    fn test_multi_layer_query() {
        let query = r#"
            SELECT x, y, z FROM data
            VISUALISE x, y
            DRAW line
            DRAW point MAPPING z AS y, 'value' AS color
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 2);
        // First layer is line, second layer is point
        assert_eq!(specs[0].layers[0].geom, Geom::Line);
        assert_eq!(specs[0].layers[1].geom, Geom::Point);

        // Second layer should have y and color
        assert_eq!(specs[0].layers[1].mappings.len(), 2);
        assert!(matches!(
            specs[0].layers[1].mappings.get("color"),
            Some(AestheticValue::Literal(LiteralValue::String(s))) if s == "value"
        ));
    }

    #[test]
    fn test_multiple_visualizations() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE x, y
            DRAW point
            VISUALIZE
            DRAW bar MAPPING x AS x, y AS y
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].layers.len(), 1);
        assert_eq!(specs[1].layers.len(), 1);
    }

    #[test]
    fn test_american_spelling() {
        let query = r#"
            SELECT x, y FROM data
            VISUALIZE x, y
            DRAW tile
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers[0].geom, Geom::Tile);
    }

    #[test]
    fn test_three_visualizations() {
        let query = r#"
            SELECT x, y, z FROM data
            VISUALISE x, y
            DRAW point
            VISUALIZE
            DRAW bar MAPPING x AS x, y AS y
            VISUALISE z AS x, y AS y
            DRAW tile
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].layers.len(), 1);
        assert_eq!(specs[1].layers.len(), 1);
        assert_eq!(specs[2].layers.len(), 1);
    }

    #[test]
    fn test_empty_visualise() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert!(specs[0].global_mappings.is_empty());
    }

    #[test]
    fn test_multiple_viz_with_different_clauses() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE x, y
            DRAW point
            LABEL title => 'Scatter Plot'
            THEME minimal
            VISUALIZE
            DRAW bar MAPPING x AS x, y AS y
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 2);

        // First viz should have layers, labels, and theme
        assert_eq!(specs[0].layers.len(), 1);
        assert!(specs[0].labels.is_some());
        assert!(specs[0].theme.is_some());

        // Second viz should have layer but no labels/theme
        assert_eq!(specs[1].layers.len(), 1);
        assert!(specs[1].labels.is_none());
        assert!(specs[1].theme.is_none());
    }

    #[test]
    fn test_mixed_spelling_multiple_viz() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE x, y
            DRAW line
            VISUALIZE
            DRAW tile MAPPING x AS x, y AS y
            VISUALISE
            DRAW bar MAPPING x AS x, y AS y
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].layers[0].geom, Geom::Line);
        assert_eq!(specs[1].layers[0].geom, Geom::Tile);
        assert_eq!(specs[2].layers[0].geom, Geom::Bar);
    }

    #[test]
    fn test_complex_multi_viz_query() {
        let query = r#"
            SELECT date, revenue, cost FROM sales
            WHERE year >= 2023
            VISUALISE date AS x, revenue AS y
            DRAW line
            DRAW line MAPPING cost AS y
            SCALE x SETTING type => 'date'
            LABEL title => 'Revenue and Cost Trends'
            THEME minimal
            VISUALIZE
            DRAW bar MAPPING date AS x, revenue AS y
            VISUALISE
            DRAW tile MAPPING date AS x, revenue AS y
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 3);

        // Plot with 2 layers, scale, labels, theme
        assert_eq!(specs[0].layers.len(), 2);
        assert_eq!(specs[0].scales.len(), 1);
        assert!(specs[0].labels.is_some());
        assert!(specs[0].theme.is_some());

        // Second viz with 1 layer
        assert_eq!(specs[1].layers.len(), 1);

        // Third viz with 1 layer
        assert_eq!(specs[2].layers.len(), 1);
    }

    #[test]
    fn test_values_subquery() {
        let query = "SELECT * FROM (VALUES (1, 2)) AS t(x, y) VISUALISE x, y DRAW point";

        // First check if tree-sitter can parse it
        let tree = parse_full_query(query);
        if let Err(ref e) = tree {
            eprintln!("Parse error: {}", e);
        }

        // Print the tree
        if let Ok(ref t) = tree {
            let root = t.root_node();
            eprintln!("Root kind: {}", root.kind());
            eprintln!("Has error: {}", root.has_error());
            eprintln!("Tree: {}", root.to_sexp());
        }

        assert!(tree.is_ok(), "Failed to parse VALUES subquery: {:?}", tree);

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
    }

    #[test]
    fn test_wildcard_global_mapping() {
        let query = r#"
            SELECT x, y FROM data
            VISUALISE *
            DRAW point
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert!(specs[0].global_mappings.wildcard);
        assert!(specs[0].global_mappings.aesthetics.is_empty());
    }

    #[test]
    fn test_explicit_global_mapping() {
        let query = r#"
            VISUALISE date AS x, revenue AS y
            DRAW line
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        let mapping = &specs[0].global_mappings;
        assert!(!mapping.wildcard);
        assert_eq!(mapping.aesthetics.len(), 2);
        assert!(mapping.aesthetics.contains_key("x"));
        assert!(mapping.aesthetics.contains_key("y"));
        assert_eq!(
            mapping.aesthetics.get("x").unwrap().column_name(),
            Some("date")
        );
        assert_eq!(
            mapping.aesthetics.get("y").unwrap().column_name(),
            Some("revenue")
        );
    }

    #[test]
    fn test_implicit_global_mapping() {
        let query = r#"
            VISUALISE x, y
            DRAW point
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        let mapping = &specs[0].global_mappings;
        assert!(!mapping.wildcard);
        assert_eq!(mapping.aesthetics.len(), 2);
        // Implicit mappings are resolved at parse time: x -> x, y -> y
        assert_eq!(
            mapping.aesthetics.get("x").unwrap().column_name(),
            Some("x")
        );
        assert_eq!(
            mapping.aesthetics.get("y").unwrap().column_name(),
            Some("y")
        );
    }

    #[test]
    fn test_mixed_global_mapping() {
        let query = r#"
            VISUALISE x, y, region AS color
            DRAW point
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        let mapping = &specs[0].global_mappings;
        assert!(!mapping.wildcard);
        assert_eq!(mapping.aesthetics.len(), 3);
        // Implicit x and y, explicit color
        assert_eq!(
            mapping.aesthetics.get("x").unwrap().column_name(),
            Some("x")
        );
        assert_eq!(
            mapping.aesthetics.get("y").unwrap().column_name(),
            Some("y")
        );
        assert_eq!(
            mapping.aesthetics.get("color").unwrap().column_name(),
            Some("region")
        );
    }

    #[test]
    fn test_visualise_from_table() {
        let query = r#"
            VISUALISE x, y FROM sales
            DRAW bar
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("sales".to_string()))
        );
    }

    #[test]
    fn test_visualise_from_cte() {
        let query = r#"
            WITH cte AS (SELECT * FROM data)
            VISUALISE x, y FROM cte
            DRAW point
        "#;

        let specs = parse_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("cte".to_string()))
        );
    }
}
