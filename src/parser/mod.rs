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
assert_eq!(specs[0].layers[0].geom, Geom::line());
# Ok(())
# }
```
*/

use crate::{Plot, Result};

pub mod builder;
pub mod source_tree;
pub mod splitter;

pub use builder::build_ast;
pub use source_tree::SourceTree;
pub use splitter::split_from_tree;

/// Main entry point for parsing ggsql queries
///
/// Takes a complete ggsql query (SQL + VISUALISE) and returns a vector of
/// parsed specifications (one per VISUALISE statement).
pub fn parse_query(query: &str) -> Result<Vec<Plot>> {
    // Parse the full query and create SourceTree (single parse!)
    let source_tree = SourceTree::new(query)?;

    // Validate the parse tree has no errors
    source_tree.validate()?;

    // Build AST from the source tree
    let specs = builder::build_ast(&source_tree)?;

    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;
    use crate::{AestheticValue, DataSource, Geom};

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
        assert_eq!(specs[0].layers[0].geom, Geom::point());
    }

    #[test]
    fn test_sql_extraction() {
        let query = r#"
            SELECT date, revenue FROM sales WHERE year = 2024
            VISUALISE date AS x, revenue AS y
            DRAW line
        "#;

        let source_tree = SourceTree::new(query).unwrap();
        let (sql, _) = splitter::split_from_tree(&source_tree).unwrap();
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
        assert_eq!(specs[0].layers[0].geom, Geom::line());
        assert_eq!(specs[0].layers[1].geom, Geom::point());

        // Second layer should have y and color
        assert_eq!(specs[0].layers[1].mappings.len(), 2);
        assert!(matches!(
            specs[0].layers[1].mappings.get("color"),
            Some(AestheticValue::Literal(ParameterValue::String(s))) if s == "value"
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
        assert_eq!(specs[0].layers[0].geom, Geom::tile());
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
        assert_eq!(specs[0].layers[0].geom, Geom::line());
        assert_eq!(specs[1].layers[0].geom, Geom::tile());
        assert_eq!(specs[2].layers[0].geom, Geom::bar());
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
        let source_tree = SourceTree::new(query);
        if let Err(ref e) = source_tree {
            eprintln!("Parse error: {}", e);
        }

        // Print the tree
        if let Ok(ref st) = source_tree {
            let root = st.root();
            eprintln!("Root kind: {}", root.kind());
            eprintln!("Has error: {}", root.has_error());
            eprintln!("Tree: {}", root.to_sexp());
        }

        assert!(
            source_tree.is_ok(),
            "Failed to parse VALUES subquery: {:?}",
            source_tree
        );

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
