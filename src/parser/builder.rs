//! AST builder - converts tree-sitter CST to typed AST
//!
//! Takes a tree-sitter parse tree and builds a typed VizSpec AST,
//! handling all the node types defined in the grammar.

use tree_sitter::{Tree, Node};
use crate::{VizqlError, Result};
use super::ast::*;
use std::collections::HashMap;

/// Build a VizSpec AST from a tree-sitter parse tree
pub fn build_ast(tree: &Tree, source: &str) -> Result<Vec<VizSpec>> {
    let root = tree.root_node();

    // For now, create a simple stub implementation
    // TODO: Implement full tree walking and AST building

    // Check if root is a query node
    if root.kind() != "query" {
        return Err(VizqlError::ParseError(format!(
            "Expected 'query' root node, got '{}'",
            root.kind()
        )));
    }

    let mut specs = Vec::new();

    // Walk through child nodes - each visualise_statement becomes a VizSpec
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "visualise_statement" {
            let spec = build_visualise_statement(&child, source)?;
            specs.push(spec);
        }
    }

    if specs.is_empty() {
        return Err(VizqlError::ParseError(
            "No VISUALISE statements found in query".to_string()
        ));
    }

    Ok(specs)
}

/// Build a single VizSpec from a visualise_statement node
fn build_visualise_statement(node: &Node, source: &str) -> Result<VizSpec> {
    let mut viz_type = VizType::Plot;
    let mut spec = VizSpec::new(VizType::Plot);

    // Walk through children of visualise_statement
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "VISUALISE" | "VISUALIZE" | "AS" => {
                // Skip keywords
                continue;
            }
            "viz_type" => {
                // Extract visualization type
                viz_type = parse_viz_type(&child, source)?;
                spec.viz_type = viz_type;
            }
            "viz_clause" => {
                // Process visualization clause
                process_viz_clause(&child, source, &mut spec)?;
            }
            _ => {
                // Unknown node type - skip for now
                continue;
            }
        }
    }

    Ok(spec)
}

/// Parse a viz_type node to determine the visualization type
fn parse_viz_type(node: &Node, source: &str) -> Result<VizType> {
    let text = get_node_text(node, source).to_uppercase();
    match text.as_str() {
        "PLOT" => Ok(VizType::Plot),
        "TABLE" => Ok(VizType::Table),
        "MAP" => Ok(VizType::Map),
        _ => Err(VizqlError::ParseError(format!("Unknown viz type: {}", text))),
    }
}

/// Process a visualization clause node
fn process_viz_clause(node: &Node, source: &str, spec: &mut VizSpec) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "with_clause" => {
                let layer = build_layer(&child, source)?;
                spec.layers.push(layer);
            }
            "scale_clause" => {
                let scale = build_scale(&child, source)?;
                spec.scales.push(scale);
            }
            "facet_clause" => {
                spec.facet = Some(build_facet(&child, source)?);
            }
            "coord_clause" => {
                spec.coord = Some(build_coord(&child, source)?);
            }
            "label_clause" => {
                let new_labels = build_labels(&child, source)?;
                // Merge with existing labels if any
                if let Some(ref mut existing_labels) = spec.labels {
                    for (key, value) in new_labels.labels {
                        existing_labels.labels.insert(key, value);
                    }
                } else {
                    spec.labels = Some(new_labels);
                }
            }
            "guide_clause" => {
                let guide = build_guide(&child, source)?;
                spec.guides.push(guide);
            }
            "theme_clause" => {
                spec.theme = Some(build_theme(&child, source)?);
            }
            _ => {
                // Unknown clause type
                continue;
            }
        }
    }

    Ok(())
}

/// Build a Layer from a with_clause node
fn build_layer(node: &Node, source: &str) -> Result<Layer> {
    // Stub implementation - create a basic point layer
    // TODO: Implement full layer building from tree-sitter nodes

    let layer = Layer::new(Geom::Point)
        .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
        .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));

    Ok(layer)
}

/// Build a Scale from a scale_clause node
fn build_scale(node: &Node, source: &str) -> Result<Scale> {
    // Stub implementation
    Ok(Scale {
        aesthetic: "x".to_string(),
        scale_type: Some(ScaleType::Linear),
        properties: HashMap::new(),
    })
}

/// Build a Facet from a facet_clause node
fn build_facet(node: &Node, source: &str) -> Result<Facet> {
    // Stub implementation
    Ok(Facet::Wrap {
        variables: vec!["category".to_string()],
        scales: FacetScales::Fixed,
    })
}

/// Build a Coord from a coord_clause node
fn build_coord(node: &Node, source: &str) -> Result<Coord> {
    // Stub implementation
    Ok(Coord {
        coord_type: CoordType::Cartesian,
        properties: HashMap::new(),
    })
}

/// Build Labels from a label_clause node
fn build_labels(node: &Node, source: &str) -> Result<Labels> {
    // Stub implementation
    let mut labels = HashMap::new();
    labels.insert("title".to_string(), "My Plot".to_string());
    Ok(Labels { labels })
}

/// Build a Guide from a guide_clause node
fn build_guide(node: &Node, source: &str) -> Result<Guide> {
    // Stub implementation
    Ok(Guide {
        aesthetic: "color".to_string(),
        guide_type: Some(GuideType::Legend),
        properties: HashMap::new(),
    })
}

/// Build a Theme from a theme_clause node
fn build_theme(node: &Node, source: &str) -> Result<Theme> {
    // Stub implementation
    Ok(Theme {
        style: Some("minimal".to_string()),
        properties: HashMap::new(),
    })
}

/// Get text content of a node
fn get_node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_implementation() {
        // This is just a placeholder test
        // Real tests will be added when the full implementation is done
        assert!(true);
    }
}