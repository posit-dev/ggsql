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
                let viz_type = parse_viz_type(&child, source)?;
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
    // Parse geom type
    let mut geom = Geom::Point; // default
    let mut aesthetics = std::collections::HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "WITH" | "USING" | "," => continue, // Skip keywords and punctuation
            "geom_type" => {
                let geom_text = get_node_text(&child, source);
                geom = parse_geom_type(&geom_text)?;
            }
            "aesthetic_mapping" => {
                let (aesthetic, value) = parse_aesthetic_mapping(&child, source)?;
                aesthetics.insert(aesthetic, value);
            }
            _ => {
                // Unknown node type
                eprintln!("Skipping unknown node type: {}", child.kind());
            }
        }
    }

    let mut layer = Layer::new(geom);
    for (aesthetic, value) in aesthetics {
        layer = layer.with_aesthetic(aesthetic, value);
    }

    Ok(layer)
}

/// Parse a geom_type node text into a Geom enum
fn parse_geom_type(text: &str) -> Result<Geom> {
    match text.to_lowercase().as_str() {
        "point" => Ok(Geom::Point),
        "line" => Ok(Geom::Line),
        "path" => Ok(Geom::Path),
        "bar" => Ok(Geom::Bar),
        "col" => Ok(Geom::Col),
        "area" => Ok(Geom::Area),
        "tile" => Ok(Geom::Tile),
        "polygon" => Ok(Geom::Polygon),
        "ribbon" => Ok(Geom::Ribbon),
        "histogram" => Ok(Geom::Histogram),
        "density" => Ok(Geom::Density),
        "smooth" => Ok(Geom::Smooth),
        "boxplot" => Ok(Geom::Boxplot),
        "violin" => Ok(Geom::Violin),
        "text" => Ok(Geom::Text),
        "label" => Ok(Geom::Label),
        "segment" => Ok(Geom::Segment),
        "arrow" => Ok(Geom::Arrow),
        "hline" => Ok(Geom::HLine),
        "vline" => Ok(Geom::VLine),
        "abline" => Ok(Geom::AbLine),
        "errorbar" => Ok(Geom::ErrorBar),
        _ => Err(VizqlError::ParseError(format!("Unknown geom type: {}", text))),
    }
}

/// Parse an aesthetic_mapping node into (aesthetic_name, aesthetic_value)
fn parse_aesthetic_mapping(node: &Node, source: &str) -> Result<(String, AestheticValue)> {
    let mut aesthetic_name = String::new();
    let mut aesthetic_value = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "aesthetic_name" => {
                aesthetic_name = get_node_text(&child, source);
            }
            "aesthetic_value" => {
                aesthetic_value = Some(parse_aesthetic_value(&child, source)?);
            }
            "=" => continue, // Skip equals sign
            _ => {}
        }
    }

    if aesthetic_name.is_empty() || aesthetic_value.is_none() {
        return Err(VizqlError::ParseError(format!(
            "Invalid aesthetic mapping: name='{}', value={:?}",
            aesthetic_name, aesthetic_value
        )));
    }

    Ok((aesthetic_name, aesthetic_value.unwrap()))
}

/// Parse an aesthetic_value node into an AestheticValue
fn parse_aesthetic_value(node: &Node, source: &str) -> Result<AestheticValue> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "column_reference" => {
                // Column reference is an identifier
                let col_name = get_node_text(&child, source);
                return Ok(AestheticValue::Column(col_name));
            }
            "literal_value" => {
                return parse_literal_value(&child, source);
            }
            _ => {}
        }
    }

    Err(VizqlError::ParseError(format!(
        "Could not parse aesthetic value from node: {}",
        node.kind()
    )))
}

/// Parse a literal_value node into an AestheticValue::Literal
fn parse_literal_value(node: &Node, source: &str) -> Result<AestheticValue> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string" => {
                let text = get_node_text(&child, source);
                let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                return Ok(AestheticValue::Literal(LiteralValue::String(unquoted.to_string())));
            }
            "number" => {
                let text = get_node_text(&child, source);
                let num = text.parse::<f64>().map_err(|e| {
                    VizqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
                })?;
                return Ok(AestheticValue::Literal(LiteralValue::Number(num)));
            }
            "boolean" => {
                let text = get_node_text(&child, source);
                let bool_val = text == "true";
                return Ok(AestheticValue::Literal(LiteralValue::Boolean(bool_val)));
            }
            _ => {}
        }
    }

    Err(VizqlError::ParseError(format!(
        "Could not parse literal value from node: {}",
        node.kind()
    )))
}

/// Build a Scale from a scale_clause node
fn build_scale(node: &Node, source: &str) -> Result<Scale> {
    let mut aesthetic = String::new();
    let mut scale_type: Option<ScaleType> = None;
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "SCALE" | "USING" | "," => continue, // Skip keywords
            "aesthetic_name" => {
                aesthetic = get_node_text(&child, source);
            }
            "scale_property" => {
                // Parse scale property: name = value
                let mut prop_cursor = child.walk();
                let mut prop_name = String::new();
                let mut prop_value: Option<ScalePropertyValue> = None;

                for prop_child in child.children(&mut prop_cursor) {
                    match prop_child.kind() {
                        "scale_property_name" => {
                            prop_name = get_node_text(&prop_child, source);
                        }
                        "scale_property_value" => {
                            prop_value = Some(parse_scale_property_value(&prop_child, source)?);
                        }
                        "=" => continue,
                        _ => {}
                    }
                }

                // If this is a 'type' property, set scale_type
                if prop_name == "type" {
                    if let Some(ScalePropertyValue::String(type_str)) = prop_value {
                        scale_type = Some(parse_scale_type(&type_str)?);
                    }
                } else if !prop_name.is_empty() && prop_value.is_some() {
                    properties.insert(prop_name, prop_value.unwrap());
                }
            }
            _ => {}
        }
    }

    if aesthetic.is_empty() {
        return Err(VizqlError::ParseError(
            "Scale clause missing aesthetic name".to_string(),
        ));
    }

    Ok(Scale {
        aesthetic,
        scale_type,
        properties,
    })
}

/// Parse scale type from text
fn parse_scale_type(text: &str) -> Result<ScaleType> {
    match text.to_lowercase().as_str() {
        "linear" => Ok(ScaleType::Linear),
        "log" | "log10" => Ok(ScaleType::Log),
        "sqrt" => Ok(ScaleType::Sqrt),
        "reverse" => Ok(ScaleType::Reverse),
        "categorical" => Ok(ScaleType::Categorical),
        "ordinal" => Ok(ScaleType::Ordinal),
        "date" => Ok(ScaleType::Date),
        "datetime" => Ok(ScaleType::DateTime),
        "viridis" => Ok(ScaleType::Viridis),
        "plasma" => Ok(ScaleType::Plasma),
        "diverging" => Ok(ScaleType::Diverging),
        _ => Err(VizqlError::ParseError(format!(
            "Unknown scale type: {}",
            text
        ))),
    }
}

/// Parse scale property value
fn parse_scale_property_value(node: &Node, source: &str) -> Result<ScalePropertyValue> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string" => {
                let text = get_node_text(&child, source);
                let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                return Ok(ScalePropertyValue::String(unquoted.to_string()));
            }
            "number" => {
                let text = get_node_text(&child, source);
                let num = text.parse::<f64>().map_err(|e| {
                    VizqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
                })?;
                return Ok(ScalePropertyValue::Number(num));
            }
            "boolean" => {
                let text = get_node_text(&child, source);
                let bool_val = text == "true";
                return Ok(ScalePropertyValue::Boolean(bool_val));
            }
            "array" => {
                // Parse array of values
                let mut values = Vec::new();
                let mut array_cursor = child.walk();
                for array_child in child.children(&mut array_cursor) {
                    match array_child.kind() {
                        "string" => {
                            let text = get_node_text(&array_child, source);
                            let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                            values.push(ArrayElement::String(unquoted.to_string()));
                        }
                        "number" => {
                            let text = get_node_text(&array_child, source);
                            if let Ok(num) = text.parse::<f64>() {
                                values.push(ArrayElement::Number(num));
                            }
                        }
                        "boolean" => {
                            let text = get_node_text(&array_child, source);
                            let bool_val = text == "true";
                            values.push(ArrayElement::Boolean(bool_val));
                        }
                        _ => continue,
                    }
                }
                return Ok(ScalePropertyValue::Array(values));
            }
            _ => {}
        }
    }

    Err(VizqlError::ParseError(format!(
        "Could not parse scale property value from node: {}",
        node.kind()
    )))
}

/// Build a Facet from a facet_clause node
fn build_facet(node: &Node, source: &str) -> Result<Facet> {
    let mut is_wrap = false;
    let mut row_vars = Vec::new();
    let mut col_vars = Vec::new();
    let mut scales = FacetScales::Fixed;

    let mut cursor = node.walk();
    let mut next_vars_are_cols = false;

    for child in node.children(&mut cursor) {
        match child.kind() {
            "FACET" | "USING" | "=" => continue,
            "WRAP" => {
                is_wrap = true;
            }
            "BY" => {
                next_vars_are_cols = true;
            }
            "facet_vars" => {
                // Parse list of variable names
                let vars = parse_facet_vars(&child, source)?;
                if is_wrap {
                    row_vars = vars;
                } else if next_vars_are_cols {
                    col_vars = vars;
                } else {
                    row_vars = vars;
                }
            }
            "facet_scales" => {
                scales = parse_facet_scales(&child, source)?;
            }
            _ => {}
        }
    }

    if is_wrap {
        Ok(Facet::Wrap {
            variables: row_vars,
            scales,
        })
    } else {
        Ok(Facet::Grid {
            rows: row_vars,
            cols: col_vars,
            scales,
        })
    }
}

/// Parse facet variables from a facet_vars node
fn parse_facet_vars(node: &Node, source: &str) -> Result<Vec<String>> {
    let mut vars = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                vars.push(get_node_text(&child, source));
            }
            "," => continue,
            _ => {}
        }
    }

    Ok(vars)
}

/// Parse facet scales from a facet_scales node
fn parse_facet_scales(node: &Node, source: &str) -> Result<FacetScales> {
    let text = get_node_text(node, source);
    match text.as_str() {
        "fixed" => Ok(FacetScales::Fixed),
        "free" => Ok(FacetScales::Free),
        "free_x" => Ok(FacetScales::FreeX),
        "free_y" => Ok(FacetScales::FreeY),
        _ => Err(VizqlError::ParseError(format!(
            "Unknown facet scales: {}",
            text
        ))),
    }
}

/// Build a Coord from a coord_clause node
fn build_coord(node: &Node, source: &str) -> Result<Coord> {
    let mut coord_type = CoordType::Cartesian;
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "COORD" | "USING" | "=" | "," => continue,
            "coord_type" => {
                coord_type = parse_coord_type(&child, source)?;
            }
            "coord_property" => {
                // Parse coord property: name = value
                let mut prop_cursor = child.walk();
                let mut prop_name = String::new();
                let mut prop_value: Option<CoordPropertyValue> = None;

                for prop_child in child.children(&mut prop_cursor) {
                    match prop_child.kind() {
                        "coord_property_name" => {
                            prop_name = get_node_text(&prop_child, source);
                        }
                        "string" | "number" | "boolean" | "array" => {
                            prop_value = Some(parse_coord_property_value(&prop_child, source)?);
                        }
                        "=" => continue,
                        _ => {}
                    }
                }

                if !prop_name.is_empty() && prop_value.is_some() {
                    properties.insert(prop_name, prop_value.unwrap());
                }
            }
            _ => {}
        }
    }

    Ok(Coord {
        coord_type,
        properties,
    })
}

/// Parse coord type from a coord_type node
fn parse_coord_type(node: &Node, source: &str) -> Result<CoordType> {
    let text = get_node_text(node, source);
    match text.to_lowercase().as_str() {
        "cartesian" => Ok(CoordType::Cartesian),
        "polar" => Ok(CoordType::Polar),
        "flip" => Ok(CoordType::Flip),
        "fixed" => Ok(CoordType::Fixed),
        "trans" => Ok(CoordType::Trans),
        "map" => Ok(CoordType::Map),
        "quickmap" => Ok(CoordType::QuickMap),
        _ => Err(VizqlError::ParseError(format!(
            "Unknown coord type: {}",
            text
        ))),
    }
}

/// Parse coord property value
fn parse_coord_property_value(node: &Node, source: &str) -> Result<CoordPropertyValue> {
    match node.kind() {
        "string" => {
            let text = get_node_text(node, source);
            let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
            Ok(CoordPropertyValue::String(unquoted.to_string()))
        }
        "number" => {
            let text = get_node_text(node, source);
            let num = text.parse::<f64>().map_err(|e| {
                VizqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
            })?;
            Ok(CoordPropertyValue::Number(num))
        }
        "boolean" => {
            let text = get_node_text(node, source);
            let bool_val = text == "true";
            Ok(CoordPropertyValue::Boolean(bool_val))
        }
        "array" => {
            // Parse array of values
            let mut values = Vec::new();
            let mut array_cursor = node.walk();
            for array_child in node.children(&mut array_cursor) {
                match array_child.kind() {
                    "string" => {
                        let text = get_node_text(&array_child, source);
                        let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                        values.push(ArrayElement::String(unquoted.to_string()));
                    }
                    "number" => {
                        let text = get_node_text(&array_child, source);
                        if let Ok(num) = text.parse::<f64>() {
                            values.push(ArrayElement::Number(num));
                        }
                    }
                    "boolean" => {
                        let text = get_node_text(&array_child, source);
                        let bool_val = text == "true";
                        values.push(ArrayElement::Boolean(bool_val));
                    }
                    _ => continue,
                }
            }
            Ok(CoordPropertyValue::Array(values))
        }
        _ => Err(VizqlError::ParseError(format!(
            "Unexpected coord property value type: {}",
            node.kind()
        ))),
    }
}

/// Build Labels from a label_clause node
fn build_labels(node: &Node, source: &str) -> Result<Labels> {
    let mut labels = HashMap::new();
    let mut cursor = node.walk();

    // Iterate through label_assignment children
    for child in node.children(&mut cursor) {
        if child.kind() == "label_assignment" {
            let mut assignment_cursor = child.walk();
            let mut label_type: Option<String> = None;
            let mut label_value: Option<String> = None;

            for assignment_child in child.children(&mut assignment_cursor) {
                match assignment_child.kind() {
                    "label_type" => {
                        label_type = Some(get_node_text(&assignment_child, source));
                    }
                    "string" => {
                        let text = get_node_text(&assignment_child, source);
                        // Remove quotes from string
                        label_value = Some(text.trim_matches(|c| c == '\'' || c == '"').to_string());
                    }
                    _ => {}
                }
            }

            if let (Some(typ), Some(val)) = (label_type, label_value) {
                labels.insert(typ, val);
            }
        }
    }

    Ok(Labels { labels })
}

/// Build a Guide from a guide_clause node
fn build_guide(node: &Node, source: &str) -> Result<Guide> {
    let mut aesthetic = String::new();
    let mut guide_type: Option<GuideType> = None;
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "GUIDE" | "USING" | "," => continue, // Skip keywords
            "aesthetic_name" => {
                aesthetic = get_node_text(&child, source);
            }
            "guide_property" => {
                // Parse guide property
                let mut prop_cursor = child.walk();
                for prop_child in child.children(&mut prop_cursor) {
                    if prop_child.kind() == "guide_type" {
                        // This is a type property: type = legend
                        let type_text = get_node_text(&prop_child, source);
                        guide_type = Some(parse_guide_type(&type_text)?);
                    } else if prop_child.kind() == "guide_property_name" {
                        // Regular property: name = value
                        let prop_name = get_node_text(&prop_child, source);

                        // Find the value (next sibling after '=')
                        let mut found_equals = false;
                        let mut value_cursor = child.walk();
                        for value_child in child.children(&mut value_cursor) {
                            if value_child.kind() == "=" {
                                found_equals = true;
                                continue;
                            }
                            if found_equals {
                                let prop_value = parse_guide_property_value(&value_child, source)?;
                                properties.insert(prop_name.clone(), prop_value);
                                break;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if aesthetic.is_empty() {
        return Err(VizqlError::ParseError(
            "Guide clause missing aesthetic name".to_string(),
        ));
    }

    Ok(Guide {
        aesthetic,
        guide_type,
        properties,
    })
}

/// Parse guide type from text
fn parse_guide_type(text: &str) -> Result<GuideType> {
    match text.to_lowercase().as_str() {
        "legend" => Ok(GuideType::Legend),
        "colorbar" => Ok(GuideType::ColorBar),
        "axis" => Ok(GuideType::Axis),
        "none" => Ok(GuideType::None),
        _ => Err(VizqlError::ParseError(format!(
            "Unknown guide type: {}",
            text
        ))),
    }
}

/// Parse guide property value
fn parse_guide_property_value(node: &Node, source: &str) -> Result<GuidePropertyValue> {
    match node.kind() {
        "string" => {
            let text = get_node_text(node, source);
            let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
            Ok(GuidePropertyValue::String(unquoted.to_string()))
        }
        "number" => {
            let text = get_node_text(node, source);
            let num = text.parse::<f64>().map_err(|e| {
                VizqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
            })?;
            Ok(GuidePropertyValue::Number(num))
        }
        "boolean" => {
            let text = get_node_text(node, source);
            let bool_val = text == "true";
            Ok(GuidePropertyValue::Boolean(bool_val))
        }
        _ => Err(VizqlError::ParseError(format!(
            "Unexpected guide property value type: {}",
            node.kind()
        ))),
    }
}

/// Build a Theme from a theme_clause node
fn build_theme(node: &Node, source: &str) -> Result<Theme> {
    let mut style: Option<String> = None;
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "THEME" | "USING" | "," => continue,
            "theme_name" => {
                style = Some(get_node_text(&child, source));
            }
            "theme_property" => {
                // Parse theme property: name = value
                let mut prop_cursor = child.walk();
                let mut prop_name = String::new();
                let mut prop_value: Option<ThemePropertyValue> = None;

                for prop_child in child.children(&mut prop_cursor) {
                    match prop_child.kind() {
                        "theme_property_name" => {
                            prop_name = get_node_text(&prop_child, source);
                        }
                        "string" | "number" | "boolean" => {
                            prop_value = Some(parse_theme_property_value(&prop_child, source)?);
                        }
                        "=" => continue,
                        _ => {}
                    }
                }

                if !prop_name.is_empty() && prop_value.is_some() {
                    properties.insert(prop_name, prop_value.unwrap());
                }
            }
            _ => {}
        }
    }

    Ok(Theme { style, properties })
}

/// Parse theme property value
fn parse_theme_property_value(node: &Node, source: &str) -> Result<ThemePropertyValue> {
    match node.kind() {
        "string" => {
            let text = get_node_text(node, source);
            let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
            Ok(ThemePropertyValue::String(unquoted.to_string()))
        }
        "number" => {
            let text = get_node_text(node, source);
            let num = text.parse::<f64>().map_err(|e| {
                VizqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
            })?;
            Ok(ThemePropertyValue::Number(num))
        }
        "boolean" => {
            let text = get_node_text(node, source);
            let bool_val = text == "true";
            Ok(ThemePropertyValue::Boolean(bool_val))
        }
        _ => Err(VizqlError::ParseError(format!(
            "Unexpected theme property value type: {}",
            node.kind()
        ))),
    }
}

/// Get text content of a node
fn get_node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stub_implementation() {
        // This is just a placeholder test
        // Real tests will be added when the full implementation is done
        assert!(true);
    }
}
