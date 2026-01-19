//! AST builder - converts tree-sitter CST to typed AST
//!
//! Takes a tree-sitter parse tree and builds a typed VizSpec AST,
//! handling all the node types defined in the grammar.

use super::ast::*;
use crate::{GgsqlError, Result};
use std::collections::HashMap;
use tree_sitter::{Node, Tree};

/// Build a VizSpec AST from a tree-sitter parse tree
pub fn build_ast(tree: &Tree, source: &str) -> Result<Vec<VizSpec>> {
    let root = tree.root_node();

    // Check if root is a query node
    if root.kind() != "query" {
        return Err(GgsqlError::ParseError(format!(
            "Expected 'query' root node, got '{}'",
            root.kind()
        )));
    }

    // Extract SQL portion node (if exists)
    let sql_portion_node = root
        .children(&mut root.walk())
        .find(|n| n.kind() == "sql_portion");

    // Check if last SQL statement is SELECT
    let last_is_select = if let Some(sql_node) = sql_portion_node {
        check_last_statement_is_select(&sql_node)
    } else {
        false
    };

    let mut specs = Vec::new();

    // Walk through child nodes - each visualise_statement becomes a VizSpec
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "visualise_statement" {
            let spec = build_visualise_statement(&child, source)?;

            // Validate VISUALISE FROM usage
            if spec.source.is_some() && last_is_select {
                return Err(GgsqlError::ParseError(
                    "Cannot use VISUALISE FROM when the last SQL statement is SELECT. \
                     Use either 'SELECT ... VISUALISE' or remove the SELECT and use \
                     'VISUALISE FROM ...'."
                        .to_string(),
                ));
            }

            specs.push(spec);
        }
    }

    if specs.is_empty() {
        return Err(GgsqlError::ParseError(
            "No VISUALISE statements found in query".to_string(),
        ));
    }

    Ok(specs)
}

/// Build a single VizSpec from a visualise_statement node
fn build_visualise_statement(node: &Node, source: &str) -> Result<VizSpec> {
    let mut spec = VizSpec::new();

    // Walk through children of visualise_statement
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "VISUALISE" | "VISUALIZE" | "FROM" => {
                // Skip keywords
                continue;
            }
            "global_mapping" => {
                // Parse global mapping (may include wildcard and/or explicit mappings)
                spec.global_mappings = parse_global_mapping(&child, source)?;
            }
            "wildcard_mapping" => {
                // Handle standalone wildcard (*) mapping
                spec.global_mappings.wildcard = true;
            }
            "identifier" | "string" => {
                // This is the FROM source (table name or file path)
                let text = get_node_text(&child, source);
                spec.source = Some(match child.kind() {
                    "string" => {
                        // Remove surrounding quotes for file paths
                        let path = text.trim_matches(|c| c == '\'' || c == '"');
                        DataSource::FilePath(path.to_string())
                    }
                    _ => DataSource::Identifier(text.to_string()),
                });
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

    // Validate no conflicts between SCALE and COORD domain specifications
    validate_scale_coord_conflicts(&spec)?;

    Ok(spec)
}

/// Parse global_mapping node into Mappings struct
/// global_mapping contains a mapping_list child node
fn parse_global_mapping(node: &Node, source: &str) -> Result<Mappings> {
    // global_mapping: $ => $.mapping_list - contains a mapping_list child node
    let mut mappings = Mappings::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "mapping_list" {
            parse_mapping_list(&child, source, &mut mappings)?;
        }
    }
    Ok(mappings)
}

/// Parse a mapping_list: comma-separated mapping_element nodes
/// Shared by both global (VISUALISE) and layer (MAPPING) mappings
fn parse_mapping_list(node: &Node, source: &str, mappings: &mut Mappings) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "mapping_element" => {
                parse_mapping_element(&child, source, mappings)?;
            }
            "," => continue,
            _ => continue,
        }
    }
    Ok(())
}

/// Parse an explicit_mapping node (value AS aesthetic)
/// Returns (aesthetic_name, value)
fn parse_explicit_mapping(node: &Node, source: &str) -> Result<(String, AestheticValue)> {
    let mut value: Option<AestheticValue> = None;
    let mut aesthetic: Option<String> = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "mapping_value" => {
                // Get the column/literal value
                let mut inner_cursor = child.walk();
                for inner_child in child.children(&mut inner_cursor) {
                    match inner_child.kind() {
                        "column_reference" => {
                            let mut ref_cursor = inner_child.walk();
                            for ref_child in inner_child.children(&mut ref_cursor) {
                                if ref_child.kind() == "identifier" {
                                    value = Some(AestheticValue::standard_column(get_node_text(
                                        &ref_child, source,
                                    )));
                                }
                            }
                        }
                        "identifier" => {
                            value = Some(AestheticValue::standard_column(get_node_text(
                                &inner_child,
                                source,
                            )));
                        }
                        "literal_value" => {
                            value = Some(parse_literal_value(&inner_child, source)?);
                        }
                        _ => {}
                    }
                }
            }
            "aesthetic_name" => {
                aesthetic = Some(get_node_text(&child, source));
            }
            "AS" => continue,
            _ => continue,
        }
    }

    match (value, aesthetic) {
        (Some(val), Some(aes)) => Ok((aes, val)),
        _ => Err(GgsqlError::ParseError(
            "Invalid explicit mapping: missing value or aesthetic".to_string(),
        )),
    }
}

/// Check for conflicts between SCALE domain and COORD aesthetic domain specifications
fn validate_scale_coord_conflicts(spec: &VizSpec) -> Result<()> {
    if let Some(ref coord) = spec.coord {
        // Get all aesthetic names that have domains in COORD
        let coord_aesthetics: Vec<String> = coord
            .properties
            .keys()
            .filter(|k| is_aesthetic_name(k))
            .cloned()
            .collect();

        // Check if any of these also have domain in SCALE
        for aesthetic in coord_aesthetics {
            for scale in &spec.scales {
                if scale.aesthetic == aesthetic {
                    // Check if this scale has a domain property
                    if scale.properties.contains_key("domain") {
                        return Err(GgsqlError::ParseError(format!(
                            "Domain for '{}' specified in both SCALE and COORD clauses. \
                            Please specify domain in only one location.",
                            aesthetic
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Process a visualization clause node
fn process_viz_clause(node: &Node, source: &str, spec: &mut VizSpec) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "draw_clause" => {
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

/// Build a Layer from a draw_clause node
/// Syntax: DRAW geom [MAPPING col AS x, ... [FROM source]] [REMAPPING stat AS aes, ...] [SETTING param => val, ...] [PARTITION BY col, ...] [FILTER condition]
fn build_layer(node: &Node, source: &str) -> Result<Layer> {
    let mut geom = Geom::Point; // default
    let mut aesthetics = Mappings::new();
    let mut remappings = Mappings::new();
    let mut parameters = HashMap::new();
    let mut partition_by = Vec::new();
    let mut filter = None;
    let mut order_by = None;
    let mut layer_source = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "geom_type" => {
                let geom_text = get_node_text(&child, source);
                geom = parse_geom_type(&geom_text)?;
            }
            "mapping_clause" => {
                let (aes, src) = parse_mapping_clause(&child, source)?;
                aesthetics = aes;
                layer_source = src;
            }
            "remapping_clause" => {
                // Reuse parse_mapping_clause - remapping has same syntax, just different semantics
                let (remap, _) = parse_mapping_clause(&child, source)?;
                remappings = remap;
            }
            "setting_clause" => {
                parameters = parse_setting_clause(&child, source)?;
            }
            "partition_clause" => {
                partition_by = parse_partition_clause(&child, source)?;
            }
            "filter_clause" => {
                filter = Some(parse_filter_clause(&child, source)?);
            }
            "order_clause" => {
                order_by = Some(parse_order_clause(&child, source)?);
            }
            _ => {
                // Skip keywords and punctuation
                continue;
            }
        }
    }

    let mut layer = Layer::new(geom);
    layer.mappings = aesthetics;
    layer.remappings = remappings;
    layer.parameters = parameters;
    layer.partition_by = partition_by;
    layer.filter = filter;
    layer.order_by = order_by;
    layer.source = layer_source;

    Ok(layer)
}

/// Parse a mapping_clause: MAPPING col AS x, "blue" AS color [FROM source]
/// Returns (aesthetics as Mappings, optional data source)
fn parse_mapping_clause(node: &Node, source: &str) -> Result<(Mappings, Option<DataSource>)> {
    let mut mappings = Mappings::new();

    // Parse mapping elements using the shared mapping_list structure
    // With the unified grammar, all aesthetic mappings come through mapping_list.
    // Bare identifiers here are part of the FROM clause, not mappings.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "mapping_list" => {
                parse_mapping_list(&child, source, &mut mappings)?;
            }
            _ => continue,
        }
    }

    // Extract layer_source field (FROM identifier or FROM 'file.csv')
    let data_source = node.child_by_field_name("layer_source").map(|child| {
        let text = get_node_text(&child, source);
        match child.kind() {
            "identifier" => DataSource::Identifier(text.to_string()),
            "string" => {
                // Remove surrounding quotes
                let path = text.trim_matches(|c| c == '\'' || c == '"');
                DataSource::FilePath(path.to_string())
            }
            _ => DataSource::Identifier(text.to_string()),
        }
    });

    Ok((mappings, data_source))
}

/// Parse a mapping_element: wildcard, explicit, or implicit mapping
/// Shared by both global (VISUALISE) and layer (MAPPING) mappings
fn parse_mapping_element(node: &Node, source: &str, mappings: &mut Mappings) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "wildcard_mapping" => {
                mappings.wildcard = true;
            }
            "explicit_mapping" => {
                let (aesthetic, value) = parse_explicit_mapping(&child, source)?;
                mappings.insert(aesthetic, value);
            }
            "implicit_mapping" | "identifier" => {
                let name = get_node_text(&child, source);
                mappings.insert(&name, AestheticValue::standard_column(&name));
            }
            _ => continue,
        }
    }
    Ok(())
}

/// Parse a setting_clause: SETTING param => value, ...
fn parse_setting_clause(node: &Node, source: &str) -> Result<HashMap<String, ParameterValue>> {
    let mut parameters = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "parameter_assignment" {
            let (param, mut value) = parse_parameter_assignment(&child, source)?;
            match param.as_str() {
                "color" | "col" | "colour" | "fill" | "stroke" => {
                    if let ParameterValue::String(color) = value {
                        value = ParameterValue::String(color_to_hex(&color));
                    }
                }
                _ => {}
            }
            parameters.insert(param, value);
        }
    }

    Ok(parameters)
}

/// Parse a partition_clause: PARTITION BY col1, col2, ...
fn parse_partition_clause(node: &Node, source: &str) -> Result<Vec<String>> {
    let mut columns = Vec::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "partition_columns" {
            let mut inner_cursor = child.walk();
            for inner_child in child.children(&mut inner_cursor) {
                if inner_child.kind() == "identifier" {
                    columns.push(get_node_text(&inner_child, source));
                }
            }
        }
    }

    Ok(columns)
}

/// Parse a parameter_assignment: param => value
fn parse_parameter_assignment(node: &Node, source: &str) -> Result<(String, ParameterValue)> {
    let mut param_name = String::new();
    let mut param_value = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "parameter_name" => {
                // parameter_name -> identifier
                let mut inner_cursor = child.walk();
                for inner_child in child.children(&mut inner_cursor) {
                    if inner_child.kind() == "identifier" {
                        param_name = get_node_text(&inner_child, source);
                    }
                }
                if param_name.is_empty() {
                    param_name = get_node_text(&child, source);
                }
            }
            "parameter_value" => {
                param_value = Some(parse_parameter_value(&child, source)?);
            }
            _ => continue,
        }
    }

    if param_name.is_empty() || param_value.is_none() {
        return Err(GgsqlError::ParseError(format!(
            "Invalid parameter assignment: param='{}', value={:?}",
            param_name, param_value
        )));
    }

    Ok((param_name, param_value.unwrap()))
}

/// Parse a parameter_value (string, number, or boolean)
fn parse_parameter_value(node: &Node, source: &str) -> Result<ParameterValue> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string" => {
                let text = get_node_text(&child, source);
                let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                return Ok(ParameterValue::String(unquoted.to_string()));
            }
            "number" => {
                let text = get_node_text(&child, source);
                let num = text.parse::<f64>().map_err(|e| {
                    GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
                })?;
                return Ok(ParameterValue::Number(num));
            }
            "boolean" => {
                let text = get_node_text(&child, source);
                let bool_val = text == "true";
                return Ok(ParameterValue::Boolean(bool_val));
            }
            _ => {}
        }
    }

    Err(GgsqlError::ParseError(format!(
        "Could not parse parameter value from node: {}",
        node.kind()
    )))
}

/// Parse a filter_clause: FILTER <raw SQL expression>
///
/// Extracts the raw SQL text from the filter_expression and returns it verbatim.
/// This allows any valid SQL WHERE expression to be passed to the database backend.
fn parse_filter_clause(node: &Node, source: &str) -> Result<SqlExpression> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "filter_expression" {
            // Extract the raw text from the filter_expression node
            let filter_text = get_node_text(&child, source).trim().to_string();
            return Ok(SqlExpression::new(filter_text));
        }
    }

    Err(GgsqlError::ParseError(
        "Could not find filter expression in filter clause".to_string(),
    ))
}

/// Parse an order_clause: ORDER BY date ASC, value DESC
fn parse_order_clause(node: &Node, source: &str) -> Result<SqlExpression> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "order_expression" {
            // Extract the raw text from the order_expression node
            let order_text = get_node_text(&child, source).trim().to_string();
            return Ok(SqlExpression::new(order_text));
        }
    }

    Err(GgsqlError::ParseError(
        "Could not find order expression in order clause".to_string(),
    ))
}

/// Parse a geom_type node text into a Geom enum
fn parse_geom_type(text: &str) -> Result<Geom> {
    match text.to_lowercase().as_str() {
        "point" => Ok(Geom::Point),
        "line" => Ok(Geom::Line),
        "path" => Ok(Geom::Path),
        "bar" => Ok(Geom::Bar),
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
        _ => Err(GgsqlError::ParseError(format!(
            "Unknown geom type: {}",
            text
        ))),
    }
}

/// Parse a literal_value node into an AestheticValue::Literal
fn parse_literal_value(node: &Node, source: &str) -> Result<AestheticValue> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string" => {
                let text = get_node_text(&child, source);
                let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                return Ok(AestheticValue::Literal(LiteralValue::String(
                    unquoted.to_string(),
                )));
            }
            "number" => {
                let text = get_node_text(&child, source);
                let num = text.parse::<f64>().map_err(|e| {
                    GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
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

    Err(GgsqlError::ParseError(format!(
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
            "SCALE" | "SETTING" | "=>" | "," => continue, // Skip keywords
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
                        "=>" => continue,
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
        return Err(GgsqlError::ParseError(
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
        _ => Err(GgsqlError::ParseError(format!(
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
                    GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
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
                    if array_child.kind() == "array_element" {
                        // Array elements wrap the actual values
                        let mut elem_cursor = array_child.walk();
                        for elem_child in array_child.children(&mut elem_cursor) {
                            match elem_child.kind() {
                                "string" => {
                                    let text = get_node_text(&elem_child, source);
                                    let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                                    values.push(ArrayElement::String(unquoted.to_string()));
                                }
                                "number" => {
                                    let text = get_node_text(&elem_child, source);
                                    if let Ok(num) = text.parse::<f64>() {
                                        values.push(ArrayElement::Number(num));
                                    }
                                }
                                "boolean" => {
                                    let text = get_node_text(&elem_child, source);
                                    let bool_val = text == "true";
                                    values.push(ArrayElement::Boolean(bool_val));
                                }
                                _ => continue,
                            }
                        }
                    }
                }
                return Ok(ScalePropertyValue::Array(values));
            }
            _ => {}
        }
    }

    Err(GgsqlError::ParseError(format!(
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
            "FACET" | "SETTING" | "=>" => continue,
            "facet_wrap" => {
                is_wrap = true;
            }
            "facet_by" => {
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
        _ => Err(GgsqlError::ParseError(format!(
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
            "COORD" | "SETTING" | "=>" | "," => continue,
            "coord_type" => {
                coord_type = parse_coord_type(&child, source)?;
            }
            "coord_properties" => {
                // New grammar structure: coord_properties contains multiple coord_property
                let mut props_cursor = child.walk();
                for prop_node in child.children(&mut props_cursor) {
                    if prop_node.kind() == "coord_property" {
                        let (prop_name, prop_value) =
                            parse_single_coord_property(&prop_node, source)?;
                        properties.insert(prop_name, prop_value);
                    }
                }
            }
            _ => {}
        }
    }

    // Validate properties for this coord type
    validate_coord_properties(&coord_type, &properties)?;

    Ok(Coord {
        coord_type,
        properties,
    })
}

/// Parse a single coord_property node into (name, value)
fn parse_single_coord_property(node: &Node, source: &str) -> Result<(String, CoordPropertyValue)> {
    let mut prop_name = String::new();
    let mut prop_value: Option<CoordPropertyValue> = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "coord_property_name" => {
                // Could be a keyword or an aesthetic_name
                let mut name_cursor = child.walk();
                for name_child in child.children(&mut name_cursor) {
                    match name_child.kind() {
                        "aesthetic_name" => {
                            prop_name = get_node_text(&name_child, source);
                        }
                        _ => {
                            // Direct keyword like xlim, ylim, theta
                            prop_name = get_node_text(&child, source);
                            break;
                        }
                    }
                }
                if prop_name.is_empty() {
                    prop_name = get_node_text(&child, source);
                }
            }
            "string" | "number" | "boolean" | "array" => {
                prop_value = Some(parse_coord_property_value(&child, source)?);
            }
            "identifier" => {
                // New: identifiers can be property values (e.g., theta = y)
                let ident = get_node_text(&child, source);
                prop_value = Some(CoordPropertyValue::String(ident));
            }
            "=" => continue,
            _ => {}
        }
    }

    if prop_name.is_empty() || prop_value.is_none() {
        return Err(GgsqlError::ParseError(format!(
            "Invalid coord property: name='{}', value present={}",
            prop_name,
            prop_value.is_some()
        )));
    }

    Ok((prop_name, prop_value.unwrap()))
}

/// Validate that properties are valid for the given coord type
fn validate_coord_properties(
    coord_type: &CoordType,
    properties: &HashMap<String, CoordPropertyValue>,
) -> Result<()> {
    for prop_name in properties.keys() {
        let valid = match coord_type {
            CoordType::Cartesian => {
                // Cartesian allows: xlim, ylim, aesthetic names
                // Not allowed: theta
                prop_name == "xlim" || prop_name == "ylim" || is_aesthetic_name(prop_name)
            }
            CoordType::Flip => {
                // Flip allows: aesthetic names only
                // Not allowed: xlim, ylim, theta
                is_aesthetic_name(prop_name)
            }
            CoordType::Polar => {
                // Polar allows: theta, aesthetic names
                // Not allowed: xlim, ylim
                prop_name == "theta" || is_aesthetic_name(prop_name)
            }
            _ => {
                // Other coord types: allow all for now (future implementation)
                true
            }
        };

        if !valid {
            let valid_props = match coord_type {
                CoordType::Cartesian => "xlim, ylim, <aesthetics>",
                CoordType::Flip => "<aesthetics>",
                CoordType::Polar => "theta, <aesthetics>",
                _ => "<varies>",
            };
            return Err(GgsqlError::ParseError(format!(
                "Property '{}' not valid for {:?} coordinates. Valid properties: {}",
                prop_name, coord_type, valid_props
            )));
        }
    }

    Ok(())
}

/// Check if a property name is an aesthetic name
fn is_aesthetic_name(name: &str) -> bool {
    matches!(
        name,
        "x" | "y"
            | "xmin"
            | "xmax"
            | "ymin"
            | "ymax"
            | "xend"
            | "yend"
            | "color"
            | "colour"
            | "fill"
            | "opacity"
            | "size"
            | "shape"
            | "linetype"
            | "linewidth"
            | "width"
            | "height"
            | "label"
            | "family"
            | "fontface"
            | "hjust"
            | "vjust"
    )
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
        _ => Err(GgsqlError::ParseError(format!(
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
                GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
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
                if array_child.kind() == "array_element" {
                    // Array elements wrap the actual values
                    let mut elem_cursor = array_child.walk();
                    for elem_child in array_child.children(&mut elem_cursor) {
                        match elem_child.kind() {
                            "string" => {
                                let text = get_node_text(&elem_child, source);
                                let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
                                values.push(ArrayElement::String(unquoted.to_string()));
                            }
                            "number" => {
                                let text = get_node_text(&elem_child, source);
                                if let Ok(num) = text.parse::<f64>() {
                                    values.push(ArrayElement::Number(num));
                                }
                            }
                            "boolean" => {
                                let text = get_node_text(&elem_child, source);
                                let bool_val = text == "true";
                                values.push(ArrayElement::Boolean(bool_val));
                            }
                            _ => continue,
                        }
                    }
                }
            }
            Ok(CoordPropertyValue::Array(values))
        }
        _ => Err(GgsqlError::ParseError(format!(
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
                        label_value =
                            Some(text.trim_matches(|c| c == '\'' || c == '"').to_string());
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
            "GUIDE" | "SETTING" | "=>" | "," => continue, // Skip keywords
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

                        // Find the value (next sibling after '=>')
                        let mut found_to = false;
                        let mut value_cursor = child.walk();
                        for value_child in child.children(&mut value_cursor) {
                            if value_child.kind() == "=>" {
                                found_to = true;
                                continue;
                            }
                            if found_to {
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
        return Err(GgsqlError::ParseError(
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
        _ => Err(GgsqlError::ParseError(format!(
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
                GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
            })?;
            Ok(GuidePropertyValue::Number(num))
        }
        "boolean" => {
            let text = get_node_text(node, source);
            let bool_val = text == "true";
            Ok(GuidePropertyValue::Boolean(bool_val))
        }
        _ => Err(GgsqlError::ParseError(format!(
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
            "THEME" | "SETTING" | "=>" | "," => continue,
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
                        "=>" => continue,
                        _ => {}
                    }
                }
                if !prop_name.is_empty() {
                    if let Some(value) = prop_value {
                        properties.insert(prop_name, value);
                    }
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
                GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e))
            })?;
            Ok(ThemePropertyValue::Number(num))
        }
        "boolean" => {
            let text = get_node_text(node, source);
            let bool_val = text == "true";
            Ok(ThemePropertyValue::Boolean(bool_val))
        }
        _ => Err(GgsqlError::ParseError(format!(
            "Unexpected theme property value type: {}",
            node.kind()
        ))),
    }
}

/// Get text content of a node
fn get_node_text(node: &Node, source: &str) -> String {
    source[node.start_byte()..node.end_byte()].to_string()
}

/// Check if the last SQL statement in sql_portion is a SELECT statement
fn check_last_statement_is_select(sql_portion_node: &Node) -> bool {
    let mut last_statement = None;
    let mut cursor = sql_portion_node.walk();

    // Find last sql_statement node
    for child in sql_portion_node.children(&mut cursor) {
        if child.kind() == "sql_statement" {
            last_statement = Some(child);
        }
    }

    // Check if last statement is or ends with a SELECT
    if let Some(stmt) = last_statement {
        let mut stmt_cursor = stmt.walk();
        for child in stmt.children(&mut stmt_cursor) {
            if child.kind() == "select_statement" {
                // Direct select_statement child
                return true;
            } else if child.kind() == "with_statement" {
                // Check if WITH has trailing SELECT
                return with_statement_has_trailing_select(&child);
            }
        }
    }

    false
}

/// Check if a with_statement has a trailing SELECT (after the CTE definitions)
fn with_statement_has_trailing_select(with_node: &Node) -> bool {
    let mut cursor = with_node.walk();
    let mut seen_cte_definition = false;

    for child in with_node.children(&mut cursor) {
        if child.kind() == "cte_definition" {
            seen_cte_definition = true;
        } else if child.kind() == "select_statement" && seen_cte_definition {
            // This is a SELECT after CTE definitions (trailing SELECT)
            return true;
        }
    }

    false
}

fn color_to_hex(value: &str) -> String {
    match csscolorparser::parse(value) {
        Ok(value) => value.to_css_hex(),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse_test_query(query: &str) -> Result<Vec<VizSpec>> {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_ggsql::language()).unwrap();

        let tree = parser.parse(query, None).unwrap();

        // Check for parse errors like the main parse_full_query does
        if tree.root_node().has_error() {
            return Err(GgsqlError::ParseError(
                "Parse tree contains errors".to_string(),
            ));
        }

        build_ast(&tree, query)
    }

    // ========================================
    // COORD Property Validation Tests
    // ========================================

    #[test]
    fn test_coord_cartesian_valid_xlim() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            COORD cartesian SETTING xlim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok(), "Parse failed: {:?}", result);
        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);

        let coord = specs[0].coord.as_ref().unwrap();
        assert_eq!(coord.coord_type, CoordType::Cartesian);
        assert!(coord.properties.contains_key("xlim"));
    }

    #[test]
    fn test_coord_cartesian_valid_ylim() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            COORD cartesian SETTING ylim => [-10, 50]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert!(coord.properties.contains_key("ylim"));
    }

    #[test]
    fn test_coord_cartesian_valid_aesthetic_domain() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            COORD cartesian SETTING color => ['red', 'green', 'blue']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert!(coord.properties.contains_key("color"));
    }

    #[test]
    fn test_coord_cartesian_invalid_property_theta() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            COORD cartesian SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'theta' not valid for Cartesian"));
    }

    #[test]
    fn test_coord_flip_valid_aesthetic_domain() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y, region AS color
            COORD flip SETTING color => ['A', 'B', 'C']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert_eq!(coord.coord_type, CoordType::Flip);
        assert!(coord.properties.contains_key("color"));
    }

    #[test]
    fn test_coord_flip_invalid_property_xlim() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD flip SETTING xlim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'xlim' not valid for Flip"));
    }

    #[test]
    fn test_coord_flip_invalid_property_ylim() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD flip SETTING ylim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'ylim' not valid for Flip"));
    }

    #[test]
    fn test_coord_flip_invalid_property_theta() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD flip SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'theta' not valid for Flip"));
    }

    #[test]
    fn test_coord_polar_valid_theta() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD polar SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert_eq!(coord.coord_type, CoordType::Polar);
        assert!(coord.properties.contains_key("theta"));
    }

    #[test]
    fn test_coord_polar_valid_aesthetic_domain() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y, region AS color
            COORD polar SETTING color => ['North', 'South', 'East', 'West']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert!(coord.properties.contains_key("color"));
    }

    #[test]
    fn test_coord_polar_invalid_property_xlim() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD polar SETTING xlim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'xlim' not valid for Polar"));
    }

    #[test]
    fn test_coord_polar_invalid_property_ylim() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            COORD polar SETTING ylim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'ylim' not valid for Polar"));
    }

    // ========================================
    // SCALE/COORD Domain Conflict Tests
    // ========================================

    #[test]
    fn test_scale_coord_conflict_x_domain() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            SCALE x SETTING domain => [0, 100]
            COORD cartesian SETTING x => [0, 50]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Domain for 'x' specified in both SCALE and COORD"));
    }

    #[test]
    fn test_scale_coord_conflict_color_domain() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            SCALE color SETTING domain => ['A', 'B']
            COORD cartesian SETTING color => ['A', 'B', 'C']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Domain for 'color' specified in both SCALE and COORD"));
    }

    #[test]
    fn test_scale_coord_no_conflict_different_aesthetics() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            SCALE color SETTING domain => ['A', 'B']
            COORD cartesian SETTING xlim => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scale_coord_no_conflict_scale_without_domain() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            SCALE x SETTING type => 'linear'
            COORD cartesian SETTING x => [0, 100]
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    // ========================================
    // Multiple Properties Tests
    // ========================================

    #[test]
    fn test_coord_cartesian_multiple_properties() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            COORD cartesian SETTING xlim => [0, 100], ylim => [-10, 50], color => ['A', 'B']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert!(coord.properties.contains_key("xlim"));
        assert!(coord.properties.contains_key("ylim"));
        assert!(coord.properties.contains_key("color"));
    }

    #[test]
    fn test_coord_polar_theta_with_aesthetic() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y, region AS color
            COORD polar SETTING theta => y, color => ['North', 'South']
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let coord = specs[0].coord.as_ref().unwrap();
        assert!(coord.properties.contains_key("theta"));
        assert!(coord.properties.contains_key("color"));
    }

    // ========================================
    // Case Insensitive Keywords Tests
    // ========================================

    #[test]
    fn test_case_insensitive_keywords_lowercase() {
        let query = r#"
            visualise
            draw point MAPPING x AS x, y AS y
            coord cartesian setting xlim => [0, 100]
            label title => 'Test Chart'
        "#;

        let result = parse_test_query(query);
        if let Err(ref e) = result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok());
        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert!(specs[0].global_mappings.is_empty());
        assert_eq!(specs[0].layers.len(), 1);
        assert!(specs[0].coord.is_some());
        assert!(specs[0].labels.is_some());
    }

    #[test]
    fn test_case_insensitive_keywords_mixed() {
        let query = r#"
            ViSuAlIsE date AS x, revenue AS y
            DrAw line
            ScAlE x SeTtInG type => 'date'
            ThEmE minimal
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);
        assert_eq!(specs[0].scales.len(), 1);
        assert!(specs[0].theme.is_some());
    }

    #[test]
    fn test_case_insensitive_american_spelling() {
        let query = r#"
            visualize category AS x, value AS y
            draw bar
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
    }

    // ========================================
    // VISUALISE FROM Validation Tests
    // ========================================

    #[test]
    fn test_visualise_from_cte() {
        let query = r#"
            WITH cte AS (SELECT * FROM x)
            VISUALISE FROM cte
            DRAW bar MAPPING a AS x, b AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("cte".to_string()))
        );
    }

    #[test]
    fn test_visualise_from_table() {
        let query = r#"
            VISUALISE FROM mtcars
            DRAW point MAPPING mpg AS x, hp AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("mtcars".to_string()))
        );
    }

    #[test]
    fn test_visualise_from_file_path() {
        let query = r#"
            VISUALISE FROM 'mtcars.csv'
            DRAW point MAPPING hp AS x, mpg AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        // Source should be stored without quotes in AST
        assert_eq!(
            specs[0].source,
            Some(DataSource::FilePath("mtcars.csv".to_string()))
        );
    }

    #[test]
    fn test_visualise_from_file_path_parquet() {
        let query = r#"
            VISUALISE FROM "data/sales.parquet"
            DRAW bar MAPPING region AS x, total AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        // Source should be stored without quotes
        assert_eq!(
            specs[0].source,
            Some(DataSource::FilePath("data/sales.parquet".to_string()))
        );
    }

    #[test]
    fn test_error_select_with_from() {
        let query = r#"
            SELECT * FROM x
            VISUALISE FROM y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Cannot use VISUALISE FROM when the last SQL statement is SELECT"));
    }

    #[test]
    fn test_allow_non_select_with_from() {
        let query = r#"
            CREATE TABLE x AS SELECT 1;
            WITH cte AS (SELECT * FROM x)
            VISUALISE FROM cte
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compat_select_visualise_as() {
        let query = r#"
            SELECT * FROM x
            VISUALISE
            DRAW bar MAPPING a AS x, b AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs[0].source, None); // No FROM clause
    }

    #[test]
    fn test_with_select_visualise_as() {
        let query = r#"
            WITH cte AS (SELECT * FROM x)
            SELECT * FROM cte
            VISUALISE
            DRAW point MAPPING a AS x, b AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(specs[0].source, None); // No FROM clause in VISUALISE
    }

    #[test]
    fn test_error_with_select_and_visualise_from() {
        let query = r#"
            WITH cte AS (SELECT * FROM x)
            SELECT * FROM cte
            VISUALISE FROM cte
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Cannot use VISUALISE FROM when the last SQL statement is SELECT"));
    }

    // ========================================
    // Complex SQL Edge Cases
    // ========================================

    #[test]
    fn test_deeply_nested_subqueries() {
        let query = r#"
            SELECT * FROM (SELECT * FROM (SELECT 1 as x, 2 as y))
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_values_rows() {
        let query = r#"
            SELECT * FROM (VALUES (1, 2), (3, 4), (5, 6)) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_ctes_no_select_with_visualise_from() {
        let query = r#"
            WITH a AS (SELECT 1 as x), b AS (SELECT 2 as y), c AS (SELECT 3 as z)
            VISUALISE FROM c
            DRAW point MAPPING z AS x, 1 AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("c".to_string()))
        );
    }

    #[test]
    fn test_union_with_visualise_as() {
        let query = r#"
            SELECT x, y FROM a UNION SELECT x, y FROM b
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_union_with_visualise_from() {
        let query = r#"
            SELECT x FROM a UNION SELECT x FROM b
            VISUALISE FROM c
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Cannot use VISUALISE FROM"));
    }

    #[test]
    fn test_subquery_in_where_clause() {
        let query = r#"
            SELECT * FROM data WHERE x IN (SELECT y FROM other)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_join_with_visualise_as() {
        let query = r#"
            SELECT a.x, b.y FROM a LEFT JOIN b ON a.id = b.id
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_window_function_with_visualise_as() {
        let query = r#"
            SELECT x, y, ROW_NUMBER() OVER (ORDER BY x) as row_num FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cte_with_join_then_visualise_from() {
        let query = r#"
            WITH joined AS (
                SELECT a.x, b.y FROM a JOIN b ON a.id = b.id
            )
            VISUALISE FROM joined
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_recursive_cte_with_visualise_from() {
        let query = r#"
            WITH RECURSIVE series AS (
                SELECT 1 as n
                UNION ALL
                SELECT n + 1 FROM series WHERE n < 10
            )
            VISUALISE FROM series
            DRAW line MAPPING n AS x, n AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_visualise_keyword_in_string_literal() {
        let query = r#"
            SELECT 'VISUALISE' as text, 1 as x, 2 as y
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_group_by_having_with_visualise_as() {
        let query = r#"
            SELECT category, SUM(value) as total FROM data
            GROUP BY category
            HAVING SUM(value) > 100
            VISUALISE
            DRAW bar MAPPING category AS x, total AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_order_by_limit_with_visualise_as() {
        let query = r#"
            SELECT * FROM data
            ORDER BY x DESC
            LIMIT 100
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_case_expression_with_visualise_as() {
        let query = r#"
            SELECT x,
                   CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END as sign
            FROM data
            VISUALISE
            DRAW point MAPPING x AS x, sign AS color
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intersect_with_visualise_as() {
        let query = r#"
            SELECT x FROM a INTERSECT SELECT x FROM b
            VISUALISE
            DRAW histogram MAPPING x AS x
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_intersect_with_visualise_from() {
        let query = r#"
            SELECT x FROM a INTERSECT SELECT x FROM b
            VISUALISE FROM c
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_except_with_visualise_as() {
        let query = r#"
            SELECT x FROM a EXCEPT SELECT x FROM b
            VISUALISE
            DRAW histogram MAPPING x AS x
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_with_semicolon_between_cte_and_visualise_from() {
        let query = r#"
            WITH cte AS (SELECT 1 as x, 2 as y);
            VISUALISE FROM cte
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_statements_with_semicolons_and_visualise_from() {
        let query = r#"
            CREATE TABLE temp AS SELECT 1 as x;
            INSERT INTO temp VALUES (2);
            WITH final AS (SELECT * FROM temp)
            VISUALISE FROM final
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_subquery_with_aggregation() {
        let query = r#"
            SELECT * FROM (
                SELECT category, AVG(value) as avg_value
                FROM data
                GROUP BY category
            )
            VISUALISE
            DRAW bar MAPPING category AS x, avg_value AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lateral_join_with_visualise_as() {
        let query = r#"
            SELECT a.*, b.*
            FROM a, LATERAL (SELECT * FROM b WHERE b.id = a.id) AS b
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_values_without_table_alias() {
        let query = r#"
            SELECT * FROM (VALUES (1, 2))
            VISUALISE
            DRAW point MAPPING column0 AS x, column1 AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nested_ctes() {
        let query = r#"
            WITH outer_cte AS (
                WITH inner_cte AS (SELECT 1 as x)
                SELECT x, x * 2 as y FROM inner_cte
            )
            VISUALISE FROM outer_cte
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_join_with_visualise_from() {
        let query = r#"
            WITH result AS (
                SELECT a.x, b.y FROM a CROSS JOIN b
            )
            VISUALISE FROM result
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_distinct_with_visualise_as() {
        let query = r#"
            SELECT DISTINCT x, y FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_with_visualise_as() {
        let query = r#"
            SELECT ALL x, y FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exists_subquery_with_visualise_as() {
        let query = r#"
            SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.id = a.id)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_not_exists_subquery_with_visualise_as() {
        let query = r#"
            SELECT * FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.id = a.id)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
    }

    // ========================================
    // Negative Test Cases - Should Error
    // ========================================

    #[test]
    fn test_error_create_with_select_and_visualise_from() {
        let query = r#"
            CREATE TABLE x AS SELECT 1;
            SELECT * FROM x
            VISUALISE FROM y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot use VISUALISE FROM"));
    }

    #[test]
    fn test_error_insert_with_select_and_visualise_from() {
        let query = r#"
            INSERT INTO x SELECT * FROM y;
            SELECT * FROM x
            VISUALISE FROM z
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_subquery_select_with_visualise_from() {
        let query = r#"
            SELECT * FROM (SELECT * FROM data)
            VISUALISE FROM other
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_join_select_with_visualise_from() {
        let query = r#"
            SELECT a.* FROM a JOIN b ON a.id = b.id
            VISUALISE FROM c
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
    }

    // ========================================
    // FILTER Clause Tests (Raw SQL)
    // ========================================

    #[test]
    fn test_filter_simple_comparison() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y FILTER value > 10
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "value > 10");
    }

    #[test]
    fn test_filter_equality() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y FILTER category = 'A'
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "category = 'A'");
    }

    #[test]
    fn test_filter_not_equal() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER status != 'inactive'
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "status != 'inactive'");
    }

    #[test]
    fn test_filter_less_than_or_equal() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER score <= 100
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "score <= 100");
    }

    #[test]
    fn test_filter_greater_than_or_equal() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER year >= 2020
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "year >= 2020");
    }

    #[test]
    fn test_filter_and_expression() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER value > 10 AND value < 100
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "value > 10 AND value < 100");
    }

    #[test]
    fn test_filter_or_expression() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER category = 'A' OR category = 'B'
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "category = 'A' OR category = 'B'");
    }

    #[test]
    fn test_filter_with_mapping_and_setting() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color SETTING size => 5 FILTER value > 50
            DRAW point SETTING fill => 'Chartreuse'
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();
        let layer = &specs[0].layers[0];

        // Check aesthetics
        assert_eq!(layer.mappings.len(), 3);
        assert!(layer.mappings.contains_key("x"));
        assert!(layer.mappings.contains_key("y"));
        assert!(layer.mappings.contains_key("color"));

        // Check parameters
        assert_eq!(layer.parameters.len(), 1);
        assert!(layer.parameters.contains_key("size"));

        // Check filter
        assert!(layer.filter.is_some());
        let filter = layer.filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "value > 50");

        // Check translation of colour name
        let layer = &specs[0].layers[1];
        assert!(layer.parameters.contains_key("fill"));

        if let ParameterValue::String(fill) = layer.parameters.get("fill").unwrap() {
            assert_eq!(fill, "#7fff00")
        } else {
            panic!("Wrong type of 'fill' parameter")
        }
    }

    #[test]
    fn test_filter_boolean_value() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER active = true
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "active = true");
    }

    #[test]
    fn test_filter_negative_number() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER temperature > -10
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        // Negative numbers are parsed as a single token with no space
        assert_eq!(filter.as_str(), "temperature > -10");
    }

    #[test]
    fn test_no_filter() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert!(specs[0].layers[0].filter.is_none());
    }

    #[test]
    fn test_multiple_layers_with_different_filters() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FILTER highlight = true
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        // First layer has no filter
        assert!(specs[0].layers[0].filter.is_none());

        // Second layer has filter
        assert!(specs[0].layers[1].filter.is_some());
        assert_eq!(
            specs[0].layers[1].filter.as_ref().unwrap().as_str(),
            "highlight = true"
        );
    }

    #[test]
    fn test_filter_column_comparison() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER start_date < end_date
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert_eq!(filter.as_str(), "start_date < end_date");
    }

    #[test]
    fn test_filter_complex_sql_expression() {
        // Test that complex SQL WHERE expressions are captured verbatim
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER category IN ('A', 'B', 'C') AND value BETWEEN 10 AND 100
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert!(filter.as_str().contains("IN"));
        assert!(filter.as_str().contains("BETWEEN"));
    }

    #[test]
    fn test_filter_like_expression() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x FILTER name LIKE '%test%'
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let filter = specs[0].layers[0].filter.as_ref().unwrap();
        assert!(filter.as_str().contains("LIKE"));
    }

    // ========================================
    // PARTITION BY Tests
    // ========================================

    #[test]
    fn test_partition_by_single_column() {
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line PARTITION BY category
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert_eq!(specs[0].layers[0].partition_by.len(), 1);
        assert_eq!(specs[0].layers[0].partition_by[0], "category");
    }

    #[test]
    fn test_partition_by_multiple_columns() {
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line PARTITION BY category, region
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert_eq!(specs[0].layers[0].partition_by.len(), 2);
        assert_eq!(specs[0].layers[0].partition_by[0], "category");
        assert_eq!(specs[0].layers[0].partition_by[1], "region");
    }

    #[test]
    fn test_partition_by_with_other_clauses() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING date AS x, value AS y SETTING opacity => 0.5 FILTER year > 2020 PARTITION BY category
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let layer = &specs[0].layers[0];
        assert_eq!(layer.partition_by.len(), 1);
        assert_eq!(layer.partition_by[0], "category");
        assert!(layer.filter.is_some());
        assert!(layer.parameters.contains_key("opacity"));
    }

    #[test]
    fn test_no_partition_by() {
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert!(specs[0].layers[0].partition_by.is_empty());
    }

    #[test]
    fn test_partition_by_case_insensitive() {
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line partition by category
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert_eq!(specs[0].layers[0].partition_by.len(), 1);
        assert_eq!(specs[0].layers[0].partition_by[0], "category");
    }

    // ========================================
    // ORDER BY Tests
    // ========================================

    #[test]
    fn test_order_by_single_column() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y ORDER BY x ASC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let order_by = specs[0].layers[0].order_by.as_ref().unwrap();
        assert_eq!(order_by.as_str(), "x ASC");
    }

    #[test]
    fn test_order_by_multiple_columns() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING x AS x, y AS y ORDER BY category, date DESC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let order_by = specs[0].layers[0].order_by.as_ref().unwrap();
        assert_eq!(order_by.as_str(), "category, date DESC");
    }

    #[test]
    fn test_order_by_with_nulls() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y ORDER BY date ASC NULLS FIRST
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let order_by = specs[0].layers[0].order_by.as_ref().unwrap();
        assert!(order_by.as_str().contains("NULLS FIRST"));
    }

    #[test]
    fn test_order_by_desc() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y ORDER BY value DESC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let order_by = specs[0].layers[0].order_by.as_ref().unwrap();
        assert_eq!(order_by.as_str(), "value DESC");
    }

    #[test]
    fn test_order_by_with_filter() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y FILTER x > 0 ORDER BY x ASC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let layer = &specs[0].layers[0];
        assert!(layer.filter.is_some());
        assert!(layer.order_by.is_some());
        assert_eq!(layer.filter.as_ref().unwrap().as_str(), "x > 0");
        assert_eq!(layer.order_by.as_ref().unwrap().as_str(), "x ASC");
    }

    #[test]
    fn test_order_by_with_partition_by() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING x AS x, y AS y PARTITION BY category ORDER BY date ASC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let layer = &specs[0].layers[0];
        assert_eq!(layer.partition_by.len(), 1);
        assert_eq!(layer.partition_by[0], "category");
        assert!(layer.order_by.is_some());
        assert_eq!(layer.order_by.as_ref().unwrap().as_str(), "date ASC");
    }

    #[test]
    fn test_order_by_with_all_clauses() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING date AS x, value AS y SETTING opacity => 0.5 FILTER year > 2020 PARTITION BY region ORDER BY date ASC, value DESC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let layer = &specs[0].layers[0];
        assert!(layer.parameters.contains_key("opacity"));
        assert!(layer.filter.is_some());
        assert_eq!(layer.partition_by.len(), 1);
        assert!(layer.order_by.is_some());
        assert_eq!(
            layer.order_by.as_ref().unwrap().as_str(),
            "date ASC, value DESC"
        );
    }

    #[test]
    fn test_no_order_by() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert!(specs[0].layers[0].order_by.is_none());
    }

    #[test]
    fn test_order_by_case_insensitive() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING x AS x, y AS y order by date asc
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert!(specs[0].layers[0].order_by.is_some());
    }

    #[test]
    fn test_multiple_layers_different_order_by() {
        let query = r#"
            VISUALISE
            DRAW line MAPPING x AS x, y AS y ORDER BY date ASC
            DRAW point MAPPING x AS x, y AS y ORDER BY value DESC
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        assert_eq!(
            specs[0].layers[0].order_by.as_ref().unwrap().as_str(),
            "date ASC"
        );
        assert_eq!(
            specs[0].layers[1].order_by.as_ref().unwrap().as_str(),
            "value DESC"
        );
    }

    // ========================================
    // Global Mapping Resolution Integration Tests
    // ========================================

    #[test]
    fn test_global_mapping_parsing() {
        let query = r#"
            VISUALISE date AS x, revenue AS y
            DRAW line
            DRAW point MAPPING region AS color
        "#;

        let specs = parse_test_query(query).unwrap();

        // Global mapping should have x and y
        assert_eq!(specs[0].global_mappings.aesthetics.len(), 2);
        assert!(specs[0].global_mappings.aesthetics.contains_key("x"));
        assert!(specs[0].global_mappings.aesthetics.contains_key("y"));
        assert!(!specs[0].global_mappings.wildcard);

        // Line layer should have no layer-specific aesthetics
        assert_eq!(specs[0].layers[0].mappings.len(), 0);

        // Point layer should have color from layer MAPPING
        assert_eq!(specs[0].layers[1].mappings.len(), 1);
        assert!(specs[0].layers[1].mappings.contains_key("color"));
    }

    #[test]
    fn test_implicit_global_mapping_parsing() {
        let query = r#"
            VISUALISE x, y
            DRAW point
        "#;

        let specs = parse_test_query(query).unwrap();

        // Implicit x, y become explicit mappings at parse time
        assert_eq!(specs[0].global_mappings.aesthetics.len(), 2);
        assert!(specs[0].global_mappings.aesthetics.contains_key("x"));
        assert!(specs[0].global_mappings.aesthetics.contains_key("y"));

        // Verify they map to columns of the same name
        let x_val = specs[0].global_mappings.aesthetics.get("x").unwrap();
        assert_eq!(x_val.column_name(), Some("x"));
        let y_val = specs[0].global_mappings.aesthetics.get("y").unwrap();
        assert_eq!(y_val.column_name(), Some("y"));
    }

    #[test]
    fn test_wildcard_global_mapping_parsing() {
        let query = r#"
            VISUALISE *
            DRAW point
        "#;

        let specs = parse_test_query(query).unwrap();

        // Wildcard flag should be set
        assert!(specs[0].global_mappings.wildcard);
        // No explicit aesthetics (wildcard expansion happens at execution time)
        assert!(specs[0].global_mappings.aesthetics.is_empty());
    }

    #[test]
    fn test_wildcard_with_explicit_mapping_parsing() {
        let query = r#"
            VISUALISE *, category AS fill
            DRAW bar
        "#;

        let specs = parse_test_query(query).unwrap();

        // Wildcard flag should be set
        assert!(specs[0].global_mappings.wildcard);
        // Plus explicit fill mapping
        assert_eq!(specs[0].global_mappings.aesthetics.len(), 1);
        assert!(specs[0].global_mappings.aesthetics.contains_key("fill"));
    }

    #[test]
    fn test_layer_wildcard_mapping_parsing() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING *
        "#;

        let specs = parse_test_query(query).unwrap();

        // Global mapping should be empty
        assert!(specs[0].global_mappings.is_empty());
        // Layer should have wildcard set
        assert!(specs[0].layers[0].mappings.wildcard);
    }

    #[test]
    fn test_layer_wildcard_with_explicit_parsing() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING *, 'red' AS color
        "#;

        let specs = parse_test_query(query).unwrap();

        // Layer should have wildcard set plus explicit color
        assert!(specs[0].layers[0].mappings.wildcard);
        assert_eq!(specs[0].layers[0].mappings.len(), 1);
        assert!(specs[0].layers[0].mappings.contains_key("color"));
    }

    // ========================================
    // Layer FROM Tests
    // ========================================

    #[test]
    fn test_layer_from_identifier() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y FROM my_cte
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);

        let layer = &specs[0].layers[0];
        assert!(layer.source.is_some());
        assert!(matches!(
            layer.source.as_ref(),
            Some(DataSource::Identifier(name)) if name == "my_cte"
        ));
    }

    #[test]
    fn test_layer_from_file_path() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y FROM 'data.csv'
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);

        let layer = &specs[0].layers[0];
        assert!(layer.source.is_some());
        assert!(matches!(
            layer.source.as_ref(),
            Some(DataSource::FilePath(path)) if path == "data.csv"
        ));
    }

    #[test]
    fn test_layer_from_empty_mapping() {
        // MAPPING FROM source (no aesthetics, inherit global)
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW point MAPPING FROM other_data
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);

        let layer = &specs[0].layers[0];
        assert!(layer.source.is_some());
        assert!(matches!(
            layer.source.as_ref(),
            Some(DataSource::Identifier(name)) if name == "other_data"
        ));
        // Layer should have no direct aesthetics (will inherit from global)
        assert!(layer.mappings.is_empty());
    }

    #[test]
    fn test_layer_without_from() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 1);

        let layer = &specs[0].layers[0];
        assert!(layer.source.is_none());
    }

    #[test]
    fn test_mixed_layers_with_and_without_from() {
        let query = r#"
            SELECT * FROM baseline
            VISUALISE
            DRAW line MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM comparison
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 2);

        // First layer uses global data (no FROM)
        assert!(specs[0].layers[0].source.is_none());

        // Second layer uses specific source
        assert!(specs[0].layers[1].source.is_some());
        assert!(matches!(
            specs[0].layers[1].source.as_ref(),
            Some(DataSource::Identifier(name)) if name == "comparison"
        ));
    }

    #[test]
    fn test_layer_from_with_cte() {
        let query = r#"
            WITH sales AS (SELECT date, revenue FROM transactions),
                 targets AS (SELECT date, goal FROM monthly_goals)
            VISUALISE
            DRAW line MAPPING date AS x, revenue AS y FROM sales
            DRAW line MAPPING date AS x, goal AS y FROM targets
        "#;

        let specs = parse_test_query(query).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layers.len(), 2);

        assert!(matches!(
            specs[0].layers[0].source.as_ref(),
            Some(DataSource::Identifier(name)) if name == "sales"
        ));
        assert!(matches!(
            specs[0].layers[1].source.as_ref(),
            Some(DataSource::Identifier(name)) if name == "targets"
        ));
    }
}
