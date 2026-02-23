//! Plot builder - converts tree-sitter CST to typed Plot
//!
//! Takes a tree-sitter parse tree and builds a typed Plot,
//! handling all the node types defined in the grammar.

use crate::plot::layer::geom::Geom;
use crate::plot::scale::{color_to_hex, is_color_aesthetic, is_facet_aesthetic, Transform};
use crate::plot::*;
use crate::{GgsqlError, Result};
use std::collections::HashMap;
use tree_sitter::Node;

use super::SourceTree;

// ============================================================================
// Basic Type Parsers
// ============================================================================

/// Extract 'name' and 'value' field nodes from an assignment-like node
///
/// Returns (name_node, value_node) without any interpretation.
/// Works for both patterns:
/// - `name => value` (SETTING, PROJECT, THEME, LABEL, RENAMING)
/// - `value AS name` (MAPPING explicit_mapping)
///
/// Caller is responsible for interpreting the nodes based on their context.
fn extract_name_value_nodes<'a>(node: &'a Node<'a>, context: &str) -> Result<(Node<'a>, Node<'a>)> {
    let name_node = node
        .child_by_field_name("name")
        .ok_or_else(|| GgsqlError::ParseError(format!("Missing 'name' field in {}", context)))?;

    let value_node = node
        .child_by_field_name("value")
        .ok_or_else(|| GgsqlError::ParseError(format!("Missing 'value' field in {}", context)))?;

    Ok((name_node, value_node))
}

/// Parse a string node, removing quotes and processing escape sequences
fn parse_string_node(node: &Node, source: &SourceTree) -> String {
    let text = source.get_text(node);
    let unquoted = text.trim_matches(|c| c == '\'' || c == '"');
    process_escape_sequences(unquoted)
}

/// Process escape sequences in a string (e.g., \n, \t, \\, \')
fn process_escape_sequences(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('\'') => result.push('\''),
                Some('"') => result.push('"'),
                Some(other) => {
                    // Unknown escape sequence - keep as-is
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'), // Trailing backslash
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Parse a number node into f64
fn parse_number_node(node: &Node, source: &SourceTree) -> Result<f64> {
    let text = source.get_text(node);
    text.parse::<f64>()
        .map_err(|e| GgsqlError::ParseError(format!("Failed to parse number '{}': {}", text, e)))
}

/// Parse a boolean node
fn parse_boolean_node(node: &Node, source: &SourceTree) -> bool {
    let text = source.get_text(node);
    text == "true"
}

/// Parse an array node into Vec<ArrayElement>
fn parse_array_node(node: &Node, source: &SourceTree) -> Result<Vec<ArrayElement>> {
    let mut values = Vec::new();

    // Find all array_element nodes
    let query = "(array_element) @elem";
    let array_elements = source.find_nodes(node, query);

    for array_element in array_elements {
        // array_element is a choice node, so it has exactly one child
        let elem_child = array_element.child(0).ok_or_else(|| {
            GgsqlError::ParseError("Invalid array_element: missing child".to_string())
        })?;

        let value = match elem_child.kind() {
            "string" => ArrayElement::String(parse_string_node(&elem_child, source)),
            "number" => ArrayElement::Number(parse_number_node(&elem_child, source)?),
            "boolean" => ArrayElement::Boolean(parse_boolean_node(&elem_child, source)),
            "null_literal" => ArrayElement::Null,
            _ => {
                return Err(GgsqlError::ParseError(format!(
                    "Invalid array element type: {}",
                    elem_child.kind()
                )));
            }
        };
        values.push(value);
    }

    Ok(values)
}

/// Parse a value node directly (string, number, boolean, array, or null)
fn parse_value_node(node: &Node, source: &SourceTree, context: &str) -> Result<ParameterValue> {
    match node.kind() {
        "string" => {
            let value = parse_string_node(node, source);
            Ok(ParameterValue::String(value))
        }
        "number" => {
            let num = parse_number_node(node, source)?;
            Ok(ParameterValue::Number(num))
        }
        "boolean" => {
            let bool_val = parse_boolean_node(node, source);
            Ok(ParameterValue::Boolean(bool_val))
        }
        "array" => {
            let values = parse_array_node(node, source)?;
            Ok(ParameterValue::Array(values))
        }
        "null_literal" => Ok(ParameterValue::Null),
        _ => Err(GgsqlError::ParseError(format!(
            "Unexpected {} value type: {}",
            context,
            node.kind()
        ))),
    }
}

/// Parse a data source node (identifier or string file path)
fn parse_data_source(node: &Node, source: &SourceTree) -> DataSource {
    match node.kind() {
        "string" => {
            let path = parse_string_node(node, source);
            DataSource::FilePath(path)
        }
        _ => {
            let text = source.get_text(node);
            DataSource::Identifier(text)
        }
    }
}

/// Parse a literal_value node into an AestheticValue::Literal
fn parse_literal_value(node: &Node, source: &SourceTree) -> Result<AestheticValue> {
    // literal_value is a choice(), so it has exactly one child
    let child = node.child(0).unwrap();
    let value = parse_value_node(&child, source, "literal")?;

    // Grammar ensures literals can't be arrays or nulls, but add safety check
    if matches!(value, ParameterValue::Array(_) | ParameterValue::Null) {
        return Err(GgsqlError::ParseError(
            "Arrays and null cannot be used as literal values in aesthetic mappings".to_string(),
        ));
    }

    Ok(AestheticValue::Literal(value))
}

// ============================================================================
// AST Building
// ============================================================================

/// Build a Plot struct from a tree-sitter parse tree
pub fn build_ast(source: &SourceTree) -> Result<Vec<Plot>> {
    let root = source.root();

    // Check if root is a query node
    if root.kind() != "query" {
        return Err(GgsqlError::ParseError(format!(
            "Expected 'query' root node, got '{}'",
            root.kind()
        )));
    }

    // Extract SQL portion node (if exists)
    let query = "(sql_portion) @sql";
    let sql_portion_node = source.find_node(&root, query);

    // Check if last SQL statement is SELECT
    let last_is_select = if let Some(sql_node) = sql_portion_node {
        check_last_statement_is_select(&sql_node, source)
    } else {
        false
    };

    // Find all visualise_statement nodes
    let query = "(visualise_statement) @viz";
    let viz_nodes = source.find_nodes(&root, query);

    let mut specs = Vec::new();
    for viz_node in viz_nodes {
        let spec = build_visualise_statement(&viz_node, source)?;

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

    if specs.is_empty() {
        return Err(GgsqlError::ParseError(
            "No VISUALISE statements found in query".to_string(),
        ));
    }

    Ok(specs)
}

/// Build a single Plot from a visualise_statement node
fn build_visualise_statement(node: &Node, source: &SourceTree) -> Result<Plot> {
    let mut spec = Plot::new();

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
                spec.global_mappings = parse_mapping(&child, source)?;
            }
            "wildcard_mapping" => {
                // Handle standalone wildcard (*) mapping
                spec.global_mappings.wildcard = true;
            }
            "from_clause" => {
                // Extract the 'table' field from table_ref (grammar: FROM table_ref)
                let query = "(table_ref table: (_) @table)";
                if let Some(table_node) = source.find_node(&child, query) {
                    spec.source = Some(parse_data_source(&table_node, source));
                }
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

/// Process a visualization clause node
fn process_viz_clause(node: &Node, source: &SourceTree, spec: &mut Plot) -> Result<()> {
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
            "project_clause" => {
                spec.project = Some(build_project(&child, source)?);
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

// ============================================================================
// Mapping Building
// ============================================================================

/// Parse mapping elements from a node containing mappings and return a Mappings struct
/// Used by global_mapping (VISUALISE), mapping_clause (MAPPING), and remapping_clause (REMAPPING)
/// Tree-sitter recursively finds mapping_element nodes within the nested mapping_list
fn parse_mapping(node: &Node, source: &SourceTree) -> Result<Mappings> {
    let mut mappings = Mappings::new();

    // Find all mapping_element nodes (recursively searches within mapping_list if present)
    let query = "(mapping_element) @elem";
    let mapping_nodes = source.find_nodes(node, query);

    for mapping_node in mapping_nodes {
        parse_mapping_element(&mapping_node, source, &mut mappings)?;
    }

    Ok(mappings)
}

/// Parse a mapping_element: wildcard, explicit, or implicit mapping
/// Shared by both global (VISUALISE) and layer (MAPPING) mappings
fn parse_mapping_element(node: &Node, source: &SourceTree, mappings: &mut Mappings) -> Result<()> {
    // mapping_element is a choice node, so it has exactly one child
    let child = node.child(0).ok_or_else(|| {
        GgsqlError::ParseError("Invalid mapping_element: missing child".to_string())
    })?;

    match child.kind() {
        "wildcard_mapping" => {
            mappings.wildcard = true;
        }
        "explicit_mapping" => {
            let (aesthetic, value) = parse_explicit_mapping(&child, source)?;
            mappings.insert(normalise_aes_name(&aesthetic), value);
        }
        "implicit_mapping" | "identifier" => {
            let name = source.get_text(&child);
            mappings.insert(
                normalise_aes_name(&name),
                AestheticValue::standard_column(&name),
            );
        }
        _ => {
            return Err(GgsqlError::ParseError(format!(
                "Invalid mapping_element child type: {}",
                child.kind()
            )));
        }
    }
    Ok(())
}

/// Parse an explicit_mapping node (value AS aesthetic)
/// Returns (aesthetic_name, value)
fn parse_explicit_mapping(node: &Node, source: &SourceTree) -> Result<(String, AestheticValue)> {
    // Extract name and value nodes using field-based queries
    let (name_node, value_node) = extract_name_value_nodes(node, "explicit mapping")?;

    // Parse aesthetic name
    let aesthetic = source.get_text(&name_node);

    // Parse value (mapping_value has exactly one child: column_reference or literal_value)
    let value_child = value_node.child(0).ok_or_else(|| {
        GgsqlError::ParseError("Invalid explicit mapping: missing value".to_string())
    })?;

    let value = match value_child.kind() {
        "column_reference" => {
            // column_reference is just an identifier wrapper, get its text directly
            AestheticValue::standard_column(source.get_text(&value_child))
        }
        "literal_value" => parse_literal_value(&value_child, source)?,
        _ => {
            return Err(GgsqlError::ParseError(format!(
                "Invalid explicit mapping value type: {}",
                value_child.kind()
            )));
        }
    };

    Ok((aesthetic, value))
}

// ============================================================================
// Layer Building
// ============================================================================

/// Build a Layer from a draw_clause node
/// Syntax: DRAW geom [MAPPING col AS x, ... [FROM source]] [REMAPPING stat AS aes, ...] [SETTING param => val, ...] [PARTITION BY col, ...] [FILTER condition]
fn build_layer(node: &Node, source: &SourceTree) -> Result<Layer> {
    let mut geom = Geom::point(); // default
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
                let geom_text = source.get_text(&child);
                geom = parse_geom_type(&geom_text)?;
            }
            "mapping_clause" => {
                // Parse aesthetic mappings and optional data source
                aesthetics = parse_mapping(&child, source)?;
                layer_source = child
                    .child_by_field_name("layer_source")
                    .map(|src| parse_data_source(&src, source));
            }
            "remapping_clause" => {
                // Parse stat result remappings (same syntax as mapping_clause)
                remappings = parse_mapping(&child, source)?;
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

/// Parse a setting_clause: SETTING param => value, ...
fn parse_setting_clause(
    node: &Node,
    source: &SourceTree,
) -> Result<HashMap<String, ParameterValue>> {
    let mut parameters = HashMap::new();

    // Find all parameter_assignment nodes
    let query = "(parameter_assignment) @param";
    let param_nodes = source.find_nodes(node, query);

    for param_node in param_nodes {
        let (param, mut value) = parse_parameter_assignment(&param_node, source)?;
        if is_color_aesthetic(&param) {
            if let ParameterValue::String(color) = value {
                value =
                    ParameterValue::String(color_to_hex(&color).map_err(GgsqlError::ParseError)?);
            }
        }
        parameters.insert(param, value);
    }

    Ok(parameters)
}

/// Parse a parameter_assignment: param => value
fn parse_parameter_assignment(
    node: &Node,
    source: &SourceTree,
) -> Result<(String, ParameterValue)> {
    // Extract name and value nodes using field-based queries
    let (name_node, value_node) = extract_name_value_nodes(node, "parameter assignment")?;

    // Parse parameter name (parameter_name is just an identifier)
    let param_name = source.get_text(&name_node);

    // Parse parameter value (parameter_value wraps the actual value node)
    let param_value = if let Some(value_child) = value_node.child(0) {
        parse_value_node(&value_child, source, "parameter")?
    } else {
        return Err(GgsqlError::ParseError(
            "Invalid parameter assignment: empty parameter_value".to_string(),
        ));
    };

    Ok((param_name, param_value))
}

/// Parse a partition_clause: PARTITION BY col1, col2, ...
fn parse_partition_clause(node: &Node, source: &SourceTree) -> Result<Vec<String>> {
    let query = r#"
        (partition_columns
          (identifier) @col)
    "#;
    Ok(source.find_texts(node, query))
}

/// Parse a filter_clause: FILTER <raw SQL expression>
///
/// Extracts the raw SQL text from the filter_expression and returns it verbatim.
/// This allows any valid SQL WHERE expression to be passed to the database backend.
fn parse_filter_clause(node: &Node, source: &SourceTree) -> Result<SqlExpression> {
    let query = "(filter_expression) @expr";

    if let Some(filter_text) = source.find_text(node, query) {
        Ok(SqlExpression::new(filter_text.trim().to_string()))
    } else {
        Err(GgsqlError::ParseError(
            "Could not find filter expression in filter clause".to_string(),
        ))
    }
}

/// Parse an order_clause: ORDER BY date ASC, value DESC
fn parse_order_clause(node: &Node, source: &SourceTree) -> Result<SqlExpression> {
    let query = "(order_expression) @expr";

    if let Some(order_text) = source.find_text(node, query) {
        Ok(SqlExpression::new(order_text.trim().to_string()))
    } else {
        Err(GgsqlError::ParseError(
            "Could not find order expression in order clause".to_string(),
        ))
    }
}

/// Parse a geom_type node text into a Geom
fn parse_geom_type(text: &str) -> Result<Geom> {
    match text.to_lowercase().as_str() {
        "point" => Ok(Geom::point()),
        "line" => Ok(Geom::line()),
        "path" => Ok(Geom::path()),
        "bar" => Ok(Geom::bar()),
        "area" => Ok(Geom::area()),
        "tile" => Ok(Geom::tile()),
        "polygon" => Ok(Geom::polygon()),
        "ribbon" => Ok(Geom::ribbon()),
        "histogram" => Ok(Geom::histogram()),
        "density" => Ok(Geom::density()),
        "smooth" => Ok(Geom::smooth()),
        "boxplot" => Ok(Geom::boxplot()),
        "violin" => Ok(Geom::violin()),
        "text" => Ok(Geom::text()),
        "label" => Ok(Geom::label()),
        "segment" => Ok(Geom::segment()),
        "arrow" => Ok(Geom::arrow()),
        "hline" => Ok(Geom::hline()),
        "vline" => Ok(Geom::vline()),
        "abline" => Ok(Geom::abline()),
        "errorbar" => Ok(Geom::errorbar()),
        _ => Err(GgsqlError::ParseError(format!(
            "Unknown geom type: {}",
            text
        ))),
    }
}

// ============================================================================
// Scale Building
// ============================================================================

/// Build a Scale from a scale_clause node
/// SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]
fn build_scale(node: &Node, source: &SourceTree) -> Result<Scale> {
    let mut aesthetic = String::new();
    let mut scale_type: Option<ScaleType> = None;
    let mut input_range: Option<Vec<ArrayElement>> = None;
    let mut explicit_input_range = false;
    let mut output_range: Option<OutputRange> = None;
    let mut transform: Option<Transform> = None;
    let mut explicit_transform = false;
    let mut properties = HashMap::new();
    let mut label_mapping: Option<HashMap<String, Option<String>>> = None;
    let mut label_template = "{}".to_string();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "SCALE" | "SETTING" | "=>" | "," | "FROM" | "TO" | "VIA" | "RENAMING" => continue, // Skip keywords
            "scale_type_identifier" => {
                // Parse scale type: CONTINUOUS, DISCRETE, BINNED, DATE, DATETIME
                let type_text = source.get_text(&child);
                scale_type = Some(parse_scale_type_identifier(&type_text)?);
            }
            "aesthetic_name" => {
                aesthetic = normalise_aes_name(&source.get_text(&child));
            }
            "scale_from_clause" => {
                // Parse FROM [array] -> input_range
                input_range = Some(parse_scale_from_clause(&child, source)?);
                // Mark as explicit input range (user specified FROM clause)
                explicit_input_range = true;
            }
            "scale_to_clause" => {
                // Parse TO [array | identifier] -> output_range
                output_range = Some(parse_scale_to_clause(&child, source)?);
            }
            "scale_via_clause" => {
                // Parse VIA identifier -> transform
                transform = Some(parse_scale_via_clause(&child, source)?);
                // Mark as explicit transform (user specified VIA clause)
                explicit_transform = true;
            }
            "setting_clause" => {
                // Reuse existing setting_clause parser
                properties = parse_setting_clause(&child, source)?;
            }
            "scale_renaming_clause" => {
                // Parse RENAMING 'A' => 'Alpha', 'B' => 'Beta', * => '{} units'
                let (mappings, template) = parse_scale_renaming_clause(&child, source)?;
                if !mappings.is_empty() {
                    label_mapping = Some(mappings);
                }
                label_template = template;
            }
            _ => {}
        }
    }

    if aesthetic.is_empty() {
        return Err(GgsqlError::ParseError(
            "Scale clause missing aesthetic name".to_string(),
        ));
    }

    // Replace colour palettes by their hex codes in output_range
    if is_color_aesthetic(&aesthetic) {
        if let Some(OutputRange::Array(ref elements)) = output_range {
            let hex_codes: Vec<ArrayElement> = elements
                .iter()
                .map(|elem| {
                    if let ArrayElement::String(color) = elem {
                        color_to_hex(color).map(ArrayElement::String)
                    } else {
                        Ok(elem.clone())
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(GgsqlError::ParseError)?;
            output_range = Some(OutputRange::Array(hex_codes));
        }
    }

    // Validate facet aesthetics cannot have output ranges (TO clause)
    if is_facet_aesthetic(&aesthetic) && output_range.is_some() {
        return Err(GgsqlError::ValidationError(format!(
            "SCALE {}: facet variables cannot have output ranges (TO clause)",
            aesthetic
        )));
    }

    Ok(Scale {
        aesthetic,
        scale_type,
        input_range,
        explicit_input_range,
        output_range,
        transform,
        explicit_transform,
        properties,
        resolved: false,
        label_mapping,
        label_template,
    })
}

/// Parse scale type identifier (CONTINUOUS, DISCRETE, BINNED, ORDINAL, IDENTITY)
fn parse_scale_type_identifier(text: &str) -> Result<ScaleType> {
    match text.to_lowercase().as_str() {
        "continuous" => Ok(ScaleType::continuous()),
        "discrete" => Ok(ScaleType::discrete()),
        "binned" => Ok(ScaleType::binned()),
        "ordinal" => Ok(ScaleType::ordinal()),
        "identity" => Ok(ScaleType::identity()),
        _ => Err(GgsqlError::ParseError(format!(
            "Unknown scale type: '{}'. Valid types: continuous, discrete, binned, ordinal, identity",
            text
        ))),
    }
}

/// Parse FROM clause: FROM [array]
fn parse_scale_from_clause(node: &Node, source: &SourceTree) -> Result<Vec<ArrayElement>> {
    let query = "(array) @arr";
    let array_node = source
        .find_node(node, query)
        .ok_or_else(|| GgsqlError::ParseError("FROM clause missing array".to_string()))?;
    parse_array_node(&array_node, source)
}

/// Parse TO clause: TO [array | identifier]
fn parse_scale_to_clause(node: &Node, source: &SourceTree) -> Result<OutputRange> {
    // Try array first
    let array_query = "(array) @arr";
    if let Some(array_node) = source.find_node(node, array_query) {
        let elements = parse_array_node(&array_node, source)?;
        return Ok(OutputRange::Array(elements));
    }

    // Try identifier (palette name)
    let ident_query = "[(identifier) (bare_identifier) (quoted_identifier)] @id";
    if let Some(ident_node) = source.find_node(node, ident_query) {
        let palette_name = source.get_text(&ident_node);
        return Ok(OutputRange::Palette(palette_name));
    }

    Err(GgsqlError::ParseError(
        "TO clause must contain either an array or identifier".to_string(),
    ))
}

/// Parse VIA clause: VIA identifier
fn parse_scale_via_clause(node: &Node, source: &SourceTree) -> Result<Transform> {
    let query = "[(identifier) (bare_identifier) (quoted_identifier)] @id";
    let ident_node = source.find_node(node, query).ok_or_else(|| {
        GgsqlError::ParseError("VIA clause missing transform identifier".to_string())
    })?;

    let transform_name = source.get_text(&ident_node);
    Transform::from_name(&transform_name).ok_or_else(|| {
        GgsqlError::ParseError(format!(
            "Unknown transform: '{}'. Valid transforms are: {}",
            transform_name,
            crate::plot::scale::ALL_TRANSFORM_NAMES.join(", ")
        ))
    })
}

/// Parse RENAMING clause: RENAMING 'A' => 'Alpha', 'B' => 'Beta', 'internal' => NULL, * => '{} units'
///
/// Returns a tuple of:
/// - HashMap where: Key = original value, Value = Some(label) or None for suppressed labels
/// - Template string for wildcard mappings (* => '...'), defaults to "{}"
fn parse_scale_renaming_clause(
    node: &Node,
    source: &SourceTree,
) -> Result<(HashMap<String, Option<String>>, String)> {
    let mut mappings = HashMap::new();
    let mut template = "{}".to_string();

    // Find all renaming_assignment nodes
    let query = "(renaming_assignment) @assign";
    let assignment_nodes = source.find_nodes(node, query);

    for assignment_node in assignment_nodes {
        // Extract name and value nodes using field-based queries
        let (name_node, value_node) = extract_name_value_nodes(&assignment_node, "scale renaming")?;

        // Check if 'name' is a wildcard
        let is_wildcard = name_node.kind() == "*";

        // Parse 'name' (from) value - wildcards, strings need unquoting, numbers are raw
        let from_value = match name_node.kind() {
            "*" => "*".to_string(),
            "string" => parse_string_node(&name_node, source),
            "number" => source.get_text(&name_node),
            "null_literal" => "null".to_string(), // null key for renaming null values
            _ => {
                return Err(GgsqlError::ParseError(format!(
                    "Invalid 'from' type in scale renaming: {}",
                    name_node.kind()
                )));
            }
        };

        // Parse 'value' (to) - string or NULL
        let to_value: Option<String> = match value_node.kind() {
            "string" => Some(parse_string_node(&value_node, source)),
            "null_literal" => None, // NULL suppresses the label
            _ => {
                return Err(GgsqlError::ParseError(format!(
                    "Invalid 'to' type in scale renaming: {}",
                    value_node.kind()
                )));
            }
        };

        if is_wildcard {
            // Wildcard: * => 'template'
            if let Some(tmpl) = to_value {
                template = tmpl;
            }
        } else {
            // Explicit mapping: 'A' => 'Alpha' or '10' => 'Ten'
            mappings.insert(from_value, to_value);
        }
    }

    Ok((mappings, template))
}

// ============================================================================
// Facet Building
// ============================================================================

/// Build a Facet from a facet_clause node
///
/// FACET vars [BY vars] [SETTING ...]
/// - Single variable = wrap layout (no WRAP keyword needed)
/// - BY clause = grid layout
fn build_facet(node: &Node, source: &SourceTree) -> Result<Facet> {
    let mut row_vars = Vec::new();
    let mut column_vars = Vec::new();
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    let mut next_vars_are_cols = false;

    for child in node.children(&mut cursor) {
        match child.kind() {
            "FACET" => continue,
            "facet_by" => {
                next_vars_are_cols = true;
            }
            "facet_vars" => {
                // Parse list of variable names
                let vars = parse_facet_vars(&child, source)?;
                if next_vars_are_cols {
                    column_vars = vars;
                } else {
                    row_vars = vars;
                }
            }
            "setting_clause" => {
                // Reuse existing setting_clause parser
                properties = parse_setting_clause(&child, source)?;
            }
            _ => {}
        }
    }

    // Determine layout variant: if column_vars is empty, it's a wrap layout
    let layout = if column_vars.is_empty() {
        FacetLayout::Wrap {
            variables: row_vars,
        }
    } else {
        FacetLayout::Grid {
            row: row_vars,
            column: column_vars,
        }
    };

    Ok(Facet {
        layout,
        properties,
        resolved: false,
    })
}

/// Parse facet variables from a facet_vars node
fn parse_facet_vars(node: &Node, source: &SourceTree) -> Result<Vec<String>> {
    let query = "(identifier) @var";
    Ok(source.find_texts(node, query))
}

// ============================================================================
// Project Building
// ============================================================================

/// Build a Projection from a project_clause node
fn build_project(node: &Node, source: &SourceTree) -> Result<Projection> {
    let mut coord = Coord::cartesian();
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "PROJECT" | "SETTING" | "=>" | "," => continue,
            "project_type" => {
                coord = parse_coord(&child, source)?;
            }
            "project_properties" => {
                // Find all project_property nodes
                let query = "(project_property) @prop";
                let prop_nodes = source.find_nodes(&child, query);

                for prop_node in prop_nodes {
                    let (prop_name, prop_value) =
                        parse_single_project_property(&prop_node, source)?;
                    properties.insert(prop_name, prop_value);
                }
            }
            _ => {}
        }
    }

    // Validate properties for this coord type
    validate_project_properties(&coord, &properties)?;

    Ok(Projection { coord, properties })
}

/// Parse a single project_property node into (name, value)
fn parse_single_project_property(
    node: &Node,
    source: &SourceTree,
) -> Result<(String, ParameterValue)> {
    // Extract name and value nodes using field-based queries
    let (name_node, value_node) = extract_name_value_nodes(node, "project property")?;

    // Parse property name (can be a literal like 'xlim' or an aesthetic_name)
    let prop_name = source.get_text(&name_node);

    // Parse property value based on its type
    let prop_value = match value_node.kind() {
        "string" | "number" | "boolean" | "array" => {
            parse_value_node(&value_node, source, "project property")?
        }
        "identifier" => {
            // identifiers can be property values (e.g., theta => y)
            ParameterValue::String(source.get_text(&value_node))
        }
        _ => {
            return Err(GgsqlError::ParseError(format!(
                "Invalid project property value type: {}",
                value_node.kind()
            )));
        }
    };

    Ok((prop_name, prop_value))
}

/// Validate that properties are valid for the given coord type
fn validate_project_properties(
    coord: &Coord,
    properties: &HashMap<String, ParameterValue>,
) -> Result<()> {
    coord
        .resolve_properties(properties)
        .map_err(GgsqlError::ParseError)?;
    Ok(())
}

/// Parse coord type from a project_type node
fn parse_coord(node: &Node, source: &SourceTree) -> Result<Coord> {
    let text = source.get_text(node);
    match text.to_lowercase().as_str() {
        "cartesian" => Ok(Coord::cartesian()),
        "polar" => Ok(Coord::polar()),
        "flip" => Ok(Coord::flip()),
        _ => Err(GgsqlError::ParseError(format!(
            "Unknown coord type: {}",
            text
        ))),
    }
}

// ============================================================================
// Labels Building
// ============================================================================

/// Build Labels from a label_clause node
fn build_labels(node: &Node, source: &SourceTree) -> Result<Labels> {
    let mut labels = HashMap::new();

    // Find all label_assignment nodes
    let query = "(label_assignment) @label";
    let label_nodes = source.find_nodes(node, query);

    for label_node in label_nodes {
        // Extract name and value nodes using field-based queries
        let (name_node, value_node) = extract_name_value_nodes(&label_node, "label assignment")?;

        // Parse label type (name)
        let label_type = source.get_text(&name_node);

        // Parse label value (must be a string)
        let label_value = match value_node.kind() {
            "string" => parse_string_node(&value_node, source),
            _ => {
                return Err(GgsqlError::ParseError(format!(
                    "Label '{}' must have a string value, got: {}",
                    label_type,
                    value_node.kind()
                )));
            }
        };

        labels.insert(label_type, label_value);
    }

    Ok(Labels { labels })
}

// ============================================================================
// Theme Building
// ============================================================================

/// Build a Theme from a theme_clause node
fn build_theme(node: &Node, source: &SourceTree) -> Result<Theme> {
    let mut style: Option<String> = None;
    let mut properties = HashMap::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "THEME" | "SETTING" | "=>" | "," => continue,
            "theme_name" => {
                style = Some(source.get_text(&child));
            }
            "theme_property" => {
                // Parse theme property: name => value using field-based queries
                let (name_node, value_node) = extract_name_value_nodes(&child, "theme property")?;

                // Parse property name
                let prop_name = source.get_text(&name_node);

                // Parse property value
                let prop_value = match value_node.kind() {
                    "string" | "number" | "boolean" => {
                        parse_value_node(&value_node, source, "theme property")?
                    }
                    _ => {
                        return Err(GgsqlError::ParseError(format!(
                            "Invalid theme property value type: {}",
                            value_node.kind()
                        )));
                    }
                };

                properties.insert(prop_name, prop_value);
            }
            _ => {}
        }
    }

    Ok(Theme { style, properties })
}

// ============================================================================
// Validation & Utilities
// ============================================================================

/// Check if the last SQL statement in sql_portion is a SELECT statement
fn check_last_statement_is_select(sql_portion_node: &Node, source: &SourceTree) -> bool {
    // Find all sql_statement nodes and get the last one (can use query for this)
    let query = "(sql_statement) @stmt";
    let statements = source.find_nodes(sql_portion_node, query);
    let last_statement = statements.last();

    // Check if last statement is or ends with a SELECT
    // But we need to check direct children only, not recursive
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
    // Need to check direct children only, not recursive search
    // A trailing SELECT means there's a select_statement after cte_definition at the same level
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

pub fn normalise_aes_name(name: &str) -> String {
    match name {
        "col" | "colour" => "color".to_string(),
        _ => name.to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_test_query(query: &str) -> Result<Vec<Plot>> {
        // Create SourceTree which parses and validates in one step
        let source = SourceTree::new(query)?;
        source.validate()?;

        build_ast(&source)
    }

    // ========================================
    // PROJECT Property Validation Tests
    // ========================================

    #[test]
    fn test_project_cartesian_invalid_property_theta() {
        let query = r#"
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            PROJECT cartesian SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'theta' not valid for cartesian"));
    }

    #[test]
    fn test_project_flip_invalid_property_theta() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            PROJECT flip SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Property 'theta' not valid for flip"));
    }

    #[test]
    fn test_project_polar_valid_theta() {
        let query = r#"
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
            PROJECT polar SETTING theta => y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());
        let specs = result.unwrap();

        let project = specs[0].project.as_ref().unwrap();
        assert_eq!(project.coord.coord_kind(), CoordKind::Polar);
        assert!(project.properties.contains_key("theta"));
    }

    // ========================================
    // Case Insensitive Keywords Tests
    // ========================================

    #[test]
    fn test_case_insensitive_keywords_lowercase() {
        let query = r#"
            visualise
            draw point MAPPING x AS x, y AS y
            project cartesian
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
        assert!(specs[0].project.is_some());
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
    fn test_visualise_from_file_path_quote_parquet() {
        let query = r#"
            VISUALISE FROM 'data/sales.parquet'
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
    fn test_visualise_from_file_path_double_quote_parquet() {
        let query = r#"
            VISUALISE FROM "data/sales.parquet"
            DRAW bar MAPPING region AS x, total AS y
        "#;

        let result = parse_test_query(query);
        assert!(result.is_ok());

        let specs = result.unwrap();
        // Source should be stored as identifier,
        // duckdb accepts this to indicate reading from file
        assert_eq!(
            specs[0].source,
            Some(DataSource::Identifier("\"data/sales.parquet\"".to_string()))
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
        // color should expand into stroke and fill
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

    #[test]
    fn test_colour_scale_hex_code_conversion() {
        let query = r#"
          VISUALISE foo AS x
          SCALE color TO ['rgb(0, 0, 255)', 'green', '#FF0000']
        "#;
        let specs = parse_test_query(query).unwrap();

        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);

        // Check output_range instead of properties.palette
        let output_range = &scales[0].output_range;
        assert!(output_range.is_some());
        let output_range = output_range.as_ref().unwrap();

        let mut ok = false;
        if let OutputRange::Array(elems) = output_range {
            ok = matches!(&elems[0], ArrayElement::String(color) if color == "#0000ff");
            ok = ok && matches!(&elems[1], ArrayElement::String(color) if color == "#008000");
            ok = ok && matches!(&elems[2], ArrayElement::String(color) if color == "#ff0000");
        }
        assert!(ok);
        eprintln!("{:?}", output_range);
    }

    // ========================================
    // Null in Scale Input Range Tests
    // ========================================

    #[test]
    fn test_scale_from_with_null_max() {
        // SCALE x FROM [0, null] - explicit min, infer max
        let query = r#"
            VISUALISE x, y
            DRAW point
            SCALE x FROM [0, null]
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "x");

        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 2);
        assert!(matches!(&input_range[0], ArrayElement::Number(n) if *n == 0.0));
        assert!(matches!(&input_range[1], ArrayElement::Null));
    }

    #[test]
    fn test_scale_from_with_null_min() {
        // SCALE x FROM [null, 100] - infer min, explicit max
        let query = r#"
            VISUALISE x, y
            DRAW point
            SCALE x FROM [null, 100]
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);

        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 2);
        assert!(matches!(&input_range[0], ArrayElement::Null));
        assert!(matches!(&input_range[1], ArrayElement::Number(n) if *n == 100.0));
    }

    #[test]
    fn test_scale_from_with_both_nulls() {
        // SCALE x FROM [null, null] - infer both (same as no FROM clause)
        let query = r#"
            VISUALISE x, y
            DRAW point
            SCALE x FROM [null, null]
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);

        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 2);
        assert!(matches!(&input_range[0], ArrayElement::Null));
        assert!(matches!(&input_range[1], ArrayElement::Null));
    }

    #[test]
    fn test_scale_from_with_null_case_insensitive() {
        // NULL should be case-insensitive
        let query = r#"
            VISUALISE x, y
            DRAW point
            SCALE x FROM [0, NULL]
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        let input_range = scales[0].input_range.as_ref().unwrap();
        assert!(matches!(&input_range[1], ArrayElement::Null));
    }

    #[test]
    fn test_scale_from_with_null() {
        // Scale with partial input range: explicit start, infer end
        // Note: DATE/DATETIME are no longer scale types - temporal handling is done
        // via transforms that are automatically inferred from column data types
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line
            SCALE x FROM ['2024-01-01', null]
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);

        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 2);
        assert!(matches!(&input_range[0], ArrayElement::String(s) if s == "2024-01-01"));
        assert!(matches!(&input_range[1], ArrayElement::Null));
    }

    #[test]
    fn test_scale_via_date_transform() {
        // Explicit date transform via VIA clause
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line
            SCALE x VIA date
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "x");
        assert!(scales[0].transform.is_some());
        assert_eq!(scales[0].transform.as_ref().unwrap().name(), "date");
    }

    #[test]
    fn test_scale_via_integer_transform() {
        // Explicit integer transform via VIA clause
        let query = r#"
            VISUALISE val AS x, count AS y
            DRAW point
            SCALE x VIA integer
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "x");
        assert!(scales[0].transform.is_some());
        assert_eq!(scales[0].transform.as_ref().unwrap().name(), "integer");
    }

    #[test]
    fn test_scale_via_int_alias() {
        // Integer transform using 'int' alias
        let query = r#"
            VISUALISE val AS x, count AS y
            DRAW point
            SCALE x VIA int
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert!(scales[0].transform.is_some());
        assert_eq!(scales[0].transform.as_ref().unwrap().name(), "integer");
    }

    #[test]
    fn test_scale_via_bigint_alias() {
        // Integer transform using 'bigint' alias
        let query = r#"
            VISUALISE val AS x, count AS y
            DRAW point
            SCALE x VIA bigint
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert!(scales[0].transform.is_some());
        assert_eq!(scales[0].transform.as_ref().unwrap().name(), "integer");
    }

    // ========================================
    // RENAMING clause tests
    // ========================================

    #[test]
    fn test_scale_renaming_basic() {
        // Basic RENAMING clause with string keys
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW bar
            SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "x");

        let label_mapping = scales[0].label_mapping.as_ref().unwrap();
        assert_eq!(label_mapping.len(), 2);
        assert_eq!(label_mapping.get("A"), Some(&Some("Alpha".to_string())));
        assert_eq!(label_mapping.get("B"), Some(&Some("Beta".to_string())));
    }

    #[test]
    fn test_scale_renaming_with_null() {
        // RENAMING with NULL to suppress labels
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW bar
            SCALE DISCRETE x RENAMING 'internal' => NULL, 'visible' => 'Shown'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        let label_mapping = scales[0].label_mapping.as_ref().unwrap();

        assert_eq!(label_mapping.get("internal"), Some(&None)); // NULL -> None
        assert_eq!(
            label_mapping.get("visible"),
            Some(&Some("Shown".to_string()))
        );
    }

    #[test]
    fn test_scale_renaming_with_numeric_keys() {
        // RENAMING with numeric keys (for binned scales)
        let query = r#"
            VISUALISE temp AS x, count AS y
            DRAW bar
            SCALE BINNED x RENAMING 0 => '0-10', 10 => '10-20', 20 => '20-30'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        let label_mapping = scales[0].label_mapping.as_ref().unwrap();

        assert_eq!(label_mapping.len(), 3);
        assert_eq!(label_mapping.get("0"), Some(&Some("0-10".to_string())));
        assert_eq!(label_mapping.get("10"), Some(&Some("10-20".to_string())));
        assert_eq!(label_mapping.get("20"), Some(&Some("20-30".to_string())));
    }

    #[test]
    fn test_scale_renaming_for_color_legend() {
        // RENAMING for color legend labels
        let query = r#"
            VISUALISE x, y, category AS color
            DRAW point
            SCALE DISCRETE color RENAMING 'cat_a' => 'Category A', 'cat_b' => 'Category B'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "color");

        let label_mapping = scales[0].label_mapping.as_ref().unwrap();
        assert_eq!(
            label_mapping.get("cat_a"),
            Some(&Some("Category A".to_string()))
        );
        assert_eq!(
            label_mapping.get("cat_b"),
            Some(&Some("Category B".to_string()))
        );
    }

    #[test]
    fn test_scale_renaming_with_setting() {
        // RENAMING combined with SETTING
        let query = r#"
            VISUALISE x, y
            DRAW bar
            SCALE DISCRETE x SETTING reverse => true RENAMING 'A' => 'First', 'B' => 'Second'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        // Check SETTING was parsed
        assert_eq!(
            scales[0].properties.get("reverse"),
            Some(&ParameterValue::Boolean(true))
        );

        // Check RENAMING was parsed
        let label_mapping = scales[0].label_mapping.as_ref().unwrap();
        assert_eq!(label_mapping.get("A"), Some(&Some("First".to_string())));
        assert_eq!(label_mapping.get("B"), Some(&Some("Second".to_string())));
    }

    #[test]
    fn test_scale_renaming_with_from_to() {
        // RENAMING combined with FROM and TO clauses
        let query = r#"
            VISUALISE x, y, cat AS color
            DRAW point
            SCALE DISCRETE color FROM ['A', 'B'] TO ['red', 'blue']
                RENAMING 'A' => 'Option A', 'B' => 'Option B'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        // Check FROM was parsed
        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 2);

        // Check TO was parsed
        assert!(scales[0].output_range.is_some());

        // Check RENAMING was parsed
        let label_mapping = scales[0].label_mapping.as_ref().unwrap();
        assert_eq!(label_mapping.get("A"), Some(&Some("Option A".to_string())));
    }

    #[test]
    fn test_scale_renaming_wildcard_template() {
        // Wildcard template for label generation
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW point
            SCALE CONTINUOUS x RENAMING * => '{} units'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        // Check label_template was parsed
        assert!(scales[0].label_mapping.is_none()); // No explicit mappings
        assert_eq!(scales[0].label_template, "{} units");
    }

    #[test]
    fn test_scale_renaming_wildcard_with_explicit() {
        // Mixed explicit mappings and wildcard template
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW point
            SCALE DISCRETE x RENAMING 'A' => 'Alpha', * => 'Category {}'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        // Check explicit mapping was parsed
        let label_mapping = scales[0].label_mapping.as_ref().unwrap();
        assert_eq!(label_mapping.get("A"), Some(&Some("Alpha".to_string())));

        // Check template was also parsed
        assert_eq!(scales[0].label_template, "Category {}");
    }

    #[test]
    fn test_scale_renaming_wildcard_uppercase() {
        // Wildcard template with uppercase transformation
        let query = r#"
            VISUALISE x AS x, y AS y
            DRAW bar
            SCALE DISCRETE x RENAMING * => '{:UPPER}'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        assert_eq!(scales[0].label_template, "{:UPPER}");
    }

    #[test]
    fn test_scale_renaming_wildcard_datetime() {
        // Wildcard template with datetime formatting
        let query = r#"
            VISUALISE date AS x, value AS y
            DRAW line
            SCALE CONTINUOUS x RENAMING * => '{:time %b %Y}'
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;

        assert_eq!(scales[0].label_template, "{:time %b %Y}");
    }

    // ========================================
    // ORDINAL scale type tests
    // ========================================

    #[test]
    fn test_scale_ordinal_basic() {
        // Basic ORDINAL scale type
        let query = r#"
            VISUALISE x AS x, y AS y, category AS fill
            DRAW point
            SCALE ORDINAL fill FROM ['low', 'medium', 'high'] TO viridis
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].aesthetic, "fill");
        assert!(scales[0].scale_type.is_some());
        assert_eq!(
            scales[0].scale_type.as_ref().unwrap().scale_type_kind(),
            crate::plot::ScaleTypeKind::Ordinal
        );

        // Check input range was parsed
        let input_range = scales[0].input_range.as_ref().unwrap();
        assert_eq!(input_range.len(), 3);

        // Check output range was parsed as palette
        assert!(scales[0].output_range.is_some());
    }

    #[test]
    fn test_scale_ordinal_with_explicit_colors() {
        // ORDINAL scale with explicit color array
        let query = r#"
            VISUALISE x AS x, y AS y, size_cat AS fill
            DRAW point
            SCALE ORDINAL fill FROM ['S', 'M', 'L'] TO ['#ff0000', '#00ff00', '#0000ff']
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(
            scales[0].scale_type.as_ref().unwrap().scale_type_kind(),
            crate::plot::ScaleTypeKind::Ordinal
        );
    }

    #[test]
    fn test_scale_ordinal_case_insensitive() {
        // ORDINAL should be case-insensitive
        let query = r#"
            VISUALISE x AS x, y AS y, cat AS color
            DRAW point
            SCALE ordinal color FROM ['a', 'b', 'c']
        "#;

        let specs = parse_test_query(query).unwrap();
        let scales = &specs[0].scales;
        assert_eq!(
            scales[0].scale_type.as_ref().unwrap().scale_type_kind(),
            crate::plot::ScaleTypeKind::Ordinal
        );
    }

    // ========================================================================
    // Basic Type Parser Tests
    // ========================================================================

    fn make_source(query: &str) -> SourceTree<'_> {
        SourceTree::new(query).unwrap()
    }

    #[test]
    fn test_parse_string_node() {
        let source = make_source("VISUALISE DRAW point LABEL title => 'hello world'");
        let root = source.root();

        let string_node = source.find_node(&root, "(string) @s").unwrap();
        let parsed = parse_string_node(&string_node, &source);
        assert_eq!(parsed, "hello world");
    }

    #[test]
    fn test_parse_number_node() {
        // Test integers
        let source = make_source("VISUALISE DRAW point SCALE x FROM [0, 100]");
        let root = source.root();

        let numbers = source.find_nodes(&root, "(number) @n");
        assert_eq!(numbers.len(), 2);
        assert_eq!(parse_number_node(&numbers[0], &source).unwrap(), 0.0);
        assert_eq!(parse_number_node(&numbers[1], &source).unwrap(), 100.0);

        // Test floats
        let source2 = make_source("VISUALISE DRAW point SCALE y FROM [-10.5, 20.75]");
        let root2 = source2.root();

        let numbers2 = source2.find_nodes(&root2, "(number) @n");
        assert_eq!(parse_number_node(&numbers2[0], &source2).unwrap(), -10.5);
        assert_eq!(parse_number_node(&numbers2[1], &source2).unwrap(), 20.75);
    }

    #[test]
    fn test_parse_array_node() {
        // Test array of strings
        let source = make_source("VISUALISE DRAW point SCALE x FROM ['a', 'b', 'c']");
        let root = source.root();

        let array_node = source.find_node(&root, "(array) @arr").unwrap();
        let parsed = parse_array_node(&array_node, &source).unwrap();

        assert_eq!(parsed.len(), 3);
        assert!(matches!(parsed[0], ArrayElement::String(ref s) if s == "a"));
        assert!(matches!(parsed[1], ArrayElement::String(ref s) if s == "b"));
        assert!(matches!(parsed[2], ArrayElement::String(ref s) if s == "c"));

        // Test array of numbers
        let source2 = make_source("VISUALISE DRAW point SCALE x FROM [0, 50, 100]");
        let root2 = source2.root();

        let array_node2 = source2.find_node(&root2, "(array) @arr").unwrap();
        let parsed2 = parse_array_node(&array_node2, &source2).unwrap();

        assert_eq!(parsed2.len(), 3);
        assert!(matches!(parsed2[0], ArrayElement::Number(n) if n == 0.0));
        assert!(matches!(parsed2[1], ArrayElement::Number(n) if n == 50.0));
        assert!(matches!(parsed2[2], ArrayElement::Number(n) if n == 100.0));
    }

    #[test]
    fn test_parse_data_source() {
        // Test identifier
        let source = make_source("VISUALISE FROM sales DRAW bar");
        let root = source.root();

        let from_node = source.find_node(&root, "(table_ref) @ref").unwrap();
        let parsed = parse_data_source(&from_node, &source);
        assert!(matches!(parsed, DataSource::Identifier(ref name) if name == "sales"));

        // Test file path - table_ref contains a string child
        let source2 = make_source("VISUALISE FROM 'data.csv' DRAW bar");
        let root2 = source2.root();

        let from_node2 = source2.find_node(&root2, "(table_ref) @ref").unwrap();
        let string_node = source2.find_node(&from_node2, "(string) @s").unwrap();
        let parsed2 = parse_data_source(&string_node, &source2);
        assert!(matches!(parsed2, DataSource::FilePath(ref path) if path == "data.csv"));
    }

    #[test]
    fn test_parse_literal_value() {
        // Test string literal
        let source = make_source("VISUALISE DRAW point MAPPING 'red' AS color");
        let root = source.root();

        let literal_node = source.find_node(&root, "(literal_value) @lit").unwrap();
        let parsed = parse_literal_value(&literal_node, &source).unwrap();
        assert!(
            matches!(parsed, AestheticValue::Literal(ParameterValue::String(ref s)) if s == "red")
        );

        // Test number literal
        let source2 = make_source("VISUALISE DRAW point MAPPING 42 AS size");
        let root2 = source2.root();

        let literal_node2 = source2.find_node(&root2, "(literal_value) @lit").unwrap();
        let parsed2 = parse_literal_value(&literal_node2, &source2).unwrap();
        assert!(matches!(parsed2, AestheticValue::Literal(ParameterValue::Number(n)) if n == 42.0));
    }
}
