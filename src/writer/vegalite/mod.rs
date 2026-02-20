//! Vega-Lite JSON writer implementation
//!
//! Converts ggsql specifications and DataFrames into Vega-Lite JSON format
//! for web-based interactive visualizations.
//!
//! # Mapping Strategy
//!
//! - ggsql Geom -> Vega-Lite mark type
//! - ggsql aesthetics -> Vega-Lite encoding channels
//! - ggsql layers -> Vega-Lite layer composition
//! - Polars DataFrame -> Vega-Lite inline data
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::writer::{Writer, VegaLiteWriter};
//!
//! let writer = VegaLiteWriter::new();
//! let vega_json = writer.write(&spec, &dataframe)?;
//! // Can be rendered in browser with vega-embed
//! ```

mod coord;
mod data;
mod encoding;
mod layer;

use crate::plot::ArrayElement;
use crate::plot::{ParameterValue, Scale, ScaleTypeKind};
use crate::writer::Writer;
use crate::{
    is_primary_positional, naming, primary_aesthetic, AestheticValue, DataFrame, GgsqlError, Plot,
    Result,
};
use serde_json::{json, Value};
use std::collections::HashMap;

// Re-export submodule functions for use in write()
use coord::apply_coord_transforms;
use data::{collect_binned_columns, is_binned_aesthetic, unify_datasets};
use encoding::{build_detail_encoding, build_encoding_channel, infer_field_type, map_aesthetic_name};
use layer::{geom_to_mark, get_renderer, validate_layer_columns, GeomRenderer, PreparedData};

/// Conversion factor from points to pixels (CSS standard: 96 DPI, 72 points/inch)
/// 1 point = 96/72 pixels = 1.333
const POINTS_TO_PIXELS: f64 = 96.0 / 72.0;

/// Conversion factor from radius (in points) to area (in square pixels)
/// Used for size aesthetic: area = pi * r^2 where r is in pixels
/// So: area_px^2 = pi * (r_pt * POINTS_TO_PIXELS)^2 = pi * r_pt^2 * (96/72)^2
const POINTS_TO_AREA: f64 = std::f64::consts::PI * POINTS_TO_PIXELS * POINTS_TO_PIXELS;

/// Result of preparing layer data for rendering
///
/// Contains the datasets, renderers, and prepared data needed to build Vega-Lite layers.
struct LayerPreparation {
    /// Individual datasets keyed by layer/component identifier
    datasets: serde_json::Map<String, Value>,
    /// Renderers for each layer (one per layer in spec.layers)
    renderers: Vec<Box<dyn GeomRenderer>>,
    /// Prepared data for each layer (one per layer in spec.layers)
    prepared: Vec<PreparedData>,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Prepare layer data using renderers
///
/// For each layer:
/// - Gets the appropriate renderer for the geom type
/// - Prepares data (handles both standard and composite cases like boxplot)
/// - Builds individual datasets map with type-specific keys
///
/// Returns the datasets map, renderers, and prepared data for layer building.
fn prepare_layer_data(
    spec: &Plot,
    data: &HashMap<String, DataFrame>,
    layer_data_keys: &[String],
    binned_columns: &HashMap<String, Vec<f64>>,
) -> Result<LayerPreparation> {
    let mut individual_datasets = serde_json::Map::new();
    let mut layer_renderers: Vec<Box<dyn GeomRenderer>> = Vec::new();
    let mut prepared_data: Vec<PreparedData> = Vec::new();

    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let data_key = &layer_data_keys[layer_idx];
        let df = data.get(data_key).ok_or_else(|| {
            GgsqlError::WriterError(format!(
                "Missing data source '{}' for layer {}",
                data_key,
                layer_idx + 1
            ))
        })?;

        // Get the appropriate renderer for this geom type
        let renderer = get_renderer(&layer.geom);

        // Prepare data using the renderer (handles both standard and composite cases)
        let prepared = renderer.prepare_data(df, data_key, binned_columns)?;

        // Add data to individual datasets based on prepared type
        match &prepared {
            PreparedData::Single { values } => {
                individual_datasets.insert(data_key.clone(), json!(values));
            }
            PreparedData::Composite { components, .. } => {
                // For composite geoms (boxplot, etc.), add each component dataset
                // with type-specific keys (e.g., "__ggsql_layer_0__lower_whisker")
                for (component_name, values) in components {
                    let type_key = format!("{}{}", data_key, component_name);
                    individual_datasets.insert(type_key, json!(values));
                }
            }
        }

        layer_renderers.push(renderer);
        prepared_data.push(prepared);
    }

    Ok(LayerPreparation {
        datasets: individual_datasets,
        renderers: layer_renderers,
        prepared: prepared_data,
    })
}

/// Build Vega-Lite layers from spec
///
/// For each layer:
/// - Creates layer spec with mark type
/// - Builds transform array with source filter
/// - Builds encoding channels for each aesthetic mapping
/// - Handles binned positional aesthetics (x2/y2 channels)
/// - Adds aesthetic parameters from SETTING as literal encodings
/// - Adds detail encoding for partition_by columns
/// - Applies geom-specific modifications via renderer
/// - Finalizes layers (may expand composite geoms into multiple layers)
///
/// The `free_x` and `free_y` flags indicate whether facet free scales are enabled.
/// When true, explicit domains should not be set for that axis.
fn build_layers(
    spec: &Plot,
    data: &HashMap<String, DataFrame>,
    layer_data_keys: &[String],
    layer_renderers: &[Box<dyn GeomRenderer>],
    prepared_data: &[PreparedData],
    free_x: bool,
    free_y: bool,
) -> Result<Vec<Value>> {
    let mut layers = Vec::new();

    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let data_key = &layer_data_keys[layer_idx];
        let df = data.get(data_key).unwrap();
        let renderer = &layer_renderers[layer_idx];
        let prepared = &prepared_data[layer_idx];

        // Layer spec with mark
        let mut layer_spec = json!({
            "mark": geom_to_mark(&layer.geom)
        });

        // Build transform array for this layer
        // Always starts with a filter to select this layer's data from unified dataset
        let mut transforms: Vec<Value> = Vec::new();

        // Add source filter transform (if the renderer needs it)
        // Composite geoms like boxplot add their own type-specific filters
        if renderer.needs_source_filter() {
            transforms.push(json!({
                "filter": {
                    "field": naming::SOURCE_COLUMN,
                    "equal": data_key
                }
            }));
        }

        // Set transform array on layer spec
        layer_spec["transform"] = json!(transforms);

        // Build encoding for this layer (pass free scale flags)
        let encoding = build_layer_encoding(layer, df, spec, free_x, free_y)?;
        layer_spec["encoding"] = Value::Object(encoding);

        // Apply geom-specific spec modifications via renderer
        renderer.modify_spec(&mut layer_spec, layer)?;

        // Finalize the layer (may expand into multiple layers for composite geoms)
        let final_layers = renderer.finalize(layer_spec, layer, data_key, prepared)?;
        layers.extend(final_layers);
    }

    Ok(layers)
}

/// Build encoding channels for a single layer
///
/// Handles:
/// - Tracking titled aesthetic families (one title per family)
/// - Building encoding channels for each aesthetic mapping
/// - Binned positional aesthetics (x2/y2 channels for bin width)
/// - Aesthetic parameters from SETTING as literal encodings
/// - Detail encoding for partition_by columns
/// - Geom-specific encoding modifications via renderer
///
/// The `free_x` and `free_y` flags indicate whether facet free scales are enabled.
/// When true, explicit domains should not be set for that axis.
fn build_layer_encoding(
    layer: &crate::plot::Layer,
    df: &DataFrame,
    spec: &Plot,
    free_x: bool,
    free_y: bool,
) -> Result<serde_json::Map<String, Value>> {
    let mut encoding = serde_json::Map::new();

    // Track which aesthetic families have been titled to ensure only one title per family
    let mut titled_families: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Collect primary aesthetics that exist in the layer (for title handling)
    // e.g., if layer has "y", then "ymin" and "ymax" should suppress their titles
    let primary_aesthetics: std::collections::HashSet<String> = layer
        .mappings
        .aesthetics
        .keys()
        .filter(|a| primary_aesthetic(a) == a.as_str())
        .cloned()
        .collect();

    // Create encoding context for this layer
    let mut enc_ctx = encoding::EncodingContext {
        df,
        spec,
        titled_families: &mut titled_families,
        primary_aesthetics: &primary_aesthetics,
        free_x,
        free_y,
    };

    // Build encoding channels for each aesthetic mapping
    for (aesthetic, value) in &layer.mappings.aesthetics {
        // Skip facet aesthetics - they are handled via top-level facet structure,
        // not as encoding channels. Adding them to encoding would create row-based
        // faceting instead of the intended wrap/grid layout.
        if matches!(aesthetic.as_str(), "panel" | "row" | "column") {
            continue;
        }

        let channel_name = map_aesthetic_name(aesthetic);
        let channel_encoding = build_encoding_channel(aesthetic, value, &mut enc_ctx)?;
        encoding.insert(channel_name, channel_encoding);

        // For binned positional aesthetics (x, y), add xend/yend channel with bin_end column
        // This enables proper bin width rendering in Vega-Lite (maps to x2/y2 channels)
        if is_primary_positional(aesthetic) && is_binned_aesthetic(aesthetic, spec) {
            if let AestheticValue::Column { name: col, .. } = value {
                let end_col = naming::bin_end_column(col);
                let end_aesthetic = format!("{}end", aesthetic); // "xend" or "yend"
                let end_channel = map_aesthetic_name(&end_aesthetic); // maps to "x2" or "y2"
                encoding.insert(end_channel, json!({"field": end_col}));
            }
        }
    }

    // Add aesthetic parameters from SETTING as literal encodings
    // (e.g., SETTING color => 'red' becomes {"color": {"value": "red"}})
    // Only parameters that are supported aesthetics for this geom type are included
    let supported_aesthetics = layer.geom.aesthetics().supported;
    for (param_name, param_value) in &layer.parameters {
        if supported_aesthetics.contains(&param_name.as_str()) {
            let channel_name = map_aesthetic_name(param_name);
            // Only add if not already set by MAPPING (MAPPING takes precedence)
            if !encoding.contains_key(&channel_name) {
                // Convert size and linewidth from points to Vega-Lite units
                let converted_value = match (param_name.as_str(), param_value) {
                    // Size: interpret as radius in points, convert to area in pixels^2
                    ("size", ParameterValue::Number(n)) => json!(n * n * POINTS_TO_AREA),
                    // Linewidth: interpret as width in points, convert to pixels
                    ("linewidth", ParameterValue::Number(n)) => json!(n * POINTS_TO_PIXELS),
                    // Other aesthetics: pass through unchanged
                    _ => param_value.to_json(),
                };
                encoding.insert(channel_name, json!({"value": converted_value}));
            }
        }
    }

    // Add detail encoding for partition_by columns (grouping)
    if let Some(detail) = build_detail_encoding(&layer.partition_by) {
        encoding.insert("detail".to_string(), detail);
    }

    // Apply geom-specific encoding modifications via renderer
    let renderer = get_renderer(&layer.geom);
    renderer.modify_encoding(&mut encoding, layer)?;

    Ok(encoding)
}

/// Apply faceting to Vega-Lite spec
///
/// Handles:
/// - FACET vars (wrap layout)
/// - FACET rows BY columns (grid layout)
/// - Moves layers into nested `spec` object
/// - Uses aesthetic column names (e.g., __ggsql_aes_panel__)
/// - Respects scale types (Binned facets use bin: "binned")
/// - Scale resolution (scales property)
/// - Label renaming (RENAMING clause)
/// - Additional properties (ncol, etc.)
fn apply_faceting(
    vl_spec: &mut Value,
    facet: &crate::plot::Facet,
    facet_df: &DataFrame,
    scales: &[Scale],
) {
    use crate::plot::FacetLayout;

    match &facet.layout {
        FacetLayout::Wrap { variables: _ } => {
            // Use the aesthetic column name for panel
            let aes_col = naming::aesthetic_column("panel");

            // Look up scale for "panel" aesthetic
            let scale = scales.iter().find(|s| s.aesthetic == "panel");

            // Build facet field definition with proper binned support
            let mut facet_def = build_facet_field_def(facet_df, &aes_col, scale);

            // Prefer scale label_mapping over facet label_mapping
            let label_mapping = scale
                .and_then(|s| s.label_mapping.as_ref())
                .or(facet.label_mapping.as_ref());

            // Apply label renaming via header.labelExpr
            apply_facet_label_renaming(&mut facet_def, label_mapping, scale);

            // Apply facet ordering from breaks/reverse
            apply_facet_ordering(&mut facet_def, scale);

            vl_spec["facet"] = facet_def;

            // Move layer into spec (data reference stays at top level)
            let mut spec_inner = json!({});
            if let Some(layer) = vl_spec.get("layer") {
                spec_inner["layer"] = layer.clone();
            }

            vl_spec["spec"] = spec_inner;
            vl_spec.as_object_mut().unwrap().remove("layer");

            // Apply scale resolution
            apply_facet_scale_resolution(vl_spec, &facet.properties);

            // Apply additional properties (columns for wrap)
            apply_facet_properties(vl_spec, &facet.properties, true);
        }
        FacetLayout::Grid { row: _, column: _ } => {
            let mut facet_spec = serde_json::Map::new();

            // Row facet: use aesthetic column "row"
            let row_aes_col = naming::aesthetic_column("row");
            if facet_df.column(&row_aes_col).is_ok() {
                let row_scale = scales.iter().find(|s| s.aesthetic == "row");
                let mut row_def = build_facet_field_def(facet_df, &row_aes_col, row_scale);

                let row_label_mapping = row_scale
                    .and_then(|s| s.label_mapping.as_ref())
                    .or(facet.label_mapping.as_ref());
                apply_facet_label_renaming(&mut row_def, row_label_mapping, row_scale);
                apply_facet_ordering(&mut row_def, row_scale);

                facet_spec.insert("row".to_string(), row_def);
            }

            // Column facet: use aesthetic column "column"
            let col_aes_col = naming::aesthetic_column("column");
            if facet_df.column(&col_aes_col).is_ok() {
                let col_scale = scales.iter().find(|s| s.aesthetic == "column");
                let mut col_def = build_facet_field_def(facet_df, &col_aes_col, col_scale);

                let col_label_mapping = col_scale
                    .and_then(|s| s.label_mapping.as_ref())
                    .or(facet.label_mapping.as_ref());
                apply_facet_label_renaming(&mut col_def, col_label_mapping, col_scale);
                apply_facet_ordering(&mut col_def, col_scale);

                facet_spec.insert("column".to_string(), col_def);
            }

            vl_spec["facet"] = Value::Object(facet_spec);

            // Move layer into spec (data reference stays at top level)
            let mut spec_inner = json!({});
            if let Some(layer) = vl_spec.get("layer") {
                spec_inner["layer"] = layer.clone();
            }

            vl_spec["spec"] = spec_inner;
            vl_spec.as_object_mut().unwrap().remove("layer");

            // Apply scale resolution
            apply_facet_scale_resolution(vl_spec, &facet.properties);

            // Apply additional properties (not columns for grid)
            apply_facet_properties(vl_spec, &facet.properties, false);
        }
    }
}

/// Build a facet field definition with proper type.
///
/// Facets always use "type": "nominal" since facet values are categorical
/// (even for binned data, the bin labels are discrete categories).
fn build_facet_field_def(df: &DataFrame, col: &str, scale: Option<&Scale>) -> Value {
    let mut field_def = json!({
        "field": col,
    });

    if let Some(scale) = scale {
        if let Some(ref scale_type) = scale.scale_type {
            match scale_type.scale_type_kind() {
                // All scale types use nominal for facets - the data column contains
                // categorical values (bin midpoints for binned, categories for discrete)
                ScaleTypeKind::Binned
                | ScaleTypeKind::Discrete
                | ScaleTypeKind::Ordinal
                | ScaleTypeKind::Continuous
                | ScaleTypeKind::Identity => {
                    field_def["type"] = json!("nominal");
                    return field_def;
                }
            }
        }
    }

    // Fall back to column type inference
    field_def["type"] = json!(infer_field_type(df, col));
    field_def
}

/// Apply facet ordering via Vega-Lite's sort property.
///
/// For discrete facets: uses input_range (FROM clause) or breaks array order
/// For binned facets: uses "descending" sort if reversed
fn apply_facet_ordering(facet_def: &mut Value, scale: Option<&Scale>) {
    let Some(scale) = scale else {
        return;
    };

    let is_reversed = matches!(
        scale.properties.get("reverse"),
        Some(ParameterValue::Boolean(true))
    );

    let is_binned = scale
        .scale_type
        .as_ref()
        .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
        .unwrap_or(false);

    if is_binned {
        // For binned facets: use "descending" sort if reversed
        if is_reversed {
            facet_def["sort"] = json!("descending");
        }
        // Default is ascending, no need to specify
    } else {
        // For discrete facets: use input_range (FROM clause) if present,
        // otherwise fall back to breaks property
        let order_values: Vec<ArrayElement> = if let Some(ref input_range) = scale.input_range {
            // Use explicit input_range from FROM clause
            input_range.clone()
        } else if let Some(ParameterValue::Array(arr)) = scale.properties.get("breaks") {
            // Fall back to breaks if present
            arr.clone()
        } else {
            return;
        };

        // Convert to JSON values, preserving null
        let mut sort_values: Vec<Value> = order_values.iter().map(|e| e.to_json()).collect();

        // Apply reverse if specified
        if is_reversed {
            sort_values.reverse();
        }

        facet_def["sort"] = json!(sort_values);
    }
}

/// Apply scale resolution to Vega-Lite spec based on facet scales property
///
/// Maps ggsql scales property to Vega-Lite resolve.scale configuration:
/// - "fixed": shared scales (Vega-Lite default, no resolve needed)
/// - "free": independent scales for both x and y
/// - "free_x": independent x scale, shared y scale
/// - "free_y": shared x scale, independent y scale
fn apply_facet_scale_resolution(vl_spec: &mut Value, properties: &HashMap<String, ParameterValue>) {
    let scales = properties
        .get("scales")
        .and_then(|v| match v {
            ParameterValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("fixed");

    match scales {
        "fixed" => {
            // Vega-Lite default is shared scales, no resolve needed
        }
        "free" => {
            vl_spec["resolve"] = json!({
                "scale": {"x": "independent", "y": "independent"}
            });
        }
        "free_x" => {
            vl_spec["resolve"] = json!({
                "scale": {"x": "independent"}
            });
        }
        "free_y" => {
            vl_spec["resolve"] = json!({
                "scale": {"y": "independent"}
            });
        }
        _ => {
            // Unknown value - resolution should have validated this
        }
    }
}

/// Apply label renaming to a facet definition via header.labelExpr
///
/// Uses Vega expression to transform facet labels:
/// - For discrete facets: 'A' => 'Alpha' becomes datum.value == 'A' ? 'Alpha' : ...
/// - For binned facets: uses build_symbol_legend_label_mapping for range-style labels
/// - NULL values suppress labels (maps to empty string)
///
/// Note: Wildcard templates are resolved during facet property resolution,
/// so by this point label_mapping contains all expanded mappings.
fn apply_facet_label_renaming(
    facet_def: &mut Value,
    label_mapping: Option<&HashMap<String, Option<String>>>,
    scale: Option<&Scale>,
) {
    // Only apply if there's a label mapping
    let has_mapping = label_mapping.is_some_and(|m| !m.is_empty());

    if !has_mapping {
        return;
    }

    // Check if this is a binned facet
    let is_binned = scale
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
        .unwrap_or(false);

    let label_expr = if is_binned {
        // For binned facets: reuse build_symbol_legend_label_mapping and build_label_expr
        build_binned_facet_label_expr(label_mapping, scale)
    } else {
        // For discrete facets: compare datum.value against string values
        build_discrete_facet_label_expr(label_mapping)
    };

    // Add to facet definition
    facet_def["header"] = json!({
        "labelExpr": label_expr
    });
}

/// Build labelExpr for binned facet values.
///
/// For binned facets, `datum.value` contains the bin midpoint (e.g., 25 for bin [20-30)).
/// This function maps midpoint values to range-style labels like "Lower – Upper",
/// using custom labels from label_mapping when available.
///
/// Unlike `build_symbol_legend_label_mapping` which maps Vega-Lite's auto-generated
/// range labels, this function maps numeric midpoints to our range labels.
fn build_binned_facet_label_expr(
    label_mapping: Option<&HashMap<String, Option<String>>>,
    scale: Option<&Scale>,
) -> String {
    let Some(scale) = scale else {
        return "datum.value".to_string();
    };

    let breaks = match scale.properties.get("breaks") {
        Some(ParameterValue::Array(arr)) => arr,
        _ => return "datum.value".to_string(),
    };

    if breaks.len() < 2 {
        return "datum.value".to_string();
    }

    // Get closed property for determining open-format labels
    let closed = scale
        .properties
        .get("closed")
        .and_then(|v| match v {
            ParameterValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("left");

    let num_bins = breaks.len() - 1;

    // Build mapping from midpoint to range label
    let mut midpoint_to_range: Vec<(String, Option<String>)> = Vec::new();

    for i in 0..num_bins {
        let lower = &breaks[i];
        let upper = &breaks[i + 1];

        // Calculate midpoint for comparison
        let midpoint_str = calculate_midpoint_string(lower, upper, scale.transform.as_ref());
        let Some(midpoint_str) = midpoint_str else {
            continue;
        };

        // Get break values as strings (for default labels)
        let lower_str = lower.to_key_string();
        let upper_str = upper.to_key_string();

        // Build the range label
        let range_label = if let Some(label_mapping) = label_mapping {
            // Check if terminals are suppressed
            let lower_suppressed = label_mapping.get(&lower_str) == Some(&None);
            let upper_suppressed = label_mapping.get(&upper_str) == Some(&None);

            // Get custom labels (fall back to break values)
            let lower_label = label_mapping
                .get(&lower_str)
                .cloned()
                .flatten()
                .unwrap_or_else(|| lower_str.clone());
            let upper_label = label_mapping
                .get(&upper_str)
                .cloned()
                .flatten()
                .unwrap_or_else(|| upper_str.clone());

            // Determine label format based on terminal suppression
            if i == 0 && lower_suppressed {
                // First bin with suppressed lower terminal → open format
                let symbol = if closed == "right" { "≤" } else { "<" };
                Some(format!("{} {}", symbol, upper_label))
            } else if i == num_bins - 1 && upper_suppressed {
                // Last bin with suppressed upper terminal → open format
                let symbol = if closed == "right" { ">" } else { "≥" };
                Some(format!("{} {}", symbol, lower_label))
            } else {
                // Standard range format: "lower – upper"
                Some(format!("{} – {}", lower_label, upper_label))
            }
        } else {
            // No label mapping - use default range format with break values
            Some(format!("{} – {}", lower_str, upper_str))
        };

        midpoint_to_range.push((midpoint_str, range_label));
    }

    if midpoint_to_range.is_empty() {
        return "datum.value".to_string();
    }

    // Build labelExpr comparing datum.value against midpoints
    build_binned_facet_value_expr(&midpoint_to_range)
}

/// Build labelExpr comparing datum.value against midpoint values
fn build_binned_facet_value_expr(mappings: &[(String, Option<String>)]) -> String {
    let mut expr_parts: Vec<String> = Vec::new();

    for (midpoint, label) in mappings {
        // Compare as number for numeric midpoints, string for temporal
        let condition = format!("datum.value == {}", midpoint);
        let result = match label {
            Some(l) => format!("'{}'", escape_vega_string(l)),
            None => "''".to_string(),
        };
        expr_parts.push(format!("{} ? {}", condition, result));
    }

    if expr_parts.is_empty() {
        return "datum.value".to_string();
    }

    // Chain: cond1 ? val1 : cond2 ? val2 : datum.value
    let mut expr = "datum.value".to_string();
    for part in expr_parts.into_iter().rev() {
        expr = format!("{} : {}", part, expr);
    }
    expr
}

/// Calculate the midpoint string for a bin
fn calculate_midpoint_string(
    lower: &ArrayElement,
    upper: &ArrayElement,
    transform: Option<&crate::plot::scale::Transform>,
) -> Option<String> {
    match (lower, upper) {
        (ArrayElement::Number(l), ArrayElement::Number(u)) => {
            let midpoint = (*l + *u) / 2.0;

            // Check if temporal transform - format as ISO string (quoted for comparison)
            if let Some(t) = transform {
                if let Some(iso) = t.format_as_iso(midpoint) {
                    return Some(format!("'{}'", iso));
                }
            }

            // Numeric: format without trailing decimals if whole number
            Some(if midpoint.fract() == 0.0 {
                format!("{}", midpoint as i64)
            } else {
                format!("{}", midpoint)
            })
        }
        // Temporal ArrayElements - calculate midpoint and format as ISO (quoted)
        (ArrayElement::Date(l), ArrayElement::Date(u)) => {
            let midpoint = ((*l as f64) + (*u as f64)) / 2.0;
            Some(format!("'{}'", ArrayElement::date_to_iso(midpoint as i32)))
        }
        (ArrayElement::DateTime(l), ArrayElement::DateTime(u)) => {
            let midpoint = ((*l as f64) + (*u as f64)) / 2.0;
            Some(format!(
                "'{}'",
                ArrayElement::datetime_to_iso(midpoint as i64)
            ))
        }
        (ArrayElement::Time(l), ArrayElement::Time(u)) => {
            let midpoint = ((*l as f64) + (*u as f64)) / 2.0;
            Some(format!("'{}'", ArrayElement::time_to_iso(midpoint as i64)))
        }
        _ => None,
    }
}

/// Build labelExpr for discrete facet values.
fn build_discrete_facet_label_expr(
    label_mapping: Option<&HashMap<String, Option<String>>>,
) -> String {
    let Some(mappings) = label_mapping else {
        return "datum.value".to_string();
    };

    // Build labelExpr for Vega-Lite
    let mut expr_parts: Vec<String> = Vec::new();

    // Add explicit mappings
    for (from, to) in mappings {
        // Handle null values: 'null' key maps to JSON null comparison
        let condition = if from == "null" {
            "datum.value == null".to_string()
        } else {
            format!("datum.value == '{}'", escape_vega_string(from))
        };
        let result = match to {
            Some(label) => format!("'{}'", escape_vega_string(label)),
            None => "''".to_string(), // NULL suppresses label
        };
        expr_parts.push(format!("{} ? {}", condition, result));
    }

    // Default case: show original value
    let default_expr = "datum.value".to_string();

    // Build the full expression as nested ternary
    if expr_parts.is_empty() {
        default_expr
    } else {
        // Chain conditions: cond1 ? val1 : cond2 ? val2 : default
        let mut expr = default_expr;
        for part in expr_parts.into_iter().rev() {
            expr = format!("{} : {}", part, expr);
        }
        expr
    }
}

/// Escape a string for use in Vega expressions
fn escape_vega_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Apply additional facet properties to Vega-Lite spec
///
/// Handles:
/// - ncol: Number of columns for wrap facets (maps to Vega-Lite's "columns")
///
/// Note: scales is handled separately by apply_facet_scale_resolution
fn apply_facet_properties(
    vl_spec: &mut Value,
    properties: &HashMap<String, ParameterValue>,
    is_wrap: bool,
) {
    for (name, value) in properties {
        match name.as_str() {
            "ncol" if is_wrap => {
                // ncol maps to Vega-Lite's "columns" property
                if let ParameterValue::Number(n) = value {
                    vl_spec["columns"] = json!(*n as i64);
                }
            }
            "scales" => {
                // Handled by apply_facet_scale_resolution
            }
            _ => {
                // Unknown properties ignored (resolution should have validated)
            }
        }
    }
}

/// Vega-Lite JSON writer
///
/// Generates Vega-Lite v6 specifications from ggsql specs and data.
pub struct VegaLiteWriter {
    /// Vega-Lite schema version
    schema: String,
}

impl VegaLiteWriter {
    /// Create a new Vega-Lite writer with default settings
    pub fn new() -> Self {
        Self {
            schema: "https://vega.github.io/schema/vega-lite/v6.json".to_string(),
        }
    }

    /// Build default Vega-Lite config matching ggplot2's theme_gray()
    ///
    /// Font sizes converted from ggplot2 points to pixels (1 pt ≈ 1.33 px at 96 DPI):
    /// - axis.text: 8.8 pts (rel(0.8) × 11) → 12 px
    /// - axis.title: 11 pts → 15 px
    /// - legend.text: 8.8 pts → 12 px
    /// - legend.title: 11 pts → 15 px
    /// - plot.title: 13.2 pts (rel(1.2) × 11) → 18 px
    /// - tick size: ~2.75 pts → 4 px
    fn default_theme_config(&self) -> Value {
        json!({
            "view": {
                "stroke": null,
                "fill": "#EBEBEB"
            },
            "axis": {
                "domain": false,
                "grid": true,
                "gridColor": "#FFFFFF",
                "gridWidth": 1,
                "tickColor": "#333333",
                "tickSize": 4,
                "labelColor": "#4D4D4D",
                "labelFontSize": 12,
                "titleColor": "#000000",
                "titleFontSize": 15,
                "titleFontWeight": "normal",
                "titlePadding": 10
            },
            "legend": {
                "labelColor": "#4D4D4D",
                "labelFontSize": 12,
                "titleColor": "#000000",
                "titleFontSize": 15,
                "titleFontWeight": "normal",
                "titlePadding": 8,
                "rowPadding": 6
            },
            "title": {
                "color": "#000000",
                "fontSize": 18,
                "fontWeight": "normal",
                "subtitleColor": "#4D4D4D",
                "subtitleFontSize": 15,
                "subtitleFontWeight": "normal",
                "anchor": "start",
                "frame": "group",
                "offset": 10
            },
            "header": {
                "labelColor": "#000000",
                "labelFontSize": 15,
                "labelFontWeight": "normal",
                "labelPadding": 5,
                "title": null
            }
        })
    }
}

impl Default for VegaLiteWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer for VegaLiteWriter {
    type Output = String;

    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<String> {
        // 1. Validate spec
        self.validate(spec)?;

        // 2. Determine if facet free scales should omit x/y domains
        // When using free scales, Vega-Lite computes independent domains per facet panel.
        // We must not set explicit domains (from SCALE or COORD) as they would override this.
        let (free_x, free_y) = if let Some(ref facet) = spec.facet {
            let scales_value = facet
                .properties
                .get("scales")
                .and_then(|v| match v {
                    ParameterValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("fixed");

            match scales_value {
                "free" => (true, true),
                "free_x" => (true, false),
                "free_y" => (false, true),
                _ => (false, false), // "fixed" or unknown
            }
        } else {
            (false, false)
        };

        // 3. Determine layer data keys
        let layer_data_keys: Vec<String> = spec
            .layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                layer
                    .data_key
                    .clone()
                    .unwrap_or_else(|| naming::layer_key(idx))
            })
            .collect();

        // 4. Validate columns for each layer
        for (layer_idx, (layer, key)) in spec.layers.iter().zip(layer_data_keys.iter()).enumerate()
        {
            let df = data.get(key).ok_or_else(|| {
                GgsqlError::WriterError(format!(
                    "Missing data source '{}' for layer {}",
                    key,
                    layer_idx + 1
                ))
            })?;
            validate_layer_columns(layer, df, layer_idx)?;
        }

        // 5. Build base Vega-Lite spec
        let mut vl_spec = json!({
            "$schema": self.schema
        });
        vl_spec["width"] = json!("container");
        vl_spec["height"] = json!("container");

        if let Some(labels) = &spec.labels {
            if let Some(title) = labels.labels.get("title") {
                vl_spec["title"] = json!(title);
            }
        }

        // 6. Collect binned columns
        let binned_columns = collect_binned_columns(spec);

        // 7. Prepare layer data
        let prep = prepare_layer_data(spec, data, &layer_data_keys, &binned_columns)?;

        // 8. Unify datasets
        let unified_data = unify_datasets(&prep.datasets)?;
        vl_spec["data"] = json!({"values": unified_data});

        // 9. Build layers (pass free scale flags for domain handling)
        let layers = build_layers(
            spec,
            data,
            &layer_data_keys,
            &prep.renderers,
            &prep.prepared,
            free_x,
            free_y,
        )?;
        vl_spec["layer"] = json!(layers);

        // 10. Apply coordinate transforms (pass free scale flags for domain handling)
        let first_df = data.get(&layer_data_keys[0]).unwrap();
        apply_coord_transforms(spec, first_df, &mut vl_spec, free_x, free_y)?;

        // 11. Apply faceting
        if let Some(facet) = &spec.facet {
            let facet_df = data.get(&layer_data_keys[0]).unwrap();
            apply_faceting(&mut vl_spec, facet, facet_df, &spec.scales);
        }

        // 12. Add default theme config (ggplot2-like gray theme)
        vl_spec["config"] = self.default_theme_config();

        // 13. Serialize
        serde_json::to_string_pretty(&vl_spec).map_err(|e| {
            GgsqlError::WriterError(format!("Failed to serialize Vega-Lite JSON: {}", e))
        })
    }

    fn validate(&self, spec: &Plot) -> Result<()> {
        // Check that we have at least one layer
        if spec.layers.is_empty() {
            return Err(GgsqlError::ValidationError(
                "VegaLiteWriter requires at least one layer".to_string(),
            ));
        }

        // Validate each layer
        for layer in &spec.layers {
            // Check required aesthetics
            layer.validate_required_aesthetics().map_err(|e| {
                GgsqlError::ValidationError(format!("Layer validation failed: {}", e))
            })?;

            // Check SETTING parameters are valid for this geom
            layer.validate_settings().map_err(|e| {
                GgsqlError::ValidationError(format!("Layer validation failed: {}", e))
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{Labels, Layer, ParameterValue};
    use crate::Geom;
    use polars::prelude::*;
    use std::collections::HashMap;

    // Re-export test functions from submodules
    use super::data::find_bin_for_value;
    use super::encoding::infer_field_type;
    use super::layer::geom_to_mark;

    /// Helper to wrap a DataFrame in a data map for testing (uses layer 0 key)
    fn wrap_data(df: DataFrame) -> HashMap<String, DataFrame> {
        wrap_data_for_layers(df, 1)
    }

    /// Helper to wrap a DataFrame for multiple layers (clones for each layer)
    fn wrap_data_for_layers(df: DataFrame, num_layers: usize) -> HashMap<String, DataFrame> {
        let mut data_map = HashMap::new();
        for i in 0..num_layers {
            data_map.insert(naming::layer_key(i), df.clone());
        }
        data_map
    }

    #[test]
    fn test_geom_to_mark_mapping() {
        // All marks should be objects with type and clip: true
        assert_eq!(
            geom_to_mark(&Geom::point()),
            json!({"type": "point", "clip": true})
        );
        assert_eq!(
            geom_to_mark(&Geom::line()),
            json!({"type": "line", "clip": true})
        );
        assert_eq!(
            geom_to_mark(&Geom::bar()),
            json!({"type": "bar", "clip": true})
        );
        assert_eq!(
            geom_to_mark(&Geom::area()),
            json!({"type": "area", "clip": true})
        );
        assert_eq!(
            geom_to_mark(&Geom::tile()),
            json!({"type": "rect", "clip": true})
        );
    }

    #[test]
    fn test_aesthetic_name_mapping() {
        // Pass-through aesthetics (including fill and stroke for separate color control)
        assert_eq!(map_aesthetic_name("x"), "x");
        assert_eq!(map_aesthetic_name("y"), "y");
        assert_eq!(map_aesthetic_name("color"), "color");
        assert_eq!(map_aesthetic_name("fill"), "fill");
        assert_eq!(map_aesthetic_name("stroke"), "stroke");
        assert_eq!(map_aesthetic_name("opacity"), "opacity");
        assert_eq!(map_aesthetic_name("size"), "size");
        assert_eq!(map_aesthetic_name("shape"), "shape");
        // Position end aesthetics (ggsql -> Vega-Lite)
        assert_eq!(map_aesthetic_name("xend"), "x2");
        assert_eq!(map_aesthetic_name("yend"), "y2");
        // Other mapped aesthetics
        assert_eq!(map_aesthetic_name("linetype"), "strokeDash");
        assert_eq!(map_aesthetic_name("linewidth"), "strokeWidth");
        assert_eq!(map_aesthetic_name("label"), "text");
    }

    #[test]
    fn test_validation_requires_layers() {
        let writer = VegaLiteWriter::new();
        let spec = Plot::new();
        assert!(writer.validate(&spec).is_err());
    }

    #[test]
    fn test_simple_point_spec() {
        let writer = VegaLiteWriter::new();

        // Create a simple spec
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Create simple DataFrame
        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify structure (uses layer array with inline data)
        assert_eq!(vl_spec["$schema"], writer.schema);
        assert!(vl_spec["layer"].is_array());
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "point");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
        assert!(vl_spec["data"]["values"].is_array());
        assert_eq!(vl_spec["data"]["values"].as_array().unwrap().len(), 3);
        assert!(vl_spec["layer"][0]["encoding"]["x"].is_object());
        assert!(vl_spec["layer"][0]["encoding"]["y"].is_object());
    }

    #[test]
    fn test_with_title() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("title".to_string(), "My Chart".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01-01", "2024-01-02"],
            "value" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "My Chart");
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "line");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
    }

    #[test]
    fn test_literal_color() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(ParameterValue::String("red".to_string())),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["color"]["value"], "red");
    }

    #[test]
    fn test_missing_column_error() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("nonexistent".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let result = writer.write(&spec, &wrap_data(df));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn test_numeric_type_inference_integers() {
        let df = df! {
            "x" => &[1i64, 2, 3],
        }
        .unwrap();

        assert_eq!(infer_field_type(&df, "x"), "quantitative");
    }

    #[test]
    fn test_nominal_type_inference_strings() {
        let df = df! {
            "category" => &["A", "B", "C"],
        }
        .unwrap();

        assert_eq!(infer_field_type(&df, "category"), "nominal");
    }

    #[test]
    fn test_numeric_string_type_inference() {
        let df = df! {
            "numbers_as_strings" => &["1.5", "2.5", "3.5"],
        }
        .unwrap();

        assert_eq!(infer_field_type(&df, "numbers_as_strings"), "quantitative");
    }

    #[test]
    fn test_find_bin_for_value() {
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        // Test value in first bin [0, 10)
        assert_eq!(find_bin_for_value(5.0, &breaks), Some((0.0, 10.0)));

        // Test value at bin boundary (belongs to next bin due to half-open interval)
        assert_eq!(find_bin_for_value(10.0, &breaks), Some((10.0, 20.0)));

        // Test value in middle bin [10, 20)
        assert_eq!(find_bin_for_value(15.0, &breaks), Some((10.0, 20.0)));

        // Test value in last bin [20, 30] (closed interval)
        assert_eq!(find_bin_for_value(25.0, &breaks), Some((20.0, 30.0)));

        // Test value at last edge (should be included in last bin)
        assert_eq!(find_bin_for_value(30.0, &breaks), Some((20.0, 30.0)));

        // Test value outside range
        assert_eq!(find_bin_for_value(-5.0, &breaks), None);
        assert_eq!(find_bin_for_value(35.0, &breaks), None);

        // Test with too few breaks
        assert_eq!(find_bin_for_value(5.0, &[10.0]), None);
    }

    #[test]
    fn test_multi_layer_composition() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // Add line layer
        let line_layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(line_layer);

        // Add point layer
        let point_layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(point_layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data_for_layers(df, 2)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have layer array with 2 layers
        assert!(vl_spec["layer"].is_array());
        assert_eq!(vl_spec["layer"].as_array().unwrap().len(), 2);
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "line");
        assert_eq!(vl_spec["layer"][1]["mark"]["type"], "point");
    }

    #[test]
    fn test_build_symbol_legend_label_mapping_basic() {
        // Test the build_symbol_legend_label_mapping function directly
        use super::encoding::build_symbol_legend_label_mapping;

        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
            ArrayElement::Number(100.0),
        ];

        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), Some("Low".to_string()));
        label_mapping.insert("25".to_string(), Some("Medium".to_string()));
        label_mapping.insert("50".to_string(), Some("High".to_string()));
        label_mapping.insert("75".to_string(), Some("Very High".to_string()));
        label_mapping.insert("100".to_string(), Some("Max".to_string())); // Will be excluded

        let result = build_symbol_legend_label_mapping(&breaks, &label_mapping, "left");

        // VL generates: "0 – 25", "25 – 50", "50 – 75", "≥ 75"
        // We map to range format using custom labels: "lower_label – upper_label"
        assert_eq!(
            result.get("0 – 25"),
            Some(&Some("Low – Medium".to_string()))
        );
        assert_eq!(
            result.get("25 – 50"),
            Some(&Some("Medium – High".to_string()))
        );
        assert_eq!(
            result.get("50 – 75"),
            Some(&Some("High – Very High".to_string()))
        );
        assert_eq!(
            result.get("≥ 75"),
            Some(&Some("Very High – Max".to_string()))
        );

        // Should not include a mapping for the last terminal value directly
        assert!(!result.contains_key("100"));
    }

    #[test]
    fn test_symbol_legend_label_expr_uses_range_format() {
        // Test that symbol legend labelExpr maps VL's range labels to our labels
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add binned color scale (symbol legend case)
        let mut scale = Scale::new("color");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        // Add label renaming
        let mut labels = HashMap::new();
        labels.insert("0".to_string(), Some("Low".to_string()));
        labels.insert("25".to_string(), Some("Medium".to_string()));
        labels.insert("50".to_string(), Some("High".to_string()));
        labels.insert("75".to_string(), Some("Very High".to_string()));
        scale.label_mapping = Some(labels);
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that labelExpr contains VL's range-style format
        let label_expr = &vl_spec["layer"][0]["encoding"]["color"]["legend"]["labelExpr"];
        assert!(label_expr.is_string());
        let expr = label_expr.as_str().unwrap();

        // Should contain mappings for VL's range format labels to our range format
        assert!(
            expr.contains("0 – 25"),
            "labelExpr should contain VL's range format '0 – 25', got: {}",
            expr
        );
        assert!(
            expr.contains("'Low – Medium'"),
            "labelExpr should map to 'Low – Medium', got: {}",
            expr
        );
        assert!(
            expr.contains("≥ 75"),
            "labelExpr should contain VL's last bin format '≥ 75', got: {}",
            expr
        );
        // Note: last bin maps "≥ 75" to "Very High – 100" (no custom label for 100 in this test)
        assert!(
            expr.contains("'Very High"),
            "labelExpr should contain 'Very High', got: {}",
            expr
        );
    }

    #[test]
    fn test_symbol_legend_open_format_with_oob_squish() {
        // Test that oob='squish' produces open format labels for symbol legends
        use super::encoding::build_symbol_legend_label_mapping;

        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
            ArrayElement::Number(100.0),
        ];

        // Suppress first and last terminals (oob='squish' behavior)
        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), None); // Suppressed
        label_mapping.insert("25".to_string(), Some("Medium".to_string()));
        label_mapping.insert("50".to_string(), Some("High".to_string()));
        label_mapping.insert("75".to_string(), Some("Very High".to_string()));
        label_mapping.insert("100".to_string(), None); // Suppressed

        // Test with closed='left' (default)
        let result_left = build_symbol_legend_label_mapping(&breaks, &label_mapping, "left");

        // First bin: suppressed lower terminal → "< 25" (open format)
        assert_eq!(
            result_left.get("0 – 25"),
            Some(&Some("< Medium".to_string())),
            "First bin with suppressed lower should use '< upper' format"
        );
        // Last bin: suppressed upper terminal → "≥ 75" (open format, same as normal)
        assert_eq!(
            result_left.get("≥ 75"),
            Some(&Some("≥ Very High".to_string())),
            "Last bin with suppressed upper should use '≥ lower' format"
        );

        // Test with closed='right'
        let result_right = build_symbol_legend_label_mapping(&breaks, &label_mapping, "right");

        // First bin: suppressed lower terminal → "≤ 25" (right-closed means upper included)
        assert_eq!(
            result_right.get("0 – 25"),
            Some(&Some("≤ Medium".to_string())),
            "First bin with closed='right' should use '≤ upper' format"
        );
        // Last bin: suppressed upper terminal → "> 75" (right-closed means lower not included)
        assert_eq!(
            result_right.get("≥ 75"),
            Some(&Some("> Very High".to_string())),
            "Last bin with closed='right' should use '> lower' format"
        );
    }

    #[test]
    fn test_facet_ordering_uses_input_range() {
        // Test that apply_facet_ordering uses input_range (FROM clause) for discrete scales
        use crate::plot::scale::Scale;

        let mut facet_def = json!({"field": "__ggsql_aes_panel__", "type": "nominal"});

        // Create a scale with input_range (simulating SCALE panel FROM ['A', 'B', 'C'])
        let mut scale = Scale::new("panel");
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
        ]);

        apply_facet_ordering(&mut facet_def, Some(&scale));

        // Verify sort order matches input_range
        assert_eq!(
            facet_def["sort"],
            json!(["A", "B", "C"]),
            "Facet sort should use input_range order"
        );
    }

    #[test]
    fn test_facet_ordering_with_null_in_input_range() {
        // Test that apply_facet_ordering preserves null values in input_range
        // This is the fix for the bug where null panels appear first
        use crate::plot::scale::Scale;

        let mut facet_def = json!({"field": "__ggsql_aes_panel__", "type": "nominal"});

        // Create a scale with input_range including null at the end
        // (simulating SCALE panel FROM ['Adelie', 'Gentoo', null])
        let mut scale = Scale::new("panel");
        scale.input_range = Some(vec![
            ArrayElement::String("Adelie".to_string()),
            ArrayElement::String("Gentoo".to_string()),
            ArrayElement::Null,
        ]);

        apply_facet_ordering(&mut facet_def, Some(&scale));

        // Verify sort order preserves null at the end
        assert_eq!(
            facet_def["sort"],
            json!(["Adelie", "Gentoo", null]),
            "Facet sort should preserve null position from input_range"
        );
    }

    #[test]
    fn test_facet_ordering_with_null_first_in_input_range() {
        // Test that null at the beginning of input_range produces null first in sort
        use crate::plot::scale::Scale;

        let mut facet_def = json!({"field": "__ggsql_aes_panel__", "type": "nominal"});

        // Create a scale with null at the beginning
        let mut scale = Scale::new("panel");
        scale.input_range = Some(vec![
            ArrayElement::Null,
            ArrayElement::String("Adelie".to_string()),
            ArrayElement::String("Gentoo".to_string()),
        ]);

        apply_facet_ordering(&mut facet_def, Some(&scale));

        // Verify null is first in sort order
        assert_eq!(
            facet_def["sort"],
            json!([null, "Adelie", "Gentoo"]),
            "Facet sort should preserve null at beginning"
        );
    }

    #[test]
    fn test_discrete_facet_label_expr_renames_null() {
        // Test that 'null' key in label_mapping generates correct Vega expression
        // for comparing against JSON null (not string 'null')
        let mut mappings = HashMap::new();
        mappings.insert("Adelie".to_string(), Some("Adelie Penguin".to_string()));
        mappings.insert("null".to_string(), Some("Missing".to_string()));

        let expr = build_discrete_facet_label_expr(Some(&mappings));

        // Should contain null comparison without quotes
        assert!(
            expr.contains("datum.value == null"),
            "Label expr should use 'datum.value == null' (not string), got: {}",
            expr
        );
        // Should map null to 'Missing'
        assert!(
            expr.contains("'Missing'"),
            "Label expr should contain 'Missing', got: {}",
            expr
        );
        // Should still use string comparison for non-null values
        assert!(
            expr.contains("datum.value == 'Adelie'"),
            "Label expr should use string comparison for Adelie, got: {}",
            expr
        );
    }

    #[test]
    fn test_binned_facet_label_expr_uses_range_labels() {
        // Test that binned facet labelExpr uses range-style labels "Lower – Upper"
        use crate::plot::scale::Scale;
        use crate::plot::{ParameterValue, ScaleType};

        // Create a binned scale with breaks [0, 20, 40, 60]
        let mut scale = Scale::new("panel");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(40.0),
                ArrayElement::Number(60.0),
            ]),
        );

        // Create label mapping keyed by lower bound
        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), Some("Low".to_string()));
        label_mapping.insert("20".to_string(), Some("Medium".to_string()));
        label_mapping.insert("40".to_string(), Some("High".to_string()));
        label_mapping.insert("60".to_string(), Some("Very High".to_string()));

        let expr = build_binned_facet_label_expr(Some(&label_mapping), Some(&scale));

        // Should contain midpoint comparisons:
        // Bin [0, 20) -> midpoint 10
        // Bin [20, 40) -> midpoint 30
        // Bin [40, 60] -> midpoint 50
        assert!(
            expr.contains("datum.value == 10"),
            "labelExpr should compare against midpoint 10, got: {}",
            expr
        );
        assert!(
            expr.contains("datum.value == 30"),
            "labelExpr should compare against midpoint 30, got: {}",
            expr
        );
        assert!(
            expr.contains("datum.value == 50"),
            "labelExpr should compare against midpoint 50, got: {}",
            expr
        );

        // Should map to range-style labels using custom label names
        assert!(
            expr.contains("'Low – Medium'"),
            "labelExpr should contain range label 'Low – Medium', got: {}",
            expr
        );
        assert!(
            expr.contains("'Medium – High'"),
            "labelExpr should contain range label 'Medium – High', got: {}",
            expr
        );
        assert!(
            expr.contains("'High – Very High'"),
            "labelExpr should contain range label 'High – Very High', got: {}",
            expr
        );
    }

    #[test]
    fn test_binned_facet_label_expr_with_suppressed_lower_terminal() {
        // Test that suppressed lower terminal creates open-format label "< Upper"
        use crate::plot::scale::Scale;
        use crate::plot::{ParameterValue, ScaleType};

        let mut scale = Scale::new("panel");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // Create label mapping with suppressed first terminal (oob='squish' behavior)
        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), None); // Suppress lower terminal
        label_mapping.insert("50".to_string(), Some("High".to_string()));
        label_mapping.insert("100".to_string(), Some("Max".to_string()));

        let expr = build_binned_facet_label_expr(Some(&label_mapping), Some(&scale));

        // First bin with suppressed lower terminal → open format "< 50" or "< High"
        // (uses upper bound label since lower is suppressed)
        assert!(
            expr.contains("'< High'"),
            "First bin with suppressed lower should use '< Upper' format, got: {}",
            expr
        );
        // Second bin should use range format
        assert!(
            expr.contains("'High – Max'"),
            "Second bin should use range format 'High – Max', got: {}",
            expr
        );
    }

    #[test]
    fn test_binned_facet_label_expr_default_range_format() {
        // Test that binned facet without label_mapping uses default range format
        use crate::plot::scale::Scale;
        use crate::plot::{ParameterValue, ScaleType};

        let mut scale = Scale::new("panel");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
            ]),
        );

        // No label_mapping - should use break values in range format
        let expr = build_binned_facet_label_expr(None, Some(&scale));

        // Should use default range format with break values
        assert!(
            expr.contains("'0 – 25'"),
            "Should use default range format '0 – 25', got: {}",
            expr
        );
        assert!(
            expr.contains("'25 – 50'"),
            "Should use default range format '25 – 50', got: {}",
            expr
        );
    }

    #[test]
    fn test_facet_free_scales_omits_domain() {
        // Test that FACET with scales => 'free' does not set explicit domains
        // This allows Vega-Lite to compute independent domains per facet panel
        use crate::plot::scale::Scale;
        use crate::plot::{Facet, FacetLayout, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add facet with free scales
        let mut facet_properties = HashMap::new();
        facet_properties.insert("scales".to_string(), ParameterValue::String("free".to_string()));
        spec.facet = Some(Facet {
            layout: FacetLayout::Wrap {
                variables: vec!["category".to_string()],
            },
            properties: facet_properties,
            label_mapping: None,
            label_template: "{}".to_string(),
            resolved: true,
        });

        // Add scale with explicit domain that should be skipped
        let mut x_scale = Scale::new("x");
        x_scale.input_range = Some(vec![
            crate::plot::ArrayElement::Number(0.0),
            crate::plot::ArrayElement::Number(100.0),
        ]);
        spec.scales.push(x_scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "A", "B"],
            "__ggsql_aes_panel__" => &["A", "A", "B"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify resolve.scale is set to independent for both axes
        assert_eq!(
            vl_spec["resolve"]["scale"]["x"], "independent",
            "x scale should be independent"
        );
        assert_eq!(
            vl_spec["resolve"]["scale"]["y"], "independent",
            "y scale should be independent"
        );

        // Verify NO explicit domain is set on x encoding (would override free scales)
        // The encoding should exist but scale.domain should not be present
        let x_encoding = &vl_spec["spec"]["layer"][0]["encoding"]["x"];
        let has_domain = x_encoding
            .get("scale")
            .and_then(|s| s.get("domain"))
            .is_some();
        assert!(
            !has_domain,
            "x encoding should NOT have explicit domain when using free scales, got: {}",
            serde_json::to_string_pretty(&x_encoding).unwrap()
        );
    }

    #[test]
    fn test_facet_free_y_only_omits_y_domain() {
        // Test that FACET with scales => 'free_y' omits y domain but keeps x domain
        use crate::plot::scale::Scale;
        use crate::plot::{Facet, FacetLayout, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add facet with free_y scales
        let mut facet_properties = HashMap::new();
        facet_properties.insert("scales".to_string(), ParameterValue::String("free_y".to_string()));
        spec.facet = Some(Facet {
            layout: FacetLayout::Wrap {
                variables: vec!["category".to_string()],
            },
            properties: facet_properties,
            label_mapping: None,
            label_template: "{}".to_string(),
            resolved: true,
        });

        // Add scales with explicit domains
        let mut x_scale = Scale::new("x");
        x_scale.input_range = Some(vec![
            crate::plot::ArrayElement::Number(0.0),
            crate::plot::ArrayElement::Number(100.0),
        ]);
        spec.scales.push(x_scale);

        let mut y_scale = Scale::new("y");
        y_scale.input_range = Some(vec![
            crate::plot::ArrayElement::Number(0.0),
            crate::plot::ArrayElement::Number(50.0),
        ]);
        spec.scales.push(y_scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "A", "B"],
            "__ggsql_aes_panel__" => &["A", "A", "B"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify only y scale is independent
        assert!(
            vl_spec["resolve"]["scale"].get("x").is_none(),
            "x scale should not be in resolve (shared)"
        );
        assert_eq!(
            vl_spec["resolve"]["scale"]["y"], "independent",
            "y scale should be independent"
        );

        // x encoding SHOULD have domain (not free)
        let x_encoding = &vl_spec["spec"]["layer"][0]["encoding"]["x"];
        let x_has_domain = x_encoding
            .get("scale")
            .and_then(|s| s.get("domain"))
            .is_some();
        assert!(
            x_has_domain,
            "x encoding SHOULD have domain when using free_y scales"
        );

        // y encoding should NOT have domain (free)
        let y_encoding = &vl_spec["spec"]["layer"][0]["encoding"]["y"];
        let y_has_domain = y_encoding
            .get("scale")
            .and_then(|s| s.get("domain"))
            .is_some();
        assert!(
            !y_has_domain,
            "y encoding should NOT have domain when using free_y scales, got: {}",
            serde_json::to_string_pretty(&y_encoding).unwrap()
        );
    }

    #[test]
    fn test_facet_fixed_scales_keeps_domain() {
        // Test that FACET with scales => 'fixed' (default) keeps explicit domains
        use crate::plot::scale::Scale;
        use crate::plot::{Facet, FacetLayout, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add facet with fixed scales (default)
        let mut facet_properties = HashMap::new();
        facet_properties.insert("scales".to_string(), ParameterValue::String("fixed".to_string()));
        spec.facet = Some(Facet {
            layout: FacetLayout::Wrap {
                variables: vec!["category".to_string()],
            },
            properties: facet_properties,
            label_mapping: None,
            label_template: "{}".to_string(),
            resolved: true,
        });

        // Add scale with explicit domain
        let mut x_scale = Scale::new("x");
        x_scale.input_range = Some(vec![
            crate::plot::ArrayElement::Number(0.0),
            crate::plot::ArrayElement::Number(100.0),
        ]);
        spec.scales.push(x_scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "A", "B"],
            "__ggsql_aes_panel__" => &["A", "A", "B"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify NO resolve.scale (fixed scales don't need it)
        assert!(
            vl_spec.get("resolve").is_none(),
            "Fixed scales should not have resolve property"
        );

        // x encoding SHOULD have domain (fixed scales keep explicit domains)
        let x_encoding = &vl_spec["spec"]["layer"][0]["encoding"]["x"];
        let has_domain = x_encoding
            .get("scale")
            .and_then(|s| s.get("domain"))
            .is_some();
        assert!(
            has_domain,
            "x encoding SHOULD have domain when using fixed scales"
        );
    }
}
