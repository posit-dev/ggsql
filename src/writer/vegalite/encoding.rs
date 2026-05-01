//! Encoding channel construction for Vega-Lite writer
//!
//! This module handles building Vega-Lite encoding channels from ggsql aesthetic mappings,
//! including type inference, scale properties, and title handling.

use crate::array_util::as_str;
use crate::plot::aesthetic::{is_position_aesthetic, AestheticContext};
use crate::plot::scale::{linetype_to_stroke_dash, shape_to_svg_path, ScaleTypeKind};
use crate::plot::ParameterValue;
use crate::{AestheticValue, DataFrame, GgsqlError, Plot, Result};
use arrow::array::Array;
use arrow::datatypes::DataType;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use super::{POINTS_TO_AREA, POINTS_TO_PIXELS};

/// Check if a position aesthetic has free scales enabled.
fn is_free(aesthetic: &str, facet: Option<&crate::plot::Facet>) -> bool {
    facet.is_some_and(|f| f.is_free(aesthetic))
}

/// Build a Vega-Lite labelExpr from label mappings
///
/// Generates a conditional expression that renames or suppresses labels:
/// - `Some(label)` -> rename to that label
/// - `None` -> suppress label (empty string)
///
/// For non-temporal scales:
/// - Uses `datum.label` for comparisons
/// - Example: `"datum.label == 'A' ? 'Alpha' : datum.label == 'B' ? 'Beta' : datum.label"`
///
/// For temporal scales:
/// - Uses `utcFormat(datum.value, 'fmt')` for comparisons (UTC to match our ISO date strings)
/// - This is necessary because `datum.label` contains Vega-Lite's formatted label (e.g., "Jan 1, 2024")
///   but our label_mapping keys are ISO format strings (e.g., "2024-01-01")
/// - Example: `"utcFormat(datum.value, '%Y-%m-%d') == '2024-01-01' ? 'Q1 Start' : datum.label"`
///
/// For threshold scales (binned legends):
/// - The `null_key` parameter specifies which key should use `datum.label == null` instead of
///   a string comparison. This is needed because Vega-Lite's threshold scale uses null for
///   the first bin's label value.
pub(super) fn build_label_expr(
    mappings: &HashMap<String, Option<String>>,
    time_format: Option<&str>,
    null_key: Option<&str>,
) -> String {
    if mappings.is_empty() {
        return "datum.label".to_string();
    }

    // Build the comparison expression based on whether this is temporal
    // Use utcFormat (not timeFormat) because ggsql writes ISO date strings as UTC.
    // timeFormat uses the browser's local timezone, causing comparison mismatches
    // (e.g., "2024-01-01" UTC midnight becomes "2023-12-31" in US timezones).
    let comparison_expr = match time_format {
        Some(fmt) => format!("utcFormat(datum.value, '{}')", fmt),
        None => "datum.label".to_string(),
    };

    let mut parts: Vec<String> = mappings
        .iter()
        .map(|(from, to)| {
            let from_escaped = super::escape_vega_string(from);

            // For threshold scales, the first terminal uses null instead of string comparison
            let condition = if null_key == Some(from.as_str()) {
                "datum.label == null".to_string()
            } else {
                format!("{} == '{}'", comparison_expr, from_escaped)
            };

            match to {
                Some(label) => {
                    let to_escaped = super::escape_vega_string(label);
                    format!("{} ? '{}'", condition, to_escaped)
                }
                None => {
                    // NULL suppresses the label (empty string)
                    format!("{} ? ''", condition)
                }
            }
        })
        .collect();

    // Fallback to original label
    parts.push("datum.label".to_string());
    parts.join(" : ")
}

/// Build label mappings for threshold scale symbol legends
///
/// Maps Vega-Lite's auto-generated range labels to our desired labels.
/// VL format: "<low> – <high>" for most bins (en-dash U+2013), "≥ <low>" for last bin.
///
/// # Arguments
/// * `breaks` - All break values including terminals [0, 25, 50, 75, 100]
/// * `label_mapping` - Our desired labels keyed by break value string
/// * `closed` - Which side of bin is closed: "left" (default) or "right"
///
/// # Returns
/// HashMap mapping Vega-Lite's predicted labels to our replacement labels
pub(super) fn build_symbol_legend_label_mapping(
    breaks: &[crate::plot::ArrayElement],
    label_mapping: &HashMap<String, Option<String>>,
    closed: &str,
) -> HashMap<String, Option<String>> {
    let mut result = HashMap::new();

    // We have N breaks = N-1 bins
    // legend.values has N-1 entries (last terminal excluded for symbol legends)
    if breaks.len() < 2 {
        return result;
    }
    let num_bins = breaks.len() - 1;

    for i in 0..num_bins {
        let lower = &breaks[i];
        let upper = &breaks[i + 1];
        let lower_str = lower.to_key_string();
        let upper_str = upper.to_key_string();

        // Get our desired label for this bin (keyed by lower bound)
        let our_label = label_mapping.get(&lower_str).cloned().flatten();

        // Predict Vega-Lite's generated label
        // All but last: "<lower> – <upper>" (en-dash U+2013 with spaces)
        // Last bin: "≥ <lower>" (greater-than-or-equal U+2265)
        let vl_label = if i == num_bins - 1 {
            format!("≥ {}", lower_str)
        } else {
            format!("{} – {}", lower_str, upper_str)
        };

        // Check if terminals are suppressed (mapped to None)
        let lower_suppressed = label_mapping.get(&lower_str) == Some(&None);
        let upper_suppressed = label_mapping.get(&upper_str) == Some(&None);

        // Get labels for building range format (fall back to break values)
        let lower_label = our_label.clone().unwrap_or_else(|| lower_str.clone());
        let upper_label = label_mapping
            .get(&upper_str)
            .cloned()
            .flatten()
            .unwrap_or_else(|| upper_str.clone());

        // Determine the replacement label
        // Priority: terminal suppression → range format with custom labels
        let replacement = if i == 0 && lower_suppressed {
            // First bin with suppressed lower terminal → open format
            let symbol = if closed == "right" { "≤" } else { "<" };
            Some(format!("{} {}", symbol, upper_label))
        } else if i == num_bins - 1 && upper_suppressed {
            // Last bin with suppressed upper terminal → open format
            let symbol = if closed == "right" { ">" } else { "≥" };
            Some(format!("{} {}", symbol, lower_label))
        } else {
            // Use range format with custom labels: "<lower_label> – <upper_label>"
            Some(format!("{} – {}", lower_label, upper_label))
        };

        result.insert(vl_label, replacement);
    }

    result
}

/// Count the number of binned material scales in the spec.
/// This is used to determine if legends should use symbol style (which requires
/// removing the last terminal value) or gradient style (which keeps all values).
pub(super) fn count_binned_legend_scales(spec: &Plot) -> usize {
    spec.scales
        .iter()
        .filter(|scale| {
            // Check if binned
            let is_binned = scale
                .scale_type
                .as_ref()
                .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
                .unwrap_or(false);

            // Check if material aesthetic
            let is_material_aesthetic = !is_position_aesthetic(&scale.aesthetic);

            is_binned && is_material_aesthetic
        })
        .count()
}

/// Check if a string (Utf8) column contains numeric values
pub(super) fn is_numeric_string_column(array: &arrow::array::ArrayRef) -> bool {
    if let Ok(ca) = as_str(array) {
        // Check first few non-null values to see if they're numeric
        for i in 0..ca.len().min(5) {
            if ca.is_null(i) {
                continue;
            }
            if ca.value(i).parse::<f64>().is_err() {
                return false;
            }
        }
        true
    } else {
        false
    }
}

/// Infer Vega-Lite field type from DataFrame column
pub(super) fn infer_field_type(df: &DataFrame, field: &str) -> String {
    if let Ok(column) = df.column(field) {
        match column.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => "quantitative",
            DataType::Boolean => "nominal",
            DataType::Utf8
                // Check if string column contains numeric values
                if is_numeric_string_column(column) =>
            {
                "quantitative"
            }
            DataType::Date32 | DataType::Timestamp(_, _) | DataType::Time64(_) => "temporal",
            _ => "nominal",
        }
        .to_string()
    } else {
        "nominal".to_string()
    }
}

/// Determine Vega-Lite field type from scale specification
pub(super) fn determine_field_type_from_scale(
    scale: &crate::plot::Scale,
    inferred: &str,
    _aesthetic: &str,
    identity_scale: &mut bool,
) -> String {
    // Use scale type if explicitly specified
    if let Some(scale_type) = &scale.scale_type {
        use crate::plot::ScaleTypeKind;
        match scale_type.scale_type_kind() {
            ScaleTypeKind::Continuous => "quantitative",
            ScaleTypeKind::Discrete => "nominal",
            ScaleTypeKind::Binned => "quantitative", // Binned data is still quantitative
            ScaleTypeKind::Ordinal => "ordinal",     // Native Vega-Lite ordinal type
            ScaleTypeKind::Identity => {
                *identity_scale = true;
                inferred
            }
        }
        .to_string()
    } else {
        // Scale exists but no type specified, use inferred
        inferred.to_string()
    }
}

// =============================================================================
// Phase 1: Utility Helpers
// =============================================================================

/// Legend display style for binned scales
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegendStyle {
    /// Gradient legend (continuous color bar)
    Gradient,
    /// Symbol legend (discrete color blocks)
    Symbol,
}

/// Determine legend style for a binned aesthetic
///
/// - fill/stroke alone: gradient legend
/// - fill/stroke with other binned material aesthetics: symbol legend
/// - all other aesthetics: symbol legend
fn determine_legend_style(aesthetic: &str, spec: &Plot) -> LegendStyle {
    let is_gradient_aesthetic = matches!(aesthetic, "fill" | "stroke");
    if !is_gradient_aesthetic {
        return LegendStyle::Symbol;
    }

    // For fill/stroke, check if there are multiple binned legend scales
    let binned_legend_count = count_binned_legend_scales(spec);
    if binned_legend_count > 1 {
        LegendStyle::Symbol
    } else {
        LegendStyle::Gradient
    }
}

/// Safely insert a property into the axis object of an encoding
///
/// Creates the axis object if it doesn't exist, preserves existing properties.
/// Does nothing if axis is explicitly set to null.
fn insert_axis_property(encoding: &mut Value, key: &str, value: Value) {
    // Skip if axis is explicitly null
    if encoding.get("axis").is_some_and(|v| v.is_null()) {
        return;
    }

    let axis = encoding.get_mut("axis").and_then(|v| v.as_object_mut());
    if let Some(axis_map) = axis {
        axis_map.insert(key.to_string(), value);
    } else {
        encoding["axis"] = json!({ key: value });
    }
}

/// Safely insert a property into the legend object of an encoding
///
/// Creates the legend object if it doesn't exist, preserves existing properties.
/// Does nothing if legend is explicitly set to null.
fn insert_legend_property(encoding: &mut Value, key: &str, value: Value) {
    // Skip if legend is explicitly null
    if encoding.get("legend").is_some_and(|v| v.is_null()) {
        return;
    }

    let legend = encoding.get_mut("legend").and_then(|v| v.as_object_mut());
    if let Some(legend_map) = legend {
        legend_map.insert(key.to_string(), value);
    } else {
        encoding["legend"] = json!({ key: value });
    }
}

// =============================================================================
// Phase 2: Logical Section Helpers
// =============================================================================

/// Determine the Vega-Lite field type for an aesthetic mapping
///
/// Checks scale specifications and transforms to determine the appropriate
/// Vega-Lite field type (quantitative, temporal, nominal, ordinal).
fn determine_field_type_for_aesthetic(
    aesthetic: &str,
    col: &str,
    df: &DataFrame,
    spec: &Plot,
    identity_scale: &mut bool,
    aesthetic_ctx: &AestheticContext,
) -> String {
    let primary = aesthetic_ctx
        .primary_internal_position(aesthetic)
        .unwrap_or(aesthetic);
    let inferred = infer_field_type(df, col);

    if let Some(scale) = spec.find_scale(primary) {
        // Check if the transform indicates temporal data
        // (Transform takes precedence since it's resolved from column dtype)
        if let Some(ref transform) = scale.transform {
            if transform.is_temporal() {
                return "temporal".to_string();
            }
        }
        // Check scale type
        determine_field_type_from_scale(scale, &inferred, aesthetic, identity_scale)
    } else {
        // No scale specification, infer from data
        inferred
    }
}

/// Apply title to encoding based on aesthetic family rules
///
/// - Primary aesthetics (x, y, color) can set the title
/// - Variant aesthetics (xmin, ymin, etc.) only get title if no primary exists
/// - When a primary exists, variants get title: null to prevent axis label conflicts
fn apply_title_to_encoding(
    encoding: &mut Value,
    aesthetic: &str,
    original_name: &Option<String>,
    spec: &Plot,
    titled_families: &mut HashSet<String>,
    primary_aesthetics: &HashSet<String>,
    aesthetic_ctx: &AestheticContext,
) {
    let primary = aesthetic_ctx
        .primary_internal_position(aesthetic)
        .unwrap_or(aesthetic);
    let is_primary = aesthetic == primary;
    let primary_exists = primary_aesthetics.contains(primary);

    if is_primary && !titled_families.contains(primary) {
        // Primary aesthetic: set title from explicit label or original_name
        let explicit_label = spec
            .labels
            .as_ref()
            .and_then(|labels| labels.labels.get(primary));

        if let Some(label_opt) = explicit_label {
            match label_opt {
                Some(label) => {
                    encoding["title"] = super::split_label_on_newlines(label);
                }
                None => {
                    encoding["title"] = Value::Null;
                }
            }
            titled_families.insert(primary.to_string());
        } else if let Some(orig) = original_name {
            // Use original column name as default title when available
            encoding["title"] = json!(orig);
            titled_families.insert(primary.to_string());
        }
    } else if !is_primary && primary_exists {
        // Variant with primary present: suppress title to avoid axis label conflicts
        encoding["title"] = Value::Null;
    } else if !is_primary && !primary_exists && !titled_families.contains(primary) {
        // Variant without primary: allow first variant to claim title (for explicit labels)
        if let Some(ref labels) = spec.labels {
            if let Some(label_opt) = labels.labels.get(primary) {
                match label_opt {
                    Some(label) => {
                        encoding["title"] = super::split_label_on_newlines(label);
                    }
                    None => {
                        encoding["title"] = Value::Null;
                    }
                }
                titled_families.insert(primary.to_string());
            }
        }
    }
}

/// Parameters for building scale properties
struct ScaleContext<'a> {
    aesthetic: &'a str,
    is_binned_legend: bool,
    #[allow(dead_code)]
    spec: &'a Plot, // Reserved for future use (e.g., multi-scale legend decisions)
}

/// Build scale properties from SCALE clause
///
/// Returns the scale object and whether a gradient legend is needed.
fn build_scale_properties(
    scale: &crate::plot::Scale,
    ctx: &ScaleContext,
) -> (serde_json::Map<String, Value>, bool) {
    use crate::plot::{OutputRange, ParameterValue};

    let mut scale_obj = serde_json::Map::new();
    let mut needs_gradient_legend = false;

    // Check if we should skip domain due to facet free scales
    // When using free scales, Vega-Lite computes independent domains per facet panel.
    // Setting an explicit domain would override this behavior.
    // Note: aesthetics are in internal format (pos1, pos2) at this stage
    let skip_domain = is_free(ctx.aesthetic, ctx.spec.facet.as_ref());

    // Apply domain from input_range (FROM clause)
    // Skip for threshold scales - they use internal breaks as domain instead
    // Skip for free facet scales - Vega-Lite should compute independent domains
    if !ctx.is_binned_legend && !skip_domain {
        if let Some(ref domain_values) = scale.input_range {
            let domain_json: Vec<Value> = domain_values.iter().map(|elem| elem.to_json()).collect();
            scale_obj.insert("domain".to_string(), json!(domain_json));
        }
    }

    // Apply range from output_range (TO clause)
    if let Some(ref output_range) = scale.output_range {
        match output_range {
            OutputRange::Array(range_values) => {
                let range_json: Vec<Value> = range_values
                    .iter()
                    .map(|elem| convert_range_element(elem, ctx.aesthetic))
                    .collect();
                scale_obj.insert("range".to_string(), json!(range_json));

                // For continuous color scales with range array, use gradient legend
                if matches!(ctx.aesthetic, "fill" | "stroke")
                    && matches!(
                        scale.scale_type.as_ref().map(|st| st.scale_type_kind()),
                        Some(ScaleTypeKind::Continuous)
                    )
                {
                    needs_gradient_legend = true;
                }
            }
            OutputRange::Palette(palette_name) => {
                scale_obj.insert("scheme".to_string(), json!(palette_name.to_lowercase()));
            }
        }
    }

    // Handle transform (VIA clause)
    if let Some(ref transform) = scale.transform {
        apply_transform_to_scale(&mut scale_obj, transform);
    }

    // Handle binned material aesthetics with threshold scale
    if ctx.is_binned_legend {
        scale_obj.insert("type".to_string(), json!("threshold"));

        // Threshold domain = internal breaks (excluding first and last terminal bounds)
        if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") {
            if breaks.len() > 2 {
                let internal_breaks: Vec<Value> = breaks[1..breaks.len() - 1]
                    .iter()
                    .map(|e| e.to_json())
                    .collect();
                scale_obj.insert("domain".to_string(), json!(internal_breaks));
            }
        }
    }

    // Handle reverse property (SETTING clause)
    if let Some(ParameterValue::Boolean(true)) = scale.properties.get("reverse") {
        scale_obj.insert("reverse".to_string(), json!(true));
    }

    (scale_obj, needs_gradient_legend)
}

/// Convert a range array element to JSON with aesthetic-specific transformations
fn convert_range_element(elem: &crate::plot::ArrayElement, aesthetic: &str) -> Value {
    use crate::plot::ArrayElement;

    match elem {
        ArrayElement::String(s) => {
            // For shape aesthetic, convert to SVG path
            if aesthetic == "shape" {
                if let Some(svg_path) = shape_to_svg_path(s) {
                    return json!(svg_path);
                }
            // For linetype aesthetic, convert to dash array
            } else if aesthetic == "linetype" {
                if let Some(dash_array) = linetype_to_stroke_dash(s) {
                    return json!(dash_array);
                }
            }
            json!(s)
        }
        ArrayElement::Number(n) => {
            match aesthetic {
                // Size: convert radius (points) to area (pixels²)
                "size" => json!(n * n * POINTS_TO_AREA),
                // Linewidth: convert points to pixels
                "linewidth" | "fontsize" => json!(n * POINTS_TO_PIXELS),
                // Other aesthetics: pass through unchanged
                _ => json!(n),
            }
        }
        other => other.to_json(),
    }
}

/// Apply transform (VIA clause) to scale object
fn apply_transform_to_scale(
    scale_obj: &mut serde_json::Map<String, Value>,
    transform: &crate::plot::scale::Transform,
) {
    use crate::plot::scale::TransformKind;

    match transform.transform_kind() {
        TransformKind::Identity => {} // Linear (default)
        TransformKind::Log10 => {
            scale_obj.insert("type".to_string(), json!("log"));
            scale_obj.insert("base".to_string(), json!(10));
            scale_obj.insert("zero".to_string(), json!(false));
        }
        TransformKind::Log => {
            scale_obj.insert("type".to_string(), json!("log"));
            scale_obj.insert("base".to_string(), json!(std::f64::consts::E));
            scale_obj.insert("zero".to_string(), json!(false));
        }
        TransformKind::Log2 => {
            scale_obj.insert("type".to_string(), json!("log"));
            scale_obj.insert("base".to_string(), json!(2));
            scale_obj.insert("zero".to_string(), json!(false));
        }
        TransformKind::Sqrt => {
            scale_obj.insert("type".to_string(), json!("sqrt"));
        }
        TransformKind::Square => {
            scale_obj.insert("type".to_string(), json!("pow"));
            scale_obj.insert("exponent".to_string(), json!(2));
        }
        TransformKind::Exp10 | TransformKind::Exp2 | TransformKind::Exp => {
            eprintln!(
                "Warning: {} transform has no native Vega-Lite equivalent, using linear scale",
                transform.name()
            );
        }
        TransformKind::Asinh | TransformKind::PseudoLog => {
            scale_obj.insert("type".to_string(), json!("symlog"));
        }
        // Temporal transforms: field type ("temporal") is set elsewhere
        TransformKind::Date | TransformKind::DateTime | TransformKind::Time => {}
        // Discrete transforms: data casting happens at SQL level
        TransformKind::String | TransformKind::Bool => {}
        // Integer transform: casting happens at SQL level
        TransformKind::Integer => {}
    }
}

/// Apply legend reversal for discrete/ordinal scales with reverse property
fn apply_reverse_legend(encoding: &mut Value, scale: &crate::plot::Scale, aesthetic: &str) {
    use crate::plot::ParameterValue;

    // Only process if reverse is true
    let Some(ParameterValue::Boolean(true)) = scale.properties.get("reverse") else {
        return;
    };

    // Only for discrete/ordinal scales
    let Some(ref scale_type) = scale.scale_type else {
        return;
    };
    let kind = scale_type.scale_type_kind();
    if !matches!(kind, ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal) {
        return;
    }

    // Only for material aesthetics (those with legends)
    if is_position_aesthetic(aesthetic) {
        return;
    }

    // Use the input_range (domain) if available
    if let Some(ref domain) = scale.input_range {
        let reversed_domain: Vec<Value> = domain.iter().rev().map(|e| e.to_json()).collect();
        insert_legend_property(encoding, "values", json!(reversed_domain));
    }
}

/// Apply breaks to encoding (axis.values or legend.values)
fn apply_breaks_to_encoding(
    encoding: &mut Value,
    scale: &crate::plot::Scale,
    aesthetic: &str,
    is_binned_legend: bool,
    spec: &Plot,
) {
    use crate::plot::ParameterValue;

    let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") else {
        return;
    };

    let all_values: Vec<Value> = breaks.iter().map(|e| e.to_json()).collect();

    if is_position_aesthetic(aesthetic) {
        // For position aesthetics (axes), filter out suppressed terminal breaks
        let axis_values: Vec<Value> = if let Some(ref label_mapping) = scale.label_mapping {
            breaks
                .iter()
                .filter(|e| {
                    let key = e.to_key_string();
                    !matches!(label_mapping.get(&key), Some(None))
                })
                .map(|e| e.to_json())
                .collect()
        } else {
            all_values
        };

        insert_axis_property(encoding, "values", json!(axis_values));
    } else {
        // For material aesthetics, determine values based on legend style
        let legend_values = if is_binned_legend {
            let legend_style = determine_legend_style(aesthetic, spec);
            if legend_style == LegendStyle::Symbol && !all_values.is_empty() {
                // Remove the last terminal for symbol legends
                all_values[..all_values.len() - 1].to_vec()
            } else {
                all_values
            }
        } else {
            all_values
        };

        insert_legend_property(encoding, "values", json!(legend_values));
    }
}

/// Apply label mapping (RENAMING clause) via labelExpr
fn apply_label_mapping_to_encoding(
    encoding: &mut Value,
    scale: &crate::plot::Scale,
    aesthetic: &str,
    is_binned_legend: bool,
    spec: &Plot,
) {
    use crate::plot::scale::TransformKind;
    use crate::plot::ParameterValue;

    let Some(ref label_mapping) = scale.label_mapping else {
        return;
    };
    if label_mapping.is_empty() {
        return;
    }

    // For temporal scales, use utcFormat() to compare against ISO keys
    let time_format = scale
        .transform
        .as_ref()
        .and_then(|t| match t.transform_kind() {
            TransformKind::Date => Some("%Y-%m-%d"),
            TransformKind::DateTime => Some("%Y-%m-%dT%H:%M:%S"),
            TransformKind::Time => Some("%H:%M:%S"),
            _ => None,
        });

    // Build the mapping and null_key based on legend style
    let (filtered_mapping, null_key) = if is_binned_legend {
        let legend_style = determine_legend_style(aesthetic, spec);

        if legend_style == LegendStyle::Symbol {
            // Symbol legend: map VL's range-style labels to our labels
            let closed = scale
                .properties
                .get("closed")
                .and_then(|v| {
                    if let ParameterValue::String(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("left");

            if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") {
                let symbol_mapping =
                    build_symbol_legend_label_mapping(breaks, label_mapping, closed);
                (symbol_mapping, None)
            } else {
                (label_mapping.clone(), None)
            }
        } else {
            // Gradient legend: use null_key for first terminal
            let first_key = scale.properties.get("breaks").and_then(|b| {
                if let ParameterValue::Array(breaks) = b {
                    breaks.first().map(|e| e.to_key_string())
                } else {
                    None
                }
            });
            (label_mapping.clone(), first_key)
        }
    } else {
        (label_mapping.clone(), None)
    };

    let label_expr = build_label_expr(&filtered_mapping, time_format, null_key.as_deref());

    if is_position_aesthetic(aesthetic) {
        insert_axis_property(encoding, "labelExpr", json!(label_expr));
    } else {
        insert_legend_property(encoding, "labelExpr", json!(label_expr));
    }
}

// =============================================================================
// Main Function
// =============================================================================

/// Context for building encoding channels
///
/// Groups shared state to reduce function argument count.
pub(super) struct EncodingContext<'a> {
    pub df: &'a DataFrame,
    pub spec: &'a Plot,
    pub titled_families: &'a mut HashSet<String>,
    pub primary_aesthetics: &'a HashSet<String>,
}

/// Build encoding channel from aesthetic mapping
///
/// The `titled_families` set tracks which aesthetic families have already received
/// a title, ensuring only one title per family (e.g., one title for x/xmin/xmax).
///
/// The `primary_aesthetics` set contains primary aesthetics that exist in the layer.
/// When a primary exists, variant aesthetics (xmin, ymin, etc.) get `title: null`.
pub(super) fn build_encoding_channel(
    aesthetic: &str,
    value: &AestheticValue,
    ctx: &mut EncodingContext,
) -> Result<Value> {
    match value {
        AestheticValue::Column {
            name: col,
            original_name,
            is_dummy,
        } => build_column_encoding(aesthetic, col, original_name, *is_dummy, true, ctx),
        AestheticValue::AnnotationColumn { name: col } => {
            // Material annotation columns use identity scale
            build_column_encoding(aesthetic, col, &None, false, false, ctx)
        }
        AestheticValue::Literal(lit) => build_literal_encoding(aesthetic, lit),
    }
}

/// Build encoding for a column-mapped aesthetic
fn build_column_encoding(
    aesthetic: &str,
    col: &str,
    original_name: &Option<String>,
    is_dummy: bool,
    is_scaled: bool,
    ctx: &mut EncodingContext,
) -> Result<Value> {
    let aesthetic_ctx = ctx.spec.get_aesthetic_context();
    let primary = aesthetic_ctx
        .primary_internal_position(aesthetic)
        .unwrap_or(aesthetic);
    let mut identity_scale = !is_scaled;

    // Determine field type from scale or infer from data
    let field_type = determine_field_type_for_aesthetic(
        aesthetic,
        col,
        ctx.df,
        ctx.spec,
        &mut identity_scale,
        &aesthetic_ctx,
    );

    // Check if this aesthetic has a binned scale
    let is_binned = ctx
        .spec
        .find_scale(primary)
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
        .unwrap_or(false);

    // Binned legend = binned + material (needs threshold scale)
    let is_binned_legend = is_binned && !is_position_aesthetic(aesthetic);

    // Build base encoding
    let mut encoding = json!({
        "field": col,
        "type": field_type,
    });

    // bin: "binned" is only valid for position channels in VL v6
    if is_binned && !is_binned_legend {
        encoding["bin"] = json!("binned");
    }

    // Apply title handling
    apply_title_to_encoding(
        &mut encoding,
        aesthetic,
        original_name,
        ctx.spec,
        ctx.titled_families,
        ctx.primary_aesthetics,
        &aesthetic_ctx,
    );

    // Build scale properties
    let (mut scale_obj, needs_gradient_legend) = if let Some(scale) = ctx.spec.find_scale(primary) {
        let scale_ctx = ScaleContext {
            aesthetic,
            spec: ctx.spec,
            is_binned_legend,
        };
        let (scale_obj, needs_gradient) = build_scale_properties(scale, &scale_ctx);

        // Apply legend reversal for discrete/ordinal scales
        apply_reverse_legend(&mut encoding, scale, aesthetic);

        // Apply breaks to axis.values or legend.values
        apply_breaks_to_encoding(&mut encoding, scale, aesthetic, is_binned_legend, ctx.spec);

        // Apply label mapping via labelExpr
        apply_label_mapping_to_encoding(
            &mut encoding,
            scale,
            aesthetic,
            is_binned_legend,
            ctx.spec,
        );

        (scale_obj, needs_gradient)
    } else {
        (serde_json::Map::new(), false)
    };

    // Position scales don't include zero by default — but only when we set
    // an explicit domain. With free facet scales (no domain), VL computes
    // the domain from data values. Setting zero:false in that case can exclude
    // 0 from the domain, breaking charts with pre-computed stacking (y2/theta2
    // starts at 0). Let VL's defaults handle it instead.
    let is_free = is_free(aesthetic, ctx.spec.facet.as_ref());
    if aesthetic_ctx.is_primary_internal(aesthetic) && !is_free {
        scale_obj.insert("zero".to_string(), json!(false));
    }

    // Apply scale object to encoding
    if identity_scale {
        encoding["scale"] = Value::Null;
    } else if !scale_obj.is_empty() {
        encoding["scale"] = json!(scale_obj);
    }

    // Apply gradient legend type for continuous color scales with range array
    if needs_gradient_legend {
        insert_legend_property(&mut encoding, "type", json!("gradient"));
    }

    // Hide axis for dummy columns
    if is_dummy {
        encoding["axis"] = Value::Null;
    }

    Ok(encoding)
}

/// Build encoding for a literal aesthetic value
fn build_literal_encoding(aesthetic: &str, lit: &ParameterValue) -> Result<Value> {
    let val = match lit {
        ParameterValue::String(s) => {
            let converted = match aesthetic {
                "linetype" => linetype_to_stroke_dash(s).map(|arr| json!(arr)),
                "shape" => shape_to_svg_path(s).map(|arr| json!(arr)),
                _ => None,
            };
            converted.unwrap_or_else(|| json!(s))
        }
        ParameterValue::Number(n) => {
            match aesthetic {
                // Size: radius (points) → area (pixels²)
                "size" => json!(n * n * POINTS_TO_AREA),
                // Linewidth: points → pixels
                "linewidth" | "fontsize" => json!(n * POINTS_TO_PIXELS),
                _ => json!(n),
            }
        }
        ParameterValue::Array(_) => {
            return Err(crate::GgsqlError::WriterError(format!(
                "The `{aes}` SETTING must be scalar, not an array.",
                aes = aesthetic
            )))
        }
        _ => lit.to_json(),
    };
    Ok(json!({"value": val}))
}

/// Map ggsql aesthetic name to Vega-Lite encoding channel name.
///
/// For internal position aesthetics (pos1, pos2, etc.), maps directly to Vega-Lite
/// channel names based on coord type:
/// - Cartesian: pos1 → "x", pos2 → "y"
/// - Polar: pos1 → "radius", pos2 → "theta"
///
/// This ensures correct Vega-Lite channel names regardless of what the user originally
/// called their position aesthetics in the PROJECT clause.
///
/// For material aesthetics, applies Vega-Lite specific mappings (e.g., linetype → strokeDash).
pub(super) fn map_aesthetic_name(
    aesthetic: &str,
    _ctx: &crate::plot::AestheticContext,
    renderer: &dyn super::projection::ProjectionRenderer,
) -> String {
    // For internal position aesthetics, map directly to Vega-Lite channel names
    // based on coord type (ignoring user-facing names)
    if let Some(vl_channel) = super::projection::map_position_to_vegalite(aesthetic, renderer) {
        return vl_channel;
    }

    // Material aesthetics: apply Vega-Lite specific mappings
    match aesthetic {
        // Line aesthetics
        "linetype" => "strokeDash".to_string(),
        "linewidth" => "strokeWidth".to_string(),
        // Text aesthetics
        "label" => "text".to_string(),
        "fontsize" => "size".to_string(),
        // All other aesthetics pass through directly
        // (fill and stroke map to Vega-Lite's separate fill/stroke channels)
        // typeface/fontweight/italic/rotation are parsed explicitly
        _ => aesthetic.to_string(),
    }
}

// =============================================================================
// RenderContext
// =============================================================================

/// Resolved Vega-Lite position channel names for the active coordinate system.
///
/// Order: `(pos1, pos1_end, pos1_offset, pos2, pos2_end, pos2_offset)`
///
/// For Cartesian: `("x", "x2", "xOffset", "y", "y2", "yOffset")`
/// For Polar: `("theta", "theta2", "thetaOffset", "radius", "radius2", "radiusOffset")`
pub type PositionChannels = (String, String, String, String, String, String);

/// Context information available to renderers during layer preparation
pub struct RenderContext<'a> {
    /// Scale definitions (for extent and properties)
    pub scales: &'a [crate::Scale],
    /// Resolved position channel names for the active coordinate system
    pub channels: PositionChannels,
    /// Aesthetic context — used to translate internal aesthetic names back to
    /// user-facing names when reporting errors.
    pub aesthetic_context: crate::plot::aesthetic::AestheticContext,
}

impl<'a> RenderContext<'a> {
    /// Create a new render context
    pub fn new(
        scales: &'a [crate::Scale],
        renderer: &dyn super::projection::ProjectionRenderer,
        aesthetic_context: crate::plot::aesthetic::AestheticContext,
    ) -> Self {
        let pos1 = super::projection::map_position_to_vegalite("pos1", renderer).unwrap();
        let pos1_end = super::projection::map_position_to_vegalite("pos1end", renderer).unwrap();
        let pos2 = super::projection::map_position_to_vegalite("pos2", renderer).unwrap();
        let pos2_end = super::projection::map_position_to_vegalite("pos2end", renderer).unwrap();

        let (pos1_offset, pos2_offset) = renderer.offset_channels();

        Self {
            scales,
            channels: (
                pos1,
                pos1_end,
                pos1_offset.to_string(),
                pos2,
                pos2_end,
                pos2_offset.to_string(),
            ),
            aesthetic_context,
        }
    }

    #[cfg(test)]
    pub fn default_for_test() -> Self {
        let renderer = super::projection::get_projection_renderer(None, None, &[]);
        Self::new(
            &[],
            renderer.as_ref(),
            crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]),
        )
    }

    /// Find a scale by aesthetic name
    pub fn find_scale(&self, aesthetic: &str) -> Option<&crate::Scale> {
        self.scales.iter().find(|s| s.aesthetic == aesthetic)
    }

    /// Get the numeric extent (min, max) for a given aesthetic from its scale
    pub fn get_extent(&self, aesthetic: &str) -> Result<(f64, f64)> {
        use crate::plot::ArrayElement;

        let display_aes = self.aesthetic_context.map_internal_to_user(aesthetic);

        // Find the scale for this aesthetic
        let scale = self.find_scale(aesthetic).ok_or_else(|| {
            GgsqlError::ValidationError(format!(
                "Cannot determine extent for aesthetic '{}': no scale found",
                display_aes
            ))
        })?;

        // Extract continuous range from input_range
        if let Some(range) = &scale.input_range {
            if range.len() >= 2 {
                if let (ArrayElement::Number(min), ArrayElement::Number(max)) =
                    (&range[0], &range[1])
                {
                    return Ok((*min, *max));
                }
            }
        }

        Err(GgsqlError::ValidationError(format!(
            "Cannot determine extent for aesthetic '{}': scale has no valid numeric range",
            display_aes
        )))
    }
}

/// Build detail encoding from partition_by columns
/// Maps partition_by columns to Vega-Lite's detail channel for grouping
pub(super) fn build_detail_encoding(partition_by: &[String]) -> Option<Value> {
    if partition_by.is_empty() {
        return None;
    }

    if partition_by.len() == 1 {
        // Single column: simple object
        Some(json!({
            "field": partition_by[0],
            "type": "nominal"
        }))
    } else {
        // Multiple columns: array of detail specifications
        let details: Vec<Value> = partition_by
            .iter()
            .map(|col| {
                json!({
                    "field": col,
                    "type": "nominal"
                })
            })
            .collect();
        Some(json!(details))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_label_expr_temporal_uses_utc_format() {
        // Temporal label comparisons must use utcFormat (not timeFormat) because
        // ggsql writes ISO date strings as UTC. timeFormat uses the browser's
        // local timezone, causing comparisons to fail in non-UTC timezones
        // (e.g., "2024-01-01" midnight UTC becomes "2023-12-31" in US Central).
        let mut mappings = HashMap::new();
        mappings.insert("2024-01-01".to_string(), Some("Jan 2024".to_string()));

        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None);

        assert!(
            expr.contains("utcFormat("),
            "temporal labelExpr should use utcFormat, got: {expr}"
        );
        assert!(
            !expr.contains("timeFormat("),
            "temporal labelExpr must not use timeFormat (local tz), got: {expr}"
        );
        assert!(
            expr.contains("utcFormat(datum.value, '%Y-%m-%d') == '2024-01-01' ? 'Jan 2024'"),
            "expected correct comparison expression, got: {expr}"
        );
    }

    #[test]
    fn test_build_label_expr_non_temporal_uses_datum_label() {
        let mut mappings = HashMap::new();
        mappings.insert("A".to_string(), Some("Alpha".to_string()));

        let expr = build_label_expr(&mappings, None, None);

        assert!(
            expr.contains("datum.label == 'A'"),
            "non-temporal should use datum.label, got: {expr}"
        );
        assert!(
            !expr.contains("utcFormat("),
            "non-temporal should not use utcFormat, got: {expr}"
        );
    }

    #[test]
    fn test_build_label_expr_fallback() {
        let mappings = HashMap::new();
        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None);
        assert_eq!(
            expr, "datum.label",
            "empty mappings should fall back to datum.label"
        );
    }

    #[test]
    fn test_build_label_expr_null_suppression() {
        let mut mappings = HashMap::new();
        mappings.insert("2024-06-01".to_string(), None); // suppress label

        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None);

        assert!(
            expr.contains("? ''"),
            "None mapping should suppress label (empty string), got: {expr}"
        );
    }

    #[test]
    fn test_literal_shape_converts_to_svg_path() {
        let lit = ParameterValue::String("square".to_string());
        let result = build_literal_encoding("shape", &lit).unwrap();
        let val = &result["value"];
        assert!(val.is_string(), "expected SVG path string, got: {val}");
        let path = val.as_str().unwrap();
        assert!(
            path.starts_with('M') && path.contains('Z'),
            "expected SVG path with M and Z commands, got: {path}"
        );
    }

    #[test]
    fn test_literal_shape_unknown_passes_through() {
        let lit = ParameterValue::String("nonexistent".to_string());
        let result = build_literal_encoding("shape", &lit).unwrap();
        assert_eq!(result, json!({"value": "nonexistent"}));
    }

    // =========================================================================
    // RenderContext::get_extent — internal aesthetic names must be translated
    // back to user-facing names in error messages.
    // =========================================================================

    mod get_extent_translation_tests {
        use crate::writer::vegalite::projection::get_projection_renderer;
        use super::*;
        use crate::plot::aesthetic::AestheticContext;
        use crate::plot::{ArrayElement, Scale};

        fn discrete_scale(aesthetic: &str) -> Scale {
            Scale {
                aesthetic: aesthetic.to_string(),
                scale_type: None,
                input_range: Some(vec![ArrayElement::String("A".to_string())]),
                explicit_input_range: false,
                output_range: None,
                transform: None,
                explicit_transform: false,
                properties: std::collections::HashMap::new(),
                resolved: false,
                label_mapping: None,
                label_template: "{}".to_string(),
            }
        }

        #[test]
        fn no_scale_found_translates_pos1_to_x_under_cartesian() {
            let scales: Vec<Scale> = vec![];
            let ctx = RenderContext::new(
                &scales,
                get_projection_renderer(None, None, &[]).as_ref(),
                AestheticContext::from_static(&["x", "y"], &[]),
            );
            let err = ctx.get_extent("pos1").unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Cannot determine extent for aesthetic 'x': no scale found"
            );
        }

        #[test]
        fn no_scale_found_translates_pos1_to_angle_under_polar() {
            let scales: Vec<Scale> = vec![];
            let ctx = RenderContext::new(
                &scales,
                get_projection_renderer(None, None, &[]).as_ref(),
                AestheticContext::from_static(&["angle", "radius"], &[]),
            );
            let err = ctx.get_extent("pos1").unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Cannot determine extent for aesthetic 'angle': no scale found"
            );
        }

        #[test]
        fn no_numeric_range_translates_pos2_to_y_under_cartesian() {
            // Scale exists, but input_range is non-numeric (discrete) so
            // get_extent returns the second error.
            let scales = vec![discrete_scale("pos2")];
            let ctx = RenderContext::new(
                &scales,
                get_projection_renderer(None, None, &[]).as_ref(),
                AestheticContext::from_static(&["x", "y"], &[]),
            );
            let err = ctx.get_extent("pos2").unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Cannot determine extent for aesthetic 'y': scale has no valid numeric range"
            );
        }
    }
}
