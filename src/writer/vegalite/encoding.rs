//! Encoding channel construction for Vega-Lite writer
//!
//! This module handles building Vega-Lite encoding channels from ggsql aesthetic mappings,
//! including type inference, scale properties, and title handling.

use crate::array_util::as_str;
use crate::plot::aesthetic::{is_position_aesthetic, AestheticContext};
use crate::plot::scale::{linetype_to_stroke_dash, shape_to_svg_path, ScaleTypeKind};
use crate::plot::{ParameterValue, Scale};
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

/// Whether a scale lays its input out as bands rather than as a continuum.
fn is_categorical(scale: &crate::Scale) -> bool {
    matches!(
        scale.scale_type.as_ref().map(|st| st.scale_type_kind()),
        Some(ScaleTypeKind::Discrete) | Some(ScaleTypeKind::Ordinal)
    )
}

/// Build a Vega-Lite labelExpr from label mappings
///
/// Generates a conditional expression that renames or suppresses labels:
/// - `Some(label)` -> rename to that label
/// - `None` -> suppress label (empty string)
///
/// For nominal/ordinal scales:
/// - Uses `datum.label` for comparisons
/// - Example: `"datum.label == 'A' ? 'Alpha' : datum.label == 'B' ? 'Beta' : datum.label"`
///
/// For quantitative scales:
/// - Uses `datum.value` for comparisons (numeric, no quotes) to avoid locale formatting
///   mismatches (e.g., `datum.label` may contain thousand separators like "2,020")
/// - Example: `"datum.value == 2020 ? '2020.0' : datum.label"`
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
    field_type: &str,
) -> String {
    if mappings.is_empty() {
        return "datum.label".to_string();
    }

    let is_numeric = field_type == "quantitative";

    // Build the comparison expression based on scale type.
    // - Temporal: use utcFormat(datum.value, fmt) because timeFormat uses local tz.
    // - Numeric: use datum.value to avoid locale thousand-separator mismatches.
    // - Otherwise: use datum.label (categorical/string).
    let comparison_expr = match time_format {
        Some(fmt) => format!("utcFormat(datum.value, '{}')", fmt),
        None if is_numeric => "datum.value".to_string(),
        None => "datum.label".to_string(),
    };

    let mut parts: Vec<String> = mappings
        .iter()
        .map(|(from, to)| {
            let from_escaped = super::escape_vega_string(from);

            // For threshold scales, the first terminal uses null instead of string comparison
            let condition = if null_key == Some(from.as_str()) {
                "datum.label == null".to_string()
            } else if is_numeric && time_format.is_none() {
                format!("{} == {}", comparison_expr, from_escaped)
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

/// Build label mappings for threshold scale symbol legends.
///
/// Vega-Lite generates its own text for each bin of a symbol legend — a
/// `"<low> – <high>"` range (en dash U+2013) for every bin but the last, which
/// it renders as `"≥ <low>"`. Those strings are the keys a `labelExpr` has to
/// match, so this pairs each with the label ggsql resolved for that bin
/// ([`Scale::binned_bins`], shared with the raster writer).
pub(super) fn build_symbol_legend_label_mapping(scale: &Scale) -> HashMap<String, Option<String>> {
    let bins = scale.binned_bins();
    let last = bins.len().saturating_sub(1);
    bins.iter()
        .enumerate()
        .map(|(i, bin)| {
            let vl_label = if i == last {
                format!("≥ {}", bin.lower.to_key_string())
            } else {
                format!(
                    "{} – {}",
                    bin.lower.to_key_string(),
                    bin.upper.to_key_string()
                )
            };
            (vl_label, Some(bin.label.clone()))
        })
        .collect()
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

/// Encode a band-fraction offset column on a position offset channel.
///
/// The offsets ggsql resolves — dodge and jitter displacements, a violin's
/// density half-width, a half-boxplot's side shift — are fractions of the band,
/// which is what a `[-0.5, 0.5]` domain makes of them: the scale's range is the
/// band width, so a value of `w` shifts the mark by `w` bands.
///
/// For the **secondary** channel the domain runs `0.5 → -0.5` instead, because a
/// ggsql offset is positive-up (matching the bottom-up categorical `y` the band
/// domain is reversed for) while a Vega-Lite `yOffset` is positive-down.
/// Flipping the domain negates the offset without touching the data, so every
/// mark — and every component of a composite one — reads the same way round as
/// the axis it sits on. The primary channel needs no flip: `xOffset` is
/// positive-right, as ggsql's offsets are.
pub(super) fn offset_encoding(field: &str, is_secondary: bool) -> Value {
    let domain = if is_secondary {
        json!([0.5, -0.5])
    } else {
        json!([-0.5, 0.5])
    };
    json!({
        "field": field,
        "type": "quantitative",
        "scale": { "domain": domain }
    })
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
            let mut domain_json: Vec<Value> =
                domain_values.iter().map(|elem| elem.to_json()).collect();
            // A categorical `y` runs bottom-up, as in ggplot2: the first level
            // sits at the bottom of the panel. Vega-Lite lays a band domain out
            // top-to-bottom, so the domain is handed over backwards to put it
            // the right way up. `scale.reverse` still composes on top, flipping
            // whatever the default now is.
            if ctx.aesthetic == "pos2" && is_categorical(scale) {
                domain_json.reverse();
            }
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

    // Handle reverse property (SETTING clause).
    //
    // A free facet dimension emits no domain — Vega-Lite computes one per panel
    // — so a categorical `y` there cannot be handed over backwards the way a
    // fixed one is above. `reverse` expresses the same bottom-up default
    // instead. A user `SETTING reverse => true` composes on top of whichever
    // default applies, which for the free case means the two cancel.
    let bottom_up_default = skip_domain && ctx.aesthetic == "pos2" && is_categorical(scale);
    let user_reverse = matches!(
        scale.properties.get("reverse"),
        Some(ParameterValue::Boolean(true))
    );
    if user_reverse != bottom_up_default {
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
        // Geographic transform: degree-aligned breaks, no VL equivalent
        TransformKind::Geographic => {}
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
    field_type: &str,
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

    let is_symbol =
        is_binned_legend && determine_legend_style(aesthetic, spec) == LegendStyle::Symbol;

    let breaks = match scale.properties.get("breaks") {
        Some(ParameterValue::Array(b)) => Some(b.as_slice()),
        _ => None,
    };

    // Symbol legends compare VL's predicted range labels (e.g. "-20 – 0")
    // as strings via datum.label, not as numeric datum.value.
    let filtered_mapping = if is_symbol {
        build_symbol_legend_label_mapping(scale)
    } else {
        label_mapping.clone()
    };

    // Gradient legends use null for the first terminal's label
    let null_key = if is_binned_legend && !is_symbol {
        breaks.and_then(|b| b.first().map(|e| e.to_key_string()))
    } else {
        None
    };

    let effective_field_type = if is_symbol { "nominal" } else { field_type };

    let label_expr = build_label_expr(
        &filtered_mapping,
        time_format,
        null_key.as_deref(),
        effective_field_type,
    );

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
    /// `calculate` transforms the built encodings need on their layer, in the
    /// order they were requested. Currently only the unit conversion an
    /// identity-scaled column needs — see [`identity_unit_conversion`].
    pub transforms: &'a mut Vec<Value>,
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

    // An identity scale hands the column to the aesthetic untouched, so each value
    // means what the same value written as a literal means. Convert it exactly as
    // `build_literal_encoding` converts that literal, per row.
    let field = match identity_conversion(aesthetic, col, ctx.spec.find_scale(primary)) {
        Some(expr) if identity_scale => {
            let converted = format!("{col}_visual");
            ctx.transforms
                .push(json!({"calculate": expr, "as": converted.clone()}));
            converted
        }
        _ => col.to_string(),
    };

    // Build base encoding
    let mut encoding = json!({
        "field": field,
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
            &field_type,
        );

        (scale_obj, needs_gradient)
    } else {
        (serde_json::Map::new(), false)
    };

    // Position scales don't include zero by default — but only when we set
    // an explicit domain. With free facet scales (no domain), VL computes
    // the domain from data values.
    if aesthetic_ctx.is_primary_internal(aesthetic) {
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

/// The conversion an identity-scaled column needs, as a Vega expression over
/// `datum`, or `None` for an aesthetic Vega-Lite already takes in ggsql's own terms.
///
/// `SCALE IDENTITY <aes>` passes the data through unscaled, which means each value is
/// read the way the same value written as a `SETTING` literal is — so it needs the
/// conversion [`build_literal_encoding`] gives that literal and [`convert_range_element`]
/// gives a resolved output range. All three paths must agree. Two shapes of conversion:
///
/// - **Units.** `size` is a radius in points and Vega-Lite wants a symbol area in px²;
///   `linewidth` / `fontsize` are points and it wants px. Plain arithmetic per row.
/// - **Names.** `shape` and `linetype` are ggsql names (`'star'`, `'dashed'`) and
///   Vega-Lite wants an SVG path and a dash array. Vega has no lookup function over an
///   inline table, so the mapping is a conditional chain over the values the scale
///   resolved, with anything unrecognised passed through — a column may already hold
///   paths or dash arrays, exactly as a literal may.
///
/// Either way the arithmetic is per-datum, which in Vega-Lite only exists as a
/// transform, hence a `calculate` feeding a derived field rather than a scale.
fn identity_conversion(aesthetic: &str, col: &str, scale: Option<&crate::Scale>) -> Option<String> {
    let datum = format!("datum['{}']", super::escape_vega_string(col));
    match aesthetic {
        // Size: radius (points) → area (pixels²)
        "size" => Some(format!("{datum} * {datum} * {POINTS_TO_AREA}")),
        // Linewidth: points → pixels
        "linewidth" | "fontsize" => Some(format!("{datum} * {POINTS_TO_PIXELS}")),
        // Shape name → SVG path
        "shape" => identity_lookup_expr(&datum, scale, |name| {
            shape_to_svg_path(name).map(|path| json!(path))
        }),
        // Linetype name → dash array
        "linetype" => identity_lookup_expr(&datum, scale, |name| {
            linetype_to_stroke_dash(name).map(|dashes| json!(dashes))
        }),
        _ => None,
    }
}

/// A Vega conditional chain mapping each value an identity scale resolved to its
/// Vega-Lite equivalent, falling through to the datum itself.
///
/// `None` when nothing needs converting — no resolved values, or none of them is a
/// name `convert` recognises — so no transform is emitted and the column reaches the
/// mark untouched, which is what a column of ready-made paths or dash arrays wants.
fn identity_lookup_expr(
    datum: &str,
    scale: Option<&crate::Scale>,
    convert: impl Fn(&str) -> Option<Value>,
) -> Option<String> {
    let values = scale?.input_range.as_ref()?;
    let mut parts: Vec<String> = Vec::new();

    for value in values {
        let crate::plot::ArrayElement::String(name) = value else {
            continue;
        };
        if let Some(converted) = convert(name) {
            parts.push(format!(
                "{datum} == '{}' ? {}",
                super::escape_vega_string(name),
                converted
            ));
        }
    }

    if parts.is_empty() {
        return None;
    }
    parts.push(datum.to_string());
    Some(parts.join(" : "))
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
    if let Some(vl_channel) = renderer.map_position(aesthetic) {
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
        let pos1 = renderer.map_position("pos1").unwrap();
        let pos1_end = renderer.map_position("pos1end").unwrap();
        let pos2 = renderer.map_position("pos2").unwrap();
        let pos2_end = renderer.map_position("pos2end").unwrap();

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
    use crate::plot::Parameters;

    #[test]
    fn test_build_label_expr_temporal_uses_utc_format() {
        // Temporal label comparisons must use utcFormat (not timeFormat) because
        // ggsql writes ISO date strings as UTC. timeFormat uses the browser's
        // local timezone, causing comparisons to fail in non-UTC timezones
        // (e.g., "2024-01-01" midnight UTC becomes "2023-12-31" in US Central).
        let mut mappings = HashMap::new();
        mappings.insert("2024-01-01".to_string(), Some("Jan 2024".to_string()));

        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None, "temporal");

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

        let expr = build_label_expr(&mappings, None, None, "nominal");

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
        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None, "temporal");
        assert_eq!(
            expr, "datum.label",
            "empty mappings should fall back to datum.label"
        );
    }

    #[test]
    fn test_build_label_expr_null_suppression() {
        let mut mappings = HashMap::new();
        mappings.insert("2024-06-01".to_string(), None); // suppress label

        let expr = build_label_expr(&mappings, Some("%Y-%m-%d"), None, "temporal");

        assert!(
            expr.contains("? ''"),
            "None mapping should suppress label (empty string), got: {expr}"
        );
    }

    #[test]
    fn test_build_label_expr_quantitative_uses_datum_value() {
        let mut mappings = HashMap::new();
        mappings.insert("2020".to_string(), Some("2020.0".to_string()));

        let expr = build_label_expr(&mappings, None, None, "quantitative");

        assert!(
            expr.contains("datum.value == 2020 ? '2020.0'"),
            "quantitative should use datum.value with unquoted comparison, got: {expr}"
        );
        assert!(
            !expr.contains("datum.label =="),
            "quantitative should not use datum.label for comparison, got: {expr}"
        );
    }

    #[test]
    fn test_symbol_legend_label_expr_uses_datum_label() {
        use crate::plot::ArrayElement;

        // Breaks: -20, 0, 20 → VL predicts labels "-20 – 0" and "≥ 0"
        let breaks = vec![
            ArrayElement::Number(-20.0),
            ArrayElement::Number(0.0),
            ArrayElement::Number(20.0),
        ];
        let mut label_mapping = HashMap::new();
        label_mapping.insert("-20".to_string(), Some("cold".to_string()));
        label_mapping.insert("0".to_string(), Some("hot".to_string()));

        let mut scale = Scale::new("fill");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale
            .properties
            .insert("breaks".to_string(), ParameterValue::Array(breaks));
        scale.label_mapping = Some(label_mapping);
        let symbol_mapping = build_symbol_legend_label_mapping(&scale);

        // The resulting mapping uses VL's range-style label strings as keys
        let expr = build_label_expr(&symbol_mapping, None, None, "nominal");

        assert!(
            expr.contains("datum.label =="),
            "symbol legend labelExpr must use datum.label (string comparison), got: {expr}"
        );
        assert!(
            !expr.contains("datum.value =="),
            "symbol legend labelExpr must not use datum.value (keys contain en-dashes), got: {expr}"
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
        use super::*;
        use crate::plot::aesthetic::AestheticContext;
        use crate::plot::{ArrayElement, Scale};
        use crate::writer::vegalite::projection::get_projection_renderer;

        fn discrete_scale(aesthetic: &str) -> Scale {
            Scale {
                aesthetic: aesthetic.to_string(),
                scale_type: None,
                input_range: Some(vec![ArrayElement::String("A".to_string())]),
                explicit_input_range: false,
                output_range: None,
                transform: None,
                explicit_transform: false,
                properties: Parameters::new(),
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
