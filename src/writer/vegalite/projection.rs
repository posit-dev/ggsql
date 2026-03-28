//! Projection transformations for Vega-Lite writer
//!
//! This module handles projection transformations (cartesian, polar)
//! that modify the Vega-Lite spec structure based on the PROJECT clause.

use crate::plot::{CoordKind, ParameterValue, Projection};
use crate::{DataFrame, GgsqlError, Plot, Result};
use serde_json::{json, Value};

/// Apply projection transformations to the spec and data
/// Returns (possibly transformed DataFrame, possibly modified spec)
pub(super) fn apply_project_transforms(
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<Option<DataFrame>> {
    if let Some(ref project) = spec.project {
        // Apply coord-specific transformations
        let result = match project.coord.coord_kind() {
            CoordKind::Cartesian => {
                apply_cartesian_project(project, vl_spec)?;
                None
            }
            CoordKind::Polar => Some(apply_polar_project(project, spec, data, vl_spec)?),
        };

        // Apply clip setting (applies to all projection types)
        if let Some(ParameterValue::Boolean(clip)) = project.properties.get("clip") {
            apply_clip_to_layers(vl_spec, *clip);
        }

        Ok(result)
    } else {
        Ok(None)
    }
}

/// Get mutable reference to the layers array, handling both flat and faceted specs.
///
/// In a flat spec: `vl_spec["layer"]`
/// In a faceted spec: `vl_spec["spec"]["layer"]`
fn get_layers_mut(vl_spec: &mut Value) -> Option<&mut Vec<Value>> {
    // Try flat structure first, then faceted
    if vl_spec.get("layer").is_some() {
        vl_spec.get_mut("layer").and_then(|l| l.as_array_mut())
    } else {
        vl_spec
            .get_mut("spec")
            .and_then(|s| s.get_mut("layer"))
            .and_then(|l| l.as_array_mut())
    }
}

/// Apply clip setting to all layers
fn apply_clip_to_layers(vl_spec: &mut Value, clip: bool) {
    if let Some(layers_arr) = get_layers_mut(vl_spec) {
        for layer in layers_arr {
            if let Some(mark) = layer.get_mut("mark") {
                if mark.is_string() {
                    // Convert "point" to {"type": "point", "clip": ...}
                    let mark_type = mark.as_str().unwrap().to_string();
                    *mark = json!({"type": mark_type, "clip": clip});
                } else if let Some(obj) = mark.as_object_mut() {
                    obj.insert("clip".to_string(), json!(clip));
                }
            }
        }
    }
}

/// Apply Cartesian projection properties
fn apply_cartesian_project(_project: &Projection, _vl_spec: &mut Value) -> Result<()> {
    // ratio - not yet implemented
    Ok(())
}

/// Apply Polar projection transformation (bar->arc, point->arc with radius)
///
/// Encoding channel names (theta/radius) are already set correctly by `map_aesthetic_name()`
/// based on coord kind. This function only:
/// 1. Converts mark types to polar equivalents (bar → arc)
/// 2. Applies start/end angle range from PROJECT clause
/// 3. Applies inner radius for donut charts
fn apply_polar_project(
    project: &Projection,
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<DataFrame> {
    // Get start angle in degrees (defaults to 0 = 12 o'clock)
    let start_degrees = project
        .properties
        .get("start")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0.0);

    // Get end angle in degrees (defaults to start + 360 = full circle)
    let end_degrees = project
        .properties
        .get("end")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(start_degrees + 360.0);

    // Get inner radius proportion (0.0 to 1.0, defaults to 0 = full pie)
    let inner = project
        .properties
        .get("inner")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0.0);

    // Convert degrees to radians for Vega-Lite
    let start_radians = start_degrees * std::f64::consts::PI / 180.0;
    let end_radians = end_degrees * std::f64::consts::PI / 180.0;

    // Convert geoms to polar equivalents and apply angle range + inner radius
    convert_geoms_to_polar(spec, vl_spec, start_radians, end_radians, inner)?;

    // No DataFrame transformation needed - Vega-Lite handles polar math
    Ok(data.clone())
}

/// Convert geoms to polar equivalents (bar->arc) and apply angle range + inner radius
///
/// Note: Encoding channel names (theta/radius) are already set correctly by
/// `map_aesthetic_name()` based on coord kind. This function handles two cases:
///
/// 1. **Arc-compatible marks** (bar, col, area → arc): Keep radius/theta channels,
///    apply angle range and inner radius directly.
///
/// 2. **Non-arc marks** (point, line): Vega-Lite only supports radius/theta channels
///    for arc and text marks. For other marks, we convert polar→cartesian using
///    calculate transforms and x/y encoding channels.
fn convert_geoms_to_polar(
    spec: &Plot,
    vl_spec: &mut Value,
    start_radians: f64,
    end_radians: f64,
    inner: f64,
) -> Result<()> {
    if let Some(layers_arr) = get_layers_mut(vl_spec) {
        for layer in layers_arr {
            if let Some(mark) = layer.get_mut("mark") {
                let polar_mark = convert_mark_to_polar(mark, spec)?;
                let is_arc = polar_mark.as_str() == Some("arc");
                *mark = polar_mark;

                if is_arc {
                    // Arc marks natively support radius/theta channels
                    if let Some(encoding) = layer.get_mut("encoding") {
                        // Remove dummy radius encoding — for pie charts the bar stat
                        // creates a dummy pos1 column (all rows same value). VL's arc
                        // mark uses full available radius by default, and a nominal
                        // radius with a single dummy value breaks faceted arc rendering.
                        strip_dummy_radius(encoding);
                        apply_polar_angle_range(encoding, start_radians, end_radians)?;
                        apply_polar_radius_range(encoding, inner)?;
                    }
                } else {
                    // Non-arc marks (point, line): convert polar to cartesian
                    convert_polar_to_cartesian(layer, start_radians, end_radians, inner)?;
                }
            }
        }
    }

    Ok(())
}

/// Convert a layer's radius/theta encoding to x/y using calculate transforms.
///
/// Vega-Lite's radius and theta channels only work with arc and text marks.
/// For point, line, and other marks, we need to:
/// 1. Extract field names and scale domains from the radius/theta encoding
/// 2. Add calculate transforms to normalize and convert polar→cartesian
/// 3. Replace radius/theta with x/y encoding channels
fn convert_polar_to_cartesian(
    layer: &mut Value,
    start_radians: f64,
    end_radians: f64,
    inner: f64,
) -> Result<()> {
    // Phase 1: Extract info from encoding (immutable read)
    let (r_field, r_domain, r_title, theta_field, theta_domain, theta_title) = {
        let encoding = layer
            .get("encoding")
            .and_then(|e| e.as_object())
            .ok_or_else(|| GgsqlError::WriterError("Layer has no encoding object".to_string()))?;

        let (r_field, r_domain, r_title) = extract_polar_channel(encoding, "radius")?;
        let (theta_field, theta_domain, theta_title) =
            extract_polar_channel(encoding, "theta")?;
        (r_field, r_domain, r_title, theta_field, theta_domain, theta_title)
    };

    // Phase 2: Build calculate transforms for polar→cartesian conversion
    //
    // theta convention: 0 = 12 o'clock (top), increases clockwise
    // cartesian: x = r * sin(θ), y = r * cos(θ)
    let angle_span = end_radians - start_radians;
    let (theta_min, theta_max) = theta_domain;
    let (r_min, r_max) = r_domain;

    let mut polar_transforms: Vec<Value> = Vec::new();

    // Normalize theta to [start_radians, end_radians]
    if (theta_max - theta_min).abs() > f64::EPSILON {
        polar_transforms.push(json!({
            "calculate": format!(
                "{start} + (datum['{field}'] - {min}) / ({max} - {min}) * {span}",
                start = start_radians,
                field = theta_field,
                min = theta_min,
                max = theta_max,
                span = angle_span,
            ),
            "as": "__polar_theta__"
        }));
    } else {
        polar_transforms.push(json!({
            "calculate": format!("{}", start_radians),
            "as": "__polar_theta__"
        }));
    }

    // Normalize radius to [inner, 1.0]
    if (r_max - r_min).abs() > f64::EPSILON {
        polar_transforms.push(json!({
            "calculate": format!(
                "{inner} + (1 - {inner}) * (datum['{field}'] - {min}) / ({max} - {min})",
                inner = inner,
                field = r_field,
                min = r_min,
                max = r_max,
            ),
            "as": "__polar_r__"
        }));
    } else {
        polar_transforms.push(json!({
            "calculate": format!("{}", (1.0 + inner) / 2.0),
            "as": "__polar_r__"
        }));
    }

    // Convert to cartesian: x = r * sin(θ), y = r * cos(θ)
    polar_transforms.push(json!({
        "calculate": "datum.__polar_r__ * sin(datum.__polar_theta__)",
        "as": "__polar_x__"
    }));
    polar_transforms.push(json!({
        "calculate": "datum.__polar_r__ * cos(datum.__polar_theta__)",
        "as": "__polar_y__"
    }));

    // Phase 3: Mutate the layer — append transforms
    if let Some(existing) = layer.get_mut("transform") {
        if let Some(arr) = existing.as_array_mut() {
            arr.extend(polar_transforms);
        }
    } else {
        layer["transform"] = json!(polar_transforms);
    }

    // Phase 4: Rewrite encoding — remove polar channels, add cartesian
    let encoding = layer
        .get_mut("encoding")
        .and_then(|e| e.as_object_mut())
        .ok_or_else(|| GgsqlError::WriterError("Layer has no encoding object".to_string()))?;

    encoding.remove("radius");
    encoding.remove("theta");

    let padding = 1.08;
    let mut x_enc = json!({
        "field": "__polar_x__",
        "type": "quantitative",
        "scale": {"domain": [-padding, padding]},
        "axis": null
    });
    let mut y_enc = json!({
        "field": "__polar_y__",
        "type": "quantitative",
        "scale": {"domain": [-padding, padding]},
        "axis": null
    });

    if let Some(title) = theta_title {
        x_enc["title"] = title;
    }
    if let Some(title) = r_title {
        y_enc["title"] = title;
    }

    encoding.insert("x".to_string(), x_enc);
    encoding.insert("y".to_string(), y_enc);

    Ok(())
}

/// Extract field name, scale domain, and title from a polar encoding channel.
/// Returns (field_name, (domain_min, domain_max), optional_title).
fn extract_polar_channel(
    encoding: &serde_json::Map<String, Value>,
    channel: &str,
) -> Result<(String, (f64, f64), Option<Value>)> {
    let channel_enc = encoding.get(channel).ok_or_else(|| {
        GgsqlError::WriterError(format!(
            "Polar projection requires '{}' encoding channel",
            channel
        ))
    })?;

    let field = channel_enc
        .get("field")
        .and_then(|f| f.as_str())
        .ok_or_else(|| {
            GgsqlError::WriterError(format!("'{}' encoding missing 'field'", channel))
        })?
        .to_string();

    // Extract domain from scale, with fallback to [0, 1]
    let domain = channel_enc
        .get("scale")
        .and_then(|s| s.get("domain"))
        .and_then(|d| d.as_array())
        .and_then(|arr| {
            let min = arr.first()?.as_f64()?;
            let max = arr.get(1)?.as_f64()?;
            Some((min, max))
        })
        .unwrap_or((0.0, 1.0));

    let title = channel_enc.get("title").cloned();

    Ok((field, domain, title))
}

/// Convert a mark type to its polar equivalent
fn convert_mark_to_polar(mark: &Value, _spec: &Plot) -> Result<Value> {
    let mark_str = if mark.is_string() {
        mark.as_str().unwrap()
    } else if let Some(mark_type) = mark.get("type") {
        mark_type.as_str().unwrap_or("bar")
    } else {
        "bar"
    };

    // Convert geom types to polar equivalents
    let polar_mark = match mark_str {
        "bar" | "col" => {
            // Bar/col in polar becomes arc (pie/donut slices)
            "arc"
        }
        "point" => {
            // Points in polar can stay as points or become arcs with radius
            // For now, keep as points (they'll plot at radius based on value)
            "point"
        }
        "line" => {
            // Lines in polar become circular/spiral lines
            "line"
        }
        "area" => {
            // Area in polar becomes arc with radius
            "arc"
        }
        _ => {
            // Other geoms: keep as-is or convert to arc
            "arc"
        }
    };

    Ok(json!(polar_mark))
}

/// Remove dummy radius encoding from arc marks.
///
/// For pie charts, the bar stat creates a dummy pos1 column where all rows have
/// the same placeholder value. In cartesian this positions all bars at the same x.
/// For arc marks, a nominal radius with a single dummy value is unnecessary (VL
/// uses full available radius by default) and breaks faceted arc rendering.
/// The dummy is identified by `"axis": null` which is set by build_column_encoding
/// for dummy columns.
fn strip_dummy_radius(encoding: &mut Value) {
    let dominated = encoding
        .get("radius")
        .and_then(|r| r.get("axis"))
        .is_some_and(|a| a.is_null());
    if dominated {
        if let Some(enc_obj) = encoding.as_object_mut() {
            enc_obj.remove("radius");
        }
    }
}

/// Apply angle range to theta encoding for polar projection
///
/// The encoding channels are already correctly named (theta/radius) by
/// `map_aesthetic_name()` based on coord kind. This function only applies
/// the optional start/end angle range from the PROJECT clause.
fn apply_polar_angle_range(
    encoding: &mut Value,
    start_radians: f64,
    end_radians: f64,
) -> Result<()> {
    // Skip if default range (0 to 2π)
    let is_default = start_radians.abs() <= f64::EPSILON
        && (end_radians - 2.0 * std::f64::consts::PI).abs() <= f64::EPSILON;
    if is_default {
        return Ok(());
    }

    let enc_obj = encoding
        .as_object_mut()
        .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

    // Apply angle range to theta encoding
    if let Some(theta_enc) = enc_obj.get_mut("theta") {
        if let Some(theta_obj) = theta_enc.as_object_mut() {
            // Merge range into existing scale object (preserving domain from expansion)
            if let Some(scale_val) = theta_obj.get_mut("scale") {
                if let Some(scale_obj) = scale_val.as_object_mut() {
                    scale_obj.insert("range".to_string(), json!([start_radians, end_radians]));
                }
            } else {
                // No existing scale, create new one with just range
                theta_obj.insert(
                    "scale".to_string(),
                    json!({
                        "range": [start_radians, end_radians]
                    }),
                );
            }
        }
    }

    Ok(())
}

/// Apply inner radius to radius encoding for donut charts
///
/// Sets the radius scale range using Vega-Lite expressions for proportional sizing.
/// The inner parameter (0.0 to 1.0) specifies the inner radius as a proportion
/// of the outer radius, creating a donut hole.
fn apply_polar_radius_range(encoding: &mut Value, inner: f64) -> Result<()> {
    // Skip if no inner radius (full pie)
    if inner <= f64::EPSILON {
        return Ok(());
    }

    let enc_obj = encoding
        .as_object_mut()
        .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

    // Use expressions for proportional sizing
    // min(width,height)/2 is the default max radius in Vega-Lite
    let inner_expr = format!("min(width,height)/2*{}", inner);
    let outer_expr = "min(width,height)/2".to_string();
    let range_value = json!([{"expr": inner_expr}, {"expr": outer_expr}]);

    // Apply scale range to radius encoding (merge with existing scale)
    if let Some(radius_enc) = enc_obj.get_mut("radius") {
        if let Some(radius_obj) = radius_enc.as_object_mut() {
            if let Some(scale_val) = radius_obj.get_mut("scale") {
                if let Some(scale_obj) = scale_val.as_object_mut() {
                    scale_obj.insert("range".to_string(), range_value.clone());
                }
            } else {
                radius_obj.insert("scale".to_string(), json!({ "range": range_value.clone() }));
            }
        }
    }

    // Also apply to radius2 if present (for arc marks)
    if let Some(radius2_enc) = enc_obj.get_mut("radius2") {
        if let Some(radius2_obj) = radius2_enc.as_object_mut() {
            if let Some(scale_val) = radius2_obj.get_mut("scale") {
                if let Some(scale_obj) = scale_val.as_object_mut() {
                    scale_obj.insert("range".to_string(), range_value.clone());
                }
            } else {
                radius2_obj.insert("scale".to_string(), json!({ "range": range_value }));
            }
        }
    }

    Ok(())
}
