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

/// Apply clip setting to all layers
fn apply_clip_to_layers(vl_spec: &mut Value, clip: bool) {
    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
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
/// `map_aesthetic_name()` based on coord kind. This function only:
/// 1. Converts mark types to polar equivalents (bar → arc)
/// 2. Applies start/end angle range from PROJECT clause
/// 3. Applies inner radius for donut charts
fn convert_geoms_to_polar(
    spec: &Plot,
    vl_spec: &mut Value,
    start_radians: f64,
    end_radians: f64,
    inner: f64,
) -> Result<()> {
    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
            for layer in layers_arr {
                if let Some(mark) = layer.get_mut("mark") {
                    *mark = convert_mark_to_polar(mark, spec)?;

                    // Apply angle range if non-default
                    if let Some(encoding) = layer.get_mut("encoding") {
                        apply_polar_angle_range(encoding, start_radians, end_radians)?;
                        apply_polar_radius_range(encoding, inner)?;
                    }
                }
            }
        }
    }

    Ok(())
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
            // Set the scale range to the specified start/end angles
            theta_obj.insert(
                "scale".to_string(),
                json!({
                    "range": [start_radians, end_radians]
                }),
            );
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

    // Apply scale range to radius encoding
    if let Some(radius_enc) = enc_obj.get_mut("radius") {
        if let Some(radius_obj) = radius_enc.as_object_mut() {
            // Use expressions for proportional sizing
            // min(width,height)/2 is the default max radius in Vega-Lite
            let inner_expr = format!("min(width,height)/2*{}", inner);
            let outer_expr = "min(width,height)/2".to_string();

            radius_obj.insert(
                "scale".to_string(),
                json!({
                    "range": [{"expr": inner_expr}, {"expr": outer_expr}]
                }),
            );
        }
    }

    // Also apply to radius2 if present (for arc marks)
    if let Some(radius2_enc) = enc_obj.get_mut("radius2") {
        if let Some(radius2_obj) = radius2_enc.as_object_mut() {
            let inner_expr = format!("min(width,height)/2*{}", inner);
            let outer_expr = "min(width,height)/2".to_string();

            radius2_obj.insert(
                "scale".to_string(),
                json!({
                    "range": [{"expr": inner_expr}, {"expr": outer_expr}]
                }),
            );
        }
    }

    Ok(())
}
