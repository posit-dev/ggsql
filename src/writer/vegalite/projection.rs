//! Projection transformations for Vega-Lite writer
//!
//! This module handles projection transformations (cartesian, flip, polar)
//! that modify the Vega-Lite spec structure based on the PROJECT clause.

use crate::plot::{CoordKind, ParameterValue, Projection};
use crate::{DataFrame, GgsqlError, Plot, Result};
use serde_json::{json, Value};

/// Apply projection transformations to the spec and data
/// Returns (possibly transformed DataFrame, possibly modified spec)
/// free_x/free_y indicate whether facets have independent scales (affects domain application)
pub(super) fn apply_project_transforms(
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
    free_x: bool,
    free_y: bool,
) -> Result<Option<DataFrame>> {
    if let Some(ref project) = spec.project {
        match project.coord.coord_kind() {
            CoordKind::Cartesian => {
                apply_cartesian_project(project, vl_spec, free_x, free_y)?;
                Ok(None) // No DataFrame transformation needed
            }
            CoordKind::Flip => {
                apply_flip_project(vl_spec)?;
                Ok(None) // No DataFrame transformation needed
            }
            CoordKind::Polar => {
                // Polar requires DataFrame transformation for percentages
                let transformed_df = apply_polar_project(project, spec, data, vl_spec)?;
                Ok(Some(transformed_df))
            }
        }
    } else {
        Ok(None)
    }
}

/// Apply Cartesian projection properties
/// Currently only ratio is supported (not yet implemented)
fn apply_cartesian_project(
    _project: &Projection,
    _vl_spec: &mut Value,
    _free_x: bool,
    _free_y: bool,
) -> Result<()> {
    // ratio, clip - not yet implemented
    Ok(())
}

/// Apply Flip projection transformation (swap x and y)
fn apply_flip_project(vl_spec: &mut Value) -> Result<()> {
    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
            for layer in layers_arr {
                if let Some(encoding) = layer.get_mut("encoding") {
                    if let Some(enc_obj) = encoding.as_object_mut() {
                        if let (Some(x), Some(y)) = (enc_obj.remove("x"), enc_obj.remove("y")) {
                            enc_obj.insert("x".to_string(), y);
                            enc_obj.insert("y".to_string(), x);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Apply Polar projection transformation (bar->arc, point->arc with radius)
fn apply_polar_project(
    project: &Projection,
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<DataFrame> {
    // Get theta field (defaults to 'y')
    let theta_field = project
        .properties
        .get("theta")
        .and_then(|v| match v {
            ParameterValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "y".to_string());

    // Convert geoms to polar equivalents
    convert_geoms_to_polar(spec, vl_spec, &theta_field)?;

    // No DataFrame transformation needed - Vega-Lite handles polar math
    Ok(data.clone())
}

/// Convert geoms to polar equivalents (bar->arc, point->arc with radius)
fn convert_geoms_to_polar(spec: &Plot, vl_spec: &mut Value, theta_field: &str) -> Result<()> {
    // Determine which aesthetic (x or y) maps to theta
    // Default: y maps to theta (pie chart style)
    let theta_aesthetic = theta_field;

    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
            for layer in layers_arr {
                if let Some(mark) = layer.get_mut("mark") {
                    *mark = convert_mark_to_polar(mark, spec)?;

                    if let Some(encoding) = layer.get_mut("encoding") {
                        update_encoding_for_polar(encoding, theta_aesthetic)?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Convert a mark type to its polar equivalent
/// Preserves `clip: true` to ensure marks don't render outside plot bounds
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

    Ok(json!({
        "type": polar_mark,
        "clip": true
    }))
}

/// Update encoding channels for polar projection
fn update_encoding_for_polar(encoding: &mut Value, theta_aesthetic: &str) -> Result<()> {
    let enc_obj = encoding
        .as_object_mut()
        .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

    // Map the theta aesthetic to theta channel
    if theta_aesthetic == "y" {
        // Standard pie chart: y -> theta, x -> color/category
        if let Some(y_enc) = enc_obj.remove("y") {
            enc_obj.insert("theta".to_string(), y_enc);
        }
        // Map x to color if not already mapped, and remove x from positional encoding
        if !enc_obj.contains_key("color") {
            if let Some(x_enc) = enc_obj.remove("x") {
                enc_obj.insert("color".to_string(), x_enc);
            }
        } else {
            // If color is already mapped, just remove x from positional encoding
            enc_obj.remove("x");
        }
    } else if theta_aesthetic == "x" {
        // Reversed: x -> theta, y -> radius
        if let Some(x_enc) = enc_obj.remove("x") {
            enc_obj.insert("theta".to_string(), x_enc);
        }
        if let Some(y_enc) = enc_obj.remove("y") {
            enc_obj.insert("radius".to_string(), y_enc);
        }
    }

    Ok(())
}

