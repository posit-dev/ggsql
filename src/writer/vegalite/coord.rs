//! Coordinate system transformations for Vega-Lite writer
//!
//! This module handles coordinate system transformations (cartesian, flip, polar)
//! that modify the Vega-Lite spec structure based on the COORD clause.

use crate::plot::aesthetic::is_aesthetic_name;
use crate::plot::{Coord, CoordType, ParameterValue};
use crate::{DataFrame, GgsqlError, Plot, Result};
use serde_json::{json, Value};

/// Apply coordinate transformations to the spec and data
/// Returns (possibly transformed DataFrame, possibly modified spec)
pub(super) fn apply_coord_transforms(
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<Option<DataFrame>> {
    if let Some(ref coord) = spec.coord {
        match coord.coord_type {
            CoordType::Cartesian => {
                apply_cartesian_coord(coord, vl_spec)?;
                Ok(None) // No DataFrame transformation needed
            }
            CoordType::Flip => {
                apply_flip_coord(vl_spec)?;
                Ok(None) // No DataFrame transformation needed
            }
            CoordType::Polar => {
                // Polar requires DataFrame transformation for percentages
                let transformed_df = apply_polar_coord(coord, spec, data, vl_spec)?;
                Ok(Some(transformed_df))
            }
            _ => {
                // Other coord types not yet implemented
                Ok(None)
            }
        }
    } else {
        Ok(None)
    }
}

/// Apply Cartesian coordinate properties (xlim, ylim, aesthetic domains)
fn apply_cartesian_coord(coord: &Coord, vl_spec: &mut Value) -> Result<()> {
    // Apply xlim/ylim to scale domains
    for (prop_name, prop_value) in &coord.properties {
        match prop_name.as_str() {
            "xlim" => {
                if let Some(limits) = extract_limits(prop_value)? {
                    apply_axis_limits(vl_spec, "x", limits)?;
                }
            }
            "ylim" => {
                if let Some(limits) = extract_limits(prop_value)? {
                    apply_axis_limits(vl_spec, "y", limits)?;
                }
            }
            _ if is_aesthetic_name(prop_name) => {
                // Aesthetic domain specification
                if let Some(domain) = extract_input_range(prop_value)? {
                    apply_aesthetic_input_range(vl_spec, prop_name, domain)?;
                }
            }
            _ => {
                // ratio, clip - not yet implemented (TODO comments added by validation)
            }
        }
    }

    Ok(())
}

/// Apply Flip coordinate transformation (swap x and y)
fn apply_flip_coord(vl_spec: &mut Value) -> Result<()> {
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

/// Apply Polar coordinate transformation (bar->arc, point->arc with radius)
fn apply_polar_coord(
    coord: &Coord,
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<DataFrame> {
    // Get theta field (defaults to 'y')
    let theta_field = coord
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

/// Update encoding channels for polar coordinates
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

// Helper methods

fn extract_limits(value: &ParameterValue) -> Result<Option<(f64, f64)>> {
    match value {
        ParameterValue::Array(arr) => {
            if arr.len() != 2 {
                return Err(GgsqlError::WriterError(format!(
                    "xlim/ylim must be exactly 2 numbers, got {}",
                    arr.len()
                )));
            }
            let min = arr[0].to_f64().ok_or_else(|| {
                GgsqlError::WriterError("xlim/ylim values must be numeric".to_string())
            })?;
            let max = arr[1].to_f64().ok_or_else(|| {
                GgsqlError::WriterError("xlim/ylim values must be numeric".to_string())
            })?;

            // Auto-swap if reversed
            let (min, max) = if min > max { (max, min) } else { (min, max) };

            Ok(Some((min, max)))
        }
        _ => Err(GgsqlError::WriterError(
            "xlim/ylim must be an array".to_string(),
        )),
    }
}

fn extract_input_range(value: &ParameterValue) -> Result<Option<Vec<Value>>> {
    match value {
        ParameterValue::Array(arr) => {
            let domain: Vec<Value> = arr.iter().map(|elem| elem.to_json()).collect();
            Ok(Some(domain))
        }
        _ => Ok(None),
    }
}

fn apply_axis_limits(vl_spec: &mut Value, axis: &str, limits: (f64, f64)) -> Result<()> {
    let domain = json!([limits.0, limits.1]);

    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
            for layer in layers_arr {
                if let Some(encoding) = layer.get_mut("encoding") {
                    if let Some(axis_enc) = encoding.get_mut(axis) {
                        axis_enc["scale"] = json!({"domain": domain});
                    }
                }
            }
        }
    }

    Ok(())
}

fn apply_aesthetic_input_range(
    vl_spec: &mut Value,
    aesthetic: &str,
    domain: Vec<Value>,
) -> Result<()> {
    let domain_json = json!(domain);

    if let Some(layers) = vl_spec.get_mut("layer") {
        if let Some(layers_arr) = layers.as_array_mut() {
            for layer in layers_arr {
                if let Some(encoding) = layer.get_mut("encoding") {
                    if let Some(aes_enc) = encoding.get_mut(aesthetic) {
                        aes_enc["scale"] = json!({"domain": domain_json});
                    }
                }
            }
        }
    }

    Ok(())
}
