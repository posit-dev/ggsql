//! Projection rendering for Vega-Lite writer
//!
//! This module provides a trait-based design for projection rendering.
//! Each projection type (cartesian, polar, and future map projections)
//! implements `ProjectionRenderer`, which owns both the VL channel mapping
//! and the spec-level transformations for that projection.

use crate::plot::{CoordKind, ParameterValue, Projection, Scale};
use crate::{DataFrame, GgsqlError, Plot, Result};
use serde_json::{json, Value};

use super::DEFAULT_POLAR_SIZE;

// =============================================================================
// ProjectionRenderer trait
// =============================================================================

/// Trait defining how a projection type maps to Vega-Lite.
///
/// Each implementation owns two concerns:
/// 1. **Channel mapping** — translating internal position aesthetics (pos1, pos2, …)
///    to Vega-Lite encoding channel names.
/// 2. **Spec transformation** — modifying the Vega-Lite spec after layers are built
///    (e.g., converting marks to arcs for polar).
pub(super) trait ProjectionRenderer {
    /// Primary and secondary VL channel names for this projection.
    ///
    /// Returns `(pos1_channel, pos2_channel)`, e.g. `("x", "y")` for cartesian,
    /// `("radius", "theta")` for polar.
    fn position_channels(&self) -> (&'static str, &'static str);

    /// Offset channel names for this projection.
    ///
    /// Returns `(pos1_offset, pos2_offset)`, e.g. `("xOffset", "yOffset")`.
    fn offset_channels(&self) -> (&'static str, &'static str);

    /// Explicit (width, height) panel dimensions for faceted specs, if needed.
    ///
    /// Polar projections need this because arc mark radius ranges depend on
    /// known dimensions; cartesian uses `"container"` sizing and returns `None`.
    fn panel_size(&self) -> Option<(f64, f64)> {
        None
    }

    /// Apply projection-specific transformations to the VL spec.
    ///
    /// Called after layers are built but before faceting. May return a
    /// transformed DataFrame (e.g., polar currently clones it unchanged).
    fn apply(
        &self,
        project: &Projection,
        spec: &Plot,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>>;

    /// Vega-Lite layers to prepend before the data layers.
    ///
    /// Called after faceting, before the theme config is applied. Receives
    /// the resolved scales so implementations can derive grid lines, axis
    /// ticks, or other decorations from scale breaks and domains.
    fn background_layers(&self, _scales: &[Scale], _theme: &mut Value) -> Vec<Value> {
        Vec::new()
    }

    /// Vega-Lite layers to append after the data layers.
    ///
    /// Same timing and access as [`background_layers`].
    fn foreground_layers(&self, _scales: &[Scale], _theme: &mut Value) -> Vec<Value> {
        Vec::new()
    }

    /// Apply projection-specific transformations and cross-cutting concerns (clip).
    fn apply_transforms(
        &self,
        spec: &Plot,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        let Some(ref project) = spec.project else {
            return Ok(None);
        };

        let result = self.apply(project, spec, data, vl_spec)?;

        if let Some(ParameterValue::Boolean(clip)) = project.properties.get("clip") {
            apply_clip_to_layers(vl_spec, *clip);
        }

        Ok(result)
    }

    /// Prepend background and append foreground decoration layers.
    ///
    /// Called after faceting so that decoration layers appear in both faceted
    /// and non-faceted specs.
    fn apply_panel_decor(&self, spec: &Plot, theme: &mut Value, vl_spec: &mut Value) {
        let mut bg = self.background_layers(&spec.scales, theme);
        let mut fg = self.foreground_layers(&spec.scales, theme);
        if bg.is_empty() && fg.is_empty() {
            return;
        }
        for layer in &mut bg {
            layer["description"] = json!("background");
        }
        for layer in &mut fg {
            layer["description"] = json!("foreground");
        }
        if let Some(layers) = get_layers_mut(vl_spec) {
            let data_layers = std::mem::take(layers);
            layers.reserve(bg.len() + data_layers.len() + fg.len());
            layers.extend(bg);
            layers.extend(data_layers);
            layers.extend(fg);
        }
    }
}

// =============================================================================
// Factory
// =============================================================================

/// Get the projection renderer for a projection spec.
///
/// Returns the appropriate renderer based on the projection's coord kind,
/// or a Cartesian renderer if no projection is specified.
pub(super) fn get_projection_renderer(project: Option<&Projection>) -> Box<dyn ProjectionRenderer> {
    match project.map(|p| p.coord.coord_kind()) {
        Some(CoordKind::Polar) => Box::new(PolarProjection),
        Some(CoordKind::Cartesian) | None => Box::new(CartesianProjection),
    }
}

// =============================================================================
// Channel mapping helpers (used by encoding.rs via the trait)
// =============================================================================

/// Map internal position aesthetic to Vega-Lite channel name using the renderer.
///
/// Returns `Some(channel_name)` for internal position aesthetics (pos1, pos2, etc.),
/// or `None` for material aesthetics.
pub(super) fn map_position_to_vegalite(
    aesthetic: &str,
    renderer: &dyn ProjectionRenderer,
) -> Option<String> {
    let (primary, secondary) = renderer.position_channels();

    // Match internal position aesthetic patterns
    // Convention: min → primary channel (x/y), max → secondary channel (x2/y2)
    match aesthetic {
        // Primary position and min variants
        "pos1" | "pos1min" => Some(primary.to_string()),
        "pos2" | "pos2min" => Some(secondary.to_string()),
        // End and max variants (Vega-Lite uses x2/y2/theta2/radius2)
        "pos1end" | "pos1max" => Some(format!("{}2", primary)),
        "pos2end" | "pos2max" => Some(format!("{}2", secondary)),
        _ => None,
    }
}

// =============================================================================
// CartesianProjection
// =============================================================================

/// Cartesian projection — standard x/y coordinates.
struct CartesianProjection;

impl ProjectionRenderer for CartesianProjection {
    fn position_channels(&self) -> (&'static str, &'static str) {
        ("x", "y")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("xOffset", "yOffset")
    }

    /// Apply Cartesian projection properties
    fn apply(
        &self,
        _project: &Projection,
        _spec: &Plot,
        _data: &DataFrame,
        _vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        // ratio - not yet implemented
        Ok(None)
    }
}

// =============================================================================
// PolarProjection
// =============================================================================

/// Normalized outer radius (proportion of `min(width, height) / 2`).
const POLAR_OUTER: f64 = 1.0;

/// Polar projection — radius/theta coordinates for pie charts, rose plots, etc.
struct PolarProjection;

impl ProjectionRenderer for PolarProjection {
    fn position_channels(&self) -> (&'static str, &'static str) {
        ("radius", "theta")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("radiusOffset", "thetaOffset")
    }

    fn panel_size(&self) -> Option<(f64, f64)> {
        Some((DEFAULT_POLAR_SIZE, DEFAULT_POLAR_SIZE))
    }

    fn apply(
        &self,
        project: &Projection,
        spec: &Plot,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        apply_polar_project(project, spec, data, vl_spec)
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

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

// =============================================================================
// Polar projection implementation
// =============================================================================

/// Extract (start_radians, end_radians, inner) from a Projection.
///
/// Defaults: start=0°, end=start+360°, inner=0.
fn polar_properties(project: Option<&Projection>) -> (f64, f64, f64) {
    let prop = |name| {
        project
            .and_then(|p| p.properties.get(name))
            .and_then(|v| match v {
                ParameterValue::Number(n) => Some(*n),
                _ => None,
            })
    };
    let start_degrees = prop("start").unwrap_or(0.0);
    let end_degrees = prop("end").unwrap_or(start_degrees + 360.0);
    let inner = prop("inner").unwrap_or(0.0);
    (
        start_degrees * std::f64::consts::PI / 180.0,
        end_degrees * std::f64::consts::PI / 180.0,
        inner,
    )
}

// =============================================================================
// Polar expression helpers
// =============================================================================
// Vega-Lite expression strings for polar ↔ pixel coordinate math.
// Used by both data-layer transforms and decoration layers.

/// Normalize a value from `[domain_min, domain_max]` to `[inner, POLAR_OUTER]`.
fn expr_normalize_radius(value: &str, domain_min: f64, domain_max: f64, inner: f64) -> String {
    let scale = (POLAR_OUTER - inner) / (domain_max - domain_min);
    format!("{inner} + {scale} * ({value} - {domain_min})")
}

/// Normalize a value from `[domain_min, domain_max]` to `[start, end]` radians.
fn expr_normalize_theta(
    value: &str,
    domain_min: f64,
    domain_max: f64,
    start: f64,
    end: f64,
) -> String {
    let scale = (end - start) / (domain_max - domain_min);
    format!("{start} + {scale} * ({value} - {domain_min})")
}

/// Pixel x-coordinate from a normalized radius expression and theta expression.
fn expr_polar_x(r: &str, theta: &str) -> String {
    format!("width / 2 + min(width, height) / 2 * {r} * sin({theta})")
}

/// Pixel y-coordinate from a normalized radius expression and theta expression.
fn expr_polar_y(r: &str, theta: &str) -> String {
    format!("height / 2 - min(width, height) / 2 * {r} * cos({theta})")
}

/// Pixel radius from a normalized radius expression.
fn expr_polar_radius(r: &str) -> String {
    format!("min(width, height) / 2 * ({r})")
}

// =============================================================================
// Polar projection transformation
// =============================================================================

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
) -> Result<Option<DataFrame>> {
    let (start_radians, end_radians, inner) = polar_properties(Some(project));

    // Convert geoms to polar equivalents and apply angle range + inner radius
    convert_geoms_to_polar(spec, vl_spec, start_radians, end_radians, inner)?;

    // No DataFrame transformation needed - Vega-Lite handles polar math
    Ok(Some(data.clone()))
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
    let is_faceted = match &spec.facet {
        Some(facet) => !facet.get_variables().is_empty(),
        _ => false,
    };

    let size = if is_faceted {
        // Try to grab size from spec if available
        let height = vl_spec.get("height").and_then(|h| h.as_f64());
        let width = vl_spec.get("width").and_then(|w| w.as_f64());

        Some(match (height, width) {
            (Some(h), Some(w)) => h.min(w),
            (Some(h), None) => h,
            (None, Some(w)) => w,
            _ => DEFAULT_POLAR_SIZE, // Fallback
        })
    } else {
        None
    };

    if let Some(layers_arr) = get_layers_mut(vl_spec) {
        for layer in layers_arr {
            if let Some(mark) = layer.get_mut("mark") {
                let polar_mark = convert_mark_to_polar(mark, spec)?;
                let is_arc = polar_mark.as_str() == Some("arc");
                *mark = polar_mark;

                if is_arc {
                    // Arc marks natively support radius/theta channels
                    if let Some(encoding) = layer.get_mut("encoding") {
                        apply_polar_angle_range(encoding, start_radians, end_radians)?;
                        apply_polar_radius_range(encoding, inner, size)?;
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
        let (theta_field, theta_domain, theta_title) = extract_polar_channel(encoding, "theta")?;
        (
            r_field,
            r_domain,
            r_title,
            theta_field,
            theta_domain,
            theta_title,
        )
    };

    let (theta_min, theta_max) = theta_domain;
    let (r_min, r_max) = r_domain;

    let mut polar_transforms: Vec<Value> = Vec::new();

    // Drop rows with null positions — Vega-Lite does this implicitly for
    // scaled channels, but with scale:null we handle it ourselves.
    polar_transforms.push(json!({
        "filter": format!(
            "isValid(datum['{r_field}']) && isValid(datum['{theta_field}'])"
        )
    }));

    let theta_expr = if (theta_max - theta_min).abs() > f64::EPSILON {
        expr_normalize_theta(
            &format!("datum['{theta_field}']"),
            theta_min,
            theta_max,
            start_radians,
            end_radians,
        )
    } else {
        format!("{start_radians}")
    };
    polar_transforms.push(json!({"calculate": theta_expr, "as": "__polar_theta__"}));

    let r_expr = if (r_max - r_min).abs() > f64::EPSILON {
        expr_normalize_radius(&format!("datum['{r_field}']"), r_min, r_max, inner)
    } else {
        format!("{}", (POLAR_OUTER + inner) / 2.0)
    };
    polar_transforms.push(json!({"calculate": r_expr, "as": "__polar_r__"}));

    polar_transforms.push(json!({
        "calculate": expr_polar_x("datum.__polar_r__", "datum.__polar_theta__"),
        "as": "__polar_x__"
    }));
    polar_transforms.push(json!({
        "calculate": expr_polar_y("datum.__polar_r__", "datum.__polar_theta__"),
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

    let mut x_enc = json!({
        "field": "__polar_x__",
        "type": "quantitative",
        "scale": null,
        "axis": null
    });
    let mut y_enc = json!({
        "field": "__polar_y__",
        "type": "quantitative",
        "scale": null,
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
        .ok_or_else(|| GgsqlError::WriterError(format!("'{}' encoding missing 'field'", channel)))?
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
fn apply_polar_radius_range(encoding: &mut Value, inner: f64, size: Option<f64>) -> Result<()> {
    let enc_obj = encoding
        .as_object_mut()
        .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

    let (inner_expr, outer_expr) = match size {
        Some(dim) => (format!("{}/2*{}", dim, inner), format!("{}/2", dim)),
        None => (
            expr_polar_radius(&format!("{inner}")),
            expr_polar_radius(&format!("{POLAR_OUTER}")),
        ),
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_inner_radius_non_faceted() {
        // Non-faceted donut should use dynamic min(width,height) expressions
        let mut encoding = json!({
            "radius": {
                "field": "dummy",
                "type": "nominal",
                "scale": {"domain": ["dummy"]}
            }
        });

        apply_polar_radius_range(&mut encoding, 0.5, None).unwrap();

        let range = encoding["radius"]["scale"]["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(
            range[0]["expr"].as_str().unwrap(),
            "min(width, height) / 2 * (0.5)"
        );
        assert_eq!(
            range[1]["expr"].as_str().unwrap(),
            "min(width, height) / 2 * (1)"
        );
    }

    #[test]
    fn test_polar_inner_radius_faceted() {
        // Faceted donut should use explicit size calculation
        let mut encoding = json!({
            "radius": {
                "field": "dummy",
                "type": "nominal",
                "scale": {"domain": ["dummy"]}
            }
        });

        apply_polar_radius_range(&mut encoding, 0.5, Some(350.0)).unwrap();

        let range = encoding["radius"]["scale"]["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0]["expr"].as_str().unwrap(), "350/2*0.5");
        assert_eq!(range[1]["expr"].as_str().unwrap(), "350/2");
    }

    #[test]
    fn test_polar_inner_radius_zero() {
        // inner = 0 should still apply range (full pie, no donut hole)
        let mut encoding = json!({
            "radius": {
                "field": "dummy",
                "type": "nominal",
                "scale": {"domain": ["dummy"]}
            }
        });

        apply_polar_radius_range(&mut encoding, 0.0, Some(350.0)).unwrap();

        // Range should be [0, 350/2] for full pie
        let range = encoding["radius"]["scale"]["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0]["expr"].as_str().unwrap(), "350/2*0");
        assert_eq!(range[1]["expr"].as_str().unwrap(), "350/2");
    }

    #[test]
    fn test_map_position_to_vegalite_cartesian() {
        let renderer = CartesianProjection;
        assert_eq!(
            map_position_to_vegalite("pos1", &renderer),
            Some("x".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos2", &renderer),
            Some("y".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos1end", &renderer),
            Some("x2".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos2end", &renderer),
            Some("y2".to_string())
        );
        assert_eq!(map_position_to_vegalite("color", &renderer), None);
        assert_eq!(renderer.offset_channels(), ("xOffset", "yOffset"));
        assert_eq!(renderer.panel_size(), None);
    }

    #[test]
    fn test_map_position_to_vegalite_polar() {
        let renderer = PolarProjection;
        assert_eq!(
            map_position_to_vegalite("pos1", &renderer),
            Some("radius".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos2", &renderer),
            Some("theta".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos1end", &renderer),
            Some("radius2".to_string())
        );
        assert_eq!(
            map_position_to_vegalite("pos2end", &renderer),
            Some("theta2".to_string())
        );
        assert_eq!(renderer.offset_channels(), ("radiusOffset", "thetaOffset"));
        assert_eq!(
            renderer.panel_size(),
            Some((DEFAULT_POLAR_SIZE, DEFAULT_POLAR_SIZE))
        );
    }

    fn polar_point_layer() -> Value {
        json!({
            "mark": "point",
            "encoding": {
                "radius": {
                    "field": "r_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "theta": {
                    "field": "t_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 100.0]}
                }
            }
        })
    }

    #[test]
    fn test_polar_to_cartesian_pixel_coordinates() {
        let mut layer = polar_point_layer();
        let start = 0.0;
        let end = 2.0 * std::f64::consts::PI;

        convert_polar_to_cartesian(&mut layer, start, end, 0.0).unwrap();

        let transforms = layer["transform"].as_array().unwrap();

        // Should contain pixel-coordinate expressions using width/height signals
        let x_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_x__")
            .unwrap();
        let x_expr = x_calc["calculate"].as_str().unwrap();
        assert!(
            x_expr.contains("width / 2") && x_expr.contains("min(width, height) / 2"),
            "x should use pixel coordinates, got: {x_expr}"
        );

        let y_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_y__")
            .unwrap();
        let y_expr = y_calc["calculate"].as_str().unwrap();
        assert!(
            y_expr.contains("height / 2") && y_expr.contains("min(width, height) / 2"),
            "y should use pixel coordinates, got: {y_expr}"
        );

        // Encoding should use scale:null (raw pixel positions)
        assert_eq!(layer["encoding"]["x"]["scale"], json!(null));
        assert_eq!(layer["encoding"]["y"]["scale"], json!(null));

        // Original polar channels should be removed
        assert!(layer["encoding"].get("radius").is_none());
        assert!(layer["encoding"].get("theta").is_none());
    }

    #[test]
    fn test_polar_to_cartesian_filters_nulls() {
        let mut layer = polar_point_layer();
        let full_circle = 2.0 * std::f64::consts::PI;

        convert_polar_to_cartesian(&mut layer, 0.0, full_circle, 0.0).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let filter = transforms
            .iter()
            .find(|t| t.get("filter").is_some())
            .expect("should have a filter transform");

        let expr = filter["filter"].as_str().unwrap();
        assert!(
            expr.contains("isValid") && expr.contains("r_col") && expr.contains("t_col"),
            "filter should check both position fields, got: {expr}"
        );
    }

    #[test]
    fn test_get_projection_renderer() {
        let cartesian = get_projection_renderer(None);
        assert_eq!(cartesian.position_channels(), ("x", "y"));

        let polar_proj = Projection::polar();
        let polar = get_projection_renderer(Some(&polar_proj));
        assert_eq!(polar.position_channels(), ("radius", "theta"));
    }

    #[test]
    fn test_expr_normalize_radius() {
        // domain [0, 10], inner 0.2 → scale = (1.0 - 0.2) / (10 - 0) = 0.08
        let expr = expr_normalize_radius("datum.v", 0.0, 10.0, 0.2);
        assert!(expr.contains("0.08"), "scale factor should be 0.08, got: {expr}");
        assert!(expr.contains("datum.v"), "should reference value, got: {expr}");

        // domain [5, 15], inner 0 → scale = 1.0 / 10 = 0.1
        let expr = expr_normalize_radius("datum.x", 5.0, 15.0, 0.0);
        assert!(expr.contains("0.1"), "scale factor should be 0.1, got: {expr}");
    }

    #[test]
    fn test_expr_normalize_theta() {
        use std::f64::consts::PI;

        // domain [0, 100], partial circle 90°–270° (π/2 to 3π/2)
        let start = PI / 2.0;
        let end = 3.0 * PI / 2.0;
        let expr = expr_normalize_theta("datum.v", 0.0, 100.0, start, end);
        // scale = (3π/2 - π/2) / (100 - 0) = π / 100 ≈ 0.031416
        let expected_scale = PI / 100.0;
        assert!(
            expr.contains(&format!("{expected_scale}")),
            "scale factor should be π/100, got: {expr}"
        );
    }
}
