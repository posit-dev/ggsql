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
    /// Whether the spec uses faceting.
    fn is_faceted(&self) -> bool;

    /// Primary and secondary VL channel names for this projection.
    ///
    /// Returns `(pos1_channel, pos2_channel)`, e.g. `("x", "y")` for cartesian,
    /// `("radius", "theta")` for polar.
    fn position_channels(&self) -> (&'static str, &'static str);

    /// Offset channel names for this projection.
    ///
    /// Returns `(pos1_offset, pos2_offset)`, e.g. `("xOffset", "yOffset")`.
    fn offset_channels(&self) -> (&'static str, &'static str);

    /// Panel dimensions as VL values (`"container"` or explicit pixels).
    ///
    /// Returns `None` for faceted cartesian (VL handles sizing).
    fn panel_size(&self) -> Option<(Value, Value)> {
        if self.is_faceted() {
            None
        } else {
            Some((json!("container"), json!("container")))
        }
    }

    /// Apply projection-specific transformations to the VL spec.
    ///
    /// Called after layers are built but before faceting. May return a
    /// transformed DataFrame (e.g., polar currently clones it unchanged).
    fn transform_layers(
        &self,
        _spec: &Plot,
        _data: &DataFrame,
        _vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        Ok(None)
    }

    /// Vega-Lite layers to prepend before the data layers.
    fn background_layers(&self, _scales: &[Scale], _theme: &mut Value) -> Vec<Value> {
        Vec::new()
    }

    /// Vega-Lite layers to append after the data layers.
    fn foreground_layers(&self, _scales: &[Scale], _theme: &mut Value) -> Vec<Value> {
        Vec::new()
    }

    /// Apply all projection-specific work: transforms, clip, and panel decoration.
    fn apply_projection(
        &self,
        spec: &Plot,
        data: &DataFrame,
        theme: &mut Value,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        let result = self.transform_layers(spec, data, vl_spec)?;

        if let Some(ref project) = spec.project {
            if let Some(ParameterValue::Boolean(clip)) = project.properties.get("clip") {
                apply_clip_to_layers(vl_spec, *clip);
            }
        }

        let mut bg = self.background_layers(&spec.scales, theme);
        let mut fg = self.foreground_layers(&spec.scales, theme);
        if !(bg.is_empty() && fg.is_empty()) {
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

        Ok(result)
    }
}

// =============================================================================
// Factory
// =============================================================================

/// Get the projection renderer for a projection spec.
///
/// Returns the appropriate renderer based on the projection's coord kind,
/// or a Cartesian renderer if no projection is specified.
pub(super) fn get_projection_renderer(
    project: Option<&Projection>,
    is_faceted: bool,
) -> Box<dyn ProjectionRenderer> {
    match project.map(|p| p.coord.coord_kind()) {
        Some(CoordKind::Polar) => Box::new(PolarProjection {
            panel: PolarPanel::new(project, is_faceted),
        }),
        Some(CoordKind::Cartesian) | None => Box::new(CartesianProjection { is_faceted }),
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
struct CartesianProjection {
    is_faceted: bool,
}

impl ProjectionRenderer for CartesianProjection {
    fn is_faceted(&self) -> bool {
        self.is_faceted
    }

    fn position_channels(&self) -> (&'static str, &'static str) {
        ("x", "y")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("xOffset", "yOffset")
    }
}

// =============================================================================
// PolarProjection
// =============================================================================

/// Normalized outer radius (proportion of `min(width, height) / 2`).
const POLAR_OUTER: f64 = 1.0;

/// Bandwidth fraction for discrete polar offsets (mirrors VL's default
/// `1 - paddingInner` for band scales, which is ~0.9).
const POLAR_BAND_FRACTION: f64 = 0.9;

/// Pre-computed panel geometry for polar specs.
///
/// Holds angular range, radius bounds, and VL expression strings for the
/// panel centre and radius. In non-faceted specs these reference the
/// `width`/`height` signals; in faceted specs they are literal pixel values
/// (VL signals don't resolve inside faceted inner specs).
struct PolarPanel {
    is_faceted: bool,
    start: f64,
    end: f64,
    inner: f64,
    outer: f64,
    size: f64,
    cx: String,
    cy: String,
    radius: String,
}

impl PolarPanel {
    fn new(project: Option<&Projection>, is_faceted: bool) -> Self {
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
        let start = start_degrees * std::f64::consts::PI / 180.0;
        let end = end_degrees * std::f64::consts::PI / 180.0;
        let inner = prop("inner").unwrap_or(0.0);
        let size = prop("size").unwrap_or(DEFAULT_POLAR_SIZE);
        let (cx, cy, radius) = if is_faceted {
            let half = size / 2.0;
            (format!("{half}"), format!("{half}"), format!("{half}"))
        } else {
            (
                "width / 2".to_string(),
                "height / 2".to_string(),
                "min(width, height) / 2".to_string(),
            )
        };
        Self {
            is_faceted,
            start,
            end,
            inner,
            outer: POLAR_OUTER,
            size,
            cx,
            cy,
            radius,
        }
    }

    fn expr_x(&self, r: &str, theta: &str) -> String {
        format!("{} + {} * ({}) * sin({})", self.cx, self.radius, r, theta)
    }

    fn expr_y(&self, r: &str, theta: &str) -> String {
        format!("{} - {} * ({}) * cos({})", self.cy, self.radius, r, theta)
    }

    fn expr_radius(&self, r: &str) -> String {
        format!("{} * ({})", self.radius, r)
    }

    fn expr_normalize_radius(&self, value: &str, domain_min: f64, domain_max: f64) -> String {
        let scale = (self.outer - self.inner) / (domain_max - domain_min);
        format!("{} + {} * ({} - {})", self.inner, scale, value, domain_min)
    }

    fn expr_normalize_theta(&self, value: &str, domain_min: f64, domain_max: f64) -> String {
        let scale = (self.end - self.start) / (domain_max - domain_min);
        format!("{} + {} * ({} - {})", self.start, scale, value, domain_min)
    }
}

/// Polar projection — radius/theta coordinates for pie charts, rose plots, etc.
struct PolarProjection {
    panel: PolarPanel,
}

impl ProjectionRenderer for PolarProjection {
    fn is_faceted(&self) -> bool {
        self.panel.is_faceted
    }

    fn position_channels(&self) -> (&'static str, &'static str) {
        ("radius", "theta")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("radiusOffset", "thetaOffset")
    }

    fn panel_size(&self) -> Option<(Value, Value)> {
        if self.panel.is_faceted {
            let size = self.panel.size;
            Some((json!(size), json!(size)))
        } else {
            Some((json!("container"), json!("container")))
        }
    }

    fn transform_layers(
        &self,
        spec: &Plot,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        apply_polar_project(&self.panel, spec, data, vl_spec)
    }

    fn background_layers(&self, scales: &[Scale], theme: &mut Value) -> Vec<Value> {
        let mut layers = Vec::new();
        layers.extend(self.panel_arc(theme));
        layers.extend(self.grid_rings(scales, theme));
        layers.extend(self.grid_spokes(scales, theme));
        layers
    }

    fn foreground_layers(&self, scales: &[Scale], theme: &mut Value) -> Vec<Value> {
        let mut layers = Vec::new();
        layers.extend(self.radial_axis(scales, theme));
        layers.extend(self.angular_axis(scales, theme));
        layers
    }
}

impl PolarProjection {
    fn grid_rings(&self, scales: &[Scale], theme: &Value) -> Vec<Value> {
        let Some(scale) = scales.iter().find(|s| s.aesthetic == "pos1") else {
            return Vec::new();
        };
        let breaks = scale.numeric_breaks();
        let Some((domain_min, domain_max)) = scale.numeric_domain() else {
            return Vec::new();
        };
        if breaks.is_empty() || (domain_max - domain_min).abs() < f64::EPSILON {
            return Vec::new();
        }

        let color = theme
            .pointer("/axis/gridColor")
            .cloned()
            .unwrap_or(json!("#FFFFFF"));
        let width = theme
            .pointer("/axis/gridWidth")
            .cloned()
            .unwrap_or(json!(1));
        let p = &self.panel;

        let values: Vec<Value> = breaks.iter().map(|&b| json!({"v": b})).collect();
        let r_norm = p.expr_normalize_radius("datum.v", domain_min, domain_max);
        let radius_expr = p.expr_radius(&r_norm);

        vec![json!({
            "data": {"values": values},
            "mark": {
                "type": "arc",
                "fill": null,
                "stroke": color,
                "strokeWidth": width,
                "theta": p.start,
                "theta2": p.end,
            },
            "encoding": {
                "radius": {
                    "value": {"expr": radius_expr}
                }
            }
        })]
    }

    fn grid_spokes(&self, scales: &[Scale], theme: &Value) -> Vec<Value> {
        let Some(scale) = scales.iter().find(|s| s.aesthetic == "pos2") else {
            return Vec::new();
        };
        let breaks = scale.numeric_breaks();
        let Some((domain_min, domain_max)) = scale.numeric_domain() else {
            return Vec::new();
        };
        if breaks.is_empty() || (domain_max - domain_min).abs() < f64::EPSILON {
            return Vec::new();
        }

        let color = theme
            .pointer("/axis/gridColor")
            .cloned()
            .unwrap_or(json!("#FFFFFF"));
        let width = theme
            .pointer("/axis/gridWidth")
            .cloned()
            .unwrap_or(json!(1));
        let p = &self.panel;

        let values: Vec<Value> = breaks.iter().map(|&b| json!({"v": b})).collect();
        let theta = p.expr_normalize_theta("datum.v", domain_min, domain_max);
        let inner_s = format!("{}", p.inner);
        let outer_s = format!("{}", p.outer);

        vec![json!({
            "data": {"values": values},
            "mark": {
                "type": "rule",
                "stroke": color,
                "strokeWidth": width,
            },
            "transform": [
                {"calculate": p.expr_x(&inner_s, &theta), "as": "x"},
                {"calculate": p.expr_y(&inner_s, &theta), "as": "y"},
                {"calculate": p.expr_x(&outer_s, &theta), "as": "x2"},
                {"calculate": p.expr_y(&outer_s, &theta), "as": "y2"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "x2": {"field": "x2"},
                "y2": {"field": "y2"},
            }
        })]
    }

    fn radial_axis(&self, scales: &[Scale], theme: &Value) -> Vec<Value> {
        let Some(scale) = scales.iter().find(|s| s.aesthetic == "pos1") else {
            return Vec::new();
        };
        let break_labels = scale.break_labels();
        let Some((domain_min, domain_max)) = scale.numeric_domain() else {
            return Vec::new();
        };
        if (domain_max - domain_min).abs() < f64::EPSILON {
            return Vec::new();
        }

        let tick_color = theme
            .pointer("/axis/tickColor")
            .cloned()
            .unwrap_or(json!("#333333"));
        let tick_size = theme
            .pointer("/axis/tickSize")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0);
        let label_color = theme
            .pointer("/axis/labelColor")
            .cloned()
            .unwrap_or(json!("#4D4D4D"));
        let label_font_size = theme
            .pointer("/axis/labelFontSize")
            .cloned()
            .unwrap_or(json!(12));
        let line_color = theme
            .pointer("/axis/domainColor")
            .cloned()
            .unwrap_or(Value::Null);

        let p = &self.panel;
        let mut layers = Vec::new();

        // Axis line: rule from inner to outer at start angle
        let inner_s = format!("{}", p.inner);
        let start_s = format!("{}", p.start);
        let outer_s = format!("{}", p.outer);
        layers.push(json!({
            "data": {"values": [{}]},
            "mark": {
                "type": "rule",
                "stroke": line_color,
            },
            "transform": [
                {"calculate": p.expr_x(&inner_s, &start_s), "as": "x"},
                {"calculate": p.expr_y(&inner_s, &start_s), "as": "y"},
                {"calculate": p.expr_x(&outer_s, &start_s), "as": "x2"},
                {"calculate": p.expr_y(&outer_s, &start_s), "as": "y2"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "x2": {"field": "x2"},
                "y2": {"field": "y2"},
            }
        }));

        if break_labels.is_empty() {
            return layers;
        }

        // Tick marks: short perpendicular segments at each break.
        // The radial axis is at `start`, so ticks extend in the tangential
        // direction. We offset by ±tick_size pixels from the axis line.
        // In pixel space, the tangential unit vector at angle θ is
        // (cos(θ), sin(θ)), so we shift by that times half the tick size.
        let values: Vec<Value> = break_labels
            .iter()
            .map(|(v, label)| json!({"v": v, "label": label}))
            .collect();
        let r_norm = p.expr_normalize_radius("datum.v", domain_min, domain_max);

        let is_full_circle = (p.end - p.start - 2.0 * std::f64::consts::PI).abs() < f64::EPSILON;
        let tick_just: f64 = if is_full_circle { 0.5 } else { 0.0 };
        let (sin_start, cos_start) = p.start.sin_cos();
        let dx_out = format!("{}", (1.0 - tick_just) * tick_size * cos_start);
        let dy_out = format!("{}", (1.0 - tick_just) * tick_size * sin_start);
        let dx_in = format!("{}", tick_just * tick_size * cos_start);
        let dy_in = format!("{}", tick_just * tick_size * sin_start);

        let cx = p.expr_x(&r_norm, &start_s);
        let cy = p.expr_y(&r_norm, &start_s);

        layers.push(json!({
            "data": {"values": values.clone()},
            "mark": {
                "type": "rule",
                "stroke": tick_color,
            },
            "transform": [
                {"calculate": cx, "as": "cx"},
                {"calculate": cy, "as": "cy"},
                {"calculate": format!("datum.cx - {dx_out}"), "as": "x"},
                {"calculate": format!("datum.cy - {dy_out}"), "as": "y"},
                {"calculate": format!("datum.cx + {dx_in}"), "as": "x2"},
                {"calculate": format!("datum.cy + {dy_in}"), "as": "y2"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "x2": {"field": "x2"},
                "y2": {"field": "y2"},
            }
        }));

        // Labels: text positioned beyond the outer end of the tick
        let label_pad = 2.0;
        let label_offset = (1.0 - tick_just) * tick_size + label_pad;
        let lx = format!("{}", -label_offset * cos_start);
        let ly = format!("{}", -label_offset * sin_start);

        layers.push(json!({
            "data": {"values": values},
            "mark": {
                "type": "text",
                "color": label_color,
                "fontSize": label_font_size,
                "align": if cos_start > 0.1 { "right" } else if cos_start < -0.1 { "left" } else { "center" },
                "baseline": if sin_start > 0.1 { "bottom" } else if sin_start < -0.1 { "top" } else { "middle" },
            },
            "transform": [
                {"calculate": format!("{cx} + {lx}"), "as": "x"},
                {"calculate": format!("{cy} + {ly}"), "as": "y"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "text": {"field": "label", "type": "nominal"},
            }
        }));

        layers
    }

    fn angular_axis(&self, scales: &[Scale], theme: &Value) -> Vec<Value> {
        let Some(scale) = scales.iter().find(|s| s.aesthetic == "pos2") else {
            return Vec::new();
        };
        let break_labels = scale.break_labels();
        let Some((domain_min, domain_max)) = scale.numeric_domain() else {
            return Vec::new();
        };
        if (domain_max - domain_min).abs() < f64::EPSILON {
            return Vec::new();
        }

        let tick_color = theme
            .pointer("/axis/tickColor")
            .cloned()
            .unwrap_or(json!("#333333"));
        let tick_size = theme
            .pointer("/axis/tickSize")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0);
        let label_color = theme
            .pointer("/axis/labelColor")
            .cloned()
            .unwrap_or(json!("#4D4D4D"));
        let label_font_size = theme
            .pointer("/axis/labelFontSize")
            .cloned()
            .unwrap_or(json!(12));
        let line_color = theme
            .pointer("/axis/domainColor")
            .cloned()
            .unwrap_or(Value::Null);

        let p = &self.panel;
        let mut layers = Vec::new();

        // Axis arc along the outer edge
        let outer_s = format!("{}", p.outer);
        let radius_expr = p.expr_radius(&outer_s);
        layers.push(json!({
            "data": {"values": [{}]},
            "mark": {
                "type": "arc",
                "fill": null,
                "stroke": line_color,
                "theta": p.start,
                "theta2": p.end,
            },
            "encoding": {
                "radius": {
                    "value": {"expr": radius_expr}
                }
            }
        }));

        if break_labels.is_empty() {
            return layers;
        }

        // Ticks: short radial segments at each theta break, pointing inward.
        // The tick direction at angle θ is along the radius vector:
        // unit = (sin(θ), -cos(θ)) in pixel space.
        let values: Vec<Value> = break_labels
            .iter()
            .map(|(v, label)| json!({"v": v, "label": label}))
            .collect();
        let theta = p.expr_normalize_theta("datum.v", domain_min, domain_max);

        let is_full_circle = (p.end - p.start - 2.0 * std::f64::consts::PI).abs() < f64::EPSILON;
        let tick_just: f64 = if is_full_circle { 0.5 } else { 0.0 };

        let outer_cx = p.expr_x(&outer_s, &theta);
        let outer_cy = p.expr_y(&outer_s, &theta);

        // Radial unit vector at angle θ is (sin(θ), -cos(θ)) in pixel space,
        // scaled by min(width,height)/2. Since the tick is small, we use the
        // precomputed center and offset by fixed pixel amounts via the
        // normalized radius direction.
        let inward = format!("{}", tick_just * tick_size);
        let outward = format!("{}", (1.0 - tick_just) * tick_size);

        layers.push(json!({
            "data": {"values": values.clone()},
            "mark": {
                "type": "rule",
                "stroke": tick_color,
            },
            "transform": [
                {"calculate": &theta, "as": "theta"},
                {"calculate": outer_cx, "as": "cx"},
                {"calculate": outer_cy, "as": "cy"},
                {"calculate": format!("datum.cx + {outward} * sin(datum.theta)"), "as": "x"},
                {"calculate": format!("datum.cy - {outward} * cos(datum.theta)"), "as": "y"},
                {"calculate": format!("datum.cx - {inward} * sin(datum.theta)"), "as": "x2"},
                {"calculate": format!("datum.cy + {inward} * cos(datum.theta)"), "as": "y2"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "x2": {"field": "x2"},
                "y2": {"field": "y2"},
            }
        }));

        // Labels: one sub-layer per (align, baseline) combination.
        // All break values live in the parent data with an `_ab` tag; each
        // child filters on its tag and sets the corresponding mark alignment.
        let label_pad = 2.0;
        let label_offset = format!("{}", (1.0 - tick_just) * tick_size + label_pad);
        let theta_scale = (p.end - p.start) / (domain_max - domain_min);

        let mut label_values = Vec::new();
        let mut alignment_keys = std::collections::BTreeSet::new();
        for &(v, ref label) in &break_labels {
            let angle = p.start + theta_scale * (v - domain_min);
            let (sin_a, cos_a) = angle.sin_cos();
            let align = if sin_a > 0.1 {
                "left"
            } else if sin_a < -0.1 {
                "right"
            } else {
                "center"
            };
            let baseline = if cos_a > 0.1 {
                "bottom"
            } else if cos_a < -0.1 {
                "top"
            } else {
                "middle"
            };
            let ab = format!("{align}/{baseline}");
            alignment_keys.insert(ab.clone());
            label_values.push(json!({"v": v, "label": label, "_ab": ab}));
        }

        let sub_layers: Vec<Value> = alignment_keys
            .into_iter()
            .map(|ab| {
                let (align, baseline) = ab.split_once('/').unwrap();
                json!({
                    "transform": [
                        {"filter": {"field": "_ab", "equal": ab}},
                    ],
                    "mark": {
                        "type": "text",
                        "color": label_color,
                        "fontSize": label_font_size,
                        "align": align,
                        "baseline": baseline,
                    },
                })
            })
            .collect();

        layers.push(json!({
            "data": {"values": label_values},
            "transform": [
                {"calculate": &theta, "as": "theta"},
                {"calculate": outer_cx, "as": "cx"},
                {"calculate": outer_cy, "as": "cy"},
                {"calculate": format!("datum.cx + {label_offset} * sin(datum.theta)"), "as": "x"},
                {"calculate": format!("datum.cy - {label_offset} * cos(datum.theta)"), "as": "y"},
            ],
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "scale": null, "axis": null},
                "y": {"field": "y", "type": "quantitative", "scale": null, "axis": null},
                "text": {"field": "label", "type": "nominal"},
            },
            "layer": sub_layers,
        }));

        layers
    }

    fn panel_arc(&self, theme: &mut Value) -> Vec<Value> {
        let Some(view) = theme.get_mut("view").and_then(|v| v.as_object_mut()) else {
            return Vec::new();
        };
        let fill = view.remove("fill").unwrap_or(Value::Null);
        let stroke = view.remove("stroke").unwrap_or(Value::Null);

        // We need a null-stroke otherwise it'll show up as a gray line
        view.insert("stroke".to_string(), Value::Null);

        let p = &self.panel;

        let inner_s = format!("{}", p.inner);
        let outer_s = format!("{}", p.outer);

        let mut mark = json!({
            "type": "arc",
            "fill": fill,
            "stroke": stroke,
            "theta":  p.start,
            "theta2": p.end,
        });
        if p.inner > 0.0 {
            mark["innerRadius"] = json!({"expr": p.expr_radius(&inner_s)});
        }
        mark["outerRadius"] = json!({"expr": p.expr_radius(&outer_s)});

        vec![json!({
            "data": {"values": [{}]},
            "mark": mark
        })]
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
    panel: &PolarPanel,
    spec: &Plot,
    data: &DataFrame,
    vl_spec: &mut Value,
) -> Result<Option<DataFrame>> {
    // Convert geoms to polar equivalents and apply angle range + inner radius
    convert_geoms_to_polar(panel, spec, vl_spec)?;

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
fn convert_geoms_to_polar(panel: &PolarPanel, spec: &Plot, vl_spec: &mut Value) -> Result<()> {
    if let Some(layers_arr) = get_layers_mut(vl_spec) {
        for layer in layers_arr {
            if let Some(mark) = layer.get_mut("mark") {
                let polar_mark = convert_mark_to_polar(mark, spec)?;
                let is_arc = polar_mark.as_str() == Some("arc");
                *mark = polar_mark;

                if is_arc {
                    // Arc marks natively support radius/theta channels
                    if let Some(encoding) = layer.get_mut("encoding") {
                        apply_polar_angle_range(encoding, panel)?;
                        apply_polar_radius_range(encoding, panel)?;
                    }
                } else {
                    // Non-arc marks (point, line): convert polar to cartesian
                    convert_polar_to_cartesian(layer, panel)?;
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
fn convert_polar_to_cartesian(layer: &mut Value, panel: &PolarPanel) -> Result<()> {
    // Phase 1: Extract info from encoding (immutable read)
    let (r_val, r_field, r_domain, r_title, r_discrete,
         theta_val, theta_field, theta_domain, theta_title, theta_discrete,
         r2_field, theta2_field, r_offset_field, theta_offset_field) = {
        let encoding = layer
            .get("encoding")
            .and_then(|e| e.as_object())
            .ok_or_else(|| GgsqlError::WriterError("Layer has no encoding object".to_string()))?;

        let (r_val, r_field, r_domain, r_title, r_disc) =
            extract_polar_channel(encoding, "radius")?;
        let (theta_val, theta_field, theta_domain, theta_title, theta_disc) =
            extract_polar_channel(encoding, "theta")?;
        let field_of = |channel: &str| {
            encoding.get(channel)
                .and_then(|e| e.get("field"))
                .and_then(|f| f.as_str())
                .map(|s| s.to_string())
        };
        (
            r_val, r_field, r_domain, r_title, r_disc,
            theta_val, theta_field, theta_domain, theta_title, theta_disc,
            field_of("radius2"), field_of("theta2"),
            field_of("radiusOffset"), field_of("thetaOffset"),
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
        panel.expr_normalize_theta(&theta_val, theta_min, theta_max)
    } else {
        format!("{}", panel.start)
    };
    polar_transforms.push(json!({"calculate": theta_expr, "as": "__polar_theta__"}));

    let r_expr = if (r_max - r_min).abs() > f64::EPSILON {
        panel.expr_normalize_radius(&r_val, r_min, r_max)
    } else {
        format!("{}", (panel.outer + panel.inner) / 2.0)
    };
    polar_transforms.push(json!({"calculate": r_expr, "as": "__polar_r__"}));

    // Offsets: fold into the normalized r/theta before computing pixel x/y.
    // If the offset has a scale domain, normalize it into the primary channel's
    // space. If no domain, treat as raw pixel displacement after conversion.
    let encoding_obj = layer.get("encoding").and_then(|e| e.as_object());
    let mut r_final = "datum.__polar_r__".to_string();
    let mut theta_final = "datum.__polar_theta__".to_string();
    let mut pixel_offsets: Vec<(String, bool)> = Vec::new(); // (field, is_radial)

    let offset_domain = |channel: &str| -> Option<(f64, f64)> {
        let arr = encoding_obj?
            .get(channel)?
            .get("scale")?
            .get("domain")?
            .as_array()?;
        Some((arr.first()?.as_f64()?, arr.get(1)?.as_f64()?))
    };

    if let Some(ref f) = r_offset_field {
        if let Some((off_min, off_max)) = offset_domain("radiusOffset") {
            let r_scale = if (r_max - r_min).abs() > f64::EPSILON {
                (panel.outer - panel.inner) / (r_max - r_min)
            } else {
                0.0
            };
            let bw = if r_discrete { POLAR_BAND_FRACTION } else { 1.0 };
            r_final = format!(
                "datum.__polar_r__ + {} * ((datum['{}'] - {}) / {} - 0.5)",
                r_scale * bw, f, off_min, off_max - off_min
            );
        } else {
            pixel_offsets.push((f.clone(), true));
        }
    }
    if let Some(ref f) = theta_offset_field {
        if let Some((off_min, off_max)) = offset_domain("thetaOffset") {
            let t_scale = if (theta_max - theta_min).abs() > f64::EPSILON {
                (panel.end - panel.start) / (theta_max - theta_min)
            } else {
                0.0
            };
            let bw = if theta_discrete { POLAR_BAND_FRACTION } else { 1.0 };
            theta_final = format!(
                "datum.__polar_theta__ + {} * ((datum['{}'] - {}) / {} - 0.5)",
                t_scale * bw, f, off_min, off_max - off_min
            );
        } else {
            pixel_offsets.push((f.clone(), false));
        }
    }

    let mut x_expr = panel.expr_x(&r_final, &theta_final);
    let mut y_expr = panel.expr_y(&r_final, &theta_final);

    // Raw pixel offsets applied after polar→cartesian conversion
    for (f, is_radial) in &pixel_offsets {
        if *is_radial {
            x_expr = format!("({x_expr}) + datum['{f}'] * sin(datum.__polar_theta__)");
            y_expr = format!("({y_expr}) - datum['{f}'] * cos(datum.__polar_theta__)");
        } else {
            x_expr = format!("({x_expr}) + datum['{f}'] * cos(datum.__polar_theta__)");
            y_expr = format!("({y_expr}) + datum['{f}'] * sin(datum.__polar_theta__)");
        }
    }

    polar_transforms.push(json!({"calculate": x_expr, "as": "__polar_x__"}));
    polar_transforms.push(json!({"calculate": y_expr, "as": "__polar_y__"}));

    // Secondary channels (radius2 → x2/y2, theta2 → x2/y2) share the
    // primary channel's domain, so we reuse the same normalization parameters.
    let has_r2 = r2_field.is_some();
    let has_theta2 = theta2_field.is_some();
    if has_r2 || has_theta2 {
        let r2_expr = if let Some(ref f) = r2_field {
            if (r_max - r_min).abs() > f64::EPSILON {
                panel.expr_normalize_radius(&format!("datum['{}']", f), r_min, r_max)
            } else {
                format!("{}", (panel.outer + panel.inner) / 2.0)
            }
        } else {
            "datum.__polar_r__".to_string()
        };
        let theta2_expr = if let Some(ref f) = theta2_field {
            if (theta_max - theta_min).abs() > f64::EPSILON {
                panel.expr_normalize_theta(&format!("datum['{}']", f), theta_min, theta_max)
            } else {
                format!("{}", panel.start)
            }
        } else {
            "datum.__polar_theta__".to_string()
        };
        polar_transforms.push(json!({"calculate": r2_expr, "as": "__polar_r2__"}));
        polar_transforms.push(json!({"calculate": theta2_expr, "as": "__polar_theta2__"}));
        polar_transforms.push(json!({
            "calculate": panel.expr_x("datum.__polar_r2__", "datum.__polar_theta2__"),
            "as": "__polar_x2__"
        }));
        polar_transforms.push(json!({
            "calculate": panel.expr_y("datum.__polar_r2__", "datum.__polar_theta2__"),
            "as": "__polar_y2__"
        }));
    }

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
    encoding.remove("radius2");
    encoding.remove("theta2");
    encoding.remove("radiusOffset");
    encoding.remove("thetaOffset");

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

    if has_r2 || has_theta2 {
        encoding.insert("x2".to_string(), json!({"field": "__polar_x2__"}));
        encoding.insert("y2".to_string(), json!({"field": "__polar_y2__"}));
    }

    Ok(())
}

/// Extract field name, numeric value expression, scale domain, and title from
/// a polar encoding channel.
///
/// Returns `(value_expr, field, (domain_min, domain_max), optional_title, is_discrete)`.
/// For continuous scales `value_expr` is `datum['field']`.
/// For discrete scales it is `indexof([...], datum['field']) + 1` with a
/// synthesized numeric domain `(0.5, n + 0.5)`.
fn extract_polar_channel(
    encoding: &serde_json::Map<String, Value>,
    channel: &str,
) -> Result<(String, String, (f64, f64), Option<Value>, bool)> {
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

    let title = channel_enc.get("title").cloned();

    let domain_arr = channel_enc
        .get("scale")
        .and_then(|s| s.get("domain"))
        .and_then(|d| d.as_array());

    // Try numeric domain first
    if let Some((min, max)) = domain_arr.and_then(|arr| {
        Some((arr.first()?.as_f64()?, arr.get(1)?.as_f64()?))
    }) {
        return Ok((format!("datum['{}']", field), field, (min, max), title, false));
    }

    // Discrete domain: string array → indexof + synthesized numeric domain
    if let Some(arr) = domain_arr {
        let strings: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        if !strings.is_empty() {
            let n = strings.len();
            let literal: String = strings
                .iter()
                .map(|s| format!("'{}'", s.replace('\'', "\\'")))
                .collect::<Vec<_>>()
                .join(",");
            // indexof returns -1 for values not in the domain; map those to null
            let arr_expr = format!("[{}]", literal);
            let expr = format!(
                "indexof({arr}, datum['{field}']) < 0 ? null : indexof({arr}, datum['{field}']) + 1",
                arr = arr_expr,
                field = field,
            );
            return Ok((expr, field, (0.5, n as f64 + 0.5), title, true));
        }
    }

    // Fallback
    Ok((format!("datum['{}']", field), field, (0.0, 1.0), title, false))
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
fn apply_polar_angle_range(encoding: &mut Value, panel: &PolarPanel) -> Result<()> {
    // Skip if default range (0 to 2π)
    let is_default = panel.start.abs() <= f64::EPSILON
        && (panel.end - 2.0 * std::f64::consts::PI).abs() <= f64::EPSILON;
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
                    scale_obj.insert("range".to_string(), json!([panel.start, panel.end]));
                }
            } else {
                // No existing scale, create new one with just range
                theta_obj.insert(
                    "scale".to_string(),
                    json!({
                        "range": [panel.start, panel.end]
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
fn apply_polar_radius_range(encoding: &mut Value, panel: &PolarPanel) -> Result<()> {
    let enc_obj = encoding
        .as_object_mut()
        .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

    let inner_s = format!("{}", panel.inner);
    let outer_s = format!("{}", panel.outer);
    let inner_expr = panel.expr_radius(&inner_s);
    let outer_expr = panel.expr_radius(&outer_s);

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

        let mut proj = Projection::polar();
        proj.properties
            .insert("inner".to_string(), ParameterValue::Number(0.5));
        let panel = PolarPanel::new(Some(&proj), false);
        apply_polar_radius_range(&mut encoding, &panel).unwrap();

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

        let mut proj = Projection::polar();
        proj.properties
            .insert("inner".to_string(), ParameterValue::Number(0.5));
        proj.properties
            .insert("size".to_string(), ParameterValue::Number(350.0));
        let panel = PolarPanel::new(Some(&proj), true);
        apply_polar_radius_range(&mut encoding, &panel).unwrap();

        let range = encoding["radius"]["scale"]["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0]["expr"].as_str().unwrap(), "175 * (0.5)");
        assert_eq!(range[1]["expr"].as_str().unwrap(), "175 * (1)");
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

        let mut proj = Projection::polar();
        proj.properties
            .insert("size".to_string(), ParameterValue::Number(350.0));
        let panel = PolarPanel::new(Some(&proj), true);
        apply_polar_radius_range(&mut encoding, &panel).unwrap();

        // Range should be [0, 350/2] for full pie
        let range = encoding["radius"]["scale"]["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0]["expr"].as_str().unwrap(), "175 * (0)");
        assert_eq!(range[1]["expr"].as_str().unwrap(), "175 * (1)");
    }

    #[test]
    fn test_map_position_to_vegalite_cartesian() {
        let renderer = CartesianProjection { is_faceted: false };
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
        assert_eq!(
            renderer.panel_size(),
            Some((json!("container"), json!("container")))
        );
    }

    #[test]
    fn test_map_position_to_vegalite_polar() {
        let renderer = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
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
            Some((json!("container"), json!("container")))
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
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

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
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

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
        let cartesian = get_projection_renderer(None, false);
        assert_eq!(cartesian.position_channels(), ("x", "y"));

        let polar_proj = Projection::polar();
        let polar = get_projection_renderer(Some(&polar_proj), false);
        assert_eq!(polar.position_channels(), ("radius", "theta"));
    }

    #[test]
    fn test_expr_normalize_radius() {
        let panel = PolarPanel::new(None, false);

        // domain [0, 10], inner 0.2 — build a panel with inner=0.2
        let mut p = panel;
        p.inner = 0.2;
        // scale = (1.0 - 0.2) / (10 - 0) = 0.08
        let expr = p.expr_normalize_radius("datum.v", 0.0, 10.0);
        assert!(
            expr.contains("0.08"),
            "scale factor should be 0.08, got: {expr}"
        );
        assert!(
            expr.contains("datum.v"),
            "should reference value, got: {expr}"
        );

        // domain [5, 15], inner 0 → scale = 1.0 / 10 = 0.1
        p.inner = 0.0;
        let expr = p.expr_normalize_radius("datum.x", 5.0, 15.0);
        assert!(
            expr.contains("0.1"),
            "scale factor should be 0.1, got: {expr}"
        );
    }

    #[test]
    fn test_expr_normalize_theta() {
        use std::f64::consts::PI;

        // domain [0, 100], partial circle 90°–270° (π/2 to 3π/2)
        let mut panel = PolarPanel::new(None, false);
        panel.start = PI / 2.0;
        panel.end = 3.0 * PI / 2.0;
        let expr = panel.expr_normalize_theta("datum.v", 0.0, 100.0);
        // scale = (3π/2 - π/2) / (100 - 0) = π / 100 ≈ 0.031416
        let expected_scale = PI / 100.0;
        assert!(
            expr.contains(&format!("{expected_scale}")),
            "scale factor should be π/100, got: {expr}"
        );
    }

    fn scale_with_breaks(aesthetic: &str, domain: (f64, f64), breaks: Vec<f64>) -> Scale {
        use crate::plot::types::ArrayElement;
        let mut scale = Scale::new(aesthetic);
        scale.input_range = Some(vec![
            ArrayElement::Number(domain.0),
            ArrayElement::Number(domain.1),
        ]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(breaks.into_iter().map(ArrayElement::Number).collect()),
        );
        scale
    }

    #[test]
    fn test_grid_rings() {
        let scales = vec![scale_with_breaks(
            "pos1",
            (0.0, 100.0),
            vec![25.0, 50.0, 75.0],
        )];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({"axis": {"gridColor": "#FFF", "gridWidth": 2}});

        let layers = proj.grid_rings(&scales, &theme);
        assert_eq!(layers.len(), 1, "should produce one layer");

        let layer = &layers[0];

        // Data should contain the break values
        let values = layer["data"]["values"].as_array().unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0]["v"], json!(25.0));
        assert_eq!(values[1]["v"], json!(50.0));
        assert_eq!(values[2]["v"], json!(75.0));

        // Mark should be a stroke-only arc
        assert_eq!(layer["mark"]["type"], "arc");
        assert_eq!(layer["mark"]["fill"], json!(null));
        assert_eq!(layer["mark"]["stroke"], "#FFF");
        assert_eq!(layer["mark"]["strokeWidth"], 2.0);

        // Radius encoding should use an expression
        let radius_expr = layer["encoding"]["radius"]["value"]["expr"]
            .as_str()
            .unwrap();
        assert!(
            radius_expr.contains("min(width, height) / 2"),
            "radius should use expr_polar_radius, got: {radius_expr}"
        );
    }

    #[test]
    fn test_grid_spokes() {
        let scales = vec![scale_with_breaks("pos2", (0.0, 60.0), vec![20.0, 40.0])];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({"axis": {"gridColor": "#CCC", "gridWidth": 1}});

        let layers = proj.grid_spokes(&scales, &theme);
        assert_eq!(layers.len(), 1, "should produce one layer");

        let layer = &layers[0];

        // Data should contain the break values
        let values = layer["data"]["values"].as_array().unwrap();
        assert_eq!(values.len(), 2);

        // Mark should be a rule
        assert_eq!(layer["mark"]["type"], "rule");
        assert_eq!(layer["mark"]["stroke"], "#CCC");

        // Should have calculate transforms for x, y, x2, y2
        let transforms = layer["transform"].as_array().unwrap();
        assert_eq!(transforms.len(), 4);
        let field_names: Vec<&str> = transforms.iter().filter_map(|t| t["as"].as_str()).collect();
        assert_eq!(field_names, vec!["x", "y", "x2", "y2"]);

        // Encoding should use scale:null for pixel positions
        assert_eq!(layer["encoding"]["x"]["scale"], json!(null));
        assert_eq!(layer["encoding"]["y"]["scale"], json!(null));
    }

    #[test]
    fn test_radial_axis() {
        let scales = vec![scale_with_breaks(
            "pos1",
            (0.0, 100.0),
            vec![25.0, 50.0, 75.0],
        )];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({
            "axis": {
                "tickColor": "#333",
                "tickSize": 6,
                "labelColor": "#4D4D4D",
                "labelFontSize": 12,
            }
        });

        let layers = proj.radial_axis(&scales, &theme);
        assert_eq!(
            layers.len(),
            3,
            "should produce axis line, ticks, and labels"
        );

        // Layer 0: axis line (single rule from inner to outer)
        let line = &layers[0];
        assert_eq!(line["mark"]["type"], "rule");
        assert_eq!(line["data"]["values"].as_array().unwrap().len(), 1);
        let transforms = line["transform"].as_array().unwrap();
        let fields: Vec<&str> = transforms.iter().filter_map(|t| t["as"].as_str()).collect();
        assert_eq!(fields, vec!["x", "y", "x2", "y2"]);

        // Layer 1: ticks (one per break)
        let ticks = &layers[1];
        assert_eq!(ticks["mark"]["type"], "rule");
        assert_eq!(ticks["data"]["values"].as_array().unwrap().len(), 3);
        let tick_transforms = ticks["transform"].as_array().unwrap();
        let tick_fields: Vec<&str> = tick_transforms
            .iter()
            .filter_map(|t| t["as"].as_str())
            .collect();
        assert_eq!(tick_fields, vec!["cx", "cy", "x", "y", "x2", "y2"]);

        // Layer 2: labels (one per break)
        let labels = &layers[2];
        assert_eq!(labels["mark"]["type"], "text");
        assert_eq!(labels["data"]["values"].as_array().unwrap().len(), 3);
        assert_eq!(labels["encoding"]["text"]["field"], "label");
        assert_eq!(labels["encoding"]["x"]["scale"], json!(null));
    }

    #[test]
    fn test_radial_axis_no_breaks() {
        let scales = vec![scale_with_breaks("pos1", (0.0, 100.0), vec![])];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({"axis": {}});

        let layers = proj.radial_axis(&scales, &theme);
        assert_eq!(
            layers.len(),
            1,
            "should produce only the axis line when no breaks"
        );
        assert_eq!(layers[0]["mark"]["type"], "rule");
    }

    #[test]
    fn test_angular_axis() {
        let scales = vec![scale_with_breaks(
            "pos2",
            (0.0, 60.0),
            vec![15.0, 30.0, 45.0],
        )];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({
            "axis": {
                "tickColor": "#333",
                "tickSize": 6,
                "labelColor": "#4D4D4D",
                "labelFontSize": 12,
            }
        });

        let layers = proj.angular_axis(&scales, &theme);
        assert_eq!(
            layers.len(),
            3,
            "should produce axis arc, ticks, and labels"
        );

        // Layer 0: axis arc along outer edge
        let arc = &layers[0];
        assert_eq!(arc["mark"]["type"], "arc");
        assert_eq!(arc["mark"]["fill"], json!(null));

        // Layer 1: ticks (one per break)
        let ticks = &layers[1];
        assert_eq!(ticks["mark"]["type"], "rule");
        assert_eq!(ticks["data"]["values"].as_array().unwrap().len(), 3);
        let tick_transforms = ticks["transform"].as_array().unwrap();
        let tick_fields: Vec<&str> = tick_transforms
            .iter()
            .filter_map(|t| t["as"].as_str())
            .collect();
        assert_eq!(tick_fields, vec!["theta", "cx", "cy", "x", "y", "x2", "y2"]);

        // Layer 2: nested label layer with shared data/transforms/encoding
        let labels = &layers[2];
        assert_eq!(labels["encoding"]["text"]["field"], "label");
        assert_eq!(labels["data"]["values"].as_array().unwrap().len(), 3);
        let sub_layers = labels["layer"].as_array().unwrap();
        assert!(
            !sub_layers.is_empty(),
            "should have at least one label sub-layer"
        );
        for sub in sub_layers {
            assert_eq!(sub["mark"]["type"], "text");
            assert!(sub["mark"]["align"].is_string());
            assert!(sub["mark"]["baseline"].is_string());
            // Each sub-layer filters by alignment tag
            assert!(sub["transform"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t.get("filter").is_some()));
        }
    }

    #[test]
    fn test_angular_axis_no_breaks() {
        let scales = vec![scale_with_breaks("pos2", (0.0, 60.0), vec![])];
        let proj = PolarProjection {
            panel: PolarPanel::new(None, false),
        };
        let theme = json!({"axis": {}});

        let layers = proj.angular_axis(&scales, &theme);
        assert_eq!(
            layers.len(),
            1,
            "should produce only the axis arc when no breaks"
        );
        assert_eq!(layers[0]["mark"]["type"], "arc");
    }

    // =========================================================================
    // Discrete channel: indexof expression
    // =========================================================================

    fn discrete_theta_layer() -> Value {
        json!({
            "mark": "point",
            "encoding": {
                "radius": {
                    "field": "r_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "theta": {
                    "field": "cat",
                    "type": "nominal",
                    "scale": {"domain": ["A", "B", "C"]}
                }
            }
        })
    }

    #[test]
    fn test_discrete_theta_uses_indexof() {
        let mut layer = discrete_theta_layer();
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let theta_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_theta__")
            .unwrap();
        let expr = theta_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains("indexof") && expr.contains("'A'") && expr.contains("datum['cat']"),
            "theta should use indexof for discrete domain, got: {expr}"
        );
        assert!(
            expr.contains("null"),
            "OOB values should map to null, got: {expr}"
        );
    }

    #[test]
    fn test_discrete_indexof_escapes_quotes() {
        let mut layer = json!({
            "mark": "point",
            "encoding": {
                "radius": {
                    "field": "r_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "theta": {
                    "field": "cat",
                    "type": "nominal",
                    "scale": {"domain": ["it's", "fine"]}
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let theta_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_theta__")
            .unwrap();
        let expr = theta_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains("it\\'s"),
            "single quotes in category names should be escaped, got: {expr}"
        );
    }

    #[test]
    fn test_discrete_theta_synthesizes_domain() {
        let mut layer = discrete_theta_layer();
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        // 3 categories → domain (0.5, 3.5), full circle → scale = 2π / 3.0
        let transforms = layer["transform"].as_array().unwrap();
        let theta_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_theta__")
            .unwrap();
        let expr = theta_calc["calculate"].as_str().unwrap();
        let expected_scale = 2.0 * std::f64::consts::PI / 3.0;
        assert!(
            expr.contains(&format!("{expected_scale}")),
            "theta scale should be 2π/3 ≈ {expected_scale}, got: {expr}"
        );
    }

    // =========================================================================
    // Secondary channels: radius2 / theta2
    // =========================================================================

    #[test]
    fn test_radius2_generates_x2_y2() {
        let mut layer = json!({
            "mark": "rule",
            "encoding": {
                "radius": {
                    "field": "r_start",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "radius2": {
                    "field": "r_end"
                },
                "theta": {
                    "field": "angle",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 100.0]}
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let has_r2 = transforms.iter().any(|t| t["as"] == "__polar_r2__");
        let has_x2 = transforms.iter().any(|t| t["as"] == "__polar_x2__");
        let has_y2 = transforms.iter().any(|t| t["as"] == "__polar_y2__");
        assert!(has_r2, "should compute __polar_r2__");
        assert!(has_x2, "should compute __polar_x2__");
        assert!(has_y2, "should compute __polar_y2__");

        assert!(layer["encoding"].get("x2").is_some());
        assert!(layer["encoding"].get("y2").is_some());
        assert!(layer["encoding"].get("radius2").is_none());
    }

    #[test]
    fn test_theta2_generates_x2_y2() {
        let mut layer = json!({
            "mark": "rule",
            "encoding": {
                "radius": {
                    "field": "r_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "theta": {
                    "field": "t_start",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 100.0]}
                },
                "theta2": {
                    "field": "t_end"
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let theta2_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_theta2__")
            .unwrap();
        let expr = theta2_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains("datum['t_end']"),
            "theta2 should use its own field, got: {expr}"
        );

        assert!(layer["encoding"].get("x2").is_some());
        assert!(layer["encoding"].get("theta2").is_none());
    }

    // =========================================================================
    // Offset channels: scaled domain
    // =========================================================================

    #[test]
    fn test_theta_offset_with_domain() {
        let mut layer = json!({
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
                },
                "thetaOffset": {
                    "field": "grp",
                    "scale": {"domain": [0.0, 4.0]}
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let x_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_x__")
            .unwrap();
        let expr = x_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains("datum['grp']"),
            "x should incorporate thetaOffset field, got: {expr}"
        );

        assert!(layer["encoding"].get("thetaOffset").is_none());
    }

    #[test]
    fn test_radius_offset_without_domain_is_pixel() {
        let mut layer = json!({
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
                },
                "radiusOffset": {
                    "field": "jitter"
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        let transforms = layer["transform"].as_array().unwrap();
        let x_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_x__")
            .unwrap();
        let expr = x_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains("datum['jitter']") && expr.contains("sin"),
            "pixel offset should apply along radial direction, got: {expr}"
        );
    }

    // =========================================================================
    // Discrete offset band fraction
    // =========================================================================

    #[test]
    fn test_discrete_theta_offset_applies_band_fraction() {
        let mut layer = json!({
            "mark": "point",
            "encoding": {
                "radius": {
                    "field": "r_col",
                    "type": "quantitative",
                    "scale": {"domain": [0.0, 10.0]}
                },
                "theta": {
                    "field": "cat",
                    "type": "nominal",
                    "scale": {"domain": ["A", "B", "C"]}
                },
                "thetaOffset": {
                    "field": "grp",
                    "scale": {"domain": [0.0, 2.0]}
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        // 3 categories → domain (0.5, 3.5), scale = 2π/3
        // With band fraction 0.9: effective scale = 2π/3 * 0.9
        let expected = 2.0 * std::f64::consts::PI / 3.0 * POLAR_BAND_FRACTION;
        let transforms = layer["transform"].as_array().unwrap();
        let x_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_x__")
            .unwrap();
        let expr = x_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains(&format!("{expected}")),
            "offset scale should include band fraction ({expected}), got: {expr}"
        );
    }

    #[test]
    fn test_continuous_theta_offset_no_band_fraction() {
        let mut layer = json!({
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
                },
                "thetaOffset": {
                    "field": "grp",
                    "scale": {"domain": [0.0, 2.0]}
                }
            }
        });
        let panel = PolarPanel::new(None, false);

        convert_polar_to_cartesian(&mut layer, &panel).unwrap();

        // Continuous → full scale = 2π/100, no band fraction
        let full_scale = 2.0 * std::f64::consts::PI / 100.0;
        let with_band = full_scale * POLAR_BAND_FRACTION;
        let transforms = layer["transform"].as_array().unwrap();
        let x_calc = transforms
            .iter()
            .find(|t| t["as"] == "__polar_x__")
            .unwrap();
        let expr = x_calc["calculate"].as_str().unwrap();
        assert!(
            expr.contains(&format!("{full_scale}"))
                && !expr.contains(&format!("{with_band}")),
            "continuous offset should use full scale ({full_scale}), not banded ({with_band}), got: {expr}"
        );
    }
}
