//! Projection rendering for Vega-Lite writer
//!
//! This module provides a trait-based design for projection rendering.
//! Each projection type (cartesian, polar, and future map projections)
//! implements `ProjectionRenderer`, which owns both the VL channel mapping
//! and the spec-level transformations for that projection.

mod cartesian;
mod map;
mod polar;

use crate::plot::{CoordKind, ParameterValue, Projection, Scale};
use crate::{Plot, Result};
use serde_json::{json, Value};

use cartesian::CartesianProjection;
use map::MapProjection;
use polar::PolarProjection;

const ANGLE_TOLERANCE: f64 = 1.49011611938476e-08; // f64::EPSILON.sqrt()

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

    /// Map internal position aesthetic to Vega-Lite channel name.
    ///
    /// Returns `Some(channel_name)` for internal position aesthetics (pos1, pos2, etc.),
    /// or `None` for material aesthetics.
    fn map_position(&self, aesthetic: &str) -> Option<String> {
        let (primary, secondary) = self.position_channels();
        match aesthetic {
            "pos1" | "pos1min" => Some(primary.to_string()),
            "pos2" | "pos2min" => Some(secondary.to_string()),
            "pos1end" | "pos1max" => Some(format!("{}2", primary)),
            "pos2end" | "pos2max" => Some(format!("{}2", secondary)),
            _ => None,
        }
    }

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
    /// Called after layers are built but before faceting.
    fn transform_layers(&self, _spec: &Plot, _vl_spec: &mut Value) -> Result<()> {
        Ok(())
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
    fn apply_projection(&self, spec: &Plot, theme: &mut Value, vl_spec: &mut Value) -> Result<()> {
        self.transform_layers(spec, vl_spec)?;

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

        Ok(())
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
    facet: Option<&crate::plot::Facet>,
    scales: &[Scale],
) -> Box<dyn ProjectionRenderer> {
    match project.map(|p| p.coord.coord_kind()) {
        Some(CoordKind::Polar) => Box::new(PolarProjection::new(project, facet, scales)),
        Some(CoordKind::Map) => Box::new(MapProjection::new(project, facet)),
        Some(CoordKind::Cartesian) | None => Box::new(CartesianProjection::new(facet)),
    }
}

// =============================================================================
// AxisInfo — reusable across projection types
// =============================================================================

pub(in crate::writer) struct AxisInfo {
    pub domain: Option<(f64, f64)>,
    pub breaks: Vec<f64>,
    pub labels: Vec<(f64, String)>,
    pub suppress: bool,
}

impl AxisInfo {
    pub fn new(aesthetic: &str, scales: &[Scale], facet: Option<&crate::plot::Facet>) -> Self {
        let scale = scales.iter().find(|s| s.aesthetic == aesthetic);
        let (domain, labels) = match scale {
            Some(s) => (s.numeric_domain(), s.break_labels()),
            None => (None, Vec::new()),
        };
        let domain = domain.filter(|(min, max)| (max - min).abs() > f64::EPSILON);
        let breaks = labels.iter().map(|(v, _)| *v).collect();
        let suppress =
            facet.is_some_and(|f| f.is_free(aesthetic)) || scale.is_some_and(|s| s.is_dummy());
        Self {
            domain,
            breaks,
            labels,
            suppress,
        }
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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::Projection;

    #[test]
    fn test_map_position_cartesian() {
        let renderer = CartesianProjection::new(None);
        assert_eq!(renderer.map_position("pos1"), Some("x".to_string()));
        assert_eq!(renderer.map_position("pos2"), Some("y".to_string()));
        assert_eq!(renderer.map_position("pos1end"), Some("x2".to_string()));
        assert_eq!(renderer.map_position("pos2end"), Some("y2".to_string()));
        assert_eq!(renderer.map_position("color"), None);
        assert_eq!(renderer.offset_channels(), ("xOffset", "yOffset"));
        assert_eq!(
            renderer.panel_size(),
            Some((json!("container"), json!("container")))
        );
    }

    #[test]
    fn test_map_position_polar() {
        let renderer = PolarProjection::new(None, None, &[]);
        assert_eq!(renderer.map_position("pos1"), Some("radius".to_string()));
        assert_eq!(renderer.map_position("pos2"), Some("theta".to_string()));
        assert_eq!(
            renderer.map_position("pos1end"),
            Some("radius2".to_string())
        );
        assert_eq!(renderer.map_position("pos2end"), Some("theta2".to_string()));
        assert_eq!(renderer.offset_channels(), ("radiusOffset", "thetaOffset"));
        assert_eq!(
            renderer.panel_size(),
            Some((json!("container"), json!("container")))
        );
    }

    #[test]
    fn test_get_projection_renderer() {
        let cartesian = get_projection_renderer(None, None, &[]);
        assert_eq!(cartesian.position_channels(), ("x", "y"));

        let polar_proj = Projection::polar();
        let polar = get_projection_renderer(Some(&polar_proj), None, &[]);
        assert_eq!(polar.position_channels(), ("radius", "theta"));
    }
}
