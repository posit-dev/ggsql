//! Map projection implementation for Vega-Lite writer
//!
//! For data that has been pre-projected server-side (via ST_Transform), Vega-Lite
//! must use an identity projection so it passes coordinates through without
//! re-projecting via d3-geo.

use crate::{Plot, Result};
use serde_json::{json, Value};

use super::ProjectionRenderer;

/// Map projection — pre-projected spatial coordinates.
pub(in crate::writer) struct MapProjection {
    is_faceted: bool,
}

impl MapProjection {
    pub(super) fn new(facet: Option<&crate::plot::Facet>) -> Self {
        Self {
            is_faceted: facet.is_some_and(|f| !f.get_variables().is_empty()),
        }
    }
}

impl ProjectionRenderer for MapProjection {
    fn is_faceted(&self) -> bool {
        self.is_faceted
    }

    fn position_channels(&self) -> (&'static str, &'static str) {
        ("x", "y")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("xOffset", "yOffset")
    }

    fn transform_layers(&self, _spec: &Plot, vl_spec: &mut Value) -> Result<()> {
        vl_spec["projection"] = json!({
            "type": "identity",
            "reflectY": true
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{Facet, FacetLayout};

    #[test]
    fn test_map_projection_identity() {
        let renderer = MapProjection::new(None);
        let mut vl_spec = json!({"layer": []});
        let spec = Plot::default();

        renderer.transform_layers(&spec, &mut vl_spec).unwrap();

        assert_eq!(vl_spec["projection"]["type"], "identity");
        assert_eq!(vl_spec["projection"]["reflectY"], true);
    }

    #[test]
    fn test_map_projection_channels() {
        let renderer = MapProjection::new(None);
        assert_eq!(renderer.position_channels(), ("x", "y"));
        assert_eq!(renderer.offset_channels(), ("xOffset", "yOffset"));
        assert_eq!(renderer.map_position("pos1"), Some("x".to_string()));
        assert_eq!(renderer.map_position("pos2"), Some("y".to_string()));
    }

    #[test]
    fn test_map_projection_faceted() {
        let facet = Facet::new(FacetLayout::Wrap {
            variables: vec!["region".to_string()],
        });
        let renderer = MapProjection::new(Some(&facet));
        assert!(renderer.is_faceted());
        assert_eq!(renderer.panel_size(), None);
    }

    #[test]
    fn test_map_projection_not_faceted() {
        let renderer = MapProjection::new(None);
        assert!(!renderer.is_faceted());
        assert_eq!(
            renderer.panel_size(),
            Some((json!("container"), json!("container")))
        );
    }
}
