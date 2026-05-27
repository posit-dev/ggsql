//! Map projection implementation for Vega-Lite writer
//!
//! For data that has been pre-projected server-side (via ST_Transform), Vega-Lite
//! must use an identity projection so it passes coordinates through without
//! re-projecting via d3-geo.

use crate::plot::types::ArrayElement;
use crate::plot::{ParameterValue, Projection, Scale};
use crate::{Plot, Result};
use serde_json::{json, Value};

use super::ProjectionRenderer;

/// Map projection — pre-projected spatial coordinates.
pub(in crate::writer) struct MapProjection {
    is_faceted: bool,
    panel_boundary_wkt: Option<String>,
    graticule_lon_wkt: Option<String>,
    graticule_lat_wkt: Option<String>,
    bbox: Option<[f64; 4]>,
}

impl MapProjection {
    pub(super) fn new(project: Option<&Projection>, facet: Option<&crate::plot::Facet>) -> Self {
        let get_string = |key: &str| -> Option<String> {
            project
                .and_then(|p| p.computed.get(key))
                .and_then(|v| match v {
                    ParameterValue::String(s) => Some(s.clone()),
                    _ => None,
                })
        };
        let panel_boundary_wkt = get_string("panel_boundary");
        let graticule_lon_wkt = get_string("graticule_lon");
        let graticule_lat_wkt = get_string("graticule_lat");
        let bbox = if let Some(ParameterValue::Array(arr)) =
            project.and_then(|p| p.computed.get("bbox"))
        {
            let nums: Vec<f64> = arr
                .iter()
                .filter_map(|e| match e {
                    ArrayElement::Number(n) => Some(*n),
                    _ => None,
                })
                .collect();
            nums.try_into().ok()
        } else {
            None
        };
        Self {
            is_faceted: facet.is_some_and(|f| !f.get_variables().is_empty()),
            panel_boundary_wkt,
            graticule_lon_wkt,
            graticule_lat_wkt,
            bbox,
        }
    }

    fn panel_boundary(&self, theme: &mut Value) -> Vec<Value> {
        let (fill, stroke) =
            if let Some(view) = theme.get_mut("view").and_then(|v| v.as_object_mut()) {
                let fill = view.remove("fill").unwrap_or(Value::Null);
                let stroke = view.remove("stroke").unwrap_or(Value::Null);
                view.insert("stroke".to_string(), Value::Null);
                (fill, stroke)
            } else {
                (Value::Null, Value::Null)
            };

        if let Some(ref wkt) = self.panel_boundary_wkt {
            let Some(geojson) = wkt_to_geojson(wkt) else {
                return Vec::new();
            };
            vec![geoshape_layer(
                geojson,
                json!({ "fill": fill, "stroke": stroke }),
            )]
        } else {
            vec![json!({
                "mark": {
                    "type": "rect",
                    "fill": fill,
                    "stroke": stroke,
                },
                "encoding": {
                    "x": {"value": 0},
                    "y": {"value": 0},
                    "x2": {"value": {"expr": "width"}},
                    "y2": {"value": {"expr": "height"}},
                }
            })]
        }
    }

    fn graticule(&self, theme: &Value) -> Vec<Value> {
        let grid_color = theme
            .pointer("/axis/gridColor")
            .cloned()
            .unwrap_or(json!("#cccccc"));
        let grid_width = theme
            .pointer("/axis/gridWidth")
            .cloned()
            .unwrap_or(json!(0.5));

        let mark = json!({
            "filled": false,
            "stroke": grid_color,
            "strokeWidth": grid_width,
        });

        [&self.graticule_lon_wkt, &self.graticule_lat_wkt]
            .into_iter()
            .flatten()
            .filter_map(|wkt| wkt_to_geojson(wkt))
            .map(|geojson| geoshape_layer(geojson, mark.clone()))
            .collect()
    }
}

impl ProjectionRenderer for MapProjection {
    fn is_faceted(&self) -> bool {
        self.is_faceted
    }

    fn position_channels(&self) -> (&'static str, &'static str) {
        ("longitude", "latitude")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("longitude", "latitude")
    }

    fn transform_layers(&self, _spec: &Plot, vl_spec: &mut Value) -> Result<()> {
        let mut proj = json!({
            "type": "identity",
            "reflectY": true
        });
        if let Some([xmin, ymin, xmax, ymax]) = self.bbox {
            let dx = (xmax - xmin) * 1.1;
            let dy = (ymax - ymin) * 1.1;
            if dx.is_finite() && dy.is_finite() && dx > 0.0 && dy > 0.0 {
                let cx = (xmin + xmax) / 2.0;
                let cy = (ymin + ymax) / 2.0;
                proj["scale"] = json!({"expr": format!(
                    "min(width / {dx}, height / {dy})"
                )});
                proj["translate"] = json!({"expr": format!(
                    "[width / 2 - min(width / {dx}, height / {dy}) * {cx}, \
                     height / 2 + min(width / {dx}, height / {dy}) * {cy}]"
                )});
            }
        }
        vl_spec["projection"] = proj;
        Ok(())
    }

    fn background_layers(&self, _scales: &[Scale], theme: &mut Value) -> Vec<Value> {
        let mut layers = Vec::new();
        layers.extend(self.panel_boundary(theme));
        layers.extend(self.graticule(theme));
        layers
    }
}

fn geoshape_layer(geojson: Value, mut mark: Value) -> Value {
    mark["type"] = json!("geoshape");
    json!({
        "data": {"values": [{"type": "Feature", "geometry": geojson}]},
        "mark": mark
    })
}

#[cfg(feature = "spatial")]
fn wkt_to_geojson(wkt: &str) -> Option<Value> {
    use geozero::geojson::GeoJsonWriter;
    use geozero::wkt::WktReader;
    use geozero::GeozeroDatasource;
    use std::io::Cursor;

    let mut reader = WktReader(wkt.as_bytes());
    let mut geojson_out = Vec::new();
    reader
        .process_geom(&mut GeoJsonWriter::new(Cursor::new(&mut geojson_out)))
        .ok()?;
    serde_json::from_slice(&geojson_out).ok()
}

#[cfg(not(feature = "spatial"))]
fn wkt_to_geojson(_wkt: &str) -> Option<Value> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{Facet, FacetLayout, Projection};

    #[test]
    fn test_map_projection_identity() {
        let renderer = MapProjection::new(None, None);
        let mut vl_spec = json!({"layer": []});
        let spec = Plot::default();

        renderer.transform_layers(&spec, &mut vl_spec).unwrap();

        assert_eq!(vl_spec["projection"]["type"], "identity");
        assert_eq!(vl_spec["projection"]["reflectY"], true);
    }

    #[test]
    fn test_map_projection_channels() {
        let renderer = MapProjection::new(None, None);
        assert_eq!(renderer.position_channels(), ("longitude", "latitude"));
        assert_eq!(renderer.offset_channels(), ("longitude", "latitude"));
        assert_eq!(renderer.map_position("pos1"), Some("longitude".to_string()));
        assert_eq!(renderer.map_position("pos2"), Some("latitude".to_string()));
    }

    #[test]
    fn test_map_projection_faceted() {
        let facet = Facet::new(FacetLayout::Wrap {
            variables: vec!["region".to_string()],
        });
        let renderer = MapProjection::new(None, Some(&facet));
        assert!(renderer.is_faceted());
        assert_eq!(renderer.panel_size(), None);
    }

    #[test]
    fn test_map_projection_not_faceted() {
        let renderer = MapProjection::new(None, None);
        assert!(!renderer.is_faceted());
        assert_eq!(
            renderer.panel_size(),
            Some((json!("container"), json!("container")))
        );
    }

    #[test]
    fn test_background_layer_without_boundary() {
        let renderer = MapProjection::new(None, None);
        let mut theme = json!({"view": {"fill": "white", "stroke": "gray"}});
        let layers = renderer.background_layers(&[], &mut theme);
        // Fallback: rect mark spanning the full view
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0]["mark"]["type"], "rect");
        assert_eq!(layers[0]["mark"]["fill"], "white");
        assert_eq!(layers[0]["mark"]["stroke"], "gray");
    }

    #[test]
    fn test_background_layer_with_boundary() {
        let mut proj = Projection::map();
        proj.computed.insert(
            "panel_boundary".to_string(),
            ParameterValue::String("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))".to_string()),
        );
        let renderer = MapProjection::new(Some(&proj), None);
        let mut theme = json!({"view": {"fill": "white", "stroke": "gray"}});
        let layers = renderer.background_layers(&[], &mut theme);

        assert_eq!(layers.len(), 1);
        let layer = &layers[0];
        assert_eq!(layer["mark"]["type"], "geoshape");
        assert_eq!(layer["mark"]["fill"], "white");
        assert_eq!(layer["mark"]["stroke"], "gray");

        let geom = &layer["data"]["values"][0]["geometry"];
        assert_eq!(geom["type"], "Polygon");
        assert!(!geom["coordinates"].is_null());

        // view stroke should be nulled out
        assert_eq!(theme["view"]["stroke"], Value::Null);
    }

    #[test]
    fn test_bbox_emits_scale_translate_exprs() {
        use crate::plot::types::ArrayElement;

        let mut proj = Projection::map();
        proj.computed.insert(
            "bbox".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(0.0),
                ArrayElement::Number(100.0),
                ArrayElement::Number(200.0),
            ]),
        );
        let renderer = MapProjection::new(Some(&proj), None);
        let mut vl_spec = json!({"layer": []});
        let spec = Plot::default();

        renderer.transform_layers(&spec, &mut vl_spec).unwrap();

        let scale = &vl_spec["projection"]["scale"]["expr"];
        let translate = &vl_spec["projection"]["translate"]["expr"];
        assert!(scale.is_string(), "scale should be an expr");
        assert!(translate.is_string(), "translate should be an expr");
        assert!(scale.as_str().unwrap().contains("width"));
        assert!(translate.as_str().unwrap().contains("height"));
    }

    #[test]
    fn test_graticule_layers_rendered() {
        let mut proj = Projection::map();
        proj.computed.insert(
            "graticule_lon".to_string(),
            ParameterValue::String("MULTILINESTRING ((0 -90, 0 90), (30 -90, 30 90))".to_string()),
        );
        proj.computed.insert(
            "graticule_lat".to_string(),
            ParameterValue::String(
                "MULTILINESTRING ((-180 0, 180 0), (-180 45, 180 45))".to_string(),
            ),
        );
        let renderer = MapProjection::new(Some(&proj), None);
        let mut theme = json!({"axis": {"gridColor": "#dddddd", "gridWidth": 1}});
        let layers = renderer.background_layers(&[], &mut theme);

        // 1 rect fallback + 2 graticule layers
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0]["mark"]["type"], "rect");
        for layer in &layers[1..] {
            assert_eq!(layer["mark"]["type"], "geoshape");
            assert_eq!(layer["mark"]["filled"], false);
            assert_eq!(layer["mark"]["stroke"], "#dddddd");
            assert_eq!(layer["mark"]["strokeWidth"], 1);
            let geom = &layer["data"]["values"][0]["geometry"];
            assert_eq!(geom["type"], "MultiLineString");
        }
    }

    #[test]
    fn test_graticule_with_panel_boundary() {
        let mut proj = Projection::map();
        proj.computed.insert(
            "panel_boundary".to_string(),
            ParameterValue::String("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))".to_string()),
        );
        proj.computed.insert(
            "graticule_lon".to_string(),
            ParameterValue::String("MULTILINESTRING ((0.5 0, 0.5 1))".to_string()),
        );
        let renderer = MapProjection::new(Some(&proj), None);
        let mut theme = json!({"view": {"fill": "white", "stroke": "gray"}});
        let layers = renderer.background_layers(&[], &mut theme);

        // Panel boundary first, then graticule
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0]["mark"]["fill"], "white");
        assert_eq!(layers[1]["mark"]["filled"], false);
    }
}
