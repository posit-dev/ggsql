//! Map coordinate system implementation

use std::collections::HashMap;

use super::{CoordKind, CoordTrait};
use crate::naming;
use crate::plot::layer::geom::GeomType;
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::plot::{Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

/// Map coordinate system - for geographic/cartographic projections
#[derive(Debug, Clone, Copy)]
pub struct Map;

impl CoordTrait for Map {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Map
    }

    fn name(&self) -> &'static str {
        "map"
    }

    fn position_aesthetic_names(&self) -> &'static [&'static str] {
        &["lon", "lat"]
    }

    fn default_properties(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "crs",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string(),
            },
            ParamDefinition {
                name: "source",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string(),
            },
            ParamDefinition {
                name: "clip",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
        ];
        PARAMS
    }

    fn apply_projection_transforms(
        &self,
        layers: &[Layer],
        layer_queries: &mut [String],
        projection: &mut super::super::Projection,
        dialect: &dyn SqlDialect,
        execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
    ) -> crate::Result<()> {
        for stmt in dialect.sql_spatial_setup() {
            execute_query(&stmt)?;
        }

        // Detect source CRS from geometry columns if not explicitly set
        if !projection.properties.contains_key("source") {
            if let Some(srid) = detect_source_srid(layers, layer_queries, execute_query) {
                projection
                    .properties
                    .insert("source".to_string(), ParameterValue::String(srid));
            }
        }

        for (idx, layer) in layers.iter().enumerate() {
            layer_queries[idx] =
                layer.geom.apply_projection(&layer_queries[idx], projection, dialect)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Returns a WKT POLYGON representing the visible hemisphere for the given projection
/// properties, or `None` if the projection doesn't require horizon clipping.
///
/// The polygon is a 72-vertex haversine boundary at 88° great-circle radius from the
/// projection center (`lon_0`, `lat_0`). Azimuthal projections (orthographic, gnomonic)
/// only display one hemisphere; geometry beyond this boundary produces degenerate output
/// after `ST_Transform` and must be clipped.
pub fn visible_area_wkt(properties: &HashMap<String, ParameterValue>) -> Option<String> {
    let crs = match properties.get("crs") {
        Some(ParameterValue::String(s)) => s,
        _ => return None,
    };

    if !needs_horizon_clip(crs) {
        return None;
    }

    let center = projection_center(crs);
    Some(hemisphere_polygon_wkt(center.0, center.1, 88.0))
}

fn needs_horizon_clip(crs: &str) -> bool {
    let lower = crs.to_ascii_lowercase();
    lower.contains("+proj=ortho") || lower.contains("+proj=gnom")
}

fn projection_center(crs: &str) -> (f64, f64) {
    let lon = extract_proj_param(crs, "+lon_0=").unwrap_or(0.0);
    let lat = extract_proj_param(crs, "+lat_0=").unwrap_or(0.0);
    (lon, lat)
}

fn extract_proj_param(crs: &str, key: &str) -> Option<f64> {
    crs.find(key).and_then(|start| {
        let after = &crs[start + key.len()..];
        let end = after.find(|c: char| c == ' ' || c == '+').unwrap_or(after.len());
        after[..end].parse().ok()
    })
}

/// Haversine boundary polygon at `radius_deg` from `(lon0, lat0)`, as WKT.
/// Returns a POLYGON when the ring doesn't cross the antimeridian, or a
/// MULTIPOLYGON split at ±180° when it does.
fn hemisphere_polygon_wkt(lon0: f64, lat0: f64, radius_deg: f64) -> String {
    let d = radius_deg.to_radians();
    let lat0_r = lat0.to_radians();
    let sin_lat0 = lat0_r.sin();
    let cos_lat0 = lat0_r.cos();
    let sin_d = d.sin();
    let cos_d = d.cos();

    let n_points = 72;
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let az = (i as f64 * 5.0).to_radians();
        let lat2 = (sin_lat0 * cos_d + cos_lat0 * sin_d * az.cos()).asin();
        let lon2 =
            lon0.to_radians() + (az.sin() * sin_d * cos_lat0).atan2(cos_d - sin_lat0 * lat2.sin());
        let mut lon_deg = lon2.to_degrees();
        // Normalize to [-180, 180]
        lon_deg = ((lon_deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        points.push((lon_deg, lat2.to_degrees()));
    }

    let includes_north_pole = lat0 + radius_deg > 90.0;
    let includes_south_pole = lat0 - radius_deg < -90.0;

    if includes_north_pole || includes_south_pole {
        build_pole_polygon(&points, includes_north_pole)
    } else if find_antimeridian_crossings(&points).len() == 2 {
        build_antimeridian_multipolygon(&points)
    } else {
        build_simple_polygon(&points)
    }
}

fn build_simple_polygon(points: &[(f64, f64)]) -> String {
    let mut coords: Vec<String> = points
        .iter()
        .map(|(lon, lat)| format!("{lon:.6} {lat:.6}"))
        .collect();
    coords.push(coords[0].clone());
    format!("POLYGON(({}))", coords.join(", "))
}

/// Routes the ring through ±90° latitude to close around a pole.
fn build_pole_polygon(points: &[(f64, f64)], north: bool) -> String {
    let mut split_idx = 0;
    let mut max_jump = 0.0_f64;
    for i in 0..points.len() {
        let next = (i + 1) % points.len();
        let jump = (points[next].0 - points[i].0).abs();
        if jump > max_jump {
            max_jump = jump;
            split_idx = next;
        }
    }

    let mut ordered: Vec<(f64, f64)> = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        ordered.push(points[(split_idx + i) % points.len()]);
    }

    let pole_lat = if north { 90.0 } else { -90.0 };
    let first = ordered.first().unwrap();
    let last = ordered.last().unwrap();

    let mut coords: Vec<String> = Vec::with_capacity(points.len() + 4);
    for (lon, lat) in &ordered {
        coords.push(format!("{lon:.6} {lat:.6}"));
    }
    coords.push(format!("{:.6} {pole_lat:.6}", last.0));
    coords.push(format!("{:.6} {pole_lat:.6}", first.0));
    coords.push(format!("{:.6} {:.6}", first.0, first.1));

    format!("POLYGON(({}))", coords.join(", "))
}

fn find_antimeridian_crossings(points: &[(f64, f64)]) -> Vec<usize> {
    let n = points.len();
    let mut crossings = Vec::new();
    for i in 0..n {
        let next = (i + 1) % n;
        if (points[next].0 - points[i].0).abs() > 180.0 {
            crossings.push(i);
        }
    }
    crossings
}

/// Splits the boundary ring into two polygons at the antimeridian (±180°).
/// Each sub-polygon closes by tracing the antimeridian between its two crossing latitudes.
fn build_antimeridian_multipolygon(points: &[(f64, f64)]) -> String {
    let n = points.len();
    let crossings = find_antimeridian_crossings(points);
    assert_eq!(crossings.len(), 2);

    let c1 = crossings[0];
    let c2 = crossings[1];

    let lat_c1 = antimeridian_crossing_lat(points[c1], points[(c1 + 1) % n]);
    let lat_c2 = antimeridian_crossing_lat(points[c2], points[(c2 + 1) % n]);

    let (east_arc, west_arc, east_start_lat, east_end_lat, west_start_lat, west_end_lat) =
        split_arcs_at_crossings(points, c1, c2, lat_c1, lat_c2);

    let east_coords = build_side_ring(&east_arc, 180.0, east_start_lat, east_end_lat);
    let west_coords = build_side_ring(&west_arc, -180.0, west_start_lat, west_end_lat);

    format!(
        "MULTIPOLYGON((({})),(({})))",
        east_coords.join(", "),
        west_coords.join(", ")
    )
}

/// Split the ring at two crossing indices into east/west arcs with their boundary latitudes.
fn split_arcs_at_crossings(
    points: &[(f64, f64)],
    c1: usize,
    c2: usize,
    lat_c1: f64,
    lat_c2: f64,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, f64, f64, f64, f64) {
    let n = points.len();

    let mut arc1: Vec<(f64, f64)> = Vec::new();
    let mut i = (c1 + 1) % n;
    loop {
        arc1.push(points[i]);
        if i == c2 {
            break;
        }
        i = (i + 1) % n;
    }

    let mut arc2: Vec<(f64, f64)> = Vec::new();
    i = (c2 + 1) % n;
    loop {
        arc2.push(points[i]);
        if i == c1 {
            break;
        }
        i = (i + 1) % n;
    }

    if arc1[0].0 > 0.0 {
        (arc1, arc2, lat_c1, lat_c2, lat_c2, lat_c1)
    } else {
        (arc2, arc1, lat_c2, lat_c1, lat_c1, lat_c2)
    }
}

fn build_side_ring(
    arc: &[(f64, f64)],
    meridian_lon: f64,
    start_lat: f64,
    end_lat: f64,
) -> Vec<String> {
    let mut coords: Vec<String> = Vec::with_capacity(arc.len() + 3);
    coords.push(format!("{meridian_lon:.6} {start_lat:.6}"));
    for (lon, lat) in arc.iter() {
        coords.push(format!("{lon:.6} {lat:.6}"));
    }
    coords.push(format!("{meridian_lon:.6} {end_lat:.6}"));
    coords.push(coords[0].clone());
    coords
}

fn antimeridian_crossing_lat(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (lon_a, lat_a) = a;
    let (lon_b, lat_b) = b;
    let (lon_a_u, lon_b_u) = if lon_a > lon_b {
        (lon_a, lon_b + 360.0)
    } else {
        (lon_a + 360.0, lon_b)
    };
    let t = (180.0 - lon_a_u) / (lon_b_u - lon_a_u);
    lat_a + t * (lat_b - lat_a)
}

fn detect_source_srid(
    layers: &[Layer],
    layer_queries: &[String],
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    let geom_col = naming::quote_ident(&naming::aesthetic_column("geometry"));

    for (idx, layer) in layers.iter().enumerate() {
        if layer.geom.geom_type() != GeomType::Spatial {
            continue;
        }
        let sql = format!(
            "SELECT ST_SRID({geom_col}) AS srid FROM ({}) WHERE {geom_col} IS NOT NULL LIMIT 1",
            layer_queries[idx]
        );
        if let Ok(df) = execute_query(&sql) {
            let batch = df.inner();
            if batch.num_rows() == 0 {
                continue;
            }
            if let Some(arr) = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
            {
                let srid = arr.value(0);
                if srid != 0 {
                    return Some(format!("EPSG:{srid}"));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;
    use std::collections::HashMap;

    #[test]
    fn test_map_properties() {
        let map = Map;
        assert_eq!(map.coord_kind(), CoordKind::Map);
        assert_eq!(map.name(), "map");
        assert_eq!(map.position_aesthetic_names(), &["lon", "lat"]);
    }

    #[test]
    fn test_map_default_properties() {
        let map = Map;
        let defaults = map.default_properties();
        let names: Vec<&str> = defaults.iter().map(|p| p.name).collect();
        assert!(names.contains(&"crs"));
        assert!(names.contains(&"source"));
        assert!(names.contains(&"clip"));
        assert_eq!(defaults.len(), 3);
    }

    #[test]
    fn test_map_accepts_crs_string() {
        let map = Map;
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );

        let resolved = map.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("crs").unwrap(),
            &ParameterValue::String("+proj=merc".to_string())
        );
    }

    #[test]
    fn test_map_rejects_unknown_property() {
        let map = Map;
        let mut props = HashMap::new();
        props.insert(
            "unknown".to_string(),
            ParameterValue::String("value".to_string()),
        );

        let resolved = map.resolve_properties(&props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("not 'unknown'"));
    }

    #[test]
    fn test_visible_area_wkt_orthographic() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=45 +lon_0=10".to_string()),
        );
        let wkt = visible_area_wkt(&props);
        assert!(wkt.is_some());
        let wkt = wkt.unwrap();
        assert!(wkt.starts_with("POLYGON(("));
        assert!(wkt.ends_with("))"));
    }

    #[test]
    fn test_visible_area_wkt_gnomonic() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=gnom +lat_0=90 +lon_0=0".to_string()),
        );
        assert!(visible_area_wkt(&props).is_some());
    }

    #[test]
    fn test_visible_area_wkt_mercator_returns_none() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        assert!(visible_area_wkt(&props).is_none());
    }

    #[test]
    fn test_visible_area_wkt_no_crs_returns_none() {
        let props = HashMap::new();
        assert!(visible_area_wkt(&props).is_none());
    }

    #[test]
    fn test_visible_area_wkt_antimeridian_crossing() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=0 +lon_0=150".to_string()),
        );
        let wkt = visible_area_wkt(&props).unwrap();
        assert!(
            wkt.starts_with("MULTIPOLYGON"),
            "lon_0=150 should cross antimeridian: {wkt}"
        );
    }

    #[test]
    fn test_visible_area_wkt_no_antimeridian_for_centered() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=0 +lon_0=0".to_string()),
        );
        let wkt = visible_area_wkt(&props).unwrap();
        assert!(
            wkt.starts_with("POLYGON(("),
            "lon_0=0 should not cross antimeridian: {wkt}"
        );
    }

    #[test]
    fn test_visible_area_wkt_pole_and_antimeridian() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=52.36 +lon_0=150.90".to_string()),
        );
        let wkt = visible_area_wkt(&props).unwrap();
        // Includes north pole (52.36 + 88 > 90), ring has one big jump → pole-routing.
        assert!(
            wkt.starts_with("POLYGON(("),
            "pole case should produce POLYGON: {wkt}"
        );
    }
}
