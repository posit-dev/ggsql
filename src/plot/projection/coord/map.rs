//! Map coordinate system implementation

use std::collections::HashMap;

use super::{CoordKind, CoordTrait};
use crate::naming;
use crate::plot::layer::geom::GeomType;
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition, TypeConstraint};
use crate::plot::{Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

pub const CLIP_BOUNDARY_TABLE: &str = "__ggsql_clip_boundary__";

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
        use crate::plot::types::{ArrayConstraint, NumberConstraint};
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
            // [xmin, ymin, xmax, ymax] in projected coordinates; null uses data bbox, Inf uses world bbox
            ParamDefinition {
                name: "bounds",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint {
                    number: TypeConstraint::Forbidden,
                    string: TypeConstraint::Forbidden,
                    boolean: TypeConstraint::Forbidden,
                    array: TypeConstraint::Constrained(
                        ArrayConstraint::of_numbers_len(NumberConstraint::unconstrained(), 4)
                            .with_null_elements(),
                    ),
                    allow_null: true,
                },
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

        // Step 1: Detect source CRS from geometry columns if not explicitly set
        if !projection.properties.contains_key("source") {
            if let Some(srid) = detect_source_srid(layers, layer_queries, execute_query) {
                projection
                    .properties
                    .insert("source".to_string(), ParameterValue::String(srid));
            }
        }

        // Step 2: For azimuthal projections, compute the hemisphere clip boundary.
        // This produces: clip_boundary (unprojected WKT), panel_boundary (projected
        // WKT for the writer's background layer), and world_bbox (bounding box of the
        // full projected visible area, used to resolve Inf in user-specified bounds).
        let world_bbox = setup_clip_boundary(projection, dialect, execute_query)?;

        // Step 3: Apply per-layer projection (ST_Transform, clip to horizon)
        for (idx, layer) in layers.iter().enumerate() {
            layer_queries[idx] =
                layer
                    .geom
                    .apply_projection(&layer_queries[idx], projection, dialect)?;
        }

        // Step 4: Materialize projected spatial layers as temp tables, compute the
        // data bbox for framing, then convert geometry to WKB for Arrow transport.
        let geom_col = naming::aesthetic_column("geometry");
        let geom_col_quoted = naming::quote_ident(&geom_col);
        let bounds_param = projection.properties.get("bounds");
        let mut computed_bbox: Option<[f64; 4]> = None;

        for (idx, layer) in layers.iter().enumerate() {
            if layer.geom.geom_type() != GeomType::Spatial {
                continue;
            }
            let table_name = format!("{}_proj", naming::layer_key(idx));
            for stmt in
                dialect.create_or_replace_temp_table_sql(&table_name, &[], &layer_queries[idx])
            {
                execute_query(&stmt)?;
            }

            let table_quoted = naming::quote_ident(&table_name);

            if needs_computed_bbox(bounds_param) {
                let sql = dialect.sql_geometry_bbox(&geom_col_quoted, &table_quoted);
                if let Ok(df) = execute_query(&sql) {
                    computed_bbox = merge_bbox(computed_bbox, bbox_from_df(&df));
                }
            }

            let wkb_expr = dialect.sql_geometry_to_wkb(&geom_col_quoted);
            layer_queries[idx] =
                format!("SELECT * REPLACE ({wkb_expr} AS {geom_col_quoted}) FROM {table_quoted}");
        }

        // Step 5: Resolve final frame bbox from user bounds + data bounds + world bounds
        if let Some(bbox) = resolve_frame_bbox(bounds_param, computed_bbox, world_bbox) {
            use crate::plot::types::ArrayElement;
            projection.computed.insert(
                "frame_bbox".to_string(),
                ParameterValue::Array(vec![
                    ArrayElement::Number(bbox[0]),
                    ArrayElement::Number(bbox[1]),
                    ArrayElement::Number(bbox[2]),
                    ArrayElement::Number(bbox[3]),
                ]),
            );
        }

        Ok(())
    }
}

impl std::fmt::Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Set up the clip boundary for azimuthal projections. Creates the clip boundary temp table,
/// projects it into the target CRS, and returns the world bbox (projected clip boundary extent).
fn setup_clip_boundary(
    projection: &mut super::super::Projection,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<[f64; 4]>> {
    let Some(wkt) = visible_area_wkt(&projection.properties) else {
        return Ok(None);
    };

    projection.computed.insert(
        "clip_boundary".to_string(),
        ParameterValue::String(wkt.clone()),
    );
    let body = format!("SELECT ST_GeomFromText('{wkt}') AS geom");
    for stmt in dialect.create_or_replace_temp_table_sql(CLIP_BOUNDARY_TABLE, &[], &body) {
        execute_query(&stmt)?;
    }

    let source = match projection.properties.get("source") {
        Some(ParameterValue::String(s)) => s.as_str(),
        _ => "EPSG:4326",
    };
    let Some(ParameterValue::String(crs)) = projection.properties.get("crs") else {
        return Ok(None);
    };

    let projected = dialect.sql_st_transform("geom", source, crs);
    let sql = format!("SELECT ST_AsText({projected}) AS wkt FROM {CLIP_BOUNDARY_TABLE}");
    if let Ok(df) = execute_query(&sql) {
        let batch = df.inner();
        if batch.num_rows() > 0 {
            if let Some(arr) = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
            {
                let projected_wkt = arr.value(0);
                projection.computed.insert(
                    "panel_boundary".to_string(),
                    ParameterValue::String(projected_wkt.to_string()),
                );
            }
        }
    }

    let world_bbox_sql = dialect.sql_geometry_bbox(&projected, CLIP_BOUNDARY_TABLE);
    let world_bbox = execute_query(&world_bbox_sql)
        .ok()
        .and_then(|df| bbox_from_df(&df));
    Ok(world_bbox)
}

/// Returns true if we need to compute a bbox (bounding box representing the extent of geometry)
/// from the data — i.e. when bounds is absent or has null elements that need filling in.
fn needs_computed_bbox(bounds_param: Option<&ParameterValue>) -> bool {
    match bounds_param {
        Some(ParameterValue::Array(arr)) => {
            use crate::plot::types::ArrayElement;
            arr.iter().any(|e| !matches!(e, ArrayElement::Number(_)))
        }
        _ => true,
    }
}

/// Extract a [xmin, ymin, xmax, ymax] bbox from a DataFrame returned by `sql_geometry_bbox`.
fn bbox_from_df(df: &DataFrame) -> Option<[f64; 4]> {
    use arrow::array::Array;
    let batch = df.inner();
    if batch.num_rows() == 0 || batch.num_columns() < 4 {
        return None;
    }
    let get_f64 = |col: usize| -> Option<f64> {
        batch
            .column(col)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .filter(|a| !a.is_null(0))
            .map(|a| a.value(0))
    };
    match (get_f64(0), get_f64(1), get_f64(2), get_f64(3)) {
        (Some(x0), Some(y0), Some(x1), Some(y1)) => Some([x0, y0, x1, y1]),
        _ => None,
    }
}

/// Expand an existing bbox to include another, or return the new one.
fn merge_bbox(existing: Option<[f64; 4]>, new: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (existing, new) {
        (Some([x0, y0, x1, y1]), Some([nx0, ny0, nx1, ny1])) => {
            Some([x0.min(nx0), y0.min(ny0), x1.max(nx1), y1.max(ny1)])
        }
        (Some(b), None) | (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Resolve the frame bbox: merge explicit bounds with computed values.
/// - Null elements fall back to the corresponding data-computed bbox.
/// - Inf/-Inf elements fall back to the clip boundary (world) bbox.
fn resolve_frame_bbox(
    bounds_param: Option<&ParameterValue>,
    computed: Option<[f64; 4]>,
    world: Option<[f64; 4]>,
) -> Option<[f64; 4]> {
    if let Some(ParameterValue::Array(arr)) = bounds_param {
        use crate::plot::types::ArrayElement;
        let data_fallback = computed.unwrap_or([f64::NAN; 4]);
        let world_fallback = world.unwrap_or([f64::NAN; 4]);
        let resolved: Vec<f64> = arr
            .iter()
            .enumerate()
            .map(|(i, e)| match e {
                ArrayElement::Number(n) if n.is_finite() => *n,
                ArrayElement::Number(_) => world_fallback[i],
                _ => data_fallback[i],
            })
            .collect();
        // [xmin, ymin, xmax, ymax] in projected CRS units
        if resolved.len() == 4 && resolved.iter().all(|v| v.is_finite()) {
            return Some([resolved[0], resolved[1], resolved[2], resolved[3]]);
        }
    }
    computed
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
        let end = after.find([' ', '+']).unwrap_or(after.len());
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
    let mut raw_points: Vec<(f64, f64)> = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let az = (i as f64 * (360.0 / n_points as f64)).to_radians();
        let lat2 = (sin_lat0 * cos_d + cos_lat0 * sin_d * az.cos()).asin();
        let lon2 =
            lon0.to_radians() + (az.sin() * sin_d * cos_lat0).atan2(cos_d - sin_lat0 * lat2.sin());
        let mut lon_deg = lon2.to_degrees();
        // Normalize to [-180, 180]
        lon_deg = ((lon_deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        raw_points.push((lon_deg, lat2.to_degrees()));
    }

    // Insert exact antimeridian vertices where consecutive points cross ±180°.
    // Uses 179.999999 to avoid ambiguity while placing vertices at the boundary.
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_points + 4);
    for i in 0..raw_points.len() {
        points.push(raw_points[i]);
        let next = (i + 1) % raw_points.len();
        if (raw_points[next].0 - raw_points[i].0).abs() > 180.0 {
            let lat = antimeridian_crossing_lat(raw_points[i], raw_points[next]);
            if raw_points[i].0 > 0.0 {
                points.push((179.999999, lat));
                points.push((-179.999999, lat));
            } else {
                points.push((-179.999999, lat));
                points.push((179.999999, lat));
            }
        }
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

    let mut coords: Vec<String> = Vec::with_capacity(points.len() + 6);
    for (lon, lat) in &ordered {
        coords.push(format!("{lon:.6} {lat:.6}"));
    }
    coords.push(format!("{:.6} {pole_lat:.6}", last.0));
    // If the closure would jump > 180° in longitude, add an intermediate
    // vertex so no single edge crosses the antimeridian.
    if (last.0 - first.0).abs() > 180.0 {
        let mid = (last.0 + first.0) / 2.0;
        coords.push(format!("{mid:.6} {pole_lat:.6}"));
    }
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

    let (east_arc, west_arc, [east_start_lat, east_end_lat, west_start_lat, west_end_lat]) =
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
type ArcSplit = (Vec<(f64, f64)>, Vec<(f64, f64)>, [f64; 4]);

fn split_arcs_at_crossings(
    points: &[(f64, f64)],
    c1: usize,
    c2: usize,
    lat_c1: f64,
    lat_c2: f64,
) -> ArcSplit {
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
        (arc1, arc2, [lat_c1, lat_c2, lat_c2, lat_c1])
    } else {
        (arc2, arc1, [lat_c2, lat_c1, lat_c1, lat_c2])
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
        assert!(names.contains(&"bounds"));
        assert_eq!(defaults.len(), 4);
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
        // Includes north pole (52.36 + 88 > 90), pole-routing produces a POLYGON.
        assert!(
            wkt.starts_with("POLYGON(("),
            "pole case should produce POLYGON: {wkt}"
        );
    }

    #[test]
    fn test_resolve_frame_bbox_no_bounds_uses_computed() {
        let computed = Some([0.0, 0.0, 100.0, 200.0]);
        assert_eq!(resolve_frame_bbox(None, computed, None), computed);
    }

    #[test]
    fn test_resolve_frame_bbox_no_bounds_no_computed() {
        assert_eq!(resolve_frame_bbox(None, None, None), None);
    }

    #[test]
    fn test_resolve_frame_bbox_explicit_bounds_override_computed() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Number(10.0),
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        let computed = Some([0.0, 0.0, 100.0, 200.0]);
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, None),
            Some([10.0, 20.0, 30.0, 40.0])
        );
    }

    #[test]
    fn test_resolve_frame_bbox_null_elements_use_computed() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Null,
            ArrayElement::Number(20.0),
            ArrayElement::Null,
            ArrayElement::Number(40.0),
        ]);
        let computed = Some([5.0, 0.0, 95.0, 0.0]);
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, None),
            Some([5.0, 20.0, 95.0, 40.0])
        );
    }

    #[test]
    fn test_resolve_frame_bbox_inf_elements_use_world() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Number(f64::NEG_INFINITY),
            ArrayElement::Number(20.0),
            ArrayElement::Number(f64::INFINITY),
            ArrayElement::Number(40.0),
        ]);
        let computed = Some([5.0, 0.0, 95.0, 0.0]);
        let world = Some([-500.0, -500.0, 500.0, 500.0]);
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, world),
            Some([-500.0, 20.0, 500.0, 40.0])
        );
    }

    #[test]
    fn test_resolve_frame_bbox_null_without_computed_falls_through() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Null,
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        // null can't resolve without computed, so result is NaN → falls through to computed
        assert_eq!(resolve_frame_bbox(Some(&bounds), None, None), None);
    }

    #[test]
    fn test_resolve_frame_bbox_inf_without_world_falls_through() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Number(f64::INFINITY),
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        let computed = Some([5.0, 0.0, 95.0, 200.0]);
        // Inf can't resolve without world, falls through to computed
        assert_eq!(resolve_frame_bbox(Some(&bounds), computed, None), computed);
    }

    #[test]
    fn test_merge_bbox() {
        let a = Some([0.0, 10.0, 50.0, 60.0]);
        let b = Some([-5.0, 15.0, 45.0, 70.0]);
        assert_eq!(merge_bbox(a, b), Some([-5.0, 10.0, 50.0, 70.0]));
        assert_eq!(merge_bbox(a, None), a);
        assert_eq!(merge_bbox(None, a), a);
        assert_eq!(merge_bbox(None, None), None);
    }
}
