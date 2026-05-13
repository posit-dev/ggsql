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

// ---------------------------------------------------------------------------
// Map coord
// ---------------------------------------------------------------------------

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

        let source = match projection.properties.get("source") {
            Some(ParameterValue::String(s)) => s.clone(),
            _ => "EPSG:4326".to_string(),
        };
        let crs = match projection.properties.get("crs") {
            Some(ParameterValue::String(s)) => s.clone(),
            _ => return Ok(()),
        };

        // Step 2: Compute the visible area boundary for this projection.
        // Azimuthal projections get a hemisphere polygon; cylindrical get a world
        // rectangle. This produces: clip_boundary (unprojected WKT), panel_boundary
        // (projected WKT for the writer's background layer), and world_bbox (bounding
        // box of the full projected visible area, used to resolve Inf in user bounds).
        let world_bbox = setup_clip_boundary(projection, &source, &crs, dialect, execute_query)?;

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
        let mut computed_bbox: Option<BBox> = None;

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
                    computed_bbox = BBox::merge(computed_bbox, BBox::from_df(&df, &crs))?;
                }
            }

            let wkb_expr = dialect.sql_geometry_to_wkb(&geom_col_quoted);
            layer_queries[idx] =
                format!("SELECT * REPLACE ({wkb_expr} AS {geom_col_quoted}) FROM {table_quoted}");
        }

        // Step 5: Resolve final frame bbox from user bounds + data bounds + world bounds
        let Some(bbox) = resolve_frame_bbox(bounds_param, computed_bbox, world_bbox) else {
            return Ok(());
        };
        projection
            .computed
            .insert("frame_bbox".to_string(), bbox.as_parameter_value());

        // Step 6: Generate graticule lines (azimuthal and interrupted projections
        // need clip-based extent and ST_Intersection; cylindrical projections don't)
        let proj_name = extract_proj_param_str(&crs, "+proj=");
        let needs_clip = matches!(
            proj_name,
            Some("ortho") | Some("gnom") | Some("stere") | Some("igh")
        );
        let (lon_wkt, lat_wkt) =
            build_graticule(&bbox, &source, needs_clip, dialect, execute_query)?;
        if let Some(wkt) = lon_wkt {
            projection
                .computed
                .insert("graticule_lon".to_string(), ParameterValue::String(wkt));
        }
        if let Some(wkt) = lat_wkt {
            projection
                .computed
                .insert("graticule_lat".to_string(), ParameterValue::String(wkt));
        }

        Ok(())
    }
}

impl std::fmt::Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// BBox
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct BBox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    crs: String,
}

impl BBox {
    fn from_df(df: &DataFrame, crs: &str) -> Option<Self> {
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
            (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) => Some(Self {
                xmin,
                ymin,
                xmax,
                ymax,
                crs: crs.to_string(),
            }),
            _ => None,
        }
    }

    fn merge(existing: Option<Self>, new: Option<Self>) -> crate::Result<Option<Self>> {
        match (existing, new) {
            (Some(a), Some(b)) => {
                if a.crs != b.crs {
                    return Err(crate::GgsqlError::InternalError(format!(
                        "Cannot merge bounding boxes with different CRS: '{}' vs '{}'",
                        a.crs, b.crs
                    )));
                }
                Ok(Some(Self {
                    xmin: a.xmin.min(b.xmin),
                    ymin: a.ymin.min(b.ymin),
                    xmax: a.xmax.max(b.xmax),
                    ymax: a.ymax.max(b.ymax),
                    crs: a.crs,
                }))
            }
            (Some(b), None) | (None, Some(b)) => Ok(Some(b)),
            (None, None) => Ok(None),
        }
    }

    fn from_array(arr: [f64; 4], crs: &str) -> Self {
        Self {
            xmin: arr[0].min(arr[2]),
            ymin: arr[1].min(arr[3]),
            xmax: arr[0].max(arr[2]),
            ymax: arr[1].max(arr[3]),
            crs: crs.to_string(),
        }
    }

    fn to_array(&self) -> [f64; 4] {
        [self.xmin, self.ymin, self.xmax, self.ymax]
    }

    fn clamp(mut self, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Self {
        self.xmin = self.xmin.clamp(xmin, xmax);
        self.ymin = self.ymin.clamp(ymin, ymax);
        self.xmax = self.xmax.clamp(xmin, xmax);
        self.ymax = self.ymax.clamp(ymin, ymax);
        self
    }

    fn xrange(&self) -> (f64, f64) {
        (self.xmin, self.xmax)
    }

    fn yrange(&self) -> (f64, f64) {
        (self.ymin, self.ymax)
    }

    fn to_polygon_wkt(&self) -> String {
        let (xmin, ymin, xmax, ymax) = (self.xmin, self.ymin, self.xmax, self.ymax);
        format!(
            "POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"
        )
    }

    fn as_parameter_value(&self) -> ParameterValue {
        use crate::plot::types::ArrayElement;
        ParameterValue::Array(vec![
            ArrayElement::Number(self.xmin),
            ArrayElement::Number(self.ymin),
            ArrayElement::Number(self.xmax),
            ArrayElement::Number(self.ymax),
        ])
    }

    fn reproject(
        &self,
        target_crs: &str,
        dialect: &dyn SqlDialect,
        execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
    ) -> Option<Self> {
        let envelope = format!(
            "ST_MakeEnvelope({}, {}, {}, {})",
            self.xmin, self.ymin, self.xmax, self.ymax
        );
        let transformed = dialect.sql_st_transform(&envelope, &self.crs, target_crs);
        let sql = format!(
            "SELECT ST_XMin(g) AS xmin, ST_YMin(g) AS ymin, \
                    ST_XMax(g) AS xmax, ST_YMax(g) AS ymax \
             FROM (SELECT {transformed} AS g)"
        );
        execute_query(&sql)
            .ok()
            .and_then(|df| Self::from_df(&df, target_crs))
    }
}

// ---------------------------------------------------------------------------
// Graticule helpers
// ---------------------------------------------------------------------------

/// Build graticule lines: determine the visible lon/lat extent, generate densified
/// meridians and parallels, clip and project them, and return projected WKT.
fn build_graticule(
    frame_bbox: &BBox,
    source: &str,
    has_clip: bool,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<(Option<String>, Option<String>)> {
    let crs = &frame_bbox.crs;
    let Some(geo_bbox) = graticule_bbox(frame_bbox, source, has_clip, dialect, execute_query)?
    else {
        return Ok((None, None));
    };

    let lon_breaks = graticule_breaks(geo_bbox.xrange());
    let lat_breaks = graticule_breaks(geo_bbox.yrange());

    if lon_breaks.is_empty() && lat_breaks.is_empty() {
        return Ok((None, None));
    }

    // Densification interval based on angular extent
    let max_range = (geo_bbox.xmax - geo_bbox.xmin).max(geo_bbox.ymax - geo_bbox.ymin);
    let step_deg = if max_range > 90.0 {
        2.0
    } else if max_range > 30.0 {
        1.0
    } else {
        0.5
    };

    // Clamp meridians away from ±180 to avoid antimeridian issues, and
    // deduplicate (e.g. if both -180 and 180 were present, they become the same)
    let lon_breaks: Vec<f64> = {
        let mut clamped: Vec<f64> = lon_breaks
            .iter()
            .map(|&v| {
                if v <= -180.0 {
                    -179.999999
                } else if v >= 180.0 {
                    179.999999
                } else {
                    v
                }
            })
            .collect();
        clamped.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        clamped
    };

    let lon_wkt = if !lon_breaks.is_empty() {
        Some(grid_lines_wkt(
            &lon_breaks,
            geo_bbox.yrange(),
            step_deg,
            true,
        ))
    } else {
        None
    };
    let lat_wkt = if !lat_breaks.is_empty() {
        Some(grid_lines_wkt(
            &lat_breaks,
            geo_bbox.xrange(),
            step_deg,
            false,
        ))
    } else {
        None
    };

    Ok((
        project_wkt(lon_wkt, has_clip, source, crs, dialect, execute_query)?,
        project_wkt(lat_wkt, has_clip, source, crs, dialect, execute_query)?,
    ))
}

/// Determine the lon/lat bounding box visible in the current frame by inverse-projecting
/// the bbox corners. Falls back to the clip boundary for azimuthal projections
/// where corners collapse to degenerate values.
fn graticule_bbox(
    frame_bbox: &BBox,
    source: &str,
    has_clip: bool,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<BBox>> {
    let mut geo_bbox = match frame_bbox.reproject(source, dialect, execute_query) {
        Some(b) => b.clamp(-180.0, -90.0, 180.0, 90.0),
        None => return Ok(None),
    };

    // For azimuthal projections the bbox corners often inverse-project to
    // degenerate or incomplete values. Use the clip boundary extent which
    // correctly represents the visible hemisphere.
    if has_clip {
        let bbox_sql = dialect.sql_geometry_bbox("geom", CLIP_BOUNDARY_TABLE);
        if let Ok(df) = execute_query(&bbox_sql) {
            if let Some(clip_bbox) = BBox::from_df(&df, source) {
                geo_bbox = clip_bbox;
            }
        }
    }

    // For projections showing the full globe, expand to full range
    if geo_bbox.xmax - geo_bbox.xmin > 300.0 {
        geo_bbox.xmin = -180.0;
        geo_bbox.xmax = 180.0;
    }
    if geo_bbox.ymax - geo_bbox.ymin > 150.0 {
        geo_bbox.ymin = -90.0;
        geo_bbox.ymax = 90.0;
    }

    Ok(Some(geo_bbox))
}

/// Pick pretty graticule break positions for a lon or lat range.
/// Uses standard angular intervals (multiples of 1, 2, 5, 10, 15, 30, 45, 90).
fn graticule_breaks((min, max): (f64, f64)) -> Vec<f64> {
    let range = max - min;
    if range <= 0.0 {
        return vec![];
    }

    const STEPS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0];

    // Pick the smallest step that gives at most ~7 lines
    let step = STEPS
        .iter()
        .copied()
        .find(|&s| range / s <= 8.0)
        .unwrap_or(90.0);

    let start = (min / step).ceil() as i64;
    let end = (max / step).floor() as i64;
    let mut breaks: Vec<f64> = (start..=end)
        .map(|i| i as f64 * step)
        .filter(|&v| v > min && v < max)
        .collect();

    // Include the boundary value when range covers the full extent,
    // so the antimeridian/pole gets a line
    if min <= -180.0 && !breaks.contains(&-180.0) {
        breaks.insert(0, -180.0);
    } else if max >= 180.0 && !breaks.contains(&180.0) {
        breaks.push(180.0);
    }
    if min <= -90.0 && !breaks.contains(&-90.0) {
        breaks.insert(0, -90.0);
    } else if max >= 90.0 && !breaks.contains(&90.0) {
        breaks.push(90.0);
    }

    breaks
}

/// Generate a MULTILINESTRING WKT with one line per break value, densified along
/// the varying axis at `step_deg` intervals.
/// - `lon_first = true`: fixed longitude (meridians), varying latitude.
/// - `lon_first = false`: fixed latitude (parallels), varying longitude.
fn grid_lines_wkt(
    breaks: &[f64],
    (vary_min, vary_max): (f64, f64),
    step_deg: f64,
    lon_first: bool,
) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(breaks.len());
    for &fixed in breaks {
        let mut coords = Vec::new();
        let mut v = vary_min;
        while v < vary_max {
            let (lon, lat) = if lon_first { (fixed, v) } else { (v, fixed) };
            coords.push(format!("{lon:.6} {lat:.6}"));
            v += step_deg;
        }
        let (lon, lat) = if lon_first {
            (fixed, vary_max)
        } else {
            (vary_max, fixed)
        };
        coords.push(format!("{lon:.6} {lat:.6}"));
        lines.push(format!("({})", coords.join(", ")));
    }
    format!("MULTILINESTRING({})", lines.join(", "))
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

/// Execute a query and extract a single string value from the first row, first column.
fn query_scalar_string(
    sql: &str,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    use arrow::array::Array;
    let df = execute_query(sql).ok()?;
    let batch = df.inner();
    if batch.num_rows() == 0 {
        return None;
    }
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()?;
    if arr.is_null(0) {
        return None;
    }
    Some(arr.value(0).to_string())
}

/// Set up the clip/visible area boundary. Creates the clip boundary temp table,
/// projects it into the target CRS, and returns the world bbox (projected extent).
/// For projections where `visible_area_wkt` returns None (e.g. LAEA), generates the
/// panel boundary directly in projected space using ST_Buffer.
fn setup_clip_boundary(
    projection: &mut super::super::Projection,
    source: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<BBox>> {
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

    let projected = dialect.sql_st_transform("geom", source, crs);
    let sql = format!("SELECT ST_AsText({projected}) AS wkt FROM {CLIP_BOUNDARY_TABLE}");
    if let Some(projected_wkt) = query_scalar_string(&sql, execute_query) {
        projection.computed.insert(
            "panel_boundary".to_string(),
            ParameterValue::String(projected_wkt),
        );
    }

    let world_bbox_sql = dialect.sql_geometry_bbox(&projected, CLIP_BOUNDARY_TABLE);
    let world_bbox = execute_query(&world_bbox_sql)
        .ok()
        .and_then(|df| BBox::from_df(&df, crs));
    Ok(world_bbox)
}

/// Clip (if needed) and project a WKT geometry string, returning the projected WKT.
fn project_wkt(
    wkt: Option<String>,
    has_clip: bool,
    source: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<String>> {
    let Some(wkt) = wkt else { return Ok(None) };
    let geom_expr = format!("ST_GeomFromText('{wkt}')");
    let clipped = if has_clip {
        // ST_CollectionExtract(..., 2) keeps only linestring components,
        // discarding stray points from vertex-on-boundary intersections.
        format!(
            "ST_CollectionExtract(ST_Intersection({geom_expr}, \
             (SELECT geom FROM {CLIP_BOUNDARY_TABLE})), 2)"
        )
    } else {
        geom_expr
    };
    let projected = dialect.sql_st_transform(&clipped, source, crs);
    let sql = format!("SELECT ST_AsText({projected}) AS wkt");
    Ok(query_scalar_string(&sql, execute_query))
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

/// Resolve the frame bbox: merge explicit bounds with computed values.
/// - Null elements fall back to the corresponding data-computed bbox.
/// - Inf/-Inf elements fall back to the clip boundary (world) bbox.
fn resolve_frame_bbox(
    bounds_param: Option<&ParameterValue>,
    computed: Option<BBox>,
    world: Option<BBox>,
) -> Option<BBox> {
    if let Some(ParameterValue::Array(arr)) = bounds_param {
        use crate::plot::types::ArrayElement;
        let data_fallback = computed.as_ref().map_or([f64::NAN; 4], |b| b.to_array());
        let world_fallback = world.as_ref().map_or([f64::NAN; 4], |b| b.to_array());
        let crs = computed
            .as_ref()
            .or(world.as_ref())
            .map(|b| b.crs.clone())
            .unwrap_or_default();
        let resolved: Vec<f64> = arr
            .iter()
            .enumerate()
            .map(|(i, e)| match e {
                ArrayElement::Number(n) if n.is_finite() => *n,
                ArrayElement::Number(_) => world_fallback[i],
                _ => data_fallback[i],
            })
            .collect();
        if resolved.len() == 4 && resolved.iter().all(|v| v.is_finite()) {
            return Some(BBox::from_array(
                [resolved[0], resolved[1], resolved[2], resolved[3]],
                &crs,
            ));
        }
    }
    computed
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

// ---------------------------------------------------------------------------
// Visible area / horizon clipping
// ---------------------------------------------------------------------------

/// Returns a WKT POLYGON representing the valid visible area for the given projection.
///
/// - Azimuthal projections (orthographic, gnomonic): a 72-vertex haversine boundary at
///   88° great-circle radius from the projection center. Geometry beyond this boundary
///   produces degenerate output after `ST_Transform`.
/// - Cylindrical projections (mercator): a rectangle at ±180° longitude, ±85° latitude
///   (the Mercator singularity is at ±85.05°).
/// - Returns `None` if no CRS is set (no projection to apply).
pub fn visible_area_wkt(properties: &HashMap<String, ParameterValue>) -> Option<String> {
    let crs = match properties.get("crs") {
        Some(ParameterValue::String(s)) => s,
        _ => return None,
    };

    let center = projection_center(crs);
    match extract_proj_param_str(crs, "+proj=") {
        Some("ortho") | Some("gnom") | Some("stere") => {
            Some(hemisphere_polygon_wkt(center.0, center.1, 88.0))
        }
        Some("laea") | Some("aeqd") => todo!("full-globe azimuthal visible area"),
        Some("igh") => Some(igh_outline_wkt()),
        Some("merc") => Some(BBox::from_array([-180.0, -85.0, 180.0, 85.0], "").to_polygon_wkt()),
        Some("mill") | Some("eqc") => {
            Some(BBox::from_array([-180.0, -90.0, 180.0, 90.0], "").to_polygon_wkt())
        }
        _ => None,
    }
}

fn projection_center(crs: &str) -> (f64, f64) {
    let lon = extract_proj_param_str(crs, "+lon_0=")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let lat = extract_proj_param_str(crs, "+lat_0=")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    (lon, lat)
}

/// Interrupted Goode Homolosine outline polygon with densified meridian edges.
/// Interrupts: -40° (north), -100°/-20°/80° (south). The outline traces
/// vertical slits at these meridians with 1° spacing for smooth projection.
fn igh_outline_wkt() -> String {
    let mut coords: Vec<String> = Vec::new();

    // Helper: densified meridian segment from lat_start to lat_end at fixed lon
    let meridian = |coords: &mut Vec<String>, lon: f64, lat_start: f64, lat_end: f64| {
        let step = if lat_end > lat_start { 5.0 } else { -5.0 };
        let n = ((lat_end - lat_start) / step).abs() as usize;
        for i in 0..n {
            let lat = lat_start + step * i as f64;
            coords.push(format!("{lon:.2} {lat:.2}"));
        }
    };

    // Counter-clockwise ring matching the R/sf approach:
    // Start top-right (180,90), down east edge, trace bottom east→west with
    // southern slits, up west edge, trace top west→east with northern slit.

    // East edge: (180, 90) down to (180, -90)
    meridian(&mut coords, 180.0, 90.0, -90.0);

    // Bottom edge east→west with southern slits at 80°, -20°, -100°
    coords.push("80.01 -90".to_string());
    meridian(&mut coords, 80.01, -90.0, 0.0);
    meridian(&mut coords, 79.99, 0.0, -90.0);
    coords.push("-19.99 -90".to_string());
    meridian(&mut coords, -19.99, -90.0, 0.0);
    meridian(&mut coords, -20.01, 0.0, -90.0);
    coords.push("-99.99 -90".to_string());
    meridian(&mut coords, -99.99, -90.0, 0.0);
    meridian(&mut coords, -100.01, 0.0, -90.0);
    coords.push("-180 -90".to_string());

    // West edge: (-180, -90) up to (-180, 90)
    meridian(&mut coords, -180.0, -90.0, 90.0);

    // Top edge west→east with northern slit at -40°
    coords.push("-40.01 90".to_string());
    meridian(&mut coords, -40.01, 90.0, 0.0);
    meridian(&mut coords, -39.99, 0.0, 90.0);

    // Close ring
    coords.push("180 90".to_string());

    format!("POLYGON(({}))", coords.join(", "))
}

fn extract_proj_param_str<'a>(crs: &'a str, key: &str) -> Option<&'a str> {
    let start = crs.find(key)?;
    let after = &crs[start + key.len()..];
    let end = after.find([' ', '+']).unwrap_or(after.len());
    Some(&after[..end])
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_visible_area_wkt_mercator_returns_rectangle() {
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        let wkt = visible_area_wkt(&props);
        assert!(wkt.is_some());
        let wkt = wkt.unwrap();
        assert!(wkt.starts_with("POLYGON(("));
        assert!(wkt.contains("-180") && wkt.contains("180"));
        assert!(wkt.contains("-85") && wkt.contains("85"));
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

    fn bbox(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> BBox {
        BBox::from_array([xmin, ymin, xmax, ymax], "EPSG:4326")
    }

    #[test]
    fn test_resolve_frame_bbox_no_bounds_uses_computed() {
        let computed = Some(bbox(0.0, 0.0, 100.0, 200.0));
        assert_eq!(resolve_frame_bbox(None, computed.clone(), None), computed);
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
        let computed = Some(bbox(0.0, 0.0, 100.0, 200.0));
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, None),
            Some(bbox(10.0, 20.0, 30.0, 40.0))
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
        let computed = Some(bbox(5.0, 0.0, 95.0, 0.0));
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, None),
            Some(bbox(5.0, 20.0, 95.0, 40.0))
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
        let computed = Some(bbox(5.0, 0.0, 95.0, 0.0));
        let world = Some(bbox(-500.0, -500.0, 500.0, 500.0));
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed, world),
            Some(bbox(-500.0, 20.0, 500.0, 40.0))
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
        let computed = Some(bbox(5.0, 0.0, 95.0, 200.0));
        assert_eq!(
            resolve_frame_bbox(Some(&bounds), computed.clone(), None),
            computed
        );
    }

    #[test]
    fn test_merge_bbox() {
        let a = Some(bbox(0.0, 10.0, 50.0, 60.0));
        let b = Some(bbox(-5.0, 15.0, 45.0, 70.0));
        assert_eq!(
            BBox::merge(a.clone(), b).unwrap(),
            Some(bbox(-5.0, 10.0, 50.0, 70.0))
        );
        assert_eq!(BBox::merge(a.clone(), None).unwrap(), a);
        assert_eq!(BBox::merge(None, a.clone()).unwrap(), a);
        assert_eq!(BBox::merge(None, None).unwrap(), None);
    }

    #[test]
    fn test_merge_bbox_crs_mismatch() {
        let a = Some(BBox::from_array([0.0, 0.0, 1.0, 1.0], "EPSG:4326"));
        let b = Some(BBox::from_array([0.0, 0.0, 1.0, 1.0], "EPSG:3857"));
        assert!(BBox::merge(a, b).is_err());
    }

    #[test]
    fn test_clamp() {
        // restricts values that exceed bounds
        let b = BBox::from_array([-200.0, -100.0, 200.0, 100.0], "EPSG:4326");
        assert_eq!(
            b.clamp(-180.0, -90.0, 180.0, 90.0),
            bbox(-180.0, -90.0, 180.0, 90.0)
        );

        // no-op when already within bounds
        let b = bbox(10.0, 20.0, 30.0, 40.0);
        assert_eq!(
            b.clamp(-180.0, -90.0, 180.0, 90.0),
            bbox(10.0, 20.0, 30.0, 40.0)
        );
    }

    #[test]
    fn test_graticule_breaks_world() {
        let breaks = graticule_breaks((-180.0, 180.0));
        assert_eq!(
            breaks,
            vec![-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]
        );
    }

    #[test]
    fn test_graticule_breaks_hemisphere() {
        let breaks = graticule_breaks((-88.0, 88.0));
        assert_eq!(breaks, vec![-60.0, -30.0, 0.0, 30.0, 60.0]);
    }

    #[test]
    fn test_graticule_breaks_small_range() {
        let breaks = graticule_breaks((10.0, 20.0));
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b > 10.0 && b < 20.0);
        }
    }

    #[test]
    fn test_graticule_breaks_empty_for_zero_range() {
        let breaks = graticule_breaks((50.0, 50.0));
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_grid_lines_wkt_meridians() {
        let wkt = grid_lines_wkt(&[0.0, 30.0], (-90.0, 90.0), 45.0, true);
        assert!(wkt.starts_with("MULTILINESTRING("), "{wkt}");
        assert!(wkt.contains("0.000000 -90.000000"), "{wkt}");
        assert!(wkt.contains("30.000000 -90.000000"), "{wkt}");
        assert!(wkt.contains("0.000000 90.000000"), "{wkt}");
        assert!(wkt.contains("30.000000 90.000000"), "{wkt}");
    }

    #[test]
    fn test_grid_lines_wkt_parallels() {
        let wkt = grid_lines_wkt(&[0.0, 45.0], (-180.0, 180.0), 90.0, false);
        assert!(wkt.starts_with("MULTILINESTRING("));
        assert!(wkt.contains("0.000000"));
        assert!(wkt.contains("45.000000"));
    }
}
