//! Map coordinate system — helper functions for map projection transforms.

use std::collections::HashMap;

use super::map_projections::MapProjectionTrait;
use crate::naming;
use crate::plot::layer::geom::GeomType;
use crate::plot::scale::breaks::graticule_breaks;
use crate::plot::{DataSource, Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

// ---------------------------------------------------------------------------
// Entry point called from the blanket CoordTrait impl
// ---------------------------------------------------------------------------

pub(crate) fn apply_map_transforms(
    map_proj: &dyn MapProjectionTrait,
    layers: &[Layer],
    layer_queries: &mut [String],
    projection: &mut super::super::Projection,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<()> {
    for stmt in dialect.sql_spatial_setup() {
        execute_query(&stmt)?;
    }

    // Read resolved CRS from properties (set by resolve_map_projection)
    let Some(ParameterValue::String(source)) = projection.properties.get("source") else {
        unreachable!("source must be resolved before apply_map_transforms");
    };
    let Some(ParameterValue::String(target)) = projection.properties.get("target") else {
        unreachable!("target must be resolved before apply_map_transforms");
    };
    let source = source.clone();
    let target = target.clone();

    // Step 2: Materialize clip boundary, panel boundary, and world bbox.
    let clip_enabled = match projection.properties.get("clip") {
        Some(ParameterValue::Boolean(b)) => *b,
        _ => true,
    };
    let mut world_bbox: Option<BBox> = None;
    let mut boundary_lonlat: Option<String> = None;

    if clip_enabled && map_proj.proj_code() != "unknown" {
        let b = materialize_clip_boundary(map_proj, &source, dialect, execute_query)?;
        if let Some(wkt) = boundary_to_target_crs(&b, &target, dialect, execute_query) {
            projection
                .computed
                .insert("panel_boundary".to_string(), ParameterValue::String(wkt));
        }
        world_bbox = compute_world_bbox(&source, &target, dialect, execute_query);
        boundary_lonlat = Some(b);
    }
    let clip = boundary_lonlat.is_some();

    // Step 3: Apply per-layer projection (ST_Transform, clip to horizon)
    for (idx, layer) in layers.iter().enumerate() {
        let columns: Vec<String> = layer
            .mappings
            .aesthetics
            .keys()
            .map(|k| naming::aesthetic_column(k))
            .collect();
        layer_queries[idx] = layer.geom.apply_projection(
            &layer_queries[idx],
            projection,
            dialect,
            clip,
            &columns,
        )?;
    }

    // Step 4: Materialize projected layers as temp tables, compute data bbox,
    // then rewrite layer queries to read from those tables.
    let user_bbox = projection.properties.get("bounds");
    let needs_data_bbox = needs_data_bbox(user_bbox);
    let mut data_bbox: Option<BBox> = None;

    for (idx, layer) in layers.iter().enumerate() {
        let is_annotation = matches!(layer.source, Some(DataSource::Annotation));
        let is_spatial = layer.geom.geom_type() == GeomType::Spatial;
        let has_projected_positions = !is_spatial
            && source != target
            && layer.mappings.contains_key("pos1")
            && layer.mappings.contains_key("pos2");

        if is_annotation || (!is_spatial && !has_projected_positions) {
            continue;
        }

        let table_quoted = materialize_layer(idx, &layer_queries[idx], dialect, execute_query)?;

        if needs_data_bbox {
            let layer_bbox =
                compute_layer_bbox(&table_quoted, is_spatial, &target, dialect, execute_query);
            data_bbox = BBox::merge(data_bbox, layer_bbox)?;
        }

        layer_queries[idx] = if is_spatial {
            let columns: Vec<String> = layer
                .mappings
                .aesthetics
                .keys()
                .map(|k| naming::aesthetic_column(k))
                .collect();
            let geom_col_quoted = naming::quote_ident(&naming::aesthetic_column("geometry"));
            let wkb_expr = dialect.sql_geometry_to_wkb(&geom_col_quoted);
            dialect.sql_select_replace(
                &wkb_expr,
                &geom_col_quoted,
                &format!("SELECT * FROM {table_quoted}"),
                &columns,
            )
        } else {
            format!("SELECT * FROM {table_quoted}")
        };
    }

    // Step 5: Resolve final frame bbox from user bounds + data bounds + world bounds
    let Some(bbox) = resolve_final_bbox(user_bbox, data_bbox, world_bbox) else {
        return Ok(());
    };
    projection
        .computed
        .insert("bbox".to_string(), bbox.as_parameter_value());

    // Step 6: Generate graticule lines. The graticule is built and clipped
    // in EPSG:4326 (independent of source), then projected to target.
    let (lon_wkt, lat_wkt) = build_graticule(
        &bbox,
        boundary_lonlat.as_deref(),
        &target,
        dialect,
        execute_query,
    )?;
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
    bbox: &BBox,
    clip_boundary_wkt: Option<&str>,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<(Option<String>, Option<String>)> {
    let Some(geo_bbox) = graticule_bbox(bbox, clip_boundary_wkt, dialect, execute_query)? else {
        return Ok((None, None));
    };

    let (lon_min, lon_max) = geo_bbox.xrange();
    let (lat_min, lat_max) = geo_bbox.yrange();
    let lon_breaks = graticule_breaks(lon_min, lon_max, 7);
    let lat_breaks = graticule_breaks(lat_min, lat_max, 7);

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
        project_graticule_wkt(lon_wkt, clip_boundary_wkt, crs, dialect, execute_query)?,
        project_graticule_wkt(lat_wkt, clip_boundary_wkt, crs, dialect, execute_query)?,
    ))
}

/// Determine the lon/lat bounding box visible in the current frame by inverse-projecting
/// the bbox corners to EPSG:4326. Falls back to the clip boundary extent for azimuthal
/// projections where corners collapse to degenerate values.
fn graticule_bbox(
    bbox: &BBox,
    clip_boundary_wkt: Option<&str>,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<BBox>> {
    let mut geo_bbox = match bbox.reproject("EPSG:4326", dialect, execute_query) {
        Some(b) => b.clamp(-180.0, -90.0, 180.0, 90.0),
        None => return Ok(None),
    };

    // For azimuthal projections the bbox corners often inverse-project to
    // degenerate or incomplete values. Use the clip boundary extent which
    // correctly represents the visible hemisphere.
    if let Some(wkt) = clip_boundary_wkt {
        let sql = format!(
            "SELECT ST_XMin(g) AS xmin, ST_YMin(g) AS ymin, \
                    ST_XMax(g) AS xmax, ST_YMax(g) AS ymax \
             FROM (SELECT ST_GeomFromText('{wkt}') AS g)"
        );
        if let Ok(df) = execute_query(&sql) {
            if let Some(clip_bbox) = BBox::from_df(&df, "EPSG:4326") {
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
        if coords.len() >= 2 {
            lines.push(format!("({})", coords.join(", ")));
        }
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

/// Compose the clip boundary (visible area minus seam slits), materialize it as a
/// temp table in source CRS for per-layer clipping, and return the WKT in EPSG:4326.
fn materialize_clip_boundary(
    map_proj: &dyn MapProjectionTrait,
    source: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<String> {
    let wkt = map_proj.visible_area_wkt().unwrap();
    let half_width = 0.005;
    let slit_wkt = map_proj.slit_wkt(half_width);

    let boundary_lonlat = if let Some(slit) = &slit_wkt {
        let sql = format!(
            "SELECT ST_AsText(ST_Difference(ST_GeomFromText('{wkt}'), ST_GeomFromText('{slit}'))) AS wkt"
        );
        query_scalar_string(&sql, execute_query).unwrap_or(wkt)
    } else {
        wkt
    };

    let source_geom = dialect.sql_st_transform(
        &format!("ST_GeomFromText('{boundary_lonlat}')"),
        "EPSG:4326",
        source,
    );
    let body = format!("SELECT {source_geom} AS geom");
    for stmt in dialect.create_or_replace_temp_table_sql(&clip_boundary_table(), &[], &body) {
        execute_query(&stmt)?;
    }

    Ok(boundary_lonlat)
}

/// Project the clip boundary from EPSG:4326 to target CRS, returning the WKT.
fn boundary_to_target_crs(
    boundary_lonlat: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    let panel_geom = dialect.sql_st_transform(
        &format!("ST_GeomFromText('{boundary_lonlat}')"),
        "EPSG:4326",
        crs,
    );
    let sql = format!("SELECT ST_AsText({panel_geom}) AS wkt");
    query_scalar_string(&sql, execute_query)
}

/// Materialize a layer query as a temp table, returning the quoted table name.
fn materialize_layer(
    idx: usize,
    query: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<String> {
    let table_name = format!("{}_proj", naming::layer_key(idx));
    for stmt in dialect.create_or_replace_temp_table_sql(&table_name, &[], query) {
        execute_query(&stmt)?;
    }
    Ok(naming::quote_ident(&table_name))
}

/// Compute the bounding box of a single materialized layer table.
fn compute_layer_bbox(
    table: &str,
    is_spatial: bool,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<BBox> {
    let sql = if is_spatial {
        let geom_col = naming::quote_ident(&naming::aesthetic_column("geometry"));
        dialect.sql_geometry_bbox(&geom_col, table)
    } else {
        let pos1_col = naming::quote_ident(&naming::aesthetic_column("pos1"));
        let pos2_col = naming::quote_ident(&naming::aesthetic_column("pos2"));
        format!(
            "SELECT MIN({pos1_col}), MIN({pos2_col}), \
             MAX({pos1_col}), MAX({pos2_col}) FROM {table}"
        )
    };
    if let Ok(df) = execute_query(&sql) {
        BBox::from_df(&df, crs)
    } else {
        None
    }
}

/// Compute the world bounding box by reading the extent of the materialized
/// clip boundary table, projected to target CRS.
fn compute_world_bbox(
    source: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<BBox> {
    let projected = dialect.sql_st_transform("geom", source, crs);
    let sql = dialect.sql_geometry_bbox(&projected, &clip_boundary_table());
    if let Ok(df) = execute_query(&sql) {
        BBox::from_df(&df, crs)
    } else {
        None
    }
}

/// Clip (if needed) and project a graticule WKT from EPSG:4326 to the target CRS.
fn project_graticule_wkt(
    wkt: Option<String>,
    clip_boundary_wkt: Option<&str>,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<String>> {
    let Some(wkt) = wkt else { return Ok(None) };
    let geom_expr = format!("ST_GeomFromText('{wkt}')");
    let clipped = if let Some(boundary) = clip_boundary_wkt {
        // ST_CollectionExtract(..., 2) keeps only linestring components,
        // discarding stray points from vertex-on-boundary intersections.
        format!(
            "ST_CollectionExtract(ST_Intersection({geom_expr}, \
             ST_GeomFromText('{boundary}')), 2)"
        )
    } else {
        geom_expr
    };
    let projected = dialect.sql_st_transform(&clipped, "EPSG:4326", crs);
    let sql = format!("SELECT ST_AsText({projected}) AS wkt");
    Ok(query_scalar_string(&sql, execute_query))
}

/// Returns true if we need to compute a bbox (bounding box representing the extent of geometry)
/// from the data — i.e. when bounds is absent or has null elements that need filling in.
fn needs_data_bbox(user_bbox: Option<&ParameterValue>) -> bool {
    match user_bbox {
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
fn resolve_final_bbox(
    user_bbox: Option<&ParameterValue>,
    computed: Option<BBox>,
    world: Option<BBox>,
) -> Option<BBox> {
    if let Some(ParameterValue::Array(arr)) = user_bbox {
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
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<String>> {
    let geom_col = naming::quote_ident(&naming::aesthetic_column("geometry"));
    let ensure_geom = dialect.sql_ensure_geometry(&geom_col);
    let mut detected: Option<String> = None;

    for (idx, layer) in layers.iter().enumerate() {
        if layer.geom.geom_type() != GeomType::Spatial {
            continue;
        }
        let sql = format!(
            "SELECT ST_SRID({ensure_geom}) AS srid FROM ({}) WHERE {geom_col} IS NOT NULL LIMIT 1",
            layer_queries[idx]
        );
        if let Ok(df) = execute_query(&sql) {
            let batch = df.inner();
            if batch.num_rows() == 0 {
                continue;
            }
            let col = batch.column(0);
            let srid = col
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .map(|a| a.value(0) as i64)
                .or_else(|| {
                    col.as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .map(|a| a.value(0))
                })
                .or_else(|| {
                    col.as_any()
                        .downcast_ref::<arrow::array::Int16Array>()
                        .map(|a| a.value(0) as i64)
                });
            if let Some(srid) = srid {
                if srid != 0 {
                    let crs = format!("EPSG:{srid}");
                    if let Some(ref prev) = detected {
                        if *prev != crs {
                            return Err(crate::GgsqlError::ValidationError(format!(
                                "Spatial layers have conflicting CRS: '{}' vs '{}'. \
                                 Set PROJECT source to specify which CRS the data is in.",
                                prev, crs
                            )));
                        }
                    } else {
                        detected = Some(crs);
                    }
                }
            }
        }
    }
    Ok(detected)
}

// ---------------------------------------------------------------------------
// Table names
// ---------------------------------------------------------------------------

pub fn clip_boundary_table() -> String {
    format!("__ggsql_clip_boundary_{}__", naming::session_id())
}

// ---------------------------------------------------------------------------
// EPSG resolution
// ---------------------------------------------------------------------------

/// Resolve numeric EPSG codes in projection properties to PROJ strings, and
/// rebuild the coord if it is still `UnknownProj`.
///
/// Called by the executor before `apply_projection_transforms`. This ensures
/// `source` and `target` properties are always PROJ/EPSG strings (not raw numbers)
/// and that the coord struct matches the resolved CRS.
pub(crate) fn resolve_map_projection(
    projection: &mut super::super::Projection,
    layers: &[Layer],
    layer_queries: &[String],
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<()> {
    // Derive target from the projection's PROJ string if not explicitly set
    if let Some(mp) = projection.coord.as_map_projection() {
        let proj_str = mp.to_proj_str();
        if !proj_str.is_empty() {
            projection
                .properties
                .entry("target".to_string())
                .or_insert_with(|| ParameterValue::String(proj_str));
        }
    }

    let properties = &mut projection.properties;

    // Step 1: Detect source CRS from geometry columns if not explicitly set
    if !properties.contains_key("source") {
        if let Some(srid) = detect_source_srid(layers, layer_queries, dialect, execute_query)? {
            properties.insert("source".to_string(), ParameterValue::String(srid));
        }
    }

    // Step 2: Resolve source and target from numeric EPSG to PROJ strings
    let source = resolve_epsg_property("source", properties, execute_query)
        .unwrap_or_else(|| "EPSG:4326".to_string());
    let target = resolve_epsg_property("target", properties, execute_query)
        .unwrap_or_else(|| source.clone());

    properties.insert("source".to_string(), ParameterValue::String(source.clone()));
    properties.insert("target".to_string(), ParameterValue::String(target.clone()));

    // Step 3: Rebuild coord when it is still UnknownProj (EPSG wasn't resolvable at parse time)
    if projection
        .coord
        .as_map_projection()
        .is_some_and(|mp| mp.proj_code() == "unknown")
    {
        projection.coord = super::Coord::map("crs", &projection.properties);
    }

    // Step 4: Validate CRS by attempting a single point transform
    let probe = dialect.sql_st_transform("ST_Point(0, 0)", &source, &target);
    let probe_sql = format!("SELECT {probe} AS g");
    match execute_query(&probe_sql) {
        Err(e) => {
            let msg = e.to_string();
            return Err(crate::GgsqlError::ValidationError(format!(
                "Invalid target CRS '{}': {}",
                target,
                msg.split(':').next_back().unwrap_or(&msg).trim()
            )));
        }
        Ok(df) => {
            if df.height() > 0 && df.column("g").is_ok() {
                let val = crate::array_util::value_to_string(df.column("g").unwrap(), 0);
                if val == "null" || val.is_empty() {
                    return Err(crate::GgsqlError::ValidationError(format!(
                        "Unsupported target CRS '{}': this backend cannot transform to \
                         arbitrary PROJ strings. Use a numeric EPSG code instead.",
                        target,
                    )));
                }
            }
        }
    }

    Ok(())
}

/// If `key` holds a numeric EPSG code or an `"EPSG:N"` string, resolve it to a PROJ
/// string. Falls back to `"EPSG:N"` format when no PROJ string can be found (the
/// database engine may still handle it). Returns `None` when the property is absent.
fn resolve_epsg_property(
    key: &str,
    properties: &HashMap<String, ParameterValue>,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    let code: u32 = match properties.get(key) {
        Some(ParameterValue::Number(n)) => *n as u32,
        Some(ParameterValue::String(s)) => {
            match s.strip_prefix("EPSG:").and_then(|n| n.parse().ok()) {
                Some(c) => c,
                // Raw PROJ strings (`+proj=foo`) don't have EPSG prefixes
                None => return Some(s.clone()),
            }
        }
        _ => return None,
    };
    let resolved = query_spatial_ref_sys(code, execute_query)
        .or_else(|| builtin_epsg_lookup(code))
        .unwrap_or_else(|| format!("EPSG:{code}"));
    Some(resolved)
}

fn query_spatial_ref_sys(
    code: u32,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    let sql = format!(
        "SELECT proj4text FROM spatial_ref_sys WHERE srid = {} LIMIT 1",
        code
    );
    let df = execute_query(&sql).ok()?;
    let batch = df.inner();
    if batch.num_rows() == 0 {
        return None;
    }
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()?;
    let proj = arr.value(0);
    if proj.is_empty() {
        return None;
    }
    Some(proj.to_string())
}

fn builtin_epsg_lookup(code: u32) -> Option<String> {
    // UTM zones north (326xx) and south (327xx)
    if (32601..=32660).contains(&code) {
        let zone = code - 32600;
        return Some(format!(
            "+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs +type=crs"
        ));
    }
    if (32701..=32760).contains(&code) {
        let zone = code - 32700;
        return Some(format!(
            "+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs +type=crs"
        ));
    }

    let proj = match code {
        // World Geodetic System 1984 (GPS)
        4326 => "+proj=longlat +datum=WGS84 +no_defs +type=crs",
        // Lambert-93 -- France
        2154 => "+proj=lcc +lat_0=46.5 +lon_0=3 +lat_1=49 +lat_2=44 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // New Zealand Transverse Mercator
        2193 => "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // UTM North America
        26914 => "+proj=utm +zone=14 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // British National Grid - United Kingdom Ordnance Survey
        27700 => "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +nadgrids=uk_os_OSTN15_NTv2_OSGBtoETRS.tif +units=m +no_defs +type=crs",
        // Equi7 Europe
        27704 => "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.82 +y_0=2121415.696 +datum=WGS84 +units=m +no_defs +type=crs",
        // Antarctic Polar Stereographic
        3031 => "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        // LCC Europe
        3034 => "+proj=lcc +lat_0=52 +lon_0=10 +lat_1=35 +lat_2=65 +x_0=4000000 +y_0=2800000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // LAEA Europe
        3035 => "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // World Mercator
        3395 => "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        // NSIDC Sea Ice Polar Stereographic North
        3413 => "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        // Australian Albers
        3577 => "+proj=aea +lat_0=0 +lon_0=132 +lat_1=-18 +lat_2=-36 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // Pseudo-Mercator
        3857 => "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs",
        // NAD83 / Conus Albers
        5070 => "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs",
        // Polar Stereographic
        9354 => "+proj=stere +lat_0=-90 +lat_ts=-65 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        _ => return None,
    };
    Some(proj.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::projection::coord::{Coord, CoordKind};
    use crate::plot::ParameterValue;
    use std::collections::HashMap;

    #[test]
    fn test_map_properties() {
        let coord = Coord::map("crs", &HashMap::new());
        assert_eq!(coord.coord_kind(), CoordKind::Map);
        assert_eq!(coord.position_aesthetic_names(), &["lon", "lat"]);
    }

    #[test]
    fn test_map_default_properties() {
        let coord = Coord::map("crs", &HashMap::new());
        let defaults = coord.default_properties();
        let names: Vec<&str> = defaults.iter().map(|p| p.name).collect();
        assert!(names.contains(&"target"));
        assert!(names.contains(&"source"));
        assert!(names.contains(&"clip"));
        assert!(names.contains(&"bounds"));
        assert!(names.contains(&"origin"));
        assert!(names.contains(&"parallel"));
        assert_eq!(defaults.len(), 6);
    }

    #[test]
    fn test_map_accepts_target_string() {
        let mut props = HashMap::new();
        props.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        let coord = Coord::map("crs", &props);

        let resolved = coord.resolve_properties(Some("crs"), &props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("target").unwrap(),
            &ParameterValue::String("+proj=merc".to_string())
        );
    }

    #[test]
    fn test_map_rejects_unknown_property() {
        let coord = Coord::map("crs", &HashMap::new());
        let mut props = HashMap::new();
        props.insert(
            "unknown".to_string(),
            ParameterValue::String("value".to_string()),
        );

        let resolved = coord.resolve_properties(Some("crs"), &props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("not 'unknown'"));
    }

    #[test]
    fn test_target_rejects_origin_and_parallel() {
        let mut props = HashMap::new();
        props.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho".to_string()),
        );
        props.insert("origin".to_string(), ParameterValue::Number(30.0));
        let coord = Coord::map("crs", &props);

        let resolved = coord.resolve_properties(Some("crs"), &props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("Cannot combine 'target'"));
    }

    #[test]
    fn test_origin_rejects_latitude_out_of_range() {
        let mut props = HashMap::new();
        props.insert(
            "origin".to_string(),
            ParameterValue::Array(vec![
                crate::plot::types::ArrayElement::Number(0.0),
                crate::plot::types::ArrayElement::Number(95.0),
            ]),
        );
        let coord = Coord::map("albers", &props);

        let resolved = coord.resolve_properties(Some("albers"), &props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("origin latitude must be between -90 and 90"));
    }

    fn bbox(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> BBox {
        BBox::from_array([xmin, ymin, xmax, ymax], "EPSG:4326")
    }

    #[test]
    fn test_resolve_final_bbox_no_bounds_uses_computed() {
        let computed = Some(bbox(0.0, 0.0, 100.0, 200.0));
        assert_eq!(resolve_final_bbox(None, computed.clone(), None), computed);
    }

    #[test]
    fn test_resolve_final_bbox_no_bounds_no_computed() {
        assert_eq!(resolve_final_bbox(None, None, None), None);
    }

    #[test]
    fn test_resolve_final_bbox_explicit_bounds_override_computed() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Number(10.0),
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        let computed = Some(bbox(0.0, 0.0, 100.0, 200.0));
        assert_eq!(
            resolve_final_bbox(Some(&bounds), computed, None),
            Some(bbox(10.0, 20.0, 30.0, 40.0))
        );
    }

    #[test]
    fn test_resolve_final_bbox_null_elements_use_computed() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Null,
            ArrayElement::Number(20.0),
            ArrayElement::Null,
            ArrayElement::Number(40.0),
        ]);
        let computed = Some(bbox(5.0, 0.0, 95.0, 0.0));
        assert_eq!(
            resolve_final_bbox(Some(&bounds), computed, None),
            Some(bbox(5.0, 20.0, 95.0, 40.0))
        );
    }

    #[test]
    fn test_resolve_final_bbox_inf_elements_use_world() {
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
            resolve_final_bbox(Some(&bounds), computed, world),
            Some(bbox(-500.0, 20.0, 500.0, 40.0))
        );
    }

    #[test]
    fn test_resolve_final_bbox_null_without_computed_falls_through() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Null,
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        assert_eq!(resolve_final_bbox(Some(&bounds), None, None), None);
    }

    #[test]
    fn test_resolve_final_bbox_inf_without_world_falls_through() {
        use crate::plot::types::ArrayElement;
        let bounds = ParameterValue::Array(vec![
            ArrayElement::Number(f64::INFINITY),
            ArrayElement::Number(20.0),
            ArrayElement::Number(30.0),
            ArrayElement::Number(40.0),
        ]);
        let computed = Some(bbox(5.0, 0.0, 95.0, 200.0));
        assert_eq!(
            resolve_final_bbox(Some(&bounds), computed.clone(), None),
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

    #[test]
    fn epsg_utm_north_zones() {
        let s = builtin_epsg_lookup(32632).unwrap();
        assert!(s.contains("+proj=utm") && s.contains("+zone=32"));
        let s = builtin_epsg_lookup(32601).unwrap();
        assert!(s.contains("+proj=utm") && s.contains("+zone=1"));
    }

    #[test]
    fn epsg_utm_south_zones() {
        let s = builtin_epsg_lookup(32755).unwrap();
        assert!(s.contains("+proj=utm") && s.contains("+zone=55") && s.contains("+south"));
    }

    #[test]
    fn epsg_common_codes() {
        assert!(builtin_epsg_lookup(4326).unwrap().contains("+proj=longlat"));
        assert!(builtin_epsg_lookup(3857).unwrap().contains("+proj=merc"));
        assert!(builtin_epsg_lookup(3035).unwrap().contains("+proj=laea"));
        assert!(builtin_epsg_lookup(5070).unwrap().contains("+proj=aea"));
    }

    #[test]
    fn epsg_unknown_code() {
        assert_eq!(builtin_epsg_lookup(99999), None);
    }
}
