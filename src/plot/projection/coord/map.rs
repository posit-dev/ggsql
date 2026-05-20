//! Map coordinate system implementation

use super::{map_projections, CoordKind, CoordTrait};
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
            if let Some(srid) = detect_source_srid(layers, layer_queries, execute_query)? {
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
            _ => {
                projection
                    .properties
                    .insert("crs".to_string(), ParameterValue::String(source.clone()));
                source.clone()
            }
        };

        // Validate CRS by attempting a single point transform
        let probe = dialect.sql_st_transform("ST_Point(0, 0)", &source, &crs);
        if let Err(e) = execute_query(&format!("SELECT {probe}")) {
            let msg = e.to_string();
            return Err(crate::GgsqlError::ValidationError(format!(
                "Invalid CRS '{}': {}",
                crs,
                msg.split(':').last().unwrap_or(&msg).trim()
            )));
        }

        // Step 2: Compute the visible area boundary for this projection.
        // Produces: clip boundary temp table (in source CRS for layer clipping),
        // panel_boundary (projected WKT for the writer's background layer), and
        // world_bbox (bounding box of the full projected visible area).
        let (world_bbox, boundary_4326) =
            setup_clip_boundary(projection, &source, &crs, dialect, execute_query)?;
        let clip = boundary_4326.is_some();

        // Step 3: Apply per-layer projection (ST_Transform, clip to horizon)
        for (idx, layer) in layers.iter().enumerate() {
            layer_queries[idx] =
                layer
                    .geom
                    .apply_projection(&layer_queries[idx], projection, dialect, clip)?;
        }

        // Step 4: Materialize projected layers as temp tables, compute data bbox,
        // then rewrite layer queries to read from those tables.
        let geom_col = naming::aesthetic_column("geometry");
        let geom_col_quoted = naming::quote_ident(&geom_col);
        let pos1_col = naming::quote_ident(&naming::aesthetic_column("pos1"));
        let pos2_col = naming::quote_ident(&naming::aesthetic_column("pos2"));
        let bounds_param = projection.properties.get("bounds");
        let mut computed_bbox: Option<BBox> = None;

        for (idx, layer) in layers.iter().enumerate() {
            let is_spatial = layer.geom.geom_type() == GeomType::Spatial;
            let has_projected_positions = !is_spatial
                && source != crs
                && layer.mappings.contains_key("pos1")
                && layer.mappings.contains_key("pos2");

            if !is_spatial && !has_projected_positions {
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
                let bbox_sql = if is_spatial {
                    dialect.sql_geometry_bbox(&geom_col_quoted, &table_quoted)
                } else {
                    format!(
                        "SELECT MIN({pos1_col}), MIN({pos2_col}), \
                         MAX({pos1_col}), MAX({pos2_col}) FROM {table_quoted}"
                    )
                };
                if let Ok(df) = execute_query(&bbox_sql) {
                    computed_bbox = BBox::merge(computed_bbox, BBox::from_df(&df, &crs))?;
                }
            }

            layer_queries[idx] = if is_spatial {
                let wkb_expr = dialect.sql_geometry_to_wkb(&geom_col_quoted);
                format!("SELECT * REPLACE ({wkb_expr} AS {geom_col_quoted}) FROM {table_quoted}")
            } else {
                format!("SELECT * FROM {table_quoted}")
            };
        }

        // Step 5: Resolve final frame bbox from user bounds + data bounds + world bounds
        let Some(bbox) = resolve_frame_bbox(bounds_param, computed_bbox, world_bbox) else {
            return Ok(());
        };
        projection
            .computed
            .insert("frame_bbox".to_string(), bbox.as_parameter_value());

        // Step 6: Generate graticule lines. The graticule is built and clipped
        // in EPSG:4326 (independent of source), then projected to target.
        let (lon_wkt, lat_wkt) = build_graticule(
            &bbox,
            boundary_4326.as_deref(),
            &crs,
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
    clip_boundary_wkt: Option<&str>,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<(Option<String>, Option<String>)> {
    let Some(geo_bbox) = graticule_bbox(frame_bbox, clip_boundary_wkt, dialect, execute_query)?
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

    // Compute the seam meridian (lon_0 + 180°) for non-cylindrical projections.
    // Graticule lines must not cross this longitude.
    let seam = if clip_boundary_wkt.is_some() {
        let lon_0 = map_projections::extract_proj_param_str(crs, "+lon_0=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        Some(map_projections::wrap_lon(lon_0 + 180.0))
    } else {
        None
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
            seam,
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
            seam,
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
    frame_bbox: &BBox,
    clip_boundary_wkt: Option<&str>,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<Option<BBox>> {
    let mut geo_bbox = match frame_bbox.reproject("EPSG:4326", dialect, execute_query) {
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
    seam: Option<f64>,
) -> String {
    let seam_epsilon = 0.01;

    let mut lines: Vec<String> = Vec::with_capacity(breaks.len());
    for &fixed in breaks {
        // For meridians (lon_first): skip if the meridian is at the seam
        if lon_first {
            if let Some(s) = seam {
                if (fixed - s).abs() < seam_epsilon {
                    continue;
                }
            }
        }

        // For parallels (lon varies): split the line at the seam
        let segments = if !lon_first {
            seam_split_ranges(vary_min, vary_max, seam, seam_epsilon)
        } else {
            vec![(vary_min, vary_max)]
        };

        for (seg_min, seg_max) in segments {
            let mut coords = Vec::new();
            let mut v = seg_min;
            while v < seg_max {
                let (lon, lat) = if lon_first { (fixed, v) } else { (v, fixed) };
                coords.push(format!("{lon:.6} {lat:.6}"));
                v += step_deg;
            }
            let (lon, lat) = if lon_first {
                (fixed, seg_max)
            } else {
                (seg_max, fixed)
            };
            coords.push(format!("{lon:.6} {lat:.6}"));
            if coords.len() >= 2 {
                lines.push(format!("({})", coords.join(", ")));
            }
        }
    }
    format!("MULTILINESTRING({})", lines.join(", "))
}

/// Split a range into sub-ranges that avoid the seam.
/// Returns one range if no seam, or two ranges if the seam falls within [min, max].
fn seam_split_ranges(min: f64, max: f64, seam: Option<f64>, epsilon: f64) -> Vec<(f64, f64)> {
    let Some(s) = seam else {
        return vec![(min, max)];
    };
    if s <= min + epsilon || s >= max - epsilon {
        return vec![(min, max)];
    }
    vec![(min, s - epsilon), (s + epsilon, max)]
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

/// Set up the clip/visible area boundary.
///
/// Returns `(world_bbox, boundary_4326_wkt)`:
/// - `world_bbox`: bounding box of the boundary projected into the target CRS
/// - `boundary_4326_wkt`: combined clip+slit boundary in EPSG:4326 (for graticule use)
///
/// Side effects:
/// - Creates `CLIP_BOUNDARY_TABLE` temp table containing the boundary in source CRS
/// - Stores `panel_boundary` in `projection.computed` (boundary projected to target CRS)
fn setup_clip_boundary(
    projection: &mut super::super::Projection,
    source: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> crate::Result<(Option<BBox>, Option<String>)> {
    let Some(map_proj) = projection.map_projection.as_ref() else {
        return Ok((None, None));
    };
    let Some(wkt) = map_proj.visible_area_wkt() else {
        return Ok((None, None));
    };

    let half_width = 0.005;
    let slit_wkt = map_proj.slit_wkt(half_width);

    // Combine clip boundary and seam slit into a single geometry in EPSG:4326.
    let boundary_4326 = if let Some(slit) = &slit_wkt {
        let sql = format!(
            "SELECT ST_AsText(ST_Difference(ST_GeomFromText('{wkt}'), ST_GeomFromText('{slit}'))) AS wkt"
        );
        query_scalar_string(&sql, execute_query).unwrap_or(wkt)
    } else {
        wkt
    };

    // Store the boundary in source CRS for per-layer clipping.
    let source_geom = dialect.sql_st_transform(
        &format!("ST_GeomFromText('{boundary_4326}')"),
        "EPSG:4326",
        source,
    );
    let body = format!("SELECT {source_geom} AS geom");
    for stmt in dialect.create_or_replace_temp_table_sql(CLIP_BOUNDARY_TABLE, &[], &body) {
        execute_query(&stmt)?;
    }

    // Project the boundary to target CRS for the panel background shape.
    let panel_geom = dialect.sql_st_transform(
        &format!("ST_GeomFromText('{boundary_4326}')"),
        "EPSG:4326",
        crs,
    );
    let sql = format!("SELECT ST_AsText({panel_geom}) AS wkt");
    if let Some(projected_wkt) = query_scalar_string(&sql, execute_query) {
        projection.computed.insert(
            "panel_boundary".to_string(),
            ParameterValue::String(projected_wkt),
        );
    }

    // World bbox: extent of the boundary in target CRS.
    let projected = dialect.sql_st_transform("geom", source, crs);
    let world_bbox_sql = dialect.sql_geometry_bbox(&projected, CLIP_BOUNDARY_TABLE);
    let world_bbox = execute_query(&world_bbox_sql)
        .ok()
        .and_then(|df| BBox::from_df(&df, crs));
    Ok((world_bbox, Some(boundary_4326)))
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
) -> crate::Result<Option<String>> {
    let geom_col = naming::quote_ident(&naming::aesthetic_column("geometry"));
    let mut detected: Option<String> = None;

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
        let wkt = grid_lines_wkt(&[0.0, 30.0], (-90.0, 90.0), 45.0, true, None);
        assert!(wkt.starts_with("MULTILINESTRING("), "{wkt}");
        assert!(wkt.contains("0.000000 -90.000000"), "{wkt}");
        assert!(wkt.contains("30.000000 -90.000000"), "{wkt}");
        assert!(wkt.contains("0.000000 90.000000"), "{wkt}");
        assert!(wkt.contains("30.000000 90.000000"), "{wkt}");
    }

    #[test]
    fn test_grid_lines_wkt_parallels() {
        let wkt = grid_lines_wkt(&[0.0, 45.0], (-180.0, 180.0), 90.0, false, None);
        assert!(wkt.starts_with("MULTILINESTRING("));
        assert!(wkt.contains("0.000000"));
        assert!(wkt.contains("45.000000"));
    }

    #[test]
    fn test_grid_lines_wkt_seam_splits_parallels() {
        // Seam at 90°: parallels spanning -180..180 should be split into two segments
        let wkt = grid_lines_wkt(&[0.0], (-180.0, 180.0), 30.0, false, Some(90.0));
        assert!(wkt.starts_with("MULTILINESTRING("));
        // Should have two linestrings (split at seam)
        let line_count = wkt.matches("(").count() - 1; // subtract outer parens
        assert!(line_count >= 2, "expected split into 2+ lines, got: {wkt}");
        // First segment should end before seam, second should start after
        assert!(
            wkt.contains("89.99"),
            "first segment should stop near seam: {wkt}"
        );
        assert!(
            wkt.contains("90.01"),
            "second segment should start past seam: {wkt}"
        );
    }

    #[test]
    fn test_grid_lines_wkt_seam_skips_meridian() {
        // Seam at 90°: a meridian at 90° should be skipped
        let wkt = grid_lines_wkt(&[60.0, 90.0, 120.0], (-90.0, 90.0), 45.0, true, Some(90.0));
        assert!(wkt.starts_with("MULTILINESTRING("));
        assert!(wkt.contains("60.000000"), "60° meridian should be present");
        assert!(
            wkt.contains("120.000000"),
            "120° meridian should be present"
        );
        assert!(
            !wkt.contains("90.000000 "),
            "90° meridian should be skipped: {wkt}"
        );
    }
}
