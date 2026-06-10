use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::CoordKind;
use crate::plot::types::{
    validate_parameter, ArrayElement, DefaultParamValue, ParamConstraint, ParamDefinition,
    TypeConstraint,
};
use crate::plot::{Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

// When adding a new named projection:
// 1. Add a struct implementing MapProjectionTrait (in this file)
// 2. Add the name to NAMED_PROJECTIONS below
// 3. Add a match arm in the build_projection! macro (proj code | named key)
pub const NAMED_PROJECTIONS: &[&str] = &[
    "geographic",
    "mercator",
    "transverse_mercator",
    "utm",
    "orthographic",
    "miller",
    "equirectangular",
    "stereographic",
    "gnomonic",
    "equal_area",
    "mollweide",
    "sinusoidal",
    "eckert4",
    "natural",
    "winkel_tripel",
    "albers",
    "lambert_conformal",
    "lambert",
    "azimuthal_equidistant",
    "igh",
    "robinson",
];

// =============================================================================
// Trait
// =============================================================================

/// Extension of `CoordTrait` for geographic map projections.
///
/// Implementing this trait gives you `CoordTrait` for free via the blanket impl below.
pub trait MapProjectionTrait: fmt::Debug + Send + Sync {
    fn proj_code(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    fn origin(&self) -> (f64, f64);
    fn to_proj_str(&self) -> String;

    fn visible_area_wkt(&self) -> Option<String> {
        Some(self.clip_shape_wkt())
    }

    fn clip_shape_wkt(&self) -> String {
        rectangle_wkt(
            -180.0,
            self.lat_bounds().0,
            180.0,
            self.lat_bounds().1,
            self.edge_segments(),
        )
    }

    fn lat_bounds(&self) -> (f64, f64) {
        (-90.0, 90.0)
    }

    fn edge_segments(&self) -> [usize; 4] {
        [1, 36, 1, 36]
    }

    fn slit_wkt(&self, epsilon: f64) -> Option<String> {
        let seam = wrap_lon(self.origin().0 + 180.0);
        if (seam - (-180.0)).abs() > epsilon && (seam - 180.0).abs() > epsilon {
            let segs = self.edge_segments()[1];
            Some(rectangle_wkt(
                seam - epsilon,
                -90.0,
                seam + epsilon,
                90.0,
                [1, segs, 1, segs],
            ))
        } else {
            None
        }
    }
}

// =============================================================================
// Blanket CoordTrait impl for all MapProjectionTrait implementors
// =============================================================================

use crate::plot::types::{ArrayConstraint, NumberConstraint};

const LON_RANGE: NumberConstraint = NumberConstraint::range(-180.0, 180.0);
const LAT_RANGE: NumberConstraint = NumberConstraint::range(-90.0, 90.0);

static MAP_PARAMS: &[ParamDefinition] = &[
    ParamDefinition {
        name: "target",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint {
            number: TypeConstraint::Constrained(NumberConstraint::count(1.0)),
            string: TypeConstraint::Any,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        },
    },
    ParamDefinition {
        name: "source",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint {
            number: TypeConstraint::Constrained(NumberConstraint::count(1.0)),
            string: TypeConstraint::Any,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        },
    },
    ParamDefinition {
        name: "clip",
        default: DefaultParamValue::Boolean(true),
        constraint: ParamConstraint::boolean(),
    },
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
    ParamDefinition {
        name: "origin",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::number_or_numeric_array(
            LON_RANGE,
            ArrayConstraint::of_numbers_len(LON_RANGE, 2),
        ),
    },
    ParamDefinition {
        name: "parallel",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::number_or_numeric_array(
            LAT_RANGE,
            ArrayConstraint::of_numbers_len(LAT_RANGE, 2),
        ),
    },
];

impl<T: MapProjectionTrait + 'static> super::CoordTrait for T {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Map
    }

    fn name(&self) -> &'static str {
        self.display_name()
    }

    fn position_aesthetic_names(&self) -> &'static [&'static str] {
        &["lon", "lat"]
    }

    fn default_properties(&self) -> &'static [ParamDefinition] {
        MAP_PARAMS
    }

    fn resolve_properties(
        &self,
        coord_type_name: Option<&str>,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        if let Some(name) = coord_type_name {
            if name != "crs" && properties.contains_key("target") {
                return Err(format!(
                    "Cannot combine a named projection ('{}') with a 'target' setting. \
                     Use either PROJECT TO {} or PROJECT TO crs SETTING target => '...'",
                    name, name
                ));
            }
        }
        let has_target = properties.contains_key("target");
        let has_origin = properties.contains_key("origin");
        let has_parallel = properties.contains_key("parallel");
        if has_target && (has_origin || has_parallel) {
            return Err(
                "Cannot combine 'target' setting with 'origin' or 'parallel'. \
                 Use either the CRS string or a named projection with 'origin'/'parallel' settings."
                    .to_string(),
            );
        }
        let defaults = self.default_properties();
        for (key, value) in properties.iter() {
            if let Some(param) = defaults.iter().find(|p| p.name == key) {
                validate_parameter(key, value, &param.constraint)?;
            } else {
                let allowed: Vec<&str> = defaults.iter().map(|p| p.name).collect();
                return Err(format!(
                    "{} projection property should be {}, not '{}'",
                    self.name(),
                    crate::or_list_quoted(&allowed, '\''),
                    key
                ));
            }
        }
        if let Some(ParameterValue::Array(arr)) = properties.get("origin") {
            if let Some(crate::plot::types::ArrayElement::Number(lat)) = arr.get(1) {
                if *lat < -90.0 || *lat > 90.0 {
                    return Err(format!(
                        "origin latitude must be between -90 and 90, got {}",
                        lat
                    ));
                }
            }
        }
        let mut resolved = properties.clone();
        for param in defaults {
            if !resolved.contains_key(param.name) {
                if let Some(default) = param.to_parameter_value() {
                    resolved.insert(param.name.to_string(), default);
                }
            }
        }
        Ok(resolved)
    }

    fn apply_projection_transforms(
        &self,
        layers: &mut [Layer],
        layer_queries: &mut [String],
        projection: &mut super::super::Projection,
        dialect: &dyn SqlDialect,
        execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
    ) -> crate::Result<()> {
        super::map::apply_map_transforms(
            self,
            layers,
            layer_queries,
            projection,
            dialect,
            execute_query,
        )
    }

    fn as_map_projection(&self) -> Option<&dyn MapProjectionTrait> {
        Some(self)
    }
}

// =============================================================================
// Factory
// =============================================================================

/// Macro that produces the match expression for building a projection struct.
/// The coercion target is determined by the type annotation at the call site.
macro_rules! build_projection {
    ($name:expr, $properties:expr) => {{
        let properties = $properties;

        // Extract parameters: from PROJ string if target is set, otherwise from properties
        let (key, lon_0, lat_0, lat_1, lat_2, raw_crs) =
            if let Some(ParameterValue::String(crs)) = properties.get("target") {
                let code = extract_proj_param_str(crs, "+proj=").unwrap_or("");
                let lon_0 = extract_f64_param(crs, "+lon_0=").unwrap_or(0.0);
                let lat_0 = extract_f64_param(crs, "+lat_0=").unwrap_or(0.0);
                let lat_1 = extract_f64_param(crs, "+lat_1=").unwrap_or(29.5);
                let lat_2 = extract_f64_param(crs, "+lat_2=").unwrap_or(45.5);
                (code, lon_0, lat_0, lat_1, lat_2, Some(crs.clone()))
            } else {
                let (lon_0, lat_0) = match properties.get("origin") {
                    Some(ParameterValue::Number(lon)) => (*lon, 0.0),
                    Some(ParameterValue::Array(arr)) => {
                        let lon = match arr.first() {
                            Some(ArrayElement::Number(n)) => *n,
                            _ => 0.0,
                        };
                        let lat = match arr.get(1) {
                            Some(ArrayElement::Number(n)) => *n,
                            _ => 0.0,
                        };
                        (lon, lat)
                    }
                    _ => (0.0, 0.0),
                };
                let (lat_1, lat_2) = match properties.get("parallel") {
                    Some(ParameterValue::Number(lat)) => (*lat, *lat),
                    Some(ParameterValue::Array(arr)) => {
                        let l1 = match arr.first() {
                            Some(ArrayElement::Number(n)) => *n,
                            _ => 29.5,
                        };
                        let l2 = match arr.get(1) {
                            Some(ArrayElement::Number(n)) => *n,
                            _ => 45.5,
                        };
                        (l1, l2)
                    }
                    _ => (29.5, 45.5),
                };
                ($name, lon_0, lat_0, lat_1, lat_2, None)
            };

        match key {
            "longlat" | "latlong" | "geographic" => Arc::new(Geographic { lon_0 }),
            "ortho" | "orthographic" => Arc::new(Orthographic { lon_0, lat_0 }),
            "stere" | "stereographic" => Arc::new(Stereographic { lon_0, lat_0 }),
            "gnom" | "gnomonic" => Arc::new(Gnomonic { lon_0, lat_0 }),
            "laea" | "lambert" => Arc::new(LambertAzimuthal { lon_0, lat_0 }),
            "aeqd" | "azimuthal_equidistant" => Arc::new(AzimuthalEquidistant { lon_0, lat_0 }),
            "tmerc" | "transverse_mercator" => Arc::new(TransverseMercator { lon_0, lat_0 }),
            "utm" => {
                if let Some(ref crs) = raw_crs {
                    let zone = extract_f64_param(crs, "+zone=")
                        .map(|z| (z as u8).clamp(1, 60))
                        .unwrap_or_else(|| Utm::from_origin(lon_0, lat_0).zone);
                    let south = crs.contains("+south");
                    Arc::new(Utm { zone, south })
                } else {
                    Arc::new(Utm::from_origin(lon_0, lat_0))
                }
            }
            "merc" | "mercator" => Arc::new(Mercator { lon_0 }),
            "mill" | "miller" => Arc::new(Miller { lon_0 }),
            "eqc" | "equirectangular" => Arc::new(Equirectangular { lon_0 }),
            "cea" | "equal_area" => Arc::new(CylindricalEqualArea { lon_0 }),
            "robin" | "robinson" => Arc::new(Robinson { lon_0 }),
            "moll" | "mollweide" => Arc::new(Mollweide { lon_0 }),
            "sinu" | "sinusoidal" => Arc::new(Sinusoidal { lon_0 }),
            "eck4" | "eckert4" => Arc::new(Eckert4 { lon_0 }),
            "natearth" | "natural" => Arc::new(NaturalEarth { lon_0 }),
            "igh" => Arc::new(Igh { lon_0 }),
            "wintri" | "winkel_tripel" => Arc::new(WinkelTripel { lon_0 }),
            "aea" | "albers" => Arc::new(AlbersEqualArea {
                lon_0,
                lat_0,
                lat_1,
                lat_2,
            }),
            "lcc" | "lambert_conformal" => Arc::new(LambertConformalConic {
                lon_0,
                lat_0,
                lat_1,
                lat_2,
            }),
            "crs" => Arc::new(UnknownProj {
                raw: raw_crs.unwrap_or_default(),
                lon_0,
                lat_0,
            }),
            _ if raw_crs.is_some() => Arc::new(UnknownProj {
                raw: raw_crs.unwrap(),
                lon_0,
                lat_0,
            }),
            _ => return None,
        }
    }};
}

/// Build a map projection as a `CoordTrait` object from a name and properties.
/// Returns `None` if the name is not recognised.
pub fn build_map_projection(
    name: Option<&str>,
    properties: &HashMap<String, ParameterValue>,
) -> Option<Arc<dyn super::CoordTrait>> {
    let name = name?;
    let obj: Arc<dyn super::CoordTrait> = build_projection!(name, properties);
    Some(obj)
}

// =============================================================================
// Azimuthal projections
// =============================================================================

#[derive(Debug, Clone)]
pub struct Orthographic {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for Orthographic {
    fn proj_code(&self) -> &'static str {
        "ortho"
    }
    fn display_name(&self) -> &'static str {
        "Orthographic"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=ortho +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        hemisphere_polygon_wkt(self.lon_0, self.lat_0, 88.0)
    }
}

#[derive(Debug, Clone)]
pub struct Stereographic {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for Stereographic {
    fn proj_code(&self) -> &'static str {
        "stere"
    }
    fn display_name(&self) -> &'static str {
        "Stereographic"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=stere +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        hemisphere_polygon_wkt(self.lon_0, self.lat_0, 88.0)
    }
}

#[derive(Debug, Clone)]
pub struct Gnomonic {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for Gnomonic {
    fn proj_code(&self) -> &'static str {
        "gnom"
    }
    fn display_name(&self) -> &'static str {
        "Gnomonic"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=gnom +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        hemisphere_polygon_wkt(self.lon_0, self.lat_0, 60.0)
    }
}

#[derive(Debug, Clone)]
pub struct LambertAzimuthal {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for LambertAzimuthal {
    fn proj_code(&self) -> &'static str {
        "laea"
    }
    fn display_name(&self) -> &'static str {
        "Lambert Azimuthal Equal-Area"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=laea +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        rectangle_wkt(-180.0, -90.0, 180.0, 90.0, [1, 36, 1, 36])
    }
    fn slit_wkt(&self, _epsilon: f64) -> Option<String> {
        Some(antipodal_cap_wkt(self.lon_0, self.lat_0))
    }
    fn edge_segments(&self) -> [usize; 4] {
        [36, 36, 36, 36]
    }
}

#[derive(Debug, Clone)]
pub struct AzimuthalEquidistant {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for AzimuthalEquidistant {
    fn proj_code(&self) -> &'static str {
        "aeqd"
    }
    fn display_name(&self) -> &'static str {
        "Azimuthal Equidistant"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=aeqd +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        rectangle_wkt(-180.0, -90.0, 180.0, 90.0, [1, 36, 1, 36])
    }
    fn slit_wkt(&self, _epsilon: f64) -> Option<String> {
        Some(antipodal_cap_wkt(self.lon_0, self.lat_0))
    }
    fn edge_segments(&self) -> [usize; 4] {
        [36, 36, 36, 36]
    }
}

// =============================================================================
// Geographic (unprojected)
// =============================================================================

#[derive(Debug, Clone)]
pub struct Geographic {
    pub lon_0: f64,
}

impl MapProjectionTrait for Geographic {
    fn proj_code(&self) -> &'static str {
        "longlat"
    }
    fn display_name(&self) -> &'static str {
        "Geographic"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=longlat +lon_0={}", self.lon_0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 1, 1, 1]
    }
}

// =============================================================================
// Cylindrical projections
// =============================================================================

#[derive(Debug, Clone)]
pub struct Mercator {
    pub lon_0: f64,
}

impl MapProjectionTrait for Mercator {
    fn proj_code(&self) -> &'static str {
        "merc"
    }
    fn display_name(&self) -> &'static str {
        "Mercator"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=merc +lon_0={}", self.lon_0)
    }
    fn lat_bounds(&self) -> (f64, f64) {
        (-85.0, 85.0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 1, 1, 1]
    }
}

#[derive(Debug, Clone)]
pub struct TransverseMercator {
    pub lon_0: f64,
    pub lat_0: f64,
}

impl MapProjectionTrait for TransverseMercator {
    fn proj_code(&self) -> &'static str {
        "tmerc"
    }
    fn display_name(&self) -> &'static str {
        "Transverse Mercator"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=tmerc +lon_0={} +lat_0={}", self.lon_0, self.lat_0)
    }
    fn clip_shape_wkt(&self) -> String {
        let west = self.lon_0 - 80.0;
        let east = self.lon_0 + 80.0;
        let segs = self.edge_segments();

        if west >= -180.0 && east <= 180.0 {
            rectangle_wkt(west, -90.0, east, 90.0, segs)
        } else {
            let (w1, e1, w2, e2) = if east > 180.0 {
                (west, 180.0, -180.0, east - 360.0)
            } else {
                (west + 360.0, 180.0, -180.0, east)
            };
            let r1 = rectangle_wkt(w1, -90.0, e1, 90.0, segs);
            let r2 = rectangle_wkt(w2, -90.0, e2, 90.0, segs);
            format!(
                "MULTIPOLYGON({}, {})",
                r1.strip_prefix("POLYGON").unwrap(),
                r2.strip_prefix("POLYGON").unwrap()
            )
        }
    }
    fn slit_wkt(&self, _epsilon: f64) -> Option<String> {
        None
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 36, 1, 36]
    }
}

#[derive(Debug, Clone)]
pub struct Utm {
    pub zone: u8,
    pub south: bool,
}

impl Utm {
    pub fn from_origin(lon_0: f64, lat_0: f64) -> Self {
        let zone = (((lon_0 + 180.0) / 6.0).floor() as u8).clamp(0, 59) + 1;
        Self {
            zone,
            south: lat_0 < 0.0,
        }
    }

    fn central_meridian(&self) -> f64 {
        self.zone as f64 * 6.0 - 183.0
    }
}

impl MapProjectionTrait for Utm {
    fn proj_code(&self) -> &'static str {
        "utm"
    }
    fn display_name(&self) -> &'static str {
        "UTM"
    }
    fn origin(&self) -> (f64, f64) {
        let lat = if self.south { -45.0 } else { 45.0 };
        (self.central_meridian(), lat)
    }
    fn to_proj_str(&self) -> String {
        if self.south {
            format!("+proj=utm +zone={} +south", self.zone)
        } else {
            format!("+proj=utm +zone={}", self.zone)
        }
    }
    fn clip_shape_wkt(&self) -> String {
        let lon_0 = self.central_meridian();
        let west = lon_0 - 80.0;
        let east = lon_0 + 80.0;
        let segs = self.edge_segments();

        if west >= -180.0 && east <= 180.0 {
            rectangle_wkt(west, -90.0, east, 90.0, segs)
        } else {
            let (w1, e1, w2, e2) = if east > 180.0 {
                (west, 180.0, -180.0, east - 360.0)
            } else {
                (west + 360.0, 180.0, -180.0, east)
            };
            let r1 = rectangle_wkt(w1, -90.0, e1, 90.0, segs);
            let r2 = rectangle_wkt(w2, -90.0, e2, 90.0, segs);
            format!(
                "MULTIPOLYGON({}, {})",
                r1.strip_prefix("POLYGON").unwrap(),
                r2.strip_prefix("POLYGON").unwrap()
            )
        }
    }
    fn slit_wkt(&self, _epsilon: f64) -> Option<String> {
        None
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 36, 1, 36]
    }
}

#[derive(Debug, Clone)]
pub struct Miller {
    pub lon_0: f64,
}

impl MapProjectionTrait for Miller {
    fn proj_code(&self) -> &'static str {
        "mill"
    }
    fn display_name(&self) -> &'static str {
        "Miller"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=mill +lon_0={}", self.lon_0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 1, 1, 1]
    }
}

#[derive(Debug, Clone)]
pub struct Equirectangular {
    pub lon_0: f64,
}

impl MapProjectionTrait for Equirectangular {
    fn proj_code(&self) -> &'static str {
        "eqc"
    }
    fn display_name(&self) -> &'static str {
        "Equirectangular"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=eqc +lon_0={}", self.lon_0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 1, 1, 1]
    }
}

#[derive(Debug, Clone)]
pub struct CylindricalEqualArea {
    pub lon_0: f64,
}

impl MapProjectionTrait for CylindricalEqualArea {
    fn proj_code(&self) -> &'static str {
        "cea"
    }
    fn display_name(&self) -> &'static str {
        "Cylindrical Equal-Area"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=cea +lon_0={}", self.lon_0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [1, 1, 1, 1]
    }
}

// =============================================================================
// Pseudocylindrical projections
// =============================================================================

#[derive(Debug, Clone)]
pub struct Robinson {
    pub lon_0: f64,
}

impl MapProjectionTrait for Robinson {
    fn proj_code(&self) -> &'static str {
        "robin"
    }
    fn display_name(&self) -> &'static str {
        "Robinson"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=robin +lon_0={}", self.lon_0)
    }
}

#[derive(Debug, Clone)]
pub struct Mollweide {
    pub lon_0: f64,
}

impl MapProjectionTrait for Mollweide {
    fn proj_code(&self) -> &'static str {
        "moll"
    }
    fn display_name(&self) -> &'static str {
        "Mollweide"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=moll +lon_0={}", self.lon_0)
    }
}

#[derive(Debug, Clone)]
pub struct Sinusoidal {
    pub lon_0: f64,
}

impl MapProjectionTrait for Sinusoidal {
    fn proj_code(&self) -> &'static str {
        "sinu"
    }
    fn display_name(&self) -> &'static str {
        "Sinusoidal"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=sinu +lon_0={}", self.lon_0)
    }
}

#[derive(Debug, Clone)]
pub struct Eckert4 {
    pub lon_0: f64,
}

impl MapProjectionTrait for Eckert4 {
    fn proj_code(&self) -> &'static str {
        "eck4"
    }
    fn display_name(&self) -> &'static str {
        "Eckert IV"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=eck4 +lon_0={}", self.lon_0)
    }
}

#[derive(Debug, Clone)]
pub struct NaturalEarth {
    pub lon_0: f64,
}

impl MapProjectionTrait for NaturalEarth {
    fn proj_code(&self) -> &'static str {
        "natearth"
    }
    fn display_name(&self) -> &'static str {
        "Natural Earth"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=natearth +lon_0={}", self.lon_0)
    }
}

// =============================================================================
// Interrupted
// =============================================================================

#[derive(Debug, Clone)]
pub struct Igh {
    pub lon_0: f64,
}

impl MapProjectionTrait for Igh {
    fn proj_code(&self) -> &'static str {
        "igh"
    }
    fn display_name(&self) -> &'static str {
        "Interrupted Goode Homolosine"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=igh +lon_0={}", self.lon_0)
    }
    fn slit_wkt(&self, epsilon: f64) -> Option<String> {
        Some(igh_slit_wkt(self.lon_0, epsilon))
    }
}

// =============================================================================
// Conic projections
// =============================================================================

#[derive(Debug, Clone)]
pub struct AlbersEqualArea {
    pub lon_0: f64,
    pub lat_0: f64,
    pub lat_1: f64,
    pub lat_2: f64,
}

impl MapProjectionTrait for AlbersEqualArea {
    fn proj_code(&self) -> &'static str {
        "aea"
    }
    fn display_name(&self) -> &'static str {
        "Albers Equal-Area"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!(
            "+proj=aea +lon_0={} +lat_0={} +lat_1={} +lat_2={}",
            self.lon_0, self.lat_0, self.lat_1, self.lat_2
        )
    }
    fn edge_segments(&self) -> [usize; 4] {
        [36, 36, 36, 36]
    }
}

#[derive(Debug, Clone)]
pub struct LambertConformalConic {
    pub lon_0: f64,
    pub lat_0: f64,
    pub lat_1: f64,
    pub lat_2: f64,
}

impl MapProjectionTrait for LambertConformalConic {
    fn proj_code(&self) -> &'static str {
        "lcc"
    }
    fn display_name(&self) -> &'static str {
        "Lambert Conformal Conic"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        format!(
            "+proj=lcc +lon_0={} +lat_0={} +lat_1={} +lat_2={}",
            self.lon_0, self.lat_0, self.lat_1, self.lat_2
        )
    }
    fn lat_bounds(&self) -> (f64, f64) {
        (-80.0, 84.0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [36, 36, 36, 36]
    }
}

// =============================================================================
// Standalone
// =============================================================================

#[derive(Debug, Clone)]
pub struct WinkelTripel {
    pub lon_0: f64,
}

impl MapProjectionTrait for WinkelTripel {
    fn proj_code(&self) -> &'static str {
        "wintri"
    }
    fn display_name(&self) -> &'static str {
        "Winkel Tripel"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, 0.0)
    }
    fn to_proj_str(&self) -> String {
        format!("+proj=wintri +lon_0={}", self.lon_0)
    }
    fn edge_segments(&self) -> [usize; 4] {
        [36, 36, 36, 36]
    }
}

// =============================================================================
// Unknown / fallback
// =============================================================================

#[derive(Debug, Clone)]
struct UnknownProj {
    raw: String,
    lon_0: f64,
    lat_0: f64,
}

impl MapProjectionTrait for UnknownProj {
    fn proj_code(&self) -> &'static str {
        "unknown"
    }
    fn display_name(&self) -> &'static str {
        "Unknown"
    }
    fn origin(&self) -> (f64, f64) {
        (self.lon_0, self.lat_0)
    }
    fn to_proj_str(&self) -> String {
        self.raw.clone()
    }
    fn visible_area_wkt(&self) -> Option<String> {
        None
    }
}

// =============================================================================
// Helpers
// =============================================================================

pub fn wrap_lon(lon: f64) -> f64 {
    ((lon + 180.0) % 360.0 + 360.0) % 360.0 - 180.0
}

fn antipodal_cap_wkt(lon_0: f64, lat_0: f64) -> String {
    let anti_lat = -lat_0;
    let radius = 1.0;

    if anti_lat + radius >= 90.0 {
        rectangle_wkt(-180.0, anti_lat - radius, 180.0, 90.0, [1, 1, 180, 1])
    } else if anti_lat - radius <= -90.0 {
        rectangle_wkt(-180.0, -90.0, 180.0, anti_lat + radius, [180, 1, 1, 1])
    } else {
        let anti_lon = wrap_lon(lon_0 + 180.0);
        hemisphere_polygon_wkt(anti_lon, anti_lat, radius)
    }
}

pub fn extract_proj_param_str<'a>(crs: &'a str, key: &str) -> Option<&'a str> {
    let start = crs.find(key)?;
    let after = &crs[start + key.len()..];
    let end = after.find([' ', '+']).unwrap_or(after.len());
    Some(&after[..end])
}

fn extract_f64_param(crs: &str, key: &str) -> Option<f64> {
    extract_proj_param_str(crs, key).and_then(|s| s.parse().ok())
}

pub fn rectangle_wkt(xmin: f64, ymin: f64, xmax: f64, ymax: f64, segments: [usize; 4]) -> String {
    let mut coords: Vec<String> = Vec::new();
    let [top, right, bottom, left] = segments.map(|s| s.max(1));
    for i in 0..top {
        let t = i as f64 / top as f64;
        coords.push(format!("{:.6} {:.6}", xmin + t * (xmax - xmin), ymax));
    }
    for i in 0..right {
        let t = i as f64 / right as f64;
        coords.push(format!("{:.6} {:.6}", xmax, ymax - t * (ymax - ymin)));
    }
    for i in 0..bottom {
        let t = i as f64 / bottom as f64;
        coords.push(format!("{:.6} {:.6}", xmax - t * (xmax - xmin), ymin));
    }
    for i in 0..left {
        let t = i as f64 / left as f64;
        coords.push(format!("{:.6} {:.6}", xmin, ymin + t * (ymax - ymin)));
    }
    coords.push(format!("{:.6} {:.6}", xmin, ymax));
    format!("POLYGON(({}))", coords.join(", "))
}

fn igh_slit_wkt(lon_0: f64, half_width: f64) -> String {
    let segs = [1, 36, 1, 36];
    let polygon_ring = |wkt: String| -> String { wkt.strip_prefix("POLYGON").unwrap().to_string() };

    let mut parts = Vec::new();

    for offset in [80.0, -20.0, -100.0] {
        let lon = wrap_lon(lon_0 + offset);
        parts.push(polygon_ring(rectangle_wkt(
            lon - half_width,
            -90.0,
            lon + half_width,
            0.0,
            segs,
        )));
    }

    let north = wrap_lon(lon_0 - 40.0);
    parts.push(polygon_ring(rectangle_wkt(
        north - half_width,
        0.0,
        north + half_width,
        90.0,
        segs,
    )));

    let seam = wrap_lon(lon_0 + 180.0);
    if (seam - (-180.0)).abs() > half_width && (seam - 180.0).abs() > half_width {
        parts.push(polygon_ring(rectangle_wkt(
            seam - half_width,
            -90.0,
            seam + half_width,
            90.0,
            segs,
        )));
    }

    format!("MULTIPOLYGON({})", parts.join(", "))
}

pub fn hemisphere_polygon_wkt(lon0: f64, lat0: f64, radius_deg: f64) -> String {
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
        lon_deg = ((lon_deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        raw_points.push((lon_deg, lat2.to_degrees()));
    }

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
    let crossings = find_antimeridian_crossings(&points);

    if includes_north_pole || includes_south_pole {
        build_pole_polygon(&points, includes_north_pole)
    } else if crossings.len() == 2 {
        build_antimeridian_multipolygon(&points)
    } else if crossings.is_empty() {
        build_simple_polygon(&points)
    } else {
        // Near-pole edge case: discrete sampling produces spurious antimeridian
        // crossings when points are at very high latitudes where longitude is
        // unstable. Route through the nearest pole to produce valid WKT.
        build_pole_polygon(&points, lat0 > 0.0)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn build_map_projection_trait(
        name: Option<&str>,
        properties: &HashMap<String, ParameterValue>,
    ) -> Option<Arc<dyn MapProjectionTrait>> {
        let name = name?;
        let obj: Arc<dyn MapProjectionTrait> = build_projection!(name, properties);
        Some(obj)
    }

    fn build_from_proj_str(crs: &str) -> Arc<dyn MapProjectionTrait> {
        let mut properties = HashMap::new();
        properties.insert(
            "target".to_string(),
            ParameterValue::String(crs.to_string()),
        );
        build_map_projection_trait(Some("crs"), &properties).unwrap()
    }

    #[test]
    fn from_proj_str_known_projections() {
        let proj = build_from_proj_str("+proj=ortho +lon_0=10 +lat_0=45");
        assert_eq!(proj.proj_code(), "ortho");
        assert_eq!(proj.origin(), (10.0, 45.0));

        let proj = build_from_proj_str("+proj=merc");
        assert_eq!(proj.proj_code(), "merc");
        assert_eq!(proj.origin(), (0.0, 0.0));

        let proj = build_from_proj_str("+proj=aea +lon_0=5 +lat_1=30 +lat_2=50");
        assert_eq!(proj.proj_code(), "aea");
        assert_eq!(
            proj.to_proj_str(),
            "+proj=aea +lon_0=5 +lat_0=0 +lat_1=30 +lat_2=50"
        );
    }

    #[test]
    fn new_from_target_albers() {
        let mut props = HashMap::new();
        props.insert(
            "target".to_string(),
            ParameterValue::String("+proj=aea +lon_0=5 +lat_0=10 +lat_1=30 +lat_2=50".to_string()),
        );
        let proj = build_map_projection_trait(Some("crs"), &props).unwrap();
        assert_eq!(proj.proj_code(), "aea");
        assert_eq!(proj.origin(), (5.0, 10.0));
        assert_eq!(
            proj.to_proj_str(),
            "+proj=aea +lon_0=5 +lat_0=10 +lat_1=30 +lat_2=50"
        );
    }

    #[test]
    fn new_named_albers_with_settings() {
        let mut props = HashMap::new();
        props.insert(
            "origin".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(-96.0),
                ArrayElement::Number(37.5),
            ]),
        );
        props.insert(
            "parallel".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(29.5), ArrayElement::Number(45.5)]),
        );
        let proj = build_map_projection_trait(Some("albers"), &props).unwrap();
        assert_eq!(proj.proj_code(), "aea");
        assert_eq!(proj.origin(), (-96.0, 37.5));
        assert_eq!(
            proj.to_proj_str(),
            "+proj=aea +lon_0=-96 +lat_0=37.5 +lat_1=29.5 +lat_2=45.5"
        );
    }

    #[test]
    fn from_proj_str_unknown_projection() {
        let proj = build_from_proj_str("+proj=fooproj +lon_0=5");
        assert_eq!(proj.proj_code(), "unknown");
        assert_eq!(proj.origin(), (5.0, 0.0));
        assert_eq!(proj.visible_area_wkt(), None);
    }

    #[test]
    fn build_from_proj_str_round_trip() {
        let proj = build_from_proj_str("+proj=robin +lon_0=10");
        let proj_str = proj.to_proj_str();
        let proj2 = build_from_proj_str(&proj_str);
        assert_eq!(proj.proj_code(), proj2.proj_code());
        assert_eq!(proj.to_proj_str(), proj2.to_proj_str());
    }

    #[test]
    fn visible_area_cylindrical() {
        let proj = build_from_proj_str("+proj=merc");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(wkt.starts_with("POLYGON(("));
        assert!(wkt.contains("85.000000"));
    }

    #[test]
    fn visible_area_azimuthal() {
        let proj = build_from_proj_str("+proj=ortho +lon_0=0 +lat_0=0");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(wkt.starts_with("POLYGON((") || wkt.starts_with("MULTIPOLYGON("));
    }

    #[test]
    fn slit_igh() {
        let proj = build_from_proj_str("+proj=igh");
        let slit = proj.slit_wkt(0.005).unwrap();
        assert!(slit.starts_with("MULTIPOLYGON("));
    }

    #[test]
    fn slit_default_antimeridian() {
        let proj = build_from_proj_str("+proj=robin +lon_0=10");
        let slit = proj.slit_wkt(0.005).unwrap();
        assert!(slit.starts_with("POLYGON(("));
        assert!(slit.contains("-170."));
    }

    #[test]
    fn slit_at_dateline_returns_none() {
        let proj = build_from_proj_str("+proj=robin +lon_0=0");
        assert!(proj.slit_wkt(0.005).is_none());
    }

    #[test]
    fn visible_area_gnomonic() {
        let proj = build_from_proj_str("+proj=gnom +lat_0=90 +lon_0=0");
        assert!(proj.visible_area_wkt().is_some());
    }

    #[test]
    fn visible_area_antimeridian_crossing() {
        let proj = build_from_proj_str("+proj=ortho +lat_0=0 +lon_0=150");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(
            wkt.starts_with("MULTIPOLYGON"),
            "lon_0=150 should cross antimeridian: {wkt}"
        );
    }

    #[test]
    fn visible_area_no_antimeridian_for_centered() {
        let proj = build_from_proj_str("+proj=ortho +lat_0=0 +lon_0=0");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(
            wkt.starts_with("POLYGON(("),
            "lon_0=0 should not cross antimeridian: {wkt}"
        );
    }

    #[test]
    fn visible_area_pole_routing() {
        let proj = build_from_proj_str("+proj=ortho +lat_0=52.36 +lon_0=150.90");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(
            wkt.starts_with("POLYGON(("),
            "pole case should produce POLYGON: {wkt}"
        );
    }

    #[test]
    fn visible_area_rectangle_always_polygon() {
        let proj = build_from_proj_str("+proj=robin +lon_0=-90");
        let wkt = proj.visible_area_wkt().unwrap();
        assert!(
            wkt.starts_with("POLYGON(("),
            "rectangle projections always produce POLYGON: {wkt}"
        );
    }

    #[test]
    fn hemisphere_polygon_always_valid_wkt() {
        // Verify all near-pole and antimeridian configurations produce valid WKT
        // (always starts with POLYGON or MULTIPOLYGON, never a bare ring).
        let cases = [
            (170.0, 80.0, 9.0),
            (-170.0, -80.0, 9.0),
            (180.0, 45.0, 60.0),
            (0.0, 89.0, 0.5),
            (179.0, 0.0, 88.0),
            (-179.0, 0.0, 88.0),
            (90.0, 89.5, 0.1),
        ];
        for (lon, lat, r) in cases {
            let wkt = hemisphere_polygon_wkt(lon, lat, r);
            assert!(
                wkt.starts_with("POLYGON((") || wkt.starts_with("MULTIPOLYGON("),
                "hemisphere_polygon_wkt({lon}, {lat}, {r}) produced invalid WKT: {wkt}"
            );
        }
    }

    #[test]
    fn seam_position() {
        let proj = build_from_proj_str("+proj=robin +lon_0=-90");
        let (lon_0, _) = proj.origin();
        let seam = wrap_lon(lon_0 + 180.0);
        assert!((seam - 90.0).abs() < 1e-6, "seam should be at 90°");
    }

    #[test]
    fn igh_slit_shift_with_lon_0() {
        let igh0 = build_from_proj_str("+proj=igh");
        let wkt0 = igh0.slit_wkt(0.005).unwrap();
        assert!(wkt0.starts_with("MULTIPOLYGON("), "{wkt0}");
        assert_eq!(wkt0.matches("((").count(), 4, "{wkt0}");

        let igh90 = build_from_proj_str("+proj=igh +lon_0=90");
        let wkt90 = igh90.slit_wkt(0.005).unwrap();
        assert_eq!(wkt90.matches("((").count(), 5, "{wkt90}");
        assert!(wkt90.contains("170.005"), "south slit near 170°: {wkt90}");
        assert!(wkt90.contains("70.005"), "south slit near 70°: {wkt90}");
        assert!(wkt90.contains("-10.005"), "south slit near -10°: {wkt90}");
        assert!(wkt90.contains("50.005"), "north slit near 50°: {wkt90}");
    }

    #[test]
    fn utm_from_proj_str() {
        let proj = build_from_proj_str("+proj=utm +zone=30");
        assert_eq!(proj.proj_code(), "utm");
        assert_eq!(proj.to_proj_str(), "+proj=utm +zone=30");

        let proj = build_from_proj_str("+proj=utm +zone=55 +south");
        assert_eq!(proj.to_proj_str(), "+proj=utm +zone=55 +south");
        let (lon, lat) = proj.origin();
        assert!((lon - 147.0).abs() < 1e-6);
        assert!(lat < 0.0);
    }

    #[test]
    fn utm_from_named_origin() {
        let mut props = HashMap::new();
        props.insert(
            "origin".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(5.0), ArrayElement::Number(52.0)]),
        );
        let proj = build_map_projection_trait(Some("utm"), &props).unwrap();
        assert_eq!(proj.proj_code(), "utm");
        assert_eq!(proj.to_proj_str(), "+proj=utm +zone=31");
    }

    #[test]
    fn utm_southern_hemisphere() {
        let mut props = HashMap::new();
        props.insert(
            "origin".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(151.0),
                ArrayElement::Number(-34.0),
            ]),
        );
        let proj = build_map_projection_trait(Some("utm"), &props).unwrap();
        assert_eq!(proj.to_proj_str(), "+proj=utm +zone=56 +south");
    }
}
