//! Geom trait and implementations
//!
//! This module provides a trait-based design for geometric objects (geoms) in ggsql.
//! Each geom type is implemented as its own struct, allowing for cleaner separation
//! of concerns and easier extensibility.
//!
//! # Architecture
//!
//! - `GeomType`: Enum for pattern matching and serialization
//! - `GeomTrait`: Trait defining geom behavior with default implementations
//! - `Geom`: Wrapper struct holding a boxed trait object
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::parser::geom::{Geom, GeomType};
//!
//! let point = Geom::point();
//! assert_eq!(point.geom_type(), GeomType::Point);
//! assert!(point.aesthetics().is_required("pos1"));
//! ```

use crate::plot::types::DefaultAestheticValue;
use crate::{naming, DataFrame, Mappings, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub mod types;

// Geom implementations
mod area;
mod arrow;
mod bar;
mod boxplot;
mod density;
mod histogram;
mod line;
mod path;
mod point;
mod polygon;
mod range;
mod ribbon;
mod rule;
mod segment;
mod smooth;
mod spatial;
pub(crate) mod stat_aggregate;
mod text;
mod tile;
mod violin;

// Re-export types
pub use types::{
    DefaultAesthetics, DefaultParamValue, ParamConstraint, ParamDefinition, StatResult,
};

// Re-export geom structs for direct access if needed
pub use area::Area;
pub use arrow::Arrow;
pub use bar::Bar;
pub use boxplot::Boxplot;
pub use density::Density;
pub use histogram::Histogram;
pub use line::Line;
pub use path::Path;
pub use point::Point;
pub use polygon::Polygon;
pub use range::Range;
pub use ribbon::Ribbon;
pub use rule::Rule;
pub use segment::Segment;
pub use smooth::Smooth;
pub use spatial::Spatial;
pub use text::Text;
pub use tile::Tile;
pub use violin::Violin;

use crate::plot::aesthetic::AestheticContext;
use crate::plot::projection::Projection;
use crate::plot::types::{ParameterValue, Schema};
use crate::reader::SqlDialect;

/// Enum of all geom types for pattern matching and serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GeomType {
    Point,
    Line,
    Path,
    Bar,
    Area,
    Tile,
    Polygon,
    Ribbon,
    Histogram,
    Density,
    Smooth,
    Boxplot,
    Violin,
    Text,
    Segment,
    Arrow,
    Rule,
    Range,
    Spatial,
}

impl std::fmt::Display for GeomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GeomType::Point => "point",
            GeomType::Line => "line",
            GeomType::Path => "path",
            GeomType::Bar => "bar",
            GeomType::Area => "area",
            GeomType::Tile => "tile",
            GeomType::Polygon => "polygon",
            GeomType::Ribbon => "ribbon",
            GeomType::Histogram => "histogram",
            GeomType::Density => "density",
            GeomType::Smooth => "smooth",
            GeomType::Boxplot => "boxplot",
            GeomType::Violin => "violin",
            GeomType::Text => "text",
            GeomType::Segment => "segment",
            GeomType::Arrow => "arrow",
            GeomType::Rule => "rule",
            GeomType::Range => "range",
            GeomType::Spatial => "spatial",
        };
        write!(f, "{}", s)
    }
}

/// Core trait for geom behavior
///
/// Each geom type implements this trait. Most methods have sensible defaults;
/// only `geom_type()` and `aesthetics()` are required implementations.
pub trait GeomTrait: std::fmt::Debug + std::fmt::Display + Send + Sync {
    /// Returns which geom type this is (for pattern matching)
    fn geom_type(&self) -> GeomType;

    /// Returns aesthetic information (REQUIRED - each geom is different)
    fn aesthetics(&self) -> DefaultAesthetics;

    /// Validate aesthetic mappings for this geom.
    ///
    /// Called during layer validation after basic checks (Required aesthetics, bidirectional)
    /// to allow geoms to implement custom validation logic (e.g., XOR constraints).
    ///
    /// Default: no additional validation
    fn validate_aesthetics(&self, _mappings: &crate::Mappings) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Returns default remappings for stat-computed columns and literals to aesthetics.
    ///
    /// Each tuple is (aesthetic_name, value) where value can be:
    /// - `DefaultAestheticValue::Column("stat_col")` - maps a stat column to the aesthetic
    /// - `DefaultAestheticValue::Number(0.0)` - maps a literal value to the aesthetic
    ///
    /// These defaults can be overridden by a REMAPPING clause.
    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics { defaults: &[] }
    }

    /// Returns valid stat column names that can be used in REMAPPING (early validation).
    ///
    /// These are the columns produced by the geom's stat transform and are used for
    /// early validation of REMAPPING clauses to provide helpful error messages.
    ///
    /// **IMPORTANT**: This static list must be kept in sync with the `stat_columns` field
    /// returned by `apply_stat_transform()` in `StatResult::Transformed`. These serve
    /// different but complementary purposes:
    ///
    /// - `valid_stat_columns()` (this method): Static compile-time list for early validation
    /// - `StatResult::stat_columns`: Dynamic runtime list of actual columns produced
    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &[]
    }

    /// Returns non-aesthetic parameters with their default values.
    ///
    /// These control stat behavior (e.g., bins for histogram).
    fn default_params(&self) -> &'static [ParamDefinition] {
        &[]
    }

    /// Returns aesthetics consumed as input by this geom's stat transform.
    ///
    /// Columns mapped to these aesthetics are used by the stat and don't need
    /// separate preservation in GROUP BY.
    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &[]
    }

    /// Whether the Aggregate stat applies to this geom, and which aesthetics
    /// stay as group keys when it does.
    ///
    /// - `None` — geom doesn't accept the `aggregate` SETTING. Used by the
    ///   statistical geoms (`histogram`, `density`, `smooth`, `boxplot`,
    ///   `violin`) that have their own bespoke stats.
    /// - `Some(&[])` — geom opts in; the stat groups by discrete mappings +
    ///   `PARTITION BY` only. Most non-statistical geoms.
    /// - `Some(&[<aes>, …])` — geom opts in *and* pins the listed aesthetics
    ///   as group keys regardless of their column's continuity. Used by
    ///   `line`/`area`/`ribbon` (domain axis) and `tile` (every spatial slot).
    ///
    /// `supports_aggregate()` is derived from this; geoms only override one
    /// method to opt in.
    fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        None
    }

    /// Whether this geom accepts the `aggregate` SETTING parameter.
    /// Derived from `aggregate_domain_aesthetics`; do not override.
    fn supports_aggregate(&self) -> bool {
        self.aggregate_domain_aesthetics().is_some()
    }

    /// Apply statistical transformation to the layer query.
    ///
    /// The default implementation:
    /// 1. Dispatches to the Aggregate stat when `supports_aggregate()` is
    ///    true and the `aggregate` parameter is set.
    /// 2. For each position axis declared as `Dummy` in `aesthetics()`,
    ///    post-wraps the result with a synthetic categorical column when
    ///    *no* aesthetic in the axis's family (e.g. `pos1`, `pos1min`,
    ///    `pos1max`, …) is mapped. The writer then suppresses the
    ///    (otherwise one-tick) axis. Geoms whose bespoke stat already
    ///    synthesises positions (`bar`, `boxplot`, `violin`, `histogram`,
    ///    …) override `apply_stat_transform` and are unaffected.
    #[allow(clippy::too_many_arguments)]
    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
        dialect: &dyn SqlDialect,
        aesthetic_ctx: &AestheticContext,
    ) -> Result<StatResult> {
        let mut result = if let (Some(domain), true) = (
            self.aggregate_domain_aesthetics(),
            has_aggregate_param(parameters),
        ) {
            stat_aggregate::apply(
                query,
                schema,
                aesthetics,
                group_by,
                parameters,
                dialect,
                aesthetic_ctx,
                domain,
            )?
        } else {
            StatResult::Identity
        };

        let aes = self.aesthetics();
        for axis in aes.dummy_axes() {
            if !types::axis_family_has_mapping(aesthetics, axis) {
                result = types::wrap_stat_with_dummy_axis(query, result, axis);
            }
        }

        Ok(result)
    }

    /// Post-process the DataFrame after stat query execution.
    ///
    /// This method is called after the stat transform query has been executed
    /// and allows geoms to modify the resulting data. The default implementation
    /// returns the data unchanged.
    ///
    /// Used by violin to scale the offset column to [0, 0.5 * width] using global
    /// max normalization before Vega-Lite rendering.
    fn post_process(
        &self,
        df: DataFrame,
        _parameters: &HashMap<String, ParameterValue>,
    ) -> Result<DataFrame> {
        Ok(df)
    }

    /// Apply coord-specific projection transformations to a layer query.
    ///
    /// Called after stat transforms, before data fetch. Each geom decides what
    /// projection means for its parameterization:
    /// - Spatial: ST_AsWKB (always), plus ST_Transform when Map coord has a CRS
    /// - Future geoms: rectangles transform corners, lines segmentize, etc.
    ///
    /// `columns` lists all column names in the query (for portable column
    /// replacement on backends that don't support `SELECT * REPLACE`).
    ///
    /// The default is a no-op (returns query unchanged).
    fn apply_projection(
        &self,
        query: &str,
        _projection: &Projection,
        _dialect: &dyn SqlDialect,
        _clip: bool,
        _columns: &[String],
    ) -> Result<String> {
        Ok(query.to_string())
    }

    /// Adjust layer mappings and parameters based on geom-specific logic.
    ///
    /// This method is called during layer execution to allow geoms to customize
    /// how aesthetics and parameters should be treated.
    /// This is called after parameters are validated, which allows for internal
    /// parameters.
    /// The default implementation does nothing.
    fn setup_layer(
        &self,
        _mappings: &mut Mappings,
        _parameters: &mut HashMap<String, ParameterValue>,
    ) -> Result<()> {
        Ok(())
    }

    /// Returns valid parameter names for SETTING clause.
    ///
    /// Combines supported aesthetics with non-aesthetic parameters from default_params.
    fn valid_settings(&self) -> Vec<&'static str> {
        let mut valid: Vec<&'static str> = self.aesthetics().supported();
        for param in self.default_params() {
            valid.push(param.name);
        }
        valid
    }
}

/// Project pos1/pos2 columns through the map CRS transform.
///
/// When the coordinate system is Map with a CRS, wraps the position columns
/// with ST_X/ST_Y(ST_Transform(ST_Point(pos1, pos2), source, target)). Returns
/// the query unchanged for non-map coords or when source == target.
pub(crate) fn project_position_columns(
    query: &str,
    projection: &Projection,
    dialect: &dyn SqlDialect,
    columns: &[String],
) -> Result<String> {
    use crate::plot::projection::coord::CoordKind;

    if projection.coord.coord_kind() != CoordKind::Map {
        return Ok(query.to_string());
    }
    let target = match projection.properties.get("target") {
        Some(ParameterValue::String(s)) => s.as_str(),
        _ => return Ok(query.to_string()),
    };
    let source = match projection.properties.get("source") {
        Some(ParameterValue::String(s)) => s.as_str(),
        _ => "EPSG:4326",
    };
    if source == target {
        return Ok(query.to_string());
    }

    let pos1 = naming::quote_ident(&naming::aesthetic_column("pos1"));
    let pos2 = naming::quote_ident(&naming::aesthetic_column("pos2"));
    let point_expr = format!("ST_Point({pos1}, {pos2})");
    let transformed = dialect.sql_st_transform(&point_expr, source, target);
    let proj_col = naming::quote_ident("__ggsql_proj_pt__");

    let inner = format!("SELECT *, {transformed} AS {proj_col} FROM ({query})");
    let x_expr = format!("ST_X({proj_col})");
    let y_expr = format!("ST_Y({proj_col})");

    // Build a column list that replaces pos1 and pos2 with projected values
    // and drops the temporary projected-point column.
    if columns.is_empty() {
        return Ok(format!(
            "SELECT {x_expr} AS {pos1}, {y_expr} AS {pos2}, * \
             FROM ({inner}) \"__ggsql_pp__\""
        ));
    }
    let select_list: Vec<String> = columns
        .iter()
        .map(|c| {
            let qc = naming::quote_ident(c);
            if qc == pos1 {
                format!("{x_expr} AS {pos1}")
            } else if qc == pos2 {
                format!("{y_expr} AS {pos2}")
            } else {
                qc
            }
        })
        .collect();
    Ok(format!(
        "SELECT {} FROM ({inner}) \"__ggsql_pp__\"",
        select_list.join(", ")
    ))
}

/// True when `parameters["aggregate"]` is set to a non-null string or array.
pub(crate) fn has_aggregate_param(parameters: &HashMap<String, ParameterValue>) -> bool {
    matches!(
        parameters.get("aggregate"),
        Some(ParameterValue::String(_)) | Some(ParameterValue::Array(_))
    )
}

/// Wrapper struct for geom trait objects
///
/// This provides a convenient interface for working with geoms while hiding
/// the complexity of trait objects.
#[derive(Clone)]
pub struct Geom(Arc<dyn GeomTrait>);

impl Geom {
    /// Create a Point geom
    pub fn point() -> Self {
        Self(Arc::new(Point))
    }

    /// Create a Line geom
    pub fn line() -> Self {
        Self(Arc::new(Line))
    }

    /// Create a Path geom
    pub fn path() -> Self {
        Self(Arc::new(Path))
    }

    /// Create a Bar geom
    pub fn bar() -> Self {
        Self(Arc::new(Bar))
    }

    /// Create an Area geom
    pub fn area() -> Self {
        Self(Arc::new(Area))
    }

    /// Create a Tile geom
    pub fn tile() -> Self {
        Self(Arc::new(Tile))
    }

    /// Create a Polygon geom
    pub fn polygon() -> Self {
        Self(Arc::new(Polygon))
    }

    /// Create a Ribbon geom
    pub fn ribbon() -> Self {
        Self(Arc::new(Ribbon))
    }

    /// Create a Histogram geom
    pub fn histogram() -> Self {
        Self(Arc::new(Histogram))
    }

    /// Create a Density geom
    pub fn density() -> Self {
        Self(Arc::new(Density))
    }

    /// Create a Smooth geom
    pub fn smooth() -> Self {
        Self(Arc::new(Smooth))
    }

    /// Create a Boxplot geom
    pub fn boxplot() -> Self {
        Self(Arc::new(Boxplot))
    }

    /// Create a Violin geom
    pub fn violin() -> Self {
        Self(Arc::new(Violin))
    }

    /// Create a Text geom
    pub fn text() -> Self {
        Self(Arc::new(Text))
    }

    /// Create a Segment geom
    pub fn segment() -> Self {
        Self(Arc::new(Segment))
    }

    /// Create an Arrow geom
    pub fn arrow() -> Self {
        Self(Arc::new(Arrow))
    }

    /// Create an Rule geom
    pub fn rule() -> Self {
        Self(Arc::new(Rule))
    }

    /// Create a Range geom
    pub fn range() -> Self {
        Self(Arc::new(Range))
    }

    /// Create a Spatial geom
    pub fn spatial() -> Self {
        Self(Arc::new(Spatial))
    }

    /// Create a Geom from a GeomType
    pub fn from_type(t: GeomType) -> Self {
        match t {
            GeomType::Point => Self::point(),
            GeomType::Line => Self::line(),
            GeomType::Path => Self::path(),
            GeomType::Bar => Self::bar(),
            GeomType::Area => Self::area(),
            GeomType::Tile => Self::tile(),
            GeomType::Polygon => Self::polygon(),
            GeomType::Ribbon => Self::ribbon(),
            GeomType::Histogram => Self::histogram(),
            GeomType::Density => Self::density(),
            GeomType::Smooth => Self::smooth(),
            GeomType::Boxplot => Self::boxplot(),
            GeomType::Violin => Self::violin(),
            GeomType::Text => Self::text(),
            GeomType::Segment => Self::segment(),
            GeomType::Arrow => Self::arrow(),
            GeomType::Rule => Self::rule(),
            GeomType::Range => Self::range(),
            GeomType::Spatial => Self::spatial(),
        }
    }

    /// Get the geom type
    pub fn geom_type(&self) -> GeomType {
        self.0.geom_type()
    }

    /// Get aesthetics information
    pub fn aesthetics(&self) -> DefaultAesthetics {
        self.0.aesthetics()
    }

    /// Get default remappings as explicitly declared by the geom.
    ///
    /// Most callers want [`implicit_default_remappings`], which also
    /// includes auto-derived entries for `Dummy` axes.
    pub fn default_remappings(&self) -> DefaultAesthetics {
        self.0.default_remappings()
    }

    /// Default remappings merged with auto-derived `(axis, Column(axis))`
    /// entries for every aesthetic declared as `Dummy` that isn't already
    /// covered by an explicit remapping. The merged list is what should be
    /// fed to the executor's rename pass.
    pub fn implicit_default_remappings(&self) -> Vec<(&'static str, DefaultAestheticValue)> {
        let explicit = self.0.default_remappings();
        let mut out: Vec<(&'static str, DefaultAestheticValue)> = explicit.defaults.to_vec();
        for axis in self.0.aesthetics().dummy_axes() {
            if !out.iter().any(|(name, _)| *name == axis) {
                out.push((axis, DefaultAestheticValue::Column(axis)));
            }
        }
        out
    }

    /// Get valid stat columns as explicitly declared by the geom.
    pub fn valid_stat_columns(&self) -> &'static [&'static str] {
        self.0.valid_stat_columns()
    }

    /// Valid stat columns merged with the axis names of every `Dummy`
    /// aesthetic declared by the geom. The executor uses this to validate
    /// REMAPPING targets.
    pub fn implicit_valid_stat_columns(&self) -> Vec<&'static str> {
        let explicit = self.0.valid_stat_columns();
        let mut out: Vec<&'static str> = explicit.to_vec();
        for axis in self.0.aesthetics().dummy_axes() {
            if !out.contains(&axis) {
                out.push(axis);
            }
        }
        out
    }

    /// Get default parameters
    pub fn default_params(&self) -> &'static [ParamDefinition] {
        self.0.default_params()
    }

    /// Get stat consumed aesthetics
    pub fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        self.0.stat_consumed_aesthetics()
    }

    /// Apply stat transform
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stat_transform(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &dyn Fn(&str) -> Result<DataFrame>,
        dialect: &dyn SqlDialect,
        aesthetic_ctx: &AestheticContext,
    ) -> Result<StatResult> {
        self.0.apply_stat_transform(
            query,
            schema,
            aesthetics,
            group_by,
            parameters,
            execute_query,
            dialect,
            aesthetic_ctx,
        )
    }

    /// Post-process DataFrame after stat query execution
    pub fn post_process(
        &self,
        df: DataFrame,
        parameters: &HashMap<String, ParameterValue>,
    ) -> Result<DataFrame> {
        self.0.post_process(df, parameters)
    }

    /// Apply coord-specific projection transformations
    pub fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        clip: bool,
        columns: &[String],
    ) -> Result<String> {
        self.0
            .apply_projection(query, projection, dialect, clip, columns)
    }

    /// Adjust layer mappings and parameters based on geom-specific logic
    pub fn setup_layer(
        &self,
        mappings: &mut Mappings,
        parameters: &mut HashMap<String, ParameterValue>,
    ) -> Result<()> {
        self.0.setup_layer(mappings, parameters)
    }

    /// Get valid settings
    pub fn valid_settings(&self) -> Vec<&'static str> {
        self.0.valid_settings()
    }

    /// Whether this geom accepts the `aggregate` SETTING parameter.
    pub fn supports_aggregate(&self) -> bool {
        self.0.supports_aggregate()
    }

    /// Aesthetics the Aggregate stat must keep as group keys rather than
    /// aggregating, even if their bound column is continuous. `None` when
    /// the geom doesn't accept the `aggregate` setting.
    pub fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        self.0.aggregate_domain_aesthetics()
    }

    /// Validate aesthetic mappings
    pub fn validate_aesthetics(&self, mappings: &Mappings) -> std::result::Result<(), String> {
        self.0.validate_aesthetics(mappings)
    }
}

impl std::fmt::Debug for Geom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geom::{:?}", self.geom_type())
    }
}

impl std::fmt::Display for Geom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for Geom {
    fn eq(&self, other: &Self) -> bool {
        self.geom_type() == other.geom_type()
    }
}

impl Eq for Geom {}

impl Serialize for Geom {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.geom_type().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Geom {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let geom_type = GeomType::deserialize(deserializer)?;
        Ok(Geom::from_type(geom_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geom_creation() {
        let point = Geom::point();
        assert_eq!(point.geom_type(), GeomType::Point);

        let line = Geom::line();
        assert_eq!(line.geom_type(), GeomType::Line);
    }

    #[test]
    fn test_geom_equality() {
        let p1 = Geom::point();
        let p2 = Geom::point();
        let l1 = Geom::line();

        assert_eq!(p1, p2);
        assert_ne!(p1, l1);
    }

    #[test]
    fn test_geom_display() {
        assert_eq!(format!("{}", Geom::point()), "point");
        assert_eq!(format!("{}", Geom::histogram()), "histogram");
    }

    #[test]
    fn test_geom_type_display() {
        assert_eq!(format!("{}", GeomType::Point), "point");
        assert_eq!(format!("{}", GeomType::Range), "range");
    }

    #[test]
    fn test_geom_from_type() {
        let geom = Geom::from_type(GeomType::Bar);
        assert_eq!(geom.geom_type(), GeomType::Bar);
    }

    #[test]
    fn test_geom_aesthetics() {
        let point = Geom::point();
        let aes = point.aesthetics();
        // Both axes are optional - omitted axes become dummy categorical axes.
        assert!(!aes.is_required("pos1"));
        assert!(!aes.is_required("pos2"));
    }

    #[test]
    fn test_geom_serialization() {
        let point = Geom::point();
        let json = serde_json::to_string(&point).unwrap();
        assert_eq!(json, "\"point\"");

        let deserialized: Geom = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.geom_type(), GeomType::Point);
    }

    #[test]
    fn test_default_remappings_are_in_aesthetics() {
        // Test that every aesthetic in default_remappings() exists in aesthetics().defaults
        // This ensures that remapped aesthetics are properly declared (usually as Delayed)

        let all_geom_types = [
            GeomType::Point,
            GeomType::Line,
            GeomType::Path,
            GeomType::Bar,
            GeomType::Area,
            GeomType::Tile,
            GeomType::Polygon,
            GeomType::Ribbon,
            GeomType::Histogram,
            GeomType::Density,
            GeomType::Smooth,
            GeomType::Boxplot,
            GeomType::Violin,
            GeomType::Text,
            GeomType::Segment,
            GeomType::Arrow,
            GeomType::Rule,
            GeomType::Range,
            GeomType::Spatial,
        ];

        // This test is rigged to trigger a compiler error when new variants are added.
        // Add the new layer to both the array above and as match arm below.
        let _exhaustive_check = |t: GeomType| match t {
            GeomType::Point
            | GeomType::Line
            | GeomType::Path
            | GeomType::Bar
            | GeomType::Area
            | GeomType::Tile
            | GeomType::Polygon
            | GeomType::Ribbon
            | GeomType::Histogram
            | GeomType::Density
            | GeomType::Smooth
            | GeomType::Boxplot
            | GeomType::Violin
            | GeomType::Text
            | GeomType::Segment
            | GeomType::Arrow
            | GeomType::Rule
            | GeomType::Range
            | GeomType::Spatial => {}
        };

        for geom_type in all_geom_types {
            let geom = Geom::from_type(geom_type);
            let remappings = geom.default_remappings();
            let aesthetics = geom.aesthetics();

            // Collect all aesthetic names from aesthetics().defaults
            let aesthetic_names: std::collections::HashSet<&str> =
                aesthetics.defaults.iter().map(|(name, _)| *name).collect();

            // Check each remapping name exists in aesthetics
            for (name, _) in remappings.defaults {
                assert!(
                    aesthetic_names.contains(name),
                    "Geom '{}' has '{}' in default_remappings() but not in aesthetics().defaults. \
                     Add it as DefaultAestheticValue::Delayed if it's a stat-produced aesthetic.",
                    geom_type,
                    name
                );
            }
        }
    }
}
