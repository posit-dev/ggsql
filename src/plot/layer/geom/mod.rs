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

use crate::{DataFrame, Mappings, Result};
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
mod errorbar;
mod histogram;
mod line;
mod linear;
mod path;
mod point;
mod polygon;
mod rect;
mod ribbon;
mod rule;
mod segment;
mod smooth;
mod text;
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
pub use errorbar::ErrorBar;
pub use histogram::Histogram;
pub use line::Line;
pub use linear::Linear;
pub use path::Path;
pub use point::Point;
pub use polygon::Polygon;
pub use rect::Rect;
pub use ribbon::Ribbon;
pub use rule::Rule;
pub use segment::Segment;
pub use smooth::Smooth;
pub use text::Text;
pub use violin::Violin;

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
    Rect,
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
    Linear,
    ErrorBar,
}

impl std::fmt::Display for GeomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GeomType::Point => "point",
            GeomType::Line => "line",
            GeomType::Path => "path",
            GeomType::Bar => "bar",
            GeomType::Area => "area",
            GeomType::Rect => "rect",
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
            GeomType::Linear => "linear",
            GeomType::ErrorBar => "errorbar",
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

    /// Check if this geom requires a statistical transformation
    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        false
    }

    /// Apply statistical transformation to the layer query.
    ///
    /// The default implementation returns identity (no transformation).
    #[allow(clippy::too_many_arguments)]
    fn apply_stat_transform(
        &self,
        _query: &str,
        _schema: &Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
        _dialect: &dyn SqlDialect,
    ) -> Result<StatResult> {
        Ok(StatResult::Identity)
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

    /// Create a Rect geom
    pub fn rect() -> Self {
        Self(Arc::new(Rect))
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

    /// Create an Linear geom
    pub fn linear() -> Self {
        Self(Arc::new(Linear))
    }

    /// Create an ErrorBar geom
    pub fn errorbar() -> Self {
        Self(Arc::new(ErrorBar))
    }

    /// Create a Geom from a GeomType
    pub fn from_type(t: GeomType) -> Self {
        match t {
            GeomType::Point => Self::point(),
            GeomType::Line => Self::line(),
            GeomType::Path => Self::path(),
            GeomType::Bar => Self::bar(),
            GeomType::Area => Self::area(),
            GeomType::Rect => Self::rect(),
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
            GeomType::Linear => Self::linear(),
            GeomType::ErrorBar => Self::errorbar(),
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

    /// Get default remappings
    pub fn default_remappings(&self) -> DefaultAesthetics {
        self.0.default_remappings()
    }

    /// Get valid stat columns
    pub fn valid_stat_columns(&self) -> &'static [&'static str] {
        self.0.valid_stat_columns()
    }

    /// Get default parameters
    pub fn default_params(&self) -> &'static [ParamDefinition] {
        self.0.default_params()
    }

    /// Get stat consumed aesthetics
    pub fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        self.0.stat_consumed_aesthetics()
    }

    /// Check if stat transform is needed
    pub fn needs_stat_transform(&self, aesthetics: &Mappings) -> bool {
        self.0.needs_stat_transform(aesthetics)
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
    ) -> Result<StatResult> {
        self.0.apply_stat_transform(
            query,
            schema,
            aesthetics,
            group_by,
            parameters,
            execute_query,
            dialect,
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

    /// Get valid settings
    pub fn valid_settings(&self) -> Vec<&'static str> {
        self.0.valid_settings()
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
        assert_eq!(format!("{}", GeomType::ErrorBar), "errorbar");
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
        assert!(aes.is_required("pos1"));
        assert!(aes.is_required("pos2"));
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
            GeomType::Rect,
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
            GeomType::Linear,
            GeomType::ErrorBar,
        ];

        // This test is rigged to trigger a compiler error when new variants are added.
        // Add the new layer to both the array above and as match arm below.
        let _exhaustive_check = |t: GeomType| match t {
            GeomType::Point
            | GeomType::Line
            | GeomType::Path
            | GeomType::Bar
            | GeomType::Area
            | GeomType::Rect
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
            | GeomType::Linear
            | GeomType::ErrorBar => {}
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
