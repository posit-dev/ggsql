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
//! assert!(point.aesthetics().required.contains(&"x"));
//! ```

use crate::{DataFrame, Mappings, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub mod types;

// Geom implementations
mod abline;
mod area;
mod arrow;
mod bar;
mod boxplot;
mod density;
mod errorbar;
mod histogram;
mod hline;
mod label;
mod line;
mod path;
mod point;
mod polygon;
mod ribbon;
mod segment;
mod smooth;
mod text;
mod tile;
mod violin;
mod vline;

// Re-export types
pub use types::{DefaultParam, DefaultParamValue, GeomAesthetics, StatResult};

// Re-export aesthetic family utilities from the central module
pub use crate::plot::aesthetic::{get_aesthetic_family, AESTHETIC_FAMILIES};

// Re-export geom structs for direct access if needed
pub use abline::AbLine;
pub use area::Area;
pub use arrow::Arrow;
pub use bar::Bar;
pub use boxplot::Boxplot;
pub use density::Density;
pub use errorbar::ErrorBar;
pub use histogram::Histogram;
pub use hline::HLine;
pub use label::Label;
pub use line::Line;
pub use path::Path;
pub use point::Point;
pub use polygon::Polygon;
pub use ribbon::Ribbon;
pub use segment::Segment;
pub use smooth::Smooth;
pub use text::Text;
pub use tile::Tile;
pub use violin::Violin;
pub use vline::VLine;

use crate::plot::types::{DefaultAestheticValue, ParameterValue, Schema};

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
    Label,
    Segment,
    Arrow,
    HLine,
    VLine,
    AbLine,
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
            GeomType::Tile => "tile",
            GeomType::Polygon => "polygon",
            GeomType::Ribbon => "ribbon",
            GeomType::Histogram => "histogram",
            GeomType::Density => "density",
            GeomType::Smooth => "smooth",
            GeomType::Boxplot => "boxplot",
            GeomType::Violin => "violin",
            GeomType::Text => "text",
            GeomType::Label => "label",
            GeomType::Segment => "segment",
            GeomType::Arrow => "arrow",
            GeomType::HLine => "hline",
            GeomType::VLine => "vline",
            GeomType::AbLine => "abline",
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
    fn aesthetics(&self) -> GeomAesthetics;

    /// Returns default remappings for stat-computed columns and literals to aesthetics.
    ///
    /// Each tuple is (aesthetic_name, value) where value can be:
    /// - `DefaultAestheticValue::Column("stat_col")` - maps a stat column to the aesthetic
    /// - `DefaultAestheticValue::Number(0.0)` - maps a literal value to the aesthetic
    ///
    /// These defaults can be overridden by a REMAPPING clause.
    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[]
    }

    /// Returns valid stat column names that can be used in REMAPPING.
    ///
    /// These are the columns produced by the geom's stat transform.
    /// REMAPPING can only map these columns to aesthetics.
    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &[]
    }

    /// Returns non-aesthetic parameters with their default values.
    ///
    /// These control stat behavior (e.g., bins for histogram).
    fn default_params(&self) -> &'static [DefaultParam] {
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
    fn apply_stat_transform(
        &self,
        _query: &str,
        _schema: &Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        Ok(StatResult::Identity)
    }

    /// Returns valid parameter names for SETTING clause.
    ///
    /// Combines supported aesthetics with non-aesthetic parameters.
    fn valid_settings(&self) -> Vec<&'static str> {
        let mut valid: Vec<&'static str> = self.aesthetics().supported.to_vec();
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

    /// Create a Label geom
    pub fn label() -> Self {
        Self(Arc::new(Label))
    }

    /// Create a Segment geom
    pub fn segment() -> Self {
        Self(Arc::new(Segment))
    }

    /// Create an Arrow geom
    pub fn arrow() -> Self {
        Self(Arc::new(Arrow))
    }

    /// Create an HLine geom
    pub fn hline() -> Self {
        Self(Arc::new(HLine))
    }

    /// Create a VLine geom
    pub fn vline() -> Self {
        Self(Arc::new(VLine))
    }

    /// Create an AbLine geom
    pub fn abline() -> Self {
        Self(Arc::new(AbLine))
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
            GeomType::Tile => Self::tile(),
            GeomType::Polygon => Self::polygon(),
            GeomType::Ribbon => Self::ribbon(),
            GeomType::Histogram => Self::histogram(),
            GeomType::Density => Self::density(),
            GeomType::Smooth => Self::smooth(),
            GeomType::Boxplot => Self::boxplot(),
            GeomType::Violin => Self::violin(),
            GeomType::Text => Self::text(),
            GeomType::Label => Self::label(),
            GeomType::Segment => Self::segment(),
            GeomType::Arrow => Self::arrow(),
            GeomType::HLine => Self::hline(),
            GeomType::VLine => Self::vline(),
            GeomType::AbLine => Self::abline(),
            GeomType::ErrorBar => Self::errorbar(),
        }
    }

    /// Get the geom type
    pub fn geom_type(&self) -> GeomType {
        self.0.geom_type()
    }

    /// Get aesthetics information
    pub fn aesthetics(&self) -> GeomAesthetics {
        self.0.aesthetics()
    }

    /// Get default remappings
    pub fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        self.0.default_remappings()
    }

    /// Get valid stat columns
    pub fn valid_stat_columns(&self) -> &'static [&'static str] {
        self.0.valid_stat_columns()
    }

    /// Get default parameters
    pub fn default_params(&self) -> &'static [DefaultParam] {
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
    pub fn apply_stat_transform(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        self.0.apply_stat_transform(
            query,
            schema,
            aesthetics,
            group_by,
            parameters,
            execute_query,
        )
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
        assert!(aes.required.contains(&"x"));
        assert!(aes.required.contains(&"y"));
    }

    #[test]
    fn test_geom_serialization() {
        let point = Geom::point();
        let json = serde_json::to_string(&point).unwrap();
        assert_eq!(json, "\"point\"");

        let deserialized: Geom = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.geom_type(), GeomType::Point);
    }
}
