//! AST (Abstract Syntax Tree) types for ggSQL specification
//!
//! This module defines the typed AST structures that represent parsed ggSQL queries.
//! The AST is built from the tree-sitter CST (Concrete Syntax Tree) and provides
//! a more convenient, typed interface for working with ggSQL specifications.
//!
//! # AST Structure
//!
//! ```text
//! VizSpec
//! ├─ global_mapping: GlobalMapping  (from VISUALISE clause mappings)
//! ├─ source: Option<String>         (optional, from VISUALISE FROM clause)
//! ├─ layers: Vec<Layer>             (1+ LayerNode, one per DRAW clause)
//! ├─ scales: Vec<Scale>             (0+ ScaleNode, one per SCALE clause)
//! ├─ facet: Option<Facet>           (optional, from FACET clause)
//! ├─ coord: Option<Coord>           (optional, from COORD clause)
//! ├─ labels: Option<Labels>         (optional, merged from LABEL clauses)
//! ├─ guides: Vec<Guide>             (0+ GuideNode, one per GUIDE clause)
//! └─ theme: Option<Theme>           (optional, from THEME clause)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Result;

/// Complete ggSQL visualization specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VizSpec {
    /// Global aesthetic mappings (from VISUALISE clause)
    pub global_mapping: GlobalMapping,
    /// FROM source name (CTE or table) when using VISUALISE FROM syntax
    pub source: Option<String>,
    /// Visual layers (one per DRAW clause)
    pub layers: Vec<Layer>,
    /// Scale configurations (one per SCALE clause)
    pub scales: Vec<Scale>,
    /// Faceting specification (from FACET clause)
    pub facet: Option<Facet>,
    /// Coordinate system (from COORD clause)
    pub coord: Option<Coord>,
    /// Text labels (merged from all LABEL clauses)
    pub labels: Option<Labels>,
    /// Guide configurations (one per GUIDE clause)
    pub guides: Vec<Guide>,
    /// Theme styling (from THEME clause)
    pub theme: Option<Theme>,
}

/// Global mapping specification from VISUALISE clause
///
/// Represents the aesthetic mappings declared at the top level in the VISUALISE clause.
/// These serve as defaults for all layers, which can override or add to them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum GlobalMapping {
    /// No global mapping specified - layers must define all aesthetics
    #[default]
    Empty,
    /// Wildcard (*) - resolve all columns at execution time
    Wildcard,
    /// Explicit list of mappings (may include implicit entries)
    Mappings(Vec<GlobalMappingItem>),
}

/// Individual mapping item in global mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GlobalMappingItem {
    /// Explicit mapping: `date AS x` → column "date" maps to aesthetic "x"
    Explicit { column: String, aesthetic: String },
    /// Implicit mapping: `x` → column "x" maps to aesthetic "x"
    Implicit { name: String },
}

/// Data source for a layer (from MAPPING ... FROM clause)
///
/// Allows layers to specify their own data source instead of using
/// the global data from the main SQL query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerSource {
    /// CTE or table name (unquoted identifier)
    Identifier(String),
    /// File path (quoted string like 'data.csv')
    FilePath(String),
}

impl LayerSource {
    /// Returns the source as a string reference
    pub fn as_str(&self) -> &str {
        match self {
            LayerSource::Identifier(s) => s,
            LayerSource::FilePath(s) => s,
        }
    }

    /// Returns true if this is a file path source
    pub fn is_file(&self) -> bool {
        matches!(self, LayerSource::FilePath(_))
    }
}

/// A single visualization layer (from DRAW clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Geometric object type
    pub geom: Geom,
    /// Aesthetic mappings (aesthetic → column or literal)
    pub aesthetics: HashMap<String, AestheticValue>,
    /// Geom parameters (not aesthetic mappings)
    pub parameters: HashMap<String, ParameterValue>,
    /// Optional data source for this layer (from MAPPING ... FROM)
    pub source: Option<LayerSource>,
    /// Optional filter expression for this layer
    pub filter: Option<FilterExpression>,
    /// Columns for grouping/partitioning (from PARTITION BY clause)
    pub partition_by: Vec<String>,
}

/// Raw SQL filter expression for layer-specific filtering (from FILTER clause)
///
/// This stores the raw SQL WHERE clause text verbatim, which is passed directly
/// to the database backend. This allows any valid SQL WHERE expression to be used.
///
/// Example filter values:
/// - `"x > 10"`
/// - `"region = 'North' AND year >= 2020"`
/// - `"category IN ('A', 'B', 'C')"`
/// - `"name LIKE '%test%'"`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterExpression(pub String);

impl FilterExpression {
    /// Create a new filter expression from raw SQL text
    pub fn new(sql: impl Into<String>) -> Self {
        Self(sql.into())
    }

    /// Get the raw SQL text
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the raw SQL text
    pub fn into_string(self) -> String {
        self.0
    }
}

/// Geometric object types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Geom {
    // Basic geoms
    Point,
    Line,
    Path,
    Bar,
    Col,
    Area,
    Tile,
    Polygon,
    Ribbon,

    // Statistical geoms
    Histogram,
    Density,
    Smooth,
    Boxplot,
    Violin,

    // Annotation geoms
    Text,
    Label,
    Segment,
    Arrow,
    HLine,
    VLine,
    AbLine,
    ErrorBar,
}

/// Aesthetic information for a geom type
#[derive(Debug, Clone, Copy)]
pub struct GeomAesthetics {
    /// All aesthetics this geom type supports
    pub supported: &'static [&'static str],
    /// Aesthetics required for this geom type to be valid
    pub required: &'static [&'static str],
}

impl std::fmt::Display for Geom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Geom::Point => "point",
            Geom::Line => "line",
            Geom::Path => "path",
            Geom::Bar => "bar",
            Geom::Col => "col",
            Geom::Area => "area",
            Geom::Tile => "tile",
            Geom::Polygon => "polygon",
            Geom::Ribbon => "ribbon",
            Geom::Histogram => "histogram",
            Geom::Density => "density",
            Geom::Smooth => "smooth",
            Geom::Boxplot => "boxplot",
            Geom::Violin => "violin",
            Geom::Text => "text",
            Geom::Label => "label",
            Geom::Segment => "segment",
            Geom::Arrow => "arrow",
            Geom::HLine => "hline",
            Geom::VLine => "vline",
            Geom::AbLine => "abline",
            Geom::ErrorBar => "errorbar",
        };
        write!(f, "{}", s)
    }
}

impl Geom {
    /// Returns aesthetic information for this geom type.
    /// Includes both supported aesthetics (for wildcard mapping) and
    /// required aesthetics (for validation).
    pub fn aesthetics(&self) -> GeomAesthetics {
        match self {
            // Position geoms
            Geom::Point => GeomAesthetics {
                supported: &[
                    "x", "y", "color", "colour", "fill", "size", "shape", "opacity",
                ],
                required: &["x", "y"],
            },
            Geom::Line => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y"],
            },
            Geom::Path => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y"],
            },
            Geom::Bar => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "width", "opacity"],
                required: &["x", "y"],
            },
            Geom::Col => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "width", "opacity"],
                required: &["x", "y"],
            },
            Geom::Area => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
            },
            Geom::Tile => GeomAesthetics {
                supported: &[
                    "x", "y", "color", "colour", "fill", "width", "height", "opacity",
                ],
                required: &["x", "y"],
            },
            Geom::Polygon => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
            },
            Geom::Ribbon => GeomAesthetics {
                supported: &["x", "ymin", "ymax", "color", "colour", "fill", "opacity"],
                required: &["x", "ymin", "ymax"],
            },

            // Statistical geoms
            Geom::Histogram => GeomAesthetics {
                supported: &["x", "color", "colour", "fill", "opacity"],
                required: &["x"],
            },
            Geom::Density => GeomAesthetics {
                supported: &["x", "color", "colour", "fill", "opacity"],
                required: &["x"],
            },
            Geom::Smooth => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "linetype", "opacity"],
                required: &["x", "y"],
            },
            Geom::Boxplot => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
            },
            Geom::Violin => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
            },

            // Annotation geoms
            Geom::Text => GeomAesthetics {
                supported: &[
                    "x", "y", "label", "color", "colour", "size", "opacity", "family", "fontface",
                    "hjust", "vjust",
                ],
                required: &["x", "y"],
            },
            Geom::Label => GeomAesthetics {
                supported: &[
                    "x", "y", "label", "color", "colour", "fill", "size", "opacity", "family",
                    "fontface", "hjust", "vjust",
                ],
                required: &["x", "y"],
            },
            Geom::Segment => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "xend",
                    "yend",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y", "xend", "yend"],
            },
            Geom::Arrow => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "xend",
                    "yend",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y", "xend", "yend"],
            },
            Geom::HLine => GeomAesthetics {
                supported: &[
                    "yintercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["yintercept"],
            },
            Geom::VLine => GeomAesthetics {
                supported: &[
                    "xintercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["xintercept"],
            },
            Geom::AbLine => GeomAesthetics {
                supported: &[
                    "slope",
                    "intercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["slope", "intercept"],
            },
            Geom::ErrorBar => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "ymin",
                    "ymax",
                    "xmin",
                    "xmax",
                    "color",
                    "colour",
                    "linewidth",
                    "opacity",
                ],
                required: &[],
            },
        }
    }
}

/// Value for aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AestheticValue {
    /// Column reference (unquoted identifier)
    Column(String),
    /// Literal value (quoted string, number, or boolean)
    Literal(LiteralValue),
}

/// Literal values in aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Value for geom parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Scale configuration (from SCALE clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    pub scale_type: Option<ScaleType>,
    /// Scale properties
    pub properties: HashMap<String, ScalePropertyValue>,
}

/// Scale types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScaleType {
    // Continuous scales
    Linear,
    Log10,
    Log,
    Log2,
    Sqrt,
    Reverse,

    // Discrete scales
    Ordinal,
    Categorical,

    // Temporal scales
    Date,
    DateTime,
    Time,

    // Color palettes
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
    Diverging,
    Sequential,
}

/// Values for scale properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
}

/// Elements in arrays
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrayElement {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Faceting specification (from FACET clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Facet {
    /// FACET WRAP variables
    Wrap {
        variables: Vec<String>,
        scales: FacetScales,
    },
    /// FACET rows BY cols
    Grid {
        rows: Vec<String>,
        cols: Vec<String>,
        scales: FacetScales,
    },
}

/// Scale sharing options for facets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FacetScales {
    Fixed,
    Free,
    FreeX,
    FreeY,
}

/// Coordinate system (from COORD clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coord {
    /// Coordinate system type
    pub coord_type: CoordType,
    /// Coordinate-specific options
    pub properties: HashMap<String, CoordPropertyValue>,
}

/// Coordinate system types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoordType {
    Cartesian,
    Polar,
    Flip,
    Fixed,
    Trans,
    Map,
    QuickMap,
}

/// Values for coordinate properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoordPropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
}

/// Text labels (from LABELS clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    /// Label assignments (label type → text)
    pub labels: HashMap<String, String>,
}

/// Guide configuration (from GUIDE clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    /// The aesthetic this guide applies to
    pub aesthetic: String,
    /// Guide type
    pub guide_type: Option<GuideType>,
    /// Guide properties
    pub properties: HashMap<String, GuidePropertyValue>,
}

/// Guide types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuideType {
    Legend,
    ColorBar,
    Axis,
    None,
}

/// Values for guide properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuidePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Theme styling (from THEME clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    /// Base theme style
    pub style: Option<String>,
    /// Theme property overrides
    pub properties: HashMap<String, ThemePropertyValue>,
}

/// Values for theme properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThemePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

impl VizSpec {
    /// Create a new empty VizSpec
    pub fn new() -> Self {
        Self {
            global_mapping: GlobalMapping::Empty,
            source: None,
            layers: Vec::new(),
            scales: Vec::new(),
            facet: None,
            coord: None,
            labels: None,
            guides: Vec::new(),
            theme: None,
        }
    }

    /// Create a new VizSpec with the given global mapping
    pub fn with_global_mapping(global_mapping: GlobalMapping) -> Self {
        Self {
            global_mapping,
            source: None,
            layers: Vec::new(),
            scales: Vec::new(),
            facet: None,
            coord: None,
            labels: None,
            guides: Vec::new(),
            theme: None,
        }
    }

    /// Check if the spec has any layers
    pub fn has_layers(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Get the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Find a scale for a specific aesthetic
    pub fn find_scale(&self, aesthetic: &str) -> Option<&Scale> {
        self.scales
            .iter()
            .find(|scale| scale.aesthetic == aesthetic)
    }

    /// Find a guide for a specific aesthetic
    pub fn find_guide(&self, aesthetic: &str) -> Option<&Guide> {
        self.guides
            .iter()
            .find(|guide| guide.aesthetic == aesthetic)
    }

    /// Resolve global mappings into layer aesthetics.
    ///
    /// For each layer, global mappings are merged as defaults.
    /// Layer-specific MAPPING clauses override global mappings.
    ///
    /// For wildcard (`VISUALISE *`), columns are mapped to aesthetics
    /// based on what each layer's geom type supports.
    pub fn resolve_global_mappings(&mut self, available_columns: &[&str]) -> Result<()> {
        // Handle non-wildcard cases first (same for all layers)
        let explicit_mappings: HashMap<String, AestheticValue> = match &self.global_mapping {
            GlobalMapping::Empty => HashMap::new(),
            GlobalMapping::Wildcard => HashMap::new(), // Handled per-layer below
            GlobalMapping::Mappings(items) => items
                .iter()
                .map(|item| match item {
                    GlobalMappingItem::Explicit { column, aesthetic } => {
                        (aesthetic.clone(), AestheticValue::Column(column.clone()))
                    }
                    GlobalMappingItem::Implicit { name } => {
                        (name.clone(), AestheticValue::Column(name.clone()))
                    }
                })
                .collect(),
        };

        // For each layer, merge mappings (layer overrides global)
        for layer in &mut self.layers {
            // For wildcard, build mappings based on this geom's supported aesthetics
            let base_aesthetics: HashMap<String, AestheticValue> =
                if matches!(self.global_mapping, GlobalMapping::Wildcard) {
                    let supported = layer.geom.aesthetics().supported;
                    available_columns
                        .iter()
                        .filter(|col| supported.contains(col))
                        .map(|col| (col.to_string(), AestheticValue::Column(col.to_string())))
                        .collect()
                } else {
                    explicit_mappings.clone()
                };

            // Merge: layer aesthetics override global
            for (aesthetic, value) in base_aesthetics {
                layer.aesthetics.entry(aesthetic).or_insert(value);
            }
        }

        Ok(())
    }
}

impl Layer {
    /// Create a new layer with the given geom
    pub fn new(geom: Geom) -> Self {
        Self {
            geom,
            aesthetics: HashMap::new(),
            parameters: HashMap::new(),
            source: None,
            filter: None,
            partition_by: Vec::new(),
        }
    }

    /// Set the filter expression
    pub fn with_filter(mut self, filter: FilterExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the data source for this layer
    pub fn with_source(mut self, source: LayerSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Add an aesthetic mapping
    pub fn with_aesthetic(mut self, aesthetic: String, value: AestheticValue) -> Self {
        self.aesthetics.insert(aesthetic, value);
        self
    }

    /// Add a parameter
    pub fn with_parameter(mut self, parameter: String, value: ParameterValue) -> Self {
        self.parameters.insert(parameter, value);
        self
    }

    /// Set the partition columns for grouping
    pub fn with_partition_by(mut self, columns: Vec<String>) -> Self {
        self.partition_by = columns;
        self
    }

    /// Get a column reference from an aesthetic, if it's mapped to a column
    pub fn get_column(&self, aesthetic: &str) -> Option<&str> {
        match self.aesthetics.get(aesthetic) {
            Some(AestheticValue::Column(col)) => Some(col),
            _ => None,
        }
    }

    /// Get a literal value from an aesthetic, if it's mapped to a literal
    pub fn get_literal(&self, aesthetic: &str) -> Option<&LiteralValue> {
        match self.aesthetics.get(aesthetic) {
            Some(AestheticValue::Literal(lit)) => Some(lit),
            _ => None,
        }
    }

    /// Check if this layer has the required aesthetics for its geom
    pub fn validate_required_aesthetics(&self) -> std::result::Result<(), String> {
        for aesthetic in self.geom.aesthetics().required {
            if !self.aesthetics.contains_key(*aesthetic) {
                return Err(format!(
                    "Geom '{}' requires aesthetic '{}' but it was not provided",
                    self.geom, aesthetic
                ));
            }
        }

        Ok(())
    }
}

impl Default for VizSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AestheticValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AestheticValue::Column(col) => write!(f, "{}", col),
            AestheticValue::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

impl std::fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralValue::String(s) => write!(f, "'{}'", s),
            LiteralValue::Number(n) => write!(f, "{}", n),
            LiteralValue::Boolean(b) => write!(f, "{}", b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viz_spec_creation() {
        let spec = VizSpec::new();
        assert_eq!(spec.global_mapping, GlobalMapping::Empty);
        assert_eq!(spec.layers.len(), 0);
        assert!(!spec.has_layers());
        assert_eq!(spec.layer_count(), 0);
    }

    #[test]
    fn test_viz_spec_with_global_mapping() {
        let mapping = GlobalMapping::Mappings(vec![
            GlobalMappingItem::Explicit {
                column: "date".to_string(),
                aesthetic: "x".to_string(),
            },
            GlobalMappingItem::Implicit {
                name: "y".to_string(),
            },
        ]);
        let spec = VizSpec::with_global_mapping(mapping.clone());
        assert_eq!(spec.global_mapping, mapping);
    }

    #[test]
    fn test_global_mapping_wildcard() {
        let spec = VizSpec::with_global_mapping(GlobalMapping::Wildcard);
        assert_eq!(spec.global_mapping, GlobalMapping::Wildcard);
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::Column("revenue".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(LiteralValue::String("blue".to_string())),
            );

        assert_eq!(layer.geom, Geom::Point);
        assert_eq!(layer.get_column("x"), Some("date"));
        assert_eq!(layer.get_column("y"), Some("revenue"));
        assert!(matches!(layer.get_literal("color"), Some(LiteralValue::String(s)) if s == "blue"));
        assert!(layer.filter.is_none());
    }

    #[test]
    fn test_layer_with_filter() {
        let filter = FilterExpression::new("year > 2020");
        let layer = Layer::new(Geom::Point).with_filter(filter);
        assert!(layer.filter.is_some());
        assert_eq!(layer.filter.as_ref().unwrap().as_str(), "year > 2020");
    }

    #[test]
    fn test_layer_validation() {
        let valid_point = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));

        assert!(valid_point.validate_required_aesthetics().is_ok());

        let invalid_point = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()));

        assert!(invalid_point.validate_required_aesthetics().is_err());

        let valid_ribbon = Layer::new(Geom::Ribbon)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic(
                "ymin".to_string(),
                AestheticValue::Column("ymin".to_string()),
            )
            .with_aesthetic(
                "ymax".to_string(),
                AestheticValue::Column("ymax".to_string()),
            );

        assert!(valid_ribbon.validate_required_aesthetics().is_ok());
    }

    #[test]
    fn test_viz_spec_layer_operations() {
        let mut spec = VizSpec::new();

        let layer1 = Layer::new(Geom::Point);
        let layer2 = Layer::new(Geom::Line);

        spec.layers.push(layer1);
        spec.layers.push(layer2);

        assert!(spec.has_layers());
        assert_eq!(spec.layer_count(), 2);
        assert_eq!(spec.layers[0].geom, Geom::Point);
        assert_eq!(spec.layers[1].geom, Geom::Line);
    }

    #[test]
    fn test_aesthetic_value_display() {
        let column = AestheticValue::Column("sales".to_string());
        let string_lit = AestheticValue::Literal(LiteralValue::String("blue".to_string()));
        let number_lit = AestheticValue::Literal(LiteralValue::Number(3.53));
        let bool_lit = AestheticValue::Literal(LiteralValue::Boolean(true));

        assert_eq!(format!("{}", column), "sales");
        assert_eq!(format!("{}", string_lit), "'blue'");
        assert_eq!(format!("{}", number_lit), "3.53");
        assert_eq!(format!("{}", bool_lit), "true");
    }

    #[test]
    fn test_geom_display() {
        assert_eq!(format!("{}", Geom::Point), "point");
        assert_eq!(format!("{}", Geom::Histogram), "histogram");
        assert_eq!(format!("{}", Geom::ErrorBar), "errorbar");
    }

    // ========================================
    // Global Mapping Resolution Tests
    // ========================================

    #[test]
    fn test_explicit_global_mapping_resolution() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Mappings(vec![
            GlobalMappingItem::Explicit {
                column: "date".to_string(),
                aesthetic: "x".to_string(),
            },
            GlobalMappingItem::Explicit {
                column: "revenue".to_string(),
                aesthetic: "y".to_string(),
            },
        ]);
        spec.layers.push(Layer::new(Geom::Point));
        spec.layers.push(Layer::new(Geom::Line));

        spec.resolve_global_mappings(&["date", "revenue", "region"])
            .unwrap();

        // Both layers should have x and y aesthetics
        assert_eq!(spec.layers[0].aesthetics.len(), 2);
        assert_eq!(spec.layers[1].aesthetics.len(), 2);
        assert!(matches!(
            spec.layers[0].aesthetics.get("x"),
            Some(AestheticValue::Column(c)) if c == "date"
        ));
        assert!(matches!(
            spec.layers[0].aesthetics.get("y"),
            Some(AestheticValue::Column(c)) if c == "revenue"
        ));
    }

    #[test]
    fn test_implicit_global_mapping_resolution() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Mappings(vec![
            GlobalMappingItem::Implicit {
                name: "x".to_string(),
            },
            GlobalMappingItem::Implicit {
                name: "y".to_string(),
            },
        ]);
        spec.layers.push(Layer::new(Geom::Point));

        spec.resolve_global_mappings(&["x", "y", "z"]).unwrap();

        assert!(matches!(
            spec.layers[0].aesthetics.get("x"),
            Some(AestheticValue::Column(c)) if c == "x"
        ));
        assert!(matches!(
            spec.layers[0].aesthetics.get("y"),
            Some(AestheticValue::Column(c)) if c == "y"
        ));
    }

    #[test]
    fn test_wildcard_global_mapping_resolution() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Wildcard;
        spec.layers.push(Layer::new(Geom::Point));

        // Point geom supports: x, y, color, size, shape, opacity, etc.
        // Columns "x", "y", "color" match supported aesthetics
        // Columns "date", "revenue" do NOT match any supported aesthetic
        spec.resolve_global_mappings(&["x", "y", "color", "date", "revenue"])
            .unwrap();

        // Should only map columns that match geom's supported aesthetics
        assert_eq!(spec.layers[0].aesthetics.len(), 3);
        assert!(spec.layers[0].aesthetics.contains_key("x"));
        assert!(spec.layers[0].aesthetics.contains_key("y"));
        assert!(spec.layers[0].aesthetics.contains_key("color"));
        assert!(!spec.layers[0].aesthetics.contains_key("date"));
        assert!(!spec.layers[0].aesthetics.contains_key("revenue"));
    }

    #[test]
    fn test_wildcard_different_geoms_get_different_aesthetics() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Wildcard;
        spec.layers.push(Layer::new(Geom::Point)); // supports size, shape
        spec.layers.push(Layer::new(Geom::Line)); // supports linetype, linewidth

        spec.resolve_global_mappings(&["x", "y", "size", "linetype"])
            .unwrap();

        // Point layer should get x, y, size (not linetype)
        assert!(spec.layers[0].aesthetics.contains_key("size"));
        assert!(!spec.layers[0].aesthetics.contains_key("linetype"));

        // Line layer should get x, y, linetype (not size)
        assert!(spec.layers[1].aesthetics.contains_key("linetype"));
        assert!(!spec.layers[1].aesthetics.contains_key("size"));
    }

    #[test]
    fn test_layer_mapping_overrides_global() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Mappings(vec![GlobalMappingItem::Explicit {
            column: "date".to_string(),
            aesthetic: "x".to_string(),
        }]);

        let mut layer = Layer::new(Geom::Point);
        layer.aesthetics.insert(
            "x".to_string(),
            AestheticValue::Column("other_date".to_string()),
        );
        spec.layers.push(layer);

        spec.resolve_global_mappings(&["date", "other_date"])
            .unwrap();

        // Layer's explicit mapping should override global
        assert!(matches!(
            spec.layers[0].aesthetics.get("x"),
            Some(AestheticValue::Column(c)) if c == "other_date"
        ));
    }

    #[test]
    fn test_empty_global_mapping_no_change() {
        let mut spec = VizSpec::new();
        spec.global_mapping = GlobalMapping::Empty;

        let mut layer = Layer::new(Geom::Point);
        layer
            .aesthetics
            .insert("x".to_string(), AestheticValue::Column("col".to_string()));
        spec.layers.push(layer);

        spec.resolve_global_mappings(&["col"]).unwrap();

        // Layer should be unchanged
        assert_eq!(spec.layers[0].aesthetics.len(), 1);
    }

    #[test]
    fn test_geom_aesthetics() {
        // Point geom
        let point = Geom::Point.aesthetics();
        assert!(point.supported.contains(&"x"));
        assert!(point.supported.contains(&"size"));
        assert!(point.supported.contains(&"shape"));
        assert!(!point.supported.contains(&"linetype"));
        assert_eq!(point.required, &["x", "y"]);

        // Line geom
        let line = Geom::Line.aesthetics();
        assert!(line.supported.contains(&"linetype"));
        assert!(line.supported.contains(&"linewidth"));
        assert!(!line.supported.contains(&"size"));
        assert_eq!(line.required, &["x", "y"]);

        // Bar geom
        let bar = Geom::Bar.aesthetics();
        assert!(bar.supported.contains(&"fill"));
        assert!(bar.supported.contains(&"width"));
        assert_eq!(bar.required, &["x", "y"]);

        // Text geom
        let text = Geom::Text.aesthetics();
        assert!(text.supported.contains(&"label"));
        assert!(text.supported.contains(&"family"));
        assert_eq!(text.required, &["x", "y"]);

        // Statistical geoms only require x
        assert_eq!(Geom::Histogram.aesthetics().required, &["x"]);
        assert_eq!(Geom::Density.aesthetics().required, &["x"]);

        // Ribbon requires ymin/ymax
        assert_eq!(Geom::Ribbon.aesthetics().required, &["x", "ymin", "ymax"]);

        // Segment/arrow require endpoints
        assert_eq!(
            Geom::Segment.aesthetics().required,
            &["x", "y", "xend", "yend"]
        );

        // Reference lines
        assert_eq!(Geom::HLine.aesthetics().required, &["yintercept"]);
        assert_eq!(Geom::VLine.aesthetics().required, &["xintercept"]);
        assert_eq!(Geom::AbLine.aesthetics().required, &["slope", "intercept"]);

        // ErrorBar has no strict requirements
        assert_eq!(Geom::ErrorBar.aesthetics().required, &[] as &[&str]);
    }
}
