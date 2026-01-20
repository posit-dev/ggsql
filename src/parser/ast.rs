//! AST (Abstract Syntax Tree) types for ggsql specification
//!
//! This module defines the typed AST structures that represent parsed ggsql queries.
//! The AST is built from the tree-sitter CST (Concrete Syntax Tree) and provides
//! a more convenient, typed interface for working with ggsql specifications.
//!
//! # AST Structure
//!
//! ```text
//! VizSpec
//! ├─ global_mappings: GlobalMapping  (from VISUALISE clause mappings)
//! ├─ source: Option<DataSource>     (optional, from VISUALISE FROM clause)
//! ├─ layers: Vec<Layer>             (1+ LayerNode, one per DRAW clause)
//! ├─ scales: Vec<Scale>             (0+ ScaleNode, one per SCALE clause)
//! ├─ facet: Option<Facet>           (optional, from FACET clause)
//! ├─ coord: Option<Coord>           (optional, from COORD clause)
//! ├─ labels: Option<Labels>         (optional, merged from LABEL clauses)
//! ├─ guides: Vec<Guide>             (0+ GuideNode, one per GUIDE clause)
//! └─ theme: Option<Theme>           (optional, from THEME clause)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};


// Re-export Geom and related types from the geom module
pub use super::geom::{Geom, GeomType, GeomTrait, GeomAesthetics, DefaultParam, DefaultParamValue, StatResult};

// Re-export Layer and SqlExpression from the layer module
pub use super::layer::{Layer, SqlExpression};

/// Column information from a data source schema
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Whether this column is discrete (suitable for grouping)
    /// Discrete: String, Boolean, Categorical, Date
    /// Continuous: numeric types, Datetime, Time
    pub is_discrete: bool,
}

/// Schema of a data source - list of columns with type info
pub type Schema = Vec<ColumnInfo>;

/// Complete ggsql visualization specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VizSpec {
    /// Global aesthetic mappings (from VISUALISE clause)
    pub global_mappings: Mappings,
    /// FROM source (CTE, table, or file) when using VISUALISE FROM syntax
    pub source: Option<DataSource>,
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

/// Unified aesthetic mapping specification
///
/// Used for both global mappings (VISUALISE clause) and layer mappings (MAPPING clause).
/// Supports wildcards combined with explicit mappings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Mappings {
    /// Whether a wildcard (*) was specified
    pub wildcard: bool,
    /// Explicit aesthetic mappings (aesthetic → value)
    pub aesthetics: HashMap<String, AestheticValue>,
}

impl Mappings {
    /// Create a new empty Mappings
    pub fn new() -> Self {
        Self {
            wildcard: false,
            aesthetics: HashMap::new(),
        }
    }

    /// Create a new Mappings with wildcard flag set
    pub fn with_wildcard() -> Self {
        Self {
            wildcard: true,
            aesthetics: HashMap::new(),
        }
    }

    /// Check if the mappings are empty (no wildcard and no aesthetics)
    pub fn is_empty(&self) -> bool {
        !self.wildcard && self.aesthetics.is_empty()
    }

    /// Insert an aesthetic mapping
    pub fn insert(&mut self, aesthetic: impl Into<String>, value: AestheticValue) {
        self.aesthetics.insert(aesthetic.into(), value);
    }

    /// Get an aesthetic value by name
    pub fn get(&self, aesthetic: &str) -> Option<&AestheticValue> {
        self.aesthetics.get(aesthetic)
    }

    /// Check if an aesthetic is mapped
    pub fn contains_key(&self, aesthetic: &str) -> bool {
        self.aesthetics.contains_key(aesthetic)
    }

    /// Get the number of explicit aesthetic mappings
    pub fn len(&self) -> usize {
        self.aesthetics.len()
    }
}

/// Data source for visualization or layer (from VISUALISE FROM or MAPPING ... FROM clause)
///
/// Allows specification of a data source - either a CTE/table name or a file path.
/// Used both for global `VISUALISE FROM` and layer-specific `MAPPING ... FROM`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSource {
    /// CTE or table name (unquoted identifier)
    Identifier(String),
    /// File path (quoted string like 'data.csv')
    FilePath(String),
}

impl DataSource {
    /// Returns the source as a string reference
    pub fn as_str(&self) -> &str {
        match self {
            DataSource::Identifier(s) => s,
            DataSource::FilePath(s) => s,
        }
    }

    /// Returns true if this is a file path source
    pub fn is_file(&self) -> bool {
        matches!(self, DataSource::FilePath(_))
    }
}

/// Value for aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AestheticValue {
    /// Column reference
    Column {
        name: String,
        /// Whether this is a dummy/placeholder column (e.g., for bar charts without x mapped)
        is_dummy: bool,
    },
    /// Literal value (quoted string, number, or boolean)
    Literal(LiteralValue),
}

impl AestheticValue {
    /// Create a column mapping
    pub fn standard_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: false,
        }
    }

    /// Create a dummy/placeholder column mapping (e.g., for bar charts without x mapped)
    pub fn dummy_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: true,
        }
    }

    /// Get column name if this is a column mapping
    pub fn column_name(&self) -> Option<&str> {
        match self {
            Self::Column { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Check if this is a dummy/placeholder column
    pub fn is_dummy(&self) -> bool {
        match self {
            Self::Column { is_dummy, .. } => *is_dummy,
            _ => false,
        }
    }

    /// Check if this is a literal value (not a column mapping)
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }
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

impl Facet {
    /// Get all variables used for faceting
    ///
    /// Returns all column names that will be used to split the data into facets.
    /// For Wrap facets, returns the variables list.
    /// For Grid facets, returns combined rows and cols variables.
    pub fn get_variables(&self) -> Vec<String> {
        match self {
            Facet::Wrap { variables, .. } => variables.clone(),
            Facet::Grid { rows, cols, .. } => {
                let mut vars = rows.clone();
                vars.extend(cols.iter().cloned());
                vars
            }
        }
    }
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
            global_mappings: Mappings::new(),
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
    pub fn with_global_mappings(global_mappings: Mappings) -> Self {
        Self {
            global_mappings,
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

    /// Compute aesthetic labels for axes and legends.
    ///
    /// For each aesthetic used in any layer, determines the appropriate label:
    /// - If user specified a label via LABEL clause, use that
    /// - Otherwise, use the first non-synthetic column name mapped to that aesthetic
    /// - Falls back to the aesthetic name itself if only constants are mapped
    ///
    /// This ensures that synthetic constant columns (like `__ggsql_const_color_0__`)
    /// don't appear as axis/legend titles.
    pub fn compute_aesthetic_labels(&mut self) {
        // Ensure Labels struct exists
        if self.labels.is_none() {
            self.labels = Some(Labels {
                labels: HashMap::new(),
            });
        }
        let labels = self.labels.as_mut().unwrap();

        // Collect all aesthetics used across all layers
        let mut all_aesthetics: HashSet<String> = HashSet::new();
        for layer in &self.layers {
            for aesthetic in layer.mappings.aesthetics.keys() {
                all_aesthetics.insert(aesthetic.clone());
            }
        }

        // For each aesthetic, compute label if not already user-specified
        for aesthetic in all_aesthetics {
            // Skip secondary/interval aesthetics (x2, y2, xmin, etc.)
            // Only primary aesthetics (x, y, color, etc.) should get labels
            if matches!(
                aesthetic.as_str(),
                "x2" | "y2" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
            ) {
                continue;
            }

            // Skip if user already specified this label
            if labels.labels.contains_key(&aesthetic) {
                continue;
            }

            // Find first non-constant column mapping
            let mut label = aesthetic.clone(); // Default to aesthetic name
            for layer in &self.layers {
                if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(&aesthetic) {
                    // Skip synthetic constant columns
                    if name.starts_with("__ggsql_const_") {
                        continue;
                    }
                    // Strip __ggsql_stat__ prefix for human-readable labels
                    if let Some(stat_name) = name.strip_prefix("__ggsql_stat__") {
                        label = stat_name.to_string();
                    } else {
                        label = name.clone();
                    }
                    break;
                }
            }

            labels.labels.insert(aesthetic, label);
        }
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
            AestheticValue::Column { name, .. } => write!(f, "{}", name),
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
        assert!(spec.global_mappings.is_empty());
        assert_eq!(spec.layers.len(), 0);
        assert!(!spec.has_layers());
        assert_eq!(spec.layer_count(), 0);
    }

    #[test]
    fn test_viz_spec_with_global_mappings() {
        let mut mapping = Mappings::new();
        mapping.insert("x", AestheticValue::standard_column("date"));
        mapping.insert("y", AestheticValue::standard_column("y"));
        let spec = VizSpec::with_global_mappings(mapping.clone());
        assert_eq!(spec.global_mappings.aesthetics.len(), 2);
        assert!(spec.global_mappings.aesthetics.contains_key("x"));
    }

    #[test]
    fn test_global_mappings_wildcard() {
        let mapping = Mappings::with_wildcard();
        let spec = VizSpec::with_global_mappings(mapping);
        assert!(spec.global_mappings.wildcard);
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("revenue"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(LiteralValue::String("blue".to_string())),
            );

        assert_eq!(layer.geom, Geom::point());
        assert_eq!(layer.get_column("x"), Some("date"));
        assert_eq!(layer.get_column("y"), Some("revenue"));
        assert!(matches!(layer.get_literal("color"), Some(LiteralValue::String(s)) if s == "blue"));
        assert!(layer.filter.is_none());
    }

    #[test]
    fn test_layer_with_filter() {
        let filter = SqlExpression::new("year > 2020");
        let layer = Layer::new(Geom::point()).with_filter(filter);
        assert!(layer.filter.is_some());
        assert_eq!(layer.filter.as_ref().unwrap().as_str(), "year > 2020");
    }

    #[test]
    fn test_layer_validation() {
        let valid_point = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("y"));

        assert!(valid_point.validate_required_aesthetics().is_ok());

        let invalid_point = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"));

        assert!(invalid_point.validate_required_aesthetics().is_err());

        let valid_ribbon = Layer::new(Geom::ribbon())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"))
            .with_aesthetic("ymin".to_string(), AestheticValue::standard_column("ymin"))
            .with_aesthetic("ymax".to_string(), AestheticValue::standard_column("ymax"));

        assert!(valid_ribbon.validate_required_aesthetics().is_ok());
    }

    #[test]
    fn test_viz_spec_layer_operations() {
        let mut spec = VizSpec::new();

        let layer1 = Layer::new(Geom::point());
        let layer2 = Layer::new(Geom::line());

        spec.layers.push(layer1);
        spec.layers.push(layer2);

        assert!(spec.has_layers());
        assert_eq!(spec.layer_count(), 2);
        assert_eq!(spec.layers[0].geom, Geom::point());
        assert_eq!(spec.layers[1].geom, Geom::line());
    }

    #[test]
    fn test_aesthetic_value_display() {
        let column = AestheticValue::standard_column("sales");
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
        assert_eq!(format!("{}", Geom::point()), "point");
        assert_eq!(format!("{}", Geom::histogram()), "histogram");
        assert_eq!(format!("{}", Geom::errorbar()), "errorbar");
    }

    // ========================================
    // Mappings Struct Tests
    // ========================================

    #[test]
    fn test_mappings_new() {
        let mappings = Mappings::new();
        assert!(!mappings.wildcard);
        assert!(mappings.aesthetics.is_empty());
        assert!(mappings.is_empty());
    }

    #[test]
    fn test_mappings_with_wildcard() {
        let mappings = Mappings::with_wildcard();
        assert!(mappings.wildcard);
        assert!(mappings.aesthetics.is_empty());
        assert!(!mappings.is_empty()); // wildcard counts as non-empty
    }

    #[test]
    fn test_mappings_insert_and_get() {
        let mut mappings = Mappings::new();
        mappings.insert("x", AestheticValue::standard_column("date"));
        mappings.insert("y", AestheticValue::standard_column("value"));

        assert_eq!(mappings.len(), 2);
        assert!(mappings.contains_key("x"));
        assert!(mappings.contains_key("y"));
        assert!(!mappings.contains_key("color"));

        let x_val = mappings.get("x").unwrap();
        assert_eq!(x_val.column_name(), Some("date"));
    }

    #[test]
    fn test_aesthetic_value_column_constructors() {
        let col = AestheticValue::standard_column("date");
        assert!(!col.is_dummy());
        assert_eq!(col.column_name(), Some("date"));

        let dummy_col = AestheticValue::dummy_column("x");
        assert!(dummy_col.is_dummy());
        assert_eq!(dummy_col.column_name(), Some("x"));
    }

    #[test]
    fn test_aesthetic_value_literal() {
        let lit = AestheticValue::Literal(LiteralValue::String("red".to_string()));
        assert!(!lit.is_dummy());
        assert_eq!(lit.column_name(), None);
    }

    #[test]
    fn test_layer_with_wildcard() {
        let layer = Layer::new(Geom::point()).with_wildcard();
        assert!(layer.mappings.wildcard);
    }

    #[test]
    fn test_geom_aesthetics() {
        // Point geom
        let point = Geom::point().aesthetics();
        assert!(point.supported.contains(&"x"));
        assert!(point.supported.contains(&"size"));
        assert!(point.supported.contains(&"shape"));
        assert!(!point.supported.contains(&"linetype"));
        assert_eq!(point.required, &["x", "y"]);

        // Line geom
        let line = Geom::line().aesthetics();
        assert!(line.supported.contains(&"linetype"));
        assert!(line.supported.contains(&"linewidth"));
        assert!(!line.supported.contains(&"size"));
        assert_eq!(line.required, &["x", "y"]);

        // Bar geom - optional x and y (stat decides aggregation)
        let bar = Geom::bar().aesthetics();
        assert!(bar.supported.contains(&"fill"));
        assert!(bar.supported.contains(&"width"));
        assert!(bar.supported.contains(&"y")); // Bar accepts optional y
        assert!(bar.supported.contains(&"x")); // Bar accepts optional x
        assert_eq!(bar.required, &[] as &[&str]); // No required aesthetics

        // Text geom
        let text = Geom::text().aesthetics();
        assert!(text.supported.contains(&"label"));
        assert!(text.supported.contains(&"family"));
        assert_eq!(text.required, &["x", "y"]);

        // Statistical geoms only require x
        assert_eq!(Geom::histogram().aesthetics().required, &["x"]);
        assert_eq!(Geom::density().aesthetics().required, &["x"]);

        // Ribbon requires ymin/ymax
        assert_eq!(Geom::ribbon().aesthetics().required, &["x", "ymin", "ymax"]);

        // Segment/arrow require endpoints
        assert_eq!(
            Geom::segment().aesthetics().required,
            &["x", "y", "xend", "yend"]
        );

        // Reference lines
        assert_eq!(Geom::hline().aesthetics().required, &["yintercept"]);
        assert_eq!(Geom::vline().aesthetics().required, &["xintercept"]);
        assert_eq!(Geom::abline().aesthetics().required, &["slope", "intercept"]);

        // ErrorBar has no strict requirements
        assert_eq!(Geom::errorbar().aesthetics().required, &[] as &[&str]);
    }
}
