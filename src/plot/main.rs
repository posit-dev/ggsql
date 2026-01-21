//! Plot types for ggsql specification
//!
//! This module defines the typed Plot structures that represent parsed ggsql queries.
//! The Plot is built from the tree-sitter CST (Concrete Syntax Tree) and provides
//! a more convenient, typed interface for working with ggsql specifications.
//!
//! # Plot Structure
//!
//! ```text
//! Plot
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
use std::collections::HashMap;

// Re-export input types
pub use super::types::{
    AestheticValue, ArrayElement, ColumnInfo, DataSource, LiteralValue, Mappings, ParameterValue,
    Schema, SqlExpression,
};

// Re-export Geom and related types from the layer::geom module
pub use super::layer::geom::{
    DefaultParam, DefaultParamValue, Geom, GeomAesthetics, GeomTrait, GeomType, StatResult,
};

// Re-export Layer from the layer module
pub use super::layer::Layer;

// Re-export Scale and Guide types from the scale module
pub use super::scale::{Guide, GuideType, Scale, ScaleType};

// Re-export Coord types from the coord module
pub use super::coord::{Coord, CoordType};

// Re-export Facet types from the facet module
pub use super::facet::{Facet, FacetScales};

/// Complete ggsql visualization specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plot {
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

/// Text labels (from LABELS clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    /// Label assignments (label type → text)
    pub labels: HashMap<String, String>,
}

/// Theme styling (from THEME clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    /// Base theme style
    pub style: Option<String>,
    /// Theme property overrides
    pub properties: HashMap<String, ParameterValue>,
}

impl Plot {
    /// Create a new empty Plot
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

    /// Create a new Plot with the given global mapping
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
    /// - Otherwise, use the first non-synthetic column name mapped to that aesthetic or its family
    /// - Aesthetic families (e.g., x, x2, xmin, xmax, xend) all contribute to the same primary label
    /// - First aesthetic encountered in a family sets the label
    ///
    /// This ensures that:
    /// - Synthetic constant columns (like `__ggsql_const_color_0__`) don't appear as axis/legend titles
    /// - Variant aesthetics like `xmin`/`xmax` can contribute to the primary aesthetic's label
    pub fn compute_aesthetic_labels(&mut self) {
        // Ensure Labels struct exists
        if self.labels.is_none() {
            self.labels = Some(Labels {
                labels: HashMap::new(),
            });
        }
        let labels = self.labels.as_mut().unwrap();

        // Process all layers and their aesthetics
        for layer in &self.layers {
            for (aesthetic, value) in &layer.mappings.aesthetics {
                if let AestheticValue::Column { name, .. } = value {
                    // Skip synthetic constant columns
                    if name.starts_with("__ggsql_const_") {
                        continue;
                    }

                    // Get the primary aesthetic for this aesthetic (e.g., "xmin" -> "x")
                    let primary = GeomAesthetics::primary_aesthetic(aesthetic);

                    // Skip if label already set (user-specified or from earlier aesthetic)
                    if labels.labels.contains_key(primary) {
                        continue;
                    }

                    // Compute the label from the column name
                    let column_name = if let Some(stat_name) = name.strip_prefix("__ggsql_stat__") {
                        stat_name.to_string()
                    } else {
                        name.clone()
                    };

                    labels.labels.insert(primary.to_string(), column_name);
                }
            }
        }
    }
}

impl Default for Plot {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_creation() {
        let spec = Plot::new();
        assert!(spec.global_mappings.is_empty());
        assert_eq!(spec.layers.len(), 0);
        assert!(!spec.has_layers());
        assert_eq!(spec.layer_count(), 0);
    }

    #[test]
    fn test_plot_with_global_mappings() {
        let mut mapping = Mappings::new();
        mapping.insert("x", AestheticValue::standard_column("date"));
        mapping.insert("y", AestheticValue::standard_column("y"));
        let spec = Plot::with_global_mappings(mapping.clone());
        assert_eq!(spec.global_mappings.aesthetics.len(), 2);
        assert!(spec.global_mappings.aesthetics.contains_key("x"));
    }

    #[test]
    fn test_global_mappings_wildcard() {
        let mapping = Mappings::with_wildcard();
        let spec = Plot::with_global_mappings(mapping);
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
    fn test_plot_layer_operations() {
        let mut spec = Plot::new();

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
        assert_eq!(
            Geom::abline().aesthetics().required,
            &["slope", "intercept"]
        );

        // ErrorBar has no strict requirements
        assert_eq!(Geom::errorbar().aesthetics().required, &[] as &[&str]);
    }

    #[test]
    fn test_aesthetic_family_primary_lookup() {
        // Test that variant aesthetics map to their primary
        assert_eq!(GeomAesthetics::primary_aesthetic("x"), "x");
        assert_eq!(GeomAesthetics::primary_aesthetic("xmin"), "x");
        assert_eq!(GeomAesthetics::primary_aesthetic("xmax"), "x");
        assert_eq!(GeomAesthetics::primary_aesthetic("x2"), "x");
        assert_eq!(GeomAesthetics::primary_aesthetic("xend"), "x");
        assert_eq!(GeomAesthetics::primary_aesthetic("y"), "y");
        assert_eq!(GeomAesthetics::primary_aesthetic("ymin"), "y");
        assert_eq!(GeomAesthetics::primary_aesthetic("ymax"), "y");
        assert_eq!(GeomAesthetics::primary_aesthetic("y2"), "y");
        assert_eq!(GeomAesthetics::primary_aesthetic("yend"), "y");

        // Non-family aesthetics return themselves
        assert_eq!(GeomAesthetics::primary_aesthetic("color"), "color");
        assert_eq!(GeomAesthetics::primary_aesthetic("size"), "size");
        assert_eq!(GeomAesthetics::primary_aesthetic("fill"), "fill");
    }

    #[test]
    fn test_compute_labels_from_variant_aesthetics() {
        // Test that variant aesthetics (xmin, xmax) can contribute to primary aesthetic labels
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::ribbon())
            .with_aesthetic(
                "xmin".to_string(),
                AestheticValue::standard_column("lower_bound"),
            )
            .with_aesthetic(
                "xmax".to_string(),
                AestheticValue::standard_column("upper_bound"),
            )
            .with_aesthetic(
                "ymin".to_string(),
                AestheticValue::standard_column("y_lower"),
            )
            .with_aesthetic(
                "ymax".to_string(),
                AestheticValue::standard_column("y_upper"),
            );
        spec.layers.push(layer);

        spec.compute_aesthetic_labels();

        let labels = spec.labels.as_ref().unwrap();
        // First variant encountered sets the label for the primary aesthetic
        // Note: HashMap iteration order may vary, so we just check both x and y have labels
        assert!(
            labels.labels.contains_key("x"),
            "x label should be set from xmin or xmax"
        );
        assert!(
            labels.labels.contains_key("y"),
            "y label should be set from ymin or ymax"
        );
    }

    #[test]
    fn test_user_label_overrides_computed() {
        // Test that user-specified labels take precedence over computed labels
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::ribbon())
            .with_aesthetic(
                "xmin".to_string(),
                AestheticValue::standard_column("lower_bound"),
            )
            .with_aesthetic(
                "xmax".to_string(),
                AestheticValue::standard_column("upper_bound"),
            )
            .with_aesthetic(
                "ymin".to_string(),
                AestheticValue::standard_column("y_lower"),
            )
            .with_aesthetic(
                "ymax".to_string(),
                AestheticValue::standard_column("y_upper"),
            );
        spec.layers.push(layer);

        // Pre-set a user label for x
        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("x".to_string(), "Custom X Label".to_string());
        spec.labels = Some(labels);

        spec.compute_aesthetic_labels();

        let labels = spec.labels.as_ref().unwrap();
        // User-specified label should be preserved
        assert_eq!(labels.labels.get("x"), Some(&"Custom X Label".to_string()));
        // y should still be computed from variants
        assert!(labels.labels.contains_key("y"));
    }

    #[test]
    fn test_primary_aesthetic_sets_label_before_variants() {
        // Test that if both primary and variant are mapped, primary takes precedence
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Add a second layer with xmin
        let layer2 = Layer::new(Geom::ribbon())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("xmin".to_string(), AestheticValue::standard_column("lower"))
            .with_aesthetic("xmax".to_string(), AestheticValue::standard_column("upper"))
            .with_aesthetic(
                "ymin".to_string(),
                AestheticValue::standard_column("y_lower"),
            )
            .with_aesthetic(
                "ymax".to_string(),
                AestheticValue::standard_column("y_upper"),
            );
        spec.layers.push(layer2);

        spec.compute_aesthetic_labels();

        let labels = spec.labels.as_ref().unwrap();
        // First layer's x mapping should win
        assert_eq!(labels.labels.get("x"), Some(&"date".to_string()));
    }
}
