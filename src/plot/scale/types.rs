//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::types::{ArrayElement, ParameterValue};
use super::scale_type::ScaleType;
use super::transform::Transform;

/// Default label template - passes through values unchanged
fn default_label_template() -> String {
    "{}".to_string()
}

/// Scale configuration (from SCALE clause)
///
/// Syntax: `SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]`
///
/// Examples:
/// - `SCALE x VIA date`
/// - `SCALE CONTINUOUS y FROM (0, 100)`
/// - `SCALE DISCRETE color FROM ('A', 'B') TO ('red', 'blue')`
/// - `SCALE color TO viridis`
/// - `SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    /// Specified as modifier: SCALE x VIA date, SCALE CONTINUOUS y
    pub scale_type: Option<ScaleType>,
    /// Input range specification (FROM clause)
    /// Maps to Vega-Lite's scale.domain
    pub input_range: Option<Vec<ArrayElement>>,
    /// Whether the input_range was explicitly specified by the user (FROM clause).
    /// Used to determine whether to apply pre-stat OOB handling in SQL.
    /// If true, the range was specified explicitly (e.g., `FROM ('A', 'B')`).
    /// If false, the range was inferred from the data.
    #[serde(default)]
    pub explicit_input_range: bool,
    /// Output range specification (TO clause)
    /// Either explicit values or a named palette
    pub output_range: Option<OutputRange>,
    /// Transformation (VIA clause)
    pub transform: Option<Transform>,
    /// Whether the transform was explicitly specified by the user (VIA clause).
    /// Used to determine whether to apply type casting in binned scales.
    /// If true, the transform was specified explicitly (e.g., `VIA date`).
    /// If false, the transform was inferred from the column data type.
    #[serde(default)]
    pub explicit_transform: bool,
    /// Additional scale properties (SETTING clause)
    /// Note: `breaks` can be either a Number (count) or Array (explicit positions).
    /// If scalar at parse time, it's converted to Array during resolution.
    pub properties: HashMap<String, ParameterValue>,
    /// Whether this scale has been resolved (set by resolve() method)
    /// Used to skip re-resolution of pre-resolved scales (e.g., Binned scales)
    #[serde(default)]
    pub resolved: bool,
    /// Label mappings for custom axis/legend labels (RENAMING clause)
    /// Maps raw data values to display labels. `None` value suppresses the label.
    /// Example: `RENAMING 'A' => 'Alpha', 'internal' => NULL`
    #[serde(default)]
    pub label_mapping: Option<HashMap<String, Option<String>>>,
    /// Template for generating labels from scale values (e.g., "{} units")
    /// Default is "{}" which passes through the value unchanged.
    /// The `{}` placeholder is replaced with each value at resolution time.
    /// Example: "{} units" -> {"0": "0 units", "25": "25 units", ...}
    #[serde(default = "default_label_template")]
    pub label_template: String,
}

impl Scale {
    /// Create a new Scale with just an aesthetic name
    pub fn new(aesthetic: impl Into<String>) -> Self {
        Self {
            aesthetic: aesthetic.into(),
            scale_type: None,
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: "{}".to_string(),
        }
    }

    /// Numeric break positions (after resolution).
    ///
    /// Delegates to the scale type for type-specific logic (e.g. discrete
    /// scales synthesize `[1, 2, …, n]` from the input range length).
    pub fn numeric_breaks(&self) -> Vec<f64> {
        match &self.scale_type {
            Some(st) => st.numeric_breaks(self),
            None => match self.properties.get("breaks") {
                Some(ParameterValue::Array(breaks)) => {
                    breaks.iter().filter_map(|b| b.to_f64()).collect()
                }
                _ => Vec::new(),
            },
        }
    }

    /// Labelled breaks: `(numeric_position, display_label)` pairs.
    ///
    /// Delegates to the scale type, then applies `label_mapping` overrides.
    /// Suppressed labels (`None` in the mapping) become empty strings.
    pub fn break_labels(&self) -> Vec<(f64, String)> {
        let raw = match &self.scale_type {
            Some(st) => st.break_labels(self),
            None => self
                .numeric_breaks()
                .into_iter()
                .map(|v| (v, format!("{v}")))
                .collect(),
        };
        let mappings = self.label_mapping.as_ref();
        let mut out = Vec::with_capacity(raw.len());
        for (pos, label) in raw {
            match mappings.and_then(|m| m.get(&label)) {
                Some(Some(renamed)) => out.push((pos, renamed.clone())),
                Some(None) => out.push((pos, String::new())),
                None => out.push((pos, label)),
            }
        }
        out
    }

    /// Numeric domain as `(min, max)` from the resolved input range.
    ///
    /// Delegates to the scale type for type-specific logic (e.g. discrete
    /// scales synthesize `(0.5, n + 0.5)` so integer positions sit at
    /// category centres).
    pub fn numeric_domain(&self) -> Option<(f64, f64)> {
        match &self.scale_type {
            Some(st) => st.numeric_domain(self),
            None => {
                let range = self.input_range.as_ref()?;
                let min = range.first()?.to_f64()?;
                let max = range.last()?.to_f64()?;
                Some((min, max))
            }
        }
    }
}

/// Output range specification (TO clause)
/// Either explicit values or a named palette identifier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputRange {
    /// Explicit array of values: TO ('red', 'blue')
    Array(Vec<ArrayElement>),
    /// Named palette identifier: TO viridis
    Palette(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn continuous_scale(domain: (f64, f64), breaks: Vec<f64>) -> Scale {
        let mut s = Scale::new("pos1");
        s.scale_type = Some(ScaleType::continuous());
        s.input_range = Some(vec![
            ArrayElement::Number(domain.0),
            ArrayElement::Number(domain.1),
        ]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(breaks.into_iter().map(ArrayElement::Number).collect()),
        );
        s
    }

    fn discrete_scale(values: &[&str]) -> Scale {
        let mut s = Scale::new("pos2");
        s.scale_type = Some(ScaleType::discrete());
        s.input_range = Some(values.iter().map(|v| ArrayElement::String(v.to_string())).collect());
        s
    }

    fn ordinal_scale(values: &[&str]) -> Scale {
        let mut s = Scale::new("pos1");
        s.scale_type = Some(ScaleType::ordinal());
        s.input_range = Some(values.iter().map(|v| ArrayElement::String(v.to_string())).collect());
        s
    }

    // =========================================================================
    // Continuous
    // =========================================================================

    #[test]
    fn test_continuous_numeric_breaks() {
        let s = continuous_scale((0.0, 100.0), vec![25.0, 50.0, 75.0]);
        assert_eq!(s.numeric_breaks(), vec![25.0, 50.0, 75.0]);
    }

    #[test]
    fn test_continuous_numeric_domain() {
        let s = continuous_scale((0.0, 100.0), vec![]);
        assert_eq!(s.numeric_domain(), Some((0.0, 100.0)));
    }

    #[test]
    fn test_continuous_no_breaks() {
        let s = continuous_scale((0.0, 100.0), vec![]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
    }

    // =========================================================================
    // Discrete
    // =========================================================================

    #[test]
    fn test_discrete_numeric_breaks() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(s.numeric_breaks(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_discrete_numeric_domain() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(s.numeric_domain(), Some((0.5, 3.5)));
    }

    #[test]
    fn test_discrete_single_category() {
        let s = discrete_scale(&["only"]);
        assert_eq!(s.numeric_breaks(), vec![1.0]);
        assert_eq!(s.numeric_domain(), Some((0.5, 1.5)));
    }

    #[test]
    fn test_discrete_empty() {
        let s = discrete_scale(&[]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
        assert_eq!(s.numeric_domain(), None);
    }

    // =========================================================================
    // Ordinal
    // =========================================================================

    #[test]
    fn test_ordinal_numeric_breaks() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(s.numeric_breaks(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ordinal_numeric_domain() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(s.numeric_domain(), Some((0.5, 3.5)));
    }

    // =========================================================================
    // Identity / no scale type
    // =========================================================================

    #[test]
    fn test_identity_string_returns_empty() {
        let mut s = Scale::new("color");
        s.scale_type = Some(ScaleType::identity());
        s.input_range = Some(vec![
            ArrayElement::String("red".to_string()),
            ArrayElement::String("blue".to_string()),
        ]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
        assert_eq!(s.numeric_domain(), None);
    }

    #[test]
    fn test_no_scale_type_falls_back() {
        let mut s = Scale::new("pos1");
        s.input_range = Some(vec![
            ArrayElement::Number(10.0),
            ArrayElement::Number(50.0),
        ]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(20.0), ArrayElement::Number(40.0)]),
        );
        assert_eq!(s.numeric_breaks(), vec![20.0, 40.0]);
        assert_eq!(s.numeric_domain(), Some((10.0, 50.0)));
    }

    // =========================================================================
    // break_labels
    // =========================================================================

    #[test]
    fn test_continuous_break_labels() {
        let s = continuous_scale((0.0, 100.0), vec![25.0, 50.0, 75.0]);
        assert_eq!(
            s.break_labels(),
            vec![(25.0, "25".to_string()), (50.0, "50".to_string()), (75.0, "75".to_string())]
        );
    }

    #[test]
    fn test_discrete_break_labels() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(
            s.break_labels(),
            vec![(1.0, "A".to_string()), (2.0, "B".to_string()), (3.0, "C".to_string())]
        );
    }

    #[test]
    fn test_ordinal_break_labels() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(
            s.break_labels(),
            vec![(1.0, "low".to_string()), (2.0, "mid".to_string()), (3.0, "high".to_string())]
        );
    }

    #[test]
    fn test_break_labels_with_mapping() {
        let mut s = discrete_scale(&["A", "B", "C"]);
        let mut mapping = HashMap::new();
        mapping.insert("A".to_string(), Some("Alpha".to_string()));
        mapping.insert("C".to_string(), None);
        s.label_mapping = Some(mapping);
        assert_eq!(
            s.break_labels(),
            vec![(1.0, "Alpha".to_string()), (2.0, "B".to_string()), (3.0, String::new())]
        );
    }
}
