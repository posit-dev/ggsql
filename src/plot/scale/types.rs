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
/// New syntax: `SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]`
///
/// Examples:
/// - `SCALE x VIA date`
/// - `SCALE CONTINUOUS y FROM [0, 100]`
/// - `SCALE DISCRETE color FROM ['A', 'B'] TO ['red', 'blue']`
/// - `SCALE color TO viridis`
/// - `SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    /// Now specified as modifier: SCALE x VIA date, SCALE CONTINUOUS y
    pub scale_type: Option<ScaleType>,
    /// Input range specification (FROM clause)
    /// Maps to Vega-Lite's scale.domain
    pub input_range: Option<Vec<ArrayElement>>,
    /// Whether the input_range was explicitly specified by the user (FROM clause).
    /// Used to determine whether to apply pre-stat OOB handling in SQL.
    /// If true, the range was specified explicitly (e.g., `FROM ['A', 'B']`).
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
}

/// Output range specification (TO clause)
/// Either explicit values or a named palette identifier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputRange {
    /// Explicit array of values: TO ['red', 'blue']
    Array(Vec<ArrayElement>),
    /// Named palette identifier: TO viridis
    Palette(String),
}
