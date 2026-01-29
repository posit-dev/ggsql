//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::types::{ArrayElement, ParameterValue};
use super::scale_type::ScaleType;
use super::transform::Transform;

/// Scale configuration (from SCALE clause)
///
/// New syntax: `SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]`
///
/// Examples:
/// - `SCALE DATE x`
/// - `SCALE CONTINUOUS y FROM [0, 100]`
/// - `SCALE DISCRETE color FROM ['A', 'B'] TO ['red', 'blue']`
/// - `SCALE color TO viridis`
/// - `SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    /// Now specified as modifier: SCALE DATE x, SCALE CONTINUOUS y
    pub scale_type: Option<ScaleType>,
    /// Input range specification (FROM clause)
    /// Maps to Vega-Lite's scale.domain
    pub input_range: Option<Vec<ArrayElement>>,
    /// Output range specification (TO clause)
    /// Either explicit values or a named palette
    pub output_range: Option<OutputRange>,
    /// Transformation (VIA clause)
    pub transform: Option<Transform>,
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
    /// Template for generating labels from break values (RENAMING * => '...')
    /// The `{}` placeholder is replaced with each break value at resolution time.
    /// Example: Some("{} units") -> {"0": "0 units", "25": "25 units", ...}
    #[serde(default)]
    pub label_template: Option<String>,
}

impl Scale {
    /// Create a new Scale with just an aesthetic name
    pub fn new(aesthetic: impl Into<String>) -> Self {
        Self {
            aesthetic: aesthetic.into(),
            scale_type: None,
            input_range: None,
            output_range: None,
            transform: None,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
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

/// Guide configuration (from GUIDE clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    /// The aesthetic this guide applies to
    pub aesthetic: String,
    /// Guide type
    pub guide_type: Option<GuideType>,
    /// Guide properties
    pub properties: HashMap<String, ParameterValue>,
}

/// Guide types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuideType {
    Legend,
    ColorBar,
    Axis,
    None,
}
