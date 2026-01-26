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
/// New syntax: `SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...]`
///
/// Examples:
/// - `SCALE DATE x`
/// - `SCALE CONTINUOUS y FROM [0, 100]`
/// - `SCALE DISCRETE color FROM ['A', 'B'] TO ['red', 'blue']`
/// - `SCALE color TO viridis`
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
    pub properties: HashMap<String, ParameterValue>,
    /// Resolved break positions (computed at execution time)
    /// Maps to Vega-Lite's axis.values or legend.values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_breaks: Option<Vec<ArrayElement>>,
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
            resolved_breaks: None,
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
