//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::types::{ArrayElement, ParameterValue};

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
    /// Transformation method (VIA clause, reserved for future use)
    pub transform_method: Option<String>,
    /// Additional scale properties (SETTING clause)
    pub properties: HashMap<String, ParameterValue>,
}

impl Scale {
    /// Create a new Scale with just an aesthetic name
    pub fn new(aesthetic: impl Into<String>) -> Self {
        Self {
            aesthetic: aesthetic.into(),
            scale_type: None,
            input_range: None,
            output_range: None,
            transform_method: None,
            properties: HashMap::new(),
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

/// Scale types - describe the nature of the data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScaleType {
    // Data type indicators (new syntax)
    /// Continuous numeric data
    Continuous,
    /// Categorical/discrete data
    Discrete,
    /// Binned/bucketed data
    Binned,

    // Temporal scales
    Date,
    DateTime,
    Time,

    // Special
    Identity,
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
