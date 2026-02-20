//! Facet types for ggsql visualization specifications
//!
//! This module defines faceting configuration for small multiples.

use crate::plot::ParameterValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default label template for facets
fn default_label_template() -> String {
    "{}".to_string()
}

/// Faceting specification (from FACET clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Facet {
    /// Layout type: wrap or grid
    pub layout: FacetLayout,
    /// Properties from SETTING clause (e.g., scales, ncol, missing)
    /// After resolution, includes validated and defaulted values
    #[serde(default)]
    pub properties: HashMap<String, ParameterValue>,
    /// Custom label mappings from RENAMING clause
    /// Key = original value, Value = Some(label) or None for suppressed labels
    #[serde(default)]
    pub label_mapping: Option<HashMap<String, Option<String>>>,
    /// Label template for wildcard mappings (* => '...'), defaults to "{}"
    #[serde(default = "default_label_template")]
    pub label_template: String,
    /// Whether properties have been resolved (validated and defaults applied)
    #[serde(skip, default)]
    pub resolved: bool,
}

/// Facet variable layout specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FacetLayout {
    /// FACET variables (wrap layout)
    Wrap { variables: Vec<String> },
    /// FACET row BY column (grid layout)
    Grid {
        row: Vec<String>,
        column: Vec<String>,
    },
}

impl Facet {
    /// Create a new Facet with the given layout
    ///
    /// Properties start empty and unresolved. Call `resolve_properties` after
    /// data is available to validate and apply defaults.
    pub fn new(layout: FacetLayout) -> Self {
        Self {
            layout,
            properties: HashMap::new(),
            label_mapping: None,
            label_template: "{}".to_string(),
            resolved: false,
        }
    }

    /// Get all variables used for faceting
    ///
    /// Returns all column names that will be used to split the data into facets.
    /// For Wrap facets, returns the variables list.
    /// For Grid facets, returns combined rows and columns variables.
    pub fn get_variables(&self) -> Vec<String> {
        self.layout.get_variables()
    }

    /// Check if this is a wrap layout facet
    pub fn is_wrap(&self) -> bool {
        self.layout.is_wrap()
    }

    /// Check if this is a grid layout facet
    pub fn is_grid(&self) -> bool {
        self.layout.is_grid()
    }
}

impl FacetLayout {
    /// Get all variables used for faceting
    ///
    /// Returns all column names that will be used to split the data into facets.
    /// For Wrap facets, returns the variables list.
    /// For Grid facets, returns combined row and column variables.
    pub fn get_variables(&self) -> Vec<String> {
        match self {
            FacetLayout::Wrap { variables } => variables.clone(),
            FacetLayout::Grid { row, column } => {
                let mut vars = row.clone();
                vars.extend(column.iter().cloned());
                vars
            }
        }
    }

    /// Check if this is a wrap layout
    pub fn is_wrap(&self) -> bool {
        matches!(self, FacetLayout::Wrap { .. })
    }

    /// Check if this is a grid layout
    pub fn is_grid(&self) -> bool {
        matches!(self, FacetLayout::Grid { .. })
    }

    /// Get variable names mapped to their aesthetic names.
    ///
    /// Returns tuples of (column_name, aesthetic_name):
    /// - Wrap: [("region", "panel")]
    /// - Grid: [("region", "row"), ("year", "column")]
    pub fn get_aesthetic_mappings(&self) -> Vec<(&str, &'static str)> {
        match self {
            FacetLayout::Wrap { variables } => {
                variables.iter().map(|v| (v.as_str(), "panel")).collect()
            }
            FacetLayout::Grid { row, column } => {
                let mut result: Vec<(&str, &'static str)> =
                    row.iter().map(|v| (v.as_str(), "row")).collect();
                result.extend(column.iter().map(|v| (v.as_str(), "column")));
                result
            }
        }
    }

    /// Get the aesthetic names used by this layout.
    ///
    /// - Wrap: ["panel"]
    /// - Grid: ["row", "column"]
    pub fn get_aesthetics(&self) -> Vec<&'static str> {
        match self {
            FacetLayout::Wrap { .. } => vec!["panel"],
            FacetLayout::Grid { .. } => vec!["row", "column"],
        }
    }
}
