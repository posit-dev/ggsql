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
pub enum Facet {
    /// FACET variables (wrap layout)
    Wrap {
        variables: Vec<String>,
        scales: FacetScales,
        /// Additional properties from SETTING clause (e.g., ncol, spacing)
        #[serde(default)]
        properties: HashMap<String, ParameterValue>,
        /// Custom label mappings from RENAMING clause
        /// Key = original value, Value = Some(label) or None for suppressed labels
        #[serde(default)]
        label_mapping: Option<HashMap<String, Option<String>>>,
        /// Label template for wildcard mappings (* => '...'), defaults to "{}"
        #[serde(default = "default_label_template")]
        label_template: String,
    },
    /// FACET rows BY cols (grid layout)
    Grid {
        rows: Vec<String>,
        cols: Vec<String>,
        scales: FacetScales,
        /// Additional properties from SETTING clause (e.g., spacing)
        #[serde(default)]
        properties: HashMap<String, ParameterValue>,
        /// Custom label mappings from RENAMING clause
        /// Key = original value, Value = Some(label) or None for suppressed labels
        #[serde(default)]
        label_mapping: Option<HashMap<String, Option<String>>>,
        /// Label template for wildcard mappings (* => '...'), defaults to "{}"
        #[serde(default = "default_label_template")]
        label_template: String,
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
