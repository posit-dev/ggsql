//! Facet types for ggsql visualization specifications
//!
//! This module defines faceting configuration for small multiples.

use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::plot::ParameterValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Faceting specification (from FACET clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Facet {
    /// Layout type: wrap or grid
    pub layout: FacetLayout,
    /// Properties from SETTING clause (e.g., scales, ncol, missing)
    /// After resolution, includes validated and defaulted values
    #[serde(default)]
    pub properties: HashMap<String, ParameterValue>,
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

    /// Whether the given position aesthetic has free (independent) scales.
    ///
    /// Accepts internal position names and their variants:
    /// `"pos1"`, `"pos1min"`, `"pos1end"`, `"pos2"`, `"pos2max"`, `"pos3"`, etc.
    pub fn is_free(&self, aesthetic: &str) -> bool {
        use crate::plot::ArrayElement;
        let Some(ParameterValue::Array(arr)) = self.properties.get("free") else {
            return false;
        };
        for (idx, prefix) in ["pos1", "pos2", "pos3"].iter().enumerate() {
            if aesthetic.starts_with(prefix) {
                return matches!(arr.get(idx), Some(ArrayElement::Boolean(true)));
            }
        }
        false
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

    /// Get variable names mapped to their user-facing aesthetic names.
    ///
    /// Returns tuples of (column_name, aesthetic_name):
    /// - Wrap: [("region", "panel")]
    /// - Grid: [("region", "row"), ("year", "column")]
    ///
    /// Note: These are user-facing names. Use AestheticContext to transform
    /// to internal names (facet1, facet2) after context initialization.
    pub fn get_aesthetic_mappings(&self) -> Vec<(&str, &'static str)> {
        let user_names = self.user_facet_names();
        match self {
            FacetLayout::Wrap { variables } => variables
                .iter()
                .map(|v| (v.as_str(), user_names[0]))
                .collect(),
            FacetLayout::Grid { row, column } => {
                let mut result: Vec<(&str, &'static str)> =
                    row.iter().map(|v| (v.as_str(), user_names[0])).collect();
                result.extend(column.iter().map(|v| (v.as_str(), user_names[1])));
                result
            }
        }
    }

    /// Get the user-facing facet aesthetic names for this layout.
    ///
    /// Used by AestheticContext for user↔internal mapping:
    /// - Wrap: ["panel"] → maps to "facet1" internally
    /// - Grid: ["row", "column"] → maps to "facet1", "facet2" internally
    pub fn user_facet_names(&self) -> &'static [&'static str] {
        match self {
            FacetLayout::Wrap { .. } => &["panel"],
            FacetLayout::Grid { .. } => &["row", "column"],
        }
    }

    /// Get the internal facet aesthetic names for this layout.
    ///
    /// Returns: "facet1" for wrap, "facet1" and "facet2" for grid.
    /// Use this after aesthetic transformation has occurred.
    pub fn internal_facet_names(&self) -> Vec<String> {
        match self {
            FacetLayout::Wrap { .. } => vec!["facet1".to_string()],
            FacetLayout::Grid { .. } => vec!["facet1".to_string(), "facet2".to_string()],
        }
    }

    /// Get variable names mapped to their internal aesthetic names.
    ///
    /// Returns tuples of (column_name, internal_aesthetic_name):
    /// - Wrap: [("region", "facet1")]
    /// - Grid: [("region", "facet1"), ("year", "facet2")]
    ///
    /// Use this after aesthetic transformation has occurred.
    pub fn get_internal_aesthetic_mappings(&self) -> Vec<(&str, String)> {
        let internal_names = self.internal_facet_names();
        match self {
            FacetLayout::Wrap { variables } => variables
                .iter()
                .map(|v| (v.as_str(), internal_names[0].clone()))
                .collect(),
            FacetLayout::Grid { row, column } => {
                let mut result: Vec<(&str, String)> = row
                    .iter()
                    .map(|v| (v.as_str(), internal_names[0].clone()))
                    .collect();
                result.extend(
                    column
                        .iter()
                        .map(|v| (v.as_str(), internal_names[1].clone())),
                );
                result
            }
        }
    }

    /// Returns the default properties for this facet layout type.
    ///
    /// Wrap facets support: free, ncol, nrow, missing
    /// Grid facets support: free, missing
    pub fn default_properties(&self) -> &'static [ParamDefinition] {
        /// Valid values for the missing property
        const MISSING_VALUES: &[&str] = &["repeat", "null"];

        match self {
            FacetLayout::Wrap { .. } => {
                const WRAP_PARAMS: &[ParamDefinition] = &[
                    ParamDefinition {
                        name: "free",
                        default: DefaultParamValue::Null,
                        constraint: ParamConstraint::unconstrained(), // Validated separately due to coord-dependent values
                    },
                    ParamDefinition {
                        name: "ncol",
                        default: DefaultParamValue::Null, // Computed from data
                        constraint: ParamConstraint::count(1.0),
                    },
                    ParamDefinition {
                        name: "nrow",
                        default: DefaultParamValue::Null,
                        constraint: ParamConstraint::count(1.0),
                    },
                    ParamDefinition {
                        name: "missing",
                        default: DefaultParamValue::Null,
                        constraint: ParamConstraint::string_option(MISSING_VALUES),
                    },
                ];
                WRAP_PARAMS
            }
            FacetLayout::Grid { .. } => {
                const GRID_PARAMS: &[ParamDefinition] = &[
                    ParamDefinition {
                        name: "free",
                        default: DefaultParamValue::Null,
                        constraint: ParamConstraint::unconstrained(), // Validated separately due to coord-dependent values
                    },
                    ParamDefinition {
                        name: "missing",
                        default: DefaultParamValue::Null,
                        constraint: ParamConstraint::string_option(MISSING_VALUES),
                    },
                ];
                GRID_PARAMS
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ArrayElement;

    fn facet_with_free(free: Vec<bool>) -> Facet {
        let mut f = Facet::new(FacetLayout::Wrap {
            variables: vec!["g".to_string()],
        });
        f.properties.insert(
            "free".to_string(),
            ParameterValue::Array(free.into_iter().map(ArrayElement::Boolean).collect()),
        );
        f
    }

    #[test]
    fn is_free_checks_position_and_variants() {
        let f = facet_with_free(vec![true, false]);
        assert!(f.is_free("pos1"));
        assert!(f.is_free("pos1min"));
        assert!(f.is_free("pos1end"));
        assert!(!f.is_free("pos2"));
        assert!(!f.is_free("pos2max"));
    }

    #[test]
    fn is_free_returns_false_for_material_aesthetics() {
        let f = facet_with_free(vec![true, true]);
        assert!(!f.is_free("color"));
        assert!(!f.is_free("fill"));
    }

    #[test]
    fn is_free_returns_false_without_free_property() {
        let f = Facet::new(FacetLayout::Wrap {
            variables: vec!["g".to_string()],
        });
        assert!(!f.is_free("pos1"));
        assert!(!f.is_free("pos2"));
    }
}
