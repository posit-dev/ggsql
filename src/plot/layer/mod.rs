//! Layer type for ggsql visualization layers
//!
//! This module defines the Layer struct and related types for representing
//! a single visualization layer (from DRAW clause) in a ggsql specification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Geom is now a submodule of layer
pub mod geom;

// Re-export geom types for convenience
pub use geom::{
    DefaultParam, DefaultParamValue, Geom, GeomAesthetics, GeomTrait, GeomType, StatResult,
};

use crate::plot::types::{
    AestheticValue, DataSource, Mappings, ParameterValue, SqlExpression,
};

/// A single visualization layer (from DRAW clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Geometric object type
    pub geom: Geom,
    /// Aesthetic mappings (from MAPPING clause)
    pub mappings: Mappings,
    /// Stat remappings (from REMAPPING clause): stat_name → aesthetic
    /// Maps stat-computed columns (e.g., "count") to aesthetic channels (e.g., "y")
    pub remappings: Mappings,
    /// Geom parameters (not aesthetic mappings)
    pub parameters: HashMap<String, ParameterValue>,
    /// Optional data source for this layer (from MAPPING ... FROM)
    pub source: Option<DataSource>,
    /// Optional filter expression for this layer (from FILTER clause)
    pub filter: Option<SqlExpression>,
    /// Optional ORDER BY expression for this layer
    pub order_by: Option<SqlExpression>,
    /// Columns for grouping/partitioning (from PARTITION BY clause)
    pub partition_by: Vec<String>,
}

impl Layer {
    /// Create a new layer with the given geom
    pub fn new(geom: Geom) -> Self {
        Self {
            geom,
            mappings: Mappings::new(),
            remappings: Mappings::new(),
            parameters: HashMap::new(),
            source: None,
            filter: None,
            order_by: None,
            partition_by: Vec::new(),
        }
    }

    /// Set the filter expression
    pub fn with_filter(mut self, filter: SqlExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the ORDER BY expression
    pub fn with_order_by(mut self, order: SqlExpression) -> Self {
        self.order_by = Some(order);
        self
    }

    /// Set the data source for this layer
    pub fn with_source(mut self, source: DataSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Add an aesthetic mapping
    pub fn with_aesthetic(mut self, aesthetic: impl Into<String>, value: AestheticValue) -> Self {
        self.mappings.insert(aesthetic, value);
        self
    }

    /// Set the wildcard flag
    pub fn with_wildcard(mut self) -> Self {
        self.mappings.wildcard = true;
        self
    }

    /// Add a parameter
    pub fn with_parameter(mut self, parameter: String, value: ParameterValue) -> Self {
        self.parameters.insert(parameter, value);
        self
    }

    /// Set the partition columns for grouping
    pub fn with_partition_by(mut self, columns: Vec<String>) -> Self {
        self.partition_by = columns;
        self
    }

    /// Get a column reference from an aesthetic, if it's mapped to a column
    pub fn get_column(&self, aesthetic: &str) -> Option<&str> {
        match self.mappings.get(aesthetic) {
            Some(AestheticValue::Column { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// Get a literal value from an aesthetic, if it's mapped to a literal
    pub fn get_literal(&self, aesthetic: &str) -> Option<&ParameterValue> {
        match self.mappings.get(aesthetic) {
            Some(AestheticValue::Literal(lit)) => Some(lit),
            _ => None,
        }
    }

    /// Check if this layer has the required aesthetics for its geom
    pub fn validate_required_aesthetics(&self) -> std::result::Result<(), String> {
        for aesthetic in self.geom.aesthetics().required {
            if !self.mappings.contains_key(aesthetic) {
                return Err(format!(
                    "Geom '{}' requires aesthetic '{}' but it was not provided",
                    self.geom, aesthetic
                ));
            }
        }

        Ok(())
    }

    /// Apply default parameter values for any params not specified by user.
    ///
    /// Call this during execution to ensure all stat params have values.
    pub fn apply_default_params(&mut self) {
        for param in self.geom.default_params() {
            if !self.parameters.contains_key(param.name) {
                let value = match &param.default {
                    DefaultParamValue::String(s) => ParameterValue::String(s.to_string()),
                    DefaultParamValue::Number(n) => ParameterValue::Number(*n),
                    DefaultParamValue::Boolean(b) => ParameterValue::Boolean(*b),
                    DefaultParamValue::Null => continue, // Don't insert null defaults
                };
                self.parameters.insert(param.name.to_string(), value);
            }
        }
    }

    /// Validate that all SETTING parameters are valid for this layer's geom
    pub fn validate_settings(&self) -> std::result::Result<(), String> {
        let valid = self.geom.valid_settings();
        for param_name in self.parameters.keys() {
            if !valid.contains(&param_name.as_str()) {
                return Err(format!(
                    "Invalid setting '{}' for geom '{}'. Valid settings are: {}",
                    param_name,
                    self.geom,
                    valid.join(", ")
                ));
            }
        }
        Ok(())
    }
}
