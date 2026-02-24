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
    DefaultAesthetics, DefaultParam, DefaultParamValue, Geom, GeomTrait, GeomType, StatResult,
};

use crate::plot::types::{AestheticValue, DataSource, Mappings, ParameterValue, SqlExpression};

/// A single visualization layer (from DRAW clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Geometric object type
    pub geom: Geom,
    /// All aesthetic mappings combined from multiple sources:
    ///
    /// 1. **MAPPING clause** (from query, highest precedence):
    ///    - Column references: `date AS x` → `AestheticValue::Column`
    ///    - Literals: `'foo' AS color` → `AestheticValue::Literal` (converted to Column during execution)
    ///
    /// 2. **SETTING clause** (from query, second precedence):
    ///    - Added during execution via `resolve_aesthetics()`
    ///    - Stored as `AestheticValue::Literal`
    ///
    /// 3. **Geom defaults** (lowest precedence):
    ///    - Added during execution via `resolve_aesthetics()`
    ///    - Stored as `AestheticValue::Literal`
    ///
    /// **Important distinction for scale application**:
    /// - Query literals (`MAPPING 'foo' AS color`) are converted to columns during query execution
    ///   via `build_layer_select_list()`, becoming `AestheticValue::Column` before reaching writers.
    ///   These columns can have scales applied.
    /// - SETTING/defaults remain as `AestheticValue::Literal` and render as constant values
    ///   without scale transformations.
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
    /// Key for this layer's data in the datamap (set during execution).
    /// Defaults to `None`. Set to `__ggsql_layer_<idx>__` during execution,
    /// but may point to another layer's data when queries are deduplicated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_key: Option<String>,
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
            data_key: None,
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
        for aesthetic in self.geom.aesthetics().required() {
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

    /// Resolve aesthetics for all supported aesthetics not in MAPPING.
    ///
    /// For each supported aesthetic that's not already mapped in MAPPING:
    /// - Check SETTING parameters first (user-specified, highest priority) and consume from parameters
    /// - Fall back to geom defaults (lower priority)
    /// - Insert into mappings as `AestheticValue::Literal`
    ///
    /// Precedence: MAPPING > SETTING > geom defaults
    ///
    /// **Important**: Query literals from MAPPING (`'foo' AS color`) have already been converted
    /// to columns during query execution, so this only adds SETTING/default literals which
    /// remain as `AestheticValue::Literal` and render without scale transformations.
    ///
    /// Call this during execution to provide a single source of truth for writers.
    pub fn resolve_aesthetics(&mut self) {
        let supported_aesthetics = self.geom.aesthetics().supported();

        for aesthetic_name in supported_aesthetics {
            // Skip if already in MAPPING (highest precedence)
            if self.mappings.contains_key(aesthetic_name) {
                continue;
            }

            // Check SETTING first (user-specified) and consume from parameters
            if let Some(value) = self.parameters.remove(aesthetic_name) {
                self.mappings
                    .insert(aesthetic_name, AestheticValue::Literal(value));
                continue;
            }

            // Fall back to geom default (filter out Null = non-literal defaults)
            if let Some(default_value) = self.geom.aesthetics().get(aesthetic_name) {
                match default_value.to_parameter_value() {
                    ParameterValue::Null => continue,
                    value => {
                        self.mappings
                            .insert(aesthetic_name, AestheticValue::Literal(value));
                    }
                }
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

    /// Update layer mappings to use prefixed aesthetic column names.
    ///
    /// After building a layer query that creates aesthetic columns with prefixed names,
    /// the layer's mappings need to be updated to point to these prefixed column names.
    ///
    /// This function converts:
    /// - `AestheticValue::Column { name: "Date", ... }` → `AestheticValue::Column { name: "__ggsql_aes_x__", ... }`
    /// - `AestheticValue::Literal(...)` → `AestheticValue::Column { name: "__ggsql_aes_color__", ... }`
    ///
    /// Note: The final rename from prefixed names to clean aesthetic names (e.g., "x")
    /// happens in Polars after query execution, before the data goes to the writer.
    pub fn update_mappings_for_aesthetic_columns(&mut self) {
        use crate::naming;

        for (aesthetic, value) in self.mappings.aesthetics.iter_mut() {
            let aes_col_name = naming::aesthetic_column(aesthetic);
            match value {
                AestheticValue::Column {
                    name,
                    original_name,
                    ..
                } => {
                    // Preserve the original column name for labels before overwriting
                    if original_name.is_none() {
                        *original_name = Some(name.clone());
                    }
                    // Column is now named with the prefixed aesthetic name
                    *name = aes_col_name;
                }
                AestheticValue::Literal(_) => {
                    // Literals are also columns with prefixed aesthetic name
                    // Note: literals don't have an original_name to preserve
                    *value = AestheticValue::standard_column(aes_col_name);
                }
            }
        }
    }

    /// Update layer mappings to use prefixed aesthetic names for remapped columns.
    ///
    /// After remappings are applied (stat columns renamed to prefixed aesthetic names),
    /// the layer mappings need to be updated so the writer uses the correct field names.
    ///
    /// For column remappings, the original name is set to the stat name (e.g., "density", "count")
    /// so axis labels show meaningful names instead of internal prefixed names.
    ///
    /// For literal remappings, the value becomes a column reference pointing to the
    /// constant column created by `apply_remappings_post_query`.
    pub fn update_mappings_for_remappings(&mut self) {
        use crate::naming;

        // For each remapping, add the target aesthetic to mappings pointing to the prefixed name
        for (target_aesthetic, value) in &self.remappings.aesthetics {
            let prefixed_name = naming::aesthetic_column(target_aesthetic);

            let new_value = match value {
                AestheticValue::Column {
                    original_name,
                    is_dummy,
                    ..
                } => {
                    // Use the stat name from remappings as the original_name for labels
                    // The stat_col_value contains the user-specified stat name (e.g., "density", "count")
                    AestheticValue::Column {
                        name: prefixed_name,
                        original_name: original_name.clone(),
                        is_dummy: *is_dummy,
                    }
                }
                AestheticValue::Literal(_) => {
                    // Literal becomes a column reference after post-query processing
                    // No original_name since it's a constant value
                    AestheticValue::Column {
                        name: prefixed_name,
                        original_name: None,
                        is_dummy: false,
                    }
                }
            };

            self.mappings
                .aesthetics
                .insert(target_aesthetic.clone(), new_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_aesthetics_from_settings() {
        // Test that resolve_aesthetics() moves aesthetic values from parameters to mappings
        let mut layer = Layer::new(Geom::point());
        layer
            .parameters
            .insert("size".to_string(), ParameterValue::Number(5.0));
        layer
            .parameters
            .insert("opacity".to_string(), ParameterValue::Number(0.8));

        layer.resolve_aesthetics();

        // Values should be moved from parameters to mappings as Literal
        assert!(!layer.parameters.contains_key("size"));
        assert!(!layer.parameters.contains_key("opacity"));
        assert_eq!(
            layer.mappings.get("size"),
            Some(&AestheticValue::Literal(ParameterValue::Number(5.0)))
        );
        assert_eq!(
            layer.mappings.get("opacity"),
            Some(&AestheticValue::Literal(ParameterValue::Number(0.8)))
        );
    }

    #[test]
    fn test_resolve_aesthetics_from_defaults() {
        // Test that resolve_aesthetics() includes geom default values
        let mut layer = Layer::new(Geom::point());

        layer.resolve_aesthetics();

        // Point geom has default shape = 'circle'
        assert_eq!(
            layer.mappings.get("shape"),
            Some(&AestheticValue::Literal(ParameterValue::String(
                "circle".to_string()
            )))
        );
    }

    #[test]
    fn test_resolve_aesthetics_skips_mapped() {
        // Test that resolve_aesthetics() skips aesthetics that are already in MAPPING
        let mut layer = Layer::new(Geom::point());
        layer.mappings.insert(
            "size",
            AestheticValue::standard_column("my_size".to_string()),
        );
        layer
            .parameters
            .insert("size".to_string(), ParameterValue::Number(5.0));

        layer.resolve_aesthetics();

        // size should stay in parameters (not moved to mappings) because it's already in MAPPING
        assert!(layer.parameters.contains_key("size"));
        // The mapping should still be the Column, not replaced with Literal
        assert!(matches!(
            layer.mappings.get("size"),
            Some(AestheticValue::Column { .. })
        ));
    }

    #[test]
    fn test_resolve_aesthetics_precedence() {
        // Test that SETTING takes precedence over geom defaults
        let mut layer = Layer::new(Geom::point());
        layer.parameters.insert(
            "shape".to_string(),
            ParameterValue::String("square".to_string()),
        );

        layer.resolve_aesthetics();

        // Should use SETTING value, not default
        assert_eq!(
            layer.mappings.get("shape"),
            Some(&AestheticValue::Literal(ParameterValue::String(
                "square".to_string()
            )))
        );
    }
}
