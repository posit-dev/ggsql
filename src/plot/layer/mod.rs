//! Layer type for ggsql visualization layers
//!
//! This module defines the Layer struct and related types for representing
//! a single visualization layer (from DRAW clause) in a ggsql specification.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Geom is a submodule of layer
pub mod geom;

// Orientation is a submodule of layer
pub mod orientation;

// Position is a submodule of layer
pub mod position;

// Re-export orientation functions and constants
pub use orientation::is_transposed;

// Re-export geom types for convenience
pub use geom::{
    DefaultAesthetics, DefaultParam, DefaultParamValue, Geom, GeomTrait, GeomType, StatResult,
};

// Re-export position types for convenience
pub use position::{Position, PositionTrait, PositionType};

use crate::{
    plot::{
        is_facet_aesthetic, parse_positional,
        types::{AestheticValue, DataSource, Mappings, ParameterValue, SqlExpression},
    },
    AestheticContext,
};

/// A single visualization layer (from DRAW clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Geometric object type
    pub geom: Geom,
    /// Position adjustment for overlapping elements
    pub position: Position,
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
    /// Adjusted width after position adjustment (e.g., for dodged bars).
    /// Set during execution by position::apply_position_adjustments().
    /// Writers can use this to know the actual element width after dodging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adjusted_width: Option<f64>,
}

impl Layer {
    /// Create a new layer with the given geom
    pub fn new(geom: Geom) -> Self {
        Self {
            geom,
            position: Position::default(),
            mappings: Mappings::new(),
            remappings: Mappings::new(),
            parameters: HashMap::new(),
            source: None,
            filter: None,
            order_by: None,
            partition_by: Vec::new(),
            data_key: None,
            adjusted_width: None,
        }
    }

    /// Set the position adjustment
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = position;
        self
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

    /// Validate layer aesthetic mappings.
    ///
    /// Performs three checks:
    /// 1. All required aesthetics are present
    /// 2. Positional requirements allow bidirectional satisfaction (handles orientation flipping)
    /// 3. No unsupported/exotic aesthetics are mapped
    ///
    /// # Parameters
    /// - `context`: Optional aesthetic context for translating internal → user-facing names
    /// - `include_delayed`: If true, allows delayed aesthetics (stat-produced). Use `true` for
    ///   writer validation, `false` for execution validation.
    ///
    /// # Returns
    /// `Ok(())` if validation passes, or `Err(message)` with a user-friendly error message.
    pub fn validate_mapping(
        &self,
        context: &Option<AestheticContext>,
        include_delayed: bool,
    ) -> std::result::Result<(), String> {
        // If there is aesthetic context, translate to user-facing form
        let translate = |aes: &str| -> String {
            let name = match context {
                Some(ctx) => ctx.map_internal_to_user(aes),
                None => aes.to_string(),
            };
            format!("`{}`", name)
        };

        // Check if all required aesthetics exist.
        let mut missing = Vec::new();
        let mut positional_reqs: Vec<(&str, u8, &str)> = Vec::new();

        for aesthetic in self.geom.aesthetics().required() {
            if let Some((slot, suffix)) = parse_positional(aesthetic) {
                positional_reqs.push((aesthetic, slot, suffix))
            } else if !self.mappings.contains_key(aesthetic) {
                missing.push(translate(aesthetic));
            }
        }

        if !missing.is_empty() {
            return Err(format!(
                "Layer '{}' mapping requires the {} aesthetic{s}.",
                self.geom,
                missing.join(", "),
                s = if missing.len() > 1 { "s" } else { "" }
            ));
        }

        // Validate positional requirements bidirectionally
        // Try both slot assignments: (1→1, 2→2) and (1→2, 2→1)
        if !positional_reqs.is_empty() {
            // Pre-compute flipped versions to avoid repeated calculation
            let pairs: Vec<_> = positional_reqs
                .iter()
                .map(|(name, slot, suffix)| {
                    let flipped_slot = if *slot == 1 { 2 } else { 1 };
                    let flipped = format!("pos{}{}", flipped_slot, suffix);
                    (*name, flipped)
                })
                .collect();

            // Find first missing aesthetic in each orientation
            let identity_missing = pairs
                .iter()
                .find(|(name, _)| !self.mappings.contains_key(name));

            let flipped_missing = pairs
                .iter()
                .find(|(_, flipped)| !self.mappings.contains_key(flipped));

            if identity_missing.is_some() && flipped_missing.is_some() {
                let (missing, flipped) = identity_missing.unwrap();

                // Check if flipped version is present (mixed orientation case)
                if self.mappings.contains_key(flipped) {
                    return Err(format!(
                        "Layer '{}' has mixed positional aesthetic orientations. \
                         Found '{}' but expected '{}' to match the orientation of other aesthetics.",
                        self.geom,
                        translate(flipped),
                        translate(missing)
                    ));
                }

                // Truly missing aesthetic
                return Err(format!(
                    "Layer '{}' mapping requires the aesthetic '{}' (or '{}').",
                    self.geom,
                    translate(missing),
                    translate(flipped)
                ));
            }
        }

        let mut supported: HashSet<String> = if include_delayed {
            self.geom.aesthetics().names()
        } else {
            self.geom.aesthetics().supported()
        }
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        // At this point in execution we don't know orientation yet,
        // so we'll approve both flipped and upflipped aesthetics.
        if let Some(ctx) = context {
            let flipped: Vec<String> = supported
                .iter()
                .map(|aes| ctx.flip_positional(aes))
                .collect();
            supported.extend(flipped);
        }

        // Check if any unsupported mappings are present
        let mut extra = Vec::new();

        for aesthetic in self.mappings.aesthetics.keys() {
            if is_facet_aesthetic(aesthetic) {
                continue;
            }
            if !supported.contains(aesthetic) {
                extra.push(translate(aesthetic));
            }
        }
        if !extra.is_empty() {
            return Err(format!(
                "Layer '{}' does not support the {} mapping{s}.",
                self.geom,
                extra.join(", "),
                s = if extra.len() > 1 { "s" } else { "" }
            ));
        }
        Ok(())
    }

    /// Apply default parameter values for any params not specified by user.
    ///
    /// Call this during execution to ensure all geom and position params have values.
    /// Geom defaults are applied first, then position defaults, so geom defaults take
    /// precedence. For example, if a geom defines width => 0.8 and the position (dodge)
    /// defines width => 0.9, the geom's 0.8 is used.
    pub fn apply_default_params(&mut self) {
        // Apply geom defaults first (higher priority)
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

        // Apply position defaults second (lower priority, won't override geom defaults)
        for param in self.position.default_params() {
            if !self.parameters.contains_key(param.name) {
                let value = match &param.default {
                    DefaultParamValue::String(s) => ParameterValue::String(s.to_string()),
                    DefaultParamValue::Number(n) => ParameterValue::Number(*n),
                    DefaultParamValue::Boolean(b) => ParameterValue::Boolean(*b),
                    DefaultParamValue::Null => continue,
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

    /// Validate that all SETTING parameters are valid for this layer's geom and position
    pub fn validate_settings(&self) -> std::result::Result<(), String> {
        // Combine valid settings from both geom and position
        let mut valid = self.geom.valid_settings();
        valid.extend(self.position.valid_settings());

        for param_name in self.parameters.keys() {
            if !valid.contains(&param_name.as_str()) {
                return Err(format!(
                    "Invalid setting '{}' for geom '{}' with position '{}'. Valid settings are: {}",
                    param_name,
                    self.geom,
                    self.position,
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
                AestheticValue::AnnotationColumn { name } => {
                    // AnnotationColumn already has identity scale behavior, just update name
                    *name = aes_col_name;
                }
                AestheticValue::Literal(_) => {
                    // Literals become standard columns with prefixed aesthetic name
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
                AestheticValue::AnnotationColumn { .. } => {
                    // Annotation columns can be remapped (e.g., stat transforms on annotation data)
                    // They remain annotation columns (identity scale)
                    AestheticValue::AnnotationColumn {
                        name: prefixed_name,
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

    #[test]
    fn test_validate_mapping_bidirectional_missing() {
        // Test error message when aesthetic is completely missing (neither identity nor flipped form)
        use crate::AestheticContext;

        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );
        // Missing both pos2min and pos1min (required by ribbon)

        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        let result = layer.validate_mapping(&Some(ctx), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("ymin") && err.contains("xmin"),
            "Expected error to mention both alternatives (ymin/xmin), got: {}",
            err
        );
    }

    #[test]
    fn test_validate_mapping_bidirectional_mixed_orientation() {
        // Test error message when aesthetics are present but in mixed orientations
        use crate::AestheticContext;

        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );
        layer.mappings.insert(
            "pos2min".to_string(),
            AestheticValue::standard_column("ymin".to_string()),
        );
        layer.mappings.insert(
            "pos1max".to_string(), // This should be pos2max to match pos2min's orientation
            AestheticValue::standard_column("xmax".to_string()),
        );

        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        let result = layer.validate_mapping(&Some(ctx), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("mixed") && err.contains("orientation"),
            "Expected error about mixed orientation, got: {}",
            err
        );
        assert!(
            err.contains("xmax") && err.contains("ymax"),
            "Expected error to mention the conflicting aesthetics (xmax/ymax), got: {}",
            err
        );
    }

    #[test]
    fn test_validate_mapping_bidirectional_identity_ok() {
        // Test that validation passes when all requirements are in identity form
        use crate::AestheticContext;

        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );
        layer.mappings.insert(
            "pos2min".to_string(),
            AestheticValue::standard_column("ymin".to_string()),
        );
        layer.mappings.insert(
            "pos2max".to_string(),
            AestheticValue::standard_column("ymax".to_string()),
        );

        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        let result = layer.validate_mapping(&Some(ctx), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_mapping_bidirectional_flipped_ok() {
        // Test that validation passes when all requirements are in flipped form
        use crate::AestheticContext;

        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("y".to_string()),
        );
        layer.mappings.insert(
            "pos1min".to_string(),
            AestheticValue::standard_column("xmin".to_string()),
        );
        layer.mappings.insert(
            "pos1max".to_string(),
            AestheticValue::standard_column("xmax".to_string()),
        );

        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        let result = layer.validate_mapping(&Some(ctx), false);
        assert!(result.is_ok());
    }
}
