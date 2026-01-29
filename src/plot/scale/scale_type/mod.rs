//! Scale type trait and implementations
//!
//! This module provides a trait-based design for scale types in ggsql.
//! Each scale type is implemented as its own struct, allowing for cleaner separation
//! of concerns and easier extensibility.
//!
//! # Architecture
//!
//! - `ScaleTypeKind`: Enum for pattern matching and serialization
//! - `ScaleTypeTrait`: Trait defining scale type behavior
//! - `ScaleType`: Wrapper struct holding an Arc<dyn ScaleTypeTrait>
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::plot::scale::{ScaleType, ScaleTypeKind};
//!
//! let continuous = ScaleType::continuous();
//! assert_eq!(continuous.scale_type_kind(), ScaleTypeKind::Continuous);
//! assert_eq!(continuous.name(), "continuous");
//! ```

use polars::prelude::{ChunkAgg, Column, DataType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::transform::{Transform, TransformKind};
use crate::plot::{ArrayElement, ColumnInfo, ParameterValue};

// Scale type implementations
mod binned;
mod continuous;
mod discrete;
mod identity;

// Re-export scale type structs for direct access if needed
pub use binned::Binned;
pub use continuous::Continuous;
pub use discrete::Discrete;
pub use identity::Identity;

// =============================================================================
// Scale Data Context
// =============================================================================

/// Input range for scale resolution
#[derive(Debug, Clone)]
pub enum InputRange {
    /// Continuous range: [min, max]
    Continuous(Vec<ArrayElement>),
    /// Discrete range: unique values
    Discrete(Vec<ArrayElement>),
}

/// Common context for scale resolution.
///
/// Aggregates data from multiple columns (across layers and aesthetic family).
/// Can be created from either schema information (pre-stat) or actual data (post-stat).
#[derive(Debug, Clone)]
pub struct ScaleDataContext {
    /// Input range: continuous [min, max] or discrete unique values
    pub range: Option<InputRange>,
    /// Data type of the column(s)
    pub dtype: Option<DataType>,
    /// Whether this is discrete data
    pub is_discrete: bool,
}

impl ScaleDataContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            range: None,
            dtype: None,
            is_discrete: false,
        }
    }

    /// Create from multiple schema ColumnInfos.
    ///
    /// Aggregates min/max across all columns for continuous data.
    /// Note: Discrete unique values are not available from schema.
    pub fn from_schemas(infos: &[ColumnInfo]) -> Self {
        if infos.is_empty() {
            return Self::new();
        }

        // Use first column's dtype and is_discrete (they should match)
        let dtype = Some(infos[0].dtype.clone());
        let is_discrete = infos[0].is_discrete;

        // Aggregate min/max across all columns
        let range = if is_discrete {
            None // Discrete unique values not available from schema
        } else {
            let mut global_min: Option<f64> = None;
            let mut global_max: Option<f64> = None;
            for info in infos {
                if let Some(ArrayElement::Number(min)) = &info.min {
                    global_min = Some(global_min.map_or(*min, |m| m.min(*min)));
                }
                if let Some(ArrayElement::Number(max)) = &info.max {
                    global_max = Some(global_max.map_or(*max, |m| m.max(*max)));
                }
            }
            match (global_min, global_max) {
                (Some(min), Some(max)) => Some(InputRange::Continuous(vec![
                    ArrayElement::Number(min),
                    ArrayElement::Number(max),
                ])),
                _ => None,
            }
        };

        Self {
            range,
            dtype,
            is_discrete,
        }
    }

    /// Create from multiple Polars Columns.
    ///
    /// Aggregates min/max or unique values across all columns.
    pub fn from_columns(columns: &[&Column], is_discrete: bool) -> Self {
        if columns.is_empty() {
            return Self::new();
        }

        let dtype = Some(columns[0].dtype().clone());

        let range = if is_discrete {
            // Aggregate unique values across all columns
            Some(InputRange::Discrete(compute_unique_values_multi(columns)))
        } else {
            // Aggregate min/max across all columns
            compute_column_range_multi(columns).map(InputRange::Continuous)
        };

        Self {
            range,
            dtype,
            is_discrete,
        }
    }

    /// Get the continuous range as [min, max] if available.
    pub fn continuous_range(&self) -> Option<&[ArrayElement]> {
        match &self.range {
            Some(InputRange::Continuous(r)) => Some(r),
            _ => None,
        }
    }

    /// Get the discrete range as unique values if available.
    pub fn discrete_range(&self) -> Option<&[ArrayElement]> {
        match &self.range {
            Some(InputRange::Discrete(r)) => Some(r),
            _ => None,
        }
    }
}

impl Default for ScaleDataContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute numeric min/max from multiple columns.
fn compute_column_range_multi(columns: &[&Column]) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<f64> = None;
    let mut global_max: Option<f64> = None;

    for column in columns {
        let series = column.as_materialized_series();
        if let Ok(ca) = series.cast(&DataType::Float64) {
            if let Ok(f64_series) = ca.f64() {
                if let Some(min) = f64_series.min() {
                    global_min = Some(global_min.map_or(min, |m| m.min(min)));
                }
                if let Some(max) = f64_series.max() {
                    global_max = Some(global_max.map_or(max, |m| m.max(max)));
                }
            }
        }
    }

    match (global_min, global_max) {
        (Some(min), Some(max)) => Some(vec![ArrayElement::Number(min), ArrayElement::Number(max)]),
        _ => None,
    }
}

/// Merge user-provided range with context-computed range.
///
/// Replaces Null values in user_range with corresponding values from context_range.
fn merge_with_context(
    user_range: &[ArrayElement],
    context_range: &[ArrayElement],
) -> Vec<ArrayElement> {
    user_range
        .iter()
        .enumerate()
        .map(|(i, elem)| {
            if matches!(elem, ArrayElement::Null) {
                // Replace Null with context value if available
                context_range.get(i).cloned().unwrap_or(ArrayElement::Null)
            } else {
                elem.clone()
            }
        })
        .collect()
}

/// Compute unique values from multiple columns, sorted.
fn compute_unique_values_multi(columns: &[&Column]) -> Vec<ArrayElement> {
    use polars::prelude::IntoSeries;
    use std::collections::BTreeSet;

    let mut unique_strings: BTreeSet<String> = BTreeSet::new();

    for column in columns {
        let series = column.as_materialized_series();
        // Try to get unique values as strings
        if let Ok(unique) = series.unique() {
            let unique_series = unique.into_series();
            if let Ok(str_series) = unique_series.str() {
                for s in str_series.into_iter().flatten() {
                    unique_strings.insert(s.to_string());
                }
            } else {
                // Non-string: convert to string representation
                for i in 0..unique_series.len() {
                    if let Ok(val) = unique_series.get(i) {
                        let s = format!("{}", val);
                        if s != "null" {
                            unique_strings.insert(s);
                        }
                    }
                }
            }
        }
    }

    unique_strings
        .into_iter()
        .map(ArrayElement::String)
        .collect()
}

/// Enum of all scale types for pattern matching and serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleTypeKind {
    /// Continuous numeric data (also used for temporal data with temporal transforms)
    Continuous,
    /// Categorical/discrete data
    Discrete,
    /// Binned/bucketed data (also supports temporal transforms)
    Binned,
    /// Identity scale (use inferred type)
    Identity,
}

impl std::fmt::Display for ScaleTypeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScaleTypeKind::Continuous => "continuous",
            ScaleTypeKind::Discrete => "discrete",
            ScaleTypeKind::Binned => "binned",
            ScaleTypeKind::Identity => "identity",
        };
        write!(f, "{}", s)
    }
}

/// Core trait for scale type behavior
///
/// Each scale type implements this trait. The trait is intentionally minimal
/// and backend-agnostic - no Vega-Lite or other writer-specific details.
pub trait ScaleTypeTrait: std::fmt::Debug + std::fmt::Display + Send + Sync {
    /// Returns which scale type this is (for pattern matching)
    fn scale_type_kind(&self) -> ScaleTypeKind;

    /// Canonical name for parsing and display
    fn name(&self) -> &'static str;

    /// Returns whether this scale type represents discrete/categorical data.
    /// Defaults to `true`. Overridden to return `false` for `Discrete` and `Identity`.
    fn is_discrete(&self) -> bool {
        true
    }

    /// Returns whether this scale type accepts the given data type
    fn allows_data_type(&self, dtype: &DataType) -> bool;

    /// Validate that all columns have compatible data types for this scale.
    /// Returns Ok(()) if valid, Err with details if any column is incompatible.
    fn validate_columns(&self, columns: &[&Column]) -> Result<(), String> {
        for col in columns {
            let dtype = col.dtype();
            if !self.allows_data_type(dtype) {
                return Err(format!(
                    "Column '{}' has type {:?} which is not compatible with {} scale",
                    col.name(),
                    dtype,
                    self.name()
                ));
            }
        }
        Ok(())
    }

    /// Resolve input range from user-provided range and data columns.
    ///
    /// Behavior varies by scale type:
    /// - Continuous/Binned: Compute min/max from data, merge with user range (nulls → computed)
    /// - Date/DateTime/Time: Compute min/max as ISO strings, merge with user range
    /// - Discrete: Collect unique values if no user range; error if user range contains nulls
    /// - Identity: Return Ok(None); error if user provided any input range
    ///
    /// The `properties` parameter provides access to SETTING values, including `expand`.
    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String>;

    /// Get default output range for an aesthetic.
    ///
    /// Returns sensible default ranges based on the aesthetic type and scale type.
    /// For example:
    /// - color/fill + discrete → standard categorical color palette (sized to input_range length)
    /// - size + continuous → [min_size, max_size] range
    /// - opacity + continuous → [0.2, 1.0] range
    ///
    /// The input_range is provided so discrete scales can size the output appropriately.
    ///
    /// Returns Ok(None) if no default is appropriate (e.g., x/y position aesthetics).
    /// Returns Err if the palette doesn't have enough colors for the input range.
    fn default_output_range(
        &self,
        _aesthetic: &str,
        _input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        Ok(None) // Default implementation: no default range
    }

    /// Returns list of allowed property names for SETTING clause.
    /// The aesthetic parameter allows different properties for different aesthetics.
    /// Default: empty (no properties allowed).
    fn allowed_properties(&self, _aesthetic: &str) -> &'static [&'static str] {
        &[]
    }

    /// Returns default value for a property, if any.
    /// Called by resolve_properties for allowed properties not in user input.
    /// The aesthetic parameter allows different defaults for different aesthetics.
    fn get_property_default(&self, _aesthetic: &str, _name: &str) -> Option<ParameterValue> {
        None
    }

    /// Returns the list of transforms this scale type supports.
    /// Transforms determine how data values are mapped to visual space.
    ///
    /// Default: only "identity" (no transformation).
    fn allowed_transforms(&self) -> &'static [TransformKind] {
        &[TransformKind::Identity]
    }

    /// Returns the default transform for this scale type, aesthetic, and column data type.
    ///
    /// The transform is inferred in order of priority:
    /// 1. Column data type (Date -> Date transform, DateTime -> DateTime transform, etc.)
    /// 2. Aesthetic defaults (`size` -> sqrt for area-proportional scaling)
    /// 3. Identity (default)
    ///
    /// The column_dtype parameter enables automatic temporal transform inference when
    /// a Date, DateTime, or Time column is mapped to an aesthetic.
    fn default_transform(&self, aesthetic: &str, column_dtype: Option<&DataType>) -> TransformKind {
        // First check column data type for temporal transforms
        if let Some(dtype) = column_dtype {
            match dtype {
                DataType::Date => return TransformKind::Date,
                DataType::Datetime(_, _) => return TransformKind::DateTime,
                DataType::Time => return TransformKind::Time,
                _ => {}
            }
        }

        // Fall back to aesthetic-based defaults
        match aesthetic {
            "size" => TransformKind::Sqrt,
            _ => TransformKind::Identity,
        }
    }

    /// Resolve and validate the transform.
    /// If user_transform is None, returns default_transform(aesthetic, column_dtype).
    /// If user_transform is Some, validates it's in allowed_transforms().
    fn resolve_transform(
        &self,
        aesthetic: &str,
        user_transform: Option<&Transform>,
        column_dtype: Option<&DataType>,
    ) -> Result<Transform, String> {
        match user_transform {
            None => Ok(Transform::from_kind(
                self.default_transform(aesthetic, column_dtype),
            )),
            Some(t) => {
                if self.allowed_transforms().contains(&t.transform_kind()) {
                    Ok(t.clone())
                } else {
                    Err(format!(
                        "Transform '{}' not supported for {} scale. Allowed: {}",
                        t.name(),
                        self.name(),
                        self.allowed_transforms()
                            .iter()
                            .map(|k| k.name())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            }
        }
    }

    /// Resolve and validate properties. NOT meant to be overridden by implementations.
    /// - Validates all properties are in allowed_properties()
    /// - Applies defaults via get_property_default()
    fn resolve_properties(
        &self,
        aesthetic: &str,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        let allowed = self.allowed_properties(aesthetic);

        // Check for unknown properties
        for key in properties.keys() {
            if !allowed.contains(&key.as_str()) {
                if allowed.is_empty() {
                    return Err(format!(
                        "{} scale does not support any SETTING properties",
                        self.name()
                    ));
                }
                return Err(format!(
                    "{} scale does not support SETTING '{}'. Allowed: {}",
                    self.name(),
                    key,
                    allowed.join(", ")
                ));
            }
        }

        // Start with user properties, add defaults for missing ones
        let mut resolved = properties.clone();
        for &prop_name in allowed {
            if !resolved.contains_key(prop_name) {
                if let Some(default) = self.get_property_default(aesthetic, prop_name) {
                    resolved.insert(prop_name.to_string(), default);
                }
            }
        }

        // Validate oob value if present
        if let Some(ParameterValue::String(oob)) = resolved.get("oob") {
            validate_oob(oob)?;

            // Discrete scales only support "censor" - no way to map unmapped values to output
            if self.scale_type_kind() == ScaleTypeKind::Discrete && oob != OOB_CENSOR {
                return Err(format!(
                    "Discrete scale only supports oob='censor'. Cannot use '{}' because \
                     values outside the input range have no corresponding output value.",
                    oob
                ));
            }
        }

        Ok(resolved)
    }

    /// Resolve break positions for this scale.
    ///
    /// Uses the resolved input range, properties, and transform to calculate
    /// appropriate break positions. This is transform-aware: log scales will
    /// produce breaks at powers of the base (or 1-2-5 pattern if pretty=true),
    /// sqrt scales will produce breaks that are evenly spaced in sqrt-space, etc.
    ///
    /// Returns None for scale types that don't support breaks (like Discrete, Identity).
    /// Returns Some(breaks) with appropriate break values otherwise.
    ///
    /// # Arguments
    /// * `input_range` - The resolved input range (min/max values)
    /// * `properties` - Resolved properties including `breaks` count and `pretty` flag
    /// * `transform` - The resolved transform
    fn resolve_breaks(
        &self,
        input_range: Option<&[ArrayElement]>,
        properties: &HashMap<String, ParameterValue>,
        transform: Option<&Transform>,
    ) -> Option<Vec<ArrayElement>> {
        // Only applicable to continuous-like scales
        if !self.supports_breaks() {
            return None;
        }

        // Extract min/max from input range
        let (min, max) = match input_range {
            Some(range) if range.len() >= 2 => {
                let min = match &range[0] {
                    ArrayElement::Number(n) => *n,
                    _ => return None,
                };
                let max = match &range[range.len() - 1] {
                    ArrayElement::Number(n) => *n,
                    _ => return None,
                };
                (min, max)
            }
            _ => return None,
        };

        if min >= max {
            return None;
        }

        // Get break count from properties
        let count = match properties.get("breaks") {
            Some(ParameterValue::Number(n)) => *n as usize,
            _ => super::breaks::DEFAULT_BREAK_COUNT,
        };

        // Get pretty flag from properties (defaults to true)
        let pretty = match properties.get("pretty") {
            Some(ParameterValue::Boolean(b)) => *b,
            _ => true,
        };

        // Use transform's calculate_breaks method if present and not identity
        let breaks = match transform {
            Some(t) if !t.is_identity() => t.calculate_breaks(min, max, count, pretty),
            _ => {
                // Identity transform or no transform - use default pretty/linear breaks
                if pretty {
                    super::breaks::pretty_breaks(min, max, count)
                } else {
                    super::breaks::linear_breaks(min, max, count)
                }
            }
        };

        if breaks.is_empty() {
            None
        } else {
            Some(breaks.into_iter().map(ArrayElement::Number).collect())
        }
    }

    /// Returns whether this scale type supports the `breaks` property.
    ///
    /// Continuous and Binned scales support breaks.
    /// Discrete and Identity scales do not.
    fn supports_breaks(&self) -> bool {
        matches!(
            self.scale_type_kind(),
            ScaleTypeKind::Continuous | ScaleTypeKind::Binned
        )
    }

    /// Resolve scale properties from data context.
    ///
    /// Called ONCE per scale, either:
    /// - Pre-stat (before build_layer_query): For Binned scales, using schema-derived context
    /// - Post-stat (after build_layer_query): For all other scales, using data-derived context
    ///
    /// Updates: input_range, transform, and properties["breaks"] on the scale.
    ///
    /// Default implementation:
    /// 1. Resolves input_range from context (or merges with existing partial range)
    /// 2. Resolves transform from context dtype if not set
    /// 3. If breaks is a scalar Number, calculates break positions and stores as Array
    fn resolve(
        &self,
        scale: &mut super::Scale,
        context: &ScaleDataContext,
        aesthetic: &str,
    ) -> Result<(), String> {
        // 1. Resolve properties (fills in defaults, validates)
        scale.properties = self.resolve_properties(aesthetic, &scale.properties)?;

        // 2. Resolve transform from context dtype and aesthetic
        let resolved_transform = self.resolve_transform(aesthetic, None, context.dtype.as_ref())?;
        scale.transform = Some(resolved_transform);

        // 3. Resolve input range
        // If scale already has input_range with Null values, merge with context
        // If scale has no input_range, use context
        if let Some(ref range) = context.range {
            let (mult, add) = get_expand_factors(&scale.properties);
            let context_range = match range {
                InputRange::Continuous(r) => expand_numeric_range(r, mult, add),
                InputRange::Discrete(r) => r.clone(),
            };

            if let Some(ref user_range) = scale.input_range {
                // Merge: replace Null values in user_range with values from context_range
                let merged = merge_with_context(user_range, &context_range);
                scale.input_range = Some(merged);
            } else {
                // No user range, use context range
                scale.input_range = Some(context_range);
            }
        }

        // 4. Calculate breaks if supports_breaks()
        // If breaks is a scalar Number (count), calculate actual break positions and store as Array
        // If breaks is already an Array, user provided explicit breaks - leave as-is
        if self.supports_breaks() {
            if let Some(ParameterValue::Number(_)) = scale.properties.get("breaks") {
                // Scalar count → calculate actual breaks and store as Array
                if let Some(breaks) = self.resolve_breaks(
                    scale.input_range.as_deref(),
                    &scale.properties,
                    scale.transform.as_ref(),
                ) {
                    scale
                        .properties
                        .insert("breaks".to_string(), ParameterValue::Array(breaks));
                }
            }
            // If breaks is already Array, user provided it - leave as-is
        }

        // 5. Apply label template if present (RENAMING * => '...')
        // For continuous/binned scales, apply to breaks array
        // For discrete scales, apply to input_range (domain values)
        if let Some(ref template) = scale.label_template {
            let values_to_label = if self.supports_breaks() {
                // Continuous/Binned: use breaks
                match scale.properties.get("breaks") {
                    Some(ParameterValue::Array(breaks)) => Some(breaks.clone()),
                    _ => None,
                }
            } else {
                // Discrete: use input_range
                scale.input_range.clone()
            };

            if let Some(values) = values_to_label {
                let generated_labels =
                    crate::format::apply_label_template(&values, template, &scale.label_mapping);
                scale.label_mapping = Some(generated_labels);
            }
        }

        // Mark scale as resolved
        scale.resolved = true;

        Ok(())
    }

    /// Pre-stat SQL transformation hook.
    ///
    /// Called inside build_layer_query to generate SQL that transforms data
    /// BEFORE stat transforms run. Returns SQL expression to wrap the column.
    ///
    /// Only Binned scales implement this (returns binning SQL).
    /// Default returns None (no transformation).
    fn pre_stat_transform_sql(&self, _column_name: &str, _scale: &super::Scale) -> Option<String> {
        None
    }
}

/// Wrapper struct for scale type trait objects
///
/// This provides a convenient interface for working with scale types while hiding
/// the complexity of trait objects.
#[derive(Clone)]
pub struct ScaleType(Arc<dyn ScaleTypeTrait>);

impl ScaleType {
    /// Create a Continuous scale type
    pub fn continuous() -> Self {
        Self(Arc::new(Continuous))
    }

    /// Create a Discrete scale type
    pub fn discrete() -> Self {
        Self(Arc::new(Discrete))
    }

    /// Create a Binned scale type
    pub fn binned() -> Self {
        Self(Arc::new(Binned))
    }

    /// Create an Identity scale type
    pub fn identity() -> Self {
        Self(Arc::new(Identity))
    }

    /// Infer scale type from a Polars data type.
    ///
    /// Maps data types to appropriate scale types:
    /// - Numeric types (Int*, UInt*, Float*) → Continuous
    /// - Temporal types (Date, Datetime, Time) → Continuous (with temporal transforms)
    /// - Boolean, String, other → Discrete
    ///
    /// Note: Temporal data uses Continuous scale type with temporal transforms
    /// (Date, DateTime, Time transforms) for break calculation and formatting.
    pub fn infer(dtype: &DataType) -> Self {
        match dtype {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Self::continuous(),
            // Temporal types are fundamentally continuous (days/µs/ns since epoch)
            // The temporal transform is inferred from the column data type
            DataType::Date | DataType::Datetime(_, _) | DataType::Time => Self::continuous(),
            DataType::Boolean | DataType::String => Self::discrete(),
            _ => Self::discrete(),
        }
    }

    /// Create a ScaleType from a ScaleTypeKind
    pub fn from_kind(kind: ScaleTypeKind) -> Self {
        match kind {
            ScaleTypeKind::Continuous => Self::continuous(),
            ScaleTypeKind::Discrete => Self::discrete(),
            ScaleTypeKind::Binned => Self::binned(),
            ScaleTypeKind::Identity => Self::identity(),
        }
    }

    /// Get the scale type kind (for pattern matching)
    pub fn scale_type_kind(&self) -> ScaleTypeKind {
        self.0.scale_type_kind()
    }

    /// Get the canonical name
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Check if this scale type represents discrete/categorical data
    pub fn is_discrete(&self) -> bool {
        self.0.is_discrete()
    }

    /// Check if this scale type accepts the given data type
    pub fn allows_data_type(&self, dtype: &DataType) -> bool {
        self.0.allows_data_type(dtype)
    }

    /// Validate that all columns have compatible data types for this scale
    pub fn validate_columns(&self, columns: &[&Column]) -> Result<(), String> {
        self.0.validate_columns(columns)
    }

    /// Resolve input range from user-provided range and data columns.
    ///
    /// Delegates to the underlying scale type implementation.
    pub fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        self.0.resolve_input_range(user_range, columns, properties)
    }

    /// Get default output range for an aesthetic.
    ///
    /// Delegates to the underlying scale type implementation.
    pub fn default_output_range(
        &self,
        aesthetic: &str,
        input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        self.0.default_output_range(aesthetic, input_range)
    }

    /// Resolve and validate properties.
    ///
    /// Validates all user-provided properties are allowed for this scale type,
    /// and fills in default values for missing properties.
    pub fn resolve_properties(
        &self,
        aesthetic: &str,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        self.0.resolve_properties(aesthetic, properties)
    }

    /// Returns the list of transforms this scale type supports.
    pub fn allowed_transforms(&self) -> &'static [TransformKind] {
        self.0.allowed_transforms()
    }

    /// Returns the default transform for this scale type, aesthetic, and column data type.
    pub fn default_transform(
        &self,
        aesthetic: &str,
        column_dtype: Option<&DataType>,
    ) -> TransformKind {
        self.0.default_transform(aesthetic, column_dtype)
    }

    /// Resolve and validate the transform.
    ///
    /// If user_transform is None, returns default_transform(aesthetic, column_dtype).
    /// If user_transform is Some, validates it's in allowed_transforms().
    pub fn resolve_transform(
        &self,
        aesthetic: &str,
        user_transform: Option<&Transform>,
        column_dtype: Option<&DataType>,
    ) -> Result<Transform, String> {
        self.0
            .resolve_transform(aesthetic, user_transform, column_dtype)
    }

    /// Resolve break positions for this scale.
    ///
    /// Uses the resolved input range, properties, and transform to calculate
    /// appropriate break positions. This is transform-aware.
    pub fn resolve_breaks(
        &self,
        input_range: Option<&[ArrayElement]>,
        properties: &HashMap<String, ParameterValue>,
        transform: Option<&Transform>,
    ) -> Option<Vec<ArrayElement>> {
        self.0.resolve_breaks(input_range, properties, transform)
    }

    /// Returns whether this scale type supports the `breaks` property.
    pub fn supports_breaks(&self) -> bool {
        self.0.supports_breaks()
    }

    /// Resolve scale properties from data context.
    ///
    /// Called ONCE per scale, either:
    /// - Pre-stat (before build_layer_query): For Binned scales, using schema-derived context
    /// - Post-stat (after build_layer_query): For all other scales, using data-derived context
    ///
    /// Updates: input_range, transform, and properties["breaks"] on the scale.
    pub fn resolve(
        &self,
        scale: &mut super::Scale,
        context: &ScaleDataContext,
        aesthetic: &str,
    ) -> Result<(), String> {
        self.0.resolve(scale, context, aesthetic)
    }

    /// Pre-stat SQL transformation hook.
    ///
    /// Called inside build_layer_query to generate SQL that transforms data
    /// BEFORE stat transforms run. Returns SQL expression to wrap the column.
    ///
    /// Only Binned scales implement this (returns binning SQL).
    pub fn pre_stat_transform_sql(
        &self,
        column_name: &str,
        scale: &super::Scale,
    ) -> Option<String> {
        self.0.pre_stat_transform_sql(column_name, scale)
    }
}

impl std::fmt::Debug for ScaleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScaleType::{:?}", self.scale_type_kind())
    }
}

impl std::fmt::Display for ScaleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for ScaleType {
    fn eq(&self, other: &Self) -> bool {
        self.scale_type_kind() == other.scale_type_kind()
    }
}

impl Eq for ScaleType {}

impl Serialize for ScaleType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.scale_type_kind().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ScaleType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let kind = ScaleTypeKind::deserialize(deserializer)?;
        Ok(ScaleType::from_kind(kind))
    }
}

// =============================================================================
// Shared helpers for input range resolution
// =============================================================================

/// Check if an aesthetic is a positional aesthetic (x, y, and variants).
/// Positional aesthetics support properties like `expand`.
pub(super) fn is_positional_aesthetic(aesthetic: &str) -> bool {
    matches!(
        aesthetic,
        "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
    )
}

/// Check if input range contains any Null placeholders
pub(super) fn input_range_has_nulls(range: &[ArrayElement]) -> bool {
    range.iter().any(|e| matches!(e, ArrayElement::Null))
}

/// Merge explicit range with inferred values (replace nulls with inferred)
pub(super) fn merge_with_inferred(
    explicit: &[ArrayElement],
    inferred: &[ArrayElement],
) -> Vec<ArrayElement> {
    explicit
        .iter()
        .enumerate()
        .map(|(i, exp)| {
            if matches!(exp, ArrayElement::Null) {
                inferred.get(i).cloned().unwrap_or(ArrayElement::Null)
            } else {
                exp.clone()
            }
        })
        .collect()
}

// =============================================================================
// Expansion helpers for continuous/temporal scales
// =============================================================================

/// Default multiplicative expansion factor for continuous/temporal scales.
pub(super) const DEFAULT_EXPAND_MULT: f64 = 0.05;

/// Default additive expansion factor for continuous/temporal scales.
pub(super) const DEFAULT_EXPAND_ADD: f64 = 0.0;

// =============================================================================
// Out-of-bounds (oob) handling constants and helpers
// =============================================================================

/// Out-of-bounds mode: set values outside range to NULL (removes from visualization)
pub const OOB_CENSOR: &str = "censor";
/// Out-of-bounds mode: clamp values to the closest limit
pub const OOB_SQUISH: &str = "squish";
/// Out-of-bounds mode: don't modify values (default for positional aesthetics)
pub const OOB_KEEP: &str = "keep";

/// Default oob mode for an aesthetic.
/// Positional aesthetics default to "keep", others default to "censor".
pub(super) fn default_oob(aesthetic: &str) -> &'static str {
    if is_positional_aesthetic(aesthetic) {
        OOB_KEEP
    } else {
        OOB_CENSOR
    }
}

/// Validate oob value is one of the allowed modes.
pub(super) fn validate_oob(value: &str) -> Result<(), String> {
    match value {
        OOB_CENSOR | OOB_SQUISH | OOB_KEEP => Ok(()),
        _ => Err(format!(
            "Invalid oob value '{}'. Must be 'censor', 'squish', or 'keep'",
            value
        )),
    }
}

/// Parse expand parameter value into (mult, add) factors.
/// Returns None if value is invalid.
///
/// Syntax:
/// - Single number: `expand => 0.05` → (0.05, 0.0)
/// - Two numbers: `expand => [0.05, 10]` → (0.05, 10.0)
pub(super) fn parse_expand_value(expand: &ParameterValue) -> Option<(f64, f64)> {
    match expand {
        ParameterValue::Number(m) => Some((*m, 0.0)),
        ParameterValue::Array(arr) if arr.len() >= 2 => {
            let mult = match &arr[0] {
                ArrayElement::Number(n) => *n,
                _ => return None,
            };
            let add = match &arr[1] {
                ArrayElement::Number(n) => *n,
                _ => return None,
            };
            Some((mult, add))
        }
        _ => None,
    }
}

/// Apply expansion to a numeric [min, max] range.
/// Returns expanded [min, max] as ArrayElements.
///
/// Formula: [min - range*mult - add, max + range*mult + add]
pub(super) fn expand_numeric_range(
    range: &[ArrayElement],
    mult: f64,
    add: f64,
) -> Vec<ArrayElement> {
    if range.len() < 2 {
        return range.to_vec();
    }

    let min = match &range[0] {
        ArrayElement::Number(n) => *n,
        _ => return range.to_vec(),
    };
    let max = match &range[1] {
        ArrayElement::Number(n) => *n,
        _ => return range.to_vec(),
    };

    let span = max - min;
    let expanded_min = min - span * mult - add;
    let expanded_max = max + span * mult + add;

    vec![
        ArrayElement::Number(expanded_min),
        ArrayElement::Number(expanded_max),
    ]
}

/// Get expand factors from properties, using defaults for continuous/temporal scales.
pub(super) fn get_expand_factors(properties: &HashMap<String, ParameterValue>) -> (f64, f64) {
    properties
        .get("expand")
        .and_then(parse_expand_value)
        .unwrap_or((DEFAULT_EXPAND_MULT, DEFAULT_EXPAND_ADD))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_type_creation() {
        let continuous = ScaleType::continuous();
        assert_eq!(continuous.scale_type_kind(), ScaleTypeKind::Continuous);

        let discrete = ScaleType::discrete();
        assert_eq!(discrete.scale_type_kind(), ScaleTypeKind::Discrete);

        let binned = ScaleType::binned();
        assert_eq!(binned.scale_type_kind(), ScaleTypeKind::Binned);
    }

    #[test]
    fn test_scale_type_equality() {
        let c1 = ScaleType::continuous();
        let c2 = ScaleType::continuous();
        let d1 = ScaleType::discrete();

        assert_eq!(c1, c2);
        assert_ne!(c1, d1);
    }

    #[test]
    fn test_scale_type_display() {
        assert_eq!(format!("{}", ScaleType::continuous()), "continuous");
        assert_eq!(format!("{}", ScaleType::binned()), "binned");
    }

    #[test]
    fn test_scale_type_kind_display() {
        assert_eq!(format!("{}", ScaleTypeKind::Continuous), "continuous");
        assert_eq!(format!("{}", ScaleTypeKind::Identity), "identity");
    }

    #[test]
    fn test_scale_type_from_kind() {
        let scale_type = ScaleType::from_kind(ScaleTypeKind::Binned);
        assert_eq!(scale_type.scale_type_kind(), ScaleTypeKind::Binned);
    }

    #[test]
    fn test_scale_type_name() {
        assert_eq!(ScaleType::continuous().name(), "continuous");
        assert_eq!(ScaleType::binned().name(), "binned");
        assert_eq!(ScaleType::identity().name(), "identity");
    }

    #[test]
    fn test_scale_type_serialization() {
        let continuous = ScaleType::continuous();
        let json = serde_json::to_string(&continuous).unwrap();
        assert_eq!(json, "\"continuous\"");

        let deserialized: ScaleType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.scale_type_kind(), ScaleTypeKind::Continuous);
    }

    #[test]
    fn test_scale_type_is_discrete() {
        // Default is true for most scale types (not truly discrete, but inverted for legacy reasons)
        assert!(ScaleType::continuous().is_discrete());
        assert!(ScaleType::binned().is_discrete());

        // Discrete and Identity return false
        assert!(!ScaleType::discrete().is_discrete());
        assert!(!ScaleType::identity().is_discrete());
    }

    #[test]
    fn test_allows_data_type() {
        use polars::prelude::TimeUnit;

        // Continuous allows numeric types AND temporal types (temporal is fundamentally continuous)
        assert!(ScaleType::continuous().allows_data_type(&DataType::Float64));
        assert!(ScaleType::continuous().allows_data_type(&DataType::Int32));
        assert!(ScaleType::continuous().allows_data_type(&DataType::UInt64));
        assert!(ScaleType::continuous().allows_data_type(&DataType::Date));
        assert!(ScaleType::continuous()
            .allows_data_type(&DataType::Datetime(TimeUnit::Microseconds, None)));
        assert!(ScaleType::continuous().allows_data_type(&DataType::Time));
        assert!(!ScaleType::continuous().allows_data_type(&DataType::String));

        // Binned allows numeric types AND temporal types (same as continuous)
        assert!(ScaleType::binned().allows_data_type(&DataType::Float64));
        assert!(ScaleType::binned().allows_data_type(&DataType::Int32));
        assert!(!ScaleType::binned().allows_data_type(&DataType::String));

        // Discrete allows string/boolean
        assert!(ScaleType::discrete().allows_data_type(&DataType::String));
        assert!(ScaleType::discrete().allows_data_type(&DataType::Boolean));
        assert!(!ScaleType::discrete().allows_data_type(&DataType::Float64));
        assert!(!ScaleType::discrete().allows_data_type(&DataType::Int32));

        // Identity allows everything
        assert!(ScaleType::identity().allows_data_type(&DataType::String));
        assert!(ScaleType::identity().allows_data_type(&DataType::Float64));
        assert!(ScaleType::identity().allows_data_type(&DataType::Date));
        assert!(ScaleType::identity().allows_data_type(&DataType::Time));
    }

    #[test]
    fn test_validate_columns() {
        use polars::prelude::*;

        let float_col: Column = Series::new("x".into(), &[1.0f64, 2.0, 3.0]).into();
        let string_col: Column = Series::new("y".into(), &["a", "b", "c"]).into();
        let int_col: Column = Series::new("z".into(), &[1i32, 2, 3]).into();

        // Continuous should accept numeric columns
        assert!(ScaleType::continuous()
            .validate_columns(&[&float_col])
            .is_ok());
        assert!(ScaleType::continuous()
            .validate_columns(&[&int_col])
            .is_ok());
        assert!(ScaleType::continuous()
            .validate_columns(&[&float_col, &int_col])
            .is_ok());

        // Continuous should reject string column
        let result = ScaleType::continuous().validate_columns(&[&string_col]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Column 'y'"));

        // Discrete should accept string column
        assert!(ScaleType::discrete()
            .validate_columns(&[&string_col])
            .is_ok());

        // Discrete should reject numeric column
        let result = ScaleType::discrete().validate_columns(&[&float_col]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Column 'x'"));

        // Identity should accept any column
        assert!(ScaleType::identity()
            .validate_columns(&[&float_col])
            .is_ok());
        assert!(ScaleType::identity()
            .validate_columns(&[&string_col])
            .is_ok());
        assert!(ScaleType::identity()
            .validate_columns(&[&float_col, &string_col, &int_col])
            .is_ok());
    }

    #[test]
    fn test_resolve_input_range_continuous_no_expand() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[1.0f64, 5.0, 10.0]).into();

        // Disable expansion for predictable test values
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.0));

        // No user range -> compute from data
        let result = ScaleType::continuous()
            .resolve_input_range(None, &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![ArrayElement::Number(1.0), ArrayElement::Number(10.0)])
        );

        // User range with nulls -> merge
        let user = vec![ArrayElement::Null, ArrayElement::Number(100.0)];
        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user), &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![ArrayElement::Number(1.0), ArrayElement::Number(100.0)])
        );

        // User range without nulls -> keep as-is
        let user = vec![ArrayElement::Number(0.0), ArrayElement::Number(50.0)];
        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user), &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(50.0)])
        );
    }

    #[test]
    fn test_resolve_input_range_discrete() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &["b", "a", "c"]).into();
        let props = HashMap::new();

        // No user range -> compute unique sorted values
        let result = ScaleType::discrete()
            .resolve_input_range(None, &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![
                ArrayElement::String("a".into()),
                ArrayElement::String("b".into()),
                ArrayElement::String("c".into()),
            ])
        );

        // User range without nulls -> keep as-is
        let user = vec![
            ArrayElement::String("x".into()),
            ArrayElement::String("y".into()),
        ];
        let result = ScaleType::discrete()
            .resolve_input_range(Some(&user), &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![
                ArrayElement::String("x".into()),
                ArrayElement::String("y".into()),
            ])
        );
    }

    #[test]
    fn test_resolve_input_range_discrete_rejects_nulls() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &["a", "b"]).into();
        let user = vec![ArrayElement::Null, ArrayElement::String("c".into())];
        let props = HashMap::new();

        let result = ScaleType::discrete().resolve_input_range(Some(&user), &[&col], &props);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("null placeholder"));
    }

    #[test]
    fn test_resolve_input_range_identity_rejects_range() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[1.0f64]).into();
        let user = vec![ArrayElement::Number(0.0), ArrayElement::Number(10.0)];
        let props = HashMap::new();

        let result = ScaleType::identity().resolve_input_range(Some(&user), &[&col], &props);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not support input range"));
    }

    #[test]
    fn test_resolve_input_range_identity_no_range() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[1.0f64]).into();
        let props = HashMap::new();

        let result = ScaleType::identity()
            .resolve_input_range(None, &[&col], &props)
            .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_input_range_binned_no_expand() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[1i32, 5, 10]).into();

        // Disable expansion for predictable test values
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.0));

        // No user range -> compute from data
        let result = ScaleType::binned()
            .resolve_input_range(None, &[&col], &props)
            .unwrap();
        assert_eq!(
            result,
            Some(vec![ArrayElement::Number(1.0), ArrayElement::Number(10.0)])
        );
    }

    #[test]
    fn test_scale_type_infer() {
        use polars::prelude::TimeUnit;

        // Numeric → Continuous
        assert_eq!(ScaleType::infer(&DataType::Int32), ScaleType::continuous());
        assert_eq!(ScaleType::infer(&DataType::Int64), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Float64),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::UInt16), ScaleType::continuous());

        // Temporal - now inferred as Continuous (with temporal transforms)
        assert_eq!(ScaleType::infer(&DataType::Date), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Datetime(TimeUnit::Microseconds, None)),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::Time), ScaleType::continuous());

        // Discrete
        assert_eq!(ScaleType::infer(&DataType::String), ScaleType::discrete());
        assert_eq!(ScaleType::infer(&DataType::Boolean), ScaleType::discrete());
    }

    // =========================================================================
    // Expansion Tests
    // =========================================================================

    #[test]
    fn test_parse_expand_value_number() {
        let val = ParameterValue::Number(0.1);
        let (mult, add) = parse_expand_value(&val).unwrap();
        assert!((mult - 0.1).abs() < 1e-10);
        assert!((add - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_expand_value_array() {
        let val =
            ParameterValue::Array(vec![ArrayElement::Number(0.05), ArrayElement::Number(10.0)]);
        let (mult, add) = parse_expand_value(&val).unwrap();
        assert!((mult - 0.05).abs() < 1e-10);
        assert!((add - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_expand_value_invalid() {
        let val = ParameterValue::String("invalid".to_string());
        assert!(parse_expand_value(&val).is_none());

        let val = ParameterValue::Array(vec![ArrayElement::Number(0.05)]);
        assert!(parse_expand_value(&val).is_none()); // Too few elements
    }

    #[test]
    fn test_expand_numeric_range_mult_only() {
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let expanded = expand_numeric_range(&range, 0.05, 0.0);
        // span = 100, expanded = [-5, 105]
        assert_eq!(expanded[0], ArrayElement::Number(-5.0));
        assert_eq!(expanded[1], ArrayElement::Number(105.0));
    }

    #[test]
    fn test_expand_numeric_range_with_add() {
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let expanded = expand_numeric_range(&range, 0.05, 10.0);
        // span = 100, mult_expansion = 5, add_expansion = 10
        // expanded = [0 - 5 - 10, 100 + 5 + 10] = [-15, 115]
        assert_eq!(expanded[0], ArrayElement::Number(-15.0));
        assert_eq!(expanded[1], ArrayElement::Number(115.0));
    }

    #[test]
    fn test_expand_numeric_range_zero_disables() {
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let expanded = expand_numeric_range(&range, 0.0, 0.0);
        // No expansion
        assert_eq!(expanded[0], ArrayElement::Number(0.0));
        assert_eq!(expanded[1], ArrayElement::Number(100.0));
    }

    #[test]
    fn test_expand_default_applied() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[0.0f64, 100.0]).into();

        // Default properties (no expand key) should use DEFAULT_EXPAND_MULT = 0.05
        let props = HashMap::new();

        let result = ScaleType::continuous()
            .resolve_input_range(None, &[&col], &props)
            .unwrap()
            .unwrap();

        // span = 100, 5% expansion = 5 on each side
        // expected: [-5, 105]
        assert_eq!(result[0], ArrayElement::Number(-5.0));
        assert_eq!(result[1], ArrayElement::Number(105.0));
    }

    #[test]
    fn test_expand_custom_value() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[0.0f64, 100.0]).into();

        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        let result = ScaleType::continuous()
            .resolve_input_range(None, &[&col], &props)
            .unwrap()
            .unwrap();

        // span = 100, 10% expansion = 10 on each side
        // expected: [-10, 110]
        assert_eq!(result[0], ArrayElement::Number(-10.0));
        assert_eq!(result[1], ArrayElement::Number(110.0));
    }

    #[test]
    fn test_expand_with_additive() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[0.0f64, 100.0]).into();

        let mut props = HashMap::new();
        props.insert(
            "expand".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.05), ArrayElement::Number(10.0)]),
        );

        let result = ScaleType::continuous()
            .resolve_input_range(None, &[&col], &props)
            .unwrap()
            .unwrap();

        // span = 100, 5% expansion = 5, additive = 10
        // expected: [-15, 115]
        assert_eq!(result[0], ArrayElement::Number(-15.0));
        assert_eq!(result[1], ArrayElement::Number(115.0));
    }

    #[test]
    fn test_expand_applied_to_user_range() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[50.0f64]).into();

        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.05));

        // User provides explicit range
        let user_range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];

        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user_range), &[&col], &props)
            .unwrap()
            .unwrap();

        // span = 100, 5% expansion = 5 on each side
        // expected: [-5, 105]
        assert_eq!(result[0], ArrayElement::Number(-5.0));
        assert_eq!(result[1], ArrayElement::Number(105.0));
    }

    #[test]
    fn test_expand_zero_disables() {
        use polars::prelude::*;
        let col: Column = Series::new("x".into(), &[0.0f64, 100.0]).into();

        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.0));

        let result = ScaleType::continuous()
            .resolve_input_range(None, &[&col], &props)
            .unwrap()
            .unwrap();

        // No expansion
        assert_eq!(result[0], ArrayElement::Number(0.0));
        assert_eq!(result[1], ArrayElement::Number(100.0));
    }

    // =========================================================================
    // resolve_properties Tests
    // =========================================================================

    #[test]
    fn test_resolve_properties_unknown_rejected() {
        let mut props = HashMap::new();
        props.insert("unknown".to_string(), ParameterValue::Number(1.0));

        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("unknown"));
        assert!(err.contains("expand")); // Should suggest allowed properties
    }

    #[test]
    fn test_resolve_properties_default_expand() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();

        assert!(resolved.contains_key("expand"));
        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => {
                assert!((n - DEFAULT_EXPAND_MULT).abs() < 1e-10);
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_resolve_properties_preserves_user_value() {
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();

        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => assert!((n - 0.1).abs() < 1e-10),
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_discrete_rejects_expand_property() {
        // Discrete scales support oob but not expand
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        let result = ScaleType::discrete().resolve_properties("color", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("discrete"));
        assert!(err.contains("does not support SETTING 'expand'"));
    }

    #[test]
    fn test_identity_rejects_properties() {
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        let result = ScaleType::identity().resolve_properties("x", &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_properties_binned_supports_expand() {
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.2));

        let resolved = ScaleType::binned().resolve_properties("x", &props).unwrap();
        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => assert!((n - 0.2).abs() < 1e-10),
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_resolve_properties_continuous_supports_expand() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();

        // Should have default expand
        assert!(resolved.contains_key("expand"));
    }

    #[test]
    fn test_resolve_properties_defaults_for_discrete() {
        // Empty properties should be allowed for discrete, defaults to oob and reverse
        let props = HashMap::new();
        let result = ScaleType::discrete().resolve_properties("color", &props);
        assert!(result.is_ok());
        let resolved = result.unwrap();
        // Discrete now supports oob and reverse with default values
        assert!(resolved.contains_key("oob"));
        assert!(resolved.contains_key("reverse"));
        assert_eq!(resolved.len(), 2); // oob and reverse
    }

    #[test]
    fn test_expand_only_for_positional_aesthetics() {
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        // Positional aesthetics should allow expand
        assert!(ScaleType::continuous()
            .resolve_properties("x", &props)
            .is_ok());
        assert!(ScaleType::continuous()
            .resolve_properties("y", &props)
            .is_ok());
        assert!(ScaleType::continuous()
            .resolve_properties("xmin", &props)
            .is_ok());
        assert!(ScaleType::continuous()
            .resolve_properties("ymax", &props)
            .is_ok());

        // Non-positional aesthetics should reject expand (but allow oob)
        let result = ScaleType::continuous().resolve_properties("color", &props);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support SETTING 'expand'"));

        let result = ScaleType::continuous().resolve_properties("size", &props);
        assert!(result.is_err());

        let result = ScaleType::continuous().resolve_properties("opacity", &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_default_expand_for_non_positional() {
        // Non-positional aesthetics should not get default expand
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("color", &props)
            .unwrap();
        assert!(!resolved.contains_key("expand"));
        // But they do get default oob
        assert!(resolved.contains_key("oob"));

        // Positional aesthetics should get default expand
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert!(resolved.contains_key("expand"));
    }

    // =========================================================================
    // OOB Tests
    // =========================================================================

    #[test]
    fn test_oob_default_positional() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(
            resolved.get("oob"),
            Some(&ParameterValue::String("keep".into()))
        );
    }

    #[test]
    fn test_oob_default_positional_variants() {
        let props = HashMap::new();
        for aesthetic in &["y", "xmin", "xmax", "ymin", "ymax", "xend", "yend"] {
            let resolved = ScaleType::continuous()
                .resolve_properties(aesthetic, &props)
                .unwrap();
            assert_eq!(
                resolved.get("oob"),
                Some(&ParameterValue::String("keep".into())),
                "Expected 'keep' default for positional aesthetic '{}'",
                aesthetic
            );
        }
    }

    #[test]
    fn test_oob_default_non_positional() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("color", &props)
            .unwrap();
        assert_eq!(
            resolved.get("oob"),
            Some(&ParameterValue::String("censor".into()))
        );
    }

    #[test]
    fn test_oob_default_non_positional_variants() {
        let props = HashMap::new();
        for aesthetic in &["size", "opacity", "fill", "stroke"] {
            let resolved = ScaleType::continuous()
                .resolve_properties(aesthetic, &props)
                .unwrap();
            assert_eq!(
                resolved.get("oob"),
                Some(&ParameterValue::String("censor".into())),
                "Expected 'censor' default for non-positional aesthetic '{}'",
                aesthetic
            );
        }
    }

    #[test]
    fn test_oob_valid_values() {
        // All valid oob values should be accepted
        for oob_value in &["censor", "squish", "keep"] {
            let mut props = HashMap::new();
            props.insert(
                "oob".to_string(),
                ParameterValue::String(oob_value.to_string()),
            );

            let result = ScaleType::continuous().resolve_properties("x", &props);
            assert!(
                result.is_ok(),
                "Expected oob='{}' to be valid, got error",
                oob_value
            );
        }
    }

    #[test]
    fn test_oob_invalid_value_rejected() {
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("invalid".into()));

        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid oob value"));
        assert!(err.contains("invalid"));
        assert!(err.contains("censor"));
        assert!(err.contains("squish"));
        assert!(err.contains("keep"));
    }

    #[test]
    fn test_oob_user_value_preserved() {
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("squish".into()));

        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(
            resolved.get("oob"),
            Some(&ParameterValue::String("squish".into()))
        );
    }

    #[test]
    fn test_oob_supported_by_all_continuous_scales() {
        // All continuous-like scales should support oob
        let props = HashMap::new();

        for scale_type in &[ScaleType::continuous(), ScaleType::binned()] {
            let resolved = scale_type.resolve_properties("color", &props).unwrap();
            assert!(
                resolved.contains_key("oob"),
                "Scale {:?} should support oob",
                scale_type.scale_type_kind()
            );
        }
    }

    #[test]
    fn test_oob_not_supported_by_identity() {
        // Identity scale should not support oob
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("censor".into()));

        let result = ScaleType::identity().resolve_properties("color", &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_discrete_supports_oob_censor_only() {
        // Discrete scale only supports oob='censor'
        let mut props = HashMap::new();

        // censor should work
        props.insert("oob".to_string(), ParameterValue::String("censor".into()));
        let result = ScaleType::discrete().resolve_properties("color", &props);
        assert!(result.is_ok());
    }

    #[test]
    fn test_discrete_rejects_oob_keep() {
        // Discrete scale should reject keep (no output value for unmapped inputs)
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("keep".into()));

        let result = ScaleType::discrete().resolve_properties("color", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Discrete scale only supports oob='censor'"));
    }

    #[test]
    fn test_discrete_rejects_oob_squish() {
        // Discrete scale should reject squish (no natural "closest" value)
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("squish".into()));

        let result = ScaleType::discrete().resolve_properties("color", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Discrete scale only supports oob='censor'"));
    }

    #[test]
    fn test_discrete_default_oob() {
        // Discrete scale should always default to 'censor'
        let props = HashMap::new();

        let resolved = ScaleType::discrete()
            .resolve_properties("color", &props)
            .unwrap();
        assert_eq!(
            resolved.get("oob"),
            Some(&ParameterValue::String("censor".into()))
        );
    }

    // =========================================================================
    // Transform Tests
    // =========================================================================

    #[test]
    fn test_continuous_default_transform_identity() {
        // Most aesthetics default to identity (when no column dtype is specified)
        assert_eq!(
            ScaleType::continuous().default_transform("x", None),
            TransformKind::Identity
        );
        assert_eq!(
            ScaleType::continuous().default_transform("y", None),
            TransformKind::Identity
        );
        assert_eq!(
            ScaleType::continuous().default_transform("color", None),
            TransformKind::Identity
        );
    }

    #[test]
    fn test_continuous_default_transform_size_is_sqrt() {
        // Size aesthetic defaults to sqrt for area-proportional scaling
        assert_eq!(
            ScaleType::continuous().default_transform("size", None),
            TransformKind::Sqrt
        );
    }

    #[test]
    fn test_continuous_default_transform_infers_temporal() {
        use polars::prelude::*;

        // Date column -> Date transform
        assert_eq!(
            ScaleType::continuous().default_transform("x", Some(&DataType::Date)),
            TransformKind::Date
        );

        // DateTime column -> DateTime transform
        assert_eq!(
            ScaleType::continuous()
                .default_transform("x", Some(&DataType::Datetime(TimeUnit::Microseconds, None))),
            TransformKind::DateTime
        );

        // Time column -> Time transform
        assert_eq!(
            ScaleType::continuous().default_transform("x", Some(&DataType::Time)),
            TransformKind::Time
        );

        // Non-temporal column -> falls back to aesthetic default
        assert_eq!(
            ScaleType::continuous().default_transform("x", Some(&DataType::Int64)),
            TransformKind::Identity
        );
        assert_eq!(
            ScaleType::continuous().default_transform("size", Some(&DataType::Float64)),
            TransformKind::Sqrt
        );
    }

    #[test]
    fn test_continuous_allows_log_transforms() {
        let transforms = ScaleType::continuous().allowed_transforms();
        assert!(transforms.contains(&TransformKind::Identity));
        assert!(transforms.contains(&TransformKind::Log10));
        assert!(transforms.contains(&TransformKind::Log2));
        assert!(transforms.contains(&TransformKind::Log));
        assert!(transforms.contains(&TransformKind::Sqrt));
        assert!(transforms.contains(&TransformKind::Asinh));
        assert!(transforms.contains(&TransformKind::PseudoLog));
    }

    #[test]
    fn test_binned_allows_log_transforms() {
        let transforms = ScaleType::binned().allowed_transforms();
        assert!(transforms.contains(&TransformKind::Identity));
        assert!(transforms.contains(&TransformKind::Log10));
        assert!(transforms.contains(&TransformKind::Sqrt));
        assert!(transforms.contains(&TransformKind::Asinh));
    }

    #[test]
    fn test_binned_default_transform_size_is_sqrt() {
        assert_eq!(
            ScaleType::binned().default_transform("size", None),
            TransformKind::Sqrt
        );
        assert_eq!(
            ScaleType::binned().default_transform("x", None),
            TransformKind::Identity
        );
    }

    #[test]
    fn test_discrete_only_allows_identity_transform() {
        let transforms = ScaleType::discrete().allowed_transforms();
        assert_eq!(transforms, &[TransformKind::Identity]);
    }

    #[test]
    fn test_discrete_rejects_log_transform() {
        let log = Transform::log();
        let result = ScaleType::discrete().resolve_transform("color", Some(&log), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("log"));
        assert!(err.contains("not supported"));
        assert!(err.contains("discrete"));
    }

    #[test]
    fn test_identity_scale_only_allows_identity_transform() {
        let transforms = ScaleType::identity().allowed_transforms();
        assert_eq!(transforms, &[TransformKind::Identity]);
    }

    #[test]
    fn test_resolve_transform_fills_default() {
        // Without user input, fills in default
        let result = ScaleType::continuous().resolve_transform("x", None, None);
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Identity);

        let result = ScaleType::continuous().resolve_transform("size", None, None);
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Sqrt);
    }

    #[test]
    fn test_resolve_transform_validates_user_input() {
        // Valid user input is accepted
        let log = Transform::log();
        let result = ScaleType::continuous().resolve_transform("y", Some(&log), None);
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Log10);

        // Invalid user input is rejected (we can't easily test this anymore since
        // Transform::from_name returns None for invalid names)
    }

    #[test]
    fn test_continuous_accepts_all_valid_transforms() {
        for kind in &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Log2,
            TransformKind::Log,
            TransformKind::Sqrt,
            TransformKind::Asinh,
            TransformKind::PseudoLog,
            TransformKind::Date,
            TransformKind::DateTime,
            TransformKind::Time,
        ] {
            let transform = Transform::from_kind(*kind);
            let result = ScaleType::continuous().resolve_transform("y", Some(&transform), None);
            assert!(
                result.is_ok(),
                "Expected transform '{:?}' to be valid for continuous scale",
                kind
            );
            assert_eq!(result.unwrap().transform_kind(), *kind);
        }
    }

    // =========================================================================
    // Reverse Property Tests
    // =========================================================================

    #[test]
    fn test_reverse_property_default_false() {
        let props = HashMap::new();

        // Continuous scale should have reverse default to false
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(
            resolved.get("reverse"),
            Some(&ParameterValue::Boolean(false))
        );

        // Same for non-positional aesthetics
        let resolved = ScaleType::continuous()
            .resolve_properties("color", &props)
            .unwrap();
        assert_eq!(
            resolved.get("reverse"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_reverse_property_accepts_true() {
        let mut props = HashMap::new();
        props.insert("reverse".to_string(), ParameterValue::Boolean(true));

        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(
            resolved.get("reverse"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    #[test]
    fn test_reverse_property_supported_by_all_scales() {
        let mut props = HashMap::new();
        props.insert("reverse".to_string(), ParameterValue::Boolean(true));

        // All scale types should support reverse property
        for scale_type in &[
            ScaleType::continuous(),
            ScaleType::binned(),
            ScaleType::discrete(),
        ] {
            let result = scale_type.resolve_properties("x", &props);
            assert!(
                result.is_ok(),
                "Scale {:?} should support reverse property",
                scale_type.scale_type_kind()
            );
            let resolved = result.unwrap();
            assert_eq!(
                resolved.get("reverse"),
                Some(&ParameterValue::Boolean(true)),
                "Scale {:?} should preserve reverse=true",
                scale_type.scale_type_kind()
            );
        }
    }

    #[test]
    fn test_identity_scale_rejects_reverse_property() {
        // Identity scale should not support reverse (no properties at all)
        let mut props = HashMap::new();
        props.insert("reverse".to_string(), ParameterValue::Boolean(true));

        let result = ScaleType::identity().resolve_properties("x", &props);
        assert!(result.is_err());
    }

    // =========================================================================
    // Breaks and Pretty Property Tests
    // =========================================================================

    #[test]
    fn test_breaks_property_default_is_5() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(resolved.get("breaks"), Some(&ParameterValue::Number(5.0)));
    }

    #[test]
    fn test_pretty_property_default_is_true() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(resolved.get("pretty"), Some(&ParameterValue::Boolean(true)));
    }

    #[test]
    fn test_breaks_property_accepts_number() {
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(10.0));

        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert_eq!(resolved.get("breaks"), Some(&ParameterValue::Number(10.0)));
    }

    #[test]
    fn test_breaks_property_accepts_array() {
        use crate::plot::ArrayElement;

        let mut props = HashMap::new();
        props.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ]),
        );

        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pretty_property_accepts_false() {
        let mut props = HashMap::new();
        props.insert("pretty".to_string(), ParameterValue::Boolean(false));

        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert_eq!(
            resolved.get("pretty"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_breaks_supported_by_continuous_scales() {
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(5.0));

        for scale_type in &[ScaleType::continuous(), ScaleType::binned()] {
            let result = scale_type.resolve_properties("x", &props);
            assert!(
                result.is_ok(),
                "Scale {:?} should support breaks property",
                scale_type.scale_type_kind()
            );
        }
    }

    #[test]
    fn test_discrete_does_not_support_breaks() {
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(5.0));

        let result = ScaleType::discrete().resolve_properties("x", &props);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support SETTING 'breaks'"));
    }

    #[test]
    fn test_identity_does_not_support_breaks() {
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(5.0));

        let result = ScaleType::identity().resolve_properties("x", &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_breaks_available_for_non_positional_aesthetics() {
        // breaks should work for color legends too
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(4.0));

        let result = ScaleType::continuous().resolve_properties("color", &props);
        assert!(result.is_ok());
    }

    // =========================================================================
    // resolve_breaks Tests
    // =========================================================================

    #[test]
    fn test_resolve_breaks_continuous_identity() {
        let input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(5.0));
        props.insert("pretty".to_string(), ParameterValue::Boolean(true));

        let identity = Transform::identity();
        let breaks =
            ScaleType::continuous().resolve_breaks(input_range.as_deref(), &props, Some(&identity));

        assert!(breaks.is_some());
        let breaks = breaks.unwrap();
        // Pretty breaks should produce nice numbers
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_resolve_breaks_continuous_log10() {
        let input_range = Some(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(1000.0),
        ]);
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(10.0));
        props.insert("pretty".to_string(), ParameterValue::Boolean(false));

        let log10 = Transform::log();
        let breaks =
            ScaleType::continuous().resolve_breaks(input_range.as_deref(), &props, Some(&log10));

        assert!(breaks.is_some());
        let breaks = breaks.unwrap();
        // Should have powers of 10: 1, 10, 100, 1000
        assert!(breaks.contains(&ArrayElement::Number(1.0)));
        assert!(breaks.contains(&ArrayElement::Number(10.0)));
        assert!(breaks.contains(&ArrayElement::Number(100.0)));
        assert!(breaks.contains(&ArrayElement::Number(1000.0)));
    }

    #[test]
    fn test_resolve_breaks_continuous_log10_pretty() {
        let input_range = Some(vec![ArrayElement::Number(1.0), ArrayElement::Number(100.0)]);
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(10.0));
        props.insert("pretty".to_string(), ParameterValue::Boolean(true));

        let log10 = Transform::log();
        let breaks =
            ScaleType::continuous().resolve_breaks(input_range.as_deref(), &props, Some(&log10));

        assert!(breaks.is_some());
        let breaks = breaks.unwrap();
        // Should have 1-2-5 pattern: 1, 2, 5, 10, 20, 50, 100
        assert!(breaks.contains(&ArrayElement::Number(1.0)));
        assert!(breaks.contains(&ArrayElement::Number(10.0)));
        assert!(breaks.contains(&ArrayElement::Number(100.0)));
    }

    #[test]
    fn test_resolve_breaks_continuous_sqrt() {
        let input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        let mut props = HashMap::new();
        props.insert("breaks".to_string(), ParameterValue::Number(5.0));
        props.insert("pretty".to_string(), ParameterValue::Boolean(false));

        let sqrt = Transform::sqrt();
        let breaks =
            ScaleType::continuous().resolve_breaks(input_range.as_deref(), &props, Some(&sqrt));

        assert!(breaks.is_some());
        let breaks = breaks.unwrap();
        assert_eq!(breaks.len(), 5);
    }

    #[test]
    fn test_resolve_breaks_discrete_returns_none() {
        let input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        let props = HashMap::new();

        let breaks = ScaleType::discrete().resolve_breaks(input_range.as_deref(), &props, None);

        // Discrete scales don't support breaks
        assert!(breaks.is_none());
    }

    #[test]
    fn test_resolve_breaks_identity_scale_returns_none() {
        let input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        let props = HashMap::new();

        let breaks = ScaleType::identity().resolve_breaks(input_range.as_deref(), &props, None);

        // Identity scales don't support breaks
        assert!(breaks.is_none());
    }

    #[test]
    fn test_resolve_breaks_no_input_range() {
        let props = HashMap::new();

        let breaks = ScaleType::continuous().resolve_breaks(None, &props, None);

        // Can't calculate breaks without input range
        assert!(breaks.is_none());
    }

    #[test]
    fn test_resolve_breaks_uses_default_count() {
        let input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        let props = HashMap::new(); // No explicit breaks count

        let identity = Transform::identity();
        let breaks =
            ScaleType::continuous().resolve_breaks(input_range.as_deref(), &props, Some(&identity));

        assert!(breaks.is_some());
        // Default is 5 breaks, should produce something close
    }

    #[test]
    fn test_supports_breaks_continuous() {
        assert!(ScaleType::continuous().supports_breaks());
    }

    #[test]
    fn test_supports_breaks_binned() {
        assert!(ScaleType::binned().supports_breaks());
    }

    #[test]
    fn test_supports_breaks_discrete_false() {
        assert!(!ScaleType::discrete().supports_breaks());
    }

    #[test]
    fn test_supports_breaks_identity_false() {
        assert!(!ScaleType::identity().supports_breaks());
    }
}
