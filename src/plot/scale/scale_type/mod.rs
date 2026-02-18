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
use crate::plot::aesthetic::is_positional_aesthetic;
use crate::plot::{ArrayElement, ColumnInfo, ParameterValue};

// Scale type implementations
mod binned;
mod continuous;
mod discrete;
mod identity;
mod ordinal;

// Re-export scale type structs for direct access if needed
use crate::plot::types::{CastTargetType, SqlTypeNames};
pub use binned::Binned;
pub use continuous::Continuous;
pub use discrete::{infer_transform_from_input_range, Discrete};
pub use identity::Identity;
pub use ordinal::Ordinal;

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
/// NULL values are included at the end of the result.
fn compute_unique_values_multi(columns: &[&Column]) -> Vec<ArrayElement> {
    compute_unique_values_native(columns, true)
}

/// Compute unique sorted values from columns, preserving native types.
///
/// For each column type:
/// - Boolean columns → `ArrayElement::Boolean` values in logical order `[false, true]`
/// - Integer/Float columns → `ArrayElement::Number` values sorted numerically
/// - Date columns → `ArrayElement::Date` values sorted chronologically
/// - DateTime columns → `ArrayElement::DateTime` values sorted chronologically
/// - Time columns → `ArrayElement::Time` values sorted chronologically
/// - String/Categorical columns → `ArrayElement::String` values sorted alphabetically
///
/// If `include_null` is true, `ArrayElement::Null` is appended at the end if any null
/// values exist in the data.
pub fn compute_unique_values_native(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    if columns.is_empty() {
        return Vec::new();
    }

    // Use first column's dtype to determine handling
    let dtype = columns[0].dtype();

    match dtype {
        DataType::Boolean => compute_unique_bool(columns, include_null),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => compute_unique_numeric(columns, include_null),
        DataType::Date => compute_unique_date(columns, include_null),
        DataType::Datetime(_, _) => compute_unique_datetime(columns, include_null),
        DataType::Time => compute_unique_time(columns, include_null),
        _ => compute_unique_string(columns, include_null), // String/Categorical/fallback
    }
}

/// Compute unique boolean values from columns.
fn compute_unique_bool(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    let mut has_false = false;
    let mut has_true = false;
    let mut has_null = false;

    for column in columns {
        if let Ok(ca) = column.as_materialized_series().bool() {
            for val in ca.into_iter() {
                match val {
                    Some(true) => has_true = true,
                    Some(false) => has_false = true,
                    None => has_null = true,
                }
                // Early exit if all values have been encountered
                if has_null && has_true && has_false {
                    break;
                }
            }
        }
        // Early exit if all values have been encountered
        if has_null && has_true && has_false {
            break;
        }
    }

    let mut result = Vec::new();
    if has_false {
        result.push(ArrayElement::Boolean(false));
    }
    if has_true {
        result.push(ArrayElement::Boolean(true));
    }
    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
}

/// Compute unique numeric values from columns, sorted numerically.
fn compute_unique_numeric(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    let mut values: Vec<f64> = Vec::new();
    let mut has_null = false;

    for column in columns {
        if let Ok(series) = column.as_materialized_series().cast(&DataType::Float64) {
            if let Ok(ca) = series.f64() {
                for val in ca.into_iter() {
                    match val {
                        Some(v) if v.is_finite() && !values.contains(&v) => {
                            values.push(v);
                        }
                        None => has_null = true,
                        _ => {} // Skip NaN/Inf or duplicates
                    }
                }
            }
        }
    }

    // Sort numerically
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<ArrayElement> = values.into_iter().map(ArrayElement::Number).collect();

    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
}

/// Compute unique date values from columns, sorted chronologically.
fn compute_unique_date(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    use std::collections::BTreeSet;

    let mut values: BTreeSet<i32> = BTreeSet::new();
    let mut has_null = false;

    for column in columns {
        if let Ok(ca) = column.as_materialized_series().date() {
            // Access the underlying physical Int32 chunked array
            for val in ca.phys.into_iter() {
                match val {
                    Some(days) => {
                        values.insert(days);
                    }
                    None => has_null = true,
                }
            }
        }
    }

    let mut result: Vec<ArrayElement> = values.into_iter().map(ArrayElement::Date).collect();

    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
}

/// Compute unique datetime values from columns, sorted chronologically.
fn compute_unique_datetime(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    use std::collections::BTreeSet;

    let mut values: BTreeSet<i64> = BTreeSet::new();
    let mut has_null = false;

    for column in columns {
        if let Ok(ca) = column.as_materialized_series().datetime() {
            // Access the underlying physical Int64 chunked array
            for val in ca.phys.into_iter() {
                match val {
                    Some(micros) => {
                        values.insert(micros);
                    }
                    None => has_null = true,
                }
            }
        }
    }

    let mut result: Vec<ArrayElement> = values.into_iter().map(ArrayElement::DateTime).collect();

    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
}

/// Compute unique time values from columns, sorted.
fn compute_unique_time(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    use std::collections::BTreeSet;

    let mut values: BTreeSet<i64> = BTreeSet::new();
    let mut has_null = false;

    for column in columns {
        if let Ok(ca) = column.as_materialized_series().time() {
            // Access the underlying physical Int64 chunked array
            for val in ca.phys.into_iter() {
                match val {
                    Some(nanos) => {
                        values.insert(nanos);
                    }
                    None => has_null = true,
                }
            }
        }
    }

    let mut result: Vec<ArrayElement> = values.into_iter().map(ArrayElement::Time).collect();

    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
}

/// Compute unique string values from columns, sorted alphabetically.
fn compute_unique_string(columns: &[&Column], include_null: bool) -> Vec<ArrayElement> {
    use std::collections::BTreeSet;

    let mut values: BTreeSet<String> = BTreeSet::new();
    let mut has_null = false;

    for column in columns {
        let series = column.as_materialized_series();
        if let Ok(unique) = series.unique() {
            for i in 0..unique.len() {
                if let Ok(val) = unique.get(i) {
                    if val.is_null() {
                        has_null = true;
                    } else {
                        let s = val.to_string();
                        // Remove surrounding quotes from string representation
                        let clean = s.trim_matches('"').to_string();
                        values.insert(clean);
                    }
                }
            }
        }
    }

    let mut result: Vec<ArrayElement> = values.into_iter().map(ArrayElement::String).collect();

    if include_null && has_null {
        result.push(ArrayElement::Null);
    }

    result
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
    /// Ordered categorical data with interpolated output
    Ordinal,
    /// Identity scale (use inferred type)
    Identity,
}

impl std::fmt::Display for ScaleTypeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScaleTypeKind::Continuous => "continuous",
            ScaleTypeKind::Discrete => "discrete",
            ScaleTypeKind::Binned => "binned",
            ScaleTypeKind::Ordinal => "ordinal",
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

    /// Returns whether this scale type uses discrete input range (unique values).
    ///
    /// When `true`, input range is computed as unique sorted values from data.
    /// When `false`, input range is computed as [min, max] from data.
    ///
    /// Defaults to `false` (continuous min/max range).
    /// Overridden to return `true` for Discrete, Identity, and Ordinal.
    fn uses_discrete_input_range(&self) -> bool {
        false
    }

    /// Get default output range for an aesthetic.
    ///
    /// Returns sensible default ranges based on the aesthetic type and scale type.
    /// For example:
    /// - color/fill + discrete → standard categorical color palette (sized to input_range length)
    /// - size + continuous → [min_size, max_size] range
    /// - opacity + continuous → [0.2, 1.0] range
    ///
    /// The scale reference is provided so implementations can access:
    /// - `scale.input_range` for sizing discrete palettes
    /// - `scale.properties["breaks"]` for binned scales to determine bin count
    ///
    /// Returns Ok(None) if no default is appropriate (e.g., x/y position aesthetics).
    /// Returns Err if the palette doesn't have enough colors for the input range.
    fn default_output_range(
        &self,
        _aesthetic: &str,
        _scale: &super::Scale,
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
    /// 2. Identity (default for all aesthetics including size)
    ///
    /// The column_dtype parameter enables automatic temporal transform inference when
    /// a Date, DateTime, or Time column is mapped to an aesthetic.
    fn default_transform(
        &self,
        _aesthetic: &str,
        column_dtype: Option<&DataType>,
    ) -> TransformKind {
        // First check column data type for temporal transforms
        if let Some(dtype) = column_dtype {
            match dtype {
                DataType::Date => return TransformKind::Date,
                DataType::Datetime(_, _) => return TransformKind::DateTime,
                DataType::Time => return TransformKind::Time,
                _ => {}
            }
        }

        // Default to identity (linear) for all aesthetics
        TransformKind::Identity
    }

    /// Resolve and validate the transform.
    ///
    /// If user_transform is Some, validates it's in allowed_transforms().
    /// If user_transform is None, infers the transform in priority order:
    /// 1. Input range type (FROM clause) - if provided
    /// 2. Column data type - if available
    /// 3. Identity (fallback for all aesthetics)
    fn resolve_transform(
        &self,
        aesthetic: &str,
        user_transform: Option<&Transform>,
        column_dtype: Option<&DataType>,
        _input_range: Option<&[ArrayElement]>,
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

            // Discrete and Ordinal scales only support "censor" - no way to map unmapped values to output
            let kind = self.scale_type_kind();
            if (kind == ScaleTypeKind::Discrete || kind == ScaleTypeKind::Ordinal)
                && oob != OOB_CENSOR
            {
                return Err(format!(
                    "{} scale only supports oob='censor'. Cannot use '{}' because \
                     values outside the input range have no corresponding output value.",
                    self.name(),
                    oob
                ));
            }

            // Binned scales support "censor" and "squish", but not "keep"
            // Values outside bins have no bin to map to, but can be squished to nearest bin edge
            if kind == ScaleTypeKind::Binned && oob == OOB_KEEP {
                return Err(format!(
                    "{} scale does not support oob='keep'. Use 'censor' to exclude values \
                     outside bins, or 'squish' to clamp them to the nearest bin edge.",
                    self.name()
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

        // Extract min/max from input range using to_f64() for temporal support
        let (min, max) = match input_range {
            Some(range) if range.len() >= 2 => {
                let min = range[0].to_f64()?;
                let max = range[range.len() - 1].to_f64()?;
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
        let breaks: Vec<ArrayElement> = match transform {
            Some(t) if !t.is_identity() => {
                let raw_breaks = t.calculate_breaks(min, max, count, pretty);
                // Wrap breaks in the appropriate ArrayElement type using transform
                raw_breaks.into_iter().map(|v| t.wrap_numeric(v)).collect()
            }
            _ => {
                // Identity transform or no transform - use default pretty/linear breaks
                let raw_breaks = if pretty {
                    super::breaks::pretty_breaks(min, max, count)
                } else {
                    super::breaks::linear_breaks(min, max, count)
                };
                raw_breaks.into_iter().map(ArrayElement::Number).collect()
            }
        };

        if breaks.is_empty() {
            None
        } else {
            Some(breaks)
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
    /// 1. Resolves properties (fills in defaults, validates)
    /// 2. Resolves transform from context dtype if not set
    /// 3. Resolves input_range from context (or merges with existing partial range)
    /// 4. Converts input_range values using transform (e.g., ISO strings → Date/DateTime/Time)
    /// 5. If breaks is a scalar Number, calculates break positions and stores as Array
    /// 6. Applies label template
    /// 7. Resolves output range
    ///
    /// Note: Binned scale overrides this method to add Binned-specific logic
    /// (implicit break handling, break/range alignment, terminal label suppression).
    fn resolve(
        &self,
        scale: &mut super::Scale,
        context: &ScaleDataContext,
        aesthetic: &str,
    ) -> Result<(), String> {
        // Steps 1-4: Common resolution logic (properties, transform, input_range, convert values)
        let common_result = resolve_common_steps(self, scale, context, aesthetic)?;
        let resolved_transform = common_result.transform;

        // 5. Calculate breaks if supports_breaks()
        // If breaks is a scalar Number (count), calculate actual break positions and store as Array
        // If breaks is already an Array, user provided explicit breaks - convert using transform
        // Then filter breaks to the input range (break algorithms may produce "nice" values outside range)
        if self.supports_breaks() {
            match scale.properties.get("breaks") {
                Some(ParameterValue::Number(_)) => {
                    // Scalar count → calculate actual breaks and store as Array
                    if let Some(breaks) = self.resolve_breaks(
                        scale.input_range.as_deref(),
                        &scale.properties,
                        scale.transform.as_ref(),
                    ) {
                        // Filter to input range
                        let filtered = if let Some(ref range) = scale.input_range {
                            super::super::breaks::filter_breaks_to_range(&breaks, range)
                        } else {
                            breaks
                        };
                        scale
                            .properties
                            .insert("breaks".to_string(), ParameterValue::Array(filtered));
                    }
                }
                Some(ParameterValue::Array(explicit_breaks)) => {
                    // User provided explicit breaks - convert using transform
                    let converted: Vec<ArrayElement> = explicit_breaks
                        .iter()
                        .map(|elem| resolved_transform.parse_value(elem))
                        .collect();
                    // Filter breaks to input range
                    let filtered = if let Some(ref range) = scale.input_range {
                        super::super::breaks::filter_breaks_to_range(&converted, range)
                    } else {
                        converted
                    };
                    scale
                        .properties
                        .insert("breaks".to_string(), ParameterValue::Array(filtered));
                }
                Some(ParameterValue::String(interval_str)) => {
                    // Temporal interval string like "2 months", "week"
                    // Only valid for temporal transforms (Date, DateTime, Time)
                    use super::super::breaks::{
                        temporal_breaks_date, temporal_breaks_datetime, temporal_breaks_time,
                        TemporalInterval,
                    };

                    if let Some(interval) = TemporalInterval::create_from_str(interval_str) {
                        if let Some(ref range) = scale.input_range {
                            let breaks: Vec<ArrayElement> = match resolved_transform
                                .transform_kind()
                            {
                                TransformKind::Date => {
                                    let min = range[0].to_f64().unwrap_or(0.0) as i32;
                                    let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i32;
                                    temporal_breaks_date(min, max, interval)
                                        .into_iter()
                                        .map(ArrayElement::String)
                                        .collect()
                                }
                                TransformKind::DateTime => {
                                    let min = range[0].to_f64().unwrap_or(0.0) as i64;
                                    let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i64;
                                    temporal_breaks_datetime(min, max, interval)
                                        .into_iter()
                                        .map(ArrayElement::String)
                                        .collect()
                                }
                                TransformKind::Time => {
                                    let min = range[0].to_f64().unwrap_or(0.0) as i64;
                                    let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i64;
                                    temporal_breaks_time(min, max, interval)
                                        .into_iter()
                                        .map(ArrayElement::String)
                                        .collect()
                                }
                                _ => vec![], // Non-temporal transforms don't support interval strings
                            };

                            if !breaks.is_empty() {
                                // Convert string breaks to appropriate temporal ArrayElement types
                                let converted: Vec<ArrayElement> = breaks
                                    .iter()
                                    .map(|elem| resolved_transform.parse_value(elem))
                                    .collect();
                                // Filter to input range
                                let filtered =
                                    super::super::breaks::filter_breaks_to_range(&converted, range);
                                scale
                                    .properties
                                    .insert("breaks".to_string(), ParameterValue::Array(filtered));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // 6. Apply label template (RENAMING * => '...')
        // Default is '{}' to ensure we control formatting instead of Vega-Lite
        // For continuous scales, apply to breaks array
        // For discrete scales, apply to input_range (domain values)
        let template = &scale.label_template;

        let values_to_label = if self.supports_breaks() {
            // Continuous: use breaks
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

        // 7. Resolve output range (TO clause)
        self.resolve_output_range(scale, aesthetic)?;

        // Mark scale as resolved
        scale.resolved = true;

        Ok(())
    }

    /// Resolve output range (TO clause) for a scale.
    ///
    /// 1. If no output_range is set, fills from `default_output_range()` (full palette)
    /// 2. Converts Palette variants to Array (expand named palette to colors)
    /// 3. Sizes the output_range based on scale type:
    ///    - Continuous: Keeps as-is (full palette for Vega-Lite interpolation)
    ///    - Discrete: Truncates to match `input_range.len()` (category count)
    ///    - Binned: Truncates/interpolates to match `breaks.len() - 1` (bin count)
    ///
    /// # Default Implementation
    ///
    /// The default implementation handles continuous scales: it converts Palette
    /// to Array (so Vega-Lite gets actual colors to interpolate), and fills from
    /// `default_output_range()` if not set. Does not size the array.
    ///
    /// Discrete and Binned scales override this to size the output appropriately.
    fn resolve_output_range(
        &self,
        scale: &mut super::Scale,
        aesthetic: &str,
    ) -> Result<(), String> {
        use super::{palettes, OutputRange};

        // Phase 1: Ensure we have an Array (convert Palette or fill default)
        match &scale.output_range {
            None => {
                // No output range - fill from default
                if let Some(default_range) = self.default_output_range(aesthetic, scale)? {
                    scale.output_range = Some(OutputRange::Array(default_range));
                }
            }
            Some(OutputRange::Palette(name)) => {
                // Named palette - convert to Array (full palette for interpolation)
                let arr = palettes::lookup_palette(aesthetic, name)?;
                scale.output_range = Some(OutputRange::Array(arr));
            }
            Some(OutputRange::Array(_)) => {
                // Already an array, nothing to do
            }
        }

        // Continuous scales: keep output_range as-is (no sizing needed)
        // Vega-Lite will interpolate across the full palette
        Ok(())
    }

    /// Validate that this scale type supports the given data type.
    ///
    /// Called when a user explicitly specifies a scale type (e.g., `SCALE DISCRETE x`)
    /// to validate that the scale type is compatible with the data being mapped.
    ///
    /// Returns Ok(()) if compatible, Err with a descriptive message if not.
    /// The error message should be actionable and suggest alternative scale types.
    ///
    /// Default implementation accepts all types (identity scales, etc.).
    /// Continuous/Binned scales override to reject non-numeric types.
    /// Discrete/Ordinal scales override to reject numeric types.
    fn validate_dtype(&self, _dtype: &DataType) -> Result<(), String> {
        Ok(()) // Default: accept all types
    }

    /// Pre-stat SQL transformation hook.
    ///
    /// Called inside build_layer_query to generate SQL that transforms data
    /// BEFORE stat transforms run. Returns SQL expression to wrap the column.
    ///
    /// Only Binned scales implement this (returns binning SQL).
    /// Default returns None (no transformation).
    ///
    /// # Arguments
    ///
    /// * `column_name` - The column to transform
    /// * `column_dtype` - The column's data type from the schema
    /// * `scale` - The resolved scale specification
    /// * `type_names` - SQL type names for casting (from Reader)
    fn pre_stat_transform_sql(
        &self,
        _column_name: &str,
        _column_dtype: &DataType,
        _scale: &super::Scale,
        _type_names: &SqlTypeNames,
    ) -> Option<String> {
        None
    }

    /// Determine if a column needs casting to match the scale's target type.
    ///
    /// This is called early in the execution pipeline to determine what columns
    /// need SQL-level casting before min/max extraction and scale resolution.
    ///
    /// # Arguments
    ///
    /// * `column_dtype` - The column's current data type
    /// * `target_dtype` - The target data type determined by type coercion across layers
    ///
    /// # Returns
    ///
    /// Returns Some(CastTargetType) if the column needs casting, None otherwise.
    ///
    /// Default implementation uses the `needs_cast` helper function.
    fn required_cast_type(
        &self,
        column_dtype: &DataType,
        target_dtype: &DataType,
    ) -> Option<CastTargetType> {
        needs_cast(column_dtype, target_dtype)
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

    /// Create an Ordinal scale type
    pub fn ordinal() -> Self {
        Self(Arc::new(Ordinal))
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
            ScaleTypeKind::Ordinal => Self::ordinal(),
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

    /// Check if this scale type uses discrete input range (unique values vs min/max)
    pub fn uses_discrete_input_range(&self) -> bool {
        self.0.uses_discrete_input_range()
    }

    /// Get default output range for an aesthetic.
    ///
    /// Delegates to the underlying scale type implementation.
    pub fn default_output_range(
        &self,
        aesthetic: &str,
        scale: &super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        self.0.default_output_range(aesthetic, scale)
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
    /// If user_transform is Some, validates it's in allowed_transforms().
    /// If user_transform is None, infers the transform in priority order:
    /// 1. Input range type (FROM clause) - if provided
    /// 2. Column data type - if available
    /// 3. Identity (fallback for all aesthetics)
    pub fn resolve_transform(
        &self,
        aesthetic: &str,
        user_transform: Option<&Transform>,
        column_dtype: Option<&DataType>,
        input_range: Option<&[ArrayElement]>,
    ) -> Result<Transform, String> {
        self.0
            .resolve_transform(aesthetic, user_transform, column_dtype, input_range)
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
    ///
    /// # Arguments
    ///
    /// * `column_name` - The column to transform
    /// * `column_dtype` - The column's data type from the schema
    /// * `scale` - The resolved scale specification
    /// * `type_names` - SQL type names for casting (from Reader)
    pub fn pre_stat_transform_sql(
        &self,
        column_name: &str,
        column_dtype: &DataType,
        scale: &super::Scale,
        type_names: &SqlTypeNames,
    ) -> Option<String> {
        self.0
            .pre_stat_transform_sql(column_name, column_dtype, scale, type_names)
    }

    /// Determine if a column needs casting to match the scale's target type.
    ///
    /// Returns Some(CastTargetType) if casting is needed, None otherwise.
    pub fn required_cast_type(
        &self,
        column_dtype: &DataType,
        target_dtype: &DataType,
    ) -> Option<CastTargetType> {
        self.0.required_cast_type(column_dtype, target_dtype)
    }

    /// Resolve output range (TO clause) for a scale.
    ///
    /// Fills from `default_output_range()` if not set, then sizes based on scale type.
    pub fn resolve_output_range(
        &self,
        scale: &mut super::Scale,
        aesthetic: &str,
    ) -> Result<(), String> {
        self.0.resolve_output_range(scale, aesthetic)
    }

    /// Validate that this scale type supports the given data type.
    ///
    /// Returns Ok(()) if compatible, Err with a descriptive message if not.
    pub fn validate_dtype(&self, dtype: &DataType) -> Result<(), String> {
        self.0.validate_dtype(dtype)
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

/// Check if input range contains any Null placeholders
pub(crate) fn input_range_has_nulls(range: &[ArrayElement]) -> bool {
    range.iter().any(|e| matches!(e, ArrayElement::Null))
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
pub fn default_oob(aesthetic: &str) -> &'static str {
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

// =============================================================================
// Output range resolution helpers
// =============================================================================

/// Interpolate numeric values to a target count.
///
/// Takes min/max from the first and last values in the array,
/// then generates `count` evenly-spaced values.
///
/// Returns `None` if the input has fewer than 2 values, count is 0,
/// or values are not numeric.
///
/// # Example
///
/// ```ignore
/// let range = vec![ArrayElement::Number(1.0), ArrayElement::Number(6.0)];
/// let interpolated = interpolate_numeric(&range, 5);
/// // Returns Some([1.0, 2.25, 3.5, 4.75, 6.0])
/// ```
pub(crate) fn interpolate_numeric(
    values: &[ArrayElement],
    count: usize,
) -> Option<Vec<ArrayElement>> {
    if values.len() < 2 || count == 0 {
        return None;
    }

    let nums: Vec<f64> = values.iter().filter_map(|e| e.to_f64()).collect();
    if nums.len() < 2 {
        return None;
    }

    let min_val = nums[0];
    let max_val = nums[nums.len() - 1];

    Some(
        (0..count)
            .map(|i| {
                let t = if count > 1 {
                    i as f64 / (count - 1) as f64
                } else {
                    0.5
                };
                ArrayElement::Number(min_val + t * (max_val - min_val))
            })
            .collect(),
    )
}

/// Size/interpolate output range to match a target count.
///
/// This is used by ordinal and binned scales to ensure the output range
/// has exactly the right number of values for the categories or bins.
///
/// Behavior by aesthetic type:
/// - **fill/stroke**: Interpolates colors using Oklab color space
/// - **size/linewidth/opacity**: Interpolates numeric values linearly
/// - **shape/linetype/other**: Truncates if too many values; errors if too few
///
/// # Arguments
///
/// * `scale` - The scale being resolved (output_range will be modified)
/// * `aesthetic` - The aesthetic type
/// * `count` - Target number of output values
///
/// # Errors
///
/// Returns an error if the output range has insufficient values for
/// non-interpolatable aesthetics (shape, linetype).
pub(crate) fn size_output_range(
    scale: &mut super::Scale,
    aesthetic: &str,
    count: usize,
) -> Result<(), String> {
    use super::colour::{interpolate_colors, ColorSpace};
    use super::OutputRange;

    if count == 0 {
        return Ok(());
    }

    if let Some(OutputRange::Array(ref arr)) = scale.output_range.clone() {
        if matches!(aesthetic, "fill" | "stroke") && arr.len() >= 2 {
            // Color interpolation using Oklab
            let hex_strs: Vec<&str> = arr
                .iter()
                .filter_map(|e| match e {
                    ArrayElement::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            let interpolated = interpolate_colors(&hex_strs, count, ColorSpace::Oklab)?;
            scale.output_range = Some(OutputRange::Array(
                interpolated.into_iter().map(ArrayElement::String).collect(),
            ));
        } else if matches!(aesthetic, "size" | "linewidth" | "opacity") && arr.len() >= 2 {
            // Numeric interpolation
            if let Some(interpolated) = interpolate_numeric(arr, count) {
                scale.output_range = Some(OutputRange::Array(interpolated));
            }
        } else {
            // Non-interpolatable aesthetics (shape, linetype): truncate/error
            if arr.len() < count {
                return Err(format!(
                    "Output range has {} values but {} {} needed",
                    arr.len(),
                    count,
                    if count == 1 { "is" } else { "are" }
                ));
            }
            if arr.len() > count {
                scale.output_range = Some(OutputRange::Array(
                    arr.iter().take(count).cloned().collect(),
                ));
            }
        }
    }

    Ok(())
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
pub(crate) fn expand_numeric_range(
    range: &[ArrayElement],
    mult: f64,
    add: f64,
) -> Vec<ArrayElement> {
    expand_numeric_range_selective(range, mult, add, None)
}

/// Apply expansion selectively to a numeric [min, max] range.
///
/// If `original_user_range` is provided, only expand values that were originally Null
/// in the user range. This preserves explicit user limits while expanding inferred values.
///
/// For example, with `FROM [0, null]`:
/// - min=0 is explicit, so it's preserved as 0
/// - max was null (inferred from data), so it gets expanded
///
/// Formula for expanded values: [min - range*mult - add, max + range*mult + add]
pub(crate) fn expand_numeric_range_selective(
    range: &[ArrayElement],
    mult: f64,
    add: f64,
    original_user_range: Option<&[ArrayElement]>,
) -> Vec<ArrayElement> {
    if range.len() < 2 {
        return range.to_vec();
    }

    // Use to_f64() to handle Number, Date, DateTime, and Time variants
    let min = match range[0].to_f64() {
        Some(n) => n,
        None => return range.to_vec(),
    };
    let max = match range[1].to_f64() {
        Some(n) => n,
        None => return range.to_vec(),
    };

    let span = max - min;

    // For singular ranges (min == max), use the absolute value to compute expansion
    // This prevents zero expansion when all data values are identical
    // If the value itself is 0, use a small default expansion (1.0)
    let effective_span = if span.abs() < 1e-10 {
        if min.abs() < 1e-10 {
            1.0 // If the value is 0, expand by ±1
        } else {
            min.abs() // Use the absolute value for expansion
        }
    } else {
        span
    };
    let expansion = effective_span * mult + add;

    // Check if min was explicitly set by user (non-null in original range)
    let min_is_explicit = original_user_range
        .and_then(|ur| ur.first())
        .map(|e| !matches!(e, ArrayElement::Null))
        .unwrap_or(false);

    // Check if max was explicitly set by user (non-null in original range)
    let max_is_explicit = original_user_range
        .and_then(|ur| ur.get(1))
        .map(|e| !matches!(e, ArrayElement::Null))
        .unwrap_or(false);

    // Only expand values that were inferred (originally null)
    let expanded_min = if min_is_explicit {
        min
    } else {
        min - expansion
    };
    let expanded_max = if max_is_explicit {
        max
    } else {
        max + expansion
    };

    vec![
        ArrayElement::Number(expanded_min),
        ArrayElement::Number(expanded_max),
    ]
}

/// Get expand factors from properties, using defaults for continuous/temporal scales.
pub(crate) fn get_expand_factors(properties: &HashMap<String, ParameterValue>) -> (f64, f64) {
    properties
        .get("expand")
        .and_then(parse_expand_value)
        .unwrap_or((DEFAULT_EXPAND_MULT, DEFAULT_EXPAND_ADD))
}

/// Clip an input range to a transform's valid domain.
///
/// This prevents expansion from producing invalid values for transforms
/// with restricted domains (e.g., log scales which exclude 0 and negatives).
pub(crate) fn clip_to_transform_domain(
    range: &[ArrayElement],
    transform: &Transform,
) -> Vec<ArrayElement> {
    if range.len() < 2 {
        return range.to_vec();
    }

    let (domain_min, domain_max) = transform.allowed_domain();
    let mut result = range.to_vec();

    if let Some(min) = result[0].to_f64() {
        if min < domain_min {
            result[0] = ArrayElement::Number(domain_min);
        }
    }

    if let Some(max) = result[1].to_f64() {
        if max > domain_max {
            result[1] = ArrayElement::Number(domain_max);
        }
    }

    result
}

// =============================================================================
// Common Scale Resolution Logic
// =============================================================================

/// Result from the common scale resolution steps.
///
/// Contains values needed by both the default resolve() implementation
/// and any scale type overrides (like Binned).
#[derive(Debug)]
pub(crate) struct ResolveCommonResult {
    /// Resolved transform
    pub transform: Transform,
    /// Expansion factors (mult, add)
    pub expand_factors: (f64, f64),
}

/// Perform the common scale resolution steps (1-4).
///
/// This handles:
/// 1. Resolve properties (fills in defaults, validates)
/// 2. Resolve transform from user input, input range (FROM clause), and context dtype
/// 3. Resolve input range (merge user range with context, apply expansion, clip to domain)
/// 4. Convert input_range values using transform
///
/// Returns the resolved transform and expand factors for use by callers.
pub(crate) fn resolve_common_steps<T: ScaleTypeTrait + ?Sized>(
    scale_type: &T,
    scale: &mut super::Scale,
    context: &ScaleDataContext,
    aesthetic: &str,
) -> Result<ResolveCommonResult, String> {
    // 1. Resolve properties (fills in defaults, validates)
    scale.properties = scale_type.resolve_properties(aesthetic, &scale.properties)?;

    // 1b. Validate input range length for continuous/binned scales
    // These scales require exactly 2 values [min, max] when explicitly specified
    if scale.explicit_input_range {
        if let Some(ref range) = scale.input_range {
            let kind = scale_type.scale_type_kind();
            if (kind == ScaleTypeKind::Continuous || kind == ScaleTypeKind::Binned)
                && range.len() != 2
            {
                return Err(format!(
                    "{} scale input range (FROM clause) must have exactly 2 values [min, max], got {}",
                    scale_type.name(),
                    range.len()
                ));
            }
        }
    }

    // 2. Resolve transform from user input, input range (FROM clause), and context dtype
    // Priority: user transform > input range inference > column dtype inference > aesthetic default
    let resolved_transform = scale_type.resolve_transform(
        aesthetic,
        scale.transform.as_ref(),
        context.dtype.as_ref(),
        scale.input_range.as_deref(),
    )?;
    scale.transform = Some(resolved_transform.clone());

    // 3. Resolve input range
    // Strategy: First merge user range with context (filling nulls), then apply expansion
    // This ensures expansion is calculated on the final range span.
    // IMPORTANT: Only expand values that were inferred (originally null), not explicit user values.
    // For example, `FROM [0, null]` should keep min=0 and only expand max.
    let (mult, add) = get_expand_factors(&scale.properties);

    // Track the original user range to know which values are explicit vs inferred
    let original_user_range = scale.input_range.clone();

    // Step 1: Determine the base range (before expansion)
    // Also track whether this is a discrete range (unique values) vs continuous (min/max)
    let (base_range, is_discrete_range): (Option<Vec<ArrayElement>>, bool) =
        if let Some(ref user_range) = scale.input_range {
            if input_range_has_nulls(user_range) {
                // User provided partial range with Nulls - merge with context (not expanded yet)
                if let Some(ref range) = context.range {
                    let (context_values, is_discrete) = match range {
                        InputRange::Continuous(r) => (r.clone(), false),
                        InputRange::Discrete(r) => (r.clone(), true),
                    };
                    (
                        Some(merge_with_context(user_range, &context_values)),
                        is_discrete,
                    )
                } else {
                    // No context range, keep user range as-is (Nulls will remain)
                    (Some(user_range.clone()), false)
                }
            } else {
                // User provided complete range - use as-is for now
                // Treat as continuous since user explicitly provided it
                (Some(user_range.clone()), false)
            }
        } else {
            match &context.range {
                Some(InputRange::Continuous(r)) => (Some(r.clone()), false),
                Some(InputRange::Discrete(r)) => (Some(r.clone()), true),
                None => (None, false),
            }
        };

    // Step 2: Apply expansion to the final merged range
    // Expansion should ONLY happen when ALL conditions are met:
    // 1. Scale uses continuous input range (not discrete/ordinal scales)
    // 2. Aesthetic is positional (x, y, xmin, xmax, etc.)
    // 3. Input range was at least partially deduced (not fully explicit)
    //
    // Then clip to the transform's valid domain to prevent invalid values
    // (e.g., expansion producing negative values for log scales)
    if let Some(range) = base_range {
        let is_positional = is_positional_aesthetic(aesthetic);
        let is_deduced = !scale.explicit_input_range
            || input_range_has_nulls(original_user_range.as_deref().unwrap_or(&[]));

        if !is_discrete_range && is_positional && is_deduced {
            let expanded =
                expand_numeric_range_selective(&range, mult, add, original_user_range.as_deref());
            scale.input_range = Some(clip_to_transform_domain(&expanded, &resolved_transform));
        } else {
            // No expansion for discrete scales, non-positional aesthetics, or fully explicit ranges
            scale.input_range = Some(range);
        }
    }

    // 4. Convert input_range values using transform (e.g., ISO strings → Date/DateTime/Time)
    // This ensures temporal scales properly parse user-provided date strings
    if let Some(ref input_range) = scale.input_range {
        let converted: Vec<ArrayElement> = input_range
            .iter()
            .map(|elem| resolved_transform.parse_value(elem))
            .collect();
        scale.input_range = Some(converted);
    }

    Ok(ResolveCommonResult {
        transform: resolved_transform,
        expand_factors: (mult, add),
    })
}

// =============================================================================
// Type Coercion (vctrs-style hierarchy)
// =============================================================================

/// Type family for coercion purposes.
///
/// Types within the same family can be coerced to a common type.
/// Types in different families coerce to String (the most general type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeFamily {
    /// Boolean, Integer, Double - upcast to more general
    Numeric,
    /// Date, Datetime, Time - no auto-coercion between them
    Temporal,
    /// String - most general type
    String,
}

/// Determine the type family for a Polars DataType.
fn type_family(dtype: &DataType) -> TypeFamily {
    match dtype {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => TypeFamily::Numeric,
        DataType::Date | DataType::Datetime(_, _) | DataType::Time => TypeFamily::Temporal,
        DataType::String => TypeFamily::String,
        _ => TypeFamily::String, // Unknown types treated as String
    }
}

/// Numeric type rank for coercion (higher = more general).
fn numeric_rank(dtype: &DataType) -> u8 {
    match dtype {
        DataType::Boolean => 0,
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 => 3,
        DataType::Int64 | DataType::UInt64 => 4,
        DataType::Float32 => 5,
        DataType::Float64 => 6,
        _ => 0,
    }
}

/// Coerce multiple Polars DataTypes to a common type following vctrs-style hierarchy.
///
/// # Type Families
///
/// 1. **Numeric family:** Boolean → Int8 → ... → Int64 → Float32 → Float64
/// 2. **Temporal family:** Date, Datetime, Time (no auto-coercion between them)
/// 3. **String family:** String (most general, can represent anything)
///
/// # Coercion Rules
///
/// - **Within numeric family:** Upcast to more general type (Boolean → Int64 → Float64)
/// - **Within temporal family:** Error if mixing different temporal types
/// - **Numeric + Temporal:** Coerce to String (incompatible families)
/// - **Any + String:** Result is String (discrete scale)
///
/// # Returns
///
/// Returns Ok(DataType) with the common type, or Err if incompatible temporal types.
pub fn coerce_dtypes(dtypes: &[DataType]) -> Result<DataType, String> {
    if dtypes.is_empty() {
        return Ok(DataType::String); // Default to String for empty
    }

    if dtypes.len() == 1 {
        return Ok(dtypes[0].clone());
    }

    // Determine families present
    let families: Vec<TypeFamily> = dtypes.iter().map(type_family).collect();

    // Check if any type is String - result is String
    if families.contains(&TypeFamily::String) {
        return Ok(DataType::String);
    }

    // Check for mixed families
    let has_numeric = families.contains(&TypeFamily::Numeric);
    let has_temporal = families.contains(&TypeFamily::Temporal);

    if has_numeric && has_temporal {
        // Incompatible families - coerce to String
        return Ok(DataType::String);
    }

    // All numeric - find highest rank
    if has_numeric && !has_temporal {
        let max_rank = dtypes.iter().map(numeric_rank).max().unwrap_or(0);
        return Ok(match max_rank {
            0 => DataType::Boolean,
            1 => DataType::Int8,
            2 => DataType::Int16,
            3 => DataType::Int32,
            4 => DataType::Int64,
            5 => DataType::Float32,
            _ => DataType::Float64,
        });
    }

    // All temporal - check they're all the same type
    if has_temporal && !has_numeric {
        let first = &dtypes[0];
        let all_same = dtypes.iter().all(|d| {
            matches!(
                (first, d),
                (DataType::Date, DataType::Date)
                    | (DataType::Datetime(_, _), DataType::Datetime(_, _))
                    | (DataType::Time, DataType::Time)
            )
        });

        if all_same {
            return Ok(first.clone());
        } else {
            // Mixed temporal types - error (requires explicit transform)
            return Err(
                "Cannot mix different temporal types (Date, Datetime, Time) without explicit transform. \
                Use VIA date, VIA datetime, or VIA time to specify the target type.".to_string()
            );
        }
    }

    // Fallback to String
    Ok(DataType::String)
}

/// Convert a Polars DataType to the corresponding CastTargetType.
///
/// Returns None if no casting is needed (identity).
pub fn dtype_to_cast_target(dtype: &DataType) -> CastTargetType {
    match dtype {
        DataType::Boolean => CastTargetType::Boolean,
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => CastTargetType::Number,
        DataType::Date => CastTargetType::Date,
        DataType::Datetime(_, _) => CastTargetType::DateTime,
        DataType::Time => CastTargetType::Time,
        DataType::String => CastTargetType::String,
        _ => CastTargetType::String, // Unknown types treated as String
    }
}

/// Check if a column dtype needs casting to match a target dtype.
///
/// Returns Some(CastTargetType) if casting is needed, None otherwise.
pub fn needs_cast(column_dtype: &DataType, target_dtype: &DataType) -> Option<CastTargetType> {
    // Same type family check
    let column_family = type_family(column_dtype);
    let target_family = type_family(target_dtype);

    // Check if already the target type
    let is_already_target = match (column_dtype, target_dtype) {
        (DataType::Boolean, DataType::Boolean) => true,
        (DataType::Date, DataType::Date) => true,
        (DataType::Datetime(_, _), DataType::Datetime(_, _)) => true,
        (DataType::Time, DataType::Time) => true,
        (DataType::String, DataType::String) => true,
        // For numeric, check if target is Float64 and column is any numeric
        (
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64,
            DataType::Float64,
        ) => {
            // Numeric columns don't need SQL-level casting to Float64
            // DuckDB handles implicit numeric conversions
            true
        }
        _ => false,
    };

    if is_already_target {
        return None;
    }

    // If families differ, need to cast
    if column_family != target_family {
        return Some(dtype_to_cast_target(target_dtype));
    }

    // Within same family, check specific cases
    match target_family {
        TypeFamily::Numeric => {
            // Numeric to numeric - DuckDB handles implicitly
            None
        }
        TypeFamily::Temporal => {
            // Different temporal types - need explicit cast
            Some(dtype_to_cast_target(target_dtype))
        }
        TypeFamily::String => None, // Already string
    }
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
    fn test_scale_type_uses_discrete_input_range() {
        // Continuous and Binned use min/max range (return false)
        assert!(!ScaleType::continuous().uses_discrete_input_range());
        assert!(!ScaleType::binned().uses_discrete_input_range());

        // Discrete, Identity, and Ordinal use unique values (return true)
        assert!(ScaleType::discrete().uses_discrete_input_range());
        assert!(ScaleType::identity().uses_discrete_input_range());
        assert!(ScaleType::ordinal().uses_discrete_input_range());
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
    fn test_expand_selective_min_explicit() {
        // User says FROM [0, null] → min is explicit, max is inferred
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let user_range = vec![ArrayElement::Number(0.0), ArrayElement::Null];

        let expanded = expand_numeric_range_selective(&range, 0.05, 0.0, Some(&user_range));

        // Min should stay at 0 (explicit), max should be expanded
        // span = 100, expansion = 5
        assert_eq!(expanded[0], ArrayElement::Number(0.0)); // NOT -5.0
        assert_eq!(expanded[1], ArrayElement::Number(105.0)); // expanded
    }

    #[test]
    fn test_expand_selective_max_explicit() {
        // User says FROM [null, 100] → min is inferred, max is explicit
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let user_range = vec![ArrayElement::Null, ArrayElement::Number(100.0)];

        let expanded = expand_numeric_range_selective(&range, 0.05, 0.0, Some(&user_range));

        // Min should be expanded, max should stay at 100 (explicit)
        // span = 100, expansion = 5
        assert_eq!(expanded[0], ArrayElement::Number(-5.0)); // expanded
        assert_eq!(expanded[1], ArrayElement::Number(100.0)); // NOT 105.0
    }

    #[test]
    fn test_expand_selective_both_explicit() {
        // User says FROM [0, 100] → both are explicit
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let user_range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];

        let expanded = expand_numeric_range_selective(&range, 0.05, 0.0, Some(&user_range));

        // Both should stay as-is (no expansion)
        assert_eq!(expanded[0], ArrayElement::Number(0.0));
        assert_eq!(expanded[1], ArrayElement::Number(100.0));
    }

    #[test]
    fn test_expand_selective_no_user_range() {
        // No user range (all inferred) → expand both
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];

        let expanded = expand_numeric_range_selective(&range, 0.05, 0.0, None);

        // Both should be expanded
        assert_eq!(expanded[0], ArrayElement::Number(-5.0));
        assert_eq!(expanded[1], ArrayElement::Number(105.0));
    }

    #[test]
    fn test_expand_singular_range_nonzero() {
        // Singular range: all values are the same (e.g., count=12 for all bars)
        // Should expand based on the value itself to create visible range
        let range = vec![ArrayElement::Number(12.0), ArrayElement::Number(12.0)];
        let expanded = expand_numeric_range(&range, 0.05, 0.0);

        // span = 0, effective_span = |12| = 12, expansion = 12 * 0.05 = 0.6
        // expanded = [12 - 0.6, 12 + 0.6] = [11.4, 12.6]
        assert_eq!(expanded[0], ArrayElement::Number(11.4));
        assert_eq!(expanded[1], ArrayElement::Number(12.6));
    }

    #[test]
    fn test_expand_singular_range_zero() {
        // Singular range at zero (e.g., all counts are 0)
        // Should use default expansion of 1.0
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(0.0)];
        let expanded = expand_numeric_range(&range, 0.05, 0.0);

        // span = 0, value = 0, effective_span = 1.0, expansion = 1.0 * 0.05 = 0.05
        // expanded = [0 - 0.05, 0 + 0.05] = [-0.05, 0.05]
        assert_eq!(expanded[0], ArrayElement::Number(-0.05));
        assert_eq!(expanded[1], ArrayElement::Number(0.05));
    }

    // =========================================================================
    // resolve_properties Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_resolve_properties_rejection_cases() {
        // Unknown property rejected
        let mut props = HashMap::new();
        props.insert("unknown".to_string(), ParameterValue::Number(1.0));
        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown"));

        // Discrete rejects expand
        let mut expand_props = HashMap::new();
        expand_props.insert("expand".to_string(), ParameterValue::Number(0.1));
        let result = ScaleType::discrete().resolve_properties("color", &expand_props);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support SETTING 'expand'"));

        // Identity rejects any property
        let result = ScaleType::identity().resolve_properties("x", &expand_props);
        assert!(result.is_err());

        // Binned rejects oob='keep'
        let mut keep_props = HashMap::new();
        keep_props.insert(
            "oob".to_string(),
            ParameterValue::String("keep".to_string()),
        );
        let result = ScaleType::binned().resolve_properties("x", &keep_props);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not support oob='keep'"));
    }

    #[test]
    fn test_resolve_properties_defaults() {
        // Continuous positional: default expand
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert!(resolved.contains_key("expand"));
        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => assert!((n - DEFAULT_EXPAND_MULT).abs() < 1e-10),
            _ => panic!("Expected Number"),
        }

        // Continuous non-positional: no default expand, but has oob
        let resolved = ScaleType::continuous()
            .resolve_properties("color", &props)
            .unwrap();
        assert!(!resolved.contains_key("expand"));
        assert!(resolved.contains_key("oob"));

        // Binned: default oob is censor
        let resolved = ScaleType::binned().resolve_properties("x", &props).unwrap();
        match resolved.get("oob") {
            Some(ParameterValue::String(s)) => assert_eq!(s, "censor"),
            _ => panic!("Expected oob to be 'censor'"),
        }

        // Discrete: only reverse property
        let resolved = ScaleType::discrete()
            .resolve_properties("color", &props)
            .unwrap();
        assert!(resolved.contains_key("reverse"));
        assert_eq!(resolved.len(), 1);
    }

    #[test]
    fn test_resolve_properties_user_values_preserved() {
        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => assert!((n - 0.1).abs() < 1e-10),
            _ => panic!("Expected Number"),
        }

        // Binned supports expand
        props.insert("expand".to_string(), ParameterValue::Number(0.2));
        let resolved = ScaleType::binned().resolve_properties("x", &props).unwrap();
        match resolved.get("expand") {
            Some(ParameterValue::Number(n)) => assert!((n - 0.2).abs() < 1e-10),
            _ => panic!("Expected Number"),
        }

        // Binned allows squish oob
        let mut oob_props = HashMap::new();
        oob_props.insert(
            "oob".to_string(),
            ParameterValue::String("squish".to_string()),
        );
        assert!(ScaleType::binned()
            .resolve_properties("x", &oob_props)
            .is_ok());
    }

    #[test]
    fn test_expand_positional_vs_non_positional() {
        use crate::plot::aesthetic::ALL_POSITIONAL;

        let mut props = HashMap::new();
        props.insert("expand".to_string(), ParameterValue::Number(0.1));

        // Positional aesthetics should allow expand
        for aes in ALL_POSITIONAL.iter() {
            assert!(
                ScaleType::continuous()
                    .resolve_properties(aes, &props)
                    .is_ok(),
                "{} should allow expand",
                aes
            );
        }

        // Non-positional aesthetics should reject expand
        for aes in &["color", "size", "opacity"] {
            let result = ScaleType::continuous().resolve_properties(aes, &props);
            assert!(result.is_err(), "{} should reject expand", aes);
        }
    }

    // =========================================================================
    // OOB Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_oob_defaults_by_aesthetic_type() {
        use crate::plot::aesthetic::ALL_POSITIONAL;

        let props = HashMap::new();

        // Positional aesthetics default to 'keep'
        for aesthetic in ALL_POSITIONAL.iter() {
            let resolved = ScaleType::continuous()
                .resolve_properties(aesthetic, &props)
                .unwrap();
            assert_eq!(
                resolved.get("oob"),
                Some(&ParameterValue::String("keep".into())),
                "Positional '{}' should default to 'keep'",
                aesthetic
            );
        }

        // Non-positional aesthetics default to 'censor'
        for aesthetic in &["color", "size", "opacity", "fill", "stroke"] {
            let resolved = ScaleType::continuous()
                .resolve_properties(aesthetic, &props)
                .unwrap();
            assert_eq!(
                resolved.get("oob"),
                Some(&ParameterValue::String("censor".into())),
                "Non-positional '{}' should default to 'censor'",
                aesthetic
            );
        }
    }

    #[test]
    fn test_oob_valid_and_invalid_values() {
        // Valid values accepted
        for oob_value in &["censor", "squish", "keep"] {
            let mut props = HashMap::new();
            props.insert(
                "oob".to_string(),
                ParameterValue::String(oob_value.to_string()),
            );
            assert!(
                ScaleType::continuous()
                    .resolve_properties("x", &props)
                    .is_ok(),
                "oob='{}' should be valid",
                oob_value
            );
        }

        // Invalid value rejected with helpful message
        let mut props = HashMap::new();
        props.insert("oob".to_string(), ParameterValue::String("invalid".into()));
        let result = ScaleType::continuous().resolve_properties("x", &props);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid oob value"));
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
    fn test_oob_scale_type_support() {
        let props = HashMap::new();

        // Continuous and binned support oob
        for scale_type in &[ScaleType::continuous(), ScaleType::binned()] {
            let resolved = scale_type.resolve_properties("color", &props).unwrap();
            assert!(
                resolved.contains_key("oob"),
                "{:?} should support oob",
                scale_type.scale_type_kind()
            );
        }

        // Identity and discrete reject oob
        let mut oob_props = HashMap::new();
        oob_props.insert("oob".to_string(), ParameterValue::String("censor".into()));
        assert!(ScaleType::identity()
            .resolve_properties("color", &oob_props)
            .is_err());
        let result = ScaleType::discrete().resolve_properties("color", &oob_props);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support SETTING 'oob'"));

        // Discrete has no oob in resolved (implicit censor)
        let resolved = ScaleType::discrete()
            .resolve_properties("color", &props)
            .unwrap();
        assert!(!resolved.contains_key("oob"));
    }

    // =========================================================================
    // Transform Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_default_transform_by_aesthetic_and_dtype() {
        use polars::prelude::*;

        // Most aesthetics default to identity (when no column dtype is specified)
        for aesthetic in &["x", "y", "color", "size"] {
            assert_eq!(
                ScaleType::continuous().default_transform(aesthetic, None),
                TransformKind::Identity,
                "{} should default to Identity",
                aesthetic
            );
        }

        // Temporal types infer their transform
        let temporal_cases = vec![
            (DataType::Date, TransformKind::Date),
            (
                DataType::Datetime(TimeUnit::Microseconds, None),
                TransformKind::DateTime,
            ),
            (DataType::Time, TransformKind::Time),
            (DataType::Int64, TransformKind::Identity), // Non-temporal fallback
        ];
        for (dtype, expected) in temporal_cases {
            assert_eq!(
                ScaleType::continuous().default_transform("x", Some(&dtype)),
                expected,
                "{:?} should infer {:?}",
                dtype,
                expected
            );
        }

        // Binned defaults to identity
        for aesthetic in &["x", "size"] {
            assert_eq!(
                ScaleType::binned().default_transform(aesthetic, None),
                TransformKind::Identity
            );
        }
    }

    #[test]
    fn test_allowed_transforms_by_scale_type() {
        // Continuous allows log transforms
        let continuous = ScaleType::continuous().allowed_transforms();
        for kind in &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Log2,
            TransformKind::Sqrt,
            TransformKind::Asinh,
            TransformKind::PseudoLog,
        ] {
            assert!(
                continuous.contains(kind),
                "Continuous should allow {:?}",
                kind
            );
        }

        // Binned allows log transforms
        let binned = ScaleType::binned().allowed_transforms();
        for kind in &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Sqrt,
            TransformKind::Asinh,
        ] {
            assert!(binned.contains(kind), "Binned should allow {:?}", kind);
        }

        // Discrete only allows identity, string, bool
        assert_eq!(
            ScaleType::discrete().allowed_transforms(),
            &[
                TransformKind::Identity,
                TransformKind::String,
                TransformKind::Bool
            ]
        );

        // Identity only allows identity
        assert_eq!(
            ScaleType::identity().allowed_transforms(),
            &[TransformKind::Identity]
        );
    }

    #[test]
    fn test_discrete_transform_acceptance() {
        // Discrete rejects log
        let log = Transform::log();
        let result = ScaleType::discrete().resolve_transform("color", Some(&log), None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not supported"));

        // Discrete accepts string and bool
        for (transform, expected_kind) in [
            (Transform::string(), TransformKind::String),
            (Transform::bool(), TransformKind::Bool),
        ] {
            let result =
                ScaleType::discrete().resolve_transform("color", Some(&transform), None, None);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().transform_kind(), expected_kind);
        }
    }

    #[test]
    fn test_resolve_transform_variations() {
        // Default fills identity
        for aesthetic in &["x", "size"] {
            let result = ScaleType::continuous().resolve_transform(aesthetic, None, None, None);
            assert_eq!(result.unwrap().transform_kind(), TransformKind::Identity);
        }

        // User input accepted for valid transforms
        let log = Transform::log();
        let result = ScaleType::continuous().resolve_transform("y", Some(&log), None, None);
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Log10);
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
            TransformKind::Integer,
            TransformKind::Date,
            TransformKind::DateTime,
            TransformKind::Time,
        ] {
            let transform = Transform::from_kind(*kind);
            let result =
                ScaleType::continuous().resolve_transform("y", Some(&transform), None, None);
            assert!(
                result.is_ok(),
                "Expected {:?} to be valid for continuous",
                kind
            );
            assert_eq!(result.unwrap().transform_kind(), *kind);
        }
    }

    #[test]
    fn test_discrete_infers_transform_from_input_range() {
        // Bool input range -> Bool transform
        let bool_range = vec![ArrayElement::Boolean(true), ArrayElement::Boolean(false)];
        let result = ScaleType::discrete().resolve_transform("fill", None, None, Some(&bool_range));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);

        // String input range -> String transform
        let string_range = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ];
        let result =
            ScaleType::discrete().resolve_transform("fill", None, None, Some(&string_range));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::String);
    }

    #[test]
    fn test_discrete_input_range_overrides_column_dtype() {
        use polars::prelude::DataType;

        // Bool input range should override String column dtype
        let bool_range = vec![ArrayElement::Boolean(true), ArrayElement::Boolean(false)];
        let result = ScaleType::discrete().resolve_transform(
            "fill",
            None,
            Some(&DataType::String), // Column is String
            Some(&bool_range),       // But input range is Bool
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);
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
    fn test_breaks_property_default_is_7() {
        let props = HashMap::new();
        let resolved = ScaleType::continuous()
            .resolve_properties("x", &props)
            .unwrap();
        assert_eq!(resolved.get("breaks"), Some(&ParameterValue::Number(7.0)));
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
        // linear_breaks now extends one step before and after
        // Negative values in sqrt space get clipped, so we get more than 5 breaks
        assert!(
            breaks.len() >= 5,
            "Should have at least 5 breaks, got {}",
            breaks.len()
        );
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

    #[test]
    fn test_resolve_string_interval_breaks_date() {
        use crate::plot::scale::Scale;

        // Set up a date scale with an interval string like "2 months"
        let mut scale = Scale::new("x");
        scale.scale_type = Some(ScaleType::continuous());
        scale.transform = Some(Transform::date());
        // Date range: 2024-01-15 to 2024-06-15 (roughly 5 months)
        // 2024-01-15 = day 19738, 2024-06-15 = day 19889
        scale.input_range = Some(vec![
            ArrayElement::Date(19738), // 2024-01-15
            ArrayElement::Date(19889), // 2024-06-15
        ]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::String("2 months".to_string()),
        );

        let context = ScaleDataContext::new();
        ScaleType::continuous()
            .resolve(&mut scale, &context, "x")
            .unwrap();

        // Should have converted to Array with date breaks
        match scale.properties.get("breaks") {
            Some(ParameterValue::Array(breaks)) => {
                assert!(!breaks.is_empty(), "breaks should not be empty");
                // Check that the breaks are Date types
                for brk in breaks {
                    assert!(
                        matches!(brk, ArrayElement::Date(_)),
                        "breaks should be Date elements"
                    );
                }
            }
            _ => panic!("breaks should be an Array after resolution"),
        }
    }

    #[test]
    fn test_resolve_string_interval_breaks_datetime() {
        use crate::plot::scale::Scale;

        // Set up a datetime scale with an interval string like "month"
        let mut scale = Scale::new("x");
        scale.scale_type = Some(ScaleType::continuous());
        scale.transform = Some(Transform::datetime());
        // DateTime range: 2024-01-01 to 2024-04-01 (3 months)
        // Microseconds since epoch for these dates
        let jan1_2024_us = 1704067200_i64 * 1_000_000; // 2024-01-01 00:00:00 UTC
        let apr1_2024_us = 1711929600_i64 * 1_000_000; // 2024-04-01 00:00:00 UTC
        scale.input_range = Some(vec![
            ArrayElement::DateTime(jan1_2024_us),
            ArrayElement::DateTime(apr1_2024_us),
        ]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::String("month".to_string()),
        );

        let context = ScaleDataContext::new();
        ScaleType::continuous()
            .resolve(&mut scale, &context, "x")
            .unwrap();

        // Should have converted to Array with datetime breaks
        match scale.properties.get("breaks") {
            Some(ParameterValue::Array(breaks)) => {
                assert!(!breaks.is_empty(), "breaks should not be empty");
                // Check that the breaks are DateTime types
                for brk in breaks {
                    assert!(
                        matches!(brk, ArrayElement::DateTime(_)),
                        "breaks should be DateTime elements"
                    );
                }
            }
            _ => panic!("breaks should be an Array after resolution"),
        }
    }

    #[test]
    fn test_resolve_string_interval_breaks_invalid_interval() {
        use crate::plot::scale::Scale;

        // Invalid interval string should be ignored (no crash)
        let mut scale = Scale::new("x");
        scale.scale_type = Some(ScaleType::continuous());
        scale.transform = Some(Transform::date());
        scale.input_range = Some(vec![ArrayElement::Date(19738), ArrayElement::Date(19889)]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::String("invalid_interval".to_string()),
        );

        let context = ScaleDataContext::new();
        // Should not error, just leave breaks as-is
        ScaleType::continuous()
            .resolve(&mut scale, &context, "x")
            .unwrap();

        // breaks should still be a String (not converted)
        match scale.properties.get("breaks") {
            Some(ParameterValue::String(_)) => {
                // Expected - invalid interval was ignored
            }
            _ => panic!("invalid interval should leave breaks unchanged"),
        }
    }

    #[test]
    fn test_resolve_string_interval_breaks_non_temporal_ignored() {
        use crate::plot::scale::Scale;

        // String interval on non-temporal transform should be ignored
        let mut scale = Scale::new("x");
        scale.scale_type = Some(ScaleType::continuous());
        scale.transform = Some(Transform::identity()); // Not temporal
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::String("2 months".to_string()),
        );

        let context = ScaleDataContext::new();
        ScaleType::continuous()
            .resolve(&mut scale, &context, "x")
            .unwrap();

        // breaks should still be a String (not converted)
        match scale.properties.get("breaks") {
            Some(ParameterValue::String(_)) => {
                // Expected - non-temporal transform ignores interval strings
            }
            _ => panic!("non-temporal transform should leave breaks unchanged"),
        }
    }

    // =========================================================================
    // Type Coercion Tests
    // =========================================================================

    #[test]
    fn test_coerce_dtypes_single_type() {
        assert_eq!(coerce_dtypes(&[DataType::Int64]).unwrap(), DataType::Int64);
        assert_eq!(
            coerce_dtypes(&[DataType::String]).unwrap(),
            DataType::String
        );
        assert_eq!(coerce_dtypes(&[DataType::Date]).unwrap(), DataType::Date);
    }

    #[test]
    fn test_coerce_dtypes_numeric_family() {
        // Boolean → Int → Float hierarchy
        assert_eq!(
            coerce_dtypes(&[DataType::Boolean, DataType::Int64]).unwrap(),
            DataType::Int64
        );
        assert_eq!(
            coerce_dtypes(&[DataType::Int32, DataType::Float64]).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            coerce_dtypes(&[DataType::Boolean, DataType::Float64]).unwrap(),
            DataType::Float64
        );
    }

    #[test]
    fn test_coerce_dtypes_string_absorbs_all() {
        // String is most general
        assert_eq!(
            coerce_dtypes(&[DataType::String, DataType::Int64]).unwrap(),
            DataType::String
        );
        assert_eq!(
            coerce_dtypes(&[DataType::String, DataType::Date]).unwrap(),
            DataType::String
        );
    }

    #[test]
    fn test_coerce_dtypes_incompatible_families_to_string() {
        // Numeric + Temporal → String
        assert_eq!(
            coerce_dtypes(&[DataType::Int64, DataType::Date]).unwrap(),
            DataType::String
        );
        assert_eq!(
            coerce_dtypes(&[DataType::Float64, DataType::Time]).unwrap(),
            DataType::String
        );
    }

    #[test]
    fn test_coerce_dtypes_temporal_same_type() {
        use polars::prelude::TimeUnit;
        // Same temporal types pass through
        assert_eq!(
            coerce_dtypes(&[DataType::Date, DataType::Date]).unwrap(),
            DataType::Date
        );
        let dt = DataType::Datetime(TimeUnit::Microseconds, None);
        assert!(coerce_dtypes(&[dt.clone(), dt.clone()]).is_ok());
    }

    #[test]
    fn test_coerce_dtypes_temporal_mixed_error() {
        use polars::prelude::TimeUnit;
        // Mixed temporal types error
        let result = coerce_dtypes(&[
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
        ]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Cannot mix different temporal types"));
    }

    #[test]
    fn test_coerce_dtypes_empty() {
        assert_eq!(coerce_dtypes(&[]).unwrap(), DataType::String);
    }

    // =========================================================================
    // needs_cast Tests
    // =========================================================================

    #[test]
    fn test_needs_cast_same_type() {
        // Same types - no cast needed
        assert!(needs_cast(&DataType::String, &DataType::String).is_none());
        assert!(needs_cast(&DataType::Date, &DataType::Date).is_none());
        assert!(needs_cast(&DataType::Boolean, &DataType::Boolean).is_none());
    }

    #[test]
    fn test_needs_cast_numeric_to_float() {
        // Numeric to Float64 - DuckDB handles implicitly
        assert!(needs_cast(&DataType::Int64, &DataType::Float64).is_none());
        assert!(needs_cast(&DataType::Int32, &DataType::Float64).is_none());
        assert!(needs_cast(&DataType::Float32, &DataType::Float64).is_none());
    }

    #[test]
    fn test_needs_cast_string_to_date() {
        // String to Date - needs explicit cast
        let result = needs_cast(&DataType::String, &DataType::Date);
        assert_eq!(result, Some(CastTargetType::Date));
    }

    #[test]
    fn test_needs_cast_int_to_string() {
        // Int to String - needs explicit cast
        let result = needs_cast(&DataType::Int64, &DataType::String);
        assert_eq!(result, Some(CastTargetType::String));
    }

    #[test]
    fn test_needs_cast_bool_to_string() {
        // Bool to String - needs explicit cast
        let result = needs_cast(&DataType::Boolean, &DataType::String);
        assert_eq!(result, Some(CastTargetType::String));
    }

    // =========================================================================
    // dtype_to_cast_target Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_cast_target() {
        assert_eq!(
            dtype_to_cast_target(&DataType::Int64),
            CastTargetType::Number
        );
        assert_eq!(
            dtype_to_cast_target(&DataType::Float64),
            CastTargetType::Number
        );
        assert_eq!(dtype_to_cast_target(&DataType::Date), CastTargetType::Date);
        assert_eq!(
            dtype_to_cast_target(&DataType::String),
            CastTargetType::String
        );
        assert_eq!(
            dtype_to_cast_target(&DataType::Boolean),
            CastTargetType::Boolean
        );
    }

    // =========================================================================
    // SqlTypeNames Tests
    // =========================================================================

    #[test]
    fn test_sql_type_names_for_target() {
        let names = SqlTypeNames {
            number: Some("DOUBLE".to_string()),
            integer: Some("BIGINT".to_string()),
            date: Some("DATE".to_string()),
            datetime: Some("TIMESTAMP".to_string()),
            time: Some("TIME".to_string()),
            string: Some("VARCHAR".to_string()),
            boolean: Some("BOOLEAN".to_string()),
        };
        assert_eq!(names.for_target(CastTargetType::Number), Some("DOUBLE"));
        assert_eq!(names.for_target(CastTargetType::Integer), Some("BIGINT"));
        assert_eq!(names.for_target(CastTargetType::Date), Some("DATE"));
        assert_eq!(
            names.for_target(CastTargetType::DateTime),
            Some("TIMESTAMP")
        );
        assert_eq!(names.for_target(CastTargetType::Time), Some("TIME"));
        assert_eq!(names.for_target(CastTargetType::String), Some("VARCHAR"));
        assert_eq!(names.for_target(CastTargetType::Boolean), Some("BOOLEAN"));
    }

    // =========================================================================
    // clip_to_transform_domain Tests
    // =========================================================================

    #[test]
    fn test_clip_to_transform_domain_identity() {
        // Identity transform allows all values, so no clipping
        let transform = Transform::identity();
        let range = vec![ArrayElement::Number(-100.0), ArrayElement::Number(100.0)];
        let clipped = clip_to_transform_domain(&range, &transform);
        assert_eq!(clipped[0], ArrayElement::Number(-100.0));
        assert_eq!(clipped[1], ArrayElement::Number(100.0));
    }

    #[test]
    fn test_clip_to_transform_domain_log() {
        // Log transform excludes 0 and negative values
        let transform = Transform::from_kind(TransformKind::Log10);
        let range = vec![ArrayElement::Number(-5.0), ArrayElement::Number(100.0)];
        let clipped = clip_to_transform_domain(&range, &transform);
        // Min should be clipped to f64::MIN_POSITIVE
        assert_eq!(clipped[0], ArrayElement::Number(f64::MIN_POSITIVE));
        assert_eq!(clipped[1], ArrayElement::Number(100.0));
    }

    #[test]
    fn test_clip_to_transform_domain_sqrt() {
        // Sqrt transform requires non-negative values
        let transform = Transform::from_kind(TransformKind::Sqrt);
        let range = vec![ArrayElement::Number(-5.0), ArrayElement::Number(100.0)];
        let clipped = clip_to_transform_domain(&range, &transform);
        // Min should be clipped to 0.0
        assert_eq!(clipped[0], ArrayElement::Number(0.0));
        assert_eq!(clipped[1], ArrayElement::Number(100.0));
    }

    #[test]
    fn test_clip_to_transform_domain_both_sides() {
        // Test clipping both min and max (though unrealistic for typical transforms)
        let transform = Transform::from_kind(TransformKind::Time);
        // Time is 0 to 24 hours in nanoseconds
        let range = vec![ArrayElement::Number(-1000.0), ArrayElement::Number(1e20)];
        let clipped = clip_to_transform_domain(&range, &transform);
        // Min should be clipped to 0.0
        assert_eq!(clipped[0], ArrayElement::Number(0.0));
        // Max should be clipped to max time nanos (24 * 3600 * 1e9)
        let max_time = 24.0 * 3600.0 * 1e9;
        assert_eq!(clipped[1], ArrayElement::Number(max_time));
    }

    #[test]
    fn test_clip_to_transform_domain_no_clipping_needed() {
        // Values already within domain - no clipping
        let transform = Transform::from_kind(TransformKind::Log10);
        let range = vec![ArrayElement::Number(0.001), ArrayElement::Number(1000.0)];
        let clipped = clip_to_transform_domain(&range, &transform);
        assert_eq!(clipped[0], ArrayElement::Number(0.001));
        assert_eq!(clipped[1], ArrayElement::Number(1000.0));
    }

    #[test]
    fn test_expansion_clipped_to_log_domain() {
        // Simulate what happens when expansion produces invalid values for log scale
        // Data: [0.001, 0.01], with 50% expansion produces negative min
        let range = vec![ArrayElement::Number(0.001), ArrayElement::Number(0.01)];
        // span = 0.009, expansion = 0.0045
        // expanded_min = 0.001 - 0.0045 = -0.0035
        // expanded_max = 0.01 + 0.0045 = 0.0145
        let expanded = expand_numeric_range(&range, 0.5, 0.0);
        assert!(expanded[0].to_f64().unwrap() < 0.0);

        // Now clip to log domain
        let transform = Transform::from_kind(TransformKind::Log10);
        let clipped = clip_to_transform_domain(&expanded, &transform);
        // Min should be clipped to f64::MIN_POSITIVE
        assert_eq!(clipped[0], ArrayElement::Number(f64::MIN_POSITIVE));
        assert_eq!(clipped[1].to_f64().unwrap(), 0.0145);
    }

    // =========================================================================
    // Output Range Helper Tests
    // =========================================================================

    #[test]
    fn test_interpolate_numeric_basic() {
        let range = vec![ArrayElement::Number(1.0), ArrayElement::Number(6.0)];
        let result = interpolate_numeric(&range, 5).unwrap();

        assert_eq!(result.len(), 5);
        assert!((result[0].to_f64().unwrap() - 1.0).abs() < 0.001);
        assert!((result[4].to_f64().unwrap() - 6.0).abs() < 0.001);
        // Check middle values are evenly spaced
        assert!((result[1].to_f64().unwrap() - 2.25).abs() < 0.001);
        assert!((result[2].to_f64().unwrap() - 3.5).abs() < 0.001);
        assert!((result[3].to_f64().unwrap() - 4.75).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_numeric_two_values() {
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let result = interpolate_numeric(&range, 2).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0].to_f64().unwrap() - 0.0).abs() < 0.001);
        assert!((result[1].to_f64().unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_numeric_single_output() {
        // With count=1, use midpoint
        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let result = interpolate_numeric(&range, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0].to_f64().unwrap() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_numeric_empty_input() {
        let range: Vec<ArrayElement> = vec![];
        assert!(interpolate_numeric(&range, 5).is_none());
    }

    #[test]
    fn test_interpolate_numeric_single_input() {
        let range = vec![ArrayElement::Number(1.0)];
        assert!(interpolate_numeric(&range, 5).is_none());
    }

    #[test]
    fn test_interpolate_numeric_zero_count() {
        let range = vec![ArrayElement::Number(1.0), ArrayElement::Number(6.0)];
        assert!(interpolate_numeric(&range, 0).is_none());
    }

    #[test]
    fn test_interpolate_numeric_non_numeric_values() {
        let range = vec![
            ArrayElement::String("a".to_string()),
            ArrayElement::String("b".to_string()),
        ];
        // Should return None because values are not numeric
        assert!(interpolate_numeric(&range, 3).is_none());
    }

    #[test]
    fn test_size_output_range_color_interpolation() {
        use super::super::OutputRange;

        let mut scale = super::super::Scale::new("fill");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("#ff0000".to_string()),
            ArrayElement::String("#0000ff".to_string()),
        ]));

        size_output_range(&mut scale, "fill", 3).unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected OutputRange::Array");
        }
    }

    #[test]
    fn test_size_output_range_size_interpolation() {
        use super::super::OutputRange;

        let mut scale = super::super::Scale::new("size");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(10.0),
        ]));

        size_output_range(&mut scale, "size", 4).unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 4);
            // Values should be 1, 4, 7, 10
            assert!((arr[0].to_f64().unwrap() - 1.0).abs() < 0.001);
            assert!((arr[3].to_f64().unwrap() - 10.0).abs() < 0.001);
        } else {
            panic!("Expected OutputRange::Array");
        }
    }

    #[test]
    fn test_size_output_range_shape_truncates() {
        use super::super::OutputRange;

        let mut scale = super::super::Scale::new("shape");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("circle".to_string()),
            ArrayElement::String("square".to_string()),
            ArrayElement::String("triangle".to_string()),
            ArrayElement::String("diamond".to_string()),
        ]));

        size_output_range(&mut scale, "shape", 2).unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("Expected OutputRange::Array");
        }
    }

    #[test]
    fn test_size_output_range_shape_error_insufficient() {
        use super::super::OutputRange;

        let mut scale = super::super::Scale::new("shape");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("circle".to_string()),
            ArrayElement::String("square".to_string()),
        ]));

        let result = size_output_range(&mut scale, "shape", 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("2 values"));
        // Note: grammar-aware "5 are needed"
    }

    // =========================================================================
    // Input Range Length Validation Tests
    // =========================================================================

    #[test]
    fn test_continuous_scale_rejects_wrong_input_range_length() {
        let scale_type = ScaleType::continuous();
        let context = ScaleDataContext::default();

        // Test with 1 value
        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0)]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 2 values"));

        // Test with 3 values
        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(100.0),
        ]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("exactly 2 values"));
        assert!(err.contains("got 3"));
    }

    #[test]
    fn test_binned_scale_rejects_wrong_input_range_length() {
        let scale_type = ScaleType::binned();
        let context = ScaleDataContext::default();

        // Test with 1 value
        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0)]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 2 values"));

        // Test with 3 values
        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(100.0),
        ]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("exactly 2 values"));
        assert!(err.contains("got 3"));
    }

    #[test]
    fn test_continuous_scale_accepts_two_element_input_range() {
        let scale_type = ScaleType::continuous();
        let context = ScaleDataContext::default();

        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_binned_scale_accepts_two_element_input_range() {
        let scale_type = ScaleType::binned();
        let context = ScaleDataContext::default();

        let mut scale = super::super::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_discrete_scale_allows_any_input_range_length() {
        let scale_type = ScaleType::discrete();
        let context = ScaleDataContext::default();

        // Test with 1 value
        let mut scale = super::super::Scale::new("color");
        scale.input_range = Some(vec![ArrayElement::String("A".to_string())]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "color");
        assert!(result.is_ok());

        // Test with 3 values
        let mut scale = super::super::Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
        ]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "color");
        assert!(result.is_ok());

        // Test with 5 values
        let mut scale = super::super::Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
            ArrayElement::String("D".to_string()),
            ArrayElement::String("E".to_string()),
        ]);
        scale.explicit_input_range = true;
        let result = resolve_common_steps(&*scale_type.0, &mut scale, &context, "color");
        assert!(result.is_ok());
    }
}
