//! Scale creation, resolution, type coercion, and OOB handling.
//!
//! This module handles creating default scales for aesthetics, resolving
//! scale properties from data, type coercion based on scale requirements,
//! and out-of-bounds (OOB) handling.

use crate::naming;
use crate::plot::aesthetic::primary_aesthetic;
use crate::plot::layer::geom::get_aesthetic_family;
use crate::plot::scale::{
    default_oob, gets_default_scale, infer_scale_target_type, infer_transform_from_input_range,
    transform::Transform, OOB_CENSOR, OOB_KEEP, OOB_SQUISH,
};
use crate::plot::{
    AestheticValue, ArrayElement, ArrayElementType, ColumnInfo, Layer, ParameterValue, Plot, Scale,
    ScaleType, ScaleTypeKind, Schema,
};
use crate::{DataFrame, GgsqlError, Result};
use polars::prelude::Column;
use std::collections::{HashMap, HashSet};

use super::schema::TypeInfo;

/// Create Scale objects for aesthetics that don't have explicit SCALE clauses.
///
/// For aesthetics with meaningful scale behavior, creates a minimal scale
/// (type will be inferred later by resolve_scales from column dtype).
/// For identity aesthetics (text, label, group, etc.), creates an Identity scale.
pub fn create_missing_scales(spec: &mut Plot) {
    let mut used_aesthetics: HashSet<String> = HashSet::new();

    // Collect from layer mappings and remappings
    // (global mappings have already been merged into layers at this point)
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = primary_aesthetic(aesthetic);
            used_aesthetics.insert(primary.to_string());
        }
        for aesthetic in layer.remappings.aesthetics.keys() {
            let primary = primary_aesthetic(aesthetic);
            used_aesthetics.insert(primary.to_string());
        }
    }

    // Find aesthetics that already have explicit scales
    let existing_scales: HashSet<String> =
        spec.scales.iter().map(|s| s.aesthetic.clone()).collect();

    // Create scales for missing aesthetics
    for aesthetic in used_aesthetics {
        if !existing_scales.contains(&aesthetic) {
            let mut scale = Scale::new(&aesthetic);
            // Set Identity scale type for aesthetics that don't get default scales
            if !gets_default_scale(&aesthetic) {
                scale.scale_type = Some(ScaleType::identity());
            }
            spec.scales.push(scale);
        }
    }
}

/// Create scales for aesthetics that appeared from stat transforms (remappings).
///
/// Called after build_layer_query() to handle aesthetics like:
/// - y → __ggsql_stat_count__ (histogram, bar)
/// - x2 → __ggsql_stat_bin_end__ (histogram)
///
/// This is necessary because stat transforms modify layer.mappings after
/// create_missing_scales() has already run, potentially adding new aesthetics
/// that don't have corresponding scales.
pub fn create_missing_scales_post_stat(spec: &mut Plot) {
    let mut current_aesthetics: HashSet<String> = HashSet::new();

    // Collect all aesthetics currently in layer mappings
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = primary_aesthetic(aesthetic);
            current_aesthetics.insert(primary.to_string());
        }
    }

    // Find aesthetics that don't have scales yet
    let existing_scales: HashSet<String> =
        spec.scales.iter().map(|s| s.aesthetic.clone()).collect();

    // Create scales for new aesthetics
    for aesthetic in current_aesthetics {
        if !existing_scales.contains(&aesthetic) {
            let mut scale = Scale::new(&aesthetic);
            if !gets_default_scale(&aesthetic) {
                scale.scale_type = Some(ScaleType::identity());
            }
            spec.scales.push(scale);
        }
    }
}

// =============================================================================
// Post-Stat Binning
// =============================================================================

/// Apply binning directly to DataFrame columns for post-stat aesthetics.
///
/// This handles cases where a user specifies `SCALE BINNED` on a remapped aesthetic
/// (e.g., binning histogram's count output if remapped to fill).
///
/// Called after resolve_scales() so that breaks have been calculated.
///
/// This handles binning for aesthetics that get their values from stat transforms
/// (e.g., SCALE BINNED fill when fill is remapped from count). Aesthetics that
/// are directly mapped from source columns are pre-stat binned via SQL transforms.
pub fn apply_post_stat_binning(
    spec: &Plot,
    data_map: &mut HashMap<String, DataFrame>,
) -> Result<()> {
    for scale in &spec.scales {
        // Only process Binned scales
        match &scale.scale_type {
            Some(st) if st.scale_type_kind() == ScaleTypeKind::Binned => {}
            _ => continue,
        }

        // Get breaks from properties (skip if no breaks calculated)
        let breaks = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(arr)) if arr.len() >= 2 => arr,
            _ => continue,
        };

        // Extract break values as f64
        let break_values: Vec<f64> = breaks.iter().filter_map(|e| e.to_f64()).collect();

        if break_values.len() < 2 {
            continue;
        }

        // Get closed property (default: left)
        let closed_left = match scale.properties.get("closed") {
            Some(ParameterValue::String(s)) => s != "right",
            _ => true,
        };

        // Find columns for this aesthetic across layers
        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        // Apply binning to each column
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
                    continue;
                }

                // Skip post-stat binning for aesthetic columns (like __ggsql_aes_x__)
                // because pre_stat_transform already binned them via SQL.
                // Post-stat binning only applies to stat columns or remapped aesthetics.
                if naming::is_aesthetic_column(&col_name) {
                    continue;
                }

                let binned_df =
                    apply_binning_to_dataframe(df, &col_name, &break_values, closed_left)?;
                data_map.insert(data_key, binned_df);
            }
        }
    }

    Ok(())
}

/// Apply binning transformation to a DataFrame column.
///
/// Replaces each value with the center of its bin based on the break values.
pub fn apply_binning_to_dataframe(
    df: &DataFrame,
    col_name: &str,
    break_values: &[f64],
    closed_left: bool,
) -> Result<DataFrame> {
    use polars::prelude::*;

    let column = df.column(col_name).map_err(|e| {
        GgsqlError::InternalError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let series = column.as_materialized_series();

    // Cast to f64 for binning
    let float_series = series.cast(&DataType::Float64).map_err(|e| {
        GgsqlError::InternalError(format!("Cannot bin column '{}': {}", col_name, e))
    })?;

    let ca = float_series
        .f64()
        .map_err(|e| GgsqlError::InternalError(e.to_string()))?;

    // Apply binning: replace values with bin centers
    let num_bins = break_values.len() - 1;
    let binned: Float64Chunked = ca.apply_values(|val| {
        for i in 0..num_bins {
            let lower = break_values[i];
            let upper = break_values[i + 1];
            let is_last = i == num_bins - 1;

            let in_bin = if closed_left {
                // Left-closed: [lower, upper) except last bin is [lower, upper]
                if is_last {
                    val >= lower && val <= upper
                } else {
                    val >= lower && val < upper
                }
            } else {
                // Right-closed: (lower, upper] except first bin is [lower, upper]
                if i == 0 {
                    val >= lower && val <= upper
                } else {
                    val > lower && val <= upper
                }
            };

            if in_bin {
                return (lower + upper) / 2.0;
            }
        }
        f64::NAN // Outside all bins
    });

    let binned_series = binned.into_series().with_name(col_name.into());

    // Replace column in DataFrame
    let mut new_df = df.clone();
    let _ = new_df
        .replace(col_name, binned_series)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))?;

    Ok(new_df)
}

// =============================================================================
// Scale Type and Transform Resolution
// =============================================================================

/// Resolve scale types and transforms early, based on column dtypes.
///
/// This function:
/// 1. Infers scale_type from column dtype if not explicitly set
/// 2. Applies type coercion across layers for same aesthetic
/// 3. Resolves transform from scale_type + dtype if not explicit
///
/// Called early in the pipeline so that type requirements can be determined
/// before min/max extraction.
pub fn resolve_scale_types_and_transforms(
    spec: &mut Plot,
    layer_type_info: &[Vec<TypeInfo>],
) -> Result<()> {
    use crate::plot::scale::coerce_dtypes;

    for scale in &mut spec.scales {
        // Skip scales that already have explicit types (user specified)
        if let Some(scale_type) = &scale.scale_type {
            // Collect all dtypes for validation and transform inference
            let all_dtypes =
                collect_dtypes_for_aesthetic(&spec.layers, &scale.aesthetic, layer_type_info);

            // Validate that explicit scale type is compatible with data type
            if !all_dtypes.is_empty() {
                if let Ok(common_dtype) = coerce_dtypes(&all_dtypes) {
                    // Validate dtype compatibility
                    scale_type.validate_dtype(&common_dtype).map_err(|e| {
                        GgsqlError::ValidationError(format!("Scale '{}': {}", scale.aesthetic, e))
                    })?;

                    // Resolve transform if not set
                    if scale.transform.is_none() && !scale.explicit_transform {
                        // For Discrete/Ordinal scales, check input range first for transform inference
                        // This allows SCALE DISCRETE x FROM [true, false] to infer Bool transform
                        // even when the column is String
                        let transform_kind = if matches!(
                            scale_type.scale_type_kind(),
                            ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal
                        ) {
                            if let Some(ref input_range) = scale.input_range {
                                if let Some(kind) = infer_transform_from_input_range(input_range) {
                                    kind
                                } else {
                                    scale_type
                                        .default_transform(&scale.aesthetic, Some(&common_dtype))
                                }
                            } else {
                                scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                            }
                        } else {
                            scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                        };
                        scale.transform = Some(Transform::from_kind(transform_kind));
                    }
                }
            }
            continue;
        }

        // Collect all dtypes for this aesthetic across layers
        let all_dtypes =
            collect_dtypes_for_aesthetic(&spec.layers, &scale.aesthetic, layer_type_info);

        if all_dtypes.is_empty() {
            continue;
        }

        // Determine common dtype through coercion
        let common_dtype = match coerce_dtypes(&all_dtypes) {
            Ok(dt) => dt,
            Err(e) => {
                return Err(GgsqlError::ValidationError(format!(
                    "Scale '{}': {}",
                    scale.aesthetic, e
                )));
            }
        };

        // Infer scale type, considering explicit transform if set
        // If user specified VIA date/datetime/time/log/sqrt/etc., use Continuous scale
        let inferred_scale_type = if scale.explicit_transform {
            if let Some(ref transform) = scale.transform {
                use crate::plot::scale::TransformKind;
                match transform.transform_kind() {
                    // Temporal transforms require Continuous scale
                    TransformKind::Date
                    | TransformKind::DateTime
                    | TransformKind::Time
                    // Numeric continuous transforms require Continuous scale
                    | TransformKind::Log10
                    | TransformKind::Log2
                    | TransformKind::Log
                    | TransformKind::Sqrt
                    | TransformKind::Square
                    | TransformKind::Exp10
                    | TransformKind::Exp2
                    | TransformKind::Exp
                    | TransformKind::Asinh
                    | TransformKind::PseudoLog
                    // Integer transform uses Continuous scale
                    | TransformKind::Integer => ScaleType::continuous(),
                    // Discrete transforms (String, Bool) use Discrete scale
                    TransformKind::String | TransformKind::Bool => ScaleType::discrete(),
                    // Identity: fall back to dtype inference
                    TransformKind::Identity => ScaleType::infer(&common_dtype),
                }
            } else {
                ScaleType::infer(&common_dtype)
            }
        } else {
            ScaleType::infer(&common_dtype)
        };
        scale.scale_type = Some(inferred_scale_type.clone());

        // Infer transform if not explicit
        if scale.transform.is_none() && !scale.explicit_transform {
            // For Discrete scales, check input range first for transform inference
            // This allows SCALE DISCRETE x FROM [true, false] to infer Bool transform
            // even when the column is String
            let transform_kind = if inferred_scale_type.scale_type_kind() == ScaleTypeKind::Discrete
            {
                if let Some(ref input_range) = scale.input_range {
                    if let Some(kind) = infer_transform_from_input_range(input_range) {
                        kind
                    } else {
                        inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                    }
                } else {
                    inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                }
            } else {
                inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
            };
            scale.transform = Some(Transform::from_kind(transform_kind));
        }
    }

    Ok(())
}

/// Collect all dtypes for an aesthetic across layers.
pub fn collect_dtypes_for_aesthetic(
    layers: &[Layer],
    aesthetic: &str,
    layer_type_info: &[Vec<TypeInfo>],
) -> Vec<polars::prelude::DataType> {
    let mut dtypes = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx >= layer_type_info.len() {
            continue;
        }
        let type_info = &layer_type_info[layer_idx];

        for aes_name in &aesthetics_to_check {
            if let Some(value) = layer.mappings.get(aes_name) {
                if let Some(col_name) = value.column_name() {
                    if let Some((_, dtype, _)) = type_info.iter().find(|(n, _, _)| n == col_name) {
                        dtypes.push(dtype.clone());
                    }
                }
            }
        }
    }
    dtypes
}

// =============================================================================
// Pre-Stat Scale Resolution (Binned Scales)
// =============================================================================

/// Pre-resolve Binned scales using schema-derived context.
///
/// This function resolves Binned scales before layer queries are built,
/// so that `pre_stat_transform_sql` has access to resolved breaks for
/// generating binning SQL.
///
/// Only Binned scales are resolved here; other scales are resolved
/// post-stat by `resolve_scales`.
pub fn apply_pre_stat_resolve(spec: &mut Plot, layer_schemas: &[Schema]) -> Result<()> {
    use crate::plot::scale::ScaleDataContext;

    for scale in &mut spec.scales {
        // Only pre-resolve Binned scales
        let scale_type = match &scale.scale_type {
            Some(st) if st.scale_type_kind() == ScaleTypeKind::Binned => st.clone(),
            _ => continue,
        };

        // Find all ColumnInfos for this aesthetic from schemas
        let column_infos =
            find_schema_columns_for_aesthetic(&spec.layers, &scale.aesthetic, layer_schemas);

        if column_infos.is_empty() {
            continue;
        }

        // Build context from schema information
        let context = ScaleDataContext::from_schemas(&column_infos);

        // Use unified resolve method
        scale_type
            .resolve(scale, &context, &scale.aesthetic.clone())
            .map_err(|e| {
                GgsqlError::ValidationError(format!("Scale '{}': {}", scale.aesthetic, e))
            })?;
    }

    Ok(())
}

/// Find ColumnInfo for an aesthetic from layer schemas.
///
/// Similar to `find_columns_for_aesthetic` but works with schema information
/// (ColumnInfo) instead of actual data (Column).
///
/// Handles both column mappings (looked up in schema) and literal mappings
/// (synthetic ColumnInfo created from the literal value).
///
/// Note: Global mappings have already been merged into layer mappings at this point.
pub fn find_schema_columns_for_aesthetic(
    layers: &[Layer],
    aesthetic: &str,
    layer_schemas: &[Schema],
) -> Vec<ColumnInfo> {
    let mut infos = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping (global mappings already merged)
    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx >= layer_schemas.len() {
            continue;
        }
        let schema = &layer_schemas[layer_idx];

        for aes_name in &aesthetics_to_check {
            if let Some(value) = layer.mappings.get(aes_name) {
                match value {
                    AestheticValue::Column { name, .. } => {
                        if let Some(info) = schema.iter().find(|c| c.name == *name) {
                            infos.push(info.clone());
                        }
                    }
                    AestheticValue::Literal(lit) => {
                        // Create synthetic ColumnInfo from literal
                        if let Some(info) = column_info_from_literal(aes_name, lit) {
                            infos.push(info);
                        }
                    }
                }
            }
        }
    }

    infos
}

/// Create a synthetic ColumnInfo from a literal value.
///
/// Used to include literal mappings in scale resolution.
pub fn column_info_from_literal(aesthetic: &str, lit: &ParameterValue) -> Option<ColumnInfo> {
    use polars::prelude::DataType;

    match lit {
        ParameterValue::Number(n) => Some(ColumnInfo {
            name: naming::const_column(aesthetic),
            dtype: DataType::Float64,
            is_discrete: false,
            min: Some(ArrayElement::Number(*n)),
            max: Some(ArrayElement::Number(*n)),
        }),
        ParameterValue::String(s) => Some(ColumnInfo {
            name: naming::const_column(aesthetic),
            dtype: DataType::String,
            is_discrete: true,
            min: Some(ArrayElement::String(s.clone())),
            max: Some(ArrayElement::String(s.clone())),
        }),
        ParameterValue::Boolean(_) => {
            // Boolean literals don't contribute to numeric ranges
            None
        }
        ParameterValue::Array(_) | ParameterValue::Null => {
            unreachable!("Grammar prevents arrays and null in literal aesthetic mappings")
        }
    }
}

// =============================================================================
// Scale Type Coercion
// =============================================================================

/// Coerce a Polars column to the target ArrayElementType.
///
/// Returns a new DataFrame with the coerced column, or an error if coercion fails.
pub fn coerce_column_to_type(
    df: &DataFrame,
    column_name: &str,
    target_type: ArrayElementType,
) -> Result<DataFrame> {
    use polars::prelude::{DataType, NamedFrom, Series, TimeUnit};

    let column = df.column(column_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", column_name, e))
    })?;

    let series = column.as_materialized_series();
    let dtype = series.dtype();

    // Check if already the target type
    let already_target_type = matches!(
        (dtype, target_type),
        (DataType::Boolean, ArrayElementType::Boolean)
            | (
                DataType::Float64 | DataType::Int64 | DataType::Int32 | DataType::Float32,
                ArrayElementType::Number,
            )
            | (DataType::Date, ArrayElementType::Date)
            | (DataType::Datetime(_, _), ArrayElementType::DateTime)
            | (DataType::Time, ArrayElementType::Time)
            | (DataType::String, ArrayElementType::String)
    );

    if already_target_type {
        return Ok(df.clone());
    }

    // Coerce based on target type
    let new_series: Series = match target_type {
        ArrayElementType::Boolean => {
            // Convert to boolean
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for boolean coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let bool_vec: Vec<Option<bool>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| match opt_s {
                            None => Ok(None),
                            Some(s) => match s.to_lowercase().as_str() {
                                "true" | "yes" | "1" => Ok(Some(true)),
                                "false" | "no" | "0" => Ok(Some(false)),
                                _ => Err(GgsqlError::ValidationError(format!(
                                    "Column '{}' row {}: Cannot coerce string '{}' to boolean",
                                    column_name, idx, s
                                ))),
                            },
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), bool_vec)
                }
                DataType::Int64 | DataType::Int32 | DataType::Float64 | DataType::Float32 => {
                    let f64_series = series.cast(&DataType::Float64).map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot cast column '{}' to float64: {}",
                            column_name, e
                        ))
                    })?;
                    let ca = f64_series.f64().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot get float64 chunked array: {}",
                            e
                        ))
                    })?;
                    let bool_vec: Vec<Option<bool>> =
                        ca.into_iter().map(|opt| opt.map(|n| n != 0.0)).collect();
                    Series::new(column_name.into(), bool_vec)
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to boolean",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::Number => {
            // Convert to float64
            series.cast(&DataType::Float64).map_err(|e| {
                GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' to number: {}",
                    column_name, e
                ))
            })?
        }

        ArrayElementType::Date => {
            // Convert to date (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for date coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let date_vec: Vec<Option<i32>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| {
                            match opt_s {
                                None => Ok(None),
                                Some(s) => {
                                    ArrayElement::from_date_string(s)
                                        .and_then(|e| match e {
                                            ArrayElement::Date(d) => Some(d),
                                            _ => None,
                                        })
                                        .ok_or_else(|| {
                                            GgsqlError::ValidationError(format!(
                                                "Column '{}' row {}: Cannot coerce string '{}' to date (expected YYYY-MM-DD)",
                                                column_name, idx, s
                                            ))
                                        })
                                        .map(Some)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), date_vec)
                        .cast(&DataType::Date)
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!("Cannot create date series: {}", e))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to date",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::DateTime => {
            // Convert to datetime (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for datetime coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let dt_vec: Vec<Option<i64>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| match opt_s {
                            None => Ok(None),
                            Some(s) => ArrayElement::from_datetime_string(s)
                                .and_then(|e| match e {
                                    ArrayElement::DateTime(dt) => Some(dt),
                                    _ => None,
                                })
                                .ok_or_else(|| {
                                    GgsqlError::ValidationError(format!(
                                        "Column '{}' row {}: Cannot coerce string '{}' to datetime",
                                        column_name, idx, s
                                    ))
                                })
                                .map(Some),
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), dt_vec)
                        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!(
                                "Cannot create datetime series: {}",
                                e
                            ))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to datetime",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::Time => {
            // Convert to time (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for time coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let time_vec: Vec<Option<i64>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| {
                            match opt_s {
                                None => Ok(None),
                                Some(s) => {
                                    ArrayElement::from_time_string(s)
                                        .and_then(|e| match e {
                                            ArrayElement::Time(t) => Some(t),
                                            _ => None,
                                        })
                                        .ok_or_else(|| {
                                            GgsqlError::ValidationError(format!(
                                                "Column '{}' row {}: Cannot coerce string '{}' to time (expected HH:MM:SS)",
                                                column_name, idx, s
                                            ))
                                        })
                                        .map(Some)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), time_vec)
                        .cast(&DataType::Time)
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!("Cannot create time series: {}", e))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to time",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::String => {
            // Convert to string
            series
                .cast(&polars::prelude::DataType::String)
                .map_err(|e| {
                    GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' to string: {}",
                        column_name, e
                    ))
                })?
        }
    };

    // Replace the column in the DataFrame
    let mut new_df = df.clone();
    let _ = new_df.replace(column_name, new_series);
    Ok(new_df)
}

/// Coerce columns mapped to an aesthetic in all relevant DataFrames.
///
/// This function finds all columns mapped to the given aesthetic across all layers
/// and coerces them to the target type.
pub fn coerce_aesthetic_columns(
    layers: &[Layer],
    data_map: &mut HashMap<String, DataFrame>,
    aesthetic: &str,
    target_type: ArrayElementType,
) -> Result<()> {
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Track which (data_key, column_name) pairs we've already coerced
    let mut coerced: HashSet<(String, String)> = HashSet::new();

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        let layer_key = naming::layer_key(i);

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                // Skip if layer doesn't have data
                if !data_map.contains_key(&layer_key) {
                    continue;
                }

                // Skip if already coerced
                let key = (layer_key.clone(), name.clone());
                if coerced.contains(&key) {
                    continue;
                }

                // Check if column exists in this DataFrame
                if let Some(df) = data_map.get(&layer_key) {
                    if df.column(name).is_ok() {
                        let coerced_df = coerce_column_to_type(df, name, target_type)?;
                        data_map.insert(layer_key.clone(), coerced_df);
                        coerced.insert(key);
                    }
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Scale Resolution
// =============================================================================

/// Resolve scale properties from data after materialization.
///
/// For each scale, this function:
/// 1. Infers target type and coerces columns if needed
/// 2. Infers scale_type from column data types if not explicitly set
/// 3. Uses the unified `resolve` method to fill in input_range, transform, and breaks
/// 4. Resolves output_range if not already set
///
/// The function inspects columns mapped to the aesthetic (including family
/// members like xmin/xmax for "x") and computes appropriate ranges.
///
/// Scales that were already resolved pre-stat (Binned scales) are skipped.
pub fn resolve_scales(spec: &mut Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    use crate::plot::scale::ScaleDataContext;

    for idx in 0..spec.scales.len() {
        // Clone aesthetic to avoid borrow issues with find_columns_for_aesthetic
        let aesthetic = spec.scales[idx].aesthetic.clone();

        // Skip scales that were already resolved pre-stat (e.g., Binned scales)
        // (resolve_output_range is now handled inside the unified resolve() method)
        if spec.scales[idx].resolved {
            continue;
        }

        // Infer target type and coerce columns if needed
        // This enables e.g. SCALE DISCRETE color FROM [true, false] to coerce string "true"/"false" to boolean
        if let Some(target_type) = infer_scale_target_type(&spec.scales[idx]) {
            coerce_aesthetic_columns(&spec.layers, data_map, &aesthetic, target_type)?;
        }

        // Find column references for this aesthetic (including family members)
        // NOTE: Must be called AFTER coercion so column types are correct
        let column_refs = find_columns_for_aesthetic(&spec.layers, &aesthetic, data_map);

        if column_refs.is_empty() {
            continue;
        }

        // Infer scale_type if not already set
        if spec.scales[idx].scale_type.is_none() {
            spec.scales[idx].scale_type = Some(ScaleType::infer(column_refs[0].dtype()));
        }

        // Clone scale_type (cheap Arc clone) to avoid borrow conflict with mutations
        let scale_type = spec.scales[idx].scale_type.clone();
        if let Some(st) = scale_type {
            // Determine if this scale uses discrete input range (unique values vs min/max)
            let use_discrete_range = st.uses_discrete_input_range();

            // Build context from actual data columns
            let context = ScaleDataContext::from_columns(&column_refs, use_discrete_range);

            // Use unified resolve method (includes resolve_output_range)
            st.resolve(&mut spec.scales[idx], &context, &aesthetic)
                .map_err(|e| {
                    GgsqlError::ValidationError(format!("Scale '{}': {}", aesthetic, e))
                })?;
        }
    }

    Ok(())
}

/// Find all columns for an aesthetic (including family members like xmin/xmax for "x").
/// Each mapping is looked up in its corresponding data source.
/// Returns references to the Columns found.
///
/// Note: Global mappings have already been merged into layer mappings at this point.
pub fn find_columns_for_aesthetic<'a>(
    layers: &[Layer],
    aesthetic: &str,
    data_map: &'a HashMap<String, DataFrame>,
) -> Vec<&'a Column> {
    let mut column_refs = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        if let Some(df) = data_map.get(&naming::layer_key(i)) {
            for aes_name in &aesthetics_to_check {
                if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                    if let Ok(column) = df.column(name) {
                        column_refs.push(column);
                    }
                }
            }
        }
    }

    column_refs
}

// =============================================================================
// Out-of-Bounds (OOB) Handling
// =============================================================================

/// Apply out-of-bounds handling to data based on scale oob properties.
///
/// For each scale with `oob != "keep"`, this function transforms the data:
/// - `censor`: Filter out rows where the aesthetic's column values fall outside the input range
/// - `squish`: Clamp column values to the input range limits (continuous only)
///
/// After all OOB transformations, filters out NULL rows for columns where:
/// - The scale has an explicit input range, AND
/// - NULL is not part of the explicit input range
pub fn apply_scale_oob(spec: &Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    // First pass: apply OOB transformations (censor sets to NULL, squish clamps)
    for scale in &spec.scales {
        // Get oob mode:
        // - If explicitly set, use that value (skip if "keep")
        // - If not set but has explicit input range, use default for aesthetic
        // - Otherwise skip
        let oob_mode = match scale.properties.get("oob") {
            Some(ParameterValue::String(s)) if s != OOB_KEEP => s.as_str(),
            Some(ParameterValue::String(_)) => continue, // explicit "keep"
            None if scale.explicit_input_range => {
                let default = default_oob(&scale.aesthetic);
                if default == OOB_KEEP {
                    continue;
                }
                default
            }
            _ => continue,
        };

        // Get input range, skip if none
        let input_range = match &scale.input_range {
            Some(r) if !r.is_empty() => r,
            _ => continue,
        };

        // Find all (data_key, column_name) pairs for this aesthetic
        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        // Helper to check if element is numeric-like (Number, Date, DateTime, Time)
        fn is_numeric_element(elem: &ArrayElement) -> bool {
            matches!(
                elem,
                ArrayElement::Number(_)
                    | ArrayElement::Date(_)
                    | ArrayElement::DateTime(_)
                    | ArrayElement::Time(_)
            )
        }

        // Helper to extract numeric value from element (dates are days, datetime is µs, etc.)
        fn extract_numeric(elem: &ArrayElement) -> Option<f64> {
            match elem {
                ArrayElement::Number(n) => Some(*n),
                ArrayElement::Date(d) => Some(*d as f64),
                ArrayElement::DateTime(dt) => Some(*dt as f64),
                ArrayElement::Time(t) => Some(*t as f64),
                _ => None,
            }
        }

        // Determine if this is a numeric or discrete range
        let is_numeric_range = is_numeric_element(&input_range[0])
            && input_range.get(1).is_some_and(is_numeric_element);

        // Apply transformation to each (data_key, column_name) pair
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
                    continue;
                }

                let transformed = if is_numeric_range {
                    // Numeric range - extract min/max (works for Number, Date, DateTime, Time)
                    let (range_min, range_max) = match (
                        extract_numeric(&input_range[0]),
                        input_range.get(1).and_then(extract_numeric),
                    ) {
                        (Some(lo), Some(hi)) => (lo, hi),
                        _ => continue,
                    };
                    apply_oob_to_column_numeric(df, &col_name, range_min, range_max, oob_mode)?
                } else {
                    // Discrete range - collect allowed values as strings using to_key_string
                    let allowed_values: HashSet<String> = input_range
                        .iter()
                        .filter(|elem| !matches!(elem, ArrayElement::Null))
                        .map(|elem| elem.to_key_string())
                        .collect();
                    apply_oob_to_column_discrete(df, &col_name, &allowed_values, oob_mode)?
                };
                data_map.insert(data_key, transformed);
            }
        }
    }

    // Second pass: filter out NULL rows for scales with explicit input ranges
    // This handles NULLs created by both pre-stat SQL censoring and post-stat OOB censor
    for scale in &spec.scales {
        // Only filter if explicit input range AND NULL is not in the range
        let should_filter_nulls = scale.explicit_input_range
            && scale
                .input_range
                .as_ref()
                .is_some_and(|range| !range.iter().any(|elem| matches!(elem, ArrayElement::Null)));

        if !should_filter_nulls {
            continue;
        }

        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                if df.column(&col_name).is_ok() {
                    let filtered = filter_null_rows(df, &col_name)?;
                    data_map.insert(data_key, filtered);
                }
            }
        }
    }

    Ok(())
}

/// Find all (data_key, column_name) pairs for an aesthetic (including family members).
/// Returns tuples of (data source key, column name) for use in transformations.
///
/// Note: Global mappings have already been merged into layer mappings at this point.
pub fn find_columns_for_aesthetic_with_sources(
    layers: &[Layer],
    aesthetic: &str,
    data_map: &HashMap<String, DataFrame>,
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        let layer_key = naming::layer_key(i);

        // Skip if layer doesn't have data
        if !data_map.contains_key(&layer_key) {
            continue;
        }

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                results.push((layer_key.clone(), name.clone()));
            }
        }
    }

    results
}

/// Apply oob transformation to a single numeric column in a DataFrame.
pub fn apply_oob_to_column_numeric(
    df: &DataFrame,
    col_name: &str,
    range_min: f64,
    range_max: f64,
    oob_mode: &str,
) -> Result<DataFrame> {
    use polars::prelude::*;

    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    // Try to cast column to f64 for comparison
    let series = col.as_materialized_series();
    let f64_col = series.cast(&DataType::Float64).map_err(|_| {
        GgsqlError::ValidationError(format!(
            "Cannot apply oob to non-numeric column '{}'",
            col_name
        ))
    })?;

    let f64_ca = f64_col.f64().map_err(|_| {
        GgsqlError::ValidationError(format!(
            "Cannot apply oob to non-numeric column '{}'",
            col_name
        ))
    })?;

    match oob_mode {
        OOB_CENSOR => {
            // Filter out rows where values are outside [range_min, range_max]
            let mask: BooleanChunked = f64_ca
                .into_iter()
                .map(|opt| opt.is_none_or(|v| v >= range_min && v <= range_max))
                .collect();

            let result = df.filter(&mask).map_err(|e| {
                GgsqlError::InternalError(format!("Failed to filter DataFrame: {}", e))
            })?;
            Ok(result)
        }
        OOB_SQUISH => {
            // Clamp values to [range_min, range_max]
            let clamped: Float64Chunked = f64_ca
                .into_iter()
                .map(|opt| opt.map(|v| v.clamp(range_min, range_max)))
                .collect();

            // Replace column with clamped values, maintaining original name
            let clamped_series = clamped.into_series().with_name(col_name.into());

            df.clone()
                .with_column(clamped_series)
                .map(|df| df.clone())
                .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))
        }
        _ => Ok(df.clone()),
    }
}

/// Filter out rows where a column has NULL values.
///
/// Used after OOB transformations to remove rows that were censored to NULL.
pub fn filter_null_rows(df: &DataFrame, col_name: &str) -> Result<DataFrame> {
    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let mask = col.is_not_null();
    df.filter(&mask)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to filter NULL rows: {}", e)))
}

/// Apply oob transformation to a single discrete/categorical column in a DataFrame.
///
/// For discrete scales, censoring sets out-of-range values to null (preserving all rows)
/// rather than filtering out entire rows. This allows other aesthetics to still be visualized.
pub fn apply_oob_to_column_discrete(
    df: &DataFrame,
    col_name: &str,
    allowed_values: &HashSet<String>,
    oob_mode: &str,
) -> Result<DataFrame> {
    use polars::prelude::*;

    // For discrete columns, only censor makes sense (squish is validated out earlier)
    if oob_mode != OOB_CENSOR {
        return Ok(df.clone());
    }

    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let series = col.as_materialized_series();

    // Build new series: keep allowed values, set others to null
    // This preserves all rows (unlike filtering) so other aesthetics can still be visualized
    let new_ca: StringChunked = (0..series.len())
        .map(|i| {
            match series.get(i) {
                Ok(val) => {
                    // Null values are kept as null
                    if val.is_null() {
                        return None;
                    }
                    // Convert value to string and check membership
                    let s = val.to_string();
                    // Remove quotes if present (polars adds quotes around strings)
                    let clean = s.trim_matches('"').to_string();
                    if allowed_values.contains(&clean) {
                        Some(clean)
                    } else {
                        None // CENSOR to null (not filter row!)
                    }
                }
                Err(_) => None,
            }
        })
        .collect();

    // Replace column (keep all rows)
    let new_series = new_ca.into_series().with_name(col_name.into());
    let mut result = df.clone();
    result
        .with_column(new_series)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ArrayElement;
    use crate::Geom;
    use polars::prelude::DataType;

    #[test]
    fn test_get_aesthetic_family() {
        // Test primary aesthetics include all family members
        let x_family = get_aesthetic_family("x");
        assert!(x_family.contains(&"x"));
        assert!(x_family.contains(&"xmin"));
        assert!(x_family.contains(&"xmax"));
        assert!(x_family.contains(&"xend"));
        assert_eq!(x_family.len(), 4);

        let y_family = get_aesthetic_family("y");
        assert!(y_family.contains(&"y"));
        assert!(y_family.contains(&"ymin"));
        assert!(y_family.contains(&"ymax"));
        assert!(y_family.contains(&"yend"));
        assert_eq!(y_family.len(), 4);

        // Test non-family aesthetics return just themselves
        let color_family = get_aesthetic_family("color");
        assert_eq!(color_family, vec!["color"]);

        // Test variant aesthetics return just themselves
        let xmin_family = get_aesthetic_family("xmin");
        assert_eq!(xmin_family, vec!["xmin"]);
    }

    #[test]
    fn test_scale_type_infer() {
        // Test numeric types -> Continuous
        assert_eq!(ScaleType::infer(&DataType::Int32), ScaleType::continuous());
        assert_eq!(ScaleType::infer(&DataType::Int64), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Float64),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::UInt16), ScaleType::continuous());

        // Temporal types now use Continuous scale (with temporal transforms)
        assert_eq!(ScaleType::infer(&DataType::Date), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Datetime(
                polars::prelude::TimeUnit::Microseconds,
                None
            )),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::Time), ScaleType::continuous());

        // Test discrete types
        assert_eq!(ScaleType::infer(&DataType::String), ScaleType::discrete());
        assert_eq!(ScaleType::infer(&DataType::Boolean), ScaleType::discrete());
    }

    #[test]
    fn test_resolve_scales_infers_input_range() {
        use polars::prelude::*;

        // Create a Plot with a scale that needs range inference
        let mut spec = Plot::new();

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("x");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with numeric values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that both scale_type and input_range were inferred
        let scale = &spec.scales[0];
        assert_eq!(scale.scale_type, Some(ScaleType::continuous()));
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        assert_eq!(range.len(), 2);
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 1.0);
                assert_eq!(*max, 10.0);
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_preserves_explicit_input_range() {
        use polars::prelude::*;

        // Create a Plot with a scale that already has a range
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with different values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that explicit range was preserved (not overwritten with [1, 10])
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 0.0); // Original explicit value
                assert_eq!(*max, 100.0); // Original explicit value
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_from_aesthetic_family_input_range() {
        use polars::prelude::*;

        // Create a Plot where "y" scale should get range from ymin and ymax columns
        let mut spec = Plot::new();

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("y");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mappings are in layer
        let layer = Layer::new(Geom::errorbar())
            .with_aesthetic("ymin".to_string(), AestheticValue::standard_column("low"))
            .with_aesthetic("ymax".to_string(), AestheticValue::standard_column("high"));
        spec.layers.push(layer);

        // Create data where ymin/ymax columns have different ranges
        let df = df! {
            "low" => &[5.0f64, 10.0, 15.0],
            "high" => &[20.0f64, 25.0, 30.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that range was inferred from both ymin and ymax columns
        let scale = &spec.scales[0];
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                // min should be 5.0 (from low column), max should be 30.0 (from high column)
                assert_eq!(*min, 5.0);
                assert_eq!(*max, 30.0);
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_partial_input_range_explicit_min_null_max() {
        use polars::prelude::*;

        // Create a Plot with a scale that has [0, null] (explicit min, infer max)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Null]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that range is [0, 10] (explicit min, inferred max)
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 0.0); // Explicit value
                assert_eq!(*max, 10.0); // Inferred from data
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_partial_input_range_null_min_explicit_max() {
        use polars::prelude::*;

        // Create a Plot with a scale that has [null, 100] (infer min, explicit max)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Null, ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that range is [1, 100] (inferred min, explicit max)
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 1.0); // Inferred from data
                assert_eq!(*max, 100.0); // Explicit value
            }
            _ => panic!("Expected Number elements"),
        }
    }
}
