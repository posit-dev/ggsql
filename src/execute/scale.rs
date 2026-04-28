//! Scale creation, resolution, type coercion, and OOB handling.
//!
//! This module handles creating default scales for aesthetics, resolving
//! scale properties from data, type coercion based on scale requirements,
//! and out-of-bounds (OOB) handling.

use crate::naming;
use crate::plot::aesthetic::AestheticContext;
use crate::plot::scale::{
    default_oob, gets_default_scale, infer_scale_target_type, infer_transform_from_input_range,
    is_facet_aesthetic, transform::Transform, OOB_CENSOR, OOB_KEEP, OOB_SQUISH,
};
use crate::plot::{
    AestheticValue, ArrayElement, ArrayElementType, ColumnInfo, Layer, ParameterValue, Plot, Scale,
    ScaleType, ScaleTypeKind, Schema,
};
use crate::{DataFrame, GgsqlError, Result};
use arrow::array::ArrayRef;
use std::collections::{HashMap, HashSet};

use super::schema::TypeInfo;

/// Create Scale objects for aesthetics that don't have explicit SCALE clauses.
///
/// For aesthetics with meaningful scale behavior, creates a minimal scale
/// (type will be inferred later by resolve_scales from column dtype).
/// For identity aesthetics (text, label, group, etc.), creates an Identity scale.
pub fn create_missing_scales(spec: &mut Plot) {
    let aesthetic_ctx = spec.get_aesthetic_context();
    let mut used_aesthetics: HashSet<String> = HashSet::new();

    // Collect from layer mappings and remappings
    // (global mappings have already been merged into layers at this point)
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = aesthetic_ctx
                .primary_internal_position(aesthetic)
                .unwrap_or(aesthetic);
            used_aesthetics.insert(primary.to_string());
        }
        for aesthetic in layer.remappings.aesthetics.keys() {
            let primary = aesthetic_ctx
                .primary_internal_position(aesthetic)
                .unwrap_or(aesthetic);
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
///
/// Also infers scale types from data for newly created scales. This must happen
/// before position adjustments so dodge/stack can correctly identify continuous
/// vs discrete axes (e.g., stat-generated count columns).
pub fn create_missing_scales_post_stat(
    spec: &mut Plot,
    data_map: &HashMap<String, DataFrame>,
) -> Result<()> {
    let aesthetic_ctx = spec.get_aesthetic_context();
    let mut current_aesthetics: HashSet<String> = HashSet::new();

    // Collect all aesthetics currently in layer mappings
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = aesthetic_ctx
                .primary_internal_position(aesthetic)
                .unwrap_or(aesthetic);
            current_aesthetics.insert(primary.to_string());
        }
    }

    // Find aesthetics that don't have scales yet and create them
    let existing_scales: HashSet<String> =
        spec.scales.iter().map(|s| s.aesthetic.clone()).collect();

    for aesthetic in current_aesthetics {
        if !existing_scales.contains(&aesthetic) {
            let mut scale = Scale::new(&aesthetic);
            if !gets_default_scale(&aesthetic) {
                scale.scale_type = Some(ScaleType::identity());
            }
            spec.scales.push(scale);
        }
    }

    // Infer types for all scales that don't have scale_type set
    // This handles both newly created scales and user-specified scales like
    // `SCALE y SETTING expand` where the type wasn't explicitly specified.
    // Position adjustments (stack, dodge) need scale types to determine axes.
    for scale in &mut spec.scales {
        if scale.scale_type.is_none() && gets_default_scale(&scale.aesthetic) {
            let column_refs = find_columns_for_aesthetic(
                &spec.layers,
                &scale.aesthetic,
                data_map,
                &aesthetic_ctx,
            );
            if !column_refs.is_empty() {
                scale.scale_type = Some(ScaleType::infer_for_aesthetic(
                    column_refs[0].data_type(),
                    &scale.aesthetic,
                ));
            }
        }
    }

    Ok(())
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
    let aesthetic_ctx = spec.get_aesthetic_context();

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
        let column_sources = find_columns_for_aesthetic_with_sources(
            &spec.layers,
            &scale.aesthetic,
            data_map,
            &aesthetic_ctx,
        );

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
    use crate::array_util::{as_f64, cast_array, new_f64_array};
    use arrow::array::Array;
    use arrow::datatypes::DataType;

    let column = df.column(col_name)?;

    // Cast to f64 for binning
    let float_col = cast_array(column, &DataType::Float64).map_err(|e| {
        GgsqlError::InternalError(format!("Cannot bin column '{}': {}", col_name, e))
    })?;

    let f64_arr = as_f64(&float_col)?;

    // Apply binning: replace values with bin centers
    let num_bins = break_values.len() - 1;
    let binned: Vec<Option<f64>> = (0..f64_arr.len())
        .map(|idx| {
            if f64_arr.is_null(idx) {
                return None;
            }
            let val = f64_arr.value(idx);
            for i in 0..num_bins {
                let lower = break_values[i];
                let upper = break_values[i + 1];
                let is_last = i == num_bins - 1;

                let in_bin = if closed_left {
                    if is_last {
                        val >= lower && val <= upper
                    } else {
                        val >= lower && val < upper
                    }
                } else {
                    if i == 0 {
                        val >= lower && val <= upper
                    } else {
                        val > lower && val <= upper
                    }
                };

                if in_bin {
                    return Some((lower + upper) / 2.0);
                }
            }
            Some(f64::NAN) // Outside all bins
        })
        .collect();

    let binned_array = new_f64_array(binned);

    // Replace column in DataFrame
    df.with_column(col_name, binned_array)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))
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

    let aesthetic_ctx = spec.get_aesthetic_context();

    for scale in &mut spec.scales {
        // Skip scales that already have explicit types (user specified)
        if let Some(scale_type) = &scale.scale_type {
            // Validate facet aesthetics cannot use Continuous scales
            if is_facet_aesthetic(&scale.aesthetic)
                && scale_type.scale_type_kind() == ScaleTypeKind::Continuous
            {
                return Err(GgsqlError::ValidationError(format!(
                    "SCALE {}: facet variables require Discrete or Binned scales, got Continuous. \
                     Use SCALE BINNED {} to bin continuous data.",
                    scale.aesthetic, scale.aesthetic
                )));
            }

            // Collect all dtypes for validation and transform inference
            let all_dtypes = collect_dtypes_for_aesthetic(
                &spec.layers,
                &scale.aesthetic,
                layer_type_info,
                &aesthetic_ctx,
            );

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
        let all_dtypes = collect_dtypes_for_aesthetic(
            &spec.layers,
            &scale.aesthetic,
            layer_type_info,
            &aesthetic_ctx,
        );

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
                    // Identity: fall back to dtype inference (considers aesthetic)
                    TransformKind::Identity => {
                        ScaleType::infer_for_aesthetic(&common_dtype, &scale.aesthetic)
                    }
                }
            } else {
                ScaleType::infer_for_aesthetic(&common_dtype, &scale.aesthetic)
            }
        } else {
            ScaleType::infer_for_aesthetic(&common_dtype, &scale.aesthetic)
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
    aesthetic_ctx: &AestheticContext,
) -> Vec<arrow::datatypes::DataType> {
    let mut dtypes = Vec::new();
    let aesthetics_to_check = aesthetic_ctx
        .internal_position_family(aesthetic)
        .map(|f| f.to_vec())
        .unwrap_or_else(|| vec![aesthetic.to_string()]);

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

    let aesthetic_ctx = spec.get_aesthetic_context();

    for scale in &mut spec.scales {
        // Only pre-resolve Binned scales
        let scale_type = match &scale.scale_type {
            Some(st) if st.scale_type_kind() == ScaleTypeKind::Binned => st.clone(),
            _ => continue,
        };

        // Find all ColumnInfos for this aesthetic from schemas
        let column_infos = find_schema_columns_for_aesthetic(
            &spec.layers,
            &scale.aesthetic,
            layer_schemas,
            &aesthetic_ctx,
        );

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
    aesthetic_ctx: &AestheticContext,
) -> Vec<ColumnInfo> {
    let mut infos = Vec::new();
    let aesthetics_to_check = aesthetic_ctx
        .internal_position_family(aesthetic)
        .map(|f| f.to_vec())
        .unwrap_or_else(|| vec![aesthetic.to_string()]);

    // Check each layer's mapping (global mappings already merged)
    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx >= layer_schemas.len() {
            continue;
        }
        let schema = &layer_schemas[layer_idx];

        for aes_name in &aesthetics_to_check {
            if let Some(value) = layer.mappings.get(aes_name) {
                match value {
                    AestheticValue::Column { name, .. }
                    | AestheticValue::AnnotationColumn { name } => {
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
    use arrow::datatypes::DataType;

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
            dtype: DataType::Utf8,
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
    use crate::array_util::*;
    use arrow::array::Array;
    use arrow::datatypes::{DataType, TimeUnit};

    let column = df.column(column_name)?;
    let dtype = column.data_type();

    // Check if already the target type
    let already_target_type = matches!(
        (dtype, target_type),
        (DataType::Boolean, ArrayElementType::Boolean)
            | (
                DataType::Float64 | DataType::Int64 | DataType::Int32 | DataType::Float32,
                ArrayElementType::Number,
            )
            | (DataType::Date32, ArrayElementType::Date)
            | (DataType::Timestamp(_, _), ArrayElementType::DateTime)
            | (DataType::Time64(_), ArrayElementType::Time)
            | (DataType::Utf8, ArrayElementType::String)
    );

    if already_target_type {
        return Ok(df.clone());
    }

    // Coerce based on target type
    let new_array: arrow::array::ArrayRef = match target_type {
        ArrayElementType::Boolean => match dtype {
            DataType::Utf8 => {
                let str_arr = as_str(column)?;
                let bool_vec: Vec<Option<bool>> = (0..str_arr.len())
                    .enumerate()
                    .map(|(idx, i)| {
                        if str_arr.is_null(i) {
                            Ok(None)
                        } else {
                            match str_arr.value(i).to_lowercase().as_str() {
                                "true" | "yes" | "1" => Ok(Some(true)),
                                "false" | "no" | "0" => Ok(Some(false)),
                                s => Err(GgsqlError::ValidationError(format!(
                                    "Column '{}' row {}: Cannot coerce string '{}' to boolean",
                                    column_name, idx, s
                                ))),
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                new_bool_array(bool_vec)
            }
            DataType::Int64 | DataType::Int32 | DataType::Float64 | DataType::Float32 => {
                let f64_col = cast_array(column, &DataType::Float64)?;
                let f64_arr = as_f64(&f64_col)?;
                let bool_vec: Vec<Option<bool>> = (0..f64_arr.len())
                    .map(|i| {
                        if f64_arr.is_null(i) {
                            None
                        } else {
                            Some(f64_arr.value(i) != 0.0)
                        }
                    })
                    .collect();
                new_bool_array(bool_vec)
            }
            _ => {
                return Err(GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' of type {:?} to boolean",
                    column_name, dtype
                )));
            }
        },

        ArrayElementType::Number => cast_array(column, &DataType::Float64).map_err(|e| {
            GgsqlError::ValidationError(format!(
                "Cannot coerce column '{}' to number: {}",
                column_name, e
            ))
        })?,

        ArrayElementType::Date => match dtype {
            DataType::Utf8 => {
                let str_arr = as_str(column)?;
                let date_vec: Vec<Option<i32>> = (0..str_arr.len())
                        .enumerate()
                        .map(|(idx, i)| {
                            if str_arr.is_null(i) {
                                Ok(None)
                            } else {
                                let s = str_arr.value(i);
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
                        })
                        .collect::<Result<Vec<_>>>()?;
                let i32_arr = new_i32_array(date_vec);
                cast_array(&i32_arr, &DataType::Date32)?
            }
            _ => {
                return Err(GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' of type {:?} to date",
                    column_name, dtype
                )));
            }
        },

        ArrayElementType::DateTime => match dtype {
            DataType::Utf8 => {
                let str_arr = as_str(column)?;
                let dt_vec: Vec<Option<i64>> = (0..str_arr.len())
                    .enumerate()
                    .map(|(idx, i)| {
                        if str_arr.is_null(i) {
                            Ok(None)
                        } else {
                            let s = str_arr.value(i);
                            ArrayElement::from_datetime_string(s)
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
                                .map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let i64_arr = new_i64_array(dt_vec);
                cast_array(&i64_arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?
            }
            _ => {
                return Err(GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' of type {:?} to datetime",
                    column_name, dtype
                )));
            }
        },

        ArrayElementType::Time => match dtype {
            DataType::Utf8 => {
                let str_arr = as_str(column)?;
                let time_vec: Vec<Option<i64>> = (0..str_arr.len())
                        .enumerate()
                        .map(|(idx, i)| {
                            if str_arr.is_null(i) {
                                Ok(None)
                            } else {
                                let s = str_arr.value(i);
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
                        })
                        .collect::<Result<Vec<_>>>()?;
                let i64_arr = new_i64_array(time_vec);
                cast_array(&i64_arr, &DataType::Time64(TimeUnit::Nanosecond))?
            }
            _ => {
                return Err(GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' of type {:?} to time",
                    column_name, dtype
                )));
            }
        },

        ArrayElementType::String => cast_array(column, &DataType::Utf8).map_err(|e| {
            GgsqlError::ValidationError(format!(
                "Cannot coerce column '{}' to string: {}",
                column_name, e
            ))
        })?,
    };

    // Replace the column in the DataFrame
    df.with_column(column_name, new_array)
        .map_err(|e| GgsqlError::ValidationError(format!("Failed to replace column: {}", e)))
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
    aesthetic_ctx: &AestheticContext,
) -> Result<()> {
    let aesthetics_to_check = aesthetic_ctx
        .internal_position_family(aesthetic)
        .map(|f| f.to_vec())
        .unwrap_or_else(|| vec![aesthetic.to_string()]);

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
    use crate::plot::projection::CoordKind;
    use crate::plot::scale::ScaleDataContext;

    let aesthetic_ctx = spec.get_aesthetic_context();

    // Determine if polar is a full circle (for zero expansion on theta)
    // A polar coord is "full circle" when end is None or equals start
    let (is_polar, polar_is_full_circle) = spec
        .project
        .as_ref()
        .map(|p| {
            let is_polar = p.coord.coord_kind() == CoordKind::Polar;
            if !is_polar {
                return (false, false);
            }
            // Check if it's a full circle: end is None or equals start
            let start = match p.properties.get("start") {
                Some(ParameterValue::Number(n)) => *n,
                _ => 0.0,
            };
            let end = match p.properties.get("end") {
                Some(ParameterValue::Number(n)) => Some(*n),
                _ => None,
            };
            let is_full_circle = end.is_none() || end == Some(start);
            (true, is_full_circle)
        })
        .unwrap_or((false, false));

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
            coerce_aesthetic_columns(
                &spec.layers,
                data_map,
                &aesthetic,
                target_type,
                &aesthetic_ctx,
            )?;
        }

        // Find column references for this aesthetic (including family members)
        // NOTE: Must be called AFTER coercion so column types are correct
        let column_refs =
            find_columns_for_aesthetic(&spec.layers, &aesthetic, data_map, &aesthetic_ctx);

        if column_refs.is_empty() {
            continue;
        }

        // Infer scale_type if not already set (fallback - usually already inferred
        // by create_missing_scales_post_stat() which runs before position adjustments)
        if spec.scales[idx].scale_type.is_none() {
            spec.scales[idx].scale_type = Some(ScaleType::infer_for_aesthetic(
                column_refs[0].data_type(),
                &aesthetic,
            ));
        }

        // Clone scale_type (cheap Arc clone) to avoid borrow conflict with mutations
        let scale_type = spec.scales[idx].scale_type.clone();
        if let Some(st) = scale_type {
            // Determine if this scale uses discrete input range (unique values vs min/max)
            let use_discrete_range = st.uses_discrete_input_range();

            // Build context from actual data columns
            let mut context = ScaleDataContext::from_columns(&column_refs, use_discrete_range);

            // For polar full-circle theta (pos2), use zero expansion
            if is_polar && polar_is_full_circle && aesthetic == "pos2" {
                context.default_expand = Some((0.0, 0.0));
            }

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
    aesthetic_ctx: &AestheticContext,
) -> Vec<&'a ArrayRef> {
    let mut column_refs = Vec::new();
    let aesthetics_to_check = aesthetic_ctx
        .internal_position_family(aesthetic)
        .map(|f| f.to_vec())
        .unwrap_or_else(|| vec![aesthetic.to_string()]);

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        if let Some(df) = data_map.get(&naming::layer_key(i)) {
            for aes_name in &aesthetics_to_check {
                if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                    // Regular columns (data and position annotations) participate in scale training
                    if let Ok(column) = df.column(name) {
                        column_refs.push(column);
                    }
                }
                // AnnotationColumn and Literal don't participate in scale training
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
    let aesthetic_ctx = spec.get_aesthetic_context();

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
        let column_sources = find_columns_for_aesthetic_with_sources(
            &spec.layers,
            &scale.aesthetic,
            data_map,
            &aesthetic_ctx,
        );

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

        let column_sources = find_columns_for_aesthetic_with_sources(
            &spec.layers,
            &scale.aesthetic,
            data_map,
            &aesthetic_ctx,
        );

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
    aesthetic_ctx: &AestheticContext,
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let aesthetics_to_check = aesthetic_ctx
        .internal_position_family(aesthetic)
        .map(|f| f.to_vec())
        .unwrap_or_else(|| vec![aesthetic.to_string()]);

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
    use crate::array_util::*;
    use arrow::array::Array;
    use arrow::datatypes::DataType;

    let col = df.column(col_name)?;

    // Try to cast column to f64 for comparison
    let f64_col = cast_array(col, &DataType::Float64).map_err(|_| {
        GgsqlError::ValidationError(format!(
            "Cannot apply oob to non-numeric column '{}'",
            col_name
        ))
    })?;

    let f64_arr = as_f64(&f64_col)?;

    match oob_mode {
        OOB_CENSOR => {
            // Filter out rows where values are outside [range_min, range_max]
            // Build a boolean mask
            let mask_values: Vec<bool> = (0..f64_arr.len())
                .map(|i| {
                    if f64_arr.is_null(i) {
                        true // Keep nulls
                    } else {
                        let v = f64_arr.value(i);
                        v >= range_min && v <= range_max
                    }
                })
                .collect();

            // Filter all columns using the mask
            let mask = arrow::array::BooleanArray::from(mask_values);
            let mut new_columns = Vec::new();
            let schema = df.schema();
            for (i, field) in schema.fields().iter().enumerate() {
                let col_arr = df.get_columns()[i].clone();
                let filtered = arrow::compute::filter(&col_arr, &mask)
                    .map_err(|e| GgsqlError::InternalError(format!("Failed to filter: {}", e)))?;
                new_columns.push((field.name().as_str(), filtered));
            }
            DataFrame::new(new_columns)
        }
        OOB_SQUISH => {
            // Clamp values to [range_min, range_max]
            let clamped: Vec<Option<f64>> = (0..f64_arr.len())
                .map(|i| {
                    if f64_arr.is_null(i) {
                        None
                    } else {
                        Some(f64_arr.value(i).clamp(range_min, range_max))
                    }
                })
                .collect();

            let clamped_array = new_f64_array(clamped);

            // Restore temporal type if original column was temporal
            let original_dtype = col.data_type().clone();
            let restored_array = match &original_dtype {
                DataType::Date32 | DataType::Timestamp(_, _) | DataType::Time64(_) => {
                    cast_array(&clamped_array, &original_dtype).map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to restore temporal type for '{}': {}",
                            col_name, e
                        ))
                    })?
                }
                _ => clamped_array,
            };

            df.with_column(col_name, restored_array)
                .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))
        }
        _ => Ok(df.clone()),
    }
}

/// Filter out rows where a column has NULL values.
///
/// Used after OOB transformations to remove rows that were censored to NULL.
pub fn filter_null_rows(df: &DataFrame, col_name: &str) -> Result<DataFrame> {
    use arrow::array::Array;

    let col = df.column(col_name)?;

    // Build boolean mask: true where NOT null
    let mask_values: Vec<bool> = (0..col.len()).map(|i| !col.is_null(i)).collect();
    let mask = arrow::array::BooleanArray::from(mask_values);

    let mut new_columns = Vec::new();
    let schema = df.schema();
    for (i, field) in schema.fields().iter().enumerate() {
        let col_arr = df.get_columns()[i].clone();
        let filtered = arrow::compute::filter(&col_arr, &mask)
            .map_err(|e| GgsqlError::InternalError(format!("Failed to filter NULL rows: {}", e)))?;
        new_columns.push((field.name().as_str(), filtered));
    }
    DataFrame::new(new_columns)
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
    use crate::array_util::*;
    use arrow::array::Array;

    // For discrete columns, only censor makes sense (squish is validated out earlier)
    if oob_mode != OOB_CENSOR {
        return Ok(df.clone());
    }

    let col = df.column(col_name)?;

    // Build new string array: keep allowed values, set others to null
    let new_values: Vec<Option<String>> = (0..col.len())
        .map(|i| {
            if col.is_null(i) {
                None
            } else {
                let s = value_to_string(col, i);
                if allowed_values.contains(&s) {
                    Some(s)
                } else {
                    None // CENSOR to null (not filter row!)
                }
            }
        })
        .collect();

    let refs: Vec<Option<&str>> = new_values.iter().map(|o| o.as_deref()).collect();
    let new_array = new_str_array(refs);

    df.with_column(col_name, new_array)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ArrayElement;
    use crate::Geom;
    use arrow::datatypes::DataType;

    #[test]
    fn test_aesthetic_context_internal_family() {
        // Test using AestheticContext for internal family lookups
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Test internal primary aesthetics include all family members
        let pos1_family = ctx.internal_position_family("pos1").unwrap();
        assert!(pos1_family.iter().any(|s| s == "pos1"));
        assert!(pos1_family.iter().any(|s| s == "pos1min"));
        assert!(pos1_family.iter().any(|s| s == "pos1max"));
        assert!(pos1_family.iter().any(|s| s == "pos1end"));
        assert_eq!(pos1_family.len(), 4); // pos1, pos1min, pos1max, pos1end

        let pos2_family = ctx.internal_position_family("pos2").unwrap();
        assert!(pos2_family.iter().any(|s| s == "pos2"));
        assert!(pos2_family.iter().any(|s| s == "pos2min"));
        assert!(pos2_family.iter().any(|s| s == "pos2max"));
        assert!(pos2_family.iter().any(|s| s == "pos2end"));
        assert_eq!(pos2_family.len(), 4); // pos2, pos2min, pos2max, pos2end

        // Test material aesthetics don't have internal family
        assert!(ctx.internal_position_family("color").is_none());

        // Test internal variant aesthetics don't have internal family
        assert!(ctx.internal_position_family("pos1min").is_none());
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
        assert_eq!(ScaleType::infer(&DataType::Date32), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                None
            )),
            ScaleType::continuous()
        );
        assert_eq!(
            ScaleType::infer(&DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond)),
            ScaleType::continuous()
        );

        // Test discrete types
        assert_eq!(ScaleType::infer(&DataType::Utf8), ScaleType::discrete());
        assert_eq!(ScaleType::infer(&DataType::Boolean), ScaleType::discrete());
    }

    #[test]
    fn test_resolve_scales_infers_input_range() {
        use crate::df;

        // Create a Plot with a scale that needs range inference
        let mut spec = Plot::new();

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("pos1");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("pos1".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with numeric values
        let df = df! {
            "value" => vec![1.0f64, 5.0, 10.0]
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
        use crate::df;

        // Create a Plot with a scale that already has a range
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("pos1");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("pos1".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with different values
        let df = df! {
            "value" => vec![1.0f64, 5.0, 10.0]
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
        use crate::df;

        // Create a Plot where "pos2" scale should get range from pos2min and pos2max columns
        let mut spec = Plot::new();

        let scale = crate::plot::Scale::new("pos2");
        spec.scales.push(scale);
        // Simulate post-transformation state: mappings use internal names
        let layer = Layer::new(Geom::range())
            .with_aesthetic(
                "pos2min".to_string(),
                AestheticValue::standard_column("low"),
            )
            .with_aesthetic(
                "pos2max".to_string(),
                AestheticValue::standard_column("high"),
            );
        spec.layers.push(layer);

        // Create data where pos2min/pos2max columns have different ranges
        let df = df! {
            "low" => vec![5.0f64, 10.0, 15.0],
            "high" => vec![20.0f64, 25.0, 30.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that range was inferred from both pos2min and pos2max columns
        let scale = &spec.scales[0];
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                // Range should cover at least 5.0 to 30.0 (from low and high columns)
                // With default expansion, the actual range may be slightly wider
                assert!(*min <= 5.0, "min should be at most 5.0, got {}", min);
                assert!(*max >= 30.0, "max should be at least 30.0, got {}", max);
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_partial_input_range_explicit_min_null_max() {
        use crate::df;

        // Create a Plot with a scale that has [0, null] (explicit min, infer max)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("pos1");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Null]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("pos1".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => vec![1.0f64, 5.0, 10.0]
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
        use crate::df;

        // Create a Plot with a scale that has [null, 100] (infer min, explicit max)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("pos1");
        scale.input_range = Some(vec![ArrayElement::Null, ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("pos1".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => vec![1.0f64, 5.0, 10.0]
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

    #[test]
    fn test_resolve_scales_polar_theta_no_expansion() {
        use crate::df;
        use crate::plot::projection::{Coord, Projection};

        // Create a Plot with a polar projection
        let mut spec = Plot::new();
        let coord = Coord::polar();
        let aesthetics = coord
            .position_aesthetic_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        spec.project = Some(Projection {
            coord,
            aesthetics,
            properties: std::collections::HashMap::new(),
        });

        // Create scale for pos2 (theta in polar) without explicit expand
        let scale = crate::plot::Scale::new("pos2");
        spec.scales.push(scale);

        // Add a layer with pos2 mapping
        let layer = Layer::new(Geom::bar())
            .with_aesthetic("pos2".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with numeric values
        let df = df! {
            "value" => vec![10.0f64, 20.0, 30.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Verify projection is set correctly
        assert!(spec.project.is_some(), "project should be set");
        let coord_kind = spec.project.as_ref().map(|p| p.coord.coord_kind());
        assert_eq!(
            coord_kind,
            Some(crate::plot::CoordKind::Polar),
            "coord_kind should be Polar"
        );

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

        // Check that no expansion was applied for polar theta
        // Without expansion, range should be exactly [10.0, 30.0]
        let scale = &spec.scales[0];
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        assert_eq!(range.len(), 2);
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 10.0, "min should be 10.0 (no expansion)");
                assert_eq!(*max, 30.0, "max should be 30.0 (no expansion)");
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_apply_oob_censor_date32() {
        // Regression: Arrow can't cast Date32 directly to Float64.
        // apply_oob_to_column_numeric must route through Int32 first.
        use arrow::array::{ArrayRef, Date32Array};
        use std::sync::Arc;

        // Days since epoch: 2024-01-01 = 19723, 2024-06-01 = 19875, 2024-12-01 = 20058
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19723, 19875, 20058]));
        let df = DataFrame::new(vec![("date", dates)]).unwrap();

        // Censor to [2024-03-01, 2024-09-01] ≈ [19783, 19967] → keeps only row 1
        let result = apply_oob_to_column_numeric(&df, "date", 19783.0, 19967.0, OOB_CENSOR)
            .expect("oob censor should handle Date32");
        assert_eq!(result.height(), 1);
    }

    #[test]
    fn test_apply_oob_squish_date32_restores_temporal_type() {
        use arrow::array::{ArrayRef, Date32Array};
        use std::sync::Arc;

        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19000, 19875, 21000]));
        let df = DataFrame::new(vec![("date", dates)]).unwrap();

        let result = apply_oob_to_column_numeric(&df, "date", 19723.0, 20089.0, OOB_SQUISH)
            .expect("oob squish should handle Date32");
        assert_eq!(
            result.column("date").unwrap().data_type(),
            &DataType::Date32
        );
    }
}
