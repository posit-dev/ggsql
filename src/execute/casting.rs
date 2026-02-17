//! Type requirements determination and casting logic.
//!
//! This module handles determining which columns need type casting based on
//! scale requirements and updating type info accordingly.

use crate::plot::scale::coerce_dtypes;
use crate::plot::{CastTargetType, Layer, ParameterValue, Plot, SqlTypeNames};
use crate::{naming, DataSource};
use polars::prelude::{DataType, TimeUnit};
use std::collections::{HashMap, HashSet};

use super::schema::TypeInfo;

/// Describes a column that needs type casting.
#[derive(Debug, Clone)]
pub struct TypeRequirement {
    /// Column name to cast
    pub column: String,
    /// Target type for casting
    pub target_type: CastTargetType,
    /// SQL type name (e.g., "DATE", "DOUBLE", "VARCHAR")
    pub sql_type_name: String,
}

/// Format a literal value as SQL
pub fn literal_to_sql(lit: &ParameterValue) -> String {
    match lit {
        ParameterValue::String(s) => format!("'{}'", s.replace('\'', "''")),
        ParameterValue::Number(n) => n.to_string(),
        ParameterValue::Boolean(b) => {
            if *b {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        ParameterValue::Array(_) | ParameterValue::Null => {
            unreachable!("Grammar prevents arrays and null in literal aesthetic mappings")
        }
    }
}

/// Determine which columns need casting based on scale requirements.
///
/// For each layer, collects columns that need casting to match the scale's
/// target type (determined by type coercion across all columns for that aesthetic).
///
/// # Arguments
///
/// * `spec` - The Plot specification with scales
/// * `layer_type_info` - Type info for each layer
/// * `type_names` - SQL type names for the database backend
///
/// # Returns
///
/// Vec of TypeRequirements for each layer.
pub fn determine_type_requirements(
    spec: &Plot,
    layer_type_info: &[Vec<TypeInfo>],
    type_names: &SqlTypeNames,
) -> Vec<Vec<TypeRequirement>> {
    use crate::plot::scale::TransformKind;

    let mut layer_requirements: Vec<Vec<TypeRequirement>> = Vec::new();

    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let mut requirements: Vec<TypeRequirement> = Vec::new();
        let type_info = &layer_type_info[layer_idx];

        // Build a map of column name to dtype for quick lookup
        let column_dtypes: HashMap<&str, &DataType> = type_info
            .iter()
            .map(|(name, dtype, _)| (name.as_str(), dtype))
            .collect();

        // For each aesthetic mapped in this layer, check if casting is needed
        for (aesthetic, value) in &layer.mappings.aesthetics {
            let col_name = match value.column_name() {
                Some(name) => name,
                None => continue, // Skip literals
            };

            // Skip synthetic columns
            if naming::is_synthetic_column(col_name) {
                continue;
            }

            let col_dtype = match column_dtypes.get(col_name) {
                Some(dtype) => *dtype,
                None => continue, // Column not in schema
            };

            // Find the scale for this aesthetic
            let scale = match spec.scales.iter().find(|s| s.aesthetic == *aesthetic) {
                Some(s) => s,
                None => continue, // No scale for this aesthetic
            };

            // Get the scale type
            let scale_type = match &scale.scale_type {
                Some(st) => st,
                None => continue, // Scale type not yet resolved
            };

            // Collect all dtypes for this aesthetic across all layers
            let all_dtypes: Vec<DataType> = layer_type_info
                .iter()
                .zip(spec.layers.iter())
                .filter_map(|(info, l)| {
                    l.mappings
                        .get(aesthetic)
                        .and_then(|v| v.column_name())
                        .and_then(|name| info.iter().find(|(n, _, _)| n == name))
                        .map(|(_, dtype, _)| dtype.clone())
                })
                .collect();

            // Determine target dtype through coercion
            let target_dtype = match coerce_dtypes(&all_dtypes) {
                Ok(dt) => dt,
                Err(_) => continue, // Skip if coercion fails
            };

            // Check if this specific column needs casting
            if let Some(cast_target) = scale_type.required_cast_type(col_dtype, &target_dtype) {
                if let Some(sql_type) = type_names.for_target(cast_target) {
                    // Don't add duplicate requirements for same column
                    if !requirements.iter().any(|r| r.column == col_name) {
                        requirements.push(TypeRequirement {
                            column: col_name.to_string(),
                            target_type: cast_target,
                            sql_type_name: sql_type.to_string(),
                        });
                    }
                }
            }

            // Check if Integer transform requires casting (float -> integer)
            if let Some(ref transform) = scale.transform {
                if transform.transform_kind() == TransformKind::Integer {
                    // Integer transform: cast non-integer numeric types to integer
                    let needs_int_cast = match col_dtype {
                        DataType::Float32 | DataType::Float64 => true,
                        // Integer types don't need casting
                        DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64 => false,
                        // Other types: no integer casting
                        _ => false,
                    };

                    if needs_int_cast {
                        if let Some(sql_type) = type_names.for_target(CastTargetType::Integer) {
                            // Don't add duplicate requirements for same column
                            if !requirements.iter().any(|r| r.column == col_name) {
                                requirements.push(TypeRequirement {
                                    column: col_name.to_string(),
                                    target_type: CastTargetType::Integer,
                                    sql_type_name: sql_type.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        layer_requirements.push(requirements);
    }

    layer_requirements
}

/// Update type info with post-cast dtypes.
///
/// After determining casting requirements, updates the type info
/// to reflect the target dtypes (so subsequent schema extraction
/// and scale resolution see the correct types).
pub fn update_type_info_for_casting(type_info: &mut [TypeInfo], requirements: &[TypeRequirement]) {
    for req in requirements {
        if let Some(entry) = type_info
            .iter_mut()
            .find(|(name, _, _)| name == &req.column)
        {
            entry.1 = match req.target_type {
                CastTargetType::Number => DataType::Float64,
                CastTargetType::Integer => DataType::Int64,
                CastTargetType::Date => DataType::Date,
                CastTargetType::DateTime => DataType::Datetime(TimeUnit::Microseconds, None),
                CastTargetType::Time => DataType::Time,
                CastTargetType::String => DataType::String,
                CastTargetType::Boolean => DataType::Boolean,
            };
            // Update is_discrete flag based on new type
            entry.2 = matches!(entry.1, DataType::String | DataType::Boolean);
        }
    }
}

/// Determine the data source table name for a layer.
///
/// Returns the table/CTE name to query from:
/// - Layer with explicit source (CTE, table, file) → that source name
/// - Layer using global data → global table name
pub fn determine_layer_source(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
) -> String {
    match &layer.source {
        Some(DataSource::Identifier(name)) => {
            if materialized_ctes.contains(name) {
                naming::cte_table(name)
            } else {
                name.clone()
            }
        }
        Some(DataSource::FilePath(path)) => {
            format!("'{}'", path)
        }
        None => {
            // Layer uses global data - caller must ensure has_global is true
            debug_assert!(has_global, "Layer has no source and no global data");
            naming::global_table()
        }
    }
}
