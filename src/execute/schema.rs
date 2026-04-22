//! Schema extraction, type inference, and min/max range computation.
//!
//! This module provides functions for extracting column types and computing
//! min/max ranges from queries. It uses a split approach:
//! 1. fetch_schema_types() - get dtypes only (before casting)
//! 2. Apply casting to queries
//! 3. complete_schema_ranges() - get min/max from cast queries

use crate::array_util::*;
use crate::plot::{AestheticValue, ArrayElement, ColumnInfo, Layer, ParameterValue, Schema};
use crate::{naming, DataFrame, Result};
use arrow::array::Array;
use arrow::datatypes::{DataType, TimeUnit};

/// Simple type info tuple: (name, dtype, is_discrete)
pub type TypeInfo = (String, DataType, bool);

/// Build SQL query to compute min and max for all columns
///
/// Generates a query that returns two rows:
/// - Row 0: MIN of each column
/// - Row 1: MAX of each column
pub fn build_minmax_query(source_query: &str, column_names: &[&str]) -> String {
    let min_exprs: Vec<String> = column_names
        .iter()
        .map(|name| {
            let q = naming::quote_ident(name);
            format!("MIN({q}) AS {q}")
        })
        .collect();

    let max_exprs: Vec<String> = column_names
        .iter()
        .map(|name| {
            let q = naming::quote_ident(name);
            format!("MAX({q}) AS {q}")
        })
        .collect();

    format!(
        "WITH \"__ggsql_source__\" AS ({}) SELECT {} FROM \"__ggsql_source__\" UNION ALL SELECT {} FROM \"__ggsql_source__\"",
        source_query,
        min_exprs.join(", "),
        max_exprs.join(", ")
    )
}

/// Extract a value from a DataFrame at a given column and row index
///
/// Converts Arrow array values to ArrayElement for storage in ColumnInfo.
pub fn extract_series_value(
    df: &DataFrame,
    column: &str,
    row: usize,
) -> Option<crate::plot::ArrayElement> {
    use crate::plot::ArrayElement;

    let col = df.column(column).ok()?;

    if row >= col.len() {
        return None;
    }

    if col.is_null(row) {
        return None;
    }

    match col.data_type() {
        DataType::Int8 => as_i8(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::Int16 => as_i16(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::Int32 => as_i32(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::Int64 => as_i64(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::UInt8 => as_u8(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::UInt16 => as_u16(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::UInt32 => as_u32(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::UInt64 => as_u64(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::Float32 => as_f32(col).ok().map(|a| ArrayElement::Number(a.value(row) as f64)),
        DataType::Float64 => as_f64(col).ok().map(|a| ArrayElement::Number(a.value(row))),
        DataType::Boolean => as_bool(col).ok().map(|a| ArrayElement::Boolean(a.value(row))),
        DataType::Utf8 => as_str(col)
            .ok()
            .map(|a| ArrayElement::String(a.value(row).to_string())),
        DataType::Date32 => {
            // Return numeric days since epoch (for range computation)
            as_date32(col)
                .ok()
                .map(|a| ArrayElement::Number(a.value(row) as f64))
        }
        DataType::Timestamp(_, _) => {
            // Return numeric microseconds since epoch (for range computation)
            as_timestamp_us(col)
                .ok()
                .map(|a| ArrayElement::Number(a.value(row) as f64))
        }
        DataType::Time64(_) => {
            // Return numeric nanoseconds since midnight (for range computation)
            as_time64_ns(col)
                .ok()
                .map(|a| ArrayElement::Number(a.value(row) as f64))
        }
        _ => None,
    }
}

/// Fetch only column types (no min/max) from a query.
///
/// Uses LIMIT 1 to get schema while minimally reading data.
/// Returns `(name, dtype, is_discrete)` tuples for each column.
///
/// This is the first phase of the split schema extraction approach:
/// 1. fetch_schema_types() - get dtypes only (before casting)
/// 2. Apply casting to queries
/// 3. complete_schema_ranges() - get min/max from cast queries
pub fn fetch_schema_types<F>(query: &str, execute_query: &F) -> Result<Vec<TypeInfo>>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    let schema_query = format!(
        "SELECT * FROM ({}) AS {} LIMIT 1",
        query,
        naming::SCHEMA_ALIAS
    );
    let schema_df = execute_query(&schema_query)?;

    let schema = schema_df.schema();
    let type_info: Vec<TypeInfo> = schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = field.data_type().clone();
            let is_discrete = matches!(dtype, DataType::Utf8 | DataType::Boolean)
                || matches!(dtype, DataType::Dictionary(_, _));
            (field.name().clone(), dtype, is_discrete)
        })
        .collect();

    Ok(type_info)
}

/// Complete schema with min/max ranges from a (possibly cast) query.
///
/// Takes pre-computed type info and extracts min/max values.
/// Called after casting is applied to queries.
pub fn complete_schema_ranges<F>(
    query: &str,
    type_info: &[TypeInfo],
    execute_query: &F,
) -> Result<Schema>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    if type_info.is_empty() {
        return Ok(Vec::new());
    }

    // Build and execute min/max query
    let column_names: Vec<&str> = type_info.iter().map(|(n, _, _)| n.as_str()).collect();
    let minmax_query = build_minmax_query(query, &column_names);
    let range_df = execute_query(&minmax_query)?;

    // Extract min (row 0) and max (row 1) for each column
    let schema = type_info
        .iter()
        .map(|(name, dtype, is_discrete)| {
            let min = extract_series_value(&range_df, name, 0);
            let max = extract_series_value(&range_df, name, 1);
            ColumnInfo {
                name: name.clone(),
                dtype: dtype.clone(),
                is_discrete: *is_discrete,
                min,
                max,
            }
        })
        .collect();

    Ok(schema)
}

/// Convert type info to schema (without min/max).
///
/// Used when we need a Schema but don't have min/max yet.
pub fn type_info_to_schema(type_info: &[TypeInfo]) -> Schema {
    type_info
        .iter()
        .map(|(name, dtype, is_discrete)| ColumnInfo {
            name: name.clone(),
            dtype: dtype.clone(),
            is_discrete: *is_discrete,
            min: None,
            max: None,
        })
        .collect()
}

/// Add type info for literal (constant) mappings to layer type info.
///
/// When a layer has literal mappings like `'blue' AS fill`, we need the type info
/// for these columns in the schema. Instead of re-querying the database, we can
/// derive the types directly from the AST.
///
/// This is called after global mappings are merged and color is split, so all
/// literal mappings are already in place.
pub fn add_literal_columns_to_type_info(layers: &[Layer], layer_type_info: &mut [Vec<TypeInfo>]) {
    for (layer, type_info) in layers.iter().zip(layer_type_info.iter_mut()) {
        for (aesthetic, value) in &layer.mappings.aesthetics {
            if let AestheticValue::Literal(lit) = value {
                let (dtype, is_discrete) = match lit {
                    ParameterValue::String(_) => (DataType::Utf8, true),
                    ParameterValue::Number(_) => (DataType::Float64, false),
                    ParameterValue::Boolean(_) => (DataType::Boolean, true),
                    ParameterValue::Array(arr) => {
                        // Infer dtype from first element (arrays are homogeneous)
                        if let Some(first) = arr.first() {
                            match first {
                                ArrayElement::String(_) => (DataType::Utf8, true),
                                ArrayElement::Number(_) => (DataType::Float64, false),
                                ArrayElement::Boolean(_) => (DataType::Boolean, true),
                                ArrayElement::Date(_) => (DataType::Date32, false),
                                ArrayElement::DateTime(_) => {
                                    (DataType::Timestamp(TimeUnit::Microsecond, None), false)
                                }
                                ArrayElement::Time(_) => (DataType::Time64(TimeUnit::Nanosecond), false),
                                ArrayElement::Null => {
                                    // Null element: default to Float64
                                    (DataType::Float64, false)
                                }
                            }
                        } else {
                            // Empty array: default to Float64
                            (DataType::Float64, false)
                        }
                    }
                    ParameterValue::Null => {
                        // Skip null literals - they don't create columns
                        continue;
                    }
                };
                let col_name = naming::aesthetic_column(aesthetic);

                // Only add if not already present
                if !type_info.iter().any(|(name, _, _)| name == &col_name) {
                    type_info.push((col_name, dtype, is_discrete));
                }
            }
        }
    }
}

/// Build a schema with prefixed aesthetic column names from the original schema.
///
/// For each aesthetic mapped to a column, looks up the original column's type
/// in the schema and adds it with the prefixed aesthetic name (e.g., `__ggsql_aes_x__`).
///
/// This schema is used by stat transforms to look up column types using the
/// prefixed names that appear in the query after `build_layer_select_list`.
pub fn build_aesthetic_schema(layer: &Layer, schema: &Schema) -> Schema {
    let mut aesthetic_schema: Schema = Vec::new();

    for (aesthetic, value) in &layer.mappings.aesthetics {
        let aes_col_name = naming::aesthetic_column(aesthetic);
        match value {
            AestheticValue::Column { name, .. } | AestheticValue::AnnotationColumn { name } => {
                // The schema already has aesthetic-prefixed column names from build_layer_base_query,
                // so we look up by aesthetic name, not the original column name.
                // Fall back to original name for backwards compatibility with older schemas.
                let col_info = schema
                    .iter()
                    .find(|c| c.name == aes_col_name)
                    .or_else(|| schema.iter().find(|c| c.name == *name));

                if let Some(original_col) = col_info {
                    aesthetic_schema.push(ColumnInfo {
                        name: aes_col_name,
                        dtype: original_col.dtype.clone(),
                        is_discrete: original_col.is_discrete,
                        min: original_col.min.clone(),
                        max: original_col.max.clone(),
                    });
                } else {
                    // Column not in schema - add with Unknown type
                    aesthetic_schema.push(ColumnInfo {
                        name: aes_col_name,
                        dtype: DataType::Utf8,
                        is_discrete: false,
                        min: None,
                        max: None,
                    });
                }
            }
            AestheticValue::Literal(lit) => {
                // Literals become columns with appropriate type
                let (dtype, is_discrete) = match lit {
                    ParameterValue::String(_) => (DataType::Utf8, true),
                    ParameterValue::Number(_) => (DataType::Float64, false),
                    ParameterValue::Boolean(_) => (DataType::Boolean, true),
                    ParameterValue::Array(arr) => {
                        // Infer dtype from first element (arrays are homogeneous)
                        if let Some(first) = arr.first() {
                            match first {
                                ArrayElement::String(_) => (DataType::Utf8, true),
                                ArrayElement::Number(_) => (DataType::Float64, false),
                                ArrayElement::Boolean(_) => (DataType::Boolean, true),
                                ArrayElement::Date(_) => (DataType::Date32, false),
                                ArrayElement::DateTime(_) => {
                                    (DataType::Timestamp(TimeUnit::Microsecond, None), false)
                                }
                                ArrayElement::Time(_) => (DataType::Time64(TimeUnit::Nanosecond), false),
                                ArrayElement::Null => {
                                    // Null element: default to Float64
                                    (DataType::Float64, false)
                                }
                            }
                        } else {
                            // Empty array: default to Float64
                            (DataType::Float64, false)
                        }
                    }
                    ParameterValue::Null => {
                        // Null: default to Float64
                        (DataType::Float64, false)
                    }
                };
                aesthetic_schema.push(ColumnInfo {
                    name: aes_col_name,
                    dtype,
                    is_discrete,
                    min: None,
                    max: None,
                });
            }
        }
    }

    // Add facet variables and partition_by columns with their original types
    for col in &layer.partition_by {
        if !aesthetic_schema.iter().any(|c| c.name == *col) {
            if let Some(original_col) = schema.iter().find(|c| c.name == *col) {
                aesthetic_schema.push(original_col.clone());
            }
        }
    }

    aesthetic_schema
}
