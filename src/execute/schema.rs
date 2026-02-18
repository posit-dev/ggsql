//! Schema extraction, type inference, and min/max range computation.
//!
//! This module provides functions for extracting column types and computing
//! min/max ranges from queries. It uses a split approach:
//! 1. fetch_schema_types() - get dtypes only (before casting)
//! 2. Apply casting to queries
//! 3. complete_schema_ranges() - get min/max from cast queries

use crate::plot::{AestheticValue, ColumnInfo, Layer, ParameterValue, Schema};
use crate::{naming, DataFrame, Result};
use polars::prelude::DataType;

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
        .map(|name| format!("MIN(\"{}\") AS \"{}\"", name, name))
        .collect();

    let max_exprs: Vec<String> = column_names
        .iter()
        .map(|name| format!("MAX(\"{}\") AS \"{}\"", name, name))
        .collect();

    format!(
        "WITH __ggsql_source__ AS ({}) SELECT {} FROM __ggsql_source__ UNION ALL SELECT {} FROM __ggsql_source__",
        source_query,
        min_exprs.join(", "),
        max_exprs.join(", ")
    )
}

/// Extract a value from a DataFrame at a given column and row index
///
/// Converts Polars values to ArrayElement for storage in ColumnInfo.
pub fn extract_series_value(
    df: &DataFrame,
    column: &str,
    row: usize,
) -> Option<crate::plot::ArrayElement> {
    use crate::plot::ArrayElement;

    let col = df.column(column).ok()?;
    let series = col.as_materialized_series();

    if row >= series.len() {
        return None;
    }

    match series.dtype() {
        DataType::Int8 => series
            .i8()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int16 => series
            .i16()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int32 => series
            .i32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int64 => series
            .i64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt8 => series
            .u8()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt16 => series
            .u16()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt32 => series
            .u32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt64 => series
            .u64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Float32 => series
            .f32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Float64 => series
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(ArrayElement::Number),
        DataType::Boolean => series
            .bool()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(ArrayElement::Boolean),
        DataType::String => series
            .str()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|s| ArrayElement::String(s.to_string())),
        DataType::Date => {
            // Return numeric days since epoch (for range computation)
            series
                .date()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|days| ArrayElement::Number(days as f64))
        }
        DataType::Datetime(_, _) => {
            // Return numeric microseconds since epoch (for range computation)
            series
                .datetime()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|us| ArrayElement::Number(us as f64))
        }
        DataType::Time => {
            // Return numeric nanoseconds since midnight (for range computation)
            series
                .time()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|ns| ArrayElement::Number(ns as f64))
        }
        _ => None,
    }
}

/// Fetch only column types (no min/max) from a query.
///
/// Uses LIMIT 0 to get schema without reading data.
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
        "SELECT * FROM ({}) AS {} LIMIT 0",
        query,
        naming::SCHEMA_ALIAS
    );
    let schema_df = execute_query(&schema_query)?;

    let type_info: Vec<TypeInfo> = schema_df
        .get_columns()
        .iter()
        .map(|col| {
            let dtype = col.dtype().clone();
            let is_discrete =
                matches!(dtype, DataType::String | DataType::Boolean) || dtype.is_categorical();
            (col.name().to_string(), is_discrete, dtype)
        })
        .map(|(name, is_discrete, dtype)| (name, dtype, is_discrete))
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
                let dtype = match lit {
                    ParameterValue::String(_) => DataType::String,
                    ParameterValue::Number(_) => DataType::Float64,
                    ParameterValue::Boolean(_) => DataType::Boolean,
                    ParameterValue::Array(_) | ParameterValue::Null => unreachable!(
                        "Grammar prevents arrays and null in literal aesthetic mappings"
                    ),
                };
                let is_discrete =
                    matches!(lit, ParameterValue::String(_) | ParameterValue::Boolean(_));
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
            AestheticValue::Column { name, .. } => {
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
                        dtype: DataType::Unknown(Default::default()),
                        is_discrete: false,
                        min: None,
                        max: None,
                    });
                }
            }
            AestheticValue::Literal(lit) => {
                // Literals become columns with appropriate type
                let dtype = match lit {
                    ParameterValue::String(_) => DataType::String,
                    ParameterValue::Number(_) => DataType::Float64,
                    ParameterValue::Boolean(_) => DataType::Boolean,
                    ParameterValue::Array(_) | ParameterValue::Null => unreachable!(
                        "Grammar prevents arrays and null in literal aesthetic mappings"
                    ),
                };
                aesthetic_schema.push(ColumnInfo {
                    name: aes_col_name,
                    dtype,
                    is_discrete: matches!(
                        lit,
                        ParameterValue::String(_) | ParameterValue::Boolean(_)
                    ),
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
