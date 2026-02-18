//! DataFrame to JSON conversion utilities for Vega-Lite writer
//!
//! This module handles converting Polars DataFrames to Vega-Lite JSON data values,
//! including temporal type handling and binned data transformations.

use crate::plot::scale::ScaleTypeKind;
// ArrayElement is used for temporal parsing
#[allow(unused_imports)]
use crate::plot::ArrayElement;
use crate::plot::ParameterValue;
use crate::{
    is_primary_positional, naming, primary_aesthetic, AestheticValue, DataFrame, GgsqlError, Plot,
    Result,
};
use polars::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Temporal type for binned date/datetime/time columns
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TemporalType {
    Date,
    DateTime,
    Time,
}

/// Convert Polars DataFrame to Vega-Lite data values (array of objects)
pub(super) fn dataframe_to_values(df: &DataFrame) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    let height = df.height();
    let column_names = df.get_column_names();

    for row_idx in 0..height {
        let mut row_obj = Map::new();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            let column = df.get_columns().get(col_idx).ok_or_else(|| {
                GgsqlError::WriterError(format!("Failed to get column {}", col_name))
            })?;

            // Get value from series and convert to JSON Value
            let value = series_value_at(column.as_materialized_series(), row_idx)?;
            row_obj.insert(col_name.to_string(), value);
        }

        values.push(Value::Object(row_obj));
    }

    Ok(values)
}

/// Get a single value from a series at a given index as JSON Value
pub(super) fn series_value_at(series: &Series, idx: usize) -> Result<Value> {
    use DataType::*;

    match series.dtype() {
        Int8 => {
            let ca = series
                .i8()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to i8: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Int16 => {
            let ca = series
                .i16()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to i16: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Int32 => {
            let ca = series
                .i32()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to i32: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Int64 => {
            let ca = series
                .i64()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to i64: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Float32 => {
            let ca = series
                .f32()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to f32: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Float64 => {
            let ca = series
                .f64()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to f64: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        Boolean => {
            let ca = series
                .bool()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to bool: {}", e)))?;
            Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
        }
        String => {
            let ca = series
                .str()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to string: {}", e)))?;
            // Try to parse as number if it looks numeric
            if let Some(val) = ca.get(idx) {
                if let Ok(num) = val.parse::<f64>() {
                    Ok(json!(num))
                } else {
                    Ok(json!(val))
                }
            } else {
                Ok(Value::Null)
            }
        }
        Date => {
            // Convert days since epoch to ISO date string: "YYYY-MM-DD"
            let ca = series
                .date()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to date: {}", e)))?;
            if let Some(days) = ca.phys.get(idx) {
                let unix_epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                let date = unix_epoch + chrono::Duration::days(days as i64);
                Ok(json!(date.format("%Y-%m-%d").to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        Datetime(time_unit, _) => {
            // Convert timestamp to ISO datetime: "YYYY-MM-DDTHH:MM:SS.sssZ"
            let ca = series.datetime().map_err(|e| {
                GgsqlError::WriterError(format!("Failed to cast to datetime: {}", e))
            })?;
            if let Some(timestamp) = ca.phys.get(idx) {
                // Convert to microseconds based on time unit
                let micros = match time_unit {
                    TimeUnit::Microseconds => timestamp,
                    TimeUnit::Milliseconds => timestamp * 1_000,
                    TimeUnit::Nanoseconds => timestamp / 1_000,
                };
                let secs = micros / 1_000_000;
                let nsecs = ((micros % 1_000_000) * 1000) as u32;
                let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nsecs)
                    .unwrap_or_else(|| {
                        chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap()
                    });
                Ok(json!(dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        Time => {
            // Convert nanoseconds since midnight to ISO time: "HH:MM:SS.sss"
            let ca = series
                .time()
                .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to time: {}", e)))?;
            if let Some(nanos) = ca.phys.get(idx) {
                let hours = nanos / 3_600_000_000_000;
                let minutes = (nanos % 3_600_000_000_000) / 60_000_000_000;
                let seconds = (nanos % 60_000_000_000) / 1_000_000_000;
                let millis = (nanos % 1_000_000_000) / 1_000_000;
                Ok(json!(format!(
                    "{:02}:{:02}:{:02}.{:03}",
                    hours, minutes, seconds, millis
                )))
            } else {
                Ok(Value::Null)
            }
        }
        _ => {
            // Fallback: convert to string
            Ok(json!(series
                .get(idx)
                .map(|v| v.to_string())
                .unwrap_or_default()))
        }
    }
}

/// Given a bin center value and breaks array, return (bin_start, bin_end).
/// Find the bin interval that contains the given value.
///
/// The breaks array contains bin edges [e0, e1, e2, ...].
/// Returns the (lower, upper) edges of the bin containing the value.
/// Uses half-open intervals [lower, upper) except for the last bin which is [lower, upper].
pub(super) fn find_bin_for_value(value: f64, breaks: &[f64]) -> Option<(f64, f64)> {
    let n = breaks.len();
    if n < 2 {
        return None;
    }

    for i in 0..n - 1 {
        let lower = breaks[i];
        let upper = breaks[i + 1];
        let is_last_bin = i == n - 2;

        // Use [lower, upper) for all bins except the last which uses [lower, upper]
        let in_bin = if is_last_bin {
            value >= lower && value <= upper
        } else {
            value >= lower && value < upper
        };

        if in_bin {
            return Some((lower, upper));
        }
    }
    None
}

/// Convert Polars DataFrame to Vega-Lite data values with bin columns.
///
/// For columns with binned scales, this replaces the center value with bin_start
/// and adds a corresponding bin_end column.
pub(super) fn dataframe_to_values_with_bins(
    df: &DataFrame,
    binned_columns: &HashMap<String, Vec<f64>>,
) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    let height = df.height();
    let column_names = df.get_column_names();

    for row_idx in 0..height {
        let mut row_obj = Map::new();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            let column = df.get_columns().get(col_idx).ok_or_else(|| {
                GgsqlError::WriterError(format!("Failed to get column {}", col_name))
            })?;

            // Get value from series and convert to JSON Value
            let value = series_value_at(column.as_materialized_series(), row_idx)?;

            // Check if this column has binned data
            let col_name_str = col_name.to_string();
            if let Some(breaks) = binned_columns.get(&col_name_str) {
                // Check if this is a temporal string (date/datetime/time)
                let temporal_info = value.as_str().and_then(parse_temporal_string);

                // Get value as f64 - works for numeric columns or parsed temporal strings
                let numeric_value = value.as_f64().or_else(|| temporal_info.map(|(val, _)| val));

                if let Some(val) = numeric_value {
                    if let Some((start, end)) = find_bin_for_value(val, breaks) {
                        // Replace value with bin_start, preserving original value type
                        if let Some((_, temporal_type)) = temporal_info {
                            // Temporal column - format bin edges as ISO strings
                            let start_str = format_temporal(start, temporal_type);
                            let end_str = format_temporal(end, temporal_type);
                            row_obj.insert(col_name_str.clone(), json!(start_str));
                            row_obj.insert(naming::bin_end_column(&col_name_str), json!(end_str));
                        } else {
                            // Numeric column - use raw values
                            row_obj.insert(col_name_str.clone(), json!(start));
                            row_obj.insert(naming::bin_end_column(&col_name_str), json!(end));
                        }
                        continue;
                    }
                }
            }

            // Not binned or couldn't resolve edges - use original value
            row_obj.insert(col_name.to_string(), value);
        }

        values.push(Value::Object(row_obj));
    }

    Ok(values)
}

/// Detect the temporal type of a string value.
/// Returns the parsed numeric value and the type.
///
/// Uses ArrayElement's parsing methods which support comprehensive format variations.
pub(super) fn parse_temporal_string(s: &str) -> Option<(f64, TemporalType)> {
    // Try date first (YYYY-MM-DD) - must check before datetime since dates are shorter
    if let Some(ArrayElement::Date(days)) = ArrayElement::from_date_string(s) {
        return Some((days as f64, TemporalType::Date));
    }
    // Try datetime (various ISO formats with/without timezone)
    if let Some(ArrayElement::DateTime(micros)) = ArrayElement::from_datetime_string(s) {
        return Some((micros as f64, TemporalType::DateTime));
    }
    // Try time (HH:MM:SS[.sss])
    if let Some(ArrayElement::Time(nanos)) = ArrayElement::from_time_string(s) {
        return Some((nanos as f64, TemporalType::Time));
    }
    None
}

/// Format a numeric temporal value back to ISO string.
pub(super) fn format_temporal(value: f64, temporal_type: TemporalType) -> String {
    match temporal_type {
        TemporalType::Date => ArrayElement::date_to_iso(value as i32),
        TemporalType::DateTime => ArrayElement::datetime_to_iso(value as i64),
        TemporalType::Time => ArrayElement::time_to_iso(value as i64),
    }
}

/// Collect binned column information from spec.
///
/// Returns a map of column name -> breaks array for all columns with binned scales.
/// The column name uses the aesthetic-prefixed format (e.g., `__ggsql_aes_x__`) since
/// that's what appears in the DataFrame after query execution.
///
/// Only x and y aesthetics are collected since only those have x2/y2 counterparts
/// in Vega-Lite for representing bin ranges.
pub(super) fn collect_binned_columns(spec: &Plot) -> HashMap<String, Vec<f64>> {
    let mut binned_columns: HashMap<String, Vec<f64>> = HashMap::new();

    for scale in &spec.scales {
        // Only x and y aesthetics support bin ranges (x2/y2) in Vega-Lite
        if !is_primary_positional(&scale.aesthetic) {
            continue;
        }

        // Check if this is a binned scale
        let is_binned = scale
            .scale_type
            .as_ref()
            .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
            .unwrap_or(false);

        if !is_binned {
            continue;
        }

        // Get breaks array from scale properties
        if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") {
            let break_values: Vec<f64> = breaks.iter().filter_map(|e| e.to_f64()).collect();

            if break_values.len() >= 2 {
                // Insert the aesthetic column name (what's in the DataFrame after execution)
                let aes_col_name = naming::aesthetic_column(&scale.aesthetic);
                binned_columns.insert(aes_col_name, break_values.clone());

                // Also insert mappings for original column names (for unit tests and
                // cases where the full pipeline isn't used)
                for layer in &spec.layers {
                    if let Some(AestheticValue::Column { name: col, .. }) =
                        layer.mappings.aesthetics.get(&scale.aesthetic)
                    {
                        binned_columns.insert(col.clone(), break_values.clone());
                    }
                }
            }
        }
    }

    binned_columns
}

/// Check if an aesthetic has a binned scale in the spec.
pub(super) fn is_binned_aesthetic(aesthetic: &str, spec: &Plot) -> bool {
    let primary = primary_aesthetic(aesthetic);
    spec.find_scale(primary)
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
        .unwrap_or(false)
}

/// Unify multiple datasets into a single dataset with source identification.
///
/// This concatenates all layer datasets into one unified dataset, adding a
/// `__ggsql_source__` field to each row that identifies which layer's data
/// the row belongs to. Each layer then uses a Vega-Lite transform filter
/// to select its data.
///
/// # Arguments
/// * `datasets` - Map of dataset key to Vega-Lite JSON values array
///
/// # Returns
/// Unified array of all rows with source identification
pub(super) fn unify_datasets(datasets: &Map<String, Value>) -> Result<Vec<Value>> {
    // 1. Collect all unique column names across all datasets
    let mut all_columns: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_key, values) in datasets {
        if let Some(arr) = values.as_array() {
            for row in arr {
                if let Some(obj) = row.as_object() {
                    for col_name in obj.keys() {
                        all_columns.insert(col_name.clone());
                    }
                }
            }
        }
    }

    // 2. For each dataset, for each row:
    //    - Include all columns (null for missing)
    //    - Add __ggsql_source__ field with dataset key
    let mut unified = Vec::new();
    for (key, values) in datasets {
        if let Some(arr) = values.as_array() {
            for row in arr {
                if let Some(obj) = row.as_object() {
                    let mut new_row = Map::new();

                    // Include all columns from union schema (null for missing)
                    for col_name in &all_columns {
                        let value = obj.get(col_name).cloned().unwrap_or(Value::Null);
                        new_row.insert(col_name.clone(), value);
                    }

                    // Add source identifier
                    new_row.insert(naming::SOURCE_COLUMN.to_string(), json!(key));

                    unified.push(Value::Object(new_row));
                }
            }
        }
    }

    Ok(unified)
}
