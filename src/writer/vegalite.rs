//! Vega-Lite JSON writer implementation
//!
//! Converts ggsql specifications and DataFrames into Vega-Lite JSON format
//! for web-based interactive visualizations.
//!
//! # Mapping Strategy
//!
//! - ggsql Geom → Vega-Lite mark type
//! - ggsql aesthetics → Vega-Lite encoding channels
//! - ggsql layers → Vega-Lite layer composition
//! - Polars DataFrame → Vega-Lite inline data
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::writer::{Writer, VegaLiteWriter};
//!
//! let writer = VegaLiteWriter::new();
//! let vega_json = writer.write(&spec, &dataframe)?;
//! // Can be rendered in browser with vega-embed
//! ```

use crate::plot::layer::geom::{GeomAesthetics, GeomType};
use crate::plot::scale::{linetype_to_stroke_dash, shape_to_svg_path, ScaleTypeKind};

/// Conversion factor from points to pixels (CSS standard: 96 DPI, 72 points/inch)
/// 1 point = 96/72 pixels ≈ 1.333
const POINTS_TO_PIXELS: f64 = 96.0 / 72.0;

/// Conversion factor from radius (in points) to area (in square pixels)
/// Used for size aesthetic: area = π × r² where r is in pixels
/// So: area_px² = π × (r_pt × POINTS_TO_PIXELS)² = π × r_pt² × (96/72)²
const POINTS_TO_AREA: f64 = std::f64::consts::PI * POINTS_TO_PIXELS * POINTS_TO_PIXELS;
// ArrayElement is used in tests and for pattern matching; suppress unused import warning
#[allow(unused_imports)]
use crate::plot::ArrayElement;
use crate::plot::{Coord, CoordType, ParameterValue};
use crate::writer::Writer;
use crate::{naming, Layer};
use crate::{AestheticValue, DataFrame, Geom, GgsqlError, Plot, Result};
use polars::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Build a Vega-Lite labelExpr from label mappings
///
/// Generates a conditional expression that renames or suppresses labels:
/// - `Some(label)` → rename to that label
/// - `None` → suppress label (empty string)
///
/// For non-temporal scales:
/// - Uses `datum.label` for comparisons
/// - Example: `"datum.label == 'A' ? 'Alpha' : datum.label == 'B' ? 'Beta' : datum.label"`
///
/// For temporal scales:
/// - Uses `timeFormat(datum.value, 'fmt')` for comparisons
/// - This is necessary because `datum.label` contains Vega-Lite's formatted label (e.g., "Jan 1, 2024")
///   but our label_mapping keys are ISO format strings (e.g., "2024-01-01")
/// - Example: `"timeFormat(datum.value, '%Y-%m-%d') == '2024-01-01' ? 'Q1 Start' : datum.label"`
fn build_label_expr(
    mappings: &HashMap<String, Option<String>>,
    time_format: Option<&str>,
    null_key: Option<&str>,
) -> String {
    if mappings.is_empty() {
        return "datum.label".to_string();
    }

    // Build the comparison expression based on whether this is temporal
    let comparison_expr = match time_format {
        Some(fmt) => format!("timeFormat(datum.value, '{}')", fmt),
        None => "datum.label".to_string(),
    };

    let mut parts: Vec<String> = mappings
        .iter()
        .map(|(from, to)| {
            let from_escaped = from.replace('\'', "\\'");

            // For threshold scales, the first terminal uses null instead of string comparison
            let condition = if null_key == Some(from.as_str()) {
                "datum.label == null".to_string()
            } else {
                format!("{} == '{}'", comparison_expr, from_escaped)
            };

            match to {
                Some(label) => {
                    let to_escaped = label.replace('\'', "\\'");
                    format!("{} ? '{}'", condition, to_escaped)
                }
                None => {
                    // NULL suppresses the label (empty string)
                    format!("{} ? ''", condition)
                }
            }
        })
        .collect();

    // Fallback to original label
    parts.push("datum.label".to_string());
    parts.join(" : ")
}

/// Build label mappings for threshold scale symbol legends
///
/// Maps Vega-Lite's auto-generated range labels to our desired labels.
/// VL format: "<low> – <high>" for most bins (en-dash U+2013), "≥ <low>" for last bin.
///
/// # Arguments
/// * `breaks` - All break values including terminals [0, 25, 50, 75, 100]
/// * `label_mapping` - Our desired labels keyed by break value string
/// * `closed` - Which side of bin is closed: "left" (default) or "right"
///
/// # Returns
/// HashMap mapping Vega-Lite's predicted labels to our replacement labels
fn build_symbol_legend_label_mapping(
    breaks: &[ArrayElement],
    label_mapping: &HashMap<String, Option<String>>,
    closed: &str,
) -> HashMap<String, Option<String>> {
    let mut result = HashMap::new();

    // We have N breaks = N-1 bins
    // legend.values has N-1 entries (last terminal excluded for symbol legends)
    if breaks.len() < 2 {
        return result;
    }
    let num_bins = breaks.len() - 1;

    for i in 0..num_bins {
        let lower = &breaks[i];
        let upper = &breaks[i + 1];
        let lower_str = lower.to_key_string();
        let upper_str = upper.to_key_string();

        // Get our desired label for this bin (keyed by lower bound)
        let our_label = label_mapping.get(&lower_str).cloned().flatten();

        // Predict Vega-Lite's generated label
        // All but last: "<lower> – <upper>" (en-dash U+2013 with spaces)
        // Last bin: "≥ <lower>" (greater-than-or-equal U+2265)
        let vl_label = if i == num_bins - 1 {
            format!("≥ {}", lower_str)
        } else {
            format!("{} – {}", lower_str, upper_str)
        };

        // Check if terminals are suppressed (mapped to None)
        let lower_suppressed = label_mapping.get(&lower_str) == Some(&None);
        let upper_suppressed = label_mapping.get(&upper_str) == Some(&None);

        // Get labels for building range format (fall back to break values)
        let lower_label = our_label.clone().unwrap_or_else(|| lower_str.clone());
        let upper_label = label_mapping
            .get(&upper_str)
            .cloned()
            .flatten()
            .unwrap_or_else(|| upper_str.clone());

        // Determine the replacement label
        // Priority: terminal suppression → range format with custom labels
        let replacement = if i == 0 && lower_suppressed {
            // First bin with suppressed lower terminal → open format
            let symbol = if closed == "right" { "≤" } else { "<" };
            Some(format!("{} {}", symbol, upper_label))
        } else if i == num_bins - 1 && upper_suppressed {
            // Last bin with suppressed upper terminal → open format
            let symbol = if closed == "right" { ">" } else { "≥" };
            Some(format!("{} {}", symbol, lower_label))
        } else {
            // Use range format with custom labels: "<lower_label> – <upper_label>"
            Some(format!("{} – {}", lower_label, upper_label))
        };

        result.insert(vl_label, replacement);
    }

    result
}

/// Vega-Lite JSON writer
/// Temporal type for binned date/datetime/time columns
#[derive(Debug, Clone, Copy, PartialEq)]
enum TemporalType {
    Date,
    DateTime,
    Time,
}

///
/// Generates Vega-Lite v6 specifications from ggsql specs and data.
pub struct VegaLiteWriter {
    /// Vega-Lite schema version
    schema: String,
}

impl VegaLiteWriter {
    /// Create a new Vega-Lite writer with default settings
    pub fn new() -> Self {
        Self {
            schema: "https://vega.github.io/schema/vega-lite/v6.json".to_string(),
        }
    }

    /// Convert Polars DataFrame to Vega-Lite data values (array of objects)
    fn dataframe_to_values(&self, df: &DataFrame) -> Result<Vec<Value>> {
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
                let value = self.series_value_at(column.as_materialized_series(), row_idx)?;
                row_obj.insert(col_name.to_string(), value);
            }

            values.push(Value::Object(row_obj));
        }

        Ok(values)
    }

    /// Get a single value from a series at a given index as JSON Value
    fn series_value_at(&self, series: &Series, idx: usize) -> Result<Value> {
        use DataType::*;

        match series.dtype() {
            Int8 => {
                let ca = series
                    .i8()
                    .map_err(|e| GgsqlError::WriterError(format!("Failed to cast to i8: {}", e)))?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int16 => {
                let ca = series.i16().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i16: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int32 => {
                let ca = series.i32().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i32: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int64 => {
                let ca = series.i64().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i64: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Float32 => {
                let ca = series.f32().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to f32: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Float64 => {
                let ca = series.f64().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to f64: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Boolean => {
                let ca = series.bool().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to bool: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            String => {
                let ca = series.str().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to string: {}", e))
                })?;
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
                let ca = series.date().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to date: {}", e))
                })?;
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
                let ca = series.time().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to time: {}", e))
                })?;
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
    fn find_bin_for_value(value: f64, breaks: &[f64]) -> Option<(f64, f64)> {
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
    fn dataframe_to_values_with_bins(
        &self,
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
                let value = self.series_value_at(column.as_materialized_series(), row_idx)?;

                // Check if this column has binned data
                let col_name_str = col_name.to_string();
                if let Some(breaks) = binned_columns.get(&col_name_str) {
                    // Check if this is a temporal string (date/datetime/time)
                    let temporal_info = value.as_str().and_then(Self::parse_temporal_string);

                    // Get value as f64 - works for numeric columns or parsed temporal strings
                    let numeric_value =
                        value.as_f64().or_else(|| temporal_info.map(|(val, _)| val));

                    if let Some(val) = numeric_value {
                        if let Some((start, end)) = Self::find_bin_for_value(val, breaks) {
                            // Replace value with bin_start, preserving original value type
                            if let Some((_, temporal_type)) = temporal_info {
                                // Temporal column - format bin edges as ISO strings
                                let start_str = Self::format_temporal(start, temporal_type);
                                let end_str = Self::format_temporal(end, temporal_type);
                                row_obj.insert(col_name_str.clone(), json!(start_str));
                                row_obj
                                    .insert(naming::bin_end_column(&col_name_str), json!(end_str));
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
    fn parse_temporal_string(s: &str) -> Option<(f64, TemporalType)> {
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
    fn format_temporal(value: f64, temporal_type: TemporalType) -> String {
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
    fn collect_binned_columns(&self, spec: &Plot) -> HashMap<String, Vec<f64>> {
        let mut binned_columns: HashMap<String, Vec<f64>> = HashMap::new();

        for scale in &spec.scales {
            // Only x and y aesthetics support bin ranges (x2/y2) in Vega-Lite
            if scale.aesthetic != "x" && scale.aesthetic != "y" {
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
    fn is_binned_aesthetic(&self, aesthetic: &str, spec: &Plot) -> bool {
        let primary = GeomAesthetics::primary_aesthetic(aesthetic);
        spec.find_scale(primary)
            .and_then(|s| s.scale_type.as_ref())
            .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
            .unwrap_or(false)
    }

    /// Count the number of binned non-positional scales in the spec.
    /// This is used to determine if legends should use symbol style (which requires
    /// removing the last terminal value) or gradient style (which keeps all values).
    fn count_binned_legend_scales(&self, spec: &Plot) -> usize {
        spec.scales
            .iter()
            .filter(|scale| {
                // Check if binned
                let is_binned = scale
                    .scale_type
                    .as_ref()
                    .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
                    .unwrap_or(false);

                // Check if non-positional (legend aesthetic)
                let is_legend_aesthetic = !matches!(
                    scale.aesthetic.as_str(),
                    "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
                );

                is_binned && is_legend_aesthetic
            })
            .count()
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
    fn unify_datasets(&self, datasets: &Map<String, Value>) -> Result<Vec<Value>> {
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

    /// Map ggsql Geom to Vega-Lite mark type
    /// Always includes `clip: true` to ensure marks don't render outside plot bounds
    fn geom_to_mark(&self, geom: &Geom) -> Value {
        let mark_type = match geom.geom_type() {
            GeomType::Point => "point",
            GeomType::Line => "line",
            GeomType::Path => "line",
            GeomType::Bar => "bar",
            GeomType::Area => "area",
            GeomType::Tile => "rect",
            GeomType::Ribbon => "area",
            GeomType::Polygon => "line",
            GeomType::Histogram => "bar",
            GeomType::Density => "area",
            GeomType::Boxplot => "boxplot",
            GeomType::Text => "text",
            GeomType::Label => "text",
            _ => "point", // Default fallback
        };
        json!({
            "type": mark_type,
            "clip": true
        })
    }

    /// Check if a string column contains numeric values
    fn is_numeric_string_column(&self, series: &Series) -> bool {
        if let Ok(ca) = series.str() {
            // Check first few non-null values to see if they're numeric
            for val in ca.into_iter().flatten().take(5) {
                if val.parse::<f64>().is_err() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Infer Vega-Lite field type from DataFrame column
    fn infer_field_type(&self, df: &DataFrame, field: &str) -> String {
        if let Ok(column) = df.column(field) {
            use DataType::*;
            match column.dtype() {
                Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 | Float32
                | Float64 => "quantitative",
                Boolean => "nominal",
                String => {
                    // Check if string column contains numeric values
                    if self.is_numeric_string_column(column.as_materialized_series()) {
                        "quantitative"
                    } else {
                        "nominal"
                    }
                }
                Date | Datetime(_, _) | Time => "temporal",
                _ => "nominal",
            }
            .to_string()
        } else {
            "nominal".to_string()
        }
    }

    /// Determine Vega-Lite field type from scale specification
    fn determine_field_type_from_scale(
        &self,
        scale: &crate::plot::Scale,
        inferred: &str,
        _aesthetic: &str,
        identity_scale: &mut bool,
    ) -> String {
        // Use scale type if explicitly specified
        if let Some(scale_type) = &scale.scale_type {
            use crate::plot::ScaleTypeKind;
            match scale_type.scale_type_kind() {
                ScaleTypeKind::Continuous => "quantitative",
                ScaleTypeKind::Discrete => "nominal",
                ScaleTypeKind::Binned => "quantitative", // Binned data is still quantitative
                ScaleTypeKind::Ordinal => "ordinal",     // Native Vega-Lite ordinal type
                ScaleTypeKind::Identity => {
                    *identity_scale = true;
                    inferred
                }
            }
            .to_string()
        } else {
            // Scale exists but no type specified, use inferred
            inferred.to_string()
        }
    }

    /// Build encoding channel from aesthetic mapping
    ///
    /// The `titled_families` set tracks which aesthetic families have already received
    /// a title, ensuring only one title per family (e.g., one title for x/xmin/xmax).
    ///
    /// The `primary_aesthetics` set contains primary aesthetics that exist in the layer.
    /// When a primary exists, variant aesthetics (xmin, ymin, etc.) get `title: null`.
    fn build_encoding_channel(
        &self,
        aesthetic: &str,
        value: &AestheticValue,
        df: &DataFrame,
        spec: &Plot,
        titled_families: &mut std::collections::HashSet<String>,
        primary_aesthetics: &std::collections::HashSet<String>,
    ) -> Result<Value> {
        match value {
            AestheticValue::Column {
                name: col,
                original_name,
                is_dummy,
            } => {
                // Check if there's a scale specification for this aesthetic or its primary
                // E.g., "xmin" should use the "x" scale
                let primary = GeomAesthetics::primary_aesthetic(aesthetic);
                let inferred = self.infer_field_type(df, col);
                let mut identity_scale = false;

                let field_type = if let Some(scale) = spec.find_scale(primary) {
                    // Check if the transform indicates temporal data
                    // (Transform takes precedence since it's resolved from column dtype)
                    if let Some(ref transform) = scale.transform {
                        if transform.is_temporal() {
                            "temporal".to_string()
                        } else {
                            // Non-temporal transform, fall through to scale type check
                            self.determine_field_type_from_scale(
                                scale,
                                &inferred,
                                aesthetic,
                                &mut identity_scale,
                            )
                        }
                    } else {
                        // No transform, check scale type
                        self.determine_field_type_from_scale(
                            scale,
                            &inferred,
                            aesthetic,
                            &mut identity_scale,
                        )
                    }
                } else {
                    // No scale specification, infer from data
                    inferred
                };

                // Check if this aesthetic has a binned scale
                let is_binned = spec
                    .find_scale(primary)
                    .and_then(|s| s.scale_type.as_ref())
                    .map(|st| st.scale_type_kind() == ScaleTypeKind::Binned)
                    .unwrap_or(false);

                let mut encoding = json!({
                    "field": col,
                    "type": field_type,
                });

                // For binned scales, add bin: "binned" to enable Vega-Lite's binned data handling
                // This allows proper axis tick placement at bin edges and range labels in legends
                if is_binned {
                    encoding["bin"] = json!("binned");
                }

                // Apply title handling:
                // - Primary aesthetics (x, y, color) can set the title
                // - Variant aesthetics (xmin, ymin, etc.) only get title if no primary exists
                // - When a primary exists, variants get title: null to prevent axis label conflicts
                let is_primary = aesthetic == primary;
                let primary_exists = primary_aesthetics.contains(primary);

                if is_primary && !titled_families.contains(primary) {
                    // Primary aesthetic: set title from explicit label or original_name
                    let explicit_label = spec
                        .labels
                        .as_ref()
                        .and_then(|labels| labels.labels.get(primary));

                    if let Some(label) = explicit_label {
                        encoding["title"] = json!(label);
                        titled_families.insert(primary.to_string());
                    } else if let Some(orig) = original_name {
                        // Use original column name as default title when available
                        // (preserves readable names when columns are renamed to internal names)
                        encoding["title"] = json!(orig);
                        titled_families.insert(primary.to_string());
                    }
                } else if !is_primary && primary_exists {
                    // Variant with primary present: suppress title to avoid axis label conflicts
                    encoding["title"] = Value::Null;
                } else if !is_primary && !primary_exists && !titled_families.contains(primary) {
                    // Variant without primary: allow first variant to claim title (for explicit labels)
                    if let Some(ref labels) = spec.labels {
                        if let Some(label) = labels.labels.get(primary) {
                            encoding["title"] = json!(label);
                            titled_families.insert(primary.to_string());
                        }
                    }
                }

                let mut scale_obj = serde_json::Map::new();
                // Track if we're using a color range array (needs gradient legend)
                let mut needs_gradient_legend = false;

                // Track if this is a binned non-positional aesthetic (needs threshold scale)
                // Computed early so we can skip normal domain handling for threshold scales
                let is_binned_legend = is_binned
                    && !matches!(
                        aesthetic,
                        "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
                    );

                // Use scale properties from the primary aesthetic's scale
                // (same scale lookup as used above for field_type)
                if let Some(scale) = spec.find_scale(primary) {
                    // Apply scale properties from SCALE if specified
                    use crate::plot::{ArrayElement, OutputRange};

                    // Apply domain from input_range (FROM clause)
                    // Skip for threshold scales - they use internal breaks as domain instead
                    if !is_binned_legend {
                        if let Some(ref domain_values) = scale.input_range {
                            let domain_json: Vec<Value> =
                                domain_values.iter().map(|elem| elem.to_json()).collect();
                            scale_obj.insert("domain".to_string(), json!(domain_json));
                        }
                    }

                    // Apply range from output_range (TO clause)

                    if let Some(ref output_range) = scale.output_range {
                        match output_range {
                            OutputRange::Array(range_values) => {
                                let range_json: Vec<Value> = range_values
                                    .iter()
                                    .map(|elem| match elem {
                                        ArrayElement::String(s) => {
                                            // For shape aesthetic, convert to SVG path
                                            if aesthetic == "shape" {
                                                if let Some(svg_path) = shape_to_svg_path(s) {
                                                    json!(svg_path)
                                                } else {
                                                    // Unknown shape, pass through
                                                    json!(s)
                                                }
                                            // For linetype aesthetic, convert to dash array
                                            } else if aesthetic == "linetype" {
                                                if let Some(dash_array) = linetype_to_stroke_dash(s)
                                                {
                                                    json!(dash_array)
                                                } else {
                                                    // Unknown linetype, pass through
                                                    json!(s)
                                                }
                                            } else {
                                                json!(s)
                                            }
                                        }
                                        ArrayElement::Number(n) => {
                                            match aesthetic {
                                                // Size: convert radius (points) to area (pixels²)
                                                // area = r² × π × (96/72)²
                                                "size" => json!(n * n * POINTS_TO_AREA),
                                                // Linewidth: convert points to pixels
                                                "linewidth" => json!(n * POINTS_TO_PIXELS),
                                                // Other aesthetics: pass through unchanged
                                                _ => json!(n),
                                            }
                                        }
                                        // All other types use to_json()
                                        other => other.to_json(),
                                    })
                                    .collect();
                                scale_obj.insert("range".to_string(), json!(range_json));

                                // For continuous color scales with range array, use gradient legend
                                if matches!(aesthetic, "fill" | "stroke")
                                    && matches!(
                                        scale.scale_type.as_ref().map(|st| st.scale_type_kind()),
                                        Some(ScaleTypeKind::Continuous)
                                    )
                                {
                                    needs_gradient_legend = true;
                                }
                            }
                            OutputRange::Palette(palette_name) => {
                                // Named palette - expand to color scheme
                                scale_obj.insert(
                                    "scheme".to_string(),
                                    json!(palette_name.to_lowercase()),
                                );
                            }
                        }
                    }

                    // Handle transform (VIA clause)
                    if let Some(ref transform) = scale.transform {
                        use crate::plot::scale::TransformKind;
                        match transform.transform_kind() {
                            TransformKind::Identity => {} // Linear (default), no additional scale properties needed
                            TransformKind::Log10 => {
                                scale_obj.insert("type".to_string(), json!("log"));
                                scale_obj.insert("base".to_string(), json!(10));
                                scale_obj.insert("zero".to_string(), json!(false));
                            }
                            TransformKind::Log => {
                                // Natural logarithm - Vega-Lite uses "log" with base e
                                scale_obj.insert("type".to_string(), json!("log"));
                                scale_obj.insert("base".to_string(), json!(std::f64::consts::E));
                                scale_obj.insert("zero".to_string(), json!(false));
                            }
                            TransformKind::Log2 => {
                                scale_obj.insert("type".to_string(), json!("log"));
                                scale_obj.insert("base".to_string(), json!(2));
                                scale_obj.insert("zero".to_string(), json!(false));
                            }
                            TransformKind::Sqrt => {
                                scale_obj.insert("type".to_string(), json!("sqrt"));
                            }
                            TransformKind::Square => {
                                scale_obj.insert("type".to_string(), json!("pow"));
                                scale_obj.insert("exponent".to_string(), json!(2));
                            }
                            TransformKind::Exp10 | TransformKind::Exp2 | TransformKind::Exp => {
                                // Vega-Lite doesn't have native exp scales
                                // Using linear scale; data is already transformed in data space
                                eprintln!(
                                    "Warning: {} transform has no native Vega-Lite equivalent, using linear scale",
                                    transform.name()
                                );
                            }
                            TransformKind::Asinh | TransformKind::PseudoLog => {
                                scale_obj.insert("type".to_string(), json!("symlog"));
                            }
                            // Temporal transforms are identity in numeric space;
                            // the field type ("temporal") is set based on the transform kind
                            TransformKind::Date | TransformKind::DateTime | TransformKind::Time => {
                            }
                            // Discrete transforms (String, Bool) don't affect Vega-Lite scale type;
                            // the data casting happens at the SQL level before reaching the writer
                            TransformKind::String | TransformKind::Bool => {}
                            // Integer transform is linear scale; casting happens at SQL level
                            TransformKind::Integer => {}
                        }
                    }

                    // Handle binned non-positional aesthetics with threshold scale
                    // For legends (fill, stroke, color, etc.), binned scales should use threshold
                    // scale type to show discrete color blocks instead of a smooth gradient
                    if is_binned_legend {
                        scale_obj.insert("type".to_string(), json!("threshold"));

                        // Threshold domain = internal breaks (excluding first and last terminal bounds)
                        // breaks = [0, 25, 50, 75, 100] → domain = [25, 50, 75]
                        if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks")
                        {
                            if breaks.len() > 2 {
                                let internal_breaks: Vec<Value> = breaks[1..breaks.len() - 1]
                                    .iter()
                                    .map(|e| e.to_json())
                                    .collect();
                                scale_obj.insert("domain".to_string(), json!(internal_breaks));
                            }
                        }
                    }

                    // Handle reverse property (SETTING clause)
                    use crate::plot::ParameterValue;
                    if let Some(ParameterValue::Boolean(true)) = scale.properties.get("reverse") {
                        scale_obj.insert("reverse".to_string(), json!(true));

                        // For discrete/ordinal scales with legends, also reverse the legend order
                        // Vega-Lite's scale.reverse only reverses the visual mapping, not the legend
                        if let Some(ref scale_type) = scale.scale_type {
                            let kind = scale_type.scale_type_kind();
                            if matches!(kind, ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal) {
                                // Only for non-positional aesthetics (those with legends)
                                if !matches!(
                                    aesthetic,
                                    "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
                                ) {
                                    // Use the input_range (domain) if available
                                    if let Some(ref domain) = scale.input_range {
                                        let reversed_domain: Vec<Value> =
                                            domain.iter().rev().map(|e| e.to_json()).collect();
                                        // Set legend.values with reversed order
                                        if !encoding.get("legend").is_some_and(|v| v.is_null()) {
                                            let legend = encoding
                                                .get_mut("legend")
                                                .and_then(|v| v.as_object_mut());
                                            if let Some(legend_map) = legend {
                                                legend_map.insert(
                                                    "values".to_string(),
                                                    json!(reversed_domain),
                                                );
                                            } else {
                                                encoding["legend"] =
                                                    json!({"values": reversed_domain});
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Handle resolved breaks -> axis.values or legend.values
                    // breaks is stored as Array in properties after resolution
                    // For binned scales, we still need to set axis.values manually because
                    // Vega-Lite's automatic tick placement with bin:"binned" only works for equal-width bins
                    if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") {
                        // Get all break values (filtering is applied selectively below)
                        let all_values: Vec<Value> = breaks.iter().map(|e| e.to_json()).collect();

                        // Positional aesthetics use axis.values, others use legend.values
                        if matches!(
                            aesthetic,
                            "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
                        ) {
                            // For positional aesthetics (axes), filter out values that have
                            // label_mapping = None (suppressed terminal breaks from oob='squish')
                            let axis_values: Vec<Value> =
                                if let Some(ref label_mapping) = scale.label_mapping {
                                    breaks
                                        .iter()
                                        .filter(|e| {
                                            let key = e.to_key_string();
                                            !matches!(label_mapping.get(&key), Some(None))
                                        })
                                        .map(|e| e.to_json())
                                        .collect()
                                } else {
                                    all_values.clone()
                                };

                            // Add to axis object
                            if !encoding.get("axis").is_some_and(|v| v.is_null()) {
                                let axis = encoding.get_mut("axis").and_then(|v| v.as_object_mut());
                                if let Some(axis_map) = axis {
                                    axis_map.insert("values".to_string(), json!(axis_values));
                                } else {
                                    encoding["axis"] = json!({"values": axis_values});
                                }
                            }
                        } else {
                            // Add to legend object for non-positional aesthetics
                            // Note: We use all_values here (no filtering of suppressed labels)
                            // because legends should show all bins, unlike axes where terminal
                            // breaks may be suppressed by oob='squish'.
                            // For threshold (binned) scales, symbol legends need the last terminal
                            // removed to avoid an extra symbol. Gradient legends (fill/stroke alone)
                            // keep all values.
                            let legend_values = if is_binned_legend {
                                // Determine if this is a symbol legend case:
                                // - Not fill/stroke (always symbol legend)
                                // - OR multiple binned legend scales (forces symbol legend)
                                let binned_legend_count = self.count_binned_legend_scales(spec);
                                let is_gradient_aesthetic = matches!(aesthetic, "fill" | "stroke");
                                let uses_symbol_legend =
                                    !is_gradient_aesthetic || binned_legend_count > 1;

                                if uses_symbol_legend && !all_values.is_empty() {
                                    // Remove the last terminal for symbol legends
                                    all_values[..all_values.len() - 1].to_vec()
                                } else {
                                    all_values
                                }
                            } else {
                                all_values
                            };

                            if !encoding.get("legend").is_some_and(|v| v.is_null()) {
                                let legend =
                                    encoding.get_mut("legend").and_then(|v| v.as_object_mut());
                                if let Some(legend_map) = legend {
                                    legend_map.insert("values".to_string(), json!(legend_values));
                                } else {
                                    encoding["legend"] = json!({"values": legend_values});
                                }
                            }
                        }
                    }

                    // Handle label_mapping -> labelExpr (RENAMING clause)
                    if let Some(ref label_mapping) = scale.label_mapping {
                        if !label_mapping.is_empty() {
                            // For temporal scales, use timeFormat() to compare against ISO keys
                            // because datum.label contains Vega-Lite's formatted label (e.g., "Jan 1, 2024")
                            // but our label_mapping keys are ISO format strings (e.g., "2024-01-01")
                            use crate::plot::scale::TransformKind;
                            let time_format =
                                scale
                                    .transform
                                    .as_ref()
                                    .and_then(|t| match t.transform_kind() {
                                        TransformKind::Date => Some("%Y-%m-%d"),
                                        TransformKind::DateTime => Some("%Y-%m-%dT%H:%M:%S"),
                                        TransformKind::Time => Some("%H:%M:%S"),
                                        _ => None,
                                    });

                            // For threshold scales (binned legends), determine if symbol legend
                            // Symbol legends need different label handling: VL generates range-style
                            // labels like "0 – 25", "25 – 50", "≥ 75" which we need to map
                            let (filtered_mapping, null_key) = if is_binned_legend {
                                let binned_legend_count = self.count_binned_legend_scales(spec);
                                let is_gradient_aesthetic = matches!(aesthetic, "fill" | "stroke");
                                let uses_symbol_legend =
                                    !is_gradient_aesthetic || binned_legend_count > 1;

                                if uses_symbol_legend {
                                    // Symbol legend: map VL's range-style labels to our labels
                                    let closed = scale
                                        .properties
                                        .get("closed")
                                        .and_then(|v| {
                                            if let ParameterValue::String(s) = v {
                                                Some(s.as_str())
                                            } else {
                                                None
                                            }
                                        })
                                        .unwrap_or("left");

                                    if let Some(ParameterValue::Array(breaks)) =
                                        scale.properties.get("breaks")
                                    {
                                        let symbol_mapping = build_symbol_legend_label_mapping(
                                            breaks,
                                            label_mapping,
                                            closed,
                                        );
                                        (symbol_mapping, None) // No null_key for symbol legends
                                    } else {
                                        (label_mapping.clone(), None)
                                    }
                                } else {
                                    // Gradient legend: use null_key for first terminal
                                    let first_key = scale.properties.get("breaks").and_then(|b| {
                                        if let ParameterValue::Array(breaks) = b {
                                            breaks.first().map(|e| e.to_key_string())
                                        } else {
                                            None
                                        }
                                    });
                                    (label_mapping.clone(), first_key)
                                }
                            } else {
                                (label_mapping.clone(), None)
                            };

                            let label_expr = build_label_expr(
                                &filtered_mapping,
                                time_format,
                                null_key.as_deref(),
                            );

                            if matches!(
                                aesthetic,
                                "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
                            ) {
                                // Add to axis object
                                let axis = encoding.get_mut("axis").and_then(|v| v.as_object_mut());
                                if let Some(axis_map) = axis {
                                    axis_map.insert("labelExpr".to_string(), json!(label_expr));
                                } else {
                                    encoding["axis"] = json!({"labelExpr": label_expr});
                                }
                            } else {
                                // Add to legend object for non-positional aesthetics
                                let legend =
                                    encoding.get_mut("legend").and_then(|v| v.as_object_mut());
                                if let Some(legend_map) = legend {
                                    legend_map.insert("labelExpr".to_string(), json!(label_expr));
                                } else {
                                    encoding["legend"] = json!({"labelExpr": label_expr});
                                }
                            }
                        }
                    }
                }
                // We don't automatically want to include 0 in our position scales
                if aesthetic == "x" || aesthetic == "y" {
                    scale_obj.insert("zero".to_string(), json!(Value::Bool(false)));
                }

                if identity_scale {
                    // When we have an identity scale, these scale properties don't matter.
                    // We should return a `"scale": null`` in the encoding channel
                    encoding["scale"] = json!(Value::Null)
                } else if !scale_obj.is_empty() {
                    encoding["scale"] = json!(scale_obj);
                }

                // For continuous color scales with range array, use gradient legend
                // (scheme-based scales automatically get gradient legends from Vega-Lite)
                if needs_gradient_legend {
                    // Merge gradient type into existing legend object (preserves values, etc.)
                    if let Some(legend_obj) =
                        encoding.get_mut("legend").and_then(|v| v.as_object_mut())
                    {
                        legend_obj.insert("type".to_string(), json!("gradient"));
                    } else if !encoding.get("legend").is_some_and(|v| v.is_null()) {
                        // No legend object yet, create one with gradient type
                        encoding["legend"] = json!({"type": "gradient"});
                    }
                    // If legend is explicitly null, leave it (user disabled legend via GUIDE)
                }

                // Hide axis for dummy columns (e.g., x when bar chart has no x mapped)
                if *is_dummy {
                    encoding["axis"] = json!(null);
                }

                Ok(encoding)
            }
            AestheticValue::Literal(lit) => {
                // For literal values, use constant value encoding
                // Size and linewidth need unit conversion from points to Vega-Lite units
                let val = match lit {
                    ParameterValue::String(s) => json!(s),
                    ParameterValue::Number(n) => {
                        match aesthetic {
                            // Size: interpret as radius in points, convert to area in pixels²
                            // area = r² × π × (96/72)²
                            "size" => json!(n * n * POINTS_TO_AREA),
                            // Linewidth: interpret as width in points, convert to pixels
                            "linewidth" => json!(n * POINTS_TO_PIXELS),
                            // Other aesthetics: pass through unchanged
                            _ => json!(n),
                        }
                    }
                    ParameterValue::Boolean(b) => json!(b),
                    // Grammar prevents arrays and null in literal aesthetic mappings
                    ParameterValue::Array(_) | ParameterValue::Null => unreachable!(
                        "Grammar prevents arrays and null in literal aesthetic mappings"
                    ),
                };
                Ok(json!({"value": val}))
            }
        }
    }

    /// Map ggsql aesthetic name to Vega-Lite encoding channel name
    fn map_aesthetic_name(&self, aesthetic: &str) -> String {
        match aesthetic {
            // Line aesthetics
            "linetype" => "strokeDash",
            "linewidth" => "strokeWidth",
            // Text aesthetics
            "label" => "text",
            // All other aesthetics pass through directly
            // (fill and stroke map to Vega-Lite's separate fill/stroke channels)
            _ => aesthetic,
        }
        .to_string()
    }

    /// Validate column references for a single layer against its specific DataFrame
    fn validate_layer_columns(
        &self,
        layer: &crate::plot::Layer,
        data: &DataFrame,
        layer_idx: usize,
    ) -> Result<()> {
        let available_columns: Vec<String> = data
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        for (aesthetic, value) in &layer.mappings.aesthetics {
            if let AestheticValue::Column { name: col, .. } = value {
                if !available_columns.contains(col) {
                    let source_desc = if let Some(src) = &layer.source {
                        format!(" (source: {})", src.as_str())
                    } else {
                        " (global data)".to_string()
                    };
                    return Err(GgsqlError::ValidationError(format!(
                        "Column '{}' referenced in aesthetic '{}' (layer {}{}) does not exist.\nAvailable columns: {}",
                        col,
                        aesthetic,
                        layer_idx + 1,
                        source_desc,
                        available_columns.join(", ")
                    )));
                }
            }
        }

        // Check partition_by columns
        for col in &layer.partition_by {
            if !available_columns.contains(col) {
                let source_desc = if let Some(src) = &layer.source {
                    format!(" (source: {})", src.as_str())
                } else {
                    " (global data)".to_string()
                };
                return Err(GgsqlError::ValidationError(format!(
                    "Column '{}' referenced in PARTITION BY (layer {}{}) does not exist.\nAvailable columns: {}",
                    col,
                    layer_idx + 1,
                    source_desc,
                    available_columns.join(", ")
                )));
            }
        }

        Ok(())
    }
}

impl Default for VegaLiteWriter {
    fn default() -> Self {
        Self::new()
    }
}

// Coordinate transformation methods
impl VegaLiteWriter {
    /// Apply coordinate transformations to the spec and data
    /// Returns (possibly transformed DataFrame, possibly modified spec)
    fn apply_coord_transforms(
        &self,
        spec: &Plot,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        if let Some(ref coord) = spec.coord {
            match coord.coord_type {
                CoordType::Cartesian => {
                    self.apply_cartesian_coord(coord, vl_spec, data)?;
                    Ok(None) // No DataFrame transformation needed
                }
                CoordType::Flip => {
                    self.apply_flip_coord(vl_spec)?;
                    Ok(None) // No DataFrame transformation needed
                }
                CoordType::Polar => {
                    // Polar requires DataFrame transformation for percentages
                    let transformed_df = self.apply_polar_coord(coord, spec, data, vl_spec)?;
                    Ok(Some(transformed_df))
                }
                _ => {
                    // Other coord types not yet implemented
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Apply Cartesian coordinate properties (xlim, ylim, aesthetic domains)
    fn apply_cartesian_coord(
        &self,
        coord: &Coord,
        vl_spec: &mut Value,
        _data: &DataFrame,
    ) -> Result<()> {
        // Apply xlim/ylim to scale domains
        for (prop_name, prop_value) in &coord.properties {
            match prop_name.as_str() {
                "xlim" => {
                    if let Some(limits) = self.extract_limits(prop_value)? {
                        self.apply_axis_limits(vl_spec, "x", limits)?;
                    }
                }
                "ylim" => {
                    if let Some(limits) = self.extract_limits(prop_value)? {
                        self.apply_axis_limits(vl_spec, "y", limits)?;
                    }
                }
                _ if self.is_aesthetic_name(prop_name) => {
                    // Aesthetic domain specification
                    if let Some(domain) = self.extract_input_range(prop_value)? {
                        self.apply_aesthetic_input_range(vl_spec, prop_name, domain)?;
                    }
                }
                _ => {
                    // ratio, clip - not yet implemented (TODO comments added by validation)
                }
            }
        }

        Ok(())
    }

    /// Apply Flip coordinate transformation (swap x and y)
    fn apply_flip_coord(&self, vl_spec: &mut Value) -> Result<()> {
        // Handle single layer
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(enc_obj) = encoding.as_object_mut() {
                // Swap x and y encodings
                if let (Some(x), Some(y)) = (enc_obj.remove("x"), enc_obj.remove("y")) {
                    enc_obj.insert("x".to_string(), y);
                    enc_obj.insert("y".to_string(), x);
                }
            }
        }

        // Handle multi-layer
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(enc_obj) = encoding.as_object_mut() {
                            if let (Some(x), Some(y)) = (enc_obj.remove("x"), enc_obj.remove("y")) {
                                enc_obj.insert("x".to_string(), y);
                                enc_obj.insert("y".to_string(), x);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply Polar coordinate transformation (bar→arc, point→arc with radius)
    fn apply_polar_coord(
        &self,
        coord: &Coord,
        spec: &Plot,
        _data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<DataFrame> {
        // Get theta field (defaults to 'y')
        let theta_field = coord
            .properties
            .get("theta")
            .and_then(|v| match v {
                ParameterValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "y".to_string());

        // Convert geoms to polar equivalents
        self.convert_geoms_to_polar(spec, vl_spec, &theta_field)?;

        // No DataFrame transformation needed - Vega-Lite handles polar math
        Ok(_data.clone())
    }

    /// Convert geoms to polar equivalents (bar→arc, point→arc with radius)
    fn convert_geoms_to_polar(
        &self,
        spec: &Plot,
        vl_spec: &mut Value,
        theta_field: &str,
    ) -> Result<()> {
        // Determine which aesthetic (x or y) maps to theta
        // Default: y maps to theta (pie chart style)
        let theta_aesthetic = theta_field;

        // Handle single layer
        if let Some(mark) = vl_spec.get_mut("mark") {
            *mark = self.convert_mark_to_polar(mark, spec)?;

            // Update encoding for polar
            if let Some(encoding) = vl_spec.get_mut("encoding") {
                self.update_encoding_for_polar(encoding, theta_aesthetic)?;
            }
        }

        // Handle multi-layer
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(mark) = layer.get_mut("mark") {
                        *mark = self.convert_mark_to_polar(mark, spec)?;

                        if let Some(encoding) = layer.get_mut("encoding") {
                            self.update_encoding_for_polar(encoding, theta_aesthetic)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert a mark type to its polar equivalent
    /// Preserves `clip: true` to ensure marks don't render outside plot bounds
    fn convert_mark_to_polar(&self, mark: &Value, _spec: &Plot) -> Result<Value> {
        let mark_str = if mark.is_string() {
            mark.as_str().unwrap()
        } else if let Some(mark_type) = mark.get("type") {
            mark_type.as_str().unwrap_or("bar")
        } else {
            "bar"
        };

        // Convert geom types to polar equivalents
        let polar_mark = match mark_str {
            "bar" | "col" => {
                // Bar/col in polar becomes arc (pie/donut slices)
                "arc"
            }
            "point" => {
                // Points in polar can stay as points or become arcs with radius
                // For now, keep as points (they'll plot at radius based on value)
                "point"
            }
            "line" => {
                // Lines in polar become circular/spiral lines
                "line"
            }
            "area" => {
                // Area in polar becomes arc with radius
                "arc"
            }
            _ => {
                // Other geoms: keep as-is or convert to arc
                "arc"
            }
        };

        Ok(json!({
            "type": polar_mark,
            "clip": true
        }))
    }

    /// Update encoding channels for polar coordinates
    fn update_encoding_for_polar(&self, encoding: &mut Value, theta_aesthetic: &str) -> Result<()> {
        let enc_obj = encoding
            .as_object_mut()
            .ok_or_else(|| GgsqlError::WriterError("Encoding is not an object".to_string()))?;

        // Map the theta aesthetic to theta channel
        if theta_aesthetic == "y" {
            // Standard pie chart: y → theta, x → color/category
            if let Some(y_enc) = enc_obj.remove("y") {
                enc_obj.insert("theta".to_string(), y_enc);
            }
            // Map x to color if not already mapped, and remove x from positional encoding
            if !enc_obj.contains_key("color") {
                if let Some(x_enc) = enc_obj.remove("x") {
                    enc_obj.insert("color".to_string(), x_enc);
                }
            } else {
                // If color is already mapped, just remove x from positional encoding
                enc_obj.remove("x");
            }
        } else if theta_aesthetic == "x" {
            // Reversed: x → theta, y → radius
            if let Some(x_enc) = enc_obj.remove("x") {
                enc_obj.insert("theta".to_string(), x_enc);
            }
            if let Some(y_enc) = enc_obj.remove("y") {
                enc_obj.insert("radius".to_string(), y_enc);
            }
        }

        Ok(())
    }

    // Helper methods

    fn extract_limits(&self, value: &ParameterValue) -> Result<Option<(f64, f64)>> {
        match value {
            ParameterValue::Array(arr) => {
                if arr.len() != 2 {
                    return Err(GgsqlError::WriterError(format!(
                        "xlim/ylim must be exactly 2 numbers, got {}",
                        arr.len()
                    )));
                }
                let min = arr[0].to_f64().ok_or_else(|| {
                    GgsqlError::WriterError("xlim/ylim values must be numeric".to_string())
                })?;
                let max = arr[1].to_f64().ok_or_else(|| {
                    GgsqlError::WriterError("xlim/ylim values must be numeric".to_string())
                })?;

                // Auto-swap if reversed
                let (min, max) = if min > max { (max, min) } else { (min, max) };

                Ok(Some((min, max)))
            }
            _ => Err(GgsqlError::WriterError(
                "xlim/ylim must be an array".to_string(),
            )),
        }
    }

    fn extract_input_range(&self, value: &ParameterValue) -> Result<Option<Vec<Value>>> {
        match value {
            ParameterValue::Array(arr) => {
                let domain: Vec<Value> = arr.iter().map(|elem| elem.to_json()).collect();
                Ok(Some(domain))
            }
            _ => Ok(None),
        }
    }

    fn apply_axis_limits(&self, vl_spec: &mut Value, axis: &str, limits: (f64, f64)) -> Result<()> {
        let domain = json!([limits.0, limits.1]);

        // Apply to encoding if present
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(axis_enc) = encoding.get_mut(axis) {
                axis_enc["scale"] = json!({"domain": domain});
            }
        }

        // Apply to layers if present
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(axis_enc) = encoding.get_mut(axis) {
                            axis_enc["scale"] = json!({"domain": domain});
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_aesthetic_input_range(
        &self,
        vl_spec: &mut Value,
        aesthetic: &str,
        domain: Vec<Value>,
    ) -> Result<()> {
        let domain_json = json!(domain);

        // Apply to encoding if present
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(aes_enc) = encoding.get_mut(aesthetic) {
                aes_enc["scale"] = json!({"domain": domain_json});
            }
        }

        // Apply to layers if present
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(aes_enc) = encoding.get_mut(aesthetic) {
                            aes_enc["scale"] = json!({"domain": domain_json});
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn is_aesthetic_name(&self, name: &str) -> bool {
        matches!(
            name,
            "x" | "y"
                | "xmin"
                | "xmax"
                | "ymin"
                | "ymax"
                | "xend"
                | "yend"
                | "color"
                | "colour"
                | "fill"
                | "stroke"
                | "opacity"
                | "size"
                | "shape"
                | "linetype"
                | "linewidth"
                | "width"
                | "height"
                | "label"
                | "family"
                | "fontface"
                | "hjust"
                | "vjust"
        )
    }

    /// Build detail encoding from partition_by columns
    /// Maps partition_by columns to Vega-Lite's detail channel for grouping
    fn build_detail_encoding(&self, partition_by: &[String]) -> Option<Value> {
        if partition_by.is_empty() {
            return None;
        }

        if partition_by.len() == 1 {
            // Single column: simple object
            Some(json!({
                "field": partition_by[0],
                "type": "nominal"
            }))
        } else {
            // Multiple columns: array of detail specifications
            let details: Vec<Value> = partition_by
                .iter()
                .map(|col| {
                    json!({
                        "field": col,
                        "type": "nominal"
                    })
                })
                .collect();
            Some(json!(details))
        }
    }
}

impl Writer for VegaLiteWriter {
    type Output = String;

    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<String> {
        // Validate spec before processing
        self.validate(spec)?;

        // Determine which dataset key each layer should use
        // Use layer.data_key if set (from execute.rs), otherwise use standard layer key
        let layer_data_keys: Vec<String> = spec
            .layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                layer
                    .data_key
                    .clone()
                    .unwrap_or_else(|| naming::layer_key(idx))
            })
            .collect();

        // Validate all required datasets exist and validate column references
        for (layer_idx, (layer, key)) in spec.layers.iter().zip(layer_data_keys.iter()).enumerate()
        {
            let df = data.get(key).ok_or_else(|| {
                GgsqlError::WriterError(format!(
                    "Missing data source '{}' for layer {}",
                    key,
                    layer_idx + 1
                ))
            })?;
            self.validate_layer_columns(layer, df, layer_idx)?;
        }

        // Build the base Vega-Lite spec
        let mut vl_spec = json!({
            "$schema": self.schema
        });

        // Responsive plot sizing
        vl_spec["width"] = json!("container");
        vl_spec["height"] = json!("container");

        // Add title if present
        if let Some(labels) = &spec.labels {
            if let Some(title) = labels.labels.get("title") {
                vl_spec["title"] = json!(title);
            }
        }

        // Collect binned column information from spec
        let binned_columns = self.collect_binned_columns(spec);

        // Build individual datasets - convert all DataFrames to Vega-Lite format
        // For binned columns, replace center values with bin_start and add bin_end columns
        let mut individual_datasets = Map::new();

        // Track boxplot info for layers that need special handling
        // Key: layer_idx, Value: BoxplotPreparedInfo
        let mut boxplot_info: HashMap<usize, BoxplotPreparedInfo> = HashMap::new();

        // For boxplot layers, prepare summary data BEFORE unification
        // so all data (including boxplot summaries) is in the unified dataset
        for (layer_idx, layer) in spec.layers.iter().enumerate() {
            let data_key = &layer_data_keys[layer_idx];

            if layer.geom.geom_type() == GeomType::Boxplot {
                let df = data.get(data_key).ok_or_else(|| {
                    GgsqlError::WriterError(format!(
                        "Missing data source '{}' for boxplot layer {}",
                        data_key,
                        layer_idx + 1
                    ))
                })?;

                // Prepare boxplot data split by type
                let (type_datasets, grouping_cols, has_outliers) =
                    prepare_boxplot_summary(df, self, &binned_columns)?;

                // Add each type's data to individual_datasets with type-specific keys
                // Keys are like: "__ggsql_layer_0__lower_whisker", "__ggsql_layer_0__box", etc.
                for (type_name, values) in type_datasets {
                    let type_key = format!("{}{}", data_key, type_name);
                    individual_datasets.insert(type_key, json!(values));
                }

                // Store info for later use during layer rendering
                boxplot_info.insert(
                    layer_idx,
                    BoxplotPreparedInfo {
                        base_key: data_key.clone(),
                        grouping_cols,
                        has_outliers,
                    },
                );
            } else {
                // Non-boxplot layers: convert DataFrame to JSON values directly
                let df = data.get(data_key).ok_or_else(|| {
                    GgsqlError::WriterError(format!(
                        "Missing data source '{}' for layer {}",
                        data_key,
                        layer_idx + 1
                    ))
                })?;
                let values = if binned_columns.is_empty() {
                    self.dataframe_to_values(df)?
                } else {
                    self.dataframe_to_values_with_bins(df, &binned_columns)?
                };
                individual_datasets.insert(data_key.clone(), json!(values));
            }
        }

        // Unify all datasets into a single dataset with source identification
        // Each row gets a __ggsql_source__ field identifying which layer it belongs to
        let unified_data = self.unify_datasets(&individual_datasets)?;

        // Store unified dataset at GLOBAL_DATA_KEY - this is the ONLY dataset
        let mut datasets = Map::new();
        datasets.insert(naming::GLOBAL_DATA_KEY.to_string(), json!(unified_data));

        // Set top-level data reference to unified dataset
        vl_spec["data"] = json!({"name": naming::GLOBAL_DATA_KEY});

        // Build layers array
        // Each layer gets a filter transform to select its data from the unified dataset
        let mut layers = Vec::new();
        for (layer_idx, layer) in spec.layers.iter().enumerate() {
            let data_key = &layer_data_keys[layer_idx];
            let df = data.get(data_key).unwrap();

            // Layer spec without per-layer data reference (uses unified top-level data)
            let mut layer_spec = json!({
                "mark": self.geom_to_mark(&layer.geom)
            });

            // Build transform array for this layer
            // Always starts with a filter to select this layer's data from unified dataset
            let mut transforms: Vec<Value> = Vec::new();

            // Add source filter transform (EXCEPT for boxplot - it adds its own type-specific filters)
            // Filter: {"field": "__ggsql_source__", "equal": "<data_key>"}
            if layer.geom.geom_type() != GeomType::Boxplot {
                transforms.push(json!({
                    "filter": {
                        "field": naming::SOURCE_COLUMN,
                        "equal": data_key
                    }
                }));
            }

            // Set transform array on layer spec
            layer_spec["transform"] = json!(transforms);

            // Build encoding for this layer
            // Track which aesthetic families have been titled to ensure only one title per family
            let mut encoding = Map::new();
            let mut titled_families: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            // Collect primary aesthetics that exist in the layer (for title handling)
            // e.g., if layer has "y", then "ymin" and "ymax" should suppress their titles
            let primary_aesthetics: std::collections::HashSet<String> = layer
                .mappings
                .aesthetics
                .keys()
                .filter(|a| GeomAesthetics::primary_aesthetic(a) == a.as_str())
                .cloned()
                .collect();

            for (aesthetic, value) in &layer.mappings.aesthetics {
                let channel_name = self.map_aesthetic_name(aesthetic);
                let channel_encoding = self.build_encoding_channel(
                    aesthetic,
                    value,
                    df,
                    spec,
                    &mut titled_families,
                    &primary_aesthetics,
                )?;
                encoding.insert(channel_name, channel_encoding);

                // For binned positional aesthetics (x, y), add x2/y2 channel with bin_end column
                // This enables proper bin width rendering in Vega-Lite
                if matches!(aesthetic.as_str(), "x" | "y")
                    && self.is_binned_aesthetic(aesthetic, spec)
                {
                    if let AestheticValue::Column { name: col, .. } = value {
                        let end_col = naming::bin_end_column(col);
                        let end_channel = format!("{}2", aesthetic); // "x2" or "y2"
                        encoding.insert(end_channel, json!({"field": end_col}));
                    }
                }
            }

            // Also add aesthetic parameters from SETTING as literal encodings
            // (e.g., SETTING color => 'red' becomes {"color": {"value": "red"}})
            // Only parameters that are supported aesthetics for this geom type are included
            let supported_aesthetics = layer.geom.aesthetics().supported;
            for (param_name, param_value) in &layer.parameters {
                if supported_aesthetics.contains(&param_name.as_str()) {
                    let channel_name = self.map_aesthetic_name(param_name);
                    // Only add if not already set by MAPPING (MAPPING takes precedence)
                    if !encoding.contains_key(&channel_name) {
                        // Convert size and linewidth from points to Vega-Lite units
                        let converted_value = match (param_name.as_str(), param_value) {
                            // Size: interpret as radius in points, convert to area in pixels²
                            ("size", ParameterValue::Number(n)) => json!(n * n * POINTS_TO_AREA),
                            // Linewidth: interpret as width in points, convert to pixels
                            ("linewidth", ParameterValue::Number(n)) => json!(n * POINTS_TO_PIXELS),
                            // Other aesthetics: pass through unchanged
                            _ => param_value.to_json(),
                        };
                        encoding.insert(channel_name, json!({"value": converted_value}));
                    }
                }
            }

            // Add detail encoding for partition_by columns (grouping)
            if let Some(detail) = self.build_detail_encoding(&layer.partition_by) {
                encoding.insert("detail".to_string(), detail);
            }

            // Handle geom-specific encoding transformations
            match layer.geom.geom_type() {
                GeomType::Bar => layer_spec = render_bar(layer_spec, layer),
                GeomType::Path => render_path(&mut encoding),
                GeomType::Ribbon => render_ribbon(&mut encoding),
                GeomType::Area => render_area(&mut encoding, layer)?,
                GeomType::Polygon => layer_spec = render_polygon(layer_spec, &mut encoding),
                _ => {}
            }

            layer_spec["encoding"] = Value::Object(encoding);

            // For boxplots we use the pre-prepared data and render multiple layers
            if layer.geom.geom_type() == GeomType::Boxplot {
                let info = boxplot_info.get(&layer_idx).ok_or_else(|| {
                    GgsqlError::InternalError(format!(
                        "Missing boxplot info for layer {}",
                        layer_idx
                    ))
                })?;

                let boxplot_layers = render_boxplot(
                    layer_spec,
                    layer,
                    &info.base_key,
                    &info.grouping_cols,
                    info.has_outliers,
                )?;
                layers.extend(boxplot_layers);
            } else {
                layers.push(layer_spec);
            }
        }

        // Assign datasets to vl_spec - there should be exactly one unified dataset
        vl_spec["datasets"] = Value::Object(datasets);

        vl_spec["layer"] = json!(layers);

        // Apply coordinate transforms (flip, polar, cartesian limits)
        // This must happen AFTER layers are built since transforms modify layer encodings
        let first_df = data.get(&layer_data_keys[0]).unwrap();
        self.apply_coord_transforms(spec, first_df, &mut vl_spec)?;

        // Handle faceting if present
        // With unified data, faceting works regardless of layer data sources
        if let Some(facet) = &spec.facet {
            // Use the unified global dataset for faceting
            let facet_data = data.get(&layer_data_keys[0]).unwrap();

            use crate::plot::Facet;
            match facet {
                Facet::Wrap { variables, .. } => {
                    if !variables.is_empty() {
                        let field_type = self.infer_field_type(facet_data, &variables[0]);
                        vl_spec["facet"] = json!({
                            "field": variables[0],
                            "type": field_type,
                        });

                        // Move layer into spec (data reference stays at top level)
                        let mut spec_inner = json!({});
                        if let Some(layer) = vl_spec.get("layer") {
                            spec_inner["layer"] = layer.clone();
                        }

                        vl_spec["spec"] = spec_inner;
                        vl_spec.as_object_mut().unwrap().remove("layer");
                    }
                }
                Facet::Grid { rows, cols, .. } => {
                    let mut facet_spec = Map::new();
                    if !rows.is_empty() {
                        let field_type = self.infer_field_type(facet_data, &rows[0]);
                        facet_spec.insert(
                            "row".to_string(),
                            json!({"field": rows[0], "type": field_type}),
                        );
                    }
                    if !cols.is_empty() {
                        let field_type = self.infer_field_type(facet_data, &cols[0]);
                        facet_spec.insert(
                            "column".to_string(),
                            json!({"field": cols[0], "type": field_type}),
                        );
                    }
                    vl_spec["facet"] = Value::Object(facet_spec);

                    // Move layer into spec (data reference stays at top level)
                    let mut spec_inner = json!({});
                    if let Some(layer) = vl_spec.get("layer") {
                        spec_inner["layer"] = layer.clone();
                    }

                    vl_spec["spec"] = spec_inner;
                    vl_spec.as_object_mut().unwrap().remove("layer");
                }
            }
        }

        serde_json::to_string_pretty(&vl_spec).map_err(|e| {
            GgsqlError::WriterError(format!("Failed to serialize Vega-Lite JSON: {}", e))
        })
    }

    fn validate(&self, spec: &Plot) -> Result<()> {
        // Check that we have at least one layer
        if spec.layers.is_empty() {
            return Err(GgsqlError::ValidationError(
                "VegaLiteWriter requires at least one layer".to_string(),
            ));
        }

        // Validate each layer
        for layer in &spec.layers {
            // Check required aesthetics
            layer.validate_required_aesthetics().map_err(|e| {
                GgsqlError::ValidationError(format!("Layer validation failed: {}", e))
            })?;

            // Check SETTING parameters are valid for this geom
            layer.validate_settings().map_err(|e| {
                GgsqlError::ValidationError(format!("Layer validation failed: {}", e))
            })?;
        }

        Ok(())
    }
}

fn render_bar(mut spec: Value, layer: &Layer) -> Value {
    let width = match layer.parameters.get("width") {
        Some(ParameterValue::Number(w)) => *w,
        _ => 0.9,
    };
    spec["mark"] = json!({
        "type": "bar",
        "width": {"band": width},
        "clip": true
    });
    spec
}

fn render_path(encoding: &mut Map<String, Value>) {
    // Use the natural data order
    encoding.insert("order".to_string(), json!({"value": Value::Null}));
}

fn render_polygon(mut spec: Value, encoding: &mut Map<String, Value>) -> Value {
    // Polygon needs both `fill` and `stroke` independently, but map_aesthetic_name()
    // converts fill → color (which works for most geoms). For closed line marks,
    // we need actual `fill` and `stroke` channels, so we undo the mapping here.
    if let Some(color) = encoding.remove("color") {
        encoding.insert("fill".to_string(), color);
    }
    // Use the natural data order
    encoding.insert("order".to_string(), json!({"value": Value::Null}));
    spec["mark"] = json!({
        "type": "line",
        "interpolate": "linear-closed", // This closes the path
        "fill": "#888888", // default values
        "stroke": "#888888"
    });
    spec
}

fn render_ribbon(encoding: &mut Map<String, Value>) {
    if let Some(ymax) = encoding.remove("ymax") {
        encoding.insert("y".to_string(), ymax);
    }
    if let Some(ymin) = encoding.remove("ymin") {
        encoding.insert("y2".to_string(), ymin);
    }
}

fn render_area(encoding: &mut Map<String, Value>, layer: &Layer) -> Result<()> {
    if let Some(mut y) = encoding.remove("y") {
        let stack_value;
        if let Some(ParameterValue::String(stack)) = layer.parameters.get("stacking") {
            stack_value = match stack.as_str() {
                "on" => json!("zero"),
                "off" => Value::Null,
                "fill" => json!("normalize"),
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Area layer's `stacking` must be \"on\", \"off\" or \"fill\", not \"{}\"",
                        stack
                    )));
                }
            }
        } else {
            stack_value = Value::Null
        }
        y["stack"] = stack_value;
        encoding.insert("y".to_string(), y);
    }
    Ok(())
}

/// Info about prepared boxplot data for a layer
struct BoxplotPreparedInfo {
    /// Base key for the layer (e.g., "__ggsql_layer_0__")
    base_key: String,
    /// Grouping column names
    grouping_cols: Vec<String>,
    /// Whether there are any outliers
    has_outliers: bool,
}

/// Prepare boxplot data by splitting into type-specific datasets (no pivot).
///
/// Returns a HashMap of type_suffix -> data_values, plus grouping_cols and has_outliers.
/// Type suffixes are: "lower_whisker", "upper_whisker", "box", "median", "outlier"
#[allow(clippy::type_complexity)]
fn prepare_boxplot_summary(
    data: &DataFrame,
    writer: &VegaLiteWriter,
    binned_columns: &HashMap<String, Vec<f64>>,
) -> Result<(HashMap<String, Vec<Value>>, Vec<String>, bool)> {
    let type_col = naming::aesthetic_column("type");
    let type_col = type_col.as_str();
    let value_col = naming::aesthetic_column("y");
    let value_col = value_col.as_str();
    let value2_col = naming::aesthetic_column("y2");
    let value2_col = value2_col.as_str();

    // Find grouping columns (all columns except type, value, value2)
    let grouping_cols: Vec<String> = data
        .get_column_names()
        .iter()
        .filter(|&col| {
            col.as_str() != type_col && col.as_str() != value_col && col.as_str() != value2_col
        })
        .map(|s| s.to_string())
        .collect();

    // Get the type column for filtering
    let type_series = data
        .column(type_col)
        .and_then(|s| s.str())
        .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

    // Check for outliers
    let has_outliers = type_series.equal("outlier").any();

    // Split data by type into separate datasets
    let mut type_datasets: HashMap<String, Vec<Value>> = HashMap::new();

    for type_name in &["lower_whisker", "upper_whisker", "box", "median", "outlier"] {
        let mask = type_series.equal(*type_name);
        let filtered = data
            .filter(&mask)
            .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

        // Skip empty datasets (e.g., no outliers)
        if filtered.height() == 0 {
            continue;
        }

        // Drop the type column since type is now encoded in the source key
        let filtered = filtered
            .drop(type_col)
            .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

        let values = if binned_columns.is_empty() {
            writer.dataframe_to_values(&filtered)?
        } else {
            writer.dataframe_to_values_with_bins(&filtered, binned_columns)?
        };

        type_datasets.insert(type_name.to_string(), values);
    }

    Ok((type_datasets, grouping_cols, has_outliers))
}

/// Render boxplot layers using filter transforms on the unified dataset.
///
/// Creates 5 layers: outliers (optional), lower whiskers, upper whiskers, box, median line.
/// All layers use filter transforms to select their data from the unified dataset.
/// Data is in long format with type-specific source keys (e.g., "__ggsql_layer_0__lower_whisker").
fn render_boxplot(
    prototype: Value,
    layer: &Layer,
    base_key: &str,
    grouping_cols: &[String],
    has_outliers: bool,
) -> Result<Vec<Value>> {
    let mut layers: Vec<Value> = Vec::new();

    let value_col = naming::aesthetic_column("y");
    let value2_col = naming::aesthetic_column("y2");

    let x_col = layer
        .mappings
        .get("x")
        .and_then(|x| x.column_name())
        .ok_or_else(|| {
            GgsqlError::WriterError("Failed to find column for 'x' aesthetic".to_string())
        })?;
    let y_col = layer
        .mappings
        .get("y")
        .and_then(|y| y.column_name())
        .ok_or_else(|| {
            GgsqlError::WriterError("Failed to find column for 'y' aesthetic".to_string())
        })?;

    // Set orientation
    let is_horizontal = x_col == value_col;
    let group_col = if is_horizontal { y_col } else { x_col };
    let offset = if is_horizontal { "yOffset" } else { "xOffset" };
    let value_var1 = if is_horizontal { "x" } else { "y" };
    let value_var2 = if is_horizontal { "x2" } else { "y2" };

    // Find dodge groups (grouping cols minus the axis group col)
    let dodge_groups: Vec<&str> = grouping_cols
        .iter()
        .filter(|col| col.as_str() != group_col)
        .map(|s| s.as_str())
        .collect();

    // Get width parameter
    let mut width = 0.9;
    if let Some(ParameterValue::Number(num)) = layer.parameters.get("width") {
        width = *num;
    }

    // Default styling
    let default_stroke = "black";
    let default_fill = "#FFFFFF00";
    let default_linewidth = 1.0;

    // Helper to create filter transform for source selection
    let make_source_filter = |type_suffix: &str| -> Value {
        let source_key = format!("{}{}", base_key, type_suffix);
        json!({
            "filter": {
                "field": naming::SOURCE_COLUMN,
                "equal": source_key
            }
        })
    };

    // Helper to create a layer with source filter and mark
    let create_layer = |proto: &Value, type_suffix: &str, mark: Value| -> Value {
        let mut layer_spec = proto.clone();
        let existing_transforms = layer_spec
            .get("transform")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        let mut new_transforms = vec![make_source_filter(type_suffix)];
        new_transforms.extend(existing_transforms);
        layer_spec["transform"] = json!(new_transforms);
        layer_spec["mark"] = mark;
        layer_spec
    };

    // Create outlier points layer (if there are outliers)
    if has_outliers {
        let mut points = create_layer(
            &prototype,
            "outlier",
            json!({
                "type": "point",
                "stroke": default_stroke,
                "strokeWidth": default_linewidth
            }),
        );
        if points["encoding"].get("color").is_some() {
            points["mark"]["filled"] = json!(true);
        }

        // Add dodging offset
        if !dodge_groups.is_empty() {
            points["encoding"][offset] = json!({"field": dodge_groups[0]});
        }

        layers.push(points);
    }

    // Clone prototype without size/shape (these apply only to points)
    let mut summary_prototype = prototype.clone();
    if let Some(Value::Object(ref mut encoding)) = summary_prototype.get_mut("encoding") {
        encoding.remove("size");
        encoding.remove("shape");
    }

    // Build encoding templates for y and y2 fields
    // y keeps its title (from original column name or explicit label)
    // y2 gets title: null to prevent Vega-Lite from combining both into the axis title
    let mut y_encoding = summary_prototype["encoding"][value_var1].clone();
    y_encoding["field"] = json!(value_col);
    let mut y2_encoding = summary_prototype["encoding"][value_var1].clone();
    y2_encoding["field"] = json!(value2_col);
    y2_encoding["title"] = Value::Null; // Suppress y2 title to prevent "y, y2" axis label

    // Lower whiskers (rule from y to y2, where y=q1 and y2=lower)
    let mut lower_whiskers = create_layer(
        &summary_prototype,
        "lower_whisker",
        json!({
            "type": "rule",
            "stroke": default_stroke,
            "size": default_linewidth
        }),
    );

    // Handle strokeWidth -> size for rule marks
    if let Some(linewidth) = lower_whiskers["encoding"].get("strokeWidth").cloned() {
        lower_whiskers["encoding"]["size"] = linewidth;
        if let Some(Value::Object(ref mut encoding)) = lower_whiskers.get_mut("encoding") {
            encoding.remove("strokeWidth");
        }
    }

    lower_whiskers["encoding"][value_var1] = y_encoding.clone();
    lower_whiskers["encoding"][value_var2] = y2_encoding.clone();

    // Upper whiskers (rule from y to y2, where y=q3 and y2=upper)
    let mut upper_whiskers = create_layer(
        &summary_prototype,
        "upper_whisker",
        json!({
            "type": "rule",
            "stroke": default_stroke,
            "size": default_linewidth
        }),
    );

    // Handle strokeWidth -> size for rule marks
    if let Some(linewidth) = upper_whiskers["encoding"].get("strokeWidth").cloned() {
        upper_whiskers["encoding"]["size"] = linewidth;
        if let Some(Value::Object(ref mut encoding)) = upper_whiskers.get_mut("encoding") {
            encoding.remove("strokeWidth");
        }
    }

    upper_whiskers["encoding"][value_var1] = y_encoding.clone();
    upper_whiskers["encoding"][value_var2] = y2_encoding.clone();

    // Box (bar from y to y2, where y=q1 and y2=q3)
    let mut box_part = create_layer(
        &summary_prototype,
        "box",
        json!({
            "type": "bar",
            "width": {"band": width},
            "align": "center",
            "stroke": default_stroke,
            "color": default_fill,
            "strokeWidth": default_linewidth
        }),
    );
    box_part["encoding"][value_var1] = y_encoding.clone();
    box_part["encoding"][value_var2] = y2_encoding.clone();

    // Median line (tick at y, where y=median)
    let mut median_line = create_layer(
        &summary_prototype,
        "median",
        json!({
            "type": "tick",
            "stroke": default_stroke,
            "width": {"band": width},
            "align": "center",
            "strokeWidth": default_linewidth
        }),
    );
    median_line["encoding"][value_var1] = y_encoding;

    // Add dodging to all summary layers
    if !dodge_groups.is_empty() {
        let offset_val = json!({"field": dodge_groups[0]});
        lower_whiskers["encoding"][offset] = offset_val.clone();
        upper_whiskers["encoding"][offset] = offset_val.clone();
        box_part["encoding"][offset] = offset_val.clone();
        median_line["encoding"][offset] = offset_val;
    }

    layers.push(lower_whiskers);
    layers.push(upper_whiskers);
    layers.push(box_part);
    layers.push(median_line);

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{ArrayElement, Labels, Layer, OutputRange, ParameterValue, Scale};
    use std::collections::HashMap;

    /// Helper to wrap a DataFrame in a data map for testing (uses layer 0 key)
    fn wrap_data(df: DataFrame) -> HashMap<String, DataFrame> {
        wrap_data_for_layers(df, 1)
    }

    /// Helper to wrap a DataFrame for multiple layers (clones for each layer)
    fn wrap_data_for_layers(df: DataFrame, num_layers: usize) -> HashMap<String, DataFrame> {
        let mut data_map = HashMap::new();
        for i in 0..num_layers {
            data_map.insert(naming::layer_key(i), df.clone());
        }
        data_map
    }

    #[test]
    fn test_geom_to_mark_mapping() {
        let writer = VegaLiteWriter::new();
        // All marks should be objects with type and clip: true
        assert_eq!(
            writer.geom_to_mark(&Geom::point()),
            json!({"type": "point", "clip": true})
        );
        assert_eq!(
            writer.geom_to_mark(&Geom::line()),
            json!({"type": "line", "clip": true})
        );
        assert_eq!(
            writer.geom_to_mark(&Geom::bar()),
            json!({"type": "bar", "clip": true})
        );
        assert_eq!(
            writer.geom_to_mark(&Geom::area()),
            json!({"type": "area", "clip": true})
        );
        assert_eq!(
            writer.geom_to_mark(&Geom::tile()),
            json!({"type": "rect", "clip": true})
        );
    }

    #[test]
    fn test_aesthetic_name_mapping() {
        let writer = VegaLiteWriter::new();
        // Pass-through aesthetics (including fill and stroke for separate color control)
        assert_eq!(writer.map_aesthetic_name("x"), "x");
        assert_eq!(writer.map_aesthetic_name("y"), "y");
        assert_eq!(writer.map_aesthetic_name("color"), "color");
        assert_eq!(writer.map_aesthetic_name("fill"), "fill");
        assert_eq!(writer.map_aesthetic_name("stroke"), "stroke");
        assert_eq!(writer.map_aesthetic_name("opacity"), "opacity");
        assert_eq!(writer.map_aesthetic_name("size"), "size");
        assert_eq!(writer.map_aesthetic_name("shape"), "shape");
        // Mapped aesthetics
        assert_eq!(writer.map_aesthetic_name("linetype"), "strokeDash");
        assert_eq!(writer.map_aesthetic_name("linewidth"), "strokeWidth");
        assert_eq!(writer.map_aesthetic_name("label"), "text");
    }

    #[test]
    fn test_validation_requires_layers() {
        let writer = VegaLiteWriter::new();
        let spec = Plot::new();
        assert!(writer.validate(&spec).is_err());
    }

    #[test]
    fn test_simple_point_spec() {
        let writer = VegaLiteWriter::new();

        // Create a simple spec
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Create simple DataFrame
        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify structure (now uses layer array and datasets)
        assert_eq!(vl_spec["$schema"], writer.schema);
        assert!(vl_spec["layer"].is_array());
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "point");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
        assert!(vl_spec["datasets"][naming::GLOBAL_DATA_KEY].is_array());
        assert_eq!(
            vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert!(vl_spec["layer"][0]["encoding"]["x"].is_object());
        assert!(vl_spec["layer"][0]["encoding"]["y"].is_object());
    }

    #[test]
    fn test_with_title() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("title".to_string(), "My Chart".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01-01", "2024-01-02"],
            "value" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "My Chart");
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "line");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
    }

    #[test]
    fn test_literal_color() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(ParameterValue::String("blue".to_string())),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["color"]["value"], "blue");
    }

    #[test]
    fn test_missing_column_error() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("foo".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let result = writer.write(&spec, &wrap_data(df));
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Column 'foo'"));
        assert!(err_msg.contains("does not exist"));
        assert!(err_msg.contains("Available columns: x, y"));
    }

    #[test]
    fn test_missing_column_in_multi_layer() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // First layer is valid
        let layer1 = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer1);

        // Second layer references non-existent column
        let layer2 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("missing_col".to_string()),
            );
        spec.layers.push(layer2);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let result = writer.write(&spec, &wrap_data_for_layers(df, 2));
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Column 'missing_col'"));
        assert!(err_msg.contains("layer 2"));
    }

    // ========================================
    // Comprehensive Grammar Coverage Tests
    // ========================================

    #[test]
    fn test_all_basic_geom_types() {
        let writer = VegaLiteWriter::new();

        let geoms = vec![
            (Geom::point(), "point"),
            (Geom::line(), "line"),
            (Geom::path(), "line"),
            (Geom::bar(), "bar"),
            (Geom::area(), "area"),
            (Geom::tile(), "rect"),
            (Geom::ribbon(), "area"),
        ];

        for (geom, expected_mark) in geoms {
            let mut spec = Plot::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                )
                .with_aesthetic(
                    "ymin".to_string(),
                    AestheticValue::standard_column("ymin".to_string()),
                )
                .with_aesthetic(
                    "ymax".to_string(),
                    AestheticValue::standard_column("ymax".to_string()),
                );
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2, 3],
                "y" => &[4, 5, 6],
                "ymin" => &[3, 4, 5],
                "ymax" => &[5, 6, 7],
            }
            .unwrap();

            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            // Handle both string marks and object marks (e.g., Bar has {"type": "bar", "width": ...})
            let mark_type = vl_spec["layer"][0]["mark"]
                .as_str()
                .or_else(|| vl_spec["layer"][0]["mark"]["type"].as_str())
                .unwrap();
            assert_eq!(mark_type, expected_mark, "Failed for geom: {:?}", geom);
        }
    }

    #[test]
    fn test_statistical_geom_types() {
        let writer = VegaLiteWriter::new();

        let geoms = vec![
            (Geom::histogram(), "bar"),
            (Geom::density(), "area"),
            // (Geom::boxplot(), "boxplot"), // Boxplot produces several layers
        ];

        for (geom, expected_mark) in geoms {
            let mut spec = Plot::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                );
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2, 3],
                "y" => &[4, 5, 6],
            }
            .unwrap();

            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            assert_eq!(
                vl_spec["layer"][0]["mark"]["type"].as_str().unwrap(),
                expected_mark
            );
            assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
        }
    }

    #[test]
    fn test_text_geom_types() {
        let writer = VegaLiteWriter::new();

        for geom in [Geom::text(), Geom::label()] {
            let mut spec = Plot::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                );
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2],
                "y" => &[3, 4],
            }
            .unwrap();

            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            assert_eq!(
                vl_spec["layer"][0]["mark"]["type"].as_str().unwrap(),
                "text"
            );
            assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
        }
    }

    #[test]
    fn test_color_aesthetic_column() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "A"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["field"],
            "category"
        );
        assert_eq!(vl_spec["layer"][0]["encoding"]["color"]["type"], "nominal");
    }

    #[test]
    fn test_size_aesthetic_column() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["size"]["field"], "value");
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["size"]["type"],
            "quantitative"
        );
    }

    #[test]
    fn test_fill_aesthetic_mapping() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("region".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "category" => &["A", "B"],
            "value" => &[10, 20],
            "region" => &["US", "EU"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // 'fill' maps directly to Vega-Lite's 'fill' channel
        assert_eq!(vl_spec["layer"][0]["encoding"]["fill"]["field"], "region");
    }

    #[test]
    fn test_multiple_aesthetics() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_aesthetic(
                "shape".to_string(),
                AestheticValue::standard_column("type".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
            "type" => &["T1", "T2", "T1"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["field"], "x");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["field"], "y");
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["field"],
            "category"
        );
        assert_eq!(vl_spec["layer"][0]["encoding"]["size"]["field"], "value");
        assert_eq!(vl_spec["layer"][0]["encoding"]["shape"]["field"], "type");
    }

    #[test]
    fn test_literal_number_value() {
        // Test that numeric literals pass through unchanged for aesthetics that
        // don't have special unit conversion (like opacity)
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "opacity".to_string(),
                AestheticValue::Literal(ParameterValue::Number(0.5)),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Opacity passes through unchanged
        assert_eq!(vl_spec["layer"][0]["encoding"]["opacity"]["value"], 0.5);
    }

    #[test]
    fn test_size_literal_radius_to_area_conversion() {
        // Test that size literals are converted from radius (points) to area (pixels²)
        // Formula: area = radius² × π × (96/72)²
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                // Radius of 5 points
                AestheticValue::Literal(ParameterValue::Number(5.0)),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Expected: 5² × π × (96/72)² = 25 × 5.585... ≈ 139.63
        let size_value = vl_spec["layer"][0]["encoding"]["size"]["value"]
            .as_f64()
            .unwrap();
        let expected = 5.0 * 5.0 * POINTS_TO_AREA;
        assert!(
            (size_value - expected).abs() < 0.01,
            "Size conversion: expected {:.2}, got {:.2}",
            expected,
            size_value
        );
    }

    #[test]
    fn test_linewidth_literal_points_to_pixels_conversion() {
        // Test that linewidth literals are converted from points to pixels
        // Formula: pixels = points × (96/72)
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "linewidth".to_string(),
                // Width of 3 points
                AestheticValue::Literal(ParameterValue::Number(3.0)),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Expected: 3 × (96/72) = 3 × 1.333... = 4.0
        // linewidth maps to strokeWidth in Vega-Lite
        let width_value = vl_spec["layer"][0]["encoding"]["strokeWidth"]["value"]
            .as_f64()
            .unwrap();
        let expected = 3.0 * POINTS_TO_PIXELS;
        assert!(
            (width_value - expected).abs() < 0.01,
            "Linewidth conversion: expected {:.2}, got {:.2}",
            expected,
            width_value
        );
    }

    #[test]
    fn test_literal_boolean_value() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "linetype".to_string(),
                AestheticValue::Literal(ParameterValue::Boolean(true)),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // linetype is mapped to strokeDash in Vega-Lite
        assert_eq!(vl_spec["layer"][0]["encoding"]["strokeDash"]["value"], true);
    }

    #[test]
    fn test_multi_layer_composition() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // First layer: line
        let layer1 = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer1);

        // Second layer: points
        let layer2 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(ParameterValue::String("red".to_string())),
            );
        spec.layers.push(layer2);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data_for_layers(df, 2)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have layer array
        assert!(vl_spec["layer"].is_array());
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        // Check first layer
        assert_eq!(layers[0]["mark"]["type"], "line");
        assert_eq!(layers[0]["mark"]["clip"], true);
        assert_eq!(layers[0]["encoding"]["x"]["field"], "x");
        assert_eq!(layers[0]["encoding"]["y"]["field"], "y");

        // Check second layer
        assert_eq!(layers[1]["mark"]["type"], "point");
        assert_eq!(layers[1]["mark"]["clip"], true);
        assert_eq!(layers[1]["encoding"]["color"]["value"], "red");
    }

    #[test]
    fn test_three_layer_composition() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // Layer 1: area
        spec.layers.push(
            Layer::new(Geom::area())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                ),
        );

        // Layer 2: line
        spec.layers.push(
            Layer::new(Geom::line())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                ),
        );

        // Layer 3: points
        spec.layers.push(
            Layer::new(Geom::point())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("x".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column("y".to_string()),
                ),
        );

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data_for_layers(df, 3)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0]["mark"]["type"], "area");
        assert_eq!(layers[0]["mark"]["clip"], true);
        assert_eq!(layers[1]["mark"]["type"], "line");
        assert_eq!(layers[1]["mark"]["clip"], true);
        assert_eq!(layers[2]["mark"]["type"], "point");
        assert_eq!(layers[2]["mark"]["clip"], true);
    }

    #[test]
    fn test_label_title() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("title".to_string(), "Test Plot".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "Test Plot");
    }

    #[test]
    fn test_label_axis_titles() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("revenue".to_string()),
            );
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("x".to_string(), "Date".to_string());
        labels
            .labels
            .insert("y".to_string(), "Revenue ($M)".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01", "2024-02", "2024-03"],
            "revenue" => &["100", "150", "200"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["title"], "Date");
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["y"]["title"],
            "Revenue ($M)"
        );
    }

    #[test]
    fn test_label_title_and_axes() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("title".to_string(), "Sales by Category".to_string());
        labels
            .labels
            .insert("x".to_string(), "Product Category".to_string());
        labels
            .labels
            .insert("y".to_string(), "Sales Volume".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 15],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "Sales by Category");
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["title"],
            "Product Category"
        );
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["y"]["title"],
            "Sales Volume"
        );
    }

    #[test]
    fn test_numeric_type_inference_integers() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_nominal_type_inference_strings() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "nominal");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_numeric_string_type_inference() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &["1", "2", "3"],
            "y" => &["4.5", "5.5", "6.5"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Numeric strings should be inferred as quantitative
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");

        // Values should be converted to numbers in JSON
        let data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data[0]["x"], 1.0);
        assert_eq!(data[0]["y"], 4.5);
    }

    #[test]
    fn test_data_conversion_all_types() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("int_col".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("float_col".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "int_col" => &[1, 2, 3],
            "float_col" => &[1.5, 2.5, 3.5],
            "string_col" => &["a", "b", "c"],
            "bool_col" => &[true, false, true],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data.len(), 3);

        // Check first row
        assert_eq!(data[0]["int_col"], 1);
        assert_eq!(data[0]["float_col"], 1.5);
        assert_eq!(data[0]["string_col"], "a");
        assert_eq!(data[0]["bool_col"], true);
    }

    #[test]
    fn test_empty_dataframe() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[] as &[i32],
            "y" => &[] as &[i32],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_large_dataset() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Create dataset with 100 rows
        let x_vals: Vec<i32> = (1..=100).collect();
        let y_vals: Vec<i32> = (1..=100).map(|i| i * 2).collect();

        let df = df! {
            "x" => x_vals,
            "y" => y_vals,
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data.len(), 100);
        assert_eq!(data[0]["x"], 1);
        assert_eq!(data[0]["y"], 2);
        assert_eq!(data[99]["x"], 100);
        assert_eq!(data[99]["y"], 200);
    }

    // ========================================
    // COORD Clause Tests
    // ========================================

    #[test]
    fn test_coord_cartesian_xlim() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD cartesian with xlim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that x scale has domain set
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["scale"]["domain"],
            json!([0.0, 100.0])
        );
    }

    #[test]
    fn test_coord_cartesian_ylim() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD cartesian with ylim
        let mut properties = HashMap::new();
        properties.insert(
            "ylim".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(-10.0),
                ArrayElement::Number(50.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that y scale has domain set
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
            json!([-10.0, 50.0])
        );
    }

    #[test]
    fn test_coord_cartesian_xlim_ylim() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD cartesian with both xlim and ylim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]),
        );
        properties.insert(
            "ylim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(200.0)]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[50, 100, 150],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check both domains
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["scale"]["domain"],
            json!([0.0, 100.0])
        );
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
            json!([0.0, 200.0])
        );
    }

    #[test]
    fn test_coord_cartesian_reversed_limits_auto_swap() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD with reversed xlim (should auto-swap)
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(100.0), ArrayElement::Number(0.0)]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should be swapped to [0, 100]
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["scale"]["domain"],
            json!([0.0, 100.0])
        );
    }

    #[test]
    fn test_coord_cartesian_aesthetic_input_range() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD with color domain
        let mut properties = HashMap::new();
        properties.insert(
            "color".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("A".to_string()),
                ArrayElement::String("B".to_string()),
                ArrayElement::String("C".to_string()),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "A"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that color scale has domain set
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["scale"]["domain"],
            json!(["A", "B", "C"])
        );
    }

    #[test]
    fn test_coord_cartesian_multi_layer() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // First layer: line
        let layer1 = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer1);

        // Second layer: points
        let layer2 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer2);

        // Add COORD with xlim and ylim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(10.0)]),
        );
        properties.insert(
            "ylim".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(-5.0), ArrayElement::Number(5.0)]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data_for_layers(df, 2)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that both layers have the limits applied
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        for layer in layers {
            assert_eq!(
                layer["encoding"]["x"]["scale"]["domain"],
                json!([0.0, 10.0])
            );
            assert_eq!(
                layer["encoding"]["y"]["scale"]["domain"],
                json!([-5.0, 5.0])
            );
        }
    }

    #[test]
    fn test_coord_flip_single_layer() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add custom axis labels
        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("x".to_string(), "Category".to_string());
        labels.labels.insert("y".to_string(), "Value".to_string());
        spec.labels = Some(labels);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // After flip: x should have "value" field, y should have "category" field
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["field"], "value");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["field"], "category");

        // But titles should preserve original aesthetic names (ggplot2 style)
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["title"], "Value");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["title"], "Category");
    }

    #[test]
    fn test_coord_flip_multi_layer() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // First layer: bar
        let layer1 = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer1);

        // Second layer: point
        let layer2 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer2);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data_for_layers(df, 2)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check both layers have flipped encodings
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        for layer in layers {
            assert_eq!(layer["encoding"]["x"]["field"], "value");
            assert_eq!(layer["encoding"]["y"]["field"], "category");
        }
    }

    #[test]
    fn test_coord_flip_preserves_other_aesthetics() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check x and y are flipped
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["field"], "y");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["field"], "x");

        // Check color and size are unchanged
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["field"],
            "category"
        );
        assert_eq!(vl_spec["layer"][0]["encoding"]["size"]["field"], "value");
    }

    #[test]
    fn test_coord_polar_basic_pie_chart() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD polar (defaults to theta = y)
        spec.coord = Some(Coord {
            coord_type: CoordType::Polar,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Bar in polar should become arc
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "arc");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);

        // y should be mapped to theta
        assert!(vl_spec["layer"][0]["encoding"]["theta"].is_object());
        assert_eq!(vl_spec["layer"][0]["encoding"]["theta"]["field"], "value");

        // x should be removed from positional encoding
        assert!(
            vl_spec["layer"][0]["encoding"]["x"].is_null()
                || !vl_spec["layer"][0]["encoding"]
                    .as_object()
                    .unwrap()
                    .contains_key("x")
        );

        // x should be mapped to color (for category differentiation)
        assert!(vl_spec["layer"][0]["encoding"]["color"].is_object());
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["field"],
            "category"
        );
    }

    #[test]
    fn test_coord_polar_with_theta_property() {
        use crate::plot::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add COORD polar with explicit theta = y
        let mut properties = HashMap::new();
        properties.insert("theta".to_string(), ParameterValue::String("y".to_string()));
        spec.coord = Some(Coord {
            coord_type: CoordType::Polar,
            properties,
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should produce same result as default
        assert_eq!(vl_spec["layer"][0]["mark"]["type"], "arc");
        assert_eq!(vl_spec["layer"][0]["mark"]["clip"], true);
        assert_eq!(vl_spec["layer"][0]["encoding"]["theta"]["field"], "value");
    }

    #[test]
    fn test_date_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Create DataFrame with Date type
        let dates = Series::new("date".into(), &[0i32, 1, 2]) // Days since epoch
            .cast(&DataType::Date)
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![dates.into(), values.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that dates are formatted as ISO strings in data
        let data_values = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data_values[0]["date"], "1970-01-01");
        assert_eq!(data_values[1]["date"], "1970-01-02");
        assert_eq!(data_values[2]["date"], "1970-01-03");
    }

    #[test]
    fn test_datetime_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("datetime".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Create DataFrame with Datetime type (microseconds since epoch)
        let datetimes = Series::new("datetime".into(), &[0i64, 1_000_000, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![datetimes.into(), values.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that datetimes are formatted as ISO strings in data
        let data_values = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data_values[0]["datetime"], "1970-01-01T00:00:00.000Z");
        assert_eq!(data_values[1]["datetime"], "1970-01-01T00:00:01.000Z");
        assert_eq!(data_values[2]["datetime"], "1970-01-01T00:00:02.000Z");
    }

    #[test]
    fn test_time_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("time".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Create DataFrame with Time type (nanoseconds since midnight)
        let times = Series::new("time".into(), &[0i64, 3_600_000_000_000, 7_200_000_000_000])
            .cast(&DataType::Time)
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![times.into(), values.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that times are formatted as ISO time strings in data
        let data_values = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data_values[0]["time"], "00:00:00.000");
        assert_eq!(data_values[1]["time"], "01:00:00.000");
        assert_eq!(data_values[2]["time"], "02:00:00.000");
    }

    #[test]
    fn test_automatic_temporal_type_inference() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("revenue".to_string()),
            );
        spec.layers.push(layer);

        // Create DataFrame with Date type - NO explicit SCALE x SETTING type => 'date' needed!
        let dates = Series::new("date".into(), &[0i32, 1, 2, 3, 4])
            .cast(&DataType::Date)
            .unwrap();
        let revenue = Series::new("revenue".into(), &[100, 120, 110, 130, 125]);
        let df = DataFrame::new(vec![dates.into(), revenue.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // CRITICAL TEST: x-axis should automatically be inferred as "temporal" type
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");

        // Dates should be formatted as ISO strings
        let data_values = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data_values[0]["date"], "1970-01-01");
        assert_eq!(data_values[1]["date"], "1970-01-02");
    }

    #[test]
    fn test_datetime_automatic_temporal_inference() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::area())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("timestamp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Create DataFrame with Datetime type
        let timestamps = Series::new("timestamp".into(), &[0i64, 86_400_000_000, 172_800_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let values = Series::new("value".into(), &[50, 75, 60]);
        let df = DataFrame::new(vec![timestamps.into(), values.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should automatically be inferred as "temporal" type
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "temporal");

        // Timestamps should be formatted as ISO datetime strings
        let data_values = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(data_values[0]["timestamp"], "1970-01-01T00:00:00.000Z");
        assert_eq!(data_values[1]["timestamp"], "1970-01-02T00:00:00.000Z");
        assert_eq!(data_values[2]["timestamp"], "1970-01-03T00:00:00.000Z");
    }

    // ========================================
    // PARTITION BY Tests
    // ========================================

    #[test]
    fn test_partition_by_single_column_generates_detail() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_partition_by(vec!["category".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02", "2024-01-03"]);
        let values = Series::new("value".into(), &[100, 120, 110]);
        let categories = Series::new("category".into(), &["A", "A", "B"]);
        let df = DataFrame::new(vec![dates.into(), values.into(), categories.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have detail encoding with the partition_by column (in layer[0])
        assert!(vl_spec["layer"][0]["encoding"]["detail"].is_object());
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["detail"]["field"],
            "category"
        );
        assert_eq!(vl_spec["layer"][0]["encoding"]["detail"]["type"], "nominal");
    }

    #[test]
    fn test_partition_by_multiple_columns_generates_detail_array() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_partition_by(vec!["category".to_string(), "region".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let categories = Series::new("category".into(), &["A", "B"]);
        let regions = Series::new("region".into(), &["North", "South"]);
        let df = DataFrame::new(vec![
            dates.into(),
            values.into(),
            categories.into(),
            regions.into(),
        ])
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have detail encoding as an array (in layer[0])
        assert!(vl_spec["layer"][0]["encoding"]["detail"].is_array());
        let details = vl_spec["layer"][0]["encoding"]["detail"]
            .as_array()
            .unwrap();
        assert_eq!(details.len(), 2);
        assert_eq!(details[0]["field"], "category");
        assert_eq!(details[0]["type"], "nominal");
        assert_eq!(details[1]["field"], "region");
        assert_eq!(details[1]["type"], "nominal");
    }

    #[test]
    fn test_no_partition_by_no_detail() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let df = DataFrame::new(vec![dates.into(), values.into()]).unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should NOT have detail encoding
        assert!(vl_spec["encoding"]["detail"].is_null());
    }

    #[test]
    fn test_partition_by_validation_missing_column() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_partition_by(vec!["nonexistent_column".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let df = DataFrame::new(vec![dates.into(), values.into()]).unwrap();

        let result = writer.write(&spec, &wrap_data(df));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent_column"));
        assert!(err.contains("PARTITION BY"));
    }

    #[test]
    fn test_facet_wrap_top_level() {
        use crate::plot::Facet;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);
        spec.facet = Some(Facet::Wrap {
            variables: vec!["region".to_string()],
            scales: crate::plot::FacetScales::Fixed,
        });

        let df = df! {
            "x" => &[1, 2, 3, 4],
            "y" => &[10, 20, 15, 25],
            "region" => &["North", "North", "South", "South"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify top-level faceting structure
        assert!(vl_spec["facet"].is_object(), "Should have top-level facet");
        assert_eq!(vl_spec["facet"]["field"], "region");
        assert!(
            vl_spec["data"].is_object(),
            "Should have top-level data reference"
        );
        assert_eq!(vl_spec["data"]["name"], naming::GLOBAL_DATA_KEY);
        assert!(
            vl_spec["datasets"][naming::GLOBAL_DATA_KEY].is_array(),
            "Should have datasets"
        );
        assert!(
            vl_spec["spec"]["layer"].is_array(),
            "Layer should be moved into spec"
        );

        // Layers inside spec should NOT have per-layer data entries
        assert!(
            vl_spec["spec"]["layer"][0].get("data").is_none(),
            "Faceted layers should not have per-layer data"
        );
    }

    #[test]
    fn test_facet_grid_top_level() {
        use crate::plot::Facet;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);
        spec.facet = Some(Facet::Grid {
            rows: vec!["region".to_string()],
            cols: vec!["category".to_string()],
            scales: crate::plot::FacetScales::Fixed,
        });

        let df = df! {
            "x" => &[1, 2, 3, 4],
            "y" => &[10, 20, 15, 25],
            "region" => &["North", "North", "South", "South"],
            "category" => &["A", "B", "A", "B"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify top-level faceting structure
        assert!(vl_spec["facet"].is_object(), "Should have top-level facet");
        assert_eq!(vl_spec["facet"]["row"]["field"], "region");
        assert_eq!(vl_spec["facet"]["column"]["field"], "category");
        assert!(
            vl_spec["data"].is_object(),
            "Should have top-level data reference"
        );
        assert_eq!(vl_spec["data"]["name"], naming::GLOBAL_DATA_KEY);
        assert!(
            vl_spec["datasets"][naming::GLOBAL_DATA_KEY].is_array(),
            "Should have datasets"
        );
        assert!(
            vl_spec["spec"]["layer"].is_array(),
            "Layer should be moved into spec"
        );

        // Layers inside spec should NOT have per-layer data entries
        assert!(
            vl_spec["spec"]["layer"][0].get("data").is_none(),
            "Faceted layers should not have per-layer data"
        );
    }

    #[test]
    fn test_aesthetic_in_setting_literal_encoding() {
        // Test that aesthetics in SETTING (e.g., SETTING stroke => 'red') are encoded as literals
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_parameter(
                "stroke".to_string(),
                ParameterValue::String("red".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "date" => &[1, 2, 3],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Stroke should be encoded as a literal value in the stroke channel
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["stroke"]["value"], "red",
            "SETTING stroke => 'red' should produce {{\"value\": \"red\"}} in stroke channel"
        );
    }

    #[test]
    fn test_aesthetic_in_setting_numeric_value() {
        // Test that numeric aesthetics in SETTING are encoded as literals
        // Note: size gets converted from radius (points) to area (pixels²)
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_parameter("size".to_string(), ParameterValue::Number(5.0)) // radius in points
            .with_parameter("opacity".to_string(), ParameterValue::Number(0.5));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Size is converted from radius (points) to area (pixels²)
        // Expected: 5² × π × (96/72)² ≈ 139.63
        let size_value = vl_spec["layer"][0]["encoding"]["size"]["value"]
            .as_f64()
            .unwrap();
        let expected_size = 5.0 * 5.0 * POINTS_TO_AREA;
        assert!(
            (size_value - expected_size).abs() < 0.01,
            "SETTING size => 5 should produce converted area value ~{:.2}, got {:.2}",
            expected_size,
            size_value
        );

        // Opacity passes through unchanged
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["opacity"]["value"], 0.5,
            "SETTING opacity => 0.5 should produce {{\"value\": 0.5}}"
        );
    }

    #[test]
    fn test_setting_linewidth_points_to_pixels() {
        // Test that SETTING linewidth is converted from points to pixels
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_parameter("linewidth".to_string(), ParameterValue::Number(3.0)); // 3 points
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Linewidth: 3 × (96/72) = 4.0 pixels
        let width_value = vl_spec["layer"][0]["encoding"]["strokeWidth"]["value"]
            .as_f64()
            .unwrap();
        let expected = 3.0 * POINTS_TO_PIXELS;
        assert!(
            (width_value - expected).abs() < 0.01,
            "SETTING linewidth => 3 should produce {:.2} pixels, got {:.2}",
            expected,
            width_value
        );
    }

    #[test]
    fn test_mapping_takes_precedence_over_setting() {
        // Test that MAPPING takes precedence over SETTING for the same aesthetic
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_parameter(
                "fill".to_string(),
                ParameterValue::String("red".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
            "category" => &["A", "B", "C"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Fill should be field-mapped (from MAPPING), not value (from SETTING)
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["fill"]["field"], "category",
            "MAPPING should take precedence over SETTING"
        );
        assert!(
            vl_spec["layer"][0]["encoding"]["fill"]["value"].is_null(),
            "Should not have value encoding when MAPPING is present"
        );
    }

    #[test]
    fn test_line_geom_no_order_encoding() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let mut layer = Layer::new(Geom::line());
        layer.mappings.insert(
            "x".to_string(),
            AestheticValue::standard_column("date".to_string()),
        );
        layer.mappings.insert(
            "y".to_string(),
            AestheticValue::standard_column("value".to_string()),
        );
        spec.layers.push(layer);

        let df = df! {
            "date" => &["2024-01", "2024-02", "2024-03"],
            "value" => &[10.0, 20.0, 30.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Line layer should only have filter transform (no window)
        let layer_spec = &vl_spec["layer"][0];
        let transforms = layer_spec["transform"]
            .as_array()
            .expect("Should have transforms");
        // Only filter transform, no window
        assert_eq!(
            transforms.len(),
            1,
            "Line geom should only have filter transform"
        );
        assert!(
            transforms[0].get("filter").is_some(),
            "Line geom transform should be filter only"
        );

        // Line should NOT have order encoding
        let encoding = &layer_spec["encoding"];
        assert!(
            encoding.get("order").is_none(),
            "Line geom should not have order encoding"
        );
    }

    #[test]
    fn test_variant_aesthetics_use_primary_label() {
        // Test that variant aesthetics (xmin, xmax, etc.) use the primary aesthetic's label
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::errorbar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "ymin".to_string(),
                AestheticValue::standard_column("lower".to_string()),
            )
            .with_aesthetic(
                "ymax".to_string(),
                AestheticValue::standard_column("upper".to_string()),
            );
        spec.layers.push(layer);

        // Set label only for the primary aesthetic
        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels
            .labels
            .insert("y".to_string(), "Value Range".to_string());
        labels.labels.insert("x".to_string(), "Date".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01", "2024-02", "2024-03"],
            "lower" => &[10.0, 15.0, 20.0],
            "upper" => &[20.0, 25.0, 30.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The x encoding should get the "Date" title
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["title"], "Date",
            "x should have the 'Date' title from labels"
        );

        // Only one of ymin/ymax should get the "Value Range" title (first one wins per family)
        // The other should not have a title set (prevents duplicate axis labels)
        let ymin_title = &vl_spec["layer"][0]["encoding"]["ymin"]["title"];
        let ymax_title = &vl_spec["layer"][0]["encoding"]["ymax"]["title"];

        // Exactly one should have the title, the other should be null
        let ymin_has_title = ymin_title == "Value Range";
        let ymax_has_title = ymax_title == "Value Range";

        assert!(
            ymin_has_title || ymax_has_title,
            "At least one of ymin/ymax should get the 'Value Range' title"
        );
        assert!(
            !(ymin_has_title && ymax_has_title),
            "Only one of ymin/ymax should get the title (first wins per family)"
        );
    }

    #[test]
    fn test_resolved_breaks_positional_axis_values() {
        // Test that breaks (as Array) for positional aesthetics maps to axis.values
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add a scale with breaks array for x
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[10, 50, 90],
            "y" => &[1, 2, 3],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The x encoding should have axis.values
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(axis_values.is_array(), "axis.values should be an array");
        assert_eq!(
            axis_values.as_array().unwrap().len(),
            5,
            "axis.values should have 5 elements"
        );
        assert_eq!(axis_values[0], 0.0);
        assert_eq!(axis_values[4], 100.0);
    }

    #[test]
    fn test_resolved_breaks_color_legend_values() {
        // Test that breaks (as Array) for non-positional aesthetics maps to legend.values
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("z".to_string()),
            );
        spec.layers.push(layer);

        // Add a scale with breaks array for color
        let mut scale = Scale::new("color");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(10.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(90.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "z" => &[10.0, 50.0, 90.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The color encoding should have legend.values
        let legend_values = &vl_spec["layer"][0]["encoding"]["color"]["legend"]["values"];
        assert!(legend_values.is_array(), "legend.values should be an array");
        assert_eq!(
            legend_values.as_array().unwrap().len(),
            3,
            "legend.values should have 3 elements"
        );
        assert_eq!(legend_values[0], 10.0);
        assert_eq!(legend_values[2], 90.0);
    }

    #[test]
    fn test_resolved_breaks_string_values() {
        // Test that breaks (as Array) with string values (e.g., dates) work correctly
        use crate::plot::scale::{Scale, Transform};
        use crate::plot::{ArrayElement, ParameterValue};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add a continuous scale with Date transform and breaks as string array
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::continuous());
        scale.transform = Some(Transform::date()); // Temporal transform
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("2024-01-01".to_string()),
                ArrayElement::String("2024-02-01".to_string()),
                ArrayElement::String("2024-03-01".to_string()),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "date" => &["2024-01-15", "2024-02-15", "2024-03-15"],
            "y" => &[1, 2, 3],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The x encoding should have axis.values with date strings
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(axis_values.is_array(), "axis.values should be an array");
        assert_eq!(axis_values[0], "2024-01-01");
        assert_eq!(axis_values[1], "2024-02-01");
        assert_eq!(axis_values[2], "2024-03-01");
    }

    #[test]
    fn test_find_bin_for_value() {
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        // Values in first bin [0, 10)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(0.0, &breaks),
            Some((0.0, 10.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(5.0, &breaks),
            Some((0.0, 10.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(9.99, &breaks),
            Some((0.0, 10.0))
        );

        // Values in second bin [10, 20)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(10.0, &breaks),
            Some((10.0, 20.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(15.0, &breaks),
            Some((10.0, 20.0))
        );

        // Values in last bin [20, 30] (closed on right)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(20.0, &breaks),
            Some((20.0, 30.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(25.0, &breaks),
            Some((20.0, 30.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(30.0, &breaks),
            Some((20.0, 30.0))
        );

        // Values outside all bins
        assert_eq!(VegaLiteWriter::find_bin_for_value(-1.0, &breaks), None);
        assert_eq!(VegaLiteWriter::find_bin_for_value(31.0, &breaks), None);
    }

    #[test]
    fn test_find_bin_for_value_uneven_breaks() {
        // Non-evenly-spaced breaks
        let breaks = vec![0.0, 10.0, 25.0, 100.0];

        // Value in [0, 10)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(5.0, &breaks),
            Some((0.0, 10.0))
        );

        // Value in [10, 25)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(17.5, &breaks),
            Some((10.0, 25.0))
        );

        // Value in [25, 100] (last bin, closed on right)
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(62.5, &breaks),
            Some((25.0, 100.0))
        );
        assert_eq!(
            VegaLiteWriter::find_bin_for_value(100.0, &breaks),
            Some((25.0, 100.0))
        );
    }

    #[test]
    fn test_binned_scale_adds_bin_encoding() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("temperature".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for x
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(30.0),
            ]),
        );
        spec.scales.push(scale);

        // Data with bin center values (5, 15, 25)
        let df = df! {
            "temperature" => &[5.0, 15.0, 25.0],
            "count" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The x encoding should have bin: "binned"
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x"]["bin"],
            json!("binned"),
            "Binned scale should add bin: \"binned\" to encoding"
        );

        // Should also have x2 channel for bin end
        assert!(
            vl_spec["layer"][0]["encoding"]["x2"].is_object(),
            "Binned x scale should add x2 channel"
        );
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["x2"]["field"],
            naming::bin_end_column("temperature")
        );
    }

    #[test]
    fn test_binned_scale_transforms_data() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("value".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );
        spec.scales.push(scale);

        // Data with bin center values: 5 (center of [0, 10]), 15 (center of [10, 20])
        let df = df! {
            "value" => &[5.0, 15.0],
            "count" => &[100, 200],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that data was transformed: center values replaced with bin_start
        let data = &vl_spec["datasets"][naming::GLOBAL_DATA_KEY];

        // First row: center 5 -> bin_start 0
        assert_eq!(
            data[0]["value"], 0.0,
            "Bin center should be replaced with bin_start"
        );
        // First row should have bin_end column
        assert_eq!(
            data[0][naming::bin_end_column("value")],
            10.0,
            "Should have bin_end column"
        );

        // Second row: center 15 -> bin_start 10
        assert_eq!(data[1]["value"], 10.0);
        assert_eq!(data[1][naming::bin_end_column("value")], 20.0);
    }

    #[test]
    fn test_binned_scale_sets_axis_values_from_breaks() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("temp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale with breaks (including uneven spacing)
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "temp" => &[5.0, 17.5, 62.5],
            "count" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // For binned scales with arbitrary breaks, axis.values should be set
        // to the breaks array for proper tick placement at bin edges
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Binned scale should set axis.values"
        );
        assert_eq!(axis_values[0], 0.0);
        assert_eq!(axis_values[1], 10.0);
        assert_eq!(axis_values[2], 25.0);
        assert_eq!(axis_values[3], 100.0);
    }

    #[test]
    fn test_non_binned_scale_still_sets_axis_values() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add a continuous (non-binned) scale with breaks
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::continuous());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[10, 60, 90],
            "y" => &[1, 2, 3],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // For non-binned scales, axis.values should still be set
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Non-binned scale should set axis.values"
        );
        assert_eq!(axis_values[0], 0.0);
        assert_eq!(axis_values[1], 50.0);
        assert_eq!(axis_values[2], 100.0);
    }

    #[test]
    fn test_binned_scale_oob_squish_removes_terminal_labels() {
        // When oob='squish' for binned scales, terminal break labels should be removed
        // since those bins extend to infinity
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("temp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale with breaks and oob='squish'
        // When resolved, label_mapping will have terminal breaks mapped to None
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(30.0),
            ]),
        );
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("squish".to_string()),
        );
        // Simulate what resolution does: terminal breaks are suppressed via label_mapping
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("0".to_string(), None); // First break suppressed
        label_mapping.insert("10".to_string(), Some("10".to_string()));
        label_mapping.insert("20".to_string(), Some("20".to_string()));
        label_mapping.insert("30".to_string(), None); // Last break suppressed
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "temp" => &[5.0, 15.0, 25.0],
            "count" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // With oob='squish', terminal breaks (0 and 30) should be removed
        // Only internal breaks (10, 20) should remain
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Binned scale should set axis.values"
        );
        let values = axis_values.as_array().unwrap();
        assert_eq!(
            values.len(),
            2,
            "Should have 2 values (terminal labels removed)"
        );
        assert_eq!(values[0], 10.0, "First value should be 10 (second break)");
        assert_eq!(values[1], 20.0, "Second value should be 20 (third break)");
    }

    #[test]
    fn test_binned_scale_oob_censor_keeps_all_labels() {
        // When oob='censor' (default) for binned scales, all break labels should be kept
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("temp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale with breaks and oob='censor'
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(30.0),
            ]),
        );
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("censor".to_string()),
        );
        spec.scales.push(scale);

        let df = df! {
            "temp" => &[5.0, 15.0, 25.0],
            "count" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // With oob='censor', all breaks should be kept
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Binned scale should set axis.values"
        );
        let values = axis_values.as_array().unwrap();
        assert_eq!(values.len(), 4, "Should have all 4 values");
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 10.0);
        assert_eq!(values[2], 20.0);
        assert_eq!(values[3], 30.0);
    }

    #[test]
    fn test_binned_scale_oob_squish_two_breaks_not_removed() {
        // When oob='squish' but only 2 breaks (1 bin), don't remove labels
        // since that would leave 0 labels
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("temp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("count".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale with only 2 breaks
        let mut scale = Scale::new("x");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]),
        );
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("squish".to_string()),
        );
        spec.scales.push(scale);

        let df = df! {
            "temp" => &[50.0],
            "count" => &[10],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // With only 2 breaks, both should be kept (values.len() <= 2 check)
        let axis_values = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Binned scale should set axis.values"
        );
        let values = axis_values.as_array().unwrap();
        assert_eq!(
            values.len(),
            2,
            "Should keep both values when only 2 breaks"
        );
    }

    // ========================================
    // RENAMING clause / labelExpr tests
    // ========================================

    #[test]
    fn test_scale_renaming_generates_axis_label_expr() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("cat".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("val".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with RENAMING
        let mut scale = Scale::new("x");
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("A".to_string(), Some("Alpha".to_string()));
        label_mapping.insert("B".to_string(), Some("Beta".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "cat" => &["A", "B"],
            "val" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr is generated
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        assert!(label_expr.is_string(), "axis.labelExpr should be a string");
        let expr = label_expr.as_str().unwrap();
        assert!(
            expr.contains("datum.label"),
            "labelExpr should reference datum.label"
        );
        assert!(
            expr.contains("Alpha"),
            "labelExpr should contain renamed label"
        );
        assert!(
            expr.contains("Beta"),
            "labelExpr should contain renamed label"
        );
    }

    #[test]
    fn test_scale_renaming_generates_legend_label_expr() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("cat".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with RENAMING for color (legend)
        let mut scale = Scale::new("color");
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("cat_a".to_string(), Some("Category A".to_string()));
        label_mapping.insert("cat_b".to_string(), Some("Category B".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
            "cat" => &["cat_a", "cat_b"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that legend.labelExpr is generated
        let label_expr = &vl_spec["layer"][0]["encoding"]["color"]["legend"]["labelExpr"];
        assert!(
            label_expr.is_string(),
            "legend.labelExpr should be a string"
        );
        let expr = label_expr.as_str().unwrap();
        assert!(
            expr.contains("Category A"),
            "labelExpr should contain renamed label"
        );
        assert!(
            expr.contains("Category B"),
            "labelExpr should contain renamed label"
        );
    }

    #[test]
    fn test_scale_renaming_with_null_suppresses_label() {
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("cat".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("val".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with RENAMING including NULL suppression
        let mut scale = Scale::new("x");
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("visible".to_string(), Some("Shown".to_string()));
        label_mapping.insert("internal".to_string(), None); // NULL -> suppress
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "cat" => &["visible", "internal"],
            "val" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr handles NULL (empty string)
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        let expr = label_expr.as_str().unwrap();
        // NULL should result in empty string
        assert!(
            expr.contains("? ''"),
            "NULL suppression should produce empty string"
        );
    }

    #[test]
    fn test_scale_renaming_temporal_uses_time_format() {
        use crate::plot::scale::{Scale, Transform};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("val".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with date transform and RENAMING
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::date());
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("2024-01-01".to_string(), Some("Q1 Start".to_string()));
        label_mapping.insert("2024-04-01".to_string(), Some("Q2 Start".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "date" => &["2024-01-01", "2024-04-01"],
            "val" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr uses timeFormat for temporal scales
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        assert!(label_expr.is_string(), "axis.labelExpr should be a string");
        let expr = label_expr.as_str().unwrap();

        // Should use timeFormat(datum.value, '%Y-%m-%d') for date scales
        assert!(
            expr.contains("timeFormat(datum.value, '%Y-%m-%d')"),
            "temporal labelExpr should use timeFormat: got {}",
            expr
        );
        // Should contain the ISO date key
        assert!(
            expr.contains("2024-01-01"),
            "labelExpr should contain ISO date key"
        );
        // Should contain the renamed label
        assert!(
            expr.contains("Q1 Start"),
            "labelExpr should contain renamed label"
        );
    }

    #[test]
    fn test_scale_renaming_datetime_uses_time_format() {
        use crate::plot::scale::{Scale, Transform};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("ts".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("val".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with datetime transform and RENAMING
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::datetime());
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert(
            "2024-01-15T10:30:00".to_string(),
            Some("Morning Meeting".to_string()),
        );
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "ts" => &["2024-01-15T10:30:00"],
            "val" => &[100],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr uses timeFormat for datetime scales
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        let expr = label_expr.as_str().unwrap();

        // Should use timeFormat with datetime format
        assert!(
            expr.contains("timeFormat(datum.value, '%Y-%m-%dT%H:%M:%S')"),
            "datetime labelExpr should use timeFormat with ISO datetime format: got {}",
            expr
        );
    }

    #[test]
    fn test_scale_renaming_time_uses_time_format() {
        use crate::plot::scale::{Scale, Transform};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("time".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("val".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with time transform and RENAMING
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::time());
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("09:00:00".to_string(), Some("Market Open".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "time" => &["09:00:00"],
            "val" => &[100],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr uses timeFormat for time scales
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        let expr = label_expr.as_str().unwrap();

        // Should use timeFormat with time format
        assert!(
            expr.contains("timeFormat(datum.value, '%H:%M:%S')"),
            "time labelExpr should use timeFormat with time format: got {}",
            expr
        );
    }

    #[test]
    fn test_scale_renaming_non_temporal_uses_datum_label() {
        use crate::plot::scale::{Scale, Transform};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with non-temporal transform (log) and RENAMING
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::log());
        let mut label_mapping = std::collections::HashMap::new();
        label_mapping.insert("1".to_string(), Some("One".to_string()));
        label_mapping.insert("10".to_string(), Some("Ten".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 10],
            "y" => &[1, 2],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that axis.labelExpr uses datum.label for non-temporal scales
        let label_expr = &vl_spec["layer"][0]["encoding"]["x"]["axis"]["labelExpr"];
        let expr = label_expr.as_str().unwrap();

        // Should use datum.label, NOT timeFormat
        assert!(
            expr.contains("datum.label =="),
            "non-temporal labelExpr should use datum.label: got {}",
            expr
        );
        assert!(
            !expr.contains("timeFormat"),
            "non-temporal labelExpr should NOT use timeFormat: got {}",
            expr
        );
    }

    // ========================================
    // Size and Linewidth Unit Conversion Tests
    // ========================================

    #[test]
    fn test_size_scale_range_conversion() {
        // Test that SCALE size TO [1, 6] converts radius (points) to area (pixels²)
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with output range [1, 6] (radius in points)
        let mut scale = Scale::new("size");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(6.0),
        ]));
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Range should be converted: [1², 6²] × π × (96/72)²
        let range = vl_spec["layer"][0]["encoding"]["size"]["scale"]["range"]
            .as_array()
            .unwrap();
        let expected_min = 1.0 * 1.0 * POINTS_TO_AREA; // ~5.585
        let expected_max = 6.0 * 6.0 * POINTS_TO_AREA; // ~201.1

        assert!(
            (range[0].as_f64().unwrap() - expected_min).abs() < 0.1,
            "Range min: expected ~{:.1}, got {:.1}",
            expected_min,
            range[0].as_f64().unwrap()
        );
        assert!(
            (range[1].as_f64().unwrap() - expected_max).abs() < 0.1,
            "Range max: expected ~{:.1}, got {:.1}",
            expected_max,
            range[1].as_f64().unwrap()
        );
    }

    #[test]
    fn test_linewidth_scale_range_conversion() {
        // Test that SCALE linewidth TO [0.5, 4] converts points to pixels
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "linewidth".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add scale with output range [0.5, 4] (width in points)
        let mut scale = Scale::new("linewidth");
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(0.5),
            ArrayElement::Number(4.0),
        ]));
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Range should be converted: [0.5, 4] × (96/72)
        let range = vl_spec["layer"][0]["encoding"]["strokeWidth"]["scale"]["range"]
            .as_array()
            .unwrap();
        let expected_min = 0.5 * POINTS_TO_PIXELS; // ~0.667
        let expected_max = 4.0 * POINTS_TO_PIXELS; // ~5.333

        assert!(
            (range[0].as_f64().unwrap() - expected_min).abs() < 0.01,
            "Range min: expected ~{:.2}, got {:.2}",
            expected_min,
            range[0].as_f64().unwrap()
        );
        assert!(
            (range[1].as_f64().unwrap() - expected_max).abs() < 0.01,
            "Range max: expected ~{:.2}, got {:.2}",
            expected_max,
            range[1].as_f64().unwrap()
        );
    }

    #[test]
    fn test_size_sqrt_transform_passes_through() {
        // Test that SCALE size VIA sqrt passes through to Vega-Lite as sqrt scale
        use crate::plot::scale::Transform;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add sqrt transform for size
        let mut scale = Scale::new("size");
        scale.transform = Some(Transform::sqrt());
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
            "value" => &[100, 400, 900],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Sqrt transform passes through to Vega-Lite
        let scale_obj = &vl_spec["layer"][0]["encoding"]["size"]["scale"];
        assert_eq!(
            scale_obj["type"], "sqrt",
            "Sqrt transform on size should pass through as sqrt scale"
        );
    }

    #[test]
    fn test_size_identity_transform_uses_linear_scale() {
        // Test that SCALE size VIA identity (linear) also results in linear scale
        use crate::plot::scale::Transform;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add identity transform for size
        let mut scale = Scale::new("size");
        scale.transform = Some(Transform::identity());
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should NOT have scale.type (linear is default)
        let scale_obj = &vl_spec["layer"][0]["encoding"]["size"]["scale"];
        assert!(
            scale_obj.get("type").is_none() || scale_obj["type"].is_null(),
            "Identity transform on size should use linear scale, got: {}",
            scale_obj
        );
    }

    #[test]
    fn test_size_log_transform_passes_through() {
        // Test that SCALE size VIA log passes through to Vega-Lite as log scale
        use crate::plot::scale::Transform;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add log transform for size
        let mut scale = Scale::new("size");
        scale.transform = Some(Transform::log());
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
            "value" => &[10, 100, 1000],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Log transform passes through to Vega-Lite
        let scale_obj = &vl_spec["layer"][0]["encoding"]["size"]["scale"];
        assert_eq!(
            scale_obj["type"], "log",
            "Log transform on size should pass through as log scale"
        );
        assert_eq!(scale_obj["base"], 10, "Log transform should have base 10");
    }

    #[test]
    fn test_non_size_sqrt_transform_unchanged() {
        // Verify that sqrt transform on non-size aesthetics still produces sqrt scale
        use crate::plot::scale::Transform;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        // Add sqrt transform for y axis
        let mut scale = Scale::new("y");
        scale.transform = Some(Transform::sqrt());
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 4, 9],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Y axis should have sqrt scale
        let scale_obj = &vl_spec["layer"][0]["encoding"]["y"]["scale"];
        assert_eq!(
            scale_obj["type"], "sqrt",
            "Sqrt transform on y should produce sqrt scale"
        );
    }

    #[test]
    fn test_other_aesthetics_pass_through_unchanged() {
        // Test that color, opacity, shape literals are not converted
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "opacity".to_string(),
                AestheticValue::Literal(ParameterValue::Number(0.75)),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Opacity should pass through unchanged
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["opacity"]["value"], 0.75,
            "Opacity literal should pass through unchanged"
        );
    }

    // ========================================
    // Unified Dataset Tests
    // ========================================

    #[test]
    fn test_unified_data_structure() {
        // Test that the writer produces a unified dataset with source column
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have a single unified dataset at GLOBAL_DATA_KEY
        assert!(
            vl_spec["datasets"][naming::GLOBAL_DATA_KEY].is_array(),
            "Should have unified dataset at global key"
        );

        // Unified data should have __ggsql_source__ column
        let unified_data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert!(!unified_data.is_empty(), "Unified data should not be empty");
        assert!(
            unified_data[0].get(naming::SOURCE_COLUMN).is_some(),
            "Each row should have source column"
        );

        // Top-level data should reference the unified dataset
        assert_eq!(
            vl_spec["data"]["name"],
            naming::GLOBAL_DATA_KEY,
            "Top-level data should reference unified dataset"
        );
    }

    #[test]
    fn test_layer_has_filter_transform() {
        // Test that each layer has a filter transform for source selection
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Layer should have transform array with filter
        let layer_spec = &vl_spec["layer"][0];
        let transforms = layer_spec["transform"].as_array();
        assert!(transforms.is_some(), "Layer should have transforms");

        let transforms = transforms.unwrap();
        assert!(!transforms.is_empty(), "Transforms should not be empty");

        // First transform should be a filter on __ggsql_source__
        let filter_transform = &transforms[0];
        assert!(
            filter_transform.get("filter").is_some(),
            "First transform should be a filter"
        );
        assert_eq!(
            filter_transform["filter"]["field"],
            naming::SOURCE_COLUMN,
            "Filter should be on source column"
        );
    }

    #[test]
    fn test_multi_layer_unified_data() {
        // Test that multiple layers are unified into a single dataset
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();

        // Layer 1: point geom
        let layer1 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer1);

        // Layer 2: line geom
        let layer2 = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer2);

        // Create data with two layer entries
        let mut data_map = HashMap::new();
        let df1 = df! {
            "x" => &[1, 2],
            "y" => &[10, 20],
        }
        .unwrap();
        let df2 = df! {
            "x" => &[3, 4],
            "y" => &[30, 40],
        }
        .unwrap();
        data_map.insert(naming::layer_key(0), df1);
        data_map.insert(naming::layer_key(1), df2);

        let json_str = writer.write(&spec, &data_map).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Unified data should have all 4 rows (2 from each layer)
        let unified_data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(unified_data.len(), 4, "Unified data should have 4 rows");

        // Each layer should have distinct filter value
        let layer0_filter = &vl_spec["layer"][0]["transform"][0]["filter"]["equal"];
        let layer1_filter = &vl_spec["layer"][1]["transform"][0]["filter"]["equal"];

        assert_eq!(
            layer0_filter,
            &naming::layer_key(0),
            "Layer 0 filter should use layer_key(0)"
        );
        assert_eq!(
            layer1_filter,
            &naming::layer_key(1),
            "Layer 1 filter should use layer_key(1)"
        );
    }

    #[test]
    fn test_unified_data_preserves_layer_separation() {
        // Test that filter transforms correctly isolate layer data
        // when multiple layers have different data sources
        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        // Layer 0: points
        let layer0 = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        // Layer 1: lines (different geom to show they're separate layers)
        let layer1 = Layer::new(Geom::line())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            );
        spec.layers.push(layer0);
        spec.layers.push(layer1);

        // Create two layer datasets with different data
        let mut data_map = HashMap::new();
        let df1 = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
        }
        .unwrap();
        let df2 = df! {
            "x" => &[100, 200],
            "y" => &[1000, 2000],
        }
        .unwrap();
        data_map.insert(naming::layer_key(0), df1);
        data_map.insert(naming::layer_key(1), df2);

        let json_str = writer.write(&spec, &data_map).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Unified data should have all 5 rows
        let unified_data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        assert_eq!(
            unified_data.len(),
            5,
            "Unified data should have 5 rows total"
        );

        // Count rows by source
        let layer0_count = unified_data
            .iter()
            .filter(|r| r[naming::SOURCE_COLUMN] == naming::layer_key(0))
            .count();
        let layer1_count = unified_data
            .iter()
            .filter(|r| r[naming::SOURCE_COLUMN] == naming::layer_key(1))
            .count();

        assert_eq!(layer0_count, 3, "Layer 0 should have 3 rows");
        assert_eq!(layer1_count, 2, "Layer 1 should have 2 rows");
    }

    // ========================================
    // render_ribbon Tests
    // ========================================

    #[test]
    fn test_render_ribbon_translates_ymin_ymax() {
        let mut encoding = Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "x", "type": "quantitative"}),
        );
        encoding.insert(
            "ymin".to_string(),
            json!({"field": "lower", "type": "quantitative"}),
        );
        encoding.insert(
            "ymax".to_string(),
            json!({"field": "upper", "type": "quantitative"}),
        );

        render_ribbon(&mut encoding);

        // ymax should become y
        assert_eq!(encoding.get("y").unwrap()["field"], "upper");

        // ymin should become y2
        assert_eq!(encoding.get("y2").unwrap()["field"], "lower");

        // Original ymin and ymax should be removed
        assert!(!encoding.contains_key("ymin"));
        assert!(!encoding.contains_key("ymax"));
    }

    // ========================================
    // render_area Tests
    // ========================================

    #[test]
    fn test_render_area_stacking_values() {
        let test_cases = vec![
            (Some("on"), json!("zero")),
            (Some("off"), Value::Null),
            (Some("fill"), json!("normalize")),
            (None, Value::Null),
        ];

        for (stacking_param, expected_stack) in test_cases {
            let mut encoding = Map::new();
            encoding.insert(
                "y".to_string(),
                json!({"field": "value", "type": "quantitative"}),
            );

            let mut layer = Layer::new(Geom::area());
            if let Some(value) = stacking_param {
                layer = layer.with_parameter(
                    "stacking".to_string(),
                    ParameterValue::String(value.to_string()),
                );
            }

            render_area(&mut encoding, &layer).unwrap();

            assert_eq!(
                encoding.get("y").unwrap()["stack"],
                expected_stack,
                "stacking={:?} should produce stack={:?}",
                stacking_param,
                expected_stack
            );
        }
    }

    #[test]
    fn test_render_area_stacking_invalid() {
        let mut encoding = Map::new();
        encoding.insert(
            "y".to_string(),
            json!({"field": "value", "type": "quantitative"}),
        );

        let layer = Layer::new(Geom::area()).with_parameter(
            "stacking".to_string(),
            ParameterValue::String("invalid".to_string()),
        );

        let result = render_area(&mut encoding, &layer);

        assert!(result.is_err());
        match result {
            Err(GgsqlError::ValidationError(msg)) => {
                assert!(msg.contains("stacking"));
                assert!(msg.contains("invalid"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[test]
    fn test_boxplot_vertical_with_outliers() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let y_col = naming::aesthetic_column("y");
        let y2_col = naming::aesthetic_column("y2");
        let type_col = naming::aesthetic_column("type");

        // Create boxplot data in visual-element format (as produced by stat_boxplot after remapping)
        // Each row represents a visual element with y (primary value) and y2 (secondary value)
        // Types: lower_whisker (y=q1, y2=lower), upper_whisker (y=q3, y2=upper),
        //        box (y=q1, y2=q3), median (y=median), outlier (y=value)
        let df = df! {
            "category" => &[
                // Category A visual elements
                "A", "A", "A", "A",
                // Category A outliers
                "A", "A", "A",
                // Category B visual elements
                "B", "B", "B", "B",
                // Category B outliers
                "B", "B"
            ],
            type_col.as_str() => &[
                // Category A
                "lower_whisker", "upper_whisker", "box", "median",
                "outlier", "outlier", "outlier",
                // Category B
                "lower_whisker", "upper_whisker", "box", "median",
                "outlier", "outlier"
            ],
            y_col.as_str() => &[
                // Category A: q1=15, q3=25, median=20, outliers: 5, 35, 40
                15.0, 25.0, 15.0, 20.0, 5.0, 35.0, 40.0,
                // Category B: q1=25, q3=35, median=30, outliers: 15, 50
                25.0, 35.0, 25.0, 30.0, 15.0, 50.0
            ],
            y2_col.as_str() => &[
                // Category A: lower=10, upper=30
                Some(10.0), Some(30.0), Some(25.0), None, None, None, None,
                // Category B: lower=20, upper=40
                Some(20.0), Some(40.0), Some(35.0), None, None, None
            ],
        }
        .unwrap();

        // Create a boxplot layer
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::boxplot())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column(y_col.clone()),
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // INVARIANT: Only one unified dataset should exist
        let datasets = vl_spec["datasets"]
            .as_object()
            .expect("datasets should be an object");
        assert_eq!(
            datasets.len(),
            1,
            "Expected exactly 1 dataset (unified), found {}. Keys: {:?}",
            datasets.len(),
            datasets.keys().collect::<Vec<_>>()
        );
        assert!(
            datasets.contains_key(naming::GLOBAL_DATA_KEY),
            "Should have unified global dataset"
        );

        // Verify that boxplot produces multiple layers (outliers + 4 boxplot components)
        assert!(vl_spec["layer"].is_array());
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 5, "Boxplot with outliers should produce 5 layers: outliers, lower whisker, upper whisker, box, median");

        // Verify first layer is outliers (point marks)
        assert_eq!(
            layers[0]["mark"]["type"], "point",
            "First layer should be outlier points"
        );

        // Verify all boxplot layers use filter transforms on __ggsql_source__
        for (i, layer) in layers.iter().enumerate() {
            let transforms = layer["transform"]
                .as_array()
                .unwrap_or_else(|| panic!("Layer {} should have transforms", i));
            assert!(
                !transforms.is_empty(),
                "Layer {} should have at least one transform",
                i
            );
            let filter = &transforms[0]["filter"];
            assert_eq!(
                filter["field"],
                naming::SOURCE_COLUMN,
                "Layer {} should filter on __ggsql_source__",
                i
            );
        }

        // Verify whiskers (rule marks)
        assert_eq!(
            layers[1]["mark"]["type"], "rule",
            "Second layer should be lower whisker"
        );
        assert_eq!(
            layers[2]["mark"]["type"], "rule",
            "Third layer should be upper whisker"
        );

        // Verify box (bar mark)
        assert_eq!(
            layers[3]["mark"]["type"], "bar",
            "Fourth layer should be box"
        );

        // Verify median (tick mark)
        assert_eq!(
            layers[4]["mark"]["type"], "tick",
            "Fifth layer should be median line"
        );

        // Verify source keys use type-specific suffixes
        let outlier_source = layers[0]["transform"][0]["filter"]["equal"]
            .as_str()
            .unwrap();
        let lower_whisker_source = layers[1]["transform"][0]["filter"]["equal"]
            .as_str()
            .unwrap();
        assert!(
            outlier_source.ends_with("outlier"),
            "Outlier source should end with 'outlier', got: {}",
            outlier_source
        );
        assert!(
            lower_whisker_source.ends_with("lower_whisker"),
            "Lower whisker source should end with 'lower_whisker', got: {}",
            lower_whisker_source
        );

        // Verify unified dataset contains data with type-specific source tags
        let unified_data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        let outlier_rows: Vec<_> = unified_data
            .iter()
            .filter(|row| row[naming::SOURCE_COLUMN].as_str() == Some(outlier_source))
            .collect();
        let lower_whisker_rows: Vec<_> = unified_data
            .iter()
            .filter(|row| row[naming::SOURCE_COLUMN].as_str() == Some(lower_whisker_source))
            .collect();
        assert_eq!(
            outlier_rows.len(),
            5,
            "Should have 5 outlier rows (3 for A, 2 for B)"
        );
        assert_eq!(
            lower_whisker_rows.len(),
            2,
            "Should have 2 lower whisker rows (one per category)"
        );

        // Verify rows have y and y2 columns (not separate stat columns)
        let first_lower_whisker = &lower_whisker_rows[0];
        assert!(
            first_lower_whisker[&y_col].is_number(),
            "Should have y column"
        );
        assert!(
            first_lower_whisker[&y2_col].is_number(),
            "Should have y2 column"
        );
        assert!(first_lower_whisker["category"].is_string());

        // Verify encodings use __ggsql_aes_y__ and __ggsql_aes_y2__
        assert!(layers[1]["encoding"]["y"].is_object());
        assert!(layers[1]["encoding"]["y2"].is_object());
        assert_eq!(layers[1]["encoding"]["y"]["field"], y_col);
        assert_eq!(layers[1]["encoding"]["y2"]["field"], y2_col);
    }

    #[test]
    fn test_boxplot_y_axis_title_uses_original_column() {
        // Verify that the y-axis title shows the original column name (e.g., "Temp")
        // not the internal column names (__ggsql_aes_y__, __ggsql_aes_y2__)
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let y_col = naming::aesthetic_column("y");
        let y2_col = naming::aesthetic_column("y2");
        let type_col = naming::aesthetic_column("type");

        // Create minimal boxplot data
        let df = df! {
            "category" => &["A", "A", "A", "A"],
            type_col.as_str() => &["lower_whisker", "upper_whisker", "box", "median"],
            y_col.as_str() => &[15.0, 25.0, 15.0, 20.0],
            y2_col.as_str() => &[Some(10.0), Some(30.0), Some(25.0), None],
        }
        .unwrap();

        // Create layer with original_name set (simulating what happens after stat remapping)
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::boxplot())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::Column {
                    name: y_col.clone(),
                    original_name: Some("Temp".to_string()), // Original column before remapping
                    is_dummy: false,
                },
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let layers = vl_spec["layer"].as_array().unwrap();

        // y encoding should have title "Temp" (original name)
        let y_encoding = &layers[1]["encoding"]["y"];
        assert_eq!(
            y_encoding["title"], "Temp",
            "y-axis title should be the original column name 'Temp', got {:?}",
            y_encoding["title"]
        );

        // y2 encoding should have title: null (suppressed)
        let y2_encoding = &layers[1]["encoding"]["y2"];
        assert!(
            y2_encoding["title"].is_null(),
            "y2 title should be null to prevent duplicate axis labels, got {:?}",
            y2_encoding["title"]
        );
    }

    #[test]
    fn test_bar_stat_y_title_not_overridden_by_y2() {
        // Verify that when bar stat creates "y" (count) and "y2" (baseline 0),
        // the y encoding gets the title "count" and y2 doesn't steal it
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let y_col = naming::aesthetic_column("y");
        let y2_col = naming::aesthetic_column("y2");

        // Create bar chart data with stat-generated y and y2
        let df = df! {
            "category" => &["A", "B", "C"],
            y_col.as_str() => &[10.0, 20.0, 30.0],
            y2_col.as_str() => &[0.0, 0.0, 0.0],
        }
        .unwrap();

        // Create layer with y from stat (original_name = "count") and y2
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::Column {
                    name: y_col.clone(),
                    original_name: Some("count".to_string()), // From bar stat
                    is_dummy: false,
                },
            )
            .with_aesthetic(
                "y2".to_string(),
                AestheticValue::Column {
                    name: y2_col.clone(),
                    original_name: None, // Default baseline, no meaningful name
                    is_dummy: false,
                },
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let layer_spec = &vl_spec["layer"][0];

        // y encoding should have title "count" (from original_name)
        let y_encoding = &layer_spec["encoding"]["y"];
        assert_eq!(
            y_encoding["title"], "count",
            "y-axis title should be 'count' (from stat), got {:?}",
            y_encoding["title"]
        );

        // y2 encoding should have title: null (suppressed because y exists)
        let y2_encoding = &layer_spec["encoding"]["y2"];
        assert!(
            y2_encoding["title"].is_null(),
            "y2 title should be null when y exists, got {:?}",
            y2_encoding["title"]
        );
    }

    #[test]
    fn test_boxplot_horizontal_with_grouping() {
        // NOTE: This test verifies the render_boxplot() logic for horizontal orientation
        // and grouping, but actual orientation detection is not yet implemented upstream
        // in the stat pipeline. This test uses manually constructed data
        // to verify the rendering logic works correctly when given horizontal data.
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let y_col = naming::aesthetic_column("y");
        let y2_col = naming::aesthetic_column("y2");
        let type_col = naming::aesthetic_column("type");

        // Create horizontal boxplot data with grouping in visual-element format
        // Horizontal means x has the values, y has the categories
        let df = df! {
            "category" => &[
                // North region visual elements
                "A", "A", "A", "A",
                // South region visual elements
                "A", "A", "A", "A"
            ],
            "region" => &[
                "North", "North", "North", "North",
                "South", "South", "South", "South"
            ],
            type_col.as_str() => &[
                // North
                "lower_whisker", "upper_whisker", "box", "median",
                // South
                "lower_whisker", "upper_whisker", "box", "median"
            ],
            y_col.as_str() => &[
                // North: q1=15, q3=25, median=20
                15.0, 25.0, 15.0, 20.0,
                // South: q1=25, q3=35, median=30
                25.0, 35.0, 25.0, 30.0
            ],
            y2_col.as_str() => &[
                // North: lower=10, upper=30
                Some(10.0), Some(30.0), Some(25.0), None,
                // South: lower=20, upper=40
                Some(20.0), Some(40.0), Some(35.0), None
            ],
        }
        .unwrap();

        // Create a horizontal boxplot layer (x = value, y = category)
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::boxplot())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column(y_col.clone()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("category".to_string()),
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // INVARIANT: Only one unified dataset should exist
        let datasets = vl_spec["datasets"]
            .as_object()
            .expect("datasets should be an object");
        assert_eq!(
            datasets.len(),
            1,
            "Expected exactly 1 dataset (unified), found {}. Keys: {:?}",
            datasets.len(),
            datasets.keys().collect::<Vec<_>>()
        );

        // Verify multiple layers (no outliers in this data)
        assert!(vl_spec["layer"].is_array());
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 4, "Boxplot should produce 4 layers");

        // Verify all layers use filter transforms
        for (i, layer) in layers.iter().enumerate() {
            let transforms = layer["transform"]
                .as_array()
                .unwrap_or_else(|| panic!("Layer {} should have transforms", i));
            assert!(
                !transforms.is_empty(),
                "Layer {} should have at least one transform",
                i
            );
            assert_eq!(
                transforms[0]["filter"]["field"],
                naming::SOURCE_COLUMN,
                "Layer {} should filter on __ggsql_source__",
                i
            );
        }

        // Verify encodings use x for values (horizontal orientation)
        // First layer is lower_whisker (rule from q1 to lower)
        assert!(layers[0]["encoding"]["x"].is_object());
        assert!(layers[0]["encoding"]["x2"].is_object());
        assert_eq!(layers[0]["encoding"]["x"]["field"], y_col);
        assert_eq!(layers[0]["encoding"]["x2"]["field"], y2_col);

        // Verify yOffset is used for dodging (since we have region grouping)
        assert!(
            layers[0]["encoding"]["yOffset"].is_object(),
            "Should have yOffset for dodging"
        );
        assert_eq!(layers[0]["encoding"]["yOffset"]["field"], "region");

        // Verify unified dataset contains data for lower_whisker type
        let lower_whisker_source = layers[0]["transform"][0]["filter"]["equal"]
            .as_str()
            .unwrap();
        let unified_data = vl_spec["datasets"][naming::GLOBAL_DATA_KEY]
            .as_array()
            .unwrap();
        let lower_whisker_rows: Vec<_> = unified_data
            .iter()
            .filter(|row| row[naming::SOURCE_COLUMN].as_str() == Some(lower_whisker_source))
            .collect();
        assert_eq!(
            lower_whisker_rows.len(),
            2,
            "Should have 2 lower whisker rows (one per region)"
        );

        let first_row = &lower_whisker_rows[0];
        assert!(first_row["category"].is_string());
        assert!(first_row["region"].is_string());
    }

    /// Test that all geom types produce only a single unified dataset
    /// This guards against regressions that might add extra datasets
    #[test]
    fn test_writer_always_produces_single_dataset() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        // Test cases: (name, geom, data, aesthetics)
        // Each should produce exactly one dataset

        // Point
        {
            let df = df! { "x" => &[1.0, 2.0], "y" => &[3.0, 4.0] }.unwrap();
            let mut spec = Plot::new();
            spec.layers.push(
                Layer::new(Geom::point())
                    .with_aesthetic(
                        "x".to_string(),
                        AestheticValue::standard_column("x".to_string()),
                    )
                    .with_aesthetic(
                        "y".to_string(),
                        AestheticValue::standard_column("y".to_string()),
                    ),
            );
            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();
            let datasets = vl_spec["datasets"]
                .as_object()
                .expect("point: datasets should be object");
            assert_eq!(
                datasets.len(),
                1,
                "point: Expected 1 dataset, found {}",
                datasets.len()
            );
        }

        // Line
        {
            let df = df! { "x" => &[1.0, 2.0], "y" => &[3.0, 4.0] }.unwrap();
            let mut spec = Plot::new();
            spec.layers.push(
                Layer::new(Geom::line())
                    .with_aesthetic(
                        "x".to_string(),
                        AestheticValue::standard_column("x".to_string()),
                    )
                    .with_aesthetic(
                        "y".to_string(),
                        AestheticValue::standard_column("y".to_string()),
                    ),
            );
            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();
            let datasets = vl_spec["datasets"]
                .as_object()
                .expect("line: datasets should be object");
            assert_eq!(
                datasets.len(),
                1,
                "line: Expected 1 dataset, found {}",
                datasets.len()
            );
        }

        // Bar
        {
            let df = df! { "x" => &["A", "B"], "y" => &[10.0, 20.0] }.unwrap();
            let mut spec = Plot::new();
            spec.layers.push(
                Layer::new(Geom::bar())
                    .with_aesthetic(
                        "x".to_string(),
                        AestheticValue::standard_column("x".to_string()),
                    )
                    .with_aesthetic(
                        "y".to_string(),
                        AestheticValue::standard_column("y".to_string()),
                    ),
            );
            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();
            let datasets = vl_spec["datasets"]
                .as_object()
                .expect("bar: datasets should be object");
            assert_eq!(
                datasets.len(),
                1,
                "bar: Expected 1 dataset, found {}",
                datasets.len()
            );
        }

        // Boxplot - this was the problematic case that motivated this fix
        // Uses aesthetic column names since remappings have been applied
        {
            let df = df! {
                "category" => &["A", "A", "A", "A", "A", "A", "A"],
                naming::aesthetic_column("type").as_str() => &["lower", "q1", "median", "q3", "upper", "min", "max"],
                naming::aesthetic_column("y").as_str() => &[10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 30.0],
            }.unwrap();
            let mut spec = Plot::new();
            spec.layers.push(
                Layer::new(Geom::boxplot())
                    .with_aesthetic(
                        "x".to_string(),
                        AestheticValue::standard_column("category".to_string()),
                    )
                    .with_aesthetic(
                        "y".to_string(),
                        AestheticValue::standard_column(naming::aesthetic_column("y")),
                    ),
            );
            let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();
            let datasets = vl_spec["datasets"]
                .as_object()
                .expect("boxplot: datasets should be object");
            assert_eq!(
                datasets.len(),
                1,
                "boxplot: Expected 1 dataset (single-dataset invariant), found {}. Keys: {:?}",
                datasets.len(),
                datasets.keys().collect::<Vec<_>>()
            );
        }
    }

    /// Test that boxplot layers all use filter transforms
    #[test]
    fn test_boxplot_uses_filter_transforms() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        // Create boxplot data with outliers (after remapping)
        // Uses aesthetic column names since remappings have been applied
        let df = df! {
            "category" => &["A", "A", "A", "A", "A", "A", "A", "A", "A"],
            naming::aesthetic_column("type").as_str() => &["lower", "q1", "median", "q3", "upper", "min", "max", "outlier", "outlier"],
            naming::aesthetic_column("y").as_str() => &[10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 30.0, 5.0, 35.0],
        }
        .unwrap();

        let mut spec = Plot::new();
        spec.layers.push(
            Layer::new(Geom::boxplot())
                .with_aesthetic(
                    "x".to_string(),
                    AestheticValue::standard_column("category".to_string()),
                )
                .with_aesthetic(
                    "y".to_string(),
                    AestheticValue::standard_column(naming::aesthetic_column("y")),
                ),
        );

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // All boxplot layers should use filter transforms on __ggsql_source__
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 5); // outliers + 4 boxplot parts

        for (i, layer) in layers.iter().enumerate() {
            let transforms = layer["transform"]
                .as_array()
                .unwrap_or_else(|| panic!("Layer {} should have transforms", i));
            assert!(
                !transforms.is_empty(),
                "Layer {} should have at least one transform",
                i
            );
            assert_eq!(
                transforms[0]["filter"]["field"],
                naming::SOURCE_COLUMN,
                "Boxplot layer {} should filter on __ggsql_source__",
                i
            );
        }
    }

    #[test]
    fn test_binned_fill_scale_uses_threshold_type() {
        // Test that binned non-positional aesthetics use threshold scale type
        // for proper discrete color legend rendering
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for fill with breaks
        let mut scale = Scale::new("fill");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The fill encoding should have scale.type = "threshold"
        let scale_type = &vl_spec["layer"][0]["encoding"]["fill"]["scale"]["type"];
        assert_eq!(
            scale_type, "threshold",
            "Binned fill scale should use threshold type"
        );
    }

    #[test]
    fn test_binned_fill_scale_threshold_domain() {
        // Test that threshold domain is internal breaks (excluding terminals)
        // breaks = [0, 25, 50, 75, 100] → domain = [25, 50, 75]
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for fill with breaks
        let mut scale = Scale::new("fill");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The fill encoding should have domain with internal breaks only
        let domain = &vl_spec["layer"][0]["encoding"]["fill"]["scale"]["domain"];
        assert!(
            domain.is_array(),
            "Threshold scale should have domain array"
        );
        let domain_arr = domain.as_array().unwrap();
        assert_eq!(
            domain_arr.len(),
            3,
            "Domain should have 3 internal breaks (excluding 0 and 100)"
        );
        assert_eq!(domain_arr[0], 25.0);
        assert_eq!(domain_arr[1], 50.0);
        assert_eq!(domain_arr[2], 75.0);
    }

    #[test]
    fn test_binned_fill_scale_legend_values() {
        // Test that legend.values contains all breaks including terminals
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for fill with breaks
        let mut scale = Scale::new("fill");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The fill encoding should have legend.values with all breaks
        let legend_values = &vl_spec["layer"][0]["encoding"]["fill"]["legend"]["values"];
        assert!(
            legend_values.is_array(),
            "Legend should have values array for all breaks"
        );
        let values_arr = legend_values.as_array().unwrap();
        assert_eq!(
            values_arr.len(),
            5,
            "Legend.values should have all 5 breaks including terminals"
        );
        assert_eq!(values_arr[0], 0.0);
        assert_eq!(values_arr[4], 100.0);
    }

    #[test]
    fn test_binned_fill_scale_label_expr_null_key() {
        // Test that labelExpr uses datum.label == null for first terminal
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};
        use std::collections::HashMap;

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for fill with breaks and label_mapping
        let mut scale = Scale::new("fill");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // Add label_mapping to trigger labelExpr generation
        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), Some("Low".to_string()));
        label_mapping.insert("25".to_string(), Some("Medium-Low".to_string()));
        label_mapping.insert("50".to_string(), Some("Medium".to_string()));
        label_mapping.insert("75".to_string(), Some("Medium-High".to_string()));
        label_mapping.insert("100".to_string(), Some("High".to_string()));
        scale.label_mapping = Some(label_mapping);
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The fill encoding should have legend.labelExpr with null check for first terminal
        let label_expr = &vl_spec["layer"][0]["encoding"]["fill"]["legend"]["labelExpr"];
        assert!(label_expr.is_string(), "labelExpr should be a string");
        let expr_str = label_expr.as_str().unwrap();

        // First terminal (0) should use datum.label == null
        assert!(
            expr_str.contains("datum.label == null"),
            "labelExpr should use null check for first terminal: {}",
            expr_str
        );

        // Other breaks should use string comparison
        assert!(
            expr_str.contains("datum.label == '25'"),
            "labelExpr should use string comparison for non-first breaks: {}",
            expr_str
        );
    }

    #[test]
    fn test_binned_positional_scale_not_threshold() {
        // Test that binned positional aesthetics (x, y) don't use threshold scale
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for y (positional)
        let mut scale = Scale::new("y");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The y encoding should NOT have scale.type = "threshold"
        let scale_type = &vl_spec["layer"][0]["encoding"]["y"]["scale"]["type"];
        assert!(
            scale_type.is_null() || scale_type != "threshold",
            "Binned positional scale should not use threshold type"
        );

        // But axis.values should still be set
        let axis_values = &vl_spec["layer"][0]["encoding"]["y"]["axis"]["values"];
        assert!(
            axis_values.is_array(),
            "Binned positional scale should have axis.values"
        );
    }

    #[test]
    fn test_build_label_expr_with_null_key() {
        // Test the build_label_expr function with null_key parameter
        use std::collections::HashMap;

        // Test with null_key for first terminal
        let mut mappings = HashMap::new();
        mappings.insert("0".to_string(), Some("Low".to_string()));
        mappings.insert("25".to_string(), Some("Medium".to_string()));

        let expr = build_label_expr(&mappings, None, Some("0"));

        // First terminal should use null comparison
        assert!(
            expr.contains("datum.label == null ? 'Low'"),
            "First terminal should use null comparison: {}",
            expr
        );

        // Second entry should use string comparison
        assert!(
            expr.contains("datum.label == '25' ? 'Medium'"),
            "Non-first entries should use string comparison: {}",
            expr
        );
    }

    #[test]
    fn test_build_label_expr_without_null_key() {
        // Test the build_label_expr function without null_key (normal case)
        use std::collections::HashMap;

        let mut mappings = HashMap::new();
        mappings.insert("0".to_string(), Some("Low".to_string()));
        mappings.insert("25".to_string(), Some("Medium".to_string()));

        let expr = build_label_expr(&mappings, None, None);

        // Both entries should use string comparison
        assert!(
            expr.contains("datum.label == '0' ? 'Low'"),
            "Without null_key, should use string comparison: {}",
            expr
        );
        assert!(
            expr.contains("datum.label == '25' ? 'Medium'"),
            "Without null_key, should use string comparison: {}",
            expr
        );
        assert!(
            !expr.contains("datum.label == null"),
            "Without null_key, should not use null comparison: {}",
            expr
        );
    }

    #[test]
    fn test_binned_color_scale_symbol_legend_removes_last_terminal() {
        // Test that binned color scale (not fill/stroke) uses symbol legend
        // which removes the last terminal from legend.values
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add a binned scale for color (not fill/stroke, so symbol legend)
        let mut scale = Scale::new("color");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // The color encoding should have legend.values with last terminal removed
        let legend_values = &vl_spec["layer"][0]["encoding"]["color"]["legend"]["values"];
        assert!(legend_values.is_array(), "Legend should have values array");
        let values_arr = legend_values.as_array().unwrap();
        assert_eq!(
            values_arr.len(),
            4,
            "Symbol legend should have 4 values (last terminal 100 removed)"
        );
        assert_eq!(values_arr[0], 0.0);
        assert_eq!(values_arr[3], 75.0);
        // 100 should NOT be in the values
        assert!(
            !values_arr.iter().any(|v| v == &json!(100.0)),
            "Last terminal (100) should be removed from symbol legend"
        );
    }

    #[test]
    fn test_multiple_binned_scales_both_use_symbol_legend() {
        // Test that when there are multiple binned non-positional scales,
        // both use symbol legend (last terminal removed)
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value1".to_string()),
            )
            .with_aesthetic(
                "size".to_string(),
                AestheticValue::standard_column("value2".to_string()),
            );
        spec.layers.push(layer);

        // Add binned scale for fill
        let mut fill_scale = Scale::new("fill");
        fill_scale.scale_type = Some(ScaleType::binned());
        fill_scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(fill_scale);

        // Add binned scale for size
        let mut size_scale = Scale::new("size");
        size_scale.scale_type = Some(ScaleType::binned());
        size_scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
            ]),
        );
        spec.scales.push(size_scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value1" => &[10.0, 60.0, 90.0],
            "value2" => &[5.0, 30.0, 45.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Fill scale (normally gradient) should use symbol legend when multiple scales exist
        let fill_legend_values = &vl_spec["layer"][0]["encoding"]["fill"]["legend"]["values"];
        assert!(fill_legend_values.is_array());
        let fill_values = fill_legend_values.as_array().unwrap();
        assert_eq!(
            fill_values.len(),
            2,
            "Fill with multiple binned scales should have 2 values (100 removed)"
        );
        assert!(!fill_values.iter().any(|v| v == &json!(100.0)));

        // Size scale should also use symbol legend
        let size_legend_values = &vl_spec["layer"][0]["encoding"]["size"]["legend"]["values"];
        assert!(size_legend_values.is_array());
        let size_values = size_legend_values.as_array().unwrap();
        assert_eq!(
            size_values.len(),
            2,
            "Size should have 2 values (50 removed)"
        );
        assert!(!size_values.iter().any(|v| v == &json!(50.0)));
    }

    #[test]
    fn test_single_binned_fill_keeps_all_terminals() {
        // Test that a single binned fill scale (gradient legend) keeps all terminals
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "fill".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add only one binned fill scale (gradient legend case)
        let mut scale = Scale::new("fill");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Single fill scale should keep all values (gradient legend)
        let legend_values = &vl_spec["layer"][0]["encoding"]["fill"]["legend"]["values"];
        assert!(legend_values.is_array());
        let values_arr = legend_values.as_array().unwrap();
        assert_eq!(
            values_arr.len(),
            5,
            "Single fill (gradient legend) should keep all 5 breaks"
        );
        assert_eq!(values_arr[0], 0.0);
        assert_eq!(values_arr[4], 100.0);
    }

    #[test]
    fn test_build_symbol_legend_label_mapping_basic() {
        // Test the build_symbol_legend_label_mapping function directly
        use super::build_symbol_legend_label_mapping;

        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
            ArrayElement::Number(100.0),
        ];

        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), Some("Low".to_string()));
        label_mapping.insert("25".to_string(), Some("Medium".to_string()));
        label_mapping.insert("50".to_string(), Some("High".to_string()));
        label_mapping.insert("75".to_string(), Some("Very High".to_string()));
        label_mapping.insert("100".to_string(), Some("Max".to_string())); // Will be excluded

        let result = build_symbol_legend_label_mapping(&breaks, &label_mapping, "left");

        // VL generates: "0 – 25", "25 – 50", "50 – 75", "≥ 75"
        // We map to range format using custom labels: "lower_label – upper_label"
        assert_eq!(
            result.get("0 – 25"),
            Some(&Some("Low – Medium".to_string()))
        );
        assert_eq!(
            result.get("25 – 50"),
            Some(&Some("Medium – High".to_string()))
        );
        assert_eq!(
            result.get("50 – 75"),
            Some(&Some("High – Very High".to_string()))
        );
        assert_eq!(
            result.get("≥ 75"),
            Some(&Some("Very High – Max".to_string()))
        );

        // Should not include a mapping for the last terminal value directly
        assert!(result.get("100").is_none());
    }

    #[test]
    fn test_symbol_legend_label_expr_uses_range_format() {
        // Test that symbol legend labelExpr maps VL's range labels to our labels
        use crate::plot::scale::Scale;
        use crate::plot::{ArrayElement, ParameterValue, ScaleType};

        let writer = VegaLiteWriter::new();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("x".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("y".to_string()),
            )
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Add binned color scale (symbol legend case)
        let mut scale = Scale::new("color");
        scale.scale_type = Some(ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );
        // Add label renaming
        let mut labels = HashMap::new();
        labels.insert("0".to_string(), Some("Low".to_string()));
        labels.insert("25".to_string(), Some("Medium".to_string()));
        labels.insert("50".to_string(), Some("High".to_string()));
        labels.insert("75".to_string(), Some("Very High".to_string()));
        scale.label_mapping = Some(labels);
        spec.scales.push(scale);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 45, 80],
            "value" => &[10.0, 45.0, 80.0],
        }
        .unwrap();

        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that labelExpr contains VL's range-style format
        let label_expr = &vl_spec["layer"][0]["encoding"]["color"]["legend"]["labelExpr"];
        assert!(label_expr.is_string());
        let expr = label_expr.as_str().unwrap();

        // Should contain mappings for VL's range format labels to our range format
        assert!(
            expr.contains("0 – 25"),
            "labelExpr should contain VL's range format '0 – 25', got: {}",
            expr
        );
        assert!(
            expr.contains("'Low – Medium'"),
            "labelExpr should map to 'Low – Medium', got: {}",
            expr
        );
        assert!(
            expr.contains("≥ 75"),
            "labelExpr should contain VL's last bin format '≥ 75', got: {}",
            expr
        );
        // Note: last bin maps "≥ 75" to "Very High – 100" (no custom label for 100 in this test)
        assert!(
            expr.contains("'Very High"),
            "labelExpr should contain 'Very High', got: {}",
            expr
        );
    }

    #[test]
    fn test_symbol_legend_open_format_with_oob_squish() {
        // Test that oob='squish' produces open format labels for symbol legends
        use super::build_symbol_legend_label_mapping;

        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
            ArrayElement::Number(100.0),
        ];

        // Suppress first and last terminals (oob='squish' behavior)
        let mut label_mapping = HashMap::new();
        label_mapping.insert("0".to_string(), None); // Suppressed
        label_mapping.insert("25".to_string(), Some("Medium".to_string()));
        label_mapping.insert("50".to_string(), Some("High".to_string()));
        label_mapping.insert("75".to_string(), Some("Very High".to_string()));
        label_mapping.insert("100".to_string(), None); // Suppressed

        // Test with closed='left' (default)
        let result_left = build_symbol_legend_label_mapping(&breaks, &label_mapping, "left");

        // First bin: suppressed lower terminal → "< 25" (open format)
        assert_eq!(
            result_left.get("0 – 25"),
            Some(&Some("< Medium".to_string())),
            "First bin with suppressed lower should use '< upper' format"
        );
        // Last bin: suppressed upper terminal → "≥ 75" (open format, same as normal)
        assert_eq!(
            result_left.get("≥ 75"),
            Some(&Some("≥ Very High".to_string())),
            "Last bin with suppressed upper should use '≥ lower' format"
        );

        // Test with closed='right'
        let result_right = build_symbol_legend_label_mapping(&breaks, &label_mapping, "right");

        // First bin: suppressed lower terminal → "≤ 25" (right-closed means upper included)
        assert_eq!(
            result_right.get("0 – 25"),
            Some(&Some("≤ Medium".to_string())),
            "First bin with closed='right' should use '≤ upper' format"
        );
        // Last bin: suppressed upper terminal → "> 75" (right-closed means lower not included)
        assert_eq!(
            result_right.get("≥ 75"),
            Some(&Some("> Very High".to_string())),
            "Last bin with closed='right' should use '> lower' format"
        );
    }
}
