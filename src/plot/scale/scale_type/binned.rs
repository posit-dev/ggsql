//! Binned scale type implementation

use std::collections::HashMap;

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait, TransformKind};
use crate::plot::{ArrayElement, ParameterValue};

/// Binned scale type - for binned/bucketed data
#[derive(Debug, Clone, Copy)]
pub struct Binned;

impl ScaleTypeTrait for Binned {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Binned
    }

    fn name(&self) -> &'static str {
        "binned"
    }

    fn allowed_transforms(&self) -> &'static [TransformKind] {
        &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Log2,
            TransformKind::Log,
            TransformKind::Sqrt,
            TransformKind::Asinh,
            TransformKind::PseudoLog,
            // Temporal transforms for date/datetime/time data
            TransformKind::Date,
            TransformKind::DateTime,
            TransformKind::Time,
        ]
    }

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
            "size" => TransformKind::Sqrt, // Area-proportional scaling
            _ => TransformKind::Identity,
        }
    }

    fn allowed_properties(&self, aesthetic: &str) -> &'static [&'static str] {
        if super::is_positional_aesthetic(aesthetic) {
            &["expand", "oob", "reverse", "breaks", "pretty", "closed"]
        } else {
            &["oob", "reverse", "breaks", "pretty", "closed"]
        }
    }

    fn get_property_default(&self, aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            "expand" if super::is_positional_aesthetic(aesthetic) => {
                Some(ParameterValue::Number(super::DEFAULT_EXPAND_MULT))
            }
            "oob" => Some(ParameterValue::String(
                super::default_oob(aesthetic).to_string(),
            )),
            "reverse" => Some(ParameterValue::Boolean(false)),
            "breaks" => Some(ParameterValue::Number(
                super::super::breaks::DEFAULT_BREAK_COUNT as f64,
            )),
            "pretty" => Some(ParameterValue::Boolean(true)),
            // "left" means bins are [lower, upper), "right" means (lower, upper]
            "closed" => Some(ParameterValue::String("left".to_string())),
            _ => None,
        }
    }

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                // Temporal types supported via temporal transforms
                | DataType::Date
                | DataType::Datetime(_, _)
                | DataType::Time
        )
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let computed = compute_numeric_range(columns);
        let (mult, add) = super::get_expand_factors(properties);

        // Apply expansion to computed range
        let expanded = computed.map(|range| super::expand_numeric_range(&range, mult, add));

        match user_range {
            None => Ok(expanded),
            Some(range) if super::input_range_has_nulls(range) => {
                // User provided partial range with nulls - merge with expanded computed
                match expanded {
                    Some(inferred) => Ok(Some(super::merge_with_inferred(range, &inferred))),
                    None => Ok(Some(range.to_vec())),
                }
            }
            Some(range) => {
                // User provided explicit full range - still apply expansion
                Ok(Some(super::expand_numeric_range(range, mult, add)))
            }
        }
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        match aesthetic {
            // Note: "color"/"colour" already split to fill/stroke before scale resolution
            "stroke" | "fill" => {
                let palette = palettes::get_color_palette("sequential")
                    .ok_or_else(|| "Default color palette 'ggsql' not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|col: &&str| ArrayElement::String(col.to_string()))
                        .collect(),
                ))
            }
            "size" | "linewidth" => Ok(Some(vec![
                ArrayElement::Number(1.0),
                ArrayElement::Number(6.0),
            ])),
            "opacity" => Ok(Some(vec![
                ArrayElement::Number(0.1),
                ArrayElement::Number(1.0),
            ])),
            _ => Ok(None),
        }
    }

    /// Generate SQL for pre-stat binning transformation.
    ///
    /// Uses the resolved breaks to compute bin boundaries via CASE WHEN,
    /// mapping each value to its bin center. Supports arbitrary (non-evenly-spaced) breaks.
    ///
    /// The `closed` property controls which side of the bin is closed:
    /// - `"left"` (default): bins are `[lower, upper)`, last bin is `[lower, upper]`
    /// - `"right"`: bins are `(lower, upper]`, first bin is `[lower, upper]`
    ///
    /// This ensures:
    /// - Values are grouped into bins defined by break boundaries
    /// - Each bin is represented by its center value `(lower + upper) / 2`
    /// - Boundary values are not lost (edge bins include endpoints)
    /// - Data is binned BEFORE any stat transforms are applied
    ///
    /// # Type Casting for Mismatched Columns
    ///
    /// When the column's data type doesn't match the transform's target type,
    /// the generated SQL includes CAST expressions:
    ///
    /// - **STRING column + DATE transform** (explicit `VIA date`):
    ///   ```sql
    ///   CASE WHEN CAST(date_col AS DATE) >= CAST('2024-01-01' AS DATE) ...
    ///   ```
    ///
    /// - **STRING column + numeric binning** (no explicit transform):
    ///   ```sql
    ///   CASE WHEN CAST(value AS DOUBLE) >= 0 AND CAST(value AS DOUBLE) < 10 ...
    ///   ```
    ///
    /// - **DATE column + DATE transform** (types match):
    ///   ```sql
    ///   CASE WHEN date_col >= 19724 AND date_col < 19755 ... -- efficient numeric comparison
    ///   ```
    ///
    /// The transform must be **explicit** (user specified `VIA date`, etc.) for casting
    /// to be applied. If no transform is specified, the column type is used as-is.
    fn pre_stat_transform_sql(
        &self,
        column_name: &str,
        column_dtype: &DataType,
        scale: &super::super::Scale,
        type_names: &super::SqlTypeNames,
    ) -> Option<String> {
        use super::super::transform::TransformKind;
        use super::CastTargetType;

        // Get breaks from scale properties (calculated in resolve)
        // breaks should be an Array after resolution
        let breaks = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(arr)) => arr,
            _ => return None,
        };

        if breaks.len() < 2 {
            return None;
        }

        // Extract numeric break values (handles Number, Date, DateTime, Time via to_f64)
        let break_values: Vec<f64> = breaks.iter().filter_map(|e| e.to_f64()).collect();

        if break_values.len() < 2 {
            return None;
        }

        // Get closed property: "left" (default) or "right"
        let closed_left = match scale.properties.get("closed") {
            Some(ParameterValue::String(s)) => s != "right",
            _ => true, // default to left-closed
        };

        // Determine if we need casting based on explicit transform and column type mismatch
        // Only consider casting if the user explicitly specified a transform (scale.transform is Some
        // and was set by the user, not inferred from dtype). We check if the transform was explicitly
        // set by looking at scale.explicit_transform flag.
        let transform = scale.transform.as_ref();
        let explicit_transform = scale.explicit_transform;

        // Determine target type and whether casting is needed
        let (cast_info, use_iso_values) = if explicit_transform {
            if let Some(t) = transform {
                match t.transform_kind() {
                    TransformKind::Date => {
                        let needs_cast = !matches!(column_dtype, DataType::Date);
                        if needs_cast {
                            (
                                type_names
                                    .for_target(CastTargetType::Date)
                                    .map(|name| (name, CastTargetType::Date)),
                                true, // Use ISO strings for temporal casts
                            )
                        } else {
                            (None, false)
                        }
                    }
                    TransformKind::DateTime => {
                        let needs_cast = !matches!(column_dtype, DataType::Datetime(..));
                        if needs_cast {
                            (
                                type_names
                                    .for_target(CastTargetType::DateTime)
                                    .map(|name| (name, CastTargetType::DateTime)),
                                true,
                            )
                        } else {
                            (None, false)
                        }
                    }
                    TransformKind::Time => {
                        let needs_cast = !matches!(column_dtype, DataType::Time);
                        if needs_cast {
                            (
                                type_names
                                    .for_target(CastTargetType::Time)
                                    .map(|name| (name, CastTargetType::Time)),
                                true,
                            )
                        } else {
                            (None, false)
                        }
                    }
                    // For non-temporal transforms (Identity, Log, Sqrt, etc.), check if column is numeric
                    _ => {
                        let is_numeric = matches!(
                            column_dtype,
                            DataType::Int8
                                | DataType::Int16
                                | DataType::Int32
                                | DataType::Int64
                                | DataType::UInt8
                                | DataType::UInt16
                                | DataType::UInt32
                                | DataType::UInt64
                                | DataType::Float32
                                | DataType::Float64
                        );
                        if !is_numeric {
                            (
                                type_names
                                    .for_target(CastTargetType::Number)
                                    .map(|name| (name, CastTargetType::Number)),
                                false, // Numeric values, not ISO strings
                            )
                        } else {
                            (None, false)
                        }
                    }
                }
            } else {
                (None, false)
            }
        } else {
            // No explicit transform - check if column is numeric for numeric binning
            let is_numeric = matches!(
                column_dtype,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Date
                    | DataType::Datetime(..)
                    | DataType::Time
            );
            if !is_numeric {
                // String column without explicit transform - cast to number
                (
                    type_names
                        .for_target(CastTargetType::Number)
                        .map(|name| (name, CastTargetType::Number)),
                    false,
                )
            } else {
                (None, false)
            }
        };

        // Build CASE WHEN clauses for each bin
        let num_bins = break_values.len() - 1;
        let mut cases = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            let lower = break_values[i];
            let upper = break_values[i + 1];
            let center = (lower + upper) / 2.0;

            let is_first = i == 0;
            let is_last = i == num_bins - 1;

            // Format column and values based on casting requirements
            let (col_expr, lower_expr, upper_expr, center_expr) =
                if let Some((type_name, _target_type)) = &cast_info {
                    let cast_col = format!("CAST({} AS {})", column_name, type_name);
                    if use_iso_values {
                        // Temporal: cast both column AND values to the target type
                        let t = transform.unwrap();
                        let lower_iso = t
                            .format_as_iso(lower)
                            .unwrap_or_else(|| format!("{}", lower));
                        let upper_iso = t
                            .format_as_iso(upper)
                            .unwrap_or_else(|| format!("{}", upper));
                        let center_iso = t
                            .format_as_iso(center)
                            .unwrap_or_else(|| format!("{}", center));
                        (
                            cast_col,
                            format!("CAST('{}' AS {})", lower_iso, type_name),
                            format!("CAST('{}' AS {})", upper_iso, type_name),
                            format!("CAST('{}' AS {})", center_iso, type_name),
                        )
                    } else {
                        // Numeric: cast column only, values are already numeric
                        (
                            cast_col,
                            format!("{}", lower),
                            format!("{}", upper),
                            format!("{}", center),
                        )
                    }
                } else {
                    // No casting needed - use raw values
                    (
                        column_name.to_string(),
                        format!("{}", lower),
                        format!("{}", upper),
                        format!("{}", center),
                    )
                };

            // Build the condition based on closed side
            // closed="left": [lower, upper) except last bin which is [lower, upper]
            // closed="right": (lower, upper] except first bin which is [lower, upper]
            let condition = if closed_left {
                if is_last {
                    // Last bin: [lower, upper] (inclusive on both ends)
                    format!(
                        "{} >= {} AND {} <= {}",
                        col_expr, lower_expr, col_expr, upper_expr
                    )
                } else {
                    // Normal bin: [lower, upper)
                    format!(
                        "{} >= {} AND {} < {}",
                        col_expr, lower_expr, col_expr, upper_expr
                    )
                }
            } else {
                // closed="right"
                if is_first {
                    // First bin: [lower, upper] (inclusive on both ends)
                    format!(
                        "{} >= {} AND {} <= {}",
                        col_expr, lower_expr, col_expr, upper_expr
                    )
                } else {
                    // Normal bin: (lower, upper]
                    format!(
                        "{} > {} AND {} <= {}",
                        col_expr, lower_expr, col_expr, upper_expr
                    )
                }
            };

            cases.push(format!("WHEN {} THEN {}", condition, center_expr));
        }

        // Build final CASE expression
        Some(format!("(CASE {} ELSE NULL END)", cases.join(" ")))
    }
}

/// Compute numeric input range as [min, max] from Columns.
fn compute_numeric_range(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<f64> = None;
    let mut global_max: Option<f64> = None;

    for column in column_refs {
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

impl std::fmt::Display for Binned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scale::{Scale, SqlTypeNames};

    /// Helper to create default type names for tests
    fn test_type_names() -> SqlTypeNames {
        SqlTypeNames::duckdb()
    }

    #[test]
    fn test_pre_stat_transform_sql_even_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(30.0),
            ]),
        );

        // Float64 column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // Should produce CASE WHEN with bin centers 5, 15, 25
        assert!(sql.contains("CASE"));
        assert!(sql.contains("WHEN value >= 0 AND value < 10 THEN 5"));
        assert!(sql.contains("WHEN value >= 10 AND value < 20 THEN 15"));
        // Last bin should be inclusive on both ends
        assert!(sql.contains("WHEN value >= 20 AND value <= 30 THEN 25"));
        assert!(sql.contains("ELSE NULL END"));
    }

    #[test]
    fn test_pre_stat_transform_sql_uneven_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        // Non-evenly-spaced breaks: [0, 10, 25, 100]
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(100.0),
            ]),
        );

        let sql = binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // Bin centers: (0+10)/2=5, (10+25)/2=17.5, (25+100)/2=62.5
        assert!(sql.contains("THEN 5")); // center of [0, 10)
        assert!(sql.contains("THEN 17.5")); // center of [10, 25)
        assert!(sql.contains("THEN 62.5")); // center of [25, 100]
    }

    #[test]
    fn test_pre_stat_transform_sql_closed_left_default() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );
        // No explicit closed property, should default to "left"

        let sql = binned
            .pre_stat_transform_sql("col", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // closed="left": [lower, upper) except last which is [lower, upper]
        assert!(sql.contains("col >= 0 AND col < 10"));
        assert!(sql.contains("col >= 10 AND col <= 20")); // last bin inclusive
    }

    #[test]
    fn test_pre_stat_transform_sql_closed_right() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );
        scale.properties.insert(
            "closed".to_string(),
            ParameterValue::String("right".to_string()),
        );

        let sql = binned
            .pre_stat_transform_sql("col", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // closed="right": first bin is [lower, upper], rest are (lower, upper]
        assert!(sql.contains("col >= 0 AND col <= 10")); // first bin inclusive
        assert!(sql.contains("col > 10 AND col <= 20"));
    }

    #[test]
    fn test_pre_stat_transform_sql_insufficient_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Only one break - not enough to form a bin
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0)]),
        );

        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_no_breaks() {
        let binned = Binned;
        let scale = Scale::new("x");
        // No breaks property at all

        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_number_breaks_returns_none() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        // breaks is still a Number (count), not resolved to Array yet
        scale
            .properties
            .insert("breaks".to_string(), ParameterValue::Number(5.0));

        // Should return None because breaks hasn't been resolved to Array
        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_closed_property_default() {
        let binned = Binned;
        let default = binned.get_property_default("x", "closed");
        assert_eq!(default, Some(ParameterValue::String("left".to_string())));
    }

    #[test]
    fn test_closed_property_allowed() {
        let binned = Binned;
        let allowed = binned.allowed_properties("x");
        assert!(allowed.contains(&"closed"));
    }

    #[test]
    fn test_pre_stat_transform_sql_with_date_breaks() {
        // Test that Date breaks are correctly handled via to_f64()
        // When column is DATE and no explicit transform, use efficient numeric comparison
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use Date variants instead of Number
        // 2024-01-01 = 19724 days, 2024-02-01 = 19755 days, 2024-03-01 = 19784 days
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(19724), // 2024-01-01
                ArrayElement::Date(19755), // 2024-02-01
                ArrayElement::Date(19784), // 2024-03-01
            ]),
        );

        // Date column - no casting needed (types match)
        let sql =
            binned.pre_stat_transform_sql("date_col", &DataType::Date, &scale, &test_type_names());

        // Should successfully generate SQL (not return None due to filtered-out breaks)
        assert!(sql.is_some(), "SQL should be generated for Date breaks");
        let sql = sql.unwrap();

        // Verify the SQL contains the expected day values (numeric comparison)
        assert!(
            sql.contains("19724"),
            "SQL should contain first break value"
        );
        assert!(
            sql.contains("19755"),
            "SQL should contain second break value"
        );
        assert!(
            sql.contains("19784"),
            "SQL should contain third break value"
        );

        // Verify bin centers: (19724+19755)/2 = 19739.5, (19755+19784)/2 = 19769.5
        assert!(
            sql.contains("THEN 19739.5"),
            "SQL should contain first bin center"
        );
        assert!(
            sql.contains("THEN 19769.5"),
            "SQL should contain second bin center"
        );
    }

    #[test]
    fn test_pre_stat_transform_sql_with_datetime_breaks() {
        // Test that DateTime breaks are correctly handled via to_f64()
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use DateTime variants (microseconds since epoch)
        // Some arbitrary microsecond values for testing
        let dt1: i64 = 1_704_067_200_000_000; // 2024-01-01 00:00:00 UTC
        let dt2: i64 = 1_706_745_600_000_000; // 2024-02-01 00:00:00 UTC
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::DateTime(dt1),
                ArrayElement::DateTime(dt2),
            ]),
        );

        use polars::prelude::TimeUnit;
        let sql = binned.pre_stat_transform_sql(
            "datetime_col",
            &DataType::Datetime(TimeUnit::Microseconds, None),
            &scale,
            &test_type_names(),
        );

        // Should successfully generate SQL
        assert!(sql.is_some(), "SQL should be generated for DateTime breaks");
    }

    #[test]
    fn test_pre_stat_transform_sql_with_time_breaks() {
        // Test that Time breaks are correctly handled via to_f64()
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use Time variants (nanoseconds since midnight)
        // 6:00 AM = 6 * 60 * 60 * 1_000_000_000 ns
        // 12:00 PM = 12 * 60 * 60 * 1_000_000_000 ns
        // 18:00 PM = 18 * 60 * 60 * 1_000_000_000 ns
        let t1: i64 = 6 * 60 * 60 * 1_000_000_000;
        let t2: i64 = 12 * 60 * 60 * 1_000_000_000;
        let t3: i64 = 18 * 60 * 60 * 1_000_000_000;
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Time(t1),
                ArrayElement::Time(t2),
                ArrayElement::Time(t3),
            ]),
        );

        let sql =
            binned.pre_stat_transform_sql("time_col", &DataType::Time, &scale, &test_type_names());

        // Should successfully generate SQL
        assert!(sql.is_some(), "SQL should be generated for Time breaks");
    }

    // ==========================================================================
    // Type Casting Tests
    // ==========================================================================

    #[test]
    fn test_string_column_with_explicit_date_transform_casts() {
        // STRING column + explicit date transform → type mismatch → CAST
        use crate::plot::scale::transform::Transform;

        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::date());
        scale.explicit_transform = true; // User specified VIA date

        // Date breaks: 2024-01-02 to 2024-03-02 (days since epoch)
        // 19724 days from 1970-01-01 = 2024-01-02
        // 19755 days from 1970-01-01 = 2024-02-02
        // 19784 days from 1970-01-01 = 2024-03-02
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(19724), // 2024-01-02
                ArrayElement::Date(19755), // 2024-02-02
                ArrayElement::Date(19784), // 2024-03-02
            ]),
        );

        // String column - needs casting
        let sql = binned
            .pre_stat_transform_sql("date_col", &DataType::String, &scale, &test_type_names())
            .unwrap();

        // Should contain CAST expressions
        assert!(
            sql.contains("CAST(date_col AS DATE)"),
            "SQL should cast column to DATE. Got: {}",
            sql
        );
        // Break values should be cast as ISO date strings
        assert!(
            sql.contains("CAST('2024-01-02' AS DATE)"),
            "SQL should cast break value to DATE. Got: {}",
            sql
        );
        assert!(
            sql.contains("CAST('2024-02-02' AS DATE)"),
            "SQL should cast break value to DATE. Got: {}",
            sql
        );
    }

    #[test]
    fn test_date_column_with_date_transform_no_cast() {
        // DATE column + date transform → types match → no cast needed
        use crate::plot::scale::transform::Transform;

        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::date());
        scale.explicit_transform = true; // User specified VIA date

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(19724), // 2024-01-01
                ArrayElement::Date(19755), // 2024-02-01
            ]),
        );

        // Date column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("date_col", &DataType::Date, &scale, &test_type_names())
            .unwrap();

        // Should NOT contain CAST expressions (efficient numeric comparison)
        assert!(
            !sql.contains("CAST("),
            "SQL should not contain CAST when types match"
        );
        assert!(
            sql.contains("date_col >= 19724"),
            "SQL should use raw numeric values"
        );
    }

    #[test]
    fn test_string_column_without_explicit_transform_casts_to_number() {
        // STRING column + no explicit transform → cast to DOUBLE
        let binned = Binned;
        let mut scale = Scale::new("x");
        // No explicit transform set

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );

        // String column - needs casting to DOUBLE
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::String, &scale, &test_type_names())
            .unwrap();

        // Should contain CAST to DOUBLE
        assert!(
            sql.contains("CAST(value AS DOUBLE)"),
            "SQL should cast string column to DOUBLE"
        );
        // Values should NOT be cast (they're already numeric)
        assert!(
            sql.contains(">= 0 AND"),
            "SQL should use raw numeric values for comparison"
        );
    }

    #[test]
    fn test_numeric_column_no_cast() {
        // INT64 column + no explicit transform → no cast needed
        let binned = Binned;
        let mut scale = Scale::new("x");

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );

        // Int64 column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::Int64, &scale, &test_type_names())
            .unwrap();

        // Should NOT contain CAST expressions
        assert!(
            !sql.contains("CAST("),
            "SQL should not contain CAST when column is numeric"
        );
        assert!(sql.contains("value >= 0"), "SQL should use raw column name");
    }
}
