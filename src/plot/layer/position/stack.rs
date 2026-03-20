//! Stack position adjustments
//!
//! Implements stacking of elements: normal, fill (normalized), and center.
//!
//! Stacking automatically detects which axis is continuous and stacks accordingly:
//! - If pos2 is continuous → stack vertically (modify pos2/pos2end)
//! - If pos1 is continuous and pos2 is discrete → stack horizontally (modify pos1/pos1end)

use super::{is_continuous_scale, Layer, PositionTrait, PositionType};
use crate::plot::types::{ParamConstraint, ParamDefinition, DefaultParamValue, ParameterValue};
use crate::{naming, DataFrame, GgsqlError, Plot, Result};
use polars::prelude::*;

/// Stack mode for position adjustments
#[derive(Clone, Copy)]
enum StackMode {
    /// Normal stacking (cumsum from 0)
    Normal,
    /// Normalized stacking (cumsum / total, then scaled to target)
    Fill(f64),
    /// Centered stacking (cumsum - total/2, centered at 0)
    Center,
}

/// Stack position - stack elements vertically
#[derive(Debug, Clone, Copy)]
pub struct Stack;

impl std::fmt::Display for Stack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stack")
    }
}

impl PositionTrait for Stack {
    fn position_type(&self) -> PositionType {
        PositionType::Stack
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "center",
                default: DefaultParamValue::Boolean(false),
                constraint: ParamConstraint::boolean(),
            },
            ParamDefinition {
                name: "total",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
        ];
        PARAMS
    }

    fn apply_adjustment(
        &self,
        df: DataFrame,
        layer: &Layer,
        spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)> {
        let center = layer
            .parameters
            .get("center")
            .and_then(|v| match v {
                ParameterValue::Boolean(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        let total = layer.parameters.get("total").and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        });

        let mode = match (center, total) {
            (true, _) => StackMode::Center,
            (false, Some(target)) => StackMode::Fill(target),
            (false, None) => StackMode::Normal,
        };
        Ok((apply_stack(df, layer, spec, mode)?, None))
    }
}

/// Direction for stacking
#[derive(Clone, Copy)]
enum StackDirection {
    /// Stack vertically (modify pos2/pos2end, group by pos1)
    Vertical,
    /// Stack horizontally (modify pos1/pos1end, group by pos2)
    Horizontal,
}

/// Check if an axis is stackable.
///
/// An axis is stackable if:
/// 1. It has a continuous scale (scale type is always known after create_missing_scales_post_stat)
/// 2. It has a pos/posend pair (e.g., pos2/pos2end) or posmin/posmax pair
/// 3. Every row has a zero baseline in one of the range columns
fn is_axis_stackable(spec: &Plot, layer: &Layer, df: &DataFrame, axis: &str) -> bool {
    // Must be continuous (scale type always known after create_missing_scales_post_stat)
    if is_continuous_scale(spec, axis) != Some(true) {
        return false;
    }

    // Helper to check if an aesthetic is defined in mappings OR remappings
    // (remappings come from default_remappings for stat geoms like bar)
    let has_aesthetic = |aes: &str| -> bool {
        layer.mappings.contains_key(aes) || layer.remappings.contains_key(aes)
    };

    // Check for pos/posend pair (e.g., pos2/pos2end)
    let end_aesthetic = format!("{}end", axis);
    let has_end_pair = has_aesthetic(axis) && has_aesthetic(&end_aesthetic);

    // Check for posmin/posmax pair (e.g., pos2min/pos2max)
    let min_aesthetic = format!("{}min", axis);
    let max_aesthetic = format!("{}max", axis);
    let has_minmax_pair = has_aesthetic(&min_aesthetic) && has_aesthetic(&max_aesthetic);

    // Check that each row has zero baseline in one of the range columns
    if has_end_pair {
        let pos_col = naming::aesthetic_column(axis);
        let end_col = naming::aesthetic_column(&end_aesthetic);
        if has_zero_baseline_per_row(df, &pos_col, &end_col) {
            return true;
        }
    }
    if has_minmax_pair {
        let min_col = naming::aesthetic_column(&min_aesthetic);
        let max_col = naming::aesthetic_column(&max_aesthetic);
        if has_zero_baseline_per_row(df, &min_col, &max_col) {
            return true;
        }
    }
    false
}

/// Check that for every row, at least one of the two columns is zero.
fn has_zero_baseline_per_row(df: &DataFrame, col_a: &str, col_b: &str) -> bool {
    let (Ok(a), Ok(b)) = (df.column(col_a), df.column(col_b)) else {
        return false;
    };

    // Cast columns to f64 for comparison - handle both Int64 and Float64 sources
    let Ok(a_casted) = a.cast(&polars::datatypes::DataType::Float64) else {
        return false;
    };
    let Ok(b_casted) = b.cast(&polars::datatypes::DataType::Float64) else {
        return false;
    };

    let Ok(a_vals) = a_casted.f64() else {
        return false;
    };
    let Ok(b_vals) = b_casted.f64() else {
        return false;
    };

    // Collect values to avoid borrow issues
    let a_vec: Vec<Option<f64>> = a_vals.into_iter().collect();
    let b_vec: Vec<Option<f64>> = b_vals.into_iter().collect();

    // For each row, either a or b must be 0
    a_vec
        .into_iter()
        .zip(b_vec)
        .all(|(a_val, b_val)| a_val == Some(0.0) || b_val == Some(0.0))
}

/// Determine stacking direction based on scale types and axis configuration.
///
/// An axis is stackable if it's continuous AND has pos/posend or posmin/posmax pairs
/// AND has zero baseline per row.
///
/// Returns:
/// - Vertical if pos2 is stackable and pos1 is not
/// - Horizontal if pos1 is stackable and pos2 is not
/// - Vertical as default (for backward compatibility)
fn determine_stack_direction(spec: &Plot, layer: &Layer, df: &DataFrame) -> Option<StackDirection> {
    let pos1_stackable = is_axis_stackable(spec, layer, df, "pos1");
    let pos2_stackable = is_axis_stackable(spec, layer, df, "pos2");

    match (pos1_stackable, pos2_stackable) {
        (false, true) => Some(StackDirection::Vertical),
        (true, false) => Some(StackDirection::Horizontal),
        _ => None,
    }
}

/// Apply stack position adjustment.
///
/// Automatically detects stacking direction based on scale types:
/// - Vertical stacking: for each unique pos1 value, compute cumulative sums of pos2
/// - Horizontal stacking: for each unique pos2 value, compute cumulative sums of pos1
///
/// Modes:
/// - Normal: standard stacking from 0
/// - Fill: normalized to sum to 1 (100% stacked)
/// - Center: centered around 0 (streamgraph style)
fn apply_stack(df: DataFrame, layer: &Layer, spec: &Plot, mode: StackMode) -> Result<DataFrame> {
    // Determine stacking direction
    let Some(direction) = determine_stack_direction(spec, layer, &df) else {
        return Ok(df);
    };

    // Set up column names based on direction
    let (stack_col, stack_end_col, group_col) = match direction {
        StackDirection::Vertical => (
            naming::aesthetic_column("pos2"),
            naming::aesthetic_column("pos2end"),
            naming::aesthetic_column("pos1"),
        ),
        StackDirection::Horizontal => (
            naming::aesthetic_column("pos1"),
            naming::aesthetic_column("pos1end"),
            naming::aesthetic_column("pos2"),
        ),
    };

    // Check if required columns exist
    if df.column(&stack_col).is_err() {
        return Ok(df);
    }

    // Stacking currently only supports non-negative values
    let min_result = df
        .clone()
        .lazy()
        .select([col(&stack_col).min()])
        .collect()
        .map_err(|e| GgsqlError::InternalError(format!("Failed to check min value: {}", e)))?;

    if let Some(min_col) = min_result.get_columns().first() {
        if let Ok(min_val) = min_col.get(0) {
            if let Ok(min) = min_val.try_extract::<f64>() {
                if min < 0.0 {
                    let axis = match direction {
                        StackDirection::Vertical => "y",
                        StackDirection::Horizontal => "x",
                    };
                    return Err(GgsqlError::ValidationError(format!(
                        "position 'stack' requires non-negative {} values",
                        axis
                    )));
                }
            }
        }
    }

    // Convert to lazy for transformations
    let lf = df.lazy();

    // Sort by group column and partition_by columns to ensure consistent stacking order
    // This ensures that within each group (e.g., x position), the stacking order is
    // consistent even if data arrives in different orders or has missing values
    let mut sort_cols = vec![col(&group_col)];
    for partition_col in &layer.partition_by {
        sort_cols.push(col(partition_col));
    }
    let sort_options = SortMultipleOptions::default();
    let lf = lf.sort_by_exprs(&sort_cols, sort_options);

    // For stacking, compute cumulative sums within each group:
    // 1. stack_col = cumulative sum (the bar top/end)
    // 2. stack_end_col = lag(stack_col, 1, 0) - the bar bottom/start (previous stack top)
    // The cumsum naturally stacks across the grouping column values

    // Treat NA heights as 0 for stacking
    // Compute cumulative sums (shared by all modes)
    let lf = lf
        .with_column(col(&stack_col).fill_null(lit(0.0)).alias(&stack_col))
        .with_column(
            col(&stack_col)
                .cum_sum(false)
                .over([col(&group_col)])
                .alias("__cumsum__"),
        )
        .with_column(
            col(&stack_col)
                .cum_sum(false)
                .shift(lit(1))
                .fill_null(lit(0.0))
                .over([col(&group_col)])
                .alias("__cumsum_lag__"),
        );

    // Apply mode-specific transformation
    let (stack_expr, stack_end_expr, temp_cols): (Expr, Expr, Vec<&str>) = match mode {
        StackMode::Normal => (
            col("__cumsum__").alias(&stack_col),
            col("__cumsum_lag__").alias(&stack_end_col),
            vec!["__cumsum__", "__cumsum_lag__"],
        ),
        StackMode::Fill(target) => {
            let total = col(&stack_col).sum().over([col(&group_col)]);
            (
                (col("__cumsum__") / total.clone() * lit(target)).alias(&stack_col),
                (col("__cumsum_lag__") / total * lit(target)).alias(&stack_end_col),
                vec!["__cumsum__", "__cumsum_lag__"],
            )
        }
        StackMode::Center => {
            let half_total = col(&stack_col).sum().over([col(&group_col)]) / lit(2.0);
            (
                (col("__cumsum__") - half_total.clone()).alias(&stack_col),
                (col("__cumsum_lag__") - half_total).alias(&stack_end_col),
                vec!["__cumsum__", "__cumsum_lag__"],
            )
        }
    };

    let mut result = lf
        .with_columns([stack_expr, stack_end_expr])
        .collect()
        .map_err(|e| {
            GgsqlError::InternalError(format!("Stack position adjustment failed: {}", e))
        })?;

    for col_name in temp_cols {
        result = result
            .drop(col_name)
            .map_err(|e| GgsqlError::InternalError(format!("Failed to drop temp column: {}", e)))?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::layer::Geom;
    use crate::plot::{AestheticValue, Mappings};

    fn make_test_df() -> DataFrame {
        df! {
            "__ggsql_aes_pos1__" => ["A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "Y", "X", "Y"],
        }
        .unwrap()
    }

    fn make_test_layer() -> Layer {
        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];
        layer
    }

    #[test]
    fn test_stack_cumsum() {
        let stack = Stack;
        assert_eq!(stack.position_type(), PositionType::Stack);

        let df = make_test_df();
        let layer = make_test_layer();
        let spec = Plot::new();

        let (result, width) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        assert!(width.is_none());
        let pos2_col = result.column("__ggsql_aes_pos2__").unwrap();
        let pos2end_col = result.column("__ggsql_aes_pos2end__").unwrap();

        assert!(pos2_col.f64().is_ok() || pos2_col.i64().is_ok());
        assert!(pos2end_col.f64().is_ok() || pos2end_col.i64().is_ok());
    }

    #[test]
    fn test_stack_default_params() {
        let stack = Stack;
        let params = stack.default_params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "center");
        assert!(matches!(
            params[0].default,
            DefaultParamValue::Boolean(false)
        ));
        assert_eq!(params[1].name, "total");
        assert!(matches!(params[1].default, DefaultParamValue::Null));
    }

    #[test]
    fn test_stack_center_parameter() {
        let stack = Stack;
        let df = make_test_df();
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));

        // Test default (center = false) - should stack from 0
        let layer = make_test_layer();
        let (result_normal, _) = stack.apply_adjustment(df.clone(), &layer, &spec).unwrap();

        // Test with center = true - should center around 0
        let mut layer_centered = make_test_layer();
        layer_centered
            .parameters
            .insert("center".to_string(), ParameterValue::Boolean(true));
        let (result_centered, _) = stack.apply_adjustment(df, &layer_centered, &spec).unwrap();

        // Normal stacking should have pos2end starting at 0
        let pos2end_normal = result_normal.column("__ggsql_aes_pos2end__").unwrap();
        let first_normal = pos2end_normal.get(0).unwrap();
        // First element's pos2end should be 0 for normal stack
        if let polars::prelude::AnyValue::Float64(v) = first_normal {
            assert_eq!(v, 0.0);
        }

        // Centered stacking should have negative values
        let pos2end_centered = result_centered.column("__ggsql_aes_pos2end__").unwrap();
        let first_centered = pos2end_centered.get(0).unwrap();
        // First element's pos2end should be negative for centered stack (shifted by -total/2)
        if let polars::prelude::AnyValue::Float64(v) = first_centered {
            assert!(
                v < 0.0,
                "Centered stack should have negative pos2end for first element"
            );
        }
    }

    fn make_continuous_scale(aesthetic: &str) -> crate::plot::Scale {
        let mut scale = crate::plot::Scale::new(aesthetic);
        scale.scale_type = Some(crate::plot::ScaleType::continuous());
        scale
    }

    fn make_discrete_scale(aesthetic: &str) -> crate::plot::Scale {
        let mut scale = crate::plot::Scale::new(aesthetic);
        scale.scale_type = Some(crate::plot::ScaleType::discrete());
        scale
    }

    #[test]
    fn test_stack_vertical_when_pos2_continuous() {
        // Default case: pos2 continuous -> stack vertically
        let stack = Stack;
        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        // pos2 should be modified (stacked)
        assert!(result.column("__ggsql_aes_pos2__").is_ok());
        assert!(result.column("__ggsql_aes_pos2end__").is_ok());
    }

    #[test]
    fn test_stack_horizontal_when_pos1_continuous() {
        // When pos1 is continuous and pos2 is discrete -> stack horizontally
        let stack = Stack;

        // Create data with numeric pos1 values and pos1end column with zero baselines
        let df = df! {
            "__ggsql_aes_pos1__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos1end__" => [0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_pos2__" => ["A", "A", "B", "B"],
            "__ggsql_aes_fill__" => ["X", "Y", "X", "Y"],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos1end",
                AestheticValue::standard_column("__ggsql_aes_pos1end__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];

        // Mark pos1 as continuous, pos2 as discrete
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        // pos1 should be modified (stacked horizontally)
        assert!(
            result.column("__ggsql_aes_pos1__").is_ok(),
            "pos1 column should exist"
        );
        assert!(
            result.column("__ggsql_aes_pos1end__").is_ok(),
            "pos1end column should be created for horizontal stacking"
        );

        // Verify stacking occurred - values should be cumulative sums
        let pos1_col = result.column("__ggsql_aes_pos1__").unwrap();
        let pos1_vals: Vec<f64> = pos1_col.f64().unwrap().into_iter().flatten().collect();

        // Should have cumulative sums (10, 30, 15, 40) for groups A and B
        assert!(
            pos1_vals.iter().any(|&v| v > 20.0),
            "Should have cumulative values > original max, got {:?}",
            pos1_vals
        );
    }

    #[test]
    fn test_stack_total_parameter() {
        let stack = Stack;
        let df = make_test_df();
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));

        // Test with total = 100 (percentage stacking)
        let mut layer = make_test_layer();
        layer
            .parameters
            .insert("total".to_string(), ParameterValue::Number(100.0));

        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        // pos2 should sum to 100 within each group (A and B)
        let pos2_col = result.column("__ggsql_aes_pos2__").unwrap();
        let pos2_vals: Vec<f64> = pos2_col.f64().unwrap().into_iter().flatten().collect();

        // For group A: values 10, 20 -> normalized: 10/30, 20/30 -> cumsum: 10/30, 30/30
        // Multiplied by 100: ~33.33, 100
        // For group B: values 15, 25 -> normalized: 15/40, 25/40 -> cumsum: 15/40, 40/40
        // Multiplied by 100: 37.5, 100
        // So max values should be 100
        let max_val = pos2_vals.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (max_val - 100.0).abs() < 0.01,
            "Expected max value ~100 for normalized stack, got {}",
            max_val
        );
    }

    #[test]
    fn test_stack_total_parameter_arbitrary_value() {
        let stack = Stack;
        let df = make_test_df();
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));

        // Test with total = 1 (normalized to 1, like old stack_fill behavior)
        let mut layer = make_test_layer();
        layer
            .parameters
            .insert("total".to_string(), ParameterValue::Number(1.0));

        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        let pos2_col = result.column("__ggsql_aes_pos2__").unwrap();
        let pos2_vals: Vec<f64> = pos2_col.f64().unwrap().into_iter().flatten().collect();

        // Max values should be 1 (normalized to sum to 1)
        let max_val = pos2_vals.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (max_val - 1.0).abs() < 0.01,
            "Expected max value ~1 for normalized stack with total=1, got {}",
            max_val
        );
    }

    #[test]
    fn test_stack_na_values_treated_as_zero() {
        let stack = Stack;

        // Create data with NA values in pos2
        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "B", "B", "B"],
            "__ggsql_aes_pos2__" => [Some(10.0), None, Some(20.0), Some(15.0), Some(25.0), None],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "Y", "Z", "X", "Y", "Z"],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];

        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));
        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        // Get pos2 values - should have no nulls after stacking
        let pos2_col = result.column("__ggsql_aes_pos2__").unwrap();
        let pos2_vals: Vec<Option<f64>> = pos2_col.f64().unwrap().into_iter().collect();

        // All values should be non-null (NA treated as 0)
        assert!(
            pos2_vals.iter().all(|v| v.is_some()),
            "Expected no null values after stacking, got {:?}",
            pos2_vals
        );

        // For group A: 10, 0 (NA), 20 -> cumsum: 10, 10, 30
        // For group B: 15, 25, 0 (NA) -> cumsum: 15, 40, 40
        // Check that the cumsum for group A ends at 30 (10 + 0 + 20)
        let group_a_max = pos2_vals[2].unwrap(); // Third row is last for group A
        assert!(
            (group_a_max - 30.0).abs() < 0.01,
            "Expected group A max ~30 (NA treated as 0), got {}",
            group_a_max
        );
    }

    #[test]
    fn test_stack_consistent_order_with_shuffled_data() {
        let stack = Stack;

        // Create data in shuffled order - categories not in order within groups
        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "B", "A", "B", "A", "B"],
            "__ggsql_aes_pos2__" => [10.0, 15.0, 30.0, 35.0, 20.0, 25.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "X", "Z", "Z", "Y", "Y"],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];

        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos2"));
        let (result, _) = stack.apply_adjustment(df, &layer, &spec).unwrap();

        // After sorting by pos1 then fill, the order should be:
        // A-X(10), A-Y(20), A-Z(30) -> cumsum: 10, 30, 60
        // B-X(15), B-Y(25), B-Z(35) -> cumsum: 15, 40, 75

        // Check that data is sorted consistently
        let pos1_col = result.column("__ggsql_aes_pos1__").unwrap();
        let fill_col = result.column("__ggsql_aes_fill__").unwrap();
        let pos2_col = result.column("__ggsql_aes_pos2__").unwrap();

        let pos1_vals: Vec<&str> = pos1_col.str().unwrap().into_iter().flatten().collect();
        let fill_vals: Vec<&str> = fill_col.str().unwrap().into_iter().flatten().collect();
        let pos2_vals: Vec<f64> = pos2_col.f64().unwrap().into_iter().flatten().collect();

        // Should be sorted: A-X, A-Y, A-Z, B-X, B-Y, B-Z
        assert_eq!(pos1_vals, vec!["A", "A", "A", "B", "B", "B"]);
        assert_eq!(fill_vals, vec!["X", "Y", "Z", "X", "Y", "Z"]);

        // Group A cumsum: 10, 30, 60
        assert!((pos2_vals[0] - 10.0).abs() < 0.01, "A-X should be 10");
        assert!((pos2_vals[1] - 30.0).abs() < 0.01, "A-Y should be 30");
        assert!((pos2_vals[2] - 60.0).abs() < 0.01, "A-Z should be 60");

        // Group B cumsum: 15, 40, 75
        assert!((pos2_vals[3] - 15.0).abs() < 0.01, "B-X should be 15");
        assert!((pos2_vals[4] - 40.0).abs() < 0.01, "B-Y should be 40");
        assert!((pos2_vals[5] - 75.0).abs() < 0.01, "B-Z should be 75");
    }
}
