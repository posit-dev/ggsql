//! Grouped window operations for position adjustments.
//!
//! Replaces polars lazy evaluation (`cum_sum().over()`, `shift()`, etc.)
//! with direct arrow compute operations. Used primarily by stack.rs.

use arrow::array::{Array, ArrayRef, Float64Array, UInt32Array};
use arrow::compute;
use arrow::compute::SortOptions;

use crate::array_util::{as_f64, fill_null_f64, value_to_string};
use crate::dataframe::DataFrame;
use crate::{GgsqlError, Result};

// ============================================================================
// Sorting
// ============================================================================

/// Sort a DataFrame by multiple columns (all ascending).
pub fn sort_dataframe(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
    if columns.is_empty() || df.height() == 0 {
        return Ok(df.clone());
    }

    // Build sort columns for lexsort
    let sort_columns: Vec<arrow::compute::SortColumn> = columns
        .iter()
        .map(|&name| {
            let col = df.column(name)?;
            Ok(arrow::compute::SortColumn {
                values: col.clone(),
                options: Some(SortOptions::default()),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let indices = compute::lexsort_to_indices(&sort_columns, None).map_err(|e| {
        GgsqlError::InternalError(format!("Failed to sort by {:?}: {}", columns, e))
    })?;

    reorder_by_indices(df, &indices)
}

/// Reorder all columns of a DataFrame using an index array.
fn reorder_by_indices(df: &DataFrame, indices: &UInt32Array) -> Result<DataFrame> {
    df.take(indices)
}

// ============================================================================
// Group identification
// ============================================================================

/// Compute a group ID (0-based) for each row based on one or more columns.
///
/// Rows with the same combination of values in `group_cols` get the same ID.
/// The IDs are assigned in order of first appearance in the (already sorted) data.
pub fn compute_group_ids(df: &DataFrame, group_cols: &[&str]) -> Result<Vec<usize>> {
    let n_rows = df.height();
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    // Collect the group column arrays
    let arrays: Vec<&ArrayRef> = group_cols
        .iter()
        .map(|&name| df.column(name))
        .collect::<Result<Vec<_>>>()?;

    // Assign group IDs by detecting where the composite key changes.
    // Since data is expected to be sorted by group columns, we just compare
    // adjacent rows.
    let mut group_ids = Vec::with_capacity(n_rows);
    group_ids.push(0usize);
    let mut current_group = 0usize;

    for i in 1..n_rows {
        let changed = arrays
            .iter()
            .any(|arr| value_to_string(arr, i) != value_to_string(arr, i - 1));
        if changed {
            current_group += 1;
        }
        group_ids.push(current_group);
    }

    Ok(group_ids)
}

// ============================================================================
// Grouped cumulative operations
// ============================================================================

/// Compute cumulative sum within groups.
///
/// For each row, the result is the running total of `values` within its group
/// (identified by `group_ids`). Null values are treated as 0.
pub fn grouped_cumsum(values: &Float64Array, group_ids: &[usize]) -> Float64Array {
    let mut result = Vec::with_capacity(values.len());
    let mut running_sum = 0.0;
    let mut current_group = group_ids.first().copied().unwrap_or(0);

    for (val_opt, &gid) in values.iter().zip(group_ids.iter()) {
        if gid != current_group {
            // New group — reset running sum
            running_sum = 0.0;
            current_group = gid;
        }
        running_sum += val_opt.unwrap_or(0.0);
        result.push(running_sum);
    }

    Float64Array::from(result)
}

/// Compute shifted cumulative sum within groups (lag by 1, fill with 0).
///
/// For each row, the result is the cumulative sum of all PREVIOUS rows in the
/// same group. The first row of each group gets 0.
pub fn grouped_cumsum_lag(values: &Float64Array, group_ids: &[usize]) -> Float64Array {
    let mut result = Vec::with_capacity(values.len());
    let mut running_sum = 0.0;
    let mut current_group = group_ids.first().copied().unwrap_or(0);

    for (val_opt, &gid) in values.iter().zip(group_ids.iter()) {
        if gid != current_group {
            // New group — reset running sum
            running_sum = 0.0;
            current_group = gid;
        }
        // Lag: output the running sum BEFORE adding current value
        result.push(running_sum);
        running_sum += val_opt.unwrap_or(0.0);
    }

    Float64Array::from(result)
}

/// Compute group sums, broadcast back to each row.
///
/// Each row gets the total sum of its group.
pub fn grouped_sum_broadcast(values: &Float64Array, group_ids: &[usize]) -> Float64Array {
    if values.is_empty() {
        return Float64Array::from(Vec::<f64>::new());
    }

    let n_groups = group_ids.iter().copied().max().unwrap_or(0) + 1;
    let mut group_sums = vec![0.0; n_groups];

    for (val_opt, &gid) in values.iter().zip(group_ids.iter()) {
        group_sums[gid] += val_opt.unwrap_or(0.0);
    }

    let result: Vec<f64> = group_ids.iter().map(|&gid| group_sums[gid]).collect();
    Float64Array::from(result)
}

// ============================================================================
// Array arithmetic helpers
// ============================================================================

/// Compute element-wise: a / b (Float64 arrays).
pub fn divide_arrays(a: &Float64Array, b: &Float64Array) -> Result<Float64Array> {
    // Manual division to handle divide-by-zero gracefully (return 0 instead of NaN/Inf)
    let result: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let divisor = bv.unwrap_or(0.0);
            if divisor == 0.0 {
                0.0
            } else {
                av.unwrap_or(0.0) / divisor
            }
        })
        .collect();
    Ok(Float64Array::from(result))
}

/// Compute element-wise: a * scalar.
pub fn multiply_scalar(a: &Float64Array, scalar: f64) -> Float64Array {
    let result: Vec<f64> = a.iter().map(|v| v.unwrap_or(0.0) * scalar).collect();
    Float64Array::from(result)
}

/// Compute element-wise: a - b.
pub fn subtract_arrays(a: &Float64Array, b: &Float64Array) -> Float64Array {
    let result: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(av, bv)| av.unwrap_or(0.0) - bv.unwrap_or(0.0))
        .collect();
    Float64Array::from(result)
}

/// Compute element-wise: a / scalar.
pub fn divide_scalar(a: &Float64Array, scalar: f64) -> Float64Array {
    if scalar == 0.0 {
        return Float64Array::from(vec![0.0; a.len()]);
    }
    let result: Vec<f64> = a.iter().map(|v| v.unwrap_or(0.0) / scalar).collect();
    Float64Array::from(result)
}

// ============================================================================
// Aggregation
// ============================================================================

/// Get the minimum value from a Float64 array, ignoring nulls.
pub fn min_f64(array: &ArrayRef) -> Result<Option<f64>> {
    let f64_array = as_f64(array)?;
    Ok(compute::min(f64_array))
}

/// Get the minimum value from a column, casting to Float64 first if needed.
pub fn column_min_f64(df: &DataFrame, col_name: &str) -> Result<Option<f64>> {
    let col = df.column(col_name)?;
    if col.data_type() == &arrow::datatypes::DataType::Float64 {
        min_f64(col)
    } else {
        let casted = crate::array_util::cast_array(col, &arrow::datatypes::DataType::Float64)?;
        min_f64(&casted)
    }
}

// ============================================================================
// Convenience: fill nulls on an ArrayRef
// ============================================================================

/// Fill nulls in an ArrayRef (expected Float64) with a value, returning a new ArrayRef.
pub fn fill_null_f64_ref(array: &ArrayRef, fill: f64) -> Result<ArrayRef> {
    let f64_arr = as_f64(array)?;
    Ok(std::sync::Arc::new(fill_null_f64(f64_arr, fill)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grouped_cumsum_single_group() {
        let values = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let group_ids = vec![0, 0, 0];
        let result = grouped_cumsum(&values, &group_ids);
        assert_eq!(result.value(0), 10.0);
        assert_eq!(result.value(1), 30.0);
        assert_eq!(result.value(2), 60.0);
    }

    #[test]
    fn test_grouped_cumsum_two_groups() {
        // Groups: A(10, 20), B(15, 25)
        let values = Float64Array::from(vec![10.0, 20.0, 15.0, 25.0]);
        let group_ids = vec![0, 0, 1, 1];
        let result = grouped_cumsum(&values, &group_ids);
        assert_eq!(result.value(0), 10.0); // A: 10
        assert_eq!(result.value(1), 30.0); // A: 10+20
        assert_eq!(result.value(2), 15.0); // B: 15
        assert_eq!(result.value(3), 40.0); // B: 15+25
    }

    #[test]
    fn test_grouped_cumsum_lag() {
        let values = Float64Array::from(vec![10.0, 20.0, 15.0, 25.0]);
        let group_ids = vec![0, 0, 1, 1];
        let result = grouped_cumsum_lag(&values, &group_ids);
        assert_eq!(result.value(0), 0.0); // A: first → 0
        assert_eq!(result.value(1), 10.0); // A: lag of 10
        assert_eq!(result.value(2), 0.0); // B: first → 0
        assert_eq!(result.value(3), 15.0); // B: lag of 15
    }

    #[test]
    fn test_grouped_sum_broadcast() {
        let values = Float64Array::from(vec![10.0, 20.0, 15.0, 25.0]);
        let group_ids = vec![0, 0, 1, 1];
        let result = grouped_sum_broadcast(&values, &group_ids);
        assert_eq!(result.value(0), 30.0); // A total
        assert_eq!(result.value(1), 30.0); // A total
        assert_eq!(result.value(2), 40.0); // B total
        assert_eq!(result.value(3), 40.0); // B total
    }

    #[test]
    fn test_grouped_cumsum_with_nulls() {
        let values = Float64Array::from(vec![Some(10.0), None, Some(20.0)]);
        let group_ids = vec![0, 0, 0];
        let result = grouped_cumsum(&values, &group_ids);
        assert_eq!(result.value(0), 10.0);
        assert_eq!(result.value(1), 10.0); // null treated as 0
        assert_eq!(result.value(2), 30.0);
    }

    #[test]
    fn test_sort_dataframe() {
        let df = crate::df! {
            "x" => vec!["B", "A", "C", "A"],
            "y" => vec![2.0, 1.0, 3.0, 0.0],
        }
        .unwrap();

        let sorted = sort_dataframe(&df, &["x"]).unwrap();
        let x_col = sorted.column("x").unwrap();
        let x_arr = crate::array_util::as_str(x_col).unwrap();
        assert_eq!(x_arr.value(0), "A");
        assert_eq!(x_arr.value(1), "A");
        assert_eq!(x_arr.value(2), "B");
        assert_eq!(x_arr.value(3), "C");
    }

    #[test]
    fn test_compute_group_ids() {
        let df = crate::df! {
            "group" => vec!["A", "A", "B", "B", "C"],
        }
        .unwrap();

        let ids = compute_group_ids(&df, &["group"]).unwrap();
        assert_eq!(ids, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_compute_group_ids_multi_column() {
        let df = crate::df! {
            "g1" => vec!["A", "A", "A", "B"],
            "g2" => vec!["X", "X", "Y", "X"],
        }
        .unwrap();

        let ids = compute_group_ids(&df, &["g1", "g2"]).unwrap();
        assert_eq!(ids, vec![0, 0, 1, 2]);
    }

    #[test]
    fn test_divide_arrays_with_zero() {
        let a = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let b = Float64Array::from(vec![2.0, 0.0, 5.0]);
        let result = divide_arrays(&a, &b).unwrap();
        assert_eq!(result.value(0), 5.0);
        assert_eq!(result.value(1), 0.0); // divide by zero → 0
        assert_eq!(result.value(2), 6.0);
    }

    #[test]
    fn test_column_min_f64() {
        let df = crate::df! {
            "x" => vec![3.0, 1.0, 2.0],
        }
        .unwrap();
        let min = column_min_f64(&df, "x").unwrap();
        assert_eq!(min, Some(1.0));
    }

    #[test]
    fn test_multiply_scalar() {
        let a = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let result = multiply_scalar(&a, 10.0);
        assert_eq!(result.value(0), 10.0);
        assert_eq!(result.value(1), 20.0);
        assert_eq!(result.value(2), 30.0);
    }

    #[test]
    fn test_subtract_arrays() {
        let a = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let b = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let result = subtract_arrays(&a, &b);
        assert_eq!(result.value(0), 9.0);
        assert_eq!(result.value(1), 18.0);
        assert_eq!(result.value(2), 27.0);
    }
}
