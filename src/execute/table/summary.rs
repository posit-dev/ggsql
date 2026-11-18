//! Statistics kernel for `SUMMARISE` clauses: aggregate operations computed
//! directly on the materialised table data.
//!
//! Only `mean` and `median` exist so far; the rest of `SUMMARY_OPERATIONS`
//! lands once the execution pipeline is wired end-to-end.

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;

use crate::array_util::{as_f64, cast_array};
use crate::{GgsqlError, Result};

/// Compute one `SUMMARISE` operation over a column. With `drop_nulls`, nulls
/// are ignored; otherwise a single null makes the result `None` (NULL), as
/// does an empty input. Matching on `operation` is case-insensitive.
// Unused until summary execution calls it.
#[allow(dead_code)]
pub fn summary_stat(operation: &str, array: &ArrayRef, drop_nulls: bool) -> Result<Option<f64>> {
    let values = numeric_values(array, operation, drop_nulls)?;
    let Some(values) = values else {
        return Ok(None);
    };
    match operation.to_ascii_lowercase().as_str() {
        "mean" => Ok(Some(values.iter().sum::<f64>() / values.len() as f64)),
        "median" => Ok(Some(quantile(values, 0.5))),
        _ => Err(GgsqlError::InternalError(format!(
            "SUMMARISE operation '{operation}' is not implemented"
        ))),
    }
}

/// Extract a column's values as f64. Returns `None` when the result should be
/// NULL: the input is empty, all-null, or holds nulls without `drop_nulls`.
fn numeric_values(array: &ArrayRef, operation: &str, drop_nulls: bool) -> Result<Option<Vec<f64>>> {
    if !array.data_type().is_numeric() {
        return Err(GgsqlError::ValidationError(format!(
            "SUMMARISE {operation} requires a numeric column, not {}",
            array.data_type()
        )));
    }
    let f64s: ArrayRef = if array.data_type() == &DataType::Float64 {
        array.clone()
    } else {
        cast_array(array, &DataType::Float64)?
    };
    let array = as_f64(&f64s)?;
    if array.is_empty() || (array.logical_null_count() > 0 && !drop_nulls) {
        return Ok(None);
    }
    let values: Vec<f64> = array.iter().flatten().collect();
    Ok((!values.is_empty()).then_some(values))
}

/// Linear-interpolated quantile.
/// Uses the `quantile_cont` convention, R's default `quantile(type = 7)`.
fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let pos = q * (values.len() - 1) as f64;
    let lower = values[pos.floor() as usize];
    let upper = values[pos.ceil() as usize];
    lower + (upper - lower) * pos.fract()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn nulls_propagate_by_default() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        assert_eq!(summary_stat("mean", &array, false).unwrap(), None);
        assert_eq!(summary_stat("median", &array, false).unwrap(), None);
    }

    #[test]
    fn drop_nulls_ignores_them() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        assert_eq!(summary_stat("mean", &array, true).unwrap(), Some(2.0));
        assert_eq!(summary_stat("median", &array, true).unwrap(), Some(2.0));
    }

    #[test]
    fn mean_accepts_integer_columns() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        assert_eq!(summary_stat("MEAN", &array, false).unwrap(), Some(2.0));
    }

    #[test]
    fn median_interpolates_even_counts() {
        let odd: ArrayRef = Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0]));
        assert_eq!(summary_stat("median", &odd, false).unwrap(), Some(2.0));

        let even: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(summary_stat("median", &even, false).unwrap(), Some(2.5));
    }

    #[test]
    fn all_null_input_is_none() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![None::<f64>, None]));
        assert_eq!(summary_stat("mean", &array, true).unwrap(), None);
    }

    #[test]
    fn non_numeric_column_is_rejected() {
        let array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        assert!(summary_stat("mean", &array, true).is_err());
    }

    #[test]
    fn unimplemented_operation_is_rejected() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        assert!(summary_stat("geomean", &array, false).is_err());
    }
}
