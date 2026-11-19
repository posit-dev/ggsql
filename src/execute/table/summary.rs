//! `SUMMARISE` execution: `setup_summaries` validates the clauses against the
//! data and prepares the frame for summary rows, `compute_summaries` runs
//! the aggregate operations on it.
//!
//! Only `mean` and `median` exist so far; the rest of `SUMMARY_OPERATIONS`
//! lands once the execution pipeline is wired end-to-end.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
use arrow::datatypes::DataType;

use crate::array_util::{as_f64, cast_array};
use crate::naming;
use crate::params::{ParameterValue, Parameters};
use crate::table::{Format, Summary, SummaryScope, TableRow};
use crate::{ColumnSection, DataFrame, GgsqlError, Result};

/// Validate `summaries` against the data and prepare the frame for summary
/// rows. Every summarised column must exist. Summary row labels need a stub
/// column to live in, so a table with summaries but no `STUB` column gets a
/// synthetic, empty one ([`naming::SUMMARY_LABEL_COLUMN`]).
pub(super) fn setup_summaries(
    df: &mut DataFrame,
    formats: &HashMap<String, Format>,
    summaries: &[Summary],
) -> Result<()> {
    if summaries.is_empty() {
        return Ok(());
    }
    for summary in summaries {
        for column in &summary.columns {
            if df.column(column).is_err() {
                return Err(GgsqlError::ValidationError(format!(
                    "SUMMARISE {}: no column '{column}' in the query result",
                    summary.operation
                )));
            }
        }
    }
    if !formats
        .values()
        .any(|f| f.target == Some(ColumnSection::Stub))
    {
        *df = df.with_column(
            naming::SUMMARY_LABEL_COLUMN,
            Arc::new(StringArray::from(vec![""; df.height()])),
        )?;
    }
    Ok(())
}

/// Compute every `SUMMARISE` clause against the (unformatted) data, one
/// DataFrame per clause, parallel to `summaries`. Every frame leads with a
/// `SUMMARY_GROUP_COLUMN` of ';'-joined group ranks, then one nullable-f64
/// column per summarised column, raw (before `FORMAT`). A `GROUP` frame
/// holds one row per group — in display order, since `setup_rows` leaves
/// rows group-contiguous; a `GRAND` frame a single row keyed by the empty
/// key (a grand total has no group). The clause itself carries the row's
/// label and placement, so nothing here duplicates them.
// Unused until the table pipeline calls it.
#[allow(dead_code)]
pub(super) fn compute_summaries(
    df: &DataFrame,
    rows: &[TableRow],
    summaries: &[Summary],
) -> Result<Vec<DataFrame>> {
    // Rows are group-contiguous after setup_rows, so each group's membership
    // is a contiguous range; a GRAND clause's single "group" is all of them.
    let mut start = 0;
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut key_strings: Vec<String> = Vec::new();
    for chunk in rows.chunk_by(|a, b| a.group == b.group) {
        ranges.push(start..start + chunk.len());
        start += chunk.len();
        key_strings.push(naming::summary_group_key(&chunk[0].group));
    }

    // A grand total has no group; the empty key matches the facet-less one.
    let grand_key = [String::new()];
    summaries
        .iter()
        .map(|summary| {
            let (groups, keys) = match summary.scope {
                SummaryScope::Grand => (None, grand_key.as_slice()),
                SummaryScope::Group => (Some(ranges.as_slice()), key_strings.as_slice()),
            };
            summary_frame(df, summary, groups, keys)
        })
        .collect()
}

/// Compute one clause's frame: the key column followed by one nullable-f64
/// column per summarised column. `groups` is `None` for a GRAND clause — a
/// single row over all rows.
fn summary_frame(
    df: &DataFrame,
    summary: &Summary,
    groups: Option<&[Range<usize>]>,
    keys: &[String],
) -> Result<DataFrame> {
    let drop_nulls = setting_string(&summary.settings, "missing").as_deref() == Some("drop");

    let mut columns: Vec<(String, ArrayRef)> = Vec::new();
    columns.push((
        naming::SUMMARY_GROUP_COLUMN.to_string(),
        Arc::new(StringArray::from(keys.to_vec())),
    ));
    for column in &summary.columns {
        let values = summary_column(df, column, summary, groups, drop_nulls)?;
        columns.push((column.clone(), Arc::new(Float64Array::from(values))));
    }
    DataFrame::new(columns)
}

/// Compute a clause's operation on one column, one value per group — or a
/// single value over the whole column when `groups` is `None` (GRAND).
fn summary_column(
    df: &DataFrame,
    column: &str,
    summary: &Summary,
    groups: Option<&[Range<usize>]>,
    drop_nulls: bool,
) -> Result<Vec<Option<f64>>> {
    let array = df.column(column).map_err(|_| {
        GgsqlError::ValidationError(format!(
            "SUMMARISE {}: no column '{column}' in the query result",
            summary.operation
        ))
    })?;
    match groups {
        Some(groups) => groups
            .iter()
            .map(|range| {
                summary_stat(
                    &summary.operation,
                    &array.slice(range.start, range.len()),
                    drop_nulls,
                )
            })
            .collect(),
        None => summary_stat(&summary.operation, array, drop_nulls).map(|value| vec![value]),
    }
}

fn setting_string(settings: &Parameters, name: &str) -> Option<String> {
    match settings.get(name) {
        Some(ParameterValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

/// Compute one `SUMMARISE` operation over a column. With `drop_nulls`, nulls
/// are ignored; otherwise a single null makes the result `None` (NULL), as
/// does an empty input. Matching on `operation` is case-insensitive.
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

    fn summary(operation: &str, scope: SummaryScope, columns: &[&str]) -> Summary {
        Summary {
            operation: operation.to_string(),
            scope,
            columns: columns.iter().map(|c| c.to_string()).collect(),
            settings: Parameters::new(),
        }
    }

    fn body_rows(keys: &[&[usize]]) -> Vec<TableRow> {
        keys.iter()
            .map(|key| TableRow {
                group: key.to_vec(),
                ..TableRow::body()
            })
            .collect()
    }

    fn column_values(frame: &DataFrame, column: &str) -> Vec<Option<f64>> {
        as_f64(frame.column(column).unwrap())
            .unwrap()
            .iter()
            .collect()
    }

    fn group_keys(frame: &DataFrame) -> Vec<String> {
        crate::array_util::as_str(frame.column(naming::SUMMARY_GROUP_COLUMN).unwrap())
            .unwrap()
            .iter()
            .map(|key| key.unwrap().to_string())
            .collect()
    }

    #[test]
    fn grand_summary_is_a_single_row_keyed_by_the_empty_key() {
        let df = crate::df! {
            "x" => vec![1.0, 2.0, 3.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let frames =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Grand, &["x"])]).unwrap();

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].height(), 1);
        assert_eq!(group_keys(&frames[0]), vec![""]);
        assert_eq!(column_values(&frames[0], "x"), vec![Some(2.0)]);
    }

    #[test]
    fn group_summary_emits_one_row_per_group_in_display_order() {
        let df = crate::df! {
            "x" => vec![1.0, 3.0, 10.0, 20.0],
        }
        .unwrap();
        // Two nested-facet groups, contiguous after setup_rows' reorder.
        let rows = body_rows(&[&[2, 0], &[2, 0], &[2, 1], &[2, 1]]);

        let frames =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Group, &["x"])]).unwrap();

        assert_eq!(frames[0].height(), 2);
        assert_eq!(group_keys(&frames[0]), vec!["2;0", "2;1"]);
        assert_eq!(column_values(&frames[0], "x"), vec![Some(2.0), Some(15.0)]);
    }

    #[test]
    fn no_facet_means_one_group_with_an_empty_key() {
        let df = crate::df! {
            "x" => vec![2.0, 4.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);

        let frames =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Group, &["x"])]).unwrap();

        assert_eq!(frames[0].height(), 1);
        assert_eq!(group_keys(&frames[0]), vec![""]);
        assert_eq!(column_values(&frames[0], "x"), vec![Some(3.0)]);
    }

    #[test]
    fn the_missing_setting_drives_null_handling() {
        let df = crate::df! {
            "x" => vec![Some(1.0), None, Some(3.0)],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let mut drop = summary("mean", SummaryScope::Grand, &["x"]);
        drop.settings.insert(
            "missing".to_string(),
            ParameterValue::String("drop".to_string()),
        );

        let frames = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["x"]), drop],
        )
        .unwrap();

        assert_eq!(column_values(&frames[0], "x"), vec![None]);
        assert_eq!(column_values(&frames[1], "x"), vec![Some(2.0)]);
    }

    fn stub_format() -> Format {
        Format {
            columns: vec!["x".to_string()],
            wildcard: false,
            target: Some(ColumnSection::Stub),
            settings: Parameters::new(),
            value_mapping: None,
            value_template: None,
        }
    }

    #[test]
    fn setup_adds_an_empty_label_column_when_no_stub_exists() {
        let mut df = crate::df! {
            "x" => vec![1.0, 2.0],
        }
        .unwrap();

        setup_summaries(
            &mut df,
            &HashMap::new(),
            &[summary("mean", SummaryScope::Group, &["x"])],
        )
        .unwrap();

        let labels =
            crate::array_util::as_str(df.column(naming::SUMMARY_LABEL_COLUMN).unwrap()).unwrap();
        assert_eq!(labels.iter().collect::<Vec<_>>(), vec![Some(""), Some("")]);
    }

    #[test]
    fn setup_leaves_the_frame_alone_when_a_stub_exists() {
        let mut df = crate::df! {
            "x" => vec![1.0, 2.0],
        }
        .unwrap();
        let formats = HashMap::from([("x".to_string(), stub_format())]);

        setup_summaries(
            &mut df,
            &formats,
            &[summary("mean", SummaryScope::Group, &["x"])],
        )
        .unwrap();

        assert!(df.column(naming::SUMMARY_LABEL_COLUMN).is_err());
    }

    #[test]
    fn setup_without_summaries_is_a_no_op() {
        let mut df = crate::df! {
            "x" => vec![1.0],
        }
        .unwrap();

        setup_summaries(&mut df, &HashMap::new(), &[]).unwrap();

        assert_eq!(df.width(), 1);
    }

    #[test]
    fn setup_rejects_a_summarised_column_the_data_lacks() {
        let mut df = crate::df! {
            "x" => vec![1.0],
        }
        .unwrap();

        let result = setup_summaries(
            &mut df,
            &HashMap::new(),
            &[summary("mean", SummaryScope::Group, &["y"])],
        );

        assert_eq!(
            result.unwrap_err().to_string(),
            "Validation error: SUMMARISE mean: no column 'y' in the query result"
        );
    }

    #[test]
    fn summarising_a_missing_column_errors() {
        let df = crate::df! {
            "x" => vec![1.0],
        }
        .unwrap();
        let rows = body_rows(&[&[]]);

        let result = compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Grand, &["y"])]);
        assert!(result.is_err());
    }
}
