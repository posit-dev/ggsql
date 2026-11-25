//! `SUMMARISE` execution: `setup_summaries` validates the clauses against the
//! data and prepares the frame for summary rows, `compute_summaries` runs
//! the aggregate operations on it.
//!
//! All of `SUMMARY_OPERATIONS` are folds over a column's values except
//! `count`, which is type-free and counts rows or non-null values.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
use arrow::datatypes::DataType;

use crate::array_util::{as_f64, cast_array, column_to_strings};
use crate::naming;
use crate::params::ParameterValue;
use crate::table::{Format, Summary, SummaryScope, TableCell, TableClass, TableColumn, TableRow};
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

/// One `SUMMARISE` row, ready for weaving into the table's grid: placement,
/// label and values all resolved. The weavers never see the `Summary`
/// clause itself.
#[derive(Debug)]
pub(super) struct SummaryRow {
    /// Whether the row summarises its group (`GROUP`) or the whole table
    /// (`GRAND`).
    pub scope: SummaryScope,
    /// Whether the row sits above (`true`) or below its group's rows.
    pub top: bool,
    /// The row's stub label: `rowlabel`, else the operation name.
    pub label: String,
    /// One-row frame of the computed values, raw (before `FORMAT`); its
    /// schema names the summarised columns.
    pub frame: DataFrame,
}

impl SummaryRow {
    /// This row's cells at grid row `row`: the label spanning the stub
    /// columns, then the computed values in their body columns. No cells in
    /// group-stub columns: a `GROUP` row's are covered by the merged stub
    /// cell; a `GRAND` row owns them by extending its label's colspan
    /// leftward. `first`/`last` mark the row's position in its block (both
    /// when the block is one row) and `above` whether the block sits above
    /// the rows it summarizes; together they pick the positional style
    /// classes.
    pub(super) fn cells(
        &self,
        row: usize,
        columns: &[TableColumn],
        first: bool,
        last: bool,
        above: bool,
    ) -> Vec<TableCell> {
        // Every cell in the row carries the same classes: the scope's base
        // class plus the block's positional markers. A group block gets the
        // solid border on both outer edges; a grand block gets it on its
        // far edge and the double border on the edge touching the body.
        let (base, first_class, last_class) = match (self.scope, above) {
            (SummaryScope::Group, _) => (
                TableClass::GroupSummary,
                TableClass::SummaryFirst,
                TableClass::SummaryLast,
            ),
            (SummaryScope::Grand, true) => (
                TableClass::GrandSummary,
                TableClass::SummaryFirst,
                TableClass::GrandSummaryLast,
            ),
            (SummaryScope::Grand, false) => (
                TableClass::GrandSummary,
                TableClass::GrandSummaryFirst,
                TableClass::SummaryLast,
            ),
        };
        let mut classes = vec![base];
        if first {
            classes.push(first_class);
        }
        if last {
            classes.push(last_class);
        }

        let mut cells = vec![self.label_cell(row, columns, &classes)];
        cells.extend(
            columns
                .iter()
                .enumerate()
                .filter(|(_, column)| column.target == ColumnSection::Body)
                .map(|(index, column)| {
                    let content = self
                        .frame
                        .column(&column.name)
                        .map(|array| column_to_strings(array).remove(0))
                        .unwrap_or_default();
                    let mut cell = TableCell::new_summary_value(row, index, content)
                        .with_properties(column.properties.clone());
                    cell.classes.extend_from_slice(&classes);
                    cell
                }),
        );
        cells
    }

    /// The row's label cell, spanning every stub column — leftward over the
    /// group-stub columns too for a `GRAND` row. `setup_summaries` adds a
    /// stub column when summaries exist without one; `move_stub_columns`
    /// puts every stub-section column in one contiguous run, so the span
    /// covers only stub columns.
    fn label_cell(&self, row: usize, columns: &[TableColumn], classes: &[TableClass]) -> TableCell {
        let label_column = columns
            .iter()
            .position(|c| c.target == ColumnSection::Stub)
            .expect("setup_summaries guarantees a stub column");
        let span_group_stubs = self.scope == SummaryScope::Grand;
        let left = if span_group_stubs && columns.first().is_some_and(|c| c.is_group_stub()) {
            0
        } else {
            label_column
        };
        let right = columns
            .iter()
            .rposition(|c| c.target == ColumnSection::Stub)
            .unwrap_or(label_column);
        debug_assert!(
            columns[left..=right].iter().all(|c| c.is_stub()),
            "the label's colspan covers only stub columns"
        );

        let mut cell = TableCell::new_summary_label(row, label_column, self.label.clone());
        cell.left = left;
        cell.right = right;
        cell.classes.extend_from_slice(classes);
        cell.with_properties(columns[label_column].properties.clone())
    }
}

/// One chunk of summary rows: the chunk's group key (`None` for the `GRAND`
/// chunk) and the clause rows computed over it, in clause order.
pub(super) type SummaryChunk = (Option<Vec<usize>>, Vec<SummaryRow>);

/// Compute every `SUMMARISE` clause against the (unformatted) data: one
/// chunk per group, in display order (`setup_rows` leaves rows
/// group-contiguous), then one `None`-keyed `GRAND` chunk over all rows.
/// Chunks with no matching clauses are omitted. Placement within a chunk is
/// the weavers' concern; they read each row's `top`.
pub(super) fn compute_summaries(
    df: &DataFrame,
    rows: &[TableRow],
    summaries: &[Summary],
) -> Result<Vec<SummaryChunk>> {
    if summaries.is_empty() {
        return Ok(Vec::new());
    }
    // Chunks the clauses emit into: one contiguous range per group, then one
    // range over all rows for the GRAND clauses.
    let mut chunks: Vec<(Option<Vec<usize>>, Range<usize>)> = Vec::new();
    let mut start = 0;
    for chunk in rows.chunk_by(|a, b| a.group == b.group) {
        chunks.push((Some(chunk[0].group.clone()), start..start + chunk.len()));
        start += chunk.len();
    }
    chunks.push((None, 0..rows.len()));

    let mut out = Vec::new();
    for (key, range) in chunks {
        let scope = match key {
            Some(_) => SummaryScope::Group,
            None => SummaryScope::Grand,
        };
        let rows = summaries
            .iter()
            .filter(|s| s.scope == scope)
            .map(|summary| summary.new_row(df, &range))
            .collect::<Result<Vec<_>>>()?;
        if !rows.is_empty() {
            out.push((key, rows));
        }
    }
    Ok(out)
}

impl Summary {
    /// Build this clause's row over `range`: placement, label and computed
    /// values.
    pub(super) fn new_row(&self, df: &DataFrame, range: &Range<usize>) -> Result<SummaryRow> {
        let label = self
            .settings
            .get("rowlabel")
            .and_then(ParameterValue::as_str)
            .map_or_else(|| self.operation.clone(), str::to_string);
        Ok(SummaryRow {
            scope: self.scope,
            top: ParameterValue::is_str(self.settings.get("place"), "top"),
            label,
            frame: summary_frame(df, self, range)?,
        })
    }
}

/// Compute one clause's frame over the rows in `range`: one row, one
/// nullable-f64 column per summarised column.
fn summary_frame(df: &DataFrame, summary: &Summary, range: &Range<usize>) -> Result<DataFrame> {
    let drop_nulls = ParameterValue::is_str(summary.settings.get("missing"), "drop");

    let columns = summary
        .columns
        .iter()
        .map(|column| {
            let array = df.column(column).map_err(|_| {
                GgsqlError::ValidationError(format!(
                    "SUMMARISE {}: no column '{column}' in the query result",
                    summary.operation
                ))
            })?;
            let value = summary_stat(
                &summary.operation,
                &array.slice(range.start, range.len()),
                drop_nulls,
            )?;
            Ok((
                column.clone(),
                Arc::new(Float64Array::from(vec![value])) as ArrayRef,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    DataFrame::new(columns)
}

/// Compute one `SUMMARISE` operation over a column. With `drop_nulls`, nulls
/// are ignored; otherwise a single null makes the result `None` (NULL), as
/// does an empty input. Matching on `operation` is case-insensitive.
pub fn summary_stat(operation: &str, array: &ArrayRef, drop_nulls: bool) -> Result<Option<f64>> {
    // `count` is type-free: rows when nulls propagate, non-null values when
    // dropped. Unlike the numeric operations, nulls never make it NULL.
    if operation.eq_ignore_ascii_case("count") {
        let count = if drop_nulls {
            array.len() - array.logical_null_count()
        } else {
            array.len()
        };
        return Ok(Some(count as f64));
    }
    let values = numeric_values(array, operation, drop_nulls)?;
    let Some(values) = values else {
        return Ok(None);
    };
    let v = &values;
    let n = v.len() as f64;
    let result = match operation.to_ascii_lowercase().as_str() {
        "sum" => v.iter().sum(),
        "prod" => v.iter().product(),
        "min" => min(v),
        "max" => max(v),
        "range" => max(v) - min(v),
        "mid" => (min(v) + max(v)) / 2.0,
        "mean" => v.iter().sum::<f64>() / n,
        "geomean" => (v.iter().map(|x| x.ln()).sum::<f64>() / n).exp(),
        "harmean" => n / v.iter().map(|x| 1.0 / x).sum::<f64>(),
        "rms" => (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt(),
        "sdev" => variance(v).sqrt(),
        "var" => variance(v),
        "iqr" => quantile(values.clone(), 0.75) - quantile(values.clone(), 0.25),
        "median" | "p50" => quantile(values.clone(), 0.5),
        "p05" => quantile(values.clone(), 0.05),
        "p10" => quantile(values.clone(), 0.10),
        "p25" => quantile(values.clone(), 0.25),
        "p75" => quantile(values.clone(), 0.75),
        "p90" => quantile(values.clone(), 0.90),
        "p95" => quantile(values.clone(), 0.95),
        _ => {
            return Err(GgsqlError::InternalError(format!(
                "SUMMARISE operation '{operation}' is not implemented"
            )))
        }
    };
    Ok(Some(result))
}

/// Population variance, matching the plot stats' `VAR_POP`.
fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn min(values: &[f64]) -> f64 {
    values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

fn max(values: &[f64]) -> f64 {
    values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
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
    use crate::params::Parameters;
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
    fn unknown_operation_is_rejected() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        assert!(summary_stat("first", &array, false).is_err());
    }

    #[test]
    fn count_is_type_free() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        assert_eq!(summary_stat("count", &strings, false).unwrap(), Some(3.0));
        assert_eq!(summary_stat("count", &strings, true).unwrap(), Some(2.0));
    }

    #[test]
    fn count_of_an_empty_slice_is_zero() {
        let array: ArrayRef = Arc::new(Float64Array::from(Vec::<f64>::new()));
        assert_eq!(summary_stat("count", &array, false).unwrap(), Some(0.0));
    }

    fn stat(operation: &str, values: &[f64]) -> f64 {
        let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
        summary_stat(operation, &array, false).unwrap().unwrap()
    }

    #[test]
    fn numeric_operations() {
        let v = &[1.0, 2.0, 3.0, 4.0];
        assert_eq!(stat("sum", v), 10.0);
        assert_eq!(stat("prod", v), 24.0);
        assert_eq!(stat("min", v), 1.0);
        assert_eq!(stat("max", v), 4.0);
        assert_eq!(stat("range", v), 3.0);
        assert_eq!(stat("mid", v), 2.5);
        assert_eq!(stat("mean", v), 2.5);
        assert_eq!(stat("geomean", v), (24.0_f64).powf(0.25));
        assert_eq!(stat("harmean", v), 4.0 / (1.0 + 0.5 + 1.0 / 3.0 + 0.25));
        assert_eq!(stat("rms", v), ((1.0 + 4.0 + 9.0 + 16.0) / 4.0_f64).sqrt());
        assert_eq!(stat("sdev", v), 1.25_f64.sqrt());
        assert_eq!(stat("var", v), 1.25);
        assert_eq!(stat("iqr", v), 1.5);
        assert_eq!(stat("median", v), 2.5);
        assert_eq!(stat("p50", v), 2.5);
        assert!((stat("p05", v) - 1.15).abs() < 1e-12);
        assert!((stat("p95", v) - 3.85).abs() < 1e-12);
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

    #[test]
    fn grand_summary_is_a_single_ungrouped_row() {
        let df = crate::df! {
            "x" => vec![1.0, 2.0, 3.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let computed =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Grand, &["x"])]).unwrap();

        assert_eq!(computed.len(), 1);
        let (key, rows) = &computed[0];
        assert_eq!(*key, None);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].top);
        assert_eq!(rows[0].label, "mean");
        assert_eq!(column_values(&rows[0].frame, "x"), vec![Some(2.0)]);
    }

    #[test]
    fn group_summary_emits_one_row_per_group_in_display_order() {
        let df = crate::df! {
            "x" => vec![1.0, 3.0, 10.0, 20.0],
        }
        .unwrap();
        // Two nested-facet groups, contiguous after setup_rows' reorder.
        let rows = body_rows(&[&[2, 0], &[2, 0], &[2, 1], &[2, 1]]);

        let computed =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Group, &["x"])]).unwrap();

        assert_eq!(computed.len(), 2);
        assert_eq!(computed[0].0, Some(vec![2, 0]));
        assert_eq!(computed[1].0, Some(vec![2, 1]));
        assert_eq!(column_values(&computed[0].1[0].frame, "x"), vec![Some(2.0)]);
        assert_eq!(
            column_values(&computed[1].1[0].frame, "x"),
            vec![Some(15.0)]
        );
    }

    #[test]
    fn rows_come_chunk_major_in_clause_order() {
        let df = crate::df! {
            "x" => vec![1.0, 10.0],
        }
        .unwrap();
        let rows = body_rows(&[&[0], &[1]]);
        let above = Summary {
            settings: crate::params! { "place" => "top" },
            ..summary("mean", SummaryScope::Group, &["x"])
        };
        let below = summary("median", SummaryScope::Group, &["x"]);

        let computed = compute_summaries(&df, &rows, &[below, above]).unwrap();

        let order: Vec<(Vec<usize>, &str, bool)> = computed
            .iter()
            .flat_map(|(key, rows)| {
                let key = key.clone().expect("GROUP chunks are keyed");
                rows.iter()
                    .map(move |row| (key.clone(), row.label.as_str(), row.top))
            })
            .collect();
        // Chunk-major, clause order within a chunk; placement is the
        // weavers' concern.
        assert_eq!(
            order,
            vec![
                (vec![0], "median", false),
                (vec![0], "mean", true),
                (vec![1], "median", false),
                (vec![1], "mean", true),
            ]
        );
    }

    #[test]
    fn rowlabel_overrides_the_operation_as_label() {
        let df = crate::df! {
            "x" => vec![1.0],
        }
        .unwrap();
        let rows = body_rows(&[&[]]);
        let labelled = Summary {
            settings: crate::params! { "rowlabel" => "Average" },
            ..summary("mean", SummaryScope::Grand, &["x"])
        };

        let computed = compute_summaries(&df, &rows, &[labelled]).unwrap();

        assert_eq!(computed[0].1[0].label, "Average");
    }

    #[test]
    fn no_facet_means_one_group_with_an_empty_key() {
        let df = crate::df! {
            "x" => vec![2.0, 4.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);

        let computed =
            compute_summaries(&df, &rows, &[summary("mean", SummaryScope::Group, &["x"])]).unwrap();

        assert_eq!(computed.len(), 1);
        assert_eq!(computed[0].0, Some(vec![]));
        assert_eq!(column_values(&computed[0].1[0].frame, "x"), vec![Some(3.0)]);
    }

    #[test]
    fn the_missing_setting_drives_null_handling() {
        let df = crate::df! {
            "x" => vec![Some(1.0), None, Some(3.0)],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let drop = Summary {
            settings: crate::params! { "missing" => "drop" },
            ..summary("mean", SummaryScope::Grand, &["x"])
        };

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["x"]), drop],
        )
        .unwrap();

        let (key, rows) = &computed[0];
        assert_eq!(*key, None);
        assert_eq!(column_values(&rows[0].frame, "x"), vec![None]);
        assert_eq!(column_values(&rows[1].frame, "x"), vec![Some(2.0)]);
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
