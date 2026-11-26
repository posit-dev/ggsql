//! `SUMMARISE` execution: `setup_summaries` validates the clauses against the
//! data and prepares the frame for summary rows, `compute_summaries` runs
//! the aggregate operations on it.
//!
//! Most of `SUMMARY_OPERATIONS` are folds over a column's numeric values.
//! The tallies `count`, `n_distinct` and `n_missing` are type-free, and
//! date/timestamp columns accept the order statistics (returning a temporal
//! value) and the spread statistics (returning a duration).

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Float64Array, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow::datatypes::{DataType, TimeUnit};
use zygos::Value;

use crate::array_util::{as_f64, cast_array, column_to_strings, value_to_string};
use crate::naming;
use crate::params::ParameterValue;
use crate::table::{
    parse_template, BinOp, Format, StatExpr, Summary, SummaryOperation, SummaryScope, TableCell,
    TableClass, TableColumn, TableRow, TemplateSegment,
};
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
    /// One-row frame of the computed values; its schema names the
    /// summarised columns. Numeric for a bare operation — the column's
    /// temporal type for a date-valued statistic on a date/timestamp
    /// column — string for a template. A `FORMAT` renaming may stringify
    /// any of these.
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
///
/// A bare operation's summary value takes its column's `FORMAT` renaming
/// just like the body cells; a template self-formats via zygos.
pub(super) fn compute_summaries(
    df: &DataFrame,
    rows: &[TableRow],
    summaries: &[Summary],
    formats: &HashMap<String, Format>,
) -> Result<Vec<SummaryChunk>> {
    if summaries.is_empty() {
        return Ok(Vec::new());
    }
    // Chunks the clauses emit into: one contiguous range per group, then one
    // range over all rows for the GRAND clauses. Each accumulates its rows.
    struct Chunk {
        key: Option<Vec<usize>>,
        range: Range<usize>,
        rows: Vec<SummaryRow>,
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0;
    for chunk in rows.chunk_by(|a, b| a.group == b.group) {
        chunks.push(Chunk {
            key: Some(chunk[0].group.clone()),
            range: start..start + chunk.len(),
            rows: Vec::new(),
        });
        start += chunk.len();
    }
    chunks.push(Chunk {
        key: None,
        range: 0..rows.len(),
        rows: Vec::new(),
    });

    // Clause-outer: the label, placement and parsed template are clause-level
    // facts, computed once and shared by every group's row.
    for summary in summaries {
        let template = match &summary.operation {
            SummaryOperation::Template(template) => {
                Some(parse_template(template).map_err(GgsqlError::ValidationError)?)
            }
            SummaryOperation::Stat(_) => None,
        };
        let label = summary
            .settings
            .get("rowlabel")
            .and_then(ParameterValue::as_str)
            .map_or_else(|| summary.operation.as_label(), str::to_string);
        let top = ParameterValue::is_str(summary.settings.get("place"), "top");
        let targets = match summary.scope {
            SummaryScope::Group => 0..chunks.len() - 1,
            SummaryScope::Grand => chunks.len() - 1..chunks.len(),
        };
        for index in targets {
            let chunk = &mut chunks[index];
            let mut frame = summary_frame(df, summary, template.as_deref(), &chunk.range)?;
            // Templates self-format; a bare operation takes the column's
            // FORMAT renaming.
            if template.is_none() {
                format_summary_frame(&mut frame, &summary.columns, formats)?;
            }
            chunk.rows.push(SummaryRow {
                scope: summary.scope,
                top,
                label: label.clone(),
                frame,
            });
        }
    }
    Ok(chunks
        .into_iter()
        .filter_map(|chunk| (!chunk.rows.is_empty()).then_some((chunk.key, chunk.rows)))
        .collect())
}

/// Compute one clause's frame over the rows in `range`: one row, one column
/// per summarised column. A bare operation yields a nullable-f64 column; a
/// template (`Some` segments, which `compute_summaries` supplies for every
/// template clause) a string column with the template rendered against it.
fn summary_frame(
    df: &DataFrame,
    summary: &Summary,
    template: Option<&[TemplateSegment]>,
    range: &Range<usize>,
) -> Result<DataFrame> {
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
            let slice = array.slice(range.start, range.len());
            let cell: ArrayRef = match template {
                Some(segments) => Arc::new(StringArray::from(vec![render_template(
                    segments, &slice, drop_nulls,
                )?])),
                None => {
                    let SummaryOperation::Stat(name) = &summary.operation else {
                        unreachable!("compute_summaries parses every template clause")
                    };
                    value_array(&summary_stat(name, &slice, drop_nulls)?, slice.data_type())
                }
            };
            Ok((column.clone(), cell))
        })
        .collect::<Result<Vec<_>>>()?;
    DataFrame::new(columns)
}

/// Render a template against one column's slice: literals verbatim, each
/// placeholder's expression evaluated over the slice and rendered through its
/// pre-compiled zygos template. A NULL expression renders as the literal
/// "null", like the body cells. Stats are memoized in `cache` across all of
/// the template's placeholders, so a repeated stat computes once per cell.
fn render_template(
    segments: &[TemplateSegment],
    slice: &ArrayRef,
    drop_nulls: bool,
) -> Result<String> {
    let mut cache = HashMap::new();
    let mut text = String::new();
    for segment in segments {
        match segment {
            TemplateSegment::Literal(literal) => text.push_str(literal),
            TemplateSegment::Stat { expr, template } => {
                match eval_stat_expr(expr, slice, drop_nulls, &mut cache)? {
                    Some(value) => {
                        text.push_str(&template.format(&value, None).unwrap_or_default())
                    }
                    None => text.push_str("null"),
                }
            }
        }
    }
    Ok(text)
}

/// Evaluate a template placeholder's expression over one column's slice:
/// stat leaves via `summary_stat` keep their type, so a date-valued leaf
/// renders through zygos's temporal formatting; any arithmetic projects the
/// operands to plain numbers via [`leaf_f64`]. A NULL leaf makes the whole
/// expression NULL.
fn eval_stat_expr(
    expr: &StatExpr,
    slice: &ArrayRef,
    drop_nulls: bool,
    cache: &mut HashMap<String, Option<Value>>,
) -> Result<Option<Value>> {
    match expr {
        StatExpr::Stat(name) => {
            let key = name.to_ascii_lowercase();
            if let Some(value) = cache.get(&key) {
                return Ok(value.clone());
            }
            let value = summary_stat(name, slice, drop_nulls)?;
            cache.insert(key, value.clone());
            Ok(value)
        }
        StatExpr::Number(n) => Ok(Some(Value::Number(*n))),
        StatExpr::Neg(inner) => Ok(eval_stat_expr(inner, slice, drop_nulls, cache)?
            .and_then(|v| leaf_f64(&v))
            .map(|v| Value::Number(-v))),
        StatExpr::Binary(op, lhs, rhs) => {
            let (Some(lhs), Some(rhs)) = (
                eval_stat_expr(lhs, slice, drop_nulls, cache)?,
                eval_stat_expr(rhs, slice, drop_nulls, cache)?,
            ) else {
                return Ok(None);
            };
            let (Some(lhs), Some(rhs)) = (leaf_f64(&lhs), leaf_f64(&rhs)) else {
                return Ok(None);
            };
            Ok(Some(Value::Number(match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
            })))
        }
    }
}

/// Project a stat leaf to a plain number for arithmetic: days for dates,
/// seconds for datetimes, itself for numbers.
fn leaf_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => Some(*n),
        Value::Date(days) => Some(*days as f64),
        Value::DateTime(us) => Some(*us as f64 / 1e6),
        _ => None,
    }
}

/// Apply each summarised column's `FORMAT` renaming to a one-row summary
/// frame, turning the formatted columns into display text — the same
/// treatment `apply_formats` gives the body cells, per value.
fn format_summary_frame(
    frame: &mut DataFrame,
    columns: &[String],
    formats: &HashMap<String, Format>,
) -> Result<()> {
    for column in columns {
        let Some(format) = formats.get(column) else {
            continue;
        };
        if format.renaming.rename_is_empty() && format.renaming.template().is_none() {
            continue;
        }
        let array = frame.column(column)?.clone();
        let mut resolved = crate::format::resolve_column_values(&array, &format.renaming)
            .map_err(GgsqlError::ValidationError)?;
        let text = resolved
            .remove(0)
            .unwrap_or_else(|| value_to_string(&array, 0));
        *frame = frame.with_column(column, Arc::new(StringArray::from(vec![text])) as ArrayRef)?;
    }
    Ok(())
}

/// Compute one `SUMMARISE` operation over a column. With `drop_nulls`, nulls
/// are ignored; otherwise a single null makes the result `None` (NULL), as
/// does an empty input. Matching on `operation` is case-insensitive.
///
/// Numeric columns yield [`Value::Number`]. Date/timestamp columns accept
/// the order statistics, which yield the column's own kind of temporal
/// value, and the spread statistics, which yield a duration as a number of
/// days (dates) or seconds (timestamps); the other folds reject them.
pub fn summary_stat(operation: &str, array: &ArrayRef, drop_nulls: bool) -> Result<Option<Value>> {
    // The type-free tallies bypass the numeric fold: `count` counts rows or
    // non-null values, `n_missing` measures nulls so never goes NULL itself.
    let op = operation.to_ascii_lowercase();
    match op.as_str() {
        "count" => {
            let count = if drop_nulls {
                array.len() - array.logical_null_count()
            } else {
                array.len()
            };
            return Ok(Some(Value::Number(count as f64)));
        }
        "n_missing" => return Ok(Some(Value::Number(array.logical_null_count() as f64))),
        "n_distinct" => return Ok(n_distinct(array, drop_nulls)?.map(Value::Number)),
        _ => {}
    }
    let Some(kind) = ValueKind::of(array.data_type()) else {
        return Err(GgsqlError::ValidationError(format!(
            "SUMMARISE {operation} requires a numeric, date or timestamp column, not {}",
            array.data_type()
        )));
    };
    kind.summary_stat(&op, array, drop_nulls)
}

/// Fold one statistic over already-extracted values.
fn fold_stat(operation: &str, values: Vec<f64>) -> Result<f64> {
    let v = &values;
    let n = v.len() as f64;
    let result = match operation {
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
        "se" => variance(v).sqrt() / n.sqrt(),
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
    Ok(result)
}

/// The flavour of column a fold operates on, fixing the unit its statistics
/// are computed in and how the result is wrapped.
enum ValueKind {
    /// Any numeric type; statistics computed and reported as-is.
    Numeric,
    /// Date32/Date64; statistics computed and durations reported in days.
    Date,
    /// Any timestamp; statistics computed in µs, durations reported in
    /// seconds.
    DateTime,
}

impl ValueKind {
    fn of(dtype: &DataType) -> Option<Self> {
        match dtype {
            _ if dtype.is_numeric() => Some(Self::Numeric),
            DataType::Date32 | DataType::Date64 => Some(Self::Date),
            DataType::Timestamp(_, _) => Some(Self::DateTime),
            _ => None,
        }
    }

    fn summary_stat(
        &self,
        operation: &str,
        array: &ArrayRef,
        drop_nulls: bool,
    ) -> Result<Option<Value>> {
        // Temporal columns accept the order statistics, which return a
        // value of the column's own kind, and the spread statistics, which
        // return a duration.
        let date_valued = matches!(
            operation,
            "min"
                | "max"
                | "mid"
                | "mean"
                | "median"
                | "p05"
                | "p10"
                | "p25"
                | "p50"
                | "p75"
                | "p90"
                | "p95"
        );
        let duration = matches!(operation, "range" | "sdev" | "se" | "iqr");
        if !matches!(self, Self::Numeric) && !date_valued && !duration {
            return Err(GgsqlError::ValidationError(format!(
                "SUMMARISE {operation} does not support {}",
                array.data_type()
            )));
        }
        let Some(values) = self.values(array, drop_nulls)? else {
            return Ok(None);
        };
        let result = fold_stat(operation, values)?;
        Ok(Some(match self {
            Self::Numeric => Value::Number(result),
            kind if date_valued => kind.typed(result),
            kind => Value::Number(kind.duration(result)),
        }))
    }

    /// Extract the column's values in the kind's compute unit (unchanged
    /// for numbers, days for dates, µs for timestamps). Returns `None` when
    /// the result should be NULL: the input is empty, all-null, or holds
    /// nulls without `drop_nulls`.
    fn values(&self, array: &ArrayRef, drop_nulls: bool) -> Result<Option<Vec<f64>>> {
        let scale = match array.data_type() {
            DataType::Date32 => 1.0,
            DataType::Date64 => 1.0 / 86_400_000.0,
            DataType::Timestamp(unit, _) => match unit {
                TimeUnit::Second => 1e6,
                TimeUnit::Millisecond => 1e3,
                TimeUnit::Microsecond => 1.0,
                TimeUnit::Nanosecond => 1e-3,
            },
            _ => 1.0,
        };
        // Timestamps don't cast to Float64 directly; going through Int64
        // first is lossless for every accepted dtype.
        let f64s: ArrayRef = if array.data_type() == &DataType::Float64 {
            array.clone()
        } else if matches!(self, Self::Numeric) {
            cast_array(array, &DataType::Float64)?
        } else {
            cast_array(&cast_array(array, &DataType::Int64)?, &DataType::Float64)?
        };
        let array = as_f64(&f64s)?;
        if array.is_empty() || (array.logical_null_count() > 0 && !drop_nulls) {
            return Ok(None);
        }
        let values: Vec<f64> = array.iter().flatten().map(|v| v * scale).collect();
        Ok((!values.is_empty()).then_some(values))
    }

    /// Wrap a date-valued result in its temporal value, rounding to the
    /// column's precision (whole days / whole µs).
    fn typed(&self, value: f64) -> Value {
        match self {
            Self::Numeric => unreachable!("the numeric arm wraps directly"),
            Self::Date => Value::Date(value.round() as i32),
            Self::DateTime => Value::DateTime(value.round() as i64),
        }
    }

    /// Convert a duration from the compute unit to the reported unit (days
    /// for dates, seconds for timestamps).
    fn duration(&self, value: f64) -> f64 {
        match self {
            Self::Numeric => unreachable!("the numeric arm wraps directly"),
            Self::Date => value,
            Self::DateTime => value / 1e6,
        }
    }
}

/// Build the one-cell array holding a bare operation's result. A date-valued
/// result keeps the summarised column's temporal type (and timezone) so it
/// displays and formats like a body cell; everything else is a Float64.
fn value_array(value: &Option<Value>, dtype: &DataType) -> ArrayRef {
    let Some(value) = value else {
        let dtype = match ValueKind::of(dtype) {
            Some(ValueKind::Date | ValueKind::DateTime) => dtype.clone(),
            _ => DataType::Float64,
        };
        return arrow::array::new_null_array(&dtype, 1);
    };
    match (value, dtype) {
        (Value::Date(days), DataType::Date32) => Arc::new(Date32Array::from(vec![Some(*days)])),
        (Value::Date(days), DataType::Date64) => {
            Arc::new(Date64Array::from(vec![Some(*days as i64 * 86_400_000)]))
        }
        (Value::DateTime(us), DataType::Timestamp(unit, tz)) => {
            let us = *us;
            match unit {
                TimeUnit::Second => Arc::new(
                    TimestampSecondArray::from(vec![Some(us / 1_000_000)])
                        .with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Millisecond => Arc::new(
                    TimestampMillisecondArray::from(vec![Some(us / 1_000)])
                        .with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Microsecond => Arc::new(
                    TimestampMicrosecondArray::from(vec![Some(us)]).with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Nanosecond => Arc::new(
                    TimestampNanosecondArray::from(vec![Some(us * 1_000)])
                        .with_timezone_opt(tz.clone()),
                ),
            }
        }
        _ => {
            let Value::Number(n) = value else {
                unreachable!("summary_stat yields Date/DateTime only on temporal columns")
            };
            Arc::new(Float64Array::from(vec![Some(*n)]))
        }
    }
}

/// Count a column's distinct non-null values: NULL when nulls propagate.
/// Numeric arrays compare on f64 bits so display rounding can't merge two
/// values; other types compare on their string representation.
fn n_distinct(array: &ArrayRef, drop_nulls: bool) -> Result<Option<f64>> {
    if array.logical_null_count() > 0 && !drop_nulls {
        return Ok(None);
    }
    let len = if array.data_type().is_numeric() {
        let f64s = cast_array(array, &DataType::Float64)?;
        as_f64(&f64s)?
            .iter()
            .flatten()
            .map(f64::to_bits)
            .collect::<HashSet<_>>()
            .len()
    } else {
        (0..array.len())
            .filter(|&idx| !array.is_null(idx))
            .map(|idx| value_to_string(array, idx))
            .collect::<HashSet<_>>()
            .len()
    };
    Ok(Some(len as f64))
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

    /// Unwrap a `summary_stat` result to its plain number.
    fn num(result: Result<Option<Value>>) -> Option<f64> {
        result.unwrap().and_then(|v| v.as_number())
    }

    #[test]
    fn nulls_propagate_by_default() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        assert_eq!(num(summary_stat("mean", &array, false)), None);
        assert_eq!(num(summary_stat("median", &array, false)), None);
    }

    #[test]
    fn drop_nulls_ignores_them() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        assert_eq!(num(summary_stat("mean", &array, true)), Some(2.0));
        assert_eq!(num(summary_stat("median", &array, true)), Some(2.0));
    }

    #[test]
    fn mean_accepts_integer_columns() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        assert_eq!(num(summary_stat("MEAN", &array, false)), Some(2.0));
    }

    #[test]
    fn median_interpolates_even_counts() {
        let odd: ArrayRef = Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0]));
        assert_eq!(num(summary_stat("median", &odd, false)), Some(2.0));

        let even: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(num(summary_stat("median", &even, false)), Some(2.5));
    }

    #[test]
    fn all_null_input_is_none() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![None::<f64>, None]));
        assert_eq!(num(summary_stat("mean", &array, true)), None);
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
        assert_eq!(num(summary_stat("count", &strings, false)), Some(3.0));
        assert_eq!(num(summary_stat("count", &strings, true)), Some(2.0));
    }

    #[test]
    fn count_of_an_empty_slice_is_zero() {
        let array: ArrayRef = Arc::new(Float64Array::from(Vec::<f64>::new()));
        assert_eq!(num(summary_stat("count", &array, false)), Some(0.0));
    }

    #[test]
    fn n_missing_counts_nulls() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        assert_eq!(num(summary_stat("n_missing", &array, false)), Some(1.0));
        assert_eq!(num(summary_stat("n_missing", &array, true)), Some(1.0));
    }

    #[test]
    fn n_distinct_counts_distinct_values() {
        let numeric: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 2, 3, 1]));
        assert_eq!(num(summary_stat("n_distinct", &numeric, false)), Some(3.0));

        let strings: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "a"]));
        assert_eq!(num(summary_stat("n_distinct", &strings, false)), Some(2.0));
    }

    #[test]
    fn n_distinct_propagates_nulls() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(2.0)]));
        assert_eq!(num(summary_stat("n_distinct", &array, false)), None);
        assert_eq!(num(summary_stat("n_distinct", &array, true)), Some(2.0));
    }

    fn eval(template: &str, array: &ArrayRef, drop_nulls: bool) -> String {
        let segments = parse_template(template).unwrap();
        render_template(&segments, array, drop_nulls).unwrap()
    }

    #[test]
    fn template_arithmetic_combines_stats() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        assert_eq!(eval("{mean+10}", &array, false), "12");
        assert_eq!(eval("{(3-mean)*2}", &array, false), "2");
        // A repeated stat computes once and combines cleanly.
        assert_eq!(eval("{mean/mean}", &array, false), "1");
        assert_eq!(eval("{-mean}", &array, false), "−2");
    }

    #[test]
    fn template_arithmetic_propagates_nulls() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None]));
        assert_eq!(eval("{mean-sdev}", &array, false), "null");
        assert_eq!(eval("{mean-sdev:%.1f}", &array, true), "1.0");
    }

    fn stat(operation: &str, values: &[f64]) -> f64 {
        let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
        num(summary_stat(operation, &array, false)).unwrap()
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
        assert_eq!(stat("se", v), 1.25_f64.sqrt() / 2.0);
        assert_eq!(stat("var", v), 1.25);
        assert_eq!(stat("iqr", v), 1.5);
        assert_eq!(stat("median", v), 2.5);
        assert_eq!(stat("p50", v), 2.5);
        assert!((stat("p05", v) - 1.15).abs() < 1e-12);
        assert!((stat("p95", v) - 3.85).abs() < 1e-12);
    }

    /// 2024-01-15, 2024-01-16, 2024-01-18 as days since the epoch.
    fn dates() -> ArrayRef {
        Arc::new(Date32Array::from(vec![19737, 19738, 19740]))
    }

    #[test]
    fn date_valued_ops_on_dates_return_dates() {
        assert!(matches!(
            summary_stat("min", &dates(), false).unwrap(),
            Some(Value::Date(19737))
        ));
        assert!(matches!(
            summary_stat("max", &dates(), false).unwrap(),
            Some(Value::Date(19740))
        ));
        assert!(matches!(
            summary_stat("median", &dates(), false).unwrap(),
            Some(Value::Date(19738))
        ));
        // (19737 + 19740) / 2 = 19738.5, rounding to 19739.
        assert!(matches!(
            summary_stat("mid", &dates(), false).unwrap(),
            Some(Value::Date(19739))
        ));
    }

    #[test]
    fn a_mean_of_dates_rounds_to_a_whole_day() {
        // Mean of 19737 and 19738 falls on 19737.5.
        let array: ArrayRef = Arc::new(Date32Array::from(vec![19737, 19738]));
        assert!(matches!(
            summary_stat("mean", &array, false).unwrap(),
            Some(Value::Date(19738))
        ));
    }

    #[test]
    fn spread_ops_on_dates_return_days() {
        assert!(matches!(
            summary_stat("range", &dates(), false).unwrap(),
            Some(Value::Number(3.0))
        ));
        assert!(matches!(
            summary_stat("iqr", &dates(), false).unwrap(),
            Some(Value::Number(1.5))
        ));
    }

    #[test]
    fn spread_ops_on_timestamps_return_seconds() {
        let array: ArrayRef = Arc::new(TimestampSecondArray::from(vec![0, 10, 50]));
        assert!(matches!(
            summary_stat("range", &array, false).unwrap(),
            Some(Value::Number(50.0))
        ));
        // Population variance of 0, 10, 50 is 1400/3.
        assert!(matches!(
            summary_stat("sdev", &array, false).unwrap(),
            Some(Value::Number(s)) if (s * s - 1400.0 / 3.0).abs() < 1e-9
        ));
    }

    #[test]
    fn date_valued_ops_on_timestamps_return_datetimes() {
        let array: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![1_000, 3_000, 5_000]));
        assert!(matches!(
            summary_stat("min", &array, false).unwrap(),
            Some(Value::DateTime(1_000_000))
        ));
        assert!(matches!(
            summary_stat("median", &array, false).unwrap(),
            Some(Value::DateTime(3_000_000))
        ));
    }

    #[test]
    fn other_folds_reject_temporal_columns() {
        for op in ["sum", "prod", "geomean", "harmean", "rms", "var"] {
            assert!(
                summary_stat(op, &dates(), false).is_err(),
                "{op} should reject a date column"
            );
        }
    }

    #[test]
    fn tallies_still_accept_temporal_columns() {
        assert_eq!(num(summary_stat("count", &dates(), false)), Some(3.0));
        assert_eq!(num(summary_stat("n_distinct", &dates(), false)), Some(3.0));
    }

    #[test]
    fn temporal_nulls_propagate_by_default() {
        let array: ArrayRef = Arc::new(Date32Array::from(vec![Some(19737), None]));
        assert!(summary_stat("min", &array, false).unwrap().is_none());
        assert!(matches!(
            summary_stat("min", &array, true).unwrap(),
            Some(Value::Date(19737))
        ));
    }

    #[test]
    fn templates_format_date_leaves_as_dates() {
        assert_eq!(eval("{min}", &dates(), false), "2024-01-15");
        assert_eq!(eval("{min:time %d %b %Y}", &dates(), false), "15 Jan 2024");
    }

    #[test]
    fn template_arithmetic_on_dates_is_a_number_of_days() {
        assert_eq!(eval("{max-min}", &dates(), false), "3");
        assert_eq!(eval("{max-min:duration d}", &dates(), false), "3d");
        assert_eq!(eval("{(max-min)/7:%.4f}", &dates(), false), "0.4286");
    }

    #[test]
    fn a_date_summary_frame_keeps_the_date_type() {
        let df = DataFrame::new(vec![("d".to_string(), dates())]).unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["d"])],
            &HashMap::new(),
        )
        .unwrap();

        let array = computed[0].1[0].frame.column("d").unwrap();
        assert_eq!(array.data_type(), &DataType::Date32);
        // 19737.67 rounds to 19738: 2024-01-16.
        assert_eq!(column_to_strings(array), vec!["2024-01-16"]);
    }

    #[test]
    fn a_timestamp_summary_frame_keeps_the_timestamp_type() {
        let column: ArrayRef = Arc::new(TimestampSecondArray::from(vec![0, 3600, 7200]));
        let df = DataFrame::new(vec![("t".to_string(), column)]).unwrap();
        let rows = body_rows(&[&[], &[], &[]]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("max", SummaryScope::Grand, &["t"])],
            &HashMap::new(),
        )
        .unwrap();

        let array = computed[0].1[0].frame.column("t").unwrap();
        assert_eq!(
            array.data_type(),
            &DataType::Timestamp(TimeUnit::Second, None)
        );
        assert_eq!(column_to_strings(array), vec!["1970-01-01T02:00:00"]);
    }

    fn summary(operation: &str, scope: SummaryScope, columns: &[&str]) -> Summary {
        Summary {
            operation: SummaryOperation::Stat(operation.to_string()),
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

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["x"])],
            &HashMap::new(),
        )
        .unwrap();

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

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Group, &["x"])],
            &HashMap::new(),
        )
        .unwrap();

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

        let computed = compute_summaries(&df, &rows, &[below, above], &HashMap::new()).unwrap();

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

        let computed = compute_summaries(&df, &rows, &[labelled], &HashMap::new()).unwrap();

        assert_eq!(computed[0].1[0].label, "Average");
    }

    #[test]
    fn no_facet_means_one_group_with_an_empty_key() {
        let df = crate::df! {
            "x" => vec![2.0, 4.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Group, &["x"])],
            &HashMap::new(),
        )
        .unwrap();

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
            &HashMap::new(),
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
            renaming: Default::default(),
        }
    }

    fn renaming_format(renaming: zygos::LabelSpec) -> Format {
        Format {
            columns: Vec::new(),
            wildcard: false,
            target: None,
            settings: Parameters::new(),
            renaming,
        }
    }

    #[test]
    fn a_columns_format_formats_its_summary_value() {
        let df = crate::df! {
            "x" => vec![1.0, 2.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);
        let mut renaming = zygos::LabelSpec::new();
        renaming.set_template("${:num %.2f}");
        let formats = HashMap::from([("x".to_string(), renaming_format(renaming))]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["x"])],
            &formats,
        )
        .unwrap();

        assert_eq!(
            column_to_strings(computed[0].1[0].frame.column("x").unwrap()),
            vec!["$1.50"]
        );
    }

    #[test]
    fn a_rename_applies_to_a_summary_value() {
        let df = crate::df! {
            "x" => vec![0.0, 0.0],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);
        let mut renaming = zygos::LabelSpec::new();
        renaming.rename(zygos::Value::String("0".into()), "-");
        let formats = HashMap::from([("x".to_string(), renaming_format(renaming))]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["x"])],
            &formats,
        )
        .unwrap();

        assert_eq!(
            column_to_strings(computed[0].1[0].frame.column("x").unwrap()),
            vec!["-"]
        );
    }

    #[test]
    fn a_renaming_formats_a_count_of_a_non_numeric_column() {
        let df = crate::df! {
            "text" => vec!["a", "b"],
        }
        .unwrap();
        let rows = body_rows(&[&[], &[]]);
        let mut renaming = zygos::LabelSpec::new();
        renaming.set_template("[{}]");
        let formats = HashMap::from([("text".to_string(), renaming_format(renaming))]);

        let computed = compute_summaries(
            &df,
            &rows,
            &[summary("count", SummaryScope::Grand, &["text"])],
            &formats,
        )
        .unwrap();

        assert_eq!(
            column_to_strings(computed[0].1[0].frame.column("text").unwrap()),
            vec!["[2]"]
        );
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

        let result = compute_summaries(
            &df,
            &rows,
            &[summary("mean", SummaryScope::Grand, &["y"])],
            &HashMap::new(),
        );
        assert!(result.is_err());
    }
}
