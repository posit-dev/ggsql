//! Canonical facet panel assignment and ordering.
//!
//! This is the single source of truth for which rows belong to which panel and
//! in what order panels are enumerated. Writers must use these functions rather
//! than deriving their own panel sets, so that per-panel structures resolved in
//! core — [`Scale::panels`](crate::plot::Scale::panels) — line up with the
//! panels a writer draws.
//!
//! Panel enumeration order (the canonical contract):
//! - Wrap: the ordered levels of `facet1`, flowed row-major.
//! - Grid: row-major over (`facet1` row, `facet2` column) — `facet1` is the
//!   outer loop.
//!
//! Level ordering follows the facet aesthetic's `SCALE` (its `input_range`,
//! then `reverse`), falling back to a numeric-aware ascending sort of the
//! values present. A binned facet sorts by its (numeric) bin centre, with NULL
//! (censored) panels last.

use std::cmp::Ordering;
use std::collections::HashSet;

use arrow::array::{Array, ArrayRef, StringArray, UInt32Array};
use arrow::datatypes::DataType;

use crate::array_util::{as_f64, as_str, cast_array};
use crate::plot::{ArrayElement, ParameterValue, Scale, ScaleTypeKind};
use crate::{DataFrame, GgsqlError, Plot, Result};

use super::types::FacetLayout;

/// The value selecting one facet cell's rows: the facet column's text form,
/// paired with whether the cell was NULL.
///
/// Text alone would make a genuine empty-string category and a NULL the same
/// panel, so the null flag travels alongside the text everywhere a level is
/// identified or matched.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LevelKey {
    pub text: String,
    pub is_null: bool,
}

/// One distinct facet level: the key selecting its rows and the numeric form of
/// the same cell (the bin-centre join key for a binned facet).
#[derive(Debug, Clone)]
pub struct FacetLevel {
    pub key: LevelKey,
    pub value: f64,
}

/// One panel in the canonical enumeration: which facet values select its rows.
/// `facet2` is `None` for Wrap layouts.
#[derive(Debug, Clone)]
pub struct PanelKey {
    /// 0-based panel index in the canonical order.
    pub index: usize,
    pub facet1: LevelKey,
    pub facet2: Option<LevelKey>,
}

/// Read a column as strings, casting non-text columns to text. Nulls become
/// empty strings; pair with the array's own null bitmap when the distinction
/// matters (see [`level_keys`]).
fn column_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let array = df.column(name)?;
    let casted;
    let str_array: &StringArray = if matches!(array.data_type(), DataType::Utf8) {
        as_str(array)?
    } else {
        casted = cast_array(array, &DataType::Utf8)?;
        as_str(&casted)?
    };
    Ok((0..str_array.len())
        .map(|i| {
            if str_array.is_null(i) {
                String::new()
            } else {
                str_array.value(i).to_string()
            }
        })
        .collect())
}

/// Read a column as `f64`, casting from any numeric/temporal source type and
/// mapping nulls to `NaN`.
fn column_to_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let array = df.column(name)?;
    let casted;
    let f64_array = if matches!(array.data_type(), DataType::Float64) {
        as_f64(array)?
    } else {
        casted = cast_array(array, &DataType::Float64)?;
        as_f64(&casted)?
    };
    Ok(f64_array.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// The per-row facet key of one facet column: its text form paired with the
/// null bitmap, so a NULL cell selects a different panel than an empty-string
/// category — see [`LevelKey`].
pub fn level_keys(df: &DataFrame, column: &str) -> Result<Vec<LevelKey>> {
    let array = df.column(column)?;
    Ok(column_to_strings(df, column)?
        .into_iter()
        .enumerate()
        .map(|(i, text)| LevelKey {
            text,
            is_null: array.is_null(i),
        })
        .collect())
}

/// Distinct facet levels present in the data, ordered per the facet scale.
///
/// * `internal_aes` is the internal facet aesthetic name (`"facet1"` /
///   `"facet2"`); the column read is its aesthetic-renamed form.
pub fn ordered_levels(spec: &Plot, df: &DataFrame, internal_aes: &str) -> Result<Vec<FacetLevel>> {
    let col = crate::naming::aesthetic_column(internal_aes);
    let keys = level_keys(df, &col)?;
    // The numeric form of the same column, for the binned join. A text facet
    // column can't cast; `value` is only read for binned scales.
    let values = column_to_f64(df, &col).unwrap_or_else(|_| vec![f64::NAN; keys.len()]);

    let mut seen = HashSet::new();
    let mut distinct: Vec<FacetLevel> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if seen.insert(key.clone()) {
            distinct.push(FacetLevel {
                key: key.clone(),
                value: values[i],
            });
        }
    }

    let scale = spec.find_scale(internal_aes);
    Ok(order_levels(distinct, scale))
}

/// Order distinct facet levels: a binned facet sorts by its (numeric) bin
/// centre; everything else follows the scale's `input_range`, then any
/// present-but-unlisted values sorted numeric-aware ascending. Reversed when
/// the scale sets `reverse => true`.
fn order_levels(mut distinct: Vec<FacetLevel>, scale: Option<&Scale>) -> Vec<FacetLevel> {
    let reverse = matches!(
        scale.and_then(|s| s.properties.get("reverse")),
        Some(ParameterValue::Boolean(true))
    );

    let mut ordered = if is_binned(scale) {
        // Bin centres sort naturally; NULL (censored) panels go last.
        distinct.sort_by(|a, b| match (a.value.is_finite(), b.value.is_finite()) {
            (true, true) => a.value.total_cmp(&b.value),
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        });
        distinct
    } else {
        match scale.and_then(|s| s.input_range.as_ref()) {
            Some(range) => {
                let order: Vec<LevelKey> = range.iter().map(element_to_key).collect();
                let (mut ranked, mut extra): (Vec<FacetLevel>, Vec<FacetLevel>) =
                    (Vec::new(), Vec::new());
                for key in &order {
                    if let Some(pos) = distinct.iter().position(|l| &l.key == key) {
                        ranked.push(distinct.remove(pos));
                    }
                }
                extra.extend(distinct);
                sort_levels(&mut extra);
                ranked.extend(extra);
                ranked
            }
            None => {
                sort_levels(&mut distinct);
                distinct
            }
        }
    };
    if reverse {
        ordered.reverse();
    }
    ordered
}

/// Numeric-aware ascending sort by key: numeric when every key parses as `f64`,
/// otherwise lexical.
fn sort_levels(levels: &mut [FacetLevel]) {
    if levels.iter().all(|l| l.key.text.parse::<f64>().is_ok()) {
        levels.sort_by(|a, b| {
            let a = a.key.text.parse::<f64>().unwrap();
            let b = b.key.text.parse::<f64>().unwrap();
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        });
    } else {
        levels.sort_by(|a, b| a.key.text.cmp(&b.key.text));
    }
}

/// Whether a facet scale is binned (numeric/temporal facet columns default to
/// it).
pub fn is_binned(scale: Option<&Scale>) -> bool {
    scale
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind())
        == Some(ScaleTypeKind::Binned)
}

/// An `input_range` element as the key the facet column carries for it: the
/// text form the column casts to (whole numbers as integers, matching an
/// integer column's cast to text), plus whether the element is the null level.
pub fn element_to_key(element: &ArrayElement) -> LevelKey {
    let text = match element {
        ArrayElement::String(s) => s.clone(),
        ArrayElement::Number(n) if n.fract() == 0.0 && n.is_finite() => format!("{}", *n as i64),
        ArrayElement::Number(n) => n.to_string(),
        ArrayElement::Boolean(b) => b.to_string(),
        ArrayElement::Null => String::new(),
        other => format!("{other:?}"),
    };
    LevelKey {
        text,
        is_null: matches!(element, ArrayElement::Null),
    }
}

/// The panels of a faceted plot, in the canonical enumeration order.
///
/// `layer0` is the first layer's data, which defines the panel set (a layer
/// contributing a level no other layer has still gets its panel via layer 0's
/// facet columns when it participates in the plot's facet; layers missing the
/// facet column entirely are repeated in every panel — see [`rows_in_panel`]).
///
/// Returns an empty vector when the plot has no `FACET` or the data has no
/// facet levels; writers collapse that case to a single panel bound to the
/// shared scales.
pub fn panel_keys(spec: &Plot, layer0: &DataFrame) -> Result<Vec<PanelKey>> {
    let Some(facet) = &spec.facet else {
        return Ok(Vec::new());
    };
    match &facet.layout {
        FacetLayout::Wrap { .. } => {
            let levels = ordered_levels(spec, layer0, "facet1")?;
            Ok(levels
                .into_iter()
                .enumerate()
                .map(|(index, level)| PanelKey {
                    index,
                    facet1: level.key,
                    facet2: None,
                })
                .collect())
        }
        FacetLayout::Grid { .. } => {
            let rows = ordered_levels(spec, layer0, "facet1")?;
            let cols = ordered_levels(spec, layer0, "facet2")?;
            let mut panels = Vec::with_capacity(rows.len() * cols.len());
            let mut index = 0;
            for row in &rows {
                for col in &cols {
                    panels.push(PanelKey {
                        index,
                        facet1: row.key.clone(),
                        facet2: Some(col.key.clone()),
                    });
                    index += 1;
                }
            }
            Ok(panels)
        }
    }
}

/// The row indices of `df` belonging to `panel`, or `None` when `df` has no
/// facet1 column — an annotation/global layer, which belongs to every panel
/// whole.
pub fn rows_in_panel(df: &DataFrame, panel: &PanelKey) -> Result<Option<Vec<u32>>> {
    let f1 = crate::naming::aesthetic_column("facet1");
    if df.column(&f1).is_err() {
        return Ok(None);
    }
    let c1 = level_keys(df, &f1)?;
    let c2 = match &panel.facet2 {
        Some(_) => Some(level_keys(df, &crate::naming::aesthetic_column("facet2"))?),
        None => None,
    };

    let mut idx: Vec<u32> = Vec::new();
    for i in 0..df.height() {
        if c1[i] != panel.facet1 {
            continue;
        }
        if let (Some(c2), Some(want2)) = (&c2, &panel.facet2) {
            if &c2[i] != want2 {
                continue;
            }
        }
        idx.push(i as u32);
    }
    Ok(Some(idx))
}

/// Take `indices` rows of an array, preserving its type.
pub fn take_rows(array: &ArrayRef, indices: &[u32]) -> Result<ArrayRef> {
    arrow::compute::take(array.as_ref(), &UInt32Array::from(indices.to_vec()), None)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to take panel rows: {}", e)))
}
