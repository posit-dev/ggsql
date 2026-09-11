//! FACET → multi-panel composition.
//!
//! ggsql resolves faceting fully at execution time: the layout (Wrap/Grid), the
//! `free` bool array, Wrap's `ncol`, and per-row facet assignment materialized in
//! the ordinary aesthetic columns `__ggsql_aes_facet1__` (and `facet2__` for
//! Grid). This module turns that into a hephaestus [`Composition`] of named
//! panels plus a [`Panel`] list the writer loops over — one hephaestus `Plot` per
//! panel, sharing the composition's scale registry.
//!
//! Panel ordering mirrors the Vega-Lite writer's `resolve_facet_ordering`: the
//! facet aesthetic's `SCALE` (its `input_range`, then `reverse`) drives the
//! order, falling back to a numeric-aware ascending sort of the present values.

use std::cmp::Ordering;
use std::collections::HashSet;

use arrow::array::{Array, UInt32Array};
use hephaestus::composition::{grid, spacer, Composition, Element, Patch};

use super::channels::{column_to_f64, column_to_strings};
use crate::naming;
use crate::plot::{ArrayElement, FacetLayout, ParameterValue, Scale, ScaleTypeKind};
use crate::{DataFrame, Plot, Result};

/// Patch id for the single (unfaceted) panel.
pub const PANEL_ID: &str = "ggsql_panel";

/// The value selecting one facet cell's rows: the facet column's text form,
/// paired with whether the cell was NULL.
///
/// [`column_to_strings`] renders a NULL as `""`, so text alone would make a
/// genuine empty-string category and a NULL the same panel. The Vega-Lite
/// writer keeps them apart (a null level labels as `"null"` and gets its own
/// facet), so this writer carries the null flag alongside the text everywhere a
/// level is identified or matched.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LevelKey {
    text: String,
    is_null: bool,
}

/// One facet cell: which facet values it holds, its grid position (for edge-only
/// axes), and the strip-label text to show.
pub struct Panel {
    /// hephaestus patch id, unique per panel.
    pub id: String,
    /// 0-based panel index (order of enumeration), for per-panel scale names.
    pub index: usize,
    /// Facet1 (Wrap panel / Grid row) value selecting this panel's rows.
    pub facet1: Option<LevelKey>,
    /// Facet2 (Grid column) value; `None` for Wrap.
    pub facet2: Option<LevelKey>,
    /// Top strip label (Wrap header, or Grid column header on the top row).
    pub strip_top: Option<String>,
    /// Right strip label (Grid row header on the right column).
    pub strip_right: Option<String>,
    /// Whether this panel is in the left column (draws the y-axis when fixed).
    pub first_col: bool,
    /// Whether this panel is the bottom-most present panel in its column (draws
    /// the x-axis when fixed).
    pub last_row: bool,
}

impl Panel {
    /// The unfaceted single panel: draws both axes, no strips.
    fn single() -> Panel {
        Panel {
            id: PANEL_ID.to_string(),
            index: 0,
            facet1: None,
            facet2: None,
            strip_top: None,
            strip_right: None,
            first_col: true,
            last_row: true,
        }
    }
}

/// The unfaceted layout: one full-size panel, no strips.
///
/// Also what a `FACET` over an empty result set collapses to: there are no
/// levels to lay out, and a grid of zero cells is not a composition hephaestus
/// will build. The figure then reads like the unfaceted empty plot — one empty
/// panel with its axes — rather than a bare canvas.
fn single_panel() -> (Composition, Vec<Panel>) {
    (
        grid(1, 1, vec![Element::from(Patch::new(PANEL_ID))]),
        vec![Panel::single()],
    )
}

/// Build the panel grid for a plot. Returns a single-cell composition + one
/// [`Panel`] when there is no `FACET`, otherwise the faceted grid.
pub fn build_panels(
    spec: &Plot,
    data: &std::collections::HashMap<String, DataFrame>,
) -> Result<(Composition, Vec<Panel>)> {
    let Some(facet) = &spec.facet else {
        return Ok(single_panel());
    };
    let layer0 = super::compose::layer_dataframe(&spec.layers[0], 0, data)?;
    match &facet.layout {
        FacetLayout::Wrap { .. } => build_wrap(spec, facet, layer0),
        FacetLayout::Grid { .. } => build_grid(spec, layer0),
    }
}

/// Wrap: N panels flowed row-major into `ncol` columns.
fn build_wrap(
    spec: &Plot,
    facet: &crate::plot::Facet,
    layer0: &DataFrame,
) -> Result<(Composition, Vec<Panel>)> {
    let levels = ordered_levels(spec, layer0, "facet1")?;
    if levels.is_empty() {
        return Ok(single_panel());
    }
    let n = levels.len();
    let ncol = wrap_ncol(facet, n);
    let nrow = n.div_ceil(ncol);

    let mut panels = Vec::with_capacity(n);
    for (idx, level) in levels.iter().enumerate() {
        let col = idx % ncol;
        // Bottom-most present panel in this column: no panel sits `ncol` cells
        // below it. Governs where the x-axis shows when the last row is partial.
        let last_row = idx + ncol >= n;
        panels.push(Panel {
            id: format!("facet_{idx}"),
            index: idx,
            facet1: Some(level.key.clone()),
            facet2: None,
            strip_top: Some(level.label.clone()),
            strip_right: None,
            first_col: col == 0,
            last_row,
        });
    }

    // Cells row-major, padding the trailing slots of a partial last row.
    let mut cells: Vec<Element> = Vec::with_capacity(nrow * ncol);
    for slot in 0..(nrow * ncol) {
        if slot < panels.len() {
            cells.push(Element::from(Patch::new(panels[slot].id.clone())));
        } else {
            cells.push(Element::from(spacer()));
        }
    }
    Ok((grid(nrow, ncol, cells), panels))
}

/// Grid: rows = facet1 levels, columns = facet2 levels. Column strips on the top
/// row, row strips on the right column.
fn build_grid(spec: &Plot, layer0: &DataFrame) -> Result<(Composition, Vec<Panel>)> {
    let rows = ordered_levels(spec, layer0, "facet1")?;
    let cols = ordered_levels(spec, layer0, "facet2")?;
    if rows.is_empty() || cols.is_empty() {
        return Ok(single_panel());
    }
    let nrow = rows.len();
    let ncol = cols.len();

    let mut panels = Vec::with_capacity(nrow * ncol);
    let mut cells: Vec<Element> = Vec::with_capacity(nrow * ncol);
    let mut index = 0;
    for (r, rowv) in rows.iter().enumerate() {
        for (c, colv) in cols.iter().enumerate() {
            let id = format!("facet_{r}_{c}");
            panels.push(Panel {
                id: id.clone(),
                index,
                facet1: Some(rowv.key.clone()),
                facet2: Some(colv.key.clone()),
                strip_top: (r == 0).then(|| colv.label.clone()),
                strip_right: (c == ncol - 1).then(|| rowv.label.clone()),
                first_col: c == 0,
                last_row: r == nrow - 1,
            });
            cells.push(Element::from(Patch::new(id)));
            index += 1;
        }
    }
    Ok((grid(nrow, ncol, cells), panels))
}

/// The resolved Wrap column count (ggsql computes it during resolution); falls
/// back to a single row if somehow absent.
fn wrap_ncol(facet: &crate::plot::Facet, n: usize) -> usize {
    match facet.properties.get("ncol") {
        Some(ParameterValue::Number(c)) if *c >= 1.0 => (*c as usize).min(n).max(1),
        _ => n.max(1),
    }
}

/// One distinct facet level: the key selecting its rows, the numeric form of the
/// same cell (the bin-join key for a binned facet), and the strip text.
struct Level {
    key: LevelKey,
    value: f64,
    label: String,
}

/// The per-row facet key of one facet column: its text form paired with the null
/// bitmap, so a NULL cell selects a different panel than an empty-string
/// category — see [`LevelKey`].
fn level_keys(df: &DataFrame, column: &str) -> Result<Vec<LevelKey>> {
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

/// Distinct facet levels present in the data, ordered per the facet scale and
/// labelled for the strip.
fn ordered_levels(spec: &Plot, df: &DataFrame, internal_aes: &str) -> Result<Vec<Level>> {
    let col = naming::aesthetic_column(internal_aes);
    let keys = level_keys(df, &col)?;
    // The numeric form of the same column, for the binned join. A text facet
    // column can't cast; `value` is only read for binned scales.
    let values = column_to_f64(df, &col).unwrap_or_else(|_| vec![f64::NAN; keys.len()]);

    let mut seen = HashSet::new();
    let mut distinct: Vec<Level> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if seen.insert(key.clone()) {
            distinct.push(Level {
                key: key.clone(),
                value: values[i],
                label: String::new(),
            });
        }
    }

    let scale = spec.find_scale(internal_aes);
    let mut ordered = order_levels(distinct, scale);
    for level in &mut ordered {
        level.label = facet_label(scale, level);
    }
    Ok(ordered)
}

/// Order distinct facet levels, mirroring the Vega-Lite writer's
/// `resolve_facet_ordering`: a binned facet sorts by its (numeric) bin centre;
/// everything else follows the scale's `input_range`, then any present-but-unlisted
/// values sorted numeric-aware ascending. Reversed when the scale sets
/// `reverse => true`.
fn order_levels(mut distinct: Vec<Level>, scale: Option<&Scale>) -> Vec<Level> {
    let reverse = super::scales::is_reversed(scale);

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
                let (mut ranked, mut extra): (Vec<Level>, Vec<Level>) = (Vec::new(), Vec::new());
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
fn sort_levels(levels: &mut [Level]) {
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

/// Whether a facet scale is binned (numeric/temporal facet columns default to it).
fn is_binned(scale: Option<&Scale>) -> bool {
    scale
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind())
        == Some(ScaleTypeKind::Binned)
}

/// Strip text for one facet level, mirroring the Vega-Lite writer's
/// `build_indexed_facet_label_expr` (discrete + `RENAMING`) and
/// `build_binned_facet_label_expr` (bin ranges). Computed here from typed values
/// rather than as a Vega expression over serialized data, so a temporal binned
/// facet — which Vega-Lite silently fails to match — labels correctly.
fn facet_label(scale: Option<&Scale>, level: &Level) -> String {
    // NULL keys as the literal string "null", matching ggsql's RENAMING key for
    // a null level (`RENAMING null => 'The rest'`).
    if level.key.is_null {
        return match scale.and_then(|s| s.label_mapping.as_ref()) {
            Some(mapping) => match mapping.get("null") {
                Some(Some(label)) => label.clone(),
                Some(None) => String::new(),
                None => "null".to_string(),
            },
            None => "null".to_string(),
        };
    }
    if is_binned(scale) {
        // The column carries the bin centre; label it with the bin's range.
        let bins = scale.map(super::scales::binned_bins).unwrap_or_default();
        if let Some(i) = super::scales::bin_at_centre(&bins, level.value) {
            return bins[i].label.clone();
        }
        return level.key.text.clone();
    }
    discrete_label(scale, level)
}

/// A discrete/ordinal level's label: the `RENAMING` override for its domain
/// value, an empty strip when suppressed, else the raw value.
fn discrete_label(scale: Option<&Scale>, level: &Level) -> String {
    let Some(scale) = scale else {
        return level.key.text.clone();
    };
    let Some(mapping) = scale.label_mapping.as_ref() else {
        return level.key.text.clone();
    };
    // `label_mapping` is keyed on the domain element's `to_key_string()`, which
    // can differ from the column's arrow-cast text (e.g. "5" vs "5.0"), so find
    // the matching domain element first.
    let key = scale
        .input_range
        .as_ref()
        .and_then(|range| {
            range
                .iter()
                .find(|e| element_matches(e, level))
                .map(|e| e.to_key_string())
        })
        .unwrap_or_else(|| level.key.text.clone());
    match mapping.get(&key) {
        Some(Some(label)) => label.clone(),
        Some(None) => String::new(),
        None => level.key.text.clone(),
    }
}

/// Whether a domain element denotes the same value as this level: by key first,
/// then numerically (a `DOUBLE` column's `"5.0"` still matches `Number(5.0)`).
fn element_matches(element: &ArrayElement, level: &Level) -> bool {
    if element_to_key(element) == level.key {
        return true;
    }
    if level.key.is_null {
        return false;
    }
    match element.to_f64() {
        Some(n) => level.value.is_finite() && n == level.value,
        None => false,
    }
}

/// An `input_range` element as the key the facet column carries for it: the text
/// form the column casts to (whole numbers as integers, matching an integer
/// column's cast to text), plus whether the element is the null level.
fn element_to_key(element: &ArrayElement) -> LevelKey {
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

/// The scale names a panel binds its position channels to, and whether each
/// dimension is free. For fixed dimensions the name is the shared `pos1`/`pos2`;
/// for free dimensions it is a per-panel name (`pos1__p{index}`), so each panel
/// resolves through its own domain.
pub struct PanelScales {
    pub pos1: String,
    pub pos2: String,
    pub free_x: bool,
    pub free_y: bool,
}

impl PanelScales {
    pub fn new(spec: &Plot, panel: &Panel) -> Self {
        let free_x = spec.facet.as_ref().is_some_and(|f| f.is_free("pos1"));
        let free_y = spec.facet.as_ref().is_some_and(|f| f.is_free("pos2"));
        PanelScales {
            pos1: if free_x {
                format!("pos1__p{}", panel.index)
            } else {
                "pos1".to_string()
            },
            pos2: if free_y {
                format!("pos2__p{}", panel.index)
            } else {
                "pos2".to_string()
            },
            free_x,
            free_y,
        }
    }

    /// Point one dimension back at the shared `pos1`/`pos2` scale, for a panel
    /// whose free per-panel scale could not be built (an empty facet cell has no
    /// extent to free the dimension over). The dimension stays flagged free, so
    /// its axis is still drawn on this panel like on every other.
    pub fn use_shared(&mut self, aesthetic: &str) {
        match aesthetic {
            "pos1" => self.pos1 = "pos1".to_string(),
            "pos2" => self.pos2 = "pos2".to_string(),
            _ => {}
        }
    }
}

/// The rows of `df` belonging to `panel`, sliced via `DataFrame::take`. A layer
/// with no facet column (annotation/global layers) is used whole for every panel.
pub fn panel_dataframe(df: &DataFrame, panel: &Panel) -> Result<DataFrame> {
    let Some(want1) = &panel.facet1 else {
        return Ok(df.clone());
    };
    let f1 = naming::aesthetic_column("facet1");
    if df.column(&f1).is_err() {
        return Ok(df.clone());
    }
    let c1 = level_keys(df, &f1)?;
    let c2 = match &panel.facet2 {
        Some(_) => Some(level_keys(df, &naming::aesthetic_column("facet2"))?),
        None => None,
    };

    let mut idx: Vec<u32> = Vec::new();
    for i in 0..df.height() {
        if &c1[i] != want1 {
            continue;
        }
        if let (Some(c2), Some(want2)) = (&c2, &panel.facet2) {
            if &c2[i] != want2 {
                continue;
            }
        }
        idx.push(i as u32);
    }
    df.take(&UInt32Array::from(idx))
}
