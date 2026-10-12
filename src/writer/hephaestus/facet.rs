//! FACET → multi-panel composition.
//!
//! ggsql resolves faceting fully at execution time: the layout (Wrap/Grid), the
//! `free` bool array, Wrap's `ncol`, and per-row facet assignment materialized in
//! the ordinary aesthetic columns `__ggsql_aes_facet1__` (and `facet2__` for
//! Grid). This module turns that into a hephaestus [`Composition`] of named
//! panels plus a [`Panel`] list the writer loops over — one hephaestus `Plot` per
//! panel, sharing the composition's scale registry.
//!
//! Panel ordering and row assignment are core's, from
//! [`crate::plot::facet::panels`] — the same order per-panel scale resolutions
//! ([`Scale::panels`](crate::plot::Scale::panels)) are stored in, so
//! `Panel::index` indexes directly into them.

use hephaestus::composition::{grid, spacer, Composition, Element, Patch};

use crate::plot::facet::{self as facet_panels, FacetLevel};
use crate::plot::{ArrayElement, FacetLayout, ParameterValue, Scale};
use crate::{DataFrame, Plot, Result};

/// Patch id for the single (unfaceted) panel.
pub const PANEL_ID: &str = "ggsql_panel";

/// One facet cell: which facet values it holds, its grid position (for edge-only
/// axes), and the strip-label text to show.
pub struct Panel {
    /// hephaestus patch id, unique per panel.
    pub id: String,
    /// 0-based panel index (the canonical order, see
    /// [`crate::plot::facet::panels`]), indexing [`Scale::panels`] and used for
    /// per-panel scale names.
    pub index: usize,
    /// Facet1 (Wrap panel / Grid row) value selecting this panel's rows.
    pub facet1: Option<facet_panels::LevelKey>,
    /// Facet2 (Grid column) value; `None` for Wrap.
    pub facet2: Option<facet_panels::LevelKey>,
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
    let levels = ordered_labelled_levels(spec, layer0, "facet1")?;
    if levels.is_empty() {
        return Ok(single_panel());
    }
    let n = levels.len();
    let ncol = wrap_ncol(facet, n);
    let nrow = n.div_ceil(ncol);

    let mut panels = Vec::with_capacity(n);
    for (idx, (level, label)) in levels.iter().enumerate() {
        let col = idx % ncol;
        // Bottom-most present panel in this column: no panel sits `ncol` cells
        // below it. Governs where the x-axis shows when the last row is partial.
        let last_row = idx + ncol >= n;
        panels.push(Panel {
            id: format!("facet_{idx}"),
            index: idx,
            facet1: Some(level.key.clone()),
            facet2: None,
            strip_top: Some(label.clone()),
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
    let rows = ordered_labelled_levels(spec, layer0, "facet1")?;
    let cols = ordered_labelled_levels(spec, layer0, "facet2")?;
    if rows.is_empty() || cols.is_empty() {
        return Ok(single_panel());
    }
    let nrow = rows.len();
    let ncol = cols.len();

    let mut panels = Vec::with_capacity(nrow * ncol);
    let mut cells: Vec<Element> = Vec::with_capacity(nrow * ncol);
    let mut index = 0;
    for (r, (rowv, row_label)) in rows.iter().enumerate() {
        for (c, (colv, col_label)) in cols.iter().enumerate() {
            let id = format!("facet_{r}_{c}");
            panels.push(Panel {
                id: id.clone(),
                index,
                facet1: Some(rowv.key.clone()),
                facet2: Some(colv.key.clone()),
                strip_top: (r == 0).then(|| col_label.clone()),
                strip_right: (c == ncol - 1).then(|| row_label.clone()),
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

/// Distinct facet levels in the canonical core order, each paired with its
/// strip-label text.
fn ordered_labelled_levels(
    spec: &Plot,
    df: &DataFrame,
    internal_aes: &str,
) -> Result<Vec<(FacetLevel, String)>> {
    let scale = spec.find_scale(internal_aes);
    Ok(facet_panels::ordered_levels(spec, df, internal_aes)?
        .into_iter()
        .map(|level| {
            let label = facet_label(scale, &level);
            (level, label)
        })
        .collect())
}

/// Strip text for one facet level, mirroring the Vega-Lite writer's
/// `build_indexed_facet_label_expr` (discrete + `RENAMING`) and
/// `build_binned_facet_label_expr` (bin ranges). Computed here from typed values
/// rather than as a Vega expression over serialized data, so a temporal binned
/// facet — which Vega-Lite silently fails to match — labels correctly.
fn facet_label(scale: Option<&Scale>, level: &FacetLevel) -> String {
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
    if facet_panels::is_binned(scale) {
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
fn discrete_label(scale: Option<&Scale>, level: &FacetLevel) -> String {
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
fn element_matches(element: &ArrayElement, level: &FacetLevel) -> bool {
    if facet_panels::element_to_key(element) == level.key {
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
    let key = facet_panels::PanelKey {
        index: panel.index,
        facet1: want1.clone(),
        facet2: panel.facet2.clone(),
    };
    match facet_panels::rows_in_panel(df, &key)? {
        Some(idx) => df.take(&arrow::array::UInt32Array::from(idx)),
        None => Ok(df.clone()),
    }
}
