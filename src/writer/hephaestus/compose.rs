//! Turning a resolved ggsql `Plot` into a live hephaestus composition.
//!
//! Everything up to the point where an output format matters. Each writer calls
//! [`build_composition`] and then rasterises, renders or serialises the result,
//! so this lives here rather than in any one writer.

use std::collections::HashMap;

use hephaestus::plot::{scale, AspectMode, Plot as HPlot, PlotComposition};
use hephaestus::scales::chrome::AxisSide;
use hephaestus::shape::ShapeRegistry;

use super::projection::apply_projection;
use super::scales::build_scale;
use super::wiring::Ctx;
use super::{channels, facet, geom, projection, scales, wiring};
use crate::naming;
use crate::plot::layer::geom::GeomType;
use crate::plot::layer::is_transposed;
use crate::plot::ParameterValue;
use crate::{DataFrame, GgsqlError, Layer, Plot, Result};

/// Fraction of a map's bounding-box span added as breathing room around it, so
/// marks on the boundary are not drawn against the panel edge. Matches the
/// Vega-Lite writer's projection fit (`span * 1.1`).
const MAP_PADDING: f64 = 0.1;

/// Reject a plot no renderer-backed writer can draw.
///
/// Phrased without naming a format: what cannot be drawn here is a limit of the
/// composition layer, not of the encoder the caller happened to pick.
///
/// # Errors
///
/// Returns `GgsqlError::WriterError` for a plot with no layers, or one whose
/// geom the composition layer cannot build.
pub fn validate_plot(spec: &Plot) -> Result<()> {
    if spec.layers.is_empty() {
        return Err(GgsqlError::WriterError(
            "a plot needs at least one layer".into(),
        ));
    }
    for layer in &spec.layers {
        let geom_type = layer.geom.geom_type();
        if !geom::is_supported(geom_type) {
            return Err(GgsqlError::WriterError(format!(
                "the plot renderer does not support the '{geom_type}' geom yet"
            )));
        }
    }
    Ok(())
}

/// Validate `spec` and build its composition — the two steps every writer runs
/// before it does anything of its own.
///
/// One function so the pair cannot drift: skipping the validation lets a
/// refused plot reach the renderer and fail in the renderer's words.
///
/// # Errors
///
/// As [`validate_plot`] and [`build_composition`].
pub fn prepare(spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<PlotComposition> {
    validate_plot(spec)?;
    build_composition(spec, data)
}

/// Build the composition for `spec`, ready to render at any size.
///
/// Layers are built in `spec.layers` order, which is DRAW order, which is
/// z-order. Callers go through [`prepare`], which validates first.
///
/// # Errors
///
/// Returns `GgsqlError::WriterError` if a layer's data is missing, a geom
/// cannot be built, or the assembled composition fails hephaestus's own
/// validation.
pub fn build_composition(
    spec: &Plot,
    data: &HashMap<String, DataFrame>,
) -> Result<PlotComposition> {
    // FACET → a grid of named panels (a single panel when unfaceted). Each
    // panel becomes one hephaestus `Plot` sharing the composition's scales.
    let (composition, panels) = facet::build_panels(spec, data)?;
    // The composition owns the shape registry backing composition-level legend
    // glyphs (point markers, line dashes).
    let mut view = PlotComposition::new(&composition)
        .shape_registry(ShapeRegistry::with_builtins())
        .theme(wiring::ggsql_theme());

    // LABEL title/subtitle/caption live on the composition rather than the
    // panels, so one label spans the whole figure — also correct unfaceted,
    // where a panel-level title would resolve to the same layout row.
    if let Some(text) = wiring::plot_label(spec, "title") {
        view = view.title(text);
    }
    if let Some(text) = wiring::plot_label(spec, "subtitle") {
        view = view.subtitle(text);
    }
    if let Some(text) = wiring::plot_label(spec, "caption") {
        view = view.caption(text);
    }

    // Axis titles are composition chrome too: one centred title per
    // dimension for the whole figure, rather than one per panel rail.
    for (side, text) in projection::composition_axis_titles(spec) {
        view = view.axis_title(side, text);
    }

    // Register the fixed (shared) scales once, globally. Every panel binds
    // its position channels to these names, giving fixed-scale faceting.
    for scale in &spec.scales {
        let kind = match scale.aesthetic.as_str() {
            "fill" | "stroke" => scales::RangeKind::Color,
            "shape" => scales::RangeKind::Shape,
            "linetype" => scales::RangeKind::Linetype,
            // The text geom's font aesthetics: a scale over them resolves a
            // range of family names / weights, not numbers.
            "typeface" => scales::RangeKind::Text,
            "fontweight" => scales::RangeKind::FontWeight,
            "italic" => scales::RangeKind::Bool,
            _ => {
                if scale.aesthetic.starts_with("pos") {
                    scales::RangeKind::Position
                } else {
                    scales::RangeKind::Number
                }
            }
        };
        if let Some(hs) = build_scale(scale, kind) {
            view.insert_scale(scale.aesthetic.clone(), hs);
        }
    }

    // Frame a map to its bounding box. Marks, clip boundary and graticules
    // share one pre-projected data space, so the position scales must span the
    // map's extent rather than the marks' or the data drifts off the boundary.
    // A spatial layer has no `pos1`/`pos2` at all, so these are its only
    // scales. The bbox is ggsql's, per the "never invent extents" principle,
    // and it is the authority here rather than a resolved `pos1`/`pos2`: a map
    // resolves those against the *graticule* extent in EPSG:4326, so their
    // domain is in degrees while the data is in the target CRS. Any
    // `SCALE lon`/`lat` limits are already folded into the bbox, and the breaks
    // those scales carry arrive as projected geometry in `computed`.
    let map_bbox = map_bbox(spec, data)?;
    if let Some((xmin, ymin, xmax, ymax)) = map_bbox {
        view.insert_scale("pos1".to_string(), scale::continuous(map_range(xmin, xmax)));
        view.insert_scale("pos2".to_string(), scale::continuous(map_range(ymin, ymax)));
    }

    // Collected from the first panel only and registered once on the
    // composition's legend ring, so a faceted plot gets one shared legend.
    // Every panel builds the same legends from the same global scales.
    let legend_sink = std::cell::RefCell::new(Vec::new());
    let mut legends_captured = false;

    for panel in &panels {
        // Slice each layer's data to this panel. A Grid cell with no matching
        // rows still becomes a panel — framed, axed and strip-labelled, just
        // without marks — so the grid stays rectangular (the ggplot2 look).
        let slices: Vec<(&Layer, DataFrame)> = spec
            .layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                Ok((
                    layer,
                    facet::panel_dataframe(layer_dataframe(layer, idx, data)?, panel)?,
                ))
            })
            .collect::<Result<_>>()?;
        let empty = slices.iter().all(|(_, df)| df.height() == 0);

        // Fixed dimensions bind the shared `pos1`/`pos2`; free ones get a
        // per-panel scale over this panel's slices — the one place the writer
        // computes an extent of its own.
        let mut ps = facet::PanelScales::new(spec, panel);
        let layer_dfs: Vec<&DataFrame> = slices.iter().map(|(_, df)| df).collect();
        if ps.free_x {
            match scales::free_position_scale(spec.find_scale("pos1"), &layer_dfs, "pos1") {
                Some(hs) => view.insert_scale(ps.pos1.clone(), hs),
                // An empty cell has no extent to free the dimension over, so
                // fall back to the shared scale rather than leave the axis and
                // channels bound to a scale that was never inserted.
                None => ps.use_shared("pos1"),
            }
        }
        if ps.free_y {
            match scales::free_position_scale(spec.find_scale("pos2"), &layer_dfs, "pos2") {
                Some(hs) => view.insert_scale(ps.pos2.clone(), hs),
                None => ps.use_shared("pos2"),
            }
        }

        // Build every layer's geom into this panel, in DRAW = z-order; geoms
        // bind channels and record legends into `legend_sink`. An empty panel
        // builds no geoms, so it cannot be the legend-capturing one.
        let panel_legends = (!legends_captured).then_some(&legend_sink);
        let mut plot = HPlot::new(&composition, panel.id.as_str())
            .shape_registry(ShapeRegistry::with_builtins());
        if !empty {
            for (layer, df) in &slices {
                let ctx = Ctx {
                    spec,
                    layer,
                    df,
                    transposed: is_transposed(layer),
                    pos1_scale: &ps.pos1,
                    pos2_scale: &ps.pos2,
                    legends: panel_legends,
                };
                geom::build_into_plot(&mut plot, &ctx)?;
            }
            legends_captured = true;
        } else {
            // Grid lines come from the scales bound to the projection's
            // channels, which a geom would normally bind. With no geoms, bind
            // them here so an empty cell keeps its neighbours' grid. A position
            // with no resolved scale stays unbound — binding one that was never
            // registered fails validation.
            for (channel, name) in [("x", &ps.pos1), ("y", &ps.pos2)] {
                if view.scale(name).is_some() {
                    plot.set_binding(channel, name.clone());
                }
            }
        }

        // Axes are created per coordinate system, edge-only for fixed scales.
        plot = apply_projection(plot, spec, panel, &ps);

        // Lock a map panel to square units so the projection keeps its
        // proportions (a globe stays round). `aspect_ratio` is the data-space
        // x-unit : y-unit ratio, not width:height — map coordinates arrive
        // pre-projected, so one unit is the same length on both axes.
        if map_bbox.is_some() {
            plot = plot.aspect_ratio(1.0).aspect_mode(AspectMode::Range);
        }

        // Facet strip labels (Wrap/Grid-column header on top, Grid-row on right).
        if let Some(text) = &panel.strip_top {
            plot = plot.strip(AxisSide::Top, text.clone());
        }
        if let Some(text) = &panel.strip_right {
            plot = plot.strip(AxisSide::Right, text.clone());
        }

        view.attach_plot(plot);
    }

    // One shared legend for the whole composition (see `legend_sink` above).
    for legend in legend_sink.into_inner() {
        view.add_legend(legend);
    }

    let issues = view.validate();
    if !issues.is_empty() {
        return Err(GgsqlError::WriterError(format!(
            "the plot renderer could not lay this plot out: {issues:?}"
        )));
    }
    Ok(view)
}

/// The map bounding box `(xmin, ymin, xmax, ymax)`, or `None` when the plot is
/// not a map. ggsql's resolved `computed["bbox"]` (set under a `PROJECT map`)
/// wins; a bare `spatial` geom with no projection falls back to the union extent
/// of its geometry data.
fn map_bbox(
    spec: &Plot,
    data: &HashMap<String, DataFrame>,
) -> Result<Option<(f64, f64, f64, f64)>> {
    if let Some(proj) = &spec.project {
        if let Some(ParameterValue::Array(arr)) = proj.computed.get("bbox") {
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            if let [xmin, ymin, xmax, ymax] = nums[..] {
                if [xmin, ymin, xmax, ymax].iter().all(|v| v.is_finite()) {
                    return Ok(Some((xmin, ymin, xmax, ymax)));
                }
            }
        }
    }

    let is_spatial = |layer: &Layer| layer.geom.geom_type() == GeomType::Spatial;
    if !spec.layers.iter().any(is_spatial) {
        return Ok(None);
    }

    let geom_col = naming::aesthetic_column("geometry");
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (idx, layer) in spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| is_spatial(l))
    {
        let df = layer_dataframe(layer, idx, data)?;
        if df.column(&geom_col).is_err() {
            continue;
        }
        for g in channels::column_to_geometry(df, &geom_col)? {
            if let Some((x0, y0, x1, y1)) = g.bounds() {
                xmin = xmin.min(x0);
                ymin = ymin.min(y0);
                xmax = xmax.max(x1);
                ymax = ymax.max(y1);
            }
        }
    }
    Ok(
        (xmin.is_finite() && ymin.is_finite() && xmax.is_finite() && ymax.is_finite())
            .then_some((xmin, ymin, xmax, ymax)),
    )
}

/// A non-degenerate inclusive range for a map's continuous position scale.
///
/// The extent is padded by [`MAP_PADDING`] around its centre, matching the
/// Vega-Lite writer, which fits the projection to `span * 1.1` centred on the
/// bbox (`vegalite/projection/map.rs`). An inverted extent is oriented first
/// and a zero-width one widened, so the scale can always map the result.
pub(super) fn map_range(min: f64, max: f64) -> std::ops::RangeInclusive<f64> {
    // Orient first: `map_bbox` only checks its four numbers for finiteness, and
    // padding a reversed bbox leaves a range no scale can map.
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let span = max - min;
    if span > f64::EPSILON {
        let pad = span * MAP_PADDING / 2.0;
        (min - pad)..=(max + pad)
    } else {
        (min - 0.5)..=(max + 0.5)
    }
}

/// Look up the DataFrame backing a layer by its execution-assigned data key,
/// falling back to the conventional key for its index as the Vega-Lite writer
/// does. Execution always assigns the key; the fallback is for a hand-built
/// `Plot`.
pub(super) fn layer_dataframe<'a>(
    layer: &Layer,
    idx: usize,
    data: &'a HashMap<String, DataFrame>,
) -> Result<&'a DataFrame> {
    let key = layer
        .data_key
        .clone()
        .unwrap_or_else(|| naming::layer_key(idx));
    data.get(&key)
        .ok_or_else(|| GgsqlError::WriterError(format!("no data found for layer key '{key}'")))
}
