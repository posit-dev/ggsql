//! `bar`, `histogram`, and `tile` geoms → hephaestus `RectGeom`.
//!
//! Bars occupy a `width`-fraction of their category band, offset per dodge
//! group (both come from ggsql: the `width` param and the `pos1offset`/
//! `pos2offset` columns + `Layer::adjusted_width`); the value axis runs
//! baseline→value. Histograms span explicit bin edges; tiles span min/max
//! extents (continuous) or fill the band (discrete). Bars/histograms are
//! orientation-aware.

use hephaestus::color::rgb8;

use super::super::channels::{aesthetic_column_name, column_to_f64};
use super::super::scales::RangeKind;
use super::super::wiring::{
    band_half_width, dodge_offsets, stroke_style_constants, Ctx, GeomSpec, LegendKind, MatDefault,
    MaterialSpec, PanelAxis, PositionSpec,
};
use crate::plot::layer::geom::GeomType;

pub fn spec(ctx: &Ctx) -> GeomSpec {
    let (positions, raw_numbers, data_channels) = match ctx.layer.geom.geom_type() {
        GeomType::Bar => {
            let (positions, bands) = bar(ctx);
            (positions, vec![], bands)
        }
        GeomType::Histogram => (histogram(ctx.transposed), vec![], vec![]),
        GeomType::Tile => {
            let (positions, bands) = tile(ctx);
            (positions, vec![], bands)
        }
        _ => (Vec::new(), vec![], vec![]),
    };

    GeomSpec {
        positions,
        material: vec![
            MaterialSpec::new(
                "fill",
                "fill",
                RangeKind::Color,
                MatDefault::Color(rgb8(0, 0, 0)),
            ),
            MaterialSpec::new("stroke", "stroke", RangeKind::Color, MatDefault::None),
            MaterialSpec::new(
                "opacity",
                "fill_opacity",
                RangeKind::Number,
                MatDefault::Number(0.8),
            ),
            MaterialSpec::new(
                "linewidth",
                "linewidth",
                RangeKind::Number,
                MatDefault::None,
            ),
            MaterialSpec::new(
                "linetype",
                "linetype",
                RangeKind::Linetype,
                MatDefault::None,
            ),
        ],
        raw_strings: &[],
        constants: stroke_style_constants(ctx.layer),
        raw_numbers,
        data_channels,
        legend_key: LegendKind::Rect,
        grouped: false,
    }
}

/// Categorical bar: x/x2 share the category column; the band edges come from
/// `width`/dodge as per-row band offsets. The value axis runs baseline→value.
fn bar(ctx: &Ctx) -> (Vec<PositionSpec>, Vec<(&'static str, Vec<f64>)>) {
    // A synthetic band — the `__ggsql_stat_dummy` a bar with no category mapping
    // sits on — has no neighbours to leave a gap for, so it takes the whole band
    // rather than the layer's `width`. Under polar that band axis is the radius,
    // where a 0.9 width would open a hole in the middle of a pie and leave a gap
    // at its rim; on a Cartesian axis it is the single full-width bar ggplot2
    // draws for an ungrouped count.
    let band_axis = if ctx.transposed { "pos2" } else { "pos1" };
    let dummy = ctx
        .spec
        .find_scale(band_axis)
        .is_some_and(|scale| scale.is_dummy());
    let half = if dummy {
        0.5
    } else {
        band_half_width(ctx.layer, 0.9)
    };
    if !ctx.transposed {
        let offsets = dodge_offsets(ctx.df, "pos1offset");
        let lo = offsets.iter().map(|o| o - half).collect();
        let hi = offsets.iter().map(|o| o + half).collect();
        (
            vec![
                PositionSpec::new("x", "pos1", PanelAxis::X),
                PositionSpec::new("x2", "pos1", PanelAxis::X),
                PositionSpec::new("y", "pos2end", PanelAxis::Y),
                PositionSpec::new("y2", "pos2", PanelAxis::Y),
            ],
            vec![("x_band", lo), ("x2_band", hi)],
        )
    } else {
        let offsets = dodge_offsets(ctx.df, "pos2offset");
        let lo = offsets.iter().map(|o| o - half).collect();
        let hi = offsets.iter().map(|o| o + half).collect();
        (
            vec![
                PositionSpec::new("y", "pos2", PanelAxis::Y),
                PositionSpec::new("y2", "pos2", PanelAxis::Y),
                PositionSpec::new("x", "pos1end", PanelAxis::X),
                PositionSpec::new("x2", "pos1", PanelAxis::X),
            ],
            vec![("y_band", lo), ("y2_band", hi)],
        )
    }
}

/// Histogram: bins span explicit edges on the main axis, value runs baseline→count.
fn histogram(transposed: bool) -> Vec<PositionSpec> {
    if !transposed {
        vec![
            PositionSpec::new("x", "pos1", PanelAxis::X),
            PositionSpec::new("x2", "pos1end", PanelAxis::X),
            PositionSpec::new("y", "pos2end", PanelAxis::Y),
            PositionSpec::new("y2", "pos2", PanelAxis::Y),
        ]
    } else {
        vec![
            PositionSpec::new("y", "pos2", PanelAxis::Y),
            PositionSpec::new("y2", "pos2end", PanelAxis::Y),
            PositionSpec::new("x", "pos1end", PanelAxis::X),
            PositionSpec::new("x2", "pos1", PanelAxis::X),
        ]
    }
}

/// Tile/heatmap. Each direction is parameterised on its own, because ggsql's
/// tile stat resolves them independently (`tile::process_direction`) and a tile
/// may be discrete on one axis and continuous on the other — which the
/// Vega-Lite writer's `TileRenderer` also handles per axis.
fn tile(ctx: &Ctx) -> (Vec<PositionSpec>, Vec<(&'static str, Vec<f64>)>) {
    let (mut positions, mut bands) = tile_axis(ctx, PanelAxis::X);
    let (y_positions, y_bands) = tile_axis(ctx, PanelAxis::Y);
    positions.extend(y_positions);
    bands.extend(y_bands);
    (positions, bands)
}

/// One tile direction: a continuous one spans the explicit min/max extents ggsql
/// resolved, a discrete one sits on the category centre and occupies a
/// `width`/`height` fraction of its band (1.0 = full band, like VL's
/// `datum.width * bandwidth`), so its edges are at ±fraction/2.
fn tile_axis(ctx: &Ctx, axis: PanelAxis) -> (Vec<PositionSpec>, Vec<(&'static str, Vec<f64>)>) {
    let (centre, min, max, size, near, far, near_band, far_band) = match axis {
        PanelAxis::X => (
            "pos1", "pos1min", "pos1max", "width", "x", "x2", "x_band", "x2_band",
        ),
        PanelAxis::Y => (
            "pos2", "pos2min", "pos2max", "height", "y", "y2", "y_band", "y2_band",
        ),
    };
    if aesthetic_column_name(ctx.layer, min).is_some() {
        (
            vec![
                PositionSpec::new(near, min, axis),
                PositionSpec::new(far, max, axis),
            ],
            vec![],
        )
    } else {
        let (lo, hi) = band_edges(ctx, size);
        (
            vec![
                PositionSpec::new(near, centre, axis),
                PositionSpec::new(far, centre, axis),
            ],
            vec![(near_band, lo), (far_band, hi)],
        )
    }
}

/// Per-row band edges (`-fraction/2`, `+fraction/2`) for a discrete tile's
/// `width`/`height` column; a missing column defaults to a full (1.0) band.
fn band_edges(ctx: &Ctx, aesthetic: &str) -> (Vec<f64>, Vec<f64>) {
    let name = crate::naming::aesthetic_column(aesthetic);
    let fracs = column_to_f64(ctx.df, &name).unwrap_or_else(|_| vec![1.0; ctx.df.height()]);
    let lo = fracs.iter().map(|f| -f / 2.0).collect();
    let hi = fracs.iter().map(|f| f / 2.0).collect();
    (lo, hi)
}
