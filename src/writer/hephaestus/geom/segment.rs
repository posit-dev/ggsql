//! `segment`, `range`, and `rule` geoms → hephaestus `SegmentGeom`.
//!
//! - segment: an explicit (pos1,pos2)→(pos1end,pos2end) line.
//! - range: a bar-less interval; aligned spans pos2min→pos2max at fixed pos1
//!   (transposed swaps).
//! - rule: a panel-spanning reference line at a fixed pos1 (vertical) or pos2
//!   (horizontal); the free axis uses scale-bypassing 0..1 panel fractions.

use hephaestus::color::rgb8;
use hephaestus::plot::geom::Raw;
use hephaestus::plot::{Plot as HPlot, SegmentGeom};

use super::super::channels::{aesthetic_column_name, column_to_channel, column_to_f64};
use super::super::scales::RangeKind;
use super::super::wiring::{
    constant_number, dodge_offsets, stroke_style_constants, wire_material, BandAxes, Ctx, GeomSpec,
    LegendKind, MatDefault, MaterialSpec, PanelAxis, PositionSpec,
};
use super::hinge::{caps, hinge_points};
use crate::plot::layer::geom::GeomType;
use crate::plot::ParameterValue;
use crate::{Layer, Result};

pub fn spec(ctx: &Ctx) -> GeomSpec {
    let (positions, raw_numbers) = match ctx.layer.geom.geom_type() {
        GeomType::Segment => (
            vec![
                PositionSpec::new("x", "pos1", PanelAxis::X),
                PositionSpec::new("y", "pos2", PanelAxis::Y),
                PositionSpec::new("x2", "pos1end", PanelAxis::X),
                PositionSpec::new("y2", "pos2end", PanelAxis::Y),
            ],
            vec![],
        ),
        GeomType::Range if !ctx.transposed => (
            vec![
                PositionSpec::new("x", "pos1", PanelAxis::X),
                PositionSpec::new("x2", "pos1", PanelAxis::X),
                PositionSpec::new("y", "pos2min", PanelAxis::Y),
                PositionSpec::new("y2", "pos2max", PanelAxis::Y),
            ],
            vec![],
        ),
        GeomType::Range => (
            vec![
                PositionSpec::new("y", "pos2", PanelAxis::Y),
                PositionSpec::new("y2", "pos2", PanelAxis::Y),
                PositionSpec::new("x", "pos1min", PanelAxis::X),
                PositionSpec::new("x2", "pos1max", PanelAxis::X),
            ],
            vec![],
        ),
        GeomType::Rule => rule(ctx),
        _ => (Vec::new(), vec![]),
    };

    GeomSpec {
        positions,
        material: material(),
        raw_strings: &[],
        constants: stroke_style_constants(ctx.layer),
        raw_numbers,
        data_channels: vec![],
        legend_key: LegendKind::Line,
        grouped: false,
    }
}

/// The stroke material table shared by every segment-family geom, including the
/// diagonal rule (which builds its positions itself but styles them the same).
fn material() -> Vec<MaterialSpec> {
    vec![
        MaterialSpec::new(
            "stroke",
            "stroke",
            RangeKind::Color,
            MatDefault::Color(rgb8(0, 0, 0)),
        ),
        MaterialSpec::new(
            "linewidth",
            "linewidth",
            RangeKind::Number,
            MatDefault::Number(1.0),
        ),
        MaterialSpec::new(
            "opacity",
            "stroke_opacity",
            RangeKind::Number,
            MatDefault::None,
        ),
        MaterialSpec::new(
            "linetype",
            "linetype",
            RangeKind::Linetype,
            MatDefault::None,
        ),
    ]
}

/// `range` end caps: the `hinge` SETTING (10pt by default, `null` to hide) draws
/// a cap across the band at **both** interval endpoints, on top of the interval's
/// own segment. Mirrors the Vega-Lite writer, which adds two `tick` layers of
/// `hinge` px beside the rule.
///
/// The caps take the same material table as the segment, so a data-mapped
/// stroke/width/dash styles them like the interval (its legend collapses into the
/// segment's, being the same scale).
pub fn build_hinges(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    let Some(hinge) = hinge_points(ctx.layer) else {
        return Ok(());
    };
    if ctx.df.height() == 0 {
        return Ok(());
    }
    let axes = BandAxes::new(ctx);
    // A range with no mapped position on the banded axis (ggsql's dummy axis) has
    // nothing to centre the caps on.
    let Some(band_col) = aesthetic_column_name(ctx.layer, axes.band()) else {
        return Ok(());
    };
    let band = column_to_channel(ctx.df, band_col)?;
    // Follow the interval's own position adjustment, which `wire_positions` put on
    // its band channels.
    let offsets = dodge_offsets(ctx.df, axes.dodge());

    for bound in ["min", "max"] {
        let aesthetic = format!("{}{bound}", axes.value());
        let Some(col) = aesthetic_column_name(ctx.layer, &aesthetic) else {
            continue;
        };
        let values = column_to_f64(ctx.df, col)?;
        let mut b = caps(ctx, axes, band.clone(), values, offsets.clone(), hinge);
        wire_material(&mut b, &material(), plot, ctx, LegendKind::Line)?;
        for (channel, value) in stroke_style_constants(ctx.layer) {
            b.set(channel, Raw(value));
        }
        plot.add_geom(b.build());
    }
    Ok(())
}

/// Whether this rule is a diagonal (abline): has a non-zero `slope`.
pub fn is_diagonal(layer: &Layer) -> bool {
    matches!(
        layer.parameters.get("diagonal"),
        Some(ParameterValue::Boolean(true))
    )
}

/// A diagonal rule (abline): **one line per data row**, each spanning the
/// position scales' resolved range with `secondary = slope * primary +
/// intercept`. Mirrors the Vega-Lite writer, whose `calculate` transforms compute
/// that expression per row from `datum.__ggsql_aes_slope__` and the intercept
/// field — so `MAPPING slope AS slope, y AS y` draws a line per row (with its own
/// slope, intercept, and material aesthetics), while `SETTING slope => 1, y => 0`
/// gives one row of literals and hence one line.
///
/// The spanning range comes straight from the scales (explicit `FROM` or
/// data-trained); when a scale is unresolved it falls back to 0..1 like any
/// continuous scale. Positions are computed rather than read from a column, so
/// this builds its own geom, but materials go through the shared `wire_material`
/// so a data-mapped `stroke`/`linetype`/`linewidth` is scaled and legended
/// exactly as on a plain segment.
pub fn build_diagonal(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    let n = ctx.df.height();
    if n == 0 {
        return Ok(());
    }
    let slopes = slope_values(ctx, n)?;

    let (x, x2, y, y2) = if !ctx.transposed {
        // y-intercept (`pos2`); x is the spanning axis.
        let intercepts = intercept_values(ctx, "pos2", n)?;
        let (x0, x1) = primary_range(ctx, "pos1");
        (
            vec![x0; n],
            vec![x1; n],
            secondary(&slopes, &intercepts, x0),
            secondary(&slopes, &intercepts, x1),
        )
    } else {
        // x-intercept (`pos1`); y is the spanning axis.
        let intercepts = intercept_values(ctx, "pos1", n)?;
        let (y0, y1) = primary_range(ctx, "pos2");
        (
            secondary(&slopes, &intercepts, y0),
            secondary(&slopes, &intercepts, y1),
            vec![y0; n],
            vec![y1; n],
        )
    };

    for (channel, scale) in [
        ("x", ctx.pos1_scale),
        ("x2", ctx.pos1_scale),
        ("y", ctx.pos2_scale),
        ("y2", ctx.pos2_scale),
    ] {
        plot.set_binding(channel, scale);
    }

    let mut b = SegmentGeom::builder();
    b.set("x", x);
    b.set("x2", x2);
    b.set("y", y);
    b.set("y2", y2);
    wire_material(&mut b, &material(), plot, ctx, LegendKind::Line)?;
    for (channel, value) in stroke_style_constants(ctx.layer) {
        b.set(channel, Raw(value));
    }
    plot.add_geom(b.build());
    Ok(())
}

/// `slope * primary + intercept` at one end of the spanning range.
fn secondary(slopes: &[f64], intercepts: &[f64], primary: f64) -> Vec<f64> {
    slopes
        .iter()
        .zip(intercepts)
        .map(|(s, i)| s * primary + i)
        .collect()
}

/// Resolved (min, max) for a position scale, or 0..1 when unresolved.
fn primary_range(ctx: &Ctx, aesthetic: &str) -> (f64, f64) {
    ctx.spec
        .find_scale(aesthetic)
        .and_then(|s| s.numeric_domain())
        .unwrap_or((0.0, 1.0))
}

/// Per-row slopes: the mapped `slope` column, else the literal or SETTING
/// parameter repeated for every row.
fn slope_values(ctx: &Ctx, n: usize) -> Result<Vec<f64>> {
    if let Some(col) = aesthetic_column_name(ctx.layer, "slope") {
        return column_to_f64(ctx.df, col);
    }
    let param = match ctx.layer.parameters.get("slope") {
        Some(ParameterValue::Number(v)) => *v,
        _ => 0.0,
    };
    Ok(vec![constant_number(ctx, "slope", param); n])
}

/// Per-row intercepts from the position aesthetic holding them (`pos2` for a
/// y-intercept, `pos1` when transposed): its column, else the literal value.
fn intercept_values(ctx: &Ctx, aesthetic: &str, n: usize) -> Result<Vec<f64>> {
    if let Some(col) = aesthetic_column_name(ctx.layer, aesthetic) {
        return column_to_f64(ctx.df, col);
    }
    Ok(vec![constant_number(ctx, aesthetic, 0.0); n])
}

/// A non-diagonal rule is a reference line spanning the whole panel on its free
/// axis. The free axis uses raw 0..1 panel fractions, so no scale/axis is
/// created for it.
fn rule(ctx: &Ctx) -> (Vec<PositionSpec>, Vec<(&'static str, f64)>) {
    if aesthetic_column_name(ctx.layer, "pos1").is_some() {
        // Vertical line at x = pos1, spanning full height.
        (
            vec![
                PositionSpec::new("x", "pos1", PanelAxis::X),
                PositionSpec::new("x2", "pos1", PanelAxis::X),
            ],
            vec![("y", 0.0), ("y2", 1.0)],
        )
    } else {
        // Horizontal line at y = pos2, spanning full width.
        (
            vec![
                PositionSpec::new("y", "pos2", PanelAxis::Y),
                PositionSpec::new("y2", "pos2", PanelAxis::Y),
            ],
            vec![("x", 0.0), ("x2", 1.0)],
        )
    }
}
