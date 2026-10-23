//! `text` geom → hephaestus `TextGeom`. A custom builder (not the generic
//! position/material path) because `vjust`/`hjust` accept keywords and flip for
//! hephaestus's top-origin `anchor_y`, and because `offset` is a layer parameter
//! rather than an aesthetic. Everything else goes through `wire_material`, so a
//! scaled `fontsize` maps through its resolved scale like any other geom's — and
//! the font face a layer holds constant reaches its legend key.

use hephaestus::color::rgb8;
use hephaestus::plot::geom::Raw;
use hephaestus::plot::{Plot as HPlot, TextGeom};

use super::super::channels::{
    aesthetic_column_name, column_to_channel, column_to_f64, column_to_strings, is_text_column,
};
use super::super::scales::RangeKind;
use super::super::wiring::{
    constant_number, require_column, wire_material, Ctx, LegendKind, MatDefault, MaterialSpec,
};
use crate::plot::types::{ArrayElement, ParameterValue};
use crate::plot::AestheticValue;
use crate::Result;

pub fn build(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    let (layer, df) = (ctx.layer, ctx.df);
    let n = df.height();

    let pos1 = require_column(ctx, "pos1")?;
    let pos2 = require_column(ctx, "pos2")?;
    let label = require_column(ctx, "label")?;

    let mut b = TextGeom::builder();

    // Positions: bind to the panel's pos1/pos2 scales (panel-aware for free).
    let p1 = column_to_channel(df, pos1)?;
    let p2 = column_to_channel(df, pos2)?;
    plot.set_binding("x", ctx.pos1_scale);
    plot.set_binding("y", ctx.pos2_scale);
    p1.apply(&mut b, "x");
    p2.apply(&mut b, "y");

    // Label string.
    b.set("text", Raw(column_to_strings(df, label)?));

    // `parse` decides whether each label is read as markdown (hephaestus's
    // `markdown` channel, which routes the row through the rich-text shaper) or
    // as a literal string. ggsql defaults it on, so the channel is always bound
    // rather than left to hephaestus's own theme default of off.
    b.set("markdown", Raw(vec![parse(layer); n]));

    // Color, glyph outline, size, opacity and the font face: the shared material
    // path, so each is honored whether it arrives as a `SETTING` literal, a
    // scaled column (`SCALE fontsize TO (6, 20)` maps through its resolved scale)
    // or an identity column — and so a data-mapped one dresses its legend key in
    // the constants the layer holds. `text_stroke` and `family` have no default
    // because ggsql's defaults for `stroke` and `typeface` are Null: hephaestus
    // skips the outline pass entirely while the channel is unset, and an empty
    // family is not "use the default" but a font lookup that misses. The glyph
    // outline's width is hephaestus's theme default, as ggsql's text geom has no
    // `linewidth` aesthetic.
    let boxed = is_textbox(layer);
    // `SETTING box => true` declares a box without naming a colour; the
    // material path cannot read a boolean literal as a colour, so the white
    // default is bound directly here.
    if let Some(AestheticValue::Literal(ParameterValue::Boolean(true))) = layer.mappings.get("box")
    {
        b.set("bg_fill", Raw(vec![rgb8(255, 255, 255); n]));
    }
    wire_material(&mut b, &material(boxed), plot, ctx, LegendKind::Text)?;

    // Box geometry: layer parameters rather than aesthetics, bound as
    // constants like `offset` — but only in box-mode, where hephaestus draws
    // the rect they describe. `linetype` has no hephaestus channel yet.
    if boxed {
        for (param, channel) in [
            ("radius", "bg_corner_radius"),
            ("padding", "bg_padding"),
            ("linewidth", "bg_linewidth"),
        ] {
            if let Some(ParameterValue::Number(v)) = layer.parameters.get(param) {
                b.set(channel, Raw(vec![*v; n]));
            }
        }
    }

    // Justification needs conversion no `RangeKind` covers, so it is resolved per
    // row here: a mapped column, else the layer's literal repeated, else centred.
    b.set("anchor_x", Raw(justification(ctx, "hjust")));
    // ggsql vjust: 0 = bottom, 1 = top; hephaestus anchor_y: 0 = top, 1 = bottom.
    let anchor_y: Vec<f64> = justification(ctx, "vjust")
        .iter()
        .map(|v| 1.0 - v)
        .collect();
    b.set("anchor_y", Raw(anchor_y));

    // `offset` nudges the label off its anchor point, in points. It is a layer
    // parameter rather than an aesthetic, so it bypasses the material table
    // entirely. hephaestus's offsets are already in points and its y grows up,
    // so both components pass through unchanged — unlike the Vega-Lite writer,
    // which converts to pixels and negates y for VL's downward axis.
    let (dx, dy) = offset(layer);
    if dx != 0.0 || dy != 0.0 {
        b.set("x_offset", Raw(vec![dx; n]));
        b.set("y_offset", Raw(vec![dy; n]));
    }

    plot.add_geom(b.build());
    Ok(())
}

/// The layer aesthetics wired through the shared material path, with ggsql's
/// text defaults. This table is also what dresses the legend key, so every
/// aesthetic hephaestus's `Text` key consumes belongs here — the face a layer
/// sets is as much part of what a `fontsize` swatch describes as its colour is.
/// Only justification is left out: the key centres its glyph in the cell.
///
/// `stroke`'s channel depends on box-mode: bare text outlines glyphs
/// (`text_stroke`), boxed text outlines the box (`bg_stroke`). `box` itself
/// defaults to `None` — unset means hephaestus draws no background rect.
fn material(boxed: bool) -> [MaterialSpec; 9] {
    [
        MaterialSpec::new(
            "fill",
            "fill",
            RangeKind::Color,
            MatDefault::Color(rgb8(0, 0, 0)),
        ),
        MaterialSpec::new(
            "stroke",
            if boxed { "bg_stroke" } else { "text_stroke" },
            RangeKind::Color,
            MatDefault::None,
        ),
        MaterialSpec::new("box", "bg_fill", RangeKind::Color, MatDefault::None),
        MaterialSpec::new(
            "fontsize",
            "size",
            RangeKind::Number,
            MatDefault::Number(11.0),
        ),
        MaterialSpec::new(
            "opacity",
            "fill_opacity",
            RangeKind::Number,
            MatDefault::Number(1.0),
        ),
        MaterialSpec::new("typeface", "family", RangeKind::Text, MatDefault::None),
        MaterialSpec::new(
            "fontweight",
            "weight",
            RangeKind::FontWeight,
            MatDefault::Number(400.0),
        ),
        MaterialSpec::new("italic", "italic", RangeKind::Bool, MatDefault::None),
        // ggsql resolves `rotation` in degrees; hephaestus angles are radians
        // (math CCW), which `RangeKind::Angle` converts. A rotated layer gets a
        // rotated key, as ggplot2's `draw_key_text` does — hephaestus sizes the
        // swatch cell from the rotated glyph, so nothing is clipped.
        MaterialSpec::new("rotation", "angle", RangeKind::Angle, MatDefault::None),
    ]
}

/// Box-mode: the layer draws boxes when `box` is declared with a non-null
/// value, mapped or set. An explicit null (`box => null`) is the opt-out and
/// counts as undeclared, matching the null-sentinel convention.
fn is_textbox(layer: &crate::Layer) -> bool {
    match layer.mappings.get("box") {
        None => false,
        Some(AestheticValue::Literal(ParameterValue::Null)) => false,
        Some(AestheticValue::Literal(ParameterValue::Boolean(false))) => false,
        Some(_) => true,
    }
}

/// The layer's `offset` parameter as `(dx, dy)` in points. A bare number offsets
/// both axes; a two-element array gives them separately.
fn offset(layer: &crate::Layer) -> (f64, f64) {
    match layer.parameters.get("offset") {
        Some(ParameterValue::Number(n)) => (*n, *n),
        Some(ParameterValue::Array(a)) if a.len() == 2 => {
            let at = |i: usize| match a[i] {
                ArrayElement::Number(n) => n,
                _ => 0.0,
            };
            (at(0), at(1))
        }
        _ => (0.0, 0.0),
    }
}

/// The layer's `parse` parameter: whether a label is markdown. Defaults to
/// `true`, matching the geom's own default — a `PLACE` layer or a query built
/// without going through parameter resolution leaves it unset.
fn parse(layer: &crate::Layer) -> bool {
    match layer.parameters.get("parse") {
        Some(ParameterValue::Boolean(b)) => *b,
        _ => true,
    }
}

/// A justification aesthetic (`hjust` / `vjust`) as a 0–1 fraction. ggsql accepts
/// either a number or a keyword, so the keywords are mapped the way the
/// Vega-Lite writer's `convert_hjust` / `convert_vjust` map them to `align` /
/// `baseline`, and anything unrecognised centres.
fn justification(ctx: &Ctx, aesthetic: &str) -> Vec<f64> {
    let n = ctx.df.height();
    if let Some(col) = aesthetic_column_name(ctx.layer, aesthetic) {
        // Dispatch on the column's own type — a keyword column casts to numbers
        // without erroring, so a numeric read cannot be tried first (see
        // `is_text_column`).
        if is_text_column(ctx.df, col) {
            if let Ok(names) = column_to_strings(ctx.df, col) {
                return names.iter().map(|s| parse_justification(s)).collect();
            }
        } else if let Ok(values) = column_to_f64(ctx.df, col) {
            return values;
        }
        return vec![0.5; n];
    }
    // `SETTING vjust => 'top'` is a string literal, which `constant_number`
    // cannot read; try it as a number first, then as a keyword.
    let value = match ctx.layer.mappings.aesthetics.get(aesthetic) {
        Some(AestheticValue::Literal(ParameterValue::String(s))) => parse_justification(s),
        _ => constant_number(ctx, aesthetic, 0.5),
    };
    vec![value; n]
}

/// A justification keyword (or numeric string) as a 0–1 fraction; 0 is
/// left/bottom, 1 is right/top.
fn parse_justification(value: &str) -> f64 {
    if let Ok(n) = value.parse::<f64>() {
        return n;
    }
    match value.to_lowercase().as_str() {
        "left" | "bottom" => 0.0,
        "right" | "top" => 1.0,
        _ => 0.5, // centre / center / middle / unknown
    }
}
