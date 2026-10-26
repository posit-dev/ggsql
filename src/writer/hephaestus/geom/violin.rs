//! `violin` composite geom. ggsql's stat emits a KDE grid per group
//! (`pos1` = category, `pos2` = value, `offset` = pre-scaled half-width). We
//! render one `RibbonGeom` band per (category, partition group): one edge sits at
//! `+offset` and the other at `-offset` of the category band (via the ribbon's
//! per-row band-offset channels), sharing the value channel. One ribbon row per
//! KDE grid sample, so the contour needs no hand-built outline.
//!
//! Which axis carries the categories follows the layer's orientation
//! (`BandAxes`); `side` collapses the band to one half, leaving the other edge on
//! the centreline (so a half-violin can pair with a half-boxplot).

use std::cmp::Ordering;
use std::collections::HashMap;

use hephaestus::color::rgb8;
use hephaestus::plot::geom::Raw;
use hephaestus::plot::{Plot as HPlot, RibbonGeom};

use super::super::channels::{
    build_group_keys, column_to_channel, column_to_f64, column_to_strings,
};
use super::super::scales::RangeKind;
use super::super::wiring::{
    apply_material, band_edges, dodge_offsets, require_column, resolve_color, resolve_material,
    side_sign, stroke_style_constants, BandAxes, Ctx, LegendKind, MatDefault, MaterialSpec,
};
use crate::Result;

/// The layer aesthetics this composite styles, with ggsql's violin defaults.
/// Used both to resolve them and to dress the legend keys in the layer's look.
fn material() -> [MaterialSpec; 5] {
    [
        MaterialSpec::new(
            "fill",
            "fill",
            RangeKind::Color,
            MatDefault::Color(rgb8(255, 255, 255)),
        ),
        MaterialSpec::new(
            "stroke",
            "stroke",
            RangeKind::Color,
            MatDefault::Color(rgb8(60, 60, 60)),
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
        MaterialSpec::new(
            "opacity",
            "fill_opacity",
            RangeKind::Number,
            MatDefault::None,
        ),
    ]
}

pub fn build(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    let (layer, df) = (ctx.layer, ctx.df);

    let axes = BandAxes::new(ctx);
    let band_aes = axes.band();
    let value_aes = axes.value();

    let band_col = require_column(ctx, band_aes)?;
    let value_col = require_column(ctx, value_aes)?;
    let offset = require_column(ctx, "offset")?;

    let p1 = column_to_channel(df, band_col)?;
    let cat = column_to_strings(df, band_col)?;
    let p2 = column_to_f64(df, value_col)?;
    let off = column_to_f64(df, offset)?;

    // One contour per (category, partition group): the category alone would merge
    // a dodged violin's groups into a single blob, since ggsql keeps position
    // aesthetics out of `partition_by`. The Vega-Lite writer composes its `detail`
    // encoding the same way.
    let partitions = build_group_keys(df, &layer.partition_by)?;
    let keys: Vec<String> = match &partitions {
        Some(parts) => cat
            .iter()
            .zip(parts)
            .map(|(c, p)| format!("{c}\u{1f}{p}"))
            .collect(),
        None => cat.clone(),
    };

    // Order rows so each violin's band is contiguous and ascending in the value
    // axis (RibbonGeom connects a mark's rows in source order).
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        let g = *index.entry(k.as_str()).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    let mut order: Vec<usize> = Vec::with_capacity(cat.len());
    for rows in &mut groups {
        rows.sort_by(|&a, &b| p2[a].partial_cmp(&p2[b]).unwrap_or(Ordering::Equal));
        order.extend_from_slice(rows);
    }

    // A channel always drives the same panel axis, whatever the orientation.
    for (channel, scale) in [
        ("x", ctx.pos1_scale),
        ("x2", ctx.pos1_scale),
        ("y", ctx.pos2_scale),
        ("y2", ctx.pos2_scale),
    ] {
        plot.set_binding(channel, scale);
    }

    // One ribbon per category, its two edges at ±offset of the category band (or
    // centreline → offset for a one-sided `side`), both shifted by the dodge
    // offset (zero when not dodged).
    let dodge = dodge_offsets(df, axes.dodge());
    let side = side_sign(layer);
    let ordered_keys: Vec<String> = order.iter().map(|&i| keys[i].clone()).collect();
    let edges: Vec<(f64, f64)> = order
        .iter()
        .map(|&i| {
            let (near, far) = band_edges(off[i], side);
            (dodge[i] + near, dodge[i] + far)
        })
        .collect();
    let band: Vec<f64> = edges.iter().map(|&(near, _)| near).collect();
    let band2: Vec<f64> = edges.iter().map(|&(_, far)| far).collect();
    let values: Vec<f64> = order.iter().map(|&i| p2[i]).collect();

    // What this composite styles, in one table: the ggsql defaults a legend key
    // should wear when nothing is mapped, and the aliasing each resolve below
    // uses. A composite has no `GeomSpec`, so it declares the same table itself.
    let material = material();

    // Resolve fill + stroke once (data-mapped → shared scale/legend, else
    // constant), mirroring the VL writer's shared-encoding model.
    let fill = resolve_color(
        ctx,
        plot,
        "fill",
        "fill",
        rgb8(255, 255, 255),
        LegendKind::Rect,
        &material,
    )?;
    let stroke = resolve_color(
        ctx,
        plot,
        "stroke",
        "stroke",
        rgb8(60, 60, 60),
        LegendKind::Rect,
        &material,
    )?;
    // Outline width + dash pattern, applied to both ribbon edges.
    let linewidth = resolve_material(
        ctx,
        plot,
        "linewidth",
        "linewidth",
        RangeKind::Number,
        LegendKind::Line,
        &material,
    )?;
    let linetype = resolve_material(
        ctx,
        plot,
        "linetype",
        "linetype",
        RangeKind::Linetype,
        LegendKind::Line,
        &material,
    )?;
    // Resolved like the rest, so a mapped column fades each violin by its own
    // datum rather than by the first row's.
    let alpha = resolve_material(
        ctx,
        plot,
        "opacity",
        "fill_opacity",
        RangeKind::Number,
        LegendKind::Rect,
        &material,
    )?;
    // Which of the ribbon's two edges take an outline. Both, normally; only the
    // far edge under a one-sided `side`, where the near edge is the centreline.
    let outline_edges: &[&str] = if side.is_some() {
        &["stroke2", "linewidth2", "linetype2"]
    } else {
        &[
            "stroke",
            "stroke2",
            "linewidth",
            "linewidth2",
            "linetype",
            "linetype2",
        ]
    };
    // The ribbon's two edges share each outline scale (the `2` suffix is the far
    // edge), so a data-mapped stroke/width/dash styles both sides alike.
    for (source, channel) in [
        (Some(&stroke), "stroke2"),
        (linewidth.as_ref(), "linewidth2"),
        (linetype.as_ref(), "linetype2"),
    ] {
        if let Some(name) = source.and_then(|s| s.scale_name()) {
            plot.set_binding(channel, name);
        }
    }

    // Both band edges carry the category; only one value channel is set, which is
    // what selects the ribbon's orientation (a vertical band when the far edge is
    // on x, a horizontal one when it is on y).
    let (band_ch, band_ch2) = axes.band_channels();
    let (frac_ch, frac_ch2) = axes.band_fraction_channels();
    let (value_ch, _) = axes.value_channels();

    let mut b = RibbonGeom::builder();
    b.keys(ordered_keys);
    p1.select(&order).apply(&mut b, band_ch);
    p1.select(&order).apply(&mut b, band_ch2);
    b.set(frac_ch, band);
    b.set(frac_ch2, band2);
    b.set(value_ch, values);
    fill.apply(&mut b, "fill", &order);
    // Under a one-sided `side`, `band_edges` collapses curve A onto the band's
    // centreline, so stroking it would draw a rule down the flat side of every
    // half-violin. Only the curve that traces the density gets an outline.
    if outline_edges.contains(&"stroke") {
        stroke.apply(&mut b, "stroke", &order);
    }
    stroke.apply(&mut b, "stroke2", &order);
    // `RibbonGeom` resolves its outline channels once per mark (from the mark's
    // first row), so a data-mapped width/dash varies per violin, not per vertex.
    for (source, channels) in [
        (linewidth.as_ref(), ["linewidth", "linewidth2"]),
        (linetype.as_ref(), ["linetype", "linetype2"]),
    ] {
        if let Some(source) = source {
            for channel in channels.iter().filter(|c| outline_edges.contains(c)) {
                source.apply(&mut b, channel, &order);
            }
        }
    }
    apply_material(&mut b, alpha.as_ref(), "fill_opacity", &order);
    // The ribbon's two edges are stroked independently, so each style goes to
    // both edges' channels.
    for (channel, value) in
        stroke_style_constants(ctx.layer)
            .into_iter()
            .flat_map(|(channel, value)| {
                let second = if channel == "cap" { "cap2" } else { "join2" };
                [(channel, value.clone()), (second, value)]
            })
    {
        b.set(channel, Raw(value));
    }
    plot.add_geom(b.build());

    Ok(())
}
