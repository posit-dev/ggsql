//! `spatial` geom → hephaestus `GeometryGeom`. A custom builder (not the generic
//! position/material path) because the geom carries no `x`/`y` columns: each row
//! is a single `Geometry` value whose coordinates resolve through the plot's
//! bound `x`/`y` scales at draw time. Under a `PROJECT map` the plot's projection
//! (a Custom clip surface built from `projection.rs`) shapes those coordinates;
//! with no `PROJECT` the geometry draws in raw data space under Cartesian.

use hephaestus::color::rgb8;
use hephaestus::plot::geom::Raw;
use hephaestus::plot::{GeometryGeom, Plot as HPlot};

use super::super::channels::column_to_geometry;
use super::super::scales::RangeKind;
use super::super::wiring::{
    stroke_style_constants, wire_material, Ctx, LegendKind, MatDefault, MaterialSpec,
};
use crate::naming;
use crate::Result;

pub fn build(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    let df = ctx.df;

    // The geometry aesthetic is always materialised to the internal WKB column.
    let geoms = column_to_geometry(df, &naming::aesthetic_column("geometry"))?;

    let mut b = GeometryGeom::builder();
    b.set("geometry", geoms);

    // Coordinates map through the panel's pos1/pos2 scales (bbox-framed; see
    // `PngWriter::write`). GeometryGeom has no x/y channel, but its draw
    // resolves each coordinate against these bound scales.
    plot.set_binding("x", ctx.pos1_scale);
    plot.set_binding("y", ctx.pos2_scale);

    // Every material aesthetic goes through the shared path, so each is honored
    // whether it's the ggsql literal default, a `SETTING` constant, or
    // data-mapped (scale-bound + legended) — a choropleth is just a data-mapped
    // `fill`. ggsql's spatial defaults (grey fill, black border, opacity 0.8,
    // linewidth 0.2, solid) arrive as literals; the `MatDefault`s match them so
    // a legend key still carries the layer's look when nothing is mapped.
    let material = [
        MaterialSpec::new(
            "fill",
            "fill",
            RangeKind::Color,
            MatDefault::Color(rgb8(0x74, 0x74, 0x74)),
        ),
        MaterialSpec::new(
            "stroke",
            "stroke",
            RangeKind::Color,
            MatDefault::Color(rgb8(0, 0, 0)),
        ),
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
            MatDefault::Number(0.2),
        ),
        MaterialSpec::new(
            "linetype",
            "linetype",
            RangeKind::Linetype,
            MatDefault::None,
        ),
    ];
    wire_material(&mut b, &material, plot, ctx, LegendKind::Rect)?;

    for (channel, value) in stroke_style_constants(ctx.layer) {
        b.set(channel, Raw(value));
    }
    plot.add_geom(b.build());
    Ok(())
}
