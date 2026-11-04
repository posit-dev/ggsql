//! `polygon` geom → hephaestus `PolygonGeom`. Rows are grouped into separate
//! closed polygons by the layer's partition columns.

use hephaestus::color::rgb8;

use super::super::scales::RangeKind;
use super::super::wiring::{
    stroke_style_constants, Ctx, GeomSpec, LegendKind, MatDefault, MaterialSpec, PanelAxis,
    PositionSpec,
};

pub fn spec(ctx: &Ctx) -> GeomSpec {
    GeomSpec {
        positions: vec![
            PositionSpec::new("x", "pos1", PanelAxis::X),
            PositionSpec::new("y", "pos2", PanelAxis::Y),
        ],
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
        raw_numbers: vec![],
        data_channels: vec![],
        legend_key: LegendKind::Rect,
        grouped: true,
    }
}
