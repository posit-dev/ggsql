//! `point` geom: one marker per row.

use hephaestus::color::rgb8;

use super::super::scales::RangeKind;
use super::super::wiring::{
    Ctx, GeomSpec, LegendKind, MatDefault, MaterialSpec, PanelAxis, PositionSpec,
};

pub fn spec(_ctx: &Ctx) -> GeomSpec {
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
            MaterialSpec::new("size", "size", RangeKind::Number, MatDefault::Number(3.0)),
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
            MaterialSpec::new("shape", "shape", RangeKind::Shape, MatDefault::None),
        ],
        raw_strings: &[],
        constant_strings: Vec::new(),
        raw_numbers: vec![],
        data_channels: vec![],
        legend_key: LegendKind::Point,
        grouped: false,
    }
}
