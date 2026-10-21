//! `line`, `path`, and `smooth` geoms → hephaestus `LineGeom`.
//!
//! Rows are grouped into separate polylines by the layer's partition columns;
//! within a group hephaestus connects rows in source order (ggsql pre-orders
//! line by pos1, path keeps raw order, smooth emits the fitted curve).

use hephaestus::color::rgb8;

use super::super::scales::RangeKind;
use super::super::wiring::{
    Ctx, GeomSpec, LegendKind, MatDefault, MaterialSpec, PanelAxis, PositionSpec,
};
use crate::plot::layer::geom::GeomType;

pub fn spec(ctx: &Ctx) -> GeomSpec {
    // ggsql defaults: line is black @ 1.5pt; smooth is blue (#3366FF) @ 2pt.
    let (stroke, linewidth) = if ctx.layer.geom.geom_type() == GeomType::Smooth {
        (rgb8(51, 102, 255), 2.0)
    } else {
        (rgb8(0, 0, 0), 1.5)
    };
    GeomSpec {
        positions: vec![
            PositionSpec::new("x", "pos1", PanelAxis::X),
            PositionSpec::new("y", "pos2", PanelAxis::Y),
        ],
        material: vec![
            MaterialSpec::new(
                "stroke",
                "stroke",
                RangeKind::Color,
                MatDefault::Color(stroke),
            ),
            MaterialSpec::new(
                "linewidth",
                "linewidth",
                RangeKind::Number,
                MatDefault::Number(linewidth),
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
        ],
        raw_strings: &[],
        constant_strings: super::super::wiring::arrow_markers(ctx.layer),
        raw_numbers: vec![],
        data_channels: vec![],
        legend_key: LegendKind::Line,
        grouped: true,
    }
}
