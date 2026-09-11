//! Layers ggsql expanded into per-vertex rows under a map `PROJECT`.
//!
//! Projecting a straight edge onto a curved surface bends it, so ggsql densifies
//! the edge in SQL (`plot/layer/geom/{segment,rule,ribbon,tile}.rs`
//! `apply_projection`): each original row becomes a run of vertex rows, the
//! extent aesthetics (`pos1end`/`pos2end`, `pos1min`/`pos2max`, …) are remapped
//! onto plain `pos1`/`pos2`, and `__ggsql_densify_id__` — appended to the
//! layer's `partition_by` — ties one row's vertices back together. The layer is
//! flagged with the `densified` parameter.
//!
//! The mark therefore changes: an open shape (`segment`, `rule`) draws as a
//! polyline and a closed one (`ribbon`, `tile`) as a filled outline. Those are
//! exactly the `line` and `polygon` geoms — same vertex columns, same grouping
//! by `partition_by`, same material tables — so their specs are reused whole.
//! The Vega-Lite writer makes the same swap, to a `line` mark with
//! `interpolate: linear-closed` for the closed shapes.

use hephaestus::plot::{LineGeom, Plot as HPlot, PolygonGeom};

use super::super::wiring::{build_and_add, Ctx};
use super::{line, polygon};
use crate::plot::layer::geom::GeomType;
use crate::plot::ParameterValue;
use crate::{GgsqlError, Layer, Result};

/// Whether ggsql expanded this layer's rows into projected vertices.
pub fn applies(layer: &Layer) -> bool {
    matches!(
        layer.parameters.get("densified"),
        Some(ParameterValue::Boolean(true))
    )
}

/// Draw the expanded vertices: a polyline per original row for segment/rule, a
/// closed polygon per original row for ribbon/tile.
pub fn build(plot: &mut HPlot, ctx: &Ctx) -> Result<()> {
    match ctx.layer.geom.geom_type() {
        GeomType::Segment | GeomType::Rule => build_and_add::<LineGeom>(plot, line::spec(ctx), ctx),
        GeomType::Ribbon | GeomType::Tile => {
            build_and_add::<PolygonGeom>(plot, polygon::spec(ctx), ctx)
        }
        other => Err(GgsqlError::WriterError(format!(
            "the plot renderer cannot draw a densified '{other}' geom"
        ))),
    }
}
