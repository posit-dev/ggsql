//! Drawing a composition into a vector scene.
//!
//! The counterpart to [`raster`](super::raster) for the backends that emit
//! drawing commands rather than pixels. `PlotComposition::render` takes
//! `&mut dyn SceneBuilder`, so this is the same call the rasteriser makes —
//! which is why SVG and PDF need no GPU adapter, no wgpu and no encoder.

use std::collections::HashMap;

use hephaestus::SceneBuilder;

use super::canvas::Canvas;
use super::compose;
use crate::{DataFrame, Plot, Result};

/// Check the plot, compose it, and draw it into `scene`.
///
/// Everything a vector writer does before serialising; what differs is the
/// scene type and how it is turned into bytes.
///
/// # Errors
///
/// Returns `GgsqlError::WriterError` if the plot cannot be drawn by this
/// renderer, or if composing it fails.
pub fn draw(
    spec: &Plot,
    data: &HashMap<String, DataFrame>,
    canvas: &Canvas,
    scene: &mut dyn SceneBuilder,
) -> Result<()> {
    let mut view = compose::prepare(spec, data)?;
    view.render(scene, canvas.size(), canvas.dpi);
    Ok(())
}
