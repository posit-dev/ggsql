//! The WebP writer.

use std::collections::HashMap;

use hephaestus::image::encode_webp;

use super::canvas::Canvas;
use super::{compose, raster, RasterRenderer};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result};

/// Writer that renders a ggsql plot to a lossless WebP image.
///
/// The best default for a raster plot delivered over a wire: lossless like PNG,
/// alpha included, about as fast as `png` at `compression=fast`, and roughly
/// half the bytes on plot content.
///
/// No `quality` and no `compression` — the VP8L lossless bitstream has no rate
/// control to expose. Every other option is the shared canvas set:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any CSS color, e.g. `white`, `#ff0000`, `transparent` | `white` |
///
/// Rendering requires a working wgpu adapter (hardware or software, e.g.
/// lavapipe) at render time.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WebpWriter {
    canvas: Canvas,
}

impl WebpWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
        }
    }

    /// Set the background color used to clear the canvas before rendering.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Render through a renderer the caller keeps, rather than building one.
    ///
    /// Constructing a [`RasterRenderer`] creates a GPU device and compiles the
    /// rasteriser's shaders, so a host rendering more than one figure should
    /// build one once and pass it here.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the plot cannot be composed, the
    /// render fails, or the encode fails.
    pub fn write_with(
        &self,
        spec: &Plot,
        data: &HashMap<String, DataFrame>,
        renderer: &mut RasterRenderer,
    ) -> Result<Vec<u8>> {
        let pixels = raster::pixels(spec, data, &self.canvas, renderer)?;
        // Straight alpha in, straight alpha out — VP8L stores exactly the
        // buffer the renderer read back.
        encode_webp(
            self.canvas.width,
            self.canvas.height,
            &pixels,
            self.canvas.dpi_hint(),
        )
        .map_err(|e| GgsqlError::WriterError(format!("webp encode failed: {e}")))
    }

    /// [`Self::write_with`] from a resolved `Spec`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_with`].
    pub fn render_with(
        &self,
        spec: &crate::reader::ResolvedPlot,
        renderer: &mut RasterRenderer,
    ) -> Result<Vec<u8>> {
        self.write_with(spec.plot(), spec.data(), renderer)
    }
}

impl Writer for WebpWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        Ok(Self {
            canvas: Canvas::from_options(options, &[])?,
        })
    }

    fn validate_plot(&self, spec: &Plot) -> Result<()> {
        compose::validate_plot(spec)
    }

    fn write_plot(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output> {
        let mut renderer = RasterRenderer::new()?;
        self.write_with(spec, data, &mut renderer)
    }
}

#[cfg(test)]
impl super::canvas::Canvased for WebpWriter {
    fn canvas(&self) -> &Canvas {
        &self.canvas
    }
}

#[cfg(test)]
mod option_tests {
    use super::*;
    use crate::writer::hephaestus::canvas::{
        assert_canvas_semantics, assert_transparent_background,
    };

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<WebpWriter>();
        assert_transparent_background::<WebpWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let options = WriterOptions::parse(&[] as &[&str]).unwrap();
        assert_eq!(
            WebpWriter::from_options(&options).unwrap(),
            WebpWriter::default()
        );
    }

    #[test]
    fn there_is_no_rate_knob_to_mistype() {
        // VP8L has no quality or compression setting, so naming one is an error
        // rather than a silently ignored request for a smaller file.
        for absent in ["quality=80", "compression=fast"] {
            let options = WriterOptions::parse([absent]).unwrap();
            let err = WebpWriter::from_options(&options).unwrap_err().to_string();
            assert!(err.contains("unknown writer option"), "{absent}: {err}");
        }
    }
}
