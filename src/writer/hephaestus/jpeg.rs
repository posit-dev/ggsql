//! The JPEG writer.

use std::collections::HashMap;

use hephaestus::image::encode_jpeg;

use super::canvas::Canvas;
use super::{compose, raster, RasterRenderer};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result};

/// Option keys [`JpegWriter`] adds to the shared canvas set.
const JPEG_OPTIONS: &[&str] = &["quality"];

/// Default JPEG quality. High enough that the ringing around a plot's thin dark
/// strokes and text stays out of the way, without being the pointless end of
/// the scale.
const DEFAULT_QUALITY: u8 = 90;

/// Writer that renders a ggsql plot to a JPEG image.
///
/// JPEG is the wrong codec for most plots: its ringing lands on exactly the thin
/// dark strokes and small text a plot is made of. Use it when something
/// downstream insists; `png` and `webp` are lossless and usually smaller on
/// plot content.
///
/// [`JpegWriter::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any **opaque** CSS color | `white` |
/// | `quality` | 1–100; higher is larger and less lossy | 90 |
///
/// `background` must be opaque: JPEG has no alpha channel, so a transparent
/// canvas has nowhere to go. Rather than silently composite the plot onto black,
/// the writer refuses the setting.
///
/// Rendering requires a working wgpu adapter (hardware or software, e.g.
/// lavapipe) at render time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JpegWriter {
    canvas: Canvas,
    quality: u8,
}

impl JpegWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            quality: DEFAULT_QUALITY,
        }
    }

    /// Set the background the plot is composited onto.
    ///
    /// Any alpha is ignored, since the format has no channel for it.
    /// [`JpegWriter::from_options`] rejects a transparent `background` rather
    /// than dropping it silently; a caller building the writer has chosen.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Set the quality, from 1 to 100. Values outside that range are clamped.
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
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
        encode_jpeg(
            self.canvas.width,
            self.canvas.height,
            &pixels,
            self.quality,
            self.canvas.background,
            self.canvas.dpi_hint(),
        )
        .map_err(|e| GgsqlError::WriterError(format!("jpeg encode failed: {e}")))
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

impl Default for JpegWriter {
    fn default() -> Self {
        Self {
            canvas: Canvas::default(),
            quality: DEFAULT_QUALITY,
        }
    }
}

impl Writer for JpegWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        let canvas = Canvas::from_options(options, JPEG_OPTIONS)?;
        if canvas.background.components[3] < 1.0 {
            return Err(GgsqlError::WriterError(
                "writer option 'background' resolves to a translucent color, but jpeg has no \
                 alpha channel; give an opaque background"
                    .to_string(),
            ));
        }
        let quality = match options.number("quality")? {
            Some(quality) if (1.0..=100.0).contains(&quality) => quality.round() as u8,
            Some(quality) => {
                return Err(GgsqlError::WriterError(format!(
                    "writer option 'quality' expects a number from 1 to 100, got '{quality}'"
                )))
            }
            None => DEFAULT_QUALITY,
        };
        Ok(Self { canvas, quality })
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
impl super::canvas::Canvased for JpegWriter {
    fn canvas(&self) -> &Canvas {
        &self.canvas
    }
}

#[cfg(test)]
mod option_tests {
    use super::*;
    use crate::writer::hephaestus::canvas::assert_canvas_semantics;

    fn writer(pairs: &[&str]) -> Result<JpegWriter> {
        JpegWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<JpegWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = JpegWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert_eq!(default.quality, DEFAULT_QUALITY);
    }

    #[test]
    fn quality_spans_one_to_a_hundred() {
        assert_eq!(writer(&["quality=1"]).unwrap().quality, 1);
        assert_eq!(writer(&["quality=100"]).unwrap().quality, 100);
        for bad in ["quality=0", "quality=101", "quality=-5"] {
            let err = writer(&[bad]).unwrap_err().to_string();
            assert!(
                err.contains("'quality' expects a number from 1 to 100"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn a_transparent_background_is_refused_rather_than_dropped() {
        for spelling in [
            "background=none",
            "background=transparent",
            "background=#00000000",
        ] {
            let err = writer(&[spelling]).unwrap_err().to_string();
            assert!(
                err.contains("jpeg has no alpha channel"),
                "{spelling}: {err}"
            );
        }
        // An opaque color is fine, whichever way it is spelled.
        assert!(writer(&["background=black"]).is_ok());
    }
}
