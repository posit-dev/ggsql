//! The PNG writer.

use std::collections::HashMap;

use hephaestus::png::{encode_png, PngCompression};

use super::canvas::Canvas;
use super::{compose, raster, RasterRenderer};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result};

/// Option keys [`PngWriter`] adds to the shared canvas set.
const PNG_OPTIONS: &[&str] = &["compression"];

/// How hard the PNG encoder works to make the file small.
const COMPRESSION_VALUES: &[&str] = &["none", "fast", "balanced", "small"];

/// Writer that renders a ggsql plot to a PNG image.
///
/// Configured with a target pixel size and DPI because raster rendering needs
/// concrete dimensions, unlike the resolution-independent Vega-Lite writer.
/// [`PngWriter::from_options`] builds the same configuration from
/// key–value [`WriterOptions`]:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any CSS color, e.g. `white`, `#ff0000`, `transparent` | `white` |
/// | `compression` | `none`, `fast`, `balanced`, or `small` | `balanced` |
///
/// `compression` trades encode time against file size, losslessly either way.
/// `balanced` is what a file wants; `fast` suits a caller on a frame deadline,
/// costing a fraction of the time for about half again the bytes.
///
/// Rendering requires a working wgpu adapter (hardware or software, e.g.
/// lavapipe) at render time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PngWriter {
    canvas: Canvas,
    compression: PngCompression,
}

impl PngWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            compression: PngCompression::Balanced,
        }
    }

    /// Set the background color used to clear the canvas before rendering.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Set how hard the encoder works to make the file small.
    pub fn compression(mut self, compression: PngCompression) -> Self {
        self.compression = compression;
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
        // `render_to_buffer` hands out straight (un-premultiplied) alpha, which
        // is exactly what PNG stores, so the buffer encodes as-is.
        encode_png(
            self.canvas.width,
            self.canvas.height,
            &pixels,
            self.compression,
            self.canvas.dpi_hint(),
        )
        .map_err(|e| GgsqlError::WriterError(format!("png encode failed: {e}")))
    }

    /// [`Self::write_with`] from a resolved `Spec`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_with`].
    pub fn render_with(
        &self,
        spec: &crate::reader::Spec,
        renderer: &mut RasterRenderer,
    ) -> Result<Vec<u8>> {
        self.write_with(spec.plot(), spec.data(), renderer)
    }
}

impl Default for PngWriter {
    fn default() -> Self {
        Self {
            canvas: Canvas::default(),
            compression: PngCompression::Balanced,
        }
    }
}

impl Writer for PngWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        let canvas = Canvas::from_options(options, PNG_OPTIONS)?;
        let compression = match options.one_of("compression", COMPRESSION_VALUES)? {
            Some("none") => PngCompression::None,
            Some("fast") => PngCompression::Fast,
            Some("small") => PngCompression::Small,
            _ => PngCompression::Balanced,
        };
        Ok(Self {
            canvas,
            compression,
        })
    }

    fn validate(&self, spec: &Plot) -> Result<()> {
        compose::validate_plot(spec)
    }

    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output> {
        let mut renderer = RasterRenderer::new()?;
        self.write_with(spec, data, &mut renderer)
    }
}

#[cfg(test)]
impl super::canvas::Canvased for PngWriter {
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

    fn writer(pairs: &[&str]) -> Result<PngWriter> {
        PngWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<PngWriter>();
        assert_transparent_background::<PngWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = PngWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert_eq!(default.compression, PngCompression::Balanced);
    }

    #[test]
    fn compression_takes_the_four_named_levels() {
        let cases = [
            ("none", PngCompression::None),
            ("fast", PngCompression::Fast),
            ("balanced", PngCompression::Balanced),
            ("small", PngCompression::Small),
        ];
        for (value, expected) in cases {
            let w = writer(&[&format!("compression={value}")]).unwrap();
            assert_eq!(w.compression, expected, "compression={value}");
        }
        let err = writer(&["compression=furlongs"]).unwrap_err().to_string();
        assert!(err.contains("'compression' expects"), "{err}");
    }
}
