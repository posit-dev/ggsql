//! The TIFF writer.

use std::collections::HashMap;

use hephaestus::image::encode_tiff;
pub use hephaestus::image::TiffCompression;

use super::canvas::Canvas;
use super::{compose, raster, RasterRenderer};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result};

/// Option keys [`TiffWriter`] adds to the shared canvas set.
const TIFF_OPTIONS: &[&str] = &["compression"];

/// How a TIFF's image data is compressed. All four are lossless.
const COMPRESSION_VALUES: &[&str] = &["none", "deflate", "lzw", "packbits"];

/// Writer that renders a ggsql plot to a TIFF image.
///
/// Lossless with alpha preserved, and the format a print workflow or an older
/// imaging tool is most likely to insist on. [`TiffWriter::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any CSS color, e.g. `white`, `#ff0000`, `transparent` | `white` |
/// | `compression` | `none`, `deflate`, `lzw`, or `packbits` | `deflate` |
///
/// All four are lossless, so `compression` is a compatibility choice: `deflate`
/// is smallest, `lzw` opens in the widest range of old readers, `packbits` is
/// cheap and does well on flat fills, and `none` stores rows verbatim.
///
/// Rendering requires a working wgpu adapter (hardware or software, e.g.
/// lavapipe) at render time.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TiffWriter {
    canvas: Canvas,
    compression: TiffCompression,
}

impl TiffWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            compression: TiffCompression::default(),
        }
    }

    /// Set the background color used to clear the canvas before rendering.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Set how the image data is compressed.
    pub fn compression(mut self, compression: TiffCompression) -> Self {
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
        encode_tiff(
            self.canvas.width,
            self.canvas.height,
            &pixels,
            self.compression,
            self.canvas.dpi_hint(),
        )
        .map_err(|e| GgsqlError::WriterError(format!("tiff encode failed: {e}")))
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

impl Writer for TiffWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        let canvas = Canvas::from_options(options, TIFF_OPTIONS)?;
        let compression = match options.one_of("compression", COMPRESSION_VALUES)? {
            Some("none") => TiffCompression::None,
            Some("lzw") => TiffCompression::Lzw,
            Some("packbits") => TiffCompression::Packbits,
            _ => TiffCompression::Deflate,
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
impl super::canvas::Canvased for TiffWriter {
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

    fn writer(pairs: &[&str]) -> Result<TiffWriter> {
        TiffWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<TiffWriter>();
        assert_transparent_background::<TiffWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = TiffWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert_eq!(default.compression, TiffCompression::Deflate);
    }

    #[test]
    fn compression_takes_the_four_named_compressors() {
        let cases = [
            ("none", TiffCompression::None),
            ("deflate", TiffCompression::Deflate),
            ("lzw", TiffCompression::Lzw),
            ("packbits", TiffCompression::Packbits),
        ];
        for (value, expected) in cases {
            let w = writer(&[&format!("compression={value}")]).unwrap();
            assert_eq!(w.compression, expected, "compression={value}");
        }
        // A png level is not a tiff compressor, and the error says which are.
        let err = writer(&["compression=fast"]).unwrap_err().to_string();
        assert!(err.contains("'compression' expects"), "{err}");
        assert!(err.contains("packbits"), "{err}");
    }
}
