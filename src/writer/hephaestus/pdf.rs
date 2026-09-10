//! The PDF writer.

use std::collections::HashMap;

use hephaestus::pdf::{encode_pdf, PdfConfig, PdfScene, PdfWarning};

use super::canvas::Canvas;
use super::{compose, vector};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, Plot, Result};

/// Option keys [`PdfWriter`] adds to the shared canvas set.
const PDF_OPTIONS: &[&str] = &["compress", "links"];

/// Writer that renders a ggsql plot to a PDF page.
///
/// Needs no GPU adapter — like the SVG writer, it records the same drawing
/// commands the rasteriser would have executed. The page carries vector geometry
/// and subset-embedded fonts, so a figure prints without resampling.
///
/// [`PdfWriter::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any CSS color; `transparent` leaves the page unpainted | `white` |
/// | `compress` | Deflate the content streams | `true` |
/// | `links` | Emit link annotations for text carrying a destination | `true` |
///
/// The page box is derived from the canvas at 72 points per inch, so
/// `width=6;height=4;units=in;dpi=300` produces a 432×288 pt page — six inches
/// wide on paper — from an 1800×1200 rendering. `units=px` gives a page sized
/// as if those pixels were rendered at `dpi`.
///
/// `compress=false` leaves the content stream as readable text, which is how
/// you inspect what was emitted or diff two figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdfWriter {
    canvas: Canvas,
    compress: bool,
    links: bool,
}

impl PdfWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            ..Self::default()
        }
    }

    /// Set the background painted behind the plot.
    ///
    /// A fully transparent color leaves the page unpainted rather than adding a
    /// full-page rect in transparent black.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Deflate the content streams. On by default; off leaves them readable.
    pub fn compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// Emit link annotations for text carrying a destination.
    pub fn links(mut self, links: bool) -> Self {
        self.links = links;
        self
    }

    /// Render, reporting anything PDF could not express.
    ///
    /// [`Writer::write`] discards the report. Take it when the output is an
    /// artifact someone will ship — a dropped gradient is a defect in the file.
    /// The list is empty for everything ggsql draws.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the plot cannot be composed.
    pub fn write_reporting(
        &self,
        spec: &Plot,
        data: &HashMap<String, DataFrame>,
    ) -> Result<(Vec<u8>, Vec<String>)> {
        let mut scene = PdfScene::with_config(self.canvas.size(), self.canvas.dpi, self.config());
        vector::draw(spec, data, &self.canvas, &mut scene)?;
        Ok((encode_pdf(&scene), describe(scene.warnings())))
    }

    /// [`Self::write_reporting`] from a resolved `Spec`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_reporting`].
    pub fn render_reporting(&self, spec: &crate::reader::Spec) -> Result<(Vec<u8>, Vec<String>)> {
        self.write_reporting(spec.plot(), spec.data())
    }

    /// The emission options this writer's settings amount to.
    fn config(&self) -> PdfConfig {
        PdfConfig::new()
            .background(self.canvas.vector_background())
            .compress(self.compress)
            .links(self.links)
    }
}

impl Default for PdfWriter {
    fn default() -> Self {
        Self {
            canvas: Canvas::default(),
            compress: true,
            links: true,
        }
    }
}

impl Writer for PdfWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        Ok(Self {
            canvas: Canvas::from_options(options, PDF_OPTIONS)?,
            compress: options.boolean("compress")?.unwrap_or(true),
            links: options.boolean("links")?.unwrap_or(true),
        })
    }

    fn validate(&self, spec: &Plot) -> Result<()> {
        compose::validate_plot(spec)
    }

    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output> {
        self.write_reporting(spec, data).map(|(pdf, _)| pdf)
    }
}

/// Put what the format could not express into ggsql's own words.
///
/// See `svg::describe` for why this translates rather than re-exports.
fn describe(warnings: &[PdfWarning]) -> Vec<String> {
    warnings
        .iter()
        .map(|warning| match warning {
            PdfWarning::SweepGradient => {
                "a sweep gradient was flattened to a solid colour; PDF has no conic shading".into()
            }
            PdfWarning::UnsupportedExtend => {
                "a gradient asked to repeat or reflect; PDF shadings only pad, so the end colours \
                 were held"
                    .into()
            }
            PdfWarning::UnsupportedCompose => {
                "a blend mode PDF cannot express was drawn as normal compositing".into()
            }
            PdfWarning::AsymmetricCaps => {
                "a stroke asked for different start and end caps; PDF has one, so both took the \
                 start cap"
                    .into()
            }
            PdfWarning::ImageBrushUnsupported => {
                "an image used as a fill or stroke was dropped; PDF cannot paint with one".into()
            }
            PdfWarning::UnembeddableImage => "an image's pixel layout could not be embedded".into(),
            PdfWarning::MissingPngFeature => {
                "a colour glyph's bitmap could not be decoded: this build has no PNG decoder".into()
            }
            PdfWarning::GlyphNotDrawable => {
                "a glyph had no outline this backend could draw and did not appear".into()
            }
            PdfWarning::NonFiniteCoordinate => {
                "a coordinate was not a finite number and was written as zero".into()
            }
            // Unbalanced layers are a defect in the writer rather than a limit
            // of the format, and the variants are non-exhaustive.
            other => format!("the plot renderer reported '{other:?}'"),
        })
        .collect()
}

#[cfg(test)]
impl super::canvas::Canvased for PdfWriter {
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

    fn writer(pairs: &[&str]) -> Result<PdfWriter> {
        PdfWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<PdfWriter>();
        assert_transparent_background::<PdfWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = PdfWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert!(default.compress);
        assert!(default.links);
    }

    #[test]
    fn the_flags_take_the_boolean_spellings() {
        assert!(!writer(&["compress=false"]).unwrap().compress);
        assert!(!writer(&["compress=no"]).unwrap().compress);
        assert!(!writer(&["links=off"]).unwrap().links);
        assert!(writer(&["compress=1", "links=yes"]).unwrap().compress);
        let err = writer(&["compress=maybe"]).unwrap_err().to_string();
        assert!(err.contains("'compress' expects true or false"), "{err}");
    }

    #[test]
    fn a_transparent_canvas_leaves_the_page_unpainted() {
        assert_eq!(
            writer(&["background=none"]).unwrap().config().background,
            None
        );
        assert!(writer(&[]).unwrap().config().background.is_some());
    }
}
