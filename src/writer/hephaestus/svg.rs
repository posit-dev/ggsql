//! The SVG writer.

use std::collections::HashMap;

use hephaestus::svg::{encode_svg, SvgConfig, SvgScene, SvgUnits, SvgWarning, TextMode};

use super::canvas::Canvas;
use super::{compose, vector};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, Plot, Result};

/// Option keys [`SvgWriter`] adds to the shared canvas set.
const SVG_OPTIONS: &[&str] = &["text", "embed-fonts", "id-prefix"];

/// How text is written into the file.
const TEXT_VALUES: &[&str] = &["text", "outline"];

/// Writer that renders a ggsql plot to SVG.
///
/// Needs no GPU adapter: SVG records the same drawing commands the rasteriser
/// would have executed, so this works on a headless box and in CI. The output
/// is resolution independent and its text is real, selectable text.
///
/// [`SvgWriter::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width, in `units` | 1500 px |
/// | `height` | Canvas height, in `units` | 1000 px |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Pixels per inch; converts physical sizes, including `units` | 300 |
/// | `background` | Any CSS color; `transparent` emits no background at all | `white` |
/// | `text` | `text` (real `<text>`) or `outline` (glyphs as `<path>`) | `text` |
/// | `embed-fonts` | Inline the font files, so the file renders identically anywhere | `false` |
/// | `id-prefix` | Prefix for generated element ids | none |
///
/// `units` is honoured in the output as well as in the input: a canvas given in
/// a physical unit declares its size in points, so
/// `width=6;height=4;units=in;dpi=300` yields an 1800×1200 `viewBox` on a
/// `432pt` root — a file that *prints* six inches wide. A pixel canvas stays in
/// pixels.
///
/// `text=outline` makes the file self-contained without embedding a font, at
/// the cost of selectable text; `embed-fonts` keeps the text but can take a
/// 30 kB plot past 3 MB. Hence neither is the default.
///
/// `id-prefix` is a correctness setting: two SVGs inlined into one HTML page
/// that both define `#lg0` will have the second's `url(#lg0)` resolve to the
/// first's gradient, in every browser.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvgWriter {
    canvas: Canvas,
    text: TextMode,
    embed_fonts: bool,
    id_prefix: Option<String>,
}

impl SvgWriter {
    /// Create a writer for the given pixel dimensions and DPI, white background.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            ..Self::default()
        }
    }

    /// Set the background painted behind the plot.
    ///
    /// A fully transparent color emits no background element at all, rather
    /// than a full-canvas rect painted in transparent black.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Write glyph outlines as `<path>` instead of `<text>` elements.
    pub fn outline_text(mut self, outline: bool) -> Self {
        self.text = if outline {
            TextMode::Outline
        } else {
            TextMode::Text
        };
        self
    }

    /// Inline the font files the plot uses.
    pub fn embed_fonts(mut self, embed: bool) -> Self {
        self.embed_fonts = embed;
        self
    }

    /// Prefix every generated element id, so two inlined files cannot collide.
    pub fn id_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.id_prefix = Some(prefix.into());
        self
    }

    /// Render, reporting anything SVG could not express.
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
    ) -> Result<(String, Vec<String>)> {
        let mut scene = SvgScene::with_config(self.canvas.size(), self.canvas.dpi, self.config());
        vector::draw(spec, data, &self.canvas, &mut scene)?;
        Ok((encode_svg(&scene), describe(scene.warnings())))
    }

    /// [`Self::write_reporting`] from a resolved `Spec`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_reporting`].
    pub fn render_reporting(&self, spec: &crate::reader::Spec) -> Result<(String, Vec<String>)> {
        self.write_reporting(spec.plot(), spec.data())
    }

    /// The emission options this writer's settings amount to.
    fn config(&self) -> SvgConfig {
        let units = if self.canvas.physical {
            // A canvas asked for in inches should print at that size, which is
            // what a `pt` root declares. `SvgUnits::Pt` leaves the viewBox in
            // pixels and only suffixes `width`/`height`.
            SvgUnits::Pt
        } else {
            SvgUnits::Px
        };
        let mut config = SvgConfig::new()
            .background(self.canvas.vector_background())
            .units(units)
            .text(self.text)
            .embed_fonts(self.embed_fonts);
        if let Some(prefix) = &self.id_prefix {
            config = config.id_prefix(prefix.clone());
        }
        config
    }
}

impl Writer for SvgWriter {
    type Output = String;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        let canvas = Canvas::from_options(options, SVG_OPTIONS)?;
        let text = match options.one_of("text", TEXT_VALUES)? {
            Some("outline") => TextMode::Outline,
            _ => TextMode::Text,
        };
        Ok(Self {
            canvas,
            text,
            embed_fonts: options.boolean("embed-fonts")?.unwrap_or(false),
            id_prefix: options.get("id-prefix").map(str::to_string),
        })
    }

    fn validate(&self, spec: &Plot) -> Result<()> {
        compose::validate_plot(spec)
    }

    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output> {
        self.write_reporting(spec, data).map(|(svg, _)| svg)
    }
}

/// Put what the format could not express into ggsql's own words.
///
/// The renderer's warning variants are `#[non_exhaustive]`, and re-exporting
/// them would leak its type names into ggsql's API. Translating at the boundary
/// is also where those names get scrubbed.
fn describe(warnings: &[SvgWarning]) -> Vec<String> {
    warnings
        .iter()
        .map(|warning| match warning {
            SvgWarning::SweepGradient => {
                "a sweep gradient was flattened to a solid colour; SVG has no conic gradient".into()
            }
            SvgWarning::UnsupportedCompose => {
                "a blend mode SVG cannot express was drawn as normal compositing".into()
            }
            SvgWarning::AsymmetricCaps => {
                "a stroke asked for different start and end caps; SVG has one, so both took the \
                 start cap"
                    .into()
            }
            SvgWarning::RadialFocalRadius => {
                "a radial gradient's focal radius was written as SVG 2's 'fr', which older \
                 viewers ignore"
                    .into()
            }
            SvgWarning::ImageBrushUnsupported => {
                "an image used as a fill or stroke was dropped; SVG cannot paint with one".into()
            }
            SvgWarning::NonFiniteCoordinate => {
                "a coordinate was not a finite number and was written as zero".into()
            }
            SvgWarning::TextWithoutSource => {
                "some text arrived with neither a string nor an outline and was not drawn".into()
            }
            SvgWarning::MissingPngFeature => {
                "an image could not be embedded: this build has no PNG encoder".into()
            }
            SvgWarning::UnembeddableImage => "an image's pixel layout could not be embedded".into(),
            SvgWarning::FontNotEmbeddable => {
                "a font could not be inlined — font collections cannot be — so its text will \
                 render in whatever font the viewer resolves"
                    .into()
            }
            // Unbalanced layers or scopes are a defect in the writer rather
            // than a limit of the format, and the variants are non-exhaustive.
            other => format!("the plot renderer reported '{other:?}'"),
        })
        .collect()
}

#[cfg(test)]
impl super::canvas::Canvased for SvgWriter {
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

    fn writer(pairs: &[&str]) -> Result<SvgWriter> {
        SvgWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<SvgWriter>();
        assert_transparent_background::<SvgWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = SvgWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert_eq!(default.text, TextMode::Text);
        assert!(!default.embed_fonts);
        assert_eq!(default.id_prefix, None);
    }

    #[test]
    fn text_takes_the_two_modes() {
        assert_eq!(writer(&["text=text"]).unwrap().text, TextMode::Text);
        assert_eq!(writer(&["text=outline"]).unwrap().text, TextMode::Outline);
        let err = writer(&["text=fancy"]).unwrap_err().to_string();
        assert!(err.contains("'text' expects 'text' or 'outline'"), "{err}");
    }

    #[test]
    fn the_flags_read_either_spelling_of_their_key() {
        for key in ["embed-fonts", "embed_fonts"] {
            assert!(
                writer(&[&format!("{key}=true")]).unwrap().embed_fonts,
                "{key}"
            );
            assert!(
                !writer(&[&format!("{key}=no")]).unwrap().embed_fonts,
                "{key}"
            );
        }
        for key in ["id-prefix", "id_prefix"] {
            let w = writer(&[&format!("{key}=fig1-")]).unwrap();
            assert_eq!(w.id_prefix.as_deref(), Some("fig1-"), "{key}");
        }
        let err = writer(&["embed-fonts=maybe"]).unwrap_err().to_string();
        assert!(err.contains("'embed_fonts' expects true or false"), "{err}");
    }

    #[test]
    fn a_transparent_canvas_emits_no_background_element() {
        let clear = writer(&["background=none"]).unwrap();
        assert_eq!(clear.config().background, None);
        let white = writer(&[]).unwrap();
        assert!(white.config().background.is_some());
    }

    #[test]
    fn a_physical_canvas_declares_its_size_in_points() {
        assert_eq!(
            writer(&["units=in", "width=6"]).unwrap().config().units,
            SvgUnits::Pt
        );
        assert_eq!(writer(&["width=600"]).unwrap().config().units, SvgUnits::Px);
    }
}
