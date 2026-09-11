//! The `hep` plot-document writer.

use std::collections::HashMap;

use hephaestus::document::{
    unsupported_items_for, write_composition, UnsupportedItem, WriteOptions,
};

use super::canvas::Canvas;
use super::{compose, CANVAS_SIZE_OPTIONS};
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result};

/// Option keys [`HepWriter`] adds to the canvas set.
const HEP_OPTIONS: &[&str] = &["lossy", "embed-fonts"];

/// Writer that captures a ggsql plot as a self-contained **`.hep`** plot
/// document.
///
/// Unlike every other writer here, this one produces no picture. It records the
/// resolved plot — scales, breaks, labels, theme, geometry and data channels —
/// so a consumer can render it itself at any size and re-render on resize
/// without going back to the query, which is what makes it the format for an
/// interactive host.
///
/// The name is the format's; ggsql does not define `.hep`. Needs no GPU adapter
/// and no encoder — it serialises the composition the other writers draw.
///
/// [`HepWriter::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Canvas width **hint**, in `units` | none |
/// | `height` | Canvas height **hint**, in `units` | none |
/// | `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width`/`height` are read | `px` |
/// | `dpi` | Resolution **hint** | none |
/// | `background` | Background a consumer should paint behind the plot | `white` |
/// | `lossy` | Drop what the format cannot carry instead of refusing | `false` |
/// | `embed-fonts` | Inline the font files the plot's text needs | `false` |
///
/// The size is a hint, not a canvas: `width`/`height`/`dpi` record what a
/// consumer should default to rather than fixing anything.
///
/// `lossy` decides what happens to a plot the format cannot fully carry.
/// Refusing is the default; with `lossy` on the same list comes back as
/// warnings from [`HepWriter::write_reporting`]. Nothing ggsql builds should
/// trip it, so a non-empty list is a bug here rather than a format limit.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HepWriter {
    canvas: Canvas,
    /// Whether a size was asked for at all, since an unset hint and a hint that
    /// happens to match the canvas default are different things to record.
    size_asked: bool,
    /// The same for `dpi`: the two are independent hints, so neither may record
    /// the other on the caller's behalf.
    dpi_asked: bool,
    lossy: bool,
    embed_fonts: bool,
}

impl HepWriter {
    /// A writer recording the given size and resolution as the consumer's
    /// default.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            canvas: Canvas::new(width, height, dpi),
            size_asked: true,
            dpi_asked: true,
            ..Self::default()
        }
    }

    /// Set the background a consumer should paint behind the plot.
    pub fn background(mut self, color: super::Color) -> Self {
        self.canvas = self.canvas.background(color);
        self
    }

    /// Drop what the format cannot carry instead of refusing to write.
    pub fn lossy(mut self, lossy: bool) -> Self {
        self.lossy = lossy;
        self
    }

    /// Inline the font files the plot's text needs.
    ///
    /// Off by default: a system family is often megabytes, and a consumer that
    /// can register its own fonts should.
    pub fn embed_fonts(mut self, embed: bool) -> Self {
        self.embed_fonts = embed;
        self
    }

    /// Write the document, reporting anything the format could not carry.
    ///
    /// With `lossy` off the same list is an error instead, so the report is
    /// non-empty only when the caller asked to degrade.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the plot cannot be composed, if it
    /// carries something the format cannot express and `lossy` is off, or if
    /// serialising fails.
    pub fn write_reporting(
        &self,
        spec: &Plot,
        data: &HashMap<String, DataFrame>,
    ) -> Result<(Vec<u8>, Vec<String>)> {
        let view = compose::prepare(spec, data)?;
        let options = self.options();

        // Checked here rather than left to `write_composition` so the error is
        // ggsql's own and names no renderer. The list is the same either way.
        let problems = unsupported_items_for(&view, &options);
        if !problems.is_empty() && !self.lossy {
            return Err(GgsqlError::WriterError(format!(
                "this plot cannot be captured as a document: {}. Pass lossy=true to write it \
                 anyway, dropping what cannot be carried",
                describe(&problems).join("; ")
            )));
        }

        let bytes = write_composition(&view, &options)
            .map_err(|e| GgsqlError::WriterError(format!("hep write failed: {e}")))?;
        Ok((bytes, describe(&problems)))
    }

    /// [`Self::write_reporting`] from a resolved `Spec`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_reporting`].
    pub fn render_reporting(
        &self,
        spec: &crate::reader::ResolvedPlot,
    ) -> Result<(Vec<u8>, Vec<String>)> {
        self.write_reporting(spec.plot(), spec.data())
    }

    /// The write options this writer's settings amount to.
    ///
    /// The canvas becomes hints. `embed_images` is left off: nothing ggsql
    /// builds registers an image, so the option cannot take effect.
    fn options(&self) -> WriteOptions {
        let mut options = WriteOptions::default();
        options.lossy = self.lossy;
        // Recorded unconditionally, unlike the size. `None` means unspecified
        // rather than transparent, so a transparent canvas has to travel as a
        // colour with zero alpha or a consumer paints white behind the plot.
        options.background = Some(self.canvas.background);
        options.size_hint = self
            .size_asked
            .then_some((self.canvas.width as f64, self.canvas.height as f64));
        options.dpi_hint = self.dpi_asked.then_some(self.canvas.dpi);
        options.embed_fonts = self.embed_fonts;
        options.embed_images = false;
        options
    }
}

impl Writer for HepWriter {
    type Output = Vec<u8>;

    fn from_options(options: &WriterOptions) -> Result<Self> {
        let canvas = Canvas::from_options(options, HEP_OPTIONS)?;
        // Each hint is recorded only when that hint was actually asked for.
        let size_asked = CANVAS_SIZE_OPTIONS
            .iter()
            .any(|key| options.get(key).is_some());
        Ok(Self {
            canvas,
            size_asked,
            dpi_asked: options.get("dpi").is_some(),
            lossy: options.boolean("lossy")?.unwrap_or(false),
            embed_fonts: options.boolean("embed-fonts")?.unwrap_or(false),
        })
    }

    fn validate_plot(&self, spec: &Plot) -> Result<()> {
        compose::validate_plot(spec)
    }

    fn write_plot(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output> {
        self.write_reporting(spec, data).map(|(bytes, _)| bytes)
    }
}

/// Put what the format could not carry into ggsql's own words.
///
/// As `svg::describe` and `pdf::describe`: the renderer's variants are
/// `#[non_exhaustive]`, and its `Display` text names renderer API and cargo
/// features a ggsql user cannot act on. Nothing ggsql builds should reach any
/// of these, so the wording aims at a writer bug rather than a format limit.
fn describe(problems: &[UnsupportedItem]) -> Vec<String> {
    problems
        .iter()
        .map(|problem| match problem {
            UnsupportedItem::CustomFormatter { scale } => format!(
                "the {scale} scale's tick labels are computed rather than listed, so a \
                 consumer cannot reproduce them"
            ),
            UnsupportedItem::UnnameableGeom { patch, index } => format!(
                "layer {} of panel {patch:?} is a mark the document cannot name, so nothing \
                 records how to draw it again",
                index + 1
            ),
            UnsupportedItem::TrackReference { location } => format!(
                "{location} is sized relative to another part of the layout, which only means \
                 something while this figure is being laid out"
            ),
            UnsupportedItem::UnnameableShape { patch, name } => format!(
                "the {name:?} marker on panel {patch:?} is a glyph with no source text, so a \
                 consumer cannot rebuild it"
            ),
            UnsupportedItem::UnembeddableImage { patch, name } => format!(
                "the {name:?} image on panel {patch:?} cannot be embedded by this build, so a \
                 consumer would have to supply it"
            ),
            // Non-exhaustive upstream, so report a problem this build has no
            // words for — without naming the renderer.
            _ => "something in this plot cannot be captured as a document".to_string(),
        })
        .collect()
}

#[cfg(test)]
impl super::canvas::Canvased for HepWriter {
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

    fn writer(pairs: &[&str]) -> Result<HepWriter> {
        HepWriter::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn canvas_options_behave_as_they_do_for_every_writer() {
        assert_canvas_semantics::<HepWriter>();
        assert_transparent_background::<HepWriter>();
    }

    #[test]
    fn the_default_writer_matches_no_options() {
        let default = HepWriter::default();
        assert_eq!(writer(&[]).unwrap(), default);
        assert!(!default.lossy);
        assert!(!default.embed_fonts);
    }

    #[test]
    fn a_hint_is_recorded_only_when_that_hint_was_asked_for() {
        // Any size works, so an unrecorded hint and a hint that happens to
        // equal the default are different things.
        let unset = writer(&[]).unwrap().options();
        assert_eq!(unset.size_hint, None);
        assert_eq!(unset.dpi_hint, None);

        // The two are independent: neither implies the other, or the document
        // would claim a default the caller never gave.
        let sized = writer(&["width=1600", "height=900"]).unwrap().options();
        assert_eq!(sized.size_hint, Some((1600.0, 900.0)));
        assert_eq!(sized.dpi_hint, None);

        let dense = writer(&["dpi=150"]).unwrap().options();
        assert_eq!(dense.size_hint, None);
        assert_eq!(dense.dpi_hint, Some(150.0));

        // A physical size resolves to pixels first, as it does everywhere —
        // and needs the dpi to do it, so both are asked for here.
        let physical = writer(&["width=6", "units=in", "dpi=100"])
            .unwrap()
            .options();
        assert_eq!(physical.size_hint.map(|(w, _)| w), Some(600.0));
        assert_eq!(physical.dpi_hint, Some(100.0));
    }

    #[test]
    fn a_background_is_always_recorded_transparent_included() {
        // `None` means unspecified rather than transparent, so a transparent
        // canvas travels as a colour with zero alpha.
        assert_eq!(
            writer(&[]).unwrap().options().background,
            Some(Canvas::default().background)
        );
        let transparent = writer(&["background=transparent"]).unwrap().options();
        let background = transparent.background.expect("recorded, not dropped");
        assert_eq!(background.components[3], 0.0);
    }

    #[test]
    fn a_problem_is_reported_without_naming_the_renderer() {
        let problems = [
            UnsupportedItem::CustomFormatter {
                scale: "pos1".into(),
            },
            UnsupportedItem::UnnameableGeom {
                patch: "panel".into(),
                index: 0,
            },
        ];
        for message in describe(&problems) {
            for leak in ["with_named_format", "Geom::kind", "hephaestus"] {
                assert!(!message.contains(leak), "{message} leaks {leak}");
            }
        }
        // And the layer is named in ggsql's own 1-based draw order.
        assert!(describe(&problems[1..])[0].contains("layer 1"));
    }

    #[test]
    fn the_flags_take_the_boolean_spellings() {
        assert!(writer(&["lossy=true"]).unwrap().lossy);
        assert!(writer(&["lossy=yes"]).unwrap().lossy);
        assert!(writer(&["embed-fonts=1"]).unwrap().embed_fonts);
        assert!(writer(&["embed_fonts=on"]).unwrap().embed_fonts);
        let err = writer(&["lossy=sometimes"]).unwrap_err().to_string();
        assert!(err.contains("'lossy' expects true or false"), "{err}");
    }

    #[test]
    fn images_are_not_an_option_to_ask_for() {
        // Nothing ggsql builds registers an image, so the setting provably
        // cannot take effect and is not offered.
        let err = writer(&["embed-images=true"]).unwrap_err().to_string();
        assert!(err.contains("unknown writer option"), "{err}");
        assert!(!writer(&[]).unwrap().options().embed_images);
    }
}
