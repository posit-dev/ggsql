//! The canvas configuration every renderer-backed writer carries.
//!
//! Raster and vector output both need concrete dimensions and a resolution, so
//! the size, DPI and background live here rather than in each writer. A writer
//! adds only its own format's keys: a JPEG quality, a TIFF compression.

use hephaestus::color::{rgba, Color};
use hephaestus::geometry::Size;

use super::scales::parse_color;
use crate::writer::WriterOptions;
use crate::{GgsqlError, Result};

/// Default canvas width in pixels.
pub(super) const DEFAULT_WIDTH: u32 = 1500;
/// Default canvas height in pixels.
pub(super) const DEFAULT_HEIGHT: u32 = 1000;
/// Default resolution. DPI converts the theme's physical sizes (text, stroke
/// widths, spacing — all in points) to pixels, so it sets how large the chrome
/// is relative to the canvas as well as the print size of a physical figure.
pub(super) const DEFAULT_DPI: f64 = 300.0;

/// Largest canvas dimension accepted, in pixels. Far beyond any real figure, but
/// small enough that a slipped unit conversion fails with a message instead of
/// exhausting GPU memory.
const MAX_DIMENSION: f64 = 32_768.0;

/// Option keys every renderer-backed writer understands.
///
/// Concatenated ahead of a writer's own keys when rejecting unknown options, so
/// the shared ones lead the "supported options" list in the error.
pub const CANVAS_OPTIONS: &[&str] = &["width", "height", "units", "dpi", "background"];

/// The canvas keys that describe a *size* rather than an appearance.
///
/// A writer whose canvas is only a hint tells "no size given" from "a size that
/// happens to equal the default" by these keys. `dpi` is its own hint and so is
/// not among them.
#[cfg(feature = "hep")]
pub const CANVAS_SIZE_OPTIONS: &[&str] = &["width", "height", "units"];

/// Units a `width` / `height` option may be given in.
const UNITS: &[&str] = &["px", "in", "cm", "mm", "pt"];

/// Size, resolution and background for one rendered figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub dpi: f64,
    pub background: Color,
    /// Whether the dimensions were given in a physical unit rather than pixels.
    ///
    /// Only the vector backends consult it: a file asked for in inches declares
    /// a physical size so it prints at that size; one in pixels stays in pixels.
    pub physical: bool,
}

impl Canvas {
    /// A canvas of the given pixel dimensions and DPI, on white.
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        Self {
            width,
            height,
            dpi,
            background: rgba(1.0, 1.0, 1.0, 1.0),
            physical: false,
        }
    }

    /// Set the background painted before anything is drawn.
    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Parse the shared keys, rejecting anything outside them or `extra` first.
    ///
    /// `extra` is the calling writer's own option names. Rejection happens
    /// before any value is read, so a mistyped key is reported rather than
    /// silently ignored.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` for an unknown key, an unusable value,
    /// or a dimension outside the renderable range.
    pub fn from_options(options: &WriterOptions, extra: &[&str]) -> Result<Self> {
        let known: Vec<&str> = CANVAS_OPTIONS.iter().chain(extra).copied().collect();
        options.reject_unknown(&known)?;

        let dpi = match options.number("dpi")? {
            Some(dpi) if dpi > 0.0 => dpi,
            Some(dpi) => {
                return Err(GgsqlError::WriterError(format!(
                    "writer option 'dpi' expects a positive number, got '{dpi}'"
                )))
            }
            None => DEFAULT_DPI,
        };
        // `units` interprets the dimensions the caller supplies; the defaults are
        // pixel counts, so they stand whatever the unit is.
        let units = options.one_of("units", UNITS)?.unwrap_or("px");
        let width = match options.number("width")? {
            Some(width) => to_pixels(width, units, dpi, "width")?,
            None => DEFAULT_WIDTH,
        };
        let height = match options.number("height")? {
            Some(height) => to_pixels(height, units, dpi, "height")?,
            None => DEFAULT_HEIGHT,
        };

        let mut canvas = Self::new(width, height, dpi);
        canvas.physical = units != "px";
        if let Some(raw) = options.get("background") {
            canvas = canvas.background(parse_background(raw)?);
        }
        Ok(canvas)
    }

    /// The canvas as a hephaestus size, for `PlotComposition::render`.
    pub fn size(&self) -> Size {
        Size::new(self.width as f64, self.height as f64)
    }

    /// The resolution to record in an output that can carry one.
    ///
    /// A file that declares nothing is read as 72 dpi by whatever opens it, so
    /// an image rendered at a higher resolution would claim the wrong physical
    /// size.
    pub fn dpi_hint(&self) -> Option<f64> {
        Some(self.dpi)
    }

    /// The background as a vector backend wants it: `None` when fully
    /// transparent.
    ///
    /// A rasteriser is always handed a colour to clear with; a vector backend
    /// takes `None` to mean "emit no background element", which is what a
    /// transparent canvas should become rather than a transparent-black rect.
    pub fn vector_background(&self) -> Option<Color> {
        if self.background.components[3] <= 0.0 {
            None
        } else {
            Some(self.background)
        }
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_DPI)
    }
}

/// Read a `background` option's value as a color.
///
/// A free function rather than a `Canvas` method because the plot viewer takes
/// a background without taking a canvas.
///
/// # Errors
///
/// Returns `GgsqlError::WriterError` if the value is not a color.
pub(super) fn parse_background(raw: &str) -> Result<Color> {
    // `none` is a familiar spelling of a transparent canvas that CSS itself
    // doesn't accept as a color.
    match raw.trim().to_lowercase().as_str() {
        "none" => Ok(rgba(0.0, 0.0, 0.0, 0.0)),
        _ => parse_color(raw).ok_or_else(|| {
            GgsqlError::WriterError(format!(
                "writer option 'background' expects a CSS color, got '{raw}'"
            ))
        }),
    }
}

/// Convert a canvas dimension given in `units` to whole pixels at `dpi`.
///
/// A physical unit goes through inches, so the same figure grows with DPI; `px`
/// is already the canvas unit, where DPI only scales the chrome.
fn to_pixels(value: f64, units: &str, dpi: f64, key: &str) -> Result<u32> {
    let per_inch = match units {
        "in" => 1.0,
        "cm" => 2.54,
        "mm" => 25.4,
        "pt" => 72.0,
        _ => return whole_pixels(value, key),
    };
    whole_pixels(value / per_inch * dpi, key)
}

/// Round a pixel count and reject one outside the renderable range.
pub(super) fn whole_pixels(pixels: f64, key: &str) -> Result<u32> {
    let rounded = pixels.round();
    if !(1.0..=MAX_DIMENSION).contains(&rounded) {
        return Err(GgsqlError::WriterError(format!(
            "writer option '{key}' resolves to {rounded} px, outside the supported range 1–{MAX_DIMENSION} px"
        )));
    }
    Ok(rounded as u32)
}

/// Test-only access to a writer's canvas.
///
/// Implemented by every renderer-backed writer so the shared option behaviour
/// can be asserted generically instead of once per format.
#[cfg(all(
    test,
    any(
        feature = "png",
        feature = "jpeg",
        feature = "tiff",
        feature = "webp",
        feature = "svg",
        feature = "pdf",
        feature = "hep"
    )
))]
pub(super) trait Canvased {
    fn canvas(&self) -> &Canvas;
}

/// Assert the five shared canvas options behave identically for `W`.
///
/// Parsed in one place, so asserted in one place; a writer's own tests cover
/// only the keys its format adds. Catches a writer that parses a canvas key
/// itself or forgets to pass its own keys to [`Canvas::from_options`].
///
/// Transparency is not covered here — see [`assert_transparent_background`].
#[cfg(all(
    test,
    any(
        feature = "png",
        feature = "jpeg",
        feature = "tiff",
        feature = "webp",
        feature = "svg",
        feature = "pdf",
        feature = "hep"
    )
))]
pub(super) fn assert_canvas_semantics<W: crate::writer::Writer + Canvased + std::fmt::Debug>() {
    let build = |pairs: &[&str]| -> Result<W> { W::from_options(&WriterOptions::parse(pairs)?) };
    let dims = |pairs: &[&str]| -> (u32, u32, f64) {
        let writer = build(pairs).unwrap();
        let c = writer.canvas();
        (c.width, c.height, c.dpi)
    };

    // No options: the documented defaults, on an opaque white canvas.
    assert_eq!(dims(&[]), (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_DPI));
    let white = *build(&[]).unwrap().canvas();
    assert_eq!(white.background.components, [1.0, 1.0, 1.0, 1.0]);
    assert!(!white.physical, "a pixel canvas is not a physical one");

    // A pixel canvas is taken verbatim, and DPI only scales the chrome on it.
    assert_eq!(
        dims(&["width=1600", "height=1200"]),
        (1600, 1200, DEFAULT_DPI)
    );
    assert_eq!(
        dims(&["width=800", "units=px", "dpi=72"]),
        (800, DEFAULT_HEIGHT, 72.0)
    );

    // A physical canvas goes through inches, so it grows with DPI.
    assert_eq!(
        dims(&["width=8", "height=6", "units=in", "dpi=100"]),
        (800, 600, 100.0)
    );
    // 2.54 cm = 1 in; 25.4 mm = 1 in; 72 pt = 1 in.
    assert_eq!(dims(&["width=2.54", "units=cm", "dpi=96"]).0, 96);
    assert_eq!(dims(&["width=25.4", "units=mm", "dpi=96"]).0, 96);
    assert_eq!(dims(&["width=72", "units=pt", "dpi=96"]).0, 96);
    // An unset dimension stays a pixel count even when the caller works in inches.
    assert_eq!(dims(&["width=5", "units=in", "dpi=200"]).1, DEFAULT_HEIGHT);
    assert!(
        build(&["width=5", "units=in"]).unwrap().canvas().physical,
        "inches are a physical unit"
    );

    // An opaque CSS color, in the spellings a user reaches for.
    let red = *build(&["background=#ff0000"]).unwrap().canvas();
    assert_eq!(red.background.components, [1.0, 0.0, 0.0, 1.0]);
    assert!(build(&["background=rgb(0, 0, 255)"]).is_ok());
    assert!(build(&["background=white"]).is_ok());

    // Every bad value names the option that carries it.
    let cases = [
        ("units=furlongs", "'units' expects"),
        ("dpi=0", "'dpi' expects a positive number"),
        ("dpi=high", "'dpi' expects a number"),
        ("width=0", "'width' resolves to 0 px"),
        ("width=-4", "'width' resolves to -4 px"),
        ("height=1e9", "'height' resolves to"),
        ("background=nope", "'background' expects a CSS color"),
    ];
    for (option, expected) in cases {
        let err = build(&[option]).unwrap_err().to_string();
        assert!(err.contains(expected), "{option}: {err}");
    }

    // And an unknown key is reported rather than ignored, with the shared keys
    // leading the list so the nearest miss is the first thing read.
    let err = build(&["with=1600"]).unwrap_err().to_string();
    assert!(err.contains("unknown writer option 'with'"), "{err}");
    assert!(err.contains("supported options: width, height"), "{err}");
}

/// Assert `W` accepts a transparent canvas, in both spellings.
///
/// Separate from [`assert_canvas_semantics`] because a format with no alpha
/// channel refuses one instead — see `JpegWriter`.
// Every writer but `jpeg`, which refuses transparency — so a jpeg-only build
// is the one config where this has no caller.
#[cfg(all(
    test,
    any(
        feature = "png",
        feature = "tiff",
        feature = "webp",
        feature = "svg",
        feature = "pdf",
        feature = "hep"
    )
))]
pub(super) fn assert_transparent_background<W: crate::writer::Writer + Canvased>() {
    for spelling in ["background=transparent", "background=none"] {
        let options = WriterOptions::parse([spelling]).unwrap();
        let writer = W::from_options(&options).unwrap();
        assert_eq!(
            writer.canvas().background.components[3],
            0.0,
            "{spelling} should be fully transparent"
        );
    }
}
