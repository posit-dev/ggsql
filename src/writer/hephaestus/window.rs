//! The plot viewer.
//!
//! Not a writer: it produces no output. It lives here because it needs the same
//! composition every writer builds, and because `ggsql-cli` uses only public
//! `ggsql::*` API and has no renderer dependency of its own.

use std::cell::Cell;
use std::rc::Rc;

use hephaestus::plot::PlotComposition;
use hephaestus::window::{self, Event, EventCtx, Frame, WindowApp, WindowConfig};

use super::canvas::{parse_background, whole_pixels};
use crate::reader::Spec;
use crate::writer::WriterOptions;
use crate::{GgsqlError, Result};

/// Option keys the viewer understands.
///
/// Notably not `units` or `dpi`: a window's size is logical pixels and its
/// resolution belongs to the display it opens on.
pub const VIEWER_OPTIONS: &[&str] = &["width", "height", "background", "title"];

/// Default window size, matching the renderer's own.
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;

/// Shows a ggsql plot in a native window.
///
/// Resizing needs no code: the composition re-solves its layout at the size and
/// resolution the window reports each frame, so the plot re-lays-out rather
/// than stretching.
///
/// [`PlotViewer::from_options`] takes:
///
/// | Option | Value | Default |
/// | --- | --- | --- |
/// | `width` | Window width in logical pixels | 800 |
/// | `height` | Window height in logical pixels | 600 |
/// | `background` | Any CSS color, e.g. `white`, `#ff0000`, `transparent` | `white` |
/// | `title` | Window title | `ggsql` |
///
/// Not a [`Writer`](crate::writer::Writer): it returns no output, blocks, and
/// must run on the main thread. `from_options` plus `show` gives the same
/// option ergonomics without claiming otherwise.
///
/// Requires a working GPU adapter, like the raster writers.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotViewer {
    width: u32,
    height: u32,
    background: super::Color,
    title: String,
}

impl PlotViewer {
    /// A viewer for a window of the given size in logical pixels.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Self::default()
        }
    }

    /// Set the window title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the color the window is cleared to before each frame.
    pub fn background(mut self, color: super::Color) -> Self {
        self.background = color;
        self
    }

    /// Build a viewer from free-form key–value options.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` for an unknown key or an unusable
    /// value. `units` and `dpi` are rejected with a reason rather than as typos.
    pub fn from_options(options: &WriterOptions) -> Result<Self> {
        for (key, why) in [
            ("units", "a window is sized in logical pixels"),
            (
                "dpi",
                "a window's resolution belongs to the display it opens on",
            ),
        ] {
            if options.get(key).is_some() {
                return Err(GgsqlError::WriterError(format!(
                    "the plot viewer takes no '{key}' option: {why}. Render to a file if you \
                     need to choose one"
                )));
            }
        }
        options.reject_unknown(VIEWER_OPTIONS)?;

        let mut viewer = Self::default();
        if let Some(width) = options.number("width")? {
            viewer.width = whole_pixels(width, "width")?;
        }
        if let Some(height) = options.number("height")? {
            viewer.height = whole_pixels(height, "height")?;
        }
        if let Some(raw) = options.get("background") {
            viewer.background = parse_background(raw)?;
        }
        if let Some(title) = options.get("title") {
            viewer.title = title.to_string();
        }
        Ok(viewer)
    }

    /// Show the plot and **block until the window closes.**
    ///
    /// Returns the frame the plot was last drawn in — logical size and dpi —
    /// as it was when the window closed, so a caller rendering the same plot
    /// to a file can match what the user last saw. Must be called from the
    /// main thread, as the platform event loops require.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the plot cannot be composed, if no
    /// GPU adapter can drive a window, or if the event loop fails.
    pub fn show(&self, spec: &Spec) -> Result<ClosingFrame> {
        let view = super::compose::prepare(spec.plot(), spec.data())?;

        let config = WindowConfig::new(self.title.clone())
            .size(self.width, self.height)
            .background(self.background);

        // The last frame the plot was drawn at, tracked across resizes so it
        // survives `window::run` consuming the app.
        let closing = Rc::new(Cell::new(ClosingFrame {
            width: self.width,
            height: self.height,
            dpi: 96.0,
        }));
        let app = SpecApp {
            view,
            closing: Rc::clone(&closing),
        };

        window::run(config, app)
            .map_err(|e| GgsqlError::WriterError(format!("the plot viewer failed: {e}")))?;

        Ok(closing.get())
    }
}

impl Default for PlotViewer {
    fn default() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            background: super::rgba(1.0, 1.0, 1.0, 1.0),
            title: "ggsql".to_string(),
        }
    }
}

/// The frame the plot was last drawn in when the window closed.
///
/// `dpi` is the display's (`96 × scale factor` — 192 on a Retina display),
/// which a file render needs to reproduce the window's text and line sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosingFrame {
    pub width: u32,
    pub height: u32,
    pub dpi: f64,
}

/// One composition, redrawn at whatever size the window currently is.
struct SpecApp {
    view: PlotComposition,
    closing: Rc<Cell<ClosingFrame>>,
}

impl WindowApp for SpecApp {
    fn draw(&mut self, frame: &mut Frame<'_>) {
        // The frame reports its own size and dpi, which is what makes a resize
        // a re-layout rather than a rescale.
        let (scene, size, dpi) = frame.parts();
        self.closing.set(ClosingFrame {
            width: size.width.round() as u32,
            height: size.height.round() as u32,
            dpi,
        });
        self.view.render(scene, size, dpi);
    }

    fn event(&mut self, ctx: &mut EventCtx<'_>, event: Event) {
        // The window stays open until the app says otherwise, so closing it is
        // the one event that needs handling.
        if matches!(event, Event::CloseRequested) {
            ctx.exit();
        }
    }
}

#[cfg(test)]
mod option_tests {
    use super::*;

    fn viewer(pairs: &[&str]) -> Result<PlotViewer> {
        PlotViewer::from_options(&WriterOptions::parse(pairs)?)
    }

    #[test]
    fn no_options_gives_the_defaults() {
        let default = PlotViewer::default();
        assert_eq!(viewer(&[]).unwrap(), default);
        assert_eq!((default.width, default.height), (800, 600));
        assert_eq!(default.title, "ggsql");
        assert_eq!(default.background.components, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn size_title_and_background_are_taken_as_given() {
        let v = viewer(&["width=1280", "height=720", "title=My plot"]).unwrap();
        assert_eq!((v.width, v.height), (1280, 720));
        assert_eq!(v.title, "My plot");
        assert_eq!(
            viewer(&["background=#ff0000"])
                .unwrap()
                .background
                .components,
            [1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            viewer(&["background=none"]).unwrap().background.components[3],
            0.0
        );
    }

    #[test]
    fn a_physical_size_is_refused_with_a_reason() {
        // Accepting a `dpi` the viewer ignores is the silent failure
        // `reject_unknown` exists to prevent, so these say why.
        for (option, expected) in [
            ("units=in", "sized in logical pixels"),
            ("dpi=300", "belongs to the display"),
        ] {
            let err = viewer(&[option]).unwrap_err().to_string();
            assert!(err.contains(expected), "{option}: {err}");
            assert!(err.contains("the plot viewer takes no"), "{option}: {err}");
        }
    }

    #[test]
    fn other_bad_values_are_reported_per_option() {
        let cases = [
            ("width=0", "'width' resolves to 0 px"),
            ("height=abc", "'height' expects a number"),
            ("background=nope", "'background' expects a CSS color"),
        ];
        for (option, expected) in cases {
            let err = viewer(&[option]).unwrap_err().to_string();
            assert!(err.contains(expected), "{option}: {err}");
        }
        let err = viewer(&["compression=fast"]).unwrap_err().to_string();
        assert!(err.contains("unknown writer option 'compression'"), "{err}");
        assert!(err.contains("width, height, background, title"), "{err}");
    }
}
