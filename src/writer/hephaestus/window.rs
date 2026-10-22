//! The plot viewer.
//!
//! Not a writer: it produces no output. It lives here because it needs the same
//! composition every writer builds, and because `ggsql-cli` uses only public
//! `ggsql::*` API and has no renderer dependency of its own.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use hephaestus::plot::PlotComposition;
use hephaestus::window::{
    self, Event, EventCtx, Frame, MessageApp, WindowApp, WindowConfig, WindowHandle, WindowSender,
};

use super::canvas::{parse_background, whole_pixels};
use crate::reader::Spec;
use crate::writer::WriterOptions;
use crate::{GgsqlError, Result};

/// Option keys the viewer understands.
///
/// Notably not `units` or `dpi`: a window's size is logical pixels and its
/// resolution belongs to the display it opens on.
const VIEWER_OPTIONS: &[&str] = &["width", "height", "background", "title"];

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
    /// Must be called from the main thread, as the platform event loops require.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the plot cannot be composed, if no
    /// GPU adapter can drive a window, or if the event loop fails.
    pub fn show(&self, spec: &Spec) -> Result<()> {
        let view = super::compose::prepare(spec.plot(), spec.data())?;

        let config = WindowConfig::new(self.title.clone())
            .size(self.width, self.height)
            .background(self.background);

        window::run(config, SpecApp { view })
            .map_err(|e| GgsqlError::WriterError(format!("the plot viewer failed: {e}")))
    }

    /// Prepare a persistent window that shows each plot it is sent.
    ///
    /// Unlike [`show`](Self::show), nothing blocks yet: the returned
    /// [`ReplWindow`] hands out a [`PlotWindowHandle`] for another thread to
    /// send plots to, and its [`run`](ReplWindow::run) blocks on the main
    /// thread until the window closes. That is the shape an interactive
    /// session needs — queries execute on its thread, the window's event loop
    /// keeps the main thread, and a winit event loop may only be created once
    /// per process, so the window is opened once and reused.
    pub fn launch(&self) -> Result<ReplWindow> {
        let handle = WindowHandle::new()
            .map_err(|e| GgsqlError::WriterError(format!("the plot viewer failed: {e}")))?;
        let config = WindowConfig::new(self.title.clone())
            .size(self.width, self.height)
            .background(self.background);
        Ok(ReplWindow {
            handle,
            config,
            closed: Arc::new(AtomicBool::new(false)),
        })
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

/// One composition, redrawn at whatever size the window currently is.
struct SpecApp {
    view: PlotComposition,
}

/// What the session thread sends the window: a new plot to show, or the end
/// of the session.
///
/// The `Spec` crosses rather than a `PlotComposition` so the window thread
/// does the composing — the plot types are single-threaded by design, and the
/// session thread's `Spec` is plain owned data.
enum ReplRequest {
    Plot(Box<Spec>),
    Shutdown,
}

/// A persistent plot window for an interactive session, not yet running.
///
/// Created by [`PlotViewer::launch`]. Hand a [`PlotWindowHandle`] to the
/// session thread with [`plot_handle`](Self::plot_handle), then call
/// [`run`](Self::run) on the main thread.
pub struct ReplWindow {
    handle: WindowHandle<ReplRequest>,
    config: WindowConfig,
    closed: Arc<AtomicBool>,
}

impl ReplWindow {
    /// The handle the session thread uses to reach the window.
    pub fn plot_handle(&self) -> PlotWindowHandle {
        PlotWindowHandle {
            sender: self.handle.sender(),
            closed: Arc::clone(&self.closed),
        }
    }

    /// Open the window and block until it closes.
    ///
    /// Must be called from the main thread, like [`PlotViewer::show`]. The
    /// window starts empty; the first plot arrives when the session thread
    /// sends one.
    pub fn run(self) -> Result<()> {
        let app = ReplApp {
            current: None,
            closed: Arc::clone(&self.closed),
        };
        self.handle
            .run(self.config, app)
            .map_err(|e| GgsqlError::WriterError(format!("the plot viewer failed: {e}")))
    }
}

/// The session thread's way to reach the plot window.
///
/// `Send` and cheap to clone. `is_closed` flips when the user closes the
/// window, so the session can notice and end rather than keep accepting
/// queries nobody will see.
#[derive(Clone)]
pub struct PlotWindowHandle {
    sender: WindowSender<ReplRequest>,
    closed: Arc<AtomicBool>,
}

impl PlotWindowHandle {
    /// Show a plot in the window, replacing whatever it currently shows.
    pub fn show(&self, spec: Spec) -> Result<()> {
        self.sender
            .send(ReplRequest::Plot(Box::new(spec)))
            .map_err(|e| GgsqlError::WriterError(format!("could not reach the plot window: {e}")))
    }

    /// Ask the window to close, e.g. because the session ended.
    ///
    /// Ignores a dead channel: the window being gone already is the goal.
    pub fn shutdown(&self) {
        let _ = self.sender.send(ReplRequest::Shutdown);
    }

    /// Whether the user has closed the window.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }
}

/// The window-side app for an interactive session: shows the latest plot it
/// was sent, records the window closing so the session thread can notice.
struct ReplApp {
    current: Option<PlotComposition>,
    closed: Arc<AtomicBool>,
}

impl WindowApp for ReplApp {
    fn draw(&mut self, frame: &mut Frame<'_>) {
        let Some(view) = self.current.as_mut() else {
            // No plot yet: the frame stays at the background color.
            return;
        };
        let (scene, size, dpi) = frame.parts();
        view.render(scene, size, dpi);
    }

    fn event(&mut self, ctx: &mut EventCtx<'_>, event: Event) {
        if matches!(event, Event::CloseRequested) {
            self.closed.store(true, Ordering::Relaxed);
            ctx.exit();
        }
    }
}

impl MessageApp<ReplRequest> for ReplApp {
    fn message(&mut self, ctx: &mut EventCtx<'_>, message: ReplRequest) {
        match message {
            // hephaestus requests a redraw after every message, so a new
            // composition is on screen at the next frame with nothing more
            // to do here.
            ReplRequest::Plot(spec) => match super::compose::prepare(spec.plot(), spec.data()) {
                Ok(view) => self.current = Some(view),
                Err(e) => eprintln!("Failed to compose the plot: {e}"),
            },
            ReplRequest::Shutdown => ctx.exit(),
        }
    }
}

impl WindowApp for SpecApp {
    fn draw(&mut self, frame: &mut Frame<'_>) {
        // The frame reports its own size and dpi, which is what makes a resize
        // a re-layout rather than a rescale.
        let (scene, size, dpi) = frame.parts();
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
