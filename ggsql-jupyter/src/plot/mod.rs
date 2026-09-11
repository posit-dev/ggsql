//! Deciding what a plot should become, and producing it.
//!
//! Three things have to agree: where the output is going
//! ([`SessionKind`](crate::display::SessionKind)), what this build and machine
//! can produce ([`PlotBackend::raster`]), and what the frontend asked for.
//! [`choose`] is the single place that reconciles them.

pub mod backend;
pub mod comm;
pub mod quarto;
pub mod sizing;

pub use backend::{PlotBackend, RenderOutcome};
pub use sizing::Canvas;

use crate::display::SessionKind;

/// An output format a plot can be rendered to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Svg,
    Pdf,
    /// Only the plot comm asks for this; nothing else routes to it.
    Tiff,
}

impl Format {
    /// The MIME type this format travels under.
    ///
    /// Copied from Positron's own backend (`plots.py`) so both ends agree.
    pub fn mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Svg => "image/svg+xml",
            Self::Pdf => "application/pdf",
            Self::Tiff => "image/tiff",
        }
    }

    /// Whether the output is text rather than bytes, and so travels in a
    /// display bundle unencoded.
    pub fn is_text(self) -> bool {
        matches!(self, Self::Svg)
    }

    /// Whether producing this needs a GPU adapter.
    pub fn needs_raster(self) -> bool {
        matches!(self, Self::Png | Self::Jpeg | Self::Tiff)
    }

    /// The nearest format this build and this machine can actually produce.
    ///
    /// The Plots pane hard-codes `png`, so without a raster writer the request
    /// degrades to SVG rather than leaving the pane empty. Nothing is hidden:
    /// the reply's `mime_type` names what was actually produced.
    pub fn available(self, backend_raster: bool) -> Self {
        if self.needs_raster() && !backend_raster {
            Self::Svg
        } else {
            self
        }
    }
}

/// What a comm render needs to send its answer back where it came from.
///
/// The render thread treats this as opaque and echoes it with the result. It
/// lives here rather than in `backend` so the thread carries no knowledge of
/// the messaging layer beyond moving this along.
#[derive(Debug, Clone)]
pub struct RenderTicket {
    pub comm_id: String,
    /// The JSON-RPC `id` to answer.
    pub rpc_id: serde_json::Value,
    /// The request being answered, which decides the reply's `mime_type` and
    /// echoed `settings`.
    pub params: comm::RenderParams,
    /// The `comm_msg` this is a reply to, and the socket identities to route
    /// it back along.
    pub parent: crate::message::JupyterMessage,
    pub identities: Vec<Vec<u8>>,
}

/// A plot to render, at a size, in a format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderRequest {
    pub format: Format,
    pub canvas: Canvas,
}

/// Where a rendered plot should be delivered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Delivery {
    /// A static bundle in the cell's output, which is what a notebook, a
    /// Quarto render and plain Jupyter all want.
    Static(RenderRequest),
    /// A `positron.plot` comm, which the kernel opens and then serves render
    /// requests on. No `execute_result` accompanies it — the comm alone creates
    /// the pane entry.
    Comm,
}

/// Decide what to do with a plot.
///
/// The rules, in the order they apply:
///
/// 1. A Positron console session opens a plot comm, whatever this build can
///    render. Positron inlines any output carrying an `image/*` mime, so a
///    static bundle would land in the console *and* leave a fixed-size pane
///    entry; the comm is the only delivery that reaches the pane alone.
/// 2. Quarto is obeyed: `QUARTO_FIG_FORMAT` and friends say exactly what figure
///    the document wants. Standalone sessions only — Quarto never drives a
///    Positron console.
/// 3. Everything else gets a PNG at the frontend's size, falling back to SVG
///    where there is no adapter or no raster writer.
pub fn choose(kind: SessionKind, backend_raster: bool, canvas: Canvas) -> Delivery {
    if kind == SessionKind::PositronConsole {
        return Delivery::Comm;
    }

    if kind == SessionKind::Standalone {
        if let Some(figure) = quarto::from_env() {
            // A document asking for a raster figure on a machine with no
            // adapter still gets a figure, just a vector one.
            return Delivery::Static(RenderRequest {
                format: figure.format.available(backend_raster),
                canvas: figure.canvas,
            });
        }
    }

    Delivery::Static(RenderRequest {
        format: Format::Png.available(backend_raster),
        canvas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANVAS: Canvas = Canvas {
        width: 800,
        height: 600,
        dpi: 96.0,
    };

    #[test]
    fn a_console_session_always_opens_a_plot_comm() {
        // Whatever this build can render: a static bundle would be inlined on
        // its mime alone and arrive twice. The comm answers in SVG where raster
        // is unavailable.
        for backend_raster in [true, false] {
            assert_eq!(
                choose(SessionKind::PositronConsole, backend_raster, CANVAS),
                Delivery::Comm,
                "backend_raster={backend_raster}"
            );
        }
    }

    #[test]
    fn a_format_degrades_only_when_it_needs_a_writer_this_build_lacks() {
        for format in [Format::Png, Format::Jpeg, Format::Tiff] {
            assert_eq!(format.available(false), Format::Svg, "{format:?}");
            assert_eq!(format.available(true), format, "{format:?}");
        }
        // The vector formats need no adapter, so they are never substituted.
        for format in [Format::Svg, Format::Pdf] {
            assert_eq!(format.available(false), format, "{format:?}");
        }
    }

    #[test]
    fn a_notebook_gets_a_static_image() {
        // A plot comm would put the picture in the Plots pane and leave the
        // cell empty, so a notebook needs a bundle whatever else is true.
        let Delivery::Static(request) = choose(SessionKind::PositronNotebook, true, CANVAS) else {
            panic!("a notebook should get a static bundle");
        };
        assert_eq!(request.format, Format::Png);
        assert_eq!(request.canvas, CANVAS);
    }

    #[test]
    fn svg_is_the_fallback_wherever_raster_is_unavailable() {
        for kind in [SessionKind::PositronNotebook, SessionKind::Standalone] {
            let Delivery::Static(request) = choose(kind, false, CANVAS) else {
                panic!("{kind:?} should get a static bundle");
            };
            assert_eq!(request.format, Format::Svg, "{kind:?}");
            // The fallback keeps the size it was asked for; only the format
            // changes.
            assert_eq!(request.canvas, CANVAS, "{kind:?}");
        }
    }

    #[test]
    fn every_format_has_positrons_mime_type() {
        // Copied from `plots.py`; both ends have to agree or a frontend
        // decodes the wrong thing.
        assert_eq!(Format::Png.mime(), "image/png");
        assert_eq!(Format::Jpeg.mime(), "image/jpeg");
        assert_eq!(Format::Svg.mime(), "image/svg+xml");
        assert_eq!(Format::Pdf.mime(), "application/pdf");
        assert_eq!(Format::Tiff.mime(), "image/tiff");
    }

    #[test]
    fn only_svg_travels_as_text() {
        assert!(Format::Svg.is_text());
        for format in [Format::Png, Format::Jpeg, Format::Pdf, Format::Tiff] {
            assert!(!format.is_text(), "{format:?}");
        }
    }

    #[test]
    fn the_vector_formats_need_no_adapter() {
        assert!(!Format::Svg.needs_raster());
        assert!(!Format::Pdf.needs_raster());
        for format in [Format::Png, Format::Jpeg, Format::Tiff] {
            assert!(format.needs_raster(), "{format:?}");
        }
    }
}
