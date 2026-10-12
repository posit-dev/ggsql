//! The render thread.
//!
//! Rendering does not happen on the message loop: `kernel.rs` awaits
//! `handle_shell_message` inline in its `select!`, so anything blocking there
//! stalls the heartbeat, the control channel and the SIGINT handler alike.
//!
//! # What the cold start actually costs
//!
//! Measured on a release build, Apple GPU, vello-hybrid:
//!
//! | | warm | cold (first process after a build) |
//! | --- | --- | --- |
//! | `RasterRenderer::new()` | 14 ms | ~185 ms |
//! | **first render** | **~85 ms** | **~1.35 s** |
//! | later renders, 3-point plot at 1200×800 | 5 ms | 5 ms |
//! | later renders, 50k points | ~200 ms | ~200 ms |
//!
//! The first render, not the renderer's construction, is the expensive part,
//! and most of it is parley/fontique loading system faces — a per-process cost.
//! Rendering an SVG first does the same text work with no GPU and drops the
//! first raster render from ~85 ms to ~20 ms.
//!
//! So the thread builds the renderer and renders a throwaway frame at startup,
//! before anyone is waiting. The renderer is `Send` but not `Sync`, which suits
//! a thread that owns it and never shares it.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};

use anyhow::{anyhow, Result};
use ggsql::reader::ResolvedPlot;

use super::{Format, RenderRequest, RenderTicket};

/// How long to wait for a GPU adapter before deciding there isn't one.
///
/// Eager, because a lazy probe would leave the first plot unable to choose a
/// path — but a driver silent for ten seconds is not one to render through.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long the message loop will wait for a pre-render before opening the
/// comm without one.
///
/// [`PlotBackend::render_stored`] blocks the loop and can queue behind the
/// pane's own jobs, so the wait is bounded — a stalled heartbeat costs more
/// than a missing pre-render.
const PRE_RENDER_BUDGET: std::time::Duration = std::time::Duration::from_millis(500);

/// A finished render, on its way back to the message loop.
pub struct RenderOutcome {
    pub ticket: Box<RenderTicket>,
    pub result: Result<Vec<u8>>,
}

/// Work for the render thread.
enum Job {
    /// Keep `spec` so the plot can be re-drawn at a new size without re-running
    /// the query. Its `DataFrame`s stay here rather than on the async task.
    Store {
        comm_id: String,
        spec: Box<ResolvedPlot>,
    },
    /// Forget a stored plot, because its comm closed or it was evicted.
    Forget { comm_id: String },
    /// Re-render a stored plot and answer `reply` directly.
    ///
    /// The blocking counterpart to `Render`, used once per plot for the
    /// pre-render that rides on `comm_open`.
    RenderStored {
        comm_id: String,
        request: RenderRequest,
        reply: Sender<Result<Vec<u8>>>,
    },
    /// Re-render a stored plot and report the result asynchronously.
    Render {
        comm_id: String,
        request: RenderRequest,
        // Boxed to keep the variants a similar size: a ticket carries a whole
        // Jupyter message, and every job would otherwise be that large.
        ticket: Box<RenderTicket>,
    },
    /// Render a plot we were handed and will not keep, answering `reply`
    /// directly. The one-shot path, for a static output bundle.
    RenderOnce {
        spec: Box<ResolvedPlot>,
        request: RenderRequest,
        reply: Sender<Result<Vec<u8>>>,
    },
    /// Stop the thread.
    Shutdown,
}

/// A handle to the render thread.
pub struct PlotBackend {
    jobs: Sender<Job>,
    /// Whether a GPU adapter was found at startup, and so whether the raster
    /// formats are available. Decided once. `false` until [`Self::finish_probe`]
    /// answers, which is the safe direction to be wrong in — it means SVG.
    raster: bool,
    /// The probe's answer, until it is collected. `None` once collected, and
    /// for a backend that never probes.
    probe: Option<Receiver<bool>>,
}

impl PlotBackend {
    /// Start the render thread and begin probing for a GPU adapter.
    ///
    /// Returns immediately; [`Self::finish_probe`] collects the answer. Eager,
    /// so the first plot can choose a path, but non-blocking so it does not
    /// delay the `starting` status.
    pub fn spawn(outcomes: tokio::sync::mpsc::UnboundedSender<RenderOutcome>) -> Self {
        Self::start(true, Some(outcomes))
    }

    /// Wait for the GPU probe, so every later [`Self::raster`] is answered.
    ///
    /// Blocks for up to [`PROBE_TIMEOUT`]. Call it *after* announcing
    /// `starting`: a wedged driver would otherwise leave the kernel silent on a
    /// bound socket, which a supervisor reads as a failed launch.
    pub fn finish_probe(&mut self) {
        let Some(probe) = self.probe.take() else {
            return;
        };

        // A thread that died before probing, or a driver that never answered,
        // both mean the same thing to us.
        self.raster = probe.recv_timeout(PROBE_TIMEOUT).unwrap_or_else(|_| {
            tracing::warn!("GPU probe did not finish within {PROBE_TIMEOUT:?}");
            false
        });

        if self.raster {
            tracing::info!("GPU adapter available; raster plot formats enabled");
        } else if cfg!(feature = "raster-plots") {
            tracing::info!("no GPU adapter; plots will render as SVG");
        } else {
            // Distinct from the line above: the machine may well have an
            // adapter, and saying otherwise misdirects the debugging.
            tracing::info!("built without the raster-plots feature; plots will render as SVG");
        }
    }

    /// A backend that never builds a GPU renderer.
    ///
    /// For tests: the probe is the one slow, machine-dependent part of startup,
    /// and the SVG path it leaves is the one that always works.
    #[cfg(test)]
    pub fn without_raster() -> Self {
        Self::start(false, None)
    }

    fn start(
        allow_raster: bool,
        outcomes: Option<tokio::sync::mpsc::UnboundedSender<RenderOutcome>>,
    ) -> Self {
        let (jobs, inbox) = mpsc::channel();
        let (probed, probe_result) = mpsc::channel();

        std::thread::Builder::new()
            .name("ggsql-render".to_string())
            .spawn(move || render_loop(inbox, probed, allow_raster, outcomes))
            .expect("failed to spawn the render thread");

        Self {
            jobs,
            raster: false,
            // Nothing to wait for in a build or a test that never probes.
            probe: allow_raster.then_some(probe_result),
        }
    }

    /// Whether this build and this machine can produce raster output.
    pub fn raster(&self) -> bool {
        self.raster
    }

    /// Render a plot once, blocking until the thread answers.
    ///
    /// Blocking, unlike [`Self::request_render`]: the static path runs once per
    /// execution and its output has to be ordered between `execute_input` and
    /// `execute_reply`, which a few milliseconds buys cheaply.
    ///
    /// # Errors
    ///
    /// Returns an error if the render thread has stopped, or if the render
    /// itself failed.
    pub fn render_once(&self, spec: Box<ResolvedPlot>, request: RenderRequest) -> Result<Vec<u8>> {
        let (reply, answer) = mpsc::channel();
        self.jobs
            .send(Job::RenderOnce {
                spec,
                request,
                reply,
            })
            .map_err(|_| anyhow!("the render thread has stopped"))?;
        answer
            .recv()
            .map_err(|_| anyhow!("the render thread stopped while rendering"))?
    }

    /// Keep a plot so its comm can re-render it at any size.
    pub fn store(&self, comm_id: String, spec: Box<ResolvedPlot>) {
        let _ = self.jobs.send(Job::Store { comm_id, spec });
    }

    /// Forget a stored plot.
    pub fn forget(&self, comm_id: &str) {
        let _ = self.jobs.send(Job::Forget {
            comm_id: comm_id.to_string(),
        });
    }

    /// Render a stored plot, blocking until the thread answers or
    /// [`PRE_RENDER_BUDGET`] runs out.
    ///
    /// The pre-render riding on `comm_open`, and the one blocking call made
    /// from the message loop. The pane's earlier jobs can queue ahead of it, so
    /// the wait is capped: giving up costs a comm without a pre-render, waiting
    /// costs the heartbeat. The pane's own renders go through
    /// [`Self::request_render`].
    ///
    /// # Errors
    ///
    /// Returns an error if the thread has stopped, the plot is unknown, the
    /// render failed, or the budget expired.
    pub fn render_stored(&self, comm_id: &str, request: RenderRequest) -> Result<Vec<u8>> {
        let (reply, answer) = mpsc::channel();
        self.jobs
            .send(Job::RenderStored {
                comm_id: comm_id.to_string(),
                request,
                reply,
            })
            .map_err(|_| anyhow!("the render thread has stopped"))?;
        // The job stays on the queue and its result is dropped, which the
        // thread treats as a caller that gave up — not as an error.
        answer
            .recv_timeout(PRE_RENDER_BUDGET)
            .map_err(|_| anyhow!("the render did not finish within {PRE_RENDER_BUDGET:?}"))?
    }

    /// Ask for a stored plot to be re-rendered, and return immediately.
    ///
    /// Why the thread exists. The reply arrives later on the outcome channel,
    /// so a render — up to a few hundred milliseconds, and asked for once per
    /// frame while a pane is dragged — never blocks the message loop.
    ///
    /// # Errors
    ///
    /// Returns an error if the render thread has stopped, in which case no
    /// outcome will ever arrive — so the caller has to answer the request
    /// itself rather than wait for one.
    pub fn request_render(
        &self,
        comm_id: &str,
        request: RenderRequest,
        ticket: RenderTicket,
    ) -> Result<()> {
        self.jobs
            .send(Job::Render {
                comm_id: comm_id.to_string(),
                request,
                ticket: Box::new(ticket),
            })
            .map_err(|_| anyhow!("the render thread has stopped"))
    }
}

impl Drop for PlotBackend {
    fn drop(&mut self) {
        let _ = self.jobs.send(Job::Shutdown);
    }
}

/// The render thread's body: probe once, then serve jobs until told to stop.
fn render_loop(
    inbox: Receiver<Job>,
    probed: Sender<bool>,
    allow_raster: bool,
    outcomes: Option<tokio::sync::mpsc::UnboundedSender<RenderOutcome>>,
) {
    // One renderer for the whole session; it handles a changing frame size
    // internally, so it serves every render whatever size is asked for.
    let mut renderer = if allow_raster {
        raster_renderer()
    } else {
        None
    };
    let _ = probed.send(renderer.is_some());

    // Pay the first-render cost now, while nobody is waiting on it. See the
    // module docs: it is mostly font loading, and it is per process.
    warm_up(renderer.as_mut());

    // The retained plots. They live here rather than beside the comm state so
    // the post-stat `DataFrame`s stay off the async task entirely.
    let mut stored: HashMap<String, Box<ResolvedPlot>> = HashMap::new();

    while let Ok(job) = inbox.recv() {
        match job {
            Job::Shutdown => break,
            Job::Store { comm_id, spec } => {
                stored.insert(comm_id, spec);
            }
            Job::Forget { comm_id } => {
                stored.remove(&comm_id);
            }
            Job::RenderOnce {
                spec,
                request,
                reply,
            } => {
                let result = render_one(&spec, &request, renderer.as_mut(), &one_shot_namespace());
                // A caller that gave up before we finished is not an error.
                let _ = reply.send(result);
            }
            Job::RenderStored {
                comm_id,
                request,
                reply,
            } => {
                let result = match stored.get(&comm_id) {
                    Some(spec) => {
                        render_one(spec, &request, renderer.as_mut(), &comm_namespace(&comm_id))
                    }
                    None => Err(anyhow!("this plot is no longer available")),
                };
                let _ = reply.send(result);
            }
            Job::Render {
                comm_id,
                request,
                ticket,
            } => {
                let result = match stored.get(&comm_id) {
                    Some(spec) => {
                        render_one(spec, &request, renderer.as_mut(), &comm_namespace(&comm_id))
                    }
                    // The comm closed, or the plot was evicted, between the
                    // request arriving and us reaching it.
                    None => Err(anyhow!("this plot is no longer available")),
                };
                if let Some(outcomes) = &outcomes {
                    let _ = outcomes.send(RenderOutcome { ticket, result });
                }
            }
        }
    }
}

/// Render a throwaway frame so the first real plot does not pay for the
/// process's font enumeration and pipeline setup.
///
/// Uses a throwaway in-memory database rather than the session's reader, which
/// would materialise ggsql's internal views in the user's session. Rendered
/// tiny, since the cost is not proportional to area, and best-effort: a failed
/// warm-up costs a slower first plot and nothing else.
fn warm_up(renderer: Option<&mut Renderer>) {
    const QUERY: &str = "SELECT 1 AS x, 1 AS y VISUALISE x AS x, y AS y DRAW point";

    let started = std::time::Instant::now();
    let spec = match ggsql::reader::connection::reader_from_uri("duckdb://memory")
        .and_then(|reader| reader.execute(QUERY))
    {
        Ok(spec) => match spec.into_plot() {
            Some(plot) => plot,
            None => {
                tracing::debug!("renderer warm-up skipped: not a plot");
                return;
            }
        },
        Err(e) => {
            tracing::debug!("renderer warm-up skipped: {e}");
            return;
        }
    };

    let request = RenderRequest {
        // SVG warms the text stack with or without a GPU, and that is most of
        // the cost; a raster pass on top would save only ~15 ms more.
        format: Format::Svg,
        canvas: super::Canvas {
            width: 64,
            height: 64,
            dpi: 96.0,
        },
    };
    match render_one(&spec, &request, renderer, "warmup-") {
        Ok(_) => tracing::debug!("renderer warmed up in {:?}", started.elapsed()),
        Err(e) => tracing::debug!("renderer warm-up failed: {e}"),
    }
}

/// Build the GPU renderer, or report that there isn't one.
#[cfg(feature = "raster-plots")]
fn raster_renderer() -> Option<ggsql::writer::RasterRenderer> {
    match ggsql::writer::RasterRenderer::new() {
        Ok(renderer) => Some(renderer),
        Err(e) => {
            tracing::info!("no GPU renderer: {e}");
            None
        }
    }
}

/// Without the feature there is nothing to build, and the probe is a constant.
#[cfg(not(feature = "raster-plots"))]
fn raster_renderer() -> Option<Never> {
    None
}

/// Stands in for the renderer in a build that has none, so `render_one` keeps
/// one signature. Uninhabited, so the raster arms are unreachable rather than
/// merely unused.
#[cfg(not(feature = "raster-plots"))]
pub enum Never {}

#[cfg(feature = "raster-plots")]
type Renderer = ggsql::writer::RasterRenderer;
#[cfg(not(feature = "raster-plots"))]
type Renderer = Never;

/// The id namespace for a plot rendered once and not kept.
///
/// SVG element ids are per-document counters (`c0`, `lg1`), so two plots in one
/// notebook both define `#lg1` and the second's `url(#lg1)` resolves to the
/// first's gradient — a silently wrong figure, in every browser. A namespace
/// per plot is what keeps them apart, since every cell's output shares one DOM.
///
/// A uuid rather than a counter, because a counter restarts with the kernel
/// while the notebook still shows the outputs it already handed out. The cost
/// is that re-running a cell rewrites the ids of an otherwise identical plot.
fn one_shot_namespace() -> String {
    format!("p{}-", uuid::Uuid::new_v4())
}

/// The id namespace for a plot the render thread keeps, from its comm id.
///
/// One namespace for the plot's whole life, so resizing redraws it to the same
/// ids. The comm id is already a uuid, and already unique per plot.
fn comm_namespace(comm_id: &str) -> String {
    format!("p{comm_id}-")
}

/// Render one plot in whichever format was asked for.
///
/// `id_namespace` prefixes the ids the SVG writer generates; the other formats
/// have no such thing and ignore it.
fn render_one(
    spec: &ResolvedPlot,
    request: &RenderRequest,
    renderer: Option<&mut Renderer>,
    id_namespace: &str,
) -> Result<Vec<u8>> {
    let canvas = request.canvas;
    match request.format {
        Format::Svg => {
            let writer = ggsql::writer::SvgWriter::new(canvas.width, canvas.height, canvas.dpi)
                .id_prefix(id_namespace);
            let (svg, warnings) = writer.render_reporting(spec)?;
            report(&warnings, "svg");
            Ok(svg.into_bytes())
        }
        Format::Pdf => {
            let writer = ggsql::writer::PdfWriter::new(canvas.width, canvas.height, canvas.dpi);
            let (pdf, warnings) = writer.render_reporting(spec)?;
            report(&warnings, "pdf");
            Ok(pdf)
        }
        #[cfg(feature = "raster-plots")]
        Format::Png | Format::Jpeg | Format::Tiff => {
            let renderer = renderer.ok_or_else(|| {
                anyhow!("this plot needs a GPU adapter, and none was found at startup")
            })?;
            match request.format {
                Format::Png => {
                    Ok(
                        ggsql::writer::PngWriter::new(canvas.width, canvas.height, canvas.dpi)
                            // The interactive path re-encodes on every resize,
                            // so trade bytes for latency there.
                            .compression(ggsql::writer::PngCompression::Fast)
                            .render_with(spec, renderer)?,
                    )
                }
                Format::Jpeg => {
                    Ok(
                        ggsql::writer::JpegWriter::new(canvas.width, canvas.height, canvas.dpi)
                            .render_with(spec, renderer)?,
                    )
                }
                Format::Tiff => {
                    Ok(
                        ggsql::writer::TiffWriter::new(canvas.width, canvas.height, canvas.dpi)
                            .render_with(spec, renderer)?,
                    )
                }
                Format::Svg | Format::Pdf => unreachable!("handled above"),
            }
        }
        #[cfg(not(feature = "raster-plots"))]
        Format::Png | Format::Jpeg | Format::Tiff => {
            let _ = renderer;
            Err(anyhow!(
                "this build has no raster plot formats; rebuild with --features raster-plots"
            ))
        }
    }
}

/// Put anything a format could not express in front of a human.
///
/// Not behind a verbosity flag: a dropped gradient is a defect in the figure a
/// document is about to embed, and the list is empty for everything ggsql draws.
fn report(warnings: &[String], format: &str) {
    for warning in warnings {
        tracing::warn!("{format}: {warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plot with a colour scale, so the SVG carries a gradient — the ids that
    /// collide are the ones a legend gradient defines and references.
    fn a_spec() -> Box<ResolvedPlot> {
        use ggsql::reader::{DuckDBReader, Reader};
        let query = "SELECT * FROM (VALUES (1,2,10),(2,3,50),(3,1,90)) t(x,y,c) \
                     VISUALISE x AS x, y AS y, c AS color DRAW point";
        Box::new(
            DuckDBReader::from_connection_string("duckdb://memory")
                .unwrap()
                .execute(query)
                .unwrap()
                .into_plot()
                .unwrap(),
        )
    }

    fn an_svg_request() -> RenderRequest {
        RenderRequest {
            format: Format::Svg,
            canvas: super::super::Canvas {
                width: 400,
                height: 300,
                dpi: 96.0,
            },
        }
    }

    fn ids(svg: &[u8]) -> Vec<String> {
        let svg = std::str::from_utf8(svg).unwrap();
        svg.match_indices("id=\"")
            .map(|(at, marker)| {
                let rest = &svg[at + marker.len()..];
                rest[..rest.find('"').unwrap()].to_string()
            })
            .collect()
    }

    #[test]
    fn a_static_plot_gets_its_own_id_namespace() {
        let backend = PlotBackend::without_raster();
        let first = backend
            .render_once(a_spec(), an_svg_request())
            .expect("the SVG path needs no adapter");
        let second = backend
            .render_once(a_spec(), an_svg_request())
            .expect("the SVG path needs no adapter");

        let (first, second) = (ids(&first), ids(&second));
        assert!(!first.is_empty(), "the plot defines no ids to namespace");
        // The whole point: the same plot twice in one notebook shares no id.
        for id in &first {
            assert!(
                !second.contains(id),
                "'{id}' would collide across two cells"
            );
        }
    }

    #[test]
    fn a_stored_plot_keeps_one_namespace_across_renders() {
        let backend = PlotBackend::without_raster();
        backend.store("comm-1".to_string(), a_spec());

        let first = backend.render_stored("comm-1", an_svg_request()).unwrap();
        let second = backend.render_stored("comm-1", an_svg_request()).unwrap();

        // A resize must not renumber the ids of a plot already on screen.
        assert_eq!(ids(&first), ids(&second));
        assert!(ids(&first).iter().all(|id| id.starts_with("pcomm-1-")));
    }

    #[test]
    fn a_namespace_is_a_valid_xml_name() {
        // An id may not start with a digit, and a uuid can.
        for namespace in [one_shot_namespace(), comm_namespace("8-4-4-4-12")] {
            let first = namespace.chars().next().unwrap();
            assert!(
                first.is_ascii_alphabetic() || first == '_',
                "'{namespace}' is not a valid XML name"
            );
        }
    }
}
