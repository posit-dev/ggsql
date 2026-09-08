//! Display data formatting for Jupyter output
//!
//! This module formats execution results as Jupyter display_data messages
//! with appropriate MIME types for rich rendering.

use crate::executor::ExecutionResult;
use crate::message::MessageHeader;
use crate::plot::{self, Canvas, Delivery, PlotBackend, RenderRequest};
use anyhow::Result;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use clap::ValueEnum;
use ggsql::reader::Spec;
use ggsql::DataFrame;
use serde_json::{json, Value};

/// What the frontend declared itself to be, via `--session-mode`.
///
/// Only a frontend that knows what it is launching passes this — in practice
/// the ggsql extension. Everything else is classified by the heuristic below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SessionMode {
    /// A Positron console session: plots belong in the Plots pane.
    Console,
    /// A Positron notebook session: plots belong in the cell.
    Notebook,
    /// A Positron background session, attached to no UI at all. Output has
    /// nowhere special to go, so it is treated exactly like a session Positron
    /// is not driving.
    Background,
}

/// Where a plot this kernel produces is meant to end up.
///
/// Positron routes a plot comm to the Plots pane whatever kind of session
/// opened it, so a notebook using the comm would leave its cell empty. Console
/// and notebook therefore need different output paths.
///
/// - `PositronConsole`: a `positron.plot` comm and no `execute_result` — the
///   comm alone creates the pane entry, and the pane re-asks on resize.
/// - `PositronNotebook`: a static image bundle in the cell, sized from
///   `output_width_px`.
/// - `Standalone`: anything else — Jupyter, Quarto, nbconvert, and a Positron
///   background session. A static bundle in `QUARTO_FIG_FORMAT`'s format.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    PositronConsole,
    PositronNotebook,
    #[default]
    Standalone,
}

impl SessionKind {
    /// Classify a session, preferring what the frontend declared.
    ///
    /// `--session-mode` is authoritative; the session-id heuristic is the
    /// fallback for Jupyter, Quarto and extensions predating the flag.
    pub fn resolve(session: &str, mode: Option<SessionMode>) -> Self {
        match mode {
            Some(SessionMode::Console) => Self::PositronConsole,
            Some(SessionMode::Notebook) => Self::PositronNotebook,
            // A background session has no pane and no cell, so there is no
            // Positron-specific slot to render into.
            Some(SessionMode::Background) => Self::Standalone,
            // Positron's supervisor prefixes every session it manages with
            // `ggsql-`; Jupyter and Quarto use bare UUIDs.
            None if !session.starts_with("ggsql-") => Self::Standalone,
            None if session.contains("notebook") => Self::PositronNotebook,
            None => Self::PositronConsole,
        }
    }
}

/// Frontend-supplied hints about the output rendering slot.
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderHints {
    pub kind: SessionKind,
    /// Width of the output slot in CSS pixels, when the frontend says.
    pub output_width_px: Option<u32>,
    /// Device pixel ratio of the display, when the frontend says.
    pub pixel_ratio: Option<f64>,
}

impl RenderHints {
    pub fn from_request(
        header: &MessageHeader,
        content: &Value,
        mode: Option<SessionMode>,
    ) -> Self {
        // Positron puts both on the execute request for a notebook or inline
        // cell — see `runtimeNotebookKernel.ts`.
        let positron = content.get("positron");
        let output_width_px = positron
            .and_then(|p| p.get("output_width_px"))
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok());
        let pixel_ratio = positron
            .and_then(|p| p.get("output_pixel_ratio"))
            .and_then(|v| v.as_f64())
            .filter(|v| *v > 0.0);
        Self {
            kind: SessionKind::resolve(header.session.as_str(), mode),
            output_width_px,
            pixel_ratio,
        }
    }

    /// The canvas a static render should use.
    ///
    /// An execute request reports a width but no height: a cell output is as
    /// wide as the cell and as tall as it is given. The height is therefore
    /// ours to pick, and the golden ratio is close to ggplot2's default figure.
    ///
    /// The Plots pane is sized elsewhere — it reports a full `{width, height}`
    /// per render, not per execution, so none of it reaches here.
    pub fn canvas(&self) -> Canvas {
        let ratio = self.pixel_ratio.unwrap_or(1.0);
        match self.output_width_px {
            Some(width) if width > 0 => {
                let width = f64::from(width);
                Canvas::from_logical(width, width / 1.618, ratio)
            }
            _ => {
                let default = Canvas::default();
                Canvas::from_logical(f64::from(default.width), f64::from(default.height), ratio)
            }
        }
    }
}

/// Format execution result as Jupyter display_data content
///
/// Returns `Some(Value)` for results that should be displayed, or `None` for
/// empty results (e.g., DDL statements like CREATE TABLE that have no columns).
///
/// Note: A SELECT that returns 0 rows but has columns will still display
/// an empty table with headers. Only truly empty DataFrames (0 columns)
/// from DDL statements return `None`.
///
/// The returned JSON matches the Jupyter display_data message format:
/// ```json
/// {
///   "data": { "mime/type": content, ... },
///   "metadata": { ... },
///   "transient": { ... }
/// }
/// ```
/// What the kernel should do with a formatted result.
pub enum Formatted {
    /// Emit this as the cell's `execute_result`.
    Bundle(Value),
    /// Open a `positron.plot` comm for this plot and emit **no**
    /// `execute_result` — the comm alone creates the pane entry.
    PlotComm(Box<Spec>),
    /// Nothing to show, as for a DDL statement.
    Nothing,
}

pub fn format_display_data(
    result: ExecutionResult,
    hints: &RenderHints,
    backend: &PlotBackend,
) -> Result<Formatted> {
    match result {
        // Rendered here rather than at execution time, so the format is chosen
        // where the destination is known — and can fail here too.
        ExecutionResult::Visualization(spec) => {
            match plot::choose(hints.kind, backend.raster(), hints.canvas()) {
                Delivery::Comm => Ok(Formatted::PlotComm(spec)),
                Delivery::Static(request) => {
                    Ok(Formatted::Bundle(format_static(spec, request, backend)?))
                }
            }
        }
        ExecutionResult::DataFrame(df) => {
            // DDL statements return DataFrames with 0 columns - don't display anything
            if df.width() == 0 {
                Ok(Formatted::Nothing)
            } else {
                Ok(Formatted::Bundle(format_dataframe(df)))
            }
        }
        ExecutionResult::ConnectionChanged { display_name, .. } => {
            Ok(Formatted::Bundle(format_connection_changed(&display_name)))
        }
    }
}

/// Format a connection-changed message
fn format_connection_changed(display_name: &str) -> Value {
    let text = format!("Connected to {}", display_name);
    json!({
        "data": {
            "text/plain": text
        },
        "metadata": {},
        "transient": {}
    })
}

/// Render a plot to an image and wrap it as a static display bundle.
///
/// No `output_location`: it routes the output to the Plots pane as well as the
/// cell, so the plot would arrive twice.
///
/// `metadata[mime].width/height` is the CSS-pixel size to display at, honoured
/// by JupyterLab and nbconvert; without it a 2x render appears twice as large.
fn format_static(spec: Box<Spec>, request: RenderRequest, backend: &PlotBackend) -> Result<Value> {
    let metadata = spec.metadata();
    let summary = format!(
        "<ggsql plot: {} layer{}, {} row{}>",
        metadata.layer_count,
        if metadata.layer_count == 1 { "" } else { "s" },
        metadata.rows,
        if metadata.rows == 1 { "" } else { "s" },
    );

    let bytes = backend.render_once(spec, request)?;
    let mime = request.format.mime();
    // SVG is text and travels as itself; everything else is bytes and travels
    // base64-encoded, which is what a display bundle expects for binary data.
    let payload = if request.format.is_text() {
        String::from_utf8(bytes)?
    } else {
        BASE64.encode(&bytes)
    };

    let (css_width, css_height) = request.canvas.css_size();
    Ok(json!({
        "data": {
            mime: payload,
            "text/plain": summary,
        },
        "metadata": {
            mime: { "width": css_width, "height": css_height }
        },
        "transient": {},
    }))
}

/// Format DataFrame as HTML table
fn format_dataframe(df: DataFrame) -> Value {
    let html = dataframe_to_html(&df);
    let text = dataframe_to_text(&df);

    json!({
        "data": {
            "text/html": html,
            "text/plain": text
        },
        "metadata": {},
        "transient": {}
    })
}

/// Convert DataFrame to HTML table
fn dataframe_to_html(df: &DataFrame) -> String {
    use ggsql::array_util::value_to_string;

    let mut html = String::from("<table border=\"1\" class=\"dataframe\">\n<thead><tr>");

    // Header row
    for col in df.get_column_names() {
        html.push_str(&format!("<th>{}</th>", escape_html(&col)));
    }
    html.push_str("</tr></thead>\n<tbody>\n");

    // Data rows (limit to first 100 for performance)
    let row_limit = df.height().min(100);
    for i in 0..row_limit {
        html.push_str("<tr>");
        for col in df.get_columns() {
            let value = value_to_string(col, i);
            html.push_str(&format!("<td>{}</td>", escape_html(&value)));
        }
        html.push_str("</tr>\n");
    }

    if df.height() > row_limit {
        html.push_str(&format!(
            "<tr><td colspan='{}' style='text-align: center;'>... {} more rows</td></tr>\n",
            df.width(),
            df.height() - row_limit
        ));
    }

    html.push_str("</tbody>\n</table>");
    html
}

/// Convert DataFrame to plain-text summary (shape + column names + first rows).
fn dataframe_to_text(df: &ggsql::DataFrame) -> String {
    use ggsql::array_util::value_to_string;

    let mut s = format!("shape: ({}, {})\n", df.height(), df.width());
    let names = df.get_column_names();
    s.push_str(&names.join("\t"));
    s.push('\n');
    let row_limit = df.height().min(10);
    for i in 0..row_limit {
        let row: Vec<String> = df
            .get_columns()
            .iter()
            .map(|c| value_to_string(c, i))
            .collect();
        s.push_str(&row.join("\t"));
        s.push('\n');
    }
    s
}

/// Escape HTML special characters
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolved plot, from a real query — the display layer renders it now,
    /// so a hand-written Vega-Lite string is no longer a stand-in for one.
    fn a_spec() -> Spec {
        use ggsql::reader::{DuckDBReader, Reader};
        DuckDBReader::from_connection_string("duckdb://memory")
            .unwrap()
            .execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap()
    }

    fn render(hints: &RenderHints) -> Value {
        match format_display_data(
            ExecutionResult::Visualization(Box::new(a_spec())),
            hints,
            &backend(),
        )
        .expect("rendering should succeed")
        {
            Formatted::Bundle(bundle) => bundle,
            Formatted::PlotComm(_) => panic!("expected a bundle, not a comm"),
            Formatted::Nothing => panic!("expected output"),
        }
    }

    #[test]
    fn test_a_console_gets_a_comm_even_without_an_adapter() {
        // No GPU here, and the console still opens a comm — Positron would
        // inline a static bundle on its `image/*` mime alone. The comm falls
        // back to SVG and says so in `mime_type`.
        let formatted = format_display_data(
            ExecutionResult::Visualization(Box::new(a_spec())),
            &positron_console(),
            &backend(),
        )
        .expect("rendering should succeed");
        assert!(
            matches!(formatted, Formatted::PlotComm(_)),
            "a console session must not produce an inline bundle"
        );
    }

    #[test]
    fn test_a_notebook_gets_a_static_image_in_its_cell() {
        let display = render(&positron_notebook());

        // SVG, because this backend has no GPU — and the fallback is the point:
        // a plot still arrives.
        let svg = display["data"]["image/svg+xml"].as_str().unwrap();
        assert!(svg.starts_with("<svg"), "{svg:.80}");
        assert!(display["data"]["text/html"].is_null(), "no CDN payload");

        // A plain-text summary for a frontend that renders neither.
        let text = display["data"]["text/plain"].as_str().unwrap();
        assert!(text.contains("layer"), "{text}");

        // **No `output_location`.** That key would route this to the Plots
        // pane as well as the cell, and the plot would arrive twice.
        assert!(
            display.get("output_location").is_none(),
            "a static bundle must not claim a plot slot"
        );
    }

    #[test]
    fn test_a_retina_notebook_renders_at_its_own_ratio() {
        // `output_pixel_ratio` rides alongside `output_width_px`, so a retina
        // cell gets a sharp plot rather than an upscaled one.
        let hints = RenderHints::from_request(
            &header("ggsql-notebook-abc"),
            &json!({"positron": {"output_width_px": 600, "output_pixel_ratio": 2.0}}),
            None,
        );
        assert_eq!(hints.pixel_ratio, Some(2.0));

        let canvas = hints.canvas();
        // Twice the device pixels, at twice the dpi...
        assert_eq!(canvas.width, 1200);
        assert_eq!(canvas.dpi, 192.0);
        // ...displayed at the size the cell actually is.
        assert_eq!(canvas.css_size(), (600, 371));
    }

    #[test]
    fn test_a_missing_ratio_falls_back_to_one() {
        // Plain Jupyter reports nothing and an older Positron only a width. 1x
        // is soft on retina, but assuming 2x wastes four times the pixels
        // everywhere else.
        let hints = RenderHints::from_request(
            &header("abcd-1234"),
            &json!({"positron": {"output_width_px": 600}}),
            None,
        );
        assert_eq!(hints.pixel_ratio, None);
        assert_eq!(hints.canvas().dpi, 96.0);
        assert_eq!(hints.canvas().width, 600);
    }

    #[test]
    fn test_a_nonsense_ratio_is_ignored_rather_than_used() {
        let hints = RenderHints::from_request(
            &header("ggsql-notebook-abc"),
            &json!({"positron": {"output_width_px": 600, "output_pixel_ratio": 0}}),
            None,
        );
        assert_eq!(hints.pixel_ratio, None);
    }

    #[test]
    fn test_a_static_bundle_declares_the_size_to_show_it_at() {
        // 589 CSS px wide, as the notebook hints report.
        let display = render(&positron_notebook());
        let metadata = &display["metadata"]["image/svg+xml"];
        assert_eq!(metadata["width"], 589);
        assert_eq!(metadata["height"], 364);
    }

    #[test]
    fn test_standalone_gets_a_static_image_and_needs_no_network() {
        // A plain Jupyter or Quarto render reaches for no CDN, so a plot works
        // offline and in CI.
        let display = render(&RenderHints::default());
        assert!(display["data"]["image/svg+xml"].is_string());
        let bundle = serde_json::to_string(&display).unwrap();
        assert!(
            !bundle.contains("jsdelivr") && !bundle.contains("vega-embed"),
            "a static bundle should carry no CDN reference"
        );
    }

    #[test]
    fn test_empty_dataframe_returns_none() {
        // DDL statements return DataFrames with 0 columns
        let df = DataFrame::empty();
        let result = ExecutionResult::DataFrame(df);
        let display = format_display_data(result, &RenderHints::default(), &backend()).unwrap();

        assert!(
            matches!(display, Formatted::Nothing),
            "Empty DataFrame (0 columns) should produce nothing"
        );
    }

    #[test]
    fn test_empty_rows_dataframe_returns_some() {
        use arrow::array::{ArrayRef, Int32Array};
        use std::sync::Arc;

        // SELECT with 0 rows but columns should still display
        let empty: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let df = DataFrame::new(vec![("x", empty)]).unwrap();
        let result = ExecutionResult::DataFrame(df);
        let display = format_display_data(result, &RenderHints::default(), &backend()).unwrap();

        assert!(
            matches!(display, Formatted::Bundle(_)),
            "DataFrame with columns but 0 rows should produce a bundle"
        );
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(
            escape_html("<script>alert('xss')</script>"),
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        );
    }

    /// A render backend with no GPU, so tests are fast and identical
    /// everywhere. The SVG path it leaves is the one that always works.
    fn backend() -> PlotBackend {
        PlotBackend::without_raster()
    }

    fn positron_console() -> RenderHints {
        RenderHints {
            kind: SessionKind::PositronConsole,
            output_width_px: None,
            pixel_ratio: None,
        }
    }

    fn positron_notebook() -> RenderHints {
        RenderHints {
            kind: SessionKind::PositronNotebook,
            output_width_px: Some(589),
            pixel_ratio: None,
        }
    }

    fn header(session: &str) -> MessageHeader {
        MessageHeader {
            msg_id: String::new(),
            session: session.to_string(),
            username: String::new(),
            date: String::new(),
            msg_type: String::new(),
            version: String::new(),
        }
    }

    fn kind(session: &str, mode: Option<SessionMode>) -> SessionKind {
        RenderHints::from_request(&header(session), &json!({}), mode).kind
    }

    #[test]
    fn test_from_request_detects_positron_sessions() {
        // The fallback path, for a frontend that passes no `--session-mode`.
        assert_eq!(kind("ggsql-c2a5a97b", None), SessionKind::PositronConsole);
        assert_eq!(
            kind("ggsql-notebook-abc", None),
            SessionKind::PositronNotebook
        );
        assert_eq!(kind("abcd-efgh-1234", None), SessionKind::Standalone);
    }

    #[test]
    fn test_session_mode_overrides_the_heuristic() {
        // A declared mode is believed whatever the session id looks like.
        assert_eq!(
            kind("abcd-efgh-1234", Some(SessionMode::Console)),
            SessionKind::PositronConsole
        );
        assert_eq!(
            kind("ggsql-c2a5a97b", Some(SessionMode::Notebook)),
            SessionKind::PositronNotebook
        );
        assert_eq!(
            kind("ggsql-notebook-abc", Some(SessionMode::Console)),
            SessionKind::PositronConsole
        );
    }

    #[test]
    fn test_a_background_session_has_no_positron_slot() {
        // Positron's session, but attached to no UI — the heuristic's answer
        // (console, from the prefix) would aim output at a pane nobody sees.
        assert_eq!(
            kind("ggsql-bg-4471", Some(SessionMode::Background)),
            SessionKind::Standalone
        );
        assert_eq!(kind("ggsql-bg-4471", None), SessionKind::PositronConsole);
    }

    #[test]
    fn test_a_non_positron_session_is_standalone_whatever_its_id_says() {
        // The heuristic keys on the "ggsql-" prefix, so a foreign id containing
        // "notebook" is still standalone.
        assert_eq!(kind("jupyter-notebook-9f2c", None), SessionKind::Standalone);
        assert_eq!(kind("notebook", None), SessionKind::Standalone);
    }
}
