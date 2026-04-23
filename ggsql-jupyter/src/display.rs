//! Display data formatting for Jupyter output
//!
//! This module formats execution results as Jupyter display_data messages
//! with appropriate MIME types for rich rendering.

use crate::executor::ExecutionResult;
use crate::message::MessageHeader;
use ggsql::DataFrame;
use serde_json::{json, Value};

/// Frontend-supplied hints about the output rendering slot.
///
/// Three render targets, identified by the Jupyter session id on the
/// incoming execute_request:
///
/// - **Positron notebook** (`ggsql-notebook-…`): inline code-chunk output
///   in an editor view. Rendered into a plain 400px container with no
///   layout-mutating observers, because Positron animates the slot during
///   its reveal transition.
/// - **Positron console** (`ggsql-…`): output lands in the Plots pane. The
///   container upgrades to `100vh` inside `.positron-output-container`, so
///   Vega-Lite's own container observer tracks pane resizes.
/// - **Standalone** (anything else — Jupyter notebook, Quarto render, …):
///   the HTML embeds in a static document. An outer/inner div wrapper with
///   a 450px design width applies a uniform CSS-transform scale when the
///   viewport is narrower, so the plot shrinks in proportion instead of
///   squashing.
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderHints {
    pub is_notebook: bool,
    pub is_positron: bool,
    pub output_width_px: Option<u32>,
}

impl RenderHints {
    pub fn from_request(header: &MessageHeader, content: &Value) -> Self {
        let session = header.session.as_str();
        // Positron's supervisor tags every session it manages with a
        // `ggsql-` prefix; standalone Jupyter/Quarto uses UUIDs without one.
        let is_positron = session.starts_with("ggsql-");
        let is_notebook = session.contains("notebook");
        let output_width_px = content
            .get("positron")
            .and_then(|p| p.get("output_width_px"))
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok());
        Self {
            is_notebook,
            is_positron,
            output_width_px,
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
pub fn format_display_data(result: ExecutionResult, hints: &RenderHints) -> Option<Value> {
    match result {
        ExecutionResult::Visualization { spec } => Some(format_vegalite(spec, hints)),
        ExecutionResult::DataFrame(df) => {
            // DDL statements return DataFrames with 0 columns - don't display anything
            if df.width() == 0 {
                None
            } else {
                Some(format_dataframe(df))
            }
        }
        ExecutionResult::ConnectionChanged { display_name, .. } => {
            Some(format_connection_changed(&display_name))
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

/// Format Vega-Lite visualization as display_data
fn format_vegalite(spec: String, hints: &RenderHints) -> Value {
    let html = vegalite_html(&spec, hints);
    json!({
        "data": {
            "text/html": html,
            "text/plain": "Vega-Lite visualization".to_string()
        },
        "metadata": {},
        "transient": {},
        "output_location": "plot"
    })
}

/// Generate the HTML wrapper that embeds a Vega-Lite spec via vega-embed.
pub fn vegalite_html(spec: &str, hints: &RenderHints) -> String {
    let spec_value: Value = serde_json::from_str(spec).unwrap_or_else(|e| {
        tracing::error!("Failed to parse Vega-Lite JSON: {}", e);
        json!({"error": "Invalid Vega-Lite JSON"})
    });

    let spec_json = serde_json::to_string(&spec_value).unwrap_or_else(|_| "{}".to_string());

    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let vis_id = format!("vis-{}", timestamp);

    if hints.is_positron {
        positron_vegalite_html(&spec_json, &vis_id, hints.is_notebook)
    } else {
        standalone_vegalite_html(&spec_json, &vis_id)
    }
}

/// Positron template: plain 400px container with no self-installed layout
/// observers. Console sessions additionally upgrade the container to `100vh`
/// when it lives inside `.positron-output-container`, letting Vega-Lite's
/// own container observer keep the Plots pane responsive. Notebook sessions
/// skip that override and keep a stable 400px box.
fn positron_vegalite_html(spec_json: &str, vis_id: &str, is_notebook: bool) -> String {
    let pane_override_js = if is_notebook {
        ""
    } else {
        "var container = document.getElementById(visId);\n\
         if (container && container.closest('.positron-output-container')) {\n\
         container.style.height = '100vh';\n\
         }\n"
    };

    format!(
        r#"<div id="{vis_id}" style="width: 100%; height: 400px;"></div>
<script type="text/javascript">
(function() {{
var spec = {spec_json};
var visId = '{vis_id}';
var options = {{"actions": true, "renderer": "svg"}};
{pane_override_js}
if (typeof window.requirejs !== 'undefined') {{
window.requirejs.config({{
paths: {{
'dom-ready': 'https://cdn.jsdelivr.net/npm/domready@1/ready.min',
'vega': 'https://cdn.jsdelivr.net/npm/vega@6/build/vega.min',
'vega-lite': 'https://cdn.jsdelivr.net/npm/vega-lite@6.4.1/build/vega-lite.min',
'vega-embed': 'https://cdn.jsdelivr.net/npm/vega-embed@7/build/vega-embed.min'
}}
}});
function docReady(fn) {{
if (document.readyState === 'complete') fn();
else window.addEventListener("load", function() {{ fn(); }});
}}
docReady(function() {{
window.requirejs(["dom-ready", "vega", "vega-embed"], function(domReady, vega, vegaEmbed) {{
domReady(function () {{
vegaEmbed('#' + visId, spec, options).catch(console.error);
}});
}});
}});
}} else {{
function loadScript(src) {{
return new Promise(function(resolve, reject) {{
var script = document.createElement('script');
script.src = src;
script.onload = resolve;
script.onerror = reject;
document.head.appendChild(script);
}});
}}
Promise.all([
loadScript('https://cdn.jsdelivr.net/npm/vega@6'),
loadScript('https://cdn.jsdelivr.net/npm/vega-lite@6.4.1'),
loadScript('https://cdn.jsdelivr.net/npm/vega-embed@7')
])
.then(function() {{ return vegaEmbed('#' + visId, spec, options); }})
.catch(function(err) {{
console.error('Failed to load Vega libraries:', err);
}});
}}
}})();
</script>
"#,
        vis_id = vis_id,
        spec_json = spec_json,
        pane_override_js = pane_override_js
    )
}

/// Standalone template: outer/inner div wrapper driving a uniform
/// scale-to-fit. The inner div holds a 450px design width; when the outer
/// container measures narrower, a CSS transform scales the inner block
/// proportionally and the outer height follows the scaled content. A
/// `ResizeObserver` on the outer div keeps the transform current as the
/// document viewport resizes.
fn standalone_vegalite_html(spec_json: &str, vis_id: &str) -> String {
    format!(
        r#"<div id="{vis_id}-outer" style="width: 100%; overflow: hidden;">
<div id="{vis_id}" style="width: 100%; min-width: 450px; height: 400px;"></div>
</div>
<script type="text/javascript">
(function() {{
var spec = {spec_json};
var visId = '{vis_id}';
var minWidth = 450;
var inner = document.getElementById(visId);
var outer = document.getElementById(visId + '-outer');
var options = {{"actions": true, "renderer": "svg"}};
function scaleToFit(o, i) {{
var available = o.clientWidth;
if (available < minWidth) {{
var scale = available / minWidth;
i.style.transform = 'scale(' + scale + ')';
i.style.transformOrigin = 'top left';
o.style.height = (i.scrollHeight * scale) + 'px';
}} else {{
i.style.transform = '';
o.style.height = '';
}}
}}
function onRendered() {{
scaleToFit(outer, inner);
var ro = new ResizeObserver(function() {{ scaleToFit(outer, inner); }});
ro.observe(outer);
}}
if (typeof window.requirejs !== 'undefined') {{
window.requirejs.config({{
paths: {{
'dom-ready': 'https://cdn.jsdelivr.net/npm/domready@1/ready.min',
'vega': 'https://cdn.jsdelivr.net/npm/vega@6/build/vega.min',
'vega-lite': 'https://cdn.jsdelivr.net/npm/vega-lite@6.4.1/build/vega-lite.min',
'vega-embed': 'https://cdn.jsdelivr.net/npm/vega-embed@7/build/vega-embed.min'
}}
}});
function docReady(fn) {{
if (document.readyState === 'complete') fn();
else window.addEventListener("load", function() {{ fn(); }});
}}
docReady(function() {{
window.requirejs(["dom-ready", "vega", "vega-embed"], function(domReady, vega, vegaEmbed) {{
domReady(function () {{
vegaEmbed('#' + visId, spec, options).then(onRendered).catch(console.error);
}});
}});
}});
}} else {{
function loadScript(src) {{
return new Promise(function(resolve, reject) {{
var script = document.createElement('script');
script.src = src;
script.onload = resolve;
script.onerror = reject;
document.head.appendChild(script);
}});
}}
Promise.all([
loadScript('https://cdn.jsdelivr.net/npm/vega@6'),
loadScript('https://cdn.jsdelivr.net/npm/vega-lite@6.4.1'),
loadScript('https://cdn.jsdelivr.net/npm/vega-embed@7')
])
.then(function() {{ return vegaEmbed('#' + visId, spec, options); }})
.then(onRendered)
.catch(function(err) {{
console.error('Failed to load Vega libraries:', err);
}});
}}
}})();
</script>
"#,
        vis_id = vis_id,
        spec_json = spec_json
    )
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

    #[test]
    fn test_vegalite_format() {
        let spec = r#"{"mark": "point"}"#.to_string();
        let result = ExecutionResult::Visualization { spec };
        let display = format_display_data(result, &RenderHints::default())
            .expect("Visualization should return Some");

        assert!(display["data"]["text/html"].is_string());
        assert!(display["data"]["text/plain"].is_string());
    }

    #[test]
    fn test_empty_dataframe_returns_none() {
        // DDL statements return DataFrames with 0 columns
        let df = DataFrame::empty();
        let result = ExecutionResult::DataFrame(df);
        let display = format_display_data(result, &RenderHints::default());

        assert!(
            display.is_none(),
            "Empty DataFrame (0 columns) should return None"
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
        let display = format_display_data(result, &RenderHints::default());

        assert!(
            display.is_some(),
            "DataFrame with columns but 0 rows should return Some"
        );
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(
            escape_html("<script>alert('xss')</script>"),
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        );
    }

    fn positron_console() -> RenderHints {
        RenderHints {
            is_notebook: false,
            is_positron: true,
            output_width_px: None,
        }
    }

    fn positron_notebook() -> RenderHints {
        RenderHints {
            is_notebook: true,
            is_positron: true,
            output_width_px: Some(589),
        }
    }

    #[test]
    fn test_positron_html_has_no_observer_feedback_loop() {
        // Positron templates must not install a `ResizeObserver` or a
        // `scaleToFit` transform: Positron animates the output slot during
        // reveal, and either one would re-lay out on every animation frame.
        for hints in [positron_console(), positron_notebook()] {
            let html = vegalite_html(r#"{"mark": "point"}"#, &hints);
            assert!(
                !html.contains("new ResizeObserver"),
                "Positron HTML must not install a ResizeObserver (hints={:?})",
                hints
            );
            assert!(
                !html.contains("scaleToFit"),
                "Positron HTML must not include scaleToFit (hints={:?})",
                hints
            );
        }
    }

    #[test]
    fn test_console_html_fills_positron_plots_pane() {
        let html = vegalite_html(r#"{"mark": "point"}"#, &positron_console());
        assert!(
            html.contains(".positron-output-container"),
            "HTML must detect Positron's plots pane for responsive height"
        );
        assert!(
            html.contains("100vh"),
            "HTML must scale to 100vh inside the plots pane"
        );
        assert!(
            html.contains("height: 400px"),
            "HTML must set a 400px baseline height for console output"
        );
    }

    #[test]
    fn test_notebook_html_skips_pane_override() {
        let html = vegalite_html(r#"{"mark": "point"}"#, &positron_notebook());
        assert!(
            html.contains("height: 400px"),
            "notebook container uses the shared 400px baseline"
        );
        assert!(
            !html.contains(".positron-output-container"),
            "notebook HTML must not carry the plots-pane 100vh override"
        );
        assert!(
            !html.contains("100vh"),
            "notebook HTML must not reach for 100vh"
        );
    }

    #[test]
    fn test_standalone_html_uses_scale_to_fit() {
        // Standalone (Jupyter/Quarto) renders into a static document and
        // wraps the plot in the outer/inner div + min-width scale-to-fit so
        // narrow viewports shrink the plot proportionally.
        let html = vegalite_html(r#"{"mark": "point"}"#, &RenderHints::default());
        assert!(
            html.contains("min-width: 450px"),
            "standalone HTML must use the 450px design width"
        );
        assert!(
            html.contains("scaleToFit"),
            "standalone HTML must uniformly scale narrow viewports"
        );
        assert!(
            html.contains("new ResizeObserver"),
            "standalone HTML must observe container resizes"
        );
        assert!(
            html.contains("-outer"),
            "standalone HTML must wrap the inner div in an overflow-hidden outer div"
        );
        assert!(
            !html.contains(".positron-output-container"),
            "standalone HTML must not carry the Positron plots-pane branch"
        );
    }

    #[test]
    fn test_from_request_detects_positron_sessions() {
        let header = |session: &str| MessageHeader {
            msg_id: String::new(),
            session: session.to_string(),
            username: String::new(),
            date: String::new(),
            msg_type: String::new(),
            version: String::new(),
        };
        let console = RenderHints::from_request(&header("ggsql-c2a5a97b"), &json!({}));
        assert!(console.is_positron && !console.is_notebook);

        let notebook = RenderHints::from_request(&header("ggsql-notebook-abc"), &json!({}));
        assert!(notebook.is_positron && notebook.is_notebook);

        let standalone = RenderHints::from_request(&header("abcd-efgh-1234"), &json!({}));
        assert!(!standalone.is_positron && !standalone.is_notebook);
    }
}
