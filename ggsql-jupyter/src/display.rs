//! Display data formatting for Jupyter output
//!
//! This module formats execution results as Jupyter display_data messages
//! with appropriate MIME types for rich rendering.

use crate::executor::ExecutionResult;
use polars::frame::DataFrame;
use serde_json::{json, Value};

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
pub fn format_display_data(result: ExecutionResult) -> Option<Value> {
    match result {
        ExecutionResult::Visualization { spec } => Some(format_vegalite(spec)),
        ExecutionResult::DataFrame(df) => {
            // DDL statements return DataFrames with 0 columns - don't display anything
            if df.width() == 0 {
                None
            } else {
                Some(format_dataframe(df))
            }
        }
    }
}

/// Format Vega-Lite visualization as display_data
fn format_vegalite(spec: String) -> Value {
    let spec_value: Value = serde_json::from_str(&spec).unwrap_or_else(|e| {
        tracing::error!("Failed to parse Vega-Lite JSON: {}", e);
        json!({"error": "Invalid Vega-Lite JSON"})
    });

    // Generate HTML with embedded Vega-Embed for universal compatibility
    // Use require.js approach for Jupyter compatibility
    let spec_json = serde_json::to_string(&spec_value).unwrap_or_else(|_| "{}".to_string());

    // Generate unique ID for this visualization
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let vis_id = format!("vis-{}", timestamp);

    let html = format!(
        r#"<div id="{vis_id}-outer" style="width: 100%; overflow: hidden;">
<div id="{vis_id}" style="min-width: 450px; height: 400px;"></div>
</div>
<script type="text/javascript">
  (function() {{
    const spec = {spec_json};
    const visId = '{vis_id}';
    const minWidth = 450;
    const inner = document.getElementById(visId);
    const outer = document.getElementById(visId + '-outer');

    // Use full height in Positron's Plots pane
    if (inner.closest('.positron-output-container')) {{
      inner.style.height = '100vh';
    }}

    const options = {{
      "actions": true,
    }};

    function scaleToFit(outer, inner) {{
      const available = outer.clientWidth;
      if (available < minWidth) {{
        const scale = available / minWidth;
        inner.style.transform = 'scale(' + scale + ')';
        inner.style.transformOrigin = 'top left';
        outer.style.height = (inner.scrollHeight * scale) + 'px';
      }} else {{
        inner.style.transform = '';
        outer.style.height = '';
      }}
    }}

    function onRendered() {{
      scaleToFit(outer, inner);
      const ro = new ResizeObserver(() => scaleToFit(outer, inner));
      ro.observe(outer);
    }}

    // Check if we're in a Jupyter environment with require.js
    if (typeof window.requirejs !== 'undefined') {{
      // Use require.js to load Vega libraries
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
        else window.addEventListener("load", () => {{
          fn();
        }});
      }}
      docReady(function() {{
        window.requirejs(["dom-ready", "vega", "vega-embed"], function(domReady, vega, vegaEmbed) {{
            domReady(function () {{
              vegaEmbed('#' + visId, spec, options).then(onRendered).catch(console.error);
            }});
        }});
      }});
    }} else {{
      // Fallback for non-Jupyter environments
      function loadScript(src) {{
        return new Promise((resolve, reject) => {{
          const script = document.createElement('script');
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
        .then(() => vegaEmbed('#' + visId, spec, options))
        .then(onRendered)
        .catch(err => {{
          console.error('Failed to load Vega libraries:', err);
        }});
    }}
  }})();
</script>"#,
        vis_id = vis_id,
        spec_json = spec_json
    );

    json!({
        "data": {
            // HTML with embedded vega-embed for rendering
            "text/html": html,

            // Text fallback
            "text/plain": "Vega-Lite visualization".to_string()
        },
        "metadata": {},
        "transient": {},
        // Route to Positron Plots pane
        "output_location": "plot"
    })
}

/// Format DataFrame as HTML table
fn format_dataframe(df: DataFrame) -> Value {
    let html = dataframe_to_html(&df);
    let text = format!("{}", df);

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
    let mut html = String::from("<table border=\"1\" class=\"dataframe\">\n<thead><tr>");

    // Header row
    for col in df.get_column_names() {
        html.push_str(&format!("<th>{}</th>", escape_html(col)));
    }
    html.push_str("</tr></thead>\n<tbody>\n");

    // Data rows (limit to first 100 for performance)
    let row_limit = df.height().min(100);
    for i in 0..row_limit {
        html.push_str("<tr>");
        for col in df.get_columns() {
            let value = col
                .get(i)
                .unwrap_or_else(|_| polars::prelude::AnyValue::Null);
            html.push_str(&format!("<td>{}</td>", escape_html(&value.to_string())));
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
        let display = format_display_data(result).expect("Visualization should return Some");

        assert!(display["data"]["text/html"].is_string());
        assert!(display["data"]["text/plain"].is_string());
    }

    #[test]
    fn test_empty_dataframe_returns_none() {
        use polars::prelude::*;

        // DDL statements return DataFrames with 0 columns
        let df = DataFrame::new(Vec::<Column>::new()).unwrap();
        let result = ExecutionResult::DataFrame(df);
        let display = format_display_data(result);

        assert!(
            display.is_none(),
            "Empty DataFrame (0 columns) should return None"
        );
    }

    #[test]
    fn test_empty_rows_dataframe_returns_some() {
        use polars::prelude::*;

        // SELECT with 0 rows but columns should still display
        let df = DataFrame::new(vec![Column::new("x".into(), Vec::<i32>::new())]).unwrap();
        let result = ExecutionResult::DataFrame(df);
        let display = format_display_data(result);

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
}
