//! Display data formatting for Jupyter output
//!
//! This module formats execution results as Jupyter display_data messages
//! with appropriate MIME types for rich rendering.

use crate::executor::ExecutionResult;
use polars::frame::DataFrame;
use serde_json::{json, Value};

/// Format execution result as Jupyter display_data content
///
/// Returns a JSON value matching the Jupyter display_data message format:
/// ```json
/// {
///   "data": { "mime/type": content, ... },
///   "metadata": { ... },
///   "transient": { ... }
/// }
/// ```
pub fn format_display_data(result: ExecutionResult) -> Value {
    match result {
        ExecutionResult::Visualization {
            spec
        } => format_vegalite(spec),
        ExecutionResult::DataFrame(df) => format_dataframe(df),
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
        r#"<div id="{}"></div>
<script type="text/javascript">
  (function() {{
    const spec = {};
    const visId = '{}';

    // Check if we're in a Jupyter environment with require.js
    if (typeof window.requirejs !== 'undefined') {{
      // Use require.js to load Vega libraries
      window.requirejs.config({{
        paths: {{
          'vega': 'https://cdn.jsdelivr.net/npm/vega@5/build/vega.min',
          'vega-lite': 'https://cdn.jsdelivr.net/npm/vega-lite@5/build/vega-lite.min',
          'vega-embed': 'https://cdn.jsdelivr.net/npm/vega-embed@6/build/vega-embed.min'
        }}
      }});

      setTimeout(() => {{
        window.requirejs(['vega-embed'], function(vegaEmbed) {{
          vegaEmbed('#' + visId, spec, {{"actions": true}}).catch(console.error);
        }});
      }}, 100);
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
        loadScript('https://cdn.jsdelivr.net/npm/vega@5'),
        loadScript('https://cdn.jsdelivr.net/npm/vega-lite@5'),
        loadScript('https://cdn.jsdelivr.net/npm/vega-embed@6')
      ])
        .then(() => {{
          vegaEmbed('#' + visId, spec, {{"actions": true}})
            .catch(console.error);
        }})
        .catch(err => {{
          console.error('Failed to load Vega libraries:', err);
        }});
    }}
  }})();
</script>"#,
        vis_id, spec_json, vis_id
    );

    json!({
        "data": {
            // Primary MIME type - JupyterLab will render this natively (if vega5 extension installed)
            "application/vnd.vegalite.v5+json": spec_value,

            // HTML fallback - works everywhere with web connection
            "text/html": html,

            // JSON fallback
            "application/json": spec_value,

            // Text fallback
            "text/plain": "Vega-Lite visualization".to_string()
        },
        "metadata": {
            "application/vnd.vegalite.v5+json": {
                "embedOptions": {
                    "actions": true  // Show export/edit actions
                }
            }
        },
        "transient": {}
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
        let result = ExecutionResult::Visualization {
            spec
        };
        let display = format_display_data(result);

        assert!(display["data"]["application/vnd.vegalite.v5+json"].is_object());
        assert!(display["data"]["text/plain"].is_string());
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(
            escape_html("<script>alert('xss')</script>"),
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        );
    }
}
