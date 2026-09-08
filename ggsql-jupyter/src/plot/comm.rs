//! The `positron.plot` comm: parsing its requests and shaping its replies.
//!
//! Everything here is pure — it turns JSON into typed values and back — so the
//! protocol can be tested without a Positron host, a GPU, or a running kernel.
//!
//! The schema is Positron's, from `positron/comms/plot-backend-openrpc.json`:
//!
//! | Request | Params | Result |
//! | --- | --- | --- |
//! | `render` | `size: {width, height}` *optional*; `pixel_ratio` required; `format` required | `{ data, mime_type, settings? }` |
//! | `get_intrinsic_size` | none | a size, **or `null`** |
//! | `get_metadata` | none | `{ name, kind, execution_id, code }` |

use serde_json::{json, Value};

use super::{Canvas, Format, RenderRequest};

/// A JSON-RPC error, as Positron's frontend expects to receive it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcError {
    /// The method is not one this comm implements.
    MethodNotFound(String),
    /// The method is right but its params are not usable.
    InvalidParams(String),
    /// The request was understood and could not be served.
    Internal(String),
}

impl RpcError {
    /// The JSON-RPC code, from the spec's reserved range.
    fn code(&self) -> i32 {
        match self {
            Self::MethodNotFound(_) => -32601,
            Self::InvalidParams(_) => -32602,
            Self::Internal(_) => -32603,
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::MethodNotFound(m) | Self::InvalidParams(m) | Self::Internal(m) => m,
        }
    }

    /// The `data.error` half of a reply.
    pub fn to_json(&self) -> Value {
        json!({ "code": self.code(), "message": self.message() })
    }
}

/// A `render` request, resolved to something the renderer can act on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderParams {
    pub request: RenderRequest,
    /// The ratio the frontend reported, echoed back in the reply's `settings`
    /// so it can tell which settings a frame was produced for.
    pub pixel_ratio: f64,
}

impl RenderParams {
    /// Read a `render` request's params.
    ///
    /// `size` is optional in the schema, so an absent one falls back to the
    /// default canvas. `pixel_ratio` and `format` are required, and an unknown
    /// format is `InvalidParams` rather than a guess.
    ///
    /// # Errors
    ///
    /// Returns [`RpcError::InvalidParams`] for a missing or unusable
    /// `pixel_ratio` or `format`.
    pub fn from_rpc(params: &Value) -> Result<Self, RpcError> {
        let pixel_ratio = params
            .get("pixel_ratio")
            .and_then(Value::as_f64)
            .ok_or_else(|| RpcError::InvalidParams("render needs a pixel_ratio".into()))?;

        let format = match params.get("format").and_then(Value::as_str) {
            Some("png") => Format::Png,
            Some("jpeg") => Format::Jpeg,
            Some("svg") => Format::Svg,
            Some("pdf") => Format::Pdf,
            Some("tiff") => Format::Tiff,
            Some(other) => {
                return Err(RpcError::InvalidParams(format!(
                    "render cannot produce the format '{other}'"
                )))
            }
            None => return Err(RpcError::InvalidParams("render needs a format".into())),
        };

        let size = params.get("size").filter(|s| !s.is_null());
        let canvas = match size {
            Some(size) => {
                let dimension = |key: &str| size.get(key).and_then(Value::as_f64).unwrap_or(0.0);
                Canvas::from_logical(dimension("width"), dimension("height"), pixel_ratio)
            }
            // No size means "use your own", and ggsql plots have no intrinsic
            // size — so the default canvas is the honest answer.
            None => {
                let default = Canvas::default();
                Canvas::from_logical(
                    f64::from(default.width),
                    f64::from(default.height),
                    pixel_ratio,
                )
            }
        };

        Ok(Self {
            request: RenderRequest { format, canvas },
            pixel_ratio,
        })
    }

    /// The same request, in a format this build and machine can produce.
    ///
    /// The pane always asks for `png`, so a build with no raster writer answers
    /// in SVG rather than leaving the pane empty, and says so in `mime_type`.
    /// See [`Format::available`](crate::plot::Format::available).
    pub fn available(mut self, backend_raster: bool) -> Self {
        self.request.format = self.request.format.available(backend_raster);
        self
    }

    /// Encode rendered bytes the way a comm reply carries them.
    ///
    /// Always base64, even for SVG: Positron builds
    /// `data:{mime_type};base64,{data}` from the reply
    /// (`languageRuntimePlotClient.ts`), so text sent as itself is an invalid
    /// URI. A static display bundle is the other way round, which is why this
    /// is the comm's one entry point.
    pub fn encode(bytes: &[u8]) -> String {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        STANDARD.encode(bytes)
    }

    /// The `result` half of a successful `render` reply.
    ///
    /// `mime_type` names the format actually produced. Positron builds its data
    /// URI from it but takes the save extension from its own request, so a
    /// substituted format would write the wrong bytes into the wrong file.
    pub fn to_result(self, encoded: String) -> Value {
        json!({
            "data": encoded,
            "mime_type": self.request.format.mime(),
            "settings": {
                "size": {
                    "width": self.request.canvas.css_size().0,
                    "height": self.request.canvas.css_size().1,
                },
                "pixel_ratio": self.pixel_ratio,
                "format": format_name(self.request.format),
            }
        })
    }
}

/// The name Positron uses for a format on the wire.
pub fn format_name(format: Format) -> &'static str {
    match format {
        Format::Png => "png",
        Format::Jpeg => "jpeg",
        Format::Svg => "svg",
        Format::Pdf => "pdf",
        Format::Tiff => "tiff",
    }
}

/// What a plot tells Positron about itself.
#[derive(Debug, Clone)]
pub struct PlotMetadata {
    /// Shown in the Plots pane's history.
    pub name: String,
    /// The `msg_id` of the execute_request that produced the plot.
    pub execution_id: String,
    /// The cell text, which the pane offers as "show the code".
    pub code: String,
}

impl PlotMetadata {
    /// The `result` half of a `get_metadata` reply.
    ///
    /// Answered from state the kernel already holds, never through the render
    /// thread: `get_metadata` gets a 5 s RPC timeout where `render` gets 30 s,
    /// so it must not queue behind one.
    pub fn to_result(&self) -> Value {
        json!({
            "name": self.name,
            "kind": "ggsql",
            "execution_id": self.execution_id,
            "code": self.code,
        })
    }
}

/// A plot's name for the pane's history list.
///
/// The `LABEL title` if the query set one, else a running count, matching how
/// the other language backends name an untitled figure.
pub fn plot_name(title: Option<&str>, sequence: u32) -> String {
    match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(title) => title.to_string(),
        None => format!("ggsql {sequence}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_render_request_becomes_a_canvas() {
        let params =
            json!({"size": {"width": 800, "height": 600}, "pixel_ratio": 2.0, "format": "png"});
        let parsed = RenderParams::from_rpc(&params).unwrap();
        assert_eq!(parsed.request.format, Format::Png);
        assert_eq!(parsed.request.canvas.width, 1600);
        assert_eq!(parsed.request.canvas.height, 1200);
        assert_eq!(parsed.request.canvas.dpi, 192.0);
    }

    #[test]
    fn a_ratio_of_one_leaves_the_size_alone() {
        let params =
            json!({"size": {"width": 800, "height": 600}, "pixel_ratio": 1, "format": "png"});
        let parsed = RenderParams::from_rpc(&params).unwrap();
        assert_eq!(parsed.request.canvas.width, 800);
        assert_eq!(parsed.request.canvas.dpi, 96.0);
    }

    #[test]
    fn an_absent_size_falls_back_rather_than_failing() {
        // The schema makes `size` optional; ggsql has no intrinsic size, so
        // the default canvas is the honest answer.
        let parsed = RenderParams::from_rpc(&json!({"pixel_ratio": 1, "format": "png"})).unwrap();
        assert_eq!(parsed.request.canvas.width, Canvas::default().width);
        // And an explicit null is the same as absent.
        let with_null =
            RenderParams::from_rpc(&json!({"size": null, "pixel_ratio": 1, "format": "png"}))
                .unwrap();
        assert_eq!(with_null.request.canvas, parsed.request.canvas);
    }

    #[test]
    fn a_transient_zero_size_is_clamped_not_rendered() {
        // Positron reports this while laying out a pane, and a zero-sized GPU
        // target is a hard error.
        let parsed = RenderParams::from_rpc(
            &json!({"size": {"width": 0, "height": 0}, "pixel_ratio": 1, "format": "png"}),
        )
        .unwrap();
        assert_eq!(parsed.request.canvas.width, 32);
        assert_eq!(parsed.request.canvas.height, 32);
    }

    #[test]
    fn every_format_positron_defines_is_accepted() {
        let cases = [
            ("png", Format::Png),
            ("jpeg", Format::Jpeg),
            ("svg", Format::Svg),
            ("pdf", Format::Pdf),
            ("tiff", Format::Tiff),
        ];
        for (name, expected) in cases {
            let params = json!({"pixel_ratio": 1, "format": name});
            assert_eq!(
                RenderParams::from_rpc(&params).unwrap().request.format,
                expected,
                "{name}"
            );
            // And the name round-trips, so a reply's `settings.format` matches
            // what was asked for.
            assert_eq!(format_name(expected), name);
        }
    }

    #[test]
    fn a_format_we_cannot_produce_is_invalid_params() {
        let err = RenderParams::from_rpc(&json!({"pixel_ratio": 1, "format": "webp"})).unwrap_err();
        assert!(matches!(err, RpcError::InvalidParams(_)));
        assert_eq!(err.code(), -32602);
        assert!(err.message().contains("webp"), "{}", err.message());
    }

    #[test]
    fn the_required_params_are_required() {
        for params in [json!({"format": "png"}), json!({"pixel_ratio": 1})] {
            let err = RenderParams::from_rpc(&params).unwrap_err();
            assert!(matches!(err, RpcError::InvalidParams(_)), "{params}");
        }
    }

    #[test]
    fn a_reply_names_the_format_it_actually_produced() {
        // Positron builds its data URI from `mime_type` but records the format
        // from its own request, so a substituted format would save the wrong
        // bytes into the wrong extension.
        let parsed = RenderParams::from_rpc(
            &json!({"size": {"width": 400, "height": 300}, "pixel_ratio": 2, "format": "png"}),
        )
        .unwrap();
        let result = parsed.to_result("AAAA".into());
        assert_eq!(result["mime_type"], "image/png");
        assert_eq!(result["data"], "AAAA");
        // The echoed settings are in CSS pixels, not device pixels.
        assert_eq!(result["settings"]["size"]["width"], 400);
        assert_eq!(result["settings"]["size"]["height"], 300);
        assert_eq!(result["settings"]["pixel_ratio"], 2.0);
        assert_eq!(result["settings"]["format"], "png");
    }

    #[test]
    fn the_panes_render_settings_parse_as_a_render_request() {
        // `did_change_plots_render_settings` carries a `plot_render_settings`,
        // which has a `render` request's shape — parsed by the same function so
        // a pre-render matches what a render would produce.
        let settings = json!({
            "size": {"width": 500, "height": 400},
            "pixel_ratio": 2,
            "format": "png"
        });
        let parsed = RenderParams::from_rpc(&settings).unwrap();
        assert_eq!(parsed.request.canvas.width, 1000);
        assert_eq!(parsed.request.canvas.height, 800);
        assert_eq!(parsed.request.format, Format::Png);
        // And the echoed settings match what the pane asked for, which is what
        // Positron gates a pre-render on.
        let result = parsed.to_result("AAAA".into());
        assert_eq!(result["settings"]["size"]["width"], 500);
        assert_eq!(result["settings"]["size"]["height"], 400);
        assert!(result["settings"].get("pixel_ratio").is_some());
    }

    #[test]
    fn svg_is_base64_encoded_like_every_other_format() {
        // Positron builds `data:<mime>;base64,<data>` from a comm reply, so
        // text sent as itself would be an invalid URI.
        let encoded = RenderParams::encode(b"<svg/>");
        assert_eq!(encoded, "PHN2Zy8+");
        assert!(
            !encoded.starts_with('<'),
            "svg must not travel as text here"
        );
    }

    #[test]
    fn an_unknown_method_is_method_not_found() {
        // An error rather than `result: null`, so a future method fails loudly
        // instead of being satisfied with garbage.
        let err = RpcError::MethodNotFound("hover".into());
        assert_eq!(err.code(), -32601);
        assert_eq!(err.to_json()["code"], -32601);
    }

    #[test]
    fn metadata_names_a_titled_plot_by_its_title() {
        assert_eq!(plot_name(Some("Sales by region"), 3), "Sales by region");
        assert_eq!(plot_name(None, 3), "ggsql 3");
        // A blank title is not a name.
        assert_eq!(plot_name(Some("   "), 7), "ggsql 7");
    }

    #[test]
    fn metadata_declares_its_kind() {
        let metadata = PlotMetadata {
            name: "ggsql 1".into(),
            execution_id: "abc-123".into(),
            code: "SELECT 1 VISUALISE …".into(),
        };
        let result = metadata.to_result();
        assert_eq!(result["kind"], "ggsql");
        assert_eq!(result["name"], "ggsql 1");
        assert_eq!(result["execution_id"], "abc-123");
        assert!(result["code"].as_str().unwrap().starts_with("SELECT"));
    }
}
