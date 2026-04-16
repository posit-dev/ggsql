# ---------------------------------------------------------------------------
# Server-side Vega-Lite rendering via V8
# ---------------------------------------------------------------------------

#' Save a ggsql spec to a file
#'
#' This function renders a specification and returns it either as a Vegalite
#' json string, an SVG or a PNG. For the latter two, the Vegalite JSON is
#' rendered to SVG using the V8 package and, potentially, converted to PNG using
#' the rsvg package.
#'
#' @param spec A `Spec` object returned by [ggsql_execute()].
#' @param file Output file path. Extension determines format: `.svg`, `.png`, or
#' `.json`.
#' @param width Width in pixels.
#' @param height Height in pixels.
#'
#' @return `file`, invisibly.
#'
#' @export
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' spec <- ggsql_execute(reader,
#'   "SELECT * FROM cars VISUALISE mpg AS x, disp AS y DRAW point"
#' )
#' svg_file <- tempfile(fileext = ".svg")
#' ggsql_save(spec, svg_file)
#'
ggsql_save <- function(spec, file, width = 600, height = 400) {
  ext <- tolower(tools::file_ext(file))
  switch(
    ext,
    svg = writeLines(ggsql_to_svg(spec, width, height), file),
    png = writeBin(ggsql_to_png(spec, width, height), file),
    json = writeLines(ggsql_render(vegalite_writer(), spec), file),
    cli::cli_abort("Unsupported format {.val {ext}}. Use svg, png, or json.")
  )
  invisible(file)
}

#' Render a ggsql spec to SVG
#'
#' Renders a visualization specification to an SVG string using Vega-Lite
#' via V8 (no browser required).
#'
#' @param spec A `Spec` object returned by [ggsql_execute()].
#' @param width Width in pixels.
#' @param height Height in pixels.
#' @return An SVG string (character).
#' @noRd
ggsql_to_svg <- function(spec, width = 600, height = 400) {
  rlang::check_installed("V8", reason = "to render Vega-Lite specs to SVG.")
  json <- ggsql_render(vegalite_writer(), spec)
  vegalite_to_svg(json, width = width, height = height)
}

#' Render a ggsql spec to PNG
#'
#' Renders a visualization specification to a PNG raw vector using Vega-Lite
#' via V8 and rsvg.
#'
#' @param spec A `Spec` object returned by [ggsql_execute()].
#' @param width Width in pixels.
#' @param height Height in pixels.
#' @return A raw vector containing PNG data.
#' @noRd
ggsql_to_png <- function(spec, width = 600, height = 400) {
  rlang::check_installed("rsvg", reason = "to render Vega-Lite specs to PNG.")
  svg <- ggsql_to_svg(spec, width = width, height = height)
  rsvg::rsvg_png(charToRaw(svg), width = width, height = height)
}

# ---------------------------------------------------------------------------
# V8 context management (cached, lazy-loaded)
# ---------------------------------------------------------------------------

get_vega_ctx <- function() {
  if (is.null(ggsql_env$v8_ctx)) {
    ggsql_env$v8_ctx <- create_vega_ctx()
  }
  ggsql_env$v8_ctx
}

create_vega_ctx <- function() {
  ctx <- V8::v8()

  # Polyfill APIs that Vega 6 expects but V8 doesn't provide
  ctx$eval(
    "
    if (typeof structuredClone === 'undefined') {
      globalThis.structuredClone = function(obj) {
        return JSON.parse(JSON.stringify(obj));
      };
    }
  "
  )

  # Load Vega libraries from CDN
  ctx$source("https://cdn.jsdelivr.net/npm/vega@6/build/vega.min.js")
  ctx$source("https://cdn.jsdelivr.net/npm/vega-lite@6/build/vega-lite.min.js")

  # Define helper function for spec → SVG conversion
  ctx$eval(
    "
    function renderToSvg(specJson, width, height) {
      var spec = JSON.parse(specJson);
      if (width) spec.width = width;
      if (height) spec.height = height;
      // Compile Vega-Lite to Vega
      var vegaSpec = vegaLite.compile(spec).spec;
      // Create a headless Vega view and render to SVG
      var view = new vega.View(vega.parse(vegaSpec), { renderer: 'none' });
      return view.toSVG();
    }
  "
  )

  ctx
}

vegalite_to_svg <- function(spec_json, width = NULL, height = NULL) {
  ctx <- get_vega_ctx()

  # Assign the spec as a variable to avoid quoting issues with large JSON
  ctx$assign("__spec_json__", spec_json)
  ctx$assign("__width__", width)
  ctx$assign("__height__", height)

  # toSVG() returns a Promise, use await
  ctx$eval("renderToSvg(__spec_json__, __width__, __height__)", await = TRUE)
}
