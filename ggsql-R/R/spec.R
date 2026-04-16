#' @noRd
Spec <- R6::R6Class(
  "Spec",
  cloneable = FALSE,
  public = list(
    .ptr = NULL,

    initialize = function(ptr) {
      self$.ptr <- ptr
    },

    print = function(...) {
      json <- ggsql_render(vegalite_writer(), self)
      html <- vegalite_html(json)
      print(htmltools::browsable(htmltools::HTML(html)))
    }
  )
)

#' @importFrom utils str
#' @export
str.Spec <- function(object, ...) {
  m <- ggsql_metadata(object)
  cli::cli_text("<ggsql_spec>")
  cli::cli_bullets(c(
    "*" = "Rows: {m$rows}",
    "*" = "Columns: {paste(m$columns, collapse = ', ')}",
    "*" = "Layers: {m$layer_count}"
  ))
  invisible()
}
#' @importFrom knitr knit_print
#' @export
knit_print.Spec <- function(x, ..., inline = FALSE) {
  options <- knitr::opts_current$get()
  writer_type <- options$writer %||% "vegalite"

  switch(
    writer_type,
    vegalite = {
      json <- ggsql_render(vegalite_writer(), x)
      html <- vegalite_html(
        json,
        width = options$fig.width,
        height = options$fig.height
      )
      x <- htmltools::browsable(htmltools::HTML(html))
      deps <- htmltools::resolveDependencies(htmltools::findDependencies(
        x,
        tagify = FALSE
      ))
      knitr::asis_output(
        sprintf("\n```{=html}\n%s\n```\n", x),
        meta = if (length(deps)) {
          list(deps)
        }
      )
    },
    vegalite_svg = ,
    vegalite_png = {
      out <- write_static_figure(x, sub("vegalite_", "", writer_type), options)
      knitr::asis_output(out)
    },
    cli::cli_abort("unknown writer {.val {writer_type}}")
  )
}

#' Utility functions for visualization specifications
#'
#' These functions allow you to extract various information from a `Spec` object
#' returned by [ggsql_execute()].
#'
#' @param spec A `Spec` object as returned by [ggsql_execute()]
#' @param index Layer index
#'
#' @return
#' * `ggsql_metadata`: A list with elements `rows`, `columns`, and `layer_count`
#' * `ggsql_sql`: A character string with the SQL portion of the query
#' * `ggsql_visual`: A character string with the visual portion of the query
#' * `ggsql_layer_count`: An integer giving the number of layers
#' * `ggsql_layer_data`: A data frame, or `NULL` if no data is available for
#' this layer
#' * `ggsql_stat_data`: A data frame, or `NULL` if the layer doesn't use a stat
#' transform
#' * `ggsql_layer_sql`: A character string with the SQL query used by the layer
#' to fetch its data, or `NULL` if the layer doesn't have any data.
#' * `ggsql_stat_sql`: A character string with the SQL query used by the layers
#' stat transform, or `NULL` if the layer doesn't have a stat transform.
#' * `ggsql_warnings`: A data.frame with columns `message`, `line`, and `column`
#' giving the validation warnings for the spec
#'
#' @name spec_utility
#' @rdname spec_utility
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' spec <- ggsql_execute(reader,
#'   "SELECT * FROM cars VISUALISE mpg AS x DRAW histogram"
#' )
#'
#' ggsql_metadata(spec)
#'
#' ggsql_visual(spec)
#'
NULL

#' @rdname spec_utility
#' @export
ggsql_metadata <- function(spec) {
  rlang::check_required(spec)
  list(
    rows = spec$.ptr$metadata_rows(),
    columns = spec$.ptr$metadata_columns(),
    layer_count = spec$.ptr$metadata_layer_count()
  )
}

#' @rdname spec_utility
#' @export
ggsql_sql <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$get_sql()
}

#' @rdname spec_utility
#' @export
ggsql_visual <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$get_visual()
}

#' @rdname spec_utility
#' @export
ggsql_layer_count <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$layer_count()
}

#' @rdname spec_utility
#' @export
ggsql_layer_data <- function(spec, index = 1L) {
  rlang::check_required(spec)
  # Convert R 1-based to Rust 0-based
  ipc_bytes <- spec$.ptr$layer_data_ipc(as.integer(index - 1L))
  if (is.null(ipc_bytes)) {
    return(NULL)
  }
  ipc_to_df(ipc_bytes)
}

#' @rdname spec_utility
#' @export
ggsql_stat_data <- function(spec, index = 1L) {
  rlang::check_required(spec)
  ipc_bytes <- spec$.ptr$stat_data_ipc(as.integer(index - 1L))
  if (is.null(ipc_bytes)) {
    return(NULL)
  }
  ipc_to_df(ipc_bytes)
}

#' @rdname spec_utility
#' @export
ggsql_layer_sql <- function(spec, index = 1L) {
  rlang::check_required(spec)
  spec$.ptr$get_layer_sql(as.integer(index - 1L))
}

#' @rdname spec_utility
#' @export
ggsql_stat_sql <- function(spec, index = 1L) {
  rlang::check_required(spec)
  spec$.ptr$get_stat_sql(as.integer(index - 1L))
}

#' @rdname spec_utility
#' @export
ggsql_warnings <- function(spec) {
  rlang::check_required(spec)
  json <- spec$.ptr$warnings_json()
  warnings_list <- jsonlite::fromJSON(json)
  if (length(warnings_list) == 0) {
    return(data.frame(
      message = character(),
      line = integer(),
      column = integer()
    ))
  }
  warnings_list
}
