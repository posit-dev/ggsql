#' Create a Vega-Lite writer
#'
#' @return A `GgsqlWriterR6` object.
#' @export
vegalite_writer <- function() {
  GgsqlWriterR6$new()
}

#' @noRd
GgsqlWriterR6 <- R6::R6Class(
  "GgsqlWriterR6",
  cloneable = FALSE,
  public = list(
    .ptr = NULL,

    initialize = function() {
      self$.ptr <- GgsqlWriter$new()
    },

    print = function(...) {
      cli::cli_text("<ggsql_writer> [vegalite]")
      invisible(self)
    }
  )
)

#' Render a spec to Vega-Lite JSON
#'
#' @param writer A `GgsqlWriterR6` object created by [vegalite_writer()].
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return A Vega-Lite JSON string.
#' @export
ggsql_render <- function(writer, spec) {
  rlang::check_required(writer)
  rlang::check_required(spec)
  writer$.ptr$render(spec$.ptr)
}
