#' Create a Vega-Lite writer
#'
#' This function creates a vegalite writer which is currently the only writer
#' type for ggsql
#'
#' @return A `Writer` object.
#'
#' @export
#'
#' @examples
#' vegalite_writer()
#'
vegalite_writer <- function() {
  Writer$new()
}

#' @noRd
Writer <- R6::R6Class(
  "Writer",
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

#' Render a spec with a writer
#'
#' This function takes a `Spec` object as returned by [ggsql_execute()] and
#' renders it with the provided writer.
#'
#' @param writer A `Writer` object created by e.g. [vegalite_writer()].
#' @param spec A `Spec` object returned by [ggsql_execute()].
#'
#' @return Writer dependent:
#'
#' * `vegalite_writer`: A string holding the vegalite JSON representation of the
#' visualization
#'
#' @export
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' spec <- ggsql_execute(reader,
#'   "SELECT * FROM cars VISUALISE mpg AS x DRAW histogram"
#' )
#'
#' ggsql_render(vegalite_writer(), spec)
#'
ggsql_render <- function(writer, spec) {
  rlang::check_required(writer)
  rlang::check_required(spec)
  writer$.ptr$render(spec$.ptr)
}
