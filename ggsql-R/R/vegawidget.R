#' Render a data frame as a vegawidget
#'
#' Convenience function that creates an in-memory DuckDB reader, registers
#' the data frame, executes the query, and returns a vegawidget.
#'
#' @param df A data frame.
#' @param viz VISUALISE spec string, e.g. `"VISUALISE x, y DRAW point"`.
#' @param connection DuckDB connection string. Defaults to in-memory.
#' @return A `vegaspec` object (from the vegawidget package).
#' @export
#' @examples
#' \dontrun{
#' render_vegawidget(mtcars, "VISUALISE mpg AS x, disp AS y DRAW point")
#' }
render_vegawidget <- function(df, viz, connection = "duckdb://memory") {
  rlang::check_required(df)
  rlang::check_required(viz)
  rlang::check_installed("vegawidget")

  reader <- duckdb_reader(connection)
  ggsql_register(reader, df, "__data__")

  query <- paste("SELECT * FROM __data__", viz)
  spec <- ggsql_execute(reader, query)

  writer <- vegalite_writer()
  json <- ggsql_render(writer, spec)

  vegawidget::as_vegaspec(json)
}
