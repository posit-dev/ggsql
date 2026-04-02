#' @noRd
GgsqlSpecR6 <- R6::R6Class(
  "GgsqlSpecR6",
  cloneable = FALSE,
  public = list(
    .ptr = NULL,

    initialize = function(ptr) {
      self$.ptr <- ptr
    },

    print = function(...) {
      m <- ggsql_metadata(self)
      cli::cli_text("<ggsql_spec>")
      cli::cli_bullets(c(
        "*" = "Rows: {m$rows}",
        "*" = "Columns: {paste(m$columns, collapse = ', ')}",
        "*" = "Layers: {m$layer_count}"
      ))
      invisible(self)
    }
  )
)

#' Get spec metadata
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return A list with elements `rows`, `columns`, and `layer_count`.
#' @export
ggsql_metadata <- function(spec) {
  rlang::check_required(spec)
  list(
    rows = spec$.ptr$metadata_rows(),
    columns = spec$.ptr$metadata_columns(),
    layer_count = spec$.ptr$metadata_layer_count()
  )
}

#' Get the SQL portion of a spec
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return A character string.
#' @export
ggsql_sql <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$get_sql()
}

#' Get the VISUALISE portion of a spec
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return A character string.
#' @export
ggsql_visual <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$get_visual()
}

#' Get the number of layers
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return An integer.
#' @export
ggsql_layer_count <- function(spec) {
  rlang::check_required(spec)
  spec$.ptr$layer_count()
}

#' Get layer data as a data frame
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @param index Layer index (1-based).
#' @return A data frame, or `NULL` if no data is available for this layer.
#' @export
ggsql_layer_data <- function(spec, index = 1L) {
  rlang::check_required(spec)
  # Convert R 1-based to Rust 0-based
  ipc_bytes <- spec$.ptr$layer_data_ipc(as.integer(index - 1L))
  if (is.null(ipc_bytes)) return(NULL)
  ipc_to_df(ipc_bytes)
}

#' Get stat transform data as a data frame
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @param index Layer index (1-based).
#' @return A data frame, or `NULL` if no stat transform for this layer.
#' @export
ggsql_stat_data <- function(spec, index = 1L) {
  rlang::check_required(spec)
  ipc_bytes <- spec$.ptr$stat_data_ipc(as.integer(index - 1L))
  if (is.null(ipc_bytes)) return(NULL)
  ipc_to_df(ipc_bytes)
}

#' Get the SQL for a specific layer
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @param index Layer index (1-based).
#' @return A character string, or `NULL`.
#' @export
ggsql_layer_sql <- function(spec, index = 1L) {
  rlang::check_required(spec)
  spec$.ptr$get_layer_sql(as.integer(index - 1L))
}

#' Get the stat transform SQL for a specific layer
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @param index Layer index (1-based).
#' @return A character string, or `NULL`.
#' @export
ggsql_stat_sql <- function(spec, index = 1L) {
  rlang::check_required(spec)
  spec$.ptr$get_stat_sql(as.integer(index - 1L))
}

#' Get validation warnings from a spec
#'
#' @param spec A `GgsqlSpecR6` object returned by [ggsql_execute()].
#' @return A data frame with columns `message`, `line`, and `column`, or an
#'   empty data frame if there are no warnings.
#' @export
ggsql_warnings <- function(spec) {
  rlang::check_required(spec)
  json <- spec$.ptr$warnings_json()
  warnings_list <- jsonlite::fromJSON(json)
  if (length(warnings_list) == 0) {
    return(data.frame(message = character(), line = integer(), column = integer()))
  }
  warnings_list
}
