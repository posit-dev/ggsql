#' Create a DuckDB reader
#'
#' Creates a DuckDB database connection that can execute SQL queries and
#' register data frames as queryable tables. The default creates an empty
#' in-memory database but you can also pass the path to a DuckDB database to
#' directly interact with that.
#'
#' @param database Path to a DuckDB database file, or `NULL` (the default)
#' for an in-memory database.
#'
#' @return A `Reader` object.
#'
#' @export
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' df <- ggsql_execute_sql(reader, "SELECT mpg, disp FROM cars LIMIT 5")
#'
duckdb_reader <- function(database = NULL) {
  if (is.null(database)) {
    connection <- "duckdb://memory"
  } else {
    check_string(database)
    if (!file.exists(database)) {
      cli::cli_abort("Database file {.path {database}} does not exist.")
    }
    connection <- paste0("duckdb://", normalizePath(database))
  }
  Reader$new(connection)
}

#' @noRd
Reader <- R6::R6Class(
  "Reader",
  cloneable = FALSE,
  public = list(
    .ptr = NULL,

    initialize = function(connection) {
      self$.ptr <- GgsqlReader$new(connection)
    },

    print = function(...) {
      cli::cli_text("<ggsql_reader>")
      invisible(self)
    }
  )
)

#' Register and unregisters a data frame as a queryable table
#'
#' After registration, the data frame can be queried by name in SQL statements.
#' You can use `ggsql_table` to extract tables from the reader (both registered
#' ones and those native to the backend) and `ggsql_table_names` to get a vector
#' of all the tables in reader.
#'
#' @param reader A `Reader` object created by e.g. [duckdb_reader()].
#' @param df A data frame to register.
#' @param name The name of the table.
#' @param replace If `TRUE`, replace an existing table with the same name.
#' Defaults to `FALSE`.
#'
#' @return `reader` for `ggsql_register()` and `ggsql_unregister()`.
#' `ggsql_table` returns a data.frame if the table exists and `NULL` if not.
#' `ggsql_table_names` return a character vector.
#'
#' @export
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#'
#' ggsql_table_names(reader)
#'
#' ggsql_table(reader, "cars")
#'
#' ggsql_unregister(reader, "cars")
#'
#' ggsql_table_names(reader)
#'
ggsql_register <- function(reader, df, name, replace = FALSE) {
  rlang::check_required(reader)
  rlang::check_required(df)
  rlang::check_required(name)
  ipc_bytes <- df_to_ipc(df)
  reader$.ptr$register_ipc(name, ipc_bytes, replace)
  invisible(reader)
}

#' @rdname ggsql_register
#' @export
ggsql_unregister <- function(reader, name) {
  rlang::check_required(reader)
  rlang::check_required(name)
  reader$.ptr$unregister(name)
  invisible(reader)
}

#' @rdname ggsql_register
#' @export
ggsql_table <- function(reader, name) {
  if (name %in% ggsql_table_names(reader)) {
    safe_name <- gsub('"', '""', name, fixed = TRUE)
    ggsql_execute_sql(reader, paste0("SELECT * FROM \"", safe_name, "\""))
  } else {
    NULL
  }
}

#' @rdname ggsql_register
#' @export
ggsql_table_names <- function(reader) {
  ggsql_execute_sql(reader, "SHOW TABLES")$name
}

#' Execute a ggsql query
#'
#' Parses the query, and execute it against the reader's database. Returns
#' either a visualization specification ready for rendering (`ggsql_execute`) or
#' a data frame with the query result (`ggsql_execute_sql`).
#'
#' @param reader A `Reader` object created by [duckdb_reader()].
#' @param query A ggsql query string (SQL + VISUALISE clause).
#' @return A `Spec` object.
#'
#' @export
#'
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' spec <- ggsql_execute(reader,
#'   "SELECT * FROM cars VISUALISE mpg AS x, disp AS y DRAW point"
#' )
#'
ggsql_execute <- function(reader, query) {
  rlang::check_required(reader)
  rlang::check_required(query)
  spec_ptr <- reader$.ptr$execute(query)
  Spec$new(spec_ptr)
}

#' @rdname ggsql_execute
#' @export
ggsql_execute_sql <- function(reader, query) {
  rlang::check_required(reader)
  rlang::check_required(query)
  ipc_bytes <- reader$.ptr$execute_sql_ipc(query)
  ipc_to_df(ipc_bytes)
}

