#' Create a DuckDB reader
#'
#' Creates a DuckDB database connection that can execute SQL queries and
#' register data frames as queryable tables.
#'
#' @param connection Connection string. Use `"duckdb://memory"` for an
#'   in-memory database or `"duckdb:///path/to/file.db"` for a file-based
#'   database.
#' @return A `GgsqlReaderR6` object.
#' @export
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' df <- ggsql_execute_sql(reader, "SELECT mpg, disp FROM cars LIMIT 5")
duckdb_reader <- function(connection = "duckdb://memory") {
  rlang::check_required(connection)
  GgsqlReaderR6$new(connection)
}

#' @noRd
GgsqlReaderR6 <- R6::R6Class(
  "GgsqlReaderR6",
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

#' Register a data frame as a queryable table
#'
#' After registration, the data frame can be queried by name in SQL statements.
#'
#' @param reader A `GgsqlReaderR6` object created by [duckdb_reader()].
#' @param df A data frame to register.
#' @param name The table name to register under.
#' @param replace If `TRUE`, replace an existing table with the same name.
#'   Defaults to `FALSE`.
#' @return `reader`, invisibly (for piping).
#' @export
ggsql_register <- function(reader, df, name, replace = FALSE) {
  rlang::check_required(reader)
  rlang::check_required(df)
  rlang::check_required(name)
  ipc_bytes <- df_to_ipc(df)
  reader$.ptr$register_ipc(name, ipc_bytes, replace)
  invisible(reader)
}

#' Unregister a previously registered table
#'
#' @param reader A `GgsqlReaderR6` object created by [duckdb_reader()].
#' @param name The table name to unregister.
#' @return `reader`, invisibly (for piping).
#' @export
ggsql_unregister <- function(reader, name) {
  rlang::check_required(reader)
  rlang::check_required(name)
  reader$.ptr$unregister(name)
  invisible(reader)
}

#' Execute a ggsql query
#'
#' Parses the query, executes the SQL portion against the reader's database,
#' and returns a visualization specification ready for rendering.
#'
#' @param reader A `GgsqlReaderR6` object created by [duckdb_reader()].
#' @param query A ggsql query string (SQL + VISUALISE clause).
#' @return A `GgsqlSpecR6` object.
#' @export
#' @examples
#' reader <- duckdb_reader()
#' ggsql_register(reader, mtcars, "cars")
#' spec <- ggsql_execute(reader,
#'   "SELECT * FROM cars VISUALISE mpg AS x, disp AS y DRAW point"
#' )
ggsql_execute <- function(reader, query) {
  rlang::check_required(reader)
  rlang::check_required(query)
  spec_ptr <- reader$.ptr$execute(query)
  GgsqlSpecR6$new(spec_ptr)
}

#' Execute SQL and return a data frame
#'
#' Executes a plain SQL query and returns the result as a data frame.
#'
#' @param reader A `GgsqlReaderR6` object created by [duckdb_reader()].
#' @param sql A SQL query string.
#' @return A data frame.
#' @export
ggsql_execute_sql <- function(reader, sql) {
  rlang::check_required(reader)
  rlang::check_required(sql)
  ipc_bytes <- reader$.ptr$execute_sql_ipc(sql)
  ipc_to_df(ipc_bytes)
}
