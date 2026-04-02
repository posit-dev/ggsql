#' Validate a ggsql query
#'
#' Checks query syntax and semantics without executing SQL. Returns a
#' validation result that can be inspected for errors and warnings.
#'
#' @param query A ggsql query string.
#' @return A `ggsql_validated` object (a list with class).
#' @export
#' @examples
#' result <- ggsql_validate("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
#' result
ggsql_validate <- function(query) {
  rlang::check_required(query)
  result <- ggsql_validate_impl(query)

  result$errors <- jsonlite::fromJSON(result$errors_json)
  result$warnings <- jsonlite::fromJSON(result$warnings_json)
  result$errors_json <- NULL
  result$warnings_json <- NULL

  structure(result, class = "ggsql_validated")
}

#' @export
print.ggsql_validated <- function(x, ...) {
  status <- if (x$valid) "valid" else "invalid"
  cli::cli_text("<ggsql_validated> [{status}]")
  if (x$has_visual) {
    cli::cli_bullets(c("*" = "Has VISUALISE clause"))
  }
  if (length(x$errors) > 0 && NROW(x$errors) > 0) {
    cli::cli_text("Errors:")
    for (i in seq_len(NROW(x$errors))) {
      cli::cli_bullets(c("x" = x$errors$message[i]))
    }
  }
  if (length(x$warnings) > 0 && NROW(x$warnings) > 0) {
    cli::cli_text("Warnings:")
    for (i in seq_len(NROW(x$warnings))) {
      cli::cli_bullets(c("!" = x$warnings$message[i]))
    }
  }
  invisible(x)
}

#' Check if a validated query has a VISUALISE clause
#' @param x A `ggsql_validated` object.
#' @return Logical.
#' @export
ggsql_has_visual <- function(x) {
  x$has_visual
}

#' Check if a validated query is valid
#' @param x A `ggsql_validated` object.
#' @return Logical.
#' @export
ggsql_is_valid <- function(x) {
  x$valid
}
