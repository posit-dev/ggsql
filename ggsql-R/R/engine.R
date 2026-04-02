# Package-level environment for persistent reader across knitr chunks
ggsql_env <- new.env(parent = emptyenv())

get_engine_reader <- function() {
  if (is.null(ggsql_env$reader)) {
    ggsql_env$reader <- duckdb_reader()
  }
  ggsql_env$reader
}

ggsql_engine <- function(options) {
  if (!options$eval) {
    return(knitr::engine_output(options, options$code, ""))
  }

  query <- paste(options$code, collapse = "\n")
  reader <- get_engine_reader()

  result <- tryCatch(
    ggsql_engine_eval(query, reader, options),
    error = function(cnd) {
      knitr::engine_output(options, options$code, conditionMessage(cnd))
    }
  )

  result
}

ggsql_engine_eval <- function(query, reader, options) {
  validated <- ggsql_validate(query)

  if (!validated$has_visual) {
    # Plain SQL: execute and render as table
    df <- ggsql_execute_sql(reader, query)
    # Suppress output for DDL/DML statements that return metadata rows
    # (e.g., COPY TO returns a "Count" column)
    is_result <- nrow(df) > 0 && ncol(df) > 0 &&
      !identical(names(df), "Count")
    if (!is_result) {
      return(knitr::engine_output(options, options$code, ""))
    }
    out <- knitr::kable(df)
    options$results <- "asis"
    return(knitr::engine_output(options, options$code, out))
  }

  # Visualization query: execute, render to Vega-Lite, display as widget
  spec <- ggsql_execute(reader, query)
  writer <- vegalite_writer()
  json <- ggsql_render(writer, spec)

  options$results <- "asis"

  widget <- vegawidget::as_vegaspec(json)
  out <- knitr::knit_print(widget, options = options)

  # When we cannot include widgets, we are handed a screenshot that we
  # must include as a png file (e.g. in PDF output)
  if (inherits(out, "html_screenshot")) {
    file_path <- knitr::sew(out, options = options)
    return(knitr::engine_output(options, out = list(file_path)))
  }

  # Add metadata manually, since we're not using the usual hooks.
  # This ensures the dependencies (<script> tags) are listed properly
  # in the output html file.
  meta <- attr(out, "knit_meta", exact = TRUE)
  knitr::knit_meta_add(meta)

  knitr::engine_output(options, options$code, out = out)
}

on_load(
  knitr::knit_engines$set(ggsql = ggsql_engine)
)
