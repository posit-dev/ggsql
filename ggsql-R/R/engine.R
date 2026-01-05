
ggsql_engine <- function(options) {
  # Write a temporary file with the query
  tmp <- basename(tempfile("ggsql", ".", paste0(".", "gsql")))
  on.exit(unlink(tmp))
  writeLines(options$code, tmp)

  out <- ''

  # Format and evaluate system command
  if (options$eval) {
    cmd <- sprintf("ggsql run %s", paste(tmp, options$engine.opts))
    out <- system(cmd, intern=TRUE)
  }

  # If the command failed, we don't have a vegalite schema and we exit early
  is_schema <- !is.na(out[2]) && startsWith(trimws(out[2]), "\"$schema")
  if (!is_schema) {
    return(knitr::engine_output(options, options$code, out))
  }

  # Interpret output as-is, i.e. raw html/pandoc
  options$results <- "asis"

  # Render the spec as html chunk to include
  widget <- vegawidget::as_vegaspec(paste0(out, collapse = "\n"))
  out <- knitr::knit_print(widget, options = options)

  # When we cannot include widgets, we are handed a screenshot that we
  # must include as a png file.
  # This happens in static formats, like PDF
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
