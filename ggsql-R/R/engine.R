# Package-level environment for persistent reader across knitr chunks
ggsql_env <- new.env(parent = emptyenv())

get_engine_reader <- function() {
  if (is.null(ggsql_env$reader)) {
    ggsql_env$reader <- duckdb_reader()
    # Inject the sql proxy into the knit environment for cross-chunk access
    proxy <- new_ggsql_tables()
    assign("sql", proxy, envir = knitr::knit_global())
    # Also inject into Python if reticulate is available, so Python chunks
    # can use sql.tablename directly (without the r. prefix)
    if (requireNamespace("reticulate", quietly = TRUE) &&
        reticulate::py_available(initialize = FALSE)) {
      reticulate::py_run_string("pass")  # ensure Python is initialized
      reticulate::py[["sql"]] <- proxy
    }
    # Register ggsql syntax highlighting with Pandoc
    register_syntax_highlighting()
  }
  ggsql_env$reader
}

register_syntax_highlighting <- function() {
  syntax_file <- system.file("ggsql.xml", package = "ggsql")
  if (!nzchar(syntax_file)) return()

  # For rmarkdown: add --syntax-definition to Pandoc args
  current <- knitr::opts_knit$get("rmarkdown.pandoc.args")
  if (!syntax_file %in% current) {
    knitr::opts_knit$set(
      rmarkdown.pandoc.args = c(current, "--syntax-definition", syntax_file)
    )
  }
}

# ---------------------------------------------------------------------------
# sql proxy: live access to tables in the ggsql DuckDB reader
# ---------------------------------------------------------------------------

new_ggsql_tables <- function() {
  structure(list(), class = "ggsql_tables")
}

#' @export
`$.ggsql_tables` <- function(x, name) {
  ggsql_execute_sql(get_engine_reader(), paste0('SELECT * FROM "', name, '"'))
}

#' @export
`[[.ggsql_tables` <- function(x, name, ...) {
  ggsql_execute_sql(get_engine_reader(), paste0('SELECT * FROM "', name, '"'))
}

#' @export
print.ggsql_tables <- function(x, ...) {
  reader <- get_engine_reader()
  tables <- try_fetch(
    ggsql_execute_sql(reader, "SHOW TABLES"),
    error = function(cnd) data.frame(name = character())
  )
  cli::cli_text("<ggsql tables>")
  if (nrow(tables) > 0) {
    cli::cli_bullets(stats::setNames(tables[[1]], rep("*", nrow(tables))))
  } else {
    cli::cli_text("(no tables)")
  }
  invisible(x)
}

#' @export
names.ggsql_tables <- function(x) {
  reader <- get_engine_reader()
  tables <- try_fetch(
    ggsql_execute_sql(reader, "SHOW TABLES"),
    error = function(cnd) data.frame(name = character())
  )
  if (nrow(tables) > 0) tables[[1]] else character()
}

# ---------------------------------------------------------------------------
# Data reference resolution (r: and py: prefixes)
# ---------------------------------------------------------------------------

resolve_data_refs <- function(query, reader) {
  refs <- gregexpr("(?:r|py):[a-zA-Z_][a-zA-Z0-9_.]*", query, ignore.case = TRUE, perl = TRUE)
  matches <- regmatches(query, refs)[[1]]

  if (length(matches) == 0) return(query)

  for (ref in unique(matches)) {
    parts <- strsplit(ref, ":", fixed = TRUE)[[1]]
    prefix <- parts[1]
    name <- parts[2]

    df <- switch(prefix,
      r = try_fetch(
        get(name, envir = knitr::knit_global()),
        error = function(cnd) {
          cli::cli_abort("Column reference {.code {ref}}: object {.val {name}} not found in R environment.")
        }
      ),
      py = {
        rlang::check_installed("reticulate", reason = "to use py: data references.")
        obj <- reticulate::py[[name]]
        if (is.null(obj)) {
          cli::cli_abort("Column reference {.code {ref}}: object {.val {name}} not found in Python environment.")
        }
        obj
      }
    )

    if (!is.data.frame(df)) {
      cli::cli_abort("{.code {ref}} does not refer to a data frame.")
    }

    internal_name <- paste0("__", prefix, "_", name, "__")
    ggsql_register(reader, df, internal_name, replace = TRUE)
    query <- gsub(ref, internal_name, query, fixed = TRUE)
  }

  query
}

# ---------------------------------------------------------------------------
# Vega-Lite HTML rendering (bypasses vegawidget for v6 compatibility)
# ---------------------------------------------------------------------------

vegalite_html <- function(spec_json, width = NULL, height = NULL, caption = NULL) {
  ggsql_env$vis_counter <- (ggsql_env$vis_counter %||% 0L) + 1L
  vis_id <- paste0("ggsql-vis-", ggsql_env$vis_counter)

  # Convert fig.width/fig.height (inches) to pixels at 96 dpi,
  # or use defaults if not specified
  css_width <- if (!is.null(width)) paste0(round(width * 96), "px") else "100%"
  css_height <- if (!is.null(height)) paste0(round(height * 96), "px") else "400px"

  html <- sprintf(
    '<div id="%s" style="width: %s; height: %s;"></div>
<script type="text/javascript">
(function() {
  const spec = %s;
  const visId = "%s";

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      // Reuse already-loaded scripts
      if (document.querySelector(\'script[src="\' + src + \'"]\')) {
        return resolve();
      }
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  loadScript("https://cdn.jsdelivr.net/npm/vega@6/build/vega.min.js")
    .then(() => loadScript("https://cdn.jsdelivr.net/npm/vega-lite@6/build/vega-lite.min.js"))
    .then(() => loadScript("https://cdn.jsdelivr.net/npm/vega-embed@7/build/vega-embed.min.js"))
    .then(() => vegaEmbed("#" + visId, spec, {"actions": true}))
    .catch(err => {
      document.getElementById(visId).innerText = "Failed to load Vega: " + err;
    });
})();
</script>',
    vis_id, css_width, css_height, spec_json, vis_id
  )

  if (!is.null(caption) && nzchar(caption)) {
    html <- sprintf(
      '<figure>\n%s\n<figcaption>%s</figcaption>\n</figure>',
      html, htmltools::htmlEscape(caption)
    )
  }

  html
}

# ---------------------------------------------------------------------------
# knitr engine
# ---------------------------------------------------------------------------

ggsql_engine <- function(options) {
  # Use SQL syntax highlighting for the source code block.
  # ggsql-specific highlighting requires adding ggsql.xml to the Quarto/Pandoc
  # config (see inst/ggsql.xml). SQL covers the base keywords well.
  options$class.source <- options$class.source %||% "sql"

  if (!options$eval) {
    return(knitr::engine_output(options, options$code, ""))
  }

  query <- paste(options$code, collapse = "\n")
  reader <- get_engine_reader()

  result <- try_fetch(
    ggsql_engine_eval(query, reader, options),
    error = function(cnd) {
      knitr::engine_output(options, options$code, conditionMessage(cnd))
    }
  )

  result
}

ggsql_engine_eval <- function(query, reader, options) {
  query <- resolve_data_refs(query, reader)
  validated <- ggsql_validate(query)

  if (!validated$has_visual) {
    # Plain SQL: execute and render as table
    df <- ggsql_execute_sql(reader, query)

    # If output.var is set, assign to knit environment instead of rendering
    if (!is.null(options$output.var)) {
      assign(options$output.var, df, envir = knitr::knit_global())
      return(knitr::engine_output(options, options$code, ""))
    }

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

  # If output.var is set, assign the Vega-Lite JSON
  if (!is.null(options$output.var)) {
    assign(options$output.var, json, envir = knitr::knit_global())
    return(knitr::engine_output(options, options$code, ""))
  }

  options$results <- "asis"

  # Embed Vega-Lite spec directly with vega-embed from CDN.
  # This avoids vegawidget version constraints (ggsql uses Vega-Lite v6).
  out <- vegalite_html(
    json,
    width = options$fig.width,
    height = options$fig.height,
    caption = options$fig.cap
  )
  knitr::engine_output(options, options$code, out = out)
}

on_load(
  knitr::knit_engines$set(ggsql = ggsql_engine)
)
