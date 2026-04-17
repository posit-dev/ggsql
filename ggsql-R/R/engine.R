# Package-level environment for persistent reader across knitr chunks
ggsql_env <- new.env(parent = emptyenv())

get_engine_reader <- function(connection = NULL) {
  if (is.null(connection)) {
    # Default in-memory DuckDB reader
    if (is.null(ggsql_env$reader)) {
      ggsql_env$reader <- duckdb_reader()
      # Inject the sql proxy into the knit environment for cross-chunk access
      proxy <- new_ggsql_tables()
      assign("sql", proxy, envir = knitr::knit_global())
      # Also inject into Python if reticulate is available, so Python chunks
      # can use sql.tablename directly (without the r. prefix)
      if (
        requireNamespace("reticulate", quietly = TRUE) &&
          reticulate::py_available(initialize = FALSE)
      ) {
        reticulate::py_run_string("pass") # ensure Python is initialized
        py <- reticulate::py
        py[["sql"]] <- proxy
      }
    }
    return(ggsql_env$reader)
  }

  # Custom connection: cache by connection string
  if (is.null(ggsql_env$readers)) {
    ggsql_env$readers <- list()
  }
  if (is.null(ggsql_env$readers[[connection]])) {
    ggsql_env$readers[[connection]] <- parse_connection(connection)
  }
  ggsql_env$readers[[connection]]
}

register_syntax_highlighting <- function() {
  syntax_file <- system.file("ggsql.xml", package = "ggsql")
  if (!nzchar(syntax_file)) {
    return()
  }

  # For rmarkdown: add --syntax-definition to Pandoc args
  current <- knitr::opts_knit$get("rmarkdown.pandoc.args")
  if (!syntax_file %in% current) {
    knitr::opts_knit$set(
      rmarkdown.pandoc.args = c(current, "--syntax-definition", syntax_file)
    )
  }
}

# ---------------------------------------------------------------------------
# Connection string parsing
# ---------------------------------------------------------------------------

parse_connection <- function(connection) {
  check_string(connection)

  if (!grepl("://", connection, fixed = TRUE)) {
    cli::cli_abort(
      "Invalid connection string {.val {connection}}. Expected format: {.code scheme://...} (e.g., {.code duckdb://memory})."
    )
  }

  scheme <- sub("://.*", "", connection)

  switch(
    tolower(scheme),
    duckdb = Reader$new(connection),
    cli::cli_abort(
      "Unsupported connection scheme {.val {scheme}}. Supported schemes: {.val duckdb}."
    )
  )
}

# ---------------------------------------------------------------------------
# sql proxy: live access to tables in the ggsql DuckDB reader
# ---------------------------------------------------------------------------

new_ggsql_tables <- function() {
  structure(list(), class = "ggsql_tables")
}

#' @export
`$.ggsql_tables` <- function(x, name) {
  safe_name <- gsub('"', '""', name, fixed = TRUE)
  ggsql_execute_sql(get_engine_reader(), paste0('SELECT * FROM "', safe_name, '"'))
}

#' @export
`[[.ggsql_tables` <- function(x, name, ...) {
  safe_name <- gsub('"', '""', name, fixed = TRUE)
  ggsql_execute_sql(get_engine_reader(), paste0('SELECT * FROM "', safe_name, '"'))
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
  refs <- gregexpr(
    "(?:r|py):[a-zA-Z_][a-zA-Z0-9_.]*",
    query,
    ignore.case = TRUE,
    perl = TRUE
  )
  matches <- regmatches(query, refs)[[1]]

  if (length(matches) == 0) {
    return(query)
  }

  for (ref in unique(matches)) {
    parts <- strsplit(ref, ":", fixed = TRUE)[[1]]
    prefix <- parts[1]
    name <- parts[2]

    df <- switch(
      tolower(prefix),
      r = try_fetch(
        get(name, envir = knitr::knit_global()),
        error = function(cnd) {
          cli::cli_abort(
            "Column reference {.code {ref}}: object {.val {name}} not found in R environment."
          )
        }
      ),
      py = {
        rlang::check_installed(
          "reticulate",
          reason = "to use py: data references."
        )
        obj <- reticulate::py[[name]]
        if (is.null(obj)) {
          cli::cli_abort(
            "Column reference {.code {ref}}: object {.val {name}} not found in Python environment."
          )
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
# Vega-Lite HTML rendering
# ---------------------------------------------------------------------------

vegalite_html <- function(
  spec_json,
  width = NULL,
  height = NULL,
  caption = NULL,
  align = "center"
) {
  ggsql_env$vis_counter <- (ggsql_env$vis_counter %||% 0L) + 1L
  vis_id <- paste0("ggsql-vis-", ggsql_env$vis_counter)

  # Convert fig.width/fig.height (inches) to pixels at 96 dpi,
  # or use defaults if not specified
  css_width <- if (!is.null(width)) {
    if (is.numeric(width)) paste0(round(width * 96), "px") else width
  } else {
    "100%"
  }
  css_height <- if (!is.null(height)) {
    if (is.numeric(height)) paste0(round(height * 96), "px") else height
  } else {
    "400px"
  }

  margin_style <- switch(
    align %||% "center",
    center = "margin-left: auto; margin-right: auto;",
    right = "margin-left: auto;",
    ""
  )

  html <- sprintf(
    '<div id="%s-outer" style="width: %s; overflow: hidden; %s">
<div id="%s" style="width: 100%%; min-width: 450px; height: %s;"></div>
</div>

<script type="text/javascript">
(function() {
  const spec = %s;
  const visId = "%s";
  const minWidth = 450;

  if (!window.__ggsql_vega_ready) {
    const loadScript = (src) => new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
    window.__ggsql_vega_ready = loadScript("https://cdn.jsdelivr.net/npm/vega@6/build/vega.min.js")
      .then(() => loadScript("https://cdn.jsdelivr.net/npm/vega-lite@6/build/vega-lite.min.js"))
      .then(() => loadScript("https://cdn.jsdelivr.net/npm/vega-embed@7/build/vega-embed.min.js"));
  }

  function scaleToFit(outer, inner) {
    const available = outer.clientWidth;
    if (available < minWidth) {
      const scale = available / minWidth;
      inner.style.transform = "scale(" + scale + ")";
      inner.style.transformOrigin = "top left";
      outer.style.height = (inner.scrollHeight * scale) + "px";
    } else {
      inner.style.transform = "";
      outer.style.height = "";
    }
  }

  window.__ggsql_vega_ready
    .then(() => vegaEmbed("#" + visId, spec, {"actions": true}))
    .then(() => {
      const outer = document.getElementById(visId + "-outer");
      const inner = document.getElementById(visId);
      scaleToFit(outer, inner);
      const ro = new ResizeObserver(() => scaleToFit(outer, inner));
      ro.observe(outer);
    })
    .catch(err => {
      document.getElementById(visId).innerText = "Failed to load Vega: " + err;
    });
})();
</script>',
    vis_id,
    css_width,
    margin_style,
    vis_id,
    css_height,
    spec_json,
    vis_id
  )

  if (!is.null(caption) && nzchar(caption)) {
    html <- sprintf(
      '<figure>\n%s\n<figcaption>%s</figcaption>\n</figure>',
      html,
      htmltools::htmlEscape(caption)
    )
  }

  html
}

# ---------------------------------------------------------------------------
# Inline chunk options (--| and #| prefix support)
# ---------------------------------------------------------------------------

parse_chunk_options <- function(options) {
  code <- options$code
  if (length(code) == 0) {
    return(options)
  }

  # Extract leading lines with --| or #| prefix
  option_pattern <- "^\\s*(?:--|#)\\|\\s?"
  is_option <- grepl(option_pattern, code, perl = TRUE)

  # Only consider contiguous leading option lines
  n_options <- 0L
  for (i in seq_along(is_option)) {
    if (!is_option[i]) {
      break
    }
    n_options <- i
  }

  if (n_options == 0L) {
    return(options)
  }

  # Extract option text (strip the prefix)
  option_lines <- sub(option_pattern, "", code[seq_len(n_options)], perl = TRUE)
  options$code <- code[-seq_len(n_options)]

  # Parse as YAML
  yaml_text <- paste(option_lines, collapse = "\n")
  parsed <- tryCatch(
    yaml::yaml.load(yaml_text),
    error = function(cnd) NULL
  )

  if (is.null(parsed) || !is.list(parsed)) {
    return(options)
  }

  # Merge into options (inline options override chunk header options)
  for (nm in names(parsed)) {
    # Convert Quarto-style kebab-case to knitr-style dot-case
    knitr_nm <- gsub("-", ".", nm, fixed = TRUE)
    options[[knitr_nm]] <- parsed[[nm]]
  }

  options
}

# ---------------------------------------------------------------------------
# knitr engine
# ---------------------------------------------------------------------------

ggsql_engine <- function(options) {
  # Parse inline chunk options (--| or #| prefixed lines)
  options <- parse_chunk_options(options)

  # Use SQL syntax highlighting for the source code block.
  # ggsql-specific highlighting requires adding ggsql.xml to the Quarto/Pandoc
  # config (see inst/ggsql.xml). SQL covers the base keywords well.
  options$class.source <- options$class.source %||% "sql"

  # Always register syntax highlighting, even for custom connections
  register_syntax_highlighting()

  if (!options$eval) {
    return(knitr::engine_output(options, options$code, ""))
  }

  query <- paste(options$code, collapse = "\n")

  result <- try_fetch(
    {
      reader <- get_engine_reader(options$connection)
      ggsql_engine_eval(query, reader, options)
    },
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
    is_result <- nrow(df) > 0 && ncol(df) > 0 && !identical(names(df), "Count")
    if (!is_result) {
      return(knitr::engine_output(options, options$code, ""))
    }
    out <- knitr::kable(df)
    options$results <- "asis"
    return(knitr::engine_output(options, options$code, out))
  }

  # Visualization query: execute and render
  spec <- ggsql_execute(reader, query)
  writer_type <- options$writer %||% "vegalite"

  # If output.var is set, always capture the Vega-Lite JSON
  if (!is.null(options$output.var)) {
    writer <- vegalite_writer()
    json <- ggsql_render(writer, spec)
    assign(options$output.var, json, envir = knitr::knit_global())
    return(knitr::engine_output(options, options$code, ""))
  }

  options$results <- "asis"

  switch(
    writer_type,
    vegalite = {
      # Embed Vega-Lite spec directly with vega-embed from CDN.
      # This avoids vegawidget version constraints (ggsql uses Vega-Lite v6).
      writer <- vegalite_writer()
      json <- ggsql_render(writer, spec)
      out <- vegalite_html(
        json,
        width = options$fig.width,
        height = options$fig.height,
        caption = options$fig.cap,
        align = options$fig.align
      )
      knitr::engine_output(options, options$code, out = out)
    },
    vegalite_svg = render_static_figure(spec, "svg", options),
    vegalite_png = render_static_figure(spec, "png", options),
    cli::cli_abort(
      c(
        "Unsupported writer {.val {writer_type}}.",
        i = "Supported writers: {.val vegalite}, {.val vegalite_svg}, {.val vegalite_png}."
      )
    )
  )
}

write_static_figure <- function(spec, format, options) {
  options$label <- options$label %||% "ggsql-chunk"
  ext <- paste0(".", format)
  fig <- knitr::fig_path(ext, options, number = 1L)
  dir.create(dirname(fig), recursive = TRUE, showWarnings = FALSE)

  switch(
    format,
    svg = writeLines(ggsql_to_svg(spec), fig),
    png = writeBin(ggsql_to_png(spec), fig)
  )

  cap <- options$fig.cap %||% ""
  sprintf("\n![%s](%s)\n", cap, fig)
}

render_static_figure <- function(spec, format, options) {
  out <- write_static_figure(spec, format, options)
  knitr::engine_output(options, options$code, out)
}

on_load(
  knitr::knit_engines$set(ggsql = ggsql_engine)
)
