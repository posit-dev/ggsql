run_query <- function(query, ...) {
  opts <- knitr::opts_current$get()
  opts$code <- query
  extra_opts <- list(...)
  if (length(extra_opts) > 0) {
    opts[names(extra_opts)] <- extra_opts
  }
  ggsql_engine(opts)
}

data_file <- "mtcars.csv"
on.exit(unlink(data_file))
write.csv(mtcars, data_file)

test_that("engine can handle a query", {
  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(query, dev = "png")

  # We expect path to png file here, since output format for knitr is undetermined
  expect_type(out, "character")
  expect_length(out, 1L)
})

test_that("engine can handle a query without visualisation statement", {
  query <- paste0("SELECT mpg, disp FROM '", data_file, "'")

  out <- run_query(query)
  expect_snapshot(cat(out))
})

test_that("engine does not return a table when merely creating data", {
  query <-
    "COPY (
      SELECT * FROM (VALUES
          (5.2, 18.5),
          (8.7, 22.3)
      ) AS t(x, y)
    ) TO 'data.csv' (HEADER, DELIMITER ',')"
  out <- run_query(query)
  expect_snapshot(cat(out))
})

# --- Data reference tests (r: and py: prefixes) ---

test_that("r: prefix resolves R objects", {
  # Put data into the knitr global environment
  assign("test_df", mtcars[1:5, c("mpg", "disp")], envir = knitr::knit_global())
  on.exit(rm("test_df", envir = knitr::knit_global()))

  query <- "SELECT mpg, disp FROM r:test_df"
  out <- run_query(query)
  expect_snapshot(cat(out))
})

test_that("r: prefix works in visualisation queries", {
  assign("test_df", mtcars[1:5, c("mpg", "disp")], envir = knitr::knit_global())
  on.exit(rm("test_df", envir = knitr::knit_global()))

  query <- c(
    "SELECT * FROM r:test_df",
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(query, dev = "png")
  expect_type(out, "character")
  expect_length(out, 1L)
})

test_that("r: prefix errors for missing objects", {
  query <- "SELECT * FROM r:nonexistent_object_xyz"
  out <- run_query(query)
  expect_match(out, "not found", ignore.case = TRUE)
})

test_that("r: prefix errors for non-data-frame objects", {
  assign("not_a_df", "just a string", envir = knitr::knit_global())
  on.exit(rm("not_a_df", envir = knitr::knit_global()))

  query <- "SELECT * FROM r:not_a_df"
  out <- run_query(query)
  expect_match(out, "data frame", ignore.case = TRUE)
})

test_that("multiple r: refs in one query work", {
  assign("df_a", data.frame(id = 1:3, x = 10:12), envir = knitr::knit_global())
  assign("df_b", data.frame(id = 1:3, y = 20:22), envir = knitr::knit_global())
  on.exit({
    rm("df_a", envir = knitr::knit_global())
    rm("df_b", envir = knitr::knit_global())
  })

  query <- "SELECT a.id, a.x, b.y FROM r:df_a a JOIN r:df_b b ON a.id = b.id"
  out <- run_query(query)
  expect_snapshot(cat(out))
})

# --- output.var and sql proxy tests ---

test_that("output.var captures SQL result as data frame", {
  query <- "SELECT 1 AS x, 2 AS y"
  run_query(query, output.var = "captured_df")
  df <- get("captured_df", envir = knitr::knit_global())
  on.exit(rm("captured_df", envir = knitr::knit_global()))

  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 1)
  expect_equal(names(df), c("x", "y"))
})

test_that("output.var captures Vega-Lite JSON for viz queries", {
  assign("test_df", mtcars[1:5, c("mpg", "disp")], envir = knitr::knit_global())
  on.exit(
    rm(list = c("test_df", "captured_json"), envir = knitr::knit_global()),
    add = TRUE
  )

  query <- c(
    "SELECT * FROM r:test_df",
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  run_query(query, output.var = "captured_json")
  json <- get("captured_json", envir = knitr::knit_global())

  expect_type(json, "character")
  expect_match(json, "vega-lite")
})

test_that("sql proxy can access registered tables", {
  reader <- get_engine_reader()
  ggsql_register(reader, mtcars[1:3, c("mpg", "disp")], "proxy_test")

  sql_obj <- get("sql", envir = knitr::knit_global())
  df <- sql_obj$proxy_test
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 3)
  expect_equal(names(df), c("mpg", "disp"))
})

test_that("sql proxy names() lists tables", {
  reader <- get_engine_reader()
  ggsql_register(reader, data.frame(a = 1), "names_test")

  sql_obj <- get("sql", envir = knitr::knit_global())
  tbl_names <- names(sql_obj)
  expect_true("names_test" %in% tbl_names)
})

# --- Inline chunk options (--| and #|) ---

test_that("--| prefix parses chunk options", {
  query <- c(
    "--| output.var: my_result",
    "SELECT 1 AS x, 2 AS y"
  )
  run_query(query)
  df <- get("my_result", envir = knitr::knit_global())
  on.exit(rm("my_result", envir = knitr::knit_global()))

  expect_s3_class(df, "data.frame")
  expect_equal(names(df), c("x", "y"))
})

test_that("#| prefix parses chunk options", {
  query <- c(
    "#| output.var: my_result2",
    "SELECT 1 AS a, 2 AS b"
  )
  run_query(query)
  df <- get("my_result2", envir = knitr::knit_global())
  on.exit(rm("my_result2", envir = knitr::knit_global()))

  expect_s3_class(df, "data.frame")
  expect_equal(names(df), c("a", "b"))
})

test_that("Quarto-style kebab-case options are converted", {
  query <- c(
    "--| output-var: my_result3",
    "SELECT 10 AS val"
  )
  run_query(query)
  df <- get("my_result3", envir = knitr::knit_global())
  on.exit(rm("my_result3", envir = knitr::knit_global()))

  expect_s3_class(df, "data.frame")
})

# --- Connection option tests ---

test_that("connection option creates a DuckDB reader", {
  query <- "SELECT 1 AS x, 2 AS y"
  out <- run_query(query, connection = "duckdb://memory")
  expect_snapshot(cat(out))
})

test_that("connection option rejects unsupported schemes", {
  query <- "SELECT 1 AS x"
  out <- run_query(query, connection = "mysql://localhost")
  expect_match(out, "Unsupported connection scheme")
})

test_that("connection option rejects invalid format", {
  query <- "SELECT 1 AS x"
  out <- run_query(query, connection = "not-a-uri")
  expect_match(out, "Invalid connection string")
})

# --- Writer option tests ---

test_that("writer defaults to interactive vegalite", {
  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(query)
  expect_match(out, "vega-embed")
})

test_that("writer = 'vegalite_svg' produces SVG output", {
  skip_if_not_installed("V8")
  skip_on_cran()
  skip_if_not_installed("withr")

  fig_dir <- withr::local_tempdir()

  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(
    query,
    writer = "vegalite_svg",
    fig.path = paste0(fig_dir, "/fig-"),
    label = "test-svg"
  )

  svg_files <- list.files(fig_dir, pattern = "\\.svg$")
  expect_length(svg_files, 1L)
  expect_match(out, "\\.svg")
})

test_that("writer = 'vegalite_png' produces PNG output", {
  skip_if_not_installed("V8")
  skip_if_not_installed("rsvg")
  skip_on_cran()
  skip_if_not_installed("withr")

  fig_dir <- withr::local_tempdir()

  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(
    query,
    writer = "vegalite_png",
    fig.path = paste0(fig_dir, "/fig-"),
    label = "test-png"
  )

  png_files <- list.files(fig_dir, pattern = "\\.png$")
  expect_length(png_files, 1L)
  expect_match(out, "\\.png")
})

test_that("writer option is ignored for plain SQL", {
  query <- "SELECT 1 AS x, 2 AS y"
  out <- run_query(query, writer = "vegalite_png")
  # Should produce a table, not an image
  expect_snapshot(cat(out))
})

test_that("invalid writer option produces error", {
  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )
  out <- run_query(query, writer = "ggplot2")
  expect_match(out, "Unsupported writer")
})

test_that("we can knit a mixed-chunk document", {
  skip_if_not_installed("withr")

  # Create a temporary working directory that will be deleted after this test
  dir <- withr::local_tempdir()
  withr::local_dir(dir)

  # We're copying the test file to working directory so side-effects,
  # like creating new figure folders, are contained
  basename <- "test_chunks.qmd"
  doc <- system.file(basename, package = "ggsql")
  in_file <- file.path(dir, basename)
  file.copy(doc, in_file)

  out_file <- file.path(dir, "test_chunks.md")

  out <- knitr::knit(input = in_file, output = out_file, quiet = TRUE)
  expect_equal(out_file, out)
  expect_true(file.exists(out))

  # Check that visualization was rendered (contains vega-embed script)
  content <- readLines(out)
  expect_true(any(grepl("vega-embed", content)))
})
