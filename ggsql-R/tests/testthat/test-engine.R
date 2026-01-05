skip_if_not_installed("png")
skip_if_not_installed("rsvg")
skip_if_not_installed("V8")

test_that("engine can handle a query", {

  data_file <- tempfile(fileext = ".csv")
  data_file <- "mtcars.csv"
  on.exit(unlink(data_file))
  write.csv(mtcars, data_file)

  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE mpg AS x, disp AS y",
    "DRAW point"
  )

  opts <- knitr::opts_current$get()
  opts$code <- query
  opts$dev <- "png"

  out <- ggsql_engine(opts)

  # We expect path to png file here, since output format for knitr is undetermined
  expect_type(out, "character")
  expect_length(out, 1L)
})

test_that("engine can handle a query without visualisation statement", {
  data_file <- tempfile(fileext = ".csv")
  data_file <- "mtcars.csv"
  on.exit(unlink(data_file))
  write.csv(mtcars, data_file)

  query <- paste0("SELECT mpg, disp FROM '", data_file, "'")

  opts <- knitr::opts_current$get()
  opts$code <- query

  out <- ggsql_engine(opts)
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
  opts <- knitr::opts_current$get()
  opts$code <- query

  out <- ggsql_engine(opts)
  expect_snapshot(cat(out))
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

  expect_snapshot_file(out)
})
