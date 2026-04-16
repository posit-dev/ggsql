test_that("duckdb_reader creates a reader", {
  reader <- duckdb_reader()
  expect_s3_class(reader, "Reader")
})

test_that("ggsql_register registers a data frame", {
  reader <- duckdb_reader()
  result <- ggsql_register(reader, mtcars, "cars")
  expect_invisible(ggsql_register(reader, iris, "iris"))
  # Returns reader for piping

  expect_s3_class(result, "Reader")
})

test_that("ggsql_register with replace works", {
  reader <- duckdb_reader()
  ggsql_register(reader, mtcars, "data")
  ggsql_register(reader, iris, "data", replace = TRUE)
  df <- ggsql_execute_sql(reader, "SELECT * FROM data LIMIT 1")
  expect_true("Sepal.Length" %in% names(df))
})

test_that("ggsql_unregister removes a table", {
  reader <- duckdb_reader()
  ggsql_register(reader, mtcars, "cars")
  ggsql_unregister(reader, "cars")
  expect_error(ggsql_execute_sql(reader, "SELECT * FROM cars"))
})

test_that("ggsql_execute_sql returns a data frame", {
  reader <- duckdb_reader()
  ggsql_register(reader, mtcars, "cars")
  df <- ggsql_execute_sql(reader, "SELECT mpg, disp FROM cars LIMIT 5")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 5)
  expect_equal(names(df), c("mpg", "disp"))
})

test_that("ggsql_execute returns a spec", {
  reader <- duckdb_reader()
  ggsql_register(reader, mtcars, "cars")
  spec <- ggsql_execute(
    reader,
    "SELECT * FROM cars VISUALISE mpg AS x, disp AS y DRAW point"
  )
  expect_s3_class(spec, "Spec")
})

test_that("piping works", {
  reader <- duckdb_reader()
  result <- reader |>
    ggsql_register(mtcars, "cars") |>
    ggsql_execute_sql("SELECT mpg FROM cars LIMIT 3")
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3)
})
