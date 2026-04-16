test_that("vegalite_writer creates a writer", {
  writer <- vegalite_writer()
  expect_s3_class(writer, "Writer")
})

test_that("ggsql_render returns Vega-Lite JSON", {
  reader <- duckdb_reader()
  ggsql_register(reader, mtcars, "cars")
  spec <- ggsql_execute(
    reader,
    "SELECT * FROM cars VISUALISE mpg AS x, disp AS y DRAW point"
  )
  writer <- vegalite_writer()
  json <- ggsql_render(writer, spec)
  expect_type(json, "character")
  expect_match(json, "vega-lite")
  # Should be valid JSON
  parsed <- jsonlite::fromJSON(json)
  expect_true("$schema" %in% names(parsed))
})
