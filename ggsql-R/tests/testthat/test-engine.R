test_that("engine can handle a query", {

  data_file <- tempfile(fileext = ".csv")
  data_file <- "mtcars.csv"
  on.exit(unlink(data_file))
  write.csv(mtcars, data_file)

  query <- c(
    paste0("SELECT mpg, disp FROM '", data_file, "'"),
    "VISUALISE AS PLOT",
    "DRAW point USING x = mpg, y = disp"
  )

  opts <- knitr::opts_current$get()
  opts$code <- query
  opts$dev <- "png"

  out <- ggsql_engine(opts)

  # We expect path to png file here, since output format for knitr is undetermined
  expect_vector(out, character(), size = 1)
})
