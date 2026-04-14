test_that("ggsql_validate returns a ggsql_validated object", {
  v <- ggsql_validate(
    "SELECT 1 AS x, 2 AS y VISUALISE DRAW point MAPPING x AS x, y AS y"
  )
  expect_s3_class(v, "ggsql_validated")
  expect_true(v$has_visual)
  expect_true(v$valid)
})

test_that("ggsql_validate detects errors", {
  v <- ggsql_validate("SELECT 1 AS x VISUALISE x DRAW point")
  expect_false(v$valid)
  expect_true(NROW(v$errors) > 0)
})

test_that("ggsql_validate handles pure SQL", {
  v <- ggsql_validate("SELECT 1 AS x")
  expect_false(v$has_visual)
  expect_true(v$valid)
})

test_that("ggsql_has_visual works", {
  v <- ggsql_validate("SELECT 1 AS x VISUALISE x DRAW point")
  expect_true(ggsql_has_visual(v))

  v2 <- ggsql_validate("SELECT 1 AS x")
  expect_false(ggsql_has_visual(v2))
})

test_that("ggsql_is_valid works", {
  v <- ggsql_validate(
    "SELECT 1 AS x, 2 AS y VISUALISE DRAW point MAPPING x AS x, y AS y"
  )
  expect_true(ggsql_is_valid(v))
})

test_that("print.ggsql_validated works", {
  v <- ggsql_validate(
    "SELECT 1 AS x, 2 AS y VISUALISE DRAW point MAPPING x AS x, y AS y"
  )
  expect_invisible(print(v))
})
