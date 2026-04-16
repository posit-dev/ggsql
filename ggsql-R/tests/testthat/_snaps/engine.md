# engine can handle a query without visualisation statement

    Code
      cat(out)
    Output
      SELECT mpg, disp FROM 'mtcars.csv'
      |  mpg|  disp|
      |----:|-----:|
      | 21.0| 160.0|
      | 21.0| 160.0|
      | 22.8| 108.0|
      | 21.4| 258.0|
      | 18.7| 360.0|
      | 18.1| 225.0|
      | 14.3| 360.0|
      | 24.4| 146.7|
      | 22.8| 140.8|
      | 19.2| 167.6|
      | 17.8| 167.6|
      | 16.4| 275.8|
      | 17.3| 275.8|
      | 15.2| 275.8|
      | 10.4| 472.0|
      | 10.4| 460.0|
      | 14.7| 440.0|
      | 32.4|  78.7|
      | 30.4|  75.7|
      | 33.9|  71.1|
      | 21.5| 120.1|
      | 15.5| 318.0|
      | 15.2| 304.0|
      | 13.3| 350.0|
      | 19.2| 400.0|
      | 27.3|  79.0|
      | 26.0| 120.3|
      | 30.4|  95.1|
      | 15.8| 351.0|
      | 19.7| 145.0|
      | 15.0| 301.0|
      | 21.4| 121.0|

# engine does not return a table when merely creating data

    Code
      cat(out)
    Output
      COPY (
            SELECT * FROM (VALUES
                (5.2, 18.5),
                (8.7, 22.3)
            ) AS t(x, y)
          ) TO 'data.csv' (HEADER, DELIMITER ',')

# r: prefix resolves R objects

    Code
      cat(out)
    Output
      SELECT mpg, disp FROM r:test_df
      |  mpg| disp|
      |----:|----:|
      | 21.0|  160|
      | 21.0|  160|
      | 22.8|  108|
      | 21.4|  258|
      | 18.7|  360|

# multiple r: refs in one query work

    Code
      cat(out)
    Output
      SELECT a.id, a.x, b.y FROM r:df_a a JOIN r:df_b b ON a.id = b.id
      | id|  x|  y|
      |--:|--:|--:|
      |  1| 10| 20|
      |  2| 11| 21|
      |  3| 12| 22|

# connection option creates a DuckDB reader

    Code
      cat(out)
    Output
      SELECT 1 AS x, 2 AS y
      |  x|  y|
      |--:|--:|
      |  1|  2|

# writer option is ignored for plain SQL

    Code
      cat(out)
    Output
      SELECT 1 AS x, 2 AS y
      |  x|  y|
      |--:|--:|
      |  1|  2|

