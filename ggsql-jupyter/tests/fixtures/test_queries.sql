-- Test Queries for ggsql-jupyter Integration Tests

-- Simple SELECT query
SELECT 1 as x, 2 as y, 'test' as label;

-- Query with aggregation
SELECT
  COUNT(*) as count,
  SUM(value) as total,
  AVG(value) as average
FROM (VALUES (1), (2), (3), (4), (5)) AS t(value);

-- Simple visualization
SELECT 1 as x, 2 as y
VISUALISE AS PLOT
DRAW point USING x = x, y = y;

-- Line chart with multiple points
SELECT n as x, n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE AS PLOT
DRAW line USING x = x, y = y
LABEL title = 'Quadratic Function', x = 'X', y = 'Y';

-- Multi-layer visualization
SELECT n as x, n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE AS PLOT
DRAW line USING x = x, y = y
DRAW point USING x = x, y = y
LABEL title = 'Line with Points';

-- Date-based visualization
SELECT
  DATE '2024-01-01' + INTERVAL (n) DAY as date,
  n * 10 as value,
  CASE WHEN n % 2 = 0 THEN 'Even' ELSE 'Odd' END as category
FROM generate_series(0, 30) as t(n)
VISUALISE AS PLOT
DRAW line USING x = date, y = value, color = category
SCALE x USING type = 'date'
LABEL title = 'Time Series', x = 'Date', y = 'Value';

-- Bar chart
SELECT
  chr(65 + n) as category,
  (n + 1) * 10 as value
FROM generate_series(0, 4) as t(n)
VISUALISE AS PLOT
DRAW bar USING x = category, y = value
LABEL title = 'Bar Chart', x = 'Category', y = 'Value';

-- Faceted visualization
SELECT
  n as x,
  n * n as y,
  CASE WHEN n <= 5 THEN 'Group A' ELSE 'Group B' END as group
FROM generate_series(1, 10) as t(n)
VISUALISE AS PLOT
DRAW point USING x = x, y = y
FACET WRAP group
LABEL title = 'Faceted Plot';

-- Error case: invalid table
SELECT * FROM nonexistent_table;

-- Error case: invalid column
SELECT invalid_column FROM (SELECT 1 as x);

-- Error case: invalid VIZ syntax
SELECT 1 as x
VISUALISE AS PLOT
DRAW invalid_geom USING x = x;
