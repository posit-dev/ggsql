-- Test Queries for ggsql-jupyter Integration Tests

-- Simple SELECT query
SELECT 1 as x, 2 as y, 'test' as label;

-- Query with aggregation
SELECT
  COUNT(*) as count,
  SUM(value) as total,
  AVG(value) as average
FROM (VALUES (1), (2), (3), (4), (5)) AS t(value);

-- Simple visualization with global mapping
SELECT 1 as x, 2 as y
VISUALISE x, y
DRAW point;

-- Line chart with multiple points
SELECT n as x, n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE x, y
DRAW line
LABEL title AS 'Quadratic Function', x AS 'X', y AS 'Y';

-- Multi-layer visualization with global mapping
SELECT n as x, n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE x, y
DRAW line
DRAW point
LABEL title AS 'Line with Points';

-- Date-based visualization with global mapping
SELECT
  DATE '2024-01-01' + INTERVAL (n) DAY as date,
  n * 10 as value,
  CASE WHEN n % 2 = 0 THEN 'Even' ELSE 'Odd' END as category
FROM generate_series(0, 30) as t(n)
VISUALISE date AS x, value AS y, category AS color
DRAW line
SCALE x SETTING type => 'date'
LABEL title => 'Time Series', x => 'Date', y => 'Value';

-- Bar chart with global mapping
SELECT
  chr(65 + n) as category,
  (n + 1) * 10 as value
FROM generate_series(0, 4) as t(n)
VISUALISE category AS x, value AS y
DRAW bar
LABEL title AS 'Bar Chart', x AS 'Category', y AS 'Value';

-- Faceted visualization with global mapping
SELECT
  n as x,
  n * n as y,
  CASE WHEN n <= 5 THEN 'Group A' ELSE 'Group B' END as group
FROM generate_series(1, 10) as t(n)
VISUALISE x, y
DRAW point
FACET WRAP group
LABEL title AS 'Faceted Plot';

-- Visualization with FILTER clause - global mapping with layer filter
SELECT
  n as x,
  n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE x, y
DRAW line
DRAW point FILTER y > 25
LABEL title AS 'Filtered Points';

-- Visualization with SETTING parameters
SELECT
  n as x,
  n * n as y
FROM generate_series(1, 10) as t(n)
VISUALISE x, y
DRAW point SETTING size => 10, opacity => 0.5
LABEL title => 'Points with Parameters';

-- Error case: invalid table
SELECT * FROM nonexistent_table;

-- Error case: invalid column
SELECT invalid_column FROM (SELECT 1 as x);

-- Error case: invalid VIZ syntax
SELECT 1 as x
VISUALISE
DRAW invalid_geom
    MAPPING x AS x;
