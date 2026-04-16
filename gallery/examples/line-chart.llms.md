# Line chart

basic

line

time-series

Time series visualization with proper date scaling

Line charts are ideal for showing trends over time. The `SCALE x VIA date` clause ensures proper date formatting on the axis.

## Code

``` ggsql
SELECT Date, Temp FROM ggsql:airquality
VISUALISE Date AS x, Temp AS y
DRAW line
SCALE x VIA date
LABEL
  title => 'Daily Temperature',
  x => 'Date',
  y => 'Temperature (F)'
```

## Explanation

- `SELECT ... FROM ggsql:airquality` queries the built-in air quality dataset
- `VISUALISE Date AS x, Temp AS y` maps the date column to x and temperature to y
- `DRAW line` connects data points with lines
- `SCALE x VIA date` ensures the x-axis is formatted as dates with appropriate tick marks
- `LABEL` provides descriptive titles for the chart and axes

## Variations

### Multiple lines by category

``` ggsql
SELECT Date, Temp, Month FROM ggsql:airquality
VISUALISE Date AS x, Temp AS y, Month AS color
DRAW line
SCALE x VIA date
LABEL
  title => 'Daily Temperature by Month',
  x => 'Date',
  y => 'Temperature (F)'
```
