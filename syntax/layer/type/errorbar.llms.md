# Errorbar

> Layers are declared with the [`DRAW` clause](../../../syntax/clause/draw.llms.md). Read the documentation for this clause for a thorough description of how to use it.

Errorbars are used to display paired metrics, typically some interval, for a variable. It is displayed as a line between the two values, often with hinges at the ends.

## Aesthetics

The following aesthetics are recognised by the errorbar layer.

### Required

- Primary axis (e.g. `x`): Position along the primary axis
- Secondary axis minimum (e.g. `ymin`): Lower position along the secondary axis.
- Secondary axis maximum (e.g. `ymax`): Upper position along the secondary axis.

### Optional

- `stroke`/`colour`: The colour of the lines in the errorbar.
- `opacity`: The opacity of the colour.
- `linewidth`: The width of the lines in the errorbar.
- `linetype`: The dash pattern of the lines in the errorbar.

## Settings

- `width`: The width of the hinges in points (must be \>= 0). Defaults to 10. Can be set to `null` to not display hinges.

## Data transformation

The errorbar layer does not transform its data but passes it through unchanged.

## Orientation

The orientation of errorbar layers is deduced directly from the mapping, because the interval is mapped to the secondary axis. To create a horizontal errorbar layer, you map the independent variable to `y` instead of `x` and the interval to `xmin` and `xmax` (assuming a default Cartesian coordinate system).

## Examples

Create example data

``` ggsql
CREATE TABLE penguin_summary AS
WITH stats AS (
  SELECT species, AVG(bill_dep) AS mean
  FROM ggsql:penguins
  GROUP BY species
)
SELECT species, mean - 1.5 AS low, mean, mean + 1.5 AS high
FROM stats
```

Classic errorbar with point at centre.

``` ggsql
VISUALISE species AS x FROM penguin_summary
DRAW errorbar 
  MAPPING low AS ymax, high AS ymin
DRAW point 
  MAPPING mean AS y
```

Dynamite plot using bars instead of points, using extra wide hinges.

``` ggsql
VISUALISE species AS x FROM penguin_summary
DRAW errorbar 
  MAPPING low AS ymax, high AS ymin
  SETTING width => 40
DRAW bar 
  MAPPING mean AS y
```

The hinges can be omitted by setting `null` as width.

``` ggsql
VISUALISE species AS x FROM penguin_summary
DRAW errorbar 
  MAPPING low AS ymax, high AS ymin
  SETTING width => null
```

A horizontal errorbar can be rendered by swapping the `x` and `y` directions.

``` ggsql
VISUALISE species AS y FROM penguin_summary
DRAW errorbar 
  MAPPING low AS xmax, high AS xmin
DRAW point 
  MAPPING mean AS x
```
