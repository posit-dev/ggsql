# Point

> Layers are declared with the [`DRAW` clause](../../../syntax/clause/draw.llms.md). Read the documentation for this clause for a thorough description of how to use it.

The point layer is used to create scatterplots. The scatterplot is most useful for displaying the relationship between two continuous variables. A bubblechart is a scatterplot with a third variable mapped to the size of points.

## Aesthetics

The following aesthetics are recognised by the point layer.

### Required

- Primary axis (e.g. `x`): Position along the primary axis.
- Secondary axis (e.g. `y`): Position along the secondary axis.

### Optional

- `size`: The size of each point
- `colour`: The default colour of each point
- `stroke`: The colour of the stroke around each point (if any). Overrides `colour`
- `fill`: The fill colour of each point (if any). Overrides `colour`
- `opacity`: The opacity of the point
- `shape`: The shape used to draw the point

## Settings

- `position`: Position adjustment. One of `'identity'` (default), `'stack'`, `'dodge'`, or `'jitter'`

## Data transformation

The point layer does not transform its data but passes it through unchanged

## Orientation

The point layer has no orientation. The axes are treated symmetrically.

## Examples

Create a classic scatterplot

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point
  MAPPING bill_len AS x, bill_dep AS y, species AS fill
```

Map to size to create a bubble chart

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point
  MAPPING bill_len AS x, bill_dep AS y, body_mass AS size
```

Use filter to only plot a subset of the data

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point
  MAPPING bill_len AS x, bill_dep AS y, species AS fill
  FILTER sex = 'female'
```

When points are plotted on a discrete scale you will likely see a lot of overplotting. Use jitter position to introduce a bit of random offset to counter that.

``` ggsql
VISUALISE species AS x, sex AS y, island AS fill FROM ggsql:penguins
DRAW point
  SETTING position => 'jitter', distribution => 'normal'
```

Use density distribution for a violin-like jitter effect, where jitter width scales with local data density.

``` ggsql
VISUALISE species AS x, bill_dep AS y FROM ggsql:penguins
DRAW point 
  SETTING position => 'jitter', distribution => 'density'
```
