# Bar

> Layers are declared with the [`DRAW` clause](../../../syntax/clause/draw.llms.md). Read the documentation for this clause for a thorough description of how to use it.

The bar layer is used to create bar plots. You can either specify the height of the bars directly or let the layer calculate it either as the count of records within the same group or as a weighted sum of the records.

## Aesthetics

The following aesthetics are recognised by the bar layer.

### Required

The bar layer has no required aesthetics

### Optional

- Primary axis (e.g. `x`): The categories to create bars for. If missing all records will be shown in the same bar
- Secondary axis (e.g. `y`): The height of the bars. If missing, it will be calculated by the layer
- `colour`: The default colour of each bar
- `stroke`: The colour of the stroke around each bar. Overrides `colour`
- `fill`: The fill colour of each bar. Overrides `colour`
- `opacity`: The opacity of the bar fill
- `linewidth`: The width of the stroke
- `linetype`: The type of stroke, i.e. the dashing pattern

## Settings

- `position`: Position adjustment. One of `'identity'`, `'stack'` (default), `'dodge'`, or `'jitter'`
- `width`: The width of the bars as a proportion of the available width (0 to 1)

## Data transformation

If the secondary axis has not been mapped the layer will calculate counts for you and display these as the secondary axis.

### Properties

- `weight`: If mapped, the sum of the weights within each group is calculated instead of the count in each group

### Calculated statistics

- `count`: The count or, if `weight` have been mapped, sum of weights in each group.
- `proportion`: The groupwise proportion, i.e. the `count` divided by the sum of `count` within each group

### Default remappings

- `count AS <secondary axis>`: By default the barplot will show count as the height of the bars

## Orientation

Bar plots have categories along their primary axis. The orientation is deduced directly from the mapping. To create a horizontal bar plot you map the categories to `y` instead of `x` (assuming a default Cartesian coordinate system).

## Examples

Show the number of each species in the data

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS x
```

Use `weight` to instead show the collective mass of each species

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS x, body_mass AS weight
```

Map fill to a discrete value to create a stacked bar chart

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS x, island AS fill
```

Or change the position setting to e.g. get a dodged bar chart

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS x, sex AS fill
  SETTING position => 'dodge'
```

Map to y if the dataset already contains the value you want to show

``` ggsql
SELECT species, MAX(body_mass) AS max_mass FROM ggsql:penguins
GROUP BY species
VISUALISE
DRAW bar
  MAPPING species AS x, max_mass AS y
```

Use together with a binned scale as an alternative to the [histogram layer](../../../syntax/layer/type/histogram.llms.md)

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING body_mass AS x
SCALE BINNED x
  SETTING breaks => 10
```

Create a horizontal bar plot by changing the mapping

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS y
```

And use with a polar coordinate system to create a pie chart

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW bar
  MAPPING species AS fill
PROJECT TO polar
```
