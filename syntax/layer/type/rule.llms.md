# Rule

> Layers are declared with the [`DRAW` clause](../../../syntax/clause/draw.llms.md). Read the documentation for this clause for a thorough description of how to use it.

The rule layer is used to draw reference lines perpendicular to an axis at specified values. This is useful for adding thresholds, means, medians, event markers, cutoff dates or other guides to the plot. The lines span the full width or height of the panels.

Additionally, the rule layer can draw diagonal reference lines by specifying a `slope` along with a position (`x` or `y`). The position acts as the intercept term in the linear equation. This is useful for adding regression lines, diagonal guides, or mathematical functions to plots.

> The rule layer doesn’t have a natural extent along the secondary axis. The secondary scale range can thus not be deduced from rule layers. If the rule layer is the only layer in the visualisation, you must specify the position scale range manually.

## Aesthetics

The following aesthetics are recognised by the rule layer.

### Required

- Primary axis (e.g. `x` or `y`): Position along the primary axis.

### Optional

- `slope` The \\\beta\\ in the equations described below. Defaults to 0.
- `colour`/`stroke`: The colour of the line
- `opacity`: The opacity of the line
- `linewidth`: The width of the line
- `linetype`: The type of the line, i.e. the dashing pattern

## Settings

- `position`: Position adjustment. One of `'identity'` (default), `'stack'`, `'dodge'`, or `'jitter'`

## Data transformation

For diagonal lines, the position aesthetic determines the intercept:

- `y = a, slope = β` creates the line: \\y = a + \beta x\\ (line passes through \\(0, a)\\)
- `x = a, slope = β` creates the line: \\x = a + \beta y\\ (line passes through \\(a, 0)\\)

Note that `slope` represents \\\Delta x/\Delta y\\ (change in x per unit change in y) when using `x` mapping, not the traditional \\\Delta y/\Delta x\\. The default slope is 0, creating horizontal lines when `y` is given and vertical lines when `x` is given.

## Orientation

The orientation is deduced directly from the mapping. See the [‘Data transformation’](#data-transformation) section for details.

## Examples

Add a horizontal threshold line to a time series plot:

``` ggsql
SELECT Date AS date, temp AS temperature
FROM ggsql:airquality
WHERE Month = 5

VISUALISE 
DRAW line 
  MAPPING date AS x, temperature AS y
PLACE rule 
  SETTING y => 70
```

Add a vertical line to mark a specific value:

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point 
  MAPPING bill_len AS x, bill_dep AS y
PLACE rule 
  SETTING x => 45
```

Add multiple threshold lines with different colors. Note that because we’re mapping data from data, we use the `DRAW` clause instead of the `PLACE` clause.

``` ggsql
WITH thresholds(value, label) AS (VALUES
  (70, 'Target'),
  (80, 'Warning'),
  (90, 'Critical')
)
SELECT Date AS date, temp AS temperature
FROM ggsql:airquality
WHERE Month = 5

VISUALISE 
DRAW line 
  MAPPING date AS x, temperature AS y
DRAW rule 
  MAPPING value AS y, label AS colour FROM thresholds
```

Add a diagnoal reference line to a scatterplot by using `slope`

``` ggsql
VISUALISE FROM ggsql:penguins
  DRAW point MAPPING bill_len AS x, bill_dep AS y
  PLACE rule SETTING slope => 0.4, y => -1
```

Add multiple reference lines with different colors from a separate dataset. Note we’re mapping from data here, so we use `DRAW` instead of `PLACE`.

``` ggsql
WITH lines AS (
    SELECT * FROM (VALUES
        (0.4, -1, 'Line A'),
        (0.2, 8, 'Line B'),
        (0.8, -19, 'Line C')
    ) AS t(slope, intercept, label)
)
VISUALISE FROM ggsql:penguins
  DRAW point MAPPING bill_len AS x, bill_dep AS y
  DRAW rule
    MAPPING 
      slope AS slope, 
      intercept AS y, 
      label AS colour
    FROM lines
```
