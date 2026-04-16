# Faceted plot

faceted

advanced

Small multiples showing data split by category

Faceted plots (small multiples) split data into separate panels by one or more categorical variables. This makes it easy to compare patterns across groups.

## Code

``` ggsql
SELECT bill_len, bill_dep, species FROM ggsql:penguins
VISUALISE bill_len AS x, bill_dep AS y
DRAW point
FACET species
LABEL
  title => 'Bill Dimensions by Species',
  x => 'Bill Length (mm)',
  y => 'Bill Depth (mm)'
```

## Explanation

- `FACET species` creates a separate panel for each penguin species
- Each panel shows the same scatter plot, filtered to that species
- This reveals species-specific patterns that might be hidden in a combined view

## Variations

### Grid layout with two variables

Use `FACET rows BY cols` to create a grid layout:

``` ggsql
SELECT bill_len, bill_dep, species, island FROM ggsql:penguins
VISUALISE bill_len AS x, bill_dep AS y
DRAW point
FACET species BY island
LABEL
  title => 'Bill Dimensions by Species and Island',
  x => 'Bill Length (mm)',
  y => 'Bill Depth (mm)'
```

### Free scales

Allow each facet to have independent axis scales with `SETTING free`:

``` ggsql
SELECT bill_len, bill_dep, species FROM ggsql:penguins
VISUALISE bill_len AS x, bill_dep AS y
DRAW point
FACET species SETTING free => 'y'
LABEL
  title => 'Bill Dimensions (Free Y Scale)',
  x => 'Bill Length (mm)',
  y => 'Bill Depth (mm)'
```
