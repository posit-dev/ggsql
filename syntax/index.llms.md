# Syntax

## Main clauses

ggsql augments the standard SQL syntax with a number of new clauses to describe a visualisation. Apart from `VISUALISE` needing to be the first, the order of these clauses is arbitrary, though grouping e.g. all `DRAW` clauses will lead to queries that are easier to reason about:

- [`VISUALISE`](../syntax/clause/visualise.llms.md) initiates the visualisation part of the query
- [`DRAW`](../syntax/clause/draw.llms.md) adds a new layer to the visualisation
  - [`PLACE`](../syntax/clause/place.llms.md) adds an annotation layer
- [`SCALE`](../syntax/clause/scale.llms.md) specify how an aesthetic should be scaled
- [`FACET`](../syntax/clause/facet.llms.md) describes how data should be split into small multiples
- [`PROJECT`](../syntax/clause/project.llms.md) is used for selecting the coordinate system to use
- [`LABEL`](../syntax/clause/label.llms.md) is used to manually add titles to the plot or the various axes and legends

## Layers

There are many different layers to choose from when visualising your data. Some are straightforward translations of your data into visual marks such as a point layer, while others perform more or less complicated calculations like e.g. the histogram layer. A layer is selected by providing the layer name after the `DRAW` clause

### Layer types

- [`point`](../syntax/layer/type/point.llms.md) is used to create a scatterplot layer.
- [`line`](../syntax/layer/type/line.llms.md) is used to produce lineplots with the data sorted along the x axis.
- [`path`](../syntax/layer/type/path.llms.md) is like `line` above but does not sort the data but plot it according to its own order.
- [`segment`](../syntax/layer/type/segment.llms.md) connects two points with a line segment.
- [`rule`](../syntax/layer/type/rule.llms.md) draws horizontal and vertical reference lines.
- [`area`](../syntax/layer/type/area.llms.md) is used to display series as an area chart.
- [`ribbon`](../syntax/layer/type/ribbon.llms.md) is used to display series extrema.
- [`polygon`](../syntax/layer/type/polygon.llms.md) is used to display arbitrary shapes as polygons.
- [`text`](../syntax/layer/type/text.llms.md) is used to render datapoints as text.
- [`bar`](../syntax/layer/type/bar.llms.md) creates a bar chart, optionally calculating y from the number of records in each bar.
- [`density`](../syntax/layer/type/density.llms.md) creates univariate kernel density estimates, showing the distribution of a variable.
- [`violin`](../syntax/layer/type/violin.llms.md) displays a rotated kernel density estimate.
- [`histogram`](../syntax/layer/type/histogram.llms.md) bins the data along the x axis and produces a bar for each bin showing the number of records in it.
- [`boxplot`](../syntax/layer/type/boxplot.llms.md) displays continuous variables as 5-number summaries.
- [`range`](../syntax/layer/type/range.llms.md) a line segment between two values along an axis, with optional hinges at the endpoints.
- [`smooth`](../syntax/layer/type/smooth.llms.md) a trendline that follows the data shape.

### Position adjustments

- [`stack`](../syntax/layer/position/stack.llms.md) places objects with a shared baseline on top of each other.
- [`dodge`](../syntax/layer/position/dodge.llms.md) places objects that share the same discrete position side by side
- [`jitter`](../syntax/layer/position/jitter.llms.md) adds a small random offset to objects sharing the same discrete position
- [`identity`](../syntax/layer/position/identity.llms.md) does nothing, i.e. turns off position adjustment

## Scales

A scale is responsible for translating a data value to an aesthetic literal, e.g. a specific color for the fill aesthetic, or a radius in points for the size aesthetic. A scale is a combination of a specific aesthetic and a scale type

### Aesthetics

- [Position](../syntax/scale/aesthetic/0_position.llms.md) aesthetics are those aesthetics related to the spatial location of the data in the coordinate system.
- [Color](../syntax/scale/aesthetic/1_color.llms.md) aesthetics are related to the color of fill and stroke
- [`opacity`](../syntax/scale/aesthetic/2_opacity.llms.md) is the aesthetic that determines the opacity of the color
- [`linetype`](../syntax/scale/aesthetic/linetype.llms.md) governs the stroke pattern of strokes
- [`linewidth`](../syntax/scale/aesthetic/linewidth.llms.md) determines the width of strokes
- [`shape`](../syntax/scale/aesthetic/shape.llms.md) determines the shape of points
- [`size`](../syntax/scale/aesthetic/size.llms.md) governs the radius of points
- [Faceting](../syntax/scale/aesthetic/Z_faceting.llms.md) aesthetics are used to determine which facet panel the data belongs to

### Scale types

- [`continuous`](../syntax/scale/type/continuous.llms.md) scales translates a continuous input to a continuous output
- [`discrete`](../syntax/scale/type/discrete.llms.md) scales translates discrete input to a discrete output
- [`binned`](../syntax/scale/type/binned.llms.md) scales translate continuous input to an ordered discrete output by binning the data
- [`ordinal`](../syntax/scale/type/ordinal.llms.md) scales translate discrete input to an ordered discrete output by enforcing an ordering to the input
- [`identity`](../syntax/scale/type/identity.llms.md) scales passes the data through unchanged

## Coordinate systems

The coordinate system defines how the abstract position aesthetics are projected onto the screen or paper where the final plot appears. As such, it has great influence over the final look of the plot.

- [`cartesian`](../syntax/coord/cartesian.llms.md) is the classic coordinate system consisting of two perpendicular axes, one being horizontal and one being vertical
- [`polar`](../syntax/coord/polar.llms.md) interprets the primary position as the angular location relative to the center and the secondary position as the distance (radius) from the center, and this creates a circular coordinate system
