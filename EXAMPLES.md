# ggSQL Examples

This document provides a collection of basic examples demonstrating how to use ggSQL for data visualization.

## Table of Contents

- [Basic Visualizations](#basic-visualizations)
- [Multiple Layers](#multiple-layers)
- [Scales and Transformations](#scales-and-transformations)
- [Coordinate Systems](#coordinate-systems)
- [Labels and Themes](#labels-and-themes)
- [Faceting](#faceting)
- [Common Table Expressions (CTEs)](#common-table-expressions-ctes)
- [Advanced Examples](#advanced-examples)

---

## Basic Visualizations

### Simple Scatter Plot

```sql
SELECT x, y FROM data
VISUALISE x, y
DRAW point
```

### Line Chart

```sql
SELECT date, revenue FROM sales
WHERE year = 2024
VISUALISE date AS x, revenue AS y
DRAW line
```

### Bar Chart

```sql
SELECT category, total FROM sales
GROUP BY category
VISUALISE category AS x, total AS y
DRAW bar
```

### Area Chart

```sql
SELECT date, cumulative FROM metrics
VISUALISE date AS x, cumulative AS y
DRAW area
```

---

## Multiple Layers

### Line with Points

Using global mapping to share aesthetics across layers:

```sql
SELECT date, value FROM timeseries
VISUALISE date AS x, value AS y
DRAW line
DRAW point
```

### Bar Chart with Colored Regions

```sql
SELECT category, revenue, region FROM sales
GROUP BY category, region
VISUALISE category AS x, revenue AS y, region AS fill
DRAW bar
```

### Multiple Lines by Group

```sql
SELECT date, value, category FROM metrics
VISUALISE date AS x, value AS y, category AS color
DRAW line
```

---

## Scales and Transformations

### Date Scale

```sql
SELECT sale_date, revenue FROM sales
VISUALISE sale_date AS x, revenue AS y
DRAW line
SCALE x SETTING type TO 'date'
```

### Logarithmic Scale

```sql
SELECT x, y FROM exponential_data
VISUALISE x, y
DRAW point
SCALE y SETTING type TO 'log10'
```

### Color Palette

```sql
SELECT date, temperature, station FROM weather
VISUALISE date AS x, temperature AS y, station AS color
DRAW line
SCALE color SETTING palette TO 'viridis'
```

### Custom Domain

```sql
SELECT category, value FROM data
VISUALISE category AS x, value AS y, category AS fill
DRAW bar
SCALE fill SETTING domain TO ['A', 'B', 'C', 'D']
```

---

## Coordinate Systems

### Cartesian with Limits

```sql
SELECT x, y FROM data
VISUALISE x, y
DRAW point
COORD cartesian SETTING xlim TO [0, 100], ylim TO [0, 50]
```

### Flipped Coordinates (Horizontal Bar Chart)

```sql
SELECT category, value FROM data
ORDER BY value DESC
VISUALISE category AS x, value AS y
DRAW bar
COORD flip
```

### Polar Coordinates (Pie Chart)

```sql
SELECT category, SUM(value) as total FROM data
GROUP BY category
VISUALISE category AS x, total AS y
DRAW bar
COORD polar
```

### Polar with Theta Specification

```sql
SELECT category, value FROM data
VISUALISE category AS x, value AS y
DRAW bar
COORD polar SETTING theta TO y
```

---

## Labels and Themes

### Chart with Title and Axis Labels

```sql
SELECT date, revenue FROM sales
VISUALISE date AS x, revenue AS y
DRAW line
LABEL title => 'Monthly Revenue Trends',
      x => 'Date',
      y => 'Revenue ($)'
```

### Multiple Labels

```sql
SELECT date, value FROM metrics
VISUALISE date AS x, value AS y
DRAW area
LABEL title => 'Performance Metrics',
      subtitle => 'Q4 2024',
      x => 'Date',
      y => 'Metric Value',
      caption => 'Data source: Analytics DB'
```

### Themed Visualization

```sql
SELECT category, value FROM data
VISUALISE category AS x, value AS y
DRAW bar
THEME minimal
```

### Theme with Custom Properties

```sql
SELECT x, y FROM data
VISUALISE x, y
DRAW point
THEME dark SETTING background TO '#1a1a1a'
```

---

## Faceting

### Facet Wrap

```sql
SELECT date, value, region FROM sales
VISUALISE date AS x, value AS y
DRAW line
FACET WRAP region
```

### Facet Grid

```sql
SELECT date, value, region, product FROM sales
VISUALISE date AS x, value AS y
DRAW line
FACET region BY product
```

### Facet with Free Scales

```sql
SELECT date, value, category FROM metrics
VISUALISE date AS x, value AS y
DRAW line
FACET WRAP category SETTING scales TO 'free_y'
```

---

## Common Table Expressions (CTEs)

ggSQL supports two patterns for working with CTEs:

1. **Traditional Pattern**: `WITH ... SELECT ... VISUALISE`
2. **Shorthand Pattern**: `WITH ... VISUALISE FROM <cte>` (injected SELECT)

### Simple CTE with VISUALISE FROM

The shorthand syntax automatically injects `SELECT * FROM <cte>`:

```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', sale_date) as month,
        SUM(revenue) as total_revenue
    FROM sales
    GROUP BY DATE_TRUNC('month', sale_date)
)
VISUALISE month AS x, total_revenue AS y FROM monthly_sales
DRAW line
SCALE x SETTING type TO 'date'
LABEL title => 'Monthly Revenue Trends',
      x => 'Month',
      y => 'Revenue ($)'
```

### Traditional CTE with SELECT

Use the traditional pattern when you need to filter or transform CTE results:

```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', sale_date) as month,
        region,
        SUM(revenue) as total_revenue
    FROM sales
    GROUP BY DATE_TRUNC('month', sale_date), region
)
SELECT * FROM monthly_sales WHERE region = 'North'
VISUALISE month AS x, total_revenue AS y
DRAW line
SCALE x SETTING type TO 'date'
```

### Multiple CTEs

Chain multiple CTEs and visualize from any of them:

```sql
WITH daily_sales AS (
    SELECT sale_date, region, SUM(revenue) as revenue
    FROM sales
    GROUP BY sale_date, region
),
regional_totals AS (
    SELECT region, SUM(revenue) as total
    FROM daily_sales
    GROUP BY region
)
VISUALISE region AS x, total AS y, region AS fill FROM regional_totals
DRAW bar
COORD flip
LABEL title => 'Total Revenue by Region',
      x => 'Region',
      y => 'Total Revenue ($)'
```

### CTE with JOIN

CTEs can contain complex queries including JOINs:

```sql
WITH product_metrics AS (
    SELECT
        p.product_name,
        p.category,
        SUM(s.quantity) as total_sold,
        SUM(s.revenue) as total_revenue
    FROM products p
    JOIN sales s ON p.product_id = s.product_id
    WHERE s.sale_date >= '2024-01-01'
    GROUP BY p.product_name, p.category
)
VISUALISE total_sold AS x, total_revenue AS y, category AS color FROM product_metrics
DRAW point
LABEL title => 'Product Performance',
      x => 'Units Sold',
      y => 'Revenue ($)'
```

### Recursive CTE

Recursive CTEs work with VISUALISE FROM:

```sql
WITH RECURSIVE series AS (
    SELECT 1 as n, 1 as value
    UNION ALL
    SELECT n + 1, value * 2
    FROM series
    WHERE n < 10
)
VISUALISE n AS x, value AS y FROM series
DRAW line
DRAW point
SCALE y SETTING type TO 'log10'
LABEL title => 'Exponential Growth',
      x => 'Step',
      y => 'Value (log scale)'
```

### CTE with Window Functions

Calculate running totals or rankings in CTEs:

```sql
WITH ranked_products AS (
    SELECT
        product_name,
        category,
        revenue,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) as rank
    FROM product_sales
)
SELECT * FROM ranked_products WHERE rank <= 5
VISUALISE product_name AS x, revenue AS y, category AS color
DRAW bar
FACET WRAP category SETTING scales TO 'free_x'
COORD flip
LABEL title => 'Top 5 Products per Category',
      x => 'Product',
      y => 'Revenue ($)'
```

### Temporal Analysis with CTEs

Create time-based aggregations:

```sql
WITH daily_metrics AS (
    SELECT
        DATE_TRUNC('day', timestamp) as day,
        AVG(temperature) as avg_temp,
        MAX(temperature) as max_temp,
        MIN(temperature) as min_temp
    FROM sensor_data
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', timestamp)
)
VISUALISE day AS x FROM daily_metrics
DRAW ribbon
    MAPPING min_temp AS ymin, max_temp AS ymax, 'lightblue' AS fill
    SETTING alpha TO 0.3
DRAW line
    MAPPING avg_temp AS y, 'blue' AS color
    SETTING size TO 2
SCALE x SETTING type TO 'date'
LABEL title => 'Temperature Range (Last 30 Days)',
      x => 'Date',
      y => 'Temperature (°C)'
```

### Important Rules

**Valid Combinations:**
- `WITH cte AS (...) VISUALISE FROM cte` - Shorthand (SELECT injected)
- `WITH cte AS (...) SELECT * FROM cte VISUALISE` - Traditional
- `CREATE TABLE x ...; WITH cte AS (...) VISUALISE FROM cte` - Multiple statements

**Invalid Combinations:**
- `WITH cte AS (...) SELECT * FROM cte VISUALISE FROM cte` - Cannot mix SELECT with VISUALISE FROM
- `SELECT * FROM x VISUALISE FROM y` - Cannot use VISUALISE FROM after SELECT

**Why VISUALISE FROM?**

The shorthand syntax (`VISUALISE FROM`) is useful when:
1. You want to visualize an entire CTE without filtering
2. You're working with tables or views directly
3. You want concise syntax for simple visualizations

Use the traditional pattern (`SELECT ... VISUALISE`) when:
1. You need to filter or transform CTE results
2. You're using set operations (UNION, INTERSECT, EXCEPT)
3. You need complex SELECT logic before visualization

---

## Advanced Examples

### Complete Regional Sales Analysis

```sql
SELECT
    sale_date,
    region,
    SUM(quantity) as total_quantity
FROM sales
WHERE sale_date >= '2024-01-01'
GROUP BY sale_date, region
ORDER BY sale_date
VISUALISE sale_date AS x, total_quantity AS y, region AS color
DRAW line
DRAW point
SCALE x SETTING type TO 'date'
FACET WRAP region
LABEL title => 'Sales Trends by Region',
      x => 'Date',
      y => 'Total Quantity'
THEME minimal
```

### Time Series with Multiple Aesthetics

```sql
SELECT
    timestamp,
    temperature,
    humidity,
    station
FROM weather_data
WHERE timestamp >= NOW() - INTERVAL '7 days'
VISUALISE timestamp AS x, temperature AS y, station AS color, station AS linetype
DRAW line
SCALE x SETTING type TO 'datetime'
SCALE color SETTING palette TO 'viridis'
LABEL title => 'Temperature Trends',
      x => 'Time',
      y => 'Temperature (°C)'
```

### Categorical Analysis with Flipped Coordinates

```sql
SELECT
    product_name,
    SUM(revenue) as total_revenue
FROM sales
GROUP BY product_name
ORDER BY total_revenue DESC
LIMIT 10
VISUALISE product_name AS x, total_revenue AS y, product_name AS fill
DRAW bar
COORD flip SETTING color TO ['red', 'orange', 'yellow', 'green', 'blue',
                          'indigo', 'violet', 'pink', 'brown', 'gray']
LABEL title => 'Top 10 Products by Revenue',
      x => 'Product',
      y => 'Revenue ($)'
THEME classic
```

### Distribution with Custom Domain

```sql
SELECT
    date,
    value,
    category
FROM measurements
WHERE category IN ('A', 'B', 'C')
VISUALISE date AS x, value AS y, category AS color, value AS size
DRAW point
SCALE x SETTING type TO 'date'
SCALE color SETTING domain TO ['A', 'B', 'C']
SCALE size SETTING limits TO [0, 100]
COORD cartesian SETTING ylim TO [0, 150]
LABEL title => 'Measurement Distribution',
      x => 'Date',
      y => 'Value'
```

### Multi-Layer Visualization with Annotations

```sql
SELECT
    x,
    y,
    category,
    label
FROM data_points
VISUALISE x, y, category AS color
DRAW point SETTING size TO 5
DRAW text MAPPING label AS label
SCALE color SETTING palette TO 'viridis'
COORD cartesian SETTING xlim TO [0, 100], ylim TO [0, 100]
LABEL title => 'Annotated Scatter Plot',
      x => 'X Axis',
      y => 'Y Axis'
```

```sql
SELECT
    cyl,
    COUNT(*) as vehicle_count
FROM data_points
WHERE cyl IN (4, 6, 8)
GROUP BY cyl
ORDER BY cyl
VISUALISE cyl AS x, vehicle_count AS y
DRAW bar
SCALE x SETTING domain TO [4, 6, 8]
LABEL title => 'Distribution of Vehicles by Number of Cylinders',
      x => 'Number of Cylinders',
      y => 'Number of Vehicles'
```

---

## Global Mapping

ggSQL supports global mappings at the VISUALISE level, which apply to all layers. This is the preferred style when multiple layers share the same aesthetic mappings.

### Implicit Global Mapping

When column names match aesthetic names, use implicit mapping:

```sql
-- Columns named x, y map to x, y aesthetics automatically
SELECT x, y FROM data
VISUALISE x, y
DRAW point
```

### Explicit Global Mapping

Map columns to aesthetics with different names:

```sql
SELECT date, revenue, region FROM sales
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
DRAW point
```

### Wildcard Global Mapping

Map all columns automatically (useful for exploration):

```sql
SELECT * FROM data
VISUALISE *
DRAW point
```

### Mixed Global and Layer Mappings

Use global mapping for shared aesthetics, layer mapping for layer-specific ones:

```sql
SELECT date, avg_temp, min_temp, max_temp FROM weather
VISUALISE date AS x
DRAW ribbon MAPPING min_temp AS ymin, max_temp AS ymax
DRAW line MAPPING avg_temp AS y
```

---

## Case Insensitivity

ggSQL keywords are case-insensitive. All of the following are valid:

```sql
-- Uppercase (traditional)
VISUALISE date AS x, revenue AS y
DRAW line

-- Lowercase
visualise date as x, revenue as y
draw line

-- Mixed case
Visualise date AS x, revenue AS y
Draw Line
```

---

## Tips and Best Practices

1. **Use Global Mappings**: Prefer global mappings (`VISUALISE date AS x, revenue AS y`) over per-layer mappings when aesthetics are shared across layers.

2. **Date Handling**: Always use `SCALE x SETTING type TO 'date'` for date columns to ensure proper axis formatting.

3. **Color Mappings**: Use `color` for continuous data and `fill` for categorical data in bars/areas.

4. **Coordinate Limits**: Set explicit limits with `COORD cartesian SETTING xlim TO [min, max]` to control axis ranges.

5. **Faceting**: Use faceting to create small multiples when comparing across categories.

6. **Multiple Layers**: Combine layers (e.g., line + point) for richer visualizations.

7. **Themes**: Apply themes last in your specification for consistent styling.

8. **Labels**: Always provide meaningful titles and axis labels for clarity.

9. **Domain Specification**: Use either SCALE or COORD for domain/limit specification, but not both for the same aesthetic.

---

## Running Examples

### Using the CLI

```bash
# Parse and validate
ggsql parse query.sql

# Execute and generate Vega-Lite JSON
ggsql exec query.sql --writer vegalite --output chart.vl.json

# Execute from file
ggsql run query.sql
```

### Using the REST API

```bash
# Start the server with sample data
ggsql-rest --load-sample-data --port 3334

# Execute a query via HTTP
curl -X POST http://localhost:3334/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM products VISUALISE name AS x, price AS y DRAW bar"}'
```

### Using the Test Application

```bash
cd test-app
npm install
npm run dev
# Open http://localhost:5173
```

---

## Further Reading

- See [CLAUDE.md](CLAUDE.md) for full system architecture and implementation details
- See [README.md](README.md) for installation and setup instructions
- See the `tree-sitter-ggsql/test/corpus/` directory for more grammar examples
