# Frequently asked questions

## Getting started

> **NOTE:**
>
> ggsql is a SQL extension for declarative data visualization based on Grammar of Graphics principles. It allows you to combine SQL data queries with visualization specifications in a single, composable syntax.
>
> ``` ggsql
> SELECT date, revenue, region FROM sales
> VISUALISE date AS x, revenue AS y, region AS color
> DRAW line
> ```

> **NOTE:**
>
> See the installation instruction in the [Get started](get_started/installation.llms.md) tutorial.

> **NOTE:**
>
> ggsql is built in a modular way so we can gradually add new backends to it. Currently, ggsql works with DuckDB and SQLite, but we are planning on expanding that soon!

> **NOTE:**
>
> We have designed ggsql to be modular, both when it comes to the database input and the final rendering. For the first phase of the development we have chosen to use Vegalite as a renderer as it has allowed us to iterate quickly, but we do not envision Vegalite to remain the only, nor default writer in the future.

## Syntax & usage

> **NOTE:**
>
> Both spellings are supported - use whichever you prefer. The grammar accepts both British (`VISUALISE`) and American (`VISUALIZE`) spellings.

> **NOTE:**
>
> Add multiple `DRAW` clauses to your query. Each `DRAW` creates a new layer:
>
> ``` ggsql
> SELECT x, y FROM data
> VISUALISE x, y
> DRAW line 
>   MAPPING x AS x, y AS y
> DRAW point 
>   MAPPING x AS x, y AS y
> ```

> **NOTE:**
>
> Use the `LABEL` clause:
>
> ``` ggsql
> SELECT date, revenue FROM sales
> VISUALISE date AS x, revenue AS y
> DRAW line
> LABEL 
>   title => 'Sales Over Time', 
>   x => 'Date', 
>   y => 'Revenue (USD)'
> ```

> **NOTE:**
>
> Some parts of the syntax are passed on directly to the database, such as the `FILTER` and `ORDER BY` clauses in `DRAW`.
>
> ``` ggsql
> SELECT date, revenue FROM sales
> VISUALISE date AS x, revenue AS y
> DRAW line
> DRAW point
>   FILTER revenue = max(revenue)
> ```
>
> Further, any query before the `VISUALISE` clause is passed directly to the database so anything supported by your backend can go in there.

> **NOTE:**
>
> ggsql integrates very deeply with the database backends and handles all statistical transformations as SQL queries. This means that if you need to make a histogram of 1 billion observations, you’ll only ever fetch the values of each histogram bin, not the full dataset.

> **NOTE:**
>
> ggsql does not yet support interactive functionality like tool tips and zooming. This is a point of focus for us and we will rather get the syntax and implementation right than rush to it.

## Troubleshooting

> **NOTE:**
>
> Add `SCALE x VIA date` to tell ggsql to treat the x-axis as temporal data:
>
> ``` ggsql
> SELECT date, value FROM data
> VISUALISE date AS x, value AS y
> DRAW line
> SCALE x VIA date
> ```

> **NOTE:**
>
> Please open an issue on our [GitHub repository](https://github.com/posit-dev/ggsql/issues).
