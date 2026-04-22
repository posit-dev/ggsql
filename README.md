# <a href="https://ggsql.org"><img src="doc/assets/logo.png" height="50px" alt="ggsql website" /></a> ggsql

A SQL extension for declarative data visualization based on the Grammar of Graphics.

ggsql allows you to write queries that combine SQL data retrieval with visualization specifications in a single, composable syntax.

## Example

```ggsql
SELECT date, revenue, region
FROM sales
WHERE year = 2024

VISUALISE date AS x, revenue AS y, region AS color
DRAW line
SCALE x
  SETTING breaks => 'month'
LABEL title => 'Sales by Region'
```

## Why?
Many data analysts are naturally at home in SQL and spend more time there than in a programming language like Python or R. Having to extract data, context switch to a new programming language, import data, etc. is cumbersome when all you want to do is understand the data you are working with *right now*.

ggsql is built for immediate familiarity and alignment with the SQL language. It is further built on the foundation of the grammar of graphics known from [ggplot2](https://ggplot2.tidyverse.org/) which affords a composable syntax capable of simple as well as arbitrarily complex visualizations.

The syntax has been designed to be easy to learn, read, and write. This also means that it is a great fit for AI agents to produce as the output query is immediately easy to understand and validate by the user so that you can have certainty in its validity.

## Project status
We are approaching an alpha release with the main architectural parts finished. Future development will focus on adding new readers (database support) and writers (output types) to compliment the DuckDB/SQLite + Vegalite setup we have focused on during early development.

## Installation
Please follow the instructions on [the website](https://ggsql.org/get_started/installation.html) for up to date information on how to install ggsql.

## Try it out
ggsql compiles to WASM and can thus be embedded in a website. You can try it out on our [playground](https://ggsql.org/wasm/) (no installation required).

## Learn more
Browse [the documentation](https://ggsql.org/syntax/) to learn of all ggsql has to offer. Complete with interactive examples to try out.
