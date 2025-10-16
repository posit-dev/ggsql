/*!
# VizQL - SQL Visualization Grammar

A SQL extension for declarative data visualization based on the Grammar of Graphics.

VizQL allows you to write queries that combine SQL data retrieval with visualization
specifications in a single, composable syntax.

## Example

```sql
SELECT date, revenue, region
FROM sales
WHERE year = 2024
VISUALISE AS PLOT
WITH line
    x = date,
    y = revenue,
    color = region
LABELS
    title = 'Sales by Region'
THEME
    style = 'minimal'
```

## Architecture

VizQL splits queries at the `VISUALISE AS` boundary:
- **SQL portion** → passed to pluggable readers (DuckDB, PostgreSQL, CSV, etc.)
- **VISUALISE portion** → parsed and compiled into visualization specifications
- **Output** → rendered via pluggable writers (ggplot2, PNG, Vega-Lite, etc.)

## Core Components

- [`parser`] - Query parsing and AST generation
- [`engine`] - Core execution engine
- [`readers`] - Data source abstraction layer
- [`writers`] - Output format abstraction layer
*/

pub mod parser;

// Re-export key types for convenience
pub use parser::{VizSpec, VizType, Layer, Scale, Geom, AestheticValue};

// Future modules - not yet implemented
// #[cfg(feature = "engine")]
// pub mod engine;

// #[cfg(any(feature = "duckdb", feature = "postgres", feature = "sqlite"))]
// pub mod readers;

// #[cfg(any(feature = "plotters"))]
// pub mod writers;

// DataFrame abstraction (wraps Polars)
pub use polars::prelude::DataFrame;

/// Main library error type
#[derive(thiserror::Error, Debug)]
pub enum VizqlError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Data source error: {0}")]
    ReaderError(String),

    #[error("Output generation error: {0}")]
    WriterError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, VizqlError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");