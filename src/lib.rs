/*!
# ggSQL - SQL Visualization Grammar

A SQL extension for declarative data visualization based on the Grammar of Graphics.

ggSQL allows you to write queries that combine SQL data retrieval with visualization
specifications in a single, composable syntax.

## Example

```sql
SELECT date, revenue, region
FROM sales
WHERE year = 2024
VISUALISE AS PLOT
DRAW line
    x = date,
    y = revenue,
    color = region
LABELS
    title = 'Sales by Region'
THEME
    style = 'minimal'
```

## Architecture

ggSQL splits queries at the `VISUALISE AS` boundary:
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

#[cfg(any(feature = "duckdb", feature = "postgres", feature = "sqlite"))]
pub mod reader;

#[cfg(any(feature = "vegalite", feature = "ggplot2", feature = "plotters"))]
pub mod writer;

// Re-export key types for convenience
pub use parser::{VizSpec, VizType, Layer, Scale, Geom, AestheticValue};

// Future modules - not yet implemented
// #[cfg(feature = "engine")]
// pub mod engine;

// DataFrame abstraction (wraps Polars)
pub use polars::prelude::DataFrame;

/// Main library error type
#[derive(thiserror::Error, Debug)]
pub enum GgsqlError {
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

pub type Result<T> = std::result::Result<T, GgsqlError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
#[cfg(all(feature = "duckdb", feature = "vegalite"))]
mod integration_tests {
    use super::*;
    use crate::parser::ast::{Layer, AestheticValue, Geom};
    use crate::reader::{DuckDBReader, Reader};
    use crate::writer::{VegaLiteWriter, Writer};

    #[test]
    fn test_end_to_end_date_type_preservation() {
        // Test complete pipeline: DuckDB → DataFrame (Date type) → VegaLite (temporal)

        // Create in-memory DuckDB with date data
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Execute SQL with DATE type
        let sql = r#"
            SELECT
                DATE '2024-01-01' + INTERVAL (n) DAY as date,
                n * 10 as revenue
            FROM generate_series(0, 4) as t(n)
        "#;

        let df = reader.execute(sql).unwrap();

        // Verify DataFrame has temporal type (DuckDB returns Datetime for DATE + INTERVAL)
        assert_eq!(df.get_column_names(), vec!["date", "revenue"]);
        let date_col = df.column("date").unwrap();
        // DATE + INTERVAL returns Datetime in DuckDB, which is still temporal
        assert!(matches!(
            date_col.dtype(),
            polars::prelude::DataType::Date | polars::prelude::DataType::Datetime(_, _)
        ));

        // Create visualization spec
        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("revenue".to_string()));
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // CRITICAL ASSERTION: x-axis should be automatically inferred as "temporal"
        assert_eq!(vl_spec["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");

        // Data values should be ISO temporal strings
        // (DuckDB returns Datetime for DATE + INTERVAL, so we get ISO datetime format)
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        let date_str = data_values[0]["date"].as_str().unwrap();
        assert!(date_str.starts_with("2024-01-01"), "Expected date starting with 2024-01-01, got {}", date_str);
    }

    #[test]
    fn test_end_to_end_datetime_type_preservation() {
        // Test complete pipeline: DuckDB → DataFrame (Datetime type) → VegaLite (temporal)

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Execute SQL with TIMESTAMP type
        let sql = r#"
            SELECT
                TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (n) HOUR as timestamp,
                n * 5 as value
            FROM generate_series(0, 3) as t(n)
        "#;

        let df = reader.execute(sql).unwrap();

        // Verify DataFrame has Datetime type
        let timestamp_col = df.column("timestamp").unwrap();
        assert!(matches!(
            timestamp_col.dtype(),
            polars::prelude::DataType::Datetime(_, _)
        ));

        // Create visualization spec
        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Area)
            .with_aesthetic("x".to_string(), AestheticValue::Column("timestamp".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should be automatically inferred as "temporal"
        assert_eq!(vl_spec["encoding"]["x"]["type"], "temporal");

        // Data values should be ISO datetime strings
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert!(data_values[0]["timestamp"].as_str().unwrap().starts_with("2024-01-01T"));
    }

    #[test]
    fn test_end_to_end_numeric_type_preservation() {
        // Test that numeric types are preserved (not converted to strings)

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Real SQL that users would write
        let sql = "SELECT 1 as int_col, 2.5 as float_col, true as bool_col";
        let df = reader.execute(sql).unwrap();

        // Verify types are preserved
        // DuckDB treats numeric literals as DECIMAL, which we convert to Float64
        assert!(matches!(df.column("int_col").unwrap().dtype(), polars::prelude::DataType::Int32));
        assert!(matches!(df.column("float_col").unwrap().dtype(), polars::prelude::DataType::Float64));
        assert!(matches!(df.column("bool_col").unwrap().dtype(), polars::prelude::DataType::Boolean));

        // Create visualization spec
        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("int_col".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("float_col".to_string()));
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Types should be inferred as quantitative
        assert_eq!(vl_spec["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");

        // Data values should be numbers (not strings!)
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["int_col"], 1);
        assert_eq!(data_values[0]["float_col"], 2.5);
        assert_eq!(data_values[0]["bool_col"], true);
    }

    #[test]
    fn test_end_to_end_mixed_types_with_nulls() {
        // Test that NULLs are handled correctly across different types

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sql = "SELECT * FROM (VALUES (1, 2.5, 'a'), (2, NULL, 'b'), (NULL, 3.5, NULL)) AS t(int_col, float_col, str_col)";
        let df = reader.execute(sql).unwrap();

        // Verify types
        assert!(matches!(df.column("int_col").unwrap().dtype(), polars::prelude::DataType::Int32));
        assert!(matches!(df.column("float_col").unwrap().dtype(), polars::prelude::DataType::Float64));
        assert!(matches!(df.column("str_col").unwrap().dtype(), polars::prelude::DataType::String));

        // Create viz spec
        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("int_col".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("float_col".to_string()));
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check null handling in JSON
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["int_col"], 1);
        assert_eq!(data_values[0]["float_col"], 2.5);
        assert_eq!(data_values[1]["float_col"], serde_json::Value::Null);
        assert_eq!(data_values[2]["int_col"], serde_json::Value::Null);
    }

    #[test]
    fn test_end_to_end_string_vs_categorical() {
        // Test that string columns are inferred as nominal type

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sql = "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 15), ('C', 30)) AS t(category, value)";
        let df = reader.execute(sql).unwrap();

        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // String columns should be inferred as nominal
        assert_eq!(vl_spec["encoding"]["x"]["type"], "nominal");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_end_to_end_time_series_aggregation() {
        // Test realistic time series query with aggregation

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create sample sales data and aggregate by day
        let sql = r#"
            WITH sales AS (
                SELECT
                    TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (n) HOUR as sale_time,
                    (n % 3) as product_id,
                    10 + (n % 5) as amount
                FROM generate_series(0, 23) as t(n)
            )
            SELECT
                DATE_TRUNC('day', sale_time) as day,
                SUM(amount) as total_sales,
                COUNT(*) as num_sales
            FROM sales
            GROUP BY day
        "#;

        let df = reader.execute(sql).unwrap();

        // Verify temporal type is preserved through aggregation
        // DATE_TRUNC returns Date type (not Datetime)
        let day_col = df.column("day").unwrap();
        assert!(matches!(
            day_col.dtype(),
            polars::prelude::DataType::Date | polars::prelude::DataType::Datetime(_, _)
        ));

        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("day".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("total_sales".to_string()));
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should be temporal
        assert_eq!(vl_spec["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_end_to_end_decimal_precision() {
        // Test that DECIMAL values with various precisions are correctly converted

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sql = "SELECT 0.1 as small, 123.456 as medium, 999999.999999 as large";
        let df = reader.execute(sql).unwrap();

        // All should be Float64
        assert!(matches!(df.column("small").unwrap().dtype(), polars::prelude::DataType::Float64));
        assert!(matches!(df.column("medium").unwrap().dtype(), polars::prelude::DataType::Float64));
        assert!(matches!(df.column("large").unwrap().dtype(), polars::prelude::DataType::Float64));

        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("small".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("medium".to_string()));
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check values are preserved
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        let small_val = data_values[0]["small"].as_f64().unwrap();
        let medium_val = data_values[0]["medium"].as_f64().unwrap();
        let large_val = data_values[0]["large"].as_f64().unwrap();

        assert!((small_val - 0.1).abs() < 0.001);
        assert!((medium_val - 123.456).abs() < 0.001);
        assert!((large_val - 999999.999999).abs() < 0.001);
    }

    #[test]
    fn test_end_to_end_integer_types() {
        // Test that different integer types are preserved

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sql = "SELECT CAST(1 AS TINYINT) as tiny, CAST(1000 AS SMALLINT) as small, CAST(1000000 AS INTEGER) as int, CAST(1000000000000 AS BIGINT) as big";
        let df = reader.execute(sql).unwrap();

        // Verify types
        assert!(matches!(df.column("tiny").unwrap().dtype(), polars::prelude::DataType::Int8));
        assert!(matches!(df.column("small").unwrap().dtype(), polars::prelude::DataType::Int16));
        assert!(matches!(df.column("int").unwrap().dtype(), polars::prelude::DataType::Int32));
        assert!(matches!(df.column("big").unwrap().dtype(), polars::prelude::DataType::Int64));

        let mut spec = VizSpec::new(VizType::Plot);
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("int".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("big".to_string()));
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // All integer types should be quantitative
        assert_eq!(vl_spec["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");

        // Check values
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["tiny"], 1);
        assert_eq!(data_values[0]["small"], 1000);
        assert_eq!(data_values[0]["int"], 1000000);
        assert_eq!(data_values[0]["big"], 1000000000000i64);
    }
}
