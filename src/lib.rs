/*!
# ggsql - SQL Visualization Grammar

A SQL extension for declarative data visualization based on the Grammar of Graphics.

ggsql allows you to write queries that combine SQL data retrieval with visualization
specifications in a single, composable syntax.

## Example

```sql
SELECT date, revenue, region
FROM sales
WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
LABEL title => 'Sales by Region'
THEME minimal
```

## Architecture

ggsql splits queries at the `VISUALISE` boundary:
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

#[cfg(feature = "duckdb")]
pub mod execute;

// Re-export key types for convenience
pub use parser::{
    AestheticValue, DataSource, Facet, Geom, Layer, Mappings, Scale, SqlExpression, VizSpec,
};

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
    use crate::parser::ast::{AestheticValue, Geom, Layer};
    use crate::reader::{DuckDBReader, Reader};
    use crate::writer::{VegaLiteWriter, Writer};
    use std::collections::HashMap;

    /// Helper to wrap a DataFrame in a data map for testing
    fn wrap_data(df: DataFrame) -> HashMap<String, DataFrame> {
        let mut data_map = HashMap::new();
        data_map.insert("__global__".to_string(), df);
        data_map
    }

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
        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("date".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("revenue".to_string()),
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // CRITICAL ASSERTION: x-axis should be automatically inferred as "temporal"
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");

        // Data values should be ISO temporal strings
        // (DuckDB returns Datetime for DATE + INTERVAL, so we get ISO datetime format)
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
        let date_str = data_values[0]["date"].as_str().unwrap();
        assert!(
            date_str.starts_with("2024-01-01"),
            "Expected date starting with 2024-01-01, got {}",
            date_str
        );
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
        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Area)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("timestamp".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should be automatically inferred as "temporal"
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "temporal");

        // Data values should be ISO datetime strings
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
        assert!(data_values[0]["timestamp"]
            .as_str()
            .unwrap()
            .starts_with("2024-01-01T"));
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
        assert!(matches!(
            df.column("int_col").unwrap().dtype(),
            polars::prelude::DataType::Int32
        ));
        assert!(matches!(
            df.column("float_col").unwrap().dtype(),
            polars::prelude::DataType::Float64
        ));
        assert!(matches!(
            df.column("bool_col").unwrap().dtype(),
            polars::prelude::DataType::Boolean
        ));

        // Create visualization spec
        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("int_col".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("float_col".to_string()),
            );
        spec.layers.push(layer);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Types should be inferred as quantitative
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");

        // Data values should be numbers (not strings!)
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
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
        assert!(matches!(
            df.column("int_col").unwrap().dtype(),
            polars::prelude::DataType::Int32
        ));
        assert!(matches!(
            df.column("float_col").unwrap().dtype(),
            polars::prelude::DataType::Float64
        ));
        assert!(matches!(
            df.column("str_col").unwrap().dtype(),
            polars::prelude::DataType::String
        ));

        // Create viz spec
        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("int_col".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("float_col".to_string()),
            );
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check null handling in JSON
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
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

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("category".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("value".to_string()),
            );
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // String columns should be inferred as nominal
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "nominal");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");
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

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("day".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("total_sales".to_string()),
            );
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should be temporal
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_end_to_end_decimal_precision() {
        // Test that DECIMAL values with various precisions are correctly converted

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sql = "SELECT 0.1 as small, 123.456 as medium, 999999.999999 as large";
        let df = reader.execute(sql).unwrap();

        // All should be Float64
        assert!(matches!(
            df.column("small").unwrap().dtype(),
            polars::prelude::DataType::Float64
        ));
        assert!(matches!(
            df.column("medium").unwrap().dtype(),
            polars::prelude::DataType::Float64
        ));
        assert!(matches!(
            df.column("large").unwrap().dtype(),
            polars::prelude::DataType::Float64
        ));

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("small".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("medium".to_string()),
            );
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check values are preserved
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
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
        assert!(matches!(
            df.column("tiny").unwrap().dtype(),
            polars::prelude::DataType::Int8
        ));
        assert!(matches!(
            df.column("small").unwrap().dtype(),
            polars::prelude::DataType::Int16
        ));
        assert!(matches!(
            df.column("int").unwrap().dtype(),
            polars::prelude::DataType::Int32
        ));
        assert!(matches!(
            df.column("big").unwrap().dtype(),
            polars::prelude::DataType::Int64
        ));

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic(
                "x".to_string(),
                AestheticValue::standard_column("int".to_string()),
            )
            .with_aesthetic(
                "y".to_string(),
                AestheticValue::standard_column("big".to_string()),
            );
        spec.layers.push(layer);

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&spec, &wrap_data(df)).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // All integer types should be quantitative
        assert_eq!(vl_spec["layer"][0]["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["layer"][0]["encoding"]["y"]["type"], "quantitative");

        // Check values
        let data_values = vl_spec["datasets"]["__global__"].as_array().unwrap();
        assert_eq!(data_values[0]["tiny"], 1);
        assert_eq!(data_values[0]["small"], 1000);
        assert_eq!(data_values[0]["int"], 1000000);
        assert_eq!(data_values[0]["big"], 1000000000000i64);
    }

    #[test]
    fn test_end_to_end_constant_mappings() {
        // Test that constant values in MAPPING clauses work correctly
        // Constants are injected into global data with layer-indexed column names
        // This allows faceting to work (all layers share same data source)

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with layer-level constants (layers use global data, no filter)
        let query = r#"
            SELECT 1 as x, 10 as y
            VISUALISE x, y
            DRAW line MAPPING 'value' AS color
            DRAW point MAPPING 'value2' AS color
        "#;

        // Prepare data - this parses, injects constants into global data, and replaces literals with columns
        let prepared =
            execute::prepare_data_with_executor(query, |sql| reader.execute(sql)).unwrap();

        // Verify constants were injected into global data (not layer-specific data)
        // Both layers share __global__ data for faceting compatibility
        assert!(
            prepared.data.contains_key("__global__"),
            "Should have global data with constants injected"
        );
        // Layers without filters should NOT have their own data entries
        assert!(
            !prepared.data.contains_key("__layer_0__"),
            "Layer 0 should use global data, not layer-specific data"
        );
        assert!(
            !prepared.data.contains_key("__layer_1__"),
            "Layer 1 should use global data, not layer-specific data"
        );
        assert_eq!(prepared.specs.len(), 1);

        // Verify global data contains layer-indexed constant columns
        let global_df = prepared.data.get("__global__").unwrap();
        let col_names = global_df.get_column_names();
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_0__"),
            "Global data should have layer 0 color constant: {:?}",
            col_names
        );
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_1__"),
            "Global data should have layer 1 color constant: {:?}",
            col_names
        );

        // Generate Vega-Lite
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&prepared.specs[0], &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Verify we have two layers
        assert_eq!(vl_spec["layer"].as_array().unwrap().len(), 2);

        // Verify the color aesthetic is mapped to layer-indexed synthetic columns
        let layer0_color = &vl_spec["layer"][0]["encoding"]["color"];
        let layer1_color = &vl_spec["layer"][1]["encoding"]["color"];

        // Color should be field-mapped to layer-indexed columns
        assert_eq!(
            layer0_color["field"].as_str().unwrap(),
            "__ggsql_const_color_0__",
            "Layer 0 color should map to layer-indexed column"
        );
        assert_eq!(
            layer1_color["field"].as_str().unwrap(),
            "__ggsql_const_color_1__",
            "Layer 1 color should map to layer-indexed column"
        );

        // All layers should use the same global data
        let global_data = &vl_spec["datasets"]["__global__"];
        assert!(global_data.is_array(), "Should have global data array");

        // Verify constant values appear in the global data with layer-indexed names
        let data_row = &global_data.as_array().unwrap()[0];
        assert_eq!(
            data_row["__ggsql_const_color_0__"], "value",
            "Layer 0 constant should be 'value'"
        );
        assert_eq!(
            data_row["__ggsql_const_color_1__"], "value2",
            "Layer 1 constant should be 'value2'"
        );
    }

    #[test]
    fn test_end_to_end_facet_with_constant_colors() {
        // Test faceting with multiple layers that have constant color mappings
        // This verifies the fix for faceting compatibility with constants

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with multiple groups for faceting
        reader
            .connection()
            .execute(
                "CREATE TABLE facet_test AS SELECT * FROM (VALUES
                    ('2023-01-01'::DATE, 100.0, 50, 'North', 'A'),
                    ('2023-02-01'::DATE, 120.0, 60, 'North', 'A'),
                    ('2023-01-01'::DATE, 80.0, 40, 'South', 'B'),
                    ('2023-02-01'::DATE, 90.0, 45, 'South', 'B')
                ) AS t(month, revenue, quantity, region, category)",
                duckdb::params![],
            )
            .unwrap();

        // Query with multiple constant-colored layers and faceting
        let query = r#"
            SELECT month, region, category, revenue, quantity * 10 as qty_scaled
            FROM facet_test
            VISUALISE month AS x
            DRAW line MAPPING revenue AS y, 'value' AS color
            DRAW point MAPPING revenue AS y, 'value2' AS color SETTING size => 30
            DRAW line MAPPING qty_scaled AS y, 'value3' AS color
            DRAW point MAPPING qty_scaled AS y, 'value4' AS color SETTING size => 30
            SCALE x SETTING type => 'date'
            FACET region BY category
        "#;

        let prepared =
            execute::prepare_data_with_executor(query, |sql| reader.execute(sql)).unwrap();

        // All layers should use global data for faceting to work
        assert!(
            prepared.data.contains_key("__global__"),
            "Should have global data"
        );
        // No layer-specific data should be created
        assert!(
            !prepared.data.contains_key("__layer_0__"),
            "Layer 0 should use global data"
        );
        assert!(
            !prepared.data.contains_key("__layer_1__"),
            "Layer 1 should use global data"
        );
        assert!(
            !prepared.data.contains_key("__layer_2__"),
            "Layer 2 should use global data"
        );
        assert!(
            !prepared.data.contains_key("__layer_3__"),
            "Layer 3 should use global data"
        );

        // Verify global data has all layer-indexed constant columns
        let global_df = prepared.data.get("__global__").unwrap();
        let col_names = global_df.get_column_names();
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_0__"),
            "Should have layer 0 color constant"
        );
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_1__"),
            "Should have layer 1 color constant"
        );
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_2__"),
            "Should have layer 2 color constant"
        );
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_3__"),
            "Should have layer 3 color constant"
        );

        // Generate Vega-Lite and verify faceting structure
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&prepared.specs[0], &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Should have facet structure (row and column)
        assert!(
            vl_spec["facet"]["row"].is_object() || vl_spec["facet"]["column"].is_object(),
            "Should have facet structure: {:?}",
            vl_spec["facet"]
        );
    }

    #[test]
    fn test_end_to_end_global_constant_in_visualise() {
        // Test that global constants in VISUALISE clause work correctly
        // e.g., VISUALISE date AS x, value AS y, 'value' AS color

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE timeseries AS SELECT * FROM (VALUES
                    ('2023-01-01'::DATE, 100.0),
                    ('2023-01-08'::DATE, 110.0),
                    ('2023-01-15'::DATE, 105.0)
                ) AS t(date, value)",
                duckdb::params![],
            )
            .unwrap();

        // Query with global constant color in VISUALISE clause
        let query = r#"
            SELECT date, value FROM timeseries
            VISUALISE date AS x, value AS y, 'value' AS color
            DRAW line
            DRAW point SETTING size => 50
            SCALE x SETTING type => 'date'
        "#;

        let prepared =
            execute::prepare_data_with_executor(query, |sql| reader.execute(sql)).unwrap();

        // Should have global data with the constant injected
        assert!(
            prepared.data.contains_key("__global__"),
            "Should have global data"
        );

        // Verify global data has the constant columns for both layers
        let global_df = prepared.data.get("__global__").unwrap();
        let col_names = global_df.get_column_names();
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_0__"),
            "Should have layer 0 color constant: {:?}",
            col_names
        );
        assert!(
            col_names.iter().any(|c| *c == "__ggsql_const_color_1__"),
            "Should have layer 1 color constant: {:?}",
            col_names
        );

        // Generate Vega-Lite and verify it works
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&prepared.specs[0], &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Both layers should have color field-mapped to their indexed constant columns
        assert_eq!(vl_spec["layer"].as_array().unwrap().len(), 2);
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["color"]["field"]
                .as_str()
                .unwrap(),
            "__ggsql_const_color_0__"
        );
        assert_eq!(
            vl_spec["layer"][1]["encoding"]["color"]["field"]
                .as_str()
                .unwrap(),
            "__ggsql_const_color_1__"
        );

        // Both constants should have the same value "value"
        let data = &vl_spec["datasets"]["__global__"].as_array().unwrap()[0];
        assert_eq!(data["__ggsql_const_color_0__"], "value");
        assert_eq!(data["__ggsql_const_color_1__"], "value");
    }
}
