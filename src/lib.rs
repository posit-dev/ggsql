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

// Allow complex types in test code (e.g., test case tuples with many elements)
#![cfg_attr(test, allow(clippy::type_complexity))]

pub mod format;
pub mod naming;
pub mod parser;
pub mod plot;

pub mod reader;

#[cfg(any(feature = "vegalite", feature = "ggplot2", feature = "plotters"))]
pub mod writer;

pub mod execute;

pub mod validate;

// Re-export key types for convenience
pub use plot::{
    AestheticValue, DataSource, Facet, Geom, Layer, Mappings, Plot, Scale, SqlExpression,
};

// Re-export aesthetic classification utilities
pub use plot::aesthetic::{
    get_aesthetic_family, is_aesthetic_name, is_positional_aesthetic, is_primary_positional,
    primary_aesthetic, AESTHETIC_FAMILIES, ALL_POSITIONAL, NON_POSITIONAL, PRIMARY_POSITIONAL,
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
    use crate::plot::{AestheticValue, Geom, Layer};
    use crate::reader::{DuckDBReader, Reader};
    use crate::writer::{VegaLiteWriter, Writer};
    use std::collections::HashMap;

    /// Helper to wrap a DataFrame in a data map for testing (uses layer 0 key)
    fn wrap_data(df: DataFrame) -> HashMap<String, DataFrame> {
        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);
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

        let df = reader.execute_sql(sql).unwrap();

        // Verify DataFrame has temporal type (DuckDB returns Datetime for DATE + INTERVAL)
        assert_eq!(df.get_column_names(), vec!["date", "revenue"]);
        let date_col = df.column("date").unwrap();
        // DATE + INTERVAL returns Datetime in DuckDB, which is still temporal
        assert!(matches!(
            date_col.dtype(),
            polars::prelude::DataType::Date | polars::prelude::DataType::Datetime(_, _)
        ));

        // Create visualization spec
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
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
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
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

        let df = reader.execute_sql(sql).unwrap();

        // Verify DataFrame has Datetime type
        let timestamp_col = df.column("timestamp").unwrap();
        assert!(matches!(
            timestamp_col.dtype(),
            polars::prelude::DataType::Datetime(_, _)
        ));

        // Create visualization spec
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::area())
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
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
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
        let df = reader.execute_sql(sql).unwrap();

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
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
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
        let df = reader.execute_sql(sql).unwrap();

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
        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
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
        let df = reader.execute_sql(sql).unwrap();

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
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

        let df = reader.execute_sql(sql).unwrap();

        // Verify temporal type is preserved through aggregation
        // DATE_TRUNC returns Date type (not Datetime)
        let day_col = df.column("day").unwrap();
        assert!(matches!(
            day_col.dtype(),
            polars::prelude::DataType::Date | polars::prelude::DataType::Datetime(_, _)
        ));

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
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
        let df = reader.execute_sql(sql).unwrap();

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

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
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
        let df = reader.execute_sql(sql).unwrap();

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

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::bar())
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
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["tiny"], 1);
        assert_eq!(data_values[0]["small"], 1000);
        assert_eq!(data_values[0]["int"], 1000000);
        assert_eq!(data_values[0]["big"], 1000000000000i64);
    }

    #[test]
    fn test_end_to_end_constant_mappings() {
        // Test that constant values in MAPPING clauses work correctly
        // Constants are injected as aesthetic-named columns in each layer's data
        // With unified data approach, all layers are merged into one dataset with source filtering

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with layer-level constants
        let query = r#"
            SELECT 1 as x, 10 as y
            VISUALISE x, y
            DRAW line MAPPING 'value' AS linetype
            DRAW point MAPPING 'value2' AS shape
        "#;

        // Prepare data - this parses and processes the query
        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();

        // Each layer has its own data (different constants = different queries)
        assert_eq!(prepared.specs.len(), 1);

        // Layer 0 should have linetype column
        let layer0_key = prepared.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer 0 should have data_key");
        let layer0_df = prepared.data.get(layer0_key).unwrap();
        let linetype_col = naming::aesthetic_column("linetype");
        let layer0_cols = layer0_df.get_column_names();
        assert!(
            layer0_cols.iter().any(|c| c.as_str() == linetype_col),
            "Layer 0 should have linetype column '{}': {:?}",
            linetype_col,
            layer0_cols
        );

        // Layer 1 should have shape column
        let layer1_key = prepared.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("Layer 1 should have data_key");
        let layer1_df = prepared.data.get(layer1_key).unwrap();
        let shape_col = naming::aesthetic_column("shape");
        let layer1_cols = layer1_df.get_column_names();
        assert!(
            layer1_cols.iter().any(|c| c.as_str() == shape_col),
            "Layer 1 should have shape column '{}': {:?}",
            shape_col,
            layer1_cols
        );

        // Generate Vega-Lite
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&prepared.specs[0], &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Verify we have two layers
        assert_eq!(vl_spec["layer"].as_array().unwrap().len(), 2);

        // Verify the aesthetic is mapped to prefixed aesthetic-named columns
        // Note: linetype is mapped to Vega-Lite's strokeDash channel
        let layer0_linetype = &vl_spec["layer"][0]["encoding"]["strokeDash"];
        let layer1_shape = &vl_spec["layer"][1]["encoding"]["shape"];

        assert_eq!(
            layer0_linetype["field"].as_str().unwrap(),
            linetype_col,
            "Layer 0 linetype should map to prefixed aesthetic-named column"
        );
        assert_eq!(
            layer1_shape["field"].as_str().unwrap(),
            shape_col,
            "Layer 1 shape should map to prefixed aesthetic-named column"
        );

        // With unified data approach, all data is in a single dataset
        // Each row has __ggsql_source__ identifying which layer's data it belongs to
        let global_data = &vl_spec["data"]["values"];
        assert!(
            global_data.is_array(),
            "Should have unified global data array"
        );
        let global_rows = global_data.as_array().unwrap();

        // Find rows for each layer by their source field
        let layer0_rows: Vec<_> = global_rows
            .iter()
            .filter(|r| r[naming::SOURCE_COLUMN] == layer0_key.as_str())
            .collect();
        let layer1_rows: Vec<_> = global_rows
            .iter()
            .filter(|r| r[naming::SOURCE_COLUMN] == layer1_key.as_str())
            .collect();

        assert!(!layer0_rows.is_empty(), "Should have layer 0 rows");
        assert!(!layer1_rows.is_empty(), "Should have layer 1 rows");

        // Verify constant values
        assert_eq!(
            layer0_rows[0][&linetype_col], "value",
            "Layer 0 linetype constant should be 'value'"
        );
        assert_eq!(
            layer1_rows[0][&shape_col], "value2",
            "Layer 1 shape constant should be 'value2'"
        );
    }

    #[test]
    fn test_end_to_end_facet_with_constant_strokes() {
        // Test faceting with multiple layers that have constant stroke mappings
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
            DRAW line MAPPING revenue AS y, 'value' AS stroke
            DRAW point MAPPING revenue AS y, 'value2' AS stroke SETTING size => 30
            DRAW line MAPPING qty_scaled AS y, 'value3' AS stroke
            DRAW point MAPPING qty_scaled AS y, 'value4' AS stroke SETTING size => 30
            FACET region BY category
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();

        // With aesthetic-named columns, each layer gets its own data
        // Each layer should have its data with prefixed aesthetic-named columns
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        let stroke_col = naming::aesthetic_column("stroke");
        for layer_idx in 0..4 {
            let layer_key = naming::layer_key(layer_idx);
            assert!(
                prepared.data.contains_key(&layer_key),
                "Should have layer {} data",
                layer_idx
            );

            let layer_df = prepared.data.get(&layer_key).unwrap();
            let col_names = layer_df.get_column_names();

            // Each layer should have prefixed aesthetic-named columns
            assert!(
                col_names.iter().any(|c| c.as_str() == x_col),
                "Layer {} should have '{}' column: {:?}",
                layer_idx,
                x_col,
                col_names
            );
            assert!(
                col_names.iter().any(|c| c.as_str() == y_col),
                "Layer {} should have '{}' column: {:?}",
                layer_idx,
                y_col,
                col_names
            );
            // Stroke constant becomes a column named with prefixed aesthetic name
            assert!(
                col_names.iter().any(|c| c.as_str() == stroke_col),
                "Layer {} should have '{}' column: {:?}",
                layer_idx,
                stroke_col,
                col_names
            );
            // Facet columns should be included
            assert!(
                col_names.iter().any(|c| c.as_str() == "region"),
                "Layer {} should have 'region' facet column: {:?}",
                layer_idx,
                col_names
            );
            assert!(
                col_names.iter().any(|c| c.as_str() == "category"),
                "Layer {} should have 'category' facet column: {:?}",
                layer_idx,
                col_names
            );
        }

        // Note: With the new aesthetic-named columns approach, each layer has its own data.
        // Faceting with multiple data sources requires query deduplication (Phase 7 of the plan).
        // For now, we verify that the data structure is correct.
        // Query deduplication will enable: identical layer queries → shared data → faceting works.

        // Verify the spec has the facet configuration
        assert!(
            prepared.specs[0].facet.is_some(),
            "Spec should have facet configuration"
        );
    }

    #[test]
    fn test_end_to_end_global_constant_in_visualise() {
        // Test that global constants in VISUALISE clause work correctly
        // e.g., VISUALISE date AS x, value AS y, 'value' AS stroke

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

        // Query with global constant stroke in VISUALISE clause
        let query = r#"
            SELECT date, value FROM timeseries
            VISUALISE date AS x, value AS y, 'value' AS stroke
            DRAW line
            DRAW point SETTING size => 50
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();

        // Each layer should have a data_key
        let layer0_key = prepared.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer 0 should have data_key");
        let _layer1_key = prepared.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("Layer 1 should have data_key");

        // Both layers have data (may be shared or separate depending on query dedup)
        // Verify layer 0 has the expected columns
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        let stroke_col = naming::aesthetic_column("stroke");

        let layer_df = prepared.data.get(layer0_key).unwrap();
        let col_names = layer_df.get_column_names();

        assert!(
            col_names.iter().any(|c| c.as_str() == x_col),
            "Should have '{}' column: {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.iter().any(|c| c.as_str() == y_col),
            "Should have '{}' column: {:?}",
            y_col,
            col_names
        );
        assert!(
            col_names.iter().any(|c| c.as_str() == stroke_col),
            "Should have '{}' column: {:?}",
            stroke_col,
            col_names
        );

        // Generate Vega-Lite and verify it works
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(&prepared.specs[0], &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Both layers should have stroke field-mapped to prefixed aesthetic-named column
        assert_eq!(vl_spec["layer"].as_array().unwrap().len(), 2);
        assert_eq!(
            vl_spec["layer"][0]["encoding"]["stroke"]["field"]
                .as_str()
                .unwrap(),
            stroke_col
        );
        assert_eq!(
            vl_spec["layer"][1]["encoding"]["stroke"]["field"]
                .as_str()
                .unwrap(),
            stroke_col
        );

        // With unified data approach, all data is in the data.values array
        // Verify the stroke value appears in the unified data
        let global_data = vl_spec["data"]["values"]
            .as_array()
            .expect("Should have unified global data");

        // Find rows belonging to layer 0 (filter by source)
        let layer0_rows: Vec<_> = global_data
            .iter()
            .filter(|r| r[naming::SOURCE_COLUMN] == layer0_key.as_str())
            .collect();
        assert!(!layer0_rows.is_empty(), "Should have layer data rows");
        assert_eq!(layer0_rows[0][&stroke_col], "value");
    }
}
