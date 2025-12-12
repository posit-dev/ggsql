//! Query execution module for ggSQL Jupyter kernel
//!
//! This module handles the execution of ggSQL queries using the existing
//! ggsql library components (parser, DuckDB reader, Vega-Lite writer).

use anyhow::Result;
use polars::frame::DataFrame;
use ggsql::{parser, reader::{DuckDBReader, Reader}, writer::{VegaLiteWriter, Writer}};

/// Result of executing a ggSQL query
#[derive(Debug)]
pub enum ExecutionResult {
    /// Pure SQL query with no visualization
    DataFrame(DataFrame),
    /// Query with visualization specification
    Visualization {
        spec: String,      // Vega-Lite JSON
        data_rows: usize,
        data_cols: usize,
    },
}

/// Query executor maintaining persistent DuckDB connection
pub struct QueryExecutor {
    reader: DuckDBReader,
    writer: VegaLiteWriter,
}

impl QueryExecutor {
    /// Create a new query executor with in-memory DuckDB database
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing query executor with in-memory DuckDB");
        let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
        let writer = VegaLiteWriter::new();

        Ok(Self { reader, writer })
    }

    /// Execute a ggSQL query
    ///
    /// This handles both pure SQL queries and queries with VISUALISE clauses.
    ///
    /// # Arguments
    ///
    /// * `code` - The ggSQL query to execute
    ///
    /// # Returns
    ///
    /// An ExecutionResult containing either a DataFrame (for pure SQL) or
    /// a Visualization (for queries with VISUALISE clause)
    pub fn execute(&self, code: &str) -> Result<ExecutionResult> {
        tracing::debug!("Executing query: {} chars", code.len());

        // 1. Split query into SQL and VIZ parts
        let (sql_part, viz_part) = parser::split_query(code)?;

        tracing::debug!("SQL part: {} chars", sql_part.len());
        tracing::debug!("VIZ part: {} chars", viz_part.len());

        // 2. Execute SQL part
        let df = self.reader.execute(&sql_part)?;
        tracing::info!("Query executed: {} rows, {} cols", df.height(), df.width());

        // 3. Check if there's a visualization
        if viz_part.is_empty() {
            // Pure SQL query - return DataFrame
            return Ok(ExecutionResult::DataFrame(df));
        }

        // 4. Parse VIZ specification
        let viz_specs = parser::parse_query(code)?;

        if viz_specs.is_empty() {
            anyhow::bail!("No visualization specification found despite VISUALISE keyword");
        }

        // 5. Generate Vega-Lite spec (use first spec if multiple)
        let vega_json = self.writer.write(&viz_specs[0], &df)?;

        tracing::debug!("Generated Vega-Lite spec: {} chars", vega_json.len());

        // 6. Return result
        Ok(ExecutionResult::Visualization {
            spec: vega_json,
            data_rows: df.height(),
            data_cols: df.width(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_visualization() {
        let executor = QueryExecutor::new().unwrap();
        let code = "SELECT 1 as x, 2 as y VISUALISE AS PLOT DRAW point USING x=x, y=y";
        let result = executor.execute(code).unwrap();

        assert!(matches!(result, ExecutionResult::Visualization { .. }));
    }

    #[test]
    fn test_pure_sql() {
        let executor = QueryExecutor::new().unwrap();
        let code = "SELECT 1 as x, 2 as y";
        let result = executor.execute(code).unwrap();

        assert!(matches!(result, ExecutionResult::DataFrame(_)));
    }

    #[test]
    fn test_error_handling() {
        let executor = QueryExecutor::new().unwrap();
        let code = "SELECT * FROM nonexistent_table";
        let result = executor.execute(code);

        assert!(result.is_err());
    }
}
