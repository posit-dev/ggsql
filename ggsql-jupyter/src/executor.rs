//! Query execution module for ggSQL Jupyter kernel
//!
//! This module handles the execution of ggSQL queries using the existing
//! ggsql library components (parser, DuckDB reader, Vega-Lite writer).

use anyhow::Result;
use ggsql::{
    execute::prepare_data,
    parser,
    reader::{DuckDBReader, Reader},
    writer::{VegaLiteWriter, Writer},
};
use polars::frame::DataFrame;

/// Result of executing a ggSQL query
#[derive(Debug)]
pub enum ExecutionResult {
    /// Pure SQL query with no visualization
    DataFrame(DataFrame),
    /// Query with visualization specification
    Visualization {
        spec: String, // Vega-Lite JSON
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

        // 1. Split query to check if there's a visualization
        let (_sql_part, viz_part) = parser::split_query(code)?;

        // 2. Check if there's a visualization
        if viz_part.is_empty() {
            // Pure SQL query - execute directly and return DataFrame
            let df = self.reader.execute(code)?;
            tracing::info!(
                "Pure SQL executed: {} rows, {} cols",
                df.height(),
                df.width()
            );
            return Ok(ExecutionResult::DataFrame(df));
        }

        // 3. Prepare data using shared execution logic (handles layer sources)
        let prepared = prepare_data(code, &self.reader)?;

        tracing::info!("Data sources prepared: {} sources", prepared.data.len());

        // 4. Generate Vega-Lite spec (use first spec if multiple)
        let vega_json = self.writer.write(&prepared.specs[0], &prepared.data)?;

        tracing::debug!("Generated Vega-Lite spec: {} chars", vega_json.len());

        // 6. Return result
        Ok(ExecutionResult::Visualization { spec: vega_json })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_visualization() {
        let executor = QueryExecutor::new().unwrap();
        let code = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";
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
