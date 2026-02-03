//! Query execution module for ggsql Jupyter kernel
//!
//! This module handles the execution of ggsql queries using the existing
//! ggsql library components (parser, DuckDB reader, Vega-Lite writer).

use anyhow::Result;
use ggsql::{
    reader::{DuckDBReader, Reader},
    validate,
    writer::{VegaLiteWriter, Writer},
};
use polars::frame::DataFrame;

/// Result of executing a ggsql query
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

    /// Execute a ggsql query
    ///
    /// This handles both pure SQL queries and queries with VISUALISE clauses.
    ///
    /// # Arguments
    ///
    /// * `code` - The ggsql query to execute
    ///
    /// # Returns
    ///
    /// An ExecutionResult containing either a DataFrame (for pure SQL) or
    /// a Visualization (for queries with VISUALISE clause)
    pub fn execute(&self, code: &str) -> Result<ExecutionResult> {
        tracing::debug!("Executing query: {} chars", code.len());

        // 1. Validate to check if there's a visualization
        let validated = validate(code)?;

        // 2. Check if there's a visualization
        if !validated.has_visual() {
            // Pure SQL query - execute directly and return DataFrame
            let df = self.reader.execute_sql(code)?;
            tracing::info!(
                "Pure SQL executed: {} rows, {} cols",
                df.height(),
                df.width()
            );
            return Ok(ExecutionResult::DataFrame(df));
        }

        // 3. Execute ggsql query using reader
        let spec = self.reader.execute(code)?;

        tracing::info!(
            "Query executed: {} rows, {} layers",
            spec.metadata().rows,
            spec.metadata().layer_count
        );

        // 4. Render to output format
        let vega_json = self.writer.render(&spec)?;

        tracing::debug!("Generated Vega-Lite spec: {} chars", vega_json.len());

        // 5. Return result
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
