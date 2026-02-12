//! Output writer abstraction layer for ggsql
//!
//! The writer module provides a pluggable interface for generating visualization
//! outputs from Plot + DataFrame combinations.
//!
//! # Architecture
//!
//! All writers implement the `Writer` trait, which provides:
//! - Spec + Data → Output conversion
//! - Validation for writer compatibility
//! - Format-specific rendering logic
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::writer::{Writer, VegaLiteWriter};
//! use ggsql::reader::{Reader, DuckDBReader};
//!
//! let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
//!
//! let writer = VegaLiteWriter::new();
//! let json = writer.render(&spec)?;
//! println!("{}", json);
//! ```

use crate::reader::Spec;
use crate::{DataFrame, Plot, Result};
use std::collections::HashMap;

#[cfg(feature = "vegalite")]
pub mod vegalite;

#[cfg(feature = "vegalite")]
pub use vegalite::VegaLiteWriter;

/// Trait for visualization output writers
///
/// Writers take a Plot and data sources and produce formatted output
/// (JSON, R code, PNG bytes, etc.).
///
/// # Associated Types
///
/// * `Output` - The type returned by `write()` and `render()`. Use `Option<String>`
///   for text output, `Option<Vec<u8>>` for binary, `()` for void writers, etc.
pub trait Writer {
    /// The output type produced by this writer.
    type Output;

    /// Generate output from a visualization specification and data sources
    ///
    /// # Arguments
    ///
    /// * `spec` - The parsed ggsql specification
    /// * `data` - A map of data source names to DataFrames. The writer decides
    ///   how to use these based on the spec's layer configurations.
    ///
    /// # Returns
    ///
    /// The writer's output, depends on writer implementation.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if:
    /// - The spec is incompatible with this writer
    /// - The data doesn't match the spec's requirements
    /// - Output generation fails
    fn write(&self, spec: &Plot, data: &HashMap<String, DataFrame>) -> Result<Self::Output>;

    /// Validate that a spec is compatible with this writer
    ///
    /// Checks whether the spec can be rendered by this writer without
    /// actually generating output.
    ///
    /// # Arguments
    ///
    /// * `spec` - The visualization specification to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if the spec is compatible, otherwise an error
    fn validate(&self, spec: &Plot) -> Result<()>;

    /// Render a Spec to output format
    ///
    /// This is the main entry point for generating visualization output.
    ///
    /// # Arguments
    ///
    /// * `spec` - The prepared visualization specification from `reader.execute()`
    ///
    /// # Returns
    ///
    /// The writer's output (type depends on writer implementation)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ggsql::reader::{Reader, DuckDBReader};
    /// use ggsql::writer::{Writer, VegaLiteWriter};
    ///
    /// let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
    /// let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
    ///
    /// let writer = VegaLiteWriter::new();
    /// let json = writer.render(&spec)?;
    /// ```
    fn render(&self, spec: &Spec) -> Result<Self::Output> {
        self.write(spec.plot(), spec.data())
    }
}
