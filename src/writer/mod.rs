//! Output writer abstraction layer for ggSQL
//!
//! The writer module provides a pluggable interface for generating visualization
//! outputs from VizSpec + DataFrame combinations.
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
//!
//! let writer = VegaLiteWriter::new();
//! let json = writer.write(&spec, &dataframe)?;
//! println!("{}", json);
//! ```

use std::collections::HashMap;
use crate::{Result, DataFrame, VizSpec};

#[cfg(feature = "vegalite")]
pub mod vegalite;

#[cfg(feature = "vegalite")]
pub use vegalite::VegaLiteWriter;

/// Trait for visualization output writers
///
/// Writers take a VizSpec and data sources and produce formatted output
/// (JSON, R code, PNG bytes, etc.).
pub trait Writer {
    /// Generate output from a visualization specification and data sources
    ///
    /// # Arguments
    ///
    /// * `spec` - The parsed ggSQL specification
    /// * `data` - A map of data source names to DataFrames. The writer decides
    ///   how to use these based on the spec's layer configurations.
    ///
    /// # Returns
    ///
    /// A string containing the formatted output (JSON, code, etc.)
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if:
    /// - The spec is incompatible with this writer
    /// - The data doesn't match the spec's requirements
    /// - Output generation fails
    fn write(&self, spec: &VizSpec, data: &HashMap<String, DataFrame>) -> Result<String>;

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
    fn validate(&self, spec: &VizSpec) -> Result<()>;
}
