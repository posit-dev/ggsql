//! Output writer abstraction layer for ggsql
//!
//! The writer module provides a pluggable interface for generating visualization
//! outputs from Plot + DataFrame combinations.
//!
//! # Architecture
//!
//! All writers implement the `Writer` trait, which provides:
//! - ResolvedPlot + Data → Output conversion
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
//!
//! Writers are configured by their own constructors, or generically from
//! key–value [`WriterOptions`] when a frontend collects settings from a user
//! without knowing which writer they picked.

use crate::reader::ResolvedSpec;
use crate::{DataFrame, GgsqlError, Plot, Result};
use std::collections::HashMap;

pub mod options;

pub use options::WriterOptions;

#[cfg(feature = "vegalite")]
pub mod vegalite;

#[cfg(feature = "vegalite")]
pub use vegalite::VegaLiteWriter;

// The raster writer is backed by the hephaestus renderer, which the module name
// records. That is an implementation detail: the writer is public as `PngWriter`
// and the module itself is not part of the API.
#[cfg(feature = "png")]
mod hephaestus;

#[cfg(feature = "png")]
pub use hephaestus::{rgba, Color, PngWriter};

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

    /// Construct the writer from free-form key–value options.
    ///
    /// This is the entry point for a frontend that collects settings from a
    /// user (`--writer-option width=1600`) and has no compile-time knowledge of
    /// the chosen writer. Implementations start by calling
    /// [`WriterOptions::reject_unknown`] so a mistyped key is reported instead
    /// of ignored, then fall back to their own defaults for anything unset.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if an option is unknown to this writer
    /// or its value cannot be interpreted.
    fn from_options(options: &WriterOptions) -> Result<Self>
    where
        Self: Sized;

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

    /// Render a ResolvedSpec (a resolved plot or table) to output format
    ///
    /// This is the main entry point for generating visualization output.
    /// Writers that don't support tables yet (all of them, as of this
    /// writing) return a `WriterError` for `ResolvedSpec::Table` rather than
    /// rejecting it at the type level — see the `ResolvedSpec::Table` arm
    /// below.
    ///
    /// # Arguments
    ///
    /// * `spec` - The resolved specification from `reader.execute()`
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
    fn render(&self, spec: &ResolvedSpec) -> Result<Self::Output> {
        match spec {
            ResolvedSpec::Plot(plot) => self.write(plot.plot(), plot.data()),
            ResolvedSpec::Table(_) => Err(GgsqlError::WriterError(
                "this writer does not support tables yet".to_string(),
            )),
        }
    }
}
