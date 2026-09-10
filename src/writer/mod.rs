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
//!
//! Writers are configured by their own constructors, or generically from
//! key–value [`WriterOptions`] when a frontend collects settings from a user
//! without knowing which writer they picked.

use crate::reader::Spec;
use crate::{DataFrame, Plot, Result};
use std::collections::HashMap;

pub mod options;

pub use options::WriterOptions;

#[cfg(feature = "vegalite")]
pub mod vegalite;

#[cfg(feature = "vegalite")]
pub use vegalite::VegaLiteWriter;

// The renderer-backed writers live in one private module named after the
// renderer they share; each is public under its own format's name. Gated on
// `graphics`, the shared composition layer, rather than on any one format.
#[cfg(feature = "graphics")]
// `graphics` and `raster-writer` are internal features the writer features turn
// on. Selecting one alone is legitimate — `cargo tree --features graphics`
// proves the vector path pulls in no wgpu — but leaves the composition layer
// with no consumer, so silence that case only. Any build with an actual writer
// still reports real dead code.
#[cfg_attr(
    not(any(
        feature = "png",
        feature = "jpeg",
        feature = "tiff",
        feature = "webp",
        feature = "svg",
        feature = "pdf",
        feature = "hep",
        feature = "window"
    )),
    allow(dead_code)
)]
mod hephaestus;

#[cfg(feature = "graphics")]
pub use hephaestus::{rgba, Canvas, Color};

#[cfg(feature = "raster-writer")]
pub use hephaestus::{RasterRenderer, MAX_RASTER_DIMENSION};

#[cfg(feature = "jpeg")]
pub use hephaestus::JpegWriter;
#[cfg(feature = "webp")]
pub use hephaestus::WebpWriter;

#[cfg(feature = "hep")]
pub use hephaestus::HepWriter;
#[cfg(feature = "pdf")]
pub use hephaestus::PdfWriter;
#[cfg(feature = "svg")]
pub use hephaestus::SvgWriter;

// Not a writer — it produces no output — but it needs the same composition, so
// it lives beside them. See its own docs for why it is not a `Writer` impl.
#[cfg(feature = "window")]
pub use hephaestus::PlotViewer;
#[cfg(feature = "png")]
pub use hephaestus::{PngCompression, PngWriter};
#[cfg(feature = "tiff")]
pub use hephaestus::{TiffCompression, TiffWriter};

/// Trait for visualization output writers
///
/// Writers take a Plot and data sources and produce formatted output
/// (JSON, R code, PNG bytes, etc.).
///
/// # Associated Types
///
/// * `Output` - The type returned by `write()` and `render()`: `String` for a
///   text format, `Vec<u8>` for a binary one. Never an `Option` — failure is
///   the `Result`'s business — and a type producing nothing is not a writer.
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
