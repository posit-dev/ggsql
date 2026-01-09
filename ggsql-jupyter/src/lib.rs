//! ggsql-jupyter library
//!
//! This module exposes the internal components for testing.

pub mod display;
pub mod executor;
pub mod message;

// Re-export commonly used types
pub use display::format_display_data;
pub use executor::{ExecutionResult, QueryExecutor};
pub use message::{ConnectionInfo, JupyterMessage, MessageHeader};
