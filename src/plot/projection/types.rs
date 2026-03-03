//! Projection types for ggsql visualization specifications
//!
//! This module defines projection configuration and types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::coord::Coord;
use crate::plot::ParameterValue;

/// Projection (from PROJECT clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    /// Coordinate system type
    pub coord: Coord,
    /// Positional aesthetic names (resolved: explicit or coord defaults)
    /// Always populated after building - never empty.
    /// e.g., ["x", "y"] for cartesian, ["theta", "radius"] for polar,
    /// or custom names like ["a", "b"] if user specifies them.
    pub aesthetics: Vec<String>,
    /// Projection-specific options
    pub properties: HashMap<String, ParameterValue>,
}

impl Projection {
    /// Get the positional aesthetic names as string slices.
    /// (aesthetics are always resolved at build time)
    pub fn positional_names(&self) -> Vec<&str> {
        self.aesthetics.iter().map(|s| s.as_str()).collect()
    }
}
