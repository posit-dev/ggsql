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
    /// Position aesthetic names (resolved: explicit or coord defaults)
    /// Always populated after building - never empty.
    /// e.g., ["x", "y"] for cartesian, ["radius", "angle"] for polar,
    /// or custom names like ["a", "b"] if user specifies them.
    pub aesthetics: Vec<String>,
    /// Projection-specific options
    pub properties: HashMap<String, ParameterValue>,
}

impl Projection {
    /// Create a default Cartesian projection (x, y).
    pub fn cartesian() -> Self {
        Self::with_defaults(Coord::cartesian())
    }

    /// Create a default Polar projection (radius, angle).
    pub fn polar() -> Self {
        Self::with_defaults(Coord::polar())
    }

    fn with_defaults(coord: Coord) -> Self {
        let aesthetics = coord
            .position_aesthetic_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self {
            coord,
            aesthetics,
            properties: HashMap::new(),
        }
    }

    /// Get the position aesthetic names as string slices.
    /// (aesthetics are always resolved at build time)
    pub fn position_names(&self) -> Vec<&str> {
        self.aesthetics.iter().map(|s| s.as_str()).collect()
    }
}
