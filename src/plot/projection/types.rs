//! Projection types for ggsql visualization specifications
//!
//! This module defines projection configuration and types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::coord::Coord;
use crate::plot::{Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

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
    /// Values computed at execution time (e.g., clip boundary WKT).
    /// Not user-facing; populated by apply_projection_transforms.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub computed: HashMap<String, ParameterValue>,
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

    /// Create a default Map projection (lon, lat).
    pub fn map() -> Self {
        Self::with_defaults(Coord::map("crs", &HashMap::new()))
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
            computed: HashMap::new(),
        }
    }

    /// Get the position aesthetic names as string slices.
    /// (aesthetics are always resolved at build time)
    pub fn position_names(&self) -> Vec<&str> {
        self.aesthetics.iter().map(|s| s.as_str()).collect()
    }

    /// Orchestrate projection transforms for all layers.
    pub fn apply_projection_transforms(
        &mut self,
        layers: &mut [Layer],
        layer_queries: &mut [String],
        dialect: &dyn SqlDialect,
        execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
    ) -> crate::Result<()> {
        let coord = self.coord.clone();
        coord.apply_projection_transforms(layers, layer_queries, self, dialect, execute_query)
    }
}
