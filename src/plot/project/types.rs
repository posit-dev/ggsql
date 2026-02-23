//! Projection types for ggsql visualization specifications
//!
//! This module defines projection configuration and types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::types::ParameterValue;

/// Projection (from PROJECT clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    /// Coordinate system type
    pub coord: Coord,
    /// Projection-specific options
    pub properties: HashMap<String, ParameterValue>,
}

/// Coordinate system types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Coord {
    Cartesian,
    Polar,
    Flip,
    Fixed,
    Trans,
    Map,
    QuickMap,
}
