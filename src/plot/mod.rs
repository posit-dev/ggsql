//! Plot types for ggsql visualization specifications
//!
//! This module contains all the types that represent a parsed ggsql visualization
//! specification, including the main Plot struct, layers, geoms, scales, facets,
//! coordinates, and input types.
//!
//! # Architecture
//!
//! The module is organized into submodules:
//!
//! - `main` - Main Plot struct and Labels/Theme types
//! - `types` - Value types: Mappings, AestheticValue, ParameterValue, etc.
//! - `layer` - Layer struct and Geom subsystem
//! - `scale` - Scale and Guide types
//! - `facet` - Facet types for small multiples
//! - `coord` - Coordinate system types

pub mod aesthetic;
pub mod coord;
pub mod facet;
pub mod layer;
pub mod main;
pub mod scale;
pub mod types;

// Re-export all types for convenience
pub use aesthetic::*;
pub use coord::*;
pub use facet::*;
pub use layer::*;
pub use main::*;
pub use scale::*;
pub use types::*;
