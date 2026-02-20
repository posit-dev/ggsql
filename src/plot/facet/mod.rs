//! Facet types for ggsql visualization specifications
//!
//! This module defines faceting configuration for small multiples.

mod resolve;
mod types;

pub use resolve::{resolve_properties, FacetDataContext};
pub use types::{Facet, FacetLayout};
