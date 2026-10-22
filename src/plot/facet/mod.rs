//! Facet types for ggsql visualization specifications
//!
//! This module defines faceting configuration for small multiples.

pub mod panels;
mod resolve;
mod types;

pub use panels::{
    element_to_key, is_binned, level_keys, ordered_levels, panel_keys, rows_in_panel, FacetLevel,
    LevelKey, PanelKey,
};
pub use resolve::{resolve_properties, FacetDataContext};
pub use types::{Facet, FacetLayout};
