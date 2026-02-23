//! Coordinate system trait and implementations
//!
//! This module provides a trait-based design for coordinate system types in ggsql.
//! Each coord type is implemented as its own struct, allowing for cleaner separation
//! of concerns and easier extensibility.
//!
//! # Architecture
//!
//! - `CoordKind`: Enum for pattern matching and serialization
//! - `CoordTrait`: Trait defining coord type behavior
//! - `Coord`: Wrapper struct holding an Arc<dyn CoordTrait>
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::plot::projection::{Coord, CoordKind};
//!
//! let cartesian = Coord::cartesian();
//! assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);
//! assert_eq!(cartesian.name(), "cartesian");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::plot::ParameterValue;

// Coord type implementations
mod cartesian;
mod flip;
mod polar;

// Re-export coord type structs
pub use cartesian::Cartesian;
pub use flip::Flip;
pub use polar::Polar;

// =============================================================================
// Coord Kind Enum
// =============================================================================

/// Enum of all coordinate system types for pattern matching and serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoordKind {
    /// Standard x/y Cartesian coordinates (default)
    Cartesian,
    /// Flipped Cartesian (swaps x and y axes)
    Flip,
    /// Polar coordinates (for pie charts, rose plots)
    Polar,
}

impl CoordKind {
    /// Get the canonical name for this coord kind
    pub fn name(&self) -> &'static str {
        match self {
            CoordKind::Cartesian => "cartesian",
            CoordKind::Flip => "flip",
            CoordKind::Polar => "polar",
        }
    }
}

// =============================================================================
// Coord Trait
// =============================================================================

/// Trait defining coordinate system behavior.
///
/// Each coord type implements this trait. The trait is intentionally minimal
/// and backend-agnostic - no Vega-Lite or other writer-specific details.
pub trait CoordTrait: std::fmt::Debug + std::fmt::Display + Send + Sync {
    /// Returns which coord type this is (for pattern matching)
    fn coord_kind(&self) -> CoordKind;

    /// Canonical name for parsing and display
    fn name(&self) -> &'static str;

    /// Returns list of allowed property names for SETTING clause.
    /// Default: empty (no properties allowed).
    fn allowed_properties(&self) -> &'static [&'static str] {
        &[]
    }

    /// Returns default value for a property, if any.
    fn get_property_default(&self, _name: &str) -> Option<ParameterValue> {
        None
    }

    /// Resolve and validate properties.
    /// Default implementation validates against allowed_properties.
    fn resolve_properties(
        &self,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        let allowed = self.allowed_properties();

        // Check for unknown properties
        for key in properties.keys() {
            if !allowed.contains(&key.as_str()) {
                let valid_props = if allowed.is_empty() {
                    "none".to_string()
                } else {
                    allowed.join(", ")
                };
                return Err(format!(
                    "Property '{}' not valid for {} projection. Valid properties: {}",
                    key,
                    self.name(),
                    valid_props
                ));
            }
        }

        // Start with user properties, add defaults for missing ones
        let mut resolved = properties.clone();
        for &prop_name in allowed {
            if !resolved.contains_key(prop_name) {
                if let Some(default) = self.get_property_default(prop_name) {
                    resolved.insert(prop_name.to_string(), default);
                }
            }
        }

        Ok(resolved)
    }
}

// =============================================================================
// Coord Wrapper Struct
// =============================================================================

/// Arc-wrapped coordinate system type.
///
/// This provides a convenient interface for working with coord types while hiding
/// the complexity of trait objects.
#[derive(Clone)]
pub struct Coord(Arc<dyn CoordTrait>);

impl Coord {
    /// Create a Cartesian coord type
    pub fn cartesian() -> Self {
        Self(Arc::new(Cartesian))
    }

    /// Create a Flip coord type
    pub fn flip() -> Self {
        Self(Arc::new(Flip))
    }

    /// Create a Polar coord type
    pub fn polar() -> Self {
        Self(Arc::new(Polar))
    }

    /// Create a Coord from a CoordKind
    pub fn from_kind(kind: CoordKind) -> Self {
        match kind {
            CoordKind::Cartesian => Self::cartesian(),
            CoordKind::Flip => Self::flip(),
            CoordKind::Polar => Self::polar(),
        }
    }

    /// Get the coord type kind (for pattern matching)
    pub fn coord_kind(&self) -> CoordKind {
        self.0.coord_kind()
    }

    /// Get the canonical name
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Returns list of allowed property names for SETTING clause.
    pub fn allowed_properties(&self) -> &'static [&'static str] {
        self.0.allowed_properties()
    }

    /// Returns default value for a property, if any.
    pub fn get_property_default(&self, name: &str) -> Option<ParameterValue> {
        self.0.get_property_default(name)
    }

    /// Resolve and validate properties.
    pub fn resolve_properties(
        &self,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        self.0.resolve_properties(properties)
    }
}

impl std::fmt::Debug for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Coord({})", self.name())
    }
}

impl std::fmt::Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl PartialEq for Coord {
    fn eq(&self, other: &Self) -> bool {
        self.coord_kind() == other.coord_kind()
    }
}

impl Eq for Coord {}

impl std::hash::Hash for Coord {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coord_kind().hash(state);
    }
}

// Implement Serialize by delegating to CoordKind
impl Serialize for Coord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.coord_kind().serialize(serializer)
    }
}

// Implement Deserialize by delegating to CoordKind
impl<'de> Deserialize<'de> for Coord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let kind = CoordKind::deserialize(deserializer)?;
        Ok(Coord::from_kind(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_kind_name() {
        assert_eq!(CoordKind::Cartesian.name(), "cartesian");
        assert_eq!(CoordKind::Flip.name(), "flip");
        assert_eq!(CoordKind::Polar.name(), "polar");
    }

    #[test]
    fn test_coord_factory_methods() {
        let cartesian = Coord::cartesian();
        assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);
        assert_eq!(cartesian.name(), "cartesian");

        let flip = Coord::flip();
        assert_eq!(flip.coord_kind(), CoordKind::Flip);
        assert_eq!(flip.name(), "flip");

        let polar = Coord::polar();
        assert_eq!(polar.coord_kind(), CoordKind::Polar);
        assert_eq!(polar.name(), "polar");
    }

    #[test]
    fn test_coord_from_kind() {
        assert_eq!(
            Coord::from_kind(CoordKind::Cartesian).coord_kind(),
            CoordKind::Cartesian
        );
        assert_eq!(
            Coord::from_kind(CoordKind::Flip).coord_kind(),
            CoordKind::Flip
        );
        assert_eq!(
            Coord::from_kind(CoordKind::Polar).coord_kind(),
            CoordKind::Polar
        );
    }

    #[test]
    fn test_coord_equality() {
        assert_eq!(Coord::cartesian(), Coord::cartesian());
        assert_eq!(Coord::flip(), Coord::flip());
        assert_eq!(Coord::polar(), Coord::polar());
        assert_ne!(Coord::cartesian(), Coord::flip());
        assert_ne!(Coord::cartesian(), Coord::polar());
        assert_ne!(Coord::flip(), Coord::polar());
    }

    #[test]
    fn test_coord_serialization() {
        let cartesian = Coord::cartesian();
        let json = serde_json::to_string(&cartesian).unwrap();
        assert_eq!(json, "\"cartesian\"");

        let flip = Coord::flip();
        let json = serde_json::to_string(&flip).unwrap();
        assert_eq!(json, "\"flip\"");

        let polar = Coord::polar();
        let json = serde_json::to_string(&polar).unwrap();
        assert_eq!(json, "\"polar\"");
    }

    #[test]
    fn test_coord_deserialization() {
        let cartesian: Coord = serde_json::from_str("\"cartesian\"").unwrap();
        assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);

        let flip: Coord = serde_json::from_str("\"flip\"").unwrap();
        assert_eq!(flip.coord_kind(), CoordKind::Flip);

        let polar: Coord = serde_json::from_str("\"polar\"").unwrap();
        assert_eq!(polar.coord_kind(), CoordKind::Polar);
    }
}
