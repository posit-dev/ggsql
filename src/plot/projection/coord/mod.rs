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

use crate::plot::types::{validate_parameter, ParamDefinition};
use crate::plot::ParameterValue;

// Coord type implementations
mod cartesian;
mod polar;

// Re-export coord type structs
pub use cartesian::Cartesian;
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
    /// Polar coordinates (for pie charts, rose plots)
    Polar,
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

    /// Primary positional aesthetic names for this coord.
    ///
    /// Returns the user-facing positional aesthetic names.
    /// e.g., ["x", "y"] for cartesian, ["radius", "theta"] for polar.
    ///
    /// These names are transformed to internal names (pos1, pos2, etc.)
    /// early in the pipeline and transformed back for output.
    fn positional_aesthetic_names(&self) -> &'static [&'static str];

    /// Returns list of allowed properties with their default values.
    /// Default: empty (no properties allowed).
    fn default_properties(&self) -> &'static [ParamDefinition] {
        &[]
    }

    /// Resolve and validate properties.
    /// Default implementation validates against default_properties.
    fn resolve_properties(
        &self,
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<HashMap<String, ParameterValue>, String> {
        let defaults = self.default_properties();

        // Validate values against constraints
        for (key, value) in properties.iter() {
            if let Some(param) = defaults.iter().find(|p| p.name == key) {
                validate_parameter(key, value, &param.constraint)?;
            } else {
                let allowed: Vec<&str> = defaults.iter().map(|p| p.name).collect();
                return Err(if allowed.is_empty() {
                    format!(
                        "{} projection does not accept any properties, but got '{}'",
                        self.name(),
                        key
                    )
                } else {
                    format!(
                        "{} projection property should be {}, not '{}'",
                        self.name(),
                        crate::or_list_quoted(&allowed, '\''),
                        key
                    )
                });
            }
        }

        // Start with user properties, add defaults for missing ones
        let mut resolved = properties.clone();
        for param in defaults {
            if !resolved.contains_key(param.name) {
                if let Some(default) = param.to_parameter_value() {
                    resolved.insert(param.name.to_string(), default);
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

    /// Create a Polar coord type
    pub fn polar() -> Self {
        Self(Arc::new(Polar))
    }

    /// Create a Coord from a CoordKind
    pub fn from_kind(kind: CoordKind) -> Self {
        match kind {
            CoordKind::Cartesian => Self::cartesian(),
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

    /// Primary positional aesthetic names for this coord.
    /// e.g., ["x", "y"] for cartesian, ["radius", "theta"] for polar.
    pub fn positional_aesthetic_names(&self) -> &'static [&'static str] {
        self.0.positional_aesthetic_names()
    }

    /// Returns list of allowed properties with their default values.
    pub fn default_properties(&self) -> &'static [ParamDefinition] {
        self.0.default_properties()
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
    fn test_coord_factory_methods() {
        let cartesian = Coord::cartesian();
        assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);
        assert_eq!(cartesian.name(), "cartesian");

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
            Coord::from_kind(CoordKind::Polar).coord_kind(),
            CoordKind::Polar
        );
    }

    #[test]
    fn test_coord_equality() {
        assert_eq!(Coord::cartesian(), Coord::cartesian());
        assert_eq!(Coord::polar(), Coord::polar());
        assert_ne!(Coord::cartesian(), Coord::polar());
    }

    #[test]
    fn test_coord_serialization() {
        let cartesian = Coord::cartesian();
        let json = serde_json::to_string(&cartesian).unwrap();
        assert_eq!(json, "\"cartesian\"");

        let polar = Coord::polar();
        let json = serde_json::to_string(&polar).unwrap();
        assert_eq!(json, "\"polar\"");
    }

    #[test]
    fn test_coord_deserialization() {
        let cartesian: Coord = serde_json::from_str("\"cartesian\"").unwrap();
        assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);

        let polar: Coord = serde_json::from_str("\"polar\"").unwrap();
        assert_eq!(polar.coord_kind(), CoordKind::Polar);
    }

    #[test]
    fn test_positional_aesthetic_names() {
        let cartesian = Coord::cartesian();
        assert_eq!(cartesian.positional_aesthetic_names(), &["x", "y"]);

        let polar = Coord::polar();
        assert_eq!(polar.positional_aesthetic_names(), &["radius", "theta"]);
    }
}
