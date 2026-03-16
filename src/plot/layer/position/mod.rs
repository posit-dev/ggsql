//! Position adjustment trait and implementations
//!
//! This module provides a trait-based design for position adjustments in ggsql.
//! Each position type is implemented as its own struct, mirroring the geom pattern.
//!
//! # Architecture
//!
//! - `PositionType`: Enum for pattern matching and serialization
//! - `PositionTrait`: Trait defining position adjustment behavior
//! - `Position`: Wrapper struct holding a boxed trait object

mod dodge;
mod identity;
mod jitter;
mod stack;

use crate::plot::types::{DefaultParam, DefaultParamValue, ParameterValue};
use crate::plot::ScaleTypeKind;
use crate::{DataFrame, Plot, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;

/// Check if an aesthetic has a continuous scale type.
/// Returns None if no scale is defined (defer to data type).
/// This is the shared helper used by position adjustments.
pub fn is_continuous_scale(spec: &Plot, aesthetic: &str) -> Option<bool> {
    spec.scales
        .iter()
        .find(|s| s.aesthetic == aesthetic)
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind() == ScaleTypeKind::Continuous)
}

/// Result of computing dodge offsets for position adjustment.
pub struct DodgeOffsets {
    /// Offset for pos1 axis (None if not dodging pos1)
    pub pos1: Option<Vec<f64>>,
    /// Offset for pos2 axis (None if not dodging pos2)
    pub pos2: Option<Vec<f64>>,
    /// Adjusted element width after dodging
    pub adjusted_width: f64,
    /// Scale factor for existing offset columns (grid_size for 2D, n_groups for 1D)
    pub offset_scale: f64,
}

/// Compute dodge offsets for each row based on group indices.
///
/// This is the shared logic used by both dodge and jitter position adjustments.
/// For 2D (both axes discrete), arranges groups in a square grid.
/// For 1D (one axis discrete), arranges groups linearly.
pub fn compute_dodge_offsets(
    indices: &[usize],
    n_groups: usize,
    width: f64,
    dodge_pos1: bool,
    dodge_pos2: bool,
) -> DodgeOffsets {
    // For 2D, use grid layout; for 1D, use linear layout
    let divisor = if dodge_pos1 && dodge_pos2 {
        (n_groups as f64).sqrt().ceil() as usize
    } else {
        n_groups
    };
    let divisor_f64 = divisor as f64;
    let adjusted_width = width / divisor_f64;
    let center_offset = (divisor_f64 - 1.0) / 2.0;

    // Helper to compute offsets given a function to extract position from index
    let compute_offsets = |pos_fn: fn(usize, usize) -> usize| -> Vec<f64> {
        indices
            .iter()
            .map(|&idx| (pos_fn(idx, divisor) as f64 - center_offset) * adjusted_width)
            .collect()
    };

    let pos1 = if dodge_pos1 {
        let pos_fn: fn(usize, usize) -> usize = if dodge_pos2 {
            |idx, div| idx % div // 2D: column position
        } else {
            |idx, _| idx // 1D: direct index
        };
        Some(compute_offsets(pos_fn))
    } else {
        None
    };

    let pos2 = if dodge_pos2 {
        let pos_fn: fn(usize, usize) -> usize = if dodge_pos1 {
            |idx, div| idx / div // 2D: row position
        } else {
            |idx, _| idx // 1D: direct index
        };
        Some(compute_offsets(pos_fn))
    } else {
        None
    };

    DodgeOffsets {
        pos1,
        pos2,
        adjusted_width,
        offset_scale: divisor_f64,
    }
}

// Re-export position implementations
pub use dodge::{compute_group_indices, Dodge, GroupIndices};
pub use identity::Identity;
pub use jitter::Jitter;
pub use stack::Stack;

use super::Layer;

/// Enum of all position types for pattern matching and serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionType {
    Identity,
    Stack,
    Dodge,
    Jitter,
}

impl std::fmt::Display for PositionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PositionType::Identity => "identity",
            PositionType::Stack => "stack",
            PositionType::Dodge => "dodge",
            PositionType::Jitter => "jitter",
        };
        write!(f, "{}", s)
    }
}

/// Core trait for position adjustment behavior
///
/// Each position type implements this trait. Most methods have sensible defaults;
/// only `position_type()` and `apply_adjustment()` are typically required.
pub trait PositionTrait: std::fmt::Debug + std::fmt::Display + Send + Sync {
    /// Returns which position type this is (for pattern matching)
    fn position_type(&self) -> PositionType;

    /// Returns default parameter values for this position
    fn default_params(&self) -> &'static [DefaultParam] {
        &[]
    }

    /// Returns valid parameter names for SETTING validation
    fn valid_settings(&self) -> Vec<&'static str> {
        self.default_params().iter().map(|p| p.name).collect()
    }

    /// Whether this position creates a pos1offset column
    fn creates_pos1offset(&self) -> bool {
        false
    }

    /// Whether this position creates a pos2offset column
    fn creates_pos2offset(&self) -> bool {
        false
    }

    /// Apply the position adjustment to the DataFrame
    ///
    /// Returns the adjusted DataFrame and optionally an adjusted width
    /// (for position types like dodge that modify element width)
    fn apply_adjustment(
        &self,
        df: DataFrame,
        layer: &Layer,
        spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)>;
}

/// Wrapper struct for position trait objects
///
/// This provides a convenient interface for working with positions while hiding
/// the complexity of trait objects.
#[derive(Clone)]
pub struct Position(Arc<dyn PositionTrait>);

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.position_type() == other.position_type()
    }
}

impl Position {
    /// Create an Identity position (no adjustment)
    pub fn identity() -> Self {
        Self(Arc::new(Identity))
    }

    /// Create a Stack position
    pub fn stack() -> Self {
        Self(Arc::new(Stack))
    }

    /// Create a Dodge position
    pub fn dodge() -> Self {
        Self(Arc::new(Dodge))
    }

    /// Create a Jitter position
    pub fn jitter() -> Self {
        Self(Arc::new(Jitter))
    }

    /// Create a Position from a PositionType
    pub fn from_type(t: PositionType) -> Self {
        match t {
            PositionType::Identity => Self::identity(),
            PositionType::Stack => Self::stack(),
            PositionType::Dodge => Self::dodge(),
            PositionType::Jitter => Self::jitter(),
        }
    }

    /// Get the position type
    pub fn position_type(&self) -> PositionType {
        self.0.position_type()
    }

    /// Get default parameters
    pub fn default_params(&self) -> &'static [DefaultParam] {
        self.0.default_params()
    }

    /// Get valid settings for SETTING validation
    pub fn valid_settings(&self) -> Vec<&'static str> {
        self.0.valid_settings()
    }

    /// Check if this position creates a pos1offset column
    pub fn creates_pos1offset(&self) -> bool {
        self.0.creates_pos1offset()
    }

    /// Check if this position creates a pos2offset column
    pub fn creates_pos2offset(&self) -> bool {
        self.0.creates_pos2offset()
    }

    /// Apply the position adjustment
    pub fn apply_adjustment(
        &self,
        df: DataFrame,
        layer: &Layer,
        spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)> {
        self.0.apply_adjustment(df, layer, spec)
    }

    /// Apply default position parameter values to a layer
    ///
    /// For each parameter defined in default_params(), if the layer doesn't
    /// already have that parameter set, insert the default value.
    pub fn apply_defaults_to_layer(&self, layer: &mut Layer) {
        for param in self.default_params() {
            if !layer.parameters.contains_key(param.name) {
                let value = match &param.default {
                    DefaultParamValue::String(s) => ParameterValue::String(s.to_string()),
                    DefaultParamValue::Number(n) => ParameterValue::Number(*n),
                    DefaultParamValue::Boolean(b) => ParameterValue::Boolean(*b),
                    DefaultParamValue::Null => continue,
                };
                layer.parameters.insert(param.name.to_string(), value);
            }
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Position::{:?}", self.position_type())
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Position {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "stack" => Self::stack(),
            "dodge" => Self::dodge(),
            "jitter" => Self::jitter(),
            _ => Self::identity(),
        })
    }
}

impl Serialize for Position {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.position_type().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Position {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let position_type = PositionType::deserialize(deserializer)?;
        Ok(Position::from_type(position_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let identity = Position::identity();
        assert_eq!(identity.position_type(), PositionType::Identity);

        let stack = Position::stack();
        assert_eq!(stack.position_type(), PositionType::Stack);

        let dodge = Position::dodge();
        assert_eq!(dodge.position_type(), PositionType::Dodge);
    }

    #[test]
    fn test_position_display() {
        assert_eq!(format!("{}", Position::identity()), "identity");
        assert_eq!(format!("{}", Position::stack()), "stack");
        assert_eq!(format!("{}", Position::dodge()), "dodge");
    }

    #[test]
    fn test_position_from_str() {
        assert_eq!(
            "stack".parse::<Position>().unwrap().position_type(),
            PositionType::Stack
        );
        assert_eq!(
            "dodge".parse::<Position>().unwrap().position_type(),
            PositionType::Dodge
        );
        assert_eq!(
            "jitter".parse::<Position>().unwrap().position_type(),
            PositionType::Jitter
        );
        assert_eq!(
            "unknown".parse::<Position>().unwrap().position_type(),
            PositionType::Identity
        );
    }

    #[test]
    fn test_position_serialization() {
        let dodge = Position::dodge();
        let json = serde_json::to_string(&dodge).unwrap();
        assert_eq!(json, "\"dodge\"");

        let deserialized: Position = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.position_type(), PositionType::Dodge);
    }

    #[test]
    fn test_creates_pos1offset() {
        assert!(!Position::identity().creates_pos1offset());
        assert!(!Position::stack().creates_pos1offset());
        assert!(Position::dodge().creates_pos1offset());
        assert!(Position::jitter().creates_pos1offset());
    }

    #[test]
    fn test_creates_pos2offset() {
        assert!(!Position::identity().creates_pos2offset());
        assert!(!Position::stack().creates_pos2offset());
        assert!(Position::dodge().creates_pos2offset()); // Dodge now supports vertical/2D
        assert!(Position::jitter().creates_pos2offset());
    }

    #[test]
    fn test_is_continuous_scale() {
        use crate::plot::{Scale, ScaleType};

        // No scale defined - returns None
        let spec = crate::plot::Plot::new();
        assert!(is_continuous_scale(&spec, "pos1").is_none());

        // Continuous scale defined
        let mut spec = crate::plot::Plot::new();
        let mut scale = Scale::new("pos1");
        scale.scale_type = Some(ScaleType::continuous());
        spec.scales.push(scale);
        assert_eq!(is_continuous_scale(&spec, "pos1"), Some(true));

        // Discrete scale defined
        let mut spec = crate::plot::Plot::new();
        let mut scale = Scale::new("pos1");
        scale.scale_type = Some(ScaleType::discrete());
        spec.scales.push(scale);
        assert_eq!(is_continuous_scale(&spec, "pos1"), Some(false));
    }
}
