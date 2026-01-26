//! Transform trait and implementations
//!
//! This module provides a trait-based design for scale transforms in ggsql.
//! Each transform type is implemented as its own struct, allowing for cleaner
//! separation of concerns and easier extensibility.
//!
//! # Architecture
//!
//! - `TransformKind`: Enum for pattern matching and serialization
//! - `TransformTrait`: Trait defining transform behavior
//! - `Transform`: Wrapper struct holding an Arc<dyn TransformTrait>
//!
//! # Supported Transforms
//!
//! | Transform    | Domain       | Description                    |
//! |--------------|--------------|--------------------------------|
//! | `identity`   | (-∞, +∞)     | No transformation (linear)     |
//! | `log10`      | (0, +∞)      | Base-10 logarithm              |
//! | `log2`       | (0, +∞)      | Base-2 logarithm               |
//! | `log`        | (0, +∞)      | Natural logarithm (base e)     |
//! | `sqrt`       | [0, +∞)      | Square root                    |
//! | `asinh`      | (-∞, +∞)     | Inverse hyperbolic sine        |
//! | `pseudo_log` | (-∞, +∞)     | Symmetric log (ggplot2 formula)|
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::plot::scale::transform::{Transform, TransformKind};
//!
//! let log10 = Transform::log10();
//! assert_eq!(log10.transform_kind(), TransformKind::Log10);
//! assert!(log10.is_value_in_domain(1.0));
//! assert!(!log10.is_value_in_domain(-1.0));
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod asinh;
mod identity;
mod log;
mod pseudo_log;
mod sqrt;

pub use self::asinh::Asinh;
pub use self::identity::Identity;
pub use self::log::Log;
pub use self::pseudo_log::PseudoLog;
pub use self::sqrt::Sqrt;

/// Enum of all transform types for pattern matching and serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformKind {
    /// No transformation (linear)
    Identity,
    /// Base-10 logarithm
    Log10,
    /// Base-2 logarithm
    Log2,
    /// Natural logarithm (base e)
    Log,
    /// Square root
    Sqrt,
    /// Inverse hyperbolic sine
    Asinh,
    /// Symmetric log (ggplot2 formula with configurable base)
    PseudoLog,
}

impl TransformKind {
    /// Returns the canonical name for this transform kind
    pub fn name(&self) -> &'static str {
        match self {
            TransformKind::Identity => "identity",
            TransformKind::Log10 => "log10",
            TransformKind::Log2 => "log2",
            TransformKind::Log => "log",
            TransformKind::Sqrt => "sqrt",
            TransformKind::Asinh => "asinh",
            TransformKind::PseudoLog => "pseudo_log",
        }
    }
}

impl std::fmt::Display for TransformKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Core trait for transform behavior
///
/// Each transform type implements this trait. The trait is intentionally
/// backend-agnostic - no Vega-Lite or other writer-specific details.
pub trait TransformTrait: std::fmt::Debug + std::fmt::Display + Send + Sync {
    /// Returns which transform type this is (for pattern matching)
    fn transform_kind(&self) -> TransformKind;

    /// Canonical name for parsing and display
    fn name(&self) -> &'static str;

    /// Returns valid input domain as (min, max)
    ///
    /// - `identity`: (-∞, +∞)
    /// - `log10`, `log2`, `log`: (0, +∞) - excludes 0 and negative
    /// - `sqrt`: [0, +∞) - includes 0
    /// - `asinh`, `pseudo_log`: (-∞, +∞)
    fn allowed_domain(&self) -> (f64, f64);

    /// Check if value is in the transform's domain
    ///
    /// Returns true if the value can be transformed without producing
    /// NaN or infinity.
    fn is_value_in_domain(&self, value: f64) -> bool;

    /// Calculate breaks for this transform
    ///
    /// Calculates appropriate break positions in data space for the
    /// given range. The algorithm varies by transform type:
    ///
    /// - `identity`: Uses Wilkinson's algorithm for pretty breaks
    /// - `log10`, `log2`, `log`: Uses powers of base with 1-2-5 pattern
    /// - `sqrt`: Calculates breaks in sqrt-space, then squares them back
    /// - `asinh`, `pseudo_log`: Uses symlog algorithm for symmetric ranges
    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64>;

    /// Forward transformation: x -> transform(x)
    ///
    /// Maps a value from data space to transformed space.
    fn transform(&self, value: f64) -> f64;

    /// Inverse transformation: transform(x) -> x
    ///
    /// Maps a value from transformed space back to data space.
    fn inverse(&self, value: f64) -> f64;
}

/// Wrapper struct for transform trait objects
///
/// This provides a convenient interface for working with transforms while
/// hiding the complexity of trait objects.
#[derive(Clone)]
pub struct Transform(Arc<dyn TransformTrait>);

impl Transform {
    /// Create an Identity transform (no transformation)
    pub fn identity() -> Self {
        Self(Arc::new(Identity))
    }

    /// Create a Log10 transform (base-10 logarithm)
    pub fn log10() -> Self {
        Self(Arc::new(Log::base10()))
    }

    /// Create a Log2 transform (base-2 logarithm)
    pub fn log2() -> Self {
        Self(Arc::new(Log::base2()))
    }

    /// Create a Log transform (natural logarithm)
    pub fn log() -> Self {
        Self(Arc::new(Log::natural()))
    }

    /// Create a Sqrt transform (square root)
    pub fn sqrt() -> Self {
        Self(Arc::new(Sqrt))
    }

    /// Create an Asinh transform (inverse hyperbolic sine)
    pub fn asinh() -> Self {
        Self(Arc::new(Asinh))
    }

    /// Create a PseudoLog transform (symmetric log, base 10)
    pub fn pseudo_log() -> Self {
        Self(Arc::new(PseudoLog::base10()))
    }

    /// Create a Transform from a string name
    ///
    /// Returns None if the name is not recognized.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use ggsql::plot::scale::transform::Transform;
    ///
    /// let t = Transform::from_name("log10").unwrap();
    /// assert_eq!(t.name(), "log10");
    ///
    /// assert!(Transform::from_name("unknown").is_none());
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "identity" => Some(Self::identity()),
            "log10" => Some(Self::log10()),
            "log2" => Some(Self::log2()),
            "log" => Some(Self::log()),
            "sqrt" => Some(Self::sqrt()),
            "asinh" => Some(Self::asinh()),
            "pseudo_log" => Some(Self::pseudo_log()),
            _ => None,
        }
    }

    /// Create a Transform from a TransformKind
    pub fn from_kind(kind: TransformKind) -> Self {
        match kind {
            TransformKind::Identity => Self::identity(),
            TransformKind::Log10 => Self::log10(),
            TransformKind::Log2 => Self::log2(),
            TransformKind::Log => Self::log(),
            TransformKind::Sqrt => Self::sqrt(),
            TransformKind::Asinh => Self::asinh(),
            TransformKind::PseudoLog => Self::pseudo_log(),
        }
    }

    /// Get the transform kind (for pattern matching)
    pub fn transform_kind(&self) -> TransformKind {
        self.0.transform_kind()
    }

    /// Get the canonical name
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Get the valid input domain as (min, max)
    pub fn allowed_domain(&self) -> (f64, f64) {
        self.0.allowed_domain()
    }

    /// Check if value is in the transform's domain
    pub fn is_value_in_domain(&self, value: f64) -> bool {
        self.0.is_value_in_domain(value)
    }

    /// Calculate breaks for this transform
    pub fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        self.0.calculate_breaks(min, max, n, pretty)
    }

    /// Forward transformation: x -> transform(x)
    pub fn transform(&self, value: f64) -> f64 {
        self.0.transform(value)
    }

    /// Inverse transformation: transform(x) -> x
    pub fn inverse(&self, value: f64) -> f64 {
        self.0.inverse(value)
    }

    /// Returns true if this is the identity transform
    pub fn is_identity(&self) -> bool {
        self.transform_kind() == TransformKind::Identity
    }
}

impl std::fmt::Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Transform::{:?}", self.transform_kind())
    }
}

impl std::fmt::Display for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for Transform {
    fn eq(&self, other: &Self) -> bool {
        self.transform_kind() == other.transform_kind()
    }
}

impl Eq for Transform {}

impl Serialize for Transform {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.transform_kind().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Transform {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let kind = TransformKind::deserialize(deserializer)?;
        Ok(Transform::from_kind(kind))
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

/// List of all valid transform names
pub const ALL_TRANSFORM_NAMES: &[&str] = &[
    "identity",
    "log10",
    "log2",
    "log",
    "sqrt",
    "asinh",
    "pseudo_log",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_creation() {
        let identity = Transform::identity();
        assert_eq!(identity.transform_kind(), TransformKind::Identity);
        assert_eq!(identity.name(), "identity");

        let log10 = Transform::log10();
        assert_eq!(log10.transform_kind(), TransformKind::Log10);
        assert_eq!(log10.name(), "log10");
    }

    #[test]
    fn test_transform_from_name() {
        assert!(Transform::from_name("identity").is_some());
        assert!(Transform::from_name("log10").is_some());
        assert!(Transform::from_name("log2").is_some());
        assert!(Transform::from_name("log").is_some());
        assert!(Transform::from_name("sqrt").is_some());
        assert!(Transform::from_name("asinh").is_some());
        assert!(Transform::from_name("pseudo_log").is_some());
        assert!(Transform::from_name("unknown").is_none());
    }

    #[test]
    fn test_transform_from_kind() {
        let t = Transform::from_kind(TransformKind::Log10);
        assert_eq!(t.transform_kind(), TransformKind::Log10);
    }

    #[test]
    fn test_transform_equality() {
        let log10_a = Transform::log10();
        let log10_b = Transform::log10();
        let log2 = Transform::log2();

        assert_eq!(log10_a, log10_b);
        assert_ne!(log10_a, log2);
    }

    #[test]
    fn test_transform_display() {
        assert_eq!(format!("{}", Transform::identity()), "identity");
        assert_eq!(format!("{}", Transform::log10()), "log10");
        assert_eq!(format!("{}", Transform::sqrt()), "sqrt");
    }

    #[test]
    fn test_transform_serialization() {
        let log10 = Transform::log10();
        let json = serde_json::to_string(&log10).unwrap();
        assert_eq!(json, "\"log10\"");

        let deserialized: Transform = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.transform_kind(), TransformKind::Log10);
    }

    #[test]
    fn test_transform_is_identity() {
        assert!(Transform::identity().is_identity());
        assert!(!Transform::log10().is_identity());
        assert!(!Transform::sqrt().is_identity());
    }

    #[test]
    fn test_transform_default() {
        let default = Transform::default();
        assert!(default.is_identity());
    }

    #[test]
    fn test_transform_kind_display() {
        assert_eq!(format!("{}", TransformKind::Identity), "identity");
        assert_eq!(format!("{}", TransformKind::Log10), "log10");
        assert_eq!(format!("{}", TransformKind::PseudoLog), "pseudo_log");
    }
}
