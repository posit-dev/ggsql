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

use crate::plot::ArrayElement;

mod asinh;
mod bool;
mod date;
mod datetime;
mod exp;
mod identity;
mod integer;
mod log;
mod pseudo_log;
mod sqrt;
mod square;
mod string;
mod time;

pub use self::asinh::Asinh;
pub use self::bool::Bool;
pub use self::date::Date;
pub use self::datetime::DateTime;
pub use self::exp::Exp;
pub use self::identity::Identity;
pub use self::integer::Integer;
pub use self::log::Log;
pub use self::pseudo_log::PseudoLog;
pub use self::sqrt::Sqrt;
pub use self::square::Square;
pub use self::string::String as StringTransform;
pub use self::time::Time;

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
    /// Square (x²) - inverse of sqrt
    Square,
    /// Base-10 exponential (10^x) - inverse of log10
    Exp10,
    /// Base-2 exponential (2^x) - inverse of log2
    Exp2,
    /// Natural exponential (e^x) - inverse of ln
    Exp,
    /// Inverse hyperbolic sine
    Asinh,
    /// Symmetric log
    PseudoLog,
    /// Date transform (days since epoch)
    Date,
    /// DateTime transform (microseconds since epoch)
    DateTime,
    /// Time transform (nanoseconds since midnight)
    Time,
    /// String transform (for discrete scales)
    String,
    /// Boolean transform (for discrete scales)
    Bool,
    /// Integer transform (linear with integer casting)
    Integer,
}

impl TransformKind {
    /// Returns the canonical name for this transform kind
    pub fn name(&self) -> &'static str {
        match self {
            TransformKind::Identity => "identity",
            TransformKind::Log10 => "log",
            TransformKind::Log2 => "log2",
            TransformKind::Log => "ln",
            TransformKind::Sqrt => "sqrt",
            TransformKind::Square => "square",
            TransformKind::Exp10 => "exp10",
            TransformKind::Exp2 => "exp2",
            TransformKind::Exp => "exp",
            TransformKind::Asinh => "asinh",
            TransformKind::PseudoLog => "pseudo_log",
            TransformKind::Date => "date",
            TransformKind::DateTime => "datetime",
            TransformKind::Time => "time",
            TransformKind::String => "string",
            TransformKind::Bool => "bool",
            TransformKind::Integer => "integer",
        }
    }

    /// Returns true if this is a temporal transform
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            TransformKind::Date | TransformKind::DateTime | TransformKind::Time
        )
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

    /// Calculate minor breaks between major breaks
    ///
    /// Places intermediate tick marks between the major breaks. The algorithm
    /// varies by transform type to produce evenly-spaced minor breaks in
    /// the transformed space.
    ///
    /// # Arguments
    /// - `major_breaks`: The major break positions
    /// - `n`: Number of minor breaks per major interval
    /// - `range`: Optional (min, max) scale input range to extend minor breaks beyond major breaks
    ///
    /// # Returns
    /// Minor break positions (excluding major breaks)
    ///
    /// # Behavior
    /// - Places n minor breaks between each consecutive pair of major breaks
    /// - If range is provided and extends beyond major breaks, extrapolates minor breaks into those regions
    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64>;

    /// Returns the default number of minor breaks per major interval for this transform
    ///
    /// - `identity`, `sqrt`: 1 (one midpoint per interval)
    /// - `log`, `asinh`, `pseudo_log`: 8 (similar density to traditional 2-9 pattern)
    fn default_minor_break_count(&self) -> usize {
        1 // Default for identity/sqrt
    }

    /// Forward transformation: x -> transform(x)
    ///
    /// Maps a value from data space to transformed space.
    fn transform(&self, value: f64) -> f64;

    /// Inverse transformation: transform(x) -> x
    ///
    /// Maps a value from transformed space back to data space.
    fn inverse(&self, value: f64) -> f64;

    /// Wrap a numeric value in the appropriate ArrayElement type.
    ///
    /// Temporal transforms override to return Date/DateTime/Time variants.
    /// Default returns ArrayElement::Number.
    fn wrap_numeric(&self, value: f64) -> ArrayElement {
        ArrayElement::Number(value)
    }

    /// Parse a value into the appropriate ArrayElement type for this transform.
    ///
    /// Temporal transforms parse ISO date/time strings into Date/DateTime/Time variants.
    /// Default passes through the value unchanged, wrapping numbers via wrap_numeric.
    fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        match elem {
            ArrayElement::Number(n) => self.wrap_numeric(*n),
            other => other.clone(),
        }
    }
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
    pub fn log() -> Self {
        Self(Arc::new(Log::base10()))
    }

    /// Create a Log2 transform (base-2 logarithm)
    pub fn log2() -> Self {
        Self(Arc::new(Log::base2()))
    }

    /// Create a Log transform (natural logarithm)
    pub fn ln() -> Self {
        Self(Arc::new(Log::natural()))
    }

    /// Create a Sqrt transform (square root)
    pub fn sqrt() -> Self {
        Self(Arc::new(Sqrt))
    }

    /// Create a Square transform (x²) - inverse of sqrt
    pub fn square() -> Self {
        Self(Arc::new(Square))
    }

    /// Create an Exp10 transform (10^x) - inverse of log10
    pub fn exp10() -> Self {
        Self(Arc::new(Exp::base10()))
    }

    /// Create an Exp2 transform (2^x) - inverse of log2
    pub fn exp2() -> Self {
        Self(Arc::new(Exp::base2()))
    }

    /// Create an Exp transform (e^x) - inverse of ln
    pub fn exp() -> Self {
        Self(Arc::new(Exp::natural()))
    }

    /// Create an Asinh transform (inverse hyperbolic sine)
    pub fn asinh() -> Self {
        Self(Arc::new(Asinh))
    }

    /// Create a PseudoLog transform (symmetric log, base 10)
    pub fn pseudo_log() -> Self {
        Self(Arc::new(PseudoLog::base10()))
    }

    /// Create a PseudoLog transform with base 2
    pub fn pseudo_log2() -> Self {
        Self(Arc::new(PseudoLog::base2()))
    }

    /// Create a PseudoLog transform with natural base (base e)
    pub fn pseudo_ln() -> Self {
        Self(Arc::new(PseudoLog::natural()))
    }

    /// Create a Date transform (for date data - days since epoch)
    pub fn date() -> Self {
        Self(Arc::new(Date))
    }

    /// Create a DateTime transform (for datetime data - microseconds since epoch)
    pub fn datetime() -> Self {
        Self(Arc::new(DateTime))
    }

    /// Create a Time transform (for time data - nanoseconds since midnight)
    pub fn time() -> Self {
        Self(Arc::new(Time))
    }

    /// Create a String transform (for discrete scales - casts to string)
    pub fn string() -> Self {
        Self(Arc::new(StringTransform))
    }

    /// Create a Bool transform (for discrete scales - casts to boolean)
    pub fn bool() -> Self {
        Self(Arc::new(Bool))
    }

    /// Create an Integer transform (linear with integer casting)
    pub fn integer() -> Self {
        Self(Arc::new(Integer))
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
            "identity" | "linear" => Some(Self::identity()),
            "log" | "log10" => Some(Self::log()),
            "log2" => Some(Self::log2()),
            "ln" => Some(Self::ln()),
            "sqrt" => Some(Self::sqrt()),
            "square" | "pow2" => Some(Self::square()),
            "exp10" => Some(Self::exp10()),
            "exp2" => Some(Self::exp2()),
            "exp" => Some(Self::exp()),
            "asinh" => Some(Self::asinh()),
            "pseudo_log" | "pseudo_log10" => Some(Self::pseudo_log()),
            "pseudo_log2" => Some(Self::pseudo_log2()),
            "pseudo_ln" => Some(Self::pseudo_ln()),
            "date" => Some(Self::date()),
            "datetime" => Some(Self::datetime()),
            "time" => Some(Self::time()),
            "string" | "str" | "varchar" => Some(Self::string()),
            "bool" | "boolean" => Some(Self::bool()),
            "integer" | "int" | "bigint" => Some(Self::integer()),
            _ => None,
        }
    }

    /// Create a Transform from a TransformKind
    pub fn from_kind(kind: TransformKind) -> Self {
        match kind {
            TransformKind::Identity => Self::identity(),
            TransformKind::Log10 => Self::log(),
            TransformKind::Log2 => Self::log2(),
            TransformKind::Log => Self::ln(),
            TransformKind::Sqrt => Self::sqrt(),
            TransformKind::Square => Self::square(),
            TransformKind::Exp10 => Self::exp10(),
            TransformKind::Exp2 => Self::exp2(),
            TransformKind::Exp => Self::exp(),
            TransformKind::Asinh => Self::asinh(),
            TransformKind::PseudoLog => Self::pseudo_log(),
            TransformKind::Date => Self::date(),
            TransformKind::DateTime => Self::datetime(),
            TransformKind::Time => Self::time(),
            TransformKind::String => Self::string(),
            TransformKind::Bool => Self::bool(),
            TransformKind::Integer => Self::integer(),
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

    /// Calculate minor breaks between major breaks
    pub fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        self.0.calculate_minor_breaks(major_breaks, n, range)
    }

    /// Returns the default number of minor breaks per major interval for this transform
    pub fn default_minor_break_count(&self) -> usize {
        self.0.default_minor_break_count()
    }

    /// Forward transformation: x -> transform(x)
    pub fn transform(&self, value: f64) -> f64 {
        self.0.transform(value)
    }

    /// Inverse transformation: transform(x) -> x
    pub fn inverse(&self, value: f64) -> f64 {
        self.0.inverse(value)
    }

    /// Wrap a numeric value in the appropriate ArrayElement type
    pub fn wrap_numeric(&self, value: f64) -> ArrayElement {
        self.0.wrap_numeric(value)
    }

    /// Parse a value into the appropriate ArrayElement type for this transform
    pub fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        self.0.parse_value(elem)
    }

    /// Returns true if this is the identity transform
    pub fn is_identity(&self) -> bool {
        self.transform_kind() == TransformKind::Identity
    }

    /// Returns true if this is a temporal transform (Date, DateTime, or Time)
    pub fn is_temporal(&self) -> bool {
        self.transform_kind().is_temporal()
    }

    /// Return the target ArrayElementType for this transform.
    ///
    /// Used by scales to determine the coercion target based on the transform.
    /// Temporal transforms target their respective temporal types;
    /// String/Bool transforms target their respective discrete types;
    /// all other transforms target Number.
    pub fn target_type(&self) -> crate::plot::ArrayElementType {
        use crate::plot::ArrayElementType;
        match self.transform_kind() {
            TransformKind::Date => ArrayElementType::Date,
            TransformKind::DateTime => ArrayElementType::DateTime,
            TransformKind::Time => ArrayElementType::Time,
            TransformKind::String => ArrayElementType::String,
            TransformKind::Bool => ArrayElementType::Boolean,
            // All other transforms (Identity, Log, Sqrt, etc.) work on numbers
            _ => ArrayElementType::Number,
        }
    }

    /// Format a numeric value as an ISO string for SQL literals.
    ///
    /// Temporal transforms convert their internal numeric representation
    /// (days/microseconds/nanoseconds) back to ISO date/time strings.
    /// Returns None for non-temporal transforms.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let date_transform = Transform::date();
    /// // 19724 days since epoch = 2024-01-01
    /// assert_eq!(date_transform.format_as_iso(19724.0), Some("2024-01-01".to_string()));
    ///
    /// let identity = Transform::identity();
    /// assert_eq!(identity.format_as_iso(100.0), None);
    /// ```
    pub fn format_as_iso(&self, value: f64) -> Option<String> {
        use chrono::{DateTime as ChronoDateTime, NaiveDate, NaiveTime};

        /// Days from CE to Unix epoch (1970-01-01)
        const UNIX_EPOCH_CE_DAYS: i32 = 719163;

        match self.transform_kind() {
            TransformKind::Date => {
                let days = value as i32;
                NaiveDate::from_num_days_from_ce_opt(days + UNIX_EPOCH_CE_DAYS)
                    .map(|d| d.format("%Y-%m-%d").to_string())
            }
            TransformKind::DateTime => {
                let micros = value as i64;
                ChronoDateTime::from_timestamp_micros(micros)
                    .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
            }
            TransformKind::Time => {
                let nanos = value as i64;
                let secs = (nanos / 1_000_000_000) as u32;
                let nano_part = (nanos % 1_000_000_000) as u32;
                NaiveTime::from_num_seconds_from_midnight_opt(secs, nano_part)
                    .map(|t| t.format("%H:%M:%S").to_string())
            }
            _ => None,
        }
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
    "linear", // alias for identity
    "log",
    "log10", // alias for log
    "log2",
    "ln",
    "sqrt",
    "square",
    "pow2", // alias for square
    "exp10",
    "exp2",
    "exp",
    "asinh",
    "pseudo_log",
    "pseudo_log10", // alias for pseudo_log
    "pseudo_log2",
    "pseudo_ln",
    "date",
    "datetime",
    "time",
    "string",
    "str",     // alias for string
    "varchar", // alias for string
    "bool",
    "boolean", // alias for bool
    "integer",
    "int",    // alias for integer
    "bigint", // alias for integer
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_creation() {
        let identity = Transform::identity();
        assert_eq!(identity.transform_kind(), TransformKind::Identity);
        assert_eq!(identity.name(), "identity");

        let log = Transform::log();
        assert_eq!(log.transform_kind(), TransformKind::Log10);
        assert_eq!(log.name(), "log");

        let ln = Transform::ln();
        assert_eq!(ln.transform_kind(), TransformKind::Log);
        assert_eq!(ln.name(), "ln");
    }

    #[test]
    fn test_transform_from_name() {
        assert!(Transform::from_name("identity").is_some());
        assert!(Transform::from_name("log").is_some());
        assert!(Transform::from_name("log10").is_some()); // alias for log
        assert!(Transform::from_name("log2").is_some());
        assert!(Transform::from_name("ln").is_some());
        assert!(Transform::from_name("sqrt").is_some());
        assert!(Transform::from_name("asinh").is_some());
        assert!(Transform::from_name("pseudo_log").is_some());
        assert!(Transform::from_name("pseudo_log10").is_some()); // alias for pseudo_log
        assert!(Transform::from_name("pseudo_log2").is_some());
        assert!(Transform::from_name("pseudo_ln").is_some());
        assert!(Transform::from_name("unknown").is_none());

        // Verify log variants return correct names
        assert_eq!(Transform::from_name("log").unwrap().name(), "log");
        assert_eq!(Transform::from_name("log10").unwrap().name(), "log");
        assert_eq!(Transform::from_name("log2").unwrap().name(), "log2");
        assert_eq!(Transform::from_name("ln").unwrap().name(), "ln");

        // Verify pseudo_log variants return correct names
        assert_eq!(
            Transform::from_name("pseudo_log").unwrap().name(),
            "pseudo_log"
        );
        assert_eq!(
            Transform::from_name("pseudo_log10").unwrap().name(),
            "pseudo_log"
        );
        assert_eq!(
            Transform::from_name("pseudo_log2").unwrap().name(),
            "pseudo_log2"
        );
        assert_eq!(
            Transform::from_name("pseudo_ln").unwrap().name(),
            "pseudo_ln"
        );
    }

    #[test]
    fn test_transform_from_kind() {
        let t = Transform::from_kind(TransformKind::Log10);
        assert_eq!(t.transform_kind(), TransformKind::Log10);
    }

    #[test]
    fn test_transform_equality() {
        let log_a = Transform::log();
        let log_b = Transform::log();
        let log2 = Transform::log2();

        assert_eq!(log_a, log_b);
        assert_ne!(log_a, log2);
    }

    #[test]
    fn test_transform_display() {
        assert_eq!(format!("{}", Transform::identity()), "identity");
        assert_eq!(format!("{}", Transform::log()), "log");
        assert_eq!(format!("{}", Transform::ln()), "ln");
        assert_eq!(format!("{}", Transform::sqrt()), "sqrt");
    }

    #[test]
    fn test_transform_serialization() {
        let log = Transform::log();
        let json = serde_json::to_string(&log).unwrap();
        assert_eq!(json, "\"log10\""); // Serializes by TransformKind enum variant name

        let deserialized: Transform = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.transform_kind(), TransformKind::Log10);
    }

    #[test]
    fn test_transform_is_identity() {
        assert!(Transform::identity().is_identity());
        assert!(!Transform::log().is_identity());
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
        assert_eq!(format!("{}", TransformKind::Log10), "log");
        assert_eq!(format!("{}", TransformKind::Log), "ln");
        assert_eq!(format!("{}", TransformKind::PseudoLog), "pseudo_log");
    }

    #[test]
    fn test_transform_target_type() {
        use crate::plot::ArrayElementType;

        // Temporal transforms target their respective types
        assert_eq!(Transform::date().target_type(), ArrayElementType::Date);
        assert_eq!(
            Transform::datetime().target_type(),
            ArrayElementType::DateTime
        );
        assert_eq!(Transform::time().target_type(), ArrayElementType::Time);

        // All other transforms target Number
        assert_eq!(
            Transform::identity().target_type(),
            ArrayElementType::Number
        );
        assert_eq!(Transform::log().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::log2().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::ln().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::sqrt().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::asinh().target_type(), ArrayElementType::Number);
        assert_eq!(
            Transform::pseudo_log().target_type(),
            ArrayElementType::Number
        );

        // Discrete transforms target their respective types
        assert_eq!(Transform::string().target_type(), ArrayElementType::String);
        assert_eq!(Transform::bool().target_type(), ArrayElementType::Boolean);
    }

    #[test]
    fn test_transform_string_creation() {
        let string = Transform::string();
        assert_eq!(string.transform_kind(), TransformKind::String);
        assert_eq!(string.name(), "string");
    }

    #[test]
    fn test_transform_bool_creation() {
        let bool_t = Transform::bool();
        assert_eq!(bool_t.transform_kind(), TransformKind::Bool);
        assert_eq!(bool_t.name(), "bool");
    }

    #[test]
    fn test_transform_from_name_string_aliases() {
        // All aliases should produce a String transform
        assert_eq!(
            Transform::from_name("string").unwrap().transform_kind(),
            TransformKind::String
        );
        assert_eq!(
            Transform::from_name("str").unwrap().transform_kind(),
            TransformKind::String
        );
        assert_eq!(
            Transform::from_name("varchar").unwrap().transform_kind(),
            TransformKind::String
        );
    }

    #[test]
    fn test_transform_from_name_bool_aliases() {
        // All aliases should produce a Bool transform
        assert_eq!(
            Transform::from_name("bool").unwrap().transform_kind(),
            TransformKind::Bool
        );
        assert_eq!(
            Transform::from_name("boolean").unwrap().transform_kind(),
            TransformKind::Bool
        );
    }

    #[test]
    fn test_transform_from_kind_string_bool() {
        let string = Transform::from_kind(TransformKind::String);
        assert_eq!(string.transform_kind(), TransformKind::String);

        let bool_t = Transform::from_kind(TransformKind::Bool);
        assert_eq!(bool_t.transform_kind(), TransformKind::Bool);
    }

    #[test]
    fn test_transform_kind_display_string_bool() {
        assert_eq!(format!("{}", TransformKind::String), "string");
        assert_eq!(format!("{}", TransformKind::Bool), "bool");
    }

    #[test]
    fn test_transform_integer_creation() {
        let integer = Transform::integer();
        assert_eq!(integer.transform_kind(), TransformKind::Integer);
        assert_eq!(integer.name(), "integer");
    }

    #[test]
    fn test_transform_from_name_integer_aliases() {
        // All aliases should produce an Integer transform
        assert_eq!(
            Transform::from_name("integer").unwrap().transform_kind(),
            TransformKind::Integer
        );
        assert_eq!(
            Transform::from_name("int").unwrap().transform_kind(),
            TransformKind::Integer
        );
        assert_eq!(
            Transform::from_name("bigint").unwrap().transform_kind(),
            TransformKind::Integer
        );
    }

    #[test]
    fn test_transform_from_kind_integer() {
        let integer = Transform::from_kind(TransformKind::Integer);
        assert_eq!(integer.transform_kind(), TransformKind::Integer);
    }

    #[test]
    fn test_transform_kind_display_integer() {
        assert_eq!(format!("{}", TransformKind::Integer), "integer");
    }

    #[test]
    fn test_transform_integer_target_type() {
        use crate::plot::ArrayElementType;
        // Integer transform targets Number (integers are numeric)
        assert_eq!(Transform::integer().target_type(), ArrayElementType::Number);
    }

    // ==================== Square Transform Tests ====================

    #[test]
    fn test_transform_square_creation() {
        let square = Transform::square();
        assert_eq!(square.transform_kind(), TransformKind::Square);
        assert_eq!(square.name(), "square");
    }

    #[test]
    fn test_transform_square_transform() {
        let sq = Transform::square();
        assert!((sq.transform(3.0) - 9.0).abs() < 1e-10);
        assert!((sq.transform(-3.0) - 9.0).abs() < 1e-10);
        assert!((sq.transform(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_square_inverse() {
        let sq = Transform::square();
        assert!((sq.inverse(9.0) - 3.0).abs() < 1e-10);
        assert!((sq.inverse(4.0) - 2.0).abs() < 1e-10);
        assert!((sq.inverse(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_from_name_square_aliases() {
        // Both "square" and "pow2" should produce a Square transform
        assert_eq!(
            Transform::from_name("square").unwrap().transform_kind(),
            TransformKind::Square
        );
        assert_eq!(
            Transform::from_name("pow2").unwrap().transform_kind(),
            TransformKind::Square
        );
    }

    #[test]
    fn test_transform_from_kind_square() {
        let square = Transform::from_kind(TransformKind::Square);
        assert_eq!(square.transform_kind(), TransformKind::Square);
    }

    #[test]
    fn test_transform_kind_display_square() {
        assert_eq!(format!("{}", TransformKind::Square), "square");
    }

    #[test]
    fn test_transform_square_is_inverse_of_sqrt() {
        let sqrt = Transform::sqrt();
        let square = Transform::square();
        // sqrt(square(x)) = x for non-negative x
        for &val in &[0.0, 1.0, 2.0, 5.0, 10.0] {
            let result = sqrt.transform(square.transform(val));
            assert!((result - val).abs() < 1e-10, "sqrt(square({})) != {}", val, val);
        }
    }

    // ==================== Exp Transform Tests ====================

    #[test]
    fn test_transform_exp10_creation() {
        let exp10 = Transform::exp10();
        assert_eq!(exp10.transform_kind(), TransformKind::Exp10);
        assert_eq!(exp10.name(), "exp10");
    }

    #[test]
    fn test_transform_exp10_transform() {
        let exp10 = Transform::exp10();
        assert!((exp10.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((exp10.transform(1.0) - 10.0).abs() < 1e-10);
        assert!((exp10.transform(2.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_exp10_inverse() {
        let exp10 = Transform::exp10();
        assert!((exp10.inverse(1.0) - 0.0).abs() < 1e-10);
        assert!((exp10.inverse(10.0) - 1.0).abs() < 1e-10);
        assert!((exp10.inverse(100.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_exp10_is_inverse_of_log10() {
        let log10 = Transform::log();
        let exp10 = Transform::exp10();
        // log10(exp10(x)) = x
        for &val in &[-1.0, 0.0, 1.0, 2.0, 3.0] {
            let result = log10.transform(exp10.transform(val));
            if val == 0.0 {
                assert!((result - val).abs() < 1e-10, "log10(exp10({})) != {}", val, val);
            } else {
                assert!((result - val).abs() / val.abs() < 1e-10, "log10(exp10({})) != {}", val, val);
            }
        }
    }

    #[test]
    fn test_transform_exp2_creation() {
        let exp2 = Transform::exp2();
        assert_eq!(exp2.transform_kind(), TransformKind::Exp2);
        assert_eq!(exp2.name(), "exp2");
    }

    #[test]
    fn test_transform_exp2_transform() {
        let exp2 = Transform::exp2();
        assert!((exp2.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((exp2.transform(1.0) - 2.0).abs() < 1e-10);
        assert!((exp2.transform(3.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_exp2_is_inverse_of_log2() {
        let log2 = Transform::log2();
        let exp2 = Transform::exp2();
        // log2(exp2(x)) = x
        for &val in &[-1.0, 0.0, 1.0, 2.0, 3.0] {
            let result = log2.transform(exp2.transform(val));
            if val == 0.0 {
                assert!((result - val).abs() < 1e-10, "log2(exp2({})) != {}", val, val);
            } else {
                assert!((result - val).abs() / val.abs() < 1e-10, "log2(exp2({})) != {}", val, val);
            }
        }
    }

    #[test]
    fn test_transform_exp_creation() {
        let exp = Transform::exp();
        assert_eq!(exp.transform_kind(), TransformKind::Exp);
        assert_eq!(exp.name(), "exp");
    }

    #[test]
    fn test_transform_exp_transform() {
        use std::f64::consts::E;
        let exp = Transform::exp();
        assert!((exp.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((exp.transform(1.0) - E).abs() < 1e-10);
    }

    #[test]
    fn test_transform_exp_is_inverse_of_ln() {
        let ln = Transform::ln();
        let exp = Transform::exp();
        // ln(exp(x)) = x
        for &val in &[-1.0, 0.0, 1.0, 2.0] {
            let result = ln.transform(exp.transform(val));
            if val == 0.0 {
                assert!((result - val).abs() < 1e-10, "ln(exp({})) != {}", val, val);
            } else {
                assert!((result - val).abs() / val.abs() < 1e-10, "ln(exp({})) != {}", val, val);
            }
        }
    }

    #[test]
    fn test_transform_from_name_exp_variants() {
        assert_eq!(
            Transform::from_name("exp10").unwrap().transform_kind(),
            TransformKind::Exp10
        );
        assert_eq!(
            Transform::from_name("exp2").unwrap().transform_kind(),
            TransformKind::Exp2
        );
        assert_eq!(
            Transform::from_name("exp").unwrap().transform_kind(),
            TransformKind::Exp
        );
    }

    #[test]
    fn test_transform_from_kind_exp_variants() {
        assert_eq!(Transform::from_kind(TransformKind::Exp10).transform_kind(), TransformKind::Exp10);
        assert_eq!(Transform::from_kind(TransformKind::Exp2).transform_kind(), TransformKind::Exp2);
        assert_eq!(Transform::from_kind(TransformKind::Exp).transform_kind(), TransformKind::Exp);
    }

    #[test]
    fn test_transform_kind_display_exp_variants() {
        assert_eq!(format!("{}", TransformKind::Exp10), "exp10");
        assert_eq!(format!("{}", TransformKind::Exp2), "exp2");
        assert_eq!(format!("{}", TransformKind::Exp), "exp");
    }

    #[test]
    fn test_transform_square_exp_target_type() {
        use crate::plot::ArrayElementType;
        // All inverse transforms target Number
        assert_eq!(Transform::square().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::exp10().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::exp2().target_type(), ArrayElementType::Number);
        assert_eq!(Transform::exp().target_type(), ArrayElementType::Number);
    }
}
