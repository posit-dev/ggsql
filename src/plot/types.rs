//! Input types for ggsql specification
//!
//! This module defines types that model user input: mappings, data sources,
//! settings, and values. These are the building blocks used in AST types
//! to capture what the user specified in their query.

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars::prelude::DataType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Schema Types (derived from input data)
// =============================================================================

/// Column information from a data source schema
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Data type of the column
    pub dtype: DataType,
    /// Whether this column is discrete (suitable for grouping)
    /// Discrete: String, Boolean, Categorical
    /// Continuous: numeric types, Date, Datetime, Time
    pub is_discrete: bool,
    /// Minimum value for this column (computed from data)
    pub min: Option<ArrayElement>,
    /// Maximum value for this column (computed from data)
    pub max: Option<ArrayElement>,
}

/// Schema of a data source - list of columns with type info
pub type Schema = Vec<ColumnInfo>;

// =============================================================================
// Mapping Types
// =============================================================================

/// Unified aesthetic mapping specification
///
/// Used for both global mappings (VISUALISE clause) and layer mappings (MAPPING clause).
/// Supports wildcards combined with explicit mappings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Mappings {
    /// Whether a wildcard (*) was specified
    pub wildcard: bool,
    /// Explicit aesthetic mappings (aesthetic → value)
    pub aesthetics: HashMap<String, AestheticValue>,
}

impl Mappings {
    /// Create a new empty Mappings
    pub fn new() -> Self {
        Self {
            wildcard: false,
            aesthetics: HashMap::new(),
        }
    }

    /// Create a new Mappings with wildcard flag set
    pub fn with_wildcard() -> Self {
        Self {
            wildcard: true,
            aesthetics: HashMap::new(),
        }
    }

    /// Check if the mappings are empty (no wildcard and no aesthetics)
    pub fn is_empty(&self) -> bool {
        !self.wildcard && self.aesthetics.is_empty()
    }

    /// Insert an aesthetic mapping
    pub fn insert(&mut self, aesthetic: impl Into<String>, value: AestheticValue) {
        self.aesthetics.insert(aesthetic.into(), value);
    }

    /// Get an aesthetic value by name
    pub fn get(&self, aesthetic: &str) -> Option<&AestheticValue> {
        self.aesthetics.get(aesthetic)
    }

    /// Check if an aesthetic is mapped
    pub fn contains_key(&self, aesthetic: &str) -> bool {
        self.aesthetics.contains_key(aesthetic)
    }

    /// Get the number of explicit aesthetic mappings
    pub fn len(&self) -> usize {
        self.aesthetics.len()
    }
}

// =============================================================================
// Data Source Types
// =============================================================================

/// Data source for visualization or layer (from VISUALISE FROM or MAPPING ... FROM clause)
///
/// Allows specification of a data source - either a CTE/table name or a file path.
/// Used both for global `VISUALISE FROM` and layer-specific `MAPPING ... FROM`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSource {
    /// CTE or table name (unquoted identifier)
    Identifier(String),
    /// File path (quoted string like 'data.csv')
    FilePath(String),
}

impl DataSource {
    /// Returns the source as a string reference
    pub fn as_str(&self) -> &str {
        match self {
            DataSource::Identifier(s) => s,
            DataSource::FilePath(s) => s,
        }
    }

    /// Returns true if this is a file path source
    pub fn is_file(&self) -> bool {
        matches!(self, DataSource::FilePath(_))
    }
}

// =============================================================================
// Value Types (used in mappings/settings)
// =============================================================================

/// Value for aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AestheticValue {
    /// Column reference
    Column {
        name: String,
        /// Whether this is a dummy/placeholder column (e.g., for bar charts without x mapped)
        is_dummy: bool,
    },
    /// Literal value (quoted string, number, or boolean)
    Literal(LiteralValue),
}

impl AestheticValue {
    /// Create a column mapping
    pub fn standard_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: false,
        }
    }

    /// Create a dummy/placeholder column mapping (e.g., for bar charts without x mapped)
    pub fn dummy_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: true,
        }
    }

    /// Get column name if this is a column mapping
    pub fn column_name(&self) -> Option<&str> {
        match self {
            Self::Column { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Check if this is a dummy/placeholder column
    pub fn is_dummy(&self) -> bool {
        match self {
            Self::Column { is_dummy, .. } => *is_dummy,
            _ => false,
        }
    }

    /// Check if this is a literal value (not a column mapping)
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }
}

impl std::fmt::Display for AestheticValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AestheticValue::Column { name, .. } => write!(f, "{}", name),
            AestheticValue::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

/// Literal values in aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

impl std::fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralValue::String(s) => write!(f, "'{}'", s),
            LiteralValue::Number(n) => write!(f, "{}", n),
            LiteralValue::Boolean(b) => write!(f, "{}", b),
        }
    }
}

/// Value for geom parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
    /// Null value to explicitly opt out of a setting
    Null,
}

/// Elements in arrays (shared type for property values)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrayElement {
    String(String),
    Number(f64),
    Boolean(bool),
    /// Null placeholder for partial input range inference (e.g., SCALE x FROM [0, null])
    Null,
    /// Date value (days since Unix epoch 1970-01-01)
    Date(i32),
    /// DateTime value (microseconds since Unix epoch)
    DateTime(i64),
    /// Time value (nanoseconds since midnight)
    Time(i64),
}

/// Days from CE to Unix epoch (1970-01-01)
const UNIX_EPOCH_CE_DAYS: i32 = 719163;

/// Convert days-since-epoch to ISO date string
fn date_to_iso_string(days: i32) -> String {
    NaiveDate::from_num_days_from_ce_opt(days + UNIX_EPOCH_CE_DAYS)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| days.to_string())
}

/// Convert microseconds-since-epoch to ISO datetime string
fn datetime_to_iso_string(micros: i64) -> String {
    DateTime::from_timestamp_micros(micros)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
        .unwrap_or_else(|| micros.to_string())
}

/// Convert nanoseconds-since-midnight to ISO time string
fn time_to_iso_string(nanos: i64) -> String {
    let secs = (nanos / 1_000_000_000) as u32;
    let nano_part = (nanos % 1_000_000_000) as u32;
    NaiveTime::from_num_seconds_from_midnight_opt(secs, nano_part)
        .map(|t| t.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| format!("{}ns", nanos))
}

/// Format number for display (remove trailing zeros for integers)
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.0}", n)
    } else {
        n.to_string()
    }
}

impl ArrayElement {
    /// Convert to f64 for numeric calculations
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            Self::Date(d) => Some(*d as f64),
            Self::DateTime(dt) => Some(*dt as f64),
            Self::Time(t) => Some(*t as f64),
            _ => None,
        }
    }

    /// Parse ISO date string "YYYY-MM-DD" to Date variant
    pub fn from_date_string(s: &str) -> Option<Self> {
        NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .ok()
            .map(|d| Self::Date(d.num_days_from_ce() - UNIX_EPOCH_CE_DAYS))
    }

    /// Parse ISO datetime string to DateTime variant
    pub fn from_datetime_string(s: &str) -> Option<Self> {
        // Try multiple formats: with/without fractional seconds, with/without Z
        for fmt in &[
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ] {
            if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
                return Some(Self::DateTime(dt.and_utc().timestamp_micros()));
            }
        }
        None
    }

    /// Parse ISO time string "HH:MM:SS[.sss]" to Time variant
    pub fn from_time_string(s: &str) -> Option<Self> {
        for fmt in &["%H:%M:%S%.f", "%H:%M:%S", "%H:%M"] {
            if let Ok(t) = NaiveTime::parse_from_str(s, fmt) {
                // Convert to nanoseconds since midnight
                let nanos =
                    t.num_seconds_from_midnight() as i64 * 1_000_000_000 + t.nanosecond() as i64;
                return Some(Self::Time(nanos));
            }
        }
        None
    }

    /// Convert to string for HashMap keys and display
    pub fn to_key_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Number(n) => format_number(*n),
            Self::Boolean(b) => b.to_string(),
            Self::Null => "null".to_string(),
            Self::Date(d) => date_to_iso_string(*d),
            Self::DateTime(dt) => datetime_to_iso_string(*dt),
            Self::Time(t) => time_to_iso_string(*t),
        }
    }

    /// Convert to a serde_json::Value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ArrayElement::String(s) => serde_json::Value::String(s.clone()),
            ArrayElement::Number(n) => serde_json::json!(n),
            ArrayElement::Boolean(b) => serde_json::Value::Bool(*b),
            ArrayElement::Null => serde_json::Value::Null,
            // Temporal types serialize as ISO strings for JSON
            ArrayElement::Date(d) => serde_json::Value::String(date_to_iso_string(*d)),
            ArrayElement::DateTime(dt) => serde_json::Value::String(datetime_to_iso_string(*dt)),
            ArrayElement::Time(t) => serde_json::Value::String(time_to_iso_string(*t)),
        }
    }
}

impl ParameterValue {
    /// Convert to a serde_json::Value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ParameterValue::String(s) => serde_json::Value::String(s.clone()),
            ParameterValue::Number(n) => serde_json::json!(n),
            ParameterValue::Boolean(b) => serde_json::Value::Bool(*b),
            ParameterValue::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|e| e.to_json()).collect())
            }
            ParameterValue::Null => serde_json::Value::Null,
        }
    }

    /// Check if this is a null value
    pub fn is_null(&self) -> bool {
        matches!(self, ParameterValue::Null)
    }

    /// Try to extract as a string value
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParameterValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as a number value
    pub fn as_number(&self) -> Option<f64> {
        match self {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to extract as a boolean value
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract as an array value
    pub fn as_array(&self) -> Option<&[ArrayElement]> {
        match self {
            ParameterValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

// =============================================================================
// SQL Expression Type
// =============================================================================

/// Raw SQL expression for layer-specific clauses (FILTER, ORDER BY)
///
/// This stores raw SQL text verbatim, which is passed directly to the database
/// backend. This allows any valid SQL expression to be used.
///
/// Example values:
/// - `"x > 10"` (filter)
/// - `"region = 'North' AND year >= 2020"` (filter)
/// - `"date ASC"` (order by)
/// - `"category, value DESC"` (order by)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SqlExpression(pub String);

impl SqlExpression {
    /// Create a new SQL expression from raw text
    pub fn new(sql: impl Into<String>) -> Self {
        Self(sql.into())
    }

    /// Get the raw SQL text
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the raw SQL text
    pub fn into_string(self) -> String {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_from_string() {
        let elem = ArrayElement::from_date_string("2024-01-15").unwrap();
        assert!(matches!(elem, ArrayElement::Date(_)));
        assert_eq!(elem.to_key_string(), "2024-01-15");
    }

    #[test]
    fn test_date_from_string_roundtrip() {
        // Test that parsing and converting back produces the same date
        let original = "2024-06-30";
        let elem = ArrayElement::from_date_string(original).unwrap();
        assert_eq!(elem.to_key_string(), original);
    }

    #[test]
    fn test_datetime_from_string() {
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
        assert!(elem.to_key_string().starts_with("2024-01-15T10:30:00"));
    }

    #[test]
    fn test_datetime_from_string_with_space() {
        let elem = ArrayElement::from_datetime_string("2024-01-15 10:30:00").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
    }

    #[test]
    fn test_time_from_string() {
        let elem = ArrayElement::from_time_string("14:30:00").unwrap();
        assert!(matches!(elem, ArrayElement::Time(_)));
        assert_eq!(elem.to_key_string(), "14:30:00");
    }

    #[test]
    fn test_time_from_string_with_millis() {
        let elem = ArrayElement::from_time_string("14:30:00.123").unwrap();
        assert!(matches!(elem, ArrayElement::Time(_)));
    }

    #[test]
    fn test_time_from_string_short() {
        let elem = ArrayElement::from_time_string("14:30").unwrap();
        assert!(matches!(elem, ArrayElement::Time(_)));
        assert_eq!(elem.to_key_string(), "14:30:00");
    }

    #[test]
    fn test_date_to_f64() {
        // 2024-01-15 is roughly 19738 days since epoch (1970-01-01)
        let elem = ArrayElement::from_date_string("2024-01-15").unwrap();
        let days = elem.to_f64().unwrap();
        // Verify the date is in a reasonable range
        assert!(days > 19000.0 && days < 20000.0);
    }

    #[test]
    fn test_time_to_f64() {
        let elem = ArrayElement::from_time_string("12:00:00").unwrap();
        let nanos = elem.to_f64().unwrap();
        // 12 hours = 12 * 60 * 60 * 1_000_000_000 nanoseconds
        assert_eq!(nanos, 43_200_000_000_000.0);
    }

    #[test]
    fn test_date_to_json() {
        let elem = ArrayElement::from_date_string("2024-01-15").unwrap();
        let json = elem.to_json();
        assert_eq!(json, serde_json::json!("2024-01-15"));
    }

    #[test]
    fn test_datetime_to_json() {
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00").unwrap();
        let json = elem.to_json();
        // Datetime serializes as ISO string
        assert!(json.is_string());
        assert!(json.as_str().unwrap().starts_with("2024-01-15T10:30:00"));
    }

    #[test]
    fn test_time_to_json() {
        let elem = ArrayElement::from_time_string("14:30:00").unwrap();
        let json = elem.to_json();
        assert_eq!(json, serde_json::json!("14:30:00"));
    }

    #[test]
    fn test_number_to_f64() {
        let elem = ArrayElement::Number(42.5);
        assert_eq!(elem.to_f64(), Some(42.5));
    }

    #[test]
    fn test_string_to_f64_returns_none() {
        let elem = ArrayElement::String("hello".to_string());
        assert_eq!(elem.to_f64(), None);
    }

    #[test]
    fn test_to_key_string_number_integer() {
        let elem = ArrayElement::Number(25.0);
        assert_eq!(elem.to_key_string(), "25");
    }

    #[test]
    fn test_to_key_string_number_decimal() {
        let elem = ArrayElement::Number(25.5);
        assert_eq!(elem.to_key_string(), "25.5");
    }

    #[test]
    fn test_invalid_date_returns_none() {
        assert!(ArrayElement::from_date_string("not-a-date").is_none());
        assert!(ArrayElement::from_date_string("2024/01/15").is_none());
    }

    #[test]
    fn test_invalid_time_returns_none() {
        assert!(ArrayElement::from_time_string("not-a-time").is_none());
        assert!(ArrayElement::from_time_string("25:00:00").is_none());
    }
}
