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
// Array Element Type (for coercion)
// =============================================================================

/// Type of an ArrayElement value, used for type inference and coercion.
///
/// This enum represents the semantic type of values in a scale's input range,
/// allowing discrete scales to infer the target type from their domain values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayElementType {
    String,
    Number,
    Boolean,
    Date,
    DateTime,
    Time,
}

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

    /// Transform aesthetic keys from user-facing to internal names.
    ///
    /// Uses the provided AestheticContext to map user-facing positional aesthetic names
    /// (e.g., "x", "y", "theta", "radius") to internal names (e.g., "pos1", "pos2").
    /// Non-positional aesthetics (e.g., "color", "size") are left unchanged.
    pub fn transform_to_internal(&mut self, ctx: &super::AestheticContext) {
        let original_aesthetics = std::mem::take(&mut self.aesthetics);
        for (aesthetic, value) in original_aesthetics {
            let internal_name = ctx
                .map_user_to_internal(&aesthetic)
                .map(|s| s.to_string())
                .unwrap_or(aesthetic);
            self.aesthetics.insert(internal_name, value);
        }
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
    /// Annotation layer (PLACE clause)
    /// Row count and array recycling handled during SQL generation
    Annotation,
}

impl DataSource {
    /// Returns the source as a string reference
    pub fn as_str(&self) -> &str {
        match self {
            DataSource::Identifier(s) => s,
            DataSource::FilePath(s) => s,
            DataSource::Annotation => "__annotation__",
        }
    }

    /// Returns true if this is a file path source
    pub fn is_file(&self) -> bool {
        matches!(self, DataSource::FilePath(_))
    }

    /// Returns true if this is an annotation layer source
    pub fn is_annotation(&self) -> bool {
        matches!(self, DataSource::Annotation)
    }
}

// =============================================================================
// Value Types (used in mappings/settings)
// =============================================================================

/// Value for aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AestheticValue {
    /// Column reference from data source
    Column {
        name: String,
        /// Original column name before internal renaming (for labels)
        /// When columns are renamed to internal names like `__ggsql_aes_x__`,
        /// this preserves the original column name (e.g., "bill_dep") for axis labels.
        original_name: Option<String>,
        /// Whether this is a dummy/placeholder column (e.g., for bar charts without x mapped)
        is_dummy: bool,
    },
    /// Annotation column for non-positional aesthetics (synthesized from PLACE literals)
    /// These columns are generated from user-specified literal values in visual space
    /// (e.g., color => 'red', size => 10) and use identity scales (no transformation).
    /// Positional annotations (x, y) use Column instead since they're in data coordinate space.
    AnnotationColumn { name: String },
    /// Literal value (quoted string, number, or boolean)
    Literal(ParameterValue),
}

impl AestheticValue {
    /// Create a standard column mapping
    pub fn standard_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            original_name: None,
            is_dummy: false,
        }
    }

    /// Create a dummy/placeholder column mapping (e.g., for bar charts without x mapped)
    pub fn dummy_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            original_name: None,
            is_dummy: true,
        }
    }

    /// Create a column mapping with an explicit original name.
    ///
    /// Used when renaming columns to internal names but preserving the original
    /// column name for labels.
    pub fn column_with_original(name: impl Into<String>, original_name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            original_name: Some(original_name.into()),
            is_dummy: false,
        }
    }

    /// Create an annotation column mapping (synthesized from PLACE literals)
    pub fn annotation_column(name: impl Into<String>) -> Self {
        Self::AnnotationColumn { name: name.into() }
    }

    /// Get column name if this is a column mapping
    pub fn column_name(&self) -> Option<&str> {
        match self {
            Self::Column { name, .. } | Self::AnnotationColumn { name } => Some(name),
            _ => None,
        }
    }

    /// Get the name to use for labels (axis titles, legend titles).
    ///
    /// Returns the original column name if available, otherwise the current name.
    /// This ensures axis labels show user-friendly names like "bill_dep" instead
    /// of internal names like "__ggsql_aes_x__".
    pub fn label_name(&self) -> Option<&str> {
        match self {
            Self::Column {
                name,
                original_name,
                ..
            } => Some(original_name.as_deref().unwrap_or(name)),
            Self::AnnotationColumn { name } => Some(name),
            _ => None,
        }
    }

    /// Check if this is a dummy/placeholder column
    pub fn is_dummy(&self) -> bool {
        matches!(self, Self::Column { is_dummy: true, .. })
    }

    /// Check if this is an annotation column
    pub fn is_annotation(&self) -> bool {
        matches!(self, Self::AnnotationColumn { .. })
    }

    /// Check if this is a literal value (not a column mapping)
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }
}

impl std::fmt::Display for AestheticValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AestheticValue::Column { name, .. } | AestheticValue::AnnotationColumn { name } => {
                write!(f, "{}", name)
            }
            AestheticValue::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

/// Static version of AestheticValue for use in default remappings.
///
/// Similar to how `ParamDefinitionValue` is the static version of `ParameterValue`,
/// this type uses `&'static str` instead of `String` so it can be used in
/// static arrays returned by `GeomTrait::default_remappings()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultAestheticValue {
    /// Column reference (stat column name)
    Column(&'static str),
    /// Literal string value
    String(&'static str),
    /// Literal number value
    Number(f64),
    /// Literal boolean value
    Boolean(bool),
    /// Supported but no default value (optional aesthetic)
    Null,
    /// Required aesthetic (must be provided via MAPPING)
    Required,
    /// Delayed aesthetic (produced by stat transform, valid for REMAPPING only, not MAPPING)
    Delayed,
}

impl DefaultAestheticValue {
    /// Convert to ParameterValue
    ///
    /// Returns String/Number/Boolean for literal defaults.
    /// Returns Null for Column/Null/Required/Delayed (non-literal variants).
    /// Use this to extract SETTING-compatible values from defaults.
    pub fn to_parameter_value(&self) -> ParameterValue {
        match self {
            Self::String(s) => ParameterValue::String(s.to_string()),
            Self::Number(n) => ParameterValue::Number(*n),
            Self::Boolean(b) => ParameterValue::Boolean(*b),
            Self::Column(_) | Self::Null | Self::Required | Self::Delayed => ParameterValue::Null,
        }
    }

    /// Convert to owned AestheticValue
    pub fn to_aesthetic_value(&self) -> AestheticValue {
        match self {
            Self::Column(name) => AestheticValue::standard_column(name.to_string()),
            // All literal variants (String/Number/Boolean) and non-literals (Null/Required/Delayed)
            _ => AestheticValue::Literal(self.to_parameter_value()),
        }
    }
}

/// Value for geom parameters (also used for literals)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
    /// Null value to explicitly opt out of a setting
    Null,
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterValue::String(s) => write!(f, "'{}'", s),
            ParameterValue::Number(n) => write!(f, "{}", n),
            ParameterValue::Boolean(b) => write!(f, "{}", b),
            ParameterValue::Array(arr) => {
                write!(f, "[")?;
                for (i, elem) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match elem {
                        ArrayElement::String(s) => write!(f, "'{}'", s)?,
                        ArrayElement::Number(n) => write!(f, "{}", n)?,
                        ArrayElement::Boolean(b) => write!(f, "{}", b)?,
                        ArrayElement::Null => write!(f, "null")?,
                        ArrayElement::Date(d) => write!(f, "'{}'", d)?,
                        ArrayElement::DateTime(dt) => write!(f, "'{}'", dt)?,
                        ArrayElement::Time(t) => write!(f, "'{}'", t)?,
                    }
                }
                write!(f, "]")
            }
            ParameterValue::Null => write!(f, "null"),
        }
    }
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
pub fn format_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.0}", n)
    } else {
        n.to_string()
    }
}

/// Get type name for error messages
fn target_type_name(t: ArrayElementType) -> &'static str {
    match t {
        ArrayElementType::String => "string",
        ArrayElementType::Number => "number",
        ArrayElementType::Boolean => "boolean",
        ArrayElementType::Date => "date",
        ArrayElementType::DateTime => "datetime",
        ArrayElementType::Time => "time",
    }
}

impl ArrayElement {
    /// Get the type of this element.
    ///
    /// Returns None for Null values.
    pub fn element_type(&self) -> Option<ArrayElementType> {
        match self {
            Self::String(_) => Some(ArrayElementType::String),
            Self::Number(_) => Some(ArrayElementType::Number),
            Self::Boolean(_) => Some(ArrayElementType::Boolean),
            Self::Date(_) => Some(ArrayElementType::Date),
            Self::DateTime(_) => Some(ArrayElementType::DateTime),
            Self::Time(_) => Some(ArrayElementType::Time),
            Self::Null => None,
        }
    }

    /// Infer the dominant type from a collection of ArrayElements.
    ///
    /// Used by discrete scales to determine the target type from their input range.
    /// Nulls are ignored. Returns None if all values are null or the slice is empty.
    ///
    /// If multiple types are present, uses priority: Boolean > Number > Date > DateTime > Time > String
    /// (Boolean is highest because it's most specific; String is lowest as it's the fallback)
    pub fn infer_type(values: &[ArrayElement]) -> Option<ArrayElementType> {
        let mut found_bool = false;
        let mut found_number = false;
        let mut found_date = false;
        let mut found_datetime = false;
        let mut found_time = false;
        let mut found_string = false;

        for elem in values {
            match elem {
                Self::Boolean(_) => found_bool = true,
                Self::Number(_) => found_number = true,
                Self::Date(_) => found_date = true,
                Self::DateTime(_) => found_datetime = true,
                Self::Time(_) => found_time = true,
                Self::String(_) => found_string = true,
                Self::Null => {}
            }
        }

        // Priority order: most specific to least specific
        if found_bool {
            Some(ArrayElementType::Boolean)
        } else if found_number {
            Some(ArrayElementType::Number)
        } else if found_date {
            Some(ArrayElementType::Date)
        } else if found_datetime {
            Some(ArrayElementType::DateTime)
        } else if found_time {
            Some(ArrayElementType::Time)
        } else if found_string {
            Some(ArrayElementType::String)
        } else {
            None
        }
    }

    /// Coerce this element to the target type.
    ///
    /// Returns Ok with the coerced value, or Err with a description if coercion is impossible.
    ///
    /// Coercion paths:
    /// - String → Boolean: "true"/"false"/"yes"/"no"/"1"/"0" (case-insensitive)
    /// - String → Number: parse as f64
    /// - String → Date/DateTime/Time: parse ISO format
    /// - Number → Boolean: 0 = false, non-zero = true
    /// - Number → String: format as string
    /// - Number → Date: interpret as days since Unix epoch
    /// - Number → DateTime: interpret as microseconds since Unix epoch
    /// - Number → Time: interpret as nanoseconds since midnight
    /// - Boolean → Number: false = 0, true = 1
    /// - Boolean → String: "true"/"false"
    /// - Null → any: stays Null
    pub fn coerce_to(&self, target: ArrayElementType) -> Result<ArrayElement, String> {
        // Already the right type?
        if self.element_type() == Some(target) {
            return Ok(self.clone());
        }

        // Null stays Null
        if matches!(self, Self::Null) {
            return Ok(Self::Null);
        }

        match (self, target) {
            // String → Boolean
            (Self::String(s), ArrayElementType::Boolean) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Ok(Self::Boolean(true)),
                "false" | "no" | "0" => Ok(Self::Boolean(false)),
                _ => Err(format!("Cannot coerce string '{}' to boolean", s)),
            },

            // String → Number
            (Self::String(s), ArrayElementType::Number) => s
                .parse::<f64>()
                .map(Self::Number)
                .map_err(|_| format!("Cannot coerce string '{}' to number", s)),

            // String → Date
            (Self::String(s), ArrayElementType::Date) => {
                Self::from_date_string(s).ok_or_else(|| {
                    format!("Cannot coerce string '{}' to date (expected YYYY-MM-DD)", s)
                })
            }

            // String → DateTime
            (Self::String(s), ArrayElementType::DateTime) => Self::from_datetime_string(s)
                .ok_or_else(|| format!("Cannot coerce string '{}' to datetime", s)),

            // String → Time
            (Self::String(s), ArrayElementType::Time) => Self::from_time_string(s)
                .ok_or_else(|| format!("Cannot coerce string '{}' to time (expected HH:MM:SS)", s)),

            // String → String (identity, already handled above but for completeness)
            (Self::String(s), ArrayElementType::String) => Ok(Self::String(s.clone())),

            // Number → Boolean
            (Self::Number(n), ArrayElementType::Boolean) => Ok(Self::Boolean(*n != 0.0)),

            // Number → String
            (Self::Number(n), ArrayElementType::String) => Ok(Self::String(format_number(*n))),

            // Number → Date (days since epoch)
            (Self::Number(n), ArrayElementType::Date) => Ok(Self::Date(*n as i32)),

            // Number → DateTime (microseconds since epoch)
            (Self::Number(n), ArrayElementType::DateTime) => Ok(Self::DateTime(*n as i64)),

            // Number → Time (nanoseconds since midnight)
            (Self::Number(n), ArrayElementType::Time) => Ok(Self::Time(*n as i64)),

            // Boolean → Number
            (Self::Boolean(b), ArrayElementType::Number) => {
                Ok(Self::Number(if *b { 1.0 } else { 0.0 }))
            }

            // Boolean → String
            (Self::Boolean(b), ArrayElementType::String) => Ok(Self::String(b.to_string())),

            // Boolean → temporal types: not supported
            (Self::Boolean(_), ArrayElementType::Date)
            | (Self::Boolean(_), ArrayElementType::DateTime)
            | (Self::Boolean(_), ArrayElementType::Time) => Err(format!(
                "Cannot coerce boolean to {}",
                target_type_name(target)
            )),

            // Date → String
            (Self::Date(d), ArrayElementType::String) => Ok(Self::String(date_to_iso_string(*d))),

            // Date → Number (days since epoch)
            (Self::Date(d), ArrayElementType::Number) => Ok(Self::Number(*d as f64)),

            // DateTime → String
            (Self::DateTime(dt), ArrayElementType::String) => {
                Ok(Self::String(datetime_to_iso_string(*dt)))
            }

            // DateTime → Number (microseconds since epoch)
            (Self::DateTime(dt), ArrayElementType::Number) => Ok(Self::Number(*dt as f64)),

            // Time → String
            (Self::Time(t), ArrayElementType::String) => Ok(Self::String(time_to_iso_string(*t))),

            // Time → Number (nanoseconds since midnight)
            (Self::Time(t), ArrayElementType::Number) => Ok(Self::Number(*t as f64)),

            // Temporal → Boolean: not supported
            (Self::Date(_), ArrayElementType::Boolean)
            | (Self::DateTime(_), ArrayElementType::Boolean)
            | (Self::Time(_), ArrayElementType::Boolean) => {
                Err(format!("Cannot coerce {} to boolean", self.type_name()))
            }

            // Cross-temporal conversions: not supported (lossy)
            (Self::Date(_), ArrayElementType::DateTime)
            | (Self::Date(_), ArrayElementType::Time)
            | (Self::DateTime(_), ArrayElementType::Date)
            | (Self::DateTime(_), ArrayElementType::Time)
            | (Self::Time(_), ArrayElementType::Date)
            | (Self::Time(_), ArrayElementType::DateTime) => Err(format!(
                "Cannot coerce {} to {}",
                self.type_name(),
                target_type_name(target)
            )),

            // Identity cases (already handled by early return, but needed for exhaustiveness)
            (Self::Number(n), ArrayElementType::Number) => Ok(Self::Number(*n)),
            (Self::Boolean(b), ArrayElementType::Boolean) => Ok(Self::Boolean(*b)),
            (Self::Date(d), ArrayElementType::Date) => Ok(Self::Date(*d)),
            (Self::DateTime(dt), ArrayElementType::DateTime) => Ok(Self::DateTime(*dt)),
            (Self::Time(t), ArrayElementType::Time) => Ok(Self::Time(*t)),

            // Null cases are handled at the top
            (Self::Null, _) => Ok(Self::Null),
        }
    }

    /// Homogenize a slice of array elements to a common type.
    ///
    /// Infers the target type from all elements, then attempts to coerce all elements
    /// to that type. If coercion fails (e.g., string + number), falls back to String type.
    ///
    /// Returns a new vector with homogenized elements.
    pub fn homogenize(values: &[Self]) -> Vec<Self> {
        // Infer target type from all elements
        let Some(target_type) = Self::infer_type(values) else {
            // All nulls or empty array - return cloned as-is
            return values.to_vec();
        };

        // Try to coerce all elements to the inferred type
        let coerced: Result<Vec<_>, _> = values
            .iter()
            .map(|elem| elem.coerce_to(target_type))
            .collect();

        match coerced {
            Ok(coerced_arr) => coerced_arr,
            Err(_) => {
                // Coercion failed - fall back to String type
                values
                    .iter()
                    .map(|elem| {
                        elem.coerce_to(ArrayElementType::String)
                            .unwrap_or(Self::Null)
                    })
                    .collect()
            }
        }
    }

    /// Convert this element to a SQL literal string.
    ///
    /// Used for generating SQL expressions from literal values.
    pub fn to_sql(&self) -> String {
        match self {
            Self::String(s) => format!("'{}'", s.replace('\'', "''")),
            Self::Number(n) => n.to_string(),
            Self::Boolean(b) => {
                if *b {
                    "TRUE".to_string()
                } else {
                    "FALSE".to_string()
                }
            }
            Self::Date(d) => {
                // Convert days since epoch to DATE
                // Use CAST to ensure result is DATE type, not TIMESTAMP
                format!("CAST(DATE '1970-01-01' + INTERVAL {} DAY AS DATE)", d)
            }
            Self::DateTime(dt) => {
                // Convert microseconds since epoch to TIMESTAMP
                format!(
                    "TIMESTAMP '1970-01-01 00:00:00' + INTERVAL {} MICROSECOND",
                    dt
                )
            }
            Self::Time(t) => {
                // Convert nanoseconds since midnight to TIME
                let seconds = t / 1_000_000_000;
                let nanos = t % 1_000_000_000;
                format!(
                    "TIME '00:00:00' + INTERVAL {} SECOND + INTERVAL {} NANOSECOND",
                    seconds, nanos
                )
            }
            Self::Null => "NULL".to_string(),
        }
    }

    /// Get the type name for error messages.
    fn type_name(&self) -> &'static str {
        match self {
            Self::String(_) => "string",
            Self::Number(_) => "number",
            Self::Boolean(_) => "boolean",
            Self::Date(_) => "date",
            Self::DateTime(_) => "datetime",
            Self::Time(_) => "time",
            Self::Null => "null",
        }
    }

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
    ///
    /// Supports timezone-aware formats:
    /// - RFC3339: `2024-01-15T10:30:00Z`, `2024-01-15T10:30:00+05:30`
    /// - With offset: `2024-01-15T10:30:00+0530`
    ///
    /// And timezone-naive formats (interpreted as UTC):
    /// - `2024-01-15T10:30:00`, `2024-01-15T10:30:00.123`
    /// - `2024-01-15 10:30:00`
    pub fn from_datetime_string(s: &str) -> Option<Self> {
        // Try RFC3339/ISO8601 with timezone first (e.g., "2024-01-15T10:30:00Z", "2024-01-15T10:30:00+05:30")
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Some(Self::DateTime(dt.timestamp_micros()));
        }

        // Try formats with explicit timezone offset (non-RFC3339 variants)
        for fmt in &[
            "%Y-%m-%dT%H:%M:%S%.f%:z", // 2024-01-15T10:30:00.123+05:30
            "%Y-%m-%dT%H:%M:%S%:z",    // 2024-01-15T10:30:00+05:30
            "%Y-%m-%dT%H:%M:%S%.f%z",  // 2024-01-15T10:30:00.123+0530
            "%Y-%m-%dT%H:%M:%S%z",     // 2024-01-15T10:30:00+0530
            "%Y-%m-%d %H:%M:%S%:z",    // 2024-01-15 10:30:00+05:30
            "%Y-%m-%d %H:%M:%S%z",     // 2024-01-15 10:30:00+0530
        ] {
            if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
                return Some(Self::DateTime(dt.timestamp_micros()));
            }
        }

        // Fall back to naive (timezone-unaware), assumed UTC
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

    /// Try to parse string as temporal type (Date/DateTime/Time).
    ///
    /// If this is a String variant, attempts to parse it as a temporal type
    /// in order of specificity: DateTime > Date > Time.
    /// If parsing succeeds, returns the temporal variant. Otherwise returns self unchanged.
    ///
    /// For non-String variants, returns self unchanged.
    ///
    /// # Example
    /// ```ignore
    /// let elem = ArrayElement::String("1973-06-01".to_string());
    /// let parsed = elem.try_as_temporal();
    /// // parsed is now ArrayElement::Date(...)
    /// ```
    pub fn try_as_temporal(self) -> Self {
        if let Self::String(ref s) = self {
            // Try DateTime first (most specific)
            if let Some(dt) = Self::from_datetime_string(s) {
                return dt;
            }
            // Try Date
            if let Some(d) = Self::from_date_string(s) {
                return d;
            }
            // Try Time
            if let Some(t) = Self::from_time_string(s) {
                return t;
            }
        }
        // Fall back to original value if not a string or parsing failed
        self
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

    /// Convert Date (days since epoch) to ISO string "YYYY-MM-DD"
    pub fn date_to_iso(days: i32) -> String {
        date_to_iso_string(days)
    }

    /// Convert DateTime (microseconds since epoch) to ISO string
    pub fn datetime_to_iso(micros: i64) -> String {
        datetime_to_iso_string(micros)
    }

    /// Convert Time (nanoseconds since midnight) to ISO string "HH:MM:SS"
    pub fn time_to_iso(nanos: i64) -> String {
        time_to_iso_string(nanos)
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

    /// Convert this parameter value to a SQL literal string.
    ///
    /// Only supports scalar values (String, Number, Boolean, Null).
    /// Arrays are handled separately in annotation layer VALUES clause generation.
    pub fn to_sql(&self) -> String {
        match self {
            ParameterValue::String(s) => format!("'{}'", s.replace('\'', "''")),
            ParameterValue::Number(n) => n.to_string(),
            ParameterValue::Boolean(b) => {
                if *b {
                    "TRUE".to_string()
                } else {
                    "FALSE".to_string()
                }
            }
            ParameterValue::Array(_) => {
                panic!("ParameterValue::to_sql() does not support arrays. Arrays in annotation layers should be handled via VALUES clause generation.")
            }
            ParameterValue::Null => "NULL".to_string(),
        }
    }

    /// Convert a scalar ParameterValue to an ArrayElement.
    ///
    /// Panics if called on an Array variant.
    fn to_array_element(&self) -> ArrayElement {
        match self {
            ParameterValue::Number(num) => ArrayElement::Number(*num),
            ParameterValue::String(s) => ArrayElement::String(s.clone()),
            ParameterValue::Boolean(b) => ArrayElement::Boolean(*b),
            ParameterValue::Null => ArrayElement::Null,
            ParameterValue::Array(_) => panic!("Cannot convert Array to single ArrayElement"),
        }
    }

    /// Recycle this value to a target array length.
    ///
    /// - Scalars (String, Number, Boolean, Null) are converted to arrays with n copies
    /// - Arrays of length 1 are recycled to n copies of that element
    /// - Arrays of target length are returned as-is (after homogenization)
    /// - Arrays of other lengths produce an error
    pub fn rep(self, n: usize) -> Result<Self, crate::GgsqlError> {
        match self {
            // Arrays: homogenize types if mixed, then recycle if needed
            ParameterValue::Array(arr) => {
                if arr.len() == 1 {
                    // Recycle the single element
                    let element = arr[0].clone();
                    Ok(ParameterValue::Array(vec![element; n]))
                } else if arr.len() == n {
                    // Already correct length - homogenize for type consistency
                    let arr = ArrayElement::homogenize(&arr);
                    Ok(ParameterValue::Array(arr))
                } else {
                    // Mismatched length - shouldn't happen if validation passed
                    Err(crate::GgsqlError::InternalError(format!(
                        "Attempted to recycle array of length {} to length {} (should have been caught earlier)",
                        arr.len(),
                        n
                    )))
                }
            }
            // Scalars: convert to ArrayElement and replicate n times
            scalar => {
                let elem = scalar.to_array_element();
                Ok(ParameterValue::Array(vec![elem; n]))
            }
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

// =============================================================================
// SQL Type Names for Casting
// =============================================================================

/// Target type for casting operations.
///
/// When a column's data type doesn't match the scale's target type
/// (e.g., STRING column with a DATE transform, or Int column needing
/// to be discrete Boolean), the SQL query needs to cast values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastTargetType {
    /// Numeric type (DOUBLE, FLOAT, etc.)
    Number,
    /// Integer type (BIGINT, INTEGER)
    Integer,
    /// Date type (DATE)
    Date,
    /// DateTime/Timestamp type (TIMESTAMP)
    DateTime,
    /// Time type (TIME)
    Time,
    /// String type (VARCHAR)
    String,
    /// Boolean type (BOOLEAN)
    Boolean,
}

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

// =============================================================================
// Default Property Types (Shared by Coord, Scale, and Geom traits)
// =============================================================================

/// Default value for a property parameter
///
/// Used by traits to declare both allowed property names and their default values
/// in a single declaration, avoiding the need to keep two separate implementations
/// in sync.
#[derive(Debug, Clone)]
pub enum ParamDefinitionValue {
    String(&'static str),
    Number(f64),
    Boolean(bool),
    Null,
}

// =============================================================================
// Parameter Constraint Types
// =============================================================================

/// Constraint state for a specific type
#[derive(Debug, Clone, Copy)]
pub enum TypeConstraint<T> {
    /// Type not allowed for this parameter
    Forbidden,
    /// Type allowed, any value accepted
    Any,
    /// Type allowed with specific constraints
    Constrained(T),
}

/// Constraint for Number values
#[derive(Debug, Clone, Copy)]
pub struct NumberConstraint {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub min_exclusive: bool,
    pub max_exclusive: bool,
    pub whole: bool,
}

impl NumberConstraint {
    /// No constraints on value
    pub const fn unconstrained() -> Self {
        Self {
            min: None,
            max: None,
            min_exclusive: false,
            max_exclusive: false,
            whole: false,
        }
    }

    /// Minimum value (inclusive)
    pub const fn min(min: f64) -> Self {
        Self {
            min: Some(min),
            max: None,
            min_exclusive: false,
            max_exclusive: false,
            whole: false,
        }
    }

    /// Minimum value (exclusive)
    pub const fn min_exclusive(min: f64) -> Self {
        Self {
            min: Some(min),
            max: None,
            min_exclusive: true,
            max_exclusive: false,
            whole: false,
        }
    }

    /// Value within range (inclusive on both ends)
    pub const fn range(min: f64, max: f64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            min_exclusive: false,
            max_exclusive: false,
            whole: false,
        }
    }

    /// Whole number (integer) with minimum value (inclusive)
    pub const fn count(min: f64) -> Self {
        Self {
            min: Some(min),
            max: None,
            min_exclusive: false,
            max_exclusive: false,
            whole: true,
        }
    }

    /// Whole number (integer) within range (inclusive on both ends)
    pub const fn count_range(min: f64, max: f64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            min_exclusive: false,
            max_exclusive: false,
            whole: true,
        }
    }
}

/// Constraint for String values
#[derive(Debug, Clone, Copy)]
pub struct StringConstraint {
    /// String must be one of these values (empty = any string allowed)
    pub allowed_values: &'static [&'static str],
}

impl StringConstraint {
    /// Any string allowed (empty slice = no restriction)
    pub const fn unconstrained() -> Self {
        Self {
            allowed_values: &[],
        }
    }

    /// String must be one of the specified values
    pub const fn one_of(values: &'static [&'static str]) -> Self {
        Self {
            allowed_values: values,
        }
    }
}

/// Element constraint - specifies type AND value constraints for array elements
#[derive(Debug, Clone, Copy)]
pub enum ArrayElementConstraint {
    /// Any element type allowed, no constraints
    Any,
    /// Must be numbers, with optional value constraint
    Number(NumberConstraint),
    /// Must be strings, with optional value constraint
    String(StringConstraint),
    /// Must be booleans
    Boolean,
}

/// Constraint for Array values
#[derive(Debug, Clone, Copy)]
pub struct ArrayConstraint {
    pub element: ArrayElementConstraint,
    pub min_len: Option<usize>,
    pub max_len: Option<usize>,
    pub allow_null_elements: bool,
}

impl ArrayConstraint {
    /// Array of numbers with value constraint
    pub const fn of_numbers(constraint: NumberConstraint) -> Self {
        Self {
            element: ArrayElementConstraint::Number(constraint),
            min_len: None,
            max_len: None,
            allow_null_elements: false,
        }
    }

    /// Array of numbers with exact length and value constraint
    pub const fn of_numbers_len(constraint: NumberConstraint, len: usize) -> Self {
        Self {
            element: ArrayElementConstraint::Number(constraint),
            min_len: Some(len),
            max_len: Some(len),
            allow_null_elements: false,
        }
    }

    /// Array of strings with value constraint
    pub const fn of_strings(constraint: StringConstraint) -> Self {
        Self {
            element: ArrayElementConstraint::String(constraint),
            min_len: None,
            max_len: None,
            allow_null_elements: false,
        }
    }

    /// Array of strings with exact length and value constraint
    #[allow(dead_code)]
    pub const fn of_strings_len(constraint: StringConstraint, len: usize) -> Self {
        Self {
            element: ArrayElementConstraint::String(constraint),
            min_len: Some(len),
            max_len: Some(len),
            allow_null_elements: false,
        }
    }

    /// Any element types allowed (including nulls)
    #[allow(dead_code)]
    pub const fn any_elements() -> Self {
        Self {
            element: ArrayElementConstraint::Any,
            min_len: None,
            max_len: None,
            allow_null_elements: true,
        }
    }

    /// Builder method to allow null elements
    #[allow(dead_code)]
    pub const fn with_null_elements(mut self) -> Self {
        self.allow_null_elements = true;
        self
    }
}

/// Validation constraint for a parameter value.
///
/// Each field specifies whether a type is forbidden, allowed (any value), or constrained.
/// Used by `ParamDefinition` to specify validation rules for parameters.
#[derive(Debug, Clone, Copy)]
pub struct ParamConstraint {
    pub number: TypeConstraint<NumberConstraint>,
    pub string: TypeConstraint<StringConstraint>,
    pub boolean: TypeConstraint<()>,
    pub array: TypeConstraint<ArrayConstraint>,
    pub allow_null: bool,
}

impl ParamConstraint {
    /// All types allowed, no constraints (default)
    pub const fn unconstrained() -> Self {
        Self {
            number: TypeConstraint::Any,
            string: TypeConstraint::Any,
            boolean: TypeConstraint::Any,
            array: TypeConstraint::Any,
            allow_null: true,
        }
    }

    /// Number only with constraint
    pub const fn number(constraint: NumberConstraint) -> Self {
        Self {
            number: TypeConstraint::Constrained(constraint),
            string: TypeConstraint::Forbidden,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        }
    }

    /// Number only, any value
    #[allow(dead_code)]
    pub const fn number_any() -> Self {
        Self {
            number: TypeConstraint::Any,
            string: TypeConstraint::Forbidden,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        }
    }

    /// String only with enum constraint
    pub const fn string_option(values: &'static [&'static str]) -> Self {
        Self {
            number: TypeConstraint::Forbidden,
            string: TypeConstraint::Constrained(StringConstraint::one_of(values)),
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        }
    }

    /// String only
    pub const fn string() -> Self {
        Self {
            number: TypeConstraint::Forbidden,
            string: TypeConstraint::Any,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        }
    }

    /// Boolean only
    pub const fn boolean() -> Self {
        Self {
            number: TypeConstraint::Forbidden,
            string: TypeConstraint::Forbidden,
            boolean: TypeConstraint::Any,
            array: TypeConstraint::Forbidden,
            allow_null: true,
        }
    }

    /// Number or Array of numbers - for `expand` parameter
    pub const fn number_or_numeric_array(num: NumberConstraint, arr: ArrayConstraint) -> Self {
        Self {
            number: TypeConstraint::Constrained(num),
            string: TypeConstraint::Forbidden,
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Constrained(arr),
            allow_null: true,
        }
    }

    /// Number, Array of numbers, or String (any) - for `breaks` parameter
    pub const fn number_or_array_or_string(num: NumberConstraint, arr: ArrayConstraint) -> Self {
        Self {
            number: TypeConstraint::Constrained(num),
            string: TypeConstraint::Any, // Any string (temporal interval validated elsewhere)
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Constrained(arr),
            allow_null: true,
        }
    }

    /// String enum or Array of strings from same enum - for `free` parameter
    #[allow(dead_code)]
    pub const fn string_or_string_array(values: &'static [&'static str]) -> Self {
        Self {
            number: TypeConstraint::Forbidden,
            string: TypeConstraint::Constrained(StringConstraint::one_of(values)),
            boolean: TypeConstraint::Forbidden,
            array: TypeConstraint::Constrained(ArrayConstraint::of_strings(
                StringConstraint::one_of(values),
            )),
            allow_null: true,
        }
    }

    /// Builder method to disallow null values
    #[allow(dead_code)]
    pub const fn required(mut self) -> Self {
        self.allow_null = false;
        self
    }

    // =========================================================================
    // Convenience constructors
    // =========================================================================

    /// Create a constraint requiring a minimum value (inclusive)
    pub const fn number_min(min: f64) -> Self {
        Self::number(NumberConstraint::min(min))
    }

    /// Create a constraint requiring a minimum value (exclusive)
    pub const fn number_min_exclusive(min: f64) -> Self {
        Self::number(NumberConstraint::min_exclusive(min))
    }

    /// Create a constraint requiring a value within a range (inclusive on both ends)
    pub const fn number_range(min: f64, max: f64) -> Self {
        Self::number(NumberConstraint::range(min, max))
    }

    /// Create a constraint requiring a whole number (integer) with minimum value
    pub const fn count(min: f64) -> Self {
        Self::number(NumberConstraint::count(min))
    }

    /// Create a constraint requiring a whole number (integer) within a range
    pub const fn count_range(min: f64, max: f64) -> Self {
        Self::number(NumberConstraint::count_range(min, max))
    }
}

/// Validate a parameter value against a constraint.
///
/// Returns Ok(()) if the value passes validation, or Err with a descriptive
/// error message if validation fails.
pub fn validate_parameter(
    param_name: &str,
    value: &ParameterValue,
    constraint: &ParamConstraint,
) -> Result<(), String> {
    match value {
        ParameterValue::Number(n) => match &constraint.number {
            TypeConstraint::Forbidden => {
                Err(type_not_allowed_error(param_name, "Number", constraint))
            }
            TypeConstraint::Any => Ok(()),
            TypeConstraint::Constrained(c) => validate_number(param_name, *n, c),
        },
        ParameterValue::String(s) => match &constraint.string {
            TypeConstraint::Forbidden => {
                Err(type_not_allowed_error(param_name, "String", constraint))
            }
            TypeConstraint::Any => Ok(()),
            TypeConstraint::Constrained(c) => validate_string(param_name, s, c),
        },
        ParameterValue::Boolean(_) => match &constraint.boolean {
            TypeConstraint::Forbidden => {
                Err(type_not_allowed_error(param_name, "Boolean", constraint))
            }
            TypeConstraint::Any | TypeConstraint::Constrained(()) => Ok(()),
        },
        ParameterValue::Array(arr) => match &constraint.array {
            TypeConstraint::Forbidden => {
                Err(type_not_allowed_error(param_name, "Array", constraint))
            }
            TypeConstraint::Any => Ok(()),
            TypeConstraint::Constrained(c) => validate_array(param_name, arr, c),
        },
        ParameterValue::Null => {
            if constraint.allow_null {
                Ok(())
            } else {
                Err(format!(
                    "Parameter '{}' is required (cannot be null)",
                    param_name
                ))
            }
        }
    }
}

fn validate_number(name: &str, n: f64, c: &NumberConstraint) -> Result<(), String> {
    // Check whole number constraint first
    if c.whole && n.fract() != 0.0 {
        return Err(format!("'{}' should be a whole number, not {}", name, n));
    }
    if let Some(min) = c.min {
        let ok = if c.min_exclusive { n > min } else { n >= min };
        if !ok {
            let op = if c.min_exclusive { ">" } else { ">=" };
            return Err(format!("'{}' should be {} {}, not {}", name, op, min, n));
        }
    }
    if let Some(max) = c.max {
        let ok = if c.max_exclusive { n < max } else { n <= max };
        if !ok {
            let op = if c.max_exclusive { "<" } else { "<=" };
            return Err(format!("'{}' should be {} {}, not {}", name, op, max, n));
        }
    }
    Ok(())
}

fn validate_string(name: &str, s: &str, c: &StringConstraint) -> Result<(), String> {
    // Empty allowed_values = unconstrained (any string)
    if c.allowed_values.is_empty() {
        return Ok(());
    }
    if !c.allowed_values.contains(&s) {
        return Err(format!(
            "'{}' should be {}, not '{}'",
            name,
            crate::or_list_quoted(&c.allowed_values, '\''),
            s
        ));
    }
    Ok(())
}

fn validate_array(name: &str, arr: &[ArrayElement], c: &ArrayConstraint) -> Result<(), String> {
    // Length validation
    let len = arr.len();
    match (c.min_len, c.max_len) {
        // Exact length required
        (Some(min), Some(max)) if min == max && len != min => {
            return Err(format!(
                "Parameter '{}' array must have exactly {} element(s) (got {})",
                name, min, len
            ));
        }
        // Range constraint (both bounds, not equal)
        (Some(min), Some(max)) if len < min || len > max => {
            return Err(format!(
                "Parameter '{}' array must have between {} and {} element(s) (got {})",
                name, min, max, len
            ));
        }
        // One-sided constraints
        (Some(min), None) if len < min => {
            return Err(format!(
                "Parameter '{}' array must have at least {} element(s) (got {})",
                name, min, len
            ));
        }
        (None, Some(max)) if len > max => {
            return Err(format!(
                "Parameter '{}' array must have at most {} element(s) (got {})",
                name, max, len
            ));
        }
        _ => {}
    }
    // Element type and value validation - collect all errors
    let mut errors: Vec<String> = Vec::new();
    for (i, elem) in arr.iter().enumerate() {
        match (&c.element, elem) {
            // Handle null elements
            (_, ArrayElement::Null) if c.allow_null_elements => {}
            (_, ArrayElement::Null) => {
                errors.push(format!("'{}[{}]' cannot be null", name, i));
            }
            // Any element type allowed
            (ArrayElementConstraint::Any, _) => {}
            // Number elements
            (ArrayElementConstraint::Number(nc), ArrayElement::Number(n)) => {
                if let Err(e) = validate_number(&format!("{}[{}]", name, i), *n, nc) {
                    errors.push(e);
                }
            }
            (ArrayElementConstraint::Number(_), _) => {
                errors.push(format!("'{}[{}]' must be a number", name, i));
            }
            // String elements
            (ArrayElementConstraint::String(sc), ArrayElement::String(s)) => {
                // Only validate if allowed_values is non-empty (empty = any string)
                if !sc.allowed_values.is_empty() {
                    if let Err(e) = validate_string(&format!("{}[{}]", name, i), s, sc) {
                        errors.push(e);
                    }
                }
            }
            (ArrayElementConstraint::String(_), _) => {
                errors.push(format!("'{}[{}]' must be a string", name, i));
            }
            // Boolean elements
            (ArrayElementConstraint::Boolean, ArrayElement::Boolean(_)) => {}
            (ArrayElementConstraint::Boolean, _) => {
                errors.push(format!("'{}[{}]' must be a boolean", name, i));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Parameter '{}' has invalid elements:\n— {}",
            name,
            errors.join("\n— ")
        ))
    }
}

fn type_not_allowed_error(name: &str, got: &str, c: &ParamConstraint) -> String {
    let mut allowed = Vec::new();
    if !matches!(c.number, TypeConstraint::Forbidden) {
        allowed.push("Number");
    }
    if !matches!(c.string, TypeConstraint::Forbidden) {
        allowed.push("String");
    }
    if !matches!(c.boolean, TypeConstraint::Forbidden) {
        allowed.push("Boolean");
    }
    if !matches!(c.array, TypeConstraint::Forbidden) {
        allowed.push("Array");
    }
    format!(
        "'{}' should be {}, not {}",
        name,
        crate::or_list(&allowed),
        got
    )
}

/// Property definition: name, default value, and validation constraint.
///
/// Used by `CoordTrait`, `ScaleTypeTrait`, and `GeomTrait` to declare their
/// allowed properties and default values in a single place.
#[derive(Debug, Clone)]
pub struct ParamDefinition {
    pub name: &'static str,
    pub default: ParamDefinitionValue,
    pub constraint: ParamConstraint,
}

impl ParamDefinition {
    /// Convert the default value to a ParameterValue, if not Null
    pub fn to_parameter_value(&self) -> Option<ParameterValue> {
        match &self.default {
            ParamDefinitionValue::String(s) => Some(ParameterValue::String(s.to_string())),
            ParamDefinitionValue::Number(n) => Some(ParameterValue::Number(*n)),
            ParamDefinitionValue::Boolean(b) => Some(ParameterValue::Boolean(*b)),
            ParamDefinitionValue::Null => None,
        }
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
    fn test_datetime_from_string_with_z() {
        // UTC timezone indicator
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00Z").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
        assert_eq!(elem.to_key_string(), "2024-01-15T10:30:00");
    }

    #[test]
    fn test_datetime_from_string_with_positive_offset() {
        // +05:30 offset (e.g., India Standard Time)
        // 10:30 IST = 05:00 UTC
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00+05:30").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
        assert_eq!(elem.to_key_string(), "2024-01-15T05:00:00");
    }

    #[test]
    fn test_datetime_from_string_with_negative_offset() {
        // -08:00 offset (e.g., Pacific Standard Time)
        // 10:30 PST = 18:30 UTC
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00-08:00").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
        assert_eq!(elem.to_key_string(), "2024-01-15T18:30:00");
    }

    #[test]
    fn test_datetime_from_string_with_zero_offset() {
        // Explicit +00:00 (same as Z)
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00+00:00").unwrap();
        assert!(matches!(elem, ArrayElement::DateTime(_)));
        assert_eq!(elem.to_key_string(), "2024-01-15T10:30:00");
    }

    #[test]
    fn test_datetime_from_string_with_fractional_and_tz() {
        // Fractional seconds with timezone
        let elem = ArrayElement::from_datetime_string("2024-01-15T10:30:00.123Z").unwrap();
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

    // =============================================================================
    // ArrayElementType tests
    // =============================================================================

    #[test]
    fn test_element_type() {
        assert_eq!(
            ArrayElement::String("hello".to_string()).element_type(),
            Some(ArrayElementType::String)
        );
        assert_eq!(
            ArrayElement::Number(42.0).element_type(),
            Some(ArrayElementType::Number)
        );
        assert_eq!(
            ArrayElement::Boolean(true).element_type(),
            Some(ArrayElementType::Boolean)
        );
        assert_eq!(
            ArrayElement::Date(100).element_type(),
            Some(ArrayElementType::Date)
        );
        assert_eq!(
            ArrayElement::DateTime(1000000).element_type(),
            Some(ArrayElementType::DateTime)
        );
        assert_eq!(
            ArrayElement::Time(1000000000).element_type(),
            Some(ArrayElementType::Time)
        );
        assert_eq!(ArrayElement::Null.element_type(), None);
    }

    #[test]
    fn test_infer_type_boolean() {
        let values = vec![ArrayElement::Boolean(true), ArrayElement::Boolean(false)];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Boolean)
        );
    }

    #[test]
    fn test_infer_type_number() {
        let values = vec![ArrayElement::Number(1.0), ArrayElement::Number(2.0)];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Number)
        );
    }

    #[test]
    fn test_infer_type_string() {
        let values = vec![
            ArrayElement::String("a".to_string()),
            ArrayElement::String("b".to_string()),
        ];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::String)
        );
    }

    #[test]
    fn test_infer_type_date() {
        let values = vec![ArrayElement::Date(100), ArrayElement::Date(200)];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Date)
        );
    }

    #[test]
    fn test_infer_type_with_nulls() {
        let values = vec![
            ArrayElement::Null,
            ArrayElement::Boolean(true),
            ArrayElement::Null,
        ];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Boolean)
        );
    }

    #[test]
    fn test_infer_type_all_nulls() {
        let values = vec![ArrayElement::Null, ArrayElement::Null];
        assert_eq!(ArrayElement::infer_type(&values), None);
    }

    #[test]
    fn test_infer_type_empty() {
        let values: Vec<ArrayElement> = vec![];
        assert_eq!(ArrayElement::infer_type(&values), None);
    }

    #[test]
    fn test_infer_type_priority_boolean_over_string() {
        // If there are mixed types, Boolean has priority over String
        let values = vec![
            ArrayElement::Boolean(true),
            ArrayElement::String("hello".to_string()),
        ];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Boolean)
        );
    }

    #[test]
    fn test_infer_type_priority_number_over_string() {
        let values = vec![
            ArrayElement::Number(42.0),
            ArrayElement::String("hello".to_string()),
        ];
        assert_eq!(
            ArrayElement::infer_type(&values),
            Some(ArrayElementType::Number)
        );
    }

    // =============================================================================
    // coerce_to tests
    // =============================================================================

    #[test]
    fn test_coerce_string_to_boolean_true() {
        let elem = ArrayElement::String("true".to_string());
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(true));

        // Also test case insensitivity
        let elem = ArrayElement::String("TRUE".to_string());
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(true));
    }

    #[test]
    fn test_coerce_string_to_boolean_false() {
        let elem = ArrayElement::String("false".to_string());
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(false));

        let elem = ArrayElement::String("no".to_string());
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(false));
    }

    #[test]
    fn test_coerce_string_to_boolean_error() {
        let elem = ArrayElement::String("maybe".to_string());
        let result = elem.coerce_to(ArrayElementType::Boolean);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Cannot coerce string 'maybe' to boolean"));
    }

    #[test]
    fn test_coerce_string_to_number() {
        let elem = ArrayElement::String("42.5".to_string());
        let result = elem.coerce_to(ArrayElementType::Number).unwrap();
        assert_eq!(result, ArrayElement::Number(42.5));
    }

    #[test]
    fn test_coerce_string_to_number_error() {
        let elem = ArrayElement::String("not a number".to_string());
        let result = elem.coerce_to(ArrayElementType::Number);
        assert!(result.is_err());
    }

    #[test]
    fn test_coerce_string_to_date() {
        let elem = ArrayElement::String("2024-01-15".to_string());
        let result = elem.coerce_to(ArrayElementType::Date).unwrap();
        assert!(matches!(result, ArrayElement::Date(_)));
        assert_eq!(result.to_key_string(), "2024-01-15");
    }

    #[test]
    fn test_coerce_string_to_date_error() {
        let elem = ArrayElement::String("not-a-date".to_string());
        let result = elem.coerce_to(ArrayElementType::Date);
        assert!(result.is_err());
    }

    #[test]
    fn test_coerce_number_to_boolean() {
        let elem = ArrayElement::Number(1.0);
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(true));

        let elem = ArrayElement::Number(0.0);
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(false));
    }

    #[test]
    fn test_coerce_number_to_string() {
        let elem = ArrayElement::Number(42.5);
        let result = elem.coerce_to(ArrayElementType::String).unwrap();
        assert_eq!(result, ArrayElement::String("42.5".to_string()));

        // Integer format
        let elem = ArrayElement::Number(42.0);
        let result = elem.coerce_to(ArrayElementType::String).unwrap();
        assert_eq!(result, ArrayElement::String("42".to_string()));
    }

    #[test]
    fn test_coerce_boolean_to_number() {
        let elem = ArrayElement::Boolean(true);
        let result = elem.coerce_to(ArrayElementType::Number).unwrap();
        assert_eq!(result, ArrayElement::Number(1.0));

        let elem = ArrayElement::Boolean(false);
        let result = elem.coerce_to(ArrayElementType::Number).unwrap();
        assert_eq!(result, ArrayElement::Number(0.0));
    }

    #[test]
    fn test_coerce_boolean_to_string() {
        let elem = ArrayElement::Boolean(true);
        let result = elem.coerce_to(ArrayElementType::String).unwrap();
        assert_eq!(result, ArrayElement::String("true".to_string()));
    }

    #[test]
    fn test_coerce_null_stays_null() {
        let elem = ArrayElement::Null;
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Null);

        let result = elem.coerce_to(ArrayElementType::Number).unwrap();
        assert_eq!(result, ArrayElement::Null);
    }

    #[test]
    fn test_coerce_same_type_identity() {
        let elem = ArrayElement::Boolean(true);
        let result = elem.coerce_to(ArrayElementType::Boolean).unwrap();
        assert_eq!(result, ArrayElement::Boolean(true));

        let elem = ArrayElement::Number(42.0);
        let result = elem.coerce_to(ArrayElementType::Number).unwrap();
        assert_eq!(result, ArrayElement::Number(42.0));
    }

    #[test]
    fn test_coerce_date_to_string() {
        let elem = ArrayElement::from_date_string("2024-01-15").unwrap();
        let result = elem.coerce_to(ArrayElementType::String).unwrap();
        assert_eq!(result, ArrayElement::String("2024-01-15".to_string()));
    }

    #[test]
    fn test_coerce_cross_temporal_not_supported() {
        let elem = ArrayElement::Date(100);
        let result = elem.coerce_to(ArrayElementType::DateTime);
        assert!(result.is_err());

        let elem = ArrayElement::DateTime(100000);
        let result = elem.coerce_to(ArrayElementType::Date);
        assert!(result.is_err());
    }

    // =========================================================================
    // ParamConstraint validation tests
    // =========================================================================

    #[test]
    fn test_number_constraint_accepts_valid() {
        let constraint = ParamConstraint::number_min(0.0);
        assert!(validate_parameter("test", &ParameterValue::Number(5.0), &constraint).is_ok());
        assert!(validate_parameter("test", &ParameterValue::Number(0.0), &constraint).is_ok());
    }

    #[test]
    fn test_number_constraint_rejects_invalid() {
        let constraint = ParamConstraint::number_min(0.0);
        let result = validate_parameter("test", &ParameterValue::Number(-1.0), &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(">= 0"));
    }

    #[test]
    fn test_number_constraint_rejects_wrong_type() {
        let constraint = ParamConstraint::number_min(0.0);
        let result = validate_parameter(
            "test",
            &ParameterValue::String("hello".to_string()),
            &constraint,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("should be Number"));
    }

    #[test]
    fn test_count_constraint_accepts_whole() {
        let constraint = ParamConstraint::count(1.0);
        assert!(validate_parameter("bins", &ParameterValue::Number(5.0), &constraint).is_ok());
        assert!(validate_parameter("bins", &ParameterValue::Number(1.0), &constraint).is_ok());
        assert!(validate_parameter("bins", &ParameterValue::Number(100.0), &constraint).is_ok());
    }

    #[test]
    fn test_count_constraint_rejects_fractional() {
        let constraint = ParamConstraint::count(1.0);
        let result = validate_parameter("bins", &ParameterValue::Number(5.5), &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("whole number"));
    }

    #[test]
    fn test_count_constraint_rejects_below_min() {
        let constraint = ParamConstraint::count(1.0);
        let result = validate_parameter("bins", &ParameterValue::Number(0.0), &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(">= 1"));
    }

    #[test]
    fn test_string_option_accepts_valid() {
        let constraint = ParamConstraint::string_option(&["a", "b", "c"]);
        assert!(validate_parameter(
            "test",
            &ParameterValue::String("a".to_string()),
            &constraint
        )
        .is_ok());
    }

    #[test]
    fn test_string_option_rejects_invalid() {
        let constraint = ParamConstraint::string_option(&["a", "b", "c"]);
        let result = validate_parameter(
            "test",
            &ParameterValue::String("d".to_string()),
            &constraint,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not 'd'"));
    }

    #[test]
    fn test_string_option_rejects_wrong_type() {
        let constraint = ParamConstraint::string_option(&["a", "b", "c"]);
        let result = validate_parameter("test", &ParameterValue::Number(1.0), &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("should be String"));
    }

    #[test]
    fn test_boolean_accepts_valid() {
        let constraint = ParamConstraint::boolean();
        assert!(validate_parameter("reverse", &ParameterValue::Boolean(true), &constraint).is_ok());
        assert!(
            validate_parameter("reverse", &ParameterValue::Boolean(false), &constraint).is_ok()
        );
    }

    #[test]
    fn test_boolean_rejects_wrong_type() {
        let constraint = ParamConstraint::boolean();
        let result = validate_parameter(
            "reverse",
            &ParameterValue::String("true".to_string()),
            &constraint,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("should be Boolean"));
    }

    #[test]
    fn test_multi_type_number_or_array_accepts_number() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers_len(NumberConstraint::unconstrained(), 2),
        );
        assert!(validate_parameter("expand", &ParameterValue::Number(0.05), &constraint).is_ok());
    }

    #[test]
    fn test_multi_type_number_or_array_accepts_array() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers_len(NumberConstraint::min(0.0), 2),
        );
        let arr =
            ParameterValue::Array(vec![ArrayElement::Number(0.05), ArrayElement::Number(10.0)]);
        assert!(validate_parameter("expand", &arr, &constraint).is_ok());
    }

    #[test]
    fn test_multi_type_number_or_array_rejects_wrong_array_length() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers_len(NumberConstraint::min(0.0), 2),
        );
        let arr = ParameterValue::Array(vec![ArrayElement::Number(0.05)]);
        let result = validate_parameter("expand", &arr, &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 2"));
    }

    #[test]
    fn test_multi_type_number_or_array_rejects_string() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers_len(NumberConstraint::min(0.0), 2),
        );
        let result = validate_parameter(
            "expand",
            &ParameterValue::String("0.05".to_string()),
            &constraint,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("should be Number or Array"));
    }

    #[test]
    fn test_multi_type_number_or_array_validates_element_values() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers_len(NumberConstraint::min(0.0), 2),
        );
        let arr = ParameterValue::Array(vec![
            ArrayElement::Number(0.05),
            ArrayElement::Number(-10.0), // Invalid: negative
        ]);
        let result = validate_parameter("expand", &arr, &constraint);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("expand[1]"));
        assert!(err.contains(">= 0"));
    }

    #[test]
    fn test_breaks_constraint_accepts_number() {
        let constraint = ParamConstraint::number_or_array_or_string(
            NumberConstraint::min(1.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        assert!(validate_parameter("breaks", &ParameterValue::Number(10.0), &constraint).is_ok());
    }

    #[test]
    fn test_breaks_constraint_accepts_array() {
        let constraint = ParamConstraint::number_or_array_or_string(
            NumberConstraint::min(1.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        let arr = ParameterValue::Array(vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
        ]);
        assert!(validate_parameter("breaks", &arr, &constraint).is_ok());
    }

    #[test]
    fn test_breaks_constraint_accepts_string() {
        let constraint = ParamConstraint::number_or_array_or_string(
            NumberConstraint::min(1.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        assert!(validate_parameter(
            "breaks",
            &ParameterValue::String("1 month".to_string()),
            &constraint
        )
        .is_ok());
    }

    #[test]
    fn test_breaks_constraint_rejects_boolean() {
        let constraint = ParamConstraint::number_or_array_or_string(
            NumberConstraint::min(1.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        let result = validate_parameter("breaks", &ParameterValue::Boolean(true), &constraint);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("should be Number, String, or Array"));
    }

    #[test]
    fn test_breaks_constraint_validates_number_value() {
        let constraint = ParamConstraint::number_or_array_or_string(
            NumberConstraint::min(1.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        let result = validate_parameter("breaks", &ParameterValue::Number(0.0), &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(">= 1"));
    }

    #[test]
    fn test_array_element_type_validation() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        let arr = ParameterValue::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::String("fifty".to_string()),
            ArrayElement::Number(100.0),
        ]);
        let result = validate_parameter("breaks", &arr, &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'breaks[1]' must be a number"));
    }

    #[test]
    fn test_null_allowed_by_default() {
        let constraint = ParamConstraint::number_min(1.0);
        assert!(validate_parameter("test", &ParameterValue::Null, &constraint).is_ok());
    }

    #[test]
    fn test_null_rejected_when_required() {
        let constraint = ParamConstraint::number_min(1.0).required();
        let result = validate_parameter("test", &ParameterValue::Null, &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("required"));
    }

    #[test]
    fn test_array_null_elements_rejected_by_default() {
        let constraint = ParamConstraint::number_or_numeric_array(
            NumberConstraint::min(0.0),
            ArrayConstraint::of_numbers(NumberConstraint::unconstrained()),
        );
        let arr = ParameterValue::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Null,
            ArrayElement::Number(3.0),
        ]);
        let result = validate_parameter("values", &arr, &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'values[1]' cannot be null"));
    }

    #[test]
    fn test_string_or_string_array_accepts_string() {
        let constraint = ParamConstraint::string_or_string_array(&["x", "y"]);
        assert!(validate_parameter(
            "free",
            &ParameterValue::String("x".to_string()),
            &constraint
        )
        .is_ok());
    }

    #[test]
    fn test_string_or_string_array_accepts_array() {
        let constraint = ParamConstraint::string_or_string_array(&["x", "y"]);
        let arr = ParameterValue::Array(vec![
            ArrayElement::String("x".to_string()),
            ArrayElement::String("y".to_string()),
        ]);
        assert!(validate_parameter("free", &arr, &constraint).is_ok());
    }

    #[test]
    fn test_string_or_string_array_validates_array_elements() {
        let constraint = ParamConstraint::string_or_string_array(&["x", "y"]);
        let arr = ParameterValue::Array(vec![
            ArrayElement::String("x".to_string()),
            ArrayElement::String("z".to_string()),
        ]);
        let result = validate_parameter("free", &arr, &constraint);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not 'z'"));
    }
    #[test]
    fn test_homogenize_mixed_number_string() {
        let arr = vec![
            ArrayElement::Number(1.0),
            ArrayElement::String("foo".to_string()),
        ];

        let homogenized = ArrayElement::homogenize(&arr);

        // Should fall back to String type since "foo" can't be coerced to Number
        assert_eq!(homogenized.len(), 2);
        assert!(matches!(homogenized[0], ArrayElement::String(_)));
        assert!(matches!(homogenized[1], ArrayElement::String(_)));

        if let ArrayElement::String(s) = &homogenized[0] {
            assert_eq!(s, "1");
        }
        if let ArrayElement::String(s) = &homogenized[1] {
            assert_eq!(s, "foo");
        }
    }

    #[test]
    fn test_try_as_temporal_date() {
        let elem = ArrayElement::String("1973-06-01".to_string());
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::Date(_)));
        assert_eq!(parsed.to_key_string(), "1973-06-01");
    }

    #[test]
    fn test_try_as_temporal_datetime() {
        let elem = ArrayElement::String("2024-03-17T14:30:00".to_string());
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::DateTime(_)));
    }

    #[test]
    fn test_try_as_temporal_time() {
        let elem = ArrayElement::String("14:30:00".to_string());
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::Time(_)));
    }

    #[test]
    fn test_try_as_temporal_non_temporal_string() {
        let elem = ArrayElement::String("not a date".to_string());
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::String(_)));
        assert_eq!(parsed.to_key_string(), "not a date");
    }

    #[test]
    fn test_try_as_temporal_non_string() {
        // Non-string elements should pass through unchanged
        let elem = ArrayElement::Number(42.0);
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::Number(_)));

        let elem = ArrayElement::Boolean(true);
        let parsed = elem.try_as_temporal();
        assert!(matches!(parsed, ArrayElement::Boolean(_)));
    }
}
