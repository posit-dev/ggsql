//! Input types for ggsql specification
//!
//! This module defines types that model user input: mappings, data sources,
//! settings, and values. These are the building blocks used in AST types
//! to capture what the user specified in their query.

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
    /// Whether this column is discrete (suitable for grouping)
    /// Discrete: String, Boolean, Categorical, Date
    /// Continuous: numeric types, Datetime, Time
    pub is_discrete: bool,
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
}

impl ArrayElement {
    /// Convert to a serde_json::Value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ArrayElement::String(s) => serde_json::Value::String(s.clone()),
            ArrayElement::Number(n) => serde_json::json!(n),
            ArrayElement::Boolean(b) => serde_json::Value::Bool(*b),
            ArrayElement::Null => serde_json::Value::Null,
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
