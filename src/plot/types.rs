//! Input types for ggsql specification
//!
//! This module defines types that model user input: mappings, data sources,
//! settings, and values. These are the building blocks used in AST types
//! to capture what the user specified in their query.

pub use crate::params::*;

use arrow::datatypes::DataType;
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

    /// Insert a standard column mapping using the internal naming convention.
    pub fn insert_column(&mut self, aesthetic: &str, column: &str) {
        self.insert(
            aesthetic,
            AestheticValue::standard_column(crate::naming::aesthetic_column(column)),
        );
    }

    /// Return the internal column names for all mapped aesthetics.
    pub fn column_names(&self) -> Vec<String> {
        self.aesthetics
            .keys()
            .map(|k| crate::naming::aesthetic_column(k))
            .collect()
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
    /// Uses the provided AestheticContext to map user-facing position aesthetic names
    /// (e.g., "x", "y", "angle", "radius") to internal names (e.g., "pos1", "pos2").
    /// Material aesthetics (e.g., "color", "size") are left unchanged.
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
    /// Annotation column for material aesthetics (synthesized from PLACE literals)
    /// These columns are generated from user-specified literal values in visual space
    /// (e.g., color => 'red', size => 10) and use identity scales (no transformation).
    /// Position annotations (x, y) use Column instead since they're in data coordinate space.
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
/// Similar to how `DefaultParamValue` is the static version of `ParameterValue`,
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
    /// Optional position aesthetic that, when the user leaves the whole
    /// axis family unmapped, is filled with a synthetic dummy categorical
    /// column by the default `apply_stat_transform`. The writer hides the
    /// resulting one-tick axis. Use only on `pos1`/`pos2`.
    ///
    /// The `Geom` wrapper auto-augments `default_remappings()` and
    /// `valid_stat_columns()` with the appropriate entries, so a geom that
    /// declares this variant doesn't need to spell those out.
    Dummy,
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
            Self::Column(_) | Self::Null | Self::Required | Self::Delayed | Self::Dummy => {
                ParameterValue::Null
            }
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
