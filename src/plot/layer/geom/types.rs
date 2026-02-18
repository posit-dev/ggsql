//! Core types for the geom trait system
//!
//! These types are used by all geom implementations and are shared across the module.

use crate::Mappings;

/// Aesthetic information for a geom type
///
/// This struct describes which aesthetics a geom supports, requires, and hides.
#[derive(Debug, Clone, Copy)]
pub struct GeomAesthetics {
    /// All aesthetics this geom type supports for user MAPPING
    pub supported: &'static [&'static str],
    /// Aesthetics required for this geom type to be valid
    pub required: &'static [&'static str],
    /// Hidden aesthetics (valid REMAPPING targets, not valid MAPPING targets)
    /// These are produced by stat transforms but shouldn't be manually mapped
    pub hidden: &'static [&'static str],
}

/// Default value for a layer parameter
#[derive(Debug, Clone)]
pub enum DefaultParamValue {
    String(&'static str),
    Number(f64),
    Boolean(bool),
    Null,
}

/// Layer parameter definition: name and default value
#[derive(Debug, Clone)]
pub struct DefaultParam {
    pub name: &'static str,
    pub default: DefaultParamValue,
}

/// Result of a statistical transformation
///
/// Stat transforms like histogram and bar count produce new columns with computed values.
/// This enum captures both the transformed query and the mappings from aesthetics to the
/// new column names.
#[derive(Debug, Clone, PartialEq)]
pub enum StatResult {
    /// No transformation needed - use original data as-is
    Identity,
    /// Transformation applied, with stat-computed columns
    Transformed {
        /// The transformed SQL query that produces the stat-computed columns
        query: String,
        /// Names of stat-computed columns (e.g., ["count", "bin", "x"])
        /// These are semantic names that will be prefixed with __ggsql_stat__
        /// and mapped to aesthetics via default_remappings or REMAPPING clause
        stat_columns: Vec<String>,
        /// Names of stat columns that are dummy/placeholder values
        /// (e.g., "x" when bar chart has no x mapped - produces a constant value)
        dummy_columns: Vec<String>,
        /// Names of aesthetics consumed by this stat transform
        /// These aesthetics were used as input to the stat and should be removed
        /// from the layer mappings after the transform completes
        consumed_aesthetics: Vec<String>,
    },
}

pub use crate::plot::types::ColumnInfo;
/// Schema of a data source - list of columns with type info
pub use crate::plot::types::Schema;

/// Helper to extract column name from aesthetic value
pub fn get_column_name(aesthetics: &Mappings, aesthetic: &str) -> Option<String> {
    use crate::AestheticValue;
    aesthetics.get(aesthetic).and_then(|v| match v {
        AestheticValue::Column { name, .. } => Some(name.clone()),
        _ => None,
    })
}
