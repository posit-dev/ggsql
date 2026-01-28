//! Centralized naming conventions for ggsql-generated identifiers.
//!
//! All synthetic column names, table names, and keys use double-underscore
//! prefix/suffix pattern to avoid collision with user-defined names.
//!
//! # Categories
//!
//! - **CTE tables**: Temp tables created from WITH clause CTEs (`__ggsql_cte_<name>_<uuid>__`)
//! - **Constant columns**: Synthetic columns for literal values (`__ggsql_const_<aesthetic>__`)
//! - **Stat columns**: Columns produced by statistical transforms (`__ggsql_stat__<name>`)
//! - **Data keys**: Keys for data sources in the data map (`__ggsql_global__`, `__ggsql_layer_<idx>__`)
//! - **Ordering column**: Window function for preserving data order (`__ggsql_order__`)
//! - **Session ID**: Process-wide UUID for temp table uniqueness

use const_format::concatcp;
use std::sync::LazyLock;
use uuid::Uuid;

// ============================================================================
// Base Building Blocks
// ============================================================================

/// Base prefix for all ggsql SQL-level identifiers
const GGSQL_PREFIX: &str = "__ggsql_";

/// Suffix for all ggsql identifiers (double underscore)
const GGSQL_SUFFIX: &str = "__";

// ============================================================================
// Session ID (Process-wide UUID for temp table uniqueness)
// ============================================================================

/// Process-wide session ID, generated once on first access.
/// Ensures temp table names are unique per process, avoiding collisions
/// when multiple processes use the same database connection.
static SESSION_ID: LazyLock<String> = LazyLock::new(|| Uuid::new_v4().simple().to_string());

/// Get the current session ID (32 hex characters, no dashes).
///
/// This ID is generated once per process and remains constant for the
/// lifetime of the process. Different processes will have different IDs.
///
/// # Example
/// ```
/// use ggsql::naming;
/// let id = naming::session_id();
/// assert_eq!(id.len(), 32); // UUID v4 simple format
/// ```
pub fn session_id() -> &'static str {
    &SESSION_ID
}

// ============================================================================
// Derived Constants
// ============================================================================

/// Full prefix for constant columns: `__ggsql_const_`
const CONST_PREFIX: &str = concatcp!(GGSQL_PREFIX, "const_");

/// Full prefix for stat columns: `__ggsql_stat__`
const STAT_PREFIX: &str = concatcp!(GGSQL_PREFIX, "stat_");

/// Full prefix for CTE tables: `__ggsql_cte_`
const CTE_PREFIX: &str = concatcp!(GGSQL_PREFIX, "cte_");

/// Full prefix for CTE tables: `__ggsql_cte_`
const LAYER_PREFIX: &str = concatcp!(GGSQL_PREFIX, "layer_");

/// Key for global data in the layer data HashMap.
/// Used as the key in PreparedData.data to store global data that applies to all layers.
/// This is NOT a SQL table name - use `global_table()` for SQL statements.
pub const GLOBAL_DATA_KEY: &str = concatcp!(GGSQL_PREFIX, "global", GGSQL_SUFFIX);

/// Column name for row ordering in Vega-Lite (used by Path geom)
pub const ORDER_COLUMN: &str = concatcp!(GGSQL_PREFIX, "order", GGSQL_SUFFIX);

/// Alias for schema extraction queries
pub const SCHEMA_ALIAS: &str = concatcp!(GGSQL_SUFFIX, "schema", GGSQL_SUFFIX);

// ============================================================================
// Constructor Functions
// ============================================================================

/// Generate SQL temp table name for global data.
///
/// Includes the session UUID to avoid collisions when multiple processes
/// use the same database connection. Format: `__ggsql_global_<uuid>__`
///
/// # Example
/// ```
/// use ggsql::naming;
/// let table = naming::global_table();
/// assert!(table.starts_with("__ggsql_global_"));
/// assert!(table.ends_with("__"));
/// // Contains 32-character UUID
/// assert_eq!(table.len(), "__ggsql_global_".len() + 32 + "__".len());
/// ```
pub fn global_table() -> String {
    format!("{}global_{}{}", GGSQL_PREFIX, session_id(), GGSQL_SUFFIX)
}

/// Generate temp table name for a materialized CTE.
///
/// Includes the session UUID to avoid collisions when multiple processes
/// use the same database connection. Format: `__ggsql_cte_<name>_<uuid>__`
///
/// # Example
/// ```
/// use ggsql::naming;
/// let table = naming::cte_table("sales");
/// assert!(table.starts_with("__ggsql_cte_sales_"));
/// assert!(table.ends_with("__"));
/// ```
pub fn cte_table(cte_name: &str) -> String {
    format!(
        "{}{}_{}{}",
        CTE_PREFIX,
        cte_name,
        session_id(),
        GGSQL_SUFFIX
    )
}

/// Generate column name for a constant aesthetic value.
///
/// Used when a single layer has a literal aesthetic value that needs
/// to be converted to a column for Vega-Lite encoding.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::const_column("color"), "__ggsql_const_color__");
/// ```
pub fn const_column(aesthetic: &str) -> String {
    format!("{}{}{}", CONST_PREFIX, aesthetic, GGSQL_SUFFIX)
}

/// Generate indexed column name for constant aesthetic (multi-layer).
///
/// Used when injecting constants into global data so different layers
/// can have different values for the same aesthetic.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::const_column_indexed("color", 0), "__ggsql_const_color_0__");
/// assert_eq!(naming::const_column_indexed("color", 1), "__ggsql_const_color_1__");
/// ```
pub fn const_column_indexed(aesthetic: &str, layer_idx: usize) -> String {
    format!(
        "{}{}_{}{}",
        CONST_PREFIX, aesthetic, layer_idx, GGSQL_SUFFIX
    )
}

/// Generate column name for statistical transform output.
///
/// These columns are produced by stat transforms like histogram and bar.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::stat_column("count"), "__ggsql_stat_count");
/// assert_eq!(naming::stat_column("bin"), "__ggsql_stat_bin");
/// ```
pub fn stat_column(stat_name: &str) -> String {
    format!("{}{}", STAT_PREFIX, stat_name)
}

/// Generate dataset key for layer-specific data.
///
/// Used when a layer has its own data source (FROM clause, filter, etc.)
/// that differs from the global data.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::layer_key(0), "__ggsql_layer_0__");
/// assert_eq!(naming::layer_key(2), "__ggsql_layer_2__");
/// ```
pub fn layer_key(layer_idx: usize) -> String {
    format!("{}{}{}", LAYER_PREFIX, layer_idx, GGSQL_SUFFIX)
}

// ============================================================================
// Detection Functions
// ============================================================================

/// Check if a column name is a synthetic constant column.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert!(naming::is_const_column("__ggsql_const_color__"));
/// assert!(naming::is_const_column("__ggsql_const_color_0__"));
/// assert!(!naming::is_const_column("color"));
/// ```
pub fn is_const_column(name: &str) -> bool {
    name.starts_with(CONST_PREFIX)
}

/// Check if a column name is a statistical transform column.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert!(naming::is_stat_column("__ggsql_stat_count"));
/// assert!(naming::is_stat_column("__ggsql_stat_bin"));
/// assert!(!naming::is_stat_column("count"));
/// ```
pub fn is_stat_column(name: &str) -> bool {
    name.starts_with(STAT_PREFIX)
}

/// Check if a column name is any synthetic ggsql column.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert!(naming::is_synthetic_column("__ggsql_const_color__"));
/// assert!(naming::is_synthetic_column("__ggsql_stat_count"));
/// assert!(!naming::is_synthetic_column("revenue"));
/// ```
pub fn is_synthetic_column(name: &str) -> bool {
    is_const_column(name) || is_stat_column(name)
}

/// Generate bin end column name for a binned column.
///
/// Used by the Vega-Lite writer to store the upper bound of a bin
/// when using `bin: "binned"` encoding with x2/y2 channels.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::bin_end_column("temperature"), "__ggsql_bin_end_temperature__");
/// assert_eq!(naming::bin_end_column("x"), "__ggsql_bin_end_x__");
/// ```
pub fn bin_end_column(column: &str) -> String {
    format!("{}bin_end_{}{}", GGSQL_PREFIX, column, GGSQL_SUFFIX)
}

/// Extract the stat name from a stat column (for display purposes).
///
/// Returns the human-readable name from a stat column name.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::extract_stat_name("__ggsql_stat_count"), Some("count"));
/// assert_eq!(naming::extract_stat_name("__ggsql_stat_bin"), Some("bin"));
/// assert_eq!(naming::extract_stat_name("regular_column"), None);
/// ```
pub fn extract_stat_name(name: &str) -> Option<&str> {
    name.strip_prefix(STAT_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id() {
        let id = session_id();
        // UUID v4 simple format is 32 hex characters
        assert_eq!(id.len(), 32);
        // Should be consistent across calls
        assert_eq!(session_id(), id);
    }

    #[test]
    fn test_global_table() {
        let table = global_table();
        assert!(table.starts_with("__ggsql_global_"));
        assert!(table.ends_with("__"));
        // Should be consistent across calls (same session)
        assert_eq!(global_table(), table);
        // Should contain session ID
        assert!(table.contains(session_id()));
    }

    #[test]
    fn test_cte_table() {
        let table = cte_table("sales");
        assert!(table.starts_with("__ggsql_cte_sales_"));
        assert!(table.ends_with("__"));
        // Should contain session ID
        assert!(table.contains(session_id()));

        let table2 = cte_table("monthly_totals");
        assert!(table2.starts_with("__ggsql_cte_monthly_totals_"));
        assert!(table2.ends_with("__"));
    }

    #[test]
    fn test_const_column() {
        assert_eq!(const_column("color"), "__ggsql_const_color__");
        assert_eq!(const_column("fill"), "__ggsql_const_fill__");
    }

    #[test]
    fn test_const_column_indexed() {
        assert_eq!(const_column_indexed("color", 0), "__ggsql_const_color_0__");
        assert_eq!(const_column_indexed("color", 1), "__ggsql_const_color_1__");
        assert_eq!(const_column_indexed("size", 5), "__ggsql_const_size_5__");
    }

    #[test]
    fn test_stat_column() {
        assert_eq!(stat_column("count"), "__ggsql_stat_count");
        assert_eq!(stat_column("bin"), "__ggsql_stat_bin");
        assert_eq!(stat_column("density"), "__ggsql_stat_density");
    }

    #[test]
    fn test_layer_key() {
        assert_eq!(layer_key(0), "__ggsql_layer_0__");
        assert_eq!(layer_key(1), "__ggsql_layer_1__");
        assert_eq!(layer_key(10), "__ggsql_layer_10__");
    }

    #[test]
    fn test_is_const_column() {
        assert!(is_const_column("__ggsql_const_color__"));
        assert!(is_const_column("__ggsql_const_color_0__"));
        assert!(is_const_column("__ggsql_const_fill__"));
        assert!(!is_const_column("color"));
        assert!(!is_const_column("__ggsql_stat_count"));
    }

    #[test]
    fn test_is_stat_column() {
        assert!(is_stat_column("__ggsql_stat_count"));
        assert!(is_stat_column("__ggsql_stat_bin"));
        assert!(!is_stat_column("count"));
        assert!(!is_stat_column("__ggsql_const_color__"));
    }

    #[test]
    fn test_is_synthetic_column() {
        assert!(is_synthetic_column("__ggsql_const_color__"));
        assert!(is_synthetic_column("__ggsql_stat_count"));
        assert!(!is_synthetic_column("revenue"));
        assert!(!is_synthetic_column("date"));
    }

    #[test]
    fn test_extract_stat_name() {
        assert_eq!(extract_stat_name("__ggsql_stat_count"), Some("count"));
        assert_eq!(extract_stat_name("__ggsql_stat_bin"), Some("bin"));
        assert_eq!(extract_stat_name("__ggsql_stat_density"), Some("density"));
        assert_eq!(extract_stat_name("regular_column"), None);
        assert_eq!(extract_stat_name("__ggsql_const_color__"), None);
    }

    #[test]
    fn test_constants() {
        assert_eq!(GLOBAL_DATA_KEY, "__ggsql_global__");
        assert_eq!(ORDER_COLUMN, "__ggsql_order__");
        assert_eq!(SCHEMA_ALIAS, "__schema__");
    }

    #[test]
    fn test_bin_end_column() {
        assert_eq!(
            bin_end_column("temperature"),
            "__ggsql_bin_end_temperature__"
        );
        assert_eq!(bin_end_column("x"), "__ggsql_bin_end_x__");
        assert_eq!(bin_end_column("value"), "__ggsql_bin_end_value__");
    }

    #[test]
    fn test_prefixes_built_from_components() {
        // Verify prefixes are correctly composed from building blocks
        assert_eq!(CONST_PREFIX, "__ggsql_const_");
        assert_eq!(STAT_PREFIX, "__ggsql_stat_");
        assert_eq!(CTE_PREFIX, "__ggsql_cte_");
        assert_eq!(LAYER_PREFIX, "__ggsql_layer_");
    }
}
