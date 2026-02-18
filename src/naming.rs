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

/// Full prefix for aesthetic columns: `__ggsql_aes_`
const AES_PREFIX: &str = concatcp!(GGSQL_PREFIX, "aes_");

/// Full prefix for builtin data tables: `__ggsql_data_`
const DATA_PREFIX: &str = concatcp!(GGSQL_PREFIX, "data_");

/// Key for global data in the layer data HashMap.
/// Used as the key in PreparedData.data to store global data that applies to all layers.
/// This is NOT a SQL table name - use `global_table()` for SQL statements.
pub const GLOBAL_DATA_KEY: &str = concatcp!(GGSQL_PREFIX, "global", GGSQL_SUFFIX);

/// Column name for row ordering in Vega-Lite (used by Path geom)
pub const ORDER_COLUMN: &str = concatcp!(GGSQL_PREFIX, "order", GGSQL_SUFFIX);

/// Column name for source identification in unified datasets
/// Added to each row to identify which layer's data the row belongs to.
/// Used with Vega-Lite filter transforms to select per-layer data.
pub const SOURCE_COLUMN: &str = concatcp!(GGSQL_PREFIX, "source", GGSQL_SUFFIX);

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

/// Generate table name for a builtin dataset.
///
/// Used when rewriting `ggsql:penguins` to the internal table name.
/// Format: `__ggsql_data_<name>__`
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::builtin_data_table("penguins"), "__ggsql_data_penguins__");
/// assert_eq!(naming::builtin_data_table("airquality"), "__ggsql_data_airquality__");
/// ```
pub fn builtin_data_table(name: &str) -> String {
    format!("{}{}{}", DATA_PREFIX, name, GGSQL_SUFFIX)
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

/// Generate column name for an aesthetic mapping.
///
/// Used when renaming source columns to aesthetic names in layer queries.
/// The prefix avoids conflicts with source data columns that might have
/// the same name as an aesthetic (e.g., a column named "x" or "color").
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::aesthetic_column("x"), "__ggsql_aes_x__");
/// assert_eq!(naming::aesthetic_column("fill"), "__ggsql_aes_fill__");
/// ```
pub fn aesthetic_column(aesthetic: &str) -> String {
    format!("{}{}{}", AES_PREFIX, aesthetic, GGSQL_SUFFIX)
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

/// Check if a column name is a synthetic aesthetic column.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert!(naming::is_aesthetic_column("__ggsql_aes_x__"));
/// assert!(naming::is_aesthetic_column("__ggsql_aes_fill__"));
/// assert!(!naming::is_aesthetic_column("x"));
/// assert!(!naming::is_aesthetic_column("__ggsql_stat_count"));
/// ```
pub fn is_aesthetic_column(name: &str) -> bool {
    name.starts_with(AES_PREFIX) && name.ends_with(GGSQL_SUFFIX)
}

/// Check if a column name is any synthetic ggsql column.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert!(naming::is_synthetic_column("__ggsql_const_color__"));
/// assert!(naming::is_synthetic_column("__ggsql_stat_count"));
/// assert!(naming::is_synthetic_column("__ggsql_aes_x__"));
/// assert!(!naming::is_synthetic_column("revenue"));
/// ```
pub fn is_synthetic_column(name: &str) -> bool {
    is_const_column(name) || is_stat_column(name) || is_aesthetic_column(name)
}

/// Generate bin end column name for a binned column.
///
/// Used by the Vega-Lite writer to store the upper bound of a bin
/// when using `bin: "binned"` encoding with xend/yend channels.
///
/// If the column is an aesthetic column (e.g., `__ggsql_aes_x__`), returns
/// the corresponding `end` aesthetic (e.g., `__ggsql_aes_xend__`).
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::bin_end_column("temperature"), "__ggsql_bin_end_temperature__");
/// assert_eq!(naming::bin_end_column("__ggsql_aes_x__"), "__ggsql_aes_xend__");
/// ```
pub fn bin_end_column(column: &str) -> String {
    // If it's an aesthetic column, use the xend/yend naming convention
    if let Some(aesthetic) = extract_aesthetic_name(column) {
        return aesthetic_column(&format!("{}end", aesthetic));
    }
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

/// Extract the aesthetic name from an aesthetic column.
///
/// Returns the aesthetic name from a prefixed column name.
///
/// # Example
/// ```
/// use ggsql::naming;
/// assert_eq!(naming::extract_aesthetic_name("__ggsql_aes_x__"), Some("x"));
/// assert_eq!(naming::extract_aesthetic_name("__ggsql_aes_fill__"), Some("fill"));
/// assert_eq!(naming::extract_aesthetic_name("regular_column"), None);
/// assert_eq!(naming::extract_aesthetic_name("__ggsql_stat_count"), None);
/// ```
pub fn extract_aesthetic_name(name: &str) -> Option<&str> {
    name.strip_prefix(AES_PREFIX)
        .and_then(|s| s.strip_suffix(GGSQL_SUFFIX))
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
        assert_eq!(SOURCE_COLUMN, "__ggsql_source__");
        assert_eq!(SCHEMA_ALIAS, "__schema__");
    }

    #[test]
    fn test_bin_end_column() {
        // Regular columns use bin_end prefix
        assert_eq!(
            bin_end_column("temperature"),
            "__ggsql_bin_end_temperature__"
        );
        assert_eq!(bin_end_column("x"), "__ggsql_bin_end_x__");
        assert_eq!(bin_end_column("value"), "__ggsql_bin_end_value__");

        // Aesthetic columns use the xend/yend convention
        assert_eq!(bin_end_column("__ggsql_aes_x__"), "__ggsql_aes_xend__");
        assert_eq!(bin_end_column("__ggsql_aes_y__"), "__ggsql_aes_yend__");
    }

    #[test]
    fn test_builtin_data_table() {
        assert_eq!(builtin_data_table("penguins"), "__ggsql_data_penguins__");
        assert_eq!(
            builtin_data_table("airquality"),
            "__ggsql_data_airquality__"
        );
    }

    #[test]
    fn test_prefixes_built_from_components() {
        // Verify prefixes are correctly composed from building blocks
        assert_eq!(CONST_PREFIX, "__ggsql_const_");
        assert_eq!(STAT_PREFIX, "__ggsql_stat_");
        assert_eq!(CTE_PREFIX, "__ggsql_cte_");
        assert_eq!(LAYER_PREFIX, "__ggsql_layer_");
        assert_eq!(AES_PREFIX, "__ggsql_aes_");
        assert_eq!(DATA_PREFIX, "__ggsql_data_");
    }

    #[test]
    fn test_aesthetic_column() {
        assert_eq!(aesthetic_column("x"), "__ggsql_aes_x__");
        assert_eq!(aesthetic_column("y"), "__ggsql_aes_y__");
        assert_eq!(aesthetic_column("fill"), "__ggsql_aes_fill__");
        assert_eq!(aesthetic_column("color"), "__ggsql_aes_color__");
    }

    #[test]
    fn test_is_aesthetic_column() {
        assert!(is_aesthetic_column("__ggsql_aes_x__"));
        assert!(is_aesthetic_column("__ggsql_aes_fill__"));
        assert!(!is_aesthetic_column("x"));
        assert!(!is_aesthetic_column("__ggsql_stat_count"));
        assert!(!is_aesthetic_column("__ggsql_const_color__"));
        // Partial matches should fail
        assert!(!is_aesthetic_column("__ggsql_aes_x")); // missing suffix
    }

    #[test]
    fn test_extract_aesthetic_name() {
        assert_eq!(extract_aesthetic_name("__ggsql_aes_x__"), Some("x"));
        assert_eq!(extract_aesthetic_name("__ggsql_aes_fill__"), Some("fill"));
        assert_eq!(extract_aesthetic_name("__ggsql_aes_color__"), Some("color"));
        assert_eq!(extract_aesthetic_name("regular_column"), None);
        assert_eq!(extract_aesthetic_name("__ggsql_stat_count"), None);
        assert_eq!(extract_aesthetic_name("__ggsql_const_color__"), None);
    }
}
