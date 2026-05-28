//! Core types for the geom trait system
//!
//! These types are used by all geom implementations and are shared across the module.

use crate::plot::aesthetic::parse_position;
use crate::{naming, plot::types::DefaultAestheticValue, Mappings};

// Re-export shared types from the central location
pub use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};

// =============================================================================
// Common constraint value arrays
// =============================================================================

/// Standard position adjustment values for the `position` parameter
pub const POSITION_VALUES: &[&str] = &["identity", "stack", "dodge", "jitter"];

/// Closed interval side values for binned data
pub const CLOSED_VALUES: &[&str] = &["left", "right"];

/// Standard `side` parameter values for layers and positions that can render
/// either both halves of a symmetric shape (or jitter range) or just one of
/// them. Used by violin, boxplot, and jitter.
pub const SIDE_VALUES: &[&str] = &["both", "left", "top", "right", "bottom"];

/// Aesthetic aliases: user-facing names that resolve to concrete aesthetics.
///
/// An alias is considered supported if any of its target aesthetics are supported
/// by a geom. For example, `color` resolves to `stroke` and/or `fill` — so any geom
/// that supports either `stroke` or `fill` also accepts `color`.
///
/// Note: Spelling variants (`colour`/`col` → `color`) are handled separately at parse
/// time by `normalise_aes_name()` in `parser/builder.rs`.
pub const AESTHETIC_ALIASES: &[(&str, &[&str])] = &[("color", &["stroke", "fill"])];

/// Shared `aggregate` parameter definition. Every geom that opts into the
/// Aggregate stat (i.e. `aggregate_domain_aesthetics()` returns `Some(_)`)
/// includes this entry in its `default_params()`. The constraint validates
/// only the structural shape (string / array-of-strings / null); the
/// per-entry vocabulary is checked by `stat_aggregate::parse_aggregate_param`.
pub const AGGREGATE_PARAM: ParamDefinition = ParamDefinition {
    name: "aggregate",
    default: DefaultParamValue::Null,
    constraint: ParamConstraint::string_or_string_array_unconstrained(),
};

/// Default aesthetic values for a geom type
///
/// This struct describes which aesthetics a geom supports, requires, and their default values.
#[derive(Debug, Clone, Copy)]
pub struct DefaultAesthetics {
    /// Aesthetic defaults: maps aesthetic name to default value
    /// - Required: Must be provided via MAPPING
    /// - Delayed: Produced by stat transform (REMAPPING only)
    /// - Null: Supported but no default
    /// - Other variants: Actual default values
    pub defaults: &'static [(&'static str, DefaultAestheticValue)],
}

impl DefaultAesthetics {
    /// Get all aesthetic names (including Delayed and aliases)
    pub fn names(&self) -> Vec<&'static str> {
        let mut result: Vec<&'static str> = self.defaults.iter().map(|(name, _)| *name).collect();
        // Include alias names if any of their targets are in the defaults
        for &(alias, targets) in AESTHETIC_ALIASES {
            if targets.iter().any(|t| result.contains(t)) {
                result.push(alias);
            }
        }
        result
    }

    /// Get supported aesthetic names (excludes Delayed, for MAPPING validation)
    ///
    /// Returns the literal names from defaults plus any aliases whose targets are
    /// supported. For bidirectional position checking, use `is_supported()` which
    /// handles pos1/pos2 equivalence.
    pub fn supported(&self) -> Vec<&'static str> {
        let mut result: Vec<&'static str> = self
            .defaults
            .iter()
            .filter(|(_, value)| !matches!(value, DefaultAestheticValue::Delayed))
            .map(|(name, _)| *name)
            .collect();
        // Include alias names if any of their targets are supported
        for &(alias, targets) in AESTHETIC_ALIASES {
            if targets.iter().any(|t| result.contains(t)) {
                result.push(alias);
            }
        }
        result
    }

    /// Get required aesthetic names (those marked as Required)
    pub fn required(&self) -> Vec<&'static str> {
        self.defaults
            .iter()
            .filter_map(|(name, value)| {
                if matches!(value, DefaultAestheticValue::Required) {
                    Some(*name)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if an aesthetic is supported (not Delayed)
    ///
    /// Position aesthetics are bidirectional: if pos1* is supported, pos2* is also
    /// considered supported (and vice versa). Aliases (e.g., `color`) are supported
    /// if any of their target aesthetics are supported.
    pub fn is_supported(&self, name: &str) -> bool {
        // Check for direct match first
        let direct_match = self
            .defaults
            .iter()
            .any(|(n, value)| !matches!(value, DefaultAestheticValue::Delayed) && *n == name);
        if direct_match {
            return true;
        }

        // Check for bidirectional position match
        if let Some((slot, suffix)) = parse_position(name) {
            let other_slot = if slot == 1 { 2 } else { 1 };
            let equivalent = format!("pos{}{}", other_slot, suffix);
            return self.defaults.iter().any(|(n, value)| {
                !matches!(value, DefaultAestheticValue::Delayed) && *n == equivalent
            });
        }

        // Check if name is an alias that resolves to a supported aesthetic
        for &(alias, targets) in AESTHETIC_ALIASES {
            if alias == name {
                return targets.iter().any(|t| self.is_supported(t));
            }
        }

        false
    }

    /// Check if an aesthetic exists (including Delayed)
    pub fn contains(&self, name: &str) -> bool {
        self.defaults.iter().any(|(n, _)| *n == name)
    }

    /// Check if an aesthetic is required
    pub fn is_required(&self, name: &str) -> bool {
        self.defaults
            .iter()
            .any(|(n, value)| *n == name && matches!(value, DefaultAestheticValue::Required))
    }

    /// Get the default value for an aesthetic by name
    pub fn get(&self, name: &str) -> Option<&'static DefaultAestheticValue> {
        self.defaults
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, value)| value)
    }

    /// Names of aesthetics declared as `Dummy` — i.e. position aesthetics
    /// the default `apply_stat_transform` should fill in with a synthetic
    /// categorical column when the user leaves the whole axis family
    /// unmapped.
    pub fn dummy_axes(&self) -> Vec<&'static str> {
        self.defaults
            .iter()
            .filter_map(|(name, value)| {
                if matches!(value, DefaultAestheticValue::Dummy) {
                    Some(*name)
                } else {
                    None
                }
            })
            .collect()
    }
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
        /// Names of stat-computed columns (e.g., ["count", "bin", "pos1"])
        /// These are semantic names that will be prefixed with __ggsql_stat__
        /// and mapped to aesthetics via default_remappings or REMAPPING clause
        stat_columns: Vec<String>,
        /// Names of stat columns that are dummy/placeholder values
        /// (e.g., "pos1" when bar chart has no x mapped - produces a constant value)
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

/// Wrap a stat result with `ORDER BY <aesthetic>`.
///
/// Used by line/area/ribbon to ensure the rendered output is sorted along the
/// domain axis whether or not the layer also goes through the Aggregate stat.
///
/// - `Identity` → becomes `Transformed` with `<input_query> ORDER BY <aes>`,
///   empty `stat_columns`/`dummy_columns`/`consumed_aesthetics`.
/// - `Transformed` → wraps the existing query in
///   `SELECT * FROM (<query>) AS "__ggsql_ord__" ORDER BY <aes>` and preserves
///   the stat metadata.
pub fn wrap_with_order_by(input_query: &str, result: StatResult, aesthetic: &str) -> StatResult {
    let order_col = naming::aesthetic_column(aesthetic);
    let order_quoted = naming::quote_ident(&order_col);
    match result {
        StatResult::Identity => StatResult::Transformed {
            query: format!("{} ORDER BY {}", input_query, order_quoted),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        },
        StatResult::Transformed {
            query,
            stat_columns,
            dummy_columns,
            consumed_aesthetics,
        } => StatResult::Transformed {
            query: format!(
                "SELECT * FROM ({}) AS \"__ggsql_ord__\" ORDER BY {}",
                query, order_quoted
            ),
            stat_columns,
            dummy_columns,
            consumed_aesthetics,
        },
    }
}

/// Wrap `query` so it produces a literal categorical column carrying the
/// dummy-axis sentinel value for `axis` (`"pos1"` or `"pos2"`). Used by
/// geoms that should still render sensibly when the user omits a position
/// aesthetic.
///
/// The wrapped query has shape:
/// ```sql
/// SELECT '__ggsql_stat_dummy' AS "__ggsql_stat_<axis>", *
/// FROM (<query>) AS "__ggsql_dummy_src__"
/// ```
///
/// This composes with any stat: pre-wrap the input (when the geom's stat
/// groups by the dummied axis, e.g. boxplot/violin) so the existing
/// `GROUP BY` collapses to a single group, or post-wrap a stat output
/// (aggregate / identity) so the dummy column is just decoration.
pub fn wrap_with_dummy_axis(query: &str, axis: &str) -> String {
    let stat_col = naming::stat_column(axis);
    let dummy_v = naming::stat_column("dummy");
    format!(
        "SELECT '{val}' AS {col}, * FROM ({q}) AS \"__ggsql_dummy_src__\"",
        val = dummy_v,
        col = naming::quote_ident(&stat_col),
        q = query,
    )
}

/// Post-wrap a `StatResult` to add a dummy column for `axis` (`"pos1"` or
/// `"pos2"`).
///
/// Wraps the inner query via [`wrap_with_dummy_axis`] (turning `Identity`
/// into a `Transformed` over the original input) and appends `axis` to
/// both `stat_columns` and `dummy_columns` so `execute/layer.rs` flips
/// `is_dummy: true` on the resulting aesthetic.
pub fn wrap_stat_with_dummy_axis(input_query: &str, result: StatResult, axis: &str) -> StatResult {
    match result {
        StatResult::Identity => StatResult::Transformed {
            query: wrap_with_dummy_axis(input_query, axis),
            stat_columns: vec![axis.to_string()],
            dummy_columns: vec![axis.to_string()],
            consumed_aesthetics: vec![],
        },
        StatResult::Transformed {
            query,
            mut stat_columns,
            mut dummy_columns,
            consumed_aesthetics,
        } => {
            // Idempotent: a stat that already produced a dummy for this axis
            // must not be re-wrapped — the SQL would gain a duplicate column.
            let already_dummied = dummy_columns.iter().any(|s| s == axis);
            let wrapped = if already_dummied {
                query
            } else {
                wrap_with_dummy_axis(&query, axis)
            };
            if !stat_columns.iter().any(|s| s == axis) {
                stat_columns.push(axis.to_string());
            }
            if !already_dummied {
                dummy_columns.push(axis.to_string());
            }
            StatResult::Transformed {
                query: wrapped,
                stat_columns,
                dummy_columns,
                consumed_aesthetics,
            }
        }
    }
}

/// Convenience wrapper for the common case of dummying `pos1`.
pub fn wrap_stat_with_dummy_pos1(input_query: &str, result: StatResult) -> StatResult {
    wrap_stat_with_dummy_axis(input_query, result, "pos1")
}

/// Returns true when at least one aesthetic in the same axis family as
/// `axis` (e.g. `pos1`, `pos1min`, `pos1max`, `pos1end`, `pos1offset`) is
/// mapped to a column.
///
/// Used by the default `apply_stat_transform` to decide whether to fill in
/// a dummy categorical column for an unmapped axis.
pub fn axis_family_has_mapping(aesthetics: &Mappings, axis: &str) -> bool {
    use crate::plot::aesthetic::parse_position;
    let Some((target_slot, _)) = parse_position(axis) else {
        return false;
    };
    aesthetics
        .aesthetics
        .iter()
        .any(|(name, value)| match parse_position(name) {
            Some((slot, _)) => slot == target_slot && value.column_name().is_some(),
            None => false,
        })
}

/// Helper to extract column name from aesthetic value
pub fn get_column_name(aesthetics: &Mappings, aesthetic: &str) -> Option<String> {
    use crate::AestheticValue;
    aesthetics.get(aesthetic).and_then(|v| match v {
        AestheticValue::Column { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Helper to extract a double-quoted column name for use in SQL expressions.
pub fn get_quoted_column_name(aesthetics: &Mappings, aesthetic: &str) -> Option<String> {
    get_column_name(aesthetics, aesthetic).map(|n| naming::quote_ident(&n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_aesthetics_methods() {
        // Create a DefaultAesthetics with various value types
        let aes = DefaultAesthetics {
            defaults: &[
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("size", DefaultAestheticValue::Number(3.0)),
                ("stroke", DefaultAestheticValue::String("black")),
                ("fill", DefaultAestheticValue::Null),
                ("yend", DefaultAestheticValue::Delayed),
            ],
        };

        // Test get() method
        assert_eq!(aes.get("x"), Some(&DefaultAestheticValue::Required));
        assert_eq!(aes.get("size"), Some(&DefaultAestheticValue::Number(3.0)));
        assert_eq!(
            aes.get("stroke"),
            Some(&DefaultAestheticValue::String("black"))
        );
        assert_eq!(aes.get("fill"), Some(&DefaultAestheticValue::Null));
        assert_eq!(aes.get("yend"), Some(&DefaultAestheticValue::Delayed));
        assert_eq!(aes.get("nonexistent"), None);

        // Test names() - includes all aesthetics + aliases
        let names = aes.names();
        assert_eq!(names.len(), 7); // 6 defaults + "color" alias (has stroke+fill)
        assert!(names.contains(&"x"));
        assert!(names.contains(&"yend"));
        assert!(names.contains(&"color")); // alias resolved from stroke+fill

        // Test supported() - excludes Delayed, includes aliases
        let supported = aes.supported();
        assert_eq!(supported.len(), 6); // 5 non-delayed + "color" alias
        assert!(supported.contains(&"x"));
        assert!(supported.contains(&"size"));
        assert!(supported.contains(&"fill"));
        assert!(supported.contains(&"color")); // alias
        assert!(!supported.contains(&"yend")); // Delayed excluded

        // Test required() - only Required variants
        let required = aes.required();
        assert_eq!(required.len(), 2);
        assert!(required.contains(&"x"));
        assert!(required.contains(&"y"));
        assert!(!required.contains(&"size"));

        // Test is_supported() - efficient membership check
        assert!(aes.is_supported("x"));
        assert!(aes.is_supported("size"));
        assert!(aes.is_supported("color")); // alias: has stroke+fill
        assert!(!aes.is_supported("yend")); // Delayed not supported
        assert!(!aes.is_supported("nonexistent"));

        // Test contains() - includes Delayed
        assert!(aes.contains("x"));
        assert!(aes.contains("yend")); // Delayed included
        assert!(!aes.contains("nonexistent"));

        // Test is_required()
        assert!(aes.is_required("x"));
        assert!(aes.is_required("y"));
        assert!(!aes.is_required("size"));
        assert!(!aes.is_required("yend"));
    }

    #[test]
    fn wrap_with_order_by_identity_appends_order() {
        let result = wrap_with_order_by("SELECT * FROM t", StatResult::Identity, "pos1");
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                dummy_columns,
                consumed_aesthetics,
            } => {
                assert_eq!(query, "SELECT * FROM t ORDER BY \"__ggsql_aes_pos1__\"");
                assert!(stat_columns.is_empty());
                assert!(dummy_columns.is_empty());
                assert!(consumed_aesthetics.is_empty());
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn wrap_with_order_by_transformed_wraps_query_and_preserves_metadata() {
        let inner = StatResult::Transformed {
            query: "SELECT * FROM grouped".to_string(),
            stat_columns: vec!["pos2".to_string(), "aggregate".to_string()],
            dummy_columns: vec!["pos1".to_string()],
            consumed_aesthetics: vec!["pos2".to_string()],
        };
        let result = wrap_with_order_by("SELECT * FROM raw", inner, "pos1");
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                dummy_columns,
                consumed_aesthetics,
            } => {
                assert_eq!(
                    query,
                    "SELECT * FROM (SELECT * FROM grouped) AS \"__ggsql_ord__\" ORDER BY \"__ggsql_aes_pos1__\""
                );
                assert_eq!(
                    stat_columns,
                    vec!["pos2".to_string(), "aggregate".to_string()]
                );
                assert_eq!(dummy_columns, vec!["pos1".to_string()]);
                assert_eq!(consumed_aesthetics, vec!["pos2".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn wrap_with_dummy_pos1_produces_expected_sql() {
        let wrapped = wrap_with_dummy_axis("SELECT * FROM t", "pos1");
        assert_eq!(
            wrapped,
            "SELECT '__ggsql_stat_dummy' AS \"__ggsql_stat_pos1\", * FROM (SELECT * FROM t) AS \"__ggsql_dummy_src__\""
        );
    }

    #[test]
    fn wrap_stat_with_dummy_pos1_promotes_identity() {
        let result = wrap_stat_with_dummy_pos1("SELECT * FROM raw", StatResult::Identity);
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                dummy_columns,
                consumed_aesthetics,
            } => {
                assert!(query.contains("__ggsql_stat_dummy"));
                assert!(query.contains("__ggsql_stat_pos1"));
                assert!(query.contains("SELECT * FROM raw"));
                assert_eq!(stat_columns, vec!["pos1".to_string()]);
                assert_eq!(dummy_columns, vec!["pos1".to_string()]);
                assert!(consumed_aesthetics.is_empty());
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn wrap_stat_with_dummy_pos1_extends_transformed_metadata() {
        let inner = StatResult::Transformed {
            query: "SELECT 1 AS x".to_string(),
            stat_columns: vec!["count".to_string()],
            dummy_columns: vec![],
            consumed_aesthetics: vec!["weight".to_string()],
        };
        let result = wrap_stat_with_dummy_pos1("SELECT * FROM raw", inner);
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                dummy_columns,
                consumed_aesthetics,
            } => {
                assert!(query.contains("__ggsql_stat_dummy"));
                assert!(query.contains("__ggsql_stat_pos1"));
                assert!(query.contains("SELECT 1 AS x"));
                assert_eq!(stat_columns, vec!["count".to_string(), "pos1".to_string()]);
                assert_eq!(dummy_columns, vec!["pos1".to_string()]);
                assert_eq!(consumed_aesthetics, vec!["weight".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn wrap_stat_with_dummy_pos1_idempotent_on_pos1() {
        // Caller already had pos1 in stat_columns/dummy_columns; helper must
        // not duplicate.
        let inner = StatResult::Transformed {
            query: "SELECT 1".to_string(),
            stat_columns: vec!["pos1".to_string()],
            dummy_columns: vec!["pos1".to_string()],
            consumed_aesthetics: vec![],
        };
        let result = wrap_stat_with_dummy_pos1("SELECT *", inner);
        match result {
            StatResult::Transformed {
                stat_columns,
                dummy_columns,
                ..
            } => {
                assert_eq!(stat_columns, vec!["pos1".to_string()]);
                assert_eq!(dummy_columns, vec!["pos1".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn test_color_alias_requires_stroke_or_fill() {
        // Geom with neither stroke nor fill: color alias should NOT be supported
        let aes = DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("size", DefaultAestheticValue::Number(3.0)),
            ],
        };

        assert!(!aes.is_supported("color"));
        assert!(!aes.supported().contains(&"color"));
        assert!(!aes.names().contains(&"color"));

        // Geom with only stroke: color alias should be supported
        let aes_stroke = DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
            ],
        };

        assert!(aes_stroke.is_supported("color"));
        assert!(aes_stroke.supported().contains(&"color"));

        // Geom with only fill: color alias should be supported
        let aes_fill = DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("black")),
            ],
        };

        assert!(aes_fill.is_supported("color"));
        assert!(aes_fill.supported().contains(&"color"));
    }
}
