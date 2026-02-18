//! Bar geom implementation

use std::collections::HashMap;
use std::collections::HashSet;

use super::types::get_column_name;
use super::{DefaultParam, DefaultParamValue, GeomAesthetics, GeomTrait, GeomType, StatResult};
use crate::naming;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::{DataFrame, GgsqlError, Mappings, Result};

use super::types::Schema;

/// Bar geom - bar charts with optional stat transform
#[derive(Debug, Clone, Copy)]
pub struct Bar;

impl GeomTrait for Bar {
    fn geom_type(&self) -> GeomType {
        GeomType::Bar
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            // Bar supports optional x and y - stat decides aggregation
            // If x is missing: single bar showing total
            // If y is missing: stat computes COUNT or SUM(weight)
            // weight: optional, if mapped uses SUM(weight) instead of COUNT(*)
            supported: &["x", "y", "weight", "fill", "stroke", "width", "opacity"],
            required: &[],
            hidden: &[],
        }
    }

    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[
            ("y", DefaultAestheticValue::Column("count")),
            ("x", DefaultAestheticValue::Column("x")),
            ("yend", DefaultAestheticValue::Number(0.0)),
        ]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["count", "x", "proportion"]
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[DefaultParam {
            name: "width",
            default: DefaultParamValue::Number(0.9),
        }]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["x", "y", "weight"]
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true // Bar stat decides COUNT vs identity based on y mapping
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        _parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        stat_bar_count(query, schema, aesthetics, group_by)
    }
}

impl std::fmt::Display for Bar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bar")
    }
}

/// Statistical transformation for bar: COUNT/SUM vs identity based on y and weight mappings
///
/// Uses pre-fetched schema to check column existence (avoiding redundant queries).
///
/// Decision logic for y:
/// - y mapped to literal → identity (use original data)
/// - y mapped to column that exists → identity (use original data)
/// - y mapped to column that doesn't exist + from wildcard → aggregation
/// - y mapped to column that doesn't exist + explicit → error
/// - y not mapped → aggregation
///
/// Decision logic for aggregation (when y triggers aggregation):
/// - weight not mapped → COUNT(*)
/// - weight mapped to literal → error (weight must be a column)
/// - weight mapped to column that exists → SUM(weight_col)
/// - weight mapped to column that doesn't exist + from wildcard → COUNT(*)
/// - weight mapped to column that doesn't exist + explicit → error
///
/// Returns `StatResult::Identity` for identity (no transformation),
/// `StatResult::Transformed` for aggregation with new y mapping.
fn stat_bar_count(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
) -> Result<StatResult> {
    // x is now optional - if not mapped, we'll use a dummy constant
    let x_col = get_column_name(aesthetics, "x");
    let use_dummy_x = x_col.is_none();

    // Build column lookup set from pre-fetched schema
    let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();

    // Check if y is mapped
    // Note: With upfront validation, if y is mapped to a column, that column must exist
    if let Some(y_value) = aesthetics.get("y") {
        // y is a literal value - use identity (no transformation)
        if y_value.is_literal() {
            return Ok(StatResult::Identity);
        }

        // y is a column reference - if it exists in schema, use identity
        // (column existence validated upfront, but we still check schema for stat decision)
        if let Some(y_col) = y_value.column_name() {
            if schema_columns.contains(y_col) {
                // y column exists - use identity (no transformation)
                return Ok(StatResult::Identity);
            }
            // y mapped but column doesn't exist in schema - fall through to aggregation
            // (this shouldn't happen with upfront validation, but handle gracefully)
        }
    }

    // y not mapped - apply aggregation (COUNT or SUM)
    // Determine aggregation expression based on weight aesthetic
    // Note: stat column is always "count" for predictability, even when using SUM
    // Note: With upfront validation, if weight is mapped to a column, that column must exist

    // Define stat column names
    let stat_count = naming::stat_column("count");
    let stat_proportion = naming::stat_column("proportion");
    let stat_x = naming::stat_column("x");
    let stat_dummy_value = naming::stat_column("dummy"); // Value used for dummy x

    let agg_expr = if let Some(weight_value) = aesthetics.get("weight") {
        // weight is mapped - check if it's valid
        if weight_value.is_literal() {
            return Err(GgsqlError::ValidationError(
                "Bar weight aesthetic must be a column, not a literal".to_string(),
            ));
        }

        if let Some(weight_col) = weight_value.column_name() {
            if schema_columns.contains(weight_col) {
                // weight column exists - use SUM (but still call it "count")
                format!("SUM({}) AS {}", weight_col, stat_count)
            } else {
                // weight mapped but column doesn't exist - fall back to COUNT
                // (this shouldn't happen with upfront validation, but handle gracefully)
                format!("COUNT(*) AS {}", stat_count)
            }
        } else {
            // Shouldn't happen (not literal, not column), fall back to COUNT
            format!("COUNT(*) AS {}", stat_count)
        }
    } else {
        // weight not mapped - use COUNT
        format!("COUNT(*) AS {}", stat_count)
    };

    // Build the query based on whether x is mapped or not
    // Use two-stage query: first GROUP BY, then calculate proportion with window function
    let (transformed_query, stat_columns, dummy_columns, consumed_aesthetics) = if use_dummy_x {
        // x is not mapped - use dummy constant, no GROUP BY on x
        let (grouped_select, final_select) = if group_by.is_empty() {
            (
                format!(
                    "'{dummy}' AS {x}, {agg}",
                    dummy = stat_dummy_value,
                    x = stat_x,
                    agg = agg_expr
                ),
                format!(
                    "*, {count} * 1.0 / SUM({count}) OVER () AS {prop}",
                    count = stat_count,
                    prop = stat_proportion
                ),
            )
        } else {
            let grp_cols = group_by.join(", ");
            (
                format!(
                    "{g}, '{dummy}' AS {x}, {agg}",
                    g = grp_cols,
                    dummy = stat_dummy_value,
                    x = stat_x,
                    agg = agg_expr
                ),
                format!(
                    "*, {count} * 1.0 / SUM({count}) OVER (PARTITION BY {grp}) AS {prop}",
                    count = stat_count,
                    grp = grp_cols,
                    prop = stat_proportion
                ),
            )
        };

        let query_str = if group_by.is_empty() {
            // No grouping at all - single aggregate
            format!(
                "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__) SELECT {final} FROM __grouped__",
                query = query,
                grouped = grouped_select,
                final = final_select
            )
        } else {
            // Group by partition/facet variables only
            let group_cols = group_by.join(", ");
            format!(
                "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __grouped__",
                query = query,
                grouped = grouped_select,
                group = group_cols,
                final = final_select
            )
        };

        // Stat columns: x (dummy), count, and proportion - x is a dummy placeholder
        // Consumed: weight (used for weighted sums)
        (
            query_str,
            vec![
                "x".to_string(),
                "count".to_string(),
                "proportion".to_string(),
            ],
            vec!["x".to_string()],
            vec!["weight".to_string()],
        )
    } else {
        // x is mapped - use existing logic with two-stage query
        let x_col = x_col.unwrap();

        // Build grouped columns (group_by includes partition_by + facet variables + x)
        let group_cols = if group_by.is_empty() {
            x_col.clone()
        } else {
            let mut cols = group_by.to_vec();
            cols.push(x_col.clone());
            cols.join(", ")
        };

        // Keep original x column name, only add the aggregated stat column
        let (grouped_select, final_select) = if group_by.is_empty() {
            (
                format!("{x}, {agg}", x = x_col, agg = agg_expr),
                format!(
                    "*, {count} * 1.0 / SUM({count}) OVER () AS {prop}",
                    count = stat_count,
                    prop = stat_proportion
                ),
            )
        } else {
            let grp_cols = group_by.join(", ");
            (
                format!("{g}, {x}, {agg}", g = grp_cols, x = x_col, agg = agg_expr),
                format!(
                    "*, {count} * 1.0 / SUM({count}) OVER (PARTITION BY {grp}) AS {prop}",
                    count = stat_count,
                    grp = grp_cols,
                    prop = stat_proportion
                ),
            )
        };

        let query_str = format!(
            "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __grouped__",
            query = query,
            grouped = grouped_select,
            group = group_cols,
            final = final_select
        );

        // count and proportion stat columns (x is preserved from original data), no dummies
        // Consumed: weight (used for weighted sums)
        (
            query_str,
            vec!["count".to_string(), "proportion".to_string()],
            vec![],
            vec!["weight".to_string()],
        )
    };

    // Return with stat column names and consumed aesthetics
    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns,
        consumed_aesthetics,
    })
}
