//! Histogram geom implementation

use std::collections::HashMap;

use super::types::get_column_name;
use super::{DefaultParam, DefaultParamValue, GeomAesthetics, GeomTrait, GeomType, StatResult};
use crate::plot::types::ParameterValue;
use crate::{DataFrame, GgsqlError, Mappings, Result};

use super::types::Schema;

/// Histogram geom - binned frequency distributions
#[derive(Debug, Clone, Copy)]
pub struct Histogram;

impl GeomTrait for Histogram {
    fn geom_type(&self) -> GeomType {
        GeomType::Histogram
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "weight", "color", "colour", "fill", "opacity"],
            required: &["x"],
            // y and x2 are produced by stat_histogram but not valid for manual MAPPING
            hidden: &["y", "x2"],
        }
    }

    fn default_remappings(&self) -> &'static [(&'static str, &'static str)] {
        &[("bin", "x"), ("bin_end", "x2"), ("count", "y")]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["bin", "bin_end", "count", "density"]
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
            DefaultParam {
                name: "bins",
                default: DefaultParamValue::Number(30.0),
            },
            DefaultParam {
                name: "closed",
                default: DefaultParamValue::String("right"),
            },
            DefaultParam {
                name: "binwidth",
                default: DefaultParamValue::Null,
            },
        ]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["x"]
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        stat_histogram(query, aesthetics, group_by, parameters, execute_query)
    }
}

impl std::fmt::Display for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "histogram")
    }
}

/// Statistical transformation for histogram: bin continuous values and count
fn stat_histogram(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    execute_query: &dyn Fn(&str) -> Result<DataFrame>,
) -> Result<StatResult> {
    // Get x column name from aesthetics
    let x_col = get_column_name(aesthetics, "x").ok_or_else(|| {
        GgsqlError::ValidationError("Histogram requires 'x' aesthetic mapping".to_string())
    })?;

    // Get bins from parameters (default: 30)
    let bins = parameters
        .get("bins")
        .and_then(|p| match p {
            ParameterValue::Number(n) => Some(*n as usize),
            _ => None,
        })
        .expect("bins is not the correct format. Expected a number");

    // Get closed parameter (default: "right")
    let closed = parameters
        .get("closed")
        .and_then(|p| match p {
            ParameterValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .expect("closed is not the correct format. Expected a string");

    // Get binwidth from parameters (default: None - use bins to calculate)
    let explicit_binwidth = parameters.get("binwidth").and_then(|p| match p {
        ParameterValue::Number(n) => Some(*n),
        _ => None,
    });

    // Query min/max to compute bin width
    let stats_query = format!(
        "SELECT MIN({x}) as min_val, MAX({x}) as max_val FROM ({query})",
        x = x_col,
        query = query
    );
    let stats_df = execute_query(&stats_query)?;

    let (min_val, max_val) = extract_histogram_min_max(&stats_df)?;

    // Compute bin width: use explicit binwidth if provided, otherwise calculate from bins
    // Round to 10 decimal places to avoid SQL DECIMAL overflow issues
    let bin_width = if let Some(bw) = explicit_binwidth {
        bw
    } else if min_val >= max_val {
        1.0 // Fallback for edge case
    } else {
        ((max_val - min_val) / (bins - 1) as f64 * 1e10).round() / 1e10
    };
    let min_val = (min_val * 1e10).round() / 1e10;

    // Build the bin expression (bin start)
    let bin_expr = if closed == "left" {
        // Left-closed [a, b): use FLOOR
        format!(
            "(FLOOR(({x} - {min} + {w} * 0.5) / {w})) * {w} + {min} - {w} * 0.5",
            x = x_col,
            min = min_val,
            w = bin_width
        )
    } else {
        // Right-closed (a, b]: use CEIL - 1 with GREATEST for min value
        format!(
            "(GREATEST(CEIL(({x} - {min} + {w} * 0.5) / {w}) - 1, 0)) * {w} + {min} - {w} * 0.5",
            x = x_col,
            min = min_val,
            w = bin_width
        )
    };
    // Build the bin end expression (bin start + bin width)
    let bin_end_expr = format!("{expr} + {w}", expr = bin_expr, w = bin_width);

    // Build grouped columns (group_by includes partition_by + facet variables)
    let group_cols = if group_by.is_empty() {
        bin_expr.clone()
    } else {
        let mut cols: Vec<String> = group_by.to_vec();
        cols.push(bin_expr.clone());
        cols.join(", ")
    };

    // Determine aggregation expression based on weight aesthetic
    let agg_expr = if let Some(weight_value) = aesthetics.get("weight") {
        if weight_value.is_literal() {
            return Err(GgsqlError::ValidationError(
                "Histogram weight aesthetic must be a column, not a literal".to_string(),
            ));
        }
        if let Some(weight_col) = weight_value.column_name() {
            format!("SUM({})", weight_col)
        } else {
            "COUNT(*)".to_string()
        }
    } else {
        "COUNT(*)".to_string()
    };

    // Use semantically meaningful column names with prefix to avoid conflicts
    // Include bin (start), bin_end (end), count/sum, and density
    // Use a two-stage query: first GROUP BY, then calculate density with window function
    let (binned_select, final_select) = if group_by.is_empty() {
        (
            format!(
                "{} AS __ggsql_stat__bin, {} AS __ggsql_stat__bin_end, {} AS __ggsql_stat__count",
                bin_expr, bin_end_expr, agg_expr
            ),
            "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER () AS __ggsql_stat__density".to_string()
        )
    } else {
        let grp_cols = group_by.join(", ");
        (
            format!(
                "{}, {} AS __ggsql_stat__bin, {} AS __ggsql_stat__bin_end, {} AS __ggsql_stat__count",
                grp_cols, bin_expr, bin_end_expr, agg_expr
            ),
            format!(
                "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER (PARTITION BY {}) AS __ggsql_stat__density",
                grp_cols
            )
        )
    };

    let transformed_query = format!(
        "WITH __stat_src__ AS ({query}), __binned__ AS (SELECT {binned} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __binned__",
        query = query,
        binned = binned_select,
        group = group_cols,
        final = final_select
    );

    // Histogram always transforms - produces bin, bin_end, count, and density columns
    // Consumed aesthetics: x (transformed into bin/bin_end) and weight (used for weighted counts)
    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns: vec![
            "bin".to_string(),
            "bin_end".to_string(),
            "count".to_string(),
            "density".to_string(),
        ],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["x".to_string(), "weight".to_string()],
    })
}

/// Extract min and max from histogram stats DataFrame
pub fn extract_histogram_min_max(df: &DataFrame) -> Result<(f64, f64)> {
    if df.height() == 0 {
        return Err(GgsqlError::ValidationError(
            "No data for histogram statistics".to_string(),
        ));
    }

    let min_val = df
        .column("min_val")
        .ok()
        .and_then(|s| s.get(0).ok())
        .and_then(|s| s.try_extract::<f64>().ok())
        .ok_or_else(|| {
            GgsqlError::ValidationError("Could not extract min value for histogram".to_string())
        })?;

    let max_val = df
        .column("max_val")
        .ok()
        .and_then(|s| s.get(0).ok())
        .and_then(|s| s.try_extract::<f64>().ok())
        .ok_or_else(|| {
            GgsqlError::ValidationError("Could not extract max value for histogram".to_string())
        })?;

    Ok((min_val, max_val))
}
