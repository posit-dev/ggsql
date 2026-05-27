//! Histogram geom implementation

use std::collections::HashMap;

use super::types::{get_quoted_column_name, CLOSED_VALUES, POSITION_VALUES};
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
    StatResult,
};
use crate::naming;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::reader::SqlDialect;
use crate::{DataFrame, GgsqlError, Mappings, Result};

use super::types::Schema;

/// Histogram geom - binned frequency distributions
#[derive(Debug, Clone, Copy)]
pub struct Histogram;

impl GeomTrait for Histogram {
    fn geom_type(&self) -> GeomType {
        GeomType::Histogram
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                // pos2 and pos1end are produced by stat_histogram but not valid for manual MAPPING
                ("pos2", DefaultAestheticValue::Delayed),
                ("pos1end", DefaultAestheticValue::Delayed),
                ("pos2end", DefaultAestheticValue::Delayed), // baseline value
            ],
        }
    }

    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Column("bin")),
                ("pos1end", DefaultAestheticValue::Column("bin_end")),
                ("pos2", DefaultAestheticValue::Column("count")),
                ("pos2end", DefaultAestheticValue::Number(0.0)),
            ],
        }
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["bin", "bin_end", "count", "density"]
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "bins",
                default: DefaultParamValue::Number(30.0),
                constraint: ParamConstraint::count(1.0),
            },
            ParamDefinition {
                name: "closed",
                default: DefaultParamValue::String("right"),
                constraint: ParamConstraint::string_option(CLOSED_VALUES),
            },
            ParamDefinition {
                name: "binwidth",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("stack"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
        ];
        PARAMS
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["pos1"]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &dyn Fn(&str) -> Result<DataFrame>,
        dialect: &dyn SqlDialect,
        _aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> Result<StatResult> {
        stat_histogram(
            query,
            aesthetics,
            group_by,
            parameters,
            execute_query,
            dialect,
        )
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
    dialect: &dyn SqlDialect,
) -> Result<StatResult> {
    // Get x column name from aesthetics
    let x_col = get_quoted_column_name(aesthetics, "pos1").ok_or_else(|| {
        GgsqlError::ValidationError("Histogram requires 'x' aesthetic mapping".to_string())
    })?;

    // Get bins from parameters (default: 30, validated by constraint)
    let ParameterValue::Number(bins) = parameters.get("bins").unwrap() else {
        unreachable!("bins validated by ParamConstraint::count")
    };
    let bins = *bins as usize;

    // Get closed parameter (default: "right", validated by constraint)
    let ParameterValue::String(closed) = parameters.get("closed").unwrap() else {
        unreachable!("closed validated by ParamConstraint::string_option")
    };
    let closed = closed.as_str();

    // Get binwidth from parameters (default: None - use bins to calculate)
    let explicit_binwidth = parameters.get("binwidth").and_then(|p| match p {
        ParameterValue::Number(n) => Some(*n),
        _ => None,
    });

    // Query min/max to compute bin width
    let stats_query = format!(
        "SELECT MIN({x}) as min_val, MAX({x}) as max_val FROM ({query}) AS \"__ggsql_stats__\"",
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
        // Right-closed (a, b]: use CEIL - 1, clamped to 0 minimum
        let ceil_expr = format!(
            "CEIL(({x} - {min} + {w} * 0.5) / {w}) - 1",
            x = x_col,
            min = min_val,
            w = bin_width
        );
        let clamped = dialect.sql_greatest(&["0", &ceil_expr]);
        format!(
            "({clamped}) * {w} + {min} - {w} * 0.5",
            clamped = clamped,
            w = bin_width,
            min = min_val
        )
    };
    // Determine aggregation expression based on weight aesthetic
    let agg_expr = if let Some(weight_value) = aesthetics.get("weight") {
        if weight_value.is_literal() {
            return Err(GgsqlError::ValidationError(
                "Histogram weight aesthetic must be a column, not a literal".to_string(),
            ));
        }
        if let Some(weight_col) = weight_value.column_name() {
            format!("SUM({})", naming::quote_ident(weight_col))
        } else {
            "COUNT(*)".to_string()
        }
    } else {
        "COUNT(*)".to_string()
    };

    // Stat output columns, prefixed to avoid clashing with user columns:
    // bin (start), bin_end (end), count/sum, density.
    let stat_bin = naming::stat_column("bin");
    let stat_bin_end = naming::stat_column("bin_end");
    let stat_count = naming::stat_column("count");
    let stat_density = naming::stat_column("density");

    let q_bin = naming::quote_ident(&stat_bin);
    let q_bin_end = naming::quote_ident(&stat_bin_end);
    let q_count = naming::quote_ident(&stat_count);
    let q_density = naming::quote_ident(&stat_density);

    // Two-stage query. `__binned__` groups rows by the bin expression and counts
    // them; its only non-facet grouping key is `bin_expr`. The outer SELECT then
    // derives bin_end (bin + width) and density from the already-grouped `bin`
    // and `count` columns. Computing the derived columns outside the GROUP BY
    // query keeps every grouped SELECT expression equal to a grouping key, which
    // strict dialects (e.g. BigQuery) require.
    let (group_cols, binned_select, density_window) = if group_by.is_empty() {
        (
            bin_expr.clone(),
            format!("{} AS {}, {} AS {}", bin_expr, q_bin, agg_expr, q_count),
            "OVER ()".to_string(),
        )
    } else {
        let grp_cols = group_by.join(", ");
        (
            format!("{}, {}", grp_cols, bin_expr),
            format!(
                "{}, {} AS {}, {} AS {}",
                grp_cols, bin_expr, q_bin, agg_expr, q_count
            ),
            format!("OVER (PARTITION BY {})", grp_cols),
        )
    };

    let transformed_query = format!(
        "WITH \"__stat_src__\" AS ({query}), \
         \"__binned__\" AS (SELECT {binned} FROM \"__stat_src__\" GROUP BY {group}) \
         SELECT *, {bin} + {width} AS {bin_end}, \
         {count} * 1.0 / SUM({count}) {density_window} AS {density} \
         FROM \"__binned__\"",
        query = query,
        binned = binned_select,
        group = group_cols,
        bin = q_bin,
        width = bin_width,
        bin_end = q_bin_end,
        count = q_count,
        density_window = density_window,
        density = q_density,
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
        consumed_aesthetics: vec!["pos1".to_string(), "weight".to_string()],
    })
}

/// Extract min and max from histogram stats DataFrame
pub fn extract_histogram_min_max(df: &DataFrame) -> Result<(f64, f64)> {
    if df.height() == 0 {
        return Err(GgsqlError::ValidationError(
            "No data for histogram statistics".to_string(),
        ));
    }

    let extract = |name: &str| -> Option<f64> {
        use arrow::array::Array;
        use arrow::datatypes::DataType;
        let col = df.column(name).ok()?;
        if col.is_null(0) {
            return None;
        }
        let casted = crate::array_util::cast_array(col, &DataType::Float64).ok()?;
        crate::array_util::as_f64(&casted).ok().map(|a| a.value(0))
    };

    let min_val = extract("min_val").ok_or_else(|| {
        GgsqlError::ValidationError("Could not extract min value for histogram".to_string())
    })?;

    let max_val = extract("max_val").ok_or_else(|| {
        GgsqlError::ValidationError("Could not extract max value for histogram".to_string())
    })?;

    Ok((min_val, max_val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    #[test]
    fn test_extract_min_max_null_errors() {
        let df = df! {
            "min_val" => vec![None::<f64>],
            "max_val" => vec![None::<f64>],
        }
        .unwrap();
        assert!(extract_histogram_min_max(&df).is_err());
    }
}
