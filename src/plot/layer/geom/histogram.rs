//! Histogram geom implementation

use super::types::{get_quoted_column_name, CLOSED_VALUES, POSITION_VALUES};
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
    StatResult,
};
use crate::naming;
use crate::plot::types::{DefaultAestheticValue, ParameterValue, Parameters};
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
        parameters: &Parameters,
        execute_query: &dyn Fn(&str) -> Result<DataFrame>,
        dialect: &dyn SqlDialect,
        aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> Result<StatResult> {
        stat_histogram(
            query,
            aesthetics,
            group_by,
            parameters,
            execute_query,
            dialect,
            aesthetic_ctx,
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
    parameters: &Parameters,
    execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    dialect: &dyn SqlDialect,
    aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
) -> Result<StatResult> {
    // Get x column name from aesthetics
    let x_col = get_quoted_column_name(aesthetics, "pos1", dialect).ok_or_else(|| {
        let name = aesthetic_ctx.map_internal_to_user("pos1");
        GgsqlError::ValidationError(format!("Histogram requires '{}' aesthetic mapping", name))
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
    let stats_query = crate::sql::select_from(
        dialect,
        &format!("MIN({x_col}) as min_val, MAX({x_col}) as max_val"),
        crate::sql::FromItem::Query(query),
        "__ggsql_stats__",
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
            "{} - 1",
            dialect.sql_ceil(&format!(
                "({x} - {min} + {w} * 0.5) / {w}",
                x = x_col,
                min = min_val,
                w = bin_width
            ))
        );
        let clamped = dialect.sql_greatest(&["0", &ceil_expr]);
        format!(
            "({clamped}) * {w} + {min} - {w} * 0.5",
            clamped = clamped,
            w = bin_width,
            min = min_val
        )
    };
    // Group by a plain column, not the bin expression itself: MonetDB does
    // not match a complex GROUP BY expression structurally against the
    // SELECT list ("cannot use non GROUP BY column ... without an aggregate
    // function"), and alias references in GROUP BY are not portable either
    // (Oracle). Computing the expression in an inner CTE and grouping by
    // the resulting column works in every dialect.
    let bin_key = dialect.quote_ident("__ggsql_bin_key__");
    let group_cols = if group_by.is_empty() {
        bin_key.clone()
    } else {
        let mut cols: Vec<String> = group_by.to_vec();
        cols.push(bin_key.clone());
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
            format!("SUM({})", dialect.quote_ident(weight_col))
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

    let q_bin = dialect.quote_ident(&stat_bin);
    let q_bin_end = dialect.quote_ident(&stat_bin_end);
    let q_count = dialect.quote_ident(&stat_count);
    let q_density = dialect.quote_ident(&stat_density);

    // Three-stage query. `__bin_src__` materializes the bin expression as a
    // plain column (see above); `__binned__` groups by it and counts; the
    // outer SELECT then derives bin_end (bin + width) and density from the
    // already-grouped `bin` and `count` columns. Computing the derived
    // columns outside the GROUP BY query keeps every grouped SELECT
    // expression equal to a grouping key, which strict dialects (e.g.
    // BigQuery) require.
    let (binned_select, density_window) = if group_by.is_empty() {
        (
            format!("{} AS {}, {} AS {}", bin_key, q_bin, agg_expr, q_count),
            "OVER ()".to_string(),
        )
    } else {
        let grp_cols = group_by
            .iter()
            .map(|c| dialect.quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!(
                "{}, {} AS {}, {} AS {}",
                grp_cols, bin_key, q_bin, agg_expr, q_count
            ),
            format!("OVER (PARTITION BY {})", grp_cols),
        )
    };

    let __stat_src__ = dialect.quote_ident("__stat_src__");
    let __bin_src__ = dialect.quote_ident("__bin_src__");
    let __binned__ = dialect.quote_ident("__binned__");
    let transformed_query = format!(
        "WITH {__stat_src__} AS ({query}), \
         {__bin_src__} AS (SELECT *, {bin_expr} AS {bin_key} FROM {__stat_src__}), \
         {__binned__} AS (SELECT {binned} FROM {__bin_src__} GROUP BY {group}) \
         SELECT *, {bin} + {width} AS {bin_end}, \
         {count} * 1.0 / SUM({count}) {density_window} AS {density} \
         FROM {__binned__}",
        query = query,
        bin_expr = bin_expr,
        bin_key = bin_key,
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
