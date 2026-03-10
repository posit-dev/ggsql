//! Smooth geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::geom::types::get_column_name;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{DefaultParam, ParameterValue, StatResult};
use crate::{naming, GgsqlError, Mappings, Result};

/// Smooth geom - smoothed conditional means (regression, LOESS, etc.)
#[derive(Debug, Clone, Copy)]
pub struct Smooth;

impl GeomTrait for Smooth {
    fn geom_type(&self) -> GeomType {
        GeomType::Smooth
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::String("#3366FF")),
                ("linewidth", DefaultAestheticValue::Number(2.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [super::DefaultParam] {
        &[
            DefaultParam {
                name: "method",
                default: super::DefaultParamValue::String("nw"),
            },
            DefaultParam {
                name: "bandwidth",
                default: super::DefaultParamValue::Null,
            },
            DefaultParam {
                name: "adjust",
                default: super::DefaultParamValue::Number(1.0),
            },
            DefaultParam {
                name: "kernel",
                default: super::DefaultParamValue::String("gaussian"),
            },
        ]
    }

    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[
            ("pos1", DefaultAestheticValue::Column("pos1")),
            ("pos2", DefaultAestheticValue::Column("intensity")),
        ]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
    ) -> crate::Result<super::StatResult> {
        let Some(ParameterValue::String(method)) = parameters.get("method") else {
            return Err(GgsqlError::ValidationError(
                "The `method` setting must be a string.".to_string(),
            ));
        };

        match method.as_str() {
            "nw" | "nadaraya-watson" => super::density::stat_density(
                query,
                aesthetics,
                "pos1",
                Some("pos2"),
                group_by,
                parameters,
                execute_query,
            ),
            "ols" => stat_lm(query, aesthetics, group_by),
            _ => Err(GgsqlError::ValidationError(
                "The `method` setting must be 'nw' or 'lm'.".to_string(),
            )),
        }
    }

    // Note: stat_smooth not yet implemented - will return Identity for now
}

impl std::fmt::Display for Smooth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "smooth")
    }
}

fn stat_lm(query: &str, aesthetics: &Mappings, group_by: &[String]) -> Result<StatResult> {
    let x_col = get_column_name(aesthetics, "pos1").ok_or_else(|| {
        GgsqlError::ValidationError("Smooth requires 'pos1' aesthetic".to_string())
    })?;
    let y_col = get_column_name(aesthetics, "pos2").ok_or_else(|| {
        GgsqlError::ValidationError("Smooth requires 'pos2' aesthetic".to_string())
    })?;

    // Build group-related SQL fragments
    let (groups_str, group_by_clause) = if group_by.is_empty() {
        (String::new(), String::new())
    } else {
        (
            format!("{}, ", group_by.join(", ")),
            format!("GROUP BY {}", group_by.join(", ")),
        )
    };

    // Compute regression coefficients and predict at min and max x values
    // We use UNION ALL to get two rows per group (one for x_min, one for x_max)
    // Slope: (E[XY] - E[X]E[Y]) / (E[X²] - E[X]²)
    // Fitted: E[Y] + slope * (x - E[X])
    let final_query = format!(
        "WITH
        coefficients AS (
          SELECT
            {groups}AVG({x}) AS x_mean,
            AVG({y}) AS y_mean,
            AVG({x} * {y}) AS xy_mean,
            AVG({x} * {x}) AS xx_mean,
            MIN({x}) AS x_min,
            MAX({x}) AS x_max
          FROM ({data})
          WHERE {x} IS NOT NULL AND {y} IS NOT NULL
          {group_by}
        )
        SELECT
          {groups}x_min AS {x_out},
          (y_mean + ((xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)) * (x_min - x_mean)) AS {y_out}
        FROM coefficients
        UNION ALL
        SELECT
          {groups}x_max AS {x_out},
          (y_mean + ((xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)) * (x_max - x_mean)) AS {y_out}
        FROM coefficients",
        groups = groups_str,
        x = x_col,
        y = y_col,
        data = query,
        x_out = naming::stat_column("pos1"),
        y_out = naming::stat_column("intensity"), // We name this 'intensity' to be consistent with the nadaraya-watson kernel
        group_by = group_by_clause
    );

    Ok(StatResult::Transformed {
        query: final_query,
        stat_columns: vec!["pos1".to_string(), "intensity".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["pos1".to_string(), "pos2".to_string()],
    })
}
