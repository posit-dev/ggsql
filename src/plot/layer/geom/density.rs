//! Density geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};
use crate::{
    plot::{
        geom::types::get_column_name, DefaultParam, DefaultParamValue, ParameterValue, StatResult,
    },
    GgsqlError, Mappings, Result,
};
use std::collections::HashMap;

/// Density geom - kernel density estimation
#[derive(Debug, Clone, Copy)]
pub struct Density;

impl GeomTrait for Density {
    fn geom_type(&self) -> GeomType {
        GeomType::Density
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "color", "colour", "fill", "stroke", "opacity"],
            required: &["x"],
            hidden: &["density"],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
            DefaultParam {
                name: "bandwidth",
                default: DefaultParamValue::Null,
            },
            DefaultParam {
                name: "adjust",
                default: DefaultParamValue::Number(1.0),
            },
        ]
    }

    fn default_remappings(&self) -> &'static [(&'static str, &'static str)] {
        &[("density", "y")]
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
        stat_density(query, aesthetics, group_by, parameters, execute_query)
    }

    // Note: stat_density not yet implemented - will return Identity for now
}

impl std::fmt::Display for Density {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "density")
    }
}

fn stat_density(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    execute: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
) -> Result<StatResult> {
    let x = get_column_name(aesthetics, "x").ok_or_else(|| {
        GgsqlError::ValidationError("Density requires 'x' aesthetic mapping".to_string())
    })?;

    let bw = density_sql_bandwidth(query, group_by, &x, parameters);

    Ok(StatResult::Transformed {
        query: "".to_string(),
        stat_columns: vec!["x".to_string(), "density".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["x".to_string()],
    })
}

fn density_sql_bandwidth(
    from: &str,
    groups: &[String],
    value: &str,
    parameters: &HashMap<String, ParameterValue>,
) -> String {
    // We have to do a little bit of torturous formatting to get the
    // absence or presence of groups right.
    let mut partition = String::new();
    let mut group_by = String::new();
    let mut comma = String::new();
    let groups = groups.join(", ");

    if !groups.is_empty() {
        partition = format!("PARTITION BY {} ", groups);
        group_by = format!("GROUP BY {}", groups);
        comma = ",".to_string()
    }

    let adjust = match parameters.get("adjust") {
        Some(ParameterValue::Number(adj)) => *adj,
        _ => 1.0,
    };

    if let Some(ParameterValue::Number(mut num)) = parameters.get("bandwidth") {
        // When we have a user-supplied bandwidth, we don't have to compute the
        // bandwidth from the data. Instead, we just make sure the query has
        // the right shape.
        num = num * adjust;
        let cte = if groups.is_empty() {
            format!("WITH bandwidth AS (SELECT {num} AS bw)", num = num)
        } else {
            format!(
                "WITH bandwidth AS (SELECT {num} AS bw, {groups} FROM ({from}) {group_by})",
                num = num,
                groups = groups,
                group_by = group_by
            )
        };
        return cte;
    }

    // The query computes Silverman's rule of thumb (R's `stats::bw.nrd0()`).
    // We absorb the adjustment in the 0.9 multiplier of the rule
    let adjust = adjust * 0.9;
    // Most complexity here comes from trying to compute quartiles in a
    // SQL dialect-agnostic fashion.
    format!(
        "WITH 
          quartiles AS (
            SELECT
              {value},
              NTILE(4) OVER ({partition}ORDER BY {value} ASC) AS _Q{comma}
              {groups}
            FROM ({from})
            WHERE {value} IS NOT NULL
          ),
          metrics AS (
            SELECT
              (MAX(CASE WHEN _Q = 3 THEN {value} END) + MIN(CASE WHEN _Q = 4 THEN {value} END)) / 2.0 -
              (MAX(CASE WHEN _Q = 1 THEN {value} END) + MIN(CASE WHEN _Q = 2 THEN {value} END)) / 2.0 AS iqr,
              COUNT(*) AS n,
              STDDEV({value}) AS sd{comma}
              {groups}
            FROM quartiles
            {group_by}
          ),
          bandwidth AS (
            SELECT
              {adjust} * LEAST(sd, iqr / 1.34) * POWER(n, -0.2) AS bw{comma}
              {groups}
            FROM metrics
          )",
      value = value,
      partition = partition,
      group_by = group_by,
      groups = groups,
      comma = comma,
      from = from,
      adjust = adjust
    )
}
