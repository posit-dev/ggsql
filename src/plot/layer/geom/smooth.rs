//! Smooth geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::geom::types::get_column_name;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{ParameterValue, StatResult};
use crate::reader::SqlDialect;
use crate::{naming, GgsqlError, Mappings, Result};

/// Valid methods for smoothing
const METHOD_VALUES: &[&str] = &["nw", "nadaraya-watson", "ols", "tls"];
/// Valid kernel types for smooth density estimation
const KERNEL_VALUES: &[&str] = &[
    "gaussian",
    "epanechnikov",
    "triangular",
    "rectangular",
    "uniform",
    "biweight",
    "quartic",
    "cosine",
];

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

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("identity"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            ParamDefinition {
                name: "method",
                default: DefaultParamValue::String("nw"),
                constraint: ParamConstraint::string_option(METHOD_VALUES),
            },
            ParamDefinition {
                name: "bandwidth",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "adjust",
                default: DefaultParamValue::Number(1.0),
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "kernel",
                default: DefaultParamValue::String("gaussian"),
                constraint: ParamConstraint::string_option(KERNEL_VALUES),
            },
        ];
        PARAMS
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Column("pos1")),
                ("pos2", DefaultAestheticValue::Column("intensity")),
            ],
        }
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
        dialect: &dyn SqlDialect,
    ) -> crate::Result<super::StatResult> {
        // Get method from parameters (validated by ParamConstraint::string_option)
        let ParameterValue::String(method) = parameters.get("method").unwrap() else {
            unreachable!("method validated by ParamConstraint::string_option")
        };

        match method.as_str() {
            "nw" | "nadaraya-watson" => {
                // Smooth geom: hardcode tails=0.0 (trim exactly to data range, no extrapolation)
                let mut params = parameters.clone();
                params.insert("tails".to_string(), ParameterValue::Number(0.0));

                super::density::stat_density(
                    query,
                    aesthetics,
                    "pos1",
                    Some("pos2"),
                    group_by,
                    &params,
                    dialect,
                )
            }
            "ols" => stat_ols(query, aesthetics, group_by),
            "tls" => stat_tls(query, aesthetics, group_by),
            _ => unreachable!("method validated by ParamConstraint::string_option"),
        }
    }
}

impl std::fmt::Display for Smooth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "smooth")
    }
}

fn stat_ols(query: &str, aesthetics: &Mappings, group_by: &[String]) -> Result<StatResult> {
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

fn stat_tls(query: &str, aesthetics: &Mappings, group_by: &[String]) -> Result<StatResult> {
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

    // Compute Total Least Squares (orthogonal regression)
    // TLS minimizes perpendicular distances, not vertical distances
    // Slope: β = (Var(y) - Var(x) + sqrt((Var(y) - Var(x))² + 4*Cov(x,y)²)) / (2*Cov(x,y))
    // Where: Var(x) = E[x²] - E[x]², Var(y) = E[y²] - E[y]², Cov(x,y) = E[xy] - E[x]E[y]
    let final_query = format!(
        "WITH
        coefficients AS (
          SELECT
            {groups}AVG({x}) AS x_mean,
            AVG({y}) AS y_mean,
            AVG({x} * {y}) AS xy_mean,
            AVG({x} * {x}) AS xx_mean,
            AVG({y} * {y}) AS yy_mean,
            MIN({x}) AS x_min,
            MAX({x}) AS x_max
          FROM ({data})
          WHERE {x} IS NOT NULL AND {y} IS NOT NULL
          {group_by}
        ),
        tls_coefficients AS (
          SELECT
            {groups}x_mean,
            y_mean,
            (yy_mean - y_mean * y_mean) - (xx_mean - x_mean * x_mean) AS var_diff,
            (xy_mean - x_mean * y_mean) AS covariance,
            x_min,
            x_max
          FROM coefficients
        )
        SELECT
          {groups}x_min AS {x_out},
          (y_mean + ((var_diff + SQRT(var_diff * var_diff + 4 * covariance * covariance)) / (2 * covariance)) * (x_min - x_mean)) AS {y_out}
        FROM tls_coefficients
        UNION ALL
        SELECT
          {groups}x_max AS {x_out},
          (y_mean + ((var_diff + SQRT(var_diff * var_diff + 4 * covariance * covariance)) / (2 * covariance)) * (x_max - x_mean)) AS {y_out}
        FROM tls_coefficients",
        groups = groups_str,
        x = x_col,
        y = y_col,
        data = query,
        x_out = naming::stat_column("pos1"),
        y_out = naming::stat_column("intensity"),
        group_by = group_by_clause
    );

    Ok(StatResult::Transformed {
        query: final_query,
        stat_columns: vec!["pos1".to_string(), "intensity".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["pos1".to_string(), "pos2".to_string()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;
    use crate::reader::duckdb::DuckDBReader;
    use crate::reader::Reader;

    #[test]
    fn test_stat_ols_ungrouped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = "SELECT x, y FROM (VALUES (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)) AS t(x, y)";
        let groups: Vec<String> = vec![];

        let mut mapping = crate::Mappings::new();
        mapping.aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::Column {
                name: "x".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );
        mapping.aesthetics.insert(
            "pos2".to_string(),
            AestheticValue::Column {
                name: "y".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );

        let result = stat_ols(query, &mapping, &groups).expect("stat_ols should succeed");

        if let StatResult::Transformed {
            query: sql,
            stat_columns,
            ..
        } = result
        {
            assert_eq!(stat_columns, vec!["pos1", "intensity"]);

            let df = reader.execute_sql(&sql).expect("SQL should execute");

            // Should have 2 rows (min and max x)
            assert_eq!(df.height(), 2);
            assert_eq!(
                df.get_column_names(),
                vec!["__ggsql_stat_pos1", "__ggsql_stat_intensity"]
            );
        } else {
            panic!("Expected Transformed result");
        }
    }

    #[test]
    fn test_stat_ols_grouped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = "SELECT x, y, category FROM (VALUES
            (1.0, 2.0, 'A'), (2.0, 4.0, 'A'), (3.0, 6.0, 'A'),
            (1.0, 3.0, 'B'), (2.0, 5.0, 'B'), (3.0, 7.0, 'B')
        ) AS t(x, y, category)";
        let groups = vec!["category".to_string()];

        let mut mapping = crate::Mappings::new();
        mapping.aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::Column {
                name: "x".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );
        mapping.aesthetics.insert(
            "pos2".to_string(),
            AestheticValue::Column {
                name: "y".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );

        let result = stat_ols(query, &mapping, &groups).expect("stat_ols should succeed");

        if let StatResult::Transformed {
            query: sql,
            stat_columns,
            ..
        } = result
        {
            assert_eq!(stat_columns, vec!["pos1", "intensity"]);

            let df = reader.execute_sql(&sql).expect("SQL should execute");

            // Should have 4 rows (2 points × 2 groups)
            assert_eq!(df.height(), 4);
            assert_eq!(
                df.get_column_names(),
                vec!["category", "__ggsql_stat_pos1", "__ggsql_stat_intensity"]
            );
        } else {
            panic!("Expected Transformed result");
        }
    }

    #[test]
    fn test_stat_tls_ungrouped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = "SELECT x, y FROM (VALUES (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)) AS t(x, y)";
        let groups: Vec<String> = vec![];

        let mut mapping = crate::Mappings::new();
        mapping.aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::Column {
                name: "x".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );
        mapping.aesthetics.insert(
            "pos2".to_string(),
            AestheticValue::Column {
                name: "y".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );

        let result = stat_tls(query, &mapping, &groups).expect("stat_tls should succeed");

        if let StatResult::Transformed {
            query: sql,
            stat_columns,
            ..
        } = result
        {
            assert_eq!(stat_columns, vec!["pos1", "intensity"]);

            let df = reader.execute_sql(&sql).expect("SQL should execute");

            // Should have 2 rows (min and max x)
            assert_eq!(df.height(), 2);
            assert_eq!(
                df.get_column_names(),
                vec!["__ggsql_stat_pos1", "__ggsql_stat_intensity"]
            );
        } else {
            panic!("Expected Transformed result");
        }
    }

    #[test]
    fn test_stat_tls_grouped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = "SELECT x, y, category FROM (VALUES
            (1.0, 2.0, 'A'), (2.0, 4.0, 'A'), (3.0, 6.0, 'A'),
            (1.0, 3.0, 'B'), (2.0, 5.0, 'B'), (3.0, 7.0, 'B')
        ) AS t(x, y, category)";
        let groups = vec!["category".to_string()];

        let mut mapping = crate::Mappings::new();
        mapping.aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::Column {
                name: "x".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );
        mapping.aesthetics.insert(
            "pos2".to_string(),
            AestheticValue::Column {
                name: "y".to_string(),
                original_name: None,
                is_dummy: false,
            },
        );

        let result = stat_tls(query, &mapping, &groups).expect("stat_tls should succeed");

        if let StatResult::Transformed {
            query: sql,
            stat_columns,
            ..
        } = result
        {
            assert_eq!(stat_columns, vec!["pos1", "intensity"]);

            let df = reader.execute_sql(&sql).expect("SQL should execute");

            // Should have 4 rows (2 points × 2 groups)
            assert_eq!(df.height(), 4);
            assert_eq!(
                df.get_column_names(),
                vec!["category", "__ggsql_stat_pos1", "__ggsql_stat_intensity"]
            );
        } else {
            panic!("Expected Transformed result");
        }
    }
}
