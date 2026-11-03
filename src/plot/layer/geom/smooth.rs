//! Smooth geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::geom::types::{get_column_name, get_quoted_column_name};
use crate::plot::types::{CastTargetType, DefaultAestheticValue, Parameters};
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
        schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &Parameters,
        _execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        dialect: &dyn SqlDialect,
        aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> crate::Result<super::StatResult> {
        // Get method from parameters (validated by ParamConstraint::string_option)
        let ParameterValue::String(method) = parameters.get("method").unwrap() else {
            unreachable!("method validated by ParamConstraint::string_option")
        };

        // Temporal x columns are computed as epoch-microsecond numbers and
        // converted back to timestamps on output.
        let x_temporal = get_column_name(aesthetics, "pos1")
            .and_then(|col| super::types::temporal_kind_of(schema, &col));

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
                    aesthetic_ctx,
                    x_temporal,
                )
            }
            "ols" => stat_ols(
                query,
                aesthetics,
                group_by,
                aesthetic_ctx,
                dialect,
                x_temporal,
            ),
            "tls" => stat_tls(
                query,
                aesthetics,
                group_by,
                aesthetic_ctx,
                dialect,
                x_temporal,
            ),
            _ => unreachable!("method validated by ParamConstraint::string_option"),
        }
    }
}

impl std::fmt::Display for Smooth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "smooth")
    }
}

/// Prepare the x expression for regression stats and the conversion of the
/// endpoint columns back to the axis type. For temporal x, the computation
/// runs on epoch-microsecond numbers and the endpoints convert back to
/// timestamps; for numeric x both are identity.
fn temporal_x_exprs(
    x_col: &str,
    x_temporal: Option<CastTargetType>,
    dialect: &dyn SqlDialect,
) -> (String, String, String) {
    match x_temporal {
        Some(kind) => (
            super::types::temporal_as_epoch_micros(dialect, x_col, kind),
            dialect.sql_number_as_temporal("x_min", CastTargetType::DateTime),
            dialect.sql_number_as_temporal("x_max", CastTargetType::DateTime),
        ),
        None => (x_col.to_string(), "x_min".to_string(), "x_max".to_string()),
    }
}

fn stat_ols(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    dialect: &dyn SqlDialect,
    x_temporal: Option<CastTargetType>,
) -> Result<StatResult> {
    let x_col = get_quoted_column_name(aesthetics, "pos1", dialect).ok_or_else(|| {
        let name = aesthetic_ctx.map_internal_to_user("pos1");
        GgsqlError::ValidationError(format!("Smooth requires '{}' aesthetic mapping", name))
    })?;
    let y_col = get_quoted_column_name(aesthetics, "pos2", dialect).ok_or_else(|| {
        let name = aesthetic_ctx.map_internal_to_user("pos2");
        GgsqlError::ValidationError(format!("Smooth requires '{}' aesthetic mapping", name))
    })?;

    let (x_sql, x_min_out, x_max_out) = temporal_x_exprs(&x_col, x_temporal, dialect);

    // Build group-related SQL fragments
    let (groups_str, group_by_clause) = if group_by.is_empty() {
        (String::new(), String::new())
    } else {
        (
            format!(
                "{}, ",
                group_by
                    .iter()
                    .map(|c| dialect.quote_ident(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "GROUP BY {}",
                group_by
                    .iter()
                    .map(|c| dialect.quote_ident(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
          FROM ({data}) AS {data_alias}
          WHERE {x} IS NOT NULL AND {y} IS NOT NULL
          {group_by}
        )
        SELECT
          {groups}{x_min_out} AS {x_out},
          (y_mean + ((xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)) * (x_min - x_mean)) AS {y_out}
        FROM coefficients
        UNION ALL
        SELECT
          {groups}{x_max_out} AS {x_out},
          (y_mean + ((xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)) * (x_max - x_mean)) AS {y_out}
        FROM coefficients",
        groups = groups_str,
        x = x_sql,
        y = y_col,
        data = query,
        data_alias = dialect.quote_ident("__ggsql_smooth__"),
        x_out = dialect.quote_ident(&naming::stat_column("pos1")),
        y_out = dialect.quote_ident(&naming::stat_column("intensity")), // We name this 'intensity' to be consistent with the nadaraya-watson kernel
        x_min_out = x_min_out,
        x_max_out = x_max_out,
        group_by = group_by_clause
    );

    Ok(StatResult::Transformed {
        query: final_query,
        stat_columns: vec!["pos1".to_string(), "intensity".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["pos1".to_string(), "pos2".to_string()],
        layer_order: vec![],
    })
}

fn stat_tls(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    dialect: &dyn SqlDialect,
    x_temporal: Option<CastTargetType>,
) -> Result<StatResult> {
    let x_col = get_quoted_column_name(aesthetics, "pos1", dialect).ok_or_else(|| {
        let name = aesthetic_ctx.map_internal_to_user("pos1");
        GgsqlError::ValidationError(format!("Smooth requires '{}' aesthetic mapping", name))
    })?;
    let y_col = get_quoted_column_name(aesthetics, "pos2", dialect).ok_or_else(|| {
        let name = aesthetic_ctx.map_internal_to_user("pos2");
        GgsqlError::ValidationError(format!("Smooth requires '{}' aesthetic mapping", name))
    })?;

    let (x_sql, x_min_out, x_max_out) = temporal_x_exprs(&x_col, x_temporal, dialect);

    // Build group-related SQL fragments
    let (groups_str, group_by_clause) = if group_by.is_empty() {
        (String::new(), String::new())
    } else {
        (
            format!(
                "{}, ",
                group_by
                    .iter()
                    .map(|c| dialect.quote_ident(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "GROUP BY {}",
                group_by
                    .iter()
                    .map(|c| dialect.quote_ident(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
          FROM ({data}) AS {data_alias}
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
          {groups}{x_min_out} AS {x_out},
          (y_mean + ((var_diff + SQRT(var_diff * var_diff + 4 * covariance * covariance)) / (2 * covariance)) * (x_min - x_mean)) AS {y_out}
        FROM tls_coefficients
        UNION ALL
        SELECT
          {groups}{x_max_out} AS {x_out},
          (y_mean + ((var_diff + SQRT(var_diff * var_diff + 4 * covariance * covariance)) / (2 * covariance)) * (x_max - x_mean)) AS {y_out}
        FROM tls_coefficients",
        groups = groups_str,
        x = x_sql,
        y = y_col,
        data = query,
        data_alias = dialect.quote_ident("__ggsql_smooth__"),
        x_out = dialect.quote_ident(&naming::stat_column("pos1")),
        y_out = dialect.quote_ident(&naming::stat_column("intensity")),
        x_min_out = x_min_out,
        x_max_out = x_max_out,
        group_by = group_by_clause
    );

    Ok(StatResult::Transformed {
        query: final_query,
        stat_columns: vec!["pos1".to_string(), "intensity".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["pos1".to_string(), "pos2".to_string()],
        layer_order: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;
    #[cfg(feature = "duckdb")]
    use crate::reader::duckdb::DuckDBReader;
    #[cfg(feature = "duckdb")]
    use crate::reader::Reader;

    #[cfg(feature = "duckdb")]
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

        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let result = stat_ols(
            query,
            &mapping,
            &groups,
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .expect("stat_ols should succeed");

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

    #[cfg(feature = "duckdb")]
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

        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let result = stat_ols(
            query,
            &mapping,
            &groups,
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .expect("stat_ols should succeed");

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

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_stat_ols_date_x_returns_timestamps() {
        use crate::plot::types::CastTargetType;
        use crate::reader::duckdb::DuckDbDialect;
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = "SELECT d, y FROM (VALUES
            (DATE '2024-01-01', 1.0), (DATE '2024-02-01', 2.0), (DATE '2024-03-01', 3.0)
        ) AS t(d, y)";
        let groups: Vec<String> = vec![];

        let mut mapping = crate::Mappings::new();
        mapping.aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::Column {
                name: "d".to_string(),
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

        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["d", "y"], &[]);
        let result = stat_ols(
            query,
            &mapping,
            &groups,
            &ctx,
            &DuckDbDialect,
            Some(CastTargetType::Date),
        )
        .expect("stat_ols should succeed");

        if let StatResult::Transformed { query: sql, .. } = result {
            let df = reader.execute_sql(&sql).expect("SQL should execute");

            // Two rows (min and max x), with x converted back to timestamps
            // so the axis stays temporal.
            assert_eq!(df.height(), 2);
            assert_eq!(
                df.column_dtype("__ggsql_stat_pos1").unwrap(),
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None
                )
            );
        } else {
            panic!("Expected Transformed result");
        }
    }

    #[cfg(feature = "duckdb")]
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

        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let result = stat_tls(
            query,
            &mapping,
            &groups,
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .expect("stat_tls should succeed");

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

    #[cfg(feature = "duckdb")]
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

        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let result = stat_tls(
            query,
            &mapping,
            &groups,
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .expect("stat_tls should succeed");

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

    // =========================================================================
    // Internal aesthetic names must not appear in stat error messages
    // =========================================================================

    #[test]
    fn stat_ols_missing_pos_aesthetics_emits_user_facing_name() {
        let mapping = crate::Mappings::new();
        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let err = stat_ols(
            "SELECT 1",
            &mapping,
            &[],
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Validation error: Smooth requires 'x' aesthetic mapping"
        );
    }

    #[test]
    fn stat_tls_missing_pos_aesthetics_emits_user_facing_name() {
        let mapping = crate::Mappings::new();
        let ctx = crate::plot::aesthetic::AestheticContext::from_static(&["x", "y"], &[]);
        let err = stat_tls(
            "SELECT 1",
            &mapping,
            &[],
            &ctx,
            &crate::reader::AnsiDialect,
            None,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Validation error: Smooth requires 'x' aesthetic mapping"
        );
    }
}
