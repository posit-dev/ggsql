//! Density geom implementation

use super::types::POSITION_VALUES;
use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::{
    naming,
    plot::{
        geom::types::get_column_name, DefaultAestheticValue, DefaultParamValue, ParamConstraint,
        ParamDefinition, ParameterValue, StatResult,
    },
    reader::SqlDialect,
    GgsqlError, Mappings, Result,
};
use std::collections::HashMap;

/// Gaussian kernel normalization constant: 1/sqrt(2*pi)
/// Precomputed at compile time to avoid repeated SQRT and PI() calls in SQL
const GAUSSIAN_NORM: f64 = 0.3989422804014327; // 1.0 / (2.0 * std::f64::consts::PI).sqrt()

/// Valid kernel types for density estimation
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

/// Density geom - kernel density estimation
#[derive(Debug, Clone, Copy)]
pub struct Density;

impl GeomTrait for Density {
    fn geom_type(&self) -> GeomType {
        GeomType::Density
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
                ("pos2", DefaultAestheticValue::Delayed), // Computed by stat
                ("pos2end", DefaultAestheticValue::Delayed),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("identity"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
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
                ("pos2", DefaultAestheticValue::Column("density")),
                ("pos2end", DefaultAestheticValue::Number(0.0)),
            ],
        }
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["pos1", "density", "intensity"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["pos1", "weight"]
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
        // Density geom: no tails limit (don't set tails parameter, defaults to None)
        stat_density(
            query, aesthetics, "pos1", None, group_by, parameters, dialect,
        )
    }
}

impl std::fmt::Display for Density {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "density")
    }
}

// Helper to add trailing comma to non-empty strings
fn with_trailing_comma(s: &str) -> String {
    if s.is_empty() {
        String::new()
    } else {
        format!("{}, ", s)
    }
}

// Helper to add leading comma to non-empty strings
fn with_leading_comma(s: &str) -> String {
    if s.is_empty() {
        String::new()
    } else {
        format!(", {}", s)
    }
}

pub(crate) fn stat_density(
    query: &str,
    aesthetics: &Mappings,
    value_aesthetic: &str,
    smooth_aesthetic: Option<&str>,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn SqlDialect,
) -> Result<StatResult> {
    let value = get_column_name(aesthetics, value_aesthetic).ok_or_else(|| {
        GgsqlError::ValidationError(format!(
            "Density requires '{}' aesthetic mapping",
            value_aesthetic
        ))
    })?;
    let smooth = smooth_aesthetic.and_then(|smth| get_column_name(aesthetics, smth));
    let weight = get_column_name(aesthetics, "weight");

    // Get tails parameter (None = unlimited)
    let tails = match parameters.get("tails") {
        Some(ParameterValue::Number(n)) => Some(*n),
        _ => None,
    };

    let bw_cte = density_sql_bandwidth(query, group_by, &value, parameters, dialect);
    let data_cte = build_data_cte(
        &value,
        smooth.as_deref(),
        weight.as_deref(),
        query,
        group_by,
    );
    let grid_cte = build_grid_cte(group_by, 512, tails, dialect);
    let kernel = choose_kde_kernel(parameters, smooth)?;

    let density_query = compute_density(
        value_aesthetic,
        group_by,
        kernel,
        &bw_cte,
        &data_cte,
        &grid_cte,
    );

    let mut consumed = vec![value_aesthetic.to_string()];
    // Stat columns produced: grid position (x), intensity (unnormalized), and density (normalized)
    let stats = vec![
        value_aesthetic.to_string(),
        "intensity".to_string(),
        "density".to_string(),
    ];
    if weight.is_some() {
        consumed.push("weight".to_string());
    }

    Ok(StatResult::Transformed {
        query: density_query,
        stat_columns: stats,
        dummy_columns: vec![],
        consumed_aesthetics: consumed,
    })
}

fn density_sql_bandwidth(
    from: &str,
    groups: &[String],
    value: &str,
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn SqlDialect,
) -> String {
    let adjust = match parameters.get("adjust") {
        Some(ParameterValue::Number(adj)) => *adj,
        _ => 1.0,
    };

    // Preformat the bandwidth expression (either explicit or computed via Silverman's rule)
    let bw_expr = if let Some(ParameterValue::Number(num)) = parameters.get("bandwidth") {
        format!("{}", num * adjust)
    } else {
        silverman_rule(adjust, value, from, groups, dialect)
    };

    // Preformat groups and GROUP BY clause together
    let (groups_select, group_by) = if groups.is_empty() {
        (String::new(), String::new())
    } else {
        let quoted_groups: Vec<String> = groups.iter().map(|g| naming::quote_ident(g)).collect();
        let groups_str = quoted_groups.join(", ");
        (
            format!("\n      {},", groups_str),
            format!("\n    GROUP BY {}", groups_str),
        )
    };

    let quoted_value = naming::quote_ident(value);
    format!(
        "WITH RECURSIVE
          bandwidth AS (
            SELECT
              {bw_expr} AS bw,{groups_select}
              MIN({value}) AS x_min,
              MAX({value}) AS x_max
            FROM ({from}) AS \"__ggsql_qt__\"
            WHERE {value} IS NOT NULL{group_by}
          )",
        bw_expr = bw_expr,
        groups_select = groups_select,
        value = quoted_value,
        from = from,
        group_by = group_by
    )
}

fn silverman_rule(
    adjust: f64,
    value_column: &str,
    from: &str,
    groups: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    // The query computes Silverman's rule of thumb (R's `stats::bw.nrd0()`).
    // We absorb the adjustment in the 0.9 multiplier of the rule
    let adjust = 0.9 * adjust;
    let v = naming::quote_ident(value_column);
    let stddev = format!("SQRT(AVG({v}*{v}) - AVG({v})*AVG({v}))", v = v);
    let q75 = dialect.sql_percentile(value_column, 0.75, from, groups);
    let q25 = dialect.sql_percentile(value_column, 0.25, from, groups);
    let iqr = format!("({q75} - {q25}) / 1.34");
    let min_expr = dialect.sql_least(&[&stddev, &iqr]);
    format!("{adjust} * {min_expr} * POW(COUNT(*), -0.2)")
}

fn choose_kde_kernel(
    parameters: &HashMap<String, ParameterValue>,
    smooth: Option<String>,
) -> Result<String> {
    let kernel = match parameters.get("kernel") {
        Some(ParameterValue::String(krnl)) => krnl.as_str(),
        _ => {
            return Err(GgsqlError::ValidationError(
                "The density's `kernel` parameter must be a string.".to_string(),
            ))
        }
    };

    // Shorthand
    let u2 = "(grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw)";
    let u_abs = "ABS(grid.x - data.val) / bandwidth.bw";

    let kernel = match kernel {
        // Gaussian: K(u) = (1/sqrt(2π)) * exp(-0.5u²)
        "gaussian" => format!("(EXP(-0.5 * {u2})) * {norm}", u2 = u2, norm = GAUSSIAN_NORM),
        // Epanechnikov: K(u) = 0.75 * (1 - u²) for |u| ≤ 1
        "epanechnikov" => format!(
            "CASE WHEN {u_abs} <= 1 THEN 0.75 * (1 - {u2}) ELSE 0 END",
            u_abs = u_abs, u2 = u2
        ),
        //  Triangular: K(u) = (1 - |u|) for |u| ≤ 1
        "triangular" => format!(
            "CASE WHEN {u_abs} <= 1 THEN 1 - {u_abs} ELSE 0 END",
            u_abs = u_abs
        ),
        // Rectangular/Uniform: K(u) = 0.5 for |u| ≤ 1
        "rectangular" | "uniform" => {
            format!("CASE WHEN {u_abs} <= 1 THEN 0.5 ELSE 0 END", u_abs = u_abs)
        }
        // Biweight = K(u) = (15/16) * (1 - u²)² for |u| ≤ 1
        "biweight" | "quartic" => format!(
            "CASE WHEN {u_abs} <= 1 THEN (15.0/16.0) * POW(1 - {u2}, 2) ELSE 0 END",
            u_abs = u_abs, u2 = u2
        ),
        // Cosine: K(u) = (π/4) * cos(πu/2) for |u| ≤ 1
        "cosine" => format!(
            "CASE WHEN {u_abs} <= 1 THEN 0.7853981633974483 * COS(1.5707963267948966 * {u_abs}) ELSE 0 END",
            u_abs = u_abs
        ),
        _ => {
            return Err(GgsqlError::ValidationError(format!(
            "The density's `kernel` parameter must be one of \"gaussian\", \"epanechnikov\", \"triangular\",
            \"rectangular\", \"uniform\", \"biweight\", \"quartic\", \"cosine\", not {kernel}.",
            kernel = kernel
        )));
        }
    };
    if let Some(smth) = smooth {
        // Nadaraya-Watson estimator: E[Y|X=x] = Σ(K((x-xi)/h) × yi) / Σ(K((x-xi)/h))
        // The bandwidth h cancels in the numerator and denominator ratio
        Ok(format!(
            "SUM(data.weight * ({kernel}) * ({smth})) / SUM(data.weight * ({kernel}))",
            kernel = kernel,
            smth = smth
        ))
    } else {
        // Use weighted sum for density computation
        // Weighted: density = (1/h) × Σ(wi × K((x-xi)/h)) / Σwi
        Ok(format!(
            "SUM(data.weight * ({kernel})) / MIN(bandwidth.bw)",
            kernel = kernel
        ))
    }
}

fn build_data_cte(
    value: &str,
    smooth: Option<&str>,
    weight: Option<&str>,
    from: &str,
    group_by: &[String],
) -> String {
    // Include weight column if provided, otherwise default to 1.0
    let weight_col = if let Some(w) = weight {
        format!(", {} AS weight", naming::quote_ident(w))
    } else {
        ", 1.0 AS weight".to_string()
    };
    let smooth_col = if let Some(s) = smooth {
        format!(", {}", naming::quote_ident(s))
    } else {
        "".to_string()
    };

    let quoted_value = naming::quote_ident(value);
    // Only filter out nulls in value column, keep NULLs in group columns
    let mut filter_valid = format!("{} IS NOT NULL", quoted_value);
    if let Some(s) = smooth {
        filter_valid = format!(
            "{filter} AND {} IS NOT NULL",
            naming::quote_ident(s),
            filter = filter_valid,
        );
    }

    let quoted_groups: Vec<String> = group_by.iter().map(|g| naming::quote_ident(g)).collect();
    format!(
        "data AS (
          SELECT {groups}{value} AS val{weight_col}{smooth_col}
          FROM ({from})
          WHERE {filter_valid}
        )",
        groups = with_trailing_comma(&quoted_groups.join(", ")),
        value = quoted_value,
        weight_col = weight_col,
        smooth_col = smooth_col,
        from = from,
        filter_valid = filter_valid
    )
}

fn build_grid_cte(
    groups: &[String],
    n_points: usize,
    tails: Option<f64>,
    dialect: &dyn SqlDialect,
) -> String {
    let has_groups = !groups.is_empty();
    let n_points_minus_1 = n_points - 1; // For formula: n-1 divisions between n points

    // Generate sequence CTE using dialect-specific SQL
    let seq_cte = dialect.sql_generate_series(n_points);

    // Shared: global_range CTE (computes range dynamically from bandwidth table)
    let global_range_cte = "global_range AS (
          SELECT
            MIN(x_min) AS min,
            MAX(x_max) AS max,
            3 * MAX(bw) AS expansion
          FROM bandwidth
        )";

    // Shared: x-coordinate formula
    let x_formula = format!(
        "(global.min - global.expansion) + (seq.n * ((global.max - global.min) + 2 * global.expansion) / {n_points})",
        n_points = n_points_minus_1
    );

    // Build base grid CTE
    let base_grid_cte = if !has_groups {
        // Simple grid without groups
        format!(
            "grid AS (
          SELECT {x_formula} AS x
          FROM global_range AS global
          CROSS JOIN \"__ggsql_seq__\" AS seq
        )",
            x_formula = x_formula
        )
    } else {
        let quoted_groups: Vec<String> = groups.iter().map(|g| naming::quote_ident(g)).collect();
        let groups_str = quoted_groups.join(", ");
        // When tails is specified, create full_grid; otherwise create grid directly
        let cte_name = if tails.is_some() { "full_grid" } else { "grid" };
        format!(
            "{cte_name} AS (
          SELECT
            {groups},
            {x_formula} AS x
          FROM global_range AS global
          CROSS JOIN \"__ggsql_seq__\" AS seq
          CROSS JOIN (SELECT DISTINCT {groups} FROM bandwidth) AS groups
        )",
            cte_name = cte_name,
            groups = groups_str,
            x_formula = x_formula
        )
    };

    // If tails is specified with groups, add the trimmed grid CTE
    if let Some(extent) = tails {
        if has_groups {
            let bandwidth_join_conds: Vec<String> = groups
                .iter()
                .map(|g| {
                    let q = naming::quote_ident(g);
                    format!("full_grid.{q} IS NOT DISTINCT FROM bandwidth.{q}")
                })
                .collect();
            let grid_groups_select: Vec<String> = groups
                .iter()
                .map(|g| format!("full_grid.{}", naming::quote_ident(g)))
                .collect();

            format!(
                "{seq_cte},
        {global_range_cte},
        {base_grid_cte},
        grid AS (
          SELECT {grid_groups}, full_grid.x
          FROM full_grid
          INNER JOIN bandwidth ON {bandwidth_join_conds}
          WHERE full_grid.x >= bandwidth.x_min - {extent} * bandwidth.bw
            AND full_grid.x <= bandwidth.x_max + {extent} * bandwidth.bw
        )",
                seq_cte = seq_cte,
                global_range_cte = global_range_cte,
                base_grid_cte = base_grid_cte,
                grid_groups = grid_groups_select.join(", "),
                bandwidth_join_conds = bandwidth_join_conds.join(" AND "),
                extent = extent
            )
        } else {
            // No groups but tail_extent specified - not meaningful, treat as no tail_extent
            format!(
                "{seq_cte},
        {global_range_cte},
        {base_grid_cte}",
                seq_cte = seq_cte,
                global_range_cte = global_range_cte,
                base_grid_cte = base_grid_cte
            )
        }
    } else {
        format!(
            "{seq_cte},
        {global_range_cte},
        {base_grid_cte}",
            seq_cte = seq_cte,
            global_range_cte = global_range_cte,
            base_grid_cte = base_grid_cte
        )
    }
}

fn compute_density(
    value_aesthetic: &str,
    group_by: &[String],
    kernel: String,
    bandwidth_cte: &str,
    data_cte: &str,
    grid_cte: &str,
) -> String {
    // Build bandwidth join condition (NULL-safe)
    let bandwidth_conditions = if group_by.is_empty() {
        "true".to_string()
    } else {
        group_by
            .iter()
            .map(|g| {
                let q = naming::quote_ident(g);
                format!("data.{q} IS NOT DISTINCT FROM bandwidth.{q}")
            })
            .collect::<Vec<String>>()
            .join(" AND ")
    };

    // Build WHERE clause to match grid to data groups (NULL-safe)
    let matching_groups = if group_by.is_empty() {
        String::new()
    } else {
        let grid_data_conds: Vec<String> = group_by
            .iter()
            .map(|g| {
                let q = naming::quote_ident(g);
                format!("grid.{q} IS NOT DISTINCT FROM data.{q}")
            })
            .collect();
        format!("WHERE {}", grid_data_conds.join(" AND "))
    };

    let join_logic = format!(
        "FROM data
        INNER JOIN bandwidth ON {bandwidth_conditions}
        CROSS JOIN grid {matching_groups}",
        bandwidth_conditions = bandwidth_conditions,
        matching_groups = matching_groups
    );

    // Build group-related SQL fragments
    let grid_groups: Vec<String> = group_by
        .iter()
        .map(|g| format!("grid.{}", naming::quote_ident(g)))
        .collect();
    let aggregation = format!(
        "GROUP BY grid.x{grid_group_by}
        ORDER BY grid.x{grid_group_by}",
        grid_group_by = with_leading_comma(&grid_groups.join(", "))
    );

    let groups = if group_by.is_empty() {
        String::new()
    } else {
        let quoted: Vec<String> = group_by.iter().map(|g| naming::quote_ident(g)).collect();
        format!("{},", quoted.join(", "))
    };

    let x_column = naming::quote_ident(&naming::stat_column(value_aesthetic));
    let intensity_column = naming::quote_ident(&naming::stat_column("intensity"));
    let density_column = naming::quote_ident(&naming::stat_column("density"));

    // Generate the density computation query
    format!(
        "{bandwidth_cte},
        {data_cte},
        {grid_cte}
        SELECT
          {x_column},
          {groups}
          {intensity_column},
          {intensity_column} / \"__norm\" AS {density_column}
        FROM (
          SELECT
            grid.x AS {x_column},
            {grid_groups}
            {kernel} AS {intensity_column},
            SUM(data.weight) AS \"__norm\"
          {join_logic}
          {aggregation}
        )",
        bandwidth_cte = bandwidth_cte,
        data_cte = data_cte,
        grid_cte = grid_cte,
        x_column = x_column,
        groups = groups,
        intensity_column = intensity_column,
        density_column = density_column,
        aggregation = aggregation,
        grid_groups = with_trailing_comma(&grid_groups.join(", "))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::duckdb::DuckDBReader;
    use crate::reader::AnsiDialect;
    use crate::reader::Reader;

    #[test]
    fn test_density_sql_no_groups() {
        let query = "SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x)";
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(0.5));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);
        let data_cte = build_data_cte("x", None, None, query, &groups);
        let grid_cte = build_grid_cte(&groups, 512, None, &AnsiDialect);
        let kernel = choose_kde_kernel(&parameters, None).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        let expected = r#"WITH RECURSIVE
          bandwidth AS (
            SELECT
              0.5 AS bw,
              MIN("x") AS x_min,
              MAX("x") AS x_max
            FROM (SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x)) AS "__ggsql_qt__"
            WHERE "x" IS NOT NULL
          ),
        data AS (
          SELECT "x" AS val, 1.0 AS weight
          FROM (SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x))
          WHERE "x" IS NOT NULL
        ),
        "__ggsql_base__"(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM "__ggsql_base__" WHERE n < 7),"__ggsql_seq__"(n) AS (SELECT CAST(a.n * 64 + b.n * 8 + c.n AS REAL) AS n FROM "__ggsql_base__" a, "__ggsql_base__" b, "__ggsql_base__" c WHERE a.n * 64 + b.n * 8 + c.n < 512),
        global_range AS (
          SELECT MIN(x_min) AS min, MAX(x_max) AS max, 3 * MAX(bw) AS expansion
          FROM bandwidth
        ),
        grid AS (
          SELECT (global.min - global.expansion) + (seq.n * ((global.max - global.min) + 2 * global.expansion) / 511) AS x
          FROM global_range AS global
          CROSS JOIN "__ggsql_seq__" AS seq
        )
        SELECT
          "__ggsql_stat_x",
          "__ggsql_stat_intensity",
          "__ggsql_stat_intensity" / "__norm" AS "__ggsql_stat_density"
        FROM (
          SELECT
            grid.x AS "__ggsql_stat_x",
            SUM(data.weight * ((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327)) / MIN(bandwidth.bw) AS "__ggsql_stat_intensity",
            SUM(data.weight) AS "__norm"
          FROM data
          INNER JOIN bandwidth ON true
          CROSS JOIN grid
          GROUP BY grid.x
          ORDER BY grid.x
        )"#;

        // Normalize whitespace for comparison
        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalize(&sql), normalize(expected));

        // Verify SQL executes and produces correct output shape
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql(&sql).expect("SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec![
                "__ggsql_stat_x",
                "__ggsql_stat_intensity",
                "__ggsql_stat_density"
            ]
        );
        assert_eq!(df.height(), 512); // 512 grid points
    }

    #[test]
    fn test_density_sql_with_two_groups() {
        let query = "SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category)";
        let groups = vec!["region".to_string(), "category".to_string()];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(0.5));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);
        let data_cte = build_data_cte("x", None, None, query, &groups);
        let grid_cte = build_grid_cte(&groups, 512, None, &AnsiDialect);
        let kernel = choose_kde_kernel(&parameters, None).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        let expected = r#"WITH RECURSIVE
          bandwidth AS (
            SELECT
              0.5 AS bw,
              "region", "category",
              MIN("x") AS x_min,
              MAX("x") AS x_max
            FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category)) AS "__ggsql_qt__"
            WHERE "x" IS NOT NULL
            GROUP BY "region", "category"
          ),
        data AS (
          SELECT "region", "category", "x" AS val, 1.0 AS weight
          FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category))
          WHERE "x" IS NOT NULL
        ),
        "__ggsql_base__"(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM "__ggsql_base__" WHERE n < 7),"__ggsql_seq__"(n) AS (SELECT CAST(a.n * 64 + b.n * 8 + c.n AS REAL) AS n FROM "__ggsql_base__" a, "__ggsql_base__" b, "__ggsql_base__" c WHERE a.n * 64 + b.n * 8 + c.n < 512),
        global_range AS (
          SELECT MIN(x_min) AS min, MAX(x_max) AS max, 3 * MAX(bw) AS expansion
          FROM bandwidth
        ),
        grid AS (
          SELECT
            "region", "category",
            (global.min - global.expansion) + (seq.n * ((global.max - global.min) + 2 * global.expansion) / 511) AS x
          FROM global_range AS global
          CROSS JOIN "__ggsql_seq__" AS seq
          CROSS JOIN (SELECT DISTINCT "region", "category" FROM bandwidth) AS groups
        )
        SELECT
          "__ggsql_stat_x",
          "region", "category",
          "__ggsql_stat_intensity",
          "__ggsql_stat_intensity" / "__norm" AS "__ggsql_stat_density"
        FROM (
          SELECT
            grid.x AS "__ggsql_stat_x",
            grid."region", grid."category",
            SUM(data.weight * ((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327)) / MIN(bandwidth.bw) AS "__ggsql_stat_intensity",
            SUM(data.weight) AS "__norm"
          FROM data
          INNER JOIN bandwidth ON data."region" IS NOT DISTINCT FROM bandwidth."region" AND data."category" IS NOT DISTINCT FROM bandwidth."category"
          CROSS JOIN grid
          WHERE grid."region" IS NOT DISTINCT FROM data."region" AND grid."category" IS NOT DISTINCT FROM data."category"
          GROUP BY grid.x, grid."region", grid."category"
          ORDER BY grid.x, grid."region", grid."category"
        )"#;

        // Normalize whitespace for comparison
        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalize(&sql), normalize(expected));

        // Verify SQL executes and produces correct output shape
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql(&sql).expect("SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec![
                "__ggsql_stat_x",
                "region",
                "category",
                "__ggsql_stat_intensity",
                "__ggsql_stat_density"
            ]
        );
        assert_eq!(df.height(), 1024); // 512 grid points × 2 groups

        // Verify density integrates to ~2 (one per group)
        // Compute grid spacing dynamically from actual data
        let x_col = df.column("__ggsql_stat_x").expect("x exists");
        // Cast to f64 if needed (AnsiDialect generates f32 from REAL)
        let x_col = x_col
            .cast(&polars::prelude::DataType::Float64)
            .expect("can cast to f64");
        let x_vals = x_col.f64().expect("x is f64");
        let x_min = x_vals.into_iter().flatten().fold(f64::INFINITY, f64::min);
        let x_max = x_vals
            .into_iter()
            .flatten()
            .fold(f64::NEG_INFINITY, f64::max);
        let dx = (x_max - x_min) / 511.0; // (n - 1) for 512 points

        let density_col = df
            .column("__ggsql_stat_density")
            .expect("density column exists");
        let total: f64 = density_col
            .f64()
            .expect("density is f64")
            .into_iter()
            .flatten()
            .sum();
        let integral = total * dx;

        // Should integrate to ~2 (one per group)
        assert!(
            (integral - 2.0).abs() < 0.01,
            "Density should integrate to ~2 (one per group), got {}",
            integral
        );
    }

    #[test]
    fn test_density_sql_computed_bandwidth() {
        // Test 1: No groups
        let query = "SELECT x FROM (VALUES (1.0), (2.0), (3.0), (4.0), (5.0)) AS t(x)";
        let groups: Vec<String> = vec![];
        let parameters = HashMap::new(); // No explicit bandwidth - will compute

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);

        // Verify SQL uses NTILE-based percentile subqueries
        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(bw_cte.contains("NTILE(4)"));
        assert!(bw_cte.contains("bandwidth AS"));
        // Verify the generated rule matches silverman_rule output
        let expected_rule = silverman_rule(1.0, "x", query, &groups, &AnsiDialect);
        assert!(normalize(&bw_cte).contains(&normalize(&expected_rule)));

        // Verify bandwidth computation executes
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader
            .execute_sql(&format!(
                "{}\nSELECT bw, x_min, x_max FROM bandwidth",
                bw_cte
            ))
            .expect("Bandwidth SQL should execute");

        assert_eq!(df.get_column_names(), vec!["bw", "x_min", "x_max"]);
        assert_eq!(df.height(), 1);

        // Test 2: With groups
        let query =
            "SELECT x, region FROM (VALUES (1.0, 'A'), (2.0, 'A'), (3.0, 'B')) AS t(x, region)";
        let groups = vec!["region".to_string()];

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);

        // Verify SQL uses NTILE-based percentile subqueries with grouping
        assert!(bw_cte.contains("NTILE(4)"));
        assert!(bw_cte.contains("GROUP BY \"region\""));
        let expected_rule = silverman_rule(1.0, "x", query, &groups, &AnsiDialect);
        assert!(normalize(&bw_cte).contains(&normalize(&expected_rule)));

        // Verify grouped bandwidth computation executes
        let df = reader
            .execute_sql(&format!(
                "{}\nSELECT bw, region, x_min, x_max FROM bandwidth",
                bw_cte
            ))
            .expect("Grouped bandwidth SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec!["bw", "region", "x_min", "x_max"]
        );
        assert_eq!(df.height(), 2); // Two groups: A and B
    }

    /// Helper function to test that a kernel integrates to 1
    fn test_kernel_integration(kernel_name: &str, tolerance: f64) {
        let query = "SELECT x FROM (VALUES (1.0), (2.0), (3.0), (4.0), (5.0)) AS t(x)";
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(1.0));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String(kernel_name.to_string()),
        );

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);
        let data_cte = build_data_cte("x", None, None, query, &groups);
        // Use wide range to capture essentially all density mass
        let grid_cte = build_grid_cte(&groups, 512, None, &AnsiDialect);
        let kernel = choose_kde_kernel(&parameters, None).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        // Execute query
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql(&sql).expect("SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec![
                "__ggsql_stat_x",
                "__ggsql_stat_intensity",
                "__ggsql_stat_density"
            ]
        );
        assert_eq!(df.height(), 512);

        // Compute integral using trapezoidal rule
        // Get actual grid spacing from the data (dynamically computed range)
        let x_col = df.column("__ggsql_stat_x").expect("x exists");
        // Cast to f64 if needed (AnsiDialect generates f32 from REAL)
        let x_col = x_col
            .cast(&polars::prelude::DataType::Float64)
            .expect("can cast to f64");
        let x_vals = x_col.f64().expect("x is f64");
        let x_min = x_vals.into_iter().flatten().fold(f64::INFINITY, f64::min);
        let x_max = x_vals
            .into_iter()
            .flatten()
            .fold(f64::NEG_INFINITY, f64::max);
        let dx = (x_max - x_min) / (df.height() as f64 - 1.0);

        let density_col = df.column("__ggsql_stat_density").expect("density exists");
        let total: f64 = density_col
            .f64()
            .expect("density is f64")
            .into_iter()
            .flatten()
            .sum();
        let integral = total * dx;

        // Verify all density values are non-negative
        let all_non_negative = density_col
            .f64()
            .expect("density is f64")
            .into_iter()
            .all(|v| v.map(|x| x >= 0.0).unwrap_or(true));
        assert!(
            all_non_negative,
            "All density values should be non-negative for kernel '{}'",
            kernel_name
        );

        // Verify integral is approximately 1
        assert!(
            (integral - 1.0).abs() < tolerance,
            "Density for kernel '{}' should integrate to ~1, got {} (tolerance: {})",
            kernel_name,
            integral,
            tolerance
        );
    }

    #[test]
    fn test_all_kernels_integrate_to_one() {
        let kernels = vec![
            "gaussian",
            "epanechnikov",
            "triangular",
            "rectangular",
            "uniform",
            "biweight",
            "quartic",
            "cosine",
        ];

        // Use 2e-3 tolerance to account for numerical integration error
        // Compact support kernels (rectangular, triangular) have larger truncation errors
        // due to sharp cutoffs, especially with discrete grid approximation
        for kernel in kernels {
            test_kernel_integration(kernel, 2e-3);
        }
    }

    #[test]
    fn test_kernel_invalid() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("invalid_kernel".to_string()),
        );

        let result = choose_kde_kernel(&parameters, None);

        assert!(result.is_err());
        match result {
            Err(GgsqlError::ValidationError(msg)) => {
                assert!(msg.contains("kernel"));
                assert!(msg.contains("invalid_kernel"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[test]
    fn test_weighted_vs_unweighted_density() {
        // Compare weighted and unweighted results
        let query = "SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x)";
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(0.5));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);
        let grid_cte = build_grid_cte(&groups, 100, None, &AnsiDialect);
        let kernel = choose_kde_kernel(&parameters, None).expect("kernel should be valid");

        // Unweighted (default weights of 1.0)
        let data_cte_unweighted = build_data_cte("x", None, None, query, &groups);
        let sql_unweighted = compute_density(
            "x",
            &groups,
            kernel.clone(),
            &bw_cte,
            &data_cte_unweighted,
            &grid_cte,
        );

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df_unweighted = reader
            .execute_sql(&sql_unweighted)
            .expect("SQL should execute");

        // With explicit uniform weights (should be equivalent)
        let query_weighted = "SELECT x, 1.0 AS weight FROM (VALUES (1.0), (2.0), (3.0)) AS t(x)";
        let data_cte_weighted = build_data_cte("x", None, Some("weight"), query_weighted, &groups);
        let sql_weighted =
            compute_density("x", &groups, kernel, &bw_cte, &data_cte_weighted, &grid_cte);
        let df_weighted = reader
            .execute_sql(&sql_weighted)
            .expect("SQL should execute");

        // Results should be identical (or very close due to floating point)
        let density_unweighted = df_unweighted
            .column("__ggsql_stat_density")
            .expect("density exists");
        let density_weighted = df_weighted
            .column("__ggsql_stat_density")
            .expect("density exists");

        let unweighted_values: Vec<f64> = density_unweighted
            .f64()
            .expect("f64")
            .into_iter()
            .flatten()
            .collect();
        let weighted_values: Vec<f64> = density_weighted
            .f64()
            .expect("f64")
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(unweighted_values.len(), weighted_values.len());

        // Check that all values are very close
        for (i, (u, w)) in unweighted_values
            .iter()
            .zip(weighted_values.iter())
            .enumerate()
        {
            assert!(
                (u - w).abs() < 1e-10,
                "Values at index {} should be equal: unweighted={}, weighted={}",
                i,
                u,
                w
            );
        }
    }

    #[test]
    fn test_density_with_intensity_remapping() {
        use crate::reader::duckdb::DuckDBReader;
        use crate::reader::Reader;

        // Create test data
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query that uses REMAPPING to map y to intensity instead of density
        let query = "
            SELECT x FROM (VALUES (1.0), (2.0), (3.0), (4.0), (5.0)) AS t(x)
            VISUALISE
            DRAW density
                MAPPING x AS x
                REMAPPING intensity AS y
                SETTING bandwidth => 1.0
        ";

        let spec = reader.execute(query).expect("Query should execute");

        // Debug: print what SQL was generated and what data we have
        println!("Generated stat SQL:");
        if let Some(sql) = spec.stat_sql(0) {
            println!("{}", sql);
        }

        // Get the stat-transformed data for layer 0
        let df = spec.stat_data(0).expect("Layer 0 should have stat data");
        println!("\nActual columns in stat_data: {:?}", df.get_column_names());
        println!("Number of rows: {}", df.height());

        // After remapping, stat columns are renamed to aesthetic columns
        // The stat transform produces: pos1, intensity, density
        // With REMAPPING intensity AS y, we get: __ggsql_aes_pos1__, __ggsql_aes_pos2__
        // (pos2 is mapped from intensity, not the default density)

        let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();

        // Should have pos1 and pos2 aesthetics after remapping (internal names)
        assert!(
            col_names.contains(&"__ggsql_aes_pos1__"),
            "Should have pos1 aesthetic, got: {:?}",
            col_names
        );
        assert!(
            col_names.contains(&"__ggsql_aes_pos2__"),
            "Should have pos2 aesthetic, got: {:?}",
            col_names
        );

        // Verify we have data
        assert!(df.height() > 0);

        // Verify pos2 values (from intensity) are non-negative
        let y_col = df
            .column("__ggsql_aes_pos2__")
            .expect("pos2 aesthetic exists");
        let all_non_negative = y_col
            .f64()
            .expect("y is f64")
            .into_iter()
            .all(|v| v.map(|x| x >= 0.0).unwrap_or(true));
        assert!(
            all_non_negative,
            "All y values (from intensity) should be non-negative"
        );

        println!("✓ Successfully used REMAPPING to map y to intensity instead of density");
    }

    #[test]
    #[ignore] // Run with: cargo test bench_density_performance -- --ignored --nocapture
    fn bench_density_performance() {
        use std::time::Instant;

        // Create test data with 1000 points
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE bench_data AS
                 SELECT (random() * 100)::DOUBLE as x
                 FROM generate_series(1, 1000)",
            )
            .expect("Failed to create test data");

        let query = "SELECT x FROM bench_data";
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(5.0));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters, &AnsiDialect);
        let data_cte = build_data_cte("x", None, None, query, &groups);
        let grid_cte = build_grid_cte(&groups, 512, None, &AnsiDialect);
        let kernel = choose_kde_kernel(&parameters, None).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        // Warm-up run
        reader.execute_sql(&sql).expect("Warm-up failed");

        // Benchmark runs
        const RUNS: usize = 10;
        let mut times = Vec::with_capacity(RUNS);

        for i in 0..RUNS {
            let start = Instant::now();
            reader.execute_sql(&sql).expect("Benchmark run failed");
            let duration = start.elapsed();
            times.push(duration);
            println!("Run {}: {:?}", i + 1, duration);
        }

        let avg = times.iter().sum::<std::time::Duration>() / RUNS as u32;
        let min = times.iter().min().unwrap();
        let max = times.iter().max().unwrap();

        println!("\n=== Benchmark Results (1000 data points, 512 grid points) ===");
        println!("Average: {:?}", avg);
        println!("Min:     {:?}", min);
        println!("Max:     {:?}", max);
        println!(
            "Ops:     0 SQRT/PI()/POW() calls, {} multiplications/divisions per run (optimized)",
            512_000
        );
    }
}
