//! Density geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::{
    naming,
    plot::{
        geom::types::get_column_name, DefaultAestheticValue, DefaultParam, DefaultParamValue,
        ParameterValue, StatResult,
    },
    GgsqlError, Mappings, Result,
};
use std::collections::HashMap;

/// Gaussian kernel normalization constant: 1/sqrt(2*pi)
/// Precomputed at compile time to avoid repeated SQRT and PI() calls in SQL
const GAUSSIAN_NORM: f64 = 0.3989422804014327; // 1.0 / (2.0 * std::f64::consts::PI).sqrt()

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
                ("x", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
                ("y", DefaultAestheticValue::Delayed), // Computed by stat
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
            DefaultParam {
                name: "stacking",
                default: DefaultParamValue::String("off"),
            },
            DefaultParam {
                name: "bandwidth",
                default: DefaultParamValue::Null,
            },
            DefaultParam {
                name: "adjust",
                default: DefaultParamValue::Number(1.0),
            },
            DefaultParam {
                name: "kernel",
                default: DefaultParamValue::String("gaussian"),
            },
        ]
    }

    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[
            ("x", DefaultAestheticValue::Column("x")),
            ("y", DefaultAestheticValue::Column("density")),
        ]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["x", "density", "intensity"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["x", "weight"]
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
        stat_density(query, aesthetics, "x", group_by, parameters, execute_query)
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
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    execute: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
) -> Result<StatResult> {
    let value = get_column_name(aesthetics, value_aesthetic).ok_or_else(|| {
        GgsqlError::ValidationError(format!(
            "Density requires '{}' aesthetic mapping",
            value_aesthetic
        ))
    })?;
    let weight = get_column_name(aesthetics, "weight");

    let (min, max) = compute_range_sql(&value, query, execute)?;
    let bw_cte = density_sql_bandwidth(query, group_by, &value, parameters);
    let data_cte = build_data_cte(&value, weight.as_deref(), query, group_by);
    let grid_cte = build_grid_cte(group_by, query, min, max, 512);
    let kernel = choose_kde_kernel(parameters)?;
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

fn compute_range_sql(
    value: &str,
    from: &str,
    execute: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
) -> Result<(f64, f64)> {
    let query = format!(
        "SELECT
          MIN({value}) AS min,
          MAX({value}) AS max
        FROM ({from})
        WHERE {value} IS NOT NULL",
        value = value,
        from = from
    );
    let result = execute(&query)?;
    let min = result
        .column("min")
        .and_then(|col| col.get(0))
        .and_then(|v| v.try_extract::<f64>());

    let max = result
        .column("max")
        .and_then(|col| col.get(0))
        .and_then(|v| v.try_extract::<f64>());

    if let (Ok(start), Ok(end)) = (min, max) {
        if !start.is_finite() || !end.is_finite() {
            return Err(GgsqlError::ValidationError(format!(
                "Density layer needs finite numbers in '{}' column.",
                value
            )));
        }
        if (end - start).abs() < 1e-8 {
            // We need to be able to compute variance for density. Having zero
            // range is guaranteed to also have zero variance.
            return Err(GgsqlError::ValidationError(format!(
                "Density layer needs non-zero range data in '{}' column.",
                value
            )));
        }
        return Ok((start, end));
    }
    Err(GgsqlError::ReaderError(format!(
        "Density layer failed to compute range for '{}' column.",
        value
    )))
}

fn density_sql_bandwidth(
    from: &str,
    groups: &[String],
    value: &str,
    parameters: &HashMap<String, ParameterValue>,
) -> String {
    let mut group_by = String::new();
    let mut comma = String::new();
    let groups = groups.join(", ");

    if !groups.is_empty() {
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
        num *= adjust;
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
    format!(
        "WITH
          bandwidth AS (
            SELECT
              {rule} AS bw{comma}
              {groups}
            FROM ({from})
            WHERE {value} IS NOT NULL
            {group_by}
          )",
        rule = silverman_rule(adjust, value),
        value = value,
        group_by = group_by,
        groups = groups,
        comma = comma,
        from = from
    )
}

fn silverman_rule(adjust: f64, value_column: &str) -> String {
    // The query computes Silverman's rule of thumb (R's `stats::bw.nrd0()`).
    // We absorb the adjustment in the 0.9 multiplier of the rule
    let adjust = 0.9 * adjust;
    format!(
        "{adjust} * LEAST(STDDEV({value}), (QUANTILE_CONT({value}, 0.75) - QUANTILE_CONT({value}, 0.25)) / 1.34) * POWER(COUNT(*), -0.2)",
        adjust = adjust,
        value = value_column
    )
}

fn choose_kde_kernel(parameters: &HashMap<String, ParameterValue>) -> Result<String> {
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
    // Use weighted sum for density computation
    // Weighted: density = (1/h) × Σ(wi × K((x-xi)/h)) / Σwi
    Ok(format!(
        "SUM(data.weight * ({kernel})) / ANY_VALUE(bandwidth.bw)",
        kernel = kernel
    ))
}

fn build_data_cte(value: &str, weight: Option<&str>, from: &str, group_by: &[String]) -> String {
    // Include weight column if provided, otherwise default to 1.0
    let weight_col = if let Some(w) = weight {
        format!(", {} AS weight", w)
    } else {
        ", 1.0 AS weight".to_string()
    };

    // Only filter out nulls in value column, keep NULLs in group columns
    let filter_valid = format!("{} IS NOT NULL", value);

    format!(
        "data AS (
          SELECT {groups}{value} AS val{weight_col}
          FROM ({from})
          WHERE {filter_valid}
        )",
        groups = with_trailing_comma(&group_by.join(", ")),
        value = value,
        weight_col = weight_col,
        from = from,
        filter_valid = filter_valid
    )
}

fn build_grid_cte(groups: &[String], from: &str, min: f64, max: f64, n_points: usize) -> String {
    let has_groups = !groups.is_empty();
    let n_points = n_points - 1; // GENERATE_SERIES gives on point for free
    let diff = (max - min).abs();

    // Expand range 10%
    let expand = 0.1;
    let min = min - (expand * diff * 0.5);
    let max = max + (expand * diff * 0.5);
    let diff = (max - min).abs();

    if !has_groups {
        return format!(
            "grid AS (
          SELECT {min} + (seq.n * {diff} / {n_points}) AS x
          FROM GENERATE_SERIES(0, {n_points}) AS seq(n)
        )",
            min = min,
            diff = diff,
            n_points = n_points
        );
    }

    let groups = groups.join(", ");
    format!(
        "grid AS (
          SELECT
            {groups},
            {min} + (seq.n * {diff} / {n_points}) AS x
          FROM GENERATE_SERIES(0, {n_points}) AS seq(n)
          CROSS JOIN (SELECT DISTINCT {groups} FROM ({from})) AS groups
        )",
        groups = groups,
        diff = diff,
        min = min,
        n_points = n_points,
        from = from
    )
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
            .map(|g| format!("data.{col} IS NOT DISTINCT FROM bandwidth.{col}", col = g))
            .collect::<Vec<String>>()
            .join(" AND ")
    };

    // Build WHERE clause to match grid to data groups (NULL-safe)
    let matching_groups = if group_by.is_empty() {
        String::new()
    } else {
        let grid_data_conds: Vec<String> = group_by
            .iter()
            .map(|g| format!("grid.{col} IS NOT DISTINCT FROM data.{col}", col = g))
            .collect();
        format!("WHERE {}", grid_data_conds.join(" AND "))
    };

    let join_logic = format!(
        "FROM data
        INNER JOIN bandwidth ON {bandwidth_conditions}
        CROSS JOIN grid {matching_groups}",
        bandwidth_conditions = bandwidth_conditions,
        matching_groups = matching_groups,
    );

    // Build group-related SQL fragments
    let grid_groups: Vec<String> = group_by.iter().map(|g| format!("grid.{}", g)).collect();
    let aggregation = format!(
        "GROUP BY grid.x{grid_group_by}
        ORDER BY grid.x{grid_group_by}",
        grid_group_by = with_leading_comma(&grid_groups.join(", "))
    );

    let groups = if group_by.is_empty() {
        String::new()
    } else {
        format!("{},", group_by.join(", "))
    };

    // Generate the density computation query
    format!(
        "{bandwidth_cte},
        {data_cte},
        {grid_cte}
        SELECT
          {x_column},
          {groups}
          {intensity_column},
          {intensity_column} / __norm AS {density_column}
        FROM (
          SELECT
            grid.x AS {x_column},
            {grid_groups}
            {kernel} AS {intensity_column},
            SUM(data.weight) AS __norm
          {join_logic}
          {aggregation}
        )",
        bandwidth_cte = bandwidth_cte,
        data_cte = data_cte,
        grid_cte = grid_cte,
        x_column = naming::stat_column(value_aesthetic),
        groups = groups,
        intensity_column = naming::stat_column("intensity"),
        density_column = naming::stat_column("density"),
        aggregation = aggregation,
        grid_groups = with_trailing_comma(&grid_groups.join(", "))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::duckdb::DuckDBReader;
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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);
        let data_cte = build_data_cte("x", None, query, &groups);
        let grid_cte = build_grid_cte(&groups, query, 0.0, 10.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        let expected = "WITH bandwidth AS (SELECT 0.5 AS bw),
        data AS (
          SELECT x AS val, 1.0 AS weight
          FROM (SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x))
          WHERE x IS NOT NULL
        ),
        grid AS (
          SELECT -0.5 + (seq.n * 11 / 511) AS x
          FROM GENERATE_SERIES(0, 511) AS seq(n)
        )
        SELECT
          __ggsql_stat_x,
          __ggsql_stat_intensity,
          __ggsql_stat_intensity / __norm AS __ggsql_stat_density
        FROM (
          SELECT
            grid.x AS __ggsql_stat_x,
            SUM(data.weight * ((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327)) / ANY_VALUE(bandwidth.bw) AS __ggsql_stat_intensity,
            SUM(data.weight) AS __norm
          FROM data
          INNER JOIN bandwidth ON true
          CROSS JOIN grid
          GROUP BY grid.x
          ORDER BY grid.x
        )";

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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);
        let data_cte = build_data_cte("x", None, query, &groups);
        let grid_cte = build_grid_cte(&groups, query, -10.0, 10.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", &groups, kernel, &bw_cte, &data_cte, &grid_cte);

        let expected = "WITH bandwidth AS (SELECT 0.5 AS bw, region, category FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category)) GROUP BY region, category),
        data AS (
          SELECT region, category, x AS val, 1.0 AS weight
          FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category))
          WHERE x IS NOT NULL
        ),
        grid AS (
          SELECT
            region, category,
            -11 + (seq.n * 22 / 511) AS x
          FROM GENERATE_SERIES(0, 511) AS seq(n)
          CROSS JOIN (SELECT DISTINCT region, category FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category))) AS groups
        )
        SELECT
          __ggsql_stat_x,
          region, category,
          __ggsql_stat_intensity,
          __ggsql_stat_intensity / __norm AS __ggsql_stat_density
        FROM (
          SELECT
            grid.x AS __ggsql_stat_x,
            grid.region, grid.category,
            SUM(data.weight * ((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327)) / ANY_VALUE(bandwidth.bw) AS __ggsql_stat_intensity,
            SUM(data.weight) AS __norm
          FROM data
          INNER JOIN bandwidth ON data.region IS NOT DISTINCT FROM bandwidth.region AND data.category IS NOT DISTINCT FROM bandwidth.category
          CROSS JOIN grid
          WHERE grid.region IS NOT DISTINCT FROM data.region AND grid.category IS NOT DISTINCT FROM data.category
          GROUP BY grid.x, grid.region, grid.category
          ORDER BY grid.x, grid.region, grid.category
        )";

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
        // Grid spacing: (max - min) / (n - 1) = 22 / 511 ≈ 0.0430
        let dx = 22.0 / 511.0;
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

        // With wide range (-10 to 10), we capture essentially all density mass
        // Tolerance of 1e-6 - error is dominated by floating point precision
        assert!(
            (integral - 2.0).abs() < 1e-6,
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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);

        // Verify exact SQL structure uses QUANTILE_CONT
        let expected = "WITH
          bandwidth AS (
            SELECT
              0.9 * LEAST(STDDEV(x), (QUANTILE_CONT(x, 0.75) - QUANTILE_CONT(x, 0.25)) / 1.34) * POWER(COUNT(*), -0.2) AS bw
            FROM (SELECT x FROM (VALUES (1.0), (2.0), (3.0), (4.0), (5.0)) AS t(x))
            WHERE x IS NOT NULL

          )";

        // Normalize whitespace for comparison
        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalize(&bw_cte), normalize(expected));

        // Verify bandwidth computation executes
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader
            .execute_sql(&format!("{}\nSELECT bw FROM bandwidth", bw_cte))
            .expect("Bandwidth SQL should execute");

        assert_eq!(df.get_column_names(), vec!["bw"]);
        assert_eq!(df.height(), 1);

        // Test 2: With groups
        let query =
            "SELECT x, region FROM (VALUES (1.0, 'A'), (2.0, 'A'), (3.0, 'B')) AS t(x, region)";
        let groups = vec!["region".to_string()];

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);

        // Verify exact SQL structure uses QUANTILE_CONT with GROUP BY
        let expected = "WITH
          bandwidth AS (
            SELECT
              0.9 * LEAST(STDDEV(x), (QUANTILE_CONT(x, 0.75) - QUANTILE_CONT(x, 0.25)) / 1.34) * POWER(COUNT(*), -0.2) AS bw,
              region
            FROM (SELECT x, region FROM (VALUES (1.0, 'A'), (2.0, 'A'), (3.0, 'B')) AS t(x, region))
            WHERE x IS NOT NULL
            GROUP BY region
          )";

        assert_eq!(normalize(&bw_cte), normalize(expected));

        // Verify grouped bandwidth computation executes
        let df = reader
            .execute_sql(&format!("{}\nSELECT bw, region FROM bandwidth", bw_cte))
            .expect("Grouped bandwidth SQL should execute");

        assert_eq!(df.get_column_names(), vec!["bw", "region"]);
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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);
        let data_cte = build_data_cte("x", None, query, &groups);
        // Use wide range to capture essentially all density mass
        let grid_cte = build_grid_cte(&groups, query, -5.0, 15.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
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
        // Grid spacing: (max - min) / (n - 1)
        let dx = 22.0 / 511.0; // (15 - (-5) expanded by 10%) / (512 - 1)
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

        let result = choose_kde_kernel(&parameters);

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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);
        let grid_cte = build_grid_cte(&groups, query, 0.0, 4.0, 100);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");

        // Unweighted (default weights of 1.0)
        let data_cte_unweighted = build_data_cte("x", None, query, &groups);
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
        let data_cte_weighted = build_data_cte("x", Some("weight"), query_weighted, &groups);
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
        // The stat transform produces: x, intensity, density
        // With REMAPPING intensity AS y, we get: __ggsql_aes_x__, __ggsql_aes_y__
        // (y is mapped from intensity, not the default density)

        let col_names: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();

        // Should have x and y aesthetics after remapping
        assert!(
            col_names.contains(&"__ggsql_aes_x__"),
            "Should have x aesthetic, got: {:?}",
            col_names
        );
        assert!(
            col_names.contains(&"__ggsql_aes_y__"),
            "Should have y aesthetic, got: {:?}",
            col_names
        );

        // Verify we have data
        assert!(df.height() > 0);

        // Verify y values (from intensity) are non-negative
        let y_col = df.column("__ggsql_aes_y__").expect("y aesthetic exists");
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

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);
        let data_cte = build_data_cte("x", None, query, &groups);
        let grid_cte = build_grid_cte(&groups, query, 0.0, 100.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
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
