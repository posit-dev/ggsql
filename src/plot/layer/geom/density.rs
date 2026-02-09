//! Density geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};
use crate::{
    naming,
    plot::{
        geom::types::get_column_name, DefaultParam, DefaultParamValue, ParameterValue, StatResult,
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

    fn default_remappings(&self) -> &'static [(&'static str, &'static str)] {
        &[("x", "x"), ("density", "y")]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["x", "density"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["x"]
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

    let (min, max) = compute_range_sql(&x, query, execute)?;
    let bw_cte = density_sql_bandwidth(query, group_by, &x, parameters);
    let grid_cte = build_grid_cte(group_by, query, min, max, 512);
    let kernel = choose_kde_kernel(parameters)?;
    let density_query = compute_density(&x, query, group_by, kernel, &bw_cte, &grid_cte);

    Ok(StatResult::Transformed {
        query: density_query,
        stat_columns: vec!["x".to_string(), "density".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["x".to_string()],
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
    // We move dividing by bandwidth outside the average computation to avoid
    // having to apply it to every element separately.
    Ok(format!(
        "AVG({kernel}) / ANY_VALUE(bandwidth.bw)",
        kernel = kernel
    ))
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
    value: &str,
    from: &str,
    group_by: &[String],
    kernel: String,
    bandwidth_cte: &str,
    grid_cte: &str,
) -> String {
    let data_cte = format!(
        "data AS (
          SELECT {groups}{value} AS val
          FROM ({from})
          WHERE {value} IS NOT NULL
        )",
        groups = with_trailing_comma(&group_by.join(", ")),
        value = value,
        from = from
    );

    // Build bandwidth join condition
    let bandwidth_conditions = if group_by.is_empty() {
        "true".to_string()
    } else {
        group_by
            .iter()
            .map(|g| format!("data.{col} = bandwidth.{col}", col = g))
            .collect::<Vec<String>>()
            .join(" AND ")
    };

    // Build WHERE clause to match grid to data groups
    let matching_groups = if group_by.is_empty() {
        String::new()
    } else {
        let grid_data_conds: Vec<String> = group_by
            .iter()
            .map(|g| format!("grid.{col} = data.{col}", col = g))
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

    // Generate the density computation query
    format!(
        "{bandwidth_cte},
        {data_cte},
        {grid_cte}
        SELECT
          grid.x AS {x_column},
          {grid_groups}
          {kernel} AS {density_column}
        {join_logic}
        {aggregation}",
        bandwidth_cte = bandwidth_cte,
        data_cte = data_cte,
        grid_cte = grid_cte,
        x_column = naming::stat_column("x"),
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
        let grid_cte = build_grid_cte(&groups, query, 0.0, 10.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", query, &groups, kernel, &bw_cte, &grid_cte);

        let expected = "WITH bandwidth AS (SELECT 0.5 AS bw),
        data AS (
          SELECT x AS val
          FROM (SELECT x FROM (VALUES (1.0), (2.0), (3.0)) AS t(x))
          WHERE x IS NOT NULL
        ),
        grid AS (
          SELECT -0.5 + (seq.n * 11 / 511) AS x
          FROM GENERATE_SERIES(0, 511) AS seq(n)
        )
        SELECT
          grid.x AS __ggsql_stat_x,
          AVG((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327) / ANY_VALUE(bandwidth.bw) AS __ggsql_stat_density
        FROM data
        INNER JOIN bandwidth ON true
        CROSS JOIN grid
        GROUP BY grid.x
        ORDER BY grid.x";

        // Normalize whitespace for comparison
        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalize(&sql), normalize(expected));

        // Verify SQL executes and produces correct output shape
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql(&sql).expect("SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec!["__ggsql_stat_x", "__ggsql_stat_density"]
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
        let grid_cte = build_grid_cte(&groups, query, -10.0, 10.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", query, &groups, kernel, &bw_cte, &grid_cte);

        let expected = "WITH bandwidth AS (SELECT 0.5 AS bw, region, category FROM (SELECT x, region, category FROM (VALUES (1.0, 'A', 'X'), (2.0, 'B', 'Y')) AS t(x, region, category)) GROUP BY region, category),
        data AS (
          SELECT region, category, x AS val
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
          grid.x AS __ggsql_stat_x,
          grid.region, grid.category,
          AVG((EXP(-0.5 * (grid.x - data.val) * (grid.x - data.val) / (bandwidth.bw * bandwidth.bw))) * 0.3989422804014327) / ANY_VALUE(bandwidth.bw) AS __ggsql_stat_density
        FROM data
        INNER JOIN bandwidth ON data.region = bandwidth.region AND data.category = bandwidth.category
        CROSS JOIN grid
        WHERE grid.region = data.region AND grid.category = data.category
        GROUP BY grid.x, grid.region, grid.category
        ORDER BY grid.x, grid.region, grid.category";

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
            .filter_map(|v| v)
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
        let query = "SELECT x FROM (VALUES (1.0), (2.0), (3.0), (4.0), (5.0)) AS t(x)";
        let groups: Vec<String> = vec![];
        let parameters = HashMap::new(); // No explicit bandwidth - will compute

        let bw_cte = density_sql_bandwidth(query, &groups, "x", &parameters);

        // Verify bandwidth computation executes
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader
            .execute_sql(&format!("{}\nSELECT bw FROM bandwidth", bw_cte))
            .expect("Bandwidth SQL should execute");

        assert_eq!(df.get_column_names(), vec!["bw"]);
        assert_eq!(df.height(), 1);
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
        // Use wide range to capture essentially all density mass
        let grid_cte = build_grid_cte(&groups, query, -5.0, 15.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", query, &groups, kernel, &bw_cte, &grid_cte);

        // Execute query
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql(&sql).expect("SQL should execute");

        assert_eq!(
            df.get_column_names(),
            vec!["__ggsql_stat_x", "__ggsql_stat_density"]
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
            .filter_map(|v| v)
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
        let grid_cte = build_grid_cte(&groups, query, 0.0, 100.0, 512);
        let kernel = choose_kde_kernel(&parameters).expect("kernel should be valid");
        let sql = compute_density("x", query, &groups, kernel, &bw_cte, &grid_cte);

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
