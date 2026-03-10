/// Portable scalar MAX across any number of SQL expressions.
/// Replaces GREATEST(a, b, ...) which is not supported by all backends.
pub fn sql_greatest(exprs: &[&str]) -> String {
    let mut result = exprs[0].to_string();
    for expr in &exprs[1..] {
        result = format!("(CASE WHEN ({result}) >= ({expr}) THEN ({result}) ELSE ({expr}) END)");
    }
    result
}

/// Portable scalar MIN across any number of SQL expressions.
/// Replaces LEAST(a, b, ...) which is not supported by all backends.
pub fn sql_least(exprs: &[&str]) -> String {
    let mut result = exprs[0].to_string();
    for expr in &exprs[1..] {
        result = format!("(CASE WHEN ({result}) <= ({expr}) THEN ({result}) ELSE ({expr}) END)");
    }
    result
}

/// Portable GENERATE_SERIES using a recursive CTE.
///
/// Returns CTE fragment(s) producing table `__ggsql_seq__` with column `n`
/// containing values `0..n-1`. Uses a cube-root decomposition to avoid
/// deep recursion: only recurses ~cbrt(n) times, then cross-joins three
/// copies to cover the full range.
pub fn sql_generate_series(n: usize) -> String {
    use crate::naming::{SERIES_BASE, SERIES_SEQ};
    let base_size = (n as f64).cbrt().ceil() as usize;
    let base_sq = base_size * base_size;
    let base_max = base_size - 1;
    format!(
        "{SERIES_BASE}(n) AS (\
           SELECT 0 UNION ALL SELECT n + 1 FROM {SERIES_BASE} WHERE n < {base_max}\
         ),\
         {SERIES_SEQ}(n) AS (\
           SELECT CAST(a.n * {base_sq} + b.n * {base_size} + c.n AS REAL) AS n \
           FROM {SERIES_BASE} a, {SERIES_BASE} b, {SERIES_BASE} c \
           WHERE a.n * {base_sq} + b.n * {base_size} + c.n < {n}\
         )"
    )
}

/// Portable PERCENTILE computation using NTILE(4) with interpolation.
///
/// Returns a scalar subquery expression that computes the specified percentile
/// of a column within an optional grouping context. Uses NTILE(4) to divide
/// data into quartiles, then interpolates between boundaries.
pub fn sql_percentile(column: &str, fraction: f64, from: &str, groups: &[String]) -> String {
    let group_filter = groups
        .iter()
        .map(|g| format!("AND _pct.{g} IS NOT DISTINCT FROM _qt.{g}"))
        .collect::<Vec<_>>()
        .join(" ");

    // NTILE(4) for quartile boundaries with midpoint interpolation.
    let lo_tile = (fraction * 4.0).ceil() as usize;
    let hi_tile = lo_tile + 1;

    format!(
        "(SELECT (\
          MAX(CASE WHEN __tile = {lo_tile} THEN __val END) + \
          MIN(CASE WHEN __tile = {hi_tile} THEN __val END)\
        ) / 2.0 \
        FROM (\
          SELECT {column} AS __val, \
                 NTILE(4) OVER (ORDER BY {column}) AS __tile \
          FROM ({from}) AS _pct \
          WHERE {column} IS NOT NULL {group_filter}\
        ))"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_max_two_args() {
        assert_eq!(
            sql_greatest(&["a", "b"]),
            "(CASE WHEN (a) >= (b) THEN (a) ELSE (b) END)"
        );
    }

    #[test]
    fn test_scalar_min_two_args() {
        assert_eq!(
            sql_least(&["a", "b"]),
            "(CASE WHEN (a) <= (b) THEN (a) ELSE (b) END)"
        );
    }

    #[test]
    fn test_scalar_max_three_args() {
        assert_eq!(
            sql_greatest(&["x", "y", "z"]),
            "(CASE WHEN ((CASE WHEN (x) >= (y) THEN (x) ELSE (y) END)) >= (z) THEN ((CASE WHEN (x) >= (y) THEN (x) ELSE (y) END)) ELSE (z) END)"
        );
    }

    #[test]
    fn test_scalar_max_single() {
        assert_eq!(sql_greatest(&["x"]), "x");
    }

    #[test]
    fn test_scalar_min_single() {
        assert_eq!(sql_least(&["x"]), "x");
    }

    #[test]
    fn test_generate_series_small() {
        let sql = sql_generate_series(8);
        // base_size = ceil(8^(1/3)) = 2, base_sq = 4, base_max = 1
        assert!(sql.contains("WHERE n < 1"), "base CTE should recurse up to base_max=1");
        assert!(sql.contains("a.n * 4 + b.n * 2 + c.n"), "cross-join arithmetic");
        assert!(sql.contains("a.n * 4 + b.n * 2 + c.n < 8"), "seq CTE filters to n");
    }

    #[test]
    fn test_generate_series_exact_cube() {
        let sql = sql_generate_series(27);
        // base_size = ceil(27^(1/3)) = 3, base_sq = 9, base_max = 2
        assert!(sql.contains("WHERE n < 2"), "base CTE should recurse up to base_max=2");
        assert!(sql.contains("a.n * 9 + b.n * 3 + c.n < 27"));
    }

    #[test]
    fn test_generate_series_one() {
        let sql = sql_generate_series(1);
        // base_size = 1, base_sq = 1, base_max = 0
        assert!(sql.contains("WHERE n < 0"), "base CTE with base_max=0");
        assert!(sql.contains("< 1"), "seq CTE filters to n=1");
    }

    #[test]
    fn test_generate_series_large() {
        let sql = sql_generate_series(1000);
        // base_size = 10, base_sq = 100, base_max = 9
        assert!(sql.contains("WHERE n < 9"));
        assert!(sql.contains("a.n * 100 + b.n * 10 + c.n"));
        assert!(sql.contains("< 1000"));
    }

    #[test]
    fn test_generate_series_contains_cte_names() {
        let sql = sql_generate_series(8);
        assert!(sql.contains("__ggsql_base__"));
        assert!(sql.contains("__ggsql_seq__"));
    }

    #[test]
    fn test_percentile_no_groups() {
        let sql = sql_percentile("val", 0.5, "SELECT * FROM t", &[]);
        // fraction=0.5: lo_tile = ceil(2.0) = 2, hi_tile = 3
        assert!(sql.contains("__tile = 2"));
        assert!(sql.contains("__tile = 3"));
        assert!(sql.contains("NTILE(4)"));
        assert!(sql.contains("val AS __val"));
        assert!(sql.contains("FROM (SELECT * FROM t) AS _pct"));
        assert!(!sql.contains("IS NOT DISTINCT FROM"));
    }

    #[test]
    fn test_percentile_with_groups() {
        let groups = vec!["region".to_string(), "year".to_string()];
        let sql = sql_percentile("price", 0.25, "SELECT * FROM sales", &groups);
        assert!(sql.contains("_pct.region IS NOT DISTINCT FROM _qt.region"));
        assert!(sql.contains("_pct.year IS NOT DISTINCT FROM _qt.year"));
    }

    #[test]
    fn test_percentile_q1() {
        let sql = sql_percentile("x", 0.25, "t", &[]);
        // fraction=0.25: lo_tile = ceil(1.0) = 1, hi_tile = 2
        assert!(sql.contains("__tile = 1"));
        assert!(sql.contains("__tile = 2"));
    }

    #[test]
    fn test_percentile_q3() {
        let sql = sql_percentile("x", 0.75, "t", &[]);
        // fraction=0.75: lo_tile = ceil(3.0) = 3, hi_tile = 4
        assert!(sql.contains("__tile = 3"));
        assert!(sql.contains("__tile = 4"));
    }
}
