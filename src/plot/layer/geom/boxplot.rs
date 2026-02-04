//! Boxplot geom implementation

use std::collections::HashMap;

use super::{GeomAesthetics, GeomTrait, GeomType};
use crate::{
    naming,
    plot::{
        geom::types::get_column_name, DefaultParam, DefaultParamValue, ParameterValue, StatResult,
    },
    DataFrame, GgsqlError, Mappings, Result,
};

/// Boxplot geom - box and whisker plots
#[derive(Debug, Clone, Copy)]
pub struct Boxplot;

impl GeomTrait for Boxplot {
    fn geom_type(&self) -> GeomType {
        GeomType::Boxplot
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x",
                "y",
                "color",
                "colour",
                "fill",
                "stroke",
                "opacity",
                "linetype",
                "linewidth",
                "size",
                "shape",
            ],
            required: &["x", "y"],
            // Internal aesthetics produced by stat transform
            hidden: &["type", "y"],
        }
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["y"]
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [super::DefaultParam] {
        &[
            DefaultParam {
                name: "outliers",
                default: super::DefaultParamValue::Boolean(true),
            },
            DefaultParam {
                name: "coef",
                default: DefaultParamValue::Number(1.5),
            },
            DefaultParam {
                name: "width",
                default: DefaultParamValue::Number(0.9),
            },
        ]
    }

    fn default_remappings(&self) -> &'static [(&'static str, &'static str)] {
        &[("value", "y")]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        stat_boxplot(query, aesthetics, group_by, parameters)
    }
}

impl std::fmt::Display for Boxplot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "boxplot")
    }
}

fn stat_boxplot(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
) -> Result<StatResult> {
    let y = get_column_name(aesthetics, "y").ok_or_else(|| {
        GgsqlError::ValidationError("Boxplot requires 'y' aesthetic mapping".to_string())
    })?;
    let x = get_column_name(aesthetics, "x").ok_or_else(|| {
        GgsqlError::ValidationError("Boxplot requires 'x' aesthetic mapping".to_string())
    })?;

    // Fetch coef parameter
    let coef = match parameters.get("coef") {
        Some(ParameterValue::Number(num)) => num,
        _ => {
            return Err(GgsqlError::InternalError(
                "The 'coef' boxplot parameter must be a numeric value.".to_string(),
            ))
        }
    };

    // Fetch outliers parameter
    let outliers = match parameters.get("outliers") {
        Some(ParameterValue::Boolean(draw)) => draw,
        _ => {
            return Err(GgsqlError::InternalError(
                "The 'outliers' parameter must be `true` or `false`.".to_string(),
            ))
        }
    };

    // Fix boxplots to be vertical, when we later have orientation this may change
    let (value_col, group_col) = (y, x);

    // The `groups` vector is never empty, it contains at least the opposite axis as column
    // This absolves us from every having to guard against empty groups
    let mut groups = group_by.to_vec();
    if !groups.contains(&group_col) {
        groups.push(group_col);
    }
    if groups.is_empty() {
        // We should never end up here, but this is just to enforce the assumption above.
        return Err(GgsqlError::InternalError(
            "Boxplots cannot have empty groups".to_string(),
        ));
    }

    // Query for boxplot summary statistics
    let summary = boxplot_sql_compute_summary(query, &groups, &value_col, coef);
    let stats_query = boxplot_sql_append_outliers(&summary, &groups, &value_col, query, outliers);

    Ok(StatResult::Transformed {
        query: stats_query,
        stat_columns: vec!["type".to_string(), "value".to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["y".to_string()],
    })
}

fn boxplot_sql_assign_quartiles(from: &str, groups: &[String], value: &str) -> String {
    // Selects all relevant columns and adds a quartile column.
    // NTILE(4) may create uneven groups
    format!(
        "SELECT
          {value},
          {groups},
          NTILE(4) OVER (PARTITION BY {groups} ORDER BY {value} ASC) AS _Q
        FROM ({from})
        WHERE {value} IS NOT NULL",
        value = value,
        groups = groups.join(", "),
        from = from
    )
}

fn boxplot_sql_quartile_minmax(from: &str, groups: &[String], value: &str) -> String {
    // Compute the min and max for every quartile.
    // The verbosity here is to pivot the table to a wide format.
    // The output is a table with 1 row per groups annotated with quartile metrics
    format!(
        "SELECT
          MIN(CASE WHEN _Q = 1 THEN {value} END) AS Q1_min,
          MAX(CASE WHEN _Q = 1 THEN {value} END) AS Q1_max,
          MIN(CASE WHEN _Q = 2 THEN {value} END) AS Q2_min,
          MAX(CASE WHEN _Q = 2 THEN {value} END) AS Q2_max,
          MIN(CASE WHEN _Q = 3 THEN {value} END) AS Q3_min,
          MAX(CASE WHEN _Q = 3 THEN {value} END) AS Q3_max,
          MIN(CASE WHEN _Q = 4 THEN {value} END) AS Q4_min,
          MAX(CASE WHEN _Q = 4 THEN {value} END) AS Q4_max,
          {groups}
        FROM ({from})
        GROUP BY {groups}",
        groups = groups.join(", "),
        value = value,
        from = from
    )
}

fn boxplot_sql_compute_fivenum(from: &str, groups: &[String], coef: &f64) -> String {
    // Here we compute the 5 statistics:
    // * lower: lower whisker
    // * upper: upper whisker
    // * q1: box start
    // * q3: box end
    // * median
    // We're assuming equally sized quartiles here, but we may have 1-member
    // differences. For large datasets this shouldn't be a problem, but in smaller
    // datasets one might notice.
    format!(
        "SELECT
          *,
          GREATEST(q1 - {coef} * (q3 - q1), min) AS lower,
          LEAST(   q3 + {coef} * (q3 - q1), max) AS upper
        FROM (
          SELECT
            Q1_min AS min,
            Q4_max AS max,
            (Q2_max + Q3_min) / 2.0 AS median,
            (Q1_max + Q2_min) / 2.0 AS q1,
            (Q3_max + Q4_min) / 2.0 AS q3,
            {groups}
          FROM ({from})
        )",
        coef = coef,
        groups = groups.join(", "),
        from = from
    )
}

fn boxplot_sql_compute_summary(from: &str, groups: &[String], value: &str, coef: &f64) -> String {
    let query = boxplot_sql_assign_quartiles(from, groups, value);
    let query = boxplot_sql_quartile_minmax(&query, groups, value);
    boxplot_sql_compute_fivenum(&query, groups, coef)
}

fn boxplot_sql_filter_outliers(groups: &[String], value: &str, from: &str) -> String {
    let mut join_pairs = Vec::new();
    let mut keep_columns = Vec::new();
    for column in groups {
        join_pairs.push(format!("raw.{} = summary.{}", column, column));
        keep_columns.push(format!("raw.{}", column));
    }

    // We're joining outliers with the summary to use the lower/upper whisker
    // values as a filter
    format!(
        "SELECT
          raw.{value} AS value,
          'outlier' AS type,
          {groups}
        FROM ({from}) raw
        JOIN summary ON {pairs}
        WHERE raw.{value} NOT BETWEEN summary.lower AND summary.upper",
        value = value,
        groups = keep_columns.join(", "),
        pairs = join_pairs.join(" AND "),
        from = from
    )
}

fn boxplot_sql_append_outliers(
    from: &str,
    groups: &[String],
    value: &str,
    raw_query: &str,
    draw_outliers: &bool,
) -> String {
    let value_name = naming::stat_column("value");
    let type_name = naming::stat_column("type");

    if !*draw_outliers {
        // Just reshape summary to long format
        let sql = format!(
            "SELECT {groups}, type AS {type_name}, value AS {value_name}
            FROM ({summary})
            UNPIVOT(value FOR type IN (min, max, median, q1, q3, upper, lower))",
            groups = groups.join(", "),
            value_name = value_name,
            type_name = type_name,
            summary = from
        );
        return sql;
    }

    // Grab query for outliers. Outcome is long format data.
    let outliers = boxplot_sql_filter_outliers(groups, value, raw_query);

    // Reshape summary to long format and combine with outliers in single table
    format!(
        "WITH
        summary AS (
          {summary}
        ),
        outliers AS (
          {outliers}
        )
        (
          SELECT {groups}, type AS {type_name}, value AS {value_name}
          FROM summary
          UNPIVOT(value FOR type IN (min, max, median, q1, q3, upper, lower))
        )
        UNION ALL
        (
          SELECT {groups}, type AS {type_name}, value AS {value_name}
          FROM outliers
        )
        ",
        summary = from,
        outliers = outliers,
        type_name = type_name,
        value_name = value_name,
        groups = groups.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;

    // ==================== Helper Functions ====================

    fn create_basic_aesthetics() -> Mappings {
        let mut aesthetics = Mappings::new();
        aesthetics.insert(
            "x".to_string(),
            AestheticValue::standard_column("category".to_string()),
        );
        aesthetics.insert(
            "y".to_string(),
            AestheticValue::standard_column("value".to_string()),
        );
        aesthetics
    }

    // ==================== SQL Generation Tests (Compact) ====================

    #[test]
    fn test_sql_assign_quartiles_basic() {
        let groups = vec!["category".to_string()];
        let result = boxplot_sql_assign_quartiles("data", &groups, "value");
        assert!(result.contains("NTILE(4)"));
        assert!(result.contains("PARTITION BY category"));
        assert!(result.contains("WHERE value IS NOT NULL"));
    }

    #[test]
    fn test_sql_assign_quartiles_multiple_groups() {
        let groups = vec!["cat".to_string(), "region".to_string()];
        let result = boxplot_sql_assign_quartiles("tbl", &groups, "val");
        assert!(result.contains("PARTITION BY cat, region"));
    }

    #[test]
    fn test_sql_quartile_minmax_structure() {
        let groups = vec!["grp".to_string()];
        let result = boxplot_sql_quartile_minmax("query", &groups, "v");
        assert!(result.contains("Q1_min"));
        assert!(result.contains("Q4_max"));
        assert!(result.contains("CASE WHEN _Q = 1"));
        assert!(result.contains("GROUP BY grp"));
    }

    #[test]
    fn test_sql_compute_fivenum_coef() {
        let groups = vec!["x".to_string()];
        let result = boxplot_sql_compute_fivenum("q", &groups, &2.5);
        assert!(result.contains("2.5"));
        assert!(result.contains("AS lower"));
        assert!(result.contains("AS upper"));
        assert!(result.contains("AS median"));
        assert!(result.contains("GREATEST"));
        assert!(result.contains("LEAST"));
    }

    #[test]
    fn test_sql_filter_outliers_join() {
        let groups = vec!["cat".to_string(), "region".to_string()];
        let result = boxplot_sql_filter_outliers(&groups, "value", "raw_data");
        assert!(result.contains("JOIN summary ON"));
        assert!(result.contains("raw.cat = summary.cat"));
        assert!(result.contains("raw.region = summary.region"));
        assert!(result.contains("NOT BETWEEN summary.lower AND summary.upper"));
        assert!(result.contains("'outlier' AS type"));
    }

    // ==================== SQL Snapshot Tests ====================

    #[test]
    fn test_boxplot_sql_compute_summary_single_group() {
        let groups = vec!["category".to_string()];
        let result = boxplot_sql_compute_summary("SELECT * FROM sales", &groups, "price", &1.5);

        let expected = r#"SELECT
          *,
          GREATEST(q1 - 1.5 * (q3 - q1), min) AS lower,
          LEAST(   q3 + 1.5 * (q3 - q1), max) AS upper
        FROM (
          SELECT
            Q1_min AS min,
            Q4_max AS max,
            (Q2_max + Q3_min) / 2.0 AS median,
            (Q1_max + Q2_min) / 2.0 AS q1,
            (Q3_max + Q4_min) / 2.0 AS q3,
            category
          FROM (SELECT
          MIN(CASE WHEN _Q = 1 THEN price END) AS Q1_min,
          MAX(CASE WHEN _Q = 1 THEN price END) AS Q1_max,
          MIN(CASE WHEN _Q = 2 THEN price END) AS Q2_min,
          MAX(CASE WHEN _Q = 2 THEN price END) AS Q2_max,
          MIN(CASE WHEN _Q = 3 THEN price END) AS Q3_min,
          MAX(CASE WHEN _Q = 3 THEN price END) AS Q3_max,
          MIN(CASE WHEN _Q = 4 THEN price END) AS Q4_min,
          MAX(CASE WHEN _Q = 4 THEN price END) AS Q4_max,
          category
        FROM (SELECT
          price,
          category,
          NTILE(4) OVER (PARTITION BY category ORDER BY price ASC) AS _Q
        FROM (SELECT * FROM sales)
        WHERE price IS NOT NULL)
        GROUP BY category)
        )"#;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_boxplot_sql_compute_summary_multiple_groups() {
        let groups = vec!["region".to_string(), "product".to_string()];
        let result = boxplot_sql_compute_summary("SELECT * FROM data", &groups, "revenue", &1.5);

        let expected = r#"SELECT
          *,
          GREATEST(q1 - 1.5 * (q3 - q1), min) AS lower,
          LEAST(   q3 + 1.5 * (q3 - q1), max) AS upper
        FROM (
          SELECT
            Q1_min AS min,
            Q4_max AS max,
            (Q2_max + Q3_min) / 2.0 AS median,
            (Q1_max + Q2_min) / 2.0 AS q1,
            (Q3_max + Q4_min) / 2.0 AS q3,
            region, product
          FROM (SELECT
          MIN(CASE WHEN _Q = 1 THEN revenue END) AS Q1_min,
          MAX(CASE WHEN _Q = 1 THEN revenue END) AS Q1_max,
          MIN(CASE WHEN _Q = 2 THEN revenue END) AS Q2_min,
          MAX(CASE WHEN _Q = 2 THEN revenue END) AS Q2_max,
          MIN(CASE WHEN _Q = 3 THEN revenue END) AS Q3_min,
          MAX(CASE WHEN _Q = 3 THEN revenue END) AS Q3_max,
          MIN(CASE WHEN _Q = 4 THEN revenue END) AS Q4_min,
          MAX(CASE WHEN _Q = 4 THEN revenue END) AS Q4_max,
          region, product
        FROM (SELECT
          revenue,
          region, product,
          NTILE(4) OVER (PARTITION BY region, product ORDER BY revenue ASC) AS _Q
        FROM (SELECT * FROM data)
        WHERE revenue IS NOT NULL)
        GROUP BY region, product)
        )"#;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_boxplot_sql_compute_summary_custom_coef() {
        let groups = vec!["x".to_string()];
        let result = boxplot_sql_compute_summary("source_query", &groups, "y", &3.0);

        // Verify coef parameter is properly interpolated
        assert!(result.contains("3 *"));
        assert!(result.contains("GREATEST(q1 - 3 * (q3 - q1), min)"));
        assert!(result.contains("LEAST(   q3 + 3 * (q3 - q1), max)"));
    }

    #[test]
    fn test_boxplot_sql_append_outliers_with_outliers() {
        let groups = vec!["category".to_string()];
        let summary = "summary_query";
        let raw = "raw_query";
        let result = boxplot_sql_append_outliers(summary, &groups, "value", raw, &true);

        // Check key components
        assert!(result.contains("WITH"));
        assert!(result.contains("summary AS ("));
        assert!(result.contains("summary_query"));
        assert!(result.contains("outliers AS ("));
        assert!(result.contains("UNION ALL"));
        assert!(
            result.contains("UNPIVOT(value FOR type IN (min, max, median, q1, q3, upper, lower))")
        );
        assert!(result.contains(&format!("AS {}", naming::stat_column("value"))));
        assert!(result.contains(&format!("AS {}", naming::stat_column("type"))));
    }

    #[test]
    fn test_boxplot_sql_append_outliers_without_outliers() {
        let groups = vec!["x".to_string()];
        let summary = "sum_query";
        let raw = "raw_query";
        let result = boxplot_sql_append_outliers(summary, &groups, "y", raw, &false);

        // Should NOT include WITH or outliers CTE
        assert!(!result.contains("WITH"));
        assert!(!result.contains("outliers AS"));
        assert!(!result.contains("UNION ALL"));

        // Should just UNPIVOT summary
        assert!(result.contains("UNPIVOT"));
        assert!(result.contains("(sum_query)"));
        assert!(result.contains(&format!("AS {}", naming::stat_column("value"))));
        assert!(result.contains(&format!("AS {}", naming::stat_column("type"))));
    }

    #[test]
    fn test_boxplot_sql_append_outliers_multi_group() {
        let groups = vec!["cat".to_string(), "region".to_string(), "year".to_string()];
        let summary = "(SELECT * FROM stats)";
        let raw = "(SELECT * FROM raw_data)";
        let result = boxplot_sql_append_outliers(summary, &groups, "val", raw, &true);

        // Verify all groups are present
        assert!(result.contains("cat, region, year"));

        // Check structure
        assert!(result.contains("WITH"));
        assert!(result.contains("summary AS"));
        assert!(result.contains("outliers AS"));

        // Verify outlier join conditions for all groups
        let outlier_section = result.split("outliers AS").nth(1).unwrap();
        assert!(outlier_section.contains("raw.cat = summary.cat"));
        assert!(outlier_section.contains("raw.region = summary.region"));
        assert!(outlier_section.contains("raw.year = summary.year"));
    }

    // ==================== Parameter Validation Tests ====================

    #[test]
    fn test_stat_boxplot_invalid_coef_type() {
        let aesthetics = create_basic_aesthetics();
        let groups = vec![];

        let mut parameters = HashMap::new();
        parameters.insert(
            "coef".to_string(),
            ParameterValue::String("invalid".to_string()),
        );
        parameters.insert("outliers".to_string(), ParameterValue::Boolean(true));

        let result = stat_boxplot("SELECT * FROM data", &aesthetics, &groups, &parameters);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("coef"));
    }

    #[test]
    fn test_stat_boxplot_missing_coef() {
        let aesthetics = create_basic_aesthetics();
        let groups = vec![];

        let mut parameters = HashMap::new();
        parameters.insert("outliers".to_string(), ParameterValue::Boolean(true));
        // Missing coef

        let result = stat_boxplot("SELECT * FROM data", &aesthetics, &groups, &parameters);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("coef"));
    }

    #[test]
    fn test_stat_boxplot_invalid_outliers_type() {
        let aesthetics = create_basic_aesthetics();
        let groups = vec![];

        let mut parameters = HashMap::new();
        parameters.insert("coef".to_string(), ParameterValue::Number(1.5));
        parameters.insert(
            "outliers".to_string(),
            ParameterValue::String("yes".to_string()),
        );

        let result = stat_boxplot("SELECT * FROM data", &aesthetics, &groups, &parameters);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outliers"));
    }

    #[test]
    fn test_stat_boxplot_missing_outliers() {
        let aesthetics = create_basic_aesthetics();
        let groups = vec![];

        let mut parameters = HashMap::new();
        parameters.insert("coef".to_string(), ParameterValue::Number(1.5));
        // Missing outliers

        let result = stat_boxplot("SELECT * FROM data", &aesthetics, &groups, &parameters);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outliers"));
    }

    // ==================== GeomTrait Implementation Tests ====================

    #[test]
    fn test_boxplot_geom_type() {
        let boxplot = Boxplot;
        assert_eq!(boxplot.geom_type(), GeomType::Boxplot);
    }

    #[test]
    fn test_boxplot_aesthetics_required() {
        let boxplot = Boxplot;
        let aes = boxplot.aesthetics();

        assert!(aes.required.contains(&"x"));
        assert!(aes.required.contains(&"y"));
        assert_eq!(aes.required.len(), 2);
    }

    #[test]
    fn test_boxplot_aesthetics_supported() {
        let boxplot = Boxplot;
        let aes = boxplot.aesthetics();

        assert!(aes.supported.contains(&"x"));
        assert!(aes.supported.contains(&"y"));
        assert!(aes.supported.contains(&"color"));
        assert!(aes.supported.contains(&"colour"));
        assert!(aes.supported.contains(&"fill"));
        assert!(aes.supported.contains(&"stroke"));
        assert!(aes.supported.contains(&"opacity"));
    }

    #[test]
    fn test_boxplot_default_params() {
        let boxplot = Boxplot;
        let params = boxplot.default_params();

        assert_eq!(params.len(), 3);

        // Find and verify outliers param
        let outliers_param = params.iter().find(|p| p.name == "outliers").unwrap();
        assert!(matches!(
            outliers_param.default,
            DefaultParamValue::Boolean(true)
        ));

        // Find and verify coef param
        let coef_param = params.iter().find(|p| p.name == "coef").unwrap();
        assert!(
            matches!(coef_param.default, DefaultParamValue::Number(v) if (v - 1.5).abs() < f64::EPSILON)
        );

        // Find and verify width param
        let width_param = params.iter().find(|p| p.name == "width").unwrap();
        assert!(
            matches!(width_param.default, DefaultParamValue::Number(v) if (v - 0.9).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn test_boxplot_default_remappings() {
        let boxplot = Boxplot;
        let remappings = boxplot.default_remappings();

        assert_eq!(remappings.len(), 1);
        assert_eq!(remappings[0], ("value", "y"));
    }

    #[test]
    fn test_boxplot_stat_consumed_aesthetics() {
        let boxplot = Boxplot;
        let consumed = boxplot.stat_consumed_aesthetics();

        assert_eq!(consumed.len(), 1);
        assert_eq!(consumed[0], "y");
    }

    #[test]
    fn test_boxplot_needs_stat_transform() {
        let boxplot = Boxplot;
        let aesthetics = Mappings::new();
        assert!(boxplot.needs_stat_transform(&aesthetics));
    }

    #[test]
    fn test_boxplot_display() {
        let boxplot = Boxplot;
        assert_eq!(format!("{}", boxplot), "boxplot");
    }
}
