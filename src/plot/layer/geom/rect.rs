//! Rect geom implementation with flexible parameter specification

use std::collections::HashMap;

use super::types::get_column_name;
use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};
use crate::naming;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::{DataFrame, GgsqlError, Mappings, Result};

use super::types::Schema;

/// Rect geom - rectangles with flexible parameter specification
///
/// Supports multiple ways to specify rectangles:
/// - X-direction: any 2 of {x (center), width, xmin, xmax}
/// - Y-direction: any 2 of {y (center), height, ymin, ymax}
///
/// For continuous scales, computes xmin/xmax and ymin/ymax
/// For discrete scales, uses x/y with width/height as band fractions
#[derive(Debug, Clone, Copy)]
pub struct Rect;

impl GeomTrait for Rect {
    fn geom_type(&self) -> GeomType {
        GeomType::Rect
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                // All positional aesthetics are optional inputs (Null)
                // They become Delayed after stat transform
                ("pos1", DefaultAestheticValue::Null), // x (center)
                ("pos1min", DefaultAestheticValue::Null), // xmin
                ("pos1max", DefaultAestheticValue::Null), // xmax
                ("width", DefaultAestheticValue::Null), // width (aesthetic, can map to column)
                ("pos2", DefaultAestheticValue::Null), // y (center)
                ("pos2min", DefaultAestheticValue::Null), // ymin
                ("pos2max", DefaultAestheticValue::Null), // ymax
                ("height", DefaultAestheticValue::Null), // height (aesthetic, can map to column)
                // Visual aesthetics
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.5)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[
            // For continuous scales: remap to min/max
            ("pos1min", DefaultAestheticValue::Column("pos1min")),
            ("pos1max", DefaultAestheticValue::Column("pos1max")),
            ("pos2min", DefaultAestheticValue::Column("pos2min")),
            ("pos2max", DefaultAestheticValue::Column("pos2max")),
            // For discrete scales: remap to center
            ("pos1", DefaultAestheticValue::Column("pos1")),
            ("pos2", DefaultAestheticValue::Column("pos2")),
            // Width/height passed through for discrete (writer validation)
            ("width", DefaultAestheticValue::Column("width")),
            ("height", DefaultAestheticValue::Column("height")),
        ]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["pos1", "pos2", "pos1min", "pos1max", "pos2min", "pos2max", "width", "height"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &[
            "pos1", "pos1min", "pos1max", "width", "pos2", "pos2min", "pos2max", "height",
        ]
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        // Always apply stat transform to validate and consolidate parameters
        true
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> Result<DataFrame>,
    ) -> Result<StatResult> {
        stat_rect(query, schema, aesthetics, group_by, parameters)
    }
}

impl std::fmt::Display for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rect")
    }
}

/// Statistical transformation for rect: consolidate parameters and compute min/max
fn stat_rect(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
    _parameters: &HashMap<String, ParameterValue>,
) -> Result<StatResult> {
    // Get aesthetic column names for SQL (at stat time, all aesthetics are columns)
    let x = get_column_name(aesthetics, "pos1");
    let xmin = get_column_name(aesthetics, "pos1min");
    let xmax = get_column_name(aesthetics, "pos1max");
    let width = get_column_name(aesthetics, "width");

    let y = get_column_name(aesthetics, "pos2");
    let ymin = get_column_name(aesthetics, "pos2min");
    let ymax = get_column_name(aesthetics, "pos2max");
    let height = get_column_name(aesthetics, "height");

    // Filter out width/height from group_by (they're position aesthetics, not grouping)
    let group_by: Vec<String> = group_by
        .iter()
        .filter(|col| {
            !width.as_ref().map_or(false, |w| col == &w)
                && !height.as_ref().map_or(false, |h| col == &h)
        })
        .cloned()
        .collect();

    // Detect if x and y are discrete by checking schema
    let is_x_discrete = x
        .as_ref()
        .and_then(|col| schema.iter().find(|c| &c.name == col))
        .map(|c| c.is_discrete)
        .unwrap_or(false);
    let is_y_discrete = y
        .as_ref()
        .and_then(|col| schema.iter().find(|c| &c.name == col))
        .map(|c| c.is_discrete)
        .unwrap_or(false);

    // Generate SQL expressions based on parameter combinations
    // Validation (exactly 2 params, discrete + min/max check) happens inside
    let (x_expr_min, x_expr_max) = generate_position_expressions(
        x.as_deref(),
        xmin.as_deref(),
        xmax.as_deref(),
        width.as_deref(),
        is_x_discrete,
        "x",
    )?;
    let (y_expr_min, y_expr_max) = generate_position_expressions(
        y.as_deref(),
        ymin.as_deref(),
        ymax.as_deref(),
        height.as_deref(),
        is_y_discrete,
        "y",
    )?;

    // Build SELECT list and stat_columns based on discrete vs continuous
    let mut select_parts = vec![];
    let mut stat_columns = vec![];

    // Add group_by columns first
    if !group_by.is_empty() {
        select_parts.push(group_by.join(", "));
    }

    // X direction
    if is_x_discrete {
        select_parts.push(format!("{} AS {}", x_expr_min, naming::stat_column("pos1")));
        stat_columns.push("pos1".to_string());
        // For discrete, pass through width if mapped (for scale training)
        if let Some(ref width_col) = width {
            select_parts.push(format!("{} AS {}", width_col, naming::stat_column("width")));
            stat_columns.push("width".to_string());
        }
    } else {
        select_parts.push(format!("{} AS {}", x_expr_min, naming::stat_column("pos1min")));
        select_parts.push(format!("{} AS {}", x_expr_max, naming::stat_column("pos1max")));
        stat_columns.push("pos1min".to_string());
        stat_columns.push("pos1max".to_string());
    }

    // Y direction
    if is_y_discrete {
        select_parts.push(format!("{} AS {}", y_expr_min, naming::stat_column("pos2")));
        stat_columns.push("pos2".to_string());
        // For discrete, pass through height if mapped (for scale training)
        if let Some(ref height_col) = height {
            select_parts.push(format!("{} AS {}", height_col, naming::stat_column("height")));
            stat_columns.push("height".to_string());
        }
    } else {
        select_parts.push(format!("{} AS {}", y_expr_min, naming::stat_column("pos2min")));
        select_parts.push(format!("{} AS {}", y_expr_max, naming::stat_column("pos2max")));
        stat_columns.push("pos2min".to_string());
        stat_columns.push("pos2max".to_string());
    }

    let select_list = select_parts.join(", ");

    // Build transformed query
    let transformed_query = format!(
        "SELECT {} FROM ({}) AS __ggsql_rect_stat__",
        select_list, query
    );

    // Build consumed aesthetics - all potentially mapped positional aesthetics
    let mut consumed = vec!["pos1", "pos1min", "pos1max", "pos2", "pos2min", "pos2max"];
    if width.is_some() {
        consumed.push("width");
    }
    if height.is_some() {
        consumed.push("height");
    }

    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns: vec![],
        consumed_aesthetics: consumed.iter().map(|s| s.to_string()).collect(),
    })
}

/// Generate SQL expressions for position min/max based on parameter combinations
///
/// Returns (min_expr, max_expr) or (center_expr, center_expr) for discrete
///
/// Validates:
/// - Discrete scales cannot use min/max aesthetics
/// - Exactly 2 parameters provided (via match statement)
fn generate_position_expressions(
    center: Option<&str>,
    min: Option<&str>,
    max: Option<&str>,
    size: Option<&str>,
    is_discrete: bool,
    axis: &str,
) -> Result<(String, String)> {
    // Validate: discrete scales cannot use min/max
    if is_discrete && (min.is_some() || max.is_some()) {
        return Err(GgsqlError::ValidationError(format!(
            "Cannot use {}min/{}max with discrete {} aesthetic. Use {} + {} instead.",
            axis,
            axis,
            axis,
            axis,
            if axis == "x" { "width" } else { "height" }
        )));
    }

    // For discrete, only center + size is valid
    if is_discrete {
        if let (Some(c), Some(_)) = (center, size) {
            return Ok((c.to_string(), c.to_string()));
        }
        return Err(GgsqlError::ValidationError(format!(
            "Discrete {} requires {} and {}.",
            axis,
            axis,
            if axis == "x" { "width" } else { "height" }
        )));
    }

    // For continuous, handle all 6 combinations
    // The _ arm catches invalid parameter counts (not exactly 2)
    match (center, min, max, size) {
        // Case 1: min + max
        (None, Some(min_col), Some(max_col), None) => {
            Ok((min_col.to_string(), max_col.to_string()))
        }
        // Case 2: center + size
        (Some(c), None, None, Some(s)) => Ok((
            format!("({} - {} / 2.0)", c, s),
            format!("({} + {} / 2.0)", c, s),
        )),
        // Case 3: center + min
        (Some(c), Some(min_col), None, None) => {
            Ok((min_col.to_string(), format!("(2 * {} - {})", c, min_col)))
        }
        // Case 4: center + max
        (Some(c), None, Some(max_col), None) => {
            Ok((format!("(2 * {} - {})", c, max_col), max_col.to_string()))
        }
        // Case 5: min + size
        (None, Some(min_col), None, Some(s)) => {
            Ok((min_col.to_string(), format!("({} + {})", min_col, s)))
        }
        // Case 6: max + size
        (None, None, Some(max_col), Some(s)) => {
            Ok((format!("({} - {})", max_col, s), max_col.to_string()))
        }
        // Invalid: wrong number of parameters or invalid combination
        _ => Err(GgsqlError::ValidationError(format!(
            "Rect requires exactly 2 {}-direction parameters from {{{}, {}min, {}max, {}}}.",
            axis,
            axis,
            axis,
            axis,
            if axis == "x" { "width" } else { "height" }
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::types::{AestheticValue, ColumnInfo};
    use polars::prelude::DataType;

    // ==================== Helper Functions ====================

    fn create_schema(discrete_cols: &[&str]) -> Schema {
        vec![
            ColumnInfo {
                name: "__ggsql_aes_pos1__".to_string(),
                dtype: if discrete_cols.contains(&"pos1") {
                    DataType::String
                } else {
                    DataType::Float64
                },
                is_discrete: discrete_cols.contains(&"pos1"),
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos1min__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos1max__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_width__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2__".to_string(),
                dtype: if discrete_cols.contains(&"pos2") {
                    DataType::String
                } else {
                    DataType::Float64
                },
                is_discrete: discrete_cols.contains(&"pos2"),
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2min__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2max__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_height__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
        ]
    }

    fn create_aesthetics(mappings: &[&str]) -> Mappings {
        let mut aesthetics = Mappings::new();
        for aesthetic in mappings {
            // Use aesthetic column naming convention
            let col_name = naming::aesthetic_column(aesthetic);
            aesthetics.insert(
                aesthetic.to_string(),
                AestheticValue::standard_column(col_name),
            );
        }
        aesthetics
    }

    // ==================== X-Direction Parameter Combinations (Continuous) ====================

    #[test]
    fn test_continuous_x_all_combinations() {
        let test_cases = vec![
            // (name, x_aesthetics, expected_min_expr, expected_max_expr)
            (
                "xmin + xmax",
                vec!["pos1min", "pos1max"],
                "__ggsql_aes_pos1min__",
                "__ggsql_aes_pos1max__",
            ),
            (
                "x + width",
                vec!["pos1", "width"],
                "(__ggsql_aes_pos1__ - __ggsql_aes_width__ / 2.0)",
                "(__ggsql_aes_pos1__ + __ggsql_aes_width__ / 2.0)",
            ),
            (
                "x + xmin",
                vec!["pos1", "pos1min"],
                "__ggsql_aes_pos1min__",
                "(2 * __ggsql_aes_pos1__ - __ggsql_aes_pos1min__)",
            ),
            (
                "x + xmax",
                vec!["pos1", "pos1max"],
                "(2 * __ggsql_aes_pos1__ - __ggsql_aes_pos1max__)",
                "__ggsql_aes_pos1max__",
            ),
            (
                "xmin + width",
                vec!["pos1min", "width"],
                "__ggsql_aes_pos1min__",
                "(__ggsql_aes_pos1min__ + __ggsql_aes_width__)",
            ),
            (
                "xmax + width",
                vec!["pos1max", "width"],
                "(__ggsql_aes_pos1max__ - __ggsql_aes_width__)",
                "__ggsql_aes_pos1max__",
            ),
        ];

        for (name, x_aesthetics, expected_min, expected_max) in test_cases {
            // Combine x aesthetics with fixed y mappings (ymin + ymax)
            let mut all_mappings = x_aesthetics.clone();
            all_mappings.extend_from_slice(&["pos2min", "pos2max"]);

            let aesthetics = create_aesthetics(&all_mappings);
            let schema = create_schema(&[]);
            let group_by = vec![];
            let parameters = HashMap::new();

            let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);

            assert!(result.is_ok(), "{}: stat_rect failed: {:?}", name, result.err());
            let stat_result = result.unwrap();

            if let StatResult::Transformed { query, stat_columns, .. } = stat_result {
                let stat_pos1min = naming::stat_column("pos1min");
                let stat_pos1max = naming::stat_column("pos1max");
                assert!(query.contains(&format!("{} AS {}", expected_min, stat_pos1min)),
                    "{}: Expected '{} AS {}' in query, got: {}", name, expected_min, stat_pos1min, query);
                assert!(query.contains(&format!("{} AS {}", expected_max, stat_pos1max)),
                    "{}: Expected '{} AS {}' in query, got: {}", name, expected_max, stat_pos1max, query);
                assert!(stat_columns.contains(&"pos1min".to_string()), "{}: Missing pos1min in stat_columns", name);
                assert!(stat_columns.contains(&"pos1max".to_string()), "{}: Missing pos1max in stat_columns", name);
            } else {
                panic!("{}: Expected Transformed result", name);
            }
        }
    }

    // ==================== Y-Direction Parameter Combinations (Continuous) ====================

    #[test]
    fn test_continuous_y_all_combinations() {
        let test_cases = vec![
            // (name, y_aesthetics, expected_min_expr, expected_max_expr)
            (
                "ymin + ymax",
                vec!["pos2min", "pos2max"],
                "__ggsql_aes_pos2min__",
                "__ggsql_aes_pos2max__",
            ),
            (
                "y + height",
                vec!["pos2", "height"],
                "(__ggsql_aes_pos2__ - __ggsql_aes_height__ / 2.0)",
                "(__ggsql_aes_pos2__ + __ggsql_aes_height__ / 2.0)",
            ),
            (
                "y + ymin",
                vec!["pos2", "pos2min"],
                "__ggsql_aes_pos2min__",
                "(2 * __ggsql_aes_pos2__ - __ggsql_aes_pos2min__)",
            ),
            (
                "y + ymax",
                vec!["pos2", "pos2max"],
                "(2 * __ggsql_aes_pos2__ - __ggsql_aes_pos2max__)",
                "__ggsql_aes_pos2max__",
            ),
            (
                "ymin + height",
                vec!["pos2min", "height"],
                "__ggsql_aes_pos2min__",
                "(__ggsql_aes_pos2min__ + __ggsql_aes_height__)",
            ),
            (
                "ymax + height",
                vec!["pos2max", "height"],
                "(__ggsql_aes_pos2max__ - __ggsql_aes_height__)",
                "__ggsql_aes_pos2max__",
            ),
        ];

        for (name, y_aesthetics, expected_min, expected_max) in test_cases {
            // Combine y aesthetics with fixed x mappings (xmin + xmax)
            let mut all_mappings = vec!["pos1min", "pos1max"];
            all_mappings.extend_from_slice(&y_aesthetics);

            let aesthetics = create_aesthetics(&all_mappings);
            let schema = create_schema(&[]);
            let group_by = vec![];
            let parameters = HashMap::new();

            let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);

            assert!(result.is_ok(), "{}: stat_rect failed: {:?}", name, result.err());
            let stat_result = result.unwrap();

            if let StatResult::Transformed { query, stat_columns, .. } = stat_result {
                let stat_pos2min = naming::stat_column("pos2min");
                let stat_pos2max = naming::stat_column("pos2max");
                assert!(query.contains(&format!("{} AS {}", expected_min, stat_pos2min)),
                    "{}: Expected '{} AS {}' in query, got: {}", name, expected_min, stat_pos2min, query);
                assert!(query.contains(&format!("{} AS {}", expected_max, stat_pos2max)),
                    "{}: Expected '{} AS {}' in query, got: {}", name, expected_max, stat_pos2max, query);
                assert!(stat_columns.contains(&"pos2min".to_string()), "{}: Missing pos2min in stat_columns", name);
                assert!(stat_columns.contains(&"pos2max".to_string()), "{}: Missing pos2max in stat_columns", name);
            } else {
                panic!("{}: Expected Transformed result", name);
            }
        }
    }

    // ==================== Discrete Scale Tests ====================

    #[test]
    fn test_discrete_x_with_width() {
        let aesthetics = create_aesthetics(&["pos1", "width", "pos2min", "pos2max"]);
        let schema = create_schema(&["pos1"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, stat_columns, .. }) = result {
            assert!(query.contains("__ggsql_aes_pos1__ AS __ggsql_stat_pos1"));
            assert!(query.contains("__ggsql_aes_width__ AS __ggsql_stat_width"));
            assert!(stat_columns.contains(&"pos1".to_string()));
            assert!(stat_columns.contains(&"width".to_string()));
            assert!(stat_columns.contains(&"pos2min".to_string()));
            assert!(stat_columns.contains(&"pos2max".to_string()));
        }
    }

    #[test]
    fn test_discrete_y_with_height() {
        let aesthetics = create_aesthetics(&["pos1min", "pos1max", "pos2", "height"]);
        let schema = create_schema(&["pos2"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, stat_columns, .. }) = result {
            assert!(query.contains("__ggsql_aes_pos2__ AS __ggsql_stat_pos2"));
            assert!(query.contains("__ggsql_aes_height__ AS __ggsql_stat_height"));
            assert!(stat_columns.contains(&"pos1min".to_string()));
            assert!(stat_columns.contains(&"pos1max".to_string()));
            assert!(stat_columns.contains(&"pos2".to_string()));
            assert!(stat_columns.contains(&"height".to_string()));
        }
    }

    #[test]
    fn test_discrete_both_directions() {
        let aesthetics = create_aesthetics(&["pos1", "width", "pos2", "height"]);
        let schema = create_schema(&["pos1", "pos2"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, stat_columns, .. }) = result {
            assert!(query.contains("__ggsql_aes_pos1__ AS __ggsql_stat_pos1"));
            assert!(query.contains("__ggsql_aes_width__ AS __ggsql_stat_width"));
            assert!(query.contains("__ggsql_aes_pos2__ AS __ggsql_stat_pos2"));
            assert!(query.contains("__ggsql_aes_height__ AS __ggsql_stat_height"));
            assert_eq!(stat_columns.len(), 4);
        }
    }

    // ==================== Validation Error Tests ====================

    #[test]
    fn test_error_too_few_x_params() {
        let aesthetics = create_aesthetics(&["pos1", "pos2min", "pos2max"]);
        let schema = create_schema(&[]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("exactly 2 x-direction parameters"));
    }

    #[test]
    fn test_error_too_many_x_params() {
        let aesthetics = create_aesthetics(&["pos1", "pos1min", "pos1max", "pos2min", "pos2max"]);
        let schema = create_schema(&[]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("exactly 2 x-direction parameters"));
    }

    #[test]
    fn test_error_discrete_with_min_max() {
        let aesthetics = create_aesthetics(&["pos1", "pos1min", "pos2min", "pos2max"]);
        let schema = create_schema(&["pos1"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Cannot use xmin/xmax with discrete x"));
    }

    #[test]
    fn test_error_discrete_requires_width() {
        let aesthetics = create_aesthetics(&["pos1", "pos2min", "pos2max"]);
        let schema = create_schema(&["pos1"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Discrete x requires x and width"));
    }

    // ==================== Group By Tests ====================

    #[test]
    fn test_width_height_filtered_from_group_by() {
        let aesthetics = create_aesthetics(&["pos1", "width", "pos2", "height"]);
        let schema = create_schema(&["pos1", "pos2"]);
        // width and height in group_by should be filtered out
        let group_by = vec![
            "__ggsql_aes_width__".to_string(),
            "__ggsql_aes_height__".to_string(),
            "__ggsql_aes_fill__".to_string(),
        ];
        let parameters = HashMap::new();

        let result = stat_rect("SELECT * FROM data", &schema, &aesthetics, &group_by, &parameters);
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, .. }) = result {
            // Should only have fill in group by, not width or height
            assert!(query.contains("SELECT __ggsql_aes_fill__,"));
            // width and height should appear as stat columns, not group by
            assert!(query.contains("__ggsql_aes_width__ AS __ggsql_stat_width"));
            assert!(query.contains("__ggsql_aes_height__ AS __ggsql_stat_height"));
        }
    }
}
