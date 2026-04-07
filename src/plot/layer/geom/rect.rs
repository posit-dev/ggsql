//! Rect geom implementation with flexible parameter specification

use std::collections::HashMap;

use super::types::POSITION_VALUES;
use super::types::{get_column_name, get_quoted_column_name};
use super::{DefaultAesthetics, GeomTrait, GeomType, ParamConstraint, StatResult};
use crate::naming;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::plot::{DefaultParamValue, ParamDefinition};
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
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
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
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "position",
            default: DefaultParamValue::String("identity"),
            constraint: ParamConstraint::string_option(POSITION_VALUES),
        }];
        PARAMS
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &[
            "pos1", "pos2", "pos1min", "pos1max", "pos2min", "pos2max", "width", "height",
        ]
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
        _dialect: &dyn crate::reader::SqlDialect,
    ) -> Result<StatResult> {
        stat_rect(query, schema, aesthetics, group_by, parameters)
    }
}

impl std::fmt::Display for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rect")
    }
}

/// Process a single direction (x or y) for rect stat transform
/// Returns (select_parts, stat_column_names)
fn process_direction(
    axis: &str,
    aesthetics: &Mappings,
    parameters: &HashMap<String, ParameterValue>,
    schema: &Schema,
) -> Result<(Vec<String>, Vec<String>)> {
    // Derive aesthetic names from axis
    let (center_aes, min_aes, max_aes, size_aes) = match axis {
        "x" => ("pos1", "pos1min", "pos1max", "width"),
        "y" => ("pos2", "pos2min", "pos2max", "height"),
        _ => unreachable!("axis must be 'x' or 'y'"),
    };

    // Get unquoted center name for schema lookup
    let center_unquoted = get_column_name(aesthetics, center_aes);
    let center = center_unquoted.as_deref().map(naming::quote_ident);
    let min = get_quoted_column_name(aesthetics, min_aes);
    let max = get_quoted_column_name(aesthetics, max_aes);
    // SETTING fallback for size is a literal value, no quoting needed.
    let size = get_quoted_column_name(aesthetics, size_aes)
        .or_else(|| parameters.get(size_aes).map(|v| v.to_string()));

    // Detect if discrete by checking schema
    let is_discrete = center_unquoted
        .as_ref()
        .and_then(|col| schema.iter().find(|c| &c.name == col))
        .map(|c| c.is_discrete)
        .unwrap_or(false);

    // Generate position expressions
    let (expr_1, expr_2) = if is_discrete {
        generate_discrete_position_expressions(
            center.as_deref(),
            min.as_deref(),
            max.as_deref(),
            size.as_deref(),
            axis,
        )?
    } else {
        generate_continuous_position_expressions(
            center.as_deref(),
            min.as_deref(),
            max.as_deref(),
            size.as_deref(),
            axis,
        )?
    };

    // Determine stat column names based on discrete vs continuous
    let stat_cols = if is_discrete {
        vec![center_aes.to_string(), size_aes.to_string()]
    } else {
        vec![min_aes.to_string(), max_aes.to_string()]
    };

    // Build SELECT parts using the stat columns
    let select_parts = vec![
        format!("{} AS {}", expr_1, naming::quote_ident(&naming::stat_column(&stat_cols[0]))),
        format!("{} AS {}", expr_2, naming::quote_ident(&naming::stat_column(&stat_cols[1]))),
    ];

    Ok((select_parts, stat_cols))
}

/// Statistical transformation for rect: consolidate parameters and compute min/max
fn stat_rect(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    _group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
) -> Result<StatResult> {
    // Process X direction
    let (x_select, x_stat_cols) = process_direction("x", aesthetics, parameters, schema)?;

    // Process Y direction
    let (y_select, y_stat_cols) = process_direction("y", aesthetics, parameters, schema)?;

    // Define consumed aesthetics (these will be transformed, not passed through)
    let consumed_aesthetic_names = [
        "pos1", "pos1min", "pos1max", "width", "pos2", "pos2min", "pos2max", "height",
    ];

    // Convert aesthetic names to column names for filtering
    let consumed_columns: Vec<String> = consumed_aesthetic_names
        .iter()
        .filter_map(|aes| get_column_name(aesthetics, aes))
        .collect();

    // Build SELECT list starting with non-consumed columns
    let mut select_parts: Vec<String> = schema
        .iter()
        .filter(|col| !consumed_columns.contains(&col.name))
        .map(|col| naming::quote_ident(&col.name))
        .collect();

    // Add X direction SELECT parts and collect stat columns
    select_parts.extend(x_select);
    let mut stat_columns = x_stat_cols;

    // Add Y direction SELECT parts and collect stat columns
    select_parts.extend(y_select);
    stat_columns.extend(y_stat_cols);

    let select_list = select_parts.join(", ");

    // Build transformed query
    let transformed_query = format!(
        "SELECT {} FROM ({}) AS \"__ggsql_rect_stat__\"",
        select_list, query
    );

    // Use the same consumed aesthetic names for StatResult
    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns: vec![],
        consumed_aesthetics: consumed_aesthetic_names
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })
}

/// Generate SQL expressions for discrete position (returns center, size)
///
/// Validates:
/// - Discrete scales cannot use min/max aesthetics
/// - Center is required
/// - Size defaults to "1.0" if not provided
fn generate_discrete_position_expressions(
    center: Option<&str>,
    min: Option<&str>,
    max: Option<&str>,
    size: Option<&str>,
    axis: &str,
) -> Result<(String, String)> {
    // Validate: discrete scales cannot use min/max
    if min.is_some() || max.is_some() {
        return Err(GgsqlError::ValidationError(format!(
            "Cannot use {}min/{}max with discrete {} aesthetic. Use {} + {} instead.",
            axis,
            axis,
            axis,
            axis,
            if axis == "x" { "width" } else { "height" }
        )));
    }

    match center {
        Some(c) => Ok((c.to_string(), size.unwrap_or("1.0").to_string())),
        None => Err(GgsqlError::ValidationError(format!(
            "Discrete {} requires {}.",
            axis, axis
        ))),
    }
}

/// Generate SQL expressions for continuous position (returns min_expr, max_expr)
///
/// Handles all 7 valid parameter combinations:
/// - min + max
/// - center + size
/// - center only (defaults size to 1.0)
/// - center + min
/// - center + max
/// - min + size
/// - max + size
fn generate_continuous_position_expressions(
    center: Option<&str>,
    min: Option<&str>,
    max: Option<&str>,
    size: Option<&str>,
    axis: &str,
) -> Result<(String, String)> {
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
        // Case 2b: center only (default size to 1.0)
        (Some(c), None, None, None) => Ok((format!("({} - 0.5)", c), format!("({} + 0.5)", c))),
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
        create_schema_with_extra(discrete_cols, &[])
    }

    fn create_schema_with_extra(discrete_cols: &[&str], extra_cols: &[&str]) -> Schema {
        let mut schema = vec![
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
        ];

        // Add extra columns (e.g., fill, color, etc.)
        for col_name in extra_cols {
            schema.push(ColumnInfo {
                name: col_name.to_string(),
                dtype: DataType::String,
                is_discrete: true,
                min: None,
                max: None,
            });
        }

        schema
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
                "\"__ggsql_aes_pos1min__\"",
                "\"__ggsql_aes_pos1max__\"",
            ),
            (
                "x + width",
                vec!["pos1", "width"],
                "(\"__ggsql_aes_pos1__\" - \"__ggsql_aes_width__\" / 2.0)",
                "(\"__ggsql_aes_pos1__\" + \"__ggsql_aes_width__\" / 2.0)",
            ),
            (
                "x only (default width 1.0)",
                vec!["pos1"],
                "(\"__ggsql_aes_pos1__\" - 0.5)",
                "(\"__ggsql_aes_pos1__\" + 0.5)",
            ),
            (
                "x + xmin",
                vec!["pos1", "pos1min"],
                "\"__ggsql_aes_pos1min__\"",
                "(2 * \"__ggsql_aes_pos1__\" - \"__ggsql_aes_pos1min__\")",
            ),
            (
                "x + xmax",
                vec!["pos1", "pos1max"],
                "(2 * \"__ggsql_aes_pos1__\" - \"__ggsql_aes_pos1max__\")",
                "\"__ggsql_aes_pos1max__\"",
            ),
            (
                "xmin + width",
                vec!["pos1min", "width"],
                "\"__ggsql_aes_pos1min__\"",
                "(\"__ggsql_aes_pos1min__\" + \"__ggsql_aes_width__\")",
            ),
            (
                "xmax + width",
                vec!["pos1max", "width"],
                "(\"__ggsql_aes_pos1max__\" - \"__ggsql_aes_width__\")",
                "\"__ggsql_aes_pos1max__\"",
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

            let result = stat_rect(
                "SELECT * FROM data",
                &schema,
                &aesthetics,
                &group_by,
                &parameters,
            );

            assert!(
                result.is_ok(),
                "{}: stat_rect failed: {:?}",
                name,
                result.err()
            );
            let stat_result = result.unwrap();

            if let StatResult::Transformed {
                query,
                stat_columns,
                ..
            } = stat_result
            {
                let stat_pos1min = naming::stat_column("pos1min");
                let stat_pos1max = naming::stat_column("pos1max");
                assert!(
                    query.contains(&format!("{} AS \"{}\"", expected_min, stat_pos1min)),
                    "{}: Expected '{} AS {}' in query, got: {}",
                    name,
                    expected_min,
                    stat_pos1min,
                    query
                );
                assert!(
                    query.contains(&format!("{} AS \"{}\"", expected_max, stat_pos1max)),
                    "{}: Expected '{} AS {}' in query, got: {}",
                    name,
                    expected_max,
                    stat_pos1max,
                    query
                );
                assert!(
                    stat_columns.contains(&"pos1min".to_string()),
                    "{}: Missing pos1min in stat_columns",
                    name
                );
                assert!(
                    stat_columns.contains(&"pos1max".to_string()),
                    "{}: Missing pos1max in stat_columns",
                    name
                );
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
                "\"__ggsql_aes_pos2min__\"",
                "\"__ggsql_aes_pos2max__\"",
            ),
            (
                "y + height",
                vec!["pos2", "height"],
                "(\"__ggsql_aes_pos2__\" - \"__ggsql_aes_height__\" / 2.0)",
                "(\"__ggsql_aes_pos2__\" + \"__ggsql_aes_height__\" / 2.0)",
            ),
            (
                "y + ymin",
                vec!["pos2", "pos2min"],
                "\"__ggsql_aes_pos2min__\"",
                "(2 * \"__ggsql_aes_pos2__\" - \"__ggsql_aes_pos2min__\")",
            ),
            (
                "y + ymax",
                vec!["pos2", "pos2max"],
                "(2 * \"__ggsql_aes_pos2__\" - \"__ggsql_aes_pos2max__\")",
                "\"__ggsql_aes_pos2max__\"",
            ),
            (
                "ymin + height",
                vec!["pos2min", "height"],
                "\"__ggsql_aes_pos2min__\"",
                "(\"__ggsql_aes_pos2min__\" + \"__ggsql_aes_height__\")",
            ),
            (
                "ymax + height",
                vec!["pos2max", "height"],
                "(\"__ggsql_aes_pos2max__\" - \"__ggsql_aes_height__\")",
                "\"__ggsql_aes_pos2max__\"",
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

            let result = stat_rect(
                "SELECT * FROM data",
                &schema,
                &aesthetics,
                &group_by,
                &parameters,
            );

            assert!(
                result.is_ok(),
                "{}: stat_rect failed: {:?}",
                name,
                result.err()
            );
            let stat_result = result.unwrap();

            if let StatResult::Transformed {
                query,
                stat_columns,
                ..
            } = stat_result
            {
                let stat_pos2min = naming::stat_column("pos2min");
                let stat_pos2max = naming::stat_column("pos2max");
                assert!(
                    query.contains(&format!("{} AS \"{}\"", expected_min, stat_pos2min)),
                    "{}: Expected '{} AS {}' in query, got: {}",
                    name,
                    expected_min,
                    stat_pos2min,
                    query
                );
                assert!(
                    query.contains(&format!("{} AS \"{}\"", expected_max, stat_pos2max)),
                    "{}: Expected '{} AS {}' in query, got: {}",
                    name,
                    expected_max,
                    stat_pos2max,
                    query
                );
                assert!(
                    stat_columns.contains(&"pos2min".to_string()),
                    "{}: Missing pos2min in stat_columns",
                    name
                );
                assert!(
                    stat_columns.contains(&"pos2max".to_string()),
                    "{}: Missing pos2max in stat_columns",
                    name
                );
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

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed {
            query,
            stat_columns,
            ..
        }) = result
        {
            assert!(query.contains("\"__ggsql_aes_pos1__\" AS \"__ggsql_stat_pos1"));
            assert!(query.contains("\"__ggsql_aes_width__\" AS \"__ggsql_stat_width"));
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

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed {
            query,
            stat_columns,
            ..
        }) = result
        {
            assert!(query.contains("\"__ggsql_aes_pos2__\" AS \"__ggsql_stat_pos2"));
            assert!(query.contains("\"__ggsql_aes_height__\" AS \"__ggsql_stat_height"));
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

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed {
            query,
            stat_columns,
            ..
        }) = result
        {
            assert!(query.contains("\"__ggsql_aes_pos1__\" AS \"__ggsql_stat_pos1"));
            assert!(query.contains("\"__ggsql_aes_width__\" AS \"__ggsql_stat_width"));
            assert!(query.contains("\"__ggsql_aes_pos2__\" AS \"__ggsql_stat_pos2"));
            assert!(query.contains("\"__ggsql_aes_height__\" AS \"__ggsql_stat_height"));
            assert_eq!(stat_columns.len(), 4);
        }
    }

    // ==================== Validation Error Tests ====================

    #[test]
    fn test_continuous_x_defaults_width() {
        // Test that continuous x without explicit width defaults to 1.0
        let aesthetics = create_aesthetics(&["pos1", "pos2min", "pos2max"]);
        let schema = create_schema(&[]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());
        let stat_result = result.unwrap();
        match stat_result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("(\"__ggsql_aes_pos1__\" - 0.5)"));
                assert!(query.contains("(\"__ggsql_aes_pos1__\" + 0.5)"));
                assert!(stat_columns.contains(&"pos1min".to_string()));
                assert!(stat_columns.contains(&"pos1max".to_string()));
            }
            _ => panic!("Expected Transformed"),
        }
    }

    #[test]
    fn test_error_too_many_x_params() {
        let aesthetics = create_aesthetics(&["pos1", "pos1min", "pos1max", "pos2min", "pos2max"]);
        let schema = create_schema(&[]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
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

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Cannot use xmin/xmax with discrete x"));
    }

    #[test]
    fn test_discrete_x_defaults_width() {
        // Test that discrete x without explicit width defaults to 1.0
        let aesthetics = create_aesthetics(&["pos1", "pos2min", "pos2max"]);
        let schema = create_schema(&["pos1"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());
        let stat_result = result.unwrap();
        match stat_result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("1.0 AS \"__ggsql_stat_width"));
                assert!(stat_columns.contains(&"width".to_string()));
            }
            _ => panic!("Expected Transformed"),
        }
    }

    // ==================== Non-Consumed Aesthetic Tests ====================

    #[test]
    fn test_non_consumed_aesthetics_passed_through() {
        let aesthetics = create_aesthetics(&["pos1", "width", "pos2", "height"]);
        // Include fill in schema (it's a non-consumed aesthetic)
        let schema = create_schema_with_extra(&["pos1", "pos2"], &["__ggsql_aes_fill__"]);
        let group_by = vec![];
        let parameters = HashMap::new();

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, .. }) = result {
            // Should include fill column (non-consumed aesthetic from schema, quoted)
            assert!(query.contains("\"__ggsql_aes_fill__\""));
            // Should NOT include width/height as pass-through (they're consumed)
            // They should only appear as stat columns
            assert!(query.contains("\"__ggsql_aes_width__\" AS \"__ggsql_stat_width"));
            assert!(query.contains("\"__ggsql_aes_height__\" AS \"__ggsql_stat_height"));
        }
    }

    #[test]
    fn test_setting_width_as_fallback() {
        // Test that SETTING width/height are used when no MAPPING is provided
        let aesthetics = create_aesthetics(&["pos1", "pos2"]);
        let schema = create_schema(&["pos1", "pos2"]);
        let group_by = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("width".to_string(), ParameterValue::Number(0.7));
        parameters.insert("height".to_string(), ParameterValue::Number(0.9));

        let result = stat_rect(
            "SELECT * FROM data",
            &schema,
            &aesthetics,
            &group_by,
            &parameters,
        );
        assert!(result.is_ok());

        if let Ok(StatResult::Transformed { query, .. }) = result {
            // Should use SETTING values as SQL literals
            assert!(query.contains("0.7 AS \"__ggsql_stat_width"));
            assert!(query.contains("0.9 AS \"__ggsql_stat_height"));
        }
    }
}
