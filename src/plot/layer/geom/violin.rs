//! Violin geom implementation

use super::types::POSITION_VALUES;
use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};
use crate::{
    naming,
    plot::{
        geom::types::get_column_name, DefaultAestheticValue, DefaultParamValue, ParamConstraint,
        ParamDefinition, ParameterValue,
    },
    DataFrame, GgsqlError, Mappings, Result,
};
use polars::prelude::*;
use std::collections::HashMap;

/// Valid kernel types for violin density estimation
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

const ALIGN_VALUES: &[&str] = &["center", "centre", "left", "top", "right", "bottom"];

/// Violin geom - violin plots (mirrored density)
#[derive(Debug, Clone, Copy)]
pub struct Violin;

impl GeomTrait for Violin {
    fn geom_type(&self) -> GeomType {
        GeomType::Violin
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
                ("offset", DefaultAestheticValue::Delayed), // Computed by stat, used for violin shape
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
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
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("dodge"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            ParamDefinition {
                name: "width",
                default: DefaultParamValue::Number(0.9),
                // We allow >1 width to make ridgeline plots
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "align",
                default: DefaultParamValue::String("center"),
                constraint: ParamConstraint::string_option(ALIGN_VALUES),
            },
            ParamDefinition {
                name: "tails",
                default: DefaultParamValue::Number(3.0),
                constraint: ParamConstraint::number_min(0.0),
            },
        ];
        PARAMS
    }

    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos2", DefaultAestheticValue::Column("pos2")),
                ("offset", DefaultAestheticValue::Column("density")),
            ],
        }
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["pos2", "density", "intensity"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["pos2", "weight"]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
        dialect: &dyn crate::reader::SqlDialect,
    ) -> Result<StatResult> {
        stat_violin(query, aesthetics, group_by, parameters, dialect)
    }

    /// Post-process the violin DataFrame to scale offset to [0, 0.5 * width].
    ///
    /// Uses global max normalization so relative differences across groups are preserved:
    /// - Narrow distributions will have higher peaks (normalized density)
    /// - Groups with more data will be wider when using intensity remapping
    fn post_process(
        &self,
        df: DataFrame,
        parameters: &HashMap<String, ParameterValue>,
    ) -> Result<DataFrame> {
        let offset_col = naming::aesthetic_column("offset");

        // Get width parameter (default 0.9)
        let width = parameters
            .get("width")
            .and_then(|v| match v {
                ParameterValue::Number(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0.9);
        let half_width = 0.5 * width;

        scale_offset_column(df, &offset_col, half_width)
    }
}

impl std::fmt::Display for Violin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "violin")
    }
}

/// Scale the offset column to [0, half_width] using global max normalization.
///
/// new_offset = offset * half_width / global_max
fn scale_offset_column(df: DataFrame, offset_col: &str, half_width: f64) -> Result<DataFrame> {
    // Check if offset column exists
    if df.column(offset_col).is_err() {
        // No offset column, return unchanged
        return Ok(df);
    }

    // Get global max of offset column
    let max_val = df
        .column(offset_col)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to get offset column: {}", e)))?
        .f64()
        .map_err(|e| GgsqlError::InternalError(format!("Offset column must be f64: {}", e)))?
        .max()
        .unwrap_or(1.0);

    if max_val <= 0.0 {
        return Ok(df);
    }

    // Scale: new_offset = offset * half_width / max_val
    let scale_factor = half_width / max_val;
    let scaled = df
        .lazy()
        .with_column((col(offset_col) * lit(scale_factor)).alias(offset_col))
        .collect()
        .map_err(|e| GgsqlError::InternalError(format!("Failed to scale offset: {}", e)))?;

    Ok(scaled)
}

fn stat_violin(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn crate::reader::SqlDialect,
) -> Result<StatResult> {
    // Verify y exists
    if get_column_name(aesthetics, "pos2").is_none() {
        return Err(GgsqlError::ValidationError(
            "Violin requires 'y' aesthetic mapping (continuous)".to_string(),
        ));
    }

    let mut group_by = group_by.to_vec();
    if let Some(x_col) = get_column_name(aesthetics, "pos1") {
        // We want to ensure x is included as a grouping
        if !group_by.contains(&x_col) {
            group_by.push(x_col);
        }
    } else {
        return Err(GgsqlError::ValidationError(
            "Violin requires 'x' aesthetic mapping (categorical)".to_string(),
        ));
    }

    // Violin uses tails parameter from user (default 3.0 set in default_params)
    super::density::stat_density(
        query,
        aesthetics,
        "pos2",
        None,
        group_by.as_slice(),
        parameters,
        dialect,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;
    use crate::reader::duckdb::DuckDBReader;
    use crate::reader::AnsiDialect;
    use crate::reader::Reader;

    // ==================== Helper Functions ====================

    fn create_basic_aesthetics() -> Mappings {
        let mut aesthetics = Mappings::new();
        aesthetics.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("species".to_string()),
        );
        aesthetics.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("flipper_length".to_string()),
        );
        aesthetics
    }

    fn create_aesthetics_with_color() -> Mappings {
        let mut aesthetics = create_basic_aesthetics();
        aesthetics.insert(
            "color".to_string(),
            AestheticValue::standard_column("island".to_string()),
        );
        aesthetics
    }

    // ==================== Basic Behavior Tests ====================

    #[test]
    fn test_violin_no_extra_groups() {
        // Test violin with just x and y (no additional grouping variables)
        let query = "SELECT species, flipper_length FROM penguins";
        let aesthetics = create_basic_aesthetics();
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(5.0));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        let setup_sql = "CREATE TABLE penguins AS SELECT * FROM (VALUES
            ('Adelie', 181.0), ('Adelie', 186.0), ('Adelie', 195.0),
            ('Gentoo', 217.0), ('Gentoo', 221.0), ('Gentoo', 230.0),
            ('Chinstrap', 192.0), ('Chinstrap', 196.0), ('Chinstrap', 201.0)
        ) AS t(species, flipper_length)";
        reader.execute_sql(setup_sql).unwrap();

        let execute = |sql: &str| reader.execute_sql(sql);

        let result = stat_violin(query, &aesthetics, &groups, &parameters, &AnsiDialect)
            .expect("stat_violin should succeed");

        // Verify the result is a transformed stat result
        match result {
            StatResult::Transformed {
                query: stat_query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                // Verify stat columns (includes intensity from density stat)
                assert_eq!(stat_columns, vec!["pos2", "intensity", "density"]);

                // Verify consumed aesthetics
                assert_eq!(consumed_aesthetics, vec!["pos2"]);

                // Execute the generated SQL and verify it works
                let df = execute(&stat_query).expect("Generated SQL should execute");

                // Should have columns: pos2 (y), density, and species (the x grouping)
                let col_names: Vec<&str> =
                    df.get_column_names().iter().map(|s| s.as_str()).collect();
                assert!(col_names.contains(&"__ggsql_stat_pos2"));
                assert!(col_names.contains(&"__ggsql_stat_density"));
                assert!(col_names.contains(&"species"));

                // Should have multiple rows per species (512 grid points per species)
                assert!(df.height() > 0);

                // Verify we have all three species
                let species_col = df.column("species").unwrap();
                let unique_species = species_col.n_unique().unwrap();
                assert_eq!(unique_species, 3, "Should have 3 unique species");
            }
            _ => panic!("Expected Transformed result"),
        }
    }

    #[test]
    fn test_violin_with_extra_groups() {
        // Test violin with x, y, and an additional color grouping variable
        let query = "SELECT species, flipper_length, island FROM penguins";
        let aesthetics = create_aesthetics_with_color();
        let groups = vec!["island".to_string()]; // Additional grouping via color aesthetic
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(5.0));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with multiple islands
        let setup_sql = "CREATE TABLE penguins AS SELECT * FROM (VALUES
            ('Adelie', 181.0, 'Torgersen'), ('Adelie', 186.0, 'Torgersen'),
            ('Adelie', 195.0, 'Biscoe'), ('Adelie', 190.0, 'Biscoe'),
            ('Gentoo', 217.0, 'Biscoe'), ('Gentoo', 221.0, 'Biscoe'),
            ('Chinstrap', 192.0, 'Dream'), ('Chinstrap', 196.0, 'Dream')
        ) AS t(species, flipper_length, island)";
        reader.execute_sql(setup_sql).unwrap();

        let execute = |sql: &str| reader.execute_sql(sql);

        let result = stat_violin(query, &aesthetics, &groups, &parameters, &AnsiDialect)
            .expect("stat_violin should succeed");

        // Verify the result is a transformed stat result
        match result {
            StatResult::Transformed {
                query: stat_query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                // Verify stat columns (includes intensity from density stat)
                assert_eq!(stat_columns, vec!["pos2", "intensity", "density"]);

                // Verify consumed aesthetics
                assert_eq!(consumed_aesthetics, vec!["pos2"]);

                // Execute the generated SQL and verify it works
                let df = execute(&stat_query).expect("Generated SQL should execute");

                // Should have columns: pos2 (y), density, species (x), and island (color group)
                let col_names: Vec<&str> =
                    df.get_column_names().iter().map(|s| s.as_str()).collect();
                assert!(col_names.contains(&"__ggsql_stat_pos2"));
                assert!(col_names.contains(&"__ggsql_stat_density"));
                assert!(col_names.contains(&"species"));
                assert!(col_names.contains(&"island"));

                // Should have multiple rows per species-island combination
                assert!(df.height() > 0);

                // Verify we have multiple species
                let species_col = df.column("species").unwrap();
                let unique_species = species_col.n_unique().unwrap();
                assert!(unique_species >= 2, "Should have at least 2 unique species");

                // Verify we have multiple islands
                let island_col = df.column("island").unwrap();
                let unique_islands = island_col.n_unique().unwrap();
                assert!(unique_islands >= 2, "Should have at least 2 unique islands");
            }
            _ => panic!("Expected Transformed result"),
        }
    }

    #[test]
    fn test_violin_width_parameter() {
        // Verify that the violin geom has a width parameter with default 0.9
        let violin = Violin;
        let params = violin.default_params();

        let width_param = params.iter().find(|p| p.name == "width");
        assert!(
            width_param.is_some(),
            "Violin should have a 'width' parameter"
        );

        if let Some(param) = width_param {
            match param.default {
                DefaultParamValue::Number(n) => {
                    assert!(
                        (n - 0.9).abs() < 1e-6,
                        "Default width should be 0.9, got {}",
                        n
                    );
                }
                _ => panic!("Width parameter should have a numeric default"),
            }
        }
    }

    #[test]
    fn test_violin_tails_parameter() {
        // Verify that the violin geom has a tails parameter with default 3.0
        let violin = Violin;
        let params = violin.default_params();

        let tails_param = params.iter().find(|p| p.name == "tails");
        assert!(
            tails_param.is_some(),
            "Violin should have a 'tails' parameter"
        );

        if let Some(param) = tails_param {
            match param.default {
                DefaultParamValue::Number(n) => {
                    assert!(
                        (n - 3.0).abs() < 1e-6,
                        "Default tails should be 3.0, got {}",
                        n
                    );
                }
                _ => panic!("Tails parameter should have a numeric default"),
            }
        }

        // Test with custom tails value
        let query = "SELECT species, flipper_length FROM penguins";
        let aesthetics = create_basic_aesthetics();
        let groups: Vec<String> = vec![];
        let mut parameters = HashMap::new();
        parameters.insert("bandwidth".to_string(), ParameterValue::Number(5.0));
        parameters.insert(
            "kernel".to_string(),
            ParameterValue::String("gaussian".to_string()),
        );
        parameters.insert("tails".to_string(), ParameterValue::Number(1.5)); // Custom tails

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        let setup_sql = "CREATE TABLE penguins AS SELECT * FROM (VALUES
            ('Adelie', 181.0), ('Adelie', 186.0), ('Adelie', 195.0),
            ('Gentoo', 217.0), ('Gentoo', 221.0), ('Gentoo', 230.0)
        ) AS t(species, flipper_length)";
        reader.execute_sql(setup_sql).unwrap();

        let execute = |sql: &str| reader.execute_sql(sql);

        let result = stat_violin(query, &aesthetics, &groups, &parameters, &AnsiDialect)
            .expect("stat_violin with custom tails should succeed");

        // Verify the SQL includes the tails constraint
        match result {
            StatResult::Transformed {
                query: stat_query, ..
            } => {
                // The generated SQL should include the tails filtering
                // We verify this by checking the SQL contains the bandwidth filtering
                assert!(
                    stat_query.contains("1.5"),
                    "SQL should contain the custom tails value 1.5"
                );

                // Execute and verify it produces results
                let df = execute(&stat_query).expect("Generated SQL should execute");
                assert!(df.height() > 0, "Should produce density data");
            }
            _ => panic!("Expected Transformed result"),
        }
    }

    // ==================== Post-Process Tests ====================

    #[test]
    fn test_violin_post_process_scales_offset() {
        let violin = Violin;
        let offset_col = naming::aesthetic_column("offset");

        // Create a DataFrame with offset values
        let df = df! {
            offset_col.as_str() => [0.0, 0.5, 1.0, 0.25],
            "__ggsql_aes_pos2__" => [1.0, 2.0, 3.0, 4.0],
        }
        .unwrap();

        // With default width 0.9, half_width = 0.45
        // Offset should be scaled to [0, 0.45]
        let parameters = HashMap::new();
        let result = violin.post_process(df, &parameters).unwrap();

        let scaled_offset = result.column(&offset_col).unwrap().f64().unwrap();
        let values: Vec<f64> = scaled_offset.into_iter().flatten().collect();

        // Max offset (1.0) should be scaled to 0.45 (half_width)
        // Other values should be proportionally scaled
        assert!((values[0] - 0.0).abs() < 1e-6, "0.0 should stay 0.0");
        assert!((values[1] - 0.225).abs() < 1e-6, "0.5 should become 0.225");
        assert!((values[2] - 0.45).abs() < 1e-6, "1.0 should become 0.45");
        assert!(
            (values[3] - 0.1125).abs() < 1e-6,
            "0.25 should become 0.1125"
        );
    }

    #[test]
    fn test_violin_post_process_custom_width() {
        let violin = Violin;
        let offset_col = naming::aesthetic_column("offset");

        // Create a DataFrame with offset values
        let df = df! {
            offset_col.as_str() => [0.0, 0.5, 1.0],
            "__ggsql_aes_pos2__" => [1.0, 2.0, 3.0],
        }
        .unwrap();

        // With width 0.6, half_width = 0.3
        let mut parameters = HashMap::new();
        parameters.insert("width".to_string(), ParameterValue::Number(0.6));

        let result = violin.post_process(df, &parameters).unwrap();

        let scaled_offset = result.column(&offset_col).unwrap().f64().unwrap();
        let values: Vec<f64> = scaled_offset.into_iter().flatten().collect();

        // Max offset (1.0) should be scaled to 0.3 (half_width)
        assert!((values[0] - 0.0).abs() < 1e-6, "0.0 should stay 0.0");
        assert!((values[1] - 0.15).abs() < 1e-6, "0.5 should become 0.15");
        assert!((values[2] - 0.3).abs() < 1e-6, "1.0 should become 0.3");
    }

    #[test]
    fn test_violin_post_process_no_offset_column() {
        let violin = Violin;

        // Create a DataFrame without offset column
        let df = df! {
            "__ggsql_aes_pos2__" => [1.0, 2.0, 3.0],
        }
        .unwrap();

        let parameters = HashMap::new();
        let result = violin.post_process(df.clone(), &parameters).unwrap();

        // Should return unchanged DataFrame
        assert_eq!(result.height(), df.height());
    }
}
