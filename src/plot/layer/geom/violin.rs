//! Violin geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType, StatResult};
use crate::{
    plot::{
        geom::types::get_column_name, DefaultAestheticValue, DefaultParam, DefaultParamValue,
        ParameterValue,
    },
    GgsqlError, Mappings, Result,
};
use std::collections::HashMap;

/// Violin geom - violin plots (mirrored density)
#[derive(Debug, Clone, Copy)]
pub struct Violin;

impl GeomTrait for Violin {
    fn geom_type(&self) -> GeomType {
        GeomType::Violin
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x",
                "y",
                "weight",
                "fill",
                "stroke",
                "opacity",
                "linewidth",
                "linetype",
            ],
            required: &["x", "y"],
            hidden: &[],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
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
            ("y", DefaultAestheticValue::Column("y")),
            ("density", DefaultAestheticValue::Column("density")),
        ]
    }

    fn valid_stat_columns(&self) -> &'static [&'static str] {
        &["y", "density", "intensity"]
    }

    fn stat_consumed_aesthetics(&self) -> &'static [&'static str] {
        &["y", "weight"]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
    ) -> Result<StatResult> {
        stat_violin(query, aesthetics, group_by, parameters, execute_query)
    }
}

impl std::fmt::Display for Violin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "violin")
    }
}

fn stat_violin(
    query: &str,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    execute: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
) -> Result<StatResult> {
    // Verify y exists
    if get_column_name(aesthetics, "y").is_none() {
        return Err(GgsqlError::ValidationError(
            "Violin requires 'y' aesthetic mapping (continuous)".to_string(),
        ));
    }

    let mut group_by = group_by.to_vec();
    if let Some(x_col) = get_column_name(aesthetics, "x") {
        // We want to ensure x is included as a grouping
        if !group_by.contains(&x_col) {
            group_by.push(x_col);
        }
    } else {
        return Err(GgsqlError::ValidationError(
            "Violin requires 'x' aesthetic mapping (categorical)".to_string(),
        ));
    }

    super::density::stat_density(
        query,
        aesthetics,
        "y",
        group_by.as_slice(),
        parameters,
        execute,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;
    use crate::reader::duckdb::DuckDBReader;
    use crate::reader::Reader;

    // ==================== Helper Functions ====================

    fn create_basic_aesthetics() -> Mappings {
        let mut aesthetics = Mappings::new();
        aesthetics.insert(
            "x".to_string(),
            AestheticValue::standard_column("species".to_string()),
        );
        aesthetics.insert(
            "y".to_string(),
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

        let result = stat_violin(query, &aesthetics, &groups, &parameters, &execute)
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
                assert_eq!(stat_columns, vec!["y", "intensity", "density"]);

                // Verify consumed aesthetics
                assert_eq!(consumed_aesthetics, vec!["y"]);

                // Execute the generated SQL and verify it works
                let df = execute(&stat_query).expect("Generated SQL should execute");

                // Should have columns: y, density, and species (the x grouping)
                let col_names: Vec<&str> =
                    df.get_column_names().iter().map(|s| s.as_str()).collect();
                assert!(col_names.contains(&"__ggsql_stat_y"));
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

        let result = stat_violin(query, &aesthetics, &groups, &parameters, &execute)
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
                assert_eq!(stat_columns, vec!["y", "intensity", "density"]);

                // Verify consumed aesthetics
                assert_eq!(consumed_aesthetics, vec!["y"]);

                // Execute the generated SQL and verify it works
                let df = execute(&stat_query).expect("Generated SQL should execute");

                // Should have columns: y, density, species (x), and island (color group)
                let col_names: Vec<&str> =
                    df.get_column_names().iter().map(|s| s.as_str()).collect();
                assert!(col_names.contains(&"__ggsql_stat_y"));
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
}
