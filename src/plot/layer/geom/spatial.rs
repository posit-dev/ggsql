use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};
use crate::plot::types::DefaultAestheticValue;
use crate::{naming, Mappings};

#[derive(Debug, Clone, Copy)]
pub struct Spatial;

impl GeomTrait for Spatial {
    fn geom_type(&self) -> GeomType {
        GeomType::Spatial
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("geometry", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("#747474")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(0.2)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        dialect: &dyn crate::reader::SqlDialect,
    ) -> crate::Result<StatResult> {
        for stmt in dialect.sql_spatial_setup() {
            execute_query(&stmt)?;
        }

        // Geometry columns use database-native types that don't have an Arrow equivalent.
        // Convert to standard WKB so the writer can parse them with geozero.
        let col = naming::quote_ident(&naming::aesthetic_column("geometry"));
        let wkb_expr = dialect.sql_geometry_to_wkb(&col);
        Ok(StatResult::Transformed {
            query: format!("SELECT * REPLACE ({wkb_expr} AS {col}) FROM ({query})"),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        })
    }
}

impl std::fmt::Display for Spatial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spatial")
    }
}
