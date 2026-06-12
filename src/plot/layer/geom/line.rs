//! Line geom implementation

use super::stat_aggregate;
use super::types::wrap_with_order_by;
use super::{
    densify_edges, has_aggregate_param, needs_projection, project_position_columns,
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
    StatResult,
};
use crate::plot::layer::orientation::{ALIGNED, ORIENTATION_VALUES};
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::reader::SqlDialect;
use crate::{naming, Mappings, Result};

/// Line geom - line charts with connected points
#[derive(Debug, Clone, Copy)]
pub struct Line;

impl GeomTrait for Line {
    fn geom_type(&self) -> GeomType {
        GeomType::Line
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.5)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "orientation",
                default: DefaultParamValue::String(ALIGNED),
                constraint: ParamConstraint::string_option(ORIENTATION_VALUES),
            },
            super::types::AGGREGATE_PARAM,
        ];
        PARAMS
    }

    fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        Some(&["pos1"])
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        dialect: &dyn crate::reader::SqlDialect,
        aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> crate::Result<StatResult> {
        let result = if has_aggregate_param(parameters) {
            stat_aggregate::apply(
                query,
                schema,
                aesthetics,
                group_by,
                parameters,
                dialect,
                aesthetic_ctx,
                self.aggregate_domain_aesthetics().unwrap_or(&[]),
            )?
        } else {
            StatResult::Identity
        };
        // Line needs ordering by pos1 (domain axis) for proper rendering, in both
        // the Identity and Aggregate paths.
        Ok(wrap_with_order_by(query, result, "pos1"))
    }

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        mappings: &mut Mappings,
        partition_by: &mut Vec<String>,
        _parameters: &mut std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> Result<String> {
        if !needs_projection(projection) {
            return Ok(query.to_string());
        }
        let columns = mappings.column_names();
        let pos1_col = naming::aesthetic_column("pos1");
        let densified = densify_edges(
            query,
            dialect,
            &columns,
            partition_by,
            Some(&pos1_col),
            false,
            1.0,
            360,
        );
        project_position_columns(&densified, projection, dialect, &columns)
    }
}

impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::types::ParameterValue;
    use crate::reader::AnsiDialect;

    #[test]
    fn test_apply_projection_densifies_and_transforms() {
        let line = Line;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=0 +lon_0=0".to_string()),
        );

        let mut mappings = Mappings::new();
        mappings.insert_column("pos1", "pos1");
        mappings.insert_column("pos2", "pos2");
        let result = line
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut mappings,
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        // Densification happened
        assert!(result.contains("__ggsql_seq__"));
        assert!(result.contains("LEAD("));
        // Projection happened
        assert!(result.contains("ST_Transform"));
    }
}
