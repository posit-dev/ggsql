//! Polygon geom implementation

use super::types::POSITION_VALUES;
use super::{
    densify_edges, needs_projection, project_position_columns, DefaultAesthetics,
    DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::reader::SqlDialect;
use crate::{Mappings, Result};

/// Polygon geom - arbitrary polygons
#[derive(Debug, Clone, Copy)]
pub struct Polygon;

impl GeomTrait for Polygon {
    fn geom_type(&self) -> GeomType {
        GeomType::Polygon
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
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
        let densified = densify_edges(query, dialect, &columns, partition_by, None, true, 1.0, 360);
        project_position_columns(&densified, projection, dialect, &columns)
    }
}

impl std::fmt::Display for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "polygon")
    }
}
