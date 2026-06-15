//! Point geom implementation

use super::types::POSITION_VALUES;
use super::{
    project_position_columns, DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType,
    ParamConstraint, ParamDefinition,
};
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::reader::SqlDialect;
use crate::{Mappings, Result};

/// Point geom - scatter plots and similar
#[derive(Debug, Clone, Copy)]
pub struct Point;

impl GeomTrait for Point {
    fn geom_type(&self) -> GeomType {
        GeomType::Point
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                // Both axes are dummy-able. Whichever the user omits is
                // synthesised as a dummy categorical column by the default
                // `apply_stat_transform`; the writer then hides that axis.
                // Mapping neither degrades to all points overlapping at a
                // single dummy spot — useful only with `aggregate`.
                ("pos1", DefaultAestheticValue::Dummy),
                ("pos2", DefaultAestheticValue::Dummy),
                ("size", DefaultAestheticValue::Number(3.0)),
                ("stroke", DefaultAestheticValue::String("black")),
                ("fill", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("shape", DefaultAestheticValue::String("circle")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("identity"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            super::types::AGGREGATE_PARAM,
        ];
        PARAMS
    }

    fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        Some(&[])
    }

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        mappings: &mut Mappings,
        _partition_by: &mut Vec<String>,
        _parameters: &mut std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> Result<String> {
        let columns = mappings.column_names();
        project_position_columns(query, projection, dialect, &columns)
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "point")
    }
}
