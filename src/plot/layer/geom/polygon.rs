//! Polygon geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

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
}

impl std::fmt::Display for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "polygon")
    }
}
