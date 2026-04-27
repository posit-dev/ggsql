//! Point geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

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
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
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
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "position",
            default: DefaultParamValue::String("identity"),
            constraint: ParamConstraint::string_option(POSITION_VALUES),
        }];
        PARAMS
    }

    fn supports_aggregate(&self) -> bool {
        true
    }

    fn aggregate_slots(&self) -> &'static [u8] {
        &[1, 2]
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "point")
    }
}
