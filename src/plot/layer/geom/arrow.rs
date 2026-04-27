//! Arrow geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

/// Arrow geom - line segments with arrowheads
#[derive(Debug, Clone, Copy)]
pub struct Arrow;

impl GeomTrait for Arrow {
    fn geom_type(&self) -> GeomType {
        GeomType::Arrow
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("pos1end", DefaultAestheticValue::Required),
                ("pos2end", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
                ("fill", DefaultAestheticValue::Null),
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

impl std::fmt::Display for Arrow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arrow")
    }
}
