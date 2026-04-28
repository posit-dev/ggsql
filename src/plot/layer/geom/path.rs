//! Path geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

/// Path geom - connected line segments in order
#[derive(Debug, Clone, Copy)]
pub struct Path;

impl GeomTrait for Path {
    fn geom_type(&self) -> GeomType {
        GeomType::Path
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
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "position",
            default: DefaultParamValue::String("identity"),
            constraint: ParamConstraint::string_option(POSITION_VALUES),
        }];
        PARAMS
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "path")
    }
}
