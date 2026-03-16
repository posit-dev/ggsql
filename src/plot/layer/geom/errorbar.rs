//! ErrorBar geom implementation

use super::{DefaultAesthetics, DefaultParam, DefaultParamValue, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// ErrorBar geom - error bars (confidence intervals)
#[derive(Debug, Clone, Copy)]
pub struct ErrorBar;

impl GeomTrait for ErrorBar {
    fn geom_type(&self) -> GeomType {
        GeomType::ErrorBar
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Null),
                ("pos2", DefaultAestheticValue::Null),
                ("pos2min", DefaultAestheticValue::Null),
                ("pos2max", DefaultAestheticValue::Null),
                ("pos1min", DefaultAestheticValue::Null),
                ("pos1max", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
            DefaultParam {
                name: "position",
                default: DefaultParamValue::String("identity"),
            },
            DefaultParam {
                name: "width",
                default: DefaultParamValue::Number(10.0),
            },
        ]
    }
}

impl std::fmt::Display for ErrorBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "errorbar")
    }
}
