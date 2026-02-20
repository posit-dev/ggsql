//! ErrorBar geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
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
                ("x", DefaultAestheticValue::Null),
                ("y", DefaultAestheticValue::Null),
                ("ymin", DefaultAestheticValue::Null),
                ("ymax", DefaultAestheticValue::Null),
                ("xmin", DefaultAestheticValue::Null),
                ("xmax", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
            ],
        }
    }
}

impl std::fmt::Display for ErrorBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "errorbar")
    }
}
