//! Linear geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// Linear geom - lines with coefficient and intercept
#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl GeomTrait for Linear {
    fn geom_type(&self) -> GeomType {
        GeomType::Linear
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("coef", DefaultAestheticValue::Required),
                ("intercept", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }
}

impl std::fmt::Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "linear")
    }
}
