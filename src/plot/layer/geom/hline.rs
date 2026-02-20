//! HLine geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// HLine geom - horizontal reference lines
#[derive(Debug, Clone, Copy)]
pub struct HLine;

impl GeomTrait for HLine {
    fn geom_type(&self) -> GeomType {
        GeomType::HLine
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("yintercept", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }
}

impl std::fmt::Display for HLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hline")
    }
}
