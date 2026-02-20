//! Point geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
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
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("size", DefaultAestheticValue::Number(3.0)),
                ("stroke", DefaultAestheticValue::String("black")),
                ("fill", DefaultAestheticValue::String("#000000B2")),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("shape", DefaultAestheticValue::String("circle")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
            ],
        }
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "point")
    }
}
