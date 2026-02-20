//! Text geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// Text geom - text labels at positions
#[derive(Debug, Clone, Copy)]
pub struct Text;

impl GeomTrait for Text {
    fn geom_type(&self) -> GeomType {
        GeomType::Text
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("label", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::Null),
                ("size", DefaultAestheticValue::Number(11.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("family", DefaultAestheticValue::Null),
                ("fontface", DefaultAestheticValue::Null),
                ("hjust", DefaultAestheticValue::Null),
                ("vjust", DefaultAestheticValue::Null),
            ],
        }
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "text")
    }
}
