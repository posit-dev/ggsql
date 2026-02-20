//! Label geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// Label geom - text labels with background
#[derive(Debug, Clone, Copy)]
pub struct Label;

impl GeomTrait for Label {
    fn geom_type(&self) -> GeomType {
        GeomType::Label
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("label", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::Null),
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

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "label")
    }
}
