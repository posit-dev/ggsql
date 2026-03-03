//! Label geom implementation
use crate::plot::{DefaultParam, DefaultParamValue};

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
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("label", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("family", DefaultAestheticValue::Null),
                ("fontsize", DefaultAestheticValue::Number(11.0)),
                ("fontface", DefaultAestheticValue::String("normal")),
                ("hjust", DefaultAestheticValue::Number(0.5)),
                ("vjust", DefaultAestheticValue::Number(0.5)),
                ("angle", DefaultAestheticValue::Number(0.0)),
            ],
        }
    }

    fn default_params(&self) -> &'static [DefaultParam] {
        &[
            DefaultParam {
                name: "nudge_x",
                default: DefaultParamValue::Null,
            },
            DefaultParam {
                name: "nudge_y",
                default: DefaultParamValue::Null,
            },
            DefaultParam {
                name: "format",
                default: DefaultParamValue::Null,
            },
        ]
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "label")
    }
}
