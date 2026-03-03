//! Text geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{DefaultParam, DefaultParamValue};

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

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "text")
    }
}
