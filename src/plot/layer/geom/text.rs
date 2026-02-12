//! Text geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Text geom - text labels at positions
#[derive(Debug, Clone, Copy)]
pub struct Text;

impl GeomTrait for Text {
    fn geom_type(&self) -> GeomType {
        GeomType::Text
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x", "y", "label", "stroke", "size", "opacity", "family", "fontface", "hjust",
                "vjust",
            ],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "text")
    }
}
