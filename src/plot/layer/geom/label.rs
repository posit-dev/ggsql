//! Label geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Label geom - text labels with background
#[derive(Debug, Clone, Copy)]
pub struct Label;

impl GeomTrait for Label {
    fn geom_type(&self) -> GeomType {
        GeomType::Label
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x", "y", "label", "color", "colour", "fill", "stroke", "size", "opacity",
                "family", "fontface", "hjust", "vjust",
            ],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "label")
    }
}
