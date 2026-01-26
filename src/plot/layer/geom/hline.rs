//! HLine geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// HLine geom - horizontal reference lines
#[derive(Debug, Clone, Copy)]
pub struct HLine;

impl GeomTrait for HLine {
    fn geom_type(&self) -> GeomType {
        GeomType::HLine
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "yintercept",
                "color",
                "colour",
                "stroke",
                "linetype",
                "linewidth",
                "opacity",
            ],
            required: &["yintercept"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for HLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hline")
    }
}
