//! VLine geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// VLine geom - vertical reference lines
#[derive(Debug, Clone, Copy)]
pub struct VLine;

impl GeomTrait for VLine {
    fn geom_type(&self) -> GeomType {
        GeomType::VLine
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "pos1intercept",
                "stroke",
                "linetype",
                "linewidth",
                "opacity",
            ],
            required: &["pos1intercept"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for VLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vline")
    }
}
