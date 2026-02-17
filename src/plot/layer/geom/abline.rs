//! AbLine geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// AbLine geom - lines with slope and intercept
#[derive(Debug, Clone, Copy)]
pub struct AbLine;

impl GeomTrait for AbLine {
    fn geom_type(&self) -> GeomType {
        GeomType::AbLine
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "slope",
                "intercept",
                "stroke",
                "linetype",
                "linewidth",
                "opacity",
            ],
            required: &["slope", "intercept"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for AbLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "abline")
    }
}
