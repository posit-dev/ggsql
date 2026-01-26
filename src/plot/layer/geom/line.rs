//! Line geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Line geom - line charts with connected points
#[derive(Debug, Clone, Copy)]
pub struct Line;

impl GeomTrait for Line {
    fn geom_type(&self) -> GeomType {
        GeomType::Line
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x",
                "y",
                "color",
                "colour",
                "stroke",
                "linetype",
                "linewidth",
                "opacity",
            ],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line")
    }
}
