//! Segment geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Segment geom - line segments between two points
#[derive(Debug, Clone, Copy)]
pub struct Segment;

impl GeomTrait for Segment {
    fn geom_type(&self) -> GeomType {
        GeomType::Segment
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x",
                "y",
                "xend",
                "yend",
                "color",
                "colour",
                "stroke",
                "linetype",
                "linewidth",
                "opacity",
            ],
            required: &["x", "y", "xend", "yend"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "segment")
    }
}
