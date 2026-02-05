//! Polygon geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Polygon geom - arbitrary polygons
#[derive(Debug, Clone, Copy)]
pub struct Polygon;

impl GeomTrait for Polygon {
    fn geom_type(&self) -> GeomType {
        GeomType::Polygon
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "y", "fill", "stroke", "opacity", "linewidth"],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "polygon")
    }
}
