//! Tile geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Tile geom - heatmaps and tile-based visualizations
#[derive(Debug, Clone, Copy)]
pub struct Tile;

impl GeomTrait for Tile {
    fn geom_type(&self) -> GeomType {
        GeomType::Tile
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x", "y", "color", "colour", "fill", "stroke", "width", "height", "opacity",
            ],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tile")
    }
}
