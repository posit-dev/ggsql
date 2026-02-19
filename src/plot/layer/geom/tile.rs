//! Tile geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// Tile geom - heatmaps and tile-based visualizations
#[derive(Debug, Clone, Copy)]
pub struct Tile;

impl GeomTrait for Tile {
    fn geom_type(&self) -> GeomType {
        GeomType::Tile
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("width", DefaultAestheticValue::Null),
                ("height", DefaultAestheticValue::Null),
                ("opacity", DefaultAestheticValue::Number(1.0)),
            ],
        }
    }
}

impl std::fmt::Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tile")
    }
}
