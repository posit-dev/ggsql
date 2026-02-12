//! Violin geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};
use crate::Mappings;

/// Violin geom - violin plots (mirrored density)
#[derive(Debug, Clone, Copy)]
pub struct Violin;

impl GeomTrait for Violin {
    fn geom_type(&self) -> GeomType {
        GeomType::Violin
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "y", "fill", "violin", "opacity"],
            required: &["x", "y"],
            hidden: &[],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    // Note: stat_violin not yet implemented - will return Identity for now
}

impl std::fmt::Display for Violin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "violin")
    }
}
