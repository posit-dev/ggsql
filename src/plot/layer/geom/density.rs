//! Density geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};
use crate::Mappings;

/// Density geom - kernel density estimation
#[derive(Debug, Clone, Copy)]
pub struct Density;

impl GeomTrait for Density {
    fn geom_type(&self) -> GeomType {
        GeomType::Density
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "fill", "stroke", "opacity"],
            required: &["x"],
            hidden: &[],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    // Note: stat_density not yet implemented - will return Identity for now
}

impl std::fmt::Display for Density {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "density")
    }
}
