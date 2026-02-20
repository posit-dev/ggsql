//! Smooth geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;
use crate::Mappings;

/// Smooth geom - smoothed conditional means (regression, LOESS, etc.)
#[derive(Debug, Clone, Copy)]
pub struct Smooth;

impl GeomTrait for Smooth {
    fn geom_type(&self) -> GeomType {
        GeomType::Smooth
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("x", DefaultAestheticValue::Required),
                ("y", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("#3366FF")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    // Note: stat_smooth not yet implemented - will return Identity for now
}

impl std::fmt::Display for Smooth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "smooth")
    }
}
