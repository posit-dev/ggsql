//! Path geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// Path geom - connected line segments in order
#[derive(Debug, Clone, Copy)]
pub struct Path;

impl GeomTrait for Path {
    fn geom_type(&self) -> GeomType {
        GeomType::Path
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &["x", "y", "stroke", "linetype", "linewidth", "opacity"],
            required: &["x", "y"],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "path")
    }
}
