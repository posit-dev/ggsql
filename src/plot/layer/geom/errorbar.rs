//! ErrorBar geom implementation

use super::{GeomAesthetics, GeomTrait, GeomType};

/// ErrorBar geom - error bars (confidence intervals)
#[derive(Debug, Clone, Copy)]
pub struct ErrorBar;

impl GeomTrait for ErrorBar {
    fn geom_type(&self) -> GeomType {
        GeomType::ErrorBar
    }

    fn aesthetics(&self) -> GeomAesthetics {
        GeomAesthetics {
            supported: &[
                "x",
                "y",
                "ymin",
                "ymax",
                "xmin",
                "xmax",
                "stroke",
                "linewidth",
                "opacity",
            ],
            required: &[],
            hidden: &[],
        }
    }
}

impl std::fmt::Display for ErrorBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "errorbar")
    }
}
