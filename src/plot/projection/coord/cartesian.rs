//! Cartesian coordinate system implementation

use super::{CoordKind, CoordTrait};
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};

/// Cartesian coordinate system - standard x/y coordinates
#[derive(Debug, Clone, Copy)]
pub struct Cartesian;

impl CoordTrait for Cartesian {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Cartesian
    }

    fn name(&self) -> &'static str {
        "cartesian"
    }

    fn position_aesthetic_names(&self) -> &'static [&'static str] {
        &["x", "y"]
    }

    fn default_properties(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "ratio",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "clip",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
        ];
        PARAMS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_properties() {
        let cartesian = Cartesian;
        assert_eq!(cartesian.coord_kind(), CoordKind::Cartesian);
        assert_eq!(cartesian.name(), "cartesian");
    }

    #[test]
    fn test_cartesian_default_properties() {
        let cartesian = Cartesian;
        let defaults = cartesian.default_properties();
        let names: Vec<&str> = defaults.iter().map(|p| p.name).collect();
        assert!(names.contains(&"ratio"));
        assert!(names.contains(&"clip"));
    }
}
