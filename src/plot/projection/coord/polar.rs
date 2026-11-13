//! Polar coordinate system implementation

use super::{CoordKind, CoordTrait};
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};

/// Polar coordinate system - for pie charts, rose plots
#[derive(Debug, Clone, Copy)]
pub struct Polar;

impl CoordTrait for Polar {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Polar
    }

    fn name(&self) -> &'static str {
        "polar"
    }

    fn position_aesthetic_names(&self) -> &'static [&'static str] {
        &["radius", "angle"]
    }

    fn default_properties(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "clip",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
            ParamDefinition {
                name: "start",
                default: DefaultParamValue::Number(0.0), // 0 degrees = 12 o'clock
                constraint: ParamConstraint::number_range(-360.0, 360.0),
            },
            ParamDefinition {
                name: "end",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_range(-360.0, 360.0),
            },
            ParamDefinition {
                name: "inner",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_range(0.0, 1.0),
            },
            ParamDefinition {
                name: "radar",
                default: DefaultParamValue::Null,
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
    fn test_polar_properties() {
        let polar = Polar;
        assert_eq!(polar.coord_kind(), CoordKind::Polar);
        assert_eq!(polar.name(), "polar");
    }

    #[test]
    fn test_polar_default_properties() {
        let polar = Polar;
        let defaults = polar.default_properties();
        let names: Vec<&str> = defaults.iter().map(|p| p.name).collect();
        assert!(names.contains(&"clip"));
        assert!(names.contains(&"start"));
        assert!(names.contains(&"end"));
        assert!(names.contains(&"inner"));
        assert!(names.contains(&"radar"));
        assert_eq!(defaults.len(), 5);
    }

    #[test]
    fn test_polar_start_default() {
        let polar = Polar;
        let defaults = polar.default_properties();
        let start_param = defaults.iter().find(|p| p.name == "start").unwrap();
        assert!(matches!(
            start_param.default,
            DefaultParamValue::Number(0.0)
        ));
    }
}
