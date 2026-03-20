//! Cartesian coordinate system implementation

use super::{CoordKind, CoordTrait};
use crate::plot::types::{ParamConstraint, ParamDefinition, DefaultParamValue};

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

    fn positional_aesthetic_names(&self) -> &'static [&'static str] {
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

impl std::fmt::Display for Cartesian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;
    use std::collections::HashMap;

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

    #[test]
    fn test_cartesian_resolve_valid_properties() {
        let cartesian = Cartesian;
        let props = HashMap::new();
        // Empty properties should resolve successfully
        let resolved = cartesian.resolve_properties(&props);
        assert!(resolved.is_ok());
    }

    #[test]
    fn test_cartesian_rejects_unknown_property() {
        let cartesian = Cartesian;
        let mut props = HashMap::new();
        props.insert(
            "unknown".to_string(),
            ParameterValue::String("value".to_string()),
        );

        let resolved = cartesian.resolve_properties(&props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("not 'unknown'"));
    }
}
