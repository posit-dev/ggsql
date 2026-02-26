//! Cartesian coordinate system implementation

use super::{CoordKind, CoordTrait};

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

    fn allowed_properties(&self) -> &'static [&'static str] {
        &["ratio", "clip"]
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
    fn test_cartesian_allowed_properties() {
        let cartesian = Cartesian;
        let allowed = cartesian.allowed_properties();
        assert!(allowed.contains(&"ratio"));
        assert!(allowed.contains(&"clip"));
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
        assert!(err.contains("unknown"));
        assert!(err.contains("not valid"));
    }
}
