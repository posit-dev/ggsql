//! Polar coordinate system implementation

use super::{CoordKind, CoordTrait};
use crate::plot::ParameterValue;

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

    fn positional_aesthetic_names(&self) -> &'static [&'static str] {
        &["theta", "radius"]
    }

    fn allowed_properties(&self) -> &'static [&'static str] {
        &["clip", "start", "end"]
    }

    fn get_property_default(&self, name: &str) -> Option<ParameterValue> {
        match name {
            "start" => Some(ParameterValue::Number(0.0)), // 0 degrees = 12 o'clock
            _ => None,
        }
    }
}

impl std::fmt::Display for Polar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_polar_properties() {
        let polar = Polar;
        assert_eq!(polar.coord_kind(), CoordKind::Polar);
        assert_eq!(polar.name(), "polar");
    }

    #[test]
    fn test_polar_allowed_properties() {
        let polar = Polar;
        let allowed = polar.allowed_properties();
        assert!(allowed.contains(&"clip"));
        assert!(allowed.contains(&"start"));
        assert!(allowed.contains(&"end"));
        assert_eq!(allowed.len(), 3);
    }

    #[test]
    fn test_polar_start_default() {
        let polar = Polar;
        let default = polar.get_property_default("start");
        assert!(default.is_some());
        assert_eq!(default.unwrap(), ParameterValue::Number(0.0));
    }

    #[test]
    fn test_polar_rejects_unknown_property() {
        let polar = Polar;
        let mut props = HashMap::new();
        props.insert("unknown".to_string(), ParameterValue::String("value".to_string()));

        let resolved = polar.resolve_properties(&props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("unknown"));
        assert!(err.contains("not valid"));
    }

    #[test]
    fn test_polar_resolve_with_explicit_start() {
        let polar = Polar;
        let mut props = HashMap::new();
        props.insert("start".to_string(), ParameterValue::Number(90.0));

        let resolved = polar.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("start").unwrap(),
            &ParameterValue::Number(90.0)
        );
    }

    #[test]
    fn test_polar_resolve_adds_start_default() {
        let polar = Polar;
        let props = HashMap::new();

        let resolved = polar.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert!(resolved.contains_key("start"));
        assert_eq!(
            resolved.get("start").unwrap(),
            &ParameterValue::Number(0.0)
        );
    }

    #[test]
    fn test_polar_resolve_with_explicit_end() {
        let polar = Polar;
        let mut props = HashMap::new();
        props.insert("end".to_string(), ParameterValue::Number(180.0));

        let resolved = polar.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("end").unwrap(),
            &ParameterValue::Number(180.0)
        );
        // start should still get its default
        assert_eq!(
            resolved.get("start").unwrap(),
            &ParameterValue::Number(0.0)
        );
    }

    #[test]
    fn test_polar_resolve_with_start_and_end() {
        let polar = Polar;
        let mut props = HashMap::new();
        props.insert("start".to_string(), ParameterValue::Number(-90.0));
        props.insert("end".to_string(), ParameterValue::Number(90.0));

        let resolved = polar.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("start").unwrap(),
            &ParameterValue::Number(-90.0)
        );
        assert_eq!(
            resolved.get("end").unwrap(),
            &ParameterValue::Number(90.0)
        );
    }
}
