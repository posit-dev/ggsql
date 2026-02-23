//! Flip coordinate system implementation

use super::{CoordKind, CoordTrait};

/// Flip coordinate system - swaps x and y axes
#[derive(Debug, Clone, Copy)]
pub struct Flip;

impl CoordTrait for Flip {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Flip
    }

    fn name(&self) -> &'static str {
        "flip"
    }

    // Flip has no SETTING properties
}

impl std::fmt::Display for Flip {
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
    fn test_flip_properties() {
        let flip = Flip;
        assert_eq!(flip.coord_kind(), CoordKind::Flip);
        assert_eq!(flip.name(), "flip");
    }

    #[test]
    fn test_flip_no_properties() {
        let flip = Flip;
        assert!(flip.allowed_properties().is_empty());
    }

    #[test]
    fn test_flip_rejects_any_property() {
        let flip = Flip;
        let mut props = HashMap::new();
        props.insert("theta".to_string(), ParameterValue::String("y".to_string()));

        let resolved = flip.resolve_properties(&props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("theta"));
        assert!(err.contains("not valid"));
    }
}
