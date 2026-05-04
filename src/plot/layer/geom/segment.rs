//! Segment geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

/// Segment geom - line segments between two points
#[derive(Debug, Clone, Copy)]
pub struct Segment;

impl GeomTrait for Segment {
    fn geom_type(&self) -> GeomType {
        GeomType::Segment
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("pos1end", DefaultAestheticValue::Required),
                ("pos2end", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "position",
            default: DefaultParamValue::String("identity"),
            constraint: ParamConstraint::string_option(POSITION_VALUES),
        }];
        PARAMS
    }

    fn supports_aggregate(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "segment")
    }
}

#[cfg(test)]
mod tests {
    use crate::plot::{AestheticContext, AestheticValue, Geom, Layer};

    /// Helper function to create a layer with given mappings and validate it
    fn validate_segment(mappings: &[(&str, &str)]) -> Result<(), String> {
        let mut layer = Layer::new(Geom::segment());
        for (aesthetic, column) in mappings {
            layer.mappings.insert(
                aesthetic.to_string(),
                AestheticValue::standard_column(column.to_string()),
            );
        }
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        layer.validate_mapping(&Some(ctx), false)
    }

    #[test]
    fn test_segment_requires_both_endpoints() {
        let result = validate_segment(&[("pos1", "x"), ("pos2", "y")]);
        assert!(result.is_err(), "Should fail when missing both endpoints");

        let result = validate_segment(&[("pos1", "x"), ("pos2", "y"), ("pos1end", "xend")]);
        assert!(result.is_err(), "Should fail when missing pos2end");

        let result = validate_segment(&[("pos1", "x"), ("pos2", "y"), ("pos2end", "yend")]);
        assert!(result.is_err(), "Should fail when missing pos1end");
    }

    #[test]
    fn test_segment_validates_with_both_endpoints() {
        let result = validate_segment(&[
            ("pos1", "x"),
            ("pos2", "y"),
            ("pos1end", "xend"),
            ("pos2end", "yend"),
        ]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with both endpoints, got error: {:?}",
            result.err()
        );
    }
}
