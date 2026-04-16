//! ErrorBar geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::types::DefaultAestheticValue;

/// ErrorBar geom - error bars (confidence intervals)
#[derive(Debug, Clone, Copy)]
pub struct ErrorBar;

impl GeomTrait for ErrorBar {
    fn geom_type(&self) -> GeomType {
        GeomType::ErrorBar
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2min", DefaultAestheticValue::Required),
                ("pos2max", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("identity"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            ParamDefinition {
                name: "width",
                default: DefaultParamValue::Number(10.0),
                constraint: ParamConstraint::number_min(0.0),
            },
        ];
        PARAMS
    }
}

impl std::fmt::Display for ErrorBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "errorbar")
    }
}

#[cfg(test)]
mod tests {
    use crate::plot::{AestheticContext, AestheticValue, Geom, Layer};

    /// Helper function to create a layer with given mappings and validate it
    fn validate_errorbar(mappings: &[(&str, &str)]) -> Result<(), String> {
        let mut layer = Layer::new(Geom::errorbar());
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
    fn test_errorbar_requires_all_aesthetics() {
        // ErrorBar requires pos1, pos2min, pos2max - test that missing any fails
        let result = validate_errorbar(&[("pos1", "x"), ("pos2max", "ymax")]);
        assert!(
            result.is_err(),
            "Should fail when missing pos2min"
        );
    }

    #[test]
    fn test_errorbar_rejects_mixed_orientation() {
        // Mixed orientation should fail: pos1 (identity) + pos1max (flipped)
        let result = validate_errorbar(&[
            ("pos1", "x"),
            ("pos2min", "ymin"),
            ("pos1max", "xmax"),
        ]);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("mixed") || err.contains("orientation"),
            "Expected error about mixed orientation, got: {}",
            err
        );
    }

    #[test]
    fn test_errorbar_validates_successfully() {
        // ErrorBar with all required aesthetics should pass validation
        let result = validate_errorbar(&[
            ("pos1", "x"),
            ("pos2min", "ymin"),
            ("pos2max", "ymax"),
        ]);

        assert!(
            result.is_ok(),
            "Expected validation to pass with all required aesthetics, got error: {:?}",
            result.err()
        );
    }
}
