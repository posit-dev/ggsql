//! Rule geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;

/// Rule geom - horizontal and vertical reference lines
#[derive(Debug, Clone, Copy)]
pub struct Rule;

impl GeomTrait for Rule {
    fn geom_type(&self) -> GeomType {
        GeomType::Rule
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("slope", DefaultAestheticValue::Number(0.0)),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn validate_aesthetics(&self, mappings: &crate::Mappings) -> std::result::Result<(), String> {
        // Rule requires exactly one of pos1 or pos2 (XOR logic)
        let has_pos1 = mappings.contains_key("pos1");
        let has_pos2 = mappings.contains_key("pos2");

        if has_pos1 && has_pos2 {
            return Err("Layer 'rule' requires exactly one of `x` or `y`, not both.".to_string());
        }

        Ok(())
    }

    fn setup_layer(
        &self,
        mappings: &mut crate::plot::layer::Mappings,
        parameters: &mut std::collections::HashMap<String, crate::plot::ParameterValue>,
    ) -> crate::Result<()> {
        use crate::plot::layer::AestheticValue;
        use crate::plot::ParameterValue;

        // For diagonal rules (slope present), convert position aesthetics to AnnotationColumn
        // so they don't participate in scale training. The position value is the intercept,
        // not the actual extent of the line.

        // Check if slope is present and non-zero (in either mappings or parameters)
        let has_diagonal_slope = mappings.get("slope").is_some_and(|mapping| {
            !matches!(mapping, AestheticValue::Literal(ParameterValue::Number(n)) if *n == 0.0)
        }) || parameters.get("slope").is_some_and(|param| {
            !matches!(param, ParameterValue::Number(n) if *n == 0.0)
        });

        if !has_diagonal_slope {
            return Ok(());
        }
        parameters.insert("diagonal".to_string(), ParameterValue::Boolean(true));

        // Determine orientation from which intercept is present.
        // We override the bidirectionality algorithm here since it uses
        // scales to determine orientation. We can't rely on that here,
        // because diagonal lines purposefully use AnnotationColumns to
        // avoid training the scale.
        let orientation = if mappings.contains_key("pos1") {
            // x-intercept → pos2 varies
            crate::plot::layer::orientation::TRANSPOSED
        } else {
            // y-intercept → pos1 varies (or default)
            crate::plot::layer::orientation::ALIGNED
        };

        parameters.insert(
            "orientation".to_string(),
            ParameterValue::String(orientation.to_string()),
        );

        // For diagonal rules, convert pos1/pos2 to AnnotationColumn so they don't participate in scale training
        // The position value is the intercept, not the actual extent of the line
        for aesthetic in ["pos1", "pos2"] {
            if let Some(mapping) = mappings.aesthetics.get_mut(aesthetic) {
                // Convert Column to AnnotationColumn
                if let AestheticValue::Column { name, .. } = &*mapping {
                    let name = name.clone();
                    *mapping = AestheticValue::AnnotationColumn { name };
                }
            }
        }

        Ok(())
    }
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rule")
    }
}

#[cfg(test)]
mod tests {
    use crate::plot::{AestheticContext, AestheticValue, Geom, Layer};

    /// Helper function to create a layer with given mappings and validate it
    fn validate_rule(mappings: &[(&str, &str)]) -> Result<(), String> {
        let mut layer = Layer::new(Geom::rule());
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
    fn test_rule_requires_exactly_one_position() {
        // Rule requires exactly one of pos1 or pos2 (XOR logic)

        // Missing both should fail
        let result = validate_rule(&[]);
        assert!(result.is_err(), "Should fail when missing both x and y");

        // Both present should fail
        let result = validate_rule(&[("pos1", "x"), ("pos2", "y")]);
        assert!(result.is_err(), "Should fail when both x and y are present");
    }

    #[test]
    fn test_rule_validates_with_x_only() {
        // Vertical rule with only x
        let result = validate_rule(&[("pos1", "x")]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with only x, got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rule_validates_with_y_only() {
        // Horizontal rule with only y
        let result = validate_rule(&[("pos2", "y")]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with only y, got error: {:?}",
            result.err()
        );
    }
}
