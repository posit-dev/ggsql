//! Ordinal scale type implementation
//!
//! Ordinal scales handle ordered categorical data with continuous output interpolation.
//! Unlike discrete scales (exact 1:1 mapping), ordinal scales interpolate output values
//! to create smooth gradients for aesthetics like color, size, and opacity.

use polars::prelude::DataType;

use super::super::transform::{Transform, TransformKind};
use super::{ScaleTypeKind, ScaleTypeTrait, SqlTypeNames};
use crate::plot::{ArrayElement, ParameterValue};

/// Ordinal scale type - for ordered categorical data with interpolated output
#[derive(Debug, Clone, Copy)]
pub struct Ordinal;

impl ScaleTypeTrait for Ordinal {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Ordinal
    }

    fn name(&self) -> &'static str {
        "ordinal"
    }

    fn validate_dtype(&self, dtype: &DataType) -> Result<(), String> {
        match dtype {
            // Accept discrete types
            DataType::String | DataType::Boolean | DataType::Categorical(_, _) => Ok(()),
            // Reject numeric types
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Err("Ordinal scale cannot be used with numeric data. \
                 Use CONTINUOUS or BINNED scale type instead, or ensure the column contains categorical data.".to_string()),
            // Reject temporal types
            DataType::Date => Err("Ordinal scale cannot be used with Date data. \
                 Use CONTINUOUS scale type instead (dates are treated as continuous temporal data).".to_string()),
            DataType::Datetime(_, _) => Err("Ordinal scale cannot be used with DateTime data. \
                 Use CONTINUOUS scale type instead (datetimes are treated as continuous temporal data).".to_string()),
            DataType::Time => Err("Ordinal scale cannot be used with Time data. \
                 Use CONTINUOUS scale type instead (times are treated as continuous temporal data).".to_string()),
            // Other types - provide generic message
            other => Err(format!(
                "Ordinal scale cannot be used with {:?} data. \
                 Ordinal scales require categorical data (String, Boolean, or Categorical).",
                other
            )),
        }
    }

    fn uses_discrete_input_range(&self) -> bool {
        true // Collects unique values like Discrete
    }

    fn allowed_transforms(&self) -> &'static [TransformKind] {
        // Same as Discrete - categorical transforms only
        &[
            TransformKind::Identity,
            TransformKind::String,
            TransformKind::Bool,
        ]
    }

    fn default_transform(
        &self,
        _aesthetic: &str,
        column_dtype: Option<&DataType>,
    ) -> TransformKind {
        // Infer from column type
        match column_dtype {
            Some(DataType::Boolean) => TransformKind::Bool,
            Some(DataType::String) | Some(DataType::Categorical(_, _)) => TransformKind::String,
            // Numeric types use Identity to preserve numeric sorting
            Some(
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64,
            ) => TransformKind::Identity,
            // Default to String for unknown types
            _ => TransformKind::String,
        }
    }

    fn resolve_transform(
        &self,
        aesthetic: &str,
        user_transform: Option<&Transform>,
        column_dtype: Option<&DataType>,
        input_range: Option<&[ArrayElement]>,
    ) -> Result<Transform, String> {
        // If user specified a transform, validate and use it
        if let Some(t) = user_transform {
            if self.allowed_transforms().contains(&t.transform_kind()) {
                return Ok(t.clone());
            } else {
                return Err(format!(
                    "Transform '{}' not supported for {} scale. Allowed: {}",
                    t.name(),
                    self.name(),
                    self.allowed_transforms()
                        .iter()
                        .map(|k| k.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        // Priority 1: Infer from input range (FROM clause) if provided
        if let Some(range) = input_range {
            if let Some(kind) = super::discrete::infer_transform_from_input_range(range) {
                return Ok(Transform::from_kind(kind));
            }
        }

        // Priority 2: Infer from column dtype
        Ok(Transform::from_kind(
            self.default_transform(aesthetic, column_dtype),
        ))
    }

    fn allowed_properties(&self, _aesthetic: &str) -> &'static [&'static str] {
        // Ordinal scales always censor OOB values (no OOB setting needed)
        &["reverse"]
    }

    fn get_property_default(&self, _aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            "reverse" => Some(ParameterValue::Boolean(false)),
            _ => None,
        }
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _scale: &super::super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        // Colors use "sequential" (like Continuous) since ordinal has inherent ordering
        // Other aesthetics same as Discrete
        match aesthetic {
            "stroke" | "fill" => {
                let palette = palettes::get_color_palette("sequential")
                    .ok_or_else(|| "Default color palette 'sequential' not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|s| ArrayElement::String(s.to_string()))
                        .collect(),
                ))
            }
            "size" | "linewidth" => Ok(Some(vec![
                ArrayElement::Number(1.0),
                ArrayElement::Number(6.0),
            ])),
            "opacity" => Ok(Some(vec![
                ArrayElement::Number(0.1),
                ArrayElement::Number(1.0),
            ])),
            "shape" => {
                let palette = palettes::get_shape_palette("default")
                    .ok_or_else(|| "Default shape palette not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|s| ArrayElement::String(s.to_string()))
                        .collect(),
                ))
            }
            "linetype" => {
                let palette = palettes::get_linetype_palette("default")
                    .ok_or_else(|| "Default linetype palette not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|s| ArrayElement::String(s.to_string()))
                        .collect(),
                ))
            }
            _ => Ok(None),
        }
    }

    fn resolve_output_range(
        &self,
        scale: &mut super::super::Scale,
        aesthetic: &str,
    ) -> Result<(), String> {
        use super::super::{palettes, OutputRange};
        use super::size_output_range;

        // Get category count from input_range (key difference from Binned which uses breaks)
        let count = scale.input_range.as_ref().map(|r| r.len()).unwrap_or(0);
        if count == 0 {
            return Ok(());
        }

        // Phase 1: Ensure we have an Array (convert Palette or fill default)
        // For linetype, use sequential ink-density palette as default (None or "sequential")
        let use_sequential_linetype = aesthetic == "linetype"
            && match &scale.output_range {
                None => true,
                Some(OutputRange::Palette(name)) => name.eq_ignore_ascii_case("sequential"),
                _ => false,
            };

        if use_sequential_linetype {
            // Generate sequential ink-density palette sized to category count
            let sequential = palettes::generate_linetype_sequential(count);
            scale.output_range = Some(OutputRange::Array(
                sequential.into_iter().map(ArrayElement::String).collect(),
            ));
        } else {
            match &scale.output_range {
                None => {
                    if let Some(default_range) = self.default_output_range(aesthetic, scale)? {
                        scale.output_range = Some(OutputRange::Array(default_range));
                    }
                }
                Some(OutputRange::Palette(name)) => {
                    let arr = palettes::lookup_palette(aesthetic, name)?;
                    scale.output_range = Some(OutputRange::Array(arr));
                }
                Some(OutputRange::Array(_)) => {}
            }
        }

        // Phase 2: Size/interpolate to category count
        size_output_range(scale, aesthetic, count)?;

        Ok(())
    }

    fn supports_breaks(&self) -> bool {
        false // No breaks for ordinal (unlike binned)
    }

    /// Pre-stat SQL transformation for ordinal scales.
    ///
    /// Ordinal scales always censor values outside the explicit input range
    /// (values not in the FROM clause have no output mapping).
    ///
    /// Only applies when input_range is explicitly specified via FROM clause.
    /// Returns CASE WHEN col IN (allowed_values) THEN col ELSE NULL END.
    fn pre_stat_transform_sql(
        &self,
        column_name: &str,
        _column_dtype: &DataType,
        scale: &super::super::Scale,
        _type_names: &SqlTypeNames,
    ) -> Option<String> {
        // Only apply if input_range is explicitly specified by user
        // (not inferred from data)
        if !scale.explicit_input_range {
            return None;
        }

        let input_range = scale.input_range.as_ref()?;
        if input_range.is_empty() {
            return None;
        }

        // Build IN clause values (excluding null - SQL IN doesn't match NULL)
        let allowed_values: Vec<String> = input_range
            .iter()
            .filter_map(|e| match e {
                ArrayElement::String(s) => Some(format!("'{}'", s.replace('\'', "''"))),
                ArrayElement::Boolean(b) => Some(if *b { "true".into() } else { "false".into() }),
                _ => None,
            })
            .collect();

        if allowed_values.is_empty() {
            return None;
        }

        // Always censor - ordinal scales have no other valid OOB behavior
        Some(format!(
            "(CASE WHEN {} IN ({}) THEN {} ELSE NULL END)",
            column_name,
            allowed_values.join(", "),
            column_name
        ))
    }
}

impl std::fmt::Display for Ordinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scale::{OutputRange, Scale};

    #[test]
    fn test_ordinal_scale_type_kind() {
        let ordinal = Ordinal;
        assert_eq!(ordinal.scale_type_kind(), ScaleTypeKind::Ordinal);
        assert_eq!(ordinal.name(), "ordinal");
    }

    #[test]
    fn test_ordinal_uses_discrete_input_range() {
        let ordinal = Ordinal;
        assert!(ordinal.uses_discrete_input_range());
    }

    #[test]
    fn test_ordinal_allowed_transforms() {
        let ordinal = Ordinal;
        let allowed = ordinal.allowed_transforms();
        assert!(allowed.contains(&TransformKind::Identity));
        assert!(allowed.contains(&TransformKind::String));
        assert!(allowed.contains(&TransformKind::Bool));
        assert!(!allowed.contains(&TransformKind::Log10));
    }

    #[test]
    fn test_resolve_output_range_color_interpolation() {
        use super::super::ScaleTypeTrait;

        let ordinal = Ordinal;
        let mut scale = Scale::new("fill");

        // 3 categories
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
        ]);

        // 2 colors to interpolate from
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("#ff0000".to_string()),
            ArrayElement::String("#0000ff".to_string()),
        ]));

        ordinal.resolve_output_range(&mut scale, "fill").unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(
                arr.len(),
                3,
                "Should interpolate to 3 colors for 3 categories"
            );
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_size_interpolation() {
        use super::super::ScaleTypeTrait;

        let ordinal = Ordinal;
        let mut scale = Scale::new("size");

        // 5 categories
        scale.input_range = Some(vec![
            ArrayElement::String("XS".to_string()),
            ArrayElement::String("S".to_string()),
            ArrayElement::String("M".to_string()),
            ArrayElement::String("L".to_string()),
            ArrayElement::String("XL".to_string()),
        ]);

        // Size range [1, 6]
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(6.0),
        ]));

        ordinal.resolve_output_range(&mut scale, "size").unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(
                arr.len(),
                5,
                "Should interpolate to 5 sizes for 5 categories"
            );
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            assert!((nums[0] - 1.0).abs() < 0.001);
            assert!((nums[4] - 6.0).abs() < 0.001);
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_shape_truncates() {
        use super::super::ScaleTypeTrait;

        let ordinal = Ordinal;
        let mut scale = Scale::new("shape");

        // 2 categories
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ]);

        // 5 shapes (more than needed)
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("circle".to_string()),
            ArrayElement::String("square".to_string()),
            ArrayElement::String("triangle".to_string()),
            ArrayElement::String("cross".to_string()),
            ArrayElement::String("diamond".to_string()),
        ]));

        ordinal.resolve_output_range(&mut scale, "shape").unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 2, "Should truncate to 2 shapes for 2 categories");
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_shape_error_insufficient() {
        use super::super::ScaleTypeTrait;

        let ordinal = Ordinal;
        let mut scale = Scale::new("shape");

        // 5 categories
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
            ArrayElement::String("D".to_string()),
            ArrayElement::String("E".to_string()),
        ]);

        // Only 2 shapes (not enough)
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::String("circle".to_string()),
            ArrayElement::String("square".to_string()),
        ]));

        let result = ordinal.resolve_output_range(&mut scale, "shape");
        assert!(result.is_err(), "Should error when shapes are insufficient");
    }

    #[test]
    fn test_resolve_output_range_opacity_interpolation() {
        use super::super::ScaleTypeTrait;

        let ordinal = Ordinal;
        let mut scale = Scale::new("opacity");

        // 4 categories
        scale.input_range = Some(vec![
            ArrayElement::String("low".to_string()),
            ArrayElement::String("medium".to_string()),
            ArrayElement::String("high".to_string()),
            ArrayElement::String("very_high".to_string()),
        ]);

        // Opacity range [0.2, 1.0]
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(0.2),
            ArrayElement::Number(1.0),
        ]));

        ordinal.resolve_output_range(&mut scale, "opacity").unwrap();

        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(
                arr.len(),
                4,
                "Should interpolate to 4 opacity values for 4 categories"
            );
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            assert!((nums[0] - 0.2).abs() < 0.001);
            assert!((nums[3] - 1.0).abs() < 0.001);
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_ordinal_default_transform_numeric() {
        use super::super::ScaleTypeTrait;
        use crate::plot::scale::TransformKind;
        use polars::prelude::DataType;

        let ordinal = Ordinal;

        // Numeric types should use Identity transform (to preserve numeric sorting)
        assert_eq!(
            ordinal.default_transform("color", Some(&DataType::Int32)),
            TransformKind::Identity
        );
        assert_eq!(
            ordinal.default_transform("color", Some(&DataType::Int64)),
            TransformKind::Identity
        );
        assert_eq!(
            ordinal.default_transform("color", Some(&DataType::Float64)),
            TransformKind::Identity
        );

        // String/Boolean use their respective transforms
        assert_eq!(
            ordinal.default_transform("color", Some(&DataType::String)),
            TransformKind::String
        );
        assert_eq!(
            ordinal.default_transform("color", Some(&DataType::Boolean)),
            TransformKind::Bool
        );
    }

    // =========================================================================
    // Dtype Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dtype_accepts_string() {
        use super::super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let ordinal = Ordinal;
        assert!(ordinal.validate_dtype(&DataType::String).is_ok());
    }

    #[test]
    fn test_validate_dtype_accepts_boolean() {
        use super::super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let ordinal = Ordinal;
        assert!(ordinal.validate_dtype(&DataType::Boolean).is_ok());
    }

    #[test]
    fn test_validate_dtype_rejects_numeric() {
        use super::super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let ordinal = Ordinal;
        let result = ordinal.validate_dtype(&DataType::Int64);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("numeric"));
        assert!(err.contains("CONTINUOUS") || err.contains("BINNED"));

        let result = ordinal.validate_dtype(&DataType::Float64);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dtype_rejects_temporal() {
        use super::super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let ordinal = Ordinal;
        let result = ordinal.validate_dtype(&DataType::Date);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Date"));
        assert!(err.contains("CONTINUOUS"));
    }
}
