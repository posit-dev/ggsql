//! Ordinal scale type implementation
//!
//! Ordinal scales handle ordered categorical data with continuous output interpolation.
//! Unlike discrete scales (exact 1:1 mapping), ordinal scales interpolate output values
//! to create smooth gradients for aesthetics like color, size, and opacity.

use std::collections::HashMap;

use polars::prelude::{Column, DataType};

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

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        // Ordinal scales accept categorical types plus numeric types
        // Numeric types are useful for ordered categories like Month (1-12), rank values, etc.
        matches!(
            dtype,
            DataType::String
                | DataType::Categorical(_, _)
                | DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        )
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        _properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        // Same as Discrete - compute unique values from columns
        let computed = compute_unique_values(columns);

        match user_range {
            None => Ok(computed),
            Some(range) if super::input_range_has_nulls(range) => match computed {
                Some(inferred) => Ok(Some(super::merge_with_inferred(range, &inferred))),
                None => Ok(Some(range.to_vec())),
            },
            Some(range) => Ok(Some(range.to_vec())),
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
        use super::super::colour::{interpolate_colors, ColorSpace};
        use super::super::{palettes, OutputRange};

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
                    let palette = match aesthetic {
                        "shape" => palettes::get_shape_palette(name),
                        "linetype" => palettes::get_linetype_palette(name),
                        _ => palettes::get_color_palette(name),
                    };
                    if let Some(palette) = palette {
                        let arr: Vec<_> = palette
                            .iter()
                            .map(|s| ArrayElement::String(s.to_string()))
                            .collect();
                        scale.output_range = Some(OutputRange::Array(arr));
                    }
                }
                Some(OutputRange::Array(_)) => {}
            }
        }

        // Phase 2: Interpolate to category count (like Binned, but using input_range.len())
        if let Some(OutputRange::Array(ref arr)) = scale.output_range.clone() {
            if matches!(aesthetic, "fill" | "stroke") && arr.len() >= 2 {
                // Color interpolation
                let hex_strs: Vec<&str> = arr
                    .iter()
                    .filter_map(|e| match e {
                        ArrayElement::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                let interpolated = interpolate_colors(&hex_strs, count, ColorSpace::Oklab)?;
                scale.output_range = Some(OutputRange::Array(
                    interpolated.into_iter().map(ArrayElement::String).collect(),
                ));
            } else if matches!(aesthetic, "size" | "linewidth" | "opacity") && arr.len() >= 2 {
                // Numeric interpolation
                let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
                if nums.len() >= 2 {
                    let min_val = nums[0];
                    let max_val = nums[nums.len() - 1];
                    let interpolated: Vec<ArrayElement> = (0..count)
                        .map(|i| {
                            let t = if count > 1 {
                                i as f64 / (count - 1) as f64
                            } else {
                                0.5
                            };
                            ArrayElement::Number(min_val + t * (max_val - min_val))
                        })
                        .collect();
                    scale.output_range = Some(OutputRange::Array(interpolated));
                }
            } else {
                // Non-interpolatable aesthetics (shape, linetype): truncate/error like Discrete
                if arr.len() < count {
                    return Err(format!(
                        "Output range has {} values but {} categories needed",
                        arr.len(),
                        count
                    ));
                }
                if arr.len() > count {
                    scale.output_range = Some(OutputRange::Array(
                        arr.iter().take(count).cloned().collect(),
                    ));
                }
            }
        }

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

/// Compute unique values from columns.
///
/// Preserves native types and sorts accordingly:
/// - Boolean columns → `ArrayElement::Boolean` values in logical order `[false, true]`
/// - Integer/Float columns → `ArrayElement::Number` values sorted numerically
/// - Date columns → `ArrayElement::Date` values sorted chronologically
/// - DateTime columns → `ArrayElement::DateTime` values sorted chronologically
/// - Time columns → `ArrayElement::Time` values sorted chronologically
/// - String/Categorical columns → `ArrayElement::String` values sorted alphabetically
///
/// Unlike discrete scales, ordinal scales do NOT include null values in the input range
/// (ordinal is for ordered data where null doesn't have a meaningful position).
fn compute_unique_values(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
    if column_refs.is_empty() {
        return None;
    }

    // Don't include nulls for ordinal scales
    let result = super::compute_unique_values_native(column_refs, false);

    if result.is_empty() {
        None
    } else {
        Some(result)
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
    fn test_ordinal_allows_numeric_data_types() {
        use super::super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let ordinal = Ordinal;

        // Ordinal should allow numeric types (e.g., Month 1-12)
        assert!(ordinal.allows_data_type(&DataType::Int8));
        assert!(ordinal.allows_data_type(&DataType::Int16));
        assert!(ordinal.allows_data_type(&DataType::Int32));
        assert!(ordinal.allows_data_type(&DataType::Int64));
        assert!(ordinal.allows_data_type(&DataType::UInt8));
        assert!(ordinal.allows_data_type(&DataType::UInt16));
        assert!(ordinal.allows_data_type(&DataType::UInt32));
        assert!(ordinal.allows_data_type(&DataType::UInt64));
        assert!(ordinal.allows_data_type(&DataType::Float32));
        assert!(ordinal.allows_data_type(&DataType::Float64));

        // Also allows categorical types
        assert!(ordinal.allows_data_type(&DataType::String));
        assert!(ordinal.allows_data_type(&DataType::Boolean));
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

    #[test]
    fn test_ordinal_numeric_input_range_sorted_numerically() {
        use polars::prelude::*;

        // Create a numeric column with values that would sort incorrectly as strings
        // String sort: ["1", "10", "2", "20"] vs numeric sort: [1, 2, 10, 20]
        let series = Series::new("month".into(), vec![10, 1, 20, 2]);
        let column = series.into_column();
        let columns = vec![&column];

        let result = compute_unique_values(&columns);
        assert!(result.is_some());
        let values = result.unwrap();

        // Should be sorted numerically: 1, 2, 10, 20
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], ArrayElement::Number(1.0));
        assert_eq!(values[1], ArrayElement::Number(2.0));
        assert_eq!(values[2], ArrayElement::Number(10.0));
        assert_eq!(values[3], ArrayElement::Number(20.0));
    }
}
