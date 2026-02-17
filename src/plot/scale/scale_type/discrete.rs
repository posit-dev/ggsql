//! Discrete scale type implementation

use polars::prelude::DataType;

use super::super::transform::{Transform, TransformKind};
use super::{ScaleTypeKind, ScaleTypeTrait, SqlTypeNames};
use crate::plot::{ArrayElement, ParameterValue};

/// Discrete scale type - for categorical/discrete data
#[derive(Debug, Clone, Copy)]
pub struct Discrete;

impl ScaleTypeTrait for Discrete {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Discrete
    }

    fn name(&self) -> &'static str {
        "discrete"
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
            | DataType::Float64 => Err("Discrete scale cannot be used with numeric data. \
                 Use CONTINUOUS or BINNED scale type instead, or ensure the column contains categorical data.".to_string()),
            // Reject temporal types
            DataType::Date => Err("Discrete scale cannot be used with Date data. \
                 Use CONTINUOUS scale type instead (dates are treated as continuous temporal data).".to_string()),
            DataType::Datetime(_, _) => Err("Discrete scale cannot be used with DateTime data. \
                 Use CONTINUOUS scale type instead (datetimes are treated as continuous temporal data).".to_string()),
            DataType::Time => Err("Discrete scale cannot be used with Time data. \
                 Use CONTINUOUS scale type instead (times are treated as continuous temporal data).".to_string()),
            // Other types - provide generic message
            other => Err(format!(
                "Discrete scale cannot be used with {:?} data. \
                 Discrete scales require categorical data (String, Boolean, or Categorical).",
                other
            )),
        }
    }

    fn uses_discrete_input_range(&self) -> bool {
        true
    }

    fn allowed_properties(&self, _aesthetic: &str) -> &'static [&'static str] {
        // Discrete scales always censor OOB values (no OOB setting needed)
        &["reverse"]
    }

    fn get_property_default(&self, _aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            "reverse" => Some(ParameterValue::Boolean(false)),
            _ => None,
        }
    }

    fn allowed_transforms(&self) -> &'static [TransformKind] {
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
        // Infer transform from column dtype
        if let Some(dtype) = column_dtype {
            match dtype {
                DataType::Boolean => return TransformKind::Bool,
                DataType::String | DataType::Categorical(_, _) => return TransformKind::String,
                _ => {}
            }
        }
        // Default to Identity for unknown/no column info
        TransformKind::Identity
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
            if let Some(kind) = infer_transform_from_input_range(range) {
                return Ok(Transform::from_kind(kind));
            }
        }

        // Priority 2: Infer from column dtype
        Ok(Transform::from_kind(
            self.default_transform(aesthetic, column_dtype),
        ))
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _scale: &super::super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        // Return full palette - sizing is done in resolve_output_range()
        match aesthetic {
            // Note: "color"/"colour" already split to fill/stroke before scale resolution
            "fill" | "stroke" => {
                let palette = palettes::get_color_palette("ggsql")
                    .ok_or_else(|| "Default color palette 'ggsql' not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|s| ArrayElement::String(s.to_string()))
                        .collect(),
                ))
            }
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

        // Phase 1: Ensure we have an Array (convert Palette or fill default)
        match &scale.output_range {
            None => {
                // No output range - fill from default
                if let Some(default_range) = self.default_output_range(aesthetic, scale)? {
                    scale.output_range = Some(OutputRange::Array(default_range));
                }
            }
            Some(OutputRange::Palette(name)) => {
                // Named palette - convert to Array
                let arr = palettes::lookup_palette(aesthetic, name)?;
                scale.output_range = Some(OutputRange::Array(arr));
            }
            Some(OutputRange::Array(_)) => {
                // Already an array, nothing to do
            }
        }

        // Phase 2: Size the Array to match category count
        // Discrete scales don't interpolate - just truncate or error
        let count = scale.input_range.as_ref().map(|r| r.len()).unwrap_or(0);
        if count == 0 {
            return Ok(());
        }

        if let Some(OutputRange::Array(ref arr)) = scale.output_range.clone() {
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

        Ok(())
    }

    /// Pre-stat SQL transformation for discrete scales.
    ///
    /// Discrete scales always censor values outside the explicit input range
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

        // Always censor - discrete scales have no other valid OOB behavior
        Some(format!(
            "(CASE WHEN {} IN ({}) THEN {} ELSE NULL END)",
            column_name,
            allowed_values.join(", "),
            column_name
        ))
    }
}

/// Infer a transform kind from input range values.
///
/// If the input range contains values of a specific type, infer the corresponding transform:
/// - String values → String transform
/// - Boolean values → Bool transform
/// - Other/mixed → None (use default)
pub fn infer_transform_from_input_range(range: &[ArrayElement]) -> Option<TransformKind> {
    if range.is_empty() {
        return None;
    }

    // Check first element to determine type
    match &range[0] {
        ArrayElement::String(_) => Some(TransformKind::String),
        ArrayElement::Boolean(_) => Some(TransformKind::Bool),
        _ => None,
    }
}

impl std::fmt::Display for Discrete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_allowed_transforms() {
        let discrete = Discrete;
        let allowed = discrete.allowed_transforms();
        assert!(allowed.contains(&TransformKind::Identity));
        assert!(allowed.contains(&TransformKind::String));
        assert!(allowed.contains(&TransformKind::Bool));
        assert!(!allowed.contains(&TransformKind::Log10));
    }

    #[test]
    fn test_discrete_default_transform_from_dtype() {
        let discrete = Discrete;

        // Boolean column → Bool transform
        assert_eq!(
            discrete.default_transform("color", Some(&DataType::Boolean)),
            TransformKind::Bool
        );

        // String column → String transform
        assert_eq!(
            discrete.default_transform("color", Some(&DataType::String)),
            TransformKind::String
        );

        // No column info → Identity
        assert_eq!(
            discrete.default_transform("color", None),
            TransformKind::Identity
        );
    }

    #[test]
    fn test_infer_transform_from_input_range_string() {
        let range = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ];
        assert_eq!(
            infer_transform_from_input_range(&range),
            Some(TransformKind::String)
        );
    }

    #[test]
    fn test_infer_transform_from_input_range_boolean() {
        let range = vec![ArrayElement::Boolean(false), ArrayElement::Boolean(true)];
        assert_eq!(
            infer_transform_from_input_range(&range),
            Some(TransformKind::Bool)
        );
    }

    #[test]
    fn test_infer_transform_from_input_range_empty() {
        let range: Vec<ArrayElement> = vec![];
        assert_eq!(infer_transform_from_input_range(&range), None);
    }

    #[test]
    fn test_infer_transform_from_input_range_numeric() {
        // Numeric values don't map to discrete transforms
        let range = vec![ArrayElement::Number(1.0), ArrayElement::Number(2.0)];
        assert_eq!(infer_transform_from_input_range(&range), None);
    }

    #[test]
    fn test_resolve_transform_explicit_string() {
        let discrete = Discrete;
        let string_transform = Transform::string();

        let result = discrete.resolve_transform("color", Some(&string_transform), None, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::String);
    }

    #[test]
    fn test_resolve_transform_explicit_bool() {
        let discrete = Discrete;
        let bool_transform = Transform::bool();

        let result = discrete.resolve_transform("color", Some(&bool_transform), None, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);
    }

    #[test]
    fn test_resolve_transform_input_range_priority_over_dtype() {
        let discrete = Discrete;

        // Bool input range should take priority over String column dtype
        let bool_range = vec![ArrayElement::Boolean(true), ArrayElement::Boolean(false)];
        let result = discrete.resolve_transform(
            "color",
            None,
            Some(&DataType::String), // String column
            Some(&bool_range),       // But bool input range
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);

        // String input range should take priority over Boolean column dtype
        let string_range = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ];
        let result = discrete.resolve_transform(
            "color",
            None,
            Some(&DataType::Boolean), // Boolean column
            Some(&string_range),      // But string input range
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::String);
    }

    #[test]
    fn test_resolve_transform_falls_back_to_dtype_when_no_input_range() {
        let discrete = Discrete;

        // No input range - should infer from column dtype
        let result = discrete.resolve_transform("color", None, Some(&DataType::Boolean), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);

        let result = discrete.resolve_transform("color", None, Some(&DataType::String), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().transform_kind(), TransformKind::String);
    }

    #[test]
    fn test_resolve_transform_numeric_input_range_falls_back_to_dtype() {
        let discrete = Discrete;

        // Numeric input range doesn't map to a discrete transform, so falls back to dtype
        let numeric_range = vec![ArrayElement::Number(1.0), ArrayElement::Number(2.0)];
        let result = discrete.resolve_transform(
            "color",
            None,
            Some(&DataType::Boolean),
            Some(&numeric_range),
        );
        assert!(result.is_ok());
        // Falls back to Boolean dtype inference
        assert_eq!(result.unwrap().transform_kind(), TransformKind::Bool);
    }

    #[test]
    fn test_resolve_transform_disallowed() {
        let discrete = Discrete;
        let log_transform = Transform::log();

        let result = discrete.resolve_transform("color", Some(&log_transform), None, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("not supported for discrete scale"));
    }

    // =========================================================================
    // Pre-Stat Transform SQL Tests
    // =========================================================================

    #[test]
    fn test_pre_stat_transform_sql_with_explicit_input_range() {
        use crate::plot::scale::Scale;

        let discrete = Discrete;
        let mut scale = Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ]);
        scale.explicit_input_range = true;

        let type_names = super::SqlTypeNames::default();
        let sql =
            discrete.pre_stat_transform_sql("category", &DataType::String, &scale, &type_names);

        assert!(sql.is_some());
        let sql = sql.unwrap();
        // Should generate CASE WHEN with IN clause
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("IN ('A', 'B')"));
        assert!(sql.contains("ELSE NULL"));
    }

    #[test]
    fn test_pre_stat_transform_sql_no_explicit_range() {
        use crate::plot::scale::Scale;

        let discrete = Discrete;
        let mut scale = Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ]);
        // explicit_input_range = false (inferred from data)
        scale.explicit_input_range = false;

        let type_names = super::SqlTypeNames::default();
        let sql =
            discrete.pre_stat_transform_sql("category", &DataType::String, &scale, &type_names);

        // Should return None (no OOB handling for inferred ranges)
        assert!(sql.is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_boolean_input_range() {
        use crate::plot::scale::Scale;

        let discrete = Discrete;
        let mut scale = Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::Boolean(true),
            ArrayElement::Boolean(false),
        ]);
        scale.explicit_input_range = true;

        let type_names = super::SqlTypeNames::default();
        let sql = discrete.pre_stat_transform_sql("flag", &DataType::Boolean, &scale, &type_names);

        assert!(sql.is_some());
        let sql = sql.unwrap();
        // Should generate CASE WHEN with IN clause for booleans
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("IN (true, false)"));
    }

    #[test]
    fn test_pre_stat_transform_sql_escapes_quotes() {
        use crate::plot::scale::Scale;

        let discrete = Discrete;
        let mut scale = Scale::new("color");
        scale.input_range = Some(vec![
            ArrayElement::String("it's".to_string()),
            ArrayElement::String("fine".to_string()),
        ]);
        scale.explicit_input_range = true;

        let type_names = super::SqlTypeNames::default();
        let sql = discrete.pre_stat_transform_sql("text", &DataType::String, &scale, &type_names);

        assert!(sql.is_some());
        let sql = sql.unwrap();
        // Should escape single quotes
        assert!(sql.contains("'it''s'"));
    }

    #[test]
    fn test_pre_stat_transform_sql_empty_range() {
        use crate::plot::scale::Scale;

        let discrete = Discrete;
        let mut scale = Scale::new("color");
        scale.input_range = Some(vec![]);
        scale.explicit_input_range = true;

        let type_names = super::SqlTypeNames::default();
        let sql =
            discrete.pre_stat_transform_sql("category", &DataType::String, &scale, &type_names);

        // Should return None for empty range
        assert!(sql.is_none());
    }

    // =========================================================================
    // Dtype Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dtype_accepts_string() {
        use super::ScaleTypeTrait;

        let discrete = Discrete;
        assert!(discrete.validate_dtype(&DataType::String).is_ok());
    }

    #[test]
    fn test_validate_dtype_accepts_boolean() {
        use super::ScaleTypeTrait;

        let discrete = Discrete;
        assert!(discrete.validate_dtype(&DataType::Boolean).is_ok());
    }

    #[test]
    fn test_validate_dtype_rejects_numeric() {
        use super::ScaleTypeTrait;

        let discrete = Discrete;
        let result = discrete.validate_dtype(&DataType::Int64);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("numeric"));
        assert!(err.contains("CONTINUOUS") || err.contains("BINNED"));

        let result = discrete.validate_dtype(&DataType::Float64);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dtype_rejects_temporal() {
        use super::ScaleTypeTrait;
        use polars::prelude::TimeUnit;

        let discrete = Discrete;
        let result = discrete.validate_dtype(&DataType::Date);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Date"));
        assert!(err.contains("CONTINUOUS"));

        let result = discrete.validate_dtype(&DataType::Datetime(TimeUnit::Microseconds, None));
        assert!(result.is_err());

        let result = discrete.validate_dtype(&DataType::Time);
        assert!(result.is_err());
    }
}
