//! Discrete scale type implementation

use std::collections::HashMap;

use polars::prelude::{Column, DataType};

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

    fn is_discrete(&self) -> bool {
        false
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

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        // Discrete scales accept string, boolean, categorical data
        // With String/Bool transforms, they can also accept other types that will be cast
        matches!(
            dtype,
            DataType::String | DataType::Boolean | DataType::Categorical(_, _)
        )
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

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        _properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        // Discrete scales don't support expansion
        match user_range {
            Some(range) if super::input_range_has_nulls(range) => {
                Err("Discrete scale input range cannot contain null placeholders".to_string())
            }
            Some(range) => Ok(Some(range.to_vec())),
            None => Ok(compute_unique_values(columns)),
        }
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
                // If palette not found, leave as Palette for Vega-Lite to handle
            }
            Some(OutputRange::Array(_)) => {
                // Already an array, nothing to do
            }
        }

        // Phase 2: Size the Array to match category count
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

/// Compute discrete input range as unique sorted values from Columns.
///
/// For boolean columns, returns `ArrayElement::Boolean` values in logical order `[false, true]`,
/// with `ArrayElement::Null` appended if null values exist in the data.
/// For other column types, returns `ArrayElement::String` values sorted alphabetically,
/// with `ArrayElement::Null` appended if null values exist in the data.
fn compute_unique_values(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
    if column_refs.is_empty() {
        return None;
    }

    // Check if all columns are boolean
    let all_boolean = column_refs.iter().all(|c| c.dtype() == &DataType::Boolean);

    if all_boolean {
        // For boolean columns, return ArrayElement::Boolean values
        // Order: [false, true, null] for consistency (logical order, null at end)
        let mut has_false = false;
        let mut has_true = false;
        let mut has_null = false;

        for column in column_refs {
            if let Ok(ca) = column.as_materialized_series().bool() {
                for val in ca.into_iter() {
                    match val {
                        Some(true) => has_true = true,
                        Some(false) => has_false = true,
                        None => has_null = true,
                    }
                }
            }
        }

        let mut result = Vec::new();
        if has_false {
            result.push(ArrayElement::Boolean(false));
        }
        if has_true {
            result.push(ArrayElement::Boolean(true));
        }
        if has_null {
            result.push(ArrayElement::Null);
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    } else {
        // String-based logic for other types (categorical, string)
        let mut unique_values: Vec<String> = Vec::new();
        let mut has_null = false;

        for column in column_refs {
            let series = column.as_materialized_series();
            if let Ok(unique) = series.unique() {
                for i in 0..unique.len() {
                    if let Ok(val) = unique.get(i) {
                        if val.is_null() {
                            has_null = true;
                        } else {
                            let s = val.to_string();
                            let clean = s.trim_matches('"').to_string();
                            if !unique_values.contains(&clean) {
                                unique_values.push(clean);
                            }
                        }
                    }
                }
            }
        }

        if unique_values.is_empty() && !has_null {
            None
        } else {
            unique_values.sort();
            let mut result: Vec<ArrayElement> = unique_values
                .into_iter()
                .map(ArrayElement::String)
                .collect();
            // Null at end of inferred range
            if has_null {
                result.push(ArrayElement::Null);
            }
            Some(result)
        }
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

    #[test]
    fn test_compute_unique_values_includes_null_for_strings() {
        use polars::prelude::*;

        // Create a column with some null values
        let series = Series::new("cat".into(), &[Some("A"), Some("B"), None, Some("C")]);
        let column = series.into_column();
        let columns = vec![&column];

        let result = compute_unique_values(&columns);
        assert!(result.is_some());
        let values = result.unwrap();

        // Should have A, B, C sorted, plus Null at the end
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], ArrayElement::String("A".to_string()));
        assert_eq!(values[1], ArrayElement::String("B".to_string()));
        assert_eq!(values[2], ArrayElement::String("C".to_string()));
        assert_eq!(values[3], ArrayElement::Null);
    }

    #[test]
    fn test_compute_unique_values_includes_null_for_booleans() {
        use polars::prelude::*;

        // Create a boolean column with null
        let series = Series::new("flag".into(), &[Some(true), Some(false), None]);
        let column = series.into_column();
        let columns = vec![&column];

        let result = compute_unique_values(&columns);
        assert!(result.is_some());
        let values = result.unwrap();

        // Should have false, true, null (logical order)
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], ArrayElement::Boolean(false));
        assert_eq!(values[1], ArrayElement::Boolean(true));
        assert_eq!(values[2], ArrayElement::Null);
    }

    #[test]
    fn test_compute_unique_values_no_null_when_none_present() {
        use polars::prelude::*;

        // Create a column without nulls
        let series = Series::new("cat".into(), vec!["A", "B", "C"]);
        let column = series.into_column();
        let columns = vec![&column];

        let result = compute_unique_values(&columns);
        assert!(result.is_some());
        let values = result.unwrap();

        // Should have A, B, C sorted, no Null
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], ArrayElement::String("A".to_string()));
        assert_eq!(values[1], ArrayElement::String("B".to_string()));
        assert_eq!(values[2], ArrayElement::String("C".to_string()));
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
}
