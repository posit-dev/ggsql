//! Continuous scale type implementation

use polars::prelude::DataType;

use super::{ScaleTypeKind, ScaleTypeTrait, SqlTypeNames, TransformKind, OOB_CENSOR, OOB_SQUISH};
use crate::plot::{ArrayElement, ParameterValue};

/// Continuous scale type - for continuous numeric data
#[derive(Debug, Clone, Copy)]
pub struct Continuous;

impl ScaleTypeTrait for Continuous {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Continuous
    }

    fn name(&self) -> &'static str {
        "continuous"
    }

    fn validate_dtype(&self, dtype: &DataType) -> Result<(), String> {
        match dtype {
            // Accept all numeric types
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Ok(()),
            // Accept temporal types
            DataType::Date | DataType::Datetime(_, _) | DataType::Time => Ok(()),
            // Reject discrete types
            DataType::String => Err("Continuous scale cannot be used with String data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            DataType::Boolean => Err("Continuous scale cannot be used with Boolean data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            DataType::Categorical(_, _) => Err("Continuous scale cannot be used with Categorical data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            // Other types - provide generic message
            other => Err(format!(
                "Continuous scale cannot be used with {:?} data. \
                 Continuous scales require numeric (Int, Float) or temporal (Date, DateTime, Time) data.",
                other
            )),
        }
    }

    fn allowed_transforms(&self) -> &'static [TransformKind] {
        &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Log2,
            TransformKind::Log,
            TransformKind::Sqrt,
            TransformKind::Square,
            TransformKind::Exp10,
            TransformKind::Exp2,
            TransformKind::Exp,
            TransformKind::Asinh,
            TransformKind::PseudoLog,
            // Integer transform for integer casting
            TransformKind::Integer,
            // Temporal transforms for date/datetime/time data
            TransformKind::Date,
            TransformKind::DateTime,
            TransformKind::Time,
        ]
    }

    fn default_transform(
        &self,
        _aesthetic: &str,
        column_dtype: Option<&DataType>,
    ) -> TransformKind {
        // First check column data type for temporal transforms
        if let Some(dtype) = column_dtype {
            match dtype {
                DataType::Date => return TransformKind::Date,
                DataType::Datetime(_, _) => return TransformKind::DateTime,
                DataType::Time => return TransformKind::Time,
                _ => {}
            }
        }

        // Default to identity (linear) for all aesthetics
        TransformKind::Identity
    }

    fn allowed_properties(&self, aesthetic: &str) -> &'static [&'static str] {
        if super::is_positional_aesthetic(aesthetic) {
            &["expand", "oob", "reverse", "breaks", "pretty"]
        } else {
            &["oob", "reverse", "breaks", "pretty"]
        }
    }

    fn get_property_default(&self, aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            "expand" if super::is_positional_aesthetic(aesthetic) => {
                Some(ParameterValue::Number(super::DEFAULT_EXPAND_MULT))
            }
            "oob" => Some(ParameterValue::String(
                super::default_oob(aesthetic).to_string(),
            )),
            "reverse" => Some(ParameterValue::Boolean(false)),
            "breaks" => Some(ParameterValue::Number(
                super::super::breaks::DEFAULT_BREAK_COUNT as f64,
            )),
            "pretty" => Some(ParameterValue::Boolean(true)),
            _ => None,
        }
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _scale: &super::super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        match aesthetic {
            // Note: "color"/"colour" already split to fill/stroke before scale resolution
            "stroke" | "fill" => {
                let palette = palettes::get_color_palette("sequential")
                    .ok_or_else(|| "Default color palette 'sequential' not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|col: &&str| ArrayElement::String(col.to_string()))
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
            _ => Ok(None),
        }
    }

    /// Pre-stat SQL transformation for continuous scales.
    ///
    /// Supports OOB modes:
    /// - "censor": CASE WHEN col >= min AND col <= max THEN col ELSE NULL END
    /// - "squish": GREATEST(min, LEAST(col, max))
    /// - "keep": No transformation (returns None)
    ///
    /// Only applies when input_range is explicitly specified via FROM clause.
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
        if input_range.len() < 2 {
            return None;
        }

        // Get min/max from input range
        let (min, max) = match (&input_range[0], &input_range[input_range.len() - 1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => (*min, *max),
            _ => return None,
        };

        // Get OOB mode from properties (default is aesthetic-dependent, set in resolve_properties)
        let oob = scale
            .properties
            .get("oob")
            .and_then(|p| match p {
                ParameterValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or(super::default_oob(&scale.aesthetic));

        match oob {
            OOB_CENSOR => Some(format!(
                "(CASE WHEN {} >= {} AND {} <= {} THEN {} ELSE NULL END)",
                column_name, min, column_name, max, column_name
            )),
            OOB_SQUISH => Some(format!(
                "GREATEST({}, LEAST({}, {}))",
                min, max, column_name
            )),
            _ => None, // "keep" = no transformation
        }
    }
}

impl std::fmt::Display for Continuous {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scale::Scale;

    /// Helper to create default type names for tests
    fn test_type_names() -> SqlTypeNames {
        SqlTypeNames::default()
    }

    #[test]
    fn test_pre_stat_transform_sql_censor() {
        let continuous = Continuous;
        let mut scale = Scale::new("y");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("censor".to_string()),
        );

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        assert!(sql.is_some());
        let sql = sql.unwrap();
        // Should generate CASE WHEN for censor
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("value >= 0"));
        assert!(sql.contains("value <= 100"));
        assert!(sql.contains("ELSE NULL"));
    }

    #[test]
    fn test_pre_stat_transform_sql_squish() {
        let continuous = Continuous;
        let mut scale = Scale::new("y");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("squish".to_string()),
        );

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        assert!(sql.is_some());
        let sql = sql.unwrap();
        // Should generate GREATEST/LEAST for squish
        assert!(sql.contains("GREATEST"));
        assert!(sql.contains("LEAST"));
    }

    #[test]
    fn test_pre_stat_transform_sql_keep() {
        let continuous = Continuous;
        let mut scale = Scale::new("x"); // positional aesthetic defaults to keep
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        scale.properties.insert(
            "oob".to_string(),
            ParameterValue::String("keep".to_string()),
        );

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        // Should return None for keep (no transformation)
        assert!(sql.is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_no_explicit_range() {
        let continuous = Continuous;
        let mut scale = Scale::new("y");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        // explicit_input_range = false (inferred from data)
        scale.explicit_input_range = false;

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        // Should return None (no OOB handling for inferred ranges)
        assert!(sql.is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_default_oob_for_positional() {
        let continuous = Continuous;
        let mut scale = Scale::new("x"); // positional aesthetic
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        // No oob property - should use default (keep for positional)

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        // Should return None since default for positional is "keep"
        assert!(sql.is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_default_oob_for_non_positional() {
        let continuous = Continuous;
        let mut scale = Scale::new("color"); // non-positional aesthetic
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        scale.explicit_input_range = true;
        // No oob property - should use default (censor for non-positional)

        let sql = continuous.pre_stat_transform_sql(
            "value",
            &DataType::Float64,
            &scale,
            &test_type_names(),
        );

        // Should generate censor SQL since default for non-positional is "censor"
        assert!(sql.is_some());
        let sql = sql.unwrap();
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("ELSE NULL"));
    }

    // =========================================================================
    // Dtype Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dtype_accepts_numeric() {
        use super::ScaleTypeTrait;

        let continuous = Continuous;
        assert!(continuous.validate_dtype(&DataType::Int8).is_ok());
        assert!(continuous.validate_dtype(&DataType::Int16).is_ok());
        assert!(continuous.validate_dtype(&DataType::Int32).is_ok());
        assert!(continuous.validate_dtype(&DataType::Int64).is_ok());
        assert!(continuous.validate_dtype(&DataType::UInt8).is_ok());
        assert!(continuous.validate_dtype(&DataType::UInt16).is_ok());
        assert!(continuous.validate_dtype(&DataType::UInt32).is_ok());
        assert!(continuous.validate_dtype(&DataType::UInt64).is_ok());
        assert!(continuous.validate_dtype(&DataType::Float32).is_ok());
        assert!(continuous.validate_dtype(&DataType::Float64).is_ok());
    }

    #[test]
    fn test_validate_dtype_accepts_temporal() {
        use super::ScaleTypeTrait;
        use polars::prelude::TimeUnit;

        let continuous = Continuous;
        assert!(continuous.validate_dtype(&DataType::Date).is_ok());
        assert!(continuous
            .validate_dtype(&DataType::Datetime(TimeUnit::Microseconds, None))
            .is_ok());
        assert!(continuous.validate_dtype(&DataType::Time).is_ok());
    }

    #[test]
    fn test_validate_dtype_rejects_string() {
        use super::ScaleTypeTrait;

        let continuous = Continuous;
        let result = continuous.validate_dtype(&DataType::String);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("String"));
        assert!(err.contains("DISCRETE"));
    }

    #[test]
    fn test_validate_dtype_rejects_boolean() {
        use super::ScaleTypeTrait;

        let continuous = Continuous;
        let result = continuous.validate_dtype(&DataType::Boolean);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Boolean"));
        assert!(err.contains("DISCRETE"));
    }
}
