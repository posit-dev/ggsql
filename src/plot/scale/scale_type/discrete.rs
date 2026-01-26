//! Discrete scale type implementation

use std::collections::HashMap;

use polars::prelude::{Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
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
        // Discrete scales support oob (censor or keep, not squish) and reverse
        &["oob", "reverse"]
    }

    fn get_property_default(&self, _aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            // Discrete scales only support "censor" - always default to it
            "oob" => Some(ParameterValue::String(super::OOB_CENSOR.to_string())),
            "reverse" => Some(ParameterValue::Boolean(false)),
            _ => None,
        }
    }

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::String | DataType::Boolean | DataType::Categorical(_, _)
        )
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
        input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        let count = input_range.map(|r| r.len()).unwrap_or(0);
        if count == 0 {
            return Ok(None);
        }

        match aesthetic {
            "color" | "colour" | "fill" | "stroke" => {
                let palette = palettes::get_color_palette("ggsql")
                    .ok_or_else(|| "Default color palette 'ggsql' not found".to_string())?;
                Ok(Some(palettes::expand_palette(palette, count, "ggsql")?))
            }
            "shape" => {
                let palette = palettes::get_shape_palette("default")
                    .ok_or_else(|| "Default shape palette not found".to_string())?;
                Ok(Some(palettes::expand_palette(palette, count, "default")?))
            }
            "linetype" => {
                let palette = palettes::get_linetype_palette("default")
                    .ok_or_else(|| "Default linetype palette not found".to_string())?;
                Ok(Some(palettes::expand_palette(palette, count, "default")?))
            }
            _ => Ok(None),
        }
    }
}

/// Compute discrete input range as unique sorted values from Columns.
fn compute_unique_values(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
    let mut unique_values: Vec<String> = Vec::new();

    for column in column_refs {
        let series = column.as_materialized_series();
        if let Ok(unique) = series.unique() {
            for i in 0..unique.len() {
                if let Ok(val) = unique.get(i) {
                    let s = val.to_string();
                    if s != "null" {
                        let clean = s.trim_matches('"').to_string();
                        if !unique_values.contains(&clean) {
                            unique_values.push(clean);
                        }
                    }
                }
            }
        }
    }

    if unique_values.is_empty() {
        None
    } else {
        unique_values.sort();
        Some(
            unique_values
                .into_iter()
                .map(ArrayElement::String)
                .collect(),
        )
    }
}

impl std::fmt::Display for Discrete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
