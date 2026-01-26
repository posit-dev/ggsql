//! Continuous scale type implementation

use std::collections::HashMap;

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait, TransformKind};
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

    fn allowed_transforms(&self) -> &'static [TransformKind] {
        &[
            TransformKind::Identity,
            TransformKind::Log10,
            TransformKind::Log2,
            TransformKind::Log,
            TransformKind::Sqrt,
            TransformKind::Asinh,
            TransformKind::PseudoLog,
        ]
    }

    fn default_transform(&self, aesthetic: &str) -> TransformKind {
        match aesthetic {
            "size" => TransformKind::Sqrt, // Area-proportional scaling
            _ => TransformKind::Identity,
        }
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

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Int8
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
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let computed = compute_numeric_range(columns);
        let (mult, add) = super::get_expand_factors(properties);

        // Apply expansion to computed range
        let expanded = computed.map(|range| super::expand_numeric_range(&range, mult, add));

        match user_range {
            None => Ok(expanded),
            Some(range) if super::input_range_has_nulls(range) => {
                // User provided partial range with nulls - merge with expanded computed
                match expanded {
                    Some(inferred) => Ok(Some(super::merge_with_inferred(range, &inferred))),
                    None => Ok(Some(range.to_vec())),
                }
            }
            Some(range) => {
                // User provided explicit full range - still apply expansion
                Ok(Some(super::expand_numeric_range(range, mult, add)))
            }
        }
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        match aesthetic {
            "stroke" | "fill" | "colour" | "color" => {
                let palette = palettes::get_color_palette("sequential")
                    .ok_or_else(|| "Default color palette 'ggsql' not found".to_string())?;
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
}

/// Compute numeric input range as [min, max] from Columns.
fn compute_numeric_range(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<f64> = None;
    let mut global_max: Option<f64> = None;

    for column in column_refs {
        let series = column.as_materialized_series();
        if let Ok(ca) = series.cast(&DataType::Float64) {
            if let Ok(f64_series) = ca.f64() {
                if let Some(min) = f64_series.min() {
                    global_min = Some(global_min.map_or(min, |m| m.min(min)));
                }
                if let Some(max) = f64_series.max() {
                    global_max = Some(global_max.map_or(max, |m| m.max(max)));
                }
            }
        }
    }

    match (global_min, global_max) {
        (Some(min), Some(max)) => Some(vec![ArrayElement::Number(min), ArrayElement::Number(max)]),
        _ => None,
    }
}

impl std::fmt::Display for Continuous {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
