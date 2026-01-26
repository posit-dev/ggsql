//! DateTime scale type implementation

use std::collections::HashMap;

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
use crate::plot::{ArrayElement, ParameterValue};

/// DateTime scale type - for datetime data (maps to temporal type)
#[derive(Debug, Clone, Copy)]
pub struct DateTime;

impl ScaleTypeTrait for DateTime {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::DateTime
    }

    fn name(&self) -> &'static str {
        "datetime"
    }

    // Uses default allowed_transforms() returning [TransformKind::Identity]

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
        matches!(dtype, DataType::Datetime(_, _))
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let (mult, add) = super::get_expand_factors(properties);

        // Compute datetime range with expansion applied (returns ISO strings)
        let expanded = compute_datetime_range_with_expansion(columns, mult, add);

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
                // User provided explicit datetime range - don't expand
                Ok(Some(range.to_vec()))
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

/// Compute datetime input range as [min, max] ISO datetime strings from Columns,
/// with expansion applied.
fn compute_datetime_range_with_expansion(
    column_refs: &[&Column],
    mult: f64,
    add: f64,
) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<i64> = None;
    let mut global_max: Option<i64> = None;

    for column in column_refs {
        let series = column.as_materialized_series();
        if let Ok(dt_ca) = series.datetime() {
            // Get the underlying physical representation (Int64) for min/max
            let physical = &dt_ca.phys;
            if let Some(min) = physical.min() {
                global_min = Some(global_min.map_or(min, |m| m.min(min)));
            }
            if let Some(max) = physical.max() {
                global_max = Some(global_max.map_or(max, |m| m.max(max)));
            }
        }
    }

    match (global_min, global_max) {
        (Some(min_ts), Some(max_ts)) => {
            // Apply expansion on the numeric microseconds
            let span = (max_ts - min_ts) as f64;
            // Note: add is in "units" - for datetime we interpret as microseconds
            let add_us = add * 1_000_000.0; // Convert seconds to microseconds
            let expanded_min = (min_ts as f64 - span * mult - add_us).floor() as i64;
            let expanded_max = (max_ts as f64 + span * mult + add_us).ceil() as i64;

            let to_iso = |ts: i64| -> Option<String> {
                // Polars Datetime is microseconds since epoch
                let secs = ts / 1_000_000;
                let nsecs = ((ts % 1_000_000).abs() * 1000) as u32;
                let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nsecs)?;
                Some(dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
            };
            Some(vec![
                ArrayElement::String(to_iso(expanded_min)?),
                ArrayElement::String(to_iso(expanded_max)?),
            ])
        }
        _ => None,
    }
}

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
