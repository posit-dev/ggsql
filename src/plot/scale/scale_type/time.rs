//! Time scale type implementation

use std::collections::HashMap;

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
use crate::plot::{ArrayElement, ParameterValue};

/// Time scale type - for time data (maps to temporal type)
#[derive(Debug, Clone, Copy)]
pub struct Time;

impl ScaleTypeTrait for Time {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Time
    }

    fn name(&self) -> &'static str {
        "time"
    }

    fn allowed_transforms(&self) -> &'static [&'static str] {
        &["identity"]
    }

    fn allowed_properties(&self, aesthetic: &str) -> &'static [&'static str] {
        if super::is_positional_aesthetic(aesthetic) {
            &["expand", "oob", "reverse"]
        } else {
            &["oob", "reverse"]
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
            _ => None,
        }
    }

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(dtype, DataType::Time)
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let (mult, add) = super::get_expand_factors(properties);

        // Compute time range with expansion applied (returns ISO strings)
        let expanded = compute_time_range_with_expansion(columns, mult, add);

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
                // User provided explicit time range - don't expand
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

/// Compute time input range as [min, max] ISO time strings from Columns,
/// with expansion applied.
fn compute_time_range_with_expansion(
    column_refs: &[&Column],
    mult: f64,
    add: f64,
) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<i64> = None;
    let mut global_max: Option<i64> = None;

    for column in column_refs {
        let series = column.as_materialized_series();
        if let Ok(time_ca) = series.time() {
            // Get the underlying physical representation (Int64 nanoseconds) for min/max
            let physical = &time_ca.phys;
            if let Some(min) = physical.min() {
                global_min = Some(global_min.map_or(min, |m| m.min(min)));
            }
            if let Some(max) = physical.max() {
                global_max = Some(global_max.map_or(max, |m| m.max(max)));
            }
        }
    }

    match (global_min, global_max) {
        (Some(min_ns), Some(max_ns)) => {
            // Apply expansion on the numeric nanoseconds
            let span = (max_ns - min_ns) as f64;
            // Note: add is in "units" - for time we interpret as nanoseconds
            let add_ns = add * 1_000_000_000.0; // Convert seconds to nanoseconds
            let expanded_min = (min_ns as f64 - span * mult - add_ns).max(0.0).floor() as i64;
            let expanded_max = (max_ns as f64 + span * mult + add_ns).ceil() as i64;

            let to_iso = |ns: i64| -> Option<String> {
                // Polars Time is nanoseconds since midnight
                let total_secs = ns / 1_000_000_000;
                let nanos = (ns % 1_000_000_000).unsigned_abs() as u32;
                let hours = total_secs / 3600;
                let mins = (total_secs % 3600) / 60;
                let secs = total_secs % 60;
                let time = chrono::NaiveTime::from_hms_nano_opt(
                    hours as u32,
                    mins as u32,
                    secs as u32,
                    nanos,
                )?;
                Some(time.format("%H:%M:%S%.3f").to_string())
            };
            Some(vec![
                ArrayElement::String(to_iso(expanded_min)?),
                ArrayElement::String(to_iso(expanded_max)?),
            ])
        }
        _ => None,
    }
}

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
