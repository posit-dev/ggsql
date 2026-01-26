//! Time scale type implementation

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
use crate::plot::ArrayElement;

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

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(dtype, DataType::Time)
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let computed = compute_time_range(columns);

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

/// Compute time input range as [min, max] ISO time strings from Columns.
fn compute_time_range(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
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
            let to_iso = |ns: i64| -> Option<String> {
                // Polars Time is nanoseconds since midnight
                let total_secs = ns / 1_000_000_000;
                let nanos = (ns % 1_000_000_000) as u32;
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
                ArrayElement::String(to_iso(min_ns)?),
                ArrayElement::String(to_iso(max_ns)?),
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
