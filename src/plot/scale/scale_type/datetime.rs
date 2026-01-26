//! DateTime scale type implementation

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
use crate::plot::ArrayElement;

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

    fn allows_data_type(&self, dtype: &DataType) -> bool {
        matches!(dtype, DataType::Datetime(_, _))
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let computed = compute_datetime_range(columns);

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

/// Compute datetime input range as [min, max] ISO datetime strings from Columns.
fn compute_datetime_range(column_refs: &[&Column]) -> Option<Vec<ArrayElement>> {
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
            let to_iso = |ts: i64| -> Option<String> {
                // Polars Datetime is microseconds since epoch
                let secs = ts / 1_000_000;
                let nsecs = ((ts % 1_000_000) * 1000) as u32;
                let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nsecs)?;
                Some(dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
            };
            Some(vec![
                ArrayElement::String(to_iso(min_ts)?),
                ArrayElement::String(to_iso(max_ts)?),
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
