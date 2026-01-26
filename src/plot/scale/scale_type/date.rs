//! Date scale type implementation

use std::collections::HashMap;

use polars::prelude::{ChunkAgg, Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
use crate::plot::{ArrayElement, ParameterValue};

/// Date scale type - for date data (maps to temporal type)
#[derive(Debug, Clone, Copy)]
pub struct Date;

impl ScaleTypeTrait for Date {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Date
    }

    fn name(&self) -> &'static str {
        "date"
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
        matches!(dtype, DataType::Date)
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        columns: &[&Column],
        properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        let (mult, add) = super::get_expand_factors(properties);

        // Compute date range with expansion applied (returns ISO strings)
        let expanded = compute_date_range_with_expansion(columns, mult, add);

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
                // User provided explicit date range - apply expansion
                // Note: For date strings, we don't expand user-provided ranges
                // since we can't easily parse arbitrary date formats
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

/// Compute date input range as [min_date, max_date] ISO strings from Columns,
/// with expansion applied.
fn compute_date_range_with_expansion(
    column_refs: &[&Column],
    mult: f64,
    add: f64,
) -> Option<Vec<ArrayElement>> {
    let mut global_min: Option<i32> = None;
    let mut global_max: Option<i32> = None;

    for column in column_refs {
        let series = column.as_materialized_series();
        if let Ok(date_ca) = series.date() {
            // Get the underlying physical representation (Int32) for min/max
            let physical = &date_ca.phys;
            if let Some(min) = physical.min() {
                global_min = Some(global_min.map_or(min, |m| m.min(min)));
            }
            if let Some(max) = physical.max() {
                global_max = Some(global_max.map_or(max, |m| m.max(max)));
            }
        }
    }

    match (global_min, global_max) {
        (Some(min_days), Some(max_days)) => {
            // Apply expansion on the numeric days
            let span = (max_days - min_days) as f64;
            let expanded_min_days = (min_days as f64 - span * mult - add).floor() as i64;
            let expanded_max_days = (max_days as f64 + span * mult + add).ceil() as i64;

            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
            let min_date = epoch + chrono::Duration::days(expanded_min_days);
            let max_date = epoch + chrono::Duration::days(expanded_max_days);
            Some(vec![
                ArrayElement::String(min_date.format("%Y-%m-%d").to_string()),
                ArrayElement::String(max_date.format("%Y-%m-%d").to_string()),
            ])
        }
        _ => None,
    }
}

impl std::fmt::Display for Date {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
