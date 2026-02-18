//! Binned scale type implementation

use std::collections::HashMap;

use polars::prelude::DataType;

use super::{
    expand_numeric_range, resolve_common_steps, ScaleDataContext, ScaleTypeKind, ScaleTypeTrait,
    TransformKind, OOB_SQUISH,
};
use crate::plot::{ArrayElement, ParameterValue};

use super::InputRange;

/// Prune breaks that would create empty edge bins.
///
/// Removes terminal breaks if both the break AND its neighbor are outside
/// the original data range. This prevents completely empty bins while
/// allowing breaks that partially extend beyond data (for nice labels).
fn prune_empty_edge_bins(breaks: &mut Vec<ArrayElement>, data_range: &[ArrayElement]) {
    if breaks.len() < 3 || data_range.len() < 2 {
        return; // Need at least 3 breaks and valid data range
    }

    let data_min = match data_range[0].to_f64() {
        Some(v) => v,
        None => return,
    };
    let data_max = match data_range[data_range.len() - 1].to_f64() {
        Some(v) => v,
        None => return,
    };

    // Check front: if first break AND second break are both < data_min, remove first
    while breaks.len() >= 3 {
        let first = breaks[0].to_f64();
        let second = breaks[1].to_f64();
        if let (Some(f), Some(s)) = (first, second) {
            if f < data_min && s < data_min {
                breaks.remove(0);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Check back: if last break AND second-to-last break are both > data_max, remove last
    while breaks.len() >= 3 {
        let last = breaks[breaks.len() - 1].to_f64();
        let second_last = breaks[breaks.len() - 2].to_f64();
        if let (Some(l), Some(sl)) = (last, second_last) {
            if l > data_max && sl > data_max {
                breaks.pop();
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

/// Binned scale type - for binned/bucketed data
#[derive(Debug, Clone, Copy)]
pub struct Binned;

impl ScaleTypeTrait for Binned {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Binned
    }

    fn name(&self) -> &'static str {
        "binned"
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
            DataType::String => Err("Binned scale cannot be used with String data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            DataType::Boolean => Err("Binned scale cannot be used with Boolean data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            DataType::Categorical(_, _) => Err("Binned scale cannot be used with Categorical data. \
                 Use DISCRETE scale type instead, or ensure the column contains numeric or temporal data.".to_string()),
            // Other types - provide generic message
            other => Err(format!(
                "Binned scale cannot be used with {:?} data. \
                 Binned scales require numeric (Int, Float) or temporal (Date, DateTime, Time) data.",
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
            &["expand", "oob", "reverse", "breaks", "pretty", "closed"]
        } else {
            &["oob", "reverse", "breaks", "pretty", "closed"]
        }
    }

    fn get_property_default(&self, aesthetic: &str, name: &str) -> Option<ParameterValue> {
        match name {
            "expand" if super::is_positional_aesthetic(aesthetic) => {
                Some(ParameterValue::Number(super::DEFAULT_EXPAND_MULT))
            }
            // Binned scales default to "censor" - "keep" is not valid for binned
            "oob" => Some(ParameterValue::String(super::OOB_CENSOR.to_string())),
            "reverse" => Some(ParameterValue::Boolean(false)),
            "breaks" => Some(ParameterValue::Number(
                super::super::breaks::DEFAULT_BREAK_COUNT as f64,
            )),
            "pretty" => Some(ParameterValue::Boolean(true)),
            // "left" means bins are [lower, upper), "right" means (lower, upper]
            "closed" => Some(ParameterValue::String("left".to_string())),
            _ => None,
        }
    }

    fn default_output_range(
        &self,
        aesthetic: &str,
        _scale: &super::super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        use super::super::palettes;

        // Return full palette - sizing/interpolation is done in resolve_output_range()
        match aesthetic {
            // Note: "color"/"colour" already split to fill/stroke before scale resolution
            "stroke" | "fill" => {
                let palette = palettes::get_color_palette("sequential")
                    .ok_or_else(|| "Default color palette 'sequential' not found".to_string())?;
                Ok(Some(
                    palette
                        .iter()
                        .map(|s| ArrayElement::String(s.to_string()))
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
        use super::size_output_range;

        // Get bin count from resolved breaks
        let bin_count = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(breaks)) if breaks.len() >= 2 => breaks.len() - 1,
            _ => return Ok(()), // No breaks resolved yet
        };

        // Phase 1: Ensure we have an Array (convert Palette or fill default)
        // For linetype, use sequential ink-density palette as default (None or "sequential")
        let use_sequential_linetype = aesthetic == "linetype"
            && match &scale.output_range {
                None => true,
                Some(OutputRange::Palette(name)) => name.eq_ignore_ascii_case("sequential"),
                _ => false,
            };

        if use_sequential_linetype {
            // Generate sequential ink-density palette sized to bin_count
            let sequential = palettes::generate_linetype_sequential(bin_count);
            scale.output_range = Some(OutputRange::Array(
                sequential.into_iter().map(ArrayElement::String).collect(),
            ));
        } else {
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
        }

        // Phase 2: Size/interpolate to bin count
        size_output_range(scale, aesthetic, bin_count)?;

        Ok(())
    }

    /// Resolve scale properties from data context.
    ///
    /// Binned scales override this to add Binned-specific logic:
    /// - Implicit break handling (skip filtering for implicit breaks)
    /// - Break/range alignment (add range boundaries to breaks or compute range from breaks)
    /// - Terminal label suppression for oob='squish'
    fn resolve(
        &self,
        scale: &mut super::super::Scale,
        context: &ScaleDataContext,
        aesthetic: &str,
    ) -> Result<(), String> {
        // Steps 1-4: Common resolution logic (properties, transform, input_range, convert values)
        let common_result = resolve_common_steps(self, scale, context, aesthetic)?;
        let resolved_transform = common_result.transform;
        let (mult, add) = common_result.expand_factors;

        // 5. Calculate breaks for binned scale
        // Track whether breaks were explicit to determine alignment strategy:
        // - Implicit (count, no explicit range): keep extended breaks (they extend past data)
        // - Explicit (explicit range OR explicit breaks array): prune breaks to range, add boundaries
        let explicit_breaks_array = matches!(
            scale.properties.get("breaks"),
            Some(ParameterValue::Array(_))
        );
        let binned_implicit = !scale.explicit_input_range && !explicit_breaks_array;

        match scale.properties.get("breaks") {
            Some(ParameterValue::Number(_)) => {
                // Scalar count → calculate actual breaks and store as Array
                if let Some(breaks) = self.resolve_breaks(
                    scale.input_range.as_deref(),
                    &scale.properties,
                    scale.transform.as_ref(),
                ) {
                    // For binned implicit, keep all breaks (they extend past data).
                    // For binned explicit, filter to input range.
                    let filtered = if binned_implicit {
                        let mut result = breaks;
                        // Prune breaks that create completely empty edge bins
                        if let Some(InputRange::Continuous(data_range)) = &context.range {
                            prune_empty_edge_bins(&mut result, data_range);
                        }
                        result
                    } else if let Some(ref range) = scale.input_range {
                        super::super::super::breaks::filter_breaks_to_range(&breaks, range)
                    } else {
                        breaks
                    };
                    scale
                        .properties
                        .insert("breaks".to_string(), ParameterValue::Array(filtered));
                }
            }
            Some(ParameterValue::Array(explicit_breaks)) => {
                // User provided explicit breaks - convert using transform
                let converted: Vec<ArrayElement> = explicit_breaks
                    .iter()
                    .map(|elem| resolved_transform.parse_value(elem))
                    .collect();
                // Filter breaks to input range (explicit breaks always filtered)
                let filtered = if let Some(ref range) = scale.input_range {
                    super::super::super::breaks::filter_breaks_to_range(&converted, range)
                } else {
                    converted
                };
                scale
                    .properties
                    .insert("breaks".to_string(), ParameterValue::Array(filtered));
            }
            Some(ParameterValue::String(interval_str)) => {
                // Temporal interval string like "2 months", "week"
                // Only valid for temporal transforms (Date, DateTime, Time)
                use super::super::super::breaks::{
                    temporal_breaks_date, temporal_breaks_datetime, temporal_breaks_time,
                    TemporalInterval,
                };

                if let Some(interval) = TemporalInterval::create_from_str(interval_str) {
                    if let Some(ref range) = scale.input_range {
                        let breaks: Vec<ArrayElement> = match resolved_transform.transform_kind() {
                            TransformKind::Date => {
                                let min = range[0].to_f64().unwrap_or(0.0) as i32;
                                let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i32;
                                temporal_breaks_date(min, max, interval)
                                    .into_iter()
                                    .map(ArrayElement::String)
                                    .collect()
                            }
                            TransformKind::DateTime => {
                                let min = range[0].to_f64().unwrap_or(0.0) as i64;
                                let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i64;
                                temporal_breaks_datetime(min, max, interval)
                                    .into_iter()
                                    .map(ArrayElement::String)
                                    .collect()
                            }
                            TransformKind::Time => {
                                let min = range[0].to_f64().unwrap_or(0.0) as i64;
                                let max = range[range.len() - 1].to_f64().unwrap_or(0.0) as i64;
                                temporal_breaks_time(min, max, interval)
                                    .into_iter()
                                    .map(ArrayElement::String)
                                    .collect()
                            }
                            _ => vec![], // Non-temporal transforms don't support interval strings
                        };

                        if !breaks.is_empty() {
                            // Convert string breaks to appropriate temporal ArrayElement types
                            let converted: Vec<ArrayElement> = breaks
                                .iter()
                                .map(|elem| resolved_transform.parse_value(elem))
                                .collect();
                            // Filter to input range
                            let filtered = super::super::super::breaks::filter_breaks_to_range(
                                &converted, range,
                            );
                            scale
                                .properties
                                .insert("breaks".to_string(), ParameterValue::Array(filtered));
                        }
                    }
                }
            }
            _ => {}
        }

        // 5b. Binned-specific: align breaks and range for proper bins
        //
        // Simple rule:
        // - If explicit input range provided → add range boundaries as terminal breaks
        // - If no explicit input range → set input_range from terminal breaks
        let maybe_breaks = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(b)) => Some(b.clone()),
            _ => None,
        };

        if let Some(mut breaks) = maybe_breaks {
            let mut new_input_range: Option<Vec<ArrayElement>> = None;

            if scale.explicit_input_range {
                // Explicit input range provided → add range as terminal breaks
                if let Some(ref range) = scale.input_range {
                    add_range_boundaries_to_breaks(&mut breaks, range);
                }
            } else if breaks.len() >= 2 {
                // No explicit range → set input_range from terminal breaks
                let terminal_range = vec![
                    breaks.first().unwrap().clone(),
                    breaks.last().unwrap().clone(),
                ];
                // Only expand for positional aesthetics (x, y, etc.)
                // Non-positional aesthetics (color, fill, size) don't get expansion
                let final_range = if super::is_positional_aesthetic(aesthetic) {
                    expand_numeric_range(&terminal_range, mult, add)
                } else {
                    terminal_range
                };
                new_input_range = Some(final_range);
            }

            // Update the breaks in the scale
            scale
                .properties
                .insert("breaks".to_string(), ParameterValue::Array(breaks));

            // Update input_range if we computed a new one
            // Convert to proper type using transform (e.g., Number → Date for temporal)
            if let Some(range) = new_input_range {
                let converted: Vec<ArrayElement> = range
                    .iter()
                    .map(|elem| resolved_transform.parse_value(elem))
                    .collect();
                scale.input_range = Some(converted);
            }
        }

        // 6. Apply label template (RENAMING * => '...')
        // Default is '{}' to ensure we control formatting instead of Vega-Lite
        // For binned scales, apply to breaks array
        let template = &scale.label_template;

        let values_to_label = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(breaks)) => Some(breaks.clone()),
            _ => None,
        };

        if let Some(values) = values_to_label {
            let generated_labels =
                crate::format::apply_label_template(&values, template, &scale.label_mapping);
            scale.label_mapping = Some(generated_labels);
        }

        // 6b. Binned-specific: suppress terminal break labels for oob='squish'
        // since those bins extend to infinity (-∞ to first internal break, last internal break to +∞)
        if let Some(ParameterValue::String(oob)) = scale.properties.get("oob") {
            if oob == OOB_SQUISH {
                if let Some(ParameterValue::Array(breaks)) = scale.properties.get("breaks") {
                    if breaks.len() > 2 {
                        // Suppress first and last break labels
                        let first_key = breaks[0].to_key_string();
                        let last_key = breaks[breaks.len() - 1].to_key_string();

                        let label_mapping = scale.label_mapping.get_or_insert_with(HashMap::new);
                        label_mapping.insert(first_key, None);
                        label_mapping.insert(last_key, None);
                    }
                }
            }
        }

        // 7. Resolve output range (TO clause)
        self.resolve_output_range(scale, aesthetic)?;

        // Mark scale as resolved
        scale.resolved = true;

        Ok(())
    }

    /// Generate SQL for pre-stat binning transformation.
    ///
    /// Uses the resolved breaks to compute bin boundaries via CASE WHEN,
    /// mapping each value to its bin center. Supports arbitrary (non-evenly-spaced) breaks.
    ///
    /// The `closed` property controls which side of the bin is closed:
    /// - `"left"` (default): bins are `[lower, upper)`, last bin is `[lower, upper]`
    /// - `"right"`: bins are `(lower, upper]`, first bin is `[lower, upper]`
    ///
    /// This ensures:
    /// - Values are grouped into bins defined by break boundaries
    /// - Each bin is represented by its center value `(lower + upper) / 2`
    /// - Boundary values are not lost (edge bins include endpoints)
    /// - Data is binned BEFORE any stat transforms are applied
    ///
    /// # Column Casting
    ///
    /// Column type casting is handled earlier in the pipeline by `apply_column_casting()`.
    /// This function assumes the column already has the correct type.
    ///
    /// However, break literal values may still need casting for temporal types:
    /// ```sql
    /// CASE WHEN date_col >= CAST('2024-01-01' AS DATE) ...
    /// ```
    fn pre_stat_transform_sql(
        &self,
        column_name: &str,
        column_dtype: &DataType,
        scale: &super::super::Scale,
        type_names: &super::SqlTypeNames,
    ) -> Option<String> {
        use super::super::transform::TransformKind;

        // Get breaks from scale properties (calculated in resolve)
        // breaks should be an Array after resolution
        let breaks = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(arr)) => arr,
            _ => return None,
        };

        if breaks.len() < 2 {
            return None;
        }

        // Extract numeric break values (handles Number, Date, DateTime, Time via to_f64)
        let break_values: Vec<f64> = breaks.iter().filter_map(|e| e.to_f64()).collect();

        if break_values.len() < 2 {
            return None;
        }

        // Get closed property: "left" (default) or "right"
        let closed_left = match scale.properties.get("closed") {
            Some(ParameterValue::String(s)) => s != "right",
            _ => true, // default to left-closed
        };

        // Get oob property: "censor" (default) or "squish"
        // With "squish", terminal bins extend to infinity
        let oob_squish = match scale.properties.get("oob") {
            Some(ParameterValue::String(s)) => s == OOB_SQUISH,
            _ => false,
        };

        // Determine if break values need temporal formatting
        // Column is already cast to correct type, but break literals may need formatting
        let transform = scale.transform.as_ref();
        let is_temporal = matches!(
            column_dtype,
            DataType::Date | DataType::Datetime(..) | DataType::Time
        );

        // Build CASE WHEN clauses for each bin
        let num_bins = break_values.len() - 1;
        let mut cases = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            let lower = break_values[i];
            let upper = break_values[i + 1];
            let center = (lower + upper) / 2.0;

            let is_first = i == 0;
            let is_last = i == num_bins - 1;

            // Format break values based on column type
            // Column is already the correct type (casting handled earlier)
            let (lower_expr, upper_expr, center_expr) = if is_temporal {
                // For temporal columns, format break values as ISO strings with CAST
                if let Some(t) = transform {
                    let type_name = match t.transform_kind() {
                        TransformKind::Date => type_names.date.as_deref(),
                        TransformKind::DateTime => type_names.datetime.as_deref(),
                        TransformKind::Time => type_names.time.as_deref(),
                        _ => None,
                    };

                    match type_name {
                        Some(type_name) => {
                            let lower_iso = t
                                .format_as_iso(lower)
                                .unwrap_or_else(|| format!("{}", lower));
                            let upper_iso = t
                                .format_as_iso(upper)
                                .unwrap_or_else(|| format!("{}", upper));
                            let center_iso = t
                                .format_as_iso(center)
                                .unwrap_or_else(|| format!("{}", center));
                            (
                                format!("CAST('{}' AS {})", lower_iso, type_name),
                                format!("CAST('{}' AS {})", upper_iso, type_name),
                                format!("CAST('{}' AS {})", center_iso, type_name),
                            )
                        }
                        None => {
                            // No type name available - use raw numeric values
                            return Some(build_case_expression_numeric(
                                column_name,
                                &break_values,
                                closed_left,
                                oob_squish,
                            ));
                        }
                    }
                } else {
                    // No transform - use raw numeric values (days/µs/ns since epoch)
                    (
                        format!("{}", lower),
                        format!("{}", upper),
                        format!("{}", center),
                    )
                }
            } else {
                // Numeric column - use raw values
                (
                    format!("{}", lower),
                    format!("{}", upper),
                    format!("{}", center),
                )
            };

            let condition = build_bin_condition(
                column_name,
                &lower_expr,
                &upper_expr,
                closed_left,
                oob_squish,
                is_first,
                is_last,
            );

            cases.push(format!("WHEN {} THEN {}", condition, center_expr));
        }

        // Build final CASE expression
        Some(format!("(CASE {} ELSE NULL END)", cases.join(" ")))
    }
}

/// Build a SQL condition for a single bin.
///
/// Handles the operator selection based on closed side and bin position,
/// and the oob_squish logic for extending first/last bins to infinity.
fn build_bin_condition(
    column_name: &str,
    lower_expr: &str,
    upper_expr: &str,
    closed_left: bool,
    oob_squish: bool,
    is_first: bool,
    is_last: bool,
) -> String {
    // Determine operators based on closed side and bin position
    // closed="left": [lower, upper) except last bin which is [lower, upper]
    // closed="right": (lower, upper] except first bin which is [lower, upper]
    let (lower_op, upper_op) = if closed_left {
        (">=", if is_last { "<=" } else { "<" })
    } else {
        (if is_first { ">=" } else { ">" }, "<=")
    };

    if oob_squish && is_first && is_last {
        // Single bin with squish: capture everything
        "TRUE".to_string()
    } else if oob_squish && is_first {
        // First bin with squish: no lower bound, extends to -∞
        format!("{} {} {}", column_name, upper_op, upper_expr)
    } else if oob_squish && is_last {
        // Last bin with squish: no upper bound, extends to +∞
        format!("{} {} {}", column_name, lower_op, lower_expr)
    } else {
        // Normal bin with both bounds
        format!(
            "{} {} {} AND {} {} {}",
            column_name, lower_op, lower_expr, column_name, upper_op, upper_expr
        )
    }
}

/// Build a CASE expression for numeric binning (helper for non-temporal cases).
fn build_case_expression_numeric(
    column_name: &str,
    break_values: &[f64],
    closed_left: bool,
    oob_squish: bool,
) -> String {
    let num_bins = break_values.len() - 1;
    let mut cases = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        let lower = break_values[i];
        let upper = break_values[i + 1];
        let center = (lower + upper) / 2.0;

        let is_first = i == 0;
        let is_last = i == num_bins - 1;

        let condition = build_bin_condition(
            column_name,
            &lower.to_string(),
            &upper.to_string(),
            closed_left,
            oob_squish,
            is_first,
            is_last,
        );

        cases.push(format!("WHEN {} THEN {}", condition, center));
    }

    format!("(CASE {} ELSE NULL END)", cases.join(" "))
}

impl std::fmt::Display for Binned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// =============================================================================
// Binned Scale Break/Range Alignment Helpers
// =============================================================================

/// Add input range boundaries as terminal breaks if not already present.
///
/// This is used when an explicit input range or explicit breaks are provided.
/// Ensures that the range boundaries are included in the breaks so bins cover
/// the full specified range.
///
/// # Arguments
///
/// * `breaks` - The breaks array (will be modified in place)
/// * `range` - The input range [min, max]
pub fn add_range_boundaries_to_breaks(breaks: &mut Vec<ArrayElement>, range: &[ArrayElement]) {
    if range.len() < 2 || breaks.is_empty() {
        return;
    }

    let range_min = match range.first().and_then(|e| e.to_f64()) {
        Some(v) => v,
        None => return,
    };
    let range_max = match range.last().and_then(|e| e.to_f64()) {
        Some(v) => v,
        None => return,
    };

    // Check and add range_min as first break if needed
    if let Some(first_break) = breaks.first().and_then(|e| e.to_f64()) {
        if (first_break - range_min).abs() > 1e-9 && first_break > range_min {
            breaks.insert(0, ArrayElement::Number(range_min));
        }
    }

    // Check and add range_max as last break if needed
    if let Some(last_break) = breaks.last().and_then(|e| e.to_f64()) {
        if (last_break - range_max).abs() > 1e-9 && last_break < range_max {
            breaks.push(ArrayElement::Number(range_max));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::scale::{Scale, SqlTypeNames};

    /// Helper to create default type names for tests
    fn test_type_names() -> SqlTypeNames {
        SqlTypeNames {
            number: Some("DOUBLE".to_string()),
            integer: Some("BIGINT".to_string()),
            date: Some("DATE".to_string()),
            datetime: Some("TIMESTAMP".to_string()),
            time: Some("TIME".to_string()),
            string: Some("VARCHAR".to_string()),
            boolean: Some("BOOLEAN".to_string()),
        }
    }

    #[test]
    fn test_pre_stat_transform_sql_even_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(30.0),
            ]),
        );

        // Float64 column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // Should produce CASE WHEN with bin centers 5, 15, 25
        assert!(sql.contains("CASE"));
        assert!(sql.contains("WHEN value >= 0 AND value < 10 THEN 5"));
        assert!(sql.contains("WHEN value >= 10 AND value < 20 THEN 15"));
        // Last bin should be inclusive on both ends
        assert!(sql.contains("WHEN value >= 20 AND value <= 30 THEN 25"));
        assert!(sql.contains("ELSE NULL END"));
    }

    #[test]
    fn test_pre_stat_transform_sql_uneven_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        // Non-evenly-spaced breaks: [0, 10, 25, 100]
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(100.0),
            ]),
        );

        let sql = binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // Bin centers: (0+10)/2=5, (10+25)/2=17.5, (25+100)/2=62.5
        assert!(sql.contains("THEN 5")); // center of [0, 10)
        assert!(sql.contains("THEN 17.5")); // center of [10, 25)
        assert!(sql.contains("THEN 62.5")); // center of [25, 100]
    }

    #[test]
    fn test_pre_stat_transform_sql_closed_left_default() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );
        // No explicit closed property, should default to "left"

        let sql = binned
            .pre_stat_transform_sql("col", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // closed="left": [lower, upper) except last which is [lower, upper]
        assert!(sql.contains("col >= 0 AND col < 10"));
        assert!(sql.contains("col >= 10 AND col <= 20")); // last bin inclusive
    }

    #[test]
    fn test_pre_stat_transform_sql_closed_right() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );
        scale.properties.insert(
            "closed".to_string(),
            ParameterValue::String("right".to_string()),
        );

        let sql = binned
            .pre_stat_transform_sql("col", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // closed="right": first bin is [lower, upper], rest are (lower, upper]
        assert!(sql.contains("col >= 0 AND col <= 10")); // first bin inclusive
        assert!(sql.contains("col > 10 AND col <= 20"));
    }

    #[test]
    fn test_pre_stat_transform_sql_insufficient_breaks() {
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Only one break - not enough to form a bin
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0)]),
        );

        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_no_breaks() {
        let binned = Binned;
        let scale = Scale::new("x");
        // No breaks property at all

        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_pre_stat_transform_sql_number_breaks_returns_none() {
        let binned = Binned;
        let mut scale = Scale::new("x");
        // breaks is still a Number (count), not resolved to Array yet
        scale
            .properties
            .insert("breaks".to_string(), ParameterValue::Number(5.0));

        // Should return None because breaks hasn't been resolved to Array
        assert!(binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .is_none());
    }

    #[test]
    fn test_closed_property_default() {
        let binned = Binned;
        let default = binned.get_property_default("x", "closed");
        assert_eq!(default, Some(ParameterValue::String("left".to_string())));
    }

    #[test]
    fn test_closed_property_allowed() {
        let binned = Binned;
        let allowed = binned.allowed_properties("x");
        assert!(allowed.contains(&"closed"));
    }

    #[test]
    fn test_pre_stat_transform_sql_with_date_breaks() {
        // Test that Date breaks are correctly handled via to_f64()
        // When column is DATE and no explicit transform, use efficient numeric comparison
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use Date variants instead of Number
        // 2024-01-01 = 19724 days, 2024-02-01 = 19755 days, 2024-03-01 = 19784 days
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(19724), // 2024-01-01
                ArrayElement::Date(19755), // 2024-02-01
                ArrayElement::Date(19784), // 2024-03-01
            ]),
        );

        // Date column - no casting needed (types match)
        let sql =
            binned.pre_stat_transform_sql("date_col", &DataType::Date, &scale, &test_type_names());

        // Should successfully generate SQL (not return None due to filtered-out breaks)
        assert!(sql.is_some(), "SQL should be generated for Date breaks");
        let sql = sql.unwrap();

        // Verify the SQL contains the expected day values (numeric comparison)
        assert!(
            sql.contains("19724"),
            "SQL should contain first break value"
        );
        assert!(
            sql.contains("19755"),
            "SQL should contain second break value"
        );
        assert!(
            sql.contains("19784"),
            "SQL should contain third break value"
        );

        // Verify bin centers: (19724+19755)/2 = 19739.5, (19755+19784)/2 = 19769.5
        assert!(
            sql.contains("THEN 19739.5"),
            "SQL should contain first bin center"
        );
        assert!(
            sql.contains("THEN 19769.5"),
            "SQL should contain second bin center"
        );
    }

    #[test]
    fn test_pre_stat_transform_sql_with_datetime_breaks() {
        // Test that DateTime breaks are correctly handled via to_f64()
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use DateTime variants (microseconds since epoch)
        // Some arbitrary microsecond values for testing
        let dt1: i64 = 1_704_067_200_000_000; // 2024-01-01 00:00:00 UTC
        let dt2: i64 = 1_706_745_600_000_000; // 2024-02-01 00:00:00 UTC
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::DateTime(dt1),
                ArrayElement::DateTime(dt2),
            ]),
        );

        use polars::prelude::TimeUnit;
        let sql = binned.pre_stat_transform_sql(
            "datetime_col",
            &DataType::Datetime(TimeUnit::Microseconds, None),
            &scale,
            &test_type_names(),
        );

        // Should successfully generate SQL
        assert!(sql.is_some(), "SQL should be generated for DateTime breaks");
    }

    #[test]
    fn test_pre_stat_transform_sql_with_time_breaks() {
        // Test that Time breaks are correctly handled via to_f64()
        let binned = Binned;
        let mut scale = Scale::new("x");

        // Use Time variants (nanoseconds since midnight)
        // 6:00 AM = 6 * 60 * 60 * 1_000_000_000 ns
        // 12:00 PM = 12 * 60 * 60 * 1_000_000_000 ns
        // 18:00 PM = 18 * 60 * 60 * 1_000_000_000 ns
        let t1: i64 = 6 * 60 * 60 * 1_000_000_000;
        let t2: i64 = 12 * 60 * 60 * 1_000_000_000;
        let t3: i64 = 18 * 60 * 60 * 1_000_000_000;
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Time(t1),
                ArrayElement::Time(t2),
                ArrayElement::Time(t3),
            ]),
        );

        let sql =
            binned.pre_stat_transform_sql("time_col", &DataType::Time, &scale, &test_type_names());

        // Should successfully generate SQL
        assert!(sql.is_some(), "SQL should be generated for Time breaks");
    }

    // ==========================================================================
    // Type Casting Tests (Updated for Unified Casting)
    // ==========================================================================
    //
    // With the unified casting approach, column casting is done earlier in the
    // pipeline (by apply_column_casting). The binned scale's pre_stat_transform_sql
    // now assumes columns already have the correct type.
    //
    // These tests verify that:
    // 1. Temporal columns use temporal literal formatting (ISO dates with CAST)
    // 2. Numeric columns use raw numeric values
    // 3. No column casting (only break literal casting for temporal types)

    #[test]
    fn test_date_column_with_date_transform_uses_temporal_literals() {
        // DATE column + date transform → temporal literals with CAST
        // (Column already has correct type; break values need formatting)
        use crate::plot::scale::transform::Transform;

        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::date());
        scale.explicit_transform = true;

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(19724), // 2024-01-02
                ArrayElement::Date(19755), // 2024-02-02
                ArrayElement::Date(19784), // 2024-03-02
            ]),
        );

        // Date column - no column casting, but break values are formatted as ISO dates
        let sql = binned
            .pre_stat_transform_sql("date_col", &DataType::Date, &scale, &test_type_names())
            .unwrap();

        // Should NOT contain column CAST (column is already DATE)
        assert!(
            !sql.contains("CAST(date_col AS"),
            "SQL should not cast column when type matches. Got: {}",
            sql
        );
        // Break values should be cast as ISO date strings
        assert!(
            sql.contains("CAST('2024-01-02' AS DATE)"),
            "SQL should format break values as ISO dates. Got: {}",
            sql
        );
        assert!(
            sql.contains("CAST('2024-02-02' AS DATE)"),
            "SQL should format break values as ISO dates. Got: {}",
            sql
        );
    }

    #[test]
    fn test_numeric_column_no_transform_uses_raw_values() {
        // Numeric column + no explicit transform → raw numeric values
        // (Column is already numeric; break values are plain numbers)
        let binned = Binned;
        let mut scale = Scale::new("x");
        // No explicit transform set

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );

        // Float64 column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::Float64, &scale, &test_type_names())
            .unwrap();

        // Should NOT contain any CAST expressions
        assert!(
            !sql.contains("CAST("),
            "SQL should not contain CAST when column is numeric. Got: {}",
            sql
        );
        assert!(
            sql.contains("value >= 0"),
            "SQL should use raw column name. Got: {}",
            sql
        );
        assert!(
            sql.contains("THEN 5"),
            "SQL should use raw numeric center values. Got: {}",
            sql
        );
    }

    #[test]
    fn test_int_column_no_cast() {
        // INT64 column + no explicit transform → no cast needed
        let binned = Binned;
        let mut scale = Scale::new("x");

        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );

        // Int64 column - no casting needed
        let sql = binned
            .pre_stat_transform_sql("value", &DataType::Int64, &scale, &test_type_names())
            .unwrap();

        // Should NOT contain CAST expressions
        assert!(
            !sql.contains("CAST("),
            "SQL should not contain CAST when column is numeric"
        );
        assert!(sql.contains("value >= 0"), "SQL should use raw column name");
    }

    #[test]
    fn test_datetime_column_with_datetime_transform() {
        // DATETIME column + datetime transform → temporal literals
        use crate::plot::scale::transform::Transform;
        use polars::prelude::TimeUnit;

        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.transform = Some(Transform::datetime());
        scale.explicit_transform = true;

        // Use DateTime variants (microseconds since epoch)
        let dt1: i64 = 1_704_067_200_000_000; // 2024-01-01 00:00:00 UTC
        let dt2: i64 = 1_706_745_600_000_000; // 2024-02-01 00:00:00 UTC
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::DateTime(dt1),
                ArrayElement::DateTime(dt2),
            ]),
        );

        let sql = binned
            .pre_stat_transform_sql(
                "datetime_col",
                &DataType::Datetime(TimeUnit::Microseconds, None),
                &scale,
                &test_type_names(),
            )
            .unwrap();

        // Should contain CAST for break values but not column
        assert!(
            !sql.contains("CAST(datetime_col AS"),
            "SQL should not cast column when type matches"
        );
        assert!(
            sql.contains("CAST('2024-01-01") && sql.contains("AS TIMESTAMP"),
            "SQL should format break values as ISO datetime with CAST. Got: {}",
            sql
        );
    }

    // ==========================================================================
    // Output Range Interpolation Tests
    // ==========================================================================

    #[test]
    fn test_resolve_output_range_size_interpolation() {
        use super::ScaleTypeTrait;
        use crate::plot::scale::OutputRange;

        let binned = Binned;
        let mut scale = Scale::new("size");

        // Set up 5 bins (4 breaks)
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // Default size range is [1, 6]
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(6.0),
        ]));

        // Resolve output range
        binned.resolve_output_range(&mut scale, "size").unwrap();

        // Should have 4 evenly spaced values from 1 to 6
        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 4, "Should have 4 size values for 4 bins");
            // Values should be: 1.0, 2.666..., 4.333..., 6.0
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            assert!((nums[0] - 1.0).abs() < 0.001, "First value should be 1.0");
            assert!((nums[3] - 6.0).abs() < 0.001, "Last value should be 6.0");
            // Check evenly spaced
            let step = (nums[1] - nums[0]).abs();
            assert!(
                ((nums[2] - nums[1]).abs() - step).abs() < 0.001,
                "Values should be evenly spaced"
            );
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_linewidth_interpolation() {
        use super::ScaleTypeTrait;
        use crate::plot::scale::OutputRange;

        let binned = Binned;
        let mut scale = Scale::new("linewidth");

        // Set up 3 bins (2 breaks)
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // Linewidth range [1, 6]
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(1.0),
            ArrayElement::Number(6.0),
        ]));

        // Resolve output range
        binned
            .resolve_output_range(&mut scale, "linewidth")
            .unwrap();

        // Should have 2 evenly spaced values: 1.0 and 6.0
        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 2, "Should have 2 linewidth values for 2 bins");
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            assert!((nums[0] - 1.0).abs() < 0.001, "First value should be 1.0");
            assert!((nums[1] - 6.0).abs() < 0.001, "Last value should be 6.0");
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_opacity_interpolation() {
        use super::ScaleTypeTrait;
        use crate::plot::scale::OutputRange;

        let binned = Binned;
        let mut scale = Scale::new("opacity");

        // Set up 5 bins
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(20.0),
                ArrayElement::Number(40.0),
                ArrayElement::Number(60.0),
                ArrayElement::Number(80.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // Opacity range [0.1, 1.0]
        scale.output_range = Some(OutputRange::Array(vec![
            ArrayElement::Number(0.1),
            ArrayElement::Number(1.0),
        ]));

        // Resolve output range
        binned.resolve_output_range(&mut scale, "opacity").unwrap();

        // Should have 5 evenly spaced values from 0.1 to 1.0
        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 5, "Should have 5 opacity values for 5 bins");
            let nums: Vec<f64> = arr.iter().filter_map(|e| e.to_f64()).collect();
            assert!((nums[0] - 0.1).abs() < 0.001, "First value should be 0.1");
            assert!((nums[4] - 1.0).abs() < 0.001, "Last value should be 1.0");
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_linetype_sequential_default() {
        use super::ScaleTypeTrait;
        use crate::plot::scale::OutputRange;

        let binned = Binned;
        let mut scale = Scale::new("linetype");

        // Set up 4 bins (5 breaks)
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(25.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(75.0),
                ArrayElement::Number(100.0),
            ]),
        );

        // No output range specified - should use sequential ink palette
        scale.output_range = None;

        binned.resolve_output_range(&mut scale, "linetype").unwrap();

        // Should have 4 linetypes with increasing ink density
        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 4, "Should have 4 linetype values for 4 bins");

            // Verify all are strings (linetype patterns)
            let linetypes: Vec<&str> = arr
                .iter()
                .filter_map(|e| match e {
                    ArrayElement::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(linetypes.len(), 4, "All values should be strings");

            // Last should be solid (highest ink)
            assert_eq!(linetypes[3], "solid", "Last linetype should be solid");

            // First should be sparse (hex pattern like "1f")
            assert!(
                linetypes[0] != "solid",
                "First linetype should not be solid"
            );
        } else {
            panic!("Output range should be an Array");
        }
    }

    #[test]
    fn test_resolve_output_range_linetype_sequential_explicit() {
        use super::ScaleTypeTrait;
        use crate::plot::scale::OutputRange;

        let binned = Binned;
        let mut scale = Scale::new("linetype");

        // Set up 3 bins
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
                ArrayElement::Number(150.0),
            ]),
        );

        // Explicitly request sequential palette
        scale.output_range = Some(OutputRange::Palette("sequential".to_string()));

        binned.resolve_output_range(&mut scale, "linetype").unwrap();

        // Should have 3 linetypes
        if let Some(OutputRange::Array(arr)) = &scale.output_range {
            assert_eq!(arr.len(), 3, "Should have 3 linetype values for 3 bins");
        } else {
            panic!("Output range should be an Array");
        }
    }

    // ==========================================================================
    // OOB Squish Tests (Consolidated)
    // ==========================================================================

    #[test]
    fn test_pre_stat_transform_sql_oob_squish_variations() {
        // Test squish mode with different closed sides and bin counts
        // Format: (closed, breaks, expected_patterns)
        let test_cases: Vec<(&str, Vec<f64>, Vec<&str>)> = vec![
            // closed="left" with 3 bins (4 breaks)
            (
                "left",
                vec![0.0, 10.0, 20.0, 30.0],
                vec![
                    "WHEN value < 10 THEN 5",                  // First bin extends to -∞
                    "WHEN value >= 10 AND value < 20 THEN 15", // Middle bin
                    "WHEN value >= 20 THEN 25",                // Last bin extends to +∞
                ],
            ),
            // closed="right" with 3 bins (4 breaks)
            (
                "right",
                vec![0.0, 10.0, 20.0, 30.0],
                vec![
                    "WHEN value <= 10 THEN 5",                 // First bin extends to -∞
                    "WHEN value > 10 AND value <= 20 THEN 15", // Middle bin
                    "WHEN value > 20 THEN 25",                 // Last bin extends to +∞
                ],
            ),
        ];

        let binned = Binned;
        for (closed, breaks, expected) in test_cases {
            let mut scale = Scale::new("x");
            scale.properties.insert(
                "breaks".to_string(),
                ParameterValue::Array(breaks.iter().map(|&v| ArrayElement::Number(v)).collect()),
            );
            scale.properties.insert(
                "oob".to_string(),
                ParameterValue::String("squish".to_string()),
            );
            if closed == "right" {
                scale.properties.insert(
                    "closed".to_string(),
                    ParameterValue::String("right".to_string()),
                );
            }

            let sql = binned
                .pre_stat_transform_sql("value", &DataType::Float64, &scale, &test_type_names())
                .unwrap();
            for pattern in expected {
                assert!(
                    sql.contains(pattern),
                    "closed={}: Missing '{}'. Got: {}",
                    closed,
                    pattern,
                    sql
                );
            }
        }
    }

    #[test]
    fn test_pre_stat_transform_sql_oob_squish_edge_cases() {
        let binned = Binned;

        // Two bins (3 breaks) - first extends to -∞, second extends to +∞
        {
            let mut scale = Scale::new("x");
            scale.properties.insert(
                "breaks".to_string(),
                ParameterValue::Array(vec![
                    ArrayElement::Number(0.0),
                    ArrayElement::Number(50.0),
                    ArrayElement::Number(100.0),
                ]),
            );
            scale.properties.insert(
                "oob".to_string(),
                ParameterValue::String("squish".to_string()),
            );
            let sql = binned
                .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
                .unwrap();
            assert!(
                sql.contains("WHEN x < 50 THEN 25"),
                "Two bins: first should extend to -∞"
            );
            assert!(
                sql.contains("WHEN x >= 50 THEN 75"),
                "Two bins: last should extend to +∞"
            );
        }

        // Single bin (2 breaks) - captures everything
        {
            let mut scale = Scale::new("x");
            scale.properties.insert(
                "breaks".to_string(),
                ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]),
            );
            scale.properties.insert(
                "oob".to_string(),
                ParameterValue::String("squish".to_string()),
            );
            let sql = binned
                .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
                .unwrap();
            assert!(
                sql.contains("WHEN TRUE THEN 50"),
                "Single bin with squish should capture all. Got: {}",
                sql
            );
        }
    }

    #[test]
    fn test_pre_stat_transform_sql_oob_censor_default() {
        // Without oob='squish' (default censor), bins should have bounds
        let binned = Binned;
        let mut scale = Scale::new("x");
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
                ArrayElement::Number(20.0),
            ]),
        );

        let sql = binned
            .pre_stat_transform_sql("x", &DataType::Float64, &scale, &test_type_names())
            .unwrap();
        assert!(
            sql.contains("x >= 0 AND x < 10"),
            "First bin should have lower bound with censor"
        );
        assert!(
            sql.contains("x >= 10 AND x <= 20"),
            "Last bin should have upper bound with censor"
        );
    }

    #[test]
    fn test_build_case_expression_numeric_helper() {
        // Test the helper function with both oob modes
        let cases = vec![
            // (oob_squish, expected_patterns)
            (
                true,
                vec![
                    "WHEN col < 10 THEN 5",
                    "WHEN col >= 10 AND col < 20 THEN 15",
                    "WHEN col >= 20 THEN 25",
                ],
            ),
            (
                false,
                vec!["col >= 0 AND col < 10", "col >= 10 AND col <= 20"],
            ),
        ];

        for (oob_squish, expected) in cases {
            let breaks = if oob_squish {
                vec![0.0, 10.0, 20.0, 30.0]
            } else {
                vec![0.0, 10.0, 20.0]
            };
            let sql = build_case_expression_numeric("col", &breaks, true, oob_squish);
            for pattern in expected {
                assert!(
                    sql.contains(pattern),
                    "oob_squish={}: Missing '{}'. Got: {}",
                    oob_squish,
                    pattern,
                    sql
                );
            }
        }
    }

    // ==========================================================================
    // Break/Range Alignment Helper Tests (Consolidated)
    // ==========================================================================

    #[test]
    fn test_add_range_boundaries_to_breaks_variations() {
        // Test various cases of adding range boundaries
        // Format: (description, initial_breaks, range, expected_len, expected_first, expected_last)
        let test_cases: Vec<(&str, Vec<f64>, Vec<f64>, usize, f64, f64)> = vec![
            (
                "adds both",
                vec![20.0, 40.0, 60.0, 80.0],
                vec![0.0, 100.0],
                6,
                0.0,
                100.0,
            ),
            (
                "adds min only",
                vec![25.0, 50.0, 75.0, 100.0],
                vec![0.0, 100.0],
                5,
                0.0,
                100.0,
            ),
            (
                "adds max only",
                vec![0.0, 25.0, 50.0, 75.0],
                vec![0.0, 100.0],
                5,
                0.0,
                100.0,
            ),
            (
                "no change needed",
                vec![0.0, 25.0, 50.0, 75.0, 100.0],
                vec![0.0, 100.0],
                5,
                0.0,
                100.0,
            ),
            (
                "uneven breaks",
                vec![10.0, 30.0, 50.0, 70.0, 90.0],
                vec![0.0, 100.0],
                7,
                0.0,
                100.0,
            ),
        ];

        for (desc, initial, range, expected_len, expected_first, expected_last) in test_cases {
            let mut breaks: Vec<ArrayElement> =
                initial.iter().map(|&v| ArrayElement::Number(v)).collect();
            let range_arr: Vec<ArrayElement> =
                range.iter().map(|&v| ArrayElement::Number(v)).collect();

            super::add_range_boundaries_to_breaks(&mut breaks, &range_arr);

            assert_eq!(
                breaks.len(),
                expected_len,
                "{}: expected {} breaks",
                desc,
                expected_len
            );
            assert_eq!(
                breaks[0],
                ArrayElement::Number(expected_first),
                "{}: first should be {}",
                desc,
                expected_first
            );
            assert_eq!(
                breaks[breaks.len() - 1],
                ArrayElement::Number(expected_last),
                "{}: last should be {}",
                desc,
                expected_last
            );
        }
    }

    // ==========================================================================
    // Prune Empty Edge Bins Tests (Consolidated)
    // ==========================================================================

    #[test]
    fn test_prune_empty_edge_bins_numeric_variations() {
        // Test various pruning scenarios with numeric breaks
        // Format: (description, breaks, data_range, expected_len, expected_first, expected_last)
        let test_cases: Vec<(&str, Vec<f64>, (f64, f64), usize, f64, f64)> = vec![
            // Remove front only: 0 < 22 and 20 < 22
            (
                "removes front",
                vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                (22.0, 95.0),
                5,
                20.0,
                100.0,
            ),
            // Remove back only: 80 > 78 and 100 > 78
            (
                "removes back",
                vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                (5.0, 78.0),
                5,
                0.0,
                80.0,
            ),
            // Remove both ends
            (
                "removes both",
                vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                (22.0, 78.0),
                4,
                20.0,
                80.0,
            ),
            // No pruning needed - data spans valid bins
            (
                "no pruning needed",
                vec![0.0, 25.0, 50.0, 75.0, 100.0],
                (5.0, 95.0),
                5,
                0.0,
                100.0,
            ),
            // Multiple empty front bins
            (
                "multiple empty front",
                vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                (45.0, 78.0),
                5,
                40.0,
                80.0,
            ),
        ];

        for (desc, breaks, (data_min, data_max), expected_len, expected_first, expected_last) in
            test_cases
        {
            let mut breaks_arr: Vec<ArrayElement> =
                breaks.iter().map(|&v| ArrayElement::Number(v)).collect();
            let data_range = vec![
                ArrayElement::Number(data_min),
                ArrayElement::Number(data_max),
            ];

            super::prune_empty_edge_bins(&mut breaks_arr, &data_range);

            assert_eq!(
                breaks_arr.len(),
                expected_len,
                "{}: expected {} breaks, got {}",
                desc,
                expected_len,
                breaks_arr.len()
            );
            assert_eq!(
                breaks_arr[0],
                ArrayElement::Number(expected_first),
                "{}: first should be {}",
                desc,
                expected_first
            );
            assert_eq!(
                breaks_arr[breaks_arr.len() - 1],
                ArrayElement::Number(expected_last),
                "{}: last should be {}",
                desc,
                expected_last
            );
        }
    }

    #[test]
    fn test_prune_empty_edge_bins_edge_cases() {
        // Too few breaks (< 3)
        {
            let mut breaks = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
            let data_range = vec![ArrayElement::Number(50.0), ArrayElement::Number(60.0)];
            super::prune_empty_edge_bins(&mut breaks, &data_range);
            assert_eq!(breaks.len(), 2, "Should not prune with < 3 breaks");
        }

        // Exactly 3 breaks - bins contain data
        {
            let mut breaks = vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(50.0),
                ArrayElement::Number(100.0),
            ];
            let data_range = vec![ArrayElement::Number(10.0), ArrayElement::Number(90.0)];
            super::prune_empty_edge_bins(&mut breaks, &data_range);
            assert_eq!(breaks.len(), 3, "Should not prune when bins contain data");
        }
    }

    #[test]
    fn test_prune_empty_edge_bins_with_dates() {
        // Test with Date ArrayElements
        let test_cases = vec![
            // No change - bins contain data
            (
                vec![19720, 19735, 19750, 19765, 19780, 19795],
                (19730, 19780),
                6,
                19720,
                19795,
            ),
            // Prunes both ends
            (
                vec![19700, 19720, 19740, 19760, 19780, 19800],
                (19740, 19760),
                4,
                19720,
                19780,
            ),
        ];

        for (breaks, (data_min, data_max), expected_len, expected_first, expected_last) in
            test_cases
        {
            let mut breaks_arr: Vec<ArrayElement> =
                breaks.iter().map(|&v| ArrayElement::Date(v)).collect();
            let data_range = vec![ArrayElement::Date(data_min), ArrayElement::Date(data_max)];

            super::prune_empty_edge_bins(&mut breaks_arr, &data_range);

            assert_eq!(breaks_arr.len(), expected_len);
            assert_eq!(breaks_arr[0], ArrayElement::Date(expected_first));
            assert_eq!(
                breaks_arr[breaks_arr.len() - 1],
                ArrayElement::Date(expected_last)
            );
        }
    }

    // =========================================================================
    // Dtype Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dtype_accepts_numeric() {
        use super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let binned = Binned;
        assert!(binned.validate_dtype(&DataType::Int64).is_ok());
        assert!(binned.validate_dtype(&DataType::Float64).is_ok());
    }

    #[test]
    fn test_validate_dtype_accepts_temporal() {
        use super::ScaleTypeTrait;
        use polars::prelude::{DataType, TimeUnit};

        let binned = Binned;
        assert!(binned.validate_dtype(&DataType::Date).is_ok());
        assert!(binned
            .validate_dtype(&DataType::Datetime(TimeUnit::Microseconds, None))
            .is_ok());
    }

    #[test]
    fn test_validate_dtype_rejects_string() {
        use super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let binned = Binned;
        let result = binned.validate_dtype(&DataType::String);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("String"));
        assert!(err.contains("DISCRETE"));
    }

    #[test]
    fn test_validate_dtype_rejects_boolean() {
        use super::ScaleTypeTrait;
        use polars::prelude::DataType;

        let binned = Binned;
        let result = binned.validate_dtype(&DataType::Boolean);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Boolean"));
        assert!(err.contains("DISCRETE"));
    }
}
