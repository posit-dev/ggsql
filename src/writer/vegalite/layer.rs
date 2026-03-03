//! Geom rendering for Vega-Lite writer
//!
//! This module provides:
//! - Basic geom-to-mark mapping and column validation
//! - A trait-based approach to rendering different ggsql geom types to Vega-Lite specs
//!
//! Each geom type can override specific phases of the rendering pipeline while using
//! sensible defaults for standard behavior.

use crate::plot::layer::geom::GeomType;
use crate::plot::ParameterValue;
use crate::writer::vegalite::POINTS_TO_PIXELS;
use crate::{naming, AestheticValue, DataFrame, Geom, GgsqlError, Layer, Result};
use polars::prelude::ChunkCompareEq;
use serde_json::{json, Map, Value};
use std::any::Any;
use std::collections::HashMap;

use super::data::{dataframe_to_values, dataframe_to_values_with_bins};

// =============================================================================
// Basic Geom Utilities
// =============================================================================

/// Map ggsql Geom to Vega-Lite mark type
/// Always includes `clip: true` to ensure marks don't render outside plot bounds
pub fn geom_to_mark(geom: &Geom) -> Value {
    let mark_type = match geom.geom_type() {
        GeomType::Point => "point",
        GeomType::Line => "line",
        GeomType::Path => "line",
        GeomType::Bar => "bar",
        GeomType::Area => "area",
        GeomType::Tile => "rect",
        GeomType::Ribbon => "area",
        GeomType::Polygon => "line",
        GeomType::Histogram => "bar",
        GeomType::Density => "area",
        GeomType::Violin => "line",
        GeomType::Boxplot => "boxplot",
        GeomType::Text => "text",
        GeomType::Label => "text",
        _ => "point", // Default fallback
    };
    json!({
        "type": mark_type,
        "clip": true
    })
}

/// Validate column references for a single layer against its specific DataFrame
pub fn validate_layer_columns(layer: &Layer, data: &DataFrame, layer_idx: usize) -> Result<()> {
    let available_columns: Vec<String> = data
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    for (aesthetic, value) in &layer.mappings.aesthetics {
        if let AestheticValue::Column { name: col, .. } = value {
            if !available_columns.contains(col) {
                let source_desc = if let Some(src) = &layer.source {
                    format!(" (source: {})", src.as_str())
                } else {
                    " (global data)".to_string()
                };
                let display_col = naming::extract_aesthetic_name(col).unwrap_or(col.as_str());
                return Err(GgsqlError::ValidationError(format!(
                    "Column '{}' referenced in aesthetic '{}' (layer {}{}) does not exist.\nAvailable columns: {}",
                    display_col,
                    aesthetic,
                    layer_idx + 1,
                    source_desc,
                    available_columns.join(", ")
                )));
            }
        }
    }

    // Check partition_by columns
    for col in &layer.partition_by {
        if !available_columns.contains(col) {
            let source_desc = if let Some(src) = &layer.source {
                format!(" (source: {})", src.as_str())
            } else {
                " (global data)".to_string()
            };
            return Err(GgsqlError::ValidationError(format!(
                "Column '{}' referenced in PARTITION BY (layer {}{}) does not exist.\nAvailable columns: {}",
                col,
                layer_idx + 1,
                source_desc,
                available_columns.join(", ")
            )));
        }
    }

    Ok(())
}

// =============================================================================
// GeomRenderer Trait System
// =============================================================================

/// Data prepared for a layer - either single dataset or multiple components
pub enum PreparedData {
    /// Standard single dataset (most geoms)
    Single { values: Vec<Value> },
    /// Multiple component datasets (boxplot, violin, errorbar)
    Composite {
        components: HashMap<String, Vec<Value>>,
        metadata: Box<dyn Any + Send + Sync>,
    },
}

/// Trait for rendering ggsql geoms to Vega-Lite layers
///
/// Provides a three-phase rendering pipeline:
/// 1. **Data Preparation**: Convert DataFrame to JSON values
/// 2. **Encoding Modifications**: Apply geom-specific encoding transformations
/// 3. **Layer Output**: Finalize and potentially expand layers
///
/// Most geoms use the default implementations. Only geoms with special requirements
/// (bar width, path ordering, boxplot decomposition) need to override specific methods.
pub trait GeomRenderer: Send + Sync {
    // === Phase 1: Data Preparation ===

    /// Prepare data for this layer.
    /// Default: convert DataFrame to JSON values (single dataset)
    fn prepare_data(
        &self,
        df: &DataFrame,
        _layer: &Layer,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        let values = if binned_columns.is_empty() {
            dataframe_to_values(df)?
        } else {
            dataframe_to_values_with_bins(df, binned_columns)?
        };
        Ok(PreparedData::Single { values })
    }

    // === Phase 2: Encoding Modifications ===

    /// Modify the encoding map for this geom.
    /// Default: no modifications
    fn modify_encoding(&self, _encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        Ok(())
    }

    /// Modify the mark/layer spec for this geom.
    /// Default: no modifications
    fn modify_spec(&self, _layer_spec: &mut Value, _layer: &Layer) -> Result<()> {
        Ok(())
    }

    // === Phase 3: Layer Output ===

    /// Whether to add the standard source filter transform.
    /// Default: true (composite geoms override to return false)
    fn needs_source_filter(&self) -> bool {
        true
    }

    /// Finalize the layer(s) for output.
    /// Default: return single layer unchanged
    /// Composite geoms override to expand into multiple layers
    fn finalize(
        &self,
        layer_spec: Value,
        _layer: &Layer,
        _data_key: &str,
        _prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        Ok(vec![layer_spec])
    }
}

// =============================================================================
// Default Renderer (for geoms with no special handling)
// =============================================================================

/// Default renderer used for geoms with standard behavior
pub struct DefaultRenderer;

impl GeomRenderer for DefaultRenderer {}

// =============================================================================
// Bar Renderer
// =============================================================================

/// Renderer for bar geom - overrides mark spec for band width
pub struct BarRenderer;

impl GeomRenderer for BarRenderer {
    fn modify_spec(&self, layer_spec: &mut Value, layer: &Layer) -> Result<()> {
        let width = match layer.parameters.get("width") {
            Some(ParameterValue::Number(w)) => *w,
            _ => 0.9,
        };
        layer_spec["mark"] = json!({
            "type": "bar",
            "width": {"band": width},
            "clip": true
        });
        Ok(())
    }
}

// =============================================================================
// Path Renderer
// =============================================================================

/// Renderer for path geom - adds order channel for natural data order
pub struct PathRenderer;

impl GeomRenderer for PathRenderer {
    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        // Use the natural data order
        encoding.insert("order".to_string(), json!({"value": Value::Null}));
        Ok(())
    }
}

// =============================================================================
// Text Renderer
// =============================================================================

/// Renderer for text geom - handles font properties via data splitting
pub struct TextRenderer;

impl TextRenderer {
    /// Apply label formatting if format parameter is specified.
    /// Returns a new DataFrame with the label column formatted, or the original if no formatting.
    fn apply_label_formatting(df: &DataFrame, layer: &Layer) -> Result<DataFrame> {
        use crate::format;
        use crate::naming;
        use crate::plot::ParameterValue;

        // Check if format parameter is specified
        let format_template = match layer.parameters.get("format") {
            Some(ParameterValue::String(template)) => template,
            _ => return Ok(df.clone()), // No formatting, return original
        };

        // Use format.rs helper to do the formatting
        let label_col_name = naming::aesthetic_column("label");
        format::format_dataframe_column(df, &label_col_name, format_template)
            .map_err(GgsqlError::WriterError)
    }

    /// Analyze DataFrame columns to build font property runs using run-length encoding.
    /// Returns:
    /// - DataFrame where each row represents a run's font properties (family, fontface, hjust, vjust, angle)
    /// - Vec<usize> of run lengths corresponding to each row
    fn build_font_rle(df: &DataFrame) -> Result<(DataFrame, Vec<usize>)> {
        use polars::prelude::*;

        let nrows = df.height();

        if nrows == 0 {
            // Return empty DataFrame and empty run lengths
            return Ok((DataFrame::default(), Vec::new()));
        }

        // Build boolean mask showing where any font property changes
        let mut changed = BooleanChunked::full("changed".into(), false, nrows);
        let mut font_columns: HashMap<&str, &polars::prelude::Column> = HashMap::new();

        for aesthetic in ["family", "fontface", "hjust", "vjust", "angle"] {
            if let Ok(col) = df.column(&naming::aesthetic_column(aesthetic)) {
                let col_changed = col.not_equal(&col.shift(1)).map_err(|e| {
                    GgsqlError::InternalError(format!("Failed to compare column: {}", e))
                })?;
                changed = &changed | &col_changed;
                font_columns.insert(aesthetic, col);
            }
        }

        // Extract change indices (where mask is true)
        // shift() creates nulls at position 0, which we treat as a change point
        let mut change_indices: Vec<usize> = Vec::new();
        for (i, val) in changed.iter().enumerate() {
            if val == Some(true) || val == None {
                // Treat null (from shift) or true as change point
                change_indices.push(i);
            }
        }

        // First row is always a change point (shift comparison is null)
        if !change_indices.is_empty() && change_indices[0] != 0 {
            change_indices.insert(0, 0);
        } else if change_indices.is_empty() {
            change_indices.push(0);
        }

        // Calculate run lengths
        let run_lengths: Vec<usize> = change_indices
            .iter()
            .enumerate()
            .map(|(i, &start)| {
                let end = change_indices.get(i + 1).copied().unwrap_or(nrows);
                end - start
            })
            .collect();

        // Extract rows at change indices (only font columns)
        let indices_ca = UInt32Chunked::from_vec(
            "indices".into(),
            change_indices.iter().map(|&i| i as u32).collect(),
        );
        let font_aesthetics = ["family", "fontface", "hjust", "vjust", "angle"];

        let mut result_cols = Vec::new();
        for aesthetic in font_aesthetics {
            if let Some(col) = font_columns.get(aesthetic) {
                let taken = col.take(&indices_ca).map_err(|e| {
                    GgsqlError::InternalError(format!(
                        "Failed to take indices from {}: {}",
                        aesthetic, e
                    ))
                })?;
                result_cols.push(taken);
            }
        }

        // Create result DataFrame (only font properties, no run_length column)
        let result_df = DataFrame::new(result_cols).map_err(|e| {
            GgsqlError::InternalError(format!("Failed to create run DataFrame: {}", e))
        })?;

        Ok((result_df, run_lengths))
    }

    /// Convert family to Vega-Lite font value
    /// Prefers literal over column value
    fn convert_family(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> Option<Value> {
        // First select which value to use (prefer literal)
        let value = if let Some(ParameterValue::String(s)) = literal {
            s.as_str()
        } else {
            column_value?
        };

        // Then apply conversion
        if !value.is_empty() {
            Some(json!(value))
        } else {
            None
        }
    }

    /// Convert fontface to Vega-Lite fontWeight and fontStyle values
    /// Prefers literal over column value
    fn convert_fontface(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> (Option<Value>, Option<Value>) {
        // First select which value to use (prefer literal)
        let value = if let Some(ParameterValue::String(s)) = literal {
            s.as_str()
        } else if let Some(s) = column_value {
            s
        } else {
            return (None, None);
        };

        // Then apply conversion
        let (weight, style) = match value {
            "bold" => (json!("bold"), json!("normal")),
            "italic" => (json!("normal"), json!("italic")),
            "bold.italic" | "bolditalic" => (json!("bold"), json!("italic")),
            _ => (json!("normal"), json!("normal")),
        };
        (Some(weight), Some(style))
    }

    /// Convert hjust to Vega-Lite align value
    /// Prefers literal over column value
    fn convert_hjust(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> Option<Value> {
        // First extract which value to use (prefer literal)
        let value_str = match literal {
            Some(ParameterValue::String(s)) => s.to_string(),
            Some(ParameterValue::Number(n)) => n.to_string(),
            _ => column_value?.to_string(),
        };

        // Then apply conversion inline
        let align = match value_str.parse::<f64>() {
            Ok(v) if v <= 0.25 => "left",
            Ok(v) if v >= 0.75 => "right",
            _ => match value_str.as_str() {
                "left" => "left",
                "right" => "right",
                _ => "center",
            },
        };

        Some(json!(align))
    }

    /// Convert vjust to Vega-Lite baseline value
    /// Prefers literal over column value
    fn convert_vjust(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> Option<Value> {
        // First extract which value to use (prefer literal)
        let value_str = match literal {
            Some(ParameterValue::String(s)) => s.to_string(),
            Some(ParameterValue::Number(n)) => n.to_string(),
            _ => column_value?.to_string(),
        };

        // Then apply conversion inline
        let baseline = match value_str.parse::<f64>() {
            Ok(v) if v <= 0.25 => "bottom",
            Ok(v) if v >= 0.75 => "top",
            _ => match value_str.as_str() {
                "top" => "top",
                "bottom" => "bottom",
                _ => "middle",
            },
        };

        Some(json!(baseline))
    }

    /// Convert angle to Vega-Lite angle value (degrees)
    /// Prefers literal over column value
    /// Normalizes angles to [0, 360) range
    fn convert_angle(literal: Option<&ParameterValue>, column_value: Option<f64>) -> Option<Value> {
        // First select which value to use (prefer literal)
        let value = if let Some(ParameterValue::Number(n)) = literal {
            *n
        } else {
            column_value?
        };

        // Then apply conversion inline
        let normalized = value % 360.0;
        let angle = if normalized < 0.0 {
            normalized + 360.0
        } else {
            normalized
        };

        Some(json!(angle))
    }

    /// Apply font properties to mark object from DataFrame row and layer literals
    /// Uses literals from layer parameters if present, otherwise uses DataFrame column values
    fn apply_font_properties(
        mark_obj: &mut Map<String, Value>,
        df: &DataFrame,
        row_idx: usize,
        layer: &Layer,
    ) -> Result<()> {
        // Helper to extract string column values using aesthetic column naming
        let get_str = |aesthetic: &str| -> Option<String> {
            let col_name = naming::aesthetic_column(aesthetic);
            df.column(&col_name)
                .ok()
                .and_then(|col| col.str().ok())
                .and_then(|ca| ca.get(row_idx))
                .map(|s| s.to_string())
        };

        // Helper to extract numeric column values (for angle)
        let get_f64 = |aesthetic: &str| -> Option<f64> {
            use polars::prelude::*;
            let col_name = naming::aesthetic_column(aesthetic);
            let col = df.column(&col_name).ok()?;

            // Try as string first (for string-encoded numbers)
            if let Ok(ca) = col.str() {
                return ca.get(row_idx).and_then(|s| s.parse::<f64>().ok());
            }

            // Try as numeric types directly
            if let Ok(casted) = col.cast(&DataType::Float64) {
                if let Ok(ca) = casted.f64() {
                    return ca.get(row_idx);
                }
            }

            None
        };

        // Convert and apply font properties
        if let Some(family_val) =
            Self::convert_family(layer.get_literal("family"), get_str("family").as_deref())
        {
            mark_obj.insert("font".to_string(), family_val);
        }

        let (font_weight_val, font_style_val) = Self::convert_fontface(
            layer.get_literal("fontface"),
            get_str("fontface").as_deref(),
        );
        if let Some(weight) = font_weight_val {
            mark_obj.insert("fontWeight".to_string(), weight);
        }
        if let Some(style) = font_style_val {
            mark_obj.insert("fontStyle".to_string(), style);
        }

        if let Some(hjust_val) =
            Self::convert_hjust(layer.get_literal("hjust"), get_str("hjust").as_deref())
        {
            mark_obj.insert("align".to_string(), hjust_val);
        }

        if let Some(vjust_val) =
            Self::convert_vjust(layer.get_literal("vjust"), get_str("vjust").as_deref())
        {
            mark_obj.insert("baseline".to_string(), vjust_val);
        }

        if let Some(angle_val) = Self::convert_angle(layer.get_literal("angle"), get_f64("angle")) {
            mark_obj.insert("angle".to_string(), angle_val);
        }

        Ok(())
    }

    /// Build transform with source filter
    fn build_transform_with_filter(prototype: &Value, source_key: &str) -> Vec<Value> {
        let source_filter = json!({
            "filter": {
                "field": naming::SOURCE_COLUMN,
                "equal": source_key
            }
        });

        let existing_transforms = prototype
            .get("transform")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let mut new_transforms = vec![source_filter];
        new_transforms.extend(existing_transforms);
        new_transforms
    }

    /// Finalize layers as nested layer with shared encoding (works for single or multiple runs)
    fn finalize_nested_layers(
        &self,
        prototype: Value,
        data_key: &str,
        font_runs_df: &DataFrame,
        run_lengths: &[usize],
        layer: &Layer,
    ) -> Result<Vec<Value>> {
        // Extract shared encoding from prototype
        let shared_encoding = prototype.get("encoding").cloned();

        // Build base mark object with fixed parameters
        let mut base_mark = json!({"type": "text"});
        if let Some(mark_map) = base_mark.as_object_mut() {
            // Extract nudge parameters (nudge_x → xOffset, nudge_y → yOffset)
            if let Some(ParameterValue::Number(x_offset)) = layer.parameters.get("nudge_x") {
                mark_map.insert("xOffset".to_string(), json!(x_offset * POINTS_TO_PIXELS));
            }
            if let Some(ParameterValue::Number(y_offset)) = layer.parameters.get("nudge_y") {
                mark_map.insert("yOffset".to_string(), json!(-y_offset * POINTS_TO_PIXELS));
            }
        }

        // Build individual layers without encoding (mark + transform only)
        // Use run_lengths to get number of runs (works even when no font columns exist)
        let nruns = run_lengths.len();
        let mut nested_layers: Vec<Value> = Vec::with_capacity(nruns);

        for run_idx in 0..nruns {
            let suffix = format!("_font_{}", run_idx);
            let source_key = format!("{}{}", data_key, suffix);

            // Clone base mark and apply font-specific properties
            let mut mark_obj = base_mark.clone();
            if let Some(mark_map) = mark_obj.as_object_mut() {
                Self::apply_font_properties(mark_map, font_runs_df, run_idx, layer)?;
            }

            // Create layer with mark and transform (no encoding)
            nested_layers.push(json!({
                "mark": mark_obj,
                "transform": Self::build_transform_with_filter(&prototype, &source_key)
            }));
        }

        // Wrap in parent spec with shared encoding
        let mut parent_spec = json!({"layer": nested_layers});

        if let Some(encoding) = shared_encoding {
            parent_spec["encoding"] = encoding;
        }

        Ok(vec![parent_spec])
    }
}

impl GeomRenderer for TextRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        layer: &Layer,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        // Apply label formatting if specified
        let df = Self::apply_label_formatting(df, layer)?;

        // Analyze font columns to get RLE runs
        let (font_runs_df, run_lengths) = Self::build_font_rle(&df)?;

        // Split data by font runs, tracking cumulative position
        let mut components: HashMap<String, Vec<Value>> = HashMap::new();
        let mut position = 0;

        for (run_idx, &length) in run_lengths.iter().enumerate() {
            let suffix = format!("_font_{}", run_idx);

            // Slice the contiguous run from the DataFrame (more efficient than boolean masking)
            let sliced = df.slice(position as i64, length);

            let values = if binned_columns.is_empty() {
                dataframe_to_values(&sliced)?
            } else {
                dataframe_to_values_with_bins(&sliced, binned_columns)?
            };

            components.insert(suffix, values);
            position += length;
        }

        Ok(PreparedData::Composite {
            components,
            metadata: Box::new((font_runs_df, run_lengths)),
        })
    }

    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        // Remove font aesthetics from encoding - they only work as mark properties
        for &aesthetic in &["family", "fontface", "hjust", "vjust", "angle"] {
            encoding.remove(aesthetic);
        }

        // Suppress legend and scale for text encoding
        if let Some(text_encoding) = encoding.get_mut("text") {
            if let Some(text_obj) = text_encoding.as_object_mut() {
                text_obj.insert("legend".to_string(), Value::Null);
                text_obj.insert("scale".to_string(), Value::Null);
            }
        }

        Ok(())
    }

    fn needs_source_filter(&self) -> bool {
        // TextRenderer handles source filtering in finalize()
        false
    }

    fn finalize(
        &self,
        prototype: Value,
        layer: &Layer,
        data_key: &str,
        prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        let PreparedData::Composite { metadata, .. } = prepared else {
            return Err(GgsqlError::InternalError(
                "TextRenderer::finalize called with non-composite data".to_string(),
            ));
        };

        // Downcast metadata to font runs
        let (font_runs_df, run_lengths) = metadata
            .downcast_ref::<(DataFrame, Vec<usize>)>()
            .ok_or_else(|| GgsqlError::InternalError("Failed to downcast font runs".to_string()))?;

        // Generate nested layers from font runs (works for single or multiple runs)
        self.finalize_nested_layers(prototype, data_key, font_runs_df, run_lengths, layer)
    }
}

// =============================================================================
// Ribbon Renderer
// =============================================================================

/// Renderer for ribbon geom - remaps ymin/ymax to y/y2
pub struct RibbonRenderer;

impl GeomRenderer for RibbonRenderer {
    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        if let Some(ymax) = encoding.remove("ymax") {
            encoding.insert("y".to_string(), ymax);
        }
        if let Some(ymin) = encoding.remove("ymin") {
            encoding.insert("y2".to_string(), ymin);
        }
        Ok(())
    }
}

// =============================================================================
// Area Renderer
// =============================================================================

/// Renderer for area geom - handles stacking options
pub struct AreaRenderer;

impl GeomRenderer for AreaRenderer {
    fn modify_encoding(&self, encoding: &mut Map<String, Value>, layer: &Layer) -> Result<()> {
        if let Some(mut y) = encoding.remove("y") {
            let stack_value;
            if let Some(ParameterValue::String(stack)) = layer.parameters.get("stacking") {
                stack_value = match stack.as_str() {
                    "on" => json!("zero"),
                    "off" => Value::Null,
                    "fill" => json!("normalize"),
                    _ => {
                        return Err(GgsqlError::ValidationError(format!(
                            "Area layer's `stacking` must be \"on\", \"off\" or \"fill\", not \"{}\"",
                            stack
                        )));
                    }
                }
            } else {
                stack_value = Value::Null
            }
            y["stack"] = stack_value;
            encoding.insert("y".to_string(), y);
        }
        Ok(())
    }
}

// =============================================================================
// Polygon Renderer
// =============================================================================

/// Renderer for polygon geom - uses closed line with fill
pub struct PolygonRenderer;

impl GeomRenderer for PolygonRenderer {
    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        // Polygon needs both `fill` and `stroke` independently, but map_aesthetic_name()
        // converts fill -> color (which works for most geoms). For closed line marks,
        // we need actual `fill` and `stroke` channels, so we undo the mapping here.
        if let Some(color) = encoding.remove("color") {
            encoding.insert("fill".to_string(), color);
        }
        // Use the natural data order
        encoding.insert("order".to_string(), json!({"value": Value::Null}));
        Ok(())
    }

    fn modify_spec(&self, layer_spec: &mut Value, _layer: &Layer) -> Result<()> {
        layer_spec["mark"] = json!({
            "type": "line",
            "interpolate": "linear-closed"
        });
        Ok(())
    }
}

// =============================================================================
// Violin Renderer
// =============================================================================

/// Renderer for violin geom - uses line
pub struct ViolinRenderer;

impl GeomRenderer for ViolinRenderer {
    fn modify_spec(&self, layer_spec: &mut Value, _layer: &Layer) -> Result<()> {
        layer_spec["mark"] = json!({
            "type": "line",
            "filled": true
        });
        let offset_col = naming::aesthetic_column("offset");

        // Mirror the density on both sides.
        // It'll be implemented as an offset.
        let violin_offset = format!("[datum.{offset}, -datum.{offset}]", offset = offset_col);

        // We use an order calculation to create a proper closed shape.
        // Right side (+ offset), sort by -y (top -> bottom)
        // Left side (- offset), sort by +y (bottom -> top)
        let calc_order = format!(
            "datum.__violin_offset > 0 ? -datum.{y} : datum.{y}",
            y = naming::aesthetic_column("pos2")
        );

        // Filter threshold to trim very low density regions (removes thin tails)
        // In theory, this depends on the grid resolution and might be better
        // handled upstream, but for now it seems not unreasonable.
        let filter_expr = format!("datum.{} > 0.001", offset_col);

        // Preserve existing transforms (e.g., source filter) and extend with violin-specific transforms
        let existing_transforms = layer_spec
            .get("transform")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let mut transforms = existing_transforms;
        transforms.extend(vec![
            json!({
                // Remove points with very low density to clean up thin tails
                "filter": filter_expr
            }),
            json!({
                "calculate": violin_offset,
                "as": "violin_offsets"
            }),
            json!({
                "flatten": ["violin_offsets"],
                "as": ["__violin_offset"]
            }),
            json!({
                "calculate": calc_order,
                "as": "__order"
            }),
        ]);

        layer_spec["transform"] = json!(transforms);
        Ok(())
    }

    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        // Ensure x is in detail encoding to create separate violins per x category
        // This is needed because line marks with filled:true require detail to create separate paths
        let x_field = encoding
            .get("x")
            .and_then(|x| x.get("field"))
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        if let Some(x_field) = x_field {
            match encoding.get_mut("detail") {
                Some(detail) if detail.is_object() => {
                    // Single field object - check if it's already x, otherwise convert to array
                    if detail.get("field").and_then(|f| f.as_str()) != Some(&x_field) {
                        let existing = detail.clone();
                        *detail = json!([existing, {"field": x_field, "type": "nominal"}]);
                    }
                }
                Some(detail) if detail.is_array() => {
                    // Array - check if x already present, add if not
                    let arr = detail.as_array_mut().unwrap();
                    let has_x = arr
                        .iter()
                        .any(|d| d.get("field").and_then(|f| f.as_str()) == Some(&x_field));
                    if !has_x {
                        arr.push(json!({"field": x_field, "type": "nominal"}));
                    }
                }
                None => {
                    // No detail encoding - add it with x field
                    encoding.insert(
                        "detail".to_string(),
                        json!({"field": x_field, "type": "nominal"}),
                    );
                }
                _ => {}
            }
        }

        // Violins use filled line marks, which don't show a fill in the legend.
        // We intercept the encoding to pupulate a different symbol to display
        for aesthetic in ["fill", "stroke"] {
            if let Some(channel) = encoding.get_mut(aesthetic) {
                // Skip if legend is explicitly null or if it's a literal value
                if channel.get("legend").is_some_and(|v| v.is_null()) {
                    continue;
                }
                if channel.get("value").is_some() {
                    continue;
                }

                // Add/update legend properties
                let legend = channel.get_mut("legend").and_then(|v| v.as_object_mut());
                if let Some(legend_map) = legend {
                    legend_map.insert("symbolType".to_string(), json!("circle"));
                } else {
                    channel["legend"] = json!({
                        "symbolType": "circle"
                    });
                }
            }
        }

        encoding.insert(
            "xOffset".to_string(),
            json!({
                "field": "__violin_offset",
                "type": "quantitative"
            }),
        );
        encoding.insert(
            "order".to_string(),
            json!({
                "field": "__order",
                "type": "quantitative"
            }),
        );
        Ok(())
    }
}

// =============================================================================
// Boxplot Renderer
// =============================================================================

/// Metadata for boxplot rendering
struct BoxplotMetadata {
    /// Grouping column names
    grouping_cols: Vec<String>,
    /// Whether there are any outliers
    has_outliers: bool,
}

/// Renderer for boxplot geom - splits into multiple component layers
pub struct BoxplotRenderer;

impl BoxplotRenderer {
    /// Prepare boxplot data by splitting into type-specific datasets.
    ///
    /// Returns a HashMap of type_suffix -> data_values, plus grouping_cols and has_outliers.
    /// Type suffixes are: "lower_whisker", "upper_whisker", "box", "median", "outlier"
    #[allow(clippy::type_complexity)]
    fn prepare_components(
        &self,
        data: &DataFrame,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<(HashMap<String, Vec<Value>>, Vec<String>, bool)> {
        let type_col = naming::aesthetic_column("type");
        let type_col = type_col.as_str();
        let value_col = naming::aesthetic_column("pos2");
        let value_col = value_col.as_str();
        let value2_col = naming::aesthetic_column("pos2end");
        let value2_col = value2_col.as_str();

        // Find grouping columns (all columns except type, value, value2)
        let grouping_cols: Vec<String> = data
            .get_column_names()
            .iter()
            .filter(|&col| {
                col.as_str() != type_col && col.as_str() != value_col && col.as_str() != value2_col
            })
            .map(|s| s.to_string())
            .collect();

        // Get the type column for filtering
        let type_series = data
            .column(type_col)
            .and_then(|s| s.str())
            .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

        // Check for outliers
        let has_outliers = type_series.equal("outlier").any();

        // Split data by type into separate datasets
        let mut type_datasets: HashMap<String, Vec<Value>> = HashMap::new();

        for type_name in &["lower_whisker", "upper_whisker", "box", "median", "outlier"] {
            let mask = type_series.equal(*type_name);
            let filtered = data
                .filter(&mask)
                .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

            // Skip empty datasets (e.g., no outliers)
            if filtered.height() == 0 {
                continue;
            }

            // Drop the type column since type is now encoded in the source key
            let filtered = filtered
                .drop(type_col)
                .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

            let values = if binned_columns.is_empty() {
                dataframe_to_values(&filtered)?
            } else {
                dataframe_to_values_with_bins(&filtered, binned_columns)?
            };

            type_datasets.insert(type_name.to_string(), values);
        }

        Ok((type_datasets, grouping_cols, has_outliers))
    }

    /// Render boxplot layers using filter transforms on the unified dataset.
    ///
    /// Creates 5 layers: outliers (optional), lower whiskers, upper whiskers, box, median line.
    fn render_layers(
        &self,
        prototype: Value,
        layer: &Layer,
        base_key: &str,
        grouping_cols: &[String],
        has_outliers: bool,
    ) -> Result<Vec<Value>> {
        let mut layers: Vec<Value> = Vec::new();

        let value_col = naming::aesthetic_column("pos2");
        let value2_col = naming::aesthetic_column("pos2end");

        let x_col = layer
            .mappings
            .get("pos1")
            .and_then(|x| x.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Boxplot requires 'x' aesthetic mapping".to_string())
            })?;
        let y_col = layer
            .mappings
            .get("pos2")
            .and_then(|y| y.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Boxplot requires 'y' aesthetic mapping".to_string())
            })?;

        // Set orientation
        let is_horizontal = x_col == value_col;
        let group_col = if is_horizontal { y_col } else { x_col };
        let offset = if is_horizontal { "yOffset" } else { "xOffset" };
        let value_var1 = if is_horizontal { "x" } else { "y" };
        let value_var2 = if is_horizontal { "x2" } else { "y2" };

        // Find dodge groups (grouping cols minus the axis group col)
        let dodge_groups: Vec<&str> = grouping_cols
            .iter()
            .filter(|col| col.as_str() != group_col)
            .map(|s| s.as_str())
            .collect();

        // Get width parameter
        let mut width = 0.9;
        if let Some(ParameterValue::Number(num)) = layer.parameters.get("width") {
            width = *num;
        }

        // Helper to create filter transform for source selection
        let make_source_filter = |type_suffix: &str| -> Value {
            let source_key = format!("{}{}", base_key, type_suffix);
            json!({
                "filter": {
                    "field": naming::SOURCE_COLUMN,
                    "equal": source_key
                }
            })
        };

        // Helper to create a layer with source filter and mark
        let create_layer = |proto: &Value, type_suffix: &str, mark: Value| -> Value {
            let mut layer_spec = proto.clone();
            let existing_transforms = layer_spec
                .get("transform")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default();
            let mut new_transforms = vec![make_source_filter(type_suffix)];
            new_transforms.extend(existing_transforms);
            layer_spec["transform"] = json!(new_transforms);
            layer_spec["mark"] = mark;
            layer_spec
        };

        // Create outlier points layer (if there are outliers)
        if has_outliers {
            let mut points = create_layer(
                &prototype,
                "outlier",
                json!({
                    "type": "point"
                }),
            );
            if points["encoding"].get("color").is_some() {
                points["mark"]["filled"] = json!(true);
            }

            // Add dodging offset
            if !dodge_groups.is_empty() {
                points["encoding"][offset] = json!({"field": dodge_groups[0]});
            }

            layers.push(points);
        }

        // Clone prototype without size/shape (these apply only to points)
        let mut summary_prototype = prototype.clone();
        if let Some(Value::Object(ref mut encoding)) = summary_prototype.get_mut("encoding") {
            encoding.remove("size");
            encoding.remove("shape");
        }

        // Build encoding templates for y and y2 fields
        let mut y_encoding = summary_prototype["encoding"][value_var1].clone();
        y_encoding["field"] = json!(value_col);
        let mut y2_encoding = summary_prototype["encoding"][value_var1].clone();
        y2_encoding["field"] = json!(value2_col);
        y2_encoding["title"] = Value::Null; // Suppress y2 title to prevent "y, y2" axis label

        // Lower whiskers (rule from y to y2, where y=q1 and y2=lower)
        let mut lower_whiskers = create_layer(
            &summary_prototype,
            "lower_whisker",
            json!({
                "type": "rule"
            }),
        );

        // Handle strokeWidth -> size for rule marks
        if let Some(linewidth) = lower_whiskers["encoding"].get("strokeWidth").cloned() {
            lower_whiskers["encoding"]["size"] = linewidth;
            if let Some(Value::Object(ref mut encoding)) = lower_whiskers.get_mut("encoding") {
                encoding.remove("strokeWidth");
            }
        }

        lower_whiskers["encoding"][value_var1] = y_encoding.clone();
        lower_whiskers["encoding"][value_var2] = y2_encoding.clone();

        // Upper whiskers (rule from y to y2, where y=q3 and y2=upper)
        let mut upper_whiskers = create_layer(
            &summary_prototype,
            "upper_whisker",
            json!({
                "type": "rule"
            }),
        );

        // Handle strokeWidth -> size for rule marks
        if let Some(linewidth) = upper_whiskers["encoding"].get("strokeWidth").cloned() {
            upper_whiskers["encoding"]["size"] = linewidth;
            if let Some(Value::Object(ref mut encoding)) = upper_whiskers.get_mut("encoding") {
                encoding.remove("strokeWidth");
            }
        }

        upper_whiskers["encoding"][value_var1] = y_encoding.clone();
        upper_whiskers["encoding"][value_var2] = y2_encoding.clone();

        // Box (bar from y to y2, where y=q1 and y2=q3)
        let mut box_part = create_layer(
            &summary_prototype,
            "box",
            json!({
                "type": "bar",
                "width": {"band": width},
                "align": "center"
            }),
        );
        box_part["encoding"][value_var1] = y_encoding.clone();
        box_part["encoding"][value_var2] = y2_encoding.clone();

        // Median line (tick at y, where y=median)
        let mut median_line = create_layer(
            &summary_prototype,
            "median",
            json!({
                "type": "tick",
                "width": {"band": width},
                "align": "center"
            }),
        );
        median_line["encoding"][value_var1] = y_encoding;

        // Add dodging to all summary layers
        if !dodge_groups.is_empty() {
            let offset_val = json!({"field": dodge_groups[0]});
            lower_whiskers["encoding"][offset] = offset_val.clone();
            upper_whiskers["encoding"][offset] = offset_val.clone();
            box_part["encoding"][offset] = offset_val.clone();
            median_line["encoding"][offset] = offset_val;
        }

        layers.push(lower_whiskers);
        layers.push(upper_whiskers);
        layers.push(box_part);
        layers.push(median_line);

        Ok(layers)
    }
}

impl GeomRenderer for BoxplotRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        _layer: &Layer,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        let (components, grouping_cols, has_outliers) =
            self.prepare_components(df, binned_columns)?;

        Ok(PreparedData::Composite {
            components,
            metadata: Box::new(BoxplotMetadata {
                grouping_cols,
                has_outliers,
            }),
        })
    }

    fn needs_source_filter(&self) -> bool {
        // Boxplot uses component-specific filters instead
        false
    }

    fn finalize(
        &self,
        prototype: Value,
        layer: &Layer,
        data_key: &str,
        prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        let PreparedData::Composite { metadata, .. } = prepared else {
            return Err(GgsqlError::InternalError(
                "BoxplotRenderer::finalize called with non-composite data".to_string(),
            ));
        };

        let info = metadata.downcast_ref::<BoxplotMetadata>().ok_or_else(|| {
            GgsqlError::InternalError("Failed to downcast boxplot metadata".to_string())
        })?;

        self.render_layers(
            prototype,
            layer,
            data_key,
            &info.grouping_cols,
            info.has_outliers,
        )
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

/// Get the appropriate renderer for a geom type
pub fn get_renderer(geom: &Geom) -> Box<dyn GeomRenderer> {
    match geom.geom_type() {
        GeomType::Path => Box::new(PathRenderer),
        GeomType::Bar => Box::new(BarRenderer),
        GeomType::Area => Box::new(AreaRenderer),
        GeomType::Ribbon => Box::new(RibbonRenderer),
        GeomType::Polygon => Box::new(PolygonRenderer),
        GeomType::Boxplot => Box::new(BoxplotRenderer),
        GeomType::Density => Box::new(AreaRenderer),
        GeomType::Violin => Box::new(ViolinRenderer),
        GeomType::Text => Box::new(TextRenderer),
        // All other geoms (Point, Line, Tile, etc.) use the default renderer
        _ => Box::new(DefaultRenderer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_violin_detail_encoding() {
        let renderer = ViolinRenderer;
        let layer = Layer::new(crate::plot::Geom::violin());

        // Case 1: No detail encoding - should add x
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        renderer.modify_encoding(&mut encoding, &layer).unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!({"field": "species", "type": "nominal"}))
        );

        // Case 2: Detail is single object (not x) - should convert to array
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "detail".to_string(),
            json!({"field": "island", "type": "nominal"}),
        );
        renderer.modify_encoding(&mut encoding, &layer).unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!([
                {"field": "island", "type": "nominal"},
                {"field": "species", "type": "nominal"}
            ]))
        );

        // Case 3: Detail is single object (already x) - should not change
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "detail".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        renderer.modify_encoding(&mut encoding, &layer).unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!({"field": "species", "type": "nominal"}))
        );

        // Case 4: Detail is array without x - should add x
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "detail".to_string(),
            json!([{"field": "island", "type": "nominal"}]),
        );
        renderer.modify_encoding(&mut encoding, &layer).unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!([
                {"field": "island", "type": "nominal"},
                {"field": "species", "type": "nominal"}
            ]))
        );

        // Case 5: Detail is array with x already - should not change
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "detail".to_string(),
            json!([
                {"field": "island", "type": "nominal"},
                {"field": "species", "type": "nominal"}
            ]),
        );
        renderer.modify_encoding(&mut encoding, &layer).unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!([
                {"field": "island", "type": "nominal"},
                {"field": "species", "type": "nominal"}
            ]))
        );
    }

    #[test]
    fn test_text_constant_font() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;
        let layer = Layer::new(crate::plot::Geom::text());

        // Create DataFrame where all rows have the same font
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("family").as_str() => &["Arial", "Arial", "Arial"],
        }
        .unwrap();

        // Prepare data - should result in single layer with _font_0 component key
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        match prepared {
            PreparedData::Composite { components, .. } => {
                // Should have single component with _font_0 key
                assert_eq!(components.len(), 1);
                assert!(components.contains_key("_font_0"));
            }
            _ => panic!("Expected Composite"),
        }
    }

    #[test]
    fn test_text_varying_font() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;
        let layer = Layer::new(crate::plot::Geom::text());

        // Create DataFrame with different fonts per row
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("family").as_str() => &["Arial", "Courier", "Times"],
        }
        .unwrap();

        // Prepare data - should result in multiple layers
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        match prepared {
            PreparedData::Composite { components, .. } => {
                // Should have 3 components (one per unique font) with suffix keys
                assert_eq!(components.len(), 3);
                assert!(components.contains_key("_font_0"));
                assert!(components.contains_key("_font_1"));
                assert!(components.contains_key("_font_2"));
            }
            _ => panic!("Expected Composite"),
        }
    }

    #[test]
    fn test_text_nested_layers_structure() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;
        let layer = Layer::new(crate::plot::Geom::text());

        // Create DataFrame with different fonts
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("family").as_str() => &["Arial", "Courier", "Arial"],
            naming::aesthetic_column("fontface").as_str() => &["bold", "italic", "bold"],
        }
        .unwrap();

        // Prepare data
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        // Get the components
        let components = match &prepared {
            PreparedData::Composite { components, .. } => components,
            _ => panic!("Expected Composite"),
        };

        // Should have 3 components due to non-contiguous indices
        // (Arial+bold at index 0, Courier+italic at index 1, Arial+bold at index 2)
        assert_eq!(components.len(), 3);

        // Build prototype spec
        let prototype = json!({
            "mark": {"type": "text"},
            "encoding": {
                "x": {"field": naming::aesthetic_column("x"), "type": "quantitative"},
                "y": {"field": naming::aesthetic_column("y"), "type": "quantitative"},
                "text": {"field": naming::aesthetic_column("label"), "type": "nominal"}
            }
        });

        // Create a dummy layer
        let layer = crate::plot::Layer::new(crate::plot::Geom::text());

        // Call finalize to get layers
        let layers = renderer
            .finalize(prototype.clone(), &layer, "test", &prepared)
            .unwrap();

        // For multiple font groups, should return single parent spec with nested layers
        assert_eq!(layers.len(), 1);

        let parent_spec = &layers[0];

        // Parent should have "layer" array
        assert!(parent_spec.get("layer").is_some());
        let nested_layers = parent_spec["layer"].as_array().unwrap();

        // Should have 3 nested layers (one per component)
        assert_eq!(nested_layers.len(), 3);

        // Parent should have shared encoding
        assert!(parent_spec.get("encoding").is_some());

        // Each nested layer should have mark and transform, but not encoding
        for nested_layer in nested_layers {
            assert!(nested_layer.get("mark").is_some());
            assert!(nested_layer.get("transform").is_some());
            assert!(nested_layer.get("encoding").is_none());

            // Mark should have font properties
            let mark = nested_layer["mark"].as_object().unwrap();
            assert!(mark.contains_key("fontWeight"));
            assert!(mark.contains_key("fontStyle"));
        }
    }

    #[test]
    fn test_text_varying_angle() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;
        let layer = Layer::new(crate::plot::Geom::text());

        // Create DataFrame with different angles
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("angle").as_str() => &["0", "45", "90"],
        }
        .unwrap();

        // Prepare data - should result in multiple layers (one per unique angle)
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        match &prepared {
            PreparedData::Composite { components, .. } => {
                // Should have 3 components (one per unique angle)
                assert_eq!(components.len(), 3);
                assert!(components.contains_key("_font_0"));
                assert!(components.contains_key("_font_1"));
                assert!(components.contains_key("_font_2"));
            }
            _ => panic!("Expected Composite"),
        }

        // Build prototype spec
        let prototype = json!({
            "mark": {"type": "text"},
            "encoding": {
                "x": {"field": naming::aesthetic_column("x"), "type": "quantitative"},
                "y": {"field": naming::aesthetic_column("y"), "type": "quantitative"},
                "text": {"field": naming::aesthetic_column("label"), "type": "nominal"}
            }
        });

        // Create a dummy layer
        let layer = crate::plot::Layer::new(crate::plot::Geom::text());

        // Call finalize to get layers
        let layers = renderer
            .finalize(prototype.clone(), &layer, "test", &prepared)
            .unwrap();

        // Should return single parent spec with nested layers
        assert_eq!(layers.len(), 1);

        let parent_spec = &layers[0];
        let nested_layers = parent_spec["layer"].as_array().unwrap();

        // Should have 3 nested layers (one per unique angle)
        assert_eq!(nested_layers.len(), 3);

        // Each layer should have angle property in mark
        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert!(mark.contains_key("angle"));
        }
    }

    #[test]
    fn test_text_varying_angle_numeric() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;
        let layer = Layer::new(crate::plot::Geom::text());

        // Create DataFrame with numeric angle column (matching actual query)
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1, 2, 3],
            naming::aesthetic_column("y").as_str() => &[1, 2, 3],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("angle").as_str() => &[0i32, 180i32, 0i32],  // integer column
        }
        .unwrap();

        // Prepare data - should result in multiple layers (one per unique angle)
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        match &prepared {
            PreparedData::Composite { components, .. } => {
                // Should have 3 components: angle 0 at row 0, angle 180 at row 1, angle 0 at row 2
                // Due to non-contiguous indices, rows 0 and 2 should be in separate components
                eprintln!("Number of components: {}", components.len());
                eprintln!(
                    "Component keys: {:?}",
                    components.keys().collect::<Vec<_>>()
                );
                assert_eq!(components.len(), 3);
            }
            _ => panic!("Expected Composite"),
        }
    }

    #[test]
    fn test_text_angle_integration() {
        use crate::execute;
        use crate::naming;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: Full pipeline from SQL query to Vega-Lite with angle aesthetic
        // This tests that angle values properly create separate layers with angle mark properties

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with text geom and varying angles
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label,
                CASE
                    WHEN n = 0 THEN 0
                    WHEN n = 1 THEN 45
                    WHEN n = 2 THEN 90
                    ELSE 0
                END as rotation
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label, rotation AS angle
            DRAW text
        "#;

        // Execute and prepare data
        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        assert_eq!(prepared.specs.len(), 1);

        let spec = &prepared.specs[0];
        assert_eq!(spec.layers.len(), 1);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Text renderer should create nested layers structure
        assert!(
            vl_spec["layer"].is_array(),
            "Should have top-level layer array"
        );
        let top_layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(top_layers.len(), 1, "Should have one parent text layer");

        // Parent layer should have shared encoding and nested layers
        let parent_layer = &top_layers[0];
        assert!(
            parent_layer["encoding"].is_object(),
            "Parent layer should have shared encoding"
        );
        assert!(
            parent_layer["layer"].is_array(),
            "Parent layer should have nested layers"
        );

        let nested_layers = parent_layer["layer"].as_array().unwrap();

        // Should have multiple nested layers (one per unique angle value)
        // We have angles: 0, 45, 90, 0 -> but non-contiguous 0s split into separate layers
        assert!(
            nested_layers.len() >= 3,
            "Should have at least 3 nested layers for different angles, got {}",
            nested_layers.len()
        );

        // Each nested layer should have mark with angle property
        for (idx, nested_layer) in nested_layers.iter().enumerate() {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert!(
                mark.contains_key("angle"),
                "Nested layer {} mark should have angle property",
                idx
            );
            assert_eq!(mark["type"], "text");

            // Should have source filter transform
            assert!(nested_layer["transform"].is_array());

            // Should NOT have encoding (inherited from parent)
            assert!(nested_layer.get("encoding").is_none());
        }

        // Verify angles are present and normalized [0, 360)
        let angles: Vec<f64> = nested_layers
            .iter()
            .filter_map(|layer| {
                layer["mark"]
                    .as_object()
                    .and_then(|m| m.get("angle"))
                    .and_then(|a| a.as_f64())
            })
            .collect();

        // Should have the three distinct angles: 0, 45, 90
        assert!(angles.contains(&0.0), "Should have 0° angle");
        assert!(angles.contains(&45.0), "Should have 45° angle");
        assert!(angles.contains(&90.0), "Should have 90° angle");

        // Verify data has angle column
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert!(!data_values.is_empty());

        let angle_col = naming::aesthetic_column("angle");
        for row in data_values {
            assert!(
                row[&angle_col].is_number(),
                "Data row should have numeric angle: {:?}",
                row
            );
        }
    }

    #[test]
    fn test_text_nudge_parameters() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: nudge_x and nudge_y parameters should map to xOffset/yOffset

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with nudge parameters
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING nudge_x => 5, nudge_y => -10
        "#;

        // Execute and prepare data
        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        assert_eq!(prepared.specs.len(), 1);

        let spec = &prepared.specs[0];
        assert_eq!(spec.layers.len(), 1);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Text renderer creates nested layers structure
        let top_layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(top_layers.len(), 1);

        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        // All nested layers should have xOffset and yOffset in mark
        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();

            assert!(
                mark.contains_key("xOffset"),
                "Mark should have xOffset from nudge_x"
            );
            assert_eq!(
                mark["xOffset"].as_f64().unwrap(),
                5.0 * POINTS_TO_PIXELS,
                "xOffset should be 5 * POINTS_TO_PIXELS"
            );

            assert!(
                mark.contains_key("yOffset"),
                "Mark should have yOffset from nudge_y"
            );
            assert_eq!(
                mark["yOffset"].as_f64().unwrap(),
                10.0 * POINTS_TO_PIXELS,
                "yOffset should be 10 * POINTS_TO_PIXELS (negated from nudge_y = -10)"
            );
        }
    }

    #[test]
    fn test_text_label_formatting() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: format parameter should transform label values

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with format parameter using Title case transformation
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                CASE
                    WHEN n = 0 THEN 'north region'
                    WHEN n = 1 THEN 'south region'
                    ELSE 'east region'
                END as region
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, region AS label
            DRAW text SETTING format => 'Region: {:Title}'
        "#;

        // Execute and prepare data
        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        assert_eq!(prepared.specs.len(), 1);

        let spec = &prepared.specs[0];
        assert_eq!(spec.layers.len(), 1);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Check that data has formatted labels
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert!(!data_values.is_empty());

        // Verify formatted labels in the data
        let label_col = crate::naming::aesthetic_column("label");

        // Check each row has properly formatted labels
        let labels: Vec<&str> = data_values
            .iter()
            .filter_map(|row| row[&label_col].as_str())
            .collect();

        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&"Region: North Region"));
        assert!(labels.contains(&"Region: South Region"));
        assert!(labels.contains(&"Region: East Region"));
    }

    #[test]
    fn test_text_label_formatting_numeric() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Test numeric formatting with printf-style format

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                n::FLOAT * 10.5 as value
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, value AS label
            DRAW text SETTING format => '${:num %.2f}'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        let label_col = crate::naming::aesthetic_column("label");

        let labels: Vec<&str> = data_values
            .iter()
            .filter_map(|row| row[&label_col].as_str())
            .collect();

        // Should have formatted currency values
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&"$0.00"));
        assert!(labels.contains(&"$10.50"));
        assert!(labels.contains(&"$21.00"));
    }

    #[test]
    fn test_text_setting_fontface() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: SETTING fontface => 'bold' should add fontWeight to base mark

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with fontface in SETTING
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING fontface => 'bold'
        "#;

        // Execute and prepare data
        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        assert_eq!(prepared.specs.len(), 1);

        let spec = &prepared.specs[0];
        assert_eq!(spec.layers.len(), 1);

        // Generate Vega-Lite JSON
        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Text renderer creates nested layers structure
        let top_layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(top_layers.len(), 1);

        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        // All nested layers should have fontWeight: "bold" in mark (from SETTING)
        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();

            assert!(
                mark.contains_key("fontWeight"),
                "Mark should have fontWeight from SETTING fontface"
            );
            assert_eq!(
                mark["fontWeight"].as_str().unwrap(),
                "bold",
                "fontWeight should be bold"
            );
        }
    }
}
