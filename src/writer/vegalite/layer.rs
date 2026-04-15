//! Geom rendering for Vega-Lite writer
//!
//! This module provides:
//! - Basic geom-to-mark mapping and column validation
//! - A trait-based approach to rendering different ggsql geom types to Vega-Lite specs
//!
//! Each geom type can override specific phases of the rendering pipeline while using
//! sensible defaults for standard behavior.

use crate::plot::layer::geom::GeomType;
use crate::plot::layer::is_transposed;
use crate::plot::{ArrayElement, ParameterValue};
use crate::writer::vegalite::POINTS_TO_PIXELS;
use crate::{naming, AestheticValue, DataFrame, Geom, GgsqlError, Layer, Result};
use polars::prelude::ChunkCompareEq;
use serde_json::{json, Map, Value};
use std::any::Any;
use std::collections::HashMap;

use super::data::{dataframe_to_values, dataframe_to_values_with_bins, ROW_INDEX_COLUMN};

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
        GeomType::Rect => "rect",
        GeomType::Ribbon => "area",
        GeomType::Polygon => "line",
        GeomType::Histogram => "bar",
        GeomType::Density => "area",
        GeomType::Violin => "line",
        GeomType::Boxplot => "boxplot",
        GeomType::Text => "text",
        GeomType::Segment => "rule",
        GeomType::Smooth => "line",
        GeomType::Rule => "rule",
        GeomType::ErrorBar => "rule",
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
                    crate::and_list(&available_columns)
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
                crate::and_list(&available_columns)
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
    Single {
        values: Vec<Value>,
        metadata: Box<dyn Any + Send + Sync>,
    },
    /// Multiple component datasets (boxplot, violin, errorbar)
    Composite {
        components: HashMap<String, Vec<Value>>,
        metadata: Box<dyn Any + Send + Sync>,
    },
}

// =============================================================================
// RenderContext
// =============================================================================

/// Context information available to renderers during layer preparation
pub struct RenderContext<'a> {
    /// Scale definitions (for extent and properties)
    pub scales: &'a [crate::Scale],
}

impl<'a> RenderContext<'a> {
    /// Create a new render context
    pub fn new(scales: &'a [crate::Scale]) -> Self {
        Self { scales }
    }

    /// Find a scale by aesthetic name
    pub fn find_scale(&self, aesthetic: &str) -> Option<&crate::Scale> {
        self.scales.iter().find(|s| s.aesthetic == aesthetic)
    }

    /// Get the numeric extent (min, max) for a given aesthetic from its scale
    pub fn get_extent(&self, aesthetic: &str) -> Result<(f64, f64)> {
        use crate::plot::ArrayElement;

        // Find the scale for this aesthetic
        let scale = self.find_scale(aesthetic).ok_or_else(|| {
            GgsqlError::ValidationError(format!(
                "Cannot determine extent for aesthetic '{}': no scale found",
                aesthetic
            ))
        })?;

        // Extract continuous range from input_range
        if let Some(range) = &scale.input_range {
            if range.len() >= 2 {
                if let (ArrayElement::Number(min), ArrayElement::Number(max)) =
                    (&range[0], &range[1])
                {
                    return Ok((*min, *max));
                }
            }
        }

        Err(GgsqlError::ValidationError(format!(
            "Cannot determine extent for aesthetic '{}': scale has no valid numeric range",
            aesthetic
        )))
    }
}

// =============================================================================
// GeomRenderer Trait System
// =============================================================================

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
        Ok(PreparedData::Single {
            values,
            metadata: Box::new(()),
        })
    }

    // === Phase 2: Encoding Modifications ===

    /// Modify the encoding map for this geom.
    /// Default: no modifications
    fn modify_encoding(
        &self,
        _encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        Ok(())
    }

    /// Modify the mark/layer spec for this geom.
    /// Default: no modifications
    fn modify_spec(
        &self,
        _layer_spec: &mut Value,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        let width = match layer.adjusted_width {
            // The adjusted width comes from position adjustments
            Some(adjusted) => adjusted,
            _ => match layer.parameters.get("width") {
                // Fallback to width parameter value if there is no adjustment
                Some(ParameterValue::Number(n)) => *n,
                _ => 0.9,
            },
        };

        // For horizontal bars, use "height" for band size; for vertical, use "width"
        let is_horizontal = is_transposed(layer);
        let axis = if is_horizontal { "y" } else { "x" };

        let size_value = match layer_spec["encoding"][axis]["bin"].as_str() {
            // I don't think binned scales obey 'band', but they don't tolerate the 'expr' option.
            Some("binned") => json!({"band": width}),
            // Use expression-based size with the adjusted width
            _ => json!({"expr": format!("bandwidth('{}') * {}", axis, width)}),
        };

        layer_spec["mark"] = if is_horizontal {
            json!({
                "type": "bar",
                "height": size_value,
                "baseline": "middle",
                "clip": true
            })
        } else {
            json!({
                "type": "bar",
                "width": size_value,
                "align": "center",
                "clip": true
            })
        };
        Ok(())
    }
}

// =============================================================================
// Path Renderer
// =============================================================================

/// Renderer for path and line geoms - preserves data order for correct rendering
///
/// Automatically detects when continuous material aesthetics (stroke, linewidth, opacity) vary
/// within partition groups and converts to segmented rendering using detail encoding.
/// Discrete material aesthetics (linetype, or discrete stroke) already define groups
/// via partition_by and don't require special handling.
///
/// Handles both `line` and `path` geoms - the only difference is the mark type used.
pub struct PathRenderer;

// =============================================================================
// Helper functions for path/line segmentation
// =============================================================================

/// Find row indices where any of the specified columns change value.
///
/// Returns a sorted vector starting with 0 (first row), followed by indices
/// where any column value differs from the previous row. Does not include
/// the final boundary (n_rows).
///
/// Used by both line segmentation and text font run-length encoding.
fn find_change_starts(df: &DataFrame, columns: &[String]) -> Result<Vec<usize>> {
    use polars::prelude::*;

    let n_rows = df.height();

    if columns.is_empty() || n_rows <= 1 {
        return Ok(vec![0]);
    }

    // Initialize change mask as all false (no changes)
    let mut change_mask = BooleanChunked::full("change_mask".into(), false, n_rows - 1);

    // For each column, OR its change mask into the accumulator
    for col_name in columns {
        let series = df.column(col_name).map_err(|e| {
            GgsqlError::InternalError(format!("Column '{}' not found: {}", col_name, e))
        })?;

        // Compare each row with the previous row
        // curr = series[1..n], prev = series[0..n-1]
        let curr = series.slice(1, n_rows - 1);
        let prev = series.slice(0, n_rows - 1);

        // Get boolean mask where values differ
        let not_equal = curr.not_equal(&prev).map_err(|e| {
            GgsqlError::InternalError(format!("Failed to compare column '{}': {}", col_name, e))
        })?;

        // OR with accumulator (change if this column OR any previous column changed)
        change_mask = &change_mask | &not_equal;
    }

    // Extract indices where mask is true (offset by 1 since we compared with previous)
    let mut change_starts = vec![0];
    for (idx, changed) in change_mask.into_iter().enumerate() {
        if changed == Some(true) {
            change_starts.push(idx + 1);
        }
    }

    Ok(change_starts)
}

/// Check if an aesthetic varies within any group segment
///
/// Uses precomputed group boundaries to efficiently check if the aesthetic
/// has multiple distinct values within any group segment.
fn aesthetic_varies_within_groups(
    df: &DataFrame,
    aesthetic_col: &str,
    group_boundaries: &[usize],
) -> Result<bool> {
    let series = df.column(aesthetic_col).map_err(|e| {
        GgsqlError::InternalError(format!("Column '{}' not found: {}", aesthetic_col, e))
    })?;

    // Check each group segment
    for window in group_boundaries.windows(2) {
        let start = window[0];
        let end = window[1];

        if end - start < 2 {
            continue; // Single-row groups can't vary
        }

        // Slice the series for this group and check uniqueness
        let segment = series.slice(start as i64, end - start);
        let n_unique = segment.n_unique().map_err(|e| {
            GgsqlError::InternalError(format!("Failed to count unique values: {}", e))
        })?;

        if n_unique > 1 {
            return Ok(true);
        }
    }

    Ok(false)
}

impl GeomRenderer for PathRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        layer: &Layer,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        // Continuous material aesthetics that can trigger segmentation
        // (linetype is always discrete and already handled via partition_by)
        let material_aesthetics: &[&'static str] = &["stroke", "linewidth", "opacity"];

        // Start with existing partition_by (includes discrete material aesthetics already)
        let partition_columns: Vec<String> = layer.partition_by.clone();

        // Compute group boundaries based on existing partitions
        let n_rows = df.height();
        let group_boundaries = if partition_columns.is_empty() || n_rows <= 1 {
            vec![0, n_rows]
        } else {
            let mut boundaries = find_change_starts(df, &partition_columns)?;
            boundaries.push(n_rows);
            boundaries
        };

        // Check continuous material aesthetics (not in partition_by) for within-group variation
        let mut varying_aesthetics: Vec<&'static str> = Vec::new();

        for &aesthetic in material_aesthetics {
            if let Some(AestheticValue::Column { name: col, .. }) = layer.mappings.get(aesthetic) {
                // Skip if already in partition_by (discrete, already defines groups)
                if !layer.partition_by.contains(col) {
                    // Continuous: check if varies within groups
                    if aesthetic_varies_within_groups(df, col, &group_boundaries)? {
                        varying_aesthetics.push(aesthetic);
                        // Don't add to partition_columns - continuous values shouldn't partition
                    }
                }
            }
        }

        // Return the data with segmentation metadata if needed
        let values = if binned_columns.is_empty() {
            dataframe_to_values(df)?
        } else {
            dataframe_to_values_with_bins(df, binned_columns)?
        };

        Ok(PreparedData::Single {
            values,
            metadata: Box::new(varying_aesthetics),
        })
    }

    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Use row index field to preserve natural data order
        // (we've already ordered in SQL via apply_stat_transform)
        encoding.insert(
            "order".to_string(),
            json!({"field": ROW_INDEX_COLUMN, "type": "quantitative"}),
        );
        Ok(())
    }

    fn finalize(
        &self,
        mut layer_spec: Value,
        layer: &Layer,
        _data_key: &str,
        prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        // Get metadata from prepared data
        let PreparedData::Single { metadata, .. } = prepared else {
            return Err(GgsqlError::InternalError(
                "PathRenderer expects PreparedData::Single".to_string(),
            ));
        };

        // Get varying aesthetics from metadata
        let Some(varying_aesthetics) = metadata.downcast_ref::<Vec<&'static str>>() else {
            return Ok(vec![layer_spec]);
        };

        // Handle varying linewidth: switch to trail mark and translate encodings
        if varying_aesthetics.contains(&"linewidth") {
            layer_spec["mark"] = json!({"type": "trail", "clip": true, "strokeWidth": 0});

            // Translate line encodings to trail encodings
            if let Some(encoding_obj) = layer_spec.get_mut("encoding") {
                if let Some(encoding_map) = encoding_obj.as_object_mut() {
                    // strokeWidth → size
                    if let Some(stroke_width) = encoding_map.remove("strokeWidth") {
                        encoding_map.insert("size".to_string(), stroke_width);
                    }

                    // stroke → fill
                    if let Some(mut stroke) = encoding_map.remove("stroke") {
                        // Add symbolStrokeColor to legend so symbols display with color
                        if let Some(stroke_obj) = stroke.as_object_mut() {
                            if let Some(legend) = stroke_obj.get_mut("legend") {
                                if let Some(legend_obj) = legend.as_object_mut() {
                                    legend_obj.insert(
                                        "symbolStrokeColor".to_string(),
                                        json!({"expr": "scale('fill', datum.value)"}),
                                    );
                                }
                            }
                        }
                        encoding_map.insert("fill".to_string(), stroke);
                    }

                    // opacity → fillOpacity
                    if let Some(opacity) = encoding_map.remove("opacity") {
                        encoding_map.insert("fillOpacity".to_string(), opacity);
                    }
                }
            }
        }

        // Handle varying stroke/opacity: apply segmentation
        if !varying_aesthetics.contains(&"stroke") && !varying_aesthetics.contains(&"opacity") {
            // Only linewidth varies, trail mark handles it natively
            return Ok(vec![layer_spec]);
        }

        // Build list of fields to segment (always x/y, plus size if linewidth varies)
        let mut segment_fields = vec![
            ("x", naming::aesthetic_column("pos1")),
            ("y", naming::aesthetic_column("pos2")),
        ];
        if varying_aesthetics.contains(&"linewidth") {
            segment_fields.push(("size", naming::aesthetic_column("linewidth")));
        }

        // Segmented rendering using detail encoding:
        // 1. Create segment IDs (row_index serves as segment ID)
        // 2. Create next row's values using window transform
        // 3. Flatten to create 2 rows per segment (point_index: 0=start, 1=end)
        // 4. Use calculate to pick current or next based on point_index
        // 5. Add segment ID to detail encoding

        // Preserve existing transforms (e.g., source filter)
        let mut transforms = layer_spec
            .get("transform")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        // Step 1 & 2: Window transform to get next row's values
        let window_ops: Vec<Value> = segment_fields
            .iter()
            .map(|(_, field)| {
                json!({
                    "op": "lead",
                    "field": field,
                    "as": format!("{}_next", field)
                })
            })
            .collect();

        let mut window_transform = json!({
            "window": window_ops,
            "sort": [{"field": ROW_INDEX_COLUMN}]
        });

        if !layer.partition_by.is_empty() {
            window_transform["groupby"] = json!(layer.partition_by);
        }

        transforms.push(window_transform);

        // Step 2b: Filter out last row in each group (no next point)
        // Check the first field (x) for null to detect end of segments
        let first_field = &segment_fields[0].1;
        transforms.push(json!({
            "filter": format!("datum.{}_next != null", first_field)
        }));

        // Step 3: Flatten to create 2 rows per segment
        // Create a constant array [0, 1] to flatten
        transforms.push(json!({
            "calculate": "[0, 1]",
            "as": "__segment_points__"
        }));

        transforms.push(json!({
            "flatten": ["__segment_points__"],
            "as": ["__point_index__"]
        }));

        // Step 4: Calculate actual field values based on point_index
        for (_, field) in &segment_fields {
            transforms.push(json!({
                "calculate": format!("datum.__point_index__ == 0 ? datum.{} : datum.{}_next", field, field),
                "as": format!("{}_final", field)
            }));
        }

        // Step 5: Create segment ID (use original row_index)
        transforms.push(json!({
            "calculate": format!("datum.{}", ROW_INDEX_COLUMN),
            "as": "__segment_id__"
        }));

        layer_spec["transform"] = json!(transforms);
        // Don't set layer_spec["data"] - use the unified top-level dataset
        // The source filter transform will select the correct rows

        // Update encodings to use final field values and add segment_id to detail
        if let Some(encoding_obj) = layer_spec.get_mut("encoding") {
            if let Some(encoding_map) = encoding_obj.as_object_mut() {
                // Update each field encoding to use _final
                for (encoding_name, field) in &segment_fields {
                    if let Some(enc) = encoding_map.get_mut(*encoding_name) {
                        if let Some(enc_obj) = enc.as_object_mut() {
                            enc_obj.insert("field".to_string(), json!(format!("{}_final", field)));
                        }
                    }
                }

                // Add segment_id to detail encoding
                encoding_map.insert(
                    "detail".to_string(),
                    json!({
                        "field": "__segment_id__",
                        "type": "nominal"
                    }),
                );
            }
        }

        Ok(vec![layer_spec])
    }
}

// =============================================================================
// Segment Renderer
// =============================================================================

pub struct SegmentRenderer;

impl GeomRenderer for SegmentRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        let has_x2 = encoding.contains_key("x2");
        let has_y2 = encoding.contains_key("y2");
        if !has_x2 && !has_y2 {
            return Err(GgsqlError::ValidationError(
                "The `segment` layer requires at least one of the `xend` or `yend` aesthetics."
                    .to_string(),
            ));
        }
        if !has_x2 {
            if let Some(x) = encoding.get("x").cloned() {
                encoding.insert("x2".to_string(), x);
            }
        }
        if !has_y2 {
            if let Some(y) = encoding.get("y").cloned() {
                encoding.insert("y2".to_string(), y);
            }
        }
        Ok(())
    }
}

// =============================================================================
// Rule Renderer
// =============================================================================

pub struct RuleRenderer;

impl GeomRenderer for RuleRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        layer: &Layer,
        context: &RenderContext,
    ) -> Result<()> {
        let has_x = encoding.contains_key("x");
        let has_y = encoding.contains_key("y");

        // Remove slope from encoding (it's never a visual encoding, only metadata)
        // and check if it's non-zero (diagonal line)
        let diagonal = matches!(
            layer.parameters.get("diagonal"),
            Some(ParameterValue::Boolean(true))
        );

        if !has_x && !has_y && !diagonal {
            return Err(GgsqlError::ValidationError(
                "The `rule` layer requires the `x` or `y` aesthetic. It currently has neither."
                    .to_string(),
            ));
        } else if has_x && has_y && !diagonal {
            return Err(GgsqlError::ValidationError(
                "The `rule` layer requires exactly one of the `x` or `y` aesthetic, not both."
                    .to_string(),
            ));
        }
        if !diagonal {
            return Ok(());
        }

        // Use layer's pre-computed orientation
        let (primary, primary2, secondary, secondary2, extent_aes) = if is_transposed(layer) {
            ("y", "y2", "x", "x2", "pos2")
        } else {
            ("x", "x2", "y", "y2", "pos1")
        };

        // Get primary axis extent from context to set explicit scale domain
        // This prevents an axis drift
        let mut primary_enco = json!({"field": "primary_min", "type": "quantitative"});
        if let Ok((min, max)) = context.get_extent(extent_aes) {
            primary_enco["scale"] = json!({"domain": [min, max]})
        };

        // Add encodings for rule mark
        // primary_min/primary_max are created by transforms (extent of the axis)
        // secondary_min/secondary_max are computed via formula
        encoding.insert(primary.to_string(), primary_enco);
        encoding.insert(
            primary2.to_string(),
            json!({
                "field": "primary_max"
            }),
        );
        encoding.insert(
            secondary.to_string(),
            json!({
                "field": "secondary_min",
                "type": "quantitative"
            }),
        );
        encoding.insert(
            secondary2.to_string(),
            json!({
                "field": "secondary_max"
            }),
        );

        Ok(())
    }

    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        layer: &Layer,
        context: &RenderContext,
    ) -> Result<()> {
        // Determine slope expression: either a literal value or a field reference
        let slope_expr = match layer.mappings.get("slope") {
            Some(AestheticValue::Literal(ParameterValue::Number(n))) if *n == 0.0 => {
                // Slope is 0 - no diagonal
                None
            }
            Some(AestheticValue::Literal(ParameterValue::Number(n))) => {
                // Literal non-zero slope - inline the value
                Some(n.to_string())
            }
            Some(AestheticValue::Column { .. }) | Some(AestheticValue::AnnotationColumn { .. }) => {
                // Column-based slope - reference the field
                let slope_field = naming::aesthetic_column("slope");
                Some(format!("datum.{}", slope_field))
            }
            _ => {
                // No slope mapping - no diagonal
                None
            }
        };

        let Some(slope_expr) = slope_expr else {
            return Ok(());
        };

        let (intercept_field, extent_aes) = if is_transposed(layer) {
            // x is intercept
            (naming::aesthetic_column("pos1"), "pos2")
        } else {
            // y is intercept
            (naming::aesthetic_column("pos2"), "pos1")
        };

        // Get extent from appropriate axis:
        let (primary_min, primary_max) = context.get_extent(extent_aes)?;

        // Add transforms:
        // 1. Create constant primary_min/primary_max fields (extent of the primary axis)
        // 2. Compute secondary values at those primary positions: secondary = slope * primary + intercept
        //    (where intercept is pos1 for x-mapped or pos2 for y-mapped)
        let transforms = json!([
            {
                "calculate": primary_min.to_string(),
                "as": "primary_min"
            },
            {
                "calculate": primary_max.to_string(),
                "as": "primary_max"
            },
            {
                "calculate": format!("{} * datum.primary_min + datum.{}", slope_expr, intercept_field),
                "as": "secondary_min"
            },
            {
                "calculate": format!("{} * datum.primary_max + datum.{}", slope_expr, intercept_field),
                "as": "secondary_max"
            }
        ]);

        // Prepend to existing transforms (if any)
        if let Some(existing) = layer_spec.get("transform") {
            if let Some(arr) = existing.as_array() {
                let mut new_transforms = transforms.as_array().unwrap().clone();
                new_transforms.extend_from_slice(arr);
                layer_spec["transform"] = json!(new_transforms);
            }
        } else {
            layer_spec["transform"] = transforms;
        }

        Ok(())
    }
}

// =============================================================================
// Text Renderer
// =============================================================================

/// Renderer for text geom - handles font properties via data splitting
pub struct TextRenderer;

impl TextRenderer {
    /// Analyze DataFrame columns to build font property runs using run-length encoding.
    /// Returns:
    /// - DataFrame where each row represents a run's font properties (family, fontweight, italic, hjust, vjust, angle)
    /// - Vec<usize> of run lengths corresponding to each row
    fn build_font_rle(df: &DataFrame) -> Result<(DataFrame, Vec<usize>)> {
        use polars::prelude::*;

        let nrows = df.height();

        if nrows == 0 {
            // Return empty DataFrame and empty run lengths
            return Ok((DataFrame::default(), Vec::new()));
        }

        // Collect font property column names that exist in the DataFrame
        let font_aesthetics = [
            "typeface",
            "fontweight",
            "italic",
            "hjust",
            "vjust",
            "rotation",
        ];

        let mut font_column_names = Vec::new();
        let mut font_columns: HashMap<&str, &polars::prelude::Column> = HashMap::new();

        for aesthetic in font_aesthetics {
            let col_name = naming::aesthetic_column(aesthetic);
            if let Ok(col) = df.column(&col_name) {
                font_column_names.push(col_name);
                font_columns.insert(aesthetic, col);
            }
        }

        // Find indices where any font property changes
        let change_indices = find_change_starts(df, &font_column_names)?;

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

    /// Split label values containing newlines into arrays of strings
    ///
    /// Uses the shared split_label_on_newlines function to ensure consistent
    /// newline handling across all label types (text data, axis labels, titles, etc.)
    fn split_label_newlines(values: &mut [Value]) -> Result<()> {
        let label_col = naming::aesthetic_column("label");

        for row in values.iter_mut() {
            // Get the object, skip if not an object
            let Some(obj) = row.as_object_mut() else {
                continue;
            };

            // Get the label value, skip if not present
            let Some(label_value) = obj.get(&label_col) else {
                continue;
            };
            // Get the string value, skip if not a string
            let Some(label_str) = label_value.as_str() else {
                continue;
            };

            // Use shared function for consistent newline splitting
            obj.insert(label_col.clone(), super::split_label_on_newlines(label_str));
        }
        Ok(())
    }

    /// Convert typeface to Vega-Lite font value
    /// Prefers literal over column value
    fn convert_typeface(
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

    /// Convert fontweight to Vega-Lite fontWeight value
    /// Prefers literal over column value
    /// Accepts all CSS font-weight keywords and numeric values:
    /// - Keywords: 'thin', 'hairline', 'extra-light', 'ultra-light', 'light',
    ///   'normal', 'regular', 'lighter', 'medium', 'semi-bold', 'demi-bold',
    ///   'bold', 'bolder', 'extra-bold', 'ultra-bold', 'black', 'heavy'
    /// - Numeric values: any number
    /// - Numeric strings from columns: '100', '400', '700', etc.
    ///
    /// Always outputs 'normal' or 'bold' for Vega-Lite compatibility:
    /// - < 500 → 'normal' (thin, light, normal, regular, lighter)
    /// - >= 500 → 'bold' (medium, semi-bold, bold, bolder, extra-bold, black, heavy)
    fn convert_fontweight(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> Option<Value> {
        // First select which value to use (prefer literal)
        let numeric = match literal {
            Some(ParameterValue::String(s)) => {
                // String literal: keyword or numeric string
                Self::parse_fontweight_to_numeric(s.as_str())
            }
            Some(ParameterValue::Number(n)) => {
                // Numeric literal: use directly
                Some(*n)
            }
            _ => {
                // Column value: try to parse
                column_value.and_then(Self::parse_fontweight_to_numeric)
            }
        }?;

        // Apply >= 500 rule to determine bold/normal
        let is_bold = numeric >= 500.0;
        Some(json!(if is_bold { "bold" } else { "normal" }))
    }

    /// Parse fontweight value from string to numeric value
    fn parse_fontweight_to_numeric(value: &str) -> Option<f64> {
        // Try parsing as number first
        if let Ok(num) = value.parse::<f64>() {
            return Some(num);
        }

        // Map CSS font-weight keywords to numeric values
        // Normalize: convert to lowercase and remove hyphens for flexible matching
        let normalized = value.to_lowercase().replace("-", "");
        match normalized.as_str() {
            "thin" | "hairline" => Some(100.0),
            "extralight" | "ultralight" => Some(200.0),
            "light" => Some(300.0),
            "normal" | "regular" | "lighter" => Some(400.0),
            "medium" => Some(500.0),
            "semibold" | "demibold" => Some(600.0),
            "bold" | "bolder" => Some(700.0),
            "extrabold" | "ultrabold" => Some(800.0),
            "black" | "heavy" => Some(900.0),
            _ => None,
        }
    }

    /// Convert italic to Vega-Lite fontStyle value
    /// Prefers literal over column value
    /// Accepts boolean literals or string column values ('true', 'false', '1', '0')
    fn convert_italic(
        literal: Option<&ParameterValue>,
        column_value: Option<&str>,
    ) -> Option<Value> {
        // First select which value to use (prefer literal)
        let value = if let Some(ParameterValue::Boolean(b)) = literal {
            *b
        } else if let Some(s) = column_value {
            // Parse string to boolean
            match s.to_lowercase().as_str() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => return None,
            }
        } else {
            return None;
        };

        // Convert boolean to fontStyle
        let style = if value { "italic" } else { "normal" };
        Some(json!(style))
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

    /// Convert rotation to Vega-Lite angle value (degrees)
    /// Prefers literal over column value
    /// Normalizes angles to [0, 360) range
    fn convert_rotation(
        literal: Option<&ParameterValue>,
        column_value: Option<f64>,
    ) -> Option<Value> {
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
        if let Some(typeface_val) = Self::convert_typeface(
            layer.get_literal("typeface"),
            get_str("typeface").as_deref(),
        ) {
            mark_obj.insert("font".to_string(), typeface_val);
        }

        if let Some(weight) = Self::convert_fontweight(
            layer.get_literal("fontweight"),
            get_str("fontweight").as_deref(),
        ) {
            mark_obj.insert("fontWeight".to_string(), weight);
        }

        if let Some(style) =
            Self::convert_italic(layer.get_literal("italic"), get_str("italic").as_deref())
        {
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

        if let Some(angle_val) =
            Self::convert_rotation(layer.get_literal("rotation"), get_f64("rotation"))
        {
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
            // Extract offset parameter (offset => [x, y] or offset => n)
            match layer.parameters.get("offset") {
                Some(ParameterValue::Array(offset_array)) if offset_array.len() == 2 => {
                    // Array case: [x, y]
                    if let ArrayElement::Number(x_offset) = offset_array[0] {
                        mark_map.insert("xOffset".to_string(), json!(x_offset * POINTS_TO_PIXELS));
                    }
                    if let ArrayElement::Number(y_offset) = offset_array[1] {
                        mark_map.insert("yOffset".to_string(), json!(-y_offset * POINTS_TO_PIXELS));
                    }
                }
                Some(ParameterValue::Number(offset)) => {
                    // Single number case: applies to both x and y
                    mark_map.insert("xOffset".to_string(), json!(offset * POINTS_TO_PIXELS));
                    mark_map.insert("yOffset".to_string(), json!(-offset * POINTS_TO_PIXELS));
                }
                _ => {}
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
        _layer: &Layer,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        // Note: Label formatting is already applied via Text::post_process() during execution

        // Analyze font columns to get RLE runs
        let (font_runs_df, run_lengths) = Self::build_font_rle(df)?;

        // Split data by font runs, tracking cumulative position
        let mut components: HashMap<String, Vec<Value>> = HashMap::new();
        let mut position = 0;

        for (run_idx, &length) in run_lengths.iter().enumerate() {
            let suffix = format!("_font_{}", run_idx);

            // Slice the contiguous run from the DataFrame (more efficient than boolean masking)
            let sliced = df.slice(position as i64, length);

            let mut values = if binned_columns.is_empty() {
                dataframe_to_values(&sliced)?
            } else {
                dataframe_to_values_with_bins(&sliced, binned_columns)?
            };

            // Post-process label values to split on newlines
            Self::split_label_newlines(&mut values)?;

            components.insert(suffix, values);
            position += length;
        }

        Ok(PreparedData::Composite {
            components,
            metadata: Box::new((font_runs_df, run_lengths)),
        })
    }

    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Remove font aesthetics from encoding - they only work as mark properties
        for &aesthetic in &[
            "typeface",
            "fontweight",
            "italic",
            "hjust",
            "vjust",
            "rotation",
        ] {
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

/// Renderer for ribbon geom - remaps ymin/ymax to y/y2 and preserves data order
pub struct RibbonRenderer;

impl GeomRenderer for RibbonRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        let is_horizontal = is_transposed(layer);

        // Remap min/max to primary/secondary based on orientation:
        // - Aligned (vertical): ymax→y, ymin→y2
        // - Transposed (horizontal): xmax→x, xmin→x2
        let (max_key, min_key, target, target2) = if is_horizontal {
            ("xmax", "xmin", "x", "x2")
        } else {
            ("ymax", "ymin", "y", "y2")
        };

        if let Some(max_val) = encoding.remove(max_key) {
            encoding.insert(target.to_string(), max_val);
        }
        if let Some(min_val) = encoding.remove(min_key) {
            encoding.insert(target2.to_string(), min_val);
        }

        // Note: Don't add order encoding for area marks - it interferes with rendering
        Ok(())
    }
}

// =============================================================================
// Rect Renderer
// =============================================================================

/// Renderer for rect geom - handles continuous and discrete rectangles
///
/// For continuous scales: remaps xmin/xmax → x/x2, ymin/ymax → y/y2
/// For discrete scales: keeps x/y as-is and applies width/height as band fractions
pub struct RectRenderer;

impl GeomRenderer for RectRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Handle x-direction: continuous if has xmin/xmax, discrete otherwise
        if let Some(xmin) = encoding.remove("xmin") {
            encoding.insert("x".to_string(), xmin);
        }
        if let Some(xmax) = encoding.remove("xmax") {
            encoding.insert("x2".to_string(), xmax);
        }

        // Handle y-direction: continuous if has ymin/ymax, discrete otherwise
        if let Some(ymin) = encoding.remove("ymin") {
            encoding.insert("y".to_string(), ymin);
        }
        if let Some(ymax) = encoding.remove("ymax") {
            encoding.insert("y2".to_string(), ymax);
        }

        Ok(())
    }

    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        let encoding = layer_spec
            .get_mut("encoding")
            .and_then(|e| e.as_object_mut());

        let Some(encoding) = encoding else {
            return Ok(());
        };

        // Check which directions are discrete
        let x_is_discrete = !encoding.contains_key("x2");
        let y_is_discrete = !encoding.contains_key("y2");

        // Early return if both continuous
        if !x_is_discrete && !y_is_discrete {
            return Ok(());
        }

        // Build mark properties for discrete directions
        let mut mark = json!({
            "type": "rect",
            "clip": true
        });

        if x_is_discrete {
            if let Some(width_enc) = encoding.remove("width") {
                // Check if it's a field encoding or literal value
                if let Some(field) = width_enc.get("field").and_then(|f| f.as_str()) {
                    // Field encoding: use expression with datum reference
                    mark["width"] = json!({
                        "expr": format!("datum.{} * bandwidth('x')", field)
                    });
                } else if let Some(value) = width_enc.get("value") {
                    // Literal value: use band syntax
                    mark["width"] = json!({"band": value});
                }
            }
        }

        if y_is_discrete {
            if let Some(height_enc) = encoding.remove("height") {
                // Check if it's a field encoding or literal value
                if let Some(field) = height_enc.get("field").and_then(|f| f.as_str()) {
                    // Field encoding: use expression with datum reference
                    mark["height"] = json!({
                        "expr": format!("datum.{} * bandwidth('y')", field)
                    });
                } else if let Some(value) = height_enc.get("value") {
                    // Literal value: use band syntax
                    mark["height"] = json!({"band": value});
                }
            }
        }

        // Only set mark if we added width or height
        if mark.get("width").is_some() || mark.get("height").is_some() {
            layer_spec["mark"] = mark;
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
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Polygon needs both `fill` and `stroke` independently, but map_aesthetic_name()
        // converts fill -> color (which works for most geoms). For closed line marks,
        // we need actual `fill` and `stroke` channels, so we undo the mapping here.
        if let Some(color) = encoding.remove("color") {
            encoding.insert("fill".to_string(), color);
        }
        // Use row index field to preserve natural data order
        encoding.insert(
            "order".to_string(),
            json!({"field": ROW_INDEX_COLUMN, "type": "quantitative"}),
        );
        Ok(())
    }

    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        layer_spec["mark"] = json!({
            "type": "line",
            "filled": true
        });
        let offset_col = naming::aesthetic_column("offset");

        // Read orientation from layer (already resolved during execution)
        let is_horizontal = is_transposed(layer);

        // It'll be implemented as an offset.
        let violin_offset = match layer.parameters.get("side") {
            Some(ParameterValue::String(side)) if side != "both" => {
                let positive = if is_horizontal {
                    matches!(side.as_str(), "bottom" | "left")
                } else {
                    matches!(side.as_str(), "top" | "right")
                };
                if positive {
                    format!("[datum.{offset}]", offset = offset_col)
                } else {
                    format!("[-datum.{offset}]", offset = offset_col)
                }
            }
            _ => format!("[datum.{offset}, -datum.{offset}]", offset = offset_col),
        };

        // Continuous axis column for order calculation:
        // - Vertical: pos2 (y-axis has continuous density values)
        // - Horizontal: pos1 (x-axis has continuous density values)
        let continuous_col = if is_horizontal {
            naming::aesthetic_column("pos1")
        } else {
            naming::aesthetic_column("pos2")
        };

        // We use an order calculation to create a proper closed shape.
        // Right side (+ offset), sort by -continuous (top -> bottom)
        // Left side (- offset), sort by +continuous (bottom -> top)
        let calc_order = format!(
            "datum.__violin_offset > 0 ? -datum.{} : datum.{}",
            continuous_col, continuous_col
        );

        // Preserve existing transforms (e.g., source filter) and extend with violin-specific transforms
        let existing_transforms = layer_spec
            .get("transform")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        // Check if pos1offset exists (from dodging) - we'll combine it with violin offset
        let pos1offset_col = naming::aesthetic_column("pos1offset");

        let mut transforms = existing_transforms;
        transforms.extend(vec![
            json!({
                // Mirror offset on both sides (offset is pre-scaled to [0, 0.5 * width])
                "calculate": violin_offset,
                "as": "violin_offsets"
            }),
            json!({
                "flatten": ["violin_offsets"],
                "as": ["__violin_offset"]
            }),
            json!({
                // Add pos1offset (dodge displacement) if it exists, otherwise use violin offset directly
                // This positions the violin correctly when dodging
                "calculate": format!(
                    "datum.{pos1offset} != null ? datum.__violin_offset + datum.{pos1offset} : datum.__violin_offset",
                    pos1offset = pos1offset_col
                ),
                "as": "__final_offset"
            }),
            json!({
                "calculate": calc_order,
                "as": "__order"
            }),
        ]);

        layer_spec["transform"] = json!(transforms);
        Ok(())
    }

    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Read orientation from layer (already resolved during execution)
        let is_horizontal = is_transposed(layer);

        encoding.remove("offset");

        // Categorical axis for detail encoding:
        // - Vertical: x channel (categorical groups on x-axis)
        // - Horizontal: y channel (categorical groups on y-axis)
        let categorical_channel = if is_horizontal { "y" } else { "x" };

        // Ensure categorical field is in detail encoding to create separate violins per category
        // This is needed because line marks with filled:true require detail to create separate paths
        let categorical_field = encoding
            .get(categorical_channel)
            .and_then(|x| x.get("field"))
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        if let Some(cat_field) = categorical_field {
            match encoding.get_mut("detail") {
                Some(detail) if detail.is_object() => {
                    // Single field object - check if it's already the categorical field, otherwise convert to array
                    if detail.get("field").and_then(|f| f.as_str()) != Some(&cat_field) {
                        let existing = detail.clone();
                        *detail = json!([existing, {"field": cat_field, "type": "nominal"}]);
                    }
                }
                Some(detail) if detail.is_array() => {
                    // Array - check if categorical field already present, add if not
                    let arr = detail.as_array_mut().unwrap();
                    let has_cat = arr
                        .iter()
                        .any(|d| d.get("field").and_then(|f| f.as_str()) == Some(&cat_field));
                    if !has_cat {
                        arr.push(json!({"field": cat_field, "type": "nominal"}));
                    }
                }
                None => {
                    // No detail encoding - add it with categorical field
                    encoding.insert(
                        "detail".to_string(),
                        json!({"field": cat_field, "type": "nominal"}),
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

        // Offset channel:
        // - Vertical: xOffset (offsets left/right from category)
        // - Horizontal: yOffset (offsets up/down from category)
        let offset_channel = if is_horizontal { "yOffset" } else { "xOffset" };
        encoding.insert(
            offset_channel.to_string(),
            json!({
                "field": "__final_offset",
                "type": "quantitative",
                "scale": {
                    "domain": [-0.5, 0.5]
                }
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
// Errorbar Renderer
// =============================================================================

struct ErrorBarRenderer;

impl GeomRenderer for ErrorBarRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Check combinations of aesthetics
        let has_x = encoding.contains_key("x");
        let has_y = encoding.contains_key("y");
        if has_x && has_y {
            Err(GgsqlError::ValidationError(
                "In errorbar layer, the `x` and `y` aesthetics are mutually exclusive".to_string(),
            ))
        } else if has_x && (encoding.contains_key("xmin") || encoding.contains_key("xmax")) {
            Err(GgsqlError::ValidationError("In errorbar layer, cannot use `x` aesthetic with `xmin` and `xmax`. `x` must be used with `ymin` and `ymax`.".to_string()))
        } else if has_y && (encoding.contains_key("ymin") || encoding.contains_key("ymax")) {
            Err(GgsqlError::ValidationError("In errorbar layer, cannot use `y` aesthetic with `ymin` and `ymax`. `y` must be used with `xmin` and `xmax`.".to_string()))
        } else if has_x {
            if let Some(ymax) = encoding.remove("ymax") {
                encoding.insert("y".to_string(), ymax);
            }
            if let Some(ymin) = encoding.remove("ymin") {
                encoding.insert("y2".to_string(), ymin);
            }
            Ok(())
        } else if has_y {
            if let Some(xmax) = encoding.remove("xmax") {
                encoding.insert("x".to_string(), xmax);
            }
            if let Some(xmin) = encoding.remove("xmin") {
                encoding.insert("x2".to_string(), xmin);
            }
            Ok(())
        } else {
            Err(GgsqlError::ValidationError(
                "In errorbar layer, aesthetics are incomplete. Either use `x`/`ymin`/`ymax` or `y`/`xmin`/`xmax` combinations.".to_string()
            ))
        }
    }

    fn finalize(
        &self,
        layer_spec: Value,
        layer: &Layer,
        _data_key: &str,
        _prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        // Get width parameter (in points)
        let width = if let Some(ParameterValue::Number(num)) = layer.parameters.get("width") {
            (*num) * POINTS_TO_PIXELS
        } else {
            // If no width specified, return just the main error bar without hinges
            return Ok(vec![layer_spec]);
        };

        let mut layers = vec![layer_spec.clone()];

        // Determine if this is a vertical or horizontal error bar and set up parameters
        let is_vertical = !is_transposed(layer);
        let (orient, position, min_field, max_field) = if is_vertical {
            (
                "horizontal",
                "y",
                naming::aesthetic_column("pos2min"),
                naming::aesthetic_column("pos2max"),
            )
        } else {
            (
                "vertical",
                "x",
                naming::aesthetic_column("pos1min"),
                naming::aesthetic_column("pos1max"),
            )
        };

        // First hinge (at min position)
        let mut hinge = layer_spec.clone();
        hinge["mark"] = json!({
            "type": "tick",
            "orient": orient,
            "size": width,
            "thickness": 0,
            "clip": true
        });
        hinge["encoding"][position]["field"] = json!(min_field);
        // Remove x2 and y2 (not needed for tick mark)
        if let Some(e) = hinge["encoding"].as_object_mut() {
            e.remove("x2");
            e.remove("y2");
        }
        layers.push(hinge.clone());

        // Second hinge (at max position) - reuse first hinge and only change position field
        hinge["encoding"][position]["field"] = json!(max_field);
        layers.push(hinge);

        Ok(layers)
    }
}

// =============================================================================
// Boxplot Renderer
// =============================================================================

/// Metadata for boxplot rendering
struct BoxplotMetadata {
    /// Whether there are any outliers
    has_outliers: bool,
}

/// Renderer for boxplot geom - splits into multiple component layers
pub struct BoxplotRenderer;

impl BoxplotRenderer {
    /// Prepare boxplot data by splitting into type-specific datasets.
    ///
    /// Returns a HashMap of type_suffix -> data_values, plus has_outliers flag.
    /// Type suffixes are: "lower_whisker", "upper_whisker", "box", "median", "outlier"
    fn prepare_components(
        &self,
        data: &DataFrame,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<(HashMap<String, Vec<Value>>, bool)> {
        let type_col = naming::aesthetic_column("type");
        let type_col = type_col.as_str();

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

        Ok((type_datasets, has_outliers))
    }

    /// Render boxplot layers using filter transforms on the unified dataset.
    ///
    /// Creates 5 layers: outliers (optional), lower whiskers, upper whiskers, box, median line.
    fn render_layers(
        &self,
        prototype: Value,
        layer: &Layer,
        base_key: &str,
        has_outliers: bool,
    ) -> Result<Vec<Value>> {
        let mut layers: Vec<Value> = Vec::new();

        // Read orientation from layer (already resolved during execution)
        let is_horizontal = is_transposed(layer);

        // Value columns depend on orientation (after DataFrame column flip):
        // - Vertical: values in pos2/pos2end (no flip)
        // - Horizontal: values in pos1/pos1end (was pos2/pos2end before flip)
        let (value_col, value2_col) = if is_horizontal {
            (
                naming::aesthetic_column("pos1"),
                naming::aesthetic_column("pos1end"),
            )
        } else {
            (
                naming::aesthetic_column("pos2"),
                naming::aesthetic_column("pos2end"),
            )
        };

        // Validate x aesthetic exists (required for boxplot)
        layer
            .mappings
            .get("pos1")
            .and_then(|x| x.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Boxplot requires 'x' aesthetic mapping".to_string())
            })?;
        // Validate y aesthetic exists (required for boxplot)
        layer
            .mappings
            .get("pos2")
            .and_then(|y| y.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Boxplot requires 'y' aesthetic mapping".to_string())
            })?;

        let value_var1 = if is_horizontal { "x" } else { "y" };
        let value_var2 = if is_horizontal { "x2" } else { "y2" };
        let axis = if is_horizontal { "y" } else { "x" };

        // Get width parameter
        let width = match layer.adjusted_width {
            // The adjusted width comes from position adjustments
            Some(adjusted) => adjusted,
            _ => match layer.parameters.get("width") {
                // Fallback to width parameter value if there is no adjustment
                Some(ParameterValue::Number(n)) => *n,
                _ => 0.9,
            },
        };
        let width_value = json!({"expr": format!("bandwidth('{}') * {}", axis, width)});

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
            if is_horizontal {
                json!({
                    "type": "bar",
                    "height": width_value,
                    "baseline": "middle"
                })
            } else {
                json!({
                    "type": "bar",
                    "width": width_value,
                    "align": "center"
                })
            },
        );
        box_part["encoding"][value_var1] = y_encoding.clone();
        box_part["encoding"][value_var2] = y2_encoding.clone();

        // Median line (tick at y, where y=median)
        let mut median_line = create_layer(
            &summary_prototype,
            "median",
            if is_horizontal {
                json!({
                    "type": "tick",
                    "height": width_value,
                    "baseline": "middle"
                })
            } else {
                json!({
                    "type": "tick",
                    "width": width_value,
                    "align": "center"
                })
            },
        );
        median_line["encoding"][value_var1] = y_encoding;

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
        let (components, has_outliers) = self.prepare_components(df, binned_columns)?;

        Ok(PreparedData::Composite {
            components,
            metadata: Box::new(BoxplotMetadata { has_outliers }),
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

        self.render_layers(prototype, layer, data_key, info.has_outliers)
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

/// Get the appropriate renderer for a geom type
pub fn get_renderer(geom: &Geom) -> Box<dyn GeomRenderer> {
    match geom.geom_type() {
        GeomType::Path => Box::new(PathRenderer),
        GeomType::Line => Box::new(PathRenderer),
        GeomType::Bar => Box::new(BarRenderer),
        GeomType::Rect => Box::new(RectRenderer),
        GeomType::Ribbon => Box::new(RibbonRenderer),
        GeomType::Polygon => Box::new(PolygonRenderer),
        GeomType::Boxplot => Box::new(BoxplotRenderer),
        GeomType::Violin => Box::new(ViolinRenderer),
        GeomType::Text => Box::new(TextRenderer),
        GeomType::Segment => Box::new(SegmentRenderer),
        GeomType::ErrorBar => Box::new(ErrorBarRenderer),
        GeomType::Rule => Box::new(RuleRenderer),
        // All other geoms (Point, Area, Density, Tile, etc.) use the default renderer
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
        let context = RenderContext::new(&[]);
        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();
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
        let context = RenderContext::new(&[]);
        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();
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
        let context = RenderContext::new(&[]);
        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();
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
        let context = RenderContext::new(&[]);
        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();
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
        let context = RenderContext::new(&[]);
        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();
        assert_eq!(
            encoding.get("detail"),
            Some(&json!([
                {"field": "island", "type": "nominal"},
                {"field": "species", "type": "nominal"}
            ]))
        );
    }

    // =============================================================================
    // RectRenderer Test Helpers
    // =============================================================================

    /// Helper to create a quantitative encoding entry
    fn quant(field: &str) -> Value {
        json!({"field": field, "type": "quantitative"})
    }

    /// Helper to create a nominal encoding entry
    fn nominal(field: &str) -> Value {
        json!({"field": field, "type": "nominal"})
    }

    /// Helper to create a literal value encoding
    fn literal(val: f64) -> Value {
        json!({"value": val})
    }

    /// Helper to create a scale encoding with explicit domain
    /// Use same min/max for constant scales, different values for variable scales
    fn scale(field: &str, min: f64, max: f64) -> Value {
        json!({
            "field": field,
            "type": "quantitative",
            "scale": {
                "domain": [min, max]
            }
        })
    }

    /// Helper to run rect rendering pipeline (modify_encoding + modify_spec)
    fn render_rect(encoding: &mut Map<String, Value>) -> Result<Value> {
        let renderer = RectRenderer;
        let layer = Layer::new(crate::plot::Geom::rect());
        let context = RenderContext::new(&[]);

        renderer.modify_encoding(encoding, &layer, &context)?;

        let mut layer_spec = json!({
            "mark": {"type": "rect", "clip": true},
            "encoding": encoding
        });

        renderer.modify_spec(&mut layer_spec, &layer, &context)?;
        Ok(layer_spec)
    }

    // =============================================================================
    // RectRenderer Tests
    // =============================================================================

    #[test]
    fn test_rect_continuous_both_axes() {
        // Test rect with continuous scales on both axes (xmin/xmax, ymin/ymax)
        // Should remap xmin->x, xmax->x2, ymin->y, ymax->y2
        let mut encoding = serde_json::Map::new();
        encoding.insert("xmin".to_string(), quant("xmin_col"));
        encoding.insert("xmax".to_string(), quant("xmax_col"));
        encoding.insert("ymin".to_string(), quant("ymin_col"));
        encoding.insert("ymax".to_string(), quant("ymax_col"));

        let spec = render_rect(&mut encoding).unwrap();

        // Should remap to x/x2/y/y2
        let enc = spec["encoding"].as_object().unwrap();
        assert_eq!(enc.get("x"), Some(&quant("xmin_col")));
        assert_eq!(enc.get("x2"), Some(&quant("xmax_col")));
        assert_eq!(enc.get("y"), Some(&quant("ymin_col")));
        assert_eq!(enc.get("y2"), Some(&quant("ymax_col")));

        // Original min/max should be removed
        assert!(enc.get("xmin").is_none());
        assert!(enc.get("xmax").is_none());
        assert!(enc.get("ymin").is_none());
        assert!(enc.get("ymax").is_none());

        // Should not have band sizing (both continuous)
        assert!(spec["mark"].get("width").is_none());
        assert!(spec["mark"].get("height").is_none());
    }

    #[test]
    fn test_rect_discrete_x_continuous_y() {
        // Test rect with discrete x scale and continuous y scale
        // x/width (discrete) and ymin/ymax (continuous)
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("width".to_string(), literal(0.8));
        encoding.insert("ymin".to_string(), quant("ymin_col"));
        encoding.insert("ymax".to_string(), quant("ymax_col"));

        let spec = render_rect(&mut encoding).unwrap();
        let enc = spec["encoding"].as_object().unwrap();

        // x should remain as x (discrete)
        assert_eq!(enc.get("x"), Some(&nominal("day")));

        // y should be remapped from ymin/ymax
        assert_eq!(enc.get("y"), Some(&quant("ymin_col")));
        assert_eq!(enc.get("y2"), Some(&quant("ymax_col")));

        // width should be removed from encoding
        assert!(enc.get("width").is_none());

        // Should have mark-level width with band sizing
        assert_eq!(spec["mark"]["width"], json!({"band": 0.8}));
        assert!(spec["mark"].get("height").is_none()); // y is continuous, no height
    }

    #[test]
    fn test_rect_discrete_both_axes_literal_width() {
        // Test rect with discrete scales on both axes with literal width/height
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("width".to_string(), literal(0.7));
        encoding.insert("y".to_string(), nominal("hour"));
        encoding.insert("height".to_string(), literal(0.9));

        let spec = render_rect(&mut encoding).unwrap();
        let enc = spec["encoding"].as_object().unwrap();

        // x and y should remain
        assert_eq!(enc.get("x"), Some(&nominal("day")));
        assert_eq!(enc.get("y"), Some(&nominal("hour")));

        // width/height should be removed from encoding
        assert!(enc.get("width").is_none());
        assert!(enc.get("height").is_none());

        // Should have mark-level width and height with band sizing
        assert_eq!(spec["mark"]["width"], json!({"band": 0.7}));
        assert_eq!(spec["mark"]["height"], json!({"band": 0.9}));
    }

    #[test]
    fn test_rect_discrete_both_axes_default_width() {
        // Test rect with discrete scales on both axes without explicit width/height
        // Should use default band size (1.0)
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("y".to_string(), nominal("hour"));

        let spec = render_rect(&mut encoding).unwrap();

        // With no width/height specified, should have no mark-level width/height
        // (rects will fill the full band by default)
        assert!(spec["mark"].get("width").is_none());
        assert!(spec["mark"].get("height").is_none());
    }

    #[test]
    fn test_rect_discrete_with_field_width() {
        // Test that field-based width on discrete scales uses datum expressions
        // (works for both variable and constant domains, or no domain)
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("width".to_string(), scale("width_col", 0.5, 0.9));

        let spec = render_rect(&mut encoding).unwrap();

        // Should use mark-level width with datum expression
        assert_eq!(
            spec["mark"]["width"],
            json!({"expr": "datum.width_col * bandwidth('x')"})
        );
    }

    #[test]
    fn test_rect_continuous_x_discrete_y() {
        // Test rect with continuous x (xmin/xmax) and discrete y (y/height)
        let mut encoding = serde_json::Map::new();
        encoding.insert("xmin".to_string(), quant("xmin_col"));
        encoding.insert("xmax".to_string(), quant("xmax_col"));
        encoding.insert("y".to_string(), nominal("category"));
        encoding.insert("height".to_string(), literal(0.6));

        let spec = render_rect(&mut encoding).unwrap();
        let enc = spec["encoding"].as_object().unwrap();

        // x should be remapped from xmin/xmax
        assert_eq!(enc.get("x"), Some(&quant("xmin_col")));
        assert_eq!(enc.get("x2"), Some(&quant("xmax_col")));

        // y should remain as y (discrete)
        assert_eq!(enc.get("y"), Some(&nominal("category")));

        // height should be removed from encoding
        assert!(enc.get("height").is_none());

        // Should have mark-level height with band sizing
        assert!(spec["mark"].get("width").is_none()); // x is continuous, no width
        assert_eq!(spec["mark"]["height"], json!({"band": 0.6}));
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
            naming::aesthetic_column("typeface").as_str() => &["Arial", "Arial", "Arial"],
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
            naming::aesthetic_column("typeface").as_str() => &["Arial", "Courier", "Times"],
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
            naming::aesthetic_column("typeface").as_str() => &["Arial", "Courier", "Arial"],
            naming::aesthetic_column("fontweight").as_str() => &["bold", "normal", "bold"],
            naming::aesthetic_column("italic").as_str() => &["false", "true", "false"],
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
        // (Arial+bold+not-italic at index 0, Courier+normal+italic at index 1, Arial+bold+not-italic at index 2)
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
            naming::aesthetic_column("rotation").as_str() => &["0", "45", "90"],
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
            assert!(mark.contains_key("angle")); // Vega-Lite uses "angle" property name
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
            naming::aesthetic_column("rotation").as_str() => &[0i32, 180i32, 0i32],  // integer column
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
                END as rot
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label, rot AS rotation
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
                mark.contains_key("angle"), // Vega-Lite uses "angle" property name
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

        let angle_col = naming::aesthetic_column("rotation");
        for row in data_values {
            assert!(
                row[&angle_col].is_number(),
                "Data row should have numeric angle: {:?}",
                row
            );
        }
    }

    #[test]
    fn test_text_offset_parameters() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: offset parameter should map to xOffset/yOffset

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with offset parameter
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING offset => [5, -10]
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
                "Mark should have xOffset from offset"
            );
            assert_eq!(
                mark["xOffset"].as_f64().unwrap(),
                5.0 * POINTS_TO_PIXELS,
                "xOffset should be 5 * POINTS_TO_PIXELS"
            );

            assert!(
                mark.contains_key("yOffset"),
                "Mark should have yOffset from offset"
            );
            assert_eq!(
                mark["yOffset"].as_f64().unwrap(),
                10.0 * POINTS_TO_PIXELS,
                "yOffset should be 10 * POINTS_TO_PIXELS (negated from offset[1] = -10)"
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
    fn test_text_label_newline_splitting() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Test that labels containing newlines are split into arrays

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with labels containing newlines in DRAW and PLACE
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                CASE
                    WHEN n = 0 THEN 'First Line\nSecond Line'
                    WHEN n = 1 THEN 'Single Line'
                    ELSE 'Line 1\nLine 2\nLine 3'
                END as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text
            PLACE text SETTING x => 5, y => 15, label => 'Annotation\nWith Newline'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        let label_col = crate::naming::aesthetic_column("label");

        // Check first label (contains newline - should be array)
        let label_0 = &data_values[0][&label_col];
        assert!(label_0.is_array(), "Label with newline should be an array");
        let lines_0 = label_0.as_array().unwrap();
        assert_eq!(lines_0.len(), 2);
        assert_eq!(lines_0[0].as_str().unwrap(), "First Line");
        assert_eq!(lines_0[1].as_str().unwrap(), "Second Line");

        // Check second label (no newline - should be string)
        let label_1 = &data_values[1][&label_col];
        assert!(
            label_1.is_string(),
            "Label without newline should be a string"
        );
        assert_eq!(label_1.as_str().unwrap(), "Single Line");

        // Check third label (multiple newlines - should be array with 3 elements)
        let label_2 = &data_values[2][&label_col];
        assert!(label_2.is_array(), "Label with newlines should be an array");
        let lines_2 = label_2.as_array().unwrap();
        assert_eq!(lines_2.len(), 3);
        assert_eq!(lines_2[0].as_str().unwrap(), "Line 1");
        assert_eq!(lines_2[1].as_str().unwrap(), "Line 2");
        assert_eq!(lines_2[2].as_str().unwrap(), "Line 3");

        // Check PLACE annotation layer (index 3, after the 3 DRAW data rows)
        assert!(data_values.len() > 3, "Should have annotation data");
        let annotation_label = &data_values[3][&label_col];
        assert!(
            annotation_label.is_array(),
            "Annotation label with newline should be an array"
        );
        let annotation_lines = annotation_label.as_array().unwrap();
        assert_eq!(annotation_lines.len(), 2);
        assert_eq!(annotation_lines[0].as_str().unwrap(), "Annotation");
        assert_eq!(annotation_lines[1].as_str().unwrap(), "With Newline");
    }

    #[test]
    fn test_text_setting_fontweight() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Integration test: SETTING fontweight => 'bold' should add fontWeight to base mark

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with fontweight in SETTING
        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 'bold'
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
                "Mark should have fontWeight from SETTING fontweight"
            );
            assert_eq!(
                mark["fontWeight"].as_str().unwrap(),
                "bold",
                "fontWeight should be bold"
            );
        }
    }

    #[test]
    fn test_text_setting_fontweight_numeric() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Test numeric fontweight values (700 should map to 'bold')
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 700
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        // Numeric 700 should map to 'bold'
        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "bold");
        }
    }

    #[test]
    fn test_text_setting_fontweight_numeric_normal() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        // Test numeric fontweight values (400 should map to 'normal')
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT
                n::INTEGER as x,
                n::INTEGER as y,
                chr(65 + n::INTEGER) as label
            FROM generate_series(0, 2) as t(n)
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 400
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        // Numeric 400 should map to 'normal'
        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "normal");
        }
    }

    #[test]
    fn test_text_setting_fontweight_keywords() {
        use crate::execute;
        use crate::reader::DuckDBReader;
        use crate::writer::vegalite::VegaLiteWriter;
        use crate::writer::Writer;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Test 'bolder' keyword (should map to 'bold')
        let query = r#"
            SELECT 1 as x, 1 as y, 'A' as label
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 'bolder'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let writer = VegaLiteWriter::new();
        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "bold");
        }

        // Test 'lighter' keyword (should map to 'normal')
        let query = r#"
            SELECT 1 as x, 1 as y, 'A' as label
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 'lighter'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "normal");
        }

        // Test 'semi-bold' keyword (should map to 'bold' since 600 >= 500)
        let query = r#"
            SELECT 1 as x, 1 as y, 'A' as label
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 'semi-bold'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "bold");
        }

        // Test 'light' keyword (should map to 'normal' since 300 < 500)
        let query = r#"
            SELECT 1 as x, 1 as y, 'A' as label
            VISUALISE x, y, label
            DRAW text SETTING fontweight => 'light'
        "#;

        let prepared = execute::prepare_data_with_reader(query, &reader).unwrap();
        let spec = &prepared.specs[0];

        let json_str = writer.write(spec, &prepared.data).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let top_layers = vl_spec["layer"].as_array().unwrap();
        let parent_layer = &top_layers[0];
        let nested_layers = parent_layer["layer"].as_array().unwrap();

        for nested_layer in nested_layers {
            let mark = nested_layer["mark"].as_object().unwrap();
            assert_eq!(mark["fontWeight"].as_str().unwrap(), "normal");
        }
    }

    #[test]
    fn test_fontweight_keyword_to_numeric_conversion() {
        // Test parse_fontweight_to_numeric helper function - all CSS keywords

        // 100 - thin/hairline
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("thin"),
            Some(100.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("hairline"),
            Some(100.0)
        );

        // 200 - extra-light/ultra-light
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("extra-light"),
            Some(200.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("extralight"),
            Some(200.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("ultra-light"),
            Some(200.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("ultralight"),
            Some(200.0)
        );

        // 300 - light
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("light"),
            Some(300.0)
        );

        // 400 - normal/regular/lighter
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("normal"),
            Some(400.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("regular"),
            Some(400.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("lighter"),
            Some(400.0)
        );

        // 500 - medium
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("medium"),
            Some(500.0)
        );

        // 600 - semi-bold/demi-bold
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("semi-bold"),
            Some(600.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("semibold"),
            Some(600.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("demi-bold"),
            Some(600.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("demibold"),
            Some(600.0)
        );

        // 700 - bold/bolder
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("bold"),
            Some(700.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("bolder"),
            Some(700.0)
        );

        // 800 - extra-bold/ultra-bold
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("extra-bold"),
            Some(800.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("extrabold"),
            Some(800.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("ultra-bold"),
            Some(800.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("ultrabold"),
            Some(800.0)
        );

        // 900 - black/heavy
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("black"),
            Some(900.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("heavy"),
            Some(900.0)
        );

        // Case insensitive
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("BOLD"),
            Some(700.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("Normal"),
            Some(400.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("SEMI-BOLD"),
            Some(600.0)
        );

        // Numeric strings pass through
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("100"),
            Some(100.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("400"),
            Some(400.0)
        );
        assert_eq!(
            TextRenderer::parse_fontweight_to_numeric("700"),
            Some(700.0)
        );

        // Invalid values
        assert_eq!(TextRenderer::parse_fontweight_to_numeric("invalid"), None);
        assert_eq!(TextRenderer::parse_fontweight_to_numeric(""), None);
    }

    #[test]
    fn test_violin_mirroring() {
        use crate::naming;

        let renderer = ViolinRenderer;
        let context = RenderContext::new(&[]);

        let layer = Layer::new(crate::plot::Geom::violin());
        let mut layer_spec = json!({
            "mark": {"type": "line"},
            "encoding": {
                "x": {"field": "species", "type": "nominal"},
                "y": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"}
            }
        });

        renderer
            .modify_spec(&mut layer_spec, &layer, &context)
            .unwrap();

        // Verify transforms include mirroring (violin_offsets)
        let transforms = layer_spec["transform"].as_array().unwrap();

        // Find the violin_offsets calculation (mirrors offset on both sides)
        let mirror_calc = transforms
            .iter()
            .find(|t| t.get("as").and_then(|a| a.as_str()) == Some("violin_offsets"));
        assert!(
            mirror_calc.is_some(),
            "Should have violin_offsets mirroring calculation"
        );

        let calc_expr = mirror_calc.unwrap()["calculate"].as_str().unwrap();
        let offset_col = naming::aesthetic_column("offset");
        // Should mirror the offset column: [datum.offset, -datum.offset]
        assert!(
            calc_expr.contains(&offset_col),
            "Mirror calculation should use offset column: {}",
            calc_expr
        );
        assert!(
            calc_expr.contains("-datum"),
            "Mirror calculation should negate: {}",
            calc_expr
        );

        // Verify flatten transform exists
        let flatten = transforms.iter().find(|t| t.get("flatten").is_some());
        assert!(
            flatten.is_some(),
            "Should have flatten transform for violin_offsets"
        );

        // Verify __final_offset calculation (combines with dodge offset)
        let final_offset = transforms
            .iter()
            .find(|t| t.get("as").and_then(|a| a.as_str()) == Some("__final_offset"));
        assert!(
            final_offset.is_some(),
            "Should have __final_offset calculation"
        );
    }

    #[test]
    fn test_violin_ridge_parameter() {
        use crate::naming;
        use crate::plot::ParameterValue;

        let offset_col = naming::aesthetic_column("offset");

        fn get_violin_offset_expr(ridge: Option<&str>, is_horizontal: bool) -> String {
            let mut layer = Layer::new(crate::plot::Geom::violin());
            if let Some(r) = ridge {
                layer
                    .parameters
                    .insert("side".to_string(), ParameterValue::String(r.to_string()));
            }

            // Set orientation parameter for horizontal case
            if is_horizontal {
                layer.parameters.insert(
                    "orientation".to_string(),
                    ParameterValue::String("transposed".to_string()),
                );
            }

            let mut layer_spec = if is_horizontal {
                json!({
                    "mark": {"type": "line"},
                    "encoding": {
                        "x": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"},
                        "y": {"field": "species", "type": "nominal"}
                    }
                })
            } else {
                json!({
                    "mark": {"type": "line"},
                    "encoding": {
                        "x": {"field": "species", "type": "nominal"},
                        "y": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"}
                    }
                })
            };

            ViolinRenderer
                .modify_spec(&mut layer_spec, &layer, &RenderContext::new(&[]))
                .unwrap();

            layer_spec["transform"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t.get("as").and_then(|a| a.as_str()) == Some("violin_offsets"))
                .unwrap()["calculate"]
                .as_str()
                .unwrap()
                .to_string()
        }

        // Default "both" - mirrors on both sides (vertical orientation)
        let expr = get_violin_offset_expr(None, false);
        assert!(
            expr.contains(&format!("[datum.{}, -datum.{}]", offset_col, offset_col))
                || expr.contains(&format!("[-datum.{}, datum.{}]", offset_col, offset_col)),
            "Default should mirror both sides: {}",
            expr
        );

        // Explicit "both" - mirrors on both sides (vertical orientation)
        let expr = get_violin_offset_expr(Some("both"), false);
        assert!(
            expr.contains(&format!("[datum.{}, -datum.{}]", offset_col, offset_col))
                || expr.contains(&format!("[-datum.{}, datum.{}]", offset_col, offset_col)),
            "Explicit 'both' should mirror both sides (vertical): {}",
            expr
        );

        // Explicit "both" - mirrors on both sides (horizontal orientation)
        let expr = get_violin_offset_expr(Some("both"), true);
        assert!(
            expr.contains(&format!("[datum.{}, -datum.{}]", offset_col, offset_col))
                || expr.contains(&format!("[-datum.{}, datum.{}]", offset_col, offset_col)),
            "Explicit 'both' should mirror both sides (horizontal): {}",
            expr
        );

        // Vertical orientation (default): x=nominal, y=quantitative
        // "left" and "bottom" - only negative offset
        assert_eq!(
            get_violin_offset_expr(Some("left"), false),
            format!("[-datum.{}]", offset_col)
        );
        assert_eq!(
            get_violin_offset_expr(Some("bottom"), false),
            format!("[-datum.{}]", offset_col)
        );

        // "right" and "top" - only positive offset
        assert_eq!(
            get_violin_offset_expr(Some("right"), false),
            format!("[datum.{}]", offset_col)
        );
        assert_eq!(
            get_violin_offset_expr(Some("top"), false),
            format!("[datum.{}]", offset_col)
        );

        // Horizontal orientation: x=quantitative, y=nominal
        // "bottom" and "left" - only positive offset
        assert_eq!(
            get_violin_offset_expr(Some("bottom"), true),
            format!("[datum.{}]", offset_col)
        );
        assert_eq!(
            get_violin_offset_expr(Some("left"), true),
            format!("[datum.{}]", offset_col)
        );

        // "top" and "right" - only negative offset
        assert_eq!(
            get_violin_offset_expr(Some("top"), true),
            format!("[-datum.{}]", offset_col)
        );
        assert_eq!(
            get_violin_offset_expr(Some("right"), true),
            format!("[-datum.{}]", offset_col)
        );
    }

    #[test]
    fn test_render_context_get_extent() {
        use crate::plot::{ArrayElement, Scale};

        // Test success case: continuous scale with numeric range
        let scales = vec![Scale {
            aesthetic: "x".to_string(),
            scale_type: None,
            input_range: Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(10.0)]),
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: std::collections::HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: "{}".to_string(),
        }];
        let context = RenderContext::new(&scales);
        let result = context.get_extent("x");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (0.0, 10.0));

        // Test error case: scale not found
        let context = RenderContext::new(&scales);
        let result = context.get_extent("y");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no scale found"));

        // Test error case: scale with no range
        let scales = vec![Scale {
            aesthetic: "x".to_string(),
            scale_type: None,
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: std::collections::HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: "{}".to_string(),
        }];
        let context = RenderContext::new(&scales);
        let result = context.get_extent("x");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no valid numeric range"));

        // Test error case: scale with non-numeric range
        let scales = vec![Scale {
            aesthetic: "x".to_string(),
            scale_type: None,
            input_range: Some(vec![
                ArrayElement::String("A".to_string()),
                ArrayElement::String("B".to_string()),
            ]),
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: std::collections::HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: "{}".to_string(),
        }];
        let context = RenderContext::new(&scales);
        let result = context.get_extent("x");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no valid numeric range"));
    }

    #[test]
    fn test_rule_renderer_multiple_diagonal_lines() {
        use crate::reader::{DuckDBReader, Reader};
        use crate::writer::{VegaLiteWriter, Writer};

        // Test that rule with 3 different slopes renders 3 separate lines
        let query = r#"
            WITH points AS (
                SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) AS t(x, y)
            ),
            lines AS (
                SELECT * FROM (VALUES
                    (2, 5, 'A'),
                    (1, 10, 'B'),
                    (3, 0, 'C')
                ) AS t(slope, y, line_id)
            )
            SELECT * FROM points
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW rule MAPPING slope AS slope, y AS y, line_id AS color FROM lines
        "#;

        // Execute query
        let reader = DuckDBReader::from_connection_string("duckdb://memory")
            .expect("Failed to create reader");
        let spec = reader.execute(query).expect("Failed to execute query");

        // Render to Vega-Lite
        let writer = VegaLiteWriter::new();
        let vl_json = writer.render(&spec).expect("Failed to render spec");

        // Parse JSON
        let vl_spec: serde_json::Value =
            serde_json::from_str(&vl_json).expect("Failed to parse Vega-Lite JSON");

        // Verify we have 2 layers (point + rule)
        let layers = vl_spec["layer"].as_array().expect("No layers found");
        assert_eq!(layers.len(), 2, "Should have 2 layers (point + rule)");

        // Get the rule layer (second layer)
        let rule_layer = &layers[1];

        // Verify it's a rule mark
        assert_eq!(
            rule_layer["mark"]["type"], "rule",
            "Rule should use rule mark"
        );

        // Verify transforms exist
        let transforms = rule_layer["transform"]
            .as_array()
            .expect("No transforms found");

        // Should have 4 calculate transforms + 1 filter = 5 total
        assert_eq!(
            transforms.len(),
            5,
            "Should have 5 transforms (primary_min, primary_max, secondary_min, secondary_max, filter)"
        );

        // Verify primary_min/primary_max transforms exist with consistent naming
        let primary_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_min")
            .expect("primary_min transform not found");
        let primary_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_max")
            .expect("primary_max transform not found");

        assert!(
            primary_min_transform["calculate"].is_string(),
            "primary_min should have calculate expression"
        );
        assert!(
            primary_max_transform["calculate"].is_string(),
            "primary_max should have calculate expression"
        );

        // Verify secondary_min and secondary_max transforms use slope and intercept with primary_min/primary_max
        let secondary_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "secondary_min")
            .expect("secondary_min transform not found");
        let secondary_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "secondary_max")
            .expect("secondary_max transform not found");

        let secondary_min_calc = secondary_min_transform["calculate"]
            .as_str()
            .expect("secondary_min calculate should be string");
        let secondary_max_calc = secondary_max_transform["calculate"]
            .as_str()
            .expect("secondary_max calculate should be string");

        // Should reference slope, pos2 (acting as intercept for y-mapped rules), and primary_min/primary_max
        assert!(
            secondary_min_calc.contains("__ggsql_aes_pos2__"),
            "secondary_min should reference pos2 (y intercept)"
        );
        assert!(
            secondary_min_calc.contains("datum.primary_min"),
            "secondary_min should reference datum.primary_min"
        );
        assert!(
            secondary_max_calc.contains("__ggsql_aes_pos2__"),
            "secondary_max should reference pos2 (y intercept)"
        );
        assert!(
            secondary_max_calc.contains("datum.primary_max"),
            "secondary_max should reference datum.primary_max"
        );

        // Verify encoding has x, x2, y, y2 with consistent field names
        let encoding = rule_layer["encoding"]
            .as_object()
            .expect("No encoding found");

        assert!(encoding.contains_key("x"), "Should have x encoding");
        assert!(encoding.contains_key("x2"), "Should have x2 encoding");
        assert!(encoding.contains_key("y"), "Should have y encoding");
        assert!(encoding.contains_key("y2"), "Should have y2 encoding");

        // Verify consistent naming: primary_min/max for x, secondary_min/max for y (default orientation)
        assert_eq!(
            encoding["x"]["field"], "primary_min",
            "x should reference primary_min field"
        );
        assert_eq!(
            encoding["x2"]["field"], "primary_max",
            "x2 should reference primary_max field"
        );
        assert_eq!(
            encoding["y"]["field"], "secondary_min",
            "y should reference secondary_min field"
        );
        assert_eq!(
            encoding["y2"]["field"], "secondary_max",
            "y2 should reference secondary_max field"
        );

        // Verify stroke encoding exists for line_id (color aesthetic becomes stroke for rule mark)
        assert!(
            encoding.contains_key("stroke"),
            "Should have stroke encoding for line_id"
        );

        // Verify data has 3 rule rows (one per slope)
        let data_values = vl_spec["data"]["values"]
            .as_array()
            .expect("No data values found");

        let rule_rows: Vec<_> = data_values
            .iter()
            .filter(|row| {
                row["__ggsql_source__"] == "__ggsql_layer_1__"
                    && row["__ggsql_aes_slope__"].is_number()
            })
            .collect();

        assert_eq!(
            rule_rows.len(),
            3,
            "Should have 3 rule rows (3 different slopes)"
        );

        // Verify we have slopes 1, 2, 3
        let mut slopes: Vec<f64> = rule_rows
            .iter()
            .map(|row| row["__ggsql_aes_slope__"].as_f64().unwrap())
            .collect();
        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            slopes,
            vec![1.0, 2.0, 3.0],
            "Should have slopes 1, 2, and 3"
        );
    }

    #[test]
    fn test_sloped_rule_renderer_horizontal_orientation() {
        use crate::reader::{DuckDBReader, Reader};
        use crate::writer::{VegaLiteWriter, Writer};

        // Test that sloped rule with x mapping (horizontal) infers y varies
        let query = r#"
            WITH points AS (
                SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) AS t(x, y)
            ),
            lines AS (
                SELECT * FROM (VALUES (0.4, -1, 'A')) AS t(slope, x, line_id)
            )
            SELECT * FROM points
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW rule MAPPING slope AS slope, x AS x, line_id AS color FROM lines
        "#;

        // Execute query
        let reader = DuckDBReader::from_connection_string("duckdb://memory")
            .expect("Failed to create reader");
        let spec = reader.execute(query).expect("Failed to execute query");

        // Render to Vega-Lite
        let writer = VegaLiteWriter::new();
        let vl_json = writer.render(&spec).expect("Failed to render spec");

        // Parse JSON
        let vl_spec: serde_json::Value =
            serde_json::from_str(&vl_json).expect("Failed to parse Vega-Lite JSON");

        // Get the rule layer (second layer)
        let layers = vl_spec["layer"].as_array().expect("No layers found");
        let rule_layer = &layers[1];

        // Verify transforms exist
        let transforms = rule_layer["transform"]
            .as_array()
            .expect("No transforms found");

        // Verify primary_min/max use pos2 extent (y-axis) for horizontal (x-mapped) orientation
        let primary_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_min")
            .expect("primary_min transform not found");
        let primary_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_max")
            .expect("primary_max transform not found");

        // The primary extent should come from the y-axis for horizontal orientation
        assert!(
            primary_min_transform["calculate"].is_string(),
            "primary_min should have calculate expression"
        );
        assert!(
            primary_max_transform["calculate"].is_string(),
            "primary_max should have calculate expression"
        );

        // Verify secondary_min and secondary_max use pos1 (x intercept) for horizontal orientation
        let secondary_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "secondary_min")
            .expect("secondary_min transform not found");
        let secondary_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "secondary_max")
            .expect("secondary_max transform not found");

        let secondary_min_calc = secondary_min_transform["calculate"]
            .as_str()
            .expect("secondary_min calculate should be string");
        let secondary_max_calc = secondary_max_transform["calculate"]
            .as_str()
            .expect("secondary_max calculate should be string");

        // Should reference pos1 (x intercept) for horizontal orientation
        assert!(
            secondary_min_calc.contains("__ggsql_aes_pos1__"),
            "secondary_min should reference pos1 (x intercept)"
        );
        assert!(
            secondary_max_calc.contains("__ggsql_aes_pos1__"),
            "secondary_max should reference pos1 (x intercept)"
        );

        // Verify encoding has y as primary axis (mapped to primary_min/max)
        let encoding = rule_layer["encoding"]
            .as_object()
            .expect("No encoding found");

        // For horizontal orientation (x-mapped): y is primary (uses primary_min/max), x is secondary
        assert_eq!(
            encoding["y"]["field"], "primary_min",
            "y should reference primary_min field for horizontal orientation"
        );
        assert_eq!(
            encoding["y2"]["field"], "primary_max",
            "y2 should reference primary_max field for horizontal orientation"
        );
        assert_eq!(
            encoding["x"]["field"], "secondary_min",
            "x should reference secondary_min field for horizontal orientation"
        );
        assert_eq!(
            encoding["x2"]["field"], "secondary_max",
            "x2 should reference secondary_max field for horizontal orientation"
        );
    }

    #[test]
    fn test_errorbar_encoding() {
        let renderer = ErrorBarRenderer;
        let layer = Layer::new(crate::plot::Geom::errorbar());
        let context = RenderContext::new(&[]);

        // Case 1: Vertical errorbar (x + ymin + ymax)
        // Should map ymax → y and ymin → y2
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "ymin".to_string(),
            json!({"field": "low", "type": "quantitative"}),
        );
        encoding.insert(
            "ymax".to_string(),
            json!({"field": "high", "type": "quantitative"}),
        );

        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();

        assert_eq!(
            encoding.get("y"),
            Some(&json!({"field": "high", "type": "quantitative"})),
            "ymax should be mapped to y"
        );
        assert_eq!(
            encoding.get("y2"),
            Some(&json!({"field": "low", "type": "quantitative"})),
            "ymin should be mapped to y2"
        );
        assert!(!encoding.contains_key("ymin"), "ymin should be removed");
        assert!(!encoding.contains_key("ymax"), "ymax should be removed");

        // Case 2: Horizontal errorbar (y + xmin + xmax)
        // Should map xmax → x and xmin → x2
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "y".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "xmin".to_string(),
            json!({"field": "low", "type": "quantitative"}),
        );
        encoding.insert(
            "xmax".to_string(),
            json!({"field": "high", "type": "quantitative"}),
        );

        renderer
            .modify_encoding(&mut encoding, &layer, &context)
            .unwrap();

        assert_eq!(
            encoding.get("x"),
            Some(&json!({"field": "high", "type": "quantitative"})),
            "xmax should be mapped to x"
        );
        assert_eq!(
            encoding.get("x2"),
            Some(&json!({"field": "low", "type": "quantitative"})),
            "xmin should be mapped to x2"
        );
        assert!(!encoding.contains_key("xmin"), "xmin should be removed");
        assert!(!encoding.contains_key("xmax"), "xmax should be removed");

        // Case 3: Error - neither x nor y is present
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "xmin".to_string(),
            json!({"field": "low", "type": "quantitative"}),
        );
        encoding.insert(
            "xmax".to_string(),
            json!({"field": "high", "type": "quantitative"}),
        );

        let result = renderer.modify_encoding(&mut encoding, &layer, &context);
        assert!(
            result.is_err(),
            "Should error when neither x nor y is present"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("aesthetics are incomplete"),
            "Error message should mention incomplete aesthetics"
        );

        // Case 4: Error - both x and y present
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "x_col", "type": "quantitative"}),
        );
        encoding.insert(
            "y".to_string(),
            json!({"field": "y_col", "type": "quantitative"}),
        );

        let result = renderer.modify_encoding(&mut encoding, &layer, &context);
        assert!(
            result.is_err(),
            "Should error when both x and y are present"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mutually exclusive"),
            "Error message should mention mutual exclusivity"
        );

        // Case 5: Error - x with xmin/xmax
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "x".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "xmin".to_string(),
            json!({"field": "low", "type": "quantitative"}),
        );
        encoding.insert(
            "xmax".to_string(),
            json!({"field": "high", "type": "quantitative"}),
        );

        let result = renderer.modify_encoding(&mut encoding, &layer, &context);
        assert!(
            result.is_err(),
            "Should error when x is used with xmin/xmax"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot use `x` aesthetic with `xmin` and `xmax`"),
            "Error message should mention conflicting aesthetics"
        );

        // Case 6: Error - y with ymin/ymax
        let mut encoding = serde_json::Map::new();
        encoding.insert(
            "y".to_string(),
            json!({"field": "species", "type": "nominal"}),
        );
        encoding.insert(
            "ymin".to_string(),
            json!({"field": "low", "type": "quantitative"}),
        );
        encoding.insert(
            "ymax".to_string(),
            json!({"field": "high", "type": "quantitative"}),
        );

        let result = renderer.modify_encoding(&mut encoding, &layer, &context);
        assert!(
            result.is_err(),
            "Should error when y is used with ymin/ymax"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot use `y` aesthetic with `ymin` and `ymax`"),
            "Error message should mention conflicting aesthetics"
        );
    }

    #[test]
    fn test_path_renderer_varying_aesthetics_metadata() {
        use crate::plot::{AestheticValue, Geom, Layer};
        use polars::prelude::*;

        let renderer = PathRenderer;
        let mut layer = Layer::new(Geom::line());

        // Create DataFrame with varying stroke
        let df = df! {
            naming::aesthetic_column("pos1").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("pos2").as_str() => &[10.0, 20.0, 30.0],
            "color".to_string().as_str() => &[1.0, 2.0, 3.0],
        }
        .unwrap();

        // Map stroke to color column (continuous, not in partition_by)
        layer.mappings.insert(
            "stroke".to_string(),
            AestheticValue::standard_column("color"),
        );

        // Prepare data - should detect varying stroke
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        match prepared {
            PreparedData::Single { metadata, .. } => {
                let varying_aesthetics = metadata
                    .downcast_ref::<Vec<&'static str>>()
                    .expect("Metadata should be Vec<&str>");
                assert_eq!(varying_aesthetics.len(), 1);
                assert!(varying_aesthetics.contains(&"stroke"));
            }
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_path_renderer_trail_mark_for_varying_linewidth() {
        use crate::plot::{AestheticValue, Geom, Layer};
        use polars::prelude::*;

        let renderer = PathRenderer;
        let mut layer = Layer::new(Geom::line());

        // Create DataFrame with varying linewidth
        let df = df! {
            naming::aesthetic_column("pos1").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("pos2").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("linewidth").as_str() => &[1.0, 3.0, 5.0],
        }
        .unwrap();

        // Map linewidth to column
        layer.mappings.insert(
            "linewidth".to_string(),
            AestheticValue::standard_column(naming::aesthetic_column("linewidth")),
        );

        // Prepare data
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        // Create a mock layer spec
        let layer_spec = json!({
            "mark": {"type": "line", "clip": true},
            "encoding": {
                "x": {"field": naming::aesthetic_column("pos1"), "type": "quantitative"},
                "y": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"},
                "strokeWidth": {"field": naming::aesthetic_column("linewidth"), "type": "quantitative"}
            }
        });

        // Finalize should switch to trail mark and translate encodings
        let result = renderer
            .finalize(layer_spec.clone(), &layer, "test", &prepared)
            .unwrap();

        assert_eq!(result.len(), 1);
        let spec = &result[0];

        // Check mark type is trail
        assert_eq!(spec["mark"]["type"], "trail");
        assert_eq!(spec["mark"]["strokeWidth"], 0);

        // Check encoding translations
        let encoding = spec["encoding"].as_object().unwrap();
        assert!(encoding.contains_key("size"), "Should have size encoding");
        assert!(
            !encoding.contains_key("strokeWidth"),
            "strokeWidth should be removed"
        );
        // No stroke mapping in this test, so no fill expected
        assert!(!encoding.contains_key("stroke"), "stroke should be removed");
    }

    #[test]
    fn test_path_renderer_trail_mark_with_stroke_legend() {
        use crate::plot::{AestheticValue, Geom, Layer};
        use polars::prelude::*;

        let renderer = PathRenderer;
        let mut layer = Layer::new(Geom::line());

        // Create DataFrame with varying linewidth and stroke
        let df = df! {
            naming::aesthetic_column("pos1").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("pos2").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("linewidth").as_str() => &[1.0, 3.0, 5.0],
            naming::aesthetic_column("stroke").as_str() => &["A", "A", "B"],
        }
        .unwrap();

        // Map linewidth and stroke to columns
        layer.mappings.insert(
            "linewidth".to_string(),
            AestheticValue::standard_column(naming::aesthetic_column("linewidth")),
        );
        layer.mappings.insert(
            "stroke".to_string(),
            AestheticValue::standard_column(naming::aesthetic_column("stroke")),
        );

        // Prepare data
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        // Create a mock layer spec with stroke legend
        let layer_spec = json!({
            "mark": {"type": "line", "clip": true},
            "encoding": {
                "x": {"field": naming::aesthetic_column("pos1"), "type": "quantitative"},
                "y": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"},
                "strokeWidth": {"field": naming::aesthetic_column("linewidth"), "type": "quantitative"},
                "stroke": {
                    "field": naming::aesthetic_column("stroke"),
                    "type": "nominal",
                    "legend": {
                        "title": "direction"
                    }
                }
            }
        });

        // Finalize should switch to trail mark and translate encodings
        let result = renderer
            .finalize(layer_spec.clone(), &layer, "test", &prepared)
            .unwrap();

        assert_eq!(result.len(), 1);
        let spec = &result[0];

        // Check mark type is trail
        assert_eq!(spec["mark"]["type"], "trail");
        assert_eq!(spec["mark"]["strokeWidth"], 0);

        // Check encoding translations
        let encoding = spec["encoding"].as_object().unwrap();
        assert!(encoding.contains_key("size"), "Should have size encoding");
        assert!(encoding.contains_key("fill"), "Should have fill encoding");
        assert!(!encoding.contains_key("stroke"), "stroke should be removed");

        // Check that fill legend has symbolStrokeColor
        let fill = &encoding["fill"];
        assert!(fill["legend"].is_object(), "fill should have legend");
        let legend = fill["legend"].as_object().unwrap();
        assert!(
            legend.contains_key("symbolStrokeColor"),
            "fill legend should have symbolStrokeColor"
        );
        assert_eq!(
            legend["symbolStrokeColor"]["expr"],
            "scale('fill', datum.value)",
            "symbolStrokeColor should use fill scale"
        );
    }

    #[test]
    fn test_path_renderer_segmentation_for_varying_stroke() {
        use crate::plot::{AestheticValue, Geom, Layer};
        use polars::prelude::*;

        let renderer = PathRenderer;
        let mut layer = Layer::new(Geom::line());

        // Create DataFrame with varying stroke
        let df = df! {
            naming::aesthetic_column("pos1").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("pos2").as_str() => &[10.0, 20.0, 30.0],
            "color".to_string().as_str() => &[1.0, 2.0, 3.0],
            ROW_INDEX_COLUMN => &[0, 1, 2],
        }
        .unwrap();

        // Map stroke to color column
        layer.mappings.insert(
            "stroke".to_string(),
            AestheticValue::standard_column("color"),
        );

        // Prepare data
        let prepared = renderer
            .prepare_data(&df, &layer, "test", &HashMap::new())
            .unwrap();

        // Create a mock layer spec
        let layer_spec = json!({
            "mark": {"type": "line", "clip": true},
            "encoding": {
                "x": {"field": naming::aesthetic_column("pos1"), "type": "quantitative"},
                "y": {"field": naming::aesthetic_column("pos2"), "type": "quantitative"},
                "stroke": {"field": "color", "type": "nominal"}
            }
        });

        // Finalize should apply segmentation transforms
        let result = renderer
            .finalize(layer_spec.clone(), &layer, "test", &prepared)
            .unwrap();

        assert_eq!(result.len(), 1);
        let spec = &result[0];

        // Check transforms exist
        let transforms = spec["transform"]
            .as_array()
            .expect("Should have transforms");
        assert!(!transforms.is_empty());

        // Check for window transform (lead operation)
        let has_window = transforms.iter().any(|t| t.get("window").is_some());
        assert!(has_window, "Should have window transform for lead");

        // Check for flatten transform
        let has_flatten = transforms.iter().any(|t| t.get("flatten").is_some());
        assert!(has_flatten, "Should have flatten transform");

        // Check for detail encoding with segment_id
        let encoding = spec["encoding"].as_object().unwrap();
        assert!(
            encoding.contains_key("detail"),
            "Should have detail encoding"
        );
        assert_eq!(
            encoding["detail"]["field"], "__segment_id__",
            "Detail should use segment_id"
        );

        // Check that x/y use _final fields
        assert!(
            encoding["x"]["field"].as_str().unwrap().ends_with("_final"),
            "x should use _final field"
        );
        assert!(
            encoding["y"]["field"].as_str().unwrap().ends_with("_final"),
            "y should use _final field"
        );
    }
}
