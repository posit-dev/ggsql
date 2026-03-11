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
        GeomType::Tile => "rect",
        GeomType::Ribbon => "area",
        GeomType::Polygon => "line",
        GeomType::Histogram => "bar",
        GeomType::Density => "area",
        GeomType::Violin => "line",
        GeomType::Boxplot => "boxplot",
        GeomType::Text => "text",
        GeomType::Label => "text",
        GeomType::Segment => "rule",
        GeomType::Rule => "rule",
        GeomType::Linear => "rule",
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
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
        _layer: &Layer,
        _context: &RenderContext,
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
        let width = match layer.parameters.get("width") {
            Some(ParameterValue::Number(w)) => *w,
            _ => 0.9,
        };

        // For horizontal bars, use "height" for band size; for vertical, use "width"
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

        // For dodged bars, use expression-based size with the adjusted width
        // For non-dodged bars, use band-relative size
        let size_value = if let Some(adjusted) = layer.adjusted_width {
            // Use bandwidth expression for dodged bars
            let axis = if is_horizontal { "y" } else { "x" };
            json!({"expr": format!("bandwidth('{}') * {}", axis, adjusted)})
        } else {
            json!({"band": width})
        };

        layer_spec["mark"] = if is_horizontal {
            json!({
                "type": "bar",
                "height": size_value,
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

/// Renderer for path geom - adds order channel for natural data order
pub struct PathRenderer;

impl GeomRenderer for PathRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Use row index field to preserve natural data order
        encoding.insert(
            "order".to_string(),
            json!({"field": ROW_INDEX_COLUMN, "type": "quantitative"}),
        );
        Ok(())
    }
}

// =============================================================================
// Line Renderer
// =============================================================================

/// Renderer for line geom - preserves data order for correct line rendering
pub struct LineRenderer;

impl GeomRenderer for LineRenderer {
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
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        let has_x = encoding.contains_key("x");
        let has_y = encoding.contains_key("y");
        if !has_x && !has_y {
            return Err(GgsqlError::ValidationError(
                "The `rule` layer requires the `x` or `y` aesthetic. It currently has neither."
                    .to_string(),
            ));
        } else if has_x && has_y {
            return Err(GgsqlError::ValidationError(
                "The `rule` layer requires exactly one of the `x` or `y` aesthetic, not both."
                    .to_string(),
            ));
        }
        Ok(())
    }
}

// =============================================================================
// Linear Renderer
// =============================================================================

/// Renderer for linear geom - draws lines based on coefficient and intercept
pub struct LinearRenderer;

impl GeomRenderer for LinearRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        _data_key: &str,
        _binned_columns: &HashMap<String, Vec<f64>>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<PreparedData> {
        // Just convert DataFrame to JSON values
        // No need to add xmin/xmax - they'll be encoded as literal values
        let values = dataframe_to_values(df)?;
        Ok(PreparedData::Single { values })
    }

    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Remove coefficient and intercept from encoding - they're only used in transforms
        encoding.remove("coef");
        encoding.remove("intercept");

        // Check orientation
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

        // For aligned (default): x is primary axis, y is computed (secondary)
        // For transposed: y is primary axis, x is computed (secondary)
        let (primary, primary2, secondary, secondary2) = if is_horizontal {
            ("y", "y2", "x", "x2")
        } else {
            ("x", "x2", "y", "y2")
        };

        // Add encodings for rule mark
        // primary_min/primary_max are created by transforms (extent of the axis)
        // secondary_min/secondary_max are computed via formula
        encoding.insert(
            primary.to_string(),
            json!({
                "field": "primary_min",
                "type": "quantitative"
            }),
        );
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
        // Field names for coef and intercept (with aesthetic column prefix)
        let coef_field = naming::aesthetic_column("coef");
        let intercept_field = naming::aesthetic_column("intercept");

        // Check orientation
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

        // Get extent from appropriate axis:
        // - Aligned (default): extent from pos1 (x-axis), compute y from x
        // - Transposed: extent from pos2 (y-axis), compute x from y
        let extent_aesthetic = if is_horizontal { "pos2" } else { "pos1" };
        let (primary_min, primary_max) = context.get_extent(extent_aesthetic)?;

        // Add transforms:
        // 1. Create constant primary_min/primary_max fields (extent of the primary axis)
        // 2. Compute secondary values at those primary positions: secondary = coef * primary + intercept
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
                "calculate": format!("datum.{} * datum.primary_min + datum.{}", coef_field, intercept_field),
                "as": "secondary_min"
            },
            {
                "calculate": format!("datum.{} * datum.primary_max + datum.{}", coef_field, intercept_field),
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
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

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

        // It'll be implemented as an offset.
        let violin_offset = format!("[datum.{offset}, -datum.{offset}]", offset = offset_col);

        // Read orientation from layer (already resolved during execution)
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

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

        // Filter threshold to trim very low density regions (removes thin tails)
        // The offset is pre-scaled to [0, 0.5 * width] by geom post_process,
        // but this filter still catches extremely low values.
        let filter_expr = format!("datum.{} > 0.001", offset_col);

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
                // Remove points with very low density to clean up thin tails
                "filter": filter_expr
            }),
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
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

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
        let is_vertical = layer_spec["encoding"]["x2"].is_null();
        let (orient, position, min_field, max_field) = if is_vertical {
            (
                "horizontal",
                "y",
                naming::aesthetic_column("ymin"),
                naming::aesthetic_column("ymax"),
            )
        } else {
            (
                "vertical",
                "x",
                naming::aesthetic_column("xmin"),
                naming::aesthetic_column("xmax"),
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
        let is_horizontal = layer
            .parameters
            .get("orientation")
            .and_then(|v| v.as_str())
            .map(|s| s == "transposed")
            .unwrap_or(false);

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

        // Get width parameter
        let base_width = layer
            .parameters
            .get("width")
            .and_then(|v| match v {
                ParameterValue::Number(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0.9);

        // For dodged boxplots, use expression-based width with adjusted_width
        // For non-dodged boxplots, use band-relative width
        let axis = if is_horizontal { "y" } else { "x" };
        let width_value = if let Some(adjusted) = layer.adjusted_width {
            json!({"expr": format!("bandwidth('{}') * {}", axis, adjusted)})
        } else {
            json!({"band": base_width})
        };

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
            json!({
                "type": "bar",
                "width": width_value,
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
                "width": width_value,
                "align": "center"
            }),
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
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
        _layer: &Layer,
        _context: &RenderContext,
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
        GeomType::Line => Box::new(LineRenderer),
        GeomType::Bar => Box::new(BarRenderer),
        GeomType::Ribbon => Box::new(RibbonRenderer),
        GeomType::Polygon => Box::new(PolygonRenderer),
        GeomType::Boxplot => Box::new(BoxplotRenderer),
        GeomType::Violin => Box::new(ViolinRenderer),
        GeomType::Segment => Box::new(SegmentRenderer),
        GeomType::Linear => Box::new(LinearRenderer),
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
    fn test_linear_renderer_multiple_lines() {
        use crate::reader::{DuckDBReader, Reader};
        use crate::writer::{VegaLiteWriter, Writer};

        // Test that linear with 3 different coefficients renders 3 separate lines
        let query = r#"
            WITH points AS (
                SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) AS t(x, y)
            ),
            lines AS (
                SELECT * FROM (VALUES
                    (2, 5, 'A'),
                    (1, 10, 'B'),
                    (3, 0, 'C')
                ) AS t(coef, intercept, line_id)
            )
            SELECT * FROM points
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW linear MAPPING coef AS coef, intercept AS intercept, line_id AS color FROM lines
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

        // Verify we have 2 layers (point + linear)
        let layers = vl_spec["layer"].as_array().expect("No layers found");
        assert_eq!(layers.len(), 2, "Should have 2 layers (point + linear)");

        // Get the linear layer (second layer)
        let linear_layer = &layers[1];

        // Verify it's a rule mark
        assert_eq!(
            linear_layer["mark"]["type"], "rule",
            "Linear should use rule mark"
        );

        // Verify transforms exist
        let transforms = linear_layer["transform"]
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

        // Verify secondary_min and secondary_max transforms use coef and intercept with primary_min/primary_max
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

        // Should reference coef, intercept, and primary_min/primary_max
        assert!(
            secondary_min_calc.contains("__ggsql_aes_coef__"),
            "secondary_min should reference coef"
        );
        assert!(
            secondary_min_calc.contains("__ggsql_aes_intercept__"),
            "secondary_min should reference intercept"
        );
        assert!(
            secondary_min_calc.contains("datum.primary_min"),
            "secondary_min should reference datum.primary_min"
        );
        assert!(
            secondary_max_calc.contains("__ggsql_aes_coef__"),
            "secondary_max should reference coef"
        );
        assert!(
            secondary_max_calc.contains("__ggsql_aes_intercept__"),
            "secondary_max should reference intercept"
        );
        assert!(
            secondary_max_calc.contains("datum.primary_max"),
            "secondary_max should reference datum.primary_max"
        );

        // Verify encoding has x, x2, y, y2 with consistent field names
        let encoding = linear_layer["encoding"]
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

        // Verify data has 3 linear rows (one per coef)
        let data_values = vl_spec["data"]["values"]
            .as_array()
            .expect("No data values found");

        let linear_rows: Vec<_> = data_values
            .iter()
            .filter(|row| {
                row["__ggsql_source__"] == "__ggsql_layer_1__"
                    && row["__ggsql_aes_coef__"].is_number()
            })
            .collect();

        assert_eq!(
            linear_rows.len(),
            3,
            "Should have 3 linear rows (3 different coefficients)"
        );

        // Verify we have coefs 1, 2, 3
        let mut coefs: Vec<f64> = linear_rows
            .iter()
            .map(|row| row["__ggsql_aes_coef__"].as_f64().unwrap())
            .collect();
        coefs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(coefs, vec![1.0, 2.0, 3.0], "Should have coefs 1, 2, and 3");
    }

    #[test]
    fn test_linear_renderer_transposed_orientation() {
        use crate::reader::{DuckDBReader, Reader};
        use crate::writer::{VegaLiteWriter, Writer};

        // Test that linear with transposed orientation swaps x/y axes
        let query = r#"
            WITH points AS (
                SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) AS t(x, y)
            ),
            lines AS (
                SELECT * FROM (VALUES (0.4, -1, 'A')) AS t(coef, intercept, line_id)
            )
            SELECT * FROM points
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW linear MAPPING coef AS coef, intercept AS intercept, line_id AS color FROM lines SETTING orientation => 'transposed'
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

        // Get the linear layer (second layer)
        let layers = vl_spec["layer"].as_array().expect("No layers found");
        let linear_layer = &layers[1];

        // Verify transforms exist
        let transforms = linear_layer["transform"]
            .as_array()
            .expect("No transforms found");

        // Verify primary_min/max use pos2 extent (y-axis) for transposed orientation
        let primary_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_min")
            .expect("primary_min transform not found");
        let primary_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "primary_max")
            .expect("primary_max transform not found");

        // The primary extent should come from the y-axis for transposed
        assert!(
            primary_min_transform["calculate"].is_string(),
            "primary_min should have calculate expression"
        );
        assert!(
            primary_max_transform["calculate"].is_string(),
            "primary_max should have calculate expression"
        );

        // Verify encoding has y as primary axis (mapped to primary_min/max)
        let encoding = linear_layer["encoding"]
            .as_object()
            .expect("No encoding found");

        // For transposed orientation: y is primary (uses primary_min/max), x is secondary
        assert_eq!(
            encoding["y"]["field"], "primary_min",
            "y should reference primary_min field for transposed"
        );
        assert_eq!(
            encoding["y2"]["field"], "primary_max",
            "y2 should reference primary_max field for transposed"
        );
        assert_eq!(
            encoding["x"]["field"], "secondary_min",
            "x should reference secondary_min field for transposed"
        );
        assert_eq!(
            encoding["x2"]["field"], "secondary_max",
            "x2 should reference secondary_max field for transposed"
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
}
