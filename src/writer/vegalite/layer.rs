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
        GeomType::Segment => "rule",
        GeomType::HLine => "rule",
        GeomType::VLine => "rule",
        GeomType::AbLine => "rule",
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
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Use the natural data order
        encoding.insert("order".to_string(), json!({"value": Value::Null}));
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
// ABLine Renderer
// =============================================================================

/// Renderer for abline geom - draws lines based on slope and intercept
pub struct ABLineRenderer;

impl GeomRenderer for ABLineRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        _data_key: &str,
        _binned_columns: &HashMap<String, Vec<f64>>,
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
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
        // Remove slope and intercept from encoding - they're only used in transforms
        encoding.remove("slope");
        encoding.remove("intercept");

        // Add x/x2/y/y2 encodings for rule mark
        // All are fields - x_min/x_max are created by transforms, y_min/y_max are computed
        encoding.insert(
            "x".to_string(),
            json!({
                "field": "x_min",
                "type": "quantitative"
            }),
        );
        encoding.insert(
            "x2".to_string(),
            json!({
                "field": "x_max"
            }),
        );
        encoding.insert(
            "y".to_string(),
            json!({
                "field": "y_min",
                "type": "quantitative"
            }),
        );
        encoding.insert(
            "y2".to_string(),
            json!({
                "field": "y_max"
            }),
        );

        Ok(())
    }

    fn modify_spec(
        &self,
        layer_spec: &mut Value,
        _layer: &Layer,
        context: &RenderContext,
    ) -> Result<()> {
        // Field names for slope and intercept (with aesthetic column prefix)
        let slope_field = naming::aesthetic_column("slope");
        let intercept_field = naming::aesthetic_column("intercept");

        // Get x extent from scale
        let (x_min, x_max) = context.get_extent("x")?;

        // Add transforms:
        // 1. Create constant x_min/x_max fields
        // 2. Compute y values at those x positions
        let transforms = json!([
            {
                "calculate": x_min.to_string(),
                "as": "x_min"
            },
            {
                "calculate": x_max.to_string(),
                "as": "x_max"
            },
            {
                "calculate": format!("datum.{} * datum.x_min + datum.{}", slope_field, intercept_field),
                "as": "y_min"
            },
            {
                "calculate": format!("datum.{} * datum.x_max + datum.{}", slope_field, intercept_field),
                "as": "y_max"
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

/// Renderer for ribbon geom - remaps ymin/ymax to y/y2
pub struct RibbonRenderer;

impl GeomRenderer for RibbonRenderer {
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
        // Use the natural data order
        encoding.insert("order".to_string(), json!({"value": Value::Null}));
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
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
            y = naming::aesthetic_column("y")
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

    fn modify_encoding(
        &self,
        encoding: &mut Map<String, Value>,
        _layer: &Layer,
        _context: &RenderContext,
    ) -> Result<()> {
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
        let value_col = naming::aesthetic_column("y");
        let value_col = value_col.as_str();
        let value2_col = naming::aesthetic_column("yend");
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

        let value_col = naming::aesthetic_column("y");
        let value2_col = naming::aesthetic_column("yend");

        let x_col = layer
            .mappings
            .get("x")
            .and_then(|x| x.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Failed to find column for 'x' aesthetic".to_string())
            })?;
        let y_col = layer
            .mappings
            .get("y")
            .and_then(|y| y.column_name())
            .ok_or_else(|| {
                GgsqlError::WriterError("Failed to find column for 'y' aesthetic".to_string())
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
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
        _context: &RenderContext,
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
        GeomType::Segment => Box::new(SegmentRenderer),
        GeomType::AbLine => Box::new(ABLineRenderer),
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
    fn test_abline_renderer_multiple_lines() {
        use crate::reader::{DuckDBReader, Reader};
        use crate::writer::{VegaLiteWriter, Writer};

        // Test that abline with 3 different slopes renders 3 separate lines
        let query = r#"
            WITH points AS (
                SELECT * FROM (VALUES (0, 5), (5, 15), (10, 25)) AS t(x, y)
            ),
            lines AS (
                SELECT * FROM (VALUES
                    (2, 5, 'A'),
                    (1, 10, 'B'),
                    (3, 0, 'C')
                ) AS t(slope, intercept, line_id)
            )
            SELECT * FROM points
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW abline MAPPING slope AS slope, intercept AS intercept, line_id AS color FROM lines
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

        // Verify we have 2 layers (point + abline)
        let layers = vl_spec["layer"].as_array().expect("No layers found");
        assert_eq!(layers.len(), 2, "Should have 2 layers (point + abline)");

        // Get the abline layer (second layer)
        let abline_layer = &layers[1];

        // Verify it's a rule mark
        assert_eq!(
            abline_layer["mark"]["type"], "rule",
            "ABLine should use rule mark"
        );

        // Verify transforms exist
        let transforms = abline_layer["transform"]
            .as_array()
            .expect("No transforms found");

        // Should have 4 calculate transforms + 1 filter = 5 total
        assert_eq!(
            transforms.len(),
            5,
            "Should have 5 transforms (x_min, x_max, y_min, y_max, filter)"
        );

        // Verify x_min/x_max transforms exist with consistent naming
        let x_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "x_min")
            .expect("x_min transform not found");
        let x_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "x_max")
            .expect("x_max transform not found");

        assert!(
            x_min_transform["calculate"].is_string(),
            "x_min should have calculate expression"
        );
        assert!(
            x_max_transform["calculate"].is_string(),
            "x_max should have calculate expression"
        );

        // Verify y_min and y_max transforms use slope and intercept with x_min/x_max
        let y_min_transform = transforms
            .iter()
            .find(|t| t["as"] == "y_min")
            .expect("y_min transform not found");
        let y_max_transform = transforms
            .iter()
            .find(|t| t["as"] == "y_max")
            .expect("y_max transform not found");

        let y_min_calc = y_min_transform["calculate"]
            .as_str()
            .expect("y_min calculate should be string");
        let y_max_calc = y_max_transform["calculate"]
            .as_str()
            .expect("y_max calculate should be string");

        // Should reference slope, intercept, and x_min/x_max
        assert!(
            y_min_calc.contains("__ggsql_aes_slope__"),
            "y_min should reference slope"
        );
        assert!(
            y_min_calc.contains("__ggsql_aes_intercept__"),
            "y_min should reference intercept"
        );
        assert!(
            y_min_calc.contains("datum.x_min"),
            "y_min should reference datum.x_min"
        );
        assert!(
            y_max_calc.contains("__ggsql_aes_slope__"),
            "y_max should reference slope"
        );
        assert!(
            y_max_calc.contains("__ggsql_aes_intercept__"),
            "y_max should reference intercept"
        );
        assert!(
            y_max_calc.contains("datum.x_max"),
            "y_max should reference datum.x_max"
        );

        // Verify encoding has x, x2, y, y2 with consistent field names
        let encoding = abline_layer["encoding"]
            .as_object()
            .expect("No encoding found");

        assert!(encoding.contains_key("x"), "Should have x encoding");
        assert!(encoding.contains_key("x2"), "Should have x2 encoding");
        assert!(encoding.contains_key("y"), "Should have y encoding");
        assert!(encoding.contains_key("y2"), "Should have y2 encoding");

        // Verify consistent naming: x_min, x_max, y_min, y_max
        assert_eq!(
            encoding["x"]["field"], "x_min",
            "x should reference x_min field"
        );
        assert_eq!(
            encoding["x2"]["field"], "x_max",
            "x2 should reference x_max field"
        );
        assert_eq!(
            encoding["y"]["field"], "y_min",
            "y should reference y_min field"
        );
        assert_eq!(
            encoding["y2"]["field"], "y_max",
            "y2 should reference y_max field"
        );

        // Verify stroke encoding exists for line_id (color aesthetic becomes stroke for rule mark)
        assert!(
            encoding.contains_key("stroke"),
            "Should have stroke encoding for line_id"
        );

        // Verify data has 3 abline rows (one per slope)
        let data_values = vl_spec["data"]["values"]
            .as_array()
            .expect("No data values found");

        let abline_rows: Vec<_> = data_values
            .iter()
            .filter(|row| {
                row["__ggsql_source__"] == "__ggsql_layer_1__"
                    && row["__ggsql_aes_slope__"].is_number()
            })
            .collect();

        assert_eq!(
            abline_rows.len(),
            3,
            "Should have 3 abline rows (3 different slopes)"
        );

        // Verify we have slopes 1, 2, 3
        let mut slopes: Vec<f64> = abline_rows
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
}
