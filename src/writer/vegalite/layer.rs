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
        GeomType::Rect => "rect",
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
// Rect Renderer
// =============================================================================

/// Renderer for rect geom - handles continuous and discrete rectangles
///
/// For continuous scales: remaps xmin/xmax → x/x2, ymin/ymax → y/y2
/// For discrete scales: keeps x/y as-is and applies width/height as band fractions
pub struct RectRenderer;

impl RectRenderer {
    /// Extract and remove band size (width/height) from encoding for discrete scales.
    /// Should only be called for discrete scales.
    fn extract_band_size(
        encoding: &mut Map<String, Value>,
        aesthetic: &str,
        axis: &str,
    ) -> Result<f64> {
        const DEFAULT_BAND_SIZE: f64 = 1.0;

        // Extract and remove the aesthetic
        let size_enc = encoding.remove(aesthetic);

        // If no aesthetic specified, use default
        let Some(size_enc) = size_enc else {
            return Ok(DEFAULT_BAND_SIZE);
        };

        // Case 1: value encoding (from SETTING parameter) - extract directly
        if let Some(value) = size_enc.get("value").and_then(|v| v.as_f64()) {
            return Ok(value);
        }

        // Case 2: field encoding (from MAPPING) - check scale domain for constant
        if size_enc.get("field").is_none() {
            // Neither value nor field - shouldn't happen
            return Err(GgsqlError::WriterError(format!(
                "Invalid {} encoding (expected value or field).",
                aesthetic
            )));
        }

        // Helper closure for repeated error message
        let domain_error = || {
            GgsqlError::WriterError(format!(
                "Could not determine {} value for discrete {} scale.",
                aesthetic, axis
            ))
        };

        // Extract domain from scale
        let domain = size_enc
            .get("scale")
            .and_then(|s| s.get("domain"))
            .and_then(|d| d.as_array())
            .ok_or_else(domain_error)?;

        if domain.len() != 2 {
            return Err(domain_error());
        }

        let (Some(min), Some(max)) = (domain[0].as_f64(), domain[1].as_f64()) else {
            return Err(domain_error());
        };

        if (min - max).abs() < 1e-10 {
            // Constant value - use it
            Ok(min)
        } else {
            // Variable - error
            Err(GgsqlError::WriterError(format!(
                "Discrete {} scale does not support variable {} columns.",
                axis, aesthetic
            )))
        }
    }
}

impl GeomRenderer for RectRenderer {
    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
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

    fn modify_spec(&self, layer_spec: &mut Value, _layer: &Layer) -> Result<()> {
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

        // Build mark spec with band sizing for discrete directions
        let mut mark = json!({
            "type": "rect",
            "clip": true
        });

        if x_is_discrete {
            let width = Self::extract_band_size(encoding, "width", "x")?;
            mark["width"] = json!({"band": width});
        }

        if y_is_discrete {
            let height = Self::extract_band_size(encoding, "height", "y")?;
            mark["height"] = json!({"band": height});
        }

        layer_spec["mark"] = mark;
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
        GeomType::Rect => Box::new(RectRenderer),
        GeomType::Ribbon => Box::new(RibbonRenderer),
        GeomType::Polygon => Box::new(PolygonRenderer),
        GeomType::Boxplot => Box::new(BoxplotRenderer),
        GeomType::Density => Box::new(AreaRenderer),
        GeomType::Violin => Box::new(ViolinRenderer),
        // All other geoms (Point, Line, etc.) use the default renderer
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

        renderer.modify_encoding(encoding, &layer)?;

        let mut layer_spec = json!({
            "mark": {"type": "rect", "clip": true},
            "encoding": encoding
        });

        renderer.modify_spec(&mut layer_spec, &layer)?;
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

        // width should be removed
        assert!(enc.get("width").is_none());

        // Should have width band sizing for discrete x
        assert_eq!(spec["mark"]["width"], json!({"band": 0.8}));
        assert!(spec["mark"].get("height").is_none()); // y is continuous, no band height
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

        // width/height should be removed
        assert!(enc.get("width").is_none());
        assert!(enc.get("height").is_none());

        // Should have both width and height band sizing
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

        // Should have default band sizing (1.0) for both
        assert_eq!(spec["mark"]["width"], json!({"band": 1.0}));
        assert_eq!(spec["mark"]["height"], json!({"band": 1.0}));
    }

    #[test]
    fn test_rect_discrete_with_constant_width_column() {
        // Test rect with discrete x scale where width is a constant-valued column
        // This should work by detecting the constant in the scale domain
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("width".to_string(), scale("width_col", 0.85, 0.85)); // constant
        encoding.insert("ymin".to_string(), quant("ymin_col"));
        encoding.insert("ymax".to_string(), quant("ymax_col"));

        let spec = render_rect(&mut encoding).unwrap();

        // Should extract the constant value 0.85
        assert_eq!(spec["mark"]["width"], json!({"band": 0.85}));
    }

    #[test]
    fn test_rect_discrete_with_variable_width_column_error() {
        // Test that variable width columns on discrete scales produce an error
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert("width".to_string(), scale("width_col", 0.5, 0.9)); // variable

        let result = render_rect(&mut encoding);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Discrete x scale"));
        assert!(err
            .to_string()
            .contains("does not support variable width columns"));
    }

    #[test]
    fn test_rect_extract_band_size_missing_domain_error() {
        // Test that missing domain in scale produces an error
        let mut encoding = serde_json::Map::new();
        encoding.insert("x".to_string(), nominal("day"));
        encoding.insert(
            "width".to_string(),
            json!({
                "field": "width_col",
                "type": "quantitative",
                "scale": {}  // missing domain
            }),
        );

        let result = render_rect(&mut encoding);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Could not determine width value"));
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

        // height should be removed
        assert!(enc.get("height").is_none());

        // Should have height band sizing for discrete y
        assert!(spec["mark"].get("width").is_none()); // x is continuous, no band width
        assert_eq!(spec["mark"]["height"], json!({"band": 0.6}));
    }
}
