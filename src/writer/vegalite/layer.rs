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
// Text Renderer
// =============================================================================

/// Strategy for handling font properties in text layers
struct FontStrategy {
    /// Each group is (properties, row_indices). Properties include both constant and varying values.
    groups: Vec<(HashMap<String, Value>, Vec<usize>)>,
}

/// Renderer for text geom - handles font properties via data splitting
pub struct TextRenderer;

impl TextRenderer {
    /// Analyze DataFrame columns to find font aesthetics
    fn analyze_font_columns(df: &DataFrame) -> Result<FontStrategy> {

        let mut varying_columns: Vec<(String, String)> = Vec::new(); // (aesthetic, column_name)
        let mut constant_values: HashMap<String, Value> = HashMap::new();

        // Check for font aesthetic columns in DataFrame
        for &aesthetic in &["family", "fontface", "hjust", "vjust"] {
            let col_name = naming::aesthetic_column(aesthetic);

            if let Ok(col) = df.column(&col_name) {
                let unique_count = col
                    .n_unique()
                    .map_err(|e| GgsqlError::WriterError(e.to_string()))?;

                if unique_count == 1 {
                    // All same → treat as constant
                    let value_str = col
                        .str()
                        .map_err(|e| GgsqlError::WriterError(e.to_string()))?
                        .get(0)
                        .unwrap_or("");
                    let converted = Self::convert_to_mark_property(aesthetic, value_str);
                    constant_values.insert(aesthetic.to_string(), converted);
                } else if unique_count > 1 {
                    // Multiple values → needs splitting
                    varying_columns.push((aesthetic.to_string(), col_name));
                }
            }
        }

        if varying_columns.is_empty() {
            // All constant or not present → single group with all rows
            let all_indices: Vec<usize> = (0..df.height()).collect();
            let groups = vec![(constant_values.clone(), all_indices)];
            Ok(FontStrategy { groups })
        } else {
            // Some varying → multi-layer
            let groups = Self::build_font_groups_from_df(df, &varying_columns, &constant_values)?;
            Ok(FontStrategy { groups })
        }
    }

    /// Build groups from DataFrame columns (used in prepare_data)
    fn build_font_groups_from_df(
        data: &DataFrame,
        varying: &[(String, String)], // (aesthetic, column_name)
        constant: &HashMap<String, Value>,
    ) -> Result<Vec<(HashMap<String, Value>, Vec<usize>)>> {
        use polars::prelude::{ChunkedArray, StringType};

        let nrows = data.height();
        let mut groups_map: HashMap<String, (HashMap<String, Value>, Vec<usize>)> = HashMap::new();

        // Pre-fetch all varying font columns
        let font_columns: Vec<(String, ChunkedArray<StringType>)> = varying
            .iter()
            .map(|(aes, col)| {
                let series = data
                    .column(col)
                    .map_err(|e| GgsqlError::WriterError(e.to_string()))?;
                let ca = series
                    .str()
                    .map_err(|e| GgsqlError::WriterError(e.to_string()))?
                    .clone();
                Ok((aes.clone(), ca))
            })
            .collect::<Result<Vec<_>>>()?;

        // Iterate rows and build groups
        for row_idx in 0..nrows {
            // Build signature for this row
            let mut sig_parts: Vec<String> = Vec::new();
            let mut properties = constant.clone();

            for (aesthetic, ca) in &font_columns {
                let value = ca.get(row_idx).unwrap_or("");
                sig_parts.push(format!("{}:{}", aesthetic, value));

                // Convert to mark property
                let prop_value = Self::convert_to_mark_property(aesthetic, value);
                properties.insert(aesthetic.to_string(), prop_value);
            }

            let signature = sig_parts.join("|");

            // Add to existing group or create new entry
            groups_map
                .entry(signature)
                .or_insert_with(|| (properties.clone(), Vec::new()))
                .1
                .push(row_idx);
        }

        // Convert to Vec and sort by first occurrence
        let mut groups: Vec<(HashMap<String, Value>, Vec<usize>)> = groups_map
            .into_values()
            .collect();

        groups.sort_by_key(|(_, indices)| indices[0]);

        Ok(groups)
    }

    /// Convert ggsql font aesthetic value to Vega-Lite mark property
    fn convert_to_mark_property(aesthetic: &str, value: &str) -> Value {
        match aesthetic {
            "family" => json!(value),
            "fontface" => {
                // Map ggplot2 fontface to fontWeight/fontStyle
                match value {
                    "bold" => json!({"fontWeight": "bold"}),
                    "italic" => json!({"fontStyle": "italic"}),
                    "bold.italic" | "bolditalic" => json!({
                        "fontWeight": "bold",
                        "fontStyle": "italic"
                    }),
                    _ => json!({"fontWeight": "normal"}),
                }
            }
            "hjust" => {
                // Map 0/0.5/1 or string to left/center/right
                let align = match value.parse::<f64>() {
                    Ok(v) if v <= 0.25 => "left",
                    Ok(v) if v >= 0.75 => "right",
                    _ => match value {
                        "left" => "left",
                        "right" => "right",
                        _ => "center",
                    },
                };
                json!(align)
            }
            "vjust" => {
                // Map 0/0.5/1 or string to bottom/middle/top
                let baseline = match value.parse::<f64>() {
                    Ok(v) if v <= 0.25 => "bottom",
                    Ok(v) if v >= 0.75 => "top",
                    _ => match value {
                        "top" => "top",
                        "bottom" => "bottom",
                        _ => "middle",
                    },
                };
                json!(baseline)
            }
            _ => json!(value),
        }
    }

    /// Filter DataFrame to specific row indices
    fn filter_by_indices(data: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
        use polars::prelude::{BooleanChunked, NamedFrom};

        let nrows = data.height();
        let mut mask_data = vec![false; nrows];
        for &idx in indices {
            if idx < nrows {
                mask_data[idx] = true;
            }
        }

        let mask = BooleanChunked::new("".into(), mask_data);

        data.filter(&mask)
            .map_err(|e| GgsqlError::WriterError(e.to_string()))
    }

    /// Map aesthetic name to Vega-Lite mark property name
    fn map_aesthetic_to_mark_property(aesthetic: &str) -> &str {
        match aesthetic {
            "family" => "font",
            "hjust" => "align",
            "vjust" => "baseline",
            _ => aesthetic,
        }
    }

    /// Apply mark property, handling special cases like fontface
    fn apply_mark_property(mark_obj: &mut Map<String, Value>, key: &str, value: &Value) {
        if key == "fontface" || value.is_object() {
            // fontface may contain multiple properties (fontWeight + fontStyle)
            if let Some(obj) = value.as_object() {
                for (k, v) in obj {
                    mark_obj.insert(k.clone(), v.clone());
                }
                return;
            }
        }
        mark_obj.insert(key.to_string(), value.clone());
    }

    /// Finalize single layer case
    /// Finalize layers from font groups (handles both single and multi-group cases)
    fn finalize_layers(
        &self,
        prototype: Value,
        data_key: &str,
        groups: &[(HashMap<String, Value>, Vec<usize>)],
    ) -> Result<Vec<Value>> {
        let mut layers = Vec::new();

        for (group_idx, (properties, _indices)) in groups.iter().enumerate() {
            let mut layer_spec = prototype.clone();
            // For single-group case (all constant), use empty suffix
            // For multi-group case, use _font_N suffix
            let suffix = if groups.len() == 1 {
                String::new()
            } else {
                format!("_font_{}", group_idx)
            };
            let source_key = format!("{}{}", data_key, suffix);

            // Apply mark properties
            if let Some(mark) = layer_spec.get_mut("mark") {
                if let Some(mark_obj) = mark.as_object_mut() {
                    for (aesthetic, value) in properties {
                        let vl_key = Self::map_aesthetic_to_mark_property(aesthetic);
                        Self::apply_mark_property(mark_obj, vl_key, value);
                    }
                }
            }

            // Add source filter for this group
            let source_filter = json!({
                "filter": {
                    "field": naming::SOURCE_COLUMN,
                    "equal": source_key
                }
            });

            let existing_transforms = layer_spec
                .get("transform")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default();

            let mut new_transforms = vec![source_filter];
            new_transforms.extend(existing_transforms);
            layer_spec["transform"] = json!(new_transforms);

            layers.push(layer_spec);
        }

        Ok(layers)
    }
}

impl GeomRenderer for TextRenderer {
    fn prepare_data(
        &self,
        df: &DataFrame,
        _data_key: &str,
        binned_columns: &HashMap<String, Vec<f64>>,
    ) -> Result<PreparedData> {
        // Analyze font columns to determine strategy
        let strategy = Self::analyze_font_columns(df)?;

        // Split data by groups (even if just 1 group for constant fonts)
        let mut components: HashMap<String, Vec<Value>> = HashMap::new();

        for (group_idx, (_properties, row_indices)) in strategy.groups.iter().enumerate() {
            // For single-group case (all constant), use empty suffix
            // For multi-group case, use _font_N suffix
            let suffix = if strategy.groups.len() == 1 {
                String::new()
            } else {
                format!("_font_{}", group_idx)
            };

            let filtered = Self::filter_by_indices(df, row_indices)?;
            let values = if binned_columns.is_empty() {
                dataframe_to_values(&filtered)?
            } else {
                dataframe_to_values_with_bins(&filtered, binned_columns)?
            };

            components.insert(suffix, values);
        }

        Ok(PreparedData::Composite {
            components,
            metadata: Box::new(strategy),
        })
    }

    fn modify_encoding(&self, encoding: &mut Map<String, Value>, _layer: &Layer) -> Result<()> {
        // Remove font aesthetics from encoding - they only work as mark properties
        for &aesthetic in &["family", "fontface", "hjust", "vjust"] {
            encoding.remove(aesthetic);
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
        _layer: &Layer,
        data_key: &str,
        prepared: &PreparedData,
    ) -> Result<Vec<Value>> {
        let PreparedData::Composite { metadata, .. } = prepared else {
            return Err(GgsqlError::InternalError(
                "TextRenderer::finalize called with non-composite data".to_string(),
            ));
        };

        // Downcast metadata to FontStrategy
        let strategy = metadata.downcast_ref::<FontStrategy>().ok_or_else(|| {
            GgsqlError::InternalError("Failed to downcast font strategy".to_string())
        })?;

        // Generate layers from groups (1 group = single layer, N groups = N layers)
        self.finalize_layers(prototype, data_key, &strategy.groups)
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
            "interpolate": "linear-closed",
            "fill": "#888888",
            "stroke": "#888888"
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

        // Default styling
        let default_stroke = "black";
        let default_fill = "#FFFFFF00";
        let default_linewidth = 1.0;

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
                    "type": "point",
                    "stroke": default_stroke,
                    "strokeWidth": default_linewidth
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
                "type": "rule",
                "stroke": default_stroke,
                "size": default_linewidth
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
                "type": "rule",
                "stroke": default_stroke,
                "size": default_linewidth
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
                "align": "center",
                "stroke": default_stroke,
                "color": default_fill,
                "strokeWidth": default_linewidth
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
                "stroke": default_stroke,
                "width": {"band": width},
                "align": "center",
                "strokeWidth": default_linewidth
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

        // Create DataFrame where all rows have the same font
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("family").as_str() => &["Arial", "Arial", "Arial"],
        }
        .unwrap();

        // Prepare data - should result in single layer with empty component key
        let prepared = renderer.prepare_data(&df, "test", &HashMap::new()).unwrap();

        match prepared {
            PreparedData::Composite { components, .. } => {
                // Should have single component with empty key
                assert_eq!(components.len(), 1);
                assert!(components.contains_key(""));
            }
            _ => panic!("Expected Composite"),
        }
    }

    #[test]
    fn test_text_varying_font() {
        use crate::naming;
        use polars::prelude::*;

        let renderer = TextRenderer;

        // Create DataFrame with different fonts per row
        let df = df! {
            naming::aesthetic_column("x").as_str() => &[1.0, 2.0, 3.0],
            naming::aesthetic_column("y").as_str() => &[10.0, 20.0, 30.0],
            naming::aesthetic_column("label").as_str() => &["A", "B", "C"],
            naming::aesthetic_column("family").as_str() => &["Arial", "Courier", "Times"],
        }
        .unwrap();

        // Prepare data - should result in multiple layers
        let prepared = renderer.prepare_data(&df, "test", &HashMap::new()).unwrap();

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
}
