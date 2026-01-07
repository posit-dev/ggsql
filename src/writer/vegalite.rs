//! Vega-Lite JSON writer implementation
//!
//! Converts ggSQL specifications and DataFrames into Vega-Lite JSON format
//! for web-based interactive visualizations.
//!
//! # Mapping Strategy
//!
//! - ggSQL Geom → Vega-Lite mark type
//! - ggSQL aesthetics → Vega-Lite encoding channels
//! - ggSQL layers → Vega-Lite layer composition
//! - Polars DataFrame → Vega-Lite inline data
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::writer::{Writer, VegaLiteWriter};
//!
//! let writer = VegaLiteWriter::new();
//! let vega_json = writer.write(&spec, &dataframe)?;
//! // Can be rendered in browser with vega-embed
//! ```

use crate::writer::Writer;
use crate::{DataFrame, Result, GgsqlError, VizSpec, Geom, AestheticValue};
use crate::parser::ast::{LiteralValue, Coord, CoordType, CoordPropertyValue, ArrayElement, FilterExpression, ComparisonOp, FilterValue};
use serde_json::{json, Value, Map};
use polars::prelude::*;

/// Vega-Lite JSON writer
///
/// Generates Vega-Lite v5 specifications from ggSQL specs and data.
pub struct VegaLiteWriter {
    /// Vega-Lite schema version
    schema: String,
}

impl VegaLiteWriter {
    /// Create a new Vega-Lite writer with default settings
    pub fn new() -> Self {
        Self {
            schema: "https://vega.github.io/schema/vega-lite/v5.json".to_string(),
        }
    }

    /// Convert Polars DataFrame to Vega-Lite data values (array of objects)
    fn dataframe_to_values(&self, df: &DataFrame) -> Result<Vec<Value>> {
        let mut values = Vec::new();
        let height = df.height();
        let column_names = df.get_column_names();

        for row_idx in 0..height {
            let mut row_obj = Map::new();

            for (col_idx, col_name) in column_names.iter().enumerate() {
                let series = df.get_columns().get(col_idx).ok_or_else(|| {
                    GgsqlError::WriterError(format!("Failed to get column {}", col_name))
                })?;

                // Get value from series and convert to JSON Value
                let value = self.series_value_at(series, row_idx)?;
                row_obj.insert(col_name.to_string(), value);
            }

            values.push(Value::Object(row_obj));
        }

        Ok(values)
    }

    /// Get a single value from a series at a given index as JSON Value
    fn series_value_at(&self, series: &Series, idx: usize) -> Result<Value> {
        use DataType::*;

        match series.dtype() {
            Int8 => {
                let ca = series.i8().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i8: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int16 => {
                let ca = series.i16().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i16: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int32 => {
                let ca = series.i32().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i32: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Int64 => {
                let ca = series.i64().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to i64: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Float32 => {
                let ca = series.f32().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to f32: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Float64 => {
                let ca = series.f64().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to f64: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            Boolean => {
                let ca = series.bool().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to bool: {}", e))
                })?;
                Ok(ca.get(idx).map(|v| json!(v)).unwrap_or(Value::Null))
            }
            String => {
                let ca = series.str().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to string: {}", e))
                })?;
                // Try to parse as number if it looks numeric
                if let Some(val) = ca.get(idx) {
                    if let Ok(num) = val.parse::<f64>() {
                        Ok(json!(num))
                    } else {
                        Ok(json!(val))
                    }
                } else {
                    Ok(Value::Null)
                }
            }
            Date => {
                // Convert days since epoch to ISO date string: "YYYY-MM-DD"
                let ca = series.date().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to date: {}", e))
                })?;
                if let Some(days) = ca.get(idx) {
                    let unix_epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let date = unix_epoch + chrono::Duration::days(days as i64);
                    Ok(json!(date.format("%Y-%m-%d").to_string()))
                } else {
                    Ok(Value::Null)
                }
            }
            Datetime(time_unit, _) => {
                // Convert timestamp to ISO datetime: "YYYY-MM-DDTHH:MM:SS.sssZ"
                let ca = series.datetime().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to datetime: {}", e))
                })?;
                if let Some(timestamp) = ca.get(idx) {
                    // Convert to microseconds based on time unit
                    let micros = match time_unit {
                        TimeUnit::Microseconds => timestamp,
                        TimeUnit::Milliseconds => timestamp * 1_000,
                        TimeUnit::Nanoseconds => timestamp / 1_000,
                    };
                    let secs = micros / 1_000_000;
                    let nsecs = ((micros % 1_000_000) * 1000) as u32;
                    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nsecs)
                        .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap());
                    Ok(json!(dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()))
                } else {
                    Ok(Value::Null)
                }
            }
            Time => {
                // Convert nanoseconds since midnight to ISO time: "HH:MM:SS.sss"
                let ca = series.time().map_err(|e| {
                    GgsqlError::WriterError(format!("Failed to cast to time: {}", e))
                })?;
                if let Some(nanos) = ca.get(idx) {
                    let hours = nanos / 3_600_000_000_000;
                    let minutes = (nanos % 3_600_000_000_000) / 60_000_000_000;
                    let seconds = (nanos % 60_000_000_000) / 1_000_000_000;
                    let millis = (nanos % 1_000_000_000) / 1_000_000;
                    Ok(json!(format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)))
                } else {
                    Ok(Value::Null)
                }
            }
            _ => {
                // Fallback: convert to string
                Ok(json!(series.get(idx).map(|v| v.to_string()).unwrap_or_default()))
            }
        }
    }

    /// Map ggSQL Geom to Vega-Lite mark type
    fn geom_to_mark(&self, geom: &Geom) -> String {
        match geom {
            Geom::Point => "point",
            Geom::Line => "line",
            Geom::Path => "line",
            Geom::Bar => "bar",
            Geom::Col => "bar",
            Geom::Area => "area",
            Geom::Tile => "rect",
            Geom::Ribbon => "area",
            Geom::Histogram => "bar",
            Geom::Density => "area",
            Geom::Boxplot => "boxplot",
            Geom::Text => "text",
            Geom::Label => "text",
            _ => "point", // Default fallback
        }
        .to_string()
    }

    /// Convert a FilterExpression to Vega-Lite filter format
    fn filter_to_vegalite(&self, filter: &FilterExpression) -> Value {
        match filter {
            FilterExpression::And(left, right) => {
                json!({
                    "and": [
                        self.filter_to_vegalite(left),
                        self.filter_to_vegalite(right)
                    ]
                })
            }
            FilterExpression::Or(left, right) => {
                json!({
                    "or": [
                        self.filter_to_vegalite(left),
                        self.filter_to_vegalite(right)
                    ]
                })
            }
            FilterExpression::Comparison { column, operator, value } => {
                let op_str = match operator {
                    ComparisonOp::Eq => "===",
                    ComparisonOp::Ne => "!==",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Ge => ">=",
                };

                let value_str = match value {
                    FilterValue::String(s) => format!("'{}'", s),
                    FilterValue::Number(n) => n.to_string(),
                    FilterValue::Boolean(b) => b.to_string(),
                    FilterValue::Column(c) => format!("datum.{}", c),
                };

                // Vega-Lite filter expression: "datum.column op value"
                json!(format!("datum.{} {} {}", column, op_str, value_str))
            }
        }
    }

    /// Check if a string column contains numeric values
    fn is_numeric_string_column(&self, series: &Series) -> bool {
        if let Ok(ca) = series.str() {
            // Check first few non-null values to see if they're numeric
            for val in ca.into_iter().flatten().take(5) {
                if val.parse::<f64>().is_err() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Infer Vega-Lite field type from DataFrame column
    fn infer_field_type(&self, df: &DataFrame, field: &str) -> String {
        if let Some(series) = df.column(field).ok() {
            use DataType::*;
            match series.dtype() {
                Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 | Float32
                | Float64 => "quantitative",
                Boolean => "nominal",
                String => {
                    // Check if string column contains numeric values
                    if self.is_numeric_string_column(series) {
                        "quantitative"
                    } else {
                        "nominal"
                    }
                }
                Date | Datetime(_, _) | Time => "temporal",
                _ => "nominal",
            }
            .to_string()
        } else {
            "nominal".to_string()
        }
    }

    /// Build encoding channel from aesthetic mapping
    fn build_encoding_channel(
        &self,
        aesthetic: &str,
        value: &AestheticValue,
        df: &DataFrame,
        spec: &VizSpec,
    ) -> Result<Value> {
        match value {
            AestheticValue::Column(col) => {
                // Check if there's a scale specification for this aesthetic
                let field_type = if let Some(scale) = spec.find_scale(aesthetic) {
                    // Use scale type if explicitly specified
                    if let Some(scale_type) = &scale.scale_type {
                        use crate::parser::ast::ScaleType;
                        match scale_type {
                            ScaleType::Linear | ScaleType::Log10 | ScaleType::Log |
                            ScaleType::Log2 | ScaleType::Sqrt | ScaleType::Reverse => "quantitative",
                            ScaleType::Ordinal | ScaleType::Categorical => "nominal",
                            ScaleType::Date | ScaleType::DateTime | ScaleType::Time => "temporal",
                            ScaleType::Viridis | ScaleType::Plasma | ScaleType::Magma |
                            ScaleType::Inferno | ScaleType::Cividis | ScaleType::Diverging |
                            ScaleType::Sequential => "quantitative", // Color scales
                        }.to_string()
                    } else if scale.properties.contains_key("domain") {
                        // If domain is specified without explicit type:
                        // - For size/opacity: keep quantitative (domain sets range, not categories)
                        // - For color/x/y: treat as ordinal (discrete categories)
                        if aesthetic == "size" || aesthetic == "opacity" || aesthetic == "alpha" {
                            "quantitative".to_string()
                        } else {
                            "ordinal".to_string()
                        }
                    } else {
                        // Scale exists but no type specified, infer from data
                        self.infer_field_type(df, col)
                    }
                } else {
                    // No scale specification, infer from data
                    self.infer_field_type(df, col)
                };

                let mut encoding = json!({
                    "field": col,
                    "type": field_type,
                });

                // Add titles for positional aesthetics
                if aesthetic == "x" || aesthetic == "y" {
                    encoding["title"] = json!(col);
                }

                // Apply scale properties from SCALE if specified
                if let Some(scale) = spec.find_scale(aesthetic) {
                    use crate::parser::ast::{ScalePropertyValue, ArrayElement};
                    let mut scale_obj = serde_json::Map::new();

                    // Apply domain
                    if let Some(domain_prop) = scale.properties.get("domain") {
                        if let ScalePropertyValue::Array(domain_values) = domain_prop {
                            let domain_json: Vec<Value> = domain_values.iter().map(|elem| {
                                match elem {
                                    ArrayElement::String(s) => json!(s),
                                    ArrayElement::Number(n) => json!(n),
                                    ArrayElement::Boolean(b) => json!(b),
                                }
                            }).collect();
                            scale_obj.insert("domain".to_string(), json!(domain_json));
                        }
                    }

                    // Apply range (explicit range property takes precedence over palette)
                    if let Some(range_prop) = scale.properties.get("range") {
                        if let ScalePropertyValue::Array(range_values) = range_prop {
                            let range_json: Vec<Value> = range_values.iter().map(|elem| {
                                match elem {
                                    ArrayElement::String(s) => json!(s),
                                    ArrayElement::Number(n) => json!(n),
                                    ArrayElement::Boolean(b) => json!(b),
                                }
                            }).collect();
                            scale_obj.insert("range".to_string(), json!(range_json));
                        }
                    } else if let Some(palette_prop) = scale.properties.get("palette") {
                        // Apply palette as range (fallback for color scales)
                        if let ScalePropertyValue::Array(palette_values) = palette_prop {
                            let range_json: Vec<Value> = palette_values.iter().map(|elem| {
                                match elem {
                                    ArrayElement::String(s) => json!(s),
                                    ArrayElement::Number(n) => json!(n),
                                    ArrayElement::Boolean(b) => json!(b),
                                }
                            }).collect();
                            scale_obj.insert("range".to_string(), json!(range_json));
                        }
                    }

                    if !scale_obj.is_empty() {
                        encoding["scale"] = json!(scale_obj);
                    }
                }

                Ok(encoding)
            }
            AestheticValue::Literal(lit) => {
                // For literal values, use constant value encoding
                let val = match lit {
                    LiteralValue::String(s) => json!(s),
                    LiteralValue::Number(n) => json!(n),
                    LiteralValue::Boolean(b) => json!(b),
                };
                Ok(json!({"value": val}))
            }
        }
    }

    /// Map ggSQL aesthetic name to Vega-Lite encoding channel name
    fn map_aesthetic_name(&self, aesthetic: &str) -> String {
        match aesthetic {
            "fill" => "color",
            "alpha" => "opacity",
            _ => aesthetic,
        }
        .to_string()
    }

    /// Apply guide configurations to encoding channels
    fn apply_guides_to_encoding(&self, encoding: &mut Map<String, Value>, spec: &VizSpec) {
        use crate::parser::ast::{GuideType, GuidePropertyValue};

        for guide in &spec.guides {
            let channel_name = self.map_aesthetic_name(&guide.aesthetic);

            // Skip if this channel doesn't exist in the encoding
            if !encoding.contains_key(&channel_name) {
                continue;
            }

            // Handle guide type
            match &guide.guide_type {
                Some(GuideType::None) => {
                    // Remove legend for this channel
                    if let Some(channel) = encoding.get_mut(&channel_name) {
                        channel["legend"] = json!(null);
                    }
                }
                Some(GuideType::Legend) => {
                    // Apply legend properties
                    if let Some(channel) = encoding.get_mut(&channel_name) {
                        let mut legend = json!({});

                        for (prop_name, prop_value) in &guide.properties {
                            let value = match prop_value {
                                GuidePropertyValue::String(s) => json!(s),
                                GuidePropertyValue::Number(n) => json!(n),
                                GuidePropertyValue::Boolean(b) => json!(b),
                            };

                            // Map property names to Vega-Lite legend properties
                            match prop_name.as_str() {
                                "title" => legend["title"] = value,
                                "position" => legend["orient"] = value,
                                "direction" => legend["direction"] = value,
                                "nrow" => legend["rowPadding"] = value,
                                "ncol" => legend["columnPadding"] = value,
                                "title_position" => legend["titleAnchor"] = value,
                                _ => {
                                    // Pass through other properties
                                    legend[prop_name] = value;
                                }
                            }
                        }

                        if !legend.as_object().unwrap().is_empty() {
                            channel["legend"] = legend;
                        }
                    }
                }
                Some(GuideType::ColorBar) => {
                    // For color bars, similar to legend but with gradient
                    if let Some(channel) = encoding.get_mut(&channel_name) {
                        let mut legend = json!({"type": "gradient"});

                        for (prop_name, prop_value) in &guide.properties {
                            let value = match prop_value {
                                GuidePropertyValue::String(s) => json!(s),
                                GuidePropertyValue::Number(n) => json!(n),
                                GuidePropertyValue::Boolean(b) => json!(b),
                            };

                            match prop_name.as_str() {
                                "title" => legend["title"] = value,
                                "position" => legend["orient"] = value,
                                _ => legend[prop_name] = value,
                            }
                        }

                        channel["legend"] = legend;
                    }
                }
                Some(GuideType::Axis) => {
                    // Apply axis properties
                    if let Some(channel) = encoding.get_mut(&channel_name) {
                        let mut axis = json!({});

                        for (prop_name, prop_value) in &guide.properties {
                            let value = match prop_value {
                                GuidePropertyValue::String(s) => json!(s),
                                GuidePropertyValue::Number(n) => json!(n),
                                GuidePropertyValue::Boolean(b) => json!(b),
                            };

                            // Map property names to Vega-Lite axis properties
                            match prop_name.as_str() {
                                "title" => axis["title"] = value,
                                "text_angle" => axis["labelAngle"] = value,
                                "text_size" => axis["labelFontSize"] = value,
                                _ => axis[prop_name] = value,
                            }
                        }

                        if !axis.as_object().unwrap().is_empty() {
                            channel["axis"] = axis;
                        }
                    }
                }
                None => {
                    // No specific guide type, just apply properties generically
                    if let Some(channel) = encoding.get_mut(&channel_name) {
                        for (prop_name, prop_value) in &guide.properties {
                            let value = match prop_value {
                                GuidePropertyValue::String(s) => json!(s),
                                GuidePropertyValue::Number(n) => json!(n),
                                GuidePropertyValue::Boolean(b) => json!(b),
                            };
                            channel[prop_name] = value;
                        }
                    }
                }
            }
        }
    }

    /// Validate that all column references in aesthetics exist in the DataFrame
    fn validate_column_references(&self, spec: &VizSpec, data: &DataFrame) -> Result<()> {
        let available_columns: Vec<String> = data
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Check all layers
        for (layer_idx, layer) in spec.layers.iter().enumerate() {
            for (aesthetic, value) in &layer.aesthetics {
                if let AestheticValue::Column(col) = value {
                    if !available_columns.contains(col) {
                        return Err(GgsqlError::ValidationError(format!(
                            "Column '{}' referenced in aesthetic '{}' (layer {}) does not exist in the query result.\nAvailable columns: {}",
                            col,
                            aesthetic,
                            layer_idx + 1,
                            available_columns.join(", ")
                        )));
                    }
                }
            }

            // Check partition_by columns
            for col in &layer.partition_by {
                if !available_columns.contains(col) {
                    return Err(GgsqlError::ValidationError(format!(
                        "Column '{}' referenced in PARTITION BY (layer {}) does not exist in the query result.\nAvailable columns: {}",
                        col,
                        layer_idx + 1,
                        available_columns.join(", ")
                    )));
                }
            }
        }

        // Check facet variables
        if let Some(facet) = &spec.facet {
            use crate::parser::ast::Facet;
            match facet {
                Facet::Wrap { variables, .. } => {
                    for var in variables {
                        if !available_columns.contains(var) {
                            return Err(GgsqlError::ValidationError(format!(
                                "Facet variable '{}' does not exist in the query result.\nAvailable columns: {}",
                                var,
                                available_columns.join(", ")
                            )));
                        }
                    }
                }
                Facet::Grid { rows, cols, .. } => {
                    for var in rows.iter().chain(cols.iter()) {
                        if !available_columns.contains(var) {
                            return Err(GgsqlError::ValidationError(format!(
                                "Facet variable '{}' does not exist in the query result.\nAvailable columns: {}",
                                var,
                                available_columns.join(", ")
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for VegaLiteWriter {
    fn default() -> Self {
        Self::new()
    }
}

// Coordinate transformation methods
impl VegaLiteWriter {
    /// Apply coordinate transformations to the spec and data
    /// Returns (possibly transformed DataFrame, possibly modified spec)
    fn apply_coord_transforms(
        &self,
        spec: &VizSpec,
        data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<Option<DataFrame>> {
        if let Some(ref coord) = spec.coord {
            match coord.coord_type {
                CoordType::Cartesian => {
                    self.apply_cartesian_coord(coord, vl_spec, data)?;
                    Ok(None) // No DataFrame transformation needed
                }
                CoordType::Flip => {
                    self.apply_flip_coord(vl_spec)?;
                    Ok(None) // No DataFrame transformation needed
                }
                CoordType::Polar => {
                    // Polar requires DataFrame transformation for percentages
                    let transformed_df = self.apply_polar_coord(coord, spec, data, vl_spec)?;
                    Ok(Some(transformed_df))
                }
                _ => {
                    // Other coord types not yet implemented
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Apply Cartesian coordinate properties (xlim, ylim, aesthetic domains)
    fn apply_cartesian_coord(
        &self,
        coord: &Coord,
        vl_spec: &mut Value,
        _data: &DataFrame,
    ) -> Result<()> {
        // Apply xlim/ylim to scale domains
        for (prop_name, prop_value) in &coord.properties {
            match prop_name.as_str() {
                "xlim" => {
                    if let Some(limits) = self.extract_limits(prop_value)? {
                        self.apply_axis_limits(vl_spec, "x", limits)?;
                    }
                }
                "ylim" => {
                    if let Some(limits) = self.extract_limits(prop_value)? {
                        self.apply_axis_limits(vl_spec, "y", limits)?;
                    }
                }
                _ if self.is_aesthetic_name(prop_name) => {
                    // Aesthetic domain specification
                    if let Some(domain) = self.extract_domain(prop_value)? {
                        self.apply_aesthetic_domain(vl_spec, prop_name, domain)?;
                    }
                }
                _ => {
                    // ratio, clip - not yet implemented (TODO comments added by validation)
                }
            }
        }

        Ok(())
    }

    /// Apply Flip coordinate transformation (swap x and y)
    fn apply_flip_coord(&self, vl_spec: &mut Value) -> Result<()> {
        // Handle single layer
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(enc_obj) = encoding.as_object_mut() {
                // Swap x and y encodings
                if let (Some(x), Some(y)) = (enc_obj.remove("x"), enc_obj.remove("y")) {
                    enc_obj.insert("x".to_string(), y);
                    enc_obj.insert("y".to_string(), x);
                }
            }
        }

        // Handle multi-layer
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(enc_obj) = encoding.as_object_mut() {
                            if let (Some(x), Some(y)) = (enc_obj.remove("x"), enc_obj.remove("y")) {
                                enc_obj.insert("x".to_string(), y);
                                enc_obj.insert("y".to_string(), x);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply Polar coordinate transformation (bar→arc, point→arc with radius)
    fn apply_polar_coord(
        &self,
        coord: &Coord,
        spec: &VizSpec,
        _data: &DataFrame,
        vl_spec: &mut Value,
    ) -> Result<DataFrame> {
        // Get theta field (defaults to 'y')
        let theta_field = coord.properties.get("theta")
            .and_then(|v| match v {
                CoordPropertyValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "y".to_string());

        // Convert geoms to polar equivalents
        self.convert_geoms_to_polar(spec, vl_spec, &theta_field)?;

        // No DataFrame transformation needed - Vega-Lite handles polar math
        Ok(_data.clone())
    }

    /// Convert geoms to polar equivalents (bar→arc, point→arc with radius)
    fn convert_geoms_to_polar(
        &self,
        spec: &VizSpec,
        vl_spec: &mut Value,
        theta_field: &str,
    ) -> Result<()> {
        // Determine which aesthetic (x or y) maps to theta
        // Default: y maps to theta (pie chart style)
        let theta_aesthetic = theta_field;

        // Handle single layer
        if let Some(mark) = vl_spec.get_mut("mark") {
            *mark = self.convert_mark_to_polar(mark, spec)?;

            // Update encoding for polar
            if let Some(encoding) = vl_spec.get_mut("encoding") {
                self.update_encoding_for_polar(encoding, theta_aesthetic)?;
            }
        }

        // Handle multi-layer
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(mark) = layer.get_mut("mark") {
                        *mark = self.convert_mark_to_polar(mark, spec)?;

                        if let Some(encoding) = layer.get_mut("encoding") {
                            self.update_encoding_for_polar(encoding, theta_aesthetic)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert a mark type to its polar equivalent
    fn convert_mark_to_polar(&self, mark: &Value, _spec: &VizSpec) -> Result<Value> {
        let mark_str = if mark.is_string() {
            mark.as_str().unwrap()
        } else if let Some(mark_type) = mark.get("type") {
            mark_type.as_str().unwrap_or("bar")
        } else {
            "bar"
        };

        // Convert geom types to polar equivalents
        match mark_str {
            "bar" | "col" => {
                // Bar/col in polar becomes arc (pie/donut slices)
                Ok(json!("arc"))
            }
            "point" => {
                // Points in polar can stay as points or become arcs with radius
                // For now, keep as points (they'll plot at radius based on value)
                Ok(json!("point"))
            }
            "line" => {
                // Lines in polar become circular/spiral lines
                Ok(json!("line"))
            }
            "area" => {
                // Area in polar becomes arc with radius
                Ok(json!("arc"))
            }
            _ => {
                // Other geoms: keep as-is or convert to arc
                Ok(json!("arc"))
            }
        }
    }

    /// Update encoding channels for polar coordinates
    fn update_encoding_for_polar(
        &self,
        encoding: &mut Value,
        theta_aesthetic: &str,
    ) -> Result<()> {
        let enc_obj = encoding.as_object_mut().ok_or_else(|| {
            GgsqlError::WriterError("Encoding is not an object".to_string())
        })?;

        // Map the theta aesthetic to theta channel
        if theta_aesthetic == "y" {
            // Standard pie chart: y → theta, x → color/category
            if let Some(y_enc) = enc_obj.remove("y") {
                enc_obj.insert("theta".to_string(), y_enc);
            }
            // Map x to color if not already mapped, and remove x from positional encoding
            if !enc_obj.contains_key("color") {
                if let Some(x_enc) = enc_obj.remove("x") {
                    enc_obj.insert("color".to_string(), x_enc);
                }
            } else {
                // If color is already mapped, just remove x from positional encoding
                enc_obj.remove("x");
            }
        } else if theta_aesthetic == "x" {
            // Reversed: x → theta, y → radius
            if let Some(x_enc) = enc_obj.remove("x") {
                enc_obj.insert("theta".to_string(), x_enc);
            }
            if let Some(y_enc) = enc_obj.remove("y") {
                enc_obj.insert("radius".to_string(), y_enc);
            }
        }

        Ok(())
    }

    // Helper methods

    fn extract_limits(&self, value: &CoordPropertyValue) -> Result<Option<(f64, f64)>> {
        match value {
            CoordPropertyValue::Array(arr) => {
                if arr.len() != 2 {
                    return Err(GgsqlError::WriterError(format!(
                        "xlim/ylim must be exactly 2 numbers, got {}",
                        arr.len()
                    )));
                }
                let min = match &arr[0] {
                    ArrayElement::Number(n) => *n,
                    _ => return Err(GgsqlError::WriterError("xlim/ylim values must be numbers".to_string())),
                };
                let max = match &arr[1] {
                    ArrayElement::Number(n) => *n,
                    _ => return Err(GgsqlError::WriterError("xlim/ylim values must be numbers".to_string())),
                };

                // Auto-swap if reversed
                let (min, max) = if min > max { (max, min) } else { (min, max) };

                Ok(Some((min, max)))
            }
            _ => Err(GgsqlError::WriterError("xlim/ylim must be an array".to_string())),
        }
    }

    fn extract_domain(&self, value: &CoordPropertyValue) -> Result<Option<Vec<Value>>> {
        match value {
            CoordPropertyValue::Array(arr) => {
                let domain: Vec<Value> = arr.iter().map(|elem| match elem {
                    ArrayElement::String(s) => json!(s),
                    ArrayElement::Number(n) => json!(n),
                    ArrayElement::Boolean(b) => json!(b),
                }).collect();
                Ok(Some(domain))
            }
            _ => Ok(None),
        }
    }

    fn apply_axis_limits(&self, vl_spec: &mut Value, axis: &str, limits: (f64, f64)) -> Result<()> {
        let domain = json!([limits.0, limits.1]);

        // Apply to encoding if present
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(axis_enc) = encoding.get_mut(axis) {
                axis_enc["scale"] = json!({"domain": domain});
            }
        }

        // Apply to layers if present
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(axis_enc) = encoding.get_mut(axis) {
                            axis_enc["scale"] = json!({"domain": domain});
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_aesthetic_domain(&self, vl_spec: &mut Value, aesthetic: &str, domain: Vec<Value>) -> Result<()> {
        let domain_json = json!(domain);

        // Apply to encoding if present
        if let Some(encoding) = vl_spec.get_mut("encoding") {
            if let Some(aes_enc) = encoding.get_mut(aesthetic) {
                aes_enc["scale"] = json!({"domain": domain_json});
            }
        }

        // Apply to layers if present
        if let Some(layers) = vl_spec.get_mut("layer") {
            if let Some(layers_arr) = layers.as_array_mut() {
                for layer in layers_arr {
                    if let Some(encoding) = layer.get_mut("encoding") {
                        if let Some(aes_enc) = encoding.get_mut(aesthetic) {
                            aes_enc["scale"] = json!({"domain": domain_json});
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn is_aesthetic_name(&self, name: &str) -> bool {
        matches!(
            name,
            "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend" |
            "color" | "colour" | "fill" | "alpha" |
            "size" | "shape" | "linetype" | "linewidth" | "width" | "height" |
            "label" | "family" | "fontface" | "hjust" | "vjust"
        )
    }

    /// Build detail encoding from partition_by columns
    /// Maps partition_by columns to Vega-Lite's detail channel for grouping
    fn build_detail_encoding(&self, partition_by: &[String]) -> Option<Value> {
        if partition_by.is_empty() {
            return None;
        }

        if partition_by.len() == 1 {
            // Single column: simple object
            Some(json!({
                "field": partition_by[0],
                "type": "nominal"
            }))
        } else {
            // Multiple columns: array of detail specifications
            let details: Vec<Value> = partition_by.iter()
                .map(|col| json!({
                    "field": col,
                    "type": "nominal"
                }))
                .collect();
            Some(json!(details))
        }
    }
}

impl Writer for VegaLiteWriter {
    fn write(&self, spec: &VizSpec, data: &DataFrame) -> Result<String> {
        // Validate that all column references exist in the DataFrame
        self.validate_column_references(spec, data)?;

        // Build the base Vega-Lite spec (without data yet)
        let mut vl_spec = json!({
            "$schema": self.schema
        });

        // Add title if present
        if let Some(labels) = &spec.labels {
            if let Some(title) = labels.labels.get("title") {
                vl_spec["title"] = json!(title);
            }
        }

        // Handle single layer vs multi-layer
        if spec.layers.len() == 1 {
            // Single layer: use flat mark + encoding
            let layer = &spec.layers[0];
            vl_spec["mark"] = json!(self.geom_to_mark(&layer.geom));

            // Add filter transform if present
            if let Some(filter) = &layer.filter {
                vl_spec["transform"] = json!([{
                    "filter": self.filter_to_vegalite(filter)
                }]);
            }

            // Build encoding from aesthetics
            let mut encoding = Map::new();
            for (aesthetic, value) in &layer.aesthetics {
                let channel_name = self.map_aesthetic_name(aesthetic);
                let channel_encoding = self.build_encoding_channel(aesthetic, value, data, spec)?;
                encoding.insert(channel_name, channel_encoding);
            }

            // Add detail encoding for partition_by columns (grouping)
            if let Some(detail) = self.build_detail_encoding(&layer.partition_by) {
                encoding.insert("detail".to_string(), detail);
            }

            // Override axis titles from labels if present
            if let Some(labels) = &spec.labels {
                if let Some(x_label) = labels.labels.get("x") {
                    if let Some(x_enc) = encoding.get_mut("x") {
                        x_enc["title"] = json!(x_label);
                    }
                }
                if let Some(y_label) = labels.labels.get("y") {
                    if let Some(y_enc) = encoding.get_mut("y") {
                        y_enc["title"] = json!(y_label);
                    }
                }
            }

            // Apply guide configurations
            self.apply_guides_to_encoding(&mut encoding, spec);

            vl_spec["encoding"] = Value::Object(encoding);
        } else if spec.layers.len() > 1 {
            // Multi-layer: use layer composition
            let mut layers = Vec::new();

            for layer in &spec.layers {
                let mut layer_spec = json!({
                    "mark": self.geom_to_mark(&layer.geom)
                });

                // Add filter transform if present
                if let Some(filter) = &layer.filter {
                    layer_spec["transform"] = json!([{
                        "filter": self.filter_to_vegalite(filter)
                    }]);
                }

                // Build encoding for this layer
                let mut encoding = Map::new();
                for (aesthetic, value) in &layer.aesthetics {
                    let channel_name = self.map_aesthetic_name(aesthetic);
                    let channel_encoding = self.build_encoding_channel(aesthetic, value, data, spec)?;
                    encoding.insert(channel_name, channel_encoding);
                }

                // Add detail encoding for partition_by columns (grouping)
                if let Some(detail) = self.build_detail_encoding(&layer.partition_by) {
                    encoding.insert("detail".to_string(), detail);
                }

                // Override axis titles from labels if present (apply to each layer)
                if let Some(labels) = &spec.labels {
                    if let Some(x_label) = labels.labels.get("x") {
                        if let Some(x_enc) = encoding.get_mut("x") {
                            x_enc["title"] = json!(x_label);
                        }
                    }
                    if let Some(y_label) = labels.labels.get("y") {
                        if let Some(y_enc) = encoding.get_mut("y") {
                            y_enc["title"] = json!(y_label);
                        }
                    }
                }

                layer_spec["encoding"] = Value::Object(encoding);
                layers.push(layer_spec);
            }

            vl_spec["layer"] = json!(layers);

            // For multi-layer plots, apply guides at the top level by creating a resolve configuration
            // and encoding the guides in the spec-level encoding (if needed)
            if !spec.guides.is_empty() {
                let mut resolve = json!({"legend": {}, "scale": {}});
                for guide in &spec.guides {
                    let channel = self.map_aesthetic_name(&guide.aesthetic);
                    // Share legends across layers
                    resolve["legend"][&channel] = json!("shared");
                    resolve["scale"][&channel] = json!("shared");
                }
                vl_spec["resolve"] = resolve;
            }
        }

        // Apply coordinate transformations (may modify vl_spec and/or transform DataFrame)
        let transformed_data = self.apply_coord_transforms(spec, data, &mut vl_spec)?;
        let data_to_use = transformed_data.as_ref().unwrap_or(data);

        // Convert DataFrame to Vega-Lite data values
        let data_values = self.dataframe_to_values(data_to_use)?;
        vl_spec["data"] = json!({"values": data_values});

        // Handle faceting if present
        if let Some(facet) = &spec.facet {
            use crate::parser::ast::Facet;
            match facet {
                Facet::Wrap { variables, .. } => {
                    if !variables.is_empty() {
                        let field_type = self.infer_field_type(data_to_use, &variables[0]);
                        vl_spec["facet"] = json!({
                            "field": variables[0],
                            "type": field_type,
                        });

                        // Move mark/encoding into spec
                        let mut spec_inner = json!({});
                        if let Some(mark) = vl_spec.get("mark") {
                            spec_inner["mark"] = mark.clone();
                        }
                        if let Some(encoding) = vl_spec.get("encoding") {
                            spec_inner["encoding"] = encoding.clone();
                        }
                        if let Some(layer) = vl_spec.get("layer") {
                            spec_inner["layer"] = layer.clone();
                        }

                        vl_spec["spec"] = spec_inner;
                        vl_spec.as_object_mut().unwrap().remove("mark");
                        vl_spec.as_object_mut().unwrap().remove("encoding");
                        vl_spec.as_object_mut().unwrap().remove("layer");
                    }
                }
                Facet::Grid { rows, cols, .. } => {
                    // Grid faceting: use row and column
                    let mut facet_spec = Map::new();
                    if !rows.is_empty() {
                        let field_type = self.infer_field_type(data_to_use, &rows[0]);
                        facet_spec.insert(
                            "row".to_string(),
                            json!({"field": rows[0], "type": field_type}),
                        );
                    }
                    if !cols.is_empty() {
                        let field_type = self.infer_field_type(data_to_use, &cols[0]);
                        facet_spec.insert(
                            "column".to_string(),
                            json!({"field": cols[0], "type": field_type}),
                        );
                    }
                    vl_spec["facet"] = Value::Object(facet_spec);

                    // Move mark/encoding/layer into spec
                    let mut spec_inner = json!({});
                    if let Some(mark) = vl_spec.get("mark") {
                        spec_inner["mark"] = mark.clone();
                    }
                    if let Some(encoding) = vl_spec.get("encoding") {
                        spec_inner["encoding"] = encoding.clone();
                    }
                    if let Some(layer) = vl_spec.get("layer") {
                        spec_inner["layer"] = layer.clone();
                    }

                    vl_spec["spec"] = spec_inner;
                    vl_spec.as_object_mut().unwrap().remove("mark");
                    vl_spec.as_object_mut().unwrap().remove("encoding");
                    vl_spec.as_object_mut().unwrap().remove("layer");
                }
            }
        }

        // Serialize to pretty JSON
        serde_json::to_string_pretty(&vl_spec).map_err(|e| {
            GgsqlError::WriterError(format!("Failed to serialize Vega-Lite JSON: {}", e))
        })
    }

    fn validate(&self, spec: &VizSpec) -> Result<()> {
        // Check that we have at least one layer
        if spec.layers.is_empty() {
            return Err(GgsqlError::ValidationError(
                "VegaLiteWriter requires at least one layer".to_string(),
            ));
        }

        // Validate each layer has required aesthetics
        for layer in &spec.layers {
            layer.validate_required_aesthetics().map_err(|e| {
                GgsqlError::ValidationError(format!("Layer validation failed: {}", e))
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Layer, Labels};
    use std::collections::HashMap;

    #[test]
    fn test_geom_to_mark_mapping() {
        let writer = VegaLiteWriter::new();
        assert_eq!(writer.geom_to_mark(&Geom::Point), "point");
        assert_eq!(writer.geom_to_mark(&Geom::Line), "line");
        assert_eq!(writer.geom_to_mark(&Geom::Bar), "bar");
        assert_eq!(writer.geom_to_mark(&Geom::Area), "area");
        assert_eq!(writer.geom_to_mark(&Geom::Tile), "rect");
    }

    #[test]
    fn test_aesthetic_name_mapping() {
        let writer = VegaLiteWriter::new();
        assert_eq!(writer.map_aesthetic_name("x"), "x");
        assert_eq!(writer.map_aesthetic_name("fill"), "color");
        assert_eq!(writer.map_aesthetic_name("alpha"), "opacity");
    }

    #[test]
    fn test_validation_requires_layers() {
        let writer = VegaLiteWriter::new();
        let spec = VizSpec::new();
        assert!(writer.validate(&spec).is_err());
    }

    #[test]
    fn test_simple_point_spec() {
        let writer = VegaLiteWriter::new();

        // Create a simple spec
        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Create simple DataFrame
        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        // Generate Vega-Lite JSON
        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Verify structure
        assert_eq!(vl_spec["$schema"], writer.schema);
        assert_eq!(vl_spec["mark"], "point");
        assert!(vl_spec["data"]["values"].is_array());
        assert_eq!(vl_spec["data"]["values"].as_array().unwrap().len(), 3);
        assert!(vl_spec["encoding"]["x"].is_object());
        assert!(vl_spec["encoding"]["y"].is_object());
    }

    #[test]
    fn test_with_title() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("title".to_string(), "My Chart".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01-01", "2024-01-02"],
            "value" => &[10, 20],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "My Chart");
        assert_eq!(vl_spec["mark"], "line");
    }

    #[test]
    fn test_literal_color() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(LiteralValue::String("blue".to_string())),
            );
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["value"], "blue");
    }

    #[test]
    fn test_missing_column_error() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("foo".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let result = writer.write(&spec, &df);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Column 'foo'"));
        assert!(err_msg.contains("does not exist"));
        assert!(err_msg.contains("Available columns: x, y"));
    }

    #[test]
    fn test_missing_column_in_multi_layer() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();

        // First layer is valid
        let layer1 = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer1);

        // Second layer references non-existent column
        let layer2 = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("missing_col".to_string()));
        spec.layers.push(layer2);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let result = writer.write(&spec, &df);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Column 'missing_col'"));
        assert!(err_msg.contains("layer 2"));
    }

    // ========================================
    // Comprehensive Grammar Coverage Tests
    // ========================================

    #[test]
    fn test_all_basic_geom_types() {
        let writer = VegaLiteWriter::new();

        let geoms = vec![
            (Geom::Point, "point"),
            (Geom::Line, "line"),
            (Geom::Path, "line"),
            (Geom::Bar, "bar"),
            (Geom::Col, "bar"),
            (Geom::Area, "area"),
            (Geom::Tile, "rect"),
            (Geom::Ribbon, "area"),
        ];

        for (geom, expected_mark) in geoms {
            let mut spec = VizSpec::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2, 3],
                "y" => &[4, 5, 6],
            }
            .unwrap();

            let json_str = writer.write(&spec, &df).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            assert_eq!(
                vl_spec["mark"].as_str().unwrap(),
                expected_mark,
                "Failed for geom: {:?}",
                geom
            );
        }
    }

    #[test]
    fn test_statistical_geom_types() {
        let writer = VegaLiteWriter::new();

        let geoms = vec![
            (Geom::Histogram, "bar"),
            (Geom::Density, "area"),
            (Geom::Boxplot, "boxplot"),
        ];

        for (geom, expected_mark) in geoms {
            let mut spec = VizSpec::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2, 3],
                "y" => &[4, 5, 6],
            }
            .unwrap();

            let json_str = writer.write(&spec, &df).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            assert_eq!(vl_spec["mark"].as_str().unwrap(), expected_mark);
        }
    }

    #[test]
    fn test_text_geom_types() {
        let writer = VegaLiteWriter::new();

        for geom in [Geom::Text, Geom::Label] {
            let mut spec = VizSpec::new();
            let layer = Layer::new(geom.clone())
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
            spec.layers.push(layer);

            let df = df! {
                "x" => &[1, 2],
                "y" => &[3, 4],
            }
            .unwrap();

            let json_str = writer.write(&spec, &df).unwrap();
            let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

            assert_eq!(vl_spec["mark"].as_str().unwrap(), "text");
        }
    }

    #[test]
    fn test_color_aesthetic_column() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "A"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["field"], "category");
        assert_eq!(vl_spec["encoding"]["color"]["type"], "nominal");
    }

    #[test]
    fn test_size_aesthetic_column() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["size"]["field"], "value");
        assert_eq!(vl_spec["encoding"]["size"]["type"], "quantitative");
    }

    #[test]
    fn test_fill_aesthetic_mapping() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()))
            .with_aesthetic("fill".to_string(), AestheticValue::Column("region".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "category" => &["A", "B"],
            "value" => &[10, 20],
            "region" => &["US", "EU"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // 'fill' should be mapped to 'color' in Vega-Lite
        assert_eq!(vl_spec["encoding"]["color"]["field"], "region");
    }

    #[test]
    fn test_alpha_aesthetic_mapping() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("alpha".to_string(), AestheticValue::Literal(LiteralValue::Number(0.5)));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // 'alpha' should be mapped to 'opacity' in Vega-Lite
        assert_eq!(vl_spec["encoding"]["opacity"]["value"], 0.5);
    }

    #[test]
    fn test_multiple_aesthetics() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Column("value".to_string()))
            .with_aesthetic("shape".to_string(), AestheticValue::Column("type".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
            "type" => &["T1", "T2", "T1"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["x"]["field"], "x");
        assert_eq!(vl_spec["encoding"]["y"]["field"], "y");
        assert_eq!(vl_spec["encoding"]["color"]["field"], "category");
        assert_eq!(vl_spec["encoding"]["size"]["field"], "value");
        assert_eq!(vl_spec["encoding"]["shape"]["field"], "type");
    }

    #[test]
    fn test_literal_number_value() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Literal(LiteralValue::Number(100.0)));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["size"]["value"], 100.0);
    }

    #[test]
    fn test_literal_boolean_value() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("linetype".to_string(), AestheticValue::Literal(LiteralValue::Boolean(true)));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["linetype"]["value"], true);
    }

    #[test]
    fn test_multi_layer_composition() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();

        // First layer: line
        let layer1 = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer1);

        // Second layer: points
        let layer2 = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Literal(LiteralValue::String("red".to_string())));
        spec.layers.push(layer2);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have layer array
        assert!(vl_spec["layer"].is_array());
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        // Check first layer
        assert_eq!(layers[0]["mark"], "line");
        assert_eq!(layers[0]["encoding"]["x"]["field"], "x");
        assert_eq!(layers[0]["encoding"]["y"]["field"], "y");

        // Check second layer
        assert_eq!(layers[1]["mark"], "point");
        assert_eq!(layers[1]["encoding"]["color"]["value"], "red");
    }

    #[test]
    fn test_three_layer_composition() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();

        // Layer 1: area
        spec.layers.push(
            Layer::new(Geom::Area)
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string())),
        );

        // Layer 2: line
        spec.layers.push(
            Layer::new(Geom::Line)
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string())),
        );

        // Layer 3: points
        spec.layers.push(
            Layer::new(Geom::Point)
                .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
                .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string())),
        );

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0]["mark"], "area");
        assert_eq!(layers[1]["mark"], "line");
        assert_eq!(layers[2]["mark"], "point");
    }

    #[test]
    fn test_label_title() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("title".to_string(), "Test Plot".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "x" => &[1, 2],
            "y" => &[3, 4],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "Test Plot");
    }

    #[test]
    fn test_label_axis_titles() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("revenue".to_string()));
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("x".to_string(), "Date".to_string());
        labels.labels.insert("y".to_string(), "Revenue ($M)".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "date" => &["2024-01", "2024-02", "2024-03"],
            "revenue" => &["100", "150", "200"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["x"]["title"], "Date");
        assert_eq!(vl_spec["encoding"]["y"]["title"], "Revenue ($M)");
    }

    #[test]
    fn test_label_title_and_axes() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("title".to_string(), "Sales by Category".to_string());
        labels.labels.insert("x".to_string(), "Product Category".to_string());
        labels.labels.insert("y".to_string(), "Sales Volume".to_string());
        spec.labels = Some(labels);

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 15],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["title"], "Sales by Category");
        assert_eq!(vl_spec["encoding"]["x"]["title"], "Product Category");
        assert_eq!(vl_spec["encoding"]["y"]["title"], "Sales Volume");
    }

    #[test]
    fn test_numeric_type_inference_integers() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_nominal_type_inference_strings() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["x"]["type"], "nominal");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");
    }

    #[test]
    fn test_numeric_string_type_inference() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &["1", "2", "3"],
            "y" => &["4.5", "5.5", "6.5"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Numeric strings should be inferred as quantitative
        assert_eq!(vl_spec["encoding"]["x"]["type"], "quantitative");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");

        // Values should be converted to numbers in JSON
        let data = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data[0]["x"], 1.0);
        assert_eq!(data[0]["y"], 4.5);
    }

    #[test]
    fn test_data_conversion_all_types() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("int_col".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("float_col".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "int_col" => &[1, 2, 3],
            "float_col" => &[1.5, 2.5, 3.5],
            "string_col" => &["a", "b", "c"],
            "bool_col" => &[true, false, true],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data.len(), 3);

        // Check first row
        assert_eq!(data[0]["int_col"], 1);
        assert_eq!(data[0]["float_col"], 1.5);
        assert_eq!(data[0]["string_col"], "a");
        assert_eq!(data[0]["bool_col"], true);
    }

    #[test]
    fn test_empty_dataframe() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        let df = df! {
            "x" => &[] as &[i32],
            "y" => &[] as &[i32],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_large_dataset() {
        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Create dataset with 100 rows
        let x_vals: Vec<i32> = (1..=100).collect();
        let y_vals: Vec<i32> = (1..=100).map(|i| i * 2).collect();

        let df = df! {
            "x" => x_vals,
            "y" => y_vals,
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        let data = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data.len(), 100);
        assert_eq!(data[0]["x"], 1);
        assert_eq!(data[0]["y"], 2);
        assert_eq!(data[99]["x"], 100);
        assert_eq!(data[99]["y"], 200);
    }

    // ========================================
    // Guide Tests
    // ========================================

    #[test]
    fn test_guide_none_hides_legend() {
        use crate::parser::ast::{Guide, GuideType};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()));
        spec.layers.push(layer);

        // Add guide to hide color legend
        spec.guides.push(Guide {
            aesthetic: "color".to_string(),
            guide_type: Some(GuideType::None),
            properties: HashMap::new(),
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["legend"], json!(null));
    }

    #[test]
    fn test_guide_legend_with_title() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()));
        spec.layers.push(layer);

        // Add guide with custom title
        let mut properties = HashMap::new();
        properties.insert("title".to_string(), GuidePropertyValue::String("Product Type".to_string()));
        spec.guides.push(Guide {
            aesthetic: "color".to_string(),
            guide_type: Some(GuideType::Legend),
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["legend"]["title"], "Product Type");
    }

    #[test]
    fn test_guide_legend_position() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add guide with custom position
        let mut properties = HashMap::new();
        properties.insert("position".to_string(), GuidePropertyValue::String("bottom".to_string()));
        spec.guides.push(Guide {
            aesthetic: "size".to_string(),
            guide_type: Some(GuideType::Legend),
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // position maps to orient in Vega-Lite
        assert_eq!(vl_spec["encoding"]["size"]["legend"]["orient"], "bottom");
    }

    #[test]
    fn test_guide_colorbar() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("temperature".to_string()));
        spec.layers.push(layer);

        // Add colorbar guide
        let mut properties = HashMap::new();
        properties.insert("title".to_string(), GuidePropertyValue::String("Temperature (°C)".to_string()));
        spec.guides.push(Guide {
            aesthetic: "color".to_string(),
            guide_type: Some(GuideType::ColorBar),
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "temperature" => &[20, 25, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["legend"]["type"], "gradient");
        assert_eq!(vl_spec["encoding"]["color"]["legend"]["title"], "Temperature (°C)");
    }

    #[test]
    fn test_guide_axis() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add axis guide for x
        let mut properties = HashMap::new();
        properties.insert("title".to_string(), GuidePropertyValue::String("Product Category".to_string()));
        properties.insert("text_angle".to_string(), GuidePropertyValue::Number(45.0));
        spec.guides.push(Guide {
            aesthetic: "x".to_string(),
            guide_type: Some(GuideType::Axis),
            properties,
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["x"]["axis"]["title"], "Product Category");
        assert_eq!(vl_spec["encoding"]["x"]["axis"]["labelAngle"], 45.0);
    }

    #[test]
    fn test_multiple_guides() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add guide for color
        let mut color_props = HashMap::new();
        color_props.insert("title".to_string(), GuidePropertyValue::String("Category".to_string()));
        color_props.insert("position".to_string(), GuidePropertyValue::String("right".to_string()));
        spec.guides.push(Guide {
            aesthetic: "color".to_string(),
            guide_type: Some(GuideType::Legend),
            properties: color_props,
        });

        // Add guide for size
        let mut size_props = HashMap::new();
        size_props.insert("title".to_string(), GuidePropertyValue::String("Value".to_string()));
        spec.guides.push(Guide {
            aesthetic: "size".to_string(),
            guide_type: Some(GuideType::Legend),
            properties: size_props,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(vl_spec["encoding"]["color"]["legend"]["title"], "Category");
        assert_eq!(vl_spec["encoding"]["color"]["legend"]["orient"], "right");
        assert_eq!(vl_spec["encoding"]["size"]["legend"]["title"], "Value");
    }

    #[test]
    fn test_guide_fill_maps_to_color() {
        use crate::parser::ast::{Guide, GuideType, GuidePropertyValue};

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()))
            .with_aesthetic("fill".to_string(), AestheticValue::Column("region".to_string()));
        spec.layers.push(layer);

        // Add guide for fill (should map to color)
        let mut properties = HashMap::new();
        properties.insert("title".to_string(), GuidePropertyValue::String("Region".to_string()));
        spec.guides.push(Guide {
            aesthetic: "fill".to_string(),
            guide_type: Some(GuideType::Legend),
            properties,
        });

        let df = df! {
            "category" => &["A", "B"],
            "value" => &[10, 20],
            "region" => &["US", "EU"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // fill should be mapped to color channel
        assert_eq!(vl_spec["encoding"]["color"]["field"], "region");
        assert_eq!(vl_spec["encoding"]["color"]["legend"]["title"], "Region");
    }

    // ========================================
    // COORD Clause Tests
    // ========================================

    #[test]
    fn test_coord_cartesian_xlim() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Add COORD cartesian with xlim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(100.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that x scale has domain set
        assert_eq!(vl_spec["encoding"]["x"]["scale"]["domain"], json!([0.0, 100.0]));
    }

    #[test]
    fn test_coord_cartesian_ylim() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Add COORD cartesian with ylim
        let mut properties = HashMap::new();
        properties.insert(
            "ylim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(-10.0),
                ArrayElement::Number(50.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that y scale has domain set
        assert_eq!(vl_spec["encoding"]["y"]["scale"]["domain"], json!([-10.0, 50.0]));
    }

    #[test]
    fn test_coord_cartesian_xlim_ylim() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Add COORD cartesian with both xlim and ylim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(100.0),
            ]),
        );
        properties.insert(
            "ylim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(200.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[50, 100, 150],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check both domains
        assert_eq!(vl_spec["encoding"]["x"]["scale"]["domain"], json!([0.0, 100.0]));
        assert_eq!(vl_spec["encoding"]["y"]["scale"]["domain"], json!([0.0, 200.0]));
    }

    #[test]
    fn test_coord_cartesian_reversed_limits_auto_swap() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer);

        // Add COORD with reversed xlim (should auto-swap)
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(100.0),
                ArrayElement::Number(0.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[10, 20, 30],
            "y" => &[4, 5, 6],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should be swapped to [0, 100]
        assert_eq!(vl_spec["encoding"]["x"]["scale"]["domain"], json!([0.0, 100.0]));
    }

    #[test]
    fn test_coord_cartesian_aesthetic_domain() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()));
        spec.layers.push(layer);

        // Add COORD with color domain
        let mut properties = HashMap::new();
        properties.insert(
            "color".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::String("A".to_string()),
                ArrayElement::String("B".to_string()),
                ArrayElement::String("C".to_string()),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "A"],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that color scale has domain set
        assert_eq!(
            vl_spec["encoding"]["color"]["scale"]["domain"],
            json!(["A", "B", "C"])
        );
    }

    #[test]
    fn test_coord_cartesian_multi_layer() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();

        // First layer: line
        let layer1 = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer1);

        // Second layer: points
        let layer2 = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()));
        spec.layers.push(layer2);

        // Add COORD with xlim and ylim
        let mut properties = HashMap::new();
        properties.insert(
            "xlim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(0.0),
                ArrayElement::Number(10.0),
            ]),
        );
        properties.insert(
            "ylim".to_string(),
            CoordPropertyValue::Array(vec![
                ArrayElement::Number(-5.0),
                ArrayElement::Number(5.0),
            ]),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Cartesian,
            properties,
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[1, 2, 3],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that both layers have the limits applied
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        for layer in layers {
            assert_eq!(layer["encoding"]["x"]["scale"]["domain"], json!([0.0, 10.0]));
            assert_eq!(layer["encoding"]["y"]["scale"]["domain"], json!([-5.0, 5.0]));
        }
    }

    #[test]
    fn test_coord_flip_single_layer() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add custom axis labels
        let mut labels = Labels {
            labels: HashMap::new(),
        };
        labels.labels.insert("x".to_string(), "Category".to_string());
        labels.labels.insert("y".to_string(), "Value".to_string());
        spec.labels = Some(labels);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // After flip: x should have "value" field, y should have "category" field
        assert_eq!(vl_spec["encoding"]["x"]["field"], "value");
        assert_eq!(vl_spec["encoding"]["y"]["field"], "category");

        // But titles should preserve original aesthetic names (ggplot2 style)
        assert_eq!(vl_spec["encoding"]["x"]["title"], "Value");
        assert_eq!(vl_spec["encoding"]["y"]["title"], "Category");
    }

    #[test]
    fn test_coord_flip_multi_layer() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();

        // First layer: bar
        let layer1 = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer1);

        // Second layer: point
        let layer2 = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer2);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check both layers have flipped encodings
        let layers = vl_spec["layer"].as_array().unwrap();
        assert_eq!(layers.len(), 2);

        for layer in layers {
            assert_eq!(layer["encoding"]["x"]["field"], "value");
            assert_eq!(layer["encoding"]["y"]["field"], "category");
        }
    }

    #[test]
    fn test_coord_flip_preserves_other_aesthetics() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("x".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("y".to_string()))
            .with_aesthetic("color".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("size".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add COORD flip
        spec.coord = Some(Coord {
            coord_type: CoordType::Flip,
            properties: HashMap::new(),
        });

        let df = df! {
            "x" => &[1, 2, 3],
            "y" => &[4, 5, 6],
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check x and y are flipped
        assert_eq!(vl_spec["encoding"]["x"]["field"], "y");
        assert_eq!(vl_spec["encoding"]["y"]["field"], "x");

        // Check color and size are unchanged
        assert_eq!(vl_spec["encoding"]["color"]["field"], "category");
        assert_eq!(vl_spec["encoding"]["size"]["field"], "value");
    }

    #[test]
    fn test_coord_polar_basic_pie_chart() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add COORD polar (defaults to theta = y)
        spec.coord = Some(Coord {
            coord_type: CoordType::Polar,
            properties: HashMap::new(),
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Bar in polar should become arc
        assert_eq!(vl_spec["mark"], "arc");

        // y should be mapped to theta
        assert!(vl_spec["encoding"]["theta"].is_object());
        assert_eq!(vl_spec["encoding"]["theta"]["field"], "value");

        // x should be removed from positional encoding
        assert!(vl_spec["encoding"]["x"].is_null() || !vl_spec["encoding"].as_object().unwrap().contains_key("x"));

        // x should be mapped to color (for category differentiation)
        assert!(vl_spec["encoding"]["color"].is_object());
        assert_eq!(vl_spec["encoding"]["color"]["field"], "category");
    }

    #[test]
    fn test_coord_polar_with_theta_property() {
        use crate::parser::ast::Coord;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Bar)
            .with_aesthetic("x".to_string(), AestheticValue::Column("category".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Add COORD polar with explicit theta = y
        let mut properties = HashMap::new();
        properties.insert(
            "theta".to_string(),
            CoordPropertyValue::String("y".to_string()),
        );
        spec.coord = Some(Coord {
            coord_type: CoordType::Polar,
            properties,
        });

        let df = df! {
            "category" => &["A", "B", "C"],
            "value" => &[10, 20, 30],
        }
        .unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should produce same result as default
        assert_eq!(vl_spec["mark"], "arc");
        assert_eq!(vl_spec["encoding"]["theta"]["field"], "value");
    }

    #[test]
    fn test_date_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Create DataFrame with Date type
        let dates = Series::new("date".into(), &[0i32, 1, 2]) // Days since epoch
            .cast(&DataType::Date)
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![dates, values]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that dates are formatted as ISO strings in data
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["date"], "1970-01-01");
        assert_eq!(data_values[1]["date"], "1970-01-02");
        assert_eq!(data_values[2]["date"], "1970-01-03");
    }

    #[test]
    fn test_datetime_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("datetime".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Create DataFrame with Datetime type (microseconds since epoch)
        let datetimes = Series::new("datetime".into(), &[0i64, 1_000_000, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![datetimes, values]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that datetimes are formatted as ISO strings in data
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["datetime"], "1970-01-01T00:00:00.000Z");
        assert_eq!(data_values[1]["datetime"], "1970-01-01T00:00:01.000Z");
        assert_eq!(data_values[2]["datetime"], "1970-01-01T00:00:02.000Z");
    }

    #[test]
    fn test_time_series_to_iso_format() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::Column("time".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Create DataFrame with Time type (nanoseconds since midnight)
        let times = Series::new("time".into(), &[0i64, 3_600_000_000_000, 7_200_000_000_000])
            .cast(&DataType::Time)
            .unwrap();
        let values = Series::new("value".into(), &[10, 20, 30]);
        let df = DataFrame::new(vec![times, values]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Check that times are formatted as ISO time strings in data
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["time"], "00:00:00.000");
        assert_eq!(data_values[1]["time"], "01:00:00.000");
        assert_eq!(data_values[2]["time"], "02:00:00.000");
    }

    #[test]
    fn test_automatic_temporal_type_inference() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("revenue".to_string()));
        spec.layers.push(layer);

        // Create DataFrame with Date type - NO explicit SCALE x SETTING type => 'date' needed!
        let dates = Series::new("date".into(), &[0i32, 1, 2, 3, 4])
            .cast(&DataType::Date)
            .unwrap();
        let revenue = Series::new("revenue".into(), &[100, 120, 110, 130, 125]);
        let df = DataFrame::new(vec![dates, revenue]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // CRITICAL TEST: x-axis should automatically be inferred as "temporal" type
        assert_eq!(vl_spec["encoding"]["x"]["type"], "temporal");
        assert_eq!(vl_spec["encoding"]["y"]["type"], "quantitative");

        // Dates should be formatted as ISO strings
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["date"], "1970-01-01");
        assert_eq!(data_values[1]["date"], "1970-01-02");
    }

    #[test]
    fn test_datetime_automatic_temporal_inference() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Area)
            .with_aesthetic("x".to_string(), AestheticValue::Column("timestamp".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        // Create DataFrame with Datetime type
        let timestamps = Series::new("timestamp".into(), &[0i64, 86_400_000_000, 172_800_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let values = Series::new("value".into(), &[50, 75, 60]);
        let df = DataFrame::new(vec![timestamps, values]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // x-axis should automatically be inferred as "temporal" type
        assert_eq!(vl_spec["encoding"]["x"]["type"], "temporal");

        // Timestamps should be formatted as ISO datetime strings
        let data_values = vl_spec["data"]["values"].as_array().unwrap();
        assert_eq!(data_values[0]["timestamp"], "1970-01-01T00:00:00.000Z");
        assert_eq!(data_values[1]["timestamp"], "1970-01-02T00:00:00.000Z");
        assert_eq!(data_values[2]["timestamp"], "1970-01-03T00:00:00.000Z");
    }

    // ========================================
    // PARTITION BY Tests
    // ========================================

    #[test]
    fn test_partition_by_single_column_generates_detail() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()))
            .with_partition_by(vec!["category".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02", "2024-01-03"]);
        let values = Series::new("value".into(), &[100, 120, 110]);
        let categories = Series::new("category".into(), &["A", "A", "B"]);
        let df = DataFrame::new(vec![dates, values, categories]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have detail encoding with the partition_by column
        assert!(vl_spec["encoding"]["detail"].is_object());
        assert_eq!(vl_spec["encoding"]["detail"]["field"], "category");
        assert_eq!(vl_spec["encoding"]["detail"]["type"], "nominal");
    }

    #[test]
    fn test_partition_by_multiple_columns_generates_detail_array() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()))
            .with_partition_by(vec!["category".to_string(), "region".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let categories = Series::new("category".into(), &["A", "B"]);
        let regions = Series::new("region".into(), &["North", "South"]);
        let df = DataFrame::new(vec![dates, values, categories, regions]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should have detail encoding as an array
        assert!(vl_spec["encoding"]["detail"].is_array());
        let details = vl_spec["encoding"]["detail"].as_array().unwrap();
        assert_eq!(details.len(), 2);
        assert_eq!(details[0]["field"], "category");
        assert_eq!(details[0]["type"], "nominal");
        assert_eq!(details[1]["field"], "region");
        assert_eq!(details[1]["type"], "nominal");
    }

    #[test]
    fn test_no_partition_by_no_detail() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()));
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let df = DataFrame::new(vec![dates, values]).unwrap();

        let json_str = writer.write(&spec, &df).unwrap();
        let vl_spec: Value = serde_json::from_str(&json_str).unwrap();

        // Should NOT have detail encoding
        assert!(vl_spec["encoding"]["detail"].is_null());
    }

    #[test]
    fn test_partition_by_validation_missing_column() {
        use polars::prelude::*;

        let writer = VegaLiteWriter::new();

        let mut spec = VizSpec::new();
        let layer = Layer::new(Geom::Line)
            .with_aesthetic("x".to_string(), AestheticValue::Column("date".to_string()))
            .with_aesthetic("y".to_string(), AestheticValue::Column("value".to_string()))
            .with_partition_by(vec!["nonexistent_column".to_string()]);
        spec.layers.push(layer);

        let dates = Series::new("date".into(), &["2024-01-01", "2024-01-02"]);
        let values = Series::new("value".into(), &[100, 120]);
        let df = DataFrame::new(vec![dates, values]).unwrap();

        let result = writer.write(&spec, &df);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent_column"));
        assert!(err.contains("PARTITION BY"));
    }
}
