//! Bridging ggsql aesthetic mappings and DataFrame columns to the typed data
//! hephaestus geoms consume.

use arrow::array::{Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
use arrow::datatypes::DataType;

use hephaestus::color::Color;
use hephaestus::plot::geom::{BuildableGeom, GeomBuilder};
use hephaestus::scales::geometry::{Coord, Geometry, Polygon as GeoPolygon};

use super::scales::parse_color;
use crate::array_util::{as_bool, as_f64, cast_array, value_to_string};
use crate::{AestheticValue, DataFrame, GgsqlError, Layer, Result};

/// A column extracted in the type hephaestus expects for a channel: numeric
/// columns become `f64`s, text columns become category strings.
#[derive(Clone)]
pub enum ChannelData {
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

impl ChannelData {
    /// Select a subset of rows by index, preserving the channel's value type.
    pub fn select(&self, idx: &[usize]) -> ChannelData {
        match self {
            ChannelData::Floats(v) => ChannelData::Floats(idx.iter().map(|&i| v[i]).collect()),
            ChannelData::Strings(v) => {
                ChannelData::Strings(idx.iter().map(|&i| v[i].clone()).collect())
            }
        }
    }

    /// Set this column on a geom builder under the given channel.
    pub fn apply<G: BuildableGeom>(self, builder: &mut GeomBuilder<G>, channel: &str) {
        match self {
            ChannelData::Floats(values) => {
                builder.set(channel, values);
            }
            ChannelData::Strings(values) => {
                builder.set(channel, values);
            }
        }
    }
}

/// The DataFrame column name backing the given internal aesthetic, if it maps
/// to a column (rather than a literal value).
pub fn aesthetic_column_name<'a>(layer: &'a Layer, aesthetic: &str) -> Option<&'a str> {
    match layer.mappings.get(aesthetic)? {
        AestheticValue::Column { name, .. } => Some(name.as_str()),
        AestheticValue::AnnotationColumn { name } => Some(name.as_str()),
        AestheticValue::Literal(_) => None,
    }
}

/// The category a null stands in for when a scaled column reaches hephaestus.
///
/// ggsql trains a categorical domain over the nulls as well, so `NULL` is a
/// level like any other and gets its own colour and legend key. hephaestus's
/// `DataColumn` has no null-carrying variant, though, so a null row cannot be
/// handed over as the `Value::Null` sitting in the domain — it would resolve to
/// nothing and the mark would draw unfilled. Both sides therefore agree on this
/// sentinel instead: [`scales::category_value`] puts it in the domain and in the
/// break positions, and [`column_to_channel`] puts it in the data. The visible
/// text is unaffected, because labels travel separately (`with_breaks_labeled`).
///
/// It carries the internal `__ggsql_` prefix so a real category cannot collide
/// with it.
pub const NULL_CATEGORY: &str = "__ggsql_null__";

/// Extract a column as the channel type implied by its arrow dtype: text and
/// booleans → category strings, everything else → `f64`.
///
/// This is the *scaled* path — the values here are looked up in a scale's
/// domain — so a null becomes [`NULL_CATEGORY`] rather than the empty string
/// [`column_to_strings`] uses for raw, unscaled text.
///
/// A boolean is a category, not a number: ggsql trains a discrete domain over
/// it (as the Vega-Lite writer's `nominal` type does) and hephaestus matches
/// data to domain by `Value` variant, so a `true` handed over as `1.0` would
/// find no `Bool` entry and resolve to nothing. Both sides therefore agree on
/// the category name — [`scales::category_value`] renders the domain the same
/// way, and `wiring::constant_position` a literal.
pub fn column_to_channel(df: &DataFrame, name: &str) -> Result<ChannelData> {
    let array = df.column(name)?;
    if is_text_column(df, name) || matches!(array.data_type(), DataType::Boolean) {
        Ok(ChannelData::Strings(
            crate::array_util::column_to_strings_or(array, NULL_CATEGORY)?,
        ))
    } else {
        Ok(ChannelData::Floats(column_to_f64(df, name)?))
    }
}

/// Whether a column holds text, and so must be read with [`column_to_strings`]
/// rather than cast to `f64`.
///
/// Arrow's cast is a *safe* one: a text column cast to `Float64` comes back as
/// all-nulls — an `Ok` full of `NaN` — rather than an error. A caller that
/// accepts either a number or a keyword therefore has to ask this first; trying
/// [`column_to_f64`] and falling back on `Err` never reaches the keywords.
pub fn is_text_column(df: &DataFrame, name: &str) -> bool {
    df.column(name)
        .is_ok_and(|array| matches!(array.data_type(), DataType::Utf8 | DataType::LargeUtf8))
}

/// Read a numeric column as `f64`, casting from any numeric/temporal source
/// type and mapping nulls to `NaN`.
pub fn column_to_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let array = df.column(name)?;
    let casted;
    let f64_array = if matches!(array.data_type(), DataType::Float64) {
        as_f64(array)?
    } else {
        casted = cast_array(array, &DataType::Float64)?;
        as_f64(&casted)?
    };
    Ok(f64_array.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Read a column as strings, casting non-text columns to text. Nulls become
/// empty strings — this is the raw, unscaled path (text labels, shape and
/// linetype names), where an empty string is the right "nothing here".
pub fn column_to_strings(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    crate::array_util::column_to_strings_or(df.column(name)?, "")
}

/// Build a per-row group key from the layer's partition columns (concatenated
/// values), used as the hephaestus `keys` for multi-vertex geoms. Returns
/// `None` when there are no partition columns (single group).
pub fn build_group_keys(df: &DataFrame, partition_by: &[String]) -> Result<Option<Vec<String>>> {
    if partition_by.is_empty() {
        return Ok(None);
    }
    let arrays: Vec<&ArrayRef> = partition_by
        .iter()
        .map(|c| df.column(c))
        .collect::<Result<_>>()?;
    let keys = (0..df.height())
        .map(|i| {
            let mut key = String::new();
            for arr in &arrays {
                key.push_str(&value_to_string(arr, i));
                key.push('\u{1f}'); // unit separator avoids cross-column collisions
            }
            key
        })
        .collect();
    Ok(Some(keys))
}

/// Read a boolean column (arrow Boolean, or text `true`/`1`). Nulls → false.
pub fn column_to_bool(df: &DataFrame, name: &str) -> Result<Vec<bool>> {
    let array = df.column(name)?;
    if matches!(array.data_type(), DataType::Boolean) {
        let a = as_bool(array)?;
        Ok((0..a.len()).map(|i| !a.is_null(i) && a.value(i)).collect())
    } else {
        Ok(column_to_strings(df, name)?
            .iter()
            .map(|s| matches!(s.to_lowercase().as_str(), "true" | "1"))
            .collect())
    }
}

/// Read a color column (visual-space literal values) as parsed colors,
/// defaulting unparseable entries to black.
pub fn column_to_colors(df: &DataFrame, name: &str) -> Result<Vec<Color>> {
    Ok(column_to_strings(df, name)?
        .iter()
        .map(|s| parse_color(s).unwrap_or(Color::BLACK))
        .collect())
}

/// Read a geometry column into hephaestus `Geometry` values. ggsql's spatial
/// pipeline re-encodes the geometry aesthetic as WKB (arrow `Binary`), which we
/// decode via `Geometry::from_wkb`; hex-encoded WKB strings (PostGIS over ODBC)
/// are decoded too, mirroring the Vega-Lite writer's `parse_geometry_from_array`.
/// Null rows become `Geometry::Empty` (drawn as nothing).
pub fn column_to_geometry(df: &DataFrame, name: &str) -> Result<Vec<Geometry>> {
    let array = df.column(name)?;
    let parse = |bytes: &[u8]| -> Result<Geometry> {
        Geometry::from_wkb(bytes)
            .map_err(|e| GgsqlError::WriterError(format!("could not parse WKB geometry: {e:?}")))
    };
    (0..array.len())
        .map(|i| {
            if array.is_null(i) {
                return Ok(Geometry::Empty);
            }
            match array.data_type() {
                DataType::Binary => parse(
                    array
                        .as_any()
                        .downcast_ref::<BinaryArray>()
                        .ok_or_else(|| geom_type_err("Binary"))?
                        .value(i),
                ),
                DataType::LargeBinary => parse(
                    array
                        .as_any()
                        .downcast_ref::<LargeBinaryArray>()
                        .ok_or_else(|| geom_type_err("LargeBinary"))?
                        .value(i),
                ),
                DataType::Utf8 => parse(&decode_hex_wkb(
                    array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| geom_type_err("Utf8"))?
                        .value(i),
                )?),
                DataType::LargeUtf8 => parse(&decode_hex_wkb(
                    array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| geom_type_err("LargeUtf8"))?
                        .value(i),
                )?),
                other => Err(GgsqlError::WriterError(format!(
                    "geometry column has unsupported type {other:?}; expected WKB (Binary)"
                ))),
            }
        })
        .collect()
}

fn geom_type_err(kind: &str) -> GgsqlError {
    GgsqlError::WriterError(format!("failed to read geometry column as {kind}"))
}

/// Decode a hex-encoded WKB string (optionally `\x`-prefixed, as PostGIS emits
/// over ODBC) into raw bytes.
fn decode_hex_wkb(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.strip_prefix("\\x").unwrap_or(hex);
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(hex.get(i..i + 2).unwrap_or(""), 16)
                .map_err(|_| GgsqlError::WriterError(format!("invalid hex in WKB at position {i}")))
        })
        .collect()
}

/// Parse a WKT string into a set of polylines (one per LineString). A
/// MultiLineString flattens to its parts; a bare LineString yields one line;
/// other geometry types contribute nothing. Used to feed graticule grid lines
/// to a map's Custom projection.
pub fn wkt_to_lines(wkt: &str) -> Vec<Vec<Coord>> {
    match Geometry::from_wkt(wkt) {
        Ok(Geometry::MultiLineString(lines)) => lines,
        Ok(Geometry::LineString(line)) => vec![line],
        _ => Vec::new(),
    }
}

/// Parse a WKT boundary string into polygon outlines for a Custom projection's
/// drawing surface. A Polygon yields one outline (with its holes); a
/// MultiPolygon yields all its parts; non-areal geometries yield nothing.
pub fn wkt_to_outline(wkt: &str) -> Vec<GeoPolygon> {
    match Geometry::from_wkt(wkt) {
        Ok(Geometry::Polygon(p)) => vec![p],
        Ok(Geometry::MultiPolygon(polys)) => polys,
        _ => Vec::new(),
    }
}
