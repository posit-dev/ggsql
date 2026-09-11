use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray,
    TimestampMillisecondArray,
};
use ggsql::array_util::value_to_string;
use ggsql::naming::DATA_PREFIX;
use ggsql::reader::sqlite::SqliteReader;
use ggsql::reader::Reader;
use ggsql::reader::Spec;
use ggsql::validate::validate;
use ggsql::writer::SvgWriter;
use ggsql::DataFrame;
use serde_json::json;
use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::prelude::*;

/// Report a panic to the console before the module aborts.
///
/// Composition asserts in a handful of documented cases and the `wasm` profile
/// sets `panic = "abort"`, so one bad plot traps the instance for the whole
/// page. The hook turns a bare `RuntimeError: unreachable executed` into a
/// console message naming the assertion. Runs on module instantiation.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// JS bridge declarations
// ============================================================================

thread_local! {
    static CONVERTERS: RefCell<Option<(js_sys::Function, js_sys::Function)>> =
        const { RefCell::new(None) };
}

/// Supply the JavaScript CSV and Parquet converters.
///
/// Called by the npm package's `init` wrapper; not intended for direct use.
#[wasm_bindgen(js_name = setConverters)]
pub fn set_converters(csv: js_sys::Function, parquet: js_sys::Function) {
    CONVERTERS.with(|converters| {
        *converters.borrow_mut() = Some((csv, parquet));
    });
}

fn converters() -> Result<(js_sys::Function, js_sys::Function), JsValue> {
    CONVERTERS.with(|converters| {
        converters.borrow().clone().ok_or_else(|| {
            JsValue::from_str(
                "CSV and Parquet converters are not configured; initialize through ggsql-wasm",
            )
        })
    })
}

fn convert_csv_js(data: &[u8]) -> Result<JsValue, JsValue> {
    let (convert_csv, _) = converters()?;
    let bytes = js_sys::Uint8Array::from(data);
    convert_csv.call1(&JsValue::UNDEFINED, &bytes)
}

async fn convert_parquet_js(data: &[u8]) -> Result<JsValue, JsValue> {
    let (_, convert_parquet) = converters()?;
    let bytes = js_sys::Uint8Array::from(data);
    let promise = convert_parquet
        .call1(&JsValue::UNDEFINED, &bytes)?
        .dyn_into::<js_sys::Promise>()?;
    wasm_bindgen_futures::JsFuture::from(promise).await
}

// ============================================================================
// SQLite VFS initialization (wasm32 only)
// ============================================================================

#[cfg(target_arch = "wasm32")]
fn ensure_vfs_initialized() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = sqlite_wasm_rs::MemVfsUtil::<sqlite_wasm_rs::WasmOsCallback>::new();
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_vfs_initialized() {
    // No VFS initialization needed on native targets
}

// ============================================================================
// Column descriptor → DataFrame conversion (for JS CSV/Parquet parsing)
// ============================================================================

/// Convert JS column descriptors to an Arrow-backed DataFrame.
fn columns_js_to_dataframe(columns_js: JsValue) -> Result<DataFrame, JsValue> {
    let columns = js_sys::Array::from(&columns_js);
    let len = columns.length();

    if len == 0 {
        return Ok(DataFrame::empty());
    }

    // Collect owned (name, array) pairs; DataFrame::new borrows the names so
    // we build a parallel Vec<String> to pin them for the lifetime of the call.
    let mut names: Vec<String> = Vec::with_capacity(len as usize);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(len as usize);

    for i in 0..len {
        let col = columns.get(i);
        let col_name = js_sys::Reflect::get(&col, &"name".into())
            .map_err(|_| JsValue::from_str("Missing column name"))?
            .as_string()
            .ok_or_else(|| JsValue::from_str("Column name is not a string"))?;
        let col_type = js_sys::Reflect::get(&col, &"type".into())
            .map_err(|_| JsValue::from_str("Missing column type"))?
            .as_string()
            .ok_or_else(|| JsValue::from_str("Column type is not a string"))?;
        let values_js = js_sys::Reflect::get(&col, &"values".into())
            .map_err(|_| JsValue::from_str("Missing column values"))?;
        let nulls_js = js_sys::Reflect::get(&col, &"nulls".into())
            .map_err(|_| JsValue::from_str("Missing column nulls"))?;

        let nulls = js_sys::Uint8Array::new(&nulls_js).to_vec();

        let array: ArrayRef = match col_type.as_str() {
            "f64" => {
                let raw = js_sys::Float64Array::new(&values_js).to_vec();
                let values: Vec<Option<f64>> = raw
                    .into_iter()
                    .zip(nulls.iter())
                    .map(|(v, &n)| if n != 0 { Some(v) } else { None })
                    .collect();
                Arc::new(Float64Array::from(values))
            }
            "i64" => {
                let raw = js_sys::Float64Array::new(&values_js).to_vec();
                let values: Vec<Option<i64>> = raw
                    .into_iter()
                    .zip(nulls.iter())
                    .map(|(v, &n)| if n != 0 { Some(v as i64) } else { None })
                    .collect();
                Arc::new(Int64Array::from(values))
            }
            "bool" => {
                let raw = js_sys::Uint8Array::new(&values_js).to_vec();
                let values: Vec<Option<bool>> = raw
                    .into_iter()
                    .zip(nulls.iter())
                    .map(|(v, &n)| if n != 0 { Some(v != 0) } else { None })
                    .collect();
                Arc::new(BooleanArray::from(values))
            }
            "string" => {
                let arr = js_sys::Array::from(&values_js);
                let values: Vec<Option<String>> = (0..arr.length())
                    .zip(nulls.iter())
                    .map(|(j, &n)| if n != 0 { arr.get(j).as_string() } else { None })
                    .collect();
                Arc::new(StringArray::from(values))
            }
            "binary" => {
                // One Uint8Array per row (e.g. WKB geometry from GeoParquet).
                let arr = js_sys::Array::from(&values_js);
                let values: Vec<Option<Vec<u8>>> = (0..arr.length())
                    .zip(nulls.iter())
                    .map(|(j, &n)| {
                        if n != 0 {
                            Some(js_sys::Uint8Array::new(&arr.get(j)).to_vec())
                        } else {
                            None
                        }
                    })
                    .collect();
                Arc::new(BinaryArray::from_iter(values.iter().map(|o| o.as_deref())))
            }
            "date" => {
                // Date32: days since Unix epoch
                let raw = js_sys::Float64Array::new(&values_js).to_vec();
                let values: Vec<Option<i32>> = raw
                    .into_iter()
                    .zip(nulls.iter())
                    .map(|(v, &n)| if n != 0 { Some(v as i32) } else { None })
                    .collect();
                Arc::new(Date32Array::from(values))
            }
            "datetime" => {
                // Timestamp(Millisecond): milliseconds since Unix epoch
                let raw = js_sys::Float64Array::new(&values_js).to_vec();
                let values: Vec<Option<i64>> = raw
                    .into_iter()
                    .zip(nulls.iter())
                    .map(|(v, &n)| if n != 0 { Some(v as i64) } else { None })
                    .collect();
                Arc::new(TimestampMillisecondArray::from(values))
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "Unknown column type: '{}'",
                    other
                )));
            }
        };

        names.push(col_name);
        arrays.push(array);
    }

    let named: Vec<(&str, ArrayRef)> = names
        .iter()
        .zip(arrays)
        .map(|(n, a)| (n.as_str(), a))
        .collect();

    DataFrame::new(named)
        .map_err(|e| JsValue::from_str(&format!("DataFrame creation error: {}", e)))
}

// ============================================================================
// GgsqlContext - public WASM API
// ============================================================================

/// Persistent ggsql context for WASM
///
/// Create once and reuse for multiple queries to avoid memory issues.
/// Uses interior mutability to avoid wasm_bindgen's &mut self aliasing issues.
#[wasm_bindgen]
pub struct GgsqlContext {
    reader: RefCell<SqliteReader>,
}

#[wasm_bindgen]
impl GgsqlContext {
    /// Create a new ggsql context
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<GgsqlContext, JsValue> {
        ensure_vfs_initialized();

        let reader = SqliteReader::new()
            .map_err(|e| JsValue::from_str(&format!("Failed to create SQLite reader: {:?}", e)))?;
        Ok(GgsqlContext {
            reader: RefCell::new(reader),
        })
    }

    /// Run a ggsql query and keep the resolved plot, ready to draw.
    ///
    /// Drawing is separate because a resize re-solves the layout, and doing
    /// that through the query would put SQL behind every frame of a drag.
    pub fn execute(&self, query: &str) -> Result<GgsqlPlot, JsValue> {
        let reader = self.reader.borrow();
        let spec = reader
            .execute(query)
            .map_err(|e| JsValue::from_str(&format!("Execute error: {:?}", e)))?;
        Ok(GgsqlPlot { spec })
    }

    /// Check whether a query contains a VISUALISE clause
    pub fn has_visual(&self, query: &str) -> bool {
        match validate(query) {
            Ok(v) => v.has_visual(),
            Err(_) => false,
        }
    }

    /// Execute SQL-only query and return JSON with columns/rows
    pub fn execute_sql(&self, query: &str) -> Result<String, JsValue> {
        let df = {
            let reader = self.reader.borrow();
            reader
                .execute_sql(query)
                .map_err(|e| JsValue::from_str(&format!("SQL error: {:?}", e)))?
        };

        let max_rows = 100usize;
        let total_rows = df.height();
        let truncated = total_rows > max_rows;
        let df = if truncated { df.slice(0, max_rows) } else { df };

        let columns: Vec<String> = df.get_column_names();
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(df.height());

        for i in 0..df.height() {
            let mut row = Vec::with_capacity(columns.len());
            for col in df.get_columns() {
                row.push(value_to_string(col, i));
            }
            rows.push(row);
        }

        let result = json!({
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
            "truncated": truncated,
        });

        serde_json::to_string(&result).map_err(|e| JsValue::from_str(&format!("JSON error: {}", e)))
    }

    /// Register a CSV file as a table from raw bytes
    pub fn register_csv(&self, name: &str, data: &[u8]) -> Result<(), JsValue> {
        let columns_js = convert_csv_js(data)
            .map_err(|e| JsValue::from_str(&format!("CSV parse error: {:?}", e)))?;
        let df = columns_js_to_dataframe(columns_js)?;
        let reader = self.reader.borrow();
        reader
            .register(name, df, true)
            .map_err(|e| JsValue::from_str(&format!("Registration error: {:?}", e)))
    }

    /// Register a Parquet file as a table from raw bytes
    pub async fn register_parquet(&self, name: &str, data: &[u8]) -> Result<(), JsValue> {
        let columns_js = convert_parquet_js(data)
            .await
            .map_err(|e| JsValue::from_str(&format!("Parquet parse error: {:?}", e)))?;
        let df = columns_js_to_dataframe(columns_js)?;
        let reader = self.reader.borrow();
        reader
            .register(name, df, true)
            .map_err(|e| JsValue::from_str(&format!("Registration error: {:?}", e)))
    }

    /// Register all known builtin datasets (e.g. ggsql:penguins)
    pub async fn register_builtin_datasets(&self) -> Result<(), JsValue> {
        for &name in ggsql::reader::data::KNOWN_DATASETS {
            if let Some(bytes) = ggsql::reader::data::builtin_parquet_bytes(name) {
                let table_name = ggsql::naming::builtin_data_table(name);
                let columns_js = convert_parquet_js(bytes).await.map_err(|e| {
                    JsValue::from_str(&format!("Parquet error for '{}': {:?}", name, e))
                })?;
                let df = columns_js_to_dataframe(columns_js)?;
                let reader = self.reader.borrow();
                reader.register(&table_name, df, true).map_err(|e| {
                    JsValue::from_str(&format!("Registration error for '{}': {:?}", name, e))
                })?;
            }
        }
        Ok(())
    }

    /// Load a previously installed SQLite extension.
    ///
    /// `entry_point` is the C init function name. If omitted, SQLite
    /// derives it from the extension name.
    pub fn load_extension(&self, name: &str, entry_point: Option<String>) -> Result<(), JsValue> {
        let reader = self.reader.borrow();
        let conn = reader.connection();
        unsafe {
            conn.load_extension_enable()
                .map_err(|e| JsValue::from_str(&format!("Enable load_extension error: {:?}", e)))?;
            conn.load_extension(name, entry_point.as_deref())
                .map_err(|e| JsValue::from_str(&format!("Load extension error: {:?}", e)))?;
        }
        Ok(())
    }

    /// Unregister a table
    pub fn unregister(&self, name: &str) -> Result<(), JsValue> {
        let reader = self.reader.borrow();
        reader
            .unregister(name)
            .map_err(|e| JsValue::from_str(&format!("Unregister error: {:?}", e)))
    }

    /// List all registered tables
    pub fn list_tables(&self) -> JsValue {
        let reader = self.reader.borrow();
        let tables = reader.list_tables(false);

        let array = js_sys::Array::new();
        for table in tables {
            array.push(&JsValue::from_str(&table));
        }

        // Builtin datasets (translate internal name → ggsql:name)
        for table in reader.list_tables(true) {
            if let Some(name) = table
                .strip_prefix(DATA_PREFIX)
                .and_then(|s| s.strip_suffix("__"))
            {
                array.push(&JsValue::from_str(&format!("ggsql:{}", name)));
            }
        }

        array.into()
    }
}

// ============================================================================
// Drawing
// ============================================================================

/// A resolved plot, ready to be drawn at whatever size the page has.
///
/// Held across redraws so a resize costs a layout pass and not a database
/// query — see [`GgsqlContext::execute`].
#[wasm_bindgen]
pub struct GgsqlPlot {
    spec: Spec,
}

#[wasm_bindgen]
impl GgsqlPlot {
    /// Draw the plot as SVG at the given size in CSS pixels.
    ///
    /// The layout is re-solved at this size rather than scaled to it, so a wider
    /// box gets more tick labels rather than stretched ones — which is why a
    /// resize calls this again instead of setting a `viewBox`.
    ///
    /// `id_prefix` namespaces every generated element id: inline SVGs share the
    /// page's id space, so two plots on one page collide without it.
    ///
    /// The canvas is opaque, matching every other ggsql writer. A transparent
    /// one would let the page show through wherever the theme's plot background
    /// does not reach — the slack a panel with a locked aspect (a polar coord,
    /// a map) leaves beside itself, which the legend sits in.
    #[wasm_bindgen(js_name = toSvg)]
    pub fn to_svg(&self, width: u32, height: u32, id_prefix: &str) -> Result<SvgRender, JsValue> {
        // 96 dpi: the caller measured its box in CSS pixels, and an SVG scales
        // for a retina screen by itself.
        let writer = SvgWriter::new(width.max(1), height.max(1), 96.0).id_prefix(id_prefix);
        let (svg, warnings) = writer
            .render_reporting(&self.spec)
            .map_err(|e| JsValue::from_str(&format!("Render error: {:?}", e)))?;
        Ok(SvgRender { svg, warnings })
    }
}

/// One drawn plot, plus whatever the format could not express.
#[wasm_bindgen]
pub struct SvgRender {
    svg: String,
    warnings: Vec<String>,
}

#[wasm_bindgen]
impl SvgRender {
    /// The SVG markup.
    #[wasm_bindgen(getter)]
    pub fn svg(&self) -> String {
        self.svg.clone()
    }

    /// What the renderer had to degrade or drop, if anything.
    #[wasm_bindgen(getter)]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

// ============================================================================
// Fonts
// ============================================================================

/// Register every font face in `bytes`, returning the family names they landed
/// under.
///
/// A page must call this before drawing anything: a browser enumerates no system
/// fonts, so the shaper starts empty and a plot comes out with no text and no
/// warning — and with wrong margins too, since text sets the layout.
///
/// Takes sfnt bytes (TTF, OTF, TTC, OTC); a WOFF or WOFF2 file has to be decoded
/// first. The returned names are what [`set_generic_family`] takes — registering
/// a face does not on its own make `sans-serif` mean it.
#[wasm_bindgen(js_name = registerFont)]
pub fn register_font(bytes: Vec<u8>) -> Result<Vec<String>, JsValue> {
    ggsql::fonts::register_font(bytes).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Whether any font family is available to shape with.
///
/// `false` means the next plot drawn will have no text in it.
#[wasm_bindgen(js_name = hasFonts)]
pub fn has_fonts() -> bool {
    !ggsql::fonts::registered_font_families().is_empty()
}

/// Point a generic family — `sans-serif`, `serif`, `monospace`, … — at concrete
/// families, in preference order.
#[wasm_bindgen(js_name = setGenericFamily)]
pub fn set_generic_family(kind: &str, families: Vec<String>) -> Result<(), JsValue> {
    ggsql::fonts::set_generic_family(kind, &families).map_err(|e| JsValue::from_str(&e.to_string()))
}
