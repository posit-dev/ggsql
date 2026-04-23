use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray,
    TimestampMillisecondArray,
};
use ggsql::array_util::value_to_string;
use ggsql::naming::DATA_PREFIX;
use ggsql::reader::sqlite::SqliteReader;
use ggsql::reader::Reader;
use ggsql::validate::validate;
use ggsql::writer::{VegaLiteWriter, Writer};
use ggsql::DataFrame;
use serde_json::json;
use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::prelude::*;

// ============================================================================
// JS bridge declarations — CSV and Parquet parsing only
// ============================================================================

#[wasm_bindgen(module = "/library/dist/lib.js")]
extern "C" {
    #[wasm_bindgen(catch)]
    async fn convert_parquet(data: &[u8]) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(catch)]
    fn convert_csv(data: &[u8]) -> Result<JsValue, JsValue>;
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
    writer: VegaLiteWriter,
}

#[wasm_bindgen]
impl GgsqlContext {
    /// Create a new ggsql context
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<GgsqlContext, JsValue> {
        ensure_vfs_initialized();

        let reader = SqliteReader::new()
            .map_err(|e| JsValue::from_str(&format!("Failed to create SQLite reader: {:?}", e)))?;
        let writer = VegaLiteWriter::new();
        Ok(GgsqlContext {
            reader: RefCell::new(reader),
            writer,
        })
    }

    /// Execute a ggsql query and return Vega-Lite JSON
    pub fn execute(&self, query: &str) -> Result<String, JsValue> {
        let spec = {
            let reader = self.reader.borrow();
            reader
                .execute(query)
                .map_err(|e| JsValue::from_str(&format!("Execute error: {:?}", e)))?
        };

        let result = self
            .writer
            .render(&spec)
            .map_err(|e| JsValue::from_str(&format!("Render error: {:?}", e)))?;

        Ok(result)
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
        let columns_js = convert_csv(data)
            .map_err(|e| JsValue::from_str(&format!("CSV parse error: {:?}", e)))?;
        let df = columns_js_to_dataframe(columns_js)?;
        let reader = self.reader.borrow();
        reader
            .register(name, df, true)
            .map_err(|e| JsValue::from_str(&format!("Registration error: {:?}", e)))
    }

    /// Register a Parquet file as a table from raw bytes
    pub async fn register_parquet(&self, name: &str, data: &[u8]) -> Result<(), JsValue> {
        let columns_js = convert_parquet(data)
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
                let columns_js = convert_parquet(bytes).await.map_err(|e| {
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
