use ggsql::reader::{PolarsReader, Reader};
use ggsql::writer::{VegaLiteWriter, Writer};
use std::cell::RefCell;

use wasm_bindgen::prelude::*;

/// Persistent ggsql context for WASM
///
/// Create once and reuse for multiple queries to avoid memory issues.
/// Uses interior mutability to avoid wasm_bindgen's &mut self aliasing issues.
#[wasm_bindgen]
pub struct GgsqlContext {
    reader: RefCell<PolarsReader>,
    writer: VegaLiteWriter,
}

#[wasm_bindgen]
impl GgsqlContext {
    /// Create a new ggsql context
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<GgsqlContext, JsValue> {
        let reader = PolarsReader::from_connection_string("polars://memory")
            .map_err(|e| JsValue::from_str(&format!("Reader error: {:?}", e)))?;
        let writer = VegaLiteWriter::new();
        Ok(GgsqlContext {
            reader: RefCell::new(reader),
            writer,
        })
    }

    /// Execute a ggsql query and return Vega-Lite JSON
    pub fn execute(&self, query: &str) -> Result<String, JsValue> {
        // Scope the mutable borrow to avoid aliasing issues
        let spec = {
            let reader = self.reader.borrow_mut();
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

    // TODO: Register a table from binary data (e.g. CSV, Parquet)
    pub fn register(&self, _name: &str) -> Result<(), JsValue> {
        Err(JsValue::from_str("Registration not yet implemented."))
    }

    /// Unregister a table
    pub fn unregister(&self, name: &str) -> Result<(), JsValue> {
        let reader = self.reader.borrow();
        reader
            .unregister(name)
            .map_err(|e| JsValue::from_str(&format!("Unregister error: {:?}", e)))?;

        Ok(())
    }

    /// List all registered tables
    pub fn list_tables(&self) -> JsValue {
        let reader = self.reader.borrow();
        let tables = reader.list_tables(false);

        let array = js_sys::Array::new();
        for table in tables {
            array.push(&JsValue::from_str(&table));
        }
        array.into()
    }
}
