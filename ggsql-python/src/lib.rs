// Allow useless_conversion due to false positive from pyo3 macro expansion
// See: https://github.com/PyO3/pyo3/issues/4327
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use ggsql::naming::GLOBAL_DATA_KEY;
use ggsql::parser::parse_query;
use ggsql::writer::{VegaLiteWriter, Writer};
use ggsql::AestheticValue;

use polars::prelude::{DataFrame, IpcReader, SerReader};

#[pyfunction]
fn split_query(query: &str) -> PyResult<(String, String)> {
    ggsql::parser::split_query(query)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (ipc_bytes, viz, *, writer = "vegalite"))]
fn render(ipc_bytes: &Bound<'_, PyBytes>, viz: &str, writer: &str) -> PyResult<String> {
    // Read DataFrame from IPC bytes
    let bytes = ipc_bytes.as_bytes();
    let cursor = Cursor::new(bytes);
    let df: DataFrame = IpcReader::new(cursor).finish().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to read IPC data: {}", e))
    })?;

    // Parse the visualization spec
    // The viz string should be a complete VISUALISE statement
    let specs = parse_query(viz)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut spec = specs.into_iter().next().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("No visualization spec found")
    })?;

    // Get column names for resolving global mappings
    let column_names: HashSet<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();

    // Merge global mappings into layers and handle wildcards
    for layer in &mut spec.layers {
        let supported_aesthetics = layer.geom.aesthetics().supported;

        // 1. Merge explicit global aesthetics into layer (layer takes precedence)
        for (aesthetic, value) in &spec.global_mappings.aesthetics {
            if supported_aesthetics.contains(&aesthetic.as_str()) {
                layer
                    .mappings
                    .aesthetics
                    .entry(aesthetic.clone())
                    .or_insert_with(|| value.clone());
            }
        }

        // 2. Handle wildcard expansion: map columns to aesthetics with matching names
        let has_wildcard = layer.mappings.wildcard || spec.global_mappings.wildcard;
        if has_wildcard {
            for &aes in supported_aesthetics {
                // Only create mapping if column exists in the dataframe
                if column_names.contains(aes) {
                    layer
                        .mappings
                        .aesthetics
                        .entry(aes.to_string())
                        .or_insert_with(|| AestheticValue::standard_column(aes));
                }
            }
        }
    }

    // Compute aesthetic labels from column names
    spec.compute_aesthetic_labels();

    // Create data map with the DataFrame as global data
    let mut data_map: HashMap<String, DataFrame> = HashMap::new();
    data_map.insert(GLOBAL_DATA_KEY.to_string(), df);

    // Write using the specified writer
    match writer {
        "vegalite" => {
            let w = VegaLiteWriter::new();
            w.write(&spec, &data_map)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown writer: {}",
            writer
        ))),
    }
}

#[pymodule]
fn _ggsql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(split_query, m)?)?;
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
