use crate::GgsqlError;

// =============================================================================
// Embedded dataset bytes
// =============================================================================
// To add new built-in datasets follow these steps:
//
// 1. Add a parquet file of your dataset to the /data/ folder
// 2. Include the bytes of that parquet file in the binary, like is done
//    beneath this block.
// 3. Add a match arm in `builtin_parquet_bytes()` for your dataset.
// 4. Add the dataset name to `KNOWN_DATASETS`.
//
// Parquet compatibility
// ---------------------
// The file must be readable by arrow-rs without `skip_arrow_metadata`.
// The test `all_builtin_parquets_load` enforces this in CI.
//
// Known-compatible writers:
//   - Python `pyarrow`            (`pq.write_table(...)`)
//   - Rust `arrow-rs` + `parquet` (`ArrowWriter`)
//   - DuckDB                      (`COPY ... TO 'file.parquet'`)
//
// Known-incompatible writers:
//   - R `nanoparquet` — writes ARROW:schema with a different flatbuffers
//     alignment that arrow-rs's strict reader rejects.
//
// If you receive a file from an incompatible source, round-trip it with a
// compatible writer. Example with pyarrow:
//   import pyarrow.parquet as pq
//   pq.write_table(pq.read_table('input.parquet'), 'output.parquet',
//                  compression='snappy')
// =============================================================================

#[cfg(feature = "builtin-data")]
static PENGUINS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/penguins.parquet"
));

#[cfg(feature = "builtin-data")]
static AIRQUALITY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/airquality.parquet"
));

#[cfg(feature = "builtin-data")]
static WORLD: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/world.parquet"));

/// Get the embedded parquet bytes for a known builtin dataset.
#[cfg(feature = "builtin-data")]
pub fn builtin_parquet_bytes(name: &str) -> Option<&'static [u8]> {
    match name {
        "penguins" => Some(PENGUINS),
        "airquality" => Some(AIRQUALITY),
        "world" => Some(WORLD),
        _ => None,
    }
}

// =============================================================================
// Arrow-based builtin data loading
// =============================================================================

#[cfg(all(feature = "builtin-data", feature = "parquet"))]
pub fn load_builtin_dataframe(name: &str) -> Result<crate::DataFrame, GgsqlError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let parquet_bytes = match name {
        "penguins" => PENGUINS,
        "airquality" => AIRQUALITY,
        "world" => WORLD,
        _ => {
            return Err(GgsqlError::ReaderError(format!(
                "Unknown builtin dataset: '{}'",
                name
            )))
        }
    };

    let bytes = bytes::Bytes::from_static(parquet_bytes);
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to read builtin dataset '{}': {}", name, e))
        })?
        .build()
        .map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to build reader for '{}': {}", name, e))
        })?;

    let batches: Vec<_> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to load builtin dataset '{}': {}", name, e))
        })?;

    if batches.is_empty() {
        return Ok(crate::DataFrame::empty());
    }

    let rb = if batches.len() == 1 {
        batches.into_iter().next().unwrap()
    } else {
        arrow::compute::concat_batches(&batches[0].schema(), &batches).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to concat batches for '{}': {}", name, e))
        })?
    };

    Ok(crate::DataFrame::from_record_batch(rb))
}

/// Known builtin dataset names in the ggsql namespace
pub const KNOWN_DATASETS: &[&str] = &["penguins", "airquality", "world"];

/// Check if a dataset name is a known builtin
pub fn is_known_builtin(name: &str) -> bool {
    KNOWN_DATASETS.contains(&name)
}

#[cfg(all(feature = "duckdb", feature = "builtin-data"))]
#[cfg(test)]
mod duckdb_tests {
    #[test]
    fn test_builtin_data_is_available() {
        use crate::naming;

        let reader =
            crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query =
            "SELECT * FROM ggsql:penguins VISUALISE DRAW point MAPPING bill_len AS x, bill_dep AS y";
        let result = crate::execute::prepare_data_with_reader(query, &reader).unwrap();
        let dataframe = result.data.get(&naming::layer_key(0)).unwrap();
        // Aesthetics are transformed to internal names (x -> pos1, y -> pos2)
        assert!(dataframe.column("__ggsql_aes_pos1__").is_ok());
        assert!(dataframe.column("__ggsql_aes_pos2__").is_ok());

        let query = "VISUALISE FROM ggsql:airquality DRAW point MAPPING Temp AS x, Ozone AS y";
        let result = crate::execute::prepare_data_with_reader(query, &reader).unwrap();
        let dataframe = result.data.get(&naming::layer_key(0)).unwrap();
        assert!(dataframe.column("__ggsql_aes_pos1__").is_ok());
        assert!(dataframe.column("__ggsql_aes_pos2__").is_ok());
    }

    #[test]
    fn test_ribbon_transposed_orientation() {
        use crate::naming;

        let reader =
            crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Ribbon with y as domain axis and xmin/xmax as value range (transposed)
        let query =
            "VISUALISE FROM ggsql:airquality DRAW ribbon MAPPING Day AS y, Temp AS xmax, 0.0 AS xmin";
        let result = crate::execute::prepare_data_with_reader(query, &reader);

        // Debug: print the error if any
        if let Err(ref e) = result {
            eprintln!("Error: {:?}", e);
        }

        let result = result.unwrap();

        // Debug: print orientation and scales
        let layer = &result.specs[0].layers[0];
        let orientation = layer.parameters.get("orientation");
        eprintln!("Layer orientation: {:?}", orientation);
        eprintln!(
            "Scales: {:?}",
            result.specs[0]
                .scales
                .iter()
                .map(|s| (&s.aesthetic, &s.scale_type))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "Layer mappings: {:?}",
            layer.mappings.aesthetics.keys().collect::<Vec<_>>()
        );

        // Check orientation was detected correctly
        assert_eq!(
            orientation.and_then(|v| v.as_str()),
            Some("transposed"),
            "Should detect Transposed orientation"
        );

        let dataframe = result.data.get(&naming::layer_key(0)).unwrap();

        // The flip-back restores user's original axis assignment:
        // After flip-back:
        // - pos2 = y (user's domain axis = Date/Day)
        // - pos1min = xmin (user's value range min = 0.0)
        // - pos1max = xmax (user's value range max = Temp)
        // The orientation flag tells the writer how to map to x/y.
        let cols: Vec<_> = dataframe.get_column_names().into_iter().collect();
        eprintln!("Columns: {:?}", cols);

        assert!(
            dataframe.column("__ggsql_aes_pos2__").is_ok(),
            "Should have pos2 (domain axis), got columns: {:?}",
            cols
        );
        assert!(
            dataframe.column("__ggsql_aes_pos1min__").is_ok(),
            "Should have pos1min (value range min), got columns: {:?}",
            cols
        );
        assert!(
            dataframe.column("__ggsql_aes_pos1max__").is_ok(),
            "Should have pos1max (value range max), got columns: {:?}",
            cols
        );
    }

    #[test]
    fn test_ribbon_transposed_vegalite_encoding() {
        use crate::reader::Reader;
        use crate::writer::{VegaLiteWriter, Writer};

        let reader =
            crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Ribbon with y as domain axis and xmin/xmax as value range (transposed)
        let query =
            "VISUALISE FROM ggsql:airquality DRAW ribbon MAPPING Day AS y, Temp AS xmax, 0.0 AS xmin";
        let spec = reader.execute(query).unwrap();

        let writer = VegaLiteWriter::new();
        let json_str = writer.render(&spec).unwrap();
        let vl_spec: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // For transposed ribbon, the encoding should have:
        // - y: domain axis (Day)
        // - x: value range max (Temp via xmax)
        // - x2: value range min (0.0 via xmin)
        // The encoding is inside layer[0] since VegaLite uses layered structure
        let encoding = &vl_spec["layer"][0]["encoding"];
        assert!(
            encoding.get("y").is_some(),
            "Should have y encoding for domain axis"
        );
        assert!(
            encoding.get("x").is_some(),
            "Should have x encoding for value max"
        );
        assert!(
            encoding.get("x2").is_some(),
            "Should have x2 encoding for value min"
        );
        // Should NOT have ymax/ymin/xmax/xmin (these should be remapped to x/x2/y/y2)
        assert!(
            encoding.get("ymax").is_none(),
            "Should not have ymax encoding"
        );
        assert!(
            encoding.get("ymin").is_none(),
            "Should not have ymin encoding"
        );
        assert!(
            encoding.get("xmax").is_none(),
            "Should not have xmax encoding"
        );
        assert!(
            encoding.get("xmin").is_none(),
            "Should not have xmin encoding"
        );
    }
}

#[cfg(all(feature = "builtin-data", feature = "parquet"))]
#[cfg(test)]
mod builtin_data_tests {
    use super::*;

    /// Every entry in `KNOWN_DATASETS` must load cleanly via arrow-rs without
    /// the `skip_arrow_metadata` workaround. If this test fails on a newly
    /// added parquet file, the file was written by an incompatible tool
    /// (see the compatibility notes at the top of this module).
    #[test]
    fn all_builtin_parquets_load() {
        for name in KNOWN_DATASETS {
            let df = load_builtin_dataframe(name).unwrap_or_else(|e| {
                panic!(
                    "Builtin dataset '{}' failed to load — likely an incompatible \
                     parquet writer. See parquet compatibility notes in \
                     src/reader/data.rs. Underlying error: {}",
                    name, e
                )
            });
            assert!(
                df.height() > 0 && df.width() > 0,
                "Builtin dataset '{}' loaded with zero rows or columns",
                name
            );
        }
    }

    #[test]
    fn test_load_builtin_parquet_unknown() {
        let result = load_builtin_dataframe("nonexistent");
        assert!(result.is_err());
    }
}
