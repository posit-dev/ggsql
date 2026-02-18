use tree_sitter::{Parser, Query, StreamingIterator};

use crate::{naming, GgsqlError};

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
// =============================================================================

#[cfg(feature = "builtin-data")]
static PENGUINS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/penguins.parquet"
));

#[cfg(feature = "builtin-data")]
static AIRQUALITY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/airquality.parquet"
));

/// Get the embedded parquet bytes for a known builtin dataset.
#[cfg(feature = "builtin-data")]
fn builtin_parquet_bytes(name: &str) -> Option<&'static [u8]> {
    match name {
        "penguins" => Some(PENGUINS),
        "airquality" => Some(AIRQUALITY),
        _ => None,
    }
}

// =============================================================================
// DuckDB builtin data registration (requires duckdb + builtin-data)
// =============================================================================

/// Register any builtin datasets referenced in the SQL with a DuckDB connection.
///
/// Finds `ggsql:X` patterns in the SQL, writes the embedded parquet data to
/// a temp file, and creates a table named `__ggsql_data_X__` in DuckDB.
#[cfg(all(feature = "duckdb", feature = "builtin-data"))]
pub fn register_builtin_datasets_duckdb(
    sql: &str,
    conn: &duckdb::Connection,
) -> Result<(), GgsqlError> {
    use std::{env, fs};

    let dataset_names = extract_builtin_dataset_names(sql)?;
    for name in dataset_names {
        let Some(parquet_bytes) = builtin_parquet_bytes(&name) else {
            continue;
        };

        let table_name = naming::builtin_data_table(&name);

        // Write parquet to temp file for DuckDB's read_parquet
        let mut tmp_path = env::temp_dir();
        tmp_path.push(format!("{}.parquet", name));
        if !tmp_path.exists() {
            fs::write(&tmp_path, parquet_bytes).expect("Failed to write dataset");
        }

        let create_sql = format!(
            "CREATE TABLE IF NOT EXISTS \"{}\" AS SELECT * FROM read_parquet('{}')",
            table_name,
            tmp_path.display()
        );

        conn.execute(&create_sql, duckdb::params![]).map_err(|e| {
            GgsqlError::ReaderError(format!(
                "Failed to register builtin dataset '{}': {}",
                name, e
            ))
        })?;
    }
    Ok(())
}

// =============================================================================
// Polars-based builtin data loading
// =============================================================================

#[cfg(feature = "builtin-data")]
pub fn load_builtin_dataframe(name: &str) -> Result<crate::DataFrame, GgsqlError> {
    use polars::prelude::*;
    use std::io::Cursor;

    let parquet_bytes = match name {
        "penguins" => PENGUINS,
        "airquality" => AIRQUALITY,
        _ => {
            return Err(GgsqlError::ReaderError(format!(
                "Unknown builtin dataset: '{}'",
                name
            )))
        }
    };

    let cursor = Cursor::new(parquet_bytes);
    ParquetReader::new(cursor).finish().map_err(|e| {
        GgsqlError::ReaderError(format!("Failed to load builtin dataset '{}': {}", name, e))
    })
}

/// Known builtin dataset names in the ggsql namespace
const KNOWN_DATASETS: &[&str] = &["penguins", "airquality"];

/// Check if a dataset name is a known builtin
pub fn is_known_builtin(name: &str) -> bool {
    KNOWN_DATASETS.contains(&name)
}

// =============================================================================
// SQL namespace rewriting (always available, including WASM)
// =============================================================================

/// Extract builtin dataset names from SQL containing namespaced identifiers.
///
/// Finds `ggsql:X` patterns via tree-sitter and returns the dataset names
/// (without the `ggsql:` prefix), deduplicated.
pub fn extract_builtin_dataset_names(sql: &str) -> Result<Vec<String>, GgsqlError> {
    let token_def = r#"(namespaced_identifier) @select"#;
    let mut tokens = tokens_from_tree(sql, token_def, "select")?;

    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    tokens.sort_unstable();
    tokens.dedup();

    let datasets: Vec<String> = tokens
        .iter()
        .filter_map(|token| token.strip_prefix("ggsql:").map(|s| s.to_string()))
        .collect();

    Ok(datasets)
}

/// Rewrite SQL to replace namespaced identifiers with internal table names.
///
/// e.g., `SELECT * FROM ggsql:penguins` -> `SELECT * FROM __ggsql_data_penguins__`
///
/// Uses tree-sitter to find the exact byte positions of namespaced identifiers,
/// then replaces them in reverse order to preserve offsets.
pub fn rewrite_namespaced_sql(sql: &str) -> Result<String, GgsqlError> {
    let token_def = r#"(namespaced_identifier) @select"#;

    // Parse to get byte positions
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_ggsql::language())
        .map_err(|e| GgsqlError::ParseError(format!("Failed to initialise parser: {}", e)))?;

    let tree = parser
        .parse(sql, None)
        .ok_or_else(|| GgsqlError::ParseError(format!("Failed to parse query: {}", sql)))?;

    let query = Query::new(&tree.language(), token_def)
        .map_err(|e| GgsqlError::ParseError(format!("Failed to initialise tree_query: {}", e)))?;

    let index = query
        .capture_index_for_name("select")
        .ok_or_else(|| GgsqlError::ParseError("Failed to capture index".to_string()))?;

    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), sql.as_bytes());

    // Collect (start_byte, end_byte, replacement) tuples
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    while let Some(matching) = matches.next() {
        for item in matching.captures {
            if item.index != index {
                continue;
            }
            let node = item.node;
            let full_text = &sql[node.start_byte()..node.end_byte()];
            if let Some(name) = full_text.strip_prefix("ggsql:") {
                replacements.push((
                    node.start_byte(),
                    node.end_byte(),
                    naming::builtin_data_table(name),
                ));
            }
        }
    }

    if replacements.is_empty() {
        return Ok(sql.to_string());
    }

    // Apply replacements in reverse byte order to preserve earlier offsets
    let mut result = sql.to_string();
    replacements.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }

    Ok(result)
}

// =============================================================================
// Shared tree-sitter helpers
// =============================================================================

fn tokens_from_tree(
    sql_query: &str,
    tree_query: &str,
    name: &str,
) -> Result<Vec<String>, GgsqlError> {
    // Setup parser
    let mut parser = Parser::new();
    if let Err(e) = parser.set_language(&tree_sitter_ggsql::language()) {
        return Err(GgsqlError::ParseError(format!(
            "Failed to initialise parser: {}",
            e
        )));
    }

    // Digest SQL to tree
    let tree = parser.parse(sql_query, None);
    if tree.is_none() {
        return Err(GgsqlError::ParseError(format!(
            "Failed to parse query: {}",
            sql_query
        )));
    }
    let tree = tree.unwrap();

    // Setup query for tree
    let query = Query::new(&tree.language(), tree_query);
    if let Err(e) = query {
        return Err(GgsqlError::ParseError(format!(
            "Failed to initialise `tree_query`: {}",
            e
        )));
    }
    let query = query.unwrap();

    // Find `name` in `tree_query`
    let index = query.capture_index_for_name(name);
    if index.is_none() {
        return Err(GgsqlError::ParseError(
            "Failed to capture index for `tree_query`".to_string(),
        ));
    }
    let index = index.unwrap();

    // Find matches of `tree_query` in the parsed tree
    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), sql_query.as_bytes());

    // Collect results
    let mut result: Vec<String> = Vec::new();
    while let Some(matching) = matches.next() {
        for item in matching.captures {
            if item.index != index {
                // We have a match with a different @keyword than the one defined by `name`.
                continue;
            }
            let node = item.node;
            let token = &sql_query[node.start_byte()..node.end_byte()];
            result.push(token.to_string());
        }
    }
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_builtin_dataset_names_single() {
        let sql = "SELECT * FROM ggsql:penguins VISUALISE DRAW point MAPPING x AS x";
        let names = extract_builtin_dataset_names(sql).unwrap();
        assert_eq!(names, vec!["penguins"]);
    }

    #[test]
    fn test_extract_builtin_dataset_names_multiple() {
        let sql =
            "SELECT * FROM ggsql:penguins, ggsql:airquality VISUALISE DRAW point MAPPING x AS x";
        let names = extract_builtin_dataset_names(sql).unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"airquality".to_string()));
        assert!(names.contains(&"penguins".to_string()));
    }

    #[test]
    fn test_extract_builtin_dataset_names_dedup() {
        let sql = "SELECT * FROM ggsql:penguins p1, ggsql:penguins p2 VISUALISE DRAW point MAPPING x AS x";
        let names = extract_builtin_dataset_names(sql).unwrap();
        assert_eq!(names, vec!["penguins"]);
    }

    #[test]
    fn test_extract_builtin_dataset_names_none() {
        let sql = "SELECT * FROM regular_table VISUALISE DRAW point MAPPING x AS x";
        let names = extract_builtin_dataset_names(sql).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn test_rewrite_namespaced_sql_simple() {
        let sql = "SELECT * FROM ggsql:penguins";
        let rewritten = rewrite_namespaced_sql(sql).unwrap();
        assert_eq!(rewritten, "SELECT * FROM __ggsql_data_penguins__");
    }

    #[test]
    fn test_rewrite_namespaced_sql_multiple() {
        let sql = "SELECT * FROM ggsql:penguins p, ggsql:airquality a WHERE p.id = a.id";
        let rewritten = rewrite_namespaced_sql(sql).unwrap();
        assert_eq!(
            rewritten,
            "SELECT * FROM __ggsql_data_penguins__ p, __ggsql_data_airquality__ a WHERE p.id = a.id"
        );
    }

    #[test]
    fn test_rewrite_namespaced_sql_no_change() {
        let sql = "SELECT * FROM regular_table WHERE x > 5";
        let rewritten = rewrite_namespaced_sql(sql).unwrap();
        assert_eq!(rewritten, sql);
    }

    #[test]
    fn test_rewrite_namespaced_sql_with_visualise() {
        let sql = "SELECT * FROM ggsql:penguins VISUALISE DRAW point MAPPING bill_len AS x, bill_dep AS y";
        let rewritten = rewrite_namespaced_sql(sql).unwrap();
        assert!(rewritten.starts_with("SELECT * FROM __ggsql_data_penguins__"));
        assert!(!rewritten.contains("ggsql:"));
    }
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
        assert!(dataframe.column("__ggsql_aes_x__").is_ok());
        assert!(dataframe.column("__ggsql_aes_y__").is_ok());

        let query = "VISUALISE FROM ggsql:airquality DRAW point MAPPING Temp AS x, Ozone AS y";
        let result = crate::execute::prepare_data_with_reader(query, &reader).unwrap();
        let dataframe = result.data.get(&naming::layer_key(0)).unwrap();
        assert!(dataframe.column("__ggsql_aes_x__").is_ok());
        assert!(dataframe.column("__ggsql_aes_y__").is_ok());
    }
}

#[cfg(feature = "builtin-data")]
#[cfg(test)]
mod builtin_data_tests {
    use super::*;

    #[test]
    fn test_load_builtin_parquet_penguins() {
        let df = load_builtin_dataframe("penguins").unwrap();
        assert!(df.height() > 0);
        assert!(df.width() > 0);
    }

    #[test]
    fn test_load_builtin_parquet_airquality() {
        let df = load_builtin_dataframe("airquality").unwrap();
        assert!(df.height() > 0);
        assert!(df.width() > 0);
    }

    #[test]
    fn test_load_builtin_parquet_unknown() {
        let result = load_builtin_dataframe("nonexistent");
        assert!(result.is_err());
    }
}
