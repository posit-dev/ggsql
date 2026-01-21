use std::{env, fs};
use tree_sitter::{Parser, Query, StreamingIterator};

use crate::GgsqlError;

// To add new built-in datasets follow these steps:
//
// 1. Add a parquet file of your dataset to the /data/ folder
// 2. Include the bytes of that parquet file in the binary, like is done
//    beneath this block.
// 3. Make a `prep_{dataset}_query()` convenience function.
// 4. In the `init_builtin_data()` function below, include a `match`-arm for your
//    dataset using the "ggsql:<dataset_name>" pattern.

static PENGUINS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/penguins.parquet"
));

static AIRQUALITY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/airquality.parquet"
));

pub fn prep_penguins_query() -> String {
    prep_builtin_dataset_query("penguins", PENGUINS)
}

pub fn prep_airquality_query() -> String {
    prep_builtin_dataset_query("airquality", AIRQUALITY)
}

fn prep_builtin_dataset_query(name: &str, data: &[u8]) -> String {
    let mut tmp_path = env::temp_dir();
    let mut filename = name.to_string();
    filename.push_str(".parquet");
    tmp_path.push(filename);
    if !tmp_path.exists() {
        fs::write(&tmp_path, data).expect("Failed to write dataset");
    }
    format!(
        "CREATE TABLE IF NOT EXISTS {} AS SELECT * FROM read_parquet('{}')",
        name,
        tmp_path.display()
    )
}

pub fn init_builtin_data(sql: &str) -> Result<Vec<String>, GgsqlError> {
    // This definition pulls out namespaced identifiers (e.g., ggsql:penguins) by
    // @select'ing the string/identifier token.
    let token_def = r#"(namespaced_identifier) @select"#;
    let mut tokens = tokens_from_tree(sql, token_def, "select")?;
    let mut result = Vec::new();
    if tokens.is_empty() {
        return Ok(result);
    }

    // Deduplicate tokens
    tokens.sort_unstable();
    tokens.dedup();

    for dataset in tokens {
        // Only process ggsql namespace datasets
        let materialize_query = match dataset.as_str() {
            "ggsql:penguins" => Some(prep_penguins_query()),
            "ggsql:airquality" => Some(prep_airquality_query()),
            _ => None, // Unknown namespace - ignored
        };
        if let Some(query) = materialize_query {
            result.push(query);
        }
    }
    Ok(result)
}

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

#[cfg(feature = "duckdb")]
#[test]
fn test_builtin_data_is_available() {
    let reader = crate::reader::DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    // We need the VISUALISE here so `prepare_data` doesn't get tripped up
    let query = "SELECT * FROM ggsql:penguins VISUALISE";
    let result = crate::execute::prepare_data(query, &reader).unwrap();
    let dataframe = result.data.get("__global__").unwrap();
    let colnames = dataframe.get_column_names();

    assert_eq!(
        colnames,
        &[
            "species",
            "island",
            "bill_len",
            "bill_dep",
            "flipper_len",
            "body_mass",
            "sex",
            "year"
        ]
    );

    let query = "VISUALISE * FROM ggsql:airquality";
    let result = crate::execute::prepare_data(query, &reader).unwrap();
    let dataframe = result.data.get("__global__").unwrap();
    let colnames = dataframe.get_column_names();

    assert_eq!(
        colnames,
        &["Ozone", "Solar.R", "Wind", "Temp", "Month", "Day", "Date"]
    );
}
