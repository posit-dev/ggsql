//! Query execution module for ggSQL
//!
//! Provides shared execution logic for building data maps from queries,
//! handling both global SQL and layer-specific data sources.

use crate::parser::ast::{AestheticValue, GlobalMapping, GlobalMappingItem, Layer, LiteralValue};
use crate::{parser, DataFrame, Facet, GgsqlError, LayerSource, Result, VizSpec};
use std::collections::{HashMap, HashSet};
use tree_sitter::{Node, Parser};

#[cfg(feature = "duckdb")]
use crate::reader::{DuckDBReader, Reader};

/// Extracted CTE (Common Table Expression) definition
#[derive(Debug, Clone)]
pub struct CteDefinition {
    /// Name of the CTE
    pub name: String,
    /// Full SQL text of the CTE body (including the SELECT statement inside)
    pub body: String,
}

/// Extract CTE definitions from SQL using tree-sitter
///
/// Parses the SQL and extracts all CTE definitions from WITH clauses.
/// Returns CTEs in declaration order (important for dependency resolution).
fn extract_ctes(sql: &str) -> Vec<CteDefinition> {
    let mut ctes = Vec::new();

    // Parse with tree-sitter
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        return ctes;
    }

    let tree = match parser.parse(sql, None) {
        Some(t) => t,
        None => return ctes,
    };

    let root = tree.root_node();

    // Walk the tree looking for WITH statements
    extract_ctes_from_node(&root, sql, &mut ctes);

    ctes
}

/// Recursively extract CTEs from a node and its children
fn extract_ctes_from_node(node: &Node, source: &str, ctes: &mut Vec<CteDefinition>) {
    // Check if this is a with_statement
    if node.kind() == "with_statement" {
        // Find all cte_definition children (in declaration order)
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "cte_definition" {
                if let Some(cte) = parse_cte_definition(&child, source) {
                    ctes.push(cte);
                }
            }
        }
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_ctes_from_node(&child, source, ctes);
    }
}

/// Parse a single CTE definition node into a CteDefinition
fn parse_cte_definition(node: &Node, source: &str) -> Option<CteDefinition> {
    let mut name: Option<String> = None;
    let mut body_start: Option<usize> = None;
    let mut body_end: Option<usize> = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                name = Some(get_node_text(&child, source).to_string());
            }
            "select_statement" => {
                // The SELECT inside the CTE
                body_start = Some(child.start_byte());
                body_end = Some(child.end_byte());
            }
            _ => {}
        }
    }

    match (name, body_start, body_end) {
        (Some(n), Some(start), Some(end)) => {
            let body = source[start..end].to_string();
            Some(CteDefinition { name: n, body })
        }
        _ => None,
    }
}

/// Get text content of a node
fn get_node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

/// Transform CTE references in SQL to use temp table names
///
/// Replaces references to CTEs (e.g., `FROM sales`, `JOIN sales`) with
/// the corresponding temp table names (e.g., `FROM __ggsql_cte_sales__`).
///
/// This handles table references after FROM and JOIN keywords, being careful
/// to only replace whole word matches (not substrings).
fn transform_cte_references(sql: &str, cte_names: &HashSet<String>) -> String {
    if cte_names.is_empty() {
        return sql.to_string();
    }

    let mut result = sql.to_string();

    for cte_name in cte_names {
        let temp_table_name = format!("__ggsql_cte_{}__", cte_name);

        // Replace table references: FROM cte_name, JOIN cte_name
        // Use word boundary matching to avoid replacing substrings
        // Pattern: (FROM|JOIN)\s+<cte_name>(\s|,|)|$)
        let patterns = [
            // FROM cte_name (case insensitive)
            (
                format!(r"(?i)(\bFROM\s+){}(\s|,|\)|$)", regex::escape(cte_name)),
                format!("${{1}}{}${{2}}", temp_table_name),
            ),
            // JOIN cte_name (case insensitive) - handles LEFT JOIN, RIGHT JOIN, etc.
            (
                format!(r"(?i)(\bJOIN\s+){}(\s|,|\)|$)", regex::escape(cte_name)),
                format!("${{1}}{}${{2}}", temp_table_name),
            ),
        ];

        for (pattern, replacement) in patterns {
            if let Ok(re) = regex::Regex::new(&pattern) {
                result = re.replace_all(&result, replacement.as_str()).to_string();
            }
        }
    }

    result
}

/// Get the temp table name for a CTE
fn get_cte_temp_table_name(cte_name: &str) -> String {
    format!("__ggsql_cte_{}__", cte_name)
}

/// Generate synthetic column name for a constant aesthetic
fn const_column_name(aesthetic: &str) -> String {
    format!("__ggsql_const_{}__", aesthetic)
}

/// Generate synthetic column name for a constant aesthetic with layer index
/// Used when injecting constants into global data so different layers can have different values
fn const_column_name_indexed(aesthetic: &str, layer_idx: usize) -> String {
    format!("__ggsql_const_{}_{}__", aesthetic, layer_idx)
}

/// Format a literal value as SQL
fn literal_to_sql(lit: &LiteralValue) -> String {
    match lit {
        LiteralValue::String(s) => format!("'{}'", s.replace('\'', "''")),
        LiteralValue::Number(n) => n.to_string(),
        LiteralValue::Boolean(b) => {
            if *b {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
    }
}

/// Extract constant aesthetics from a layer
fn extract_constants(layer: &Layer) -> Vec<(String, LiteralValue)> {
    layer
        .aesthetics
        .iter()
        .filter_map(|(aesthetic, value)| {
            if let AestheticValue::Literal(lit) = value {
                Some((aesthetic.clone(), lit.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Replace literal aesthetic values with column references to synthetic constant columns
///
/// After data has been fetched with constants injected as columns, this function
/// updates the spec so that aesthetics point to the synthetic column names instead
/// of literal values.
///
/// For layers using global data (no source, no filter), uses layer-indexed column names
/// (e.g., `__ggsql_const_color_0__`) since constants are injected into global data.
/// For other layers, uses non-indexed column names (e.g., `__ggsql_const_color__`).
fn replace_literals_with_columns(spec: &mut VizSpec) {
    for (layer_idx, layer) in spec.layers.iter_mut().enumerate() {
        for (aesthetic, value) in layer.aesthetics.iter_mut() {
            if matches!(value, AestheticValue::Literal(_)) {
                // Use layer-indexed column name for layers using global data (no source, no filter)
                // Use non-indexed name for layers with their own data (filter or explicit source)
                let col_name = if layer.source.is_none() && layer.filter.is_none() {
                    const_column_name_indexed(aesthetic, layer_idx)
                } else {
                    const_column_name(aesthetic)
                };
                *value = AestheticValue::Column(col_name);
            }
        }
    }
}

/// Materialize CTEs as temporary tables in the database
///
/// Creates a temp table for each CTE in declaration order. When a CTE
/// references an earlier CTE, the reference is transformed to use the
/// temp table name.
///
/// Returns the set of CTE names that were materialized.
fn materialize_ctes<F>(ctes: &[CteDefinition], execute_sql: &F) -> Result<HashSet<String>>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    let mut materialized = HashSet::new();

    for cte in ctes {
        // Transform the CTE body to replace references to earlier CTEs
        let transformed_body = transform_cte_references(&cte.body, &materialized);

        let temp_table_name = get_cte_temp_table_name(&cte.name);
        let create_sql = format!(
            "CREATE OR REPLACE TEMP TABLE {} AS {}",
            temp_table_name, transformed_body
        );

        execute_sql(&create_sql).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to materialize CTE '{}': {}", cte.name, e))
        })?;

        materialized.insert(cte.name.clone());
    }

    Ok(materialized)
}

/// Extract the trailing SELECT statement from a WITH clause
///
/// Given SQL like `WITH a AS (...), b AS (...) SELECT * FROM a`, extracts
/// just the `SELECT * FROM a` part. Returns None if there's no trailing SELECT.
fn extract_trailing_select(sql: &str) -> Option<String> {
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        return None;
    }

    let tree = parser.parse(sql, None)?;
    let root = tree.root_node();

    // Find sql_portion → sql_statement → with_statement → select_statement
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "sql_portion" {
            let mut sql_cursor = child.walk();
            for sql_child in child.children(&mut sql_cursor) {
                if sql_child.kind() == "sql_statement" {
                    let mut stmt_cursor = sql_child.walk();
                    for stmt_child in sql_child.children(&mut stmt_cursor) {
                        if stmt_child.kind() == "with_statement" {
                            // Find trailing select_statement in with_statement
                            let mut with_cursor = stmt_child.walk();
                            let mut seen_cte = false;
                            for with_child in stmt_child.children(&mut with_cursor) {
                                if with_child.kind() == "cte_definition" {
                                    seen_cte = true;
                                } else if with_child.kind() == "select_statement" && seen_cte {
                                    // This is the trailing SELECT
                                    return Some(get_node_text(&with_child, sql).to_string());
                                }
                            }
                        } else if stmt_child.kind() == "select_statement" {
                            // Direct SELECT (no WITH clause)
                            return Some(get_node_text(&stmt_child, sql).to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Transform global SQL for execution with temp tables
///
/// If the SQL has a WITH clause followed by SELECT, extracts just the SELECT
/// portion and transforms CTE references to temp table names.
/// For SQL without WITH clause, just transforms any CTE references.
fn transform_global_sql(sql: &str, materialized_ctes: &HashSet<String>) -> Option<String> {
    // Try to extract trailing SELECT from WITH clause
    if let Some(trailing_select) = extract_trailing_select(sql) {
        // Transform CTE references in the SELECT
        Some(transform_cte_references(
            &trailing_select,
            materialized_ctes,
        ))
    } else if has_executable_sql(sql) {
        // No WITH clause but has executable SQL - just transform references
        Some(transform_cte_references(sql, materialized_ctes))
    } else {
        // No executable SQL (just CTEs)
        None
    }
}

/// Build a layer query handling all source types
///
/// Handles:
/// - `None` source with filter, constants, or stat transform needed → queries `__ggsql_global__`
/// - `None` source without filter, constants, or stat transform → returns `None` (use global directly)
/// - `Identifier` source → checks if CTE, uses temp table or table name
/// - `FilePath` source → wraps path in single quotes
///
/// Constants are injected as synthetic columns (e.g., `'blue' AS __ggsql_const_color__`).
/// Also applies statistical transformations for geoms that need them
/// (e.g., histogram binning, bar counting).
///
/// Returns:
/// - `Ok(Some(query))` - execute this query and store result
/// - `Ok(None)` - layer uses `__global__` directly (no source, no filter, no constants, no stat transform)
/// - `Err(...)` - validation error (e.g., filter without global data)
fn build_layer_query<F>(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
    layer_idx: usize,
    facet: Option<&Facet>,
    constants: &[(String, LiteralValue)],
    execute_query: &F,
) -> Result<Option<String>>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    let filter = layer.filter.as_ref().map(|f| f.as_str());

    let table_name = match &layer.source {
        Some(LayerSource::Identifier(name)) => {
            // Check if it's a materialized CTE
            if materialized_ctes.contains(name) {
                get_cte_temp_table_name(name)
            } else {
                name.clone()
            }
        }
        Some(LayerSource::FilePath(path)) => {
            // File paths need single quotes
            format!("'{}'", path)
        }
        None => {
            // No source - validate and use global if filter or constants present
            if filter.is_some() || !constants.is_empty() {
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} has a FILTER or constants but no data source. Either provide a SQL query or use MAPPING FROM.",
                        layer_idx + 1
                    )));
                }
                "__ggsql_global__".to_string()
            } else {
                // Check if stat transform is needed
                if layer.geom.needs_stat_transform(&layer.aesthetics) {
                    if !has_global {
                        return Err(GgsqlError::ValidationError(format!(
                            "Layer {} requires data for statistical transformation but no data source.",
                            layer_idx + 1
                        )));
                    }
                    "__ggsql_global__".to_string()
                } else {
                    // No source, no filter, no constants, no stat transform - use __global__ data directly
                    return Ok(None);
                }
            }
        }
    };

    // Build base query with optional constant columns
    let mut query = if constants.is_empty() {
        format!("SELECT * FROM {}", table_name)
    } else {
        let const_cols: Vec<String> = constants
            .iter()
            .map(|(aes, lit)| format!("{} AS {}", literal_to_sql(lit), const_column_name(aes)))
            .collect();
        format!("SELECT *, {} FROM {}", const_cols.join(", "), table_name)
    };

    // Apply filter
    if let Some(f) = filter {
        query = format!("{} WHERE {}", query, f);
    }

    // Combine partition_by and facet variables for grouping
    let mut group_by = layer.partition_by.clone();
    if let Some(f) = facet {
        for var in f.get_variables() {
            if !group_by.contains(&var) {
                group_by.push(var);
            }
        }
    }

    // Apply statistical transformation (after filter, uses combined group_by)
    query = layer
        .geom
        .apply_stat_transform(&query, &layer.aesthetics, &group_by, execute_query)?;

    Ok(Some(query))
}

/// Check if SQL contains executable statements (SELECT, INSERT, UPDATE, DELETE, CREATE)
///
/// Returns false if the SQL is just CTE definitions without a trailing statement.
/// This handles cases like `WITH a AS (...), b AS (...) VISUALISE` where the WITH
/// clause has no trailing SELECT - these CTEs are still extracted for layer use
/// but shouldn't be executed as global data.
fn has_executable_sql(sql: &str) -> bool {
    // Parse with tree-sitter to check for executable statements
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_ggsql::language()).is_err() {
        // If we can't parse, assume it's executable (fail safely)
        return true;
    }

    let tree = match parser.parse(sql, None) {
        Some(t) => t,
        None => return true, // Assume executable if parse fails
    };

    let root = tree.root_node();

    // Look for sql_portion which should contain actual SQL statements
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "sql_portion" {
            // Check if sql_portion contains actual statement nodes
            let mut sql_cursor = child.walk();
            for sql_child in child.children(&mut sql_cursor) {
                if sql_child.kind() == "sql_statement" {
                    // Check if this is a WITH-only statement (no trailing SELECT)
                    let mut stmt_cursor = sql_child.walk();
                    for stmt_child in sql_child.children(&mut stmt_cursor) {
                        match stmt_child.kind() {
                            "select_statement" | "create_statement" | "insert_statement"
                            | "update_statement" | "delete_statement" => return true,
                            "with_statement" => {
                                // Check if WITH has trailing SELECT
                                if with_has_trailing_select(&stmt_child) {
                                    return true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    false
}

/// Check if a with_statement node has a trailing SELECT (after CTEs)
fn with_has_trailing_select(with_node: &Node) -> bool {
    let mut cursor = with_node.walk();
    let mut seen_cte = false;

    for child in with_node.children(&mut cursor) {
        if child.kind() == "cte_definition" {
            seen_cte = true;
        } else if child.kind() == "select_statement" && seen_cte {
            return true;
        }
    }

    false
}

/// Result of preparing data for visualization
pub struct PreparedData {
    /// Data map with global and layer-specific DataFrames
    pub data: HashMap<String, DataFrame>,
    /// Parsed and resolved visualization specifications
    pub specs: Vec<VizSpec>,
}

/// Build data map from a query using a custom query executor function
///
/// This is the most flexible variant that works with any query execution strategy,
/// including shared state readers in REST API contexts.
///
/// # Arguments
/// * `query` - The full ggSQL query string
/// * `execute_query` - A function that executes SQL and returns a DataFrame
pub fn prepare_data_with_executor<F>(query: &str, execute_query: F) -> Result<PreparedData>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    // Split query into SQL and viz portions
    let (sql_part, viz_part) = parser::split_query(query)?;

    // Parse visualization portion
    let mut specs = parser::parse_query(query)?;

    if specs.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No visualization specifications found".to_string(),
        ));
    }

    // Check if we have any visualization content
    if viz_part.trim().is_empty() {
        return Err(GgsqlError::ValidationError(
            "The visualization portion is empty".to_string(),
        ));
    }

    // Extract CTE definitions from the global SQL (in declaration order)
    let ctes = extract_ctes(&sql_part);

    // Materialize CTEs as temporary tables
    // This creates __ggsql_cte_<name>__ tables that persist for the session
    let materialized_ctes = materialize_ctes(&ctes, &execute_query)?;

    // Build data map for multi-source support
    let mut data_map: HashMap<String, DataFrame> = HashMap::new();

    // Collect constants from layers that use global data (no source, no filter)
    // These get injected into the global data table so all layers share the same data source
    // (required for faceting to work). Use layer-indexed column names to allow different
    // constant values per layer (e.g., layer 0: 'blue' AS color, layer 1: 'red' AS color)
    let first_spec = &specs[0];

    // First, extract global constants from VISUALISE clause (e.g., VISUALISE 'blue' AS color)
    // These apply to all layers that use global data
    let global_mapping_constants: Vec<(String, LiteralValue)> =
        if let GlobalMapping::Mappings(items) = &first_spec.global_mapping {
            items
                .iter()
                .filter_map(|item| {
                    if let GlobalMappingItem::Literal { value, aesthetic } = item {
                        Some((aesthetic.clone(), value.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![]
        };

    // Find layers that use global data (no source, no filter)
    let global_data_layer_indices: Vec<usize> = first_spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| layer.source.is_none() && layer.filter.is_none())
        .map(|(idx, _)| idx)
        .collect();

    // Collect all constants: layer-specific constants + global constants for each global-data layer
    let mut global_constants: Vec<(usize, String, LiteralValue)> = Vec::new();

    // Add layer-specific constants (from MAPPING clauses)
    for (layer_idx, layer) in first_spec.layers.iter().enumerate() {
        if layer.source.is_none() && layer.filter.is_none() {
            for (aes, lit) in extract_constants(layer) {
                global_constants.push((layer_idx, aes, lit));
            }
        }
    }

    // Add global mapping constants for each layer that uses global data
    // (these will be propagated to layers during resolve_global_mappings)
    for layer_idx in &global_data_layer_indices {
        for (aes, lit) in &global_mapping_constants {
            // Only add if this layer doesn't already have this aesthetic from its own MAPPING
            let layer = &first_spec.layers[*layer_idx];
            if !layer.aesthetics.contains_key(aes) {
                global_constants.push((*layer_idx, aes.clone(), lit.clone()));
            }
        }
    }

    // Execute global SQL if present
    // If there's a WITH clause, extract just the trailing SELECT and transform CTE references.
    // The global result is stored as a temp table so filtered layers can query it efficiently.
    if !sql_part.trim().is_empty() {
        if let Some(transformed_sql) = transform_global_sql(&sql_part, &materialized_ctes) {
            // Inject global constants into the query (with layer-indexed names)
            let global_query = if global_constants.is_empty() {
                transformed_sql
            } else {
                let const_cols: Vec<String> = global_constants
                    .iter()
                    .map(|(layer_idx, aes, lit)| {
                        format!(
                            "{} AS {}",
                            literal_to_sql(lit),
                            const_column_name_indexed(aes, *layer_idx)
                        )
                    })
                    .collect();
                format!(
                    "SELECT *, {} FROM ({})",
                    const_cols.join(", "),
                    transformed_sql
                )
            };

            // Create temp table for global result
            let create_global = format!(
                "CREATE OR REPLACE TEMP TABLE __ggsql_global__ AS {}",
                global_query
            );
            execute_query(&create_global)?;

            // Read back into DataFrame for data_map
            let df = execute_query("SELECT * FROM __ggsql_global__")?;
            data_map.insert("__global__".to_string(), df);
        }
    }

    // Execute layer-specific queries
    // build_layer_query() handles all cases:
    // - Layer with source (CTE, table, or file) → query that source
    // - Layer with filter but no source → query __ggsql_global__ with filter and constants
    // - Layer with no source, no filter → returns None (use global directly, constants already injected)
    let has_global = data_map.contains_key("__global__");

    for (idx, layer) in first_spec.layers.iter().enumerate() {
        // For layers using global data without filter, constants are already in global data
        // (injected with layer-indexed names). For other layers, extract constants for injection.
        let constants = if layer.source.is_none() && layer.filter.is_none() {
            vec![] // Constants already in global data
        } else {
            extract_constants(layer)
        };

        if let Some(layer_query) = build_layer_query(
            layer,
            &materialized_ctes,
            has_global,
            idx,
            first_spec.facet.as_ref(),
            &constants,
            &execute_query,
        )? {
            let df = execute_query(&layer_query).map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "Failed to fetch data for layer {}: {}",
                    idx + 1,
                    e
                ))
            })?;
            data_map.insert(format!("__layer_{}__", idx), df);
        }
        // If None returned, layer uses __global__ data directly (no entry needed)
    }

    // Validate we have some data
    if data_map.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No data sources found. Either provide a SQL query or use MAPPING FROM in layers."
                .to_string(),
        ));
    }

    // For layers without specific sources, ensure global data exists
    let has_layer_without_source = first_spec
        .layers
        .iter()
        .any(|l| l.source.is_none() && l.filter.is_none());
    if has_layer_without_source && !data_map.contains_key("__global__") {
        return Err(GgsqlError::ValidationError(
            "Some layers use global data but no SQL query was provided.".to_string(),
        ));
    }

    // Resolve global mappings using global data if available, otherwise first layer data
    let resolve_df = data_map
        .get("__global__")
        .or_else(|| data_map.values().next())
        .ok_or_else(|| GgsqlError::InternalError("No data available".to_string()))?;

    let column_names: Vec<&str> = resolve_df
        .get_column_names()
        .iter()
        .map(|s| s.as_str())
        .collect();

    for spec in &mut specs {
        spec.resolve_global_mappings(&column_names)?;
        // Replace literal aesthetic values with column references to synthetic constant columns
        replace_literals_with_columns(spec);
        // Compute aesthetic labels (uses first non-constant column, respects user-specified labels)
        spec.compute_aesthetic_labels();
    }

    Ok(PreparedData {
        data: data_map,
        specs,
    })
}

/// Build data map from a query using DuckDB reader
///
/// Convenience wrapper around `prepare_data_with_executor` for direct DuckDB reader usage.
#[cfg(feature = "duckdb")]
pub fn prepare_data(query: &str, reader: &DuckDBReader) -> Result<PreparedData> {
    prepare_data_with_executor(query, |sql| reader.execute(sql))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FilterExpression, Geom};

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_global_only() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";

        let result = prepare_data(query, &reader).unwrap();

        assert!(result.data.contains_key("__global__"));
        assert_eq!(result.specs.len(), 1);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_no_viz() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y";

        let result = prepare_data(query, &reader);
        assert!(result.is_err());
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_layer_source() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create a table first
        reader
            .connection()
            .execute(
                "CREATE TABLE test_data AS SELECT 1 as a, 2 as b",
                duckdb::params![],
            )
            .unwrap();

        let query = "VISUALISE DRAW point MAPPING a AS x, b AS y FROM test_data";

        let result = prepare_data(query, &reader).unwrap();

        assert!(result.data.contains_key("__layer_0__"));
        assert!(!result.data.contains_key("__global__"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_with_filter_on_global() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with multiple rows
        reader
            .connection()
            .execute(
                "CREATE TABLE filter_test AS SELECT * FROM (VALUES
                (1, 10, 'A'),
                (2, 20, 'B'),
                (3, 30, 'A'),
                (4, 40, 'B')
            ) AS t(id, value, category)",
                duckdb::params![],
            )
            .unwrap();

        // Query with filter on layer using global data
        let query = "SELECT * FROM filter_test VISUALISE DRAW point MAPPING id AS x, value AS y FILTER category = 'A'";

        let result = prepare_data(query, &reader).unwrap();

        // Should have global data (unfiltered) and layer 0 data (filtered)
        assert!(result.data.contains_key("__global__"));
        assert!(result.data.contains_key("__layer_0__"));

        // Global should have all 4 rows
        let global_df = result.data.get("__global__").unwrap();
        assert_eq!(global_df.height(), 4);

        // Layer 0 should have only 2 rows (filtered to category = 'A')
        let layer_df = result.data.get("__layer_0__").unwrap();
        assert_eq!(layer_df.height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_with_filter_on_layer_source() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE layer_filter_test AS SELECT * FROM (VALUES
                (1, 100),
                (2, 200),
                (3, 300),
                (4, 400)
            ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Query with layer-specific source and filter
        let query =
            "VISUALISE DRAW point MAPPING x AS x, y AS y FROM layer_filter_test FILTER y > 200";

        let result = prepare_data(query, &reader).unwrap();

        // Should only have layer 0 data (no global)
        assert!(!result.data.contains_key("__global__"));
        assert!(result.data.contains_key("__layer_0__"));

        // Layer 0 should have only 2 rows (y > 200)
        let layer_df = result.data.get("__layer_0__").unwrap();
        assert_eq!(layer_df.height(), 2);
    }

    // ========================================
    // CTE Extraction Tests
    // ========================================

    #[test]
    fn test_extract_ctes_single() {
        let sql = "WITH sales AS (SELECT * FROM raw_sales) SELECT * FROM sales";
        let ctes = extract_ctes(sql);

        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].name, "sales");
        assert!(ctes[0].body.contains("SELECT * FROM raw_sales"));
    }

    #[test]
    fn test_extract_ctes_multiple() {
        let sql = "WITH
            sales AS (SELECT * FROM raw_sales),
            targets AS (SELECT * FROM goals)
        SELECT * FROM sales";
        let ctes = extract_ctes(sql);

        assert_eq!(ctes.len(), 2);
        // Verify order is preserved
        assert_eq!(ctes[0].name, "sales");
        assert_eq!(ctes[1].name, "targets");
    }

    #[test]
    fn test_extract_ctes_none() {
        let sql = "SELECT * FROM sales WHERE year = 2024";
        let ctes = extract_ctes(sql);

        assert!(ctes.is_empty());
    }

    // ========================================
    // CTE Reference Transformation Tests
    // ========================================

    #[test]
    fn test_transform_cte_references_single() {
        let sql = "SELECT * FROM sales WHERE year = 2024";
        let mut cte_names = HashSet::new();
        cte_names.insert("sales".to_string());

        let result = transform_cte_references(sql, &cte_names);

        assert_eq!(
            result,
            "SELECT * FROM __ggsql_cte_sales__ WHERE year = 2024"
        );
    }

    #[test]
    fn test_transform_cte_references_multiple() {
        let sql = "SELECT * FROM sales JOIN targets ON sales.date = targets.date";
        let mut cte_names = HashSet::new();
        cte_names.insert("sales".to_string());
        cte_names.insert("targets".to_string());

        let result = transform_cte_references(sql, &cte_names);

        assert!(result.contains("FROM __ggsql_cte_sales__"));
        assert!(result.contains("JOIN __ggsql_cte_targets__"));
    }

    #[test]
    fn test_transform_cte_references_no_match() {
        let sql = "SELECT * FROM other_table";
        let mut cte_names = HashSet::new();
        cte_names.insert("sales".to_string());

        let result = transform_cte_references(sql, &cte_names);

        assert_eq!(result, "SELECT * FROM other_table");
    }

    #[test]
    fn test_transform_cte_references_empty() {
        let sql = "SELECT * FROM sales";
        let cte_names = HashSet::new();

        let result = transform_cte_references(sql, &cte_names);

        assert_eq!(result, "SELECT * FROM sales");
    }

    // ========================================
    // Build Layer Query Tests
    // ========================================

    /// Mock execute function for tests that don't need actual data
    fn mock_execute(_sql: &str) -> Result<DataFrame> {
        // Return empty DataFrame - tests that need real data use DuckDB
        Ok(DataFrame::default())
    }

    #[test]
    fn test_build_layer_query_with_cte() {
        let mut materialized = HashSet::new();
        materialized.insert("sales".to_string());

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::Identifier("sales".to_string()));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        // Should use temp table name
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM __ggsql_cte_sales__".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_with_cte_and_filter() {
        let mut materialized = HashSet::new();
        materialized.insert("sales".to_string());

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::Identifier("sales".to_string()));
        layer.filter = Some(FilterExpression::new("year = 2024"));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM __ggsql_cte_sales__ WHERE year = 2024".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_without_cte() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::Identifier("some_table".to_string()));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        // Should use table name directly
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM some_table".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_table_with_filter() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::Identifier("some_table".to_string()));
        layer.filter = Some(FilterExpression::new("value > 100"));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM some_table WHERE value > 100".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::FilePath("data/sales.csv".to_string()));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        // File paths should be wrapped in single quotes
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM 'data/sales.csv'".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path_with_filter() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::FilePath("data.parquet".to_string()));
        layer.filter = Some(FilterExpression::new("x > 10"));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &[], &mock_execute);

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM 'data.parquet' WHERE x > 10".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_none_source_with_filter() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.filter = Some(FilterExpression::new("category = 'A'"));

        let result = build_layer_query(&layer, &materialized, true, 0, None, &[], &mock_execute);

        // Should query __ggsql_global__ with filter
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM __ggsql_global__ WHERE category = 'A'".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_none_source_no_filter() {
        let materialized = HashSet::new();

        let layer = Layer::new(Geom::Point);

        let result = build_layer_query(&layer, &materialized, true, 0, None, &[], &mock_execute);

        // Should return None - layer uses __global__ directly
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_build_layer_query_filter_without_global_errors() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::Point);
        layer.filter = Some(FilterExpression::new("x > 10"));

        let result = build_layer_query(&layer, &materialized, false, 2, None, &[], &mock_execute);

        // Should return validation error
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Layer 3")); // layer_idx 2 -> Layer 3 in message
        assert!(err.contains("FILTER"));
    }

    #[test]
    fn test_build_layer_query_with_constants() {
        let materialized = HashSet::new();
        let constants = vec![
            (
                "color".to_string(),
                LiteralValue::String("blue".to_string()),
            ),
            ("size".to_string(), LiteralValue::Number(5.0)),
        ];

        let mut layer = Layer::new(Geom::Point);
        layer.source = Some(LayerSource::Identifier("some_table".to_string()));

        let result = build_layer_query(&layer, &materialized, false, 0, None, &constants, &mock_execute);

        // Should inject constants as columns
        let query = result.unwrap().unwrap();
        assert!(query.contains("SELECT *"));
        assert!(query.contains("'blue' AS __ggsql_const_color__"));
        assert!(query.contains("5 AS __ggsql_const_size__"));
        assert!(query.contains("FROM some_table"));
    }

    #[test]
    fn test_build_layer_query_constants_on_global() {
        let materialized = HashSet::new();
        let constants = vec![("fill".to_string(), LiteralValue::String("red".to_string()))];

        // No source but has constants - should use __ggsql_global__
        let layer = Layer::new(Geom::Point);

        let result = build_layer_query(&layer, &materialized, true, 0, None, &constants, &mock_execute);

        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM __ggsql_global__"));
        assert!(query.contains("'red' AS __ggsql_const_fill__"));
    }

    // ========================================
    // End-to-End CTE Reference Tests
    // ========================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_layer_references_cte_from_global() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with CTE defined in global SQL, referenced by layer
        let query = r#"
            WITH sales AS (
                SELECT 1 as date, 100 as revenue, 'A' as region
                UNION ALL
                SELECT 2, 200, 'B'
            ),
            targets AS (
                SELECT 1 as date, 150 as goal
                UNION ALL
                SELECT 2, 180
            )
            SELECT * FROM sales
            VISUALISE
            DRAW line MAPPING date AS x, revenue AS y
            DRAW point MAPPING date AS x, goal AS y FROM targets
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have global data (from sales) and layer 1 data (from targets CTE)
        assert!(result.data.contains_key("__global__"));
        assert!(result.data.contains_key("__layer_1__"));

        // Global should have 2 rows (from sales)
        let global_df = result.data.get("__global__").unwrap();
        assert_eq!(global_df.height(), 2);

        // Layer 1 should have 2 rows (from targets CTE)
        let layer_df = result.data.get("__layer_1__").unwrap();
        assert_eq!(layer_df.height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_layer_references_cte_with_filter() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with CTE and layer that references it with a filter
        let query = r#"
            WITH data AS (
                SELECT 1 as x, 10 as y, 'A' as category
                UNION ALL SELECT 2, 20, 'B'
                UNION ALL SELECT 3, 30, 'A'
                UNION ALL SELECT 4, 40, 'B'
            )
            SELECT * FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM data FILTER category = 'A'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Global should have all 4 rows
        let global_df = result.data.get("__global__").unwrap();
        assert_eq!(global_df.height(), 4);

        // Layer 1 should have 2 rows (filtered to category = 'A')
        let layer_df = result.data.get("__layer_1__").unwrap();
        assert_eq!(layer_df.height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_multiple_layers_reference_different_ctes() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query with multiple CTEs, each referenced by different layers
        let query = r#"
            WITH
                line_data AS (SELECT 1 as x, 100 as y UNION ALL SELECT 2, 200),
                point_data AS (SELECT 1 as x, 150 as y UNION ALL SELECT 2, 250),
                bar_data AS (SELECT 1 as x, 50 as y UNION ALL SELECT 2, 75)
            VISUALISE
            DRAW line MAPPING x AS x, y AS y FROM line_data
            DRAW point MAPPING x AS x, y AS y FROM point_data
            DRAW bar MAPPING x AS x, y AS y FROM bar_data
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have 3 layer datasets, no global (since no trailing SELECT)
        assert!(!result.data.contains_key("__global__"));
        assert!(result.data.contains_key("__layer_0__"));
        assert!(result.data.contains_key("__layer_1__"));
        assert!(result.data.contains_key("__layer_2__"));

        // Each layer should have 2 rows
        assert_eq!(result.data.get("__layer_0__").unwrap().height(), 2);
        assert_eq!(result.data.get("__layer_1__").unwrap().height(), 2);
        assert_eq!(result.data.get("__layer_2__").unwrap().height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_cte_chain_dependencies() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // CTE b references CTE a - tests that transform_cte_references works during materialization
        let query = r#"
            WITH
                raw_data AS (
                    SELECT 1 as id, 100 as value
                    UNION ALL SELECT 2, 200
                    UNION ALL SELECT 3, 300
                ),
                filtered AS (
                    SELECT * FROM raw_data WHERE value > 150
                ),
                aggregated AS (
                    SELECT COUNT(*) as cnt, SUM(value) as total FROM filtered
                )
            VISUALISE
            DRAW point MAPPING cnt AS x, total AS y FROM aggregated
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data from aggregated CTE
        assert!(result.data.contains_key("__layer_0__"));
        let layer_df = result.data.get("__layer_0__").unwrap();
        assert_eq!(layer_df.height(), 1); // Single aggregated row
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_visualise_from_cte() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // WITH clause with VISUALISE FROM (parser injects SELECT * FROM monthly)
        let query = r#"
            WITH monthly AS (
                SELECT 1 as month, 1000 as revenue
                UNION ALL SELECT 2, 1200
                UNION ALL SELECT 3, 1100
            )
            VISUALISE month AS x, revenue AS y FROM monthly
            DRAW line
            DRAW point
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // VISUALISE FROM causes SELECT injection, so we have global data
        assert!(result.data.contains_key("__global__"));
        // Layers without their own FROM use global directly (no separate entry)
        assert!(!result.data.contains_key("__layer_0__"));
        assert!(!result.data.contains_key("__layer_1__"));

        // Global should have 3 rows
        assert_eq!(result.data.get("__global__").unwrap().height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_multiple_ctes_no_global_select() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // WITH clause without trailing SELECT - each layer uses its own CTE
        let query = r#"
            WITH
                series_a AS (SELECT 1 as x, 10 as y UNION ALL SELECT 2, 20),
                series_b AS (SELECT 1 as x, 15 as y UNION ALL SELECT 2, 25)
            VISUALISE
            DRAW line MAPPING x AS x, y AS y FROM series_a
            DRAW point MAPPING x AS x, y AS y FROM series_b
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // No global data since no trailing SELECT
        assert!(!result.data.contains_key("__global__"));
        // Each layer has its own data
        assert!(result.data.contains_key("__layer_0__"));
        assert!(result.data.contains_key("__layer_1__"));

        assert_eq!(result.data.get("__layer_0__").unwrap().height(), 2);
        assert_eq!(result.data.get("__layer_1__").unwrap().height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_layer_from_cte_mixed_with_global() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // First layer uses global data, second layer uses CTE
        let query = r#"
            WITH targets AS (
                SELECT 1 as x, 50 as target
                UNION ALL SELECT 2, 60
            )
            SELECT 1 as x, 100 as actual
            UNION ALL SELECT 2, 120
            VISUALISE
            DRAW line MAPPING x AS x, actual AS y
            DRAW point MAPPING x AS x, target AS y FROM targets
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Global from SELECT, layer 1 from CTE
        assert!(result.data.contains_key("__global__"));
        assert!(result.data.contains_key("__layer_1__"));
        // Layer 0 has no entry (uses global directly)
        assert!(!result.data.contains_key("__layer_0__"));

        assert_eq!(result.data.get("__global__").unwrap().height(), 2);
        assert_eq!(result.data.get("__layer_1__").unwrap().height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_cte_with_complex_filter_expression() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Test complex filter expressions work correctly with temp tables
        let query = r#"
            WITH data AS (
                SELECT 1 as x, 10 as y, 'A' as cat, true as active
                UNION ALL SELECT 2, 20, 'B', true
                UNION ALL SELECT 3, 30, 'A', false
                UNION ALL SELECT 4, 40, 'B', false
                UNION ALL SELECT 5, 50, 'A', true
            )
            SELECT * FROM data
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM data FILTER cat = 'A' AND active = true
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Global should have all 5 rows
        assert_eq!(result.data.get("__global__").unwrap().height(), 5);

        // Layer 1 should have 2 rows (cat='A' AND active=true)
        assert_eq!(result.data.get("__layer_1__").unwrap().height(), 2);
    }

    // ========================================
    // Statistical Transformation Tests
    // ========================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_histogram_stat_transform() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with continuous values
        reader
            .connection()
            .execute(
                "CREATE TABLE hist_test AS SELECT RANDOM() * 100 as value FROM range(100)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM hist_test
            VISUALISE
            DRAW histogram MAPPING value AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data with binned results
        assert!(result.data.contains_key("__layer_0__"));
        let layer_df = result.data.get("__layer_0__").unwrap();

        // Should have x (bin) and y (count) columns
        let col_names: Vec<&str> = layer_df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert!(col_names.contains(&"x"));
        assert!(col_names.contains(&"y"));

        // Should have fewer rows than original (binned)
        assert!(layer_df.height() < 100);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_count_stat_transform() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories
        reader
            .connection()
            .execute(
                "CREATE TABLE bar_test AS SELECT * FROM (VALUES ('A'), ('B'), ('A'), ('C'), ('A'), ('B')) AS t(category)",
                duckdb::params![],
            )
            .unwrap();

        // Bar with only x mapped - should apply count stat
        let query = r#"
            SELECT * FROM bar_test
            VISUALISE
            DRAW bar MAPPING category AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data with counted results
        assert!(result.data.contains_key("__layer_0__"));
        let layer_df = result.data.get("__layer_0__").unwrap();

        // Should have 3 rows (3 unique categories: A, B, C)
        assert_eq!(layer_df.height(), 3);

        // Should have x and y columns
        let col_names: Vec<&str> = layer_df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert!(col_names.contains(&"x"));
        assert!(col_names.contains(&"y"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_no_transform_when_y_mapped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories and values
        reader
            .connection()
            .execute(
                "CREATE TABLE bar_y_test AS SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)",
                duckdb::params![],
            )
            .unwrap();

        // Bar with x and y mapped - should NOT apply count stat
        let query = r#"
            SELECT * FROM bar_y_test
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should NOT have layer 0 data (no transformation needed, uses global)
        assert!(!result.data.contains_key("__layer_0__"));
        assert!(result.data.contains_key("__global__"));

        // Global should have original 3 rows
        let global_df = result.data.get("__global__").unwrap();
        assert_eq!(global_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_histogram_with_facet() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with region facet
        reader
            .connection()
            .execute(
                "CREATE TABLE facet_hist_test AS SELECT * FROM (VALUES
                    (10.0, 'North'), (20.0, 'North'), (30.0, 'North'), (40.0, 'North'), (50.0, 'North'),
                    (15.0, 'South'), (25.0, 'South'), (35.0, 'South'), (45.0, 'South'), (55.0, 'South')
                ) AS t(value, region)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM facet_hist_test
            VISUALISE
            DRAW histogram MAPPING value AS x
            FACET WRAP region
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data with binned results
        assert!(result.data.contains_key("__layer_0__"));
        let layer_df = result.data.get("__layer_0__").unwrap();

        // Should have region column preserved for faceting
        let col_names: Vec<&str> = layer_df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert!(col_names.contains(&"region"));
        assert!(col_names.contains(&"x"));
        assert!(col_names.contains(&"y"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_count_with_partition_by() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories and groups
        reader
            .connection()
            .execute(
                "CREATE TABLE bar_partition_test AS SELECT * FROM (VALUES
                    ('A', 'G1'), ('B', 'G1'), ('A', 'G1'),
                    ('A', 'G2'), ('B', 'G2'), ('C', 'G2')
                ) AS t(category, grp)",
                duckdb::params![],
            )
            .unwrap();

        // Bar with only x mapped and partition by
        let query = r#"
            SELECT * FROM bar_partition_test
            VISUALISE
            DRAW bar MAPPING category AS x PARTITION BY grp
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data with counted results
        assert!(result.data.contains_key("__layer_0__"));
        let layer_df = result.data.get("__layer_0__").unwrap();

        // Should have grp column preserved for grouping
        let col_names: Vec<&str> = layer_df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert!(col_names.contains(&"grp"));
        assert!(col_names.contains(&"x"));
        assert!(col_names.contains(&"y"));

        // G1 has A(2), B(1) = 2 rows; G2 has A(1), B(1), C(1) = 3 rows; total = 5 rows
        assert_eq!(layer_df.height(), 5);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_point_no_stat_transform() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE point_test AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Point geom should NOT apply any stat transform
        let query = r#"
            SELECT * FROM point_test
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should NOT have layer 0 data (no transformation, uses global)
        assert!(!result.data.contains_key("__layer_0__"));
        assert!(result.data.contains_key("__global__"));

        // Global should have original 3 rows
        let global_df = result.data.get("__global__").unwrap();
        assert_eq!(global_df.height(), 3);
    }
}
