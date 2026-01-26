//! Query execution module for ggsql
//!
//! Provides shared execution logic for building data maps from queries,
//! handling both global SQL and layer-specific data sources.

use crate::naming;
use crate::plot::layer::geom::{GeomAesthetics, AESTHETIC_FAMILIES};
use crate::plot::{
    AestheticValue, ColumnInfo, Layer, LiteralValue, OutputRange, ScaleType, Schema, StatResult,
};
use crate::{parser, DataFrame, DataSource, Facet, GgsqlError, Plot, Result};
use polars::prelude::Column;
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
        let temp_table_name = naming::cte_table(cte_name);

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

/// Fetch schema for a query using LIMIT 0
///
/// Executes a schema-only query to determine column names and types.
/// Used to:
/// 1. Resolve wildcard mappings to actual columns
/// 2. Filter group_by to discrete columns only
/// 3. Pass to stat transforms for column validation
fn fetch_layer_schema<F>(query: &str, execute_query: &F) -> Result<Schema>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    let schema_query = format!(
        "SELECT * FROM ({}) AS {} LIMIT 0",
        query,
        naming::SCHEMA_ALIAS
    );
    let df = execute_query(&schema_query)?;

    Ok(df
        .get_columns()
        .iter()
        .map(|col| {
            use polars::prelude::DataType;
            let dtype = col.dtype();
            // Discrete: String, Boolean, Date (grouping by day makes sense), Categorical
            // Continuous: numeric types, Datetime, Time (too granular for grouping)
            let is_discrete =
                matches!(dtype, DataType::String | DataType::Boolean | DataType::Date)
                    || dtype.is_categorical();
            ColumnInfo {
                name: col.name().to_string(),
                is_discrete,
            }
        })
        .collect())
}

/// Determine the data source table name for a layer
///
/// Returns the table/CTE name to query from:
/// - Layer with explicit source (CTE, table, file) → that source name
/// - Layer using global data → None (caller should use global schema)
fn determine_layer_source(layer: &Layer, materialized_ctes: &HashSet<String>) -> Option<String> {
    match &layer.source {
        Some(DataSource::Identifier(name)) => {
            // Check if it's a materialized CTE
            if materialized_ctes.contains(name) {
                Some(naming::cte_table(name))
            } else {
                Some(name.clone())
            }
        }
        Some(DataSource::FilePath(path)) => {
            // File paths need single quotes for DuckDB
            Some(format!("'{}'", path))
        }
        None => {
            // Layer uses global data
            None
        }
    }
}

/// Validate all layers against their schemas
///
/// Validates:
/// - Required aesthetics exist for each geom
/// - SETTING parameters are valid for each geom
/// - Aesthetic columns exist in schema
/// - Partition_by columns exist in schema
/// - Remapping target aesthetics are supported by geom
/// - Remapping source columns are valid stat columns for geom
fn validate(layers: &[Layer], layer_schemas: &[Schema]) -> Result<()> {
    for (idx, (layer, schema)) in layers.iter().zip(layer_schemas.iter()).enumerate() {
        let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();
        let supported = layer.geom.aesthetics().supported;

        // Validate required aesthetics for this geom
        layer
            .validate_required_aesthetics()
            .map_err(|e| GgsqlError::ValidationError(format!("Layer {}: {}", idx + 1, e)))?;

        // Validate SETTING parameters are valid for this geom
        layer
            .validate_settings()
            .map_err(|e| GgsqlError::ValidationError(format!("Layer {}: {}", idx + 1, e)))?;

        // Validate aesthetic columns exist in schema
        for (aesthetic, value) in &layer.mappings.aesthetics {
            // Only validate aesthetics supported by this geom
            if !supported.contains(&aesthetic.as_str()) {
                continue;
            }

            if let Some(col_name) = value.column_name() {
                // Skip synthetic columns (stat-generated or constants)
                if naming::is_synthetic_column(col_name) {
                    continue;
                }
                if !schema_columns.contains(col_name) {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {}: aesthetic '{}' references non-existent column '{}'",
                        idx + 1,
                        aesthetic,
                        col_name
                    )));
                }
            }
        }

        // Validate partition_by columns exist in schema
        for col in &layer.partition_by {
            if !schema_columns.contains(col.as_str()) {
                return Err(GgsqlError::ValidationError(format!(
                    "Layer {}: PARTITION BY references non-existent column '{}'",
                    idx + 1,
                    col
                )));
            }
        }

        // Validate remapping target aesthetics are supported by geom
        // Target can be in supported OR hidden (hidden = valid REMAPPING targets but not MAPPING targets)
        let aesthetics_info = layer.geom.aesthetics();
        for target_aesthetic in layer.remappings.aesthetics.keys() {
            let is_supported = aesthetics_info
                .supported
                .contains(&target_aesthetic.as_str());
            let is_hidden = aesthetics_info.hidden.contains(&target_aesthetic.as_str());
            if !is_supported && !is_hidden {
                return Err(GgsqlError::ValidationError(format!(
                    "Layer {}: REMAPPING targets unsupported aesthetic '{}' for geom '{}'",
                    idx + 1,
                    target_aesthetic,
                    layer.geom
                )));
            }
        }

        // Validate remapping source columns are valid stat columns for this geom
        let valid_stat_columns = layer.geom.valid_stat_columns();
        for stat_value in layer.remappings.aesthetics.values() {
            if let Some(stat_col) = stat_value.column_name() {
                if !valid_stat_columns.contains(&stat_col) {
                    if valid_stat_columns.is_empty() {
                        return Err(GgsqlError::ValidationError(format!(
                            "Layer {}: REMAPPING not supported for geom '{}' (no stat transform)",
                            idx + 1,
                            layer.geom
                        )));
                    } else {
                        return Err(GgsqlError::ValidationError(format!(
                            "Layer {}: REMAPPING references unknown stat column '{}'. Valid stat columns for geom '{}' are: {}",
                            idx + 1,
                            stat_col,
                            layer.geom,
                            valid_stat_columns.join(", ")
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Add discrete mapped columns to partition_by for all layers
///
/// For each layer, examines all aesthetic mappings and adds any that map to
/// discrete columns (string, boolean, date, categorical) to the layer's
/// partition_by. This ensures proper grouping for all layers, not just stat geoms.
///
/// Columns already in partition_by (from explicit PARTITION BY clause) are skipped.
/// Stat-consumed aesthetics (x for bar, x for histogram) are also skipped.
fn add_discrete_columns_to_partition_by(layers: &mut [Layer], layer_schemas: &[Schema]) {
    // Positional aesthetics should NOT be auto-added to grouping.
    // Stats that need to group by positional aesthetics (like bar/histogram)
    // already handle this themselves via stat_consumed_aesthetics().
    const POSITIONAL_AESTHETICS: &[&str] =
        &["x", "y", "xmin", "xmax", "ymin", "ymax", "xend", "yend"];

    for (layer, schema) in layers.iter_mut().zip(layer_schemas.iter()) {
        let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();
        let discrete_columns: HashSet<&str> = schema
            .iter()
            .filter(|c| c.is_discrete)
            .map(|c| c.name.as_str())
            .collect();

        // Get aesthetics consumed by stat transforms (if any)
        let consumed_aesthetics = layer.geom.stat_consumed_aesthetics();

        for (aesthetic, value) in &layer.mappings.aesthetics {
            // Skip positional aesthetics - these should not trigger auto-grouping
            if POSITIONAL_AESTHETICS.contains(&aesthetic.as_str()) {
                continue;
            }

            // Skip stat-consumed aesthetics (they're transformed, not grouped)
            if consumed_aesthetics.contains(&aesthetic.as_str()) {
                continue;
            }

            if let Some(col) = value.column_name() {
                // Skip if column doesn't exist in schema
                if !schema_columns.contains(col) {
                    continue;
                }

                // Skip if column is not discrete
                if !discrete_columns.contains(col) {
                    continue;
                }

                // Skip if already in partition_by
                if layer.partition_by.contains(&col.to_string()) {
                    continue;
                }

                layer.partition_by.push(col.to_string());
            }
        }
    }
}

/// Extract constant aesthetics from a layer
fn extract_constants(layer: &Layer) -> Vec<(String, LiteralValue)> {
    layer
        .mappings
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
fn replace_literals_with_columns(spec: &mut Plot) {
    for (layer_idx, layer) in spec.layers.iter_mut().enumerate() {
        for (aesthetic, value) in layer.mappings.aesthetics.iter_mut() {
            if matches!(value, AestheticValue::Literal(_)) {
                // Use layer-indexed column name for layers using global data (no source, no filter)
                // Use non-indexed name for layers with their own data (filter or explicit source)
                let col_name = if layer.source.is_none() && layer.filter.is_none() {
                    naming::const_column_indexed(aesthetic, layer_idx)
                } else {
                    naming::const_column(aesthetic)
                };
                *value = AestheticValue::standard_column(col_name);
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

        let temp_table_name = naming::cte_table(&cte.name);
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
/// Constants are injected as synthetic columns (e.g., `'value' AS __ggsql_const_color__`).
/// Also applies statistical transformations for geoms that need them
/// (e.g., histogram binning, bar counting).
///
/// Returns:
/// - `Ok(Some(query))` - execute this query and store result
/// - `Ok(None)` - layer uses `__global__` directly (no source, no filter, no constants, no stat transform)
/// - `Err(...)` - validation error (e.g., filter without global data)
///
/// Note: This function takes `&mut Layer` because stat transforms may add new aesthetic mappings
/// (e.g., mapping y to `__ggsql_stat__count` for histogram or bar count).
fn build_layer_query<F>(
    layer: &mut Layer,
    schema: &Schema,
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
    // Apply default parameter values (e.g., bins=30 for histogram)
    // Must be done before any immutable borrows of layer
    layer.apply_default_params();

    let filter = layer.filter.as_ref().map(|f| f.as_str());
    let order_by = layer.order_by.as_ref().map(|f| f.as_str());

    let table_name = match &layer.source {
        Some(DataSource::Identifier(name)) => {
            // Check if it's a materialized CTE
            if materialized_ctes.contains(name) {
                naming::cte_table(name)
            } else {
                name.clone()
            }
        }
        Some(DataSource::FilePath(path)) => {
            // File paths need single quotes
            format!("'{}'", path)
        }
        None => {
            // No source - validate and use global if filter, order_by or constants present
            if filter.is_some() || order_by.is_some() || !constants.is_empty() {
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} has a FILTER, ORDER BY, or constants but no data source. Either provide a SQL query or use MAPPING FROM.",
                        layer_idx + 1
                    )));
                }
                naming::global_table()
            } else if layer.geom.needs_stat_transform(&layer.mappings) {
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} requires data for statistical transformation but no data source.",
                        layer_idx + 1
                    )));
                }
                naming::global_table()
            } else {
                // No source, no filter, no constants, no stat transform - use __global__ data directly
                return Ok(None);
            }
        }
    };

    // Build base query with optional constant columns
    let mut query = if constants.is_empty() {
        format!("SELECT * FROM {}", table_name)
    } else {
        let const_cols: Vec<String> = constants
            .iter()
            .map(|(aes, lit)| format!("{} AS {}", literal_to_sql(lit), naming::const_column(aes)))
            .collect();
        format!("SELECT *, {} FROM {}", const_cols.join(", "), table_name)
    };

    // Combine partition_by (which includes discrete mapped columns) and facet variables for grouping
    // Note: partition_by is pre-populated with discrete columns by add_discrete_columns_to_partition_by()
    let mut group_by = layer.partition_by.clone();
    if let Some(f) = facet {
        for var in f.get_variables() {
            if !group_by.contains(&var) {
                group_by.push(var);
            }
        }
    }

    // Apply filter
    if let Some(f) = filter {
        query = format!("{} WHERE {}", query, f);
    }

    // Apply statistical transformation (after filter, uses combined group_by)
    // Returns StatResult::Identity for no transformation, StatResult::Transformed for transformed query
    let stat_result = layer.geom.apply_stat_transform(
        &query,
        schema,
        &layer.mappings,
        &group_by,
        &layer.parameters,
        execute_query,
    )?;

    match stat_result {
        StatResult::Transformed {
            query: transformed_query,
            stat_columns,
            dummy_columns,
            consumed_aesthetics,
        } => {
            // Build final remappings: start with geom defaults, override with user remappings
            let mut final_remappings: HashMap<String, String> = layer
                .geom
                .default_remappings()
                .iter()
                .map(|(stat, aes)| (stat.to_string(), aes.to_string()))
                .collect();

            // User REMAPPING overrides defaults
            // In remappings, the aesthetic key is the target, and the column name is the stat name
            for (aesthetic, value) in &layer.remappings.aesthetics {
                if let Some(stat_name) = value.column_name() {
                    // stat_name maps to this aesthetic
                    final_remappings.insert(stat_name.to_string(), aesthetic.clone());
                }
            }

            // FIRST: Remove consumed aesthetics - they were used as stat input, not visual output
            for aes in &consumed_aesthetics {
                layer.mappings.aesthetics.remove(aes);
            }

            // THEN: Apply stat_columns to layer aesthetics using the remappings
            for stat in &stat_columns {
                if let Some(aesthetic) = final_remappings.get(stat) {
                    let col = naming::stat_column(stat);
                    let is_dummy = dummy_columns.contains(stat);
                    layer.mappings.insert(
                        aesthetic.clone(),
                        if is_dummy {
                            AestheticValue::dummy_column(col)
                        } else {
                            AestheticValue::standard_column(col)
                        },
                    );
                }
            }

            // Use the transformed query
            let mut final_query = transformed_query;
            if let Some(o) = order_by {
                final_query = format!("{} ORDER BY {}", final_query, o);
            }
            Ok(Some(final_query))
        }
        StatResult::Identity => {
            // Identity - no stat transformation
            // If the layer has no explicit source, no filter, no order_by, and no constants,
            // we can use __global__ directly (return None)
            if layer.source.is_none()
                && filter.is_none()
                && order_by.is_none()
                && constants.is_empty()
            {
                Ok(None)
            } else {
                // Layer has filter, order_by, or constants - still need the query
                let mut final_query = query;
                if let Some(o) = order_by {
                    final_query = format!("{} ORDER BY {}", final_query, o);
                }
                Ok(Some(final_query))
            }
        }
    }
}

/// Merge global mappings into layer aesthetics and expand wildcards
///
/// This function performs smart wildcard expansion with schema awareness:
/// 1. Merges explicit global aesthetics into layers (layer aesthetics take precedence)
/// 2. Only merges aesthetics that the geom supports
/// 3. Expands wildcards by adding mappings only for supported aesthetics that:
///    - Are not already mapped (either from global or layer)
///    - Have a matching column in the layer's schema
/// 4. Moreover it propagates 'color' to 'fill' and 'stroke'
fn merge_global_mappings_into_layers(specs: &mut [Plot], layer_schemas: &[Schema]) {
    for spec in specs {
        for (layer, schema) in spec.layers.iter_mut().zip(layer_schemas.iter()) {
            let supported = layer.geom.aesthetics().supported;
            let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();

            // 1. First merge explicit global aesthetics (layer overrides global)
            for (aesthetic, value) in &spec.global_mappings.aesthetics {
                if supported.contains(&aesthetic.as_str()) {
                    layer
                        .mappings
                        .aesthetics
                        .entry(aesthetic.clone())
                        .or_insert(value.clone());
                }
            }

            // 2. Smart wildcard expansion: only expand to columns that exist in schema
            let has_wildcard = layer.mappings.wildcard || spec.global_mappings.wildcard;
            if has_wildcard {
                for &aes in supported {
                    // Only create mapping if column exists in the schema
                    if schema_columns.contains(aes) {
                        layer
                            .mappings
                            .aesthetics
                            .entry(crate::parser::builder::normalise_aes_name(aes))
                            .or_insert(AestheticValue::standard_column(aes));
                    }
                }
            }

            // Clear wildcard flag since it's been resolved
            layer.mappings.wildcard = false;
        }
    }
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

// Let 'color' aesthetics fill defaults for the 'stroke' and 'fill' aesthetics
fn split_color_aesthetic(layers: &mut Vec<Layer>) {
    for layer in layers {
        if !layer.mappings.aesthetics.contains_key("color") {
            continue;
        }
        let supported = layer.geom.aesthetics().supported;
        for &aes in &["stroke", "fill"] {
            if !supported.contains(&aes) {
                continue;
            }
            let color = layer.mappings.aesthetics.get("color").unwrap().clone();
            layer
                .mappings
                .aesthetics
                .entry(aes.to_string())
                .or_insert(color);
        }
    }
}

/// Result of preparing data for visualization
pub struct PreparedData {
    /// Data map with global and layer-specific DataFrames
    pub data: HashMap<String, DataFrame>,
    /// Parsed and resolved visualization specifications
    pub specs: Vec<Plot>,
}

/// Build data map from a query using a custom query executor function
///
/// This is the most flexible variant that works with any query execution strategy,
/// including shared state readers in REST API contexts.
///
/// # Arguments
/// * `query` - The full ggsql query string
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
    // constant values per layer (e.g., layer 0: 'value' AS color, layer 1: 'value2' AS color)
    let first_spec = &specs[0];

    // First, extract global constants from VISUALISE clause (e.g., VISUALISE 'value' AS color)
    // These apply to all layers that use global data
    let global_mappings_constants: Vec<(String, LiteralValue)> = first_spec
        .global_mappings
        .aesthetics
        .iter()
        .filter_map(|(aesthetic, value)| {
            if let AestheticValue::Literal(lit) = value {
                Some((aesthetic.clone(), lit.clone()))
            } else {
                None
            }
        })
        .collect();

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
    // (these will be injected into the global data table)
    for layer_idx in &global_data_layer_indices {
        for (aes, lit) in &global_mappings_constants {
            // Only add if this layer doesn't already have this aesthetic from its own MAPPING
            let layer = &first_spec.layers[*layer_idx];
            if !layer.mappings.contains_key(aes) {
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
                            naming::const_column_indexed(aes, *layer_idx)
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
                "CREATE OR REPLACE TEMP TABLE {} AS {}",
                naming::global_table(),
                global_query
            );
            execute_query(&create_global)?;

            // Read back into DataFrame for data_map
            let df = execute_query(&format!("SELECT * FROM {}", naming::global_table()))?;
            data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);
        }
    }

    // Fetch schemas upfront for smart wildcard expansion and validation
    let has_global = data_map.contains_key(naming::GLOBAL_DATA_KEY);

    // Fetch global schema (used by layers without explicit source)
    let global_schema = if has_global {
        fetch_layer_schema(
            &format!("SELECT * FROM {}", naming::global_table()),
            &execute_query,
        )?
    } else {
        Vec::new()
    };

    // Fetch schemas for all layers
    let mut layer_schemas: Vec<Schema> = Vec::new();
    for layer in &specs[0].layers {
        let source = determine_layer_source(layer, &materialized_ctes);
        let schema = match source {
            Some(src) => {
                let base_query = format!("SELECT * FROM {}", src);
                fetch_layer_schema(&base_query, &execute_query)?
            }
            None => {
                // Layer uses global data - use global schema
                global_schema.clone()
            }
        };
        layer_schemas.push(schema);
    }

    // Merge global mappings into layer aesthetics and expand wildcards
    // Smart wildcard expansion only creates mappings for columns that exist in schema
    merge_global_mappings_into_layers(&mut specs, &layer_schemas);

    // Validate all layers against their schemas
    // This catches errors early with clear error messages:
    // - Missing required aesthetics
    // - Invalid SETTING parameters
    // - Non-existent columns in mappings
    // - Non-existent columns in PARTITION BY
    // - Unsupported aesthetics in REMAPPING
    // - Invalid stat columns in REMAPPING
    validate(&specs[0].layers, &layer_schemas)?;

    // Add discrete mapped columns to partition_by for all layers
    // This ensures proper grouping for color, fill, shape, etc. aesthetics
    add_discrete_columns_to_partition_by(&mut specs[0].layers, &layer_schemas);

    // Execute layer-specific queries
    // build_layer_query() handles all cases:
    // - Layer with source (CTE, table, or file) → query that source
    // - Layer with filter/order_by but no source → query __ggsql_global__ with filter/order_by and constants
    // - Layer with no source, no filter, no order_by → returns None (use global directly, constants already injected)
    let facet = specs[0].facet.clone();

    for (idx, layer) in specs[0].layers.iter_mut().enumerate() {
        // For layers using global data without filter, constants are already in global data
        // (injected with layer-indexed names). For other layers, extract constants for injection.
        let constants = if layer.source.is_none() && layer.filter.is_none() {
            vec![] // Constants already in global data
        } else {
            extract_constants(layer)
        };

        // Get mutable reference to layer for stat transform to update aesthetics
        if let Some(layer_query) = build_layer_query(
            layer,
            &layer_schemas[idx],
            &materialized_ctes,
            has_global,
            idx,
            facet.as_ref(),
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
            data_map.insert(naming::layer_key(idx), df);
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
    let has_layer_without_source = specs[0]
        .layers
        .iter()
        .any(|l| l.source.is_none() && l.filter.is_none());
    if has_layer_without_source && !data_map.contains_key(naming::GLOBAL_DATA_KEY) {
        return Err(GgsqlError::ValidationError(
            "Some layers use global data but no SQL query was provided.".to_string(),
        ));
    }

    // Post-process specs: replace literals with column references and compute labels
    for spec in &mut specs {
        // Replace literal aesthetic values with column references to synthetic constant columns
        replace_literals_with_columns(spec);
        // Compute aesthetic labels (uses first non-constant column, respects user-specified labels)
        spec.compute_aesthetic_labels();
        // Divide 'color' over 'stroke' and 'fill'. This needs to happens after
        // literals have associated columns.
        split_color_aesthetic(&mut spec.layers);
    }

    // Resolve scale types from data for scales without explicit types
    for spec in &mut specs {
        resolve_scales(spec, &data_map)?;
    }

    // Apply out-of-bounds handling to data based on scale oob properties
    for spec in &specs {
        apply_scale_oob(spec, &mut data_map)?;
    }

    Ok(PreparedData {
        data: data_map,
        specs,
    })
}

// =============================================================================
// Scale Resolution
// =============================================================================

/// Resolve scale properties from data after materialization.
///
/// For each scale, this function:
/// 1. Infers scale_type from column data types if not explicitly set
/// 2. Resolves input_range (domain) using the scale type's `resolve_input_range` method
///
/// The function inspects columns mapped to the aesthetic (including family
/// members like xmin/xmax for "x") and computes appropriate ranges.
fn resolve_scales(spec: &mut Plot, data_map: &HashMap<String, DataFrame>) -> Result<()> {
    for idx in 0..spec.scales.len() {
        // Clone aesthetic to avoid borrow issues with find_columns_for_aesthetic
        let aesthetic = spec.scales[idx].aesthetic.clone();

        // Find column references for this aesthetic (including family members)
        let column_refs =
            find_columns_for_aesthetic(&spec.global_mappings, &spec.layers, &aesthetic, data_map);

        if column_refs.is_empty() {
            continue;
        }

        // Infer scale_type if not already set
        if spec.scales[idx].scale_type.is_none() {
            spec.scales[idx].scale_type = Some(ScaleType::infer(column_refs[0].dtype()));
        }

        // Clone scale_type (cheap Arc clone) to avoid borrow conflict with mutations
        let scale_type = spec.scales[idx].scale_type.clone();
        if let Some(st) = scale_type {
            // Resolve and validate properties (fills in defaults, rejects unknown)
            spec.scales[idx].properties = st
                .resolve_properties(&aesthetic, &spec.scales[idx].properties)
                .map_err(|e| {
                    GgsqlError::ValidationError(format!("Scale '{}': {}", aesthetic, e))
                })?;

            // Resolve transform method (fills in default, validates user input)
            let resolved_transform = st
                .resolve_transform(&aesthetic, spec.scales[idx].transform_method.as_deref())
                .map_err(|e| {
                    GgsqlError::ValidationError(format!("Scale '{}': {}", aesthetic, e))
                })?;
            spec.scales[idx].transform_method = Some(resolved_transform);

            // Resolve input range using the scale type's method
            let resolved_range = st
                .resolve_input_range(
                    spec.scales[idx].input_range.as_deref(),
                    &column_refs,
                    &spec.scales[idx].properties,
                )
                .map_err(|e| {
                    GgsqlError::ValidationError(format!("Scale '{}': {}", aesthetic, e))
                })?;

            if let Some(range) = resolved_range {
                spec.scales[idx].input_range = Some(range);
            }

            // Resolve output range (only if not already set)
            if spec.scales[idx].output_range.is_none() {
                if let Some(default_range) = st
                    .default_output_range(&aesthetic, spec.scales[idx].input_range.as_deref())
                    .map_err(GgsqlError::ValidationError)?
                {
                    spec.scales[idx].output_range = Some(OutputRange::Array(default_range));
                }
            }
        }

        // Expand named palettes to explicit arrays
        if let Some(OutputRange::Palette(ref name)) = spec.scales[idx].output_range.clone() {
            use crate::plot::scale::palettes;

            // Determine if this is a color or shape aesthetic
            let palette_values = match aesthetic.as_str() {
                "shape" => palettes::get_shape_palette(name),
                _ => palettes::get_color_palette(name),
            };

            if let Some(palette) = palette_values {
                // Size to input_range length, or use full palette
                let count = spec.scales[idx]
                    .input_range
                    .as_ref()
                    .map(|r| r.len())
                    .unwrap_or(palette.len());
                let expanded = palettes::expand_palette(palette, count, name)
                    .map_err(GgsqlError::ValidationError)?;
                spec.scales[idx].output_range = Some(OutputRange::Array(expanded));
            }
            // If palette not found, leave as Palette variant for Vega-Lite to handle
        }
    }

    Ok(())
}

/// Find all columns for an aesthetic (including family members like xmin/xmax for "x").
/// Each mapping is looked up in its corresponding data source.
/// Returns references to the Columns found.
fn find_columns_for_aesthetic<'a>(
    global_mappings: &crate::plot::Mappings,
    layers: &[Layer],
    aesthetic: &str,
    data_map: &'a HashMap<String, DataFrame>,
) -> Vec<&'a Column> {
    let mut column_refs = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);
    let global_df = data_map.get(naming::GLOBAL_DATA_KEY);

    // Check global mapping → look up in global data
    if let Some(df) = global_df {
        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = global_mappings.get(aes_name) {
                if let Ok(column) = df.column(name) {
                    column_refs.push(column);
                }
            }
        }
    }

    // Check each layer's mapping → look up in layer data OR global data
    for (i, layer) in layers.iter().enumerate() {
        // Use layer-specific data if available, otherwise fall back to global
        let df = data_map.get(&naming::layer_key(i)).or(global_df);
        if let Some(df) = df {
            for aes_name in &aesthetics_to_check {
                if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                    if let Ok(column) = df.column(name) {
                        column_refs.push(column);
                    }
                }
            }
        }
    }

    column_refs
}

/// Get all aesthetics in the same family as the given aesthetic.
/// For primary aesthetics like "x", returns ["x", "xmin", "xmax", "x2", "xend"].
/// For non-family aesthetics like "color", returns just ["color"].
fn get_aesthetic_family(aesthetic: &str) -> Vec<&str> {
    // First, determine the primary aesthetic
    let primary = GeomAesthetics::primary_aesthetic(aesthetic);

    // If aesthetic is not a primary (it's a variant), just return the aesthetic itself
    // since scales should be defined for primary aesthetics
    if primary != aesthetic {
        return vec![aesthetic];
    }

    // Collect primary + all variants that map to this primary
    let mut family = vec![primary];
    for (variant, prim) in AESTHETIC_FAMILIES {
        if *prim == primary {
            family.push(*variant);
        }
    }

    family
}

// =============================================================================
// Out-of-Bounds (OOB) Handling
// =============================================================================

use crate::plot::scale::{OOB_CENSOR, OOB_KEEP, OOB_SQUISH};
use crate::plot::{ArrayElement, ParameterValue};

/// Apply out-of-bounds handling to data based on scale oob properties.
///
/// For each scale with `oob != "keep"`, this function transforms the data:
/// - `censor`: Filter out rows where the aesthetic's column values fall outside the input range
/// - `squish`: Clamp column values to the input range limits (continuous only)
///
/// For discrete scales, only `censor` is supported - values not in the allowed set are filtered.
fn apply_scale_oob(spec: &Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    for scale in &spec.scales {
        // Get oob mode, skip if "keep"
        let oob_mode = match scale.properties.get("oob") {
            Some(ParameterValue::String(s)) if s != OOB_KEEP => s.as_str(),
            _ => continue,
        };

        // Get input range, skip if none
        let input_range = match &scale.input_range {
            Some(r) if !r.is_empty() => r,
            _ => continue,
        };

        // Find all (data_key, column_name) pairs for this aesthetic
        let column_sources = find_columns_for_aesthetic_with_sources(
            &spec.global_mappings,
            &spec.layers,
            &scale.aesthetic,
        );

        // Determine if this is a numeric or discrete range
        let is_numeric_range = matches!(
            (&input_range[0], input_range.get(1)),
            (ArrayElement::Number(_), Some(ArrayElement::Number(_)))
        );

        // Apply transformation to each (data_key, column_name) pair
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
                    continue;
                }

                let transformed = if is_numeric_range {
                    // Numeric range - extract min/max
                    let (range_min, range_max) = match (&input_range[0], &input_range[1]) {
                        (ArrayElement::Number(lo), ArrayElement::Number(hi)) => (*lo, *hi),
                        _ => continue,
                    };
                    apply_oob_to_column_numeric(df, &col_name, range_min, range_max, oob_mode)?
                } else {
                    // Discrete range - collect allowed values as strings
                    let allowed_values: std::collections::HashSet<String> = input_range
                        .iter()
                        .filter_map(|elem| match elem {
                            ArrayElement::String(s) => Some(s.clone()),
                            ArrayElement::Number(n) => Some(n.to_string()),
                            ArrayElement::Boolean(b) => Some(b.to_string()),
                            ArrayElement::Null => None,
                        })
                        .collect();
                    apply_oob_to_column_discrete(df, &col_name, &allowed_values, oob_mode)?
                };
                data_map.insert(data_key, transformed);
            }
        }
    }
    Ok(())
}

/// Find all (data_key, column_name) pairs for an aesthetic (including family members).
/// Returns tuples of (data source key, column name) for use in transformations.
fn find_columns_for_aesthetic_with_sources(
    global_mappings: &crate::plot::Mappings,
    layers: &[Layer],
    aesthetic: &str,
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check global mapping → uses global data
    for aes_name in &aesthetics_to_check {
        if let Some(AestheticValue::Column { name, .. }) = global_mappings.get(aes_name) {
            results.push((naming::GLOBAL_DATA_KEY.to_string(), name.clone()));
        }
    }

    // Check each layer's mapping → uses layer data or global data
    for (i, layer) in layers.iter().enumerate() {
        // Determine which data source this layer uses
        let data_key = if layer.source.is_some() || layer.filter.is_some() {
            naming::layer_key(i)
        } else {
            naming::GLOBAL_DATA_KEY.to_string()
        };

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                results.push((data_key.clone(), name.clone()));
            }
        }
    }

    results
}

/// Apply oob transformation to a single numeric column in a DataFrame.
fn apply_oob_to_column_numeric(
    df: &DataFrame,
    col_name: &str,
    range_min: f64,
    range_max: f64,
    oob_mode: &str,
) -> Result<DataFrame> {
    use polars::prelude::*;

    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    // Try to cast column to f64 for comparison
    let series = col.as_materialized_series();
    let f64_col = series.cast(&DataType::Float64).map_err(|_| {
        GgsqlError::ValidationError(format!(
            "Cannot apply oob to non-numeric column '{}'",
            col_name
        ))
    })?;

    let f64_ca = f64_col.f64().map_err(|_| {
        GgsqlError::ValidationError(format!(
            "Cannot apply oob to non-numeric column '{}'",
            col_name
        ))
    })?;

    match oob_mode {
        OOB_CENSOR => {
            // Filter out rows where values are outside [range_min, range_max]
            let mask: BooleanChunked = f64_ca
                .into_iter()
                .map(|opt| opt.is_none_or(|v| v >= range_min && v <= range_max))
                .collect();

            df.filter(&mask).map_err(|e| {
                GgsqlError::InternalError(format!("Failed to filter DataFrame: {}", e))
            })
        }
        OOB_SQUISH => {
            // Clamp values to [range_min, range_max]
            let clamped: Float64Chunked = f64_ca
                .into_iter()
                .map(|opt| opt.map(|v| v.clamp(range_min, range_max)))
                .collect();

            // Replace column with clamped values, maintaining original name
            let clamped_series = clamped.into_series().with_name(col_name.into());

            df.clone()
                .with_column(clamped_series)
                .map(|df| df.clone())
                .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))
        }
        _ => Ok(df.clone()),
    }
}

/// Apply oob transformation to a single discrete/categorical column in a DataFrame.
fn apply_oob_to_column_discrete(
    df: &DataFrame,
    col_name: &str,
    allowed_values: &std::collections::HashSet<String>,
    oob_mode: &str,
) -> Result<DataFrame> {
    use polars::prelude::*;

    // For discrete columns, only censor makes sense (squish is validated out earlier)
    if oob_mode != OOB_CENSOR {
        return Ok(df.clone());
    }

    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let series = col.as_materialized_series();

    // Build mask: keep rows where value is null OR value is in allowed set
    let mask: BooleanChunked = (0..series.len())
        .map(|i| {
            match series.get(i) {
                Ok(val) => {
                    // Null values are kept (similar to numeric behavior)
                    if val.is_null() {
                        return true;
                    }
                    // Convert value to string and check membership
                    let s = val.to_string();
                    // Remove quotes if present (polars adds quotes around strings)
                    let clean = s.trim_matches('"');
                    allowed_values.contains(clean)
                }
                Err(_) => true, // Keep on error
            }
        })
        .collect();

    df.filter(&mask)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to filter DataFrame: {}", e)))
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
    use crate::naming;
    use crate::plot::{ArrayElement, SqlExpression};
    use crate::Geom;
    use polars::prelude::DataType;

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_global_only() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";

        let result = prepare_data(query, &reader).unwrap();

        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
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

        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
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
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Global should have all 4 rows
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 4);

        // Layer 0 should have only 2 rows (filtered to category = 'A')
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
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
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer 0 should have only 2 rows (y > 200)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
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

        // CTE table names now include session UUID
        assert!(result.starts_with("SELECT * FROM __ggsql_cte_sales_"));
        assert!(result.ends_with("__ WHERE year = 2024"));
        assert!(result.contains(naming::session_id()));
    }

    #[test]
    fn test_transform_cte_references_multiple() {
        let sql = "SELECT * FROM sales JOIN targets ON sales.date = targets.date";
        let mut cte_names = HashSet::new();
        cte_names.insert("sales".to_string());
        cte_names.insert("targets".to_string());

        let result = transform_cte_references(sql, &cte_names);

        // CTE table names now include session UUID
        assert!(result.contains("FROM __ggsql_cte_sales_"));
        assert!(result.contains("JOIN __ggsql_cte_targets_"));
        assert!(result.contains(naming::session_id()));
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
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("sales".to_string()));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should use temp table name with session UUID
        let query = result.unwrap().unwrap();
        assert!(query.starts_with("SELECT * FROM __ggsql_cte_sales_"));
        assert!(query.ends_with("__"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_with_cte_and_filter() {
        let mut materialized = HashSet::new();
        materialized.insert("sales".to_string());
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("sales".to_string()));
        layer.filter = Some(SqlExpression::new("year = 2024"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should use temp table name with session UUID and filter
        let query = result.unwrap().unwrap();
        assert!(query.contains("__ggsql_cte_sales_"));
        assert!(query.ends_with(" WHERE year = 2024"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_without_cte() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should use table name directly
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM some_table".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_table_with_filter() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.filter = Some(SqlExpression::new("value > 100"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM some_table WHERE value > 100".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::FilePath("data/sales.csv".to_string()));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        // File paths should be wrapped in single quotes
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM 'data/sales.csv'".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path_with_filter() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::FilePath("data.parquet".to_string()));
        layer.filter = Some(SqlExpression::new("x > 10"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM 'data.parquet' WHERE x > 10".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_none_source_with_filter() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.filter = Some(SqlExpression::new("category = 'A'"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            true,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should query global table with session UUID and filter
        let query = result.unwrap().unwrap();
        assert!(query.starts_with("SELECT * FROM __ggsql_global_"));
        assert!(query.ends_with("__ WHERE category = 'A'"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_none_source_no_filter() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            true,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should return None - layer uses __global__ directly
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_build_layer_query_filter_without_global_errors() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.filter = Some(SqlExpression::new("x > 10"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            2,
            None,
            &[],
            &mock_execute,
        );

        // Should return validation error
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Layer 3")); // layer_idx 2 -> Layer 3 in message
        assert!(err.contains("FILTER"));
    }

    #[test]
    fn test_build_layer_query_with_order_by() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.order_by = Some(SqlExpression::new("date ASC"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM some_table ORDER BY date ASC".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_with_filter_and_order_by() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.filter = Some(SqlExpression::new("year = 2024"));
        layer.order_by = Some(SqlExpression::new("date DESC, value ASC"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &[],
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some(
                "SELECT * FROM some_table WHERE year = 2024 ORDER BY date DESC, value ASC"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_build_layer_query_none_source_with_order_by() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.order_by = Some(SqlExpression::new("x ASC"));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            true,
            0,
            None,
            &[],
            &mock_execute,
        );

        // Should query global table with session UUID and order_by
        let query = result.unwrap().unwrap();
        assert!(query.starts_with("SELECT * FROM __ggsql_global_"));
        assert!(query.ends_with("__ ORDER BY x ASC"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_with_constants() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();
        let constants = vec![
            (
                "color".to_string(),
                LiteralValue::String("value".to_string()),
            ),
            (
                "size".to_string(),
                LiteralValue::String("value2".to_string()),
            ),
        ];

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            false,
            0,
            None,
            &constants,
            &mock_execute,
        );

        // Should inject constants as columns
        let query = result.unwrap().unwrap();
        assert!(query.contains("SELECT *"));
        assert!(query.contains("'value' AS __ggsql_const_color__"));
        assert!(query.contains("'value2' AS __ggsql_const_size__"));
        assert!(query.contains("FROM some_table"));
    }

    #[test]
    fn test_build_layer_query_constants_on_global() {
        let materialized = HashSet::new();
        let empty_schema: Schema = Vec::new();
        let constants = vec![(
            "fill".to_string(),
            LiteralValue::String("value".to_string()),
        )];

        // No source but has constants - should use global table with session UUID
        let mut layer = Layer::new(Geom::point());

        let result = build_layer_query(
            &mut layer,
            &empty_schema,
            &materialized,
            true,
            0,
            None,
            &constants,
            &mock_execute,
        );

        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM __ggsql_global_"));
        assert!(query.contains(naming::session_id()));
        assert!(query.contains("'value' AS __ggsql_const_fill__"));
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
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(1)));

        // Global should have 2 rows (from sales)
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 2);

        // Layer 1 should have 2 rows (from targets CTE)
        let layer_df = result.data.get(&naming::layer_key(1)).unwrap();
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
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 4);

        // Layer 1 should have 2 rows (filtered to category = 'A')
        let layer_df = result.data.get(&naming::layer_key(1)).unwrap();
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
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(&naming::layer_key(1)));
        assert!(result.data.contains_key(&naming::layer_key(2)));

        // Each layer should have 2 rows
        assert_eq!(result.data.get(&naming::layer_key(0)).unwrap().height(), 2);
        assert_eq!(result.data.get(&naming::layer_key(1)).unwrap().height(), 2);
        assert_eq!(result.data.get(&naming::layer_key(2)).unwrap().height(), 2);
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
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
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
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
        // Layers without their own FROM use global directly (no separate entry)
        assert!(!result.data.contains_key(&naming::layer_key(0)));
        assert!(!result.data.contains_key(&naming::layer_key(1)));

        // Global should have 3 rows
        assert_eq!(
            result.data.get(naming::GLOBAL_DATA_KEY).unwrap().height(),
            3
        );
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
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
        // Each layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(&naming::layer_key(1)));

        assert_eq!(result.data.get(&naming::layer_key(0)).unwrap().height(), 2);
        assert_eq!(result.data.get(&naming::layer_key(1)).unwrap().height(), 2);
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
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(1)));
        // Layer 0 has no entry (uses global directly)
        assert!(!result.data.contains_key(&naming::layer_key(0)));

        assert_eq!(
            result.data.get(naming::GLOBAL_DATA_KEY).unwrap().height(),
            2
        );
        assert_eq!(result.data.get(&naming::layer_key(1)).unwrap().height(), 2);
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
        assert_eq!(
            result.data.get(naming::GLOBAL_DATA_KEY).unwrap().height(),
            5
        );

        // Layer 1 should have 2 rows (cat='A' AND active=true)
        assert_eq!(result.data.get(&naming::layer_key(1)).unwrap().height(), 2);
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
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have stat bin and count columns
        let col_names: Vec<&str> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&naming::stat_column("bin").as_str()));
        assert!(col_names.contains(&naming::stat_column("count").as_str()));

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
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 3 rows (3 unique categories: A, B, C)
        assert_eq!(layer_df.height(), 3);

        // Should have category (original x) and stat count columns
        let col_names: Vec<&str> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"category"));
        assert!(col_names.contains(&naming::stat_column("count").as_str()));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_uses_y_when_mapped() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories and values
        reader
            .connection()
            .execute(
                "CREATE TABLE bar_y_test AS SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)",
                duckdb::params![],
            )
            .unwrap();

        // Bar geom with x and y mapped - should NOT apply count stat (uses y values)
        let query = r#"
            SELECT * FROM bar_y_test
            VISUALISE
            DRAW bar MAPPING category AS x, value AS y
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should NOT have layer 0 data (no transformation needed, uses global)
        assert!(!result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        // Global should have original 3 rows
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
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
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have region column preserved for faceting
        let col_names: Vec<&str> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"region"));
        assert!(col_names.contains(&naming::stat_column("bin").as_str()));
        assert!(col_names.contains(&naming::stat_column("count").as_str()));
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
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have grp column preserved for grouping
        let col_names: Vec<&str> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(col_names.contains(&"grp"));
        assert!(col_names.contains(&"category"));
        assert!(col_names.contains(&naming::stat_column("count").as_str()));

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
        assert!(!result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        // Global should have original 3 rows
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_global_mapping_x_and_y() {
        // Test that bar charts with x and y in global VISUALISE mapping work correctly
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories and pre-aggregated values
        reader
            .connection()
            .execute(
                "CREATE TABLE sales AS SELECT * FROM (VALUES ('Electronics', 1000), ('Clothing', 800), ('Furniture', 600)) AS t(category, total)",
                duckdb::params![],
            )
            .unwrap();

        // Bar geom with x and y from global mapping - should NOT apply count stat (uses y values)
        let query = r#"
            SELECT * FROM sales
            VISUALISE category AS x, total AS y
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should NOT have layer 0 data (no transformation needed, y is mapped and exists)
        assert!(
            !result.data.contains_key(&naming::layer_key(0)),
            "Bar with y mapped should use global data directly"
        );
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        // Global should have original 3 rows
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 3);

        // Verify spec has x and y aesthetics merged into layer
        assert_eq!(result.specs.len(), 1);
        let layer = &result.specs[0].layers[0];
        assert!(
            layer.mappings.contains_key("x"),
            "Layer should have x from global mapping"
        );
        assert!(
            layer.mappings.contains_key("y"),
            "Layer should have y from global mapping"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_wildcard_uses_y_when_present() {
        // With the new smart stat logic, if wildcard expands y and y column exists,
        // bar uses existing y values (identity, no COUNT)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE wildcard_test AS SELECT * FROM (VALUES
                    ('A', 100), ('B', 200), ('C', 300)
                ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // VISUALISE * with bar chart - uses existing y values since y column exists
        let query = r#"
            SELECT * FROM wildcard_test
            VISUALISE *
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // With wildcard and y column present, bar uses identity (no layer 0 data)
        assert!(
            !result.data.contains_key(&naming::layer_key(0)),
            "Bar with wildcard + y column should use identity (no COUNT)"
        );
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        // Global should have original 3 rows
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_explicit_y_uses_data_directly() {
        // Bar geom uses existing y column directly when y is mapped and exists, no stat transform
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE bar_explicit AS SELECT * FROM (VALUES
                    ('A', 100), ('B', 200), ('C', 300)
                ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Explicit x, y mapping with bar geom - no COUNT transform (y exists)
        let query = r#"
            SELECT * FROM bar_explicit
            VISUALISE x, y
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should NOT have layer 0 data (no transformation, y is explicitly mapped and exists)
        assert!(
            !result.data.contains_key(&naming::layer_key(0)),
            "Bar with explicit y should use global data directly"
        );
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        // Global should have original 3 rows (no COUNT applied)
        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_wildcard_mapping_only_x_column() {
        // Wildcard with only x column - SHOULD apply COUNT stat transform
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE wildcard_x_only AS SELECT * FROM (VALUES
                    ('A'), ('B'), ('A'), ('C'), ('A'), ('B')
                ) AS t(x)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM wildcard_x_only
            VISUALISE *
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data (COUNT transformation applied)
        assert!(
            result.data.contains_key(&naming::layer_key(0)),
            "Bar without y should apply COUNT stat"
        );
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 3 rows (3 unique x values: A, B, C)
        assert_eq!(layer_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_aliased_columns_with_bar_geom() {
        // Test explicit mappings with SQL column aliases using bar geom
        // Bar geom uses existing y values directly when y is mapped and exists
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE sales_aliased AS SELECT * FROM (VALUES
                    ('Electronics', 1000), ('Clothing', 800), ('Furniture', 600)
                ) AS t(category, revenue)",
                duckdb::params![],
            )
            .unwrap();

        // Column aliases create columns named 'x' and 'y'
        // Bar geom uses them directly (no stat transform since y exists)
        let query = r#"
            SELECT category AS x, SUM(revenue) AS y
            FROM sales_aliased
            GROUP BY category
            VISUALISE x, y
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Bar geom with y mapped - no stat transform (y column exists)
        assert!(
            !result.data.contains_key(&naming::layer_key(0)),
            "Bar with explicit y should use global data directly"
        );
        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));

        let global_df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(global_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_weight_uses_sum() {
        // Bar with weight aesthetic should use SUM(weight) instead of COUNT(*)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE weight_test AS SELECT * FROM (VALUES
                    ('A', 10), ('A', 20), ('B', 30)
                ) AS t(category, amount)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM weight_test
            VISUALISE
            DRAW bar MAPPING category AS x, amount AS weight
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data (SUM transformation applied)
        assert!(
            result.data.contains_key(&naming::layer_key(0)),
            "Bar with weight should apply SUM stat"
        );
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 2 rows (2 unique categories: A, B)
        assert_eq!(layer_df.height(), 2);

        // Verify y values are sums: A=30 (10+20), B=30
        // SUM returns f64, but stat column is always named "count" for consistency
        let stat_count_col = naming::stat_column("count");
        let y_col = layer_df
            .column(&stat_count_col)
            .expect("stat count column should exist");
        let y_values: Vec<f64> = y_col
            .f64()
            .expect("stat count should be f64 (SUM result)")
            .into_iter()
            .flatten()
            .collect();

        // Sum of A should be 30, sum of B should be 30
        assert!(
            y_values.contains(&30.0),
            "Should have sum of 30 for category A"
        );
        assert!(
            y_values.contains(&30.0),
            "Should have sum of 30 for category B"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_without_weight_uses_count() {
        // Bar without weight aesthetic should use COUNT(*)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE count_test AS SELECT * FROM (VALUES
                    ('A', 10), ('A', 20), ('B', 30)
                ) AS t(category, amount)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM count_test
            VISUALISE
            DRAW bar MAPPING category AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data (COUNT transformation applied)
        assert!(
            result.data.contains_key(&naming::layer_key(0)),
            "Bar without weight should apply COUNT stat"
        );
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 2 rows (2 unique categories: A, B)
        assert_eq!(layer_df.height(), 2);

        // Verify y values are counts: A=2, B=1
        let stat_count_col = naming::stat_column("count");
        let y_col = layer_df
            .column(&stat_count_col)
            .expect("stat count column should exist");
        let y_values: Vec<i64> = y_col
            .i64()
            .expect("stat count should be i64")
            .into_iter()
            .flatten()
            .collect();

        assert!(
            y_values.contains(&2),
            "Should have count of 2 for category A"
        );
        assert!(
            y_values.contains(&1),
            "Should have count of 1 for category B"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_weight_from_wildcard_missing_column_falls_back_to_count() {
        // Wildcard mapping with no 'weight' column should fall back to COUNT
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE no_weight_col AS SELECT * FROM (VALUES
                    ('A'), ('A'), ('B')
                ) AS t(x)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM no_weight_col
            VISUALISE *
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data (COUNT transformation applied)
        assert!(
            result.data.contains_key(&naming::layer_key(0)),
            "Bar with wildcard (no weight column) should apply COUNT stat"
        );
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 2 rows (2 unique x values: A, B)
        assert_eq!(layer_df.height(), 2);

        // Verify y values are counts: A=2, B=1
        let stat_count_col = naming::stat_column("count");
        let y_col = layer_df
            .column(&stat_count_col)
            .expect("stat count column should exist");
        let y_values: Vec<i64> = y_col
            .i64()
            .expect("stat count should be i64")
            .into_iter()
            .flatten()
            .collect();

        assert!(y_values.contains(&2), "Should have count of 2 for A");
        assert!(y_values.contains(&1), "Should have count of 1 for B");
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_explicit_weight_missing_column_errors() {
        // Explicitly mapping weight to non-existent column should error
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE no_weight_explicit AS SELECT * FROM (VALUES
                    ('A'), ('B')
                ) AS t(category)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM no_weight_explicit
            VISUALISE
            DRAW bar MAPPING category AS x, nonexistent AS weight
        "#;

        let result = prepare_data(query, &reader);
        assert!(
            result.is_err(),
            "Bar with explicit weight mapping to non-existent column should error"
        );

        if let Err(err) = result {
            let err_msg = format!("{}", err);
            assert!(
                err_msg.contains("weight") && err_msg.contains("nonexistent"),
                "Error should mention weight and the missing column name, got: {}",
                err_msg
            );
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_weight_literal_errors() {
        // Mapping a literal value to weight should error
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE literal_weight AS SELECT * FROM (VALUES
                    ('A'), ('B')
                ) AS t(category)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM literal_weight
            VISUALISE
            DRAW bar MAPPING category AS x, 5 AS weight
        "#;

        let result = prepare_data(query, &reader);
        assert!(result.is_err(), "Bar with literal weight should error");

        if let Err(err) = result {
            let err_msg = format!("{}", err);
            assert!(
                err_msg.contains("weight") && err_msg.contains("literal"),
                "Error should mention weight must be a column, not literal, got: {}",
                err_msg
            );
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_with_wildcard_uses_weight_when_present() {
        // Wildcard mapping with 'weight' column should use SUM(weight)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE wildcard_weight AS SELECT * FROM (VALUES
                    ('A', 10), ('A', 20), ('B', 30)
                ) AS t(x, weight)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM wildcard_weight
            VISUALISE *
            DRAW bar
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have layer 0 data (SUM transformation applied)
        assert!(
            result.data.contains_key(&naming::layer_key(0)),
            "Bar with wildcard + weight column should apply SUM stat"
        );
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 2 rows (2 unique x values: A, B)
        assert_eq!(layer_df.height(), 2);

        // Verify y values are sums: A=30, B=30
        // SUM returns f64, but stat column is always named "count" for consistency
        let stat_count_col = naming::stat_column("count");
        let y_col = layer_df
            .column(&stat_count_col)
            .expect("stat count column should exist");
        let y_values: Vec<f64> = y_col
            .f64()
            .expect("stat count should be f64 (SUM result)")
            .into_iter()
            .flatten()
            .collect();

        assert!(y_values.contains(&30.0), "Should have sum values");
    }

    // =============================================================================
    // Scale Resolution Tests
    // =============================================================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_numeric_to_continuous() {
        // Test that numeric columns infer Continuous scale type
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 1.0 as x, 2.0 as y FROM (VALUES (1))
            VISUALISE x, y
            DRAW point
            SCALE x FROM [0, 100]
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let spec = &result.specs[0];

        // Find the x scale
        let x_scale = spec.find_scale("x").expect("x scale should exist");

        // Should be inferred as Continuous from numeric column
        assert_eq!(
            x_scale.scale_type,
            Some(ScaleType::continuous()),
            "Numeric column should infer Continuous scale type"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_date_to_temporal() {
        // Test that date columns infer Date scale type
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT '2024-01-01'::DATE as date, 100 as value
            VISUALISE date AS x, value AS y
            DRAW line
            SCALE x
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let spec = &result.specs[0];

        // Find the x scale
        let x_scale = spec.find_scale("x").expect("x scale should exist");

        // Should be inferred as Date from Date column
        assert_eq!(
            x_scale.scale_type,
            Some(ScaleType::date()),
            "Date column should infer Date scale type"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_string_to_discrete() {
        // Test that string columns infer Discrete scale type
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 'A' as category, 100 as value FROM (VALUES (1))
            VISUALISE category AS x, value AS y
            DRAW bar
            SCALE x FROM ['A', 'B', 'C']
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let spec = &result.specs[0];

        // Find the x scale
        let x_scale = spec.find_scale("x").expect("x scale should exist");

        // Should be inferred as Discrete from String column
        assert_eq!(
            x_scale.scale_type,
            Some(ScaleType::discrete()),
            "String column should infer Discrete scale type"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_explicit_type_preserved() {
        // Test that explicit scale types are not overwritten
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 'A' as category, 100 as value FROM (VALUES (1))
            VISUALISE category AS x, value AS y
            DRAW bar
            SCALE CONTINUOUS x
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let spec = &result.specs[0];

        // Find the x scale
        let x_scale = spec.find_scale("x").expect("x scale should exist");

        // Should preserve explicit Continuous type even though column is string
        assert_eq!(
            x_scale.scale_type,
            Some(ScaleType::continuous()),
            "Explicit scale type should be preserved"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_from_aesthetic_family() {
        // Test that scales can infer type from aesthetic family members (ymin, ymax -> y)
        // Ribbon requires x, ymin, ymax - we test that SCALE y infers type from ymin/ymax columns
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 1.0 as date, 0.0 as ymin, 10.0 as ymax FROM (VALUES (1))
            VISUALISE
            DRAW ribbon MAPPING date AS x, ymin AS ymin, ymax AS ymax
            SCALE y FROM [0, 20]
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let spec = &result.specs[0];

        // Find the y scale
        let y_scale = spec.find_scale("y").expect("y scale should exist");

        // Should infer Continuous from ymin/ymax columns (family members of y)
        assert_eq!(
            y_scale.scale_type,
            Some(ScaleType::continuous()),
            "Scale should infer type from aesthetic family members (ymin/ymax -> y)"
        );
    }

    #[test]
    fn test_get_aesthetic_family() {
        // Test primary aesthetics include all family members
        let x_family = get_aesthetic_family("x");
        assert!(x_family.contains(&"x"));
        assert!(x_family.contains(&"xmin"));
        assert!(x_family.contains(&"xmax"));
        assert!(x_family.contains(&"x2"));
        assert!(x_family.contains(&"xend"));

        let y_family = get_aesthetic_family("y");
        assert!(y_family.contains(&"y"));
        assert!(y_family.contains(&"ymin"));
        assert!(y_family.contains(&"ymax"));
        assert!(y_family.contains(&"y2"));
        assert!(y_family.contains(&"yend"));

        // Test non-family aesthetics return just themselves
        let color_family = get_aesthetic_family("color");
        assert_eq!(color_family, vec!["color"]);

        // Test variant aesthetics return just themselves
        let xmin_family = get_aesthetic_family("xmin");
        assert_eq!(xmin_family, vec!["xmin"]);
    }

    #[test]
    fn test_scale_type_infer() {
        // Test numeric types -> Continuous
        assert_eq!(ScaleType::infer(&DataType::Int32), ScaleType::continuous());
        assert_eq!(ScaleType::infer(&DataType::Int64), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Float64),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::UInt16), ScaleType::continuous());

        // Test temporal types
        assert_eq!(ScaleType::infer(&DataType::Date), ScaleType::date());
        assert_eq!(
            ScaleType::infer(&DataType::Datetime(
                polars::prelude::TimeUnit::Microseconds,
                None
            )),
            ScaleType::datetime()
        );
        assert_eq!(ScaleType::infer(&DataType::Time), ScaleType::time());

        // Test discrete types
        assert_eq!(ScaleType::infer(&DataType::String), ScaleType::discrete());
        assert_eq!(ScaleType::infer(&DataType::Boolean), ScaleType::discrete());
    }

    // =========================================================================
    // Input Range Inference Tests (using ScaleType::resolve_input_range)
    // =========================================================================

    #[test]
    fn test_infer_input_range_numeric() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Create numeric column
        let column: Column = Series::new("x".into(), &[1.0f64, 5.0, 10.0, 3.0]).into();

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let range = ScaleType::continuous()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.len(), 2);

        // Should be [min, max] = [1.0, 10.0]
        assert_eq!(range[0], ArrayElement::Number(1.0));
        assert_eq!(range[1], ArrayElement::Number(10.0));
    }

    #[test]
    fn test_infer_input_range_numeric_integer() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Create integer column (should cast to f64)
        let column: Column = Series::new("y".into(), &[10i32, 20, 30, 5]).into();

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let range = ScaleType::continuous()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();

        assert_eq!(range[0], ArrayElement::Number(5.0));
        assert_eq!(range[1], ArrayElement::Number(30.0));
    }

    #[test]
    fn test_infer_input_range_numeric_multiple_columns() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Two columns with different ranges - should combine
        let column1: Column = Series::new("x".into(), &[1.0f64, 5.0]).into();
        let column2: Column = Series::new("xmax".into(), &[10.0f64, 20.0]).into();

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let range = ScaleType::continuous()
            .resolve_input_range(None, &[&column1, &column2], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();

        // Should combine: min=1.0 (from column1), max=20.0 (from column2)
        assert_eq!(range[0], ArrayElement::Number(1.0));
        assert_eq!(range[1], ArrayElement::Number(20.0));
    }

    #[test]
    fn test_infer_input_range_date() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Create date column: 2024-01-15, 2024-03-20, 2024-02-01
        // Days since epoch (1970-01-01):
        // 2024-01-15 = 19737 days
        // 2024-02-01 = 19754 days
        // 2024-03-20 = 19802 days
        let column: Column = Series::new("date".into(), &[19737i32, 19802, 19754])
            .cast(&DataType::Date)
            .unwrap()
            .into();

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let range = ScaleType::date()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.len(), 2);

        assert_eq!(range[0], ArrayElement::String("2024-01-15".into()));
        assert_eq!(range[1], ArrayElement::String("2024-03-20".into()));
    }

    #[test]
    fn test_infer_input_range_discrete() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Create string column with duplicates
        let column: Column = Series::new("category".into(), &["B", "A", "C", "A", "B"]).into();
        let props = StdHashMap::new();

        let range = ScaleType::discrete()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();

        // Should be sorted unique values: ["A", "B", "C"]
        assert_eq!(range.len(), 3);
        assert_eq!(range[0], ArrayElement::String("A".into()));
        assert_eq!(range[1], ArrayElement::String("B".into()));
        assert_eq!(range[2], ArrayElement::String("C".into()));
    }

    #[test]
    fn test_infer_input_range_discrete_with_nulls() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // Create string column with null values
        let column: Column =
            Series::new("category".into(), &[Some("B"), None, Some("A"), Some("B")]).into();
        let props = StdHashMap::new();

        let range = ScaleType::discrete()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();

        // Nulls should be excluded, result should be ["A", "B"]
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], ArrayElement::String("A".into()));
        assert_eq!(range[1], ArrayElement::String("B".into()));
    }

    #[test]
    fn test_resolve_scales_infers_input_range() {
        use polars::prelude::*;

        // Create a Plot with a scale that needs range inference
        let mut spec = Plot::new();
        spec.global_mappings
            .insert("x", AestheticValue::standard_column("value"));

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("x");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        spec.layers.push(Layer::new(Geom::point()));

        // Create data with numeric values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

        // Resolve scales
        resolve_scales(&mut spec, &data_map).unwrap();

        // Check that both scale_type and input_range were inferred
        let scale = &spec.scales[0];
        assert_eq!(scale.scale_type, Some(ScaleType::continuous()));
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        assert_eq!(range.len(), 2);
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 1.0);
                assert_eq!(*max, 10.0);
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_preserves_explicit_input_range() {
        use polars::prelude::*;

        // Create a Plot with a scale that already has a range
        let mut spec = Plot::new();
        spec.global_mappings
            .insert("x", AestheticValue::standard_column("value"));

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        spec.layers.push(Layer::new(Geom::point()));

        // Create data with different values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

        // Resolve scales
        resolve_scales(&mut spec, &data_map).unwrap();

        // Check that explicit range was preserved (not overwritten with [1, 10])
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 0.0); // Original explicit value
                assert_eq!(*max, 100.0); // Original explicit value
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_from_aesthetic_family_input_range() {
        use polars::prelude::*;

        // Create a Plot where "y" scale should get range from ymin and ymax columns
        let mut spec = Plot::new();
        spec.global_mappings
            .insert("ymin", AestheticValue::standard_column("low"));
        spec.global_mappings
            .insert("ymax", AestheticValue::standard_column("high"));

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("y");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        spec.layers.push(Layer::new(Geom::errorbar()));

        // Create data where ymin/ymax columns have different ranges
        let df = df! {
            "low" => &[5.0f64, 10.0, 15.0],
            "high" => &[20.0f64, 25.0, 30.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

        // Resolve scales
        resolve_scales(&mut spec, &data_map).unwrap();

        // Check that range was inferred from both ymin and ymax columns
        let scale = &spec.scales[0];
        assert!(scale.input_range.is_some());

        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                // min should be 5.0 (from low column), max should be 30.0 (from high column)
                assert_eq!(*min, 5.0);
                assert_eq!(*max, 30.0);
            }
            _ => panic!("Expected Number elements"),
        }
    }

    // =========================================================================
    // Partial Input Range Inference Tests (null placeholders)
    // These tests now use ScaleType::resolve_input_range which handles the merging internally
    // =========================================================================

    #[test]
    fn test_resolve_input_range_merge_min_null() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // FROM [null, 100] with data [1, 10] → [1, 100] (with expansion disabled)
        let column: Column = Series::new("x".into(), &[1.0f64, 5.0, 10.0]).into();
        let user_range = vec![ArrayElement::Null, ArrayElement::Number(100.0)];

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user_range), &[&column], &props)
            .unwrap();
        assert!(result.is_some());
        let range = result.unwrap();

        assert_eq!(range[0], ArrayElement::Number(1.0)); // From data
        assert_eq!(range[1], ArrayElement::Number(100.0)); // From explicit
    }

    #[test]
    fn test_resolve_input_range_merge_max_null() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // FROM [0, null] with data [1, 10] → [0, 10] (with expansion disabled)
        let column: Column = Series::new("x".into(), &[1.0f64, 5.0, 10.0]).into();
        let user_range = vec![ArrayElement::Number(0.0), ArrayElement::Null];

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user_range), &[&column], &props)
            .unwrap();
        assert!(result.is_some());
        let range = result.unwrap();

        assert_eq!(range[0], ArrayElement::Number(0.0)); // From explicit
        assert_eq!(range[1], ArrayElement::Number(10.0)); // From data
    }

    #[test]
    fn test_resolve_input_range_merge_both_null() {
        use polars::prelude::*;
        use std::collections::HashMap as StdHashMap;

        // FROM [null, null] with data [1, 10] → [1, 10] (with expansion disabled)
        let column: Column = Series::new("x".into(), &[1.0f64, 5.0, 10.0]).into();
        let user_range = vec![ArrayElement::Null, ArrayElement::Null];

        // Disable expansion for predictable test values
        let mut props = StdHashMap::new();
        props.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );

        let result = ScaleType::continuous()
            .resolve_input_range(Some(&user_range), &[&column], &props)
            .unwrap();
        assert!(result.is_some());
        let range = result.unwrap();

        assert_eq!(range[0], ArrayElement::Number(1.0)); // From data
        assert_eq!(range[1], ArrayElement::Number(10.0)); // From data
    }

    #[test]
    fn test_resolve_scales_partial_input_range_explicit_min_null_max() {
        use polars::prelude::*;

        // Create a Plot with a scale that has [0, null] (explicit min, infer max)
        let mut spec = Plot::new();
        spec.global_mappings
            .insert("x", AestheticValue::standard_column("value"));

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Null]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        spec.layers.push(Layer::new(Geom::point()));

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

        // Resolve scales
        resolve_scales(&mut spec, &data_map).unwrap();

        // Check that range is [0, 10] (explicit min, inferred max)
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 0.0); // Explicit value
                assert_eq!(*max, 10.0); // Inferred from data
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[test]
    fn test_resolve_scales_partial_input_range_null_min_explicit_max() {
        use polars::prelude::*;

        // Create a Plot with a scale that has [null, 100] (infer min, explicit max)
        let mut spec = Plot::new();
        spec.global_mappings
            .insert("x", AestheticValue::standard_column("value"));

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Null, ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        spec.layers.push(Layer::new(Geom::point()));

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

        // Resolve scales
        resolve_scales(&mut spec, &data_map).unwrap();

        // Check that range is [1, 100] (inferred min, explicit max)
        let scale = &spec.scales[0];
        let range = scale.input_range.as_ref().unwrap();
        match (&range[0], &range[1]) {
            (ArrayElement::Number(min), ArrayElement::Number(max)) => {
                assert_eq!(*min, 1.0); // Inferred from data
                assert_eq!(*max, 100.0); // Explicit value
            }
            _ => panic!("Expected Number elements"),
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_expansion_of_color_aesthetic() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Colors as standard columns
        let query = r#"
            VISUALISE bill_len AS x, bill_dep AS y FROM ggsql:penguins
            DRAW point MAPPING species AS color, island AS fill
        "#;

        let result = prepare_data(query, &reader).unwrap();

        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        assert!(aes.contains_key("stroke"));
        assert!(aes.contains_key("fill"));

        let stroke = aes.get("stroke").unwrap().column_name().unwrap();
        assert_eq!(stroke, "species");

        let fill = aes.get("fill").unwrap().column_name().unwrap();
        assert_eq!(fill, "island");

        // Colors as global constant
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, 'blue' AS color FROM ggsql:penguins
          DRAW point MAPPING island AS stroke
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(stroke.column_name().unwrap(), "island");

        let fill = aes.get("fill").unwrap();
        assert_eq!(fill.column_name().unwrap(), "__ggsql_const_color_0__");

        // Colors as layer constant
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, island AS fill FROM ggsql:penguins
          DRAW point MAPPING 'blue' AS color
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(stroke.column_name().unwrap(), "__ggsql_const_color_0__");

        let fill = aes.get("fill").unwrap();
        assert_eq!(fill.column_name().unwrap(), "island");
    }

    // =========================================================================
    // OOB Tests
    // =========================================================================

    #[test]
    fn test_apply_oob_censor_filters_data() {
        use polars::prelude::*;

        // Create DataFrame with values 1..10
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0f64, 5.0, 10.0, 15.0, 20.0]).into(),
            Series::new("y".into(), vec![10.0f64, 20.0, 30.0, 40.0, 50.0]).into(),
        ])
        .unwrap();

        // Apply censor to keep only values in [5, 15]
        let result = apply_oob_to_column_numeric(&df, "x", 5.0, 15.0, "censor").unwrap();

        // Should have 3 rows: 5, 10, 15
        assert_eq!(result.height(), 3);

        // Check values
        let x_col = result.column("x").unwrap();
        let x_series = x_col.as_materialized_series().f64().unwrap();
        let values: Vec<f64> = x_series.into_iter().flatten().collect();
        assert_eq!(values, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_apply_oob_squish_clamps_data() {
        use polars::prelude::*;

        // Create DataFrame with values 1..10
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0f64, 5.0, 10.0, 15.0, 20.0]).into(),
            Series::new("y".into(), vec![10.0f64, 20.0, 30.0, 40.0, 50.0]).into(),
        ])
        .unwrap();

        // Apply squish to clamp values to [5, 15]
        let result = apply_oob_to_column_numeric(&df, "x", 5.0, 15.0, "squish").unwrap();

        // Should still have 5 rows
        assert_eq!(result.height(), 5);

        // Check clamped values: [1→5, 5, 10, 15, 20→15]
        let x_col = result.column("x").unwrap();
        let x_series = x_col.as_materialized_series().f64().unwrap();
        let values: Vec<f64> = x_series.into_iter().flatten().collect();
        assert_eq!(values, vec![5.0, 5.0, 10.0, 15.0, 15.0]);
    }

    #[test]
    fn test_apply_oob_keep_preserves_data() {
        use polars::prelude::*;

        // Create DataFrame with values 1..10
        let df = DataFrame::new(vec![Series::new(
            "x".into(),
            vec![1.0f64, 5.0, 10.0, 15.0, 20.0],
        )
        .into()])
        .unwrap();

        // Apply keep - should not modify data
        let result = apply_oob_to_column_numeric(&df, "x", 5.0, 15.0, "keep").unwrap();

        // Should still have 5 rows with original values
        assert_eq!(result.height(), 5);
        let x_col = result.column("x").unwrap();
        let x_series = x_col.as_materialized_series().f64().unwrap();
        let values: Vec<f64> = x_series.into_iter().flatten().collect();
        assert_eq!(values, vec![1.0, 5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_apply_oob_censor_preserves_null_values() {
        use polars::prelude::*;

        // Create DataFrame with some null values
        let df = DataFrame::new(vec![Series::new(
            "x".into(),
            vec![Some(1.0f64), None, Some(10.0), Some(20.0)],
        )
        .into()])
        .unwrap();

        // Apply censor to keep only values in [5, 15]
        let result = apply_oob_to_column_numeric(&df, "x", 5.0, 15.0, "censor").unwrap();

        // Should have 2 rows: null (preserved) and 10
        assert_eq!(result.height(), 2);
    }

    #[test]
    fn test_apply_oob_discrete_censor_filters_data() {
        use polars::prelude::*;

        // Create DataFrame with categorical values
        let df = DataFrame::new(vec![
            Series::new("category".into(), vec!["A", "B", "C", "D", "E"]).into(),
            Series::new("value".into(), vec![1, 2, 3, 4, 5]).into(),
        ])
        .unwrap();

        // Only allow A, B, C
        let allowed: std::collections::HashSet<String> =
            ["A", "B", "C"].iter().map(|s| s.to_string()).collect();

        let result = apply_oob_to_column_discrete(&df, "category", &allowed, "censor").unwrap();

        // Should have 3 rows: A, B, C
        assert_eq!(result.height(), 3);

        // Check values
        let cat_col = result.column("category").unwrap();
        let values: Vec<String> = (0..cat_col.len())
            .map(|i| {
                cat_col
                    .as_materialized_series()
                    .get(i)
                    .unwrap()
                    .to_string()
                    .trim_matches('"')
                    .to_string()
            })
            .collect();
        assert_eq!(values, vec!["A", "B", "C"]);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_censor_integration() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with values spanning outside the range
        // Note: expand => 0 disables range expansion for predictable filtering
        let query = r#"
            SELECT * FROM (VALUES
                (1, 5),
                (2, 15),
                (3, 25)
            ) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            SCALE y FROM [0, 20] SETTING oob => 'censor', expand => 0
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Check that the data was filtered (only 2 rows should remain)
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_squish_integration() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with values spanning outside the range
        // Note: expand => 0 disables range expansion for predictable clamping
        let query = r#"
            SELECT * FROM (VALUES
                (1, 5),
                (2, 15),
                (3, 25)
            ) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            SCALE y FROM [0, 20] SETTING oob => 'squish', expand => 0
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Check that the data was clamped (all 3 rows should remain, y clamped to [0, 20])
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 3);

        // Check clamped values
        let y_col = df.column("y").unwrap();
        let y_series = y_col
            .as_materialized_series()
            .cast(&polars::prelude::DataType::Float64)
            .unwrap();
        let y_f64 = y_series.f64().unwrap();
        let values: Vec<f64> = y_f64.into_iter().flatten().collect();
        // Values should be: 5 (within range), 15 (within range), 20 (clamped from 25)
        assert_eq!(values, vec![5.0, 15.0, 20.0]);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_default_keep_for_positional() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with values spanning outside the range
        // Default for positional (y) is 'keep', so data should not be modified
        let query = r#"
            SELECT * FROM (VALUES
                (1, 5),
                (2, 15),
                (3, 25)
            ) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            SCALE y FROM [0, 20]
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // All rows should be present (default oob='keep' for positional)
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_discrete_censor_integration() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with categories, some outside allowed range
        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 'A'),
                (2, 20, 'B'),
                (3, 30, 'C'),
                (4, 40, 'D')
            ) AS t(x, y, category)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            SCALE DISCRETE color FROM ['A', 'B'] SETTING oob => 'censor'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Only rows with 'A' and 'B' should remain (C and D are out of range)
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 2);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_discrete_default_censor_for_non_positional() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Non-positional aesthetics (like color) should default to censor
        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 'A'),
                (2, 20, 'B'),
                (3, 30, 'C'),
                (4, 40, 'D')
            ) AS t(x, y, category)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, category AS color
            SCALE DISCRETE color FROM ['A', 'B']
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Default oob='censor' for non-positional, so C and D should be filtered
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 2);
    }
}
