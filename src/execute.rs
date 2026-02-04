//! Query execution module for ggsql
//!
//! Provides shared execution logic for building data maps from queries,
//! handling both global SQL and layer-specific data sources.

use crate::naming;
use crate::plot::{AestheticValue, ColumnInfo, Layer, LiteralValue, Schema, StatResult};
use crate::{parser, DataFrame, DataSource, Facet, GgsqlError, Plot, Result};
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

/// Result of building a layer query
///
/// Contains information about the queries executed for a layer,
/// distinguishing between base filter queries and stat transform queries.
#[derive(Debug, Default)]
pub struct LayerQueryResult {
    /// The final query to execute (if any)
    /// None means layer uses global data directly
    pub query: Option<String>,
    /// The base query before stat transform (filter/source only)
    /// None if layer uses global data directly without filter
    pub layer_sql: Option<String>,
    /// The stat transform query (if a stat transform was applied)
    /// None if no stat transform was needed
    pub stat_sql: Option<String>,
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
/// - `Ok(LayerQueryResult)` with information about queries executed
/// - `Err(...)` - validation error (e.g., filter without global data)
///
/// Note: This function takes `&mut Layer` because stat transforms may add new aesthetic mappings
/// (e.g., mapping y to `__ggsql_stat__count` for histogram or bar count).
#[allow(clippy::too_many_arguments)]
fn build_layer_query<F>(
    layer: &mut Layer,
    schema: &Schema,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
    layer_idx: usize,
    facet: Option<&Facet>,
    constants: &[(String, LiteralValue)],
    execute_query: &F,
) -> Result<LayerQueryResult>
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
                return Ok(LayerQueryResult::default());
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

    // Save the base query (with filter) before stat transform
    let base_query = query.clone();

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
            let mut final_query = transformed_query.clone();
            if let Some(o) = order_by {
                final_query = format!("{} ORDER BY {}", final_query, o);
            }
            Ok(LayerQueryResult {
                query: Some(final_query),
                layer_sql: Some(base_query),
                stat_sql: Some(transformed_query),
            })
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
                Ok(LayerQueryResult::default())
            } else {
                // Layer has filter, order_by, or constants - still need the query
                let mut final_query = query;
                if let Some(o) = order_by {
                    final_query = format!("{} ORDER BY {}", final_query, o);
                }
                Ok(LayerQueryResult {
                    query: Some(final_query.clone()),
                    layer_sql: Some(final_query),
                    stat_sql: None,
                })
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
    /// Parsed and resolved visualization specification
    pub spec: Plot,
    /// The main SQL query that was executed
    pub sql: String,
    /// The raw VISUALISE portion text
    pub visual: String,
    /// Per-layer filter/source queries (None = uses global data directly)
    pub layer_sql: Vec<Option<String>>,
    /// Per-layer stat transform queries (None = no stat transform)
    pub stat_sql: Vec<Option<String>>,
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

    // TODO: Support multiple VISUALISE statements in future
    if specs.len() > 1 {
        return Err(GgsqlError::ValidationError(
            "Multiple VISUALISE statements are not yet supported. Please use a single VISUALISE statement.".to_string(),
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

    // Track layer and stat queries for introspection
    let mut layer_sql_vec: Vec<Option<String>> = Vec::new();
    let mut stat_sql_vec: Vec<Option<String>> = Vec::new();

    for (idx, layer) in specs[0].layers.iter_mut().enumerate() {
        // For layers using global data without filter, constants are already in global data
        // (injected with layer-indexed names). For other layers, extract constants for injection.
        let constants = if layer.source.is_none() && layer.filter.is_none() {
            vec![] // Constants already in global data
        } else {
            extract_constants(layer)
        };

        // Get mutable reference to layer for stat transform to update aesthetics
        let query_result = build_layer_query(
            layer,
            &layer_schemas[idx],
            &materialized_ctes,
            has_global,
            idx,
            facet.as_ref(),
            &constants,
            &execute_query,
        )?;

        // Store query information for introspection
        layer_sql_vec.push(query_result.layer_sql);
        stat_sql_vec.push(query_result.stat_sql);

        // Execute the query if one was generated
        if let Some(layer_query) = query_result.query {
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

    let mut spec = specs.into_iter().next().unwrap();

    // Post-process spec: replace literals with column references and compute labels
    // Replace literal aesthetic values with column references to synthetic constant columns
    replace_literals_with_columns(&mut spec);
    // Compute aesthetic labels (uses first non-constant column, respects user-specified labels)
    spec.compute_aesthetic_labels();
    // Divide 'color' over 'stroke' and 'fill'. This needs to happens after
    // literals have associated columns.
    split_color_aesthetic(&mut spec.layers);

    Ok(PreparedData {
        data: data_map,
        spec,
        sql: sql_part,
        visual: viz_part,
        layer_sql: layer_sql_vec,
        stat_sql: stat_sql_vec,
    })
}

/// Build data map from a query using DuckDB reader
///
/// Convenience wrapper around `prepare_data_with_executor` for direct DuckDB reader usage.
#[cfg(feature = "duckdb")]
pub fn prepare_data(query: &str, reader: &DuckDBReader) -> Result<PreparedData> {
    prepare_data_with_executor(query, |sql| reader.execute_sql(sql))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::naming;
    use crate::plot::SqlExpression;
    use crate::Geom;

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_global_only() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";

        let result = prepare_data(query, &reader).unwrap();

        assert!(result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert_eq!(result.spec.layers.len(), 1);
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
        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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
        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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
        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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

        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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
        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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

        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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
        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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

        // Should return empty result - layer uses __global__ directly
        let query_result = result.unwrap();
        assert!(query_result.query.is_none());
        assert!(query_result.layer_sql.is_none());
        assert!(query_result.stat_sql.is_none());
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

        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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

        let query_result = result.unwrap();
        assert_eq!(
            query_result.query,
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
        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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
        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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

        let query_result = result.unwrap();
        let query = query_result.query.unwrap();
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
        assert_eq!(result.spec.layers.len(), 1);
        let layer = &result.spec.layers[0];
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

        let aes = &result.spec.layers[0].mappings.aesthetics;

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
        let aes = &result.spec.layers[0].mappings.aesthetics;

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
        let aes = &result.spec.layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(stroke.column_name().unwrap(), "__ggsql_const_color_0__");

        let fill = aes.get("fill").unwrap();
        assert_eq!(fill.column_name().unwrap(), "island");
    }
}
