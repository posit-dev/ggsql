//! Query execution module for ggsql
//!
//! Provides shared execution logic for building data maps from queries,
//! handling both global SQL and layer-specific data sources.

use crate::naming;
use crate::plot::layer::geom::{GeomAesthetics, AESTHETIC_FAMILIES};
use crate::plot::{
    AestheticValue, ArrayElement, ArrayElementType, ColumnInfo, Layer, LiteralValue,
    ParameterValue, Scale, ScaleType, ScaleTypeKind, Schema, StatResult,
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

/// Build SQL query to compute min and max for all columns
///
/// Generates a query that returns two rows:
/// - Row 0: MIN of each column
/// - Row 1: MAX of each column
fn build_minmax_query(source_query: &str, column_names: &[&str]) -> String {
    let min_exprs: Vec<String> = column_names
        .iter()
        .map(|name| format!("MIN(\"{}\") AS \"{}\"", name, name))
        .collect();

    let max_exprs: Vec<String> = column_names
        .iter()
        .map(|name| format!("MAX(\"{}\") AS \"{}\"", name, name))
        .collect();

    format!(
        "WITH __ggsql_source__ AS ({}) SELECT {} FROM __ggsql_source__ UNION ALL SELECT {} FROM __ggsql_source__",
        source_query,
        min_exprs.join(", "),
        max_exprs.join(", ")
    )
}

/// Extract a value from a DataFrame at a given column and row index
///
/// Converts Polars values to ArrayElement for storage in ColumnInfo.
fn extract_series_value(
    df: &DataFrame,
    column: &str,
    row: usize,
) -> Option<crate::plot::ArrayElement> {
    use crate::plot::ArrayElement;
    use polars::prelude::DataType;

    let col = df.column(column).ok()?;
    let series = col.as_materialized_series();

    if row >= series.len() {
        return None;
    }

    match series.dtype() {
        DataType::Int8 => series
            .i8()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int16 => series
            .i16()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int32 => series
            .i32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Int64 => series
            .i64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt8 => series
            .u8()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt16 => series
            .u16()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt32 => series
            .u32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::UInt64 => series
            .u64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Float32 => series
            .f32()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|v| ArrayElement::Number(v as f64)),
        DataType::Float64 => series
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(ArrayElement::Number),
        DataType::Boolean => series
            .bool()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(ArrayElement::Boolean),
        DataType::String => series
            .str()
            .ok()
            .and_then(|ca| ca.get(row))
            .map(|s| ArrayElement::String(s.to_string())),
        DataType::Date => {
            // Return numeric days since epoch (for range computation)
            series
                .date()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|days| ArrayElement::Number(days as f64))
        }
        DataType::Datetime(_, _) => {
            // Return numeric microseconds since epoch (for range computation)
            series
                .datetime()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|us| ArrayElement::Number(us as f64))
        }
        DataType::Time => {
            // Return numeric nanoseconds since midnight (for range computation)
            series
                .time()
                .ok()
                .and_then(|ca| ca.physical().get(row))
                .map(|ns| ArrayElement::Number(ns as f64))
        }
        _ => None,
    }
}

// =============================================================================
// Type-only Schema Extraction (for early resolution)
// =============================================================================

/// Simple type info tuple: (name, dtype, is_discrete)
pub type TypeInfo = (String, polars::prelude::DataType, bool);

/// Fetch only column types (no min/max) from a query.
///
/// Uses LIMIT 0 to get schema without reading data.
/// Returns `(name, dtype, is_discrete)` tuples for each column.
///
/// This is the first phase of the split schema extraction approach:
/// 1. fetch_schema_types() - get dtypes only (before casting)
/// 2. Apply casting to queries
/// 3. complete_schema_ranges() - get min/max from cast queries
fn fetch_schema_types<F>(query: &str, execute_query: &F) -> Result<Vec<TypeInfo>>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    use polars::prelude::DataType;

    let schema_query = format!(
        "SELECT * FROM ({}) AS {} LIMIT 0",
        query,
        naming::SCHEMA_ALIAS
    );
    let schema_df = execute_query(&schema_query)?;

    let type_info: Vec<TypeInfo> = schema_df
        .get_columns()
        .iter()
        .map(|col| {
            let dtype = col.dtype().clone();
            let is_discrete =
                matches!(dtype, DataType::String | DataType::Boolean) || dtype.is_categorical();
            (col.name().to_string(), is_discrete, dtype)
        })
        .map(|(name, is_discrete, dtype)| (name, dtype, is_discrete))
        .collect();

    Ok(type_info)
}

/// Complete schema with min/max ranges from a (possibly cast) query.
///
/// Takes pre-computed type info and extracts min/max values.
/// Called after casting is applied to queries.
fn complete_schema_ranges<F>(
    query: &str,
    type_info: &[TypeInfo],
    execute_query: &F,
) -> Result<Schema>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    if type_info.is_empty() {
        return Ok(Vec::new());
    }

    // Build and execute min/max query
    let column_names: Vec<&str> = type_info.iter().map(|(n, _, _)| n.as_str()).collect();
    let minmax_query = build_minmax_query(query, &column_names);
    let range_df = execute_query(&minmax_query)?;

    // Extract min (row 0) and max (row 1) for each column
    let schema = type_info
        .iter()
        .map(|(name, dtype, is_discrete)| {
            let min = extract_series_value(&range_df, name, 0);
            let max = extract_series_value(&range_df, name, 1);
            ColumnInfo {
                name: name.clone(),
                dtype: dtype.clone(),
                is_discrete: *is_discrete,
                min,
                max,
            }
        })
        .collect();

    Ok(schema)
}

/// Convert type info to schema (without min/max).
///
/// Used when we need a Schema but don't have min/max yet.
fn type_info_to_schema(type_info: &[TypeInfo]) -> Schema {
    type_info
        .iter()
        .map(|(name, dtype, is_discrete)| ColumnInfo {
            name: name.clone(),
            dtype: dtype.clone(),
            is_discrete: *is_discrete,
            min: None,
            max: None,
        })
        .collect()
}

/// Add type info for literal (constant) mappings to layer type info.
///
/// When a layer has literal mappings like `'blue' AS fill`, we need the type info
/// for these columns in the schema. Instead of re-querying the database, we can
/// derive the types directly from the AST.
///
/// This is called after global mappings are merged and color is split, so all
/// literal mappings are already in place.
fn add_literal_columns_to_type_info(layers: &[Layer], layer_type_info: &mut [Vec<TypeInfo>]) {
    use polars::prelude::DataType;

    for (layer, type_info) in layers.iter().zip(layer_type_info.iter_mut()) {
        for (aesthetic, value) in &layer.mappings.aesthetics {
            if let AestheticValue::Literal(lit) = value {
                let dtype = match lit {
                    LiteralValue::String(_) => DataType::String,
                    LiteralValue::Number(_) => DataType::Float64,
                    LiteralValue::Boolean(_) => DataType::Boolean,
                };
                let is_discrete = matches!(lit, LiteralValue::String(_) | LiteralValue::Boolean(_));
                let col_name = naming::aesthetic_column(aesthetic);

                // Only add if not already present
                if !type_info.iter().any(|(name, _, _)| name == &col_name) {
                    type_info.push((col_name, dtype, is_discrete));
                }
            }
        }
    }
}

// =============================================================================
// Type Requirements and Casting
// =============================================================================

use crate::plot::{CastTargetType, SqlTypeNames};

/// Describes a column that needs type casting.
#[derive(Debug, Clone)]
pub struct TypeRequirement {
    /// Column name to cast
    pub column: String,
    /// Target type for casting
    pub target_type: CastTargetType,
    /// SQL type name (e.g., "DATE", "DOUBLE", "VARCHAR")
    pub sql_type_name: String,
}

/// Determine which columns need casting based on scale requirements.
///
/// For each layer, collects columns that need casting to match the scale's
/// target type (determined by type coercion across all columns for that aesthetic).
///
/// # Arguments
///
/// * `spec` - The Plot specification with scales
/// * `layer_type_info` - Type info for each layer
/// * `type_names` - SQL type names for the database backend
///
/// # Returns
///
/// Vec of TypeRequirements for each layer.
fn determine_type_requirements(
    spec: &Plot,
    layer_type_info: &[Vec<TypeInfo>],
    type_names: &SqlTypeNames,
) -> Vec<Vec<TypeRequirement>> {
    use crate::plot::scale::coerce_dtypes;
    let mut layer_requirements: Vec<Vec<TypeRequirement>> = Vec::new();

    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        let mut requirements: Vec<TypeRequirement> = Vec::new();
        let type_info = &layer_type_info[layer_idx];

        // Build a map of column name to dtype for quick lookup
        let column_dtypes: HashMap<&str, &polars::prelude::DataType> = type_info
            .iter()
            .map(|(name, dtype, _)| (name.as_str(), dtype))
            .collect();

        // For each aesthetic mapped in this layer, check if casting is needed
        for (aesthetic, value) in &layer.mappings.aesthetics {
            let col_name = match value.column_name() {
                Some(name) => name,
                None => continue, // Skip literals
            };

            // Skip synthetic columns
            if naming::is_synthetic_column(col_name) {
                continue;
            }

            let col_dtype = match column_dtypes.get(col_name) {
                Some(dtype) => *dtype,
                None => continue, // Column not in schema
            };

            // Find the scale for this aesthetic
            let scale = match spec.scales.iter().find(|s| s.aesthetic == *aesthetic) {
                Some(s) => s,
                None => continue, // No scale for this aesthetic
            };

            // Get the scale type
            let scale_type = match &scale.scale_type {
                Some(st) => st,
                None => continue, // Scale type not yet resolved
            };

            // Collect all dtypes for this aesthetic across all layers
            let all_dtypes: Vec<polars::prelude::DataType> = layer_type_info
                .iter()
                .zip(spec.layers.iter())
                .filter_map(|(info, l)| {
                    l.mappings
                        .get(aesthetic)
                        .and_then(|v| v.column_name())
                        .and_then(|name| info.iter().find(|(n, _, _)| n == name))
                        .map(|(_, dtype, _)| dtype.clone())
                })
                .collect();

            // Determine target dtype through coercion
            let target_dtype = match coerce_dtypes(&all_dtypes) {
                Ok(dt) => dt,
                Err(_) => continue, // Skip if coercion fails
            };

            // Check if this specific column needs casting
            if let Some(cast_target) = scale_type.required_cast_type(col_dtype, &target_dtype) {
                if let Some(sql_type) = type_names.for_target(cast_target) {
                    // Don't add duplicate requirements for same column
                    if !requirements.iter().any(|r| r.column == col_name) {
                        requirements.push(TypeRequirement {
                            column: col_name.to_string(),
                            target_type: cast_target,
                            sql_type_name: sql_type.to_string(),
                        });
                    }
                }
            }

            // Check if Integer transform requires casting (float -> integer)
            use crate::plot::scale::TransformKind;
            use crate::plot::CastTargetType;
            if let Some(ref transform) = scale.transform {
                if transform.transform_kind() == TransformKind::Integer {
                    // Integer transform: cast non-integer numeric types to integer
                    let needs_int_cast = match col_dtype {
                        polars::prelude::DataType::Float32 | polars::prelude::DataType::Float64 => {
                            true
                        }
                        // Integer types don't need casting
                        polars::prelude::DataType::Int8
                        | polars::prelude::DataType::Int16
                        | polars::prelude::DataType::Int32
                        | polars::prelude::DataType::Int64
                        | polars::prelude::DataType::UInt8
                        | polars::prelude::DataType::UInt16
                        | polars::prelude::DataType::UInt32
                        | polars::prelude::DataType::UInt64 => false,
                        // Other types: no integer casting
                        _ => false,
                    };

                    if needs_int_cast {
                        if let Some(sql_type) = type_names.for_target(CastTargetType::Integer) {
                            // Don't add duplicate requirements for same column
                            if !requirements.iter().any(|r| r.column == col_name) {
                                requirements.push(TypeRequirement {
                                    column: col_name.to_string(),
                                    target_type: CastTargetType::Integer,
                                    sql_type_name: sql_type.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        layer_requirements.push(requirements);
    }

    layer_requirements
}

/// Update type info with post-cast dtypes.
///
/// After determining casting requirements, updates the type info
/// to reflect the target dtypes (so subsequent schema extraction
/// and scale resolution see the correct types).
fn update_type_info_for_casting(type_info: &mut [TypeInfo], requirements: &[TypeRequirement]) {
    use polars::prelude::{DataType, TimeUnit};

    for req in requirements {
        if let Some(entry) = type_info
            .iter_mut()
            .find(|(name, _, _)| name == &req.column)
        {
            entry.1 = match req.target_type {
                CastTargetType::Number => DataType::Float64,
                CastTargetType::Integer => DataType::Int64,
                CastTargetType::Date => DataType::Date,
                CastTargetType::DateTime => DataType::Datetime(TimeUnit::Microseconds, None),
                CastTargetType::Time => DataType::Time,
                CastTargetType::String => DataType::String,
                CastTargetType::Boolean => DataType::Boolean,
            };
            // Update is_discrete flag based on new type
            entry.2 = matches!(entry.1, DataType::String | DataType::Boolean);
        }
    }
}

/// Determine the data source table name for a layer.
///
/// Returns the table/CTE name to query from:
/// - Layer with explicit source (CTE, table, file) → that source name
/// - Layer using global data → global table name
fn determine_layer_source(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
) -> String {
    match &layer.source {
        Some(DataSource::Identifier(name)) => {
            if materialized_ctes.contains(name) {
                naming::cte_table(name)
            } else {
                name.clone()
            }
        }
        Some(DataSource::FilePath(path)) => {
            format!("'{}'", path)
        }
        None => {
            // Layer uses global data - caller must ensure has_global is true
            debug_assert!(has_global, "Layer has no source and no global data");
            naming::global_table()
        }
    }
}

/// Build the source query for a layer.
///
/// Returns `SELECT * FROM source` where source is either:
/// - The layer's explicit source (table, CTE, file)
/// - The global table if layer has no explicit source
///
/// Note: This is distinct from `build_layer_base_query()` which builds a full
/// SELECT with aesthetic column renames and type casts.
fn layer_source_query(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
) -> String {
    let source = determine_layer_source(layer, materialized_ctes, has_global);
    format!("SELECT * FROM {}", source)
}

/// Build the SELECT list for a layer query with aesthetic-renamed columns and casting.
///
/// This function builds SELECT expressions that:
/// 1. Rename source columns to prefixed aesthetic names
/// 2. Apply type casts based on scale requirements
///
/// # Arguments
///
/// * `layer` - The layer configuration with aesthetic mappings
/// * `type_requirements` - Columns that need type casting
///
/// # Returns
///
/// A vector of SQL SELECT expressions starting with `*` followed by aesthetic columns:
/// - `*` (preserves all original columns)
/// - `CAST("Date" AS DATE) AS "__ggsql_aes_x__"` (cast + rename)
/// - `"Temp" AS "__ggsql_aes_y__"` (rename only, no cast needed)
/// - `'red' AS "__ggsql_aes_color__"` (literal value as aesthetic column)
///
/// The prefix `__ggsql_aes_` avoids conflicts with source columns that might
/// have names matching aesthetics (e.g., a column named "x" or "color").
///
/// Note: Facet variables are preserved automatically via `SELECT *`.
fn build_layer_select_list(layer: &Layer, type_requirements: &[TypeRequirement]) -> Vec<String> {
    let mut select_exprs = Vec::new();

    // Start with * to preserve all original columns
    // This ensures facet variables, partition_by columns, and any other
    // columns are available for downstream processing (stat transforms, etc.)
    select_exprs.push("*".to_string());

    // Build a map of column -> cast requirement for quick lookup
    let cast_map: std::collections::HashMap<&str, &TypeRequirement> = type_requirements
        .iter()
        .map(|r| (r.column.as_str(), r))
        .collect();

    // Add aesthetic-mapped columns with prefixed names (and casts where needed)
    for (aesthetic, value) in &layer.mappings.aesthetics {
        let aes_col_name = naming::aesthetic_column(aesthetic);
        let select_expr = match value {
            AestheticValue::Column { name, .. } => {
                // Check if this column needs casting
                if let Some(req) = cast_map.get(name.as_str()) {
                    // Cast and rename to prefixed aesthetic name
                    format!(
                        "CAST(\"{}\" AS {}) AS \"{}\"",
                        name, req.sql_type_name, aes_col_name
                    )
                } else {
                    // Just rename to prefixed aesthetic name
                    format!("\"{}\" AS \"{}\"", name, aes_col_name)
                }
            }
            AestheticValue::Literal(lit) => {
                // Literals become columns with prefixed aesthetic name
                format!("{} AS \"{}\"", literal_to_sql(lit), aes_col_name)
            }
        };

        select_exprs.push(select_expr);
    }

    select_exprs
}

/// Update layer mappings to use prefixed aesthetic column names.
///
/// After building a layer query that creates aesthetic columns with prefixed names,
/// the layer's mappings need to be updated to point to these prefixed column names.
///
/// This function converts:
/// - `AestheticValue::Column { name: "Date", ... }` → `AestheticValue::Column { name: "__ggsql_aes_x__", ... }`
/// - `AestheticValue::Literal(...)` → `AestheticValue::Column { name: "__ggsql_aes_color__", ... }`
///
/// Note: The final rename from prefixed names to clean aesthetic names (e.g., "x")
/// happens in Polars after query execution, before the data goes to the writer.
fn update_mappings_for_aesthetic_columns(layer: &mut Layer) {
    for (aesthetic, value) in layer.mappings.aesthetics.iter_mut() {
        let aes_col_name = naming::aesthetic_column(aesthetic);
        match value {
            AestheticValue::Column {
                name,
                original_name,
                ..
            } => {
                // Preserve the original column name for labels before overwriting
                if original_name.is_none() {
                    *original_name = Some(name.clone());
                }
                // Column is now named with the prefixed aesthetic name
                *name = aes_col_name;
            }
            AestheticValue::Literal(_) => {
                // Literals are also columns with prefixed aesthetic name
                // Note: literals don't have an original_name to preserve
                *value = AestheticValue::standard_column(aes_col_name);
            }
        }
    }
}

/// Build a schema with prefixed aesthetic column names from the original schema.
///
/// For each aesthetic mapped to a column, looks up the original column's type
/// in the schema and adds it with the prefixed aesthetic name (e.g., `__ggsql_aes_x__`).
///
/// This schema is used by stat transforms to look up column types using the
/// prefixed names that appear in the query after `build_layer_select_list`.
fn build_aesthetic_schema(layer: &Layer, schema: &Schema) -> Schema {
    use polars::prelude::DataType;

    let mut aesthetic_schema: Schema = Vec::new();

    for (aesthetic, value) in &layer.mappings.aesthetics {
        let aes_col_name = naming::aesthetic_column(aesthetic);
        match value {
            AestheticValue::Column { name, .. } => {
                // The schema already has aesthetic-prefixed column names from build_layer_base_query,
                // so we look up by aesthetic name, not the original column name.
                // Fall back to original name for backwards compatibility with older schemas.
                let col_info = schema
                    .iter()
                    .find(|c| c.name == aes_col_name)
                    .or_else(|| schema.iter().find(|c| c.name == *name));

                if let Some(original_col) = col_info {
                    aesthetic_schema.push(ColumnInfo {
                        name: aes_col_name,
                        dtype: original_col.dtype.clone(),
                        is_discrete: original_col.is_discrete,
                        min: original_col.min.clone(),
                        max: original_col.max.clone(),
                    });
                } else {
                    // Column not in schema - add with Unknown type
                    aesthetic_schema.push(ColumnInfo {
                        name: aes_col_name,
                        dtype: DataType::Unknown(Default::default()),
                        is_discrete: false,
                        min: None,
                        max: None,
                    });
                }
            }
            AestheticValue::Literal(lit) => {
                // Literals become columns with appropriate type
                let dtype = match lit {
                    LiteralValue::String(_) => DataType::String,
                    LiteralValue::Number(_) => DataType::Float64,
                    LiteralValue::Boolean(_) => DataType::Boolean,
                };
                aesthetic_schema.push(ColumnInfo {
                    name: aes_col_name,
                    dtype,
                    is_discrete: matches!(lit, LiteralValue::String(_) | LiteralValue::Boolean(_)),
                    min: None,
                    max: None,
                });
            }
        }
    }

    // Add facet variables and partition_by columns with their original types
    for col in &layer.partition_by {
        if !aesthetic_schema.iter().any(|c| c.name == *col) {
            if let Some(original_col) = schema.iter().find(|c| c.name == *col) {
                aesthetic_schema.push(original_col.clone());
            }
        }
    }

    aesthetic_schema
}

/// Rename columns in DataFrame after query execution.
///
/// This function performs two types of renames:
/// 1. Prefixed aesthetic columns to clean names: `__ggsql_aes_x__` → `x`
/// 2. Stat columns to aesthetic names via remappings: `__ggsql_stat_count` → `y`
///
/// This keeps SQL queries using safe prefixed names while producing clean
/// output for the Vega-Lite writer.
/// Apply remappings to rename stat columns to their target aesthetic's prefixed name.
///
/// After stat transforms, columns like `__ggsql_stat_count` need to be renamed
/// to the target aesthetic's prefixed name (e.g., `__ggsql_aes_y__`).
///
/// Note: Prefixed aesthetic names persist through the entire pipeline.
/// We do NOT rename `__ggsql_aes_x__` back to `x`.
fn apply_remappings_post_query(df: DataFrame, layer: &Layer) -> Result<DataFrame> {
    let mut df = df;

    // Apply remappings: stat columns → prefixed aesthetic names
    // e.g., __ggsql_stat_count → __ggsql_aes_y__
    // Remappings structure: HashMap<target_aesthetic, AestheticValue pointing to stat column>
    for (target_aesthetic, stat_col_value) in &layer.remappings.aesthetics {
        if let Some(stat_col_name) = stat_col_value.column_name() {
            // Check if this stat column exists in the DataFrame
            if df.column(stat_col_name).is_ok() {
                let target_col_name = naming::aesthetic_column(target_aesthetic);
                df.rename(stat_col_name, target_col_name.into())
                    .map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to rename stat column '{}' to '{}': {}",
                            stat_col_name, target_aesthetic, e
                        ))
                    })?;
            }
        }
    }

    Ok(df)
}

/// Update layer mappings to use prefixed aesthetic names for remapped columns.
///
/// After remappings are applied (stat columns renamed to prefixed aesthetic names),
/// the layer mappings need to be updated so the writer uses the correct field names.
///
/// The original name is set to the stat name (e.g., "density", "count") so axis labels
/// show meaningful names instead of internal prefixed names.
fn update_mappings_for_remappings(layer: &mut Layer) {
    // For each remapping, add the target aesthetic to mappings pointing to the prefixed name
    for (target_aesthetic, stat_col_value) in &layer.remappings.aesthetics {
        let prefixed_name = naming::aesthetic_column(target_aesthetic);

        // Use the stat name from remappings as the original_name for labels
        // The stat_col_value contains the user-specified stat name (e.g., "density", "count")
        let original_name = stat_col_value.column_name().map(|s| s.to_string());

        let value = AestheticValue::Column {
            name: prefixed_name,
            original_name,
            is_dummy: false,
        };

        layer
            .mappings
            .aesthetics
            .insert(target_aesthetic.clone(), value);
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
/// discrete columns to the layer's partition_by. This ensures proper grouping
/// for all layers, not just stat geoms.
///
/// Discreteness is determined by:
/// 1. If the aesthetic has an explicit scale with a scale_type:
///    - ScaleTypeKind::Discrete or Binned → discrete (add to partition_by)
///    - ScaleTypeKind::Continuous → not discrete (skip)
///    - ScaleTypeKind::Identity → fall back to schema
/// 2. Otherwise, use schema's is_discrete flag (based on column data type)
///
/// Columns already in partition_by (from explicit PARTITION BY clause) are skipped.
/// Stat-consumed aesthetics (x for bar, x for histogram) are also skipped.
fn add_discrete_columns_to_partition_by(
    layers: &mut [Layer],
    layer_schemas: &[Schema],
    scales: &[Scale],
) {
    // Positional aesthetics should NOT be auto-added to grouping.
    // Stats that need to group by positional aesthetics (like bar/histogram)
    // already handle this themselves via stat_consumed_aesthetics().
    const POSITIONAL_AESTHETICS: &[&str] =
        &["x", "y", "xmin", "xmax", "ymin", "ymax", "xend", "yend"];

    // Build a map of aesthetic -> scale for quick lookup
    let scale_map: HashMap<&str, &Scale> =
        scales.iter().map(|s| (s.aesthetic.as_str(), s)).collect();

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

                // Determine if this aesthetic is discrete:
                // 1. Check if there's an explicit scale with a scale_type
                // 2. Fall back to schema's is_discrete
                //
                // Discrete and Binned scales produce categorical groupings.
                // Continuous scales don't group. Identity defers to column type.
                let primary_aesthetic = GeomAesthetics::primary_aesthetic(aesthetic);
                let is_discrete = if let Some(scale) = scale_map.get(primary_aesthetic) {
                    if let Some(ref scale_type) = scale.scale_type {
                        match scale_type.scale_type_kind() {
                            ScaleTypeKind::Discrete
                            | ScaleTypeKind::Binned
                            | ScaleTypeKind::Ordinal => true,
                            ScaleTypeKind::Continuous => false,
                            ScaleTypeKind::Identity => discrete_columns.contains(col),
                        }
                    } else {
                        // Scale exists but no explicit type - use schema
                        discrete_columns.contains(col)
                    }
                } else {
                    // No scale for this aesthetic - use schema
                    discrete_columns.contains(col)
                };

                // Skip if not discrete
                if !is_discrete {
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

// =============================================================================
// Pre-Stat Transform
// =============================================================================

/// Apply pre-stat transformations for scales that require data modification before stats.
///
/// Handles multiple scale types:
/// - **Binned**: Wraps columns with bin centers based on resolved breaks
/// - **Discrete/Ordinal**: Censors values outside explicit input_range (FROM clause)
/// - **Continuous**: Applies OOB handling (censor/squish) when input_range is explicit
///
/// This must happen BEFORE stat transforms so that data is transformed first.
/// For example, censoring species='Gentoo' before COUNT(*) ensures Gentoo isn't counted.
///
/// # Arguments
///
/// * `query` - The base query to transform
/// * `layer` - The layer configuration
/// * `schema` - The layer's schema (used for column dtype lookup)
/// * `scales` - All resolved scales
/// * `type_names` - SQL type names for the database backend
fn apply_pre_stat_transform(
    query: &str,
    layer: &Layer,
    schema: &Schema,
    scales: &[crate::plot::Scale],
    type_names: &SqlTypeNames,
) -> String {
    use polars::prelude::DataType;

    let mut transform_exprs: Vec<(String, String)> = vec![];
    let mut transformed_columns: HashSet<String> = HashSet::new();

    // Check layer mappings for aesthetics with scales that need pre-stat transformation
    // Handles both column mappings and literal mappings (which are injected as synthetic columns)
    for (aesthetic, value) in &layer.mappings.aesthetics {
        // The query has already renamed columns to aesthetic names via build_layer_base_query,
        // so we use the aesthetic column name for SQL generation and schema lookup.
        let aes_col_name = naming::aesthetic_column(aesthetic);

        // Skip if we already have a transform for this aesthetic column
        // (can happen when fill and stroke both map to the same column)
        if transformed_columns.contains(&aes_col_name) {
            continue;
        }

        // Skip if this aesthetic is not mapped to a column or literal
        if value.column_name().is_none() && !value.is_literal() {
            continue;
        }

        // Find column dtype from schema using aesthetic column name
        let col_dtype = schema
            .iter()
            .find(|c| c.name == aes_col_name)
            .map(|c| c.dtype.clone())
            .unwrap_or(DataType::String); // Default to String if not found

        // Find scale for this aesthetic
        if let Some(scale) = scales.iter().find(|s| s.aesthetic == *aesthetic) {
            if let Some(ref scale_type) = scale.scale_type {
                // Get pre-stat SQL transformation from scale type (if applicable)
                // Each scale type's pre_stat_transform_sql() returns None if not applicable
                if let Some(sql) =
                    scale_type.pre_stat_transform_sql(&aes_col_name, &col_dtype, scale, type_names)
                {
                    transformed_columns.insert(aes_col_name.clone());
                    transform_exprs.push((aes_col_name, sql));
                } else {
                }
            } else {
            }
        } else {
        }
    }

    if transform_exprs.is_empty() {
        return query.to_string();
    }

    // Build wrapper: SELECT {transformed_cols}, other_cols FROM ({query})
    // For each transformed column, use the SQL expression; for others, keep as-is
    let transformed_col_names: HashSet<&str> =
        transform_exprs.iter().map(|(c, _)| c.as_str()).collect();

    // Build column list: all columns, with transformed ones replaced by their expressions
    let col_exprs: Vec<String> = transform_exprs
        .iter()
        .map(|(col, sql)| format!("{} AS {}", sql, col))
        .collect();

    // Build the excluded columns list for the * expansion
    // We need to select *, but exclude the columns we're replacing
    if col_exprs.is_empty() {
        return query.to_string();
    }

    // Use EXCLUDE to remove the original columns, then add the transformed versions
    let exclude_clause = if transformed_col_names.len() == 1 {
        format!("EXCLUDE ({})", transformed_col_names.iter().next().unwrap())
    } else {
        format!(
            "EXCLUDE ({})",
            transformed_col_names
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let final_query = format!(
        "SELECT * {}, {} FROM ({})",
        exclude_clause,
        col_exprs.join(", "),
        query
    );
    final_query
}

/// Part 1: Build the initial layer query with SELECT, casts, filters, and aesthetic renames.
///
/// This function builds a query that:
/// 1. Applies filter (uses original column names - that's what users write)
/// 2. Renames columns to aesthetic names (e.g., "Date" AS "__ggsql_aes_x__")
/// 3. Applies type casts based on scale requirements
///
/// The resulting query can be used for:
/// - Schema completion (fetching min/max values)
/// - Scale input range resolution
///
/// Does NOT apply stat transforms or ORDER BY - those require completed schemas.
///
/// # Arguments
///
/// * `layer` - The layer configuration with aesthetic mappings
/// * `source_query` - The base query for the layer's data source
/// * `type_requirements` - Columns that need type casting
///
/// # Returns
///
/// The base query string with SELECT/casts/filters applied.
fn build_layer_base_query(
    layer: &Layer,
    source_query: &str,
    type_requirements: &[TypeRequirement],
) -> String {
    // Build SELECT list with aesthetic renames, casts
    let select_exprs = build_layer_select_list(layer, type_requirements);
    let select_clause = if select_exprs.is_empty() {
        "*".to_string()
    } else {
        select_exprs.join(", ")
    };

    // Build query with optional WHERE clause
    if let Some(ref f) = layer.filter {
        format!(
            "SELECT {} FROM ({}) WHERE {}",
            select_clause,
            source_query,
            f.as_str()
        )
    } else {
        format!("SELECT {} FROM ({})", select_clause, source_query)
    }
}

/// Part 2: Apply stat transforms and ORDER BY to a base query.
///
/// This function:
/// 1. Builds the aesthetic-named schema for stat transforms
/// 2. Updates layer mappings to use prefixed aesthetic names
/// 3. Applies pre-stat transforms (e.g., binning, discrete censoring)
/// 4. Builds group_by columns from partition_by and facet
/// 5. Applies statistical transformation
/// 6. Applies ORDER BY
///
/// Should be called AFTER schema completion and scale input range resolution,
/// since stat transforms may depend on resolved breaks.
///
/// # Arguments
///
/// * `layer` - The layer to transform (modified by stat transforms)
/// * `base_query` - The base query from build_layer_base_query
/// * `schema` - The layer's schema (with min/max from base_query)
/// * `facet` - Optional facet configuration (needed for group_by columns)
/// * `scales` - All resolved scales
/// * `type_names` - SQL type names for the database backend
/// * `execute_query` - Function to execute queries (needed for some stat transforms)
///
/// # Returns
///
/// The final query string with stat transforms and ORDER BY applied.
fn apply_layer_transforms<F>(
    layer: &mut Layer,
    base_query: &str,
    schema: &Schema,
    facet: Option<&Facet>,
    scales: &[crate::plot::Scale],
    type_names: &SqlTypeNames,
    execute_query: &F,
) -> Result<String>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    // Clone order_by early to avoid borrow conflicts
    let order_by = layer.order_by.clone();

    // Build the aesthetic-named schema for stat transforms
    let aesthetic_schema: Schema = build_aesthetic_schema(layer, schema);

    // Update mappings to use prefixed aesthetic names
    // This must happen BEFORE stat transforms so they use aesthetic names
    update_mappings_for_aesthetic_columns(layer);

    // Apply pre-stat transforms (e.g., binning, discrete censoring)
    // Uses aesthetic names since columns are now renamed and mappings updated
    let query = apply_pre_stat_transform(base_query, layer, &aesthetic_schema, scales, type_names);

    // Build group_by columns from partition_by and facet variables
    let mut group_by: Vec<String> = Vec::new();
    for col in &layer.partition_by {
        group_by.push(col.clone());
    }
    if let Some(f) = facet {
        for var in f.get_variables() {
            if !group_by.contains(&var) {
                group_by.push(var);
            }
        }
    }

    // Apply statistical transformation (uses aesthetic names)
    let stat_result = layer.geom.apply_stat_transform(
        &query,
        &aesthetic_schema,
        &layer.mappings,
        &group_by,
        &layer.parameters,
        execute_query,
    )?;

    let final_query = match stat_result {
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
            // When user maps a stat to an aesthetic, remove any default mapping to that aesthetic
            for (aesthetic, value) in &layer.remappings.aesthetics {
                if let Some(stat_name) = value.column_name() {
                    // Remove any existing mapping to this aesthetic (from defaults)
                    final_remappings.retain(|_, aes| aes != aesthetic);
                    // Add the user's mapping
                    final_remappings.insert(stat_name.to_string(), aesthetic.clone());
                }
            }

            // Capture original names from consumed aesthetics before removing them.
            // This allows stat-generated replacements to use the original column name for labels.
            // e.g., "revenue AS x" with histogram → x gets label "revenue" not "bin_start"
            let mut consumed_original_names: HashMap<String, String> = HashMap::new();
            for aes in &consumed_aesthetics {
                if let Some(value) = layer.mappings.get(aes) {
                    // Use label_name() to get the best available name for labels
                    if let Some(label) = value.label_name() {
                        consumed_original_names.insert(aes.clone(), label.to_string());
                    }
                }
            }

            // Remove consumed aesthetics - they were used as stat input, not visual output
            for aes in &consumed_aesthetics {
                layer.mappings.aesthetics.remove(aes);
            }

            // Apply stat_columns to layer aesthetics using the remappings
            for stat in &stat_columns {
                if let Some(aesthetic) = final_remappings.get(stat) {
                    let is_dummy = dummy_columns.contains(stat);
                    let prefixed_name = naming::aesthetic_column(aesthetic);

                    // Determine the original_name for labels:
                    // - If this aesthetic was consumed, use the original column name
                    // - Otherwise, use the stat name (e.g., "density", "count")
                    let original_name = consumed_original_names
                        .get(aesthetic)
                        .cloned()
                        .or_else(|| Some(stat.clone()));

                    let value = AestheticValue::Column {
                        name: prefixed_name,
                        original_name,
                        is_dummy,
                    };
                    layer.mappings.insert(aesthetic.clone(), value);
                }
            }

            // Wrap transformed query to rename stat columns to prefixed aesthetic names
            let stat_rename_exprs: Vec<String> = stat_columns
                .iter()
                .filter_map(|stat| {
                    final_remappings.get(stat).map(|aes| {
                        let stat_col = naming::stat_column(stat);
                        let prefixed_aes = naming::aesthetic_column(aes);
                        format!("\"{}\" AS \"{}\"", stat_col, prefixed_aes)
                    })
                })
                .collect();

            if stat_rename_exprs.is_empty() {
                transformed_query
            } else {
                let stat_col_names: Vec<String> = stat_columns
                    .iter()
                    .map(|s| naming::stat_column(s))
                    .collect();
                let exclude_clause = format!("EXCLUDE ({})", stat_col_names.join(", "));
                format!(
                    "SELECT * {}, {} FROM ({})",
                    exclude_clause,
                    stat_rename_exprs.join(", "),
                    transformed_query
                )
            }
        }
        StatResult::Identity => query,
    };

    // Apply ORDER BY
    let final_query = if let Some(ref o) = order_by {
        format!("{} ORDER BY {}", final_query, o.as_str())
    } else {
        final_query
    };

    Ok(final_query)
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
            // Note: "color"/"colour" are accepted even though not in supported,
            // because split_color_aesthetic will convert them to fill/stroke later
            for (aesthetic, value) in &spec.global_mappings.aesthetics {
                let is_color_alias = matches!(aesthetic.as_str(), "color" | "colour");
                if supported.contains(&aesthetic.as_str()) || is_color_alias {
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

// Let 'color' aesthetics fill defaults for the 'stroke' and 'fill' aesthetics.
// Also splits 'color' scale to 'fill' and 'stroke' scales.
// Removes 'color' from both mappings and scales after splitting to avoid
// non-deterministic behavior from HashMap iteration order.
fn split_color_aesthetic(spec: &mut Plot) {
    // 1. Split color SCALE to fill/stroke scales
    if let Some(color_scale_idx) = spec.scales.iter().position(|s| s.aesthetic == "color") {
        let color_scale = spec.scales[color_scale_idx].clone();

        // Add fill scale if not already present
        if !spec.scales.iter().any(|s| s.aesthetic == "fill") {
            let mut fill_scale = color_scale.clone();
            fill_scale.aesthetic = "fill".to_string();
            spec.scales.push(fill_scale);
        }

        // Add stroke scale if not already present
        if !spec.scales.iter().any(|s| s.aesthetic == "stroke") {
            let mut stroke_scale = color_scale.clone();
            stroke_scale.aesthetic = "stroke".to_string();
            spec.scales.push(stroke_scale);
        }

        // Remove the color scale
        spec.scales.remove(color_scale_idx);
    }

    // 2. Split color mapping to fill/stroke in layers, then remove color
    for layer in &mut spec.layers {
        if let Some(color_value) = layer.mappings.aesthetics.get("color").cloned() {
            let supported = layer.geom.aesthetics().supported;

            for &aes in &["stroke", "fill"] {
                if supported.contains(&aes) {
                    layer
                        .mappings
                        .aesthetics
                        .entry(aes.to_string())
                        .or_insert(color_value.clone());
                }
            }

            // Remove color after splitting
            layer.mappings.aesthetics.remove("color");
        }
    }

    // 3. Split color parameter (SETTING) to fill/stroke in layers
    for layer in &mut spec.layers {
        if let Some(color_value) = layer.parameters.get("color").cloned() {
            let supported = layer.geom.aesthetics().supported;

            for &aes in &["stroke", "fill"] {
                if supported.contains(&aes) {
                    layer
                        .parameters
                        .entry(aes.to_string())
                        .or_insert(color_value.clone());
                }
            }

            // Remove color after splitting
            layer.parameters.remove("color");
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
/// * `type_names` - SQL type names for the database backend
pub fn prepare_data_with_executor<F>(
    query: &str,
    execute_query: F,
    type_names: &SqlTypeNames,
) -> Result<PreparedData>
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

    // Execute global SQL if present
    // If there's a WITH clause, extract just the trailing SELECT and transform CTE references.
    // The global result is stored as a temp table so filtered layers can query it efficiently.
    // Track whether we actually create the temp table (depends on transform_global_sql succeeding)
    //
    // Note: Constants (literals in mappings) are no longer injected into the global table.
    // Each layer now builds its own query via build_layer_select_list which includes
    // literals as aesthetic-named columns (e.g., 'red' AS "color").
    let mut has_global_table = false;
    if !sql_part.trim().is_empty() {
        if let Some(transformed_sql) = transform_global_sql(&sql_part, &materialized_ctes) {
            // Create temp table for global result
            let create_global = format!(
                "CREATE OR REPLACE TEMP TABLE {} AS {}",
                naming::global_table(),
                transformed_sql
            );
            execute_query(&create_global)?;

            // NOTE: Don't read into data_map yet - defer until after casting is determined
            // The temp table exists and can be used for schema fetching
            has_global_table = true;
        }
    }

    // Validate all layers have a data source (explicit source or global data)
    for (idx, layer) in specs[0].layers.iter().enumerate() {
        if layer.source.is_none() && !has_global_table {
            return Err(GgsqlError::ValidationError(format!(
                "Layer {} has no data source. Either provide a SQL query before VISUALISE or use FROM in the layer.",
                idx + 1
            )));
        }
    }

    // Build source queries for each layer to fetch initial type info
    // Every layer now has its own source query (either explicit source or global table)
    let layer_source_queries: Vec<String> = specs[0]
        .layers
        .iter()
        .map(|layer| layer_source_query(layer, &materialized_ctes, has_global_table))
        .collect();

    // Get types for each layer from source queries (Phase 1: types only, no min/max yet)
    let mut layer_type_info: Vec<Vec<TypeInfo>> = Vec::new();
    for source_query in &layer_source_queries {
        let type_info = fetch_schema_types(source_query, &execute_query)?;
        layer_type_info.push(type_info);
    }

    // Initial schemas (types only, no min/max - will be completed after base queries)
    let mut layer_schemas: Vec<Schema> = layer_type_info
        .iter()
        .map(|ti| type_info_to_schema(ti))
        .collect();

    // Merge global mappings into layer aesthetics and expand wildcards
    // Smart wildcard expansion only creates mappings for columns that exist in schema
    merge_global_mappings_into_layers(&mut specs, &layer_schemas);

    // Split 'color' aesthetic to 'fill' and 'stroke' early in the pipeline
    // This must happen before validation so fill/stroke are validated (not color)
    for spec in &mut specs {
        split_color_aesthetic(spec);
    }

    // Add literal (constant) columns to type info programmatically
    // This avoids re-querying the database - we derive types from the AST
    add_literal_columns_to_type_info(&specs[0].layers, &mut layer_type_info);

    // Rebuild layer schemas with constant columns included
    layer_schemas = layer_type_info
        .iter()
        .map(|ti| type_info_to_schema(ti))
        .collect();

    // Validate all layers against their schemas
    // This must happen BEFORE build_layer_query because stat transforms remove consumed aesthetics
    validate(&specs[0].layers, &layer_schemas)?;

    // Create scales for all mapped aesthetics that don't have explicit SCALE clauses
    create_missing_scales(&mut specs[0]);

    // Resolve scale types and transforms early based on column dtypes
    resolve_scale_types_and_transforms(&mut specs[0], &layer_type_info)?;

    // Determine which columns need type casting
    let type_requirements = determine_type_requirements(&specs[0], &layer_type_info, type_names);

    // Update type info with post-cast dtypes
    // This ensures subsequent schema extraction and scale resolution see the correct types
    for (layer_idx, requirements) in type_requirements.iter().enumerate() {
        if layer_idx < layer_type_info.len() {
            update_type_info_for_casting(&mut layer_type_info[layer_idx], requirements);
        }
    }

    // Build layer base queries using build_layer_base_query()
    // These include: SELECT with aesthetic renames, casts from type_requirements, filters
    // Note: This is Part 1 of the split - base queries that can be used for schema completion
    let layer_base_queries: Vec<String> = specs[0]
        .layers
        .iter()
        .enumerate()
        .map(|(idx, layer)| {
            build_layer_base_query(layer, &layer_source_queries[idx], &type_requirements[idx])
        })
        .collect();

    // Clone facet for apply_layer_transforms
    let facet = specs[0].facet.clone();

    // Complete schemas with min/max from base queries (Phase 2: ranges from cast data)
    // Base queries include casting via build_layer_select_list, so min/max reflect cast types
    for (idx, base_query) in layer_base_queries.iter().enumerate() {
        layer_schemas[idx] =
            complete_schema_ranges(base_query, &layer_type_info[idx], &execute_query)?;
    }

    // Pre-resolve Binned scales using schema-derived context
    // This must happen before apply_layer_transforms so pre_stat_transform_sql has resolved breaks
    apply_pre_stat_resolve(&mut specs[0], &layer_schemas)?;

    // Add discrete mapped columns to partition_by for all layers
    let scales = specs[0].scales.clone();
    add_discrete_columns_to_partition_by(&mut specs[0].layers, &layer_schemas, &scales);

    // Clone scales for apply_layer_transforms
    let scales = specs[0].scales.clone();

    // Build final layer queries using apply_layer_transforms (Part 2 of the split)
    // This applies: pre-stat transforms, stat transforms, ORDER BY
    let mut layer_queries: Vec<String> = Vec::new();

    for (idx, layer) in specs[0].layers.iter_mut().enumerate() {
        // Validate weight aesthetic is a column, not a literal
        if let Some(weight_value) = layer.mappings.aesthetics.get("weight") {
            if weight_value.is_literal() {
                return Err(GgsqlError::ValidationError(
                    "Bar weight aesthetic must be a column, not a literal".to_string(),
                ));
            }
        }

        // Apply default parameter values (e.g., bins=30 for histogram)
        layer.apply_default_params();

        // Apply stat transforms and ORDER BY (Part 2)
        let layer_query = apply_layer_transforms(
            layer,
            &layer_base_queries[idx],
            &layer_schemas[idx],
            facet.as_ref(),
            &scales,
            type_names,
            &execute_query,
        )?;
        layer_queries.push(layer_query);
    }

    // Phase 2: Deduplicate and execute unique queries
    let mut query_to_result: HashMap<String, DataFrame> = HashMap::new();
    for (idx, query) in layer_queries.iter().enumerate() {
        if !query_to_result.contains_key(query) {
            let df = execute_query(query).map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "Failed to fetch data for layer {}: {}",
                    idx + 1,
                    e
                ))
            })?;
            query_to_result.insert(query.clone(), df);
        }
    }

    // Phase 3: Assign data to layers (clone only when needed)
    // Key by (query, serialized_remappings) to detect when layers can share data
    // Layers with identical query AND remappings share data via data_key
    let mut config_to_key: HashMap<(String, String), String> = HashMap::new();

    for (idx, query) in layer_queries.iter().enumerate() {
        let layer = &mut specs[0].layers[idx];
        let remappings_key = serde_json::to_string(&layer.remappings).unwrap_or_default();
        let config_key = (query.clone(), remappings_key);

        if let Some(existing_key) = config_to_key.get(&config_key) {
            // Same query AND same remappings - share data
            layer.data_key = Some(existing_key.clone());
        } else {
            // Need own data entry (either first occurrence or different remappings)
            let layer_key = naming::layer_key(idx);
            let df = query_to_result.get(query).unwrap().clone();
            data_map.insert(layer_key.clone(), df);
            config_to_key.insert(config_key, layer_key.clone());
            layer.data_key = Some(layer_key);
        }
    }

    // Phase 4: Apply remappings (rename stat columns to prefixed aesthetic names)
    // e.g., __ggsql_stat_count → __ggsql_aes_y__
    // Note: Prefixed aesthetic names persist through the entire pipeline
    // Track processed keys to avoid duplicate work on shared datasets
    let mut processed_keys: HashSet<String> = HashSet::new();
    for layer in specs[0].layers.iter_mut() {
        if let Some(ref key) = layer.data_key {
            if processed_keys.insert(key.clone()) {
                // First time seeing this data - process it
                if let Some(df) = data_map.remove(key) {
                    let df_with_remappings = apply_remappings_post_query(df, layer)?;
                    data_map.insert(key.clone(), df_with_remappings);
                }
            }
            // Update layer mappings for all layers (even if data shared)
            update_mappings_for_remappings(layer);
        }
    }

    // Validate we have some data (every layer should have its own data)
    if data_map.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No data sources found. Either provide a SQL query or use MAPPING FROM in layers."
                .to_string(),
        ));
    }

    // Create scales for aesthetics added by stat transforms (e.g., y from histogram)
    // This must happen after build_layer_query() which applies stat transforms
    // and modifies layer.mappings with new aesthetics like y → __ggsql_stat_count__
    for spec in &mut specs {
        create_missing_scales_post_stat(spec);
    }

    // Post-process specs: compute aesthetic labels
    // Note: Literal to column conversion is now handled by update_mappings_for_aesthetic_columns()
    // inside build_layer_query(), so replace_literals_with_columns() is no longer needed
    for spec in &mut specs {
        // Compute aesthetic labels (uses first non-constant column, respects user-specified labels)
        spec.compute_aesthetic_labels();
    }

    // Resolve scale types from data for scales without explicit types
    for spec in &mut specs {
        resolve_scales(spec, &mut data_map)?;
    }

    // Apply post-stat binning for Binned scales on remapped aesthetics
    // This handles cases like SCALE BINNED fill when fill is remapped from count
    for spec in &specs {
        apply_post_stat_binning(spec, &mut data_map)?;
    }

    // Apply out-of-bounds handling to data based on scale oob properties
    for spec in &specs {
        apply_scale_oob(spec, &mut data_map)?;
    }

    // Prune unnecessary columns from each layer's DataFrame
    prune_dataframes_per_layer(&specs, &mut data_map)?;

    Ok(PreparedData {
        data: data_map,
        specs,
    })
}

// =============================================================================
// Automatic Scale Creation
// =============================================================================

/// Check if an aesthetic gets a default scale (type inferred from data).
///
/// Returns true for aesthetics that benefit from scale resolution
/// (input range, output range, transforms, breaks).
/// Returns false for aesthetics that should use Identity scale.
fn gets_default_scale(aesthetic: &str) -> bool {
    matches!(
        aesthetic,
        // Position aesthetics
        "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend" | "x2" | "y2"
        // Color aesthetics (color/colour/col already split to fill/stroke)
        | "fill" | "stroke"
        // Size aesthetics
        | "size" | "linewidth"
        // Other visual aesthetics
        | "opacity" | "shape" | "linetype"
    )
}

/// Create Scale objects for aesthetics that don't have explicit SCALE clauses.
///
/// For aesthetics with meaningful scale behavior, creates a minimal scale
/// (type will be inferred later by resolve_scales from column dtype).
/// For identity aesthetics (text, label, group, etc.), creates an Identity scale.
fn create_missing_scales(spec: &mut Plot) {
    let mut used_aesthetics: HashSet<String> = HashSet::new();

    // Collect from layer mappings and remappings
    // (global mappings have already been merged into layers at this point)
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = GeomAesthetics::primary_aesthetic(aesthetic);
            used_aesthetics.insert(primary.to_string());
        }
        for aesthetic in layer.remappings.aesthetics.keys() {
            let primary = GeomAesthetics::primary_aesthetic(aesthetic);
            used_aesthetics.insert(primary.to_string());
        }
    }

    // Find aesthetics that already have explicit scales
    let existing_scales: HashSet<String> =
        spec.scales.iter().map(|s| s.aesthetic.clone()).collect();

    // Create scales for missing aesthetics
    for aesthetic in used_aesthetics {
        if !existing_scales.contains(&aesthetic) {
            let mut scale = crate::plot::Scale::new(&aesthetic);
            // Set Identity scale type for aesthetics that don't get default scales
            if !gets_default_scale(&aesthetic) {
                scale.scale_type = Some(ScaleType::identity());
            }
            spec.scales.push(scale);
        }
    }
}

/// Create scales for aesthetics that appeared from stat transforms (remappings).
///
/// Called after build_layer_query() to handle aesthetics like:
/// - y → __ggsql_stat_count__ (histogram, bar)
/// - x2 → __ggsql_stat_bin_end__ (histogram)
///
/// This is necessary because stat transforms modify layer.mappings after
/// create_missing_scales() has already run, potentially adding new aesthetics
/// that don't have corresponding scales.
fn create_missing_scales_post_stat(spec: &mut Plot) {
    let mut current_aesthetics: HashSet<String> = HashSet::new();

    // Collect all aesthetics currently in layer mappings
    for layer in &spec.layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            let primary = GeomAesthetics::primary_aesthetic(aesthetic);
            current_aesthetics.insert(primary.to_string());
        }
    }

    // Find aesthetics that don't have scales yet
    let existing_scales: HashSet<String> =
        spec.scales.iter().map(|s| s.aesthetic.clone()).collect();

    // Create scales for new aesthetics
    for aesthetic in current_aesthetics {
        if !existing_scales.contains(&aesthetic) {
            let mut scale = crate::plot::Scale::new(&aesthetic);
            if !gets_default_scale(&aesthetic) {
                scale.scale_type = Some(ScaleType::identity());
            }
            spec.scales.push(scale);
        }
    }
}

/// Apply binning directly to DataFrame columns for post-stat aesthetics.
///
/// This handles cases where a user specifies `SCALE BINNED` on a remapped aesthetic
/// (e.g., binning histogram's count output if remapped to fill).
///
/// Called after resolve_scales() so that breaks have been calculated.
///
/// This handles binning for aesthetics that get their values from stat transforms
/// (e.g., SCALE BINNED fill when fill is remapped from count). Aesthetics that
/// are directly mapped from source columns are pre-stat binned via SQL transforms.
fn apply_post_stat_binning(spec: &Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    for scale in &spec.scales {
        // Only process Binned scales
        match &scale.scale_type {
            Some(st) if st.scale_type_kind() == ScaleTypeKind::Binned => {}
            _ => continue,
        }

        // Get breaks from properties (skip if no breaks calculated)
        let breaks = match scale.properties.get("breaks") {
            Some(ParameterValue::Array(arr)) if arr.len() >= 2 => arr,
            _ => continue,
        };

        // Extract break values as f64
        let break_values: Vec<f64> = breaks.iter().filter_map(|e| e.to_f64()).collect();

        if break_values.len() < 2 {
            continue;
        }

        // Get closed property (default: left)
        let closed_left = match scale.properties.get("closed") {
            Some(ParameterValue::String(s)) => s != "right",
            _ => true,
        };

        // Find columns for this aesthetic across layers
        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        // Apply binning to each column
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
                    continue;
                }

                // Skip post-stat binning for aesthetic columns (like __ggsql_aes_x__)
                // because pre_stat_transform already binned them via SQL.
                // Post-stat binning only applies to stat columns or remapped aesthetics.
                if naming::is_aesthetic_column(&col_name) {
                    continue;
                }

                let binned_df =
                    apply_binning_to_dataframe(df, &col_name, &break_values, closed_left)?;
                data_map.insert(data_key, binned_df);
            }
        }
    }

    Ok(())
}

/// Apply binning transformation to a DataFrame column.
///
/// Replaces each value with the center of its bin based on the break values.
fn apply_binning_to_dataframe(
    df: &DataFrame,
    col_name: &str,
    break_values: &[f64],
    closed_left: bool,
) -> Result<DataFrame> {
    use polars::prelude::*;

    let column = df.column(col_name).map_err(|e| {
        GgsqlError::InternalError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let series = column.as_materialized_series();

    // Cast to f64 for binning
    let float_series = series.cast(&DataType::Float64).map_err(|e| {
        GgsqlError::InternalError(format!("Cannot bin column '{}': {}", col_name, e))
    })?;

    let ca = float_series
        .f64()
        .map_err(|e| GgsqlError::InternalError(e.to_string()))?;

    // Apply binning: replace values with bin centers
    let num_bins = break_values.len() - 1;
    let binned: Float64Chunked = ca.apply_values(|val| {
        for i in 0..num_bins {
            let lower = break_values[i];
            let upper = break_values[i + 1];
            let is_last = i == num_bins - 1;

            let in_bin = if closed_left {
                // Left-closed: [lower, upper) except last bin is [lower, upper]
                if is_last {
                    val >= lower && val <= upper
                } else {
                    val >= lower && val < upper
                }
            } else {
                // Right-closed: (lower, upper] except first bin is [lower, upper]
                if i == 0 {
                    val >= lower && val <= upper
                } else {
                    val > lower && val <= upper
                }
            };

            if in_bin {
                return (lower + upper) / 2.0;
            }
        }
        f64::NAN // Outside all bins
    });

    let binned_series = binned.into_series().with_name(col_name.into());

    // Replace column in DataFrame
    let mut new_df = df.clone();
    let _ = new_df
        .replace(col_name, binned_series)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))?;

    Ok(new_df)
}

/// Resolve scale types and transforms early, based on column dtypes.
///
/// This function:
/// 1. Infers scale_type from column dtype if not explicitly set
/// 2. Applies type coercion across layers for same aesthetic
/// 3. Resolves transform from scale_type + dtype if not explicit
///
/// Called early in the pipeline so that type requirements can be determined
/// before min/max extraction.
fn resolve_scale_types_and_transforms(
    spec: &mut Plot,
    layer_type_info: &[Vec<TypeInfo>],
) -> Result<()> {
    use crate::plot::scale::coerce_dtypes;
    use crate::plot::scale::transform::Transform;

    for scale in &mut spec.scales {
        // Skip scales that already have explicit types (user specified)
        if scale.scale_type.is_some() {
            // Still need to resolve transform if not set
            if scale.transform.is_none() && !scale.explicit_transform {
                // Collect all dtypes and coerce to common type (same as inference branch)
                let all_dtypes =
                    collect_dtypes_for_aesthetic(&spec.layers, &scale.aesthetic, layer_type_info);
                if !all_dtypes.is_empty() {
                    if let Ok(common_dtype) = coerce_dtypes(&all_dtypes) {
                        let scale_type = scale.scale_type.as_ref().unwrap();
                        // For Discrete/Ordinal scales, check input range first for transform inference
                        // This allows SCALE DISCRETE x FROM [true, false] to infer Bool transform
                        // even when the column is String
                        let transform_kind = if matches!(
                            scale_type.scale_type_kind(),
                            crate::plot::scale::ScaleTypeKind::Discrete
                                | crate::plot::scale::ScaleTypeKind::Ordinal
                        ) {
                            if let Some(ref input_range) = scale.input_range {
                                use crate::plot::scale::infer_transform_from_input_range;
                                if let Some(kind) = infer_transform_from_input_range(input_range) {
                                    kind
                                } else {
                                    scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                                }
                            } else {
                                scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                            }
                        } else {
                            scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                        };
                        scale.transform = Some(Transform::from_kind(transform_kind));
                    }
                }
            }
            continue;
        }

        // Collect all dtypes for this aesthetic across layers
        let all_dtypes =
            collect_dtypes_for_aesthetic(&spec.layers, &scale.aesthetic, layer_type_info);

        if all_dtypes.is_empty() {
            continue;
        }

        // Determine common dtype through coercion
        let common_dtype = match coerce_dtypes(&all_dtypes) {
            Ok(dt) => dt,
            Err(e) => {
                return Err(GgsqlError::ValidationError(format!(
                    "Scale '{}': {}",
                    scale.aesthetic, e
                )));
            }
        };

        // Infer scale type, considering explicit transform if set
        // If user specified VIA date/datetime/time/log/sqrt/etc., use Continuous scale
        let inferred_scale_type = if scale.explicit_transform {
            if let Some(ref transform) = scale.transform {
                use crate::plot::scale::TransformKind;
                match transform.transform_kind() {
                    // Temporal transforms require Continuous scale
                    TransformKind::Date
                    | TransformKind::DateTime
                    | TransformKind::Time
                    // Numeric continuous transforms require Continuous scale
                    | TransformKind::Log10
                    | TransformKind::Log2
                    | TransformKind::Log
                    | TransformKind::Sqrt
                    | TransformKind::Square
                    | TransformKind::Exp10
                    | TransformKind::Exp2
                    | TransformKind::Exp
                    | TransformKind::Asinh
                    | TransformKind::PseudoLog
                    // Integer transform uses Continuous scale
                    | TransformKind::Integer => ScaleType::continuous(),
                    // Discrete transforms (String, Bool) use Discrete scale
                    TransformKind::String | TransformKind::Bool => ScaleType::discrete(),
                    // Identity: fall back to dtype inference
                    TransformKind::Identity => ScaleType::infer(&common_dtype),
                }
            } else {
                ScaleType::infer(&common_dtype)
            }
        } else {
            ScaleType::infer(&common_dtype)
        };
        scale.scale_type = Some(inferred_scale_type.clone());

        // Infer transform if not explicit
        if scale.transform.is_none() && !scale.explicit_transform {
            // For Discrete scales, check input range first for transform inference
            // This allows SCALE DISCRETE x FROM [true, false] to infer Bool transform
            // even when the column is String
            let transform_kind = if inferred_scale_type.scale_type_kind()
                == crate::plot::scale::ScaleTypeKind::Discrete
            {
                if let Some(ref input_range) = scale.input_range {
                    use crate::plot::scale::infer_transform_from_input_range;
                    if let Some(kind) = infer_transform_from_input_range(input_range) {
                        kind
                    } else {
                        inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                    }
                } else {
                    inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
                }
            } else {
                inferred_scale_type.default_transform(&scale.aesthetic, Some(&common_dtype))
            };
            scale.transform = Some(Transform::from_kind(transform_kind));
        }
    }

    Ok(())
}

/// Collect all dtypes for an aesthetic across layers.
fn collect_dtypes_for_aesthetic(
    layers: &[Layer],
    aesthetic: &str,
    layer_type_info: &[Vec<TypeInfo>],
) -> Vec<polars::prelude::DataType> {
    let mut dtypes = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx >= layer_type_info.len() {
            continue;
        }
        let type_info = &layer_type_info[layer_idx];

        for aes_name in &aesthetics_to_check {
            if let Some(value) = layer.mappings.get(aes_name) {
                if let Some(col_name) = value.column_name() {
                    if let Some((_, dtype, _)) = type_info.iter().find(|(n, _, _)| n == col_name) {
                        dtypes.push(dtype.clone());
                    }
                }
            }
        }
    }
    dtypes
}

// =============================================================================
// Pre-Stat Scale Resolution (Binned Scales)
// =============================================================================

/// Pre-resolve Binned scales using schema-derived context.
///
/// This function resolves Binned scales before layer queries are built,
/// so that `pre_stat_transform_sql` has access to resolved breaks for
/// generating binning SQL.
///
/// Only Binned scales are resolved here; other scales are resolved
/// post-stat by `resolve_scales`.
fn apply_pre_stat_resolve(spec: &mut Plot, layer_schemas: &[Schema]) -> Result<()> {
    use crate::plot::scale::{ScaleDataContext, ScaleTypeKind};

    for scale in &mut spec.scales {
        // Only pre-resolve Binned scales
        let scale_type = match &scale.scale_type {
            Some(st) if st.scale_type_kind() == ScaleTypeKind::Binned => st.clone(),
            _ => continue,
        };

        // Find all ColumnInfos for this aesthetic from schemas
        let column_infos =
            find_schema_columns_for_aesthetic(&spec.layers, &scale.aesthetic, layer_schemas);

        if column_infos.is_empty() {
            continue;
        }

        // Build context from schema information
        let context = ScaleDataContext::from_schemas(&column_infos);

        // Use unified resolve method
        scale_type
            .resolve(scale, &context, &scale.aesthetic.clone())
            .map_err(|e| {
                GgsqlError::ValidationError(format!("Scale '{}': {}", scale.aesthetic, e))
            })?;
    }

    Ok(())
}

/// Find ColumnInfo for an aesthetic from layer schemas.
///
/// Similar to `find_columns_for_aesthetic` but works with schema information
/// (ColumnInfo) instead of actual data (Column).
///
/// Handles both column mappings (looked up in schema) and literal mappings
/// (synthetic ColumnInfo created from the literal value).
///
/// Note: Global mappings have already been merged into layer mappings at this point.
fn find_schema_columns_for_aesthetic(
    layers: &[Layer],
    aesthetic: &str,
    layer_schemas: &[Schema],
) -> Vec<ColumnInfo> {
    let mut infos = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping (global mappings already merged)
    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer_idx >= layer_schemas.len() {
            continue;
        }
        let schema = &layer_schemas[layer_idx];

        for aes_name in &aesthetics_to_check {
            if let Some(value) = layer.mappings.get(aes_name) {
                match value {
                    AestheticValue::Column { name, .. } => {
                        if let Some(info) = schema.iter().find(|c| c.name == *name) {
                            infos.push(info.clone());
                        }
                    }
                    AestheticValue::Literal(lit) => {
                        // Create synthetic ColumnInfo from literal
                        if let Some(info) = column_info_from_literal(aes_name, lit) {
                            infos.push(info);
                        }
                    }
                }
            }
        }
    }

    infos
}

/// Create a synthetic ColumnInfo from a literal value.
///
/// Used to include literal mappings in scale resolution.
fn column_info_from_literal(aesthetic: &str, lit: &LiteralValue) -> Option<ColumnInfo> {
    use polars::prelude::DataType;

    match lit {
        LiteralValue::Number(n) => Some(ColumnInfo {
            name: naming::const_column(aesthetic),
            dtype: DataType::Float64,
            is_discrete: false,
            min: Some(ArrayElement::Number(*n)),
            max: Some(ArrayElement::Number(*n)),
        }),
        LiteralValue::String(s) => Some(ColumnInfo {
            name: naming::const_column(aesthetic),
            dtype: DataType::String,
            is_discrete: true,
            min: Some(ArrayElement::String(s.clone())),
            max: Some(ArrayElement::String(s.clone())),
        }),
        LiteralValue::Boolean(_) => {
            // Boolean literals don't contribute to numeric ranges
            None
        }
    }
}

// =============================================================================
// Scale Type Coercion
// =============================================================================

/// Infer the target type for coercion based on scale kind.
///
/// Different scale kinds determine type differently:
/// - **Discrete**: Type from input range (e.g., `FROM [true, false]` → Boolean)
/// - **Continuous**: Type from transform (e.g., `VIA date` → Date, `VIA log10` → Number)
/// - **Binned**: No coercion (binning happens in SQL before DataFrame)
/// - **Identity**: No coercion
fn infer_scale_target_type(scale: &Scale) -> Option<ArrayElementType> {
    let scale_type = scale.scale_type.as_ref()?;

    match scale_type.scale_type_kind() {
        // Discrete/Ordinal: type from input range
        ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal => scale
            .input_range
            .as_ref()
            .and_then(|r| ArrayElement::infer_type(r)),
        // Continuous: type from transform
        ScaleTypeKind::Continuous => scale.transform.as_ref().map(|t| t.target_type()),
        // Binned: no coercion (binning happens in SQL before DataFrame)
        ScaleTypeKind::Binned => None,
        // Identity: no coercion
        ScaleTypeKind::Identity => None,
    }
}

/// Coerce a Polars column to the target ArrayElementType.
///
/// Returns a new DataFrame with the coerced column, or an error if coercion fails.
fn coerce_column_to_type(
    df: &DataFrame,
    column_name: &str,
    target_type: ArrayElementType,
) -> Result<DataFrame> {
    use polars::prelude::{DataType, NamedFrom, Series};

    let column = df.column(column_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", column_name, e))
    })?;

    let series = column.as_materialized_series();
    let dtype = series.dtype();

    // Check if already the target type
    let already_target_type = matches!(
        (dtype, target_type),
        (DataType::Boolean, ArrayElementType::Boolean)
            | (
                DataType::Float64 | DataType::Int64 | DataType::Int32 | DataType::Float32,
                ArrayElementType::Number,
            )
            | (DataType::Date, ArrayElementType::Date)
            | (DataType::Datetime(_, _), ArrayElementType::DateTime)
            | (DataType::Time, ArrayElementType::Time)
            | (DataType::String, ArrayElementType::String)
    );

    if already_target_type {
        return Ok(df.clone());
    }

    // Coerce based on target type
    let new_series: Series = match target_type {
        ArrayElementType::Boolean => {
            // Convert to boolean
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for boolean coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let bool_vec: Vec<Option<bool>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| match opt_s {
                            None => Ok(None),
                            Some(s) => match s.to_lowercase().as_str() {
                                "true" | "yes" | "1" => Ok(Some(true)),
                                "false" | "no" | "0" => Ok(Some(false)),
                                _ => Err(GgsqlError::ValidationError(format!(
                                    "Column '{}' row {}: Cannot coerce string '{}' to boolean",
                                    column_name, idx, s
                                ))),
                            },
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), bool_vec)
                }
                DataType::Int64 | DataType::Int32 | DataType::Float64 | DataType::Float32 => {
                    let f64_series = series.cast(&DataType::Float64).map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot cast column '{}' to float64: {}",
                            column_name, e
                        ))
                    })?;
                    let ca = f64_series.f64().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot get float64 chunked array: {}",
                            e
                        ))
                    })?;
                    let bool_vec: Vec<Option<bool>> =
                        ca.into_iter().map(|opt| opt.map(|n| n != 0.0)).collect();
                    Series::new(column_name.into(), bool_vec)
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to boolean",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::Number => {
            // Convert to float64
            series.cast(&DataType::Float64).map_err(|e| {
                GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' to number: {}",
                    column_name, e
                ))
            })?
        }

        ArrayElementType::Date => {
            // Convert to date (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for date coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let date_vec: Vec<Option<i32>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| {
                            match opt_s {
                                None => Ok(None),
                                Some(s) => {
                                    ArrayElement::from_date_string(s)
                                        .and_then(|e| match e {
                                            ArrayElement::Date(d) => Some(d),
                                            _ => None,
                                        })
                                        .ok_or_else(|| {
                                            GgsqlError::ValidationError(format!(
                                                "Column '{}' row {}: Cannot coerce string '{}' to date (expected YYYY-MM-DD)",
                                                column_name, idx, s
                                            ))
                                        })
                                        .map(Some)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), date_vec)
                        .cast(&DataType::Date)
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!("Cannot create date series: {}", e))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to date",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::DateTime => {
            // Convert to datetime (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for datetime coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let dt_vec: Vec<Option<i64>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| match opt_s {
                            None => Ok(None),
                            Some(s) => ArrayElement::from_datetime_string(s)
                                .and_then(|e| match e {
                                    ArrayElement::DateTime(dt) => Some(dt),
                                    _ => None,
                                })
                                .ok_or_else(|| {
                                    GgsqlError::ValidationError(format!(
                                        "Column '{}' row {}: Cannot coerce string '{}' to datetime",
                                        column_name, idx, s
                                    ))
                                })
                                .map(Some),
                        })
                        .collect::<Result<Vec<_>>>()?;

                    use polars::prelude::TimeUnit;
                    Series::new(column_name.into(), dt_vec)
                        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!(
                                "Cannot create datetime series: {}",
                                e
                            ))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to datetime",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::Time => {
            // Convert to time (from string)
            match dtype {
                DataType::String => {
                    let str_series = series.str().map_err(|e| {
                        GgsqlError::ValidationError(format!(
                            "Cannot convert column '{}' to string for time coercion: {}",
                            column_name, e
                        ))
                    })?;

                    let time_vec: Vec<Option<i64>> = str_series
                        .into_iter()
                        .enumerate()
                        .map(|(idx, opt_s)| {
                            match opt_s {
                                None => Ok(None),
                                Some(s) => {
                                    ArrayElement::from_time_string(s)
                                        .and_then(|e| match e {
                                            ArrayElement::Time(t) => Some(t),
                                            _ => None,
                                        })
                                        .ok_or_else(|| {
                                            GgsqlError::ValidationError(format!(
                                                "Column '{}' row {}: Cannot coerce string '{}' to time (expected HH:MM:SS)",
                                                column_name, idx, s
                                            ))
                                        })
                                        .map(Some)
                                }
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    Series::new(column_name.into(), time_vec)
                        .cast(&DataType::Time)
                        .map_err(|e| {
                            GgsqlError::ValidationError(format!("Cannot create time series: {}", e))
                        })?
                }
                _ => {
                    return Err(GgsqlError::ValidationError(format!(
                        "Cannot coerce column '{}' of type {:?} to time",
                        column_name, dtype
                    )));
                }
            }
        }

        ArrayElementType::String => {
            // Convert to string
            series.cast(&DataType::String).map_err(|e| {
                GgsqlError::ValidationError(format!(
                    "Cannot coerce column '{}' to string: {}",
                    column_name, e
                ))
            })?
        }
    };

    // Replace the column in the DataFrame
    let mut new_df = df.clone();
    let _ = new_df.replace(column_name, new_series);
    Ok(new_df)
}

/// Coerce columns mapped to an aesthetic in all relevant DataFrames.
///
/// This function finds all columns mapped to the given aesthetic across all layers
/// and coerces them to the target type.
fn coerce_aesthetic_columns(
    layers: &[Layer],
    data_map: &mut HashMap<String, DataFrame>,
    aesthetic: &str,
    target_type: ArrayElementType,
) -> Result<()> {
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Track which (data_key, column_name) pairs we've already coerced
    let mut coerced: HashSet<(String, String)> = HashSet::new();

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        let layer_key = naming::layer_key(i);

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                // Skip if layer doesn't have data
                if !data_map.contains_key(&layer_key) {
                    continue;
                }

                // Skip if already coerced
                let key = (layer_key.clone(), name.clone());
                if coerced.contains(&key) {
                    continue;
                }

                // Check if column exists in this DataFrame
                if let Some(df) = data_map.get(&layer_key) {
                    if df.column(name).is_ok() {
                        let coerced_df = coerce_column_to_type(df, name, target_type)?;
                        data_map.insert(layer_key.clone(), coerced_df);
                        coerced.insert(key);
                    }
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Scale Resolution
// =============================================================================

/// Resolve scale properties from data after materialization.
///
/// For each scale, this function:
/// 1. Infers target type and coerces columns if needed
/// 2. Infers scale_type from column data types if not explicitly set
/// 3. Uses the unified `resolve` method to fill in input_range, transform, and breaks
/// 4. Resolves output_range if not already set
///
/// The function inspects columns mapped to the aesthetic (including family
/// members like xmin/xmax for "x") and computes appropriate ranges.
///
/// Scales that were already resolved pre-stat (Binned scales) are skipped.
fn resolve_scales(spec: &mut Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    use crate::plot::scale::ScaleDataContext;

    for idx in 0..spec.scales.len() {
        // Clone aesthetic to avoid borrow issues with find_columns_for_aesthetic
        let aesthetic = spec.scales[idx].aesthetic.clone();

        // Skip scales that were already resolved pre-stat (e.g., Binned scales)
        // (resolve_output_range is now handled inside the unified resolve() method)
        if spec.scales[idx].resolved {
            continue;
        }

        // NEW: Infer target type and coerce columns if needed
        // This enables e.g. SCALE DISCRETE color FROM [true, false] to coerce string "true"/"false" to boolean
        if let Some(target_type) = infer_scale_target_type(&spec.scales[idx]) {
            coerce_aesthetic_columns(&spec.layers, data_map, &aesthetic, target_type)?;
        }

        // Find column references for this aesthetic (including family members)
        // NOTE: Must be called AFTER coercion so column types are correct
        let column_refs = find_columns_for_aesthetic(&spec.layers, &aesthetic, data_map);

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
            // Determine if this scale uses discrete input range (unique values vs min/max)
            let use_discrete_range = st.uses_discrete_input_range();

            // Build context from actual data columns
            let context = ScaleDataContext::from_columns(&column_refs, use_discrete_range);

            // Use unified resolve method (includes resolve_output_range)
            st.resolve(&mut spec.scales[idx], &context, &aesthetic)
                .map_err(|e| {
                    GgsqlError::ValidationError(format!("Scale '{}': {}", aesthetic, e))
                })?;
        }
    }

    Ok(())
}

/// Find all columns for an aesthetic (including family members like xmin/xmax for "x").
/// Each mapping is looked up in its corresponding data source.
/// Returns references to the Columns found.
///
/// Note: Global mappings have already been merged into layer mappings at this point.
fn find_columns_for_aesthetic<'a>(
    layers: &[Layer],
    aesthetic: &str,
    data_map: &'a HashMap<String, DataFrame>,
) -> Vec<&'a Column> {
    let mut column_refs = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        if let Some(df) = data_map.get(&naming::layer_key(i)) {
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

/// Apply out-of-bounds handling to data based on scale oob properties.
///
/// For each scale with `oob != "keep"`, this function transforms the data:
/// - `censor`: Filter out rows where the aesthetic's column values fall outside the input range
/// - `squish`: Clamp column values to the input range limits (continuous only)
///
/// After all OOB transformations, filters out NULL rows for columns where:
/// - The scale has an explicit input range, AND
/// - NULL is not part of the explicit input range
fn apply_scale_oob(spec: &Plot, data_map: &mut HashMap<String, DataFrame>) -> Result<()> {
    use crate::plot::scale::default_oob;

    // First pass: apply OOB transformations (censor sets to NULL, squish clamps)
    for scale in &spec.scales {
        // Get oob mode:
        // - If explicitly set, use that value (skip if "keep")
        // - If not set but has explicit input range, use default for aesthetic
        // - Otherwise skip
        let oob_mode = match scale.properties.get("oob") {
            Some(ParameterValue::String(s)) if s != OOB_KEEP => s.as_str(),
            Some(ParameterValue::String(_)) => continue, // explicit "keep"
            None if scale.explicit_input_range => {
                let default = default_oob(&scale.aesthetic);
                if default == OOB_KEEP {
                    continue;
                }
                default
            }
            _ => continue,
        };

        // Get input range, skip if none
        let input_range = match &scale.input_range {
            Some(r) if !r.is_empty() => r,
            _ => continue,
        };

        // Find all (data_key, column_name) pairs for this aesthetic
        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        // Helper to check if element is numeric-like (Number, Date, DateTime, Time)
        fn is_numeric_element(elem: &ArrayElement) -> bool {
            matches!(
                elem,
                ArrayElement::Number(_)
                    | ArrayElement::Date(_)
                    | ArrayElement::DateTime(_)
                    | ArrayElement::Time(_)
            )
        }

        // Helper to extract numeric value from element (dates are days, datetime is µs, etc.)
        fn extract_numeric(elem: &ArrayElement) -> Option<f64> {
            match elem {
                ArrayElement::Number(n) => Some(*n),
                ArrayElement::Date(d) => Some(*d as f64),
                ArrayElement::DateTime(dt) => Some(*dt as f64),
                ArrayElement::Time(t) => Some(*t as f64),
                _ => None,
            }
        }

        // Determine if this is a numeric or discrete range
        let is_numeric_range = is_numeric_element(&input_range[0])
            && input_range.get(1).map_or(false, is_numeric_element);

        // Apply transformation to each (data_key, column_name) pair
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
                    continue;
                }

                let transformed = if is_numeric_range {
                    // Numeric range - extract min/max (works for Number, Date, DateTime, Time)
                    let (range_min, range_max) = match (
                        extract_numeric(&input_range[0]),
                        input_range.get(1).and_then(extract_numeric),
                    ) {
                        (Some(lo), Some(hi)) => (lo, hi),
                        _ => continue,
                    };
                    apply_oob_to_column_numeric(df, &col_name, range_min, range_max, oob_mode)?
                } else {
                    // Discrete range - collect allowed values as strings using to_key_string
                    let allowed_values: std::collections::HashSet<String> = input_range
                        .iter()
                        .filter(|elem| !matches!(elem, ArrayElement::Null))
                        .map(|elem| elem.to_key_string())
                        .collect();
                    apply_oob_to_column_discrete(df, &col_name, &allowed_values, oob_mode)?
                };
                data_map.insert(data_key, transformed);
            }
        }
    }

    // Second pass: filter out NULL rows for scales with explicit input ranges
    // This handles NULLs created by both pre-stat SQL censoring and post-stat OOB censor
    for scale in &spec.scales {
        // Only filter if explicit input range AND NULL is not in the range
        let should_filter_nulls = scale.explicit_input_range
            && scale
                .input_range
                .as_ref()
                .is_some_and(|range| !range.iter().any(|elem| matches!(elem, ArrayElement::Null)));

        if !should_filter_nulls {
            continue;
        }

        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic, data_map);

        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                if df.column(&col_name).is_ok() {
                    let filtered = filter_null_rows(df, &col_name)?;
                    data_map.insert(data_key, filtered);
                }
            }
        }
    }

    Ok(())
}

/// Find all (data_key, column_name) pairs for an aesthetic (including family members).
/// Returns tuples of (data source key, column name) for use in transformations.
///
/// Note: Global mappings have already been merged into layer mappings at this point.
fn find_columns_for_aesthetic_with_sources(
    layers: &[Layer],
    aesthetic: &str,
    data_map: &HashMap<String, DataFrame>,
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping - every layer has its own data
    for (i, layer) in layers.iter().enumerate() {
        let layer_key = naming::layer_key(i);

        // Skip if layer doesn't have data
        if !data_map.contains_key(&layer_key) {
            continue;
        }

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                results.push((layer_key.clone(), name.clone()));
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

            let result = df.filter(&mask).map_err(|e| {
                GgsqlError::InternalError(format!("Failed to filter DataFrame: {}", e))
            })?;
            Ok(result)
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

/// Filter out rows where a column has NULL values.
///
/// Used after OOB transformations to remove rows that were censored to NULL.
fn filter_null_rows(df: &DataFrame, col_name: &str) -> Result<DataFrame> {
    let col = df.column(col_name).map_err(|e| {
        GgsqlError::ValidationError(format!("Column '{}' not found: {}", col_name, e))
    })?;

    let mask = col.is_not_null();
    df.filter(&mask)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to filter NULL rows: {}", e)))
}

/// Apply oob transformation to a single discrete/categorical column in a DataFrame.
///
/// For discrete scales, censoring sets out-of-range values to null (preserving all rows)
/// rather than filtering out entire rows. This allows other aesthetics to still be visualized.
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

    // Build new series: keep allowed values, set others to null
    // This preserves all rows (unlike filtering) so other aesthetics can still be visualized
    let new_ca: StringChunked = (0..series.len())
        .map(|i| {
            match series.get(i) {
                Ok(val) => {
                    // Null values are kept as null
                    if val.is_null() {
                        return None;
                    }
                    // Convert value to string and check membership
                    let s = val.to_string();
                    // Remove quotes if present (polars adds quotes around strings)
                    let clean = s.trim_matches('"').to_string();
                    if allowed_values.contains(&clean) {
                        Some(clean)
                    } else {
                        None // CENSOR to null (not filter row!)
                    }
                }
                Err(_) => None,
            }
        })
        .collect();

    // Replace column (keep all rows)
    let new_series = new_ca.into_series().with_name(col_name.into());
    let mut result = df.clone();
    result
        .with_column(new_series)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to replace column: {}", e)))?;
    Ok(result)
}

// =============================================================================
// Column Pruning
// =============================================================================

/// Collect the set of column names required for a specific layer.
///
/// Returns column names needed for:
/// - Aesthetic mappings (e.g., `__ggsql_aes_x__`, `__ggsql_aes_y__`)
/// - Bin end columns for binned scales (e.g., `__ggsql_aes_x2__`)
/// - Facet variables (shared across all layers)
/// - Partition columns (for Vega-Lite detail encoding)
/// - Order column for Path geoms
fn collect_layer_required_columns(layer: &Layer, spec: &Plot) -> HashSet<String> {
    use crate::plot::layer::geom::GeomType;

    let mut required = HashSet::new();

    // Facet variables (shared across all layers)
    if let Some(ref facet) = spec.facet {
        for var in facet.get_variables() {
            required.insert(var);
        }
    }

    // Aesthetic columns for this layer
    for aesthetic in layer.mappings.aesthetics.keys() {
        let aes_col = naming::aesthetic_column(aesthetic);
        required.insert(aes_col.clone());

        // Check if this aesthetic has a binned scale
        if let Some(scale) = spec.find_scale(aesthetic) {
            if let Some(ref scale_type) = scale.scale_type {
                if scale_type.scale_type_kind() == ScaleTypeKind::Binned {
                    required.insert(naming::bin_end_column(&aes_col));
                }
            }
        }
    }

    // Partition columns for this layer (used by Vega-Lite detail encoding)
    for col in &layer.partition_by {
        required.insert(col.clone());
    }

    // Order column for Path geoms
    if layer.geom.geom_type() == GeomType::Path {
        required.insert(naming::ORDER_COLUMN.to_string());
    }

    required
}

/// Prune columns from a DataFrame to only include required columns.
///
/// Columns that don't exist in the DataFrame are silently ignored.
fn prune_dataframe(df: &DataFrame, required: &HashSet<String>) -> Result<DataFrame> {
    let columns_to_keep: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|name| required.contains(name.as_str()))
        .map(|name| name.to_string())
        .collect();

    if columns_to_keep.is_empty() {
        return Err(GgsqlError::InternalError(format!(
            "No columns remain after pruning. Required columns: {:?}",
            required
        )));
    }

    df.select(&columns_to_keep)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to prune columns: {}", e)))
}

/// Prune all DataFrames in the data map based on layer requirements.
///
/// Each layer's DataFrame is pruned to only include columns needed by that layer.
fn prune_dataframes_per_layer(
    specs: &[Plot],
    data_map: &mut HashMap<String, DataFrame>,
) -> Result<()> {
    for spec in specs {
        for layer in &spec.layers {
            if let Some(ref data_key) = layer.data_key {
                if let Some(df) = data_map.get(data_key) {
                    let required = collect_layer_required_columns(layer, spec);
                    let pruned = prune_dataframe(df, &required)?;
                    data_map.insert(data_key.clone(), pruned);
                }
            }
        }
    }
    Ok(())
}

/// Build data map from a query using DuckDB reader
///
/// Convenience wrapper around `prepare_data_with_executor` for direct DuckDB reader usage.
#[cfg(feature = "duckdb")]
pub fn prepare_data(query: &str, reader: &DuckDBReader) -> Result<PreparedData> {
    prepare_data_with_executor(query, |sql| reader.execute(sql), &reader.sql_type_names())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::naming;
    use crate::plot::ArrayElement;
    use crate::Geom;
    use polars::prelude::{DataType, IntoColumn};

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_global_only() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";

        let result = prepare_data(query, &reader).unwrap();

        // With the new approach, every layer has its own data (no GLOBAL_DATA_KEY)
        assert!(result.data.contains_key(&naming::layer_key(0)));
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

        // Layer with filter creates its own data - global data is NOT needed in data_map
        // (the filter query uses the global temp table internally, but the result is layer-specific)
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(0)));

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
    fn test_transform_cte_references() {
        // Test cases: (sql, cte_names, expected_contains, expected_not_contains)
        let test_cases: Vec<(
            &str,
            Vec<&str>,
            Vec<&str>,    // strings that should be in result
            Option<&str>, // exact match (if result should equal this)
        )> = vec![
            // Single CTE reference
            (
                "SELECT * FROM sales WHERE year = 2024",
                vec!["sales"],
                vec!["FROM __ggsql_cte_sales_", "__ WHERE year = 2024"],
                None,
            ),
            // Multiple CTE references
            (
                "SELECT * FROM sales JOIN targets ON sales.date = targets.date",
                vec!["sales", "targets"],
                vec!["FROM __ggsql_cte_sales_", "JOIN __ggsql_cte_targets_"],
                None,
            ),
            // No matching CTE (unchanged)
            (
                "SELECT * FROM other_table",
                vec!["sales"],
                vec![],
                Some("SELECT * FROM other_table"),
            ),
            // Empty CTE names (unchanged)
            (
                "SELECT * FROM sales",
                vec![],
                vec![],
                Some("SELECT * FROM sales"),
            ),
        ];

        for (sql, cte_names_vec, expected_contains, exact_match) in test_cases {
            let cte_names: HashSet<String> =
                cte_names_vec.iter().map(|s| s.to_string()).collect();
            let result = transform_cte_references(sql, &cte_names);

            if let Some(expected) = exact_match {
                assert_eq!(
                    result, expected,
                    "SQL '{}' should remain unchanged",
                    sql
                );
            } else {
                for expected in &expected_contains {
                    assert!(
                        result.contains(expected),
                        "Result '{}' should contain '{}' for SQL '{}'",
                        result,
                        expected,
                        sql
                    );
                }
                // When CTEs are transformed, result should contain session UUID
                if !cte_names_vec.is_empty() {
                    assert!(
                        result.contains(naming::session_id()),
                        "Result should contain session UUID"
                    );
                }
            }
        }
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

        // With new approach, all layers have their own data
        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(&naming::layer_key(1)));

        // Layer 0 should have 2 rows (from sales via global)
        let layer0_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer0_df.height(), 2);

        // Layer 1 should have 2 rows (from targets CTE)
        let layer1_df = result.data.get(&naming::layer_key(1)).unwrap();
        assert_eq!(layer1_df.height(), 2);
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

        // Layer 0 should have all 4 rows (from global)
        let layer0_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer0_df.height(), 4);

        // Layer 1 should have 2 rows (filtered to category = 'A')
        let layer1_df = result.data.get(&naming::layer_key(1)).unwrap();
        assert_eq!(layer1_df.height(), 2);
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

        // Both layers should have data_keys
        let layer0_key = result.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer 0 should have data_key");
        let layer1_key = result.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("Layer 1 should have data_key");

        // Both layer data should exist
        assert!(
            result.data.contains_key(layer0_key),
            "Should have layer 0 data"
        );
        assert!(
            result.data.contains_key(layer1_key),
            "Should have layer 1 data"
        );

        // Both should have 3 rows
        assert_eq!(result.data.get(layer0_key).unwrap().height(), 3);
        assert_eq!(result.data.get(layer1_key).unwrap().height(), 3);
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

        // With new approach, all layers have their own data
        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert!(result.data.contains_key(&naming::layer_key(1)));

        assert_eq!(result.data.get(&naming::layer_key(0)).unwrap().height(), 2);
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

        // Layer 0 should have all 5 rows
        assert_eq!(result.data.get(&naming::layer_key(0)).unwrap().height(), 5);

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

        // Should have prefixed aesthetic-named columns (stat columns are renamed to aesthetics)
        // __ggsql_stat__bin -> __ggsql_aes_x__, __ggsql_stat__count -> __ggsql_aes_y__
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        assert!(
            col_names.contains(&x_col),
            "Should have '{}' column (from stat bin): {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Should have '{}' column (from stat count): {:?}",
            y_col,
            col_names
        );

        // Should have fewer rows than original (binned)
        assert!(layer_df.height() < 100);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_histogram_creates_y_scale_post_stat() {
        // Tests that scales are created for remapped aesthetics (e.g., y from histogram's count)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with continuous values
        reader
            .connection()
            .execute(
                "CREATE TABLE hist_scale_test AS SELECT RANDOM() * 100 as value FROM range(100)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM hist_scale_test
            VISUALISE
            DRAW histogram MAPPING value AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have scales for both x and y
        // x is from the original mapping, y is from the stat transform (count)
        let x_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "x");
        let y_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "y");

        assert!(
            x_scale.is_some(),
            "Should have x scale from original mapping"
        );
        assert!(
            y_scale.is_some(),
            "Should have y scale from stat transform remapping"
        );

        // y scale should have been resolved (scale_type inferred from count column)
        let y_scale = y_scale.unwrap();
        assert!(
            y_scale.scale_type.is_some(),
            "y scale should have scale_type resolved"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_creates_y_scale_post_stat() {
        // Tests that bar geom creates y scale for count stat
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with categories
        reader
            .connection()
            .execute(
                "CREATE TABLE bar_scale_test AS SELECT * FROM (VALUES ('A'), ('B'), ('A'), ('C')) AS t(category)",
                duckdb::params![],
            )
            .unwrap();

        // Bar with only x mapped - should apply count stat and create y scale
        let query = r#"
            SELECT * FROM bar_scale_test
            VISUALISE
            DRAW bar MAPPING category AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have scales for both x and y
        let x_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "x");
        let y_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "y");

        assert!(
            x_scale.is_some(),
            "Should have x scale from original mapping"
        );
        assert!(
            y_scale.is_some(),
            "Should have y scale from count stat remapping"
        );

        // y scale should have been resolved
        let y_scale = y_scale.unwrap();
        assert!(
            y_scale.scale_type.is_some(),
            "y scale should have scale_type resolved"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_histogram_creates_x2_scale_post_stat() {
        // Tests that histogram creates x2 scale for bin_end remapping
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE hist_x2_test AS SELECT RANDOM() * 100 as value FROM range(50)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM hist_x2_test
            VISUALISE
            DRAW histogram MAPPING value AS x
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have x2 scale from bin_end remapping
        // Note: x2 is part of the x aesthetic family, so it may not have its own scale
        // but x scale should exist and handle the x family
        let _x2_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "x2");
        let x_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "x");

        assert!(
            x_scale.is_some(),
            "Should have x scale for x aesthetic family"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_multi_layer_stat_and_non_stat() {
        // Tests that a plot with both stat geom (histogram) and non-stat geom (point)
        // correctly creates scales for all aesthetics including remapped ones
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE multi_layer_test AS SELECT n as id, n * 2.0 as x_val, n * 3.0 as y_val FROM range(50) t(n)",
                duckdb::params![],
            )
            .unwrap();

        // Histogram on x_val, point on x_val and y_val
        let query = r#"
            SELECT * FROM multi_layer_test
            VISUALISE
            DRAW histogram MAPPING x_val AS x
            DRAW point MAPPING x_val AS x, y_val AS y
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Both layers should have x scale
        let x_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "x");
        assert!(x_scale.is_some(), "Should have x scale");

        // Should have y scale (from histogram's count remapping AND point's y mapping)
        let y_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "y");
        assert!(y_scale.is_some(), "Should have y scale");

        // y scale should be resolved
        let y_scale = y_scale.unwrap();
        assert!(
            y_scale.scale_type.is_some(),
            "y scale should have scale_type resolved"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_post_stat_binning_on_count() {
        // Tests that SCALE BINNED can be applied to a remapped aesthetic
        // This is an edge case where user bins the count output from histogram/bar
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data with values
        reader
            .connection()
            .execute(
                "CREATE TABLE binned_count_test AS SELECT * FROM range(100) t(value)",
                duckdb::params![],
            )
            .unwrap();

        // Histogram with SCALE BINNED on the y (count) aesthetic
        let query = r#"
            SELECT * FROM binned_count_test
            VISUALISE
            DRAW histogram MAPPING value AS x
            SCALE BINNED y SETTING breaks => 5
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have both x and y scales
        let y_scale = result.specs[0].scales.iter().find(|s| s.aesthetic == "y");
        assert!(y_scale.is_some(), "Should have y scale");

        let y_scale = y_scale.unwrap();
        assert!(y_scale.resolved, "y scale should be resolved");

        // The y scale should have breaks calculated
        let breaks = y_scale.properties.get("breaks");
        assert!(breaks.is_some(), "y scale should have breaks property");

        // Verify the data has been binned (count values should be bin centers)
        // With prefixed aesthetic-named columns, stat count is renamed to "__ggsql_aes_y__"
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let y_col_name = naming::aesthetic_column("y");
        let count_col = layer_df.column(&y_col_name);
        assert!(
            count_col.is_ok(),
            "Should have '{}' column (from stat count) in layer data",
            y_col_name
        );
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

        // With new approach, columns are renamed to prefixed aesthetic names
        // So "category" is renamed to "__ggsql_aes_x__" and stat count column is renamed to "__ggsql_aes_y__"
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        assert!(
            col_names.contains(&x_col),
            "Expected '{}' in {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Expected '{}' in {:?}",
            y_col,
            col_names
        );
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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have original 3 rows (no stat transform when y is mapped)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
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
        // Stat columns are renamed to prefixed aesthetic names (bin->x, count->y)
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        assert!(
            col_names.contains(&"region".to_string()),
            "Should have 'region' facet column: {:?}",
            col_names
        );
        assert!(
            col_names.contains(&x_col),
            "Should have '{}' column (from stat bin): {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Should have '{}' column (from stat count): {:?}",
            y_col,
            col_names
        );
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

        // Should have grp column preserved for grouping (partition_by)
        // category is renamed to prefixed x, count is renamed to prefixed y
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        assert!(
            col_names.contains(&"grp".to_string()),
            "Expected 'grp' (partition_by column) in {:?}",
            col_names
        );
        assert!(
            col_names.contains(&x_col),
            "Expected '{}' in {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Expected '{}' in {:?}",
            y_col,
            col_names
        );

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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have 3 rows (no transformation, but still layer-specific)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have original 3 rows (no transformation when y is mapped and exists)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);

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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have original 3 rows (wildcard with y uses identity)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have original 3 rows (no COUNT applied when y exists)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
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

        // With new approach, every layer has its own data
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer should have 3 rows (no stat transform since y column exists)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
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
        // With new approach, stat count column is renamed to prefixed "__ggsql_aes_y__"
        let y_col_name = naming::aesthetic_column("y");
        let y_col = layer_df.column(&y_col_name).unwrap_or_else(|_| {
            panic!(
                "'{}' column should exist (stat count renamed to prefixed aesthetic)",
                y_col_name
            )
        });
        // SUM may return f64 depending on DB/type handling
        let y_values: Vec<f64> = y_col
            .f64()
            .expect("y should be f64 (SUM result)")
            .into_iter()
            .flatten()
            .collect();

        // Sum of A should be 30, sum of B should be 30
        assert!(
            y_values.contains(&30.0),
            "Should have sum of 30 for category A, got: {:?}",
            y_values
        );
        assert!(
            y_values.contains(&30.0),
            "Should have sum of 30 for category B, got: {:?}",
            y_values
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
        // With new approach, stat count column is renamed to prefixed "__ggsql_aes_y__"
        let y_col_name = naming::aesthetic_column("y");
        let y_col = layer_df.column(&y_col_name).unwrap_or_else(|_| {
            panic!(
                "'{}' column should exist (stat count renamed to prefixed aesthetic)",
                y_col_name
            )
        });
        let y_values: Vec<i64> = y_col
            .i64()
            .expect("y should be i64")
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
        // With prefixed aesthetic-named columns, stat count is renamed to "__ggsql_aes_y__"
        let y_col_name = naming::aesthetic_column("y");
        let y_col = layer_df
            .column(&y_col_name)
            .unwrap_or_else(|_| panic!("'{}' column (stat count) should exist", y_col_name));
        let y_values: Vec<i64> = y_col
            .i64()
            .expect("y should be i64")
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
        // With prefixed aesthetic-named columns, stat count is renamed to "__ggsql_aes_y__"
        let y_col_name = naming::aesthetic_column("y");
        let y_col = layer_df
            .column(&y_col_name)
            .unwrap_or_else(|_| panic!("'{}' column (stat count) should exist", y_col_name));
        let y_values: Vec<f64> = y_col
            .f64()
            .expect("y should be f64 (SUM result)")
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

        // Date columns now use Continuous scale type with temporal transform
        assert_eq!(
            x_scale.scale_type,
            Some(ScaleType::continuous()),
            "Date column should infer Continuous scale type"
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

        // Temporal types now use Continuous scale (with temporal transforms)
        assert_eq!(ScaleType::infer(&DataType::Date), ScaleType::continuous());
        assert_eq!(
            ScaleType::infer(&DataType::Datetime(
                polars::prelude::TimeUnit::Microseconds,
                None
            )),
            ScaleType::continuous()
        );
        assert_eq!(ScaleType::infer(&DataType::Time), ScaleType::continuous());

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

        // Date columns now use Continuous scale which treats dates as numeric (days since epoch)
        let range = ScaleType::continuous()
            .resolve_input_range(None, &[&column], &props)
            .unwrap();
        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.len(), 2);

        // Continuous scale returns numeric range (days since epoch)
        assert_eq!(range[0], ArrayElement::Number(19737.0));
        assert_eq!(range[1], ArrayElement::Number(19802.0));
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

        // Nulls should be included at the end, result should be ["A", "B", null]
        assert_eq!(range.len(), 3);
        assert_eq!(range[0], ArrayElement::String("A".into()));
        assert_eq!(range[1], ArrayElement::String("B".into()));
        assert_eq!(range[2], ArrayElement::Null);
    }

    #[test]
    fn test_resolve_scales_infers_input_range() {
        use polars::prelude::*;

        // Create a Plot with a scale that needs range inference
        // (global mappings are merged into layers before resolve_scales is called)
        let mut spec = Plot::new();

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("x");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with numeric values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

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
        // (global mappings are merged into layers before resolve_scales is called)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with different values
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

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
        // (global mappings are merged into layers before resolve_scales is called)
        let mut spec = Plot::new();

        // Disable expansion for predictable test values
        let mut scale = crate::plot::Scale::new("y");
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mappings are in layer
        let layer = Layer::new(Geom::errorbar())
            .with_aesthetic("ymin".to_string(), AestheticValue::standard_column("low"))
            .with_aesthetic("ymax".to_string(), AestheticValue::standard_column("high"));
        spec.layers.push(layer);

        // Create data where ymin/ymax columns have different ranges
        let df = df! {
            "low" => &[5.0f64, 10.0, 15.0],
            "high" => &[20.0f64, 25.0, 30.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

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
        // (global mappings are merged into layers before resolve_scales is called)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Number(0.0), ArrayElement::Null]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

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
        // (global mappings are merged into layers before resolve_scales is called)
        let mut spec = Plot::new();

        let mut scale = crate::plot::Scale::new("x");
        scale.input_range = Some(vec![ArrayElement::Null, ArrayElement::Number(100.0)]);
        // Disable expansion for predictable test values
        scale.properties.insert(
            "expand".to_string(),
            crate::plot::ParameterValue::Number(0.0),
        );
        spec.scales.push(scale);
        // Simulate post-merge state: mapping is in layer
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Create data with values 1-10
        let df = df! {
            "value" => &[1.0f64, 5.0, 10.0]
        }
        .unwrap();

        let mut data_map = HashMap::new();
        data_map.insert(naming::layer_key(0), df);

        // Resolve scales
        resolve_scales(&mut spec, &mut data_map).unwrap();

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

        // With prefixed aesthetic-named columns, mappings are updated to use prefixed aesthetic names
        let stroke = aes.get("stroke").unwrap().column_name().unwrap();
        assert_eq!(stroke, naming::aesthetic_column("stroke")); // was "species", renamed to "__ggsql_aes_stroke__"

        let fill = aes.get("fill").unwrap().column_name().unwrap();
        assert_eq!(fill, naming::aesthetic_column("fill")); // was "island", renamed to "__ggsql_aes_fill__"

        // Colors as global constant
        // With aesthetic-named columns, constants become columns named after the aesthetic
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, 'blue' AS color FROM ggsql:penguins
          DRAW point MAPPING island AS stroke
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(
            stroke.column_name().unwrap(),
            naming::aesthetic_column("stroke")
        ); // was "island", renamed to "__ggsql_aes_stroke__"

        let fill = aes.get("fill").unwrap();
        assert_eq!(
            fill.column_name().unwrap(),
            naming::aesthetic_column("fill")
        ); // constant 'blue' -> column "__ggsql_aes_fill__"

        // Colors as layer constant
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, island AS fill FROM ggsql:penguins
          DRAW point MAPPING 'blue' AS color
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(
            stroke.column_name().unwrap(),
            naming::aesthetic_column("stroke")
        ); // constant 'blue' -> column "__ggsql_aes_stroke__"

        let fill = aes.get("fill").unwrap();
        assert_eq!(
            fill.column_name().unwrap(),
            naming::aesthetic_column("fill")
        ); // was "island", renamed to "__ggsql_aes_fill__"

        // Verify color is removed after splitting
        assert!(
            !aes.contains_key("color"),
            "color should be removed after splitting to stroke/fill"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_scale_colour_alias_normalization() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Test that SCALE colour gets normalized to color and split to fill/stroke
        let query = r#"
            VISUALISE bill_len AS x, bill_dep AS y, species AS color FROM ggsql:penguins
            DRAW point
            SCALE DISCRETE colour
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let scales = &result.specs[0].scales;

        // Should have fill and stroke scales, not color
        assert!(
            scales.iter().any(|s| s.aesthetic == "fill"),
            "should have fill scale"
        );
        assert!(
            scales.iter().any(|s| s.aesthetic == "stroke"),
            "should have stroke scale"
        );
        assert!(
            !scales.iter().any(|s| s.aesthetic == "color"),
            "color scale should be removed after splitting"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_scale_color_splits_to_fill_and_stroke() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Test that SCALE color gets split to fill and stroke scales
        let query = r#"
            VISUALISE bill_len AS x, bill_dep AS y, species AS color FROM ggsql:penguins
            DRAW point
            SCALE DISCRETE color TO viridis
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let scales = &result.specs[0].scales;

        // Should have fill and stroke scales with viridis palette, not color
        let fill_scale = scales.iter().find(|s| s.aesthetic == "fill");
        let stroke_scale = scales.iter().find(|s| s.aesthetic == "stroke");

        assert!(fill_scale.is_some(), "should have fill scale");
        assert!(stroke_scale.is_some(), "should have stroke scale");
        assert!(
            !scales.iter().any(|s| s.aesthetic == "color"),
            "color scale should be removed"
        );

        // Both fill and stroke scales should inherit the viridis palette (or its expanded hex colors)
        let fill = fill_scale.unwrap();
        // The palette may be expanded to hex codes in the parser, so check for either variant
        assert!(
            fill.output_range.is_some(),
            "fill scale should have output_range: {:?}",
            fill
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_scale_color_does_not_override_existing_fill_scale() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Test that existing fill scale is preserved when splitting color
        let query = r#"
            VISUALISE bill_len AS x, bill_dep AS y, species AS color FROM ggsql:penguins
            DRAW point
            SCALE DISCRETE color TO viridis
            SCALE DISCRETE fill TO plasma
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let scales = &result.specs[0].scales;

        // Should have fill scale with plasma (not overwritten by color split)
        let fill_scale = scales.iter().find(|s| s.aesthetic == "fill").unwrap();
        // Palettes may be expanded to hex arrays in the parser
        assert!(
            fill_scale.output_range.is_some(),
            "fill scale should have output_range (from plasma): {:?}",
            fill_scale
        );

        // stroke scale should have viridis from color split
        let stroke_scale = scales.iter().find(|s| s.aesthetic == "stroke").unwrap();
        assert!(
            stroke_scale.output_range.is_some(),
            "stroke scale should have output_range (from viridis): {:?}",
            stroke_scale
        );
    }

    // =========================================================================
    // OOB Tests
    // =========================================================================

    #[test]
    fn test_apply_oob_numeric_operations() {
        use polars::prelude::*;

        // Test cases: (oob_mode, expected_height, expected_values)
        let test_cases: Vec<(&str, usize, Vec<f64>)> = vec![
            // Censor: filters out values outside range [5, 15]
            ("censor", 3, vec![5.0, 10.0, 15.0]),
            // Squish: clamps values to range [5, 15]
            ("squish", 5, vec![5.0, 5.0, 10.0, 15.0, 15.0]),
            // Keep: preserves all values unchanged
            ("keep", 5, vec![1.0, 5.0, 10.0, 15.0, 20.0]),
        ];

        for (oob_mode, expected_height, expected_values) in test_cases {
            let df = DataFrame::new(vec![
                Series::new("x".into(), vec![1.0f64, 5.0, 10.0, 15.0, 20.0]).into(),
                Series::new("y".into(), vec![10.0f64, 20.0, 30.0, 40.0, 50.0]).into(),
            ])
            .unwrap();

            let result = apply_oob_to_column_numeric(&df, "x", 5.0, 15.0, oob_mode).unwrap();

            assert_eq!(
                result.height(),
                expected_height,
                "OOB mode '{}' should have {} rows",
                oob_mode,
                expected_height
            );

            let x_col = result.column("x").unwrap();
            let x_series = x_col.as_materialized_series().f64().unwrap();
            let values: Vec<f64> = x_series.into_iter().flatten().collect();
            assert_eq!(
                values, expected_values,
                "OOB mode '{}' produced wrong values",
                oob_mode
            );
        }
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
    fn test_apply_oob_discrete_censor_sets_null() {
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

        // Should have all 5 rows preserved (censoring sets to null, doesn't filter)
        assert_eq!(result.height(), 5);

        // Check values: A, B, C should be preserved, D, E should be null
        let cat_col = result.column("category").unwrap();
        let series = cat_col.as_materialized_series();

        // First 3 values should be A, B, C
        assert_eq!(series.get(0).unwrap().to_string().trim_matches('"'), "A");
        assert_eq!(series.get(1).unwrap().to_string().trim_matches('"'), "B");
        assert_eq!(series.get(2).unwrap().to_string().trim_matches('"'), "C");

        // D and E (rows 3 and 4) should be null
        assert!(series.get(3).unwrap().is_null());
        assert!(series.get(4).unwrap().is_null());

        // Value column should be unchanged
        let val_col = result.column("value").unwrap();
        let values: Vec<i32> = val_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
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
        // With pre-stat OOB transform, data is in layer key; otherwise in global key
        let layer_key = naming::layer_key(0);
        let df = result
            .data
            .get(&layer_key)
            .or_else(|| result.data.get(naming::GLOBAL_DATA_KEY))
            .expect("Should have layer or global data");
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
        // With pre-stat OOB transform, data is in layer key; otherwise in global key
        let layer_key = naming::layer_key(0);
        let df = result
            .data
            .get(&layer_key)
            .or_else(|| result.data.get(naming::GLOBAL_DATA_KEY))
            .expect("Should have layer or global data");
        assert_eq!(df.height(), 3);

        // Check clamped values (column is now named with prefixed aesthetic name)
        let y_col_name = naming::aesthetic_column("y");
        let y_col = df
            .column(&y_col_name)
            .unwrap_or_else(|_| panic!("Should have '{}' column", y_col_name));
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
        // With aesthetic-named columns, data is in layer_key(0)
        let df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_discrete_censor_integration() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with categories, some outside allowed range
        // Discrete scales censor OOB values to NULL, then NULL rows are filtered out
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

        // Only rows with allowed categories ('A', 'B') should remain
        // Rows with 'C' and 'D' are censored to NULL, then filtered out
        let layer_key = naming::layer_key(0);
        let df = result
            .data
            .get(&layer_key)
            .or_else(|| result.data.get(naming::GLOBAL_DATA_KEY))
            .expect("Should have layer or global data");

        // Debug: print column names and data
        eprintln!(
            "DEBUG censor_integration: Column names: {:?}",
            df.get_column_names()
        );
        eprintln!(
            "DEBUG censor_integration: DataFrame height: {}",
            df.height()
        );
        eprintln!(
            "DEBUG censor_integration: All scales: {:?}",
            result.specs[0].scales
        );
        eprintln!(
            "DEBUG censor_integration: stroke scale: {:?}",
            result.specs[0]
                .scales
                .iter()
                .find(|s| s.aesthetic == "stroke")
        );
        eprintln!(
            "DEBUG censor_integration: fill scale: {:?}",
            result.specs[0]
                .scales
                .iter()
                .find(|s| s.aesthetic == "fill")
        );

        assert_eq!(df.height(), 2);

        // Verify only rows with x=1,2 (categories A,B) remain
        // After pruning, columns are renamed to aesthetic names
        let x_col_name = naming::aesthetic_column("x");
        let x_col = df.column(&x_col_name).unwrap();
        let x_values: Vec<i32> = x_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(x_values, vec![1, 2]);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_discrete_bar_chart_filters_null_after_stat() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Bar chart with discrete scale - tests the full flow:
        // 1. Pre-stat SQL censors 'C' to NULL
        // 2. Stat transform (COUNT) groups by category, including NULL
        // 3. Post-stat should filter out the NULL row
        let query = r#"
            SELECT * FROM (VALUES
                ('A'), ('A'), ('A'),
                ('B'), ('B'),
                ('C')
            ) AS t(category)
            VISUALISE
            DRAW bar MAPPING category AS x
            SCALE x FROM ['A', 'B']
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have 2 rows (A and B), NOT 3 (with NULL)
        let df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(
            df.height(),
            2,
            "Expected 2 rows (A and B), but got {} - NULL row should be filtered",
            df.height()
        );

        // Verify only A and B are present (column is now named with prefixed aesthetic name)
        let x_col = naming::aesthetic_column("x");
        let cat_col = df
            .column(&x_col)
            .unwrap_or_else(|_| panic!("Should have '{}' column", x_col));
        let values: Vec<String> = cat_col
            .as_materialized_series()
            .str()
            .unwrap()
            .into_iter()
            .flatten()
            .map(|s| s.to_string())
            .collect();
        assert!(values.contains(&"A".to_string()));
        assert!(values.contains(&"B".to_string()));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_oob_discrete_default_censor_for_non_positional() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Non-positional aesthetics (like color) default to censor
        // Rows with OOB values are filtered out after censoring to NULL
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

        // Only rows with allowed categories ('A', 'B') should remain
        // Rows with 'C' and 'D' are censored to NULL, then filtered out
        let layer_key = naming::layer_key(0);
        let df = result
            .data
            .get(&layer_key)
            .or_else(|| result.data.get(naming::GLOBAL_DATA_KEY))
            .expect("Should have layer or global data");
        assert_eq!(df.height(), 2);

        // Verify only y values for rows with allowed categories remain
        // After pruning, columns are renamed to aesthetic names
        let y_col_name = naming::aesthetic_column("y");
        let y_col = df.column(&y_col_name).unwrap();
        let y_values: Vec<i32> = y_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(y_values, vec![10, 20]);
    }

    #[test]
    fn test_oob_discrete_censor_preserves_other_columns() {
        use polars::prelude::*;

        // Create DataFrame with multiple columns
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1, 2, 3, 4]).into(),
            Series::new("y".into(), vec![10.0, 20.0, 30.0, 40.0]).into(),
            Series::new("category".into(), vec!["A", "B", "C", "D"]).into(),
            Series::new("label".into(), vec!["one", "two", "three", "four"]).into(),
        ])
        .unwrap();

        // Only allow A, B in category
        let allowed: std::collections::HashSet<String> =
            ["A", "B"].iter().map(|s| s.to_string()).collect();

        let result = apply_oob_to_column_discrete(&df, "category", &allowed, "censor").unwrap();

        // All rows preserved
        assert_eq!(result.height(), 4);

        // x column unchanged
        let x_col = result.column("x").unwrap();
        let x_values: Vec<i32> = x_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(x_values, vec![1, 2, 3, 4]);

        // y column unchanged
        let y_col = result.column("y").unwrap();
        let y_values: Vec<f64> = y_col
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(y_values, vec![10.0, 20.0, 30.0, 40.0]);

        // label column unchanged
        let label_col = result.column("label").unwrap();
        let series = label_col.as_materialized_series();
        assert_eq!(series.get(0).unwrap().to_string().trim_matches('"'), "one");
        assert_eq!(series.get(1).unwrap().to_string().trim_matches('"'), "two");
        assert_eq!(
            series.get(2).unwrap().to_string().trim_matches('"'),
            "three"
        );
        assert_eq!(series.get(3).unwrap().to_string().trim_matches('"'), "four");

        // category column: A, B preserved, C, D set to null
        let cat_col = result.column("category").unwrap();
        let cat_series = cat_col.as_materialized_series();
        assert_eq!(
            cat_series.get(0).unwrap().to_string().trim_matches('"'),
            "A"
        );
        assert_eq!(
            cat_series.get(1).unwrap().to_string().trim_matches('"'),
            "B"
        );
        assert!(cat_series.get(2).unwrap().is_null());
        assert!(cat_series.get(3).unwrap().is_null());
    }

    // ==========================================================================
    // Schema with min/max range tests
    // ==========================================================================

    /// Helper function to fetch schema using the split approach (types + ranges)
    #[cfg(feature = "duckdb")]
    fn fetch_schema_for_test<F>(query: &str, execute_query: &F) -> Result<Schema>
    where
        F: Fn(&str) -> Result<DataFrame>,
    {
        let type_info = fetch_schema_types(query, execute_query)?;
        complete_schema_ranges(query, &type_info, execute_query)
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_numeric_columns() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        let query = "SELECT * FROM (VALUES (1, 10.5), (5, 20.0), (3, 15.5)) AS t(x, y)";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 2);

        // Column x: integer
        let col_x = &schema[0];
        assert_eq!(col_x.name, "x");
        assert!(!col_x.is_discrete);
        assert_eq!(col_x.min, Some(ArrayElement::Number(1.0)));
        assert_eq!(col_x.max, Some(ArrayElement::Number(5.0)));

        // Column y: float
        let col_y = &schema[1];
        assert_eq!(col_y.name, "y");
        assert!(!col_y.is_discrete);
        assert_eq!(col_y.min, Some(ArrayElement::Number(10.5)));
        assert_eq!(col_y.max, Some(ArrayElement::Number(20.0)));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_string_columns() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        let query = "SELECT * FROM (VALUES ('apple'), ('cherry'), ('banana')) AS t(fruit)";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 1);

        let col = &schema[0];
        assert_eq!(col.name, "fruit");
        assert!(col.is_discrete);
        // Lexicographic min/max
        assert_eq!(col.min, Some(ArrayElement::String("apple".to_string())));
        assert_eq!(col.max, Some(ArrayElement::String("cherry".to_string())));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_date_columns() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        let query = "SELECT * FROM (VALUES
            ('2024-01-15'::DATE),
            ('2024-03-01'::DATE),
            ('2024-02-10'::DATE)
        ) AS t(date_col)";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 1);

        let col = &schema[0];
        assert_eq!(col.name, "date_col");
        // Date is now continuous (not discrete)
        assert!(!col.is_discrete);
        // Date min/max as numeric days since epoch (for range computation)
        assert_eq!(col.min, Some(ArrayElement::Number(19737.0))); // 2024-01-15
        assert_eq!(col.max, Some(ArrayElement::Number(19783.0))); // 2024-03-01
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_empty_table() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        // Empty result set (WHERE 1=0 filters out all rows)
        let query = "SELECT 1 as x, 2 as y WHERE 1=0";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 2);

        // For empty tables, min/max should be None
        let col_x = &schema[0];
        assert_eq!(col_x.name, "x");
        assert!(col_x.min.is_none());
        assert!(col_x.max.is_none());

        let col_y = &schema[1];
        assert_eq!(col_y.name, "y");
        assert!(col_y.min.is_none());
        assert!(col_y.max.is_none());
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_mixed_column_types() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        let query = "SELECT * FROM (VALUES
            (1, 'A', true, '2024-01-01'::DATE),
            (5, 'C', false, '2024-12-31'::DATE),
            (3, 'B', true, '2024-06-15'::DATE)
        ) AS t(num, str_col, bool_col, date_col)";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 4);

        // Numeric column
        let col_num = &schema[0];
        assert_eq!(col_num.name, "num");
        assert!(!col_num.is_discrete);
        assert_eq!(col_num.min, Some(ArrayElement::Number(1.0)));
        assert_eq!(col_num.max, Some(ArrayElement::Number(5.0)));

        // String column (discrete)
        let col_str = &schema[1];
        assert_eq!(col_str.name, "str_col");
        assert!(col_str.is_discrete);
        assert_eq!(col_str.min, Some(ArrayElement::String("A".to_string())));
        assert_eq!(col_str.max, Some(ArrayElement::String("C".to_string())));

        // Boolean column (discrete)
        let col_bool = &schema[2];
        assert_eq!(col_bool.name, "bool_col");
        assert!(col_bool.is_discrete);
        assert_eq!(col_bool.min, Some(ArrayElement::Boolean(false)));
        assert_eq!(col_bool.max, Some(ArrayElement::Boolean(true)));

        // Date column (continuous) - stored as numeric days since epoch
        let col_date = &schema[3];
        assert_eq!(col_date.name, "date_col");
        assert!(!col_date.is_discrete);
        assert_eq!(col_date.min, Some(ArrayElement::Number(19723.0))); // 2024-01-01
        assert_eq!(col_date.max, Some(ArrayElement::Number(20088.0))); // 2024-12-31
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_fetch_schema_with_nulls() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let execute_query = |sql: &str| reader.execute(sql);
        let query = "SELECT * FROM (VALUES
            (1, 'A'),
            (NULL, 'B'),
            (5, NULL),
            (3, 'C')
        ) AS t(num, str_col)";

        let schema = fetch_schema_for_test(query, &execute_query).unwrap();

        assert_eq!(schema.len(), 2);

        // Numeric column - nulls should be ignored for min/max
        let col_num = &schema[0];
        assert_eq!(col_num.name, "num");
        assert_eq!(col_num.min, Some(ArrayElement::Number(1.0)));
        assert_eq!(col_num.max, Some(ArrayElement::Number(5.0)));

        // String column - nulls should be ignored for min/max
        let col_str = &schema[1];
        assert_eq!(col_str.name, "str_col");
        assert_eq!(col_str.min, Some(ArrayElement::String("A".to_string())));
        assert_eq!(col_str.max, Some(ArrayElement::String("C".to_string())));
    }

    // =========================================================================
    // Type Casting Tests
    // =========================================================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_type_casting_integration_string_to_date() {
        // Test that STRING column with DATE scale gets properly cast
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT * FROM (VALUES
                ('2024-01-01', 100),
                ('2024-01-15', 200),
                ('2024-02-01', 150)
            ) AS t(date_str, value)
            VISUALISE
            DRAW point MAPPING date_str AS x, value AS y
            SCALE x VIA date
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Verify the data was prepared successfully
        // With aesthetic-named columns, data is in layer_key(0)
        let df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(df.height(), 3);

        // The scale should be properly resolved
        let spec = &result.specs[0];
        let x_scale = spec.scales.iter().find(|s| s.aesthetic == "x").unwrap();
        assert!(x_scale.scale_type.is_some());
    }

    // =========================================================================
    // create_missing_scales Tests
    // =========================================================================

    #[test]
    fn test_create_missing_scales_from_global_mapping() {
        // Test that scales are created for aesthetics in layer mappings
        // (global mappings are merged into layers before create_missing_scales is called)
        use crate::plot::{AestheticValue, Geom, Layer, Plot};

        let mut spec = Plot::new();
        // Simulate post-merge state: mappings are in layer
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // No explicit scales defined
        assert!(spec.scales.is_empty());

        create_missing_scales(&mut spec);

        // Should have created scales for x and y
        assert_eq!(spec.scales.len(), 2);
        let scale_aesthetics: HashSet<&str> =
            spec.scales.iter().map(|s| s.aesthetic.as_str()).collect();
        assert!(scale_aesthetics.contains("x"));
        assert!(scale_aesthetics.contains("y"));

        // Both should have no explicit scale_type (will be inferred from data)
        for scale in &spec.scales {
            if scale.aesthetic == "x" || scale.aesthetic == "y" {
                assert!(scale.scale_type.is_none());
            }
        }
    }

    #[test]
    fn test_create_missing_scales_from_layer_mapping() {
        // Test that scales are created for aesthetics in layer mappings
        use crate::plot::{AestheticValue, Geom, Layer, Plot};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("col_x"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("col_y"));
        spec.layers.push(layer);

        assert!(spec.scales.is_empty());

        create_missing_scales(&mut spec);

        // Should have created scales for x and y
        assert_eq!(spec.scales.len(), 2);
        let scale_aesthetics: HashSet<&str> =
            spec.scales.iter().map(|s| s.aesthetic.as_str()).collect();
        assert!(scale_aesthetics.contains("x"));
        assert!(scale_aesthetics.contains("y"));
    }

    #[test]
    fn test_create_missing_scales_preserves_explicit_scales() {
        // Test that explicit scales are not overwritten
        // (global mappings are merged into layers before create_missing_scales is called)
        use crate::plot::{AestheticValue, Geom, Layer, Plot, Scale, ScaleType};

        let mut spec = Plot::new();
        // Simulate post-merge state: mappings are in layer
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"));
        spec.layers.push(layer);

        // Add explicit scale for x with specific type
        let mut x_scale = Scale::new("x");
        x_scale.scale_type = Some(ScaleType::continuous());
        spec.scales.push(x_scale);

        create_missing_scales(&mut spec);

        // Should have 2 scales now (x preserved, y created)
        assert_eq!(spec.scales.len(), 2);

        // Find x scale - should preserve the explicit type
        let x_scale = spec.scales.iter().find(|s| s.aesthetic == "x").unwrap();
        assert_eq!(x_scale.scale_type, Some(ScaleType::continuous()));

        // Find y scale - should have no type (will be inferred)
        let y_scale = spec.scales.iter().find(|s| s.aesthetic == "y").unwrap();
        assert!(y_scale.scale_type.is_none());
    }

    #[test]
    fn test_create_missing_scales_includes_literals() {
        // Test that scales are created for literal aesthetic values too
        use crate::plot::{AestheticValue, Geom, Layer, LiteralValue, Plot};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::point())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("col_x"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("col_y"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(LiteralValue::String("red".to_string())),
            );
        spec.layers.push(layer);

        create_missing_scales(&mut spec);

        // Should have scales for x, y, and color (even though color is a literal)
        assert_eq!(spec.scales.len(), 3);
        let scale_aesthetics: HashSet<&str> =
            spec.scales.iter().map(|s| s.aesthetic.as_str()).collect();
        assert!(scale_aesthetics.contains("x"));
        assert!(scale_aesthetics.contains("y"));
        assert!(scale_aesthetics.contains("color"));
    }

    #[test]
    fn test_create_missing_scales_identity_for_text() {
        // Test that text/label/group aesthetics get Identity scale type
        use crate::plot::{AestheticValue, Geom, Layer, Plot, ScaleType};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::text())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("col_x"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("col_y"))
            .with_aesthetic(
                "label".to_string(),
                AestheticValue::standard_column("text_col"),
            );
        spec.layers.push(layer);

        create_missing_scales(&mut spec);

        // Find label scale - should have Identity type
        let label_scale = spec.scales.iter().find(|s| s.aesthetic == "label").unwrap();
        assert_eq!(label_scale.scale_type, Some(ScaleType::identity()));

        // x and y should have no type (will be inferred from data)
        let x_scale = spec.scales.iter().find(|s| s.aesthetic == "x").unwrap();
        assert!(x_scale.scale_type.is_none());
    }

    #[test]
    fn test_create_missing_scales_normalizes_aesthetic_families() {
        // Test that variant aesthetics (xmin, xmax, ymin, ymax) are normalized to primary
        use crate::plot::{AestheticValue, Geom, Layer, Plot};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::ribbon())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("ymin".to_string(), AestheticValue::standard_column("lower"))
            .with_aesthetic("ymax".to_string(), AestheticValue::standard_column("upper"));
        spec.layers.push(layer);

        create_missing_scales(&mut spec);

        // Should have scales for x and y only (ymin/ymax normalized to y)
        assert_eq!(spec.scales.len(), 2);
        let scale_aesthetics: HashSet<&str> =
            spec.scales.iter().map(|s| s.aesthetic.as_str()).collect();
        assert!(scale_aesthetics.contains("x"));
        assert!(scale_aesthetics.contains("y")); // ymin/ymax normalized to y
        assert!(!scale_aesthetics.contains("ymin")); // Should not create separate scale
        assert!(!scale_aesthetics.contains("ymax")); // Should not create separate scale
    }

    #[test]
    fn test_create_missing_scales_from_remappings() {
        // Test that scales are created for aesthetics in remappings
        use crate::plot::{AestheticValue, Geom, Layer, Plot};

        let mut spec = Plot::new();
        let mut layer = Layer::new(Geom::bar());
        layer
            .mappings
            .insert("x", AestheticValue::standard_column("category"));
        // Remapping: stat column "count" maps to "y" aesthetic
        layer
            .remappings
            .insert("y", AestheticValue::standard_column("count"));
        spec.layers.push(layer);

        create_missing_scales(&mut spec);

        // Should have scales for x and y (from remapping)
        let scale_aesthetics: HashSet<&str> =
            spec.scales.iter().map(|s| s.aesthetic.as_str()).collect();
        assert!(scale_aesthetics.contains("x"));
        assert!(scale_aesthetics.contains("y"));
    }

    #[test]
    fn test_gets_default_scale() {
        // Aesthetics that SHOULD get default scale (type inferred from data)
        let should_get_scale = [
            // Position aesthetics
            "x", "y", "xmin", "xmax", "ymin", "ymax", "xend", "yend", "x2", "y2",
            // Color aesthetics (color/colour/col are split to fill/stroke)
            "fill", "stroke",
            // Size, opacity, shape, linetype
            "size", "linewidth", "opacity", "shape", "linetype",
        ];
        for aes in should_get_scale {
            assert!(gets_default_scale(aes), "'{}' should get default scale", aes);
        }

        // Aesthetics that should NOT get default scale (use Identity)
        let should_not_get_scale = ["text", "label", "group", "detail", "tooltip", "unknown_aesthetic"];
        for aes in should_not_get_scale {
            assert!(!gets_default_scale(aes), "'{}' should NOT get default scale", aes);
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_automatic_temporal_scale_without_explicit_scale() {
        // Test end-to-end: date column without explicit SCALE should get temporal transform
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE temporal_test AS SELECT * FROM (VALUES
                    ('2024-01-01'::DATE, 100),
                    ('2024-02-01'::DATE, 200),
                    ('2024-03-01'::DATE, 300)
                ) AS t(date, value)",
                duckdb::params![],
            )
            .unwrap();

        // Query without explicit SCALE clause
        let query = r#"
            SELECT * FROM temporal_test
            VISUALISE date AS x, value AS y
            DRAW line
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Should have x scale automatically created
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .expect("x scale should be automatically created");

        // Scale type should be inferred as continuous (for temporal)
        assert!(x_scale.scale_type.is_some());
        assert_eq!(
            x_scale.scale_type.as_ref().unwrap().name(),
            "continuous",
            "Date column should infer continuous scale type"
        );

        // Transform should be Date (temporal)
        assert!(x_scale.transform.is_some());
        assert_eq!(
            x_scale.transform.as_ref().unwrap().name(),
            "date",
            "Date column should infer Date transform"
        );
    }

    #[test]
    fn test_discrete_scale_triggers_partition_by() {
        // Test that explicit SCALE DISCRETE on an integer column adds it to partition_by
        use crate::plot::{AestheticValue, Geom, Layer, Plot, Scale, ScaleType};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("group_id"),
            );
        spec.layers.push(layer);

        // Create a discrete scale for color (even though group_id might be an integer)
        let mut color_scale = Scale::new("color");
        color_scale.scale_type = Some(ScaleType::discrete());
        spec.scales.push(color_scale);

        // Schema where group_id is NOT discrete (integer column)
        let schema = vec![
            ColumnInfo {
                name: "date".to_string(),
                dtype: polars::prelude::DataType::Date,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "value".to_string(),
                dtype: polars::prelude::DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "group_id".to_string(),
                dtype: polars::prelude::DataType::Int64,
                is_discrete: false, // Integer column, NOT discrete by schema
                min: None,
                max: None,
            },
        ];

        // Before: partition_by should be empty
        assert!(spec.layers[0].partition_by.is_empty());

        add_discrete_columns_to_partition_by(&mut spec.layers, &[schema], &spec.scales);

        // After: group_id should be added to partition_by because SCALE DISCRETE color
        assert!(
            spec.layers[0]
                .partition_by
                .contains(&"group_id".to_string()),
            "Integer column with explicit SCALE DISCRETE should be added to partition_by"
        );
    }

    #[test]
    fn test_continuous_scale_prevents_partition_by() {
        // Test that explicit SCALE CONTINUOUS on a string column does NOT add it to partition_by
        use crate::plot::{AestheticValue, Geom, Layer, Plot, Scale, ScaleType};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category"),
            );
        spec.layers.push(layer);

        // Create a continuous scale for color (overriding schema's discrete)
        let mut color_scale = Scale::new("color");
        color_scale.scale_type = Some(ScaleType::continuous());
        spec.scales.push(color_scale);

        // Schema where category IS discrete (string column)
        let schema = vec![
            ColumnInfo {
                name: "date".to_string(),
                dtype: polars::prelude::DataType::Date,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "value".to_string(),
                dtype: polars::prelude::DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "category".to_string(),
                dtype: polars::prelude::DataType::String,
                is_discrete: true, // String column, discrete by schema
                min: None,
                max: None,
            },
        ];

        add_discrete_columns_to_partition_by(&mut spec.layers, &[schema], &spec.scales);

        // category should NOT be added because SCALE CONTINUOUS overrides schema
        assert!(
            !spec.layers[0]
                .partition_by
                .contains(&"category".to_string()),
            "String column with explicit SCALE CONTINUOUS should NOT be added to partition_by"
        );
    }

    #[test]
    fn test_identity_scale_falls_back_to_schema() {
        // Test that Identity scale falls back to schema's is_discrete
        use crate::plot::{AestheticValue, Geom, Layer, Plot, Scale, ScaleType};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("category"),
            );
        spec.layers.push(layer);

        // Create an identity scale for color
        let mut color_scale = Scale::new("color");
        color_scale.scale_type = Some(ScaleType::identity());
        spec.scales.push(color_scale);

        // Schema where category IS discrete
        let schema = vec![
            ColumnInfo {
                name: "date".to_string(),
                dtype: polars::prelude::DataType::Date,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "value".to_string(),
                dtype: polars::prelude::DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "category".to_string(),
                dtype: polars::prelude::DataType::String,
                is_discrete: true,
                min: None,
                max: None,
            },
        ];

        add_discrete_columns_to_partition_by(&mut spec.layers, &[schema], &spec.scales);

        // category SHOULD be added because Identity falls back to schema (which says discrete)
        assert!(
            spec.layers[0]
                .partition_by
                .contains(&"category".to_string()),
            "Discrete column with Identity scale should be added to partition_by"
        );
    }

    #[test]
    fn test_binned_scale_triggers_partition_by() {
        // Test that SCALE BINNED on a continuous column adds it to partition_by
        // (binned data creates discrete categories/bins)
        use crate::plot::{AestheticValue, Geom, Layer, Plot, Scale, ScaleType};

        let mut spec = Plot::new();
        let layer = Layer::new(Geom::line())
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("value"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::standard_column("temperature"),
            );
        spec.layers.push(layer);

        // Create a binned scale for color
        let mut color_scale = Scale::new("color");
        color_scale.scale_type = Some(ScaleType::binned());
        spec.scales.push(color_scale);

        // Schema where temperature is NOT discrete (continuous float column)
        let schema = vec![
            ColumnInfo {
                name: "date".to_string(),
                dtype: polars::prelude::DataType::Date,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "value".to_string(),
                dtype: polars::prelude::DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "temperature".to_string(),
                dtype: polars::prelude::DataType::Float64,
                is_discrete: false, // Continuous column
                min: None,
                max: None,
            },
        ];

        // Before: partition_by should be empty
        assert!(spec.layers[0].partition_by.is_empty());

        add_discrete_columns_to_partition_by(&mut spec.layers, &[schema], &spec.scales);

        // After: temperature should be added to partition_by because SCALE BINNED creates categories
        assert!(
            spec.layers[0]
                .partition_by
                .contains(&"temperature".to_string()),
            "Continuous column with SCALE BINNED should be added to partition_by"
        );
    }

    // =============================================================================
    // Label Template Tests
    // =============================================================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_label_template_continuous_with_breaks() {
        // Test that wildcard template is applied to break values for continuous scales
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES (0, 10), (50, 20), (100, 30)) AS t(x, y)
            VISUALISE x AS x, y AS y
            DRAW point
            SCALE CONTINUOUS x SETTING breaks => 3 RENAMING * => '{} units'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Find the x scale
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .unwrap();

        // Check that label_mapping was populated from the template
        assert!(
            x_scale.label_mapping.is_some(),
            "label_mapping should be set"
        );
        let label_mapping = x_scale.label_mapping.as_ref().unwrap();

        // The exact break values depend on pretty breaks algorithm, but they should have ' units' suffix
        for (key, label) in label_mapping {
            if let Some(label_str) = label {
                assert!(
                    label_str.ends_with(" units"),
                    "Label '{}' for key '{}' should end with ' units'",
                    label_str,
                    key
                );
            }
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_label_template_discrete_uppercase() {
        // Test that wildcard template is applied to discrete scale domain values
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('north', 10), ('south', 20), ('east', 30)) AS t(region, value)
            VISUALISE region AS x, value AS y
            DRAW bar
            SCALE DISCRETE x RENAMING * => '{:UPPER}'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Find the x scale
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .unwrap();

        // Check that label_mapping was populated from the template
        assert!(
            x_scale.label_mapping.is_some(),
            "label_mapping should be set"
        );
        let label_mapping = x_scale.label_mapping.as_ref().unwrap();

        // Check uppercase transformations
        assert_eq!(
            label_mapping.get("north"),
            Some(&Some("NORTH".to_string())),
            "north should be transformed to NORTH"
        );
        assert_eq!(
            label_mapping.get("south"),
            Some(&Some("SOUTH".to_string())),
            "south should be transformed to SOUTH"
        );
        assert_eq!(
            label_mapping.get("east"),
            Some(&Some("EAST".to_string())),
            "east should be transformed to EAST"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_label_template_explicit_takes_priority() {
        // Test that explicit mappings take priority over template
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(cat, value)
            VISUALISE cat AS x, value AS y
            DRAW bar
            SCALE DISCRETE x RENAMING 'A' => 'Alpha', * => 'Category {}'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Find the x scale
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .unwrap();

        // Check that label_mapping was populated
        assert!(
            x_scale.label_mapping.is_some(),
            "label_mapping should be set"
        );
        let label_mapping = x_scale.label_mapping.as_ref().unwrap();

        // A should have explicit mapping
        assert_eq!(
            label_mapping.get("A"),
            Some(&Some("Alpha".to_string())),
            "A should keep explicit mapping 'Alpha'"
        );

        // B and C should have template-generated labels
        assert_eq!(
            label_mapping.get("B"),
            Some(&Some("Category B".to_string())),
            "B should get template label 'Category B'"
        );
        assert_eq!(
            label_mapping.get("C"),
            Some(&Some("Category C".to_string())),
            "C should get template label 'Category C'"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_label_template_title_case() {
        // Test Title case transformation
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('us east', 10), ('eu west', 20)) AS t(region, value)
            VISUALISE region AS x, value AS y
            DRAW bar
            SCALE DISCRETE x RENAMING * => 'Region: {:Title}'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Find the x scale
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .unwrap();

        // Check that label_mapping was populated
        assert!(
            x_scale.label_mapping.is_some(),
            "label_mapping should be set"
        );
        let label_mapping = x_scale.label_mapping.as_ref().unwrap();

        assert_eq!(
            label_mapping.get("us east"),
            Some(&Some("Region: Us East".to_string())),
            "Should apply Title case transformation with prefix"
        );
        assert_eq!(
            label_mapping.get("eu west"),
            Some(&Some("Region: Eu West".to_string())),
            "Should apply Title case transformation with prefix"
        );
    }

    // =============================================================================
    // Scale Type Coercion Tests
    // =============================================================================

    #[test]
    fn test_infer_scale_target_type() {
        use crate::plot::scale::{ScaleType, Transform};

        // Helper to create a base scale
        fn make_scale(
            scale_type: ScaleType,
            input_range: Option<Vec<ArrayElement>>,
            transform: Option<Transform>,
        ) -> Scale {
            let explicit_transform = transform.is_some();
            Scale {
                aesthetic: "x".to_string(),
                scale_type: Some(scale_type),
                input_range,
                explicit_input_range: false,
                output_range: None,
                transform,
                explicit_transform,
                properties: HashMap::new(),
                resolved: false,
                label_mapping: None,
                label_template: None,
            }
        }

        // Test cases: (scale_type, input_range, transform, expected_target_type, description)
        let test_cases: Vec<(ScaleType, Option<Vec<ArrayElement>>, Option<Transform>, Option<ArrayElementType>, &str)> = vec![
            // Discrete scales infer type from input_range
            (
                ScaleType::discrete(),
                Some(vec![ArrayElement::Boolean(true), ArrayElement::Boolean(false)]),
                None,
                Some(ArrayElementType::Boolean),
                "Discrete with boolean range → Boolean",
            ),
            (
                ScaleType::discrete(),
                Some(vec![ArrayElement::String("A".to_string()), ArrayElement::String("B".to_string())]),
                None,
                Some(ArrayElementType::String),
                "Discrete with string range → String",
            ),
            // Continuous scales infer from transform
            (
                ScaleType::continuous(),
                None,
                Some(Transform::date()),
                Some(ArrayElementType::Date),
                "Continuous with date transform → Date",
            ),
            (
                ScaleType::continuous(),
                None,
                Some(Transform::log()),
                Some(ArrayElementType::Number),
                "Continuous with log transform → Number",
            ),
            // Scales that return None (no coercion)
            (
                ScaleType::binned(),
                None,
                None,
                None,
                "Binned → None (binning in SQL)",
            ),
            (
                ScaleType::identity(),
                None,
                None,
                None,
                "Identity → None (no coercion)",
            ),
            (
                ScaleType::continuous(),
                None,
                None,
                None,
                "Continuous without transform → None",
            ),
        ];

        for (scale_type, input_range, transform, expected, description) in test_cases {
            let scale = make_scale(scale_type, input_range, transform);
            let result = infer_scale_target_type(&scale);
            assert_eq!(result, expected, "{}", description);
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_coerce_column_string_to_boolean() {
        use polars::prelude::{NamedFrom, Series};

        // Create a DataFrame with string column
        let series = Series::new("flag".into(), vec!["true", "false", "TRUE", "FALSE"]);
        let df = DataFrame::new(vec![series.into_column()]).unwrap();

        // Coerce to boolean
        let result = coerce_column_to_type(&df, "flag", ArrayElementType::Boolean).unwrap();

        // Verify the column was converted
        let col = result.column("flag").unwrap();
        assert_eq!(col.dtype(), &polars::prelude::DataType::Boolean);

        let bool_series = col.as_materialized_series();
        let bool_vec: Vec<Option<bool>> = bool_series.bool().unwrap().into_iter().collect();
        assert_eq!(
            bool_vec,
            vec![Some(true), Some(false), Some(true), Some(false)]
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_coerce_column_string_to_boolean_error() {
        use polars::prelude::{NamedFrom, Series};

        // Create a DataFrame with invalid string values
        let series = Series::new("flag".into(), vec!["true", "maybe", "false"]);
        let df = DataFrame::new(vec![series.into_column()]).unwrap();

        // Coerce should fail
        let result = coerce_column_to_type(&df, "flag", ArrayElementType::Boolean);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            GgsqlError::ValidationError(msg) => {
                assert!(msg.contains("Cannot coerce string 'maybe' to boolean"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_coerce_column_string_to_date() {
        use polars::prelude::{NamedFrom, Series};

        // Create a DataFrame with date strings
        let series = Series::new("date".into(), vec!["2024-01-15", "2024-06-30"]);
        let df = DataFrame::new(vec![series.into_column()]).unwrap();

        // Coerce to date
        let result = coerce_column_to_type(&df, "date", ArrayElementType::Date).unwrap();

        // Verify the column was converted
        let col = result.column("date").unwrap();
        assert_eq!(col.dtype(), &polars::prelude::DataType::Date);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_coerce_column_already_correct_type() {
        use polars::prelude::{NamedFrom, Series};

        // Create a DataFrame with boolean column
        let series = Series::new("flag".into(), vec![true, false]);
        let df = DataFrame::new(vec![series.into_column()]).unwrap();

        // Coerce to boolean (same type) should be a no-op
        let result = coerce_column_to_type(&df, "flag", ArrayElementType::Boolean).unwrap();

        // Verify the column is still boolean
        let col = result.column("flag").unwrap();
        assert_eq!(col.dtype(), &polars::prelude::DataType::Boolean);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_discrete_boolean_range_coerces_string_column() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Query that produces string "true"/"false" values with discrete boolean scale
        let query = r#"
            SELECT
                n,
                n * 10 as value,
                CASE WHEN n % 2 = 0 THEN 'true' ELSE 'false' END as is_even
            FROM generate_series(1, 6) AS t(n)
            VISUALISE n AS x, value AS y, is_even AS color
            DRAW point
            SCALE DISCRETE color FROM [true, false]
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Verify the scale has boolean input range
        let color_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "fill" || s.aesthetic == "stroke")
            .expect("Should have fill or stroke scale");

        // The input_range should contain boolean values (coerced from string column)
        let input_range = color_scale
            .input_range
            .as_ref()
            .expect("Should have input_range");
        for elem in input_range {
            assert!(
                matches!(elem, ArrayElement::Boolean(_)),
                "Input range should contain Boolean values, got {:?}",
                elem
            );
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_ordinal_numeric_column_resolves_unique_values() {
        // Test that ordinal scale with numeric column resolves unique values, not min/max
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            VISUALISE Ozone AS x, Temp AS y FROM ggsql:airquality
            DRAW point
                MAPPING Month AS color
            SCALE ORDINAL color TO lapaz
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Find the fill scale (color gets split to fill/stroke)
        let fill_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "fill")
            .expect("Should have fill scale");

        // Verify it's ORDINAL
        assert!(
            matches!(
                fill_scale.scale_type.as_ref().map(|st| st.scale_type_kind()),
                Some(crate::plot::scale::ScaleTypeKind::Ordinal)
            ),
            "Should be ORDINAL scale type, got {:?}",
            fill_scale.scale_type
        );

        // Verify uses_discrete_input_range returns true
        assert!(
            fill_scale
                .scale_type
                .as_ref()
                .map(|st| st.uses_discrete_input_range())
                .unwrap_or(false),
            "Ordinal.uses_discrete_input_range() should return true"
        );

        // Verify input_range has unique month values, not just min/max
        let input_range = fill_scale
            .input_range
            .as_ref()
            .expect("Should have input_range");

        // The airquality dataset has months 5-9, so should have 5 unique values
        assert!(
            input_range.len() > 2,
            "Ordinal should have unique values, not just min/max. Got {} values: {:?}",
            input_range.len(),
            input_range
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_binned_date_scale_with_visualise_from() {
        // Regression test: binned date scale should apply binning transformation to Date columns
        // Bug: The StatResult::Identity branch was incorrectly returning None (use global directly)
        // even when pre-stat transform (binning) was applied to the query.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            VISUALISE Date AS x, Temp AS y FROM ggsql:airquality
            DRAW boxplot
            SCALE BINNED x VIA date SETTING breaks => 'month'
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Verify the scale has breaks computed
        let x_scale = result.specs[0]
            .scales
            .iter()
            .find(|s| s.aesthetic == "x")
            .expect("Should have x scale");

        let breaks = x_scale.properties.get("breaks");
        assert!(
            matches!(breaks, Some(ParameterValue::Array(_))),
            "Breaks should be computed as Array, got: {:?}",
            breaks
        );

        // Verify we have layer data with transformed x values (not using global directly)
        // This is the key assertion - if binning is applied, layer data must be created
        let layer_key = naming::layer_key(0);
        assert!(
            result.data.contains_key(&layer_key),
            "Should have layer data from binning transformation. Available keys: {:?}",
            result.data.keys().collect::<Vec<_>>()
        );

        // Verify the x column (was Date, renamed to prefixed x) has binned values (bin centers, not original dates)
        let layer_data = result.data.get(&layer_key).unwrap();
        let x_col_name = naming::aesthetic_column("x");
        let date_col = layer_data.column(&x_col_name).unwrap_or_else(|_| {
            panic!(
                "Should have '{}' column (was Date, renamed to prefixed aesthetic)",
                x_col_name
            )
        });
        // Get unique values - should be fewer than original dates due to binning
        let series = date_col.as_materialized_series();
        let unique_count = series.n_unique().unwrap();
        // airquality has ~150 rows spanning May-September 1973
        // With monthly breaks, we should have at most 6 unique bin centers
        assert!(
            unique_count <= 6,
            "Binned x column should have at most 6 unique values (monthly bins), got {}",
            unique_count
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_query_deduplication_identical_layers() {
        // Test that multiple layers work correctly
        // Each layer has its own data_key which may be shared if queries are identical
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Two identical point layers
        let query = r#"
            SELECT * FROM (VALUES
                (1, 10), (2, 20), (3, 30)
            ) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = prepare_data(query, &reader).unwrap();

        // Both layers should have data_keys
        let layer0_data_key = result.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer 0 should have data_key");
        let layer1_data_key = result.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("Layer 1 should have data_key");

        // Both layer data should exist in the data map
        assert!(
            result.data.contains_key(layer0_data_key),
            "Should have layer 0 data at key {}",
            layer0_data_key
        );
        assert!(
            result.data.contains_key(layer1_data_key),
            "Should have layer 1 data at key {}",
            layer1_data_key
        );

        // Both should have 3 rows
        let layer0_data = result.data.get(layer0_data_key).unwrap();
        let layer1_data = result.data.get(layer1_data_key).unwrap();
        assert_eq!(layer0_data.height(), 3);
        assert_eq!(layer1_data.height(), 3);

        // Verify the data content is correct (DuckDB may return i32 or i64 depending on literals)
        let x_col = layer0_data
            .column(&naming::aesthetic_column("x"))
            .unwrap()
            .as_materialized_series();
        // Cast to i64 to handle both i32 and i64 input types
        let x_values: Vec<i64> = x_col
            .cast(&polars::prelude::DataType::Int64)
            .unwrap()
            .i64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(x_values, vec![1, 2, 3]);
    }
}
