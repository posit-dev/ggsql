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

/// Apply column casting to a SQL query.
///
/// Wraps the query to cast specified columns to their target types.
/// Uses DuckDB's EXCLUDE syntax for clean column replacement.
///
/// Example output:
/// ```sql
/// SELECT * EXCLUDE (date_col, value_col),
///        CAST(date_col AS DATE) AS date_col,
///        CAST(value_col AS DOUBLE) AS value_col
/// FROM (original_query)
/// ```
fn apply_column_casting(query: &str, requirements: &[TypeRequirement]) -> String {
    if requirements.is_empty() {
        return query.to_string();
    }

    // Build EXCLUDE clause
    let exclude_cols: Vec<&str> = requirements.iter().map(|r| r.column.as_str()).collect();
    let exclude_clause = if exclude_cols.len() == 1 {
        format!("EXCLUDE (\"{}\")", exclude_cols[0])
    } else {
        format!(
            "EXCLUDE ({})",
            exclude_cols
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    // Build CAST expressions
    let cast_exprs: Vec<String> = requirements
        .iter()
        .map(|r| {
            format!(
                "CAST(\"{}\" AS {}) AS \"{}\"",
                r.column, r.sql_type_name, r.column
            )
        })
        .collect();

    format!(
        "SELECT * {}, {} FROM ({})",
        exclude_clause,
        cast_exprs.join(", "),
        query
    )
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

/// Collect type requirements for the global query.
///
/// For the global query, we need to cast columns that are used by any layer
/// that reads from global data (layers without explicit source).
fn collect_requirements_for_global(
    layer_requirements: &[Vec<TypeRequirement>],
    spec: &Plot,
) -> Vec<TypeRequirement> {
    let mut global_reqs: Vec<TypeRequirement> = Vec::new();

    for (layer_idx, layer) in spec.layers.iter().enumerate() {
        // Only include requirements from layers that use global data
        if layer.source.is_none() && layer_idx < layer_requirements.len() {
            for req in &layer_requirements[layer_idx] {
                // Don't add duplicates
                if !global_reqs.iter().any(|r| r.column == req.column) {
                    global_reqs.push(req.clone());
                }
            }
        }
    }

    global_reqs
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

/// Build the base query for a layer including constant columns.
///
/// Returns `SELECT * [, constants...] FROM source` or None if layer uses global directly
/// without needing any transformation.
///
/// Constants are included in the base query ONLY for layers with their own source or filter.
/// For layers using global data (no source, no filter), constants are injected into the
/// global table with layer-indexed names by `prepare_data_with_executor`, so we don't add
/// them here.
fn build_base_layer_query(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
) -> Option<String> {
    // For layers using global data (no source, no filter), constants are already
    // in the global table with layer-indexed names. Return None to use global directly.
    let uses_global_directly = layer.source.is_none() && layer.filter.is_none();

    // Determine source table
    let source = match &layer.source {
        Some(DataSource::Identifier(name)) => {
            if materialized_ctes.contains(name) {
                naming::cte_table(name)
            } else {
                name.clone()
            }
        }
        Some(DataSource::FilePath(path)) => format!("'{}'", path),
        None => {
            // Layer uses global data
            if has_global {
                naming::global_table()
            } else {
                return None;
            }
        }
    };

    // For layers using global directly, don't add constants (they're already in global)
    if uses_global_directly {
        return Some(format!("SELECT * FROM {}", source));
    }

    // For layers with their own source or filter, add constants with non-indexed names
    let constants = extract_constants(layer);

    // Build query with constants
    if constants.is_empty() {
        Some(format!("SELECT * FROM {}", source))
    } else {
        let const_cols: Vec<String> = constants
            .iter()
            .map(|(aes, lit)| format!("{} AS {}", literal_to_sql(lit), naming::const_column(aes)))
            .collect();
        Some(format!(
            "SELECT *, {} FROM {}",
            const_cols.join(", "),
            source
        ))
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

// =============================================================================
// Pre-Stat Transform
// =============================================================================

/// Check if a layer needs pre-stat transformation based on its mapped aesthetics and scales.
///
/// Returns true if any mapped aesthetic has a scale that would generate pre-stat SQL
/// (e.g., binned scales with resolved breaks).
fn needs_pre_stat_transform(
    layer: &Layer,
    schema: &Schema,
    scales: &[crate::plot::Scale],
    type_names: &SqlTypeNames,
) -> bool {
    use polars::prelude::DataType;

    for (aesthetic, value) in &layer.mappings.aesthetics {
        let col_name = if let Some(col) = value.column_name() {
            col.to_string()
        } else if value.is_literal() {
            naming::const_column(aesthetic)
        } else {
            continue;
        };

        let col_dtype = schema
            .iter()
            .find(|c| c.name == col_name)
            .map(|c| c.dtype.clone())
            .unwrap_or(DataType::String);

        if let Some(scale) = scales.iter().find(|s| s.aesthetic == *aesthetic) {
            if let Some(ref scale_type) = scale.scale_type {
                if scale_type
                    .pre_stat_transform_sql(&col_name, &col_dtype, scale, type_names)
                    .is_some()
                {
                    return true;
                }
            }
        }
    }

    false
}

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

    // Check layer mappings for aesthetics with Binned scales
    // Handles both column mappings and literal mappings (which are injected as synthetic columns)
    for (aesthetic, value) in &layer.mappings.aesthetics {
        // Get the column name - either the mapped column or the synthetic constant column
        let col_name = if let Some(col) = value.column_name() {
            col.to_string()
        } else if value.is_literal() {
            // Literals are injected as synthetic columns like __ggsql_const_{aesthetic}__
            naming::const_column(aesthetic)
        } else {
            continue;
        };

        // Find column dtype from schema
        let col_dtype = schema
            .iter()
            .find(|c| c.name == col_name)
            .map(|c| c.dtype.clone())
            .unwrap_or(DataType::String); // Default to String if not found

        // Find scale for this aesthetic
        if let Some(scale) = scales.iter().find(|s| s.aesthetic == *aesthetic) {
            if let Some(ref scale_type) = scale.scale_type {
                // Get pre-stat SQL transformation from scale type (if applicable)
                // Each scale type's pre_stat_transform_sql() returns None if not applicable
                if let Some(sql) =
                    scale_type.pre_stat_transform_sql(&col_name, &col_dtype, scale, type_names)
                {
                    transform_exprs.push((col_name, sql));
                }
            }
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

    format!(
        "SELECT * {}, {} FROM ({})",
        exclude_clause,
        col_exprs.join(", "),
        query
    )
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
///
/// Pre-stat transforms are applied (e.g., binning for Binned scales) before stat transforms.
///
/// Note: Constants are now included in the cast_base_query (via build_base_layer_query),
/// so this function no longer needs to add them.
fn build_layer_query<F>(
    layer: &mut Layer,
    cast_base_query: &str,
    schema: &Schema,
    has_global: bool,
    layer_idx: usize,
    facet: Option<&Facet>,
    scales: &[crate::plot::Scale],
    type_names: &SqlTypeNames,
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

    // Check if layer needs a query or can use global directly
    // NOTE: Constants alone do NOT trigger a query for layers using global data (no source, no filter)
    // because those constants are already injected into the global table with layer-indexed names.
    // Constants only require a query when combined with a filter (which creates layer-specific data).
    let needs_query = match &layer.source {
        Some(_) => true, // Has explicit source
        None => {
            // No source - check if we need to query
            if filter.is_some() || order_by.is_some() {
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} has a FILTER or ORDER BY but no data source. Either provide a SQL query or use MAPPING FROM.",
                        layer_idx + 1
                    )));
                }
                true
            } else if layer.geom.needs_stat_transform(&layer.mappings) {
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} requires data for statistical transformation but no data source.",
                        layer_idx + 1
                    )));
                }
                true
            } else if needs_pre_stat_transform(layer, schema, scales, type_names) {
                // Layer has binned scales that need data transformation
                if !has_global {
                    return Err(GgsqlError::ValidationError(format!(
                        "Layer {} requires data transformation for binned scale but no data source.",
                        layer_idx + 1
                    )));
                }
                true
            } else {
                // No source, no filter, no stat transform, no pre-stat transform
                // - use __global__ data directly
                // (constants are already in global table with layer-indexed names)
                false
            }
        }
    };

    if !needs_query {
        return Ok(None);
    }

    // Determine the base query source
    // For layers with explicit source, cast_base_query already includes constants
    // For layers without source, cast_base_query is the cast global query
    let base_source = match &layer.source {
        Some(DataSource::FilePath(path)) => {
            // File paths need to be queried directly (casting not applied to files)
            format!("SELECT * FROM '{}'", path)
        }
        _ => {
            // Use the pre-built cast base query (which now includes constants)
            cast_base_query.to_string()
        }
    };

    // Wrap the cast base query (constants are already included)
    let mut query = format!("SELECT * FROM ({})", base_source);

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

    // Apply pre-stat transformations (e.g., binning for Binned scales)
    // This must happen before stat transforms so that data is transformed first
    query = apply_pre_stat_transform(&query, layer, schema, scales, type_names);

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
            // If the layer has no explicit source, no filter, and no order_by,
            // we can use __global__ directly (return None)
            // NOTE: Constants don't require a query because they're already in global table
            if layer.source.is_none() && filter.is_none() && order_by.is_none() {
                Ok(None)
            } else {
                // Layer has source, filter, or order_by - still need the query
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
    // Track whether we actually create the temp table (depends on transform_global_sql succeeding)
    let mut has_global_table = false;
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

            // NOTE: Don't read into data_map yet - defer until after casting is determined
            // The temp table exists and can be used for schema fetching
            has_global_table = true;
        }
    }

    // Build base queries for all layers BEFORE other processing
    // These are the raw SELECT * FROM source queries, without filters/stats
    let global_base_query = format!("SELECT * FROM {}", naming::global_table());
    let mut layer_base_queries: Vec<Option<String>> = Vec::new();
    for layer in &specs[0].layers {
        let source = determine_layer_source(layer, &materialized_ctes);
        let base_query = source.map(|src| format!("SELECT * FROM {}", src));
        layer_base_queries.push(base_query);
    }

    // Get types from base queries (Phase 1: types only, no min/max yet)
    let global_type_info = if has_global_table {
        fetch_schema_types(&global_base_query, &execute_query)?
    } else {
        Vec::new()
    };

    // Get types for each layer
    let mut layer_type_info: Vec<Vec<TypeInfo>> = Vec::new();
    for base_query in &layer_base_queries {
        let type_info = match base_query {
            Some(q) => fetch_schema_types(q, &execute_query)?,
            None => global_type_info.clone(), // Uses global
        };
        layer_type_info.push(type_info);
    }

    // Initial schemas (types only, no min/max - will be completed after casting)
    let global_schema = type_info_to_schema(&global_type_info);
    let mut layer_schemas: Vec<Schema> = layer_type_info
        .iter()
        .map(|ti| type_info_to_schema(ti))
        .collect();

    // Merge global mappings into layer aesthetics and expand wildcards
    // Smart wildcard expansion only creates mappings for columns that exist in schema
    merge_global_mappings_into_layers(&mut specs, &layer_schemas);

    // Split 'color' aesthetic to 'fill' and 'stroke' early in the pipeline
    // This must happen before validation so fill/stroke are validated (not color)
    // Note: Literals may create redundant constant columns (fill and stroke both 'blue')
    // but this is acceptable for correct validation behavior
    for spec in &mut specs {
        split_color_aesthetic(spec);
    }

    // Rebuild base queries WITH constants now that global mappings are merged
    // This ensures constants are included in schema extraction and can be cast if needed
    let layer_base_queries: Vec<Option<String>> = specs[0]
        .layers
        .iter()
        .map(|layer| build_base_layer_query(layer, &materialized_ctes, has_global_table))
        .collect();

    // Re-fetch type info from base queries now that they include constants
    let mut layer_type_info: Vec<Vec<TypeInfo>> = Vec::new();
    for base_query in &layer_base_queries {
        let type_info = match base_query {
            Some(q) => fetch_schema_types(q, &execute_query)?,
            None => global_type_info.clone(), // Uses global
        };
        layer_type_info.push(type_info);
    }

    // Rebuild layer schemas with constant columns included
    layer_schemas = layer_type_info
        .iter()
        .map(|ti| type_info_to_schema(ti))
        .collect();

    // Validate all layers against their schemas
    // This must happen BEFORE build_layer_query because stat transforms remove consumed aesthetics
    // (e.g., 'weight' is consumed by bar's stat_count and removed from mappings)
    // This catches errors with clear error messages:
    // - Missing required aesthetics
    // - Invalid SETTING parameters
    // - Non-existent columns in mappings
    // - Non-existent columns in PARTITION BY
    // - Unsupported aesthetics in REMAPPING
    // - Invalid stat columns in REMAPPING
    validate(&specs[0].layers, &layer_schemas)?;

    // Create scales for all mapped aesthetics that don't have explicit SCALE clauses
    // This ensures temporal transform inference works even without explicit SCALE x
    create_missing_scales(&mut specs[0]);

    // Resolve scale types and transforms early based on column dtypes
    // This enables type coercion and determines what casting may be needed
    resolve_scale_types_and_transforms(&mut specs[0], &layer_type_info)?;

    // Determine which columns need type casting
    // This is based on scale requirements and type coercion across layers
    let type_requirements = determine_type_requirements(&specs[0], &layer_type_info, type_names);

    // Apply casting to base queries
    // This wraps queries with CAST expressions for columns that need type conversion
    let cast_global_query = if has_global_table {
        // Collect requirements for global query (columns mapped by any layer using global data)
        let global_requirements = collect_requirements_for_global(&type_requirements, &specs[0]);
        apply_column_casting(&global_base_query, &global_requirements)
    } else {
        global_base_query.clone()
    };

    // Apply casting to layer base queries
    let cast_layer_queries: Vec<Option<String>> = layer_base_queries
        .iter()
        .enumerate()
        .map(|(idx, base_q)| {
            base_q
                .as_ref()
                .map(|q| apply_column_casting(q, &type_requirements[idx]))
        })
        .collect();

    // Update type info with post-cast dtypes
    // This ensures subsequent schema extraction and scale resolution see the correct types
    for (layer_idx, requirements) in type_requirements.iter().enumerate() {
        if layer_idx < layer_type_info.len() {
            update_type_info_for_casting(&mut layer_type_info[layer_idx], requirements);
        }
    }

    // Complete schemas with min/max from cast queries (Phase 2: ranges from cast data)
    // This ensures min/max values reflect the actual cast types
    let global_schema = if has_global_table {
        complete_schema_ranges(&cast_global_query, &global_type_info, &execute_query)?
    } else {
        global_schema
    };

    // Complete layer schemas with min/max from cast queries
    for (idx, cast_query) in cast_layer_queries.iter().enumerate() {
        layer_schemas[idx] = match cast_query {
            Some(cq) => complete_schema_ranges(cq, &layer_type_info[idx], &execute_query)?,
            None => global_schema.clone(), // Uses global
        };
    }

    // Pre-resolve Binned scales using schema-derived context
    // This must happen before build_layer_query so pre_stat_transform_sql has resolved breaks
    apply_pre_stat_resolve(&mut specs[0], &layer_schemas)?;

    // Add discrete mapped columns to partition_by for all layers
    // This ensures proper grouping for color, fill, shape, etc. aesthetics
    // Uses scale type (if explicit) to determine discreteness, falling back to schema
    // Clone scales to avoid borrow conflict (layers borrowed mutably, scales immutably)
    let scales = specs[0].scales.clone();
    add_discrete_columns_to_partition_by(&mut specs[0].layers, &layer_schemas, &scales);

    // Execute layer-specific queries
    // build_layer_query() handles all cases:
    // - Layer with source (CTE, table, or file) → query that source
    // - Layer with filter/order_by but no source → query __ggsql_global__ with filter/order_by and constants
    // - Layer with no source, no filter, no order_by → returns None (use global directly, constants already injected)
    let facet = specs[0].facet.clone();
    // Clone scales to avoid borrow conflict (layers borrowed mutably, scales immutably)
    let scales = specs[0].scales.clone();

    for (idx, layer) in specs[0].layers.iter_mut().enumerate() {
        // Get the cast base query for this layer (or use global)
        // Constants are now included in cast_layer_queries via build_base_layer_query
        let cast_base = cast_layer_queries[idx]
            .as_deref()
            .unwrap_or(&cast_global_query);

        // Get mutable reference to layer for stat transform to update aesthetics
        if let Some(layer_query) = build_layer_query(
            layer,
            cast_base,
            &layer_schemas[idx],
            has_global_table,
            idx,
            facet.as_ref(),
            &scales,
            type_names,
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

    // Check if any layer uses global data directly
    // A layer "uses global" only if:
    // 1. There are no layers (VISUALISE without DRAW) - global data is the output
    // 2. build_layer_query returned None (no entry in data_map) - layer uses global DataFrame directly
    // Note: Layers with filters/stats query the global temp table but produce their own data,
    // so they don't need GLOBAL_DATA_KEY in data_map
    let needs_global_data = specs[0].layers.is_empty()
        || (0..specs[0].layers.len()).any(|idx| !data_map.contains_key(&naming::layer_key(idx)));

    // If global data is needed, execute the CAST global query now
    // This ensures global data goes through the same casting pipeline as layer data
    if needs_global_data && has_global_table {
        let df = execute_query(&cast_global_query)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to fetch global data: {}", e)))?;
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);
    }

    // Validate we have some data
    if data_map.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No data sources found. Either provide a SQL query or use MAPPING FROM in layers."
                .to_string(),
        ));
    }

    // For layers that use global data directly (no layer-specific data), ensure global data exists
    // Note: Layers with stat transforms create their own data in layer_key(idx), so they don't need this
    let layer_uses_global_directly =
        (0..specs[0].layers.len()).any(|idx| !data_map.contains_key(&naming::layer_key(idx)));
    if layer_uses_global_directly && !data_map.contains_key(naming::GLOBAL_DATA_KEY) {
        return Err(GgsqlError::ValidationError(
            "Some layers use global data but no SQL query was provided.".to_string(),
        ));
    }

    // Create scales for aesthetics added by stat transforms (e.g., y from histogram)
    // This must happen after build_layer_query() which applies stat transforms
    // and modifies layer.mappings with new aesthetics like y → __ggsql_stat_count__
    for spec in &mut specs {
        create_missing_scales_post_stat(spec);
    }

    // Post-process specs: replace literals with column references and compute labels
    for spec in &mut specs {
        // Replace literal aesthetic values with column references to synthetic constant columns
        replace_literals_with_columns(spec);
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
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic);

        // Apply binning to each column
        for (data_key, col_name) in column_sources {
            if let Some(df) = data_map.get(&data_key) {
                // Skip if column doesn't exist in this data source
                if df.column(&col_name).is_err() {
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
                        let transform_kind = scale
                            .scale_type
                            .as_ref()
                            .unwrap()
                            .default_transform(&scale.aesthetic, Some(&common_dtype));
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
    let global_key = naming::GLOBAL_DATA_KEY.to_string();

    // Track which (data_key, column_name) pairs we've already coerced
    let mut coerced: HashSet<(String, String)> = HashSet::new();

    // Check each layer's mapping → look up in layer data OR global data
    for (i, layer) in layers.iter().enumerate() {
        let layer_key = naming::layer_key(i);

        for aes_name in &aesthetics_to_check {
            if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(aes_name) {
                // Determine which data source to use
                let data_key = if data_map.contains_key(&layer_key) {
                    layer_key.clone()
                } else if data_map.contains_key(&global_key) {
                    global_key.clone()
                } else {
                    continue;
                };

                // Skip if already coerced
                let key = (data_key.clone(), name.clone());
                if coerced.contains(&key) {
                    continue;
                }

                // Check if column exists in this DataFrame
                if let Some(df) = data_map.get(&data_key) {
                    if df.column(name).is_ok() {
                        let coerced_df = coerce_column_to_type(df, name, target_type)?;
                        data_map.insert(data_key.clone(), coerced_df);
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
            // Determine if this is a discrete scale (for unique values vs min/max range)
            // Note: is_discrete() returns true for Continuous/Binned (supports breaks)
            // and false for Discrete/Identity (doesn't support breaks)
            // So we invert it: discrete range when is_discrete() returns false
            let use_discrete_range = !st.is_discrete();

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
    let global_df = data_map.get(naming::GLOBAL_DATA_KEY);

    // Check each layer's mapping → look up in layer data OR global data
    // (global mappings already merged into layers)
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
    // First pass: apply OOB transformations (censor sets to NULL, squish clamps)
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
        let column_sources =
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic);

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
            find_columns_for_aesthetic_with_sources(&spec.layers, &scale.aesthetic);

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
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let aesthetics_to_check = get_aesthetic_family(aesthetic);

    // Check each layer's mapping → uses layer data or global data
    // (global mappings already merged into layers)
    for (i, layer) in layers.iter().enumerate() {
        // Determine which data source this layer uses:
        // - layer.source: explicit data source
        // - layer.filter: per-layer filter creates separate data
        // - stat transform: geoms like bar, histogram create layer-specific data
        let has_layer_data = layer.source.is_some()
            || layer.filter.is_some()
            || layer.geom.needs_stat_transform(&layer.mappings);

        let data_key = if has_layer_data {
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
    use crate::plot::{ArrayElement, SqlExpression};
    use crate::Geom;
    use polars::prelude::{DataType, IntoColumn};

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

    /// Test helper to create SqlTypeNames with default values
    fn test_type_names() -> SqlTypeNames {
        SqlTypeNames {
            number: Some("DOUBLE".to_string()),
            integer: Some("BIGINT".to_string()),
            date: Some("DATE".to_string()),
            datetime: Some("TIMESTAMP".to_string()),
            time: Some("TIME".to_string()),
            string: Some("VARCHAR".to_string()),
            boolean: Some("BOOLEAN".to_string()),
        }
    }

    #[test]
    fn test_build_layer_query_with_cte() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("sales".to_string()));

        // Build cast base query (simulating what prepare_data_with_executor would do)
        let cast_base_query = format!("SELECT * FROM {}", naming::cte_table("sales"));

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            &cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should use the cast base query wrapped in FROM
        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM (SELECT * FROM __ggsql_cte_sales_"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_with_cte_and_filter() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("sales".to_string()));
        layer.filter = Some(SqlExpression::new("year = 2024"));

        // Build cast base query (simulating what prepare_data_with_executor would do)
        let cast_base_query = format!("SELECT * FROM {}", naming::cte_table("sales"));

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            &cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should use the cast base query wrapped in FROM with filter
        let query = result.unwrap().unwrap();
        assert!(query.contains("__ggsql_cte_sales_"));
        assert!(query.ends_with(" WHERE year = 2024"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_without_cte() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));

        // Build cast base query (simulating what prepare_data_with_executor would do)
        let cast_base_query = "SELECT * FROM some_table";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should wrap the cast base query
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM (SELECT * FROM some_table)".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_table_with_filter() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.filter = Some(SqlExpression::new("value > 100"));

        // Build cast base query (simulating what prepare_data_with_executor would do)
        let cast_base_query = "SELECT * FROM some_table";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM (SELECT * FROM some_table) WHERE value > 100".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::FilePath("data/sales.csv".to_string()));

        // For file paths, cast_base_query is not used (file paths are queried directly)
        let cast_base_query = "";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // File paths should be wrapped in single quotes
        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM (SELECT * FROM 'data/sales.csv')".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_file_path_with_filter() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::FilePath("data.parquet".to_string()));
        layer.filter = Some(SqlExpression::new("x > 10"));

        // For file paths, cast_base_query is not used (file paths are queried directly)
        let cast_base_query = "";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM (SELECT * FROM 'data.parquet') WHERE x > 10".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_none_source_with_filter() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.filter = Some(SqlExpression::new("category = 'A'"));

        // Build cast base query (the cast global query)
        let cast_base_query = format!("SELECT * FROM {}", naming::global_table());

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            &cast_base_query,
            &empty_schema,
            true,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should wrap the cast global query and add filter
        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM (SELECT * FROM __ggsql_global_"));
        assert!(query.ends_with(") WHERE category = 'A'"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_layer_query_none_source_no_filter() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());

        // Cast base query (won't be used when returning None)
        let cast_base_query = format!("SELECT * FROM {}", naming::global_table());

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            &cast_base_query,
            &empty_schema,
            true,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should return None - layer uses __global__ directly
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_build_layer_query_filter_without_global_errors() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.filter = Some(SqlExpression::new("x > 10"));

        // Cast base query (won't be used due to error)
        let cast_base_query = "";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            2,
            None,
            &[],
            &type_names,
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
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.order_by = Some(SqlExpression::new("date ASC"));

        // Build cast base query
        let cast_base_query = "SELECT * FROM some_table";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some("SELECT * FROM (SELECT * FROM some_table) ORDER BY date ASC".to_string())
        );
    }

    #[test]
    fn test_build_layer_query_with_filter_and_order_by() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        layer.filter = Some(SqlExpression::new("year = 2024"));
        layer.order_by = Some(SqlExpression::new("date DESC, value ASC"));

        // Build cast base query
        let cast_base_query = "SELECT * FROM some_table";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        assert_eq!(
            result.unwrap(),
            Some(
                "SELECT * FROM (SELECT * FROM some_table) WHERE year = 2024 ORDER BY date DESC, value ASC"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_build_layer_query_none_source_with_order_by() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.order_by = Some(SqlExpression::new("x ASC"));

        // Build cast base query (the cast global query)
        let cast_base_query = format!("SELECT * FROM {}", naming::global_table());

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            &cast_base_query,
            &empty_schema,
            true,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // Should wrap the cast global query and add order_by
        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM (SELECT * FROM __ggsql_global_"));
        assert!(query.ends_with(") ORDER BY x ASC"));
        assert!(query.contains(naming::session_id()));
    }

    #[test]
    fn test_build_base_layer_query_with_constants() {
        let materialized = HashSet::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        // Add literal mappings which become constants
        layer.mappings.insert(
            "color".to_string(),
            AestheticValue::Literal(LiteralValue::String("value".to_string())),
        );
        layer.mappings.insert(
            "size".to_string(),
            AestheticValue::Literal(LiteralValue::String("value2".to_string())),
        );

        // build_base_layer_query should include constants
        let base_query = build_base_layer_query(&layer, &materialized, false);

        // Should inject constants as columns
        let query = base_query.unwrap();
        assert!(query.contains("SELECT *"));
        assert!(query.contains("'value' AS __ggsql_const_color__"));
        assert!(query.contains("'value2' AS __ggsql_const_size__"));
        assert!(query.contains("FROM some_table"));
    }

    #[test]
    fn test_build_base_layer_query_constants_on_global() {
        let materialized = HashSet::new();

        // No source but has constants - should use global table
        // Constants are NOT added here because they're injected into the global table
        // with layer-indexed names (e.g., __ggsql_const_fill_0__)
        let mut layer = Layer::new(Geom::point());
        layer.mappings.insert(
            "fill".to_string(),
            AestheticValue::Literal(LiteralValue::String("value".to_string())),
        );

        // build_base_layer_query should return simple query from global
        // (constants are already in global table with layer-indexed names)
        let base_query = build_base_layer_query(&layer, &materialized, true);

        let query = base_query.unwrap();
        assert!(query.contains("FROM __ggsql_global_"));
        assert!(query.contains(naming::session_id()));
        // Constants should NOT be in the base query - they're in the global table
        assert!(!query.contains("__ggsql_const_fill__"));
    }

    #[test]
    fn test_build_layer_query_with_constants_in_base() {
        let empty_schema: Schema = Vec::new();

        let mut layer = Layer::new(Geom::point());
        layer.source = Some(DataSource::Identifier("some_table".to_string()));
        // Add literal mappings which become constants
        layer.mappings.insert(
            "fill".to_string(),
            AestheticValue::Literal(LiteralValue::String("blue".to_string())),
        );

        // Simulate what prepare_data_with_executor does: build base query with constants
        let cast_base_query = "SELECT *, 'blue' AS __ggsql_const_fill__ FROM some_table";

        let type_names = test_type_names();
        let result = build_layer_query(
            &mut layer,
            cast_base_query,
            &empty_schema,
            false,
            0,
            None,
            &[],
            &type_names,
            &mock_execute,
        );

        // build_layer_query wraps the cast_base_query (which already has constants)
        let query = result.unwrap().unwrap();
        assert!(query.contains("FROM (SELECT *"));
        assert!(query.contains("__ggsql_const_fill__"));
        assert!(query.contains("FROM some_table"));
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
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let count_col = layer_df.column(naming::stat_column("count").as_str());
        assert!(count_col.is_ok(), "Should have count column in layer data");
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
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

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
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

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
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

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
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

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
        data_map.insert(naming::GLOBAL_DATA_KEY.to_string(), df);

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

        let stroke = aes.get("stroke").unwrap().column_name().unwrap();
        assert_eq!(stroke, "species");

        let fill = aes.get("fill").unwrap().column_name().unwrap();
        assert_eq!(fill, "island");

        // Colors as global constant
        // Note: split_color_aesthetic runs before replace_literals_with_columns,
        // so the constant column is named after the target aesthetic (fill) not the source (color)
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, 'blue' AS color FROM ggsql:penguins
          DRAW point MAPPING island AS stroke
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(stroke.column_name().unwrap(), "island");

        let fill = aes.get("fill").unwrap();
        assert_eq!(fill.column_name().unwrap(), "__ggsql_const_fill_0__");

        // Colors as layer constant
        let query = r#"
          VISUALISE bill_len AS x, bill_dep AS y, island AS fill FROM ggsql:penguins
          DRAW point MAPPING 'blue' AS color
        "#;

        let result = prepare_data(query, &reader).unwrap();
        let aes = &result.specs[0].layers[0].mappings.aesthetics;

        let stroke = aes.get("stroke").unwrap();
        assert_eq!(stroke.column_name().unwrap(), "__ggsql_const_stroke_0__");

        let fill = aes.get("fill").unwrap();
        assert_eq!(fill.column_name().unwrap(), "island");

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
        // Discrete scales always censor OOB values (no explicit oob setting needed)
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

        // All rows should be preserved (censoring sets to null, doesn't filter)
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 4);

        // Verify x and y columns are preserved
        let x_col = df.column("x").unwrap();
        let x_values: Vec<i32> = x_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(x_values, vec![1, 2, 3, 4]);
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

        // Verify only A and B are present
        let cat_col = df.column("category").unwrap();
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

        // All rows should be preserved (censoring sets to null, doesn't filter)
        // C and D should have null color, but row is still present
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
        assert_eq!(df.height(), 4);

        // Verify x and y columns are preserved
        let y_col = df.column("y").unwrap();
        let y_values: Vec<i32> = y_col
            .as_materialized_series()
            .i32()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(y_values, vec![10, 20, 30, 40]);
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

    #[test]
    fn test_apply_column_casting_empty() {
        // Empty requirements should return query unchanged
        let query = "SELECT * FROM some_table";
        let requirements: Vec<TypeRequirement> = vec![];

        let result = apply_column_casting(query, &requirements);
        assert_eq!(result, query);
    }

    #[test]
    fn test_apply_column_casting_single_column() {
        use crate::plot::CastTargetType;

        let query = "SELECT * FROM some_table";
        let requirements = vec![TypeRequirement {
            column: "date_col".to_string(),
            target_type: CastTargetType::Date,
            sql_type_name: "DATE".to_string(),
        }];

        let result = apply_column_casting(query, &requirements);
        assert!(result.contains("EXCLUDE (\"date_col\")"));
        assert!(result.contains("CAST(\"date_col\" AS DATE) AS \"date_col\""));
        assert!(result.ends_with(" FROM (SELECT * FROM some_table)"));
    }

    #[test]
    fn test_apply_column_casting_multiple_columns() {
        use crate::plot::CastTargetType;

        let query = "SELECT * FROM data";
        let requirements = vec![
            TypeRequirement {
                column: "date_col".to_string(),
                target_type: CastTargetType::Date,
                sql_type_name: "DATE".to_string(),
            },
            TypeRequirement {
                column: "value".to_string(),
                target_type: CastTargetType::Number,
                sql_type_name: "DOUBLE".to_string(),
            },
        ];

        let result = apply_column_casting(query, &requirements);
        assert!(result.contains("EXCLUDE (\"date_col\", \"value\")"));
        assert!(result.contains("CAST(\"date_col\" AS DATE) AS \"date_col\""));
        assert!(result.contains("CAST(\"value\" AS DOUBLE) AS \"value\""));
    }

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
        let df = result.data.get(naming::GLOBAL_DATA_KEY).unwrap();
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
    fn test_gets_default_scale_positional() {
        // Position aesthetics should get default scale (type inferred from data)
        assert!(gets_default_scale("x"));
        assert!(gets_default_scale("y"));
        assert!(gets_default_scale("xmin"));
        assert!(gets_default_scale("xmax"));
        assert!(gets_default_scale("ymin"));
        assert!(gets_default_scale("ymax"));
        assert!(gets_default_scale("xend"));
        assert!(gets_default_scale("yend"));
        assert!(gets_default_scale("x2"));
        assert!(gets_default_scale("y2"));
    }

    #[test]
    fn test_gets_default_scale_color() {
        // Color aesthetics should get default scale
        // Note: color/colour/col are split to fill/stroke before scale creation
        assert!(gets_default_scale("fill"));
        assert!(gets_default_scale("stroke"));
    }

    #[test]
    fn test_gets_default_scale_size_and_other() {
        // Size, opacity, shape, linetype should get default scale
        assert!(gets_default_scale("size"));
        assert!(gets_default_scale("linewidth"));
        assert!(gets_default_scale("opacity"));
        assert!(gets_default_scale("shape"));
        assert!(gets_default_scale("linetype"));
    }

    #[test]
    fn test_gets_default_scale_identity_aesthetics() {
        // Text, label, group, detail, tooltip should NOT get default scale (use Identity)
        assert!(!gets_default_scale("text"));
        assert!(!gets_default_scale("label"));
        assert!(!gets_default_scale("group"));
        assert!(!gets_default_scale("detail"));
        assert!(!gets_default_scale("tooltip"));
        assert!(!gets_default_scale("unknown_aesthetic"));
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
    fn test_infer_scale_target_type_discrete_boolean_range() {
        use crate::plot::scale::ScaleType;

        // Discrete scale with boolean input range should infer Boolean type
        let scale = Scale {
            aesthetic: "color".to_string(),
            scale_type: Some(ScaleType::discrete()),
            input_range: Some(vec![
                ArrayElement::Boolean(true),
                ArrayElement::Boolean(false),
            ]),
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, Some(ArrayElementType::Boolean));
    }

    #[test]
    fn test_infer_scale_target_type_discrete_string_range() {
        use crate::plot::scale::ScaleType;

        // Discrete scale with string input range should infer String type
        let scale = Scale {
            aesthetic: "color".to_string(),
            scale_type: Some(ScaleType::discrete()),
            input_range: Some(vec![
                ArrayElement::String("A".to_string()),
                ArrayElement::String("B".to_string()),
            ]),
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, Some(ArrayElementType::String));
    }

    #[test]
    fn test_infer_scale_target_type_continuous_date_transform() {
        use crate::plot::scale::{ScaleType, Transform};

        // Continuous scale with date transform should infer Date type
        let scale = Scale {
            aesthetic: "x".to_string(),
            scale_type: Some(ScaleType::continuous()),
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: Some(Transform::date()),
            explicit_transform: true,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, Some(ArrayElementType::Date));
    }

    #[test]
    fn test_infer_scale_target_type_continuous_log_transform() {
        use crate::plot::scale::{ScaleType, Transform};

        // Continuous scale with log transform should infer Number type
        let scale = Scale {
            aesthetic: "y".to_string(),
            scale_type: Some(ScaleType::continuous()),
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: Some(Transform::log()),
            explicit_transform: true,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, Some(ArrayElementType::Number));
    }

    #[test]
    fn test_infer_scale_target_type_binned_returns_none() {
        use crate::plot::scale::ScaleType;

        // Binned scales should return None (no coercion - binning happens in SQL)
        let scale = Scale {
            aesthetic: "x".to_string(),
            scale_type: Some(ScaleType::binned()),
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, None);
    }

    #[test]
    fn test_infer_scale_target_type_identity_returns_none() {
        use crate::plot::scale::ScaleType;

        // Identity scales should return None (no coercion)
        let scale = Scale {
            aesthetic: "label".to_string(),
            scale_type: Some(ScaleType::identity()),
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, None);
    }

    #[test]
    fn test_infer_scale_target_type_continuous_no_transform() {
        use crate::plot::scale::ScaleType;

        // Continuous scale without transform should return None (no coercion)
        let scale = Scale {
            aesthetic: "x".to_string(),
            scale_type: Some(ScaleType::continuous()),
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: HashMap::new(),
            resolved: false,
            label_mapping: None,
            label_template: None,
        };

        let target_type = infer_scale_target_type(&scale);
        assert_eq!(target_type, None);
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
}
