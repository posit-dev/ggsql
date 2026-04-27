//! Layer query building, data transforms, and stat application.
//!
//! This module handles building SQL queries for layers, applying pre-stat
//! transformations, stat transforms, and post-query operations.

use crate::plot::aesthetic::{self, AestheticContext};
use crate::plot::layer::is_transposed;
use crate::plot::layer::orientation::{flip_position_aesthetics, resolve_orientation};
use crate::plot::{
    AestheticValue, DefaultAestheticValue, Layer, ParameterValue, Scale, Schema, StatResult,
};
use crate::reader::SqlDialect;
use crate::{naming, DataFrame, GgsqlError, Result};
use arrow::datatypes::DataType;
use std::collections::{HashMap, HashSet};

use super::casting::TypeRequirement;
use super::schema::build_aesthetic_schema;

/// Build the source query for a layer.
///
/// Returns a complete query that can be executed to retrieve the layer's data:
/// - Annotation layers → VALUES clause with all aesthetic columns (modifies layer mappings)
/// - Table/CTE layers → `SELECT * FROM table_or_cte`
/// - File path layers → `SELECT * FROM 'path'`
/// - Layers without explicit source → `SELECT * FROM __ggsql_global__`
///
/// For annotation layers, this function processes parameters and converts them to
/// Column/AnnotationColumn mappings, so the layer is modified in place.
pub fn layer_source_query(
    layer: &mut Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
    dialect: &dyn SqlDialect,
) -> Result<String> {
    match &layer.source {
        Some(crate::DataSource::Annotation) => {
            // Annotation layers: process parameters and return complete VALUES clause (with on-the-fly recycling)
            process_annotation_layer(layer, dialect)
        }
        Some(crate::DataSource::Identifier(name)) => {
            // Regular table or CTE
            let source = if materialized_ctes.contains(name) {
                naming::cte_table(name)
            } else {
                name.clone()
            };
            Ok(format!("SELECT * FROM {}", source))
        }
        Some(crate::DataSource::FilePath(path)) => {
            // File path source
            Ok(format!("SELECT * FROM '{}'", path))
        }
        None => {
            // Layer uses global data
            debug_assert!(has_global, "Layer has no source and no global data");
            Ok(format!(
                "SELECT * FROM {}",
                naming::quote_ident(&naming::global_table())
            ))
        }
    }
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
pub fn build_layer_select_list(
    layer: &Layer,
    type_requirements: &[TypeRequirement],
    dialect: &dyn SqlDialect,
) -> Vec<String> {
    let mut select_exprs = Vec::new();

    // Start with * to preserve all original columns
    // This ensures facet variables, partition_by columns, and any other
    // columns are available for downstream processing (stat transforms, etc.)
    select_exprs.push("*".to_string());

    // Build a map of column -> cast requirement for quick lookup
    let cast_map: HashMap<&str, &TypeRequirement> = type_requirements
        .iter()
        .map(|r| (r.column.as_str(), r))
        .collect();

    // Add aesthetic-mapped columns with prefixed names (and casts where needed)
    for (aesthetic, value) in &layer.mappings.aesthetics {
        let aes_col_name = naming::aesthetic_column(aesthetic);
        let select_expr = match value {
            AestheticValue::Column { name, .. } | AestheticValue::AnnotationColumn { name } => {
                // Check if this column needs casting
                if let Some(req) = cast_map.get(name.as_str()) {
                    // Cast and rename to prefixed aesthetic name
                    format!(
                        "CAST({} AS {}) AS {}",
                        naming::quote_ident(name),
                        req.sql_type_name,
                        naming::quote_ident(&aes_col_name)
                    )
                } else {
                    // Just rename to prefixed aesthetic name
                    format!(
                        "{} AS {}",
                        naming::quote_ident(name),
                        naming::quote_ident(&aes_col_name)
                    )
                }
            }
            AestheticValue::Literal(lit) => {
                // Literals become columns with prefixed aesthetic name
                format!(
                    "{} AS {}",
                    lit.to_sql(dialect),
                    naming::quote_ident(&aes_col_name)
                )
            }
        };

        select_exprs.push(select_expr);
    }

    select_exprs
}

/// Apply remappings to rename stat columns to their target aesthetic's prefixed name,
/// and add constant columns for literal remappings.
///
/// After stat transforms, columns like `__ggsql_stat_count` need to be renamed
/// to the target aesthetic's prefixed name (e.g., `__ggsql_aes_y__`).
///
/// For literal values (e.g., `ymin=0`), this creates a constant column.
///
/// Note: Prefixed aesthetic names persist through the entire pipeline.
/// We do NOT rename `__ggsql_aes_x__` back to `x`.
pub fn apply_remappings_post_query(df: DataFrame, layer: &Layer) -> Result<DataFrame> {
    let mut df = df;
    let row_count = df.height();

    // Apply remappings: stat columns → prefixed aesthetic names
    // e.g., __ggsql_stat_count → __ggsql_aes_y__
    // Remappings structure: HashMap<target_aesthetic, AestheticValue pointing to stat column>
    for (target_aesthetic, value) in &layer.remappings.aesthetics {
        let target_col_name = naming::aesthetic_column(target_aesthetic);

        match value {
            AestheticValue::Column { name, .. } | AestheticValue::AnnotationColumn { name } => {
                // Check if this stat column exists in the DataFrame
                if df.column(name).is_ok() {
                    df = df.rename(name, &target_col_name).map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to rename stat column '{}' to '{}': {}",
                            name, target_aesthetic, e
                        ))
                    })?;
                }
            }
            AestheticValue::Literal(lit) => {
                // Add constant column for literal values
                let array = literal_to_array(lit, row_count);
                df = df.with_column(&target_col_name, array).map_err(|e| {
                    GgsqlError::InternalError(format!(
                        "Failed to add literal column '{}': {}",
                        target_col_name, e
                    ))
                })?;
            }
        }
    }

    // Drop any remaining __ggsql_stat_* columns that weren't consumed by remappings.
    let stat_cols: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|name| naming::is_stat_column(name))
        .collect();
    if !stat_cols.is_empty() {
        df = df.drop_many(&stat_cols)?;
    }

    Ok(df)
}

/// Convert a literal value to an Arrow ArrayRef with constant values.
///
/// For string literals, attempts to parse as temporal types (date/datetime/time)
/// using the same format precedence as the rest of ggsql. Falls back to string
/// if parsing fails.
pub fn literal_to_array(lit: &ParameterValue, len: usize) -> arrow::array::ArrayRef {
    use crate::array_util::{cast_array, new_constant_bool, new_constant_f64, new_constant_str};
    use crate::plot::ArrayElement;
    use arrow::datatypes::{DataType, TimeUnit};
    use std::sync::Arc;

    match lit {
        ParameterValue::Number(n) => new_constant_f64(*n, len),
        ParameterValue::String(s) => {
            // Try to parse as temporal types (DateTime > Date > Time)
            match ArrayElement::String(s.clone()).try_as_temporal() {
                ArrayElement::DateTime(micros) => {
                    let arr: arrow::array::ArrayRef =
                        Arc::new(arrow::array::Int64Array::from(vec![micros; len]));
                    cast_array(&arr, &DataType::Timestamp(TimeUnit::Microsecond, None))
                        .expect("DateTime cast should not fail")
                }
                ArrayElement::Date(days) => {
                    let arr: arrow::array::ArrayRef =
                        Arc::new(arrow::array::Int32Array::from(vec![days; len]));
                    cast_array(&arr, &DataType::Date32).expect("Date cast should not fail")
                }
                ArrayElement::Time(nanos) => {
                    let arr: arrow::array::ArrayRef =
                        Arc::new(arrow::array::Int64Array::from(vec![nanos; len]));
                    cast_array(&arr, &DataType::Time64(TimeUnit::Nanosecond))
                        .expect("Time cast should not fail")
                }
                ArrayElement::String(_) => {
                    // Parsing failed, use original string
                    new_constant_str(s, len)
                }
                _ => unreachable!("try_as_temporal only returns String or temporal types"),
            }
        }
        ParameterValue::Boolean(b) => new_constant_bool(*b, len),
        ParameterValue::Array(_) | ParameterValue::Null => {
            unreachable!("Arrays are never moved to mappings; NULL is filtered in process_annotation_layers()")
        }
    }
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
/// * `dialect` - SQL dialect for the database backend
pub fn apply_pre_stat_transform(
    query: &str,
    layer: &Layer,
    full_schema: &Schema,
    aesthetic_schema: &Schema,
    scales: &[Scale],
    dialect: &dyn SqlDialect,
) -> String {
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

        // Find column dtype from aesthetic schema using aesthetic column name
        let col_dtype = aesthetic_schema
            .iter()
            .find(|c| c.name == aes_col_name)
            .map(|c| c.dtype.clone())
            .unwrap_or(DataType::Utf8); // Default to Utf8 if not found

        // Find scale for this aesthetic
        if let Some(scale) = scales.iter().find(|s| s.aesthetic == *aesthetic) {
            if let Some(ref scale_type) = scale.scale_type {
                // Get pre-stat SQL transformation from scale type (if applicable)
                // Each scale type's pre_stat_transform_sql() returns None if not applicable
                if let Some(sql) =
                    scale_type.pre_stat_transform_sql(&aes_col_name, &col_dtype, scale, dialect)
                {
                    transformed_columns.insert(aes_col_name.clone());
                    transform_exprs.push((aes_col_name, sql));
                }
            }
        }
    }

    if transform_exprs.is_empty() {
        return query.to_string();
    }

    // Build explicit column list from full_schema (original columns) and
    // aesthetic_schema (aesthetic columns added by build_layer_base_query).
    // The base query produces SELECT *, col AS __ggsql_aes_x__, ... so the
    // actual SQL output has both, but they come from different schema sources.
    // This avoids SELECT * EXCLUDE which has portability issues across SQL backends.
    let mut seen: HashSet<&str> = HashSet::new();
    let combined_cols = full_schema.iter().chain(aesthetic_schema.iter());

    let select_exprs: Vec<String> = combined_cols
        .filter(|col| seen.insert(&col.name))
        .map(|col| {
            if let Some((_, sql)) = transform_exprs.iter().find(|(c, _)| c == &col.name) {
                format!("{} AS {}", sql, naming::quote_ident(&col.name))
            } else {
                naming::quote_ident(&col.name)
            }
        })
        .collect();

    format!(
        "SELECT {} FROM ({}) AS \"__ggsql_pre__\"",
        select_exprs.join(", "),
        query
    )
}

/// Part 1: Build the initial layer query with SELECT, casts, filters, and aesthetic renames.
///
/// This function builds a query that:
/// 1. Applies filter (uses original column names - that's what users write)
/// 2. Renames columns to aesthetic names (e.g., "Date" AS "__ggsql_aes_x__")
/// 3. Applies type casts based on scale requirements
///
/// For annotation layers, the source_query is already the complete VALUES clause,
/// so it's returned as-is (no wrapping, filtering, or casting needed).
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
/// * `source_query` - The base query for the layer's data source (for annotations, this is already the VALUES clause)
/// * `type_requirements` - Columns that need type casting (not applicable to annotations)
///
/// # Returns
///
/// The base query string with SELECT/casts/filters applied.
pub fn build_layer_base_query(
    layer: &Layer,
    source_query: &str,
    type_requirements: &[TypeRequirement],
    dialect: &dyn SqlDialect,
) -> String {
    // Annotation layers now go through the same pipeline as regular layers.
    // The source_query for annotations is a VALUES clause with raw column names,
    // and this function wraps it with SELECT expressions that rename to prefixed aesthetic names.

    // Build SELECT list with aesthetic renames, casts
    let select_exprs = build_layer_select_list(layer, type_requirements, dialect);
    let select_clause = if select_exprs.is_empty() {
        "*".to_string()
    } else {
        select_exprs.join(", ")
    };

    // Build query with optional WHERE clause
    if let Some(ref f) = layer.filter {
        format!(
            "SELECT {} FROM ({}) AS \"__ggsql_src__\" WHERE {}",
            select_clause,
            source_query,
            f.as_str()
        )
    } else {
        format!(
            "SELECT {} FROM ({}) AS \"__ggsql_src__\"",
            select_clause, source_query
        )
    }
}

/// Part 2: Apply stat transforms and ORDER BY to a base query.
///
/// This function:
/// 1. Builds the aesthetic-named schema for stat transforms
/// 2. Updates layer mappings to use prefixed aesthetic names
/// 3. Applies pre-stat transforms (e.g., binning, discrete censoring)
/// 4. Builds group_by columns from partition_by
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
/// * `scales` - All resolved scales
/// * `dialect` - SQL dialect for the database backend
/// * `execute_query` - Function to execute queries (needed for some stat transforms)
///
/// # Returns
///
/// The final query string with stat transforms and ORDER BY applied.
pub fn apply_layer_transforms<F>(
    layer: &mut Layer,
    base_query: &str,
    schema: &Schema,
    scales: &[Scale],
    dialect: &dyn SqlDialect,
    execute_query: &F,
) -> Result<String>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    use crate::plot::layer::orientation::flip_position_aesthetics;

    // Clone order_by early to avoid borrow conflicts
    let order_by = layer.order_by.clone();

    // Orientation detection and initial flip was already done in mod.rs before
    // build_layer_base_query. We just check if we need to flip back after stat.
    let needs_flip = is_transposed(layer);

    // Build the aesthetic-named schema for stat transforms
    // Note: Mappings were already flipped in mod.rs if needed, so schema reflects normalized orientation
    let aesthetic_schema: Schema = build_aesthetic_schema(layer, schema);

    // Collect literal aesthetic column names BEFORE conversion to Column values.
    // Literal columns contain constant values (same for every row), so adding them to
    // GROUP BY doesn't affect aggregation results - they're simply preserved through grouping.
    let literal_columns: Vec<String> = layer
        .mappings
        .aesthetics
        .iter()
        .filter_map(|(aesthetic, value)| {
            if value.is_literal() {
                Some(naming::aesthetic_column(aesthetic))
            } else {
                None
            }
        })
        .collect();

    // Update mappings to use prefixed aesthetic names
    // This must happen BEFORE stat transforms so they use aesthetic names
    layer.update_mappings_for_aesthetic_columns();

    // Apply pre-stat transforms (e.g., binning, discrete censoring)
    // Uses aesthetic names since columns are now renamed and mappings updated
    let query = apply_pre_stat_transform(
        base_query,
        layer,
        schema,
        &aesthetic_schema,
        scales,
        dialect,
    );

    // Build group_by columns from partition_by
    // Note: Facet aesthetics are already in partition_by via add_discrete_columns_to_partition_by,
    // so we don't add facet.get_variables() here (which would add original column names
    // instead of aesthetic column names, breaking pre-stat transforms like domain censoring).
    let mut group_by: Vec<String> = Vec::new();
    for col in &layer.partition_by {
        group_by.push(col.clone());
    }

    // Add literal aesthetic columns to group_by so they survive stat transforms.
    // Since literal columns contain constant values (same for every row), adding them
    // to GROUP BY doesn't affect aggregation results - they're simply preserved.
    for col in &literal_columns {
        if !group_by.contains(col) {
            group_by.push(col.clone());
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
        dialect,
    )?;

    // Flip user remappings BEFORE merging defaults for Transposed orientation.
    // User remappings are in user orientation (e.g., `count AS x` for horizontal histogram).
    // We flip them to aligned orientation so they're uniform with defaults.
    // At the end, we flip everything back together.
    if needs_flip {
        flip_position_aesthetics(&mut layer.remappings.aesthetics);
    }

    // Apply literal default remappings from geom defaults (e.g., y2 => 0.0 for bar baseline).
    // These apply regardless of stat transform, but only if user hasn't overridden them.
    // Defaults are always in aligned orientation.
    for (aesthetic, default_value) in layer.geom.default_remappings().defaults {
        // Only process literal values here (Column values are handled in Transformed branch)
        if !matches!(default_value, DefaultAestheticValue::Column(_)) {
            // Only add if user hasn't already specified this aesthetic in remappings or mappings
            if !layer.remappings.aesthetics.contains_key(*aesthetic)
                && !layer.mappings.aesthetics.contains_key(*aesthetic)
            {
                layer
                    .remappings
                    .insert(aesthetic.to_string(), default_value.to_aesthetic_value());
            }
        }
    }

    let final_query = match stat_result {
        StatResult::Transformed {
            query: transformed_query,
            stat_columns,
            dummy_columns,
            consumed_aesthetics,
        } => {
            // Build stat column -> aesthetic mappings from geom defaults for renaming
            let mut final_remappings: HashMap<String, String> = HashMap::new();

            for (aesthetic, default_value) in layer.geom.default_remappings().defaults {
                if let DefaultAestheticValue::Column(stat_col) = default_value {
                    // Stat column mapping: stat_col -> aesthetic (for rename)
                    final_remappings.insert(stat_col.to_string(), aesthetic.to_string());
                }
            }

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

            // Auto-remap stat columns whose names are position aesthetics that were
            // consumed by the stat (e.g. Aggregate's `pos1`/`pos2` outputs). The geom
            // can't list these in `default_remappings` because the set of position
            // aesthetics in play is dynamic per layer.
            for stat in &stat_columns {
                if final_remappings.contains_key(stat) {
                    continue;
                }
                if aesthetic::is_position_aesthetic(stat) && consumed_aesthetics.contains(stat) {
                    final_remappings.insert(stat.clone(), stat.clone());
                }
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
                        .or_else(|| {
                            // For variant position aesthetics (e.g., pos1min, pos2max),
                            // fall back to the primary aesthetic's original name (pos1, pos2).
                            // This ensures tile's expanded min/max aesthetics inherit the
                            // original column name from the user's x/y mapping.
                            aesthetic::parse_position(aesthetic).and_then(|(slot, suffix)| {
                                if !suffix.is_empty() {
                                    let primary = format!("pos{}", slot);
                                    consumed_original_names.get(&primary).cloned()
                                } else {
                                    None
                                }
                            })
                        })
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
                        format!(
                            "{} AS {}",
                            naming::quote_ident(&stat_col),
                            naming::quote_ident(&prefixed_aes)
                        )
                    })
                })
                .collect();

            if stat_rename_exprs.is_empty() {
                transformed_query
            } else {
                format!(
                    "SELECT *, {} FROM ({}) AS \"__ggsql_stat__\"",
                    stat_rename_exprs.join(", "),
                    transformed_query
                )
            }
        }
        StatResult::Identity => query,
    };

    // Flip mappings back after stat transforms if we flipped them earlier
    // Now pos1/pos2 map to the user's intended x/y positions
    // Note: We only flip mappings here, not remappings. Remappings are flipped
    // later in mod.rs after apply_remappings_post_query creates the columns,
    // so that Phase 4.5 can flip those columns along with everything else.
    if needs_flip {
        flip_position_aesthetics(&mut layer.mappings.aesthetics);

        // Normalize mapping column names to match their aesthetic keys.
        // After flipping, pos1 might point to __ggsql_aes_pos2__ (and vice versa).
        // We update the column names so pos1 → __ggsql_aes_pos1__, etc.
        // The DataFrame columns will be renamed correspondingly in mod.rs.
        normalize_mapping_column_names(layer);
    }

    // Apply explicit ORDER BY if provided
    let final_query = if let Some(ref o) = order_by {
        format!("{} ORDER BY {}", final_query, o.as_str())
    } else {
        final_query
    };

    Ok(final_query)
}

/// Build a VALUES clause for an annotation layer with all aesthetic columns.
///
/// Generates SQL like: `WITH t(col1, col2) AS (VALUES (...), (...)) SELECT * FROM t`
///
/// This function:
/// 1. Moves position/required/array parameters from layer.parameters to layer.mappings
/// 2. Handles array recycling on-the-fly (determines max length, replicates scalars)
/// 3. Validates that all arrays have compatible lengths (1 or max)
/// 4. Builds the VALUES clause with raw aesthetic column names
/// 5. Converts parameter values to Column/AnnotationColumn mappings
///
/// For annotation layers:
/// - Position aesthetics (pos1, pos2): use Column (data coordinate space, participate in scales)
/// - Material aesthetics (color, size): use AnnotationColumn (visual space, identity scale)
///
/// # Arguments
///
/// * `layer` - The annotation layer with aesthetics in parameters (will be modified)
///
/// # Returns
///
/// A complete SQL expression ready to use as a FROM clause
fn process_annotation_layer(layer: &mut Layer, dialect: &dyn SqlDialect) -> Result<String> {
    use crate::plot::ArrayElement;

    // Step 1: Identify which parameters to use for annotation data
    // Only process position aesthetics, required aesthetics, and array parameters
    // (material non-required scalars stay in parameters as geom settings)
    let required_aesthetics = layer.geom.aesthetics().required();
    let supported_aesthetics = layer.geom.aesthetics().supported();
    let param_keys: Vec<String> = layer.parameters.keys().cloned().collect();

    // Collect parameters we'll use, checking criteria and filtering NULLs
    let mut annotation_params: Vec<(String, ParameterValue)> = Vec::new();

    for param_name in param_keys {
        // Skip if already in mappings
        if layer.mappings.contains_key(&param_name) {
            continue;
        }

        let Some(value) = layer.parameters.get(&param_name) else {
            continue;
        };

        // Filter out NULL aesthetics - they mean "use geom default"
        if value.is_null() {
            continue;
        }

        // Check if this is a position aesthetic OR a required aesthetic OR an array for supported aesthetic
        let is_position = crate::plot::aesthetic::is_position_aesthetic(&param_name);
        let is_required = required_aesthetics.contains(&param_name.as_str());
        let is_array = matches!(value, ParameterValue::Array(_))
            && supported_aesthetics.contains(&param_name.as_str());

        // Only process position/required/array parameters
        if is_position || is_required || is_array {
            annotation_params.push((param_name.clone(), value.clone()));
        }
    }

    // Step 2: Handle empty annotation_params by adding a dummy column
    // This occurs when geoms have no required aesthetics and user provides only
    // non-positional scalar parameters (e.g., PLACE rule SETTING stroke => 'red')
    if annotation_params.is_empty() {
        // Add a dummy column so we can generate a valid VALUES clause
        annotation_params.push(("__ggsql_dummy__".to_string(), ParameterValue::Number(1.0)));
    }

    // Step 3: Determine max array length from all annotation parameters
    let mut max_length = 1;

    for (aesthetic, value) in &annotation_params {
        // Only check array values
        let ParameterValue::Array(arr) = value else {
            continue;
        };

        let len = arr.len();
        if len <= 1 {
            continue;
        }

        if max_length > 1 && len != max_length {
            // Multiple different non-1 lengths - error
            return Err(GgsqlError::ValidationError(format!(
                "PLACE annotation layer has mismatched array lengths: '{}' has length {}, but another has length {}",
                aesthetic, len, max_length
            )));
        }
        if len > max_length {
            max_length = len;
        }
    }

    // Step 4: Build VALUES clause and create final mappings simultaneously
    let mut columns: Vec<Vec<ArrayElement>> = Vec::new();
    let mut column_names = Vec::new();

    for (aesthetic, param) in &annotation_params {
        // Build column data for VALUES clause using rep() to handle scalars and arrays uniformly
        let mut column_values = match param.clone().rep(max_length)? {
            ParameterValue::Array(arr) => arr,
            _ => unreachable!("rep() always returns Array variant"),
        };

        // Try to parse string elements as temporal types (Date/DateTime/Time)
        // This ensures literals like '1973-06-01' become Date columns, not String columns
        column_values = column_values
            .into_iter()
            .map(|elem| elem.try_as_temporal())
            .collect();

        columns.push(column_values);
        // Use raw aesthetic names (not prefixed) so annotations go through
        // the same column→aesthetic renaming pipeline as regular layers
        column_names.push(aesthetic.clone());

        // Skip creating mappings for dummy columns (they're just for valid SQL)
        if aesthetic == "__ggsql_dummy__" {
            continue;
        }

        // Create final mapping directly (no intermediate Literal step)
        let is_position = crate::plot::aesthetic::is_position_aesthetic(aesthetic);
        let mapping_value = if is_position {
            // Position aesthetics use Column (participate in scales)
            AestheticValue::Column {
                name: aesthetic.clone(), // Raw aesthetic name from VALUES clause
                original_name: None,
                is_dummy: false,
            }
        } else {
            // Material aesthetics use AnnotationColumn (identity scale)
            AestheticValue::AnnotationColumn {
                name: aesthetic.clone(), // Raw aesthetic name from VALUES clause
            }
        };

        layer.mappings.insert(aesthetic.clone(), mapping_value);
        // Remove from parameters now that it's in mappings
        layer.parameters.remove(aesthetic);
    }

    // Step 5: Build VALUES rows
    let values_clause = (0..max_length)
        .map(|i| {
            let row: Vec<String> = columns.iter().map(|col| col[i].to_sql(dialect)).collect();
            format!("({})", row.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ");

    // Step 6: Build complete SQL query
    let column_list = column_names
        .iter()
        .map(|c| naming::quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");

    let sql = format!(
        "WITH __ggsql_values__({}) AS (VALUES {}) SELECT * FROM __ggsql_values__",
        column_list, values_clause
    );

    Ok(sql)
}

/// Normalize mapping column names to match their aesthetic keys after flip-back.
///
/// After flipping position aesthetics, the mapping values (column names) may not match the keys.
/// For example, pos1 might point to `__ggsql_aes_pos2__`.
/// This function updates the column names so pos1 → `__ggsql_aes_pos1__`, etc.
///
/// This should be called after flipping during flip-back.
/// The DataFrame columns should be renamed correspondingly using `flip_dataframe_columns`.
fn normalize_mapping_column_names(layer: &mut Layer) {
    // Collect the aesthetics to update (to avoid borrowing issues)
    let aesthetics_to_update: Vec<String> = layer
        .mappings
        .aesthetics
        .keys()
        .filter(|aes| crate::plot::aesthetic::is_position_aesthetic(aes))
        .cloned()
        .collect();

    for aesthetic in aesthetics_to_update {
        // Literals are already converted to Columns by update_mappings_for_aesthetic_columns()
        if let Some(AestheticValue::Column { name, .. }) =
            layer.mappings.aesthetics.get_mut(&aesthetic)
        {
            *name = naming::aesthetic_column(&aesthetic);
        }
    }
}

/// Resolve orientation for all layers and apply mapping flips.
///
/// This function:
/// 1. Resolves orientation via auto-detection or explicit setting
/// 2. Stores resolved orientation in layer parameters
/// 3. Flips mappings for transposed layers
/// 4. Flips type_info column names to match flipped mappings
///
/// Must be called BEFORE building base queries, since build_layer_base_query
/// uses layer.mappings to create SQL like `column AS __ggsql_aes_pos1__`.
///
/// Note: Validation of orientation settings is handled by `validate_settings()`,
/// which rejects orientation for geoms that don't have it in default_params.
pub fn resolve_orientations(
    layers: &mut [Layer],
    scales: &[Scale],
    layer_type_info: &mut [Vec<super::schema::TypeInfo>],
    aesthetic_ctx: &AestheticContext,
) {
    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        let orientation = resolve_orientation(layer, scales);
        // Store resolved orientation in parameters for downstream use (writers need it)
        layer.parameters.insert(
            "orientation".to_string(),
            ParameterValue::String(orientation.to_string()),
        );
        if is_transposed(layer) {
            flip_position_aesthetics(&mut layer.mappings.aesthetics);
            // Also flip column names in type_info to match the flipped mappings
            if layer_idx < layer_type_info.len() {
                for (name, _, _) in &mut layer_type_info[layer_idx] {
                    if let Some(aesthetic) = naming::extract_aesthetic_name(name) {
                        let flipped = aesthetic_ctx.flip_position(aesthetic);
                        if flipped != aesthetic {
                            *name = naming::aesthetic_column(&flipped);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{ArrayElement, DataSource, Geom, Layer, ParameterValue};
    use crate::reader::AnsiDialect;

    #[test]
    fn test_annotation_single_scalar() {
        let mut layer = Layer::new(Geom::text());
        layer.source = Some(DataSource::Annotation);
        // Put values in parameters (not mappings) - process_annotation_layer will process them
        layer
            .parameters
            .insert("pos1".to_string(), ParameterValue::Number(5.0));
        layer
            .parameters
            .insert("pos2".to_string(), ParameterValue::Number(10.0));
        layer.parameters.insert(
            "label".to_string(),
            ParameterValue::String("Test".to_string()),
        );

        let result = process_annotation_layer(&mut layer, &AnsiDialect).unwrap();

        // Uses CTE form: WITH __ggsql_values__(cols) AS (VALUES (...)) SELECT * FROM __ggsql_values__
        // Check all values are present (order may vary due to HashMap)
        assert!(result.contains("VALUES"));
        assert!(result.contains("5"));
        assert!(result.contains("10"));
        assert!(result.contains("'Test'"));
        // Raw aesthetic names in column list
        assert!(result.contains("\"pos1\""));
        assert!(result.contains("\"pos2\""));
        assert!(result.contains("\"label\""));

        // After processing, mappings should have Column/AnnotationColumn values
        assert!(layer.mappings.contains_key("pos1"));
        assert!(layer.mappings.contains_key("pos2"));
        assert!(layer.mappings.contains_key("label"));
    }

    #[test]
    fn test_annotation_array_recycling() {
        let mut layer = Layer::new(Geom::text());
        layer.source = Some(DataSource::Annotation);
        layer.parameters.insert(
            "pos1".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(1.0),
                ArrayElement::Number(2.0),
                ArrayElement::Number(3.0),
            ]),
        );
        layer
            .parameters
            .insert("pos2".to_string(), ParameterValue::Number(10.0));
        layer.parameters.insert(
            "label".to_string(),
            ParameterValue::String("Same".to_string()),
        );

        let result = process_annotation_layer(&mut layer, &AnsiDialect).unwrap();

        // Should recycle scalar pos2 and label to match array length (3)
        assert!(result.contains("VALUES"));
        // Check that all values appear (order may vary due to HashMap)
        assert!(result.contains("1") && result.contains("2") && result.contains("3"));
        assert!(result.contains("10"));
        assert!(result.contains("'Same'"));
        // Check row count by counting value tuples (3 rows)
        assert_eq!(result.matches("), (").count() + 1, 3, "Should have 3 rows");
    }

    #[test]
    fn test_annotation_mismatched_arrays() {
        let mut layer = Layer::new(Geom::text());
        layer.source = Some(DataSource::Annotation);
        layer.parameters.insert(
            "pos1".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Number(1.0),
                ArrayElement::Number(2.0),
                ArrayElement::Number(3.0),
            ]),
        );
        layer.parameters.insert(
            "pos2".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(10.0), ArrayElement::Number(20.0)]),
        );

        let result = process_annotation_layer(&mut layer, &AnsiDialect);

        // Should error with mismatched lengths
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("mismatched array lengths"),
            "Error message should mention mismatched arrays"
        );
        // Error should mention one of the aesthetics (order may vary)
        assert!(
            err_msg.contains("pos1") || err_msg.contains("pos2"),
            "Error message should mention at least one aesthetic"
        );
    }

    #[test]
    fn test_annotation_multiple_arrays_same_length() {
        let mut layer = Layer::new(Geom::text());
        layer.source = Some(DataSource::Annotation);
        layer.parameters.insert(
            "pos1".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(1.0), ArrayElement::Number(2.0)]),
        );
        layer.parameters.insert(
            "pos2".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(10.0), ArrayElement::Number(20.0)]),
        );

        let result = process_annotation_layer(&mut layer, &AnsiDialect).unwrap();

        // Both arrays have length 2, should work (order may vary)
        assert!(result.contains("VALUES"));
        assert!(result.contains("1") && result.contains("2"));
        assert!(result.contains("10") && result.contains("20"));
        // Check row count by counting value tuples (2 rows)
        assert_eq!(result.matches("), (").count() + 1, 2, "Should have 2 rows");
    }

    #[test]
    fn test_annotation_mixed_types() {
        let mut layer = Layer::new(Geom::text());
        layer.source = Some(DataSource::Annotation);
        layer
            .parameters
            .insert("pos1".to_string(), ParameterValue::Number(5.0));
        layer
            .parameters
            .insert("pos2".to_string(), ParameterValue::Number(10.0));
        layer.parameters.insert(
            "label".to_string(),
            ParameterValue::String("Text".to_string()),
        );

        let result = process_annotation_layer(&mut layer, &AnsiDialect).unwrap();

        // Should handle different types (order may vary)
        assert!(result.contains("VALUES"));
        assert!(result.contains("5"));
        assert!(result.contains("10"));
        assert!(result.contains("'Text'"));
    }

    #[test]
    fn test_literal_to_array_date_parsing() {
        use arrow::array::Array;
        use arrow::datatypes::DataType;

        // Date literal should parse to Date32 type
        let array = literal_to_array(&ParameterValue::String("1973-06-01".to_string()), 5);
        assert_eq!(
            array.data_type(),
            &DataType::Date32,
            "Date string should parse to Date32 type"
        );
        assert_eq!(array.len(), 5);
    }

    #[test]
    fn test_literal_to_array_datetime_parsing() {
        use arrow::array::Array;
        use arrow::datatypes::{DataType, TimeUnit};

        // DateTime literal should parse to Timestamp type
        let array = literal_to_array(
            &ParameterValue::String("2024-03-17T14:30:00".to_string()),
            3,
        );
        assert!(
            matches!(
                array.data_type(),
                DataType::Timestamp(TimeUnit::Microsecond, None)
            ),
            "DateTime string should parse to Timestamp type"
        );
        assert_eq!(array.len(), 3);
    }

    #[test]
    fn test_literal_to_array_time_parsing() {
        use arrow::array::Array;
        use arrow::datatypes::{DataType, TimeUnit};

        // Time literal should parse to Time64 type
        let array = literal_to_array(&ParameterValue::String("14:30:00".to_string()), 4);
        assert_eq!(
            array.data_type(),
            &DataType::Time64(TimeUnit::Nanosecond),
            "Time string should parse to Time64 type"
        );
        assert_eq!(array.len(), 4);
    }

    #[test]
    fn test_literal_to_array_string_fallback() {
        use arrow::array::Array;
        use arrow::datatypes::DataType;

        // Non-temporal string should remain Utf8 type
        let array = literal_to_array(&ParameterValue::String("not a date".to_string()), 2);
        assert_eq!(
            array.data_type(),
            &DataType::Utf8,
            "Non-temporal string should remain Utf8 type"
        );
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_annotation_no_required_aesthetics() {
        // Rule geom has no required aesthetics, only optional ones
        let mut layer = Layer::new(Geom::rule());
        layer.source = Some(DataSource::Annotation);
        // Only non-positional, non-required scalar parameters
        layer.parameters.insert(
            "stroke".to_string(),
            ParameterValue::String("red".to_string()),
        );
        layer
            .parameters
            .insert("linewidth".to_string(), ParameterValue::Number(2.0));

        let result = process_annotation_layer(&mut layer, &AnsiDialect);

        // Should generate valid SQL with a dummy column
        match result {
            Ok(sql) => {
                // Check that SQL is valid (has VALUES with at least one column)
                assert!(
                    !sql.contains("(VALUES ) AS t()"),
                    "Should not generate empty VALUES clause"
                );
                assert!(
                    sql.contains("VALUES") && sql.contains("WITH __ggsql_values__"),
                    "Should have VALUES with at least one column"
                );
                // Should contain the dummy column
                assert!(
                    sql.contains("__ggsql_dummy__"),
                    "Should have dummy column when no data columns exist"
                );
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }

        // Verify that stroke and linewidth remain in parameters (not moved to mappings)
        assert!(
            layer.parameters.contains_key("stroke"),
            "Non-positional, non-required parameters should stay in parameters"
        );
        assert!(
            layer.parameters.contains_key("linewidth"),
            "Non-positional, non-required parameters should stay in parameters"
        );

        // Verify dummy column is not in mappings
        assert!(
            !layer.mappings.contains_key("__ggsql_dummy__"),
            "Dummy column should not be added to mappings"
        );
    }
}
