//! Layer query building, data transforms, and stat application.
//!
//! This module handles building SQL queries for layers, applying pre-stat
//! transformations, stat transforms, and post-query operations.

use crate::plot::aesthetic::{is_positional_aesthetic, AestheticContext};
use crate::plot::layer::is_transposed;
use crate::plot::layer::orientation::{flip_positional_aesthetics, resolve_orientation};
use crate::plot::{
    AestheticValue, DefaultAestheticValue, Layer, ParameterValue, Scale, Schema, SqlTypeNames,
    StatResult,
};
use crate::{naming, DataFrame, GgsqlError, Result};
use polars::prelude::DataType;
use std::collections::{HashMap, HashSet};

use super::casting::{literal_to_sql, TypeRequirement};
use super::schema::build_aesthetic_schema;

/// Build the source query for a layer.
///
/// Returns `SELECT * FROM source` where source is either:
/// - The layer's explicit source (table, CTE, file)
/// - The global table if layer has no explicit source
///
/// Note: This is distinct from `build_layer_base_query()` which builds a full
/// SELECT with aesthetic column renames and type casts.
pub fn layer_source_query(
    layer: &Layer,
    materialized_ctes: &HashSet<String>,
    has_global: bool,
) -> String {
    let source = super::casting::determine_layer_source(layer, materialized_ctes, has_global);
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
pub fn build_layer_select_list(
    layer: &Layer,
    type_requirements: &[TypeRequirement],
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
    use polars::prelude::IntoColumn;

    let mut df = df;
    let row_count = df.height();

    // Apply remappings: stat columns → prefixed aesthetic names
    // e.g., __ggsql_stat_count → __ggsql_aes_y__
    // Remappings structure: HashMap<target_aesthetic, AestheticValue pointing to stat column>
    for (target_aesthetic, value) in &layer.remappings.aesthetics {
        let target_col_name = naming::aesthetic_column(target_aesthetic);

        match value {
            AestheticValue::Column { name, .. } => {
                // Check if this stat column exists in the DataFrame
                if df.column(name).is_ok() {
                    df.rename(name, target_col_name.into()).map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to rename stat column '{}' to '{}': {}",
                            name, target_aesthetic, e
                        ))
                    })?;
                }
            }
            AestheticValue::Literal(lit) => {
                // Add constant column for literal values
                let series = literal_to_series(&target_col_name, lit, row_count);
                df = df
                    .with_column(series.into_column())
                    .map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to add literal column '{}': {}",
                            target_col_name, e
                        ))
                    })?
                    .clone();
            }
        }
    }

    Ok(df)
}

/// Convert a literal value to a Polars Series with constant values.
pub fn literal_to_series(name: &str, lit: &ParameterValue, len: usize) -> polars::prelude::Series {
    use polars::prelude::{NamedFrom, Series};

    match lit {
        ParameterValue::Number(n) => Series::new(name.into(), vec![*n; len]),
        ParameterValue::String(s) => Series::new(name.into(), vec![s.as_str(); len]),
        ParameterValue::Boolean(b) => Series::new(name.into(), vec![*b; len]),
        ParameterValue::Array(_) | ParameterValue::Null => {
            unreachable!("Grammar prevents arrays and null in literal aesthetic mappings")
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
/// * `type_names` - SQL type names for the database backend
pub fn apply_pre_stat_transform(
    query: &str,
    layer: &Layer,
    full_schema: &Schema,
    aesthetic_schema: &Schema,
    scales: &[Scale],
    type_names: &SqlTypeNames,
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
    // This avoids SELECT * EXCLUDE which has portability issues
    // (Polars SQL silently drops re-added columns with the same name).
    let mut seen: HashSet<&str> = HashSet::new();
    let combined_cols = full_schema.iter().chain(aesthetic_schema.iter());

    let select_exprs: Vec<String> = combined_cols
        .filter(|col| seen.insert(&col.name))
        .map(|col| {
            if let Some((_, sql)) = transform_exprs.iter().find(|(c, _)| c == &col.name) {
                format!("{} AS \"{}\"", sql, col.name)
            } else {
                format!("\"{}\"", col.name)
            }
        })
        .collect();

    format!(
        "SELECT {} FROM ({}) AS __ggsql_pre__",
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
pub fn build_layer_base_query(
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
            "SELECT {} FROM ({}) AS __ggsql_src__ WHERE {}",
            select_clause,
            source_query,
            f.as_str()
        )
    } else {
        format!(
            "SELECT {} FROM ({}) AS __ggsql_src__",
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
/// * `type_names` - SQL type names for the database backend
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
    type_names: &SqlTypeNames,
    execute_query: &F,
) -> Result<String>
where
    F: Fn(&str) -> Result<DataFrame>,
{
    use crate::plot::layer::orientation::flip_positional_aesthetics;

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
        type_names,
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
    )?;

    // Flip user remappings BEFORE merging defaults for Transposed orientation.
    // User remappings are in user orientation (e.g., `count AS x` for horizontal histogram).
    // We flip them to aligned orientation so they're uniform with defaults.
    // At the end, we flip everything back together.
    if needs_flip {
        flip_positional_aesthetics(&mut layer.remappings.aesthetics);
    }

    // Apply literal default remappings from geom defaults (e.g., y2 => 0.0 for bar baseline).
    // These apply regardless of stat transform, but only if user hasn't overridden them.
    // Defaults are always in aligned orientation.
    for (aesthetic, default_value) in layer.geom.default_remappings() {
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

            for (aesthetic, default_value) in layer.geom.default_remappings() {
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

                // If the transformed query uses CTEs (WITH ... SELECT ...),
                // we can't wrap it in a subquery because Polars SQL doesn't
                // support CTEs inside subqueries. Instead, split into CTE
                // prefix + trailing SELECT, then append the trailing SELECT
                // as another CTE and add the rename SELECT on top.
                if let Some((cte_prefix, trailing_select)) =
                    crate::parser::SourceTree::new(&transformed_query)
                        .ok()
                        .as_ref()
                        .and_then(super::cte::split_with_query)
                {
                    format!(
                        "{}, __ggsql_stat__ AS ({}) SELECT * {}, {} FROM __ggsql_stat__",
                        cte_prefix,
                        trailing_select,
                        exclude_clause,
                        stat_rename_exprs.join(", ")
                    )
                } else {
                    format!(
                        "SELECT * {}, {} FROM ({}) AS __ggsql_stat__",
                        exclude_clause,
                        stat_rename_exprs.join(", "),
                        transformed_query
                    )
                }
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
        flip_positional_aesthetics(&mut layer.mappings.aesthetics);

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

/// Normalize mapping column names to match their aesthetic keys after flip-back.
///
/// After flipping positional aesthetics, the mapping values (column names) may not match the keys.
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
        .filter(|aes| crate::plot::aesthetic::is_positional_aesthetic(aes))
        .cloned()
        .collect();

    for aesthetic in aesthetics_to_update {
        if let Some(value) = layer.mappings.aesthetics.get_mut(&aesthetic) {
            let expected_col = naming::aesthetic_column(&aesthetic);
            match value {
                AestheticValue::Column { name, .. } => {
                    *name = expected_col;
                }
                AestheticValue::Literal(_) => {
                    *value = AestheticValue::standard_column(expected_col);
                }
            }
        }
    }
}

/// Flip positional column names in a DataFrame for Transposed orientation layers.
///
/// Swaps column names like `__ggsql_aes_pos1__` ↔ `__ggsql_aes_pos2__` so that
/// the data matches the flipped mapping names.
///
/// This is called after query execution for layers with Transposed orientation,
/// in coordination with `normalize_mapping_column_names` which updates the mappings.
pub fn flip_dataframe_positional_columns(
    df: DataFrame,
    aesthetic_ctx: &AestheticContext,
) -> DataFrame {
    use polars::prelude::*;

    // Collect renames needed before consuming df
    let renames: Vec<(String, String)> = df
        .get_column_names()
        .iter()
        .filter_map(|col_name| {
            naming::extract_aesthetic_name(col_name).and_then(|aesthetic| {
                if is_positional_aesthetic(aesthetic) {
                    let flipped = aesthetic_ctx.flip_positional(aesthetic);
                    if flipped != aesthetic {
                        return Some((col_name.to_string(), naming::aesthetic_column(&flipped)));
                    }
                }
                None
            })
        })
        .collect();

    if renames.is_empty() {
        return df;
    }

    let mut lazy = df.lazy();

    // First pass: rename to temp names
    for (from, to) in &renames {
        let temp = format!("{}_temp", to);
        lazy = lazy.rename([from.as_str()], [temp.as_str()], true);
    }

    // Second pass: remove temp suffix
    for (_, to) in &renames {
        let temp = format!("{}_temp", to);
        lazy = lazy.rename([temp.as_str()], [to.as_str()], true);
    }

    lazy.collect().expect("rename should not fail")
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
            flip_positional_aesthetics(&mut layer.mappings.aesthetics);
            // Also flip column names in type_info to match the flipped mappings
            if layer_idx < layer_type_info.len() {
                for (name, _, _) in &mut layer_type_info[layer_idx] {
                    if let Some(aesthetic) = naming::extract_aesthetic_name(name) {
                        let flipped = aesthetic_ctx.flip_positional(aesthetic);
                        if flipped != aesthetic {
                            *name = naming::aesthetic_column(&flipped);
                        }
                    }
                }
            }
        }
    }
}
