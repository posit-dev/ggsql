//! Query execution module for ggsql
//!
//! Provides shared execution logic for building data maps from queries,
//! handling both global SQL and layer-specific data sources.
//!
//! This module is organized into submodules:
//! - `cte`: CTE extraction, transformation, and materialization
//! - `schema`: Schema extraction, type inference, and min/max ranges
//! - `casting`: Type requirements determination and casting logic
//! - `layer`: Layer query building, data transforms, and stat application
//! - `scale`: Scale creation, resolution, type coercion, and OOB handling

mod casting;
mod cte;
mod layer;
mod scale;
mod schema;

// Re-export public API
pub use casting::TypeRequirement;
pub use cte::CteDefinition;
pub use schema::TypeInfo;

use crate::naming;
use crate::parser;
use crate::plot::aesthetic::{primary_aesthetic, ALL_POSITIONAL};
use crate::plot::{AestheticValue, Layer, Scale, ScaleTypeKind, Schema};
use crate::{DataFrame, GgsqlError, Plot, Result};
use std::collections::{HashMap, HashSet};

use crate::reader::Reader;

#[cfg(all(feature = "duckdb", test))]
use crate::reader::DuckDBReader;

// =============================================================================
// Validation
// =============================================================================

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

// =============================================================================
// Global Mapping & Color Splitting
// =============================================================================

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

/// Let 'color' aesthetics fill defaults for the 'stroke' and 'fill' aesthetics.
/// Also splits 'color' scale to 'fill' and 'stroke' scales.
/// Removes 'color' from both mappings and scales after splitting to avoid
/// non-deterministic behavior from HashMap iteration order.
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

// =============================================================================
// Discrete Column Handling
// =============================================================================

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
            // Skip positional aesthetics - these should not trigger auto-grouping.
            // Stats that need to group by positional aesthetics (like bar/histogram)
            // already handle this themselves via stat_consumed_aesthetics().
            if ALL_POSITIONAL.iter().any(|s| s == aesthetic) {
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
                let primary_aesthetic = primary_aesthetic(aesthetic);
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

                // Use the prefixed aesthetic column name, since the query renames
                // columns to prefixed names (e.g., island → __ggsql_aes_fill__)
                let aes_col_name = naming::aesthetic_column(aesthetic);

                // Skip if already in partition_by
                if layer.partition_by.contains(&aes_col_name) {
                    continue;
                }

                layer.partition_by.push(aes_col_name);
            }
        }
    }
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

// =============================================================================
// Public API: PreparedData
// =============================================================================

/// Result of preparing data for visualization
pub struct PreparedData {
    /// Data map with global and layer-specific DataFrames
    pub data: HashMap<String, DataFrame>,
    /// Parsed and resolved visualization specifications
    pub specs: Vec<Plot>,
    /// The SQL portion of the query
    pub sql: String,
    /// The VISUALISE portion of the query
    pub visual: String,
}

/// Build data map from a query using a Reader
///
/// This is the main entry point for preparing visualization data from a ggsql query.
///
/// # Arguments
/// * `query` - The full ggsql query string
/// * `reader` - A Reader implementation for executing SQL
pub fn prepare_data_with_reader<R: Reader>(query: &str, reader: &R) -> Result<PreparedData> {
    let execute_query = |sql: &str| reader.execute_sql(sql);
    let type_names = reader.sql_type_names();

    // Parse once and create SourceTree
    let source_tree = parser::SourceTree::new(query)?;
    source_tree.validate()?;

    // Check if query has VISUALISE statements
    let root = source_tree.root();
    if source_tree
        .find_node(&root, "(visualise_statement) @viz")
        .is_none()
    {
        return Err(GgsqlError::ValidationError(
            "No visualization specifications found".to_string(),
        ));
    }

    // Build AST from existing tree
    let mut specs = parser::build_ast(&source_tree)?;

    if specs.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No visualization specifications found".to_string(),
        ));
    }

    // Extract CTE definitions from the source tree (in declaration order)
    let ctes = cte::extract_ctes(&source_tree);

    // Materialize CTEs as registered tables via reader.register()
    let materialized_ctes = cte::materialize_ctes(&ctes, reader)?;

    // Build data map for multi-source support
    let mut data_map: HashMap<String, DataFrame> = HashMap::new();

    // Extract SQL once (reused later for PreparedData)
    let sql_part = source_tree.extract_sql();

    // Execute global SQL if present
    // If there's a WITH clause, extract just the trailing SELECT and transform CTE references.
    // The global result is stored as a temp table so filtered layers can query it efficiently.
    // Track whether we actually create the temp table (depends on transform_global_sql succeeding)
    let mut has_global_table = false;
    if sql_part.is_some() {
        if let Some(transformed_sql) = cte::transform_global_sql(&source_tree, &materialized_ctes) {
            // Execute global result SQL and register result as a temp table
            let df = execute_query(&transformed_sql)?;
            reader.register(&naming::global_table(), df, true)?;

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
        .map(|l| layer::layer_source_query(l, &materialized_ctes, has_global_table))
        .collect();

    // Get types for each layer from source queries (Phase 1: types only, no min/max yet)
    let mut layer_type_info: Vec<Vec<schema::TypeInfo>> = Vec::new();
    for source_query in &layer_source_queries {
        let type_info = schema::fetch_schema_types(source_query, &execute_query)?;
        layer_type_info.push(type_info);
    }

    // Initial schemas (types only, no min/max - will be completed after base queries)
    let mut layer_schemas: Vec<Schema> = layer_type_info
        .iter()
        .map(|ti| schema::type_info_to_schema(ti))
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
    schema::add_literal_columns_to_type_info(&specs[0].layers, &mut layer_type_info);

    // Rebuild layer schemas with constant columns included
    layer_schemas = layer_type_info
        .iter()
        .map(|ti| schema::type_info_to_schema(ti))
        .collect();

    // Validate all layers against their schemas
    // This must happen BEFORE build_layer_query because stat transforms remove consumed aesthetics
    validate(&specs[0].layers, &layer_schemas)?;

    // Create scales for all mapped aesthetics that don't have explicit SCALE clauses
    scale::create_missing_scales(&mut specs[0]);

    // Resolve scale types and transforms early based on column dtypes
    scale::resolve_scale_types_and_transforms(&mut specs[0], &layer_type_info)?;

    // Determine which columns need type casting
    let type_requirements =
        casting::determine_type_requirements(&specs[0], &layer_type_info, &type_names);

    // Update type info with post-cast dtypes
    // This ensures subsequent schema extraction and scale resolution see the correct types
    for (layer_idx, requirements) in type_requirements.iter().enumerate() {
        if layer_idx < layer_type_info.len() {
            casting::update_type_info_for_casting(&mut layer_type_info[layer_idx], requirements);
        }
    }

    // Build layer base queries using build_layer_base_query()
    // These include: SELECT with aesthetic renames, casts from type_requirements, filters
    // Note: This is Part 1 of the split - base queries that can be used for schema completion
    let layer_base_queries: Vec<String> = specs[0]
        .layers
        .iter()
        .enumerate()
        .map(|(idx, l)| {
            layer::build_layer_base_query(l, &layer_source_queries[idx], &type_requirements[idx])
        })
        .collect();

    // Clone facet for apply_layer_transforms
    let facet = specs[0].facet.clone();

    // Complete schemas with min/max from base queries (Phase 2: ranges from cast data)
    // Base queries include casting via build_layer_select_list, so min/max reflect cast types
    for (idx, base_query) in layer_base_queries.iter().enumerate() {
        layer_schemas[idx] =
            schema::complete_schema_ranges(base_query, &layer_type_info[idx], &execute_query)?;
    }

    // Pre-resolve Binned scales using schema-derived context
    // This must happen before apply_layer_transforms so pre_stat_transform_sql has resolved breaks
    scale::apply_pre_stat_resolve(&mut specs[0], &layer_schemas)?;

    // Add discrete mapped columns to partition_by for all layers
    let scales = specs[0].scales.clone();
    add_discrete_columns_to_partition_by(&mut specs[0].layers, &layer_schemas, &scales);

    // Clone scales for apply_layer_transforms
    let scales = specs[0].scales.clone();

    // Build final layer queries using apply_layer_transforms (Part 2 of the split)
    // This applies: pre-stat transforms, stat transforms, ORDER BY
    let mut layer_queries: Vec<String> = Vec::new();

    for (idx, l) in specs[0].layers.iter_mut().enumerate() {
        // Validate weight aesthetic is a column, not a literal
        if let Some(weight_value) = l.mappings.aesthetics.get("weight") {
            if weight_value.is_literal() {
                return Err(GgsqlError::ValidationError(
                    "Bar weight aesthetic must be a column, not a literal".to_string(),
                ));
            }
        }

        // Apply default parameter values (e.g., bins=30 for histogram)
        l.apply_default_params();

        // Apply stat transforms and ORDER BY (Part 2)
        let layer_query = layer::apply_layer_transforms(
            l,
            &layer_base_queries[idx],
            &layer_schemas[idx],
            facet.as_ref(),
            &scales,
            &type_names,
            &execute_query,
        )?;
        layer_queries.push(layer_query);
    }

    // Phase 2: Deduplicate and execute unique queries
    let mut query_to_result: HashMap<String, DataFrame> = HashMap::new();
    for (idx, q) in layer_queries.iter().enumerate() {
        if !query_to_result.contains_key(q) {
            let df = execute_query(q).map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "Failed to fetch data for layer {}: {}",
                    idx + 1,
                    e
                ))
            })?;
            query_to_result.insert(q.clone(), df);
        }
    }

    // Phase 3: Assign data to layers (clone only when needed)
    // Key by (query, serialized_remappings) to detect when layers can share data
    // Layers with identical query AND remappings share data via data_key
    let mut config_to_key: HashMap<(String, String), String> = HashMap::new();

    for (idx, q) in layer_queries.iter().enumerate() {
        let layer = &mut specs[0].layers[idx];
        let remappings_key = serde_json::to_string(&layer.remappings).unwrap_or_default();
        let config_key = (q.clone(), remappings_key);

        if let Some(existing_key) = config_to_key.get(&config_key) {
            // Same query AND same remappings - share data
            layer.data_key = Some(existing_key.clone());
        } else {
            // Need own data entry (either first occurrence or different remappings)
            let layer_key = naming::layer_key(idx);
            let df = query_to_result.get(q).unwrap().clone();
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
    for l in specs[0].layers.iter_mut() {
        if let Some(ref key) = l.data_key {
            if processed_keys.insert(key.clone()) {
                // First time seeing this data - process it
                if let Some(df) = data_map.remove(key) {
                    let df_with_remappings = layer::apply_remappings_post_query(df, l)?;
                    data_map.insert(key.clone(), df_with_remappings);
                }
            }
            // Update layer mappings for all layers (even if data shared)
            l.update_mappings_for_remappings();
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
        scale::create_missing_scales_post_stat(spec);
    }

    // Post-process specs: compute aesthetic labels
    for spec in &mut specs {
        // Compute aesthetic labels (uses first non-constant column, respects user-specified labels)
        spec.compute_aesthetic_labels();
    }

    // Resolve scale types from data for scales without explicit types
    for spec in &mut specs {
        scale::resolve_scales(spec, &mut data_map)?;
    }

    // Apply post-stat binning for Binned scales on remapped aesthetics
    // This handles cases like SCALE BINNED fill when fill is remapped from count
    for spec in &specs {
        scale::apply_post_stat_binning(spec, &mut data_map)?;
    }

    // Apply out-of-bounds handling to data based on scale oob properties
    for spec in &specs {
        scale::apply_scale_oob(spec, &mut data_map)?;
    }

    // Prune unnecessary columns from each layer's DataFrame
    prune_dataframes_per_layer(&specs, &mut data_map)?;

    // Extract VISUALISE text for PreparedData (SQL already extracted earlier)
    let visual_part = source_tree.extract_visualise().unwrap_or_default();

    Ok(PreparedData {
        data: data_map,
        specs,
        sql: sql_part.unwrap_or_default(),
        visual: visual_part,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_global_only() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point";

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // With the new approach, every layer has its own data (no GLOBAL_DATA_KEY)
        assert!(result.data.contains_key(&naming::layer_key(0)));
        assert_eq!(result.specs.len(), 1);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_prepare_data_no_viz() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y";

        let result = prepare_data_with_reader(query, &reader);
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

        let result = prepare_data_with_reader(query, &reader).unwrap();

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

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer with filter creates its own data - global data is NOT needed in data_map
        assert!(!result.data.contains_key(naming::GLOBAL_DATA_KEY));
        assert!(result.data.contains_key(&naming::layer_key(0)));

        // Layer 0 should have only 2 rows (filtered to category = 'A')
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 2);
    }

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

        let result = prepare_data_with_reader(query, &reader).unwrap();

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

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have layer 0 data with binned results
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have prefixed aesthetic-named columns
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        assert!(
            col_names.contains(&x_col),
            "Should have '{}' column: {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Should have '{}' column: {:?}",
            y_col,
            col_names
        );

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

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have layer 0 data with counted results
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have 3 rows (3 unique categories: A, B, C)
        assert_eq!(layer_df.height(), 3);

        // With new approach, columns are renamed to prefixed aesthetic names
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

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer should have original 3 rows (no stat transform when y is mapped)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer_df.height(), 3);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_bar_adds_y2_zero_for_baseline() {
        // Bar geom should add y2=0 to ensure bars have a baseline
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE bar_y2_test AS SELECT * FROM (VALUES
                    ('A', 10), ('B', 20), ('C', 30)
                ) AS t(category, value)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM bar_y2_test
            VISUALISE category AS x, value AS y
            DRAW bar
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();
        let layer = &result.specs[0].layers[0];

        // Layer should have yend in mappings (added by default for bar)
        assert!(
            layer.mappings.aesthetics.contains_key("yend"),
            "Bar should have yend mapping for baseline: {:?}",
            layer.mappings.aesthetics.keys().collect::<Vec<_>>()
        );

        // The DataFrame should have the yend column with 0 values
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let yend_col = naming::aesthetic_column("yend");
        assert!(
            layer_df.column(&yend_col).is_ok(),
            "DataFrame should have '{}' column: {:?}",
            yend_col,
            layer_df.get_column_names_str()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_resolve_scales_numeric_to_continuous() {
        // Test that numeric columns infer Continuous scale type
        use crate::plot::ScaleType;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 1.0 as x, 2.0 as y FROM (VALUES (1))
            VISUALISE x, y
            DRAW point
            SCALE x FROM [0, 100]
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();
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
    fn test_resolve_scales_string_to_discrete() {
        // Test that string columns infer Discrete scale type
        use crate::plot::ScaleType;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 'A' as category, 100 as value FROM (VALUES (1))
            VISUALISE category AS x, value AS y
            DRAW bar
            SCALE x FROM ['A', 'B', 'C']
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();
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

        let result = prepare_data_with_reader(query, &reader).unwrap();

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

    /// Test that literal mappings survive stat transforms (e.g., histogram grouping).
    ///
    /// This tests the fix for issue #129 where literal aesthetic columns like
    /// `'foo' AS stroke` were lost during stat transforms because they weren't
    /// included in the GROUP BY clause.
    #[cfg(feature = "duckdb")]
    #[test]
    fn test_histogram_with_literal_mapping() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE hist_literal_test AS SELECT RANDOM() * 100 as value FROM range(100)",
                duckdb::params![],
            )
            .unwrap();

        // Histogram with a literal stroke mapping - should preserve the literal column
        let query = r#"
            SELECT * FROM hist_literal_test
            VISUALISE value AS x
            DRAW histogram MAPPING 'foo' AS stroke
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have layer 0 data with binned results
        assert!(result.data.contains_key(&naming::layer_key(0)));
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();

        // Should have prefixed aesthetic-named columns
        let col_names: Vec<String> = layer_df
            .get_column_names_str()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let x_col = naming::aesthetic_column("x");
        let y_col = naming::aesthetic_column("y");
        let stroke_col = naming::aesthetic_column("stroke");

        assert!(
            col_names.contains(&x_col),
            "Should have '{}' column: {:?}",
            x_col,
            col_names
        );
        assert!(
            col_names.contains(&y_col),
            "Should have '{}' column: {:?}",
            y_col,
            col_names
        );
        // The literal stroke column should survive the stat transform
        assert!(
            col_names.contains(&stroke_col),
            "Should have '{}' column (literal mapping should survive stat transform): {:?}",
            stroke_col,
            col_names
        );

        // Should have fewer rows than original (binned)
        assert!(layer_df.height() < 100);
    }
}
