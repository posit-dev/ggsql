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
mod position;
mod scale;
mod schema;

// Re-export public API
pub use casting::TypeRequirement;
pub use cte::CteDefinition;
pub use schema::TypeInfo;

use crate::naming;
use crate::parser;
use crate::plot::aesthetic::{is_position_aesthetic, AestheticContext};
use crate::plot::facet::{resolve_properties as resolve_facet_properties, FacetDataContext};
use crate::plot::layer::is_transposed;
use crate::plot::{AestheticValue, Layer, Scale, ScaleTypeKind, Schema};
use crate::{DataFrame, DataSource, GgsqlError, Plot, Result};
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
fn validate(
    layers: &[Layer],
    layer_schemas: &[Schema],
    aesthetic_context: &Option<AestheticContext>,
) -> Result<()> {
    let translate = |aes: &str| -> String {
        match aesthetic_context {
            Some(ctx) => ctx.map_internal_to_user(aes),
            None => aes.to_string(),
        }
    };

    for (idx, (layer, schema)) in layers.iter().zip(layer_schemas.iter()).enumerate() {
        let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();
        let supported = layer.geom.aesthetics().supported();

        // Validate required aesthetics for this geom
        layer
            .validate_mapping(aesthetic_context, false)
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
                        translate(aesthetic),
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
        // REMAPPING can target any aesthetic (including Delayed ones from stat transforms)
        let aesthetics_info = layer.geom.aesthetics();
        for target_aesthetic in layer.remappings.aesthetics.keys() {
            if !aesthetics_info.contains(target_aesthetic) {
                return Err(GgsqlError::ValidationError(format!(
                    "Layer {}: REMAPPING targets unsupported aesthetic '{}' for geom '{}'",
                    idx + 1,
                    translate(target_aesthetic),
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
                            crate::and_list(valid_stat_columns)
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// Column Name Normalization
// =============================================================================

/// Rewrite `name` to match the schema's casing (case-insensitive resolution).
///
/// SQL treats unquoted identifiers as case-insensitive, so users may write
/// `VISUALISE CATEGORY AS x` even when DuckDB returns the column as `category`.
/// ggsql's own validator and generated SQL treat column names case-sensitively,
/// so we reconcile by rewriting the user-written name to the schema's casing
/// before either runs.
///
/// Exact match wins. Otherwise, if exactly one case-insensitive match exists,
/// `name` is rewritten to that match. Ambiguous matches (e.g. schema has both
/// `"Foo"` and `"foo"` and user wrote `FOO`) and missing references are left
/// untouched so the existing validator can report them with its normal error.
fn normalize_column_ref(name: &mut String, schema_names: &[&str]) {
    if schema_names.contains(&name.as_str()) {
        return;
    }
    let name_lower = name.to_lowercase();
    let mut match_iter = schema_names
        .iter()
        .filter(|s| s.to_lowercase() == name_lower);
    if let Some(first) = match_iter.next() {
        if match_iter.next().is_none() {
            *name = (*first).to_string();
        }
    }
}

/// Normalize all user-written column references in `specs` against their layer
/// schemas.
///
/// Runs after `merge_global_mappings_into_layers` so every aesthetic that a
/// layer will consult is already attached to that layer's `mappings`; each
/// layer can then be normalized against its own schema. This matters for
/// multi-source layers (e.g. `MAPPING ... FROM temps` vs `... FROM ozone`),
/// where the schemas — and the column casings — can legitimately differ.
///
/// Covers aesthetic `Column` values and `partition_by` per layer, plus
/// user-written facet variables on the plot-level `FACET` clause.
fn normalize_column_references(specs: &mut [Plot], layer_schemas: &[Schema]) {
    for spec in specs {
        for (layer, schema) in spec.layers.iter_mut().zip(layer_schemas.iter()) {
            if matches!(layer.source, Some(DataSource::Annotation)) {
                continue;
            }
            let names: Vec<&str> = schema.iter().map(|c| c.name.as_str()).collect();
            for value in layer.mappings.aesthetics.values_mut() {
                if let AestheticValue::Column { name, .. } = value {
                    normalize_column_ref(name, &names);
                }
            }
            for col in &mut layer.partition_by {
                normalize_column_ref(col, &names);
            }
        }

        // Facet variables are plot-level. Normalize against the first layer
        // whose schema contains the variable (case-insensitively). If no
        // layer matches, leave it — `add_facet_mappings_to_layers` simply
        // won't inject a mapping for layers that don't have the column.
        if let Some(facet) = spec.facet.as_mut() {
            let normalize_var = |var: &mut String| {
                for schema in layer_schemas {
                    let names: Vec<&str> = schema.iter().map(|c| c.name.as_str()).collect();
                    let before = var.clone();
                    normalize_column_ref(var, &names);
                    if *var != before || names.contains(&var.as_str()) {
                        break;
                    }
                }
            };
            match &mut facet.layout {
                crate::plot::FacetLayout::Wrap { variables } => {
                    variables.iter_mut().for_each(normalize_var);
                }
                crate::plot::FacetLayout::Grid { row, column } => {
                    row.iter_mut().for_each(normalize_var);
                    column.iter_mut().for_each(normalize_var);
                }
            }
        }
    }
}

// =============================================================================
// Global Mapping & Color Splitting
// =============================================================================

/// Check if an aesthetic value is a null sentinel (explicit removal marker)
fn is_null_sentinel(value: &AestheticValue) -> bool {
    matches!(
        value,
        AestheticValue::Literal(crate::plot::ParameterValue::Null)
    )
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
        let aesthetic_ctx = spec.get_aesthetic_context();
        for (layer, schema) in spec.layers.iter_mut().zip(layer_schemas.iter()) {
            // Skip annotation layers - they don't inherit global mappings
            if matches!(layer.source, Some(DataSource::Annotation)) {
                continue;
            }

            let supported = layer.geom.aesthetics().supported();
            let all_names = layer.geom.aesthetics().names();
            let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();

            // 1. First merge explicit global aesthetics (layer overrides global)
            // Note: facet aesthetics (panel, row, column) are also accepted,
            // as they apply to all layers regardless of geom support
            // Note: Use all_names (not supported) so that Delayed aesthetics like
            // pos2 on histogram can be targeted by explicit global mappings, matching
            // the behavior of layer-level MAPPING
            // Note: Also accept flipped position counterparts so bidirectional geoms
            // (e.g., range: pos1+pos2min+pos2max or pos2+pos1min+pos1max) can
            // receive globals from either orientation.
            for (aesthetic, value) in &spec.global_mappings.aesthetics {
                let is_facet_aesthetic = crate::plot::scale::is_facet_aesthetic(aesthetic.as_str());
                let flipped = aesthetic_ctx.flip_position(aesthetic);
                if all_names.contains(&aesthetic.as_str())
                    || all_names.contains(&flipped.as_str())
                    || is_facet_aesthetic
                {
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
                for aes in &supported {
                    // Convert internal name to user-facing name for schema matching
                    let user_name = aesthetic_ctx.map_internal_to_user(aes);
                    // Only create mapping if the user-facing column exists in the schema
                    if schema_columns.contains(user_name.as_str()) {
                        layer
                            .mappings
                            .aesthetics
                            .entry(crate::parser::builder::normalise_aes_name(aes))
                            .or_insert(AestheticValue::standard_column(&user_name));
                    }
                }
            }

            // Clear wildcard flag since it's been resolved
            layer.mappings.wildcard = false;

            // Remove null sentinel mappings (explicit "don't inherit" markers)
            layer
                .mappings
                .aesthetics
                .retain(|_, value| !is_null_sentinel(value));
        }
    }
}

/// Resolve aesthetic aliases in a plot specification.
///
/// For each alias defined in [`AESTHETIC_ALIASES`], splits the alias in scales,
/// layer mappings, and layer parameters into the concrete target aesthetics
/// (only where the geom supports them). Removes the alias after splitting to
/// avoid non-deterministic behavior from HashMap iteration order.
fn resolve_aesthetic_aliases(spec: &mut Plot) {
    use crate::plot::layer::geom::types::AESTHETIC_ALIASES;

    for &(alias, targets) in AESTHETIC_ALIASES {
        // 1. Split alias SCALE to target scales
        if let Some(idx) = spec.scales.iter().position(|s| s.aesthetic == alias) {
            let alias_scale = spec.scales[idx].clone();
            for &target in targets {
                if !spec.scales.iter().any(|s| s.aesthetic == target) {
                    let mut target_scale = alias_scale.clone();
                    target_scale.aesthetic = target.to_string();
                    spec.scales.push(target_scale);
                }
            }
            spec.scales.remove(idx);
        }

        // 2. Split alias mapping and parameters in each layer
        for layer in &mut spec.layers {
            let aesthetics = layer.geom.aesthetics();

            // Split mapping
            if let Some(value) = layer.mappings.aesthetics.get(alias).cloned() {
                for &target in targets {
                    if aesthetics.is_supported(target) {
                        layer
                            .mappings
                            .aesthetics
                            .entry(target.to_string())
                            .or_insert(value.clone());
                    }
                }
                layer.mappings.aesthetics.remove(alias);
            }

            // Split parameter (SETTING)
            if let Some(value) = layer.parameters.get(alias).cloned() {
                for &target in targets {
                    if aesthetics.is_supported(target) {
                        layer
                            .parameters
                            .entry(target.to_string())
                            .or_insert(value.clone());
                    }
                }
                layer.parameters.remove(alias);
            }
        }
    }
}

// =============================================================================
// Facet Mapping Injection
// =============================================================================

/// Add facet variable mappings to each layer's mappings.
///
/// This allows facet aesthetics to flow through the same code paths as
/// regular aesthetics (scale resolution, type casting, SELECT list building,
/// partition_by handling, etc.).
///
/// Skips injection if:
/// - The layer already has the facet aesthetic mapped (from MAPPING or global)
/// - The variables list is empty (inferred from layer mappings, not FACET clause)
/// - The column doesn't exist in this layer's schema (different data source)
fn add_facet_mappings_to_layers(
    layers: &mut [Layer],
    facet: &crate::plot::Facet,
    layer_type_info: &[Vec<schema::TypeInfo>],
) {
    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        if layer_idx >= layer_type_info.len() {
            continue;
        }
        let type_info = &layer_type_info[layer_idx];

        // Use internal aesthetic names (facet1, facet2) since transformation has already occurred
        for (var, aesthetic) in facet.layout.get_internal_aesthetic_mappings() {
            // Skip if layer already has this facet aesthetic mapped (from MAPPING or global)
            if layer.mappings.aesthetics.contains_key(&aesthetic) {
                continue;
            }

            // Only inject if the column exists in this layer's schema
            // (variables list is empty when inferred from layer mappings - no injection needed)
            if type_info.iter().any(|(col, _, _)| col == var) {
                // Add mapping: variable → facet aesthetic (internal name)
                layer.mappings.aesthetics.insert(
                    aesthetic,
                    AestheticValue::Column {
                        name: var.to_string(),
                        original_name: Some(var.to_string()),
                        is_dummy: false,
                    },
                );
            }
        }
    }
}

// =============================================================================
// Facet Missing Column Detection and Handling
// =============================================================================

/// Identify which layers are missing the facet column.
///
/// Returns a vector of booleans, one per layer. A layer is considered "missing"
/// the facet column if ANY of the facet variables are not present in the layer's
/// schema (type_info).
///
/// This is used to determine which layers need data duplication when
/// `missing => 'repeat'` is set on the facet.
fn identify_layers_missing_facet_column(
    layers: &[Layer],
    facet: &crate::plot::Facet,
    layer_type_info: &[Vec<schema::TypeInfo>],
) -> Vec<bool> {
    let facet_variables = facet.get_variables();

    // If variables list is empty (inferred from layer mappings), no layers are "missing"
    if facet_variables.is_empty() {
        return vec![false; layers.len()];
    }

    layers
        .iter()
        .enumerate()
        .map(|(layer_idx, _layer)| {
            if layer_idx >= layer_type_info.len() {
                return false;
            }
            let type_info = &layer_type_info[layer_idx];
            let schema_columns: std::collections::HashSet<&str> =
                type_info.iter().map(|(name, _, _)| name.as_str()).collect();

            // Layer is missing if ANY facet variable is absent from its schema
            facet_variables
                .iter()
                .any(|var| !schema_columns.contains(var.as_str()))
        })
        .collect()
}

/// Get unique facet values from layers that have the facet column.
///
/// Collects all unique values for a facet aesthetic from layers that have the column,
/// to be used for cross-joining with layers that are missing the column.
fn get_unique_facet_values(
    data_map: &HashMap<String, DataFrame>,
    facet_aesthetic: &str,
    layers: &[Layer],
    layers_missing_facet: &[bool],
) -> Option<arrow::array::ArrayRef> {
    let aes_col = naming::aesthetic_column(facet_aesthetic);
    let mut all_arrays: Vec<arrow::array::ArrayRef> = Vec::new();

    for (idx, layer) in layers.iter().enumerate() {
        // Skip layers that are missing the facet column
        if idx < layers_missing_facet.len() && layers_missing_facet[idx] {
            continue;
        }

        if let Some(ref data_key) = layer.data_key {
            if let Some(df) = data_map.get(data_key) {
                if let Ok(col) = df.column(&aes_col) {
                    all_arrays.push(col.clone());
                }
            }
        }
    }

    if all_arrays.is_empty() {
        return None;
    }

    // Concatenate all arrays
    let refs: Vec<&dyn arrow::array::Array> = all_arrays.iter().map(|a| a.as_ref()).collect();
    let combined = arrow::compute::concat(&refs).ok()?;

    // Get unique values by collecting strings into a set
    use crate::array_util::value_to_string;
    let mut seen = std::collections::HashSet::new();
    let mut unique_indices = Vec::new();
    for i in 0..combined.len() {
        let key = value_to_string(&combined, i);
        if seen.insert(key) {
            unique_indices.push(i as u32);
        }
    }
    let indices = arrow::array::UInt32Array::from(unique_indices);
    arrow::compute::take(&*combined, &indices, None).ok()
}

/// Cross-join a DataFrame with facet values (duplicate for each facet panel).
///
/// Creates a new DataFrame where every row is duplicated for each unique facet value.
/// The facet column is added with the appropriate values.
fn cross_join_with_facet_values(
    df: &DataFrame,
    unique_values: &arrow::array::ArrayRef,
    facet_aesthetic: &str,
) -> Result<DataFrame> {
    use arrow::array::{Array, UInt32Array};

    let aes_col = naming::aesthetic_column(facet_aesthetic);
    let n_values = unique_values.len();

    if n_values == 0 {
        return Ok(df.clone());
    }

    let n_rows = df.height();

    // 1. Repeat each original column n_values times
    // [a, b, c] with n_values=2 -> [a, a, b, b, c, c]
    let repeat_indices: Vec<u32> = (0..n_rows)
        .flat_map(|i| std::iter::repeat_n(i as u32, n_values))
        .collect();
    let repeat_idx = UInt32Array::from(repeat_indices);

    let col_names = df.get_column_names();
    let mut new_columns: Vec<(&str, arrow::array::ArrayRef)> = Vec::new();
    for name in &col_names {
        let col = df.column(name)?;
        let repeated = arrow::compute::take(col.as_ref(), &repeat_idx, None).map_err(|e| {
            GgsqlError::InternalError(format!("Failed to repeat column '{}': {}", name, e))
        })?;
        new_columns.push((name, repeated));
    }

    // 2. Create the facet column: tile unique_values for each row
    // [v1, v2, v1, v2, v1, v2] for n_rows=3, n_values=2
    let facet_indices: Vec<u32> = (0..n_rows)
        .flat_map(|_| (0..n_values).map(|j| j as u32))
        .collect();
    let facet_idx = UInt32Array::from(facet_indices);
    let facet_col = arrow::compute::take(unique_values.as_ref(), &facet_idx, None)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to create facet column: {}", e)))?;
    new_columns.push((&aes_col, facet_col));

    DataFrame::new(new_columns)
}

/// Handle layers missing the facet column based on facet.missing setting.
///
/// - `repeat` (default): Cross-join layer data with all unique facet values,
///   effectively duplicating the layer's data across all facet panels.
/// - `null`: Do nothing (current behavior - nulls added during unification,
///   layer appears only in null panel if null is in scale's input range).
fn handle_missing_facet_columns(
    spec: &Plot,
    data_map: &mut HashMap<String, DataFrame>,
    layers_missing_facet: &[bool],
) -> Result<()> {
    use crate::plot::ParameterValue;

    let facet = match &spec.facet {
        Some(f) => f,
        None => return Ok(()),
    };

    // Get the missing setting (default to "repeat")
    let missing_setting = facet
        .properties
        .get("missing")
        .and_then(|v| {
            if let ParameterValue::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        })
        .unwrap_or("repeat");

    // If null, do nothing (existing behavior handles this)
    if missing_setting == "null" {
        return Ok(());
    }

    // Get internal facet aesthetics from layout (facet1, facet2)
    let facet_aesthetics = facet.layout.internal_facet_names();

    // Process each facet aesthetic
    for facet_aesthetic in &facet_aesthetics {
        // Get unique values from layers that HAVE the column
        let unique_values = match get_unique_facet_values(
            data_map,
            facet_aesthetic,
            &spec.layers,
            layers_missing_facet,
        ) {
            Some(v) => v,
            None => continue, // No layers have this column, skip
        };

        // For each layer MISSING the column, cross-join with facet values
        for (idx, layer) in spec.layers.iter().enumerate() {
            if idx >= layers_missing_facet.len() || !layers_missing_facet[idx] {
                continue;
            }

            if let Some(ref data_key) = layer.data_key {
                if let Some(df) = data_map.get(data_key) {
                    // Only process if this DataFrame doesn't already have the column
                    let aes_col = naming::aesthetic_column(facet_aesthetic);
                    if df.column(&aes_col).is_err() {
                        let expanded_df =
                            cross_join_with_facet_values(df, &unique_values, facet_aesthetic)?;
                        data_map.insert(data_key.clone(), expanded_df);
                    }
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Facet Resolution from Layer Mappings
// =============================================================================

/// Resolve facet configuration from layer mappings and FACET clause.
///
/// Logic:
/// 1. Collect all facet aesthetic mappings from layers (after global merge)
/// 2. Validate no conflicting layout types (cannot mix 'panel' with 'row'/'column')
/// 3. Validate Grid layout has both 'row' and 'column' if either is used
/// 4. If FACET clause exists:
///    - Validate layer mappings are compatible with layout type
///    - Layer mappings take precedence (override FACET clause columns)
/// 5. If no FACET clause: infer layout from layer mappings
///
/// Returns:
/// - `Ok(Some(Facet))` - Resolved facet configuration
/// - `Ok(None)` - No faceting needed
/// - `Err(...)` - Validation error
fn resolve_facet(
    layers: &[crate::plot::Layer],
    existing_facet: Option<crate::plot::Facet>,
) -> Result<Option<crate::plot::Facet>> {
    use crate::plot::facet::FacetLayout;
    use crate::plot::scale::is_facet_aesthetic;

    // Collect facet aesthetic mappings from all layers
    // After transformation: panel → facet1, row → facet1, column → facet2
    // If only facet1 exists → wrap layout (panel only)
    // If facet1 AND facet2 exist → grid layout (row AND column)
    let mut has_facet1 = false;
    let mut has_facet2 = false;

    for layer in layers {
        for aesthetic in layer.mappings.aesthetics.keys() {
            if is_facet_aesthetic(aesthetic) {
                match aesthetic.as_str() {
                    "facet1" => has_facet1 = true,
                    "facet2" => has_facet2 = true,
                    _ => {}
                }
            }
        }
    }

    // Validate: Grid requires both facet1 and facet2 (row and column)
    // Having only facet2 is an error (column without row)
    if has_facet2 && !has_facet1 {
        return Err(GgsqlError::ValidationError(
            "Grid facet layout requires both 'row' and 'column' aesthetics. Missing: 'row'"
                .to_string(),
        ));
    }

    // Determine inferred layout from layer mappings
    // facet1 only → wrap layout (originally 'panel')
    // facet1 AND facet2 → grid layout (originally 'row' AND 'column')
    let inferred_layout = if has_facet1 && has_facet2 {
        Some(FacetLayout::Grid {
            row: vec![],    // Empty - each layer has its own mapping
            column: vec![], // Empty - each layer has its own mapping
        })
    } else if has_facet1 {
        Some(FacetLayout::Wrap {
            variables: vec![], // Empty - each layer has its own mapping
        })
    } else {
        None
    };

    // If no layer mappings and no FACET clause, no faceting
    if inferred_layout.is_none() && existing_facet.is_none() {
        return Ok(None);
    }

    // If FACET clause exists, validate compatibility with layer mappings
    if let Some(ref facet) = existing_facet {
        let is_wrap = facet.is_wrap();

        // Wrap layout (FACET var) but layer has both facet1 AND facet2 (row/column)
        // This indicates the layer was declared with Grid aesthetics
        if is_wrap && has_facet2 {
            return Err(GgsqlError::ValidationError(
                "FACET clause uses Wrap layout, but layer mappings use 'row'/'column' (Grid layout). \
                 Remove FACET clause to infer Grid layout, or use 'panel' aesthetic instead.".to_string()
            ));
        }

        // Grid layout (FACET row BY col) but layer has only facet1 without facet2
        // This indicates the layer was declared with Wrap aesthetic (panel only)
        // Note: Grid layout declared by user means they expect both row and column
        // If layer only has facet1, it's compatible (will use only row mapping)
        // This is actually okay - we don't need to error here

        // FACET clause exists and is compatible - use it (layer mappings will override columns)
        return Ok(Some(facet.clone()));
    }

    // No FACET clause - infer from layer mappings
    if let Some(layout) = inferred_layout {
        return Ok(Some(crate::plot::Facet::new(layout)));
    }

    Ok(None)
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
    aesthetic_ctx: &AestheticContext,
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

        // Build set of excluded aesthetics that should not trigger auto-grouping:
        // - Stat-consumed aesthetics (transformed, not grouped)
        // - 'label' aesthetic (text content to display, not grouping categories)
        let consumed_aesthetics = layer.geom.stat_consumed_aesthetics();
        let mut excluded_aesthetics: HashSet<&str> = consumed_aesthetics.iter().copied().collect();
        excluded_aesthetics.insert("label");

        for (aesthetic, value) in &layer.mappings.aesthetics {
            // Skip position aesthetics - these should not trigger auto-grouping.
            // Stats that need to group by position aesthetics (like bar/histogram)
            // already handle this themselves via stat_consumed_aesthetics().
            if is_position_aesthetic(aesthetic) {
                continue;
            }

            // Skip excluded aesthetics (stat-consumed or label)
            if excluded_aesthetics.contains(aesthetic.as_str()) {
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
                let primary_aes = aesthetic_ctx
                    .primary_internal_position(aesthetic)
                    .unwrap_or(aesthetic);
                let is_discrete = if let Some(scale) = scale_map.get(primary_aes) {
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

    // Facet aesthetic columns (shared across all layers)
    // Only the aesthetic-prefixed columns are needed for Vega-Lite output.
    // The original variable names (e.g., "species") are not needed after
    // the aesthetic columns (e.g., "__ggsql_aes_facet1__") have been created.
    if let Some(ref facet) = spec.facet {
        for aesthetic in facet.layout.internal_facet_names() {
            required.insert(naming::aesthetic_column(&aesthetic));
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

    // Position offset column for position adjustments that create pos1offset
    // This column is created by dodge/jitter positions and is not in layer.mappings
    if layer.position.creates_pos1offset() {
        required.insert(naming::aesthetic_column("pos1offset"));
    }

    // Position offset column for position adjustments that create pos2offset
    // This column is created by jitter position for vertical jittering
    if layer.position.creates_pos2offset() {
        required.insert(naming::aesthetic_column("pos2offset"));
    }

    required
}

/// Prune columns from a DataFrame to only include required columns.
///
/// Columns that don't exist in the DataFrame are silently ignored.
/// If no required columns exist in the DataFrame (e.g., annotation layers with only
/// literal aesthetics), returns a 0-column DataFrame with the same row count.
fn prune_dataframe(df: &DataFrame, required: &HashSet<String>) -> Result<DataFrame> {
    let columns_to_keep: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|name| required.contains(name.as_str()))
        .map(|name| name.to_string())
        .collect();

    if columns_to_keep.is_empty() {
        // Return a 0-column DataFrame with the same row count
        // This happens for annotation layers with only literal aesthetics (e.g., PLACE rule SETTING slope => 0.4)
        // The row count determines how many marks to draw; aesthetics come from Literal values in mappings
        let row_count = df.height();

        if row_count > 0 {
            // Create a 0-column DataFrame with the correct row count
            // We do this by creating a dummy column and then dropping it
            let with_rows = crate::df! {
                "__dummy__" => vec![0i32; row_count]
            }?;
            return with_rows.drop("__dummy__");
        } else {
            return Ok(DataFrame::empty());
        }
    }

    // Keep only the columns in columns_to_keep
    let drop_cols: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|name| !columns_to_keep.contains(name))
        .collect();
    df.drop_many(&drop_cols)
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
pub fn prepare_data_with_reader(query: &str, reader: &dyn Reader) -> Result<PreparedData> {
    let execute_query = |sql: &str| reader.execute_sql(sql);
    let dialect = reader.dialect();

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

    // Execute setup statements (INSTALL, LOAD, SET, etc.) before the main query
    for stmt in source_tree.find_texts(&root, "(sql_statement (other_sql_statement) @stmt)") {
        execute_query(&stmt)?;
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
    let mut has_global_table = false;
    if sql_part.is_some() {
        let (side_effects, query) = cte::transform_global_sql(&source_tree, &materialized_ctes);

        for stmt in &side_effects {
            execute_query(stmt)?;
        }

        if let Some(query) = query {
            // Materialize global result as a temp table directly on the backend
            // (no roundtrip through Rust).
            let statements = reader.dialect().create_or_replace_temp_table_sql(
                &naming::global_table(),
                &[],
                &query,
            );
            for stmt in &statements {
                execute_query(stmt)?;
            }

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
    // For annotation layers, this is where array recycling and parameter→mapping conversion happens
    let layer_source_queries: Vec<String> = specs[0]
        .layers
        .iter_mut()
        .map(|l| layer::layer_source_query(l, &materialized_ctes, has_global_table, dialect))
        .collect::<Result<Vec<_>>>()?;

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
    // NOTE: Both global and layer aesthetics are already in internal format (pos1, pos2)
    // because transformation happens in builder.rs right after parsing
    merge_global_mappings_into_layers(&mut specs, &layer_schemas);

    // Reconcile user-written column casing with schema casing (DuckDB lowercases
    // unquoted identifiers). Must run after the global→layer merge so each layer
    // is normalized against its own schema, which matters for multi-source layers.
    normalize_column_references(&mut specs, &layer_schemas);

    // Resolve aesthetic aliases (e.g., 'color' → 'fill'/'stroke') early in the pipeline
    // This must happen before validation so concrete aesthetics are validated
    for spec in &mut specs {
        resolve_aesthetic_aliases(spec);
    }

    // Add literal (constant) columns to type info programmatically
    // This avoids re-querying the database - we derive types from the AST
    schema::add_literal_columns_to_type_info(&specs[0].layers, &mut layer_type_info);

    // Rebuild layer schemas with constant columns included
    layer_schemas = layer_type_info
        .iter()
        .map(|ti| schema::type_info_to_schema(ti))
        .collect();

    // Resolve facet: infer from layer mappings or validate against FACET clause
    // This must happen AFTER merge_global_mappings_into_layers so layer mappings include global aesthetics
    specs[0].facet = resolve_facet(&specs[0].layers, specs[0].facet.clone())?;

    // Inject facet variable mappings into layers (only for missing aesthetics)
    // This allows facet aesthetics (facet1, facet2) to flow through the same
    // code paths as regular aesthetics - scale creation, type resolution, etc.
    if let Some(facet) = specs[0].facet.clone() {
        add_facet_mappings_to_layers(&mut specs[0].layers, &facet, &layer_type_info);
    }

    // Identify layers missing the facet column (for later data duplication)
    let layers_missing_facet = if let Some(ref facet) = specs[0].facet {
        identify_layers_missing_facet_column(&specs[0].layers, facet, &layer_type_info)
    } else {
        vec![false; specs[0].layers.len()]
    };

    // Validate all layers against their schemas
    // This must happen BEFORE build_layer_query because stat transforms remove consumed aesthetics
    validate(
        &specs[0].layers,
        &layer_schemas,
        &specs[0].aesthetic_context,
    )?;

    // Allow geoms to adjust mappings based on their specific logic
    // (e.g., rule geom converts pos1/pos2 to AnnotationColumn when slope is present)
    for spec in &mut specs {
        for layer in &mut spec.layers {
            layer
                .geom
                .setup_layer(&mut layer.mappings, &mut layer.parameters)?;
        }
    }

    // Create scales for all mapped aesthetics that don't have explicit SCALE clauses
    scale::create_missing_scales(&mut specs[0]);

    // Resolve scale types and transforms early based on column dtypes
    scale::resolve_scale_types_and_transforms(&mut specs[0], &layer_type_info)?;

    // Determine which columns need type casting
    let type_requirements =
        casting::determine_type_requirements(&specs[0], &layer_type_info, dialect);

    // Update type info with post-cast dtypes
    // This ensures subsequent schema extraction and scale resolution see the correct types
    for (layer_idx, requirements) in type_requirements.iter().enumerate() {
        if layer_idx < layer_type_info.len() {
            casting::update_type_info_for_casting(&mut layer_type_info[layer_idx], requirements);
        }
    }

    // Detect orientation and flip mappings BEFORE building base queries.
    // This ensures the SQL query uses the correct aesthetic column names.
    let scales = specs[0].scales.clone();
    let aesthetic_ctx = specs[0].get_aesthetic_context();
    layer::resolve_orientations(
        &mut specs[0].layers,
        &scales,
        &mut layer_type_info,
        &aesthetic_ctx,
    );

    // Build layer base queries using build_layer_base_query()
    // These include: SELECT with aesthetic renames, casts from type_requirements, filters
    // Note: This is Part 1 of the split - base queries that can be used for schema completion
    let layer_base_queries: Vec<String> = specs[0]
        .layers
        .iter()
        .enumerate()
        .map(|(idx, l)| {
            layer::build_layer_base_query(
                l,
                &layer_source_queries[idx],
                &type_requirements[idx],
                dialect,
            )
        })
        .collect();

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
    let aesthetic_ctx = specs[0].get_aesthetic_context();
    add_discrete_columns_to_partition_by(
        &mut specs[0].layers,
        &layer_schemas,
        &scales,
        &aesthetic_ctx,
    );

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
            &scales,
            dialect,
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
    // Key by (query, serialized_remappings, orientation) to detect when layers can share data
    // Layers with identical query AND remappings AND orientation share data via data_key
    let mut config_to_key: HashMap<(String, String, bool), String> = HashMap::new();

    for (idx, q) in layer_queries.iter().enumerate() {
        let layer = &mut specs[0].layers[idx];
        let remappings_key = serde_json::to_string(&layer.remappings).unwrap_or_default();
        let needs_flip = is_transposed(layer);
        let config_key = (q.clone(), remappings_key, needs_flip);

        if let Some(existing_key) = config_to_key.get(&config_key) {
            // Same query AND same remappings AND same orientation - share data
            layer.data_key = Some(existing_key.clone());
        } else {
            // Need own data entry (either first occurrence or different config)
            let layer_key = naming::layer_key(idx);
            let df = query_to_result.get(q).unwrap().clone();

            data_map.insert(layer_key.clone(), df);
            config_to_key.insert(config_key, layer_key.clone());
            layer.data_key = Some(layer_key);
        }
    }

    // Phase 4: Apply remappings (rename stat columns and add literal columns)
    // e.g., __ggsql_stat_count → __ggsql_aes_y__, or add __ggsql_aes_pos2end__ = 0.0
    // Note: Prefixed aesthetic names persist through the entire pipeline
    // Track processed keys to avoid duplicate work on shared datasets
    let mut processed_keys: HashSet<String> = HashSet::new();
    for l in specs[0].layers.iter_mut() {
        if let Some(ref key) = l.data_key {
            if processed_keys.insert(key.clone()) {
                // First time seeing this data - process it
                if let Some(df) = data_map.remove(key) {
                    let df_with_remappings = layer::apply_remappings_post_query(df, l)?;
                    // Apply geom post_process (e.g., violin scales offset to [0, 0.5 * width])
                    let df_post_processed =
                        l.geom.post_process(df_with_remappings, &l.parameters)?;
                    data_map.insert(key.clone(), df_post_processed);
                }
            }

            // Flip remappings for Transposed orientation layers.
            // This must happen AFTER apply_remappings_post_query (which creates columns
            // with ALIGNED orientation names) but BEFORE update_mappings_for_remappings
            // (which uses remapping keys to create mapping entries).
            // Phase 4.5 will then flip the DataFrame columns to match.
            if is_transposed(l) {
                crate::plot::layer::orientation::flip_position_aesthetics(
                    &mut l.remappings.aesthetics,
                );
            }

            // Update layer mappings for all layers (even if data shared)
            l.update_mappings_for_remappings();
        }

        // Resolve aesthetics (SETTING/defaults) after all mapping updates
        // This ensures query literals have been converted to columns, and SETTING/defaults
        // are added as new Literal entries that remain as constant values
        l.resolve_aesthetics();
    }

    // Phase 4.5: Flip DataFrame columns for Transposed orientation layers
    // This must happen AFTER remappings (Phase 4) because remappings create columns
    // with ALIGNED orientation names, and the flip converts them to USER orientation.
    // All position columns (stat-produced and literal) are flipped together.
    let mut flipped_keys: HashSet<String> = HashSet::new();
    for layer in specs[0].layers.iter() {
        if is_transposed(layer) {
            if let Some(ref key) = layer.data_key {
                if flipped_keys.insert(key.clone()) {
                    // First time flipping this data key
                    if let Some(df) = data_map.remove(key) {
                        let flipped_df =
                            crate::plot::layer::orientation::flip_dataframe_position_columns(
                                df,
                                &aesthetic_ctx,
                            );
                        data_map.insert(key.clone(), flipped_df);
                    }
                }
            }
        }
    }

    // Create scales for aesthetics added by stat transforms (e.g., y from histogram)
    // This must happen after build_layer_query() which applies stat transforms
    // and modifies layer.mappings with new aesthetics like y → __ggsql_stat_count__
    for spec in &mut specs {
        scale::create_missing_scales_post_stat(spec, &data_map)?;
    }

    // Apply position adjustments (stack, dodge) to layer data
    // Must be after scale type inference but before full scale resolution
    // so scales can see the adjusted values (e.g., stacked maxima)
    for spec in &mut specs {
        position::apply_position_adjustments(spec, &mut data_map)?;
    }

    // Validate we have some data (every layer should have its own data)
    if data_map.is_empty() {
        return Err(GgsqlError::ValidationError(
            "No data sources found. Either provide a SQL query or use MAPPING FROM in layers."
                .to_string(),
        ));
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

    // Resolve facet properties (after data is available)
    for spec in &mut specs {
        // Get position aesthetic names from the aesthetic context (coord-specific)
        // This must be done before mutably borrowing facet
        let position_names: Vec<String> = spec.get_aesthetic_context().user_position().to_vec();
        // Convert to &str slice for resolve_facet_properties
        let position_refs: Vec<&str> = position_names.iter().map(|s| s.as_str()).collect();

        if let Some(ref mut facet) = spec.facet {
            // Get the first layer's data for computing facet defaults
            let facet_df = data_map.get(&naming::layer_key(0)).ok_or_else(|| {
                GgsqlError::InternalError("Missing layer 0 data for facet resolution".to_string())
            })?;
            // Use aesthetic column names (e.g., __ggsql_aes_facet1__) since the DataFrame
            // has been transformed to use aesthetic columns at this point
            let aesthetic_cols: Vec<String> = facet
                .layout
                .internal_facet_names()
                .iter()
                .map(|aes| naming::aesthetic_column(aes))
                .collect();
            let context = FacetDataContext::from_dataframe(facet_df, &aesthetic_cols);
            resolve_facet_properties(facet, &context, &position_refs)
                .map_err(|e| GgsqlError::ValidationError(format!("Facet: {}", e)))?;
        }
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

    // Handle layers missing the facet column based on facet.missing setting
    // This must happen after OOB handling but before pruning
    for spec in &specs {
        handle_missing_facet_columns(spec, &mut data_map, &layers_missing_facet)?;
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
    fn test_layer_references_cte_with_column_aliases() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            WITH t(value, label) AS (
                SELECT * FROM (VALUES
                    (70, 'Target'),
                    (80, 'Warning'),
                    (90, 'Critical')
                )
            )
            SELECT 1 AS date, 75 AS temperature
            VISUALISE
            DRAW line MAPPING date AS x, temperature AS y
            DRAW rule MAPPING value AS y, label AS colour FROM t
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer 0: line from global data
        let layer0_df = result.data.get(&naming::layer_key(0)).unwrap();
        assert_eq!(layer0_df.height(), 1);

        // Layer 1: rule from CTE with column aliases
        let layer1_df = result.data.get(&naming::layer_key(1)).unwrap();
        assert_eq!(layer1_df.height(), 3);
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

        // Should have prefixed aesthetic-named columns (using internal names)
        let col_names: Vec<String> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("pos1");
        let y_col = naming::aesthetic_column("pos2");
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

        // With new approach, columns are renamed to prefixed aesthetic names (using internal names)
        let col_names: Vec<String> = layer_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x_col = naming::aesthetic_column("pos1");
        let y_col = naming::aesthetic_column("pos2");
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

        // Layer should have pos2end in mappings (yend is transformed to pos2end)
        assert!(
            layer.mappings.aesthetics.contains_key("pos2end"),
            "Bar should have pos2end mapping for baseline: {:?}",
            layer.mappings.aesthetics.keys().collect::<Vec<_>>()
        );

        // The DataFrame should have the pos2end column with 0 values
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let yend_col = naming::aesthetic_column("pos2end");
        assert!(
            layer_df.column(&yend_col).is_ok(),
            "DataFrame should have '{}' column: {:?}",
            yend_col,
            layer_df.get_column_names()
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

        // Find the pos1 scale (x is transformed to pos1)
        let x_scale = spec.find_scale("pos1").expect("pos1 scale should exist");

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

        // Find the pos1 scale (x is transformed to pos1)
        let x_scale = spec.find_scale("pos1").expect("pos1 scale should exist");

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

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_visualise_from_after_create() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            CREATE TEMP TABLE data(x, y) AS (VALUES
              ('A', 5),
              ('B', 2),
              ('C', 4),
              ('D', 7),
              ('E', 6)
            )
            VISUALISE x, y FROM data
            DRAW area
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();
        let key = result.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer should have data_key");
        assert_eq!(result.data.get(key).unwrap().height(), 5);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_visualise_from_after_create_and_insert() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            CREATE TEMP TABLE data(x INTEGER, y INTEGER);
            INSERT INTO data VALUES (1, 10), (2, 20), (3, 30);
            VISUALISE x, y FROM data
            DRAW point
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();
        let key = result.specs[0].layers[0]
            .data_key
            .as_ref()
            .expect("Layer should have data_key");
        assert_eq!(result.data.get(key).unwrap().height(), 3);
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
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Use internal aesthetic names
        let x_col = naming::aesthetic_column("pos1");
        let y_col = naming::aesthetic_column("pos2");
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

    // =========================================================================
    // Facet Aesthetic Mapping Tests
    // =========================================================================

    mod resolve_facet_tests {
        use super::*;
        use crate::plot::facet::FacetLayout;
        use crate::plot::layer::geom::Geom;
        use crate::plot::layer::Layer;
        use crate::plot::Facet;

        fn make_layer_with_mapping(aesthetic: &str, column: &str) -> Layer {
            let mut layer = Layer::new(Geom::point());
            layer.mappings.aesthetics.insert(
                aesthetic.to_string(),
                AestheticValue::standard_column(column),
            );
            layer
        }

        #[test]
        fn test_resolve_facet_infers_wrap_from_layer_mapping() {
            // Use internal name "facet1" since resolve_facet is called after transformation
            let layers = vec![make_layer_with_mapping("facet1", "region")];

            let result = resolve_facet(&layers, None).unwrap();

            assert!(result.is_some());
            let facet = result.unwrap();
            assert!(facet.is_wrap());
            // Variables should be empty (each layer has its own mapping)
            assert!(facet.get_variables().is_empty());
        }

        #[test]
        fn test_resolve_facet_infers_grid_from_layer_mappings() {
            // Use internal names "facet1" and "facet2" since resolve_facet is called after transformation
            let mut layer = Layer::new(Geom::point());
            layer.mappings.aesthetics.insert(
                "facet1".to_string(),
                AestheticValue::standard_column("region"),
            );
            layer.mappings.aesthetics.insert(
                "facet2".to_string(),
                AestheticValue::standard_column("year"),
            );
            let layers = vec![layer];

            let result = resolve_facet(&layers, None).unwrap();

            assert!(result.is_some());
            let facet = result.unwrap();
            assert!(facet.is_grid());
            // Variables should be empty
            assert!(facet.get_variables().is_empty());
        }

        #[test]
        fn test_resolve_facet_error_incomplete_grid() {
            // Only facet2 without facet1 is an error (column without row)
            let layers = vec![make_layer_with_mapping("facet2", "region")];

            let result = resolve_facet(&layers, None);

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("requires both"));
            assert!(err.contains("row"));
        }

        #[test]
        fn test_resolve_facet_uses_existing_facet_clause() {
            let layers = vec![Layer::new(Geom::point())]; // No facet mappings

            let existing_facet = Facet::new(FacetLayout::Wrap {
                variables: vec!["region".to_string()],
            });

            let result = resolve_facet(&layers, Some(existing_facet.clone())).unwrap();

            assert!(result.is_some());
            let facet = result.unwrap();
            assert!(facet.is_wrap());
            assert_eq!(facet.get_variables(), vec!["region".to_string()]);
        }

        #[test]
        fn test_resolve_facet_error_wrap_clause_with_grid_mapping() {
            // When layer has both facet1 AND facet2 (grid), but FACET clause is Wrap
            // This should error because the user declared grid aesthetics but FACET says wrap
            let mut layer = Layer::new(Geom::point());
            layer.mappings.aesthetics.insert(
                "facet1".to_string(),
                AestheticValue::standard_column("category"),
            );
            layer.mappings.aesthetics.insert(
                "facet2".to_string(),
                AestheticValue::standard_column("year"),
            );
            let layers = vec![layer];

            let existing_facet = Facet::new(FacetLayout::Wrap {
                variables: vec!["region".to_string()],
            });

            let result = resolve_facet(&layers, Some(existing_facet));

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("Wrap layout"));
            assert!(err.contains("row")); // mentions the user-facing name in error
        }

        #[test]
        fn test_resolve_facet_no_mappings_no_clause() {
            let layers = vec![Layer::new(Geom::point())];

            let result = resolve_facet(&layers, None).unwrap();

            assert!(result.is_none());
        }

        #[test]
        fn test_resolve_facet_layer_override_compatible_with_clause() {
            // Layer has facet1 mapping (from panel), FACET clause is Wrap - compatible
            let layers = vec![make_layer_with_mapping("facet1", "category")];

            let existing_facet = Facet::new(FacetLayout::Wrap {
                variables: vec!["region".to_string()],
            });

            // Should succeed - layer mapping takes precedence over FACET clause columns
            let result = resolve_facet(&layers, Some(existing_facet)).unwrap();
            assert!(result.is_some());
            assert!(result.unwrap().is_wrap());
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_aesthetic_mapping_wrap() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE facet_test AS SELECT * FROM (VALUES
                    (1, 10, 'A'), (2, 20, 'A'), (3, 30, 'B'), (4, 40, 'B')
                ) AS t(x, y, region)",
                duckdb::params![],
            )
            .unwrap();

        // Use panel aesthetic in layer mapping (not FACET clause)
        let query = r#"
            SELECT * FROM facet_test
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, region AS panel
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have a facet configuration inferred from layer mapping
        assert!(result.specs[0].facet.is_some());
        let facet = result.specs[0].facet.as_ref().unwrap();
        assert!(facet.is_wrap());

        // Data should have facet1 aesthetic column (internal name for panel)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let facet_col = naming::aesthetic_column("facet1");
        assert!(
            layer_df.column(&facet_col).is_ok(),
            "Should have '{}' column: {:?}",
            facet_col,
            layer_df.get_column_names()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_aesthetic_mapping_grid() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE grid_facet_test AS SELECT * FROM (VALUES
                    (1, 10, 'A', 2020), (2, 20, 'B', 2020),
                    (3, 30, 'A', 2021), (4, 40, 'B', 2021)
                ) AS t(x, y, region, year)",
                duckdb::params![],
            )
            .unwrap();

        // Use row/column aesthetics in layer mapping
        let query = r#"
            SELECT * FROM grid_facet_test
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, region AS row, year AS column
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have a grid facet configuration
        assert!(result.specs[0].facet.is_some());
        let facet = result.specs[0].facet.as_ref().unwrap();
        assert!(facet.is_grid());

        // Data should have facet1 (row) and facet2 (column) aesthetic columns (internal names)
        let layer_df = result.data.get(&naming::layer_key(0)).unwrap();
        let row_col = naming::aesthetic_column("facet1");
        let col_col = naming::aesthetic_column("facet2");
        assert!(
            layer_df.column(&row_col).is_ok(),
            "Should have '{}' column",
            row_col
        );
        assert!(
            layer_df.column(&col_col).is_ok(),
            "Should have '{}' column",
            col_col
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_global_mapping() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE global_facet_test AS SELECT * FROM (VALUES
                    (1, 10, 'A'), (2, 20, 'B')
                ) AS t(x, y, region)",
                duckdb::params![],
            )
            .unwrap();

        // Use panel aesthetic in global VISUALISE mapping
        let query = r#"
            SELECT * FROM global_facet_test
            VISUALISE region AS panel
            DRAW point MAPPING x AS x, y AS y
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should have a facet configuration
        assert!(result.specs[0].facet.is_some());
        assert!(result.specs[0].facet.as_ref().unwrap().is_wrap());
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_layer_override_of_facet_clause() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE override_test AS SELECT * FROM (VALUES
                    (1, 10, 'A', 'X'), (2, 20, 'B', 'Y')
                ) AS t(x, y, region, category)",
                duckdb::params![],
            )
            .unwrap();

        // FACET clause specifies region, but layer mapping uses category
        let query = r#"
            SELECT * FROM override_test
            VISUALISE
            FACET region
            DRAW point MAPPING x AS x, y AS y, category AS panel
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Should succeed - layer mapping overrides FACET clause
        let layer = &result.specs[0].layers[0];
        // Use internal name "facet1" since transformation has occurred
        let facet_mapping = layer.mappings.aesthetics.get("facet1").unwrap();
        // Use label_name() which returns original column name before internal renaming
        assert_eq!(
            facet_mapping.label_name(),
            Some("category"),
            "Layer should override FACET clause with category column"
        );
    }

    // =========================================================================
    // Facet Missing Column Tests
    // =========================================================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_missing_repeat_broadcasts_layer() {
        // Test that missing => 'repeat' (default) broadcasts a layer without the facet column
        // across all facet panels
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create main data with facet column
        reader
            .connection()
            .execute(
                "CREATE TABLE main_data AS SELECT * FROM (VALUES
                    (1, 10, 'A'), (2, 20, 'A'), (3, 30, 'B'), (4, 40, 'B')
                ) AS t(x, y, region)",
                duckdb::params![],
            )
            .unwrap();

        // Create reference line data WITHOUT the facet column
        reader
            .connection()
            .execute(
                "CREATE TABLE ref_data AS SELECT * FROM (VALUES
                    (0, 25)
                ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Query with two layers: main data has facet, ref line doesn't
        // Default missing => 'repeat' should broadcast ref line to both panels
        let query = r#"
            SELECT * FROM main_data
            VISUALISE
            FACET region
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM ref_data
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer 1 (ref_data point) should have its data expanded to include both facet values
        let ref_key = result.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("ref layer should have data_key");
        let ref_df = result.data.get(ref_key).unwrap();

        // With repeat, the ref_data should have 2 rows (one per facet value: A and B)
        assert_eq!(
            ref_df.height(),
            2,
            "ref layer should be repeated for each facet panel (A and B)"
        );

        // The facet column should exist in the ref_data (internal name facet1)
        let facet_col = naming::aesthetic_column("facet1");
        assert!(
            ref_df.column(&facet_col).is_ok(),
            "ref data should have facet column after broadcast: {:?}",
            ref_df.get_column_names()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_missing_null_no_broadcast() {
        // Test that missing => 'null' does NOT broadcast layers
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create main data with facet column
        reader
            .connection()
            .execute(
                "CREATE TABLE main_data_null AS SELECT * FROM (VALUES
                    (1, 10, 'A'), (2, 20, 'A'), (3, 30, 'B'), (4, 40, 'B')
                ) AS t(x, y, region)",
                duckdb::params![],
            )
            .unwrap();

        // Create reference line data WITHOUT the facet column
        reader
            .connection()
            .execute(
                "CREATE TABLE ref_data_null AS SELECT * FROM (VALUES
                    (0, 25)
                ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Query with missing => 'null'
        let query = r#"
            SELECT * FROM main_data_null
            VISUALISE
            FACET region SETTING missing => 'null'
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM ref_data_null
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer 1 should NOT have its data expanded
        let ref_key = result.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("ref layer should have data_key");
        let ref_df = result.data.get(ref_key).unwrap();

        // With null, the ref data should have 1 row (not repeated)
        assert_eq!(
            ref_df.height(),
            1,
            "ref layer should NOT be repeated with missing => 'null'"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_missing_repeat_grid_layout() {
        // Test repeat behavior with grid facets (row + column)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create main data with row and column facet variables
        reader
            .connection()
            .execute(
                "CREATE TABLE grid_main AS SELECT * FROM (VALUES
                    (1, 10, 'A', 2020), (2, 20, 'A', 2021),
                    (3, 30, 'B', 2020), (4, 40, 'B', 2021)
                ) AS t(x, y, region, year)",
                duckdb::params![],
            )
            .unwrap();

        // Create reference data WITHOUT facet columns
        reader
            .connection()
            .execute(
                "CREATE TABLE grid_ref AS SELECT * FROM (VALUES
                    (0, 25)
                ) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        // Grid facet with default repeat
        let query = r#"
            SELECT * FROM grid_main
            VISUALISE
            FACET region BY year
            DRAW point MAPPING x AS x, y AS y
            DRAW point MAPPING x AS x, y AS y FROM grid_ref
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Layer 1 should be expanded for both row and column
        let ref_key = result.specs[0].layers[1]
            .data_key
            .as_ref()
            .expect("ref layer should have data_key");
        let ref_df = result.data.get(ref_key).unwrap();

        // With grid (2 regions x 2 years = 4 panels), the ref should have 4 rows
        assert_eq!(
            ref_df.height(),
            4,
            "ref layer should be repeated for each grid panel (2 regions x 2 years)"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_facet_missing_layer_with_facet_column_unchanged() {
        // Ensure layers that DO have the facet column are not affected
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data where both layers have the facet column
        reader
            .connection()
            .execute(
                "CREATE TABLE both_have_facet AS SELECT * FROM (VALUES
                    (1, 10, 'A'), (2, 20, 'B')
                ) AS t(x, y, region)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM both_have_facet
            VISUALISE
            FACET region
            DRAW point MAPPING x AS x, y AS y
            DRAW line MAPPING x AS x, y AS y
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Both layers should have 2 rows (original data, not expanded)
        let point_key = result.specs[0].layers[0].data_key.as_ref().unwrap();
        let line_key = result.specs[0].layers[1].data_key.as_ref().unwrap();

        let point_df = result.data.get(point_key).unwrap();
        let line_df = result.data.get(line_key).unwrap();

        assert_eq!(
            point_df.height(),
            2,
            "point layer with facet column should not be expanded"
        );
        assert_eq!(
            line_df.height(),
            2,
            "line layer with facet column should not be expanded"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_annotation_layer() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create test data
        reader
            .connection()
            .execute(
                "CREATE TABLE test_place AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y)",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT * FROM test_place
            VISUALISE x, y
            DRAW point
            PLACE text SETTING x => 2, y => 25, label => 'Annotation', fontsize => 14
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        assert_eq!(result.specs.len(), 1);
        assert_eq!(
            result.specs[0].layers.len(),
            2,
            "Should have DRAW + PLACE layers"
        );

        // First layer: regular DRAW point
        let point_layer = &result.specs[0].layers[0];
        assert_eq!(point_layer.geom, crate::Geom::point());
        assert!(
            point_layer.source.is_none(),
            "DRAW layer should have no explicit source"
        );

        // Second layer: PLACE text annotation
        let annotation_layer = &result.specs[0].layers[1];
        assert_eq!(annotation_layer.geom, crate::Geom::text());
        assert!(
            matches!(annotation_layer.source, Some(DataSource::Annotation)),
            "PLACE layer should have Annotation source"
        );

        // Verify annotation layer has 1-row data
        let annotation_key = annotation_layer.data_key.as_ref().unwrap();
        let annotation_df = result.data.get(annotation_key).unwrap();
        assert_eq!(
            annotation_df.height(),
            1,
            "Annotation layer should have exactly 1 row"
        );

        // Verify position aesthetics are moved from SETTING to mappings with transformed names
        // They become Column references (not Literals) so they can participate in scale training
        assert!(
            matches!(
                annotation_layer.mappings.get("pos1"),
                Some(AestheticValue::Column { name, .. }) if name == "__ggsql_aes_pos1__"
            ),
            "x should be transformed to pos1, moved to mappings, and materialized as column"
        );
        assert!(
            matches!(
                annotation_layer.mappings.get("pos2"),
                Some(AestheticValue::Column { name, .. }) if name == "__ggsql_aes_pos2__"
            ),
            "y should be transformed to pos2, moved to mappings, and materialized as column"
        );

        // Verify required material aesthetic (label) is in mappings as AnnotationColumn
        // After process_annotation_layer, required aesthetics are converted to AnnotationColumn
        assert!(
            matches!(
                annotation_layer.mappings.get("label"),
                Some(AestheticValue::AnnotationColumn { name }) if name == "__ggsql_aes_label__"
            ),
            "label (required) should be in mappings as AnnotationColumn with prefixed name"
        );

        // Non-required, material, non-array aesthetics like size may be processed
        // by resolve_aesthetics or other downstream logic, so we don't strictly check
        // where they end up. The key point is that required/position aesthetics are
        // correctly moved to mappings.
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_annotation_with_stat_geom() {
        // Test that annotation layers work with stat geoms (e.g., histogram)
        // This was previously broken due to naming conflicts:
        // - Annotation layers created __ggsql_aes_pos1__ directly
        // - Stat transforms tried to rename __ggsql_stat_bin → __ggsql_aes_pos1__ (conflict!)
        // Now fixed: annotations use raw names (pos1), go through normal renaming pipeline

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            VISUALISE
            PLACE histogram SETTING x => [1.2, 2.5, 3.1, 2.8, 1.9, 2.2, 3.5, 2.1, 1.8, 2.9], bins => 5
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        assert_eq!(result.specs.len(), 1);
        assert_eq!(
            result.specs[0].layers.len(),
            1,
            "Should have one PLACE layer"
        );

        let histogram_layer = &result.specs[0].layers[0];
        assert_eq!(histogram_layer.geom, crate::Geom::histogram());
        assert!(
            matches!(histogram_layer.source, Some(DataSource::Annotation)),
            "PLACE layer should have Annotation source"
        );

        // After stat transform, pos1 should be remapped to bin
        assert!(
            histogram_layer.mappings.contains_key("pos1"),
            "Histogram should have pos1 aesthetic (bin start)"
        );
        assert!(
            histogram_layer.mappings.contains_key("pos1end"),
            "Histogram should have pos1end aesthetic (bin end)"
        );
        assert!(
            histogram_layer.mappings.contains_key("pos2"),
            "Histogram should have pos2 aesthetic (count)"
        );

        // Verify the data has binned results
        let histogram_key = histogram_layer.data_key.as_ref().unwrap();
        let histogram_df = result.data.get(histogram_key).unwrap();

        assert!(
            histogram_df.height() > 0,
            "Histogram should produce binned data"
        );
        assert!(
            histogram_df.height() <= 5,
            "Histogram with 5 bins should produce at most 5 rows"
        );

        // Verify the binned data has the expected columns
        assert!(
            histogram_df.column("__ggsql_aes_pos1__").is_ok(),
            "Should have bin start column"
        );
        assert!(
            histogram_df.column("__ggsql_aes_pos1end__").is_ok(),
            "Should have bin end column"
        );
        assert!(
            histogram_df.column("__ggsql_aes_pos2__").is_ok(),
            "Should have count column"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_missing_required_aesthetic() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT 1 AS x, 2 AS y
            VISUALISE x, y
            DRAW point
            PLACE text SETTING x => 5, label => 'Missing y!'
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(result.is_err(), "Should fail validation");

        match result {
            Err(GgsqlError::ValidationError(msg)) => {
                assert!(
                    msg.contains("y"),
                    "Error should mention missing y aesthetic: {}",
                    msg
                );
            }
            Err(e) => panic!("Expected ValidationError, got: {}", e),
            Ok(_) => panic!("Expected error, got success"),
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_affects_scale_ranges() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Data has x from 1-3, but PLACE annotation at x=10 should extend the range
        let query = r#"
            SELECT 1 AS x, 10 AS y UNION ALL
            SELECT 2 AS x, 20 AS y UNION ALL
            SELECT 3 AS x, 30 AS y
            VISUALISE x, y
            DRAW point
            PLACE text SETTING x => 10, y => 50, label => 'Extended'
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(result.is_ok(), "Query should execute: {:?}", result.err());

        let prep = result.unwrap();

        // Check that x scale input_range includes both data range (1-3) and annotation point (10)
        let x_scale = prep.specs[0].find_scale("pos1");
        assert!(x_scale.is_some(), "Should have x scale");

        let scale = x_scale.unwrap();
        if let Some(input_range) = &scale.input_range {
            // Input range should include the annotation x value (10)
            // The scale input_range is resolved from the combined data + annotation DataFrames
            assert!(
                input_range.len() >= 2,
                "Scale input_range should have min/max values"
            );
            // Check that the range max is at least 10 (the annotation x value)
            if let Some(max_val) = input_range.last() {
                match max_val {
                    crate::plot::types::ArrayElement::Number(n) => {
                        assert!(
                            *n >= 10.0,
                            "Scale input_range max should include annotation point at x=10, got: {}",
                            n
                        );
                    }
                    _ => panic!("Expected numeric input_range value"),
                }
            }
        } else {
            panic!("Scale should have an input_range");
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_no_global_mapping_inheritance() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT 1 AS x, 2 AS y, 'red' AS color
            VISUALISE x, y, color
            DRAW point
            PLACE text SETTING x => 5, y => 10, label => 'Test'
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // DRAW layer should have inherited global color mapping
        let point_layer = &result.specs[0].layers[0];
        assert!(
            point_layer.mappings.contains_key("color")
                || point_layer.mappings.contains_key("fill")
                || point_layer.mappings.contains_key("stroke"),
            "DRAW layer should inherit color from global mappings"
        );

        // PLACE layer should NOT have inherited global mappings
        let annotation_layer = &result.specs[0].layers[1];
        assert!(
            !annotation_layer.mappings.contains_key("color"),
            "PLACE layer should not inherit color from global mappings"
        );

        // PLACE layer should have geom default fill='black', not global color='red'
        assert!(
            annotation_layer.mappings.contains_key("fill"),
            "PLACE layer should have default fill from text geom"
        );
        match annotation_layer.mappings.aesthetics.get("fill") {
            Some(AestheticValue::Literal(crate::plot::types::ParameterValue::String(s)))
                if s == "black" =>
            {
                // Correct: geom default fill
            }
            Some(AestheticValue::Column { name, .. }) if name == "color" => {
                panic!("PLACE layer incorrectly inherited global color mapping as fill");
            }
            other => {
                panic!("Expected fill=Literal('black'), got: {:?}", other);
            }
        }

        assert!(
            !annotation_layer.mappings.contains_key("stroke"),
            "PLACE layer should not have stroke (text geom default is null)"
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_place_array_parameter_not_recycled() {
        // Test that array parameters that are NOT supported aesthetics
        // should not trigger row recycling in PLACE layers.
        // Example: offset is a PARAMETER for text geom, not an aesthetic,
        // so `offset => [0, 1]` should NOT create 2 rows.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            VISUALISE
            PLACE text SETTING x => 5, y => 10, label => 'Test', offset => [0, 1]
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        assert_eq!(result.specs.len(), 1);
        assert_eq!(
            result.specs[0].layers.len(),
            1,
            "Should have one PLACE layer"
        );

        let text_layer = &result.specs[0].layers[0];
        assert_eq!(text_layer.geom, crate::Geom::text());
        assert!(
            matches!(text_layer.source, Some(DataSource::Annotation)),
            "PLACE layer should have Annotation source"
        );

        // Verify annotation layer has exactly 1 row (not 2)
        // offset is a parameter, not an aesthetic, so it should NOT be recycled
        let annotation_key = text_layer.data_key.as_ref().unwrap();
        let annotation_df = result.data.get(annotation_key).unwrap();
        assert_eq!(
            annotation_df.height(),
            1,
            "Annotation layer should have exactly 1 row (offset array should not be recycled)"
        );

        // Verify offset remains as a parameter (not moved to aesthetics)
        assert!(
            text_layer.parameters.contains_key("offset"),
            "offset should remain as a parameter"
        );
        assert!(
            !text_layer.mappings.contains_key("offset"),
            "offset should NOT be moved to aesthetics/mappings"
        );

        // Verify offset has the original array value
        match text_layer.parameters.get("offset") {
            Some(crate::plot::types::ParameterValue::Array(arr)) => {
                assert_eq!(arr.len(), 2, "offset should have 2 elements");
                assert!(
                    matches!(arr[0], crate::plot::types::ArrayElement::Number(n) if (n - 0.0).abs() < 1e-10),
                    "offset[0] should be 0"
                );
                assert!(
                    matches!(arr[1], crate::plot::types::ArrayElement::Number(n) if (n - 1.0).abs() < 1e-10),
                    "offset[1] should be 1"
                );
            }
            other => panic!("Expected offset to be Array, got: {:?}", other),
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_null_mapping_removes_global_aesthetic() {
        // Global mapping sets fill=region, but second layer uses null AS fill to opt out
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 1 as x, 2 as y, 'A' as region
            VISUALISE x, y, region AS fill
            DRAW point
            DRAW line MAPPING null AS fill
        "#;

        let result = prepare_data_with_reader(query, &reader).unwrap();

        // Point layer (first) should have fill inherited from global
        let point_layer = &result.specs[0].layers[0];
        assert!(
            point_layer.mappings.aesthetics.contains_key("fill"),
            "point layer should inherit fill from global mapping"
        );

        // Line layer (second) should NOT have fill because of null AS fill
        let line_layer = &result.specs[0].layers[1];
        assert!(
            !line_layer.mappings.aesthetics.contains_key("fill"),
            "line layer should not have fill due to null AS fill"
        );
    }

    // ========================================================================
    // Validation Error Message Tests (User-facing aesthetic names)
    // ========================================================================

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_validation_error_shows_user_facing_names_for_missing_aesthetics() {
        // Test that validation errors show user-facing names (x, y) instead of internal (pos1, pos2)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE test_data AS SELECT * FROM (VALUES (1, 2)) AS t(a, b)",
                duckdb::params![],
            )
            .unwrap();

        // Query missing required aesthetic 'y' - should show 'y' not 'pos2'
        let query = r#"
            SELECT * FROM test_data
            VISUALISE
            DRAW point MAPPING a AS x
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(result.is_err(), "Expected validation error");

        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(
            err_msg.contains("`y`"),
            "Error should mention user-facing name 'y', got: {}",
            err_msg
        );
        assert!(
            !err_msg.contains("pos2"),
            "Error should not mention internal name 'pos2', got: {}",
            err_msg
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_validation_error_shows_user_facing_names_for_unsupported_aesthetics() {
        // Test that validation errors show user-facing names for unsupported aesthetics
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute(
                "CREATE TABLE test_data AS SELECT * FROM (VALUES (1, 2, 3)) AS t(a, b, c)",
                duckdb::params![],
            )
            .unwrap();

        // Query with unsupported aesthetic 'xmin' for point - should show 'xmin' not 'pos1min'
        let query = r#"
            SELECT * FROM test_data
            VISUALISE
            DRAW point MAPPING a AS x, b AS y, c AS xmin
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(result.is_err(), "Expected validation error");

        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(
            err_msg.contains("`xmin`"),
            "Error should mention user-facing name 'xmin', got: {}",
            err_msg
        );
        assert!(
            !err_msg.contains("pos1min"),
            "Error should not mention internal name 'pos1min', got: {}",
            err_msg
        );
    }

    // =========================================================================
    // Case-insensitive column reference normalization
    // =========================================================================

    /// Original reproducer: DuckDB lowercases unquoted identifiers, so
    /// `SELECT category` returns `category`. `VISUALISE CATEGORY AS x` must
    /// resolve to `category` before validation.
    #[cfg(feature = "duckdb")]
    #[test]
    fn test_case_insensitive_visualise_refs() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE case_test AS SELECT 'A' AS category, 10 AS value \
                 UNION ALL SELECT 'B', 20",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT category, value FROM case_test
            VISUALISE CATEGORY AS x, VALUE AS y
            DRAW bar
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(
            result.is_ok(),
            "Uppercase VISUALISE refs should resolve to lowercase schema: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_mixed_case_visualise_refs() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE mixed AS SELECT 'A' AS category, 10 AS value \
                 UNION ALL SELECT 'B', 20",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT category, value FROM mixed
            VISUALISE CaTeGoRy AS x, VaLuE AS y
            DRAW bar
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(
            result.is_ok(),
            "Mixed-case VISUALISE refs should normalize: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_case_insensitive_partition_by() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE pb AS SELECT 1 AS x, 10 AS y, 'A' AS category \
                 UNION ALL SELECT 2, 20, 'B'",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT x, y, category FROM pb
            VISUALISE x AS x, y AS y
            DRAW line PARTITION BY CATEGORY
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(
            result.is_ok(),
            "Uppercase PARTITION BY should resolve to lowercase schema: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_case_insensitive_facet() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE facet_case AS SELECT 1 AS x, 10 AS y, 'N' AS region \
                 UNION ALL SELECT 2, 20, 'S'",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            SELECT x, y, region FROM facet_case
            VISUALISE x AS x, y AS y
            DRAW point
            FACET REGION
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(
            result.is_ok(),
            "Uppercase FACET variable should resolve to lowercase schema: {:?}",
            result.err()
        );
    }

    /// Multi-source layers with different schemas — the case the old PR #143
    /// got wrong. Each layer's mapping must be normalized against that layer's
    /// own schema, not the first layer's.
    #[cfg(feature = "duckdb")]
    #[test]
    fn test_multi_source_layers_case_insensitive() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE temps AS SELECT 1 AS date, 20.0 AS value \
                 UNION ALL SELECT 2, 21.0",
                duckdb::params![],
            )
            .unwrap();
        reader
            .connection()
            .execute(
                "CREATE TABLE ozone AS SELECT 1 AS date, 0.05 AS value \
                 UNION ALL SELECT 2, 0.06",
                duckdb::params![],
            )
            .unwrap();

        let query = r#"
            VISUALISE Date AS x, Value AS y
            DRAW line MAPPING Date AS x, Value AS y FROM temps
            DRAW line MAPPING DATE AS x, VALUE AS y FROM ozone
        "#;

        let result = prepare_data_with_reader(query, &reader);
        assert!(
            result.is_ok(),
            "Per-layer normalization should work across multi-source layers: {:?}",
            result.err()
        );
    }

    // Direct tests for the normalizer's match-resolution contract.
    // (DuckDB itself rejects case-colliding column names even when quoted,
    // so the ambiguous/exact-match scenarios can't be exercised end-to-end.)

    #[test]
    fn test_normalize_column_ref_exact_match_wins() {
        let mut name = "foo".to_string();
        normalize_column_ref(&mut name, &["Foo", "foo"]);
        assert_eq!(name, "foo");
    }

    #[test]
    fn test_normalize_column_ref_unique_case_match_rewrites() {
        let mut name = "CATEGORY".to_string();
        normalize_column_ref(&mut name, &["category", "value"]);
        assert_eq!(name, "category");
    }

    #[test]
    fn test_normalize_column_ref_ambiguous_left_alone() {
        let mut name = "FOO".to_string();
        normalize_column_ref(&mut name, &["Foo", "foo"]);
        assert_eq!(name, "FOO", "ambiguous match must not be silently resolved");
    }

    #[test]
    fn test_normalize_column_ref_no_match_left_alone() {
        let mut name = "missing".to_string();
        normalize_column_ref(&mut name, &["a", "b"]);
        assert_eq!(name, "missing");
    }

    // =========================================================================
    // validate() — internal aesthetic names must not leak into error messages
    // =========================================================================

    mod validate_translation_tests {
        use super::*;
        use crate::plot::layer::geom::Geom;
        use crate::plot::layer::Layer;
        use crate::plot::types::{AestheticValue, ColumnInfo};
        use arrow::datatypes::DataType;

        fn col(name: &str) -> ColumnInfo {
            ColumnInfo {
                name: name.to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            }
        }

        fn point_layer_with_mapping(aesthetic: &str, column: &str) -> Layer {
            let mut layer = Layer::new(Geom::point());
            // Point geom requires both pos1 and pos2; map both, but only the
            // one we're testing points at a missing column.
            layer.mappings.aesthetics.insert(
                "pos1".to_string(),
                AestheticValue::standard_column("present_x"),
            );
            layer.mappings.aesthetics.insert(
                "pos2".to_string(),
                AestheticValue::standard_column("present_y"),
            );
            layer.mappings.aesthetics.insert(
                aesthetic.to_string(),
                AestheticValue::standard_column(column),
            );
            layer
        }

        #[test]
        fn aesthetic_column_missing_translates_pos1_to_x_under_cartesian() {
            let mut layer = Layer::new(Geom::point());
            layer.mappings.aesthetics.insert(
                "pos1".to_string(),
                AestheticValue::standard_column("missing"),
            );
            layer.mappings.aesthetics.insert(
                "pos2".to_string(),
                AestheticValue::standard_column("present_y"),
            );
            let schema: Schema = vec![col("present_y")];
            let ctx = Some(AestheticContext::from_static(&["x", "y"], &[]));

            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: aesthetic 'x' references non-existent column 'missing'"
            );
        }

        #[test]
        fn aesthetic_column_missing_translates_pos1_to_angle_under_polar() {
            let mut layer = Layer::new(Geom::point());
            layer.mappings.aesthetics.insert(
                "pos1".to_string(),
                AestheticValue::standard_column("missing"),
            );
            layer.mappings.aesthetics.insert(
                "pos2".to_string(),
                AestheticValue::standard_column("present_radius"),
            );
            let schema: Schema = vec![col("present_radius")];
            let ctx = Some(AestheticContext::from_static(&["angle", "radius"], &[]));

            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: aesthetic 'angle' references non-existent column 'missing'"
            );
        }

        #[test]
        fn aesthetic_column_missing_translates_pos2_to_y_under_cartesian() {
            let layer = point_layer_with_mapping("pos2", "missing");
            let schema: Schema = vec![col("present_x"), col("present_y")];
            let ctx = Some(AestheticContext::from_static(&["x", "y"], &[]));

            // pos2 is overridden to point at "missing" by point_layer_with_mapping
            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: aesthetic 'y' references non-existent column 'missing'"
            );
        }

        #[test]
        fn material_aesthetic_column_missing_keeps_color_name() {
            // Material aesthetics (color, size, etc.) should round-trip unchanged.
            let layer = point_layer_with_mapping("color", "missing");
            let schema: Schema = vec![col("present_x"), col("present_y")];
            let ctx = Some(AestheticContext::from_static(&["x", "y"], &[]));

            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: aesthetic 'color' references non-existent column 'missing'"
            );
        }

        #[test]
        fn remapping_unsupported_target_translates_pos2_to_y_under_cartesian() {
            // Histogram has a stat transform and supports REMAPPING, but does
            // not support `shape`. Aim a remapping at an aesthetic histogram
            // does support but aliased so we can verify pos→user translation:
            // route via `pos2` mapped target (the histogram stat output is
            // `count`).  We need a target the geom actually rejects.  Simpler:
            // build a Layer manually with a remapping at "pos9" (truly unsupported)
            // and verify pos translation runs even on an unrecognized name.
            //
            // For an end-to-end position translation case, we use a layer with
            // a histogram geom and a remapping target that becomes "pos1max"
            // after parser transformation — histogram does not support pos1max
            // (only pos1 and the delayed pos2). The test here builds the
            // post-transform state directly.
            let mut layer = Layer::new(Geom::histogram());
            layer.mappings.aesthetics.insert(
                "pos1".to_string(),
                AestheticValue::standard_column("present_x"),
            );
            layer.remappings.aesthetics.insert(
                "pos1max".to_string(),
                AestheticValue::standard_column("count"),
            );
            let schema: Schema = vec![col("present_x")];
            let ctx = Some(AestheticContext::from_static(&["x", "y"], &[]));

            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: REMAPPING targets unsupported aesthetic 'xmax' for geom 'histogram'"
            );
        }

        #[test]
        fn remapping_unsupported_target_translates_pos1max_to_anglemax_under_polar() {
            let mut layer = Layer::new(Geom::histogram());
            layer.mappings.aesthetics.insert(
                "pos1".to_string(),
                AestheticValue::standard_column("present_x"),
            );
            layer.remappings.aesthetics.insert(
                "pos1max".to_string(),
                AestheticValue::standard_column("count"),
            );
            let schema: Schema = vec![col("present_x")];
            let ctx = Some(AestheticContext::from_static(&["angle", "radius"], &[]));

            let err = validate(&[layer], &[schema], &ctx).unwrap_err().to_string();
            assert_eq!(
                err,
                "Validation error: Layer 1: REMAPPING targets unsupported aesthetic 'anglemax' for geom 'histogram'"
            );
        }
    }
}
