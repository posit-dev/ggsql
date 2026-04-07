//! Dodge position adjustment
//!
//! Positions elements side-by-side within groups. Dodge automatically detects
//! which axes are discrete and applies dodge accordingly:
//! - If only pos1 is discrete → dodge horizontally (pos1offset)
//! - If only pos2 is discrete → dodge vertically (pos2offset)
//! - If both are discrete → 2D grid dodge (both offsets, arranged in a grid)

use super::{
    compute_dodge_offsets, is_continuous_scale, non_facet_partition_cols, Layer, PositionTrait,
    PositionType,
};
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition, ParameterValue};
use crate::{naming, DataFrame, GgsqlError, Plot, Result};
use polars::prelude::*;
use std::collections::HashMap;

/// Result of computing group indices for dodge/jitter operations.
///
/// Contains the number of unique groups and the group index for each row.
pub struct GroupIndices {
    /// Number of unique groups
    pub n_groups: usize,
    /// Group index (0 to n_groups-1) for each row
    pub indices: Vec<usize>,
}

/// Compute group indices from partition_by columns.
///
/// Returns None if there are no grouping columns or columns don't exist.
/// Returns Some(GroupIndices) with n_groups=1 if there's only one group.
pub fn compute_group_indices(
    df: &DataFrame,
    group_cols: &[String],
) -> Result<Option<GroupIndices>> {
    if group_cols.is_empty() {
        return Ok(None);
    }

    // Check if required grouping columns exist
    for col_name in group_cols {
        if df.column(col_name).is_err() {
            return Ok(None);
        }
    }

    // Create composite key for each row by concatenating all grouping column values
    let n_rows = df.height();
    let mut composite_keys: Vec<String> = Vec::with_capacity(n_rows);

    for row_idx in 0..n_rows {
        let mut key_parts: Vec<String> = Vec::with_capacity(group_cols.len());
        for col_name in group_cols {
            let col = df.column(col_name).unwrap();
            let val = col.get(row_idx).map_err(|e| {
                GgsqlError::InternalError(format!("Failed to get value at row {}: {}", row_idx, e))
            })?;
            key_parts.push(format!("{}", val));
        }
        composite_keys.push(key_parts.join("\x00")); // Use null byte as separator
    }

    // Get unique composite keys and sort them for consistent ordering
    let mut unique_keys: Vec<String> = composite_keys.to_vec();
    unique_keys.sort();
    unique_keys.dedup();

    let n_groups = unique_keys.len();

    // Create mapping from composite key to index
    let key_to_idx: HashMap<String, usize> = unique_keys
        .into_iter()
        .enumerate()
        .map(|(idx, key)| (key, idx))
        .collect();

    // Create index column by mapping each row's composite key
    let indices: Vec<usize> = composite_keys
        .iter()
        .map(|key| *key_to_idx.get(key).unwrap())
        .collect();

    Ok(Some(GroupIndices { n_groups, indices }))
}

/// Dodge position - position elements side-by-side
#[derive(Debug, Clone, Copy)]
pub struct Dodge;

impl std::fmt::Display for Dodge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dodge")
    }
}

impl PositionTrait for Dodge {
    fn position_type(&self) -> PositionType {
        PositionType::Dodge
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "width",
            default: DefaultParamValue::Number(0.9),
            constraint: ParamConstraint::number_range(0.0, 1.0),
        }];
        PARAMS
    }

    fn creates_pos1offset(&self) -> bool {
        true
    }

    fn creates_pos2offset(&self) -> bool {
        true // May create pos2offset when pos2 is discrete
    }

    fn apply_adjustment(
        &self,
        df: DataFrame,
        layer: &Layer,
        spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)> {
        apply_dodge_with_width(df, layer, spec)
    }
}

/// Apply dodge position adjustment and compute adjusted bar width.
///
/// Automatically detects which axes are discrete and applies dodge accordingly:
/// - Discrete pos1 only → creates pos1offset column (horizontal dodge)
/// - Discrete pos2 only → creates pos2offset column (vertical dodge)
/// - Both discrete → creates both offset columns (2D grid arrangement)
/// - Neither discrete → returns unchanged (no dodge applied)
///
/// For 2D grid dodge, groups are arranged in a square grid pattern. For example:
/// - 4 groups → 2x2 grid
/// - 8 groups → 3x3 grid (one cell empty)
/// - 9 groups → 3x3 grid (all cells filled)
///
/// If an existing "offset" column exists (e.g., from violin geom), scales it by n_groups
/// so the layer can use adjusted values for its shape rendering.
/// Also returns the adjusted bar width (original width / n_groups for 1D, or
/// original width / grid_size for 2D).
fn apply_dodge_with_width(
    df: DataFrame,
    layer: &Layer,
    spec: &Plot,
) -> Result<(DataFrame, Option<f64>)> {
    let offset_col = naming::aesthetic_column("offset");
    let pos1offset_col = naming::aesthetic_column("pos1offset");
    let pos2offset_col = naming::aesthetic_column("pos2offset");

    // Check which axes should be dodged (discrete axes)
    // Since create_missing_scales_post_stat() runs before position adjustments,
    // scale types are always known, so we use explicit discrete checks.
    let dodge_pos1 = is_continuous_scale(spec, "pos1") == Some(false);
    let dodge_pos2 = is_continuous_scale(spec, "pos2") == Some(false);

    // If neither is discrete, nothing to dodge
    if !dodge_pos1 && !dodge_pos2 {
        return Ok((df, None));
    }

    // Compute group indices, excluding facet columns so group count
    // reflects within-panel groups (not cross-panel composites)
    let group_cols = non_facet_partition_cols(&layer.partition_by, spec);
    let group_info = match compute_group_indices(&df, &group_cols)? {
        Some(info) => info,
        None => return Ok((df, None)),
    };

    let GroupIndices { n_groups, indices } = group_info;

    if n_groups <= 1 {
        // Only one group - no dodging needed
        return Ok((df, None));
    }

    // Get the default bar width from layer parameters (or use 0.9 as default)
    let bar_width = layer
        .parameters
        .get("width")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0.9);

    // Check if layer has an existing offset column (e.g., violin density offset)
    let has_offset_col = df.column(&offset_col).is_ok();

    // Compute dodge offsets using shared logic
    let offsets = compute_dodge_offsets(&indices, n_groups, bar_width, dodge_pos1, dodge_pos2);

    let mut lf = df.lazy();

    // Apply the computed offsets
    if let Some(pos1_offsets) = offsets.pos1 {
        lf = lf.with_column(
            lit(Series::new(pos1offset_col.clone().into(), pos1_offsets)).alias(&pos1offset_col),
        );
    }
    if let Some(pos2_offsets) = offsets.pos2 {
        lf = lf.with_column(
            lit(Series::new(pos2offset_col.clone().into(), pos2_offsets)).alias(&pos2offset_col),
        );
    }

    // If offset column exists (e.g., violin), scale it by the offset scale factor
    if has_offset_col {
        lf = lf.with_column((col(&offset_col) / lit(offsets.offset_scale)).alias(&offset_col));
    }

    // Collect the result
    let final_df = lf.collect().map_err(|e| {
        GgsqlError::InternalError(format!("Dodge position adjustment failed: {}", e))
    })?;

    Ok((final_df, Some(offsets.adjusted_width)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::layer::Geom;
    use crate::plot::{AestheticValue, Mappings, Scale, ScaleType};

    fn make_test_df() -> DataFrame {
        df! {
            "__ggsql_aes_pos1__" => ["A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "Y", "X", "Y"],
        }
        .unwrap()
    }

    fn make_test_layer() -> Layer {
        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];
        layer
    }

    fn make_continuous_scale(aesthetic: &str) -> Scale {
        let mut scale = Scale::new(aesthetic);
        scale.scale_type = Some(ScaleType::continuous());
        scale
    }

    fn make_discrete_scale(aesthetic: &str) -> Scale {
        let mut scale = Scale::new(aesthetic);
        scale.scale_type = Some(ScaleType::discrete());
        scale
    }

    #[test]
    fn test_dodge_horizontal_only() {
        // When pos1 is discrete and pos2 is continuous, only pos1offset is created
        let dodge = Dodge;
        assert_eq!(dodge.position_type(), PositionType::Dodge);

        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos1 as discrete and pos2 as continuous via scales
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify pos1offset column was created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_ok(),
            "pos1offset column should be created"
        );
        // Verify pos2offset column was NOT created
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_err(),
            "pos2offset column should NOT be created when pos2 is continuous"
        );

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();

        // With 2 groups (X, Y) and default width 0.9:
        // - adjusted_width = 0.9 / 2 = 0.45
        // - center_offset = 0.5
        // - Group X: center = (0 - 0.5) * 0.45 = -0.225
        // - Group Y: center = (1 - 0.5) * 0.45 = +0.225
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
        assert!(
            offsets.iter().any(|&v| (v - (-0.225)).abs() < 0.001),
            "Should have offset -0.225 for group X, got {:?}",
            offsets
        );
        assert!(
            offsets.iter().any(|&v| (v - 0.225).abs() < 0.001),
            "Should have offset +0.225 for group Y, got {:?}",
            offsets
        );

        // Verify adjusted_width was returned
        assert!(width.is_some());
        assert!(
            (width.unwrap() - 0.45).abs() < 0.001,
            "adjusted_width should be 0.9/2 = 0.45, got {:?}",
            width
        );
    }

    #[test]
    fn test_dodge_vertical_only() {
        // When pos1 is continuous and pos2 is discrete, only pos2offset is created
        let dodge = Dodge;

        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos1 as continuous and pos2 as discrete via scales
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify pos1offset column was NOT created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_err(),
            "pos1offset column should NOT be created when pos1 is continuous"
        );
        // Verify pos2offset column was created
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_ok(),
            "pos2offset column should be created"
        );

        let offset = result
            .column("__ggsql_aes_pos2offset__")
            .unwrap()
            .f64()
            .unwrap();

        // With 2 groups (X, Y) and default width 0.9:
        // - adjusted_width = 0.9 / 2 = 0.45
        // - center_offset = 0.5
        // - Group X: center = (0 - 0.5) * 0.45 = -0.225
        // - Group Y: center = (1 - 0.5) * 0.45 = +0.225
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
        assert!(
            offsets.iter().any(|&v| (v - (-0.225)).abs() < 0.001),
            "Should have offset -0.225 for group X, got {:?}",
            offsets
        );
        assert!(
            offsets.iter().any(|&v| (v - 0.225).abs() < 0.001),
            "Should have offset +0.225 for group Y, got {:?}",
            offsets
        );

        // Verify adjusted_width was returned
        assert!(width.is_some());
        assert!(
            (width.unwrap() - 0.45).abs() < 0.001,
            "adjusted_width should be 0.9/2 = 0.45, got {:?}",
            width
        );
    }

    #[test]
    fn test_dodge_bidirectional_2x2_grid() {
        // When both axes are discrete, groups are arranged in a 2D grid
        let dodge = Dodge;

        let df = make_test_df();
        let layer = make_test_layer();

        // Both axes must be explicitly marked as discrete
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify both offset columns were created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_ok(),
            "pos1offset column should be created"
        );
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_ok(),
            "pos2offset column should be created"
        );

        // With 2 groups in 2D mode, grid_size = ceil(sqrt(2)) = 2
        // adjusted_width = 0.9 / 2 = 0.45
        // center_offset = (2 - 1) / 2 = 0.5
        // Group 0 (X): col=0, row=0 → pos1=(-0.5)*0.45=-0.225, pos2=(-0.5)*0.45=-0.225
        // Group 1 (Y): col=1, row=0 → pos1=(0.5)*0.45=0.225, pos2=(-0.5)*0.45=-0.225
        let pos1_offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let pos2_offset = result
            .column("__ggsql_aes_pos2offset__")
            .unwrap()
            .f64()
            .unwrap();

        let pos1_offsets: Vec<f64> = pos1_offset.into_iter().flatten().collect();
        let pos2_offsets: Vec<f64> = pos2_offset.into_iter().flatten().collect();

        // Verify we have both expected pos1 offsets
        assert!(
            pos1_offsets.iter().any(|&v| (v - (-0.225)).abs() < 0.001),
            "Should have pos1offset -0.225, got {:?}",
            pos1_offsets
        );
        assert!(
            pos1_offsets.iter().any(|&v| (v - 0.225).abs() < 0.001),
            "Should have pos1offset +0.225, got {:?}",
            pos1_offsets
        );

        // Verify pos2 offsets (in 2x2 grid with 2 groups, both groups are in row 0)
        // Group 0: row=0, Group 1: row=0
        // So all pos2 offsets should be (0 - 0.5) * 0.45 = -0.225
        for &v in &pos2_offsets {
            assert!(
                (v - (-0.225)).abs() < 0.001,
                "All pos2 offsets should be -0.225 for 2 groups in 2x2 grid, got {}",
                v
            );
        }

        // Verify adjusted_width
        assert!(width.is_some());
        assert!(
            (width.unwrap() - 0.45).abs() < 0.001,
            "adjusted_width should be 0.9/2 = 0.45, got {:?}",
            width
        );
    }

    #[test]
    fn test_dodge_bidirectional_3x3_grid() {
        // Test with 4 groups to verify 2x2 arrangement within 2x2 grid
        let dodge = Dodge;

        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_fill__" => ["G1", "G2", "G3", "G4"],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::point());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];

        // Both axes must be explicitly marked as discrete
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        // With 4 groups in 2D mode, grid_size = ceil(sqrt(4)) = 2
        // This gives a 2x2 grid layout:
        // G1: col=0, row=0 → (-0.5, -0.5) * adjusted_width
        // G2: col=1, row=0 → (+0.5, -0.5) * adjusted_width
        // G3: col=0, row=1 → (-0.5, +0.5) * adjusted_width
        // G4: col=1, row=1 → (+0.5, +0.5) * adjusted_width

        let pos1_offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let pos2_offset = result
            .column("__ggsql_aes_pos2offset__")
            .unwrap()
            .f64()
            .unwrap();

        let pos1_offsets: Vec<f64> = pos1_offset.into_iter().flatten().collect();
        let pos2_offsets: Vec<f64> = pos2_offset.into_iter().flatten().collect();

        // Verify we have both positive and negative offsets in both dimensions
        assert!(
            pos1_offsets.iter().any(|&v| v < 0.0),
            "Should have negative pos1 offsets"
        );
        assert!(
            pos1_offsets.iter().any(|&v| v > 0.0),
            "Should have positive pos1 offsets"
        );
        assert!(
            pos2_offsets.iter().any(|&v| v < 0.0),
            "Should have negative pos2 offsets"
        );
        assert!(
            pos2_offsets.iter().any(|&v| v > 0.0),
            "Should have positive pos2 offsets"
        );

        // Verify adjusted_width = 0.9 / 2 = 0.45
        assert!(width.is_some());
        assert!(
            (width.unwrap() - 0.45).abs() < 0.001,
            "adjusted_width should be 0.9/2 = 0.45 for 4 groups (2x2 grid), got {:?}",
            width
        );
    }

    #[test]
    fn test_dodge_neither_discrete() {
        // When both axes are continuous, no offset columns are created
        let dodge = Dodge;

        let df = make_test_df();
        let layer = make_test_layer();

        // Mark both as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify neither offset column was created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_err(),
            "pos1offset column should NOT be created when pos1 is continuous"
        );
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_err(),
            "pos2offset column should NOT be created when pos2 is continuous"
        );

        // No adjusted width when no dodging occurs
        assert!(width.is_none());
    }

    #[test]
    fn test_dodge_custom_width() {
        let dodge = Dodge;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer
            .parameters
            .insert("width".to_string(), ParameterValue::Number(0.6));

        // Mark pos1 as discrete and pos2 as continuous so only pos1offset is created
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, width) = dodge.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();

        // With 2 groups and custom width 0.6:
        // - adjusted_width = 0.6 / 2 = 0.3
        // - center_offset = 0.5
        // - Group X: center = (0 - 0.5) * 0.3 = -0.15
        // - Group Y: center = (1 - 0.5) * 0.3 = +0.15
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
        assert!(
            offsets.iter().any(|&v| (v - (-0.15)).abs() < 0.001),
            "Should have offset -0.15 for group X, got {:?}",
            offsets
        );
        assert!(
            offsets.iter().any(|&v| (v - 0.15).abs() < 0.001),
            "Should have offset +0.15 for group Y, got {:?}",
            offsets
        );

        assert!((width.unwrap() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_dodge_creates_pos1offset() {
        assert!(Dodge.creates_pos1offset());
    }

    #[test]
    fn test_dodge_creates_pos2offset() {
        assert!(Dodge.creates_pos2offset());
    }

    #[test]
    fn test_dodge_default_params() {
        let dodge = Dodge;
        let params = dodge.default_params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "width");
        assert!(matches!(params[0].default, DefaultParamValue::Number(0.9)));
    }
}
