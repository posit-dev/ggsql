//! Position adjustment dispatch for layers
//!
//! This module applies position adjustments to layers after DataFrame materialization
//! but before scale training. This ensures scales see the adjusted values.
//!
//! The actual position adjustment algorithms are implemented in the position module
//! (`src/plot/layer/position/`). This module provides the dispatch logic.

use crate::plot::{Plot, PositionType};
use crate::{DataFrame, Result};
use std::collections::HashMap;

/// Apply position adjustments to all layers in the spec.
///
/// For each layer with a non-identity position:
/// - Stack: modifies pos2/pos2end columns with cumulative sums
/// - Dodge: creates pos1offset column for horizontal displacement, adjusts bar width
/// - Jitter: creates pos1offset/pos2offset columns with random displacement
///
/// Must be called after resolve_aesthetics() but before resolve_scales().
pub fn apply_position_adjustments(
    spec: &mut Plot,
    data_map: &mut HashMap<String, DataFrame>,
) -> Result<()> {
    for idx in 0..spec.layers.len() {
        // Skip identity position (no adjustment needed)
        if spec.layers[idx].position.position_type() == PositionType::Identity {
            continue;
        }

        let Some(key) = spec.layers[idx].data_key.clone() else {
            continue;
        };

        let Some(df) = data_map.remove(&key) else {
            continue;
        };

        // Delegate to the position's apply_adjustment implementation
        // Each position validates its own requirements internally
        let (adjusted_df, adjusted_width) =
            spec.layers[idx]
                .position
                .apply_adjustment(df, &spec.layers[idx], spec)?;

        data_map.insert(key.clone(), adjusted_df);

        // Store adjusted width on layer (for writers that need it)
        // This does NOT override the user's width parameter
        if let Some(width) = adjusted_width {
            spec.layers[idx].adjusted_width = Some(width);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_util::as_f64;
    use crate::df;
    use crate::plot::facet::{Facet, FacetLayout};
    use crate::plot::layer::{Geom, Position};
    use crate::plot::{AestheticValue, Mappings, ParameterValue, Scale, ScaleType};
    use arrow::array::Array;

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

    fn make_test_df() -> DataFrame {
        df! {
            "__ggsql_aes_pos1__" => vec!["A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => vec![10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos2end__" => vec![0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => vec!["X", "Y", "X", "Y"],
        }
        .unwrap()
    }

    fn make_test_layer() -> crate::plot::Layer {
        let mut layer = crate::plot::Layer::new(Geom::bar());
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
        // Add fill to partition_by (simulates what add_discrete_columns_to_partition_by does)
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];
        layer
    }

    #[test]
    fn test_identity_no_change() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::identity();

        let spec = Plot::new();
        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df.clone());

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        // Data should be unchanged
        let result_df = data_map.get("__ggsql_layer_0__").unwrap();
        assert_eq!(result_df.height(), 4);
    }

    #[test]
    fn test_stack_cumsum() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::stack();

        let spec = Plot::new();
        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();
        let pos2_col = result_df.column("__ggsql_aes_pos2__").unwrap();
        let pos2end_col = result_df.column("__ggsql_aes_pos2end__").unwrap();

        // Verify stacking was applied (column should be numeric)
        assert!(
            matches!(
                pos2_col.data_type(),
                arrow::datatypes::DataType::Float64
                    | arrow::datatypes::DataType::Int64
                    | arrow::datatypes::DataType::Int32
            ),
            "pos2 should be numeric"
        );
        assert!(
            matches!(
                pos2end_col.data_type(),
                arrow::datatypes::DataType::Float64
                    | arrow::datatypes::DataType::Int64
                    | arrow::datatypes::DataType::Int32
            ),
            "pos2end should be numeric"
        );
    }

    #[test]
    fn test_dodge_offset() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::dodge();

        // Create spec with pos1 as discrete and pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();

        // Verify pos1offset column was created
        let offset_col = result_df.column("__ggsql_aes_pos1offset__");
        assert!(offset_col.is_ok(), "pos1offset column should be created");

        let offset = as_f64(offset_col.unwrap()).unwrap();

        // With 2 groups (X, Y) and default width 0.9:
        // - adjusted_width = 0.9 / 2 = 0.45
        // - center_offset = 0.5
        // - Group X: center = (0 - 0.5) * 0.45 = -0.225
        // - Group Y: center = (1 - 0.5) * 0.45 = +0.225
        let offsets: Vec<f64> = (0..offset.len())
            .filter(|&i| !offset.is_null(i))
            .map(|i| offset.value(i))
            .collect();
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

        // Verify adjusted_width was set
        let adjusted = spec_with_layer.layers[0].adjusted_width;
        assert!(adjusted.is_some());
        assert!((adjusted.unwrap() - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_dodge_custom_width() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::dodge();
        layer
            .parameters
            .insert("width".to_string(), ParameterValue::Number(0.6));

        // Create spec with pos1 as discrete and pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();
        let offset_col = result_df.column("__ggsql_aes_pos1offset__").unwrap();
        let offset = as_f64(offset_col).unwrap();

        // With 2 groups and custom width 0.6:
        // - adjusted_width = 0.6 / 2 = 0.3
        let offsets: Vec<f64> = (0..offset.len())
            .filter(|&i| !offset.is_null(i))
            .map(|i| offset.value(i))
            .collect();
        assert!(offsets.iter().any(|&v| (v - (-0.15)).abs() < 0.001));
        assert!(offsets.iter().any(|&v| (v - 0.15).abs() < 0.001));

        let adjusted = spec_with_layer.layers[0].adjusted_width;
        assert!((adjusted.unwrap() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_jitter_offset() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::jitter();

        // Create spec with pos1 as discrete and pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();

        // Verify pos1offset column was created
        let offset_col = result_df.column("__ggsql_aes_pos1offset__");
        assert!(offset_col.is_ok());

        let offset = as_f64(offset_col.unwrap()).unwrap();
        let offsets: Vec<f64> = (0..offset.len())
            .filter(|&i| !offset.is_null(i))
            .map(|i| offset.value(i))
            .collect();

        // With default width 0.9, offsets should be in range [-0.45, 0.45]
        for &v in &offsets {
            assert!((-0.45..=0.45).contains(&v));
        }

        // No adjusted_width for jitter
        assert!(spec_with_layer.layers[0].adjusted_width.is_none());
    }

    #[test]
    fn test_jitter_custom_width() {
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.position = Position::jitter();
        layer
            .parameters
            .insert("width".to_string(), ParameterValue::Number(0.6));

        // Create spec with pos1 as discrete and pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let mut data_map = HashMap::new();
        layer.data_key = Some("__ggsql_layer_0__".to_string());
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();
        let offset_col = result_df.column("__ggsql_aes_pos1offset__").unwrap();
        let offset = as_f64(offset_col).unwrap();
        let offsets: Vec<f64> = (0..offset.len())
            .filter(|&i| !offset.is_null(i))
            .map(|i| offset.value(i))
            .collect();

        // With custom width 0.6, offsets should be in range [-0.3, 0.3]
        for &v in &offsets {
            assert!((-0.3..=0.3).contains(&v));
        }
    }

    #[test]
    fn test_stack_resets_per_facet_panel() {
        // Stacking should compute independently within each facet panel.
        // Without this, bars in the second facet panel stack on top of
        // cumulative values from the first panel (see issue #244).
        //
        // Two facet panels (F1, F2) each with the same x="A" and two
        // fill groups (X, Y). Stacking within each panel should start from 0.
        let df = df! {
            "__ggsql_aes_pos1__" => vec!["A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => vec![10.0, 20.0, 30.0, 40.0],
            "__ggsql_aes_pos2end__" => vec![0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => vec!["X", "Y", "X", "Y"],
            "__ggsql_aes_facet1__" => vec!["F1", "F1", "F2", "F2"],
        }
        .unwrap();

        let mut layer = crate::plot::Layer::new(Geom::bar());
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
            m.insert(
                "facet1",
                AestheticValue::standard_column("__ggsql_aes_facet1__"),
            );
            m
        };
        layer.partition_by = vec![
            "__ggsql_aes_fill__".to_string(),
            "__ggsql_aes_facet1__".to_string(),
        ];
        layer.position = Position::stack();
        layer.data_key = Some("__ggsql_layer_0__".to_string());

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        spec.facet = Some(Facet::new(FacetLayout::Wrap {
            variables: vec!["facet_var".to_string()],
        }));
        let mut data_map = HashMap::new();
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        let mut spec_with_layer = spec;
        spec_with_layer.layers.push(layer);

        apply_position_adjustments(&mut spec_with_layer, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();

        // Sort by facet then fill so we can assert in predictable order
        // Build sort indices based on (facet, fill) lexicographic order
        let facet_col =
            crate::array_util::as_str(result_df.column("__ggsql_aes_facet1__").unwrap()).unwrap();
        let fill_col =
            crate::array_util::as_str(result_df.column("__ggsql_aes_fill__").unwrap()).unwrap();
        let mut indices: Vec<usize> = (0..result_df.height()).collect();
        indices.sort_by(|&a, &b| {
            let fa = facet_col.value(a);
            let fb = facet_col.value(b);
            let cmp1 = fa.cmp(fb);
            if cmp1 != std::cmp::Ordering::Equal {
                return cmp1;
            }
            fill_col.value(a).cmp(fill_col.value(b))
        });

        let pos2_arr = as_f64(result_df.column("__ggsql_aes_pos2__").unwrap()).unwrap();
        let pos2end_arr = as_f64(result_df.column("__ggsql_aes_pos2end__").unwrap()).unwrap();

        let pos2_vals: Vec<f64> = indices.iter().map(|&i| pos2_arr.value(i)).collect();
        let pos2end_vals: Vec<f64> = indices.iter().map(|&i| pos2end_arr.value(i)).collect();

        // Expected (sorted by facet, fill):
        // F1/X: pos2end=0,  pos2=10  (first in panel, starts at 0)
        // F1/Y: pos2end=10, pos2=30  (stacks on X)
        // F2/X: pos2end=0,  pos2=30  (first in panel, should reset to 0)
        // F2/Y: pos2end=30, pos2=70  (stacks on X)
        assert_eq!(
            pos2end_vals[2], 0.0,
            "F2 panel first bar should start at 0, not carry over from F1. pos2end={:?}, pos2={:?}",
            pos2end_vals, pos2_vals
        );
    }

    #[test]
    fn test_stack_groups_by_facet_not_fill_order() {
        // Regression: when partition_by listed fill before facet, the sort order
        // put fill first, interleaving facet panels. compute_group_ids then
        // treated each row as its own group and stacking had no effect.
        //
        // Data is pre-sorted by (fill, facet) — the worst case for the old code.
        // Three facet panels (F1, F2, F3) each with fill groups (X, Y).
        let df = df! {
            "__ggsql_aes_pos1__" => vec!["A", "A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "__ggsql_aes_pos2end__" => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => vec!["X", "X", "X", "Y", "Y", "Y"],
            "__ggsql_aes_facet1__" => vec!["F1", "F2", "F3", "F1", "F2", "F3"],
        }
        .unwrap();

        let mut layer = crate::plot::Layer::new(Geom::bar());
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
            m.insert(
                "facet1",
                AestheticValue::standard_column("__ggsql_aes_facet1__"),
            );
            m
        };
        // fill before facet — the order that triggered the bug
        layer.partition_by = vec![
            "__ggsql_aes_fill__".to_string(),
            "__ggsql_aes_facet1__".to_string(),
        ];
        layer.position = Position::stack();
        layer.data_key = Some("__ggsql_layer_0__".to_string());

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        spec.facet = Some(Facet::new(FacetLayout::Wrap {
            variables: vec!["facet_var".to_string()],
        }));
        let mut data_map = HashMap::new();
        data_map.insert("__ggsql_layer_0__".to_string(), df);
        spec.layers.push(layer);

        apply_position_adjustments(&mut spec, &mut data_map).unwrap();

        let result_df = data_map.get("__ggsql_layer_0__").unwrap();

        let facet_col =
            crate::array_util::as_str(result_df.column("__ggsql_aes_facet1__").unwrap()).unwrap();
        let fill_col =
            crate::array_util::as_str(result_df.column("__ggsql_aes_fill__").unwrap()).unwrap();
        let pos2_arr = as_f64(result_df.column("__ggsql_aes_pos2__").unwrap()).unwrap();
        let pos2end_arr = as_f64(result_df.column("__ggsql_aes_pos2end__").unwrap()).unwrap();

        // Collect (facet, fill) → (pos2, pos2end)
        let mut by_key: std::collections::HashMap<(&str, &str), (f64, f64)> =
            std::collections::HashMap::new();
        for i in 0..result_df.height() {
            by_key.insert(
                (facet_col.value(i), fill_col.value(i)),
                (pos2_arr.value(i), pos2end_arr.value(i)),
            );
        }

        // Within each facet the two fill groups must stack:
        // F1: X=10 → [0,10], Y=40 → [10,50]
        // F2: X=20 → [0,20], Y=50 → [20,70]
        // F3: X=30 → [0,30], Y=60 → [30,90]
        assert_eq!(by_key[&("F1", "X")], (10.0, 0.0));
        assert_eq!(by_key[&("F1", "Y")], (50.0, 10.0));
        assert_eq!(by_key[&("F2", "X")], (20.0, 0.0));
        assert_eq!(by_key[&("F2", "Y")], (70.0, 20.0));
        assert_eq!(by_key[&("F3", "X")], (30.0, 0.0));
        assert_eq!(by_key[&("F3", "Y")], (90.0, 30.0));
    }

    #[test]
    fn test_dodge_ignores_facet_columns_in_group_count() {
        // Dodge should compute n_groups per facet panel, not globally.
        // With fill=["X","Y"] and facet=["F1","F2"], dodge should see
        // 2 groups (X, Y) not 4 (X-F1, X-F2, Y-F1, Y-F2).
        //
        // With 2 groups and default width 0.9:
        //   adjusted_width = 0.9 / 2 = 0.45
        //   offsets: -0.225 (group X), +0.225 (group Y)
        //
        // If facet columns incorrectly inflate n_groups to 4:
        //   adjusted_width = 0.9 / 4 = 0.225
        //   offsets would be different (spread across 4 positions)
        let df = df! {
            "__ggsql_aes_pos1__" => vec!["A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => vec![10.0, 20.0, 30.0, 40.0],
            "__ggsql_aes_pos2end__" => vec![0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => vec!["X", "Y", "X", "Y"],
            "__ggsql_aes_facet1__" => vec!["F1", "F1", "F2", "F2"],
        }
        .unwrap();

        let mut layer = crate::plot::Layer::new(Geom::bar());
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
            m.insert(
                "facet1",
                AestheticValue::standard_column("__ggsql_aes_facet1__"),
            );
            m
        };
        layer.partition_by = vec![
            "__ggsql_aes_fill__".to_string(),
            "__ggsql_aes_facet1__".to_string(),
        ];
        layer.position = Position::dodge();
        layer.data_key = Some("__ggsql_layer_0__".to_string());

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        spec.facet = Some(Facet::new(FacetLayout::Wrap {
            variables: vec!["facet_var".to_string()],
        }));
        let mut data_map = HashMap::new();
        data_map.insert("__ggsql_layer_0__".to_string(), df);

        spec.layers.push(layer);

        apply_position_adjustments(&mut spec, &mut data_map).unwrap();

        // With 2 groups (X, Y), adjusted_width should be 0.45
        let adjusted = spec.layers[0].adjusted_width.unwrap();
        assert!(
            (adjusted - 0.45).abs() < 0.001,
            "adjusted_width should be 0.45 (2 groups), got {} (facet columns inflated group count)",
            adjusted
        );
    }
}
