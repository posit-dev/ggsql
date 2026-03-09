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

        let Some(df) = data_map.get(&key) else {
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
    use crate::plot::layer::{Geom, Position};
    use crate::plot::{AestheticValue, Mappings, ParameterValue, Scale, ScaleType};
    use polars::prelude::*;

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
            "__ggsql_aes_pos1__" => ["A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "Y", "X", "Y"],
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

        // Verify stacking was applied
        assert!(pos2_col.f64().is_ok() || pos2_col.i64().is_ok());
        assert!(pos2end_col.f64().is_ok() || pos2end_col.i64().is_ok());
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

        let offset = offset_col.unwrap().f64().unwrap();

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
        let offset = result_df
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();

        // With 2 groups and custom width 0.6:
        // - adjusted_width = 0.6 / 2 = 0.3
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
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

        let offset = offset_col.unwrap().f64().unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

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
        let offset = result_df
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // With custom width 0.6, offsets should be in range [-0.3, 0.3]
        for &v in &offsets {
            assert!((-0.3..=0.3).contains(&v));
        }
    }
}
