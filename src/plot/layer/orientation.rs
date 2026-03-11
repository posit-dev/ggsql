//! Layer orientation detection and mapping flipping.
//!
//! This module provides orientation detection for geoms with implicit orientation
//! (bar, histogram, boxplot, violin, density, ribbon) and handles flipping positional
//! aesthetic mappings before stat computation.
//!
//! # Orientation
//!
//! Some geoms have a "main axis" (categorical/domain axis) and a "value axis":
//! - Bar: main axis = categories, value axis = bar height
//! - Histogram: main axis = bins, value axis = count
//! - Boxplot: main axis = groups, value axis = distribution
//! - Ribbon: main axis = domain (e.g., time), value axis = range (min/max)
//!
//! Orientation describes how the layer's main axis aligns with the coordinate's
//! primary axis (pos1):
//! - **"aligned"**: main axis = pos1 (vertical bars, x-axis bins)
//! - **"transposed"**: main axis = pos2 (horizontal bars, y-axis bins)
//!
//! # Auto-Detection
//!
//! Orientation is auto-detected from scale types:
//! - For two-axis geoms (bar, boxplot): if pos1 is continuous and pos2 is discrete → "transposed"
//! - For single-axis geoms (histogram, density): if pos2 has a scale but pos1 doesn't → "transposed"

use super::geom::GeomType;
use super::Layer;
use crate::plot::scale::ScaleTypeKind;
use crate::plot::{Mappings, Scale};

/// Orientation value for aligned/vertical orientation.
pub const ALIGNED: &str = "aligned";

/// Orientation value for transposed/horizontal orientation.
pub const TRANSPOSED: &str = "transposed";

/// Determine effective orientation for a layer.
///
/// Auto-detects orientation from scales for geoms with implicit orientation.
/// Geoms without implicit orientation always return "aligned".
pub fn resolve_orientation(layer: &Layer, scales: &[Scale]) -> &'static str {
    // Only auto-detect for geoms with implicit orientation
    if !geom_has_implicit_orientation(&layer.geom.geom_type()) {
        return ALIGNED;
    }

    detect_from_scales(
        scales,
        &layer.geom.geom_type(),
        &layer.mappings,
        &layer.remappings,
    )
}

/// Check if a layer is transposed (horizontal orientation).
///
/// Convenience helper for downstream code that needs to check orientation.
pub fn is_transposed(layer: &Layer, scales: &[Scale]) -> bool {
    resolve_orientation(layer, scales) == TRANSPOSED
}

/// Check if a geom type supports orientation auto-detection.
///
/// Returns true for geoms with inherent orientation assumptions:
/// - Bar, Histogram, Boxplot, Violin, Density
///
/// Returns false for geoms without inherent orientation:
/// - Point, Line, Path, Area, etc.
pub fn geom_has_implicit_orientation(geom: &GeomType) -> bool {
    matches!(
        geom,
        GeomType::Bar
            | GeomType::Histogram
            | GeomType::Boxplot
            | GeomType::Violin
            | GeomType::Density
            | GeomType::Ribbon
    )
}

/// Detect orientation from scales, mappings, and remappings.
///
/// Applies unified rules in order:
///
/// 0. **Remapping without mapping**: If no positional mappings exist but remappings
///    target a positional axis, the remapping target is the value axis:
///    - Remapping to pos1 only → Transposed (pos1 is value axis, main axis must be pos2)
///    - Remapping to pos2 only → Aligned (pos2 is value axis, main axis is pos1)
///
/// 1. **Single scale present**: The present scale defines the primary axis
///    - Only pos1 → Primary
///    - Only pos2 → Secondary
///
/// 2. **Both continuous**: The axis with range mappings is secondary (value axis)
///    - pos1 has range mappings → Secondary
///    - pos2 has range mappings (or neither) → Primary (default)
///
/// 3. **Mixed types**: The discrete scale is the primary (domain) axis
///    - pos1 discrete, pos2 continuous → Primary
///    - pos1 continuous, pos2 discrete → Secondary
///
/// 4. **Default**: Primary
fn detect_from_scales(
    scales: &[Scale],
    _geom: &GeomType,
    mappings: &Mappings,
    remappings: &Mappings,
) -> &'static str {
    // Check for positional mappings
    let has_pos1_mapping = mappings.contains_key("pos1");
    let has_pos2_mapping = mappings.contains_key("pos2");

    // Rule 0: Remapping without mapping - remapping target is the value axis
    if !has_pos1_mapping && !has_pos2_mapping {
        let has_pos1_remapping = remappings.contains_key("pos1");
        let has_pos2_remapping = remappings.contains_key("pos2");

        if has_pos1_remapping && !has_pos2_remapping {
            return TRANSPOSED;
        }
        if has_pos2_remapping && !has_pos1_remapping {
            return ALIGNED;
        }
    }

    let pos1_scale = scales.iter().find(|s| s.aesthetic == "pos1");
    let pos2_scale = scales.iter().find(|s| s.aesthetic == "pos2");

    let has_pos1 = pos1_scale.is_some();
    let has_pos2 = pos2_scale.is_some();

    // Rule 1: Single scale present - that axis is primary
    // Only apply when there are explicit positional mappings; otherwise the user
    // is just customizing a scale (e.g., SCALE y SETTING expand) without intending
    // to change orientation. The geom's default_remappings will define orientation.
    if has_pos1_mapping || has_pos2_mapping {
        if has_pos2 && !has_pos1 {
            return TRANSPOSED;
        }
        if has_pos1 && !has_pos2 {
            return ALIGNED;
        }
    }

    // Both scales present
    let pos1_continuous = pos1_scale.is_some_and(is_continuous_scale);
    let pos2_continuous = pos2_scale.is_some_and(is_continuous_scale);

    // Rule 2: Both continuous - range mapping axis is secondary
    // Range mappings include min/max pairs and primary/end pairs
    if pos1_continuous && pos2_continuous {
        let has_pos1_range = mappings.contains_key("pos1min")
            || mappings.contains_key("pos1max")
            || mappings.contains_key("pos1end");
        let has_pos2_range = mappings.contains_key("pos2min")
            || mappings.contains_key("pos2max")
            || mappings.contains_key("pos2end");

        if has_pos1_range && !has_pos2_range {
            return TRANSPOSED;
        }
        return ALIGNED;
    }

    // Rule 3: Mixed types - discrete axis is primary
    let pos1_discrete = pos1_scale.is_some_and(is_discrete_scale);
    let pos2_discrete = pos2_scale.is_some_and(is_discrete_scale);

    if pos1_continuous && pos2_discrete {
        return TRANSPOSED;
    }
    if pos1_discrete && pos2_continuous {
        return ALIGNED;
    }

    // Default
    ALIGNED
}

/// Check if a scale is continuous (numeric/temporal).
fn is_continuous_scale(scale: &Scale) -> bool {
    scale
        .scale_type
        .as_ref()
        .is_some_and(|st| st.scale_type_kind() == ScaleTypeKind::Continuous)
}

/// Check if a scale is discrete (categorical/ordinal).
fn is_discrete_scale(scale: &Scale) -> bool {
    scale.scale_type.as_ref().is_some_and(|st| {
        matches!(
            st.scale_type_kind(),
            ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal
        )
    })
}

/// Swap positional aesthetic pairs in layer mappings.
///
/// Swaps the following pairs:
/// - pos1 ↔ pos2
/// - pos1min ↔ pos2min
/// - pos1max ↔ pos2max
/// - pos1end ↔ pos2end
/// - pos1offset ↔ pos2offset
///
/// This is called before stat transforms for Secondary orientation layers,
/// so stats always see "standard" orientation. After stat transforms,
/// this is called again to flip back to the correct output positions.
pub fn flip_mappings(layer: &mut Layer) {
    let pairs = [
        ("pos1", "pos2"),
        ("pos1min", "pos2min"),
        ("pos1max", "pos2max"),
        ("pos1end", "pos2end"),
        ("pos1offset", "pos2offset"),
    ];

    for (a, b) in pairs {
        let val_a = layer.mappings.aesthetics.remove(a);
        let val_b = layer.mappings.aesthetics.remove(b);

        if let Some(v) = val_a {
            layer.mappings.aesthetics.insert(b.to_string(), v);
        }
        if let Some(v) = val_b {
            layer.mappings.aesthetics.insert(a.to_string(), v);
        }
    }
}

/// Swap positional aesthetic pairs in remappings.
///
/// Same as flip_mappings but for remappings (stat output mappings).
pub fn flip_remappings(layer: &mut Layer) {
    let pairs = [
        ("pos1", "pos2"),
        ("pos1min", "pos2min"),
        ("pos1max", "pos2max"),
        ("pos1end", "pos2end"),
        ("pos1offset", "pos2offset"),
    ];

    for (a, b) in pairs {
        let val_a = layer.remappings.aesthetics.remove(a);
        let val_b = layer.remappings.aesthetics.remove(b);

        if let Some(v) = val_a {
            layer.remappings.aesthetics.insert(b.to_string(), v);
        }
        if let Some(v) = val_b {
            layer.remappings.aesthetics.insert(a.to_string(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{AestheticValue, Geom, ScaleType};

    #[test]
    fn test_orientation_constants() {
        assert_eq!(ALIGNED, "aligned");
        assert_eq!(TRANSPOSED, "transposed");
    }

    #[test]
    fn test_geom_has_implicit_orientation() {
        assert!(geom_has_implicit_orientation(&GeomType::Bar));
        assert!(geom_has_implicit_orientation(&GeomType::Histogram));
        assert!(geom_has_implicit_orientation(&GeomType::Boxplot));
        assert!(geom_has_implicit_orientation(&GeomType::Violin));
        assert!(geom_has_implicit_orientation(&GeomType::Density));
        assert!(geom_has_implicit_orientation(&GeomType::Ribbon));

        assert!(!geom_has_implicit_orientation(&GeomType::Point));
        assert!(!geom_has_implicit_orientation(&GeomType::Line));
        assert!(!geom_has_implicit_orientation(&GeomType::Path));
        assert!(!geom_has_implicit_orientation(&GeomType::Area));
    }

    #[test]
    fn test_resolve_orientation_no_implicit() {
        // Point geom has no implicit orientation
        let layer = Layer::new(Geom::point());
        let scales = vec![];
        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_is_transposed_helper() {
        // Helper function should return true for transposed orientation
        let mut layer = Layer::new(Geom::histogram());
        layer
            .mappings
            .insert("pos2", AestheticValue::standard_column("y_col"));
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        assert!(is_transposed(&layer, &scales));

        // Should return false for aligned orientation
        let layer2 = Layer::new(Geom::histogram());
        let mut scale2 = Scale::new("pos1");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales2 = vec![scale2];

        assert!(!is_transposed(&layer2, &scales2));
    }

    #[test]
    fn test_resolve_orientation_histogram_default() {
        // Histogram with pos1 scale → Aligned
        let layer = Layer::new(Geom::histogram());
        let mut scale = Scale::new("pos1");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_histogram_horizontal() {
        // Histogram with pos2 mapping (y binned) → Transposed
        // Real-world: `VISUALISE y AS y DRAW histogram` or `MAPPING y AS pos2`
        let mut layer = Layer::new(Geom::histogram());
        layer
            .mappings
            .insert("pos2", AestheticValue::standard_column("y_col"));
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_scale_only_no_flip() {
        // Scale specification without positional mapping shouldn't flip orientation
        // Real-world: `VISUALISE FROM data DRAW bar SCALE y SETTING expand => [...]`
        // The bar stat will produce pos1=category, pos2=count → should stay Aligned
        let layer = Layer::new(Geom::bar());
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        // Without positional mappings, scale existence doesn't imply orientation
        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_bar_horizontal() {
        // Bar with pos1 continuous, pos2 discrete → Transposed
        let layer = Layer::new(Geom::bar());
        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::discrete());
        let scales = vec![scale1, scale2];

        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_bar_vertical() {
        // Bar with pos1 discrete, pos2 continuous → Aligned
        let layer = Layer::new(Geom::bar());
        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::discrete());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_flip_mappings() {
        let mut layer = Layer::new(Geom::bar());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("category".to_string()),
        );
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("value".to_string()),
        );
        layer.mappings.insert(
            "pos1end".to_string(),
            AestheticValue::standard_column("x2".to_string()),
        );

        flip_mappings(&mut layer);

        // pos1 ↔ pos2
        assert_eq!(
            layer.mappings.get("pos2").unwrap().column_name(),
            Some("category")
        );
        assert_eq!(
            layer.mappings.get("pos1").unwrap().column_name(),
            Some("value")
        );
        // pos1end → pos2end
        assert_eq!(
            layer.mappings.get("pos2end").unwrap().column_name(),
            Some("x2")
        );
        assert!(layer.mappings.get("pos1end").is_none());
    }

    #[test]
    fn test_flip_mappings_empty() {
        let mut layer = Layer::new(Geom::point());
        // No crash with empty mappings
        flip_mappings(&mut layer);
        assert!(layer.mappings.aesthetics.is_empty());
    }

    #[test]
    fn test_flip_mappings_partial() {
        let mut layer = Layer::new(Geom::bar());
        // Only pos1 mapped
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );

        flip_mappings(&mut layer);

        // pos1 moves to pos2
        assert!(layer.mappings.get("pos1").is_none());
        assert_eq!(layer.mappings.get("pos2").unwrap().column_name(), Some("x"));
    }

    #[test]
    fn test_resolve_orientation_ribbon_both_continuous_pos2_range() {
        // Ribbon with both continuous scales and pos2 range → Aligned
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );
        layer.mappings.insert(
            "pos2min".to_string(),
            AestheticValue::standard_column("ymin".to_string()),
        );
        layer.mappings.insert(
            "pos2max".to_string(),
            AestheticValue::standard_column("ymax".to_string()),
        );

        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_ribbon_both_continuous_pos1_range() {
        // Ribbon with both continuous scales and pos1 range → Secondary
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("y".to_string()),
        );
        layer.mappings.insert(
            "pos1min".to_string(),
            AestheticValue::standard_column("xmin".to_string()),
        );
        layer.mappings.insert(
            "pos1max".to_string(),
            AestheticValue::standard_column("xmax".to_string()),
        );

        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_ribbon_pos1_continuous_pos2_discrete() {
        // Ribbon with pos1 continuous, pos2 discrete → Secondary
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("value".to_string()),
        );
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("category".to_string()),
        );

        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::discrete());
        let scales = vec![scale1, scale2];

        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_ribbon_pos1_discrete_pos2_continuous() {
        // Ribbon with pos1 discrete, pos2 continuous → Primary
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("category".to_string()),
        );
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("value".to_string()),
        );

        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::discrete());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_ribbon_pos1_range_with_scales() {
        // Ribbon with pos2 mapping and pos1 range (xmin/xmax) with continuous scales → Transposed
        // This covers: DRAW ribbon MAPPING Date AS y, Temp AS xmax, 0.0 AS xmin
        // Rule 2: Both continuous, pos1 has range → Transposed
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("Date".to_string()),
        );
        layer.mappings.insert(
            "pos1min".to_string(),
            AestheticValue::Literal(crate::plot::ParameterValue::Number(0.0)),
        );
        layer.mappings.insert(
            "pos1max".to_string(),
            AestheticValue::standard_column("Temp".to_string()),
        );

        // Scales are created and typed by execute pipeline
        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_ribbon_pos2_range_with_scales() {
        // Ribbon with pos1 mapping and pos2 range (ymin/ymax) with continuous scales → Aligned
        // This covers: DRAW ribbon MAPPING Date AS x, Temp AS ymax, 0.0 AS ymin
        // Rule 2: Both continuous, pos2 has range (or neither) → Aligned
        let mut layer = Layer::new(Geom::ribbon());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("Date".to_string()),
        );
        layer.mappings.insert(
            "pos2min".to_string(),
            AestheticValue::Literal(crate::plot::ParameterValue::Number(0.0)),
        );
        layer.mappings.insert(
            "pos2max".to_string(),
            AestheticValue::standard_column("Temp".to_string()),
        );

        // Scales are created and typed by execute pipeline
        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::continuous());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_remapping_to_pos1() {
        // Bar with no mappings but remapping to pos1 → Transposed
        // This covers: VISUALISE FROM data DRAW bar REMAPPING proportion AS x
        let mut layer = Layer::new(Geom::bar());
        layer.remappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("proportion".to_string()),
        );

        let scales = vec![];
        assert_eq!(
            resolve_orientation(&layer, &scales),
            TRANSPOSED
        );
    }

    #[test]
    fn test_resolve_orientation_remapping_to_pos2() {
        // Bar with no mappings but remapping to pos2 → Aligned (default)
        let mut layer = Layer::new(Geom::bar());
        layer.remappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("count".to_string()),
        );

        let scales = vec![];
        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_remapping_both_axes() {
        // Bar with remappings to both axes → falls through to default (Aligned)
        let mut layer = Layer::new(Geom::bar());
        layer.remappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x_val".to_string()),
        );
        layer.remappings.insert(
            "pos2".to_string(),
            AestheticValue::standard_column("y_val".to_string()),
        );

        let scales = vec![];
        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_mapping_overrides_remapping() {
        // Bar with pos1 mapping AND pos1 remapping → mapping takes precedence
        // The remapping rule only applies when NO positional mappings exist
        let mut layer = Layer::new(Geom::bar());
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("category".to_string()),
        );
        layer.remappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("proportion".to_string()),
        );

        // With pos1 discrete scale → Aligned (normal rule 3)
        let mut scale1 = Scale::new("pos1");
        scale1.scale_type = Some(ScaleType::discrete());
        let mut scale2 = Scale::new("pos2");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale1, scale2];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }
}
