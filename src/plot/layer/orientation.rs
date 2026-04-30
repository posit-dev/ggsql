//! Layer orientation detection and mapping flipping.
//!
//! This module provides orientation detection for geoms with implicit orientation
//! (bar, histogram, boxplot, violin, density, ribbon) and handles flipping position
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
use crate::plot::aesthetic::{is_position_aesthetic, AestheticContext};
use crate::plot::scale::ScaleTypeKind;
use crate::plot::{AestheticValue, Mappings, Scale};
use crate::{naming, DataFrame};

/// Orientation value for aligned/vertical orientation.
pub const ALIGNED: &str = "aligned";

/// Orientation value for transposed/horizontal orientation.
pub const TRANSPOSED: &str = "transposed";

/// Valid orientation values for constraint validation.
pub const ORIENTATION_VALUES: &[&str] = &[ALIGNED, TRANSPOSED];

/// Determine effective orientation for a layer.
///
/// Returns explicit orientation if set via SETTING, otherwise auto-detects
/// from scales for geoms with implicit orientation.
/// Geoms without implicit orientation return "aligned" unless explicitly set.
pub fn resolve_orientation(layer: &Layer, scales: &[Scale]) -> &'static str {
    // Check for explicit orientation setting first
    if let Some(orientation) = layer.parameters.get("orientation").and_then(|v| v.as_str()) {
        return if orientation == TRANSPOSED {
            TRANSPOSED
        } else {
            ALIGNED
        };
    }

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
/// Reads the orientation from the layer's parameters, which must have been
/// set by `resolve_orientations()` during execution.
pub fn is_transposed(layer: &Layer) -> bool {
    layer
        .parameters
        .get("orientation")
        .and_then(|v| v.as_str())
        .map(|s| s == TRANSPOSED)
        .unwrap_or(false)
}

/// Check if a geom type supports orientation auto-detection.
///
/// Returns true for geoms with inherent orientation assumptions:
/// - Bar, Histogram, Boxplot, Violin, Density, Ribbon, Rule, Range
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
            | GeomType::Rule
            | GeomType::Range
    )
}

/// Detect orientation from scales, mappings, and remappings.
///
/// Applies unified rules in order:
///
/// 0. **Remapping without mapping**: If no position mappings exist but remappings
///    target a position axis, the remapping target is the value axis:
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
    // Check for position mappings
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
    // Only apply when there are explicit position mappings; otherwise the user
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

/// Check if a scale is discrete for orientation purposes.
///
/// Includes categorical, ordinal, and binned scales - all represent
/// discrete categories rather than continuous values.
fn is_discrete_scale(scale: &Scale) -> bool {
    scale.scale_type.as_ref().is_some_and(|st| {
        matches!(
            st.scale_type_kind(),
            ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal | ScaleTypeKind::Binned
        )
    })
}

/// Swap position aesthetic pairs in an aesthetics map.
///
/// Swaps the following pairs:
/// - pos1 ↔ pos2
/// - pos1min ↔ pos2min
/// - pos1max ↔ pos2max
/// - pos1end ↔ pos2end
/// - pos1offset ↔ pos2offset
///
/// Used for both mappings and remappings when handling transposed orientation.
pub fn flip_position_aesthetics(
    aesthetics: &mut std::collections::HashMap<String, AestheticValue>,
) {
    const PAIRS: [(&str, &str); 5] = [
        ("pos1", "pos2"),
        ("pos1min", "pos2min"),
        ("pos1max", "pos2max"),
        ("pos1end", "pos2end"),
        ("pos1offset", "pos2offset"),
    ];

    for (a, b) in PAIRS {
        let val_a = aesthetics.remove(a);
        let val_b = aesthetics.remove(b);

        if let Some(v) = val_a {
            aesthetics.insert(b.to_string(), v);
        }
        if let Some(v) = val_b {
            aesthetics.insert(a.to_string(), v);
        }
    }
}

/// Flip position column names in a DataFrame for Transposed orientation layers.
///
/// Swaps column names like `__ggsql_aes_pos1__` ↔ `__ggsql_aes_pos2__` so that
/// the data matches the flipped mapping names.
///
/// This is called after query execution for layers with Transposed orientation,
/// in coordination with `normalize_mapping_column_names` which updates the mappings.
pub fn flip_dataframe_position_columns(
    df: DataFrame,
    aesthetic_ctx: &AestheticContext,
) -> DataFrame {
    // Collect renames needed
    let renames: Vec<(String, String)> = df
        .get_column_names()
        .iter()
        .filter_map(|col_name| {
            naming::extract_aesthetic_name(col_name).and_then(|aesthetic| {
                if is_position_aesthetic(aesthetic) {
                    let flipped = aesthetic_ctx.flip_position(aesthetic);
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

    let mut result = df;

    // First pass: rename to temp names
    for (from, to) in &renames {
        let temp = format!("{}_temp", to);
        result = result.rename(from, &temp).expect("rename should not fail");
    }

    // Second pass: remove temp suffix
    for (_, to) in &renames {
        let temp = format!("{}_temp", to);
        result = result.rename(&temp, to).expect("rename should not fail");
    }

    result
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
        assert!(geom_has_implicit_orientation(&GeomType::Rule));

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
        use crate::plot::ParameterValue;

        // Helper function should return true for transposed orientation
        let mut layer = Layer::new(Geom::histogram());
        layer
            .mappings
            .insert("pos2", AestheticValue::standard_column("y_col"));
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        // Resolve and store orientation (as done by resolve_orientations)
        let orientation = resolve_orientation(&layer, &scales);
        layer.parameters.insert(
            "orientation".to_string(),
            ParameterValue::String(orientation.to_string()),
        );
        assert!(is_transposed(&layer));

        // Should return false for aligned orientation
        let mut layer2 = Layer::new(Geom::histogram());
        let mut scale2 = Scale::new("pos1");
        scale2.scale_type = Some(ScaleType::continuous());
        let scales2 = vec![scale2];

        let orientation2 = resolve_orientation(&layer2, &scales2);
        layer2.parameters.insert(
            "orientation".to_string(),
            ParameterValue::String(orientation2.to_string()),
        );
        assert!(!is_transposed(&layer2));
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

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
    }

    #[test]
    fn test_resolve_orientation_scale_only_no_flip() {
        // Scale specification without position mapping shouldn't flip orientation
        // Real-world: `VISUALISE FROM data DRAW bar SCALE y SETTING expand => [...]`
        // The bar stat will produce pos1=category, pos2=count → should stay Aligned
        let layer = Layer::new(Geom::bar());
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        // Without position mappings, scale existence doesn't imply orientation
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

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
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
    fn test_flip_position_aesthetics() {
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

        flip_position_aesthetics(&mut layer.mappings.aesthetics);

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
    fn test_flip_position_aesthetics_empty() {
        let mut layer = Layer::new(Geom::point());
        // No crash with empty mappings
        flip_position_aesthetics(&mut layer.mappings.aesthetics);
        assert!(layer.mappings.aesthetics.is_empty());
    }

    #[test]
    fn test_flip_position_aesthetics_partial() {
        let mut layer = Layer::new(Geom::bar());
        // Only pos1 mapped
        layer.mappings.insert(
            "pos1".to_string(),
            AestheticValue::standard_column("x".to_string()),
        );

        flip_position_aesthetics(&mut layer.mappings.aesthetics);

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

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
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

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
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

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
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
        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
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
        // The remapping rule only applies when NO position mappings exist
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

    #[test]
    fn test_resolve_orientation_rule_vertical() {
        // Rule with pos1 scale → Aligned (vertical rule)
        // Real-world: `DRAW rule MAPPING 2.5 AS pos1` with `SCALE CONTINUOUS pos1`
        let mut layer = Layer::new(Geom::rule());
        layer
            .mappings
            .insert("pos1", AestheticValue::standard_column("x_val"));
        let mut scale = Scale::new("pos1");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }

    #[test]
    fn test_resolve_orientation_rule_horizontal() {
        // Rule with pos2 scale → Transposed (horizontal rule)
        // Real-world: `DRAW rule MAPPING 15 AS pos1` with `SCALE CONTINUOUS pos2`
        let mut layer = Layer::new(Geom::rule());
        layer
            .mappings
            .insert("pos1", AestheticValue::standard_column("y_val"));
        let mut scale = Scale::new("pos2");
        scale.scale_type = Some(ScaleType::continuous());
        let scales = vec![scale];

        assert_eq!(resolve_orientation(&layer, &scales), TRANSPOSED);
    }

    #[test]
    fn test_resolve_orientation_rule_default() {
        // Rule with no scales → defaults to Aligned
        let layer = Layer::new(Geom::rule());
        let scales = vec![];

        assert_eq!(resolve_orientation(&layer, &scales), ALIGNED);
    }
}
