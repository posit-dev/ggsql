//! Coordinate system resolution
//!
//! Resolves the default coordinate system by inspecting aesthetic mappings,
//! and post-scale resolution of projection properties like `radar`.

use std::collections::HashMap;

use super::coord::{Coord, CoordKind};
use super::Projection;
use crate::plot::aesthetic::{MATERIAL_AESTHETICS, POSITION_SUFFIXES};
use crate::plot::scale::ScaleTypeKind;
use crate::plot::{Mappings, ParameterValue, Scale};
use crate::GgsqlError;

/// Cartesian primary aesthetic names
const CARTESIAN_PRIMARIES: &[&str] = &["x", "y"];

/// Polar primary aesthetic names
const POLAR_PRIMARIES: &[&str] = &["angle", "radius"];

/// Resolve coordinate system for a Plot
///
/// If `project` is `Some`, returns `Ok(None)` (keep existing, no changes needed).
/// If `project` is `None`, infers coord from aesthetic mappings:
/// - x/y/xmin/xmax/ymin/ymax → Cartesian
/// - angle/radius/anglemin/... → Polar
/// - Both → Error
/// - Neither → Ok(None) (caller should use default Cartesian)
///
/// Called early in the pipeline, before AestheticContext construction.
pub fn resolve_coord(
    project: Option<&Projection>,
    global_mappings: &Mappings,
    layer_mappings: &[&Mappings],
) -> Result<Option<Projection>, String> {
    // If project is explicitly specified, keep it as-is
    if project.is_some() {
        return Ok(None);
    }

    // Collect all explicit aesthetic keys from global and layer mappings
    let mut found_cartesian = false;
    let mut found_polar = false;

    // Check global mappings
    for aesthetic in global_mappings.aesthetics.keys() {
        check_aesthetic(aesthetic, &mut found_cartesian, &mut found_polar);
    }

    // Check layer mappings
    for layer_map in layer_mappings {
        for aesthetic in layer_map.aesthetics.keys() {
            check_aesthetic(aesthetic, &mut found_cartesian, &mut found_polar);
        }
    }

    // Determine result
    if found_cartesian && found_polar {
        return Err(
            "Conflicting aesthetics: cannot use both cartesian (x/y) and polar (angle/radius) \
             aesthetics in the same plot. Use PROJECT TO cartesian or PROJECT TO polar to \
             specify the coordinate system explicitly."
                .to_string(),
        );
    }

    if found_polar {
        // Infer polar coordinate system
        let coord = Coord::from_kind(CoordKind::Polar);
        let aesthetics = coord
            .position_aesthetic_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        return Ok(Some(Projection {
            coord,
            aesthetics,
            properties: HashMap::new(),
        }));
    }

    if found_cartesian {
        // Infer cartesian coordinate system
        let coord = Coord::from_kind(CoordKind::Cartesian);
        let aesthetics = coord
            .position_aesthetic_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        return Ok(Some(Projection {
            coord,
            aesthetics,
            properties: HashMap::new(),
        }));
    }

    // Neither found - return None (caller uses default)
    Ok(None)
}

/// Check if an aesthetic name indicates cartesian or polar coordinate system.
/// Updates the found flags accordingly.
fn check_aesthetic(aesthetic: &str, found_cartesian: &mut bool, found_polar: &mut bool) {
    // Skip material aesthetics (color, size, etc.)
    if MATERIAL_AESTHETICS.contains(&aesthetic) {
        return;
    }

    // Strip position suffix if present (xmin -> x, anglemax -> angle)
    let primary = strip_position_suffix(aesthetic);

    // Check against cartesian primaries
    if CARTESIAN_PRIMARIES.contains(&primary) {
        *found_cartesian = true;
    }

    // Check against polar primaries
    if POLAR_PRIMARIES.contains(&primary) {
        *found_polar = true;
    }
}

/// Strip position suffix from an aesthetic name.
/// e.g., "xmin" -> "x", "anglemax" -> "angle", "y" -> "y"
fn strip_position_suffix(name: &str) -> &str {
    for suffix in POSITION_SUFFIXES {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

/// Resolve projection properties that depend on scale types.
///
/// Called after `resolve_scales()`. Currently handles:
/// - **`radar`** (polar only): When null (auto), sets to `true` if the theta
///   (pos2) scale is discrete/ordinal. When explicitly `true`, validates that
///   the theta scale is indeed discrete.
pub fn resolve_projection_properties(
    project: &mut Projection,
    scales: &[Scale],
) -> crate::Result<()> {
    if project.coord.coord_kind() != CoordKind::Polar {
        return Ok(());
    }

    let theta_scale = scales.iter().find(|s| s.aesthetic == "pos2");

    let theta_is_discrete = theta_scale
        .and_then(|s| s.scale_type.as_ref())
        .is_some_and(|st| {
            matches!(
                st.scale_type_kind(),
                ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal
            )
        });

    let too_few_categories = theta_scale
        .and_then(|s| s.input_range.as_ref())
        .is_some_and(|r| r.len() <= 2);

    match project.properties.get("radar") {
        Some(ParameterValue::Boolean(true)) if !theta_is_discrete => {
            return Err(GgsqlError::ValidationError(
                "SETTING radar => true requires a discrete angle scale, \
                 but the angle aesthetic is continuous"
                    .to_string(),
            ));
        }
        Some(ParameterValue::Boolean(true)) if too_few_categories => {
            return Err(GgsqlError::ValidationError(
                "SETTING radar => true requires more than 2 categories \
                 on the angle aesthetic"
                    .to_string(),
            ));
        }
        Some(ParameterValue::Boolean(_)) => {
            // Explicit true (valid) or false — keep as-is
        }
        _ => {
            // Null / absent — auto-detect: discrete with >2 categories
            let use_radar = theta_is_discrete && !too_few_categories;
            project
                .properties
                .insert("radar".to_string(), ParameterValue::Boolean(use_radar));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{AestheticValue, ArrayElement, ScaleType};

    /// Helper to create Mappings with given aesthetic names
    fn mappings_with(aesthetics: &[&str]) -> Mappings {
        let mut m = Mappings::new();
        for aes in aesthetics {
            m.insert(aes.to_string(), AestheticValue::standard_column("col"));
        }
        m
    }

    // ========================================
    // Test: Explicit project is preserved
    // ========================================

    #[test]
    fn test_resolve_keeps_explicit_project() {
        let project = Projection::cartesian();
        let global = mappings_with(&["angle", "radius"]); // Would infer polar
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(Some(&project), &global, &layers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // None means keep existing
    }

    // ========================================
    // Test: Infer Cartesian
    // ========================================

    #[test]
    fn test_infer_cartesian_from_x_y() {
        let global = mappings_with(&["x", "y"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Cartesian);
        assert_eq!(proj.aesthetics, vec!["x", "y"]);
    }

    #[test]
    fn test_infer_cartesian_from_variants() {
        let global = mappings_with(&["xmin", "ymax"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Cartesian);
    }

    #[test]
    fn test_infer_cartesian_from_layer() {
        let global = Mappings::new();
        let layer = mappings_with(&["x", "y"]);
        let layers: Vec<&Mappings> = vec![&layer];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Cartesian);
    }

    // ========================================
    // Test: Infer Polar
    // ========================================

    #[test]
    fn test_infer_polar_from_angle_radius() {
        let global = mappings_with(&["angle", "radius"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Polar);
        assert_eq!(proj.aesthetics, vec!["radius", "angle"]);
    }

    #[test]
    fn test_infer_polar_from_variants() {
        let global = mappings_with(&["anglemin", "radiusmax"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Polar);
    }

    #[test]
    fn test_infer_polar_from_layer() {
        let global = Mappings::new();
        let layer = mappings_with(&["angle", "radius"]);
        let layers: Vec<&Mappings> = vec![&layer];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Polar);
    }

    // ========================================
    // Test: Material aesthetics ignored
    // ========================================

    #[test]
    fn test_ignore_material() {
        let global = mappings_with(&["color", "size", "fill", "opacity"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Neither cartesian nor polar
    }

    #[test]
    fn test_material_with_cartesian() {
        let global = mappings_with(&["x", "y", "color", "size"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Cartesian);
    }

    // ========================================
    // Test: Conflict error
    // ========================================

    #[test]
    fn test_conflict_error() {
        let global = mappings_with(&["x", "angle"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Conflicting"));
        assert!(err.contains("cartesian"));
        assert!(err.contains("polar"));
    }

    #[test]
    fn test_conflict_across_global_and_layer() {
        let global = mappings_with(&["x", "y"]);
        let layer = mappings_with(&["angle"]);
        let layers: Vec<&Mappings> = vec![&layer];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Conflicting"));
    }

    // ========================================
    // Test: Empty returns None (default)
    // ========================================

    #[test]
    fn test_empty_returns_none() {
        let global = Mappings::new();
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    // ========================================
    // Test: Wildcard doesn't affect inference
    // ========================================

    #[test]
    fn test_wildcard_with_polar() {
        let mut global = Mappings::with_wildcard();
        global.insert("angle", AestheticValue::standard_column("cat"));
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Polar);
    }

    #[test]
    fn test_wildcard_alone_returns_none() {
        let global = Mappings::with_wildcard();
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Wildcard alone doesn't infer coord
    }

    // ========================================
    // Test: Helper functions
    // ========================================

    // ========================================
    // Test: resolve_projection_properties
    // ========================================

    fn scale_with_type(aesthetic: &str, discrete: bool) -> Scale {
        let mut s = Scale::new(aesthetic);
        s.scale_type = Some(if discrete {
            ScaleType::discrete()
        } else {
            ScaleType::continuous()
        });
        s
    }

    fn discrete_scale_with_n(aesthetic: &str, n: usize) -> Scale {
        let mut s = Scale::new(aesthetic);
        s.scale_type = Some(ScaleType::discrete());
        s.input_range = Some(
            (0..n)
                .map(|i| ArrayElement::String(format!("cat{i}")))
                .collect(),
        );
        s
    }

    #[test]
    fn test_radar_auto_true_for_discrete_theta() {
        let mut proj = Projection::polar();
        let scales = vec![discrete_scale_with_n("pos2", 5)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    #[test]
    fn test_radar_auto_false_for_continuous_theta() {
        let mut proj = Projection::polar();
        let scales = vec![scale_with_type("pos2", false)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_radar_auto_true_for_discrete_theta_no_range() {
        let mut proj = Projection::polar();
        let scales = vec![scale_with_type("pos2", true)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    #[test]
    fn test_radar_auto_false_for_discrete_theta_2_categories() {
        let mut proj = Projection::polar();
        let scales = vec![discrete_scale_with_n("pos2", 2)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_radar_auto_false_for_discrete_theta_1_category() {
        let mut proj = Projection::polar();
        let scales = vec![discrete_scale_with_n("pos2", 1)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_radar_auto_true_for_discrete_theta_3_categories() {
        let mut proj = Projection::polar();
        let scales = vec![discrete_scale_with_n("pos2", 3)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    #[test]
    fn test_radar_explicit_true_with_discrete_ok() {
        let mut proj = Projection::polar();
        proj.properties
            .insert("radar".to_string(), ParameterValue::Boolean(true));
        let scales = vec![discrete_scale_with_n("pos2", 4)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    #[test]
    fn test_radar_explicit_true_with_2_categories_errors() {
        let mut proj = Projection::polar();
        proj.properties
            .insert("radar".to_string(), ParameterValue::Boolean(true));
        let scales = vec![discrete_scale_with_n("pos2", 2)];
        let result = resolve_projection_properties(&mut proj, &scales);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("more than 2"),
            "error should mention 'more than 2': {err}"
        );
    }

    #[test]
    fn test_radar_explicit_true_with_continuous_errors() {
        let mut proj = Projection::polar();
        proj.properties
            .insert("radar".to_string(), ParameterValue::Boolean(true));
        let scales = vec![scale_with_type("pos2", false)];
        let result = resolve_projection_properties(&mut proj, &scales);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("discrete"),
            "error should mention discrete: {err}"
        );
    }

    #[test]
    fn test_radar_explicit_false_with_discrete_preserved() {
        let mut proj = Projection::polar();
        proj.properties
            .insert("radar".to_string(), ParameterValue::Boolean(false));
        let scales = vec![scale_with_type("pos2", true)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert_eq!(
            proj.properties.get("radar"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    #[test]
    fn test_radar_noop_for_cartesian() {
        let mut proj = Projection::cartesian();
        let scales = vec![scale_with_type("pos2", true)];
        resolve_projection_properties(&mut proj, &scales).unwrap();
        assert!(!proj.properties.contains_key("radar"));
    }

    #[test]
    fn test_strip_position_suffix() {
        assert_eq!(strip_position_suffix("x"), "x");
        assert_eq!(strip_position_suffix("y"), "y");
        assert_eq!(strip_position_suffix("xmin"), "x");
        assert_eq!(strip_position_suffix("xmax"), "x");
        assert_eq!(strip_position_suffix("xend"), "x");
        assert_eq!(strip_position_suffix("ymin"), "y");
        assert_eq!(strip_position_suffix("ymax"), "y");
        assert_eq!(strip_position_suffix("angle"), "angle");
        assert_eq!(strip_position_suffix("anglemin"), "angle");
        assert_eq!(strip_position_suffix("radiusmax"), "radius");
    }
}
