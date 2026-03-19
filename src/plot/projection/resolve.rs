//! Coordinate system resolution
//!
//! Resolves the default coordinate system by inspecting aesthetic mappings.

use std::collections::HashMap;

use super::coord::{Coord, CoordKind};
use super::Projection;
use crate::plot::aesthetic::{MATERIAL_AESTHETICS, POSITIONAL_SUFFIXES};
use crate::plot::Mappings;

/// Cartesian primary aesthetic names
const CARTESIAN_PRIMARIES: &[&str] = &["x", "y"];

/// Polar primary aesthetic names
const POLAR_PRIMARIES: &[&str] = &["theta", "radius"];

/// Resolve coordinate system for a Plot
///
/// If `project` is `Some`, returns `Ok(None)` (keep existing, no changes needed).
/// If `project` is `None`, infers coord from aesthetic mappings:
/// - x/y/xmin/xmax/ymin/ymax → Cartesian
/// - theta/radius/thetamin/... → Polar
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
            "Conflicting aesthetics: cannot use both cartesian (x/y) and polar (theta/radius) \
             aesthetics in the same plot. Use PROJECT TO cartesian or PROJECT TO polar to \
             specify the coordinate system explicitly."
                .to_string(),
        );
    }

    if found_polar {
        // Infer polar coordinate system
        let coord = Coord::from_kind(CoordKind::Polar);
        let aesthetics = coord
            .positional_aesthetic_names()
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
            .positional_aesthetic_names()
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

    // Strip positional suffix if present (xmin -> x, thetamax -> theta)
    let primary = strip_positional_suffix(aesthetic);

    // Check against cartesian primaries
    if CARTESIAN_PRIMARIES.contains(&primary) {
        *found_cartesian = true;
    }

    // Check against polar primaries
    if POLAR_PRIMARIES.contains(&primary) {
        *found_polar = true;
    }
}

/// Strip positional suffix from an aesthetic name.
/// e.g., "xmin" -> "x", "thetamax" -> "theta", "y" -> "y"
fn strip_positional_suffix(name: &str) -> &str {
    for suffix in POSITIONAL_SUFFIXES {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::AestheticValue;

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
        let project = Projection {
            coord: Coord::cartesian(),
            aesthetics: vec!["x".to_string(), "y".to_string()],
            properties: HashMap::new(),
        };
        let global = mappings_with(&["theta", "radius"]); // Would infer polar
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
    fn test_infer_polar_from_theta_radius() {
        let global = mappings_with(&["theta", "radius"]);
        let layers: Vec<&Mappings> = vec![];

        let result = resolve_coord(None, &global, &layers);
        assert!(result.is_ok());
        let inferred = result.unwrap();
        assert!(inferred.is_some());
        let proj = inferred.unwrap();
        assert_eq!(proj.coord.coord_kind(), CoordKind::Polar);
        assert_eq!(proj.aesthetics, vec!["radius", "theta"]);
    }

    #[test]
    fn test_infer_polar_from_variants() {
        let global = mappings_with(&["thetamin", "radiusmax"]);
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
        let layer = mappings_with(&["theta", "radius"]);
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
        let global = mappings_with(&["x", "theta"]);
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
        let layer = mappings_with(&["theta"]);
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
        global.insert("theta", AestheticValue::standard_column("cat"));
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

    #[test]
    fn test_strip_positional_suffix() {
        assert_eq!(strip_positional_suffix("x"), "x");
        assert_eq!(strip_positional_suffix("y"), "y");
        assert_eq!(strip_positional_suffix("xmin"), "x");
        assert_eq!(strip_positional_suffix("xmax"), "x");
        assert_eq!(strip_positional_suffix("xend"), "x");
        assert_eq!(strip_positional_suffix("ymin"), "y");
        assert_eq!(strip_positional_suffix("ymax"), "y");
        assert_eq!(strip_positional_suffix("theta"), "theta");
        assert_eq!(strip_positional_suffix("thetamin"), "theta");
        assert_eq!(strip_positional_suffix("radiusmax"), "radius");
    }
}
