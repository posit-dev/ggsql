//! Aesthetic classification and validation utilities
//!
//! This module provides centralized functions and constants for working with
//! aesthetic names in ggsql. Aesthetics are visual properties that can be mapped
//! to data columns or set to literal values.
//!
//! # Positional vs Legend Aesthetics
//!
//! Aesthetics fall into two categories:
//! - **Positional**: Map to axes (x, y, and variants like xmin, xmax, etc.)
//! - **Legend**: Map to visual properties shown in legends (color, size, shape, etc.)
//!
//! # Aesthetic Families
//!
//! Some aesthetics belong to "families" where variants map to a primary aesthetic.
//! For example, `xmin`, `xmax`, and `xend` all belong to the "x" family.
//! This is used for scale resolution and label computation.

/// Primary positional aesthetics (x and y only)
pub const PRIMARY_POSITIONAL: &[&str] = &["x", "y"];

/// All positional aesthetics (primary + variants)
pub const ALL_POSITIONAL: &[&str] = &["x", "xmin", "xmax", "xend", "y", "ymin", "ymax", "yend"];

/// Maps variant aesthetics to their primary aesthetic family.
///
/// For example, `xmin`, `xmax`, and `xend` all belong to the "x" family.
/// When computing labels, all family members can contribute to the primary aesthetic's label,
/// with the first aesthetic encountered in a family setting the label.
pub const AESTHETIC_FAMILIES: &[(&str, &str)] = &[
    ("xmin", "x"),
    ("xmax", "x"),
    ("xend", "x"),
    ("ymin", "y"),
    ("ymax", "y"),
    ("yend", "y"),
];

/// Non-positional aesthetics (visual properties shown in legends or applied to marks)
///
/// These include:
/// - Color aesthetics: color, colour, fill, stroke, opacity
/// - Size/shape aesthetics: size, shape, linetype, linewidth
/// - Dimension aesthetics: width, height
/// - Text aesthetics: label, family, fontface, hjust, vjust
pub const NON_POSITIONAL: &[&str] = &[
    "color",
    "colour",
    "fill",
    "stroke",
    "opacity",
    "size",
    "shape",
    "linetype",
    "linewidth",
    "width",
    "height",
    "label",
    "family",
    "fontface",
    "hjust",
    "vjust",
];

/// Check if aesthetic is primary positional (x or y only)
#[inline]
pub fn is_primary_positional(aesthetic: &str) -> bool {
    PRIMARY_POSITIONAL.contains(&aesthetic)
}

/// Check if aesthetic is positional (maps to axis, not legend)
///
/// Positional aesthetics include x, y, and their variants (xmin, xmax, ymin, ymax, xend, yend).
/// These aesthetics map to axis positions rather than legend entries.
#[inline]
pub fn is_positional_aesthetic(name: &str) -> bool {
    ALL_POSITIONAL.contains(&name)
}

/// Check if name is a recognized aesthetic
///
/// This includes all positional aesthetics plus visual aesthetics like color, size, shape, etc.
#[inline]
pub fn is_aesthetic_name(name: &str) -> bool {
    is_positional_aesthetic(name) || NON_POSITIONAL.contains(&name)
}

/// Get the primary aesthetic for a given aesthetic name.
///
/// Returns the primary family aesthetic if the input is a variant (e.g., "xmin" -> "x"),
/// or returns the aesthetic itself if it's already primary (e.g., "x" -> "x", "fill" -> "fill").
#[inline]
pub fn primary_aesthetic(aesthetic: &str) -> &str {
    AESTHETIC_FAMILIES
        .iter()
        .find(|(variant, _)| *variant == aesthetic)
        .map(|(_, primary)| *primary)
        .unwrap_or(aesthetic)
}

/// Get all aesthetics in the same family as the given aesthetic.
///
/// For primary aesthetics like "x", returns all family members: `["x", "xmin", "xmax", "x2", "xend"]`.
/// For variant aesthetics like "xmin", returns just `["xmin"]` since scales should be
/// defined for primary aesthetics.
/// For non-family aesthetics like "color", returns just `["color"]`.
///
/// This is used by scale resolution to find all columns that contribute to a scale's
/// input range (e.g., both `ymin` and `ymax` columns contribute to the "y" scale).
pub fn get_aesthetic_family(aesthetic: &str) -> Vec<&str> {
    // First, determine the primary aesthetic
    let primary = primary_aesthetic(aesthetic);

    // If aesthetic is not a primary (it's a variant), just return the aesthetic itself
    // since scales should be defined for primary aesthetics
    if primary != aesthetic {
        return vec![aesthetic];
    }

    // Collect primary + all variants that map to this primary
    let mut family = vec![primary];
    for (variant, prim) in AESTHETIC_FAMILIES {
        if *prim == primary {
            family.push(*variant);
        }
    }

    family
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_positional() {
        assert!(is_primary_positional("x"));
        assert!(is_primary_positional("y"));
        assert!(!is_primary_positional("xmin"));
        assert!(!is_primary_positional("color"));
    }

    #[test]
    fn test_positional_aesthetic() {
        // Primary
        assert!(is_positional_aesthetic("x"));
        assert!(is_positional_aesthetic("y"));

        // Variants
        assert!(is_positional_aesthetic("xmin"));
        assert!(is_positional_aesthetic("xmax"));
        assert!(is_positional_aesthetic("ymin"));
        assert!(is_positional_aesthetic("ymax"));
        assert!(is_positional_aesthetic("xend"));
        assert!(is_positional_aesthetic("yend"));

        // Non-positional
        assert!(!is_positional_aesthetic("color"));
        assert!(!is_positional_aesthetic("size"));
        assert!(!is_positional_aesthetic("fill"));
    }

    #[test]
    fn test_all_positional_contents() {
        assert!(ALL_POSITIONAL.contains(&"x"));
        assert!(ALL_POSITIONAL.contains(&"y"));
        assert!(ALL_POSITIONAL.contains(&"xmin"));
        assert!(ALL_POSITIONAL.contains(&"xmax"));
        assert!(ALL_POSITIONAL.contains(&"ymin"));
        assert!(ALL_POSITIONAL.contains(&"ymax"));
        assert!(ALL_POSITIONAL.contains(&"xend"));
        assert!(ALL_POSITIONAL.contains(&"yend"));
        assert_eq!(ALL_POSITIONAL.len(), 8);
    }

    #[test]
    fn test_is_aesthetic_name() {
        // Positional
        assert!(is_aesthetic_name("x"));
        assert!(is_aesthetic_name("y"));
        assert!(is_aesthetic_name("xmin"));
        assert!(is_aesthetic_name("yend"));

        // Visual
        assert!(is_aesthetic_name("color"));
        assert!(is_aesthetic_name("colour"));
        assert!(is_aesthetic_name("fill"));
        assert!(is_aesthetic_name("stroke"));
        assert!(is_aesthetic_name("opacity"));
        assert!(is_aesthetic_name("size"));
        assert!(is_aesthetic_name("shape"));
        assert!(is_aesthetic_name("linetype"));
        assert!(is_aesthetic_name("linewidth"));

        // Text
        assert!(is_aesthetic_name("label"));
        assert!(is_aesthetic_name("family"));
        assert!(is_aesthetic_name("fontface"));
        assert!(is_aesthetic_name("hjust"));
        assert!(is_aesthetic_name("vjust"));

        // Not aesthetics
        assert!(!is_aesthetic_name("foo"));
        assert!(!is_aesthetic_name("data"));
        assert!(!is_aesthetic_name("z"));
    }

    #[test]
    fn test_primary_aesthetic() {
        // Primary aesthetics return themselves
        assert_eq!(primary_aesthetic("x"), "x");
        assert_eq!(primary_aesthetic("y"), "y");
        assert_eq!(primary_aesthetic("color"), "color");
        assert_eq!(primary_aesthetic("fill"), "fill");

        // Variants return their primary
        assert_eq!(primary_aesthetic("xmin"), "x");
        assert_eq!(primary_aesthetic("xmax"), "x");
        assert_eq!(primary_aesthetic("xend"), "x");
        assert_eq!(primary_aesthetic("ymin"), "y");
        assert_eq!(primary_aesthetic("ymax"), "y");
        assert_eq!(primary_aesthetic("yend"), "y");
    }

    #[test]
    fn test_get_aesthetic_family() {
        // Primary aesthetics return full family
        let x_family = get_aesthetic_family("x");
        assert!(x_family.contains(&"x"));
        assert!(x_family.contains(&"xmin"));
        assert!(x_family.contains(&"xmax"));
        assert!(x_family.contains(&"xend"));
        assert_eq!(x_family.len(), 4);

        let y_family = get_aesthetic_family("y");
        assert!(y_family.contains(&"y"));
        assert!(y_family.contains(&"ymin"));
        assert!(y_family.contains(&"ymax"));
        assert!(y_family.contains(&"yend"));
        assert_eq!(y_family.len(), 4);

        // Variants return just themselves
        assert_eq!(get_aesthetic_family("xmin"), vec!["xmin"]);
        assert_eq!(get_aesthetic_family("ymax"), vec!["ymax"]);

        // Non-family aesthetics return just themselves
        assert_eq!(get_aesthetic_family("color"), vec!["color"]);
        assert_eq!(get_aesthetic_family("fill"), vec!["fill"]);
    }
}
