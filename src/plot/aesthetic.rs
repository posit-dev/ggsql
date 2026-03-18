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
//!
//! # Internal vs User-Facing Aesthetics
//!
//! The pipeline uses internal positional aesthetic names (pos1, pos2, etc.) that are
//! transformed from user-facing names (x/y or theta/radius) early in the pipeline
//! and transformed back for output. This is handled by `AestheticContext`.

use std::collections::HashMap;

// =============================================================================
// Positional Suffixes (applied to primary names automatically)
// =============================================================================

/// Positional aesthetic suffixes - applied to primary names to create variant aesthetics
/// e.g., "x" + "min" = "xmin", "pos1" + "end" = "pos1end"
///
/// Note: "offset" is intentionally NOT included here because it's a positioning
/// adjustment that shouldn't influence scale training or be part of aesthetic families.
/// The `flip_positional` method handles offset correctly via prefix detection.
pub const POSITIONAL_SUFFIXES: &[&str] = &["min", "max", "end"];

// =============================================================================
// Static Constants (for backward compatibility with existing code)
// =============================================================================

/// User-facing facet aesthetics (for creating small multiples)
///
/// These aesthetics control faceting layout:
/// - `panel`: Single variable faceting (wrap layout)
/// - `row`: Row variable for grid faceting
/// - `column`: Column variable for grid faceting
///
/// After aesthetic transformation, these become internal names:
/// - `panel` → `facet1`
/// - `row` → `facet1`, `column` → `facet2`
pub const USER_FACET_AESTHETICS: &[&str] = &["panel", "row", "column"];

/// Non-positional aesthetics (visual properties shown in legends or applied to marks)
///
/// These include:
/// - Color aesthetics: color, colour, fill, stroke, opacity
/// - Size/shape aesthetics: size, shape, linetype, linewidth
/// - Dimension aesthetics: width, height
/// - Text aesthetics: label, typeface, fontweight, italic, hjust, vjust
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
    "typeface",
    "fontweight",
    "italic",
    "fontsize",
    "hjust",
    "vjust",
];

// =============================================================================
// AestheticContext - Comprehensive context for aesthetic operations
// =============================================================================

/// Comprehensive context for aesthetic operations.
///
/// Uses HashMaps for efficient O(1) lookups between user-facing and internal aesthetic names.
/// Used to transform between user-facing aesthetic names (x/y or theta/radius)
/// and internal names (pos1/pos2), as well as facet aesthetics (panel/row/column)
/// to internal facet names (facet1/facet2).
///
/// # Example
///
/// ```ignore
/// use ggsql::plot::AestheticContext;
///
/// // For cartesian coords
/// let ctx = AestheticContext::from_static(&["x", "y"], &[]);
/// assert_eq!(ctx.map_user_to_internal("x"), Some("pos1"));
/// assert_eq!(ctx.map_user_to_internal("ymin"), Some("pos2min"));
///
/// // For polar coords
/// let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);
/// assert_eq!(ctx.map_user_to_internal("theta"), Some("pos1"));
/// assert_eq!(ctx.map_user_to_internal("radius"), Some("pos2"));
///
/// // With facets
/// let ctx = AestheticContext::from_static(&["x", "y"], &["panel"]);
/// assert_eq!(ctx.map_user_to_internal("panel"), Some("facet1"));
///
/// let ctx = AestheticContext::from_static(&["x", "y"], &["row", "column"]);
/// assert_eq!(ctx.map_user_to_internal("row"), Some("facet1"));
/// assert_eq!(ctx.map_user_to_internal("column"), Some("facet2"));
/// ```
#[derive(Debug, Clone)]
pub struct AestheticContext {
    // User → Internal mapping (O(1) lookups)
    user_to_internal: HashMap<String, String>,

    // Family lookups (internal names only)
    internal_to_primary: HashMap<String, String>,
    primary_to_internal_family: HashMap<String, Vec<String>>,

    // For iteration (ordered lists)
    user_primaries: Vec<String>,
    internal_primaries: Vec<String>,

    // Facet mappings
    user_facet: Vec<&'static str>,
    internal_facet: Vec<String>,

    // Non-positional (static reference)
    non_positional: &'static [&'static str],
}

impl AestheticContext {
    /// Create context from coord's positional names and facet's aesthetic names.
    ///
    /// # Arguments
    ///
    /// * `positional_names` - Primary positional aesthetic names (e.g., ["x", "y"] or custom names)
    /// * `facet_names` - User-facing facet aesthetic names from facet layout
    ///   (e.g., ["panel"] for wrap, ["row", "column"] for grid)
    pub fn new(positional_names: &[String], facet_names: &[&'static str]) -> Self {
        // Initialize all HashMaps and vectors
        let mut user_to_internal = HashMap::new();
        let mut internal_to_primary = HashMap::new();
        let mut primary_to_internal_family = HashMap::new();

        let mut user_primaries = Vec::new();
        let mut internal_primaries = Vec::new();

        // Build positional mappings
        for (i, user_primary) in positional_names.iter().enumerate() {
            let pos_num = i + 1;
            let internal_primary = format!("pos{}", pos_num);

            // Track primaries
            user_primaries.push(user_primary.clone());
            internal_primaries.push(internal_primary.clone());

            // Build internal family
            let mut internal_family = vec![internal_primary.clone()];

            // Add primary to mappings
            user_to_internal.insert(user_primary.clone(), internal_primary.clone());
            internal_to_primary.insert(internal_primary.clone(), internal_primary.clone());

            // Add suffixed variants
            for suffix in POSITIONAL_SUFFIXES {
                let user_variant = format!("{}{}", user_primary, suffix);
                let internal_variant = format!("{}{}", internal_primary, suffix);

                user_to_internal.insert(user_variant, internal_variant.clone());
                internal_to_primary.insert(internal_variant.clone(), internal_primary.clone());
                internal_family.push(internal_variant);
            }

            // Store internal family
            primary_to_internal_family.insert(internal_primary, internal_family);
        }

        // Build internal facet names for active facets (from FACET clause or layer mappings)
        let internal_facet: Vec<String> = (1..=facet_names.len())
            .map(|i| format!("facet{}", i))
            .collect();

        Self {
            user_to_internal,
            internal_to_primary,
            primary_to_internal_family,
            user_primaries,
            internal_primaries,
            user_facet: facet_names.to_vec(),
            internal_facet,
            non_positional: NON_POSITIONAL,
        }
    }

    /// Create context from static positional names and facet names.
    ///
    /// Convenience method for creating context from static string slices (e.g., from coord defaults).
    pub fn from_static(positional_names: &[&'static str], facet_names: &[&'static str]) -> Self {
        let owned_positional: Vec<String> =
            positional_names.iter().map(|s| s.to_string()).collect();
        Self::new(&owned_positional, facet_names)
    }

    // === Mapping: User → Internal ===

    /// Map user aesthetic (positional or facet) to internal name.
    ///
    /// Positional: "x" → "pos1", "ymin" → "pos2min", "theta" → "pos1"
    /// Facet: "panel" → "facet1", "row" → "facet1", "column" → "facet2"
    ///
    /// Note: Facet mappings work regardless of whether a FACET clause exists,
    /// allowing layer-declared facet aesthetics to be transformed.
    pub fn map_user_to_internal(&self, user_aesthetic: &str) -> Option<&str> {
        // Check positional first (O(1) HashMap lookup)
        if let Some(internal) = self.user_to_internal.get(user_aesthetic) {
            return Some(internal.as_str());
        }

        // Check active facet (from FACET clause)
        if let Some(idx) = self.user_facet.iter().position(|u| *u == user_aesthetic) {
            return Some(self.internal_facet[idx].as_str());
        }

        // Always map user-facing facet names to internal names,
        // even when no FACET clause exists (allows layer-declared facets)
        // panel → facet1 (wrap layout)
        // row → facet1, column → facet2 (grid layout)
        match user_aesthetic {
            "panel" => Some("facet1"),
            "row" => Some("facet1"),
            "column" => Some("facet2"),
            _ => None,
        }
    }

    /// Map internal aesthetic to user-facing name (reverse of map_user_to_internal).
    ///
    /// Positional: "pos1" → "x", "pos2min" → "ymin", "pos1" → "theta" (for polar)
    /// Facet: "facet1" → "panel" (wrap), "facet1" → "row" (grid), "facet2" → "column" (grid)
    /// Non-positional: "color" → "color" (unchanged)
    ///
    /// Returns None if the internal aesthetic is not recognized.
    pub fn map_internal_to_user(&self, internal_aesthetic: &str) -> String {
        // Check internal facet (facet1, facet2)
        if let Some(idx) = self
            .internal_facet
            .iter()
            .position(|i| i == internal_aesthetic)
        {
            return self.user_facet[idx].to_string();
        }

        // Check internal positional (pos1, pos1min, pos2, etc.)
        // Iterate through user_to_internal to find reverse mapping
        for (user, internal) in &self.user_to_internal {
            if internal == internal_aesthetic {
                return user.to_string();
            }
        }

        // Non-positional aesthetics (color, size, etc.)
        // Internal is the same as external
        internal_aesthetic.to_string()
    }

    // === Checking (O(1) HashMap lookups) ===

    /// Check if internal aesthetic is primary positional (pos1, pos2, ...)
    pub fn is_primary_internal(&self, name: &str) -> bool {
        self.internal_primaries.iter().any(|s| s == name)
    }

    /// Check if aesthetic is non-positional (color, size, etc.)
    pub fn is_non_positional(&self, name: &str) -> bool {
        self.non_positional.contains(&name)
    }

    /// Check if name is a user-facing facet aesthetic (panel, row, column)
    pub fn is_user_facet(&self, name: &str) -> bool {
        self.user_facet.contains(&name)
    }

    /// Check if name is an internal facet aesthetic (facet1, facet2)
    pub fn is_internal_facet(&self, name: &str) -> bool {
        self.internal_facet.iter().any(|f| f == name)
    }

    /// Check if name is a facet aesthetic (user or internal)
    pub fn is_facet(&self, name: &str) -> bool {
        self.is_user_facet(name) || self.is_internal_facet(name)
    }

    // === Aesthetic Families (O(1) HashMap lookups) ===

    /// Get the primary aesthetic for an internal family member.
    ///
    /// e.g., "pos1min" → "pos1", "pos2end" → "pos2"
    /// Non-positional aesthetics return themselves.
    pub fn primary_internal_positional<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        // Check internal positional (O(1) lookup)
        if let Some(primary) = self.internal_to_primary.get(name) {
            return Some(primary.as_str());
        }
        // Non-positional aesthetics are their own primary
        if self.is_non_positional(name) {
            return Some(name);
        }
        None
    }

    /// Get the internal aesthetic family for a primary aesthetic.
    ///
    /// e.g., "pos1" → ["pos1", "pos1min", "pos1max", "pos1end"]
    pub fn internal_positional_family(&self, primary: &str) -> Option<&[String]> {
        self.primary_to_internal_family
            .get(primary)
            .map(|v| v.as_slice())
    }

    // === Accessors ===

    /// Get primary internal positional aesthetics (pos1, pos2, ...)
    pub fn internal_positional(&self) -> &[String] {
        &self.internal_primaries
    }

    /// Get user positional aesthetics (x, y or theta, radius or custom names)
    pub fn user_positional(&self) -> &[String] {
        &self.user_primaries
    }

    /// Get user-facing facet aesthetics (panel, row, column)
    pub fn user_facet(&self) -> &[&'static str] {
        &self.user_facet
    }

    // === Orientation Flipping ===

    /// Flip a positional aesthetic to its opposite position.
    ///
    /// Swaps pos1 ↔ pos2 (and their suffixed variants like pos1min ↔ pos2min).
    /// Non-positional aesthetics are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ctx = AestheticContext::from_static(&["x", "y"], &[]);
    /// assert_eq!(ctx.flip_positional("pos1"), "pos2");
    /// assert_eq!(ctx.flip_positional("pos2min"), "pos1min");
    /// assert_eq!(ctx.flip_positional("pos1end"), "pos2end");
    /// assert_eq!(ctx.flip_positional("color"), "color"); // unchanged
    /// ```
    pub fn flip_positional(&self, name: &str) -> String {
        // Only flip if we have exactly 2 positional aesthetics
        if self.internal_primaries.len() != 2 {
            return name.to_string();
        }

        // Check if it's a pos1 or pos2 variant
        if let Some(rest) = name.strip_prefix("pos1") {
            return format!("pos2{}", rest);
        }
        if let Some(rest) = name.strip_prefix("pos2") {
            return format!("pos1{}", rest);
        }

        // Not a positional aesthetic, return unchanged
        name.to_string()
    }
}

/// Check if aesthetic is a user-facing facet aesthetic (panel, row, column)
///
/// Use this function for checks BEFORE aesthetic transformation.
/// For checks after transformation, use `is_facet_aesthetic`.
#[inline]
pub fn is_user_facet_aesthetic(aesthetic: &str) -> bool {
    USER_FACET_AESTHETICS.contains(&aesthetic)
}

/// Check if aesthetic is an internal facet aesthetic (facet1, facet2, etc.)
///
/// Facet aesthetics control the creation of small multiples (faceted plots).
/// They only support Discrete and Binned scale types, and cannot have output ranges (TO clause).
///
/// This function works with **internal** aesthetic names after transformation.
/// For user-facing checks before transformation, use `is_user_facet_aesthetic`.
#[inline]
pub fn is_facet_aesthetic(aesthetic: &str) -> bool {
    // Check pattern: facet followed by digits only (facet1, facet2, etc.)
    if aesthetic.starts_with("facet") && aesthetic.len() > 5 {
        return aesthetic[5..].chars().all(|c| c.is_ascii_digit());
    }
    false
}

/// Check if aesthetic is an internal positional (pos1, pos1min, pos2max, etc.)
///
/// This function works with **internal** aesthetic names after transformation.
/// Matches patterns like: pos1, pos2, pos1min, pos2max, pos1end, etc.
///
/// For user-facing checks before transformation, use `AestheticContext::is_user_positional()`.
#[inline]
pub fn is_positional_aesthetic(name: &str) -> bool {
    if !name.starts_with("pos") || name.len() <= 3 {
        return false;
    }

    // Check for primary: pos followed by only digits (pos1, pos2, pos10, etc.)
    let after_pos = &name[3..];
    if after_pos.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }

    // Check for variants: posN followed by a suffix
    for suffix in POSITIONAL_SUFFIXES {
        if let Some(base) = name.strip_suffix(suffix) {
            if base.starts_with("pos") && base.len() > 3 {
                let num_part = &base[3..];
                if num_part.chars().all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
    }

    false
}

/// Parse a positional aesthetic name to extract its slot number and suffix.
///
/// Returns `Some((slot, suffix))` for positional aesthetics:
/// - `pos1` → (1, "")
/// - `pos2min` → (2, "min")
/// - `pos1end` → (1, "end")
///
/// Returns `None` for non-positional aesthetics.
pub fn parse_positional(name: &str) -> Option<(u8, &str)> {
    if !name.starts_with("pos") {
        return None;
    }
    let rest = &name[3..];
    let slot_char = rest.chars().next()?;
    let slot = slot_char.to_digit(10)? as u8;
    let suffix = &rest[1..];
    Some((slot, suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facet_aesthetic() {
        // Internal facet aesthetics (after transformation)
        assert!(is_facet_aesthetic("facet1"));
        assert!(is_facet_aesthetic("facet2"));
        assert!(is_facet_aesthetic("facet10")); // supports any number
        assert!(!is_facet_aesthetic("facet")); // too short
        assert!(!is_facet_aesthetic("facetx")); // not a number

        // User-facing names are NOT internal facet aesthetics
        assert!(!is_facet_aesthetic("panel"));
        assert!(!is_facet_aesthetic("row"));
        assert!(!is_facet_aesthetic("column"));

        // Other aesthetics
        assert!(!is_facet_aesthetic("x"));
        assert!(!is_facet_aesthetic("color"));
        assert!(!is_facet_aesthetic("pos1"));
    }

    #[test]
    fn test_user_facet_aesthetic() {
        // User-facing facet aesthetics (before transformation)
        assert!(is_user_facet_aesthetic("panel"));
        assert!(is_user_facet_aesthetic("row"));
        assert!(is_user_facet_aesthetic("column"));

        // Internal names are NOT user-facing
        assert!(!is_user_facet_aesthetic("facet1"));
        assert!(!is_user_facet_aesthetic("facet2"));

        // Other aesthetics
        assert!(!is_user_facet_aesthetic("x"));
        assert!(!is_user_facet_aesthetic("color"));
    }

    #[test]
    fn test_positional_aesthetic() {
        // Checks internal positional names (pos1, pos2, etc. and variants)
        // For user-facing checks, use AestheticContext::is_user_positional()

        // Primary internal
        assert!(is_positional_aesthetic("pos1"));
        assert!(is_positional_aesthetic("pos2"));
        assert!(is_positional_aesthetic("pos10")); // supports any number

        // Variants
        assert!(is_positional_aesthetic("pos1min"));
        assert!(is_positional_aesthetic("pos1max"));
        assert!(is_positional_aesthetic("pos2min"));
        assert!(is_positional_aesthetic("pos2max"));
        assert!(is_positional_aesthetic("pos1end"));
        assert!(is_positional_aesthetic("pos2end"));

        // User-facing names are NOT positional (handled by AestheticContext)
        assert!(!is_positional_aesthetic("x"));
        assert!(!is_positional_aesthetic("y"));
        assert!(!is_positional_aesthetic("xmin"));
        assert!(!is_positional_aesthetic("theta"));

        // Non-positional
        assert!(!is_positional_aesthetic("color"));
        assert!(!is_positional_aesthetic("size"));
        assert!(!is_positional_aesthetic("fill"));

        // Edge cases
        assert!(!is_positional_aesthetic("pos")); // too short
        assert!(!is_positional_aesthetic("position")); // not a valid pattern
    }

    // ========================================================================
    // AestheticContext Tests
    // ========================================================================

    #[test]
    fn test_aesthetic_context_cartesian() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // User positional names
        assert_eq!(ctx.user_positional(), &["x", "y"]);

        // Primary internal names
        let primary: Vec<&str> = ctx
            .internal_positional()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(primary, vec!["pos1", "pos2"]);
    }

    #[test]
    fn test_aesthetic_context_polar() {
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // User positional names
        assert_eq!(ctx.user_positional(), &["theta", "radius"]);

        // Primary internal names
        let primary: Vec<&str> = ctx
            .internal_positional()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(primary, vec!["pos1", "pos2"]);
    }

    #[test]
    fn test_aesthetic_context_user_to_internal() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Primary aesthetics
        assert_eq!(ctx.map_user_to_internal("x"), Some("pos1"));
        assert_eq!(ctx.map_user_to_internal("y"), Some("pos2"));

        // Variants
        assert_eq!(ctx.map_user_to_internal("xmin"), Some("pos1min"));
        assert_eq!(ctx.map_user_to_internal("xmax"), Some("pos1max"));
        assert_eq!(ctx.map_user_to_internal("xend"), Some("pos1end"));
        assert_eq!(ctx.map_user_to_internal("ymin"), Some("pos2min"));
        assert_eq!(ctx.map_user_to_internal("ymax"), Some("pos2max"));
        assert_eq!(ctx.map_user_to_internal("yend"), Some("pos2end"));

        // Non-positional returns None
        assert_eq!(ctx.map_user_to_internal("color"), None);
        assert_eq!(ctx.map_user_to_internal("fill"), None);
    }

    #[test]
    fn test_aesthetic_context_polar_mapping() {
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // User to internal
        assert_eq!(ctx.map_user_to_internal("theta"), Some("pos1"));
        assert_eq!(ctx.map_user_to_internal("radius"), Some("pos2"));
        assert_eq!(ctx.map_user_to_internal("thetaend"), Some("pos1end"));
        assert_eq!(ctx.map_user_to_internal("radiusmin"), Some("pos2min"));
    }

    #[test]
    fn test_aesthetic_context_is_primary_internal() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Primary internal
        assert!(ctx.is_primary_internal("pos1"));
        assert!(ctx.is_primary_internal("pos2"));
        assert!(!ctx.is_primary_internal("pos1min"));
        assert!(!ctx.is_primary_internal("x"));
        assert!(!ctx.is_primary_internal("color"));
    }

    #[test]
    fn test_aesthetic_context_with_facets() {
        let ctx = AestheticContext::from_static(&["x", "y"], &["panel"]);

        // Check user facet
        assert!(ctx.is_user_facet("panel"));
        assert!(!ctx.is_user_facet("row"));
        assert_eq!(ctx.user_facet(), &["panel"]);

        // Check internal facet
        assert!(ctx.is_internal_facet("facet1"));
        assert!(!ctx.is_internal_facet("panel"));

        // Check mapping
        assert_eq!(ctx.map_user_to_internal("panel"), Some("facet1"));

        // Check combined is_facet
        assert!(ctx.is_facet("panel")); // user
        assert!(ctx.is_facet("facet1")); // internal
    }

    #[test]
    fn test_aesthetic_context_with_grid_facets() {
        let ctx = AestheticContext::from_static(&["x", "y"], &["row", "column"]);

        // Check user facet
        assert!(ctx.is_user_facet("row"));
        assert!(ctx.is_user_facet("column"));
        assert!(!ctx.is_user_facet("panel"));
        assert_eq!(ctx.user_facet(), &["row", "column"]);

        // Check internal facet
        assert!(ctx.is_internal_facet("facet1"));
        assert!(ctx.is_internal_facet("facet2"));

        // Check mappings
        assert_eq!(ctx.map_user_to_internal("row"), Some("facet1"));
        assert_eq!(ctx.map_user_to_internal("column"), Some("facet2"));
    }

    #[test]
    fn test_aesthetic_context_families() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Get internal family (offset not included - it's a positioning adjustment)
        let pos1_family = ctx.internal_positional_family("pos1").unwrap();
        let pos1_strs: Vec<&str> = pos1_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(pos1_strs, vec!["pos1", "pos1min", "pos1max", "pos1end"]);

        // Primary internal aesthetic
        assert_eq!(ctx.primary_internal_positional("pos1"), Some("pos1"));
        assert_eq!(ctx.primary_internal_positional("pos1min"), Some("pos1"));
        assert_eq!(ctx.primary_internal_positional("pos2end"), Some("pos2"));
        assert_eq!(ctx.primary_internal_positional("color"), Some("color"));
    }

    #[test]
    fn test_aesthetic_context_internal_to_user_cartesian() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Primary aesthetics
        assert_eq!(ctx.map_internal_to_user("pos1"), "x");
        assert_eq!(ctx.map_internal_to_user("pos2"), "y");

        // Variants
        assert_eq!(ctx.map_internal_to_user("pos1min"), "xmin");
        assert_eq!(ctx.map_internal_to_user("pos1max"), "xmax");
        assert_eq!(ctx.map_internal_to_user("pos1end"), "xend");
        assert_eq!(ctx.map_internal_to_user("pos2min"), "ymin");
        assert_eq!(ctx.map_internal_to_user("pos2max"), "ymax");
        assert_eq!(ctx.map_internal_to_user("pos2end"), "yend");

        // Non-positional aesthetics remain unchanged
        assert_eq!(ctx.map_internal_to_user("color"), "color");
        assert_eq!(ctx.map_internal_to_user("size"), "size");
        assert_eq!(ctx.map_internal_to_user("fill"), "fill");
    }

    #[test]
    fn test_aesthetic_context_internal_to_user_polar() {
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // Primary aesthetics map to polar names
        assert_eq!(ctx.map_internal_to_user("pos1"), "theta");
        assert_eq!(ctx.map_internal_to_user("pos2"), "radius");

        // Variants
        assert_eq!(ctx.map_internal_to_user("pos1end"), "thetaend");
        assert_eq!(ctx.map_internal_to_user("pos2min"), "radiusmin");
        assert_eq!(ctx.map_internal_to_user("pos2max"), "radiusmax");
    }

    #[test]
    fn test_aesthetic_context_internal_to_user_facets() {
        // Wrap facet (panel)
        let ctx_wrap = AestheticContext::from_static(&["x", "y"], &["panel"]);
        assert_eq!(ctx_wrap.map_internal_to_user("facet1"), "panel");

        // Grid facet (row, column)
        let ctx_grid = AestheticContext::from_static(&["x", "y"], &["row", "column"]);
        assert_eq!(ctx_grid.map_internal_to_user("facet1"), "row");
        assert_eq!(ctx_grid.map_internal_to_user("facet2"), "column");
    }

    #[test]
    fn test_aesthetic_context_roundtrip() {
        // Test that user -> internal -> user roundtrips correctly
        let ctx = AestheticContext::from_static(&["x", "y"], &["panel"]);

        // Positional
        let internal = ctx.map_user_to_internal("x").unwrap();
        assert_eq!(ctx.map_internal_to_user(internal), "x");

        let internal = ctx.map_user_to_internal("ymin").unwrap();
        assert_eq!(ctx.map_internal_to_user(internal), "ymin");

        // Facet
        let internal = ctx.map_user_to_internal("panel").unwrap();
        assert_eq!(ctx.map_internal_to_user(internal), "panel");
    }
    #[test]
    fn test_parse_positional() {
        // Primary positional
        assert_eq!(parse_positional("pos1"), Some((1, "")));
        assert_eq!(parse_positional("pos2"), Some((2, "")));

        // Variants with suffixes
        assert_eq!(parse_positional("pos1min"), Some((1, "min")));
        assert_eq!(parse_positional("pos2max"), Some((2, "max")));
        assert_eq!(parse_positional("pos1end"), Some((1, "end")));

        // Non-positional
        assert_eq!(parse_positional("color"), None);
        assert_eq!(parse_positional("x"), None);
        assert_eq!(parse_positional("xmin"), None);
    }
}
