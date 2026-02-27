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

// =============================================================================
// Positional Suffixes (applied to primary names automatically)
// =============================================================================

/// Positional aesthetic suffixes - applied to primary names to create variant aesthetics
/// e.g., "x" + "min" = "xmin", "pos1" + "end" = "pos1end"
pub const POSITIONAL_SUFFIXES: &[&str] = &["min", "max", "end", "intercept"];

/// Family size: primary + all suffixes (used for slicing family arrays)
const FAMILY_SIZE: usize = 1 + POSITIONAL_SUFFIXES.len();

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

// =============================================================================
// AestheticContext - Comprehensive context for aesthetic operations
// =============================================================================

/// Comprehensive context for aesthetic operations.
///
/// Pre-computes all mappings at creation time for efficient lookups.
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
/// assert_eq!(ctx.map_internal_to_user("pos1"), Some("x"));
///
/// // For polar coords
/// let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);
/// assert_eq!(ctx.map_user_to_internal("theta"), Some("pos1"));
/// assert_eq!(ctx.map_user_to_internal("radius"), Some("pos2"));
/// assert_eq!(ctx.map_internal_to_user("pos1"), Some("theta"));
///
/// // With facets
/// let ctx = AestheticContext::from_static(&["x", "y"], &["panel"]);
/// assert_eq!(ctx.map_user_to_internal("panel"), Some("facet1"));
/// assert_eq!(ctx.map_internal_to_user("facet1"), Some("panel"));
///
/// let ctx = AestheticContext::from_static(&["x", "y"], &["row", "column"]);
/// assert_eq!(ctx.map_user_to_internal("row"), Some("facet1"));
/// assert_eq!(ctx.map_user_to_internal("column"), Some("facet2"));
/// ```
#[derive(Debug, Clone)]
pub struct AestheticContext {
    /// User-facing positional names: ["x", "y"] or ["theta", "radius"] or custom names
    user_positional: Vec<String>,
    /// All user positional (with suffixes): ["x", "xmin", "xmax", "xend", "y", ...]
    all_user_positional: Vec<String>,
    /// Primary internal positional: ["pos1", "pos2", ...]
    primary_internal: Vec<String>,
    /// All internal positional: ["pos1", "pos1min", ..., "pos2", ...]
    all_internal_positional: Vec<String>,
    /// User-facing facet names: ["panel"] or ["row", "column"]
    user_facet: Vec<&'static str>,
    /// All internal facet names: ["facet1"] or ["facet1", "facet2"]
    all_internal_facet: Vec<String>,
    /// Non-positional aesthetics (static list)
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
        // Build positional mappings
        let mut all_user = Vec::new();
        let mut primary_internal = Vec::new();
        let mut all_internal = Vec::new();

        for (i, primary_name) in positional_names.iter().enumerate() {
            let pos_num = i + 1;
            let internal_base = format!("pos{}", pos_num);
            primary_internal.push(internal_base.clone());

            // Add primary first (e.g., "x", "pos1")
            all_user.push(primary_name.clone());
            all_internal.push(internal_base.clone());

            // Then add suffixed variants (e.g., "xmin", "pos1min")
            for suffix in POSITIONAL_SUFFIXES {
                all_user.push(format!("{}{}", primary_name, suffix));
                all_internal.push(format!("{}{}", internal_base, suffix));
            }
        }

        // Build internal facet names for active facets (from FACET clause or layer mappings)
        // These are used for internal→user mapping (to know which user name to show)
        let all_internal_facet: Vec<String> = (1..=facet_names.len())
            .map(|i| format!("facet{}", i))
            .collect();

        Self {
            user_positional: positional_names.to_vec(),
            all_user_positional: all_user,
            primary_internal,
            all_internal_positional: all_internal,
            user_facet: facet_names.to_vec(),
            all_internal_facet,
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
        // Check positional first
        if let Some(idx) = self
            .all_user_positional
            .iter()
            .position(|u| u == user_aesthetic)
        {
            return Some(self.all_internal_positional[idx].as_str());
        }

        // Check active facet (from FACET clause)
        if let Some(idx) = self.user_facet.iter().position(|u| *u == user_aesthetic) {
            return Some(self.all_internal_facet[idx].as_str());
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

    // === Mapping: Internal → User ===

    /// Map internal aesthetic (positional or facet) to user-facing name.
    ///
    /// Positional: "pos1" → "x", "pos2min" → "ymin"
    /// Facet: "facet1" → "panel" (or "row"), "facet2" → "column"
    pub fn map_internal_to_user(&self, internal_aesthetic: &str) -> Option<&str> {
        // Check positional first
        if let Some(idx) = self
            .all_internal_positional
            .iter()
            .position(|i| i == internal_aesthetic)
        {
            return Some(self.all_user_positional[idx].as_str());
        }
        // Check facet
        if let Some(idx) = self
            .all_internal_facet
            .iter()
            .position(|i| i == internal_aesthetic)
        {
            return Some(self.user_facet[idx]);
        }
        None
    }

    // === Checking (simple lookups in pre-computed lists) ===

    /// Check if user aesthetic is a positional (x, y, xmin, theta, etc.)
    pub fn is_user_positional(&self, name: &str) -> bool {
        self.all_user_positional.iter().any(|s| s == name)
    }

    /// Check if internal aesthetic is positional (pos1, pos1min, etc.)
    pub fn is_internal_positional(&self, name: &str) -> bool {
        self.all_internal_positional.iter().any(|s| s == name)
    }

    /// Check if internal aesthetic is primary positional (pos1, pos2, ...)
    pub fn is_primary_internal(&self, name: &str) -> bool {
        self.primary_internal.iter().any(|s| s == name)
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
        self.all_internal_facet.iter().any(|f| f == name)
    }

    /// Check if name is a facet aesthetic (user or internal)
    pub fn is_facet(&self, name: &str) -> bool {
        self.is_user_facet(name) || self.is_internal_facet(name)
    }

    // === Aesthetic Families ===

    /// Get the primary aesthetic for a family member.
    ///
    /// e.g., "pos1min" → "pos1", "pos2end" → "pos2"
    /// Non-positional aesthetics return themselves.
    pub fn primary_internal_aesthetic<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        // Check internal positional - find which primary it belongs to
        for (i, primary) in self.primary_internal.iter().enumerate() {
            let start = i * FAMILY_SIZE;
            let end = start + FAMILY_SIZE;
            if self.all_internal_positional[start..end]
                .iter()
                .any(|s| s == name)
            {
                return Some(primary.as_str());
            }
        }
        // Non-positional aesthetics are their own primary
        if self.is_non_positional(name) {
            return Some(name);
        }
        None
    }

    /// Get the aesthetic family for a primary aesthetic.
    ///
    /// e.g., "pos1" → ["pos1", "pos1min", "pos1max", "pos1end"]
    pub fn get_internal_family(&self, primary: &str) -> Option<&[String]> {
        for (i, p) in self.primary_internal.iter().enumerate() {
            if p == primary {
                let start = i * FAMILY_SIZE;
                let end = start + FAMILY_SIZE;
                return Some(&self.all_internal_positional[start..end]);
            }
        }
        None
    }

    /// Get the user-facing family for a user primary aesthetic.
    ///
    /// e.g., "x" → ["x", "xmin", "xmax", "xend"]
    pub fn get_user_family(&self, user_primary: &str) -> Option<&[String]> {
        for (i, p) in self.user_positional.iter().enumerate() {
            if *p == user_primary {
                let start = i * FAMILY_SIZE;
                let end = start + FAMILY_SIZE;
                return Some(&self.all_user_positional[start..end]);
            }
        }
        None
    }

    /// Get the primary user-facing aesthetic for a user variant.
    ///
    /// e.g., "xmin" → "x", "thetamax" → "theta", "color" → "color"
    /// Returns None if the aesthetic is not recognized.
    pub fn primary_user_aesthetic<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        // Check user positional - find which primary it belongs to
        for (i, primary) in self.user_positional.iter().enumerate() {
            let start = i * FAMILY_SIZE;
            let end = start + FAMILY_SIZE;
            if self.all_user_positional[start..end]
                .iter()
                .any(|s| s == name)
            {
                return Some(primary);
            }
        }
        // Non-positional aesthetics are their own primary
        if self.is_non_positional(name) {
            return Some(name);
        }
        None
    }

    // === Accessors ===

    /// Get all internal positional aesthetics (pos1, pos1min, ..., pos2, ...)
    pub fn all_internal_positional(&self) -> &[String] {
        &self.all_internal_positional
    }

    /// Get primary internal positional aesthetics (pos1, pos2, ...)
    pub fn primary_internal(&self) -> &[String] {
        &self.primary_internal
    }

    /// Get user positional aesthetics (x, y or theta, radius or custom names)
    pub fn user_positional(&self) -> &[String] {
        &self.user_positional
    }

    /// Get all user positional aesthetics with suffixes (x, xmin, xmax, xend, ...)
    pub fn all_user_positional(&self) -> &[String] {
        &self.all_user_positional
    }

    /// Get user-facing facet aesthetics (panel, row, column)
    pub fn user_facet(&self) -> &[&'static str] {
        &self.user_facet
    }

    /// Get all internal facet aesthetics (facet1, facet2)
    pub fn all_internal_facet(&self) -> &[String] {
        &self.all_internal_facet
    }

    /// Get non-positional aesthetics
    pub fn non_positional(&self) -> &'static [&'static str] {
        self.non_positional
    }
}

/// Check if aesthetic is a primary internal positional (pos1, pos2, etc.)
///
/// This function works with **internal** aesthetic names after transformation.
/// For user-facing checks before transformation, use `AestheticContext::is_user_positional()`.
#[inline]
pub fn is_primary_positional(aesthetic: &str) -> bool {
    // Check if it matches pattern: pos followed by digits only
    if aesthetic.starts_with("pos") && aesthetic.len() > 3 {
        return aesthetic[3..].chars().all(|c| c.is_ascii_digit());
    }
    false
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
/// Matches patterns like: pos1, pos2, pos1min, pos2max, pos1end, pos2intercept, etc.
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

/// Get the primary aesthetic for a given aesthetic name.
///
/// This function works with **internal** aesthetic names (pos1, pos2, etc.) and non-positional
/// aesthetics. After aesthetic transformation, all positional aesthetics are in internal format.
///
/// For internal positional variants: "pos1min" → "pos1", "pos2end" → "pos2"
/// For non-positional aesthetics: "color" → "color", "fill" → "fill"
///
/// Note: For user-facing aesthetic families (before transformation), use
/// `AestheticContext::primary_user_aesthetic()` instead.
#[inline]
pub fn primary_aesthetic(aesthetic: &str) -> &str {
    // Handle internal positional variants (pos1min -> pos1, pos2end -> pos2, etc.)
    if aesthetic.starts_with("pos") {
        for suffix in POSITIONAL_SUFFIXES {
            if let Some(base) = aesthetic.strip_suffix(suffix) {
                // Extract the base: pos1min -> pos1, pos2end -> pos2
                // Verify it's a valid positional (pos followed by digits)
                if base.len() > 3 && base[3..].chars().all(|c| c.is_ascii_digit()) {
                    // Return static str by leaking - this is acceptable for a small fixed set
                    // In practice this is only called with a limited set of aesthetics
                    return Box::leak(base.to_string().into_boxed_str());
                }
            }
        }
    }

    // Non-positional aesthetics (and internal primaries) return themselves
    aesthetic
}

/// Get all aesthetics in the same family as the given aesthetic.
///
/// This function works with **internal** aesthetic names (pos1, pos2, etc.) and non-positional
/// aesthetics. After aesthetic transformation, all positional aesthetics are in internal format.
///
/// For internal positional primary "pos1": returns `["pos1", "pos1min", "pos1max", "pos1end", "pos1intercept"]`
/// For internal positional variant "pos1min": returns just `["pos1min"]` (scales defined on primaries)
/// For non-positional aesthetics "color": returns just `["color"]`
///
/// This is used by scale resolution to find all columns that contribute to a scale's
/// input range (e.g., both `pos2min` and `pos2max` columns contribute to the "pos2" scale).
///
/// Note: For user-facing aesthetic families (before transformation), use
/// `AestheticContext::get_user_family()` instead.
pub fn get_aesthetic_family(aesthetic: &str) -> Vec<String> {
    // First, determine the primary aesthetic
    let primary = primary_aesthetic(aesthetic);

    // If aesthetic is not a primary (it's a variant), just return the aesthetic itself
    // since scales should be defined for primary aesthetics
    if primary != aesthetic {
        return vec![aesthetic.to_string()];
    }

    // Check if this is an internal positional (pos1, pos2, etc.)
    if primary.starts_with("pos")
        && primary.len() > 3
        && primary[3..].chars().all(|c| c.is_ascii_digit())
    {
        // Build the internal family: pos1 -> [pos1, pos1min, pos1max, pos1end, pos1intercept]
        let mut family = vec![primary.to_string()];
        for suffix in POSITIONAL_SUFFIXES {
            family.push(format!("{}{}", primary, suffix));
        }
        return family;
    }

    // Non-positional aesthetics don't have families, just return themselves
    vec![aesthetic.to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_positional() {
        // is_primary_positional() checks for internal names (pos1, pos2, etc.)
        assert!(is_primary_positional("pos1"));
        assert!(is_primary_positional("pos2"));
        assert!(is_primary_positional("pos10")); // supports any number

        // Variants are not primary
        assert!(!is_primary_positional("pos1min"));
        assert!(!is_primary_positional("pos2max"));

        // User-facing names are NOT primary positional (handled by AestheticContext)
        assert!(!is_primary_positional("x"));
        assert!(!is_primary_positional("y"));

        // Non-positional
        assert!(!is_primary_positional("color"));

        // Edge cases
        assert!(!is_primary_positional("pos")); // too short
        assert!(!is_primary_positional("position")); // not a valid pattern
    }

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
        assert!(is_positional_aesthetic("pos1intercept"));
        assert!(is_positional_aesthetic("pos2intercept"));

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

    #[test]
    fn test_primary_aesthetic() {
        // Handles internal names (pos1, pos2, etc.) and non-positional aesthetics.
        // For user-facing families, use AestheticContext.

        // Internal positional primaries return themselves
        assert_eq!(primary_aesthetic("pos1"), "pos1");
        assert_eq!(primary_aesthetic("pos2"), "pos2");

        // Internal positional variants return their primary
        assert_eq!(primary_aesthetic("pos1min"), "pos1");
        assert_eq!(primary_aesthetic("pos1max"), "pos1");
        assert_eq!(primary_aesthetic("pos1end"), "pos1");
        assert_eq!(primary_aesthetic("pos1intercept"), "pos1");
        assert_eq!(primary_aesthetic("pos2min"), "pos2");
        assert_eq!(primary_aesthetic("pos2max"), "pos2");
        assert_eq!(primary_aesthetic("pos2end"), "pos2");
        assert_eq!(primary_aesthetic("pos2intercept"), "pos2");

        // Non-positional aesthetics return themselves
        assert_eq!(primary_aesthetic("color"), "color");
        assert_eq!(primary_aesthetic("fill"), "fill");
        assert_eq!(primary_aesthetic("size"), "size");

        // User-facing names without internal family handling return themselves
        // (user-facing family resolution is handled by AestheticContext)
        assert_eq!(primary_aesthetic("x"), "x");
        assert_eq!(primary_aesthetic("y"), "y");
        assert_eq!(primary_aesthetic("xmin"), "xmin");
        assert_eq!(primary_aesthetic("ymax"), "ymax");
    }

    #[test]
    fn test_get_aesthetic_family() {
        // Handles internal names (pos1, pos2, etc.) and non-positional aesthetics.
        // For user-facing families, use AestheticContext.

        // Internal positional primary returns full family
        let pos1_family = get_aesthetic_family("pos1");
        assert!(pos1_family.iter().any(|s| s == "pos1"));
        assert!(pos1_family.iter().any(|s| s == "pos1min"));
        assert!(pos1_family.iter().any(|s| s == "pos1max"));
        assert!(pos1_family.iter().any(|s| s == "pos1end"));
        assert!(pos1_family.iter().any(|s| s == "pos1intercept"));
        assert_eq!(pos1_family.len(), 5);

        let pos2_family = get_aesthetic_family("pos2");
        assert!(pos2_family.iter().any(|s| s == "pos2"));
        assert!(pos2_family.iter().any(|s| s == "pos2min"));
        assert!(pos2_family.iter().any(|s| s == "pos2max"));
        assert!(pos2_family.iter().any(|s| s == "pos2end"));
        assert!(pos2_family.iter().any(|s| s == "pos2intercept"));
        assert_eq!(pos2_family.len(), 5);

        // Internal positional variants return just themselves
        assert_eq!(get_aesthetic_family("pos1min"), vec!["pos1min"]);
        assert_eq!(get_aesthetic_family("pos2max"), vec!["pos2max"]);

        // Non-positional aesthetics return just themselves (no family)
        assert_eq!(get_aesthetic_family("color"), vec!["color"]);
        assert_eq!(get_aesthetic_family("fill"), vec!["fill"]);

        // User-facing names without families return just themselves
        // (user-facing family resolution is handled by AestheticContext)
        assert_eq!(get_aesthetic_family("x"), vec!["x"]);
        assert_eq!(get_aesthetic_family("y"), vec!["y"]);
        assert_eq!(get_aesthetic_family("xmin"), vec!["xmin"]);
    }

    // ========================================================================
    // AestheticContext Tests
    // ========================================================================

    #[test]
    fn test_aesthetic_context_cartesian() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // User positional names
        assert_eq!(ctx.user_positional(), &["x", "y"]);

        // All user positional (with suffixes)
        let all_user: Vec<&str> = ctx
            .all_user_positional()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(all_user.contains(&"x"));
        assert!(all_user.contains(&"xmin"));
        assert!(all_user.contains(&"xmax"));
        assert!(all_user.contains(&"xend"));
        assert!(all_user.contains(&"xintercept"));
        assert!(all_user.contains(&"y"));
        assert!(all_user.contains(&"ymin"));
        assert!(all_user.contains(&"ymax"));
        assert!(all_user.contains(&"yend"));
        assert!(all_user.contains(&"yintercept"));

        // Primary internal names
        let primary: Vec<&str> = ctx.primary_internal().iter().map(|s| s.as_str()).collect();
        assert_eq!(primary, vec!["pos1", "pos2"]);
    }

    #[test]
    fn test_aesthetic_context_polar() {
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // User positional names
        assert_eq!(ctx.user_positional(), &["theta", "radius"]);

        // All user positional (with suffixes)
        let all_user: Vec<&str> = ctx
            .all_user_positional()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(all_user.contains(&"theta"));
        assert!(all_user.contains(&"thetamin"));
        assert!(all_user.contains(&"thetamax"));
        assert!(all_user.contains(&"thetaend"));
        assert!(all_user.contains(&"thetaintercept"));
        assert!(all_user.contains(&"radius"));
        assert!(all_user.contains(&"radiusmin"));
        assert!(all_user.contains(&"radiusmax"));
        assert!(all_user.contains(&"radiusend"));
        assert!(all_user.contains(&"radiusintercept"));
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
    fn test_aesthetic_context_internal_to_user() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Primary aesthetics
        assert_eq!(ctx.map_internal_to_user("pos1"), Some("x"));
        assert_eq!(ctx.map_internal_to_user("pos2"), Some("y"));

        // Variants
        assert_eq!(ctx.map_internal_to_user("pos1min"), Some("xmin"));
        assert_eq!(ctx.map_internal_to_user("pos1max"), Some("xmax"));
        assert_eq!(ctx.map_internal_to_user("pos1end"), Some("xend"));
        assert_eq!(ctx.map_internal_to_user("pos2min"), Some("ymin"));
        assert_eq!(ctx.map_internal_to_user("pos2max"), Some("ymax"));
        assert_eq!(ctx.map_internal_to_user("pos2end"), Some("yend"));

        // Unknown internal returns None
        assert_eq!(ctx.map_internal_to_user("pos3"), None);
        assert_eq!(ctx.map_internal_to_user("color"), None);
    }

    #[test]
    fn test_aesthetic_context_polar_mapping() {
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // User to internal
        assert_eq!(ctx.map_user_to_internal("theta"), Some("pos1"));
        assert_eq!(ctx.map_user_to_internal("radius"), Some("pos2"));
        assert_eq!(ctx.map_user_to_internal("thetaend"), Some("pos1end"));
        assert_eq!(ctx.map_user_to_internal("radiusmin"), Some("pos2min"));

        // Internal to user
        assert_eq!(ctx.map_internal_to_user("pos1"), Some("theta"));
        assert_eq!(ctx.map_internal_to_user("pos2"), Some("radius"));
        assert_eq!(ctx.map_internal_to_user("pos1end"), Some("thetaend"));
        assert_eq!(ctx.map_internal_to_user("pos2min"), Some("radiusmin"));
    }

    #[test]
    fn test_aesthetic_context_is_checks() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // User positional
        assert!(ctx.is_user_positional("x"));
        assert!(ctx.is_user_positional("ymin"));
        assert!(!ctx.is_user_positional("color"));
        assert!(!ctx.is_user_positional("pos1"));

        // Internal positional
        assert!(ctx.is_internal_positional("pos1"));
        assert!(ctx.is_internal_positional("pos2min"));
        assert!(!ctx.is_internal_positional("x"));
        assert!(!ctx.is_internal_positional("color"));

        // Primary internal
        assert!(ctx.is_primary_internal("pos1"));
        assert!(ctx.is_primary_internal("pos2"));
        assert!(!ctx.is_primary_internal("pos1min"));

        // Non-positional
        assert!(ctx.is_non_positional("color"));
        assert!(ctx.is_non_positional("fill"));
        assert!(!ctx.is_non_positional("x"));
        assert!(!ctx.is_non_positional("pos1"));
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
        assert_eq!(ctx.map_internal_to_user("facet1"), Some("panel"));

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
        assert_eq!(ctx.map_internal_to_user("facet1"), Some("row"));
        assert_eq!(ctx.map_internal_to_user("facet2"), Some("column"));
    }

    #[test]
    fn test_aesthetic_context_families() {
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);

        // Get internal family
        let pos1_family = ctx.get_internal_family("pos1").unwrap();
        let pos1_strs: Vec<&str> = pos1_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            pos1_strs,
            vec!["pos1", "pos1min", "pos1max", "pos1end", "pos1intercept"]
        );

        // Get user family
        let x_family = ctx.get_user_family("x").unwrap();
        let x_strs: Vec<&str> = x_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(x_strs, vec!["x", "xmin", "xmax", "xend", "xintercept"]);

        // Primary internal aesthetic
        assert_eq!(ctx.primary_internal_aesthetic("pos1"), Some("pos1"));
        assert_eq!(ctx.primary_internal_aesthetic("pos1min"), Some("pos1"));
        assert_eq!(ctx.primary_internal_aesthetic("pos2end"), Some("pos2"));
        assert_eq!(
            ctx.primary_internal_aesthetic("pos1intercept"),
            Some("pos1")
        );
        assert_eq!(ctx.primary_internal_aesthetic("color"), Some("color"));
    }

    #[test]
    fn test_aesthetic_context_user_family_resolution() {
        // Cartesian: user-facing families are x/y based
        let cartesian = AestheticContext::from_static(&["x", "y"], &[]);
        assert_eq!(cartesian.primary_user_aesthetic("x"), Some("x"));
        assert_eq!(cartesian.primary_user_aesthetic("xmin"), Some("x"));
        assert_eq!(cartesian.primary_user_aesthetic("xmax"), Some("x"));
        assert_eq!(cartesian.primary_user_aesthetic("xend"), Some("x"));
        assert_eq!(cartesian.primary_user_aesthetic("xintercept"), Some("x"));
        assert_eq!(cartesian.primary_user_aesthetic("y"), Some("y"));
        assert_eq!(cartesian.primary_user_aesthetic("ymin"), Some("y"));
        assert_eq!(cartesian.primary_user_aesthetic("ymax"), Some("y"));
        assert_eq!(cartesian.primary_user_aesthetic("color"), Some("color"));

        // Polar: user-facing families are theta/radius based
        let polar = AestheticContext::from_static(&["theta", "radius"], &[]);
        assert_eq!(polar.primary_user_aesthetic("theta"), Some("theta"));
        assert_eq!(polar.primary_user_aesthetic("thetamin"), Some("theta"));
        assert_eq!(polar.primary_user_aesthetic("thetamax"), Some("theta"));
        assert_eq!(polar.primary_user_aesthetic("thetaend"), Some("theta"));
        assert_eq!(polar.primary_user_aesthetic("radius"), Some("radius"));
        assert_eq!(polar.primary_user_aesthetic("radiusmin"), Some("radius"));
        assert_eq!(polar.primary_user_aesthetic("radiusmax"), Some("radius"));
        assert_eq!(polar.primary_user_aesthetic("color"), Some("color"));

        // Polar doesn't know about cartesian aesthetics
        assert_eq!(polar.primary_user_aesthetic("x"), None);
        assert_eq!(polar.primary_user_aesthetic("xmin"), None);
    }

    #[test]
    fn test_aesthetic_context_polar_user_families() {
        // Verify polar coords have correct user families
        let ctx = AestheticContext::from_static(&["theta", "radius"], &[]);

        // Get user family for theta
        let theta_family = ctx.get_user_family("theta").unwrap();
        let theta_strs: Vec<&str> = theta_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            theta_strs,
            vec![
                "theta",
                "thetamin",
                "thetamax",
                "thetaend",
                "thetaintercept"
            ]
        );

        // Get user family for radius
        let radius_family = ctx.get_user_family("radius").unwrap();
        let radius_strs: Vec<&str> = radius_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            radius_strs,
            vec![
                "radius",
                "radiusmin",
                "radiusmax",
                "radiusend",
                "radiusintercept"
            ]
        );

        // But internal families are the same for all coords
        let pos1_family = ctx.get_internal_family("pos1").unwrap();
        let pos1_strs: Vec<&str> = pos1_family.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            pos1_strs,
            vec!["pos1", "pos1min", "pos1max", "pos1end", "pos1intercept"]
        );
    }
}
