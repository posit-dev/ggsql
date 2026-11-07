//! Color utilities for ggsql visualization
//!
//! Provides color parsing, conversion, and interpolation functions.
//! Interpolation itself lives in `zygos::color`; this module adapts it to
//! ggsql's CSS-string interface.

pub use zygos::color::ColorSpace;
use zygos::color::{ramp, Color};

// =============================================================================
// Color Utilities
// =============================================================================

/// Convert a CSS color name/value to hex format.
/// Supports named colors (e.g., "red"), hex (#FF0000), rgb(), rgba(), hsl(), etc.
pub fn color_to_hex(value: &str) -> Result<String, String> {
    csscolorparser::parse(value)
        .map(|c| c.to_css_hex())
        .map_err(|e| format!("Invalid color '{}': {}", value, e))
}

/// Check if an aesthetic name is color-related.
pub fn is_color_aesthetic(aesthetic: &str) -> bool {
    matches!(aesthetic, "color" | "col" | "colour" | "fill" | "stroke")
}

// =============================================================================
// Color Interpolation
// =============================================================================

/// Interpolate between colors, returning `count` evenly-spaced colors.
///
/// Colors can be any CSS color format supported by `csscolorparser`:
/// - Named colors: "red", "blue", "coral"
/// - Hex: "#ff0000", "#f00"
/// - RGB: "rgb(255, 0, 0)"
/// - HSL: "hsl(0, 100%, 50%)"
///
/// # Arguments
/// * `colors` - Input color stops (at least 1 color required)
/// * `count` - Number of output colors to generate
/// * `space` - Color space to use for interpolation
///
/// # Returns
/// A vector of hex color strings (e.g., "#ff0000")
///
/// # Example
/// ```
/// use ggsql::plot::scale::colour::{interpolate_colors, ColorSpace};
///
/// // Generate a 5-color gradient from red to blue
/// let colors = interpolate_colors(&["red", "blue"], 5, ColorSpace::Oklab).unwrap();
/// assert_eq!(colors.len(), 5);
/// ```
pub fn interpolate_colors(
    colors: &[&str],
    count: usize,
    space: ColorSpace,
) -> Result<Vec<String>, String> {
    if colors.is_empty() {
        return Err("At least one color is required".to_string());
    }

    let stops: Vec<Color> = colors
        .iter()
        .map(|c| parse_to_color(c))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ramp(&stops, count, space)
        .iter()
        .map(color_to_css_hex)
        .collect())
}

/// Convenience function for creating a two-color gradient.
///
/// # Arguments
/// * `start` - Starting color (any CSS format)
/// * `end` - Ending color (any CSS format)
/// * `count` - Number of output colors
/// * `space` - Color space for interpolation
///
/// # Example
/// ```
/// use ggsql::plot::scale::colour::{gradient, ColorSpace};
///
/// let colors = gradient("white", "black", 5, ColorSpace::Oklab).unwrap();
/// assert_eq!(colors.len(), 5);
/// ```
pub fn gradient(
    start: &str,
    end: &str,
    count: usize,
    space: ColorSpace,
) -> Result<Vec<String>, String> {
    interpolate_colors(&[start, end], count, space)
}

/// Parse a CSS color string to a zygos Color.
fn parse_to_color(color: &str) -> Result<Color, String> {
    let parsed =
        csscolorparser::parse(color).map_err(|e| format!("Invalid color '{}': {}", color, e))?;

    Ok(Color::new([
        parsed.r as f32,
        parsed.g as f32,
        parsed.b as f32,
        parsed.a as f32,
    ]))
}

/// Convert a zygos Color to a hex string, dropping alpha.
pub(crate) fn color_to_css_hex(color: &Color) -> String {
    let [r, g, b, _] = color.components;
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", to_u8(r), to_u8(g), to_u8(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_hex_named_colors() {
        assert_eq!(color_to_hex("red").unwrap(), "#ff0000");
        assert_eq!(color_to_hex("blue").unwrap(), "#0000ff");
        assert_eq!(color_to_hex("green").unwrap(), "#008000");
        assert_eq!(color_to_hex("white").unwrap(), "#ffffff");
        assert_eq!(color_to_hex("black").unwrap(), "#000000");
    }

    #[test]
    fn test_color_to_hex_hex_values() {
        assert_eq!(color_to_hex("#ff0000").unwrap(), "#ff0000");
        assert_eq!(color_to_hex("#FF0000").unwrap(), "#ff0000");
        assert_eq!(color_to_hex("#f00").unwrap(), "#ff0000");
    }

    #[test]
    fn test_color_to_hex_invalid() {
        assert!(color_to_hex("notacolor").is_err());
        assert!(color_to_hex("").is_err());
    }

    #[test]
    fn test_is_color_aesthetic() {
        assert!(is_color_aesthetic("color"));
        assert!(is_color_aesthetic("col"));
        assert!(is_color_aesthetic("colour"));
        assert!(is_color_aesthetic("fill"));
        assert!(is_color_aesthetic("stroke"));
        assert!(!is_color_aesthetic("x"));
        assert!(!is_color_aesthetic("y"));
        assert!(!is_color_aesthetic("size"));
        assert!(!is_color_aesthetic("shape"));
    }

    // =========================================================================
    // Color Interpolation Tests
    // =========================================================================

    #[test]
    fn test_interpolate_colors_basic() {
        let colors = interpolate_colors(&["red", "blue"], 5, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], "#ff0000");
        assert_eq!(colors[4], "#0000ff");
    }

    #[test]
    fn test_interpolate_colors_empty_input() {
        let result = interpolate_colors(&[], 5, ColorSpace::Oklab);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("At least one color"));
    }

    #[test]
    fn test_interpolate_colors_invalid_color() {
        let result = interpolate_colors(&["red", "notacolor"], 5, ColorSpace::Oklab);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid color"));
    }

    #[test]
    fn test_gradient_convenience() {
        let colors = gradient("red", "blue", 5, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], "#ff0000");
        assert_eq!(colors[4], "#0000ff");
    }

    #[test]
    fn test_color_space_default() {
        assert_eq!(ColorSpace::default(), ColorSpace::Oklab);
    }
}
