//! Color utilities for ggsql visualization
//!
//! Provides color parsing, conversion, and interpolation functions.

use palette::{FromColor, IntoColor, LinSrgb, Mix, Oklab, Srgb};

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

/// Color space options for interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Oklab color space - perceptually uniform (recommended for most uses).
    /// Produces visually pleasing gradients that avoid muddy colors.
    #[default]
    Oklab,
    /// Linear RGB color space - simple linear interpolation in RGB.
    /// Can produce darker intermediate colors for complementary hues.
    LinearRgb,
}

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

    if count == 0 {
        return Ok(vec![]);
    }

    // Parse all input colors to Srgb
    let srgb_colors: Vec<Srgb<f32>> = colors
        .iter()
        .map(|c| parse_to_srgb(c))
        .collect::<Result<Vec<_>, _>>()?;

    // Single color: return it `count` times
    if srgb_colors.len() == 1 {
        let hex = srgb_to_hex(&srgb_colors[0]);
        return Ok(vec![hex; count]);
    }

    // Two or more colors: interpolate
    let result = match space {
        ColorSpace::Oklab => interpolate_in_oklab(&srgb_colors, count),
        ColorSpace::LinearRgb => interpolate_in_linear_rgb(&srgb_colors, count),
    };

    Ok(result)
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

/// Parse a CSS color string to Srgb<f32>.
fn parse_to_srgb(color: &str) -> Result<Srgb<f32>, String> {
    let parsed =
        csscolorparser::parse(color).map_err(|e| format!("Invalid color '{}': {}", color, e))?;

    Ok(Srgb::new(parsed.r as f32, parsed.g as f32, parsed.b as f32))
}

/// Convert Srgb<f32> to hex string.
fn srgb_to_hex(color: &Srgb<f32>) -> String {
    let r = (color.red.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (color.green.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (color.blue.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Interpolate colors in Oklab color space.
fn interpolate_in_oklab(colors: &[Srgb<f32>], count: usize) -> Vec<String> {
    // Convert to Oklab
    let oklab_colors: Vec<Oklab<f32>> = colors
        .iter()
        .map(|c| Oklab::from_color(LinSrgb::from(*c)))
        .collect();

    if count == 1 {
        let lin: LinSrgb<f32> = oklab_colors[0].into_color();
        return vec![srgb_to_hex(&Srgb::from(lin))];
    }

    let num_segments = oklab_colors.len() - 1;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let segment_float = t * num_segments as f32;
        let segment = (segment_float.floor() as usize).min(num_segments - 1);
        let segment_t = segment_float - segment as f32;

        let interpolated = oklab_colors[segment].mix(oklab_colors[segment + 1], segment_t);
        let lin: LinSrgb<f32> = interpolated.into_color();
        result.push(srgb_to_hex(&Srgb::from(lin)));
    }

    result
}

/// Interpolate colors in linear RGB color space.
fn interpolate_in_linear_rgb(colors: &[Srgb<f32>], count: usize) -> Vec<String> {
    // Convert to linear RGB
    let lin_colors: Vec<LinSrgb<f32>> = colors.iter().map(|c| LinSrgb::from(*c)).collect();

    if count == 1 {
        return vec![srgb_to_hex(&Srgb::from(lin_colors[0]))];
    }

    let num_segments = lin_colors.len() - 1;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let segment_float = t * num_segments as f32;
        let segment = (segment_float.floor() as usize).min(num_segments - 1);
        let segment_t = segment_float - segment as f32;

        let interpolated = lin_colors[segment].mix(lin_colors[segment + 1], segment_t);
        result.push(srgb_to_hex(&Srgb::from(interpolated)));
    }

    result
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
        // Two colors, 5 output colors
        let colors = interpolate_colors(&["red", "blue"], 5, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 5);
        // First and last should be close to input colors
        assert_eq!(colors[0], "#ff0000"); // red
        assert_eq!(colors[4], "#0000ff"); // blue
    }

    #[test]
    fn test_interpolate_colors_linear_rgb() {
        let colors = interpolate_colors(&["white", "black"], 3, ColorSpace::LinearRgb).unwrap();
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], "#ffffff"); // white
        assert_eq!(colors[2], "#000000"); // black
    }

    #[test]
    fn test_interpolate_colors_single_input() {
        // Single color input should return that color repeated
        let colors = interpolate_colors(&["red"], 3, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], "#ff0000");
        assert_eq!(colors[1], "#ff0000");
        assert_eq!(colors[2], "#ff0000");
    }

    #[test]
    fn test_interpolate_colors_count_zero() {
        let colors = interpolate_colors(&["red", "blue"], 0, ColorSpace::Oklab).unwrap();
        assert!(colors.is_empty());
    }

    #[test]
    fn test_interpolate_colors_count_one() {
        let colors = interpolate_colors(&["red", "blue"], 1, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 1);
        assert_eq!(colors[0], "#ff0000"); // should be first color
    }

    #[test]
    fn test_interpolate_colors_count_two() {
        let colors = interpolate_colors(&["red", "blue"], 2, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[0], "#ff0000"); // red
        assert_eq!(colors[1], "#0000ff"); // blue
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
    fn test_interpolate_colors_multi_stop() {
        // Three colors: red -> white -> blue
        let colors = interpolate_colors(&["red", "white", "blue"], 5, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], "#ff0000"); // red
        assert_eq!(colors[2], "#ffffff"); // white (middle)
        assert_eq!(colors[4], "#0000ff"); // blue
    }

    #[test]
    fn test_interpolate_colors_hex_input() {
        let colors = interpolate_colors(&["#ff0000", "#0000ff"], 3, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], "#ff0000");
        assert_eq!(colors[2], "#0000ff");
    }

    #[test]
    fn test_gradient_convenience() {
        let colors = gradient("red", "blue", 5, ColorSpace::Oklab).unwrap();
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], "#ff0000");
        assert_eq!(colors[4], "#0000ff");
    }

    #[test]
    fn test_oklab_vs_linear_rgb_red_cyan() {
        // Red to cyan: Oklab should produce lighter intermediates,
        // while linear RGB produces darker/muddier intermediates
        let oklab = interpolate_colors(&["red", "cyan"], 5, ColorSpace::Oklab).unwrap();
        let linear = interpolate_colors(&["red", "cyan"], 5, ColorSpace::LinearRgb).unwrap();

        // Both should have same start and end
        assert_eq!(oklab[0], "#ff0000");
        assert_eq!(oklab[4], "#00ffff");
        assert_eq!(linear[0], "#ff0000");
        assert_eq!(linear[4], "#00ffff");

        // Middle colors should differ - Oklab tends to be brighter
        // We just verify they're different (the specific values depend on the algorithm)
        assert_ne!(oklab[2], linear[2]);
    }

    #[test]
    fn test_color_space_default() {
        // Default should be Oklab
        assert_eq!(ColorSpace::default(), ColorSpace::Oklab);
    }

    #[test]
    fn test_interpolate_preserves_endpoints() {
        // Verify that interpolation preserves exact endpoint colors
        let test_cases = vec![("black", "white"), ("red", "green"), ("#123456", "#abcdef")];

        for (start, end) in test_cases {
            let colors = interpolate_colors(&[start, end], 10, ColorSpace::Oklab).unwrap();
            // First color should match start (parsed and converted back)
            let start_hex = color_to_hex(start).unwrap();
            let end_hex = color_to_hex(end).unwrap();
            assert_eq!(
                colors[0], start_hex,
                "Start mismatch for {}->{}",
                start, end
            );
            assert_eq!(colors[9], end_hex, "End mismatch for {}->{}", start, end);
        }
    }

    #[test]
    fn test_interpolate_many_stops() {
        // Rainbow gradient with 6 stops
        let colors = interpolate_colors(
            &["red", "orange", "yellow", "green", "blue", "violet"],
            11,
            ColorSpace::Oklab,
        )
        .unwrap();
        assert_eq!(colors.len(), 11);
        // First and last should match
        assert_eq!(colors[0], "#ff0000"); // red
        assert_eq!(colors[10], "#ee82ee"); // violet
    }
}
