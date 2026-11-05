//! Named palette definitions for color and shape aesthetics.
//!
//! The tables and lookup functions live in `zygos::palettes`; this module
//! adapts them to ggsql's string-based `ArrayElement` vocabulary (hex color
//! strings, ggplot2-style hex linetype patterns) and adds the
//! `ArrayElement`-typed lookup used during scale resolution.

pub use zygos::palettes::linetype::get_linetype_palette;
pub use zygos::palettes::shape::get_shape_palette;

use zygos::LinetypeStep;

use super::colour::color_to_css_hex;
use crate::plot::ArrayElement;

/// Look up a color palette by name, as hex color strings ("#rrggbb").
pub fn get_color_palette(name: &str) -> Option<Vec<String>> {
    zygos::palettes::color::get_color_palette(name)
        .map(|colors| colors.iter().map(color_to_css_hex).collect())
}

/// Generate linetypes with evenly-spaced ink densities, as ggplot2-style
/// hex patterns ("1f", "88") with "solid" last — the string vocabulary the
/// linetype aesthetic's output range uses.
pub fn generate_linetype_sequential(count: usize) -> Vec<String> {
    zygos::palettes::linetype::generate_linetype_sequential(count)
        .iter()
        .map(|steps| steps_to_hex_pattern(steps))
        .collect()
}

/// Convert a generated Dash/Gap pattern back to its hex string form. The
/// sequential generator only emits whole-number Dash/Gap steps (an empty
/// pattern is "solid"), so the conversion is lossless.
fn steps_to_hex_pattern(steps: &[LinetypeStep]) -> String {
    if steps.is_empty() {
        return "solid".to_string();
    }
    steps
        .iter()
        .filter_map(|step| match step {
            LinetypeStep::Dash(v) | LinetypeStep::Gap(v) => char::from_digit(*v as u32, 16),
            LinetypeStep::Marker(_) => None,
        })
        .collect()
}

/// Look up a palette by aesthetic and name, returning as ArrayElements.
///
/// This helper consolidates the aesthetic-based palette lookup with proper error handling,
/// eliminating duplicate code across scale type implementations.
///
/// # Arguments
///
/// * `aesthetic` - The aesthetic type ("shape", "linetype", "color", "fill", "stroke")
/// * `name` - The palette name to look up
///
/// # Returns
///
/// Returns `Ok(Vec<ArrayElement>)` with the palette values, or an error if:
/// - The aesthetic doesn't support palettes
/// - The palette name is unknown
pub fn lookup_palette(aesthetic: &str, name: &str) -> Result<Vec<ArrayElement>, String> {
    let strings: Vec<String> = match aesthetic {
        "shape" => get_shape_palette(name).map(|p| p.iter().map(|s| s.to_string()).collect()),
        "linetype" => get_linetype_palette(name).map(|p| p.iter().map(|s| s.to_string()).collect()),
        "color" | "fill" | "stroke" => get_color_palette(name),
        _ => {
            return Err(
                "Palettes are only valid for color, fill, stroke, shape, and linetype aesthetics"
                    .to_string(),
            );
        }
    }
    .ok_or_else(|| format!("Unknown {} palette: '{}'", aesthetic, name))?;

    Ok(strings.into_iter().map(ArrayElement::String).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_color_palette() {
        let viridis = get_color_palette("viridis").unwrap();
        assert_eq!(viridis.len(), 256);
        assert_eq!(viridis[0], "#440154");
        assert!(get_color_palette("VIRIDIS").is_some()); // case insensitive
        assert!(get_color_palette("unknown").is_none());
    }

    #[test]
    fn test_generate_linetype_sequential() {
        assert_eq!(generate_linetype_sequential(0), Vec::<String>::new());
        assert_eq!(generate_linetype_sequential(1), vec!["solid"]);
        assert_eq!(generate_linetype_sequential(3), vec!["1f", "88", "solid"]);
    }

    #[test]
    fn test_lookup_palette_color() {
        let result = lookup_palette("fill", "viridis");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 256);

        assert_eq!(lookup_palette("stroke", "tableau10").unwrap().len(), 10);
        assert_eq!(lookup_palette("color", "ggsql").unwrap().len(), 10);
    }

    #[test]
    fn test_lookup_palette_shape() {
        assert_eq!(lookup_palette("shape", "default").unwrap().len(), 9);
        assert_eq!(lookup_palette("linetype", "default").unwrap().len(), 6);
    }

    #[test]
    fn test_lookup_palette_unknown_palette() {
        let err = lookup_palette("fill", "nonexistent_palette").unwrap_err();
        assert!(err.contains("Unknown"));
        assert!(err.contains("nonexistent_palette"));
    }

    #[test]
    fn test_lookup_palette_invalid_aesthetic() {
        // The error must not leak the internal aesthetic name (e.g. `pos1`)
        // or the palette name.
        let err = lookup_palette("pos1", "viridis").unwrap_err();
        assert_eq!(
            err,
            "Palettes are only valid for color, fill, stroke, shape, and linetype aesthetics"
        );
    }
}
