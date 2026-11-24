//! Shape-name → SVG path conversion for Vega-Lite symbol marks.
//!
//! The shape geometry lives in `zygos::patterns::shape` as kurbo paths;
//! serializing them to SVG path strings is a Vega-Lite concern, so it
//! lives here in the writer.

/// Convert a named shape to an SVG path string for Vega-Lite.
///
/// Composite shapes (e.g. "square-cross") produce multiple subpaths joined
/// by spaces; Vega-Lite fills them with the evenodd fill-rule. Returns
/// `None` for unknown shapes.
pub fn shape_to_svg_path(name: &str) -> Option<String> {
    let shape = zygos::patterns::shape::builtin_shape(name)?;
    Some(
        shape
            .paths()
            .iter()
            .map(|p| p.to_svg())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_shapes_have_paths() {
        for name in [
            "circle",
            "square",
            "diamond",
            "triangle-up",
            "star",
            "cross",
        ] {
            assert!(
                shape_to_svg_path(name).is_some(),
                "shape '{name}' should have an SVG path"
            );
        }
    }

    #[test]
    fn unknown_shape_is_none() {
        assert!(shape_to_svg_path("not-a-shape").is_none());
    }

    #[test]
    fn composite_shape_has_multiple_subpaths() {
        let path = shape_to_svg_path("square-cross").unwrap();
        assert!(path.matches('M').count() > 1);
    }
}
