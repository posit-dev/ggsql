/// Get normalized coordinates for a shape.
/// Returns `Vec<Vec<(f64, f64)>>` where each inner Vec is a path/polygon.
/// Coordinates are normalized to [-1, 1] range centered at origin (0, 0).
/// This is the format expected by Vega-Lite SVG paths.
///
/// # Examples
/// - Simple shapes (circle, square): Single path `vec![vec![(x1,y1), (x2,y2), ...]]`
/// - Composite shapes (square-cross): Multiple paths `vec![vec![square coords], vec![cross coords]]`
pub fn get_shape_coordinates(name: &str) -> Option<Vec<Vec<(f64, f64)>>> {
    match name.to_lowercase().as_str() {
        "circle" => Some(circle_coords()),
        "square" => Some(square_coords()),
        "diamond" => Some(diamond_coords()),
        "triangle-up" => Some(triangle_up_coords()),
        "triangle-down" => Some(triangle_down_coords()),
        "star" => Some(star_coords()),
        "cross" => Some(cross_coords()),
        "plus" => Some(plus_coords()),
        "stroke" => Some(stroke_coords()),
        "vline" => Some(vline_coords()),
        "asterisk" => Some(asterisk_coords()),
        "bowtie" => Some(bowtie_coords()),
        // Composite shapes
        "square-cross" => Some(combine_shapes(square_coords(), cross_coords())),
        "circle-plus" => Some(combine_shapes(circle_coords(), plus_coords())),
        "square-plus" => Some(combine_shapes(square_coords(), plus_coords())),
        _ => None,
    }
}

/// Convert shape coordinates to SVG path string for Vega-Lite.
/// Coordinates are in [-1, 1] range centered at origin.
///
/// Returns None for unknown shapes.
pub fn shape_to_svg_path(name: &str) -> Option<String> {
    let paths = get_shape_coordinates(name)?;

    let svg_paths: Vec<String> = paths
        .iter()
        .map(|path| {
            let mut svg = String::new();
            for (i, &(x, y)) in path.iter().enumerate() {
                let cmd = if i == 0 { "M" } else { "L" };
                svg.push_str(&format!("{}{:.3},{:.3} ", cmd, x, y));
            }
            // Close path for polygons (3+ points)
            if path.len() >= 3 {
                svg.push('Z');
            }
            svg.trim().to_string()
        })
        .collect();

    Some(svg_paths.join(" "))
}

/// Combine two shapes' coordinate sets into one.
fn combine_shapes(a: Vec<Vec<(f64, f64)>>, b: Vec<Vec<(f64, f64)>>) -> Vec<Vec<(f64, f64)>> {
    let mut result = a;
    result.extend(b);
    result
}

/// Circle approximated with 32-point polygon.
/// Radius 0.8 centered at origin.
fn circle_coords() -> Vec<Vec<(f64, f64)>> {
    let n = 32;
    let radius = 0.8;
    let points: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    vec![points]
}

/// Square with corners at (-0.8, -0.8) to (0.8, 0.8).
fn square_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)]]
}

/// Diamond (square rotated 45 degrees).
fn diamond_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(0.0, -0.8), (0.8, 0.0), (0.0, 0.8), (-0.8, 0.0)]]
}

/// Triangle pointing up.
fn triangle_up_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(0.0, -0.8), (0.8, 0.8), (-0.8, 0.8)]]
}

/// Triangle pointing down.
fn triangle_down_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)]]
}

/// 5-pointed star with alternating outer (0.8) and inner (0.4) radii.
fn star_coords() -> Vec<Vec<(f64, f64)>> {
    let outer_radius = 0.8;
    let inner_radius = 0.4;
    let points: Vec<(f64, f64)> = (0..10)
        .map(|i| {
            // Start from top (-PI/2) and go clockwise
            let angle = -std::f64::consts::PI / 2.0 + std::f64::consts::PI * (i as f64) / 5.0;
            let radius = if i % 2 == 0 {
                outer_radius
            } else {
                inner_radius
            };
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    vec![points]
}

/// X shape (diagonal cross) - two line segments.
fn cross_coords() -> Vec<Vec<(f64, f64)>> {
    vec![
        vec![(-0.8, -0.8), (0.8, 0.8)], // diagonal from bottom-left to top-right
        vec![(-0.8, 0.8), (0.8, -0.8)], // diagonal from top-left to bottom-right
    ]
}

/// + shape (axis-aligned cross) - two line segments.
fn plus_coords() -> Vec<Vec<(f64, f64)>> {
    vec![
        vec![(-0.8, 0.0), (0.8, 0.0)], // horizontal line
        vec![(0.0, -0.8), (0.0, 0.8)], // vertical line
    ]
}

/// Horizontal line at y=0.
fn stroke_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(-0.8, 0.0), (0.8, 0.0)]]
}

/// Vertical line at x=0.
fn vline_coords() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(0.0, -0.8), (0.0, 0.8)]]
}

/// Asterisk (*) - three lines through center.
fn asterisk_coords() -> Vec<Vec<(f64, f64)>> {
    vec![
        vec![(-0.8, 0.0), (0.8, 0.0)],  // horizontal
        vec![(-0.6, -0.7), (0.6, 0.7)], // diagonal /
        vec![(-0.6, 0.7), (0.6, -0.7)], // diagonal \
    ]
}

/// Bowtie - two triangles meeting at center.
fn bowtie_coords() -> Vec<Vec<(f64, f64)>> {
    vec![
        vec![(-0.8, -0.8), (0.0, 0.0), (-0.8, 0.8)], // left triangle
        vec![(0.8, -0.8), (0.0, 0.0), (0.8, 0.8)],   // right triangle
    ]
}

#[cfg(test)]
mod tests {
    use super::{get_shape_coordinates, shape_to_svg_path};
    use crate::plot::palettes::SHAPES;

    #[test]
    fn test_get_shape_coordinates_simple_shapes() {
        // Simple closed shapes return single path
        assert_eq!(get_shape_coordinates("circle").unwrap().len(), 1);
        assert_eq!(get_shape_coordinates("square").unwrap().len(), 1);
        assert_eq!(get_shape_coordinates("diamond").unwrap().len(), 1);
        assert_eq!(get_shape_coordinates("triangle-up").unwrap().len(), 1);
        assert_eq!(get_shape_coordinates("triangle-down").unwrap().len(), 1);
        assert_eq!(get_shape_coordinates("star").unwrap().len(), 1);
    }

    #[test]
    fn test_get_shape_coordinates_open_shapes() {
        // Open/stroke shapes may have multiple line segments
        assert!(get_shape_coordinates("cross").is_some());
        assert!(get_shape_coordinates("plus").is_some());
        assert!(get_shape_coordinates("stroke").is_some());
        assert!(get_shape_coordinates("vline").is_some());
        assert!(get_shape_coordinates("asterisk").is_some());
        assert!(get_shape_coordinates("bowtie").is_some());
    }

    #[test]
    fn test_get_shape_coordinates_composite_shapes() {
        // Composite shapes return multiple paths (base + overlay)
        let sq_cross = get_shape_coordinates("square-cross").unwrap();
        assert!(
            sq_cross.len() > 1,
            "square-cross should have multiple paths"
        );

        let circ_plus = get_shape_coordinates("circle-plus").unwrap();
        assert!(
            circ_plus.len() > 1,
            "circle-plus should have multiple paths"
        );

        let sq_plus = get_shape_coordinates("square-plus").unwrap();
        assert!(sq_plus.len() > 1, "square-plus should have multiple paths");
    }

    #[test]
    fn test_get_shape_coordinates_all_shapes_supported() {
        // All shapes in the SHAPES palette should have coordinates
        for shape in SHAPES.iter() {
            assert!(
                get_shape_coordinates(shape).is_some(),
                "Shape '{}' should have coordinates",
                shape
            );
        }
    }

    #[test]
    fn test_get_shape_coordinates_normalized() {
        // All coordinates should be in [-1, 1] range
        for shape in SHAPES.iter() {
            if let Some(paths) = get_shape_coordinates(shape) {
                for path in &paths {
                    for &(x, y) in path {
                        assert!((-1.0..=1.0).contains(&x), "{} x={} out of range", shape, x);
                        assert!((-1.0..=1.0).contains(&y), "{} y={} out of range", shape, y);
                    }
                }
            }
        }
    }

    #[test]
    fn test_get_shape_coordinates_unknown() {
        assert!(get_shape_coordinates("unknown_shape").is_none());
    }

    #[test]
    fn test_get_shape_coordinates_case_insensitive() {
        assert!(get_shape_coordinates("CIRCLE").is_some());
        assert!(get_shape_coordinates("Square").is_some());
        assert!(get_shape_coordinates("TRIANGLE-UP").is_some());
    }

    #[test]
    fn test_shape_to_svg_path_square() {
        let path = shape_to_svg_path("square").unwrap();
        assert!(path.starts_with('M'));
        assert!(path.contains('L'));
        assert!(path.ends_with('Z'));
    }

    #[test]
    fn test_shape_to_svg_path_all_shapes() {
        for shape in SHAPES.iter() {
            assert!(
                shape_to_svg_path(shape).is_some(),
                "Shape '{}' should produce SVG path",
                shape
            );
        }
    }

    #[test]
    fn test_shape_to_svg_path_unknown() {
        assert!(shape_to_svg_path("unknown").is_none());
    }

    #[test]
    fn test_shape_to_svg_path_composite() {
        let path = shape_to_svg_path("square-cross").unwrap();
        // Should contain multiple M commands (one per sub-path)
        assert!(path.matches('M').count() > 1);
    }

    #[test]
    fn test_shape_to_svg_path_open_shapes_not_closed() {
        // Open shapes (lines) should NOT end with Z
        let stroke = shape_to_svg_path("stroke").unwrap();
        assert!(!stroke.ends_with('Z'));

        let vline = shape_to_svg_path("vline").unwrap();
        assert!(!vline.ends_with('Z'));

        let cross = shape_to_svg_path("cross").unwrap();
        assert!(!cross.ends_with('Z'));

        let plus = shape_to_svg_path("plus").unwrap();
        assert!(!plus.ends_with('Z'));
    }
}
