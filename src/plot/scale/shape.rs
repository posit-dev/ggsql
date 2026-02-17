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
        "hline" => Some(hline_coords()),
        "vline" => Some(vline_coords()),
        "asterisk" => Some(asterisk_coords()),
        "bowtie" => Some(bowtie_coords()),
        // Composite shapes with cutouts (use evenodd fill-rule)
        "square-cross" => Some(square_cross_coords()),
        "circle-plus" => Some(circle_plus_coords()),
        "square-plus" => Some(square_plus_coords()),
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

// =============================================================================
// Area-equalized shape coordinates
// =============================================================================
// All closed shapes are scaled to have approximately equal visual area.
// Reference: circle with radius 0.8 has area π×0.8² ≈ 2.01
//
// Scale factors derived from:
// - Square: area = 4s² → s = √(π×0.8²/4) ≈ 0.71
// - Diamond: area = 2d² → d = √(π×0.8²/2) ≈ 0.89
// - Triangle: area = √3×r² (equilateral) → needs ~15% larger radius
// - Star: area ≈ 0.5 × circle → needs ~40% larger radius

/// Circle approximated with 32-point polygon.
/// Radius 0.8 centered at origin. Area ≈ 2.01 (reference shape).
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

/// Square scaled for equal area with circle.
/// Half-side 0.71 gives area ≈ 2.01.
fn square_coords() -> Vec<Vec<(f64, f64)>> {
    let s = 0.71; // √(π×0.8²/4) ≈ 0.709
    vec![vec![(-s, -s), (s, -s), (s, s), (-s, s)]]
}

/// Diamond (square rotated 45 degrees) scaled for equal area.
/// Half-diagonal 0.89 gives area ≈ 2.01.
fn diamond_coords() -> Vec<Vec<(f64, f64)>> {
    let d = 0.89; // √(π×0.8²/2) ≈ 0.892
    vec![vec![(0.0, -d), (d, 0.0), (0.0, d), (-d, 0.0)]]
}

/// Triangle pointing up, scaled for equal area.
/// Base and height adjusted for area ≈ 2.01.
fn triangle_up_coords() -> Vec<Vec<(f64, f64)>> {
    // Equilateral-ish triangle: area = 0.5 × base × height
    // For area 2.01 with base = 2r and height = 1.5r: r ≈ 0.92
    let r = 0.92;
    let h = r * 0.75; // height offset from center
    vec![vec![(0.0, -r), (r, h), (-r, h)]]
}

/// Triangle pointing down, scaled for equal area.
fn triangle_down_coords() -> Vec<Vec<(f64, f64)>> {
    let r = 0.92;
    let h = r * 0.75;
    vec![vec![(-r, -h), (r, -h), (0.0, r)]]
}

/// 5-pointed star scaled for equal area.
/// Outer radius increased to compensate for inner concavity.
fn star_coords() -> Vec<Vec<(f64, f64)>> {
    // Star area ≈ 0.5 × circle area for same outer radius
    // Scale up by √2 ≈ 1.41 to compensate
    let outer_radius = 0.95; // 0.8 × ~1.19, capped to stay in bounds
    let inner_radius = outer_radius * 0.4; // maintain proportions
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
/// Scaled by 1/√2 so diagonal length matches the plus's axis-aligned length.
fn cross_coords() -> Vec<Vec<(f64, f64)>> {
    // 0.8 / √2 ≈ 0.566 gives same line length as plus (1.6 units)
    let c = 0.8 / std::f64::consts::SQRT_2;
    vec![
        vec![(-c, -c), (c, c)], // diagonal from bottom-left to top-right
        vec![(-c, c), (c, -c)], // diagonal from top-left to bottom-right
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
fn hline_coords() -> Vec<Vec<(f64, f64)>> {
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

/// Square divided into 4 triangles by X-shaped gap.
/// Creates 4 separate triangular pieces (top, right, bottom, left).
fn square_cross_coords() -> Vec<Vec<(f64, f64)>> {
    let s = 0.71; // area-equalized square size
    let g = 0.12; // half-gap width (perpendicular to diagonal)

    // 4 triangles pointing inward from each edge
    vec![
        // Top triangle
        vec![(-s + g, -s), (s - g, -s), (0.0, -g)],
        // Right triangle
        vec![(s, -s + g), (s, s - g), (g, 0.0)],
        // Bottom triangle
        vec![(s - g, s), (-s + g, s), (0.0, g)],
        // Left triangle
        vec![(-s, s - g), (-s, -s + g), (-g, 0.0)],
    ]
}

/// Circle divided into 4 quarters by +-shaped gap with constant width.
/// Creates 4 separate quarter-circle pieces.
fn circle_plus_coords() -> Vec<Vec<(f64, f64)>> {
    let r: f64 = 0.8; // circle radius
    let g: f64 = 0.12 / std::f64::consts::SQRT_2; // half-gap width, scaled to match X visually
    let n = 8; // points per quarter arc

    // Where the circle intersects the gap edge: sqrt(r² - g²)
    let edge = (r * r - g * g).sqrt();

    // Start and end angles for the arc (where circle intersects gap)
    let start_angle = (g / r).asin(); // angle where y = g on circle
    let end_angle = std::f64::consts::FRAC_PI_2 - start_angle; // angle where x = g

    let mut quarters = Vec::new();

    for q in 0..4 {
        let base_angle = (q as f64) * std::f64::consts::FRAC_PI_2;
        let mut points = Vec::new();

        // Inner corner point
        let (cx, cy) = match q {
            0 => (g, g),   // top-right
            1 => (-g, g),  // top-left
            2 => (-g, -g), // bottom-left
            _ => (g, -g),  // bottom-right
        };
        points.push((cx, cy));

        // Point where gap meets circle (start of arc)
        let (sx, sy) = match q {
            0 => (edge, g),   // right edge of horizontal gap
            1 => (-g, edge),  // top edge of vertical gap
            2 => (-edge, -g), // left edge of horizontal gap
            _ => (g, -edge),  // bottom edge of vertical gap
        };
        points.push((sx, sy));

        // Arc points
        let arc_start = base_angle + start_angle;
        let arc_span = end_angle - start_angle;
        for i in 0..=n {
            let t = (i as f64) / (n as f64);
            let angle = arc_start + t * arc_span;
            points.push((r * angle.cos(), r * angle.sin()));
        }

        // Point where arc meets gap (end of arc)
        let (ex, ey) = match q {
            0 => (g, edge),   // top edge of vertical gap
            1 => (-edge, g),  // left edge of horizontal gap
            2 => (-g, -edge), // bottom edge of vertical gap
            _ => (edge, -g),  // right edge of horizontal gap
        };
        points.push((ex, ey));

        quarters.push(points);
    }

    quarters
}

/// Square divided into 4 smaller squares by +-shaped gap.
/// Creates 4 separate square pieces in each corner.
fn square_plus_coords() -> Vec<Vec<(f64, f64)>> {
    let s = 0.71; // area-equalized square size
    let g = 0.12 / std::f64::consts::SQRT_2; // half-gap width, scaled to match X visually

    // 4 smaller squares in each corner
    vec![
        // Top-left square
        vec![(-s, -s), (-g, -s), (-g, -g), (-s, -g)],
        // Top-right square
        vec![(g, -s), (s, -s), (s, -g), (g, -g)],
        // Bottom-right square
        vec![(g, g), (s, g), (s, s), (g, s)],
        // Bottom-left square
        vec![(-s, g), (-g, g), (-g, s), (-s, s)],
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
        assert!(get_shape_coordinates("hline").is_some());
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
        let hline = shape_to_svg_path("hline").unwrap();
        assert!(!hline.ends_with('Z'));

        let vline = shape_to_svg_path("vline").unwrap();
        assert!(!vline.ends_with('Z'));

        let cross = shape_to_svg_path("cross").unwrap();
        assert!(!cross.ends_with('Z'));

        let plus = shape_to_svg_path("plus").unwrap();
        assert!(!plus.ends_with('Z'));
    }
}
