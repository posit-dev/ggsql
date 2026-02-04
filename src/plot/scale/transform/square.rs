//! Square transform implementation (x²) - inverse of sqrt

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{linear_breaks, minor_breaks_linear};

/// Square transform (x²) - inverse of sqrt
///
/// Domain: (-∞, +∞) - all real numbers
/// Range: [0, +∞) - non-negative values
#[derive(Debug, Clone, Copy)]
pub struct Square;

impl TransformTrait for Square {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Square
    }

    fn name(&self) -> &'static str {
        "square"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, _pretty: bool) -> Vec<f64> {
        // Data-space even breaks (simple linear breaks in input space)
        // These won't be visually evenly spaced after squaring, but that's expected
        linear_breaks(min, max, n)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        minor_breaks_linear(major_breaks, n, range)
    }

    fn transform(&self, value: f64) -> f64 {
        value * value
    }

    fn inverse(&self, value: f64) -> f64 {
        if value >= 0.0 {
            value.sqrt()
        } else {
            -((-value).sqrt())
        }
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_domain() {
        let t = Square;
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min < 0.0);
        assert!(max.is_infinite() && max > 0.0);
    }

    #[test]
    fn test_square_is_value_in_domain() {
        let t = Square;
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(1.0));
        assert!(t.is_value_in_domain(-1.0));
        assert!(t.is_value_in_domain(100.0));
        assert!(t.is_value_in_domain(-100.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
        assert!(!t.is_value_in_domain(f64::NAN));
    }

    #[test]
    fn test_square_transform() {
        let t = Square;
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
        assert!((t.transform(1.0) - 1.0).abs() < 1e-10);
        assert!((t.transform(2.0) - 4.0).abs() < 1e-10);
        assert!((t.transform(3.0) - 9.0).abs() < 1e-10);
        assert!((t.transform(10.0) - 100.0).abs() < 1e-10);
        // Negative values also square to positive
        assert!((t.transform(-2.0) - 4.0).abs() < 1e-10);
        assert!((t.transform(-3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_square_inverse() {
        let t = Square;
        assert!((t.inverse(0.0) - 0.0).abs() < 1e-10);
        assert!((t.inverse(1.0) - 1.0).abs() < 1e-10);
        assert!((t.inverse(4.0) - 2.0).abs() < 1e-10);
        assert!((t.inverse(9.0) - 3.0).abs() < 1e-10);
        assert!((t.inverse(100.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_square_roundtrip_positive() {
        let t = Square;
        for &val in &[0.0, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            if val == 0.0 {
                assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
            } else {
                assert!(
                    (back - val).abs() / val < 1e-10,
                    "Roundtrip failed for {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_square_is_inverse_of_sqrt() {
        // Verify that Square::transform is the same as Sqrt::inverse
        use super::super::Sqrt;
        let square = Square;
        let sqrt = Sqrt;

        for &val in &[0.0_f64, 1.0, 4.0, 9.0, 25.0, 100.0] {
            assert!(
                (square.transform(val.sqrt()) - sqrt.inverse(val.sqrt())).abs() < 1e-10,
                "Square::transform != Sqrt::inverse for {}",
                val.sqrt()
            );
        }
    }

    #[test]
    fn test_square_inverse_is_sqrt_transform() {
        // Verify that Square::inverse is the same as Sqrt::transform
        use super::super::Sqrt;
        let square = Square;
        let sqrt = Sqrt;

        for &val in &[0.0, 1.0, 4.0, 9.0, 25.0, 100.0] {
            assert!(
                (square.inverse(val) - sqrt.transform(val)).abs() < 1e-10,
                "Square::inverse != Sqrt::transform for {}",
                val
            );
        }
    }

    #[test]
    fn test_square_breaks() {
        let t = Square;
        let breaks = t.calculate_breaks(0.0, 10.0, 5, false);
        // linear_breaks gives exact coverage from min to max
        assert_eq!(breaks.len(), 5, "Should have exactly 5 breaks");
        // First break should be at 0
        assert!(
            (breaks.first().unwrap() - 0.0).abs() < 1e-10,
            "First break should be at 0"
        );
        // Last break should be at 10
        assert!(
            (breaks.last().unwrap() - 10.0).abs() < 1e-10,
            "Last break should be at 10"
        );
    }

    #[test]
    fn test_square_kind_and_name() {
        let t = Square;
        assert_eq!(t.transform_kind(), TransformKind::Square);
        assert_eq!(t.name(), "square");
    }

    #[test]
    fn test_square_display() {
        assert_eq!(format!("{}", Square), "square");
    }
}
