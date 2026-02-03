//! Sqrt transform implementation (square root)

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{minor_breaks_sqrt, sqrt_breaks};

/// Sqrt transform - square root
///
/// Domain: [0, +∞) - non-negative values (includes 0)
#[derive(Debug, Clone, Copy)]
pub struct Sqrt;

impl TransformTrait for Sqrt {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Sqrt
    }

    fn name(&self) -> &'static str {
        "sqrt"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value >= 0.0 && value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        sqrt_breaks(min, max, n, pretty)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        minor_breaks_sqrt(major_breaks, n, range)
    }

    fn transform(&self, value: f64) -> f64 {
        value.sqrt()
    }

    fn inverse(&self, value: f64) -> f64 {
        value * value
    }
}

impl std::fmt::Display for Sqrt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_domain() {
        let t = Sqrt;
        let (min, max) = t.allowed_domain();
        assert_eq!(min, 0.0);
        assert!(max.is_infinite());
    }

    #[test]
    fn test_sqrt_is_value_in_domain() {
        let t = Sqrt;
        assert!(t.is_value_in_domain(0.0)); // sqrt includes 0
        assert!(t.is_value_in_domain(1.0));
        assert!(t.is_value_in_domain(100.0));
        assert!(!t.is_value_in_domain(-1.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
    }

    #[test]
    fn test_sqrt_transform() {
        let t = Sqrt;
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
        assert!((t.transform(1.0) - 1.0).abs() < 1e-10);
        assert!((t.transform(4.0) - 2.0).abs() < 1e-10);
        assert!((t.transform(9.0) - 3.0).abs() < 1e-10);
        assert!((t.transform(100.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_inverse() {
        let t = Sqrt;
        assert!((t.inverse(0.0) - 0.0).abs() < 1e-10);
        assert!((t.inverse(1.0) - 1.0).abs() < 1e-10);
        assert!((t.inverse(2.0) - 4.0).abs() < 1e-10);
        assert!((t.inverse(3.0) - 9.0).abs() < 1e-10);
        assert!((t.inverse(10.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_roundtrip() {
        let t = Sqrt;
        for &val in &[0.0, 1.0, 4.0, 9.0, 25.0, 100.0] {
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
    fn test_sqrt_breaks() {
        let t = Sqrt;
        let breaks = t.calculate_breaks(0.0, 100.0, 5, false);
        // linear_breaks now extends one step before and after
        // Negative values in sqrt space get clipped
        assert!(
            breaks.len() >= 5,
            "Should have at least 5 breaks, got {}",
            breaks.len()
        );
        // First break should be >= 0 (sqrt clips negatives)
        assert!(breaks.first().unwrap() >= &0.0);
        // Last break should be >= 100
        assert!(breaks.last().unwrap() >= &100.0);
    }
}
