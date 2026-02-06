//! Asinh transform implementation (inverse hyperbolic sine)

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{minor_breaks_symlog, symlog_breaks};

/// Asinh transform - inverse hyperbolic sine
///
/// Domain: (-∞, +∞) - all real numbers
///
/// The asinh transform is useful for data that spans multiple orders of
/// magnitude and includes zero or negative values. It behaves like log
/// for large values but is well-defined for zero and negative values.
///
/// Formula: asinh(x) = ln(x + sqrt(x² + 1))
#[derive(Debug, Clone, Copy)]
pub struct Asinh;

impl TransformTrait for Asinh {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Asinh
    }

    fn name(&self) -> &'static str {
        "asinh"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        symlog_breaks(min, max, n, pretty)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        minor_breaks_symlog(major_breaks, n, range)
    }

    fn default_minor_break_count(&self) -> usize {
        8 // Similar density to traditional 2-9 pattern on log axes
    }

    fn transform(&self, value: f64) -> f64 {
        value.asinh()
    }

    fn inverse(&self, value: f64) -> f64 {
        value.sinh()
    }
}

impl std::fmt::Display for Asinh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asinh_domain() {
        let t = Asinh;
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_asinh_transform() {
        let t = Asinh;
        // asinh(0) = 0
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
        // asinh is odd function
        assert!((t.transform(-1.0) + t.transform(1.0)).abs() < 1e-10);
        // For large values, asinh(x) ≈ ln(2x)
        let large: f64 = 1000.0;
        let expected = (2.0 * large).ln();
        assert!((t.transform(large) - expected).abs() < 0.01);
    }

    #[test]
    fn test_asinh_inverse() {
        let t = Asinh;
        // sinh(0) = 0
        assert!((t.inverse(0.0) - 0.0).abs() < 1e-10);
        // sinh is odd function
        assert!((t.inverse(-1.0) + t.inverse(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_asinh_roundtrip() {
        let t = Asinh;
        for &val in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            if val == 0.0 {
                assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
            } else {
                assert!(
                    (back - val).abs() / val.abs() < 1e-10,
                    "Roundtrip failed for {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_asinh_breaks_symmetric() {
        let t = Asinh;
        let breaks = t.calculate_breaks(-1000.0, 1000.0, 10, false);
        // Should have negative, zero, and positive values
        assert!(breaks.contains(&0.0));
        assert!(breaks.iter().any(|&v| v < 0.0));
        assert!(breaks.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_asinh_works_with_zero() {
        let t = Asinh;
        // Unlike log, asinh works with zero
        let breaks = t.calculate_breaks(0.0, 100.0, 5, false);
        assert!(!breaks.is_empty());
    }
}
