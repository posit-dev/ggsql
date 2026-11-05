//! Asinh transform implementation (inverse hyperbolic sine)

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{extent_minor_breaks, symlog_breaks, symlog_minor_breaks};

// Branches either side of zero follow the base-10 1-2-5 pattern: asinh
// grows like a natural log, but powers of e would label as 2.718….
const BREAK_BASE: f64 = 10.0;

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
        ZygosMath::of(TransformKind::Asinh).allowed_domain()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, _pretty: bool) -> Vec<f64> {
        symlog_breaks(min, max, n, BREAK_BASE)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        extent_minor_breaks(major_breaks, n, range, |min, max| {
            symlog_minor_breaks(min, max, BREAK_BASE)
        })
    }

    fn default_minor_break_count(&self) -> usize {
        // The whole-number multiples between powers fix the density; any
        // non-zero count asks for them.
        8
    }

    fn transform(&self, value: f64) -> f64 {
        ZygosMath::of(TransformKind::Asinh).forward(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        ZygosMath::of(TransformKind::Asinh).inverse(value)
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
        let breaks = t.calculate_breaks(-1000.0, 1000.0, 10, true);
        // Decade 1-2-5 breaks on each branch, mirrored around zero.
        let positive = [
            0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
        ];
        let mut expected: Vec<f64> = positive.iter().rev().map(|v| -v).collect();
        expected.push(0.0);
        expected.extend(positive);
        assert_eq!(breaks, expected);
    }

    #[test]
    fn test_asinh_works_with_zero() {
        let t = Asinh;
        // Unlike log, asinh works with zero
        let breaks = t.calculate_breaks(0.0, 100.0, 5, true);
        assert_eq!(
            breaks,
            vec![0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        );
    }

    #[test]
    fn test_asinh_minor_breaks() {
        let t = Asinh;
        let minors = t.calculate_minor_breaks(&[-10.0, 0.0, 10.0], 8, Some((-20.0, 20.0)));
        // Whole-number multiples on each branch; the majors' own positions
        // are left to the majors.
        assert!(minors.contains(&-20.0) && minors.contains(&20.0));
        assert!(minors.contains(&2.0) && minors.contains(&-9.0));
        assert!(!minors.contains(&10.0) && !minors.contains(&-10.0));
        assert!(minors.windows(2).all(|w| w[0] < w[1]));
        assert!(t
            .calculate_minor_breaks(&[-10.0, 0.0, 10.0], 0, None)
            .is_empty());
    }
}
