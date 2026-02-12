//! Integer transform implementation (linear with integer rounding)

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{integer_breaks, minor_breaks_linear};
use crate::plot::ArrayElement;

/// Integer transform - linear scale with integer rounding
///
/// This transform works like Identity (linear) but signals that the data
/// should be cast to integer type in SQL. The transform and inverse are
/// identity functions, but breaks are rounded to integers.
#[derive(Debug, Clone, Copy)]
pub struct Integer;

impl TransformTrait for Integer {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Integer
    }

    fn name(&self) -> &'static str {
        "integer"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        // Use dedicated integer breaks function for proper even spacing
        integer_breaks(min, max, n, pretty)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        // For integer scales, minor breaks should also be integers
        // Filter out any that would round to a major break
        let minors = minor_breaks_linear(major_breaks, n, range);
        let rounded: Vec<f64> = minors.iter().map(|v| v.round()).collect();

        // Deduplicate and remove any that coincide with major breaks
        let mut result = Vec::new();
        for r in rounded {
            if !major_breaks.iter().any(|&m| (m - r).abs() < 0.5)
                && !result.iter().any(|&v: &f64| (v - r).abs() < 0.5)
            {
                result.push(r);
            }
        }
        result
    }

    fn transform(&self, value: f64) -> f64 {
        value
    }

    fn inverse(&self, value: f64) -> f64 {
        value
    }

    fn wrap_numeric(&self, value: f64) -> ArrayElement {
        ArrayElement::Number(value.round())
    }
}

impl std::fmt::Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_domain() {
        let t = Integer;
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_integer_transform() {
        let t = Integer;
        assert_eq!(t.transform(1.0), 1.0);
        assert_eq!(t.transform(-5.0), -5.0);
        assert_eq!(t.transform(0.0), 0.0);
        assert_eq!(t.transform(100.5), 100.5);
    }

    #[test]
    fn test_integer_inverse() {
        let t = Integer;
        assert_eq!(t.inverse(1.0), 1.0);
        assert_eq!(t.inverse(-5.0), -5.0);
    }

    #[test]
    fn test_integer_roundtrip() {
        let t = Integer;
        for &val in &[0.0, 1.0, -1.0, 100.0, -100.0, 0.001] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_integer_breaks_rounded() {
        let t = Integer;
        // Breaks should be rounded to integers
        let breaks = t.calculate_breaks(0.0, 100.0, 5, true);
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be rounded", b);
        }
    }

    #[test]
    fn test_integer_breaks_evenly_spaced() {
        let t = Integer;
        // Breaks should be evenly spaced (all gaps equal)
        let breaks = t.calculate_breaks(0.0, 100.0, 5, true);
        if breaks.len() >= 2 {
            let step = breaks[1] - breaks[0];
            for i in 1..breaks.len() {
                let gap = breaks[i] - breaks[i - 1];
                assert!(
                    (gap - step).abs() < 0.01,
                    "Uneven spacing: gap {} != step {} at index {}",
                    gap,
                    step,
                    i
                );
            }
        }
    }

    #[test]
    fn test_integer_breaks_small_range() {
        let t = Integer;
        // Small range should give consecutive integers
        let breaks = t.calculate_breaks(0.0, 5.0, 5, true);
        // Should be [0, 1, 2, 3, 4, 5] or similar consecutive sequence
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
        // Verify even spacing
        if breaks.len() >= 2 {
            let step = breaks[1] - breaks[0];
            for i in 1..breaks.len() {
                let gap = breaks[i] - breaks[i - 1];
                assert!(
                    (gap - step).abs() < 0.01,
                    "Uneven spacing in small range: gap {} != step {}",
                    gap,
                    step
                );
            }
        }
    }

    #[test]
    fn test_integer_breaks_small_range_linear() {
        let t = Integer;
        // Test the problematic case: range 0-5 with n=5
        // Previously this would give [0, 1.25, 2.5, 3.75, 5] → rounded [0, 1, 3, 4, 5]
        // Now it should give evenly spaced integers
        let breaks = t.calculate_breaks(0.0, 5.0, 5, false);
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
        // The breaks should not have the "2.5 rounds to 3" problem
        // i.e., should not skip 2 and have both 1 and 3
        if breaks.contains(&1.0) && breaks.contains(&3.0) {
            assert!(
                breaks.contains(&2.0),
                "Should include 2.0, got {:?}",
                breaks
            );
        }
    }

    #[test]
    fn test_integer_wrap_numeric() {
        let t = Integer;
        // wrap_numeric should round to integer
        assert_eq!(t.wrap_numeric(5.5), ArrayElement::Number(6.0));
        assert_eq!(t.wrap_numeric(5.4), ArrayElement::Number(5.0));
        assert_eq!(t.wrap_numeric(-2.7), ArrayElement::Number(-3.0));
    }

    #[test]
    fn test_integer_default_minor_break_count() {
        let t = Integer;
        assert_eq!(t.default_minor_break_count(), 1);
    }
}
