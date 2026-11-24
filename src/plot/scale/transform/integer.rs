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
    fn test_integer_breaks_rounded() {
        let t = Integer;
        // Breaks should be rounded to integers
        let breaks = t.calculate_breaks(0.0, 100.0, 5, true);
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be rounded", b);
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
