//! Sqrt transform implementation (square root)

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{linear_breaks, minor_breaks_linear, sqrt_breaks};

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
        ZygosMath::of(TransformKind::Sqrt).allowed_domain()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        if pretty {
            sqrt_breaks(min, max, n)
        } else {
            // Evenly spaced in sqrt space, squared back.
            let min = min.max(0.0);
            if min >= max {
                return vec![];
            }
            linear_breaks(min.sqrt(), max.sqrt(), n)
                .into_iter()
                .map(|v| v * v)
                .collect()
        }
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
        ZygosMath::of(TransformKind::Sqrt).forward(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        ZygosMath::of(TransformKind::Sqrt).inverse(value)
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
    fn test_sqrt_breaks() {
        let t = Sqrt;
        // Nice numbers in sqrt space, squared back.
        assert_eq!(
            t.calculate_breaks(0.0, 100.0, 5, true),
            vec![0.0, 4.0, 16.0, 36.0, 64.0, 100.0]
        );
        // Evenly spaced in sqrt space, squared back.
        assert_eq!(
            t.calculate_breaks(0.0, 100.0, 5, false),
            vec![0.0, 6.25, 25.0, 56.25, 100.0]
        );
    }
}
