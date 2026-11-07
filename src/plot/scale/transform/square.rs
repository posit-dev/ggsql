//! Square transform implementation (x²) - inverse of sqrt

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{extended_breaks, linear_breaks, minor_breaks_linear};

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
        ZygosMath::of(TransformKind::Square).allowed_domain()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        // Breaks in data space: they won't be visually evenly spaced after
        // squaring, but their labels stay round.
        if pretty {
            extended_breaks(min, max, n)
        } else {
            linear_breaks(min, max, n)
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
        ZygosMath::of(TransformKind::Square).forward(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        // Sign-preserving, unlike zygos's non-negative branch, so a domain padded
        // below zero in transformed space maps back below zero rather than
        // folding onto its mirror image.
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
    fn test_square_breaks_pretty() {
        let t = Square;
        assert_eq!(
            t.calculate_breaks(0.0, 10.0, 5, true),
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
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
