//! Identity transform implementation (no transformation)

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{extended_breaks, linear_breaks, minor_breaks_linear};

/// Identity transform - no transformation (linear scale)
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl TransformTrait for Identity {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Identity
    }

    fn name(&self) -> &'static str {
        "identity"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        ZygosMath::of(TransformKind::Identity).allowed_domain()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
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
        ZygosMath::of(TransformKind::Identity).forward(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        ZygosMath::of(TransformKind::Identity).inverse(value)
    }
}

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_breaks_delegate_to_zygos() {
        // Break placement is zygos's contract and tested there; ggsql only
        // owns the delegation.
        let t = Identity;
        assert!(!t.calculate_breaks(0.0, 100.0, 5, true).is_empty());
    }

    #[test]
    fn test_identity_minor_breaks_delegate_to_zygos() {
        // Minor-break placement is likewise zygos's contract.
        let t = Identity;
        let majors = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        assert!(!t.calculate_minor_breaks(&majors, 1, None).is_empty());
        let majors = vec![25.0, 50.0, 75.0];
        assert!(!t
            .calculate_minor_breaks(&majors, 1, Some((0.0, 100.0)))
            .is_empty());
    }

    #[test]
    fn test_identity_default_minor_break_count() {
        let t = Identity;
        assert_eq!(t.default_minor_break_count(), 1);
    }
}
