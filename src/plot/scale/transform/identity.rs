//! Identity transform implementation (no transformation)

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{linear_breaks, pretty_breaks};

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
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        if pretty {
            pretty_breaks(min, max, n)
        } else {
            linear_breaks(min, max, n)
        }
    }

    fn transform(&self, value: f64) -> f64 {
        value
    }

    fn inverse(&self, value: f64) -> f64 {
        value
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
    fn test_identity_domain() {
        let t = Identity;
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_identity_is_value_in_domain() {
        let t = Identity;
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(-1000.0));
        assert!(t.is_value_in_domain(1000.0));
        assert!(t.is_value_in_domain(0.00001));
        assert!(!t.is_value_in_domain(f64::INFINITY));
        assert!(!t.is_value_in_domain(f64::NAN));
    }

    #[test]
    fn test_identity_transform() {
        let t = Identity;
        assert_eq!(t.transform(1.0), 1.0);
        assert_eq!(t.transform(-5.0), -5.0);
        assert_eq!(t.transform(0.0), 0.0);
        assert_eq!(t.transform(100.0), 100.0);
    }

    #[test]
    fn test_identity_inverse() {
        let t = Identity;
        assert_eq!(t.inverse(1.0), 1.0);
        assert_eq!(t.inverse(-5.0), -5.0);
    }

    #[test]
    fn test_identity_roundtrip() {
        let t = Identity;
        for &val in &[0.0, 1.0, -1.0, 100.0, -100.0, 0.001] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_identity_breaks_pretty() {
        let t = Identity;
        let breaks = t.calculate_breaks(0.0, 100.0, 5, true);
        assert!(!breaks.is_empty());
        // Pretty breaks should produce nice numbers
    }

    #[test]
    fn test_identity_breaks_linear() {
        let t = Identity;
        let breaks = t.calculate_breaks(0.0, 100.0, 5, false);
        assert_eq!(breaks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }
}
