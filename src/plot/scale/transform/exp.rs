//! Exponential transform implementation (base^x) - inverse of log

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{exp_pretty_breaks, linear_breaks, minor_breaks_linear};

/// Exponential transform (base^x) - inverse of log
///
/// Domain: (-∞, +∞) - all real numbers
/// Range: (0, +∞) - positive values
#[derive(Debug, Clone, Copy)]
pub struct Exp {
    base: f64,
}

impl Exp {
    /// Create an exponential transform with the given base
    pub fn new(base: f64) -> Self {
        assert!(
            base > 0.0 && base != 1.0,
            "Exp base must be positive and not 1"
        );
        Self { base }
    }

    /// Create a base-10 exponential transform (10^x) - inverse of log10
    pub fn base10() -> Self {
        Self { base: 10.0 }
    }

    /// Create a base-2 exponential transform (2^x) - inverse of log2
    pub fn base2() -> Self {
        Self { base: 2.0 }
    }

    /// Create a natural exponential transform (e^x) - inverse of ln
    pub fn natural() -> Self {
        Self {
            base: std::f64::consts::E,
        }
    }

    /// Get the base of this exponential
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Check if this is a base-10 exp (within floating point tolerance)
    fn is_base10(&self) -> bool {
        (self.base - 10.0).abs() < 1e-10
    }

    /// Check if this is a base-2 exp (within floating point tolerance)
    fn is_base2(&self) -> bool {
        (self.base - 2.0).abs() < 1e-10
    }

    /// Check if this is a natural exp (within floating point tolerance)
    fn is_natural(&self) -> bool {
        (self.base - std::f64::consts::E).abs() < 1e-10
    }
}

impl TransformTrait for Exp {
    fn transform_kind(&self) -> TransformKind {
        if self.is_base10() {
            TransformKind::Exp10
        } else if self.is_base2() {
            TransformKind::Exp2
        } else {
            // Natural exp and any other base map to Exp
            TransformKind::Exp
        }
    }

    fn name(&self) -> &'static str {
        if self.is_base10() {
            "exp10"
        } else if self.is_base2() {
            "exp2"
        } else {
            "exp"
        }
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        // Breaks are in data space (exponents), not output space
        if pretty && (self.is_base10() || self.is_base2()) {
            exp_pretty_breaks(min, max, n, self.base)
        } else {
            // Default: nice linear breaks (0, 1, 2, 3, ...)
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
        self.base.powf(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        value.log(self.base)
    }
}

impl std::fmt::Display for Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    // ==================== Base-10 (Exp10) Tests ====================

    #[test]
    fn test_exp10_domain() {
        let t = Exp::base10();
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min < 0.0);
        assert!(max.is_infinite() && max > 0.0);
    }

    #[test]
    fn test_exp10_is_value_in_domain() {
        let t = Exp::base10();
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(1.0));
        assert!(t.is_value_in_domain(-1.0));
        assert!(t.is_value_in_domain(100.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
        assert!(!t.is_value_in_domain(f64::NAN));
    }

    #[test]
    fn test_exp10_transform() {
        let t = Exp::base10();
        assert!((t.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((t.transform(1.0) - 10.0).abs() < 1e-10);
        assert!((t.transform(2.0) - 100.0).abs() < 1e-10);
        assert!((t.transform(3.0) - 1000.0).abs() < 1e-10);
        assert!((t.transform(-1.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_exp10_inverse() {
        let t = Exp::base10();
        assert!((t.inverse(1.0) - 0.0).abs() < 1e-10);
        assert!((t.inverse(10.0) - 1.0).abs() < 1e-10);
        assert!((t.inverse(100.0) - 2.0).abs() < 1e-10);
        assert!((t.inverse(0.1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exp10_roundtrip() {
        let t = Exp::base10();
        for &val in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
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
    fn test_exp10_is_inverse_of_log10() {
        // Verify that Exp10::transform is the same as Log10::inverse
        use super::super::Log;
        let exp10 = Exp::base10();
        let log10 = Log::base10();

        for &val in &[-1.0, 0.0, 1.0, 2.0, 3.0] {
            assert!(
                (exp10.transform(val) - log10.inverse(val)).abs() < 1e-10,
                "Exp10::transform != Log10::inverse for {}",
                val
            );
        }
    }

    #[test]
    fn test_exp10_inverse_is_log10_transform() {
        // Verify that Exp10::inverse is the same as Log10::transform
        use super::super::Log;
        let exp10 = Exp::base10();
        let log10 = Log::base10();

        for &val in &[0.001, 0.1, 1.0, 10.0, 100.0] {
            assert!(
                (exp10.inverse(val) - log10.transform(val)).abs() < 1e-10,
                "Exp10::inverse != Log10::transform for {}",
                val
            );
        }
    }

    #[test]
    fn test_exp10_kind_and_name() {
        let t = Exp::base10();
        assert_eq!(t.transform_kind(), TransformKind::Exp10);
        assert_eq!(t.name(), "exp10");
    }

    // ==================== Base-2 (Exp2) Tests ====================

    #[test]
    fn test_exp2_domain() {
        let t = Exp::base2();
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min < 0.0);
        assert!(max.is_infinite() && max > 0.0);
    }

    #[test]
    fn test_exp2_is_value_in_domain() {
        let t = Exp::base2();
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(1.0));
        assert!(t.is_value_in_domain(-1.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
    }

    #[test]
    fn test_exp2_transform() {
        let t = Exp::base2();
        assert!((t.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((t.transform(1.0) - 2.0).abs() < 1e-10);
        assert!((t.transform(2.0) - 4.0).abs() < 1e-10);
        assert!((t.transform(3.0) - 8.0).abs() < 1e-10);
        assert!((t.transform(-1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_exp2_inverse() {
        let t = Exp::base2();
        assert!((t.inverse(1.0) - 0.0).abs() < 1e-10);
        assert!((t.inverse(2.0) - 1.0).abs() < 1e-10);
        assert!((t.inverse(4.0) - 2.0).abs() < 1e-10);
        assert!((t.inverse(8.0) - 3.0).abs() < 1e-10);
        assert!((t.inverse(0.5) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exp2_roundtrip() {
        let t = Exp::base2();
        for &val in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0] {
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
    fn test_exp2_is_inverse_of_log2() {
        // Verify that Exp2::transform is the same as Log2::inverse
        use super::super::Log;
        let exp2 = Exp::base2();
        let log2 = Log::base2();

        for &val in &[-1.0, 0.0, 1.0, 2.0, 3.0] {
            assert!(
                (exp2.transform(val) - log2.inverse(val)).abs() < 1e-10,
                "Exp2::transform != Log2::inverse for {}",
                val
            );
        }
    }

    #[test]
    fn test_exp2_kind_and_name() {
        let t = Exp::base2();
        assert_eq!(t.transform_kind(), TransformKind::Exp2);
        assert_eq!(t.name(), "exp2");
    }

    // ==================== Natural Exp (base e) Tests ====================

    #[test]
    fn test_exp_domain() {
        let t = Exp::natural();
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min < 0.0);
        assert!(max.is_infinite() && max > 0.0);
    }

    #[test]
    fn test_exp_is_value_in_domain() {
        let t = Exp::natural();
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(1.0));
        assert!(t.is_value_in_domain(-1.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
    }

    #[test]
    fn test_exp_transform() {
        let t = Exp::natural();
        assert!((t.transform(0.0) - 1.0).abs() < 1e-10);
        assert!((t.transform(1.0) - E).abs() < 1e-10);
        assert!((t.transform(2.0) - E * E).abs() < 1e-10);
        assert!((t.transform(-1.0) - (1.0 / E)).abs() < 1e-10);
    }

    #[test]
    fn test_exp_inverse() {
        let t = Exp::natural();
        assert!((t.inverse(1.0) - 0.0).abs() < 1e-10);
        assert!((t.inverse(E) - 1.0).abs() < 1e-10);
        assert!((t.inverse(E * E) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_roundtrip() {
        let t = Exp::natural();
        for &val in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
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
    fn test_exp_is_inverse_of_ln() {
        // Verify that Exp::transform is the same as Log(natural)::inverse
        use super::super::Log;
        let exp = Exp::natural();
        let ln = Log::natural();

        for &val in &[-1.0, 0.0, 1.0, 2.0] {
            assert!(
                (exp.transform(val) - ln.inverse(val)).abs() < 1e-10,
                "Exp::transform != Log(natural)::inverse for {}",
                val
            );
        }
    }

    #[test]
    fn test_exp_kind_and_name() {
        let t = Exp::natural();
        assert_eq!(t.transform_kind(), TransformKind::Exp);
        assert_eq!(t.name(), "exp");
    }

    // ==================== General Tests ====================

    #[test]
    fn test_base_accessor() {
        assert!((Exp::base10().base() - 10.0).abs() < 1e-10);
        assert!((Exp::base2().base() - 2.0).abs() < 1e-10);
        assert!((Exp::natural().base() - E).abs() < 1e-10);
    }

    #[test]
    fn test_custom_base() {
        let t = Exp::new(5.0);
        // 5^2 = 25
        assert!((t.transform(2.0) - 25.0).abs() < 1e-10);
        assert!((t.inverse(25.0) - 2.0).abs() < 1e-10);
        // Custom base maps to TransformKind::Exp
        assert_eq!(t.transform_kind(), TransformKind::Exp);
        assert_eq!(t.name(), "exp");
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_zero() {
        Exp::new(0.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_one() {
        Exp::new(1.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_negative() {
        Exp::new(-2.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Exp::base10()), "exp10");
        assert_eq!(format!("{}", Exp::base2()), "exp2");
        assert_eq!(format!("{}", Exp::natural()), "exp");
    }

    #[test]
    fn test_exp_breaks() {
        let t = Exp::base10();
        let breaks = t.calculate_breaks(0.0, 3.0, 5, false);
        assert!(!breaks.is_empty());
    }
}
