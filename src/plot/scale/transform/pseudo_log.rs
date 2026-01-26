//! PseudoLog transform implementation (symmetric log with configurable base)
//!
//! This module provides a symmetric logarithm transform that handles zero and
//! negative values. The base parameter controls which logarithm is approximated
//! for large values.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::symlog_breaks;

/// PseudoLog transform - symmetric logarithm with configurable base
///
/// Domain: (-∞, +∞) - all real numbers
///
/// The pseudo-log transform is a symmetric logarithm that handles zero and
/// negative values. It is based on the inverse hyperbolic sine (asinh) scaled
/// to approximate log of the given base for large values.
///
/// Formula (ggplot2's `pseudo_log_trans` with sigma=1):
/// - transform: `asinh(x / 2) / ln(base)`
/// - inverse: `sinh(y * ln(base)) * 2`
///
/// Properties:
/// - Linear near zero (smooth transition)
/// - Logarithmic (given base) for large |x|
/// - Symmetric around zero: f(-x) = -f(x)
/// - f(0) = 0
/// - For large x: f(x) ≈ log_base(x)
///
/// The base determines which logarithm is approximated:
/// - Base 10: approximates log10 for large values
/// - Base 2: approximates log2 for large values
/// - Base e: approximates ln for large values (equivalent to asinh scaling)
#[derive(Debug, Clone, Copy)]
pub struct PseudoLog {
    base: f64,
    ln_base: f64, // cached for performance
}

impl PseudoLog {
    /// Create a pseudo-log transform with the given base
    pub fn new(base: f64) -> Self {
        assert!(
            base > 0.0 && base != 1.0,
            "PseudoLog base must be positive and not 1"
        );
        Self {
            base,
            ln_base: base.ln(),
        }
    }

    /// Create a base-10 pseudo-log transform (default)
    ///
    /// Approximates log10 for large values.
    pub fn base10() -> Self {
        Self::new(10.0)
    }

    /// Create a base-2 pseudo-log transform
    ///
    /// Approximates log2 for large values.
    pub fn base2() -> Self {
        Self::new(2.0)
    }

    /// Create a natural pseudo-log transform (base e)
    ///
    /// Approximates ln for large values.
    pub fn natural() -> Self {
        Self::new(std::f64::consts::E)
    }

    /// Get the base of this pseudo-log
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Check if this is a base-10 pseudo-log (within floating point tolerance)
    fn is_base10(&self) -> bool {
        (self.base - 10.0).abs() < 1e-10
    }

    /// Check if this is a base-2 pseudo-log (within floating point tolerance)
    fn is_base2(&self) -> bool {
        (self.base - 2.0).abs() < 1e-10
    }
}

impl TransformTrait for PseudoLog {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::PseudoLog
    }

    fn name(&self) -> &'static str {
        if self.is_base10() {
            "pseudo_log"
        } else if self.is_base2() {
            "pseudo_log2"
        } else {
            "pseudo_ln"
        }
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        symlog_breaks(min, max, n, pretty)
    }

    fn transform(&self, value: f64) -> f64 {
        (value * 0.5).asinh() / self.ln_base
    }

    fn inverse(&self, value: f64) -> f64 {
        (value * self.ln_base).sinh() * 2.0
    }
}

impl std::fmt::Display for PseudoLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    // ==================== Base-10 Tests (default) ====================

    #[test]
    fn test_pseudo_log_domain() {
        let t = PseudoLog::base10();
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_pseudo_log_is_value_in_domain() {
        let t = PseudoLog::base10();
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(-1000.0));
        assert!(t.is_value_in_domain(1000.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
    }

    #[test]
    fn test_pseudo_log_transform_at_zero() {
        let t = PseudoLog::base10();
        // f(0) = asinh(0) / ln(10) = 0
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudo_log_transform_values() {
        let t = PseudoLog::base10();
        let ln10 = 10.0_f64.ln();

        // f(10) = asinh(5) / ln(10) ≈ 1.004
        let expected_10 = 5.0_f64.asinh() / ln10;
        assert!(
            (t.transform(10.0) - expected_10).abs() < 1e-10,
            "transform(10) = {}, expected {}",
            t.transform(10.0),
            expected_10
        );

        // f(100) = asinh(50) / ln(10) ≈ 2.000
        let expected_100 = 50.0_f64.asinh() / ln10;
        assert!(
            (t.transform(100.0) - expected_100).abs() < 1e-10,
            "transform(100) = {}, expected {}",
            t.transform(100.0),
            expected_100
        );

        // f(1000) = asinh(500) / ln(10) ≈ 3.000
        let expected_1000 = 500.0_f64.asinh() / ln10;
        assert!(
            (t.transform(1000.0) - expected_1000).abs() < 1e-10,
            "transform(1000) = {}, expected {}",
            t.transform(1000.0),
            expected_1000
        );
    }

    #[test]
    fn test_pseudo_log_approximates_log10_for_large_values() {
        let t = PseudoLog::base10();
        // For large x, pseudo_log(x) ≈ log10(x)
        for &x in &[1000.0, 10000.0, 100000.0] {
            let pseudo = t.transform(x);
            let log10 = x.log10();
            let error = (pseudo - log10).abs();
            assert!(
                error < 0.01,
                "For x={}, pseudo_log={}, log10={}, error={}",
                x,
                pseudo,
                log10,
                error
            );
        }
    }

    #[test]
    fn test_pseudo_log_is_symmetric() {
        let t = PseudoLog::base10();
        // f(-x) = -f(x) for all x
        for &val in &[0.1, 1.0, 10.0, 100.0, 1000.0] {
            let pos = t.transform(val);
            let neg = t.transform(-val);
            assert!(
                (pos + neg).abs() < 1e-10,
                "Not symmetric: f({}) = {}, f({}) = {}",
                val,
                pos,
                -val,
                neg
            );
        }
    }

    #[test]
    fn test_pseudo_log_inverse() {
        let t = PseudoLog::base10();
        let ln10 = 10.0_f64.ln();

        // inverse(0) = sinh(0) * 2 = 0
        assert!((t.inverse(0.0) - 0.0).abs() < 1e-10);

        // inverse(1) = sinh(ln(10)) * 2 ≈ 9.9
        let expected_1 = (1.0 * ln10).sinh() * 2.0;
        assert!(
            (t.inverse(1.0) - expected_1).abs() < 1e-10,
            "inverse(1) = {}, expected {}",
            t.inverse(1.0),
            expected_1
        );

        // inverse(2) = sinh(2*ln(10)) * 2 ≈ 99.98
        let expected_2 = (2.0 * ln10).sinh() * 2.0;
        assert!(
            (t.inverse(2.0) - expected_2).abs() < 1e-10,
            "inverse(2) = {}, expected {}",
            t.inverse(2.0),
            expected_2
        );
    }

    #[test]
    fn test_pseudo_log_roundtrip() {
        let t = PseudoLog::base10();
        for &val in &[-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            if val == 0.0 {
                assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
            } else {
                assert!(
                    (back - val).abs() / val.abs() < 1e-10,
                    "Roundtrip failed for {}: got {}",
                    val,
                    back
                );
            }
        }
    }

    #[test]
    fn test_pseudo_log_different_from_asinh() {
        let pseudo = PseudoLog::base10();
        // pseudo_log and asinh are NOT the same
        // asinh(10) ≈ 2.998, pseudo_log(10) ≈ 1.004
        let asinh_10 = 10.0_f64.asinh();
        let pseudo_10 = pseudo.transform(10.0);
        assert!(
            (asinh_10 - pseudo_10).abs() > 1.0,
            "pseudo_log should differ from asinh: asinh(10)={}, pseudo_log(10)={}",
            asinh_10,
            pseudo_10
        );
    }

    #[test]
    fn test_pseudo_log_breaks() {
        let t = PseudoLog::base10();
        let breaks = t.calculate_breaks(-100.0, 100.0, 7, false);
        assert!(breaks.contains(&0.0));
    }

    #[test]
    fn test_pseudo_log_kind_and_name() {
        let t = PseudoLog::base10();
        assert_eq!(t.transform_kind(), TransformKind::PseudoLog);
        assert_eq!(t.name(), "pseudo_log");
    }

    // ==================== Base-2 Tests ====================

    #[test]
    fn test_pseudo_log2_transform_at_zero() {
        let t = PseudoLog::base2();
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudo_log2_approximates_log2_for_large_values() {
        let t = PseudoLog::base2();
        // For large x, pseudo_log2(x) ≈ log2(x)
        for &x in &[64.0, 1024.0, 65536.0] {
            let pseudo = t.transform(x);
            let log2 = x.log2();
            let error = (pseudo - log2).abs();
            assert!(
                error < 0.05,
                "For x={}, pseudo_log2={}, log2={}, error={}",
                x,
                pseudo,
                log2,
                error
            );
        }
    }

    #[test]
    fn test_pseudo_log2_is_symmetric() {
        let t = PseudoLog::base2();
        for &val in &[1.0, 4.0, 16.0, 64.0] {
            let pos = t.transform(val);
            let neg = t.transform(-val);
            assert!(
                (pos + neg).abs() < 1e-10,
                "Not symmetric: f({}) = {}, f({}) = {}",
                val,
                pos,
                -val,
                neg
            );
        }
    }

    #[test]
    fn test_pseudo_log2_roundtrip() {
        let t = PseudoLog::base2();
        for &val in &[-64.0, -8.0, -1.0, 0.0, 1.0, 8.0, 64.0] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            if val == 0.0 {
                assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
            } else {
                assert!(
                    (back - val).abs() / val.abs() < 1e-10,
                    "Roundtrip failed for {}: got {}",
                    val,
                    back
                );
            }
        }
    }

    #[test]
    fn test_pseudo_log2_kind_and_name() {
        let t = PseudoLog::base2();
        assert_eq!(t.transform_kind(), TransformKind::PseudoLog);
        assert_eq!(t.name(), "pseudo_log2");
    }

    // ==================== Natural (base e) Tests ====================

    #[test]
    fn test_pseudo_ln_transform_at_zero() {
        let t = PseudoLog::natural();
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudo_ln_approximates_ln_for_large_values() {
        let t = PseudoLog::natural();
        // For large x, pseudo_ln(x) ≈ ln(x)
        for &x in &[100.0, 1000.0, 10000.0] {
            let pseudo = t.transform(x);
            let ln = x.ln();
            let error = (pseudo - ln).abs();
            assert!(
                error < 0.02,
                "For x={}, pseudo_ln={}, ln={}, error={}",
                x,
                pseudo,
                ln,
                error
            );
        }
    }

    #[test]
    fn test_pseudo_ln_is_symmetric() {
        let t = PseudoLog::natural();
        for &val in &[1.0, E, E * E, 100.0] {
            let pos = t.transform(val);
            let neg = t.transform(-val);
            assert!(
                (pos + neg).abs() < 1e-10,
                "Not symmetric: f({}) = {}, f({}) = {}",
                val,
                pos,
                -val,
                neg
            );
        }
    }

    #[test]
    fn test_pseudo_ln_roundtrip() {
        let t = PseudoLog::natural();
        for &val in &[-100.0, -E, -1.0, 0.0, 1.0, E, 100.0] {
            let transformed = t.transform(val);
            let back = t.inverse(transformed);
            if val == 0.0 {
                assert!((back - val).abs() < 1e-10, "Roundtrip failed for {}", val);
            } else {
                assert!(
                    (back - val).abs() / val.abs() < 1e-10,
                    "Roundtrip failed for {}: got {}",
                    val,
                    back
                );
            }
        }
    }

    #[test]
    fn test_pseudo_ln_kind_and_name() {
        let t = PseudoLog::natural();
        assert_eq!(t.transform_kind(), TransformKind::PseudoLog);
        assert_eq!(t.name(), "pseudo_ln");
    }

    // ==================== General Tests ====================

    #[test]
    fn test_base_accessor() {
        assert!((PseudoLog::base10().base() - 10.0).abs() < 1e-10);
        assert!((PseudoLog::base2().base() - 2.0).abs() < 1e-10);
        assert!((PseudoLog::natural().base() - E).abs() < 1e-10);
    }

    #[test]
    fn test_custom_base() {
        let t = PseudoLog::new(5.0);
        // f(0) = 0
        assert!((t.transform(0.0) - 0.0).abs() < 1e-10);
        // Roundtrip works
        let val = 125.0;
        let transformed = t.transform(val);
        let back = t.inverse(transformed);
        assert!(
            (back - val).abs() / val < 1e-10,
            "Roundtrip failed for {}",
            val
        );
        // Custom base maps to TransformKind::PseudoLog
        assert_eq!(t.transform_kind(), TransformKind::PseudoLog);
        // Custom base name falls back to pseudo_ln
        assert_eq!(t.name(), "pseudo_ln");
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_zero() {
        PseudoLog::new(0.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_one() {
        PseudoLog::new(1.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_base_negative() {
        PseudoLog::new(-2.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", PseudoLog::base10()), "pseudo_log");
        assert_eq!(format!("{}", PseudoLog::base2()), "pseudo_log2");
        assert_eq!(format!("{}", PseudoLog::natural()), "pseudo_ln");
    }

    #[test]
    fn test_all_bases_symmetric_around_zero() {
        for t in &[
            PseudoLog::base10(),
            PseudoLog::base2(),
            PseudoLog::natural(),
        ] {
            for &val in &[0.1, 1.0, 10.0, 100.0] {
                let pos = t.transform(val);
                let neg = t.transform(-val);
                assert!(
                    (pos + neg).abs() < 1e-10,
                    "{}: Not symmetric for {}",
                    t.name(),
                    val
                );
            }
        }
    }

    #[test]
    fn test_all_bases_zero_at_origin() {
        for t in &[
            PseudoLog::base10(),
            PseudoLog::base2(),
            PseudoLog::natural(),
        ] {
            assert!(t.transform(0.0).abs() < 1e-10, "{}: f(0) != 0", t.name());
        }
    }
}
