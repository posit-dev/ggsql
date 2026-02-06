//! PseudoLog transform implementation (symmetric log with configurable base)
//!
//! This module provides a symmetric logarithm transform that handles zero and
//! negative values. The base parameter controls which logarithm is approximated
//! for large values.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{minor_breaks_symlog, symlog_breaks};

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

    // ==================== Consolidated Transform Tests ====================

    /// Test data for all pseudo-log bases
    fn get_transforms() -> Vec<(PseudoLog, &'static str)> {
        vec![
            (PseudoLog::base10(), "pseudo_log"),
            (PseudoLog::base2(), "pseudo_log2"),
            (PseudoLog::natural(), "pseudo_ln"),
        ]
    }

    #[test]
    fn test_all_bases_domain() {
        for (t, name) in get_transforms() {
            let (min, max) = t.allowed_domain();
            assert!(
                min.is_infinite() && min.is_sign_negative(),
                "{}: domain min should be -∞",
                name
            );
            assert!(
                max.is_infinite() && max.is_sign_positive(),
                "{}: domain max should be +∞",
                name
            );
        }
    }

    #[test]
    fn test_all_bases_zero_at_origin() {
        for (t, name) in get_transforms() {
            assert!(t.transform(0.0).abs() < 1e-10, "{}: f(0) should be 0", name);
        }
    }

    #[test]
    fn test_all_bases_symmetric_around_zero() {
        let test_values = [0.1, 1.0, 10.0, 100.0, 1000.0];
        for (t, name) in get_transforms() {
            for &val in &test_values {
                let pos = t.transform(val);
                let neg = t.transform(-val);
                assert!(
                    (pos + neg).abs() < 1e-10,
                    "{}: Not symmetric for {} (f({})={}, f({})={})",
                    name,
                    val,
                    val,
                    pos,
                    -val,
                    neg
                );
            }
        }
    }

    #[test]
    fn test_all_bases_roundtrip() {
        let test_values = [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0];
        for (t, name) in get_transforms() {
            for &val in &test_values {
                let transformed = t.transform(val);
                let back = t.inverse(transformed);
                if val == 0.0 {
                    assert!(
                        (back - val).abs() < 1e-10,
                        "{}: Roundtrip failed for {}",
                        name,
                        val
                    );
                } else {
                    assert!(
                        (back - val).abs() / val.abs() < 1e-10,
                        "{}: Roundtrip failed for {} (got {})",
                        name,
                        val,
                        back
                    );
                }
            }
        }
    }

    #[test]
    fn test_all_bases_kind_and_name() {
        for (t, expected_name) in get_transforms() {
            assert_eq!(
                t.transform_kind(),
                TransformKind::PseudoLog,
                "Kind should be PseudoLog"
            );
            assert_eq!(t.name(), expected_name);
        }
    }

    #[test]
    fn test_approximates_log_for_large_values() {
        // Test that each pseudo-log approximates its corresponding log for large values
        let test_cases = vec![
            (PseudoLog::base10(), vec![1000.0, 10000.0, 100000.0], 0.01), // log10
            (PseudoLog::base2(), vec![64.0, 1024.0, 65536.0], 0.05),      // log2
            (PseudoLog::natural(), vec![100.0, 1000.0, 10000.0], 0.02),   // ln
        ];

        for (t, values, tolerance) in test_cases {
            for x in values {
                let pseudo = t.transform(x);
                let actual_log = x.log(t.base());
                let error = (pseudo - actual_log).abs();
                assert!(
                    error < tolerance,
                    "{}: For x={}, pseudo={}, log={}, error={}",
                    t.name(),
                    x,
                    pseudo,
                    actual_log,
                    error
                );
            }
        }
    }

    #[test]
    fn test_all_bases_display() {
        assert_eq!(format!("{}", PseudoLog::base10()), "pseudo_log");
        assert_eq!(format!("{}", PseudoLog::base2()), "pseudo_log2");
        assert_eq!(format!("{}", PseudoLog::natural()), "pseudo_ln");
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
    fn test_invalid_bases() {
        // Test all invalid base cases in one test
        let invalid_bases = [(0.0, "zero"), (1.0, "one"), (-2.0, "negative")];
        for (base, desc) in invalid_bases {
            let result = std::panic::catch_unwind(|| PseudoLog::new(base));
            assert!(
                result.is_err(),
                "PseudoLog::new({}) should panic for {} base",
                base,
                desc
            );
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
    fn test_default_minor_break_count() {
        for (t, name) in get_transforms() {
            assert_eq!(
                t.default_minor_break_count(),
                8,
                "{} should have default minor count of 8",
                name
            );
        }
    }
}
