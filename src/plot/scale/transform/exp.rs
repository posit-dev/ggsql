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

    // ==================== Consolidated Transform Tests ====================

    /// Test data for all exp bases
    fn get_transforms() -> Vec<(Exp, TransformKind, &'static str)> {
        vec![
            (Exp::base10(), TransformKind::Exp10, "exp10"),
            (Exp::base2(), TransformKind::Exp2, "exp2"),
            (Exp::natural(), TransformKind::Exp, "exp"),
        ]
    }

    #[test]
    fn test_all_bases_domain() {
        for (t, _, name) in get_transforms() {
            let (min, max) = t.allowed_domain();
            assert!(
                min.is_infinite() && min < 0.0,
                "{}: domain min should be -∞",
                name
            );
            assert!(
                max.is_infinite() && max > 0.0,
                "{}: domain max should be +∞",
                name
            );
        }
    }

    #[test]
    fn test_all_bases_transform_and_inverse() {
        // Test cases: (transform, input, expected_transform)
        let test_cases = vec![
            // Exp10: 10^0=1, 10^1=10, 10^2=100, 10^-1=0.1
            (
                Exp::base10(),
                vec![(0.0, 1.0), (1.0, 10.0), (2.0, 100.0), (-1.0, 0.1)],
            ),
            // Exp2: 2^0=1, 2^1=2, 2^2=4, 2^-1=0.5
            (
                Exp::base2(),
                vec![(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (-1.0, 0.5)],
            ),
            // Natural: e^0=1, e^1=e, e^2=e²
            (Exp::natural(), vec![(0.0, 1.0), (1.0, E), (2.0, E * E)]),
        ];

        for (t, cases) in test_cases {
            for (input, expected) in cases {
                assert!(
                    (t.transform(input) - expected).abs() < 1e-10,
                    "{}: transform({}) should be {}, got {}",
                    t.name(),
                    input,
                    expected,
                    t.transform(input)
                );
                // Test inverse too
                assert!(
                    (t.inverse(expected) - input).abs() < 1e-9,
                    "{}: inverse({}) should be {}, got {}",
                    t.name(),
                    expected,
                    input,
                    t.inverse(expected)
                );
            }
        }
    }

    #[test]
    fn test_all_bases_roundtrip() {
        let test_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for (t, _, name) in get_transforms() {
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
                        "{}: Roundtrip failed for {}",
                        name,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_all_bases_kind_and_name() {
        for (t, expected_kind, expected_name) in get_transforms() {
            assert_eq!(
                t.transform_kind(),
                expected_kind,
                "Kind mismatch for {}",
                expected_name
            );
            assert_eq!(t.name(), expected_name);
        }
    }

    #[test]
    fn test_all_bases_is_inverse_of_log() {
        use super::super::Log;

        let pairs = vec![
            (Exp::base10(), Log::base10()),
            (Exp::base2(), Log::base2()),
            (Exp::natural(), Log::natural()),
        ];

        let test_values = [-1.0, 0.0, 1.0, 2.0];
        for (exp, log) in pairs {
            for &val in &test_values {
                assert!(
                    (exp.transform(val) - log.inverse(val)).abs() < 1e-10,
                    "{}::transform != {}::inverse for {}",
                    exp.name(),
                    log.name(),
                    val
                );
            }
        }
    }

    #[test]
    fn test_all_bases_display() {
        assert_eq!(format!("{}", Exp::base10()), "exp10");
        assert_eq!(format!("{}", Exp::base2()), "exp2");
        assert_eq!(format!("{}", Exp::natural()), "exp");
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
    fn test_invalid_bases() {
        // Test all invalid base cases in one test
        let invalid_bases = [(0.0, "zero"), (1.0, "one"), (-2.0, "negative")];
        for (base, desc) in invalid_bases {
            let result = std::panic::catch_unwind(|| Exp::new(base));
            assert!(
                result.is_err(),
                "Exp::new({}) should panic for {} base",
                base,
                desc
            );
        }
    }

    #[test]
    fn test_exp_breaks() {
        let t = Exp::base10();
        let breaks = t.calculate_breaks(0.0, 3.0, 5, false);
        assert!(!breaks.is_empty());
    }
}
