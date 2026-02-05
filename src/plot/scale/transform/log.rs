//! Log transform implementation (parameterized by base)
//!
//! This module provides a unified logarithm transform that supports any base.
//! Common bases (10, 2, e) have named constructors for convenience.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{log_breaks, minor_breaks_log};

/// Log transform - logarithm with configurable base
///
/// Domain: (0, +∞) - positive values only
///
/// The base determines which `TransformKind` is returned:
/// - Base 10 → `TransformKind::Log10`
/// - Base 2 → `TransformKind::Log2`
/// - Base e → `TransformKind::Log`
#[derive(Debug, Clone, Copy)]
pub struct Log {
    base: f64,
}

impl Log {
    /// Create a log transform with the given base
    pub fn new(base: f64) -> Self {
        assert!(
            base > 0.0 && base != 1.0,
            "Log base must be positive and not 1"
        );
        Self { base }
    }

    /// Create a base-10 logarithm transform
    pub fn base10() -> Self {
        Self { base: 10.0 }
    }

    /// Create a base-2 logarithm transform
    pub fn base2() -> Self {
        Self { base: 2.0 }
    }

    /// Create a natural logarithm transform (base e)
    pub fn natural() -> Self {
        Self {
            base: std::f64::consts::E,
        }
    }

    /// Get the base of this logarithm
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Check if this is a base-10 log (within floating point tolerance)
    fn is_base10(&self) -> bool {
        (self.base - 10.0).abs() < 1e-10
    }

    /// Check if this is a base-2 log (within floating point tolerance)
    fn is_base2(&self) -> bool {
        (self.base - 2.0).abs() < 1e-10
    }

    /// Check if this is a natural log (within floating point tolerance)
    fn is_natural(&self) -> bool {
        (self.base - std::f64::consts::E).abs() < 1e-10
    }
}

impl TransformTrait for Log {
    fn transform_kind(&self) -> TransformKind {
        if self.is_base10() {
            TransformKind::Log10
        } else if self.is_base2() {
            TransformKind::Log2
        } else {
            // Natural log and any other base map to Log
            TransformKind::Log
        }
    }

    fn name(&self) -> &'static str {
        if self.is_base10() {
            "log"
        } else if self.is_base2() {
            "log2"
        } else {
            "ln"
        }
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::MIN_POSITIVE, f64::INFINITY)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        value > 0.0 && value.is_finite()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        log_breaks(min, max, n, self.base, pretty)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        minor_breaks_log(major_breaks, n, self.base, range)
    }

    fn default_minor_break_count(&self) -> usize {
        8 // Similar density to traditional 2-9 pattern on log axes
    }

    fn transform(&self, value: f64) -> f64 {
        value.log(self.base)
    }

    fn inverse(&self, value: f64) -> f64 {
        self.base.powf(value)
    }
}

impl std::fmt::Display for Log {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    // ==================== Consolidated Transform Tests ====================

    /// Test data for all log bases
    fn get_transforms() -> Vec<(Log, TransformKind, &'static str)> {
        vec![
            (Log::base10(), TransformKind::Log10, "log"),
            (Log::base2(), TransformKind::Log2, "log2"),
            (Log::natural(), TransformKind::Log, "ln"),
        ]
    }

    #[test]
    fn test_all_bases_domain() {
        for (t, _, name) in get_transforms() {
            let (min, max) = t.allowed_domain();
            assert!(min > 0.0, "{}: domain min should be > 0", name);
            assert!(max.is_infinite(), "{}: domain max should be infinite", name);
        }
    }

    #[test]
    fn test_all_bases_is_value_in_domain() {
        for (t, _, name) in get_transforms() {
            // Valid values
            assert!(
                t.is_value_in_domain(1.0),
                "{}: 1.0 should be in domain",
                name
            );
            assert!(
                t.is_value_in_domain(0.0001),
                "{}: 0.0001 should be in domain",
                name
            );
            assert!(
                t.is_value_in_domain(1000.0),
                "{}: 1000.0 should be in domain",
                name
            );
            // Invalid values
            assert!(
                !t.is_value_in_domain(0.0),
                "{}: 0.0 should not be in domain",
                name
            );
            assert!(
                !t.is_value_in_domain(-1.0),
                "{}: -1.0 should not be in domain",
                name
            );
            assert!(
                !t.is_value_in_domain(f64::INFINITY),
                "{}: infinity should not be in domain",
                name
            );
            assert!(
                !t.is_value_in_domain(f64::NAN),
                "{}: NaN should not be in domain",
                name
            );
        }
    }

    #[test]
    fn test_all_bases_transform_and_inverse() {
        // Test cases: (transform, input, expected_transform, inverse_test_val, expected_inverse)
        let test_cases = vec![
            // Log10: log10(1)=0, log10(10)=1, log10(100)=2, log10(0.1)=-1
            (
                Log::base10(),
                vec![(1.0, 0.0), (10.0, 1.0), (100.0, 2.0), (0.1, -1.0)],
            ),
            // Log2: log2(1)=0, log2(2)=1, log2(4)=2, log2(0.5)=-1
            (
                Log::base2(),
                vec![(1.0, 0.0), (2.0, 1.0), (4.0, 2.0), (0.5, -1.0)],
            ),
            // Natural: ln(1)=0, ln(e)=1, ln(e²)=2
            (Log::natural(), vec![(1.0, 0.0), (E, 1.0), (E * E, 2.0)]),
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
        let test_values = [0.001, 0.1, 1.0, 2.0, 10.0, 100.0, 1000.0];
        for (t, _, name) in get_transforms() {
            for &val in &test_values {
                let transformed = t.transform(val);
                let back = t.inverse(transformed);
                assert!(
                    (back - val).abs() / val < 1e-10,
                    "{}: Roundtrip failed for {}",
                    name,
                    val
                );
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
    fn test_all_bases_breaks_contain_powers() {
        // Log10: 1, 10, 100, 1000
        let t10 = Log::base10();
        let breaks10 = t10.calculate_breaks(1.0, 1000.0, 10, false);
        for &v in &[1.0, 10.0, 100.0, 1000.0] {
            assert!(breaks10.contains(&v), "log10 breaks should contain {}", v);
        }

        // Log2: 1, 2, 4, 8, 16
        let t2 = Log::base2();
        let breaks2 = t2.calculate_breaks(1.0, 16.0, 10, false);
        for &v in &[1.0, 2.0, 4.0, 8.0, 16.0] {
            assert!(breaks2.contains(&v), "log2 breaks should contain {}", v);
        }

        // Natural log - just verify non-empty
        let tn = Log::natural();
        let breaksn = tn.calculate_breaks(1.0, 100.0, 10, false);
        assert!(
            !breaksn.is_empty(),
            "natural log breaks should not be empty"
        );
    }

    #[test]
    fn test_all_bases_display() {
        assert_eq!(format!("{}", Log::base10()), "log");
        assert_eq!(format!("{}", Log::base2()), "log2");
        assert_eq!(format!("{}", Log::natural()), "ln");
    }

    // ==================== General Tests ====================

    #[test]
    fn test_base_accessor() {
        assert!((Log::base10().base() - 10.0).abs() < 1e-10);
        assert!((Log::base2().base() - 2.0).abs() < 1e-10);
        assert!((Log::natural().base() - E).abs() < 1e-10);
    }

    #[test]
    fn test_custom_base() {
        let t = Log::new(5.0);
        // 5^2 = 25, so log_5(25) = 2
        assert!((t.transform(25.0) - 2.0).abs() < 1e-10);
        assert!((t.inverse(2.0) - 25.0).abs() < 1e-10);
        // Custom base maps to TransformKind::Log
        assert_eq!(t.transform_kind(), TransformKind::Log);
        assert_eq!(t.name(), "ln");
    }

    #[test]
    fn test_invalid_bases() {
        // Test all invalid base cases in one test
        let invalid_bases = [(0.0, "zero"), (1.0, "one"), (-2.0, "negative")];
        for (base, desc) in invalid_bases {
            let result = std::panic::catch_unwind(|| Log::new(base));
            assert!(
                result.is_err(),
                "Log::new({}) should panic for {} base",
                base,
                desc
            );
        }
    }

    // ==================== Minor Breaks Tests ====================

    #[test]
    fn test_minor_breaks_all_bases() {
        // Test minor breaks work for all bases
        let test_cases = vec![
            (Log::base10(), vec![1.0, 10.0, 100.0], 8, 16), // 8 per decade × 2 decades
            (Log::base2(), vec![1.0, 2.0, 4.0, 8.0], 1, 3), // 1 per interval × 3 intervals
        ];

        for (t, majors, n, expected_len) in test_cases {
            let minors = t.calculate_minor_breaks(&majors, n, None);
            assert_eq!(
                minors.len(),
                expected_len,
                "{}: expected {} minor breaks, got {}",
                t.name(),
                expected_len,
                minors.len()
            );
            assert!(
                minors.iter().all(|&x| x > 0.0),
                "{}: all minor breaks should be positive",
                t.name()
            );
        }
    }

    #[test]
    fn test_minor_breaks_geometric_mean() {
        let t = Log::base10();
        let majors = vec![1.0, 10.0];
        let minors = t.calculate_minor_breaks(&majors, 1, None);
        // Single minor break should be at geometric mean: sqrt(1 * 10) ≈ 3.16
        assert_eq!(minors.len(), 1);
        assert!((minors[0] - (1.0_f64 * 10.0).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_minor_breaks_with_extension() {
        let t = Log::base10();
        let majors = vec![10.0, 100.0];
        let minors = t.calculate_minor_breaks(&majors, 8, Some((1.0, 1000.0)));
        // Should extend into [1, 10) and (100, 1000]
        assert_eq!(minors.len(), 24); // 8 per decade × 3 decades
    }

    #[test]
    fn test_default_minor_break_count() {
        // All log transforms should have the same default
        for (t, _, name) in get_transforms() {
            assert_eq!(
                t.default_minor_break_count(),
                8,
                "{} should have default minor count of 8",
                name
            );
        }
    }
}
