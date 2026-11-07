//! Log transform implementation (parameterized by base)
//!
//! This module provides a unified logarithm transform that supports any base.
//! Common bases (10, 2, e) have named constructors for convenience.

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{extent_minor_breaks, log_minor_breaks, log_pretty_breaks};

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

    /// zygos's log for this base: its own member for bases 10, 2 and e, and the
    /// natural log divided by `ln(base)` for any other.
    fn math(&self) -> ZygosMath {
        match self.transform_kind() {
            TransformKind::Log if (self.base - std::f64::consts::E).abs() >= 1e-10 => {
                ZygosMath::with_scale(TransformKind::Log, self.base.ln())
            }
            kind => ZygosMath::of(kind),
        }
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
        // Strictly positive, not zygos's `(0, +∞)` bound: scale resolution clips
        // an out-of-domain limit to this minimum, and zero has no logarithm.
        (f64::MIN_POSITIVE, f64::INFINITY)
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, _pretty: bool) -> Vec<f64> {
        log_pretty_breaks(min, max, n, self.base)
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        extent_minor_breaks(major_breaks, n, range, |min, max| {
            log_minor_breaks(min, max, self.base)
        })
    }

    fn default_minor_break_count(&self) -> usize {
        // The whole-number multiples between powers fix the density (2-9 for
        // base 10); any non-zero count asks for them.
        8
    }

    fn transform(&self, value: f64) -> f64 {
        self.math().forward(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        self.math().inverse(value)
    }

    fn expand_range(&self, min: f64, max: f64, mult: (f64, f64), add: (f64, f64)) -> (f64, f64) {
        self.math().expand_range(min, max, mult, add)
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
    fn test_all_bases_breaks() {
        // Log10: the 1-2-5 pattern across every decade.
        let t10 = Log::base10();
        assert_eq!(
            t10.calculate_breaks(1.0, 1000.0, 5, true),
            vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        );
        // `pretty` has no powers-only alternative.
        assert_eq!(
            t10.calculate_breaks(1.0, 1000.0, 5, false),
            t10.calculate_breaks(1.0, 1000.0, 5, true)
        );

        // Log2: only multipliers below the base survive, leaving the powers.
        let t2 = Log::base2();
        assert_eq!(
            t2.calculate_breaks(1.0, 16.0, 10, true),
            vec![1.0, 2.0, 4.0, 8.0, 16.0]
        );

        // Natural log over more than four e-folds: the powers of e alone.
        let tn = Log::natural();
        let expected = [0, 1, 2, 3, 4].map(|d| std::f64::consts::E.powi(d));
        let breaksn = tn.calculate_breaks(1.0, 100.0, 10, true);
        assert_eq!(breaksn.len(), expected.len(), "{breaksn:?}");
        for (b, x) in breaksn.iter().zip(expected) {
            assert!((b - x).abs() < 1e-9, "{breaksn:?}");
        }
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
