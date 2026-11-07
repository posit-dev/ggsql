//! Exponential transform implementation (base^x) - inverse of log

use super::{TransformKind, TransformTrait, ZygosMath};
use crate::plot::scale::breaks::{extended_breaks, linear_breaks, minor_breaks_linear};

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

    /// zygos's exponential for this base: its own member for bases 10, 2 and e,
    /// and the natural exponential of `x · ln(base)` for any other.
    fn math(&self) -> ZygosMath {
        match self.transform_kind() {
            TransformKind::Exp if (self.base - std::f64::consts::E).abs() >= 1e-10 => {
                ZygosMath::with_scale(TransformKind::Exp, self.base.ln())
            }
            kind => ZygosMath::of(kind),
        }
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
        self.math().allowed_domain()
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        // Breaks in data space (the exponents), so the labels stay round
        // numbers; the axis spaces them out exponentially.
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
        self.math().forward_exp(value)
    }

    fn inverse(&self, value: f64) -> f64 {
        self.math().inverse_exp(value)
    }

    fn expand_range(&self, min: f64, max: f64, mult: (f64, f64), add: (f64, f64)) -> (f64, f64) {
        self.math().expand_range_exp(min, max, mult, add)
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
        // Round exponents; the axis spaces them out exponentially.
        assert_eq!(
            t.calculate_breaks(0.0, 3.0, 5, true),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            t.calculate_breaks(0.0, 3.0, 5, false),
            vec![0.0, 0.75, 1.5, 2.25, 3.0]
        );
    }
}
