use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::graticule_breaks;

#[derive(Debug, Clone, Copy)]
pub struct Geographic;

impl TransformTrait for Geographic {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Geographic
    }

    fn name(&self) -> &'static str {
        "geographic"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, _pretty: bool) -> Vec<f64> {
        graticule_breaks(min, max, n)
    }

    fn calculate_minor_breaks(
        &self,
        _major_breaks: &[f64],
        _n: usize,
        _range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        Vec::new()
    }

    fn transform(&self, value: f64) -> f64 {
        value
    }

    fn inverse(&self, value: f64) -> f64 {
        value
    }
}

impl std::fmt::Display for Geographic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breaks_world_longitude() {
        let t = Geographic;
        let breaks = t.calculate_breaks(-180.0, 180.0, 7, true);
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b >= -180.0 && b <= 180.0);
        }
        // Should pick nice degree intervals (multiples of 30° or 45°)
        assert!(breaks.iter().all(|b| b % 30.0 == 0.0 || b % 45.0 == 0.0));
    }

    #[test]
    fn test_breaks_small_extent() {
        let t = Geographic;
        let breaks = t.calculate_breaks(5.0, 15.0, 5, true);
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b > 5.0 && b < 15.0);
        }
    }

    #[test]
    fn test_breaks_respects_count() {
        let t = Geographic;
        let breaks_few = t.calculate_breaks(-90.0, 90.0, 3, true);
        let breaks_many = t.calculate_breaks(-90.0, 90.0, 10, true);
        assert!(breaks_many.len() >= breaks_few.len());
    }

    #[test]
    fn test_identity_transform() {
        let t = Geographic;
        assert_eq!(t.transform(42.0), 42.0);
        assert_eq!(t.inverse(42.0), 42.0);
    }
}
