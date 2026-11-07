//! Date transform implementation
//!
//! Transforms Date data (days since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals (years, months, weeks, days).

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{integer_breaks, minor_breaks_linear};
use crate::plot::ArrayElement;
use zygos::breaks::temporal_breaks_from_f64;

/// Date transform - for date data (days since epoch)
///
/// This transform works on the numeric representation of dates (days since Unix epoch).
/// The transform/inverse functions are identity (pass-through), but break calculation
/// produces sensible temporal intervals.
#[derive(Debug, Clone, Copy)]
pub struct Date;

impl TransformTrait for Date {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Date
    }

    fn name(&self) -> &'static str {
        "date"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn transform(&self, value: f64) -> f64 {
        // Identity transform - dates stay in days-since-epoch space
        value
    }

    fn inverse(&self, value: f64) -> f64 {
        // Identity inverse
        value
    }

    fn calculate_breaks(&self, min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
        if n == 0 || min >= max {
            return vec![];
        }

        if pretty {
            // A calendar interval (years, quarters, months, weeks, days) sized
            // to about `n` breaks, stepped from its calendar grid.
            temporal_breaks_from_f64(min, max, zygos::TemporalUnit::Date, n)
        } else {
            // For non-pretty, use integer breaks since dates are whole days
            integer_breaks(min, max, n, false)
        }
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        // Evenly spaced minors for an explicit count, or for majors that don't
        // step by a calendar interval; calendar-stepped majors get calendar
        // minors during scale resolution instead.
        minor_breaks_linear(major_breaks, n, range)
    }

    fn default_minor_break_count(&self) -> usize {
        3
    }

    fn wrap_numeric(&self, value: f64) -> ArrayElement {
        ArrayElement::Date(value as i32)
    }

    fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        match elem {
            ArrayElement::String(s) => {
                ArrayElement::from_date_string(s).unwrap_or_else(|| elem.clone())
            }
            ArrayElement::Number(n) => self.wrap_numeric(*n),
            // Date values pass through unchanged
            ArrayElement::Date(_) => elem.clone(),
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for Date {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_transform_kind() {
        let t = Date;
        assert_eq!(t.transform_kind(), TransformKind::Date);
    }

    #[test]
    fn test_date_name() {
        let t = Date;
        assert_eq!(t.name(), "date");
    }

    #[test]
    fn test_date_domain() {
        let t = Date;
        let (min, max) = t.allowed_domain();
        // Date domain allows negative (before epoch) and positive dates
        assert!(min < 0.0);
        assert!(max > 0.0);
    }

    #[test]
    fn test_date_transform_is_identity() {
        let t = Date;
        assert_eq!(t.transform(100.0), 100.0);
        assert_eq!(t.transform(-50.0), -50.0);
        assert_eq!(t.inverse(100.0), 100.0);
        assert_eq!(t.inverse(-50.0), -50.0);
    }

    #[test]
    fn test_date_breaks_follow_the_calendar() {
        let t = Date;
        // Five years → year starts.
        assert_eq!(
            t.calculate_breaks(0.0, 1826.0, 5, true),
            vec![0.0, 365.0, 730.0, 1096.0, 1461.0, 1826.0]
        );
        // Six months → month starts.
        assert_eq!(
            t.calculate_breaks(0.0, 180.0, 6, true),
            vec![0.0, 31.0, 59.0, 90.0, 120.0, 151.0]
        );
        // Four weeks → Mondays (1970-01-05 is the first).
        assert_eq!(
            t.calculate_breaks(0.0, 28.0, 5, true),
            vec![4.0, 11.0, 18.0, 25.0]
        );
    }
}
