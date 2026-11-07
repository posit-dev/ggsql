//! DateTime transform implementation
//!
//! Transforms DateTime data (microseconds since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{linear_breaks, minor_breaks_linear};
use crate::plot::ArrayElement;
use zygos::breaks::temporal_breaks_from_f64;

/// DateTime transform - for datetime data (microseconds since epoch)
///
/// This transform works on the numeric representation of datetimes (microseconds since Unix epoch).
/// The transform/inverse functions are identity (pass-through), but break calculation
/// produces sensible temporal intervals.
#[derive(Debug, Clone, Copy)]
pub struct DateTime;

impl TransformTrait for DateTime {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::DateTime
    }

    fn name(&self) -> &'static str {
        "datetime"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        // Roughly year 1 to year 9999 in microseconds since epoch
        // i64::MIN/MAX is about +/- 292,000 years, so we can be generous
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn transform(&self, value: f64) -> f64 {
        // Identity transform - datetimes stay in microseconds-since-epoch space
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
            // A calendar interval (years down to seconds) sized to about `n`
            // breaks, stepped from its calendar grid.
            temporal_breaks_from_f64(min, max, zygos::TemporalUnit::DateTime, n)
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
        // Evenly spaced minors for an explicit count, or for majors that don't
        // step by a calendar interval; calendar-stepped majors get calendar
        // minors during scale resolution instead.
        minor_breaks_linear(major_breaks, n, range)
    }

    fn default_minor_break_count(&self) -> usize {
        3
    }

    fn wrap_numeric(&self, value: f64) -> ArrayElement {
        ArrayElement::DateTime(value as i64)
    }

    fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        match elem {
            ArrayElement::String(s) => {
                ArrayElement::from_datetime_string(s).unwrap_or_else(|| elem.clone())
            }
            ArrayElement::Number(n) => self.wrap_numeric(*n),
            // DateTime values pass through unchanged
            ArrayElement::DateTime(_) => elem.clone(),
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Microseconds per time unit
    const MICROS_PER_SECOND: f64 = 1_000_000.0;
    const MICROS_PER_MINUTE: f64 = 60.0 * MICROS_PER_SECOND;
    const MICROS_PER_HOUR: f64 = 60.0 * MICROS_PER_MINUTE;
    const MICROS_PER_DAY: f64 = 24.0 * MICROS_PER_HOUR;

    #[test]
    fn test_datetime_transform_kind() {
        let t = DateTime;
        assert_eq!(t.transform_kind(), TransformKind::DateTime);
    }

    #[test]
    fn test_datetime_name() {
        let t = DateTime;
        assert_eq!(t.name(), "datetime");
    }

    #[test]
    fn test_datetime_domain() {
        let t = DateTime;
        let (min, max) = t.allowed_domain();
        // DateTime has infinite domain
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_datetime_transform_is_identity() {
        let t = DateTime;
        assert_eq!(t.transform(100.0), 100.0);
        assert_eq!(t.transform(-50.0), -50.0);
        assert_eq!(t.inverse(100.0), 100.0);
    }

    #[test]
    fn test_datetime_breaks_follow_the_clock() {
        let t = DateTime;
        // A day with about eight breaks → every three hours from midnight.
        let breaks = t.calculate_breaks(0.0, 24.0 * MICROS_PER_HOUR, 8, true);
        let hours: Vec<f64> = breaks.iter().map(|b| b / MICROS_PER_HOUR).collect();
        assert_eq!(
            hours,
            vec![0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
        );
        // An hour with about six breaks → every ten minutes.
        let breaks = t.calculate_breaks(0.0, 60.0 * MICROS_PER_MINUTE, 6, true);
        let minutes: Vec<f64> = breaks.iter().map(|b| b / MICROS_PER_MINUTE).collect();
        assert_eq!(minutes, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }
}
