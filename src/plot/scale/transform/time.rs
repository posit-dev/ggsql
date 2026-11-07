//! Time transform implementation
//!
//! Transforms Time data (nanoseconds since midnight) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice time intervals.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{linear_breaks, minor_breaks_linear};
use crate::plot::ArrayElement;
use zygos::breaks::temporal_breaks_from_f64;

/// Time transform - for time data (nanoseconds since midnight)
///
/// This transform works on the numeric representation of time (nanoseconds since midnight).
/// The transform/inverse functions are identity (pass-through), but break calculation
/// produces sensible time intervals.
#[derive(Debug, Clone, Copy)]
pub struct Time;

// Nanoseconds per time unit
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
const NANOS_PER_MINUTE: f64 = 60.0 * NANOS_PER_SECOND;
const NANOS_PER_HOUR: f64 = 60.0 * NANOS_PER_MINUTE;

// Maximum time value (24 hours in nanoseconds)
const MAX_TIME_NANOS: f64 = 24.0 * NANOS_PER_HOUR;

impl TransformTrait for Time {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::Time
    }

    fn name(&self) -> &'static str {
        "time"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        // Time is nanoseconds since midnight: 0 to 24 hours
        (0.0, MAX_TIME_NANOS)
    }

    fn transform(&self, value: f64) -> f64 {
        // Identity transform - time stays in nanoseconds-since-midnight space
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

        // Clamp to valid time range
        let min = min.max(0.0);
        let max = max.min(MAX_TIME_NANOS);

        if pretty {
            // A clock interval (hours, minutes, seconds) sized to about `n`
            // breaks, stepped from midnight.
            temporal_breaks_from_f64(min, max, zygos::TemporalUnit::Time, n)
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
        ArrayElement::Time(value as i64)
    }

    fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        match elem {
            ArrayElement::String(s) => {
                ArrayElement::from_time_string(s).unwrap_or_else(|| elem.clone())
            }
            ArrayElement::Number(n) => self.wrap_numeric(*n),
            // Time values pass through unchanged
            ArrayElement::Time(_) => elem.clone(),
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_transform_kind() {
        let t = Time;
        assert_eq!(t.transform_kind(), TransformKind::Time);
    }

    #[test]
    fn test_time_name() {
        let t = Time;
        assert_eq!(t.name(), "time");
    }

    #[test]
    fn test_time_domain() {
        let t = Time;
        let (min, max) = t.allowed_domain();
        // Time domain is 0 to 24 hours in nanoseconds
        assert_eq!(min, 0.0);
        assert_eq!(max, 24.0 * NANOS_PER_HOUR);
    }

    #[test]
    fn test_time_transform_is_identity() {
        let t = Time;
        assert_eq!(t.transform(100.0), 100.0);
        assert_eq!(t.inverse(100.0), 100.0);
    }

    #[test]
    fn test_time_breaks_follow_the_clock() {
        let t = Time;
        // Two hours with about eight breaks → every fifteen minutes.
        let breaks = t.calculate_breaks(0.0, 2.0 * NANOS_PER_HOUR, 8, true);
        let minutes: Vec<f64> = breaks.iter().map(|b| b / NANOS_PER_MINUTE).collect();
        assert_eq!(
            minutes,
            vec![0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
        );
    }
}
