//! Time transform implementation
//!
//! Transforms Time data (nanoseconds since midnight) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice time intervals.

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::minor_breaks_linear;
use crate::plot::ArrayElement;

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

// Time interval types for break calculation
#[derive(Debug, Clone, Copy, PartialEq)]
enum TimeInterval {
    Hour,
    Minute,
    Second,
    Millisecond,
}

impl TimeInterval {
    /// Nanoseconds in each interval
    fn nanos(&self) -> f64 {
        match self {
            TimeInterval::Hour => NANOS_PER_HOUR,
            TimeInterval::Minute => NANOS_PER_MINUTE,
            TimeInterval::Second => NANOS_PER_SECOND,
            TimeInterval::Millisecond => 1_000_000.0,
        }
    }

    /// Calculate expected number of breaks for this interval over the given span
    fn expected_breaks(&self, span_nanos: f64) -> f64 {
        span_nanos / self.nanos()
    }

    /// Select appropriate interval and step based on span and desired break count.
    /// Uses tolerance-based search: tries each interval from largest to smallest,
    /// stops when within ~20% of requested n, then calculates a nice step multiplier.
    fn select(span_nanos: f64, n: usize) -> (Self, usize) {
        let n_f64 = n as f64;
        let tolerance = 0.2; // 20% tolerance
        let min_breaks = n_f64 * (1.0 - tolerance);
        let max_breaks = n_f64 * (1.0 + tolerance);

        // Intervals from largest to smallest
        let intervals = [
            TimeInterval::Hour,
            TimeInterval::Minute,
            TimeInterval::Second,
            TimeInterval::Millisecond,
        ];

        for &interval in &intervals {
            let expected = interval.expected_breaks(span_nanos);

            // Skip if this interval produces too few breaks
            if expected < 1.0 {
                continue;
            }

            // If within tolerance, use step=1
            if expected >= min_breaks && expected <= max_breaks {
                return (interval, 1);
            }

            // If too many breaks, calculate a nice step
            if expected > max_breaks {
                let raw_step = expected / n_f64;
                let nice = match interval {
                    TimeInterval::Hour => nice_hour_step(raw_step) as usize,
                    TimeInterval::Minute => nice_minute_step(raw_step) as usize,
                    TimeInterval::Second => nice_second_step(raw_step) as usize,
                    TimeInterval::Millisecond => nice_step(raw_step) as usize,
                };
                let step = nice.max(1);

                // Verify the stepped interval is reasonable
                let stepped_breaks = expected / step as f64;
                if stepped_breaks >= 1.0 {
                    return (interval, step);
                }
            }
        }

        // Fallback: use Millisecond with step calculation
        let expected = TimeInterval::Millisecond.expected_breaks(span_nanos);
        let step = (nice_step(expected / n_f64) as usize).max(1);
        (TimeInterval::Millisecond, step)
    }
}

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

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite() && (0.0..=MAX_TIME_NANOS).contains(&value)
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

        let span = max - min;
        let (interval, step) = TimeInterval::select(span, n);

        if pretty {
            calculate_pretty_time_breaks(min, max, interval, step)
        } else {
            calculate_linear_time_breaks(min, max, n)
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

/// Calculate pretty time breaks aligned to interval boundaries
fn calculate_pretty_time_breaks(min: f64, max: f64, interval: TimeInterval, step: usize) -> Vec<f64> {
    let mut breaks = Vec::new();

    match interval {
        TimeInterval::Hour => {
            let step_nanos = (step as i64) * NANOS_PER_HOUR as i64;

            let start_nanos = (min / NANOS_PER_HOUR).floor() as i64 * NANOS_PER_HOUR as i64;

            let mut nanos = start_nanos;
            while (nanos as f64) <= max {
                let ns = nanos as f64;
                if ns >= min && ns <= max {
                    breaks.push(ns);
                }
                nanos += step_nanos;
            }
        }
        TimeInterval::Minute => {
            let step_nanos = (step as i64) * NANOS_PER_MINUTE as i64;

            let start_nanos = (min / NANOS_PER_MINUTE).floor() as i64 * NANOS_PER_MINUTE as i64;

            let mut nanos = start_nanos;
            while (nanos as f64) <= max {
                let ns = nanos as f64;
                if ns >= min && ns <= max {
                    breaks.push(ns);
                }
                nanos += step_nanos;
            }
        }
        TimeInterval::Second => {
            let step_nanos = (step as i64) * NANOS_PER_SECOND as i64;

            let start_nanos = (min / NANOS_PER_SECOND).floor() as i64 * NANOS_PER_SECOND as i64;

            let mut nanos = start_nanos;
            while (nanos as f64) <= max {
                let ns = nanos as f64;
                if ns >= min && ns <= max {
                    breaks.push(ns);
                }
                nanos += step_nanos;
            }
        }
        TimeInterval::Millisecond => {
            let step_nanos = (step as i64) * 1_000_000;

            let start_nanos = (min / 1_000_000.0).floor() as i64 * 1_000_000;

            let mut nanos = start_nanos;
            while (nanos as f64) <= max {
                let ns = nanos as f64;
                if ns >= min && ns <= max {
                    breaks.push(ns);
                }
                nanos += step_nanos;
            }
        }
    }

    if breaks.is_empty() {
        breaks.push(min);
        if max > min {
            breaks.push(max);
        }
    }

    breaks
}

/// Calculate linear breaks in nanosecond-space
fn calculate_linear_time_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![min];
    }

    let step = (max - min) / (n - 1) as f64;
    (0..n).map(|i| min + i as f64 * step).collect()
}

/// Round to a "nice" step value
fn nice_step(step: f64) -> f64 {
    if step <= 0.0 {
        return 1.0;
    }

    let magnitude = 10_f64.powf(step.log10().floor());
    let residual = step / magnitude;

    let nice = if residual <= 1.5 {
        1.0
    } else if residual <= 3.0 {
        2.0
    } else if residual <= 7.0 {
        5.0
    } else {
        10.0
    };

    nice * magnitude
}

/// Nice step values for hours (1, 2, 3, 4, 6, 12)
fn nice_hour_step(step: f64) -> f64 {
    if step <= 1.0 {
        1.0
    } else if step <= 2.0 {
        2.0
    } else if step <= 3.0 {
        3.0
    } else if step <= 4.0 {
        4.0
    } else if step <= 6.0 {
        6.0
    } else {
        12.0
    }
}

/// Nice step values for minutes (1, 2, 5, 10, 15, 30)
fn nice_minute_step(step: f64) -> f64 {
    if step <= 1.0 {
        1.0
    } else if step <= 2.0 {
        2.0
    } else if step <= 5.0 {
        5.0
    } else if step <= 10.0 {
        10.0
    } else if step <= 15.0 {
        15.0
    } else {
        30.0
    }
}

/// Nice step values for seconds (1, 2, 5, 10, 15, 30)
fn nice_second_step(step: f64) -> f64 {
    if step <= 1.0 {
        1.0
    } else if step <= 2.0 {
        2.0
    } else if step <= 5.0 {
        5.0
    } else if step <= 10.0 {
        10.0
    } else if step <= 15.0 {
        15.0
    } else {
        30.0
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
        assert_eq!(min, 0.0);
        assert_eq!(max, 24.0 * NANOS_PER_HOUR);

        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(12.0 * NANOS_PER_HOUR));
        assert!(!t.is_value_in_domain(-1.0));
        assert!(!t.is_value_in_domain(25.0 * NANOS_PER_HOUR));
    }

    #[test]
    fn test_time_transform_is_identity() {
        let t = Time;
        assert_eq!(t.transform(100.0), 100.0);
        assert_eq!(t.inverse(100.0), 100.0);
    }

    #[test]
    fn test_time_breaks_hour_span() {
        let t = Time;
        // Full day
        let min = 0.0;
        let max = 24.0 * NANOS_PER_HOUR;
        let breaks = t.calculate_breaks(min, max, 8, true);
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b >= min && b <= max);
        }
    }

    #[test]
    fn test_time_breaks_minute_span() {
        let t = Time;
        // 2 hours
        let min = 0.0;
        let max = 2.0 * NANOS_PER_HOUR;
        let breaks = t.calculate_breaks(min, max, 8, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_time_breaks_second_span() {
        let t = Time;
        // 5 minutes
        let min = 0.0;
        let max = 5.0 * NANOS_PER_MINUTE;
        let breaks = t.calculate_breaks(min, max, 5, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_time_breaks_linear() {
        let t = Time;
        let breaks = t.calculate_breaks(0.0, NANOS_PER_HOUR, 5, false);
        assert_eq!(breaks.len(), 5);
        assert_eq!(breaks[0], 0.0);
        assert_eq!(breaks[4], NANOS_PER_HOUR);
    }

    #[test]
    fn test_time_interval_selection() {
        // Full day (24 hours, n=8) -> hour with step
        let (interval, step) = TimeInterval::select(24.0 * NANOS_PER_HOUR, 8);
        assert_eq!(interval, TimeInterval::Hour);
        assert!(step >= 1);

        // Hour span (1 hour, n=6) -> minute with step
        let (interval, step) = TimeInterval::select(NANOS_PER_HOUR, 6);
        assert_eq!(interval, TimeInterval::Minute);
        assert!(step >= 1);

        // Minute span (1 minute, n=6) -> second with step
        let (interval, step) = TimeInterval::select(NANOS_PER_MINUTE, 6);
        assert_eq!(interval, TimeInterval::Second);
        assert!(step >= 1);

        // Second span (1 second, n=10) -> millisecond with step
        let (interval, step) = TimeInterval::select(NANOS_PER_SECOND, 10);
        assert_eq!(interval, TimeInterval::Millisecond);
        assert!(step >= 1);
    }

    #[test]
    fn test_nice_hour_step() {
        assert_eq!(nice_hour_step(1.0), 1.0);
        assert_eq!(nice_hour_step(2.5), 3.0);
        assert_eq!(nice_hour_step(5.0), 6.0);
        assert_eq!(nice_hour_step(10.0), 12.0);
    }

    #[test]
    fn test_nice_minute_step() {
        assert_eq!(nice_minute_step(1.0), 1.0);
        assert_eq!(nice_minute_step(3.0), 5.0);
        assert_eq!(nice_minute_step(12.0), 15.0);
        assert_eq!(nice_minute_step(25.0), 30.0);
    }

    #[test]
    fn test_nice_second_step() {
        assert_eq!(nice_second_step(1.0), 1.0);
        assert_eq!(nice_second_step(3.0), 5.0);
        assert_eq!(nice_second_step(12.0), 15.0);
        assert_eq!(nice_second_step(25.0), 30.0);
    }
}
