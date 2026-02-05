//! DateTime transform implementation
//!
//! Transforms DateTime data (microseconds since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals.

use chrono::Datelike;

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::minor_breaks_linear;
use crate::plot::ArrayElement;

/// DateTime transform - for datetime data (microseconds since epoch)
///
/// This transform works on the numeric representation of datetimes (microseconds since Unix epoch).
/// The transform/inverse functions are identity (pass-through), but break calculation
/// produces sensible temporal intervals.
#[derive(Debug, Clone, Copy)]
pub struct DateTime;

// Microseconds per time unit
const MICROS_PER_SECOND: f64 = 1_000_000.0;
const MICROS_PER_MINUTE: f64 = 60.0 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: f64 = 60.0 * MICROS_PER_MINUTE;
const MICROS_PER_DAY: f64 = 24.0 * MICROS_PER_HOUR;

// DateTime interval types for break calculation
#[derive(Debug, Clone, Copy, PartialEq)]
enum DateTimeInterval {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

impl DateTimeInterval {
    /// Approximate microseconds in each interval
    fn micros(&self) -> f64 {
        match self {
            DateTimeInterval::Year => 365.25 * MICROS_PER_DAY,
            DateTimeInterval::Month => 30.4375 * MICROS_PER_DAY,
            DateTimeInterval::Day => MICROS_PER_DAY,
            DateTimeInterval::Hour => MICROS_PER_HOUR,
            DateTimeInterval::Minute => MICROS_PER_MINUTE,
            DateTimeInterval::Second => MICROS_PER_SECOND,
        }
    }

    /// Calculate expected number of breaks for this interval over the given span
    fn expected_breaks(&self, span_micros: f64) -> f64 {
        span_micros / self.micros()
    }

    /// Select appropriate interval and step based on span and desired break count.
    /// Uses tolerance-based search: tries each interval from largest to smallest,
    /// stops when within ~20% of requested n, then calculates a nice step multiplier.
    fn select(span_micros: f64, n: usize) -> (Self, usize) {
        let n_f64 = n as f64;
        let tolerance = 0.2; // 20% tolerance
        let min_breaks = n_f64 * (1.0 - tolerance);
        let max_breaks = n_f64 * (1.0 + tolerance);

        // Intervals from largest to smallest
        let intervals = [
            DateTimeInterval::Year,
            DateTimeInterval::Month,
            DateTimeInterval::Day,
            DateTimeInterval::Hour,
            DateTimeInterval::Minute,
            DateTimeInterval::Second,
        ];

        for &interval in &intervals {
            let expected = interval.expected_breaks(span_micros);

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
                    DateTimeInterval::Year => nice_step(raw_step) as usize,
                    DateTimeInterval::Month => nice_month_step(raw_step),
                    DateTimeInterval::Day => nice_step(raw_step) as usize,
                    DateTimeInterval::Hour => nice_hour_step(raw_step) as usize,
                    DateTimeInterval::Minute => nice_minute_step(raw_step) as usize,
                    DateTimeInterval::Second => nice_step(raw_step) as usize,
                };
                let step = nice.max(1);

                // Verify the stepped interval is reasonable
                let stepped_breaks = expected / step as f64;
                if stepped_breaks >= 1.0 {
                    return (interval, step);
                }
            }
        }

        // Fallback: use Second with step calculation
        let expected = DateTimeInterval::Second.expected_breaks(span_micros);
        let step = (nice_step(expected / n_f64) as usize).max(1);
        (DateTimeInterval::Second, step)
    }
}

/// Nice step values for months (1, 2, 3, 6, 12)
fn nice_month_step(step: f64) -> usize {
    if step <= 1.0 {
        1
    } else if step <= 2.0 {
        2
    } else if step <= 4.0 {
        3
    } else if step <= 8.0 {
        6
    } else {
        12
    }
}

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

    fn is_value_in_domain(&self, value: f64) -> bool {
        value.is_finite()
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

        let span = max - min;
        let (interval, step) = DateTimeInterval::select(span, n);

        if pretty {
            calculate_pretty_datetime_breaks(min, max, interval, step)
        } else {
            calculate_linear_datetime_breaks(min, max, n)
        }
    }

    fn calculate_minor_breaks(
        &self,
        major_breaks: &[f64],
        n: usize,
        range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        // Use linear minor breaks in microsecond-space
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

/// Calculate pretty datetime breaks aligned to interval boundaries
fn calculate_pretty_datetime_breaks(
    min: f64,
    max: f64,
    interval: DateTimeInterval,
    step: usize,
) -> Vec<f64> {
    let mut breaks = Vec::new();

    match interval {
        DateTimeInterval::Year => {
            let min_dt = micros_to_datetime(min as i64);
            let max_dt = micros_to_datetime(max as i64);

            let start_year = min_dt.year();
            let end_year = max_dt.year();

            let step = step as i32;
            let aligned_start = (start_year / step) * step;

            let mut year = aligned_start;
            while year <= end_year + step {
                if let Some(dt) = chrono::NaiveDate::from_ymd_opt(year, 1, 1) {
                    let micros = datetime_to_micros(dt.and_hms_opt(0, 0, 0).unwrap());
                    if micros >= min && micros <= max {
                        breaks.push(micros);
                    }
                }
                year += step;
            }
        }
        DateTimeInterval::Month => {
            let min_dt = micros_to_datetime(min as i64);
            let max_dt = micros_to_datetime(max as i64);

            let start_year = min_dt.year();
            let start_month = min_dt.month();
            let end_year = max_dt.year();
            let end_month = max_dt.month();

            let mut year = start_year;
            let mut month = ((start_month - 1) / step as u32) * step as u32 + 1;

            while year < end_year || (year == end_year && month <= end_month) {
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, 1) {
                    let micros = datetime_to_micros(date.and_hms_opt(0, 0, 0).unwrap());
                    if micros >= min && micros <= max {
                        breaks.push(micros);
                    }
                }
                month += step as u32;
                if month > 12 {
                    month -= 12;
                    year += 1;
                }
            }
        }
        DateTimeInterval::Day => {
            let step_micros = (step as i64) * MICROS_PER_DAY as i64;

            let start_micros = (min / MICROS_PER_DAY).floor() as i64 * MICROS_PER_DAY as i64;

            let mut micros = start_micros;
            while (micros as f64) <= max {
                let m = micros as f64;
                if m >= min && m <= max {
                    breaks.push(m);
                }
                micros += step_micros;
            }
        }
        DateTimeInterval::Hour => {
            let step_micros = (step as i64) * MICROS_PER_HOUR as i64;

            let start_micros = (min / MICROS_PER_HOUR).floor() as i64 * MICROS_PER_HOUR as i64;

            let mut micros = start_micros;
            while (micros as f64) <= max {
                let m = micros as f64;
                if m >= min && m <= max {
                    breaks.push(m);
                }
                micros += step_micros;
            }
        }
        DateTimeInterval::Minute => {
            let step_micros = (step as i64) * MICROS_PER_MINUTE as i64;

            let start_micros = (min / MICROS_PER_MINUTE).floor() as i64 * MICROS_PER_MINUTE as i64;

            let mut micros = start_micros;
            while (micros as f64) <= max {
                let m = micros as f64;
                if m >= min && m <= max {
                    breaks.push(m);
                }
                micros += step_micros;
            }
        }
        DateTimeInterval::Second => {
            let step_micros = (step as i64) * MICROS_PER_SECOND as i64;

            let start_micros = (min / MICROS_PER_SECOND).floor() as i64 * MICROS_PER_SECOND as i64;

            let mut micros = start_micros;
            while (micros as f64) <= max {
                let m = micros as f64;
                if m >= min && m <= max {
                    breaks.push(m);
                }
                micros += step_micros;
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

/// Calculate linear breaks in microsecond-space
fn calculate_linear_datetime_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![min];
    }

    let step = (max - min) / (n - 1) as f64;
    (0..n).map(|i| min + i as f64 * step).collect()
}

/// Convert microseconds since epoch to NaiveDateTime
fn micros_to_datetime(micros: i64) -> chrono::NaiveDateTime {
    let secs = micros / 1_000_000;
    let nsecs = ((micros % 1_000_000).abs() * 1000) as u32;
    chrono::DateTime::from_timestamp(secs, nsecs)
        .map(|dt| dt.naive_utc())
        .unwrap_or_default()
}

/// Convert NaiveDateTime to microseconds since epoch
fn datetime_to_micros(dt: chrono::NaiveDateTime) -> f64 {
    dt.and_utc().timestamp_micros() as f64
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

/// Nice step values for hours (1, 2, 3, 4, 6, 12, 24)
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
    } else if step <= 12.0 {
        12.0
    } else {
        24.0
    }
}

/// Nice step values for minutes (1, 2, 5, 10, 15, 30, 60)
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
    } else if step <= 30.0 {
        30.0
    } else {
        60.0
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
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(-1_000_000_000_000.0));
        assert!(t.is_value_in_domain(1_000_000_000_000.0));
        assert!(!t.is_value_in_domain(f64::INFINITY));
    }

    #[test]
    fn test_datetime_transform_is_identity() {
        let t = DateTime;
        assert_eq!(t.transform(100.0), 100.0);
        assert_eq!(t.transform(-50.0), -50.0);
        assert_eq!(t.inverse(100.0), 100.0);
    }

    #[test]
    fn test_datetime_breaks_year_span() {
        let t = DateTime;
        // ~5 years span (in microseconds)
        let min = 0.0;
        let max = 5.0 * 365.25 * MICROS_PER_DAY;
        let breaks = t.calculate_breaks(min, max, 5, true);
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b >= min && b <= max);
        }
    }

    #[test]
    fn test_datetime_breaks_hour_span() {
        let t = DateTime;
        // ~24 hours span
        let min = 0.0;
        let max = 24.0 * MICROS_PER_HOUR;
        let breaks = t.calculate_breaks(min, max, 8, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_datetime_breaks_minute_span() {
        let t = DateTime;
        // ~60 minutes span
        let min = 0.0;
        let max = 60.0 * MICROS_PER_MINUTE;
        let breaks = t.calculate_breaks(min, max, 6, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_datetime_breaks_linear() {
        let t = DateTime;
        let breaks = t.calculate_breaks(0.0, 1_000_000.0, 5, false);
        assert_eq!(breaks.len(), 5);
        assert_eq!(breaks[0], 0.0);
        assert_eq!(breaks[4], 1_000_000.0);
    }

    #[test]
    fn test_datetime_interval_selection() {
        // Large span (5 years, n=5) -> year with step
        let (interval, step) = DateTimeInterval::select(365.0 * MICROS_PER_DAY * 5.0, 5);
        assert_eq!(interval, DateTimeInterval::Year);
        assert!(step >= 1);

        // Day span (30 days, n=5) -> day with step
        let (interval, step) = DateTimeInterval::select(30.0 * MICROS_PER_DAY, 5);
        assert_eq!(interval, DateTimeInterval::Day);
        assert!(step >= 1);

        // Hour span (24 hours, n=8) -> hour with step
        let (interval, step) = DateTimeInterval::select(24.0 * MICROS_PER_HOUR, 8);
        assert_eq!(interval, DateTimeInterval::Hour);
        assert!(step >= 1);

        // Minute span (60 minutes, n=6) -> minute with step
        let (interval, step) = DateTimeInterval::select(60.0 * MICROS_PER_MINUTE, 6);
        assert_eq!(interval, DateTimeInterval::Minute);
        assert!(step >= 1);
    }

    #[test]
    fn test_nice_hour_step() {
        assert_eq!(nice_hour_step(1.0), 1.0);
        assert_eq!(nice_hour_step(1.5), 2.0);
        assert_eq!(nice_hour_step(2.5), 3.0);
        assert_eq!(nice_hour_step(5.0), 6.0);
        assert_eq!(nice_hour_step(10.0), 12.0);
        assert_eq!(nice_hour_step(20.0), 24.0);
    }

    #[test]
    fn test_nice_minute_step() {
        assert_eq!(nice_minute_step(1.0), 1.0);
        assert_eq!(nice_minute_step(3.0), 5.0);
        assert_eq!(nice_minute_step(7.0), 10.0);
        assert_eq!(nice_minute_step(12.0), 15.0);
        assert_eq!(nice_minute_step(20.0), 30.0);
        assert_eq!(nice_minute_step(45.0), 60.0);
    }
}
