//! DateTime transform implementation
//!
//! Transforms DateTime data (microseconds since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals.

use chrono::Datelike;

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::minor_breaks_linear;

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

    /// Select appropriate interval based on span and desired break count
    fn select(span_micros: f64, n: usize) -> Self {
        let target_interval = span_micros / n as f64;

        if target_interval >= 365.0 * MICROS_PER_DAY {
            DateTimeInterval::Year
        } else if target_interval >= 28.0 * MICROS_PER_DAY {
            DateTimeInterval::Month
        } else if target_interval >= MICROS_PER_DAY {
            DateTimeInterval::Day
        } else if target_interval >= MICROS_PER_HOUR {
            DateTimeInterval::Hour
        } else if target_interval >= MICROS_PER_MINUTE {
            DateTimeInterval::Minute
        } else {
            DateTimeInterval::Second
        }
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
        let interval = DateTimeInterval::select(span, n);

        if pretty {
            calculate_pretty_datetime_breaks(min, max, n, interval)
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
}

/// Calculate pretty datetime breaks aligned to interval boundaries
fn calculate_pretty_datetime_breaks(
    min: f64,
    max: f64,
    n: usize,
    interval: DateTimeInterval,
) -> Vec<f64> {
    let mut breaks = Vec::new();

    match interval {
        DateTimeInterval::Year => {
            let min_dt = micros_to_datetime(min as i64);
            let max_dt = micros_to_datetime(max as i64);

            let start_year = min_dt.year();
            let end_year = max_dt.year();

            let year_span = (end_year - start_year + 1) as usize;
            let step = nice_step(year_span as f64 / n as f64) as i32;

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

            let total_months =
                (end_year - start_year) * 12 + (end_month as i32 - start_month as i32 + 1);
            let step = ((total_months as usize) / n).max(1);
            let step = match step {
                1 => 1,
                2 => 2,
                3..=4 => 3,
                5..=8 => 6,
                _ => 12,
            };

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
            let span = max - min;
            let days = span / MICROS_PER_DAY;
            let step_days = nice_step(days / n as f64) as i64;

            let start_micros = (min / MICROS_PER_DAY).floor() as i64 * MICROS_PER_DAY as i64;
            let step_micros = step_days * MICROS_PER_DAY as i64;

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
            let span = max - min;
            let hours = span / MICROS_PER_HOUR;
            let step_hours = nice_hour_step(hours / n as f64) as i64;

            let start_micros = (min / MICROS_PER_HOUR).floor() as i64 * MICROS_PER_HOUR as i64;
            let step_micros = step_hours * MICROS_PER_HOUR as i64;

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
            let span = max - min;
            let minutes = span / MICROS_PER_MINUTE;
            let step_minutes = nice_minute_step(minutes / n as f64) as i64;

            let start_micros = (min / MICROS_PER_MINUTE).floor() as i64 * MICROS_PER_MINUTE as i64;
            let step_micros = step_minutes * MICROS_PER_MINUTE as i64;

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
            let span = max - min;
            let seconds = span / MICROS_PER_SECOND;
            let step_seconds = nice_step(seconds / n as f64);

            let start_micros =
                (min / MICROS_PER_SECOND).floor() as i64 * MICROS_PER_SECOND as i64;
            let step_micros = (step_seconds * MICROS_PER_SECOND) as i64;

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
        .unwrap_or_else(|| chrono::NaiveDateTime::default())
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
        // Large span -> year
        assert_eq!(
            DateTimeInterval::select(365.0 * MICROS_PER_DAY * 5.0, 5),
            DateTimeInterval::Year
        );

        // Day span -> day
        assert_eq!(
            DateTimeInterval::select(30.0 * MICROS_PER_DAY, 5),
            DateTimeInterval::Day
        );

        // Hour span -> hour
        assert_eq!(
            DateTimeInterval::select(24.0 * MICROS_PER_HOUR, 8),
            DateTimeInterval::Hour
        );

        // Minute span -> minute
        assert_eq!(
            DateTimeInterval::select(60.0 * MICROS_PER_MINUTE, 6),
            DateTimeInterval::Minute
        );
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
