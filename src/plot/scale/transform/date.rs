//! Date transform implementation
//!
//! Transforms Date data (days since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals (years, months, weeks, days).

use chrono::Datelike;

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::{integer_breaks, minor_breaks_linear};
use crate::plot::ArrayElement;

/// Date transform - for date data (days since epoch)
///
/// This transform works on the numeric representation of dates (days since Unix epoch).
/// The transform/inverse functions are identity (pass-through), but break calculation
/// produces sensible temporal intervals.
#[derive(Debug, Clone, Copy)]
pub struct Date;

// Date interval types for break calculation
#[derive(Debug, Clone, Copy, PartialEq)]
enum DateInterval {
    Year,
    Quarter,
    Month,
    Week,
    Day,
}

impl DateInterval {
    /// Approximate number of days in each interval
    fn days(&self) -> f64 {
        match self {
            DateInterval::Year => 365.25,
            DateInterval::Quarter => 91.3125, // 365.25 / 4
            DateInterval::Month => 30.4375,   // 365.25 / 12
            DateInterval::Week => 7.0,
            DateInterval::Day => 1.0,
        }
    }

    /// Calculate expected number of breaks for this interval over the given span
    fn expected_breaks(&self, span_days: f64) -> f64 {
        span_days / self.days()
    }

    /// Select appropriate interval and step based on span in days and desired break count.
    /// Uses tolerance-based search: tries each interval from largest to smallest,
    /// stops when within ~20% of requested n, then calculates a nice step multiplier.
    fn select(span_days: f64, n: usize) -> (Self, usize) {
        let n_f64 = n as f64;
        let tolerance = 0.2; // 20% tolerance
        let min_breaks = n_f64 * (1.0 - tolerance);
        let max_breaks = n_f64 * (1.0 + tolerance);

        // Intervals from largest to smallest
        let intervals = [
            DateInterval::Year,
            DateInterval::Quarter,
            DateInterval::Month,
            DateInterval::Week,
            DateInterval::Day,
        ];

        for &interval in &intervals {
            let expected = interval.expected_breaks(span_days);

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
                    DateInterval::Year => nice_step(raw_step) as usize,
                    DateInterval::Quarter => nice_quarter_step(raw_step),
                    DateInterval::Month => nice_month_step(raw_step),
                    DateInterval::Week => nice_week_step(raw_step),
                    DateInterval::Day => nice_step(raw_step) as usize,
                };
                let step = nice.max(1);

                // Verify the stepped interval is reasonable
                let stepped_breaks = expected / step as f64;
                if stepped_breaks >= 1.0 {
                    return (interval, step);
                }
            }
        }

        // Fallback: use Day with step calculation
        let expected = DateInterval::Day.expected_breaks(span_days);
        let step = (nice_step(expected / n_f64) as usize).max(1);
        (DateInterval::Day, step)
    }
}

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

        let span = max - min;
        let (interval, step) = DateInterval::select(span, n);

        if pretty {
            calculate_pretty_date_breaks(min, max, interval, step)
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
        // Use linear minor breaks in day-space
        minor_breaks_linear(major_breaks, n, range)
    }

    fn default_minor_break_count(&self) -> usize {
        // 3 minor ticks per major interval works well for dates
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

/// Calculate pretty date breaks aligned to interval boundaries
fn calculate_pretty_date_breaks(
    min: f64,
    max: f64,
    interval: DateInterval,
    step: usize,
) -> Vec<f64> {
    let unix_epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();

    // Convert min/max to dates
    let min_date = unix_epoch + chrono::Duration::days(min.floor() as i64);
    let max_date = unix_epoch + chrono::Duration::days(max.ceil() as i64);

    let mut breaks = Vec::new();

    match interval {
        DateInterval::Year => {
            // Start at the beginning of the year containing min_date
            let start_year = min_date.year();
            let end_year = max_date.year();

            // Use the step from interval selection
            let step = step as i32;
            let aligned_start = (start_year / step) * step;

            let mut year = aligned_start;
            while year <= end_year + step {
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, 1, 1) {
                    let days = (date - unix_epoch).num_days() as f64;
                    if days >= min && days <= max {
                        breaks.push(days);
                    }
                }
                year += step;
            }
        }
        DateInterval::Quarter => {
            // Start at the beginning of the quarter containing min_date
            let start_year = min_date.year();
            let start_quarter = (min_date.month() - 1) / 3;

            let end_year = max_date.year();
            let end_quarter = (max_date.month() - 1) / 3;

            // Align to step boundary
            let aligned_start_quarter = (start_quarter / step as u32) * step as u32;

            let mut year = start_year;
            let mut quarter = aligned_start_quarter;

            while year < end_year || (year == end_year && quarter <= end_quarter) {
                let month = quarter * 3 + 1;
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, 1) {
                    let days = (date - unix_epoch).num_days() as f64;
                    if days >= min && days <= max {
                        breaks.push(days);
                    }
                }
                quarter += step as u32;
                if quarter > 3 {
                    // Calculate how many years to advance and remaining quarters
                    let years_advance = quarter / 4;
                    quarter %= 4;
                    year += years_advance as i32;
                }
            }
        }
        DateInterval::Month => {
            // Start at the beginning of the month containing min_date
            let start_year = min_date.year();
            let start_month = min_date.month();

            let end_year = max_date.year();
            let end_month = max_date.month();

            // Use the step from interval selection
            let mut year = start_year;
            let mut month = ((start_month - 1) / step as u32) * step as u32 + 1;

            while year < end_year || (year == end_year && month <= end_month) {
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, 1) {
                    let days = (date - unix_epoch).num_days() as f64;
                    if days >= min && days <= max {
                        breaks.push(days);
                    }
                }
                month += step as u32;
                if month > 12 {
                    month -= 12;
                    year += 1;
                }
            }
        }
        DateInterval::Week => {
            // Start at the Monday on or before min_date
            let start_days = min.floor() as i64;
            // weekday() returns 0 for Monday, 6 for Sunday
            let weekday = (start_days.rem_euclid(7) + 3) % 7; // Convert to Mon=0
            let first_monday = start_days - weekday;

            let end_days = max.ceil() as i64;
            let step_days = (step * 7) as i64; // step weeks in days

            let mut day = first_monday;
            while day <= end_days {
                let days = day as f64;
                if days >= min && days <= max {
                    breaks.push(days);
                }
                day += step_days;
            }
        }
        DateInterval::Day => {
            // Use the step from interval selection
            let step = step as i64;

            let start_day = (min / step as f64).floor() as i64 * step;
            let end_day = max.ceil() as i64;

            let mut day = start_day;
            while day <= end_day {
                let days = day as f64;
                if days >= min && days <= max {
                    breaks.push(days);
                }
                day += step;
            }
        }
    }

    // Ensure we have at least min and max if the algorithm produced nothing
    if breaks.is_empty() {
        breaks.push(min);
        if max > min {
            breaks.push(max);
        }
    }

    breaks
}

/// Round to a "nice" step value (1, 2, 5, 10, 20, 50, etc.)
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

/// Nice step values for weeks (1, 2, 4)
fn nice_week_step(step: f64) -> usize {
    if step <= 1.5 {
        1
    } else if step <= 3.0 {
        2
    } else {
        4
    }
}

/// Nice step values for quarters (1, 2, 4)
fn nice_quarter_step(step: f64) -> usize {
    if step <= 1.5 {
        1
    } else if step <= 3.0 {
        2
    } else {
        4
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
    fn test_date_breaks_year_span() {
        let t = Date;
        // ~5 years span (in days)
        let min = 0.0; // 1970-01-01
        let max = 365.0 * 5.0; // ~1975
        let breaks = t.calculate_breaks(min, max, 5, true);
        assert!(!breaks.is_empty());
        // All breaks should be within range
        for &b in &breaks {
            assert!(b >= min && b <= max);
        }
    }

    #[test]
    fn test_date_breaks_month_span() {
        let t = Date;
        // ~6 months span
        let min = 0.0;
        let max = 180.0;
        let breaks = t.calculate_breaks(min, max, 6, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_date_breaks_week_span() {
        let t = Date;
        // ~4 weeks span
        let min = 0.0;
        let max = 28.0;
        let breaks = t.calculate_breaks(min, max, 5, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_date_breaks_day_span() {
        let t = Date;
        // ~7 days span
        let min = 0.0;
        let max = 7.0;
        let breaks = t.calculate_breaks(min, max, 7, true);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_date_breaks_linear() {
        let t = Date;
        let breaks = t.calculate_breaks(0.0, 100.0, 5, false);
        // Should have integer day breaks
        assert!(!breaks.is_empty());
        assert!(breaks[0] <= 0.0);
        assert!(*breaks.last().unwrap() >= 100.0);
        // All breaks should be integers (whole days)
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be a whole day", b);
        }
        // Breaks should be evenly spaced
        if breaks.len() >= 2 {
            let step = breaks[1] - breaks[0];
            for i in 1..breaks.len() {
                let gap = breaks[i] - breaks[i - 1];
                assert!(
                    (gap - step).abs() < 0.01,
                    "Date breaks should be evenly spaced"
                );
            }
        }
    }

    #[test]
    fn test_date_interval_selection() {
        // Large span (10 years, n=5) -> year with step
        let (interval, step) = DateInterval::select(3650.0, 5);
        assert_eq!(interval, DateInterval::Year);
        assert!(step >= 1);

        // Medium span (6 months, n=6) -> month
        let (interval, step) = DateInterval::select(180.0, 6);
        assert_eq!(interval, DateInterval::Month);
        assert!(step >= 1);

        // Small span (4 weeks, n=4) -> week
        let (interval, step) = DateInterval::select(28.0, 4);
        assert_eq!(interval, DateInterval::Week);
        assert!(step >= 1);

        // Very small span (7 days, n=7) -> day
        let (interval, step) = DateInterval::select(7.0, 7);
        assert_eq!(interval, DateInterval::Day);
        assert!(step >= 1);
    }

    #[test]
    fn test_date_interval_selection_airquality() {
        // airquality data: ~150 days, n=7
        // Previously: selected Week, generated ~22 breaks
        // Now: should select Month (150/30 ≈ 5 breaks, within 20% of 7)
        let (interval, step) = DateInterval::select(150.0, 7);
        // Month gives ~5 breaks (within tolerance of 7), or
        // Week with step would give ~5 breaks
        let expected_breaks = interval.expected_breaks(150.0) / step as f64;
        assert!(
            (3.0..=10.0).contains(&expected_breaks),
            "Expected 3-10 breaks for 150 days, n=7, got {} ({:?} with step {})",
            expected_breaks,
            interval,
            step
        );
    }

    #[test]
    fn test_date_breaks_airquality_count() {
        let t = Date;
        // ~150 days (May-September), n=7
        let min = 0.0;
        let max = 150.0;
        let breaks = t.calculate_breaks(min, max, 7, true);

        // Should have roughly 5-10 breaks, not 22
        assert!(
            breaks.len() >= 3 && breaks.len() <= 12,
            "Expected 3-12 breaks for 150 days, n=7, got {}",
            breaks.len()
        );
    }

    #[test]
    fn test_nice_step() {
        assert_eq!(nice_step(1.0), 1.0);
        assert_eq!(nice_step(1.5), 1.0); // 1.5 rounds down to 1.0
        assert_eq!(nice_step(1.6), 2.0); // 1.6 rounds up to 2.0
        assert_eq!(nice_step(3.0), 2.0); // 3.0 rounds to 2.0
        assert_eq!(nice_step(3.5), 5.0); // 3.5 rounds up to 5.0
        assert_eq!(nice_step(7.0), 5.0);
        assert_eq!(nice_step(8.0), 10.0);
        assert_eq!(nice_step(15.0), 10.0); // 15 = 1.5 * 10, rounds to 1.0 * 10
        assert_eq!(nice_step(16.0), 20.0); // 16 = 1.6 * 10, rounds to 2.0 * 10
    }
}
