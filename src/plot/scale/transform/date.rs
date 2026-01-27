//! Date transform implementation
//!
//! Transforms Date data (days since epoch) to appropriate break positions.
//! The transform itself is identity (no numerical transformation), but the
//! break calculation produces nice temporal intervals (years, months, weeks, days).

use chrono::Datelike;

use super::{TransformKind, TransformTrait};
use crate::plot::scale::breaks::minor_breaks_linear;

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

    /// Select appropriate interval based on span in days and desired break count
    fn select(span_days: f64, n: usize) -> Self {
        let target_interval = span_days / n as f64;

        // Choose interval that gives roughly the right number of breaks
        if target_interval >= 365.0 {
            DateInterval::Year
        } else if target_interval >= 90.0 {
            DateInterval::Quarter
        } else if target_interval >= 28.0 {
            DateInterval::Month
        } else if target_interval >= 5.0 {
            DateInterval::Week
        } else {
            DateInterval::Day
        }
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
        // Roughly ~4000 BC to ~4000 AD in days since epoch
        (-2_000_000.0, 2_000_000.0)
    }

    fn is_value_in_domain(&self, value: f64) -> bool {
        let (min, max) = self.allowed_domain();
        value.is_finite() && value >= min && value <= max
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
        let interval = DateInterval::select(span, n);

        if pretty {
            calculate_pretty_date_breaks(min, max, n, interval)
        } else {
            // For non-pretty, fall back to linear breaks in day-space
            calculate_linear_date_breaks(min, max, n)
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
}

/// Calculate pretty date breaks aligned to interval boundaries
fn calculate_pretty_date_breaks(min: f64, max: f64, n: usize, interval: DateInterval) -> Vec<f64> {
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

            // Calculate year step to get roughly n breaks
            let year_span = (end_year - start_year + 1) as usize;
            let step = (year_span / n).max(1);

            // Align to nice year boundaries (1, 2, 5, 10, etc.)
            let step = nice_step(step as f64) as i32;

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

            let mut year = start_year;
            let mut quarter = start_quarter;

            while year < end_year || (year == end_year && quarter <= end_quarter) {
                let month = (quarter * 3 + 1) as u32;
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, 1) {
                    let days = (date - unix_epoch).num_days() as f64;
                    if days >= min && days <= max {
                        breaks.push(days);
                    }
                }
                quarter += 1;
                if quarter > 3 {
                    quarter = 0;
                    year += 1;
                }
            }
        }
        DateInterval::Month => {
            // Start at the beginning of the month containing min_date
            let start_year = min_date.year();
            let start_month = min_date.month();

            let end_year = max_date.year();
            let end_month = max_date.month();

            // Calculate total months and step
            let total_months =
                (end_year - start_year) * 12 + (end_month as i32 - start_month as i32 + 1);
            let step = ((total_months as usize) / n).max(1);

            // Align step to nice values (1, 2, 3, 6, 12)
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

            let mut day = first_monday;
            while day <= end_days {
                let days = day as f64;
                if days >= min && days <= max {
                    breaks.push(days);
                }
                day += 7;
            }
        }
        DateInterval::Day => {
            // Calculate step size for days
            let span = max - min;
            let step = (span / n as f64).max(1.0);
            let step = nice_step(step) as i64;

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

/// Calculate linear breaks in day-space
fn calculate_linear_date_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![min];
    }

    let step = (max - min) / (n - 1) as f64;
    (0..n).map(|i| min + i as f64 * step).collect()
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
        assert!(min < 0.0);
        assert!(max > 0.0);
        assert!(t.is_value_in_domain(0.0));
        assert!(t.is_value_in_domain(-1000.0));
        assert!(t.is_value_in_domain(1000.0));
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
        assert_eq!(breaks.len(), 5);
        assert_eq!(breaks[0], 0.0);
        assert_eq!(breaks[4], 100.0);
    }

    #[test]
    fn test_date_interval_selection() {
        // Large span -> year
        assert_eq!(DateInterval::select(3650.0, 5), DateInterval::Year);

        // Medium span -> month
        assert_eq!(DateInterval::select(180.0, 6), DateInterval::Month);

        // Small span -> week
        assert_eq!(DateInterval::select(28.0, 4), DateInterval::Week);

        // Very small span -> day
        assert_eq!(DateInterval::select(7.0, 7), DateInterval::Day);
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
