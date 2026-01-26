//! Break calculation algorithms for scales
//!
//! Provides functions to calculate axis/legend break positions.

use crate::plot::ArrayElement;

/// Default number of breaks
pub const DEFAULT_BREAK_COUNT: usize = 5;

/// Calculate pretty breaks using "nice numbers" algorithm.
/// Based on Wilkinson's algorithm (similar to R's pretty()).
pub fn pretty_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 || min >= max {
        return vec![];
    }

    let range = max - min;
    let rough_step = range / (n as f64);

    // Find a "nice" step size (1, 2, 5, 10, 20, 25, 50, etc.)
    let magnitude = 10f64.powf(rough_step.log10().floor());
    let residual = rough_step / magnitude;

    let nice_step = if residual <= 1.0 {
        1.0 * magnitude
    } else if residual <= 2.0 {
        2.0 * magnitude
    } else if residual <= 5.0 {
        5.0 * magnitude
    } else {
        10.0 * magnitude
    };

    // Calculate nice min/max
    let nice_min = (min / nice_step).floor() * nice_step;
    let nice_max = (max / nice_step).ceil() * nice_step;

    // Generate breaks
    let mut breaks = vec![];
    let mut value = nice_min;
    while value <= nice_max + nice_step * 0.5 {
        breaks.push(value);
        value += nice_step;
    }
    breaks
}

/// Calculate simple linear breaks (evenly spaced).
pub fn linear_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![(min + max) / 2.0];
    }

    let step = (max - min) / (n - 1) as f64;
    (0..n).map(|i| min + step * i as f64).collect()
}

/// Filter breaks to only those within the given range.
pub fn filter_breaks_to_range(
    breaks: &[ArrayElement],
    range: &[ArrayElement],
) -> Vec<ArrayElement> {
    let (min, max) = match (range.first(), range.last()) {
        (Some(ArrayElement::Number(min)), Some(ArrayElement::Number(max))) => (*min, *max),
        _ => return breaks.to_vec(), // Can't filter non-numeric
    };

    breaks
        .iter()
        .filter(|b| {
            if let ArrayElement::Number(v) = b {
                *v >= min && *v <= max
            } else {
                true // Keep non-numeric breaks
            }
        })
        .cloned()
        .collect()
}

// =============================================================================
// Transform-Aware Break Calculations
// =============================================================================

/// Main entry point for transform-aware break calculation.
///
/// Dispatches to the appropriate break calculation function based on the
/// transform type. For identity transforms (None), uses Wilkinson's algorithm
/// for pretty breaks or simple linear spacing.
pub fn calculate_breaks(
    min: f64,
    max: f64,
    n: usize,
    transform: Option<&str>,
    pretty: bool,
) -> Vec<f64> {
    match transform {
        Some("log10") => log_breaks(min, max, n, 10.0, pretty),
        Some("log2") => log_breaks(min, max, n, 2.0, pretty),
        Some("log") => log_breaks(min, max, n, std::f64::consts::E, pretty),
        Some("sqrt") => sqrt_breaks(min, max, n, pretty),
        Some("asinh") | Some("pseudo_log") => symlog_breaks(min, max, n, pretty),
        _ => {
            if pretty {
                pretty_breaks(min, max, n)
            } else {
                linear_breaks(min, max, n)
            }
        }
    }
}

/// Calculate breaks for log scales.
///
/// For `pretty=true`: Uses 1-2-5 pattern across decades (e.g., 1, 2, 5, 10, 20, 50, 100).
/// For `pretty=false`: Returns only powers of the base (e.g., 1, 10, 100, 1000 for base 10).
///
/// Non-positive values are filtered out since log is undefined for them.
pub fn log_breaks(min: f64, max: f64, n: usize, base: f64, pretty: bool) -> Vec<f64> {
    // Filter to positive values only
    let pos_min = if min <= 0.0 { f64::MIN_POSITIVE } else { min };
    let pos_max = if max <= 0.0 {
        return vec![];
    } else {
        max
    };

    if pos_min >= pos_max || n == 0 {
        return vec![];
    }

    let min_exp = pos_min.log(base).floor() as i32;
    let max_exp = pos_max.log(base).ceil() as i32;

    if pretty {
        log_breaks_extended(pos_min, pos_max, base, min_exp, max_exp, n)
    } else {
        // Simple: just powers of base
        (min_exp..=max_exp)
            .map(|e| base.powi(e))
            .filter(|&v| v >= pos_min && v <= pos_max)
            .collect()
    }
}

/// Extended log breaks using 1-2-5 pattern.
///
/// Generates breaks at each power of the base, multiplied by 1, 2, and 5,
/// then thins the result to approximately n values.
fn log_breaks_extended(
    min: f64,
    max: f64,
    base: f64,
    min_exp: i32,
    max_exp: i32,
    n: usize,
) -> Vec<f64> {
    let multipliers = [1.0, 2.0, 5.0];

    let mut breaks = Vec::new();
    for exp in min_exp..=max_exp {
        let power = base.powi(exp);
        for &mult in &multipliers {
            let value = power * mult;
            if value >= min && value <= max {
                breaks.push(value);
            }
        }
    }

    // Sort to ensure proper order (multipliers can cause interleaving)
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    breaks.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(b.abs()));

    thin_breaks(breaks, n)
}

/// Calculate breaks for sqrt scales.
///
/// Calculates breaks in sqrt-transformed space, then squares them back.
/// Non-negative values only (sqrt is undefined for negative numbers).
pub fn sqrt_breaks(min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
    let pos_min = min.max(0.0);
    if pos_min >= max || n == 0 {
        return vec![];
    }

    let sqrt_min = pos_min.sqrt();
    let sqrt_max = max.sqrt();

    // Calculate breaks in sqrt space, then square
    let sqrt_space_breaks = if pretty {
        pretty_breaks(sqrt_min, sqrt_max, n)
    } else {
        linear_breaks(sqrt_min, sqrt_max, n)
    };

    sqrt_space_breaks
        .into_iter()
        .map(|v| v * v)
        .filter(|&v| v >= pos_min && v <= max)
        .collect()
}

/// Calculate breaks for symlog scales (handles zero and negatives).
///
/// Symmetric log scale that can handle the full range of values including
/// zero and negative numbers. Uses log breaks for positive and negative
/// portions separately, with zero included if in range.
pub fn symlog_breaks(min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }

    let mut breaks = Vec::new();

    // Handle negative portion
    if min < 0.0 {
        let neg_max = min.abs();
        let neg_min = if max < 0.0 { max.abs() } else { 1.0 };
        let neg_breaks = log_breaks(neg_min, neg_max, n / 2 + 1, 10.0, pretty);
        breaks.extend(neg_breaks.into_iter().map(|v| -v).rev());
    }

    // Include zero if in range
    if min <= 0.0 && max >= 0.0 {
        breaks.push(0.0);
    }

    // Handle positive portion
    if max > 0.0 {
        let pos_min = if min > 0.0 { min } else { 1.0 };
        breaks.extend(log_breaks(pos_min, max, n / 2 + 1, 10.0, pretty));
    }

    breaks
}

/// Thin a break vector to approximately n values.
///
/// Keeps the first and last values and selects evenly-spaced indices
/// from the middle to achieve the target count.
fn thin_breaks(breaks: Vec<f64>, n: usize) -> Vec<f64> {
    if breaks.len() <= n || n == 0 {
        return breaks;
    }

    if n == 1 {
        // Return middle value
        return vec![breaks[breaks.len() / 2]];
    }

    // Keep first and last, thin middle
    let step = (breaks.len() - 1) as f64 / (n - 1) as f64;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let idx = (i as f64 * step).round() as usize;
        result.push(breaks[idx.min(breaks.len() - 1)]);
    }
    result.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(b.abs()));
    result
}

/// Temporal interval unit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalUnit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

/// Temporal interval with optional count (e.g., "2 months", "3 days")
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalInterval {
    pub count: u32,
    pub unit: TemporalUnit,
}

impl TemporalInterval {
    /// Parse interval string like "month", "2 months", "3 days"
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        let parts: Vec<&str> = s.split_whitespace().collect();

        match parts.as_slice() {
            // Just unit: "month", "day"
            [unit] => {
                let unit = Self::parse_unit(unit)?;
                Some(Self { count: 1, unit })
            }
            // Count + unit: "2 months", "3 days"
            [count, unit] => {
                let count: u32 = count.parse().ok()?;
                let unit = Self::parse_unit(unit)?;
                Some(Self { count, unit })
            }
            _ => None,
        }
    }

    fn parse_unit(s: &str) -> Option<TemporalUnit> {
        match s {
            "second" | "seconds" => Some(TemporalUnit::Second),
            "minute" | "minutes" => Some(TemporalUnit::Minute),
            "hour" | "hours" => Some(TemporalUnit::Hour),
            "day" | "days" => Some(TemporalUnit::Day),
            "week" | "weeks" => Some(TemporalUnit::Week),
            "month" | "months" => Some(TemporalUnit::Month),
            "year" | "years" => Some(TemporalUnit::Year),
            _ => None,
        }
    }
}

/// Calculate temporal breaks at interval boundaries for Date scale.
/// min/max are days since epoch for Date.
pub fn temporal_breaks_date(
    min_days: i32,
    max_days: i32,
    interval: TemporalInterval,
) -> Vec<String> {
    use chrono::NaiveDate;

    let epoch = match NaiveDate::from_ymd_opt(1970, 1, 1) {
        Some(d) => d,
        None => return vec![],
    };
    let min_date = epoch + chrono::Duration::days(min_days as i64);
    let max_date = epoch + chrono::Duration::days(max_days as i64);

    let mut breaks = vec![];
    let mut current = align_date_to_interval(min_date, &interval);

    while current <= max_date {
        breaks.push(current.format("%Y-%m-%d").to_string());
        current = advance_date_by_interval(current, &interval);
    }
    breaks
}

fn align_date_to_interval(
    date: chrono::NaiveDate,
    interval: &TemporalInterval,
) -> chrono::NaiveDate {
    use chrono::{Datelike, NaiveDate};

    match interval.unit {
        TemporalUnit::Day => date,
        TemporalUnit::Week => {
            // Align to Monday
            let days_from_monday = date.weekday().num_days_from_monday();
            date - chrono::Duration::days(days_from_monday as i64)
        }
        TemporalUnit::Month => {
            NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap_or(date)
        }
        TemporalUnit::Year => NaiveDate::from_ymd_opt(date.year(), 1, 1).unwrap_or(date),
        _ => date, // Second/minute/hour not applicable to Date
    }
}

fn advance_date_by_interval(
    date: chrono::NaiveDate,
    interval: &TemporalInterval,
) -> chrono::NaiveDate {
    use chrono::{Datelike, NaiveDate};

    let count = interval.count as i64;
    match interval.unit {
        TemporalUnit::Day => date + chrono::Duration::days(count),
        TemporalUnit::Week => date + chrono::Duration::weeks(count),
        TemporalUnit::Month => {
            // Add N months
            let total_months = date.year() * 12 + date.month() as i32 - 1 + count as i32;
            let year = total_months / 12;
            let month = (total_months % 12) as u32 + 1;
            NaiveDate::from_ymd_opt(year, month, 1).unwrap_or(date)
        }
        TemporalUnit::Year => {
            NaiveDate::from_ymd_opt(date.year() + count as i32, 1, 1).unwrap_or(date)
        }
        _ => date + chrono::Duration::days(count),
    }
}

/// Calculate temporal breaks at interval boundaries for DateTime scale.
/// min/max are microseconds since epoch.
pub fn temporal_breaks_datetime(
    min_us: i64,
    max_us: i64,
    interval: TemporalInterval,
) -> Vec<String> {
    use chrono::{DateTime, Utc};

    let to_datetime = |us: i64| -> Option<DateTime<Utc>> {
        let secs = us / 1_000_000;
        let nsecs = ((us % 1_000_000).abs() * 1000) as u32;
        DateTime::<Utc>::from_timestamp(secs, nsecs)
    };

    let min_dt = match to_datetime(min_us) {
        Some(dt) => dt,
        None => return vec![],
    };
    let max_dt = match to_datetime(max_us) {
        Some(dt) => dt,
        None => return vec![],
    };

    let mut breaks = vec![];
    let mut current = align_datetime_to_interval(min_dt, &interval);

    while current <= max_dt {
        breaks.push(current.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string());
        current = advance_datetime_by_interval(current, &interval);
    }
    breaks
}

fn align_datetime_to_interval(
    dt: chrono::DateTime<chrono::Utc>,
    interval: &TemporalInterval,
) -> chrono::DateTime<chrono::Utc> {
    use chrono::{Datelike, TimeZone, Timelike, Utc};

    match interval.unit {
        TemporalUnit::Second => Utc
            .with_ymd_and_hms(
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                dt.second(),
            )
            .single()
            .unwrap_or(dt),
        TemporalUnit::Minute => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), 0)
            .single()
            .unwrap_or(dt),
        TemporalUnit::Hour => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), 0, 0)
            .single()
            .unwrap_or(dt),
        TemporalUnit::Day => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
            .single()
            .unwrap_or(dt),
        TemporalUnit::Week => {
            let days_from_monday = dt.weekday().num_days_from_monday();
            let aligned = dt - chrono::Duration::days(days_from_monday as i64);
            Utc.with_ymd_and_hms(aligned.year(), aligned.month(), aligned.day(), 0, 0, 0)
                .single()
                .unwrap_or(dt)
        }
        TemporalUnit::Month => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
            .single()
            .unwrap_or(dt),
        TemporalUnit::Year => Utc
            .with_ymd_and_hms(dt.year(), 1, 1, 0, 0, 0)
            .single()
            .unwrap_or(dt),
    }
}

fn advance_datetime_by_interval(
    dt: chrono::DateTime<chrono::Utc>,
    interval: &TemporalInterval,
) -> chrono::DateTime<chrono::Utc> {
    use chrono::{Datelike, TimeZone, Timelike, Utc};

    let count = interval.count as i64;
    match interval.unit {
        TemporalUnit::Second => dt + chrono::Duration::seconds(count),
        TemporalUnit::Minute => dt + chrono::Duration::minutes(count),
        TemporalUnit::Hour => dt + chrono::Duration::hours(count),
        TemporalUnit::Day => dt + chrono::Duration::days(count),
        TemporalUnit::Week => dt + chrono::Duration::weeks(count),
        TemporalUnit::Month => {
            let total_months = dt.year() * 12 + dt.month() as i32 - 1 + count as i32;
            let year = total_months / 12;
            let month = (total_months % 12) as u32 + 1;
            Utc.with_ymd_and_hms(
                year,
                month,
                dt.day().min(28),
                dt.hour(),
                dt.minute(),
                dt.second(),
            )
            .single()
            .unwrap_or(dt)
        }
        TemporalUnit::Year => Utc
            .with_ymd_and_hms(
                dt.year() + count as i32,
                dt.month(),
                dt.day().min(28),
                dt.hour(),
                dt.minute(),
                dt.second(),
            )
            .single()
            .unwrap_or(dt),
    }
}

/// Calculate temporal breaks at interval boundaries for Time scale.
/// min/max are nanoseconds since midnight.
pub fn temporal_breaks_time(min_ns: i64, max_ns: i64, interval: TemporalInterval) -> Vec<String> {
    use chrono::NaiveTime;

    let to_time = |ns: i64| -> Option<NaiveTime> {
        let total_secs = ns / 1_000_000_000;
        let nanos = (ns % 1_000_000_000).unsigned_abs() as u32;
        let hours = (total_secs / 3600) as u32;
        let mins = ((total_secs % 3600) / 60) as u32;
        let secs = (total_secs % 60) as u32;
        NaiveTime::from_hms_nano_opt(hours.min(23), mins, secs, nanos)
    };

    let min_time = match to_time(min_ns) {
        Some(t) => t,
        None => return vec![],
    };
    let max_time = match to_time(max_ns) {
        Some(t) => t,
        None => return vec![],
    };

    let mut breaks = vec![];
    let mut current = align_time_to_interval(min_time, &interval);

    while current <= max_time {
        breaks.push(current.format("%H:%M:%S%.3f").to_string());
        current = match advance_time_by_interval(current, &interval) {
            Some(t) if t > current => t,
            _ => break, // Overflow past 24 hours
        };
    }
    breaks
}

fn align_time_to_interval(
    time: chrono::NaiveTime,
    interval: &TemporalInterval,
) -> chrono::NaiveTime {
    use chrono::{NaiveTime, Timelike};

    match interval.unit {
        TemporalUnit::Second => {
            NaiveTime::from_hms_opt(time.hour(), time.minute(), time.second()).unwrap_or(time)
        }
        TemporalUnit::Minute => {
            NaiveTime::from_hms_opt(time.hour(), time.minute(), 0).unwrap_or(time)
        }
        TemporalUnit::Hour => NaiveTime::from_hms_opt(time.hour(), 0, 0).unwrap_or(time),
        _ => time, // Day/week/month/year not applicable to Time
    }
}

fn advance_time_by_interval(
    time: chrono::NaiveTime,
    interval: &TemporalInterval,
) -> Option<chrono::NaiveTime> {
    use chrono::Timelike;

    let count = interval.count as u32;
    match interval.unit {
        TemporalUnit::Second => time.with_second((time.second() + count) % 60),
        TemporalUnit::Minute => time.with_minute((time.minute() + count) % 60),
        TemporalUnit::Hour => time.with_hour((time.hour() + count) % 24),
        _ => Some(time), // Day/week/month/year not applicable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Pretty Breaks Tests
    // =========================================================================

    #[test]
    fn test_pretty_breaks_basic() {
        let breaks = pretty_breaks(0.0, 100.0, 5);
        // Should produce nice round numbers
        assert!(breaks.len() >= 3);
        assert!(breaks[0] <= 0.0);
        assert!(*breaks.last().unwrap() >= 100.0);
    }

    #[test]
    fn test_pretty_breaks_small_range() {
        let breaks = pretty_breaks(0.1, 0.9, 5);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_pretty_breaks_large_range() {
        let breaks = pretty_breaks(0.0, 10000.0, 5);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_pretty_breaks_zero_count() {
        let breaks = pretty_breaks(0.0, 100.0, 0);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_pretty_breaks_equal_min_max() {
        let breaks = pretty_breaks(50.0, 50.0, 5);
        assert!(breaks.is_empty());
    }

    // =========================================================================
    // Linear Breaks Tests
    // =========================================================================

    #[test]
    fn test_linear_breaks_basic() {
        let breaks = linear_breaks(0.0, 100.0, 5);
        assert_eq!(breaks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }

    #[test]
    fn test_linear_breaks_single() {
        let breaks = linear_breaks(0.0, 100.0, 1);
        assert_eq!(breaks, vec![50.0]);
    }

    #[test]
    fn test_linear_breaks_two() {
        let breaks = linear_breaks(0.0, 100.0, 2);
        assert_eq!(breaks, vec![0.0, 100.0]);
    }

    #[test]
    fn test_linear_breaks_zero_count() {
        let breaks = linear_breaks(0.0, 100.0, 0);
        assert!(breaks.is_empty());
    }

    // =========================================================================
    // Filter Breaks Tests
    // =========================================================================

    #[test]
    fn test_filter_breaks_to_range() {
        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
            ArrayElement::Number(100.0),
        ];

        let range = vec![ArrayElement::Number(0.5), ArrayElement::Number(99.5)];
        let filtered = filter_breaks_to_range(&breaks, &range);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], ArrayElement::Number(25.0));
        assert_eq!(filtered[1], ArrayElement::Number(50.0));
        assert_eq!(filtered[2], ArrayElement::Number(75.0));
    }

    #[test]
    fn test_filter_breaks_all_inside() {
        let breaks = vec![
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
            ArrayElement::Number(75.0),
        ];

        let range = vec![ArrayElement::Number(0.0), ArrayElement::Number(100.0)];
        let filtered = filter_breaks_to_range(&breaks, &range);

        assert_eq!(filtered.len(), 3);
    }

    // =========================================================================
    // Log Break Tests
    // =========================================================================

    #[test]
    fn test_log10_breaks_powers_only() {
        // pretty=false should give powers of 10
        let breaks = log_breaks(1.0, 10000.0, 10, 10.0, false);
        assert_eq!(breaks, vec![1.0, 10.0, 100.0, 1000.0, 10000.0]);
    }

    #[test]
    fn test_log10_breaks_extended() {
        // pretty=true should give 1-2-5 pattern
        let breaks = log_breaks(1.0, 100.0, 10, 10.0, true);
        // Should contain: 1, 2, 5, 10, 20, 50, 100
        assert!(breaks.contains(&1.0));
        assert!(breaks.contains(&2.0));
        assert!(breaks.contains(&5.0));
        assert!(breaks.contains(&10.0));
        assert!(breaks.contains(&100.0));
    }

    #[test]
    fn test_log10_breaks_extended_full_pattern() {
        // Test the complete 1-2-5 pattern across decades
        let breaks = log_breaks(1.0, 1000.0, 20, 10.0, true);
        // Should have breaks at 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
        assert!(breaks.contains(&1.0));
        assert!(breaks.contains(&2.0));
        assert!(breaks.contains(&5.0));
        assert!(breaks.contains(&10.0));
        assert!(breaks.contains(&20.0));
        assert!(breaks.contains(&50.0));
        assert!(breaks.contains(&100.0));
    }

    #[test]
    fn test_log2_breaks() {
        let breaks = log_breaks(1.0, 16.0, 10, 2.0, false);
        assert_eq!(breaks, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    }

    #[test]
    fn test_log_breaks_filters_negative() {
        // Range includes negative - should only return positive breaks
        let breaks = log_breaks(-10.0, 1000.0, 10, 10.0, false);
        assert!(breaks.iter().all(|&v| v > 0.0));
        assert!(breaks.contains(&1.0));
        assert!(breaks.contains(&10.0));
        assert!(breaks.contains(&100.0));
        assert!(breaks.contains(&1000.0));
    }

    #[test]
    fn test_log_breaks_all_negative_returns_empty() {
        let breaks = log_breaks(-100.0, -1.0, 5, 10.0, true);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_log_breaks_zero_count() {
        let breaks = log_breaks(1.0, 100.0, 0, 10.0, true);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_log_breaks_fractional_range() {
        // Test range from 0.01 to 100
        let breaks = log_breaks(0.01, 100.0, 10, 10.0, false);
        assert!(breaks.contains(&0.01));
        assert!(breaks.contains(&0.1));
        assert!(breaks.contains(&1.0));
        assert!(breaks.contains(&10.0));
        assert!(breaks.contains(&100.0));
    }

    // =========================================================================
    // Sqrt Break Tests
    // =========================================================================

    #[test]
    fn test_sqrt_breaks_basic() {
        let breaks = sqrt_breaks(0.0, 100.0, 5, false);
        // Linear in sqrt space: sqrt(100)=10, steps of 2.5
        // Squared back: 0, 6.25, 25, 56.25, 100
        assert_eq!(breaks.len(), 5);
        assert!((breaks[0] - 0.0).abs() < 0.01);
        assert!((breaks[4] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_sqrt_breaks_filters_negative() {
        let breaks = sqrt_breaks(-10.0, 100.0, 5, true);
        assert!(breaks.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_sqrt_breaks_zero_count() {
        let breaks = sqrt_breaks(0.0, 100.0, 0, true);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_sqrt_breaks_pretty() {
        let breaks = sqrt_breaks(0.0, 100.0, 5, true);
        // Should use Wilkinson in sqrt space
        assert!(!breaks.is_empty());
        // First break should be <= 0, last should be >= 100
        // (Wilkinson expands range to nice numbers)
    }

    // =========================================================================
    // Symlog Break Tests
    // =========================================================================

    #[test]
    fn test_symlog_breaks_symmetric() {
        let breaks = symlog_breaks(-1000.0, 1000.0, 10, false);
        // Should have negative powers, zero, and positive powers
        assert!(breaks.contains(&0.0));
        assert!(breaks.iter().any(|&v| v < 0.0));
        assert!(breaks.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_symlog_breaks_positive_only() {
        let breaks = symlog_breaks(1.0, 1000.0, 5, false);
        // Should behave like regular log for positive-only range
        assert!(breaks.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_symlog_breaks_negative_only() {
        let breaks = symlog_breaks(-1000.0, -1.0, 5, false);
        // Should have only negative values
        assert!(breaks.iter().all(|&v| v < 0.0));
    }

    #[test]
    fn test_symlog_breaks_includes_zero() {
        let breaks = symlog_breaks(-100.0, 100.0, 7, false);
        assert!(breaks.contains(&0.0));
    }

    #[test]
    fn test_symlog_breaks_zero_count() {
        let breaks = symlog_breaks(-100.0, 100.0, 0, true);
        assert!(breaks.is_empty());
    }

    // =========================================================================
    // Calculate Breaks Dispatch Tests
    // =========================================================================

    #[test]
    fn test_calculate_breaks_dispatches_log10() {
        let breaks = calculate_breaks(1.0, 1000.0, 5, Some("log10"), false);
        assert!(breaks.contains(&10.0));
        assert!(breaks.contains(&100.0));
    }

    #[test]
    fn test_calculate_breaks_dispatches_log2() {
        let breaks = calculate_breaks(1.0, 16.0, 10, Some("log2"), false);
        assert!(breaks.contains(&2.0));
        assert!(breaks.contains(&4.0));
        assert!(breaks.contains(&8.0));
    }

    #[test]
    fn test_calculate_breaks_dispatches_sqrt() {
        let breaks = calculate_breaks(0.0, 100.0, 5, Some("sqrt"), false);
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_calculate_breaks_dispatches_asinh() {
        let breaks = calculate_breaks(-100.0, 100.0, 7, Some("asinh"), false);
        assert!(breaks.contains(&0.0));
    }

    #[test]
    fn test_calculate_breaks_dispatches_pseudo_log() {
        let breaks = calculate_breaks(-100.0, 100.0, 7, Some("pseudo_log"), false);
        assert!(breaks.contains(&0.0));
    }

    #[test]
    fn test_calculate_breaks_dispatches_identity_pretty() {
        let breaks = calculate_breaks(0.0, 100.0, 5, None, true);
        // Should use pretty_breaks
        assert!(!breaks.is_empty());
        // Pretty breaks produces nice numbers
    }

    #[test]
    fn test_calculate_breaks_dispatches_identity_linear() {
        let breaks = calculate_breaks(0.0, 100.0, 5, None, false);
        // Should use linear_breaks
        assert_eq!(breaks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }

    #[test]
    fn test_calculate_breaks_unknown_transform() {
        // Unknown transform should fall through to identity
        let breaks = calculate_breaks(0.0, 100.0, 5, Some("unknown"), true);
        assert!(!breaks.is_empty());
    }

    // =========================================================================
    // Thin Breaks Tests
    // =========================================================================

    #[test]
    fn test_thin_breaks_no_thinning_needed() {
        let breaks = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = thin_breaks(breaks.clone(), 10);
        assert_eq!(result, breaks);
    }

    #[test]
    fn test_thin_breaks_to_smaller() {
        let breaks = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = thin_breaks(breaks, 5);
        assert_eq!(result.len(), 5);
        // Should keep first and last
        assert_eq!(result[0], 1.0);
        assert_eq!(result[4], 10.0);
    }

    #[test]
    fn test_thin_breaks_single() {
        let breaks = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = thin_breaks(breaks, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 3.0); // Middle value
    }

    // =========================================================================
    // Temporal Interval Tests
    // =========================================================================

    #[test]
    fn test_temporal_interval_from_str_simple() {
        let interval = TemporalInterval::from_str("month").unwrap();
        assert_eq!(interval.count, 1);
        assert_eq!(interval.unit, TemporalUnit::Month);
    }

    #[test]
    fn test_temporal_interval_from_str_with_count() {
        let interval = TemporalInterval::from_str("2 months").unwrap();
        assert_eq!(interval.count, 2);
        assert_eq!(interval.unit, TemporalUnit::Month);

        let interval = TemporalInterval::from_str("3 days").unwrap();
        assert_eq!(interval.count, 3);
        assert_eq!(interval.unit, TemporalUnit::Day);
    }

    #[test]
    fn test_temporal_interval_all_units() {
        assert!(TemporalInterval::from_str("second").is_some());
        assert!(TemporalInterval::from_str("seconds").is_some());
        assert!(TemporalInterval::from_str("minute").is_some());
        assert!(TemporalInterval::from_str("hour").is_some());
        assert!(TemporalInterval::from_str("day").is_some());
        assert!(TemporalInterval::from_str("week").is_some());
        assert!(TemporalInterval::from_str("month").is_some());
        assert!(TemporalInterval::from_str("year").is_some());
    }

    #[test]
    fn test_temporal_interval_invalid() {
        assert!(TemporalInterval::from_str("invalid").is_none());
        assert!(TemporalInterval::from_str("foo bar baz").is_none());
        assert!(TemporalInterval::from_str("").is_none());
    }

    // =========================================================================
    // Temporal Date Breaks Tests
    // =========================================================================

    #[test]
    fn test_temporal_breaks_monthly() {
        // 2024-01-15 = day 19738, 2024-04-15 = day 19828
        let interval = TemporalInterval::from_str("month").unwrap();
        let breaks = temporal_breaks_date(19738, 19828, interval);
        assert!(!breaks.is_empty());
        assert_eq!(breaks[0], "2024-01-01");
        assert!(breaks.contains(&"2024-02-01".to_string()));
        assert!(breaks.contains(&"2024-03-01".to_string()));
        assert!(breaks.contains(&"2024-04-01".to_string()));
    }

    #[test]
    fn test_temporal_breaks_bimonthly() {
        // Test "2 months" interval
        // 2024-01-01 = day 19724, 2024-07-01 = day 19907
        let interval = TemporalInterval::from_str("2 months").unwrap();
        let breaks = temporal_breaks_date(19724, 19907, interval);
        assert_eq!(breaks[0], "2024-01-01");
        assert!(breaks.contains(&"2024-03-01".to_string()));
        assert!(breaks.contains(&"2024-05-01".to_string()));
        // Should NOT contain Feb, Apr, Jun
        assert!(!breaks.contains(&"2024-02-01".to_string()));
    }

    #[test]
    fn test_temporal_breaks_yearly() {
        // Test yearly breaks spanning multiple years
        // 2022-01-01 = day 18993, 2024-12-31 = day 20089
        let interval = TemporalInterval::from_str("year").unwrap();
        let breaks = temporal_breaks_date(18993, 20089, interval);
        assert!(breaks.contains(&"2022-01-01".to_string()));
        assert!(breaks.contains(&"2023-01-01".to_string()));
        assert!(breaks.contains(&"2024-01-01".to_string()));
    }

    #[test]
    fn test_temporal_breaks_weekly() {
        // Test weekly breaks
        // About 30 days
        let interval = TemporalInterval::from_str("week").unwrap();
        let breaks = temporal_breaks_date(19724, 19754, interval);
        assert!(!breaks.is_empty());
        // Should be multiple weeks
        assert!(breaks.len() >= 4);
    }
}
