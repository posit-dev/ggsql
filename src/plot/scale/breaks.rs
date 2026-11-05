//! Break calculation for scales.
//!
//! Tick selection itself lives in zygos — Wilkinson Extended
//! ([`extended_breaks`]), [`linear_breaks`], the log / sqrt / symlog
//! families and the calendar-aware temporal breaks — and transforms call it
//! directly. This module re-exports those and keeps only what zygos has no
//! counterpart for: integer-constrained breaks, graticules for map
//! projections, minor breaks that continue past the outermost majors,
//! filtering typed breaks to a domain, and parsing ggsql's interval strings
//! (`'2 months'`) into a zygos [`TemporalInterval`].

pub use zygos::breaks::{
    extended_breaks, linear_breaks, linear_minor_breaks_between, log_minor_breaks,
    log_pretty_breaks, sqrt_breaks, symlog_breaks, symlog_minor_breaks, CalendarUnit,
    TemporalInterval, DEFAULT_BREAK_COUNT,
};
use zygos::TemporalUnit;

use crate::plot::ArrayElement;

/// Calculate breaks for integer scales with even spacing.
///
/// Unlike simply rounding the output of [`extended_breaks`], this function
/// ensures that breaks are evenly spaced integers. For small ranges where
/// the natural step would be < 1, it uses step = 1 and generates consecutive
/// integers.
///
/// # Arguments
/// - `min`: Minimum data value
/// - `max`: Maximum data value
/// - `n`: Target number of breaks
/// - `pretty`: If true, use "nice" integer step sizes (1, 2, 5, 10, 20, ...).
///   If false, use exact linear spacing rounded to integers.
pub fn integer_breaks(min: f64, max: f64, n: usize, pretty: bool) -> Vec<f64> {
    if n == 0 || min >= max || !min.is_finite() || !max.is_finite() {
        return vec![];
    }

    let range = max - min;
    let int_min = min.floor() as i64;
    let int_max = max.ceil() as i64;
    let int_range = int_max - int_min;

    // For very small ranges, just return consecutive integers
    if int_range <= n as i64 {
        return (int_min..=int_max).map(|i| i as f64).collect();
    }

    if pretty {
        // Use "nice" integer step sizes: 1, 2, 5, 10, 20, 25, 50, 100, ...
        let rough_step = range / (n as f64);

        // Find nice integer step (must be >= 1)
        let nice_step = if rough_step < 1.0 {
            1
        } else {
            let magnitude = 10f64.powf(rough_step.log10().floor()) as i64;
            let residual = rough_step / magnitude as f64;

            let multiplier = if residual <= 1.0 {
                1
            } else if residual <= 2.0 {
                2
            } else if residual <= 5.0 {
                5
            } else {
                10
            };

            (magnitude * multiplier).max(1)
        };

        // Find starting point (nice_min <= min, aligned to step)
        let nice_min = (int_min / nice_step) * nice_step;

        // Generate breaks
        let mut breaks = vec![];
        let mut value = nice_min;
        while value <= int_max {
            breaks.push(value as f64);
            value += nice_step;
        }
        breaks
    } else {
        // Linear spacing with integer step (at least 1)
        // Extend one step before min and one step after max for binned scales
        let step = ((int_range as f64) / (n as f64 - 1.0)).ceil() as i64;
        let step = step.max(1);

        let mut breaks = vec![];
        // Start one step before int_min
        let mut value = int_min - step;
        // Generate until one step past int_max
        while value <= int_max + step {
            breaks.push(value as f64);
            value += step;
        }
        breaks
    }
}

/// Filter breaks to only those within the given range.
///
/// Both sides are compared through [`ArrayElement::to_f64`], which is the space a
/// scale's domain and breaks share once its transform has parsed them, so a
/// temporal break is constrained by a temporal domain just as a number is by a
/// numeric one. An element with no numeric form (a category, a boolean) is kept.
pub fn filter_breaks_to_range(
    breaks: &[ArrayElement],
    range: &[ArrayElement],
) -> Vec<ArrayElement> {
    let (min, max) = match (
        range.first().and_then(|e| e.to_f64()),
        range.last().and_then(|e| e.to_f64()),
    ) {
        (Some(min), Some(max)) => (min, max),
        _ => return breaks.to_vec(), // Can't filter against a non-numeric range
    };

    breaks
        .iter()
        .filter(|b| b.to_f64().is_none_or(|v| v >= min && v <= max))
        .cloned()
        .collect()
}

// =============================================================================
// Graticule breaks (for map projections)
// =============================================================================

/// Calculate graticule break positions for a geographic range.
///
/// Picks the smallest step from a predefined set that produces at most `n` lines,
/// then returns values within (min, max) at that interval. Includes boundary
/// values (-180/180 for longitude, -90/90 for latitude) when the range spans
/// the full extent.
pub fn graticule_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    // Graticules never span more than the globe. max/min also tame non-finite
    // inputs (e.g. infinite coordinates from degenerate projections), which
    // otherwise make the step loop below iterate ~i64::MAX times.
    let min = min.max(-180.0);
    let max = max.min(180.0);

    let range = max - min;
    if range <= 0.0 || n == 0 {
        return vec![];
    }

    const STEPS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0];

    let step = STEPS
        .iter()
        .copied()
        .find(|&s| range / s <= n as f64 + 1.0)
        .unwrap_or(90.0);

    let start = (min / step).ceil() as i64;
    let end = (max / step).floor() as i64;
    let mut breaks: Vec<f64> = (start..=end)
        .map(|i| i as f64 * step)
        .filter(|&v| v > min && v < max)
        .collect();

    // min/max are clamped to ±180 above, so these boundaries are equality checks
    if min == -180.0 && !breaks.contains(&-180.0) {
        breaks.insert(0, -180.0);
    } else if max == 180.0 && !breaks.contains(&180.0) {
        breaks.push(180.0);
    }
    if min <= -90.0 && !breaks.contains(&-90.0) {
        breaks.insert(0, -90.0);
    } else if max >= 90.0 && !breaks.contains(&90.0) {
        breaks.push(90.0);
    }

    breaks
}

// =============================================================================
// Minor Break Calculations
// =============================================================================

/// Calculate minor breaks by evenly dividing intervals (linear space)
///
/// Between each pair of major breaks, inserts n evenly-spaced minor breaks.
/// If range extends beyond major breaks, extrapolates minor breaks into those regions.
///
/// # Arguments
/// - `major_breaks`: The major break positions (must be sorted)
/// - `n`: Number of minor breaks per major interval
/// - `range`: Optional (min, max) scale input range to extend minor breaks beyond major breaks
///
/// # Returns
/// Minor break positions (excluding major breaks)
///
/// # Example
/// ```ignore
/// let majors = vec![20.0, 40.0, 60.0];
/// let minors = minor_breaks_linear(&majors, 1, Some((0.0, 80.0)));
/// // Returns [10, 30, 50, 70] - extends before 20 and after 60
/// ```
pub fn minor_breaks_linear(major_breaks: &[f64], n: usize, range: Option<(f64, f64)>) -> Vec<f64> {
    if major_breaks.len() < 2 || n == 0 {
        return vec![];
    }

    let mut minors = Vec::new();

    // Calculate interval between consecutive major breaks
    let interval = major_breaks[1] - major_breaks[0];
    if interval <= 0.0 {
        return vec![];
    }

    let step = interval / (n + 1) as f64;

    // Extrapolating past the outermost majors only makes sense when the majors
    // share one rhythm — then the minors outside continue the pattern of the
    // ones inside. With user-supplied uneven majors (`breaks => (37, 42, 55)`)
    // there is no such rhythm: `step` comes from the *first* interval only, so
    // extrapolated minors sit at a spacing matching no part of the axis and read
    // as stray ticks beyond the last label. Those axes get interior minors only.
    let evenly_spaced = major_breaks
        .windows(2)
        .all(|w| (w[1] - w[0] - interval).abs() <= interval.abs() * 1e-9);

    // If range extends before first major break, extrapolate backwards
    if let (true, Some((min, _))) = (evenly_spaced, range) {
        let first_major = major_breaks[0];
        let mut pos = first_major - step;
        while pos >= min {
            minors.push(pos);
            pos -= step;
        }
    }

    // Add minor breaks between each pair of major breaks
    minors.extend(linear_minor_breaks_between(major_breaks, n));

    // If range extends beyond last major break, extrapolate forwards
    if let (true, Some((_, max))) = (evenly_spaced, range) {
        let last_major = *major_breaks.last().unwrap();
        let mut pos = last_major + step;
        while pos <= max {
            minors.push(pos);
            pos += step;
        }
    }

    minors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    minors
}

/// Parse an interval string like `"month"`, `"2 months"` or `"3 days"`.
///
/// Returns `None` for anything that isn't an optional whole-number count followed
/// by a calendar unit (`second`, `minute`, `hour`, `day`, `week`, `month`, `year`,
/// singular or plural, case-insensitive).
pub fn parse_temporal_interval(s: &str) -> Option<TemporalInterval> {
    let s = s.trim().to_lowercase();
    let parts: Vec<&str> = s.split_whitespace().collect();

    match parts.as_slice() {
        // Just unit: "month", "day"
        [unit] => Some(TemporalInterval::new(1, parse_calendar_unit(unit)?)),
        // Count + unit: "2 months", "3 days"
        [count, unit] => {
            let count: u32 = count.parse().ok()?;
            Some(TemporalInterval::new(count, parse_calendar_unit(unit)?))
        }
        _ => None,
    }
}

fn parse_calendar_unit(s: &str) -> Option<CalendarUnit> {
    match s {
        "second" | "seconds" => Some(CalendarUnit::Second),
        "minute" | "minutes" => Some(CalendarUnit::Minute),
        "hour" | "hours" => Some(CalendarUnit::Hour),
        "day" | "days" => Some(CalendarUnit::Day),
        "week" | "weeks" => Some(CalendarUnit::Week),
        "month" | "months" => Some(CalendarUnit::Month),
        "year" | "years" => Some(CalendarUnit::Year),
        _ => None,
    }
}

/// Calendar breaks every `interval` from the interval's calendar grid at or
/// before `min` through `max`, plus one terminal break past `max` so the last
/// bin of a binned scale closes over the data. Positions are in `unit`'s raw
/// encoding: days for a date, microseconds for a datetime, nanoseconds since
/// midnight for a time (which stops at midnight rather than wrapping).
pub fn interval_breaks(
    interval: TemporalInterval,
    min: f64,
    max: f64,
    unit: TemporalUnit,
) -> Vec<f64> {
    use zygos::breaks::{temporal_breaks_date, temporal_breaks_datetime, temporal_breaks_time};
    match unit {
        TemporalUnit::Date => temporal_breaks_date(min as i32, max as i32, interval)
            .into_iter()
            .map(f64::from)
            .collect(),
        TemporalUnit::DateTime | TemporalUnit::Duration => {
            temporal_breaks_datetime(min as i64, max as i64, interval)
                .into_iter()
                .map(|v| v as f64)
                .collect()
        }
        TemporalUnit::Time => temporal_breaks_time(min as i64, max as i64, interval)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
    }
}

/// The calendar interval automatic temporal breaks step by over `[min, max]`
/// when about `n` breaks are asked for — the interval
/// [`zygos::breaks::temporal_breaks_from_f64`] picks, exposed so the minor
/// breaks can subdivide the same interval the majors use.
pub fn auto_temporal_interval(
    min: f64,
    max: f64,
    n: usize,
    unit: TemporalUnit,
) -> TemporalInterval {
    zygos::breaks::pick_temporal_interval(max - min, n.max(2) as f64, unit)
}

/// Minor breaks for a transform whose zygos minor algorithm works from the
/// scale's extent rather than by subdividing between majors — the log
/// family's whole-number multiples and the symlog branches built on them.
///
/// `minors_for` receives the extent: `range` when given, otherwise the span of
/// the majors. Those algorithms fix their own density, so `n` only decides
/// whether there are minors at all (`0` asks for none). Positions a major
/// already occupies are dropped.
pub fn extent_minor_breaks(
    major_breaks: &[f64],
    n: usize,
    range: Option<(f64, f64)>,
    minors_for: impl FnOnce(f64, f64) -> Vec<f64>,
) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    let (min, max) = match range {
        Some(extent) => extent,
        None => match (major_breaks.first(), major_breaks.last()) {
            (Some(&first), Some(&last)) => (first, last),
            _ => return vec![],
        },
    };
    let mut minors = minors_for(min, max);
    minors.retain(|minor| {
        !major_breaks
            .iter()
            .any(|major| (minor - major).abs() <= major.abs().max(minor.abs()) * 1e-9)
    });
    minors
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Integer Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_integer_breaks_variations() {
        // Test various ranges - all should produce integer, evenly-spaced breaks
        let test_cases = vec![
            (0.0, 100.0, 5, true),
            (0.0, 1_000_000.0, 5, true),
            (-50.0, 50.0, 5, true),
            (0.0, 5.0, 5, false),
        ];
        for (min, max, n, pretty) in test_cases {
            let breaks = integer_breaks(min, max, n, pretty);
            assert!(
                !breaks.is_empty(),
                "integer_breaks({}, {}, {}, {}) should not be empty",
                min,
                max,
                n,
                pretty
            );
            // All breaks should be integers
            for b in &breaks {
                assert_eq!(
                    *b,
                    b.round(),
                    "Break {} should be integer for ({}, {}, {}, {})",
                    b,
                    min,
                    max,
                    n,
                    pretty
                );
            }
            // All gaps should be equal (evenly spaced)
            if breaks.len() >= 2 {
                let step = breaks[1] - breaks[0];
                for i in 1..breaks.len() {
                    let gap = breaks[i] - breaks[i - 1];
                    assert!(
                        (gap - step).abs() < 0.01,
                        "Uneven spacing for ({}, {}, {}, {}): {:?}",
                        min,
                        max,
                        n,
                        pretty,
                        breaks
                    );
                }
            }
        }
    }

    #[test]
    fn test_integer_breaks_small_range() {
        // For range 0-5, should get consecutive integers
        let breaks = integer_breaks(0.0, 5.0, 10, true);
        assert_eq!(breaks, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_integer_breaks_edge_cases() {
        let edge_cases = vec![
            (0.0, 100.0, 0, "zero count"),
            (100.0, 0.0, 5, "min > max"),
            (50.0, 50.0, 5, "min == max"),
            (f64::NAN, 100.0, 5, "NaN min"),
            (0.0, f64::INFINITY, 5, "infinite max"),
        ];
        for (min, max, n, desc) in edge_cases {
            assert!(
                integer_breaks(min, max, n, true).is_empty(),
                "integer_breaks with {} should be empty",
                desc
            );
        }
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
    // Temporal Interval Tests (Consolidated)
    // =========================================================================

    /// Days since the Unix epoch for an ISO date.
    fn days(iso: &str) -> i32 {
        let date = chrono::NaiveDate::parse_from_str(iso, "%Y-%m-%d").unwrap();
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        (date - epoch).num_days() as i32
    }

    #[test]
    fn test_temporal_interval_parsing() {
        // Simple unit names
        let simple = parse_temporal_interval("month").unwrap();
        assert_eq!(simple.count, 1);
        assert_eq!(simple.unit, CalendarUnit::Month);

        // With count prefix
        let with_count = parse_temporal_interval("2 months").unwrap();
        assert_eq!(with_count.count, 2);
        assert_eq!(with_count.unit, CalendarUnit::Month);

        // All unit names should parse
        for unit in &[
            "second", "seconds", "minute", "hour", "day", "week", "month", "year",
        ] {
            assert!(
                parse_temporal_interval(unit).is_some(),
                "{} should parse",
                unit
            );
        }

        // Invalid inputs
        for invalid in &["invalid", "foo bar baz", ""] {
            assert!(
                parse_temporal_interval(invalid).is_none(),
                "{} should not parse",
                invalid
            );
        }
    }

    // =========================================================================
    // Minor Breaks Linear Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_minor_breaks_linear_variations() {
        // Basic case - one midpoint per interval
        let minors_basic = minor_breaks_linear(&[0.0, 10.0, 20.0], 1, None);
        assert_eq!(minors_basic, vec![5.0, 15.0]);

        // Multiple minor breaks per interval
        let minors_multi = minor_breaks_linear(&[0.0, 10.0, 20.0], 4, None);
        assert_eq!(minors_multi.len(), 8);
        for &v in &[2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 18.0] {
            assert!(minors_multi.contains(&v), "Should contain {}", v);
        }

        // With extension beyond majors
        let minors_ext = minor_breaks_linear(&[20.0, 40.0, 60.0], 1, Some((0.0, 80.0)));
        for &v in &[10.0, 30.0, 50.0, 70.0] {
            assert!(minors_ext.contains(&v), "Extended should contain {}", v);
        }
    }

    #[test]
    fn test_minor_breaks_linear_edge_cases() {
        assert!(
            minor_breaks_linear(&[10.0], 1, None).is_empty(),
            "Single major should return empty"
        );
        assert!(
            minor_breaks_linear(&[0.0, 10.0, 20.0], 0, None).is_empty(),
            "Zero count should return empty"
        );
    }

    // =========================================================================
    // Extent-driven Minor Breaks Tests
    // =========================================================================

    #[test]
    fn test_extent_minor_breaks_drops_majors_and_honours_zero() {
        let majors = [1.0, 10.0, 100.0];
        let log10 = |min: f64, max: f64| log_minor_breaks(min, max, 10.0);
        let minors = extent_minor_breaks(&majors, 8, None, log10);
        assert_eq!(minors.len(), 16);
        assert!(minors.iter().all(|m| !majors.contains(m)));
        assert!(extent_minor_breaks(&majors, 0, None, log10).is_empty());

        // 1-2-5 majors: the 2 and 5 multiples belong to the majors.
        let minors = extent_minor_breaks(&[1.0, 2.0, 5.0, 10.0], 8, None, log10);
        assert_eq!(minors, vec![3.0, 4.0, 6.0, 7.0, 8.0, 9.0]);
    }

    // =========================================================================
    // Terminal Break Tests (for binning coverage)
    // =========================================================================

    #[test]
    fn test_temporal_breaks_date_includes_terminal_break() {
        // Test that interval_breaks includes a terminal break AFTER max_date
        // to ensure all data falls within a bin.
        //
        // Data spanning Jan 15 to Mar 15 with monthly breaks should produce:
        // [Jan-01, Feb-01, Mar-01, Apr-01] - the Apr-01 is the terminal break
        // that closes the [Mar-01, Apr-01] bin for data in March.

        // 2024-01-15 to 2024-03-15 (days since epoch)
        // Jan 15, 2024 = 19738 days since epoch
        // Mar 15, 2024 = 19798 days since epoch
        let monthly = parse_temporal_interval("month").unwrap();
        let breaks: Vec<i32> = interval_breaks(monthly, 19738.0, 19798.0, TemporalUnit::Date)
            .into_iter()
            .map(|d| d as i32)
            .collect();

        // Should include terminal break (Apr-01) after max_date (Mar-15)
        assert!(
            breaks.contains(&days("2024-04-01")),
            "Should include terminal break Apr-01 to close the last bin. Got: {:?}",
            breaks
        );

        // Verify all expected breaks are present
        assert!(breaks.contains(&days("2024-01-01")));
        assert!(breaks.contains(&days("2024-02-01")));
        assert!(breaks.contains(&days("2024-03-01")));

        // The terminal break ensures data from Mar 2-15 falls within [Mar-01, Apr-01]
        assert_eq!(breaks.len(), 4, "Should have 4 breaks for 3 bins");
    }

    #[test]
    fn test_interval_breaks_align_to_count_sized_grid() {
        // "3 months" from Feb 15 steps along the quarter grid (Jan / Apr / Jul),
        // not from February.
        let quarterly = parse_temporal_interval("3 months").unwrap();
        let breaks: Vec<i32> = interval_breaks(
            quarterly,
            days("2024-02-15") as f64,
            days("2024-06-30") as f64,
            TemporalUnit::Date,
        )
        .into_iter()
        .map(|d| d as i32)
        .collect();
        assert_eq!(
            breaks,
            vec![days("2024-01-01"), days("2024-04-01"), days("2024-07-01")]
        );
    }

    #[test]
    fn test_interval_breaks_time_stops_at_midnight() {
        // A terminal break would land past 24:00, so there is none.
        const HOUR_NS: f64 = 3_600_000_000_000.0;
        let hourly = parse_temporal_interval("hour").unwrap();
        let breaks = interval_breaks(
            hourly,
            22.0 * HOUR_NS,
            23.0 * HOUR_NS + 1.0,
            TemporalUnit::Time,
        );
        assert_eq!(breaks, vec![22.0 * HOUR_NS, 23.0 * HOUR_NS]);
    }

    // =========================================================================
    // Graticule Breaks Tests
    // =========================================================================

    #[test]
    fn test_graticule_breaks_world() {
        let breaks = graticule_breaks(-180.0, 180.0, 7);
        assert_eq!(
            breaks,
            vec![-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]
        );
    }

    #[test]
    fn test_graticule_breaks_hemisphere() {
        let breaks = graticule_breaks(-88.0, 88.0, 7);
        assert_eq!(breaks, vec![-60.0, -30.0, 0.0, 30.0, 60.0]);
    }

    #[test]
    fn test_graticule_breaks_small_range() {
        let breaks = graticule_breaks(10.0, 20.0, 7);
        assert!(!breaks.is_empty());
        for &b in &breaks {
            assert!(b > 10.0 && b < 20.0);
        }
    }

    #[test]
    fn test_graticule_breaks_empty_for_zero_range() {
        let breaks = graticule_breaks(50.0, 50.0, 7);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_graticule_breaks_infinite_range_falls_back_to_globe() {
        let breaks = graticule_breaks(f64::NEG_INFINITY, f64::INFINITY, 7);
        assert_eq!(
            breaks,
            vec![-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]
        );
    }

    #[test]
    fn test_graticule_breaks_nan_range_falls_back_to_globe() {
        let breaks = graticule_breaks(f64::NAN, f64::NAN, 7);
        assert_eq!(breaks, graticule_breaks(-180.0, 180.0, 7));
    }

    #[test]
    fn test_graticule_breaks_one_sided_infinity() {
        let breaks = graticule_breaks(0.0, f64::INFINITY, 7);
        assert!(breaks.iter().all(|&b| b > 0.0 && b <= 180.0));
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_graticule_breaks_huge_finite_range_is_clamped() {
        let breaks = graticule_breaks(-1e15, 1e15, 7);
        assert!(breaks.iter().all(|&b| (-180.0..=180.0).contains(&b)));
    }
}
