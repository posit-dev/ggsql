//! Break calculation algorithms for scales
//!
//! Provides functions to calculate axis/legend break positions.

use crate::plot::ArrayElement;

/// Default number of breaks
pub const DEFAULT_BREAK_COUNT: usize = 7;

// =============================================================================
// Wilkinson Extended Algorithm
// =============================================================================

/// "Nice" step multipliers in order of preference (most preferred first).
/// From Talbot et al. "An Extension of Wilkinson's Algorithm for Positioning Tick Labels on Axes"
const Q: &[f64] = &[1.0, 5.0, 2.0, 2.5, 4.0, 3.0];

/// Default scoring weights
const W_SIMPLICITY: f64 = 0.2;
const W_COVERAGE: f64 = 0.25;
const W_DENSITY: f64 = 0.5;
const W_LEGIBILITY: f64 = 0.05;

/// Calculate breaks using Wilkinson Extended labeling algorithm.
///
/// This algorithm searches for optimal axis labeling by scoring candidates
/// on simplicity, coverage, density, and legibility.
///
/// Reference: Talbot, Lin, Hanrahan (2010) "An Extension of Wilkinson's Algorithm
/// for Positioning Tick Labels on Axes"
pub fn wilkinson_extended(min: f64, max: f64, target_count: usize) -> Vec<f64> {
    if target_count == 0 || min >= max || !min.is_finite() || !max.is_finite() {
        return vec![];
    }

    let range = max - min;

    let mut best_score = f64::NEG_INFINITY;
    let mut best_breaks: Vec<f64> = vec![];

    // Search through possible labelings
    // j = skip factor (1 = every Q value, 2 = every other, etc.)
    for j in 1..=target_count.max(10) {
        // q_index = which Q value to use
        for (q_index, &q) in Q.iter().enumerate() {
            // Simplicity score for this q
            let q_score = simplicity_score(q_index, Q.len(), j);

            // Early termination: if best possible score can't beat current best
            if q_score + W_COVERAGE + W_DENSITY + W_LEGIBILITY < best_score {
                continue;
            }

            // k = actual number of ticks (varies around target)
            for k in 2..=(target_count * 2).max(10) {
                let density = density_score(k, target_count);

                // Early termination check
                if q_score + W_COVERAGE + density + W_LEGIBILITY < best_score {
                    continue;
                }

                // Calculate step size
                let delta = (range / (k as f64 - 1.0)) * (j as f64);
                let step = q * nice_step_size(delta / q);

                // Find nice min that covers data
                let nice_min = (min / step).floor() * step;
                let nice_max = nice_min + step * (k as f64 - 1.0);

                // Check coverage
                if nice_max < max {
                    continue; // Doesn't cover data
                }

                let coverage = coverage_score(min, max, nice_min, nice_max);
                let legibility = 1.0; // Simplified: all formats equally legible

                let score = W_SIMPLICITY * q_score
                    + W_COVERAGE * coverage
                    + W_DENSITY * density
                    + W_LEGIBILITY * legibility;

                if score > best_score {
                    best_score = score;
                    best_breaks = generate_breaks(nice_min, step, k);
                }
            }
        }
    }

    // Fallback to simple algorithm if search failed
    if best_breaks.is_empty() {
        return pretty_breaks_simple(min, max, target_count);
    }

    best_breaks
}

/// Simplicity score: prefer earlier Q values and smaller skip factors
fn simplicity_score(q_index: usize, q_len: usize, j: usize) -> f64 {
    1.0 - (q_index as f64) / (q_len as f64) - (j as f64 - 1.0) / 10.0
}

/// Coverage score: penalize extending too far beyond data range
fn coverage_score(data_min: f64, data_max: f64, label_min: f64, label_max: f64) -> f64 {
    let data_range = data_max - data_min;
    let label_range = label_max - label_min;

    if label_range == 0.0 {
        return 0.0;
    }

    // Penalize for extending beyond data
    let extension = (label_range - data_range) / data_range;
    (1.0 - 0.5 * extension).max(0.0)
}

/// Density score: prefer getting close to target count
fn density_score(actual: usize, target: usize) -> f64 {
    let ratio = actual as f64 / target as f64;
    // Prefer slight under-density to over-density
    if ratio >= 1.0 {
        2.0 - ratio
    } else {
        ratio
    }
}

/// Round to nearest power of 10
fn nice_step_size(x: f64) -> f64 {
    10f64.powf(x.log10().round())
}

/// Generate break positions
fn generate_breaks(start: f64, step: f64, count: usize) -> Vec<f64> {
    (0..count).map(|i| start + step * i as f64).collect()
}

/// Wilkinson Extended with preference for including zero.
///
/// Useful for bar charts and other visualizations where zero is meaningful.
pub fn wilkinson_extended_include_zero(min: f64, max: f64, target_count: usize) -> Vec<f64> {
    // If zero is already in range, use standard algorithm
    if min <= 0.0 && max >= 0.0 {
        return wilkinson_extended(min, max, target_count);
    }

    // Extend range to include zero
    let extended_min = if min > 0.0 { 0.0 } else { min };
    let extended_max = if max < 0.0 { 0.0 } else { max };

    wilkinson_extended(extended_min, extended_max, target_count)
}

// =============================================================================
// Pretty Breaks (Public API)
// =============================================================================

/// Calculate pretty breaks using Wilkinson Extended labeling algorithm.
///
/// This is the main entry point for "nice" axis break calculation.
/// Uses an optimization-based approach to find breaks that balance
/// simplicity, coverage, and density.
pub fn pretty_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    wilkinson_extended(min, max, n)
}

/// Legacy simple "nice numbers" algorithm.
///
/// Kept for comparison and fallback purposes.
pub fn pretty_breaks_simple(min: f64, max: f64, n: usize) -> Vec<f64> {
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
///
/// Generates exactly n evenly-spaced breaks from min to max.
/// Use this when `pretty => false` for exact data coverage.
pub fn linear_breaks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        // Single break at midpoint
        return vec![(min + max) / 2.0];
    }

    let step = (max - min) / (n - 1) as f64;
    // Generate exactly n breaks from min to max
    (0..n).map(|i| min + step * i as f64).collect()
}

/// Calculate breaks for integer scales with even spacing.
///
/// Unlike simply rounding the output of `pretty_breaks`, this function
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

/// Calculate "pretty" breaks for exponential scales.
///
/// Mirrors the log 1-2-5 pattern: for base 10, breaks at 0, log10(2), log10(5), 1, ...
/// This produces output values at 1, 2, 5, 10, 20, 50, 100... when exponentiated.
///
/// For exponential transforms, the input space (exponents) is linear, so we want
/// breaks at values that will produce "nice" output values when exponentiated.
pub fn exp_pretty_breaks(min: f64, max: f64, n: usize, base: f64) -> Vec<f64> {
    if n == 0 || min >= max {
        return vec![];
    }

    // The 1-2-5 multipliers in log space
    // For base 10: log10(1)=0, log10(2)≈0.301, log10(5)≈0.699
    let multipliers: [f64; 3] = [1.0, 2.0, 5.0];
    let log_mults: Vec<f64> = multipliers.iter().map(|&m| m.log(base)).collect();

    let floor_min = min.floor();
    let ceil_max = max.ceil();

    let mut breaks = Vec::new();
    let mut exp = floor_min;
    while exp <= ceil_max {
        for &log_mult in &log_mults {
            let val = exp + log_mult;
            if val >= min && val <= max {
                breaks.push(val);
            }
        }
        exp += 1.0;
    }

    // Thin to approximately n breaks if we have too many
    thin_breaks(breaks, n)
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

    // If range extends before first major break, extrapolate backwards
    if let Some((min, _)) = range {
        let first_major = major_breaks[0];
        let mut pos = first_major - step;
        while pos >= min {
            minors.push(pos);
            pos -= step;
        }
    }

    // Add minor breaks between each pair of major breaks
    for window in major_breaks.windows(2) {
        let start = window[0];
        let end = window[1];
        let local_step = (end - start) / (n + 1) as f64;

        for i in 1..=n {
            let pos = start + local_step * i as f64;
            minors.push(pos);
        }
    }

    // If range extends beyond last major break, extrapolate forwards
    if let Some((_, max)) = range {
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

/// Calculate minor breaks for log scales (equal ratios in log space)
///
/// Transforms major breaks to log space, divides evenly, transforms back.
/// This produces minor breaks that are evenly spaced in log space (equal ratios).
///
/// # Arguments
/// - `major_breaks`: The major break positions (must be positive and sorted)
/// - `n`: Number of minor breaks per major interval
/// - `base`: The logarithm base (e.g., 10.0, 2.0, E)
/// - `range`: Optional (min, max) scale input range to extend minor breaks beyond major breaks
///
/// # Returns
/// Minor break positions (excluding major breaks)
pub fn minor_breaks_log(
    major_breaks: &[f64],
    n: usize,
    base: f64,
    range: Option<(f64, f64)>,
) -> Vec<f64> {
    if major_breaks.len() < 2 || n == 0 {
        return vec![];
    }

    // Filter to positive values only
    let positive_majors: Vec<f64> = major_breaks.iter().copied().filter(|&x| x > 0.0).collect();

    if positive_majors.len() < 2 {
        return vec![];
    }

    // Transform to log space
    let log_majors: Vec<f64> = positive_majors.iter().map(|&x| x.log(base)).collect();

    // Calculate minor breaks in log space
    let log_range = range.map(|(min, max)| {
        let log_min = if min > 0.0 {
            min.log(base)
        } else {
            log_majors[0] - (log_majors[1] - log_majors[0])
        };
        let log_max = max.log(base);
        (log_min, log_max)
    });

    let log_minors = minor_breaks_linear(&log_majors, n, log_range);

    // Transform back to data space
    log_minors.into_iter().map(|x| base.powf(x)).collect()
}

/// Calculate minor breaks in sqrt space
///
/// Transforms to sqrt space, divides evenly, squares back.
///
/// # Arguments
/// - `major_breaks`: The major break positions (must be non-negative and sorted)
/// - `n`: Number of minor breaks per major interval
/// - `range`: Optional (min, max) scale input range to extend minor breaks beyond major breaks
///
/// # Returns
/// Minor break positions (excluding major breaks)
pub fn minor_breaks_sqrt(major_breaks: &[f64], n: usize, range: Option<(f64, f64)>) -> Vec<f64> {
    if major_breaks.len() < 2 || n == 0 {
        return vec![];
    }

    // Filter to non-negative values only
    let nonneg_majors: Vec<f64> = major_breaks.iter().copied().filter(|&x| x >= 0.0).collect();

    if nonneg_majors.len() < 2 {
        return vec![];
    }

    // Transform to sqrt space
    let sqrt_majors: Vec<f64> = nonneg_majors.iter().map(|&x| x.sqrt()).collect();

    // Calculate minor breaks in sqrt space
    let sqrt_range = range.map(|(min, max)| (min.max(0.0).sqrt(), max.sqrt()));

    let sqrt_minors = minor_breaks_linear(&sqrt_majors, n, sqrt_range);

    // Transform back to data space (square)
    sqrt_minors.into_iter().map(|x| x * x).collect()
}

/// Calculate minor breaks for symlog scales
///
/// Uses asinh transform space for even division. Handles negative values.
///
/// # Arguments
/// - `major_breaks`: The major break positions (sorted)
/// - `n`: Number of minor breaks per major interval
/// - `range`: Optional (min, max) scale input range to extend minor breaks beyond major breaks
///
/// # Returns
/// Minor break positions (excluding major breaks)
pub fn minor_breaks_symlog(major_breaks: &[f64], n: usize, range: Option<(f64, f64)>) -> Vec<f64> {
    if major_breaks.len() < 2 || n == 0 {
        return vec![];
    }

    // Transform to asinh space
    let asinh_majors: Vec<f64> = major_breaks.iter().map(|&x| x.asinh()).collect();

    // Calculate minor breaks in asinh space
    let asinh_range = range.map(|(min, max)| (min.asinh(), max.asinh()));

    let asinh_minors = minor_breaks_linear(&asinh_majors, n, asinh_range);

    // Transform back to data space
    asinh_minors.into_iter().map(|x| x.sinh()).collect()
}

/// Trim breaks to be within the specified range (inclusive)
///
/// # Arguments
/// - `breaks`: The break positions to filter
/// - `range`: The (min, max) range to keep
///
/// # Returns
/// Break positions that fall within [min, max]
pub fn trim_breaks(breaks: &[f64], range: (f64, f64)) -> Vec<f64> {
    breaks
        .iter()
        .copied()
        .filter(|&b| b >= range.0 && b <= range.1)
        .collect()
}

/// Trim temporal breaks to be within the specified range (inclusive)
///
/// Uses string comparison for ISO-format dates (works for Date, DateTime, Time).
///
/// # Arguments
/// - `breaks`: The break positions as ISO strings
/// - `range`: The (min, max) range as ISO strings
///
/// # Returns
/// Break positions that fall within the range
pub fn trim_temporal_breaks(breaks: &[String], range: (&str, &str)) -> Vec<String> {
    breaks
        .iter()
        .filter(|b| b.as_str() >= range.0 && b.as_str() <= range.1)
        .cloned()
        .collect()
}

/// Temporal minor break specification
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MinorBreakSpec {
    /// Derive minor interval from major interval (default)
    #[default]
    Auto,
    /// Explicit count per major interval
    Count(usize),
    /// Explicit interval string
    Interval(String),
}

/// Derive minor interval from major interval (keeps count below 10)
///
/// Returns the recommended minor interval string for a given major interval.
///
/// | Major Epoch | Minor Epoch  | Approx Count |
/// |-------------|--------------|--------------|
/// | year        | 3 months     | 4            |
/// | quarter     | month        | 3            |
/// | month       | week         | ~4           |
/// | week        | day          | 7            |
/// | day         | 6 hours      | 4            |
/// | hour        | 15 minutes   | 4            |
/// | minute      | 15 seconds   | 4            |
/// | second      | 100 ms       | 10           |
pub fn derive_minor_interval(major_interval: &str) -> &'static str {
    let interval = TemporalInterval::create_from_str(major_interval);
    match interval {
        Some(TemporalInterval {
            unit: TemporalUnit::Year,
            ..
        }) => "3 months",
        Some(TemporalInterval {
            unit: TemporalUnit::Month,
            count,
        }) if count >= 3 => "month", // quarter -> month
        Some(TemporalInterval {
            unit: TemporalUnit::Month,
            ..
        }) => "week",
        Some(TemporalInterval {
            unit: TemporalUnit::Week,
            ..
        }) => "day",
        Some(TemporalInterval {
            unit: TemporalUnit::Day,
            ..
        }) => "6 hours",
        Some(TemporalInterval {
            unit: TemporalUnit::Hour,
            ..
        }) => "15 minutes",
        Some(TemporalInterval {
            unit: TemporalUnit::Minute,
            ..
        }) => "15 seconds",
        Some(TemporalInterval {
            unit: TemporalUnit::Second,
            ..
        }) => "100 ms",
        None => "day", // fallback
    }
}

/// Calculate temporal minor breaks for Date scale
///
/// # Arguments
/// - `major_breaks`: Major break positions as ISO date strings ("YYYY-MM-DD")
/// - `major_interval`: The major interval string (e.g., "month", "year")
/// - `spec`: Minor break specification (Auto, Count, or Interval)
/// - `range`: Optional (min, max) as ISO date strings to extend minor breaks
///
/// # Returns
/// Minor break positions as ISO date strings
pub fn temporal_minor_breaks_date(
    major_breaks: &[String],
    major_interval: &str,
    spec: MinorBreakSpec,
    range: Option<(&str, &str)>,
) -> Vec<String> {
    use chrono::NaiveDate;

    if major_breaks.len() < 2 {
        return vec![];
    }

    // Parse major breaks to dates
    let major_dates: Vec<NaiveDate> = major_breaks
        .iter()
        .filter_map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    if major_dates.len() < 2 {
        return vec![];
    }

    let minor_interval = match spec {
        MinorBreakSpec::Auto => derive_minor_interval(major_interval).to_string(),
        MinorBreakSpec::Count(n) => {
            // Calculate interval between first two majors and divide by n+1
            let days = (major_dates[1] - major_dates[0]).num_days();
            let minor_days = days / (n + 1) as i64;
            format!("{} days", minor_days.max(1))
        }
        MinorBreakSpec::Interval(s) => s,
    };

    let interval = match TemporalInterval::create_from_str(&minor_interval) {
        Some(i) => i,
        None => return vec![],
    };

    let mut minors = Vec::new();

    // Parse range bounds
    let range_dates = range.and_then(|(min, max)| {
        let min_date = NaiveDate::parse_from_str(min, "%Y-%m-%d").ok()?;
        let max_date = NaiveDate::parse_from_str(max, "%Y-%m-%d").ok()?;
        Some((min_date, max_date))
    });

    // If range extends before first major, extrapolate backwards
    if let Some((min_date, _)) = range_dates {
        let first_major = major_dates[0];
        let mut current = retreat_date_by_interval(first_major, &interval);
        while current >= min_date {
            minors.push(current.format("%Y-%m-%d").to_string());
            current = retreat_date_by_interval(current, &interval);
        }
    }

    // Add minors between each pair of major breaks
    for window in major_dates.windows(2) {
        let start = window[0];
        let end = window[1];
        let mut current = advance_date_by_interval(start, &interval);
        while current < end {
            minors.push(current.format("%Y-%m-%d").to_string());
            current = advance_date_by_interval(current, &interval);
        }
    }

    // If range extends beyond last major, extrapolate forwards
    if let Some((_, max_date)) = range_dates {
        let last_major = *major_dates.last().unwrap();
        let mut current = advance_date_by_interval(last_major, &interval);
        while current <= max_date {
            minors.push(current.format("%Y-%m-%d").to_string());
            current = advance_date_by_interval(current, &interval);
        }
    }

    minors.sort();
    minors
}

/// Retreat a date by the given interval (go backwards)
fn retreat_date_by_interval(
    date: chrono::NaiveDate,
    interval: &TemporalInterval,
) -> chrono::NaiveDate {
    use chrono::{Datelike, NaiveDate};

    let count = interval.count as i64;
    match interval.unit {
        TemporalUnit::Day => date - chrono::Duration::days(count),
        TemporalUnit::Week => date - chrono::Duration::weeks(count),
        TemporalUnit::Month => {
            let total_months = date.year() * 12 + date.month() as i32 - 1 - count as i32;
            let year = total_months.div_euclid(12);
            let month = (total_months.rem_euclid(12)) as u32 + 1;
            NaiveDate::from_ymd_opt(year, month, 1).unwrap_or(date)
        }
        TemporalUnit::Year => {
            NaiveDate::from_ymd_opt(date.year() - count as i32, 1, 1).unwrap_or(date)
        }
        _ => date - chrono::Duration::days(count),
    }
}

/// Calculate temporal minor breaks for DateTime scale
///
/// # Arguments
/// - `major_breaks`: Major break positions as ISO datetime strings
/// - `major_interval`: The major interval string
/// - `spec`: Minor break specification
/// - `range`: Optional (min, max) as ISO datetime strings
///
/// # Returns
/// Minor break positions as ISO datetime strings
pub fn temporal_minor_breaks_datetime(
    major_breaks: &[String],
    major_interval: &str,
    spec: MinorBreakSpec,
    range: Option<(&str, &str)>,
) -> Vec<String> {
    use chrono::{DateTime, Utc};

    if major_breaks.len() < 2 {
        return vec![];
    }

    // Parse major breaks to datetimes
    let major_dts: Vec<DateTime<Utc>> = major_breaks
        .iter()
        .filter_map(|s| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        })
        .collect();

    if major_dts.len() < 2 {
        return vec![];
    }

    let minor_interval = match spec {
        MinorBreakSpec::Auto => derive_minor_interval(major_interval).to_string(),
        MinorBreakSpec::Count(n) => {
            let duration = major_dts[1] - major_dts[0];
            let minor_secs = duration.num_seconds() / (n + 1) as i64;
            if minor_secs >= 3600 {
                format!("{} hours", minor_secs / 3600)
            } else if minor_secs >= 60 {
                format!("{} minutes", minor_secs / 60)
            } else {
                format!("{} seconds", minor_secs.max(1))
            }
        }
        MinorBreakSpec::Interval(s) => s,
    };

    let interval = match TemporalInterval::create_from_str(&minor_interval) {
        Some(i) => i,
        None => return vec![],
    };

    let mut minors = Vec::new();

    // Parse range bounds
    let range_dts = range.and_then(|(min, max)| {
        let min_dt = DateTime::parse_from_rfc3339(min)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))?;
        let max_dt = DateTime::parse_from_rfc3339(max)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))?;
        Some((min_dt, max_dt))
    });

    // If range extends before first major, extrapolate backwards
    if let Some((min_dt, _)) = range_dts {
        let first_major = major_dts[0];
        let mut current = retreat_datetime_by_interval(first_major, &interval);
        while current >= min_dt {
            minors.push(current.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string());
            current = retreat_datetime_by_interval(current, &interval);
        }
    }

    // Add minors between each pair of major breaks
    for window in major_dts.windows(2) {
        let start = window[0];
        let end = window[1];
        let mut current = advance_datetime_by_interval(start, &interval);
        while current < end {
            minors.push(current.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string());
            current = advance_datetime_by_interval(current, &interval);
        }
    }

    // If range extends beyond last major, extrapolate forwards
    if let Some((_, max_dt)) = range_dts {
        let last_major = *major_dts.last().unwrap();
        let mut current = advance_datetime_by_interval(last_major, &interval);
        while current <= max_dt {
            minors.push(current.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string());
            current = advance_datetime_by_interval(current, &interval);
        }
    }

    minors.sort();
    minors
}

/// Retreat a datetime by the given interval (go backwards)
fn retreat_datetime_by_interval(
    dt: chrono::DateTime<chrono::Utc>,
    interval: &TemporalInterval,
) -> chrono::DateTime<chrono::Utc> {
    use chrono::{Datelike, TimeZone, Timelike, Utc};

    let count = interval.count as i64;
    match interval.unit {
        TemporalUnit::Second => dt - chrono::Duration::seconds(count),
        TemporalUnit::Minute => dt - chrono::Duration::minutes(count),
        TemporalUnit::Hour => dt - chrono::Duration::hours(count),
        TemporalUnit::Day => dt - chrono::Duration::days(count),
        TemporalUnit::Week => dt - chrono::Duration::weeks(count),
        TemporalUnit::Month => {
            let total_months = dt.year() * 12 + dt.month() as i32 - 1 - count as i32;
            let year = total_months.div_euclid(12);
            let month = (total_months.rem_euclid(12)) as u32 + 1;
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
                dt.year() - count as i32,
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

/// Calculate temporal minor breaks for Time scale
///
/// # Arguments
/// - `major_breaks`: Major break positions as time strings ("HH:MM:SS.mmm")
/// - `major_interval`: The major interval string
/// - `spec`: Minor break specification
/// - `range`: Optional (min, max) as time strings
///
/// # Returns
/// Minor break positions as time strings
pub fn temporal_minor_breaks_time(
    major_breaks: &[String],
    major_interval: &str,
    spec: MinorBreakSpec,
    range: Option<(&str, &str)>,
) -> Vec<String> {
    use chrono::NaiveTime;

    if major_breaks.len() < 2 {
        return vec![];
    }

    // Parse major breaks to times
    let major_times: Vec<NaiveTime> = major_breaks
        .iter()
        .filter_map(|s| NaiveTime::parse_from_str(s, "%H:%M:%S%.3f").ok())
        .collect();

    if major_times.len() < 2 {
        return vec![];
    }

    let minor_interval = match spec {
        MinorBreakSpec::Auto => derive_minor_interval(major_interval).to_string(),
        MinorBreakSpec::Count(n) => {
            let duration = major_times[1] - major_times[0];
            let minor_secs = duration.num_seconds() / (n + 1) as i64;
            if minor_secs >= 60 {
                format!("{} minutes", minor_secs / 60)
            } else {
                format!("{} seconds", minor_secs.max(1))
            }
        }
        MinorBreakSpec::Interval(s) => s,
    };

    let interval = match TemporalInterval::create_from_str(&minor_interval) {
        Some(i) => i,
        None => return vec![],
    };

    let mut minors = Vec::new();

    // Parse range bounds
    let range_times = range.and_then(|(min, max)| {
        let min_time = NaiveTime::parse_from_str(min, "%H:%M:%S%.3f").ok()?;
        let max_time = NaiveTime::parse_from_str(max, "%H:%M:%S%.3f").ok()?;
        Some((min_time, max_time))
    });

    // If range extends before first major, extrapolate backwards
    if let Some((min_time, _)) = range_times {
        let first_major = major_times[0];
        if let Some(mut current) = retreat_time_by_interval(first_major, &interval) {
            while current >= min_time && current < first_major {
                minors.push(current.format("%H:%M:%S%.3f").to_string());
                match retreat_time_by_interval(current, &interval) {
                    Some(prev) if prev < current => current = prev,
                    _ => break,
                }
            }
        }
    }

    // Add minors between each pair of major breaks
    for window in major_times.windows(2) {
        let start = window[0];
        let end = window[1];
        if let Some(mut current) = advance_time_by_interval(start, &interval) {
            while current < end {
                minors.push(current.format("%H:%M:%S%.3f").to_string());
                match advance_time_by_interval(current, &interval) {
                    Some(next) if next > current => current = next,
                    _ => break,
                }
            }
        }
    }

    // If range extends beyond last major, extrapolate forwards
    if let Some((_, max_time)) = range_times {
        let last_major = *major_times.last().unwrap();
        if let Some(mut current) = advance_time_by_interval(last_major, &interval) {
            while current <= max_time && current > last_major {
                minors.push(current.format("%H:%M:%S%.3f").to_string());
                match advance_time_by_interval(current, &interval) {
                    Some(next) if next > current => current = next,
                    _ => break,
                }
            }
        }
    }

    minors.sort();
    minors
}

/// Retreat a time by the given interval (go backwards)
fn retreat_time_by_interval(
    time: chrono::NaiveTime,
    interval: &TemporalInterval,
) -> Option<chrono::NaiveTime> {
    let count = interval.count as i64;
    let duration = match interval.unit {
        TemporalUnit::Second => chrono::Duration::seconds(count),
        TemporalUnit::Minute => chrono::Duration::minutes(count),
        TemporalUnit::Hour => chrono::Duration::hours(count),
        _ => return Some(time), // Day/week/month/year not applicable
    };
    time.overflowing_sub_signed(duration).0.into()
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
    pub fn create_from_str(s: &str) -> Option<Self> {
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

    let count = interval.count;
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
    // Pretty Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_pretty_breaks_variations() {
        // Test various range sizes
        let test_cases: Vec<(f64, f64, usize)> = vec![
            (0.0, 100.0, 5),   // Basic range
            (0.1, 0.9, 5),     // Small range
            (0.0, 10000.0, 5), // Large range
        ];
        for (min, max, n) in test_cases {
            let breaks = pretty_breaks(min, max, n);
            assert!(
                !breaks.is_empty(),
                "pretty_breaks({}, {}, {}) should not be empty",
                min,
                max,
                n
            );
            assert!(
                breaks[0] <= min,
                "pretty_breaks({}, {}, {}): first should be <= min",
                min,
                max,
                n
            );
            assert!(
                *breaks.last().unwrap() >= max,
                "pretty_breaks({}, {}, {}): last should be >= max",
                min,
                max,
                n
            );
        }
    }

    #[test]
    fn test_pretty_breaks_edge_cases() {
        assert!(
            pretty_breaks(0.0, 100.0, 0).is_empty(),
            "zero count should return empty"
        );
        assert!(
            pretty_breaks(50.0, 50.0, 5).is_empty(),
            "equal min/max should return empty"
        );
    }

    // =========================================================================
    // Linear Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_linear_breaks_variations() {
        // Test various counts
        // Format: (min, max, n, expected_result)
        let test_cases: Vec<(f64, f64, usize, Vec<f64>)> = vec![
            (0.0, 100.0, 5, vec![0.0, 25.0, 50.0, 75.0, 100.0]),
            (0.0, 100.0, 1, vec![50.0]),       // Single break at midpoint
            (0.0, 100.0, 2, vec![0.0, 100.0]), // Two breaks at endpoints
            (10.0, 90.0, 5, vec![10.0, 30.0, 50.0, 70.0, 90.0]), // Non-zero start
        ];
        for (min, max, n, expected) in test_cases {
            let breaks = linear_breaks(min, max, n);
            assert_eq!(breaks, expected, "linear_breaks({}, {}, {})", min, max, n);
        }
    }

    #[test]
    fn test_linear_breaks_edge_cases() {
        assert!(
            linear_breaks(0.0, 100.0, 0).is_empty(),
            "zero count should return empty"
        );
    }

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
    // Log Break Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_log_breaks_powers() {
        // pretty=false should give powers of base
        let test_cases = vec![
            // (min, max, base, expected)
            (1.0, 10000.0, 10.0, vec![1.0, 10.0, 100.0, 1000.0, 10000.0]),
            (1.0, 16.0, 2.0, vec![1.0, 2.0, 4.0, 8.0, 16.0]),
            (0.01, 100.0, 10.0, vec![0.01, 0.1, 1.0, 10.0, 100.0]),
        ];
        for (min, max, base, expected) in test_cases {
            let breaks = log_breaks(min, max, 10, base, false);
            assert_eq!(
                breaks, expected,
                "log_breaks({}, {}, base={})",
                min, max, base
            );
        }
    }

    #[test]
    fn test_log_breaks_pretty_1_2_5_pattern() {
        // pretty=true should give 1-2-5 pattern
        let breaks = log_breaks(1.0, 100.0, 10, 10.0, true);
        for &v in &[1.0, 2.0, 5.0, 10.0, 100.0] {
            assert!(
                breaks.contains(&v),
                "log_breaks pretty should contain {}",
                v
            );
        }
    }

    #[test]
    fn test_log_breaks_filters_negative() {
        // Range includes negative - should only return positive breaks
        let breaks = log_breaks(-10.0, 1000.0, 10, 10.0, false);
        assert!(breaks.iter().all(|&v| v > 0.0));
        for &v in &[1.0, 10.0, 100.0, 1000.0] {
            assert!(
                breaks.contains(&v),
                "log_breaks should contain {} after filtering negative",
                v
            );
        }
    }

    #[test]
    fn test_log_breaks_edge_cases() {
        assert!(
            log_breaks(-100.0, -1.0, 5, 10.0, true).is_empty(),
            "all negative should return empty"
        );
        assert!(
            log_breaks(1.0, 100.0, 0, 10.0, true).is_empty(),
            "zero count should return empty"
        );
    }

    // =========================================================================
    // Sqrt Break Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_sqrt_breaks_variations() {
        // Basic case
        let breaks = sqrt_breaks(0.0, 100.0, 5, false);
        assert!(breaks.len() >= 5, "Should have at least 5 breaks");
        assert!(
            breaks.first().unwrap() >= &0.0,
            "First break should be >= 0"
        );
        assert!(
            breaks.last().unwrap() >= &100.0,
            "Last break should be >= 100"
        );

        // With negative input (should filter)
        let breaks_neg = sqrt_breaks(-10.0, 100.0, 5, true);
        assert!(
            breaks_neg.iter().all(|&v| v >= 0.0),
            "Should filter negative values"
        );

        // Pretty mode
        let breaks_pretty = sqrt_breaks(0.0, 100.0, 5, true);
        assert!(!breaks_pretty.is_empty());
    }

    #[test]
    fn test_sqrt_breaks_edge_cases() {
        assert!(
            sqrt_breaks(0.0, 100.0, 0, true).is_empty(),
            "zero count should return empty"
        );
    }

    // =========================================================================
    // Symlog Break Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_symlog_breaks_variations() {
        // Symmetric range - should have negatives, zero, positives
        let breaks_sym = symlog_breaks(-1000.0, 1000.0, 10, false);
        assert!(
            breaks_sym.contains(&0.0),
            "Symmetric range should contain 0"
        );
        assert!(
            breaks_sym.iter().any(|&v| v < 0.0),
            "Should have negative values"
        );
        assert!(
            breaks_sym.iter().any(|&v| v > 0.0),
            "Should have positive values"
        );

        // Positive only
        let breaks_pos = symlog_breaks(1.0, 1000.0, 5, false);
        assert!(
            breaks_pos.iter().all(|&v| v > 0.0),
            "Positive-only should have only positive"
        );

        // Negative only
        let breaks_neg = symlog_breaks(-1000.0, -1.0, 5, false);
        assert!(
            breaks_neg.iter().all(|&v| v < 0.0),
            "Negative-only should have only negative"
        );

        // Crossing zero should include zero
        let breaks_cross = symlog_breaks(-100.0, 100.0, 7, false);
        assert!(
            breaks_cross.contains(&0.0),
            "Crossing zero should include 0"
        );
    }

    #[test]
    fn test_symlog_breaks_edge_cases() {
        assert!(
            symlog_breaks(-100.0, 100.0, 0, true).is_empty(),
            "zero count should return empty"
        );
    }

    // =========================================================================
    // Calculate Breaks Dispatch Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_calculate_breaks_dispatch() {
        // Test that calculate_breaks dispatches to correct algorithm
        // Format: (min, max, n, transform, pretty, expected_values_to_contain)
        let test_cases: Vec<(f64, f64, usize, Option<&str>, bool, Vec<f64>)> = vec![
            (1.0, 1000.0, 5, Some("log10"), false, vec![10.0, 100.0]),
            (1.0, 16.0, 10, Some("log2"), false, vec![2.0, 4.0, 8.0]),
            (-100.0, 100.0, 7, Some("asinh"), false, vec![0.0]),
            (-100.0, 100.0, 7, Some("pseudo_log"), false, vec![0.0]),
        ];
        for (min, max, n, transform, pretty, expected) in test_cases {
            let breaks = calculate_breaks(min, max, n, transform, pretty);
            for v in expected {
                assert!(
                    breaks.contains(&v),
                    "calculate_breaks({}, {}, {:?}) should contain {}",
                    min,
                    max,
                    transform,
                    v
                );
            }
        }
    }

    #[test]
    fn test_calculate_breaks_identity_mode() {
        // Identity mode (None transform)
        let breaks_pretty = calculate_breaks(0.0, 100.0, 5, None, true);
        assert!(!breaks_pretty.is_empty(), "Identity pretty should work");

        let breaks_linear = calculate_breaks(0.0, 100.0, 5, None, false);
        assert_eq!(
            breaks_linear,
            vec![0.0, 25.0, 50.0, 75.0, 100.0],
            "Identity linear should be evenly spaced"
        );

        // Unknown transform should fall back to identity
        let breaks_unknown = calculate_breaks(0.0, 100.0, 5, Some("unknown"), true);
        assert!(
            !breaks_unknown.is_empty(),
            "Unknown transform should fall back to identity"
        );
    }

    // =========================================================================
    // Thin Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_thin_breaks_variations() {
        // No thinning needed
        let result_none = thin_breaks(vec![1.0, 2.0, 3.0, 4.0, 5.0], 10);
        assert_eq!(result_none, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Thin to smaller
        let result_thin = thin_breaks(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        assert_eq!(result_thin.len(), 5);
        assert_eq!(result_thin[0], 1.0);
        assert_eq!(result_thin[4], 10.0);

        // Thin to single - should be middle
        let result_single = thin_breaks(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1);
        assert_eq!(result_single, vec![3.0]);
    }

    // =========================================================================
    // Temporal Interval Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_temporal_interval_parsing() {
        // Simple unit names
        let simple = TemporalInterval::create_from_str("month").unwrap();
        assert_eq!(simple.count, 1);
        assert_eq!(simple.unit, TemporalUnit::Month);

        // With count prefix
        let with_count = TemporalInterval::create_from_str("2 months").unwrap();
        assert_eq!(with_count.count, 2);
        assert_eq!(with_count.unit, TemporalUnit::Month);

        // All unit names should parse
        for unit in &[
            "second", "seconds", "minute", "hour", "day", "week", "month", "year",
        ] {
            assert!(
                TemporalInterval::create_from_str(unit).is_some(),
                "{} should parse",
                unit
            );
        }

        // Invalid inputs
        for invalid in &["invalid", "foo bar baz", ""] {
            assert!(
                TemporalInterval::create_from_str(invalid).is_none(),
                "{} should not parse",
                invalid
            );
        }
    }

    // =========================================================================
    // Temporal Date Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_temporal_breaks_date_various_intervals() {
        // Monthly breaks: 2024-01-15 to 2024-04-15
        let monthly = TemporalInterval::create_from_str("month").unwrap();
        let breaks_monthly = temporal_breaks_date(19738, 19828, monthly);
        assert_eq!(breaks_monthly[0], "2024-01-01");
        for month in &["2024-02-01", "2024-03-01", "2024-04-01"] {
            assert!(
                breaks_monthly.contains(&month.to_string()),
                "Monthly should contain {}",
                month
            );
        }

        // Bimonthly: 2024-01-01 to 2024-07-01
        let bimonthly = TemporalInterval::create_from_str("2 months").unwrap();
        let breaks_bi = temporal_breaks_date(19724, 19907, bimonthly);
        assert!(breaks_bi.contains(&"2024-03-01".to_string()));
        assert!(
            !breaks_bi.contains(&"2024-02-01".to_string()),
            "Bimonthly should skip Feb"
        );

        // Yearly: 2022-01-01 to 2024-12-31
        let yearly = TemporalInterval::create_from_str("year").unwrap();
        let breaks_yearly = temporal_breaks_date(18993, 20089, yearly);
        for year in &["2022-01-01", "2023-01-01", "2024-01-01"] {
            assert!(
                breaks_yearly.contains(&year.to_string()),
                "Yearly should contain {}",
                year
            );
        }

        // Weekly: ~30 days
        let weekly = TemporalInterval::create_from_str("week").unwrap();
        let breaks_weekly = temporal_breaks_date(19724, 19754, weekly);
        assert!(
            breaks_weekly.len() >= 4,
            "Weekly should have at least 4 breaks"
        );
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
    // Minor Breaks Log/Sqrt/Symlog Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_minor_breaks_log_variations() {
        // Basic case
        let minors_basic = minor_breaks_log(&[1.0, 10.0, 100.0], 8, 10.0, None);
        assert_eq!(minors_basic.len(), 16, "8 per decade × 2 decades");

        // Single minor (geometric mean)
        let minors_single = minor_breaks_log(&[1.0, 10.0, 100.0], 1, 10.0, None);
        assert_eq!(minors_single.len(), 2);
        assert!((minors_single[0] - (1.0_f64 * 10.0).sqrt()).abs() < 0.01);

        // With extension
        let minors_ext = minor_breaks_log(&[10.0, 100.0], 8, 10.0, Some((1.0, 1000.0)));
        assert_eq!(minors_ext.len(), 24, "8 per decade × 3 decades");

        // Filters negative
        let minors_neg = minor_breaks_log(&[-10.0, 1.0, 10.0, 100.0], 1, 10.0, None);
        assert!(minors_neg.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_minor_breaks_sqrt_variations() {
        // Basic case - midpoints in sqrt space, squared back
        let minors_basic = minor_breaks_sqrt(&[0.0, 25.0, 100.0], 1, None);
        assert_eq!(minors_basic.len(), 2);
        assert!((minors_basic[0] - 6.25).abs() < 0.01);
        assert!((minors_basic[1] - 56.25).abs() < 0.01);

        // With extension
        let minors_ext = minor_breaks_sqrt(&[25.0, 100.0], 1, Some((0.0, 225.0)));
        assert!(minors_ext.len() >= 2);

        // Filters negative
        let minors_neg = minor_breaks_sqrt(&[-10.0, 0.0, 25.0, 100.0], 1, None);
        assert!(minors_neg.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_minor_breaks_symlog_variations() {
        // Basic case - one minor per interval
        let minors_basic = minor_breaks_symlog(&[-100.0, -10.0, 0.0, 10.0, 100.0], 1, None);
        assert_eq!(minors_basic.len(), 4);

        // Crossing zero - midpoint should be near 0
        let minors_cross = minor_breaks_symlog(&[-10.0, 10.0], 1, None);
        assert_eq!(minors_cross.len(), 1);
        assert!(
            minors_cross[0].abs() < 1.0,
            "Midpoint crossing zero should be near 0"
        );

        // With extension
        let minors_ext = minor_breaks_symlog(&[0.0, 100.0], 1, Some((-100.0, 200.0)));
        assert!(minors_ext.len() >= 2);
    }

    // =========================================================================
    // Trim Breaks Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_trim_breaks_variations() {
        // Trim from both ends
        let trimmed = trim_breaks(&[5.0, 10.0, 15.0, 20.0, 25.0, 30.0], (10.0, 25.0));
        assert_eq!(trimmed, vec![10.0, 15.0, 20.0, 25.0]);

        // All outside range
        let empty = trim_breaks(&[5.0, 10.0, 15.0], (20.0, 30.0));
        assert!(empty.is_empty());

        // All inside range
        let all = trim_breaks(&[15.0, 20.0, 25.0], (10.0, 30.0));
        assert_eq!(all, vec![15.0, 20.0, 25.0]);
    }

    #[test]
    fn test_trim_temporal_breaks_variations() {
        // Trim to middle
        let breaks = vec![
            "2024-01-01".to_string(),
            "2024-02-01".to_string(),
            "2024-03-01".to_string(),
        ];
        let trimmed = trim_temporal_breaks(&breaks, ("2024-01-15", "2024-02-15"));
        assert_eq!(trimmed, vec!["2024-02-01".to_string()]);

        // All inside
        let all_inside = trim_temporal_breaks(
            &["2024-02-01".to_string(), "2024-02-15".to_string()],
            ("2024-01-01", "2024-03-01"),
        );
        assert_eq!(all_inside.len(), 2);
    }

    // =========================================================================
    // Derive Minor Interval Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_derive_minor_interval_all_units() {
        let expected = vec![
            ("year", "3 months"),
            ("3 months", "month"),
            ("month", "week"),
            ("week", "day"),
            ("day", "6 hours"),
            ("hour", "15 minutes"),
            ("minute", "15 seconds"),
            ("invalid", "day"), // Falls back to day
        ];
        for (input, expected_output) in expected {
            assert_eq!(
                derive_minor_interval(input),
                expected_output,
                "derive_minor_interval({}) should be {}",
                input,
                expected_output
            );
        }
    }

    // =========================================================================
    // Temporal Minor Breaks & MinorBreakSpec Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_temporal_minor_breaks_date_variations() {
        let majors = vec![
            "2024-01-01".to_string(),
            "2024-02-01".to_string(),
            "2024-03-01".to_string(),
        ];

        // Auto mode - derives "week" from "month"
        let minors_auto = temporal_minor_breaks_date(&majors, "month", MinorBreakSpec::Auto, None);
        assert!(!minors_auto.is_empty());
        assert!(minors_auto.iter().any(|d| d.starts_with("2024-01")));
        assert!(minors_auto.iter().any(|d| d.starts_with("2024-02")));

        // By count
        let minors_count =
            temporal_minor_breaks_date(&majors[..2], "month", MinorBreakSpec::Count(3), None);
        assert!(!minors_count.is_empty());

        // By interval
        let minors_interval = temporal_minor_breaks_date(
            &majors[..2],
            "month",
            MinorBreakSpec::Interval("week".to_string()),
            None,
        );
        assert!(minors_interval.len() >= 3, "January has about 4 weeks");

        // With extension
        let minors_ext = temporal_minor_breaks_date(
            &["2024-02-01".to_string(), "2024-03-01".to_string()],
            "month",
            MinorBreakSpec::Interval("week".to_string()),
            Some(("2024-01-01", "2024-04-01")),
        );
        assert!(
            minors_ext.iter().any(|d| d.starts_with("2024-01")),
            "Should extend into January"
        );
        assert!(
            minors_ext.iter().any(|d| d.starts_with("2024-03")),
            "Should extend into March"
        );

        // Single major returns empty
        let minors_single = temporal_minor_breaks_date(
            &["2024-01-01".to_string()],
            "month",
            MinorBreakSpec::Auto,
            None,
        );
        assert!(minors_single.is_empty());
    }

    #[test]
    fn test_minor_break_spec_types() {
        assert_eq!(MinorBreakSpec::default(), MinorBreakSpec::Auto);
        assert_eq!(MinorBreakSpec::Count(4), MinorBreakSpec::Count(4));
        assert_eq!(
            MinorBreakSpec::Interval("week".to_string()),
            MinorBreakSpec::Interval("week".to_string())
        );
    }

    // =========================================================================
    // Wilkinson Extended Tests (Consolidated)
    // =========================================================================

    #[test]
    fn test_wilkinson_basic_properties() {
        // Test various ranges - should produce nice round numbers
        let test_cases = vec![
            (0.0, 100.0, 5),
            (0.1, 0.9, 5),
            (0.0, 1_000_000.0, 5),
            (-50.0, 50.0, 5),
            (0.0, 152.0, 5), // penguin scenario
        ];
        for (min, max, n) in test_cases {
            let breaks = wilkinson_extended(min, max, n);
            assert!(
                !breaks.is_empty(),
                "wilkinson_extended({}, {}, {}) should not be empty",
                min,
                max,
                n
            );
            assert!(
                breaks.len() >= 3 && breaks.len() <= 10,
                "wilkinson({}, {}, {}) count should be reasonable",
                min,
                max,
                n
            );
        }
    }

    #[test]
    fn test_wilkinson_prefers_nice_numbers() {
        let breaks = wilkinson_extended(0.0, 97.0, 5);
        for b in &breaks {
            let normalized = b / 10.0;
            let is_nice = normalized.fract() == 0.0
                || (normalized * 2.0).fract() == 0.0
                || (normalized * 4.0).fract() == 0.0;
            assert!(is_nice, "Break {} should be a nice number", b);
        }
    }

    #[test]
    fn test_wilkinson_covers_data() {
        let breaks = wilkinson_extended(7.3, 94.2, 5);
        assert!(*breaks.first().unwrap() <= 7.3);
        assert!(*breaks.last().unwrap() >= 94.2);
    }

    #[test]
    fn test_wilkinson_edge_cases() {
        let edge_cases = vec![
            (0.0, 100.0, 0, "zero count"),
            (100.0, 0.0, 5, "min > max"),
            (50.0, 50.0, 5, "min == max"),
            (f64::NAN, 100.0, 5, "NaN min"),
            (0.0, f64::INFINITY, 5, "infinite max"),
        ];
        for (min, max, n, desc) in edge_cases {
            assert!(
                wilkinson_extended(min, max, n).is_empty(),
                "wilkinson_extended with {} should be empty",
                desc
            );
        }
    }

    #[test]
    fn test_wilkinson_include_zero_variants() {
        // Positive-only range - should extend to include zero
        let breaks_pos = wilkinson_extended_include_zero(20.0, 80.0, 5);
        assert!(
            *breaks_pos.first().unwrap() <= 0.0,
            "Positive range should extend to include 0"
        );

        // Negative-only range - should extend to include zero
        let breaks_neg = wilkinson_extended_include_zero(-80.0, -20.0, 5);
        assert!(
            *breaks_neg.last().unwrap() >= 0.0,
            "Negative range should extend to include 0"
        );

        // Range already includes zero
        let breaks_both = wilkinson_extended_include_zero(-10.0, 90.0, 5);
        assert!(!breaks_both.is_empty());
        assert!(*breaks_both.first().unwrap() <= -10.0);
        assert!(*breaks_both.last().unwrap() >= 90.0);
    }

    #[test]
    fn test_pretty_breaks_simple_preserved() {
        let breaks = pretty_breaks_simple(0.0, 100.0, 5);
        assert!(!breaks.is_empty());
        assert!(breaks[0] <= 0.0);
        assert!(*breaks.last().unwrap() >= 100.0);
    }
}
