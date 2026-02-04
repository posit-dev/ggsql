//! Break calculation algorithms for scales
//!
//! Provides functions to calculate axis/legend break positions.

use crate::plot::ArrayElement;

/// Default number of breaks
pub const DEFAULT_BREAK_COUNT: usize = 5;

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
        // linear_breaks returns exactly n evenly-spaced breaks from min to max
        let breaks = linear_breaks(0.0, 100.0, 5);
        // step = 25, so we get: 0, 25, 50, 75, 100
        assert_eq!(breaks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }

    #[test]
    fn test_linear_breaks_single() {
        // Single break at midpoint
        let breaks = linear_breaks(0.0, 100.0, 1);
        assert_eq!(breaks, vec![50.0]);
    }

    #[test]
    fn test_linear_breaks_two() {
        // Two breaks at min and max
        let breaks = linear_breaks(0.0, 100.0, 2);
        assert_eq!(breaks, vec![0.0, 100.0]);
    }

    #[test]
    fn test_linear_breaks_zero_count() {
        let breaks = linear_breaks(0.0, 100.0, 0);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_linear_breaks_exact_coverage() {
        // Verify that breaks exactly cover min to max (no extension)
        let breaks = linear_breaks(10.0, 90.0, 5);
        // step = 20, so: 10, 30, 50, 70, 90
        assert_eq!(
            breaks.first().unwrap(),
            &10.0,
            "First break should be exactly min"
        );
        assert_eq!(
            breaks.last().unwrap(),
            &90.0,
            "Last break should be exactly max"
        );
        assert_eq!(breaks.len(), 5);
    }

    // =========================================================================
    // Integer Breaks Tests
    // =========================================================================

    #[test]
    fn test_integer_breaks_pretty_basic() {
        let breaks = integer_breaks(0.0, 100.0, 5, true);
        // Should produce nice round integers
        assert!(!breaks.is_empty());
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
        // Should cover the range
        assert!(*breaks.first().unwrap() <= 0.0);
        assert!(*breaks.last().unwrap() >= 100.0);
    }

    #[test]
    fn test_integer_breaks_evenly_spaced() {
        let breaks = integer_breaks(0.0, 100.0, 5, true);
        // All gaps should be equal (evenly spaced)
        if breaks.len() >= 2 {
            let step = breaks[1] - breaks[0];
            for i in 1..breaks.len() {
                let gap = breaks[i] - breaks[i - 1];
                assert!(
                    (gap - step).abs() < 0.01,
                    "Uneven spacing: gap {} != step {} at breaks {:?}",
                    gap,
                    step,
                    breaks
                );
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
    fn test_integer_breaks_no_rounding_problem() {
        // This tests the specific bug: linear breaks [0, 1.25, 2.5, 3.75, 5]
        // when rounded become [0, 1, 3, 4, 5] (uneven because 2.5 rounds to 3)
        let breaks = integer_breaks(0.0, 5.0, 5, false);
        // All breaks should be integers
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
        // Should be evenly spaced
        if breaks.len() >= 2 {
            let step = breaks[1] - breaks[0];
            for i in 1..breaks.len() {
                let gap = breaks[i] - breaks[i - 1];
                assert!(
                    (gap - step).abs() < 0.01,
                    "Uneven spacing (the rounding bug): {:?}",
                    breaks
                );
            }
        }
    }

    #[test]
    fn test_integer_breaks_large_range() {
        let breaks = integer_breaks(0.0, 1_000_000.0, 5, true);
        assert!(!breaks.is_empty());
        // Should have nice round numbers like 0, 200000, 400000, ...
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
    }

    #[test]
    fn test_integer_breaks_negative_range() {
        let breaks = integer_breaks(-50.0, 50.0, 5, true);
        assert!(!breaks.is_empty());
        for b in &breaks {
            assert_eq!(*b, b.round(), "Break {} should be integer", b);
        }
    }

    #[test]
    fn test_integer_breaks_edge_cases() {
        assert!(integer_breaks(0.0, 100.0, 0, true).is_empty());
        assert!(integer_breaks(100.0, 0.0, 5, true).is_empty()); // min > max
        assert!(integer_breaks(50.0, 50.0, 5, true).is_empty()); // min == max
        assert!(integer_breaks(f64::NAN, 100.0, 5, true).is_empty());
        assert!(integer_breaks(0.0, f64::INFINITY, 5, true).is_empty());
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
        // Linear in sqrt space with extension: sqrt(100)=10, steps of 2.5
        // linear_breaks now extends one step before and after
        // Squared back: ~6.25 (step before 0 gets clipped), 0, 6.25, 25, 56.25, 100, ~156.25
        // But negative values in sqrt space get clipped
        assert!(
            breaks.len() >= 5,
            "Should have at least 5 breaks, got {}",
            breaks.len()
        );
        // First break should be >= 0 (sqrt clips negatives)
        assert!(breaks.first().unwrap() >= &0.0);
        // Last break should be >= 100
        assert!(breaks.last().unwrap() >= &100.0);
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
        // Should use linear_breaks for exact coverage
        // step = 25, so: 0, 25, 50, 75, 100
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
        let interval = TemporalInterval::create_from_str("month").unwrap();
        assert_eq!(interval.count, 1);
        assert_eq!(interval.unit, TemporalUnit::Month);
    }

    #[test]
    fn test_temporal_interval_from_str_with_count() {
        let interval = TemporalInterval::create_from_str("2 months").unwrap();
        assert_eq!(interval.count, 2);
        assert_eq!(interval.unit, TemporalUnit::Month);

        let interval = TemporalInterval::create_from_str("3 days").unwrap();
        assert_eq!(interval.count, 3);
        assert_eq!(interval.unit, TemporalUnit::Day);
    }

    #[test]
    fn test_temporal_interval_all_units() {
        assert!(TemporalInterval::create_from_str("second").is_some());
        assert!(TemporalInterval::create_from_str("seconds").is_some());
        assert!(TemporalInterval::create_from_str("minute").is_some());
        assert!(TemporalInterval::create_from_str("hour").is_some());
        assert!(TemporalInterval::create_from_str("day").is_some());
        assert!(TemporalInterval::create_from_str("week").is_some());
        assert!(TemporalInterval::create_from_str("month").is_some());
        assert!(TemporalInterval::create_from_str("year").is_some());
    }

    #[test]
    fn test_temporal_interval_invalid() {
        assert!(TemporalInterval::create_from_str("invalid").is_none());
        assert!(TemporalInterval::create_from_str("foo bar baz").is_none());
        assert!(TemporalInterval::create_from_str("").is_none());
    }

    // =========================================================================
    // Temporal Date Breaks Tests
    // =========================================================================

    #[test]
    fn test_temporal_breaks_monthly() {
        // 2024-01-15 = day 19738, 2024-04-15 = day 19828
        let interval = TemporalInterval::create_from_str("month").unwrap();
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
        let interval = TemporalInterval::create_from_str("2 months").unwrap();
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
        let interval = TemporalInterval::create_from_str("year").unwrap();
        let breaks = temporal_breaks_date(18993, 20089, interval);
        assert!(breaks.contains(&"2022-01-01".to_string()));
        assert!(breaks.contains(&"2023-01-01".to_string()));
        assert!(breaks.contains(&"2024-01-01".to_string()));
    }

    #[test]
    fn test_temporal_breaks_weekly() {
        // Test weekly breaks
        // About 30 days
        let interval = TemporalInterval::create_from_str("week").unwrap();
        let breaks = temporal_breaks_date(19724, 19754, interval);
        assert!(!breaks.is_empty());
        // Should be multiple weeks
        assert!(breaks.len() >= 4);
    }

    // =========================================================================
    // Minor Breaks Linear Tests
    // =========================================================================

    #[test]
    fn test_minor_breaks_linear_basic() {
        let majors = vec![0.0, 10.0, 20.0];
        let minors = minor_breaks_linear(&majors, 1, None);
        // Expect: one midpoint per interval
        assert_eq!(minors, vec![5.0, 15.0]);
    }

    #[test]
    fn test_minor_breaks_linear_multiple() {
        let majors = vec![0.0, 10.0, 20.0];
        let minors = minor_breaks_linear(&majors, 4, None);
        // Expect: 4 minor breaks per interval
        assert_eq!(minors.len(), 8); // 4 * 2 intervals
        assert!(minors.contains(&2.0));
        assert!(minors.contains(&4.0));
        assert!(minors.contains(&6.0));
        assert!(minors.contains(&8.0));
        assert!(minors.contains(&12.0));
        assert!(minors.contains(&14.0));
        assert!(minors.contains(&16.0));
        assert!(minors.contains(&18.0));
    }

    #[test]
    fn test_minor_breaks_linear_with_extension() {
        let majors = vec![20.0, 40.0, 60.0];
        let minors = minor_breaks_linear(&majors, 1, Some((0.0, 80.0)));
        // Expect: extends before 20 and after 60
        assert!(minors.contains(&10.0)); // Before first major
        assert!(minors.contains(&30.0)); // Between 20-40
        assert!(minors.contains(&50.0)); // Between 40-60
        assert!(minors.contains(&70.0)); // After last major
    }

    #[test]
    fn test_minor_breaks_linear_empty_for_single_major() {
        let majors = vec![10.0];
        let minors = minor_breaks_linear(&majors, 1, None);
        assert!(minors.is_empty());
    }

    #[test]
    fn test_minor_breaks_linear_empty_for_zero_count() {
        let majors = vec![0.0, 10.0, 20.0];
        let minors = minor_breaks_linear(&majors, 0, None);
        assert!(minors.is_empty());
    }

    // =========================================================================
    // Minor Breaks Log Tests
    // =========================================================================

    #[test]
    fn test_minor_breaks_log_basic() {
        let majors = vec![1.0, 10.0, 100.0];
        let minors = minor_breaks_log(&majors, 8, 10.0, None);
        // Expect: 8 breaks per decade, evenly spaced in log space
        assert_eq!(minors.len(), 16); // 8 between 1-10, 8 between 10-100
    }

    #[test]
    fn test_minor_breaks_log_single_minor() {
        let majors = vec![1.0, 10.0, 100.0];
        let minors = minor_breaks_log(&majors, 1, 10.0, None);
        // Expect: one midpoint per decade in log space
        // Between 1 and 10: sqrt(1 * 10) ≈ 3.16
        // Between 10 and 100: sqrt(10 * 100) ≈ 31.6
        assert_eq!(minors.len(), 2);
        // Check geometric mean relationship
        assert!((minors[0] - (1.0_f64 * 10.0).sqrt()).abs() < 0.01);
        assert!((minors[1] - (10.0_f64 * 100.0).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_minor_breaks_log_with_extension() {
        let majors = vec![10.0, 100.0];
        let minors = minor_breaks_log(&majors, 8, 10.0, Some((1.0, 1000.0)));
        // Should have minors in [1, 10), (10, 100), and (100, 1000]
        assert_eq!(minors.len(), 24); // 8 per decade × 3 decades
    }

    #[test]
    fn test_minor_breaks_log_filters_negative() {
        let majors = vec![-10.0, 1.0, 10.0, 100.0];
        let minors = minor_breaks_log(&majors, 1, 10.0, None);
        // Should only use positive majors
        assert!(minors.iter().all(|&x| x > 0.0));
    }

    // =========================================================================
    // Minor Breaks Sqrt Tests
    // =========================================================================

    #[test]
    fn test_minor_breaks_sqrt_basic() {
        let majors = vec![0.0, 25.0, 100.0];
        let minors = minor_breaks_sqrt(&majors, 1, None);
        // sqrt(0)=0, sqrt(25)=5, sqrt(100)=10
        // Midpoints in sqrt space: 2.5, 7.5
        // Squared back: 6.25, 56.25
        assert_eq!(minors.len(), 2);
        assert!((minors[0] - 6.25).abs() < 0.01);
        assert!((minors[1] - 56.25).abs() < 0.01);
    }

    #[test]
    fn test_minor_breaks_sqrt_with_extension() {
        let majors = vec![25.0, 100.0];
        let minors = minor_breaks_sqrt(&majors, 1, Some((0.0, 225.0)));
        // sqrt(0)=0, sqrt(25)=5, sqrt(100)=10, sqrt(225)=15
        // Should extend before 25 and after 100
        assert!(minors.len() >= 2);
    }

    #[test]
    fn test_minor_breaks_sqrt_filters_negative() {
        let majors = vec![-10.0, 0.0, 25.0, 100.0];
        let minors = minor_breaks_sqrt(&majors, 1, None);
        // Should only use non-negative majors
        assert!(minors.iter().all(|&x| x >= 0.0));
    }

    // =========================================================================
    // Minor Breaks Symlog Tests
    // =========================================================================

    #[test]
    fn test_minor_breaks_symlog_basic() {
        let majors = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
        let minors = minor_breaks_symlog(&majors, 1, None);
        // Should have minors between each pair of majors
        assert_eq!(minors.len(), 4);
    }

    #[test]
    fn test_minor_breaks_symlog_crosses_zero() {
        let majors = vec![-10.0, 10.0];
        let minors = minor_breaks_symlog(&majors, 1, None);
        // Midpoint in asinh space should be near 0
        assert_eq!(minors.len(), 1);
        assert!(minors[0].abs() < 1.0); // Should be near zero
    }

    #[test]
    fn test_minor_breaks_symlog_with_extension() {
        let majors = vec![0.0, 100.0];
        let minors = minor_breaks_symlog(&majors, 1, Some((-100.0, 200.0)));
        // Should extend into negative and beyond 100
        assert!(minors.len() >= 2);
    }

    // =========================================================================
    // Trim Breaks Tests
    // =========================================================================

    #[test]
    fn test_trim_breaks() {
        let breaks = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let trimmed = trim_breaks(&breaks, (10.0, 25.0));
        assert_eq!(trimmed, vec![10.0, 15.0, 20.0, 25.0]);
    }

    #[test]
    fn test_trim_breaks_empty() {
        let breaks = vec![5.0, 10.0, 15.0];
        let trimmed = trim_breaks(&breaks, (20.0, 30.0));
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_trim_breaks_all_inside() {
        let breaks = vec![15.0, 20.0, 25.0];
        let trimmed = trim_breaks(&breaks, (10.0, 30.0));
        assert_eq!(trimmed, breaks);
    }

    // =========================================================================
    // Trim Temporal Breaks Tests
    // =========================================================================

    #[test]
    fn test_trim_temporal_breaks() {
        let breaks = vec![
            "2024-01-01".to_string(),
            "2024-02-01".to_string(),
            "2024-03-01".to_string(),
        ];
        let trimmed = trim_temporal_breaks(&breaks, ("2024-01-15", "2024-02-15"));
        assert_eq!(trimmed, vec!["2024-02-01".to_string()]);
    }

    #[test]
    fn test_trim_temporal_breaks_all_inside() {
        let breaks = vec!["2024-02-01".to_string(), "2024-02-15".to_string()];
        let trimmed = trim_temporal_breaks(&breaks, ("2024-01-01", "2024-03-01"));
        assert_eq!(trimmed.len(), 2);
    }

    // =========================================================================
    // Derive Minor Interval Tests
    // =========================================================================

    #[test]
    fn test_derive_minor_interval() {
        assert_eq!(derive_minor_interval("year"), "3 months");
        assert_eq!(derive_minor_interval("3 months"), "month"); // quarter
        assert_eq!(derive_minor_interval("month"), "week");
        assert_eq!(derive_minor_interval("week"), "day");
        assert_eq!(derive_minor_interval("day"), "6 hours");
        assert_eq!(derive_minor_interval("hour"), "15 minutes");
        assert_eq!(derive_minor_interval("minute"), "15 seconds");
    }

    #[test]
    fn test_derive_minor_interval_invalid() {
        // Invalid interval falls back to "day"
        assert_eq!(derive_minor_interval("invalid"), "day");
    }

    // =========================================================================
    // Temporal Minor Breaks Date Tests
    // =========================================================================

    #[test]
    fn test_temporal_minor_breaks_date_auto() {
        let majors = vec![
            "2024-01-01".to_string(),
            "2024-02-01".to_string(),
            "2024-03-01".to_string(),
        ];
        let minors = temporal_minor_breaks_date(&majors, "month", MinorBreakSpec::Auto, None);
        // Auto should derive "week" from "month"
        // Should have weekly dates within January and February
        assert!(!minors.is_empty());
        assert!(minors.iter().any(|d| d.starts_with("2024-01")));
        assert!(minors.iter().any(|d| d.starts_with("2024-02")));
    }

    #[test]
    fn test_temporal_minor_breaks_date_by_count() {
        let majors = vec!["2024-01-01".to_string(), "2024-02-01".to_string()];
        let minors = temporal_minor_breaks_date(&majors, "month", MinorBreakSpec::Count(3), None);
        // Count(3) means 3 minor breaks per month (dividing by 4)
        // ~7-8 days per minor interval
        assert!(!minors.is_empty());
    }

    #[test]
    fn test_temporal_minor_breaks_date_by_interval() {
        let majors = vec!["2024-01-01".to_string(), "2024-02-01".to_string()];
        let minors = temporal_minor_breaks_date(
            &majors,
            "month",
            MinorBreakSpec::Interval("week".to_string()),
            None,
        );
        // Should have weekly dates within January
        assert!(!minors.is_empty());
        // January has about 4 weeks
        assert!(minors.len() >= 3);
    }

    #[test]
    fn test_temporal_minor_breaks_date_with_extension() {
        let majors = vec!["2024-02-01".to_string(), "2024-03-01".to_string()];
        let minors = temporal_minor_breaks_date(
            &majors,
            "month",
            MinorBreakSpec::Interval("week".to_string()),
            Some(("2024-01-01", "2024-04-01")),
        );
        // Should extend weekly breaks into January and March
        assert!(minors.iter().any(|d| d.starts_with("2024-01"))); // Before first major
        assert!(minors.iter().any(|d| d.starts_with("2024-03"))); // After last major
    }

    #[test]
    fn test_temporal_minor_breaks_date_empty_for_single() {
        let majors = vec!["2024-01-01".to_string()];
        let minors = temporal_minor_breaks_date(&majors, "month", MinorBreakSpec::Auto, None);
        assert!(minors.is_empty());
    }

    // =========================================================================
    // MinorBreakSpec Tests
    // =========================================================================

    #[test]
    fn test_minor_break_spec_default() {
        let spec = MinorBreakSpec::default();
        assert_eq!(spec, MinorBreakSpec::Auto);
    }

    #[test]
    fn test_minor_break_spec_variants() {
        let auto = MinorBreakSpec::Auto;
        let count = MinorBreakSpec::Count(4);
        let interval = MinorBreakSpec::Interval("week".to_string());

        assert_eq!(auto, MinorBreakSpec::Auto);
        assert_eq!(count, MinorBreakSpec::Count(4));
        assert_eq!(interval, MinorBreakSpec::Interval("week".to_string()));
    }

    // =========================================================================
    // Wilkinson Extended Tests
    // =========================================================================

    #[test]
    fn test_wilkinson_basic() {
        let breaks = wilkinson_extended(0.0, 100.0, 5);
        assert!(!breaks.is_empty());
        assert!(breaks.len() >= 3 && breaks.len() <= 10);
        // Should produce nice round numbers
        assert!(breaks.iter().all(|&b| b == b.round()));
    }

    #[test]
    fn test_wilkinson_prefers_nice_numbers() {
        let breaks = wilkinson_extended(0.0, 97.0, 5);
        // Should prefer 0, 20, 40, 60, 80, 100 over something like 0, 24.25, ...
        for b in &breaks {
            let normalized = b / 10.0;
            // Check divisible by nice numbers (1, 2, 2.5, 5, 10)
            let is_nice = normalized.fract() == 0.0
                || (normalized * 2.0).fract() == 0.0
                || (normalized * 4.0).fract() == 0.0;
            assert!(is_nice, "Break {} is not a nice number", b);
        }
    }

    #[test]
    fn test_wilkinson_covers_data() {
        let breaks = wilkinson_extended(7.3, 94.2, 5);
        assert!(*breaks.first().unwrap() <= 7.3);
        assert!(*breaks.last().unwrap() >= 94.2);
    }

    #[test]
    fn test_wilkinson_penguin_count_scenario() {
        // Simulate bar chart of penguins by species:
        // min=0 (explicitly set), max≈152 (max species count)
        let breaks = wilkinson_extended(0.0, 152.0, 5);
        eprintln!("Breaks for [0, 152] with target=5: {:?}", breaks);
        assert!(
            breaks.len() >= 4,
            "Expected at least 4 breaks for [0, 152] but got {:?}",
            breaks
        );
    }

    #[test]
    fn test_wilkinson_explicit_min_preserved() {
        // When user sets explicit min (FROM [0, null]), only max gets expanded.
        // This test verifies the Wilkinson algorithm produces good breaks
        // when min=0 is preserved and only max is expanded.
        //
        // Scenario: bar chart with max count ~152, after selective expansion → [0, ~160]
        let breaks = wilkinson_extended(0.0, 159.6, 5);
        assert!(
            breaks.len() >= 4,
            "Expected at least 4 breaks for [0, 159.6] but got {:?}",
            breaks
        );
        // Should start at 0 (preserving user's explicit min)
        assert!(
            breaks[0] <= 0.0,
            "First break should be <= 0, got {}",
            breaks[0]
        );
    }

    #[test]
    fn test_wilkinson_small_range() {
        let breaks = wilkinson_extended(0.1, 0.9, 5);
        assert!(!breaks.is_empty());
        // Should handle fractional ranges
    }

    #[test]
    fn test_wilkinson_large_range() {
        let breaks = wilkinson_extended(0.0, 1_000_000.0, 5);
        assert!(!breaks.is_empty());
        // Should produce nice round numbers in millions
    }

    #[test]
    fn test_wilkinson_negative_range() {
        let breaks = wilkinson_extended(-50.0, 50.0, 5);
        assert!(!breaks.is_empty());
        // Should likely include zero or values near zero
        assert!(breaks.iter().any(|&b| b.abs() < 20.0));
    }

    #[test]
    fn test_wilkinson_edge_cases() {
        assert!(wilkinson_extended(0.0, 100.0, 0).is_empty());
        assert!(wilkinson_extended(100.0, 0.0, 5).is_empty()); // min > max
        assert!(wilkinson_extended(50.0, 50.0, 5).is_empty()); // min == max
        assert!(wilkinson_extended(f64::NAN, 100.0, 5).is_empty());
        assert!(wilkinson_extended(0.0, f64::INFINITY, 5).is_empty());
    }

    #[test]
    fn test_wilkinson_vs_simple_quality() {
        // Test case where Wilkinson should produce results
        let wilkinson = wilkinson_extended(0.0, 97.0, 5);
        let simple = pretty_breaks_simple(0.0, 97.0, 5);

        // Both should produce non-empty results
        assert!(!wilkinson.is_empty());
        assert!(!simple.is_empty());

        // Wilkinson should get reasonably close to target count
        assert!(wilkinson.len() >= 3 && wilkinson.len() <= 10);
    }

    #[test]
    fn test_wilkinson_include_zero() {
        // Test the include_zero variant
        let breaks = wilkinson_extended_include_zero(20.0, 80.0, 5);
        // Should extend to include zero
        assert!(*breaks.first().unwrap() <= 0.0);
    }

    #[test]
    fn test_wilkinson_include_zero_already_in_range() {
        // When zero is already in range, should behave like regular wilkinson
        let breaks = wilkinson_extended_include_zero(-10.0, 90.0, 5);
        assert!(!breaks.is_empty());
        // Should cover the range
        assert!(*breaks.first().unwrap() <= -10.0);
        assert!(*breaks.last().unwrap() >= 90.0);
    }

    #[test]
    fn test_wilkinson_include_zero_negative_range() {
        // Test with negative-only range
        let breaks = wilkinson_extended_include_zero(-80.0, -20.0, 5);
        // Should extend to include zero
        assert!(*breaks.last().unwrap() >= 0.0);
    }

    #[test]
    fn test_pretty_breaks_simple_preserved() {
        // Ensure the simple algorithm is still available and works
        let breaks = pretty_breaks_simple(0.0, 100.0, 5);
        assert!(!breaks.is_empty());
        assert!(breaks[0] <= 0.0);
        assert!(*breaks.last().unwrap() >= 100.0);
    }
}
