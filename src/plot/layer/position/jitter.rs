//! Jitter position adjustment
//!
//! Adds random displacement to elements to avoid overplotting.
//! Jitter automatically detects which axes are discrete and applies
//! jitter to those axes:
//! - If only pos1 is discrete → jitter horizontally (pos1offset)
//! - If only pos2 is discrete → jitter vertically (pos2offset)
//! - If both are discrete → jitter in both directions
//!
//! When `dodge=true` (default), jitter first applies dodge positioning to separate
//! groups, then applies random jitter within the reduced width of each group's space.
//!
//! The `distribution` parameter controls the shape of the jitter:
//! - `uniform` (default): uniform random distribution across the width
//! - `normal`: normal/Gaussian distribution with ~95% of points within the width

use super::{
    compute_dodge_offsets, compute_group_indices, is_continuous_scale, Layer, PositionTrait,
    PositionType,
};
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition, ParameterValue};
use crate::{naming, DataFrame, GgsqlError, Plot, Result};
use polars::prelude::*;
use rand::Rng;

/// Valid distribution types for jitter position
const DISTRIBUTION_VALUES: &[&str] = &["uniform", "normal", "density", "intensity"];

/// Jitter distribution type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JitterDistribution {
    Uniform,
    Normal,
    /// Per-group normalized density (area under curve = 1).
    /// Narrow distributions have higher peaks than wide distributions.
    Density,
    /// Count-weighted density (not normalized by group size).
    /// Groups with more observations have higher peaks.
    /// Both density and intensity use global max normalization.
    Intensity,
}

impl JitterDistribution {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "normal" | "gaussian" => Self::Normal,
            "density" => Self::Density,
            "intensity" => Self::Intensity,
            _ => Self::Uniform,
        }
    }

    /// Generate a random jitter value within the given width.
    ///
    /// For uniform: values are in [-width/2, width/2]
    /// For normal: σ = width/4, so ~95% of values fall within [-width/2, width/2]
    /// For density/intensity: not applicable, density scaling is handled separately
    fn sample<R: Rng>(&self, rng: &mut R, width: f64) -> f64 {
        match self {
            Self::Uniform | Self::Density | Self::Intensity => (rng.gen::<f64>() - 0.5) * width,
            Self::Normal => {
                // Box-Muller transform for normal distribution
                // Use σ = width/4 so 95% of values fall within ±2σ = ±width/2
                let sigma = width / 4.0;
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                // Avoid log(0) by ensuring u1 > 0
                let u1 = if u1 == 0.0 { f64::MIN_POSITIVE } else { u1 };
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * sigma
            }
        }
    }
}

// ============================================================================
// Density estimation for density-based jitter
// ============================================================================

/// Compute Silverman's rule of thumb bandwidth for KDE.
///
/// Uses the formula: h = 0.9 * adjust * min(σ, IQR/1.34) * n^(-0.2)
///
/// This matches the bandwidth calculation used by the density and violin geoms,
/// ensuring consistent density estimates when using `distribution => 'density'`
/// with a violin layer.
fn silverman_bandwidth(values: &[f64], adjust: f64) -> f64 {
    let n = values.len() as f64;
    if n <= 1.0 {
        return 1.0;
    }

    // Compute mean and standard deviation (population stddev to match SQL STDDEV)
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Compute IQR (interquartile range) using linear interpolation
    // This matches SQL's QUANTILE_CONT behavior
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = quantile_cont(&sorted, 0.25);
    let q3 = quantile_cont(&sorted, 0.75);
    let iqr = q3 - q1;

    // Silverman's rule: 0.9 * adjust * min(σ, IQR/1.34) * n^(-0.2)
    let scale = if iqr > 0.0 {
        std_dev.min(iqr / 1.34)
    } else {
        std_dev
    };

    if scale == 0.0 {
        return 1.0; // Fallback for constant data
    }

    0.9 * adjust * scale * n.powf(-0.2)
}

/// Compute continuous quantile using linear interpolation.
/// Matches SQL QUANTILE_CONT behavior.
fn quantile_cont(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let n = sorted.len() as f64;
    let idx = p * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;

    if lo == hi || hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Compute density at each point using Gaussian KDE (normalized PDF).
///
/// For each point xi, computes: f(xi) = (1/nh) * Σ K((xi - xj) / h)
/// where K is the Gaussian kernel.
///
/// This produces a normalized PDF where the area under the curve equals 1.
/// Narrow distributions will have higher peaks than wide distributions.
fn compute_densities(values: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = values.len() as f64;
    compute_intensities(values, bandwidth)
        .into_iter()
        .map(|i| i / n)
        .collect()
}

/// Compute intensity at each point using Gaussian KDE (count-weighted, not normalized).
///
/// For each point xi, computes: f(xi) = (1/h) * Σ K((xi - xj) / h)
/// where K is the Gaussian kernel.
///
/// Unlike `compute_densities`, this does NOT divide by n, so groups with more
/// observations will have higher values. This makes the width proportional to
/// the number of data points.
fn compute_intensities(values: &[f64], bandwidth: f64) -> Vec<f64> {
    let norm_factor = 1.0 / (bandwidth * (2.0 * std::f64::consts::PI).sqrt());

    values
        .iter()
        .map(|&xi| {
            // Sum kernel contributions from all points
            let intensity: f64 = values
                .iter()
                .map(|&xj| {
                    let u = (xi - xj) / bandwidth;
                    (-0.5 * u * u).exp()
                })
                .sum();
            intensity * norm_factor
        })
        .collect()
}

/// Compute density/intensity scales for grouped data with global normalization.
///
/// When groups exist, compute density/intensity separately per group, but normalize
/// using the global max across ALL groups. This preserves relative differences:
/// - For density: narrow distributions appear wider (higher peaks)
/// - For intensity: groups with more data appear wider
///
/// # Arguments
/// * `values` - All values from the continuous axis
/// * `group_indices` - Group index for each value
/// * `n_groups` - Number of distinct groups
/// * `explicit_bandwidth` - Optional explicit bandwidth (overrides Silverman's rule)
/// * `adjust` - Bandwidth adjustment multiplier
/// * `use_intensity` - If true, use intensity (count-weighted); if false, use density (normalized PDF)
fn compute_grouped_scales(
    values: &[f64],
    group_indices: &[usize],
    n_groups: usize,
    explicit_bandwidth: Option<f64>,
    adjust: f64,
    use_intensity: bool,
) -> Vec<f64> {
    // Group values by their group index
    let mut grouped_values: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
    let mut grouped_original_indices: Vec<Vec<usize>> = vec![Vec::new(); n_groups];

    for (i, (&value, &group_idx)) in values.iter().zip(group_indices.iter()).enumerate() {
        grouped_values[group_idx].push(value);
        grouped_original_indices[group_idx].push(i);
    }

    // Compute raw density/intensity for each group (before normalization)
    let mut all_raw_values = vec![0.0; values.len()];

    for group_idx in 0..n_groups {
        let group_vals = &grouped_values[group_idx];
        if group_vals.is_empty() {
            continue;
        }

        // Use explicit bandwidth if provided, otherwise compute per-group using Silverman's rule
        // This matches how violin/density compute bandwidth per group
        let bandwidth = explicit_bandwidth
            .map(|bw| bw * adjust)
            .unwrap_or_else(|| silverman_bandwidth(group_vals, adjust));

        // Compute raw values using appropriate formula
        let raw = if use_intensity {
            compute_intensities(group_vals, bandwidth)
        } else {
            compute_densities(group_vals, bandwidth)
        };

        // Map back to original indices
        for (within_group_idx, &original_idx) in
            grouped_original_indices[group_idx].iter().enumerate()
        {
            all_raw_values[original_idx] = raw[within_group_idx];
        }
    }

    // Global normalization: divide by max across ALL groups
    let global_max = all_raw_values.iter().fold(0.0_f64, |a, &b| a.max(b));
    if global_max > 0.0 {
        all_raw_values.iter().map(|v| v / global_max).collect()
    } else {
        vec![1.0; values.len()]
    }
}

/// Jitter position - add random displacement
#[derive(Debug, Clone, Copy)]
pub struct Jitter;

impl std::fmt::Display for Jitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jitter")
    }
}

impl PositionTrait for Jitter {
    fn position_type(&self) -> PositionType {
        PositionType::Jitter
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "width",
                default: DefaultParamValue::Number(0.9),
                constraint: ParamConstraint::number_range(0.0, 1.0),
            },
            ParamDefinition {
                name: "dodge",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
            ParamDefinition {
                name: "distribution",
                default: DefaultParamValue::String("uniform"),
                constraint: ParamConstraint::string_option(DISTRIBUTION_VALUES),
            },
            // Density distribution parameters (match violin/density geoms)
            ParamDefinition {
                name: "bandwidth",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
            ParamDefinition {
                name: "adjust",
                default: DefaultParamValue::Number(1.0),
                constraint: ParamConstraint::number_min_exclusive(0.0),
            },
        ];
        PARAMS
    }

    fn creates_pos1offset(&self) -> bool {
        true
    }

    fn creates_pos2offset(&self) -> bool {
        true
    }

    fn apply_adjustment(
        &self,
        df: DataFrame,
        layer: &Layer,
        spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)> {
        Ok((apply_jitter(df, layer, spec)?, None))
    }
}

/// Compute density/intensity scales for density-based jitter distribution.
///
/// Returns a Vec of scale factors (0 to 1) for each row, where higher values
/// indicate denser regions that should have wider jitter spread.
///
/// For density distribution: scales are normalized per-group (area = 1 per group).
/// For intensity distribution: scales are normalized globally (larger groups have higher peaks).
fn compute_density_scales(
    df: &DataFrame,
    layer: &Layer,
    pos1_continuous: bool,
    use_intensity: bool,
    dodge: bool,
    explicit_bandwidth: Option<f64>,
    adjust: f64,
) -> Result<Option<Vec<f64>>> {
    // Identify axes
    let continuous_col = if pos1_continuous { "pos1" } else { "pos2" };
    let discrete_col = if pos1_continuous { "pos2" } else { "pos1" };
    let continuous_col_name = naming::aesthetic_column(continuous_col);
    let discrete_col_name = naming::aesthetic_column(discrete_col);

    // Extract values from the continuous axis
    let values: Vec<f64> = df
        .column(&continuous_col_name)
        .map_err(|_| {
            GgsqlError::InternalError(format!(
                "Missing {} column for density jitter",
                continuous_col
            ))
        })?
        .cast(&DataType::Float64)
        .map_err(|_| {
            GgsqlError::InternalError(format!(
                "{} must be numeric for density jitter",
                continuous_col
            ))
        })?
        .f64()
        .map_err(|_| {
            GgsqlError::InternalError(format!(
                "{} must be numeric for density jitter",
                continuous_col
            ))
        })?
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    // Build density grouping columns: discrete axis + relevant partition_by columns
    // This matches how violin computes density per group
    let mut density_group_cols = vec![discrete_col_name.clone()];
    for col in &layer.partition_by {
        if density_group_cols.contains(col) {
            continue;
        }
        // When dodge is false, only include facet variables (not color/fill groups)
        // Facet variables have predictable names: __ggsql_aes_facet1__, __ggsql_aes_facet2__
        if !dodge && !col.contains("_facet") {
            continue;
        }
        density_group_cols.push(col.clone());
    }

    // Compute density grouping
    let density_group_info = compute_group_indices(df, &density_group_cols)?;

    // Compute density/intensity scales per group with global normalization
    if let Some(info) = density_group_info {
        Ok(Some(compute_grouped_scales(
            &values,
            &info.indices,
            info.n_groups,
            explicit_bandwidth,
            adjust,
            use_intensity,
        )))
    } else {
        // Single group - compute global density/intensity
        let bandwidth = explicit_bandwidth
            .map(|bw| bw * adjust)
            .unwrap_or_else(|| silverman_bandwidth(&values, adjust));
        let raw = if use_intensity {
            compute_intensities(&values, bandwidth)
        } else {
            compute_densities(&values, bandwidth)
        };
        // Normalize to [0, 1]
        let max_val = raw.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_val > 0.0 {
            Ok(Some(raw.iter().map(|v| v / max_val).collect()))
        } else {
            Ok(Some(vec![1.0; values.len()]))
        }
    }
}

/// Apply jitter position adjustment.
///
/// Automatically detects which axes are discrete and applies jitter accordingly:
/// - Discrete pos1 → creates pos1offset column
/// - Discrete pos2 → creates pos2offset column
/// - Both discrete → creates both offset columns
/// - Neither discrete → returns unchanged (no jitter applied)
///
/// When `dodge=true` (default), groups are first dodged to separate positions,
/// then jitter is applied within each group's reduced space.
///
/// The width parameter controls the total jitter range. When dodging is applied,
/// the effective jitter range is reduced by the number of groups.
///
/// The `distribution` parameter controls the jitter shape:
/// - `uniform`: uniform random distribution across the width (default)
/// - `normal`: Gaussian distribution with ~95% of points within the width
/// - `density`: scales jitter width by local density (requires exactly one continuous axis)
fn apply_jitter(df: DataFrame, layer: &Layer, spec: &Plot) -> Result<DataFrame> {
    // Check which axes should be jittered (discrete axes)
    // Since create_missing_scales_post_stat() runs before position adjustments,
    // scale types are always known, so we use explicit discrete checks.
    let jitter_pos1 = is_continuous_scale(spec, "pos1") == Some(false);
    let jitter_pos2 = is_continuous_scale(spec, "pos2") == Some(false);

    // Get width parameter (default 0.9)
    let width = layer
        .parameters
        .get("width")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0.9);

    // Get dodge parameter (default true)
    let dodge = layer
        .parameters
        .get("dodge")
        .and_then(|v| match v {
            ParameterValue::Boolean(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(true);

    // Get distribution parameter (default "uniform")
    let distribution = layer
        .parameters
        .get("distribution")
        .and_then(|v| match v {
            ParameterValue::String(s) => Some(JitterDistribution::from_str(s)),
            _ => None,
        })
        .unwrap_or(JitterDistribution::Uniform);

    // Density/intensity distribution validation: requires exactly one continuous axis
    // (one discrete axis to jitter along, one continuous axis for density)
    let pos1_continuous = !jitter_pos1;
    let pos2_continuous = !jitter_pos2;
    let use_density_scaling = distribution == JitterDistribution::Density
        || distribution == JitterDistribution::Intensity;
    if use_density_scaling && (pos1_continuous == pos2_continuous) {
        let dist_name = if distribution == JitterDistribution::Intensity {
            "intensity"
        } else {
            "density"
        };
        return Err(GgsqlError::ValidationError(format!(
            "Jitter distribution '{}' requires exactly one continuous axis",
            dist_name
        )));
    }

    let mut rng = rand::thread_rng();
    let n_rows = df.height();

    // Compute group info for dodge-first behavior
    let group_info = if dodge {
        compute_group_indices(&df, &layer.partition_by)?
    } else {
        None
    };

    // Extract group info for dodge behavior
    let (n_groups, group_indices) = match &group_info {
        Some(info) if info.n_groups > 1 => (info.n_groups, Some(&info.indices)),
        _ => (1, None),
    };

    // Get density-specific parameters (match violin/density geoms)
    let explicit_bandwidth = layer.parameters.get("bandwidth").and_then(|v| match v {
        ParameterValue::Number(n) => Some(*n),
        _ => None,
    });

    let adjust = layer
        .parameters
        .get("adjust")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(1.0);

    // Compute density scales if using density/intensity distribution
    let use_intensity = distribution == JitterDistribution::Intensity;
    let density_scales = if use_density_scaling {
        compute_density_scales(
            &df,
            layer,
            pos1_continuous,
            use_intensity,
            dodge,
            explicit_bandwidth,
            adjust,
        )?
    } else {
        None
    };

    let pos1offset_col = naming::aesthetic_column("pos1offset");
    let pos2offset_col = naming::aesthetic_column("pos2offset");

    let mut result = df.lazy();

    // Compute dodge centers if we have groups to dodge
    let dodge_offsets = if n_groups > 1 {
        let indices = group_indices.unwrap();
        Some(compute_dodge_offsets(
            indices,
            n_groups,
            width,
            jitter_pos1,
            jitter_pos2,
        ))
    } else {
        None
    };

    // Helper to generate jitter with optional density scaling
    let make_jitter =
        |rng: &mut rand::rngs::ThreadRng, jitter_width: f64, count: usize| -> Vec<f64> {
            (0..count)
                .map(|i| {
                    let jitter = distribution.sample(rng, jitter_width);
                    if let Some(ref scales) = density_scales {
                        jitter * scales[i]
                    } else {
                        jitter
                    }
                })
                .collect()
        };

    // Add pos1offset if pos1 is discrete
    if jitter_pos1 {
        let jitter_width = dodge_offsets
            .as_ref()
            .map(|d| d.adjusted_width)
            .unwrap_or(width);
        let jitters = make_jitter(&mut rng, jitter_width, n_rows);

        let offsets: Vec<f64> = if let Some(ref dodge) = dodge_offsets {
            if let Some(ref centers) = dodge.pos1 {
                // Dodge + jitter
                centers
                    .iter()
                    .zip(jitters.iter())
                    .map(|(c, j)| c + j)
                    .collect()
            } else {
                jitters
            }
        } else {
            jitters
        };

        result = result.with_column(
            lit(Series::new(pos1offset_col.clone().into(), offsets)).alias(&pos1offset_col),
        );
    }

    // Add pos2offset if pos2 is discrete
    if jitter_pos2 {
        let jitter_width = dodge_offsets
            .as_ref()
            .map(|d| d.adjusted_width)
            .unwrap_or(width);
        let jitters = make_jitter(&mut rng, jitter_width, n_rows);

        let offsets: Vec<f64> = if let Some(ref dodge) = dodge_offsets {
            if let Some(ref centers) = dodge.pos2 {
                // Dodge + jitter
                centers
                    .iter()
                    .zip(jitters.iter())
                    .map(|(c, j)| c + j)
                    .collect()
            } else {
                jitters
            }
        } else {
            jitters
        };

        result = result.with_column(
            lit(Series::new(pos2offset_col.clone().into(), offsets)).alias(&pos2offset_col),
        );
    }

    result
        .collect()
        .map_err(|e| GgsqlError::InternalError(format!("Jitter position adjustment failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::layer::Geom;
    use crate::plot::{AestheticValue, Mappings, Scale, ScaleType};

    fn make_test_df() -> DataFrame {
        df! {
            "__ggsql_aes_pos1__" => ["A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => [10.0, 20.0, 15.0, 25.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "Y", "X", "Y"],
        }
        .unwrap()
    }

    fn make_test_layer() -> Layer {
        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];
        layer
    }

    fn make_continuous_scale(aesthetic: &str) -> Scale {
        let mut scale = Scale::new(aesthetic);
        scale.scale_type = Some(ScaleType::continuous());
        scale
    }

    fn make_discrete_scale(aesthetic: &str) -> Scale {
        let mut scale = Scale::new(aesthetic);
        scale.scale_type = Some(ScaleType::discrete());
        scale
    }

    #[test]
    fn test_jitter_horizontal_only_with_dodge() {
        // When pos1 is discrete and pos2 is continuous, only pos1offset is created
        // With default dodge=true and 2 groups, offsets should be dodge + jitter
        let jitter = Jitter;
        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos1 as discrete and pos2 as continuous via scales
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, width) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify pos1offset column was created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_ok(),
            "pos1offset column should be created"
        );
        // Verify pos2offset column was NOT created
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_err(),
            "pos2offset column should NOT be created when pos2 is continuous"
        );

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // With default width 0.9 and 2 groups (dodge=true):
        // effective_width = 0.9 / 2 = 0.45
        // Group X center: -0.225, Group Y center: +0.225
        // With jitter in range [-0.225, +0.225] around each center
        // Total range: [-0.45, 0.45]
        for &v in &offsets {
            assert!(
                (-0.45..=0.45).contains(&v),
                "Jitter+dodge offset {} should be in range [-0.45, 0.45]",
                v
            );
        }

        // Verify no adjusted_width is returned for jitter
        assert!(width.is_none());
    }

    #[test]
    fn test_jitter_horizontal_no_dodge() {
        // With dodge=false, should behave like classic jitter
        let jitter = Jitter;
        let df = make_test_df();
        let mut layer = make_test_layer();
        layer
            .parameters
            .insert("dodge".to_string(), ParameterValue::Boolean(false));

        // Mark pos1 as discrete and pos2 as continuous via scales
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // With dodge=false and width 0.9, pure jitter in range [-0.45, 0.45]
        for &v in &offsets {
            assert!(
                (-0.45..=0.45).contains(&v),
                "Pure jitter offset {} should be in range [-0.45, 0.45]",
                v
            );
        }
    }

    #[test]
    fn test_jitter_vertical_only() {
        // When pos1 is continuous and pos2 is discrete, only pos2offset is created
        let jitter = Jitter;
        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos1 as continuous and pos2 as discrete via scales
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify pos1offset column was NOT created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_err(),
            "pos1offset column should NOT be created when pos1 is continuous"
        );
        // Verify pos2offset column was created
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_ok(),
            "pos2offset column should be created"
        );

        let offset = result
            .column("__ggsql_aes_pos2offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // With default width 0.9 and 2 groups (dodge=true), effective range is [-0.45, 0.45]
        for &v in &offsets {
            assert!(
                (-0.45..=0.45).contains(&v),
                "Jitter+dodge offset {} should be in range [-0.45, 0.45]",
                v
            );
        }
    }

    #[test]
    fn test_jitter_bidirectional() {
        // When both axes are discrete, both offset columns are created
        let jitter = Jitter;
        let df = make_test_df();
        let layer = make_test_layer();

        // Both axes must be explicitly marked as discrete
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify both offset columns were created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_ok(),
            "pos1offset column should be created"
        );
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_ok(),
            "pos2offset column should be created"
        );
    }

    #[test]
    fn test_jitter_neither_discrete() {
        // When both axes are continuous, no offset columns are created
        let jitter = Jitter;
        let df = make_test_df();
        let layer = make_test_layer();

        // Mark both as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify neither offset column was created
        assert!(
            result.column("__ggsql_aes_pos1offset__").is_err(),
            "pos1offset column should NOT be created when pos1 is continuous"
        );
        assert!(
            result.column("__ggsql_aes_pos2offset__").is_err(),
            "pos2offset column should NOT be created when pos2 is continuous"
        );
    }

    #[test]
    fn test_jitter_custom_width_with_dodge() {
        let jitter = Jitter;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer
            .parameters
            .insert("width".to_string(), ParameterValue::Number(0.6));

        // Mark pos1 as discrete and pos2 as continuous so only pos1offset is created
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // With custom width 0.6 and 2 groups (dodge=true):
        // effective_width = 0.6 / 2 = 0.3
        // Total range: [-0.3, 0.3]
        for &v in &offsets {
            assert!(
                (-0.3..=0.3).contains(&v),
                "Jitter+dodge offset {} should be in range [-0.3, 0.3] with width 0.6",
                v
            );
        }
    }

    #[test]
    fn test_jitter_groups_separate_with_dodge() {
        // With dodge=true, different groups should have different center positions
        let jitter = Jitter;

        let df = make_test_df();
        let layer = make_test_layer();

        // Mark pos1 as discrete and pos2 as continuous so only pos1offset is created
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let fill_col = result.column("__ggsql_aes_fill__").unwrap();

        // Collect offsets by group
        let mut group_x_offsets = vec![];
        let mut group_y_offsets = vec![];

        for i in 0..result.height() {
            let fill_val = fill_col.get(i).unwrap();
            let offset_val = offset.get(i).unwrap();
            if fill_val.to_string().contains("X") {
                group_x_offsets.push(offset_val);
            } else {
                group_y_offsets.push(offset_val);
            }
        }

        // With dodge, group X should have negative-centered offsets
        // and group Y should have positive-centered offsets
        let x_mean: f64 = group_x_offsets.iter().sum::<f64>() / group_x_offsets.len() as f64;
        let y_mean: f64 = group_y_offsets.iter().sum::<f64>() / group_y_offsets.len() as f64;

        // The means should be on opposite sides of 0 (X negative, Y positive)
        // Allow some variance due to jitter randomness
        assert!(
            x_mean < y_mean,
            "Group X mean ({}) should be less than Group Y mean ({})",
            x_mean,
            y_mean
        );
    }

    #[test]
    fn test_jitter_no_groups_no_dodge() {
        // Without partition_by columns, no dodge is applied
        let jitter = Jitter;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.partition_by = vec![]; // No grouping

        // Mark pos1 as discrete and pos2 as continuous so only pos1offset is created
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // Without groups, pure jitter with full width range [-0.45, 0.45]
        for &v in &offsets {
            assert!(
                (-0.45..=0.45).contains(&v),
                "Pure jitter offset {} should be in range [-0.45, 0.45]",
                v
            );
        }
    }

    #[test]
    fn test_jitter_creates_pos1offset() {
        assert!(Jitter.creates_pos1offset());
    }

    #[test]
    fn test_jitter_creates_pos2offset() {
        assert!(Jitter.creates_pos2offset());
    }

    #[test]
    fn test_jitter_default_params() {
        let jitter = Jitter;
        let params = jitter.default_params();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0].name, "width");
        assert!(matches!(params[0].default, DefaultParamValue::Number(0.9)));
        assert_eq!(params[1].name, "dodge");
        assert!(matches!(
            params[1].default,
            DefaultParamValue::Boolean(true)
        ));
        assert_eq!(params[2].name, "distribution");
        assert!(matches!(
            params[2].default,
            DefaultParamValue::String("uniform")
        ));
        // Density distribution parameters (match violin/density geoms)
        assert_eq!(params[3].name, "bandwidth");
        assert!(matches!(params[3].default, DefaultParamValue::Null));
        assert_eq!(params[4].name, "adjust");
        assert!(matches!(params[4].default, DefaultParamValue::Number(1.0)));
    }

    #[test]
    fn test_jitter_normal_distribution() {
        // Normal distribution should have ~95% of values within the width
        let jitter = Jitter;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.partition_by = vec![]; // No grouping for pure jitter
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("normal".to_string()),
        );

        // Mark pos1 as discrete and pos2 as continuous so only pos1offset is created
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // Normal distribution is centered at 0
        // Values can exceed the width bounds (unlike uniform), but should be centered
        // With only 4 samples and σ = width/4 = 0.225, the standard error is ~0.11
        // Use a wide tolerance to avoid flaky tests with small sample sizes
        let mean: f64 = offsets.iter().sum::<f64>() / offsets.len() as f64;
        assert!(
            mean.abs() < 0.5,
            "Normal distribution mean {} should be close to 0",
            mean
        );
    }

    #[test]
    fn test_jitter_distribution_from_str() {
        assert_eq!(
            JitterDistribution::from_str("uniform"),
            JitterDistribution::Uniform
        );
        assert_eq!(
            JitterDistribution::from_str("normal"),
            JitterDistribution::Normal
        );
        assert_eq!(
            JitterDistribution::from_str("gaussian"),
            JitterDistribution::Normal
        );
        assert_eq!(
            JitterDistribution::from_str("density"),
            JitterDistribution::Density
        );
        assert_eq!(
            JitterDistribution::from_str("DENSITY"),
            JitterDistribution::Density
        );
        assert_eq!(
            JitterDistribution::from_str("NORMAL"),
            JitterDistribution::Normal
        );
        assert_eq!(
            JitterDistribution::from_str("intensity"),
            JitterDistribution::Intensity
        );
        assert_eq!(
            JitterDistribution::from_str("INTENSITY"),
            JitterDistribution::Intensity
        );
        assert_eq!(
            JitterDistribution::from_str("unknown"),
            JitterDistribution::Uniform
        );
    }

    #[test]
    fn test_jitter_density_requires_one_continuous_axis() {
        // Density distribution requires exactly one continuous axis
        let jitter = Jitter;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.partition_by = vec![]; // No grouping
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("density".to_string()),
        );

        // Test 1: Both axes discrete - should fail
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));
        let result = jitter.apply_adjustment(df.clone(), &layer, &spec);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires exactly one continuous axis"));

        // Test 2: Both axes continuous - should fail
        let mut spec = Plot::new();
        spec.scales.push(make_continuous_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        let result = jitter.apply_adjustment(df.clone(), &layer, &spec);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires exactly one continuous axis"));

        // Test 3: Only pos2 continuous (pos1 discrete) - should succeed
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        let result = jitter.apply_adjustment(df, &layer, &spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jitter_density_distribution() {
        // Density distribution should create violin-like spread
        // Points in dense regions should have larger jitter amplitude (due to density scaling)
        let jitter = Jitter;

        // Create data with clear density peaks
        // Values 1.0 appears 5 times, values 2.0 and 3.0 appear once each
        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m
        };
        layer.partition_by = vec![];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("density".to_string()),
        );

        // Mark pos1 as discrete and pos2 as continuous (density computed along pos2)
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // Due to randomness, we can't assert exact values
        // But we can verify that offsets were generated
        assert_eq!(offsets.len(), 7);
    }

    #[test]
    fn test_jitter_density_per_group() {
        // When groups exist, density should be computed separately per group
        let jitter = Jitter;

        // Create data with two groups, each with different density distributions
        // Group X: dense at 1.0
        // Group Y: dense at 3.0
        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "__ggsql_aes_fill__" => ["X", "X", "X", "Y", "Y", "Y"],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m.insert(
                "fill",
                AestheticValue::standard_column("__ggsql_aes_fill__"),
            );
            m
        };
        layer.partition_by = vec!["__ggsql_aes_fill__".to_string()];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("density".to_string()),
        );

        // Mark pos1 as discrete and pos2 as continuous
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify offsets were created and are within expected bounds
        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
        assert_eq!(offsets.len(), 6);

        // With 2 groups, we should see separated dodge positions
        // Group X centered at negative, Group Y centered at positive
        let fill_col = result.column("__ggsql_aes_fill__").unwrap();
        let mut group_x_offsets = vec![];
        let mut group_y_offsets = vec![];

        for i in 0..result.height() {
            let fill_val = fill_col.get(i).unwrap();
            let offset_val = offset.get(i).unwrap();
            if fill_val.to_string().contains("X") {
                group_x_offsets.push(offset_val);
            } else {
                group_y_offsets.push(offset_val);
            }
        }

        // Groups should be separated due to dodge
        let x_mean: f64 = group_x_offsets.iter().sum::<f64>() / group_x_offsets.len() as f64;
        let y_mean: f64 = group_y_offsets.iter().sum::<f64>() / group_y_offsets.len() as f64;
        assert!(
            x_mean < y_mean,
            "Group X mean ({}) should be less than Group Y mean ({})",
            x_mean,
            y_mean
        );
    }

    #[test]
    fn test_silverman_bandwidth() {
        // Test Silverman bandwidth computation with default adjust=1.0
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bandwidth = super::silverman_bandwidth(&values, 1.0);
        // Should return a positive value
        assert!(bandwidth > 0.0);

        // Constant data should return fallback
        let constant = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let bandwidth = super::silverman_bandwidth(&constant, 1.0);
        assert_eq!(bandwidth, 1.0);

        // Single value should return fallback
        let single = vec![5.0];
        let bandwidth = super::silverman_bandwidth(&single, 1.0);
        assert_eq!(bandwidth, 1.0);

        // Test adjust parameter
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bw_default = super::silverman_bandwidth(&values, 1.0);
        let bw_double = super::silverman_bandwidth(&values, 2.0);
        assert!(
            (bw_double - bw_default * 2.0).abs() < 1e-10,
            "Bandwidth with adjust=2.0 should be twice the default"
        );
    }

    #[test]
    fn test_compute_densities() {
        // Test that densities are computed correctly
        let values = vec![0.0, 0.0, 0.0, 5.0, 10.0];
        let bandwidth = 1.0;
        let densities = super::compute_densities(&values, bandwidth);

        // Values near 0.0 (3 points) should have higher density than value at 10.0
        assert!(densities[0] > densities[4]);
        assert!(densities[1] > densities[4]);
        assert!(densities[2] > densities[4]);
    }

    #[test]
    fn test_compute_intensities() {
        // Test that intensities differ from densities by not dividing by n
        let values = vec![1.0, 1.0, 1.0, 5.0, 10.0];
        let bandwidth = 1.0;
        let densities = super::compute_densities(&values, bandwidth);
        let intensities = super::compute_intensities(&values, bandwidth);

        // Intensities should be n times larger than densities
        let n = values.len() as f64;
        for (d, i) in densities.iter().zip(intensities.iter()) {
            assert!(
                (i - d * n).abs() < 1e-10,
                "Intensity {} should be {} times density {}",
                i,
                n,
                d
            );
        }
    }

    #[test]
    fn test_jitter_intensity_requires_one_continuous_axis() {
        // Intensity distribution requires exactly one continuous axis
        let jitter = Jitter;

        let df = make_test_df();
        let mut layer = make_test_layer();
        layer.partition_by = vec![]; // No grouping
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("intensity".to_string()),
        );

        // Both axes discrete - should fail
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_discrete_scale("pos2"));
        let result = jitter.apply_adjustment(df.clone(), &layer, &spec);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires exactly one continuous axis"));

        // Only pos2 continuous (pos1 discrete) - should succeed
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));
        let result = jitter.apply_adjustment(df, &layer, &spec);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jitter_intensity_distribution() {
        // Intensity distribution should create violin-like spread
        let jitter = Jitter;

        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m
        };
        layer.partition_by = vec![];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("intensity".to_string()),
        );

        // Mark pos1 as discrete and pos2 as continuous (density computed along pos2)
        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();

        // Due to randomness, we can't assert exact values
        // But we can verify that offsets were generated
        assert_eq!(offsets.len(), 7);
    }

    #[test]
    fn test_jitter_intensity_global_normalization() {
        // Test that intensity uses global max normalization across groups
        // Group A has 5 points, Group B has 2 points
        // With intensity distribution, Group A should have larger scales (more data)
        let jitter = Jitter;

        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A", "B", "B"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m
        };
        layer.partition_by = vec![];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("intensity".to_string()),
        );
        // Use dodge=false to avoid group separation
        layer
            .parameters
            .insert("dodge".to_string(), ParameterValue::Boolean(false));

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let (result, _) = jitter.apply_adjustment(df, &layer, &spec).unwrap();

        // Verify offsets were created
        let offset = result
            .column("__ggsql_aes_pos1offset__")
            .unwrap()
            .f64()
            .unwrap();
        let offsets: Vec<f64> = offset.into_iter().flatten().collect();
        assert_eq!(offsets.len(), 7);
    }

    #[test]
    fn test_jitter_density_explicit_bandwidth() {
        // Test that explicit bandwidth parameter is used
        let jitter = Jitter;

        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 2.0, 3.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m
        };
        layer.partition_by = vec![];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("density".to_string()),
        );
        // Set explicit bandwidth matching what violin might use
        layer
            .parameters
            .insert("bandwidth".to_string(), ParameterValue::Number(0.5));

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let result = jitter.apply_adjustment(df, &layer, &spec);
        assert!(result.is_ok(), "Should succeed with explicit bandwidth");
    }

    #[test]
    fn test_jitter_density_adjust_parameter() {
        // Test that adjust parameter scales bandwidth
        let jitter = Jitter;

        let df = df! {
            "__ggsql_aes_pos1__" => ["A", "A", "A", "A", "A"],
            "__ggsql_aes_pos2__" => [1.0, 1.0, 1.0, 2.0, 3.0],
            "__ggsql_aes_pos2end__" => [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        .unwrap();

        let mut layer = Layer::new(Geom::bar());
        layer.mappings = {
            let mut m = Mappings::new();
            m.insert(
                "pos1",
                AestheticValue::standard_column("__ggsql_aes_pos1__"),
            );
            m.insert(
                "pos2",
                AestheticValue::standard_column("__ggsql_aes_pos2__"),
            );
            m.insert(
                "pos2end",
                AestheticValue::standard_column("__ggsql_aes_pos2end__"),
            );
            m
        };
        layer.partition_by = vec![];
        layer.parameters.insert(
            "distribution".to_string(),
            ParameterValue::String("density".to_string()),
        );
        // Set adjust parameter (scales auto-computed bandwidth)
        layer
            .parameters
            .insert("adjust".to_string(), ParameterValue::Number(2.0));

        let mut spec = Plot::new();
        spec.scales.push(make_discrete_scale("pos1"));
        spec.scales.push(make_continuous_scale("pos2"));

        let result = jitter.apply_adjustment(df, &layer, &spec);
        assert!(result.is_ok(), "Should succeed with adjust parameter");
    }

    #[test]
    fn test_quantile_cont() {
        // Test quantile interpolation
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Exact quartiles
        let q0 = super::quantile_cont(&sorted, 0.0);
        assert!((q0 - 1.0).abs() < 1e-10);

        let q1 = super::quantile_cont(&sorted, 1.0);
        assert!((q1 - 5.0).abs() < 1e-10);

        // Median
        let q50 = super::quantile_cont(&sorted, 0.5);
        assert!((q50 - 3.0).abs() < 1e-10);

        // Interpolated values
        let q25 = super::quantile_cont(&sorted, 0.25);
        assert!((q25 - 2.0).abs() < 1e-10);

        let q75 = super::quantile_cont(&sorted, 0.75);
        assert!((q75 - 4.0).abs() < 1e-10);
    }
}
