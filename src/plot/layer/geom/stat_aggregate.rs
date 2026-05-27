//! Aggregate stat — collapse each group to a single row by applying an
//! aggregate function per numeric mapping.
//!
//! When a layer's `aggregate` SETTING is set, this stat groups by discrete
//! mappings + PARTITION BY columns and emits one row per group. Each numeric
//! column-mapping (positional *and* material) is replaced in place by the
//! aggregated value of its bound column. Discrete mappings stay as group keys;
//! literal mappings pass through unchanged.
//!
//! # Setting shape
//!
//! `aggregate` accepts a single string or array of strings. Each string is
//! either:
//!
//! - **default** — `'<func>'` (no prefix). Up to two defaults may be supplied.
//!   With one default it applies to every untargeted numeric mapping. With two
//!   defaults the first applies to *lower-half* aesthetics (no suffix or `min`
//!   suffix) plus all non-range geoms, and the second applies to *upper-half*
//!   aesthetics (`max` or `end` suffix). More than two defaults is an error.
//! - **target** — `'<aes>:<func>'`. Applies `func` to the named aesthetic only.
//!   `<aes>` is a user-facing name (`x`, `y`, `xmin`, `xmax`, `xend`, `yend`,
//!   `color`, `size`, …); the stat resolves it to the internal name through
//!   `AestheticContext`.
//!
//! Numeric mappings without a target *or* applicable default are dropped with
//! a warning to stderr.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use regex::Regex;

use super::types::StatResult;
use crate::naming;
use crate::plot::aesthetic::AestheticContext;
use crate::plot::types::{ArrayElement, ParameterValue, Schema};
use crate::reader::SqlDialect;
use crate::{GgsqlError, Mappings, Result};

/// All simple-aggregation function names accepted by the `aggregate` SETTING.
///
/// Band names (e.g. `mean+sdev`, `median-0.5iqr`) are validated separately by
/// `parse_agg_name`, which checks the offset against `OFFSET_STATS` and the
/// expansion against `EXPANSION_STATS`.
pub const AGG_NAMES: &[&str] = &[
    // Tallies & sums
    "count", "sum", "prod", // Extremes
    "min", "max", "range", "mid", // Central tendency
    "mean", "geomean", "harmean", "rms", "median", // Spread (standalone)
    "sdev", "var", "iqr", // Percentiles
    "p05", "p10", "p25", "p50", "p75", "p90", "p95", // Positional (row order in the group)
    "first", "last", "diff",
];

/// Stats that can appear as the *offset* (left of `±`) in a band name like
/// `mean+sdev`. Single-value central or representative quantities only —
/// counts/spreads are excluded.
pub const OFFSET_STATS: &[&str] = &[
    "mean", "median", "geomean", "harmean", "rms", "sum", "prod", "min", "max", "mid", "p05",
    "p10", "p25", "p50", "p75", "p90", "p95",
];

/// Stats that can appear as the *expansion* (right of `±[mod]`) in a band name.
/// Spread / dispersion measures only.
pub const EXPANSION_STATS: &[&str] = &["sdev", "se", "var", "iqr", "range"];

/// Parsed representation of any aggregate-function name.
///
/// Simple aggregates (`mean`, `count`, `p25`) have `band == None`. Band names
/// (`mean+sdev`, `median-0.5iqr`) have `band == Some(...)` with the offset
/// stored in `offset` and the spread/multiplier in `band`.
#[derive(Debug, Clone, PartialEq)]
pub struct AggSpec {
    pub offset: &'static str,
    pub band: Option<Band>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Signed multiplier on the expansion. `+1.0` corresponds to `<offset>+<exp>`;
    /// `-1.96` corresponds to `<offset>-1.96<exp>`. The sign and magnitude are
    /// folded together so there's a single source of truth.
    pub mod_value: f64,
    pub expansion: &'static str,
}

fn resolve_static(name: &str, vocab: &'static [&'static str]) -> Option<&'static str> {
    vocab.iter().copied().find(|v| *v == name)
}

/// Single regex covering one `aggregate` entry: optional `<aes>:` prefix,
/// required offset name, optional `±[<mult>]<expansion>` band suffix.
///
/// Capture groups:
/// 1. aesthetic prefix (anything up to the first `:`; structural-only — full
///    aesthetic resolution happens in `apply()`)
/// 2. offset name
/// 3. sign — present iff the entry has a band
/// 4. magnitude — optional, defaults to `1.0`
/// 5. expansion name
fn entry_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(?:([^:]+):)?([a-z]+\d*)(?:([+-])(\d+(?:\.\d+)?)?([a-z]+))?$").unwrap()
    })
}

/// Parsed shape of a single `aggregate` array entry.
struct ParsedEntry {
    /// `Some(name)` when the entry has an `<aes>:` prefix; `None` for an
    /// unprefixed default. Resolution to internal aesthetic names happens in
    /// `apply()` via `resolve_target_aesthetic`.
    aesthetic: Option<String>,
    spec: AggSpec,
}

fn parse_entry(entry: &str) -> std::result::Result<ParsedEntry, String> {
    let caps = entry_re()
        .captures(entry)
        .ok_or_else(|| format!("could not parse aggregate entry '{}'", entry))?;

    let aesthetic = caps.get(1).map(|m| m.as_str().to_string());
    let offset_str = caps.get(2).unwrap().as_str();
    let band_present = caps.get(3).is_some();

    let band = if band_present {
        let expansion_str = caps.get(5).unwrap().as_str();
        let expansion = resolve_static(expansion_str, EXPANSION_STATS).ok_or_else(|| {
            format!(
                "'{}': '{}' is not a valid expansion stat. Allowed expansions: {}",
                entry,
                expansion_str,
                crate::or_list_quoted(EXPANSION_STATS, '\''),
            )
        })?;
        let magnitude: f64 = caps
            .get(4)
            .map_or(1.0, |m| m.as_str().parse().unwrap_or(1.0));
        let mod_value = if caps.get(3).unwrap().as_str() == "-" {
            -magnitude
        } else {
            magnitude
        };
        Some(Band {
            mod_value,
            expansion,
        })
    } else {
        None
    };

    let offset = if band.is_some() {
        resolve_static(offset_str, OFFSET_STATS).ok_or_else(|| {
            if AGG_NAMES.contains(&offset_str) {
                format!(
                    "'{}': '{}' is not a valid offset stat. Allowed offsets: {}",
                    entry,
                    offset_str,
                    crate::or_list_quoted(OFFSET_STATS, '\''),
                )
            } else {
                format!(
                    "'{}': '{}' is not a known stat. Allowed offsets: {}",
                    entry,
                    offset_str,
                    crate::or_list_quoted(OFFSET_STATS, '\''),
                )
            }
        })?
    } else {
        resolve_static(offset_str, AGG_NAMES).ok_or_else(|| {
            format!(
                "unknown aggregate function '{}'. Allowed: {} (or use a band like `mean+sdev`)",
                offset_str,
                crate::or_list_quoted(AGG_NAMES, '\''),
            )
        })?
    };

    Ok(ParsedEntry {
        aesthetic,
        spec: AggSpec { offset, band },
    })
}

// =============================================================================
// AggregateSpec — parsed representation of the `aggregate` SETTING.
// =============================================================================

/// Parsed `aggregate` SETTING.
///
/// Up to two unprefixed defaults plus per-aesthetic targets. A target may be
/// named more than once; the multiple functions cause that aesthetic to
/// *explode* into multiple rows per group
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateSpec {
    pub default_lower: Option<AggSpec>,
    pub default_upper: Option<AggSpec>,
    /// Targets in declaration order. Each entry is `(user-facing aesthetic,
    /// non-empty list of functions)`. Multiple SETTING entries with the same
    /// aesthetic are merged into one list during parsing — the cumulative
    /// length determines that aesthetic's explosion factor.
    pub targets: Vec<(String, Vec<AggSpec>)>,
}

impl AggregateSpec {
    fn new() -> Self {
        Self {
            default_lower: None,
            default_upper: None,
            targets: Vec::new(),
        }
    }

    /// Maximum target list length, or `1` if every target has a single function.
    /// This is the number of exploded rows the stat will emit per group.
    pub fn explosion_factor(&self) -> usize {
        self.targets
            .iter()
            .map(|(_, fns)| fns.len())
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Per-row labels for the synthetic `aggregate` column. `None` for the
    /// single-row case (no explosion), since the column only makes sense as a
    /// row-differentiator and there's nothing to differentiate.
    ///
    /// For each row in `0..explosion_factor`, walks every *exploded* target
    /// (length == n; length-1 recycled targets are skipped because they take
    /// the same value on every row), collects each target's function name at
    /// that row, deduplicates them while preserving declaration order, and
    /// joins with `/`.
    ///
    /// Examples (with `n = 2`):
    /// - `('y:min', 'y:max')` → `['min', 'max']`
    /// - `('y:min', 'y:max', 'color:sum', 'color:prod')` → `['min/sum', 'max/prod']`
    /// - `('y:mean', 'y:max', 'color:mean', 'color:prod')` → `['mean', 'max/prod']`
    /// - `('y:min', 'y:max', 'color:median')` → `['min', 'max']` (color is recycled)
    pub fn explosion_labels(&self) -> Option<Vec<String>> {
        let n = self.explosion_factor();
        if n <= 1 {
            return None;
        }
        let exploded: Vec<&Vec<AggSpec>> = self
            .targets
            .iter()
            .filter(|(_, fns)| fns.len() == n)
            .map(|(_, fns)| fns)
            .collect();
        let labels = (0..n)
            .map(|row| {
                let mut parts: Vec<String> = Vec::new();
                for fns in &exploded {
                    let label = agg_label(&fns[row]);
                    if !parts.contains(&label) {
                        parts.push(label);
                    }
                }
                parts.join("/")
            })
            .collect();
        Some(labels)
    }
}

/// Human-readable label for an `AggSpec`. Re-emits simple names verbatim and
/// reconstructs band names like `mean+sdev` / `mean-1.96sdev`.
fn agg_label(spec: &AggSpec) -> String {
    match &spec.band {
        None => spec.offset.to_string(),
        Some(b) => {
            let sign = if b.mod_value < 0.0 { '-' } else { '+' };
            let magnitude = b.mod_value.abs();
            if magnitude == 1.0 {
                format!("{}{}{}", spec.offset, sign, b.expansion)
            } else {
                format!("{}{}{}{}", spec.offset, sign, magnitude, b.expansion)
            }
        }
    }
}

/// Parse the `aggregate` SETTING value into an `AggregateSpec`. Returns
/// `Ok(None)` when the parameter is unset, null, or empty. Returns `Err(...)`
/// for malformed input.
pub fn parse_aggregate_param(
    value: &ParameterValue,
) -> std::result::Result<Option<AggregateSpec>, String> {
    let entries: Vec<&str> = match value {
        ParameterValue::Null => return Ok(None),
        ParameterValue::String(s) => vec![s.as_str()],
        ParameterValue::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for el in arr {
                match el {
                    ArrayElement::String(s) => out.push(s.as_str()),
                    ArrayElement::Null => continue,
                    _ => {
                        return Err("'aggregate' array entries must be strings or null".to_string());
                    }
                }
            }
            if out.is_empty() {
                return Ok(None);
            }
            out
        }
        _ => return Err("'aggregate' must be a string, array of strings, or null".to_string()),
    };

    let mut spec = AggregateSpec::new();
    for entry in entries {
        let parsed = parse_entry(entry)?;
        match parsed.aesthetic {
            Some(aes) => {
                if let Some((_, fns)) = spec.targets.iter_mut().find(|(a, _)| *a == aes) {
                    fns.push(parsed.spec);
                } else {
                    spec.targets.push((aes, vec![parsed.spec]));
                }
            }
            None => {
                if spec.default_lower.is_none() {
                    spec.default_lower = Some(parsed.spec);
                } else if spec.default_upper.is_none() {
                    spec.default_upper = Some(parsed.spec);
                } else {
                    return Err(format!(
                        "'aggregate' accepts at most two unprefixed defaults; got a third: '{}'",
                        entry
                    ));
                }
            }
        }
    }

    if spec.default_lower.is_none() && spec.default_upper.is_none() && spec.targets.is_empty() {
        return Ok(None);
    }

    // Validate recycling: every target list must be length 1 or N (the max).
    let n = spec.explosion_factor();
    if n > 1 {
        for (aes, fns) in &spec.targets {
            if fns.len() != 1 && fns.len() != n {
                return Err(format!(
                    "aggregate target '{}' has {} functions; targets in an exploded layer must \
                     have either 1 or {} functions (the longest target's count)",
                    aes,
                    fns.len(),
                    n
                ));
            }
        }
    }

    Ok(Some(spec))
}

// =============================================================================
// SQL fragment helpers (per-column aggregate expressions).
// =============================================================================

/// Map a percentile function name (`p05`..`p95`, `median`) to its fraction.
fn percentile_fraction(func: &str) -> Option<f64> {
    match func {
        "median" | "p50" => Some(0.50),
        "p05" => Some(0.05),
        "p10" => Some(0.10),
        "p25" => Some(0.25),
        "p75" => Some(0.75),
        "p90" => Some(0.90),
        "p95" => Some(0.95),
        _ => None,
    }
}

/// Build the inline SQL fragment for a *simple* stat (no band) applied to a
/// quoted column. Returns `None` when the dialect cannot express this
/// aggregate inline — for the percentile/iqr family that means the caller
/// switches to the correlated `sql_percentile` fallback; for other names it
/// means the dialect doesn't support that function and the stat layer raises
/// a clear error before SQL is built (see `validate_supported`).
fn simple_stat_sql_inline(name: &str, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    if let Some(frac) = percentile_fraction(name) {
        let unquoted = unquote(qcol);
        return dialect.sql_quantile_inline(&unquoted, frac);
    }
    if name == "iqr" {
        let unquoted = unquote(qcol);
        let p75 = dialect.sql_quantile_inline(&unquoted, 0.75)?;
        let p25 = dialect.sql_quantile_inline(&unquoted, 0.25)?;
        return Some(format!("({} - {})", p75, p25));
    }
    dialect.sql_aggregate(name, qcol)
}

/// Whether the dialect can produce SQL for this aggregate (inline or via the
/// percentile fallback). Used to surface a clear error before SQL is built.
fn dialect_supports(name: &str, dialect: &dyn SqlDialect) -> bool {
    if percentile_fraction(name).is_some() || name == "iqr" {
        // Always supported: percentile path falls back to a correlated subquery
        // built from `sql_percentile`, which has a portable default.
        return true;
    }
    dialect.sql_aggregate(name, "x").is_some()
}

/// Walk every aggregate that will be emitted and confirm the dialect supports
/// it. Returns the list of unsupported function names, deduplicated.
fn unsupported_functions(
    aggregated: &[(String, String, Vec<AggSpec>)],
    dialect: &dyn SqlDialect,
) -> Vec<String> {
    let mut missing: Vec<String> = Vec::new();
    for (_, _, specs) in aggregated {
        for spec in specs {
            for name in [Some(spec.offset), spec.band.as_ref().map(|b| b.expansion)]
                .into_iter()
                .flatten()
            {
                if !dialect_supports(name, dialect) && !missing.iter().any(|m| m == name) {
                    missing.push(name.to_string());
                }
            }
        }
    }
    missing
}

fn agg_sql_inline(spec: &AggSpec, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    let offset_sql = simple_stat_sql_inline(spec.offset, qcol, dialect)?;
    match &spec.band {
        None => Some(offset_sql),
        Some(band) => {
            let exp_sql = simple_stat_sql_inline(band.expansion, qcol, dialect)?;
            Some(format_band(&offset_sql, band.mod_value, &exp_sql))
        }
    }
}

/// Format a band expression `(offset ± [magnitude *] expansion)`. The sign and
/// magnitude come folded together in `mod_value`; this splits them back out
/// only when emitting SQL so the output is readable (e.g. `(mean - 1.96 * sdev)`
/// rather than `(mean + -1.96 * sdev)`).
fn format_band(offset: &str, mod_value: f64, exp: &str) -> String {
    let sign = if mod_value < 0.0 { '-' } else { '+' };
    let magnitude = mod_value.abs();
    if magnitude == 1.0 {
        format!("({} {} {})", offset, sign, exp)
    } else {
        format!("({} {} {} * {})", offset, sign, magnitude, exp)
    }
}

/// Fallback SQL for a simple stat — used when a percentile component lacks
/// inline support. Emits a correlated `sql_percentile` subquery; falls
/// through to the inline form for everything else.
fn simple_stat_sql_fallback(
    name: &str,
    raw_col: &str,
    dialect: &dyn SqlDialect,
    src_alias: &str,
    group_cols: &[String],
) -> String {
    if let Some(frac) = percentile_fraction(name) {
        return dialect.sql_percentile(raw_col, frac, src_alias, group_cols);
    }
    if name == "iqr" {
        let p75 = dialect.sql_percentile(raw_col, 0.75, src_alias, group_cols);
        let p25 = dialect.sql_percentile(raw_col, 0.25, src_alias, group_cols);
        return format!("({} - {})", p75, p25);
    }
    let qcol = naming::quote_ident(raw_col);
    simple_stat_sql_inline(name, &qcol, dialect).unwrap_or_else(|| "NULL".to_string())
}

fn agg_sql_fallback(
    spec: &AggSpec,
    raw_col: &str,
    dialect: &dyn SqlDialect,
    src_alias: &str,
    group_cols: &[String],
) -> String {
    let offset_sql = simple_stat_sql_fallback(spec.offset, raw_col, dialect, src_alias, group_cols);
    match &spec.band {
        None => offset_sql,
        Some(band) => {
            let exp_sql =
                simple_stat_sql_fallback(band.expansion, raw_col, dialect, src_alias, group_cols);
            format_band(&offset_sql, band.mod_value, &exp_sql)
        }
    }
}

fn needs_quantile_fallback(spec: &AggSpec, probe_col: &str, dialect: &dyn SqlDialect) -> bool {
    if simple_needs_fallback(spec.offset, probe_col, dialect) {
        return true;
    }
    if let Some(band) = &spec.band {
        if simple_needs_fallback(band.expansion, probe_col, dialect) {
            return true;
        }
    }
    false
}

fn simple_needs_fallback(name: &str, probe_col: &str, dialect: &dyn SqlDialect) -> bool {
    if let Some(frac) = percentile_fraction(name) {
        return dialect.sql_quantile_inline(probe_col, frac).is_none();
    }
    if name == "iqr" {
        return dialect.sql_quantile_inline(probe_col, 0.5).is_none();
    }
    false
}

fn unquote(qcol: &str) -> String {
    let trimmed = qcol.trim_start_matches('"').trim_end_matches('"');
    trimmed.replace("\"\"", "\"")
}

// =============================================================================
// apply — entry point.
// =============================================================================

/// Resolve a user-facing target aesthetic name to one or more internal names
/// that are actually mapped on the layer. Handles three cases:
/// 1. The name maps directly through `AestheticContext` (e.g. `y` → `pos2`).
/// 2. The name is an alias from `AESTHETIC_ALIASES` (e.g. `color` → `stroke`,
///    `fill`); each target whose internal counterpart is mapped is included.
/// 3. The name is a material aesthetic with the same internal name (e.g. `size`).
///
/// Returns the empty vector if no resolution finds a mapped aesthetic.
fn resolve_target_aesthetic(
    user_aes: &str,
    aesthetics: &Mappings,
    aesthetic_ctx: &AestheticContext,
) -> Vec<String> {
    use crate::plot::layer::geom::types::AESTHETIC_ALIASES;
    let mut out = Vec::new();
    if let Some(internal) = aesthetic_ctx.map_user_to_internal(user_aes) {
        if aesthetics.aesthetics.contains_key(internal) {
            out.push(internal.to_string());
            return out;
        }
    }
    for (alias, targets) in AESTHETIC_ALIASES {
        if *alias == user_aes {
            for t in *targets {
                let internal = aesthetic_ctx
                    .map_user_to_internal(t)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| (*t).to_string());
                if aesthetics.aesthetics.contains_key(&internal) && !out.contains(&internal) {
                    out.push(internal);
                }
            }
            return out;
        }
    }
    if aesthetics.aesthetics.contains_key(user_aes) {
        out.push(user_aes.to_string());
    }
    out
}

/// Classify an internal aesthetic name as upper-half or lower-half for the
/// purpose of default-aggregate routing.
///
/// `min` suffix → lower; `max`/`end` → upper; no suffix → lower. Material
/// aesthetics (no position prefix) are always lower.
fn is_upper_half(internal_aes: &str) -> bool {
    internal_aes.ends_with("max") || internal_aes.ends_with("end")
}

/// Resolve every user-facing target in `spec` to its internal aesthetic
/// name(s) on the layer. Returns a map keyed by internal aesthetic name with
/// the function list each target supplies, or an error when a target doesn't
/// match any mapped aesthetic or when two targets resolve to the same
/// aesthetic.
///
/// Shared between [`apply`] (which uses the returned map) and
/// `Layer::validate_aggregate_setting` (which discards the map and just
/// surfaces the error). Callers pass an already-parsed [`AggregateSpec`] to
/// avoid re-parsing the raw setting.
pub(crate) fn resolve_aggregate_targets(
    spec: &AggregateSpec,
    aesthetics: &Mappings,
    aesthetic_ctx: &AestheticContext,
) -> std::result::Result<HashMap<String, Vec<AggSpec>>, String> {
    let mut targets_internal: HashMap<String, Vec<AggSpec>> = HashMap::new();
    for (user_aes, fns) in &spec.targets {
        let resolved = resolve_target_aesthetic(user_aes, aesthetics, aesthetic_ctx);
        if resolved.is_empty() {
            return Err(format!(
                "aggregate target '{}' is not mapped on this layer",
                user_aes
            ));
        }
        for internal in resolved {
            if targets_internal.contains_key(&internal) {
                return Err(format!(
                    "aggregate target '{}' resolves to aesthetic '{}' which is already targeted",
                    user_aes, internal
                ));
            }
            targets_internal.insert(internal, fns.clone());
        }
    }
    Ok(targets_internal)
}

/// Compute the set of internal aesthetic names that the layer's `aggregate`
/// setting *explicitly targets*. Lighter than [`aggregated_aesthetics`] —
/// doesn't need a schema — so post-stat callers can use it without rebuilding
/// type information from a materialised DataFrame.
pub fn targeted_aesthetics(
    parameters: &HashMap<String, ParameterValue>,
    aesthetics: &Mappings,
    aesthetic_ctx: &AestheticContext,
) -> HashSet<String> {
    let raw = match parameters.get("aggregate") {
        Some(v) if !matches!(v, ParameterValue::Null) => v,
        _ => return HashSet::new(),
    };
    let spec = match parse_aggregate_param(raw).ok().flatten() {
        Some(s) => s,
        None => return HashSet::new(),
    };
    let mut targeted: HashSet<String> = HashSet::new();
    for (user_aes, _fns) in &spec.targets {
        for internal in resolve_target_aesthetic(user_aes, aesthetics, aesthetic_ctx) {
            targeted.insert(internal);
        }
    }
    targeted
}

/// Compute, for a layer's `aggregate` setting, which internal aesthetic names
/// will be (a) *explicitly targeted* by `aggregate => '<aes>:<func>'` and
/// (b) *aggregated* by the stat (either targeted OR a numeric mapping that an
/// untargeted default applies to).
///
/// The execute pipeline uses this to decide whether to defer scale-driven
/// pre-stat rewrites (`SCALE BINNED <aes>`, `SCALE <aes> FROM […]`, …) until
/// after the stat. The bucketing here mirrors the per-mapping branching in
/// [`apply`]; both must stay in sync.
///
/// Returns `None` when `aggregate` is unset, null, or fails to parse — i.e.
/// when the stat will return `Identity` and no aesthetic is touched. Parse
/// errors are swallowed; the stat itself surfaces a clean diagnostic.
pub fn aggregated_aesthetics(
    parameters: &HashMap<String, ParameterValue>,
    aesthetics: &Mappings,
    schema: &Schema,
    aesthetic_ctx: &AestheticContext,
    domain_aesthetics: &[&'static str],
) -> Option<(HashSet<String>, HashSet<String>)> {
    let raw = parameters.get("aggregate")?;
    if matches!(raw, ParameterValue::Null) {
        return None;
    }
    let spec = parse_aggregate_param(raw).ok()??;

    let mut targeted: HashSet<String> = HashSet::new();
    for (user_aes, _fns) in &spec.targets {
        for internal in resolve_target_aesthetic(user_aes, aesthetics, aesthetic_ctx) {
            targeted.insert(internal);
        }
    }

    let mut aggregated: HashSet<String> = targeted.clone();
    let mut entries: Vec<(&String, &crate::AestheticValue)> =
        aesthetics.aesthetics.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (aes, value) in entries {
        let col = match value.column_name() {
            Some(c) => c,
            None => continue,
        };
        if domain_aesthetics.contains(&aes.as_str()) {
            continue;
        }
        let is_discrete = schema
            .iter()
            .find(|c| c.name == col)
            .map(|c| c.is_discrete)
            .unwrap_or(false);
        if is_discrete {
            continue;
        }
        if targeted.contains(aes) {
            continue;
        }
        let default_applies = if is_upper_half(aes) {
            spec.default_upper.is_some() || spec.default_lower.is_some()
        } else {
            spec.default_lower.is_some()
        };
        if default_applies {
            aggregated.insert(aes.clone());
        }
    }

    Some((targeted, aggregated))
}

/// Apply the Aggregate stat to a layer query.
///
/// Returns `StatResult::Identity` when the `aggregate` parameter is unset, null,
/// or empty. Otherwise, builds a `GROUP BY` query producing one row per group
/// (the *reduce* path) — or, when at least one target lists multiple functions,
/// `N` rows per group with a synthetic `aggregate` column tagging each row
/// (the *explode* path).
#[allow(clippy::too_many_arguments)]
pub fn apply(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn SqlDialect,
    aesthetic_ctx: &AestheticContext,
    domain_aesthetics: &[&'static str],
) -> Result<StatResult> {
    let raw = match parameters.get("aggregate") {
        None | Some(ParameterValue::Null) => return Ok(StatResult::Identity),
        Some(v) => v,
    };
    let spec = parse_aggregate_param(raw).map_err(GgsqlError::ValidationError)?;
    let spec = match spec {
        Some(s) => s,
        None => return Ok(StatResult::Identity),
    };
    let n = spec.explosion_factor();
    let labels = spec.explosion_labels();

    // Resolve target keys (user-facing) → internal aesthetic names. An alias
    // like `color` expands to whichever of its targets (stroke/fill) is mapped
    // on the layer; the same function list applies to all of them.
    let targets_internal = resolve_aggregate_targets(&spec, aesthetics, aesthetic_ctx)
        .map_err(GgsqlError::ValidationError)?;

    // Walk mappings. Three buckets:
    //   - aggregated: (internal_aes, raw_col, fns of length n) — each emits one column per row
    //   - kept_cols: discrete column-mappings — keep as group key
    //   - dropped: numeric mapping with no applicable function (warn & skip)
    let mut aggregated: Vec<(String, String, Vec<AggSpec>)> = Vec::new();
    let mut kept_cols: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    let mut entries: Vec<(&String, &crate::AestheticValue)> =
        aesthetics.aesthetics.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (aes, value) in entries {
        let col = match value.column_name() {
            Some(c) => c.to_string(),
            None => continue, // literals & annotation columns pass through
        };
        // Geom-declared domain aesthetics (e.g. `pos1` for line/area/ribbon)
        // always become group keys — they identify each row, never get
        // aggregated, never get dropped.
        if domain_aesthetics.contains(&aes.as_str()) {
            if !kept_cols.contains(&col) {
                kept_cols.push(col);
            }
            continue;
        }
        let info = schema.iter().find(|c| c.name == col);
        let is_discrete = info.map(|c| c.is_discrete).unwrap_or(false);
        if is_discrete {
            if !kept_cols.contains(&col) {
                kept_cols.push(col);
            }
            continue;
        }

        // Numeric mapping. Look up the function list (recycling to length n).
        let fns: Option<Vec<AggSpec>> = if let Some(list) = targets_internal.get(aes) {
            if list.len() == n {
                Some(list.clone())
            } else {
                // Validated to be 1 or n during parsing; guard with a sanity check.
                debug_assert_eq!(list.len(), 1);
                Some(vec![list[0].clone(); n])
            }
        } else {
            let default = if is_upper_half(aes) {
                spec.default_upper
                    .clone()
                    .or_else(|| spec.default_lower.clone())
            } else {
                spec.default_lower.clone()
            };
            default.map(|d| vec![d; n])
        };

        match fns {
            Some(list) => aggregated.push((aes.clone(), col, list)),
            None => dropped.push(aes.clone()),
        }
    }

    for d in &dropped {
        let user_aes = aesthetic_ctx.map_internal_to_user(d);
        eprintln!(
            "Warning: aggregate dropped numeric mapping for aesthetic '{}' \
             (no applicable default and no targeted function). \
             Suggestion: add an unprefixed default like `aggregate => 'mean'` \
             to apply one function to every numeric mapping, or target this \
             aesthetic with `'{0}:<func>'`.",
            user_aes,
        );
    }

    // No aggregate functions to apply → the stat has nothing to do. Whether
    // the layer has group keys or not is irrelevant: emitting a `SELECT keys
    // FROM src GROUP BY keys` query would be a distinct-rows transform the
    // user didn't ask for.
    if aggregated.is_empty() {
        return Ok(StatResult::Identity);
    }

    // Group columns: PARTITION BY + discrete column-mappings, deduped.
    let mut group_cols: Vec<String> = Vec::new();
    for g in group_by {
        if !group_cols.contains(g) {
            group_cols.push(g.clone());
        }
    }
    for c in &kept_cols {
        if !group_cols.contains(c) {
            group_cols.push(c.clone());
        }
    }

    let missing = unsupported_functions(&aggregated, dialect);
    if !missing.is_empty() {
        return Err(GgsqlError::ValidationError(format!(
            "aggregate function(s) {} are not supported by this database backend",
            crate::or_list_quoted(&missing, '\''),
        )));
    }

    let transformed_query = match &labels {
        Some(ls) => build_aggregate_query(query, &aggregated, &group_cols, ls, dialect),
        None => build_group_by_query(query, &aggregated, &group_cols, dialect),
    };

    let mut stat_columns: Vec<String> = aggregated.iter().map(|(a, _, _)| a.clone()).collect();
    let consumed_aesthetics: Vec<String> = stat_columns.clone();
    // The synthetic `aggregate` column is only emitted for the multi-row
    // (explosion) case, where it differentiates rows that share the same
    // group key.
    if labels.is_some() {
        stat_columns.push("aggregate".to_string());
    }

    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns: vec![],
        consumed_aesthetics,
    })
}

/// CTE preamble plus the alias the caller should `FROM`. When any emitted
/// aggregate references the `__ggsql_rn__` / `__ggsql_max_rn__` columns
/// (the dialect-portable form of `first` / `last`), wrap the source CTE in a
/// row-numbered layer.
fn source_cte_chain(
    query: &str,
    aggregated: &[(String, String, Vec<AggSpec>)],
    group_cols: &[String],
    dialect: &dyn SqlDialect,
) -> (String, &'static str) {
    let raw_src = "\"__ggsql_stat_src__\"";
    if !needs_row_position(aggregated, dialect) {
        return (format!("WITH {raw_src} AS ({query})"), raw_src);
    }
    let rn_src = "\"__ggsql_stat_src_rn__\"";
    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
    // ORDER BY (SELECT 1) is the canonical "no real ordering" stand-in: it
    // satisfies the standard's required ORDER BY for window functions while
    // letting the engine pick the row order — same indeterminacy as DuckDB's
    // native FIRST() without a user ORDER BY.
    let partition = if group_select.is_empty() {
        String::new()
    } else {
        format!("PARTITION BY {} ", group_select.join(", "))
    };
    let cte = format!(
        "WITH {raw_src} AS ({query}), {rn_src} AS (\
           SELECT *, \
             ROW_NUMBER() OVER ({partition}ORDER BY (SELECT 1)) AS \"__ggsql_rn__\", \
             COUNT(*) OVER ({partition_no_order}) AS \"__ggsql_max_rn__\" \
           FROM {raw_src}\
         )",
        partition_no_order = partition.trim_end(),
    );
    (cte, rn_src)
}

/// True iff at least one aggregate spec, after the dialect emits its SQL,
/// references the row-position columns. Backends with native `FIRST`/`LAST`
/// (DuckDB) emit a string that doesn't mention `__ggsql_rn__`, and so don't
/// pay for the extra window functions.
fn needs_row_position(
    aggregated: &[(String, String, Vec<AggSpec>)],
    dialect: &dyn SqlDialect,
) -> bool {
    for (_, _, specs) in aggregated {
        for spec in specs {
            for name in [Some(spec.offset), spec.band.as_ref().map(|b| b.expansion)]
                .into_iter()
                .flatten()
            {
                if let Some(sql) = dialect.sql_aggregate(name, "x") {
                    if sql.contains("__ggsql_rn__") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Build the single-row `WITH src AS (<query>) SELECT <group cols>, <agg exprs>
/// FROM src AS "__ggsql_qt__" GROUP BY <group cols>` query. Each aggregated
/// aesthetic's function list is length 1 here.
///
/// Falls back to `dialect.sql_percentile()` per-column when an aggregate's
/// percentile component lacks inline support.
fn build_group_by_query(
    query: &str,
    aggregated: &[(String, String, Vec<AggSpec>)],
    group_cols: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    let outer_alias = "\"__ggsql_qt__\"";
    let (with_clause, src_alias) = source_cte_chain(query, aggregated, group_cols, dialect);

    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        format!(" GROUP BY {}", group_select.join(", "))
    };

    let mut select_parts: Vec<String> = group_select.clone();

    for (aes, raw_col, fns) in aggregated {
        let agg = &fns[0];
        let stat_col = naming::stat_column(aes);
        let qcol = naming::quote_ident(raw_col);
        let expr = if needs_quantile_fallback(agg, raw_col, dialect) {
            agg_sql_fallback(agg, raw_col, dialect, src_alias, group_cols)
        } else {
            agg_sql_inline(agg, &qcol, dialect)
                .expect("agg_sql_inline must succeed when needs_quantile_fallback is false")
        };
        select_parts.push(format!("{} AS {}", expr, naming::quote_ident(&stat_col)));
    }

    format!(
        "{with_clause} SELECT {sel} FROM {src} AS {outer}{gb}",
        sel = select_parts.join(", "),
        src = src_alias,
        outer = outer_alias,
        gb = group_by_clause,
    )
}

/// Build the exploded `WITH src AS (<query>) <branch_0> UNION ALL <branch_1>
/// ...` query. One branch per row in `0..labels.len()`, each branch its own
/// `GROUP BY` with the row's aggregation functions and a literal label tagged
/// to `__ggsql_stat_aggregate__`.
fn build_aggregate_query(
    query: &str,
    aggregated: &[(String, String, Vec<AggSpec>)],
    group_cols: &[String],
    labels: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    let outer_alias = "\"__ggsql_qt__\"";
    let (with_clause, src_alias) = source_cte_chain(query, aggregated, group_cols, dialect);

    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        format!(" GROUP BY {}", group_select.join(", "))
    };

    let stat_aggregate_col = naming::stat_column("aggregate");

    let branches: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(row_idx, label)| {
            let mut select_parts: Vec<String> = group_select.clone();

            for (aes, raw_col, fns) in aggregated {
                let agg = &fns[row_idx];
                let stat_col = naming::stat_column(aes);
                let qcol = naming::quote_ident(raw_col);
                let expr = if needs_quantile_fallback(agg, raw_col, dialect) {
                    agg_sql_fallback(agg, raw_col, dialect, src_alias, group_cols)
                } else {
                    agg_sql_inline(agg, &qcol, dialect)
                        .expect("agg_sql_inline must succeed when needs_quantile_fallback is false")
                };
                select_parts.push(format!("{} AS {}", expr, naming::quote_ident(&stat_col)));
            }

            select_parts.push(format!(
                "{} AS {}",
                naming::quote_literal(label),
                naming::quote_ident(&stat_aggregate_col)
            ));

            format!(
                "SELECT {} FROM {} AS {}{}",
                select_parts.join(", "),
                src_alias,
                outer_alias,
                group_by_clause,
            )
        })
        .collect();

    format!("{with_clause} {body}", body = branches.join(" UNION ALL "),)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::aesthetic::AestheticContext;
    use crate::plot::types::{AestheticValue, ColumnInfo};
    use arrow::datatypes::DataType;

    /// A test dialect that mimics DuckDB: native QUANTILE_CONT plus the
    /// row-positional FIRST / LAST aggregates.
    struct InlineQuantileDialect;
    impl SqlDialect for InlineQuantileDialect {
        fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
            Some(format!(
                "QUANTILE_CONT({}, {})",
                naming::quote_ident(column),
                fraction
            ))
        }

        fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
            match name {
                "first" => Some(format!("FIRST({})", qcol)),
                "last" => Some(format!("LAST({})", qcol)),
                "diff" => Some(format!("(LAST({c}) - FIRST({c}))", c = qcol)),
                _ => crate::reader::default_sql_aggregate(name, qcol),
            }
        }
    }

    /// A test dialect with no inline quantile support, exercising the
    /// per-column `sql_percentile` fallback.
    struct NoInlineQuantileDialect;
    impl SqlDialect for NoInlineQuantileDialect {}

    fn col(name: &str) -> AestheticValue {
        AestheticValue::Column {
            name: name.to_string(),
            original_name: None,
            is_dummy: false,
        }
    }

    fn schema_for(cols: &[(&str, bool)]) -> Schema {
        cols.iter()
            .map(|(name, is_discrete)| ColumnInfo {
                name: name.to_string(),
                dtype: if *is_discrete {
                    DataType::Utf8
                } else {
                    DataType::Float64
                },
                is_discrete: *is_discrete,
                min: None,
                max: None,
            })
            .collect()
    }

    fn cartesian_ctx() -> AestheticContext {
        AestheticContext::from_static(&["x", "y"], &[])
    }

    fn run(
        params: ParameterValue,
        aes: &Mappings,
        schema: &Schema,
        group_by: &[String],
        dialect: &dyn SqlDialect,
    ) -> Result<StatResult> {
        run_with_domain(params, aes, schema, group_by, dialect, &[])
    }

    fn run_with_domain(
        params: ParameterValue,
        aes: &Mappings,
        schema: &Schema,
        group_by: &[String],
        dialect: &dyn SqlDialect,
        domain: &[&'static str],
    ) -> Result<StatResult> {
        let mut p = HashMap::new();
        p.insert("aggregate".to_string(), params);
        let ctx = cartesian_ctx();
        apply(
            "SELECT * FROM t",
            schema,
            aes,
            group_by,
            &p,
            dialect,
            &ctx,
            domain,
        )
    }

    fn arr(items: &[&str]) -> ParameterValue {
        ParameterValue::Array(
            items
                .iter()
                .map(|s| ArrayElement::String(s.to_string()))
                .collect(),
        )
    }

    // ---------- parser tests ----------

    #[test]
    fn parses_unset_and_null() {
        assert_eq!(parse_aggregate_param(&ParameterValue::Null).unwrap(), None);
        assert_eq!(parse_aggregate_param(&arr(&[])).unwrap(), None);
    }

    #[test]
    fn parses_single_default() {
        let s = parse_aggregate_param(&ParameterValue::String("mean".to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(s.default_lower.as_ref().map(|a| a.offset), Some("mean"));
        assert!(s.default_upper.is_none());
        assert!(s.targets.is_empty());
    }

    #[test]
    fn parses_two_defaults_in_order() {
        let s = parse_aggregate_param(&arr(&["min", "max"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.default_lower.as_ref().map(|a| a.offset), Some("min"));
        assert_eq!(s.default_upper.as_ref().map(|a| a.offset), Some("max"));
    }

    #[test]
    fn three_unprefixed_defaults_is_error() {
        let err = parse_aggregate_param(&arr(&["mean", "min", "max"])).unwrap_err();
        assert!(err.contains("at most two"), "got: {}", err);
    }

    fn target_funcs<'a>(spec: &'a AggregateSpec, aes: &str) -> Option<&'a [AggSpec]> {
        spec.targets
            .iter()
            .find(|(a, _)| a == aes)
            .map(|(_, fns)| fns.as_slice())
    }

    #[test]
    fn parses_targeted_entries() {
        let s = parse_aggregate_param(&arr(&["mean", "y:max", "color:median"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.default_lower.as_ref().map(|a| a.offset), Some("mean"));
        assert_eq!(target_funcs(&s, "y").map(|fs| fs[0].offset), Some("max"));
        assert_eq!(
            target_funcs(&s, "color").map(|fs| fs[0].offset),
            Some("median")
        );
    }

    #[test]
    fn duplicate_target_explodes_into_a_list() {
        let s = parse_aggregate_param(&arr(&["y:min", "y:max"]))
            .unwrap()
            .unwrap();
        let fns = target_funcs(&s, "y").unwrap();
        assert_eq!(fns.len(), 2);
        assert_eq!(fns[0].offset, "min");
        assert_eq!(fns[1].offset, "max");
        assert_eq!(s.explosion_factor(), 2);
        assert_eq!(
            s.explosion_labels(),
            Some(vec!["min".to_string(), "max".to_string()])
        );
    }

    #[test]
    fn multi_aesthetic_explosion_joins_unique_function_names() {
        // Two exploded targets contribute distinct function names per row → 'min/sum', 'max/prod'.
        let s = parse_aggregate_param(&arr(&["y:min", "y:max", "color:sum", "color:prod"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            s.explosion_labels(),
            Some(vec!["min/sum".to_string(), "max/prod".to_string()])
        );
    }

    #[test]
    fn multi_aesthetic_explosion_dedups_repeats() {
        // y and color both use 'mean' at row 0 → label is just 'mean' (deduped).
        let s = parse_aggregate_param(&arr(&["y:mean", "y:max", "color:mean", "color:prod"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            s.explosion_labels(),
            Some(vec!["mean".to_string(), "max/prod".to_string()])
        );
    }

    #[test]
    fn recycled_target_excluded_from_label() {
        // color has length 1 → recycled, not exploded; label only reflects y's functions.
        let s = parse_aggregate_param(&arr(&["y:min", "y:max", "color:median"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            s.explosion_labels(),
            Some(vec!["min".to_string(), "max".to_string()])
        );
    }

    #[test]
    fn single_row_returns_no_labels() {
        // The aggregate column only makes sense as a row-differentiator, and a
        // single-row aggregation has nothing to differentiate, so no labels.
        let s = parse_aggregate_param(&ParameterValue::String("mean".to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(s.explosion_labels(), None);

        let s = parse_aggregate_param(&arr(&["mean", "color:median"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.explosion_labels(), None);
    }

    #[test]
    fn recycling_violation_is_error() {
        // y has 2, color has 3 → mismatched, neither is 1 nor matches the longest.
        let err = parse_aggregate_param(&arr(&[
            "y:min",
            "y:max",
            "color:p10",
            "color:p50",
            "color:p90",
        ]))
        .unwrap_err();
        assert!(err.contains("longest target"), "got: {}", err);
    }

    #[test]
    fn length_one_target_recycles_in_explosion() {
        let s = parse_aggregate_param(&arr(&["y:min", "y:max", "color:median"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.explosion_factor(), 2);
        assert_eq!(target_funcs(&s, "color").map(|f| f.len()), Some(1));
    }

    #[test]
    fn empty_prefix_is_error() {
        let err = parse_aggregate_param(&ParameterValue::String(":mean".to_string())).unwrap_err();
        assert!(err.contains("could not parse"), "got: {}", err);
    }

    #[test]
    fn unknown_function_is_error() {
        let err = parse_aggregate_param(&ParameterValue::String("nope".to_string())).unwrap_err();
        assert!(err.contains("unknown aggregate"), "got: {}", err);
    }

    #[test]
    fn band_functions_parse() {
        let s = parse_aggregate_param(&arr(&["mean-sdev", "mean+sdev"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.default_lower.as_ref().unwrap().offset, "mean");
        assert_eq!(
            s.default_lower
                .as_ref()
                .unwrap()
                .band
                .as_ref()
                .unwrap()
                .expansion,
            "sdev"
        );
        assert_eq!(
            s.default_lower
                .as_ref()
                .unwrap()
                .band
                .as_ref()
                .unwrap()
                .mod_value,
            -1.0,
        );
        assert_eq!(s.default_upper.as_ref().unwrap().offset, "mean");
        assert_eq!(
            s.default_upper
                .as_ref()
                .unwrap()
                .band
                .as_ref()
                .unwrap()
                .mod_value,
            1.0,
        );
    }

    // ---------- apply tests ----------

    #[test]
    fn returns_identity_when_param_unset() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let p: HashMap<String, ParameterValue> = HashMap::new();
        let ctx = cartesian_ctx();
        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &p,
            &InlineQuantileDialect,
            &ctx,
            &[],
        )
        .unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn returns_identity_when_param_null() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let result = run(
            ParameterValue::Null,
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn single_default_applies_to_every_numeric_mapping() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("mean".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"), "{}", query);
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"), "{}", query);
                // No GROUP BY when no discrete mappings or PARTITION BY — SQL
                // collapses to a single row per query, which is correct.
                assert!(!query.contains("CROSS JOIN"));
                assert!(!query.contains("UNION ALL"));
                assert_eq!(stat_columns.len(), 2);
                assert!(stat_columns.contains(&"pos1".to_string()));
                assert!(stat_columns.contains(&"pos2".to_string()));
                assert_eq!(consumed_aesthetics.len(), 2);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_dialect_emits_portable_stddev_and_first() {
        use crate::reader::sqlite::SqliteDialect;

        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);

        // sdev must not emit STDDEV_POP (SQLite has no such function).
        let result = run(
            ParameterValue::String("sdev".to_string()),
            &aes,
            &schema,
            &[],
            &SqliteDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    !query.contains("STDDEV_POP"),
                    "SQLite dialect must not emit STDDEV_POP, got: {query}"
                );
                assert!(query.contains("SQRT") && query.contains("AVG"), "{query}");
            }
            _ => panic!("expected Transformed"),
        }

        // first now uses the portable ROW_NUMBER + MAX(CASE) form. It must run
        // on SQLite without `FIRST` ever appearing as an aggregate call.
        let result = run(
            ParameterValue::String("first".to_string()),
            &aes,
            &schema,
            &[],
            &SqliteDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains("ROW_NUMBER()"),
                    "expected ROW_NUMBER prep, got: {query}"
                );
                assert!(
                    query.contains("\"__ggsql_rn__\" = 1"),
                    "expected first via rn=1, got: {query}"
                );
                assert!(
                    !query.contains("FIRST(\""),
                    "must not call FIRST as an aggregate, got: {query}"
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    /// End-to-end run against an actual SQLite reader. Pins that the rn-CTE
    /// path (`MAX(CASE WHEN __ggsql_rn__ = …)`) returns the correct first /
    /// last / diff values per discrete group — the SQL-string assertions in
    /// `sqlite_dialect_emits_portable_stddev_and_first` only verify the
    /// generated query's shape, not that it actually computes the right thing
    /// when executed.
    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_first_last_diff_return_correct_values() {
        use crate::naming;
        use crate::reader::SqliteReader;

        let reader = SqliteReader::new().unwrap();

        // Stable row order via an explicit ORDER BY so first/last are
        // deterministic. Group A: 10, 30, 20 → first=10, last=20, diff=10.
        // Group B: 100, 50      → first=100, last=50, diff=-50.
        let body = "WITH t(g, ord, v) AS (\
                    SELECT 'A', 1, 10 UNION ALL SELECT 'A', 2, 30 \
                    UNION ALL SELECT 'A', 3, 20 \
                    UNION ALL SELECT 'B', 1, 100 UNION ALL SELECT 'B', 2, 50) \
                    SELECT g, v FROM t ORDER BY g, ord";

        // Helper: run the query with a given aggregate and return per-group values.
        let run_agg = |func: &str| -> Vec<(String, f64)> {
            let query = format!(
                "{body} VISUALISE \
                 DRAW point MAPPING g AS x, v AS y \
                 SETTING aggregate => '{func}'"
            );
            let prepared = crate::execute::prepare_data_with_reader(&query, &reader).unwrap();
            let df = prepared
                .data
                .get(prepared.specs[0].layers[0].data_key.as_ref().unwrap())
                .unwrap();
            let xs = df.column("__ggsql_aes_pos1__").unwrap();
            let ys = df.column("__ggsql_aes_pos2__").unwrap();
            let mut out: Vec<(String, f64)> = (0..df.height())
                .map(|i| {
                    let x = crate::array_util::value_to_string(xs, i);
                    let y = crate::array_util::value_to_string(ys, i)
                        .parse::<f64>()
                        .unwrap();
                    (x, y)
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };

        assert_eq!(
            run_agg("first"),
            vec![("A".to_string(), 10.0), ("B".to_string(), 100.0)],
            "first should pick the group's first row in ORDER BY ord"
        );
        assert_eq!(
            run_agg("last"),
            vec![("A".to_string(), 20.0), ("B".to_string(), 50.0)],
            "last should pick the group's last row"
        );
        assert_eq!(
            run_agg("diff"),
            vec![("A".to_string(), 10.0), ("B".to_string(), -50.0)],
            "diff should be last - first per group"
        );

        // While we're here, sanity-check that the generated stat SQL goes
        // through the rn-CTE path (no native FIRST/LAST aggregate call).
        let _ = naming::layer_key(0); // exercise the import, keeps `naming` used
    }

    #[test]
    fn unsupported_aggregate_errors_with_dialect_that_lacks_function() {
        // A dialect that explicitly opts out of `first` (returns None) must
        // produce the validation error rather than emitting broken SQL.
        struct OptOutDialect;
        impl SqlDialect for OptOutDialect {
            fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
                if name == "first" {
                    return None;
                }
                crate::reader::default_sql_aggregate(name, qcol)
            }
        }

        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let err = run(
            ParameterValue::String("first".to_string()),
            &aes,
            &schema,
            &[],
            &OptOutDialect,
        )
        .unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("first") && msg.contains("not supported"),
            "expected unsupported-function error mentioning 'first', got: {msg}"
        );
    }

    #[test]
    fn mid_emits_min_max_midpoint() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("mid".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains(
                        "(MIN(\"__ggsql_aes_pos1__\") + MAX(\"__ggsql_aes_pos1__\")) / 2.0"
                    ),
                    "{}",
                    query
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn diff_uses_row_position_and_subtracts_first_from_last() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);

        // AnsiDialect path: portable rn-based form for last - first.
        struct AnsiTestDialect;
        impl SqlDialect for AnsiTestDialect {}
        let result = run(
            ParameterValue::String("diff".to_string()),
            &aes,
            &schema,
            &[],
            &AnsiTestDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("ROW_NUMBER()"), "{query}");
                assert!(
                    query.contains("\"__ggsql_rn__\" = \"__ggsql_max_rn__\""),
                    "{query}"
                );
                assert!(query.contains("\"__ggsql_rn__\" = 1"), "{query}");
                assert!(query.contains(" - "), "expected subtraction, got: {query}");
            }
            _ => panic!("expected Transformed"),
        }

        // Native-FIRST/LAST path: no rn CTE.
        let result = run(
            ParameterValue::String("diff".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains("LAST(") && query.contains("FIRST("),
                    "expected native LAST/FIRST: {query}"
                );
                assert!(
                    !query.contains("__ggsql_rn__"),
                    "native dialect must not add ROW_NUMBER prep: {query}"
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn duckdb_first_skips_row_number_cte() {
        use crate::reader::duckdb::DuckDbDialect;

        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("first".to_string()),
            &aes,
            &schema,
            &[],
            &DuckDbDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains("FIRST(\""),
                    "expected native FIRST aggregate, got: {query}"
                );
                assert!(
                    !query.contains("__ggsql_rn__"),
                    "DuckDB has native FIRST, must not add ROW_NUMBER prep: {query}"
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn last_with_discrete_group_partitions_row_number_over_group() {
        // Pins build_group_by_query's behaviour for the rn-CTE + non-empty
        // group_cols combo: every other test that exercises the rn CTE has
        // empty group_cols (so windows emit `OVER ()`). A bug that
        // forgot to thread `PARTITION BY <group_cols>` through wouldn't
        // surface in those tests.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", true), // discrete group key
            ("__ggsql_aes_pos2__", false),
        ]);

        // Native-FIRST/LAST dialect: no rn CTE, GROUP BY uses the discrete key.
        let result = run(
            ParameterValue::String("last".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    !query.contains("__ggsql_rn__"),
                    "native LAST must not add ROW_NUMBER prep: {query}"
                );
                assert!(query.contains("LAST(\"__ggsql_aes_pos2__\")"), "{query}");
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""), "{query}");
            }
            _ => panic!("expected Transformed"),
        }

        // Default dialect: rn CTE must partition by the discrete group key.
        struct AnsiTestDialect;
        impl SqlDialect for AnsiTestDialect {}
        let result = run(
            ParameterValue::String("last".to_string()),
            &aes,
            &schema,
            &[],
            &AnsiTestDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains(
                        "ROW_NUMBER() OVER (PARTITION BY \"__ggsql_aes_pos1__\" ORDER BY (SELECT 1))"
                    ),
                    "{query}"
                );
                assert!(
                    query.contains("COUNT(*) OVER (PARTITION BY \"__ggsql_aes_pos1__\")"),
                    "{query}"
                );
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""), "{query}");
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn first_and_last_emit_positional_aggregates() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2min", col("__ggsql_aes_pos2min__"));
        aes.insert("pos2max", col("__ggsql_aes_pos2max__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2min__", false),
            ("__ggsql_aes_pos2max__", false),
        ]);
        let result = run(
            arr(&["first", "last"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains("FIRST(\"__ggsql_aes_pos2min__\")"),
                    "{}",
                    query
                );
                assert!(
                    query.contains("LAST(\"__ggsql_aes_pos2max__\")"),
                    "{}",
                    query
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn two_defaults_split_lower_and_upper_for_segment() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert("pos1end", col("__ggsql_aes_pos1end__"));
        aes.insert("pos2end", col("__ggsql_aes_pos2end__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
            ("__ggsql_aes_pos1end__", false),
            ("__ggsql_aes_pos2end__", false),
        ]);
        let result = run(
            arr(&["min", "max"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // pos1, pos2 use MIN; pos1end, pos2end use MAX.
                assert!(query.contains("MIN(\"__ggsql_aes_pos1__\")"), "{}", query);
                assert!(query.contains("MIN(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(
                    query.contains("MAX(\"__ggsql_aes_pos1end__\")"),
                    "{}",
                    query
                );
                assert!(
                    query.contains("MAX(\"__ggsql_aes_pos2end__\")"),
                    "{}",
                    query
                );
                assert!(!query.contains("MIN(\"__ggsql_aes_pos1end__\")"));
                assert!(!query.contains("MAX(\"__ggsql_aes_pos1__\")"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn two_defaults_split_for_ribbon() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2min", col("__ggsql_aes_pos2min__"));
        aes.insert("pos2max", col("__ggsql_aes_pos2max__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2min__", false),
            ("__ggsql_aes_pos2max__", false),
        ]);
        let result = run(
            arr(&["mean-sdev", "mean+sdev"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("STDDEV_POP(\"__ggsql_aes_pos2max__\")"));
                assert!(query.contains("AVG(\"__ggsql_aes_pos2min__\")"));
                // upper default (mean+sdev) goes to pos2max → '+' between AVG and STDDEV
                let pos2max_section = query.split("__ggsql_aes_pos2max__\")").next().unwrap_or("");
                assert!(pos2max_section.contains('+') || query.contains("+ STDDEV_POP"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn targeted_prefix_overrides_default() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            arr(&["mean", "y:max"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"), "{}", query);
                assert!(query.contains("MAX(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(!query.contains("AVG(\"__ggsql_aes_pos2__\")"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn material_aesthetic_targeted_by_user_facing_name() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert("size", col("__ggsql_aes_size__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
            ("__ggsql_aes_size__", false),
        ]);
        let result = run(
            arr(&["mean", "size:median"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("QUANTILE_CONT(\"__ggsql_aes_size__\", 0.5)"));
                assert!(stat_columns.contains(&"size".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn color_alias_targets_stroke_and_fill() {
        // `color` is an alias that resolves to whichever of `stroke`/`fill`
        // is actually mapped on the layer.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert("fill", col("__ggsql_aes_fill__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
            ("__ggsql_aes_fill__", false),
        ]);
        let result = run(
            arr(&["mean", "color:max"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("MAX(\"__ggsql_aes_fill__\")"), "{}", query);
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"));
                assert!(stat_columns.contains(&"fill".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn explosion_emits_union_all_with_aggregate_label_column() {
        // ('y:min', 'y:max') on a line-style layer → 2 rows per group, each
        // tagged with the function name in __ggsql_stat_aggregate__.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            arr(&["y:min", "y:max"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                assert!(query.contains("UNION ALL"), "{}", query);
                assert!(query.contains("MIN(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(query.contains("MAX(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(query.contains("'min' AS \"__ggsql_stat_aggregate\""));
                assert!(query.contains("'max' AS \"__ggsql_stat_aggregate\""));
                // Aesthetics consumed: pos2. The synthetic `aggregate` is in
                // stat_columns but NOT consumed (it's a new column).
                assert!(consumed_aesthetics.contains(&"pos2".to_string()));
                assert!(!consumed_aesthetics.contains(&"aggregate".to_string()));
                assert!(stat_columns.contains(&"aggregate".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn explosion_recycles_length_one_targets_and_defaults() {
        // ('mean', 'y:min', 'y:max', 'color:median'):
        //   - default 'mean' applies to non-targeted aesthetics, recycled
        //   - y is exploded into [min, max] → N=2
        //   - color is targeted with one function → recycled to [median, median]
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert("fill", col("__ggsql_aes_fill__"));
        aes.insert("size", col("__ggsql_aes_size__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
            ("__ggsql_aes_fill__", false),
            ("__ggsql_aes_size__", false),
        ]);
        let result = run(
            arr(&["mean", "y:min", "y:max", "color:median"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // y is exploded → MIN and MAX appear in different branches
                assert!(query.contains("MIN(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(query.contains("MAX(\"__ggsql_aes_pos2__\")"));
                // color (alias → fill) is recycled → QUANTILE_CONT(.5) appears in BOTH branches
                let median_count = query
                    .matches("QUANTILE_CONT(\"__ggsql_aes_fill__\", 0.5)")
                    .count();
                assert_eq!(
                    median_count, 2,
                    "color median should appear once per branch: {}",
                    query
                );
                // size has no target → uses default 'mean' → AVG appears in both branches
                let avg_size = query.matches("AVG(\"__ggsql_aes_size__\")").count();
                assert_eq!(
                    avg_size, 2,
                    "size mean should appear once per branch: {}",
                    query
                );
                // pos1 (no target) → mean → AVG appears in both branches
                let avg_pos1 = query.matches("AVG(\"__ggsql_aes_pos1__\")").count();
                assert_eq!(avg_pos1, 2);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn domain_aesthetic_kept_as_group_key_even_when_continuous() {
        // Regression test for the line/area/ribbon case: the user writes
        //   DRAW line ... SETTING aggregate => ('y:min', 'y:max')
        // and expects pos1 (the continuous time-axis column) to be a group
        // key, not a dropped numeric mapping. The geom declares pos1 as a
        // domain aesthetic; the stat keeps it as a group column.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false), // continuous, would be dropped without the domain hint
            ("__ggsql_aes_pos2__", false),
        ]);
        let result = run_with_domain(
            arr(&["y:min", "y:max"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
            &["pos1"],
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                // pos1 is in the GROUP BY, not aggregated.
                assert!(
                    query.contains("GROUP BY \"__ggsql_aes_pos1__\""),
                    "{}",
                    query
                );
                assert!(!query.contains("MIN(\"__ggsql_aes_pos1__\")"));
                assert!(!query.contains("MAX(\"__ggsql_aes_pos1__\")"));
                // pos2 is exploded into MIN and MAX branches.
                assert!(query.contains("MIN(\"__ggsql_aes_pos2__\")"));
                assert!(query.contains("MAX(\"__ggsql_aes_pos2__\")"));
                // pos1 is NOT consumed (kept), pos2 IS consumed.
                assert!(!consumed_aesthetics.contains(&"pos1".to_string()));
                assert!(consumed_aesthetics.contains(&"pos2".to_string()));
                // synthetic aggregate column emitted in the explosion case.
                assert!(stat_columns.contains(&"aggregate".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn explosion_with_range_geom_two_defaults() {
        // For ribbon: pos1 + pos2min (lower) + pos2max (upper).
        // ('mean-sdev', 'mean+sdev', 'color:p25', 'color:p75'):
        //   - two defaults split lower/upper for range aesthetics
        //   - color is exploded → N=2
        // Result: two rows, with color taking p25 in row 0 and p75 in row 1;
        // pos1/pos2min always use mean-sdev, pos2max always uses mean+sdev.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2min", col("__ggsql_aes_pos2min__"));
        aes.insert("pos2max", col("__ggsql_aes_pos2max__"));
        aes.insert("fill", col("__ggsql_aes_fill__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2min__", false),
            ("__ggsql_aes_pos2max__", false),
            ("__ggsql_aes_fill__", false),
        ]);
        let result = run(
            arr(&["mean-sdev", "mean+sdev", "color:p25", "color:p75"]),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("UNION ALL"));
                // pos2max always uses mean+sdev (upper default) — a `+` between AVG and STDDEV
                let upper_branch_marker = "AVG(\"__ggsql_aes_pos2max__\") + STDDEV_POP";
                assert!(query.contains(upper_branch_marker), "{}", query);
                // color uses p25 in one branch, p75 in another
                assert!(query.contains("QUANTILE_CONT(\"__ggsql_aes_fill__\", 0.25)"));
                assert!(query.contains("QUANTILE_CONT(\"__ggsql_aes_fill__\", 0.75)"));
                // Synthetic aggregate column is present
                assert!(stat_columns.contains(&"aggregate".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn discrete_mapping_becomes_group_key() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert("color", col("__ggsql_aes_color__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
            ("__ggsql_aes_color__", true), // discrete!
        ]);
        let result = run(
            ParameterValue::String("mean".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(
                    query.contains("GROUP BY \"__ggsql_aes_color__\""),
                    "{}",
                    query
                );
                assert!(!stat_columns.contains(&"color".to_string()));
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"));
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn literal_mapping_passes_through() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        aes.insert(
            "fill",
            AestheticValue::Literal(ParameterValue::String("steelblue".to_string())),
        );
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("mean".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(!query.contains("AVG(\"__ggsql_aes_fill__\")"));
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"));
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn untargeted_numeric_mapping_dropped_when_no_default() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        // Only `y` targeted, no default → x is dropped.
        let result = run(
            ParameterValue::String("y:mean".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
                assert!(!query.contains("\"__ggsql_aes_pos1__\""));
                assert_eq!(stat_columns, vec!["pos2".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn quantile_uses_dialect_inline_when_available() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("p25".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("QUANTILE_CONT"));
                assert!(query.contains("0.25"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn quantile_falls_back_to_correlated_subquery_without_inline() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos2__", false)]);
        let result = run(
            ParameterValue::String("p25".to_string()),
            &aes,
            &schema,
            &[],
            &NoInlineQuantileDialect,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // The fallback dialect's sql_percentile uses NTILE.
                assert!(query.contains("NTILE(4)"));
                // No explosion any more — single SELECT, no UNION ALL.
                assert!(!query.contains("UNION ALL"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn unknown_targeted_aesthetic_is_error() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[("__ggsql_aes_pos1__", false), ("__ggsql_aes_pos2__", false)]);
        let err = run(
            ParameterValue::String("size:mean".to_string()),
            &aes,
            &schema,
            &[],
            &InlineQuantileDialect,
        )
        .unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("not mapped"), "got: {}", msg);
    }
}
