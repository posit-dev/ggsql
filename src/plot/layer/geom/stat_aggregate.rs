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

use std::collections::HashMap;

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
    "min", "max", "range", // Central tendency
    "mean", "geomean", "harmean", "rms", "median", // Spread (standalone)
    "sdev", "var", "iqr", // Percentiles
    "p05", "p10", "p25", "p50", "p75", "p90", "p95",
];

/// Stats that can appear as the *offset* (left of `±`) in a band name like
/// `mean+sdev`. Single-value central or representative quantities only —
/// counts/spreads are excluded.
pub const OFFSET_STATS: &[&str] = &[
    "mean", "median", "geomean", "harmean", "rms", "sum", "prod", "min", "max", "p05", "p10",
    "p25", "p50", "p75", "p90", "p95",
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
    pub sign: char,
    pub mod_value: f64,
    pub expansion: &'static str,
}

fn resolve_static(name: &str, vocab: &'static [&'static str]) -> Option<&'static str> {
    vocab.iter().copied().find(|v| *v == name)
}

/// Parse an aggregate-function name into an `AggSpec`. Returns `None` on
/// invalid input (unknown stat, malformed band, or band with vocabulary
/// violation).
pub fn parse_agg_name(name: &str) -> Option<AggSpec> {
    if let Some(spec) = parse_band(name) {
        return Some(spec);
    }
    resolve_static(name, AGG_NAMES).map(|offset| AggSpec { offset, band: None })
}

fn parse_band(name: &str) -> Option<AggSpec> {
    // Walk offsets longest-first so `median` matches before `mean`.
    let mut offsets: Vec<&'static str> = OFFSET_STATS.to_vec();
    offsets.sort_by_key(|s| std::cmp::Reverse(s.len()));

    for offset in offsets {
        let rest = match name.strip_prefix(offset) {
            Some(r) => r,
            None => continue,
        };
        let (sign, after_sign) = match rest.chars().next() {
            Some('+') => ('+', &rest[1..]),
            Some('-') => ('-', &rest[1..]),
            _ => continue,
        };

        let (mod_value, expansion_str) = parse_mod_and_remainder(after_sign);
        let expansion = match resolve_static(expansion_str, EXPANSION_STATS) {
            Some(e) => e,
            None => continue,
        };

        return Some(AggSpec {
            offset,
            band: Some(Band {
                sign,
                mod_value,
                expansion,
            }),
        });
    }
    None
}

fn parse_mod_and_remainder(s: &str) -> (f64, &str) {
    let mut idx = 0;
    let bytes = s.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    if idx < bytes.len() && bytes[idx] == b'.' {
        let mut after_dot = idx + 1;
        while after_dot < bytes.len() && bytes[after_dot].is_ascii_digit() {
            after_dot += 1;
        }
        if after_dot > idx + 1 {
            idx = after_dot;
        }
    }
    if idx == 0 {
        return (1.0, s);
    }
    let num_str = &s[..idx];
    let value: f64 = num_str.parse().unwrap_or(1.0);
    (value, &s[idx..])
}

// =============================================================================
// AggregateSpec — parsed representation of the `aggregate` SETTING.
// =============================================================================

/// Parsed `aggregate` SETTING: zero, one, or two unprefixed defaults plus an
/// optional set of per-aesthetic targets keyed by user-facing aesthetic name.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateSpec {
    pub default_lower: Option<AggSpec>,
    pub default_upper: Option<AggSpec>,
    /// Targets keyed by user-facing aesthetic name (e.g. `"y"`, `"xmax"`,
    /// `"color"`). Resolved to internal names at apply-time.
    pub targets: HashMap<String, AggSpec>,
}

impl AggregateSpec {
    fn new() -> Self {
        Self {
            default_lower: None,
            default_upper: None,
            targets: HashMap::new(),
        }
    }
}

/// Parse the `aggregate` SETTING value into an `AggregateSpec`. Returns `Ok(None)`
/// when the parameter is unset or null. Returns `Err(...)` for malformed input.
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
        if let Some((aes, func)) = split_target(entry) {
            if aes.is_empty() {
                return Err(format!("'{}': aesthetic prefix is empty", entry));
            }
            if func.is_empty() {
                return Err(format!("'{}': aggregate function is empty", entry));
            }
            let agg = parse_agg_name(func).ok_or_else(|| {
                format!(
                    "'{}': {}",
                    entry,
                    diagnose_invalid_function_name(func)
                )
            })?;
            if spec.targets.contains_key(aes) {
                return Err(format!(
                    "aesthetic '{}' is targeted by more than one aggregate",
                    aes
                ));
            }
            spec.targets.insert(aes.to_string(), agg);
        } else {
            let agg = parse_agg_name(entry)
                .ok_or_else(|| diagnose_invalid_function_name(entry))?;
            if spec.default_lower.is_none() {
                spec.default_lower = Some(agg);
            } else if spec.default_upper.is_none() {
                spec.default_upper = Some(agg);
            } else {
                return Err(format!(
                    "'aggregate' accepts at most two unprefixed defaults; got a third: '{}'",
                    entry
                ));
            }
        }
    }

    if spec.default_lower.is_none() && spec.default_upper.is_none() && spec.targets.is_empty() {
        return Ok(None);
    }
    Ok(Some(spec))
}

/// Split an entry into `(aesthetic, function)` if it contains a `:`. Returns
/// `None` for an unprefixed entry like `'mean'`.
fn split_target(entry: &str) -> Option<(&str, &str)> {
    entry.split_once(':')
}

/// Validate the `aggregate` SETTING value at parse-time. Used by
/// `Layer::validate_settings`. Aesthetic-name resolution is deferred to
/// `apply()` because `AestheticContext` isn't available here.
pub fn validate_aggregate_param(value: &ParameterValue) -> std::result::Result<(), String> {
    parse_aggregate_param(value).map(|_| ())
}

/// Build a per-role error message for a name that didn't parse. Re-walks the
/// input with looser rules to identify which side (offset / expansion) failed.
fn diagnose_invalid_function_name(name: &str) -> String {
    if let Some(sign_idx) = name.find(['+', '-']) {
        let offset_str = &name[..sign_idx];
        let after_sign = &name[sign_idx + 1..];
        let (_mod_value, expansion_str) = parse_mod_and_remainder(after_sign);

        let offset_known_simple = AGG_NAMES.contains(&offset_str);
        let offset_known_band = OFFSET_STATS.contains(&offset_str);
        let expansion_known_band = EXPANSION_STATS.contains(&expansion_str);

        if !offset_known_band {
            if offset_known_simple {
                return format!(
                    "'{}': '{}' is not a valid offset stat. Allowed offsets: {}",
                    name,
                    offset_str,
                    crate::or_list_quoted(OFFSET_STATS, '\''),
                );
            }
            return format!(
                "'{}': '{}' is not a known stat. Allowed offsets: {}",
                name,
                offset_str,
                crate::or_list_quoted(OFFSET_STATS, '\''),
            );
        }
        if !expansion_known_band {
            return format!(
                "'{}': '{}' is not a valid expansion stat. Allowed expansions: {}",
                name,
                expansion_str,
                crate::or_list_quoted(EXPANSION_STATS, '\''),
            );
        }
        return format!("'{}' is not a valid aggregate function name", name);
    }
    format!(
        "unknown aggregate function '{}'. Allowed: {} (or use a band like `mean+sdev`)",
        name,
        crate::or_list_quoted(AGG_NAMES, '\''),
    )
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
/// quoted column. Returns `None` for percentile-based stats when the dialect
/// lacks an inline quantile aggregate (caller switches to the correlated
/// `sql_percentile` fallback).
fn simple_stat_sql_inline(name: &str, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    if name == "count" {
        // `count` in this position is COUNT(col): non-null tally for that column.
        return Some(format!("COUNT({})", qcol));
    }
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
    Some(match name {
        "sum" => format!("SUM({})", qcol),
        "prod" => format!("EXP(SUM(LN({})))", qcol),
        "min" => format!("MIN({})", qcol),
        "max" => format!("MAX({})", qcol),
        "range" => format!("(MAX({c}) - MIN({c}))", c = qcol),
        "mean" => format!("AVG({})", qcol),
        "geomean" => format!("EXP(AVG(LN({})))", qcol),
        "harmean" => format!("(COUNT({c}) * 1.0 / SUM(1.0 / {c}))", c = qcol),
        "rms" => format!("SQRT(AVG({c} * {c}))", c = qcol),
        "sdev" => format!("STDDEV_POP({})", qcol),
        "se" => format!("(STDDEV_POP({c}) / SQRT(COUNT({c})))", c = qcol),
        "var" => format!("VAR_POP({})", qcol),
        _ => return None,
    })
}

fn agg_sql_inline(spec: &AggSpec, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    let offset_sql = simple_stat_sql_inline(spec.offset, qcol, dialect)?;
    match &spec.band {
        None => Some(offset_sql),
        Some(band) => {
            let exp_sql = simple_stat_sql_inline(band.expansion, qcol, dialect)?;
            Some(format_band(
                &offset_sql,
                band.sign,
                band.mod_value,
                &exp_sql,
            ))
        }
    }
}

fn format_band(offset: &str, sign: char, mod_value: f64, exp: &str) -> String {
    if mod_value == 1.0 {
        format!("({} {} {})", offset, sign, exp)
    } else {
        format!("({} {} {} * {})", offset, sign, mod_value, exp)
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
            format_band(&offset_sql, band.sign, band.mod_value, &exp_sql)
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

/// Apply the Aggregate stat to a layer query.
///
/// Returns `StatResult::Identity` when the `aggregate` parameter is unset, null,
/// or empty. Otherwise, builds a single-pass `GROUP BY` query producing one row
/// per group with one aggregated column per kept numeric mapping.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn SqlDialect,
    aesthetic_ctx: &AestheticContext,
) -> Result<StatResult> {
    let raw = match parameters.get("aggregate") {
        None | Some(ParameterValue::Null) => return Ok(StatResult::Identity),
        Some(v) => v,
    };
    let spec = parse_aggregate_param(raw)
        .map_err(GgsqlError::ValidationError)?;
    let spec = match spec {
        Some(s) => s,
        None => return Ok(StatResult::Identity),
    };

    // Resolve target keys (user-facing) → internal aesthetic names. An alias
    // like `color` expands to whichever of its targets (stroke/fill) is mapped
    // on the layer; the function applies to all of them.
    let mut targets_internal: HashMap<String, AggSpec> = HashMap::new();
    for (user_aes, agg) in &spec.targets {
        let resolved = resolve_target_aesthetic(user_aes, aesthetics, aesthetic_ctx);
        if resolved.is_empty() {
            return Err(GgsqlError::ValidationError(format!(
                "aggregate target '{}' is not mapped on this layer",
                user_aes
            )));
        }
        for internal in resolved {
            if targets_internal.contains_key(&internal) {
                return Err(GgsqlError::ValidationError(format!(
                    "aggregate target '{}' resolves to aesthetic '{}' which is already targeted",
                    user_aes, internal
                )));
            }
            targets_internal.insert(internal, agg.clone());
        }
    }

    // Walk mappings. Three buckets:
    //   - aggregated: (internal_aes, raw_col, AggSpec) — each emits one column
    //   - kept_cols: discrete column-mappings — keep as group key
    //   - dropped: numeric mapping with no applicable function (warn & skip)
    let mut aggregated: Vec<(String, String, AggSpec)> = Vec::new();
    let mut kept_cols: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    let mut entries: Vec<(&String, &crate::AestheticValue)> = aesthetics.aesthetics.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (aes, value) in entries {
        let col = match value.column_name() {
            Some(c) => c.to_string(),
            None => continue, // literals & annotation columns pass through
        };
        let info = schema.iter().find(|c| c.name == col);
        let is_discrete = info.map(|c| c.is_discrete).unwrap_or(false);
        if is_discrete {
            if !kept_cols.contains(&col) {
                kept_cols.push(col);
            }
            continue;
        }

        // Numeric mapping. Look up the aggregation function.
        let agg = if let Some(targeted) = targets_internal.get(aes) {
            Some(targeted.clone())
        } else if is_upper_half(aes) {
            spec.default_upper
                .clone()
                .or_else(|| spec.default_lower.clone())
                .filter(|_| spec.default_upper.is_some() || spec.default_lower.is_some())
        } else {
            spec.default_lower.clone()
        };

        match agg {
            Some(a) => aggregated.push((aes.clone(), col, a)),
            None => dropped.push(aes.clone()),
        }
    }

    // The *only* time we have nothing to aggregate but should still transform
    // is when defaults exist but every numeric mapping was dropped — we still
    // emit a GROUP BY to honour the grouping. If there are no aggregations and
    // no kept columns and no group_by, return Identity.
    if aggregated.is_empty() && kept_cols.is_empty() && group_by.is_empty() {
        for d in &dropped {
            eprintln!(
                "Warning: aggregate dropped numeric mapping for aesthetic '{}' (no applicable default and no targeted function)",
                aesthetic_ctx.map_internal_to_user(d)
            );
        }
        return Ok(StatResult::Identity);
    }

    for d in &dropped {
        eprintln!(
            "Warning: aggregate dropped numeric mapping for aesthetic '{}' (no applicable default and no targeted function)",
            aesthetic_ctx.map_internal_to_user(d)
        );
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

    let transformed_query =
        build_group_by_query(query, &aggregated, &group_cols, dialect);

    let stat_columns: Vec<String> = aggregated.iter().map(|(a, _, _)| a.clone()).collect();
    let consumed_aesthetics: Vec<String> = stat_columns.clone();

    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns: vec![],
        consumed_aesthetics,
    })
}

/// Build the `WITH src AS (<query>) SELECT <group cols>, <agg exprs> FROM src
/// AS "__ggsql_qt__" GROUP BY <group cols>` query.
///
/// Falls back to `dialect.sql_percentile()` per-column when an aggregate's
/// percentile component lacks inline support.
fn build_group_by_query(
    query: &str,
    aggregated: &[(String, String, AggSpec)],
    group_cols: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    let src_alias = "\"__ggsql_stat_src__\"";
    let outer_alias = "\"__ggsql_qt__\"";

    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        format!(" GROUP BY {}", group_select.join(", "))
    };

    let mut select_parts: Vec<String> = group_select.clone();

    for (aes, raw_col, agg) in aggregated {
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
        "WITH {src} AS ({query}) SELECT {sel} FROM {src} AS {outer}{gb}",
        src = src_alias,
        query = query,
        sel = select_parts.join(", "),
        outer = outer_alias,
        gb = group_by_clause,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::aesthetic::AestheticContext;
    use crate::plot::types::{AestheticValue, ColumnInfo};
    use arrow::datatypes::DataType;

    /// A test dialect that mimics DuckDB's native QUANTILE_CONT support.
    struct InlineQuantileDialect;
    impl SqlDialect for InlineQuantileDialect {
        fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
            Some(format!(
                "QUANTILE_CONT({}, {})",
                naming::quote_ident(column),
                fraction
            ))
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
        let mut p = HashMap::new();
        p.insert("aggregate".to_string(), params);
        let ctx = cartesian_ctx();
        apply("SELECT * FROM t", schema, aes, group_by, &p, dialect, &ctx)
    }

    fn arr(items: &[&str]) -> ParameterValue {
        ParameterValue::Array(items.iter().map(|s| ArrayElement::String(s.to_string())).collect())
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
        let s = parse_aggregate_param(&arr(&["min", "max"])).unwrap().unwrap();
        assert_eq!(s.default_lower.as_ref().map(|a| a.offset), Some("min"));
        assert_eq!(s.default_upper.as_ref().map(|a| a.offset), Some("max"));
    }

    #[test]
    fn three_unprefixed_defaults_is_error() {
        let err = parse_aggregate_param(&arr(&["mean", "min", "max"])).unwrap_err();
        assert!(err.contains("at most two"), "got: {}", err);
    }

    #[test]
    fn parses_targeted_entries() {
        let s = parse_aggregate_param(&arr(&["mean", "y:max", "color:median"]))
            .unwrap()
            .unwrap();
        assert_eq!(s.default_lower.as_ref().map(|a| a.offset), Some("mean"));
        assert_eq!(s.targets.get("y").map(|a| a.offset), Some("max"));
        assert_eq!(s.targets.get("color").map(|a| a.offset), Some("median"));
    }

    #[test]
    fn duplicate_target_is_error() {
        let err = parse_aggregate_param(&arr(&["y:mean", "y:median"])).unwrap_err();
        assert!(err.contains("more than one aggregate"), "got: {}", err);
    }

    #[test]
    fn empty_prefix_is_error() {
        let err = parse_aggregate_param(&ParameterValue::String(":mean".to_string())).unwrap_err();
        assert!(err.contains("aesthetic prefix"), "got: {}", err);
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
            s.default_lower.as_ref().unwrap().band.as_ref().unwrap().expansion,
            "sdev"
        );
        assert_eq!(
            s.default_lower.as_ref().unwrap().band.as_ref().unwrap().sign,
            '-'
        );
        assert_eq!(s.default_upper.as_ref().unwrap().offset, "mean");
    }

    // ---------- apply tests ----------

    #[test]
    fn returns_identity_when_param_unset() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let p: HashMap<String, ParameterValue> = HashMap::new();
        let ctx = cartesian_ctx();
        let result = apply("SELECT * FROM t", &schema, &aes, &[], &p, &InlineQuantileDialect, &ctx)
            .unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn returns_identity_when_param_null() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let result = run(ParameterValue::Null, &aes, &schema, &[], &InlineQuantileDialect).unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn single_default_applies_to_every_numeric_mapping() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
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
        let result = run(arr(&["min", "max"]), &aes, &schema, &[], &InlineQuantileDialect)
            .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // pos1, pos2 use MIN; pos1end, pos2end use MAX.
                assert!(query.contains("MIN(\"__ggsql_aes_pos1__\")"), "{}", query);
                assert!(query.contains("MIN(\"__ggsql_aes_pos2__\")"), "{}", query);
                assert!(query.contains("MAX(\"__ggsql_aes_pos1end__\")"), "{}", query);
                assert!(query.contains("MAX(\"__ggsql_aes_pos2end__\")"), "{}", query);
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
                let pos2max_section = query
                    .split("__ggsql_aes_pos2max__\")")
                    .next()
                    .unwrap_or("");
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
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
        ]);
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
            StatResult::Transformed { query, stat_columns, .. } => {
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
            StatResult::Transformed { query, stat_columns, .. } => {
                assert!(query.contains("MAX(\"__ggsql_aes_fill__\")"), "{}", query);
                assert!(query.contains("AVG(\"__ggsql_aes_pos1__\")"));
                assert!(stat_columns.contains(&"fill".to_string()));
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
                assert!(query.contains("GROUP BY \"__ggsql_aes_color__\""), "{}", query);
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
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
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
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
        ]);
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
        let schema = schema_for(&[
            ("__ggsql_aes_pos1__", false),
            ("__ggsql_aes_pos2__", false),
        ]);
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
