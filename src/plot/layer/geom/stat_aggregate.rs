//! Aggregate stat - groups data and applies one or more aggregation functions per group.
//!
//! When a layer's `aggregate` SETTING is set to a function name (or array of names),
//! this stat groups by discrete mappings + PARTITION BY columns and produces one row
//! per (group × function), aggregating numeric position aesthetics.
//!
//! Output columns:
//! - One column per numeric position aesthetic (named `pos1`, `pos2`, etc.) holding the
//!   aggregated value. NULL for `count` rows.
//! - `aggregate` - the function name for the row.
//! - `count` (only when `count` is requested) - the row tally for that group.

use std::collections::HashMap;

use super::types::StatResult;
use crate::naming;
use crate::plot::aesthetic::{is_position_aesthetic, parse_position};
use crate::plot::types::{ParameterValue, Schema};
use crate::reader::SqlDialect;
use crate::{GgsqlError, Mappings, Result};

/// All simple-aggregation function names accepted by the `aggregate` SETTING.
///
/// Band names (e.g. `mean+sdev`, `median-0.5iqr`) are validated separately by
/// `parse_agg_name`, which checks the offset against `OFFSET_STATS` and the
/// expansion against `EXPANSION_STATS`.
pub const AGG_NAMES: &[&str] = &[
    // Tallies & sums
    "count",
    "sum",
    "prod",
    // Extremes
    "min",
    "max",
    "range",
    // Central tendency
    "mean",
    "geomean",
    "harmean",
    "rms",
    "median",
    // Spread (standalone)
    "sdev",
    "var",
    "iqr",
    // Percentiles
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
];

/// Stats that can appear as the *offset* (left of `±`) in a band name like
/// `mean+sdev`. Single-value central or representative quantities only —
/// counts/spreads are excluded.
pub const OFFSET_STATS: &[&str] = &[
    "mean",
    "median",
    "geomean",
    "harmean",
    "rms",
    "sum",
    "prod",
    "min",
    "max",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
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

/// Resolve a name to its canonical `&'static str` from the given vocabulary,
/// or `None` if the input doesn't match any entry.
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

/// Try to parse `name` as a band: `<offset><sign><mod>?<expansion>`. Returns
/// `None` if it doesn't match the band shape OR if either half is outside its
/// allowed vocabulary.
fn parse_band(name: &str) -> Option<AggSpec> {
    // Walk offsets longest-first so `median` matches before `mean`.
    let mut offsets: Vec<&'static str> = OFFSET_STATS.to_vec();
    offsets.sort_by_key(|s| std::cmp::Reverse(s.len()));

    for offset in offsets {
        let rest = match name.strip_prefix(offset) {
            Some(r) => r,
            None => continue, // doesn't start with this offset
        };
        let (sign, after_sign) = match rest.chars().next() {
            Some('+') => ('+', &rest[1..]),
            Some('-') => ('-', &rest[1..]),
            _ => continue, // wrong sign char — try next offset
        };

        let (mod_value, expansion_str) = parse_mod_and_remainder(after_sign);
        let expansion = match resolve_static(expansion_str, EXPANSION_STATS) {
            Some(e) => e,
            None => continue, // expansion doesn't match — try next offset
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

/// Parse a leading `<digits>(.<digits>)?` modifier from `s`. Returns
/// `(parsed_value, rest_of_string)`. If no leading digits, returns
/// `(1.0, s)` — modifier defaults to 1.
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
            // need at least one digit after '.'
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

/// Validate the `aggregate` SETTING value: null, a single function name, or
/// an array of function names. Each name must be parseable by `parse_agg_name`.
pub fn validate_aggregate_param(value: &ParameterValue) -> std::result::Result<(), String> {
    use crate::plot::types::ArrayElement;
    match value {
        ParameterValue::Null => Ok(()),
        ParameterValue::String(s) => validate_function_name(s),
        ParameterValue::Array(arr) => {
            for el in arr {
                match el {
                    ArrayElement::String(s) => validate_function_name(s)?,
                    ArrayElement::Null => continue,
                    _ => {
                        return Err(
                            "'aggregate' array entries must be strings or null".to_string()
                        );
                    }
                }
            }
            Ok(())
        }
        _ => Err("'aggregate' must be a string, array of strings, or null".to_string()),
    }
}

fn validate_function_name(name: &str) -> std::result::Result<(), String> {
    match parse_agg_name(name) {
        Some(_) => Ok(()),
        None => Err(diagnose_invalid_function_name(name)),
    }
}

/// Build a per-role error message for a name that didn't parse. Re-walks the
/// input with looser rules to identify which side (offset / expansion) failed.
fn diagnose_invalid_function_name(name: &str) -> String {
    // Look for a sign character. If there is one, examine the offset and
    // expansion halves separately.
    if let Some(sign_idx) = name.find(|c| c == '+' || c == '-') {
        let offset_str = &name[..sign_idx];
        let after_sign = &name[sign_idx + 1..];
        let (_mod_value, expansion_str) = parse_mod_and_remainder(after_sign);

        let offset_known_simple = AGG_NAMES.contains(&offset_str);
        let offset_known_band = OFFSET_STATS.contains(&offset_str);
        let expansion_known_band = EXPANSION_STATS.contains(&expansion_str);

        if !offset_known_band {
            // The offset half is the problem.
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
        // Both halves are individually valid but band parsing failed for some
        // other reason (e.g. malformed modifier).
        return format!("'{}' is not a valid aggregate function name", name);
    }
    format!(
        "unknown aggregate function '{}'. Allowed: {} (or use a band like `mean+sdev`)",
        name,
        crate::or_list_quoted(AGG_NAMES, '\''),
    )
}

/// Apply the Aggregate stat to a layer query.
///
/// Returns `StatResult::Identity` when the `aggregate` parameter is unset or null.
/// Otherwise, builds a grouped-aggregation query and returns `StatResult::Transformed`.
///
/// Strategy:
/// - **Single-pass** (preferred): one `GROUP BY` produces a wide row per group, then
///   `CROSS JOIN VALUES(...)` of function names explodes to one row per (group × function).
///   Used when all requested functions are inline-able.
/// - **UNION ALL fallback**: when a quantile is requested but the dialect doesn't
///   provide `sql_quantile_inline`, fall back to per-function subqueries using
///   `dialect.sql_percentile`.
pub fn apply(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
    parameters: &HashMap<String, ParameterValue>,
    dialect: &dyn SqlDialect,
    agg_slots: &[u8],
    range_pair: Option<(&'static str, &'static str)>,
) -> Result<StatResult> {
    let funcs = match extract_aggregate_param(parameters) {
        None => return Ok(StatResult::Identity),
        Some(funcs) => funcs,
    };

    if let Some((lo, hi)) = range_pair {
        return apply_range_mode(query, schema, aesthetics, group_by, &funcs, dialect, lo, hi);
    }

    // Walk the layer's position aesthetics and route each by (slot, type):
    //   in-axis slot && numeric  → aggregated (numeric_pos)
    //   in-axis slot && discrete → kept as group column (kept_pos_cols)
    //   out-of-axis (any type)   → kept as group column (kept_pos_cols)
    let mut numeric_pos: Vec<(String, String)> = Vec::new(); // (aesthetic, prefixed col)
    let mut kept_pos_cols: Vec<String> = Vec::new();
    for (aesthetic, value) in &aesthetics.aesthetics {
        if !is_position_aesthetic(aesthetic) {
            continue;
        }
        let col = match value.column_name() {
            Some(c) => c.to_string(),
            None => continue,
        };
        let slot = parse_position(aesthetic).map(|(s, _)| s).unwrap_or(0);
        let in_axis = agg_slots.contains(&slot);
        let info = schema.iter().find(|c| c.name == col);
        let is_discrete = info.map(|c| c.is_discrete).unwrap_or(false);

        if !in_axis || is_discrete {
            kept_pos_cols.push(col);
        } else {
            numeric_pos.push((aesthetic.clone(), col));
        }
    }
    numeric_pos.sort_by(|a, b| a.0.cmp(&b.0));
    kept_pos_cols.sort();

    if numeric_pos.is_empty() && !funcs.iter().any(|f| f == "count") {
        return Err(GgsqlError::ValidationError(
            "aggregate requires at least one numeric position aesthetic, or the 'count' function"
                .to_string(),
        ));
    }

    // Group columns: PARTITION BY + discrete mappings (already in group_by) + any
    // position-aesthetic columns we kept (out-of-axis or in-axis-but-discrete).
    // Deduplicated, preserving order.
    let mut group_cols: Vec<String> = Vec::new();
    for g in group_by {
        if !group_cols.contains(g) {
            group_cols.push(g.clone());
        }
    }
    for c in &kept_pos_cols {
        if !group_cols.contains(c) {
            group_cols.push(c.clone());
        }
    }

    let needs_count_col = funcs.iter().any(|f| f == "count");

    // Decide strategy: single-pass when every percentile component can be inlined.
    let probe = numeric_pos
        .first()
        .map(|(_, c)| c.as_str())
        .unwrap_or("__ggsql_probe__");
    let needs_fallback = funcs.iter().any(|f| {
        parse_agg_name(f)
            .map(|spec| needs_quantile_fallback(&spec, probe, dialect))
            .unwrap_or(false)
    });

    let transformed_query = if needs_fallback {
        build_union_all_query(query, &funcs, &numeric_pos, &group_cols, dialect)
    } else {
        build_single_pass_query(query, &funcs, &numeric_pos, &group_cols, dialect)
    };

    let mut stat_columns: Vec<String> = numeric_pos.iter().map(|(a, _)| a.clone()).collect();
    stat_columns.push("aggregate".to_string());
    if needs_count_col {
        stat_columns.push("count".to_string());
    }

    let consumed_aesthetics: Vec<String> = numeric_pos.into_iter().map(|(a, _)| a).collect();

    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns,
        dummy_columns: vec![],
        consumed_aesthetics,
    })
}

/// Extract the `aggregate` parameter as a list of function names, or `None` when
/// the parameter is unset/null.
fn extract_aggregate_param(parameters: &HashMap<String, ParameterValue>) -> Option<Vec<String>> {
    use crate::plot::types::ArrayElement;
    match parameters.get("aggregate") {
        None | Some(ParameterValue::Null) => None,
        Some(ParameterValue::String(s)) => Some(vec![s.clone()]),
        Some(ParameterValue::Array(arr)) => {
            let names: Vec<String> = arr
                .iter()
                .filter_map(|el| match el {
                    ArrayElement::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if names.is_empty() {
                None
            } else {
                Some(names)
            }
        }
        _ => None,
    }
}

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
/// quoted column.
///
/// Returns `None` for `count` (which doesn't take a column) and for percentile-
/// based stats (`p05..p95`, `median`, `iqr`) when the dialect lacks an inline
/// quantile aggregate (caller should switch to UNION ALL strategy).
fn simple_stat_sql_inline(name: &str, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    if name == "count" {
        return None;
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

/// Inline SQL for a parsed `AggSpec`. Combines the offset and (optional)
/// expansion halves with the appropriate sign and modifier.
fn agg_sql_inline(spec: &AggSpec, qcol: &str, dialect: &dyn SqlDialect) -> Option<String> {
    let offset_sql = simple_stat_sql_inline(spec.offset, qcol, dialect)?;
    match &spec.band {
        None => Some(offset_sql),
        Some(band) => {
            let exp_sql = simple_stat_sql_inline(band.expansion, qcol, dialect)?;
            Some(format_band(&offset_sql, band.sign, band.mod_value, &exp_sql))
        }
    }
}

/// Build the SQL fragment `(offset ± mod * exp)`, omitting the `mod *` prefix
/// when `mod_value == 1.0`.
fn format_band(offset: &str, sign: char, mod_value: f64, exp: &str) -> String {
    if mod_value == 1.0 {
        format!("({} {} {})", offset, sign, exp)
    } else {
        format!("({} {} {} * {})", offset, sign, mod_value, exp)
    }
}

/// Fallback SQL for a simple stat. Used by the UNION-ALL path for percentile
/// components (which need correlated `sql_percentile`) and falls through to
/// the inline form for everything else.
fn simple_stat_sql_fallback(
    name: &str,
    raw_col: &str,
    dialect: &dyn SqlDialect,
    src_alias: &str,
    group_cols: &[String],
) -> String {
    if name == "count" {
        return "NULL".to_string();
    }
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

/// Fallback SQL for a parsed `AggSpec` (UNION-ALL path).
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

/// Whether this spec has any percentile component that the dialect can't
/// inline (in which case the caller must use the UNION-ALL fallback).
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

/// Strip surrounding double quotes from an identifier, undoing `naming::quote_ident`.
fn unquote(qcol: &str) -> String {
    let trimmed = qcol.trim_start_matches('"').trim_end_matches('"');
    trimmed.replace("\"\"", "\"")
}

/// SQL for a function name literal, properly escaped.
fn func_literal(func: &str) -> String {
    format!("'{}'", func.replace('\'', "''"))
}

// =============================================================================
// Range-mode strategy: exactly two functions filling a (lower, upper) aesthetic
// pair on the same row. Used by ribbon/range.
// =============================================================================

fn apply_range_mode(
    query: &str,
    schema: &Schema,
    aesthetics: &Mappings,
    group_by: &[String],
    funcs: &[String],
    dialect: &dyn SqlDialect,
    lo: &'static str,
    hi: &'static str,
) -> Result<StatResult> {
    if funcs.len() != 2 {
        return Err(GgsqlError::ValidationError(format!(
            "aggregate on a range geom must be an array of exactly two functions (lower, upper), got {}",
            funcs.len()
        )));
    }

    // Range mode requires `pos2` mapped to a numeric input column. The user
    // writes `MAPPING value AS y` and the stat consumes it to produce both
    // bounds.
    let input_col = match aesthetics.get("pos2").and_then(|v| v.column_name()) {
        Some(c) => c.to_string(),
        None => {
            return Err(GgsqlError::ValidationError(
                "aggregate on a range geom requires a `y` (pos2) mapping as the input column"
                    .to_string(),
            ));
        }
    };
    let info = schema.iter().find(|c| c.name == input_col);
    if info.map(|c| c.is_discrete).unwrap_or(false) {
        return Err(GgsqlError::ValidationError(
            "aggregate on a range geom requires a numeric `y` (pos2) input, not a discrete column"
                .to_string(),
        ));
    }
    let qcol = naming::quote_ident(&input_col);

    // Group columns: PARTITION BY + discrete mappings (already in group_by) +
    // any discrete position aesthetics on the layer (e.g. pos1 if it's a string).
    let mut group_cols: Vec<String> = Vec::new();
    for g in group_by {
        if !group_cols.contains(g) {
            group_cols.push(g.clone());
        }
    }
    for (aesthetic, value) in &aesthetics.aesthetics {
        if !is_position_aesthetic(aesthetic) || aesthetic == "pos2" {
            continue;
        }
        let col = match value.column_name() {
            Some(c) => c.to_string(),
            None => continue,
        };
        if !group_cols.contains(&col) {
            group_cols.push(col);
        }
    }

    let src_alias = "\"__ggsql_stat_src__\"";
    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        let qcols: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
        format!(" GROUP BY {}", qcols.join(", "))
    };

    // Parse and emit each bound. Use the inline form when the dialect supports
    // every percentile component; otherwise fall back to `sql_percentile`
    // correlated to the outer alias used in the FROM (`__ggsql_qt__`).
    let lo_expr = build_range_function_sql(&funcs[0], &qcol, &input_col, dialect, &group_cols)?;
    let hi_expr = build_range_function_sql(&funcs[1], &qcol, &input_col, dialect, &group_cols)?;

    let stat_lo = naming::stat_column(lo);
    let stat_hi = naming::stat_column(hi);

    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
    let mut select_parts = group_select.clone();
    select_parts.push(format!("{} AS {}", lo_expr, naming::quote_ident(&stat_lo)));
    select_parts.push(format!("{} AS {}", hi_expr, naming::quote_ident(&stat_hi)));

    let transformed_query = format!(
        "WITH {src} AS ({query}) SELECT {sel} FROM {src} AS \"__ggsql_qt__\"{gb}",
        src = src_alias,
        query = query,
        sel = select_parts.join(", "),
        gb = group_by_clause,
    );

    // consumed_aesthetics: pos2 carries the original-name capture for axis
    // labels; lo/hi flag the auto-rename in execute/layer.rs (their stat-column
    // names match the position aesthetics they fill).
    Ok(StatResult::Transformed {
        query: transformed_query,
        stat_columns: vec![lo.to_string(), hi.to_string()],
        dummy_columns: vec![],
        consumed_aesthetics: vec!["pos2".to_string(), lo.to_string(), hi.to_string()],
    })
}

/// Build the SQL fragment for one function in range mode. Parses the function
/// name into an `AggSpec` (which validates the offset/expansion vocabulary)
/// and emits inline SQL when the dialect supports every percentile component,
/// otherwise the correlated fallback.
fn build_range_function_sql(
    func: &str,
    qcol: &str,
    raw_col: &str,
    dialect: &dyn SqlDialect,
    group_cols: &[String],
) -> Result<String> {
    if func == "count" {
        return Err(GgsqlError::ValidationError(
            "aggregate on a range geom does not support 'count' (it has no range semantics)"
                .to_string(),
        ));
    }
    let spec = parse_agg_name(func).ok_or_else(|| {
        GgsqlError::ValidationError(format!(
            "aggregate on a range geom: {}",
            diagnose_invalid_function_name(func)
        ))
    })?;
    if needs_quantile_fallback(&spec, raw_col, dialect) {
        Ok(agg_sql_fallback(
            &spec,
            raw_col,
            dialect,
            "\"__ggsql_stat_src__\"",
            group_cols,
        ))
    } else {
        agg_sql_inline(&spec, qcol, dialect).ok_or_else(|| {
            GgsqlError::ValidationError(format!(
                "aggregate on a range geom does not support function '{}' on this dialect",
                func
            ))
        })
    }
}

// =============================================================================
// Single-pass strategy: GROUP BY produces a wide CTE, then CROSS JOIN explodes
// rows per requested function.
// =============================================================================

fn build_single_pass_query(
    query: &str,
    funcs: &[String],
    numeric_pos: &[(String, String)],
    group_cols: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    let src_alias = "\"__ggsql_stat_src__\"";
    let agg_alias = "\"__ggsql_stat_agg__\"";
    let funcs_alias = "\"__ggsql_stat_funcs__\"";

    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        let qcols: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
        format!(" GROUP BY {}", qcols.join(", "))
    };

    // Build the wide aggregation SELECT: one column per (function × position).
    let mut wide_select_exprs: Vec<String> =
        group_cols.iter().map(|c| naming::quote_ident(c)).collect();

    // Track the synthetic column names for each (aesthetic, function) pair.
    let mut wide_col_for: HashMap<(String, String), String> = HashMap::new();

    for (aes, col) in numeric_pos {
        let qcol = naming::quote_ident(col);
        for func in funcs {
            if func == "count" {
                continue;
            }
            let key = (aes.clone(), func.clone());
            if wide_col_for.contains_key(&key) {
                continue;
            }
            let wide_name = synthetic_col_name(aes, func);
            let spec = parse_agg_name(func)
                .expect("aggregate function names are validated upstream of single-pass");
            let expr = agg_sql_inline(&spec, &qcol, dialect)
                .expect("agg_sql_inline must be Some when single-pass is selected");
            wide_select_exprs.push(format!("{} AS {}", expr, naming::quote_ident(&wide_name)));
            wide_col_for.insert(key, wide_name);
        }
    }

    let needs_count_col = funcs.iter().any(|f| f == "count");
    let count_wide = if needs_count_col {
        let c = "__ggsql_stat_cnt__";
        wide_select_exprs.push(format!("COUNT(*) AS {}", naming::quote_ident(c)));
        Some(c.to_string())
    } else {
        None
    };

    let wide_select = wide_select_exprs.join(", ");

    // Build the CROSS JOIN VALUES table of function names.
    let funcs_values: Vec<String> = funcs
        .iter()
        .map(|f| format!("({})", func_literal(f)))
        .collect();
    let funcs_cte = format!(
        "{}(name) AS (VALUES {})",
        funcs_alias,
        funcs_values.join(", ")
    );

    // Build the outer SELECT: group cols + per-aesthetic CASE + count CASE + name AS aggregate.
    let mut outer_exprs: Vec<String> = group_cols
        .iter()
        .map(|c| format!("{}.{}", agg_alias, naming::quote_ident(c)))
        .collect();

    for (aes, _) in numeric_pos {
        let stat_col = naming::stat_column(aes);
        let mut whens: Vec<String> = Vec::new();
        for func in funcs {
            if let Some(wide_name) = wide_col_for.get(&(aes.clone(), func.clone())) {
                whens.push(format!(
                    "WHEN {} THEN {}.{}",
                    func_literal(func),
                    agg_alias,
                    naming::quote_ident(wide_name)
                ));
            }
        }
        let case_expr = if whens.is_empty() {
            "NULL".to_string()
        } else {
            format!(
                "CASE {}.name {} ELSE NULL END",
                funcs_alias,
                whens.join(" ")
            )
        };
        outer_exprs.push(format!(
            "{} AS {}",
            case_expr,
            naming::quote_ident(&stat_col)
        ));
    }

    if let Some(count_wide) = count_wide {
        let stat_col = naming::stat_column("count");
        let case_expr = format!(
            "CASE {f}.name WHEN {lit} THEN {a}.{c} ELSE NULL END",
            f = funcs_alias,
            a = agg_alias,
            lit = func_literal("count"),
            c = naming::quote_ident(&count_wide)
        );
        outer_exprs.push(format!(
            "{} AS {}",
            case_expr,
            naming::quote_ident(&stat_col)
        ));
    }

    let stat_aggregate_col = naming::stat_column("aggregate");
    outer_exprs.push(format!(
        "{}.name AS {}",
        funcs_alias,
        naming::quote_ident(&stat_aggregate_col)
    ));

    format!(
        "WITH {src} AS ({query}), \
         {agg_alias_def} AS (SELECT {wide_select} FROM {src}{group_by}), \
         {funcs_cte} \
         SELECT {outer} FROM {agg} CROSS JOIN {funcs}",
        src = src_alias,
        query = query,
        agg_alias_def = agg_alias,
        wide_select = wide_select,
        group_by = group_by_clause,
        funcs_cte = funcs_cte,
        outer = outer_exprs.join(", "),
        agg = agg_alias,
        funcs = funcs_alias,
    )
}

/// Synthetic name for a (aesthetic, function) intermediate column in the wide CTE.
/// Includes a sanitized form of the function name to avoid collisions on `+`/`-`.
fn synthetic_col_name(aes: &str, func: &str) -> String {
    let safe: String = func
        .chars()
        .map(|c| match c {
            '+' => 'p',
            '-' => 'm',
            _ if c.is_ascii_alphanumeric() => c,
            _ => '_',
        })
        .collect();
    format!("__ggsql_stat_{}_{}", aes, safe)
}

// =============================================================================
// UNION ALL fallback strategy: one SELECT per requested function.
// =============================================================================

fn build_union_all_query(
    query: &str,
    funcs: &[String],
    numeric_pos: &[(String, String)],
    group_cols: &[String],
    dialect: &dyn SqlDialect,
) -> String {
    let src_alias = "\"__ggsql_stat_src__\"";

    let group_by_clause = if group_cols.is_empty() {
        String::new()
    } else {
        let qcols: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();
        format!(" GROUP BY {}", qcols.join(", "))
    };

    let group_select: Vec<String> = group_cols.iter().map(|c| naming::quote_ident(c)).collect();

    let needs_count_col = funcs.iter().any(|f| f == "count");
    let stat_aggregate_col = naming::stat_column("aggregate");
    let stat_count_col = naming::stat_column("count");

    let branches: Vec<String> = funcs
        .iter()
        .map(|func| {
            let mut select_parts: Vec<String> = group_select.clone();

            // Parse the function name once per branch. Falls through to a
            // string-NULL value column if parsing fails (shouldn't happen
            // because validation runs upstream, but stay defensive).
            let parsed_spec = parse_agg_name(func);
            for (aes, col) in numeric_pos {
                let stat_col = naming::stat_column(aes);
                let value_expr = if func == "count" {
                    "NULL".to_string()
                } else if let Some(spec) = &parsed_spec {
                    agg_sql_fallback(spec, col, dialect, src_alias, group_cols)
                } else {
                    "NULL".to_string()
                };
                select_parts.push(format!(
                    "{} AS {}",
                    value_expr,
                    naming::quote_ident(&stat_col)
                ));
            }

            if needs_count_col {
                let value_expr = if func == "count" {
                    "COUNT(*)".to_string()
                } else {
                    "NULL".to_string()
                };
                select_parts.push(format!(
                    "{} AS {}",
                    value_expr,
                    naming::quote_ident(&stat_count_col)
                ));
            }

            select_parts.push(format!(
                "{} AS {}",
                func_literal(func),
                naming::quote_ident(&stat_aggregate_col)
            ));

            // Quantile fallbacks (sql_percentile) need the outer alias `__ggsql_qt__`
            // so their correlated WHERE clause can find group columns.
            format!(
                "SELECT {} FROM {} AS \"__ggsql_qt__\"{}",
                select_parts.join(", "),
                src_alias,
                group_by_clause
            )
        })
        .collect();

    format!(
        "WITH {src} AS ({query}) {branches}",
        src = src_alias,
        query = query,
        branches = branches.join(" UNION ALL ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A test dialect with no inline quantile support, exercising the UNION ALL fallback.
    struct NoInlineQuantileDialect;
    impl SqlDialect for NoInlineQuantileDialect {}

    fn col(name: &str) -> AestheticValue {
        AestheticValue::Column {
            name: name.to_string(),
            original_name: None,
            is_dummy: false,
        }
    }

    fn numeric_schema(cols: &[&str]) -> Schema {
        cols.iter()
            .map(|c| ColumnInfo {
                name: c.to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            })
            .collect()
    }

    #[test]
    fn returns_identity_when_param_unset() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let params = HashMap::new();
        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn returns_identity_when_param_null() {
        let aes = Mappings::new();
        let schema: Schema = vec![];
        let mut params = HashMap::new();
        params.insert("aggregate".to_string(), ParameterValue::Null);
        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        assert_eq!(result, StatResult::Identity);
    }

    #[test]
    fn single_pass_for_mean_emits_avg() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();

        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos2__\")"),
                    "query: {}",
                    query
                );
                assert!(query.contains("CROSS JOIN"));
                assert!(stat_columns.contains(&"pos2".to_string()));
                assert!(stat_columns.contains(&"aggregate".to_string()));
                assert!(!stat_columns.contains(&"count".to_string()));
                assert_eq!(consumed_aesthetics, vec!["pos2".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn count_emits_count_star_and_keeps_count_column() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("count".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();

        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("COUNT(*)"));
                assert!(stat_columns.contains(&"count".to_string()));
                assert!(stat_columns.contains(&"aggregate".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn mixed_count_and_mean_produces_two_rows_per_group() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("count".to_string()),
                ArrayElement::String("mean".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
                assert!(query.contains("COUNT(*)"));
                assert!(query.contains("'count'"));
                assert!(query.contains("'mean'"));
                // The count CASE must reference the agg CTE for the value column,
                // not the funcs CTE (regression: previously emitted funcs.cnt which
                // doesn't exist).
                assert!(
                    query.contains("\"__ggsql_stat_agg__\".\"__ggsql_stat_cnt__\""),
                    "count CASE should reference the agg CTE, query was: {}",
                    query
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn quantile_uses_dialect_inline_when_available() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("p25".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("QUANTILE_CONT"));
                assert!(query.contains("0.25"));
                assert!(!query.contains("UNION ALL"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn quantile_falls_back_to_union_all_without_dialect_support() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("p25".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &NoInlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // Fallback dialect uses NTILE-based correlated subquery via UNION ALL.
                assert!(query.contains("NTILE(4)"));
                assert!(query.contains("UNION ALL") || !query.contains("CROSS JOIN"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn mean_sdev_emits_avg_and_stddev() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("mean-sdev".to_string()),
                ArrayElement::String("mean+sdev".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("STDDEV_POP"));
                assert!(query.contains("AVG"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn mean_se_includes_sqrt_count() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean+se".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("SQRT(COUNT"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn prod_emits_exp_sum_ln() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("prod".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("EXP(SUM(LN"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn iqr_emits_p75_minus_p25() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("iqr".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("0.75"));
                assert!(query.contains("0.25"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn discrete_position_aesthetic_becomes_group_column() {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = vec![
            ColumnInfo {
                name: "__ggsql_aes_pos1__".to_string(),
                dtype: DataType::Utf8,
                is_discrete: true,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
        ];
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                // pos1 (discrete) is in GROUP BY, not aggregated.
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                // pos2 is aggregated.
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
                // Only pos2 is consumed.
                assert_eq!(consumed_aesthetics, vec!["pos2".to_string()]);
                // Only pos2 (numeric) appears in stat_columns; pos1 stays as-is.
                assert!(stat_columns.contains(&"pos2".to_string()));
                assert!(!stat_columns.contains(&"pos1".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn explicit_group_by_columns_appear_in_query() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &["region".to_string()],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("GROUP BY \"region\""));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn line_style_groups_by_pos1_and_aggregates_pos2() {
        // slots=[2]: pos1 stays as group (even though numeric), pos2 gets aggregated.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos1__", "__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("max".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                consumed_aesthetics,
                stat_columns,
                ..
            } => {
                assert!(
                    query.contains("MAX(\"__ggsql_aes_pos2__\")"),
                    "query: {}",
                    query
                );
                assert!(!query.contains("MAX(\"__ggsql_aes_pos1__\")"));
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                assert_eq!(consumed_aesthetics, vec!["pos2".to_string()]);
                assert!(stat_columns.contains(&"pos2".to_string()));
                assert!(!stat_columns.contains(&"pos1".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn point_style_aggregates_both_slots() {
        // slots=[1,2]: both pos1 and pos2 (numeric) get aggregated → centroid.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos1__", "__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[1, 2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                consumed_aesthetics,
                stat_columns,
                ..
            } => {
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos1__\")"),
                    "query: {}",
                    query
                );
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos2__\")"),
                    "query: {}",
                    query
                );
                assert!(!query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                let mut consumed = consumed_aesthetics.clone();
                consumed.sort();
                assert_eq!(consumed, vec!["pos1".to_string(), "pos2".to_string()]);
                assert!(stat_columns.contains(&"pos1".to_string()));
                assert!(stat_columns.contains(&"pos2".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn range_geom_aggregates_pos2_minmax() {
        // slots=[2]: pos1 fixed (group), pos2min and pos2max both aggregated.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2min", col("__ggsql_aes_pos2min__"));
        aes.insert("pos2max", col("__ggsql_aes_pos2max__"));
        let schema = numeric_schema(&[
            "__ggsql_aes_pos1__",
            "__ggsql_aes_pos2min__",
            "__ggsql_aes_pos2max__",
        ]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                consumed_aesthetics,
                ..
            } => {
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos2min__\")"),
                    "query: {}",
                    query
                );
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos2max__\")"),
                    "query: {}",
                    query
                );
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                let mut consumed = consumed_aesthetics.clone();
                consumed.sort();
                assert_eq!(consumed, vec!["pos2max".to_string(), "pos2min".to_string()]);
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn out_of_axis_numeric_pos_stays_as_group() {
        // slots=[2], numeric pos1 → still goes to GROUP BY (not aggregated).
        // Same expectation as line_style_groups_by_pos1_and_aggregates_pos2 but
        // explicit about the "numeric out-of-axis" path.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos1__", "__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn discrete_in_axis_pos_stays_as_group_on_centroid_geom() {
        // slots=[1,2], pos1 discrete + pos2 numeric → only pos2 aggregated,
        // pos1 stays as GROUP BY. Confirms numeric check is preserved on
        // slot=[1,2] geoms (e.g. point with category AS x, value AS y).
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = vec![
            ColumnInfo {
                name: "__ggsql_aes_pos1__".to_string(),
                dtype: DataType::Utf8,
                is_discrete: true,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
        ];
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[1, 2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                consumed_aesthetics,
                stat_columns,
                ..
            } => {
                assert!(query.contains("AVG(\"__ggsql_aes_pos2__\")"));
                assert!(!query.contains("AVG(\"__ggsql_aes_pos1__\")"));
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                assert_eq!(consumed_aesthetics, vec!["pos2".to_string()]);
                assert!(stat_columns.contains(&"pos2".to_string()));
                assert!(!stat_columns.contains(&"pos1".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn count_works_with_no_numeric_pos() {
        // slots=[2], only discrete pos1 mapped, aggregate=count → no
        // "needs numeric" error; query has COUNT(*) and groups by pos1.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        let schema = vec![ColumnInfo {
            name: "__ggsql_aes_pos1__".to_string(),
            dtype: DataType::Utf8,
            is_discrete: true,
            min: None,
            max: None,
        }];
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("count".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                ..
            } => {
                assert!(query.contains("COUNT(*)"));
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                assert!(stat_columns.contains(&"count".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    // ========================================================================
    // Range-mode tests (ribbon / range)
    // ========================================================================

    fn range_pair() -> Option<(&'static str, &'static str)> {
        Some(("pos2min", "pos2max"))
    }

    fn range_input_aes_with_group() -> (Mappings, Schema) {
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = vec![
            ColumnInfo {
                name: "__ggsql_aes_pos1__".to_string(),
                dtype: DataType::Utf8,
                is_discrete: true,
                min: None,
                max: None,
            },
            ColumnInfo {
                name: "__ggsql_aes_pos2__".to_string(),
                dtype: DataType::Float64,
                is_discrete: false,
                min: None,
                max: None,
            },
        ];
        (aes, schema)
    }

    #[test]
    fn range_mode_two_functions_emits_one_row_per_group() {
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("mean-sdev".to_string()),
                ArrayElement::String("mean+sdev".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap();
        match result {
            StatResult::Transformed {
                query,
                stat_columns,
                consumed_aesthetics,
                ..
            } => {
                assert!(
                    query.contains(
                        "AVG(\"__ggsql_aes_pos2__\") - STDDEV_POP(\"__ggsql_aes_pos2__\")"
                    ),
                    "lower bound expr missing: {}",
                    query
                );
                assert!(
                    query.contains(
                        "AVG(\"__ggsql_aes_pos2__\") + STDDEV_POP(\"__ggsql_aes_pos2__\")"
                    ),
                    "upper bound expr missing: {}",
                    query
                );
                assert!(query.contains("GROUP BY \"__ggsql_aes_pos1__\""));
                assert!(!query.contains("UNION ALL"));
                assert!(!query.contains("CROSS JOIN"));
                // No `aggregate` tag column in range mode.
                assert!(!query.contains("__ggsql_stat_aggregate__"));
                assert_eq!(
                    stat_columns,
                    vec!["pos2min".to_string(), "pos2max".to_string()]
                );
                assert!(consumed_aesthetics.contains(&"pos2".to_string()));
                assert!(consumed_aesthetics.contains(&"pos2min".to_string()));
                assert!(consumed_aesthetics.contains(&"pos2max".to_string()));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn range_mode_rejects_single_function() {
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exactly two"),
            "expected 'exactly two' in error, got: {}",
            err
        );
    }

    #[test]
    fn range_mode_rejects_three_functions() {
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("min".to_string()),
                ArrayElement::String("mean".to_string()),
                ArrayElement::String("max".to_string()),
            ]),
        );

        let err = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("exactly two"));
    }

    #[test]
    fn range_mode_quantile_uses_inline_when_available() {
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("p25".to_string()),
                ArrayElement::String("p75".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("QUANTILE_CONT"));
                assert!(query.contains("0.25"));
                assert!(query.contains("0.75"));
                assert!(!query.contains("NTILE(4)"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn range_mode_quantile_falls_back_without_dialect_support() {
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("p25".to_string()),
                ArrayElement::String("p75".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &NoInlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("NTILE(4)"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn range_mode_requires_pos2_input() {
        // Range geom but pos2 not mapped → error.
        let mut aes = Mappings::new();
        aes.insert("pos1", col("__ggsql_aes_pos1__"));
        let schema = vec![ColumnInfo {
            name: "__ggsql_aes_pos1__".to_string(),
            dtype: DataType::Utf8,
            is_discrete: true,
            min: None,
            max: None,
        }];
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("mean-sdev".to_string()),
                ArrayElement::String("mean+sdev".to_string()),
            ]),
        );

        let err = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("pos2") || err.contains("`y`"),
            "expected pos2/y mention in error, got: {}",
            err
        );
    }

    // ========================================================================
    // Parser tests (parse_agg_name)
    // ========================================================================

    #[test]
    fn parse_simple_names() {
        assert_eq!(
            parse_agg_name("mean"),
            Some(AggSpec { offset: "mean", band: None })
        );
        assert_eq!(
            parse_agg_name("count"),
            Some(AggSpec { offset: "count", band: None })
        );
        assert_eq!(
            parse_agg_name("p25"),
            Some(AggSpec { offset: "p25", band: None })
        );
    }

    #[test]
    fn parse_band_default_modifier() {
        let spec = parse_agg_name("mean+sdev").unwrap();
        assert_eq!(spec.offset, "mean");
        let band = spec.band.unwrap();
        assert_eq!(band.sign, '+');
        assert_eq!(band.mod_value, 1.0);
        assert_eq!(band.expansion, "sdev");
    }

    #[test]
    fn parse_band_integer_modifier() {
        let spec = parse_agg_name("mean-2sdev").unwrap();
        let band = spec.band.unwrap();
        assert_eq!(band.sign, '-');
        assert_eq!(band.mod_value, 2.0);
        assert_eq!(band.expansion, "sdev");
    }

    #[test]
    fn parse_band_decimal_modifier() {
        let spec = parse_agg_name("mean+1.96sdev").unwrap();
        let band = spec.band.unwrap();
        assert_eq!(band.mod_value, 1.96);
    }

    #[test]
    fn parse_band_longest_offset_wins() {
        // 'median+sdev' must match offset 'median', not 'me' (which isn't an
        // offset anyway, but more pertinently the parser must not stop at a
        // shorter prefix).
        let spec = parse_agg_name("median+sdev").unwrap();
        assert_eq!(spec.offset, "median");
    }

    #[test]
    fn parse_band_percentile_offset() {
        let spec = parse_agg_name("p25+0.5range").unwrap();
        assert_eq!(spec.offset, "p25");
        let band = spec.band.unwrap();
        assert_eq!(band.mod_value, 0.5);
        assert_eq!(band.expansion, "range");
    }

    #[test]
    fn parse_band_rejects_invalid_offset() {
        assert!(parse_agg_name("count+sdev").is_none());
        assert!(parse_agg_name("iqr+sdev").is_none());
    }

    #[test]
    fn parse_band_rejects_invalid_expansion() {
        assert!(parse_agg_name("mean+count").is_none());
        assert!(parse_agg_name("mean+median").is_none());
    }

    #[test]
    fn parse_rejects_unknown() {
        assert!(parse_agg_name("foo").is_none());
        assert!(parse_agg_name("").is_none());
    }

    // ========================================================================
    // Validation tests (validate_aggregate_param)
    // ========================================================================

    #[test]
    fn validate_accepts_simple_names_and_bands() {
        use crate::plot::types::ArrayElement;
        validate_aggregate_param(&ParameterValue::String("mean".to_string())).unwrap();
        validate_aggregate_param(&ParameterValue::String("mean+sdev".to_string())).unwrap();
        validate_aggregate_param(&ParameterValue::String("median-0.5iqr".to_string())).unwrap();
        validate_aggregate_param(&ParameterValue::Array(vec![
            ArrayElement::String("mean".to_string()),
            ArrayElement::String("mean+1.96sdev".to_string()),
        ]))
        .unwrap();
    }

    #[test]
    fn validate_diagnostic_for_invalid_offset() {
        let err = validate_aggregate_param(&ParameterValue::String("count+sdev".to_string()))
            .unwrap_err();
        assert!(err.contains("count"), "err: {}", err);
        assert!(err.contains("offset"), "err: {}", err);
    }

    #[test]
    fn validate_diagnostic_for_invalid_expansion() {
        let err = validate_aggregate_param(&ParameterValue::String("mean+count".to_string()))
            .unwrap_err();
        assert!(err.contains("count"), "err: {}", err);
        assert!(err.contains("expansion"), "err: {}", err);
    }

    #[test]
    fn validate_diagnostic_for_unknown() {
        let err =
            validate_aggregate_param(&ParameterValue::String("foo".to_string())).unwrap_err();
        assert!(err.contains("unknown"), "err: {}", err);
        assert!(err.contains("foo"), "err: {}", err);
    }

    // ========================================================================
    // SQL emission for parametric bands
    // ========================================================================

    #[test]
    fn band_decimal_modifier_emits_in_sql() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean+1.96sdev".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(
                    query.contains("AVG(\"__ggsql_aes_pos2__\") + 1.96 * STDDEV_POP(\"__ggsql_aes_pos2__\")"),
                    "query: {}",
                    query
                );
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn band_with_percentile_offset_inline() {
        // median-0.5iqr on a dialect with inline quantile support.
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("median-0.5iqr".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // median uses QUANTILE_CONT(col, 0.5); iqr uses QUANTILE_CONT(.., 0.75) and 0.25.
                assert!(
                    query.contains("QUANTILE_CONT") && query.contains("0.5"),
                    "query: {}",
                    query
                );
                assert!(query.contains("0.75") && query.contains("0.25"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn band_with_percentile_offset_falls_back() {
        // median+2sdev on a dialect WITHOUT inline quantile support → UNION-ALL
        // path with sql_percentile for median, inline STDDEV_POP for sdev.
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("median+2sdev".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &NoInlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("NTILE(4)"));
                assert!(query.contains("STDDEV_POP"));
                assert!(query.contains("2 * "));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn band_with_default_modifier_omits_one_prefix() {
        let mut aes = Mappings::new();
        aes.insert("pos2", col("__ggsql_aes_pos2__"));
        let schema = numeric_schema(&["__ggsql_aes_pos2__"]);
        let mut params = HashMap::new();
        params.insert(
            "aggregate".to_string(),
            ParameterValue::String("mean+sdev".to_string()),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            None,
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                // mod=1 case: (offset + exp), no `1 *` prefix.
                assert!(
                    query.contains(
                        "AVG(\"__ggsql_aes_pos2__\") + STDDEV_POP(\"__ggsql_aes_pos2__\")"
                    ),
                    "expected `(AVG + STDDEV_POP)` form, got: {}",
                    query
                );
                assert!(!query.contains("1 * STDDEV_POP"));
            }
            _ => panic!("expected Transformed"),
        }
    }

    #[test]
    fn range_mode_supports_decimal_band() {
        // Ribbon range mode + 95% CI band.
        let (aes, schema) = range_input_aes_with_group();
        let mut params = HashMap::new();
        use crate::plot::types::ArrayElement;
        params.insert(
            "aggregate".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::String("mean-1.96sdev".to_string()),
                ArrayElement::String("mean+1.96sdev".to_string()),
            ]),
        );

        let result = apply(
            "SELECT * FROM t",
            &schema,
            &aes,
            &[],
            &params,
            &InlineQuantileDialect,
            &[2],
            range_pair(),
        )
        .unwrap();
        match result {
            StatResult::Transformed { query, .. } => {
                assert!(query.contains("- 1.96 * STDDEV_POP"));
                assert!(query.contains("+ 1.96 * STDDEV_POP"));
            }
            _ => panic!("expected Transformed"),
        }
    }
}
