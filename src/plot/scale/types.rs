//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::types::{ArrayElement, ParameterValue, Parameters};
use super::scale_type::ScaleType;
use super::transform::Transform;

/// One bin of a resolved binned scale: the edges bounding it and the text that
/// names it. See [`Scale::binned_bins`].
#[derive(Debug, Clone, PartialEq)]
pub struct BinLabel {
    pub lower: ArrayElement,
    pub upper: ArrayElement,
    pub label: String,
}

/// Scale configuration (from SCALE clause)
///
/// Syntax: `SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]`
///
/// Examples:
/// - `SCALE x VIA date`
/// - `SCALE CONTINUOUS y FROM (0, 100)`
/// - `SCALE DISCRETE color FROM ('A', 'B') TO ('red', 'blue')`
/// - `SCALE color TO viridis`
/// - `SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    /// Specified as modifier: SCALE x VIA date, SCALE CONTINUOUS y
    pub scale_type: Option<ScaleType>,
    /// Input range specification (FROM clause)
    /// Maps to Vega-Lite's scale.domain
    pub input_range: Option<Vec<ArrayElement>>,
    /// Whether the input_range was explicitly specified by the user (FROM clause).
    /// Used to determine whether to apply pre-stat OOB handling in SQL.
    /// If true, the range was specified explicitly (e.g., `FROM ('A', 'B')`).
    /// If false, the range was inferred from the data.
    #[serde(default)]
    pub explicit_input_range: bool,
    /// Output range specification (TO clause)
    /// Either explicit values or a named palette
    pub output_range: Option<OutputRange>,
    /// Transformation (VIA clause)
    pub transform: Option<Transform>,
    /// Whether the transform was explicitly specified by the user (VIA clause).
    /// Used to determine whether to apply type casting in binned scales.
    /// If true, the transform was specified explicitly (e.g., `VIA date`).
    /// If false, the transform was inferred from the column data type.
    #[serde(default)]
    pub explicit_transform: bool,
    /// Additional scale properties (SETTING clause)
    /// Note: `breaks` and `minor_breaks` can each be a Number (count), Array
    /// (explicit positions) or String (temporal interval). Whatever the user wrote,
    /// both are converted to an Array of positions during resolution.
    pub properties: Parameters,
    /// Whether this scale has been resolved (set by resolve() method)
    /// Used to skip re-resolution of pre-resolved scales (e.g., Binned scales)
    #[serde(default)]
    pub resolved: bool,
    /// Label mappings for custom axis/legend labels (RENAMING clause)
    /// Maps raw data values to display labels. `None` value suppresses the label.
    /// Example: `RENAMING 'A' => 'Alpha', 'internal' => NULL`
    #[serde(default)]
    pub label_mapping: Option<HashMap<String, Option<String>>>,
    /// Template for generating labels from scale values (e.g., "{} units")
    /// Default is "{}" which passes through the value unchanged.
    /// The `{}` placeholder is replaced with each value at resolution time.
    /// Example: "{} units" -> {"0": "0 units", "25": "25 units", ...}
    #[serde(default = "crate::format::default_template")]
    pub label_template: String,
}

impl Scale {
    /// Create a new Scale with just an aesthetic name
    pub fn new(aesthetic: impl Into<String>) -> Self {
        Self {
            aesthetic: aesthetic.into(),
            scale_type: None,
            input_range: None,
            explicit_input_range: false,
            output_range: None,
            transform: None,
            explicit_transform: false,
            properties: Parameters::new(),
            resolved: false,
            label_mapping: None,
            label_template: "{}".to_string(),
        }
    }

    /// Whether the scale's domain consists solely of the stat-dummy sentinel.
    ///
    /// Dummy scales are injected when a stat requires a position channel that
    /// the user didn't map (e.g., a bar chart's categorical axis for a
    /// pie chart). Axes for dummy scales should be suppressed.
    pub fn is_dummy(&self) -> bool {
        matches!(
            self.input_range.as_deref(),
            Some([ArrayElement::String(s)]) if s == &crate::naming::stat_column("dummy")
        )
    }

    /// Numeric break positions (after resolution).
    ///
    /// Delegates to the scale type for type-specific logic (e.g. discrete
    /// scales synthesize `[1, 2, …, n]` from the input range length).
    pub fn numeric_breaks(&self) -> Vec<f64> {
        match &self.scale_type {
            Some(st) => st.numeric_breaks(self),
            None => match self.properties.get("breaks") {
                Some(ParameterValue::Array(breaks)) => {
                    breaks.iter().filter_map(|b| b.to_f64()).collect()
                }
                _ => Vec::new(),
            },
        }
    }

    /// Numeric minor break positions (after resolution), from the `minor_breaks`
    /// setting.
    ///
    /// Minor breaks carry no labels — they are the sub-ticks / sub-gridlines between
    /// majors. Returns an `Option`, unlike [`numeric_breaks`](Self::numeric_breaks),
    /// because "resolved to none" and "not resolved" have to be told apart:
    ///
    /// - `Some(positions)`, possibly **empty** — resolution ran. An empty vector is
    ///   the user asking for no minors (`SETTING minor_breaks => 0`) and a consumer
    ///   must honour it by drawing none.
    /// - `None` — no minor breaks were resolved, either because the scale type has
    ///   none (discrete, ordinal, binned — a binned axis's ticks are its bin edges)
    ///   or because the scale is unresolved. A consumer is free to fall back on its
    ///   own algorithm.
    pub fn numeric_minor_breaks(&self) -> Option<Vec<f64>> {
        match self.properties.get("minor_breaks") {
            Some(ParameterValue::Array(breaks)) => {
                Some(breaks.iter().filter_map(|b| b.to_f64()).collect())
            }
            _ => None,
        }
    }

    /// Labelled breaks: `(numeric_position, display_label)` pairs.
    ///
    /// Delegates to the scale type, then applies `label_mapping` overrides.
    /// Suppressed labels (`None` in the mapping) become empty strings — the
    /// break is kept, but goes unlabelled. Use [`Self::visible_break_labels`]
    /// when a suppressed break should disappear entirely.
    pub fn break_labels(&self) -> Vec<(f64, String)> {
        self.labelled_breaks()
            .into_iter()
            .map(|(pos, label)| (pos, label.unwrap_or_default()))
            .collect()
    }

    /// Labelled breaks with suppressed ones **dropped**, not blanked.
    ///
    /// A binned scale under `oob => 'squish'` suppresses its two terminal
    /// breaks: the outermost bins are open-ended, so the edge values they would
    /// be labelled with are not real boundaries. Leaving the break in place with
    /// an empty label still draws its tick and gridline, which reads as a
    /// boundary that isn't there — so the whole break goes.
    pub fn visible_break_labels(&self) -> Vec<(f64, String)> {
        self.labelled_breaks()
            .into_iter()
            .filter_map(|(pos, label)| label.map(|l| (pos, l)))
            .collect()
    }

    /// Breaks paired with their resolved label, where `None` means the label was
    /// explicitly suppressed (as opposed to merely empty). The two public break
    /// accessors differ only in what they do with that `None`.
    fn labelled_breaks(&self) -> Vec<(f64, Option<String>)> {
        let raw = match &self.scale_type {
            Some(st) => st.break_labels(self),
            None => self
                .numeric_breaks()
                .into_iter()
                .map(|v| (v, format!("{v}")))
                .collect(),
        };
        let mappings = self.label_mapping.as_ref();
        let mut out = Vec::with_capacity(raw.len());
        for (pos, label) in raw {
            match mappings.and_then(|m| m.get(&label)) {
                Some(Some(renamed)) => out.push((pos, Some(renamed.clone()))),
                Some(None) => out.push((pos, None)),
                None => out.push((pos, Some(label))),
            }
        }
        out
    }

    /// The bins of a binned scale, each with its edges and display label.
    ///
    /// One definition of the bin-labelling contract, for every consumer that has
    /// to name a bin: `"lower – upper"` (en dash) with per-edge `RENAMING`
    /// applied to each side, and an open form (`≤`/`<` at the bottom, `>`/`≥` at
    /// the top, per the scale's `closed` side) where a *terminal* edge's label is
    /// suppressed — which is what `oob => 'squish'` does, because the outermost
    /// bins then reach past the edge value and it is no longer a real boundary.
    ///
    /// Empty for a scale with no resolved break array, and for any scale type
    /// other than binned.
    pub fn binned_bins(&self) -> Vec<BinLabel> {
        if self.scale_type.as_ref().map(|st| st.scale_type_kind())
            != Some(super::ScaleTypeKind::Binned)
        {
            return Vec::new();
        }
        let Some(ParameterValue::Array(breaks)) = self.properties.get("breaks") else {
            return Vec::new();
        };
        if breaks.len() < 2 {
            return Vec::new();
        }
        let closed_right = matches!(
            self.properties.get("closed"),
            Some(ParameterValue::String(s)) if s == "right"
        );
        let mapping = self.label_mapping.as_ref();
        let last = breaks.len() - 2;

        (0..=last)
            .map(|i| {
                let (lower, upper) = (breaks[i].clone(), breaks[i + 1].clone());
                let (lo_key, hi_key) = (lower.to_key_string(), upper.to_key_string());
                let suppressed = |key: &str| matches!(mapping.and_then(|m| m.get(key)), Some(None));
                let label_of = |key: &str| {
                    mapping
                        .and_then(|m| m.get(key))
                        .cloned()
                        .flatten()
                        .unwrap_or_else(|| key.to_string())
                };
                let label = if i == 0 && suppressed(&lo_key) {
                    let symbol = if closed_right { "≤" } else { "<" };
                    format!("{symbol} {}", label_of(&hi_key))
                } else if i == last && suppressed(&hi_key) {
                    let symbol = if closed_right { ">" } else { "≥" };
                    format!("{symbol} {}", label_of(&lo_key))
                } else {
                    format!("{} – {}", label_of(&lo_key), label_of(&hi_key))
                };
                BinLabel {
                    lower,
                    upper,
                    label,
                }
            })
            .collect()
    }

    /// Numeric domain as `(min, max)` from the resolved input range.
    ///
    /// Delegates to the scale type for type-specific logic (e.g. discrete
    /// scales synthesize `(0.5, n + 0.5)` so integer positions sit at
    /// category centres).
    pub fn numeric_domain(&self) -> Option<(f64, f64)> {
        match &self.scale_type {
            Some(st) => st.numeric_domain(self),
            None => {
                let range = self.input_range.as_ref()?;
                let min = range.first()?.to_f64()?;
                let max = range.last()?.to_f64()?;
                Some((min, max))
            }
        }
    }

    /// Apply this scale's resolved expansion to a caller-computed `(min, max)`.
    ///
    /// [`numeric_domain`](Self::numeric_domain) is already expanded, so this is
    /// only for a consumer that had to derive a range ggsql did *not* resolve —
    /// today, a writer computing a per-panel domain for a **free** facet
    /// dimension. Going through this method rather than re-deriving the formula
    /// keeps a free panel padded exactly like a fixed axis, honours
    /// `SETTING expand`, and picks up context-dependent factors the caller can't
    /// see (a polar full-circle theta resolves to zero expansion).
    ///
    /// Mirrors resolution: expand, then clip to the transform's allowed domain.
    /// On an unresolved scale the factors fall back to the continuous defaults.
    pub fn expand_range(&self, min: f64, max: f64) -> (f64, f64) {
        let (mult, add) = super::scale_type::get_expand_factors(&self.properties);
        let expanded = super::scale_type::expand_numeric_range(
            &[ArrayElement::Number(min), ArrayElement::Number(max)],
            mult,
            add,
        );
        let clipped = match &self.transform {
            Some(t) => super::scale_type::clip_to_transform_domain(&expanded, t),
            None => expanded,
        };
        match (
            clipped.first().and_then(|e| e.to_f64()),
            clipped.last().and_then(|e| e.to_f64()),
        ) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => (min, max),
        }
    }
}

/// Output range specification (TO clause)
/// Either explicit values or a named palette identifier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputRange {
    /// Explicit array of values: TO ('red', 'blue')
    Array(Vec<ArrayElement>),
    /// Named palette identifier: TO viridis
    Palette(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn continuous_scale(domain: (f64, f64), breaks: Vec<f64>) -> Scale {
        let mut s = Scale::new("pos1");
        s.scale_type = Some(ScaleType::continuous());
        s.input_range = Some(vec![
            ArrayElement::Number(domain.0),
            ArrayElement::Number(domain.1),
        ]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(breaks.into_iter().map(ArrayElement::Number).collect()),
        );
        s
    }

    fn discrete_scale(values: &[&str]) -> Scale {
        let mut s = Scale::new("pos2");
        s.scale_type = Some(ScaleType::discrete());
        s.input_range = Some(
            values
                .iter()
                .map(|v| ArrayElement::String(v.to_string()))
                .collect(),
        );
        s
    }

    fn ordinal_scale(values: &[&str]) -> Scale {
        let mut s = Scale::new("pos1");
        s.scale_type = Some(ScaleType::ordinal());
        s.input_range = Some(
            values
                .iter()
                .map(|v| ArrayElement::String(v.to_string()))
                .collect(),
        );
        s
    }

    // =========================================================================
    // Continuous
    // =========================================================================

    #[test]
    fn test_continuous_numeric_breaks() {
        let s = continuous_scale((0.0, 100.0), vec![25.0, 50.0, 75.0]);
        assert_eq!(s.numeric_breaks(), vec![25.0, 50.0, 75.0]);
    }

    #[test]
    fn test_continuous_numeric_domain() {
        let s = continuous_scale((0.0, 100.0), vec![]);
        assert_eq!(s.numeric_domain(), Some((0.0, 100.0)));
    }

    #[test]
    fn test_continuous_no_breaks() {
        let s = continuous_scale((0.0, 100.0), vec![]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
    }

    // =========================================================================
    // Discrete
    // =========================================================================

    #[test]
    fn test_discrete_numeric_breaks() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(s.numeric_breaks(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_discrete_numeric_domain() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(s.numeric_domain(), Some((0.5, 3.5)));
    }

    #[test]
    fn test_discrete_single_category() {
        let s = discrete_scale(&["only"]);
        assert_eq!(s.numeric_breaks(), vec![1.0]);
        assert_eq!(s.numeric_domain(), Some((0.5, 1.5)));
    }

    #[test]
    fn test_discrete_empty() {
        let s = discrete_scale(&[]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
        assert_eq!(s.numeric_domain(), None);
    }

    // =========================================================================
    // Ordinal
    // =========================================================================

    #[test]
    fn test_ordinal_numeric_breaks() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(s.numeric_breaks(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ordinal_numeric_domain() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(s.numeric_domain(), Some((0.5, 3.5)));
    }

    // =========================================================================
    // Identity / no scale type
    // =========================================================================

    #[test]
    fn test_identity_string_returns_empty() {
        let mut s = Scale::new("color");
        s.scale_type = Some(ScaleType::identity());
        s.input_range = Some(vec![
            ArrayElement::String("red".to_string()),
            ArrayElement::String("blue".to_string()),
        ]);
        assert_eq!(s.numeric_breaks(), Vec::<f64>::new());
        assert_eq!(s.numeric_domain(), None);
    }

    #[test]
    fn test_no_scale_type_falls_back() {
        let mut s = Scale::new("pos1");
        s.input_range = Some(vec![ArrayElement::Number(10.0), ArrayElement::Number(50.0)]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(20.0), ArrayElement::Number(40.0)]),
        );
        assert_eq!(s.numeric_breaks(), vec![20.0, 40.0]);
        assert_eq!(s.numeric_domain(), Some((10.0, 50.0)));
    }

    // =========================================================================
    // break_labels
    // =========================================================================

    #[test]
    fn test_continuous_break_labels() {
        let s = continuous_scale((0.0, 100.0), vec![25.0, 50.0, 75.0]);
        assert_eq!(
            s.break_labels(),
            vec![
                (25.0, "25".to_string()),
                (50.0, "50".to_string()),
                (75.0, "75".to_string())
            ]
        );
    }

    #[test]
    fn test_discrete_break_labels() {
        let s = discrete_scale(&["A", "B", "C"]);
        assert_eq!(
            s.break_labels(),
            vec![
                (1.0, "A".to_string()),
                (2.0, "B".to_string()),
                (3.0, "C".to_string())
            ]
        );
    }

    #[test]
    fn test_ordinal_break_labels() {
        let s = ordinal_scale(&["low", "mid", "high"]);
        assert_eq!(
            s.break_labels(),
            vec![
                (1.0, "low".to_string()),
                (2.0, "mid".to_string()),
                (3.0, "high".to_string())
            ]
        );
    }

    #[test]
    fn test_temporal_break_labels_are_iso_strings() {
        // 1208 days since epoch = 1973-04-23. A temporal break labels itself from
        // its own value, not from the epoch number its position projects to.
        let mut s = continuous_scale((1208.0, 1264.0), vec![]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![
                ArrayElement::Date(1208),
                ArrayElement::Date(1236),
                ArrayElement::Date(1264),
            ]),
        );
        assert_eq!(
            s.break_labels(),
            vec![
                (1208.0, "1973-04-23".to_string()),
                (1236.0, "1973-05-21".to_string()),
                (1264.0, "1973-06-18".to_string())
            ]
        );
    }

    #[test]
    fn test_temporal_break_labels_honour_mapping() {
        // `label_mapping` is keyed by `to_key_string()`, so a RENAMING override on
        // a temporal break has to be found under the ISO key.
        let mut s = continuous_scale((1208.0, 1236.0), vec![]);
        s.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Date(1208), ArrayElement::Date(1236)]),
        );
        let mut mapping = HashMap::new();
        mapping.insert("1973-04-23".to_string(), Some("Apr 23".to_string()));
        mapping.insert("1973-05-21".to_string(), None);
        s.label_mapping = Some(mapping);
        assert_eq!(
            s.break_labels(),
            vec![(1208.0, "Apr 23".to_string()), (1236.0, String::new())]
        );
    }

    #[test]
    fn test_break_labels_with_mapping() {
        let mut s = discrete_scale(&["A", "B", "C"]);
        let mut mapping = HashMap::new();
        mapping.insert("A".to_string(), Some("Alpha".to_string()));
        mapping.insert("C".to_string(), None);
        s.label_mapping = Some(mapping);
        assert_eq!(
            s.break_labels(),
            vec![
                (1.0, "Alpha".to_string()),
                (2.0, "B".to_string()),
                (3.0, String::new())
            ]
        );
    }

    #[test]
    fn test_numeric_minor_breaks_is_none_until_resolved() {
        // Unresolved, so a consumer may fall back to its own algorithm. A setting
        // value that resolution hasn't converted yet reads the same way.
        let mut s = continuous_scale((0.0, 100.0), vec![0.0, 50.0, 100.0]);
        assert_eq!(s.numeric_minor_breaks(), None);
        s.properties
            .insert("minor_breaks".to_string(), ParameterValue::Number(3.0));
        assert_eq!(s.numeric_minor_breaks(), None);
    }

    #[test]
    fn test_numeric_minor_breaks_reads_resolved_positions() {
        let mut s = continuous_scale((0.0, 100.0), vec![0.0, 50.0, 100.0]);
        s.properties.insert(
            "minor_breaks".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(25.0), ArrayElement::Number(75.0)]),
        );
        assert_eq!(s.numeric_minor_breaks(), Some(vec![25.0, 75.0]));
    }

    #[test]
    fn test_numeric_minor_breaks_distinguishes_resolved_none() {
        // `minor_breaks => 0` resolves to an empty array, which must not read as
        // "unresolved" — a consumer has to draw none rather than invent some.
        let mut s = continuous_scale((0.0, 100.0), vec![0.0, 50.0, 100.0]);
        s.properties.insert(
            "minor_breaks".to_string(),
            ParameterValue::Array(Vec::new()),
        );
        assert_eq!(s.numeric_minor_breaks(), Some(Vec::new()));
    }

    #[test]
    fn test_expand_range_uses_the_continuous_default() {
        // No `expand` property → the 5%-of-span default, both ends.
        let s = continuous_scale((0.0, 100.0), vec![]);
        assert_eq!(s.expand_range(0.0, 100.0), (-5.0, 105.0));
    }

    #[test]
    fn test_expand_range_honours_the_setting() {
        // Multiplier only, then the [mult, add] pair resolution writes back.
        let mut s = continuous_scale((0.0, 100.0), vec![]);
        s.properties
            .insert("expand".to_string(), ParameterValue::Number(0.1));
        assert_eq!(s.expand_range(0.0, 100.0), (-10.0, 110.0));

        s.properties.insert(
            "expand".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.1), ArrayElement::Number(1.0)]),
        );
        assert_eq!(s.expand_range(0.0, 100.0), (-11.0, 111.0));
    }

    #[test]
    fn test_expand_range_zero_is_exact() {
        // What a polar full-circle theta resolves to: no padding at all, so a
        // free panel doesn't open a gap in the pie.
        let mut s = continuous_scale((0.0, 100.0), vec![]);
        s.properties.insert(
            "expand".to_string(),
            ParameterValue::Array(vec![ArrayElement::Number(0.0), ArrayElement::Number(0.0)]),
        );
        assert_eq!(s.expand_range(0.0, 100.0), (0.0, 100.0));
    }

    #[test]
    fn test_expand_range_clips_to_the_transform_domain() {
        // Mirrors resolution: expanding below zero on a log scale clips to the
        // transform's allowed minimum rather than producing an invalid domain.
        let mut s = continuous_scale((1.0, 1000.0), vec![]);
        s.transform = Some(Transform::log());
        let (lo, hi) = s.expand_range(1.0, 1000.0);
        assert_eq!(lo, f64::MIN_POSITIVE);
        assert!(
            (hi - 1049.95).abs() < 1e-9,
            "upper end expands normally: {hi}"
        );
    }
}
