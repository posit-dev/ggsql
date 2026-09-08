//! Translating resolved ggsql scales into hephaestus scales.
//!
//! ggsql resolves the scale configuration (type, domain, transform, breaks,
//! formatted labels, and — for material aesthetics — a concrete output range);
//! we build the matching hephaestus `Scale`, which performs the value→output
//! mapping at draw time. Palettes are already resolved to concrete values by
//! ggsql's execution stage, so the output range is always an explicit `Array`.

use std::sync::Arc;

use hephaestus::color::{rgba, Color};
use hephaestus::plot::geom::linetype::{dash, gap, pattern, solid};
use hephaestus::plot::scale::{self, Scale as HScale, TransformKind as HTransform};
use hephaestus::scales::value::{
    Date as HDate, DateTime as HDateTime, LinetypeStep, Time as HTime, Value as HValue,
};
use hephaestus::scales::Direction;

use super::channels::{column_to_channel, column_to_f64, ChannelData, NULL_CATEGORY};
use crate::naming;
use crate::plot::aesthetic::POSITION_SUFFIXES;
use crate::plot::scale::{linetype_to_stroke_dash, TransformKind as GTransform};
use crate::plot::{ArrayElement, OutputRange, ParameterValue, Scale as GScale, ScaleTypeKind};
use crate::DataFrame;

/// What kind of visual output a scale's range produces. Selects how a resolved
/// `OutputRange::Array` is mapped onto a hephaestus range — and, for the values
/// a scale never touches, how a literal or an identity column converts into a
/// hephaestus value (`wiring::set_literal_channel` / `constant_material`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum RangeKind {
    /// Position scale — no output range; maps to a `[0, 1]` panel fraction.
    Position,
    /// Color-family aesthetic (fill / stroke): hex/name strings → `Color`.
    Color,
    /// Numeric aesthetic (size / linewidth / opacity): numbers passed through.
    Number,
    /// Marker shape: names resolved against the plot's `ShapeRegistry`.
    Shape,
    /// Line dash pattern: names → builtin linetype patterns.
    Linetype,
    /// Free-form string aesthetic (a font family): names passed through, to be
    /// resolved by whatever consumes them.
    Text,
    /// Boolean aesthetic (italic): flags passed through. A boolean has no
    /// meaningful output range, so only the literal / identity paths use it.
    Bool,
    /// CSS font weight: keyword (`bold`) or numeric string → `100..=900`.
    FontWeight,
    /// Rotation in degrees, as ggsql resolves it → radians, hephaestus's unit.
    Angle,
}

/// Build a hephaestus scale from a resolved ggsql scale. `None` when ggsql
/// resolved no scale type, so there is nothing to register.
pub fn build_scale(scale: &GScale, kind: RangeKind) -> Option<HScale> {
    // No resolved scale type → no scale to register. ggsql is the source of scale
    // truth; the writer never fabricates one.
    let type_kind = scale.scale_type.as_ref().map(|st| st.scale_type_kind())?;
    let transform = scale.transform.as_ref().map(|t| t.transform_kind());

    let mut hs = match type_kind {
        ScaleTypeKind::Discrete => scale::discrete(domain_values(Some(scale))),
        ScaleTypeKind::Ordinal => scale::ordinal(domain_values(Some(scale))),
        ScaleTypeKind::Identity => scale::identity(),
        ScaleTypeKind::Binned => {
            let h_transform = transform.and_then(map_transform);
            let (min, max) = continuous_domain(Some(scale));
            // A binned scale needs at least two edges to have a bin at all;
            // resolution normally supplies them, and the domain's own ends are
            // the only honest stand-in when it hasn't.
            let breaks = Some(scale.numeric_breaks())
                .filter(|edges| edges.len() >= 2)
                .unwrap_or_else(|| vec![min, max]);
            let mut c = scale::binned(min..=max, breaks);
            if let Some(t) = h_transform {
                c = c.with_transform(t);
            }
            c
        }
        ScaleTypeKind::Continuous => {
            let (min, max) = continuous_domain(Some(scale));
            // A temporal channel becomes a calendar-aware scale, so the ticks
            // hephaestus generates for itself (and their labels) are dates
            // rather than epoch numbers. ggsql's own breaks still win where it
            // resolved them — see `apply_breaks`.
            match temporal_scale(transform, min, max) {
                Some(t) => t,
                None => {
                    let mut c = scale::continuous(min..=max);
                    if let Some(t) = transform.and_then(map_transform) {
                        c = c.with_transform(t);
                    }
                    c
                }
            }
        }
    };

    // `SETTING reverse => true` is a property ggsql resolves but does not apply,
    // leaving each writer to flip its own scale — the Vega-Lite writer emits VL's
    // `scale.reverse`, and this is hephaestus's equivalent. Reversal is a property
    // of the *mapping*, not of the domain, so one flag covers every scale kind and
    // both roles: a position axis runs backwards and a material scale walks its
    // palette from the far end, while the domain (and therefore the breaks, the
    // bin edges and the order a legend lists its keys in) stays as ggsql resolved
    // it. That is also what VL's `reverse` means — it flips the range, not the
    // domain — so the two writers order a reversed legend the same way.
    if is_reversed(Some(scale)) {
        hs = hs.with_direction(Direction::Reversed);
    }

    if kind != RangeKind::Position {
        if let Some(OutputRange::Array(values)) = scale.output_range.as_ref() {
            hs = apply_output_range(hs, kind, values);
        }
    }

    // Feed ggsql's resolved breaks + formatted labels for every scale (including
    // under a non-identity transform), so axis/legend ticks match ggsql — and the
    // Vega-Lite writer — exactly. ggsql's breaks pair with the same resolved
    // domain hephaestus reads, so they line up. `apply_breaks` is a no-op when
    // the scale has no resolved breaks.
    Some(apply_breaks(hs, scale, type_kind))
}

/// Build a per-panel position scale for a **free** facet dimension, computing
/// the domain from this panel's own data slices.
///
/// This is a deliberate, scoped exception to ggsql owning all scale domains
/// (fixed dimensions still pass `numeric_domain()` straight through): only free
/// facet dimensions derive a per-panel domain here. Continuous dimensions take
/// the numeric extent of the position family (`pos1`, `pos1min/max/end`, …)
/// present in the slices; discrete/ordinal take the panel's distinct categories;
/// binned dimensions keep ggsql's global bin edges, narrowed to the bins the panel
/// occupies (see [`free_binned_scale`]). ggsql's resolved *continuous* breaks are
/// for the global domain and don't fit a per-panel one, so those ticks are left to
/// hephaestus.
///
/// The *padding* around a computed extent is still ggsql's:
/// [`Scale::expand_range`](crate::plot::Scale::expand_range) applies the scale's
/// own resolved `expand` factors, so a free panel is padded exactly like a fixed
/// axis. Only the extent is derived here, never the expansion policy.
pub fn free_position_scale(
    global: Option<&GScale>,
    dfs: &[&DataFrame],
    base: &str,
) -> Option<HScale> {
    let type_kind = global
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind())
        .unwrap_or(ScaleTypeKind::Continuous);
    let transform = global
        .and_then(|s| s.transform.as_ref())
        .map(|t| t.transform_kind());

    let hs = match type_kind {
        ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal => {
            let vals = panel_categories(global, dfs, base);
            // An empty cell has no categories to free the dimension over; `None`
            // sends the panel back to the shared scale (`PanelScales::use_shared`)
            // rather than registering a domainless axis.
            if vals.is_empty() {
                return None;
            }
            if matches!(type_kind, ScaleTypeKind::Ordinal) {
                scale::ordinal(vals)
            } else {
                scale::discrete(vals)
            }
        }
        ScaleTypeKind::Identity => scale::identity(),
        ScaleTypeKind::Binned => global
            .and_then(|g| free_binned_scale(g, dfs, base))
            // No usable break array → fall back to a plain continuous panel scale.
            .or_else(|| free_continuous_scale(global, dfs, base, transform))?,
        ScaleTypeKind::Continuous => free_continuous_scale(global, dfs, base, transform)?,
    };

    // The same flag the fixed path sets (see [`build_scale`]): freeing a
    // dimension narrows its domain, it does not undo `SETTING reverse => true`.
    Some(if is_reversed(global) {
        hs.with_direction(Direction::Reversed)
    } else {
        hs
    })
}

/// A per-panel continuous position scale over the panel's own data extent.
///
/// A temporal dimension becomes a temporal scale, and keeps ggsql's global break
/// labels narrowed to the panel — the same treatment [`free_binned_scale`] gives
/// bin edges, and what the Vega-Lite writer does with a free temporal axis. The
/// alternative, letting hephaestus pick per-panel calendar ticks, invents breaks
/// ggsql didn't resolve and packs full ISO labels into a panel too narrow to hold
/// them (the writer does no label thinning). A panel no global break
/// falls inside keeps hephaestus's own ticks rather than a bare axis; they are
/// dates either way, because the scale carries the calendar unit.
fn free_continuous_scale(
    global: Option<&GScale>,
    dfs: &[&DataFrame],
    base: &str,
    transform: Option<GTransform>,
) -> Option<HScale> {
    let (min, max) = panel_extent(dfs, base)?;
    // A panel extent is raw data, where `numeric_domain()` would already be
    // expanded, so pad it with the scale's own resolved expansion — otherwise a
    // free panel's marks sit hard against the panel edge while a fixed axis gets
    // 5%, and `SETTING expand` silently stops applying once a dimension is freed.
    let (min, max) = match global {
        Some(g) => g.expand_range(min, max),
        None => (min, max),
    };
    let (min, max) = pad_degenerate(min, max);
    // ggsql's global minors, narrowed to this panel — the same treatment its majors
    // get below. Pinning these is what keeps a panel showing one major from being
    // filled with hephaestus's own sub-unit minors: ggsql derives minors from the
    // global major spacing, so the survivors stay on that grid. `None` (no minors
    // resolved) stays None so the fallback survives; an empty list after filtering is
    // a panel that genuinely contains none.
    let minors: Option<Vec<f64>> = global
        .and_then(|g| g.numeric_minor_breaks())
        .map(|positions| {
            positions
                .into_iter()
                .filter(|pos| *pos >= min && *pos <= max)
                .collect()
        });
    if let Some(hs) = temporal_scale(transform, min, max) {
        let labels: Vec<(HValue, String)> = global
            .map(|g| g.break_labels())
            .unwrap_or_default()
            .into_iter()
            .filter(|(pos, _)| *pos >= min && *pos <= max)
            .map(|(pos, label)| (temporal_value(transform, pos), label))
            .collect();
        // Leave the whole tick set automatic when no global break lands in the panel;
        // pinning minors around ticks hephaestus chose itself would mix two grids.
        return Some(if labels.is_empty() {
            hs
        } else {
            apply_pinned_minors(hs.with_breaks_labeled(labels), minors.as_deref(), transform)
        });
    }
    let mut c = scale::continuous(min..=max);
    if let Some(t) = transform.and_then(map_transform) {
        c = c.with_transform(t);
    }
    Some(apply_pinned_minors(c, minors.as_deref(), transform))
}

/// Pin `minors` (ggsql positions, already narrowed to the target domain) on `hs`,
/// wrapping each as the transform's value variant.
///
/// `None` leaves hephaestus's automatic minors in place — ggsql resolved none, so
/// there is nothing to pass through. `Some(&[])` pins an empty list, which is how
/// hephaestus is told to draw no minors at all: that is `SETTING minor_breaks => 0`
/// arriving intact rather than being mistaken for "nothing to say".
fn apply_pinned_minors(
    hs: HScale,
    minors: Option<&[f64]>,
    transform: Option<GTransform>,
) -> HScale {
    match minors {
        Some(positions) => hs.with_minor_breaks(
            positions
                .iter()
                .map(|pos| temporal_value(transform, *pos))
                .collect(),
        ),
        None => hs,
    }
}

/// A per-panel **binned** position scale: ggsql's globally resolved bin edges,
/// narrowed to the window of bins this panel's data occupies.
///
/// The writer never invents bin boundaries — it only selects from the edges ggsql
/// resolved, and labels them with ggsql's own edge labels. Edges and domain narrow
/// together because a hephaestus binned scale derives band width from its edge
/// count as `1 / (edges - 1)`: keeping every global edge while shrinking the domain
/// would leave each bar a global bin-width wide, hanging off the panel.
///
/// Neither `expand_range` nor pinned minors here: the band width a bar is drawn at
/// assumes the domain spans exactly the edges, so padding the domain would
/// desynchronise bar width from bin width, and a binned axis's ticks are its edges,
/// with nothing to subdivide.
fn free_binned_scale(global: &GScale, dfs: &[&DataFrame], base: &str) -> Option<HScale> {
    let bins = binned_bins(global);
    if bins.is_empty() {
        return None;
    }
    let (lo, hi) = panel_extent(dfs, base)?;
    // The inclusive window of bins covering the panel's extent.
    let first = bins.iter().rposition(|b| b.lower <= lo).unwrap_or(0);
    let last = bins
        .iter()
        .position(|b| b.upper >= hi)
        .unwrap_or(bins.len() - 1);
    let (first, last) = (first.min(last), last);
    let window = &bins[first..=last];

    let mut edges = Vec::with_capacity(window.len() + 1);
    edges.push(window[0].lower);
    edges.extend(window.iter().map(|b| b.upper));
    let mut hs = scale::binned(window[0].lower..=window[window.len() - 1].upper, edges);
    if let Some(t) = global
        .transform
        .as_ref()
        .map(|t| t.transform_kind())
        .and_then(map_transform)
    {
        hs = hs.with_transform(t);
    }
    // ggsql's edge labels, restricted to the edges this panel's window keeps.
    let labels: Vec<(HValue, String)> = global
        .break_labels()
        .into_iter()
        .filter(|(pos, _)| *pos >= window[0].lower && *pos <= window[window.len() - 1].upper)
        .map(|(pos, label)| (HValue::Number(pos), label))
        .collect();
    Some(if labels.is_empty() {
        hs
    } else {
        hs.with_breaks_labeled(labels)
    })
}

/// The finite numeric extent of a position family across the given slices.
fn panel_extent(dfs: &[&DataFrame], base: &str) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for df in dfs {
        // The base aesthetic plus its whole position family, so a panel holding
        // only extents (a bar's `pos2end`, a ribbon's `pos2min`/`max`) still
        // sizes its axis — the same family `execute/scale.rs` trains a fixed
        // scale over.
        for suffix in std::iter::once("").chain(POSITION_SUFFIXES.iter().copied()) {
            let name = naming::aesthetic_column(&format!("{base}{suffix}"));
            if df.column(&name).is_ok() {
                if let Ok(values) = column_to_f64(df, &name) {
                    for v in values.into_iter().filter(|v| v.is_finite()) {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
        }
    }
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// The categories a panel occupies: ggsql's globally resolved domain, narrowed
/// to the levels these slices actually contain and left in the global order.
///
/// Selecting from `input_range` is what keeps a free panel agreeing with a fixed
/// one — the same level order, the same [`channels::NULL_CATEGORY`] sentinel for
/// a null level, and the same value *type* [`column_to_channel`] hands over.
/// Re-deriving the domain from the column's text would break all three. The same
/// narrowing [`free_binned_scale`] does for bin edges.
fn panel_categories(global: Option<&GScale>, dfs: &[&DataFrame], base: &str) -> Vec<HValue> {
    let name = naming::aesthetic_column(base);
    let domain = domain_values(global);
    let mut present = vec![false; domain.len()];
    for df in dfs {
        let Ok(data) = column_to_channel(df, &name) else {
            continue;
        };
        // Matched with `key_eq`, exactly as hephaestus matches data to domain at
        // draw time, so a level counts as present here only if it would resolve
        // there too.
        for value in channel_values(data) {
            if let Some(i) = domain.iter().position(|level| level.key_eq(&value)) {
                present[i] = true;
            }
        }
    }
    domain
        .into_iter()
        .zip(present)
        .filter_map(|(level, present)| present.then_some(level))
        .collect()
}

/// A column's values as the hephaestus values a scale domain is matched against.
fn channel_values(data: ChannelData) -> Vec<HValue> {
    match data {
        ChannelData::Strings(values) => values
            .into_iter()
            .map(|v| HValue::String(Arc::from(v.as_str())))
            .collect(),
        ChannelData::Floats(values) => values.into_iter().map(HValue::Number).collect(),
    }
}

/// Domain for a continuous scale. ggsql's resolved `numeric_domain` is
/// authoritative — it carries ggsql's global, expanded, transform-aware training
/// over every layer and the whole position family — so pass it straight through,
/// exactly as the Vega-Lite writer uses `input_range`.
fn continuous_domain(scale: Option<&GScale>) -> (f64, f64) {
    let domain = scale
        .and_then(|s| s.numeric_domain())
        .filter(|(min, max)| min.is_finite() && max.is_finite())
        .unwrap_or((0.0, 1.0));
    pad_degenerate(domain.0, domain.1)
}

/// Whether the scale carries `SETTING reverse => true`.
pub fn is_reversed(scale: Option<&GScale>) -> bool {
    matches!(
        scale.and_then(|s| s.properties.get("reverse")),
        Some(ParameterValue::Boolean(true))
    )
}

/// Category domain for a discrete/ordinal scale, as hephaestus values, in the
/// order ggsql resolved. `reverse` is a direction on the scale rather than a
/// reordering here — see [`build_scale`].
fn domain_values(scale: Option<&GScale>) -> Vec<HValue> {
    scale
        .and_then(|s| s.input_range.as_ref())
        .map(|range| range.iter().map(category_value).collect())
        .unwrap_or_default()
}

/// A categorical domain entry as a hephaestus value. Identical to
/// [`array_element_to_value`] except for the two levels the data side cannot
/// hand over as themselves, which both sides therefore spell as a string: a
/// null becomes [`channels::NULL_CATEGORY`], and a boolean its category name
/// (`column_to_channel` reads a boolean column as strings — see there).
fn category_value(element: &ArrayElement) -> HValue {
    match element {
        ArrayElement::Null => HValue::String(Arc::from(NULL_CATEGORY)),
        ArrayElement::Boolean(_) => HValue::String(Arc::from(element.to_key_string().as_str())),
        other => array_element_to_value(other),
    }
}

/// Attach the resolved output range to a material scale.
fn apply_output_range(hs: HScale, kind: RangeKind, values: &[ArrayElement]) -> HScale {
    let values: Vec<&ArrayElement> = values.iter().collect();
    match kind {
        RangeKind::Color => hs.range_colors(values.into_iter().filter_map(array_element_to_color)),
        RangeKind::Number => hs.range_numbers(values.into_iter().filter_map(|e| e.to_f64())),
        RangeKind::Shape | RangeKind::Text => hs.range_strings(
            values
                .into_iter()
                .map(|e| Arc::from(e.to_key_string().as_str())),
        ),
        RangeKind::Linetype => {
            hs.range_linetypes(values.into_iter().map(|e| map_linetype(&e.to_key_string())))
        }
        // hephaestus takes a font weight as a number and an angle in radians, so
        // the range converts exactly as a literal on the same channel does.
        RangeKind::FontWeight => hs.range_numbers(
            values
                .into_iter()
                .map(|e| parse_font_weight(&e.to_key_string())),
        ),
        RangeKind::Angle => hs.range_numbers(
            values
                .into_iter()
                .filter_map(|e| e.to_f64())
                .map(f64::to_radians),
        ),
        // Neither a position nor a boolean has an output range: the former maps to
        // a panel fraction, the latter is only ever a literal or identity value.
        RangeKind::Position | RangeKind::Bool => hs,
    }
}

/// Map a ggsql linetype to a hephaestus dash pattern; unknown → solid.
///
/// ggsql accepts both names (`dashed`, `twodash`, …) and ggplot2-style hex
/// patterns (`"1343"` = 1 on, 3 off, 4 on, 3 off), and resolves an *ordinal*
/// linetype scale's range entirely to hex. Both forms go through core's
/// [`linetype_to_stroke_dash`], the same parser the Vega-Lite writer uses, so
/// the two writers draw a given linetype identically — matching a name against
/// hephaestus's own builtins would silently render every hex pattern solid and
/// alias `longdash`/`twodash` onto the wrong ones.
///
/// The resulting on/off lengths are points, which is what hephaestus's linetype
/// steps take.
pub fn map_linetype(name: &str) -> Arc<[LinetypeStep]> {
    let Some(lengths) = linetype_to_stroke_dash(name) else {
        return solid();
    };
    // `pattern` requires strict dash/gap alternation, so an odd-length pattern
    // would panic. The parser doesn't produce one; treat it as unknown anyway.
    if lengths.is_empty() || lengths.len() % 2 != 0 {
        return solid();
    }
    pattern(lengths.iter().enumerate().map(|(i, len)| {
        if i % 2 == 0 {
            dash(*len as f64)
        } else {
            gap(*len as f64)
        }
    }))
}

/// Map a ggsql `fontweight` to hephaestus's numeric CSS weight (100–900);
/// unknown → 400. ggsql accepts either a keyword or a number, matching the
/// Vega-Lite writer's `parse_fontweight_to_numeric`.
pub fn parse_font_weight(value: &str) -> f64 {
    if let Ok(n) = value.parse::<f64>() {
        return n;
    }
    match value.to_lowercase().replace('-', "").as_str() {
        "thin" | "hairline" => 100.0,
        "extralight" | "ultralight" => 200.0,
        "light" => 300.0,
        "medium" => 500.0,
        "semibold" | "demibold" => 600.0,
        "bold" | "bolder" => 700.0,
        "extrabold" | "ultrabold" => 800.0,
        "black" | "heavy" => 900.0,
        _ => 400.0, // normal / regular / unknown
    }
}

/// Feed ggsql's resolved breaks + formatted labels into the hephaestus scale so
/// axis/legend ticks match ggsql exactly (including RENAMING overrides).
///
/// Minor breaks travel the same way, via [`apply_minor_breaks`]: break positions are
/// ggsql's to own, majors and minors alike, so nothing here invents either.
fn apply_breaks(hs: HScale, scale: &GScale, type_kind: ScaleTypeKind) -> HScale {
    let hs = apply_minor_breaks(hs, scale, type_kind);
    let categorical = matches!(type_kind, ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal);
    // A suppressed label means different things either side of this line. On a
    // categorical scale it is `RENAMING <level> => null`, i.e. hide the text but
    // keep the category — dropping it would misalign the axis. On a numeric one
    // it is a binned `oob => 'squish'` terminal, where the edge is not a real
    // boundary and its tick and gridline must go too, exactly as the Vega-Lite
    // writer filters them out of the axis.
    let labels = if categorical {
        scale.break_labels()
    } else {
        scale.visible_break_labels()
    };
    if labels.is_empty() {
        return hs;
    }
    match type_kind {
        ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal => {
            // Pair each label with the category at its resolved position, which
            // for a categorical scale is the 1-based index into `input_range`.
            // Keyed by position rather than zipped, so a break set that doesn't
            // cover every category can't shift every label onto the wrong one.
            let Some(range) = scale.input_range.as_ref() else {
                return hs;
            };
            let pairs: Vec<(HValue, String)> = labels
                .into_iter()
                .filter_map(|(pos, label)| {
                    let index = (pos.round() as usize).checked_sub(1)?;
                    range.get(index).map(|e| (category_value(e), label))
                })
                .collect();
            hs.with_breaks_labeled(pairs)
        }
        // Binned scales included: their breaks are the bin **edges**, labelled by
        // ggsql. Placing an edge break on a binned axis is hephaestus's job, via
        // `Scale::map_break` (a break takes its own domain fraction, where a data
        // value goes to its bin's centre). Composite "lower – upper" range labels
        // belong to keyed legends and facet strips, not axes.
        //
        // A continuous temporal scale takes its breaks as temporal values, matching
        // the variant its own generated breaks come back as, so hephaestus formats
        // any break we don't label as a date rather than as an epoch number.
        _ => {
            let temporal = matches!(type_kind, ScaleTypeKind::Continuous)
                .then(|| scale.transform.as_ref().map(|t| t.transform_kind()))
                .flatten();
            hs.with_breaks_labeled(
                labels
                    .into_iter()
                    .map(|(pos, label)| (temporal_value(temporal, pos), label))
                    .collect(),
            )
        }
    }
}

/// Pin ggsql's resolved minor breaks (sub-ticks / sub-gridlines) so they subdivide
/// ggsql's majors instead of being generated from the domain. Without this a sparse
/// major set — a fixed temporal axis narrowed to one break in a facet panel — gets
/// hephaestus's own sub-unit minors, which read as a dotted rail.
fn apply_minor_breaks(hs: HScale, scale: &GScale, type_kind: ScaleTypeKind) -> HScale {
    // Same variant rule as the majors: a temporal scale's positions go back as
    // typed temporal values, everything else as plain numbers.
    let temporal = matches!(type_kind, ScaleTypeKind::Continuous)
        .then(|| scale.transform.as_ref().map(|t| t.transform_kind()))
        .flatten();
    apply_pinned_minors(hs, scale.numeric_minor_breaks().as_deref(), temporal)
}

/// One bin of a resolved ggsql binned scale: its numeric edges, its centre (the
/// value a binned data column actually carries — see `Binned::pre_stat_transform_sql`),
/// and its display label.
pub struct Bin {
    pub lower: f64,
    pub upper: f64,
    pub centre: f64,
    pub label: String,
}

/// The bins of a resolved binned scale, in the numeric form the writer needs:
/// ggsql's own [`Scale::binned_bins`](crate::plot::Scale::binned_bins) labelling
/// — shared with the Vega-Lite writer, so both name a bin the same way — plus
/// each bin's centre, which is the value a binned data column carries.
pub fn binned_bins(scale: &GScale) -> Vec<Bin> {
    scale
        .binned_bins()
        .into_iter()
        .filter_map(|bin| {
            let (lower, upper) = (bin.lower.to_f64()?, bin.upper.to_f64()?);
            Some(Bin {
                lower,
                upper,
                centre: (lower + upper) / 2.0,
                label: bin.label,
            })
        })
        .collect()
}

/// The bin whose centre is closest to `value` — the join from a binned data cell
/// back to its bin. Nearest-centre rather than an interval test because the
/// column carries centres, which are one per bin and a bin width apart (and a
/// temporal centre is truncated to whole days/seconds on the way through SQL).
pub fn bin_at_centre(bins: &[Bin], value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    bins.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.centre - value)
                .abs()
                .total_cmp(&(b.centre - value).abs())
        })
        .map(|(i, _)| i)
}

/// A hephaestus temporal scale over `min..=max`, in the unit the ggsql temporal
/// transform names: days since epoch for `Date`, microseconds since epoch for
/// `DateTime`, nanoseconds since midnight for `Time` — the same units ggsql's
/// `ArrayElement` uses, which is what a temporal column projects to f64 as.
/// `None` for any non-temporal transform.
fn temporal_scale(transform: Option<GTransform>, min: f64, max: f64) -> Option<HScale> {
    match transform? {
        GTransform::Date => Some(scale::temporal(
            HDate::from_days(min as i32)..=HDate::from_days(max as i32),
        )),
        GTransform::DateTime => Some(scale::temporal(
            HDateTime::from_micros(min as i64)..=HDateTime::from_micros(max as i64),
        )),
        GTransform::Time => Some(scale::temporal(
            HTime::from_nanos(min as i64)..=HTime::from_nanos(max as i64),
        )),
        _ => None,
    }
}

/// Wrap a break position as the value variant its scale works in — the temporal
/// variant under a temporal transform, a plain number otherwise.
fn temporal_value(transform: Option<GTransform>, pos: f64) -> HValue {
    match transform {
        Some(GTransform::Date) => HValue::Date(pos as i32),
        Some(GTransform::DateTime) => HValue::DateTime(pos as i64),
        Some(GTransform::Time) => HValue::Time(pos as i64),
        _ => HValue::Number(pos),
    }
}

/// Map a ggsql transform to its hephaestus equivalent. Cast/temporal transforms
/// have no spacing effect (values arrive already projected to f64), and
/// `Geographic` is identity in value space — it differs from `Identity` only in
/// the breaks it picks, which arrive resolved through `apply_breaks`. All of them
/// map to identity (`None` — hephaestus defaults to identity).
fn map_transform(kind: GTransform) -> Option<HTransform> {
    match kind {
        GTransform::Log10 => Some(HTransform::Log10),
        GTransform::Log2 => Some(HTransform::Log2),
        GTransform::Log => Some(HTransform::Log),
        GTransform::Sqrt => Some(HTransform::Sqrt),
        GTransform::Square => Some(HTransform::Square),
        GTransform::Exp10 => Some(HTransform::Exp10),
        GTransform::Exp2 => Some(HTransform::Exp2),
        GTransform::Exp => Some(HTransform::Exp),
        GTransform::Asinh => Some(HTransform::Asinh),
        GTransform::PseudoLog => Some(HTransform::PseudoLog),
        GTransform::Identity
        | GTransform::Date
        | GTransform::DateTime
        | GTransform::Time
        | GTransform::String
        | GTransform::Bool
        | GTransform::Integer
        | GTransform::Geographic => None,
    }
}

/// Convert a ggsql array element to a hephaestus domain value.
pub fn array_element_to_value(element: &ArrayElement) -> HValue {
    match element {
        ArrayElement::String(s) => HValue::String(Arc::from(s.as_str())),
        ArrayElement::Number(n) => HValue::Number(*n),
        ArrayElement::Boolean(b) => HValue::Bool(*b),
        ArrayElement::Date(d) => HValue::Date(*d),
        ArrayElement::DateTime(dt) => HValue::DateTime(*dt),
        ArrayElement::Time(t) => HValue::Time(*t),
        ArrayElement::Null => HValue::Null,
    }
}

/// Parse a color output-range element (hex or CSS name) into a hephaestus color.
fn array_element_to_color(element: &ArrayElement) -> Option<Color> {
    match element {
        ArrayElement::String(s) => parse_color(s),
        _ => None,
    }
}

/// Parse a CSS color string (hex, name, rgb(), …) into a hephaestus color.
pub fn parse_color(value: &str) -> Option<Color> {
    csscolorparser::parse(value)
        .ok()
        .map(|c| rgba(c.r, c.g, c.b, c.a))
}

/// Widen a domain that is non-finite or zero-width so a continuous scale can map
/// it without dividing by zero.
fn pad_degenerate(min: f64, max: f64) -> (f64, f64) {
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if (max - min).abs() < f64::EPSILON {
        return (min - 0.5, max + 0.5);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A binned scale with the given edges, plus optional per-edge label overrides
    /// and properties, as ggsql's resolution would leave it.
    fn binned_scale(edges: &[f64], props: &[(&str, ParameterValue)]) -> GScale {
        let mut scale = GScale::new("facet1");
        scale.scale_type = Some(crate::plot::ScaleType::binned());
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(edges.iter().map(|e| ArrayElement::Number(*e)).collect()),
        );
        for (key, value) in props {
            scale.properties.insert(key.to_string(), value.clone());
        }
        // ggsql always populates `label_mapping` for a resolved scale (the default
        // `{}` template applied to every edge).
        let mut mapping: HashMap<String, Option<String>> = HashMap::new();
        for edge in edges {
            let key = ArrayElement::Number(*edge).to_key_string();
            mapping.insert(key.clone(), Some(key));
        }
        scale.label_mapping = Some(mapping);
        scale
    }

    fn labels(bins: &[Bin]) -> Vec<&str> {
        bins.iter().map(|b| b.label.as_str()).collect()
    }

    #[test]
    fn binned_bins_labels_ranges() {
        let bins = binned_bins(&binned_scale(&[0.0, 10.0, 20.0], &[]));
        assert_eq!(labels(&bins), vec!["0 – 10", "10 – 20"]);
        assert_eq!(bins[0].centre, 5.0);
        assert_eq!(bins[1].centre, 15.0);
    }

    #[test]
    fn binned_bins_honors_edge_renaming() {
        let mut scale = binned_scale(&[0.0, 10.0, 20.0], &[]);
        scale
            .label_mapping
            .as_mut()
            .unwrap()
            .insert("10".to_string(), Some("ten".to_string()));
        assert_eq!(labels(&binned_bins(&scale)), vec!["0 – ten", "ten – 20"]);
    }

    #[test]
    fn binned_bins_opens_suppressed_terminals() {
        // `oob => 'squish'` suppresses the terminal edge labels.
        let mut scale = binned_scale(&[0.0, 10.0, 20.0], &[]);
        let mapping = scale.label_mapping.as_mut().unwrap();
        mapping.insert("0".to_string(), None);
        mapping.insert("20".to_string(), None);
        assert_eq!(labels(&binned_bins(&scale)), vec!["< 10", "≥ 10"]);

        scale.properties.insert(
            "closed".to_string(),
            ParameterValue::String("right".to_string()),
        );
        assert_eq!(labels(&binned_bins(&scale)), vec!["≤ 10", "> 10"]);
    }

    #[test]
    fn binned_bins_empty_without_breaks() {
        assert!(binned_bins(&GScale::new("facet1")).is_empty());
        assert!(binned_bins(&binned_scale(&[5.0], &[])).is_empty());
    }

    /// A resolved continuous Date scale: domain and breaks in days since epoch,
    /// labelled the way ggsql's resolution leaves them (ISO keys).
    fn date_scale(domain: (i32, i32), breaks: &[i32]) -> GScale {
        let mut scale = GScale::new("pos1");
        scale.scale_type = Some(crate::plot::scale::ScaleType::continuous());
        scale.transform = Some(crate::plot::scale::transform::Transform::date());
        scale.input_range = Some(vec![
            ArrayElement::Date(domain.0),
            ArrayElement::Date(domain.1),
        ]);
        scale.properties.insert(
            "breaks".to_string(),
            ParameterValue::Array(breaks.iter().map(|d| ArrayElement::Date(*d)).collect()),
        );
        scale
    }

    #[test]
    fn temporal_scale_labels_ggsql_breaks_as_dates() {
        let scale = date_scale((1208, 1264), &[1208, 1236, 1264]);
        let hs = build_scale(&scale, RangeKind::Position).expect("scale");
        let locale = hephaestus::scales::locale::Locale::EN_US;
        let labels: Vec<String> = hs.breaks(5).iter().map(|b| hs.format(b, &locale)).collect();
        assert_eq!(labels, vec!["1973-04-23", "1973-05-21", "1973-06-18"]);
    }

    #[test]
    fn temporal_scale_is_calendar_aware() {
        // Breaks hephaestus generates for itself come back as dates, not as the
        // epoch-day numbers the domain is stored in — that is what a panel scale
        // with no in-window ggsql break falls back on.
        let hs = temporal_scale(Some(GTransform::Date), 1208.0, 1400.0).expect("temporal scale");
        let locale = hephaestus::scales::locale::Locale::EN_US;
        for label in hs.breaks(5).iter().map(|b| hs.format(b, &locale)) {
            assert!(
                label.starts_with("197"),
                "expected a date label, got {label}"
            );
        }
        assert!(temporal_scale(Some(GTransform::Log10), 1.0, 10.0).is_none());
        assert!(temporal_scale(None, 1.0, 10.0).is_none());
    }

    #[test]
    fn boolean_domain_matches_the_data_side() {
        // hephaestus matches data to domain by `Value` variant, so a boolean
        // column's two representations have to agree: `column_to_channel` reads
        // one as category strings, and the domain must spell them the same way.
        let mut scale = GScale::new("fill");
        scale.scale_type = Some(crate::plot::ScaleType::discrete());
        scale.input_range = Some(vec![
            ArrayElement::Boolean(false),
            ArrayElement::Boolean(true),
        ]);
        let domain = domain_values(Some(&scale));
        let expected = [
            HValue::String(Arc::from("false")),
            HValue::String(Arc::from("true")),
        ];
        assert_eq!(domain.len(), expected.len());
        for (got, want) in domain.iter().zip(&expected) {
            assert!(got.key_eq(want), "expected {want:?}, got {got:?}");
        }

        let df = crate::df! { "flag" => vec![true, false] }.unwrap();
        let ChannelData::Strings(values) = column_to_channel(&df, "flag").unwrap() else {
            panic!("a boolean column should arrive as category strings");
        };
        for value in values {
            let value = HValue::String(Arc::from(value.as_str()));
            assert!(
                domain.iter().any(|level| level.key_eq(&value)),
                "no domain level matches {value:?}"
            );
        }
    }

    #[test]
    fn bin_at_centre_finds_nearest_bin() {
        let bins = binned_bins(&binned_scale(&[0.0, 10.0, 20.0, 30.0], &[]));
        assert_eq!(bin_at_centre(&bins, 5.0), Some(0));
        assert_eq!(bin_at_centre(&bins, 15.0), Some(1));
        assert_eq!(bin_at_centre(&bins, 25.0), Some(2));
        // Tolerant of a centre truncated on the way through SQL.
        assert_eq!(bin_at_centre(&bins, 14.0), Some(1));
        // Out of range still lands on the closest bin; NaN does not.
        assert_eq!(bin_at_centre(&bins, 99.0), Some(2));
        assert_eq!(bin_at_centre(&bins, f64::NAN), None);
        assert_eq!(bin_at_centre(&[], 5.0), None);
    }

    #[test]
    fn font_weights_parse_like_vegalite() {
        // Keywords, in either casing and with or without the hyphen.
        assert_eq!(parse_font_weight("bold"), 700.0);
        assert_eq!(parse_font_weight("Bold"), 700.0);
        assert_eq!(parse_font_weight("semi-bold"), 600.0);
        assert_eq!(parse_font_weight("extralight"), 200.0);
        // Numbers pass through, as a string or as ggsql's own number formatting.
        assert_eq!(parse_font_weight("350"), 350.0);
        assert_eq!(parse_font_weight("350.0"), 350.0);
        // Anything unrecognised is regular, never a missing glyph.
        assert_eq!(parse_font_weight("normal"), 400.0);
        assert_eq!(parse_font_weight("wingdings"), 400.0);
    }
}
