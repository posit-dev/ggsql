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

use super::channels::NULL_CATEGORY;
use crate::plot::scale::{linetype_to_stroke_dash, TransformKind as GTransform};
use crate::plot::{ArrayElement, OutputRange, ParameterValue, Scale as GScale, ScaleTypeKind};

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

/// Build a per-panel position scale for a **free** facet dimension from core's
/// per-panel resolution ([`GScale::panels`]).
///
/// Core resolves the per-panel domain, breaks, labels, and minor breaks with
/// the same machinery as the global scale, so this writer only translates —
/// it never derives a positional extent itself. `None` when core resolved no
/// per-panel entry (an empty facet cell), sending the panel back to the shared
/// scale (`PanelScales::use_shared`).
pub fn free_position_scale(global: Option<&GScale>, panel_index: usize) -> Option<HScale> {
    let g = global?;
    let panel = g.panels.as_ref()?.get(panel_index)?.as_ref()?;
    let type_kind = g
        .scale_type
        .as_ref()
        .map(|st| st.scale_type_kind())
        .unwrap_or(ScaleTypeKind::Continuous);
    let transform = g.transform.as_ref().map(|t| t.transform_kind());

    let hs = match type_kind {
        ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal => {
            let vals: Vec<HValue> = panel.input_range.iter().map(category_value).collect();
            // An empty cell has no categories to free the dimension over.
            if vals.is_empty() {
                return None;
            }
            let mut hs = if matches!(type_kind, ScaleTypeKind::Ordinal) {
                scale::ordinal(vals)
            } else {
                scale::discrete(vals)
            };
            // Pair each label with the category at its resolved position — the
            // 1-based index into the panel's domain, exactly as the fixed path
            // (`apply_breaks`) pairs against the global domain. A suppressed
            // label blanks the text but keeps the category.
            let pairs: Vec<(HValue, String)> = panel
                .breaks
                .iter()
                .filter_map(|(pos, label)| {
                    let index = (pos.round() as usize).checked_sub(1)?;
                    Some((
                        category_value(panel.input_range.get(index)?),
                        label.clone().unwrap_or_default(),
                    ))
                })
                .collect();
            if !pairs.is_empty() {
                hs = hs.with_breaks_labeled(pairs);
            }
            hs
        }
        ScaleTypeKind::Identity => scale::identity(),
        ScaleTypeKind::Binned => {
            // The panel's domain is the narrowed bin-edge window and its breaks
            // are those edges — a hephaestus binned scale derives band width
            // from its edge count, so the two must narrow together.
            let (min, max) = panel_range(&panel.input_range)?;
            let edges: Vec<f64> = panel.breaks.iter().map(|(pos, _)| *pos).collect();
            let edges = if edges.len() >= 2 {
                edges
            } else {
                vec![min, max]
            };
            let mut hs = scale::binned(min..=max, edges);
            if let Some(t) = transform.and_then(map_transform) {
                hs = hs.with_transform(t);
            }
            let labels = visible_numbered_labels(&panel.breaks, None);
            if !labels.is_empty() {
                hs = hs.with_breaks_labeled(labels);
            }
            hs
        }
        ScaleTypeKind::Continuous => {
            // The panel domain arrives already expanded and transform-clipped —
            // core ran the same resolve as for a fixed axis.
            let (min, max) = panel_range(&panel.input_range)?;
            let (min, max) = pad_degenerate(min, max);
            let labels = visible_numbered_labels(&panel.breaks, transform);
            match temporal_scale(transform, min, max) {
                Some(hs) => {
                    if labels.is_empty() {
                        // Leave the tick set automatic when core resolved no
                        // breaks; pinning minors around ticks hephaestus chose
                        // itself would mix two grids.
                        hs
                    } else {
                        apply_pinned_minors(
                            hs.with_breaks_labeled(labels),
                            panel.minor_breaks.as_deref(),
                            transform,
                        )
                    }
                }
                None => {
                    let mut c = scale::continuous(min..=max);
                    if let Some(t) = transform.and_then(map_transform) {
                        c = c.with_transform(t);
                    }
                    if !labels.is_empty() {
                        c = c.with_breaks_labeled(labels);
                    }
                    apply_pinned_minors(c, panel.minor_breaks.as_deref(), transform)
                }
            }
        }
    };

    // The same flag the fixed path sets (see [`build_scale`]): freeing a
    // dimension narrows its domain, it does not undo `SETTING reverse => true`.
    Some(if is_reversed(global) {
        hs.with_direction(Direction::Reversed)
    } else {
        hs
    })
}

/// A panel domain's `(min, max)` from its two endpoint elements.
fn panel_range(input_range: &[crate::plot::ArrayElement]) -> Option<(f64, f64)> {
    let min = input_range.first()?.to_f64()?;
    let max = input_range.last()?.to_f64()?;
    Some((min, max))
}

/// A panel's visible breaks (suppressed labels dropped) as hephaestus
/// `(value, label)` pairs, wrapping positions in the transform's value variant.
fn visible_numbered_labels(
    breaks: &[(f64, Option<String>)],
    transform: Option<GTransform>,
) -> Vec<(HValue, String)> {
    breaks
        .iter()
        .filter_map(|(pos, label)| {
            label
                .clone()
                .map(|label| (temporal_value(transform, *pos), label))
        })
        .collect()
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
    /// Kept for completeness of the bin representation; the writer currently
    /// only joins on `centre` and reads `label`.
    #[allow(dead_code)]
    pub lower: f64,
    #[allow(dead_code)]
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
    use crate::writer::hephaestus::channels::{column_to_channel, ChannelData};
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
