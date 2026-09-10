//! Shared geom-wiring: position/material channels, scales, axes, legends, and
//! group keys. Each geom module declares its channel specs; these helpers do the
//! repetitive work, generic over the concrete geom builder.

use std::collections::HashSet;

use hephaestus::color::{rgb8, Color};
use hephaestus::plot::chrome::legend::{Legend, LegendKeySpec};
use hephaestus::plot::geom::{BuildableGeom, Geom, GeomBuilder, Raw};
use hephaestus::plot::theme::{Element, Length, RectElement, Theme};
use hephaestus::plot::Plot as HPlot;
use hephaestus::scales::chrome::LegendSide;
use hephaestus::scales::value::{DataColumn, Value as HValue};

use super::channels::{
    aesthetic_column_name, build_group_keys, column_to_bool, column_to_channel, column_to_colors,
    column_to_f64, column_to_strings, ChannelData,
};
use super::scales::{map_linetype, parse_color, parse_font_weight, RangeKind};
use crate::plot::{ParameterValue, ScaleTypeKind};
use crate::{AestheticValue, DataFrame, GgsqlError, Layer, Plot, Result};

/// The chrome ggsql renders with: hephaestus's default theme with the handful of
/// deviations ggsql needs. This is the single hook for chrome — ggsql has no
/// theme concept of its own yet, so anything the two writers must agree on that
/// isn't a scale or a channel belongs here.
///
/// Three deviations so far:
///
/// - **A colorbar's frame.** hephaestus's `BarTheme` leaves `linewidth_pt` unset,
///   which cascades to the 1pt ink border every `RectElement` gets by default, so
///   a continuous color legend arrives boxed. Vega-Lite draws no gradient border
///   (its `gradientStrokeWidth` default is 0), and the discrete `KeyTheme` next to
///   it zeroes its own border, so the box is out of place in either comparison.
/// - **Markdown chrome.** ggsql treats chrome strings as rich text, so `**bold**`
///   and `{.red word}` render as styled text rather than literal markers. Set on
///   the root `text` element, it cascades to *every* text slot — which is how a
///   future ggsql theme concept would drive it. What actually parses is whichever
///   of those slots hephaestus consults `markdown` on: today the plot title,
///   subtitle and caption, the axis titles and the facet strip labels. Legend
///   titles and break labels cascade the flag too but shape through
///   `TextRun::new` regardless, so they still draw their markers; they start
///   parsing with no change here once hephaestus reads the flag at those sites
///   (see [Known gaps](CLAUDE.md)).
pub fn ggsql_theme() -> Theme {
    let mut theme = Theme::default();
    theme.legend.bar.frame = Element::Set(RectElement {
        // The bar's own gradient fills the interior; only the border changes.
        fill: None,
        linewidth_pt: Some(Length::Abs(0.0)),
        ..RectElement::default()
    });
    theme.text.markdown = Some(true);
    theme
}

/// Read-only context for building one layer's geom.
pub struct Ctx<'a> {
    pub spec: &'a Plot,
    pub layer: &'a Layer,
    pub df: &'a DataFrame,
    /// Whether the layer is in transposed (horizontal) orientation.
    pub transposed: bool,
    /// Scale name this panel binds `pos1` (x) to: the shared `"pos1"` when fixed,
    /// a per-panel name when the facet dimension is free.
    pub pos1_scale: &'a str,
    /// Scale name this panel binds `pos2` (y) to.
    pub pos2_scale: &'a str,
    /// Sink collecting the legends a geom would draw. Legends are registered once
    /// on the composition (never on the per-panel plot), so faceted plots get a
    /// single shared legend rather than one per panel. `Some` only while building
    /// the first panel — every panel produces the same legends (all built from the
    /// globally resolved scales), so one capture suffices.
    pub legends: Option<&'a std::cell::RefCell<Vec<Legend>>>,
}

impl Ctx<'_> {
    /// The scale name to bind a position channel on `axis` to (panel-aware for
    /// free facet scales).
    pub fn pos_scale(&self, axis: PanelAxis) -> &str {
        match axis {
            PanelAxis::X => self.pos1_scale,
            PanelAxis::Y => self.pos2_scale,
        }
    }

    /// Record a legend for later registration on the composition. A no-op once
    /// the first panel has been captured (`legends` is `None`).
    pub fn push_legend(&self, legend: Legend) {
        if let Some(sink) = self.legends {
            sink.borrow_mut().push(legend);
        }
    }
}

/// Which panel axis a position channel drives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PanelAxis {
    X,
    Y,
}

/// A position channel: hephaestus `channel` ← ggsql `aesthetic`, on `axis`.
pub struct PositionSpec {
    pub channel: &'static str,
    pub aesthetic: String,
    pub axis: PanelAxis,
}

impl PositionSpec {
    pub fn new(channel: &'static str, aesthetic: impl Into<String>, axis: PanelAxis) -> Self {
        Self {
            channel,
            aesthetic: aesthetic.into(),
            axis,
        }
    }
}

/// A material aesthetic: ggsql `aesthetic` → hephaestus `channel`, producing
/// `kind`, with a fallback `default` applied when the aesthetic isn't mapped.
pub struct MaterialSpec {
    pub aesthetic: &'static str,
    pub channel: &'static str,
    pub kind: RangeKind,
    pub default: MatDefault,
}

impl MaterialSpec {
    pub fn new(
        aesthetic: &'static str,
        channel: &'static str,
        kind: RangeKind,
        default: MatDefault,
    ) -> Self {
        Self {
            aesthetic,
            channel,
            kind,
            default,
        }
    }
}

/// Fallback for an unmapped material channel, so output matches ggsql defaults.
pub enum MatDefault {
    None,
    Color(Color),
    Number(f64),
}

/// The legend swatch a geom's data-mapped scales use, so the key matches the
/// mark (a colored line for line geoms, a filled rect for bars/areas, etc.).
#[derive(Clone, Copy)]
pub enum LegendKind {
    Point,
    Line,
    Rect,
    /// A glyph, for a text layer: a scaled `fontsize` says what it does by
    /// drawing letters at each size rather than discs.
    Text,
}

/// What a geom needs wired: its position channels, material table, any raw
/// (unscaled) string channels (e.g. text labels), and whether it groups rows.
pub struct GeomSpec {
    pub positions: Vec<PositionSpec>,
    pub material: Vec<MaterialSpec>,
    /// Unscaled string channels set from a mapped aesthetic (e.g. text labels):
    /// (hephaestus channel, ggsql aesthetic).
    pub raw_strings: &'static [(&'static str, &'static str)],
    /// Constant panel-space channel values, scale-bypassing (e.g. a rule's
    /// 0..1 span, discrete-tile band edges): (hephaestus channel, value).
    ///
    /// Materialised per row, not set as a scalar: a hephaestus geom whose
    /// geometry varies per row (`SegmentGeom`, `RectGeom`, …) requires *every*
    /// position channel to be a column, and panics on a constant.
    pub raw_numbers: Vec<(&'static str, f64)>,
    /// Per-row unscaled channel data the geom computes itself (e.g. bar band
    /// edges from width/dodge): (hephaestus channel, one value per row).
    pub data_channels: Vec<(&'static str, Vec<f64>)>,
    /// Legend swatch style for this geom's data-mapped scales.
    pub legend_key: LegendKind,
    pub grouped: bool,
}

/// Build a concrete geom from its spec and attach it to the plot. Bindings are
/// written onto `plot`; legends are recorded on `ctx` for one-shot registration
/// on the composition; scales are registered globally from `spec.scales` (see
/// `PngWriter::write`), and axes are created per coordinate system in
/// `projection`.
pub fn build_and_add<G>(plot: &mut HPlot, spec: GeomSpec, ctx: &Ctx) -> Result<()>
where
    G: BuildableGeom + Geom + 'static,
{
    let mut builder = GeomBuilder::<G>::new();
    if spec.grouped {
        if let Some(keys) = build_group_keys(ctx.df, &ctx.layer.partition_by)? {
            builder.keys(keys);
        }
    }
    // Band channels the geom computes itself (bar/tile edges) already carry the
    // position adjustment, so `wire_positions` must not overwrite them.
    let claimed: Vec<&str> = spec.data_channels.iter().map(|(c, _)| *c).collect();
    wire_positions(&mut builder, &spec.positions, plot, ctx, &claimed)?;
    for (channel, aesthetic) in spec.raw_strings {
        if let Some(col) = aesthetic_column_name(ctx.layer, aesthetic) {
            builder.set(*channel, Raw(column_to_strings(ctx.df, col)?));
        }
    }
    for (channel, value) in &spec.raw_numbers {
        builder.set(*channel, Raw(vec![*value; ctx.df.height()]));
    }
    for (channel, values) in spec.data_channels {
        builder.set(channel, values);
    }
    wire_material(&mut builder, &spec.material, plot, ctx, spec.legend_key)?;
    plot.add_geom(builder.build());
    Ok(())
}

/// Set position channels on the builder and bind them to the `pos1`/`pos2`
/// scales. `set_binding` is idempotent, so repeated bindings across layers are
/// harmless. Axis chrome is created later, per coordinate system, in `projection`.
///
/// Each position also picks up the layer's position adjustment: `dodge` and
/// `jitter` are resolved by ggsql into per-row band fractions on the adjusted
/// axis, which map onto the geom's matching `_band` channel. Channels listed in
/// `claimed` are skipped — a geom that derives its own band edges (bar, tile) has
/// already folded the same offsets in.
fn wire_positions<G: BuildableGeom>(
    builder: &mut GeomBuilder<G>,
    positions: &[PositionSpec],
    plot: &mut HPlot,
    ctx: &Ctx,
    claimed: &[&str],
) -> Result<()> {
    let offsets = AxisOffsets::new(ctx.df);
    for p in positions {
        let data = match aesthetic_column_name(ctx.layer, &p.aesthetic) {
            Some(col) => column_to_channel(ctx.df, col)?,
            // A position given as a bare constant. It is repeated per row rather
            // than set as a scalar, because a geom whose geometry varies per row
            // rejects a constant position channel.
            None => constant_position(ctx, &p.aesthetic)
                .ok_or_else(|| missing_aesthetic(ctx, &p.aesthetic))?,
        };
        data.apply(builder, p.channel);
        plot.set_binding(p.channel, ctx.pos_scale(p.axis));

        if let Some(values) = offsets.for_axis(p.axis) {
            let band = format!("{}_band", p.channel);
            if !claimed.contains(&band.as_str()) {
                builder.set(band, values.clone());
            }
        }
    }
    Ok(())
}

/// A position aesthetic mapped to a bare `Literal`, materialised into one
/// data-space value per row so it still travels through its position scale.
///
/// ggsql delivers an aesthetic three ways and the writer honours all three
/// everywhere (see `wire_material`); positions are no exception. A number is a
/// continuous coordinate, a string a discrete category, and a boolean is read
/// as its category name, matching how the same values arrive in a column.
fn constant_position(ctx: &Ctx, aesthetic: &str) -> Option<ChannelData> {
    let n = ctx.df.height();
    match ctx.layer.mappings.get(aesthetic) {
        Some(AestheticValue::Literal(ParameterValue::Number(value))) => {
            Some(ChannelData::Floats(vec![*value; n]))
        }
        Some(AestheticValue::Literal(ParameterValue::String(value))) => {
            Some(ChannelData::Strings(vec![value.clone(); n]))
        }
        Some(AestheticValue::Literal(ParameterValue::Boolean(value))) => {
            Some(ChannelData::Strings(vec![value.to_string(); n]))
        }
        _ => None,
    }
}

/// The per-row band-fraction offsets ggsql resolved for a position adjustment,
/// per panel axis. `None` for an axis the layer wasn't adjusted along — which is
/// every axis for `position => 'identity'`, and the value axis always (`stack`
/// rewrites the value columns instead of offsetting).
struct AxisOffsets {
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
}

impl AxisOffsets {
    fn new(df: &DataFrame) -> Self {
        Self {
            x: offset_column(df, "pos1offset"),
            y: offset_column(df, "pos2offset"),
        }
    }

    fn for_axis(&self, axis: PanelAxis) -> Option<&Vec<f64>> {
        match axis {
            PanelAxis::X => self.x.as_ref(),
            PanelAxis::Y => self.y.as_ref(),
        }
    }
}

/// Set material channels: data-mapped → bind channel to its (globally
/// registered) scale + record a legend; literal → constant visual value; identity/
/// annotation → `Raw` per-row values; unmapped → the spec's default. Public so
/// custom-builder geoms (e.g. `spatial`) can wire their scalar aesthetics through
/// the same data-mapped-capable path the generic geoms use.
pub fn wire_material<G: BuildableGeom>(
    builder: &mut GeomBuilder<G>,
    material: &[MaterialSpec],
    plot: &mut HPlot,
    ctx: &Ctx,
    legend_kind: LegendKind,
) -> Result<()> {
    let mut handled: HashSet<&str> = HashSet::new();
    // One legend per *aesthetic*, not per channel. A geom may drive several
    // channels from one aesthetic — a ribbon sends `stroke` to both of its edge
    // curves — and each is a separate `MaterialSpec`, but they all describe the
    // same scale and want one swatch between them. Recording a second legend
    // does not merely duplicate: its key is `scaled` on the mirror channel
    // (`stroke2`), which no legend key kind consumes, so it resolves to neither
    // fill nor stroke and hephaestus paints its "row isn't empty" placeholder —
    // a black outline over the real key.
    let mut legended: HashSet<&str> = HashSet::new();

    for m in material {
        if handled.contains(m.channel) {
            continue;
        }
        // ggsql delivers material values three ways (mirroring the Vega-Lite
        // writer's `build_encoding_channel`): a bare `Literal` (a fixed value,
        // from a geom default or `SETTING`), a data-mapped `Column` (scaled +
        // legend), or an identity `AnnotationColumn` (per-row constant, from
        // PLACE).
        if let Some(AestheticValue::Literal(lit)) = ctx.layer.mappings.get(m.aesthetic) {
            if set_literal_channel(builder, m.channel, m.kind, lit) {
                handled.insert(m.channel);
            }
            continue;
        }
        let Some(col) = aesthetic_column_name(ctx.layer, m.aesthetic) else {
            continue;
        };
        handled.insert(m.channel);

        let scale = ctx.spec.find_scale(m.aesthetic);
        let type_kind = scale
            .and_then(|s| s.scale_type.as_ref())
            .map(|st| st.scale_type_kind());
        let data_mapped = scale.is_some() && type_kind != Some(ScaleTypeKind::Identity);

        if data_mapped {
            let data = column_to_channel(ctx.df, col)?;
            data.apply(builder, m.channel);
            // Bind the channel to the aesthetic's scale (registered globally) and
            // record a legend the first time this aesthetic is seen. hephaestus
            // collapses compatible legends, so repeated records across *layers*
            // for the same scale still merge at registration.
            plot.set_binding(m.channel, m.aesthetic);
            if legended.insert(m.aesthetic) {
                ctx.push_legend(material_legend(
                    ctx,
                    m.aesthetic,
                    m.channel,
                    m.kind,
                    legend_kind,
                    material,
                ));
            }
        } else {
            match m.kind {
                RangeKind::Color => {
                    builder.set(m.channel, Raw(column_to_colors(ctx.df, col)?));
                }
                RangeKind::Shape | RangeKind::Text => {
                    builder.set(m.channel, Raw(column_to_strings(ctx.df, col)?));
                }
                // A linetype column holds ggsql names or hex patterns, which the
                // channel cannot read as strings — it takes dash patterns. Map each
                // row through the same parser a literal goes through. Built as a
                // `DataColumn` because hephaestus has no `Raw(Vec<Arc<[LinetypeStep]>>)`
                // conversion, only the `Raw(DataColumn)` one.
                RangeKind::Linetype => {
                    let patterns: Vec<_> = column_to_strings(ctx.df, col)?
                        .iter()
                        .map(|s| map_linetype(s))
                        .collect();
                    builder.set(m.channel, Raw(DataColumn::from(patterns)));
                }
                RangeKind::Bool => {
                    builder.set(m.channel, Raw(column_to_bool(ctx.df, col)?));
                }
                // A weight column may hold keywords or numbers; both parse.
                RangeKind::FontWeight => {
                    let weights: Vec<f64> = column_to_strings(ctx.df, col)?
                        .iter()
                        .map(|s| parse_font_weight(s))
                        .collect();
                    builder.set(m.channel, Raw(weights));
                }
                RangeKind::Angle => {
                    let radians: Vec<f64> = column_to_f64(ctx.df, col)?
                        .into_iter()
                        .map(f64::to_radians)
                        .collect();
                    builder.set(m.channel, Raw(radians));
                }
                _ => {
                    builder.set(m.channel, Raw(column_to_f64(ctx.df, col)?));
                }
            }
        }
    }

    // Defaults for channels no spec mapped. `Raw` for the same reason literals
    // are: a default is a visual value, and a sibling layer may have bound this
    // channel to a scale that would otherwise swallow it.
    for m in material {
        if handled.contains(m.channel) {
            continue;
        }
        match m.default {
            MatDefault::Color(c) => {
                builder.set(m.channel, Raw(c));
                handled.insert(m.channel);
            }
            MatDefault::Number(n) => {
                builder.set(m.channel, Raw(n));
                handled.insert(m.channel);
            }
            MatDefault::None => {}
        }
    }
    Ok(())
}

/// Set a material channel to a constant from a `Literal` aesthetic value,
/// converting by the channel's `RangeKind` (mirrors the Vega-Lite writer's
/// `build_literal_encoding`). Hephaestus takes sizes/widths in the same units
/// ggsql resolves them to (points), so numbers pass through unscaled. Returns
/// whether the value was applicable (an unparseable color / type mismatch is
/// left to the geom's default).
///
/// The constant is set **`Raw`**, i.e. scale-bypassing. A literal is already a
/// visual-space value, and a hephaestus binding is per *plot channel*, not per
/// geom: one layer mapping `colour` binds `stroke` to a categorical scale for
/// every layer in the panel, and a sibling layer's plain (non-`Raw`) black
/// would then be looked up in that scale's domain, resolve to `Null`, and
/// vanish. Bypassing keeps each layer's constants its own.
fn set_literal_channel<G: BuildableGeom>(
    builder: &mut GeomBuilder<G>,
    channel: &str,
    kind: RangeKind,
    lit: &ParameterValue,
) -> bool {
    match (kind, lit) {
        (RangeKind::Color, ParameterValue::String(s)) => match parse_color(s) {
            Some(c) => {
                builder.set(channel, Raw(c));
                true
            }
            None => false,
        },
        (RangeKind::Shape, ParameterValue::String(s)) => {
            builder.set(channel, Raw(s.clone()));
            true
        }
        // An empty string is not "use the default": it is a lookup (a font family,
        // say) that misses. Leave the channel unset so hephaestus's default holds.
        (RangeKind::Text, ParameterValue::String(s)) if !s.is_empty() => {
            builder.set(channel, Raw(s.clone()));
            true
        }
        (RangeKind::Bool, ParameterValue::Boolean(b)) => {
            builder.set(channel, Raw(*b));
            true
        }
        // ggsql's `fontweight` takes a keyword or a number; hephaestus takes 100–900.
        (RangeKind::FontWeight, ParameterValue::String(s)) => {
            builder.set(channel, Raw(parse_font_weight(s)));
            true
        }
        (RangeKind::FontWeight, ParameterValue::Number(n)) if n.is_finite() => {
            builder.set(channel, Raw(*n));
            true
        }
        (RangeKind::Angle, ParameterValue::Number(n)) if n.is_finite() => {
            builder.set(channel, Raw(n.to_radians()));
            true
        }
        (RangeKind::Linetype, ParameterValue::String(s)) => {
            builder.set(channel, Raw(HValue::Linetype(map_linetype(s))));
            true
        }
        (RangeKind::Number, ParameterValue::Number(n)) if n.is_finite() => {
            builder.set(channel, Raw(*n));
            true
        }
        _ => false,
    }
}

/// The axis roles of a banded geom (boxplot / violin / range): its categories —
/// or, for a range, its fixed positions — sit on one axis and its values on the
/// other. ggsql flips the position columns of a transposed (horizontal) layer, so
/// the banded axis becomes `pos2` and the values land in the `pos1` family;
/// every channel name follows from that.
///
/// Hephaestus channels are named per panel axis (`x`, `y_band`, …), so a geom
/// asks this for the channel that drives its banded or value axis instead of
/// hardcoding `x`/`y`. The `pos1`/`pos2` *bindings* need no swap: a channel
/// always belongs to the same panel axis.
#[derive(Clone, Copy)]
pub struct BandAxes {
    transposed: bool,
}

impl BandAxes {
    pub fn new(ctx: &Ctx) -> Self {
        Self {
            transposed: ctx.transposed,
        }
    }

    /// The aesthetic family holding the banded-axis positions (`"pos1"`).
    pub fn band(&self) -> &'static str {
        if self.transposed {
            "pos2"
        } else {
            "pos1"
        }
    }

    /// The aesthetic family holding the values (`"pos2"`).
    pub fn value(&self) -> &'static str {
        if self.transposed {
            "pos1"
        } else {
            "pos2"
        }
    }

    /// The aesthetic carrying position-adjustment (dodge) offsets on the banded
    /// axis.
    pub fn dodge(&self) -> &'static str {
        if self.transposed {
            "pos2offset"
        } else {
            "pos1offset"
        }
    }

    /// The two banded-axis position channels (`("x", "x2")`).
    pub fn band_channels(&self) -> (&'static str, &'static str) {
        if self.transposed {
            ("y", "y2")
        } else {
            ("x", "x2")
        }
    }

    /// The two value-axis position channels (`("y", "y2")`).
    pub fn value_channels(&self) -> (&'static str, &'static str) {
        if self.transposed {
            ("x", "x2")
        } else {
            ("y", "y2")
        }
    }

    /// The banded axis's band-fraction offset channels (`("x_band", "x2_band")`),
    /// which shift a mark's two edges within the category band.
    pub fn band_fraction_channels(&self) -> (&'static str, &'static str) {
        if self.transposed {
            ("y_band", "y2_band")
        } else {
            ("x_band", "x2_band")
        }
    }

    /// The banded axis's absolute-pt offset channels (`("x_offset",
    /// "x2_offset")`), for marks sized in points rather than band fractions
    /// (hinge caps).
    pub fn band_offset_channels(&self) -> (&'static str, &'static str) {
        if self.transposed {
            ("y_offset", "y2_offset")
        } else {
            ("x_offset", "x2_offset")
        }
    }
}

/// The `side` SETTING as a signed direction along the banded axis: `None` for
/// `'both'` (a full-width mark centred on the band), else the sign of the half
/// the mark occupies.
///
/// Hephaestus band offsets are positive-right on x and positive-up on y, so
/// `'top'`/`'right'` are positive in either orientation — which reproduces the
/// Vega-Lite writer's visual outcome (there the sign flips with orientation
/// because Vega-Lite's y offsets point down).
pub fn side_sign(layer: &Layer) -> Option<f64> {
    match layer.parameters.get("side")? {
        ParameterValue::String(s) => match s.as_str() {
            "top" | "right" => Some(1.0),
            "bottom" | "left" => Some(-1.0),
            _ => None,
        },
        _ => None,
    }
}

/// The two edges of a banded mark, as offsets from the band centre: the full
/// `±half` band for `side => 'both'`, else the centreline → `±half` half-band.
/// Used for band fractions (box, median, violin edge) and for pt-sized marks
/// (hinge caps) alike.
pub fn band_edges(half: f64, side: Option<f64>) -> (f64, f64) {
    match side {
        None => (-half, half),
        Some(sign) => (0.0, sign * half),
    }
}

/// Half the band width a banded geom (bar/box/violin) occupies: the
/// dodge-narrowed width if set, else the `width` parameter (or `default`).
pub fn band_half_width(layer: &Layer, default: f64) -> f64 {
    let width = layer
        .parameters
        .get("width")
        .and_then(|v| match v {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(default);
    layer.adjusted_width.unwrap_or(width).abs() / 2.0
}

/// A constant number from an aesthetic, or `default` when unmapped. Reads an
/// annotation column first, then a bare `Literal` number (e.g. `slope => 1`).
pub fn constant_number(ctx: &Ctx, aesthetic: &str, default: f64) -> f64 {
    if let Some(n) = aesthetic_column_name(ctx.layer, aesthetic)
        .and_then(|c| column_to_f64(ctx.df, c).ok())
        .and_then(|v| v.first().copied())
        .filter(|x| x.is_finite())
    {
        return n;
    }
    if let Some(AestheticValue::Literal(ParameterValue::Number(n))) =
        ctx.layer.mappings.aesthetics.get(aesthetic)
    {
        if n.is_finite() {
            return *n;
        }
    }
    default
}

/// A position-adjustment offset column (per-row band fractions), or zeros when
/// the layer wasn't adjusted along that axis. For geoms that derive their own band
/// edges and so need the offsets as numbers; the generic path wires the same
/// column onto a `_band` channel in [`wire_positions`].
pub fn dodge_offsets(df: &DataFrame, aesthetic: &str) -> Vec<f64> {
    offset_column(df, aesthetic).unwrap_or_else(|| vec![0.0; df.height()])
}

/// ggsql's resolved offset column for one axis, when the layer carries one.
fn offset_column(df: &DataFrame, aesthetic: &str) -> Option<Vec<f64>> {
    let name = crate::naming::aesthetic_column(aesthetic);
    df.column(&name).ok()?;
    column_to_f64(df, &name).ok()
}

/// Resolve a label for an aesthetic: explicit `LABEL` wins (`None` suppresses),
/// else the original mapped column name is the default.
pub fn aesthetic_label(spec: &Plot, layer: &Layer, aesthetic: &str) -> Option<String> {
    if let Some(labels) = &spec.labels {
        if let Some(entry) = labels.labels.get(aesthetic) {
            return entry.clone();
        }
    }
    match layer.mappings.get(aesthetic) {
        Some(AestheticValue::Column {
            original_name: Some(name),
            ..
        }) => Some(name.clone()),
        _ => None,
    }
}

/// Resolve a plot-level label (`title`, `subtitle`, `caption`) from the `LABEL`
/// clause. `None` covers both "not set" and `LABEL <key> => NULL` (suppressed).
///
/// Literal `\n` in the SQL string literal becomes a real newline, matching the
/// Vega-Lite writer's `split_label_on_newlines`.
pub fn plot_label(spec: &Plot, key: &str) -> Option<String> {
    let text = spec.labels.as_ref()?.labels.get(key)?.as_ref()?;
    Some(text.replace("\\n", "\n"))
}

/// A resolved material aesthetic for a composite geom, mirroring the Vega-Lite
/// writer's shared-encoding model: either a data-mapped column (scaled through a
/// registered scale that is bound + legended once, carrying that scale's name) or
/// a constant visual value. Components select the rows they cover and apply it to
/// a channel, so one resolved aesthetic styles every part of the composite.
pub enum MaterialSource {
    Data { data: ChannelData, scale: String },
    Constant(HValue),
}

impl MaterialSource {
    /// Set `channel` for the `idx` rows: the scaled data subset, or the constant.
    pub fn apply<G: BuildableGeom>(
        &self,
        builder: &mut GeomBuilder<G>,
        channel: &str,
        idx: &[usize],
    ) {
        match self {
            MaterialSource::Data { data, .. } => data.select(idx).apply(builder, channel),
            // `Raw`: a resolved constant is a visual value and must not be looked
            // up in whatever scale another layer bound to this channel.
            MaterialSource::Constant(v) => {
                builder.set(channel, Raw(v.clone()));
            }
        }
    }

    /// The registered scale name, when data-mapped (for binding extra channels,
    /// e.g. a ribbon's far edge, to the same scale).
    pub fn scale_name(&self) -> Option<&str> {
        match self {
            MaterialSource::Data { scale, .. } => Some(scale),
            MaterialSource::Constant(_) => None,
        }
    }
}

/// The column backing an aesthetic a geom cannot draw without.
pub fn require_column<'a>(ctx: &'a Ctx, aesthetic: &str) -> Result<&'a str> {
    aesthetic_column_name(ctx.layer, aesthetic).ok_or_else(|| missing_aesthetic(ctx, aesthetic))
}

/// The error for a geom that reached the writer without an aesthetic it needs,
/// named in the user's own terms: `map_internal_to_user` turns `pos1` into `x`
/// (`y` for a transposed layer, `theta`/`radius` under polar), so the message
/// reads like the query that produced it. Core validation normally catches this
/// first; the writer's check is the backstop.
pub fn missing_aesthetic(ctx: &Ctx, aesthetic: &str) -> GgsqlError {
    let user = ctx
        .spec
        .get_aesthetic_context()
        .map_internal_to_user(aesthetic);
    GgsqlError::WriterError(format!(
        "{} layer has no '{user}' mapping",
        ctx.layer.geom.geom_type()
    ))
}

/// [`MaterialSource::apply`] for an aesthetic that may not have resolved.
/// `None` leaves the channel unset, so hephaestus's own default stands.
pub fn apply_material<G: BuildableGeom>(
    builder: &mut GeomBuilder<G>,
    source: Option<&MaterialSource>,
    channel: &str,
    idx: &[usize],
) {
    if let Some(source) = source {
        source.apply(builder, channel, idx);
    }
}

/// Resolve a color aesthetic (`fill`, `stroke`, …) for a composite geom, falling
/// back to `default` when unmapped. See [`resolve_material`].
pub fn resolve_color(
    ctx: &Ctx,
    plot: &mut HPlot,
    aesthetic: &'static str,
    channel: &'static str,
    default: Color,
    legend_kind: LegendKind,
    material: &[MaterialSpec],
) -> Result<MaterialSource> {
    Ok(resolve_material(
        ctx,
        plot,
        aesthetic,
        channel,
        RangeKind::Color,
        legend_kind,
        material,
    )?
    .unwrap_or(MaterialSource::Constant(HValue::Color(default))))
}

/// Resolve a material aesthetic for a composite geom, dispatching the same three
/// ways as [`wire_material`] does for simple geoms, but returning a value the
/// caller can apply to a row subset (which `wire_material`, being whole-column,
/// cannot).
///
/// A data-mapped non-identity scale binds `channel` to the aesthetic's (globally
/// registered) scale and records one legend; the full column is returned for
/// components to select. Otherwise the constant visual value: an identity /
/// annotation column's first value, else the mapped literal (`SETTING linewidth
/// => 3`). `None` when the aesthetic isn't mapped at all.
///
/// hephaestus collapses compatible legends, so repeated binds across a
/// composite's components merge at registration.
pub fn resolve_material(
    ctx: &Ctx,
    plot: &mut HPlot,
    aesthetic: &'static str,
    channel: &'static str,
    kind: RangeKind,
    legend_kind: LegendKind,
    material: &[MaterialSpec],
) -> Result<Option<MaterialSource>> {
    if is_data_mapped(ctx, aesthetic) {
        let col = aesthetic_column_name(ctx.layer, aesthetic);
        plot.set_binding(channel, aesthetic);
        ctx.push_legend(material_legend(
            ctx,
            aesthetic,
            channel,
            kind,
            legend_kind,
            material,
        ));
        return Ok(Some(MaterialSource::Data {
            data: column_to_channel(ctx.df, col.unwrap())?,
            scale: aesthetic.to_string(),
        }));
    }
    Ok(constant_material(ctx, aesthetic, kind).map(MaterialSource::Constant))
}

/// Whether an aesthetic maps a data column through a scale that actually
/// transforms it — i.e. it is scaled and legended, rather than carrying
/// visual-space values (an identity scale / annotation column) or a constant.
fn is_data_mapped(ctx: &Ctx, aesthetic: &str) -> bool {
    let scale = ctx.spec.find_scale(aesthetic);
    let type_kind = scale
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind());
    aesthetic_column_name(ctx.layer, aesthetic).is_some()
        && scale.is_some()
        && type_kind != Some(ScaleTypeKind::Identity)
}

/// The constant visual value of an unscaled material aesthetic: an identity /
/// annotation column's first value, else a bare `Literal`, converted by the
/// channel's `RangeKind` (hephaestus takes widths in points, as ggsql resolves
/// them, so numbers pass through). `None` when unmapped or inapplicable.
fn constant_material(ctx: &Ctx, aesthetic: &str, kind: RangeKind) -> Option<HValue> {
    let col = aesthetic_column_name(ctx.layer, aesthetic);
    let literal = match ctx.layer.mappings.aesthetics.get(aesthetic) {
        Some(AestheticValue::Literal(lit)) => Some(lit),
        _ => None,
    };
    match kind {
        RangeKind::Color => {
            if let Some(c) = col
                .and_then(|c| column_to_colors(ctx.df, c).ok())
                .and_then(|v| v.first().copied())
            {
                return Some(HValue::Color(c));
            }
            match literal {
                Some(ParameterValue::String(s)) => parse_color(s).map(HValue::Color),
                _ => None,
            }
        }
        RangeKind::Number | RangeKind::Position => {
            if let Some(n) = col
                .and_then(|c| column_to_f64(ctx.df, c).ok())
                .and_then(|v| v.first().copied())
                .filter(|x| x.is_finite())
            {
                return Some(HValue::Number(n));
            }
            match literal {
                Some(ParameterValue::Number(n)) if n.is_finite() => Some(HValue::Number(*n)),
                _ => None,
            }
        }
        RangeKind::Linetype => {
            let name = col
                .and_then(|c| column_to_strings(ctx.df, c).ok())
                .and_then(|v| v.first().cloned())
                .or_else(|| match literal {
                    Some(ParameterValue::String(s)) => Some(s.clone()),
                    _ => None,
                })?;
            Some(HValue::Linetype(map_linetype(&name)))
        }
        RangeKind::Shape | RangeKind::Text => {
            let name = col
                .and_then(|c| column_to_strings(ctx.df, c).ok())
                .and_then(|v| v.first().cloned())
                .or_else(|| match literal {
                    Some(ParameterValue::String(s)) => Some(s.clone()),
                    _ => None,
                })
                .filter(|s| !s.is_empty())?;
            Some(HValue::String(name.into()))
        }
        RangeKind::Bool => {
            if let Some(b) = col
                .and_then(|c| column_to_bool(ctx.df, c).ok())
                .and_then(|v| v.first().copied())
            {
                return Some(HValue::Bool(b));
            }
            match literal {
                Some(ParameterValue::Boolean(b)) => Some(HValue::Bool(*b)),
                _ => None,
            }
        }
        RangeKind::FontWeight => {
            let weight = col
                .and_then(|c| column_to_strings(ctx.df, c).ok())
                .and_then(|v| v.first().cloned())
                .or_else(|| match literal {
                    Some(ParameterValue::String(s)) => Some(s.clone()),
                    Some(ParameterValue::Number(n)) => Some(n.to_string()),
                    _ => None,
                })?;
            Some(HValue::Number(parse_font_weight(&weight)))
        }
        RangeKind::Angle => {
            let degrees = col
                .and_then(|c| column_to_f64(ctx.df, c).ok())
                .and_then(|v| v.first().copied())
                .or(match literal {
                    Some(ParameterValue::Number(n)) => Some(*n),
                    _ => None,
                })
                .filter(|d| d.is_finite())?;
            Some(HValue::Number(degrees.to_radians()))
        }
    }
}

/// Build a legend for a data-mapped material scale. Continuous color uses a
/// colorbar; everything else a keyed legend (swatch per `legend_kind`) at the
/// scale's breaks.
///
/// A binned scale flips whichever body it got into hephaestus's **binned** mode,
/// because ggsql's binned breaks are the bin *edges*: `N + 1` breaks describe `N`
/// bins. Binned mode draws one key (or one constant-color block) per bin and puts
/// the edge labels on a tick rail *between* them, so every edge is labelled once
/// at the boundary it names. That is why neither writer needs a compound
/// `"lower – upper"` label here — unlike the Vega-Lite writer, which has no
/// between-keys rail and so must reverse-engineer Vega's own range labels in
/// `encoding::build_symbol_legend_label_mapping`.
/// `scale_name` is the ggsql aesthetic, which is also the key the scale is
/// registered under — so the scale's type and the legend's title both follow
/// from it rather than being passed in.
pub fn material_legend(
    ctx: &Ctx,
    scale_name: &str,
    channel: &str,
    kind: RangeKind,
    legend_kind: LegendKind,
    material: &[MaterialSpec],
) -> Legend {
    let type_kind = ctx
        .spec
        .find_scale(scale_name)
        .and_then(|s| s.scale_type.as_ref())
        .map(|st| st.scale_type_kind());
    let title = aesthetic_label(ctx.spec, ctx.layer, scale_name);
    let continuous_color = kind == RangeKind::Color
        && matches!(
            type_kind,
            Some(ScaleTypeKind::Continuous) | Some(ScaleTypeKind::Binned)
        );
    let mut legend = if continuous_color {
        Legend::colorbar(scale_name).side(LegendSide::Right)
    } else {
        let key = match legend_kind {
            LegendKind::Point => LegendKeySpec::point(),
            LegendKind::Line => LegendKeySpec::line(),
            LegendKind::Rect => LegendKeySpec::rect(),
            LegendKind::Text => LegendKeySpec::text(),
        }
        .scaled(channel, scale_name);
        Legend::new(scale_name)
            .side(LegendSide::Right)
            .key(pin_constants(
                ctx,
                key,
                material,
                channel,
                legend_kind,
                kind,
            ))
    };
    if type_kind == Some(ScaleTypeKind::Binned) {
        legend = legend.binned();
    }
    if let Some(title) = title {
        legend = legend.title(title);
    }
    legend
}

/// Dress a legend key in everything the layer holds constant, so the swatch
/// looks like the marks it describes: a translucent area's key is translucent, a
/// map layer's key carries its border color, a dashed line's key is dashed.
///
/// A key paints only what it is told to paint — nothing is inherited from the
/// plot — so every constant has to be pinned explicitly. The geom's own
/// `MaterialSpec` table is the source: it already names each ggsql aesthetic's
/// hephaestus channel *and* that geom's aliasing (`color` → `fill` for an area,
/// → `stroke` for a line), so the key is styled exactly like the geom is.
/// `LegendKeySpec::fixed` ignores channels the key kind doesn't consume, which
/// is what lets one table serve point, line and rect keys.
///
/// Two channels are deliberately left alone: the one the legend is *scaled* on
/// (pinning it would override the very thing being shown), and any channel a
/// scale owns — a data-mapped aesthetic's column holds domain values, not visual
/// ones, and it carries its own legend anyway. Everything else pins, including
/// the channels that decide how much room the glyph takes (`size`, `linewidth`,
/// `shape`): hephaestus sizes each swatch cell from the key it holds, so a
/// `SETTING size => 12` marker gets a cell that fits it.
fn pin_constants(
    ctx: &Ctx,
    mut key: LegendKeySpec,
    material: &[MaterialSpec],
    scaled_channel: &str,
    legend_kind: LegendKind,
    kind: RangeKind,
) -> LegendKeySpec {
    // `claimed` is "do not pin this channel again"; `pinned` is "this channel
    // actually got a value". They differ for a channel a *scale* owns: nothing
    // may pin over it, but it has no constant either.
    let mut claimed: HashSet<&str> = HashSet::from([scaled_channel]);
    let mut pinned: HashSet<&str> = HashSet::from([scaled_channel]);
    for m in material {
        if claimed.contains(m.channel) {
            continue;
        }
        // A channel another scale drives is spoken for, whichever aesthetic
        // reached it first — claim it so a later alias can't pin over it.
        if is_data_mapped(ctx, m.aesthetic) {
            claimed.insert(m.channel);
            continue;
        }
        let value = constant_material(ctx, m.aesthetic, m.kind).or(match m.default {
            MatDefault::Color(c) => Some(HValue::Color(c)),
            MatDefault::Number(n) => Some(HValue::Number(n)),
            MatDefault::None => None,
        });
        if let Some(value) = value {
            claimed.insert(m.channel);
            pinned.insert(m.channel);
            key = key.fixed(m.channel, value);
        }
    }
    // Last resort: a key whose body color is neither scaled nor constant renders
    // as an empty swatch next to its label. That happens when the geom leaves
    // the body unmapped with no default, and — more often — when a *different*
    // scale owns it: a `size` legend on a layer that also maps `fill` cannot
    // borrow the fill column, since it holds domain values rather than colors.
    // A neutral grey is the honest stand-in; that scale carries its own legend.
    //
    // A color-scaled legend never needs it: the scale itself paints the key. It
    // must also not get it, because ggsql maps `color` onto *both* `fill` and
    // `stroke`, and hephaestus only collapses those two legends into one swatch
    // while their keys stay equivalent — a grey body on just the `stroke` one
    // splits them, leaving a second key drawn over the first.
    if kind != RangeKind::Color {
        let body = match legend_kind {
            LegendKind::Line => "stroke",
            LegendKind::Point | LegendKind::Rect | LegendKind::Text => "fill",
        };
        // A layer that fades its body out keeps the grey harmlessly: the
        // `fill_opacity` / `stroke_opacity` pinned above is what hephaestus
        // paints it at, so `opacity => 0` leaves the key as unfilled as the
        // marks are.
        if !pinned.contains(body) {
            key = key.fixed(body, HValue::Color(rgb8(64, 64, 64)));
        }
    }
    key
}
