//! `area`, `ribbon`, and `density` geoms → hephaestus `RibbonGeom` (a filled
//! band between two curves). Orientation-aware: aligned bands run along x with
//! the extent on y (`y`/`y2`); transposed bands run along y with the extent on
//! x (`x`/`x2`).

use std::collections::HashMap;

use hephaestus::color::rgb8;

use super::super::channels::{aesthetic_column_name, build_group_keys, column_to_f64};
use super::super::scales::RangeKind;
use super::super::wiring::{
    Ctx, GeomSpec, LegendKind, MatDefault, MaterialSpec, PanelAxis, PositionSpec,
};
use crate::plot::layer::geom::GeomType;

pub fn spec(ctx: &Ctx) -> GeomSpec {
    let ribbon = ctx.layer.geom.geom_type() == GeomType::Ribbon;

    let baseline = if ctx.transposed { "pos1end" } else { "pos2end" };

    let positions = if !ctx.transposed {
        // Band along x; extent on y. ribbon → [pos2min, pos2max]; area/density
        // → [pos2end (baseline), pos2].
        let (lo, hi) = if ribbon {
            ("pos2min", "pos2max")
        } else {
            ("pos2end", "pos2")
        };
        vec![
            PositionSpec::new("x", "pos1", PanelAxis::X),
            PositionSpec::new("y", lo, PanelAxis::Y),
            PositionSpec::new("y2", hi, PanelAxis::Y),
        ]
    } else {
        // Band along y; extent on x.
        let (lo, hi) = if ribbon {
            ("pos1min", "pos1max")
        } else {
            ("pos1end", "pos1")
        };
        vec![
            PositionSpec::new("y", "pos2", PanelAxis::Y),
            PositionSpec::new("x", lo, PanelAxis::X),
            PositionSpec::new("x2", hi, PanelAxis::X),
        ]
    };

    GeomSpec {
        positions,
        material: vec![
            MaterialSpec::new(
                "fill",
                "fill",
                RangeKind::Color,
                MatDefault::Color(rgb8(0, 0, 0)),
            ),
            // A ribbon's two edge curves are stroked independently: `stroke`
            // outlines curve A (the baseline / lower edge), `stroke2` curve B
            // (the data curve). Wiring only the first leaves the band's visible
            // silhouette unbordered, so every outline aesthetic is sent to both.
            // Whether curve A's outline is *visible* is then decided per mark by
            // `baseline_outline` — see there for why an area's is usually not.
            MaterialSpec::new("stroke", "stroke", RangeKind::Color, MatDefault::None),
            MaterialSpec::new("stroke", "stroke2", RangeKind::Color, MatDefault::None),
            MaterialSpec::new(
                "opacity",
                "fill_opacity",
                RangeKind::Number,
                MatDefault::Number(0.8),
            ),
            MaterialSpec::new(
                "linewidth",
                "linewidth",
                RangeKind::Number,
                MatDefault::None,
            ),
            MaterialSpec::new(
                "linewidth",
                "linewidth2",
                RangeKind::Number,
                MatDefault::None,
            ),
            MaterialSpec::new(
                "linetype",
                "linetype",
                RangeKind::Linetype,
                MatDefault::None,
            ),
            MaterialSpec::new(
                "linetype",
                "linetype2",
                RangeKind::Linetype,
                MatDefault::None,
            ),
        ],
        raw_strings: &[],
        constant_strings: Vec::new(),
        raw_numbers: vec![],
        // A ribbon's two edges are both data, so both are outlined unconditionally.
        data_channels: if ribbon {
            vec![]
        } else {
            vec![("stroke_opacity", baseline_outline(ctx, baseline))]
        },
        legend_key: LegendKind::Rect,
        grouped: true,
    }
}

/// Whether each row's mark takes an outline on curve A — the baseline of an
/// `area` or `density` band — as a per-row `stroke_opacity` for that curve.
///
/// A baseline that holds one value is the axis, not part of the shape: stroking
/// it draws a rule along `y = 0` under the chart, which is why ggplot2's
/// `geom_area`/`geom_density` outline only their upper edge. A baseline that
/// *wanders* is genuine silhouette — the bottom band of a centred stack
/// (streamgraph) rides on `-total/2` and wants its own border just as much as
/// its upper edge does. So the test is per mark on the resolved data rather than
/// per geom: within a normal stack the bottom band sits on the axis (no outline)
/// while the bands above it ride on their neighbour's upper edge (outlined, and
/// coincident with the outline that neighbour already draws).
///
/// hephaestus resolves a ribbon's outline channels once per mark, from the
/// mark's first row, so a per-row 0/1 here switches whole marks. Marks are the
/// `keys` [`build_and_add`](super::super::wiring::build_and_add) derives from
/// `partition_by`, so the grouping below has to match it. Zeroing the opacity is
/// what expresses "no outline" per mark: curve A is stroked whenever its channel
/// is bound at all, and the binding belongs to the geom as a whole.
fn baseline_outline(ctx: &Ctx, baseline: &str) -> Vec<f64> {
    let n = ctx.df.height();
    // No column at all: the baseline is a bare constant, so it cannot wander.
    let Some(column) = aesthetic_column_name(ctx.layer, baseline) else {
        return vec![AXIS; n];
    };
    let (Ok(values), Ok(keys)) = (
        column_to_f64(ctx.df, column),
        build_group_keys(ctx.df, &ctx.layer.partition_by),
    ) else {
        return vec![AXIS; n];
    };
    silhouette_opacity(&values, keys.as_deref())
}

/// Opacity for a baseline that is part of the shape, and for one that is the
/// axis.
const SILHOUETTE: f64 = 1.0;
const AXIS: f64 = 0.0;

/// The rule itself: a mark's baseline is silhouette when its values are not all
/// the same. `keys` groups rows into marks (`None` = one mark).
fn silhouette_opacity(values: &[f64], keys: Option<&[String]>) -> Vec<f64> {
    let mut marks: HashMap<&str, Vec<usize>> = HashMap::new();
    for i in 0..values.len() {
        marks.entry(keys.map_or("", |k| &k[i])).or_default().push(i);
    }

    let mut opacity = vec![AXIS; values.len()];
    for rows in marks.values() {
        // Nulls arrive as NaN and are not drawn, so they say nothing about the
        // baseline's shape.
        let mut finite = rows.iter().map(|&i| values[i]).filter(|v| v.is_finite());
        let wanders = match finite.next() {
            Some(first) => finite.any(|v| v != first),
            None => false,
        };
        if wanders {
            for &i in rows {
                opacity[i] = SILHOUETTE;
            }
        }
    }
    opacity
}

#[cfg(test)]
mod tests {
    use super::{silhouette_opacity, AXIS, SILHOUETTE};

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flat_baseline_is_the_axis() {
        // A plain area: `pos2end` is 0 everywhere.
        let opacity = silhouette_opacity(&[0.0, 0.0, 0.0], None);
        assert_eq!(opacity, vec![AXIS; 3]);
    }

    #[test]
    fn wandering_baseline_is_silhouette() {
        // A centred stack's only band, riding on -total/2.
        let opacity = silhouette_opacity(&[-3.0, -4.5, -2.0], None);
        assert_eq!(opacity, vec![SILHOUETTE; 3]);
    }

    #[test]
    fn normal_stack_outlines_every_band_but_the_bottom_one() {
        // Group "a" sits on the axis; "b" rides on a's upper edge.
        let opacity = silhouette_opacity(
            &[0.0, 0.0, 0.0, 3.0, 5.0, 4.0],
            Some(&keys(&["a", "a", "a", "b", "b", "b"])),
        );
        assert_eq!(
            opacity,
            vec![AXIS, AXIS, AXIS, SILHOUETTE, SILHOUETTE, SILHOUETTE]
        );
    }

    #[test]
    fn centred_stack_outlines_every_band() {
        let opacity =
            silhouette_opacity(&[-4.0, -6.0, -1.0, 0.5], Some(&keys(&["a", "a", "b", "b"])));
        assert_eq!(opacity, vec![SILHOUETTE; 4]);
    }

    #[test]
    fn nulls_do_not_make_a_baseline_wander() {
        let opacity = silhouette_opacity(&[0.0, f64::NAN, 0.0], None);
        assert_eq!(opacity, vec![AXIS; 3]);
    }
}
