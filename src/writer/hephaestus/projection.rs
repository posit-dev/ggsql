//! Apply a ggsql `PROJECT` clause (coordinate system) to the hephaestus plot,
//! including the coord-appropriate axes. Cartesian (the default when there is no
//! `PROJECT`) gets bottom/left rails; polar gets angular + radial rings.

use hephaestus::plot::chrome::axis::{Axis, AxisPlacement, PolarRing};
use hephaestus::plot::projection::{CustomProjection, PolarProjection, Projection as HProj};
use hephaestus::plot::AspectMode;
use hephaestus::plot::Plot as HPlot;
use hephaestus::scales::chrome::AxisSide;

use super::channels::{wkt_to_lines, wkt_to_outline};
use super::facet::{Panel, PanelScales};
use super::wiring::aesthetic_label;
use crate::plot::projection::{coord::CoordKind, Projection};
use crate::plot::ParameterValue;
use crate::Plot;

/// Apply the plot's coordinate system to one panel. No `PROJECT` clause is
/// treated as Cartesian.
pub fn apply_projection(plot: HPlot, spec: &Plot, panel: &Panel, ps: &PanelScales) -> HPlot {
    match spec.project.as_ref().map(|p| p.coord.coord_kind()) {
        None | Some(CoordKind::Cartesian) => {
            apply_proj_cartesian(plot, spec.project.as_ref(), spec, panel, ps)
        }
        Some(CoordKind::Polar) => apply_proj_polar(plot, spec.project.as_ref().unwrap(), spec, ps),
        Some(CoordKind::Map) => apply_proj_map(plot, spec.project.as_ref().unwrap()),
    }
}

fn apply_proj_cartesian(
    mut plot: HPlot,
    proj: Option<&Projection>,
    spec: &Plot,
    panel: &Panel,
    ps: &PanelScales,
) -> HPlot {
    if let Some(proj) = proj {
        if let Some(ParameterValue::Boolean(false)) = proj.properties.get("clip") {
            plot = plot.clip(false);
        }
        // The two conventions are inverses: ggsql's `ratio` is ggplot2's
        // `coord_fixed` (vertical step : horizontal step), while hephaestus's
        // `aspect_ratio` is the screen space one x unit takes per y unit. ggsql's
        // parameter constraint keeps it > 0.
        if let Some(ParameterValue::Number(ratio)) = proj.properties.get("ratio") {
            plot = plot
                .aspect_ratio(1.0 / *ratio)
                .aspect_mode(AspectMode::Range);
        }
    }
    // Edge-only axes for fixed scales (ggplot2 look): x on the bottom-most panel
    // of each column, y on the left column. A free dimension has a per-panel
    // domain, so its axis is drawn on every panel.
    if panel.last_row || ps.free_x {
        add_cartesian_axis(&mut plot, spec, "pos1", &ps.pos1, AxisSide::Bottom);
    }
    if panel.first_col || ps.free_y {
        add_cartesian_axis(&mut plot, spec, "pos2", &ps.pos2, AxisSide::Left);
    }
    plot
}

/// Whether a position scale warrants an axis. A missing scale, or a synthetic
/// single-category dummy (`__ggsql_stat_dummy` — e.g. a bar with no x mapped, or
/// a pie's radius), gets none; drawing it would expose the internal placeholder.
/// Mirrors the Vega-Lite writer's `AxisInfo::suppress`.
fn has_real_axis(spec: &Plot, name: &str) -> bool {
    spec.find_scale(name).is_some_and(|s| !s.is_dummy())
}

/// Add one bottom/left rail bound to `scale_name`. Skipped for absent or dummy
/// scales. `aesthetic` is the ggsql position name (`pos1`/`pos2`); `scale_name`
/// is the registered scale the rail reads (they differ only for a free per-panel
/// scale). The rail carries no title — that belongs to the composition, see
/// [`composition_axis_titles`].
fn add_cartesian_axis(
    plot: &mut HPlot,
    spec: &Plot,
    aesthetic: &str,
    scale_name: &str,
    side: AxisSide,
) {
    if !has_real_axis(spec, aesthetic) {
        return;
    }
    plot.add_axis(Axis::rail(scale_name, AxisPlacement::Cartesian(side)));
}

/// The figure's axis titles, as `(side, text)` pairs for the **composition**.
///
/// Axis titles live in the outer chrome — one centred title per dimension for
/// the whole figure — rather than on each panel's rail: a faceted plot would
/// otherwise title every row and column, and a free dimension (whose axis is
/// drawn on every panel) would repeat the title inside the grid. That also
/// matches the plot-level labels, which sit on the composition for the same
/// reason. Only Cartesian coords carry them: polar rails are untitled and a map
/// has no rails at all.
pub fn composition_axis_titles(spec: &Plot) -> Vec<(AxisSide, String)> {
    match spec.project.as_ref().map(|p| p.coord.coord_kind()) {
        None | Some(CoordKind::Cartesian) => {}
        _ => return Vec::new(),
    }
    let Some(layer) = spec.layers.first() else {
        return Vec::new();
    };
    [(AxisSide::Bottom, "pos1"), (AxisSide::Left, "pos2")]
        .into_iter()
        .filter(|(_, aesthetic)| has_real_axis(spec, aesthetic))
        .filter_map(|(side, aesthetic)| {
            aesthetic_label(spec, layer, aesthetic).map(|title| (side, title))
        })
        .collect()
}

fn apply_proj_polar(mut plot: HPlot, proj: &Projection, spec: &Plot, ps: &PanelScales) -> HPlot {
    plot.clear_axes();
    if let Some(ParameterValue::Boolean(false)) = proj.properties.get("clip") {
        plot = plot.clip(false);
    }
    // Sweep angles in degrees, clockwise from 12 o'clock (ggplot2 / Vega-Lite
    // pie convention). `start` defaults to 0 (12 o'clock); `end` defaults to a
    // full turn past `start`, so setting only `start` rotates a full circle
    // rather than truncating it (matches the VL writer's `start + 360`).
    // A categorical angle makes a radar rather than a pie: ggsql resolves that
    // and records it as `properties["radar"]` (the Vega-Lite writer reads the
    // same flag). `PolarProjection::radar` differs from `full_circle` in two
    // ways — `Chord` edges, so a polyline bends at each category boundary
    // instead of arcing between them, and `theta_break_fracs` at the band
    // centres `(i + 0.5) / N`, which is exactly where `Scale::map` puts a
    // discrete scale's categories, so spokes, grid polygons and data line up.
    let categories = matches!(
        proj.properties.get("radar"),
        Some(ParameterValue::Boolean(true))
    )
    .then(|| spec.find_scale("pos2").and_then(|s| s.input_range.as_ref()))
    .flatten()
    .map(|range| range.len());
    let base = match categories {
        Some(n) => PolarProjection::radar(n),
        None => PolarProjection::full_circle(),
    };
    let num = |k| match proj.properties.get(k) {
        Some(ParameterValue::Number(n)) => Some(*n),
        _ => None,
    };
    let start_deg = num("start").unwrap_or(0.0);
    let end_deg = num("end").unwrap_or(start_deg + 360.0);
    let deg = |d: f64| base.theta_start() - d * std::f64::consts::PI / 180.0;
    let start = deg(start_deg);
    let end = deg(end_deg);
    let inner = num("inner").unwrap_or(0.0);
    // ggsql assigns pos1→radius, pos2→theta (as the Vega-Lite writer does), so a
    // value on `y` (pos2) drives the slice angle and `x` (pos1) the radius.
    plot = plot.projection(HProj::Polar(
        base.channels("y", "x")
            .theta_range(start, end)
            .inner_radius(inner),
    ));
    // Suppress an axis whose position scale is a synthetic dummy (e.g. a pie's
    // radius), same as the Cartesian path.
    if has_real_axis(spec, "pos2") {
        plot.add_axis(Axis::rail(
            ps.pos2.as_str(),
            AxisPlacement::PolarAngular(PolarRing::Outer),
        ));
    }
    if has_real_axis(spec, "pos1") {
        // The radial rail runs along the spoke at the *start* of the sweep.
        // `theta_frac` is a 0–1 fraction of the sweep, not an angle — the sweep's
        // own start is 0.0 whatever `theta_start` works out to be.
        plot.add_axis(Axis::rail(
            ps.pos1.as_str(),
            AxisPlacement::PolarRadius { theta_frac: 0.0 },
        ));
    }
    plot
}

/// Map projection. Coordinates arrive **pre-projected from SQL**, so hephaestus
/// performs no reprojection: a `Custom` projection uses the projected clip
/// boundary as its drawing surface (clip + background) and the projected
/// graticule lines as its grid. Position scales are the bbox-framed `pos1`/`pos2`
/// registered in `compose::build_composition`. Mirrors the Vega-Lite writer's
/// identity `MapProjection` (`panel_boundary` + `graticule_*` from `computed`).
fn apply_proj_map(mut plot: HPlot, proj: &Projection) -> HPlot {
    // A map has no Cartesian rails; the boundary + graticules are the chrome.
    plot.clear_axes();

    let computed_str = |key: &str| match proj.computed.get(key) {
        Some(ParameterValue::String(s)) => Some(s.as_str()),
        _ => None,
    };

    // The projected clip boundary becomes the Custom projection's outline (a full
    // MultiPolygon — every part with its holes); when absent (a map with no
    // CRS/clip), fall back to the default Cartesian identity over the bbox-framed
    // scales.
    let outline = computed_str("panel_boundary")
        .map(wkt_to_outline)
        .unwrap_or_default();
    if !outline.is_empty() {
        let mut custom = CustomProjection::new(outline);
        if let Some(lon) = computed_str("graticule_lon") {
            custom = custom.x_major(wkt_to_lines(lon));
        }
        if let Some(lat) = computed_str("graticule_lat") {
            custom = custom.y_major(wkt_to_lines(lat));
        }
        // `clip` applies to every coord kind, as it does in the Vega-Lite writer
        // (`apply_clip_to_layers`): the boundary is still the drawing surface
        // when clipping is off, marks outside it just aren't cut away.
        let clip = !matches!(
            proj.properties.get("clip"),
            Some(ParameterValue::Boolean(false))
        );
        plot = plot.projection(HProj::Custom(custom)).clip(clip);
    }
    plot
}
