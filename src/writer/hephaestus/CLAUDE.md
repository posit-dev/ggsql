# `writer/hephaestus/` — renderer-backed writer internals

The writers here render a resolved ggsql `Spec` through
[hephaestus](https://github.com/posit-dev/hephaestus), a 2D scene renderer with a
grammar-of-graphics plot API. Seven of them exist, each behind its own cargo
feature — the three GPU-free ones on by default, the four raster ones not:

| Writer | Feature | Default | GPU | Output | Its own options |
| --- | --- | --- | --- | --- | --- |
| `PngWriter` | `png` | — | yes | PNG bytes, lossless, alpha preserved | `compression` = `none`/`fast`/`balanced`/`small` |
| `JpegWriter` | `jpeg` | — | yes | JPEG bytes, lossy, **no alpha** | `quality` 1–100 |
| `TiffWriter` | `tiff` | — | yes | TIFF bytes, lossless, alpha preserved | `compression` = `none`/`deflate`/`lzw`/`packbits` |
| `WebpWriter` | `webp` | — | yes | WebP bytes, lossless VP8L, alpha preserved | — |
| `SvgWriter` | `svg` | ✓ | **no** | SVG text, resolution independent | `text` = `text`/`outline`, `embed-fonts`, `id-prefix` |
| `PdfWriter` | `pdf` | ✓ | **no** | one PDF page, fonts subset in | `compress`, `links` |
| `HepWriter` | `hep` | ✓ | **no** | a `.hep` plot document — no picture at all | `lossy`, `embed-fonts` |

**Each format exposes the axis it actually has, and they share no knob they
would have to reinterpret.** PNG's `compression` trades encode time for size,
TIFF's trades reader compatibility for size (all four of its compressors are
lossless, so `deflate` simply *is* the small one), JPEG's `quality` is a rate
knob, and VP8L has no rate control at all. A shared `compression` across the
four raster writers would mean four different things.

**The three GPU-free writers are the important ones architecturally.** They go
through the same `PlotComposition` and the same `render` call — the composition
takes `&mut dyn SceneBuilder`, so a vector scene slots in exactly where the
rasteriser does. That is why they need no adapter, pull in no wgpu and need no
`-dev` package to build, which is what lets them be **default features**. It is
also what makes them the test surface: see [Testing](#testing).

**hephaestus is not a public name.** The user-facing names are the formats
(`--writer png`, `--features webp`, `ggsql::writer::TiffWriter`); the module is
named after the renderer it wraps and is private. What leaves the crate is the
seven writers, `PlotViewer`, and this list ([`../mod.rs`](../mod.rs)):

| Re-export | Origin |
| --- | --- |
| `Canvas` | ggsql's own |
| `MAX_RASTER_DIMENSION` | ggsql's own |
| `Color`, `rgba` | **the renderer's** (`hephaestus::color`) |
| `PngCompression`, `TiffCompression` | **the renderer's** |

The bottom two rows are why a hephaestus bump is a breaking change to ggsql
even when the renderer's own API holds still: those types are in ggsql's public
API by re-export, so a rename or an added variant upstream lands here. Anything
new that has to go public should be ggsql's own type unless there is a reason
it cannot be. Keep the renderer's name out of anything a user reads — CLI help,
error messages, `/doc/`.

The one carve-out: **a foreign format may be called by its own name.**
`HepWriter` writes hephaestus's own `.hep` plot-document format, whose magic
bytes are `HEPHPLOT`. ggsql does not define that format, so naming it after its
owner is accurate rather than a leak — and "document" would have been the worse
name, implying a generic container ggsql defines. ggsql's *own* writers still
may not be named after the renderer.

This file is the **architecture**: the abstractions, the invariants, and how to
extend them. For how the writer's behaviour got here, read
[`/CHANGELOG.md`](../../../CHANGELOG.md) and the commit history; what is
deliberately not done yet is in [Known gaps](#known-gaps) below.

For ggsql language semantics see [`/doc/syntax/`](../../../doc/syntax/); for the
sibling writer's internals, [`../vegalite/CLAUDE.md`](../vegalite/CLAUDE.md).

## The governing principle

**ggsql owns every scale domain; the writer never computes its own extents.**
The `Spec` arrives with each `Scale` fully resolved — type, domain (already
expanded, transform-aware, trained globally over all layers and the whole
position family), transform, breaks, formatted labels, and a concrete output
range for material aesthetics. The writer's job is to *pass those through* to
hephaestus, which performs the value→pixel mapping at draw time. This mirrors
the Vega-Lite writer, which passes `input_range` into `scale.domain`.

The practical consequence: when something looks wrong, the fix is usually a
missing *pass-through*, not a better computation here. The same shape held
upstream — every hephaestus gap this writer hit was a missing setter, not a
missing algorithm.

There are exactly **two scoped exceptions**, both flagged in the code and both
debt that would disappear if ggsql resolved more:

| Exception | Where | Why |
| --- | --- | --- |
| Free facet dimensions | `scales::{free_position_scale, free_binned_scale}` | ggsql resolves one global domain; a `free` panel needs its own. Only the *extent* is computed — the padding around it is still ggsql's, via `Scale::expand_range`. |
| Spatial `pos1`/`pos2` | `compose.rs::map_bbox` | A spatial layer positions by geometry, so ggsql resolves no position scales. The bbox still comes from ggsql (`Projection.computed["bbox"]`), falling back to the geometry extent only for a bare `spatial` geom. |

## Configuration

Rendering needs concrete dimensions, so unlike the Vega-Lite writer these
carry state — and they carry the *same* state, in [`canvas.rs`](canvas.rs):
`Canvas { width, height, dpi, background, physical }`. Each writer's `new` +
`.background()` set it directly; `Writer::from_options` builds the same thing
from the frontend-agnostic key–value [`WriterOptions`](../options.rs)
(`-D width=1600` on the CLI), via `Canvas::from_options(options, extra)` where
`extra` is the writer's own keys.

`CANVAS_OPTIONS` leads the concatenation `reject_unknown` sees, so the shared
keys come first in the "supported options" list — the nearest miss for a
mistyped key is almost always one of them. `physical` records that `units`
resolved to a physical unit; only a vector backend consults it, to decide
whether to declare a print size.

The user-facing table of keys lives in each writer's rustdoc and in
[`/doc/get_started/tooling/cli.qmd`](../../../doc/get_started/tooling/cli.qmd);
what matters here:

- **`units` interprets supplied dimensions only.** `to_pixels` converts a
  physical unit through inches at `dpi`, so a figure given in inches grows with
  resolution. The defaults are pixel counts and so are unit-independent.
- **DPI is not just print resolution.** hephaestus converts the theme's physical
  sizes (text, strokes, spacing — all points) at render DPI, so `dpi` also sets
  how large the chrome is relative to a pixel canvas.
- **Every option is validated, none is ignored.** `reject_unknown` first, then a
  per-key error naming the option; `whole_pixels` rejects a dimension outside
  `1..=MAX_DIMENSION` so a slipped unit conversion fails with a message rather
  than by exhausting GPU memory.

## Render flow

Unlike the Vega-Lite writer, which emits a declarative document and lets the VL
runtime do layout and scale application, hephaestus **is** the runtime. So
`write` builds a live object graph and renders it.

```
compose::build_composition(&Plot, &HashMap<String, DataFrame>)
 │
 ├─ facet::build_panels(spec, data)      → (Composition, Vec<Panel>)
 │      1×1 grid + one Panel when unfaceted; else grid(nrow, ncol, cells)
 ├─ PlotComposition::new(&composition).shape_registry(..)
 ├─ wiring::plot_label → composition title / subtitle / caption
 ├─ projection::composition_axis_titles → one centred x / y title, outer chrome
 ├─ for scale in spec.scales:            scales::build_scale(scale, RangeKind)
 │      → view.insert_scale(scale.aesthetic, hs)      ← the fixed/shared scales
 ├─ map_bbox → insert continuous "pos1"/"pos2" for a map / spatial plot
 │
 ├─ for panel in panels:                              ← one hephaestus Plot each
 │    ├─ facet::panel_dataframe(layer_df, panel)      per-layer row slice
 │    ├─ facet::PanelScales::new(spec, panel)         free dims → "pos1__p{idx}"
 │    ├─ free dims: scales::free_position_scale → view.insert_scale
 │    ├─ for (layer, df) in slices:  geom::build_into_plot(&mut plot, &Ctx{..})
 │    │      geoms set channels, plot.set_binding(channel, scale), push legends
 │    ├─ projection::apply_projection(plot, spec, panel, &ps)   ← axes live here
 │    ├─ map: plot.aspect_ratio(1.0).aspect_mode(Range)   ← square units
 │    ├─ panel.strip_top / strip_right → plot.strip(AxisSide::…)
 │    └─ view.attach_plot(plot)
 │
 ├─ legend_sink (captured from the *first* panel) → view.add_legend(..)
 └─ view.validate()

raster::pixels(spec, data, canvas, renderer)
 ├─ compose::validate_plot                    ← rejects a zero-layer plot, and
 │                                               the `arrow` stub no writer draws
 ├─ compose::build_composition                ← the diagram above
 └─ raster::render_rgba8: HybridRenderer → straight-alpha RGBA8 buffer

<raster>::write_with  = raster::pixels, then one encoder call
<vector>::write_reporting = vector::draw into an SvgScene / PdfScene, then encode
HepWriter::write_reporting = write_composition — no scene, no pixels
```

A writer is therefore its option parsing plus one line. `PlotComposition` is
where the work is, and it is format-independent — which is why the module gate
is the internal `graphics` feature and only `raster` pulls in wgpu.

**Degradation is reported, not returned.** `SvgScene` and `PdfScene` collect
what the format could not express, and the `hep` writer can list what the
document cannot carry. `Writer::write` has nowhere to put that, and widening
the trait for three of eight writers would be wrong, so those three add an
inherent `write_reporting` / `render_reporting` returning `(output, Vec<String>)`
and `Writer::write` discards the second half. `Vec<String>` rather than a ggsql
enum: the renderer's variants are `#[non_exhaustive]`, so mirroring them means
re-deriving a growing list every release and re-exporting them leaks its type
names — each writer's `describe()` translates at the boundary instead, which is
also where the renderer's name gets scrubbed. The list should be empty for
everything ggsql draws, and the corpus tests assert exactly that.

Layers draw in `spec.layers` order, which is DRAW order, which is z-order.

## Module map

| File | Role |
| --- | --- |
| [`mod.rs`](mod.rs) | Module wiring and the public re-exports, plus the shared `renders_*` corpus driven through the PNG writer. No writer lives here. |
| [`canvas.rs`](canvas.rs) | `Canvas`, `CANVAS_OPTIONS`, unit conversion and the dimension bound — the configuration every writer shares. Plus the test-only `Canvased` / `assert_canvas_semantics`, so the shared option behaviour is asserted once per writer rather than restated per writer. |
| [`compose.rs`](compose.rs) | `validate_plot` and `build_composition` — the orchestration above, and `map_bbox` / `map_range`. Format-independent, and where nearly all the code is. |
| [`raster.rs`](raster.rs) | `RasterRenderer`, `render_rgba8`, and `pixels`. **The only file that names a GPU renderer**, and the only part needing an adapter. |
| [`vector.rs`](vector.rs) | `draw` — the same three steps as `raster::pixels`, into a `&mut dyn SceneBuilder` instead of a pixel buffer. No GPU. |
| [`png.rs`](png.rs), [`jpeg.rs`](jpeg.rs), [`tiff.rs`](tiff.rs), [`webp.rs`](webp.rs) | One raster writer each: its rustdoc option table, `from_options`, and one encoder call. |
| [`svg.rs`](svg.rs), [`pdf.rs`](pdf.rs) | One vector writer each, plus a `describe()` translating what the format could not express into ggsql's vocabulary. |
| [`hep.rs`](hep.rs) | The plot-document writer. Serialises the composition; builds no scene at all. |
| [`window.rs`](window.rs) | `PlotViewer` — **not a writer.** Shows the composition in a native window and blocks until it closes. |
| [`wiring.rs`](wiring.rs) | The shared, geom-generic machinery: `Ctx`, `GeomSpec` + its parts, `build_and_add`, `wire_positions`, `wire_material`, `MaterialSource`/`resolve_material`, `BandAxes`, `side`/band helpers, `material_legend`, label resolution. |
| [`scales.rs`](scales.rs) | ggsql `Scale` → hephaestus `Scale`. `RangeKind`, transform + palette + break mapping, temporal scales, free-panel scales, `binned_bins`/`bin_at_centre`. |
| [`channels.rs`](channels.rs) | DataFrame column → typed channel data (`ChannelData`, `column_to_*`), group keys, WKB/WKT geometry decoding. |
| [`facet.rs`](facet.rs) | `FACET` → `Composition` + `Vec<Panel>`; level ordering, strip labelling, per-panel row slicing, `PanelScales`. |
| [`projection.rs`](projection.rs) | `PROJECT` → hephaestus `Projection`, **and the axes** (they depend on the coord). |
| [`geom/`](geom/) | One module per geom family, each declaring a `GeomSpec` or supplying a custom builder. `geom/mod.rs` is the dispatch + `is_supported`. |

## Core abstractions

### `Ctx` — what a geom is given

Read-only per (layer, panel): the `Plot`, the `Layer`, that panel's sliced
`DataFrame`, `transposed`, the scale names to bind positions to
(`pos1_scale`/`pos2_scale` — panel-aware, so free facets work), and the legend
sink. Geoms write bindings directly onto the `HPlot` and push legends through
`Ctx::push_legend`; there is no accumulator to thread.

### `GeomSpec` — the declarative path

Most geoms are just data. A module returns a `GeomSpec` and
`wiring::build_and_add::<G>` does the rest (see [`geom/point.rs`](geom/point.rs)
for the minimal case):

| Field | Meaning |
| --- | --- |
| `positions: Vec<PositionSpec>` | hephaestus `channel` ← ggsql `aesthetic`, plus which `PanelAxis` it drives (so the right `pos` scale is bound and the right dodge/jitter offsets picked up). |
| `material: Vec<MaterialSpec>` | ggsql aesthetic → hephaestus channel, a `RangeKind`, and a `MatDefault` fallback matching ggsql's own geom default. Several aesthetics may target one channel (`fill`/`color`/`colour` → `fill`); the first that resolves wins. |
| `raw_strings` | Unscaled string channels from a mapped aesthetic (text labels). |
| `raw_numbers` | Constant panel-space values that bypass scales (a rule's 0..1 span), materialised one per row. |
| `data_channels` | Per-row values the geom computes itself (bar/tile band edges, an area's per-mark baseline-outline gate). Channels listed here are *claimed*: `wire_positions` won't overwrite them with the raw offsets, because the geom already folded those in. |
| `legend_key: LegendKind` | Point / Line / Rect / Text swatch, so a line legend shows a line and a text legend shows a glyph. |
| `grouped: bool` | Derive hephaestus `keys` from `layer.partition_by`, for multi-vertex marks (line, area, polygon). |

### The three ways ggsql delivers an aesthetic

This is the single most important thing to get right, and it mirrors the
Vega-Lite writer's `build_encoding_channel` exactly. `wire_material` (whole
column) and `resolve_material` (row-subsettable, for composites) both dispatch
the same three ways:

| `AestheticValue` | Meaning | Handling |
| --- | --- | --- |
| `Literal(..)` | A fixed value — **every geom default and every `SETTING` constant** arrives this way, not as a materialized column | `Raw` constant channel value (`set_literal_channel` / `constant_material`), converted by `RangeKind` |
| `Column` with a non-identity scale | Data-mapped | Set the column, `plot.set_binding(channel, aesthetic)`, record one legend |
| `Column` with an identity scale, or `AnnotationColumn` | Visual-space values already | Per-row `Raw` |

`MatDefault` is a true last-resort fallback, only for an aesthetic ggsql didn't
map at all. Keying off columns alone silently drops every literal — which is how
`SETTING color => 'red'` once rendered black.

**Only a ggsql-mapped column goes through a scale; everything the writer
resolves itself is `Raw`.** A hephaestus binding belongs to the *plot channel*,
not to the geom that set it, so one layer mapping `colour` binds `stroke` to a
categorical scale for **every** layer in the panel. A plain (non-`Raw`) constant
on that channel — a literal, a `MatDefault`, a composite's
`MaterialSource::Constant` — is then looked up in that scale's domain, resolves
to `Null`, and the mark silently disappears. `Raw` bypasses the binding, which
is what a value already in visual space wants anyway. The same holds for a
position given as a constant: `wire_positions` materialises it per row through
`constant_position` so it still travels through its position scale, because a
hephaestus geom whose geometry varies per row rejects a constant position
channel outright (`"x" must be data, not constant`).

### `MaterialSource` — composites

A composite geom (boxplot, violin) decomposes one ggsql layer into several
hephaestus geoms that must all be styled *identically*, each from its own row
subset. `wiring::resolve_material` resolves an aesthetic once — registering the
binding and one legend if data-mapped — and returns a `MaterialSource` that
components `.apply(&mut builder, channel, &row_indices)`. `resolve_color` /
`resolve_optional_color` are the color-typed wrappers (the latter for aesthetics
whose ggsql default is `Null`, e.g. a text geom's `stroke`, where "unmapped"
must leave the channel unset). This is the raster analog of Vega-Lite's *shared
encoding* on a composite mark.

### `BandAxes` — banded geoms and orientation

Boxplot, violin and range measure *across* the axis they sit on. `BandAxes`
names channels by **role** rather than by axis — `band()`, `value()`, `dodge()`,
`band_channels()`, `band_fraction_channels()`, `band_offset_channels()` — and
flips them when ggsql transposed the layer. Bindings never swap: a hephaestus
channel always drives the same panel axis; only the column feeding it moves.

`side_sign` + `band_edges(half, side)` halve a mark onto one side of the band
(`'both'` → `±half`, else centreline → `±half`). hephaestus band offsets are
positive-right on x and positive-up on y — the convention every ggsql offset
uses — so `'top'`/`'right'` are positive in *either* orientation and one
predicate covers both. The Vega-Lite writer's `side_is_positive` reads the same
way; it reverses the *scale domain* of a `yOffset` channel rather than the sign,
because VL's y offsets point down.

## Channel naming

hephaestus channels are named per **panel axis**; scales are registered under the
**ggsql aesthetic**. `plot.set_binding(channel, scale_name)` ties them together
and is idempotent, so repeated bindings across layers and components are
harmless.

**Each hephaestus geom declares the channels it accepts, and setting one it
doesn't declare panics** (`geom::state::validate_known_channels`, at build time
rather than at draw). So the names below are per-geom, not global: only the
fill-bearing geoms take `fill_opacity`, only `RibbonGeom` takes the `2`-suffixed
far-edge channels. A misnamed channel fails loudly rather than rendering as the
default — check the target geom's `CHANNELS` catalog upstream when adding one.

| Concept | hephaestus channel |
| --- | --- |
| Positions | `x`, `x2`, `y`, `y2` |
| Band fraction offsets (dodge/jitter, width) | `x_band`, `x2_band`, `y_band`, `y2_band` |
| Absolute (pt) offsets — hinge caps | `x_offset`, `x2_offset`, `y_offset`, `y2_offset` |
| Color | `fill`, `stroke` (`stroke2` = a ribbon's far edge; `text_stroke` = a glyph outline) |
| Scalars | `size`, `linewidth`, `linetype`, `shape`, `fill_opacity` / `stroke_opacity` |
| Geometry / text | `geometry`; `text`, `markdown`, `anchor_x`, `anchor_y`, `angle`, `weight`, `italic`, `family` |

| Scale registry key | Source |
| --- | --- |
| `pos1`, `pos2`, `fill`, `stroke`, `size`, `shape`, `linetype`, … | The ggsql aesthetic name, from `spec.scales` |
| `pos1__p{index}`, `pos2__p{index}` | A **free** facet dimension's per-panel scale |

Under Polar, ggsql assigns pos1→radius and pos2→theta (as the Vega-Lite writer
does); `projection.rs` tells hephaestus so via `PolarProjection`'s
`angle_channel`/`radius_channel` rather than renaming anything.

## Scales

`scales::build_scale(Option<&GScale>, RangeKind) -> Option<HScale>` is the whole
translation. It returns `None` when ggsql resolved no scale type — the writer
registers nothing rather than fabricating a scale.

- `ScaleTypeKind` maps 1:1 (Continuous / Discrete / Ordinal / Binned /
  Identity), as do the transforms; cast and temporal transforms map to identity
  because values arrive already projected to `f64`.
- A **temporal** continuous scale becomes `scale::temporal` in the unit its
  transform names (days / µs since epoch, ns since midnight — exactly
  `ArrayElement`'s units), so hephaestus's own ticks read as dates.
- `RangeKind` selects how a resolved `OutputRange::Array` becomes a hephaestus
  range: `range_colors` / `range_numbers` / `range_strings` / `range_linetypes`,
  and nothing for `Position`. Palettes are already concrete by the time the
  writer runs.
- **`RangeKind` is also the one place a value converts**, for the aesthetics no
  scale touches: `wire_material`, `set_literal_channel` and `constant_material`
  all dispatch on it, so a conversion written once serves the literal, the
  identity column, *and* the legend key. That is why the text geom's face is
  expressed as kinds (`Text` for a family, `Bool` for italic, `FontWeight` for a
  CSS keyword or number, `Angle` for ggsql's degrees → hephaestus's radians)
  rather than as per-row code in the geom: a channel converted inside a geom
  cannot be pinned onto a key. A new unit-converted aesthetic belongs here.
- **Breaks are ggsql's, majors and minors alike.** `apply_breaks` feeds
  `break_labels()` in as `with_breaks_labeled`, so ticks (and `RENAMING`
  overrides) match ggsql — and therefore the Vega-Lite writer — exactly.
  `apply_minor_breaks` does the same for `numeric_minor_breaks()`, where
  `Some(vec![])` ("resolved to none") must stay distinct from `None` ("not
  resolved, fall back to hephaestus's automatic minors"). A *suppressed* label
  means different things on either side of the categorical divide, so the two
  take different accessors: a categorical scale keeps the break and blanks it
  (`break_labels`, since `RENAMING <level> => null` must not shift the axis),
  a numeric one drops it whole (`visible_break_labels`, since a binned
  `oob => 'squish'` terminal is not a real boundary).
- **`reverse` is the writer's to apply**, like VL's `scale.reverse`: ggsql
  resolves the property but never touches the domain, so the writer sets
  hephaestus's `Direction::Reversed` on the scale. Reversal is a property of the
  *mapping*, so one flag covers every scale kind and both roles — a position axis
  runs backwards, a material scale walks its palette from the far end — while the
  domain, the breaks, the bin edges, and the order a legend lists its keys in all
  stay as ggsql resolved them. VL's `reverse` flips the range rather than the
  domain too, so both writers order a reversed legend the same way.
- **Linetypes go through core's `linetype_to_stroke_dash`**, not hephaestus's
  builtins by name: ggsql resolves an ordinal linetype range to ggplot2-style hex
  patterns, and core's parser is what VL uses, so routing through it is what keeps
  the two writers drawing the same dashes.
- **A null category travels as `channels::NULL_CATEGORY`.** ggsql trains a
  categorical domain over nulls, but hephaestus's `DataColumn` has no
  null-carrying variant, so domain and data agree on a sentinel string instead
  (`scales::category_value` and `column_to_channel`). Labels are unaffected —
  they travel separately through `with_breaks_labeled`.

## Faceting and panels

ggsql resolves faceting fully (layout, `free` flags, Wrap's `ncol`, per-row
`__ggsql_aes_facet1__`/`facet2__`), so `facet.rs` only lays it out: a
`Composition` of named patches plus a `Vec<Panel>` the write loop iterates.
Unfaceted is the same path with one panel, so there is no branch.

- **Ordering** mirrors the Vega-Lite writer's `resolve_facet_ordering`: binned
  facets by bin centre, else the facet scale's `input_range` then numeric-aware
  ascending, then `reverse`.
- **Slicing** is `DataFrame::take` on matching row indices. A layer with no facet
  column is used whole in every panel.
- **Every cell is a panel, empty or not.** A Grid row × column combination absent
  from the data still gets its `HPlot` — background, grid, edge axis and strip —
  because the grid must stay rectangular and the strips must keep describing every
  row and column (ggplot2's `facet_grid`, and the Vega-Lite writer). Two things
  follow, both in the `write` loop: the panel builds no geoms (nothing to draw over
  zero rows), so it must bind `x`/`y` itself — hephaestus derives the panel grid
  from the scales bound to the projection's channels, which a geom would otherwise
  have bound — and it must not count as the legend-capturing panel. A **free**
  dimension has no extent to compute there either, so `PanelScales::use_shared`
  points that dimension back at the global scale.
- **Strip labels** come from `Level { key, value, is_null, label }` — `key`
  selects rows, `label` is the text. Discrete levels honour `RENAMING`
  (suppressed → `Some("")`, *not* `None`, so hephaestus still reserves the strip
  slot and panels stay aligned); binned levels join the column's bin **centre**
  back to its bin via `scales::bin_at_centre` and label it with the bin's range.
- **Edge-only axes** (the ggplot2 look): `Panel::{first_col, last_row}` are
  honoured in `projection::apply_proj_cartesian`. A free dimension forces its
  axis onto every panel.

## Projections and axes

`projection::apply_projection` dispatches on `CoordKind` and **creates the axes**,
because the axis kind depends on the coord: Cartesian gets bottom/left rails,
Polar an angular ring + a radial rail, Map neither (the clip boundary and
graticules are the chrome). `has_real_axis` suppresses an axis whose position
scale is a synthetic `__ggsql_stat_dummy` (a pie's radius, a bar with no x),
mirroring the VL writer's `AxisInfo::suppress`.

**Axis titles are not on the rails.** A rail is per panel, so titling it would
label every facet row and column — and every panel of a free dimension. The
figure has one x and one y, so it gets one centred title each, installed on the
composition by `write` from `projection::composition_axis_titles` alongside the
plot-level labels. Same suppression rules (`has_real_axis`, Cartesian only), and
no unfaceted special case: a 1×1 composition puts the title where a panel's own
would have gone.

A **categorical angle** makes a radar rather than a pie. ggsql resolves that and
records `properties["radar"]`; the writer swaps `PolarProjection::full_circle`
for `::radar(n)`, which brings `PolarEdgeStyle::Chord` (polylines bend at each
category boundary instead of arcing between them) and `theta_break_fracs` at the
band centres `(i + 0.5) / n` — exactly where `Scale::map` puts a discrete scale's
categories, so spokes, grid polygons and data line up with no further wiring. The
radial rail's `theta_frac` is a **0–1 fraction of the sweep**, not an angle.

Map coordinates arrive **pre-projected from SQL**, so hephaestus reprojects
nothing: a `CustomProjection` takes `computed["panel_boundary"]` as its clip
surface and `graticule_lon`/`graticule_lat` as its grid, all decoded from WKT by
`channels::wkt_to_*`. Custom's coordinate math equals Cartesian, which is exactly
what pre-projected data wants. Because those coordinates are already in one
linear unit on both axes, the panel's `aspect_ratio` is **1.0** — it is the
data-space x-unit : y-unit ratio, not a panel width:height ratio, so feeding it
the bbox's own proportions stretches every map by exactly that factor.

The bbox becomes the `pos1`/`pos2` domains through `map_range`, which pads a real
span by `MAP_PADDING` (10%, split around its centre) so the framing matches the
Vega-Lite writer's projection fit (`span * 1.1`) and a shape on the boundary is
not drawn against the panel edge.

## Legends

Legends live on the **composition**, never on a per-panel plot, so a faceted plot
gets one shared legend. `Ctx` carries a `RefCell<Vec<Legend>>` sink that is
`Some` only while building the **first** panel — every panel produces the same
legends, since all are built from the globally resolved scales — and `write`
registers the captured set once. This covers the single-panel case with no
special-casing.

Beyond that, deduplication is hephaestus's: `collapse_legends` merges legends
whose scales are equivalent, which is what makes `color AS <var>` (mapped onto
*both* `fill` and `stroke`, hence two scales) render as one swatch. Do not build
a writer-side dedup map.

A legend key paints only what it is told to paint — nothing is inherited from the
plot — so `pin_constants` dresses each key in the layer's own constants, walking
the same `MaterialSpec` table the geom wired itself from. That table already
encodes the geom's aliasing (`color` → `fill` for an area, → `stroke` for a
line), so the key ends up styled like the marks it describes. Exactly two rules,
and everything else pins:

- **Never pin the scaled channel**, or the key overrides the thing it exists to show.
- **Never pin a channel a scale owns.** A data-mapped aesthetic's column holds
  domain values, not visual ones, and it carries its own legend. When that
  channel is the key's *body* colour, a non-colour legend falls back to a neutral
  grey; a colour-scaled legend takes no fallback at all, because ggsql maps
  `color` onto both `fill` and `stroke` and hephaestus only collapses those two
  legends while their keys stay equivalent.

That includes the channels that decide how much room the glyph takes. hephaestus
sizes each swatch **cell** from the key it holds, so `size`, `linewidth` and
`shape` pin like any other constant — `SETTING shape => 'star'` puts stars in the
legend, and `SETTING size => 12` gets a cell that fits the marker. It also
includes `fill_opacity` / `stroke_opacity`, which a key carries separately just as
a geom does: `opacity => 0` on a point geom pins straight through and leaves the
key as open a circle as the marks are.

**A geom's `MaterialSpec` table is therefore the whole vocabulary of what its key
can wear** — a channel the geom sets outside that table is invisible to
`pin_constants`, however constant it is. So every aesthetic the key kind consumes
belongs in the table, even one that needs converting first: a text key takes
`family`, `weight`, `italic` and `angle`, which is why those are `RangeKind`s
rather than per-row code in `geom/text.rs`. `SETTING typeface => 'Times New
Roman', italic => true` dresses the swatch in the same face as the marks, and a
rotated layer gets a rotated key (as ggplot2's `draw_key_text` does — the cell is
sized from the rotated glyph, so nothing clips).

One legend is recorded per **aesthetic**, not per channel. A geom may drive
several channels from one aesthetic — a ribbon sends `stroke` to both edge curves
— and they describe one scale, so they get one swatch. Recording a second does
not merely duplicate it: the extra key is `scaled` on the mirror channel
(`stroke2`), which no key kind consumes, so it resolves to nothing and hephaestus
paints its "row isn't empty" placeholder in ink over the real key. Cross-*layer*
dedup is still hephaestus's `collapse_legends`.

`ggsql_theme()` is the one hook for chrome the writer overrides — currently just
suppressing the colorbar frame hephaestus otherwise inherits from its default
`RectElement`. Anything the two writers must agree on that is neither a scale nor
a channel belongs there.

## The viewer is not a writer

[`window.rs`](window.rs) holds `PlotViewer`, which produces no output at all. It
is not a `Writer` impl on purpose: `Output = ()` would put "blocks, main thread
only, native only" into that trait's contract for one implementor's sake.
`from_options` plus `show(&Spec)` gives the same option ergonomics without
claiming it writes anything.

It lives in this crate rather than in `ggsql-cli` because the CLI uses only
public `ggsql::*` API and has no renderer dependency — see
[`/ggsql-cli/CLAUDE.md`](../../../ggsql-cli/CLAUDE.md). So the *behaviour* goes
public as a type instead of `build_composition` going public.

Two things follow from what a window is:

- **Resize needs no code.** `Frame::parts()` reports the surface's own size and
  dpi each frame, and the composition re-solves its layout for them — so a
  resize is a re-layout, not a rescale. That is the same property the `hep`
  format exists to preserve.
- **`units` and `dpi` are rejected, with a reason.** A window is sized in
  logical pixels and its resolution belongs to the display (`frame.dpi()` wins
  every draw), so accepting either would be accepting a setting that is then
  ignored — exactly the silent failure `reject_unknown` exists to prevent. The
  error says so rather than reporting them as typos. Option parsing reuses
  `canvas::{whole_pixels, parse_background}`, which is why those are free
  functions rather than `Canvas` methods.

## Adding a geom

1. Add a module under [`geom/`](geom/) returning a `GeomSpec`, and dispatch it in
   `geom/mod.rs::build_into_plot` via `build_and_add::<TheHephaestusGeom>`. Add
   the `GeomType` to `is_supported` — `validate` rejects anything not listed.
2. Get the ggsql defaults right in the `material` table: check what the geom's
   ggsql definition actually sets, since those arrive as `Literal`s and the
   `MatDefault` only fires when nothing is mapped.
3. Reach for a **custom builder** (as [`geom/text.rs`](geom/text.rs),
   [`geom/spatial.rs`](geom/spatial.rs) and the composites do) only when the geom
   has no plain x/y columns, computes its positions, or reads a layer *parameter*
   rather than an aesthetic. Even then, route materials through `wire_material` /
   `resolve_material` — that is what keeps data-mapped aesthetics working and what
   dresses the legend key. A material aesthetic needing a **unit or keyword
   conversion** is not a reason to hand-roll it: add a `RangeKind` and keep it in
   the table (see `text`'s font face).
4. Check the **densified** path: under a map `PROJECT`, ggsql expands segment /
   rule / ribbon / tile into per-vertex rows and remaps the extent aesthetics
   onto plain `pos1`/`pos2`. [`geom/densified.rs`](geom/densified.rs) runs
   *before* the `GeomType` match and reuses `line::spec` / `polygon::spec` whole.
   A geom that densifies but isn't handled there fails *silently*, not loudly.
5. When in doubt about behaviour, read what the Vega-Lite writer does for the
   same geom and match it — the two writers are meant to agree.

## Testing

The shared corpus lives at the bottom of [`mod.rs`](mod.rs); each writer's own
option tests live beside it in its own file:

```sh
# Everything. `hep-read` is test-only and unlocks the round trip.
cargo test --features all-writers,hep-read --lib writer::hephaestus

# The GPU-free subset — hard assertions, and what CI can rely on.
cargo test --features svg,pdf,hep,hep-read --lib writer::hephaestus
```

**Option tests do not repeat themselves.** `canvas::assert_canvas_semantics::<W>()`
covers the five shared keys — defaults, unit conversion, the `MAX_DIMENSION`
bound, the background spellings, and that a bad value names its own option — and
is called once per writer, which is what catches a writer that parses a canvas
key itself or forgets to pass its own keys through. Transparency is separate
(`assert_transparent_background`), because JPEG has no alpha channel and refuses
it. A writer's own file then tests only the keys its format adds.

### The corpus runs through every writer

`assert_renders(query)` drives **each compiled writer** over one query, so a
corpus entry is written once and checked by all of them. The ~78 `renders_*`
tests are that corpus: one query per geom, facet mode, scale kind, position
adjustment and projection.

**The vector assertions are what makes this a regression net.** They need no
adapter, so they run in CI and on a headless box: `<svg …>` opened and closed,
a non-zero `<path>` count, `%PDF-` and `%%EOF`, and — the real one —
**`warnings()` empty**, meaning nothing in the whole corpus reached a case a
vector format cannot express. The raster assertion still skips where there is
no adapter (`assert_png_or_skip` matches on the substring `"GPU renderer"`), so
a green run has never proved a *raster* render happened. Before the vector
writers existed, that was the only kind of end-to-end assertion there was.

### The assertions only readable output can make

`mod svg_text` checks the [governing principle](#the-governing-principle)
*directly*, which no raster test can: SVG output is text, so the breaks, labels
and titles ggsql resolved can be read back out of it.

- Tick labels appear verbatim, in ggsql's own number formatting.
- Facet strip labels appear **once each, in panel order** — previously asserted
  only against `build_panels`, never against rendered output.
- `RENAMING` reaches both an axis rail and a legend key.
- A binned scale's resolved edges reach the colorbar.
- Every `LABEL` slot appears, and markdown is **parsed** — no literal `*`, and
  the emphasised run carries a style.
- `text=outline` → zero `<text>` and more `<path>`; `id-prefix` rewrites every
  id *and* every `url(#…)` reference.
- `units=in` → a `pt` root over a pixel `viewBox`, so the file prints at the
  size it was asked for.

`mod pdf_structure` does the same for what PDF's structure exposes: the
`/MediaBox` at 72 pt per inch, `compress=false` leaving no `/FlateDecode`, and
`/FontFile2` proving the fonts are subset in.

`mod hep_roundtrip` (behind the test-only `hep-read` feature) is the strongest
single test here: write a document, read it back into a **new** composition,
render both to SVG and compare **byte for byte**. Any loss anywhere in the
format — a scale, a break, a theme entry, a channel column, a geom — shows up as
different drawing commands. SVG is the comparison surface precisely because it
is deterministic text; a raster comparison would be at the mercy of GPU
antialiasing, which is not bit-reproducible even between two runs of the same
code.

### Still eyeballing

- **Exact-text assertions** — `facet_strips_*` and the `binned_bins` /
  `bin_at_centre` / temporal-scale unit tests need no GPU either.
- **Snapshot tests do not exist.** Whole-picture correctness is still verified
  by eye, usually against the Vega-Lite render of the same query. Assume a
  hephaestus version bump needs re-eyeballing:

```sh
cargo run -p ggsql-cli --features png -- exec "<query>" \
    --reader "duckdb://memory" --writer png --output /tmp/out.png
```

For eyeballing *at scale* — after a hephaestus bump, or when hunting the kind of
small omission that only shows up across the whole feature surface — use the
visual-test harness instead of one-off queries. It renders every executable
```` ```{ggsql} ```` cell in [`/doc/`](../../../doc/) (≈190 in `doc/syntax/`
alone) and writes one HTML report pairing each query with its render, optionally
beside the Vega-Lite render of the same `Spec`:

```sh
cargo run -p ggsql-cli --features png --example visual_test -- --compare
open target/visual-test/index.html
```

It never stops on a failure — an error or a panic is captured against its cell —
so one run inventories every gap at once. Implementation notes:
[`/ggsql-cli/CLAUDE.md`](../../../ggsql-cli/CLAUDE.md).

## Operational constraints

- **A GPU adapter is required by the four raster writers**, at render time, and
  by nothing else. CI installs Mesa's lavapipe; headless containers need
  something equivalent. The vector and document writers need neither an adapter
  nor wgpu, which is why they are the default ones.
- **The backend is Vello Hybrid, not vello classic**, and the choice is named
  in exactly one place ([`raster.rs`](raster.rs)) so it stays swappable. Hybrid
  computes coverage on the CPU and gives the GPU a plain render pipeline, which
  buys two things: its GPU buffers are sized to the scene's actual content
  instead of fixed caps, so a dense plot has **no draw-count ceiling**; and it
  can paint binary coverage, so a hit test reports exactly one id per pixel
  rather than a blend of two — vello classic antialiases its pick pass and can
  report an id that was never drawn. The second matters only once interaction
  lands, but it is the reason not to defer the choice. Output differs from
  vello classic by antialiasing alone (~2% of pixels on a scatter, max channel
  delta under 70, geometry identical). `hephaestus/vello-hybrid` transitively
  enables `hephaestus/png`, so a webp-only build still compiles the PNG codec.
- **The raster ceiling is the GPU's, not the renderer's.** The device is asked
  for as much as it grants up to 16384 px per dimension, which is
  `MAX_RASTER_DIMENSION` and what `check_size` guards before anything is
  allocated — so the error can name the limit and point at `svg`/`pdf`, which
  have none. A device offering less rejects the frame itself with its own limit
  named. Verified on Apple silicon: 4600×3100, 8000×2000, 16000×1000 and a
  faceted 10000×6000 all render; 17000×1000 is refused up front.
- **fontconfig is a *runtime* dependency on Linux, not a build-time one.** Text
  layout goes through parley/fontique, which enumerates fonts through the
  system fontconfig whatever backend draws — so this applies to `svg` and `pdf`
  just as much as to `png`. But `src/Cargo.toml` names `fontique` directly for
  one reason: to turn on `fontconfig-dlopen`, which parley does not re-export.
  With it, `yeslogic-fontconfig-sys`'s build script makes no pkg-config call at
  all and `libfontconfig.so.1` is loaded on use, so **`libfontconfig1-dev` is
  not needed to build** — which is what lets `svg`/`pdf`/`hep` be default
  features without breaking `cargo install ggsql-cli` on a bare box. Verified
  in `debian:bookworm-slim` and `manylinux_2_28`, neither of which ships the
  `-dev` package; CI installs only the runtime library so every run re-checks
  it. macOS uses CoreText and needs nothing.

  **With no fontconfig at all, a render silently loses all text**: geometry
  draws, every `<text>` disappears, and the exit code is 0. Worth knowing when
  a minimal container produces an unlabelled plot.
- **A GPU is needed for raster output, not to see a plot.** `svg`, `pdf` and
  `hep` need no adapter and no wgpu, so they are the fallback for a machine
  that has none — and the reason CI has hard assertions at all.
- **The vector and document writers are default features *and* MSRV-clean.**
  `svg`, `pdf` and `hep` need no adapter, no wgpu and no `-dev` package, so
  there is nothing to opt into — and they still compile on CRAN's 1.86, which
  is what keeps them available to the R package's vendored copy. What refuses
  on 1.86 is cargo's *declaration* check, because `parley` declares 1.88 while
  compiling fine on it, and `--ignore-rust-version` bypasses a declaration:

  ```sh
  cargo +1.86 check --ignore-rust-version -p ggsql   # default features; passes
  ```

  CI runs exactly that, so the claim is checked rather than asserted. The four
  raster writers are genuinely 1.88+ (wgpu), which is one more reason they are
  not default. **Do not use a 1.87+ std API anywhere in the library** —
  `rust-version = "1.86"` keeps clippy flagging one as a lint, and that guard
  is why the declaration stays at 1.86 rather than following `parley`'s.
- **The dependency is the published `0.4.1` crate** (`src/Cargo.toml`), pinned
  with `default-features = false` so the GPU rasteriser arrives only with
  `raster`. So
  nothing here blocks publishing ggsql. hephaestus's own semver contract extends
  to the `kurbo`, `peniko` and `wgpu` types in its public API, so a bump in any
  of those is a breaking change to this writer even when hephaestus's own API
  holds still.

## Known gaps

Deliberately not done, in rough order of how likely they are to bite:

- **No committed snapshot fixtures** (see [Testing](#testing)) — whole-picture
  correctness is checked by eyeballing, with the harness's `--baseline` for
  doing it at scale. The SVG corpus is the natural fixture surface, being
  deterministic text where a 2 px panel shift reads as a hunk rather than as a
  changed hash; whether to commit it is still open.
- **Log-scale tick labels are wrong, and not because of this writer.** ggsql
  resolves a 1–100 `log10` domain to breaks of
  `[5e-308, 2e-256, …, 100]`, and both writers faithfully print those. The fix
  is in scale resolution; nothing changes here. Recorded as an ignored test
  (`svg_text::log_tick_labels_should_be_decades`).
- **No axis label thinning.** ggsql's resolved breaks are drawn as-is, so a
  narrow facet panel can crowd or overlap long labels — which is why
  `free_continuous_scale` narrows the *global* breaks to a panel rather than
  letting hephaestus invent per-panel ones.
- **No switch on rich-text chrome.** [`ggsql_theme`](wiring.rs) turns markdown on
  for the whole chrome cascade, so a title that wants a literal `*` has no way to
  ask for one. The text layer has `parse`; chrome waits for ggsql to grow a theme
  concept, which is where the same switch belongs.

## See also

- [`../vegalite/CLAUDE.md`](../vegalite/CLAUDE.md) — the sibling writer to mirror.
- [`../../CLAUDE.md`](../../CLAUDE.md) — the core crate: feature flags, pipeline.
- [`../../plot/CLAUDE.md`](../../plot/CLAUDE.md) — the AST and scale types this
  writer consumes.
- [`/doc/syntax/`](../../../doc/syntax/) — authoritative ggsql syntax reference.
