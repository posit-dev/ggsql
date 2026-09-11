## 0.5.0 - 2026-09-10

### Added

- Seven new writers render a plot directly, through [hephaestus](https://github.com/posit-dev/hephaestus): `svg`, `pdf` and `hep` as default features, and `png`, `jpeg`, `tiff` and `webp` off by default and needing a GPU adapter at render time. Each takes the canvas settings `width`, `height`, `units` (`px`, `in`, `cm`, `mm`, `pt`), `dpi` and `background`, plus what its own format offers: `compression` for `png` and `tiff`, `quality` for `jpeg`, `text`, `embed-fonts` and `id-prefix` for `svg`, `compress` and `links` for `pdf`, `lossy` and `embed-fonts` for `hep`. `webp` has none — it is lossless with no rate control.

- `LABEL caption` is honored by the new writers. It has no Vega-Lite equivalent and is ignored there.

- New `ggsql view` subcommand shows a query’s plot in a native window, blocking until it is closed. Resizing re-lays-out the plot rather than stretching it. `-D` (`--viewer-option`) takes `width`, `height`, `background` and `title`; `units` and `dpi` are refused, since a window is sized in logical pixels and its resolution belongs to the display. Behind a new off-by-default `window` feature, and needs a GPU adapter; the subcommand exists either way and says what would enable it.

- New `equal_earth` map projection (`+proj=eqearth`).

- New caching layer that wraps any `Reader` with an in-memory, writeable cache backend (currently duckdb or sqlite), making write-constrained databases usable and avoiding repeated remote reads during interactive iteration. Memoized reads are bounded by a TTL and an LRU byte budget, configurable per connection. Selected by the composite connection scheme `<cache>+<primary>://…` (e.g. `duckdb+odbc://…`) or, in the CLI, by `--cache <duckdb|sqlite>` on `exec`, `run` and `view`. The cache can be cleared mid-session with the `-- @uncache` meta-command.

- Writers can be configured from key–value options: `Writer::from_options` takes a `WriterOptions` set, and the CLI collects them from a repeatable `--writer-option key=value` flag on `exec` and `run` (short `-D`, also spellable `--writer-options`). Several settings can be collapsed into one flag separated by `;` — `-D 'width=1600;dpi=150'`, quoted because shells read `;` themselves — and the two forms mix. An unknown key or unusable value is an error naming the option, not a silently ignored setting.

- `--reader`, `--writer`, and `--output` gained the short forms `-r`, `-w`, and `-o` on `exec` and `run`; `validate --reader` also takes `-r`.

- `--output`’s extension picks the writer when `--writer` is omitted: `svg`, `pdf`, `hep`, `png`, `jpg`/`jpeg`, `tif`/`tiff`, `webp` and `json`/`vl.json` each name their own. An explicit `--writer` still wins, warning on stderr if it disagrees with the extension. An unrecognised extension falls back to Vega-Lite; an extension naming a writer the build lacks is an error.

- New `ggsql::fonts::{register_font, registered_font_families, set_generic_family}` let a host register font faces itself, for a platform with no font database to enumerate. A browser is the case that needs it: it enumerates nothing, so the wasm package ships four Roboto faces and registers them before drawing — without them a plot has no text at all, and, since text is what sets the layout, the wrong margins with it. A page wanting its own typography calls `registerFontFromUrl(url, { genericFor })` instead, which registers the face and points a generic at whatever family name the file turned out to carry — the one place that name exists.

- New off-by-default `webfonts` feature: `fonts::register_font` also accepts the WOFF and WOFF2 containers a font CDN serves a browser, unwrapping them to the sfnt inside. That is how a font arrives at a web page, so `ggsql-wasm` turns it on. Without the feature such a container is refused by name rather than reaching the shaper and registering nothing, which would draw a plot with no text and no indication why.

- Text is rendered as rich text (markdown) by the new writers. A text layer’s `label` is parsed for `**bold**`, `*italic*`, `_underline_`, `~~strike~~`, `` `code` `` and marquee-style `{selector body}` spans that set a colour or size (`{.red hot}`, `{#0072B2 blue}`, `{.20 big}`), and so are the plot title, subtitle, caption, axis titles, legend titles and break labels. The new `parse` setting on the text layer turns it off for that layer (`SETTING parse => false`), drawing the label exactly as given; it defaults to `true`. Chrome text has no switch yet. The Vega-Lite writer has no rich-text equivalent and ignores `parse`, always drawing text literally.

- New `minor_breaks` setting on continuous scales, controlling the unlabelled subdivisions between breaks: a whole number of minor breaks *per interval between two breaks* (`0` removes them), an array of exact positions, or — for temporal scales — an interval such as `'week'`. Defaults to a value chosen by the transformation. This has no Vega-Lite equivalent and is ignored by that writer.

- The VS Code / Positron extension now ships the `ggsql-jupyter` kernel, so installing the extension is all that is needed to run queries. It is offered alongside every ggsql kernel found on the machine — a Jupyter kernelspec, a native install, one on `PATH`, or the path in `ggsql.kernelPath` — each named for the version it reports, so the New Console Session picker shows which is which. A kernel too old to report one is still offered, named without a version. The bundled kernel is the default.

- `ggsql-jupyter` accepts `--version`.

### Changed

- The wasm bundle draws plots with ggsql’s own renderer instead of emitting Vega-Lite. A query is executed in the browser and drawn straight to SVG, so the playground and the live examples on the docs site look like every other ggsql output rather than like a second implementation, and `vega`, `vega-lite` and `vega-embed` are gone from the page — about 1.8 MB less JavaScript. A plot re-solves its layout when its box changes size, so a wider pane gets more tick labels rather than stretched ones. **Breaking:** `GgsqlContext.execute` returns a `GgsqlPlot` to draw rather than a Vega-Lite JSON string; the npm package is entered through a new TypeScript client that adds `PlotView` and `registerDefaultFonts` beside it. `init()` — and a new `initSync()` — wire the extension loader and the CSV/Parquet converters themselves, so `initExtensionLoader` is gone and entering through the package rather than the generated glue is now required.
- Plots in a Positron console now open a `positron.plot` comm, so the Plots pane renders them at its own size, re-renders sharp when resized, and its save, copy and zoom affordances work on them. A new `--max-plots` (default 32) caps the retained history, closing the oldest first. Once the pane has reported a size, a new plot arrives already rendered at it.
- Plots in notebooks and documents are now rendered by the kernel and no longer need network access. A `VISUALISE` query in JupyterLab, a Positron notebook or a Quarto render previously emitted HTML that fetched vega, vega-lite and vega-embed from a CDN on every render; it now emits a rendered image.
- Quarto’s figure settings are honoured: `QUARTO_FIG_FORMAT` selects the writer (`png`, `jpeg`, `svg`, `pdf`) and `QUARTO_FIG_WIDTH`/`_HEIGHT` are read as inches at `QUARTO_FIG_DPI`, so `fig-width: 6` means six inches and a PDF document gets a vector figure.
- Kernel plots render as SVG wherever raster output is unavailable — no GPU adapter, or a build with the new `raster-plots` feature turned off — so a GPU is needed for raster output, not to see a plot.
- An unknown writer, a writer whose feature is off, and an unusable writer setting are now reported **before** the query runs rather than after. `--writer` and `-D` list every writer and its settings in their long help, marking the ones this build does not have and naming the feature that would add each.
- A `FROM` on the `VISUALISE` clause now takes exactly one bare source. It previously reused the SQL `FROM` grammar while only ever reading the first source, so `VISUALISE FROM a, b` silently plotted `a` alone and `VISUALISE FROM a JOIN b ON …` silently dropped the join. Both are now parse errors, as is an alias (`VISUALISE FROM tbl AS t`), which parsed before but had no effect. Join two tables in a `SELECT` and visualise its result.
- Dodging now only takes effect where groups actually meet on a position. A layer whose grouping gives every group a position of its own — `colour` mapped to the same column as the discrete axis, say — is drawn at its full width instead of being squeezed into `1/n` of the band and shifted off its own category, which made a coloured ridgeline plot (`DRAW violin SETTING side => 'top'`) land its violins between the axis ticks or outside the panel altogether. Groups in different facet panels don’t meet either. Where any position does hold several groups the whole layer still dodges, so an element keeps the same slot in every position. Jitter, which dodges before jittering, follows the same rule.
- Categorical `y` axes now run bottom-up, so the first level sits at the bottom of the panel as it does in ggplot2. This affects every plot with a discrete or ordinal `y` — horizontal bars, boxplots and violins by category, points and 2D jitter — and brings the Vega-Lite writer in line with the new writers, which already read this way.
- Banded marks now measure against the full step in the Vega-Lite writer. A band fraction (a bar’s `width`, a dodge displacement, a jitter spread, a violin or boxplot half-width, a discrete tile’s extent) is a fraction of the whole category step, so `width => 0.9` leaves a 10% gap — ggplot2’s convention. Vega-Lite previously subtracted its own default band padding first, and a further 20% of every step for dodged, jittered and half-sided layers, so their marks were narrower, their displacements smaller and their category ticks pulled toward the middle of the panel.
- Position scales like `SCALE lon` and `SCALE lat` transfer their limits to map projections, and transfer their `breaks` setting to the graticule (#492).

### Fixed

- Positron no longer offers a ggsql runtime on a machine that has no kernel.
- A dodged violin or half-boxplot on a categorical `y` axis is no longer flipped in the Vega-Lite writer. Both took their band displacement from an encoding of their own that read a ggsql offset as pointing down the screen, so their groups came out in the opposite order to every other mark — a violin put the first group above the second where a boxplot of the same data put it below, and a half-boxplot’s box parted company with its own whiskers once dodged. Violins are also clipped to the panel now, as every other mark is.
- An identity-scaled column is now read exactly like the equivalent literal. `SCALE IDENTITY <aes>` hands its values straight to the aesthetic, so they mean what the same value written with `SETTING` means, but several were passed to the renderer unconverted: a `size` column was read as a symbol area in pixels² rather than the radius in points `SETTING size => 3` gives (markers far too small), a `shape` column of names such as `'star'` made Vega-Lite fail to render at all, and a `linetype` column of names such as `'dashed'` drew a solid line in both writers. `size`, `linewidth`, `fontsize`, `shape` and `linetype` identity columns now convert per row, so an identity column and a setting produce the same drawing. A value the aesthetic already understands still passes through untouched.
- `DRAW bar MAPPING <category> AS y` produced a single bar against a synthetic axis instead of horizontal bars. A layer whose geom synthesises its primary position (bar, boxplot) now transposes when the user maps a *discrete* `y`, and stays put when they map a continuous one — that being the value axis, where a lone `DRAW boxplot MAPPING <value> AS y` already belongs.
- `RENAMING` was ignored on a discrete or ordinal scale over a non-string domain (`SCALE ORDINAL color RENAMING 6 => 'June'` on a numeric month), because the break label was formatted as `6.0` while the rename was keyed on `6`.
- A temporal axis given a calendar interval (`SETTING breaks => '2 months'`) no longer draws ticks outside its own domain. The generator steps a whole interval past each end, and the filter that trims them back compared only plain numbers, so a date break was never constrained at all.
- Minor breaks are no longer extrapolated beyond the outermost major break when the majors are unevenly spaced, as they are when set by hand (`SETTING breaks => (37, 42, 55)`). Their spacing was taken from the first interval alone, so they matched no part of the axis. Evenly spaced majors still extend to the edge of the range.
- `Scale::break_labels()` — what a writer reads to label an axis, colorbar or legend tick — labels a temporal break with its own date (`1973-04-23`) instead of the epoch number its position projects to (`1208`), and keys `RENAMING` overrides by that same string, so a rename on a temporal scale is found rather than missed. Numeric and categorical labels are unchanged.
- A scale with an explicit input range that no layer trains — `SCALE x FROM (0, 10)` alongside a diagonal `rule`, whose position is deliberately kept out of scale training — takes its type from that range (numeric or temporal → continuous, string or boolean → discrete) instead of staying untyped, so consumers get a fully resolved scale.
- The VS Code / Positron extension now offers its “Source Current File” button and code cells in plain `.sql` files, so existing SQL can be run against a ggsql kernel without renaming it. `.sql` files keep their usual SQL syntax highlighting; to get ggsql highlighting as well, map them to the `ggsql` language type with `files.associations`, which the extension points out the first time a `.sql` file is opened. The new `ggsql.enableSqlFiles` setting turns the whole behaviour off.
- The VS Code / Positron extension contributes a “ggsql File” entry to the New File dialog.
- The VS Code / Positron extension now ships a language icon that renders in the session picker, editor tabs and the Explorer. It previously pointed at a file that did not exist, which left the icon blank.
- In plain VS Code, the extension no longer offers run buttons, keybindings or Command Palette entries for commands that need the Positron runtime and so had no handler there.
- ggsql interpreter sessions in Positron now come back after an extension host restart as well as after a window reload. A session the user renamed also keeps its name across the restore, and ggsql runtimes are rediscovered on every window open rather than risking a stale cache hit.

## 0.4.1 - 2026-06-22

### Changed

- Pinned the Windows image for GitHub Actions runners to Windows Server 2022.

## 0.4.0 - 2026-06-22

### Added

- New `AdbcReader<D: Driver>` for connecting to data sources via [ADBC](https://arrow.apache.org/adbc/) (Arrow Database Connectivity), behind a new off-by-default `adbc` feature flag. Generic over any concrete `adbc_core::sync::Driver`, so concrete drivers (Flight SQL, Snowflake, etc.) compose at the call site. Tested against `adbc_datafusion` for in-process unit coverage.
- New `aggregate` SETTING on Identity-stat layers (point, line, area, bar, ribbon, range, segment, arrow, rule, text). By default it collapses each group to a single row by replacing every numeric mapping in place with its aggregated value. See the `DRAW` documentation for details (#384).
- Added panel decorations (grid lines, axes, background) for polar coordinates (#156).
- Added `radar` setting to polar coordinates for making radar plots (#418).
- New `side` SETTING on the `boxplot` layer and the `jitter` position, mirroring the existing `violin` setting (#439).
- New `hinge` SETTING on the `boxplot` layer, mirroring the existing `range` setting (#438)
- New `DRAW spatial` layer for rendering simple features (WKT/WKB) for drawing maps and choropleths (#370).
- New builtin dataset `ggsql:world` for showcasing spatial examples. Data is a subset of columns from the [Natural Earth](https://www.naturalearthdata.com/) country data at 1:110m resolution (#370).
- New `PROJECT TO <map>` family of spatial map projections. For general projections, one can use `PROJECT TO crs SETTING target => '+proj=...'`. Several named projections have explicit support using e.g. `PROJECT TO mollweide`. Works for a subset of layers, notably `spatial`, `point`, `text`, `path`, `polygon` and `tile`. Requires a spatial backend like PostGIS, SpatiaLite, or DuckDB spatial extension (#455).

### Fixed

- Quoted SQL identifiers (e.g. `"variable.dotted"`) in `VISUALISE` column references are now unquoted at parse time, so they correctly match the underlying Arrow schema during validation.
- Dodging of horizontal violin plots were broken due to a bad orientation assumption in the VegaLite writer. We now correctly use the orientation to dodge in the correct dimension (#439).
- Fixed misbehaviour of numeric scale’s `RENAMING` clause due to pre-formatting issues (#461)

### Changed

- `boxplot`, `violin`, and `range` now support omitting the categorical aesthetic, matching `bar`. `point` now treats both position aesthetics as optional.
- Upgraded dependencies: duckdb-rs v1.10502, arrow v58 (#447).
- Renamed the `width` setting in the `range` layer to `hinge`. This prevents it from clashing with `width` needed by `position => 'dodge'` (#437).
- Pinned the minimum supported Rust version to 1.86 (the maximum Rust version CRAN ships) so the crate keeps building for the R bindings.

## 0.3.3 - 2026-05-27

### Fixed

- Add CASE expression support to tree-sitter grammar (#432)
- Fix Vega-Lite spec emitted for boxplot (#449)
- Support predicates in function arguments (#457)

## 0.3.2 - 2026-05-05

### Fixed

- Side effects like `CREATE TEMP TABLE` before the `VISUALISE` statement are now separated from directly feeding into the visualisation data (#415)
- Fixed bug where panel axes were unintentionally anchored to zero when using `FACET ... SETTING free => 'x'/'y'` (#410).
- Fixed bug where faceted data were matched to the incorrect panels (#409)

### Changed

- Restructured how ggsql integrates with ODBC drivers to use the system ODBC, rather than bundling unixodbc as part of binary releases. This fixes several issues on Linux and macOS caused by relative paths to dynamic libraries.

## 0.3.1 - 2026-04-30

### Fixed

- Fixed stacking in faceted plots (#403)

## 0.3.0 - 2026-04-29

### Added

- Add cell delimiters and code lens actions to the Positron extension (#366)
- ODBC is now turned on for the CLI as well (#344)
- `FROM` can now come before `VISUALIZE`, mirroring the DuckDB style. This means that `FROM table VISUALIZE x, y` and `VISUALIZE x, y FROM table` are equivalent queries (#369)
- CLI now has built-in documentation through the `docs` command as well as a skill for llms through the `skill` command (#361)
- The ggsql wasm package is now published on GitHub Releases and NPM (#367)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we handled auto-resizing in the plot pane. We now have a per-output-location path in the Jupyter kernel (#360)
- Passing the shape aesthetic via `SETTING` now correctly translates named shapes (#368)
- Asterisk shape now has lines 60 degrees apart, giving an even shape
- `validate()` now reports an actionable error when a SQL expression (e.g. `CAST(...)` or a function call) appears inside a `VISUALISE` mapping, instead of silently treating the entire query as SQL (#389)
- Error messages no longer leak internal aesthetic names. Validation, scale, and writer errors now report user-facing aesthetic names (`x`, `y`, `panel`, `row`, …) instead of internal forms (`pos1`, `pos2`, `facet1`, …), translated based on the active coordinate system and facet layout (#388).
- Fixed opacity calculation in point layers with Vega-Lite (#393)
- Fixed an issue with case-sensitive column references in mappings (#374)
- Fixed SQL function set quantifiers in the ggsql grammar (#395)
- Fixed loading of dynamic libraries in PyPI build of `ggsql-jupyter` (#355, \#392)
- Fixed an issue with OOB null-filtering, leading to missing median lines in boxplots (#394)

### Changed

- Reverted an earlier decision to materialize CTEs and the global query in Rust before registering them back to the backend. We now keep the data purely on the backend until the layer query as was always intended (#363)
- Relieved some grammatical constraints on the SQL-portion before the VISUALISE portion (#364).
- Simplified internal approach to DataFrame with DuckDB reader (#365)
- Moved the CLI to its own module rather than be part of the main crate (#379)
- Restructured CLAUDE.md to better deal with the rising complexity of the project (#382)
- Renamed the `errorbar` layer to `range`. The geom was never error-specific and is generally useful for displaying intervals (min/max ranges, candlestick wicks, percentile bands, etc.).
- The `segment` layer now requires both `xend` and `yend` (rather than auto-filling a missing endpoint from the start position). For axis-aligned 1D intervals — lollipops, candlestick wicks, etc. — use the `range` layer instead.

### Removed

- Removed polars from dependency list along with all its transient dependencies. Rewrote DataFrame struct on top of arrow (#350)
- Moved ggsql-python to its own repo (posit-dev/ggsql-python) and cleaned up any additional references to it
- Moved ggsql-r to its own repo (posit-dev/ggsql-r)

## \[2.7.0\] - 2026-04-20

- First alpha release. No changes tracked before this
