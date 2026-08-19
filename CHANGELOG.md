## [Unreleased]

### Added
- New `PngWriter` renders a plot to a PNG raster image via
  [hephaestus](https://github.com/posit-dev/hephaestus), behind a new
  off-by-default `png` feature (`--writer png` in the CLI). `LABEL caption`
  and the new `minor_breaks` setting have no Vega-Lite equivalent and render
  only here. Requires a working GPU adapter — hardware or software, e.g.
  lavapipe — at render time.
- Writers can be configured from key–value options: `Writer::from_options` takes
  a `WriterOptions` set, and the CLI collects them from a repeatable
  `--writer-option key=value` flag on `exec` and `run` (short `-D`, also
  spellable `--writer-options`). Several settings can be collapsed into one flag
  separated by `;` — `-D 'width=1600;dpi=150'`, quoted because shells read `;`
  themselves — and the two forms mix. The png writer
  takes `width`, `height`, `units` (`px`, `in`, `cm`, `mm`, `pt`), `dpi`, and
  `background` (any CSS color, including `transparent`), defaulting to a
  1500×1000 px white canvas at 300 dpi; the Vega-Lite writer takes none. An
  unknown key or unusable value is an error naming the option, not a silently
  ignored setting.
- `--reader`, `--writer`, and `--output` gained the short forms `-r`, `-w`, and
  `-o` on `exec` and `run`; `validate --reader` also takes `-r`.
- Text is rendered as rich text (markdown) by the png writer. A text layer's
  `label` is parsed for `**bold**`, `*italic*`, `_underline_`, `~~strike~~`,
  `` `code` `` and marquee-style `{selector body}` spans that set a colour or
  size (`{.red hot}`, `{#0072B2 blue}`, `{.20 big}`), and so are the plot title,
  subtitle, caption and axis titles set with `LABEL`. Legend titles and break
  labels (axis tick labels, legend keys) do not parse yet and show their markers.
  The new `parse` setting on the text layer turns it off for that layer
  (`SETTING parse => false`), drawing the label exactly as given; it defaults to
  `true`. Chrome text has no switch yet. The Vega-Lite writer has no rich-text
  equivalent and ignores `parse`, always drawing text literally.
- New `minor_breaks` setting on continuous scales, controlling the unlabelled
  subdivisions between breaks: a whole number of minor breaks *per interval between
  two breaks* (`0` removes them), an array of exact positions, or — for temporal
  scales — an interval such as `'week'`. Defaults to a value chosen by the
  transformation. This has no Vega-Lite equivalent and is ignored by that writer;
  the png writer draws them.

- The VS Code / Positron extension now ships the `ggsql-jupyter` kernel, so
  installing the extension is all that is needed to run queries — no separate
  native install. The per-platform builds (`darwin-arm64`, `darwin-x64`,
  `linux-arm64`, `linux-x64`, `win32-x64`) each carry the same signed kernel
  binary the matching installer does. `win32-arm64` and a universal build are
  published without a kernel, and still require the native installer. A new
  `ggsql.kernelStrategy` setting picks between the bundled kernel (the default),
  a kernel installed on the machine, and a fixed path in `ggsql.kernelPath`;
  configuring `ggsql.kernelPath` alone continues to mean that path is used.

### Changed
- Dodging now only takes effect where groups actually meet on a position. A
  layer whose grouping gives every group a position of its own — `colour` mapped
  to the same column as the discrete axis, say — is drawn at its full width
  instead of being squeezed into `1/n` of the band and shifted off its own
  category, which made a coloured ridgeline plot (`DRAW violin SETTING side =>
  'top'`) land its violins between the axis ticks or outside the panel
  altogether. Groups in different facet panels don't meet either. Where any
  position does hold several groups the whole layer still dodges, so an element
  keeps the same slot in every position. Jitter, which dodges before jittering,
  follows the same rule.
- Categorical `y` axes now run bottom-up, so the first level sits at the bottom
  of the panel as it does in ggplot2. This affects every plot with a discrete or
  ordinal `y` — horizontal bars, boxplots and violins by category, points and
  2D jitter — and brings the Vega-Lite writer in line with the raster one, which
  already read this way.
- Banded marks now measure against the full step in the VegaLite writer. A band fraction
  (a bar's `width`, a dodge displacement, a jitter spread, a violin or boxplot
  half-width, a discrete tile's extent) is a fraction of the whole category step,
  so `width => 0.9` leaves a 10% gap — ggplot2's convention. Vega-Lite previously
  subtracted its own default band padding first, making every banded mark there
  narrower than the same query rendered as a raster. This applies to dodged,
  jittered and half-sided layers too, where Vega-Lite reserved a further 20% of
  every step: their marks were narrower, their displacements smaller, and their
  category ticks pulled toward the middle of the panel.

### Fixed
- Positron no longer offers a ggsql runtime on a machine that has no kernel.
  Discovery added `ggsql-jupyter` as a candidate whether or not it was on `PATH`,
  and the accessibility check waved through any bare name, so starting the
  console failed with `KS-19: Kernel path not found` instead of ggsql simply not
  being listed.
- A dodged violin or half-boxplot on a categorical `y` axis is no longer flipped
  in the Vega-Lite writer. Both took their band displacement from an encoding of
  their own that read a ggsql offset as pointing down the screen, so their groups
  came out in the opposite order to every other mark — a violin put the first
  group above the second where a boxplot of the same data put it below, and a
  half-boxplot's box parted company with its own whiskers once dodged. Violins
  are also clipped to the panel now, as every other mark is.
- An identity-scaled column is now read exactly like the equivalent literal.
  `SCALE IDENTITY <aes>` hands its values straight to the aesthetic, so they mean
  what the same value written with `SETTING` means, but several were passed to the
  renderer unconverted: a `size` column was read as a symbol area in pixels²
  rather than the radius in points `SETTING size => 3` gives (markers far too
  small), a `shape` column of names such as `'star'` made Vega-Lite fail to render
  at all, and a `linetype` column of names such as `'dashed'` drew a solid line in
  both writers. `size`, `linewidth`, `fontsize`, `shape` and `linetype` identity
  columns now convert per row, so an identity column and a setting produce the
  same drawing. A value the aesthetic already understands still passes through
  untouched.
- `DRAW bar MAPPING <category> AS y` produced a single bar against a synthetic
  axis instead of horizontal bars. A layer whose geom synthesises its primary
  position (bar, boxplot) now transposes when the user maps a *discrete* `y`, and
  stays put when they map a continuous one — that being the value axis, where a
  lone `DRAW boxplot MAPPING <value> AS y` already belongs.
- `RENAMING` was ignored on a discrete or ordinal scale over a non-string domain
  (`SCALE ORDINAL color RENAMING 6 => 'June'` on a numeric month), because the
  break label was formatted as `6.0` while the rename was keyed on `6`.
- A temporal axis given a calendar interval (`SETTING breaks => '2 months'`) no
  longer draws ticks outside its own domain. The generator steps a whole interval
  past each end, and the filter that trims them back compared only plain numbers,
  so a date break was never constrained at all.
- Minor breaks are no longer extrapolated beyond the outermost major break when
  the majors are unevenly spaced, as they are when set by hand
  (`SETTING breaks => (37, 42, 55)`). Their spacing was taken from the first
  interval alone, so they matched no part of the axis. Evenly spaced majors still
  extend to the edge of the range.
- `Scale::break_labels()` — what a writer reads to label an axis, colorbar or
  legend tick — labels a temporal break with its own date (`1973-04-23`) instead
  of the epoch number its position projects to (`1208`), and keys `RENAMING`
  overrides by that same string, so a rename on a temporal scale is found rather
  than missed. Numeric and categorical labels are unchanged.
- A scale with an explicit input range that no layer trains — `SCALE x FROM (0, 10)`
  alongside a diagonal `rule`, whose position is deliberately kept out of scale
  training — takes its type from that range (numeric or temporal → continuous,
  string or boolean → discrete) instead of staying untyped, so consumers get a
  fully resolved scale.
- The VS Code / Positron extension now offers its "Source Current File" button
  and code cells in plain `.sql` files, so existing SQL can be run against a
  ggsql kernel without renaming it. `.sql` files keep their usual SQL syntax
  highlighting; to get ggsql highlighting as well, map them to the `ggsql`
  language type with `files.associations`, which the extension points out the
  first time a `.sql` file is opened. The new `ggsql.enableSqlFiles` setting
  turns the whole behaviour off.
- The VS Code / Positron extension contributes a "ggsql File" entry to the
  New File dialog.
- The VS Code / Positron extension now ships a language icon that renders in the
  session picker, editor tabs and the Explorer. It previously pointed at a file
  that did not exist, which left the icon blank.
- In plain VS Code, the extension no longer offers run buttons, keybindings or
  Command Palette entries for commands that need the Positron runtime and so
  had no handler there.
- Plots in Positron notebooks no longer come out blank when the cell output is
  rendered before Positron has laid the slot out, which happened on the first
  execution after a kernel started and when reopening a saved notebook. The
  plot sizes itself from its container, so a zero-width first measurement drew
  it at zero size with nothing left to correct it. It now recovers once the
  container has a real width.
- ggsql interpreter sessions in Positron now come back after an extension host
  restart as well as after a window reload. A session the user renamed also
  keeps its name across the restore, and ggsql runtimes are rediscovered on
  every window open rather than risking a stale cache hit.

## 0.4.1 - 2026-06-22

### Changed

- Pinned the Windows image for GitHub Actions runners to Windows Server 2022.

## 0.4.0 - 2026-06-22

### Added
- New `AdbcReader<D: Driver>` for connecting to data sources via
  [ADBC](https://arrow.apache.org/adbc/) (Arrow Database Connectivity), behind
  a new off-by-default `adbc` feature flag. Generic over any concrete
  `adbc_core::sync::Driver`, so concrete drivers (Flight SQL, Snowflake, etc.)
  compose at the call site. Tested against `adbc_datafusion` for in-process
  unit coverage.
- New `aggregate` SETTING on Identity-stat layers (point, line, area, bar, ribbon,
  range, segment, arrow, rule, text). By default it collapses each group to a
  single row by replacing every numeric mapping in place with its aggregated
  value. See the `DRAW` documentation for details (#384).
- Added panel decorations (grid lines, axes, background) for polar coordinates (#156).
- Added `radar` setting to polar coordinates for making radar plots (#418).
- New `side` SETTING on the `boxplot` layer and the `jitter` position, mirroring
  the existing `violin` setting (#439).
- New `hinge` SETTING on the `boxplot` layer, mirroring the existing `range`
  setting (#438)
- New `DRAW spatial` layer for rendering simple features (WKT/WKB) for drawing
  maps and choropleths (#370).
- New builtin dataset `ggsql:world` for showcasing spatial examples. Data is
  a subset of columns from the [Natural Earth](https://www.naturalearthdata.com/)
  country data at 1:110m resolution (#370).
- New `PROJECT TO <map>` family of spatial map projections. For general
  projections, one can use `PROJECT TO crs SETTING target => '+proj=...'`.
  Several named projections have explicit support using e.g.
  `PROJECT TO mollweide`. Works for a subset of layers, notably `spatial`,
  `point`, `text`, `path`, `polygon` and `tile`. Requires a spatial backend
  like PostGIS, SpatiaLite, or DuckDB spatial extension (#455).

### Fixed

- Quoted SQL identifiers (e.g. `"variable.dotted"`) in `VISUALISE` column
  references are now unquoted at parse time, so they correctly match the
  underlying Arrow schema during validation.
- Dodging of horizontal violin plots were broken due to a bad orientation
  assumption in the VegaLite writer. We now correctly use the orientation to
  dodge in the correct dimension (#439).
- Fixed misbehaviour of numeric scale's `RENAMING` clause due to pre-formatting
  issues (#461)

### Changed

- `boxplot`, `violin`, and `range` now support omitting the categorical
  aesthetic, matching `bar`. `point` now treats both position aesthetics as
  optional.
- Upgraded dependencies: duckdb-rs v1.10502, arrow v58 (#447).
- Renamed the `width` setting in the `range` layer to `hinge`. This prevents
  it from clashing with `width` needed by `position => 'dodge'` (#437).
- Pinned the minimum supported Rust version to 1.86 (the maximum Rust version
  CRAN ships) so the crate keeps building for the R bindings.

## 0.3.3 - 2026-05-27

### Fixed

- Add CASE expression support to tree-sitter grammar (#432)
- Fix Vega-Lite spec emitted for boxplot (#449)
- Support predicates in function arguments (#457)

## 0.3.2 - 2026-05-05

### Fixed

- Side effects like `CREATE TEMP TABLE` before the `VISUALISE` statement are now
  separated from directly feeding into the visualisation data (#415)
- Fixed bug where panel axes were unintentionally anchored to zero when using
  `FACET ... SETTING free => 'x'/'y'` (#410).
- Fixed bug where faceted data were matched to the incorrect panels (#409)

### Changed

- Restructured how ggsql integrates with ODBC drivers to use the system ODBC,
rather than bundling unixodbc as part of binary releases. This fixes several
issues on Linux and macOS caused by relative paths to dynamic libraries.

## 0.3.1 - 2026-04-30

### Fixed

- Fixed stacking in faceted plots (#403)

## 0.3.0 - 2026-04-29

### Added

- Add cell delimiters and code lens actions to the Positron extension (#366)
- ODBC is now turned on for the CLI as well (#344)
- `FROM` can now come before `VISUALIZE`, mirroring the DuckDB style. This means
that `FROM table VISUALIZE x, y` and `VISUALIZE x, y FROM table` are equivalent
queries (#369)
- CLI now has built-in documentation through the `docs` command as well as a
skill for llms through the `skill` command (#361)
- The ggsql wasm package is now published on GitHub Releases and NPM (#367)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we
handled auto-resizing in the plot pane. We now have a per-output-location path
in the Jupyter kernel (#360)
- Passing the shape aesthetic via `SETTING` now correctly translates named
shapes (#368)
- Asterisk shape now has lines 60 degrees apart, giving an even shape
- `validate()` now reports an actionable error when a SQL expression (e.g.
`CAST(...)` or a function call) appears inside a `VISUALISE` mapping, instead
of silently treating the entire query as SQL (#389)
- Error messages no longer leak internal aesthetic names. Validation, scale,
and writer errors now report user-facing aesthetic names (`x`, `y`, `panel`,
`row`, …) instead of internal forms (`pos1`, `pos2`, `facet1`, …), translated
based on the active coordinate system and facet layout (#388).
- Fixed opacity calculation in point layers with Vega-Lite (#393)
- Fixed an issue with case-sensitive column references in mappings (#374)
- Fixed SQL function set quantifiers in the ggsql grammar (#395)
- Fixed loading of dynamic libraries in PyPI build of `ggsql-jupyter` (#355, #392)
- Fixed an issue with OOB null-filtering, leading to missing median lines in boxplots (#394)

### Changed

- Reverted an earlier decision to materialize CTEs and the global query in Rust
before registering them back to the backend. We now keep the data purely on the
backend until the layer query as was always intended (#363)
- Relieved some grammatical constraints on the SQL-portion before the VISUALISE
portion (#364).
- Simplified internal approach to DataFrame with DuckDB reader (#365)
- Moved the CLI to its own module rather than be part of the main crate (#379)
- Restructured CLAUDE.md to better deal with the rising complexity of the project (#382)
- Renamed the `errorbar` layer to `range`. The geom was never error-specific and is generally useful for displaying intervals (min/max ranges, candlestick wicks, percentile bands, etc.).
- The `segment` layer now requires both `xend` and `yend` (rather than auto-filling a missing endpoint from the start position). For axis-aligned 1D intervals — lollipops, candlestick wicks, etc. — use the `range` layer instead.

### Removed

- Removed polars from dependency list along with all its transient dependencies. Rewrote DataFrame struct on top of arrow (#350)
- Moved ggsql-python to its own repo (posit-dev/ggsql-python) and cleaned up any additional references to it
- Moved ggsql-r to its own repo (posit-dev/ggsql-r)

## [2.7.0] - 2026-04-20

- First alpha release. No changes tracked before this
