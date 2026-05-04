## [Unreleased]

### Added

- New `aggregate` SETTING on Identity-stat layers (point, line, area, bar, ribbon,
range, segment, arrow, rule, text). Collapses each group to a single row by
replacing every numeric mapping in place with its aggregated value. Accepts a
single string or array of strings; entries are either unprefixed defaults
(`'mean'`) or per-aesthetic targets (`'y:max'`, `'color:median'`). Up to two
defaults may be supplied — the first applies to lower-half aesthetics plus all
non-range layers, the second to upper-half (`max`/`end` suffix). Numeric
mappings without a target or applicable default are dropped with a warning.
- Add cell delimiters and code lens actions to the Positron extension (#366)
- ODBC is now turned on for the CLI as well (#344)
- `FROM` can now come before `VISUALIZE`, mirroring the DuckDB style. This means
that `FROM table VISUALIZE x, y` and `VISUALIZE x, y FROM table` are equivalent
queries (#369)
- CLI now has built-in documentation through the `docs` command as well as a
skill for llms through the `skill` command (#361)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we
handled auto-resizing in the plot pane. We now have a per-output-location path
in the Jupyter kernel (#360)
- Passing the shape aesthetic via `SETTING` now correctly translates named
shapes (#368)
- Asterisk shape now has lines 60 degrees apart, giving an even shape

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
- The `orientation` setting on `ribbon` and `range` layers. With explicit `xmin`/`xmax` or `ymin`/`ymax` mappings, orientation is unambiguous and is auto-detected from the mappings; the override is no longer needed.

## [2.7.0] - 2026-04-20

- First alpha release. No changes tracked before this
