## \[Unreleased\]

### Added

- Add cell delimiters and code lens actions to the Positron extension (#366)
- ODBC is now turned on for the CLI as well (#344)
- `FROM` can now come before `VISUALIZE`, mirroring the DuckDB style. This means that `FROM table VISUALIZE x, y` and `VISUALIZE x, y FROM table` are equivalent queries (#369)
- CLI now has built-in documentation through the `docs` command as well as a skill for llms through the `skill` command (#361)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we handled auto-resizing in the plot pane. We now have a per-output-location path in the Jupyter kernel (#360)
- Passing the shape aesthetic via `SETTING` now correctly translates named shapes (#368)
- Asterisk shape now has lines 60 degrees apart, giving an even shape
- Error messages no longer leak internal aesthetic names. Validation, scale, and writer errors now report user-facing aesthetic names (`x`, `y`, `panel`, `row`, …) instead of internal forms (`pos1`, `pos2`, `facet1`, …), translated based on the active coordinate system and facet layout (#388).
- Fixed opacity calculation in point layers with Vega-Lite (#393)
- Fixed an issue with case-sensitive column references in mappings (#374)
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
