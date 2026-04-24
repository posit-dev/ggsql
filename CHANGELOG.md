## [Unreleased]

### Added

- ODBC is now turned on for the CLI as well (#344)
- `FROM` can now come before `VISUALIZE`, mirroring the DuckDB style. This means
that `FROM table VISUALIZE x, y` and `VISUALIZE x, y FROM table` are equivalent
queries (#369)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we
handled auto-resizing in the plot pane. We now have a per-output-location path
in the Jupyter kernel (#360)

### Removed

- Removed polars from dependency list along with all its transient dependencies. Rewrote DataFrame struct on top of arrow (#350)
- Moved ggsql-python to its own repo (posit-dev/ggsql-python) and cleaned up any additional references to it
- Moved ggsql-r to its own repo (posit-dev/ggsql-r)

## [2.7.0] - 2026-04-20

- First alpha release. No changes tracked before this
