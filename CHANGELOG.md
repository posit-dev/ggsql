## [Unreleased]

### Added

- Add cell delimiters and code lens actions to the Positron extension (#366)
- ODBC is now turned on for the CLI as well (#344)

### Fixed

- Rendering of inline plots in Positron had a bad interaction with how we
handled auto-resizing in the plot pane. We now have a per-output-location path
in the Jupyter kernel (#360)

### Changed

- Reverted an earlier decision to materialize CTEs and the global query in Rust
before registering them back to the backend. We now keep the data purely on the
backend until the layer query as was always intended (#363)
- Simplified internal approach to DataFrame with DuckDB reader (#365)

### Removed

- Removed polars from dependency list along with all its transient dependencies. Rewrote DataFrame struct on top of arrow (#350)
- Moved ggsql-python to its own repo (posit-dev/ggsql-python) and cleaned up any additional references to it
- Moved ggsql-r to its own repo (posit-dev/ggsql-r)

## [2.7.0] - 2026-04-20

- First alpha release. No changes tracked before this
