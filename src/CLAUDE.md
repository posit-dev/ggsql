# `src/` — ggsql core Rust crate

The core library. Crate name: `ggsql`. Workspace member, declared in `/Cargo.toml`. The `ggsql` CLI binary lives separately in [`/ggsql-cli/`](../ggsql-cli/) and depends on this crate.

For ggsql language semantics, see [`/doc/syntax/`](../doc/syntax/). For Vega-Lite renderer internals, see [`writer/vegalite/CLAUDE.md`](writer/vegalite/CLAUDE.md). For AST types, see [`plot/CLAUDE.md`](plot/CLAUDE.md).

## Entry points

- **`lib.rs`** — library root. Declares modules, re-exports the headline types (`Plot`, `Layer`, `Geom`, `Scale`, `Mappings`, `AestheticValue`, `DataSource`, `Facet`, `FacetLayout`, `SqlExpression`, `DataFrame`), and defines `GgsqlError` + `Result`.

## Module map

```
src/
├── lib.rs                      Library root
├── array_util.rs, compute.rs    Arrow array helpers
├── dataframe.rs                 DataFrame wrapper around arrow RecordBatch
├── format.rs                    Label/number/date formatting
├── naming.rs                    Internal column-name conventions (__ggsql_*)
├── util.rs                      String helpers (and_list, or_list, …)
├── validate.rs                  validate(): syntax + semantic checks without SQL execution
│
├── parser/      Tree-sitter integration → typed AST (Plot)
├── plot/        AST: Plot, Layer, Geom, Scale, Facet, Projection, Mappings  (see plot/CLAUDE.md)
├── reader/      Reader trait + drivers (DuckDB, SQLite, ODBC, Snowflake, …)
├── execute/     Pipeline that turns Plot + Reader → executed Spec
├── writer/      Writer trait + Vega-Lite implementation  (see writer/vegalite/CLAUDE.md)
├── data/        Bundled sample datasets (penguins, airquality)
└── doc/         API.md — public Rust API reference
```

### `parser/`

- `mod.rs` exposes `parse_query()` which builds a `Vec<Plot>` from a query string.
- `source_tree.rs` is the parse-once wrapper: holds the tree-sitter `Tree`, source text, and language; offers a declarative query API (`find_node`, `find_text`, …) plus lazy `extract_sql()` / `extract_visualise()` extractors. It also handles the `VISUALISE FROM <source>` shorthand by injecting `SELECT * FROM <source>`.
- `builder.rs` walks the CST and produces typed `Plot` values. This is where new grammar nodes become `Plot` fields.

Grammar lives in [`/tree-sitter-ggsql/`](../tree-sitter-ggsql/) — when adding syntax, edit `grammar.js`, regenerate, then teach `builder.rs` about the new nodes.

### `reader/`

`Reader` trait exposes `execute_sql()` for SQL → `DataFrame` and `execute()` (default method) for the full ggsql pipeline. Drivers each live in their own file:

| File | Backend | Feature flag |
| --- | --- | --- |
| `duckdb.rs` | DuckDB (in-memory or file) | `duckdb` (default) |
| `sqlite.rs` | SQLite | `sqlite` (default) |
| `odbc.rs` | ODBC | `odbc` (default) |
| `snowflake.rs` | Snowflake (very small placeholder today) | — |
| `connection.rs` | Connection-string parsing for all of the above | — |
| `data.rs`, `spec.rs` | `Spec` type returned by `execute()`, plus DataFrame conversion | — |

`SqlDialect` trait in `mod.rs` lets each driver supply its own type names and information-schema queries. PostgreSQL has a feature flag (`postgres`) but no driver file yet.

### `execute/`

The pipeline that takes a parsed `Plot` plus a `Reader` and produces a fully-resolved `Spec` (typed data per layer, scales resolved, casts applied). Submodules:

- `mod.rs` — top-level `prepare_data_with_reader()` and validation glue.
- `cte.rs` — CTE extraction / materialization for shared subqueries.
- `schema.rs` — schema inspection, type inference, range computation.
- `casting.rs` — `TypeRequirement` derivation and cast-target selection.
- `layer.rs` — per-layer SQL building, transforms, stat application.
- `scale.rs` — scale resolution, type coercion, out-of-bounds handling.
- `position.rs` — position adjustment (stack/dodge/jitter) at execution time.

### `writer/`

`Writer` trait in `mod.rs` (associated `Output` type so writers can return text or bytes). Only Vega-Lite is implemented today; `ggplot2`, `plotters` are reserved feature flags. Implementation deep-dive: [`writer/vegalite/CLAUDE.md`](writer/vegalite/CLAUDE.md).

### `plot/`

Sufficiently large to have its own [`plot/CLAUDE.md`](plot/CLAUDE.md). It holds the AST types and the registries for geoms, scale types, transforms, positions, and coords.

### `doc/`

Just `API.md` — the public Rust API reference for `Reader::execute`, `Writer::render`, `validate`, `Spec`, `Validated`, `Metadata`. End-user docs live in `/doc/`, not here.

## Public API quick reference

Two-stage pipeline:

1. **`reader.execute(query)`** → `Spec` (parses, runs SQL, resolves mappings, applies stats).
2. **`writer.render(&spec)`** → output (Vega-Lite JSON for `VegaLiteWriter`).

`validate(query)` performs syntax + semantic checks without touching a reader.

Full method-by-method reference: [`doc/API.md`](doc/API.md).

## Feature flags

Defined in `Cargo.toml`:

| Flag | Default | Purpose |
| --- | --- | --- |
| `duckdb` | ✓ | DuckDB reader |
| `sqlite` | ✓ | SQLite reader |
| `odbc` | ✓ | ODBC reader |
| `parquet` | ✓ | Parquet support in readers/data |
| `ipc` | ✓ | Arrow IPC support |
| `vegalite` | ✓ | Vega-Lite writer |
| `builtin-data` | ✓ | Bundled penguins/airquality datasets |
| `postgres` | — | PostgreSQL reader (declared, not yet implemented) |
| `ggplot2` | — | ggplot2 writer (declared, not yet implemented) |
| `all-readers` | — | `duckdb` + `postgres` + `sqlite` + `odbc` |
| `all-writers` | — | `vegalite` + `ggplot2` + `plotters` |

`ggsql-wasm` builds with `default-features = false` plus `vegalite`, `sqlite`, `builtin-data`. `ggsql-jupyter` builds with `duckdb`, `sqlite`, `odbc`, `vegalite`.

## Testing

```sh
# All crates in workspace
cargo test --workspace

# Just this crate
cargo test --package ggsql

# A specific feature combination
cargo test --package ggsql --no-default-features --features "duckdb,vegalite"
```

Unit tests live alongside the code (`#[cfg(test)] mod tests`). Integration tests at the bottom of `lib.rs` exercise the end-to-end pipeline against DuckDB and Vega-Lite (gated on both features).

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview, build/test for everything.
- [`plot/CLAUDE.md`](plot/CLAUDE.md) — AST types.
- [`writer/vegalite/CLAUDE.md`](writer/vegalite/CLAUDE.md) — Vega-Lite renderer internals.
- [`/doc/syntax/`](../doc/syntax/) — authoritative ggsql syntax reference.
- [`doc/API.md`](doc/API.md) — Rust public API reference.
- [`/ggsql-cli/CLAUDE.md`](../ggsql-cli/CLAUDE.md) — the `ggsql` CLI binary that wraps this library.
- [`/INSTALLERS.md`](../INSTALLERS.md) — cross-platform installer build (driven from `ggsql-cli`).
