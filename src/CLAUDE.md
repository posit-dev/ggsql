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
├── spec.rs                      Spec: parse-time result of one VISUALISE/TABULATE statement (Plot or Table)
├── util.rs                      String helpers (and_list, or_list, …)
├── validate.rs                  validate(): syntax + semantic checks without SQL execution
├── fonts.rs                     Font registration, for hosts with no font database
│
├── parser/      Tree-sitter integration → typed AST (Spec: Plot or Table)
├── plot/        AST: Plot, Layer, Geom, Scale, Facet, Projection, Mappings  (see plot/CLAUDE.md)
├── table/       AST stub for TABULATE, parallel to plot/ (no fields yet)
├── reader/      Reader trait + drivers (DuckDB, SQLite, ODBC, Snowflake, …)
├── execute/     Pipeline that turns Plot + Reader → ResolvedPlot
├── writer/      Writer trait + Vega-Lite implementation  (see writer/vegalite/CLAUDE.md)
├── data/        Bundled sample datasets (penguins, airquality)
└── doc/         API.md — public Rust API reference
```

### `parser/`

- `mod.rs` exposes `parse_query()` which builds a `Vec<Spec>` from a query string — one `Spec` per `VISUALISE`/`TABULATE` statement, in source order. Today `build_ast` only ever produces `Spec::Plot`; `TABULATE` isn't wired into the grammar yet.
- `source_tree.rs` is the parse-once wrapper: holds the tree-sitter `Tree`, source text, and language; offers a declarative query API (`find_node`, `find_text`, …) plus lazy `extract_sql()` / `extract_spec()` extractors (the latter covers both `VISUALISE` and `TABULATE`). It also handles the `VISUALISE FROM <source>` shorthand by injecting `SELECT * FROM <source>`.
- `builder.rs` walks the CST and produces typed `Spec` values (`Plot`, boxed for size, or `Table`). This is where new grammar nodes become `Plot`/`Table` fields.
- `sql.rs` extracts structure from SQL fragments over the parse tree.

Grammar lives in [`/tree-sitter-ggsql/`](../tree-sitter-ggsql/) — when adding syntax, edit `grammar.js`, regenerate, then teach `builder.rs` about the new nodes.

### `reader/`

`Reader` trait exposes `execute_sql()` for SQL → `DataFrame` and `execute()` (default method) for the full ggsql pipeline. Drivers each live in their own file:

| File | Backend | Feature flag |
| --- | --- | --- |
| `duckdb.rs` | DuckDB (in-memory or file) | `duckdb` (default) |
| `sqlite.rs` | SQLite | `sqlite` (default) |
| `odbc.rs` | ODBC | `odbc` (default) |
| `cache.rs` | `CachingReader` — wraps any primary `Reader` with an in-memory cache | `duckdb` or `sqlite` |
| `connection.rs` | Connection-string parsing for all of the above | — |
| `spec.rs` | `ResolvedPlot` type returned by `execute()`, plus DataFrame conversion | — |
| `data.rs` | Bundled sample datasets — the `ggsql:` builtins | `builtin-data` |

`SqlDialect` trait in `mod.rs` lets each driver supply its own type names, information-schema queries, and spatial helper methods (`sql_st_transform`, `sql_geometry_to_wkb`, `sql_geometry_bbox`, `sql_ensure_geometry`, `sql_select_replace`, `sql_spatial_setup`).

**Caching layer.** `CachingReader` (`cache.rs`) wraps a primary reader plus an in-memory `CacheBackend`, splitting work across two `Reader` surfaces. **`execute_sql` = source**: base reads of the user's data plus user setup/DML run on the primary (with result memoization), except `ggsql:` builtins, the `__ggsql_cache_meta__` table, and reads that reference a cache-resident internal table, which go to the cache. **`execute_sql_cached` = compute**: all dialect-generated/derived SQL (schema probes, stats, projection/map transforms, spatial setup, final layer queries — everything operating on `__ggsql_*` tables) runs on the cache; it defaults to `execute_sql` so a plain reader runs everything on one connection. Cache routing is by **exact-identifier membership** in the set of tables registered into the cache. Memoization keys on `hash(primary_uri + sql)` and is tracked in the `__ggsql_cache_meta__` table inside the cache backend. Each memoized read is bounded by a **TTL** (default 300s) and the whole memo by an **LRU byte budget** (default 512 MB); both are configurable via `CacheConfig` (env `GGSQL_CACHE_DISABLED`/`GGSQL_CACHE_TTL`/`GGSQL_CACHE_MAX_BYTES`, or per-connection URI query parameters `?cache_ttl=…&cache_max_bytes=…&cache_disabled=…`). The `__ggsql_cache_meta__` table is queryable for introspection (`SELECT * FROM __ggsql_cache_meta__`). Pure/non-visual SQL (CLI table fallback, Jupyter) goes through `execute_sql` so it reads the primary rather than the empty cache. `Reader::materialize_table` (default = `CREATE TEMP TABLE` on the reader, no Rust roundtrip) is overridden to read the body via the source surface and `register()` the result into the cache, so the primary is never written to; `Reader::caches_sources()` (default `false`, `true` for `CachingReader`) gates the executor's per-layer source staging: file sources are staged on the cache surface, while identifiers go through `materialize_table`, which routes the read to the cache (CTEs, builtins, cache-resident tables) or the primary as needed. `dialect()` returns the **cache** dialect, and every compute-surface failure is prefixed with the cache backend's scheme (``on the `duckdb` cache backend: …``) so a cache-dialect driver error is not mistaken for one from the user's own connection. Selected via the composite `<cache>+<primary>://` scheme (`reader_from_uri` / `split_cache_uri`) or the CLI `--cache` flag; off by default.

### `execute/`

The pipeline that takes a parsed `Plot` plus a `Reader` and produces a `ResolvedPlot` (typed data per layer, scales resolved, casts applied). Submodules:

- `mod.rs` — top-level `prepare_data_with_reader()` and validation glue.
- `cte.rs` — CTE extraction / materialization for shared subqueries.
- `schema.rs` — schema inspection, type inference, range computation.
- `casting.rs` — `TypeRequirement` derivation and cast-target selection.
- `layer.rs` — per-layer SQL building, transforms, stat application.
- `scale.rs` — scale resolution, type coercion, out-of-bounds handling.
- `position.rs` — position adjustment (stack/dodge/jitter) at execution time.

### `writer/`

`Writer` trait in `mod.rs` (associated `Output` type so writers can return text or bytes, and `from_options` for configuration a frontend collects as key–value pairs — `options.rs`'s `WriterOptions`, parsed from the CLI's `--writer-option`). Two families:

- **Vega-Lite** (`vegalite` feature, default) — emits Vega-Lite JSON. Deep-dive: [`writer/vegalite/CLAUDE.md`](writer/vegalite/CLAUDE.md).
- **HTML** (`html` feature, default) — `HtmlWriter` renders a resolved `Table` (a `TABULATE` query) as a bare `<table>`; the only writer that supports tables at all. Plot-only otherwise (`vegalite` and the hephaestus writers below).
- **The renderer-backed writers** (seven of them; `svg`, `pdf` and `hep` default, the four raster ones not) — all live in `writer/hephaestus/`, named after the renderer they wrap; that name is internal, and the module is private so only the writers, `Canvas` and `RasterRenderer` are public. They share their whole pipeline — `Canvas` for configuration, `compose` for the plot composition, then either `raster` for pixels or `vector` for drawing commands — and differ only in what they do with the result. Deep-dive (architecture + known gaps): [`writer/hephaestus/CLAUDE.md`](writer/hephaestus/CLAUDE.md).

  | Feature | Default | Writer | Output | GPU |
  | --- | --- | --- | --- | --- |
  | `png` / `jpeg` / `tiff` / `webp` | — | `PngWriter`, `JpegWriter`, `TiffWriter`, `WebpWriter` | image bytes | required |
  | `svg` / `pdf` | ✓ | `SvgWriter`, `PdfWriter` | vector text / one PDF page | **none** |
  | `hep` | ✓ | `HepWriter` | a `.hep` plot document — no picture | **none** |

  Plus `PlotViewer` behind the `window` feature — not a writer, since it returns no output, blocks, and must run on the main thread. It shows the same composition in a native window, re-laying-out on resize.

  The three GPU-free writers go through the same composition and the same `render` call (which takes `&mut dyn SceneBuilder`), so they need no adapter and pull in no wgpu. **That is why they are default**: nothing about them has to be opted into, including on Linux, where `fontconfig-dlopen` removes the build-time `libfontconfig1-dev` requirement (see [`writer/hephaestus/CLAUDE.md`](writer/hephaestus/CLAUDE.md)). They also still compile on the CRAN MSRV — `cargo +1.86 check --ignore-rust-version -p ggsql`, where the flag is needed only because `parley` *declares* 1.88 while compiling fine on 1.86. Only the raster writers need an adapter and are genuinely 1.88+, and a raster dimension is capped at what the GPU grants, up to 16384 px.

Three **internal** features carry the split, enabled by the writer features rather than named directly: `graphics` is the shared composition layer, and `raster = graphics + hephaestus/vello-hybrid` adds the GPU rasteriser. Only `raster` pulls in wgpu, vello_hybrid and pollster, which is what lets a vector-only build skip them — `cargo tree --features graphics` shows none of the three, `--features png` shows 19. `raster-writer` then narrows `raster` once more, to "some writer actually reads pixels back" — the viewer needs the rasteriser without ever doing that. `graphics` is the single module gate for `writer/hephaestus/`, so adding a format needs no change there.

### `fonts.rs`

Registering font faces with the shaper, behind `graphics`. Natively the operating
system enumerates fonts and nothing here is needed; a browser enumerates none, so
a wasm host has to hand the faces over itself or every plot comes out with no
text and — since text is what sets the margins — the wrong layout too.

`register_font` takes sfnt bytes, and with the optional `webfonts` feature the
WOFF and WOFF2 containers a font CDN serves a browser as well. Without it those
are refused by name, because compressed bytes hold no recognisable face and
registering nothing is a silent failure.

### `plot/`

Sufficiently large to have its own [`plot/CLAUDE.md`](plot/CLAUDE.md). It holds the AST types and the registries for geoms, scale types, transforms, positions, and coords.

### `doc/`

Just `API.md` — the public Rust API reference for `Reader::execute`, `Writer::render`, `validate`, `ResolvedPlot`, `Validated`, `Metadata`. End-user docs live in `/doc/`, not here.

## Public API quick reference

Two-stage pipeline:

1. **`reader.execute(query)`** → `ResolvedPlot` (parses, runs SQL, resolves mappings, applies stats).
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
| `spatial` | ✓ | Spatial/geometry support (geozero for WKT↔GeoJSON) |
| `vegalite` | ✓ | Vega-Lite writer |
| `html` | ✓ | `HtmlWriter` — renders a `TABULATE` query as a bare `<table>` |
| `graphics` | — | *Internal.* The shared plot-composition layer; no GPU |
| `raster` | — | *Internal.* `graphics` + the GPU rasteriser (wgpu/vello-hybrid) |
| `png` | — | PNG writer (`raster`; genuinely 1.88+, excluded from the MSRV check) |
| `jpeg` | — | JPEG writer (`raster`) |
| `tiff` | — | TIFF writer (`raster`) |
| `webp` | — | WebP writer (`raster`) |
| `svg` | ✓ | SVG writer (`graphics`; no GPU, MSRV-clean) |
| `pdf` | ✓ | PDF writer (`graphics`; no GPU, MSRV-clean) |
| `hep` | ✓ | `.hep` plot-document writer (`graphics`; no GPU, MSRV-clean) |
| `hep-read` | — | **Test-only.** Reading a `.hep` back, for the round-trip test |
| `window` | — | `PlotViewer` — a native plot window (`raster`; not a writer) |
| `webfonts` | — | `fonts::register_font` also accepts WOFF / WOFF2 (`graphics`) |
| `builtin-data` | ✓ | Bundled penguins/airquality datasets |
| `all-readers` | — | `duckdb` + `sqlite` + `odbc` |
| `all-writers` | — | every writer above except the test-only `hep-read` |

`ggsql-wasm` builds with `default-features = false` plus `svg`, `webfonts`, `sqlite`, `builtin-data`, `spatial` — it draws plots in the browser with `SvgWriter`, which needs no GPU adapter, and `webfonts` is what lets a page hand it the WOFF/WOFF2 a font CDN serves. `ggsql-jupyter` builds with `duckdb`, `svg`, `pdf` plus ggsql's defaults, and its own default `raster-plots` feature adds `png`, `jpeg` and `tiff`; `svg` and `pdf` are non-optional there because the no-adapter fallback has to be compiled in whatever else is.

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
