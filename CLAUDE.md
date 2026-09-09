# ggsql

A SQL extension for declarative data visualization based on the Grammar of Graphics. Queries combine SQL data retrieval with a visualization spec in one composable syntax:

```ggsql
SELECT date, revenue, region FROM sales WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
LABEL title => 'Sales by Region'
```

The user-facing site is at <https://ggsql.org>. The README at [`README.md`](README.md) is the public introduction.

## Authoritative docs

**Anything about ggsql syntax or semantics belongs in [`doc/`](doc/), not in any CLAUDE.md file.** That includes clause behaviour (`VISUALISE`, `DRAW`, `PLACE`, `SCALE`, `FACET`, `PROJECT`, `LABEL`), layer types, scales, aesthetics, and coordinate systems. CLAUDE.md files describe the implementation around those features — they should link to `doc/syntax/` rather than restate.

**Writing ggsql queries:** when you need to author or modify a ggsql query, use the vendored skill at [`doc/vendor/SKILL.md`](doc/vendor/SKILL.md). It is the source of truth for the syntax Claude should produce; do not invent clauses, settings, aesthetics, or layer types beyond what it documents.

## Workspace layout

| Folder | Role | Type | Per-folder CLAUDE.md |
| --- | --- | --- | --- |
| [`src/`](src/) | Core Rust library (crate `ggsql`) | Cargo workspace member | [`src/CLAUDE.md`](src/CLAUDE.md) |
| [`ggsql-cli/`](ggsql-cli/) | `ggsql` command-line binary | Cargo workspace member | [`ggsql-cli/CLAUDE.md`](ggsql-cli/CLAUDE.md) |
| [`tree-sitter-ggsql/`](tree-sitter-ggsql/) | Tree-sitter grammar + multi-language bindings | Cargo workspace member (also npm + PyPI) | [`tree-sitter-ggsql/CLAUDE.md`](tree-sitter-ggsql/CLAUDE.md) |
| [`ggsql-jupyter/`](ggsql-jupyter/) | Jupyter kernel | Cargo workspace member (also PyPI via maturin) | [`ggsql-jupyter/CLAUDE.md`](ggsql-jupyter/CLAUDE.md) |
| [`ggsql-wasm/`](ggsql-wasm/) | WebAssembly bindings + browser playground | Cargo workspace member | [`ggsql-wasm/CLAUDE.md`](ggsql-wasm/CLAUDE.md) |
| [`ggsql-vscode/`](ggsql-vscode/) | VS Code / Positron extension | Standalone TypeScript / npm | [`ggsql-vscode/CLAUDE.md`](ggsql-vscode/CLAUDE.md) |
| [`doc/`](doc/) | Quarto documentation site (ggsql.org) | Quarto project | [`doc/CLAUDE.md`](doc/CLAUDE.md) |

The Cargo workspace (`/Cargo.toml`) has five members: `tree-sitter-ggsql`, `src`, `ggsql-cli`, `ggsql-jupyter`, `ggsql-wasm`. Default workspace members exclude `ggsql-wasm` (it needs the wasm32 target and is built separately).

## High-level pipeline

```
ggsql query  ──►  parser  ──►  Plot AST  ──►  executor  ──►  Spec  ──►  writer  ──►  output
                  (tree-sitter)              (Reader runs SQL,            (Vega-Lite JSON
                                              applies stats,               or PNG)
                                              resolves scales)
```

- The parser splits the query at the `VISUALISE` boundary. SQL goes to a pluggable `Reader` (DuckDB, SQLite, ODBC); the VISUALISE part becomes a typed `Plot`.
- The executor ties the two together: SQL → DataFrame, AST resolved against actual schema, stats and scales applied per layer.
- The writer renders the resolved `Spec` to an output format: Vega-Lite JSON (the default writer), SVG, PDF or a `.hep` plot document (all default features), or one of the four raster formats — PNG, JPEG, TIFF, WebP (non-default; they need a GPU adapter). Everything but Vega-Lite is implemented on top of the hephaestus renderer — a name that stays internal; users see the format names.

For details — module layout, traits, where extension points live — see [`src/CLAUDE.md`](src/CLAUDE.md). For a specific renderer, [`src/writer/vegalite/CLAUDE.md`](src/writer/vegalite/CLAUDE.md) (Vega-Lite) or [`src/writer/hephaestus/CLAUDE.md`](src/writer/hephaestus/CLAUDE.md) (the other seven). For the AST types, [`src/plot/CLAUDE.md`](src/plot/CLAUDE.md).

## Building

**Prerequisite: `tree-sitter-cli`.** Any Rust build regenerates the parser from `grammar.js` via `tree-sitter-ggsql`'s build script, which runs `tree-sitter generate` and **fails if `tree-sitter-cli` is not on `PATH`**. Install it once with `npm install -g tree-sitter-cli`. To build against the committed `tree-sitter-ggsql/src/parser.c` without the CLI (e.g. if you're not touching the grammar), set `GGSQL_SKIP_GENERATE=1`.

```sh
# Rust workspace (default members: tree-sitter-ggsql, src, ggsql-cli, ggsql-jupyter)
cargo build --workspace
cargo build --release --workspace

# Just the library
cargo build --package ggsql

# Just the CLI binary
cargo build --package ggsql-cli

# Wasm build (separate, not in default workspace members)
cd ggsql-wasm && ./build-wasm.sh

# VS Code extension
cd ggsql-vscode && npm install && npm run package

# Tree-sitter parser (regenerate after editing grammar.js)
cd tree-sitter-ggsql && npx tree-sitter generate
```

### Rust version (MSRV)

The MSRV is **Rust 1.86**, declared as `rust-version` in `/Cargo.toml`. This is the maximum Rust version CRAN ships, and the R package's vendored copy of ggsql has to build against it — **only bump it when CRAN does.** `rust-version` also points clippy's MSRV-aware lints at 1.86, so an accidental 1.87+ std API is flagged as a lint rather than surfacing as a cryptic `E0658` at vendoring time. Keeping that guard is the main reason not to raise the declaration to match a dependency's.

**`parley` declares 1.88 while compiling fine on 1.86.** It is non-optional for the default `svg`/`pdf`/`hep` writers, so cargo's floor check refuses on 1.86 until `--ignore-rust-version` bypasses the *declaration*. That flag is not papering over a real incompatibility — the library genuinely compiles, which CI proves:

```sh
# The CRAN-toolchain claim, checked rather than asserted. Library only:
# --all-targets pulls the `adbc` dev-dependency path, and datafusion uses
# let-chains, so 1.86 fails there for real.
cargo +1.86 check --ignore-rust-version -p ggsql
```

There is deliberately **no root `rust-toolchain.toml`**: pinning to 1.86 would force local `cargo test` and rust-analyzer onto a toolchain where the `adbc` test path can't build. CI's default toolchain is stable — which still guards the MSRV, because clippy reads `rust-version`, not the toolchain — with the 1.86 check as its own step.

Two things are exempt from the 1.86 MSRV:

- **The `adbc` test path.** The experimental `adbc` feature depends (dev-only) on `adbc_datafusion` → `datafusion` ≥53.1.0, which uses let-chains and so requires rustc ≥1.88 for real. The shipped library still builds on 1.86 (it uses only `adbc_core`); only the test / `--all-targets` build pulls `datafusion`.
- **The wasm bindings (`ggsql-wasm`).** R doesn't use wasm, and some wasm-only dependencies require a newer rustc, so the crate has no `rust-version` and a nested `ggsql-wasm/rust-toolchain.toml` selects **stable** for builds run from that directory (`./build-wasm.sh`, Cargo, or `pkg/`).

### Rendering plots on Linux

`svg`, `pdf` and `hep` are **default features** and need no GPU adapter, no wgpu and — importantly — **no `libfontconfig1-dev`**: `src/Cargo.toml` names `fontique` directly to enable `fontconfig-dlopen`, so fontconfig is loaded at run time rather than linked at build time. That is what makes them safe to have on by default; see [`src/writer/hephaestus/CLAUDE.md`](src/writer/hephaestus/CLAUDE.md). The four raster writers (`png`, `jpeg`, `tiff`, `webp`) and `ggsql view` are **not** default and need an adapter at run time.

Cross-platform installers (NSIS / MSI / DMG / Deb): see [`INSTALLERS.md`](INSTALLERS.md). Releases are tag-driven via `.github/workflows/`.

## Testing

```sh
# Whole Rust workspace
cargo test --workspace

# A single crate
cargo test --package ggsql
cargo test --package ggsql-jupyter

# Tree-sitter corpus
cd tree-sitter-ggsql && npm test

# Jupyter kernel protocol tests (Python)
cd ggsql-jupyter/tests && pip install -r requirements.txt && pytest
```

Per-folder CLAUDE.md files cover component-specific test guidance.

## Coding style

- **Reuse existing infrastructure and architectural choices.** When adding new code, prefer extending or adapting what is already there over introducing a parallel implementation. If reuse requires changes elsewhere to accommodate the new caller, that is more palatable than implementing the same thing twice.
- **Comments describe the current state of the code.** Do not reference past states, how something used to work, what was changed, or why an earlier approach was abandoned — that history belongs in commit messages and [`CHANGELOG.md`](CHANGELOG.md).
- **[`CHANGELOG.md`](CHANGELOG.md) is the record of user-visible change over time.** Consult it when you need to know when something landed or how behaviour evolved. Update it when adding a feature, changing behaviour, or removing something — but write **one entry per feature**, added when the feature is complete. Don't gradually accrete bullets during development.
- **Always run `cargo fmt` and resolve all `cargo clippy` warnings at the end of any task that involves writing code.** Do this before considering the task complete.

## Where to ask which question

- *What does clause/layer/scale X do?* → [`doc/syntax/`](doc/syntax/).
- *How does the `ggsql` CLI work? Where do its subcommands live?* → [`ggsql-cli/CLAUDE.md`](ggsql-cli/CLAUDE.md).
- *How does the parser work? How is a `Plot` built?* → [`src/CLAUDE.md`](src/CLAUDE.md), then `src/parser/`.
- *How do I add a new geom / scale type / coord?* → [`src/plot/CLAUDE.md`](src/plot/CLAUDE.md).
- *How does Vega-Lite output get assembled?* → [`src/writer/vegalite/CLAUDE.md`](src/writer/vegalite/CLAUDE.md).
- *How do the raster, SVG, PDF and `.hep` writers work?* → [`src/writer/hephaestus/CLAUDE.md`](src/writer/hephaestus/CLAUDE.md), which also lists their known gaps.
- *How does `ggsql view` show a plot in a window?* → [`ggsql-cli/CLAUDE.md`](ggsql-cli/CLAUDE.md), then `PlotViewer` in [`src/writer/hephaestus/window.rs`](src/writer/hephaestus/window.rs).
- *How does a query become rendered output end-to-end?* → [`src/CLAUDE.md`](src/CLAUDE.md) (execution pipeline), then `src/execute/`.
- *How does the Jupyter kernel route messages?* → [`ggsql-jupyter/CLAUDE.md`](ggsql-jupyter/CLAUDE.md).
- *How does the VS Code / Positron extension talk to the kernel?* → [`ggsql-vscode/CLAUDE.md`](ggsql-vscode/CLAUDE.md).
- *How is the wasm playground built and embedded into the docs?* → [`ggsql-wasm/CLAUDE.md`](ggsql-wasm/CLAUDE.md) and [`doc/CLAUDE.md`](doc/CLAUDE.md).
- *How do I add new ggsql syntax?* → grammar in [`tree-sitter-ggsql/CLAUDE.md`](tree-sitter-ggsql/CLAUDE.md), then AST building in `src/parser/builder.rs` (covered in [`src/CLAUDE.md`](src/CLAUDE.md)), then docs in [`doc/CLAUDE.md`](doc/CLAUDE.md).
