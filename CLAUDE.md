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
| [`src/`](src/) | Core Rust library + `ggsql` CLI | Cargo workspace member | [`src/CLAUDE.md`](src/CLAUDE.md) |
| [`tree-sitter-ggsql/`](tree-sitter-ggsql/) | Tree-sitter grammar + multi-language bindings | Cargo workspace member (also npm + PyPI) | [`tree-sitter-ggsql/CLAUDE.md`](tree-sitter-ggsql/CLAUDE.md) |
| [`ggsql-jupyter/`](ggsql-jupyter/) | Jupyter kernel | Cargo workspace member (also PyPI via maturin) | [`ggsql-jupyter/CLAUDE.md`](ggsql-jupyter/CLAUDE.md) |
| [`ggsql-wasm/`](ggsql-wasm/) | WebAssembly bindings + browser playground | Cargo workspace member | [`ggsql-wasm/CLAUDE.md`](ggsql-wasm/CLAUDE.md) |
| [`ggsql-vscode/`](ggsql-vscode/) | VS Code / Positron extension | Standalone TypeScript / npm | [`ggsql-vscode/CLAUDE.md`](ggsql-vscode/CLAUDE.md) |
| [`doc/`](doc/) | Quarto documentation site (ggsql.org) | Quarto project | [`doc/CLAUDE.md`](doc/CLAUDE.md) |

The Cargo workspace (`/Cargo.toml`) has four members: `tree-sitter-ggsql`, `src`, `ggsql-jupyter`, `ggsql-wasm`. Default workspace members exclude `ggsql-wasm` (it needs the wasm32 target and is built separately).

## High-level pipeline

```
ggsql query  ──►  parser  ──►  Plot AST  ──►  executor  ──►  Spec  ──►  writer  ──►  output
                  (tree-sitter)              (Reader runs SQL,            (Vega-Lite JSON)
                                              applies stats,
                                              resolves scales)
```

- The parser splits the query at the `VISUALISE` boundary. SQL goes to a pluggable `Reader` (DuckDB, SQLite, ODBC); the VISUALISE part becomes a typed `Plot`.
- The executor ties the two together: SQL → DataFrame, AST resolved against actual schema, stats and scales applied per layer.
- The writer renders the resolved `Spec` to an output format (today: Vega-Lite JSON).

For details — module layout, traits, where extension points live — see [`src/CLAUDE.md`](src/CLAUDE.md). For the Vega-Lite renderer specifically, [`src/writer/vegalite/CLAUDE.md`](src/writer/vegalite/CLAUDE.md). For the AST types, [`src/plot/CLAUDE.md`](src/plot/CLAUDE.md).

## Building

```sh
# Rust workspace (default members: tree-sitter-ggsql, src, ggsql-jupyter)
cargo build --workspace
cargo build --release --workspace

# Just the CLI / library
cargo build --package ggsql

# Wasm build (separate, not in default workspace members)
cd ggsql-wasm && ./build-wasm.sh

# VS Code extension
cd ggsql-vscode && npm install && npm run package

# Tree-sitter parser (regenerate after editing grammar.js)
cd tree-sitter-ggsql && npx tree-sitter generate
```

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

## Where to ask which question

- *What does clause/layer/scale X do?* → [`doc/syntax/`](doc/syntax/).
- *How does the parser work? How is a `Plot` built?* → [`src/CLAUDE.md`](src/CLAUDE.md), then `src/parser/`.
- *How do I add a new geom / scale type / coord?* → [`src/plot/CLAUDE.md`](src/plot/CLAUDE.md).
- *How does Vega-Lite output get assembled?* → [`src/writer/vegalite/CLAUDE.md`](src/writer/vegalite/CLAUDE.md).
- *How does a query become rendered output end-to-end?* → [`src/CLAUDE.md`](src/CLAUDE.md) (execution pipeline), then `src/execute/`.
- *How does the Jupyter kernel route messages?* → [`ggsql-jupyter/CLAUDE.md`](ggsql-jupyter/CLAUDE.md).
- *How does the VS Code / Positron extension talk to the kernel?* → [`ggsql-vscode/CLAUDE.md`](ggsql-vscode/CLAUDE.md).
- *How is the wasm playground built and embedded into the docs?* → [`ggsql-wasm/CLAUDE.md`](ggsql-wasm/CLAUDE.md) and [`doc/CLAUDE.md`](doc/CLAUDE.md).
- *How do I add new ggsql syntax?* → grammar in [`tree-sitter-ggsql/CLAUDE.md`](tree-sitter-ggsql/CLAUDE.md), then AST building in `src/parser/builder.rs` (covered in [`src/CLAUDE.md`](src/CLAUDE.md)), then docs in [`doc/CLAUDE.md`](doc/CLAUDE.md).
