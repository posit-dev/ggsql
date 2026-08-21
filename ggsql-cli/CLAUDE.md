# `ggsql-cli/` — `ggsql` command-line binary

Standalone Rust binary that wraps the `ggsql` library. Workspace member; published to crates.io as `ggsql-cli` and shipped as the `ggsql` executable in the cross-platform installers.

End-user installation lives in [`/doc/get_started/installation.qmd`](../doc/get_started/installation.qmd); CLI usage in [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-cli/
├── Cargo.toml          Binary def, depends on ggsql; holds [package.metadata.packager]
├── build.rs            Generates docs_data.rs by reading /doc/syntax/ + /doc/vendor/SKILL.md
├── examples/
│   └── visual_test.rs  Dev harness: renders the doc examples into an HTML report
└── src/
    └── main.rs         clap CLI: exec, run, parse, validate, docs, skill
```

The binary name is `ggsql` (not `ggsql-cli`) — that's what release artifacts and `$PATH` see.

`build.rs` finds `/doc/` via `CARGO_MANIFEST_DIR/..` (workspace root). It walks `/doc/syntax/*.qmd` to embed clause/layer/scale/aesthetic/coord docs as constants in `OUT_DIR/docs_data.rs`, and reads `/doc/vendor/SKILL.md` (with optional `GGSQL_UPDATE_SKILL=1` to refresh from GitHub) for the `skill` subcommand. The `docs` and `skill` commands therefore work offline once the binary is built.

## Subcommands

| Command | Purpose |
| --- | --- |
| `exec` | Run a ggsql query string (default reader `duckdb://memory`, writer `vegalite`) |
| `run` | Like `exec`, but reads the query from a file |
| `parse` | Print the parsed AST (formats: `pretty`, `debug`, `json`) — debugging aid |
| `validate` | Syntax + semantic check without executing SQL |
| `docs` | Render embedded ggsql syntax docs (TTY → ANSI via termimad, pipe → markdown, `--format json` → structured) |
| `skill` | Render the AI-assistant skill from `/doc/vendor/SKILL.md` |
| `agent-info` | Alias for `skill` |

Only public `ggsql::*` API is used (`reader`, `writer`, `validate`, `parser`, `VERSION`) — this crate has no awareness of internal modules.

`exec`/`run` build their reader via the library factory `ggsql::reader::connection::reader_from_uri`. They accept an in-memory caching layer (off by default) selected either by the composite connection scheme `<cache>+<primary>://…` (e.g. `duckdb+odbc://…`) or the `--cache <duckdb|sqlite>` flag; the two cannot be combined.

`exec` and `run` share a `WriterSpec { name, options }`: `--writer` names the writer and repeated `--writer-option key=value` flags (short `-D`, visible alias `--writer-options`, several settings per flag when separated by `;`) become a `ggsql::writer::WriterOptions`, parsed up front in `main` so a malformed pair fails before any SQL runs. The two travel together down `cmd_exec` → `exec_with_reader` → `render_spec`, which dispatches on the name and hands the options to `Writer::from_options`. Adding a setting to a writer therefore needs no CLI change; which keys exist is the writer's business, and an unknown one is its error to report. User-facing keys are documented in [`/doc/get_started/tooling/cli.qmd`](../doc/get_started/tooling/cli.qmd).

## Build & install

```sh
# Dev
cargo build --release --package ggsql-cli
./target/release/ggsql --version

# From crates.io
cargo install ggsql-cli

# Refresh the embedded skill at build time
GGSQL_UPDATE_SKILL=1 cargo build --package ggsql-cli
```

Cross-platform installers — see [`/INSTALLERS.md`](../INSTALLERS.md). Windows (NSIS / MSI) and Linux (Deb) installers are built via `cargo packager` from this crate's `[package.metadata.packager]`, with output in `ggsql-cli/target/release/packager/`. macOS `.pkg` installers are built directly with Apple's `pkgbuild` (the `[package.metadata.packager]` block is not consulted there). All three flows bundle both `ggsql` and `ggsql-jupyter` binaries.

The macOS codesign step uses [`/entitlements.plist`](../entitlements.plist) at the workspace root (shared with `ggsql-jupyter`).

## Features

```toml
default = ["duckdb", "sqlite", "vegalite", "ipc", "parquet", "builtin-data", "odbc"]
```

Each feature passes through to `ggsql/<feature>`. The `vegalite` flag also gates the writer-rendering path in `main.rs` via `#[cfg(feature = "vegalite")]`.

## Testing

```sh
cargo test --package ggsql-cli
```

Library-level coverage lives in `ggsql` itself — this crate is thin glue, so its own test suite is small. Smoke test the binary end-to-end:

```sh
./target/release/ggsql --version
./target/release/ggsql exec "SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point"
./target/release/ggsql docs draw
./target/release/ggsql skill
```

## The `visual_test` example

[`examples/visual_test.rs`](examples/visual_test.rs) is a **developer harness, not a shipped feature**: it treats every executable ```` ```{ggsql} ```` cell in [`/doc/`](../doc/) as a test corpus, renders each one, and writes a single HTML report pairing every query with its output. It lives here because this is the crate that already owns clap and the public `ggsql` API; it adds nothing to the binary.

```sh
cargo run -p ggsql-cli --features png --example visual_test              # doc/syntax + doc/gallery
cargo run -p ggsql-cli --features png --example visual_test -- --compare # + Vega-Lite side by side
cargo run -p ggsql-cli --features png --example visual_test -- doc/gallery -f pie
open target/visual-test/index.html
```

`[[example]]`'s `required-features` keeps it out of `cargo test --workspace`, so a build without a GPU stack never compiles it.

Four properties are worth preserving when changing it:

- **One reader per source file, cells in document order.** Doc pages build a table in one cell and plot it in the next, so per-cell isolation would break the corpus. A cell with no `VISUALISE` (`validate(..).has_visual()` is false) runs as setup through `execute_sql`.
- **Cells run in their own page's directory**, as Quarto runs them, so a query reading `FROM 'minard_troops.csv'` finds the CSV sitting beside the `.qmd`. The report and its `assets/` are resolved to an absolute path up front, since they outlive that switch.
- **Nothing aborts the run.** Execution errors, render errors and *panics* inside a writer are captured per cell (`capture`), so one report surfaces every problem in the corpus at once. This is the point of the tool — a run that stops at the first failure tells you almost nothing.
- **Renders are files, specs are inline.** PNGs are written to `assets/`; Vega-Lite specs are embedded in `<script type="application/json">` and mounted lazily, so the report works opened straight off disk (`fetch` would be blocked on `file://`) without paying for 200 charts up front.

The report itself is plain HTML with a client-side filter and an *only problems* toggle; the only network dependency is the vega CDN under `--compare`.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/writer/hephaestus/CLAUDE.md`](../src/writer/hephaestus/CLAUDE.md) — the PNG writer this harness is mostly used to check.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/INSTALLERS.md`](../INSTALLERS.md) — cross-platform installer build (Windows/Linux from this crate's packager metadata; macOS via `pkgbuild`).
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing CLI docs.
