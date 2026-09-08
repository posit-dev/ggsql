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
    ├── main.rs         clap CLI: exec, run, parse, validate, docs, skill
    └── writers.rs      The writer registry — one row per writer, plus dispatch
```

The binary name is `ggsql` (not `ggsql-cli`) — that's what release artifacts and `$PATH` see.

`build.rs` finds `/doc/` via `CARGO_MANIFEST_DIR/..` (workspace root). It walks `/doc/syntax/*.qmd` to embed clause/layer/scale/aesthetic/coord docs as constants in `OUT_DIR/docs_data.rs`, and reads `/doc/vendor/SKILL.md` (with optional `GGSQL_UPDATE_SKILL=1` to refresh from GitHub) for the `skill` subcommand. The `docs` and `skill` commands therefore work offline once the binary is built.

**`doc/vendor/SKILL.md` is a *cache*, not a source.** `GGSQL_UPDATE_SKILL=1` overwrites it wholesale from [`posit-dev/skills`](https://github.com/posit-dev/skills/blob/main/ggsql/ggsql/SKILL.md), so an edit made here survives only until the next refresh. Anything that has to stick — a new writer, a changed setting, a corrected claim about a format — belongs in that repository; editing the cache is how the local build sees it in the meantime, and the upstream PR is what keeps it.

## Subcommands

| Command | Purpose |
| --- | --- |
| `exec` | Run a ggsql query string (default reader `duckdb://memory`, writer `vegalite`) |
| `run` | Like `exec`, but reads the query from a file |
| `view` | Show a query's plot in a native window; blocks until it closes (`window` feature) |
| `parse` | Print the parsed AST (formats: `pretty`, `debug`, `json`) — debugging aid |
| `validate` | Syntax + semantic check without executing SQL |
| `docs` | Render embedded ggsql syntax docs (TTY → ANSI via termimad, pipe → markdown, `--format json` → structured) |
| `skill` | Render the AI-assistant skill from `/doc/vendor/SKILL.md` |
| `agent-info` | Alias for `skill` |

The subcommand list does not change with features: `view` is always defined, and every writer is always a `--writer` name. What changes is whether it can do anything, and it says so.

Only public `ggsql::*` API is used (`reader`, `writer`, `validate`, `parser`, `VERSION`) — this crate has no awareness of internal modules.

`exec` and `run` share their flags through one `#[derive(Args)] RenderArgs` (`--reader`, `--cache`, `--writer`, `-D`, `--output`, `--verbose`) that both subcommands `#[command(flatten)]`, so a flag's help text and default exist once. `RenderArgs::writer()` resolves them into a `WriterSpec { info, options }` **in `main`, before any SQL runs** — an unknown `--writer`, a writer whose feature is off, and a `-D` pair that is not `key=value` all fail there rather than after the query has executed. `WriterSpec` then travels down `cmd_exec` → `exec_with_reader` → `render_spec`.

Which keys a writer accepts is the writer's business, and an unknown one is its error to report — so adding a setting needs no CLI change. User-facing keys are documented in [`/doc/get_started/tooling/cli.qmd`](../doc/get_started/tooling/cli.qmd).

### The writer registry

[`src/writers.rs`](src/writers.rs) holds one `WriterInfo` row per writer: its name and aliases, the filename extensions that imply it, the cargo feature that compiles it, the `label` used in messages ("PNG", "Vega-Lite JSON"), a `blurb` and an `options` line for help, `compiled: cfg!(feature = "…")`, and a `render` function pointer. Dispatch, `--writer`'s long help, `-D`'s long help and the "unknown writer" message are all generated from that list, so **adding a writer means adding a row and its render function** — nothing else in the CLI changes. Because `compiled` is a field rather than a `#[cfg]` around the row, the help and the error can name a writer this build lacks and say which feature would bring it in, which is the more common mistake than a misspelled name.

Render functions return `Result<(Output, Vec<String>), String>`: the output plus anything the writer had to degrade to produce it. They report failure rather than exiting, so `render_spec` owns how a problem is presented. **Warnings go to stderr unconditionally, not behind `-v`** — something the writer could not express is a defect in the file the user is about to ship, and stderr keeps it out of a piped artifact.

**`RenderArgs::resolve_writer` decides which writer runs**, and the order is deliberate:

1. An explicit `--writer` wins — it is what the user said. If it disagrees with `--output`'s extension the flag is still obeyed, with a note on stderr; writing SVG to a `.txt` to read it is legitimate, so this is a warning, not an error.
2. Otherwise `--output`'s extension picks one, via `writers::for_extension`. Longest extension first, so a two-part `vl.json` cannot be shadowed by its own `json` tail.
3. **An extension naming a writer this build lacks is an error**, with the feature named — the same message an explicit `--writer` gives. Falling back here would write Vega-Lite JSON into a file called `.png`, which is the mistake the feature exists to prevent.
4. An unrecognised extension, or no `--output`, falls back to `writers::DEFAULT_WRITER`.

This is why `RenderArgs::writer` is `Option<String>` with no clap `default_value`: "unset" has to be distinguishable from "explicitly vegalite", or step 2 could never fire. The default is stated in the long help instead.

`open_reader(uri, cache) -> Result<Box<dyn Reader + Send>, String>` is the matching single place for connection strings, and it delegates to the library factory `ggsql::reader::connection::reader_from_uri`. Which schemes exist, which of them this build has, and how a cache wraps a primary are the library's business; `ggsql::reader::Reader` is object-safe on purpose, so `exec`, `run` and `view` all go through this one function.

`--cache <duckdb|sqlite>` on `exec` and `run` wraps the reader in an in-memory caching layer, off by default. It is sugar for the composite connection scheme `<cache>+<primary>://…` (e.g. `duckdb+odbc://…`) that `reader_from_uri` already understands, so `open_reader` rewrites the flag into that URI and refuses the two forms together — there would be no saying which cache was meant. The composite scheme works on `view` too, which has no flag of its own.

### `view`, and why the window code is not here

`view` flattens its own `ViewArgs` rather than `RenderArgs`: there is no `--writer` to pick and no `--output` to write, and its `-D` (`--viewer-option`) carries the viewer's settings rather than a writer's.

**The window itself lives in the library, as `ggsql::writer::PlotViewer`** — and that is the decision most likely to be re-litigated, so: *only public `ggsql::*` API is used; this crate has no awareness of internal modules.* For the CLI to call the renderer's `window::run` itself it would have to take a direct hephaestus dependency, name `PlotComposition` and `WindowConfig` in its own source, and pin hephaestus in a second place — breaking that invariant three ways. So the *behaviour* goes public as a type instead, and `cmd_view` stays thin: parse options, open the reader, execute, call `show`. `show` blocks on the main thread until the window closes.

**The subcommand is defined unconditionally.** Without the `window` feature it prints what would bring it back. A subcommand that vanishes between builds is worse than one that explains itself — the same reasoning as `WriterInfo::compiled`.

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
default = ["duckdb", "sqlite", "vegalite", "parquet", "builtin-data", "odbc", "svg", "pdf", "hep"]
```

Each feature passes through to `ggsql/<feature>`. A writer feature gates only its own row's render function in `writers.rs`; the row itself is always present.

`svg`, `pdf` and `hep` are default because they cost nothing to have: no GPU adapter, no wgpu, and on Linux no `libfontconfig1-dev` at build time. `png`, `jpeg`, `tiff`, `webp` and `window` are not, since those do need an adapter at run time.

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
