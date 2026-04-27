# `ggsql-cli/` — `ggsql` command-line binary

Standalone Rust binary that wraps the `ggsql` library. Workspace member; published to crates.io as `ggsql-cli` and shipped as the `ggsql` executable in the cross-platform installers.

End-user installation lives in [`/doc/get_started/installation.qmd`](../doc/get_started/installation.qmd); CLI usage in [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-cli/
├── Cargo.toml          Binary def, depends on ggsql; holds [package.metadata.packager]
├── build.rs            Generates docs_data.rs by reading /doc/syntax/ + /doc/vendor/SKILL.md
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

Only public `ggsql::*` API is used (`reader`, `writer`, `validate`, `parser`, `VERSION`) — this crate has no awareness of internal modules.

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

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/INSTALLERS.md`](../INSTALLERS.md) — cross-platform installer build (Windows/Linux from this crate's packager metadata; macOS via `pkgbuild`).
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing CLI docs.
