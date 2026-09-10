# `ggsql-wasm/` — WebAssembly bindings

Compiles the `ggsql` core to WebAssembly for browsers. It powers the playground
at [`/doc/wasm/`](../doc/wasm/) and is published as the `ggsql-wasm` npm
package. Plots use ggsql's own SVG renderer; there is no JavaScript charting
library.

End-user playground: <https://ggsql.org/wasm/>. This file describes the build.

## Layout

```
ggsql-wasm/
├── Cargo.toml            cdylib; ggsql with default-features = false + svg, webfonts, sqlite, builtin-data, spatial
├── build-wasm.sh         cargo → wasm-bindgen → wasm-opt → package → demo → doc/wasm
├── src/lib.rs            wasm-bindgen API; receives JS converters during init
├── fonts/                Roboto faces served beside the wasm
├── pkg/                  checked-in source for the published npm package
│   ├── package.json      package manifest, including the release version
│   ├── build.mjs         bundles the TypeScript client and copies fonts
│   ├── tsconfig.json     emits declarations into dist/
│   ├── src/              client, converters and extension loader
│   └── dist/             generated publish contents (gitignored)
└── demo/                 browser demo and Quarto integration
    ├── package.json      consumes file:../pkg
    ├── build.mjs
    └── src/
```

`pkg/` is the npm package, not generated wasm-pack output. Its source, manifest,
license and lock file are committed. Only `pkg/dist/`, `demo/dist/` and
`doc/wasm/` are generated and ignored.

The package entry point is `pkg/src/ggsql.ts`. It wraps wasm-bindgen's `init` and
`initSync`, supplies the CSV and Parquet converters to Rust through
`setConverters`, wires the extension loader, re-exports the generated API, and
adds `PlotView` and font loading. wasm-bindgen writes its glue directly into
`pkg/dist/`, next to the client bundle and generated declarations.

The re-export list is written out rather than `export *`, so that
`setConverters` stays internal: it is the hook `init` uses to hand the
converters to Rust, and a caller who replaced them would break every registered
format. Adding a binding to `lib.rs` therefore means adding it to that list too.

## Toolchain

- **Rust stable, not the workspace 1.86 MSRV.** The nested
  [`rust-toolchain.toml`](rust-toolchain.toml) selects stable for builds run from
  this directory. The wasm crate has no `rust-version`.
- Rust target `wasm32-unknown-unknown`.
- `wasm-bindgen-cli` with exactly the version recorded for `wasm-bindgen` in
  `/Cargo.lock`. The build checks this because the CLI and crate schema versions
  must agree. Install the required version with:

  ```sh
  cargo install -f wasm-bindgen-cli --version <version>
  ```

- A clang/LLVM with wasm backend support. The build script verifies it with a
  one-line compile probe.
- `wasm-opt` from binaryen for the `-Oz` optimization step. This is not
  `wasm-tools`, which is a different project and has no equivalent; use
  `brew install binaryen` or `cargo install wasm-opt`. The optimizer is passed
  the wasm features rustc emits by name rather than `--all-features`: that flag
  turns on everything binaryen knows, and binaryen 132 emits compact imports
  under it, which browsers refuse to compile ("Invalid import kind 127").
- Node.js for `pkg/` and `demo/`.

On macOS, the system clang may lack the wasm backend. A Homebrew LLVM setup is:

```sh
export PATH=/opt/homebrew/opt/llvm/bin:$PATH
export CC=/opt/homebrew/opt/llvm/bin/clang
export AR=/opt/homebrew/opt/llvm/bin/llvm-ar
```

## Build

The full build is:

```sh
cd ggsql-wasm
./build-wasm.sh
```

It runs these phases in order:

1. `cargo build --target wasm32-unknown-unknown --profile wasm -p ggsql-wasm`.
2. `wasm-bindgen --target web --keep-lld-exports` into `pkg/dist/`.
   `--keep-lld-exports` preserves symbols needed by loadable SQLite extensions.
3. `wasm-opt -Oz` on `pkg/dist/ggsql_wasm_bg.wasm`.
4. `npm install && npm run build` in `pkg/`. esbuild bundles the TypeScript
   client and hyparquet, TypeScript emits declarations, and the fonts are copied
   into `dist/fonts/`.
5. Downloads or reuses `mod_spatialite.wasm` under
   `/target/wasm-extensions/`.
6. `npm install && npm run build` in `demo/`, then copies SpatiaLite into
   `demo/dist/`.
7. Copies `demo/dist/` to `/doc/wasm/`.

Phase 4 also stamps `pkg/package.json`'s version from the crate's, so the npm
version cannot drift from the Rust one; `cargo pkgid` reads it, and the `npm
install` that follows carries it into `pkg/package-lock.json`.

Flags:

- `--skip-binary` reuses `pkg/dist/ggsql_wasm_bg.wasm` and rebuilds the client
  and demo (phases 4–7). Use it while iterating on TypeScript or demo code. It
  fails until a full build has produced the glue and wasm once.
- `--skip-opt` compiles the binary but skips the optimization pass.

Each prerequisite is checked only when the step that needs it will run, which is
what makes those flags worth having: `--skip-binary` needs neither clang,
`wasm-bindgen` nor `wasm-opt`, and `--skip-opt` does not need `wasm-opt`. CI's
non-release wasm job takes `--skip-opt` and installs no binaryen at all.

`pkg/tsconfig.json` uses `rootDirs: ["./src", "./dist"]` so the client import
of `./ggsql_wasm.js` resolves to wasm-bindgen's declaration in `dist/`.
Consequently `npm run typecheck` in `pkg/` requires a prior wasm-bindgen run;
the glue is deliberately not hand-maintained.

## Fonts are the thing that surprises people

**A browser enumerates no system fonts.** The font collection starts empty,
`sans-serif` resolves to nothing, and a plot is drawn with no text and incorrect
layout measurements.

`registerDefaultFonts()` fetches the four Roboto faces in `fonts/`, registers
them with the Rust shaper, points `sans-serif` at the family names returned by
the font files, and declares matching browser `@font-face` rules. `PlotView`
then names the registered family on the SVG root so browser rendering uses the
same face that ggsql measured.

Three details are load-bearing:

- **`registerFont` returns family names, and they matter.** A generic is an
  indirection through the font context rather than a name, so registering Roboto
  does not on its own make `sans-serif` mean Roboto — `setGenericFamily` does,
  and it takes names. The only place a family's name exists is inside the file;
  guessing it from the filename resolves to nothing at shaping time.
- **One file per weight and style.** The shaper selects faces by weight, width
  and style and does not understand CSS `unicode-range`; multiple subsets
  sharing a family name can let one without basic Latin win the attribute match,
  turning every tick label to tofu while the bold title still renders.
- **The browser has to resolve the same face the shaper measured.** Every run is
  placed with one anchor plus `textLength`, so the browser fits whatever it
  resolves into the width the shaper measured — a different face gets
  horizontally scaled to fit, which reads as plausible and is wrong differently
  on every platform. Hence both the `@font-face` rules and the family named on
  the SVG root.

`registerFontFromUrl(url, { genericFor })` supports a page's own typography.
The `webfonts` feature on `ggsql` is what lets `registerFont` take the WOFF and
WOFF2 containers a font CDN serves as well as sfnt bytes, unwrapping them before
the shaper sees them; it costs about 55 kB brotli, and is worth it because WOFF2
is how a font arrives at a web page. Without the feature a container is refused
*by name* rather than reaching the shaper and silently registering nothing.
Font registration is process-global, permanent, and must happen before the first
draw.

## Wasm-specific constraints

`Cargo.toml` contains wasm32-only dependency overrides:

- `getrandom` and `uuid` use their JavaScript randomness support.
- `sqlite-wasm-rs` provides SQLite in the browser.
- `tokio` disables its default host I/O features.

ODBC is unavailable in the browser.

The `svg` writer rather than a rasterising one: it needs no GPU adapter and no
WebGL2 context, and it pulls no `vello_hybrid`, `vello_common`, `glifo` or
`naga` — 25 fewer crates than the canvas path. It also puts no ceiling on how
many plots a page can show, which a docs page carrying several would otherwise
run into. The cost is one DOM element per mark, so a very dense plot is heavier
here than it would be on a canvas.

The CSV and Parquet converters live in TypeScript, and the package's `init`
hands them to Rust with `setConverters`. So a page has to enter through the
package: a context built on the glue's own `init` has no converters, and
`register_csv`, `register_parquet` and `register_builtin_datasets` fail with
"CSV and Parquet converters are not configured".

## Distribution

- **npm:** publish `pkg/`. `package.json` includes only `dist/`, while npm
  automatically includes `LICENSE` and the manifest. Its version is stamped from
  the crate's by `build-wasm.sh`, so there is nothing to bump by hand.
- **GitHub Releases:** the wasm binary is also attached to releases.
- **Docs:** `build-wasm.sh` copies `demo/dist/` to
  [`/doc/wasm/`](../doc/wasm/), which Quarto serves. Both are generated.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/CLAUDE.md`](../doc/CLAUDE.md) — docs and playground embedding.
