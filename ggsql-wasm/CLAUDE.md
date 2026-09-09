# `ggsql-wasm/` — WebAssembly bindings

Compiles the `ggsql` core to WebAssembly for browsers. It powers the playground
at [`/doc/wasm/`](../doc/wasm/) and is published as the `ggsql-wasm` npm
package. Plots use ggsql's own SVG renderer; there is no JavaScript charting
library.

End-user playground: <https://ggsql.org/wasm/>. This file describes the build.

## Layout

```
ggsql-wasm/
├── Cargo.toml            cdylib; ggsql with svg, webfonts, sqlite, builtin-data and spatial
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
  `wasm-tools`; use `brew install binaryen` or `cargo install wasm-opt`.
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

Flags:

- `--skip-binary` reuses `pkg/dist/ggsql_wasm_bg.wasm` and rebuilds the client
  and demo. Use it while iterating on TypeScript or demo code. It fails until a
  full build has produced the glue and wasm once.
- `--skip-opt` compiles the binary but skips the optimization pass.

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

Keep one file per weight and style. The shaper selects faces by those attributes
and does not understand CSS `unicode-range`; multiple subsets with the same
family name can select a subset that lacks the required glyphs.

`registerFontFromUrl(url, { genericFor })` supports a page's own typography.
The `webfonts` feature accepts WOFF and WOFF2 as well as sfnt formats. Font
registration is process-global and must happen before the first draw.

## Wasm-specific constraints

`Cargo.toml` contains wasm32-only dependency overrides:

- `getrandom` and `uuid` use their JavaScript randomness support.
- `sqlite-wasm-rs` provides SQLite in the browser.
- `tokio` disables its default host I/O features.

ODBC is unavailable in the browser. The SVG writer is used because it needs no
GPU or WebGL context and permits any number of plots on a page.

The CSV and Parquet converters live in TypeScript. The package's init wrapper
passes them to Rust using `setConverters`; constructing a context through the
package after `init()` is therefore required before registering those formats
or builtin datasets.

## Distribution

- **npm:** publish `pkg/`. `package.json` includes only `dist/`, while npm
  automatically includes `LICENSE` and the manifest. The committed package
  version is bumped in the release checklist.
- **GitHub Releases:** the wasm binary is also attached to releases.
- **Docs:** `build-wasm.sh` copies `demo/dist/` to
  [`/doc/wasm/`](../doc/wasm/), which Quarto serves. Both are generated.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/CLAUDE.md`](../doc/CLAUDE.md) — docs and playground embedding.
