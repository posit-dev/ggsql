# `ggsql-wasm/` — WebAssembly bindings

Compiles the `ggsql` core to WebAssembly so it can run in browsers. Used by the playground at [`/doc/wasm/`](../doc/wasm/) and published as an npm package. Workspace member.

Plots are drawn by the same renderer the rest of ggsql uses: the query is executed in wasm, `SvgWriter` draws it to SVG markup, and the page puts that markup in the document. There is no Vega-Lite here and no JavaScript charting library.

End-user playground: <https://ggsql.org/wasm/>. This file describes the *build*.

## Layout

```
ggsql-wasm/
├── Cargo.toml            cdylib; ggsql with default-features = false + svg, webfonts, sqlite, builtin-data, spatial
├── build-wasm.sh         End-to-end build orchestrator (library + wasm + demo → doc/wasm)
├── src/
│   └── lib.rs            wasm-bindgen entry points (the only Rust here)
├── js/                   Hand-written client: the package entry point
│   ├── ggsql.js          re-exports the glue, plus PlotView and font loading
│   └── ggsql.d.ts        hand-written types for it
├── fonts/                Roboto faces served beside the wasm (see Fonts)
├── library/              TypeScript wrapper distributed on npm
│   ├── package.json      npm package (build with `npm run build`)
│   ├── build.mjs         esbuild script
│   └── src/
├── demo/                 Browser demo + playground used by the doc site
│   ├── package.json
│   ├── build.mjs
│   └── src/              UI code (editor + SVG plot preview)
└── pkg/                  wasm-pack output, assembled by build-wasm.sh
    ├── ggsql_wasm_bg.wasm
    ├── mod_spatialite.wasm
    ├── ggsql_wasm.js, .d.ts   generated glue
    ├── ggsql.js, ggsql.d.ts   copied from js/; the package entry point
    ├── fonts/                 copied from fonts/
    └── package.json
```

**Nothing here is committed.** `/.gitignore` covers `pkg/`, `demo/dist/` and
`library/dist/`, and `doc/.gitignore` covers [`/doc/wasm/`](../doc/wasm/) — every
one of them is produced by `build-wasm.sh`, which CI runs before the docs are
rendered and published.

The package entry point is `ggsql.js`, the hand-written wrapper, not wasm-pack's
generated glue: `build-wasm.sh` points `main` / `module` / `exports` at it and it
re-exports everything the glue has. The split follows the renderer's own client —
wasm bytes are expensive and JavaScript is not — so the Rust side draws SVG and
stops, while `ResizeObserver`, font fetching and DOM insertion live in JS.

## Toolchain

- **Rust stable, not the workspace 1.86 MSRV.** The rest of the workspace targets the 1.86 MSRV for the R/CRAN bindings (see [`/CLAUDE.md`](../CLAUDE.md)), but R doesn't use wasm and some wasm-only deps need a newer rustc. A nested [`rust-toolchain.toml`](rust-toolchain.toml) selects stable for any build run from this directory; this crate has no `rust-version`. Every CI job installs `dtolnay/rust-toolchain@stable` and there is **no root `rust-toolchain.toml`**, so nothing here needs a `cargo +stable` — the 1.86 claim is enforced by its own check step in `build.yaml`, not by pinning the toolchain.
- Rust target `wasm32-unknown-unknown` and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) for compilation.
- A clang/llvm with wasm backend support (the build script verifies this with a one-line probe).
- `wasm-opt` (from binaryen) for the `-Oz` optimization step. **Not** `wasm-tools`,
  which is a different project and has no equivalent — `brew install binaryen`, or
  `cargo install wasm-opt` as CI does. The optimizer is passed the wasm features
  rustc emits by name rather than `--all-features`: that flag turns on everything
  binaryen knows, and binaryen 132 emits compact imports under it, which browsers
  refuse to compile ("Invalid import kind 127").
- Node.js for `library/` and `demo/`.

## Build

The full build:

```sh
cd ggsql-wasm
./build-wasm.sh
```

This sequentially:

1. `npm install && npm run build` in `library/` — produces the typed JS wrapper.
2. `wasm-pack build --target web --profile wasm --no-opt` — compiles `src/lib.rs` to `pkg/`. The `wasm` profile is defined in the workspace `Cargo.toml` (release-style, `opt-level = "z"`, LTO, `panic = "abort"`).
3. `wasm-opt pkg/ggsql_wasm_bg.wasm -o pkg/ggsql_wasm_bg.wasm -Oz` — shrinks the binary further.
4. Copies `js/ggsql.js`, `js/ggsql.d.ts` and `fonts/` into `pkg/`, and points the
   generated `package.json` at the wrapper (`main`, `module`, `exports`, `types`).
5. Downloads the prebuilt `mod_spatialite.wasm` from the [ggsql-dev/sqlite-wasm-rs releases](https://github.com/ggsql-dev/sqlite-wasm-rs/releases) into `pkg/`, caching it under `/target/wasm-extensions/`.
6. `npm install && npm run build` in `demo/` — bundles the playground UI (copies extension wasm from `pkg/` into `dist/`).
7. Copies `demo/dist/` to `/doc/wasm/` so Quarto can serve it under the docs site.

Flags:

- `--skip-binary` — reuse the existing `pkg/` wasm (skip steps 2–3); useful when iterating on `js/`, `library/` or `demo/`. **Step 4 still runs**, so an edit to the hand-written wrapper reaches the bundle without a recompile — that is the point of the flag. Fails if there is no `pkg/` to reuse.
- `--skip-opt` — compile but skip `wasm-opt` (faster, larger binary).

## Fonts are the thing that surprises people

**A browser enumerates no system fonts.** `fontique` falls back to a dummy
backend, so the collection starts empty, `sans-serif` resolves to nothing, and a
plot comes out with its chrome drawn and *no text at all* — no error, no warning.
Text is also what sets the layout, so a fontless plot has the wrong margins and
legend widths as well as no labels.

So the client registers fonts before drawing anything. `registerDefaultFonts()`
fetches the four Roboto faces in `fonts/`, hands them to `registerFont`, and
points `sans-serif` at the family names it gets back. Three details are
load-bearing:

- **`registerFont` returns family names, and they matter.** A generic is an
  indirection through the font context rather than a name, so registering Roboto
  does not on its own make `sans-serif` mean Roboto — `setGenericFamily` does,
  and it takes names. The only place a family's name exists is inside the file;
  guessing it from the filename resolves to nothing at shaping time.
- **One file per (weight, style).** The shaper selects within a family by weight,
  width and style and has no notion of CSS `unicode-range`. Register several
  subset files sharing a family name and one without basic Latin can win the
  attribute match — every tick label becomes tofu while the bold title still
  renders.
- **The browser has to resolve the same face the shaper measured.** Every run is
  placed with one anchor plus `textLength`, so the browser fits whatever it
  resolves into the width the shaper measured. The SVG names only the generic the
  theme asked for, which the browser would resolve to its own default — a
  different face, horizontally scaled to fit, which reads as plausible and is
  wrong, differently on every platform. `registerDefaultFonts` therefore also
  declares an `@font-face` for the same files, and `PlotView` names the
  registered family on the drawn SVG's root. `font-family` inherits, so a span
  that named its own — `code`, which must stay monospace — keeps it.

Registration is process-global and permanent: once per page, not once per plot.

`registerFont` takes sfnt bytes (TTF, OTF, TTC, OTC) **and** the WOFF and WOFF2
containers a font CDN serves, unwrapping them before the shaper sees them — that
is the `webfonts` feature on `ggsql`, which this crate enables, and it costs
about 55 kB brotli. It is worth that because WOFF2 is how a font arrives at a
web page, and it was otherwise the single likeliest input to fail. Without the
feature a container is refused *by name* rather than reaching the shaper and
silently registering nothing.

`registerFontFromUrl(url, { genericFor })` is the entry point for a page using
its own typography: it fetches, registers, and points the generic at the family
names that came back, since only the file knows what it is called.

The faces are fetched at run time from beside the wasm, not baked into it, so a
page supplying its own transfers none of them.

## Wasm-specific feature constraints

`Cargo.toml` carves out wasm32-only dependency overrides:

- `getrandom` and `uuid` are forced to the `js` feature so they get randomness from the browser.
- `sqlite-wasm-rs` replaces `rusqlite` for SQLite support in the browser.
- `tokio` is reduced to `default-features = false` (no I/O reactor on wasm).

ODBC is not enabled here — it requires host APIs that aren't available in the browser.

The `svg` writer rather than a rasterising one: it needs no GPU adapter and no
WebGL2 context, and it pulls no `vello_hybrid`, `vello_common`, `glifo` or `naga`
— 25 fewer crates than the canvas path. It also puts no ceiling on how many plots
a page can show, which a docs page carrying several would otherwise run into.
The cost is one DOM element per mark, so a very dense plot is heavier here than
it would be on a canvas.

## Distribution

- **npm**: `pkg/` is the published package, entered through `ggsql.js`. `library/`
  is a build-time dependency of the wasm rather than a separate publish: `lib.rs`
  imports it with `#[wasm_bindgen(module = "/library/dist/lib.js")]`, which is why
  it has to be built *before* `wasm-pack` runs.
- **GitHub Releases**: the wasm binary is also attached to releases (see commit `071cff6`).
- **Docs site**: `build-wasm.sh` copies `demo/dist/` to [`/doc/wasm/`](../doc/wasm/), which Quarto serves and embeds into every page via `_quarto.yml`. Generated, gitignored, and rebuilt by `publish.yaml`.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/CLAUDE.md`](../doc/CLAUDE.md) — how the playground gets embedded into the Quarto site.
