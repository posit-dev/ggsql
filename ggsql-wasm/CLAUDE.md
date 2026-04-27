# `ggsql-wasm/` — WebAssembly bindings

Compiles the `ggsql` core to WebAssembly so it can run in browsers. Used by the playground at [`/doc/wasm/`](../doc/wasm/) and published as an npm package. Workspace member.

End-user playground: <https://ggsql.org/wasm/>. This file describes the *build*.

## Layout

```
ggsql-wasm/
├── Cargo.toml            cdylib; ggsql with default-features = false + vegalite, sqlite, builtin-data
├── build-wasm.sh         End-to-end build orchestrator (library + wasm + demo → doc/wasm)
├── src/
│   └── lib.rs            wasm-bindgen entry points (the only Rust here)
├── library/              TypeScript wrapper distributed on npm
│   ├── package.json      npm package (build with `npm run build`)
│   ├── build.mjs         esbuild script
│   └── src/
├── demo/                 Browser demo + playground used by the doc site
│   ├── package.json
│   ├── build.mjs
│   └── src/              UI code (editor + Vega-Lite preview)
└── pkg/                  wasm-pack output (committed; consumed by library/ and demo/)
    ├── ggsql_wasm_bg.wasm
    ├── ggsql_wasm.js, .d.ts
    └── package.json
```

`pkg/` is generated but committed so contributors don't need a wasm toolchain just to run the docs.

## Toolchain

- Rust target `wasm32-unknown-unknown` and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) for compilation.
- A clang/llvm with wasm backend support (the build script verifies this with a one-line probe).
- `wasm-opt` (from binaryen) for the `-Oz` optimization step.
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
4. `npm install && npm run build` in `demo/` — bundles the playground UI.
5. Copies `demo/dist/` to `/doc/wasm/` so Quarto can serve it under the docs site.

Flags:

- `--skip-binary` — reuse the existing `pkg/` (skip steps 2–3); useful when iterating on `library/` or `demo/`.
- `--skip-opt` — compile but skip `wasm-opt` (faster, larger binary).

## Wasm-specific feature constraints

`Cargo.toml` carves out wasm32-only dependency overrides:

- `getrandom` and `uuid` are forced to the `js` feature so they get randomness from the browser.
- `sqlite-wasm-rs` replaces `rusqlite` for SQLite support in the browser.
- `tokio` is reduced to `default-features = false` (no I/O reactor on wasm).

ODBC is not enabled here — it requires host APIs that aren't available in the browser.

## Distribution

- **npm**: `library/` is published as the user-facing JS/TS wrapper. The `pkg/` artifact is bundled with it.
- **GitHub Releases**: the wasm binary is also attached to releases (see commit `071cff6`).
- **Docs site**: `demo/dist/` is committed into [`/doc/wasm/`](../doc/wasm/) by `build-wasm.sh` and embedded in Quarto pages via `_quarto.yml`.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/CLAUDE.md`](../doc/CLAUDE.md) — how the playground gets embedded into the Quarto site.
