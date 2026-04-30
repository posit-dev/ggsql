## Dependency audit recommendations

Ordered by effort-to-impact ratio (easiest wins first).

- **Remove `proptest`** (dev-dep in `src/Cargo.toml`). Never imported anywhere — no `proptest!` macros or imports exist. Drops 31 transitive deps (5 unique). One-line removal.

- **Remove `csv`** (dep in `ggsql-wasm/Cargo.toml`). Never imported — CSV parsing is handled on the JS side via the `convert_csv()` bridge. Drops 6 transitive deps (1 unique). One-line removal.

- **Remove `tokio-test`** (dev-dep in `ggsql-jupyter/Cargo.toml`). Never imported — the Jupyter kernel's tests are Python-based. Drops 21 transitive deps (1 unique). One-line removal.

- **Disable `prompt` feature on `odbc-api`**. Change to `default-features = false, features = ["odbc_version_3_80"]`. No prompt/driver-connect calls exist anywhere. Drops ~55 unique deps including the entire `winit` windowing library and its platform backends (`objc2-*`, `ndk`, `jni`, `xkbcommon-dl`, etc.).

- **Disable `resolve-http` on `jsonschema`** (dev-dep in `src/Cargo.toml`). Change to `default-features = false, features = ["resolve-file"]`. The Vega-Lite schema is vendored via `include_str!`, so HTTP resolution is never used. Drops ~30 unique deps (`reqwest`, `rustls`, TLS platform crates).

- **Disable `named_from_str` on `palette`**. Change to `default-features = false, features = ["std", "approx"]`. Named color lookup is done by `csscolorparser`, not `palette` — only color space math (`Srgb`, `LinSrgb`, `Oklab`, `Mix`) is used. Drops 4 unique deps (the `phf` v0.11 family).

- **Remove `functions` and `window` features from `rusqlite`**. No `create_scalar_function` or `create_window_function` calls exist. Zero dep savings but a cleaner feature surface.

- **Remove `ipc` feature from `arrow`** (and the empty `ipc = []` feature flag in `src/Cargo.toml`). No `arrow::ipc` imports exist anywhere. Marginal savings — `flatbuffers` is still pulled in by `duckdb`.

- **Remove `postgres` and `plotters`** placeholder deps. Both are optional deps behind feature flags with no implementation. No compiled impact today, but removing them avoids confusion and keeps the manifest honest until drivers are actually written.
