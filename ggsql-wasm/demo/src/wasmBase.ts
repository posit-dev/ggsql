// Base URL for shared WASM assets (ggsql_wasm_bg.wasm, onig.wasm, etc.)
// Derived from import.meta.url so it resolves relative to the bundle,
// not the page that loads it. Works for both the playground (co-located)
// and quarto pages (loaded cross-directory via dynamic import).
export const WASM_BASE = new URL(".", import.meta.url).href;
