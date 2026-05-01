# PR1 spike: adbc_driver_duckdb viability

Date: 2026-05-01
Outcome: FAIL
Entrypoint that worked (if SUCCESS): n/a
libduckdb path used: `target/debug/build/libduckdb-sys-a628876b20a388b6/out/libduckdb.a`
adbc_driver_manager version: 0.23

## Summary

The `duckdb` crate (with its `bundled` feature, as configured in this
workspace's `Cargo.toml`) does not produce a shared library that
`adbc_driver_manager::ManagedDriver::load_dynamic_from_filename` can
`dlopen`. After `cargo build --package ggsql --features duckdb`, the
only `libduckdb*` artifacts under `target/debug/` are static archives
(`libduckdb.a`) plus rlibs / object files; there is no
`libduckdb.dylib` (or `.so` / `.dll`) anywhere in the build tree, and
no system libduckdb is installed on this machine.

The `libduckdb.a` static archive *does* contain ADBC symbols, including
the canonical entrypoint `_duckdb_adbc_init`, confirmed via `nm -gU`
(see "Symbol evidence" below). DuckDB does ship the C-ABI ADBC
implementation as part of its source. But `dlopen` cannot load a
static archive, so `ManagedDriver` cannot bridge to it — even though
the symbols are linked into the test binary itself.

## Test result

The spike test points `LIBDUCKDB_PATH` at the static archive that
`duckdb-sys` produces and tries each known entrypoint name. All three
attempts fail at the `dlopen` stage with the same error:

```
attempt with entrypoint='duckdb_adbc_init' failed: Internal: Error with
  dynamic library: dlopen(.../libduckdb.a, 0x0085): tried:
  '.../libduckdb.a' (slice is not valid mach-o file), ...
attempt with entrypoint='AdbcDriverDuckDBInit' failed: <same dlopen error>
attempt with entrypoint='AdbcDriverInit' failed: <same dlopen error>

Could not load .../libduckdb.a as an ADBC driver via any known
entrypoint. Last error: Internal: Error with dynamic library:
dlopen(.../libduckdb.a, 0x0085): tried: '.../libduckdb.a' (slice is not
valid mach-o file) ...
```

The failure mode is platform-level (Mach-O loader refusing a static
archive), not an ADBC-init-level failure, so trying additional
entrypoint names would not change the outcome.

## Symbol evidence

`nm -gU` on `libduckdb.a` shows the ADBC entrypoint and full ADBC API
surface are compiled in:

```
0000000000000000 T _duckdb_adbc_init
000000000000c8d4 T _AdbcConnectionInit
0000000000009c20 T _AdbcDatabaseInit
000000000000a488 T _AdbcLoadDriverFromInitFunc
... (full Adbc{Database,Connection,Statement}* API)
```

So if a `libduckdb.dylib` were available with the same symbols, the
canonical `duckdb_adbc_init` entrypoint should work.

## Why we end up with `.a` only

`Cargo.toml` declares:

```toml
duckdb = { version = "~1.4", features = ["bundled", "vtab-arrow"] }
```

The `bundled` feature on `duckdb-rs` tells `libduckdb-sys` to compile
DuckDB from vendored C++ sources via `cc` and emit a static library to
link directly into downstream rlibs/binaries. There is no
`build.rs`-level switch in `libduckdb-sys` that flips this to a
`cdylib`.

To get a shared `libduckdb.dylib` we would have to either:
1. Drop `bundled` and link against a system-installed libduckdb
   (regression in build hermeticity, adds a non-Rust install
   prerequisite for every contributor).
2. Add a separate build step that produces a shared library — either
   by post-processing `libduckdb.a` into a dylib, or by depending on
   the upstream prebuilt `libduckdb` release artifact, only for
   testing.

Neither is appropriate for PR1.

## Decision

FAIL → proceed to Task 7B: skip the cross-driver equivalence tests
in PR1; the existing `adbc_datafusion`-backed `#[ignore]` test in
`src/reader/adbc.rs` remains the only end-to-end ADBC coverage, which
exercises the `AdbcReader<D: Driver>` generic against a real driver
without needing `ManagedDriver`. A follow-up could add an equivalence
suite either by (a) bundling a `libduckdb.dylib` for tests via a
separate dev-only build step, or (b) standing up a SQLite-based
equivalence (the upstream `adbc_driver_sqlite` ships as a real
shared library) that compares ADBC-routed results against
`rusqlite`-routed results.
