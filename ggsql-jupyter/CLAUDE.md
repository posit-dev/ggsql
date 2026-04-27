# `ggsql-jupyter/` — Jupyter kernel

Standalone Rust binary that speaks the Jupyter messaging protocol over ZeroMQ. Embeds the `ggsql` library and renders results as Vega-Lite (visual queries) or HTML tables (pure SQL). Workspace member; published to crates.io and to PyPI (as a binary wheel via maturin).

End-user installation and usage live in [`README.md`](README.md). End-user notebook docs live in [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-jupyter/
├── Cargo.toml          Rust binary + library, depends on ggsql with duckdb + vegalite
├── pyproject.toml      maturin config (bindings = "bin") for the PyPI wheel
├── README.md           User-facing install + usage
├── src/
│   ├── main.rs         Binary entry: clap CLI (start kernel, --install)
│   ├── lib.rs          Library root (so internals can be unit-tested)
│   ├── kernel.rs       Jupyter messaging loop (ZMQ, message dispatch)
│   ├── executor.rs     Runs queries via ggsql::Reader, returns rendered output
│   ├── connection.rs   Reader lifecycle, connection-string parsing
│   ├── data_explorer.rs Positron data-explorer comm channel
│   ├── display.rs      Output formatting (Vega-Lite + vega-embed HTML, SQL → HTML table)
│   ├── message.rs      Jupyter message structs (ZMQ frames, HMAC signing)
│   └── util.rs
└── tests/
    ├── test_compliance.py   Jupyter protocol conformance
    ├── test_integration.py  End-to-end via jupyter_client
    ├── fixtures/            Sample notebooks / queries
    └── requirements.txt
```

## How it runs

1. `ggsql-jupyter --install` writes a kernelspec into the active Python environment (Jupyter, conda, uv, virtualenv — auto-detected).
2. `ggsql-jupyter <connection-file>` is the entry point Jupyter invokes; it reads the connection JSON, opens the five ZMQ sockets (shell, control, iopub, stdin, heartbeat), and runs `kernel.rs`'s message loop.
3. Each `execute_request` is dispatched through `executor.rs` → `ggsql::reader::DuckDBReader::execute(...)`. The kernel keeps a single persistent in-memory DuckDB session so cells share state.
4. The result is wrapped by `display.rs` into a Jupyter `display_data` message — Vega-Lite specs go through vega-embed in an HTML payload (works in classic Jupyter, JupyterLab, and Positron); pure SQL goes out as an HTML table.

## Positron-specific bits

- Kernel info advertises `"output_location": "plot"` so visualizations route to Positron's Plot pane.
- `data_explorer.rs` implements Positron's data-explorer comm channel (registered query results become explorable tables).
- The companion VS Code extension (`ggsql-vscode/`) discovers this binary via the `ggsql.kernelPath` setting, the active Jupyter kernelspec, or `PATH`.

## Build & install

```sh
# Dev: build the binary and register with the active env
cargo build --release --package ggsql-jupyter
./target/release/ggsql-jupyter --install

# Run a one-off install from crates.io
cargo install ggsql-jupyter
ggsql-jupyter --install

# PyPI distribution (built via maturin in CI; pyproject.toml is a wheel-builder shim)
pip install ggsql-jupyter && ggsql-jupyter --install
```

`pyproject.toml` declares `bindings = "bin"` — there is no Python module, the wheel just delivers the binary cross-platform.

## Features

```toml
default = ["all-readers"]
all-readers = ["sqlite", "odbc", "duckdb"]
```

Each feature passes through to `ggsql/<feature>`. The default install therefore supports DuckDB, SQLite, and ODBC connection strings.

## Testing

The Rust side has unit tests inline (`cargo test -p ggsql-jupyter`). The Jupyter protocol tests are Python:

```sh
cd ggsql-jupyter/tests
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`test_compliance.py` verifies handler coverage (`execute_request`, `kernel_info_request`, `is_complete_request`, `shutdown_request`); `test_integration.py` drives a real kernel via `jupyter_client`.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/ggsql-vscode/CLAUDE.md`](../ggsql-vscode/CLAUDE.md) — the VS Code / Positron extension that drives this kernel.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing notebook docs.
