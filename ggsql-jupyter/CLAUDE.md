# `ggsql-jupyter/` — Jupyter kernel

Standalone Rust binary that speaks the Jupyter messaging protocol over ZeroMQ. Embeds the `ggsql` library and renders results as images (visual queries) or HTML tables (pure SQL). Workspace member; published to crates.io and to PyPI (as a binary wheel via maturin).

End-user installation and usage live in [`README.md`](README.md). End-user notebook docs live in [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-jupyter/
├── Cargo.toml          Rust binary + library, depends on ggsql with duckdb + svg/pdf + raster-plots
├── pyproject.toml      maturin config (bindings = "bin") for the PyPI wheel
├── README.md           User-facing install + usage
├── src/
│   ├── main.rs         Binary entry: clap CLI (start kernel, --install)
│   ├── lib.rs          Library root (so internals can be unit-tested)
│   ├── kernel.rs       Jupyter messaging loop (ZMQ, message dispatch)
│   ├── executor.rs     Runs queries via ggsql::Reader, returns rendered output
│   ├── connection.rs   Reader lifecycle, connection-string parsing
│   ├── data_explorer/  Positron data-explorer comm channel
│   │   ├── mod.rs      State, RPC dispatch, paging and profiles
│   │   ├── filter.rs   Row filters, sort keys, schema search, Convert to Code → SQL
│   │   └── export.rs   Copying a selection as CSV / TSV / HTML
│   ├── results.rs      Query results retained for the data explorer
│   ├── display.rs      Output formatting (image bundles, SQL → HTML table)
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
3. Each `execute_request` is dispatched through `executor.rs` → `Reader::execute(...)`. The kernel keeps a single persistent in-memory DuckDB session so cells share state. Readers are built by the library factory `ggsql::reader::connection::reader_from_uri`, so a composite `<cache>+<primary>://` connection string (via `-- @connect:`, say) wraps the reader in an in-memory caching layer — the persistent session is what makes that worth having, since later cells reuse cached remote reads. `-- @uncache` clears it mid-session.
4. The result is wrapped by `display.rs` into a Jupyter message. A plot is **rendered here, in the kernel** (see [Rendering](#rendering)); pure SQL goes out as an HTML table.

## Where output goes: `SessionKind`

`display.rs`'s `SessionKind` decides which of three output slots a result is aimed at, and it is the thing every rendering decision keys off:

| `SessionKind` | Slot |
| --- | --- |
| `PositronConsole` | Positron's Plots pane |
| `PositronNotebook` | The notebook cell |
| `Standalone` | A static document — Jupyter, Quarto, nbconvert |

**The console/notebook split is not cosmetic.** Positron routes a plot comm to the Plots pane whatever kind of session opened it, so a notebook session that used one would put its picture in the pane and leave its cell empty. The two need different output paths, and this is what tells them apart.

`SessionKind::resolve(session, mode)` prefers what the frontend *declared* over what its session id looks like:

- **`--session-mode console|notebook|background`** is authoritative. Only a frontend that creates the session can say, which in practice means the ggsql extension — `manager.ts`'s `createKernelSpec` appends it from `sessionMetadata.sessionMode`. The enum's values are already the flag's spelling, so nothing is translated. `background` maps to `Standalone`: it is Positron's session, but attached to no UI, so there is no Positron slot to render into.
- **The session-id heuristic** is the fallback: a `ggsql-` prefix means Positron (its supervisor tags every session it manages), and `notebook` in the id means a notebook. It exists for external Jupyter and Quarto, which pass no flag, and for extension versions predating it.

Two places deliberately **do not** pass the flag. `writeKernelJson` and `ggsql-jupyter --install` write kernelspecs for *external* frontends, which are exactly the ones that should classify as `Standalone`. And `restoreSession` doesn't rebuild the spec at all — the supervisor replays the argv the session was created with, and a session's mode never changes.

## Rendering

Plots are rendered in the kernel and travel as images. **Nothing fetches a renderer from a CDN**, so a plot works offline, in CI, and behind a firewall.

`plot/` holds the whole of it:

| File | Role |
| --- | --- |
| `plot/mod.rs` | `Format`, `RenderRequest`, and `choose` — the one function that decides what a plot becomes |
| `plot/backend.rs` | `PlotBackend`: the render thread, and the GPU probe |
| `plot/sizing.rs` | `Canvas`: logical size × device pixel ratio → device pixels + dpi |
| `plot/quarto.rs` | `QUARTO_FIG_*` → a format and a canvas |

### `choose`

Three things have to agree — where the output is going, what this build and machine can produce, and what the frontend asked for — so they are reconciled in one readable function rather than spread through the formatting code:

| `SessionKind` | Result |
| --- | --- |
| `PositronConsole` | A `positron.plot` comm, with **no `execute_result`** — whatever this build can render |
| `PositronNotebook` | A static image bundle in the cell |
| `Standalone` | What `QUARTO_FIG_FORMAT` asked for, else a static image |

**SVG is the fallback wherever raster output is unavailable** — no GPU adapter, or a build with `raster-plots` off. That is why `ggsql/svg` and `ggsql/pdf` are *non-optional* dependencies: the path that always works must always be compiled in, and it costs nothing to hold that line since the vector writers pull in no wgpu.

`raster-plots` itself **is** default, because the Plots pane asks for `png` and every shipped build enables it — a plain `cargo build` quietly producing SVG-only plots differed from the released kernel in the way hardest to notice from the outside. `--no-default-features --features all-readers` builds without wgpu. The fallback is a change of *format*, never of delivery: `Format::available` substitutes SVG for a raster request and `mime_type` reports what was produced, so a console keeps its comm either way.

### The render thread

`kernel.rs` awaits `handle_shell_message` **inline** in its `select!`, so anything blocking there stalls the heartbeat, the control channel and the SIGINT handler alike. So `PlotBackend` owns a thread, probes for an adapter once at startup (10 s ceiling), and keeps **one** renderer for the session, which handles a changing frame size internally.

The probe is eager rather than lazy because a lazy one would leave the *first* plot unable to choose a path.

### What the cold start actually costs

Measured on a release build, Apple GPU, vello-hybrid:

| | warm | cold (first process after a build) |
| --- | --- | --- |
| `RasterRenderer::new()` | 14 ms | ~185 ms |
| **first render** | **~85 ms** | **~1.35 s** |
| later renders, 3-point plot at 1200×800 | 5 ms | 5 ms |
| later renders, 50k points | ~200 ms | ~200 ms |
| a render at a size not seen before | 9–22 ms | — |

**Constructing the renderer is not the expensive part — the first render is**, and most of that is text: parley/fontique enumerating and loading system faces. Rendering an SVG first (no GPU, same text work) drops the first raster render from ~85 ms to ~20 ms, which is what identifies the cost. It is per *process*, not per renderer, so the SVG fallback pays it too.

That is why the thread renders a **throwaway 64×64 SVG frame at startup**, before anything is waiting on it: with the warm-up, the first plot of a session renders in ~14 ms rather than ~85 ms, and far better than that on a genuinely cold start. `backend::warm_up` builds its own in-memory database to do it — never the session's reader, since executing a query through that would materialise ggsql's internal views in the user's session.

### The plot comm

A console session gets one `positron.plot` comm per plot, modelled on `positron.dataExplorer` rather than the singleton connection comm — the pane shows plots as a history. `plot/comm.rs` holds the protocol (all pure, so it is tested without a Positron host); `kernel.rs` holds the transport.

Five things about it are load-bearing:

- **The comm alone creates the pane entry, so the kernel emits no `execute_result` alongside it.** Positron's `createActivityItemOutput` inlines *any* output whose data carries an `image/*` mime, consulting neither `output_location` nor the output kind, so a bundle sent as well would appear in the console *and* leave a second, fixed-size entry in the pane. It is also why a console with no raster writer keeps the comm and answers its renders in SVG instead of falling back to a bundle: there is no way to put a picture in the pane alone without one.
- **`comm_open` must follow `execute_input`** and be parented to the `execute_request`. Positron populates `_recentExecutions` from `execute_input`, and that is where the plot's `code` metadata comes from.
- **`render` is answered asynchronously; `get_metadata` is not.** A render goes to the thread and its reply comes back through the `select!` outcome arm, so the message loop stays free. `get_metadata` is answered from `plot_comms` because it gets Positron's default 5 s timeout where `render` gets 30 s — it must never queue behind a render.
- **`get_intrinsic_size` returns `null`.** ggsql has no figure-size syntax, so there is no intrinsic size, and `null` stops Positron offering an "Intrinsic" option that would be a lie. Sizing then follows whichever policy the Plots pane has selected — **`Auto` by default, which caps the aspect ratio at the golden ratio**, so a pane wider than 1.618:1 asks for less width than it has. The pane applies its policy *before* the kernel hears anything (`PlotSizingPolicyAuto.getPlotSize` feeds `setPlotsRenderSettings`, which arrives as `did_change_plots_render_settings`), so that shape is Positron's and not ours. `Fill` uses the whole viewport.
- **`show`/`update` are never sent, and are `MethodNotFound` inbound.** They mean "the backend mutated this figure, re-fetch it"; a ggsql `Spec` is immutable per execution, so re-running a cell opens a *new* comm, as the R and matplotlib backends do. An unknown method is an error rather than `result: null`, so a future Positron method fails visibly instead of being silently satisfied with garbage.

Plots are retained on the render thread and capped by `--max-plots` (default 32), evicted oldest-first with `comm_close` on iopub — which cleanly removes the plot from the pane, the right semantic for "the kernel no longer keeps that plot". Positron imposes no cap of its own, so this is where a long console session's memory is bounded.

### Why renders are asynchronous

Verified rather than assumed: with a large dense render in flight, a `kernel_info_request` issued immediately after it was answered in **1 ms**, while the render's own reply arrived much later. The message loop — and with it the heartbeat, the control channel and the interrupt handler — is genuinely free during a render.

That matters most for the interactive path, where dragging the Plots pane asks for a frame per event. Positron's `PositronPlotRenderQueue` already serialises renders per session and cancels superseded ones, so the kernel never sees overlapping renders for the same comm and needs no cancellation of its own.

### Pre-rendering, and the one notification we subscribe to

`manager.ts` sets `uiSubscriptions: ['did_change_plots_render_settings']`. Without it the frontend never tells the kernel how large the Plots pane is, and the kernel needs that for exactly one thing: rendering a **new** plot at the right size so `comm_open` can carry it as `pre_render`, and the pane shows it immediately instead of blank until its own render request lands. Every other render already carries its own size on the request.

Three details are easy to get wrong:

- **It is a notification, not a request.** It has no JSON-RPC `id`, and replying to one is a protocol error — so the ui branch checks `rpc_id.is_null()` before building a reply. It previously answered *everything* on that comm with `result: null`.
- **The first plot of a session carries no pre-render.** The pane has not reported yet, and rendering at a guessed size would show the wrong-sized picture and have it replaced the moment the pane asks properly. The flash is worse than the wait; the reference Python backend skips it for the same reason.
- **A pre-render without `settings` is silently discarded** (`languageRuntimePlotClient.ts` gates on `pre_render?.settings`), so it always goes through `RenderParams::to_result`, which includes them.

`plot_render_settings` has the same shape as a `render` request's params, so the same parser handles both — which is what keeps a pre-render identical to what a render would have produced.

### Base64, twice over, differently

The two transports disagree, and both are right:

| Transport | `image/svg+xml` | Binary formats |
| --- | --- | --- |
| Static display bundle | as text | base64 |
| Plot comm reply, and `pre_render` | **base64** | base64 |

The comm's convention is forced: Positron builds `data:{mime_type};base64,{data}` from the reply, so text sent as itself yields an invalid URI. The reference Python backend encodes unconditionally for the same reason. `RenderParams::encode` is the comm's one entry point so this cannot drift.

### The size clamp

`sizing::MAX_PX` is **16384**, matching `ggsql::writer::MAX_RASTER_DIMENSION` —
the most a GPU is asked to grant, and beyond what any pane can ask for. So the
clamp is a guard against a nonsense `size` or `pixel_ratio` arriving over the
comm rather than a limit a real plot meets: a large pane at 2x renders at its
full device size. A GPU whose own limit is lower rejects the frame and names
it. The number is spelled out rather than imported because a default kernel
build has no raster writer to import it from.

### Sizing

`Canvas::from_logical` scales pixels **and** dpi by the device pixel ratio together. Scaling the pixels alone renders the same chrome into more pixels — a blurry plot at the right size; scaling dpi alone grows the chrome instead of the resolution. This matches matplotlib's Positron backend. `metadata[mime].width/height` then carries the CSS size to display at, so a 2× render appears sharp rather than twice as big.

**Two different channels report size, and they report different things:**

| Channel | Carries | Sizes |
| --- | --- | --- |
| `execute_request`'s `positron` dict | `output_width_px`, `output_pixel_ratio` | A **cell output** slot — as wide as the cell, as tall as whatever it is given |
| plot comm `render` params, and the ui comm's `did_change_plots_render_settings` | a required `{width, height}` plus `pixel_ratio` and `format` | The **Plots pane**, which is why a plot in the pane fits it exactly |

So `RenderHints::canvas` picks a height (golden ratio, close to ggplot2's default figure) because a cell genuinely reports none — while the pane never comes through that function at all, since its size arrives per render rather than per execution.

**A static bundle carries no `output_location`.** That key routes an output to Positron's plot widget, which would show the picture in the Plots pane *as well as* in the cell — one plot arriving twice.

## Positron-specific bits

- `data_explorer/` implements Positron's data-explorer comm channel. An explorer's `Source` says which database its table is in: the session's own, for a table opened from the Connections pane, or the private results database below.

### The data explorer as SQL

Every page, filter, sort, profile and export is a query; the table is never loaded whole. A few things are deliberate:

- **Filter values arrive as strings and are checked against the column's type** before they reach the SQL: a number must parse as a finite number, a boolean as true/false, and everything else becomes a quoted literal. Filters are always ANDed (`supports_conditions: unsupported`).
- **A bad filter is left out, not fatal.** One the kernel cannot build, or one the database rejects (counted on its own to find out which), comes back with `is_valid: false` and an `error_message`; the rest still apply.
- **Profiles and exports see the filtered rows**, through `relation()`; pages and exports also follow the sort.
- **Sorting a retained result breaks ties on DuckDB's `rowid`**, its original row order, so paging is stable and clearing the sort restores the query's own `ORDER BY`. A Connections-pane table has no portable equivalent and gets no tiebreak.
- **Regular-expression search needs `Capabilities::regex`** (`regexp_matches`, DuckDB only). The results database always has it; the session's database has it when its URI is `duckdb://`.
- **Convert to Code writes SQL for the session's database**, not the private one, wrapping a retained result's original query as a derived table. A filter it cannot express there is left out with a `--` comment saying so.
- **A failed request is a JSON-RPC error**, with `plot::comm::RpcError`'s encoding, not a `null` result.

### Opening a console result in the data explorer

The console renders a tabular result as an HTML table and has no output toolbar to hang a button on, so the affordance is a link, and it needs no change to Positron:

1. **A console result is retained.** `results.rs` copies it into a private in-memory DuckDB under an unguessable key, bounded by `--max-results` (default 16). Only results no explorer has open count against the bound or are evicted, so an open tab never loses its table. Querying a copy rather than the source means one SQL dialect for everything the explorer asks, no remote query re-run per page, and nothing visible in the user's session.
2. **A `stream` follows the `execute_result`**, carrying an OSC 8 hyperlink to `x-ggsql-view:<session-id>/<key>`. It cannot go in the bundle's `text/plain`: the console shows the HTML and ignores it. So that the link stays in view under a long result, a console table shows rows the way R's tibble does — all of them up to twice `--console-rows` (default 10), else the first `--console-rows` — where elsewhere it shows up to 100 (`display.rs`'s `html_rows_shown`). The option is also read from `GGSQL_CONSOLE_ROWS`, which is how the extension sets it.
3. **Positron offers a clicked hyperlink to `openResource`** on the last active console session of each language. The extension's session wrapper (`ggsql-vscode/src/resultLinks.ts`) handles the scheme and runs `-- @view-result <key>` silently, in the session the link names.
4. **`-- @view-result` opens a `positron.dataExplorer` comm with `inline_only: false`**, which Positron opens as an editor tab immediately. A result that already has an explorer open is not opened again: the kernel sends `open_editor` on the UI comm with that explorer's editor URI (`positron-data-explorer:positron-data-explorer-<comm id>`), which focuses the tab — what Positron's Variables pane does for a variable already in view. A key that is no longer retained is a cell error, which surfaces as a warning from the extension.

Only `SessionKind::PositronConsole` gets this: anywhere else the escape sequence would be stray text. Notebooks have a better route — the inline data explorer — that this does not yet use.

A query under the user-facing `-- @view` directive (see [the Positron docs](../doc/get_started/tooling/positron-vscode.qmd)) takes the same path from step 1, minus the click: it is retained in any Positron session, silent executions included, and opened at once. The table still shows in the cell, and a console still gets the link under it, to reopen a closed tab; a silent execution shows nothing but the tab. A `Standalone` session has no data explorer to open, so it gets the inline table and a `stderr` note instead.
- The companion VS Code extension (`ggsql-vscode/`) ships a copy of this binary and also discovers installed ones via the `ggsql.kernelPath` setting, the Jupyter kernelspec directories, the native install locations, or `PATH`. It runs each one with `--version` to name the runtime it registers, so **keep `--version` working**: a kernel that does not answer it is still offered, but without a version in the picker. See [Finding the kernel](../ggsql-vscode/CLAUDE.md#finding-the-kernel).

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
default = ["all-readers", "raster-plots"]
all-readers = ["sqlite", "odbc", "duckdb"]
```

The reader features pass through to `ggsql/<feature>`, so the default install supports DuckDB, SQLite and ODBC connection strings. `raster-plots` adds `ggsql/png`, `ggsql/jpeg` and `ggsql/tiff` — **default on purpose**, for the reason in [`choose`](#choose): the Plots pane asks for `png`, every shipped build enables it, and a plain `cargo build` that quietly produced SVG-only plots differed from the released kernel in the way hardest to notice. `--no-default-features --features all-readers` builds without wgpu; plots then render as SVG. The release workflow passes no feature flags at all, so a wheel is exactly what a plain build produces.

The vector writers are not features here: `ggsql`'s `svg` and `pdf` are named unconditionally in `[dependencies]`, because the fallback path has to be compiled in whatever else is.

## Testing

The Rust side has unit tests inline (`cargo test -p ggsql-jupyter`). The Jupyter protocol tests are Python:

```sh
cd ggsql-jupyter/tests
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`test_compliance.py` verifies handler coverage (`execute_request`, `kernel_info_request`, `is_complete_request`, `shutdown_request`); `test_integration.py` drives a real kernel via `jupyter_client`. Both run in CI (`jupyter-protocol-tests` job in `/.github/workflows/build.yaml`).

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/ggsql-vscode/CLAUDE.md`](../ggsql-vscode/CLAUDE.md) — the VS Code / Positron extension that drives this kernel.
- [`/src/CLAUDE.md`](../src/CLAUDE.md) — the underlying `ggsql` library.
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing notebook docs.
