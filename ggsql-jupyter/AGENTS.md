# `ggsql-jupyter/` — Jupyter kernel

Standalone Rust binary that speaks the Jupyter messaging protocol over ZeroMQ. Embeds the `ggsql` library and renders results as images (visual queries) or HTML tables (pure SQL). Workspace member; published to crates.io and to PyPI (as a binary wheel via maturin).

End-user installation and usage live in [`README.md`](README.md). End-user notebook docs live in [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-jupyter/
├── Cargo.toml          Rust binary + library, depends on ggsql with duckdb + svg/pdf + raster-plots
├── examples/results_bench.rs  Results API benchmark (DuckDB cache vs. Arrow row indices)
├── pyproject.toml      maturin config (bindings = "bin") for the PyPI wheel
├── README.md           User-facing install + usage
├── src/
│   ├── main.rs         Binary entry: clap CLI (start kernel, --install)
│   ├── lib.rs          Library root (so internals can be unit-tested)
│   ├── kernel.rs       Jupyter messaging loop (ZMQ, message dispatch)
│   ├── executor.rs     Runs queries via ggsql::Reader, returns rendered output
│   ├── connection.rs   Reader lifecycle, connection-string parsing
│   ├── data_explorer.rs  Positron data-explorer comm, for tables opened from the Connections pane
│   ├── results/        Frontend-neutral results API over recent console results
│   │   ├── mod.rs      ResultStore: pending results, the results cache, views
│   │   ├── view.rs     One view: filters, sort, order table, pages, profiles, export
│   │   ├── filter.rs   Filter and SortKey, and the SQL for them
│   │   ├── methods.rs  The API as named JSON calls (`result_open`, `result_page`, …)
│   │   └── export.rs   A selection as CSV / TSV / HTML
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

- `data_explorer.rs` implements Positron's data-explorer comm for a table opened from the Connections pane, querying the session's own database.
- The companion VS Code extension (`ggsql-vscode/`) ships a copy of this binary and also discovers installed ones via the `ggsql.kernelPath` setting, the Jupyter kernelspec directories, the native install locations, or `PATH`. It runs each one with `--version` to name the runtime it registers, so **keep `--version` working**: a kernel that does not answer it is still offered, but without a version in the picker. See [Finding the kernel](../ggsql-vscode/AGENTS.md#finding-the-kernel).

### The results API

`results/` keeps recent query results and answers a small, frontend-neutral set of calls on them (`methods.rs` documents each: `result_open`, `result_set_filters`, `result_set_sort`, `result_page`, `result_profile`, `result_export`, `result_sql`, `result_close`). Nothing in it knows about Positron: the ggsql extension translates Positron's Data Explorer protocol into these calls (see [`ggsql-vscode/AGENTS.md`](../ggsql-vscode/AGENTS.md)). The kernel's only Positron-specific part is the route: a `call_method` request on the `positron.ui` comm — what an extension's `session.callMethod(name, params)` sends — whose method starts with `result_` goes to `methods::call`, and its answer or error comes back as the JSON-RPC reply (errors with `plot::comm::RpcError`'s encoding). A `call_method` for any other name is an error, like an unknown plot-comm method, so a caller expecting something the kernel lacks fails visibly; any other UI request gets `result: null`.

- **Which results are kept.** Only when the frontend asks, with `--keep-results` or `GGSQL_KEEP_RESULTS` — the extension sets it for a console session whose results it can open — so a session nobody explores holds nothing beyond what it displays. Then every tabular result of a non-silent execution is kept, and its key — unguessable hex, so a stale link from another kernel process names nothing — goes in the `execute_result`'s `metadata` as `{"ggsql": {"result": key}}`; the kernel prints no link itself. Unopened results are held as their Arrow data.
- **One byte budget covers everything kept**, `--results-max-bytes` / `GGSQL_RESULTS_MAX_BYTES` (default 1 GB, by Arrow's estimate of the data): unopened results, the cache's tables, pinned ones included, and every view's order table. Making room drops closed results first, then the oldest unopened ones. An open view is never pulled from under its frontend, so when open views alone fill the budget a new result is not kept (it is shown without a key) and a sort or filter that needs room fails with an error saying so.
- **The cache's DuckDB has a hard memory limit of twice the budget** (`memory_limit_for`, at least 64 MB). The budget counts what is kept, by Arrow's estimate; DuckDB holds a table less compactly than that and needs working memory on top to sort and profile, so the limit is what bounds the process. A budget too large to set as a limit leaves DuckDB's default, most of the machine's memory.
- **Opening copies the result into the results cache.** A `CacheTables` (the caching reader's bookkeeping, `ggsql::reader::cache`) on its own in-memory DuckDB, which `ResultStore` creates itself. The session's reader is never wrapped, so the user's dialect is untouched. The copy adds `__ggsql_row__`, each row's position in the query's output, and drops the Arrow data, so a result is held once. It stays **pinned** while any view of it is open; once closed, it goes when the budget needs the room, and reopening before then reuses it. A result can have several views, each with its own filters and sort.
- **Views end with the UI comm.** Closing it closes every view, since nothing could close them once it has gone; that is what releases the pins of a frontend that reloaded or crashed.
- **Nothing is written to disk.** The cache sets `temp_directory = ''`, so DuckDB cannot spill: a query that needs more memory than the limit fails with DuckDB's out-of-memory error, and the view it was for is left as it was.
- **Sorting and filtering build an order table** of `(pos, __ggsql_row__)` per view (`view.rs`); a page joins it back to the result by `pos`, so scrolling never sorts again. Ties break on `__ggsql_row__`, so paging is stable and clearing the sort restores the query's own order. An unfiltered, unsorted view has none and pages by `__ggsql_row__` directly. A new sort or set of filters is built under a new table name and only then replaces the old one, so one that fails leaves the view as it was.
- **A page is bounded.** At most `MAX_PAGE_ROWS`; a range is cut short at the end of the view, and an index outside it is an error, so a cell never lands in another row's place. Scattered indices fetch just those rows.
- **Filter values arrive as strings and are checked against the column's kind** before they reach the SQL: a number must parse as a finite number, a boolean as true/false, anything else becomes a quoted literal. A filter that cannot be built, or that the database rejects (tried on its own to find out which), is left out and its message returned in `errors`; the rest still apply. Filters are always ANDed. Identifiers, booleans and regular expressions come from the dialect the SQL is for (`SqlDialect::sql_regex_match`, `None` where a backend has none), and `LIKE` escapes with `!`, which no backend reads as a string escape.
- **Kinds are DuckDB type names matched exactly** (`kind_of`): a column type that only contains `INT`, like `POINT`, is not an integer, and an `ENUM` is a string.
- **`result_sql` writes SQL for the session's database**, not the cache: the cell's statements before the last as they were, then the last as a derived table with the view's filters and sort, in the session's dialect. It is in the kernel because only the kernel knows that dialect. A filter it cannot express there, or one on a column whose name the query repeats (renamed `a_1` in the cache, but ambiguous in SQL), is left out with a `--` comment.

`examples/results_bench.rs` measures it. At 10M rows (617 MB of Arrow data) on an Apple Silicon laptop: opening takes about 3.4 s, a sort 0.4–0.6 s, a page 20 ms, and memory levels off near 2 GB across repeated open/close cycles. Sorting the Arrow data directly avoids the copy but takes 2–5 s per stable sort, which is why the cache is DuckDB.

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

- [`/AGENTS.md`](../AGENTS.md) — workspace overview.
- [`/ggsql-vscode/AGENTS.md`](../ggsql-vscode/AGENTS.md) — the VS Code / Positron extension that drives this kernel.
- [`/src/AGENTS.md`](../src/AGENTS.md) — the underlying `ggsql` library.
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing notebook docs.
