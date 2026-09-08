# `ggsql-vscode/` — VS Code / Positron extension

TypeScript extension that adds ggsql language support to VS Code and Positron: syntax highlighting, code-cell execution, connection management, and (in Positron) a registered language runtime that drives the `ggsql-jupyter` kernel.

Not a Cargo workspace member — this is a standalone npm project. End-user docs live in [`README.md`](README.md). Tooling overview for users: [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd). This file describes the *implementation*.

## Layout

```
ggsql-vscode/
├── package.json              Extension manifest (commands, keybindings, languages, runtime)
├── tsconfig.json
├── esbuild.js                Bundler config (builds out/extension.js)
├── eslint.config.mjs
├── language-configuration.json   Bracket pairs, comment markers
├── logo.png                  Marketplace icon
├── src/
│   ├── extension.ts          activate(): registers commands, manager, code lenses
│   ├── manager.ts            Kernel discovery + Positron language-runtime registration
│   ├── positronApi.ts        Acquires the Positron API so Positron can attribute it to this extension
│   ├── connections.ts        Connection-string handling for the Connections pane
│   ├── cellParser.ts         Splits .ggsql files into cells for Run-Cell commands
│   ├── codelens.ts           "▶ Run cell" lens above each cell
│   ├── decorations.ts        Cell separator decorations
│   ├── context.ts            Sets editor context keys (e.g. ggsql.hasCodeCells)
│   ├── sqlAssociation.ts     One-time notice pointing at files.associations for .sql highlighting
│   ├── types.ts              Shared interfaces
│   └── test/                 Mocha suites (unit + activation) and the grammar fixture
├── syntaxes/
│   └── ggsql.tmLanguage.json TextMate grammar (used for tokenization in VS Code)
├── bundled/bin/              Kernel shipped inside the platform VSIXes (staged at release time, not in git)
├── examples/                 Sample .ggsql files
├── resources/                Static assets bundled with the extension
│   ├── ggsql-icon.svg        Full-colour logo; read by manager.ts for base64EncodedIconSvg
│   └── ggsql-lang-icon.svg   contributes.languages[].icon; tuned for the 16px file-icon slot
└── ggsql-0.1.0.vsix          Packaged extension (build artifact, may be stale)
```

## File extensions and language ID

`package.json` registers `id: ggsql` for `.gsql`, `.ggsql`, and `.ggsql.sql`. **Order matters:** the first entry is the primary extension, and it is what the workbench suggests when saving an untitled ggsql document (`textFileService.suggestFilename` uses `extensions.at(0)`). `.gsql` is first to match the documented extension in [`/doc/get_started/tooling/positron-vscode.qmd`](../doc/get_started/tooling/positron-vscode.qmd).

The TextMate grammar at `syntaxes/ggsql.tmLanguage.json` provides tokenization. Tree-sitter highlights — used by editors that prefer the grammar package directly — live in [`/tree-sitter-ggsql/queries/highlights.scm`](../tree-sitter-ggsql/queries/highlights.scm).

## Commands and keybindings

Declared in `package.json` and wired up in `extension.ts`:

| Command | Default key | Purpose |
| --- | --- | --- |
| `ggsql.runCurrentAdvance` | Cmd/Ctrl+Enter, Shift+Enter | Run current cell, advance to next |
| `ggsql.runQuery` | Cmd/Ctrl+Shift+Enter | Run current cell only |
| `ggsql.runNextCell` | — | Run the next cell |
| `ggsql.runCellsAbove` | — | Run all cells above the cursor |
| `ggsql.sourceCurrentFile` | — | Run the entire file (also exposed as the editor "Run" button) |
| `ggsql.createNewFile` | — | Open a new untitled ggsql document (also in the New File dialog) |

Cells are detected by `cellParser.ts`; `codelens.ts` puts a CodeLens above each cell.

`ggsql.createNewFile` is contributed to the `file/newFile` menu under its own `ggsql` group, so the New File dialog lists it beneath the built-in File and Notebook sections rather than alongside them. It is registered before the Positron API check in `extension.ts`, since opening an untitled document does not need Positron.

## Attaching to `.sql` files

ggsql is a SQL superset, so the kernel can execute a plain `.sql` file. Rather than claiming the `.sql` file association, the extension attaches its run affordances to documents already tokenized as `sql`:

- [`src/languages.ts`](src/languages.ts) is the single source of truth. `isGgsqlDocument()` decides whether ggsql will act on a document; every command guard, decoration and context key routes through it. Add new language-scoped behaviour there rather than comparing `languageId` inline.
- `CELL_LANGUAGE_IDS` registers the CodeLens provider for both ids. The provider gates on `isGgsqlDocument()` at request time and fires `onDidChangeCodeLenses` when `ggsql.enableSqlFiles` changes, so the setting takes effect without re-registering.
- The `editor/title/run` `when` clauses gate on `config.ggsql.enableSqlFiles` directly, so no context key is needed for the buttons.

Why not claim `.sql` in `contributes.languages`: extension-contributed associations are resolved last-registered-wins (`languagesAssociations.ts`), which is not a contract and is fragile against other SQL extensions. It would also replace the richer built-in `source.sql` grammar and break language-scoped features from other SQL tooling.

[`src/sqlAssociation.ts`](src/sqlAssociation.ts) shows a one-time notice pointing at `files.associations`, which is the right mechanism for users who want ggsql highlighting in `.sql` files because user-configured associations sit in the highest precedence tier and so win deterministically.

**It does not write the setting.** Mapping `*.sql` rewrites how every `.sql` file in every workspace is treated, which is not an extension's call, and running SQL in the ggsql console does not need it. An earlier version did write it on a button press, and that turned out to need a reset command, an offer to un-write, and care over which config level to base the update on. None of that is needed now.

Three constraints worth knowing before editing the notice:

- **Notification text is not markdown.** It is parsed by `parseLinkedText`, which matches markdown links and nothing else, so code spans and emphasis render literally. Hrefs are limited to `https:`, `command:` and `file:`, and may contain no spaces or `)`. The setting name is therefore a `command:` link, and the mapping is spelled out in prose rather than backticked.
- **Any button press dismisses the notification**, so every button has to be a complete answer. The documentation link lives on the `ggsql.enableSqlFiles` setting rather than being a button here.
- **The notice fires on every `.sql` open, and only "Don't show again" persists.** An Info notification with buttons is *not* sticky (`notifications.ts` makes actions sticky only at Error severity), so it auto-hides after 10 seconds and extensions cannot opt out. A show-once notice is therefore trivially missed, which would defeat the point. A module-level `noticeVisible` guard stops that becoming a pile-up when many `.sql` files open at once. `ggsql: Reset .sql File Association Prompt` clears the persisted flag, the only route back from "Don't show again" short of editing the global storage database.

The `ggsql.enableSqlFiles` description uses `markdownDescription` rather than `description`, which matters for two reasons: only the markdown path runs `fixSettingLinks`, and only it renders links at all. It uses both link forms available there:

- `` `#files.associations#` `` renders as a link to that setting. The backticks (or single quotes) are required; a bare `#files.associations#` is left as literal text.
- A plain markdown link to [`/doc/get_started/tooling/positron-vscode.qmd`](../doc/get_started/tooling/positron-vscode.qmd), whose `## Using ggsql with .sql files` heading carries an explicit `{#sql-files}` anchor so the URL survives a rewording. Keep the anchor if you edit that heading.

## Positron integration

The extension declares `contributes.languageRuntimes` for `ggsql` (see `package.json`) and depends on `@posit-dev/positron`. When activated under Positron, `manager.ts`:

1. Discovers `ggsql-jupyter` binaries as described in [Finding the kernel](#finding-the-kernel) below.
2. Registers each as a Positron language runtime so `▶ Run` and the Console route to the kernel.
3. Tells the kernel what kind of session this is, and subscribes to the one notification it needs — see below.

### The two things the extension has to tell the kernel

**Plots are rendered in the kernel and reach the Plots pane over a `positron.plot` comm.** The extension routes nothing itself: there is no `output_location` metadata and no client-side renderer. What it does do is supply the two pieces of context the kernel cannot work out for itself.

- **`createKernelSpec` appends `--session-mode <console|notebook|background>`** from `sessionMetadata.sessionMode`. The kernel keys every output decision off that: a console session gets the comm, a notebook session gets a static image in its cell, and background gets the standalone path. Without the flag the kernel falls back to guessing from its session id, which works but is a heuristic. `restoreSession` deliberately does *not* rebuild the spec — the supervisor replays the argv the session was created with, and a session's mode never changes.
- **`uiSubscriptions: ['did_change_plots_render_settings']`** on the runtime metadata. Without it the frontend never tells the kernel how large the Plots pane is, and the kernel needs that for exactly one thing: rendering a *new* plot at the right size so its `comm_open` can carry it, and the pane shows it immediately instead of blank. Every other render carries its own size on the request.

Both live in `manager.ts`. The kernel side of the contract — why the comm alone creates the pane entry, and why an output bundle alongside it would show the plot twice — is in [`/ggsql-jupyter/CLAUDE.md`](../ggsql-jupyter/CLAUDE.md).

Outside Positron there is no way to execute a query: `activate()` returns early, so every command that runs code stays unregistered. To avoid offering actions that cannot work, everything execution-related gates on Positron's built-in **`isPositron`** context key ([extension development docs](https://positron.posit.co/extension-development.html#option-1-context-keys)):

- the two `editor/title/run` buttons
- the three keybindings
- the five run commands, hidden from the Command Palette via a `commandPalette` menu entry

`isPositron` is declared with a default of `true`, so it is correct in Positron from startup with no code and no activation-order window. In VS Code the key is never declared, so it evaluates falsy. Prefer it over a hand-rolled `setContext` key for anything purely declarative; `getPositronApi()` from [`src/positronApi.ts`](src/positronApi.ts) is the right check when the code needs the API object itself.

**Acquiring the Positron API.** `src/positronApi.ts` calls `require('positron')` directly, and `esbuild.js` lists `positron` in `external` so the call is still there in `out/extension.js`. This is load bearing, not a style choice. Positron's require interceptor works out which extension owns an API object from the filesystem path of the requiring file, and that identity becomes the `extensionId` on every runtime the extension registers. Reaching the API through the global accessor that `tryAcquirePositronApi()` uses puts the requiring path inside Positron's own bootstrap, which the interceptor cannot place, so the runtime is filed under `nullExtensionDescription` and Positron cannot activate this extension when it restores sessions after a window reload. `src/test/bundle.test.ts` guards the esbuild half of this.

The Positron Supervisor is a soft dependency, reached through `getSupervisorApi()` in `manager.ts`. It deliberately is not in `extensionDependencies`: that field is static, and an entry for `positron.positron-supervisor` would stop the extension activating at all in VS Code, where the supervisor does not exist.

`GgsqlRuntimeManager.alwaysRediscover` is `true` because ggsql runtimes are never marked `cacheable`, so Positron must run discovery on every window open rather than trusting its cross-window cache. The typings declare it as an optional member, so `tsc` checks the value's type but not the name: a misspelling would compile as a harmless extra property and silently disable the flag. `src/test/manager.test.ts` is the guard, because the property access there fails to compile if the name changes.

Anything that does *not* need the runtime (`ggsql.createNewFile`, `ggsql.resetSqlAssociationPrompt`, syntax highlighting) is registered before the early return and works in plain VS Code. Add new commands on the correct side of that line, and gate them if they execute code.

## Finding the kernel

The extension ships the kernel: the per-platform VSIXes carry `ggsql-jupyter` at `bundled/bin/`, so installing the extension is enough and no native installer is needed. The platform-neutral VSIX carries none, and users on a platform without a build install the kernel themselves.

**Every kernel found is offered**, and the user picks in the New Console Session picker. There is one order, not a set of strategies:

| Order | Source | Where |
| --- | --- | --- |
| 1 | `Setting` | `ggsql.kernelPath`, when set |
| 2 | `Bundled` | `bundled/bin/` inside the extension |
| 3 | `Jupyter` | Jupyter kernelspec directories, user then system |
| 4 | `System` | the native package install location for the platform |
| 5 | `Path` | `PATH` |

Order decides which the picker lists first — hence which is the default — and, when two paths name one file, which occurrence survives `dedupeCandidates()`. The bundled kernel leads unless the user named one, so a machine with nothing installed gets a working runtime and a machine with an install keeps being offered it.

`selectKernelCandidates()` is the whole rule with no filesystem in it, which is what `src/test/kernelDiscovery.test.ts` exercises; `discoverKernelPaths()` supplies it with what is actually on disk.

Five things here are load bearing:

- **A kernel is run before it is offered, and what it says is what the picker shows.** `probeKernel()` runs `ggsql-jupyter --version` and reads the version out of its output, so the runtime is named `ggsql 0.4.1` the way Positron's own runtimes are. Filesystem checks cannot tell whether a binary starts: the bundled kernel is built for the platform but not for every system it can be installed on, and one linked against newer shared libraries than the host provides is exec'd successfully and then killed by the dynamic linker, which no `stat` or `access` call can see. A kernel released before `--version` existed rejects the argument and exits non-zero, which by exit status alone is the same answer the loader-killed binary gives. The probe settles the two apart by asking a question every version answers — `--help` — and only a kernel that fails *that* is treated as unable to run; one that passes it is offered as plain `ggsql (<source>)` with no version, since dropping it would take away the install the user already had. **Beyond that, only the bundled kernel has to pass at all:** a kernel the user installed is their own business and is offered even when it would not start.
- **Successful probes are cached, failures are not.** `ProbeCache` keeps them in `globalState` keyed by kernel path, with the file's mtime and size in the entry, so an install upgraded in place is probed again instead of reporting the version it used to have, and a pass costs one spawn per kernel per install rather than one per window. It rewrites the map with only the kernels the pass saw, or the paths of every superseded extension version would accumulate. A failure is not stored: it is cheap to repeat, and a host that gains the missing shared libraries should start working without waiting for an update.
- **Every candidate is an absolute path.** A candidate that is only a binary name satisfies each existence check further down and so registers a runtime that fails at session start with `KS-19: Kernel path not found`. `findOnPath()` returns `undefined` rather than the bare name, and `isKernelAccessible()` rejects any non-absolute path, so no kernel anywhere means **zero** runtimes rather than an unusable one. The single exception is a `ggsql.kernelPath` that resolves to nothing: it is passed through so discovery can report it as inaccessible in the log instead of ignoring the setting silently.
- **The bundled kernel's `runtimeId` is fixed, not derived from its path.** Every other source hashes `kernelPath` to get one id per installed kernel, but the bundled path contains the versioned extension directory, so hashing it would mint a new runtime on every extension update and lose the workspace's runtime affinity and its restorable sessions. That only holds up because `validateMetadata()` regenerates the metadata Positron stored for the workspace: the id survives the update, the path in the stored copy does not. It rejects metadata with no matching candidate, which is how Positron learns to drop a runtime whose kernel has been uninstalled.
- **The bundled runtime carries no `(<source>)` qualifier.** It is the default, so there is nothing to distinguish it from. `runtimeShortName` stays plain `ggsql` for every kernel: it labels a console tab, where the version adds nothing.

Discovery also writes the user-level Jupyter kernelspec for the *leading* runnable kernel, so Quarto and Jupyter can find ggsql without a session ever being started, and so the spec stops pointing into an extension directory an update has removed. Only one is written, and never for a kernel that was itself found as a kernelspec — that one is already where Jupyter looks. Only a kernel that has passed the probe is written there: the spec outlives the window and is what Quarto resolves, and it has no fallback of its own.

The one case that interrupts the user is the dead end: nothing runnable anywhere, whether because the bundled kernel failed its probe or because the build carries none (the platform-neutral VSIX). `reportNoUsableKernel()` then shows a non-modal warning once per extension version, offering the install docs and the log. A kernel that is merely skipped — an unusable `ggsql.kernelPath`, say, with others still available — is reported in the log only.

## Settings

```json
{
  "ggsql.kernelPath": "string"   // an extra kernel to offer, listed first
}
```

## Build & package

```sh
cd ggsql-vscode
npm install                # one-time
npm run check-types        # tsc --noEmit
npm run package            # esbuild → out/extension.js (production)
npx vsce package           # produces ggsql-<version>.vsix
code --install-extension ggsql-<version>.vsix
```

A local `vsce package` produces the kernel-less VSIX, since `bundled/` only exists in a release build.

**Release builds** live in [`/.github/workflows/release-packages.yml`](../.github/workflows/release-packages.yml), not in a workflow of their own. Its `build-vsix` job runs a matrix of six — the five platform targets plus `universal` — downloading the `ggsql-jupyter-<target>` artifact each platform job uploaded between signing and installer packaging, restoring the executable bit, and running `vsce package --target <target>`. `publish-openvsx` then publishes the packaged file to Open VSX.

Four things about that arrangement are deliberate:

- **The VSIX build cannot live in its own workflow.** Actions artifacts are scoped to a single workflow run, and two workflows triggered by the same tag run in parallel, so a separate workflow could not download the kernels. Building in the same run also means the kernel and the extension always come from one commit.
- **The executable bit has to be restored after download.** Artifact upload and download drop it. It does survive `vsce package` into the VSIX itself, so restoring it once in CI is enough; `ensureExecutable()` in `manager.ts` is belt-and-braces for an install that loses it.
- **The published artefact is the packaged `.vsix`, with no `target` passed to the publish action.** Open VSX reads the platform from the `TargetPlatform` attribute that `vsce package --target` writes into `extension.vsixmanifest`, and defaults to `universal` when it is absent; `ovsx` discards a target option when handed an already-packaged vsix.
- **`win32-arm64` is not built.** No runner produces that kernel yet. Positron's bootstrap appends `?targetPlatform=<target>` and gets an HTTP 403 rather than the universal build for a target that was never published, so the universal VSIX is not a fallback for it — see posit-dev/positron#14954.

Watch mode for development: `npm run watch` (runs esbuild + tsc in parallel).

For an interactive session, open the **repo root** in Positron and press <kbd>F5</kbd> ("Run Extension"). [`/.vscode/launch.json`](../.vscode/launch.json) runs the `build-ggsql-vscode` task, which is `npm run watch` in this folder, then opens an Extension Development Host with `--extensionDevelopmentPath`, so the extension loads from source with no VSIX. Launch from Positron rather than VS Code, or the dev host has no Positron API and the runtime manager never registers. The watcher rebuilds `out/extension.js` on save, but the host does not hot-reload: run _Developer: Reload Window_ in the Extension Development Host to pick up a change.

## Testing

```sh
cd ggsql-vscode
npm test                  # grammar scopes, then the VS Code suites
npm run test:grammar      # TextMate scopes only; no Electron, fast
npm run test:extension
npm run test:integration  # downloads Positron; needs a staged kernel (see below)
```

Tests live in `src/test/` and compile to `out-test/` via `tsconfig.test.json`, deliberately not to `out/`, which `esbuild.js` owns. The whole of `src/` compiles there, not just `src/test/`, because the unit tests import the extension's own modules. `@vscode/test-cli` launches a real VS Code instance, so a window appears while the suites run; CI wraps the same command in `xvfb-run`.

Note that `tsc` does not prune output for deleted sources: if you delete or rename a test, remove its `.js` and `.js.map` from `out-test/test/` or the runner keeps executing the stale copy. `npm run test:extension` on its own does not recompile, so run `npm test` (or `npm run compile-tests` first) after editing any `.ts`.

The suites cover the extension as stock VS Code sees it: activation, language resolution, cell parsing, `.sql` gating, CodeLens placement, TextMate scopes, kernel discovery, and the parts of `manager.ts` and `positronApi.ts` that are reachable without a Positron host. `bundle.test.ts` additionally asserts against the built `out/extension.js`. The rest of the Positron surface (session creation, connection drivers, cell execution) is not covered, since it needs a Positron host, and `sqlAssociation.ts` and `connections.ts` are untested.

Add new tests as `src/test/<name>.test.ts`; no config change is needed. `.vscode-test.mjs` globs `out-test/test/*.test.js` — one level only, deliberately, so the Positron suite in `test/integration/` does not run under stock VS Code, where it cannot pass.

### The Positron integration suite

`src/test/integration/` is the only place a kernel is actually launched. The unit suites cover discovery precedence and metadata, and `build-vsix` proves the binary is inside the VSIX; neither can tell whether it *starts*, which is the failure the bundling work exists to fix. `npm run test:integration` builds nothing itself — stage a kernel first:

```sh
cargo build --release --bin ggsql-jupyter
mkdir -p ggsql-vscode/bundled/bin && cp target/release/ggsql-jupyter ggsql-vscode/bundled/bin/
```

`src/test/runIntegration.ts` then downloads Positron via [`@posit-dev/positron-test-electron`](https://github.com/posit-dev/positron-test-electron) and runs the suite in its extension host. Three details are load bearing:

- **`disableExtensions: false`.** Session creation goes through `positron.positron-supervisor`, one of Positron's bundled extensions. Under the harness's default `--disable-extensions` there is no supervisor and every session start fails.
- **The suite drives mocha itself.** `extensionTestsPath` must resolve to a module exporting `run()`, which is why `test/integration/index.ts` exists instead of the `@vscode/test-cli` config the other suites use. Its timeout is 120s: a session start spawns the binary and completes a Jupyter handshake.
- **`channel: 'daily'`.** Positron's stable channel is not published for every platform. Pin `version` instead once a known-good build is worth freezing.

The download is cached in `.positron-test/`, gitignored like `.vscode-test/`. It keeps a directory per Positron version, so it grows as dailies move on — around 3 GB after one run, and worth clearing occasionally rather than a leak to fix.

**Nothing in the integration job is cached in CI**, and both halves of that are deliberate.

Positron is re-downloaded every run, which is not what the job costs: on `ubuntu-latest`, downloading it and running the suite together take about 75s. Caching `.positron-test/` would add an entry of roughly a gigabyte per Positron version, and `channel: 'daily'` mints a new version most days, which is how a cache becomes the problem instead of the fix.

The kernel build is the expensive step — around 16 min cold against under 3 warm — and **the job gets it warm without owning a cache, by restoring `publish.yaml`'s.** That workflow runs on every PR as well, and builds this very binary with `cargo build --release --package ggsql-jupyter` on the same 1.86 toolchain and lockfile, so `shared-key: ${{ runner.os }}-publish` is an exact key match (`full match: true` in the log, restored in ~11s). `save-if: false` is the whole point: this job only ever reads, so it adds nothing to a 10 GB repo-wide limit that other workflows have already mostly spent — `build.yaml`'s cargo cache alone is over 5 GB.

`rust-cache` prunes the workspace's own crates and keeps dependencies, so what comes back is the expensive part (duckdb, arrow, parquet) and only the final link is left to do.

What that couples to is publish.yaml's `shared-key` and its toolchain. Rename either and this job silently goes cold, as it would if the entry were evicted — slower, never broken. The real fix would be to stop building the kernel twice per PR, but the two live in separate workflows and Actions artifacts are run-scoped, so nothing can pass between them; see [`/.github/workflows/test-extension.yaml`](../.github/workflows/test-extension.yaml).

The assertions worth keeping: a runtime with `runtimeId` `ggsql-bundled` whose path is under `bundled/bin/`, its name matching `ggsql <version>` — the only check that the version really comes back from the binary — and `executeCode` returning a result against a session started from that id, which covers spawn, handshake and execution in one call. The suite starts the runtime by id rather than by language, because every ggsql install on the machine is registered and a developer running this locally has their own.

### Testing discovery without wrecking the developer's machine

Two seams exist because discovery reads and writes real state:

- `GgsqlRuntimeManager` takes `{ kernelSpecDir }`. Discovery advertises the kernel by writing a Jupyter kernel spec, so a test that called `discoverAllRuntimes()` with the default would repoint the *real* kernelspec — the one Quarto resolves — at a temp fixture.
- `kernelDiscovery.test.ts` redirects `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA` and `PATH` to stage host kernels, restoring them in teardown. The native-installer locations (`/usr/local/bin`, `/usr/bin`, `/Applications`) are hard-coded absolutes that no environment variable can redirect, so any test asserting the whole list of candidates or runtimes — every install being offered, that is most of them — calls `systemInstallPresent()` and skips on a machine that has one. CI never does, which is where those regressions matter.

**Any test that calls `discoverAllRuntimes()` and asserts on the whole result must call `isolateHostEnv()` first**, not just the ones staging a host kernel. A count of probes or an assertion of "nothing was registered" is a claim about every kernel discovery can see, and on a developer's machine that includes their own installs. Running the extension is enough to create one: discovery writes the user kernelspec, so a session in the Extension Development Host leaves a kernel at `~/Library/Jupyter/kernels/ggsql/` that the suite then discovers. `systemInstallPresent()` does not cover this — it checks the native-installer paths only.

### Editing the grammar fixture

`src/test/grammar/highlight.gsql` uses `vscode-tmgrammar-test`'s annotation format, which has three rules worth knowing before you touch it:

- The header must be exactly `-- SYNTAX TEST "source.ggsql"`. A trailing `>>`, which some examples show, makes the tool reject the file with a parse error rather than an assertion failure.
- A `^` caret's column is the comment token length plus its index in the assertion line, matched against the source line's 0-based columns. Carets therefore cannot target source columns 0 and 1, which is why the `<---` form exists for line-initial tokens.
- For `<---`, the dash count sets the assertion's right edge from column 0. It must be at least 1 and no more than the target token's length, so it is not cosmetic.

Assertion lines are stripped before tokenization, so they never tokenize as ggsql comments.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/ggsql-jupyter/CLAUDE.md`](../ggsql-jupyter/CLAUDE.md) — the kernel this extension drives.
- [`/tree-sitter-ggsql/CLAUDE.md`](../tree-sitter-ggsql/CLAUDE.md) — grammar that powers more advanced editor highlighting.
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing tooling docs.
