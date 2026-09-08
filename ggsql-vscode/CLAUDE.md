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

1. Discovers a `ggsql-jupyter` binary via, in order: the `ggsql.kernelPath` setting, an installed Jupyter kernelspec named `ggsql`, or `ggsql-jupyter` on `PATH`.
2. Registers it as a Positron language runtime so `▶ Run` and the Console route to the kernel.
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

## Settings

```json
{
  "ggsql.kernelPath": "string"   // empty → use 'ggsql-jupyter' from PATH
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

Watch mode for development: `npm run watch` (runs esbuild + tsc in parallel).

For an interactive session, open the **repo root** in Positron and press <kbd>F5</kbd> ("Run Extension"). [`/.vscode/launch.json`](../.vscode/launch.json) runs the `build-ggsql-vscode` task, which is `npm run watch` in this folder, then opens an Extension Development Host with `--extensionDevelopmentPath`, so the extension loads from source with no VSIX. Launch from Positron rather than VS Code, or the dev host has no Positron API and the runtime manager never registers. The watcher rebuilds `out/extension.js` on save, but the host does not hot-reload: run _Developer: Reload Window_ in the Extension Development Host to pick up a change.

## Testing

```sh
cd ggsql-vscode
npm test              # grammar scopes, then the VS Code suites
npm run test:grammar  # TextMate scopes only; no Electron, fast
npm run test:extension
```

Tests live in `src/test/` and compile to `out-test/` via `tsconfig.test.json`, deliberately not to `out/`, which `esbuild.js` owns. The whole of `src/` compiles there, not just `src/test/`, because the unit tests import the extension's own modules. `@vscode/test-cli` launches a real VS Code instance, so a window appears while the suites run; CI wraps the same command in `xvfb-run`.

Note that `tsc` does not prune output for deleted sources: if you delete or rename a test, remove its `.js` and `.js.map` from `out-test/test/` or the runner keeps executing the stale copy. `npm run test:extension` on its own does not recompile, so run `npm test` (or `npm run compile-tests` first) after editing any `.ts`.

The suites cover the extension as stock VS Code sees it: activation, language resolution, cell parsing, `.sql` gating, CodeLens placement, TextMate scopes, and the parts of `manager.ts` and `positronApi.ts` that are reachable without a Positron host. `bundle.test.ts` additionally asserts against the built `out/extension.js`. The rest of the Positron surface (session creation, connection drivers, cell execution) is not covered, since it needs a Positron host, and `sqlAssociation.ts` and `connections.ts` are untested.

Add new tests as `src/test/<name>.test.ts`; no config change is needed.

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
