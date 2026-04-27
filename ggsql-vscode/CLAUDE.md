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
├── logo.png, icon.png
├── src/
│   ├── extension.ts          activate(): registers commands, manager, code lenses
│   ├── manager.ts            Kernel discovery + Positron language-runtime registration
│   ├── connections.ts        Connection-string handling for the Connections pane
│   ├── cellParser.ts         Splits .ggsql files into cells for Run-Cell commands
│   ├── codelens.ts           "▶ Run cell" lens above each cell
│   ├── decorations.ts        Cell separator decorations
│   ├── context.ts            Sets editor context keys (e.g. ggsql.hasCodeCells)
│   └── types.ts              Shared interfaces
├── syntaxes/
│   └── ggsql.tmLanguage.json TextMate grammar (used for tokenization in VS Code)
├── examples/                 Sample .ggsql files
├── resources/                Static assets bundled with the extension
└── ggsql-0.1.0.vsix          Packaged extension (build artifact, may be stale)
```

## File extensions and language ID

`package.json` registers `id: ggsql` for `.ggsql`, `.ggsql.sql`, and `.gsql`. The TextMate grammar at `syntaxes/ggsql.tmLanguage.json` provides tokenization. Tree-sitter highlights — used by editors that prefer the grammar package directly — live in [`/tree-sitter-ggsql/queries/highlights.scm`](../tree-sitter-ggsql/queries/highlights.scm).

## Commands and keybindings

Declared in `package.json` and wired up in `extension.ts`:

| Command | Default key | Purpose |
| --- | --- | --- |
| `ggsql.runCurrentAdvance` | Cmd/Ctrl+Enter, Shift+Enter | Run current cell, advance to next |
| `ggsql.runQuery` | Cmd/Ctrl+Shift+Enter | Run current cell only |
| `ggsql.runNextCell` | — | Run the next cell |
| `ggsql.runCellsAbove` | — | Run all cells above the cursor |
| `ggsql.sourceCurrentFile` | — | Run the entire file (also exposed as the editor "Run" button) |

Cells are detected by `cellParser.ts`; `codelens.ts` puts a CodeLens above each cell.

## Positron integration

The extension declares `contributes.languageRuntimes` for `ggsql` (see `package.json`) and depends on `@posit-dev/positron`. When activated under Positron, `manager.ts`:

1. Discovers a `ggsql-jupyter` binary via, in order: the `ggsql.kernelPath` setting, an installed Jupyter kernelspec named `ggsql`, or `ggsql-jupyter` on `PATH`.
2. Registers it as a Positron language runtime so `▶ Run` and the Console route to the kernel.
3. Routes plot output to Positron's Plot pane via metadata coming back from the kernel (`output_location: "plot"`).

Outside Positron, the same commands fall back to writing query output to the active terminal.

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

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/ggsql-jupyter/CLAUDE.md`](../ggsql-jupyter/CLAUDE.md) — the kernel this extension drives.
- [`/tree-sitter-ggsql/CLAUDE.md`](../tree-sitter-ggsql/CLAUDE.md) — grammar that powers more advanced editor highlighting.
- [`/doc/get_started/tooling.qmd`](../doc/get_started/tooling.qmd) — user-facing tooling docs.
