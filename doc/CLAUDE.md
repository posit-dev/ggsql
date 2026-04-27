# `doc/` — Quarto documentation site

Source for the public ggsql documentation at <https://ggsql.org>. **This directory is the authoritative reference for ggsql syntax and semantics.** Other CLAUDE.md files in the repo link here rather than restate.

If you are about to document a clause, layer type, scale, or aesthetic anywhere else: stop and put it here instead.

## Layout

```
doc/
├── _quarto.yml             Quarto project config (navbar, sidebar, theme, llms-txt)
├── _brand.yml, styles.scss, ggsql.xml   Branding + the ggsql syntax-highlight definition
├── index.qmd               Landing page
├── faq.qmd
├── get_started/
│   ├── installation.qmd    Platform-specific install + CLI quick reference
│   ├── first_plot.qmd      Tutorial: first visualization
│   ├── grammar.qmd         Grammar of Graphics conceptual foundation
│   ├── anatomy.qmd         Anatomy of a ggsql query
│   ├── tooling.qmd         VS Code / Positron / Jupyter / Python / R / CLI integrations
│   └── the_rest.qmd        Advanced features
├── syntax/
│   ├── index.qmd
│   ├── clause/             draw, facet, label, place, project, scale, visualise
│   ├── layer/
│   │   ├── type/           one .qmd per layer type (point, line, bar, …)
│   │   └── position/       identity, stack, dodge, jitter
│   ├── scale/
│   │   ├── type/           binned, continuous, discrete, identity, ordinal
│   │   └── aesthetic/      position, color, opacity, size, linewidth, shape, linetype, faceting
│   └── coord/              cartesian, polar
├── gallery/
│   ├── index.qmd
│   └── examples/           Runnable example queries
├── assets/, vendor/        Static resources
├── data/CSVs               Sample datasets used by examples (data.csv, sales.csv, …)
└── wasm/                   Built playground (copied from /ggsql-wasm/demo/dist by build-wasm.sh)
```

Generated artefacts not to edit by hand:

- `_site/` — Quarto build output. Not committed (see `.gitignore`).
- `*.quarto_ipynb*` — Quarto's intermediate notebook files for `.qmd` pages with executable cells. Regenerated on build.
- `wasm/` — produced by [`/ggsql-wasm/build-wasm.sh`](../ggsql-wasm/build-wasm.sh).

## Authoring conventions

- Pages are Quarto markdown (`.qmd`) — markdown plus YAML front-matter.
- Code blocks tagged ```` ```ggsql ```` use the syntax definition in `ggsql.xml` (referenced from `_quarto.yml` under `syntax-definitions`). Use that fence for runnable / illustrative ggsql snippets.
- Each clause page under `syntax/clause/` follows the same shape: short narrative, a "Clause syntax" code block listing all subclauses, then sections per subclause with examples. Mirror that shape when adding pages.
- Examples that need data reference the CSV files at the top of `doc/` (e.g. `data.csv`, `sales.csv`, `metrics.csv`, `timeseries.csv`).
- The site has `llms-txt: true` in `_quarto.yml`, so an `llms.txt` is generated for AI tooling — keep page titles and descriptions clean.

## Site structure (from `_quarto.yml`)

- **Navbar**: Get started · Syntax (drop-down per clause) · Gallery · FAQ · News · Playground · Python · R.
- **Syntax sidebar**: clauses → layers (type + position) → scales (type + aesthetic) → coordinate systems. New `.qmd` files in those folders are picked up automatically via `auto:` glob entries.
- **Get-started sidebar**: hand-ordered list (`installation` → `first_plot` → `grammar` → `anatomy` → `tooling` → `the_rest`).

## Build & preview

```sh
cd doc
quarto preview     # local server with hot reload
quarto render      # full site build → _site/
```

The site is published from the `gh-pages` branch (see `_quarto.yml`'s `repo-branch`). Quarto extensions live in `_extensions/`.

## What goes here vs. CLAUDE.md

- **Here**: anything a ggsql user might want to read — clause syntax, layer types, scales, aesthetics, examples, CLI usage, installation.
- **In CLAUDE.md files**: implementation details for contributors — module layouts, build pipelines, where to add a new geom, internal types.

If you find clause/layer/scale syntax described in any CLAUDE.md file, that is a duplication bug — move it here and link from there.

## See also

- [`/CLAUDE.md`](../CLAUDE.md) — workspace overview.
- [`/ggsql-wasm/CLAUDE.md`](../ggsql-wasm/CLAUDE.md) — how the embedded playground at `wasm/` is built.
