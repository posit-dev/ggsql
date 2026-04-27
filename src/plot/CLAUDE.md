# `src/plot/` — Visualization AST

The typed Abstract Syntax Tree representing a parsed ggsql visualization spec. Built by the parser (`src/parser/`), consumed by the executor (`src/execute/`) and writers (`src/writer/`).

For ggsql language semantics, see [`/doc/syntax/`](../../doc/syntax/). This file describes the *Rust types*, not the language they model.

## Top-level types

Defined in `main.rs` (the `Plot` struct + `Labels`) and `types.rs` (input/value types). All are re-exported through `mod.rs`.

| Type | Where | Role |
| --- | --- | --- |
| `Plot` | `main.rs` | Root spec: `global_mappings`, optional `source`, `layers`, `scales`, optional `facet`, `project`, `labels`, plus a derived `aesthetic_context`. |
| `Labels` | `main.rs` | Title/subtitle/axis/aesthetic labels merged from all `LABEL` clauses. |
| `Mappings` | `types.rs` | Unified aesthetic mapping (`wildcard: bool` + `aesthetics: HashMap<String, AestheticValue>`); used for both global and per-layer mappings. |
| `AestheticValue` | `types.rs` | One side of a mapping: column reference, literal, annotation literal, or null. |
| `ParameterValue` | `types.rs` | Value used in `SETTING` clauses: string / number / boolean / array. |
| `DataSource` | `types.rs` | `Identifier`, `FilePath`, or `Annotation` (PLACE) — the right-hand side of `FROM`. |
| `SqlExpression` | `types.rs` | Captured raw SQL fragment (e.g. for `FILTER`). |
| `Schema` / `ColumnInfo` / `ArrayElement` | `types.rs` | Schema info computed from data; carries dtype, discreteness, min/max. |
| `AestheticContext` | `aesthetic.rs` | Coord-aware mapping of user aesthetic names (`x`, `y`) to internal names (`pos1`, `pos2`). Computed once and stored on `Plot` so executor and writers agree. |

`aesthetic.rs` also exports the position-aesthetic predicates (`is_position_aesthetic`, `MATERIAL_AESTHETICS`, `POSITION_SUFFIXES`) that the rest of the pipeline reaches for.

## Subdirectory map

```
plot/
├── aesthetic.rs       Aesthetic naming + classification (position/facet/material)
├── main.rs            Plot, Labels
├── types.rs           Mappings, AestheticValue, DataSource, Schema, ArrayElement, …
├── facet/             FACET clause
├── layer/             DRAW clause (layers)
│   ├── geom/          one file per layer type (point, line, bar, …)
│   ├── orientation.rs Layer transposition (horizontal vs vertical)
│   └── position/      identity, stack, dodge, jitter
├── projection/        PROJECT clause
│   └── coord/         cartesian, polar
└── scale/             SCALE clause
    ├── scale_type/    binned, continuous, discrete, identity, ordinal
    └── transform/     identity, log, sqrt, asinh, exp, square, pseudo_log,
                       date, datetime, time, integer, bool, string
```

### `layer/`

`Layer` itself lives in `layer/mod.rs`. The two big substructures:

- **`layer/geom/`** — one `.rs` per geom plus `mod.rs` (registry) and `types.rs`. The current set: `area`, `arrow`, `bar`, `boxplot`, `density`, `errorbar`, `histogram`, `line`, `path`, `point`, `polygon`, `ribbon`, `rule`, `segment`, `smooth`, `text`, `tile`, `violin`. Each implements `GeomTrait` (declares supported aesthetics, parameters, defaults, and any stat behaviour). User-facing semantics live in [`/doc/syntax/layer/type/`](../../doc/syntax/layer/type/).
- **`layer/position/`** — `identity`, `stack`, `dodge`, `jitter`. Each implements `PositionTrait`. Semantics: [`/doc/syntax/layer/position/`](../../doc/syntax/layer/position/).

`layer/orientation.rs` decides whether a layer is "transposed" (horizontal bar vs vertical bar etc.) based on the mappings + geom type.

### `scale/`

`Scale` in `types.rs`, the kind enum + traits in `scale_type/`, the transformation logic in `transform/`. Other top-level files in `scale/`:

- `breaks.rs` — break/tick computation for axes.
- `colour.rs` — colour aesthetic helpers, gradient interpolation.
- `linetype.rs`, `shape.rs` — categorical aesthetic mappings.
- `palettes.rs` — built-in colour palettes (large file, mostly data tables).

Scale type / transform docs: [`/doc/syntax/scale/type/`](../../doc/syntax/scale/type/) and [`/doc/syntax/scale/aesthetic/`](../../doc/syntax/scale/aesthetic/).

### `facet/` and `projection/`

Smaller subsystems. Each has a `types.rs` (data structure) and `resolve.rs` (logic that runs during execution). `projection/coord/` currently has `cartesian` and `polar`. Docs: [`/doc/syntax/clause/facet.qmd`](../../doc/syntax/clause/facet.qmd), [`/doc/syntax/coord/`](../../doc/syntax/coord/).

## Adding a new geom / scale type / coord

1. **Geom**: add `layer/geom/<name>.rs` implementing `GeomTrait`; register it in `layer/geom/mod.rs` and `layer/geom/types.rs` (`GeomType` enum). Update `src/parser/builder.rs` so the parser recognises the keyword. Add a doc page under `/doc/syntax/layer/type/`.
2. **Scale type**: add `scale/scale_type/<name>.rs` implementing `ScaleTypeTrait`; register in `scale/scale_type/mod.rs` (`ScaleTypeKind`). Update parser if a new keyword is needed. Doc page under `/doc/syntax/scale/type/`.
3. **Transform**: add `scale/transform/<name>.rs` implementing `TransformTrait`; register in `scale/transform/mod.rs` and `ALL_TRANSFORM_NAMES`.
4. **Coord**: add `projection/coord/<name>.rs` implementing `CoordTrait`; register in `projection/coord/mod.rs` (`CoordKind`). Doc page under `/doc/syntax/coord/`.

In every case the writer (`src/writer/vegalite/`) needs corresponding rendering support — see [`src/writer/vegalite/CLAUDE.md`](../writer/vegalite/CLAUDE.md).

## See also

- [`src/CLAUDE.md`](../CLAUDE.md) — module overview for the whole crate.
- [`src/writer/vegalite/CLAUDE.md`](../writer/vegalite/CLAUDE.md) — how the AST is rendered.
- [`/doc/syntax/`](../../doc/syntax/) — authoritative ggsql syntax reference.
