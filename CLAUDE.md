# VizQL: SQL Visualization Grammar

A SQL extension for declarative data visualization based on Grammar of Graphics principles.

## Core Concept

```sql
SELECT date, revenue, region FROM sales WHERE year = 2024
VISUALISE AS PLOT
WITH line USING x = date, y = revenue, color = region
LABEL title = 'Sales by Region'
THEME minimal
```

**Key Principles:**
- Everything before `VISUALISE AS` is standard SQL
- Separation of concerns: data (SQL) vs visualization (VISUALISE)
- Terminal operation: produces visual output, not relational data
- Based on Grammar of Graphics (ggplot2, Vega-Lite concepts)
- **Multiple outputs**: Supports multiple VISUALISE statements in a single query
- **Spelling flexibility**: Both `VISUALISE` (British) and `VISUALIZE` (American) spellings supported

## Grammar Structure

```sql
SELECT <columns> FROM <table> [WHERE ...] [GROUP BY ...]
[VISUALISE|VISUALIZE AS <type>
  [WITH <geom> USING <aesthetics> [AS <name>]]...
  [SCALE <aesthetic> USING <properties>]...
  [FACET {<vars> BY <vars> | WRAP <vars>} [USING scales = <sharing>]]
  [COORD USING type = <type> [, <options>]]
  [LABEL [title = <string>] [x = <string>] [y = <string>] ...]
  [GUIDE <aesthetic> USING <properties>]...
  [THEME <name> [USING <overrides>]]
]...
```

**Visualization Types:**
- `PLOT` - Standard Grammar of Graphics visualizations
- `TABLE` - Tabular data output
- `MAP` - Geographic/spatial visualizations

## Keywords

**Repeatable:** WITH, SCALE, GUIDE, LABEL, VISUALISE/VISUALIZE
**Singular:** PLOT/TABLE/MAP, FACET, COORD, THEME

## Core Components

### WITH Clause (Layers)
Defines geometric objects with aesthetic mappings. Layers render bottom-to-top.

```sql
WITH <geom_type> USING
    <aesthetic> = <column_name | literal>,
    [<geom_parameter> = <value>, ...]
    [AS <layer_name>]
```

**Literal vs Column Rules:**
- Quoted strings/numbers = literals: `color = 'blue'`, `size = 3`
- Unquoted identifiers = columns: `color = region`, `x = date`

**Common Geoms:**
- Basic: `point`, `line`, `bar`, `area`, `tile`, `ribbon`
- Statistical: `histogram`, `density`, `smooth`, `boxplot`
- Annotation: `text`, `segment`, `hline`, `vline`

**Common Aesthetics:**
- Position: `x`, `y`, `xmin`, `xmax`, `ymin`, `ymax`
- Color: `color`, `fill`, `alpha`
- Size/Shape: `size`, `shape`, `linetype`, `linewidth`
- Text: `label`, `family`, `fontface`

### SCALE Clause
Maps data values to visual values.

```sql
SCALE <aesthetic> USING
    [type = <scale_type>]
    [limits = [min, max]]
    [breaks = <array | interval>]
    [palette = <name>]
```

**Scale Types:**
- Continuous: `linear` (default numeric), `log10`, `sqrt`, `reverse`
- Discrete: `categorical` (default string), `ordinal`
- Temporal: `date`, `datetime`
- Color: `viridis`, `plasma`, `categorical`, `diverging`

### FACET Clause
Creates small multiples.

```sql
-- Grid layout
FACET <row_vars> BY <col_vars> [USING scales = <sharing>]
-- Wrapped layout
FACET WRAP <vars> [USING scales = <sharing>]
```

**Scale Sharing:** `'fixed'` (default), `'free'`, `'free_x'`, `'free_y'`

### Other Clauses

**COORD:** Coordinate transformations
- `cartesian` (default), `polar`, `flip`, `fixed`

**LABEL:** Text labels
- `title`, `subtitle`, `x`, `y`, `<aesthetic>`, `caption`, `tag`

**GUIDE:** Legend/axis configuration
- Types: `legend`, `colorbar`, `axis`, `none`

**THEME:** Visual styling
- Base themes: `minimal`, `classic`, `gray`, `bw`, `dark`, `void`

## Examples

### Basic Patterns

```sql
-- Scatter plot
SELECT x, y FROM data
VISUALISE AS PLOT
WITH point USING x = x, y = y

-- Line with points
WITH line USING x = date, y = value
WITH point USING x = date, y = value

-- Grouped bars
WITH bar USING x = category, y = value, fill = group, position = 'dodge'

-- Faceted plot
FACET WRAP region

-- Styled plot
LABEL title = 'My Plot', x = 'X Axis', y = 'Y Axis'
THEME minimal
```

### Complex Example

```sql
SELECT date, revenue, cost, region FROM sales
WHERE date >= '2023-01-01'
VISUALISE AS PLOT
WITH ribbon USING x = date, ymin = cost, ymax = revenue, fill = region, alpha = 0.3
WITH line USING x = date, y = revenue, color = region, size = 1.5
WITH point USING x = date, y = revenue, color = region, size = 2
SCALE x USING type = 'date', breaks = '2 months'
SCALE color USING palette = 'viridis'
FACET WRAP region
LABEL title = 'Revenue vs Cost by Region', x = 'Date', y = 'Amount'
GUIDE color USING position = 'right'
THEME minimal
```

### Multiple Outputs Example

```sql
-- Generate both a visualization and a data table from the same query
SELECT date, revenue, region FROM sales
WHERE year = 2024
VISUALISE AS PLOT
WITH line USING x = date, y = revenue, color = region
LABEL title = 'Sales Trends by Region'
THEME minimal
VISUALIZE AS TABLE
```

## Data Types

**Arrays:** `[0, 100]`, `['2023-01-01', '2024-01-01']`
**Strings:** `'date'`, `'viridis'`
**Numbers:** `2`, `0.7`, `45`
**Booleans:** `true`, `false`

## Validation

**Parse-time:** Syntax, required aesthetics, clause structure
**Execution-time:** Column existence, data types, scale compatibility

## Quick Reference

### Keyword Summary
| Keyword | Repeatable | Purpose |
|---------|------------|---------|
| `VISUALISE/VISUALIZE AS <type>` | Yes | Entry point (required, can have multiple) |
| `WITH` | Yes | Define layers |
| `SCALE` | Yes | Configure scales |
| `FACET` | No | Small multiples |
| `COORD` | No | Coordinate system |
| `LABEL` | No | Text labels |
| `GUIDE` | Yes | Legends/axes |
| `THEME` | No | Visual styling |

### Aesthetic Cheat Sheet
| Aesthetic | Example Values |
|-----------|----------------|
| `x`, `y` | Column name |
| `color` | Column or `'blue'` |
| `size` | Column or `3` |
| `alpha` | `0.5`, `0.7` |
| `linetype` | `'solid'`, `'dashed'` |

### Geom Requirements
| Geom | Required | Optional |
|------|----------|----------|
| `point` | `x`, `y` | `color`, `size`, `shape` |
| `line` | `x`, `y` | `color`, `linetype`, `size` |
| `bar` | `x`, `y` | `fill`, `color` |
| `histogram` | `x` | `fill`, `bins` |

## Design Principles

1. **Grammar of Graphics** - Composable layers, explicit mappings
2. **SQL-Native** - Natural extension, consistent patterns
3. **Separation of Concerns** - Data vs visual encoding
4. **Terminal Operation** - Produces visualization, not relation
5. **Explicit over Implicit** - Clear specifications

---

# Implementation Architecture

## System Overview

**VizQL** splits queries at `VISUALISE AS`:
- **SQL portion** → pluggable readers (DuckDB, PostgreSQL, CSV)
- **VISUALISE portion** → parsed into visualization specifications
- **Output** → pluggable writers (ggplot2, PNG, Vega-Lite, etc.)

## Architecture Flow

```
Query → Tree-sitter Splitter → Reader + Parser
                            ↓
                      Data + VizSpec
                            ↓
                      Writer → Output
```

## Technology Stack

**Language:** Rust (speed, safety, FFI, ecosystem) ✅
**Parser:** Tree-sitter (robust, incremental, editor support) ✅
**Grammar:** Simplified approach without external C scanner ✅
**Bindings:** Rust ✅, C ✅, Python ✅, Node.js ✅
**Data:** Polars DataFrames (planned)
**Readers:** DuckDB, PostgreSQL, SQLite, CSV (planned)
**Writers:** ggplot2, Vega-Lite, PNG renderer (planned)

## Current Project Structure

```
vizql/
├── Cargo.toml                  # Workspace configuration
├── README.md                   # Project documentation
├── CLAUDE.md                   # Complete specification (this file)
│
├── tree-sitter-vizql/         # Tree-sitter grammar package ✅ IMPLEMENTED
│   ├── grammar.js              # Grammar definition (281 lines, simplified)
│   ├── src/parser.c            # Generated parser (~50k lines)
│   ├── src/grammar.json        # Generated grammar representation
│   ├── bindings/
│   │   ├── rust/               # Rust language bindings ✅
│   │   ├── c/                  # C language bindings ✅
│   │   ├── python/             # Python language bindings ✅
│   │   │   ├── tree_sitter_vizql/  # Python package
│   │   │   └── tests/          # Python tests
│   │   └── node/               # Node.js language bindings ✅
│   │       ├── index.js        # Main Node.js entry point
│   │       ├── binding.cc      # C++ binding code
│   │       └── binding_test.js # Node.js tests
│   ├── setup.py                # Python package build
│   ├── pyproject.toml          # Python project configuration
│   ├── binding.gyp             # Node.js native binding build
│   ├── package.json            # NPM package configuration
│   └── tree-sitter.json        # Grammar metadata
│
└── src/                        # Main Rust library ✅ IMPLEMENTED
    ├── Cargo.toml              # Package configuration
    ├── lib.rs                  # Library entry point with public API
    ├── cli.rs                  # Command-line interface ✅ WORKING
    │
    └── parser/                 # Parsing subsystem ✅ IMPLEMENTED
        ├── mod.rs              # Public parsing API + query splitting
        ├── ast.rs              # AST type definitions (VizSpec, Layer, etc.)
        ├── builder.rs          # Tree-sitter CST → AST conversion
        ├── splitter.rs         # Regex-based query splitter
        └── error.rs            # Parse error types
```

## Implementation Status

### ✅ **Completed (Phase 1-2)**
- **Tree-sitter Grammar**: Simplified grammar without external scanner
- **Multiple Visualization Support**: Can parse multiple VISUALISE/VISUALIZE statements in one query
- **Spelling Flexibility**: Both British (VISUALISE) and American (VISUALIZE) spellings supported
- **Visualization Types**: Supports PLOT, TABLE, MAP types
- **Parser Integration**: Rust bindings working, all 20 tests passing
- **AST System**: Complete type definitions for VizSpec, VizType, Layer, Geom, etc.
- **Query Splitting**: Regex-based splitter separates SQL from VISUALISE portions (supports both spellings)
- **CLI Interface**: Working commands (parse, validate, exec, run) with multi-visualization support

### 🚧 **In Progress**
- **Documentation**: Updated specification and examples
- **Code Quality**: Stub implementations need full logic

### 📋 **Planned (Phase 3+)**
- **Readers**: Data source abstraction (DuckDB, PostgreSQL, CSV)
- **Engine**: Query execution pipeline, validation logic
- **Writers**: Output format generation (ggplot2, Vega-Lite, PNG)
- **Advanced Features**: Full AST builder implementation
- **Polish**: Optimization, CI/CD, comprehensive testing

### 🎯 **Current Capabilities**
```bash
# Working CLI commands:
cargo run -- parse "SELECT * FROM data VISUALISE AS PLOT WITH point USING x=x, y=y"

# Multiple visualizations in one query:
cargo run -- parse "SELECT * FROM data VISUALISE AS PLOT WITH point USING x=x, y=y VISUALIZE AS TABLE"

# American spelling support:
cargo run -- parse "SELECT * FROM data VISUALIZE AS MAP WITH tile USING x=x, y=y"

# JSON output for programmatic processing:
cargo run -- parse "SELECT * FROM data VISUALISE AS PLOT WITH line USING x=x, y=y" --format json
```

## Future Extensions

- **Interactivity:** `INTERACT` clause for tooltips, brushing
- **Animation:** `ANIMATE BY` for temporal transitions
- **Advanced Stats:** Regression lines, confidence bands
- **Plugin System:** Custom readers/writers
- **Python Bindings:** PyO3 integration
- **Editor Support:** LSP, syntax highlighting

---

This specification defines a complete grammar for SQL-native data visualization, designed to be implementation-agnostic while providing a unified, composable interface for creating rich visualizations.
