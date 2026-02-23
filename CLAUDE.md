# ggsql System Architecture & Implementation Summary

## Overview

**ggsql** is a SQL extension for declarative data visualization based on Grammar of Graphics principles. It allows users to combine SQL data queries with visualization specifications in a single, composable syntax.

**Core Innovation**: ggsql extends standard SQL with a `VISUALISE` clause that separates data retrieval (SQL) from visual encoding (Grammar of Graphics), enabling terminal visualization operations that produce charts instead of relational data.

```sql
SELECT date, revenue, region FROM sales WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
SCALE x VIA date
SCALE y FROM [0, 100000]
LABEL title => 'Sales by Region', x => 'Date', y => 'Revenue'
THEME minimal
```

**Statistics**:

- ~7,500 lines of Rust code (including PROJECT implementation)
- 507-line Tree-sitter grammar (simplified, no external scanner)
- Full bindings: Rust, C, Python, Node.js with tree-sitter integration
- Syntax highlighting support via Tree-sitter queries
- 916 total tests (174 parser tests, comprehensive builder and integration tests)
- End-to-end working pipeline: SQL → Data → Visualization
- Projectinate transformations: Cartesian, Flip, Polar
- VISUALISE FROM shorthand syntax with automatic SELECT injection

---

## Global Mapping Feature

ggsql supports global aesthetic mappings at the VISUALISE level that apply to all layers:

### Explicit Global Mapping

Map columns to specific aesthetics at the VISUALISE level:

```sql
SELECT * FROM sales WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
DRAW point
-- Both layers inherit x, y, and color mappings
```

### Implicit Global Mapping

Use column names directly when they match aesthetic names:

```sql
SELECT x, y FROM data
VISUALISE x, y
DRAW point
-- Equivalent to: VISUALISE x AS x, y AS y
```

### Wildcard Mapping

Map all columns automatically (resolved at execution time):

```sql
SELECT * FROM data
VISUALISE *
DRAW point
-- All columns mapped to aesthetics with matching names
```

### VISUALISE FROM Shorthand

Direct visualization from tables/CTEs (auto-injects `SELECT * FROM`):

```sql
-- Direct table visualization
VISUALISE FROM sales
DRAW bar MAPPING region AS x, total AS y

-- CTE visualization
WITH monthly_totals AS (
    SELECT DATE_TRUNC('month', sale_date) as month, SUM(revenue) as total
    FROM sales
    GROUP BY month
)
VISUALISE FROM monthly_totals
DRAW line MAPPING month AS x, total AS y
```

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                       ggsql Query                            │
│  "SELECT ... FROM ... WHERE ... VISUALISE x, y DRAW ..."     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │        SourceTree             │
         │  (Parse once, reuse CST)      │
         │  • extract_sql()              │
         │  • extract_visualise()        │
         └───────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  ┌─────────────┐        ┌──────────────┐
  │  SQL Part   │        │   VIZ Part   │
  │ "SELECT..." │        │ "VISUALISE..." │
  └──────┬──────┘        └──────┬───────┘
         │                      │
         ▼                      ▼
  ┌─────────────┐        ┌──────────────┐
  │   Reader    │        │    Parser    │
  │  (DuckDB,   │        │ (tree-sitter)│
  │   Postgres) │        │   → AST      │
  └──────┬──────┘        └──────┬───────┘
         │                      │
         ▼                      │
  ┌─────────────┐               │
  │  DataFrame  │               │
  │  (Polars)   │               │
  └──────┬──────┘               │
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │      Writer          │
         │  (Vega-Lite, ggplot) │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │   Visualization      │
         │  (JSON, PNG, HTML)   │
         └─────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: SQL handles data retrieval, VISUALISE handles visual encoding
2. **Pluggable Architecture**: Readers and Writers are trait-based, enabling multiple backends
3. **Grammar of Graphics**: Composable layers, explicit aesthetic mappings, scale transformations
4. **Terminal Operation**: Produces visualizations, not relational data (like SQL's `SELECT`)
5. **Type Safety**: Strong typing through AST with Rust's type system

---

## Public API

### Quick Start

```rust
use ggsql::reader::{DuckDBReader, Reader};
use ggsql::writer::VegaLiteWriter;

// Create a reader
let reader = DuckDBReader::from_connection_string("duckdb://memory")?;

// Execute the ggsql query
let spec = reader.execute(
    "SELECT x, y FROM data VISUALISE x, y DRAW point"
)?;

// Render to Vega-Lite JSON
let writer = VegaLiteWriter::new();
let json = writer.render(&spec)?;
```

### Core Functions

| Function                | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `reader.execute(query)` | Main entry point: parse, execute SQL, resolve mappings |
| `writer.render(spec)`   | Generate output from a Spec                            |
| `validate(query)`       | Validate syntax + semantics, inspect query structure   |

### Key Types

**`Validated`** - Result of `validate()`:

- `has_visual()` - Whether query has VISUALISE clause
- `sql()` - The SQL portion (before VISUALISE)
- `visual()` - The VISUALISE portion (raw text)
- `tree()` - CST for advanced inspection
- `valid()` - Whether query is valid
- `errors()` - Validation errors
- `warnings()` - Validation warnings

**`Spec`** - Result of `reader.execute()`, ready for rendering:

- `plot()` - Resolved plot specification
- `metadata()` - Rows, columns, layer count
- `warnings()` - Validation warnings from execution
- `data()` / `layer_data(i)` / `stat_data(i)` - Access DataFrames
- `sql()` / `visual()` / `layer_sql(i)` / `stat_sql(i)` - Query introspection

**`Metadata`**:

- `rows` - Number of rows in primary data
- `columns` - Column names
- `layer_count` - Number of layers

### Reader & Writer

**Reader trait** (data source abstraction):

- `execute_sql(sql)` - Run SQL, return DataFrame
- `register(name, df)` - Register DataFrame as table
- `unregister(name)` - Unregister a previously registered table
- Implementation: `DuckDBReader`

**Writer trait** (output format abstraction):

- `write(spec, data)` - Generate output string
- Implementation: `VegaLiteWriter` (Vega-Lite v6 JSON)

For detailed API documentation, see [`src/doc/API.md`](src/doc/API.md).

---

## Component Breakdown

### 1. Parser Module (`src/parser/`)

**Responsibility**: Split queries and parse visualization specifications into typed AST.

#### SourceTree (`source_tree.rs`)

**Parse-once architecture** that eliminates duplicate parsing throughout the pipeline.

**Core Design**:

- Wraps tree-sitter `Tree` + source text + `Language`
- Parses query once, reuses CST for all operations
- Declarative tree-sitter query API instead of manual tree walking
- Lazy extraction methods for SQL and VISUALISE portions

**High-Level Query API**:

- `find_node(query)` - Find first matching node via tree-sitter query
- `find_nodes(query)` - Find all matching nodes
- `find_text(query)` - Extract text of first match
- `find_texts(query)` - Extract text of all matches

**Lazy Extraction Methods**:

- `extract_sql()` - Lazily extract SQL portion (before VISUALISE)
- `extract_visualise()` - Lazily extract VISUALISE portion
- Both methods use declarative tree-sitter queries
- Handles VISUALISE FROM by automatically injecting `SELECT * FROM <source>`

#### Tree-sitter Integration (`mod.rs`)

- Uses `tree-sitter-ggsql` grammar (507 lines, simplified approach)
- Parses **full query** (SQL + VISUALISE) into concrete syntax tree (CST)
- Grammar supports: PLOT/TABLE/MAP types, DRAW/SCALE/FACET/PROJECT/LABEL/THEME clauses
- British and American spellings: `VISUALISE` / `VISUALIZE`
- **SQL portion parsing**: Basic SQL structure (SELECT, WITH, CREATE, INSERT, subqueries)
- **Recursive subquery support**: Fully recursive grammar for complex SQL

**Grammar Structure** (`tree-sitter-ggsql/grammar.js`):

Key grammar rules:

- `query`: Root node containing SQL + VISUALISE portions
- `sql_portion`: Zero or more SQL statements before VISUALISE
- `with_statement`: WITH clause with optional trailing SELECT (compound statement)
- `subquery`: Fully recursive subquery rule supporting nested parentheses
- `visualise_statement`: VISUALISE clause with optional global mappings and FROM source

**Critical Grammar Features**:

1. **Compound WITH statements**: `WITH cte AS (...) SELECT * FROM cte` parses as single statement
2. **Recursive subqueries**: `SELECT * FROM (SELECT * FROM (VALUES (1, 2)))` fully supported
3. **VISUALISE FROM extraction**: Grammar identifies FROM identifier for SELECT injection

```rust
pub fn parse_query(query: &str) -> Result<Vec<Plot>> {
    // Parse once with SourceTree
    let source_tree = SourceTree::new(query)?;

    // Validate query structure
    source_tree.validate()?;

    // Build AST from parse tree
    let specs = builder::build_ast(&source_tree)?;
    Ok(specs)
}
```

#### Plot Types (`plot.rs`)

Core data structures representing visualization specifications:

```rust
pub struct Plot {
    pub global_mapping: GlobalMapping, // From VISUALISE clause: Empty, Wildcard, or Mappings
    pub source: Option<String>,        // FROM source (for VISUALISE FROM)
    pub layers: Vec<Layer>,            // DRAW clauses
    pub scales: Vec<Scale>,            // SCALE clauses
    pub facet: Option<Facet>,          // FACET clause
    pub project: Option<Project>,          // PROJECT clause
    pub labels: Option<Labels>,        // LABEL clause
    pub theme: Option<Theme>,          // THEME clause
}

/// Global mapping specification from VISUALISE clause
pub enum GlobalMapping {
    Empty,                           // No mapping: VISUALISE
    Wildcard,                        // Wildcard: VISUALISE *
    Mappings(Vec<GlobalMappingItem>), // List: VISUALISE x, y, date AS x
}

pub enum GlobalMappingItem {
    Explicit { column: String, aesthetic: String }, // date AS x
    Implicit { name: String },                      // x (maps column x to aesthetic x)
}

pub struct Layer {
    pub geom: Geom,                  // Geometric object type
    pub aesthetics: HashMap<String, AestheticValue>,  // Aesthetic mappings (from MAPPING)
    pub parameters: HashMap<String, ParameterValue>,  // Geom parameters (from SETTING)
    pub filter: Option<FilterExpression>,  // Layer filter (from FILTER)
    pub partition_by: Vec<String>,   // Grouping columns (from PARTITION BY)
}

pub enum Geom {
    // Basic geoms
    Point, Line, Path, Bar, Col, Area, Tile, Polygon, Ribbon,
    // Statistical geoms
    Histogram, Density, Smooth, Boxplot, Violin,
    // Annotation geoms
    Text, Label, Segment, Arrow, HLine, VLine, AbLine, ErrorBar,
}

pub enum AestheticValue {
    Column(String),                  // Unquoted column reference: revenue AS x
    Literal(ParameterValue),         // Quoted literal: 'value' AS fill
}

pub enum ParameterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ParameterValue>),      // Array values for properties
}

pub struct Scale {
    pub aesthetic: String,
    pub scale_type: Option<ScaleType>,
    pub properties: HashMap<String, Value>,
}

pub enum ScaleType {
    // Continuous scales
    Linear, Log10, Log, Log2, Sqrt, Reverse,
    // Discrete scales
    Ordinal, Categorical,
    // Temporal scales
    Date, DateTime, Time,
    // Color palettes
    Viridis, Plasma, Magma, Inferno, Cividis, Diverging, Sequential,
}

pub enum Facet {
    /// FACET variables (wrap layout)
    Wrap {
        variables: Vec<String>,
        scales: FacetScales,
        properties: HashMap<String, ParameterValue>,  // From SETTING clause
    },
    /// FACET rows BY cols (grid layout)
    Grid {
        rows: Vec<String>,
        cols: Vec<String>,
        scales: FacetScales,
        properties: HashMap<String, ParameterValue>,  // From SETTING clause
    },
}

pub enum FacetScales {
    Fixed,      // 'fixed' - same scales across all facets
    Free,       // 'free' - independent scales for each facet
    FreeX,      // 'free_x' - independent x-axis, shared y-axis
    FreeY,      // 'free_y' - independent y-axis, shared x-axis
}

pub struct Project {
    pub project_type: ProjectType,
    pub properties: HashMap<String, ParameterValue>,
}

pub enum ProjectType {
    Cartesian,  // Standard x/y projectinates
    Polar,      // Polar projectinates (pie charts, rose plots)
    Flip,       // Flipped Cartesian (swaps x and y)
    Fixed,      // Fixed aspect ratio
    Trans,      // Transformed projectinates
    Map,        // Map projections
    QuickMap,   // Quick map approximation
}

pub struct Labels {
    pub labels: HashMap<String, String>,  // label type → text
}

pub struct Theme {
    pub style: Option<String>,
    pub properties: HashMap<String, ParameterValue>,
}
```

**Key Methods**:

**Plot methods:**

- `Plot::new()` - Create a new empty Plot
- `Plot::with_global_mapping(mapping)` - Create Plot with a global mapping
- `Plot::find_scale(aesthetic)` - Look up scale specification for an aesthetic
- `Plot::find_guide(aesthetic)` - Find a guide specification for an aesthetic
- `Plot::has_layers()` - Check if Plot has any layers
- `Plot::layer_count()` - Get the number of layers

**Layer methods:**

- `Layer::new(geom)` - Create a new layer with a geom
- `Layer::with_filter(filter)` - Set the layer filter (builder pattern)
- `Layer::with_aesthetic(aesthetic, value)` - Add an aesthetic mapping (builder pattern)
- `Layer::with_parameter(parameter, value)` - Add a geom parameter (builder pattern)
- `Layer::get_column(aesthetic)` - Get column name for an aesthetic (if mapped to column)
- `Layer::get_literal(aesthetic)` - Get literal value for an aesthetic (if literal)
- `Layer::validate_required_aesthetics()` - Validate that required aesthetics are present for the geom type

**Type conversions:**

- JSON serialization/deserialization for all AST types via Serde

#### Error Handling (`error.rs`)

**Detailed parse error reporting** with location information:

```rust
pub struct ParseError {
    pub message: String,      // Error description
    pub line: usize,          // Line number (0-based)
    pub column: usize,        // Column number (0-based)
    pub context: String,      // Parsing context
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {}, column {} (in {})",
            self.message,
            self.line + 1,    // Display as 1-based
            self.column + 1,
            self.context
        )
    }
}
```

**Benefits**:

- Precise error location reporting for user-friendly diagnostics
- Context information helps identify where parsing failed
- Converts to GgsqlError for unified error handling

#### Main Error Type (`lib.rs`)

The library uses a unified error type for all operations:

```rust
pub enum GgsqlError {
    ParseError(String),        // Query parsing errors
    ValidationError(String),   // Semantic validation errors
    ReaderError(String),       // Data source errors
    WriterError(String),       // Output generation errors
    InternalError(String),     // Unexpected internal errors
}

pub type Result<T> = std::result::Result<T, GgsqlError>;
```

**Error Types**:

- **ParseError**: Tree-sitter parsing failures, invalid syntax
- **ValidationError**: Semantic errors (missing required aesthetics, type mismatches)
- **ReaderError**: Database connection failures, SQL errors, missing tables/columns
- **WriterError**: Vega-Lite generation errors, file I/O errors
- **InternalError**: Unexpected conditions that should not occur

---

### 2. Reader Module (`src/reader/`)

**Responsibility**: Abstract data source access, execute SQL, return DataFrames.

#### Reader Trait (`mod.rs`)

```rust
pub trait Reader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame>;
    fn supports_query(&self, sql: &str) -> bool;
}
```

#### DuckDB Reader (`duckdb.rs`)

**Current Production Reader** - Fully implemented and tested.

**Features**:

- In-memory databases: `duckdb://memory`
- File-based databases: `duckdb://path/to/file.db`
- SQL execution → Polars DataFrame conversion
- Comprehensive type handling

**Connection Parsing** (`connection.rs`):

```rust
pub fn parse_connection_string(uri: &str) -> Result<ConnectionInfo> {
    // duckdb://memory → In-memory database
    // duckdb:///path/to/file.db → File-based database
}
```

#### Planned Readers (Not Yet Implemented)

The codebase includes connection string parsing and feature flags for additional readers, but they are not yet implemented:

- **PostgreSQL Reader** (`postgres://...`)
  - Feature flag: `postgres`
  - Connection string parsing exists in `connection.rs`
  - Reader implementation: Not yet available

- **SQLite Reader** (`sqlite://...`)
  - Feature flag: `sqlite`
  - Connection string parsing exists in `connection.rs`
  - Reader implementation: Not yet available

**Current Status**: Only DuckDB reader is fully implemented and production-ready.

---

### 3. Writer Module (`src/writer/`)

**Responsibility**: Convert DataFrame + Plot → output format (JSON, PNG, R code, etc.)

#### Writer Trait (`mod.rs`)

```rust
pub trait Writer {
    fn write(&self, df: &DataFrame, spec: &Plot) -> Result<String>;
    fn file_extension(&self) -> &str;
}
```

#### Vega-Lite Writer (`vegalite.rs`)

**Current Production Writer** - Fully implemented and tested.

**Features**:

- Converts Plot → Vega-Lite JSON specification
- Multi-layer composition support
- Scale type → Vega field type mapping
- Faceting (wrap and grid layouts)
- Axis label customization
- Inline data embedding

**Architecture**:

```rust
impl Writer for VegaLiteWriter {
    fn write(&self, df: &DataFrame, spec: &Plot) -> Result<String> {
        // 1. Convert DataFrame to JSON values
        let data_values = self.dataframe_to_json(df)?;

        // 2. Build Vega-Lite spec
        let mut vl_spec = json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "data": {"values": data_values},
            "width": 600,
            "autosize": {"type": "fit", "contains": "padding"}
        });

        // 3. Handle single vs multi-layer
        if spec.layers.len() == 1 {
            // Single layer: flat structure
            vl_spec["mark"] = self.geom_to_mark(&spec.layers[0].geom);
            vl_spec["encoding"] = self.build_encoding(&spec.layers[0], df, spec)?;
        } else {
            // Multi-layer: layered structure
            let layers: Vec<Value> = spec.layers.iter()
                .map(|layer| {
                    let mut layer_spec = json!({
                        "mark": self.geom_to_mark(&layer.geom),
                        "encoding": self.build_encoding(layer, df, spec)?
                    });
                    // Apply axis labels to each layer
                    apply_axis_labels(&mut layer_spec, &spec.labels);
                    Ok(layer_spec)
                })
                .collect::<Result<Vec<_>>>()?;
            vl_spec["layer"] = json!(layers);
        }

        // 4. Add faceting, title, etc.
        self.add_faceting(&mut vl_spec, spec)?;
        if let Some(labels) = &spec.labels {
            if let Some(title) = labels.labels.get("title") {
                vl_spec["title"] = json!(title);
            }
        }

        Ok(serde_json::to_string_pretty(&vl_spec)?)
    }
}
```

---

### 4. REST API (`src/rest.rs`)

**Responsibility**: HTTP interface for executing ggsql queries.

**Technology**: Axum web framework with CORS support

**Endpoints**:

```rust
// POST /api/v1/query - Execute ggsql query
// Request:
{
  "query": "SELECT ... VISUALISE ...",
  "reader": "duckdb://memory",  // optional, default
  "writer": "vegalite"            // optional, default
}

// Response (success):
{
  "status": "success",
  "data": {
    "spec": { /* Vega-Lite JSON */ },
    "metadata": {
      "rows": 100,
      "columns": ["date", "revenue", "region", "..."],
      "global_mapping": "Mappings",
      "layers": 2
    }
  }
}

// Response (error):
{
  "status": "error",
  "error": {
    "type": "ParseError",
    "message": "..."
  }
}
```

**Additional Endpoints**:

- `GET /` - Root endpoint (returns API information and status)
- `POST /api/v1/parse` - Parse query and return AST (debugging)
- `GET /api/v1/health` - Health check
- `GET /api/v1/version` - Version info

**CORS Configuration**: Allows cross-origin requests for web frontends

**CLI Options**:

```bash
# Basic usage
ggsql-rest --host 127.0.0.1 --port 3334

# With sample data (pre-loaded products, sales, employees tables)
ggsql-rest --load-sample-data

# Load custom data files (CSV, Parquet, JSON)
ggsql-rest --load-data data.csv --load-data other.parquet

# Configure CORS origins
ggsql-rest --cors-origin "http://localhost:5173,http://localhost:3000"
```

**Sample Data Loading**:

- `--load-sample-data`: Loads built-in sample data (products, sales, employees)
- `--load-data <file>`: Loads data from CSV, Parquet, or JSON files into in-memory database
- Multiple `--load-data` flags can be used to load multiple files
- Pre-loaded data persists for the lifetime of the server session

---

### 5. CLI (`src/cli.rs`)

**Responsibility**: Command-line interface for local query execution.

**Commands**:

```bash
# Parse query and show AST
ggsql parse "SELECT ... VISUALISE ..."

# Execute query and generate output
ggsql exec "SELECT ... VISUALISE ..." \
  --reader duckdb://memory \
  --writer vegalite \
  --output viz.vl.json

# Execute query from file
ggsql run query.sql --writer vegalite

# Validate query syntax
ggsql validate query.sql
```

**Features**:

- JSON or human-readable output formats
- File or stdin input
- Pluggable reader/writer selection
- Error reporting with context

---

### 6. Jupyter Kernel (`ggsql-jupyter/`)

**Responsibility**: Jupyter kernel for executing ggsql queries with rich inline visualizations.

**Features**:

- Execute ggsql queries directly in Jupyter notebooks
- Rich Vega-Lite visualizations rendered inline
- Persistent DuckDB session across cells
- Pure SQL support with HTML table output
- Interactive data exploration and visualization

**Architecture**:

The Jupyter kernel implements the Jupyter messaging protocol to:

1. Receive ggsql query code from notebook cells
2. Maintain a persistent in-memory DuckDB session
3. Execute queries using the ggsql engine
4. Return Vega-Lite JSON for visualization cells
5. Return HTML tables for pure SQL queries

**Installation**:

```bash
# Install from crates.io
cargo install ggsql-jupyter
ggsql-jupyter --install

# Or build from source
cargo build --release --package ggsql-jupyter
./target/release/ggsql-jupyter --install
```

**Usage**:

```sql
-- Cell 1: Create data
CREATE TABLE sales AS
SELECT * FROM (VALUES
    ('2024-01-01'::DATE, 100, 'North'),
    ('2024-01-02'::DATE, 120, 'South')
) AS t(date, revenue, region)

-- Cell 2: Visualize
SELECT * FROM sales
VISUALISE
DRAW line MAPPING date AS x, revenue AS y, region AS color
SCALE x VIA date
LABEL title => 'Sales Trends'
```

**Key Implementation Details**:

- Uses Jupyter messaging protocol (ZMQ)
- Supports `execute_request`, `kernel_info_request`, `shutdown_request`, `is_complete_request`
- Returns `display_data` messages with HTML-embedded Vega-Lite visualizations (using vega-embed)
- Maintains kernel state across cell executions

**Positron IDE Integration**:

The kernel includes enhanced support for Positron IDE:

- **`is_complete_request` handler**: Enables code completeness checking for multi-line input
- **HTML-based rendering**: Uses vega-embed to render Vega-Lite specs as HTML, providing broader compatibility
- **Plot pane routing**: Includes Positron-specific `"output_location": "plot"` metadata to automatically route visualizations to the Plots pane
- **Comm channels**: Implements Positron communication channels (comms) for variables, UI, and plot output

---

### 7. VS Code Extension (`ggsql-vscode/`)

**Responsibility**: Syntax highlighting for ggsql in Visual Studio Code and Positron IDE, with language runtime integration for Positron.

**Features**:

- Complete syntax highlighting for ggsql queries
- SQL keyword support (SELECT, FROM, WHERE, JOIN, WITH, etc.)
- ggsql clause highlighting (VISUALISE, SCALE, PROJECT, FACET, LABEL, etc.)
- Aesthetic highlighting (x, y, color, size, shape, etc.)
- String and number literals
- Comment support (`--` and `/* */`)
- Bracket matching and auto-closing
- `.gsql` file extension support
- Positron IDE language runtime integration

**Installation**:

```bash
# Install from VSIX file
code --install-extension ggsql-0.1.0.vsix

# Or from source
cd ggsql-vscode
npm install -g @vscode/vsce
vsce package
code --install-extension ggsql-0.1.0.vsix
```

**Implementation**:

- **TypeScript extension** with full Positron API integration
- Uses TextMate grammar (`syntaxes/ggsql.tmLanguage.json`)
- Tree-sitter syntax highlighting queries (`tree-sitter-ggsql/queries/highlights.scm`)
- Language configuration for bracket matching and comments
- File extension association (`.gsql`)
- SVG icon asset for branding

**Positron IDE Integration**:

When running in Positron IDE, the extension provides enhanced functionality:

- **Language runtime registration**: Registers ggsql as a language runtime, enabling direct query execution
- **Kernel discovery**: Automatically discovers the ggsql-jupyter kernel from:
  - Extension settings (`ggsql.kernelPath`)
  - Jupyter kernelspec directories
  - System PATH
- **Plot pane integration**: Visualizations are routed to Positron's Plots pane via kernel metadata

**Syntax Scopes**:

- `keyword.control.ggsql` - VISUALISE, DRAW, SCALE, PROJECT, etc.
- `keyword.other.sql` - SELECT, FROM, WHERE, etc.
- `entity.name.function.geom.ggsql` - point, line, bar, etc.
- `variable.parameter.aesthetic.ggsql` - x, y, color, size, etc.
- `constant.language.scale-type.ggsql` - linear, log10, date, etc.

---

### 8. Python Bindings (`ggsql-python/`)

**Responsibility**: Python bindings for ggsql, enabling Python users to create visualizations using ggsql's VISUALISE syntax.

**Features**:

- PyO3-based Rust bindings compiled to a native Python extension
- Two-stage API mirroring the Rust API: `reader.execute()` → `render()`
- DuckDB reader with DataFrame registration
- Custom Python reader support: any object with `execute_sql(sql) -> DataFrame` method
- Works with any narwhals-compatible DataFrame (polars, pandas, etc.)
- LazyFrames are collected automatically
- Returns native `altair.Chart` objects via `render_altair()` convenience function
- Query validation and introspection (SQL, layer queries, stat queries)

**Installation**:

```bash
# From source (requires Rust toolchain and maturin)
cd ggsql-python
pip install maturin
maturin develop
```

**API**:

```python
import ggsql
import polars as pl

# Create reader and register data
reader = ggsql.DuckDBReader("duckdb://memory")
df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
reader.register("data", df)

# Execute visualization
spec = reader.execute(
    "SELECT * FROM data VISUALISE x, y DRAW point"
)

# Inspect metadata
print(f"Rows: {spec.metadata()['rows']}")
print(f"Columns: {spec.metadata()['columns']}")
print(f"SQL: {spec.sql()}")

# Render to Vega-Lite JSON
writer = ggsql.VegaLiteWriter()
json_output = writer.render(spec)
```

**Convenience Function** (`render_altair`):

For quick visualizations without explicit reader setup:

```python
import ggsql
import polars as pl

df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

# Render DataFrame to Altair chart in one call
chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
chart.display()  # In Jupyter
```

**Query Validation**:

```python
# Validate syntax without execution
validated = ggsql.validate(
    "SELECT x, y FROM data VISUALISE x, y DRAW point"
)
print(f"Valid: {validated.valid()}")
print(f"Has VISUALISE: {validated.has_visual()}")
print(f"SQL portion: {validated.sql()}")
print(f"Errors: {validated.errors()}")
```

**Classes**:

| Class                      | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `DuckDBReader(connection)` | Database reader with DataFrame registration       |
| `VegaLiteWriter()`         | Vega-Lite JSON output writer                      |
| `Validated`                | Result of `validate()` with query inspection      |
| `Spec`                     | Result of `reader.execute()`, ready for rendering |

**Functions**:

| Function                 | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `validate(query)`        | Syntax/semantic validation with query inspection |
| `reader.execute(query)`  | Execute ggsql query, return Spec                 |
| `execute(query, reader)` | Execute with custom reader (bridge path)         |
| `render_altair(df, viz)` | Convenience: render DataFrame to Altair chart    |

**Spec Methods**:

| Method           | Description                                  |
| ---------------- | -------------------------------------------- |
| `render(writer)` | Generate Vega-Lite JSON                      |
| `metadata()`     | Get rows, columns, layer_count               |
| `sql()`          | Get the SQL portion                          |
| `visual()`       | Get the VISUALISE portion                    |
| `layer_count()`  | Number of DRAW layers                        |
| `data()`         | Get the main DataFrame                       |
| `layer_data(i)`  | Get layer-specific DataFrame (if filtered)   |
| `stat_data(i)`   | Get stat transform DataFrame (if applicable) |
| `layer_sql(i)`   | Get layer filter SQL (if applicable)         |
| `stat_sql(i)`    | Get stat transform SQL (if applicable)       |
| `warnings()`     | Get validation warnings                      |

**Custom Python Readers**:

Any Python object with an `execute_sql(sql: str) -> polars.DataFrame` method can be used as a reader:

```python
import ggsql
import polars as pl

class MyReader:
    """Custom reader that returns static data."""

    def execute_sql(self, sql: str) -> pl.DataFrame:
        return pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

# Use custom reader with ggsql.execute()
reader = MyReader()
spec = ggsql.execute(
    "SELECT * FROM data VISUALISE x, y DRAW point",
    reader
)
```

Required methods for custom readers (in addition to `execute_sql`):

- `register(name: str, df: polars.DataFrame, replace: bool = False) -> None` - Register a DataFrame as a table

Optional methods for custom readers:

- `unregister(name: str) -> None` - Unregister a previously registered table

Native readers (e.g., `DuckDBReader`) use an optimized fast path, while custom Python readers are automatically bridged via IPC serialization.

**Dependencies**:

- Python >= 3.10
- altair >= 5.0
- narwhals >= 2.15
- polars >= 1.0

---

## Feature Flags and Build Configuration

ggsql uses Cargo feature flags to enable optional functionality and reduce binary size.

### Available Features

**Default features** (`default = ["duckdb", "sqlite", "vegalite"]`):

- `duckdb` - DuckDB reader (fully implemented)
- `sqlite` - SQLite reader (planned, not implemented)
- `vegalite` - Vega-Lite writer (fully implemented)

**Reader features**:

- `duckdb` - Enable DuckDB database reader
- `postgres` - Enable PostgreSQL reader (planned, not implemented)
- `sqlite` - Enable SQLite reader (planned, not implemented)
- `all-readers` - Enable all reader implementations

**Writer features**:

- `vegalite` - Enable Vega-Lite JSON writer (default)
- `ggplot2` - Enable R/ggplot2 code generation (planned, not implemented)
- `plotters` - Enable plotters-based rendering (planned, not implemented)
- `all-writers` - Enable all writer implementations

**API features**:

- `rest-api` - Enable REST API server (`ggsql-rest` binary)
  - Includes: `axum`, `tokio`, `tower-http`, `tracing`, `duckdb`, `vegalite`
  - Required for building `ggsql-rest` server

**Python bindings**:

- `ggsql-python` - Python bindings via PyO3 (separate crate, not a feature flag)

### Building with Custom Features

```bash
# Minimal build (library only, no default features)
cargo build --no-default-features

# With specific features
cargo build --features "duckdb,vegalite"

# REST API server
cargo build --bin ggsql-rest --features rest-api

# All features
cargo build --all-features
```

### Feature Dependencies

**Feature flag → Dependencies mapping**:

- `duckdb` → `duckdb` crate
- `postgres` → `postgres` crate (future)
- `sqlite` → `rusqlite` crate (future)
- `rest-api` → `axum`, `tokio`, `tower-http`, `tracing`, `tracing-subscriber`
- `ggsql-python` → `pyo3`, `narwhals`, `altair` (separate workspace crate)

---

## Release & Distribution

### Cross-Platform Installers

ggsql uses [cargo-packager](https://github.com/crabnebula-dev/cargo-packager) to create native installers for Windows, macOS, and Linux. See [INSTALLERS.md](../INSTALLERS.md) for detailed build instructions.

**Supported Formats**:

- **Windows**: NSIS (.exe), MSI (.msi)
- **macOS**: DMG (.dmg)
- **Linux**: Debian (.deb)

**What Gets Packaged**:

- ✅ `ggsql` CLI binary only
- ❌ `ggsql-rest` API server (install separately with `cargo install ggsql --features rest-api`)
- ❌ `ggsql-jupyter` kernel (install separately with `cargo install ggsql-jupyter`)

### Release Process

**Creating a Release**:

1. **Update version** in workspace `Cargo.toml`
2. **Update CHANGELOG** (if exists) with release notes
3. **Test installers locally** (at least one platform):
   ```bash
   cd src
   cargo packager --release --formats nsis    # Windows
   cargo packager --release --formats dmg     # macOS
   cargo packager --release --formats deb     # Linux
   ```
4. **Create and push version tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
5. **GitHub Actions will automatically**:
   - Build installers for all platforms
   - Create GitHub Release
   - Attach installers as release assets
   - Generate release notes

**Manual Workflow Trigger** (for testing):
```bash
gh workflow run release-installers.yml
```

### Known Limitations

**Installers**:

- Installers are unsigned (users may see security warnings)
- macOS installers require users to run `xattr -cr /Applications/ggsql.app` if unsigned
- Windows SmartScreen may show "Windows protected your PC" warning
- Production releases should use code signing certificates

**License Display**:

- LICENSE.md must use ASCII quotes (not Unicode smart quotes) for proper Windows installer display
- Verify with: `cargo packager --release --formats nsis` and test installer UI

### Distribution Channels

**Current**:

- GitHub Releases (automated via GitHub Actions)
- Manual local builds via cargo-packager

**Future** (not yet implemented):

- Homebrew tap (macOS/Linux)
- Chocolatey (Windows)
- Scoop (Windows)
- apt repository (Debian/Ubuntu)
- crates.io (Rust library)
- PyPI (Python bindings)

---

## Grammar Deep Dive

### ggsql Grammar Structure

```sql
[SELECT ...] VISUALISE [<global_mapping>] [FROM <source>] [clauses]...
```

Where `<global_mapping>` can be:

- Empty: `VISUALISE` (layers must define all mappings)
- Mappings: `VISUALISE x, y, date AS x` (mixed implicit/explicit)
- Wildcard: `VISUALISE *` (map all columns)

### Clause Types

| Clause      | Repeatable | Purpose           | Example                                   |
| ----------- | ---------- | ----------------- | ----------------------------------------- |
| `VISUALISE` | ✅ Yes     | Entry point       | `VISUALISE date AS x, revenue AS y`       |
| `DRAW`      | ✅ Yes     | Define layers     | `DRAW line MAPPING date AS x, value AS y` |
| `SCALE`     | ✅ Yes     | Configure scales  | `SCALE x VIA date`                        |
| `FACET`     | ❌ No      | Small multiples   | `FACET region`                            |
| `PROJECT`   | ❌ No      | Coordinate system | `PROJECT flip` |
| `LABEL`     | ❌ No      | Text labels       | `LABEL title => 'My Chart', x => 'Date'`  |
| `THEME`     | ❌ No      | Visual styling    | `THEME minimal`                           |

### DRAW Clause (Layers)

**Syntax**:

```sql
DRAW <geom>
    [MAPPING <value> AS <aesthetic>, ...]
    [SETTING <param> => <value>, ...]
    [PARTITION BY <column>, ...]
    [FILTER <condition>]
```

All clauses (MAPPING, SETTING, PARTITION BY, FILTER) are optional.

**Geom Types**:

- **Basic**: `point`, `line`, `path`, `bar`, `col`, `area`, `tile`, `polygon`, `ribbon`
- **Statistical**: `histogram`, `density`, `smooth`, `boxplot`, `violin`
- **Annotation**: `text`, `label`, `segment`, `arrow`, `hline`, `vline`, `abline`, `errorbar`

**MAPPING Clause** (Aesthetic Mappings):

Maps data values (columns or literals) to visual aesthetics. Syntax: `value AS aesthetic`

- **Position**: `x`, `y`, `xmin`, `xmax`, `ymin`, `ymax`
- **Color**: `color`, `fill`, `stroke`, `opacity`
- **Size/Shape**: `size`, `shape`, `linetype`, `linewidth`
- **Text**: `label`, `family`, `fontface`

**Literal vs Column**:

- Unquoted → column reference: `region AS color`
- Quoted → literal value: `'value' AS color`

**SETTING Clause** (Parameters):

Sets layer/geom parameters (not mapped to data). Syntax: `param => value`

- Parameters like `opacity`, `size` (fixed), `stroke_width`, etc.

**PARTITION BY Clause** (Grouping):

Groups data for geoms that need grouping (e.g., lines, paths, polygons). Maps to Vega-Lite's `detail` encoding channel, which groups data without adding visual differentiation (unlike `color`).

- Accepts a comma-separated list of column names
- Useful for drawing separate lines/paths for each group without assigning different colors
- Similar to SQL's `PARTITION BY` in window functions

**FILTER Clause** (Layer Filtering):

Applies a filter to the layer data. Supports basic comparison operators.

- Operators: `=`, `!=`, `<>`, `<`, `>`, `<=`, `>=`
- Logical: `AND`, `OR`, parentheses for grouping

**Examples**:

```sql
-- Basic mapping
DRAW line
    MAPPING date AS x, revenue AS y, region AS color

-- Mapping with literal
DRAW point
    MAPPING date AS x, revenue AS y, 'value' AS color

-- Setting parameters
DRAW point
    MAPPING x AS x, y AS y
    SETTING size => 5, opacity => 0.7

-- With filter
DRAW point
    MAPPING x AS x, y AS y, category AS color
    FILTER value > 100

-- Combined
DRAW line
    MAPPING date AS x, value AS y
    SETTING stroke_width => 2
    FILTER category = 'A' AND year >= 2024

-- With PARTITION BY (single column)
DRAW line
    MAPPING date AS x, value AS y
    PARTITION BY category

-- With PARTITION BY (multiple columns)
DRAW line
    MAPPING date AS x, value AS y
    PARTITION BY category, region

-- PARTITION BY with color (grouped lines with different colors)
DRAW line
    MAPPING date AS x, value AS y, region AS color
    PARTITION BY category

-- All clauses combined
DRAW line
    MAPPING date AS x, value AS y
    SETTING stroke_width => 2
    PARTITION BY category, region
    FILTER year >= 2020
```

### SCALE Clause

**Syntax**:

```sql
SCALE [TYPE] <aesthetic> [FROM <input>] [TO <output>] [VIA <transform>] [SETTING <properties>]
```

**Type Modifiers** (optional, placed before aesthetic):

- **`CONTINUOUS`** - Continuous numeric data
- **`DISCRETE`** - Categorical/discrete data
- **`BINNED`** - Binned/bucketed data
- **`DATE`** - Date data (maps to Vega-Lite temporal type)
- **`DATETIME`** - Datetime data (maps to Vega-Lite temporal type)

**Subclauses**:

- **`FROM [...]`** - Input range specification (maps to Vega-Lite `scale.domain`)
- **`TO [...]`** or **`TO palette`** - Output range as array or named palette (maps to Vega-Lite `scale.range` or `scale.scheme`)
- **`VIA transform`** - Transformation method (reserved for future use)
- **`SETTING ...`** - Additional properties (e.g., `breaks`)

**Named Palettes** (used with `TO`):

- `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `diverging`, `sequential`

**Critical for Date Formatting**:

```sql
SCALE x VIA date
-- Maps to Vega-Lite field type = "temporal"
-- Enables proper date axis formatting
```

**Input Range Specification** (FROM clause):

The `FROM` clause explicitly sets the input range for a scale:

```sql
-- Set range for discrete scale
SCALE DISCRETE color FROM ['A', 'B', 'C']

-- Set range for continuous scale
SCALE CONTINUOUS x FROM [0, 100]
```

**Range Specification** (TO clause):

The `TO` clause sets the output range - either explicit values or a named palette:

```sql
-- Explicit color values
SCALE color FROM ['A', 'B'] TO ['red', 'blue']

-- Named palette
SCALE color TO viridis
```

**Examples**:

```sql
-- Date scale
SCALE x VIA date

-- Continuous scale with input range
SCALE CONTINUOUS y FROM [0, 100]

-- Discrete color scale with input range and output range
SCALE DISCRETE color FROM ['A', 'B', 'C'] TO ['red', 'green', 'blue']

-- Color scale with named palette
SCALE color TO viridis

-- Scale with input range and additional settings
SCALE x VIA date FROM ['2024-01-01', '2024-12-31'] SETTING breaks => '1 month'
```

### FACET Clause

**Syntax**:

```sql
-- Wrap layout (single variable = automatic wrap)
FACET <vars> [SETTING <param> => <value>, ...]

-- Grid layout (BY clause for row × column)
FACET <row_vars> BY <col_vars> [SETTING ...]
```

**SETTING Properties**:

- `free => <axes>` - Which axes have independent scales (see below)
- `ncol => <number>` - Number of columns for wrap layout
- `spacing => <number>` - Space between facets

**Free Scales** (`free` property):

- `null` or omitted (default) - Shared/fixed scales across all facets
- `'x'` - Independent x-axis, shared y-axis
- `'y'` - Shared x-axis, independent y-axis
- `['x', 'y']` - Independent scales for both axes

**Customizing Strip Labels**:

To customize facet strip labels, use `SCALE panel RENAMING ...` (for wrap) or `SCALE row/column RENAMING ...` (for grid).

**Examples**:

```sql
-- Simple wrap facet
FACET region

-- Grid facet with BY
FACET region BY category

-- With free y-axis scales
FACET region SETTING free => 'y'

-- With column count for wrap
FACET region SETTING ncol => 3

-- With label renaming via scale
FACET region
SCALE panel RENAMING 'N' => 'North', 'S' => 'South'

-- Combined grid with settings
FACET region BY category
    SETTING free => ['x', 'y'], spacing => 10
```

### PROJECT Clause

**Syntax**:

```sql
-- With projectinate type
PROJECT <type> [SETTING <properties>]

-- With properties only (defaults to cartesian)
PROJECT SETTING <properties>
```

**Projectinate Types**:

- **`cartesian`** - Standard x/y Cartesian projectinates (default)
- **`flip`** - Flipped Cartesian (swaps x and y axes)
- **`polar`** - Polar projectinates (for pie charts, rose plots)
- **`fixed`** - Fixed aspect ratio
- **`trans`** - Transformed projectinates
- **`map`** - Map projections
- **`quickmap`** - Quick approximation for maps

**Common Properties** (all projection types):

- `clip => <boolean>` - Whether to clip marks outside the plot area (default: unset)

**Type-Specific Properties**:

**Cartesian**:

- `ratio => <number>` - Set aspect ratio (not yet implemented)

Note: For axis limits, use `SCALE x FROM [min, max]` or `SCALE y FROM [min, max]`.

**Flip**:

- No additional properties

**Polar**:

- `theta => <aesthetic>` - Which aesthetic maps to angle (defaults to `y`)

**Important Notes**:

1. **Axis limits**: Use `SCALE x/y FROM [min, max]` to set axis limits
2. **Aesthetic domains**: Use `SCALE <aesthetic> FROM [...]` to set aesthetic domains
3. **ggplot2 compatibility**: `project_flip` preserves axis label names (labels stay with aesthetic names, not visual position)
4. **Multi-layer support**: All projection transforms apply to all layers

**Status**:

- ✅ **Cartesian**: Fully implemented and tested
- ✅ **Flip**: Fully implemented and tested
- ✅ **Polar**: Fully implemented and tested
- ❌ **Other types**: Not yet implemented

**Examples**:

```sql
-- Flip projection for horizontal bar chart
PROJECT flip

-- Polar for pie chart (theta defaults to y)
PROJECT polar

-- Polar with explicit theta mapping
PROJECT polar SETTING theta => x

-- Clip marks to plot area
PROJECT cartesian SETTING clip => true

-- Combined with other clauses
DRAW bar MAPPING category AS x, value AS y
SCALE x FROM [0, 100]
SCALE y FROM [0, 200]
PROJECT flip SETTING clip => true
LABEL x => 'Category', y => 'Count'
```

### LABEL Clause

**Syntax**:

```sql
LABEL
  [title => <string>]
  [subtitle => <string>]
  [x => <string>]
  [y => <string>]
  [<aesthetic> => <string>]
  [caption => <string>]
  [tag => <string>]
```

**Example**:

```sql
LABEL
  title => 'Sales by Region',
  x => 'Date',
  y => 'Revenue (USD)',
  caption => 'Data from Q4 2024'
```

### THEME Clause

**Syntax**:

```sql
THEME <name> [SETTING <overrides>]
```

**Base Themes**: `minimal`, `classic`, `gray`, `bw`, `dark`, `void`

**Example**:

```sql
THEME minimal
THEME dark SETTING background => '#1a1a1a'
```

---

## Complete Example Walkthrough

### Query

```sql
SELECT sale_date, region, SUM(quantity) as total
FROM sales
WHERE sale_date >= '2024-01-01'
GROUP BY sale_date, region
ORDER BY sale_date
VISUALISE
DRAW line
    MAPPING sale_date AS x, total AS y, region AS color
DRAW point
    MAPPING sale_date AS x, total AS y, region AS color
SCALE x VIA date
FACET region
LABEL title => 'Sales Trends by Region', x => 'Date', y => 'Total Quantity'
THEME minimal
```

### Execution Flow

**1. Query Splitting**

```rust
// splitter.rs
SQL:  "SELECT sale_date, region, SUM(quantity) as total FROM sales ..."
VIZ:  "VISUALISE DRAW line MAPPING sale_date AS x, ..."
```

**2. SQL Execution** (DuckDB Reader)

```rust
// duckdb.rs
connection.execute(sql) → ResultSet
ResultSet → DataFrame (Polars)

// DataFrame columns: sale_date (Date32), region (String), total (Int64)
// Date32 values converted to ISO format: "2024-01-01"
```

**3. VIZ Parsing** (tree-sitter)

```rust
// parser/mod.rs
Tree-sitter CST → AST

Plot {
  global_mapping: GlobalMapping::Empty,
  layers: [
    Layer { geom: Geom::Line, aesthetics: {"x": "sale_date", "y": "total", "color": "region"} },
    Layer { geom: Geom::Point, aesthetics: {"x": "sale_date", "y": "total", "color": "region"} }
  ],
  scales: [
    Scale { aesthetic: "x", scale_type: Some(ScaleType::Date) }
  ],
  facet: Some(Facet::Wrap { variables: ["region"], scales: "fixed" }),
  labels: Some(Labels { labels: {"title": "...", "x": "Date", "y": "Total Quantity"} }),
  theme: Some(Theme::Minimal)
}
```

**4. Vega-Lite Generation** (VegaLite Writer)

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "data": {
    "values": [
      {"sale_date": "2024-01-01", "region": "North", "total": 150},
      {"sale_date": "2024-01-01", "region": "South", "total": 120},
      ...
    ]
  },
  "title": "Sales Trends by Region",
  "width": 600,
  "autosize": {"type": "fit", "contains": "padding"},
  "facet": {
    "field": "region",
    "type": "nominal"
  },
  "spec": {
    "layer": [
      {
        "mark": "line",
        "encoding": {
          "x": {"field": "sale_date", "type": "temporal", "title": "Date"},
          "y": {"field": "total", "type": "quantitative", "title": "Total Quantity"},
          "color": {"field": "region", "type": "nominal"}
        }
      },
      {
        "mark": "point",
        "encoding": {
          "x": {"field": "sale_date", "type": "temporal", "title": "Date"},
          "y": {"field": "total", "type": "quantitative", "title": "Total Quantity"},
          "color": {"field": "region", "type": "nominal"}
        }
      }
    ]
  }
}
```

**5. Rendering** (Browser/Vega-Lite)

- Vega-Lite JSON → SVG/Canvas visualization
- Faceted multi-line chart with points
- Temporal x-axis with proper date formatting
- Color-coded regions
- Interactive tooltips
