# ggSQL - SQL Visualization Grammar

A SQL extension for declarative data visualization based on the Grammar of Graphics.

ggSQL allows you to write queries that combine SQL data retrieval with visualization specifications in a single, composable syntax.

## Example

```sql
SELECT date, revenue, region
FROM sales
WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
LABEL title => 'Sales by Region'
THEME minimal
```

## Project Status

✨ **Active Development** - Core functionality is working with ongoing feature additions.

**Completed:**

- ✅ Complete tree-sitter grammar with SQL + VISUALISE parsing
- ✅ Full AST type system with validation
- ✅ DuckDB reader with comprehensive type handling
- ✅ Vega-Lite writer with multi-layer support
- ✅ CLI tool (`ggsql`) with parse, exec, and validate commands
- ✅ REST API server (`ggsql-rest`) with CORS support
- ✅ Jupyter kernel (`ggsql-jupyter`) with inline Vega-Lite visualizations
- ✅ VS Code extension (`ggsql-vscode`) with syntax highlighting for `.ggsql` files

**Planned:**

- 📋 Additional readers
- 📋 Additional writers
- 📋 More geom types and statistical transformations
- 📋 Enhanced theme system

## Architecture

ggSQL splits queries at the `VISUALISE` boundary:

- **SQL portion** → passed to pluggable readers (DuckDB, PostgreSQL, CSV, etc.)
- **VISUALISE portion** → parsed and compiled into visualization specifications
- **Output** → rendered via pluggable writers (ggplot2, PNG, Vega-Lite, etc.)

## Development Setup

### Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/georgestagg/ggsql
   cd ggsql
   ```

2. **Install tree-sitter CLI:**

   ```bash
   npm install -g tree-sitter-cli
   ```

3. **Build the project:**

   ```bash
   cargo build
   ```

4. **Run tests:**
   ```bash
   cargo test
   ```

## Project Structure

```
ggsql/
├── Cargo.toml                       # Workspace root configuration
├── README.md                        # This file
│
├── tree-sitter-ggsql/               # Tree-sitter grammar package
│
├── src/                             # Main library
│   ├── lib.rs                       # Public API and re-exports
│   ├── cli.rs                       # Command-line interface
│   ├── rest.rs                      # REST API server
│   ├── parser/                      # Parsing subsystem
│   ├── reader/                      # Data source readers
│   └── writer/                      # Visualization writers
│
├── ggsql-jupyter/                   # Jupyter kernel
│
└── ggsql-vscode/                    # VS Code extension
```

## Development Workflow

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test modules
cargo test ast                       # AST type tests
cargo test splitter                  # Query splitter tests
cargo test parser                    # All parser tests

# Run without default features (avoids database dependencies)
cargo test --no-default-features

# Run with specific features
cargo test --features duckdb,sqlite
```

### Working with the Grammar

The tree-sitter grammar is in `tree-sitter-ggsql/grammar.js`. After making changes:

1. **Regenerate the parser:**

   ```bash
   cd tree-sitter-ggsql
   tree-sitter generate
   ```

2. **Test the grammar:**

   ```bash
   # Test parsing a specific file
   tree-sitter parse test/corpus/basic.txt

   # Test all corpus files
   tree-sitter test
   ```

3. **Debug parsing issues:**

   ```bash
   # Enable debug mode
   tree-sitter parse --debug test/corpus/basic.txt

   # Check for conflicts
   tree-sitter generate --report-states-for-rule=query
   ```

### Code Organization

- **AST Types** (`src/parser/ast.rs`): Core data structures representing parsed ggSQL
- **Query Splitter** (`src/parser/splitter.rs`): Separates SQL from VISUALISE portions
- **AST Builder** (`src/parser/builder.rs`): Converts tree-sitter parse trees to typed AST
- **Error Handling** (`src/parser/error.rs`): Parse-time error types and formatting

### Adding New Grammar Features

1. **Update the grammar** in `tree-sitter-ggsql/grammar.js`
2. **Add corresponding AST types** in `src/parser/ast.rs`
3. **Update the AST builder** in `src/parser/builder.rs`
4. **Add test cases** for the new feature
5. **Update syntax highlighting** in `tree-sitter-ggsql/queries/highlights.scm`

## Testing Strategy

### Unit Tests

Located alongside the code they test:

- `src/parser/ast.rs` - AST type functionality and validation
- `src/parser/splitter.rs` - Query splitting edge cases
- `src/parser/builder.rs` - CST to AST conversion

### Integration Tests

- Full parsing pipeline tests in `src/parser/mod.rs`
- End-to-end query processing (planned)

### Grammar Tests

- `tree-sitter-ggsql/test/corpus/` - Example queries with expected parse trees
- Run with `tree-sitter test`

### Running Specific Test Categories

```bash
# Core AST functionality
cargo test ast::tests

# Query splitting logic
cargo test splitter::tests

# Tree-sitter grammar
cd tree-sitter-ggsql && tree-sitter test

# All parser integration tests
cargo test parser
```

## Grammar Specification

See [CLAUDE.md](CLAUDE.md) for the in-progress ggSQL grammar specification, including:

- Syntax reference
- AST structure
- Implementation phases and architecture
- Design principles and philosophy

Key grammar elements:

- `VISUALISE [mappings] [FROM source]` - Entry point with global aesthetic mappings
- `DRAW <geom> [MAPPING] [SETTING] [FILTER]` - Define geometric layers (point, line, bar, etc.)
- `SCALE <aesthetic> SETTING` - Configure data-to-visual mappings
- `FACET` - Create small multiples (WRAP for flowing layout, BY for grid)
- `COORD` - Coordinate transformations (cartesian, flip, polar)
- `LABEL`, `THEME`, `GUIDE` - Styling and annotation

## Jupyter Kernel

The `ggsql-jupyter` package provides a Jupyter kernel for interactive ggSQL queries with inline Vega-Lite visualizations.

### Installation

```bash
cargo build --release --package ggsql-jupyter
./target/release/ggsql-jupyter --install
```

### Usage

After installation, create a new notebook with the "ggSQL" kernel or use `%kernel ggsql` in an existing notebook.

```sql
-- Create data
CREATE TABLE sales AS
SELECT * FROM (VALUES
    ('2024-01-01'::DATE, 100, 'North'),
    ('2024-01-02'::DATE, 120, 'South')
) AS t(date, revenue, region)

-- Visualize with ggSQL using global mapping
SELECT * FROM sales
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
SCALE x SETTING type => 'date'
LABEL title => 'Sales Trends'
```

The kernel maintains a persistent DuckDB session across cells, so you can create tables in one cell and query them in another.

### Quarto

A Quarto example can be found in `ggsql-jupyter/tests/quarto/doc.qmd`.

## VS Code Extension

The `ggsql-vscode` extension provides syntax highlighting for ggSQL files in Visual Studio Code.

### Installation

```bash
# Package the extension
cd ggsql-vscode
npm install -g @vscode/vsce
vsce package

# Install the VSIX file
code --install-extension ggsql-0.1.0.vsix
```

### Features

- **Syntax highlighting** for ggSQL keywords, geoms, aesthetics, and SQL
- **File association** for `.ggsql`, `.ggsql.sql`, and `.gsql` extensions
- **Bracket matching** and auto-closing for parentheses and brackets
- **Comment support** for `--` single-line and `/* */` multi-line comments

The extension uses a TextMate grammar that highlights:

- SQL keywords (SELECT, FROM, WHERE, JOIN, etc.)
- ggSQL clauses (VISUALISE, DRAW, SCALE, COORD, FACET, etc.)
- Geometric objects (point, line, bar, area, etc.)
- Aesthetics (x, y, color, size, shape, etc.)
- Scale types (linear, log10, date, viridis, etc.)

## CLI

### Installation

```bash
cargo install --path src
```
