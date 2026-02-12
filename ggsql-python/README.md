# ggsql

Python bindings for [ggsql](https://github.com/georgestagg/ggsql), a SQL extension for declarative data visualization.

This package provides Python bindings to the Rust `ggsql` crate, enabling Python users to create visualizations using ggsql's VISUALISE syntax with native Altair chart output.

## Installation

### From PyPI (when published)

```bash
pip install ggsql
```

### From source

Building from source requires:

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.10+
- [maturin](https://github.com/PyO3/maturin)

```bash
# Clone the monorepo
git clone https://github.com/georgestagg/ggsql.git
cd ggsql/ggsql-python

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install build dependencies
pip install maturin

# Build and install in development mode
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/ggsql-*.whl
```

## Quick Start

### Simple Usage with `render_altair`

For quick visualizations, use the `render_altair` convenience function:

```python
import ggsql
import polars as pl

# Create a DataFrame
df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 15, 30, 25],
    "category": ["A", "B", "A", "B", "A"]
})

# Render to Altair chart
chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")

# Display or save
chart.display()  # In Jupyter
chart.save("chart.html")  # Save to file
```

### Two-Stage API

For more control, use the two-stage API with explicit reader and writer:

```python
import ggsql
import polars as pl

# 1. Create a DuckDB reader
reader = ggsql.DuckDBReader("duckdb://memory")

# 2. Register your DataFrame as a table
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "revenue": [100, 150, 120],
    "region": ["North", "South", "North"]
})
reader.register("sales", df)

# 3. Execute the ggsql query
spec = reader.execute(
    """
    SELECT * FROM sales
    VISUALISE date AS x, revenue AS y, region AS color
    DRAW line
    LABEL title => 'Sales by Region'
    """
)

# 4. Inspect metadata
print(f"Rows: {spec.metadata()['rows']}")
print(f"Columns: {spec.metadata()['columns']}")
print(f"Layers: {spec.layer_count()}")

# 5. Inspect SQL/VISUALISE portions and data
print(f"SQL: {spec.sql()}")
print(f"Visual: {spec.visual()}")
print(spec.layer_data(0))  # Returns polars DataFrame

# 6. Render to Vega-Lite JSON
writer = ggsql.VegaLiteWriter()
vegalite_json = writer.render(spec)
print(vegalite_json)
```

## API Reference

### Classes

#### `DuckDBReader(connection: str)`

Database reader that executes SQL and manages DataFrames.

```python
reader = ggsql.DuckDBReader("duckdb://memory")  # In-memory database
reader = ggsql.DuckDBReader("duckdb:///path/to/file.db")  # File database
```

**Methods:**

- `register(name: str, df: polars.DataFrame)` - Register a DataFrame as a queryable table
- `unregister(name: str)` - Unregister a previously registered table
- `execute_sql(sql: str) -> polars.DataFrame` - Execute SQL and return results
- `supports_register() -> bool` - Check if registration is supported

#### `VegaLiteWriter()`

Writer that generates Vega-Lite v6 JSON specifications.

```python
writer = ggsql.VegaLiteWriter()
json_output = writer.render(spec)
```

#### `Validated`

Result of `validate()` containing query analysis without SQL execution.

**Methods:**

- `valid() -> bool` - Whether the query is syntactically and semantically valid
- `has_visual() -> bool` - Whether the query contains a VISUALISE clause
- `sql() -> str` - The SQL portion (before VISUALISE)
- `visual() -> str` - The VISUALISE portion
- `errors() -> list[dict]` - Validation errors with messages and locations
- `warnings() -> list[dict]` - Validation warnings

#### `Spec`

Result of `reader.execute()`, containing resolved visualization ready for rendering.

**Methods:**

- `metadata() -> dict` - Get `{"rows": int, "columns": list[str], "layer_count": int}`
- `sql() -> str` - The executed SQL query
- `visual() -> str` - The VISUALISE clause
- `layer_count() -> int` - Number of DRAW layers
- `data() -> polars.DataFrame | None` - Main query result DataFrame
- `layer_data(index: int) -> polars.DataFrame | None` - Layer-specific data (if filtered)
- `stat_data(index: int) -> polars.DataFrame | None` - Statistical transform data
- `layer_sql(index: int) -> str | None` - Layer filter SQL
- `stat_sql(index: int) -> str | None` - Stat transform SQL
- `warnings() -> list[dict]` - Validation warnings from execution

### Functions

#### `validate(query: str) -> Validated`

Validate query syntax and semantics without executing SQL.

```python
validated = ggsql.validate("SELECT x, y FROM data VISUALISE x, y DRAW point")
if validated.valid():
    print("Query is valid!")
else:
    for error in validated.errors():
        print(f"Error: {error['message']}")
```

#### `reader.execute(query: str) -> Spec`

Execute a ggsql query and return the visualization specification.

```python
reader = ggsql.DuckDBReader("duckdb://memory")
spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
```

#### `render_altair(df, viz: str, **kwargs) -> altair.Chart`

Convenience function to render a DataFrame with a VISUALISE spec to an Altair chart.

**Parameters:**

- `df` - Any narwhals-compatible DataFrame (polars, pandas, etc.). LazyFrames are collected automatically.
- `viz` - The VISUALISE specification string
- `**kwargs` - Additional arguments passed to `altair.Chart.from_json()` (e.g., `validate=False`)

**Returns:** An Altair chart object (Chart, LayerChart, FacetChart, etc.)

```python
import polars as pl
import ggsql

df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
```

## Examples

### Mapping Styles

```python
df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "category": ["A", "B", "A"]})

# Explicit mapping
ggsql.render_altair(df, "VISUALISE x AS x, y AS y DRAW point")

# Implicit mapping (column name = aesthetic name)
ggsql.render_altair(df, "VISUALISE x, y DRAW point")

# Wildcard mapping (map all matching columns)
ggsql.render_altair(df, "VISUALISE * DRAW point")

# With color encoding
ggsql.render_altair(df, "VISUALISE x, y, category AS color DRAW point")
```

### Custom Readers

You can use any Python object with an `execute_sql(sql: str) -> polars.DataFrame` method as a reader. This enables integration with any data source.

```python
import ggsql
import polars as pl

class CSVReader:
    """Custom reader that loads data from CSV files."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def execute_sql(self, sql: str) -> pl.DataFrame:
        # Simple implementation: ignore SQL and return fixed data
        # A real implementation would parse SQL to determine which file to load
        return pl.read_csv(f"{self.data_dir}/data.csv")

# Use custom reader with ggsql.execute()
reader = CSVReader("/path/to/data")
spec = ggsql.execute(
    "SELECT * FROM data VISUALISE x, y DRAW point",
    reader
)
writer = ggsql.VegaLiteWriter()
json_output = writer.render(spec)
```

**Optional methods** for custom readers:

- `supports_register() -> bool` - Return `True` if your reader supports DataFrame registration
- `register(name: str, df: polars.DataFrame) -> None` - Register a DataFrame as a queryable table
- `unregister(name: str) -> None` - Unregister a previously registered table

```python
class AdvancedReader:
    """Custom reader with registration support."""

    def __init__(self):
        self.tables = {}

    def execute_sql(self, sql: str) -> pl.DataFrame:
        # Your SQL execution logic here
        ...

    def supports_register(self) -> bool:
        return True

    def register(self, name: str, df: pl.DataFrame) -> None:
        self.tables[name] = df

    def unregister(self, name: str) -> None:
        del self.tables[name]
```

Native readers like `DuckDBReader` use an optimized fast path, while custom Python readers are automatically bridged via IPC serialization.

### Ibis Reader Example

[Ibis](https://ibis-project.org/) provides a unified Python API for SQL operations across multiple backends. Here's how to create an ibis-based custom reader:

```python
import ggsql
import polars as pl
import ibis

class IbisReader:
    """Custom reader using ibis as the SQL backend."""

    def __init__(self, backend="duckdb"):
        if backend == "duckdb":
            self.con = ibis.duckdb.connect()
        elif backend == "sqlite":
            self.con = ibis.sqlite.connect()
        # Add other backends as needed

    def execute_sql(self, sql: str) -> pl.DataFrame:
        return self.con.con.execute(sql).pl()

    def supports_register(self) -> bool:
        return True

    def register(self, name: str, df: pl.DataFrame) -> None:
        self.con.create_table(name, df.to_arrow(), overwrite=True)

    def unregister(self, name: str) -> None:
        self.con.drop_table(name)

# Usage
reader = IbisReader()
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "revenue": [100, 150, 120],
})
reader.register("sales", df)

spec = ggsql.execute(
    "SELECT * FROM sales VISUALISE date AS x, revenue AS y DRAW line",
    reader
)
writer = ggsql.VegaLiteWriter()
print(writer.render(spec))
```

## Development

### Keeping in sync with the monorepo

The `ggsql-python` package is part of the [ggsql monorepo](https://github.com/posit-dev/ggsql) and depends on the Rust `ggsql` crate via a path dependency. When the Rust crate is updated, you may need to rebuild:

```bash
cd ggsql-python

# Rebuild after Rust changes
maturin develop

# If tree-sitter grammar changed, clean and rebuild
cd .. && cargo clean -p tree-sitter-ggsql && cd ggsql-python
maturin develop
```

### Running tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- altair >= 5.0
- narwhals >= 2.15
- polars >= 1.0

## License

MIT
