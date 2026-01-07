# ggSQL Jupyter Kernel

A Jupyter kernel for executing ggSQL queries with rich inline Vega-Lite visualizations.

## Overview

The ggSQL Jupyter kernel enables you to run ggSQL queries directly in Jupyter notebooks, with automatic rendering of visualizations using Vega-Lite. It maintains a persistent DuckDB session across cells, allowing you to build up datasets and create visualizations interactively.

## Features

- **Execute ggSQL queries** in Jupyter notebooks
- **Rich visualizations** with Vega-Lite rendered inline
- **Persistent DuckDB session** across cells
- **Pure SQL support** with HTML table output
- **Grammar of Graphics** syntax for declarative visualization

## Installation

### Prerequisites

- Jupyter Lab or Notebook installed
- Python 3.8+ (for Jupyter)

### Option 1: Install from crates.io (Recommended)

If you have Rust installed:

```bash
cargo install ggsql-jupyter
ggsql-jupyter --install
```

This will:

1. Download and compile the kernel
2. Install it into your current environment (respects virtualenvs, conda, uv)

### Option 2: Download Pre-built Binary from GitHub Releases

For users without Rust:

1. **Download the binary** for your platform from [GitHub Releases](https://github.com/georgestagg/ggsql/releases)

   - Linux: `ggsql-jupyter-linux-x64`
   - macOS (Intel): `ggsql-jupyter-macos-x64`
   - macOS (Apple Silicon): `ggsql-jupyter-macos-arm64`
   - Windows: `ggsql-jupyter-windows-x64.exe`

2. **Rename and make executable** (Linux/macOS):

   ```bash
   mv ggsql-jupyter-linux-x64 ggsql-jupyter
   chmod +x ggsql-jupyter
   ```

3. **Install the kernel**:

   ```bash
   ./ggsql-jupyter --install
   ```

   On Windows (PowerShell):

   ```powershell
   .\ggsql-jupyter-windows-x64.exe --install
   ```

The `--install` flag automatically:

- Creates a temporary directory with the kernel spec
- Copies the binary to the appropriate location
- Runs `jupyter kernelspec install` with the correct flags
- Respects your current environment (virtualenv, conda, etc.)
- Cleans up temporary files

### Option 3: Build from Source

From the workspace root:

```bash
cargo build --release --package ggsql-jupyter
./target/release/ggsql-jupyter --install
```

### Installation Flags

- `--install`: Install the kernel (default: user install)
- `--install --user`: Explicitly install for current user
- `--install --sys-prefix`: Install into sys.prefix (for conda envs)

### Verify Installation

```bash
jupyter kernelspec list
```

You should see `ggsql` in the list of available kernels.

## Usage

### Start Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### Create a ggSQL Notebook

1. In Jupyter, click "New" and select "ggSQL" from the dropdown
2. Start writing ggSQL queries!

### Example Queries

#### Simple Point Plot

```sql
SELECT 1 as x, 2 as y, 'A' as category
UNION ALL
SELECT 2, 4, 'A'
UNION ALL
SELECT 3, 3, 'B'
VISUALISE x, y, category AS color
DRAW point
```

#### Time Series

```sql
SELECT
    '2024-01-01'::DATE + INTERVAL (n) DAY as date,
    n * 10 as revenue
FROM generate_series(0, 30) as t(n)
VISUALISE date AS x, revenue AS y
DRAW line
SCALE x SETTING type TO 'date'
LABEL title => 'Revenue Growth', x => 'Date', y => 'Revenue ($)'
```

#### Multi-Layer Plot with Global Mapping

```sql
SELECT x, x*x as y, x*x*x as z
FROM generate_series(1, 10) as t(x)
VISUALISE x AS x
DRAW line MAPPING y AS y
DRAW line MAPPING z AS y
LABEL title => 'Polynomial Functions'
```

#### Pure SQL (Data Tables)

```sql
SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, name)
```

This will display as an HTML table without visualization.

#### Building Up Data Across Cells

Cell 1:

```sql
CREATE TABLE products AS
SELECT * FROM (VALUES
    (1, 'Widget', 10.99),
    (2, 'Gadget', 24.99),
    (3, 'Doohickey', 5.99)
) AS t(id, name, price)
```

Cell 2:

```sql
SELECT * FROM products
VISUALISE name AS x, price AS y
DRAW bar
LABEL title => 'Product Prices', y => 'Price ($)'
```
