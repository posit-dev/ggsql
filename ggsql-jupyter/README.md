# ggsql Jupyter Kernel

A Jupyter kernel for executing ggsql queries with rich inline Vega-Lite visualizations.

## Overview

The ggsql Jupyter kernel enables you to run ggsql queries directly in Jupyter notebooks, with automatic rendering of visualizations using Vega-Lite. It maintains a persistent DuckDB session across cells, allowing you to build up datasets and create visualizations interactively.

## Features

- **Execute ggsql queries** in Jupyter notebooks
- **Rich visualizations** with Vega-Lite rendered inline
- **Persistent DuckDB session** across cells
- **Pure SQL support** with HTML table output
- **Grammar of Graphics** syntax for declarative visualization

## Installation

### Prerequisites

- Jupyter Lab or Notebook installed

### Option 1: Install from PyPI (Recommended)

The easiest way to install the ggsql kernel is from PyPI. This provides pre-built binaries for Linux, macOS, and Windows.

Using pip:

```bash
pip install ggsql-jupyter
ggsql-jupyter --install
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv tool install ggsql-jupyter
ggsql-jupyter --install
```

The `--install` flag registers the kernel with Jupyter. It automatically detects and respects your current environment (virtualenv, conda, uv, etc.).

### Option 2: Install from crates.io

Requires a [Rust toolchain](https://rustup.rs/):

```bash
cargo install ggsql-jupyter
ggsql-jupyter --install
```

### Option 3: Download Pre-built Binary

Pre-built binaries are available from [GitHub Releases](https://github.com/georgestagg/ggsql/releases):

| Platform              | Binary                          |
| --------------------- | ------------------------------- |
| Linux (x86_64)        | `ggsql-jupyter-linux-x64`       |
| Linux (ARM64)         | `ggsql-jupyter-linux-arm64`     |
| macOS (Intel)         | `ggsql-jupyter-macos-x64`       |
| macOS (Apple Silicon) | `ggsql-jupyter-macos-arm64`     |
| Windows (x64)         | `ggsql-jupyter-windows-x64.exe` |

After downloading, make it executable and install:

```bash
chmod +x ggsql-jupyter-*
./ggsql-jupyter-linux-x64 --install
```

On Windows (PowerShell):

```powershell
.\ggsql-jupyter-windows-x64.exe --install
```

### Option 4: Build from Source

Requires a [Rust toolchain](https://rustup.rs/). From the workspace root:

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

### Create a ggsql Notebook

1. In Jupyter, click "New" and select "ggsql" from the dropdown
2. Start writing ggsql queries!

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
SCALE x
  SETTING type => 'date'
LABEL title => 'Revenue Growth', x => 'Date', y => 'Revenue ($)'
```

#### Multi-Layer Plot with Global Mapping

```sql
SELECT x, x*x as y, x*x*x as z
FROM generate_series(1, 10) as t(x)

VISUALISE x AS x
DRAW line
  MAPPING y AS y
DRAW line
  MAPPING z AS y
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
