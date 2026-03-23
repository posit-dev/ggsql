# ggsql

[ggsql](https://ggsql.org) is a SQL extension for declarative data visualization based on Grammar of Graphics principles. It combines SQL data queries with visualization specifications in a single, composable syntax.

## Features

- Complete syntax highlighting for ggsql queries.
- `.ggsql` file extension support.
- Language runtime integration for [Positron IDE](https://positron.posit.co).

## Example

```sql
SELECT date, revenue, region
FROM sales
WHERE year = 2024
VISUALISE date AS x, revenue AS y, region AS color
DRAW line
SCALE x SETTING breaks => 'month'
LABEL title => 'Sales by Region'
```

## Installation

You can either install ggsql from the extension marketplace or download and install it manually:

1. Download [`ggsql.vsix`](https://github.com/posit-dev/ggsql/releases/latest/download/ggsql.vsix)
2. Install via the command line:

```bash
# VS Code
code --install-extension ggsql.vsix

# Positron
positron --install-extension ggsql.vsix
```

Or install from within the editor: open the Extensions view, click the `...` menu, select "Install from VSIX...", and choose the downloaded file.

## Learn More

Visit [ggsql.org](https://ggsql.org) to get started with ggsql, explore the documentation, and see more examples.
