# ggSQL Syntax Highlighting for VSCode

Syntax highlighting for ggSQL - SQL with declarative visualization based on Grammar of Graphics.

## Features

- **Complete syntax highlighting** for ggSQL queries
- **SQL keyword support** (SELECT, FROM, WHERE, JOIN, WITH, etc.)
- **ggSQL clause highlighting**: (SCALE, COORD, FACET, LABEL, etc.)
- **Aesthetic highlighting** (x, y, color, size, shape, etc.)
- **String and number literals**
- **Comment support** (`--` and `/* */`)
- **Bracket matching and auto-closing**

## Installation

### From VSIX File (Recommended)

1. Download the latest `.vsix` file from the releases
2. Open VSCode
3. Go to Extensions view (Ctrl+Shift+X / Cmd+Shift+X)
4. Click the "..." menu → "Install from VSIX..."
5. Select the downloaded `.vsix` file

Or via command line:

```bash
code --install-extension ggsql-0.1.0.vsix
```

### From Source

```bash
# Clone the repository
git clone https://github.com/georgestagg/ggsql.git
cd ggsql/ggsql-vscode

# Install vsce (VSCode Extension Manager)
npm install -g @vscode/vsce

# Package the extension
vsce package

# Install the generated .vsix file
code --install-extension ggsql-0.1.0.vsix
```

### Development and Testing

```bash
# Open extension folder in VSCode
cd ggsql-vscode
code .

# Press F5 to launch Extension Development Host
# This opens a new VSCode window with the extension loaded
```

### Test with Sample File

1. After installation, open the example file:

   ```bash
   code ggsql-vscode/examples/sample.gsql
   ```

2. You should see syntax highlighting for:
   - SQL keywords in one color
   - ggSQL keywords (VISUALISE, DRAW, SCALE, etc.) in another
   - Geom types, aesthetics, scale types highlighted distinctly
   - Comments, strings, and numbers properly highlighted

### Create a Test File

Create a new file with `.gsql` extension:

```sql
-- test.gsql
SELECT * FROM sales WHERE year = 2024
VISUALISE date AS x, revenue AS y
DRAW line
SCALE x SETTING type => 'date'
LABEL title => 'Sales Trends'
```

Verify that all keywords are highlighted correctly.

### Debug Highlighting Issues

If highlighting seems incorrect:

1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Run: "Developer: Inspect Editor Tokens and Scopes"
3. Click on any token to see its scope name
4. Compare with scopes defined in `syntaxes/ggsql.tmLanguage.json`

## About ggSQL

ggSQL is a SQL extension for declarative data visualization. It combines SQL queries with Grammar of Graphics-inspired visualization specifications, enabling you to query data and define visualizations in a single, composable syntax.

ggSQL extends SQL with a `VISUALISE` clause that acts as a terminal operation, producing visualizations instead of relational data. Global aesthetic mappings can be specified directly after VISUALISE (e.g., `VISUALISE date AS x, revenue AS y`).
