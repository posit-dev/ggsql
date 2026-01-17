# Changelog

## 0.1.1

- Rename the WITH clause to DRAW

## 0.1.0

- Initial release of ggsql syntax highlighting
- Complete TextMate grammar for ggsql language
- Support for SQL keywords (SELECT, FROM, WHERE, JOIN, WITH, etc.)
- Support for ggsql visualization clauses:
  - VISUALISE/VISUALIZE AS statements
  - WITH clause with geom types (point, line, bar, area, histogram, etc.)
  - SCALE clause with scale types (linear, log10, date, viridis, etc.)
  - COORD clause with coordinate types (cartesian, polar, flip)
  - FACET clause (WRAP, BY with scale options)
  - LABEL clause (title, subtitle, axis labels, caption)
  - GUIDE clause (legend, colorbar, axis configuration)
  - THEME clause (minimal, classic, dark, etc.)
- Aesthetic name highlighting (x, y, color, fill, size, shape, etc.)
- String and number literal highlighting
- Comment support (line comments `--` and block comments `/* */`)
- Bracket matching and auto-closing for `()`, `[]`, `{}`, `''`, `""`
- File extension associations: `.ggsql`, `.ggsql.sql` and `.gsql`.
- Language configuration for proper editor behavior
- Comprehensive README with examples and usage instructions
- Syntax highlighting for all ggsql constructs
- Compatible with all VSCode color themes
- Auto-closing pairs for quotes and brackets
- Comment toggling with keyboard shortcuts
- Word pattern configuration for proper word selection
