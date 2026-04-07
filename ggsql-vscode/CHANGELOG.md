# Changelog

## 0.1.9

Pre-alpha release.

- Improvements to website (#225, #247, #260, #261, #263)
- Add bidirectionality to layers (#183)
- Text layer (#155)
- Rectangles (#168)
- Fix: remove invalid `stack` property from secondary position channels (#237)
- Update color palettes (#198)
- Annotations (#172)
- Smooth layer (#223)
- Validate mapping (#230)
- Change dimension name to angle (#243)
- Add input validation (#235)
- CI: Audit node packages, update ESLint & uri-js, update Node in workflows (#246)
- Allow for using other readers in ggsql cli (#251)
- Fix: compute position stacking per-facet-panel instead of globally (#245)
- Add logo and update README.md for the VSCode/Positron extension (#255)
- Fix warnings/errors in wasm sysroot (#253)
- Universal query for place (#249)
- Tweaks to avoid cache rate limit (#258, #266, #267, #273)
- Various rect fixes (#262)
- Last batch of housekeeping (#265)
- Add code links to docs (#269)

## 0.1.8

- Initial pre-alpha release.

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
  - PROJECT clause with projection types (cartesian, polar, flip)
  - FACET clause (WRAP, BY with scale options)
  - LABEL clause (title, subtitle, axis labels, caption)
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
