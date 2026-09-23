//! A minimal HTML table writer.
//!
//! Maps `ResolvedTable`'s three parts onto distinct pieces of the `<table>`:
//! - `cells` → `<thead>`/`<tbody>` rows. How a cell's styling is expressed
//!   depends on the `css_mode` option: `class` (the default) puts `ggsql_*`
//!   classes on the elements — from the cell's recorded `TableCellClass`es
//!   plus property-derived classes like alignment — backed by a `<style>`
//!   block; `inline` renders the same declarations into each element's
//!   `style` attribute instead (the two gt `as_raw_html()` modes).
//! - `columns` → a `<colgroup>`, one `<col>` per column, with a `style`
//!   attribute from that column's resolved `width` (a bare `<col>` for a
//!   column with none) in both modes — targeted column styling stays inline,
//!   as gt keeps `tab_style()` rules inline.
//! - `rows` → not consumed yet; no row-wide property exists to render.
//!
//! No footnotes, since `Table` has no fields to describe those yet. Spanner
//! rows are rendered (as `colspan`, one `<tr>` per level, above the column
//! labels); `render_cell`/`render_row` can also render a `rowspan` cell,
//! though nothing in the resolution pipeline produces one yet, so a column
//! with no spanner at a given level still gets a blank filler cell rather
//! than a merged one. This is a stub to prove the Table → writer plumbing
//! end to end, not the real grammar-of-tables output; it deliberately does
//! not reuse `ggsql-jupyter`'s existing `dataframe_to_html`, which works
//! directly off a `DataFrame` rather than resolved `TableCell`s.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::plot::{ParameterValue, Parameters};
use crate::util::escape_html;
use crate::writer::{Writer, WriterOptions};
use crate::{
    DataFrame, GgsqlError, Plot, Result, TableCell, TableCellClass, TableColumn, TableRow,
};

/// Renders a resolved table as a bare HTML `<table>`. Does not support plots.
#[derive(Debug, Default)]
pub struct HtmlWriter {
    css_mode: CssMode,
}

/// How cell styling is expressed in the output HTML — the two gt
/// `as_raw_html()` modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CssMode {
    /// `ggsql_*` classes on the elements, backed by a `<style>` block
    /// carrying the style classes' rules (gt's `inline_css = FALSE`).
    #[default]
    Class,
    /// Declarations rendered into each element's `style` attribute; no
    /// classes, no `<style>` block (gt's `inline_css = TRUE`).
    Inline,
}

impl HtmlWriter {
    /// Create a new HtmlWriter (class mode; see `css_mode`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Writer for HtmlWriter {
    type Output = String;

    /// Takes one option: `css_mode` — `class` (default) renders `ggsql_*`
    /// classes backed by a `<style>` block; `inline` renders the same
    /// declarations into each element's `style` attribute.
    fn from_options(options: &WriterOptions) -> Result<Self> {
        options.reject_unknown(&["css_mode"])?;
        let css_mode = match options.one_of("css_mode", &["inline", "class"])? {
            Some("inline") => CssMode::Inline,
            _ => CssMode::Class,
        };
        Ok(Self { css_mode })
    }

    fn write_plot(&self, _spec: &Plot, _data: &HashMap<String, DataFrame>) -> Result<String> {
        Err(GgsqlError::WriterError(
            "HtmlWriter does not support plots".to_string(),
        ))
    }

    fn validate_plot(&self, _spec: &Plot) -> Result<()> {
        Err(GgsqlError::WriterError(
            "HtmlWriter does not support plots".to_string(),
        ))
    }

    fn write_table(
        &self,
        cells: &[TableCell],
        columns: Option<&[TableColumn]>,
        rows: Option<&[TableRow]>,
    ) -> Result<String> {
        // Not consumed yet — no property needs whole-row rendering yet.
        let _ = rows;

        // Fold `hjust` into alignment classes up front, so rendering reads
        // `classes` alone.
        let cells: Vec<TableCell> = cells
            .iter()
            .cloned()
            .map(TableCell::discretise_hjust)
            .collect();

        let ncol = cells
            .iter()
            .map(|cell| cell.right)
            .max()
            .map_or(0, |r| r + 1);

        // BTreeMap because essentially Vec<Vec<&TableCell>> but compact for missing rows
        let mut header_rows: BTreeMap<usize, Vec<&TableCell>> = BTreeMap::new();
        let mut body_rows: BTreeMap<usize, Vec<&TableCell>> = BTreeMap::new();

        // Sorting Hat: Hmmm... Yes... BODY ROW!!! *applause*
        for cell in &cells {
            if cell.is_header() {
                header_rows.entry(cell.top).or_default().push(cell);
            } else {
                body_rows.entry(cell.top).or_default().push(cell);
            }
        }

        // Not a general TableCell invariant — just what this writer's split
        // into two blocks requires.
        if let (Some(&max_header_row), Some(&min_body_row)) =
            (header_rows.keys().next_back(), body_rows.keys().next())
        {
            if max_header_row >= min_body_row {
                return Err(GgsqlError::WriterError(format!(
                    "HtmlWriter renders headers and body as separate <thead>/<tbody> blocks, so \
                     every header cell must sit above every body cell; found a header row at \
                     {max_header_row} at or below a body row at {min_body_row}"
                )));
            }
        }

        let nrow = cells
            .iter()
            .map(|cell| cell.bottom)
            .max()
            .map_or(0, |r| r + 1);
        let occupied = occupied_columns_per_row(&cells, nrow);

        let mut html = String::new();
        if self.css_mode == CssMode::Class {
            html.push_str(&render_style_block());
        }
        html.push_str("<table>\n");

        if let Some(colgroup) = render_colgroup(columns) {
            html.push_str(&colgroup);
        }

        if !header_rows.is_empty() {
            html.push_str("<thead>\n");
            for (row, row_cells) in header_rows {
                html.push_str(&render_row(row_cells, ncol, &occupied[row], self.css_mode));
            }
            html.push_str("</thead>\n");
        }

        if !body_rows.is_empty() {
            html.push_str("<tbody>\n");
            for (row, row_cells) in body_rows {
                html.push_str(&render_row(row_cells, ncol, &occupied[row], self.css_mode));
            }
            html.push_str("</tbody>\n");
        }

        html.push_str("</table>");

        Ok(html)
    }
}

/// What a `TableCellClass` stands for in CSS, as property–value pairs. The
/// single source of truth both `css_mode`s render from — class mode puts
/// these in the `<style>` block, inline mode folds them into the element's
/// `style` attribute. Structural classes carry no declarations — they're
/// pure semantic hooks for the embedder.
fn class_declarations(class: TableCellClass) -> &'static [(&'static str, &'static str)] {
    match class {
        TableCellClass::Row
        | TableCellClass::ColHeading
        | TableCellClass::Spanner
        | TableCellClass::SpannerOuter => &[],
        TableCellClass::AlignLeft => &[("text-align", "left")],
        TableCellClass::AlignCenter => &[("text-align", "center")],
        TableCellClass::AlignRight => &[
            ("text-align", "right"),
            // Right-aligned data reads as numeric — keep digit widths uniform
            // so they still line up under one another.
            ("font-variant-numeric", "tabular-nums"),
        ],
    }
}

/// The styled classes (those carrying declarations), in `<style>`-block rule
/// order. With equal specificity the later rule wins, so this decides
/// precedence in class mode; `inline_style`'s own precedence instead follows
/// the order of the cell's `classes` field (see its doc comment).
const STYLED_CLASSES: &[TableCellClass] = &[
    TableCellClass::AlignLeft,
    TableCellClass::AlignCenter,
    TableCellClass::AlignRight,
];

/// The `<style>` block class mode prepends — one rule per styled class,
/// generated from the same lookup inline mode folds into `style` attributes,
/// so the modes can't drift. Emitted unconditionally: three short rules
/// aren't worth a pass over the cells to see which are used.
fn render_style_block() -> String {
    let mut block = String::from("<style>\n");
    for &class in STYLED_CLASSES {
        let name = format!("ggsql_{class}");
        let declarations = class_declarations(class)
            .iter()
            .map(|(property, value)| format!("{property}: {value}"))
            .collect::<Vec<_>>()
            .join("; ");
        block.push_str(&format!(".{name} {{ {declarations}; }}\n"));
    }
    block.push_str("</style>\n");
    block
}

/// Fold classes' declarations into a `style` attribute value, each CSS
/// property appearing at most once — a later class's declaration overrides
/// an earlier one's (CSS's last-wins rule, made explicit so the output
/// doesn't rely on browsers' duplicate-declaration handling).
fn inline_style(classes: &[TableCellClass]) -> Option<String> {
    let mut folded: Vec<(&str, &str)> = Vec::new();
    for class in classes {
        for &(property, value) in class_declarations(*class) {
            match folded.iter_mut().find(|(p, _)| *p == property) {
                Some(entry) => entry.1 = value,
                None => folded.push((property, value)),
            }
        }
    }
    (!folded.is_empty()).then(|| {
        folded
            .iter()
            .map(|(property, value)| format!("{property}: {value}"))
            .collect::<Vec<_>>()
            .join("; ")
    })
}

/// Render one `TableCell` as an HTML tag — `<th>`/`<td>` from
/// `cell.is_header()`, with a `colspan`/`rowspan` attribute only when the
/// cell actually spans more than one column/row. Styling per `mode`: class
/// mode emits a `class` attribute naming the cell's `TableCellClass`es;
/// inline mode folds their declarations into a `style` attribute. Each
/// attribute appears only when non-empty.
fn render_cell(cell: &TableCell, mode: CssMode) -> String {
    let tag = if cell.is_header() { "th" } else { "td" };
    let colspan = cell.width();
    let rowspan = cell.height();
    let mut attrs = String::new();
    if colspan > 1 {
        attrs.push_str(&format!(" colspan=\"{colspan}\""));
    }
    if rowspan > 1 {
        attrs.push_str(&format!(" rowspan=\"{rowspan}\""));
    }

    match mode {
        CssMode::Class => {
            if !cell.classes.is_empty() {
                let names = cell
                    .classes
                    .iter()
                    .map(|&class| format!("ggsql_{class}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                attrs.push_str(&format!(" class=\"{names}\""));
            }
        }
        CssMode::Inline => {
            if let Some(style) = inline_style(&cell.classes) {
                attrs.push_str(&format!(" style=\"{style}\""));
            }
        }
    }
    format!("<{tag}{attrs}>{}</{tag}>", escape_html(&cell.content))
}

/// Render a `<colgroup>` block, one `<col>` per column, or `None` if
/// `columns` is absent or none of them resolve a `style`. Skipping the block
/// entirely in that case avoids emitting a run of bare, attribute-less
/// `<col>` tags that would render identically to omitting them.
fn render_colgroup(columns: Option<&[TableColumn]>) -> Option<String> {
    let styles: Vec<Option<String>> = columns?
        .iter()
        .map(|c| column_style(&c.properties))
        .collect();
    if styles.iter().all(Option::is_none) {
        return None;
    }

    let mut html = String::from("<colgroup>\n");
    for style in styles {
        match style {
            Some(style) => html.push_str(&format!("<col style=\"{style}\">\n")),
            None => html.push_str("<col>\n"),
        }
    }
    html.push_str("</colgroup>\n");
    Some(html)
}

/// Translate a column's resolved `FORMAT` properties into a `<col>`'s
/// `style` attribute value, or `None` if it has no `width`. `width`'s value
/// is already validated (by `ParamConstraint::string_numeric_with_unit`) to
/// be a number immediately followed by `px` or `%`, which is exactly CSS's
/// own `width` syntax, so it's used as-is.
fn column_style(properties: &Parameters) -> Option<String> {
    match properties.get("width") {
        Some(ParameterValue::String(width)) => Some(format!("width: {width}")),
        _ => None,
    }
}

/// Per row, the column positions already covered by a cell that started in
/// an earlier row and hasn't ended yet (a rowspan declared above that row) —
/// those get no cell and no filler when the row is rendered, since the
/// earlier cell already claims that grid position.
fn occupied_columns_per_row(cells: &[TableCell], nrow: usize) -> Vec<HashSet<usize>> {
    let mut occupied = vec![HashSet::new(); nrow];
    for cell in cells {
        // Excludes `top` itself — that row renders the cell normally.
        let spanned_rows = (cell.top + 1)..=cell.bottom;
        for occupied_row in &mut occupied[spanned_rows] {
            occupied_row.extend(cell.left..=cell.right);
        }
    }
    occupied
}

/// Render one row — cells sharing a `top` — as `<tr>...</tr>`, walking every
/// column position from `0..ncol` and padding any gap with a synthesized
/// empty cell of the same kind. `occupied` (from `occupied_columns_per_row`)
/// names the columns a rowspan from an earlier row already claims — those
/// get neither a cell nor a filler here.
fn render_row(
    mut cells: Vec<&TableCell>,
    ncol: usize,
    occupied: &HashSet<usize>,
    mode: CssMode,
) -> String {
    // write_table's only caller never hands this an empty `cells` (an entry
    // is only ever created already holding a cell), but an empty row has no
    // `kind` to synthesize fillers with, so this returns early rather than
    // assume that guarantee holds forever.
    if cells.is_empty() {
        return String::new();
    }

    // The walk below depends on `left`-ascending order.
    cells.sort_by_key(|cell| cell.left);
    // Known gap: `kind` and `filler_classes` both assume every real cell in
    // this row shares one `TableCellKind`/structural class, true of every row
    // today but not guaranteed to stay true once row stubs (labels/groups)
    // can share a row with `Body` cells — revisit then.
    let kind = cells[0].kind;
    // A real cell's classes are uniform across the row by construction, so
    // any of them names the filler's correctly.
    let filler_classes = cells[0].classes.clone();
    let mut html = String::from("<tr>");

    // `ncol` comes from the whole table, not this row's own cells — a row
    // that doesn't reach the last column must still pad out to it.
    let mut col = 0;
    let mut i = 0;
    while col < ncol {
        if occupied.contains(&col) {
            col += 1;
        } else if i < cells.len() && cells[i].left == col {
            html.push_str(&render_cell(cells[i], mode));
            col = cells[i].right + 1;
            i += 1;
        } else {
            // A gap (a spanner not reaching every column, or one fragmented
            // by another spanner's occupancy) renders through `render_cell`
            // too, so one place decides which tag a `TableCellKind` gets,
            // not two.
            let filler = TableCell::new(kind, cells[0].top, cells[0].top, col, col, String::new())
                .with_classes(filler_classes.clone());
            html.push_str(&render_cell(&filler, mode));
            col += 1;
        }
    }

    html.push_str("</tr>\n");
    html
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::{TableCellClass, TableCellKind};

    fn cell(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> TableCell {
        TableCell::new(kind, top, bottom, left, right, String::new())
    }

    #[test]
    fn from_options_defaults_to_class_mode() {
        assert_eq!(HtmlWriter::new().css_mode, CssMode::Class);
        let writer = HtmlWriter::from_options(&WriterOptions::new()).unwrap();
        assert_eq!(writer.css_mode, CssMode::Class);
    }

    #[test]
    fn from_options_accepts_inline_and_class_case_insensitively() {
        let options = WriterOptions::parse(["css-mode=INLINE"]).unwrap();
        assert_eq!(
            HtmlWriter::from_options(&options).unwrap().css_mode,
            CssMode::Inline
        );
        let options = WriterOptions::parse(["css_mode=Class"]).unwrap();
        assert_eq!(
            HtmlWriter::from_options(&options).unwrap().css_mode,
            CssMode::Class
        );
    }

    #[test]
    fn from_options_rejects_bad_values_and_unknown_keys() {
        let err = HtmlWriter::from_options(&WriterOptions::parse(["css_mode=banana"]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("expects 'inline' or 'class'"), "{err}");
        assert!(HtmlWriter::from_options(&WriterOptions::parse(["widdth=3"]).unwrap()).is_err());
    }

    #[test]
    fn render_row_returns_empty_string_for_no_cells() {
        assert_eq!(
            render_row(Vec::new(), 3, &HashSet::new(), CssMode::Inline),
            ""
        );
    }

    #[test]
    fn render_cell_emits_rowspan_when_height_is_greater_than_one() {
        let c = cell(TableCellKind::ColumnLabel, 0, 1, 0, 0);
        assert_eq!(render_cell(&c, CssMode::Inline), "<th rowspan=\"2\"></th>");
    }

    #[test]
    fn render_cell_emits_a_style_attribute_from_properties_in_inline_mode() {
        // Declaration content is inline_style's tests' job — this just checks
        // render_cell routes `hjust` through it as a `style="..."` attribute.
        let mut c = cell(TableCellKind::Body, 0, 0, 0, 0);
        c.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let html = render_cell(&c.discretise_hjust(), CssMode::Inline);
        assert!(html.starts_with("<td style=\"text-align: right;"));
        assert!(html.ends_with("\"></td>"));
    }

    #[test]
    fn render_cell_emits_classes_in_class_mode() {
        let mut body =
            cell(TableCellKind::Body, 0, 0, 0, 0).with_classes(vec![TableCellClass::Row]);
        body.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let body = body.discretise_hjust();
        assert_eq!(
            render_cell(&body, CssMode::Class),
            "<td class=\"ggsql_row ggsql_right\"></td>"
        );

        let label = cell(TableCellKind::ColumnLabel, 0, 0, 0, 0)
            .with_classes(vec![TableCellClass::ColHeading]);
        assert_eq!(
            render_cell(&label, CssMode::Class),
            "<th class=\"ggsql_col_heading\"></th>"
        );

        let spanner =
            cell(TableCellKind::Spanner, 0, 0, 0, 1).with_classes(vec![TableCellClass::Spanner]);
        assert_eq!(
            render_cell(&spanner, CssMode::Class),
            "<th colspan=\"2\" class=\"ggsql_spanner\"></th>"
        );

        let outer = cell(TableCellKind::Spanner, 0, 0, 0, 1)
            .with_classes(vec![TableCellClass::SpannerOuter]);
        assert_eq!(
            render_cell(&outer, CssMode::Class),
            "<th colspan=\"2\" class=\"ggsql_spanner_outer\"></th>"
        );
    }

    #[test]
    fn inline_style_folds_declarations_per_class() {
        assert_eq!(inline_style(&[]), None);
        assert_eq!(
            inline_style(&[TableCellClass::AlignLeft]),
            Some("text-align: left".to_string())
        );
        assert_eq!(
            inline_style(&[TableCellClass::AlignRight]),
            Some("text-align: right; font-variant-numeric: tabular-nums".to_string())
        );
        // Structural classes carry no declarations.
        assert_eq!(inline_style(&[TableCellClass::Row]), None);
    }

    #[test]
    fn inline_style_lets_a_later_class_override_an_earlier_ones_property() {
        // AlignLeft and AlignRight both declare text-align — last wins.
        assert_eq!(
            inline_style(&[TableCellClass::AlignLeft, TableCellClass::AlignRight]),
            Some("text-align: right; font-variant-numeric: tabular-nums".to_string())
        );
    }

    #[test]
    fn column_style_reads_a_string_width_as_a_css_width() {
        let mut properties = Parameters::new();
        properties.insert(
            "width".to_string(),
            ParameterValue::String("20%".to_string()),
        );
        assert_eq!(column_style(&properties), Some("width: 20%".to_string()));

        assert_eq!(column_style(&Parameters::new()), None);
    }

    fn column(properties: Parameters) -> TableColumn {
        TableColumn {
            name: String::new(),
            label: String::new(),
            properties,
        }
    }

    #[test]
    fn render_colgroup_returns_none_without_columns() {
        assert_eq!(render_colgroup(None), None);
    }

    #[test]
    fn render_colgroup_returns_none_when_no_column_has_a_width() {
        let columns = vec![column(Parameters::new()), column(Parameters::new())];
        assert_eq!(render_colgroup(Some(&columns)), None);
    }

    #[test]
    fn render_colgroup_emits_a_bare_col_for_a_column_with_no_width() {
        let mut widened = Parameters::new();
        widened.insert(
            "width".to_string(),
            ParameterValue::String("20%".to_string()),
        );
        let columns = vec![column(widened), column(Parameters::new())];

        assert_eq!(
            render_colgroup(Some(&columns)).unwrap(),
            "<colgroup>\n<col style=\"width: 20%\">\n<col>\n</colgroup>\n"
        );
    }

    #[test]
    fn render_row_skips_a_column_occupied_by_a_rowspan_from_above() {
        let a = cell(TableCellKind::ColumnLabel, 1, 1, 0, 0);
        let c = cell(TableCellKind::ColumnLabel, 1, 1, 2, 2);
        let occupied = HashSet::from([1]);

        let row = render_row(vec![&a, &c], 3, &occupied, CssMode::Inline);

        assert_eq!(row, "<tr><th></th><th></th></tr>\n");
    }

    #[test]
    fn render_row_gives_a_filler_the_same_classes_as_its_row_siblings() {
        let a = cell(TableCellKind::Spanner, 0, 0, 0, 0)
            .with_classes(vec![TableCellClass::SpannerOuter]);
        let b = cell(TableCellKind::Spanner, 0, 0, 2, 2)
            .with_classes(vec![TableCellClass::SpannerOuter]);

        let row = render_row(vec![&a, &b], 3, &HashSet::new(), CssMode::Class);

        assert_eq!(
            row,
            "<tr><th class=\"ggsql_spanner_outer\"></th>\
             <th class=\"ggsql_spanner_outer\"></th>\
             <th class=\"ggsql_spanner_outer\"></th></tr>\n"
        );
    }

    #[test]
    fn write_table_errors_when_a_header_cell_is_not_above_every_body_cell() {
        let cells = vec![
            cell(TableCellKind::Body, 0, 0, 0, 0),
            cell(TableCellKind::ColumnLabel, 0, 0, 1, 1),
        ];

        assert!(HtmlWriter::new().write_table(&cells, None, None).is_err());
    }

    #[test]
    fn write_table_renders_a_rowspan_cell_once_and_skips_it_in_the_next_row() {
        // Column 1 has no spanner, so its label rowspans up through row 0
        // instead of row 0 getting a blank filler there.
        let cells = vec![
            cell(TableCellKind::Spanner, 0, 0, 0, 0),
            cell(TableCellKind::ColumnLabel, 0, 1, 1, 1),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
            cell(TableCellKind::Body, 2, 2, 0, 0),
            cell(TableCellKind::Body, 2, 2, 1, 1),
        ];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&cells, None, None)
        .unwrap();

        assert_eq!(
            html,
            "<table>\n\
             <thead>\n\
             <tr><th></th><th rowspan=\"2\"></th></tr>\n\
             <tr><th></th></tr>\n\
             </thead>\n\
             <tbody>\n\
             <tr><td></td><td></td></tr>\n\
             </tbody>\n\
             </table>"
        );
    }
}

#[cfg(test)]
#[cfg(feature = "duckdb")]
mod tests {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};

    #[test]
    fn test_write_table_renders_rows_and_escapes_html() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES (1, '<b>a</b>'), (2, 'b')) AS t(id, name)",
            )
            .unwrap();
        let spec = reader.execute("TABULATE FROM sales").unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        // Precise per-dtype alignment is covered directly by
        // resolve_column_properties's/discretise_hjust's own tests — this
        // just checks the columns render and escaping works end to end.
        assert!(html.starts_with("<style>"));
        assert!(html.contains("<table>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        // Class mode: "id"/1 is numeric (right), "name" is text (left) —
        // alignment is a class on the cell, declarations in the <style>
        // block, nothing inline.
        assert!(html.contains("<td class=\"ggsql_row ggsql_right\">1</td>"));
        assert!(html.contains("ggsql_left"));
        assert!(html
            .contains(".ggsql_right { text-align: right; font-variant-numeric: tabular-nums; }"));
        assert!(!html.contains("style=\"text-align"));
        assert!(html.contains("&lt;b&gt;a&lt;/b&gt;"));
        assert!(!html.contains("<b>a</b>"));
    }

    #[test]
    fn test_write_table_inline_mode_keeps_declarations_on_the_cells() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader.execute("TABULATE FROM sales").unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=inline"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        assert!(html.starts_with("<table>"));
        assert!(!html.contains("<style>"));
        assert!(!html.contains("class="));
        assert!(html.contains("style=\"text-align: right")); // "id"/1, numeric
        assert!(html.contains("style=\"text-align: left")); // "name", text
    }

    #[test]
    fn test_write_table_renders_a_spanner_row_with_colspan() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a', 10)) AS t(id, name, amount)",
            )
            .unwrap();
        let spec = reader
            .execute("TABULATE FROM sales SPAN 'Info' ACROSS id, name")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        // "amount" has no spanner, so its label stretches up into the
        // spanner row (rowspan) instead of a blank filler cell there.
        // Alignment styling is incidental here (amount/id are numeric) and
        // covered precisely by resolve_column_properties's own tests — this
        // checks colspan/rowspan/ordering, not exact style content. The
        // single spanner level is the topmost one, hence `spanner_outer`.
        assert!(html.contains(
            "<tr><th colspan=\"2\" class=\"ggsql_spanner_outer\">Info</th><th rowspan=\"2\""
        ));
        assert!(html.contains(">amount</th>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        // The spanner row renders above the column-label row.
        assert!(html.find("Info").unwrap() < html.find(">id</th>").unwrap());
    }

    #[test]
    fn test_write_table_renders_a_colgroup_for_a_formats_width() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader
            .execute("TABULATE FROM sales FORMAT id SETTING width => '20%'")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        assert!(html.contains("<colgroup>"));
        assert!(html.contains("<col style=\"width: 20%\">"));
        // "name" has no FORMAT width, so its <col> stays bare.
        assert!(html.contains("<col>"));
        // <colgroup> comes before <thead>, per the HTML spec.
        assert!(html.find("<colgroup>").unwrap() < html.find("<thead>").unwrap());
    }

    #[test]
    fn test_write_table_omits_tbody_for_a_zero_row_result() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader
            .execute("SELECT * FROM sales WHERE 1 = 0 TABULATE")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        assert!(html.contains("<thead>"));
        assert!(!html.contains("<tbody>"));
    }

    #[test]
    fn test_write_plot_is_unsupported() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap();

        let writer = HtmlWriter::new();
        assert!(writer.render(&spec).is_err());
    }
}
