//! A minimal HTML table writer.
//!
//! Renders a `ResolvedTable`'s cells as a bare `<table>` — inline `style`
//! attributes from each cell's resolved `FORMAT` properties, but no
//! footnotes, since `Table` has no fields to describe those yet. Spanner
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
use crate::{DataFrame, GgsqlError, Plot, Result, TableCell, TableColumn, TableRow};

/// Renders a resolved table as a bare HTML `<table>`. Does not support plots.
#[derive(Debug, Default)]
pub struct HtmlWriter;

impl HtmlWriter {
    /// Create a new HtmlWriter.
    pub fn new() -> Self {
        Self
    }
}

impl Writer for HtmlWriter {
    type Output = String;

    /// This writer takes no options and rejects any.
    fn from_options(options: &WriterOptions) -> Result<Self> {
        options.reject_unknown(&[])?;
        Ok(Self::new())
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
        // Not consumed yet — no property needs whole-column/whole-row
        // rendering (like `width`) rather than a per-cell one yet.
        let _ = (columns, rows);

        let ncol = cells
            .iter()
            .map(|cell| cell.right)
            .max()
            .map_or(0, |r| r + 1);

        // BTreeMap because essentially Vec<Vec<&TableCell>> but compact for missing rows
        let mut header_rows: BTreeMap<usize, Vec<&TableCell>> = BTreeMap::new();
        let mut body_rows: BTreeMap<usize, Vec<&TableCell>> = BTreeMap::new();

        // Sorting Hat: Hmmm... Yes... BODY ROW!!! *applause*
        for cell in cells {
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
        let occupied = occupied_columns_per_row(cells, nrow);

        let mut html = String::from("<table>\n");

        if !header_rows.is_empty() {
            html.push_str("<thead>\n");
            for (row, row_cells) in header_rows {
                html.push_str(&render_row(row_cells, ncol, &occupied[row]));
            }
            html.push_str("</thead>\n");
        }

        if !body_rows.is_empty() {
            html.push_str("<tbody>\n");
            for (row, row_cells) in body_rows {
                html.push_str(&render_row(row_cells, ncol, &occupied[row]));
            }
            html.push_str("</tbody>\n");
        }

        html.push_str("</table>");

        Ok(html)
    }
}

/// Render one `TableCell` as an HTML tag — `<th>`/`<td>` from
/// `cell.is_header()`, with a `colspan`/`rowspan` attribute only when the
/// cell actually spans more than one column/row, and a `style` attribute
/// only when `cell.properties` resolves to one.
fn render_cell(cell: &TableCell) -> String {
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
    if let Some(style) = cell_style(&cell.properties) {
        attrs.push_str(&format!(" style=\"{style}\""));
    }
    format!("<{tag}{attrs}>{}</{tag}>", escape_html(&cell.content))
}

/// Translate a cell's resolved `FORMAT` properties into a `style` attribute
/// value, or `None` if none of them produce a CSS declaration.
fn cell_style(properties: &Parameters) -> Option<String> {
    let mut declarations = Vec::new();

    if let Some(align) = text_align(properties.get("hjust")) {
        declarations.push(format!("text-align: {align}"));
        // Right-aligned data reads as numeric — keep digit widths uniform
        // so they still line up under one another.
        if align == "right" {
            declarations.push("font-variant-numeric: tabular-nums".to_string());
        }
    }

    (!declarations.is_empty()).then(|| declarations.join("; "))
}

/// Map a resolved `hjust` number to a CSS `text-align` keyword, bucketed
/// with the same `0.25`/`0.75` thresholds `VegaLiteWriter`'s `convert_hjust`
/// uses for its own `align` conversion, so `hjust` means the same alignment
/// in both writers. `resolve_column_properties` already standardises every
/// `hjust` (however the user wrote it) to a number before it reaches a
/// `TableCell`, so a non-`Number` here is unreachable in practice.
fn text_align(hjust: Option<&ParameterValue>) -> Option<&'static str> {
    match hjust? {
        ParameterValue::Number(n) if *n <= 0.25 => Some("left"),
        ParameterValue::Number(n) if *n >= 0.75 => Some("right"),
        ParameterValue::Number(_) => Some("center"),
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
fn render_row(mut cells: Vec<&TableCell>, ncol: usize, occupied: &HashSet<usize>) -> String {
    // write_table's only caller never hands this an empty `cells` (an entry
    // is only ever created already holding a cell), but an empty row has no
    // `kind` to synthesize fillers with, so this returns early rather than
    // assume that guarantee holds forever.
    if cells.is_empty() {
        return String::new();
    }

    // The walk below depends on `left`-ascending order.
    cells.sort_by_key(|cell| cell.left);
    let kind = cells[0].kind;
    let mut html = String::from("<tr>");

    // `ncol` comes from the whole table, not this row's own cells — a row
    // that doesn't reach the last column must still pad out to it.
    let mut col = 0;
    let mut i = 0;
    while col < ncol {
        if occupied.contains(&col) {
            col += 1;
        } else if i < cells.len() && cells[i].left == col {
            html.push_str(&render_cell(cells[i]));
            col = cells[i].right + 1;
            i += 1;
        } else {
            // A gap (a spanner not reaching every column, or one fragmented
            // by another spanner's occupancy) renders through `render_cell`
            // too, so one place decides which tag a `TableCellKind` gets,
            // not two.
            let filler = TableCell::new(kind, cells[0].top, cells[0].top, col, col, String::new());
            html.push_str(&render_cell(&filler));
            col += 1;
        }
    }

    html.push_str("</tr>\n");
    html
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::TableCellKind;

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
    fn render_row_returns_empty_string_for_no_cells() {
        assert_eq!(render_row(Vec::new(), 3, &HashSet::new()), "");
    }

    #[test]
    fn render_cell_emits_rowspan_when_height_is_greater_than_one() {
        let c = cell(TableCellKind::ColumnLabel, 0, 1, 0, 0);
        assert_eq!(render_cell(&c), "<th rowspan=\"2\"></th>");
    }

    #[test]
    fn render_cell_emits_a_style_attribute_from_properties() {
        // The exact declarations a given `hjust` produces are `cell_style`'s
        // own tests' job — this just checks render_cell embeds whatever
        // `cell_style` returns as a `style="..."` attribute.
        let mut c = cell(TableCellKind::Body, 0, 0, 0, 0);
        c.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let style = cell_style(&c.properties).unwrap();
        assert_eq!(render_cell(&c), format!("<td style=\"{style}\"></td>"));
    }

    #[test]
    fn text_align_buckets_a_resolved_hjust_number() {
        assert_eq!(text_align(None), None);
        assert_eq!(text_align(Some(&ParameterValue::Number(0.0))), Some("left"));
        assert_eq!(text_align(Some(&ParameterValue::Number(0.1))), Some("left"));
        assert_eq!(
            text_align(Some(&ParameterValue::Number(0.5))),
            Some("center")
        );
        assert_eq!(
            text_align(Some(&ParameterValue::Number(0.9))),
            Some("right")
        );
        assert_eq!(
            text_align(Some(&ParameterValue::Number(1.0))),
            Some("right")
        );
    }

    #[test]
    fn cell_style_only_adds_tabular_nums_when_right_aligned() {
        let mut left = Parameters::new();
        left.insert("hjust".to_string(), ParameterValue::Number(0.0));
        assert_eq!(cell_style(&left), Some("text-align: left".to_string()));

        let mut right = Parameters::new();
        right.insert("hjust".to_string(), ParameterValue::Number(1.0));
        assert_eq!(
            cell_style(&right),
            Some("text-align: right; font-variant-numeric: tabular-nums".to_string())
        );

        assert_eq!(cell_style(&Parameters::new()), None);
    }

    #[test]
    fn render_row_skips_a_column_occupied_by_a_rowspan_from_above() {
        let a = cell(TableCellKind::ColumnLabel, 1, 1, 0, 0);
        let c = cell(TableCellKind::ColumnLabel, 1, 1, 2, 2);
        let occupied = HashSet::from([1]);

        let row = render_row(vec![&a, &c], 3, &occupied);

        assert_eq!(row, "<tr><th></th><th></th></tr>\n");
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

        let html = HtmlWriter::new().write_table(&cells, None, None).unwrap();

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
        // resolve_column_properties's/cell_style's own tests — this just
        // checks the columns render and escaping works end to end.
        assert!(html.starts_with("<table>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        assert!(html.contains(">1</td>"));
        assert!(html.contains("text-align: right")); // "id"/1, numeric
        assert!(html.contains("text-align: left")); // "name", text
        assert!(html.contains("&lt;b&gt;a&lt;/b&gt;"));
        assert!(!html.contains("<b>a</b>"));
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
        // checks colspan/rowspan/ordering, not exact style content.
        assert!(html.contains("<tr><th colspan=\"2\">Info</th><th rowspan=\"2\""));
        assert!(html.contains(">amount</th>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        // The spanner row renders above the column-label row.
        assert!(html.find("Info").unwrap() < html.find(">id</th>").unwrap());
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
