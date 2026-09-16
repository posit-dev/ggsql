//! A minimal HTML table writer.
//!
//! Renders a `ResolvedTable`'s cells as a bare `<table>` — no styling, no
//! footnotes, since `Table` has no fields to describe those yet. Spanner
//! rows are rendered (as `colspan`, one `<tr>` per level, above the column
//! labels), but a column with no spanner at a given level gets a blank
//! filler cell rather than a `rowspan`-merged one — see `render_row`. This
//! is a stub to prove the Table → writer plumbing end to end, not the real
//! grammar-of-tables output; it deliberately does not reuse
//! `ggsql-jupyter`'s existing `dataframe_to_html`, which works directly off
//! a `DataFrame` rather than resolved `TableCell`s.

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::util::escape_html;
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result, TableCell};

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

    fn write_table(&self, cells: &[TableCell]) -> Result<String> {
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

        let mut html = String::from("<table>\n");

        if !header_rows.is_empty() {
            html.push_str("<thead>\n");
            for (_, row) in header_rows {
                html.push_str(&render_row(row, ncol));
            }
            html.push_str("</thead>\n");
        }

        if !body_rows.is_empty() {
            html.push_str("<tbody>\n");
            for (_, row) in body_rows {
                html.push_str(&render_row(row, ncol));
            }
            html.push_str("</tbody>\n");
        }

        html.push_str("</table>");

        Ok(html)
    }
}

/// Render one `TableCell` as an HTML tag — `<th>`/`<td>` from
/// `cell.is_header()`, with a `colspan` attribute only when the cell
/// actually covers more than one column.
fn render_cell(cell: &TableCell) -> String {
    let tag = if cell.is_header() { "th" } else { "td" };
    let colspan = cell.width();
    let attrs = if colspan > 1 {
        format!(" colspan=\"{colspan}\"")
    } else {
        String::new()
    };
    format!("<{tag}{attrs}>{}</{tag}>", escape_html(&cell.content))
}

/// Render one row — cells sharing a `top` — as `<tr>...</tr>`, walking every
/// column position from `0..ncol` and padding any gap with a synthesized
/// empty cell of the same kind.
fn render_row(mut cells: Vec<&TableCell>, ncol: usize) -> String {
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
        // Check if column has cell
        if i < cells.len() && cells[i].left == col {
            html.push_str(&render_cell(cells[i]));
            col = cells[i].right + 1;
            i += 1;
        } else {
            // A gap (a spanner not reaching every column, or one fragmented
            // by another spanner's occupancy) renders through `render_cell`
            // too, so one place decides which tag a `TableCellKind` gets,
            // not two.
            let filler = TableCell {
                kind,
                top: cells[0].top,
                bottom: cells[0].top,
                left: col,
                right: col,
                content: String::new(),
            };
            html.push_str(&render_cell(&filler));
            col += 1;
        }
    }

    html.push_str("</tr>\n");
    html
}

#[cfg(test)]
mod render_row_tests {
    use super::*;

    #[test]
    fn render_row_returns_empty_string_for_no_cells() {
        assert_eq!(render_row(Vec::new(), 3), "");
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

        assert!(html.starts_with("<table>"));
        assert!(html.contains("<th>id</th>"));
        assert!(html.contains("<th>name</th>"));
        assert!(html.contains("<td>1</td>"));
        assert!(html.contains("&lt;b&gt;a&lt;/b&gt;"));
        assert!(!html.contains("<b>a</b>"));
    }

    #[test]
    fn test_write_table_renders_a_spanner_row_with_colspan_and_fills_gaps() {
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

        // The spanner covers id+name (colspan 2), with an empty header cell
        // filling the gap above "amount", which has no spanner over it.
        assert!(html.contains("<tr><th colspan=\"2\">Info</th><th></th></tr>"));
        assert!(html.contains("<th>id</th>"));
        assert!(html.contains("<th>name</th>"));
        assert!(html.contains("<th>amount</th>"));
        // The spanner row renders above the column-label row.
        assert!(html.find("Info").unwrap() < html.find("<th>id</th>").unwrap());
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
