//! A minimal HTML table writer.
//!
//! Renders a `ResolvedTable`'s body as a bare `<table>` — no styling, no
//! headings/spanners/footnotes, since `Table` has no fields to describe
//! those yet. This is a stub to prove the Table → writer plumbing end to
//! end, not the real grammar-of-tables output; it deliberately does not
//! reuse `ggsql-jupyter`'s existing `dataframe_to_html`, since that's built
//! around `DataFrame` specifically, and `ResolvedTable.body`'s type is
//! itself still provisional (see the note on that field).

use std::collections::HashMap;

use crate::array_util::value_to_string;
use crate::util::escape_html;
use crate::writer::{Writer, WriterOptions};
use crate::{DataFrame, GgsqlError, Plot, Result, Table};

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

    fn write_table(&self, _table: &Table, body: &DataFrame) -> Result<String> {
        let mut html = String::from("<table>\n<thead>\n<tr>");
        for name in body.get_column_names() {
            html.push_str(&format!("<th>{}</th>", escape_html(&name)));
        }
        html.push_str("</tr>\n</thead>\n<tbody>\n");

        let columns = body.get_columns();
        for row in 0..body.height() {
            html.push_str("<tr>");
            for column in columns {
                html.push_str(&format!(
                    "<td>{}</td>",
                    escape_html(&value_to_string(column, row))
                ));
            }
            html.push_str("</tr>\n");
        }
        html.push_str("</tbody>\n</table>");

        Ok(html)
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
    fn test_write_plot_is_unsupported() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap();

        let writer = HtmlWriter::new();
        assert!(writer.render(&spec).is_err());
    }
}
