//! Aligned, CSV-like table formatting for query results with nothing to draw.
//!
//! Shared by `exec`/`run` (the no-VISUALISE fallback) and the REPL, so a plain
//! SQL statement looks the same however it was submitted.

use ggsql::parser;
use ggsql::reader::Reader;

/// Run `query`'s SQL part and format up to `max_rows` rows as an aligned,
/// CSV-like table.
pub fn format(query: &str, reader: &dyn Reader, max_rows: usize) -> Result<String, String> {
    let source_tree =
        parser::SourceTree::new(query).map_err(|e| format!("Failed to parse query: {e}"))?;

    let sql_part = source_tree.extract_sql().unwrap_or_default();

    let data = reader
        .execute_sql(&sql_part)
        .map_err(|e| format!("Failed to execute SQL query: {e}"))?;

    let nrow = data.height().min(max_rows);
    let ncol = data.width();
    let colnames = data.get_column_names();

    // We add an extra 'row' for the column names
    let mut rows: Vec<String> = vec![String::from(""); nrow + 1];

    let columns = data.get_columns();
    for (col_id, (col_name, column_data)) in colnames.iter().zip(columns.iter()).enumerate() {
        let mut width = col_name.chars().count();

        // End last column without comma
        let suffix = if col_id == ncol - 1 { "" } else { ", " };

        // Prepopulate formatted column with column name
        let mut col_fmt: Vec<String> = vec![format!("{}{}", col_name, suffix)];

        // Format every cell in column, tracking width
        for row_idx in 0..nrow {
            let cell = ggsql::array_util::value_to_string(column_data, row_idx);
            let cell_fmt = format!("{}{}", cell, suffix);
            let nchar = cell_fmt.chars().count();
            if nchar > width {
                width = nchar;
            }
            col_fmt.push(cell_fmt);
        }
        // Pad strings with spaces
        let col_fmt: Vec<String> = col_fmt
            .into_iter()
            .map(|s| format!("{:width$}", s, width = width))
            .collect();

        // Push columns to row string
        for (row, fmt) in rows.iter_mut().zip(col_fmt.iter()) {
            row.push_str(fmt.as_str());
        }
    }

    Ok(rows.join("\n"))
}
