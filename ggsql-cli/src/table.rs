//! Aligned, CSV-like table formatting for query results with nothing to draw.
//!
//! Shared by `exec`/`run` (the no-VISUALISE fallback) and the REPL, so a plain
//! SQL statement looks the same however it was submitted. A result truncated
//! at `max_rows` says how many rows were left out.

use ggsql::parser;
use ggsql::reader::Reader;
use ggsql::DataFrame;

/// Run `query`'s SQL part and format up to `max_rows` rows as an aligned,
/// CSV-like table.
pub fn format(query: &str, reader: &dyn Reader, max_rows: usize) -> Result<String, String> {
    let source_tree =
        parser::SourceTree::new(query).map_err(|e| format!("Failed to parse query: {e}"))?;

    let sql_part = source_tree.extract_sql().unwrap_or_default();

    let data = reader
        .execute_sql(&sql_part)
        .map_err(|e| format!("Failed to execute SQL query: {e}"))?;

    Ok(format_frame(&data, max_rows))
}

/// Format up to `max_rows` of `data` as an aligned, CSV-like table.
fn format_frame(data: &DataFrame, max_rows: usize) -> String {
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

    let hidden = data.height() - nrow;
    if hidden > 0 {
        rows.push(format!(
            "… {hidden} more row{} not shown",
            if hidden == 1 { "" } else { "s" }
        ));
    }

    rows.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggsql::array_util::{new_i64_array, new_str_array};
    use ggsql::DataFrame;

    fn frame() -> DataFrame {
        DataFrame::new(vec![
            (
                "name",
                new_str_array(vec![Some("alpha"), Some("b"), Some("gamma")]),
            ),
            ("count", new_i64_array(vec![Some(10), Some(2), Some(300)])),
        ])
        .unwrap()
    }

    #[test]
    fn rows_and_header_are_aligned() {
        let out = format_frame(&frame(), 100);
        assert_eq!(
            out,
            "name,  count\n\
             alpha, 10   \n\
             b,     2    \n\
             gamma, 300  "
        );
    }

    #[test]
    fn truncated_results_have_a_footer() {
        let out = format_frame(&frame(), 2);
        assert_eq!(out.lines().last().unwrap(), "… 1 more row not shown");

        let out = format_frame(&frame(), 1);
        assert_eq!(out.lines().last().unwrap(), "… 2 more rows not shown");

        // Nothing truncated, no footer.
        let out = format_frame(&frame(), 100);
        assert!(!out.contains("not shown"));
    }
}
