//! `TABULATE FORMAT STUB` resolution: building the `StubHead`/`StubRowLabel`
//! cells.

use crate::array_util::value_to_string;
use crate::table::{TableCell, TableColumn};
use crate::DataFrame;

/// Build one `StubHead` cell per stub column, numbered from `top == 0`.
/// Skips every non-stub column.
pub(crate) fn create_stubhead(columns: &[TableColumn]) -> Vec<TableCell> {
    columns
        .iter()
        .enumerate()
        .filter(|(_, column)| column.is_stub())
        .map(|(index, column)| {
            TableCell::new_stub_head(index, column.label.clone())
                .with_properties(column.properties.clone())
        })
        .collect()
}

/// Build one `StubRowLabel` cell per `DataFrame` value in every stub
/// column, numbered from `top == 0`. Skips every non-stub column.
pub(crate) fn create_row_labels(df: &DataFrame, columns: &[TableColumn]) -> Vec<TableCell> {
    let mut cells = Vec::new();

    for (index, column) in columns.iter().enumerate() {
        if !column.is_stub() {
            continue;
        }

        let array = df
            .column(&column.name)
            .expect("TableColumn.name always names a column of df");

        for row in 0..df.height() {
            cells.push(
                TableCell::new_stub_row_label(row, index, value_to_string(array, row))
                    .with_properties(column.properties.clone()),
            );
        }
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;
    use crate::plot::Parameters;
    use crate::table::TableCellKind;
    use crate::ColumnSection;

    fn column(name: &str, label: &str) -> TableColumn {
        TableColumn {
            name: name.to_string(),
            label: label.to_string(),
            properties: Parameters::new(),
            target: ColumnSection::Body,
        }
    }

    fn stub_column(name: &str) -> TableColumn {
        TableColumn {
            target: ColumnSection::Stub,
            ..column(name, name)
        }
    }

    #[test]
    fn create_stubhead_only_builds_stub_columns() {
        let columns = vec![stub_column("region"), column("sales", "sales")];

        let labels = create_stubhead(&columns);

        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].kind, TableCellKind::StubHead);
        assert_eq!(labels[0].left, 0);
        assert_eq!(labels[0].content, "region");
    }

    #[test]
    fn create_stubhead_is_empty_with_no_stub_columns() {
        let columns = vec![column("id", "id")];

        assert!(create_stubhead(&columns).is_empty());
    }

    #[test]
    fn create_row_labels_only_builds_stub_columns() {
        let frame = df! {
            "region" => vec!["north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2i32],
        }
        .unwrap();
        let columns = vec![stub_column("region"), column("sales", "sales")];

        let labels = create_row_labels(&frame, &columns);

        assert_eq!(labels.len(), 2);
        assert!(labels
            .iter()
            .all(|cell| cell.kind == TableCellKind::StubRowLabel));
        assert_eq!(labels[0].content, "north");
        assert_eq!(labels[0].left, 0);
        assert_eq!(labels[1].content, "south");
    }

    #[test]
    fn create_row_labels_is_empty_with_no_stub_columns() {
        let frame = df! { "sales" => vec![1i32] }.unwrap();
        let columns = vec![column("sales", "sales")];

        assert!(create_row_labels(&frame, &columns).is_empty());
    }
}
