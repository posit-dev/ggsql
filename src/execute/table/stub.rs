//! `TABULATE FORMAT STUB` resolution: moving stub columns to the front and
//! building their `StubHead`/`StubRowLabel` cells.

use super::layout::TableColumn;
use super::{TableCell, TableCellKind, TableClass};
use crate::array_util::value_to_string;
use crate::DataFrame;

/// Move every STUB-targeted column to the front, preserving each group's
/// relative order. Runs after `reorder_table_columns`, as the final word on
/// column position — a SPAN can never cover a stub column
/// (`Table::validate_span_stub_boundary`).
pub(crate) fn move_stub_columns(columns: Vec<TableColumn>) -> Vec<TableColumn> {
    let (mut stub, body): (Vec<_>, Vec<_>) = columns.into_iter().partition(TableColumn::is_stub);
    stub.extend(body);
    stub
}

/// Build one `StubHead` cell per stub column, numbered from `top == 0`.
/// Skips every non-stub column.
pub(crate) fn create_stubhead(columns: &[TableColumn]) -> Vec<TableCell> {
    columns
        .iter()
        .enumerate()
        .filter(|(_, column)| column.is_stub())
        .map(|(index, column)| {
            TableCell::new(
                TableCellKind::StubHead,
                0,
                0,
                index,
                index,
                column.label.clone(),
            )
            .with_properties(column.properties.clone())
            .with_classes(vec![TableClass::StubHead])
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
                TableCell::new(
                    TableCellKind::StubRowLabel,
                    row,
                    row,
                    index,
                    index,
                    value_to_string(array, row),
                )
                .with_properties(column.properties.clone())
                .with_classes(vec![TableClass::Stub]),
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
    fn move_stub_columns_moves_stub_columns_to_the_front_preserving_relative_order() {
        let columns = vec![
            column("a", "a"),
            stub_column("b"),
            column("c", "c"),
            stub_column("d"),
        ];

        let result = move_stub_columns(columns);

        assert_eq!(
            result.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["b", "d", "a", "c"]
        );
    }

    #[test]
    fn move_stub_columns_is_a_no_op_with_no_stub_columns() {
        let columns = vec![column("a", "a"), column("b", "b")];

        let result = move_stub_columns(columns);

        assert_eq!(
            result.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
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
