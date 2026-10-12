//! Table-cell resolution: turns already-resolved columns, spans and a
//! `FORMAT`-applied `DataFrame` into the `TableLayout` `ResolvedTable`
//! holds, assembling parts with the `create_*` helpers from
//! `crate::table::layout`.

use std::collections::HashMap;

use super::format::resolve_column_properties;
use super::stub::{create_row_labels, create_stubhead};
use crate::array_util::value_to_string;
use crate::plot::Labels;
use crate::table::{
    compose_header, create_caption, create_column_labels, create_heading, create_spanners,
    move_stub_columns, reorder_table_columns, TableCell, TableClass, TableColumn, TableLayout,
    TableRow,
};
use crate::{DataFrame, Format, Result, Spanner};

/// Resolve a table's columns: one `TableColumn` per `DataFrame` column
/// (labelled, with `formats`' resolved `SETTING` properties), reordered for
/// any `gather`-ing spanner and then for STUB columns. `spans` must already
/// be resolved (`Table::resolve_spanners`).
pub(super) fn setup_columns(
    df: &DataFrame,
    spans: &[Spanner],
    labels: &Labels,
    formats: &HashMap<String, Format>,
) -> Vec<TableColumn> {
    let columns = create_table_columns(df, labels, formats);
    let columns = reorder_table_columns(columns, spans);
    move_stub_columns(columns)
}

/// Build the resolved `TableLayout` for a table from its already-resolved
/// `columns` and already-`FORMAT`-applied `df` — spanner rows, column
/// labels, and body, composed together and validated.
pub(super) fn build_cells(
    df: &DataFrame,
    columns: Vec<TableColumn>,
    spans: &[Spanner],
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
) -> Result<TableLayout> {
    let spanners = create_spanners(&columns, spans);
    // Every label empty means `LABEL col => NULL` (or `=> ''`) on every
    // column, since an unlabeled column keeps its (non-empty) name — omit
    // the row rather than render blank header cells.
    let column_labels = if columns.iter().all(|c| c.label.is_empty()) {
        TableLayout::new(columns.clone(), Vec::new(), Vec::new())
    } else {
        let mut label_cells = create_stubhead(&columns);
        label_cells.extend(create_column_labels(&columns));
        TableLayout::new(
            columns.clone(),
            vec![TableRow {
                classes: vec![TableClass::ColHeadingRow],
                ..TableRow::header()
            }],
            label_cells,
        )
    };
    let header = compose_header(spanners, column_labels);
    let heading = create_heading(title, subtitle, &columns);
    let header = heading.rowbind(header);
    let mut body_cells = create_row_labels(df, &columns);
    body_cells.extend(create_body(df, &columns));
    let table_body = TableLayout::new(
        columns.clone(),
        vec![TableRow::body(); df.height()],
        body_cells,
    );
    let layout = header.rowbind(table_body);
    let caption_layout = create_caption(caption, &columns);
    let layout = layout.rowbind(caption_layout);

    layout.validate()?;

    Ok(layout)
}

/// Build one `TableColumn` per `DataFrame` column, in the `DataFrame`'s own
/// order — `reorder_table_columns` is what may reorder this list
/// afterward, not this function. `labels` is the one authority for an
/// explicit column label; `formats` (already reshaped to one `Format` per
/// column by `setup_formats`) is the one authority for its properties.
fn create_table_columns(
    df: &DataFrame,
    labels: &Labels,
    formats: &HashMap<String, Format>,
) -> Vec<TableColumn> {
    let mut columns = Vec::new();
    for name in df.get_column_names() {
        // Not stored on TableColumn: nothing needs it once `properties`
        // (which may default from it) is resolved.
        let dtype = df
            .column(&name)
            .expect("name comes from df's own columns")
            .data_type();
        let format = formats.get(&name);
        let properties = resolve_column_properties(dtype, format);
        let target = format.map(|f| f.target).unwrap_or_default();
        let label = match labels.labels.get(&name) {
            // A stub head is a row-label header, not a data column header —
            // default it blank rather than to the column's own name.
            None if target.is_stub() => String::new(),
            None => name.clone(),
            Some(None) => String::new(),
            Some(Some(label)) => label.clone(),
        };
        columns.push(TableColumn {
            name,
            label,
            properties,
            target,
        });
    }
    columns
}

/// Build one `Body` cell per `DataFrame` value, numbered from `top == 0`, in
/// `columns`' order. Looks each column up in `df` **by name**, not position,
/// since `columns` may already be reordered relative to `df`. Skips every
/// STUB column — `create_row_labels` builds those.
fn create_body(df: &DataFrame, columns: &[TableColumn]) -> Vec<TableCell> {
    let mut cells = Vec::new();

    for (index, column) in columns.iter().enumerate() {
        if column.is_stub() {
            continue;
        }

        // Looked up once per column, outside the row loop: `DataFrame::column`
        // is an `O(ncol)` scan over the schema, so doing this per row instead
        // would cost `O(nrow * ncol)` lookups rather than `O(ncol)`.
        let array = df
            .column(&column.name)
            .expect("TableColumn.name always names a column of df");

        for row in 0..df.height() {
            cells.push(
                TableCell::new_body(row, index, value_to_string(array, row))
                    .with_properties(column.properties.clone()),
            );
        }
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::super::format::{apply_formats, setup_formats};
    use super::*;
    use crate::df;
    use crate::plot::{ParameterValue, Parameters};
    use crate::table::{extract_heading_labels, TableCellKind};
    use crate::{ColumnSection, GgsqlError, Spanner, Table};

    /// Runs the same steps `resolve_table_with_reader` does, minus the
    /// `Reader`/SQL execution — lets a test build a `Table`'s resolved cells
    /// directly against a `df!()`-built `DataFrame`.
    fn resolve_layout(df: &DataFrame, table: &Table) -> Result<TableLayout> {
        let mut labels = table.labels.clone();
        let (title, subtitle, caption) = extract_heading_labels(&mut labels);
        let column_names = df.get_column_names();
        let spans = table
            .resolve_spanners(&labels, Some(&column_names))
            .map_err(GgsqlError::ValidationError)?;
        let formats = setup_formats(df, &table.formats, &spans)?;
        let columns = setup_columns(df, &spans, &labels, &formats);
        let df = apply_formats(df, &formats)?;
        build_cells(
            &df,
            columns,
            &spans,
            title.as_deref(),
            subtitle.as_deref(),
            caption.as_deref(),
        )
    }

    /// `resolve_layout`, for the (common) tests that only care about cells.
    fn resolve_cells(df: &DataFrame, table: &Table) -> Result<Vec<TableCell>> {
        resolve_layout(df, table).map(|layout| layout.cells)
    }

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
    fn create_table_columns_resolves_default_suppress_and_override() {
        let frame = df! {
            "id" => vec![1i32],
            "name" => vec!["a".to_string()],
            "extra" => vec![true],
        }
        .unwrap();

        let mut labels = Labels::default();
        labels
            .labels
            .insert("id".to_string(), Some("ID".to_string()));
        labels.labels.insert("name".to_string(), None);
        // "extra" has no entry at all: no LABEL clause mentioned it.

        let columns = create_table_columns(&frame, &labels, &HashMap::new());

        assert_eq!(columns[0].name, "id");
        assert_eq!(columns[0].label, "ID"); // overridden
        assert_eq!(columns[1].label, ""); // explicitly suppressed
        assert_eq!(columns[2].label, "extra"); // absent: kept as-is
    }

    #[test]
    fn create_table_columns_defaults_a_stub_column_to_a_blank_label_but_label_can_override() {
        let frame = df! {
            "region" => vec!["north".to_string()],
        }
        .unwrap();
        let formats = HashMap::from([(
            "region".to_string(),
            Format {
                columns: vec!["region".to_string()],
                target: ColumnSection::Stub,
                settings: Parameters::new(),
                value_mapping: None,
                value_template: None,
            },
        )]);

        let columns = create_table_columns(&frame, &Labels::default(), &formats);
        assert_eq!(columns[0].label, "");

        let mut labels = Labels::default();
        labels
            .labels
            .insert("region".to_string(), Some("Region".to_string()));
        let columns = create_table_columns(&frame, &labels, &formats);
        assert_eq!(columns[0].label, "Region");
    }

    fn spanner_with(columns: &[&str], label: Option<&str>, settings: Parameters) -> Spanner {
        Spanner {
            id: label.unwrap_or("").to_string(),
            label: label.map(str::to_string),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            settings,
        }
    }

    fn spanner(columns: &[&str]) -> Spanner {
        spanner_with(columns, Some(""), Parameters::new())
    }

    fn spanner_with_setting(columns: &[&str], key: &str, value: ParameterValue) -> Spanner {
        let mut settings = Parameters::new();
        settings.insert(key.to_string(), value);
        spanner_with(columns, Some(""), settings)
    }

    fn labeled_spanner(columns: &[&str], label: &str) -> Spanner {
        spanner_with(columns, Some(label), Parameters::new())
    }

    #[test]
    fn create_body_numbers_rows_from_zero() {
        let frame = df! {
            "id" => vec![1i32, 2],
            "name" => vec!["a".to_string(), "b".to_string()],
        }
        .unwrap();
        let columns = create_table_columns(&frame, &Labels::default(), &HashMap::new());

        let body = create_body(&frame, &columns);

        assert_eq!(body.len(), 4);
        assert!(body.iter().all(|cell| cell.kind == TableCellKind::Body));
        // Column 0 ("id"): both rows, before column 1 starts — cells are
        // pushed column-major, not row-major (see create_body's inline
        // comment on why `array` is looked up once per column).
        assert_eq!(body[0].top, 0);
        assert_eq!(body[0].bottom, 0);
        assert_eq!(body[0].left, 0);
        assert_eq!(body[0].content, "1");
        assert_eq!(body[1].top, 1);
        assert_eq!(body[1].bottom, 1);
        assert_eq!(body[1].left, 0);
        assert_eq!(body[1].content, "2");
        // Column 1 ("name")
        assert_eq!(body[2].top, 0);
        assert_eq!(body[2].left, 1);
        assert_eq!(body[2].content, "a");
        assert_eq!(body[3].top, 1);
        assert_eq!(body[3].left, 1);
        assert_eq!(body[3].content, "b");
    }

    #[test]
    fn create_body_looks_up_columns_by_name_not_position() {
        // `columns` reordered relative to `frame`'s own column order —
        // `create_body` must follow `columns`, not `df`'s raw position, for
        // spanner-driven reordering to actually reach the body.
        let frame = df! {
            "id" => vec![1i32],
            "name" => vec!["a".to_string()],
        }
        .unwrap();
        let columns = vec![column("name", "name"), column("id", "id")];

        let body = create_body(&frame, &columns);

        assert_eq!(body[0].content, "a"); // "name" column, placed first
        assert_eq!(body[1].content, "1"); // "id" column, placed second
    }

    #[test]
    fn create_body_skips_stub_columns() {
        let frame = df! {
            "region" => vec!["north".to_string()],
            "sales" => vec![1i32],
        }
        .unwrap();
        let columns = vec![stub_column("region"), column("sales", "sales")];

        let body = create_body(&frame, &columns);

        assert_eq!(body.len(), 1);
        assert_eq!(body[0].content, "1");
        assert_eq!(body[0].left, 1);
    }

    #[test]
    fn build_cells_composes_spanners_labels_and_body_together() {
        let frame = df! {
            "a" => vec![1i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "G")];

        let cells = resolve_cells(&frame, &table).unwrap();

        let spanner_cell = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Spanner)
            .unwrap();
        assert_eq!(spanner_cell.top, 0);
        assert_eq!(spanner_cell.left, 0);
        assert_eq!(spanner_cell.right, 1);
        assert_eq!(spanner_cell.content, "G");

        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::ColumnLabel)
            .all(|c| c.top == 1));
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .all(|c| c.top == 2));
    }

    #[test]
    fn build_cells_overrides_a_spanners_default_label_via_labels_clause() {
        let frame = df! {
            "a" => vec![1i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "g")];
        table
            .labels
            .labels
            .insert("g".to_string(), Some("Group".to_string()));

        let cells = resolve_cells(&frame, &table).unwrap();

        let spanner_cell = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Spanner)
            .unwrap();
        assert_eq!(spanner_cell.content, "Group");
    }

    #[test]
    fn build_cells_suppresses_a_named_spanners_cell_via_labels_null() {
        let frame = df! {
            "a" => vec![1i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "g")];
        table.labels.labels.insert("g".to_string(), None);

        let cells = resolve_cells(&frame, &table).unwrap();

        assert!(!cells.iter().any(|c| c.kind == TableCellKind::Spanner));
    }

    #[test]
    fn build_cells_records_style_classes() {
        let frame = df! {
            "a" => vec![1i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "G")];

        let cells = resolve_cells(&frame, &table).unwrap();

        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::ColumnLabel)
            .all(|c| c.classes == [TableClass::ColHeading]));
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .all(|c| c.classes == [TableClass::Row]));
        // Spanner cells carry both spanner classes, centred by default.
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Spanner)
            .all(|c| c.classes
                == [
                    TableClass::Spanner,
                    TableClass::SpannerLabel,
                    TableClass::AlignCenter
                ]));
    }

    #[test]
    fn build_cells_applies_a_formats_renaming_to_body_cells() {
        let frame = df! {
            "price" => vec![0.0f64, 5.0],
        }
        .unwrap();
        let mut table = Table::new();
        table.formats = vec![crate::Format {
            columns: vec!["price".to_string()],
            target: crate::ColumnSection::Body,
            settings: Parameters::new(),
            value_mapping: Some(std::collections::HashMap::from([(
                "0".to_string(),
                Some("-".to_string()),
            )])),
            value_template: Some("${:num %.2f}".to_string()),
        }];

        let cells = resolve_cells(&frame, &table).unwrap();

        let mut body_cells: Vec<_> = cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .collect();
        body_cells.sort_by_key(|c| c.top);
        assert_eq!(body_cells[0].content, "-");
        assert_eq!(body_cells[1].content, "$5.00");
    }

    #[test]
    fn build_cells_defaults_hjust_by_dtype_on_label_and_body_cells() {
        let frame = df! {
            "price" => vec![1.0f64],
            "name" => vec!["a".to_string()],
        }
        .unwrap();
        let table = Table::new();

        let cells = resolve_cells(&frame, &table).unwrap();

        let hjust = |left: usize, kind: TableCellKind| {
            cells
                .iter()
                .find(|c| c.left == left && c.kind == kind)
                .unwrap()
                .properties
                .get("hjust")
                .cloned()
        };
        assert_eq!(
            hjust(0, TableCellKind::ColumnLabel),
            Some(ParameterValue::Number(1.0))
        );
        assert_eq!(
            hjust(0, TableCellKind::Body),
            Some(ParameterValue::Number(1.0))
        );
        assert_eq!(
            hjust(1, TableCellKind::Body),
            Some(ParameterValue::Number(0.0))
        );
    }

    #[test]
    fn build_cells_reorders_columns_for_a_spanner_before_building_labels_and_body() {
        let frame = df! {
            "a" => vec![1i32],
            "x" => vec![9i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "G")];

        let cells = resolve_cells(&frame, &table).unwrap();

        let mut label_cells: Vec<_> = cells
            .iter()
            .filter(|c| c.kind == TableCellKind::ColumnLabel)
            .collect();
        label_cells.sort_by_key(|c| c.left);
        let label_order: Vec<&str> = label_cells.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(label_order, vec!["a", "b", "x"]);

        let mut body_cells: Vec<_> = cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .collect();
        body_cells.sort_by_key(|c| c.left);
        let body_order: Vec<&str> = body_cells.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(body_order, vec!["1", "2", "9"]);
    }

    #[test]
    fn build_cells_surfaces_a_genuine_spanner_overlap_as_an_error() {
        // (a,b) auto-assigns level 1; (b,c) is explicitly pinned to level 1
        // too, and neither reordering nor level assignment resolves that —
        // build_cells should surface this via TableLayout::validate, not
        // silently produce a broken layout.
        let frame = df! {
            "a" => vec![1i32],
            "b" => vec![2i32],
            "c" => vec![3i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![
            spanner(&["a", "b"]),
            spanner_with_setting(&["b", "c"], "level", ParameterValue::Number(1.0)),
        ];

        assert!(resolve_cells(&frame, &table).is_err());
    }

    #[test]
    fn build_cells_omits_the_column_label_row_when_every_label_is_null() {
        let frame = df! {
            "id" => vec![1i32],
            "name" => vec!["a".to_string()],
        }
        .unwrap();
        let mut table = Table::new();
        table.labels.labels.insert("id".to_string(), None);
        table.labels.labels.insert("name".to_string(), None);

        let cells = resolve_cells(&frame, &table).unwrap();

        assert!(!cells.iter().any(|c| c.kind == TableCellKind::ColumnLabel));
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .all(|c| c.top == 0));
    }

    #[test]
    fn build_cells_stacks_title_and_subtitle_above_the_column_labels() {
        let frame = df! {
            "id" => vec![1i32],
        }
        .unwrap();
        let mut table = Table::new();
        table
            .labels
            .labels
            .insert("title".to_string(), Some("Title".to_string()));
        table
            .labels
            .labels
            .insert("subtitle".to_string(), Some("Subtitle".to_string()));

        let cells = resolve_cells(&frame, &table).unwrap();

        let title = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Title)
            .unwrap();
        let subtitle = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Subtitle)
            .unwrap();
        let label = cells
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel)
            .unwrap();
        let body = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Body)
            .unwrap();

        assert_eq!(
            (title.top, title.bottom, title.left, title.right),
            (0, 0, 0, 0)
        );
        assert_eq!(title.classes, [TableClass::Title]);
        assert_eq!((subtitle.top, subtitle.bottom), (1, 1));
        assert_eq!(subtitle.classes, [TableClass::Subtitle]);
        assert_eq!(label.top, 2);
        assert_eq!(body.top, 3);
    }

    #[test]
    fn build_cells_omits_a_heading_row_for_an_unset_title_or_subtitle() {
        let frame = df! {
            "id" => vec![1i32],
        }
        .unwrap();
        let mut table = Table::new();
        table
            .labels
            .labels
            .insert("subtitle".to_string(), Some("Subtitle only".to_string()));

        let cells = resolve_cells(&frame, &table).unwrap();

        assert!(!cells.iter().any(|c| c.kind == TableCellKind::Title));
        let subtitle = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Subtitle)
            .unwrap();
        // No title row above it, so subtitle takes row 0 itself.
        assert_eq!(subtitle.top, 0);
    }

    #[test]
    fn build_cells_places_the_caption_after_the_body() {
        let frame = df! {
            "id" => vec![1i32, 2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table
            .labels
            .labels
            .insert("caption".to_string(), Some("Source: test".to_string()));

        let cells = resolve_cells(&frame, &table).unwrap();

        let caption = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Caption)
            .unwrap();
        let max_other_bottom = cells
            .iter()
            .filter(|c| c.kind != TableCellKind::Caption)
            .map(|c| c.bottom)
            .max()
            .unwrap();
        assert_eq!(caption.top, max_other_bottom + 1);
        assert_eq!(caption.left, 0);
        assert_eq!(caption.right, 0);
        assert_eq!(caption.classes, [TableClass::Caption]);
    }

    #[test]
    fn resolve_layout_marks_heading_and_col_heading_rows() {
        let frame = df! {
            "id" => vec![1i32],
        }
        .unwrap();
        let mut table = Table::new();
        table
            .labels
            .labels
            .insert("title".to_string(), Some("Title".to_string()));
        table
            .labels
            .labels
            .insert("subtitle".to_string(), Some("Subtitle".to_string()));

        let layout = resolve_layout(&frame, &table).unwrap();

        // Title (row 0), Subtitle (row 1), ColumnLabel (row 2), Body (row 3).
        assert_eq!(layout.rows.len(), 4);
        assert_eq!(layout.rows[0].classes, vec![TableClass::Heading]);
        assert_eq!(layout.rows[1].classes, vec![TableClass::Heading]);
        assert_eq!(layout.rows[2].classes, vec![TableClass::ColHeadingRow]);
        assert!(layout.rows[3].classes.is_empty());
        // Title/Subtitle/ColumnLabel are all header rows; only the body row
        // isn't.
        assert!(layout.rows[0].is_header);
        assert!(layout.rows[1].is_header);
        assert!(layout.rows[2].is_header);
        assert!(!layout.rows[3].is_header);
    }

    #[test]
    fn reserved_label_keys_always_win_over_a_same_named_column() {
        // A column literally named "title" can't get a header override via
        // LABEL — the reserved key always wins, and the column just keeps
        // its own name, exactly as if no LABEL entry existed for it.
        let frame = df! {
            "title" => vec![1i32],
        }
        .unwrap();
        let mut table = Table::new();
        table
            .labels
            .labels
            .insert("title".to_string(), Some("The Table's Title".to_string()));

        let cells = resolve_cells(&frame, &table).unwrap();

        let heading = cells
            .iter()
            .find(|c| c.kind == TableCellKind::Title)
            .unwrap();
        assert_eq!(heading.content, "The Table's Title");
        let column_label = cells
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel)
            .unwrap();
        assert_eq!(column_label.content, "title");
    }
}
