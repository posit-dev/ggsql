//! Table-cell resolution: turns already-resolved columns, spans and a
//! `FORMAT`-applied `DataFrame` into the `TableLayout` `ResolvedTable`
//! holds, assembling parts with the `create_*` helpers from
//! `crate::table::layout`.

use std::collections::HashMap;

use super::format::resolve_column_properties;
use super::stub::{create_row_labels, create_stubhead};
use crate::array_util::{column_to_strings, value_to_string};
use crate::plot::facet::Facet;
use crate::plot::Labels;
use crate::table::{
    compose_header, create_caption, create_column_labels, create_heading, create_spanners,
    groups_are_woven, move_stub_columns, reorder_table_columns, TableCell, TableClass, TableColumn,
    TableLayout, TableRow, TableTitles,
};
use crate::{ColumnSection, DataFrame, Format, GgsqlError, Result, Spanner};

/// Resolve a table's columns: one `TableColumn` per `DataFrame` column
/// (labelled, with `formats`' resolved `SETTING` properties), reordered for
/// any `gather`-ing spanner and then for stub columns. A `place => 'top'`
/// facet consumes its columns as group labels; a `place => 'left'` facet
/// keeps them as group stub columns. `spans` must already be resolved
/// (`Table::resolve_spanners`).
pub(super) fn setup_columns(
    df: &DataFrame,
    spans: &[Spanner],
    labels: &Labels,
    formats: &HashMap<String, Format>,
    facet: Option<&Facet>,
) -> Vec<TableColumn> {
    let columns = create_table_columns(df, labels, formats, facet);
    let columns = reorder_table_columns(columns, spans);
    let mut columns = move_stub_columns(columns);
    let Some(facet) = facet else {
        return columns;
    };
    if groups_are_woven(facet) {
        // If we weave groups as rows, we omit the group columns
        let variables = facet.get_variables();
        columns.retain(|c| !variables.contains(&c.name));
    }
    columns
}

/// Resolve a table's body rows: one `TableRow` per `df` row, stamped with
/// its facet group key. A facet groups on the raw query values (before
/// `FORMAT`'s RENAMING rewrites them) and reorders `df` so groups are
/// contiguous. Group-label rows are not part of the result — `build_cells`
/// interleaves those once labels can be read from the formatted frame.
pub(super) fn setup_rows(
    df: DataFrame,
    facet: Option<&Facet>,
) -> Result<(DataFrame, Vec<TableRow>)> {
    let Some(facet) = facet else {
        let rows = vec![TableRow::body(); df.height()];
        return Ok((df, rows));
    };
    let df = super::facet::drop_null_rows(facet, &df).map_err(GgsqlError::ValidationError)?;
    let keys = super::facet::setup_groups(facet, &df).map_err(GgsqlError::ValidationError)?;
    let (df, keys) = super::facet::reorder_rows(&df, keys).map_err(GgsqlError::ValidationError)?;
    let mut rows = Vec::with_capacity(keys.len());
    for key in keys {
        let first_in_group = rows.last().is_none_or(|row: &TableRow| row.group != key);
        let mut row = TableRow {
            group: key,
            ..TableRow::body()
        };
        if first_in_group {
            row.classes.push(TableClass::RowGroupFirst);
        }
        rows.push(row);
    }
    Ok((df, rows))
}

/// Build the resolved `TableLayout` for a table from its already-resolved
/// `columns`, already-`FORMAT`-applied `df`, and `setup_rows` body rows —
/// spanner rows, column labels, and body, composed together and validated.
pub(super) fn build_cells(
    df: &DataFrame,
    columns: Vec<TableColumn>,
    rows: Vec<TableRow>,
    spans: &[Spanner],
    titles: &TableTitles,
    facet: Option<&Facet>,
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
    let heading = create_heading(
        titles.title.as_deref(),
        titles.subtitle.as_deref(),
        &columns,
    );
    let header = heading.rowbind(header);
    let mut body_cells = create_row_labels(df, &columns);
    body_cells.extend(create_body(df, &columns));
    let (body_rows, body_cells) = match facet {
        None => (rows, body_cells),
        Some(facet) => {
            if groups_are_woven(facet) {
                weave_group_labels(df, &columns, &rows, body_cells, facet)?
            } else {
                let body_cells = merge_group_stub_cells(body_cells, &columns, &rows);
                (rows, body_cells)
            }
        }
    };
    let table_body = TableLayout::new(columns.clone(), body_rows, body_cells);
    let layout = header.rowbind(table_body);
    let caption_layout = create_caption(titles.caption.as_deref(), &columns);
    let layout = layout.rowbind(caption_layout);

    layout.validate()?;

    Ok(layout)
}

/// Build one `TableColumn` per `DataFrame` column, in the `DataFrame`'s own
/// order — `reorder_table_columns` is what may reorder this list
/// afterward, not this function. `labels` is the one authority for an
/// explicit column label; `formats` (already reshaped to one `Format` per
/// column by `setup_formats`) is the one authority for its properties. A
/// `place => 'left'` facet marks its variables `GroupStub`.
fn create_table_columns(
    df: &DataFrame,
    labels: &Labels,
    formats: &HashMap<String, Format>,
    facet: Option<&Facet>,
) -> Vec<TableColumn> {
    let group_stub_variables = match facet {
        Some(facet) if !groups_are_woven(facet) => facet.get_variables(),
        _ => Vec::new(),
    };
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
        // Group stubs record nesting order
        let target = group_stub_variables
            .iter()
            .position(|v| *v == name)
            .map(ColumnSection::GroupStub)
            .unwrap_or_else(|| format.map(|f| f.target).unwrap_or_default());
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
        let values = column_to_strings(
            df.column(&column.name)
                .expect("TableColumn.name always names a column of df"),
        );

        for (row, value) in values.into_iter().enumerate() {
            cells.push(
                TableCell::new_body(row, index, value).with_properties(column.properties.clone()),
            );
        }
    }

    cells
}

/// Interleave group-label rows into the body. At each group boundary, emit
/// one `GroupHeading` row per facet-variable level that changed, labels read
/// from the group's first row (so an unchanged outer variable isn't
/// restated). Body cells keep their data-row numbering, shifted down by
/// the label rows before them.
fn weave_group_labels(
    df: &DataFrame,
    columns: &[TableColumn],
    rows: &[TableRow],
    body_cells: Vec<TableCell>,
    facet: &Facet,
) -> Result<(Vec<TableRow>, Vec<TableCell>)> {
    let facet_arrays: Vec<_> = facet
        .get_variables()
        .iter()
        .map(|name| {
            df.column(name).map_err(|_| {
                GgsqlError::ValidationError(format!(
                    "TABULATE FACET: no column '{name}' in the query result"
                ))
            })
        })
        .collect::<Result<_>>()?;

    let mut out_rows = Vec::new();
    let mut label_cells = Vec::new();
    // body_offset[data row] = how many label rows precede it in the grid.
    let mut body_offset = vec![0usize; df.height()];
    // None until the first group's key is read: it emits every level.
    let mut previous_key: Option<&[usize]> = None;
    let mut row = 0usize;

    for group_rows in rows.chunk_by(|a, b| a.group == b.group) {
        let key = group_rows[0].group.as_slice();
        let labels: Vec<String> = facet_arrays
            .iter()
            .map(|array| value_to_string(array, row))
            .collect();
        // When multiple labels exist, we have a nesting pattern.
        // We don't need to repeat every level label at every group,
        // just from the level(s) that changed onward.
        let first_changed = match previous_key {
            None => 0,
            Some(previous) => previous
                .iter()
                .zip(key)
                .position(|(old, new)| old != new)
                .unwrap_or(key.len()),
        };
        for label in &labels[first_changed..] {
            label_cells.push(TableCell::new_group_heading(
                out_rows.len(),
                columns.len(),
                label.clone(),
            ));
            out_rows.push(TableRow::group_heading(key.to_vec()));
        }
        previous_key = Some(key);

        for table_row in group_rows.iter() {
            body_offset[row] = out_rows.len() - row;
            out_rows.push(table_row.clone());
            row += 1;
        }
    }

    let mut body_cells = body_cells;
    for cell in &mut body_cells {
        cell.offset_rows(body_offset[cell.top]);
    }
    label_cells.extend(body_cells);

    Ok((out_rows, label_cells))
}

/// Merge each group stub column's row labels into rowspanned cells. A cell
/// at depth `d` spans rows while the `d + 1` outermost ranks of the rows'
/// group keys are equal, so a group never merges into a cell opened under
/// a different outer level even when its label repeats.
fn merge_group_stub_cells(
    cells: Vec<TableCell>,
    columns: &[TableColumn],
    rows: &[TableRow],
) -> Vec<TableCell> {
    if !columns.iter().any(|c| c.is_group_stub()) {
        return cells;
    }

    // Column index of a group stub column -> position of its open cell in
    // `out`, for the column whose cells are currently being walked.
    let mut open: HashMap<usize, usize> = HashMap::new();
    let mut out: Vec<TableCell> = Vec::with_capacity(cells.len());
    for cell in cells {
        if let ColumnSection::GroupStub(depth) = columns[cell.left].target {
            if let Some(&index) = open.get(&cell.left) {
                let current_key = &rows[cell.top].group[..=depth];
                let previous_key = &rows[cell.top - 1].group[..=depth];
                if current_key == previous_key {
                    out[index].bottom = cell.bottom;
                    continue;
                }
            }
            open.insert(cell.left, out.len());
        }
        out.push(cell);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::format::{apply_formats, setup_formats};
    use super::*;
    use crate::df;
    use crate::plot::{ParameterValue, Parameters};
    use crate::table::{extract_table_titles, TableCellKind};
    use crate::{ColumnSection, GgsqlError, Spanner, Table};

    /// Runs the same steps `resolve_table_with_reader` does, minus the
    /// `Reader`/SQL execution — lets a test build a `Table`'s resolved cells
    /// directly against a `df!()`-built `DataFrame`.
    fn resolve_layout(df: &DataFrame, table: &Table) -> Result<TableLayout> {
        let mut labels = table.labels.clone();
        let titles = extract_table_titles(&mut labels);
        let column_names = df.get_column_names();
        let spans = table
            .resolve_spanners(&labels, Some(&column_names))
            .map_err(GgsqlError::ValidationError)?;
        let formats = setup_formats(df, &table.formats, &spans)?;
        let columns = setup_columns(df, &spans, &labels, &formats, None);
        let df = apply_formats(df, &formats)?;
        let rows = vec![TableRow::body(); df.height()];
        build_cells(&df, columns, rows, &spans, &titles, None)
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

        let columns = create_table_columns(&frame, &labels, &HashMap::new(), None);

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
                value_template: "{}".to_string(),
            },
        )]);

        let columns = create_table_columns(&frame, &Labels::default(), &formats, None);
        assert_eq!(columns[0].label, "");

        let mut labels = Labels::default();
        labels
            .labels
            .insert("region".to_string(), Some("Region".to_string()));
        let columns = create_table_columns(&frame, &labels, &formats, None);
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

    fn left_facet(variables: &[&str]) -> Facet {
        let mut properties = Parameters::new();
        properties.insert(
            "place".to_string(),
            ParameterValue::String("left".to_string()),
        );
        Facet {
            layout: crate::plot::facet::FacetLayout::Wrap {
                variables: variables.iter().map(|s| s.to_string()).collect(),
            },
            properties,
            resolved: true,
        }
    }

    #[test]
    fn setup_columns_left_facet_leads_with_group_stubs_in_variable_order() {
        let frame = df! {
            "region" => vec!["north".to_string()],
            "quarter" => vec!["q1".to_string()],
            "sales" => vec![1i32],
        }
        .unwrap();
        let facet = left_facet(&["quarter", "region"]);

        let columns = setup_columns(
            &frame,
            &[],
            &Labels::default(),
            &HashMap::new(),
            Some(&facet),
        );

        let names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["quarter", "region", "sales"]);
        assert_eq!(columns[0].target, ColumnSection::GroupStub(0));
        assert_eq!(columns[1].target, ColumnSection::GroupStub(1));
        assert_eq!(columns[2].target, ColumnSection::Body);
    }

    #[test]
    fn create_body_numbers_rows_from_zero() {
        let frame = df! {
            "id" => vec![1i32, 2],
            "name" => vec!["a".to_string(), "b".to_string()],
        }
        .unwrap();
        let columns = create_table_columns(&frame, &Labels::default(), &HashMap::new(), None);

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
            value_template: "${:num %.2f}".to_string(),
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
