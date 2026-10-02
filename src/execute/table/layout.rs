//! Cell-layout construction: turns already-resolved columns, spans and a
//! `FORMAT`-applied `DataFrame` into the positioned `TableCell`/`TableRow`
//! grid `ResolvedTable` holds, via the `Section` building block that
//! `build_cells` and `spanner::create_spanners` produce and `rowbind`
//! combines.

use std::collections::{HashMap, HashSet};

use super::format::resolve_column_properties;
use super::spanner::{create_spanners, reorder_table_columns};
use super::stub::{create_row_labels, create_stubhead, move_stub_columns};
use super::{TableCell, TableCellKind, TableClass};
use crate::array_util::value_to_string;
use crate::plot::{Labels, Parameters};
use crate::{ColumnSection, DataFrame, Format, GgsqlError, Result, Spanner};

/// Pop the reserved `title`/`subtitle`/`caption` keys out of a `TABULATE`
/// `LABEL` clause's resolved map, leaving only genuine column-name
/// overrides behind. `.flatten()` treats "key absent" and "key present but
/// NULL" the same: neither produces a title/subtitle/caption.
pub(super) fn extract_heading_labels(
    labels: &mut Labels,
) -> (Option<String>, Option<String>, Option<String>) {
    let title = labels.labels.remove("title").flatten();
    let subtitle = labels.labels.remove("subtitle").flatten();
    let caption = labels.labels.remove("caption").flatten();
    (title, subtitle, caption)
}

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

/// Build the resolved cell layout for a table from its already-resolved
/// `columns` and already-`FORMAT`-applied `df` — spanner rows, column
/// labels, and body, composed together and checked for overlaps.
pub(super) fn build_cells(
    df: &DataFrame,
    columns: &[TableColumn],
    spans: &[Spanner],
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
) -> Result<(Vec<TableCell>, Vec<TableRow>)> {
    let ncol = columns.len();
    let spanners = create_spanners(columns, spans);
    // Every label empty means `LABEL col => NULL` (or `=> ''`) on every
    // column, since an unlabeled column keeps its (non-empty) name — omit
    // the row rather than render blank header cells.
    let column_labels = if columns.iter().all(|c| c.label.is_empty()) {
        Section::new(Vec::new(), Vec::new())
    } else {
        let mut label_cells = create_stubhead(columns);
        label_cells.extend(create_column_labels(columns));
        Section::new(
            vec![TableRow {
                classes: vec![TableClass::ColHeadingRow],
                ..TableRow::header()
            }],
            label_cells,
        )
    };
    let header = compose_header(spanners, column_labels);
    let heading = create_heading(title, subtitle, ncol);
    let header = rowbind(heading, header);
    let mut body_cells = create_row_labels(df, columns);
    body_cells.extend(create_body(df, columns));
    let table_body = Section::new(vec![TableRow::default(); df.height()], body_cells);
    let section = rowbind(header, table_body);
    let caption_section = create_caption(caption, ncol);
    let section = rowbind(section, caption_section);

    validate_overlaps(&section.cells)?;

    Ok((section.cells, section.rows))
}

/// One layout part's cells alongside one `TableRow` **per row index it
/// occupies** — not one per cell, since several cells can share a row (a
/// column-label row has one cell per column, but is still a single row).
/// Every `create_*` helper below builds both together, so `rows` is never
/// derived from `cells` and a row's classification (`is_header`, `Heading`)
/// is decided where its cells are built.
///
/// `pub(super)` so the sibling `spanner` module can build one; the fields
/// stay private so `new` is the only constructor.
pub(super) struct Section {
    rows: Vec<TableRow>,
    cells: Vec<TableCell>,
}

impl Section {
    /// Build a `Section`, `debug_assert`ing that `rows.len()` matches the
    /// row extent `cells` implies.
    pub(super) fn new(rows: Vec<TableRow>, cells: Vec<TableCell>) -> Section {
        debug_assert_eq!(
            rows.len(),
            count_cell_rows(&cells),
            "Section's row count doesn't match its own cells' row extent"
        );
        Section { rows, cells }
    }

    /// Consume `self` into just its cells.
    #[cfg(test)]
    pub(super) fn into_cells(self) -> Vec<TableCell> {
        self.cells
    }
}

/// Stack `bottom` below `top`, offsetting `bottom`'s cells down by `top`'s
/// row extent and concatenating both sides' `rows` — no offsetting needed
/// there, since a `TableRow`'s position *is* its index, unlike a
/// `TableCell`'s rectangle.
fn rowbind(top: Section, mut bottom: Section) -> Section {
    if top.cells.is_empty() {
        return bottom;
    }
    if bottom.cells.is_empty() {
        return top;
    }

    let row_offset = top.rows.len();
    for cell in &mut bottom.cells {
        cell.offset_rows(row_offset);
    }

    let mut rows = top.rows;
    rows.extend(bottom.rows);
    let mut cells = top.cells;
    cells.extend(bottom.cells);

    Section::new(rows, cells)
}

/// The row extent a set of cells implies — the highest `bottom` plus one,
/// or `0` for no cells.
pub(crate) fn count_cell_rows(cells: &[TableCell]) -> usize {
    cells
        .iter()
        .map(|cell| cell.bottom)
        .max()
        .map_or(0, |r| r + 1)
}

/// The column extent a set of cells implies — the highest `right` plus one,
/// or `0` for no cells.
pub(crate) fn count_cell_cols(cells: &[TableCell]) -> usize {
    cells
        .iter()
        .map(|cell| cell.right)
        .max()
        .map_or(0, |r| r + 1)
}

/// Build the title/subtitle heading rows (gt's `title_row`/`subtitle_row`),
/// each spanning the table's full width and numbered locally from
/// `top == 0`. Each row carries `TableClass::Heading` (gt's `gt_heading`).
fn create_heading(title: Option<&str>, subtitle: Option<&str>, ncol: usize) -> Section {
    let right = ncol.saturating_sub(1);
    let mut cells = Vec::new();
    if let Some(title) = title {
        cells.push(
            TableCell::new(TableCellKind::Title, 0, 0, 0, right, title.to_string())
                .with_classes(vec![TableClass::Title]),
        );
    }
    if let Some(subtitle) = subtitle {
        let row = cells.len();
        cells.push(
            TableCell::new(
                TableCellKind::Subtitle,
                row,
                row,
                0,
                right,
                subtitle.to_string(),
            )
            .with_classes(vec![TableClass::Subtitle]),
        );
    }
    let rows = vec![
        TableRow {
            classes: vec![TableClass::Heading],
            ..TableRow::header()
        };
        cells.len()
    ];
    Section::new(rows, cells)
}

/// Build the caption cell (gt's `tab_caption()`), spanning the table's full
/// width at local row 0; `rowbind` places it at the very end of the layout.
/// `HtmlWriter` pulls this cell out of the grid before rendering, since HTML
/// requires `<caption>` outside `<thead>`/`<tbody>`. Its row carries no
/// class — a caption never lands in a rendered `<tr>`.
fn create_caption(caption: Option<&str>, ncol: usize) -> Section {
    match caption {
        Some(caption) => Section::new(
            vec![TableRow::default()],
            vec![TableCell::new(
                TableCellKind::Caption,
                0,
                0,
                0,
                ncol.saturating_sub(1),
                caption.to_string(),
            )
            .with_classes(vec![TableClass::Caption])],
        ),
        None => Section::new(Vec::new(), Vec::new()),
    }
}

/// One column's identity within a table layout: its source name and
/// resolved display label/properties, as resolved by
/// `create_table_columns`.
#[derive(Debug, Clone)]
pub struct TableColumn {
    /// The column's name in the resolved `DataFrame` — used to look its
    /// values up in `create_body`, independent of display order.
    pub name: String,
    /// The resolved `ColumnLabel` cell content for this column.
    pub label: String,
    /// Resolved `SETTING` properties for this column's cells (e.g. `hjust`),
    /// carried onto every `ColumnLabel`/`Body` cell in this column. A
    /// writer wanting a whole-column property (e.g. `width`) reads it here
    /// instead of the same value repeated across the column's cells.
    pub properties: Parameters,
    /// Which section of the table this column belongs to (`FORMAT`'s
    /// `BODY`/`STUB` target, `Body` if no `FORMAT` clause named it).
    pub target: ColumnSection,
}

impl TableColumn {
    /// Whether this column is a `TABULATE FORMAT STUB` column.
    pub fn is_stub(&self) -> bool {
        self.target.is_stub()
    }
}

/// One row's resolved properties within a table layout. `classes` carries
/// row-scoped classes (e.g. `Heading`, `ColHeadingRow`) that belong on the
/// enclosing `<tr>` rather than on any cell.
#[derive(Debug, Clone, Default)]
pub struct TableRow {
    /// Resolved properties for this row's cells.
    pub properties: Parameters,
    /// Style classes recorded at build time (see `TableClass`) — row-scoped
    /// ones, e.g. `Heading`. Ordered, like `TableCell::classes`.
    pub classes: Vec<TableClass>,
    /// Whether this row belongs in a table's header rather than its body.
    /// `false` by default; set `true` for a spanner, column-label, title or
    /// subtitle row.
    pub is_header: bool,
}

impl TableRow {
    /// A header row — everything else at its default.
    pub fn header() -> Self {
        Self {
            is_header: true,
            ..Self::default()
        }
    }
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

/// Build one `ColumnLabel` cell per non-stub column, numbered locally from
/// `top == 0`. Skips every stub column; `create_stubhead` builds those.
fn create_column_labels(columns: &[TableColumn]) -> Vec<TableCell> {
    columns
        .iter()
        .enumerate()
        .filter(|(_, column)| !column.is_stub())
        .map(|(index, column)| {
            TableCell::new(
                TableCellKind::ColumnLabel,
                0,
                0,
                index,
                index,
                column.label.clone(),
            )
            .with_properties(column.properties.clone())
            .with_classes(vec![TableClass::ColHeading])
        })
        .collect()
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
                TableCell::new(
                    TableCellKind::Body,
                    row,
                    row,
                    index,
                    index,
                    value_to_string(array, row),
                )
                .with_properties(column.properties.clone())
                .with_classes(vec![TableClass::Row]),
            );
        }
    }

    cells
}

/// Stack spanner rows above column labels, letting a column with no spanner
/// covering it stretch its own label upward over the gap, then fill the
/// remaining gaps.
fn compose_header(spanners: Section, column_labels: Section) -> Section {
    let num_spanner_rows = spanners.rows.len();
    let header = rowbind(spanners, column_labels);
    let header = stretch_unspanned_column_labels(header, num_spanner_rows);
    fill_spanner_gaps(header, num_spanner_rows)
}

/// Grow a column's label cell upward into every consecutive spanner row
/// above it (starting from the row closest to the labels) that has no
/// spanner covering that column, stopping at the first one that does —
/// turning those rows into a `rowspan` on the label instead of separate
/// blank filler cells. gt only stretches into the single row immediately
/// above the labels, even when rows further up are also empty. Only touches
/// cells — row count/order is unaffected.
fn stretch_unspanned_column_labels(mut header: Section, num_spanner_rows: usize) -> Section {
    if num_spanner_rows == 0 {
        return header;
    }

    let ncol = count_cell_cols(&header.cells);
    let mut stretch_depth = vec![0usize; ncol];
    for (column, depth) in stretch_depth.iter_mut().enumerate() {
        for row in (0..num_spanner_rows).rev() {
            let covered = header.cells.iter().any(|c| {
                c.kind == TableCellKind::Spanner
                    && c.top == row
                    && c.left <= column
                    && column <= c.right
            });
            if covered {
                break;
            }
            *depth += 1;
        }
    }

    for label in header
        .cells
        .iter_mut()
        // A stub column can never be spanned (`Table::validate_span_stub_boundary`),
        // so its stretch always reaches the full spanner height.
        .filter(|c| matches!(c.kind, TableCellKind::ColumnLabel | TableCellKind::StubHead))
    {
        // Assumes every label cell is exactly one column wide (true of
        // everything create_column_labels/create_stubhead produce) — a wider
        // one would need its own stretch depth reconciled across its whole
        // span, not just `left`.
        label.top -= stretch_depth[label.left];
    }

    header
}

/// Fill every spanner-row position the stretch above couldn't reach with a
/// real, empty `Filler` cell.
fn fill_spanner_gaps(mut header: Section, num_spanner_rows: usize) -> Section {
    let ncol = count_cell_cols(&header.cells);

    // Coverage set, O(area) like `validate_overlaps`.
    let mut occupied: HashSet<(usize, usize)> = HashSet::new();
    for cell in &header.cells {
        for row in cell.top..=cell.bottom {
            for col in cell.left..=cell.right {
                occupied.insert((row, col));
            }
        }
    }

    for row in 0..num_spanner_rows {
        for col in 0..ncol {
            if occupied.contains(&(row, col)) {
                continue;
            }
            header.cells.push(
                TableCell::new(TableCellKind::Filler, row, row, col, col, String::new())
                    .with_classes(vec![TableClass::Spanner]),
            );
        }
    }

    header
}

/// Check that no two cells in a resolved layout claim the same grid position.
///
/// Walks every cell's full footprint into a map of occupied positions to the
/// `TableCellKind` that claimed each one, erroring — naming both kinds — on
/// the first double claim. `O(total cell area)`.
fn validate_overlaps(cells: &[TableCell]) -> Result<()> {
    let mut occupied: std::collections::HashMap<(usize, usize), TableCellKind> =
        std::collections::HashMap::new();

    for cell in cells {
        for row in cell.top..=cell.bottom {
            for col in cell.left..=cell.right {
                if let Some(existing_kind) = occupied.insert((row, col), cell.kind) {
                    return Err(GgsqlError::ValidationError(format!(
                        "Table layout has a {existing_kind} cell and a {} cell clashing at row {row}, column {col}",
                        cell.kind
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::format::{apply_formats, setup_formats};
    use super::*;
    use crate::df;
    use crate::plot::ParameterValue;
    use crate::{Spanner, Table};

    /// Runs the same steps `resolve_table_with_reader` does, minus the
    /// `Reader`/SQL execution — lets a test build a `Table`'s resolved cells
    /// directly against a `df!()`-built `DataFrame`.
    fn resolve_section(df: &DataFrame, table: &Table) -> Result<Section> {
        let mut labels = table.labels.clone();
        let (title, subtitle, caption) = extract_heading_labels(&mut labels);
        let column_names = df.get_column_names();
        let spans = table
            .resolve_spanners(&labels, Some(&column_names))
            .map_err(GgsqlError::ValidationError)?;
        let formats = setup_formats(df, &table.formats, &spans)?;
        let columns = setup_columns(df, &spans, &labels, &formats);
        let df = apply_formats(df, &formats)?;
        let (cells, rows) = build_cells(
            &df,
            &columns,
            &spans,
            title.as_deref(),
            subtitle.as_deref(),
            caption.as_deref(),
        )?;
        Ok(Section::new(rows, cells))
    }

    /// `resolve_section`, for the (common) tests that only care about cells.
    fn resolve_cells(df: &DataFrame, table: &Table) -> Result<Vec<TableCell>> {
        resolve_section(df, table).map(|section| section.cells)
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
                value_template: "{}".to_string(),
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
    fn create_column_labels_builds_one_cell_per_column_at_row_zero() {
        let columns = vec![column("id", "id"), column("name", "name")];

        let labels = create_column_labels(&columns);

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].kind, TableCellKind::ColumnLabel);
        assert_eq!(labels[0].top, 0);
        assert_eq!(labels[0].bottom, 0);
        assert_eq!(labels[0].left, 0);
        assert_eq!(labels[0].right, 0);
        assert_eq!(labels[0].content, "id");
        assert_eq!(labels[1].left, 1);
        assert_eq!(labels[1].right, 1);
        assert_eq!(labels[1].content, "name");
    }

    #[test]
    fn create_column_labels_skips_a_stub_column() {
        let columns = vec![stub_column("region"), column("sales", "sales")];

        let labels = create_column_labels(&columns);

        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].kind, TableCellKind::ColumnLabel);
        assert_eq!(labels[0].left, 1);
        assert_eq!(labels[0].content, "sales");
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

    fn cell(kind: TableCellKind, top: usize, bottom: usize, content: &str) -> TableCell {
        TableCell::new(kind, top, bottom, 0, 0, content.to_string())
    }

    /// Wrap a hand-built `Vec<TableCell>` into a `Section` for a test, with
    /// exactly as many default rows as `cells` implies (via
    /// `count_cell_rows`) — satisfies `Section::new`'s row-count invariant
    /// without every test having to spell out a `rows` vec of its own.
    fn section(cells: Vec<TableCell>) -> Section {
        let rows = vec![TableRow::default(); count_cell_rows(&cells)];
        Section::new(rows, cells)
    }

    #[test]
    fn rowbind_shifts_the_bottom_below_a_single_top_row() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];
        let body = vec![
            cell(TableCellKind::Body, 0, 0, "1"),
            cell(TableCellKind::Body, 1, 1, "2"),
        ];

        let cells = rowbind(section(column_labels), section(body)).cells;

        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0].kind, TableCellKind::ColumnLabel);
        assert_eq!(cells[0].top, 0);
        assert_eq!(cells[1].kind, TableCellKind::Body);
        assert_eq!(cells[1].top, 1);
        assert_eq!(cells[1].bottom, 1);
        assert_eq!(cells[2].top, 2);
        assert_eq!(cells[2].bottom, 2);
    }

    #[test]
    fn rowbind_offsets_by_the_top_rows_actual_extent_not_a_hardcoded_one() {
        // `rowbind` offsets by `top.rows.len()` rather than assuming exactly
        // one row — pin that down directly, independent of whichever caller
        // happens to produce a multi-row `top`.
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 1, "id")];
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = rowbind(section(column_labels), section(body)).cells;

        assert_eq!(cells[1].top, 2);
        assert_eq!(cells[1].bottom, 2);
    }

    #[test]
    fn rowbind_returns_bottom_unchanged_when_top_is_empty() {
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = rowbind(section(Vec::new()), section(body)).cells;

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
    }

    #[test]
    fn rowbind_returns_top_unchanged_when_bottom_is_empty() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];

        let cells = rowbind(section(column_labels), section(Vec::new())).cells;

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
    }

    #[test]
    fn compose_header_stretches_an_unspanned_columns_label_over_the_gap() {
        // "G" covers a, b (columns 0, 1) at the one spanner row; c has no
        // spanner at all. c is a StubHead: a stub column is never spanned,
        // so it stretches the same way an unspanned ColumnLabel does.
        let spanners = vec![cell_at(TableCellKind::Spanner, 0, 0, 0, 1)];
        let column_labels = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::StubHead, 0, 0, 2, 2),
        ];

        let header = compose_header(section(spanners), section(column_labels)).cells;

        let a = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 0)
            .unwrap();
        let c = header
            .iter()
            .find(|c| c.kind == TableCellKind::StubHead && c.left == 2)
            .unwrap();
        assert_eq!((a.top, a.bottom), (1, 1));
        assert_eq!((c.top, c.bottom), (0, 1));
    }

    #[test]
    fn compose_header_stops_stretching_at_the_first_row_that_covers_the_column() {
        // "Outer" (row 0, the far level) covers both a and b; "Inner" (row
        // 1, next to the labels) covers only a. b's row-1 gap is contiguous
        // with the labels, so it stretches by one row — but row 0 already
        // covers b, so the stretch must stop there, not skip past it.
        let spanners = vec![
            cell_at(TableCellKind::Spanner, 0, 0, 0, 1),
            cell_at(TableCellKind::Spanner, 1, 1, 0, 0),
        ];
        let column_labels = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
        ];

        let header = compose_header(section(spanners), section(column_labels)).cells;

        let a = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 0)
            .unwrap();
        let b = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 1)
            .unwrap();
        assert_eq!((a.top, a.bottom), (2, 2));
        assert_eq!((b.top, b.bottom), (1, 2));
    }

    #[test]
    fn compose_header_stretches_across_every_row_when_none_of_them_cover_the_column() {
        // "G" (row 1) covers only a; "H" (row 0) covers only b; c has no
        // spanner at either level, so its label absorbs both rows.
        let spanners = vec![
            cell_at(TableCellKind::Spanner, 0, 0, 1, 1),
            cell_at(TableCellKind::Spanner, 1, 1, 0, 0),
        ];
        let column_labels = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 2, 2),
        ];

        let header = compose_header(section(spanners), section(column_labels)).cells;

        let c = header
            .iter()
            .find(|cell| cell.kind == TableCellKind::ColumnLabel && cell.left == 2)
            .unwrap();
        assert_eq!((c.top, c.bottom), (0, 2));
    }

    #[test]
    fn compose_header_fills_a_spanner_gap_a_stretch_cant_reach() {
        // Stub column a, plus X(b,c) level 1, Y(c,d) level 2, Z(b,c) level 3:
        // d at Z's row (topmost) and b at Y's row (middle) are gaps neither
        // the stub head nor d's own label stretch reaches.
        let spanners = vec![
            cell_at(TableCellKind::Spanner, 0, 0, 1, 2), // Z, topmost
            cell_at(TableCellKind::Spanner, 1, 1, 2, 3), // Y, middle
            cell_at(TableCellKind::Spanner, 2, 2, 1, 2), // X, bottom-most
        ];
        let column_labels = vec![
            cell_at(TableCellKind::StubHead, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 2, 2),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 3, 3),
        ];

        let header = compose_header(section(spanners), section(column_labels)).cells;

        let fillers: Vec<_> = header
            .iter()
            .filter(|c| c.kind == TableCellKind::Filler)
            .collect();
        assert_eq!(fillers.len(), 2);
        let d3 = fillers.iter().find(|c| c.top == 0 && c.left == 3).unwrap();
        assert_eq!(d3.classes, vec![TableClass::Spanner]);
        let b2 = fillers.iter().find(|c| c.top == 1 && c.left == 1).unwrap();
        assert_eq!(b2.classes, vec![TableClass::Spanner]);

        // d's own label absorbs the third gap (row 2, X's row) via the
        // stretch — no filler there.
        let d_label = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 3)
            .unwrap();
        assert_eq!((d_label.top, d_label.bottom), (2, 3));
    }

    #[test]
    fn compose_header_does_nothing_when_there_are_no_spanners() {
        let column_labels = vec![cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0)];

        let header = compose_header(section(Vec::new()), section(column_labels)).cells;

        assert_eq!((header[0].top, header[0].bottom), (0, 0));
    }

    fn cell_at(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> TableCell {
        TableCell::new(kind, top, bottom, left, right, String::new())
    }

    #[test]
    fn validate_overlaps_accepts_a_disjoint_layout() {
        let cells = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::Body, 1, 1, 0, 0),
            cell_at(TableCellKind::Body, 1, 1, 1, 1),
        ];

        assert!(validate_overlaps(&cells).is_ok());
    }

    #[test]
    fn validate_overlaps_rejects_two_cells_at_the_same_position() {
        let cells = vec![
            cell_at(TableCellKind::Body, 0, 0, 0, 0),
            cell_at(TableCellKind::Body, 0, 0, 0, 0),
        ];

        let error = validate_overlaps(&cells).unwrap_err();
        assert!(matches!(error, GgsqlError::ValidationError(msg)
            if msg.contains("row 0, column 0") && msg.contains("body cell and a body cell")));
    }

    #[test]
    fn validate_overlaps_rejects_a_spanning_cell_overlapping_a_later_one() {
        // A cell spanning columns 0..=1 on row 0 overlapping a second cell
        // that only touches column 1 on the same row — the shape a spanner
        // bug or a bad spanner declaration would produce, not something
        // disjoint labels/body can create on their own.
        let cells = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 1),
            cell_at(TableCellKind::Spanner, 0, 0, 1, 1),
        ];

        let error = validate_overlaps(&cells).unwrap_err();
        assert!(matches!(error, GgsqlError::ValidationError(msg)
            if msg.contains("column label cell and a spanner cell")));
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
        // build_cells should surface this via validate_overlaps, not
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
    fn resolve_section_marks_heading_and_col_heading_rows() {
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

        let section = resolve_section(&frame, &table).unwrap();

        // Title (row 0), Subtitle (row 1), ColumnLabel (row 2), Body (row 3).
        assert_eq!(section.rows.len(), 4);
        assert_eq!(section.rows[0].classes, vec![TableClass::Heading]);
        assert_eq!(section.rows[1].classes, vec![TableClass::Heading]);
        assert_eq!(section.rows[2].classes, vec![TableClass::ColHeadingRow]);
        assert!(section.rows[3].classes.is_empty());
        // Title/Subtitle/ColumnLabel are all header rows; only the body row
        // isn't.
        assert!(section.rows[0].is_header);
        assert!(section.rows[1].is_header);
        assert!(section.rows[2].is_header);
        assert!(!section.rows[3].is_header);
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
