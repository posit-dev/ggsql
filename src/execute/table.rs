//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling to do here — just the one query that
//! produces `body`, plus resolving that data into positioned `TableCell`s.
//! `SPAN`-specific resolution (column reordering, header-row assignment)
//! lives in the sibling `table_spanner` module, the way Plot's own
//! resolution logic is split across `schema.rs`/`casting.rs`/`layer.rs`/
//! `scale.rs`/`position.rs`/`cte.rs` rather than left in one file.
//! `Table::resolve_spanner_ids` is the exception — it needs no `DataFrame`,
//! so it lives on `Table` itself, reachable from `validate()` too.

use super::table_spanner::{create_spanners, reorder_table_columns};
use crate::array_util::value_to_string;
use crate::parser::{self, SourceTree};
use crate::plot::Labels;
use crate::reader::{Reader, ResolvedTable};
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, GgsqlError, Result, Spec, Table};

/// Resolve a TABULATE query into a `ResolvedTable`.
///
/// This is the Table-side substitute for *two* Plot-side functions combined:
/// `execute::prepare_data_with_reader` (parses, resolves layers/scales/facets,
/// returns the intermediate `PreparedData`) and `reader::resolve_plot_with_reader`
/// (takes the first `Plot` from that, wraps it into `ResolvedPlot`). Table
/// collapses both into one function because there's no per-layer/scale/facet
/// resolution step for a `PreparedTable`-equivalent to do — `ResolvedTable`
/// already holds everything this function produces.
///
/// Takes the *first* `Table` spec found in the query (mirroring how Plot
/// execution takes the first `Plot` spec) — a query with several TABULATE
/// statements, or a mix of VISUALISE and TABULATE, isn't disambiguated any
/// further than that yet.
///
/// Setup statements (INSTALL, LOAD, SET, etc.) ahead of a TABULATE are
/// executed here too, via the same `execute_setup_statements` helper
/// `prepare_data_with_reader` uses — structured DML (CREATE, INSERT, UPDATE,
/// DELETE) ahead of a TABULATE isn't handled, since there's no CTE/side-effect
/// extraction step in this pipeline to mirror `prepare_data_with_reader`'s use
/// of `cte::extract_side_effects`.
pub fn resolve_table_with_reader(query: &str, reader: &dyn Reader) -> Result<ResolvedTable> {
    let validated = validate(query)?;
    let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

    let source_tree = SourceTree::new(query)?;
    source_tree.validate()?;

    let table = parser::build_ast(&source_tree)?
        .into_iter()
        .find_map(Spec::into_table)
        .ok_or_else(|| GgsqlError::ValidationError("No table specification found".to_string()))?;

    super::execute_setup_statements(&source_tree, reader)?;

    let sql = source_tree.extract_sql().ok_or_else(|| {
        GgsqlError::ValidationError(
            "TABULATE has no data source: add a FROM, or a SQL query before it".to_string(),
        )
    })?;

    let df = reader.execute_sql(&sql)?;
    let cells = build_cells(&df, &table)?;

    Ok(ResolvedTable::new(table, cells, sql, warnings))
}

/// Build the resolved cell layout for a table — columns (reordered for any
/// spanners), spanner rows, column labels, and body, composed together and
/// checked for overlaps. Split out from `resolve_table_with_reader` so the
/// whole layout pipeline can be tested directly against a `df!()`-built
/// `DataFrame` and a `Table`, without needing a `Reader`/real SQL execution.
fn build_cells(df: &DataFrame, table: &Table) -> Result<Vec<TableCell>> {
    for (idx, spanner) in table.spans.iter().enumerate() {
        spanner
            .validate_settings()
            .map_err(|e| GgsqlError::ValidationError(format!("SPAN {}: {}", idx + 1, e)))?;
    }
    let spans = table
        .resolve_spanner_ids()
        .map_err(GgsqlError::ValidationError)?;

    let columns = create_table_columns(df, &table.labels);
    let columns = reorder_table_columns(columns, &spans)?;
    let spanners = create_spanners(&columns, &spans)?;
    let column_labels = create_column_labels(&columns);
    let header = compose_header(spanners, column_labels);
    let table_body = create_body(df, &columns);
    let cells = rowbind_cells(header, table_body);
    validate_overlaps(&cells)?;

    Ok(cells)
}

/// One column's identity within a table layout: its source name and resolved
/// display label. `create_table_columns` is the one place `Labels` gets consulted —
/// `create_column_labels` and `create_body` both work off `name`/`label`
/// directly instead of asking `Labels` again, and both follow `columns`'
/// order rather than `df`'s raw column order, so a future spanner-driven
/// reordering of this list carries through to cell positions automatically.
///
/// `dtype: DataType` is expected to join this once column alignment is
/// tackled — left out for now since nothing would read it yet, and an unread
/// struct field is a dead-code warning, not just an early add.
pub(crate) struct TableColumn {
    /// The column's name in the resolved `DataFrame` — used to look its
    /// values up in `create_body`, independent of display order.
    pub(crate) name: String,
    /// The resolved `ColumnLabel` cell content for this column.
    pub(crate) label: String,
}

/// Build one `TableColumn` per `DataFrame` column, in the `DataFrame`'s own
/// order — `reorder_table_columns` is what may reorder this list
/// afterward, not this function. `labels` (from a `TABULATE LABEL` clause)
/// is the one authority for a column's label.
fn create_table_columns(df: &DataFrame, labels: &Labels) -> Vec<TableColumn> {
    df.get_column_names()
        .into_iter()
        .map(|name| {
            let label = match labels.labels.get(&name) {
                None => name.clone(),
                Some(None) => String::new(),
                Some(Some(label)) => label.clone(),
            };
            TableColumn { name, label }
        })
        .collect()
}

/// Build one `ColumnLabel` cell per column, numbered from `top == 0`, in
/// `columns`' order.
///
/// Row numbering here is local to this function alone — `compose_header`/
/// `rowbind_cells` are what decide where this sits relative to spanners and
/// the body, not this function.
fn create_column_labels(columns: &[TableColumn]) -> Vec<TableCell> {
    // The only way every column's label ends up empty is `LABEL col => NULL`
    // (or `=> ''`) on every column, since an unlabeled column keeps its
    // (non-empty) name — so a wholly suppressed row omits the row entirely
    // rather than rendering a row of blank header cells.
    if columns.iter().all(|c| c.label.is_empty()) {
        return Vec::new();
    }

    columns
        .iter()
        .enumerate()
        .map(|(index, column)| TableCell {
            kind: TableCellKind::ColumnLabel,
            top: 0,
            bottom: 0,
            left: index,
            right: index,
            content: column.label.clone(),
        })
        .collect()
}

/// Build one `Body` cell per `DataFrame` value, numbered from `top == 0`, in
/// `columns`' order rather than `df`'s raw column order — the same seam
/// `create_column_labels` uses, so the two stay in sync under a future
/// reordering. Looks each column up in `df` **by name**, not position, since
/// `columns` may already be reordered relative to `df` by the time this runs.
fn create_body(df: &DataFrame, columns: &[TableColumn]) -> Vec<TableCell> {
    let mut cells = Vec::new();

    for (index, column) in columns.iter().enumerate() {
        // Looked up once per column, outside the row loop: `DataFrame::column`
        // is an `O(ncol)` scan over the schema, so doing this per row instead
        // would cost `O(nrow * ncol)` lookups rather than `O(ncol)`.
        let array = df
            .column(&column.name)
            .expect("TableColumn.name always names a column of df");

        for row in 0..df.height() {
            cells.push(TableCell {
                kind: TableCellKind::Body,
                top: row,
                bottom: row,
                left: index,
                right: index,
                content: value_to_string(array, row),
            });
        }
    }

    cells
}

/// Stack `bottom` below `top`, offsetting `bottom` down by whatever row
/// extent `top` actually occupies — a pure function of its two arguments,
/// with no `DataFrame`/SQL knowledge of its own, and no assumption about
/// either side's row count. R's `rbind()` for already-positioned cells:
/// every part of the table (spanners, column labels, body, ...) gets
/// stacked together with the same primitive rather than each combination
/// hardcoding its own offset arithmetic.
fn rowbind_cells(top: Vec<TableCell>, mut bottom: Vec<TableCell>) -> Vec<TableCell> {
    if top.is_empty() {
        return bottom;
    }
    if bottom.is_empty() {
        return top;
    }

    let row_offset = top
        .iter()
        .map(|cell| cell.bottom)
        .max()
        .map_or(0, |bottom| bottom + 1);

    for cell in &mut bottom {
        cell.offset_rows(row_offset);
    }

    let mut cells = top;
    cells.extend(bottom);
    cells
}

/// Stack spanner rows above column labels — the header half of a table's
/// layout — then let a column with no spanner covering it stretch its own
/// label upward over the gap rather than leave it a separate blank cell.
/// Kept as its own named step (rather than folded into `rowbind_cells`)
/// since a stubhead (another header part) is expected to join it.
fn compose_header(spanners: Vec<TableCell>, column_labels: Vec<TableCell>) -> Vec<TableCell> {
    let num_spanner_rows = spanners.iter().map(|c| c.bottom).max().map_or(0, |r| r + 1);
    let header = rowbind_cells(spanners, column_labels);
    stretch_unspanned_column_labels(header, num_spanner_rows)
}

/// Grow a column's label cell upward into every consecutive spanner row
/// above it (starting from the row closest to the labels) that has no
/// spanner covering that column, stopping at the first one that does —
/// turning those rows into a `rowspan` on the label instead of separate
/// blank filler cells. Deliberately diverges from gt here: gt only ever
/// stretches into the single row immediately above the labels, even when
/// rows further up are also empty for that column (verified directly
/// against gt's own output).
fn stretch_unspanned_column_labels(
    mut header: Vec<TableCell>,
    num_spanner_rows: usize,
) -> Vec<TableCell> {
    if num_spanner_rows == 0 {
        return header;
    }

    let ncol = header.iter().map(|c| c.right).max().map_or(0, |r| r + 1);
    let mut stretch_depth = vec![0usize; ncol];
    for (column, depth) in stretch_depth.iter_mut().enumerate() {
        for row in (0..num_spanner_rows).rev() {
            let covered = header.iter().any(|c| {
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
        .iter_mut()
        .filter(|c| c.kind == TableCellKind::ColumnLabel)
    {
        // Assumes every ColumnLabel cell is exactly one column wide (true of
        // everything create_column_labels produces) — a wider one would need
        // its own stretch depth reconciled across its whole span, not just
        // `left`.
        label.top -= stretch_depth[label.left];
    }

    header
}

/// Check that no two cells in a resolved layout claim the same grid position.
///
/// Walks every cell's full footprint (`top..=bottom` × `left..=right`, not
/// just its corners) into a map of occupied positions to the `TableCellKind`
/// that claimed each one, erroring — naming both kinds involved — as soon as
/// a position is claimed twice. `O(total cell area)` rather than the O(n²)
/// cost of comparing every pair of cells — cheap for the common case (one
/// 1x1 cell per data value, so area == cell count) and only grows with the
/// footprint spanning cells actually cover, not with `cells.len()` squared.
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

// =============================================================================
// Public API: TableCell
// =============================================================================

/// What role a `TableCell` plays in the table's layout.
///
/// Naming follows R's gt package (`column_labels`, `body`, ...), since ggsql's
/// table grammar is expected to keep drawing on its part vocabulary as more
/// of it (spanners, stub, footnotes, source notes) gets built out here.
///
/// Lets a writer tell cells apart (e.g. `<th>` vs `<td>`) without relying on
/// position — a column label is a `ColumnLabel` cell, not "whatever's in row
/// 0". A caption is expected to become a `TableCell` too once `Table` can
/// resolve one (still just text with a position, spanning the full width) —
/// not added yet, since nothing produces one today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableCellKind {
    /// A column label (gt's `column_labels`).
    ColumnLabel,
    /// A data value (gt's `body`).
    Body,
    /// A spanner cell, grouping several columns under one label (`TABULATE
    /// SPAN`).
    Spanner,
}

impl std::fmt::Display for TableCellKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TableCellKind::ColumnLabel => "column label",
            TableCellKind::Body => "body",
            TableCellKind::Spanner => "spanner",
        };
        write!(f, "{text}")
    }
}

/// A single positioned cell within a resolved table layout.
///
/// Parallel to `PreparedData` on the Plot side (an intermediate resolution
/// type, not the final `ResolvedTable` envelope) — but there is no Plot-side
/// equivalent to the shape itself, since Plot resolves at `DataFrame`
/// granularity, not per-cell.
///
/// Position is an inclusive grid rectangle: `top`/`bottom` are row indices,
/// `left`/`right` are column indices, 0-based, inclusive on both ends. A
/// non-spanning cell has `top == bottom` and `left == right`. Colspan/rowspan
/// and adjacency helpers beyond `offset_rows`/`offset_cols` are expected to
/// live elsewhere and account for the inclusive convention themselves,
/// rather than each caller doing `+ 1` arithmetic against these fields
/// directly. Style/formatting fields are deliberately not included yet — add
/// them once a feature needs them.
#[derive(Debug, Clone)]
pub struct TableCell {
    /// What role this cell plays (column label, body, ...).
    pub kind: TableCellKind,
    /// Top row index (inclusive).
    pub top: usize,
    /// Bottom row index (inclusive).
    pub bottom: usize,
    /// Left column index (inclusive).
    pub left: usize,
    /// Right column index (inclusive).
    pub right: usize,
    /// The cell's text content.
    pub content: String,
}

impl TableCell {
    /// Shift this cell down by `rows`, moving `top` and `bottom` together so
    /// a spanning cell keeps its height.
    pub fn offset_rows(&mut self, rows: usize) {
        self.top += rows;
        self.bottom += rows;
    }

    /// Shift this cell right by `cols`, moving `left` and `right` together
    /// so a spanning cell keeps its width.
    pub fn offset_cols(&mut self, cols: usize) {
        self.left += cols;
        self.right += cols;
    }

    /// Whether this cell belongs in a table's header (`ColumnLabel`,
    /// `Spanner`) rather than its body (`Body`) — lets a writer pick `<th>`
    /// vs `<td>` (or an equivalent) off the cell itself, without matching on
    /// `TableCellKind` at every call site.
    pub fn is_header(&self) -> bool {
        matches!(
            self.kind,
            TableCellKind::ColumnLabel | TableCellKind::Spanner
        )
    }

    /// This cell's width in columns — `right - left + 1`, since both bounds
    /// are inclusive.
    pub fn width(&self) -> usize {
        self.right - self.left + 1
    }

    /// This cell's height in rows — `bottom - top + 1`, since both bounds
    /// are inclusive. No caller yet; kept alongside `width` for symmetry.
    pub fn height(&self) -> usize {
        self.bottom - self.top + 1
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::df;
    use crate::plot::{ParameterValue, Parameters};
    use crate::Spanner;

    fn column(name: &str, label: &str) -> TableColumn {
        TableColumn {
            name: name.to_string(),
            label: label.to_string(),
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

        let columns = create_table_columns(&frame, &labels);

        assert_eq!(columns[0].name, "id");
        assert_eq!(columns[0].label, "ID"); // overridden
        assert_eq!(columns[1].label, ""); // explicitly suppressed
        assert_eq!(columns[2].label, "extra"); // absent: kept as-is
    }

    fn spanner_with(columns: &[&str], label: Option<&str>, settings: Parameters) -> Spanner {
        Spanner {
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
    fn create_body_numbers_rows_from_zero() {
        let frame = df! {
            "id" => vec![1i32, 2],
            "name" => vec!["a".to_string(), "b".to_string()],
        }
        .unwrap();
        let columns = create_table_columns(&frame, &Labels::default());

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

    fn cell(kind: TableCellKind, top: usize, bottom: usize, content: &str) -> TableCell {
        TableCell {
            kind,
            top,
            bottom,
            left: 0,
            right: 0,
            content: content.to_string(),
        }
    }

    #[test]
    fn rowbind_cells_shifts_the_bottom_below_a_single_top_row() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];
        let body = vec![
            cell(TableCellKind::Body, 0, 0, "1"),
            cell(TableCellKind::Body, 1, 1, "2"),
        ];

        let cells = rowbind_cells(column_labels, body);

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
    fn rowbind_cells_offsets_by_the_top_rows_actual_extent_not_a_hardcoded_one() {
        // `rowbind_cells` computes the offset from `top` itself rather than
        // assuming exactly one row — pin that down directly, independent of
        // whichever caller happens to produce a multi-row `top`.
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 1, "id")];
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = rowbind_cells(column_labels, body);

        assert_eq!(cells[1].top, 2);
        assert_eq!(cells[1].bottom, 2);
    }

    #[test]
    fn rowbind_cells_returns_bottom_unchanged_when_top_is_empty() {
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = rowbind_cells(Vec::new(), body);

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
    }

    #[test]
    fn rowbind_cells_returns_top_unchanged_when_bottom_is_empty() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];

        let cells = rowbind_cells(column_labels, Vec::new());

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
    }

    #[test]
    fn compose_header_stretches_an_unspanned_columns_label_over_the_gap() {
        // "G" covers a, b (columns 0, 1) at the one spanner row; c has no
        // spanner at all.
        let spanners = vec![cell_at(TableCellKind::Spanner, 0, 0, 0, 1)];
        let column_labels = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 2, 2),
        ];

        let header = compose_header(spanners, column_labels);

        let a = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 0)
            .unwrap();
        let c = header
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 2)
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

        let header = compose_header(spanners, column_labels);

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

        let header = compose_header(spanners, column_labels);

        let c = header
            .iter()
            .find(|cell| cell.kind == TableCellKind::ColumnLabel && cell.left == 2)
            .unwrap();
        assert_eq!((c.top, c.bottom), (0, 2));
    }

    #[test]
    fn compose_header_does_nothing_when_there_are_no_spanners() {
        let column_labels = vec![cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0)];

        let header = compose_header(Vec::new(), column_labels);

        assert_eq!((header[0].top, header[0].bottom), (0, 0));
    }

    fn cell_at(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> TableCell {
        TableCell {
            kind,
            top,
            bottom,
            left,
            right,
            content: String::new(),
        }
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
    fn table_cell_kind_display_names_all_three_variants() {
        assert_eq!(TableCellKind::ColumnLabel.to_string(), "column label");
        assert_eq!(TableCellKind::Body.to_string(), "body");
        assert_eq!(TableCellKind::Spanner.to_string(), "spanner");
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

        let cells = build_cells(&frame, &table).unwrap();

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
    fn build_cells_reorders_columns_for_a_spanner_before_building_labels_and_body() {
        let frame = df! {
            "a" => vec![1i32],
            "x" => vec![9i32],
            "b" => vec![2i32],
        }
        .unwrap();
        let mut table = Table::new();
        table.spans = vec![labeled_spanner(&["a", "b"], "G")];

        let cells = build_cells(&frame, &table).unwrap();

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

        assert!(build_cells(&frame, &table).is_err());
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

        let cells = build_cells(&frame, &table).unwrap();

        assert!(!cells.iter().any(|c| c.kind == TableCellKind::ColumnLabel));
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Body)
            .all(|c| c.top == 0));
    }
}

#[cfg(test)]
#[cfg(feature = "duckdb")]
mod tests {
    use super::*;
    use crate::reader::DuckDBReader;

    fn reader_with_sales() -> DuckDBReader {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, name)")
            .unwrap();
        reader
    }

    #[test]
    fn test_tabulate_from() {
        let reader = reader_with_sales();
        let resolved = resolve_table_with_reader("TABULATE FROM sales", &reader).unwrap();

        assert_eq!(resolved.sql(), "SELECT * FROM sales");
        assert_eq!(resolved.nrow(), 3);
        assert_eq!(resolved.ncol(), 2);
    }

    #[test]
    fn test_bare_tabulate_uses_preceding_select() {
        let reader = reader_with_sales();

        let from_only = resolve_table_with_reader("TABULATE FROM sales", &reader).unwrap();
        let select_then_tabulate =
            resolve_table_with_reader("SELECT * FROM sales TABULATE", &reader).unwrap();

        assert_eq!(from_only.sql(), select_then_tabulate.sql());
        assert_eq!(from_only.nrow(), select_then_tabulate.nrow());
    }

    #[test]
    fn test_tabulate_with_no_source_errors() {
        let reader = reader_with_sales();
        let result = resolve_table_with_reader("TABULATE", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_does_not_borrow_a_later_visualise_from() {
        // A source-less TABULATE followed by an unrelated VISUALISE FROM must
        // still error "no data source", not silently resolve against the
        // VISUALISE's FROM.
        let reader = reader_with_sales();
        let result = resolve_table_with_reader("TABULATE VISUALISE FROM sales DRAW point", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_label_applies_under_the_tab_clause_wrapper() {
        // label_clause is nested under tab_clause in the grammar (so LABEL
        // and SPAN can appear in any order) — build_tabulate_statement must
        // unwrap it, not match "label_clause" as a direct child.
        let reader = reader_with_sales();
        let resolved =
            resolve_table_with_reader("TABULATE FROM sales LABEL id => 'ID'", &reader).unwrap();

        let label_cell = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 0)
            .unwrap();
        assert_eq!(label_cell.content, "ID");
    }
}
