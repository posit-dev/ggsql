//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling to do here — just the one query that
//! produces `body`, plus (as `Table` grows headings/spanners/footnotes)
//! resolving that data into positioned `TableCell`s. As those concerns grow
//! they're expected to split into sibling files here, the way Plot's own
//! resolution logic is split across `schema.rs`/`casting.rs`/`layer.rs`/
//! `scale.rs`/`position.rs`/`cte.rs` rather than left in one file.

use crate::array_util::value_to_string;
use crate::parser::{self, SourceTree};
use crate::plot::Labels;
use crate::reader::{Reader, ResolvedTable};
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, GgsqlError, Result, Spec};

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
    let columns = create_table_columns(&df, &table.labels);
    let column_labels = create_column_labels(&columns);
    let table_body = create_body(&df, &columns);
    let cells = compose_cells(column_labels, table_body);
    validate_overlaps(&cells)?;

    Ok(ResolvedTable::new(table, cells, sql, warnings))
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
struct TableColumn {
    /// The column's name in the resolved `DataFrame` — used to look its
    /// values up in `create_body`, independent of display order.
    name: String,
    /// The resolved `ColumnLabel` cell content for this column.
    label: String,
}

/// Build one `TableColumn` per `DataFrame` column, in the `DataFrame`'s own
/// order (nothing reorders it yet).
///
/// `labels` (from a `TABULATE LABEL` clause) is the one authority for a
/// column's label. Three outcomes: a name absent from `labels` keeps the
/// column name; an explicit `LABEL col => NULL` empties the label;
/// `LABEL col => 'text'` sets it to `text`.
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
/// Row numbering here is local to this function alone — `compose_cells`
/// is what decides where this sits relative to the body, not this function.
fn create_column_labels(columns: &[TableColumn]) -> Vec<TableCell> {
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

/// Compose a column-label row and a body into one layout: a pure function of
/// its two arguments, with no `DataFrame`/SQL knowledge of its own. Shifts
/// `body` down by however many rows `column_labels` occupies — today always
/// one, but this is what lets the two stay ignorant of each other's size. As
/// `Table` grows (headings/spanners/footnotes), more of these intermediate
/// composers are expected alongside this one (e.g. for a header or footer),
/// each combining a subset of parts the same way.
fn compose_cells(column_labels: Vec<TableCell>, mut body: Vec<TableCell>) -> Vec<TableCell> {
    let row_offset = column_labels
        .iter()
        .map(|cell| cell.bottom)
        .max()
        .map_or(0, |bottom| bottom + 1);

    for cell in &mut body {
        cell.offset_rows(row_offset);
    }

    let mut cells = column_labels;
    cells.extend(body);
    cells
}

/// Check that no two cells in a resolved layout claim the same grid position.
///
/// Walks every cell's full footprint (`top..=bottom` × `left..=right`, not
/// just its corners) into a set of occupied positions, erroring as soon as a
/// position is claimed twice. `O(total cell area)` rather than the O(n²) cost
/// of comparing every pair of cells — cheap for the common case (one 1x1
/// cell per data value, so area == cell count) and only grows with the
/// footprint spanning cells actually cover, not with `cells.len()` squared.
fn validate_overlaps(cells: &[TableCell]) -> Result<()> {
    let mut occupied = std::collections::HashSet::new();

    for cell in cells {
        for row in cell.top..=cell.bottom {
            for col in cell.left..=cell.right {
                if !occupied.insert((row, col)) {
                    return Err(GgsqlError::ValidationError(format!(
                        "Table layout has more than one cell at row {row}, column {col}"
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
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::df;

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
    fn compose_cells_shifts_the_body_below_a_single_label_row() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];
        let body = vec![
            cell(TableCellKind::Body, 0, 0, "1"),
            cell(TableCellKind::Body, 1, 1, "2"),
        ];

        let cells = compose_cells(column_labels, body);

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
    fn compose_cells_offsets_by_the_label_rows_actual_extent_not_a_hardcoded_one() {
        // Nothing produces a multi-row column-label section today, but
        // `compose_cells` computes the offset from `column_labels` itself
        // rather than assuming exactly one row — pin that down directly.
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 1, "id")];
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = compose_cells(column_labels, body);

        assert_eq!(cells[1].top, 2);
        assert_eq!(cells[1].bottom, 2);
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
        assert!(
            matches!(error, GgsqlError::ValidationError(msg) if msg.contains("row 0, column 0"))
        );
    }

    #[test]
    fn validate_overlaps_rejects_a_spanning_cell_overlapping_a_later_one() {
        // A cell spanning columns 0..=1 on row 0 overlapping a second cell
        // that only touches column 1 on the same row — the shape a spanner
        // bug or a bad spanner declaration would produce, not something
        // disjoint labels/body can create on their own.
        let cells = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 1),
            cell_at(TableCellKind::Body, 0, 0, 1, 1),
        ];

        assert!(validate_overlaps(&cells).is_err());
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
        // VISUALISE's FROM — regression for a bug where extract_sql matched
        // any statement's FROM in the whole query, not just the one being
        // resolved.
        let reader = reader_with_sales();
        let result = resolve_table_with_reader("TABULATE VISUALISE FROM sales DRAW point", &reader);
        assert!(result.is_err());
    }
}
