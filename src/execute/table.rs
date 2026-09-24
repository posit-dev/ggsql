//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling to do here — just the one query that
//! produces `body`, plus resolving that data into positioned `TableCell`s.
//! `SPAN`-specific resolution (column reordering, header-row assignment)
//! lives in the sibling `table_spanner` module, and `FORMAT`-specific
//! resolution (replacing a column's values with its resolved display text)
//! lives in `table_format`, the way Plot's own resolution logic is split
//! across `schema.rs`/`casting.rs`/`layer.rs`/`scale.rs`/`position.rs`/
//! `cte.rs` rather than left in one file. `Table::resolve_spanner_ids` is the
//! exception — it needs no `DataFrame`, so it lives on `Table` itself,
//! reachable from `validate()` too.

use std::collections::HashMap;

use super::table_format::{
    apply_formats, resolve_column_properties, setup_formats, standardise_hjust,
};
use super::table_spanner::{create_spanners, reorder_table_columns};
use crate::array_util::value_to_string;
use crate::parser::{self, SourceTree};
use crate::plot::{Labels, Parameters};
use crate::reader::{Reader, ResolvedTable};
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, Format, GgsqlError, Result, Spanner, Spec, Table};

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
    let mut labels = table.labels.clone();
    let (title, subtitle, caption) = extract_heading_labels(&mut labels);

    super::execute_setup_statements(&source_tree, reader)?;

    let sql = source_tree.extract_sql().ok_or_else(|| {
        GgsqlError::ValidationError(
            "TABULATE has no data source: add a FROM, or a SQL query before it".to_string(),
        )
    })?;

    let df = reader.execute_sql(&sql)?;

    // The shape both create_table_columns (SETTING) and apply_formats
    // (RENAMING) read from.
    let formats = setup_formats(&df, &table.formats)?;
    let (columns, spans) = setup_columns(&df, &table, &labels, &formats)?;
    let df = apply_formats(&df, &formats)?;
    let (cells, rows) = build_cells(
        &df,
        &columns,
        &spans,
        title.as_deref(),
        subtitle.as_deref(),
        caption.as_deref(),
    )?;

    Ok(ResolvedTable::new(
        cells,
        Some(columns),
        Some(rows),
        sql,
        warnings,
    ))
}

/// Pop the reserved `title`/`subtitle`/`caption` keys out of a `TABULATE`
/// `LABEL` clause's resolved map, leaving only genuine column-name
/// overrides behind — `create_table_columns` never needs to know these
/// three names are special, since by the time it runs they're already gone.
/// `.flatten()` treats "key absent" and "key present but NULL" the same:
/// neither produces a title/subtitle/caption, and there's no default value
/// for either to suppress the way an aesthetic's computed label has one.
fn extract_heading_labels(labels: &mut Labels) -> (Option<String>, Option<String>, Option<String>) {
    let title = labels.labels.remove("title").flatten();
    let subtitle = labels.labels.remove("subtitle").flatten();
    let caption = labels.labels.remove("caption").flatten();
    (title, subtitle, caption)
}

/// Resolve a table's columns: validated SPAN settings, spanner ids
/// expanded, one `TableColumn` per `DataFrame` column (labelled, with
/// `formats`' resolved `SETTING` properties), reordered for any
/// `gather`-ing spanner. Also returns `spans` alongside `columns` — already
/// resolved here, and still needed by `build_cells` for the spanner cells
/// themselves, so recomputing it there would just repeat this work.
fn setup_columns(
    df: &DataFrame,
    table: &Table,
    labels: &Labels,
    formats: &HashMap<String, Format>,
) -> Result<(Vec<TableColumn>, Vec<Spanner>)> {
    for (idx, spanner) in table.spans.iter().enumerate() {
        spanner
            .validate_settings()
            .map_err(|e| GgsqlError::ValidationError(format!("SPAN {}: {}", idx + 1, e)))?;
    }
    let spans = table
        .resolve_spanner_ids()
        .map_err(GgsqlError::ValidationError)?;

    let columns = create_table_columns(df, labels, formats);
    let columns = reorder_table_columns(columns, &spans)?;

    Ok((columns, spans))
}

/// Build the resolved cell layout for a table from its already-resolved
/// `columns` and already-`FORMAT`-applied `df` — spanner rows, column
/// labels, and body, composed together and checked for overlaps. Split out
/// from `resolve_table_with_reader` so the layout pipeline can be tested
/// directly against a `df!()`-built `DataFrame` and a `Table`, without
/// needing a `Reader`/real SQL execution.
fn build_cells(
    df: &DataFrame,
    columns: &[TableColumn],
    spans: &[Spanner],
    title: Option<&str>,
    subtitle: Option<&str>,
    caption: Option<&str>,
) -> Result<(Vec<TableCell>, Vec<TableRow>)> {
    let ncol = columns.len();
    let spanners = create_spanners(columns, spans)?;
    let column_labels = create_column_labels(columns);
    let header = compose_header(spanners, column_labels);
    let heading = create_heading(title, subtitle, ncol);
    let header = rowbind(heading, header);
    let table_body = create_body(df, columns);
    let section = rowbind(header, table_body);
    let caption_section = create_caption(caption, ncol);
    let section = rowbind(section, caption_section);

    validate_overlaps(&section.cells)?;

    Ok((section.cells, section.rows))
}

/// One layout part's cells alongside one `TableRow` **per row index it
/// occupies** — not one per cell, since several cells can share a row (a
/// column-label row has one cell per column, but is still a single row).
/// Every `create_*` helper below builds both together, sized correctly from
/// the row count it already knows (column labels: always 1; body:
/// `df.height()`; spanners: however many levels `assign_spanner_levels`
/// used; heading: 0/1/2) — `rows` is therefore never *derived* from `cells`
/// anywhere, which is what keeps `TableRow`'s classification (e.g.
/// `Heading`) a decision made exactly once, where a cell is built.
///
/// Kept private: this is build-pipeline plumbing between this file and
/// `table_spanner`, not part of `ResolvedTable`'s shape — `build_cells`
/// unpacks it into the plain `(Vec<TableCell>, Vec<TableRow>)` a
/// `ResolvedTable` actually holds. `pub(super)` on the type itself (rather
/// than fully private) because `table_spanner::create_spanners` needs to
/// name it and build one via `Section::new`; the fields stay private to
/// this module so `new` is the only way to construct one, from anywhere.
pub(super) struct Section {
    rows: Vec<TableRow>,
    cells: Vec<TableCell>,
}

impl Section {
    /// Build a `Section`, `debug_assert`ing that `rows.len()` matches the
    /// row extent `cells` implies — the invariant every `create_*` function
    /// is expected to already satisfy. Checked here, where a `Section`
    /// comes into existence, rather than only once something later
    /// consumes it — so it covers every `Section` ever built, not just ones
    /// that happen to reach `rowbind`.
    pub(super) fn new(rows: Vec<TableRow>, cells: Vec<TableCell>) -> Section {
        debug_assert_eq!(
            rows.len(),
            count_cell_rows(&cells),
            "Section's row count doesn't match its own cells' row extent"
        );
        Section { rows, cells }
    }

    /// Consume `self` into just its cells — `table_spanner`'s own tests are
    /// the one place outside this module that need to read a `Section`
    /// (rather than only ever construct one), to check `create_spanners`'
    /// cells directly.
    #[cfg(test)]
    pub(super) fn into_cells(self) -> Vec<TableCell> {
        self.cells
    }
}

/// Stack `bottom` below `top`, offsetting `bottom`'s cells down by whatever
/// row extent `top` actually occupies, and plain-concatenating both sides'
/// `rows` — no offsetting needed there, since a `TableRow`'s position *is*
/// its index, unlike a `TableCell`'s rectangle. Both inputs already went
/// through `Section::new`'s check when they were built, so there's nothing
/// left to (re-)validate here.
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

/// The row extent a set of cells implies — the highest `bottom` plus one, or
/// `0` for no cells. Shared by `Section::new`'s `debug_assert`,
/// `stretch_unspanned_column_labels`, `HtmlWriter::write_table`, and
/// `ResolvedTable::nrow`/`ncol`, alongside its column counterpart
/// `count_cell_cols`.
pub(crate) fn count_cell_rows(cells: &[TableCell]) -> usize {
    cells
        .iter()
        .map(|cell| cell.bottom)
        .max()
        .map_or(0, |r| r + 1)
}

/// The column extent a set of cells implies — the highest `right` plus one,
/// or `0` for no cells. `count_cell_rows`'s column counterpart.
pub(crate) fn count_cell_cols(cells: &[TableCell]) -> usize {
    cells
        .iter()
        .map(|cell| cell.right)
        .max()
        .map_or(0, |r| r + 1)
}

/// Build the title/subtitle heading rows (gt's `title_row`/`subtitle_row`),
/// each spanning the table's full width and numbered locally from
/// `top == 0` — `rowbind` is what stacks this above the rest of the header,
/// the same primitive every other part of the layout stacks with. Each row
/// carries `TableClass::Heading` (gt's `gt_heading`, applied to the whole
/// `<tr>` rather than to the cell inside it — see `TableClass`'s own doc
/// comment for why that split exists).
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
            ..TableRow::default()
        };
        cells.len()
    ];
    Section::new(rows, cells)
}

/// Build the caption cell (gt's `tab_caption()`), spanning the table's full
/// width at row 0 (locally numbered; `rowbind` places it after everything
/// else, at the very end of the layout). `HtmlWriter` pulls this cell out of
/// the grid before rendering — see its own module doc — since HTML requires
/// `<caption>` outside `<thead>`/`<tbody>`. Its row carries no class — a
/// caption never lands in a rendered `<tr>` at all.
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
/// resolved display label/properties. `create_table_columns` is the one
/// place `Labels` and `FORMAT`'s `SETTING` get consulted — `create_column_labels`
/// and `create_body` both work off this resolved data directly instead of
/// asking again, and both follow `columns`' order rather than `df`'s raw
/// column order, so a future spanner-driven reordering of this list carries
/// through to cell positions automatically.
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
}

/// One row's resolved properties within a table layout. `properties` has no
/// row-wide `TABULATE` clause to populate it yet, but `classes` does: a
/// `Title`/`Subtitle` row gets `TableClass::Heading` here (see
/// `create_heading`) — a class on the enclosing `<tr>` has nowhere else to
/// live, which is this type's whole reason to exist as a sibling to
/// `TableColumn` rather than being read off `TableCell` directly.
#[derive(Debug, Clone, Default)]
pub struct TableRow {
    /// Resolved properties for this row's cells.
    pub properties: Parameters,
    /// Style classes recorded at build time (see `TableClass`) — row-scoped
    /// ones, e.g. `Heading`. Ordered, like `TableCell::classes`.
    pub classes: Vec<TableClass>,
}

/// Build one `TableColumn` per `DataFrame` column, in the `DataFrame`'s own
/// order — `reorder_table_columns` is what may reorder this list
/// afterward, not this function. `labels` is the one authority for a
/// column's label; `formats` (already reshaped to one `Format` per column
/// by `setup_formats`) is the one authority for its properties.
fn create_table_columns(
    df: &DataFrame,
    labels: &Labels,
    formats: &HashMap<String, Format>,
) -> Vec<TableColumn> {
    df.get_column_names()
        .into_iter()
        .map(|name| {
            let label = match labels.labels.get(&name) {
                None => name.clone(),
                Some(None) => String::new(),
                Some(Some(label)) => label.clone(),
            };
            // Not stored on TableColumn: nothing needs it once `properties`
            // (which may default from it) is resolved.
            let dtype = df
                .column(&name)
                .expect("name comes from df's own columns")
                .data_type();
            let properties = resolve_column_properties(dtype, formats.get(&name));
            TableColumn {
                name,
                label,
                properties,
            }
        })
        .collect()
}

/// Build one `ColumnLabel` cell per column, numbered from `top == 0`, in
/// `columns`' order — always exactly one row.
///
/// Row numbering here is local to this function alone — `compose_header`/
/// `rowbind` are what decide where this sits relative to spanners and the
/// body, not this function.
fn create_column_labels(columns: &[TableColumn]) -> Section {
    // The only way every column's label ends up empty is `LABEL col => NULL`
    // (or `=> ''`) on every column, since an unlabeled column keeps its
    // (non-empty) name — so a wholly suppressed row omits the row entirely
    // rather than rendering a row of blank header cells.
    if columns.iter().all(|c| c.label.is_empty()) {
        return Section::new(Vec::new(), Vec::new());
    }

    let cells = columns
        .iter()
        .enumerate()
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
        .collect();

    Section::new(vec![TableRow::default()], cells)
}

/// Build one `Body` cell per `DataFrame` value, numbered from `top == 0`, in
/// `columns`' order rather than `df`'s raw column order — the same seam
/// `create_column_labels` uses, so the two stay in sync under a future
/// reordering. Looks each column up in `df` **by name**, not position, since
/// `columns` may already be reordered relative to `df` by the time this runs.
/// Always exactly `df.height()` rows.
fn create_body(df: &DataFrame, columns: &[TableColumn]) -> Section {
    let mut cells = Vec::new();

    for (index, column) in columns.iter().enumerate() {
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

    Section::new(vec![TableRow::default(); df.height()], cells)
}

/// Stack spanner rows above column labels — the header half of a table's
/// layout — then let a column with no spanner covering it stretch its own
/// label upward over the gap rather than leave it a separate blank cell.
/// Kept as its own named step (rather than folded into `rowbind`) since a
/// stubhead (another header part) is expected to join it.
fn compose_header(spanners: Section, column_labels: Section) -> Section {
    let num_spanner_rows = spanners.rows.len();
    let header = rowbind(spanners, column_labels);
    stretch_unspanned_column_labels(header, num_spanner_rows)
}

/// Grow a column's label cell upward into every consecutive spanner row
/// above it (starting from the row closest to the labels) that has no
/// spanner covering that column, stopping at the first one that does —
/// turning those rows into a `rowspan` on the label instead of separate
/// blank filler cells. Deliberately diverges from gt here: gt only ever
/// stretches into the single row immediately above the labels, even when
/// rows further up are also empty for that column (verified directly
/// against gt's own output). Only touches cells — the row count/order is
/// unaffected, since this never adds, removes, or reassigns a row, only
/// grows an existing `ColumnLabel` cell's `top` upward.
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
/// 0".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableCellKind {
    /// A column label (gt's `column_labels`).
    ColumnLabel,
    /// A data value (gt's `body`).
    Body,
    /// A spanner cell, grouping several columns under one label (`TABULATE
    /// SPAN`).
    Spanner,
    /// The table's title (`LABEL title => ...`, gt's `gt_title`).
    Title,
    /// The table's subtitle (`LABEL subtitle => ...`, gt's `gt_subtitle`).
    Subtitle,
    /// The table's caption (`LABEL caption => ...`, gt's `tab_caption()`).
    /// Full-width like `Title`/`Subtitle`, but rendered outside the normal
    /// row grid entirely — see `HtmlWriter`'s module doc.
    Caption,
}

impl TableCellKind {
    /// Whether a cell of this kind belongs in a table's header (`ColumnLabel`,
    /// `Spanner`, `Title`, `Subtitle`) rather than its body (`Body`) — the
    /// kind alone decides it, which is what lets this be asked off a bare
    /// `TableCellKind` (a synthesized filler has no real `TableCell` of its
    /// own) as well as off a full cell, via `TableCell::is_header` below.
    /// `Caption` stays out of this: a writer is expected to pull it out of
    /// the grid entirely rather than render it as either.
    pub fn is_header(self) -> bool {
        matches!(
            self,
            TableCellKind::ColumnLabel
                | TableCellKind::Spanner
                | TableCellKind::Title
                | TableCellKind::Subtitle
        )
    }
}

impl std::fmt::Display for TableCellKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TableCellKind::ColumnLabel => "column label",
            TableCellKind::Body => "body",
            TableCellKind::Spanner => "spanner",
            TableCellKind::Title => "title",
            TableCellKind::Subtitle => "subtitle",
            TableCellKind::Caption => "caption",
        };
        write!(f, "{text}")
    }
}

/// A writer-neutral style class, one shared vocabulary for whichever part of
/// a table layout carries a `classes` field.
///
/// Styling roles are recorded here rather than derived from `TableCellKind`:
/// `kind` informs layout (header vs body, spans), while classes inform style,
/// and positional variants like `SpannerOuter` are only known while the
/// layout is being built. A writer maps each variant to its own class
/// vocabulary — the HTML writer prefixes its `Display` with `ggsql_`.
///
/// Not every variant applies to every carrier: `Heading` is row-scoped
/// (`TableRow::classes`, the `<tr>` wrapping a `Title`/`Subtitle` cell, gt's
/// `gt_heading`); every other variant is cell-scoped (`TableCell::classes`).
/// Nothing in the type enforces that split — it's a per-variant convention,
/// since a writer maps every carrier through the same class-name/declaration
/// lookup regardless of where it came from. A future `TableColumn`-scoped
/// class (a `<col>`/`<colgroup>` concern) or a table-wide one
/// (`<table>`/`<thead>`/`<tbody>` itself) belongs in this same enum too.
///
/// Naming follows gt's classes (`gt_row`, `gt_col_heading`,
/// `gt_column_spanner_outer`, ...). The structural variants are recorded by
/// `build_cells()`; the alignment variants are a cell's resolved `hjust`
/// expressed as a class, appended by `TableCell::discretise_hjust` rather
/// than recorded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableClass {
    /// A body cell (gt's `gt_row`).
    Row,
    /// A column-label cell (gt's `gt_col_heading`).
    ColHeading,
    /// A spanner cell below the topmost spanner level.
    Spanner,
    /// A spanner cell in the topmost spanner level, supplanting `Spanner`
    /// (gt's `gt_column_spanner_outer`).
    SpannerOuter,
    /// Left/center/right-aligned cell content (gt's
    /// `gt_left`/`gt_center`/`gt_right`).
    AlignLeft,
    AlignCenter,
    AlignRight,
    /// The title cell (gt's `gt_title`).
    Title,
    /// The subtitle cell (gt's `gt_subtitle`).
    Subtitle,
    /// The caption cell (gt's `gt_caption`).
    Caption,
    /// Row-scoped: the `<tr>` wrapping a `Title`/`Subtitle` cell (gt's
    /// `gt_heading`).
    Heading,
}

impl std::fmt::Display for TableClass {
    /// The class-name suffix a writer hangs its own prefix on, e.g.
    /// `ggsql_row` for `Row` in the HTML writer.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TableClass::Row => "row",
            TableClass::ColHeading => "col_heading",
            TableClass::Spanner => "spanner",
            TableClass::SpannerOuter => "spanner_outer",
            TableClass::AlignLeft => "left",
            TableClass::AlignCenter => "center",
            TableClass::AlignRight => "right",
            TableClass::Title => "title",
            TableClass::Subtitle => "subtitle",
            TableClass::Caption => "caption",
            TableClass::Heading => "heading",
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
/// directly.
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
    /// Display properties for this cell (e.g. `hjust`), resolved from its
    /// column's `FORMAT` `SETTING`. `ColumnLabel` and `Body` cells inherit
    /// their column's properties; a `Spanner` cell covers several columns
    /// at once, so it has none of its own.
    pub properties: Parameters,
    /// Style classes recorded at build time (see `TableClass`). Ordered;
    /// a writer renders them in order.
    pub classes: Vec<TableClass>,
}

impl TableCell {
    /// Build a cell with no display properties or classes — the common case
    /// for a `Spanner` or filler cell, which covers several columns rather
    /// than resolving from one. A `ColumnLabel`/`Body` cell should follow
    /// this with `with_properties` and `with_classes` instead of leaving the
    /// defaults.
    pub(crate) fn new(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        content: String,
    ) -> Self {
        Self {
            kind,
            top,
            bottom,
            left,
            right,
            content,
            properties: Parameters::new(),
            classes: Vec::new(),
        }
    }

    /// Set this cell's display properties, e.g. to a column's resolved
    /// `FORMAT` `SETTING`.
    pub(crate) fn with_properties(mut self, properties: Parameters) -> Self {
        self.properties = properties;
        self
    }

    /// Record this cell's style classes, e.g. `TableClass::Row` for a
    /// body cell.
    pub(crate) fn with_classes(mut self, classes: Vec<TableClass>) -> Self {
        self.classes = classes;
        self
    }

    /// Fold this cell's resolved `hjust` into an alignment class appended to
    /// `classes`, removing `hjust` from `properties` so no writer renders it
    /// twice. Numbers and keyword spellings standardise via
    /// `standardise_hjust`; the number is then bucketed with the same
    /// `0.25`/`0.75` thresholds `VegaLiteWriter`'s `convert_hjust` uses for
    /// its own `align` conversion, so `hjust` means the same alignment in
    /// both writers.
    pub fn discretise_hjust(mut self) -> Self {
        let Some(hjust) = self.properties.remove("hjust") else {
            return self;
        };
        let Some(n) = standardise_hjust(&hjust) else {
            // Unrecognized value — don't silently drop it.
            self.properties.insert("hjust".to_string(), hjust);
            return self;
        };
        let class = if n <= 0.25 {
            TableClass::AlignLeft
        } else if n >= 0.75 {
            TableClass::AlignRight
        } else {
            TableClass::AlignCenter
        };
        self.classes.push(class);
        self
    }

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

    /// Whether this cell belongs in a table's header rather than its body —
    /// lets a writer pick `<th>` vs `<td>` (or an equivalent) off the cell
    /// itself, without matching on `TableCellKind` at every call site. See
    /// `TableCellKind::is_header`, which this just delegates to.
    pub fn is_header(&self) -> bool {
        self.kind.is_header()
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

    /// Runs the same steps `resolve_table_with_reader` does, minus the
    /// `Reader`/SQL execution — lets a test build a `Table`'s resolved cells
    /// directly against a `df!()`-built `DataFrame`.
    fn resolve_section(df: &DataFrame, table: &Table) -> Result<Section> {
        let mut labels = table.labels.clone();
        let (title, subtitle, caption) = extract_heading_labels(&mut labels);
        let formats = setup_formats(df, &table.formats)?;
        let (columns, spans) = setup_columns(df, table, &labels, &formats)?;
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

        let section = create_column_labels(&columns);
        assert_eq!(section.rows.len(), 1);
        let labels = section.cells;

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
        let columns = create_table_columns(&frame, &Labels::default(), &HashMap::new());

        let section = create_body(&frame, &columns);
        assert_eq!(section.rows.len(), 2);
        let body = section.cells;

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

        let body = create_body(&frame, &columns).cells;

        assert_eq!(body[0].content, "a"); // "name" column, placed first
        assert_eq!(body[1].content, "1"); // "id" column, placed second
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
        // spanner at all.
        let spanners = vec![cell_at(TableCellKind::Spanner, 0, 0, 0, 1)];
        let column_labels = vec![
            cell_at(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 1, 1),
            cell_at(TableCellKind::ColumnLabel, 0, 0, 2, 2),
        ];

        let header = compose_header(section(spanners), section(column_labels)).cells;

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
    fn table_cell_kind_display_names_every_variant() {
        assert_eq!(TableCellKind::ColumnLabel.to_string(), "column label");
        assert_eq!(TableCellKind::Body.to_string(), "body");
        assert_eq!(TableCellKind::Spanner.to_string(), "spanner");
        assert_eq!(TableCellKind::Title.to_string(), "title");
        assert_eq!(TableCellKind::Subtitle.to_string(), "subtitle");
        assert_eq!(TableCellKind::Caption.to_string(), "caption");
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
    fn discretise_hjust_folds_hjust_into_a_class() {
        let discretised = |n: f64| {
            let mut properties = Parameters::new();
            properties.insert("hjust".to_string(), ParameterValue::Number(n));
            TableCell::new(TableCellKind::Body, 0, 0, 0, 0, String::new())
                .with_properties(properties)
                .discretise_hjust()
        };
        assert_eq!(discretised(0.0).classes, [TableClass::AlignLeft]);
        assert_eq!(discretised(0.1).classes, [TableClass::AlignLeft]);
        assert_eq!(discretised(0.5).classes, [TableClass::AlignCenter]);
        assert_eq!(discretised(0.9).classes, [TableClass::AlignRight]);
        assert_eq!(discretised(1.0).classes, [TableClass::AlignRight]);
        // hjust is consumed, not copied.
        assert!(discretised(1.0).properties.is_empty());

        // The keyword spellings classify identically to their numbers.
        let discretised_str = |s: &str| {
            let mut properties = Parameters::new();
            properties.insert("hjust".to_string(), ParameterValue::String(s.to_string()));
            TableCell::new(TableCellKind::Body, 0, 0, 0, 0, String::new())
                .with_properties(properties)
                .discretise_hjust()
        };
        assert_eq!(discretised_str("left").classes, [TableClass::AlignLeft]);
        assert_eq!(discretised_str("right").classes, [TableClass::AlignRight]);
        assert_eq!(discretised_str("center").classes, [TableClass::AlignCenter]);
        assert_eq!(discretised_str("centre").classes, [TableClass::AlignCenter]);
        // No hjust → no class.
        assert!(
            TableCell::new(TableCellKind::Body, 0, 0, 0, 0, String::new())
                .discretise_hjust()
                .classes
                .is_empty()
        );
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
        // The only spanner level is the topmost one.
        assert!(cells
            .iter()
            .filter(|c| c.kind == TableCellKind::Spanner)
            .all(|c| c.classes == [TableClass::SpannerOuter]));
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
    fn resolve_section_marks_only_the_heading_rows() {
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
        assert!(section.rows[2].classes.is_empty());
        assert!(section.rows[3].classes.is_empty());
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
        // 3 data rows + 1 column-label row: nrow() is the whole grid, not
        // just the data rows.
        assert_eq!(resolved.nrow(), 4);
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
