//! Resolved table layout vocabulary: the positioned `TableCell`, its
//! `TableCellKind`, and the writer-neutral `TableClass` style classes —
//! plus the `Section` assembly that composes cells, column labels,
//! headings and captions into a grid (`rowbind`, `compose_header`, ...).

use std::collections::HashSet;

use super::ColumnSection;
use crate::labels::Labels;
use crate::params::Parameters;
use crate::plot::ParameterValue;
use crate::{GgsqlError, Result};

/// What role a `TableCell` plays in the table's layout.
///
/// Naming follows R's gt package (`column_labels`, `body`, `stub`, ...).
/// Lets a writer tell cells apart (e.g. `<th>` vs `<td>`) without relying on
/// position — a column label is a `ColumnLabel` cell, not "whatever's in row
/// 0".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableCellKind {
    /// A column label (gt's `column_labels`).
    ColumnLabel,
    /// The stub's own header cell (gt's `tab_stubhead()`) — sits on the
    /// column-label row, over a `TABULATE FORMAT STUB` column.
    StubHead,
    /// A data value (gt's `body`).
    Body,
    /// A row-label value in the stub (`TABULATE FORMAT STUB`, gt's `stub`).
    StubRowLabel,
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
    /// An empty cell synthesized in `execute` to fill a grid position no
    /// real cell reaches. Distinct from whatever kind it visually stands in
    /// for, so kind-based filtering can't mistake it for real structure.
    Filler,
}

impl TableCellKind {
    /// Whether a cell of this kind is header-shaped rather than a plain data
    /// cell — decided by kind alone. `Caption` is excluded; a writer pulls it
    /// out of the grid instead of rendering it as either.
    pub fn is_header(self) -> bool {
        matches!(
            self,
            TableCellKind::ColumnLabel
                | TableCellKind::StubHead
                | TableCellKind::Spanner
                | TableCellKind::Title
                | TableCellKind::Subtitle
                | TableCellKind::Filler // every current case is header-shaped
                | TableCellKind::StubRowLabel
        )
    }
}

impl std::fmt::Display for TableCellKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TableCellKind::ColumnLabel => "column label",
            TableCellKind::StubHead => "stub head",
            TableCellKind::Body => "body",
            TableCellKind::StubRowLabel => "stub row label",
            TableCellKind::Spanner => "spanner",
            TableCellKind::Title => "title",
            TableCellKind::Subtitle => "subtitle",
            TableCellKind::Caption => "caption",
            TableCellKind::Filler => "filler",
        };
        write!(f, "{text}")
    }
}

/// A writer-neutral style class, one shared vocabulary for whichever part of
/// a table layout carries a `classes` field.
///
/// Styling roles are recorded here rather than derived from `TableCellKind`:
/// `kind` informs layout (header vs body, spans), classes inform style. A
/// writer maps each variant to its own class vocabulary — the HTML writer
/// prefixes its `Display` with `ggsql_`.
///
/// Variants are grouped by scope below (table/row/cell). Nothing in the
/// type enforces the grouping — it's a per-variant convention.
///
/// Naming follows gt's classes (`gt_row`, `gt_col_heading`,
/// `gt_column_spanner_outer`, ...). The structural variants are recorded by
/// `build_cells()`; the alignment variants are a cell's resolved `hjust`
/// expressed as a class, appended by `TableCell::discretise_hjust`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableClass {
    // ------------------------- Table-scoped -------------------------
    // Applied by a writer to its own top-level elements (`<table>`,
    // `<tbody>`) — there's no resolved layout type to record them on.
    /// The whole table (gt's `gt_table`).
    Table,
    /// The `<tbody>` wrapping every body row (gt's `gt_table_body`). Not to
    /// be confused with the per-cell `Row`.
    TableBody,

    // -------------------------- Row-scoped --------------------------
    // Recorded on `TableRow::classes`, rendered on the `<tr>`.
    /// The `<tr>` wrapping a `Title`/`Subtitle` cell (gt's `gt_heading`).
    Heading,
    /// The `<tr>` wrapping the row of column-label cells (gt's
    /// `gt_col_headings` — note the plural, distinguishing it from the
    /// singular per-cell `gt_col_heading`).
    ColHeadingRow,
    /// The `<tr>` wrapping a spanner row (gt's `gt_spanner_row`).
    SpannerRow,

    // -------------------------- Cell-scoped -------------------------
    // Recorded on `TableCell::classes`, rendered on the `<th>`/`<td>`.
    /// A body cell (gt's `gt_row`).
    Row,
    /// A row-label cell in the stub (`TableCellKind::StubRowLabel`).
    Stub,
    /// The stub's own header cell (`TableCellKind::StubHead`). No gt
    /// equivalent.
    StubHead,
    /// A column-label cell (gt's `gt_col_heading`).
    ColHeading,
    /// Every spanner cell, at every level (gt's `gt_column_spanner_outer`).
    Spanner,
    /// A spanner cell's label text (gt's `gt_column_spanner`). Special
    /// meaning for the HTML writer: it goes on an element inside the cell
    /// rather than on the cell itself.
    SpannerLabel,
    /// The title cell (gt's `gt_title`).
    Title,
    /// The subtitle cell (gt's `gt_subtitle`).
    Subtitle,
    /// The caption cell (gt's `gt_caption`).
    Caption,
    /// Left/center/right-aligned cell content (gt's
    /// `gt_left`/`gt_center`/`gt_right`).
    AlignLeft,
    AlignCenter,
    AlignRight,
}

impl std::fmt::Display for TableClass {
    /// The class-name suffix a writer hangs its own prefix on, e.g.
    /// `ggsql_row` for `Row` in the HTML writer.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TableClass::Table => "table",
            TableClass::TableBody => "table_body",
            TableClass::Row => "row",
            TableClass::Stub => "stub",
            TableClass::ColHeading => "col_heading",
            TableClass::ColHeadingRow => "col_heading_row",
            TableClass::SpannerRow => "spanner_row",
            TableClass::StubHead => "stub_head",
            TableClass::Spanner => "spanner",
            TableClass::SpannerLabel => "spanner_label",
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
/// Position is an inclusive grid rectangle: `top`/`bottom` are row indices,
/// `left`/`right` are column indices, 0-based, inclusive on both ends. A
/// non-spanning cell has `top == bottom` and `left == right`.
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
    /// Build a cell with no display properties or classes. A
    /// `ColumnLabel`/`Body` cell should follow with `with_properties` and
    /// `with_classes`.
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
    /// `standardise_hjust`; the number is bucketed with the same
    /// `0.25`/`0.75` thresholds `VegaLiteWriter`'s `convert_hjust` uses.
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
    /// delegates to `TableCellKind::is_header`.
    pub fn is_header(&self) -> bool {
        self.kind.is_header()
    }

    /// This cell's width in columns — `right - left + 1`, since both bounds
    /// are inclusive.
    pub fn width(&self) -> usize {
        self.right - self.left + 1
    }

    /// This cell's height in rows — `bottom - top + 1`, since both bounds
    /// are inclusive.
    pub fn height(&self) -> usize {
        self.bottom - self.top + 1
    }
}

/// One column's identity within a table layout: its source name and
/// resolved display label/properties, as resolved by
/// `create_table_columns`.
#[derive(Debug, Clone, PartialEq)]
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

/// Standardise an `hjust` value to a number: `"left"`,
/// `"center"`/`"centre"`, `"right"` become `0.0`, `0.5`, `1.0` (any other
/// string is `0.5`); a number passes through; anything else is `None`.
pub(crate) fn standardise_hjust(value: &ParameterValue) -> Option<f64> {
    match value {
        ParameterValue::String(s) if s == "left" => Some(0.0),
        ParameterValue::String(s) if s == "right" => Some(1.0),
        ParameterValue::String(_) => Some(0.5),
        ParameterValue::Number(n) => Some(*n),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;

    #[test]
    fn table_cell_kind_display_names_every_variant() {
        assert_eq!(TableCellKind::ColumnLabel.to_string(), "column label");
        assert_eq!(TableCellKind::StubHead.to_string(), "stub head");
        assert_eq!(TableCellKind::Body.to_string(), "body");
        assert_eq!(TableCellKind::StubRowLabel.to_string(), "stub row label");
        assert_eq!(TableCellKind::Spanner.to_string(), "spanner");
        assert_eq!(TableCellKind::Title.to_string(), "title");
        assert_eq!(TableCellKind::Subtitle.to_string(), "subtitle");
        assert_eq!(TableCellKind::Caption.to_string(), "caption");
        assert_eq!(TableCellKind::Filler.to_string(), "filler");
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
}

/// A table's layout: its cells alongside one `TableRow` **per row index they
/// occupy** — not one per cell, since several cells can share a row (a
/// column-label row has one cell per column, but is still a single row) —
/// and one `TableColumn` per column index. Both the build-time composition
/// unit and the resolved handoff to a writer: every `create_*` helper below
/// builds all three together, so `rows` is never derived from `cells` and a
/// row's classification (`is_header`, `Heading`) is decided where its cells
/// are built.
///
/// The type is public because `Writer::write_table` takes one; construction
/// and composition stay crate-internal.
pub struct TableLayout {
    pub(crate) columns: Vec<TableColumn>,
    pub(crate) rows: Vec<TableRow>,
    pub(crate) cells: Vec<TableCell>,
}

impl TableLayout {
    /// Build a `TableLayout`, `debug_assert`ing that `rows.len()` matches
    /// the row extent `cells` implies and that `cells` stays within
    /// `columns`' width — a part-way layout (e.g. spanners covering only
    /// some columns) may be narrower; `validate` requires the full width
    /// once composed.
    pub(crate) fn new(
        columns: Vec<TableColumn>,
        rows: Vec<TableRow>,
        cells: Vec<TableCell>,
    ) -> TableLayout {
        debug_assert_eq!(
            rows.len(),
            count_cell_rows(&cells),
            "TableLayout's row count doesn't match its own cells' row extent"
        );
        debug_assert!(
            columns.len() >= count_cell_cols(&cells),
            "TableLayout's cells exceed its columns' width"
        );
        TableLayout {
            columns,
            rows,
            cells,
        }
    }

    /// Number of rows in the layout — heading, spanner, and column-label
    /// rows included, not just data rows.
    pub fn nrow(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns in the layout.
    pub fn ncol(&self) -> usize {
        self.columns.len()
    }

    /// The layout's cells: one per column label and per data value.
    pub fn cells(&self) -> &[TableCell] {
        &self.cells
    }

    /// Resolved per-column properties — a writer wanting a whole-column
    /// value (e.g. `width`) reads it here instead of the same value repeated
    /// across the column's cells.
    pub fn columns(&self) -> &[TableColumn] {
        &self.columns
    }

    /// Resolved per-row properties, symmetric with `columns`.
    pub fn rows(&self) -> &[TableRow] {
        &self.rows
    }

    /// Stack `bottom` below `self`, offsetting `bottom`'s cells down by
    /// `self`'s row extent and concatenating both sides' `rows` — no
    /// offsetting needed there, since a `TableRow`'s position *is* its
    /// index, unlike a `TableCell`'s rectangle.
    pub(crate) fn rowbind(self, mut bottom: TableLayout) -> TableLayout {
        debug_assert_eq!(
            self.columns, bottom.columns,
            "rowbind both sides to have identical columns"
        );
        if self.cells.is_empty() {
            return bottom;
        }
        if bottom.cells.is_empty() {
            return self;
        }

        let row_offset = self.nrow();
        for cell in &mut bottom.cells {
            cell.offset_rows(row_offset);
        }

        let mut rows = self.rows;
        rows.extend(bottom.rows);
        let mut cells = self.cells;
        cells.extend(bottom.cells);

        TableLayout::new(self.columns, rows, cells)
    }

    /// Check a fully composed layout: `rows` and `columns` match the extents
    /// `cells` implies, and no two cells claim the same grid position.
    ///
    /// The overlap check walks every cell's full footprint into a map of
    /// occupied positions to the `TableCellKind` that claimed each one,
    /// erroring — naming both kinds — on the first double claim.
    /// `O(total cell area)`.
    pub fn validate(&self) -> Result<()> {
        let row_extent = count_cell_rows(&self.cells);
        if self.rows.len() != row_extent {
            return Err(GgsqlError::ValidationError(format!(
                "Table layout has {} rows but its cells span {row_extent}",
                self.rows.len()
            )));
        }
        if !self.cells.is_empty() {
            let col_extent = count_cell_cols(&self.cells);
            if self.columns.len() != col_extent {
                return Err(GgsqlError::ValidationError(format!(
                    "Table layout has {} columns but its cells span {col_extent}",
                    self.columns.len()
                )));
            }
        }

        let mut occupied: std::collections::HashMap<(usize, usize), TableCellKind> =
            std::collections::HashMap::new();

        for cell in &self.cells {
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

    /// Consume `self` into just its cells.
    #[cfg(test)]
    pub(crate) fn into_cells(self) -> Vec<TableCell> {
        self.cells
    }
}
/// Build the title/subtitle heading rows (gt's `title_row`/`subtitle_row`),
/// each spanning the table's full width and numbered locally from
/// `top == 0`. Each row carries `TableClass::Heading` (gt's `gt_heading`).
pub(crate) fn create_heading(
    title: Option<&str>,
    subtitle: Option<&str>,
    columns: &[TableColumn],
) -> TableLayout {
    let right = columns.len().saturating_sub(1);
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
    TableLayout::new(columns.to_vec(), rows, cells)
}
/// Build the caption cell (gt's `tab_caption()`), spanning the table's full
/// width at local row 0; `rowbind` places it at the very end of the layout.
/// `HtmlWriter` pulls this cell out of the grid before rendering, since HTML
/// requires `<caption>` outside `<thead>`/`<tbody>`. Its row carries no
/// class — a caption never lands in a rendered `<tr>`.
pub(crate) fn create_caption(caption: Option<&str>, columns: &[TableColumn]) -> TableLayout {
    match caption {
        Some(caption) => TableLayout::new(
            columns.to_vec(),
            vec![TableRow::default()],
            vec![TableCell::new(
                TableCellKind::Caption,
                0,
                0,
                0,
                columns.len().saturating_sub(1),
                caption.to_string(),
            )
            .with_classes(vec![TableClass::Caption])],
        ),
        None => TableLayout::new(columns.to_vec(), Vec::new(), Vec::new()),
    }
}
/// Build one `ColumnLabel` cell per non-stub column, numbered locally from
/// `top == 0`. Skips every stub column; `create_stubhead` builds those.
pub(crate) fn create_column_labels(columns: &[TableColumn]) -> Vec<TableCell> {
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

/// Pop the reserved `title`/`subtitle`/`caption` keys out of a `TABULATE`
/// `LABEL` clause's resolved map, leaving only genuine column-name
/// overrides behind. `.flatten()` treats "key absent" and "key present but
/// NULL" the same: neither produces a title/subtitle/caption.
pub(crate) fn extract_heading_labels(
    labels: &mut Labels,
) -> (Option<String>, Option<String>, Option<String>) {
    let title = labels.labels.remove("title").flatten();
    let subtitle = labels.labels.remove("subtitle").flatten();
    let caption = labels.labels.remove("caption").flatten();
    (title, subtitle, caption)
}

/// Move every STUB-targeted column to the front, preserving each group's
/// relative order. Runs after `reorder_table_columns`, as the final word on
/// column position — a SPAN can never cover a stub column
/// (`Table::validate_span_stub_boundary`).
pub(crate) fn move_stub_columns(columns: Vec<TableColumn>) -> Vec<TableColumn> {
    let (mut stub, body): (Vec<_>, Vec<_>) = columns.into_iter().partition(TableColumn::is_stub);
    stub.extend(body);
    stub
}
/// Stack spanner rows above column labels, letting a column with no spanner
/// covering it stretch its own label upward over the gap, then fill the
/// remaining gaps.
pub(crate) fn compose_header(spanners: TableLayout, column_labels: TableLayout) -> TableLayout {
    let num_spanner_rows = spanners.nrow();
    let header = spanners.rowbind(column_labels);
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
fn stretch_unspanned_column_labels(
    mut header: TableLayout,
    num_spanner_rows: usize,
) -> TableLayout {
    if num_spanner_rows == 0 {
        return header;
    }

    let ncol = header.ncol();
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
fn fill_spanner_gaps(mut header: TableLayout, num_spanner_rows: usize) -> TableLayout {
    let ncol = header.ncol();

    // Coverage set, O(area) like `TableLayout::validate`.
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

#[cfg(test)]
mod assembly_tests {
    use super::*;
    use crate::params::Parameters;
    use crate::{ColumnSection, GgsqlError};

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
    fn cell(kind: TableCellKind, top: usize, bottom: usize, content: &str) -> TableCell {
        TableCell::new(kind, top, bottom, 0, 0, content.to_string())
    }

    /// Wrap a hand-built `Vec<TableCell>` into a `TableLayout` for a test,
    /// with exactly as many default rows and columns as `cells` implies —
    /// satisfies `TableLayout::new`'s invariants without every test having
    /// to spell out `rows`/`columns` vecs of its own.
    fn layout(cells: Vec<TableCell>) -> TableLayout {
        let ncol = count_cell_cols(&cells);
        layout_wide(cells, ncol)
    }

    /// `layout`, with an explicit column count — for a part that doesn't
    /// itself touch every column of the table it belongs to (a spanner row
    /// covering only some columns, or an empty part).
    fn layout_wide(cells: Vec<TableCell>, ncol: usize) -> TableLayout {
        let columns = (0..ncol).map(|i| column(&i.to_string(), "")).collect();
        let rows = vec![TableRow::default(); count_cell_rows(&cells)];
        TableLayout::new(columns, rows, cells)
    }

    #[test]
    fn rowbind_shifts_the_bottom_below_a_single_top_row() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];
        let body = vec![
            cell(TableCellKind::Body, 0, 0, "1"),
            cell(TableCellKind::Body, 1, 1, "2"),
        ];

        let cells = layout(column_labels).rowbind(layout(body)).cells;

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

        let cells = layout(column_labels).rowbind(layout(body)).cells;

        assert_eq!(cells[1].top, 2);
        assert_eq!(cells[1].bottom, 2);
    }

    #[test]
    fn rowbind_returns_bottom_unchanged_when_top_is_empty() {
        let body = vec![cell(TableCellKind::Body, 0, 0, "1")];

        let cells = layout_wide(Vec::new(), 1).rowbind(layout(body)).cells;

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
    }

    #[test]
    fn rowbind_returns_top_unchanged_when_bottom_is_empty() {
        let column_labels = vec![cell(TableCellKind::ColumnLabel, 0, 0, "id")];

        let cells = layout(column_labels)
            .rowbind(layout_wide(Vec::new(), 1))
            .cells;

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

        let header = compose_header(layout_wide(spanners, 3), layout(column_labels)).cells;

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

        let header = compose_header(layout(spanners), layout(column_labels)).cells;

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

        let header = compose_header(layout_wide(spanners, 3), layout(column_labels)).cells;

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

        let header = compose_header(layout(spanners), layout(column_labels)).cells;

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

        let header = compose_header(layout_wide(Vec::new(), 1), layout(column_labels)).cells;

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

        assert!(layout(cells).validate().is_ok());
    }

    #[test]
    fn validate_overlaps_rejects_two_cells_at_the_same_position() {
        let cells = vec![
            cell_at(TableCellKind::Body, 0, 0, 0, 0),
            cell_at(TableCellKind::Body, 0, 0, 0, 0),
        ];

        let error = layout(cells).validate().unwrap_err();
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

        let error = layout(cells).validate().unwrap_err();
        assert!(matches!(error, GgsqlError::ValidationError(msg)
            if msg.contains("column label cell and a spanner cell")));
    }
}
