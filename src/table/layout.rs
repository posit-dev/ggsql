//! Resolved table layout vocabulary: the positioned `TableCell`, its
//! `TableCellKind`, and the writer-neutral `TableClass` style classes.

use crate::params::Parameters;
use crate::plot::ParameterValue;

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
