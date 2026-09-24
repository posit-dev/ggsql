//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling to do here — just the one query that
//! produces `body`, plus resolving that data into positioned `TableCell`s.
//! `SPAN`-specific resolution (column reordering, header-row assignment)
//! lives in the child `spanner` module, `FORMAT`-specific resolution
//! (replacing a column's values with its resolved display text) lives in
//! `format`, and building the resolved `TableCell`/`TableRow` grid out of
//! both lives in `layout` — all three used only from here, unlike Plot's own
//! resolution logic, which is split across the flat siblings `schema.rs`/
//! `casting.rs`/`layer.rs`/`scale.rs`/`position.rs`/`cte.rs` because those
//! are each reachable from more than one place. `Table::resolve_spanner_ids`
//! is the exception — it needs no `DataFrame`, so it lives on `Table`
//! itself, reachable from `validate()` too.

mod format;
mod layout;
mod spanner;

pub use layout::{TableColumn, TableRow};
// Crate-internal only (not part of the public API): the row/column-extent-
// from-cells helpers, needed by `writer::html` and `reader::spec` as well as
// `layout` itself.
pub(crate) use layout::{count_cell_cols, count_cell_rows};

use format::{apply_formats, setup_formats, standardise_hjust};
use layout::{build_cells, extract_heading_labels, setup_columns};

use crate::parser::{self, SourceTree};
use crate::plot::Parameters;
use crate::reader::{Reader, ResolvedTable};
use crate::validate::{validate, ValidationWarning};
use crate::{GgsqlError, Result, Spec};

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

    Ok(ResolvedTable::new(cells, columns, rows, sql, warnings))
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
mod tests {
    use super::*;
    use crate::plot::ParameterValue;

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

#[cfg(test)]
#[cfg(feature = "duckdb")]
mod integration_tests {
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
