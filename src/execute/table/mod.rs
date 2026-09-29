//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling to do here — just the one query that
//! produces `body`, plus resolving that data into positioned `TableCell`s.
//! `SPAN`-specific resolution (column reordering, header-row assignment)
//! lives in the child `spanner` module, `TABULATE FORMAT STUB`-specific
//! resolution (moving stub columns to the front, building their
//! `StubHead`/`StubRowLabel` cells) lives in `stub`, `FORMAT`-specific
//! resolution (replacing a column's values with its resolved display text)
//! lives in `format`, and building the resolved `TableCell`/`TableRow` grid
//! out of all three lives in `layout` — all four used only from here, unlike
//! Plot's own resolution logic, which is split across the flat siblings
//! `schema.rs`/`casting.rs`/`layer.rs`/`scale.rs`/`position.rs`/`cte.rs`
//! because those are each reachable from more than one place.
//! `Table::resolve_spanner_ids` is the exception — it needs no `DataFrame`,
//! so it lives on `Table` itself, reachable from `validate()` too.

mod format;
mod layout;
mod spanner;
mod stub;

pub use layout::{TableColumn, TableRow};
// Crate-internal only (not part of the public API): the row/column-extent-
// from-cells helpers, needed by `writer::html` and `reader::spec` as well as
// `layout` itself.
pub(crate) use layout::{count_cell_cols, count_cell_rows};

use format::{apply_formats, setup_formats, standardise_hjust};
use layout::{build_cells, extract_heading_labels, setup_columns};

use super::cte::split_with_query;
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

    let sql = build_table_sql(&source_tree, &table.selection).ok_or_else(|| {
        GgsqlError::ValidationError(
            "TABULATE has no data source: add a FROM, or a SQL query before it".to_string(),
        )
    })?;

    let df = reader.execute_sql(&sql)?;

    let column_names = df.get_column_names();
    let spans = table
        .resolve_spanners(&labels, Some(&column_names))
        .map_err(GgsqlError::ValidationError)?;
    // The shape both create_table_columns (SETTING) and apply_formats
    // (RENAMING) read from. Resolved after spans so a FORMAT column entry
    // can name a SPAN id in place of the columns it covers.
    let formats = setup_formats(&df, &table.formats, &spans)?;
    let columns = setup_columns(&df, &spans, &labels, &formats);
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

/// Builds the SQL a `TABULATE` query executes, folding `selection`
/// (`Table::selection`) in as the outer projection. Table-side counterpart
/// to `execute::cte::transform_global_sql`, without its CTE-rewriting or
/// cache-staging — a `Table` has neither.
fn build_table_sql(source_tree: &SourceTree, selection: &str) -> Option<String> {
    let root = source_tree.root();

    // A WITH...SELECT tail, or a plain trailing SELECT.
    let select_sql = split_with_query(source_tree)
        .map(|(_, select)| select)
        .or_else(|| source_tree.find_text(&root, "(sql_statement (select_statement) @select)"));

    if let Some(select_sql) = select_sql {
        return Some(if selection == "*" {
            select_sql
        } else {
            format!("SELECT {selection} FROM ({select_sql})")
        });
    }

    // No trailing SELECT: fall back to TABULATE FROM <source>.
    let first_stmt = source_tree.first_stmt(&root)?;
    let from_source = source_tree.find_text(
        &first_stmt,
        r#"(tabulate_statement (single_source_from source: (_) @source))"#,
    );

    if let Some(source) = from_source {
        return Some(format!("SELECT {selection} FROM {source}"));
    }

    // Neither: e.g. a bare DuckDB-style `FROM t`. This text may carry a
    // leading setup-statement prefix (INSTALL/LOAD/SET), which a non-"*"
    // selection then wraps into an invalid subquery — a narrow, accepted gap.
    let fallback = source_tree.extract_sql()?;
    Some(if selection == "*" {
        fallback
    } else {
        format!("SELECT {selection} FROM ({fallback})")
    })
}

// =============================================================================
// Public API: TableCell
// =============================================================================

/// What role a `TableCell` plays in the table's layout.
///
/// Naming follows R's gt package (`column_labels`, `body`, `stub`, ...),
/// since ggsql's table grammar is expected to keep drawing on its part
/// vocabulary as more of it (footnotes, source notes) gets built out here.
///
/// Lets a writer tell cells apart (e.g. `<th>` vs `<td>`) without relying on
/// position — a column label is a `ColumnLabel` cell, not "whatever's in row
/// 0".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableCellKind {
    /// A column label (gt's `column_labels`).
    ColumnLabel,
    /// The stub's own header cell (gt's `tab_stubhead()`) — same row as
    /// `ColumnLabel`, but over a `TABULATE FORMAT STUB` column rather than
    /// a regular one, hence its own kind.
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
    /// Whether a cell of this kind belongs in a table's header (`ColumnLabel`,
    /// `StubHead`, `Spanner`, `Title`, `Subtitle`, `Filler`) rather than its
    /// body (`Body`, `StubRowLabel`) — the kind alone decides it, via
    /// `TableCell::is_header` below. `Caption` stays out of this: a writer
    /// is expected to pull it out of the grid entirely rather than render
    /// it as either.
    pub fn is_header(self) -> bool {
        matches!(
            self,
            TableCellKind::ColumnLabel
                | TableCellKind::StubHead
                | TableCellKind::Spanner
                | TableCellKind::Title
                | TableCellKind::Subtitle
                | TableCellKind::Filler // every case today is header-shaped
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
/// `kind` informs layout (header vs body, spans), while classes inform style,
/// and positional variants like `SpannerOuter` are only known while the
/// layout is being built. A writer maps each variant to its own class
/// vocabulary — the HTML writer prefixes its `Display` with `ggsql_`.
///
/// Not every variant applies to every carrier: `Heading` and `ColHeadingRow`
/// are row-scoped (`TableRow::classes` — the `<tr>` wrapping a
/// `Title`/`Subtitle` cell, and the `<tr>` wrapping the column-label row,
/// respectively); every other structural/alignment variant is cell-scoped
/// (`TableCell::classes`). `Table` and `TableBody` are scoped to the table as
/// a whole (or a whole section of it) — there is no resolved layout type
/// representing "the whole table"/"the whole body" for either to be recorded
/// on, so a writer applies them directly to its own top-level elements (the
/// `<table>` and `<tbody>` respectively) rather than reading them off
/// `cells`/`rows`. Nothing in the type enforces any of this — it's a
/// per-variant convention, since a writer maps every carrier through the same
/// class-name/declaration lookup regardless of where it came from. A future
/// `TableColumn`-scoped class (a `<col>`/`<colgroup>` concern) belongs in this
/// same enum too.
///
/// Naming follows gt's classes (`gt_row`, `gt_col_heading`,
/// `gt_column_spanner_outer`, ...). The structural variants are recorded by
/// `build_cells()`; the alignment variants are a cell's resolved `hjust`
/// expressed as a class, appended by `TableCell::discretise_hjust` rather
/// than recorded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableClass {
    /// The whole table (gt's `gt_table`).
    Table,
    /// The `<tbody>` wrapping every body row (gt's `gt_table_body`). Not to
    /// be confused with `Row`, which is per-cell rather than for the whole
    /// section.
    TableBody,
    /// A body cell (gt's `gt_row`).
    Row,
    /// A row-label cell in the stub (`TableCellKind::StubRowLabel`).
    Stub,
    /// A column-label cell (gt's `gt_col_heading`).
    ColHeading,
    /// Row-scoped: the `<tr>` wrapping the row of column-label cells (gt's
    /// `gt_col_headings` — note the plural, distinguishing it from the
    /// singular per-cell `gt_col_heading`).
    ColHeadingRow,
    /// The stub's own header cell (`TableCellKind::StubHead`). No gt
    /// equivalent — gt's own stubhead has no dedicated CSS class of its
    /// own, unlike this one.
    StubHead,
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
            TableClass::Table => "table",
            TableClass::TableBody => "table_body",
            TableClass::Row => "row",
            TableClass::Stub => "stub",
            TableClass::ColHeading => "col_heading",
            TableClass::ColHeadingRow => "col_heading_row",
            TableClass::StubHead => "stub_head",
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
        let resolved = resolve_table_with_reader("TABULATE * FROM sales", &reader).unwrap();

        assert_eq!(resolved.sql(), "SELECT * FROM sales");
        // 3 data rows + 1 column-label row: nrow() is the whole grid, not
        // just the data rows.
        assert_eq!(resolved.nrow(), 4);
        assert_eq!(resolved.ncol(), 2);
    }

    #[test]
    fn test_tabulate_selection_picks_and_renames_columns() {
        let reader = reader_with_sales();

        let picked =
            resolve_table_with_reader("TABULATE name, id AS Number FROM sales", &reader).unwrap();
        assert_eq!(picked.sql(), "SELECT name, id AS Number FROM sales");
        assert_eq!(picked.ncol(), 2);
    }

    #[test]
    fn test_tabulate_selection_wildcard_with_rename() {
        let reader = reader_with_sales();

        let mixed =
            resolve_table_with_reader("TABULATE *, id AS Number FROM sales", &reader).unwrap();
        assert_eq!(mixed.sql(), "SELECT *, id AS Number FROM sales");
    }

    #[test]
    fn test_tabulate_selection_with_no_tabulate_from() {
        let reader = reader_with_sales();

        let no_from =
            resolve_table_with_reader("SELECT * FROM sales TABULATE name, id AS Number", &reader)
                .unwrap();
        assert_eq!(
            no_from.sql(),
            "SELECT name, id AS Number FROM (SELECT * FROM sales)"
        );
        assert_eq!(no_from.ncol(), 2);
    }

    #[test]
    fn test_bare_tabulate_uses_preceding_select() {
        let reader = reader_with_sales();

        let from_only = resolve_table_with_reader("TABULATE * FROM sales", &reader).unwrap();
        let select_then_tabulate =
            resolve_table_with_reader("SELECT * FROM sales TABULATE *", &reader).unwrap();

        assert_eq!(from_only.sql(), select_then_tabulate.sql());
        assert_eq!(from_only.nrow(), select_then_tabulate.nrow());
    }

    #[test]
    fn test_tabulate_with_no_source_errors() {
        let reader = reader_with_sales();
        let result = resolve_table_with_reader("TABULATE *", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_does_not_borrow_a_later_visualise_from() {
        // A source-less TABULATE followed by an unrelated VISUALISE FROM must
        // still error "no data source", not silently resolve against the
        // VISUALISE's FROM.
        let reader = reader_with_sales();
        let result =
            resolve_table_with_reader("TABULATE * VISUALISE FROM sales DRAW point", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_label_applies_under_the_tab_clause_wrapper() {
        // label_clause is nested under tab_clause in the grammar (so LABEL
        // and SPAN can appear in any order) — build_tabulate_statement must
        // unwrap it, not match "label_clause" as a direct child.
        let reader = reader_with_sales();
        let resolved =
            resolve_table_with_reader("TABULATE * FROM sales LABEL id => 'ID'", &reader).unwrap();

        let label_cell = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::ColumnLabel && c.left == 0)
            .unwrap();
        assert_eq!(label_cell.content, "ID");
    }

    #[test]
    fn test_tabulate_format_stub_moves_the_column_to_the_front() {
        let reader = reader_with_sales();
        let resolved =
            resolve_table_with_reader("TABULATE * FROM sales FORMAT STUB name", &reader).unwrap();

        let columns = resolved.columns();
        assert_eq!(columns[0].name, "name");
        assert_eq!(columns[0].target, crate::ColumnSection::Stub);
        assert_eq!(columns[1].name, "id");
        assert_eq!(columns[1].target, crate::ColumnSection::Body);

        let stub_cell = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::StubRowLabel)
            .unwrap();
        assert_eq!(stub_cell.left, 0);
        let stub_head = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::StubHead)
            .unwrap();
        assert_eq!(stub_head.left, 0);
    }

    #[test]
    fn test_tabulate_format_stub_head_is_blank_by_default_but_label_can_override() {
        let reader = reader_with_sales();
        let resolved =
            resolve_table_with_reader("TABULATE * FROM sales FORMAT STUB name", &reader).unwrap();
        let stub_head = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::StubHead)
            .unwrap();
        assert_eq!(stub_head.content, "");

        let resolved = resolve_table_with_reader(
            "TABULATE * FROM sales FORMAT STUB name LABEL name => 'Name'",
            &reader,
        )
        .unwrap();
        let stub_head = resolved
            .cells()
            .iter()
            .find(|c| c.kind == TableCellKind::StubHead)
            .unwrap();
        assert_eq!(stub_head.content, "Name");
    }

    #[test]
    fn test_tabulate_span_over_a_stub_column_errors() {
        let reader = reader_with_sales();
        let result = resolve_table_with_reader(
            "TABULATE * FROM sales FORMAT STUB name SPAN G ACROSS name, id",
            &reader,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_span_unknown_column_errors_even_with_gather_disabled() {
        // Table::resolve_spanners checks every ACROSS entry regardless of
        // gather, so this errors even though gather_columns itself (the
        // only place that also checks column existence) is skipped here.
        let reader = reader_with_sales();
        match resolve_table_with_reader(
            "TABULATE * FROM sales SPAN G ACROSS nope, id SETTING gather => false",
            &reader,
        ) {
            Ok(_) => panic!("expected an unknown ACROSS column to error"),
            Err(GgsqlError::ValidationError(msg)) => {
                assert!(msg.contains("SPAN references unknown column 'nope'"))
            }
            Err(e) => panic!("expected a ValidationError, got {e:?}"),
        }
    }
}
