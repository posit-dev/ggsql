//! A minimal HTML table writer.
//!
//! Maps `ResolvedTable`'s three parts onto distinct pieces of the `<table>`:
//! - `cells` → `<thead>`/`<tbody>` rows, plus one carve-out: a `Caption`-kind
//!   cell is pulled out of the grid before row rendering and emitted as a
//!   `<caption>` (HTML requires it as `<table>`'s first child, outside
//!   `<thead>`/`<tbody>` entirely) — see `write_table`. Everything else,
//!   including the new `Title`/`Subtitle` heading rows, renders through the
//!   normal row logic below. How a cell's (or a heading row's) styling is
//!   expressed depends on the `css_mode` option: `class` (the default) puts
//!   `ggsql_*` classes on the elements — from the recorded `TableClass`es
//!   plus property-derived classes like alignment — backed by a `<style>`
//!   block; `inline` renders the same declarations into each element's
//!   `style` attribute instead (the two gt `as_raw_html()` modes).
//! - `columns` → a `<colgroup>`, one `<col>` per column, with a `style`
//!   attribute from that column's resolved `width` (a bare `<col>` for a
//!   column with none) in both modes — targeted column styling stays inline,
//!   as gt keeps `tab_style()` rules inline.
//! - `rows` → a class on the `<tr>` itself (`Heading`, for a `Title`/
//!   `Subtitle` row) — the only row-wide property that exists to render so
//!   far.
//!
//! No footnotes yet, since `Table` has no field for those. Spanner rows are
//! rendered (as `colspan`, one `<tr>` per level, above the column labels);
//! `render_cell`/`render_row` can also render a `rowspan` cell, though
//! nothing in the resolution pipeline produces one for a data column yet, so
//! a column with no spanner at a given level still gets a blank filler cell
//! rather than a merged one. This is a stub to prove the Table → writer
//! plumbing end to end, not the real grammar-of-tables output; it
//! deliberately does not reuse `ggsql-jupyter`'s existing
//! `dataframe_to_html`, which works directly off a `DataFrame` rather than
//! resolved `TableCell`s.

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::execute::{count_cell_cols, count_cell_rows};
use crate::plot::{ParameterValue, Parameters};
use crate::util::escape_html;
use crate::writer::{Writer, WriterOptions};
use crate::{
    DataFrame, GgsqlError, Plot, Result, TableCell, TableCellKind, TableClass, TableColumn,
    TableRow,
};

/// Renders a resolved table as a bare HTML `<table>`. Does not support plots.
#[derive(Debug, Default)]
pub struct HtmlWriter {
    css_mode: CssMode,
}

/// How cell styling is expressed in the output HTML — the two gt
/// `as_raw_html()` modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CssMode {
    /// `ggsql_*` classes on the elements, backed by a `<style>` block
    /// carrying the style classes' rules (gt's `inline_css = FALSE`).
    #[default]
    Class,
    /// Declarations rendered into each element's `style` attribute; no
    /// classes, no `<style>` block (gt's `inline_css = TRUE`).
    Inline,
}

impl HtmlWriter {
    /// Create a new HtmlWriter (class mode; see `css_mode`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Writer for HtmlWriter {
    type Output = String;

    /// Takes one option: `css_mode` — `class` (default) renders `ggsql_*`
    /// classes backed by a `<style>` block; `inline` renders the same
    /// declarations into each element's `style` attribute.
    fn from_options(options: &WriterOptions) -> Result<Self> {
        options.reject_unknown(&["css_mode"])?;
        let css_mode = match options.one_of("css_mode", &["inline", "class"])? {
            Some("inline") => CssMode::Inline,
            _ => CssMode::Class,
        };
        Ok(Self { css_mode })
    }

    fn write_plot(&self, _spec: &Plot, _data: &HashMap<String, DataFrame>) -> Result<String> {
        Err(GgsqlError::WriterError(
            "HtmlWriter does not support plots".to_string(),
        ))
    }

    fn validate_plot(&self, _spec: &Plot) -> Result<()> {
        Err(GgsqlError::WriterError(
            "HtmlWriter does not support plots".to_string(),
        ))
    }

    fn write_table(
        &self,
        cells: &[TableCell],
        columns: &[TableColumn],
        rows: &[TableRow],
    ) -> Result<String> {
        // Fold `hjust` into alignment classes up front, so rendering reads
        // `classes` alone.
        let cells: Vec<TableCell> = cells
            .iter()
            .cloned()
            .map(TableCell::discretise_hjust)
            .collect();

        // HTML requires `<caption>` outside `<thead>`/`<tbody>`, as
        // `<table>`'s first child, so it's rendered separately below
        // through `render_caption` rather than `render_row`/`render_cell`.
        // Splitting it out here, before `ncol`/`nrow` are computed, means
        // the header/body grid and everything below only ever see genuine
        // header/body rows — `create_caption`/`rowbind` always place it as
        // the grid's trailing row, so nothing further down needs its own
        // caption check.
        let (caption_cells, cells): (Vec<TableCell>, Vec<TableCell>) = cells
            .into_iter()
            .partition(|cell| cell.kind == TableCellKind::Caption);

        let ncol = count_cell_cols(&cells);
        let nrow = count_cell_rows(&cells);

        let (header_rows, body_rows) = split_rows(rows, nrow);

        // Not a general TableRow invariant — just what this writer's split
        // into two blocks requires.
        if let (Some(&max_header_row), Some(&min_body_row)) =
            (header_rows.last(), body_rows.first())
        {
            if max_header_row >= min_body_row {
                return Err(GgsqlError::WriterError(format!(
                    "HtmlWriter renders headers and body as separate <thead>/<tbody> blocks, so \
                     every header cell must sit above every body cell; found a header row at \
                     {max_header_row} at or below a body row at {min_body_row}"
                )));
            }
        }

        let mut all_slots = build_all_slots(&cells, ncol, nrow);

        let mut html = String::new();
        if self.css_mode == CssMode::Class {
            html.push_str(&render_style_block());
        }
        html.push_str("<table>\n");

        for caption in &caption_cells {
            html.push_str(&render_caption(caption, self.css_mode));
            html.push('\n');
        }

        if let Some(colgroup) = render_colgroup(columns) {
            html.push_str(&colgroup);
        }

        if !header_rows.is_empty() {
            html.push_str("<thead>\n");
            for row in &header_rows {
                let slots = all_slots
                    .remove(row)
                    .expect("a header row's own cell always puts it in build_all_slots' result");
                html.push_str(&render_row(&slots, &rows[*row], self.css_mode));
            }
            html.push_str("</thead>\n");
        }

        if !body_rows.is_empty() {
            html.push_str("<tbody>\n");
            for row in &body_rows {
                let slots = all_slots
                    .remove(row)
                    .expect("a body row's own cell always puts it in build_all_slots' result");
                html.push_str(&render_row(&slots, &rows[*row], self.css_mode));
            }
            html.push_str("</tbody>\n");
        }

        html.push_str("</table>");

        Ok(html)
    }
}

/// What a `TableClass` stands for in CSS, as property–value pairs. The
/// single source of truth both `css_mode`s render from — class mode puts
/// these in the `<style>` block, inline mode folds them into the element's
/// `style` attribute. Structural classes carry no declarations — they're
/// pure semantic hooks for the embedder.
fn class_declarations(class: TableClass) -> &'static [(&'static str, &'static str)] {
    match class {
        TableClass::Row
        | TableClass::ColHeading
        | TableClass::Spanner
        | TableClass::SpannerOuter
        | TableClass::Title
        | TableClass::Subtitle
        | TableClass::Caption
        | TableClass::Heading => &[],
        TableClass::AlignLeft => &[("text-align", "left")],
        TableClass::AlignCenter => &[("text-align", "center")],
        TableClass::AlignRight => &[
            ("text-align", "right"),
            // Right-aligned data reads as numeric — keep digit widths uniform
            // so they still line up under one another.
            ("font-variant-numeric", "tabular-nums"),
        ],
    }
}

/// The styled classes (those carrying declarations), in `<style>`-block rule
/// order. With equal specificity the later rule wins, so this decides
/// precedence in class mode; `inline_style`'s own precedence instead follows
/// the order of the cell's `classes` field (see its doc comment).
const STYLED_CLASSES: &[TableClass] = &[
    TableClass::AlignLeft,
    TableClass::AlignCenter,
    TableClass::AlignRight,
];

/// The `<style>` block class mode prepends — one rule per styled class,
/// generated from the same lookup inline mode folds into `style` attributes,
/// so the modes can't drift. Emitted unconditionally: three short rules
/// aren't worth a pass over the cells to see which are used.
fn render_style_block() -> String {
    let mut block = String::from("<style>\n");
    for &class in STYLED_CLASSES {
        let name = format!("ggsql_{class}");
        let declarations = class_declarations(class)
            .iter()
            .map(|(property, value)| format!("{property}: {value}"))
            .collect::<Vec<_>>()
            .join("; ");
        block.push_str(&format!(".{name} {{ {declarations}; }}\n"));
    }
    block.push_str("</style>\n");
    block
}

/// Fold classes' declarations into a `style` attribute value, each CSS
/// property appearing at most once — a later class's declaration overrides
/// an earlier one's (CSS's last-wins rule, made explicit so the output
/// doesn't rely on browsers' duplicate-declaration handling).
fn inline_style(classes: &[TableClass]) -> Option<String> {
    let mut folded: Vec<(&str, &str)> = Vec::new();
    for class in classes {
        for &(property, value) in class_declarations(*class) {
            match folded.iter_mut().find(|(p, _)| *p == property) {
                Some(entry) => entry.1 = value,
                None => folded.push((property, value)),
            }
        }
    }
    (!folded.is_empty()).then(|| {
        folded
            .iter()
            .map(|(property, value)| format!("{property}: {value}"))
            .collect::<Vec<_>>()
            .join("; ")
    })
}

/// Render a `class` (class mode) or `style` (inline mode) attribute from a
/// set of recorded classes — shared by `render_cell` (a `<td>`/`<th>`),
/// `render_caption` (the extracted `<caption>`) and `render_row` (the
/// `<tr>`), since the same class list means the same declarations wherever
/// it ends up. Empty when `classes` is empty (class mode) or resolves no
/// declarations (inline mode).
fn styling_attr(classes: &[TableClass], mode: CssMode) -> String {
    match mode {
        CssMode::Class => {
            if classes.is_empty() {
                String::new()
            } else {
                let names = classes
                    .iter()
                    .map(|&class| format!("ggsql_{class}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!(" class=\"{names}\"")
            }
        }
        CssMode::Inline => inline_style(classes)
            .map(|style| format!(" style=\"{style}\""))
            .unwrap_or_default(),
    }
}

/// Render one `TableCell` as an HTML `<th>`/`<td>`, picked from
/// `cell.is_header()`, with a `colspan`/`rowspan` attribute only when the
/// cell actually spans more than one column/row. Styling per `mode` via
/// `styling_attr`. Not used for a `Caption`-kind cell — see `render_caption`.
fn render_cell(cell: &TableCell, mode: CssMode) -> String {
    let tag = if cell.is_header() { "th" } else { "td" };
    let colspan = cell.width();
    let rowspan = cell.height();
    let mut attrs = String::new();
    if colspan > 1 {
        attrs.push_str(&format!(" colspan=\"{colspan}\""));
    }
    if rowspan > 1 {
        attrs.push_str(&format!(" rowspan=\"{rowspan}\""));
    }
    attrs.push_str(&styling_attr(&cell.classes, mode));
    format!("<{tag}{attrs}>{}</{tag}>", escape_html(&cell.content))
}

/// Render a `Caption`-kind `TableCell` as a `<caption>` element — no
/// `colspan`/`rowspan`, since HTML doesn't take either there, and
/// `write_table` already pulled it out of the row grid before this is ever
/// called. Styling per `mode` via `styling_attr`, same as `render_cell`.
fn render_caption(cell: &TableCell, mode: CssMode) -> String {
    let attrs = styling_attr(&cell.classes, mode);
    format!("<caption{attrs}>{}</caption>", escape_html(&cell.content))
}

/// Render a `<colgroup>` block, one `<col>` per column, or `None` if
/// `columns` is empty or none of them resolve a `style`. Skipping the block
/// entirely in that case avoids emitting a run of bare, attribute-less
/// `<col>` tags that would render identically to omitting them.
fn render_colgroup(columns: &[TableColumn]) -> Option<String> {
    let styles: Vec<Option<String>> = columns
        .iter()
        .map(|c| column_style(&c.properties))
        .collect();
    if styles.iter().all(Option::is_none) {
        return None;
    }

    let mut html = String::from("<colgroup>\n");
    for style in styles {
        match style {
            Some(style) => html.push_str(&format!("<col style=\"{style}\">\n")),
            None => html.push_str("<col>\n"),
        }
    }
    html.push_str("</colgroup>\n");
    Some(html)
}

/// Translate a column's resolved `FORMAT` properties into a `<col>`'s
/// `style` attribute value, or `None` if it has no `width`. `width`'s value
/// is already validated (by `ParamConstraint::string_numeric_with_unit`) to
/// be a number immediately followed by `px` or `%`, which is exactly CSS's
/// own `width` syntax, so it's used as-is.
fn column_style(properties: &Parameters) -> Option<String> {
    match properties.get("width") {
        Some(ParameterValue::String(width)) => Some(format!("width: {width}")),
        _ => None,
    }
}

/// What belongs at one column of a rendered row, decided once for the whole
/// table by `build_all_slots` rather than worked out inline while
/// `render_row` walks columns.
enum Slot<'a> {
    /// A real cell starting at this column.
    Cell(&'a TableCell),
    /// A genuine gap: no real cell reaches this column, so a blank cell is
    /// synthesized matching the row's own dominant kind/classes.
    Filler {
        kind: TableCellKind,
        classes: &'a [TableClass],
    },
    /// A column a rowspan from an earlier row already claims — nothing
    /// rendered here at all, not even a filler.
    Skip,
}

/// Split every row index in `0..nrow` into header or body, per
/// `TableRow::is_header` — `rows` may carry one trailing entry for a
/// caption's own row (see `write_table`'s own caption split); bounding by
/// `nrow` is what keeps that entry out of either set. Each returned `Vec` is
/// already ascending, since `idx` only ever increases as `rows` is walked.
///
/// Sorting Hat: Hmmm... Yes... BODY ROW!!! *applause*
fn split_rows(rows: &[TableRow], nrow: usize) -> (Vec<usize>, Vec<usize>) {
    let mut header_rows = Vec::new();
    let mut body_rows = Vec::new();

    for (idx, row) in rows.iter().take(nrow).enumerate() {
        if row.is_header {
            header_rows.push(idx);
        } else {
            body_rows.push(idx);
        }
    }

    (header_rows, body_rows)
}

/// Decide what belongs at every column of every row, for the whole table, in
/// one pass over `cells` — a cell's rowspan/colspan footprint is marked as
/// the cell itself is placed. `ncol`/`nrow` size the grid; `write_table`
/// already has them, computed from `cells` itself, which by this point is
/// caption-free (see `write_table`'s own caption split).
///
/// Returns one entry per row that has at least one cell actually starting
/// in it — a row entirely swallowed by an earlier rowspan (nothing in it
/// but continuation) is omitted, since it gets no `<tr>` of its own either.
/// The caller is expected to only ever look up rows it already knows have a
/// real cell (e.g. from grouping the same `cells` by `top`), which is
/// exactly the rows this always produces.
fn build_all_slots(
    cells: &[TableCell],
    ncol: usize,
    nrow: usize,
) -> BTreeMap<usize, Vec<Slot<'_>>> {
    let mut grid: Vec<Vec<Option<Slot<'_>>>> = (0..nrow)
        .map(|_| (0..ncol).map(|_| None).collect())
        .collect();
    for cell in cells {
        for row in &mut grid[cell.top..=cell.bottom] {
            for col in &mut row[cell.left..=cell.right] {
                *col = Some(Slot::Skip);
            }
        }
        grid[cell.top][cell.left] = Some(Slot::Cell(cell));
    }

    grid.into_iter()
        .enumerate()
        .filter_map(|(row, row_slots)| {
            // A gap gets a filler matching the row's own dominant
            // kind/classes — read off whichever real cell started this row.
            // Known gap: assumes every real cell in a row shares one
            // `TableCellKind`/structural class, true of every row today but
            // not guaranteed to stay true once row stubs (labels/groups)
            // can share a row with `Body` cells — revisit then.
            let (kind, classes) = row_slots.iter().find_map(|slot| match slot {
                Some(Slot::Cell(cell)) => Some((cell.kind, cell.classes.as_slice())),
                _ => None,
            })?;
            let slots = row_slots
                .into_iter()
                .map(|slot| slot.unwrap_or(Slot::Filler { kind, classes }))
                .collect();
            Some((row, slots))
        })
        .collect()
}

/// Render a synthesized filler — a genuine gap in the grid with no real
/// cell reaching it — as a blank `<th>`/`<td>` matching its row's own
/// dominant kind/classes. Takes `kind`/`classes` directly rather than a
/// `&TableCell`, since a filler was never a real cell to begin with.
fn render_filler(kind: TableCellKind, classes: &[TableClass], mode: CssMode) -> String {
    let tag = if kind.is_header() { "th" } else { "td" };
    let attrs = styling_attr(classes, mode);
    format!("<{tag}{attrs}></{tag}>")
}

/// Render one row's already-decided `Slot`s as `<tr>...</tr>`. A `Skip`
/// slot renders nothing at all; a `Cell`/`Filler` slot renders through
/// `render_cell`/`render_filler`. Styling on the `<tr>` itself comes from
/// `row`'s recorded classes, same as any other element, via `styling_attr`.
fn render_row(slots: &[Slot], row: &TableRow, mode: CssMode) -> String {
    if slots.is_empty() {
        return String::new();
    }

    // Every `Cell`/`Filler` in a row shares one `TableCellKind` (a `Filler`
    // inherits it from whichever real cell started the row — see
    // `build_all_slots`), so the first one found is enough to cross-check
    // against `row.is_header` — the two are meant to always agree. Gated on
    // `debug_assertions` so the scan itself, not just the assertion, is
    // compiled out of a release build.
    #[cfg(debug_assertions)]
    if let Some(kind) = slots.iter().find_map(|slot| match slot {
        Slot::Cell(cell) => Some(cell.kind),
        Slot::Filler { kind, .. } => Some(*kind),
        Slot::Skip => None,
    }) {
        debug_assert_eq!(
            kind.is_header(),
            row.is_header,
            "row's TableRow::is_header disagrees with its own cells' TableCellKind::is_header"
        );
    }

    let mut html = format!("<tr{}>", styling_attr(&row.classes, mode));

    for slot in slots {
        match slot {
            Slot::Cell(cell) => html.push_str(&render_cell(cell, mode)),
            Slot::Filler { kind, classes } => {
                html.push_str(&render_filler(*kind, classes, mode));
            }
            Slot::Skip => {}
        }
    }

    html.push_str("</tr>\n");
    html
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::{TableCellKind, TableClass};

    fn cell(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> TableCell {
        TableCell::new(kind, top, bottom, left, right, String::new())
    }

    #[test]
    fn from_options_defaults_to_class_mode() {
        assert_eq!(HtmlWriter::new().css_mode, CssMode::Class);
        let writer = HtmlWriter::from_options(&WriterOptions::new()).unwrap();
        assert_eq!(writer.css_mode, CssMode::Class);
    }

    #[test]
    fn from_options_accepts_inline_and_class_case_insensitively() {
        let options = WriterOptions::parse(["css-mode=INLINE"]).unwrap();
        assert_eq!(
            HtmlWriter::from_options(&options).unwrap().css_mode,
            CssMode::Inline
        );
        let options = WriterOptions::parse(["css_mode=Class"]).unwrap();
        assert_eq!(
            HtmlWriter::from_options(&options).unwrap().css_mode,
            CssMode::Class
        );
    }

    #[test]
    fn from_options_rejects_bad_values_and_unknown_keys() {
        let err = HtmlWriter::from_options(&WriterOptions::parse(["css_mode=banana"]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("expects 'inline' or 'class'"), "{err}");
        assert!(HtmlWriter::from_options(&WriterOptions::parse(["widdth=3"]).unwrap()).is_err());
    }

    #[test]
    fn build_all_slots_returns_empty_for_no_cells() {
        assert!(build_all_slots(&[], 0, 0).is_empty());
    }

    #[test]
    fn render_row_returns_empty_string_for_no_slots() {
        assert_eq!(render_row(&[], &TableRow::default(), CssMode::Inline), "");
    }

    #[test]
    fn render_cell_emits_rowspan_when_height_is_greater_than_one() {
        let c = cell(TableCellKind::ColumnLabel, 0, 1, 0, 0);
        assert_eq!(render_cell(&c, CssMode::Inline), "<th rowspan=\"2\"></th>");
    }

    #[test]
    fn render_cell_emits_a_style_attribute_from_properties_in_inline_mode() {
        // Declaration content is inline_style's tests' job — this just checks
        // render_cell routes `hjust` through it as a `style="..."` attribute.
        let mut c = cell(TableCellKind::Body, 0, 0, 0, 0);
        c.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let html = render_cell(&c.discretise_hjust(), CssMode::Inline);
        assert!(html.starts_with("<td style=\"text-align: right;"));
        assert!(html.ends_with("\"></td>"));
    }

    #[test]
    fn render_cell_emits_classes_in_class_mode() {
        let mut body = cell(TableCellKind::Body, 0, 0, 0, 0).with_classes(vec![TableClass::Row]);
        body.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let body = body.discretise_hjust();
        assert_eq!(
            render_cell(&body, CssMode::Class),
            "<td class=\"ggsql_row ggsql_right\"></td>"
        );

        let label =
            cell(TableCellKind::ColumnLabel, 0, 0, 0, 0).with_classes(vec![TableClass::ColHeading]);
        assert_eq!(
            render_cell(&label, CssMode::Class),
            "<th class=\"ggsql_col_heading\"></th>"
        );

        let spanner =
            cell(TableCellKind::Spanner, 0, 0, 0, 1).with_classes(vec![TableClass::Spanner]);
        assert_eq!(
            render_cell(&spanner, CssMode::Class),
            "<th colspan=\"2\" class=\"ggsql_spanner\"></th>"
        );

        let outer =
            cell(TableCellKind::Spanner, 0, 0, 0, 1).with_classes(vec![TableClass::SpannerOuter]);
        assert_eq!(
            render_cell(&outer, CssMode::Class),
            "<th colspan=\"2\" class=\"ggsql_spanner_outer\"></th>"
        );
    }

    #[test]
    fn inline_style_folds_declarations_per_class() {
        assert_eq!(inline_style(&[]), None);
        assert_eq!(
            inline_style(&[TableClass::AlignLeft]),
            Some("text-align: left".to_string())
        );
        assert_eq!(
            inline_style(&[TableClass::AlignRight]),
            Some("text-align: right; font-variant-numeric: tabular-nums".to_string())
        );
        // Structural classes carry no declarations.
        assert_eq!(inline_style(&[TableClass::Row]), None);
    }

    #[test]
    fn inline_style_lets_a_later_class_override_an_earlier_ones_property() {
        // AlignLeft and AlignRight both declare text-align — last wins.
        assert_eq!(
            inline_style(&[TableClass::AlignLeft, TableClass::AlignRight]),
            Some("text-align: right; font-variant-numeric: tabular-nums".to_string())
        );
    }

    #[test]
    fn column_style_reads_a_string_width_as_a_css_width() {
        let mut properties = Parameters::new();
        properties.insert(
            "width".to_string(),
            ParameterValue::String("20%".to_string()),
        );
        assert_eq!(column_style(&properties), Some("width: 20%".to_string()));

        assert_eq!(column_style(&Parameters::new()), None);
    }

    fn column(properties: Parameters) -> TableColumn {
        TableColumn {
            name: String::new(),
            label: String::new(),
            properties,
        }
    }

    #[test]
    fn render_colgroup_returns_none_without_columns() {
        assert_eq!(render_colgroup(&[]), None);
    }

    #[test]
    fn render_colgroup_returns_none_when_no_column_has_a_width() {
        let columns = vec![column(Parameters::new()), column(Parameters::new())];
        assert_eq!(render_colgroup(&columns), None);
    }

    #[test]
    fn render_colgroup_emits_a_bare_col_for_a_column_with_no_width() {
        let mut widened = Parameters::new();
        widened.insert(
            "width".to_string(),
            ParameterValue::String("20%".to_string()),
        );
        let columns = vec![column(widened), column(Parameters::new())];

        assert_eq!(
            render_colgroup(&columns).unwrap(),
            "<colgroup>\n<col style=\"width: 20%\">\n<col>\n</colgroup>\n"
        );
    }

    #[test]
    fn build_all_slots_marks_a_rowspans_continuation_row_as_skip() {
        // Spans rows 0-1 at column 1 — row 1's column 1 is a continuation,
        // not a gap, even though nothing else in row 1 covers it.
        let cells = [
            cell(TableCellKind::ColumnLabel, 0, 1, 1, 1),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
            cell(TableCellKind::ColumnLabel, 1, 1, 2, 2),
        ];

        let slots = build_all_slots(&cells, 3, 2);

        let row1 = &slots[&1];
        assert_eq!(row1.len(), 3);
        assert!(matches!(&row1[0], Slot::Cell(cell) if cell.left == 0));
        assert!(matches!(&row1[1], Slot::Skip));
        assert!(matches!(&row1[2], Slot::Cell(cell) if cell.left == 2));
    }

    #[test]
    fn render_row_renders_nothing_for_a_skip_slot() {
        let a = cell(TableCellKind::ColumnLabel, 1, 1, 0, 0);
        let c = cell(TableCellKind::ColumnLabel, 1, 1, 2, 2);
        let slots = vec![Slot::Cell(&a), Slot::Skip, Slot::Cell(&c)];

        let row = render_row(&slots, &TableRow::header(), CssMode::Inline);

        assert_eq!(row, "<tr><th></th><th></th></tr>\n");
    }

    #[test]
    fn build_all_slots_fills_a_gap_with_the_row_siblings_kind_and_classes() {
        let cells = [
            cell(TableCellKind::Spanner, 0, 0, 0, 0).with_classes(vec![TableClass::SpannerOuter]),
            cell(TableCellKind::Spanner, 0, 0, 2, 2).with_classes(vec![TableClass::SpannerOuter]),
        ];

        let slots = build_all_slots(&cells, 3, 1);

        let row0 = &slots[&0];
        assert_eq!(row0.len(), 3);
        assert!(matches!(&row0[0], Slot::Cell(cell) if cell.left == 0));
        match &row0[1] {
            Slot::Filler { kind, classes } => {
                assert_eq!(*kind, TableCellKind::Spanner);
                assert_eq!(classes.to_vec(), vec![TableClass::SpannerOuter]);
            }
            _ => panic!("expected a Filler slot at column 1"),
        }
        assert!(matches!(&row0[2], Slot::Cell(cell) if cell.left == 2));
    }

    #[test]
    fn render_row_renders_a_filler_with_its_own_kind_and_classes() {
        let classes = [TableClass::SpannerOuter];
        let slots = vec![Slot::Filler {
            kind: TableCellKind::Spanner,
            classes: &classes,
        }];

        let row = render_row(&slots, &TableRow::header(), CssMode::Class);

        assert_eq!(row, "<tr><th class=\"ggsql_spanner_outer\"></th></tr>\n");
    }

    #[test]
    fn write_table_errors_when_a_header_row_is_not_above_every_body_row() {
        let cells = vec![
            cell(TableCellKind::Body, 0, 0, 0, 0),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
        ];
        let rows = vec![TableRow::default(), TableRow::header()];

        assert!(HtmlWriter::new().write_table(&cells, &[], &rows).is_err());
    }

    #[test]
    fn write_table_renders_a_rowspan_cell_once_and_skips_it_in_the_next_row() {
        // Column 1 has no spanner, so its label rowspans up through row 0
        // instead of row 0 getting a blank filler there.
        let cells = vec![
            cell(TableCellKind::Spanner, 0, 0, 0, 0),
            cell(TableCellKind::ColumnLabel, 0, 1, 1, 1),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
            cell(TableCellKind::Body, 2, 2, 0, 0),
            cell(TableCellKind::Body, 2, 2, 1, 1),
        ];
        let rows = vec![TableRow::header(), TableRow::header(), TableRow::default()];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&cells, &[], &rows)
        .unwrap();

        assert_eq!(
            html,
            "<table>\n\
             <thead>\n\
             <tr><th></th><th rowspan=\"2\"></th></tr>\n\
             <tr><th></th></tr>\n\
             </thead>\n\
             <tbody>\n\
             <tr><td></td><td></td></tr>\n\
             </tbody>\n\
             </table>"
        );
    }

    #[test]
    fn write_table_renders_a_caption_as_tables_first_child() {
        let cells = vec![
            cell(TableCellKind::ColumnLabel, 0, 0, 0, 0),
            cell(TableCellKind::Body, 1, 1, 0, 0),
            TableCell::new(
                TableCellKind::Caption,
                2,
                2,
                0,
                0,
                "<b>Source</b>".to_string(),
            ),
        ];
        let rows = vec![TableRow::header(), TableRow::default(), TableRow::default()];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&cells, &[], &rows)
        .unwrap();

        // <caption> is table's first child, before <thead>/<tbody>, and its
        // content is escaped like any other cell's.
        assert!(html.starts_with("<table>\n<caption>&lt;b&gt;Source&lt;/b&gt;</caption>\n"));
        assert!(html.find("<caption>").unwrap() < html.find("<thead>").unwrap());
        // Not part of the grid: no <th>/<td> for it, and it doesn't trip the
        // "header above body" check despite sitting below the body row.
        assert!(!html.contains("<th>&lt;b&gt;Source&lt;/b&gt;"));
        assert!(!html.contains("<td>&lt;b&gt;Source&lt;/b&gt;"));
    }

    #[test]
    fn write_table_renders_a_heading_rows_tr_class_from_the_rows_param() {
        let cells = vec![
            cell(TableCellKind::Title, 0, 0, 0, 0).with_classes(vec![TableClass::Title]),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
            cell(TableCellKind::Body, 2, 2, 0, 0),
        ];
        let rows = vec![
            TableRow {
                classes: vec![TableClass::Heading],
                ..TableRow::header()
            },
            TableRow::header(),
            TableRow::default(),
        ];

        let html = HtmlWriter::new().write_table(&cells, &[], &rows).unwrap();

        assert!(html.contains("<tr class=\"ggsql_heading\"><th class=\"ggsql_title\">"));
        // The column-label row has no recorded row classes, so its <tr> is
        // bare.
        assert!(html.contains("<tr><th></th></tr>"));
    }

    #[test]
    fn write_table_omits_an_inline_mode_heading_class_with_no_declarations() {
        // `Heading` carries no CSS declarations, so inline mode's <tr> stays
        // bare even though a row class was recorded.
        let cells = vec![
            cell(TableCellKind::Title, 0, 0, 0, 0).with_classes(vec![TableClass::Title]),
            cell(TableCellKind::Body, 1, 1, 0, 0),
        ];
        let rows = vec![
            TableRow {
                classes: vec![TableClass::Heading],
                ..TableRow::header()
            },
            TableRow::default(),
        ];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&cells, &[], &rows)
        .unwrap();

        assert!(html.contains("<tr><th></th></tr>"));
    }
}

#[cfg(test)]
#[cfg(feature = "duckdb")]
mod tests {
    use super::*;
    use crate::reader::{DuckDBReader, Reader};

    #[test]
    fn test_write_table_renders_rows_and_escapes_html() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES (1, '<b>a</b>'), (2, 'b')) AS t(id, name)",
            )
            .unwrap();
        let spec = reader.execute("TABULATE FROM sales").unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        // Precise per-dtype alignment is covered directly by
        // resolve_column_properties's/discretise_hjust's own tests — this
        // just checks the columns render and escaping works end to end.
        assert!(html.starts_with("<style>"));
        assert!(html.contains("<table>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        // Class mode: "id"/1 is numeric (right), "name" is text (left) —
        // alignment is a class on the cell, declarations in the <style>
        // block, nothing inline.
        assert!(html.contains("<td class=\"ggsql_row ggsql_right\">1</td>"));
        assert!(html.contains("ggsql_left"));
        assert!(html
            .contains(".ggsql_right { text-align: right; font-variant-numeric: tabular-nums; }"));
        assert!(!html.contains("style=\"text-align"));
        assert!(html.contains("&lt;b&gt;a&lt;/b&gt;"));
        assert!(!html.contains("<b>a</b>"));
    }

    #[test]
    fn test_write_table_inline_mode_keeps_declarations_on_the_cells() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader.execute("TABULATE FROM sales").unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=inline"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        assert!(html.starts_with("<table>"));
        assert!(!html.contains("<style>"));
        assert!(!html.contains("class="));
        assert!(html.contains("style=\"text-align: right")); // "id"/1, numeric
        assert!(html.contains("style=\"text-align: left")); // "name", text
    }

    #[test]
    fn test_write_table_renders_a_spanner_row_with_colspan() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a', 10)) AS t(id, name, amount)",
            )
            .unwrap();
        let spec = reader
            .execute("TABULATE FROM sales SPAN 'Info' ACROSS id, name")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        // "amount" has no spanner, so its label stretches up into the
        // spanner row (rowspan) instead of a blank filler cell there.
        // Alignment styling is incidental here (amount/id are numeric) and
        // covered precisely by resolve_column_properties's own tests — this
        // checks colspan/rowspan/ordering, not exact style content. The
        // single spanner level is the topmost one, hence `spanner_outer`.
        assert!(html.contains(
            "<tr><th colspan=\"2\" class=\"ggsql_spanner_outer\">Info</th><th rowspan=\"2\""
        ));
        assert!(html.contains(">amount</th>"));
        assert!(html.contains(">id</th>"));
        assert!(html.contains(">name</th>"));
        // The spanner row renders above the column-label row.
        assert!(html.find("Info").unwrap() < html.find(">id</th>").unwrap());
    }

    #[test]
    fn test_write_table_renders_a_colgroup_for_a_formats_width() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader
            .execute("TABULATE FROM sales FORMAT id SETTING width => '20%'")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        assert!(html.contains("<colgroup>"));
        assert!(html.contains("<col style=\"width: 20%\">"));
        // "name" has no FORMAT width, so its <col> stays bare.
        assert!(html.contains("<col>"));
        // <colgroup> comes before <thead>, per the HTML spec.
        assert!(html.find("<colgroup>").unwrap() < html.find("<thead>").unwrap());
    }

    #[test]
    fn test_write_table_omits_tbody_for_a_zero_row_result() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader
            .execute("SELECT * FROM sales WHERE 1 = 0 TABULATE")
            .unwrap();

        let writer = HtmlWriter::new();
        let html = writer.render(&spec).unwrap();

        assert!(html.contains("<thead>"));
        assert!(!html.contains("<tbody>"));
    }

    #[test]
    fn test_write_plot_is_unsupported() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
            .unwrap();

        let writer = HtmlWriter::new();
        assert!(writer.render(&spec).is_err());
    }
}
