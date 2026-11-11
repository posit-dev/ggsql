//! A minimal HTML table writer.
//!
//! Maps a `TableLayout`'s three parts onto pieces of the `<table>`:
//! - `cells` → `<thead>`/`<tbody>` rows. A `Caption`-kind cell is pulled out
//!   of the grid before row rendering and emitted as a `<caption>` (HTML
//!   requires it as `<table>`'s first child) — see `write_table`. Styling is
//!   expressed per the `css_mode` option: `inline` (the default) renders
//!   declarations into each element's `style` attribute; `class` puts
//!   `ggsql_*` classes on the elements, backed by a `<style>` block (the two
//!   gt `as_raw_html()` modes).
//! - `columns` → a `<colgroup>`, one `<col>` per column, with a `style`
//!   attribute from that column's resolved `width` (a bare `<col>` for a
//!   column with none) in both modes — targeted column styling stays inline,
//!   as gt keeps `tab_style()` rules inline.
//! - `rows` → a class on the `<tr>` itself (`Heading` for a `Title`/
//!   `Subtitle` row, `ColHeadingRow` for the column-label row).
//! - The `<table>`/`<tbody>` elements always carry `TableClass::Table`/
//!   `TableClass::TableBody` respectively.
//!
//! Spanner rows render as `colspan`, one `<tr>` per level, above the column
//! labels; a column label no spanner covers renders as a `rowspan` cell
//! reaching up into the spanner rows.

use std::collections::HashMap;

use crate::plot::{ParameterValue, Parameters};
use crate::table::Slot;
use crate::util::escape_html;
use crate::writer::{Writer, WriterOptions};
use crate::{
    DataFrame, GgsqlError, Plot, Result, TableCell, TableCellKind, TableClass, TableColumn,
    TableLayout, TableRow,
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
    /// Declarations rendered into each element's `style` attribute; no
    /// classes, no `<style>` block (gt's `inline_css = TRUE`).
    #[default]
    Inline,
    /// `ggsql_*` classes on the elements, backed by a `<style>` block
    /// carrying the style classes' rules (gt's `inline_css = FALSE`).
    Class,
}

impl HtmlWriter {
    /// Create a new HtmlWriter (inline mode; see `css_mode`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Writer for HtmlWriter {
    type Output = String;

    /// Takes one option: `css_mode` — `inline` (default) renders declarations
    /// into each element's `style` attribute; `class` renders `ggsql_*`
    /// classes backed by a `<style>` block.
    fn from_options(options: &WriterOptions) -> Result<Self> {
        options.reject_unknown(&["css_mode"])?;
        let css_mode = match options.one_of("css_mode", &["inline", "class"])? {
            Some("class") => CssMode::Class,
            _ => CssMode::Inline,
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

    fn write_table(&self, layout: &TableLayout) -> Result<String> {
        layout.validate()?;

        let columns = layout.columns();
        let rows = layout.rows();

        // HTML requires `<caption>` outside `<thead>`/`<tbody>`, as
        // `<table>`'s first child — rendered separately via
        // `render_caption`; `slots_by_row` keeps caption cells out of the
        // grid below.
        let caption_cells: Vec<TableCell> = layout
            .cells()
            .iter()
            .filter(|cell| cell.kind == TableCellKind::Caption)
            .cloned()
            .map(TableCell::discretise_hjust)
            .collect();

        let (header_rows, body_rows) = layout.header_and_body_rows();
        let mut all_slots = layout.slots_by_row();

        let mut html = String::new();
        if self.css_mode == CssMode::Class {
            html.push_str(&render_style_block());
        }
        let table_attrs = styling_attr(&[TableClass::Table], self.css_mode);
        // Opt out of Quarto's HTML table processing (the Bootstrap
        // `.table`/`.table-striped` classes it adds to a raw <table>):
        // `data-quarto-disable-processing` has to be `"true"` — "false"
        // means "processing is not disabled", i.e. go ahead.
        let quarto_opt_out =
            " data-quarto-disable-processing=\"true\" data-quarto-bootstrap=\"false\"";
        html.push_str(&format!("<table{quarto_opt_out}{table_attrs}>\n"));

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
                    .expect("a header row's own cell always puts it in slots_by_row's result");
                html.push_str(&render_row(&slots, &rows[*row], self.css_mode));
            }
            html.push_str("</thead>\n");
        }

        if !body_rows.is_empty() {
            let tbody_attrs = styling_attr(&[TableClass::TableBody], self.css_mode);
            // One <tbody> per facet group, so a scope="rowgroup" heading
            // covers exactly its group; group-less tables get a single tbody.
            let woven = body_rows
                .iter()
                .any(|&row| rows[row].classes.contains(&TableClass::GroupHeadingRow));

            for run in body_rows.chunk_by(|&a, &b| same_tbody(&rows[a], &rows[b], woven)) {
                html.push_str(&format!("<tbody{tbody_attrs}>\n"));
                for row in run {
                    let slots = all_slots
                        .remove(row)
                        .expect("a body row's own cell always puts it in slots_by_row's result");
                    html.push_str(&render_row(&slots, &rows[*row], self.css_mode));
                }
                html.push_str("</tbody>\n");
            }
        }

        html.push_str("</table>");

        Ok(html)
    }
}

/// Whether two consecutive body rows share a `<tbody>` — the `chunk_by`
/// predicate behind the one-tbody-per-group split.
/// * Woven groups (heading rows) split at leaf groups — a deeper heading keeps
///   the tbody open, so an outer heading joins its first leaf.
/// * Stub-column groups split at the outermost group: a rowspan may not cross
///   a tbody boundary, and the outer stub cell spans the whole group.
fn same_tbody(prev: &TableRow, row: &TableRow, woven: bool) -> bool {
    if prev.group.is_empty() || row.group.is_empty() {
        return true;
    }
    if woven {
        let deeper = row.group.len() > prev.group.len() && row.group.starts_with(&prev.group);
        row.group == prev.group || deeper
    } else {
        row.group.first() == prev.group.first()
    }
}

/// What a `TableClass` stands for in CSS, as property–value pairs. The
/// single source of truth both `css_mode`s render from — class mode puts
/// these in the `<style>` block, inline mode folds them into the element's
/// `style` attribute. Structural classes carry no declarations — they're
/// pure semantic hooks for the embedder.
fn class_declarations(class: TableClass) -> &'static [(&'static str, &'static str)] {
    match class {
        TableClass::Table => &[
            ("display", "table"),
            ("border-collapse", "collapse"),
            ("margin-left", "auto"),
            ("margin-right", "auto"),
            ("color", "#333333"),
            ("font-size", "16px"),
            ("font-weight", "normal"),
            ("background-color", "#FFFFFF"),
            ("width", "auto"),
            ("border-top-style", "solid"),
            ("border-top-width", "2px"),
            ("border-top-color", "#A8A8A8"),
            ("border-right-style", "none"),
            ("border-right-width", "2px"),
            ("border-right-color", "#D3D3D3"),
            ("border-bottom-style", "solid"),
            ("border-bottom-width", "2px"),
            ("border-bottom-color", "#A8A8A8"),
            ("border-left-style", "none"),
            ("border-left-width", "2px"),
            ("border-left-color", "#D3D3D3"),
        ],
        TableClass::TableBody => &[
            ("border-top-style", "solid"),
            ("border-top-width", "2px"),
            ("border-top-color", "#D3D3D3"),
            ("border-bottom-style", "solid"),
            ("border-bottom-width", "2px"),
            ("border-bottom-color", "#D3D3D3"),
        ],
        TableClass::Row => &[
            ("padding-top", "8px"),
            ("padding-bottom", "8px"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("margin", "10px"),
            ("border-top-style", "solid"),
            ("border-top-width", "1px"),
            ("border-top-color", "#D3D3D3"),
            ("border-left-style", "none"),
            ("border-left-width", "1px"),
            ("border-left-color", "#D3D3D3"),
            ("border-right-style", "none"),
            ("border-right-width", "1px"),
            ("border-right-color", "#D3D3D3"),
            ("vertical-align", "middle"),
            ("overflow-x", "hidden"),
        ],
        TableClass::Stub => &[
            ("color", "#333333"),
            ("background-color", "#FFFFFF"),
            ("font-size", "100%"),
            ("font-weight", "initial"),
            ("text-transform", "inherit"),
            ("border-right-style", "solid"),
            ("border-right-width", "2px"),
            ("border-right-color", "#D3D3D3"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
        ],
        TableClass::StubRowGroup => &[
            ("color", "#333333"),
            ("background-color", "#FFFFFF"),
            ("font-size", "100%"),
            ("font-weight", "initial"),
            ("text-transform", "inherit"),
            ("border-right-style", "solid"),
            ("border-right-width", "2px"),
            ("border-right-color", "#D3D3D3"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("vertical-align", "top"),
        ],
        TableClass::ColHeading => &[
            ("color", "#333333"),
            ("background-color", "#FFFFFF"),
            ("font-size", "100%"),
            ("font-weight", "normal"),
            ("text-transform", "inherit"),
            ("border-left-style", "none"),
            ("border-left-width", "1px"),
            ("border-left-color", "#D3D3D3"),
            ("border-right-style", "none"),
            ("border-right-width", "1px"),
            ("border-right-color", "#D3D3D3"),
            ("vertical-align", "bottom"),
            ("padding-top", "5px"),
            ("padding-bottom", "6px"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("overflow-x", "hidden"),
        ],
        TableClass::StubHead => &[],
        TableClass::SpannerRow => &[("border-bottom-style", "hidden")],
        TableClass::ColHeadingRow => &[
            ("border-top-style", "solid"),
            ("border-top-width", "2px"),
            ("border-top-color", "#D3D3D3"),
            ("border-bottom-style", "solid"),
            ("border-bottom-width", "2px"),
            ("border-bottom-color", "#D3D3D3"),
            ("border-left-style", "none"),
            ("border-left-width", "1px"),
            ("border-left-color", "#D3D3D3"),
            ("border-right-style", "none"),
            ("border-right-width", "1px"),
            ("border-right-color", "#D3D3D3"),
        ],
        TableClass::SpannerLabel => &[
            ("border-bottom-style", "solid"),
            ("border-bottom-width", "2px"),
            ("border-bottom-color", "#D3D3D3"),
            ("vertical-align", "bottom"),
            ("padding-top", "5px"),
            ("padding-bottom", "5px"),
            ("overflow-x", "hidden"),
            ("display", "inline-block"),
            ("width", "100%"),
        ],
        TableClass::Spanner => &[
            ("color", "#333333"),
            ("background-color", "#FFFFFF"),
            ("font-size", "100%"),
            ("font-weight", "normal"),
            ("text-transform", "inherit"),
            ("padding-top", "0"),
            ("padding-bottom", "0"),
            ("padding-left", "4px"),
            ("padding-right", "4px"),
        ],
        TableClass::Title => &[
            ("color", "#333333"),
            ("font-size", "125%"),
            ("font-weight", "initial"),
            ("padding-top", "4px"),
            ("padding-bottom", "4px"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("border-bottom-color", "#FFFFFF"),
            ("border-bottom-width", "0"),
        ],
        TableClass::Subtitle => &[
            ("color", "#333333"),
            ("font-size", "85%"),
            ("font-weight", "initial"),
            ("padding-top", "3px"),
            ("padding-bottom", "5px"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("border-top-color", "#FFFFFF"),
            ("border-top-width", "0"),
        ],
        TableClass::Caption => &[
            ("caption-side", "bottom"),
            ("padding-top", "4px"),
            ("padding-bottom", "4px"),
        ],
        TableClass::Heading => &[
            ("background-color", "#FFFFFF"),
            ("text-align", "center"),
            ("border-bottom-color", "#FFFFFF"),
            ("border-left-style", "none"),
            ("border-left-width", "1px"),
            ("border-left-color", "#D3D3D3"),
            ("border-right-style", "none"),
            ("border-right-width", "1px"),
            ("border-right-color", "#D3D3D3"),
        ],
        TableClass::GroupHeading => &[
            ("padding-top", "8px"),
            ("padding-bottom", "8px"),
            ("padding-left", "5px"),
            ("padding-right", "5px"),
            ("color", "#333333"),
            ("background-color", "#FFFFFF"),
            ("font-size", "100%"),
            ("font-weight", "initial"),
            ("border-top-style", "solid"),
            ("border-top-width", "2px"),
            ("border-top-color", "#D3D3D3"),
            ("border-bottom-style", "solid"),
            ("border-bottom-width", "2px"),
            ("border-bottom-color", "#D3D3D3"),
            ("border-left-style", "none"),
            ("border-left-width", "1px"),
            ("border-left-color", "#D3D3D3"),
            ("border-right-style", "none"),
            ("border-right-width", "1px"),
            ("border-right-color", "#D3D3D3"),
            ("vertical-align", "middle"),
            ("text-align", "left"),
        ],
        TableClass::GroupHeadingRow => &[],
        TableClass::RowGroupFirst => &[
            ("border-top-style", "solid"),
            ("border-top-width", "2px"),
            ("border-top-color", "#D3D3D3"),
        ],
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

/// Classes rendered on a `<div>` around the cell's content instead of on the
/// `<th>`/`<td>` itself (gt's `gt_column_spanner` pattern).
const INNER_CLASSES: &[TableClass] = &[TableClass::SpannerLabel];

/// The `<style>` block class mode prepends — one rule per styled class,
/// generated from the same lookup inline mode folds into `style` attributes,
/// so the modes can't drift. Emitted unconditionally, without checking
/// which classes the table actually uses.
fn render_style_block() -> String {
    let mut block = String::from("<style>\n");
    for &class in TableClass::PRECEDENCE {
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
/// doesn't rely on browsers' duplicate-declaration handling). Folding
/// happens in `TableClass::PRECEDENCE` order, not class-list order, so a
/// conflict resolves the same way it would in class mode and the modes
/// can't drift.
fn inline_style(classes: &[TableClass]) -> Option<String> {
    let mut ordered: Vec<&TableClass> = classes.iter().collect();
    // Unstyled classes declare nothing, so their fallback position is
    // irrelevant.
    ordered.sort_by_key(|c| {
        TableClass::PRECEDENCE
            .iter()
            .position(|s| s == *c)
            .unwrap_or(0)
    });
    let mut folded: Vec<(&str, &str)> = Vec::new();
    for class in ordered {
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

/// Render one `TableCell` as an HTML `<th>`/`<td>`, styled per `mode` (see
/// `styling_attr`). Not used for a `Caption`-kind cell — see `render_caption`.
fn render_cell(cell: &TableCell, mode: CssMode) -> String {
    // A Filler labels nothing, so it renders as a <td> even in the header;
    // a GroupHeading heads its rows even though it sits in the body.
    let tag = if (cell.is_header() && cell.kind != TableCellKind::Filler)
        || cell.kind == TableCellKind::GroupHeading
    {
        "th"
    } else {
        "td"
    };
    let colspan = cell.width();
    let rowspan = cell.height();
    let mut attrs = String::new();
    if colspan > 1 {
        attrs.push_str(&format!(" colspan=\"{colspan}\""));
    }
    if rowspan > 1 {
        attrs.push_str(&format!(" rowspan=\"{rowspan}\""));
    }
    if let Some(scope) = header_scope(cell.kind) {
        attrs.push_str(&format!(" scope=\"{scope}\""));
    }
    // `slots_by_row`'s slots borrow the layout's cells, so `hjust` can't be
    // folded into `classes` up front — its class joins the fold here.
    // INNER_CLASSES go on a <div> around the content, not on the cell itself.
    let (inner, outer): (Vec<TableClass>, Vec<TableClass>) = cell
        .classes
        .iter()
        .copied()
        .chain(cell.hjust_class())
        .partition(|class| INNER_CLASSES.contains(class));
    attrs.push_str(&styling_attr(&outer, mode));
    if inner.is_empty() {
        format!("<{tag}{attrs}>{}</{tag}>", escape_html(&cell.content))
    } else {
        let inner_attrs = styling_attr(&inner, mode);
        format!(
            "<{tag}{attrs}><div{inner_attrs}>{}</div></{tag}>",
            escape_html(&cell.content)
        )
    }
}

/// `scope` value for a header cell of `kind`, or `None` if `scope` doesn't
/// apply. `ColumnLabel`/`StubHead` are always single-column (`"col"`); a
/// `Spanner` is a column-group header even when its group has one column
/// (`"colgroup"`); `StubRowLabel` heads its row (`"row"`); a `GroupHeading`
/// heads its facet group (`"rowgroup"`); everything else gets none.
fn header_scope(kind: TableCellKind) -> Option<&'static str> {
    match kind {
        TableCellKind::ColumnLabel | TableCellKind::StubHead => Some("col"),
        TableCellKind::Spanner => Some("colgroup"),
        TableCellKind::StubRowLabel => Some("row"),
        TableCellKind::GroupHeading => Some("rowgroup"),
        TableCellKind::Body
        | TableCellKind::Title
        | TableCellKind::Subtitle
        | TableCellKind::Caption
        | TableCellKind::Filler => None,
    }
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
/// is already validated (by `ParamConstraint::number_with_unit`) to be a
/// number immediately followed by `px` or `%`, which is exactly CSS's own
/// `width` syntax, so it's used as-is.
fn column_style(properties: &Parameters) -> Option<String> {
    match properties.get("width") {
        Some(ParameterValue::String(width)) => Some(format!("width: {width}")),
        _ => None,
    }
}

/// Render one row's already-decided `Slot`s as `<tr>...</tr>`. A `Skip`
/// slot renders nothing at all; a `Cell` slot renders through `render_cell`.
/// Styling on the `<tr>` itself comes from `row`'s recorded classes, same as
/// any other element, via `styling_attr`.
fn render_row(slots: &[Slot], row: &TableRow, mode: CssMode) -> String {
    // Nothing to render if every slot is a Skip continuation (vacuously
    // true for an empty `slots` too).
    if slots.iter().all(|slot| matches!(slot, Slot::Skip)) {
        return String::new();
    }

    let mut html = format!("<tr{}>", styling_attr(&row.classes, mode));

    for slot in slots {
        match slot {
            Slot::Cell(cell) => html.push_str(&render_cell(cell, mode)),
            Slot::Skip => {}
        }
    }

    html.push_str("</tr>\n");
    html
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::{ColumnSection, TableCellKind, TableClass};

    fn cell(
        kind: TableCellKind,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> TableCell {
        TableCell::new(kind, top, bottom, left, right, String::new())
    }

    /// Wrap hand-built `cells`/`rows` into a `TableLayout` for `write_table`
    /// tests, with as many default columns as `cells` implies.
    fn layout(cells: Vec<TableCell>, rows: Vec<TableRow>) -> TableLayout {
        let ncol = cells
            .iter()
            .map(|cell| cell.right)
            .max()
            .map_or(0, |r| r + 1);
        let columns = (0..ncol)
            .map(|i| TableColumn {
                name: i.to_string(),
                label: String::new(),
                properties: Parameters::new(),
                target: ColumnSection::Body,
            })
            .collect();
        TableLayout::new(columns, rows, cells)
    }

    #[test]
    fn from_options_defaults_to_inline_mode() {
        assert_eq!(HtmlWriter::new().css_mode, CssMode::Inline);
        let writer = HtmlWriter::from_options(&WriterOptions::new()).unwrap();
        assert_eq!(writer.css_mode, CssMode::Inline);
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
    fn render_row_returns_empty_string_for_no_slots() {
        assert_eq!(render_row(&[], &TableRow::body(), CssMode::Inline), "");
    }

    #[test]
    fn render_row_returns_empty_string_when_every_slot_is_a_skip() {
        // A row entirely swallowed by rowspans from earlier rows gets no
        // `<tr>` of its own either.
        let slots = vec![Slot::Skip, Slot::Skip];
        assert_eq!(render_row(&slots, &TableRow::body(), CssMode::Inline), "");
    }

    #[test]
    fn render_cell_emits_rowspan_when_height_is_greater_than_one() {
        let c = cell(TableCellKind::ColumnLabel, 0, 1, 0, 0);
        assert_eq!(
            render_cell(&c, CssMode::Inline),
            "<th rowspan=\"2\" scope=\"col\"></th>"
        );
    }

    #[test]
    fn render_cell_emits_a_group_heading_as_a_rowgroup_th() {
        let c = cell(TableCellKind::GroupHeading, 2, 3, 0, 0);
        assert_eq!(
            render_cell(&c, CssMode::Inline),
            "<th rowspan=\"2\" scope=\"rowgroup\"></th>"
        );
    }

    #[test]
    fn render_cell_emits_a_filler_as_td_even_in_the_header() {
        // A filler labels nothing, so it must not announce as a header cell.
        let c = cell(TableCellKind::Filler, 0, 1, 0, 0);
        assert_eq!(render_cell(&c, CssMode::Inline), "<td rowspan=\"2\"></td>");
    }

    #[test]
    fn render_cell_emits_a_single_column_spanner_with_colgroup_scope() {
        // A spanner is a column-group header even when the group has one
        // column.
        let c = cell(TableCellKind::Spanner, 0, 0, 0, 0);
        assert_eq!(
            render_cell(&c, CssMode::Inline),
            "<th scope=\"colgroup\"></th>"
        );
    }

    #[test]
    fn render_cell_emits_a_style_attribute_from_properties_in_inline_mode() {
        // Declaration content is inline_style's tests' job — this just checks
        // render_cell routes `hjust` through it as a `style="..."` attribute.
        let mut c = cell(TableCellKind::Body, 0, 0, 0, 0);
        c.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        let html = render_cell(&c, CssMode::Inline);
        assert!(html.starts_with("<td style=\"text-align: right;"));
        assert!(html.ends_with("\"></td>"));
    }

    #[test]
    fn render_cell_emits_classes_in_class_mode() {
        let mut body = cell(TableCellKind::Body, 0, 0, 0, 0).with_classes(vec![TableClass::Row]);
        body.properties
            .insert("hjust".to_string(), ParameterValue::Number(1.0));
        assert_eq!(
            render_cell(&body, CssMode::Class),
            "<td class=\"ggsql_row ggsql_right\"></td>"
        );

        let label =
            cell(TableCellKind::ColumnLabel, 0, 0, 0, 0).with_classes(vec![TableClass::ColHeading]);
        assert_eq!(
            render_cell(&label, CssMode::Class),
            "<th scope=\"col\" class=\"ggsql_col_heading\"></th>"
        );

        let spanner =
            cell(TableCellKind::Spanner, 0, 0, 0, 1).with_classes(vec![TableClass::SpannerLabel]);
        assert_eq!(
            render_cell(&spanner, CssMode::Class),
            "<th colspan=\"2\" scope=\"colgroup\"><div class=\"ggsql_spanner_label\"></div></th>"
        );

        let outer =
            cell(TableCellKind::Spanner, 0, 0, 0, 1).with_classes(vec![TableClass::Spanner]);
        assert_eq!(
            render_cell(&outer, CssMode::Class),
            "<th colspan=\"2\" scope=\"colgroup\" class=\"ggsql_spanner\"></th>"
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
        // `StubHead` is the one structural class with no gt equivalent and
        // so no declarations.
        assert_eq!(inline_style(&[TableClass::StubHead]), None);
    }

    #[test]
    fn inline_style_resolves_conflicts_by_precedence_regardless_of_class_order() {
        // AlignLeft and AlignRight both declare text-align; AlignRight comes
        // later in TableClass::PRECEDENCE, whichever order the classes
        // arrive in.
        let expected = Some("text-align: right; font-variant-numeric: tabular-nums".to_string());
        assert_eq!(
            inline_style(&[TableClass::AlignLeft, TableClass::AlignRight]),
            expected
        );
        assert_eq!(
            inline_style(&[TableClass::AlignRight, TableClass::AlignLeft]),
            expected
        );
    }

    #[test]
    fn inline_style_is_order_independent_for_every_class_pair() {
        // Class mode resolves conflicts by stylesheet (PRECEDENCE) order, so
        // inline folding must give the same result for any permutation of a
        // cell's classes — otherwise the modes drift.
        let all: Vec<TableClass> = TableClass::PRECEDENCE
            .iter()
            .copied()
            .chain([TableClass::StubHead])
            .collect();
        for &a in &all {
            for &b in &all {
                assert_eq!(
                    inline_style(&[a, b]),
                    inline_style(&[b, a]),
                    "{a} vs {b} folds differently under permutation"
                );
            }
        }
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
            target: ColumnSection::Body,
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
    fn render_row_renders_nothing_for_a_skip_slot() {
        let a = cell(TableCellKind::ColumnLabel, 1, 1, 0, 0);
        let c = cell(TableCellKind::ColumnLabel, 1, 1, 2, 2);
        let slots = vec![Slot::Cell(&a), Slot::Skip, Slot::Cell(&c)];

        let row = render_row(&slots, &TableRow::header(), CssMode::Inline);

        assert_eq!(
            row,
            "<tr><th scope=\"col\"></th><th scope=\"col\"></th></tr>\n"
        );
    }

    #[test]
    fn write_table_errors_when_a_header_row_is_not_above_every_body_row() {
        let cells = vec![
            cell(TableCellKind::Body, 0, 0, 0, 0),
            cell(TableCellKind::ColumnLabel, 1, 1, 0, 0),
        ];
        let rows = vec![TableRow::body(), TableRow::header()];

        assert!(HtmlWriter::new().write_table(&layout(cells, rows)).is_err());
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
        let rows = vec![TableRow::header(), TableRow::header(), TableRow::body()];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&layout(cells, rows))
        .unwrap();

        // The <table>/<tbody> tags' own attributes are `TableClass::Table`/
        // `TableClass::TableBody`'s concern, not this test's — split both off
        // and check only that each is the tag it claims to be, then
        // assert_eq! the rest verbatim.
        let (table_tag, rest) = html.split_once('\n').unwrap();
        assert!(table_tag.starts_with("<table") && table_tag.ends_with('>'));
        let (before_tbody, rest) = rest.split_once("<tbody").unwrap();
        let (tbody_tag, rest) = rest.split_once('\n').unwrap();
        assert!(tbody_tag.ends_with('>'));
        assert_eq!(
            before_tbody,
            "<thead>\n\
             <tr><th scope=\"colgroup\"></th><th rowspan=\"2\" scope=\"col\"></th></tr>\n\
             <tr><th scope=\"col\"></th></tr>\n\
             </thead>\n"
        );
        assert_eq!(
            rest,
            "<tr><td></td><td></td></tr>\n\
             </tbody>\n\
             </table>"
        );
    }

    #[test]
    fn write_table_splits_tbody_per_leaf_group_when_groups_are_woven() {
        // Two leaf groups under one outer group: the outer heading joins its
        // first leaf's tbody rather than sitting alone.
        let cells = vec![
            cell(TableCellKind::GroupHeading, 0, 0, 0, 0),
            cell(TableCellKind::GroupHeading, 1, 1, 0, 0),
            cell(TableCellKind::Body, 2, 2, 0, 0),
            cell(TableCellKind::GroupHeading, 3, 3, 0, 0),
            cell(TableCellKind::Body, 4, 4, 0, 0),
        ];
        let rows = vec![
            TableRow::group_heading(vec![0]),
            TableRow::group_heading(vec![0, 0]),
            TableRow {
                group: vec![0, 0],
                ..TableRow::body()
            },
            TableRow::group_heading(vec![0, 1]),
            TableRow {
                group: vec![0, 1],
                ..TableRow::body()
            },
        ];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&layout(cells, rows))
        .unwrap();

        assert_eq!(html.matches("<tbody").count(), 2);
        let first_tbody = html
            .split_once("<tbody")
            .unwrap()
            .1
            .split_once("</tbody>")
            .unwrap()
            .0;
        assert_eq!(first_tbody.matches("<tr>").count(), 3);
    }

    #[test]
    fn write_table_splits_tbody_per_outermost_group_for_stub_columns() {
        // No GroupHeadingRow rows: group labels live in stub columns. Split
        // at the outermost group only, so a rowspanned stub cell never
        // crosses a tbody boundary.
        let cells = vec![
            cell(TableCellKind::GroupHeading, 0, 1, 0, 0),
            cell(TableCellKind::Body, 0, 0, 1, 1),
            cell(TableCellKind::Body, 1, 1, 1, 1),
            cell(TableCellKind::GroupHeading, 2, 2, 0, 0),
            cell(TableCellKind::Body, 2, 2, 1, 1),
        ];
        let rows = vec![
            TableRow {
                group: vec![0, 0],
                ..TableRow::body()
            },
            TableRow {
                group: vec![0, 1],
                ..TableRow::body()
            },
            TableRow {
                group: vec![1, 0],
                ..TableRow::body()
            },
        ];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&layout(cells, rows))
        .unwrap();

        assert_eq!(html.matches("<tbody").count(), 2);
        let first_tbody = html
            .split_once("<tbody")
            .unwrap()
            .1
            .split_once("</tbody>")
            .unwrap()
            .0;
        assert_eq!(first_tbody.matches("<tr>").count(), 2);
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
        let rows = vec![TableRow::header(), TableRow::body(), TableRow::caption()];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&layout(cells, rows))
        .unwrap();

        // <caption> is table's first child, before <thead>/<tbody>, and its
        // content is escaped like any other cell's. The <table> tag's own
        // attributes are `TableClass::Table`'s concern, not this test's.
        let (table_tag, rest) = html.split_once('\n').unwrap();
        assert!(table_tag.starts_with("<table") && table_tag.ends_with('>'));
        assert!(rest.starts_with("<caption>&lt;b&gt;Source&lt;/b&gt;</caption>\n"));
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
            TableRow::body(),
        ];

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=class"]).unwrap()).unwrap();
        let html = writer.write_table(&layout(cells, rows)).unwrap();

        assert!(html.contains("<tr class=\"ggsql_heading\"><th class=\"ggsql_title\">"));
        // The column-label row has no recorded row classes, so its <tr> is
        // bare.
        assert!(html.contains("<tr><th scope=\"col\"></th></tr>"));
    }

    #[test]
    fn write_table_renders_an_inline_mode_heading_row_style_from_row_classes() {
        // `Heading` carries real declarations, so inline mode's <tr> gets a
        // style attribute from the recorded row class.
        let cells = vec![
            cell(TableCellKind::Title, 0, 0, 0, 0).with_classes(vec![TableClass::Title]),
            cell(TableCellKind::Body, 1, 1, 0, 0),
        ];
        let rows = vec![
            TableRow {
                classes: vec![TableClass::Heading],
                ..TableRow::header()
            },
            TableRow::body(),
        ];

        let html = HtmlWriter {
            css_mode: CssMode::Inline,
        }
        .write_table(&layout(cells, rows))
        .unwrap();

        assert!(html.contains("<tr style=\"background-color: #FFFFFF"));
        // The body row has no recorded row classes, so its <tr> stays bare.
        assert!(html.contains("<tr><td></td></tr>"));
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
        let spec = reader.execute("TABULATE * FROM sales").unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=class"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        // Precise per-dtype alignment is covered directly by
        // resolve_column_properties's/discretise_hjust's own tests — this
        // just checks the columns render and escaping works end to end.
        assert!(html.starts_with("<style>"));
        assert!(html.contains("data-quarto-bootstrap=\"false\" class=\"ggsql_table\">"));
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
        let spec = reader.execute("TABULATE * FROM sales").unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=inline"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        assert!(html.starts_with("<table"));
        assert!(!html.contains("<style>"));
        assert!(!html.contains("class="));
        // "id"/1 is numeric (right), "name" is text (left) — both cells also
        // carry `Row`'s own declarations, folded in ahead of alignment's.
        assert!(html.contains("text-align: right"));
        assert!(html.contains("text-align: left"));
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
            .execute("TABULATE * FROM sales SPAN Info OVER id, name")
            .unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=class"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        // "amount" has no spanner, so its label stretches up into the
        // spanner row (rowspan) instead of a blank filler cell there.
        // Alignment styling is incidental here (amount/id are numeric) and
        // covered precisely by resolve_column_properties's own tests — this
        // checks colspan/rowspan/ordering, not exact style content. The
        // spanner cell carries both spanner classes, and it's centred by
        // default.
        assert!(html.contains(
            "<tr class=\"ggsql_col_heading_row ggsql_spanner_row\"><th colspan=\"2\" scope=\"colgroup\" class=\"ggsql_spanner ggsql_center\"><div class=\"ggsql_spanner_label\">Info</div></th><th rowspan=\"2\" scope=\"col\""
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
            .execute("TABULATE * FROM sales FORMAT id SETTING width => '20%'")
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
    fn test_write_table_renders_a_stub_head_and_row_labels() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE sales AS SELECT * FROM (VALUES ('north', 1), ('south', 2)) AS t(region, amount)",
            )
            .unwrap();
        let spec = reader
            .execute("TABULATE * FROM sales FORMAT STUB region")
            .unwrap();

        let writer =
            HtmlWriter::from_options(&WriterOptions::parse(["css_mode=class"]).unwrap()).unwrap();
        let html = writer.render(&spec).unwrap();

        // The stub head renders as a <th scope="col">, blank by default (no
        // LABEL given); the row labels render as <th scope="row">, both
        // right-aligned (resolve_column_properties' STUB default).
        assert!(html.contains("<th scope=\"col\" class=\"ggsql_stub_head ggsql_right\"></th>"));
        assert!(html.contains("<th scope=\"row\" class=\"ggsql_stub ggsql_right\">north</th>"));
        assert!(html.contains("<td class=\"ggsql_row ggsql_right\">1</td>")); // "amount" unaffected
        assert!(html.find("ggsql_stub_head").unwrap() < html.find(">amount</th>").unwrap());
    }

    #[test]
    fn test_write_table_omits_tbody_for_a_zero_row_result() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'a')) AS t(id, name)")
            .unwrap();
        let spec = reader
            .execute("SELECT * FROM sales WHERE 1 = 0 TABULATE *")
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
