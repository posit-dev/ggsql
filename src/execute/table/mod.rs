//! Table resolution: turns a TABULATE query + Reader into a ResolvedTable.
//!
//! A Table has no layers, so there's no per-layer CTE materialization, scale
//! resolution, or facet handling here — just the one data query, plus
//! resolving that data into positioned `TableCell`s. `FORMAT STUB` lives in
//! `stub`, `FORMAT` values and settings in `format`, and cell/row grid
//! construction in `layout`; the `TableLayout` assembly and `SPAN` resolution
//! it composes live in `crate::table`.

mod facet;
mod format;
mod layout;
mod stub;

use format::{apply_formats, setup_formats};
use layout::{build_cells, setup_columns, setup_rows};

use crate::table::extract_table_titles;

use super::cte::split_with_query;
use crate::naming;
use crate::parser::{self, SourceTree};
use crate::reader::SqlDialect;
use crate::reader::{Reader, ResolvedTable};
use crate::sql;
use crate::validate::{validate, ValidationWarning};
use crate::{DataFrame, GgsqlError, Result, SelectionItem, Spec};

/// Resolve a TABULATE query into a `ResolvedTable`.
///
/// Takes the *first* `Table` spec found in the query — several TABULATE
/// statements, or a mix of VISUALISE and TABULATE, aren't disambiguated
/// further. Setup (INSTALL, LOAD, SET) and side-effect DML (CREATE, INSERT,
/// UPDATE, DELETE) ahead of a TABULATE run here.
pub fn resolve_table_with_reader(query: &str, reader: &dyn Reader) -> Result<ResolvedTable> {
    let validated = validate(query)?;
    let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

    let source_tree = SourceTree::new(query)?;
    source_tree.validate()?;

    let table = parser::build_ast(&source_tree)?
        .into_iter()
        .find_map(Spec::into_table)
        .ok_or_else(|| GgsqlError::ValidationError("No table specification found".to_string()))?;

    let facet = crate::table::setup_facet(&table.facet).map_err(GgsqlError::ValidationError)?;
    let mut labels = table.labels.clone();
    let titles = extract_table_titles(&mut labels);

    super::execute_setup_statements(&source_tree, reader)?;

    // Run structured DML (CREATE, INSERT, UPDATE, DELETE) so any table it
    // creates exists before the TABULATE query reads it.
    for stmt in super::cte::extract_side_effects(&source_tree) {
        reader.execute_sql(&stmt)?;
    }

    let sql =
        build_table_sql(&source_tree, &table.selection, reader.dialect()).ok_or_else(|| {
            GgsqlError::ValidationError(
                "TABULATE has no data source: add a FROM, or a SQL query before it".to_string(),
            )
        })?;

    let df = resolve_selection(reader.execute_sql(&sql)?, &table.selection)?;

    let (df, rows) = setup_rows(df, facet.as_ref())?;

    let column_names = df.get_column_names();
    let spans = table
        .resolve_spanners(&labels, Some(&column_names))
        .map_err(GgsqlError::ValidationError)?;
    // The shape both create_table_columns (SETTING) and apply_formats
    // (RENAMING) read from. Resolved after spans so a FORMAT column entry
    // can name a SPAN id in place of the columns it covers.
    let formats = setup_formats(&df, &table.formats, &spans)?;
    let columns = setup_columns(&df, &spans, &labels, &formats, facet.as_ref());
    // Only FACET removes columns today; if that changes, revisit this error.
    if columns.is_empty() {
        return Err(GgsqlError::ValidationError(
            "TABULATE FACET covers every column: no body columns left to show".to_string(),
        ));
    }
    let df = apply_formats(&df, &formats)?;
    let layout = build_cells(&df, columns, rows, &spans, &titles, facet.as_ref())?;

    Ok(ResolvedTable::new(layout, sql, warnings))
}

/// Resolves a mixed selection like `TABULATE foo, bar AS Baz, *`: named
/// columns are kept only at the position they were written (a rename
/// included), the wildcard contributes the rest of the source columns.
/// Selections without both a wildcard and a named column pass through
/// unchanged. Duplicate output column names are an error.
fn resolve_selection(df: DataFrame, selection: &[SelectionItem]) -> Result<DataFrame> {
    let n_named_items = selection
        .iter()
        .filter(|item| matches!(item, SelectionItem::Column { .. }))
        .count();
    let n_wildcards = selection.len() - n_named_items;

    let df = if n_named_items == 0 || n_wildcards == 0 {
        df
    } else {
        let names = df.get_column_names();
        let wildcard_width = names
            .len()
            .checked_sub(n_named_items)
            .filter(|remaining| remaining % n_wildcards == 0)
            .map(|remaining| remaining / n_wildcards)
            .ok_or_else(|| {
                GgsqlError::ValidationError(
                    "TABULATE selection resolved an unexpected number of SQL columns".to_string(),
                )
            })?;

        // A column item — bare or renamed — already surfaces its source
        // column once, at its written position; drop the wildcard's copy of
        // that same source column so a rename doesn't show it twice. A
        // constant column (`'x' AS label`, no source) has nothing to drop.
        let claimed_sources: std::collections::HashSet<&str> = selection
            .iter()
            .filter_map(|item| match item {
                SelectionItem::Column {
                    source: Some(source),
                    ..
                } => Some(source.as_str()),
                _ => None,
            })
            .collect();

        let mut columns: Vec<(String, arrow::array::ArrayRef)> = Vec::new();
        let mut cursor = 0usize;
        for item in selection {
            match item {
                SelectionItem::Column { name, .. } => {
                    columns.push((name.clone(), df.inner().column(cursor).clone()));
                    cursor += 1;
                }
                SelectionItem::Wildcard => {
                    for (idx, name) in names.iter().enumerate().skip(cursor).take(wildcard_width) {
                        if !claimed_sources.contains(name.as_str()) {
                            columns.push((name.clone(), df.inner().column(idx).clone()));
                        }
                    }
                    cursor += wildcard_width;
                }
            }
        }
        DataFrame::new(columns)?
    };

    // Downstream steps look columns up by name, so a duplicate output name
    // would silently shadow rather than error.
    let mut seen = std::collections::HashSet::new();
    if let Some(dup) = df
        .get_column_names()
        .into_iter()
        .find(|name| !seen.insert(name.clone()))
    {
        return Err(GgsqlError::ValidationError(format!(
            "TABULATE selection produces the column '{dup}' more than once: give each occurrence a distinct name"
        )));
    }

    Ok(df)
}

/// The selection as a SQL select list (e.g. `foo, bar AS Baz, *`).
fn selection_sql(selection: &[SelectionItem]) -> String {
    selection
        .iter()
        .map(|item| match item {
            SelectionItem::Wildcard => "*",
            SelectionItem::Column { sql, .. } => sql,
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Builds the SQL a `TABULATE` query executes, folding `selection` in as
/// the outer projection. Table-side counterpart to
/// `execute::cte::transform_global_sql`, without CTE-rewriting or
/// cache-staging.
fn build_table_sql(
    source_tree: &SourceTree,
    selection: &[SelectionItem],
    dialect: &dyn SqlDialect,
) -> Option<String> {
    let wildcard_only = matches!(selection, [SelectionItem::Wildcard]);
    let root = source_tree.root();

    // A WITH...SELECT tail, or a plain trailing SELECT.
    let select_sql = split_with_query(source_tree)
        .map(|(_, select)| select)
        .or_else(|| source_tree.find_text(&root, "(sql_statement (select_statement) @select)"));

    if let Some(select_sql) = select_sql {
        return Some(if wildcard_only {
            select_sql
        } else {
            wrap_selection(dialect, selection, &select_sql)
        });
    }

    // No trailing SELECT: fall back to TABULATE FROM <source>.
    let first_stmt = source_tree.first_stmt(&root)?;
    let from_source = source_tree.find_text(
        &first_stmt,
        r#"(tabulate_statement (single_source_from source: (_) @source))"#,
    );

    if let Some(source) = from_source {
        return Some(format!("SELECT {} FROM {source}", selection_sql(selection)));
    }

    // Neither: e.g. a bare DuckDB-style `FROM t`. The text may carry a
    // leading setup-statement prefix (INSTALL/LOAD/SET), which a non-"*"
    // selection then wraps into an invalid subquery — a known gap.
    let fallback = source_tree.extract_sql()?;
    Some(if wildcard_only {
        fallback
    } else {
        wrap_selection(dialect, selection, &fallback)
    })
}

/// `SELECT <selection> FROM (<query>)`, aliased as the dialect requires and
/// with a leading WITH hoisted out where the dialect forbids CTEs in a
/// derived table.
fn wrap_selection(dialect: &dyn SqlDialect, selection: &[SelectionItem], query: &str) -> String {
    sql::select_from(
        dialect,
        &selection_sql(selection),
        sql::FromItem::Query(query),
        naming::TABLE_ALIAS,
    )
}

#[cfg(test)]
#[cfg(feature = "duckdb")]
mod integration_tests {
    use super::*;
    use crate::reader::DuckDBReader;
    use crate::table::{TableCellKind, TableClass};

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
    fn test_tabulate_selection_wildcard_with_rename_does_not_duplicate_the_source_column() {
        let reader = reader_with_sales();

        // sales is (id, name): `id AS Number` renames `id`, so the
        // wildcard's own copy of `id` is dropped rather than shown twice.
        let trailing =
            resolve_table_with_reader("TABULATE *, id AS Number FROM sales", &reader).unwrap();
        assert_eq!(trailing.sql(), "SELECT *, id AS Number FROM sales");
        assert_eq!(trailing.ncol(), 2);
        let order: Vec<_> = trailing.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(order, ["name", "Number"]);

        let leading =
            resolve_table_with_reader("TABULATE id AS Number, * FROM sales", &reader).unwrap();
        assert_eq!(leading.ncol(), 2);
        let order: Vec<_> = leading.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(order, ["Number", "name"]);
    }

    #[test]
    fn test_tabulate_selection_wildcard_dedupes_named_columns() {
        let reader = reader_with_sales();

        let resolved = resolve_table_with_reader("TABULATE name, * FROM sales", &reader).unwrap();
        assert_eq!(resolved.sql(), "SELECT name, * FROM sales");
        // sales is (id, name): the wildcard's second `name` is dropped.
        assert_eq!(resolved.ncol(), 2);

        let trailing = resolve_table_with_reader("TABULATE *, name FROM sales", &reader).unwrap();
        assert_eq!(trailing.ncol(), 2);
        // The named column keeps its written position.
        let order: Vec<_> = trailing.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(order, ["id", "name"]);
    }

    #[test]
    fn test_tabulate_selection_duplicate_alias_without_wildcard_errors() {
        let reader = reader_with_sales();
        match resolve_table_with_reader("TABULATE id AS x, name AS x FROM sales", &reader) {
            Ok(_) => panic!("expected a duplicate output column to error"),
            Err(GgsqlError::ValidationError(msg)) => assert!(msg.contains("'x'")),
            Err(e) => panic!("expected a ValidationError, got {e:?}"),
        }
    }

    #[test]
    fn test_tabulate_selection_rename_colliding_with_wildcard_column_name_errors() {
        let reader = reader_with_sales();
        match resolve_table_with_reader("TABULATE id AS name, * FROM sales", &reader) {
            Ok(_) => panic!("expected a duplicate output column to error"),
            Err(GgsqlError::ValidationError(msg)) => assert!(msg.contains("'name'")),
            Err(e) => panic!("expected a ValidationError, got {e:?}"),
        }
    }

    #[test]
    fn test_tabulate_from_table_created_by_preceding_create() {
        let reader = reader_with_sales();
        let resolved = resolve_table_with_reader(
            "CREATE TEMP TABLE my_data AS SELECT * FROM sales LIMIT 2 TABULATE * FROM my_data",
            &reader,
        )
        .unwrap();
        // 2 data rows + 1 column-label row.
        assert_eq!(resolved.nrow(), 3);
    }

    #[test]
    fn test_tabulate_selection_with_no_tabulate_from() {
        // StubReader with the ANSI dialect pins the aliasing assertion to a
        // known dialect rather than whatever DuckDB happens to use.
        let (reader, _log) =
            crate::reader::test_support::StubReader::new(&crate::reader::AnsiDialect);

        let no_from =
            resolve_table_with_reader("SELECT * FROM sales TABULATE name, id AS Number", &reader)
                .unwrap();
        assert_eq!(
            no_from.sql(),
            "SELECT name, id AS Number FROM (SELECT * FROM sales) AS \"__table__\""
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
            "TABULATE * FROM sales FORMAT STUB name SPAN G OVER name, id",
            &reader,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_facet_interleaves_group_heading_rows() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE t AS SELECT * FROM (VALUES \
                 ('north', 'q1', 1), ('north', 'q2', 2), ('south', 'q1', 3)) \
                 AS t(region, quarter, sales)",
            )
            .unwrap();

        let resolved =
            resolve_table_with_reader("TABULATE * FROM t FACET region", &reader).unwrap();

        // FACET consumes its column: only quarter and sales remain.
        let names: Vec<&str> = resolved.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["quarter", "sales"]);

        let mut headings: Vec<_> = resolved
            .cells()
            .iter()
            .filter(|c| c.kind == TableCellKind::GroupHeading)
            .collect();
        headings.sort_by_key(|c| c.top);
        assert_eq!(headings.len(), 2);
        // Row 0 is the column-label row; each heading spans both columns.
        assert_eq!(
            (headings[0].top, headings[0].left, headings[0].right),
            (1, 0, 1)
        );
        assert_eq!(headings[0].content, "north");
        // north has two data rows, so south's heading sits at row 4.
        assert_eq!(headings[1].top, 4);
        assert_eq!(headings[1].content, "south");

        assert_eq!(
            resolved
                .cells()
                .iter()
                .filter(|c| c.kind == TableCellKind::Body)
                .count(),
            6
        );
    }

    #[test]
    fn test_tabulate_facet_nests_label_rows_for_multiple_variables() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE t AS SELECT * FROM (VALUES \
                 ('north', 'q1', 1), ('north', 'q2', 2), ('south', 'q1', 3)) \
                 AS t(region, quarter, sales)",
            )
            .unwrap();

        let resolved =
            resolve_table_with_reader("TABULATE * FROM t FACET region, quarter", &reader).unwrap();

        let mut headings: Vec<_> = resolved
            .cells()
            .iter()
            .filter(|c| c.kind == TableCellKind::GroupHeading)
            .collect();
        headings.sort_by_key(|c| c.top);
        let texts: Vec<&str> = headings.iter().map(|c| c.content.as_str()).collect();
        // q1 is not restated while region stays north; south restates q1
        // under it. One data row per group: headings interleave singly.
        assert_eq!(texts, ["north", "q1", "q2", "south", "q1"]);
        let tops: Vec<usize> = headings.iter().map(|c| c.top).collect();
        assert_eq!(tops, [1, 2, 4, 6, 7]);
    }

    #[test]
    fn test_tabulate_facet_place_left_splits_groups_with_equal_labels() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE t AS SELECT * FROM (VALUES ('null', 1), (NULL, 2)) \
                 AS t(name, sales)",
            )
            .unwrap();

        // NULL and the string 'null' render identically but are distinct
        // groups; their cells must not merge into one rowspan.
        let resolved = resolve_table_with_reader(
            "TABULATE * FROM t FACET name SETTING place => 'left'",
            &reader,
        )
        .unwrap();

        let mut labels: Vec<_> = resolved
            .cells()
            .iter()
            .filter(|c| c.kind == TableCellKind::GroupHeading)
            .collect();
        labels.sort_by_key(|c| c.top);
        let extents: Vec<(&str, usize, usize)> = labels
            .iter()
            .map(|c| (c.content.as_str(), c.top, c.bottom))
            .collect();
        assert_eq!(extents, [("null", 1, 1), ("null", 2, 2)]);
    }

    #[test]
    fn test_tabulate_facet_missing_drop_removes_null_rows() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE t AS SELECT * FROM (VALUES ('a', 1), (NULL, 2), ('b', 3)) \
                 AS t(name, sales)",
            )
            .unwrap();

        let resolved = resolve_table_with_reader(
            "TABULATE * FROM t FACET name SETTING missing => 'drop'",
            &reader,
        )
        .unwrap();

        // Header row, and per surviving group a heading row plus one body row.
        assert_eq!(resolved.rows().len(), 5);
        let headings: Vec<_> = resolved
            .cells()
            .iter()
            .filter(|c| c.kind == TableCellKind::GroupHeading)
            .map(|c| c.content.as_str())
            .collect();
        assert_eq!(headings, ["a", "b"]);
    }

    #[test]
    fn test_tabulate_facet_covering_every_column_errors() {
        let reader = reader_with_sales();
        let result = resolve_table_with_reader("TABULATE name FROM sales FACET name", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_facet_rejects_a_spanner_over_a_facet_column() {
        let reader = reader_with_sales();
        let result =
            resolve_table_with_reader("TABULATE * FROM sales FACET name SPAN G OVER name", &reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_tabulate_facet_place_left_merges_group_labels_into_rowspans() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE t AS SELECT * FROM (VALUES \
                 ('north', 'q1', 1), ('north', 'q2', 2), ('south', 'q2', 3), \
                 ('south', 'q2', 4)) \
                 AS t(region, quarter, sales)",
            )
            .unwrap();

        let resolved = resolve_table_with_reader(
            "TABULATE * FROM t FACET region, quarter SETTING place => 'left'",
            &reader,
        )
        .unwrap();

        // Facet columns lead as group stubs, in nesting order.
        let names: Vec<&str> = resolved.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["region", "quarter", "sales"]);
        assert!(resolved.columns()[0].is_stub());
        assert!(resolved.columns()[1].is_stub());

        let mut labels: Vec<_> = resolved
            .cells()
            .iter()
            .filter(|c| c.kind == TableCellKind::GroupHeading)
            .collect();
        labels.sort_by_key(|c| (c.left, c.top));
        let extents: Vec<(&str, usize, usize, usize)> = labels
            .iter()
            .map(|c| (c.content.as_str(), c.left, c.top, c.bottom))
            .collect();
        // Row 0 is the column-label row. region spans both north rows and
        // both south rows; quarter's q2 splits at the region boundary even
        // though the label repeats, and merges within south's group.
        assert_eq!(
            extents,
            [
                ("north", 0, 1, 2),
                ("south", 0, 3, 4),
                ("q1", 1, 1, 1),
                ("q2", 1, 2, 2),
                ("q2", 1, 3, 4),
            ]
        );

        // The first body row of each group carries the group separator.
        let firsts: Vec<usize> = resolved
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, row)| row.classes.contains(&TableClass::RowGroupFirst))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(firsts, [1, 2, 3]);
    }

    #[test]
    fn test_tabulate_span_unknown_column_errors_even_with_gather_disabled() {
        // Table::resolve_spanners checks every OVER entry regardless of
        // gather, so this errors even though gather_columns itself (the
        // only place that also checks column existence) is skipped here.
        let reader = reader_with_sales();
        match resolve_table_with_reader(
            "TABULATE * FROM sales SPAN G OVER nope, id SETTING gather => false",
            &reader,
        ) {
            Ok(_) => panic!("expected an unknown OVER column to error"),
            Err(GgsqlError::ValidationError(msg)) => {
                assert!(msg.contains("SPAN references unknown column 'nope'"))
            }
            Err(e) => panic!("expected a ValidationError, got {e:?}"),
        }
    }
}
