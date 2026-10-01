//! `TABULATE SPAN` resolution: column reordering (`gather`) and header-row
//! (level) assignment for spanners.

use super::layout::{Section, TableColumn};
use crate::{Spanner, TableCell, TableCellKind, TableClass, TableRow};

/// Reorder `columns` so every `gather`-enabled spanner's members become
/// contiguous, folding spanners in declaration order. `SETTING gather =>
/// false` leaves a spanner's columns in place. Touches column order only;
/// header-row assignment is `assign_spanner_levels`' concern.
pub(crate) fn reorder_table_columns(
    mut columns: Vec<TableColumn>,
    spans: &[Spanner],
) -> Vec<TableColumn> {
    for span in spans {
        let gather = span
            .settings
            .get("gather")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);

        if gather {
            columns = gather_columns(columns, &span.columns);
        }
    }

    columns
}

/// Move `members` (in the order given) to sit contiguously where the first
/// of them currently is, pulling the rest in around it — the minimal move
/// that unifies them, leaving every other column's relative order untouched.
fn gather_columns(columns: Vec<TableColumn>, members: &[String]) -> Vec<TableColumn> {
    let Some(anchor) = members.first() else {
        return columns;
    };

    let anchor_index = columns
        .iter()
        .position(|c| &c.name == anchor)
        .expect("Table::resolve_spanners already validated every ACROSS column exists");
    let insertion_index = columns[..anchor_index]
        .iter()
        .filter(|c| !members.contains(&c.name))
        .count();

    let mut remainder = Vec::with_capacity(columns.len());
    let mut by_name: std::collections::HashMap<String, TableColumn> =
        std::collections::HashMap::new();
    for column in columns {
        if members.contains(&column.name) {
            by_name.insert(column.name.clone(), column);
        } else {
            remainder.push(column);
        }
    }

    let mut result: Vec<TableColumn> = remainder.drain(..insertion_index).collect();
    for name in members {
        let column = by_name
            .remove(name)
            .expect("Table::resolve_spanners already validated every ACROSS column exists");
        result.push(column);
    }
    result.extend(remainder);

    result
}

/// Assign each spanner a 1-indexed level (header row). An explicit `SETTING
/// level => N` pins a spanner there with no conflict check —
/// `validate_overlaps` catches a genuine clash once real cells exist. A
/// spanner with no explicit level is assigned greedily (see the `None`
/// arm). Levels are then compacted to remove gaps explicit levels can leave
/// (e.g. 1, 3, 4 → 1, 2, 3).
///
/// Infallible: `level`'s type is checked by `Spanner::validate_settings`.
fn assign_spanner_levels(spans: &[Spanner]) -> Vec<usize> {
    let mut levels: Vec<usize> = Vec::with_capacity(spans.len());

    for span in spans {
        let level = match span.settings.get("level") {
            Some(value) => value
                .as_number()
                .expect("Spanner::validate_settings already checked 'level' is a number")
                as usize,
            None => {
                // One more than the highest level of any already-assigned
                // spanner whose columns intersect this one's. Never revisits
                // a lower level, so a chain of pairwise-but-not-all
                // conflicting spanners can use more levels than strictly
                // necessary.
                spans
                    .iter()
                    .zip(&levels)
                    .filter(|(other, _)| other.columns.iter().any(|c| span.columns.contains(c)))
                    .map(|(_, &level)| level)
                    .max()
                    .unwrap_or(0)
                    + 1
            }
        };
        levels.push(level);
    }

    // Compact gaps left by explicit levels (e.g. 1, 3, 4) into a dense
    // range (1, 2, 3).
    let mut distinct = levels.clone();
    distinct.sort_unstable();
    distinct.dedup();

    levels
        .into_iter()
        .map(|level| distinct.binary_search(&level).unwrap() + 1)
        .collect()
}

/// Build one `TableCell` per contiguous run of a spanner's columns, calling
/// `assign_spanner_levels` itself. Rows are numbered locally from
/// `top == 0`. `Section::rows` gets exactly `max_level` entries, each
/// carrying `TableClass::ColHeadingRow`, like the column-label row does.
pub(crate) fn create_spanners(columns: &[TableColumn], spans: &[Spanner]) -> Section {
    if spans.is_empty() {
        return Section::new(Vec::new(), Vec::new());
    }

    // Spanners with no display text (`SPAN NULL`, or `LABEL <id> => NULL`)
    // contribute no cells, so their level shouldn't affect other levels.
    let spans: Vec<Spanner> = spans
        .iter()
        .filter(|s| s.label.is_some())
        .cloned()
        .collect();

    let levels = assign_spanner_levels(&spans);
    let max_level = levels.iter().copied().max().unwrap_or(0);

    let mut cells = Vec::new();

    for (span, &level) in spans.iter().zip(&levels) {
        let label = span
            .label
            .as_ref()
            .expect("spans is filtered to only Some(label) spanners");
        // Row 0 is topmost. Level 1 is bottom-most.
        let row = max_level - level;
        let classes = vec![
            TableClass::Spanner,
            TableClass::SpannerLabel,
            TableClass::AlignCenter,
        ];

        // Run-length encode the span's columns; disjoint columns give
        // multiple runs.
        let mut run_start = None;
        for (index, column) in columns.iter().enumerate() {
            // Does column belong to span?
            let in_span = span.columns.contains(&column.name);
            match (in_span, run_start) {
                // Found column in span, initiate new run
                (true, None) => run_start = Some(index),
                // Column not in span, end run and push cell
                (false, Some(start)) => {
                    cells.push(
                        TableCell::new(
                            TableCellKind::Spanner,
                            row,
                            row,
                            start,
                            index - 1,
                            label.clone(),
                        )
                        .with_classes(classes.clone()),
                    );
                    run_start = None;
                }
                _ => {}
            }
        }
        // Close a run still open at the last column.
        if let Some(start) = run_start {
            cells.push(
                TableCell::new(
                    TableCellKind::Spanner,
                    row,
                    row,
                    start,
                    columns.len() - 1,
                    label.clone(),
                )
                .with_classes(classes),
            );
        }
    }

    Section::new(
        vec![
            TableRow {
                classes: vec![TableClass::ColHeadingRow, TableClass::SpannerRow],
                ..TableRow::header()
            };
            max_level
        ],
        cells,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::{ParameterValue, Parameters};
    use crate::ColumnSection;

    fn column(name: &str, label: &str) -> TableColumn {
        TableColumn {
            name: name.to_string(),
            label: label.to_string(),
            properties: Parameters::new(),
            target: ColumnSection::Body,
        }
    }

    fn names(columns: &[TableColumn]) -> Vec<&str> {
        columns.iter().map(|c| c.name.as_str()).collect()
    }

    fn spanner_with(columns: &[&str], label: Option<&str>, settings: Parameters) -> Spanner {
        Spanner {
            id: label.unwrap_or("").to_string(),
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

    fn null_spanner(columns: &[&str]) -> Spanner {
        spanner_with(columns, None, Parameters::new())
    }

    #[test]
    fn gather_columns_moves_members_next_to_the_anchor_pulling_from_the_right() {
        let columns = vec![
            column("x", "x"),
            column("a", "a"),
            column("y", "y"),
            column("b", "b"),
            column("z", "z"),
            column("c", "c"),
            column("w", "w"),
        ];

        let result = gather_columns(
            columns,
            &["a".to_string(), "b".to_string(), "c".to_string()],
        );

        assert_eq!(names(&result), vec!["x", "a", "b", "c", "y", "z", "w"]);
    }

    #[test]
    fn gather_columns_uses_declared_order_not_original_order() {
        // Members are scattered as C, A, B originally, but declared as
        // [A, B, C] — gathering must produce A, B, C (declared order), not
        // C, A, B (original relative order).
        let columns = vec![
            column("c", "c"),
            column("x", "x"),
            column("a", "a"),
            column("y", "y"),
            column("b", "b"),
        ];

        let result = gather_columns(
            columns,
            &["a".to_string(), "b".to_string(), "c".to_string()],
        );

        assert_eq!(names(&result), vec!["x", "a", "b", "c", "y"]);
    }

    #[test]
    fn gather_columns_is_a_no_op_when_already_contiguous() {
        let columns = vec![column("a", "a"), column("b", "b"), column("c", "c")];

        let result = gather_columns(columns, &["a".to_string(), "b".to_string()]);

        assert_eq!(names(&result), vec!["a", "b", "c"]);
    }

    #[test]
    #[should_panic(expected = "Table::resolve_spanners already validated")]
    fn gather_columns_panics_on_an_unvalidated_unknown_column() {
        let columns = vec![column("a", "a"), column("b", "b")];

        gather_columns(columns, &["a".to_string(), "nope".to_string()]);
    }

    #[test]
    fn reorder_table_columns_defaults_gather_to_true() {
        let columns = vec![
            column("a", "a"),
            column("x", "x"),
            column("b", "b"),
            column("y", "y"),
        ];
        let spans = vec![spanner(&["a", "b"])];

        let result = reorder_table_columns(columns, &spans);

        assert_eq!(names(&result), vec!["a", "b", "x", "y"]);
    }

    #[test]
    fn reorder_table_columns_skips_gather_false() {
        let columns = vec![
            column("a", "a"),
            column("x", "x"),
            column("b", "b"),
            column("y", "y"),
        ];
        let spans = vec![spanner_with_setting(
            &["a", "b"],
            "gather",
            ParameterValue::Boolean(false),
        )];

        let result = reorder_table_columns(columns, &spans);

        assert_eq!(names(&result), vec!["a", "x", "b", "y"]);
    }

    #[test]
    fn reorder_table_columns_folds_spanners_in_declaration_order() {
        let columns = vec![
            column("a", "a"),
            column("x", "x"),
            column("b", "b"),
            column("y", "y"),
        ];
        // Second span's gather runs against the *result* of the first, not
        // the original order.
        let spans = vec![spanner(&["a", "b"]), spanner(&["b", "y"])];

        let result = reorder_table_columns(columns, &spans);

        assert_eq!(names(&result), vec!["a", "b", "y", "x"]);
    }

    #[test]
    fn assign_spanner_levels_needs_three_levels_for_mutually_crossing_spanners() {
        let spans = vec![
            spanner(&["a", "b"]),
            spanner(&["b", "c"]),
            spanner(&["a", "c"]),
        ];

        let levels = assign_spanner_levels(&spans);

        assert_eq!(levels, vec![1, 2, 3]);
    }

    #[test]
    fn assign_spanner_levels_pushes_a_chained_conflict_past_a_free_level() {
        // X(a,b) and Z(c,d) share no column and could both sit at level 1,
        // but Z conflicts with Y(b,c), which conflicts with X — matching
        // gt, Z gets pushed to level 3 rather than reusing X's level 1.
        let spans = vec![
            spanner(&["a", "b"]),
            spanner(&["b", "c"]),
            spanner(&["c", "d"]),
        ];

        let levels = assign_spanner_levels(&spans);

        assert_eq!(levels, vec![1, 2, 3]);
    }

    #[test]
    fn assign_spanner_levels_pins_explicit_level_without_a_conflict_check() {
        // (a,b) auto-assigns to level 1; (b,c) is explicitly pinned to level
        // 1 too, despite conflicting with (a,b) — this function doesn't
        // error on that, validate_overlaps does once real cells exist.
        let spans = vec![
            spanner(&["a", "b"]),
            spanner_with_setting(&["b", "c"], "level", ParameterValue::Number(1.0)),
        ];

        let levels = assign_spanner_levels(&spans);

        assert_eq!(levels, vec![1, 1]);
    }

    #[test]
    fn assign_spanner_levels_compacts_gaps_left_by_explicit_levels() {
        let spans = vec![
            spanner_with_setting(&["a"], "level", ParameterValue::Number(1.0)),
            spanner_with_setting(&["b"], "level", ParameterValue::Number(3.0)),
            spanner_with_setting(&["c"], "level", ParameterValue::Number(4.0)),
        ];

        let levels = assign_spanner_levels(&spans);

        assert_eq!(levels, vec![1, 2, 3]);
    }

    #[test]
    fn create_spanners_builds_one_cell_per_disjoint_spanner_on_the_same_row() {
        let columns = vec![
            column("a", "a"),
            column("b", "b"),
            column("c", "c"),
            column("d", "d"),
        ];
        let spans = vec![
            labeled_spanner(&["a", "b"], "G1"),
            labeled_spanner(&["c", "d"], "G2"),
        ];

        let cells = create_spanners(&columns, &spans).into_cells();

        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].top, 0);
        assert_eq!(cells[0].bottom, 0);
        assert_eq!(cells[0].left, 0);
        assert_eq!(cells[0].right, 1);
        assert_eq!(cells[0].content, "G1");
        assert_eq!(cells[1].left, 2);
        assert_eq!(cells[1].right, 3);
        assert_eq!(cells[1].content, "G2");
        assert!(cells.iter().all(|c| c.kind == TableCellKind::Spanner));
    }

    #[test]
    fn create_spanners_puts_crossing_spanners_on_separate_rows() {
        let columns = vec![column("a", "a"), column("b", "b"), column("c", "c")];
        let spans = vec![
            labeled_spanner(&["a", "b"], "G1"),
            labeled_spanner(&["b", "c"], "G2"),
        ];

        let cells = create_spanners(&columns, &spans).into_cells();

        // Level 1 (G1, closest to the columns) is the bottom spanner row —
        // the higher local row number, since row 0 is the topmost row.
        let g1 = cells.iter().find(|c| c.content == "G1").unwrap();
        let g2 = cells.iter().find(|c| c.content == "G2").unwrap();
        assert_eq!(g1.top, 1);
        assert_eq!(g1.left, 0);
        assert_eq!(g1.right, 1);
        assert_eq!(g2.top, 0);
        assert_eq!(g2.left, 1);
        assert_eq!(g2.right, 2);
    }

    #[test]
    fn create_spanners_fragments_a_non_contiguous_spanner_into_multiple_cells() {
        let columns = vec![column("a", "a"), column("x", "x"), column("b", "b")];
        let spans = vec![labeled_spanner(&["a", "b"], "G")];

        let cells = create_spanners(&columns, &spans).into_cells();

        assert_eq!(cells.len(), 2);
        assert!(cells.iter().all(|c| c.content == "G"));
        assert_eq!(cells[0].left, 0);
        assert_eq!(cells[0].right, 0);
        assert_eq!(cells[1].left, 2);
        assert_eq!(cells[1].right, 2);
    }

    #[test]
    fn create_spanners_skips_null_labeled_spanners_and_their_levels() {
        // If the NULL spanner participated in level assignment, it would
        // cross (a,b) and (b,c) and force "G" to level 2 — filtering it out
        // first means "G" is the only spanner left, so it stays at level 1.
        let columns = vec![column("a", "a"), column("b", "b"), column("c", "c")];
        let spans = vec![null_spanner(&["a", "b"]), labeled_spanner(&["b", "c"], "G")];

        let cells = create_spanners(&columns, &spans).into_cells();

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].top, 0);
        assert_eq!(cells[0].content, "G");
        assert_eq!(cells[0].left, 1);
        assert_eq!(cells[0].right, 2);
    }

    #[test]
    fn create_spanners_returns_empty_for_no_spanners() {
        let columns = vec![column("a", "a"), column("b", "b")];

        let cells = create_spanners(&columns, &[]).into_cells();

        assert!(cells.is_empty());
    }

    fn spanner_with_id(columns: &[&str], id: &str) -> Spanner {
        Spanner {
            id: id.to_string(),
            ..spanner_with(columns, Some(""), Parameters::new())
        }
    }

    fn table_with_spans(spans: Vec<Spanner>) -> crate::Table {
        crate::Table {
            spans,
            ..crate::Table::new()
        }
    }

    #[test]
    fn resolve_spanner_ids_expands_a_reference_to_an_earlier_spanners_columns() {
        let table = table_with_spans(vec![
            spanner_with_id(&["a", "b"], "x"),
            spanner(&["x", "c"]),
        ]);

        let resolved = table.resolve_spanner_ids().unwrap();

        assert_eq!(resolved[1].columns, vec!["a", "b", "c"]);
    }

    #[test]
    fn resolve_spanner_ids_leaves_a_forward_reference_unresolved() {
        // "x" is declared after it's referenced here — left as a literal
        // string, to be caught downstream as an unknown column.
        let table = table_with_spans(vec![
            spanner(&["x", "c"]),
            spanner_with_id(&["a", "b"], "x"),
        ]);

        let resolved = table.resolve_spanner_ids().unwrap();

        assert_eq!(resolved[0].columns, vec!["x", "c"]);
    }

    #[test]
    fn resolve_spanner_ids_rejects_a_duplicate_id() {
        let table = table_with_spans(vec![
            spanner_with_id(&["a"], "dup"),
            spanner_with_id(&["b"], "dup"),
        ]);

        assert!(table.resolve_spanner_ids().is_err());
    }

    #[test]
    fn resolve_spanner_ids_resolves_transitively() {
        // z -> y -> x: z references y, which already expanded its own
        // reference to x by the time z is processed.
        let table = table_with_spans(vec![
            spanner_with_id(&["a", "b"], "x"),
            spanner_with_id(&["x", "c"], "y"),
            spanner(&["y", "d"]),
        ]);

        let resolved = table.resolve_spanner_ids().unwrap();

        assert_eq!(resolved[2].columns, vec!["a", "b", "c", "d"]);
    }
}
