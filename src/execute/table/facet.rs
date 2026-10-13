//! `TABULATE FACET` grouping: assigns each `DataFrame` row a group id from
//! its facet values, then reorders rows so groups are contiguous.

use std::collections::HashMap;

use arrow::array::UInt32Array;

use crate::array_util::value_to_string;
use crate::DataFrame;

/// Assign each `df` row a group id. Groups are ordered nested by variable:
/// each variable's levels are ranked by first appearance, and group ids
/// follow lexicographic order of the rank tuples, so the first variable
/// forms the outermost grouping. Rows that are null in every facet variable
/// share one group.
pub(super) fn setup_groups(variables: &[String], df: &DataFrame) -> Result<Vec<usize>, String> {
    let columns: Vec<_> = variables
        .iter()
        .map(|name| {
            df.column(name)
                .map_err(|_| format!("TABULATE FACET: no column '{}' in the query result", name))
        })
        .collect::<Result<_, _>>()?;

    // Rank each variable's values by first appearance.
    let mut levels: Vec<HashMap<Option<String>, usize>> = vec![HashMap::new(); columns.len()];
    let mut tuples = Vec::with_capacity(df.height());
    for row in 0..df.height() {
        let tuple: Vec<usize> = columns
            .iter()
            .zip(&mut levels)
            .map(|(column, ranks)| {
                let value = if column.is_null(row) {
                    None
                } else {
                    Some(value_to_string(column, row))
                };
                let next = ranks.len();
                *ranks.entry(value).or_insert(next)
            })
            .collect();
        tuples.push(tuple);
    }

    // Group ids follow the nested (lexicographic) order of rank tuples.
    let mut unique = tuples.clone();
    unique.sort();
    unique.dedup();
    let group_of: HashMap<Vec<usize>, usize> = unique
        .into_iter()
        .enumerate()
        .map(|(id, tuple)| (tuple, id))
        .collect();

    Ok(tuples.into_iter().map(|tuple| group_of[&tuple]).collect())
}

/// Reorder `df` so rows with the same group id are contiguous, keeping the
/// query's row order within each group. Returns the reordered frame and the
/// group ids in their new order.
pub(super) fn reorder_rows(
    df: &DataFrame,
    ids: Vec<usize>,
) -> Result<(DataFrame, Vec<usize>), String> {
    let mut order: Vec<u32> = (0..df.height() as u32).collect();
    // sort_by_key is stable: within-group row order is preserved.
    order.sort_by_key(|&i| ids[i as usize]);

    let df = df
        .take(&UInt32Array::from(order.clone()))
        .map_err(|e| e.to_string())?;
    let ids = order.into_iter().map(|i| ids[i as usize]).collect();

    Ok((df, ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    fn variables(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn setup_groups_assigns_ids_in_first_appearance_order() {
        let frame = df! {
            "region" => vec!["south".to_string(), "north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2, 3],
        }
        .unwrap();

        let ids = setup_groups(&variables(&["region"]), &frame).unwrap();

        assert_eq!(ids, vec![0, 1, 0]);
    }

    #[test]
    fn setup_groups_uses_the_whole_key_for_multiple_variables() {
        let frame = df! {
            "region" => vec!["north".to_string(), "north".to_string()],
            "quarter" => vec!["q1".to_string(), "q2".to_string()],
        }
        .unwrap();

        let ids = setup_groups(&variables(&["region", "quarter"]), &frame).unwrap();

        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn setup_groups_orders_ids_nested_by_variable() {
        // Row order interleaves the variables; ids follow the nested order
        // (a,x), (a,y), (b,x), (b,y) instead.
        let frame = df! {
            "a" => vec!["a".to_string(), "b".to_string(), "a".to_string(), "b".to_string()],
            "b" => vec!["x".to_string(), "x".to_string(), "y".to_string(), "y".to_string()],
        }
        .unwrap();

        let ids = setup_groups(&variables(&["a", "b"]), &frame).unwrap();

        assert_eq!(ids, vec![0, 2, 1, 3]);
    }

    #[test]
    fn setup_groups_errors_on_an_unknown_column() {
        let frame = df! { "sales" => vec![1i32] }.unwrap();

        let err = setup_groups(&variables(&["region"]), &frame).unwrap_err();

        assert!(err.contains("region"));
    }

    #[test]
    fn reorder_rows_clusters_groups_and_preserves_within_group_order() {
        let frame = df! {
            "region" => vec!["south".to_string(), "north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2, 3],
        }
        .unwrap();
        let ids = setup_groups(&variables(&["region"]), &frame).unwrap();

        let (frame, ids) = reorder_rows(&frame, ids).unwrap();

        assert_eq!(ids, vec![0, 0, 1]);
        assert_eq!(
            frame
                .column("sales")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap()
                .values(),
            &[1, 3, 2]
        );
    }

    #[test]
    fn setup_groups_buckets_nulls_into_one_group() {
        let frame = DataFrame::new(vec![(
            "region",
            crate::dataframe::into_array_ref(vec![Some("north"), None, None]),
        )])
        .unwrap();

        let ids = setup_groups(&variables(&["region"]), &frame).unwrap();

        assert_eq!(ids, vec![0, 1, 1]);
    }
}
