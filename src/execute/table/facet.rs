//! `TABULATE FACET` grouping: assigns each `DataFrame` row a group key from
//! its facet values, then reorders rows so groups are contiguous.

use std::collections::HashMap;

use arrow::array::UInt32Array;

use crate::array_util::value_to_string;
use crate::DataFrame;

/// Assign each `df` row its group key: one rank per facet variable, levels
/// ranked by first appearance. Lexicographic comparison of keys gives the
/// nested group order (first variable outermost); a key's prefix identifies
/// the row's group at that nesting depth. Rows that are null in a variable
/// share one rank for it.
pub(super) fn setup_groups(
    variables: &[String],
    df: &DataFrame,
) -> Result<Vec<Vec<usize>>, String> {
    let columns: Vec<_> = variables
        .iter()
        .map(|name| {
            df.column(name)
                .map_err(|_| format!("TABULATE FACET: no column '{}' in the query result", name))
        })
        .collect::<Result<_, _>>()?;

    // Rank each variable's values by first appearance.
    let mut levels: Vec<HashMap<Option<String>, usize>> = vec![HashMap::new(); columns.len()];
    let mut keys = Vec::with_capacity(df.height());
    for row in 0..df.height() {
        let key: Vec<usize> = columns
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
        keys.push(key);
    }

    Ok(keys)
}

/// Reorder `df` so rows with equal group keys are contiguous in
/// lexicographic key order, keeping the query's row order within each
/// group. Returns the reordered frame and the keys in their new order.
pub(super) fn reorder_rows(
    df: &DataFrame,
    keys: Vec<Vec<usize>>,
) -> Result<(DataFrame, Vec<Vec<usize>>), String> {
    let mut order: Vec<u32> = (0..df.height() as u32).collect();
    // sort_by is stable: within-group row order is preserved.
    order.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));

    let df = df
        .take(&UInt32Array::from(order.clone()))
        .map_err(|e| e.to_string())?;
    let keys = order
        .into_iter()
        .map(|i| keys[i as usize].clone())
        .collect();

    Ok((df, keys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    fn variables(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn setup_groups_assigns_keys_in_first_appearance_order() {
        let frame = df! {
            "region" => vec!["south".to_string(), "north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2, 3],
        }
        .unwrap();

        let keys = setup_groups(&variables(&["region"]), &frame).unwrap();

        assert_eq!(keys, vec![vec![0], vec![1], vec![0]]);
    }

    #[test]
    fn setup_groups_uses_the_whole_key_for_multiple_variables() {
        let frame = df! {
            "region" => vec!["north".to_string(), "north".to_string()],
            "quarter" => vec!["q1".to_string(), "q2".to_string()],
        }
        .unwrap();

        let keys = setup_groups(&variables(&["region", "quarter"]), &frame).unwrap();

        assert_eq!(keys, vec![vec![0, 0], vec![0, 1]]);
    }

    #[test]
    fn setup_groups_orders_keys_nested_by_variable() {
        // Row order interleaves the variables; keys follow the nested order
        // (a,x), (a,y), (b,x), (b,y) instead.
        let frame = df! {
            "a" => vec!["a".to_string(), "b".to_string(), "a".to_string(), "b".to_string()],
            "b" => vec!["x".to_string(), "x".to_string(), "y".to_string(), "y".to_string()],
        }
        .unwrap();

        let keys = setup_groups(&variables(&["a", "b"]), &frame).unwrap();

        assert_eq!(keys, vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]);
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
        let keys = setup_groups(&variables(&["region"]), &frame).unwrap();

        let (frame, keys) = reorder_rows(&frame, keys).unwrap();

        assert_eq!(keys, vec![vec![0], vec![0], vec![1]]);
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

        let keys = setup_groups(&variables(&["region"]), &frame).unwrap();

        assert_eq!(keys, vec![vec![0], vec![1], vec![1]]);
    }
}
