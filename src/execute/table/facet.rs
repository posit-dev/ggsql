//! `TABULATE FACET` grouping: assigns each `DataFrame` row a group key from
//! its facet values, then reorders rows so groups are contiguous.

use std::collections::HashMap;

use arrow::array::{ArrayRef, UInt32Array};

use crate::array_util::value_to_string;
use crate::dataframe::into_array_ref;
use crate::plot::facet::{Facet, FacetLayout};
use crate::table::{TableColumn, TableRow};
use crate::{ColumnSection, DataFrame};

/// Assign each `df` row its group key: one rank per facet variable (see
/// `rank_column`). Lexicographic comparison of keys gives the nested group
/// order (first variable outermost); a key's prefix identifies the row's
/// group at that nesting depth.
pub(super) fn setup_groups(facet: &Facet, df: &DataFrame) -> Result<Vec<Vec<usize>>, String> {
    let columns = facet_columns(facet, df)?;

    let nulls_first = facet.properties.get("missing")
        == Some(&crate::plot::ParameterValue::String("first".to_string()));
    let mut keys = vec![Vec::with_capacity(columns.len()); df.height()];
    for column in &columns {
        let ranks = rank_column(column, nulls_first);
        for (key, rank) in keys.iter_mut().zip(ranks) {
            key.push(rank);
        }
    }

    Ok(keys)
}

/// Remove rows that are null in any of `facet`'s variables. Returns `df`
/// unchanged unless the facet drops nulls (`missing => 'drop'`).
pub(super) fn drop_null_rows(facet: &Facet, df: &DataFrame) -> Result<DataFrame, String> {
    if facet.properties.get("missing")
        != Some(&crate::plot::ParameterValue::String("drop".to_string()))
    {
        return Ok(df.clone());
    }
    let columns = facet_columns(facet, df)?;
    let keep: Vec<u32> = (0..df.height() as u32)
        .filter(|&row| columns.iter().all(|c| !c.is_null(row as usize)))
        .collect();
    df.take(&UInt32Array::from(keep)).map_err(|e| e.to_string())
}

/// Resolve `facet`'s variables to `df` columns.
fn facet_columns<'a>(facet: &Facet, df: &'a DataFrame) -> Result<Vec<&'a ArrayRef>, String> {
    facet
        .get_variables()
        .iter()
        .map(|name| {
            df.column(name)
                .map_err(|_| format!("TABULATE FACET: no column '{}' in the query result", name))
        })
        .collect()
}

/// Collapse a separated facet into one variable: join the facet variables'
/// (formatted) values per row into a column named for the first variable,
/// drop the originals, renumber the rows' group keys to single ids, and
/// reduce the facet to that variable. Group stub columns in `columns` are
/// replaced by one, keeping the first's label and properties. Passes
/// everything through unchanged when the facet sets no `separator` or has a
/// single variable.
/// The table pieces `collapse_group_columns` rewrites for a separated facet.
type Collapsed = (DataFrame, Vec<TableColumn>, Vec<TableRow>, Option<Facet>);

pub(super) fn collapse_group_columns(
    df: DataFrame,
    mut columns: Vec<TableColumn>,
    mut rows: Vec<TableRow>,
    facet: Option<&Facet>,
) -> Result<Collapsed, String> {
    let Some(facet) = facet else {
        return Ok((df, columns, rows, None));
    };
    let separator = match facet.properties.get("separator") {
        Some(crate::plot::ParameterValue::String(separator)) => separator.clone(),
        _ => return Ok((df, columns, rows, Some(facet.clone()))),
    };

    // One variable has nothing to join.
    let variables = facet.get_variables();
    if variables.len() == 1 {
        return Ok((df, columns, rows, Some(facet.clone())));
    }
    // The joined column takes the first variable's name.
    let name = variables[0].clone();

    let arrays: Vec<&ArrayRef> = variables
        .iter()
        .map(|v| df.column(v).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let joined: Vec<String> = (0..df.height())
        .map(|row| {
            arrays
                .iter()
                .map(|array| value_to_string(array, row))
                .collect::<Vec<_>>()
                .join(&separator)
        })
        .collect();
    let df = df
        .drop_many(&variables)
        .and_then(|df| df.with_column(&name, into_array_ref(joined)))
        .map_err(|e| e.to_string())?;

    // Single-id keys: rows are group-sorted, so the id is the chunk index.
    let mut id = 0;
    let mut previous: Option<Vec<usize>> = None;
    for row in &mut rows {
        if previous.as_ref().is_some_and(|key| *key != row.group) {
            id += 1;
        }
        previous = Some(std::mem::take(&mut row.group));
        row.group = vec![id];
    }

    // Replace the group stub columns with one, at the first's position.
    if let Some(position) = columns.iter().position(|c| c.is_group_stub()) {
        let mut joined_column = columns[position].clone();
        joined_column.target = ColumnSection::GroupStub(0);
        columns.retain(|c| !c.is_group_stub());
        columns.insert(position, joined_column);
    }

    let mut properties = facet.properties.clone();
    properties.remove("separator");
    let facet = Facet {
        layout: FacetLayout::Wrap {
            variables: vec![name],
        },
        properties,
        resolved: facet.resolved,
    };

    Ok((df, columns, rows, Some(facet)))
}

/// Rank one facet variable's rows: non-null levels by first appearance,
/// nulls at a fixed rank (0 with `nulls_first`, otherwise past the non-null
/// levels) so their placement doesn't depend on where they occur.
fn rank_column(column: &ArrayRef, nulls_first: bool) -> Vec<usize> {
    let mut levels: HashMap<String, usize> = HashMap::new();
    let mut values: Vec<Option<String>> = Vec::with_capacity(column.len());
    for row in 0..column.len() {
        if column.is_null(row) {
            values.push(None);
            continue;
        }
        let value = value_to_string(column, row);
        let next = levels.len();
        levels.entry(value.clone()).or_insert(next);
        values.push(Some(value));
    }

    let null_rank = if nulls_first { 0 } else { levels.len() };
    let offset = usize::from(nulls_first);
    values
        .into_iter()
        .map(|value| match value {
            Some(value) => levels[&value] + offset,
            None => null_rank,
        })
        .collect()
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
    use crate::plot::facet::FacetLayout;
    use crate::plot::{ParameterValue, Parameters};

    fn facet(names: &[&str], missing: &str) -> Facet {
        let mut properties = Parameters::new();
        properties.insert(
            "missing".to_string(),
            ParameterValue::String(missing.to_string()),
        );
        Facet {
            layout: FacetLayout::Wrap {
                variables: names.iter().map(|n| n.to_string()).collect(),
            },
            properties,
            resolved: true,
        }
    }

    #[test]
    fn setup_groups_assigns_keys_in_first_appearance_order() {
        let frame = df! {
            "region" => vec!["south".to_string(), "north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2, 3],
        }
        .unwrap();

        let keys = setup_groups(&facet(&["region"], "last"), &frame).unwrap();

        assert_eq!(keys, vec![vec![0], vec![1], vec![0]]);
    }

    #[test]
    fn setup_groups_uses_the_whole_key_for_multiple_variables() {
        let frame = df! {
            "region" => vec!["north".to_string(), "north".to_string()],
            "quarter" => vec!["q1".to_string(), "q2".to_string()],
        }
        .unwrap();

        let keys = setup_groups(&facet(&["region", "quarter"], "last"), &frame).unwrap();

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

        let keys = setup_groups(&facet(&["a", "b"], "last"), &frame).unwrap();

        assert_eq!(keys, vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]);
    }

    #[test]
    fn setup_groups_errors_on_an_unknown_column() {
        let frame = df! { "sales" => vec![1i32] }.unwrap();

        let err = setup_groups(&facet(&["region"], "last"), &frame).unwrap_err();

        assert!(err.contains("region"));
    }

    #[test]
    fn reorder_rows_clusters_groups_and_preserves_within_group_order() {
        let frame = df! {
            "region" => vec!["south".to_string(), "north".to_string(), "south".to_string()],
            "sales" => vec![1i32, 2, 3],
        }
        .unwrap();
        let keys = setup_groups(&facet(&["region"], "last"), &frame).unwrap();

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

        let keys = setup_groups(&facet(&["region"], "last"), &frame).unwrap();

        assert_eq!(keys, vec![vec![0], vec![1], vec![1]]);
    }

    #[test]
    fn setup_groups_ranks_nulls_first_when_asked() {
        let frame = DataFrame::new(vec![(
            "region",
            crate::dataframe::into_array_ref(vec![Some("north"), None, Some("south")]),
        )])
        .unwrap();

        let keys = setup_groups(&facet(&["region"], "first"), &frame).unwrap();

        assert_eq!(keys, vec![vec![1], vec![0], vec![2]]);
    }

    #[test]
    fn setup_groups_ranks_nulls_last_regardless_of_where_they_occur() {
        let frame = DataFrame::new(vec![(
            "region",
            crate::dataframe::into_array_ref(vec![Some("north"), None, Some("south")]),
        )])
        .unwrap();

        let keys = setup_groups(&facet(&["region"], "last"), &frame).unwrap();

        assert_eq!(keys, vec![vec![0], vec![2], vec![1]]);
    }

    #[test]
    fn drop_null_rows_removes_rows_null_in_any_variable() {
        let frame = DataFrame::new(vec![
            (
                "region",
                crate::dataframe::into_array_ref(vec![Some("north"), None, Some("south")]),
            ),
            (
                "quarter",
                crate::dataframe::into_array_ref(vec![Some("q1"), Some("q2"), None]),
            ),
        ])
        .unwrap();

        let frame = drop_null_rows(&facet(&["region", "quarter"], "drop"), &frame).unwrap();

        assert_eq!(frame.height(), 1);
    }
}
