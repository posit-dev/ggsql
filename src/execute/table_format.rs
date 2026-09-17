//! `TABULATE FORMAT` resolution: replacing each `FORMAT`-covered column in a
//! table's `DataFrame` with its resolved display text, called from
//! `table::build_cells` before column/cell layout runs — so `create_body`'s
//! ordinary `value_to_string` rendering already reflects any `RENAMING`, and
//! never needs to know `FORMAT` exists.

use std::collections::HashMap;

use crate::array_util::{new_str_array, value_to_string};
use crate::{DataFrame, Format, GgsqlError, Result};

/// Replace every `FORMAT`-covered column in `df` with its resolved display
/// text. A later `FORMAT` clause wins over an earlier one naming the same
/// column. A column whose `FORMAT` has no `RENAMING` at all (`value_template`
/// is the default `"{}"` and `value_mapping` is empty) is left untouched.
pub(crate) fn apply_formats(df: &DataFrame, formats: &[Format]) -> Result<DataFrame> {
    let mut by_column: HashMap<&str, &Format> = HashMap::new();
    for format in formats {
        for name in &format.columns {
            if df.column(name).is_err() {
                return Err(GgsqlError::ValidationError(format!(
                    "FORMAT references unknown column '{name}'"
                )));
            }
            by_column.insert(name.as_str(), format);
        }
    }

    let mut df = df.clone();
    for (name, format) in by_column {
        let has_renaming = format.value_template != "{}"
            || format.value_mapping.as_ref().is_some_and(|m| !m.is_empty());
        if !has_renaming {
            // Skip the per-row formatting pass for a column nobody asked to
            // reformat — the common `FORMAT ... SETTING ...` case.
            continue;
        }

        let array = df
            .column(name)
            .expect("name was just checked above, before df was cloned")
            .clone();
        let resolved = crate::format::resolve_column_values(
            &array,
            &format.value_template,
            &format.value_mapping,
        )
        .map_err(|e| GgsqlError::ValidationError(format!("FORMAT column '{name}': {e}")))?;

        let display: Vec<Option<String>> = resolved
            .into_iter()
            .enumerate()
            .map(|(row, value)| Some(value.unwrap_or_else(|| value_to_string(&array, row))))
            .collect();
        let display_refs: Vec<Option<&str>> = display.iter().map(|s| s.as_deref()).collect();

        df = df.with_column(name, new_str_array(display_refs))?;
    }

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_util::value_to_string;
    use crate::df;

    fn format_with(
        columns: &[&str],
        template: &str,
        mapping: Option<HashMap<String, Option<String>>>,
    ) -> Format {
        Format {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            settings: crate::plot::Parameters::new(),
            value_mapping: mapping,
            value_template: template.to_string(),
        }
    }

    fn column_content(df: &DataFrame, name: &str) -> Vec<String> {
        let array = df.column(name).unwrap();
        (0..df.height())
            .map(|row| value_to_string(array, row))
            .collect()
    }

    #[test]
    fn leaves_a_column_untouched_when_its_format_has_no_renaming() {
        let frame = df! { "price" => vec![1.0f64, 2.0] }.unwrap();
        let formats = vec![format_with(&["price"], "{}", None)];

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["1", "2"]);
    }

    #[test]
    fn applies_a_wildcard_template() {
        let frame = df! { "price" => vec![1.0f64, 2.0] }.unwrap();
        let formats = vec![format_with(&["price"], "${:num %.2f}", None)];

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["$1.00", "$2.00"]);
    }

    #[test]
    fn explicit_mapping_wins_over_the_template() {
        let frame = df! { "price" => vec![0.0f64, 5.0] }.unwrap();
        let mapping = Some(HashMap::from([("0".to_string(), Some("-".to_string()))]));
        let formats = vec![format_with(&["price"], "{:num %.2f}", mapping)];

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["-", "5.00"]);
    }

    #[test]
    fn later_format_wins_for_a_column_named_by_two_formats() {
        let frame = df! { "price" => vec![1.0f64] }.unwrap();
        let formats = vec![
            format_with(&["price"], "{:num %.0f}", None),
            format_with(&["price"], "{:num %.2f}", None),
        ];

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["1.00"]);
    }

    #[test]
    fn errors_on_a_format_naming_an_unknown_column() {
        let frame = df! { "price" => vec![1.0f64] }.unwrap();
        let formats = vec![format_with(&["typo"], "{}", None)];

        let error = apply_formats(&frame, &formats).unwrap_err();
        assert!(matches!(error, GgsqlError::ValidationError(msg)
            if msg.contains("FORMAT references unknown column 'typo'")));
    }

    #[test]
    fn a_null_with_no_explicit_entry_still_falls_back_to_value_to_string() {
        let frame = df! { "price" => vec![Some(1.0f64), None] }.unwrap();
        let formats = vec![format_with(&["price"], "{:num %.2f}", None)];

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["1.00", "null"]);
    }
}
