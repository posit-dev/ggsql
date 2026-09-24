//! `TABULATE FORMAT` resolution: validating `Table.formats`' `SETTING`
//! parameters and reshaping it (one entry per `FORMAT` clause, each naming
//! several columns) into one `Format` per column (`setup_formats`), then
//! applying that per column — `RENAMING` by replacing the column's values in
//! the `DataFrame` (`apply_formats`), `SETTING` by resolving display
//! properties for `TableColumn` (`resolve_column_properties`). `setup_formats`
//! and `apply_formats` are both called from `table::resolve_table_with_reader`.

use std::collections::HashMap;

use arrow::datatypes::DataType;

use crate::array_util::{new_str_array, value_to_string};
use crate::plot::{ParameterValue, Parameters};
use crate::{DataFrame, Format, GgsqlError, Result};

/// Validate every FORMAT's `SETTING` parameters, then reshape `formats`
/// into one `Format` per column it covers.
pub(super) fn setup_formats(df: &DataFrame, formats: &[Format]) -> Result<HashMap<String, Format>> {
    for (idx, format) in formats.iter().enumerate() {
        format
            .validate_settings()
            .map_err(|e| GgsqlError::ValidationError(format!("FORMAT {}: {}", idx + 1, e)))?;
    }

    // `formats` has one entry per FORMAT clause, each naming several
    // columns — reshaped here into one `Format` per column, where a later
    // clause wins over an earlier one naming the same column.
    let mut resolved = HashMap::new();
    for format in formats {
        for name in &format.columns {
            if df.column(name).is_err() {
                return Err(GgsqlError::ValidationError(format!(
                    "FORMAT references unknown column '{name}'"
                )));
            }
            let mut format = format.clone();
            // Redundant now: this map's own key names the column instead.
            format.columns = Vec::new();
            resolved.insert(name.clone(), format);
        }
    }
    Ok(resolved)
}

/// Replace every column in `df` named in `formats` with its resolved
/// display text. A column whose `Format` has no `RENAMING` at all
/// (`value_template` is the default `"{}"` and `value_mapping` is empty) is
/// left untouched.
pub(super) fn apply_formats(
    df: &DataFrame,
    formats: &HashMap<String, Format>,
) -> Result<DataFrame> {
    let mut df = df.clone();
    for (name, format) in formats {
        let has_renaming = format.value_template != "{}"
            || format.value_mapping.as_ref().is_some_and(|m| !m.is_empty());
        if !has_renaming {
            // Skip the per-row formatting pass for a column nobody asked to
            // reformat — the common `FORMAT ... SETTING ...` case.
            continue;
        }

        let array = df
            .column(name)
            .expect("setup_formats already checked this column exists")
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

/// Resolve `SETTING` properties for one column: `format`'s own settings (if
/// any), with `hjust` standardised to a number (see `standardise_hjust`);
/// an absent setting defaults from `dtype` (numeric columns right,
/// everything else left). Every writer reads a plain number for `hjust` and
/// buckets it into left/center/right itself — none of them see the keyword
/// form.
pub(super) fn resolve_column_properties(dtype: &DataType, format: Option<&Format>) -> Parameters {
    let mut properties = format.map(|f| f.settings.clone()).unwrap_or_default();

    let hjust = properties
        .get("hjust")
        .and_then(standardise_hjust)
        .unwrap_or(if dtype.is_numeric() { 1.0 } else { 0.0 });
    properties.insert("hjust".to_string(), ParameterValue::Number(hjust));

    properties
}

/// Standardise an `hjust` value to a number: the keywords `"left"`,
/// `"center"`/`"centre"` and `"right"` become `0.0`, `0.5` and `1.0` (any
/// other string is `0.5` — validation rejects unrecognized spellings before
/// this runs); a number passes through unchanged; anything else is `None`.
pub(super) fn standardise_hjust(value: &ParameterValue) -> Option<f64> {
    match value {
        ParameterValue::String(s) if s == "left" => Some(0.0),
        ParameterValue::String(s) if s == "right" => Some(1.0),
        ParameterValue::String(_) => Some(0.5),
        ParameterValue::Number(n) => Some(*n),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_util::value_to_string;
    use crate::df;

    fn format_with(template: &str, mapping: Option<HashMap<String, Option<String>>>) -> Format {
        Format {
            columns: Vec::new(),
            settings: Parameters::new(),
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
    fn setup_formats_lets_a_later_clause_win_and_clears_columns() {
        let frame = df! { "price" => vec![1.0f64] }.unwrap();
        let formats = vec![
            Format {
                columns: vec!["price".to_string()],
                ..format_with("{:num %.0f}", None)
            },
            Format {
                columns: vec!["price".to_string()],
                ..format_with("{:num %.2f}", None)
            },
        ];

        let resolved = setup_formats(&frame, &formats).unwrap();

        let price = resolved.get("price").unwrap();
        assert_eq!(price.value_template, "{:num %.2f}");
        assert!(price.columns.is_empty());
    }

    #[test]
    fn setup_formats_errors_on_an_unknown_column() {
        let frame = df! { "price" => vec![1.0f64] }.unwrap();
        let formats = vec![Format {
            columns: vec!["typo".to_string()],
            ..format_with("{}", None)
        }];

        let error = setup_formats(&frame, &formats).unwrap_err();
        assert!(matches!(error, GgsqlError::ValidationError(msg)
            if msg.contains("FORMAT references unknown column 'typo'")));
    }

    #[test]
    fn leaves_a_column_untouched_when_its_format_has_no_renaming() {
        let frame = df! { "price" => vec![1.0f64, 2.0] }.unwrap();
        let formats = HashMap::from([("price".to_string(), format_with("{}", None))]);

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["1", "2"]);
    }

    #[test]
    fn applies_a_wildcard_template() {
        let frame = df! { "price" => vec![1.0f64, 2.0] }.unwrap();
        let formats = HashMap::from([("price".to_string(), format_with("${:num %.2f}", None))]);

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["$1.00", "$2.00"]);
    }

    #[test]
    fn explicit_mapping_wins_over_the_template() {
        let frame = df! { "price" => vec![0.0f64, 5.0] }.unwrap();
        let mapping = Some(HashMap::from([("0".to_string(), Some("-".to_string()))]));
        let formats = HashMap::from([("price".to_string(), format_with("{:num %.2f}", mapping))]);

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["-", "5.00"]);
    }

    #[test]
    fn a_null_with_no_explicit_entry_still_falls_back_to_value_to_string() {
        let frame = df! { "price" => vec![Some(1.0f64), None] }.unwrap();
        let formats = HashMap::from([("price".to_string(), format_with("{:num %.2f}", None))]);

        let result = apply_formats(&frame, &formats).unwrap();

        assert_eq!(column_content(&result, "price"), vec!["1.00", "null"]);
    }

    fn format_with_hjust(hjust: ParameterValue) -> Format {
        let mut settings = Parameters::new();
        settings.insert("hjust".to_string(), hjust);
        Format {
            columns: Vec::new(),
            settings,
            value_mapping: None,
            value_template: "{}".to_string(),
        }
    }

    #[test]
    fn resolve_column_properties_defaults_hjust_by_dtype_when_absent() {
        assert_eq!(
            resolve_column_properties(&DataType::Float64, None).get("hjust"),
            Some(&ParameterValue::Number(1.0))
        );
        assert_eq!(
            resolve_column_properties(&DataType::Utf8, None).get("hjust"),
            Some(&ParameterValue::Number(0.0))
        );
    }

    #[test]
    fn resolve_column_properties_standardises_keywords_to_numbers() {
        let format = format_with_hjust(ParameterValue::String("left".to_string()));
        assert_eq!(
            resolve_column_properties(&DataType::Utf8, Some(&format)).get("hjust"),
            Some(&ParameterValue::Number(0.0))
        );

        let format = format_with_hjust(ParameterValue::String("centre".to_string()));
        assert_eq!(
            resolve_column_properties(&DataType::Utf8, Some(&format)).get("hjust"),
            Some(&ParameterValue::Number(0.5))
        );

        let format = format_with_hjust(ParameterValue::String("right".to_string()));
        assert_eq!(
            resolve_column_properties(&DataType::Utf8, Some(&format)).get("hjust"),
            Some(&ParameterValue::Number(1.0))
        );
    }

    #[test]
    fn resolve_column_properties_keeps_an_explicit_number_unchanged() {
        let format = format_with_hjust(ParameterValue::Number(0.25));

        assert_eq!(
            resolve_column_properties(&DataType::Float64, Some(&format)).get("hjust"),
            Some(&ParameterValue::Number(0.25))
        );
    }
}
