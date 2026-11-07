//! Break-label generation, delegated to [`zygos::LabelSpec`]
//!
//! ggsql's RENAMING clause maps to zygos's label spec: explicit
//! `'value' => 'label'` entries become renames/suppressions and the
//! wildcard `* => '...'` becomes the fallback template. Labels are
//! rendered against the *current* break set on every call, so
//! re-breaking re-expands templates and re-derives shared numeric
//! precision.
//!
//! Rename matching reproduces ggsql's historical string-key behaviour
//! on top of zygos's typed [`Value::key_eq`]: user rename keys come
//! from SQL literals (`'25'`, `'2024-01-01'`) while breaks are typed,
//! so a break matches a key either by `key_eq` or by comparing the
//! break's `to_key_string()` against the key's own string form.

use zygos::{LabelSpec, Value};

use crate::plot::{format_number, ArrayElement};

/// Convert a break element to the [`Value`] it is labelled as.
///
/// Numeric breaks stay numeric so `{}` shared precision and `{:num}`
/// work; temporal breaks go as their ISO key string, which zygos's
/// `{:time}` parses (and which rename keys written as ISO string
/// literals match by `key_eq`).
pub(crate) fn break_value(elem: &ArrayElement) -> Value {
    match elem {
        ArrayElement::String(s) => Value::String(s.as_str().into()),
        ArrayElement::Number(n) => Value::Number(*n),
        ArrayElement::Boolean(b) => Value::Bool(*b),
        ArrayElement::Null => Value::Null,
        ArrayElement::Date(_) | ArrayElement::DateTime(_) | ArrayElement::Time(_) => {
            Value::String(elem.to_key_string().into())
        }
    }
}

/// Convert a rename *key* element to a [`Value`], keeping the
/// literal's type: `'A'` stays a string, `25` a number, `NULL` null.
/// The typed rule lets an unquoted `10` match the string category
/// `"10"`; quoted keys still match through the text rule in
/// [`label_elements`].
pub(crate) fn key_value(elem: &ArrayElement) -> Value {
    match elem {
        ArrayElement::String(s) => Value::String(s.as_str().into()),
        ArrayElement::Number(n) => Value::Number(*n),
        ArrayElement::Boolean(b) => Value::Bool(*b),
        ArrayElement::Null => Value::Null,
        ArrayElement::Date(d) => Value::Date(*d),
        ArrayElement::DateTime(dt) => Value::DateTime(*dt),
        ArrayElement::Time(t) => Value::Time(*t),
    }
}

/// The string form of a rename key, for the text matching rule:
/// `'25'` renames numeric break 25 because the break's
/// `to_key_string()` equals the key's string.
fn key_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.to_string(),
        Value::Number(n) => format_number(*n),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        // Temporal and other keys never arise from RENAMING literals;
        // give them their default rendering so the rule stays total.
        other => zygos::default_label(other, None).unwrap_or_default(),
    }
}

/// Render the labels for a break set against a [`LabelSpec`], one
/// entry per break, aligned by index. `None` means the label is
/// suppressed.
///
/// Resolution order per break (ggplot2's `labels` semantics, extended
/// with ggsql's historical string-key matching):
///
/// 1. **Typed rename** — the break's [`break_value`] `key_eq`-matches
///    a rename key.
/// 2. **Text rename** — the break's `to_key_string()` equals a rename
///    key's string form.
/// 3. **Template** — the spec's template (default `{}`), with shared
///    numeric precision derived from the un-renamed breaks.
///
/// A bare null break with no rename renders `"null"`, as it always
/// has (zygos suppresses [`Value::Null`] by default).
pub fn label_elements(spec: &LabelSpec, breaks: &[ArrayElement]) -> Vec<Option<String>> {
    let values: Vec<Value> = breaks.iter().map(break_value).collect();

    // Resolve renames first: typed match, then the historical text rule.
    let renames: Vec<(&Value, Option<&str>)> = spec.renames().collect();
    let matched: Vec<Option<Option<String>>> = values
        .iter()
        .zip(breaks)
        .map(|(value, elem)| {
            if let Some(entry) = spec.lookup(value) {
                return Some(entry.map(str::to_string));
            }
            let key = elem.to_key_string();
            renames
                .iter()
                .find(|(k, _)| key_string(k) == key)
                .map(|(_, r)| r.map(str::to_string))
        })
        .collect();

    // Shared precision over the un-renamed breaks, for plain `{}`.
    let has_plain = spec.template().is_none_or(zygos::Template::has_plain);
    let precision = if has_plain {
        let unrenamed: Vec<Value> = values
            .iter()
            .zip(&matched)
            .filter(|(_, m)| m.is_none())
            .map(|(v, _)| v.clone())
            .collect();
        zygos::shared_precision(&unrenamed)
    } else {
        None
    };

    values
        .iter()
        .zip(breaks)
        .zip(&matched)
        .map(|((value, elem), m)| match m {
            Some(entry) => entry.clone(),
            None => {
                let rendered = match spec.template() {
                    Some(t) => t.format(value, precision),
                    None => zygos::default_label(value, precision),
                };
                match (rendered, elem) {
                    // zygos suppresses nulls; ggsql's bare null break
                    // has always rendered the text "null".
                    (None, ArrayElement::Null) => Some("null".to_string()),
                    (rendered, _) => rendered,
                }
            }
        })
        .collect()
}

/// Apply label formatting template to a DataFrame column.
///
/// Returns a new DataFrame with the specified column formatted according to the template.
///
/// # Arguments
/// * `df` - DataFrame containing the column to format
/// * `column_name` - Name of the column to format
/// * `template` - Template string with placeholders (e.g., "{:Title}", "{:num %.2f}")
///
/// # Returns
/// New DataFrame with formatted column
///
/// # Example
/// ```ignore
/// let formatted_df = format_dataframe_column(&df, "_aesthetic_label", "Region: {:Title}")?;
/// ```
pub fn format_dataframe_column(
    df: &crate::DataFrame,
    column_name: &str,
    template: &str,
) -> Result<crate::DataFrame, String> {
    use crate::array_util::{as_f64, as_str, cast_array, new_str_array};
    use arrow::array::Array;
    use arrow::datatypes::DataType;

    // Get the column
    let column = df
        .column(column_name)
        .map_err(|e| format!("Column '{}' not found: {}", column_name, e))?;

    // Convert the column to label values: strings stay strings,
    // numerics stay numeric so `{:num}` and `{}` work.
    let values: Vec<Option<Value>> = if let Ok(str_col) = as_str(column) {
        // String column (includes temporal data auto-converted to ISO format)
        (0..str_col.len())
            .map(|i| {
                if str_col.is_null(i) {
                    None
                } else {
                    Some(Value::String(str_col.value(i).into()))
                }
            })
            .collect()
    } else if let Ok(cast) = cast_array(column, &DataType::Float64) {
        let f64_col = as_f64(&cast).map_err(|e| format!("Failed to cast column to f64: {}", e))?;

        (0..f64_col.len())
            .map(|i| {
                if f64_col.is_null(i) {
                    None
                } else {
                    Some(Value::Number(f64_col.value(i)))
                }
            })
            .collect()
    } else {
        return Err(format!(
            "Formatting doesn't support type {:?} in column '{}'. Try string or numeric types instead.",
            column.data_type(),
            column_name
        ));
    };

    let template = zygos::Template::parse(template);
    let formatted_owned: Vec<Option<String>> = values
        .iter()
        .map(|opt| opt.as_ref().and_then(|v| template.format(v, None)))
        .collect();

    let formatted_refs: Vec<Option<&str>> =
        formatted_owned.iter().map(|opt| opt.as_deref()).collect();
    let formatted_col = new_str_array(formatted_refs);

    // Replace column in DataFrame
    df.with_column(column_name, formatted_col)
        .map_err(|e| format!("Failed to replace column: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with_template(template: &str) -> LabelSpec {
        let mut spec = LabelSpec::new();
        spec.set_template(template);
        spec
    }

    fn numbers(breaks: &[f64]) -> Vec<ArrayElement> {
        breaks.iter().map(|n| ArrayElement::Number(*n)).collect()
    }

    fn strings(breaks: &[&str]) -> Vec<ArrayElement> {
        breaks
            .iter()
            .map(|s| ArrayElement::String(s.to_string()))
            .collect()
    }

    fn labels(spec: &LabelSpec, breaks: &[ArrayElement]) -> Vec<Option<String>> {
        label_elements(spec, breaks)
    }

    #[test]
    fn test_plain_placeholder() {
        let result = labels(&LabelSpec::new(), &numbers(&[0.0, 25.0, 50.0]));
        assert_eq!(
            result,
            vec![
                Some("0".to_string()),
                Some("25".to_string()),
                Some("50".to_string())
            ]
        );
    }

    #[test]
    fn test_upper_placeholder() {
        let result = labels(
            &spec_with_template("{:UPPER}"),
            &strings(&["north", "south"]),
        );
        assert_eq!(
            result,
            vec![Some("NORTH".to_string()), Some("SOUTH".to_string())]
        );
    }

    #[test]
    fn test_lower_placeholder() {
        let result = labels(
            &spec_with_template("{:lower}"),
            &strings(&["HELLO", "WORLD"]),
        );
        assert_eq!(
            result,
            vec![Some("hello".to_string()), Some("world".to_string())]
        );
    }

    #[test]
    fn test_title_placeholder() {
        let result = labels(
            &spec_with_template("Region: {:Title}"),
            &strings(&["us east", "eu west"]),
        );
        assert_eq!(
            result,
            vec![
                Some("Region: Us East".to_string()),
                Some("Region: Eu West".to_string())
            ]
        );
    }

    #[test]
    fn test_datetime_placeholder() {
        let result = labels(
            &spec_with_template("{:time %b %Y}"),
            &strings(&["2024-01-15", "2024-02-15"]),
        );
        assert_eq!(
            result,
            vec![Some("Jan 2024".to_string()), Some("Feb 2024".to_string())]
        );
    }

    #[test]
    fn test_rename_takes_priority() {
        let mut spec = spec_with_template("Category {}");
        spec.rename(Value::String("A".into()), "Alpha");
        let result = labels(&spec, &strings(&["A", "B", "C"]));
        assert_eq!(
            result,
            vec![
                Some("Alpha".to_string()),
                Some("Category B".to_string()),
                Some("Category C".to_string())
            ]
        );
    }

    #[test]
    fn test_quoted_number_key_matches_numeric_break() {
        // The historical text rule: '25' renames numeric break 25.
        let mut spec = LabelSpec::new();
        spec.rename(Value::String("25".into()), "twenty-five");
        let result = labels(&spec, &numbers(&[25.0, 50.0]));
        assert_eq!(
            result,
            vec![Some("twenty-five".to_string()), Some("50".to_string())]
        );
    }

    #[test]
    fn test_unquoted_number_key_matches_string_category() {
        // The typed rule: 10 renames the string category "10".
        let mut spec = LabelSpec::new();
        spec.rename(Value::Number(10.0), "ten");
        let result = labels(&spec, &strings(&["10", "20"]));
        assert_eq!(
            result,
            vec![Some("ten".to_string()), Some("20".to_string())]
        );
    }

    #[test]
    fn test_multiple_placeholders() {
        let result = labels(&spec_with_template("{} - {:UPPER}"), &strings(&["hello"]));
        assert_eq!(result, vec![Some("hello - HELLO".to_string())]);
    }

    #[test]
    fn test_no_placeholder_literal() {
        let result = labels(&spec_with_template("Constant Label"), &strings(&["A", "B"]));
        assert_eq!(
            result,
            vec![
                Some("Constant Label".to_string()),
                Some("Constant Label".to_string())
            ]
        );
    }

    #[test]
    fn test_datetime_with_time() {
        let result = labels(
            &spec_with_template("{:time %Y-%m-%d %H:%M}"),
            &strings(&["2024-01-15T10:30:00"]),
        );
        assert_eq!(result, vec![Some("2024-01-15 10:30".to_string())]);
    }

    #[test]
    fn test_invalid_datetime_fallback() {
        let result = labels(
            &spec_with_template("{:time %Y-%m-%d}"),
            &strings(&["not-a-date"]),
        );
        assert_eq!(result, vec![Some("not-a-date".to_string())]);
    }

    #[test]
    fn test_null_renders_null_text() {
        let result = labels(
            &LabelSpec::new(),
            &[
                ArrayElement::String("A".to_string()),
                ArrayElement::Null,
                ArrayElement::String("B".to_string()),
            ],
        );
        assert_eq!(
            result,
            vec![
                Some("A".to_string()),
                Some("null".to_string()),
                Some("B".to_string())
            ]
        );
    }

    #[test]
    fn test_null_rename_by_null_literal_key() {
        let mut spec = LabelSpec::new();
        spec.rename(Value::Null, "The rest");
        let result = labels(&spec, &[ArrayElement::Null]);
        assert_eq!(result, vec![Some("The rest".to_string())]);
    }

    #[test]
    fn test_number_format_decimal_places() {
        let result = labels(
            &spec_with_template("${:num %.2f}"),
            &numbers(&[25.5, 100.0]),
        );
        assert_eq!(
            result,
            vec![Some("$25.50".to_string()), Some("$100.00".to_string())]
        );
    }

    #[test]
    fn test_number_format_no_decimals() {
        let result = labels(&spec_with_template("{:num %.0f} items"), &numbers(&[25.7]));
        assert_eq!(result, vec![Some("26 items".to_string())]);
    }

    #[test]
    fn test_number_format_scientific() {
        let result = labels(&spec_with_template("{:num %.2e}"), &numbers(&[1234.5]));
        assert_eq!(result, vec![Some("1.23e+03".to_string())]);
    }

    #[test]
    fn test_number_format_non_numeric_fallback() {
        let result = labels(&spec_with_template("{:num %.2f}"), &strings(&["hello"]));
        assert_eq!(result, vec![Some("hello".to_string())]);
    }

    #[test]
    fn test_number_format_integer() {
        let result = labels(&spec_with_template("{:num %d}"), &numbers(&[42.0]));
        assert_eq!(result, vec![Some("42".to_string())]);
    }

    #[test]
    fn test_plain_placeholder_consistent_decimal_precision() {
        let result = labels(&LabelSpec::new(), &numbers(&[54.5, 55.0, 55.5]));
        assert_eq!(
            result,
            vec![
                Some("54.5".to_string()),
                Some("55.0".to_string()),
                Some("55.5".to_string())
            ]
        );
    }

    #[test]
    fn test_plain_placeholder_all_integers_no_decimals() {
        let result = labels(&LabelSpec::new(), &numbers(&[10.0, 20.0, 30.0]));
        assert_eq!(
            result,
            vec![
                Some("10".to_string()),
                Some("20".to_string()),
                Some("30".to_string())
            ]
        );
    }

    #[test]
    fn test_plain_placeholder_small_fractional_steps() {
        let result = labels(&LabelSpec::new(), &numbers(&[0.0, 0.05, 0.1]));
        assert_eq!(
            result,
            vec![
                Some("0.00".to_string()),
                Some("0.05".to_string()),
                Some("0.10".to_string())
            ]
        );
    }

    #[test]
    fn test_suppression() {
        let mut spec = LabelSpec::new();
        spec.suppress(Value::String("B".into()));
        let result = labels(&spec, &strings(&["A", "B", "C"]));
        assert_eq!(
            result,
            vec![Some("A".to_string()), None, Some("C".to_string())]
        );
    }
}
