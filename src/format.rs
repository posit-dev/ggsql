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

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, UInt64Array};
use arrow::datatypes::DataType;
use zygos::{LabelSpec, Value};

use crate::array_util::{as_f32, as_f64, as_str, cast_array, new_str_array};
use crate::plot::ArrayElement;

/// Serde mirror for [`zygos::LabelSpec`], which is dependency-free and has no
/// serde impls of its own. Rename keys round-trip as [`ArrayElement`]s under
/// the same key-typing rule the parser uses (`'A'` → string, `25` → number,
/// `NULL` → null); the template round-trips as its source text. Shared by
/// every spec type that stores a `LabelSpec` (`Scale::label_spec`,
/// `Format::renaming`).
pub(crate) mod label_spec_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use zygos::{LabelSpec, Value};

    use crate::plot::ArrayElement;

    #[derive(Serialize, Deserialize, Default)]
    struct LabelSpecSerde {
        #[serde(default)]
        renames: Vec<(ArrayElement, Option<String>)>,
        #[serde(default)]
        template: Option<String>,
    }

    fn value_to_element(value: &Value) -> ArrayElement {
        match value {
            Value::String(s) => ArrayElement::String(s.to_string()),
            Value::Number(n) => ArrayElement::Number(*n),
            Value::Bool(b) => ArrayElement::Boolean(*b),
            Value::Null => ArrayElement::Null,
            // Only RENAMING literals reach a LabelSpec — string, number,
            // boolean, null. Anything else stringifies.
            other => ArrayElement::String(zygos::default_label(other, None).unwrap_or_default()),
        }
    }

    pub fn serialize<S: Serializer>(spec: &LabelSpec, serializer: S) -> Result<S::Ok, S::Error> {
        let mirror = LabelSpecSerde {
            renames: spec
                .renames()
                .map(|(k, r)| (value_to_element(k), r.map(str::to_string)))
                .collect(),
            template: spec.template().map(|t| t.source().to_string()),
        };
        mirror.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<LabelSpec, D::Error> {
        let mirror = LabelSpecSerde::deserialize(deserializer)?;
        let mut spec = LabelSpec::new();
        for (key, rename) in mirror.renames {
            let value = crate::format::key_value(&key);
            match rename {
                Some(label) => spec.rename(value, label),
                None => spec.suppress(value),
            };
        }
        if let Some(template) = mirror.template {
            spec.set_template(&template);
        }
        Ok(spec)
    }
}

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
        Value::Number(n) => n.to_string(),
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
    // Scale labels share numeric precision across the break set; a
    // suppressed label is dropped (`None`); a bare null renders "null".
    render_labels(spec, &values, true, None, Some("null"))
}

/// Resolve one display text per row of a column, for `TABULATE FORMAT`'s
/// `RENAMING` clause: an explicit rename wins outright (a suppression
/// blanks the cell's text); every other value goes through the spec's
/// template or default rendering, except a null with no explicit `NULL`
/// entry, which is left unresolved.
///
/// `Ok(None)` at a row means "no resolution, use the caller's own default
/// content for that row"; `Ok(Some(s))` (possibly empty) is the resolved
/// display text.
pub fn resolve_column_values(
    column: &ArrayRef,
    spec: &LabelSpec,
) -> Result<Vec<Option<String>>, String> {
    let values = column_to_values(column)?;
    // Table cells render per value without shared precision; a
    // suppression blanks the cell; a bare null stays unresolved so the
    // caller falls back to its own default rendering.
    Ok(render_labels(spec, &values, false, Some(""), None))
}

/// Like [`resolve_column_values`], but consulting only the spec's direct
/// renames — the wildcard template is ignored. For already-rendered text
/// (a `SUMMARISE` template cell), where re-rendering a string through a
/// numeric template is meaningless.
pub fn resolve_column_values_direct(
    column: &ArrayRef,
    spec: &LabelSpec,
) -> Result<Vec<Option<String>>, String> {
    let mut direct = LabelSpec::new();
    for (key, label) in spec.renames() {
        match label {
            Some(label) => direct.rename(key.clone(), label),
            None => direct.suppress(key.clone()),
        };
    }
    resolve_column_values(column, &direct)
}

/// Shared label rendering behind [`label_elements`] and
/// [`resolve_column_values`]. `share_precision` derives `{}` decimal
/// precision across the un-renamed values (scale breaks) or renders each
/// value on its own (table cells). `suppressed` and `bare_null` give the
/// caller's rendering of a suppressed entry and of an un-renamed null;
/// `None` leaves the slot `None`.
fn render_labels(
    spec: &LabelSpec,
    values: &[Value],
    share_precision: bool,
    suppressed: Option<&str>,
    bare_null: Option<&str>,
) -> Vec<Option<String>> {
    // Resolve renames first: typed match, then the historical text rule.
    let renames: Vec<(&Value, Option<&str>)> = spec.renames().collect();
    let matched: Vec<Option<Option<String>>> = values
        .iter()
        .map(|value| {
            if let Some(entry) = spec.lookup(value) {
                return Some(entry.map(str::to_string));
            }
            let key = key_string(value);
            renames
                .iter()
                .find(|(k, _)| key_string(k) == key)
                .map(|(_, r)| r.map(str::to_string))
        })
        .collect();

    // Shared precision over the un-renamed values, for plain `{}`.
    let has_plain = spec.template().is_none_or(zygos::Template::has_plain);
    let precision = if share_precision && has_plain {
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
        .zip(&matched)
        .map(|(value, m)| match m {
            Some(entry) => entry.clone().or_else(|| suppressed.map(str::to_string)),
            None => {
                let rendered = match spec.template() {
                    Some(t) => t.format(value, precision),
                    None => zygos::default_label(value, precision),
                };
                match (rendered, value) {
                    // zygos suppresses nulls; a bare null gets the
                    // caller's fallback, if any.
                    (None, Value::Null) => bare_null.map(str::to_string),
                    (rendered, _) => rendered,
                }
            }
        })
        .collect()
}

/// Convert an Arrow column to one zygos [`Value`] per row ([`Value::Null`]
/// for a null). Floats and exactly representable integers stay numeric so
/// `{:num}` works; larger integers keep their exact text form; temporal
/// values and strings go as their display text (ISO for temporal, which
/// `{:time}` parses).
fn column_to_values(column: &ArrayRef) -> Result<Vec<Value>, String> {
    // Exactness cut-off: integers beyond 2^53 are not representable as
    // the f64 inside `Value::Number`, so they stay text.
    const MAX_EXACT: i64 = 1 << 53;

    match column.data_type() {
        DataType::Int64 => {
            let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok((0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        Value::Null
                    } else {
                        let v = array.value(i);
                        if v.unsigned_abs() <= MAX_EXACT as u64 {
                            Value::Number(v as f64)
                        } else {
                            Value::String(v.to_string().into())
                        }
                    }
                })
                .collect())
        }
        DataType::UInt64 => {
            let array = column.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok((0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        Value::Null
                    } else {
                        let v = array.value(i);
                        if v <= MAX_EXACT as u64 {
                            Value::Number(v as f64)
                        } else {
                            Value::String(v.to_string().into())
                        }
                    }
                })
                .collect())
        }
        dtype if dtype.is_numeric() => {
            let cast = cast_array(column, &DataType::Float64)
                .map_err(|e| format!("Failed to cast column to f64: {e}"))?;
            let f64_col = as_f64(&cast).map_err(|e| e.to_string())?;
            Ok((0..f64_col.len())
                .map(|i| {
                    if f64_col.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(f64_col.value(i))
                    }
                })
                .collect())
        }
        DataType::Boolean => {
            let array = column.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok((0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        Value::Null
                    } else {
                        Value::Bool(array.value(i))
                    }
                })
                .collect())
        }
        dtype => {
            // Strings render as themselves; everything else Arrow can
            // display (temporal renders ISO text).
            if let Ok(str_col) = as_str(column) {
                return Ok((0..str_col.len())
                    .map(|i| {
                        if str_col.is_null(i) {
                            Value::Null
                        } else {
                            Value::String(str_col.value(i).into())
                        }
                    })
                    .collect());
            }
            let formatter =
                arrow::util::display::ArrayFormatter::try_new(column.as_ref(), &Default::default())
                    .map_err(|e| format!("Formatting doesn't support type {dtype:?}: {e}"))?;
            Ok((0..column.len())
                .map(|i| {
                    if column.is_null(i) {
                        Value::Null
                    } else {
                        Value::String(formatter.value(i).to_string().into())
                    }
                })
                .collect())
        }
    }
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

/// Decimal places to display a float column consistently: the most any
/// non-null value needs, each value asking for the fewest decimals that
/// round-trip within float noise (`f64::EPSILON.sqrt()`, relative).
/// Non-float columns need none.
pub fn display_decimals(column: &ArrayRef) -> usize {
    let decimals: Box<dyn Iterator<Item = f64> + '_> = match column.data_type() {
        DataType::Float64 => Box::new(
            as_f64(column)
                .unwrap()
                .iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        ),
        DataType::Float32 => Box::new(
            as_f32(column)
                .unwrap()
                .iter()
                .map(|v| f64::from(v.unwrap_or(f32::NAN))),
        ),
        _ => return 0,
    };
    // Search each value only upward from the running maximum: a value
    // fitting at fewer decimals cannot lower it, and a fit at `d` fits at
    // every larger `d`, so nothing below the maximum needs checking.
    let mut max = 0;
    for v in decimals {
        max = minimal_decimals(v, max);
        if max == 15 {
            break;
        }
    }
    max
}

/// Fewest decimal places that represent `v` within float noise: the
/// smallest `d >= from` whose rounded value is within
/// `f64::EPSILON.sqrt()` (relative) of `v`. Caps at 15, past which an f64
/// carries no real information. Non-finite values contribute nothing.
fn minimal_decimals(v: f64, from: usize) -> usize {
    if !v.is_finite() {
        return from;
    }
    let tol = f64::EPSILON.sqrt() * v.abs().max(1.0);
    (from..=15)
        .find(|&d| {
            let factor = 10f64.powi(d as i32);
            ((v * factor).round() / factor - v).abs() <= tol
        })
        .unwrap_or(15)
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

    fn floats(values: &[f64]) -> ArrayRef {
        std::sync::Arc::new(arrow::array::Float64Array::from(values.to_vec()))
    }

    #[test]
    fn display_decimals_pads_to_the_widest_value() {
        assert_eq!(display_decimals(&floats(&[39.1, 37.7, 50.0])), 1);
        assert_eq!(display_decimals(&floats(&[1.25, 3.75])), 2);
    }

    #[test]
    fn display_decimals_trims_binary_float_noise() {
        assert_eq!(display_decimals(&floats(&[0.1 + 0.2, 1.5])), 1);
        assert_eq!(display_decimals(&floats(&[std::f64::consts::PI])), 7);
    }

    #[test]
    fn display_decimals_is_zero_for_whole_and_non_float_columns() {
        assert_eq!(display_decimals(&floats(&[50.0, 60.0])), 0);
        let ints: ArrayRef = std::sync::Arc::new(arrow::array::Int64Array::from(vec![1, 2]));
        assert_eq!(display_decimals(&ints), 0);
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

    #[test]
    fn resolve_column_values_null_with_explicit_entry_is_resolved() {
        use crate::array_util::new_f64_array;

        let column = new_f64_array(vec![None, Some(1.0)]);
        let mut spec = spec_with_template("{:num %.0f}");
        spec.suppress(Value::Null); // NULL => NULL suppresses

        let result = resolve_column_values(&column, &spec).unwrap();

        assert_eq!(result[0], Some(String::new())); // suppressed, not templated
        assert_eq!(result[1], Some("1".to_string()));
    }

    #[test]
    fn resolve_column_values_formats_temporal_columns_as_iso_text() {
        use arrow::array::Date32Array;
        use std::sync::Arc;

        // 19737 days after the epoch
        let column: ArrayRef = Arc::new(Date32Array::from(vec![Some(19737)]));
        let spec = spec_with_template("{:time %b %Y}");
        let result = resolve_column_values(&column, &spec).unwrap();
        assert_eq!(result, vec![Some("Jan 2024".to_string())]);

        let result = resolve_column_values(&column, &LabelSpec::new()).unwrap();
        assert_eq!(result, vec![Some("2024-01-15".to_string())]);
    }

    #[test]
    fn resolve_column_values_formats_timestamps_for_time_placeholders() {
        use arrow::array::TimestampMicrosecondArray;
        use std::sync::Arc;

        // 2024-01-15 10:30:00 UTC, in microseconds
        let column: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(
            1_705_315_800_000_000,
        )]));
        let spec = spec_with_template("{:time %b %Y}");
        let result = resolve_column_values(&column, &spec).unwrap();
        assert_eq!(result, vec![Some("Jan 2024".to_string())]);
    }

    #[test]
    fn resolve_column_values_keeps_large_integers_exact() {
        use arrow::array::Int64Array;
        use std::sync::Arc;

        // Beyond 2^53: not representable as f64
        let column: ArrayRef = Arc::new(Int64Array::from(vec![9007199254740993]));
        let result = resolve_column_values(&column, &LabelSpec::new()).unwrap();
        assert_eq!(result, vec![Some("9007199254740993".to_string())]);
    }

    #[test]
    fn resolve_column_values_renders_booleans_as_true_and_false() {
        use arrow::array::BooleanArray;
        use std::sync::Arc;

        let column: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let result = resolve_column_values(&column, &LabelSpec::new()).unwrap();
        assert_eq!(
            result,
            vec![Some("true".to_string()), Some("false".to_string())]
        );
    }

    #[test]
    fn resolve_column_values_does_not_align_precision_across_cells() {
        use crate::array_util::new_f64_array;

        let column = new_f64_array(vec![Some(1.5), Some(3.0), Some(3.0001)]);
        let result = resolve_column_values(&column, &LabelSpec::new()).unwrap();
        assert_eq!(
            result,
            vec![
                Some("1.5".to_string()),
                Some("3".to_string()),
                Some("3.0001".to_string())
            ]
        );
    }

    #[test]
    fn resolve_column_values_leaves_numeric_looking_strings_alone() {
        use crate::array_util::new_str_array;

        let column = new_str_array(vec![Some("007"), Some("3.0001")]);
        let result = resolve_column_values(&column, &LabelSpec::new()).unwrap();
        assert_eq!(
            result,
            vec![Some("007".to_string()), Some("3.0001".to_string())]
        );
    }

    #[test]
    fn resolve_column_values_null_without_an_explicit_entry_is_left_unresolved() {
        use crate::array_util::new_f64_array;

        let column = new_f64_array(vec![None, Some(1.0)]);
        let spec = spec_with_template("{:num %.2f}");

        let result = resolve_column_values(&column, &spec).unwrap();

        // Caller falls back to its own default rendering for this row.
        assert_eq!(result[0], None);
        assert_eq!(result[1], Some("1.00".to_string()));
    }
}
