//! Template-based label generation for scale RENAMING wildcards
//!
//! Supports placeholder syntax in templates:
//! - `{}` - Insert value as-is
//! - `{:UPPER}` - Convert to UPPERCASE
//! - `{:lower}` - Convert to lowercase
//! - `{:Title}` - Convert to Title Case
//! - `{:time %fmt}` - DateTime strftime format (e.g., `{:time %b %Y}` -> "Jan 2024")
//! - `{:num %fmt}` - Number printf format (e.g., `{:num %.2f}` -> "25.50")

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::plot::ArrayElement;

/// Placeholder types supported in label templates
#[derive(Debug, Clone)]
enum Placeholder {
    /// `{}` - Insert value as-is
    Plain,
    /// `{:UPPER}` - Convert to UPPERCASE
    Upper,
    /// `{:lower}` - Convert to lowercase
    Lower,
    /// `{:Title}` - Convert to Title Case
    Title,
    /// `{:time %Y-%m-%d}` or similar strftime format
    DateTime(String),
    /// `{:num %.2f}` or similar printf format
    Number(String),
}

/// Parsed placeholder with its full match text
#[derive(Debug, Clone)]
struct ParsedPlaceholder {
    placeholder: Placeholder,
    match_text: String,
}

/// Regex for matching placeholders in templates
fn placeholder_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{([^}]*)\}").expect("Invalid placeholder regex"))
}

/// Parse all placeholders from a template string
fn parse_placeholders(template: &str) -> Vec<ParsedPlaceholder> {
    placeholder_regex()
        .find_iter(template)
        .map(|cap| {
            let inner = &template[cap.start() + 1..cap.end() - 1];
            let placeholder = match inner {
                "" => Placeholder::Plain,
                ":UPPER" => Placeholder::Upper,
                ":lower" => Placeholder::Lower,
                ":Title" => Placeholder::Title,
                s if s.starts_with(":time ") => {
                    Placeholder::DateTime(s.strip_prefix(":time ").unwrap().to_string())
                }
                s if s.starts_with(":num ") => {
                    Placeholder::Number(s.strip_prefix(":num ").unwrap().to_string())
                }
                _ => Placeholder::Plain, // Unknown, treat as plain
            };
            ParsedPlaceholder {
                placeholder,
                match_text: cap.as_str().to_string(),
            }
        })
        .collect()
}

/// Apply transformation based on placeholder type
fn apply_transformation(value: &str, placeholder: &Placeholder) -> String {
    match placeholder {
        Placeholder::Plain => value.to_string(),
        Placeholder::Upper => value.to_uppercase(),
        Placeholder::Lower => value.to_lowercase(),
        Placeholder::Title => to_title_case(value),
        Placeholder::DateTime(fmt) => format_datetime(value, fmt),
        Placeholder::Number(fmt) => format_number_with_spec(value, fmt),
    }
}

/// Convert string to Title Case
fn to_title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format datetime string using strftime format
fn format_datetime(value: &str, fmt: &str) -> String {
    // Try parsing as NaiveDateTime first (various formats)
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return dt.format(fmt).to_string();
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f") {
        return dt.format(fmt).to_string();
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return dt.format(fmt).to_string();
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
        return dt.format(fmt).to_string();
    }
    // Try parsing as NaiveDate
    if let Ok(d) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return d.format(fmt).to_string();
    }
    // Try parsing as NaiveTime
    if let Ok(d) = NaiveTime::parse_from_str(value, "%H:%M:%S") {
        return d.format(fmt).to_string();
    }
    if let Ok(d) = NaiveTime::parse_from_str(value, "%H:%M:%S%.f") {
        return d.format(fmt).to_string();
    }
    // Fallback: return original value if parsing fails
    value.to_string()
}

/// Format number using printf-style format specifier (e.g., "%.2f", "%d", "%e")
fn format_number_with_spec(value: &str, fmt: &str) -> String {
    // Try to parse as f64
    if let Ok(n) = value.parse::<f64>() {
        // Use sprintf crate for full printf compatibility
        // Supports: %d, %i, %f, %e, %E, %g, %G, %o, %x, %X, width, precision, flags
        return sprintf::sprintf!(fmt, n).unwrap_or_else(|_| value.to_string());
    }
    // Fallback: return original value if parsing fails
    value.to_string()
}

/// Apply a label template to an array of break values.
///
/// Each break value is formatted using the template string.
/// Explicit mappings (from existing label_mapping) take priority over template-generated ones.
///
/// # Arguments
/// * `breaks` - Array elements to apply template to
/// * `template` - Template string with placeholders (e.g., "{} units", "{:UPPER}")
/// * `existing` - Optional existing label mappings (explicit mappings take priority)
///
/// # Returns
/// HashMap of original value -> formatted label
///
/// # Example
/// ```ignore
/// let breaks = vec![ArrayElement::Number(0.0), ArrayElement::Number(25.0)];
/// let result = apply_label_template(&breaks, "{} units", &None);
/// // result: {"0" => Some("0 units"), "25" => Some("25 units")}
/// ```
pub fn apply_label_template(
    breaks: &[ArrayElement],
    template: &str,
    existing: &Option<HashMap<String, Option<String>>>,
) -> HashMap<String, Option<String>> {
    let mut result = existing.clone().unwrap_or_default();

    // Parse all placeholders once
    let placeholders = parse_placeholders(template);

    for elem in breaks {
        // Skip null values
        if matches!(elem, ArrayElement::Null) {
            continue;
        }
        let key = elem.to_key_string();

        let break_val = key.clone();
        // Only apply template if no explicit mapping exists
        result.entry(key).or_insert_with(|| {
            let label = if placeholders.is_empty() {
                // No placeholders - use template as literal string
                template.to_string()
            } else {
                // Replace each placeholder with its transformed value
                // Process in reverse order to preserve string indices
                let mut label = template.to_string();
                for parsed in placeholders.iter().rev() {
                    let transformed = apply_transformation(&break_val, &parsed.placeholder);
                    label = label.replace(&parsed.match_text, &transformed);
                }
                label
            };
            Some(label)
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_placeholder() {
        let breaks = vec![
            ArrayElement::Number(0.0),
            ArrayElement::Number(25.0),
            ArrayElement::Number(50.0),
        ];
        let result = apply_label_template(&breaks, "{} units", &None);

        assert_eq!(result.get("0"), Some(&Some("0 units".to_string())));
        assert_eq!(result.get("25"), Some(&Some("25 units".to_string())));
        assert_eq!(result.get("50"), Some(&Some("50 units".to_string())));
    }

    #[test]
    fn test_upper_placeholder() {
        let breaks = vec![
            ArrayElement::String("north".to_string()),
            ArrayElement::String("south".to_string()),
        ];
        let result = apply_label_template(&breaks, "{:UPPER}", &None);

        assert_eq!(result.get("north"), Some(&Some("NORTH".to_string())));
        assert_eq!(result.get("south"), Some(&Some("SOUTH".to_string())));
    }

    #[test]
    fn test_lower_placeholder() {
        let breaks = vec![
            ArrayElement::String("HELLO".to_string()),
            ArrayElement::String("WORLD".to_string()),
        ];
        let result = apply_label_template(&breaks, "{:lower}", &None);

        assert_eq!(result.get("HELLO"), Some(&Some("hello".to_string())));
        assert_eq!(result.get("WORLD"), Some(&Some("world".to_string())));
    }

    #[test]
    fn test_title_placeholder() {
        let breaks = vec![
            ArrayElement::String("us east".to_string()),
            ArrayElement::String("eu west".to_string()),
        ];
        let result = apply_label_template(&breaks, "Region: {:Title}", &None);

        assert_eq!(
            result.get("us east"),
            Some(&Some("Region: Us East".to_string()))
        );
        assert_eq!(
            result.get("eu west"),
            Some(&Some("Region: Eu West".to_string()))
        );
    }

    #[test]
    fn test_datetime_placeholder() {
        let breaks = vec![
            ArrayElement::String("2024-01-15".to_string()),
            ArrayElement::String("2024-02-15".to_string()),
        ];
        let result = apply_label_template(&breaks, "{:time %b %Y}", &None);

        assert_eq!(
            result.get("2024-01-15"),
            Some(&Some("Jan 2024".to_string()))
        );
        assert_eq!(
            result.get("2024-02-15"),
            Some(&Some("Feb 2024".to_string()))
        );
    }

    #[test]
    fn test_explicit_takes_priority() {
        let breaks = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
            ArrayElement::String("C".to_string()),
        ];
        let mut existing = HashMap::new();
        existing.insert("A".to_string(), Some("Alpha".to_string()));

        let result = apply_label_template(&breaks, "Category {}", &Some(existing));

        // A should keep explicit mapping
        assert_eq!(result.get("A"), Some(&Some("Alpha".to_string())));
        // B and C should get template
        assert_eq!(result.get("B"), Some(&Some("Category B".to_string())));
        assert_eq!(result.get("C"), Some(&Some("Category C".to_string())));
    }

    #[test]
    fn test_multiple_placeholders() {
        let breaks = vec![ArrayElement::String("hello".to_string())];
        let result = apply_label_template(&breaks, "{} - {:UPPER}", &None);

        assert_eq!(
            result.get("hello"),
            Some(&Some("hello - HELLO".to_string()))
        );
    }

    #[test]
    fn test_no_placeholder_literal() {
        let breaks = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::String("B".to_string()),
        ];
        let result = apply_label_template(&breaks, "Constant Label", &None);

        assert_eq!(result.get("A"), Some(&Some("Constant Label".to_string())));
        assert_eq!(result.get("B"), Some(&Some("Constant Label".to_string())));
    }

    #[test]
    fn test_to_key_string_number_integer() {
        assert_eq!(ArrayElement::Number(0.0).to_key_string(), "0");
        assert_eq!(ArrayElement::Number(25.0).to_key_string(), "25");
        assert_eq!(ArrayElement::Number(-100.0).to_key_string(), "-100");
    }

    #[test]
    fn test_to_key_string_number_decimal() {
        assert_eq!(ArrayElement::Number(25.5).to_key_string(), "25.5");
        assert_eq!(ArrayElement::Number(0.123).to_key_string(), "0.123");
    }

    #[test]
    fn test_to_title_case() {
        assert_eq!(to_title_case("hello world"), "Hello World");
        assert_eq!(to_title_case("HELLO WORLD"), "Hello World");
        assert_eq!(to_title_case("hello"), "Hello");
        assert_eq!(to_title_case(""), "");
    }

    #[test]
    fn test_datetime_with_time() {
        let breaks = vec![ArrayElement::String("2024-01-15T10:30:00".to_string())];
        let result = apply_label_template(&breaks, "{:time %Y-%m-%d %H:%M}", &None);

        assert_eq!(
            result.get("2024-01-15T10:30:00"),
            Some(&Some("2024-01-15 10:30".to_string()))
        );
    }

    #[test]
    fn test_invalid_datetime_fallback() {
        let breaks = vec![ArrayElement::String("not-a-date".to_string())];
        let result = apply_label_template(&breaks, "{:time %Y-%m-%d}", &None);

        // Should fall back to original value
        assert_eq!(
            result.get("not-a-date"),
            Some(&Some("not-a-date".to_string()))
        );
    }

    #[test]
    fn test_null_skipped() {
        let breaks = vec![
            ArrayElement::String("A".to_string()),
            ArrayElement::Null,
            ArrayElement::String("B".to_string()),
        ];
        let result = apply_label_template(&breaks, "{}", &None);

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("A"));
        assert!(result.contains_key("B"));
    }

    #[test]
    fn test_number_format_decimal_places() {
        let breaks = vec![ArrayElement::Number(25.5), ArrayElement::Number(100.0)];
        let result = apply_label_template(&breaks, "${:num %.2f}", &None);

        assert_eq!(result.get("25.5"), Some(&Some("$25.50".to_string())));
        assert_eq!(result.get("100"), Some(&Some("$100.00".to_string())));
    }

    #[test]
    fn test_number_format_no_decimals() {
        let breaks = vec![ArrayElement::Number(25.7)];
        let result = apply_label_template(&breaks, "{:num %.0f} items", &None);

        assert_eq!(result.get("25.7"), Some(&Some("26 items".to_string())));
    }

    #[test]
    fn test_number_format_scientific() {
        let breaks = vec![ArrayElement::Number(1234.5)];
        let result = apply_label_template(&breaks, "{:num %.2e}", &None);

        assert_eq!(result.get("1234.5"), Some(&Some("1.23e+03".to_string())));
    }

    #[test]
    fn test_number_format_non_numeric_fallback() {
        let breaks = vec![ArrayElement::String("hello".to_string())];
        let result = apply_label_template(&breaks, "{:num %.2f}", &None);

        // Non-numeric should fall back to original value
        assert_eq!(result.get("hello"), Some(&Some("hello".to_string())));
    }

    #[test]
    fn test_number_format_integer() {
        let breaks = vec![ArrayElement::Number(42.0)];
        let result = apply_label_template(&breaks, "{:num %d}", &None);

        assert_eq!(result.get("42"), Some(&Some("42".to_string())));
    }
}
