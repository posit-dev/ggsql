//! General-purpose utility functions for consistent string formatting across the codebase.

use std::fmt::Display;

/// Format a list of items with "and", following Oxford comma rules.
///
/// Works with any type implementing `Display` (strings, enums, numbers, etc.)
///
/// # Examples
/// - `and_list(&["a", "b", "c"])` → `a, b, and c`
/// - `and_list(&[Color::Red, Color::Green])` → `red and green`
/// - `and_list(&[1, 2, 3])` → `1, 2, and 3`
pub fn and_list<T: Display>(items: &[T]) -> String {
    format_list(items, "and")
}

/// Format a list of items with "or", following Oxford comma rules.
///
/// Works with any type implementing `Display` (strings, enums, numbers, etc.)
///
/// # Examples
/// - `or_list(&["a", "b", "c"])` → `a, b, or c`
/// - `or_list(&[Color::Red, Color::Green])` → `red or green`
pub fn or_list<T: Display>(items: &[T]) -> String {
    format_list(items, "or")
}

fn format_list<T: Display>(items: &[T], conjunction: &str) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].to_string(),
        2 => format!("{} {} {}", items[0], conjunction, items[1]),
        _ => {
            let mut result = String::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                if i == items.len() - 1 {
                    result.push_str(conjunction);
                    result.push(' ');
                }
                result.push_str(&item.to_string());
            }
            result
        }
    }
}

/// Format a list of items with quotes and "and", following Oxford comma rules.
///
/// Works with any type implementing `Display` (strings, enums, numbers, etc.)
///
/// # Examples
/// - `and_list_quoted(&["a", "b", "c"], '\'')` → `'a', 'b', and 'c'`
/// - `and_list_quoted(&[Color::Red, Color::Green], '\'')` → `'red' and 'green'`
pub fn and_list_quoted<T: Display>(items: &[T], quote: char) -> String {
    format_quoted_list(items, quote, "and")
}

/// Format a list of items with quotes and "or", following Oxford comma rules.
///
/// Works with any type implementing `Display` (strings, enums, numbers, etc.)
///
/// # Examples
/// - `or_list_quoted(&["a", "b", "c"], '\'')` → `'a', 'b', or 'c'`
/// - `or_list_quoted(&[Color::Red, Color::Green], '\'')` → `'red' or 'green'`
pub fn or_list_quoted<T: Display>(items: &[T], quote: char) -> String {
    format_quoted_list(items, quote, "or")
}

fn format_quoted_list<T: Display>(items: &[T], quote: char, conjunction: &str) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|item| format!("{}{}{}", quote, item, quote))
        .collect();
    format_list(&quoted, conjunction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_list_empty() {
        let empty: &[&str] = &[];
        assert_eq!(and_list(empty), "");
    }

    #[test]
    fn test_and_list_single() {
        assert_eq!(and_list(&["apple"]), "apple");
    }

    #[test]
    fn test_and_list_two() {
        assert_eq!(and_list(&["apple", "banana"]), "apple and banana");
    }

    #[test]
    fn test_and_list_three() {
        assert_eq!(
            and_list(&["apple", "banana", "cherry"]),
            "apple, banana, and cherry"
        );
    }

    #[test]
    fn test_or_list_two() {
        assert_eq!(or_list(&["apple", "banana"]), "apple or banana");
    }

    #[test]
    fn test_or_list_three() {
        assert_eq!(
            or_list(&["apple", "banana", "cherry"]),
            "apple, banana, or cherry"
        );
    }

    #[test]
    fn test_and_list_quoted_empty() {
        let empty: &[&str] = &[];
        assert_eq!(and_list_quoted(empty, '\''), "");
    }

    #[test]
    fn test_and_list_quoted_single() {
        assert_eq!(and_list_quoted(&["apple"], '\''), "'apple'");
    }

    #[test]
    fn test_and_list_quoted_two() {
        assert_eq!(
            and_list_quoted(&["apple", "banana"], '\''),
            "'apple' and 'banana'"
        );
    }

    #[test]
    fn test_and_list_quoted_three() {
        assert_eq!(
            and_list_quoted(&["apple", "banana", "cherry"], '\''),
            "'apple', 'banana', and 'cherry'"
        );
    }

    #[test]
    fn test_and_list_quoted_four() {
        assert_eq!(
            and_list_quoted(&["a", "b", "c", "d"], '\''),
            "'a', 'b', 'c', and 'd'"
        );
    }

    #[test]
    fn test_or_list_quoted_two() {
        assert_eq!(
            or_list_quoted(&["apple", "banana"], '"'),
            "\"apple\" or \"banana\""
        );
    }

    #[test]
    fn test_or_list_quoted_three() {
        assert_eq!(
            or_list_quoted(&["apple", "banana", "cherry"], '`'),
            "`apple`, `banana`, or `cherry`"
        );
    }

    #[test]
    fn test_with_string_vec() {
        let items = vec!["one".to_string(), "two".to_string()];
        assert_eq!(and_list_quoted(&items, '\''), "'one' and 'two'");
    }

    // Enum tests
    #[derive(Debug)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    impl std::fmt::Display for Color {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Color::Red => write!(f, "red"),
                Color::Green => write!(f, "green"),
                Color::Blue => write!(f, "blue"),
            }
        }
    }

    #[test]
    fn test_and_list_enum() {
        assert_eq!(
            and_list(&[Color::Red, Color::Green, Color::Blue]),
            "red, green, and blue"
        );
    }

    #[test]
    fn test_or_list_enum() {
        assert_eq!(or_list(&[Color::Red, Color::Green]), "red or green");
    }

    #[test]
    fn test_and_list_quoted_enum() {
        assert_eq!(
            and_list_quoted(&[Color::Red, Color::Green, Color::Blue], '\''),
            "'red', 'green', and 'blue'"
        );
    }

    #[test]
    fn test_or_list_quoted_enum() {
        assert_eq!(
            or_list_quoted(&[Color::Red, Color::Green], '"'),
            "\"red\" or \"green\""
        );
    }

    #[test]
    fn test_with_numbers() {
        assert_eq!(and_list(&[1, 2, 3]), "1, 2, and 3");
        assert_eq!(or_list(&[42, 99]), "42 or 99");
    }
}
