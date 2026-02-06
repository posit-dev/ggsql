//! Linetype definitions and conversion to Vega-Lite strokeDash arrays.

/// Parse a ggplot2-style hex string linetype pattern.
///
/// Format: Even number (2-8) of hex digits, each specifying on/off lengths.
/// Examples: "33" = [3,3], "1343" = [1,3,4,3], "44" = [4,4]
fn parse_hex_linetype(s: &str) -> Option<Vec<u32>> {
    let len = s.len();

    // Must be even length, 2-8 characters, all hex digits
    if !(2..=8).contains(&len) || !len.is_multiple_of(2) {
        return None;
    }

    // Parse each character as a hex digit
    let mut result = Vec::with_capacity(len);
    for c in s.chars() {
        let digit = c.to_digit(16)?;
        if digit == 0 {
            return None; // ggplot2 requires non-zero digits
        }
        result.push(digit);
    }

    Some(result)
}

/// Get the strokeDash array for a linetype specification.
///
/// Supports:
/// - Named linetypes: "solid", "dashed", "dotted", "dotdash", "longdash", "twodash"
/// - Hex string patterns: "33", "1343", "44", etc. (2-8 hex digits)
///
/// # Named linetype patterns
/// - `solid`: continuous line (empty array)
/// - `dashed`: regular dashes `[6, 4]`
/// - `dotted`: dots `[1, 2]`
/// - `dotdash`: alternating dots and dashes `[1, 2, 6, 2]`
/// - `longdash`: longer dashes `[10, 4]`
/// - `twodash`: two-dash pattern `[6, 2, 2, 2]`
///
/// # Hex string patterns
/// A string of 2-8 hex digits (even count), where each digit specifies
/// the length of alternating on/off segments. For example:
/// - `"33"` = 3 units on, 3 units off
/// - `"1343"` = 1 on, 3 off, 4 on, 3 off
/// - `"af"` = 10 on, 15 off (hex digits a-f supported)
pub fn linetype_to_stroke_dash(name: &str) -> Option<Vec<u32>> {
    // First try named linetypes (case-insensitive)
    match name.to_lowercase().as_str() {
        "solid" => return Some(vec![]),
        "dashed" => return Some(vec![6, 4]),
        "dotted" => return Some(vec![1, 2]),
        "dotdash" => return Some(vec![1, 2, 6, 2]),
        "longdash" => return Some(vec![10, 4]),
        "twodash" => return Some(vec![6, 2, 2, 2]),
        _ => {}
    }

    // Then try hex string pattern
    parse_hex_linetype(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linetype_to_stroke_dash_known() {
        assert_eq!(linetype_to_stroke_dash("solid"), Some(vec![]));
        assert_eq!(linetype_to_stroke_dash("dashed"), Some(vec![6, 4]));
        assert_eq!(linetype_to_stroke_dash("dotted"), Some(vec![1, 2]));
        assert_eq!(linetype_to_stroke_dash("dotdash"), Some(vec![1, 2, 6, 2]));
        assert_eq!(linetype_to_stroke_dash("longdash"), Some(vec![10, 4]));
        assert_eq!(linetype_to_stroke_dash("twodash"), Some(vec![6, 2, 2, 2]));
    }

    #[test]
    fn test_linetype_to_stroke_dash_case_insensitive() {
        assert!(linetype_to_stroke_dash("SOLID").is_some());
        assert!(linetype_to_stroke_dash("Dashed").is_some());
        assert!(linetype_to_stroke_dash("DoTdAsH").is_some());
    }

    #[test]
    fn test_linetype_to_stroke_dash_unknown() {
        assert!(linetype_to_stroke_dash("unknown").is_none());
        assert!(linetype_to_stroke_dash("").is_none());
    }

    #[test]
    fn test_hex_linetype_basic() {
        // Simple two-digit patterns
        assert_eq!(linetype_to_stroke_dash("33"), Some(vec![3, 3]));
        assert_eq!(linetype_to_stroke_dash("44"), Some(vec![4, 4]));
        assert_eq!(linetype_to_stroke_dash("13"), Some(vec![1, 3]));
    }

    #[test]
    fn test_hex_linetype_four_digit() {
        assert_eq!(linetype_to_stroke_dash("1343"), Some(vec![1, 3, 4, 3]));
        assert_eq!(linetype_to_stroke_dash("2262"), Some(vec![2, 2, 6, 2]));
    }

    #[test]
    fn test_hex_linetype_six_and_eight_digit() {
        assert_eq!(
            linetype_to_stroke_dash("123456"),
            Some(vec![1, 2, 3, 4, 5, 6])
        );
        assert_eq!(
            linetype_to_stroke_dash("12345678"),
            Some(vec![1, 2, 3, 4, 5, 6, 7, 8])
        );
    }

    #[test]
    fn test_hex_linetype_ggplot2_standards() {
        // ggplot2's standard dash-dot patterns
        assert_eq!(linetype_to_stroke_dash("44"), Some(vec![4, 4])); // dashed
        assert_eq!(linetype_to_stroke_dash("13"), Some(vec![1, 3])); // dotted
        assert_eq!(linetype_to_stroke_dash("1343"), Some(vec![1, 3, 4, 3])); // dotdash
        assert_eq!(linetype_to_stroke_dash("73"), Some(vec![7, 3])); // longdash
        assert_eq!(linetype_to_stroke_dash("2262"), Some(vec![2, 2, 6, 2])); // twodash
    }

    #[test]
    fn test_hex_linetype_with_letters() {
        // Hex digits a-f should work
        assert_eq!(linetype_to_stroke_dash("af"), Some(vec![10, 15]));
        assert_eq!(linetype_to_stroke_dash("AF"), Some(vec![10, 15]));
        assert_eq!(linetype_to_stroke_dash("1a2b"), Some(vec![1, 10, 2, 11]));
    }

    #[test]
    fn test_hex_linetype_invalid() {
        // Odd length
        assert!(linetype_to_stroke_dash("123").is_none());
        // Too short
        assert!(linetype_to_stroke_dash("1").is_none());
        // Too long (>8)
        assert!(linetype_to_stroke_dash("1234567890").is_none());
        // Contains zero (invalid in ggplot2)
        assert!(linetype_to_stroke_dash("10").is_none());
        assert!(linetype_to_stroke_dash("01").is_none());
        // Non-hex characters
        assert!(linetype_to_stroke_dash("gg").is_none());
        assert!(linetype_to_stroke_dash("1x").is_none());
    }
}
