//! Linetype definitions and conversion to Vega-Lite strokeDash arrays.

/// Get the strokeDash array for a named linetype.
/// Returns None for unknown linetypes.
///
/// # Linetype patterns
/// - `solid`: continuous line (empty array)
/// - `dashed`: regular dashes `[6, 4]`
/// - `dotted`: dots `[1, 2]`
/// - `dotdash`: alternating dots and dashes `[1, 2, 6, 2]`
/// - `longdash`: longer dashes `[10, 4]`
/// - `twodash`: two-dash pattern `[6, 2, 2, 2]`
pub fn linetype_to_stroke_dash(name: &str) -> Option<Vec<u32>> {
    match name.to_lowercase().as_str() {
        "solid" => Some(vec![]),
        "dashed" => Some(vec![6, 4]),
        "dotted" => Some(vec![1, 2]),
        "dotdash" => Some(vec![1, 2, 6, 2]),
        "longdash" => Some(vec![10, 4]),
        "twodash" => Some(vec![6, 2, 2, 2]),
        _ => None,
    }
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
}
