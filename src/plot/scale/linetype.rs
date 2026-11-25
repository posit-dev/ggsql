//! Linetype definitions and conversion to stroke-dash arrays.
//!
//! The patterns live in `zygos::patterns::linetype`; this module adapts
//! them to the `Vec<u32>` stroke-dash form the Vega-Lite writer emits.

/// Get the strokeDash array for a linetype specification.
///
/// Supports:
/// - Named linetypes: "solid", "dashed", "dotted", "dotdash", "longdash", "twodash"
/// - Hex string patterns: "33", "1343", "44", etc. (2-8 hex digits)
///
/// `solid` resolves to an empty array. Lengths are multiples of the line
/// width — the convention zygos, hephaestus and ggsql share (a `6,4`
/// dashed line at 2 px width dashes 12 px on, 8 px off).
pub fn linetype_to_stroke_dash(name: &str) -> Option<Vec<u32>> {
    zygos::patterns::linetype::linetype_pattern(name).map(|steps| {
        steps
            .iter()
            .filter_map(|step| match step {
                zygos::LinetypeStep::Dash(v) | zygos::LinetypeStep::Gap(v) => Some(*v as u32),
                zygos::LinetypeStep::Marker(_) => None,
            })
            .collect()
    })
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
    }

    #[test]
    fn test_linetype_to_stroke_dash_unknown() {
        assert!(linetype_to_stroke_dash("unknown").is_none());
        assert!(linetype_to_stroke_dash("").is_none());
    }

    #[test]
    fn test_hex_linetype() {
        assert_eq!(linetype_to_stroke_dash("33"), Some(vec![3, 3]));
        assert_eq!(linetype_to_stroke_dash("1343"), Some(vec![1, 3, 4, 3]));
        assert_eq!(linetype_to_stroke_dash("af"), Some(vec![10, 15]));
    }
}
