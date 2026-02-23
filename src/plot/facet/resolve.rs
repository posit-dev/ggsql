//! Facet property resolution
//!
//! Validates facet properties and applies data-aware defaults.

use crate::plot::ParameterValue;
use crate::DataFrame;
use std::collections::HashMap;

use super::types::Facet;

/// Context for facet resolution with data-derived information
pub struct FacetDataContext {
    /// Number of unique values in the first facet variable
    pub num_levels: usize,
    /// Unique values for each facet variable (as strings for label formatting)
    pub unique_values: HashMap<String, Vec<String>>,
}

impl FacetDataContext {
    /// Create context from a DataFrame and facet variables
    ///
    /// Extracts unique values from each facet variable for label resolution.
    pub fn from_dataframe(df: &DataFrame, variables: &[String]) -> Self {
        let mut unique_values = HashMap::new();
        let mut num_levels = 1;

        for (i, var) in variables.iter().enumerate() {
            if let Ok(col) = df.column(var) {
                let unique = col.unique().ok();
                let values: Vec<String> = unique
                    .as_ref()
                    .map(|u| {
                        (0..u.len())
                            .filter_map(|j| u.get(j).ok().map(|v| format!("{}", v)))
                            .collect()
                    })
                    .unwrap_or_default();

                if i == 0 {
                    num_levels = values.len().max(1);
                }
                unique_values.insert(var.clone(), values);
            }
        }

        Self {
            num_levels,
            unique_values,
        }
    }
}

/// Allowed properties for wrap facets
const WRAP_ALLOWED: &[&str] = &["free", "ncol", "missing"];

/// Allowed properties for grid facets
const GRID_ALLOWED: &[&str] = &["free", "missing"];

/// Valid values for the missing property
const MISSING_VALUES: &[&str] = &["repeat", "null"];

/// Valid string values for the free property
const FREE_STRING_VALUES: &[&str] = &["x", "y"];

/// Compute smart default ncol for wrap facets based on number of levels
///
/// Returns an optimal column count that creates a balanced grid:
/// - n ≤ 3: ncol = n (single row)
/// - n ≤ 6: ncol = 3
/// - n ≤ 12: ncol = 4
/// - n > 12: ncol = 5
fn compute_default_ncol(num_levels: usize) -> i64 {
    if num_levels <= 3 {
        num_levels as i64
    } else if num_levels <= 6 {
        3
    } else if num_levels <= 12 {
        4
    } else {
        5
    }
}

/// Resolve and validate facet properties
///
/// This function:
/// 1. Skips if already resolved
/// 2. Validates all properties are allowed for this layout
/// 3. Validates property values:
///    - `free`: must be null, 'x', 'y', or ['x', 'y']
///    - `ncol`: positive integer
/// 4. Applies defaults for missing properties:
///    - `ncol` (wrap only): computed from `context.num_levels`
/// 5. Sets `resolved = true`
pub fn resolve_properties(facet: &mut Facet, context: &FacetDataContext) -> Result<(), String> {
    // Skip if already resolved
    if facet.resolved {
        return Ok(());
    }

    let is_wrap = facet.is_wrap();

    // Step 1: Validate all properties are allowed for this layout
    let allowed = if is_wrap { WRAP_ALLOWED } else { GRID_ALLOWED };
    for key in facet.properties.keys() {
        if !allowed.contains(&key.as_str()) {
            if key == "ncol" && !is_wrap {
                return Err(
                    "Setting `ncol` is only allowed for 1 dimensional facets, not 2 dimensional facets".to_string(),
                );
            }
            return Err(format!(
                "Unknown setting: '{}'. Allowed settings: {}",
                key,
                allowed.join(", ")
            ));
        }
    }

    // Step 2: Validate property values
    validate_free_property(facet)?;
    validate_ncol_property(facet)?;
    validate_missing_property(facet)?;

    // Step 3: Apply defaults for missing properties
    apply_defaults(facet, context);

    // Mark as resolved
    facet.resolved = true;

    Ok(())
}

/// Validate free property value
///
/// Accepts:
/// - `null` (ParameterValue::Null) - shared scales (default when absent)
/// - `'x'` or `'y'` (strings) - independent scale for that axis only
/// - `['x', 'y']` or `['y', 'x']` (arrays) - independent scales for both axes
fn validate_free_property(facet: &Facet) -> Result<(), String> {
    if let Some(value) = facet.properties.get("free") {
        match value {
            ParameterValue::Null => {
                // Explicit null means shared scales (same as default)
                Ok(())
            }
            ParameterValue::String(s) => {
                if !FREE_STRING_VALUES.contains(&s.as_str()) {
                    return Err(format!(
                        "invalid 'free' value '{}'. Expected 'x', 'y', ['x', 'y'], or null",
                        s
                    ));
                }
                Ok(())
            }
            ParameterValue::Array(arr) => {
                // Must be exactly ['x', 'y'] or ['y', 'x']
                if arr.len() != 2 {
                    return Err(format!(
                        "invalid 'free' array: expected ['x', 'y'], got {} elements",
                        arr.len()
                    ));
                }
                let mut has_x = false;
                let mut has_y = false;
                for elem in arr {
                    match elem {
                        crate::plot::ArrayElement::String(s) if s == "x" => has_x = true,
                        crate::plot::ArrayElement::String(s) if s == "y" => has_y = true,
                        _ => {
                            return Err(
                                "invalid 'free' array: elements must be 'x' or 'y'".to_string()
                            );
                        }
                    }
                }
                if !has_x || !has_y {
                    return Err(
                        "invalid 'free' array: expected ['x', 'y'] with both 'x' and 'y'"
                            .to_string(),
                    );
                }
                Ok(())
            }
            _ => Err(
                "'free' must be null, a string ('x' or 'y'), or an array ['x', 'y']".to_string(),
            ),
        }
    } else {
        Ok(())
    }
}

/// Validate ncol property value
fn validate_ncol_property(facet: &Facet) -> Result<(), String> {
    if let Some(value) = facet.properties.get("ncol") {
        match value {
            ParameterValue::Number(n) => {
                if *n <= 0.0 || n.fract() != 0.0 {
                    return Err(format!("`ncol` must be a positive integer, got {}", n));
                }
            }
            _ => {
                return Err("'ncol' must be a number".to_string());
            }
        }
    }
    Ok(())
}

/// Validate missing property value
fn validate_missing_property(facet: &Facet) -> Result<(), String> {
    if let Some(value) = facet.properties.get("missing") {
        match value {
            ParameterValue::String(s) => {
                if !MISSING_VALUES.contains(&s.as_str()) {
                    return Err(format!(
                        "invalid 'missing' value '{}'. Expected one of: {}",
                        s,
                        MISSING_VALUES.join(", ")
                    ));
                }
            }
            _ => {
                return Err("'missing' must be a string ('repeat' or 'null')".to_string());
            }
        }
    }
    Ok(())
}

/// Apply default values for missing properties
fn apply_defaults(facet: &mut Facet, context: &FacetDataContext) {
    // Note: absence of 'free' property means fixed/shared scales (default)
    // No need to insert a default value

    // Default ncol for wrap facets (computed from data)
    if facet.is_wrap() && !facet.properties.contains_key("ncol") {
        let default_cols = compute_default_ncol(context.num_levels);
        facet.properties.insert(
            "ncol".to_string(),
            ParameterValue::Number(default_cols as f64),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::facet::FacetLayout;
    use polars::prelude::*;

    fn make_wrap_facet() -> Facet {
        Facet::new(FacetLayout::Wrap {
            variables: vec!["category".to_string()],
        })
    }

    fn make_grid_facet() -> Facet {
        Facet::new(FacetLayout::Grid {
            row: vec!["row_var".to_string()],
            column: vec!["col_var".to_string()],
        })
    }

    fn make_context(num_levels: usize) -> FacetDataContext {
        FacetDataContext {
            num_levels,
            unique_values: HashMap::new(),
        }
    }

    #[test]
    fn test_compute_default_ncol() {
        assert_eq!(compute_default_ncol(1), 1);
        assert_eq!(compute_default_ncol(2), 2);
        assert_eq!(compute_default_ncol(3), 3);
        assert_eq!(compute_default_ncol(4), 3);
        assert_eq!(compute_default_ncol(6), 3);
        assert_eq!(compute_default_ncol(7), 4);
        assert_eq!(compute_default_ncol(12), 4);
        assert_eq!(compute_default_ncol(13), 5);
        assert_eq!(compute_default_ncol(100), 5);
    }

    #[test]
    fn test_resolve_applies_defaults() {
        let mut facet = make_wrap_facet();
        let context = make_context(5);

        resolve_properties(&mut facet, &context).unwrap();

        assert!(facet.resolved);
        // Note: absence of 'free' means fixed scales (no default inserted)
        assert!(!facet.properties.contains_key("free"));
        assert_eq!(
            facet.properties.get("ncol"),
            Some(&ParameterValue::Number(3.0))
        );
    }

    #[test]
    fn test_resolve_preserves_user_values() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::Array(vec![
                crate::plot::ArrayElement::String("x".to_string()),
                crate::plot::ArrayElement::String("y".to_string()),
            ]),
        );
        facet
            .properties
            .insert("ncol".to_string(), ParameterValue::Number(2.0));

        let context = make_context(10);
        resolve_properties(&mut facet, &context).unwrap();

        // free => ['x', 'y'] preserved
        assert!(facet.properties.contains_key("free"));
        assert_eq!(
            facet.properties.get("ncol"),
            Some(&ParameterValue::Number(2.0))
        );
    }

    #[test]
    fn test_resolve_skips_if_already_resolved() {
        let mut facet = make_wrap_facet();
        facet.resolved = true;

        let context = make_context(5);
        resolve_properties(&mut facet, &context).unwrap();

        // Should not have applied defaults since it was already resolved
        assert!(!facet.properties.contains_key("ncol"));
    }

    #[test]
    fn test_error_columns_is_unknown_property() {
        // "columns" is Vega-Lite's name, we use "ncol"
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("columns".to_string(), ParameterValue::Number(4.0));

        let context = make_context(10);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown setting"));
        assert!(err.contains("columns"));
    }

    #[test]
    fn test_error_ncol_on_grid() {
        let mut facet = make_grid_facet();
        facet
            .properties
            .insert("ncol".to_string(), ParameterValue::Number(3.0));

        let context = make_context(10);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("ncol"));
        assert!(err.contains("1 dimensional"));
    }

    #[test]
    fn test_error_unknown_property() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("unknown".to_string(), ParameterValue::Number(1.0));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown setting"));
    }

    #[test]
    fn test_error_invalid_free_value() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::String("invalid".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("invalid"));
        assert!(err.contains("free"));
    }

    #[test]
    fn test_error_negative_ncol() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("ncol".to_string(), ParameterValue::Number(-1.0));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("ncol"));
        assert!(err.contains("positive"));
    }

    #[test]
    fn test_error_non_integer_ncol() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("ncol".to_string(), ParameterValue::Number(2.5));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("ncol"));
        assert!(err.contains("integer"));
    }

    #[test]
    fn test_grid_no_ncol_default() {
        let mut facet = make_grid_facet();
        let context = make_context(10);

        resolve_properties(&mut facet, &context).unwrap();

        // Grid facets should not get ncol default
        assert!(!facet.properties.contains_key("ncol"));
        // No free property by default (means fixed/shared scales)
        assert!(!facet.properties.contains_key("free"));
        assert!(facet.resolved);
    }

    #[test]
    fn test_context_from_dataframe() {
        let df = df! {
            "category" => &["A", "B", "C", "A", "B", "C"],
            "value" => &[1, 2, 3, 4, 5, 6],
        }
        .unwrap();

        let context = FacetDataContext::from_dataframe(&df, &["category".to_string()]);
        assert_eq!(context.num_levels, 3);
    }

    #[test]
    fn test_context_from_dataframe_missing_column() {
        let df = df! {
            "other" => &[1, 2, 3],
        }
        .unwrap();

        let context = FacetDataContext::from_dataframe(&df, &["missing".to_string()]);
        assert_eq!(context.num_levels, 1); // Falls back to 1
    }

    #[test]
    fn test_context_from_dataframe_empty_variables() {
        let df = df! {
            "x" => &[1, 2, 3],
        }
        .unwrap();

        let context = FacetDataContext::from_dataframe(&df, &[]);
        assert_eq!(context.num_levels, 1);
    }

    // ========================================
    // Missing Property Tests
    // ========================================

    #[test]
    fn test_missing_property_repeat_valid() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "missing".to_string(),
            ParameterValue::String("repeat".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_property_null_valid() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "missing".to_string(),
            ParameterValue::String("null".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_invalid_missing_value() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "missing".to_string(),
            ParameterValue::String("invalid".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("invalid"));
        assert!(err.contains("missing"));
    }

    #[test]
    fn test_error_missing_not_string() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("missing".to_string(), ParameterValue::Number(1.0));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("missing"));
        assert!(err.contains("string"));
    }

    #[test]
    fn test_missing_allowed_on_grid_facet() {
        let mut facet = make_grid_facet();
        facet.properties.insert(
            "missing".to_string(),
            ParameterValue::String("repeat".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    // ========================================
    // Free Property Tests
    // ========================================

    #[test]
    fn test_free_property_x_valid() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("free".to_string(), ParameterValue::String("x".to_string()));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_free_property_y_valid() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("free".to_string(), ParameterValue::String("y".to_string()));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_free_property_array_valid() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::Array(vec![
                crate::plot::ArrayElement::String("x".to_string()),
                crate::plot::ArrayElement::String("y".to_string()),
            ]),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_free_property_array_reversed_valid() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::Array(vec![
                crate::plot::ArrayElement::String("y".to_string()),
                crate::plot::ArrayElement::String("x".to_string()),
            ]),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_free_property_null_valid() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("free".to_string(), ParameterValue::Null);

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_free_array_single_element() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::Array(vec![crate::plot::ArrayElement::String("x".to_string())]),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("free"));
        // Single element fails both the length check (1 != 2) and the "both x and y" check
        assert!(
            err.contains("1 elements") || err.contains("both 'x' and 'y'"),
            "Expected error about array length or missing elements, got: {}",
            err
        );
    }

    #[test]
    fn test_error_free_array_invalid_element() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "free".to_string(),
            ParameterValue::Array(vec![
                crate::plot::ArrayElement::String("x".to_string()),
                crate::plot::ArrayElement::String("z".to_string()),
            ]),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("free"));
        assert!(err.contains("'x' or 'y'"));
    }

    #[test]
    fn test_error_free_wrong_type() {
        let mut facet = make_wrap_facet();
        facet
            .properties
            .insert("free".to_string(), ParameterValue::Number(1.0));

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("free"));
    }
}
