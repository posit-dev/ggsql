//! Facet property resolution
//!
//! Validates facet properties and applies data-aware defaults.

use crate::format::apply_label_template;
use crate::plot::ArrayElement;
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
const WRAP_ALLOWED: &[&str] = &["scales", "ncol", "missing"];

/// Allowed properties for grid facets
const GRID_ALLOWED: &[&str] = &["scales", "missing"];

/// Valid values for the missing property
const MISSING_VALUES: &[&str] = &["repeat", "null"];

/// Valid values for the scales property
const SCALES_VALUES: &[&str] = &["fixed", "free", "free_x", "free_y"];

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
///    - `scales`: must be fixed/free/free_x/free_y
///    - `ncol`: positive integer
/// 4. Applies defaults for missing properties:
///    - `scales`: "fixed"
///    - `ncol` (wrap only): computed from `context.num_levels`
/// 5. Resolves label mappings by applying wildcard template to unique values
/// 6. Sets `resolved = true`
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
                    "property 'ncol' is only allowed for wrap facets, not grid facets".to_string(),
                );
            }
            return Err(format!(
                "unknown property '{}'. Allowed properties: {}",
                key,
                allowed.join(", ")
            ));
        }
    }

    // Step 2: Validate property values
    validate_scales_property(facet)?;
    validate_ncol_property(facet)?;
    validate_missing_property(facet)?;

    // Step 3: Apply defaults for missing properties
    apply_defaults(facet, context);

    // Step 4: Resolve label mappings (apply wildcard template to unique values)
    resolve_label_mapping(facet, context);

    // Mark as resolved
    facet.resolved = true;

    Ok(())
}

/// Resolve label mappings by applying wildcard template to unique values
///
/// If a wildcard template is specified (e.g., `* => 'Region: {}'`), this function
/// expands it into explicit mappings for each unique value in the facet variables.
///
/// Uses the same `apply_label_template` function as scales, which supports:
/// - `{}` - plain substitution
/// - `{:UPPER}`, `{:lower}`, `{:Title}` - case transformations
/// - `{:time %fmt}` - datetime formatting
/// - `{:num %fmt}` - number formatting
fn resolve_label_mapping(facet: &mut Facet, context: &FacetDataContext) {
    let template = &facet.label_template;

    // If default template and no explicit mappings, nothing to do
    if template == "{}" && facet.label_mapping.is_none() {
        return;
    }

    // Collect all unique values from facet variables as ArrayElements
    let variables = facet.get_variables();
    let mut values: Vec<ArrayElement> = Vec::new();
    for var in &variables {
        if let Some(var_values) = context.unique_values.get(var) {
            for v in var_values {
                values.push(ArrayElement::String(v.clone()));
            }
        }
    }

    // Apply label template using the same function as scales
    let generated_labels = apply_label_template(&values, template, &facet.label_mapping);
    facet.label_mapping = Some(generated_labels);

    // Reset template (same as scales - ensures ggsql controls formatting)
    facet.label_template = "{}".to_string();
}

/// Validate scales property value
fn validate_scales_property(facet: &Facet) -> Result<(), String> {
    if let Some(value) = facet.properties.get("scales") {
        match value {
            ParameterValue::String(s) => {
                if !SCALES_VALUES.contains(&s.as_str()) {
                    return Err(format!(
                        "invalid 'scales' value '{}'. Expected one of: {}",
                        s,
                        SCALES_VALUES.join(", ")
                    ));
                }
            }
            _ => {
                return Err(
                    "'scales' must be a string (e.g., 'fixed', 'free', 'free_x', 'free_y')"
                        .to_string(),
                );
            }
        }
    }
    Ok(())
}

/// Validate ncol property value
fn validate_ncol_property(facet: &Facet) -> Result<(), String> {
    if let Some(value) = facet.properties.get("ncol") {
        match value {
            ParameterValue::Number(n) => {
                if *n <= 0.0 || n.fract() != 0.0 {
                    return Err(format!("'ncol' must be a positive integer, got {}", n));
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
    // Default scales to "fixed"
    if !facet.properties.contains_key("scales") {
        facet.properties.insert(
            "scales".to_string(),
            ParameterValue::String("fixed".to_string()),
        );
    }

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

    fn make_context_with_values(variable: &str, values: Vec<&str>) -> FacetDataContext {
        let mut unique_values = HashMap::new();
        unique_values.insert(
            variable.to_string(),
            values.iter().map(|s| s.to_string()).collect(),
        );
        FacetDataContext {
            num_levels: values.len(),
            unique_values,
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
        assert_eq!(
            facet.properties.get("scales"),
            Some(&ParameterValue::String("fixed".to_string()))
        );
        assert_eq!(
            facet.properties.get("ncol"),
            Some(&ParameterValue::Number(3.0))
        );
    }

    #[test]
    fn test_resolve_preserves_user_values() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "scales".to_string(),
            ParameterValue::String("free".to_string()),
        );
        facet
            .properties
            .insert("ncol".to_string(), ParameterValue::Number(2.0));

        let context = make_context(10);
        resolve_properties(&mut facet, &context).unwrap();

        assert_eq!(
            facet.properties.get("scales"),
            Some(&ParameterValue::String("free".to_string()))
        );
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
        assert!(!facet.properties.contains_key("scales"));
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
        assert!(err.contains("unknown property"));
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
        assert!(err.contains("wrap"));
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
        assert!(err.contains("unknown property"));
    }

    #[test]
    fn test_error_invalid_scales_value() {
        let mut facet = make_wrap_facet();
        facet.properties.insert(
            "scales".to_string(),
            ParameterValue::String("invalid".to_string()),
        );

        let context = make_context(5);
        let result = resolve_properties(&mut facet, &context);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("invalid"));
        assert!(err.contains("scales"));
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
        // But should still get scales default
        assert!(facet.properties.contains_key("scales"));
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
    // Label Resolution Tests
    // ========================================

    #[test]
    fn test_label_template_expansion() {
        let mut facet = make_wrap_facet();
        facet.label_template = "Region: {}".to_string();

        let context = make_context_with_values("category", vec!["A", "B", "C"]);
        resolve_properties(&mut facet, &context).unwrap();

        // Template should be reset
        assert_eq!(facet.label_template, "{}");

        // All values should have expanded mappings
        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(mappings.get("A"), Some(&Some("Region: A".to_string())));
        assert_eq!(mappings.get("B"), Some(&Some("Region: B".to_string())));
        assert_eq!(mappings.get("C"), Some(&Some("Region: C".to_string())));
    }

    #[test]
    fn test_label_explicit_mapping_preserved() {
        let mut facet = make_wrap_facet();
        facet.label_template = "Region: {}".to_string();
        let mut mappings = HashMap::new();
        mappings.insert("A".to_string(), Some("Alpha".to_string()));
        facet.label_mapping = Some(mappings);

        let context = make_context_with_values("category", vec!["A", "B", "C"]);
        resolve_properties(&mut facet, &context).unwrap();

        // Explicit mapping should be preserved, template applied to others
        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(mappings.get("A"), Some(&Some("Alpha".to_string()))); // Preserved
        assert_eq!(mappings.get("B"), Some(&Some("Region: B".to_string()))); // Template
        assert_eq!(mappings.get("C"), Some(&Some("Region: C".to_string()))); // Template
    }

    #[test]
    fn test_label_null_suppression_preserved() {
        let mut facet = make_wrap_facet();
        facet.label_template = "Region: {}".to_string();
        let mut mappings = HashMap::new();
        mappings.insert("A".to_string(), None); // Suppressed
        facet.label_mapping = Some(mappings);

        let context = make_context_with_values("category", vec!["A", "B"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(mappings.get("A"), Some(&None)); // Still suppressed
        assert_eq!(mappings.get("B"), Some(&Some("Region: B".to_string())));
    }

    #[test]
    fn test_no_template_no_expansion() {
        let mut facet = make_wrap_facet();
        // Default template "{}" - no expansion needed

        let context = make_context_with_values("category", vec!["A", "B", "C"]);
        resolve_properties(&mut facet, &context).unwrap();

        // No label_mapping should be created
        assert!(facet.label_mapping.is_none());
    }

    #[test]
    fn test_context_stores_unique_values() {
        let df = df! {
            "category" => &["A", "B", "C", "A", "B"],
        }
        .unwrap();

        let context = FacetDataContext::from_dataframe(&df, &["category".to_string()]);

        let values = context.unique_values.get("category").unwrap();
        assert_eq!(values.len(), 3);
        // Values should include A, B, C (order may vary)
        assert!(values.iter().any(|v| v.contains('A')));
        assert!(values.iter().any(|v| v.contains('B')));
        assert!(values.iter().any(|v| v.contains('C')));
    }

    // ========================================
    // Advanced Placeholder Tests (via format.rs)
    // ========================================

    #[test]
    fn test_label_upper_placeholder() {
        let mut facet = make_wrap_facet();
        facet.label_template = "{:UPPER}".to_string();

        let context = make_context_with_values("category", vec!["north", "south"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(mappings.get("north"), Some(&Some("NORTH".to_string())));
        assert_eq!(mappings.get("south"), Some(&Some("SOUTH".to_string())));
    }

    #[test]
    fn test_label_lower_placeholder() {
        let mut facet = make_wrap_facet();
        facet.label_template = "{:lower}".to_string();

        let context = make_context_with_values("category", vec!["HELLO", "WORLD"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(mappings.get("HELLO"), Some(&Some("hello".to_string())));
        assert_eq!(mappings.get("WORLD"), Some(&Some("world".to_string())));
    }

    #[test]
    fn test_label_title_placeholder() {
        let mut facet = make_wrap_facet();
        facet.label_template = "Region: {:Title}".to_string();

        let context = make_context_with_values("category", vec!["us east", "eu west"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(
            mappings.get("us east"),
            Some(&Some("Region: Us East".to_string()))
        );
        assert_eq!(
            mappings.get("eu west"),
            Some(&Some("Region: Eu West".to_string()))
        );
    }

    #[test]
    fn test_label_datetime_placeholder() {
        let mut facet = make_wrap_facet();
        facet.label_template = "{:time %b %Y}".to_string();

        let context = make_context_with_values("category", vec!["2024-01-15", "2024-06-15"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        assert_eq!(
            mappings.get("2024-01-15"),
            Some(&Some("Jan 2024".to_string()))
        );
        assert_eq!(
            mappings.get("2024-06-15"),
            Some(&Some("Jun 2024".to_string()))
        );
    }

    #[test]
    fn test_label_explicit_takes_priority_with_template() {
        let mut facet = make_wrap_facet();
        facet.label_template = "{:UPPER}".to_string();
        let mut mappings = HashMap::new();
        mappings.insert("north".to_string(), Some("Northern Region".to_string()));
        facet.label_mapping = Some(mappings);

        let context = make_context_with_values("category", vec!["north", "south", "east"]);
        resolve_properties(&mut facet, &context).unwrap();

        let mappings = facet.label_mapping.as_ref().unwrap();
        // Explicit mapping should be preserved
        assert_eq!(
            mappings.get("north"),
            Some(&Some("Northern Region".to_string()))
        );
        // Others get template applied
        assert_eq!(mappings.get("south"), Some(&Some("SOUTH".to_string())));
        assert_eq!(mappings.get("east"), Some(&Some("EAST".to_string())));
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
}
