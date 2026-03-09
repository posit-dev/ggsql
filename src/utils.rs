/// Portable scalar MAX across any number of SQL expressions.
/// Replaces GREATEST(a, b, ...) which is not supported by all backends.
/// Generates: `(SELECT MAX(v) FROM (VALUES (a), (b), ...) AS t(v))`
pub fn scalar_max(exprs: &[&str]) -> String {
    let values = exprs
        .iter()
        .map(|e| format!("({e})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("(SELECT MAX(v) FROM (VALUES {values}) AS t(v))")
}

/// Portable scalar MIN across any number of SQL expressions.
/// Replaces LEAST(a, b, ...) which is not supported by all backends.
/// Generates: `(SELECT MIN(v) FROM (VALUES (a), (b), ...) AS t(v))`
pub fn scalar_min(exprs: &[&str]) -> String {
    let values = exprs
        .iter()
        .map(|e| format!("({e})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("(SELECT MIN(v) FROM (VALUES {values}) AS t(v))")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_max_two_args() {
        assert_eq!(
            scalar_max(&["a", "b"]),
            "(SELECT MAX(v) FROM (VALUES (a), (b)) AS t(v))"
        );
    }

    #[test]
    fn test_scalar_min_two_args() {
        assert_eq!(
            scalar_min(&["a", "b"]),
            "(SELECT MIN(v) FROM (VALUES (a), (b)) AS t(v))"
        );
    }

    #[test]
    fn test_scalar_max_three_args() {
        assert_eq!(
            scalar_max(&["x", "y", "z"]),
            "(SELECT MAX(v) FROM (VALUES (x), (y), (z)) AS t(v))"
        );
    }
}
