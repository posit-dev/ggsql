//! Query validation without SQL execution.
//!
//! This module provides query syntax and semantic validation without executing
//! any SQL. Use this for IDE integration, syntax checking, and query inspection.

use crate::parser;
use crate::Result;

// ============================================================================
// Core Types
// ============================================================================

/// Result of `validate()` - query inspection and validation without SQL execution.
pub struct Validated {
    sql: String,
    visual: String,
    has_visual: bool,
    tree: Option<tree_sitter::Tree>,
    valid: bool,
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
}

impl Validated {
    /// Whether the query contains a VISUALISE clause.
    pub fn has_visual(&self) -> bool {
        self.has_visual
    }

    /// The SQL portion (before VISUALISE).
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// The VISUALISE portion (raw text).
    pub fn visual(&self) -> &str {
        &self.visual
    }

    /// CST for advanced inspection.
    pub fn tree(&self) -> Option<&tree_sitter::Tree> {
        self.tree.as_ref()
    }

    /// Whether the query is valid (no errors).
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Validation errors.
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Validation warnings.
    pub fn warnings(&self) -> &[ValidationWarning] {
        &self.warnings
    }
}

/// A validation error (fatal).
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub location: Option<Location>,
}

/// A validation warning (non-fatal).
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
    pub location: Option<Location>,
}

/// Location within a query string (0-based).
#[derive(Debug, Clone)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

// ============================================================================
// Validation Function
// ============================================================================

/// Validate query syntax and semantics without executing SQL.
pub fn validate(query: &str) -> Result<Validated> {
    let mut errors = Vec::new();
    let warnings = Vec::new();

    // Split to determine if there's a viz portion
    let (sql_part, viz_part) = match parser::split_query(query) {
        Ok((sql, viz)) => (sql, viz),
        Err(e) => {
            // Split error - return as validation error
            errors.push(ValidationError {
                message: e.to_string(),
                location: None,
            });
            return Ok(Validated {
                sql: String::new(),
                visual: String::new(),
                has_visual: false,
                tree: None,
                valid: false,
                errors,
                warnings,
            });
        }
    };

    let has_visual = !viz_part.trim().is_empty();

    // Parse the full query to get the CST
    let tree = if has_visual {
        let mut ts_parser = tree_sitter::Parser::new();
        ts_parser
            .set_language(&tree_sitter_ggsql::language())
            .map_err(|e| {
                crate::GgsqlError::InternalError(format!("Failed to set language: {}", e))
            })?;
        ts_parser.parse(query, None)
    } else {
        None
    };

    // If no visualization, just syntax check passed
    if !has_visual {
        return Ok(Validated {
            sql: sql_part,
            visual: viz_part,
            has_visual,
            tree,
            valid: true,
            errors,
            warnings,
        });
    }

    // Parse to get plot specifications for validation
    let plots = match parser::parse_query(query) {
        Ok(p) => p,
        Err(e) => {
            errors.push(ValidationError {
                message: e.to_string(),
                location: None,
            });
            return Ok(Validated {
                sql: sql_part,
                visual: viz_part,
                has_visual,
                tree,
                valid: false,
                errors,
                warnings,
            });
        }
    };

    // Validate the single plot (we only support one VISUALISE statement)
    if let Some(plot) = plots.first() {
        // Validate each layer
        for (layer_idx, layer) in plot.layers.iter().enumerate() {
            let context = format!("Layer {}", layer_idx + 1);

            // Check required aesthetics
            // Note: Without schema data, we can only check if mappings exist,
            // not if the columns are valid. We skip this check for wildcards.
            if !layer.mappings.wildcard {
                if let Err(e) = layer.validate_required_aesthetics() {
                    errors.push(ValidationError {
                        message: format!("{}: {}", context, e),
                        location: None,
                    });
                }
            }

            // Validate SETTING parameters
            if let Err(e) = layer.validate_settings() {
                errors.push(ValidationError {
                    message: format!("{}: {}", context, e),
                    location: None,
                });
            }
        }
    }

    Ok(Validated {
        sql: sql_part,
        visual: viz_part,
        has_visual,
        tree,
        valid: errors.is_empty(),
        errors,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_with_visual() {
        let validated =
            validate("SELECT 1 as x, 2 as y VISUALISE DRAW point MAPPING x AS x, y AS y").unwrap();
        assert!(validated.has_visual());
        assert_eq!(validated.sql(), "SELECT 1 as x, 2 as y");
        assert!(validated.visual().starts_with("VISUALISE"));
        assert!(validated.tree().is_some());
        assert!(validated.valid());
    }

    #[test]
    fn test_validate_without_visual() {
        let validated = validate("SELECT 1 as x, 2 as y").unwrap();
        assert!(!validated.has_visual());
        assert_eq!(validated.sql(), "SELECT 1 as x, 2 as y");
        assert!(validated.visual().is_empty());
        assert!(validated.tree().is_none());
        assert!(validated.valid());
    }

    #[test]
    fn test_validate_valid_query() {
        let validated =
            validate("SELECT 1 as x, 2 as y VISUALISE DRAW point MAPPING x AS x, y AS y").unwrap();
        assert!(
            validated.valid(),
            "Expected valid query: {:?}",
            validated.errors()
        );
        assert!(validated.errors().is_empty());
    }

    #[test]
    fn test_validate_missing_required_aesthetic() {
        // Point requires x and y, but we only provide x
        let validated =
            validate("SELECT 1 as x, 2 as y VISUALISE DRAW point MAPPING x AS x").unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
        assert!(validated.errors()[0].message.contains("y"));
    }

    #[test]
    fn test_validate_syntax_error() {
        let validated = validate("SELECT 1 VISUALISE DRAW invalidgeom").unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
    }

    #[test]
    fn test_validate_sql_and_visual_content() {
        let query = "SELECT 1 as x, 2 as y VISUALISE DRAW point MAPPING x AS x, y AS y DRAW line MAPPING x AS x, y AS y";
        let validated = validate(query).unwrap();

        assert!(validated.has_visual());
        assert_eq!(validated.sql(), "SELECT 1 as x, 2 as y");
        assert!(validated.visual().contains("DRAW point"));
        assert!(validated.visual().contains("DRAW line"));
        assert!(validated.valid());
    }

    #[test]
    fn test_validate_sql_only() {
        let query = "SELECT 1 as x, 2 as y";
        let validated = validate(query).unwrap();

        // SQL-only queries should be valid (just syntax check)
        assert!(validated.valid());
        assert!(validated.errors().is_empty());
    }
}
