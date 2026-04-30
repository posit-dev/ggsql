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

fn has_error_ancestor(node: &tree_sitter::Node) -> bool {
    let mut cur = node.parent();
    while let Some(p) = cur {
        if p.is_error() {
            return true;
        }
        cur = p.parent();
    }
    false
}

/// Validate query syntax and semantics without executing SQL.
pub fn validate(query: &str) -> Result<Validated> {
    let mut errors = Vec::new();
    let warnings = Vec::new();

    // Parse once and create SourceTree
    let source_tree = match parser::SourceTree::new(query) {
        Ok(st) => st,
        Err(e) => {
            // Parse error - return as validation error
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

    // Extract SQL and viz portions using existing tree
    let sql_part = source_tree.extract_sql().unwrap_or_default();
    let viz_part = source_tree.extract_visualise().unwrap_or_default();

    let root = source_tree.root();
    let visualise_stmt = source_tree.find_node(&root, "(visualise_statement) @viz");
    let has_visual = visualise_stmt.is_some();

    if let Err(e) = source_tree.validate() {
        // The lexer always tokenises VISUALISE / VISUALIZE as
        // `visualise_keyword` (token prec 10 in grammar.js), so the keyword
        // survives even when parsing fails. We give a ggsql-aware message
        // only when the parse error lies on the VISUALISE side: either no
        // visualise_statement was recovered (keyword stranded in an ERROR
        // node) or one was recovered as a fragment under an ERROR ancestor
        // (partial recovery — common when a mapping holds a SQL expression
        // and the parser rolls forward into the SQL portion). When the
        // visualise_statement is a clean top-level child of `query`, the
        // error is in the SQL portion and the generic message is more honest.
        let kw_pos = source_tree
            .find_node(&root, "(visualise_keyword) @kw")
            .map(|n| n.start_position());
        let visualise_side_failed = match &visualise_stmt {
            Some(node) => node.has_error() || has_error_ancestor(node),
            None => kw_pos.is_some(),
        };
        let (message, location) = if visualise_side_failed {
            (
                "VISUALISE clause was not recognized. Mappings accept column \
                 names only — not SQL expressions like CAST() or function \
                 calls. Move data transformations to the SELECT clause and \
                 reference the resulting column by name in VISUALISE."
                    .to_string(),
                kw_pos.map(|p| Location {
                    line: p.row,
                    column: p.column,
                }),
            )
        } else {
            (e.to_string(), None)
        };
        errors.push(ValidationError { message, location });
        return Ok(Validated {
            sql: sql_part,
            visual: viz_part,
            has_visual,
            tree: Some(source_tree.tree),
            valid: false,
            errors,
            warnings,
        });
    }

    // Genuine SQL-only query (no parse errors, no VISUALISE clause).
    if !has_visual {
        return Ok(Validated {
            sql: sql_part,
            visual: viz_part,
            has_visual: false,
            tree: None,
            valid: true,
            errors,
            warnings,
        });
    }

    // Build AST from existing tree for validation
    let plots = match parser::build_ast(&source_tree) {
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
                tree: Some(source_tree.tree),
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
            // not if the columns are valid. We skip this check for wildcards
            // (either layer or global).
            let is_annotation = matches!(
                layer.source,
                Some(crate::plot::types::DataSource::Annotation)
            );
            let has_wildcard =
                layer.mappings.wildcard || (!is_annotation && plot.global_mappings.wildcard);
            if !has_wildcard {
                // Merge global mappings into a temporary copy for validation
                // (mirrors execution-time merge, layer takes precedence)
                let mut merged = layer.clone();
                if !is_annotation {
                    for (aesthetic, value) in &plot.global_mappings.aesthetics {
                        merged
                            .mappings
                            .aesthetics
                            .entry(aesthetic.clone())
                            .or_insert(value.clone());
                    }
                }
                if let Err(e) = merged.validate_mapping(&plot.aesthetic_context, false) {
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
        tree: Some(source_tree.tree),
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

    #[test]
    fn test_validate_color_aesthetic_on_line() {
        // color should be valid on line geom (has stroke)
        let validated = validate(
            "SELECT 1 as x, 2 as y VISUALISE DRAW line MAPPING x AS x, y AS y, region AS color",
        )
        .unwrap();
        assert!(
            validated.valid(),
            "color should be accepted on line geom: {:?}",
            validated.errors()
        );
    }

    #[test]
    fn test_validate_color_aesthetic_on_point() {
        // color should be valid on point geom (has stroke + fill)
        let validated = validate(
            "SELECT 1 as x, 2 as y VISUALISE DRAW point MAPPING x AS x, y AS y, cat AS color",
        )
        .unwrap();
        assert!(
            validated.valid(),
            "color should be accepted on point geom: {:?}",
            validated.errors()
        );
    }

    #[test]
    fn test_validate_colour_spelling() {
        // British spelling 'colour' should work (normalized by parser to 'color')
        let validated = validate(
            "SELECT 1 as x, 2 as y VISUALISE DRAW line MAPPING x AS x, y AS y, region AS colour",
        )
        .unwrap();
        assert!(
            validated.valid(),
            "colour (British) should be accepted: {:?}",
            validated.errors()
        );
    }

    #[test]
    fn test_validate_global_color_mapping() {
        // Global color mapping should validate correctly
        let validated =
            validate("SELECT 1 as x, 2 as y VISUALISE x AS x, y AS y, region AS color DRAW line")
                .unwrap();
        assert!(
            validated.valid(),
            "global color mapping should be accepted: {:?}",
            validated.errors()
        );
    }

    // Issue #256: SQL expressions in VISUALISE mappings used to be silently
    // consumed as SQL, with validate() reporting valid=true and has_visual=false.
    // The fix detects a stray visualise_keyword node (one that didn't make it
    // into a visualise_statement) and emits an actionable error.

    #[test]
    fn test_validate_cast_in_visualise_mapping() {
        let query = "SELECT sex, survived, COUNT(*) AS n FROM titanic GROUP BY sex, survived\n\
                     VISUALISE sex AS x, n AS y, CAST(survived AS VARCHAR) AS fill\n\
                     DRAW bar";
        let validated = validate(query).unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
        let msg = &validated.errors()[0].message;
        assert!(
            msg.contains("VISUALISE") && msg.contains("column"),
            "expected helpful message, got: {msg}"
        );
        assert!(validated.errors()[0].location.is_some());
    }

    #[test]
    fn test_validate_function_call_in_visualise_mapping() {
        let query = "SELECT t, v FROM data VISUALISE date_trunc('day', t) AS x, v AS y DRAW line";
        let validated = validate(query).unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
        assert!(validated.errors()[0].message.contains("VISUALISE"));
    }

    #[test]
    fn test_validate_lowercase_visualise_keyword_with_expression() {
        let query = "SELECT a, b FROM t visualise cast(a as varchar) as x, b as y draw point";
        let validated = validate(query).unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
    }

    #[test]
    fn test_validate_us_visualize_spelling_with_expression() {
        let query = "SELECT a, b FROM t VISUALIZE CAST(a AS VARCHAR) AS x, b AS y DRAW point";
        let validated = validate(query).unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
    }

    #[test]
    fn test_validate_sql_side_error_does_not_blame_visualise() {
        // When the parse error is in the SQL portion but the VISUALISE clause
        // is fully recovered, we must not emit the "VISUALISE clause was not
        // recognized" message — the VISUALISE clause is fine.
        let query = "SELECT @@@ FROM t VISUALISE a AS x, b AS y DRAW point";
        let validated = validate(query).unwrap();
        assert!(!validated.valid());
        assert!(!validated.errors().is_empty());
        let msg = &validated.errors()[0].message;
        assert!(
            !msg.contains("VISUALISE clause was not recognized"),
            "SQL-side error should not be reported as a VISUALISE error, got: {msg}"
        );
    }

    #[test]
    fn test_validate_visualise_in_string_literal_is_valid() {
        // VISUALISE inside a string literal must NOT trigger the new error —
        // tree-sitter classifies it as part of a string node.
        let validated = validate("SELECT 'VISUALISE' AS s").unwrap();
        assert!(
            validated.valid(),
            "string literal containing VISUALISE should be valid: {:?}",
            validated.errors()
        );
        assert!(!validated.has_visual());
    }

    #[test]
    fn test_validate_visualise_in_comment_is_valid() {
        // VISUALISE inside a comment must NOT trigger the new error.
        let validated = validate("SELECT 1 AS x -- VISUALISE here\n").unwrap();
        assert!(
            validated.valid(),
            "comment containing VISUALISE should be valid: {:?}",
            validated.errors()
        );
        assert!(!validated.has_visual());
    }
}
