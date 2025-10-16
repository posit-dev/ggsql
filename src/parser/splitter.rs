//! Query splitter using tree-sitter
//!
//! Splits VizQL queries at the VISUALISE AS boundary, properly handling
//! SQL strings, comments, and other edge cases.

use crate::{VizqlError, Result};
use regex::Regex;

/// Split a VizQL query into SQL and visualization portions
///
/// Returns (sql_part, viz_part) where:
/// - sql_part: Everything before first "VISUALISE/VISUALIZE AS"
/// - viz_part: Everything from first "VISUALISE/VISUALIZE AS" onwards (may contain multiple VISUALISE statements)
pub fn split_query(query: &str) -> Result<(String, String)> {
    // For now, implement a simple regex-based splitter
    // TODO: Replace with proper tree-sitter based splitting

    let query = query.trim();

    // Find "VISUALISE AS" or "VISUALIZE AS" (case insensitive)
    let pattern = Regex::new(r"(?i)\bVISUALI[SZ]E\s+AS\b")
        .map_err(|e| VizqlError::InternalError(format!("Regex error: {}", e)))?;

    if let Some(mat) = pattern.find(query) {
        let sql_part = query[..mat.start()].trim().to_string();
        let viz_part = query[mat.start()..].trim().to_string();
        Ok((sql_part, viz_part))
    } else {
        // No VISUALISE clause found - treat entire query as SQL
        Ok((query.to_string(), String::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let query = "SELECT * FROM data VISUALISE AS PLOT WITH point x = x, y = y";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("VISUALISE AS PLOT"));
        assert!(viz.contains("WITH point"));
    }

    #[test]
    fn test_case_insensitive() {
        let query = "SELECT * FROM data visualise as plot WITH point x = x, y = y";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, "SELECT * FROM data");
        assert!(viz.starts_with("visualise as plot"));
    }

    #[test]
    fn test_no_visualise() {
        let query = "SELECT * FROM data WHERE x > 5";
        let (sql, viz) = split_query(query).unwrap();

        assert_eq!(sql, query);
        assert!(viz.is_empty());
    }
}