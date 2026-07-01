//! SQL structural extraction over the tree-sitter parse tree.
//!
//! These helpers parse a SQL fragment and pull structure out of it for
//! the reader and execution layers.

use crate::parser::SourceTree;
use crate::{naming, Result};

/// A `table_ref` occurrence in a SQL query.
pub struct TableRefSite {
    /// Byte offset of the start of the `table` node.
    pub start: usize,
    /// Byte offset of the end of the `table` node.
    pub end: usize,
    /// Raw source text of the `table` node.
    pub raw: String,
    /// Whether the ref carries an explicit `alias`.
    pub has_alias: bool,
}

/// Locate every `table_ref` in a SQL query, with the byte range of its `table`
/// target and whether it is aliased.
///
/// Subquery sources have no `table` field and are skipped.
pub fn extract_table_ref_sites(sql: &str) -> Result<Vec<TableRefSite>> {
    let source_tree = SourceTree::new(sql)?;
    let root = source_tree.root();

    let mut sites = Vec::new();
    for node in source_tree.find_nodes(&root, "(table_ref) @ref") {
        let Some(table) = node.child_by_field_name("table") else {
            continue;
        };
        sites.push(TableRefSite {
            start: table.start_byte(),
            end: table.end_byte(),
            raw: source_tree.get_text(&table),
            has_alias: node.child_by_field_name("alias").is_some(),
        });
    }
    Ok(sites)
}

/// Extract the table names referenced in a SQL query's `FROM`/`JOIN` clauses.
///
/// Returns the `table_ref` targets, with surrounding quotes stripped. Subquery
/// sources contribute no name.
pub fn extract_table_refs(sql: &str) -> Result<Vec<String>> {
    let source_tree = SourceTree::new(sql)?;
    let root = source_tree.root();

    let mut names: Vec<String> = source_tree
        .find_texts(&root, "(table_ref table: (_) @table)")
        .iter()
        .map(|t| naming::unquote_ident(t))
        .collect();
    names.sort_unstable();
    names.dedup();
    Ok(names)
}

/// Byte ranges of string literals in `sql`.
pub fn string_literal_ranges(sql: &str) -> Vec<(usize, usize)> {
    let Ok(source_tree) = SourceTree::new(sql) else {
        return Vec::new();
    };
    let root = source_tree.root();
    source_tree
        .find_nodes(&root, "(string) @s")
        .iter()
        .map(|n| (n.start_byte(), n.end_byte()))
        .collect()
}

/// Extract builtin dataset names from SQL containing namespaced identifiers.
pub fn extract_builtin_dataset_names(sql: &str) -> Result<Vec<String>> {
    let source_tree = SourceTree::new(sql)?;
    let root = source_tree.root();

    let mut names: Vec<String> = source_tree
        .find_texts(&root, "(namespaced_identifier) @select")
        .iter()
        .filter_map(|token| token.strip_prefix("ggsql:").map(|s| s.to_string()))
        .collect();
    names.sort_unstable();
    names.dedup();
    Ok(names)
}

/// Rewrite SQL to replace namespaced identifiers with internal table names.
///
/// e.g. `SELECT * FROM ggsql:penguins` → `SELECT * FROM __ggsql_data_penguins__`.
///
/// Uses the parse tree to find the positions of namespaced identifiers, then
/// replaces them.
pub fn rewrite_namespaced_sql(sql: &str) -> Result<String> {
    let source_tree = SourceTree::new(sql)?;
    let root = source_tree.root();

    // Collect (start_byte, end_byte, replacement) tuples.
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    for node in source_tree.find_nodes(&root, "(namespaced_identifier) @select") {
        let full_text = source_tree.get_text(&node);
        if let Some(name) = full_text.strip_prefix("ggsql:") {
            replacements.push((
                node.start_byte(),
                node.end_byte(),
                naming::quote_ident(&naming::builtin_data_table(name)),
            ));
        }
    }

    if replacements.is_empty() {
        return Ok(sql.to_string());
    }

    // Apply replacements in reverse byte order to preserve earlier offsets.
    let mut result = sql.to_string();
    replacements.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_builtin_dataset_names_single() {
        let sql = "SELECT * FROM ggsql:penguins VISUALISE DRAW point MAPPING x AS x";
        assert_eq!(
            extract_builtin_dataset_names(sql).unwrap(),
            vec!["penguins"]
        );
    }

    #[test]
    fn test_extract_builtin_dataset_names_multiple() {
        let sql =
            "SELECT * FROM ggsql:penguins, ggsql:airquality VISUALISE DRAW point MAPPING x AS x";
        let names = extract_builtin_dataset_names(sql).unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"airquality".to_string()));
        assert!(names.contains(&"penguins".to_string()));
    }

    #[test]
    fn test_extract_builtin_dataset_names_dedup() {
        let sql = "SELECT * FROM ggsql:penguins p1, ggsql:penguins p2 VISUALISE DRAW point MAPPING x AS x";
        assert_eq!(
            extract_builtin_dataset_names(sql).unwrap(),
            vec!["penguins"]
        );
    }

    #[test]
    fn test_extract_builtin_dataset_names_none() {
        let sql = "SELECT * FROM regular_table VISUALISE DRAW point MAPPING x AS x";
        assert!(extract_builtin_dataset_names(sql).unwrap().is_empty());
    }

    #[test]
    fn test_rewrite_namespaced_sql_simple() {
        let sql = "SELECT * FROM ggsql:penguins";
        assert_eq!(
            rewrite_namespaced_sql(sql).unwrap(),
            "SELECT * FROM \"__ggsql_data_penguins__\""
        );
    }

    #[test]
    fn test_rewrite_namespaced_sql_multiple() {
        let sql = "SELECT * FROM ggsql:penguins p, ggsql:airquality a WHERE p.id = a.id";
        assert_eq!(
            rewrite_namespaced_sql(sql).unwrap(),
            "SELECT * FROM \"__ggsql_data_penguins__\" p, \"__ggsql_data_airquality__\" a WHERE p.id = a.id"
        );
    }

    #[test]
    fn test_rewrite_namespaced_sql_no_change() {
        let sql = "SELECT * FROM regular_table WHERE x > 5";
        assert_eq!(rewrite_namespaced_sql(sql).unwrap(), sql);
    }

    #[test]
    fn test_rewrite_namespaced_sql_with_visualise() {
        let sql = "SELECT * FROM ggsql:penguins VISUALISE DRAW point MAPPING bill_len AS x, bill_dep AS y";
        let rewritten = rewrite_namespaced_sql(sql).unwrap();
        assert!(rewritten.starts_with("SELECT * FROM \"__ggsql_data_penguins__\""));
        assert!(!rewritten.contains("ggsql:"));
    }

    #[test]
    fn test_extract_table_refs_basic() {
        let mut refs = extract_table_refs("SELECT * FROM a JOIN \"b c\" ON a.k = 1").unwrap();
        refs.sort();
        assert_eq!(refs, vec!["a".to_string(), "b c".to_string()]);
    }

    #[test]
    fn test_extract_table_ref_sites_alias_and_range() {
        let sites =
            extract_table_ref_sites("SELECT * FROM orders o JOIN items ON o.k = 1").unwrap();
        assert_eq!(sites.len(), 2);
        assert_eq!(sites[0].raw, "orders");
        assert!(sites[0].has_alias);
        assert_eq!(sites[1].raw, "items");
        assert!(!sites[1].has_alias);
    }

    #[test]
    fn test_string_literal_ranges_finds_literals() {
        let sql = "SELECT * FROM t WHERE note = 'hello'";
        let ranges = string_literal_ranges(sql);
        assert_eq!(ranges.len(), 1);
        let (s, e) = ranges[0];
        assert_eq!(&sql[s..e], "'hello'");
    }
}
