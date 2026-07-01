//! CTE (Common Table Expression) extraction, transformation, and materialization.
//!
//! This module handles extracting CTE definitions from SQL using tree-sitter,
//! materializing them as temporary tables, and transforming CTE references
//! in SQL queries.

use crate::reader::Reader;
use crate::{naming, parser::SourceTree, GgsqlError, Result};
use std::collections::HashSet;
use tree_sitter::Node;

/// Extracted CTE (Common Table Expression) definition
#[derive(Debug, Clone)]
pub struct CteDefinition {
    /// Name of the CTE
    pub name: String,
    /// Full SQL text of the CTE body (including the SELECT statement inside)
    pub body: String,
    /// Optional column aliases: WITH t(value, label) AS (...) → ["value", "label"]
    pub column_aliases: Vec<String>,
}

/// Extract CTE definitions from the source tree
///
/// Extracts all CTE definitions from WITH clauses using the existing parse tree.
/// Returns CTEs in declaration order (important for dependency resolution).
pub fn extract_ctes(source_tree: &SourceTree) -> Vec<CteDefinition> {
    let root = source_tree.root();

    // Use declarative tree-sitter query to find all CTE definitions
    source_tree
        .find_nodes(&root, "(cte_definition) @cte")
        .into_iter()
        .filter_map(|node| parse_cte_definition(&node, source_tree.source))
        .collect()
}

/// Parse a single CTE definition node into a CteDefinition
fn parse_cte_definition(node: &Node, source: &str) -> Option<CteDefinition> {
    let mut name: Option<String> = None;
    let mut column_aliases: Vec<String> = Vec::new();
    let mut body_start: Option<usize> = None;
    let mut body_end: Option<usize> = None;

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                // First identifier is the CTE name, subsequent ones are column aliases
                if name.is_none() {
                    name = Some(get_node_text(&child, source).to_string());
                } else {
                    column_aliases.push(get_node_text(&child, source).to_string());
                }
            }
            "select_statement" | "subquery_body" | "with_statement" => {
                body_start = Some(child.start_byte());
                body_end = Some(child.end_byte());
            }
            _ => {}
        }
    }

    match (name, body_start, body_end) {
        (Some(n), Some(start), Some(end)) => {
            let body = source[start..end].to_string();
            Some(CteDefinition {
                name: n,
                body,
                column_aliases,
            })
        }
        _ => None,
    }
}

/// Get text content of a node
pub(crate) fn get_node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

/// Transform CTE references in SQL to use temp table names.
///
/// Replaces references to CTEs (e.g. `FROM sales`, `JOIN sales`, `sales.col`)
/// with the corresponding temp table names (e.g. `__ggsql_cte_sales__`).
///
/// Table references are found via the parser; column references are rewritten
/// tolerant of whitespace around the dot and never inside string literals.
pub fn transform_cte_references(sql: &str, cte_names: &HashSet<String>) -> String {
    if cte_names.is_empty() {
        return sql.to_string();
    }

    // On a parse failure leave the SQL unchanged.
    let Ok(sites) = crate::parser::extract_table_ref_sites(sql) else {
        return sql.to_string();
    };
    let string_ranges = crate::parser::string_literal_ranges(sql);
    let in_string = |pos: usize| string_ranges.iter().any(|&(s, e)| pos >= s && pos < e);

    // Match CTE names against the unquoted reference text, mirroring the
    // definition names.
    let temp_of = |raw: &str| -> Option<String> {
        let name = naming::unquote_ident(raw);
        cte_names
            .iter()
            .find(|c| naming::unquote_ident(c).eq_ignore_ascii_case(&name))
            .map(|c| naming::quote_ident(&naming::cte_table(c)))
    };

    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    // Rewrite each table_ref that names a CTE to its temp table.
    for site in &sites {
        if let Some(temp) = temp_of(&site.raw) {
            replacements.push((site.start, site.end, temp));
        }
    }

    // Rewrite qualified column references `cte.col` -> `temp.col`.
    let site_starts: HashSet<usize> = sites.iter().map(|s| s.start).collect();
    for cte in cte_names {
        let temp = naming::quote_ident(&naming::cte_table(cte));
        let bare = naming::unquote_ident(cte);
        let pattern = format!(r"((?i:{}))\s*\.", regex::escape(&bare));
        let Ok(re) = regex::Regex::new(&pattern) else {
            continue;
        };
        for caps in re.captures_iter(sql) {
            let name = caps.get(1).unwrap();
            let start = name.start();
            if site_starts.contains(&start) || in_string(start) {
                continue;
            }
            // Require a boundary before the name.
            let ok_prefix = sql[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '"'));
            if ok_prefix {
                replacements.push((name.start(), name.end(), temp.clone()));
            }
        }
    }

    // Apply in reverse byte order so earlier offsets stay valid.
    let mut result = sql.to_string();
    replacements.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    // Distinct offsets only.
    replacements.dedup_by_key(|(start, _, _)| *start);

    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }

    result
}

/// Byte offsets of the `.` separators in a (possibly quoted) dotted identifier.
fn unquoted_dot_positions(raw: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut in_quote = false;
    for (i, b) in raw.bytes().enumerate() {
        match b {
            b'"' => in_quote = !in_quote,
            b'.' if !in_quote => positions.push(i),
            _ => {}
        }
    }
    positions
}

/// Split a (possibly quoted) dotted identifier into its components, trimming
/// surrounding whitespace: `schema . base` → `["schema","base"]`.
fn identifier_components(raw: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for pos in unquoted_dot_positions(raw) {
        parts.push(raw[start..pos].trim());
        start = pos + 1;
    }
    parts.push(raw[start..].trim());
    parts
}

/// The final component of a (possibly quoted) dotted identifier: `base` for
/// `schema.base`, `"base"` for `"schema"."base"`, and the whole (trimmed) string
/// for a single (possibly quoted) name.
fn last_identifier_component(raw: &str) -> &str {
    match unquoted_dot_positions(raw).last() {
        Some(&pos) => raw[pos + 1..].trim(),
        None => raw.trim(),
    }
}

/// Stage the primary base tables in a body destined for the cache.
///
/// A body that is entirely primary (no cache-resident reference) or entirely
/// cache-resident is returned unchanged.
pub fn transform_source_references(sql: &str, reader: &dyn Reader) -> Result<String> {
    if !reader.caches_sources() {
        return Ok(sql.to_string());
    }

    // Discover table references via the parser.
    let sites = match crate::parser::extract_table_ref_sites(sql) {
        Ok(sites) => sites,
        Err(_) => return Ok(sql.to_string()),
    };

    // A ref resolves against the cache backend (rather than the primary) when it
    // is an internal `__ggsql_*` table, a `ggsql:` builtin, or a file-path string.
    let is_cache_resolvable = |raw: &str| {
        raw.starts_with('\'')
            || raw.starts_with("ggsql:")
            || naming::is_internal_table(&naming::unquote_ident(raw))
    };

    let has_cache_ref = sites.iter().any(|s| is_cache_resolvable(&s.raw));
    let primary_sites: Vec<&crate::parser::TableRefSite> = sites
        .iter()
        .filter(|s| !is_cache_resolvable(&s.raw))
        .collect();

    // Only mixed bodies need staging.
    if !has_cache_ref || primary_sites.is_empty() {
        return Ok(sql.to_string());
    }

    // The final identifier component of a ref. This is how an unaliased ref
    // is spelled at column sites, so it doubles as the alias to attach.
    let last_of = |raw: &str| -> String { last_identifier_component(raw).to_string() };

    // Stage each distinct primary table into the cache once.
    let mut staged_for: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for site in &primary_sites {
        if !staged_for.contains_key(&site.raw) {
            let t = naming::staged_source_table(staged_for.len());
            reader.materialize_table(&t, &[], &format!("SELECT * FROM {}", site.raw))?;
            staged_for.insert(site.raw.clone(), t);
        }
    }

    // An unaliased ref is aliased back to its last component so `base.col` keeps
    // resolving. Two unaliased refs sharing a last component (`a.base`+`b.base`)
    // would collide, so those use their unique staged name instead.
    let mut last_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut seen_unaliased: HashSet<&str> = HashSet::new();
    for site in &primary_sites {
        if !site.has_alias && seen_unaliased.insert(site.raw.as_str()) {
            *last_counts.entry(last_of(&site.raw)).or_default() += 1;
        }
    }
    let last_collides = |raw: &str| last_counts.get(&last_of(raw)).copied().unwrap_or(0) > 1;

    let string_ranges = crate::parser::string_literal_ranges(sql);
    let in_string = |pos: usize| string_ranges.iter().any(|&(s, e)| pos >= s && pos < e);
    let site_starts: HashSet<usize> = primary_sites.iter().map(|s| s.start).collect();

    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    // Rewrite each table_ref occurrence to the staged table.
    for site in &primary_sites {
        let quoted = naming::quote_ident(&staged_for[&site.raw]);
        let replacement = if site.has_alias || last_collides(&site.raw) {
            quoted
        } else {
            format!("{} AS {}", quoted, last_of(&site.raw))
        };
        replacements.push((site.start, site.end, replacement));
    }

    // Rewrite full column qualifiers (`schema.base.col`) to the alias, or the
    // staged table when it collides — skipping string literals and table_ref sites.
    for raw in staged_for.keys() {
        // Only multi-part refs can appear as a full qualifier at column sites.
        if unquoted_dot_positions(raw).is_empty() {
            continue;
        }
        let target = if last_collides(raw) {
            naming::quote_ident(&staged_for[raw])
        } else {
            last_of(raw)
        };
        // Match the qualifier tolerant of whitespace around its dots.
        let pattern = format!(
            r"({})\s*\.",
            identifier_components(raw)
                .iter()
                .map(|c| {
                    if c.starts_with('"') {
                        regex::escape(c)
                    } else {
                        format!("(?i:{})", regex::escape(c))
                    }
                })
                .collect::<Vec<_>>()
                .join(r"\s*\.\s*")
        );
        let Ok(re) = regex::Regex::new(&pattern) else {
            continue;
        };
        for caps in re.captures_iter(sql) {
            let qualifier = caps.get(1).unwrap();
            let start = qualifier.start();
            if site_starts.contains(&start) || in_string(start) {
                continue;
            }
            // Require a boundary before the qualifier.
            let ok_prefix = sql[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '"'));
            if ok_prefix {
                replacements.push((start, qualifier.end(), target.clone()));
            }
        }
    }

    // Apply in reverse byte order so earlier offsets stay valid.
    let mut result = sql.to_string();
    replacements.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }

    Ok(result)
}

/// Materialize CTEs as temporary tables in the database
///
/// Creates a temp table for each CTE in declaration order. When a CTE
/// references an earlier CTE, the reference is transformed to use the
/// temp table name.
///
/// Returns the set of CTE names that were materialized.
pub fn materialize_ctes(ctes: &[CteDefinition], reader: &dyn Reader) -> Result<HashSet<String>> {
    let mut materialized = HashSet::new();

    for cte in ctes {
        // Transform the CTE body to replace references to earlier CTEs
        let transformed_body = transform_cte_references(&cte.body, &materialized);
        // Stage any primary base tables this body joins against into the cache.
        let transformed_body = transform_source_references(&transformed_body, reader)?;

        let temp_table_name = naming::cte_table(&cte.name);

        reader
            .materialize_table(&temp_table_name, &cte.column_aliases, &transformed_body)
            .map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to materialize CTE '{}': {}", cte.name, e))
            })?;

        materialized.insert(cte.name.clone());
    }

    Ok(materialized)
}

/// Split a WITH...SELECT query into its CTE prefix and trailing SELECT.
///
/// Given SQL like `WITH a AS (...), b AS (...) SELECT * FROM a`, returns:
/// - CTE prefix: `"WITH a AS (...), b AS (...)"`
/// - Trailing SELECT: `"SELECT * FROM a"`
///
/// Returns `None` if the query is not a WITH statement, has no trailing SELECT,
/// or parsing fails.
pub fn split_with_query(source_tree: &SourceTree) -> Option<(String, String)> {
    let root = source_tree.root();
    let with_node = source_tree.find_node(&root, "(with_statement) @with")?;

    let mut cursor = with_node.walk();
    let mut last_cte_end: Option<usize> = None;
    let mut tail_node = None;
    let mut seen_cte = false;

    for child in with_node.children(&mut cursor) {
        match child.kind() {
            "cte_definition" => {
                seen_cte = true;
                last_cte_end = Some(child.end_byte());
            }
            // WITH's tail may be a SELECT or a bare FROM (DuckDB-style).
            // For from_statement, we rewrite the tail to `SELECT * <from_stmt>`.
            "select_statement" if seen_cte => {
                tail_node = Some((child, false));
                break;
            }
            "from_statement" if seen_cte => {
                tail_node = Some((child, true));
                break;
            }
            _ => {}
        }
    }

    let cte_prefix = source_tree.source[with_node.start_byte()..last_cte_end?].to_string();
    let (node, is_from) = tail_node?;
    let trailing = if is_from {
        format!("SELECT * {}", source_tree.get_text(&node))
    } else {
        source_tree.get_text(&node)
    };
    Some((cte_prefix, trailing))
}

/// Collect side-effect statements (CREATE, INSERT, UPDATE, DELETE) that
/// need to run before the main query.
///
/// Only structured DML is handled here — other_sql_statement nodes
/// (INSTALL, LOAD, SET, …) are pre-executed in prepare_data_with_reader.
pub fn extract_side_effects(source_tree: &SourceTree) -> Vec<String> {
    let root = source_tree.root();
    let side_effect_stmts = r#"
        (sql_statement
          [(create_statement)
           (insert_statement)
           (update_statement)
           (delete_statement)] @stmt)
    "#;
    source_tree
        .find_texts(&root, side_effect_stmts)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Transform global SQL for execution as the global temp table.
///
/// Returns the query whose result should be wrapped as the global temp table,
/// with CTE references rewritten and primary base tables staged.
pub fn transform_global_sql(
    source_tree: &SourceTree,
    materialized_ctes: &HashSet<String>,
    reader: &dyn Reader,
) -> Result<Option<String>> {
    let root = source_tree.root();

    // Try to extract trailing SELECT (WITH...SELECT or direct SELECT)
    let select_sql = split_with_query(source_tree)
        .map(|(_, select)| select)
        .or_else(|| {
            // Fallback: direct SELECT statement (no WITH clause)
            source_tree.find_text(&root, "(sql_statement (select_statement) @select)")
        });

    if let Some(select_sql) = select_sql {
        let select_sql = transform_cte_references(&select_sql, materialized_ctes);
        return Ok(Some(transform_source_references(&select_sql, reader)?));
    }

    if !has_executable_sql(source_tree) {
        return Ok(None);
    }

    // We have non-SELECT executable SQL and/or VISUALISE FROM. VISUALISE FROM
    // becomes the queryable part; a bare WITH clause without a trailing
    // statement is not executable on its own (its CTEs are materialized
    // separately).
    let viz_from_query = source_tree
        .find_text(
            &root,
            r#"(visualise_statement (from_clause (table_ref) @table))"#,
        )
        .map(|table| {
            let q = format!("SELECT * FROM {}", table);
            let q = transform_cte_references(&q, materialized_ctes);
            transform_source_references(&q, reader)
        })
        .transpose()?;

    if viz_from_query.is_some() || !extract_side_effects(source_tree).is_empty() {
        Ok(viz_from_query)
    } else {
        // has_executable_sql was true but we found no specific statements or
        // VISUALISE FROM — fall back to extract_sql as the query.
        source_tree
            .extract_sql()
            .map(|s| {
                let s = transform_cte_references(&s, materialized_ctes);
                transform_source_references(&s, reader)
            })
            .transpose()
    }
}

/// Check if SQL contains executable statements (SELECT, INSERT, UPDATE, DELETE, CREATE)
///
/// Returns false if the SQL is just CTE definitions without a trailing statement.
/// This handles cases like `WITH a AS (...), b AS (...) VISUALISE` where the WITH
/// clause has no trailing SELECT - these CTEs are still extracted for layer use
/// but shouldn't be executed as global data.
pub fn has_executable_sql(source_tree: &SourceTree) -> bool {
    let root = source_tree.root();

    // Check for direct executable statements (SELECT, CREATE, INSERT, UPDATE,
    // DELETE, or bare FROM (DuckDB-style FROM-first — rewritten to SELECT *))
    let direct_statements = r#"
        (sql_statement
          [(select_statement)
           (create_statement)
           (insert_statement)
           (update_statement)
           (delete_statement)
           (from_statement)] @stmt)
    "#;

    if source_tree.find_node(&root, direct_statements).is_some() {
        return true;
    }

    // Check for WITH statements that have trailing SELECT
    if split_with_query(source_tree).is_some() {
        return true;
    }

    // Check for VISUALISE FROM (which injects SELECT * FROM <source>)
    let visualise_from = r#"
        (visualise_statement
          (from_clause) @from)
    "#;
    if source_tree.find_node(&root, visualise_from).is_some() {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ctes_single() {
        let sql = "WITH sales AS (SELECT * FROM raw_sales) SELECT * FROM sales";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].name, "sales");
        assert!(ctes[0].body.contains("SELECT * FROM raw_sales"));
    }

    #[test]
    fn test_extract_ctes_multiple() {
        let sql = "WITH
            sales AS (SELECT * FROM raw_sales),
            targets AS (SELECT * FROM goals)
        SELECT * FROM sales";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 2);
        // Verify order is preserved
        assert_eq!(ctes[0].name, "sales");
        assert_eq!(ctes[1].name, "targets");
    }

    #[test]
    fn test_extract_ctes_with_column_aliases() {
        let sql = "WITH t(value, label) AS (SELECT * FROM (VALUES (70, 'Target'))) SELECT * FROM t";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].name, "t");
        assert_eq!(ctes[0].column_aliases, vec!["value", "label"]);
    }

    #[test]
    fn test_extract_ctes_without_column_aliases() {
        let sql = "WITH sales AS (SELECT * FROM raw_sales) SELECT * FROM sales";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].name, "sales");
        assert!(ctes[0].column_aliases.is_empty());
    }

    #[test]
    fn test_extract_ctes_none() {
        let sql = "SELECT * FROM sales WHERE year = 2024";
        let source_tree = SourceTree::new(sql).unwrap();
        let ctes = extract_ctes(&source_tree);

        assert!(ctes.is_empty());
    }

    #[test]
    fn test_transform_cte_references() {
        // Test cases: (sql, cte_names, expected_contains, exact_match)
        let test_cases: Vec<(
            &str,
            Vec<&str>,
            Vec<&str>,    // strings that should be in result
            Option<&str>, // exact match (if result should equal this)
        )> = vec![
            // Single CTE reference
            (
                "SELECT * FROM sales WHERE year = 2024",
                vec!["sales"],
                vec!["FROM \"__ggsql_cte_sales_", "__\" WHERE year = 2024"],
                None,
            ),
            // Multiple CTE references with qualified columns
            (
                "SELECT sales.date, targets.revenue FROM sales JOIN targets ON sales.id = targets.id",
                vec!["sales", "targets"],
                vec![
                    "FROM \"__ggsql_cte_sales_",
                    "JOIN \"__ggsql_cte_targets_",
                    "__ggsql_cte_sales_",  // qualified reference sales.date
                    "__ggsql_cte_targets_", // qualified reference targets.revenue
                ],
                None,
            ),
            // Qualified column references only (no FROM/JOIN transformation needed)
            (
                "WHERE sales.date > '2024-01-01' AND sales.revenue > 100",
                vec!["sales"],
                vec!["__ggsql_cte_sales_"],
                None,
            ),
            // No matching CTE (unchanged)
            (
                "SELECT * FROM other_table",
                vec!["sales"],
                vec![],
                Some("SELECT * FROM other_table"),
            ),
            // Empty CTE names (unchanged)
            (
                "SELECT * FROM sales",
                vec![],
                vec![],
                Some("SELECT * FROM sales"),
            ),
            // No false positives on substrings (wholesale should not match 'sales')
            (
                "SELECT wholesale.date FROM wholesale",
                vec!["sales"],
                vec![],
                Some("SELECT wholesale.date FROM wholesale"),
            ),
        ];

        for (sql, cte_names_vec, expected_contains, exact_match) in test_cases {
            let cte_names: HashSet<String> = cte_names_vec.iter().map(|s| s.to_string()).collect();
            let result = transform_cte_references(sql, &cte_names);

            if let Some(expected) = exact_match {
                assert_eq!(result, expected, "SQL '{}' should remain unchanged", sql);
            } else {
                for expected in &expected_contains {
                    assert!(
                        result.contains(expected),
                        "Result '{}' should contain '{}' for SQL '{}'",
                        result,
                        expected,
                        sql
                    );
                }
                // When CTEs are transformed, result should contain session UUID
                if !cte_names_vec.is_empty() {
                    assert!(
                        result.contains(naming::session_id()),
                        "Result should contain session UUID"
                    );
                }
            }
        }
    }

    #[test]
    fn test_transform_cte_references_comma_join_second_position() {
        // A CTE in a non-first comma position must still be rewritten.
        let ctes: HashSet<String> = ["cte"].iter().map(|s| s.to_string()).collect();
        let out = transform_cte_references("SELECT * FROM base, cte WHERE base.k = cte.k", &ctes);
        assert!(
            !out.contains("FROM base, cte "),
            "cte table ref not rewritten: {out}"
        );
        assert!(out.contains("__ggsql_cte_cte_"));
        // Both the table ref and the qualified column ref are rewritten.
        assert_eq!(out.matches("__ggsql_cte_cte_").count(), 2);
    }

    #[test]
    fn test_transform_cte_references_preserves_string_literals() {
        // A CTE name inside a string literal must not be rewritten.
        let ctes: HashSet<String> = ["cte"].iter().map(|s| s.to_string()).collect();
        let out = transform_cte_references("SELECT cte.k, 'cte.k' AS lit FROM cte", &ctes);
        assert!(out.contains("'cte.k'"), "literal was corrupted: {out}");
        // The real qualifier and table ref are still rewritten.
        assert_eq!(out.matches("__ggsql_cte_cte_").count(), 2);
    }

    #[test]
    fn test_transform_cte_references_whitespace_around_dot() {
        let ctes: HashSet<String> = ["cte"].iter().map(|s| s.to_string()).collect();
        let out = transform_cte_references("SELECT cte . v FROM cte", &ctes);
        // The whitespace-separated qualifier is rewritten too.
        assert!(!out.contains("cte . v"), "qualifier not rewritten: {out}");
        assert_eq!(out.matches("__ggsql_cte_cte_").count(), 2);
    }

    #[test]
    fn test_transform_cte_references_case_insensitive() {
        let ctes: HashSet<String> = ["cte"].iter().map(|s| s.to_string()).collect();
        let out = transform_cte_references("SELECT CTE.v FROM CTE", &ctes);
        assert_eq!(out.matches("__ggsql_cte_cte_").count(), 2);
    }

    /// Minimal reader that records `materialize_table` calls, used to unit-test
    /// source-reference staging without a database.
    struct MockReader {
        caches: bool,
        staged: std::cell::RefCell<Vec<(String, String)>>,
    }

    impl MockReader {
        fn new(caches: bool) -> Self {
            Self {
                caches,
                staged: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    impl Reader for MockReader {
        fn execute_sql(&self, _sql: &str) -> Result<crate::DataFrame> {
            unreachable!("staging must not touch execute_sql in these tests")
        }
        fn register(&self, _name: &str, _df: crate::DataFrame, _replace: bool) -> Result<()> {
            Ok(())
        }
        fn execute(&self, _query: &str) -> Result<crate::reader::Spec> {
            unreachable!()
        }
        fn caches_sources(&self) -> bool {
            self.caches
        }
        fn materialize_table(
            &self,
            name: &str,
            _column_aliases: &[String],
            body_sql: &str,
        ) -> Result<()> {
            self.staged
                .borrow_mut()
                .push((name.to_string(), body_sql.to_string()));
            Ok(())
        }
    }

    #[test]
    fn test_transform_source_references_quoted_primary_ref() {
        let reader = MockReader::new(true);
        // A quoted primary base table joined against a cache-resident CTE temp.
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN \"my base\" ON 1 = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].1, "SELECT * FROM \"my base\"");
        assert!(out.contains("__ggsql_staged_0_"));
        assert!(!out.contains("JOIN \"my base\""));
    }

    #[test]
    fn test_transform_source_references_non_caching_reader_unchanged() {
        let reader = MockReader::new(false);
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN base ON base.k = 1";
        let out = transform_source_references(sql, &reader).unwrap();
        assert_eq!(out, sql);
        assert!(reader.staged.borrow().is_empty());
    }

    #[test]
    fn test_transform_source_references_all_primary_unchanged() {
        let reader = MockReader::new(true);
        let out = transform_source_references("SELECT * FROM a JOIN base ON a.k = base.k", &reader)
            .unwrap();
        assert_eq!(out, "SELECT * FROM a JOIN base ON a.k = base.k");
        assert!(reader.staged.borrow().is_empty());
    }

    #[test]
    fn test_transform_source_references_stages_mixed_body() {
        let reader = MockReader::new(true);
        // A cache-resident CTE temp joined against a primary base table.
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN base ON base.k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1, "base should be staged exactly once");
        assert!(staged[0].0.starts_with("__ggsql_staged_0_"));
        assert_eq!(staged[0].1, "SELECT * FROM base");

        assert!(out.contains("__ggsql_staged_0_"));
        assert!(out.contains("\"__ggsql_cte_t__\"")); // cte ref preserved
        assert!(!out.contains("JOIN base"));
    }

    #[test]
    fn test_transform_source_references_reversed_from_join() {
        let reader = MockReader::new(true);
        // Primary base table in FROM, cache-resident CTE temp in JOIN.
        let sql = "SELECT * FROM base JOIN \"__ggsql_cte_t__\" ON base.k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        assert_eq!(reader.staged.borrow().len(), 1);
        assert!(out.contains("FROM \"__ggsql_staged_0_"));
        assert!(!out.contains("FROM base"));
    }

    #[test]
    fn test_transform_source_references_case_insensitive_qualifier() {
        let reader = MockReader::new(true);
        // The full qualifier is spelled in a different case than the table_ref;
        // unquoted identifiers fold, so it must still be rewritten.
        let sql = "SELECT MYSCHEMA.BASE.w FROM \"__ggsql_cte_t__\" \
                   JOIN myschema.base ON MYSCHEMA.BASE.k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        assert_eq!(reader.staged.borrow().len(), 1);
        assert!(
            !out.contains("MYSCHEMA"),
            "case-variant qualifier not rewritten: {out}"
        );
    }

    #[test]
    fn test_last_identifier_component() {
        assert_eq!(last_identifier_component("base"), "base");
        assert_eq!(last_identifier_component("schema.base"), "base");
        assert_eq!(last_identifier_component("cat.schema.base"), "base");
        assert_eq!(last_identifier_component("\"schema\".\"base\""), "\"base\"");
        assert_eq!(last_identifier_component("schema.\"base\""), "\"base\"");
        // A dot inside a quoted component is not a separator.
        assert_eq!(last_identifier_component("\"my.base\""), "\"my.base\"");
    }

    #[test]
    fn test_transform_source_references_fully_quoted_qualified() {
        let reader = MockReader::new(true);
        // `"s"."t"` must stage and alias to the (quoted) last component.
        let sql = "SELECT \"myschema\".\"base\".w FROM \"__ggsql_cte_t__\" \
                   JOIN \"myschema\".\"base\" ON \"myschema\".\"base\".k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].1, "SELECT * FROM \"myschema\".\"base\"");
        // The staged table is aliased AS "base" and full qualifiers rewritten.
        assert!(out.contains("AS \"base\""));
        assert!(!out.contains("\"myschema\".\"base\""));
    }

    #[test]
    fn test_transform_source_references_whitespace_around_dot() {
        let reader = MockReader::new(true);
        // Whitespace/newlines around the dots (in both the table_ref and the full
        // column qualifier) must not defeat staging or the qualifier rewrite.
        let sql = "SELECT myschema .\nbase . w FROM \"__ggsql_cte_t__\" \
                   JOIN myschema . base ON myschema . base . k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1);
        // The full column qualifiers are rewritten to the alias despite spacing.
        assert!(!out.contains("myschema"), "qualifier not rewritten: {out}");
        assert!(out.contains("AS base"));
    }

    #[test]
    fn test_transform_source_references_schema_qualified() {
        let reader = MockReader::new(true);
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN myschema.base ON base.k = 1";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].1, "SELECT * FROM myschema.base");
        assert!(out.contains("__ggsql_staged_0_"));
        assert!(!out.contains("JOIN myschema.base"));
    }

    #[test]
    fn test_transform_source_references_same_ref_staged_once() {
        let reader = MockReader::new(true);
        let sql = "SELECT * FROM base JOIN \"__ggsql_cte_t__\" ON base.k = base.j";
        let _ = transform_source_references(sql, &reader).unwrap();
        assert_eq!(reader.staged.borrow().len(), 1);
    }

    #[test]
    fn test_transform_source_references_comma_join() {
        // A comma join between a cache-resident temp and a primary base table:
        // the primary side must be staged (the old FROM/JOIN regex missed this).
        let reader = MockReader::new(true);
        let sql = "SELECT * FROM \"__ggsql_cte_t__\", base";
        let out = transform_source_references(sql, &reader).unwrap();

        let staged = reader.staged.borrow();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].1, "SELECT * FROM base");
        assert!(out.contains("__ggsql_staged_0_"));
        assert!(out.contains("\"__ggsql_cte_t__\""));
    }

    #[test]
    fn test_transform_source_references_preserves_string_literals() {
        // A string literal that happens to look like a qualified reference must
        // not be rewritten.
        let reader = MockReader::new(true);
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN myschema.base \
                   ON note = 'myschema.base.k'";
        let out = transform_source_references(sql, &reader).unwrap();

        assert_eq!(reader.staged.borrow().len(), 1);
        // The literal is untouched; the real qualifier in the table_ref is staged.
        assert!(out.contains("'myschema.base.k'"));
        assert!(!out.contains("JOIN myschema.base "));
    }

    #[test]
    fn test_transform_source_references_builtin_not_staged() {
        // A `ggsql:` builtin resolves against the cache, so a JOIN against it and
        // a resident temp is entirely cache-side and needs no staging.
        let reader = MockReader::new(true);
        let sql = "SELECT * FROM \"__ggsql_cte_t__\" JOIN ggsql:penguins ON 1 = 1";
        let out = transform_source_references(sql, &reader).unwrap();
        assert!(reader.staged.borrow().is_empty());
        assert_eq!(out, sql);
    }

    #[test]
    fn test_split_with_query_basic() {
        let sql = "WITH cte AS (SELECT * FROM x) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT * FROM x)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_multiple_ctes() {
        let sql = "WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a JOIN b";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH a AS (SELECT 1), b AS (SELECT 2)");
        assert_eq!(select, "SELECT * FROM a JOIN b");
    }

    #[test]
    fn test_split_with_query_nested_subquery() {
        let sql = "WITH cte AS (SELECT * FROM (SELECT 1)) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT * FROM (SELECT 1))");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_string_with_select_keyword() {
        let sql = "WITH cte AS (SELECT 'SELECT' AS col) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT 'SELECT' AS col)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_string_with_parens() {
        let sql = "WITH cte AS (SELECT '()' AS col) SELECT * FROM cte";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert_eq!(prefix, "WITH cte AS (SELECT '()' AS col)");
        assert_eq!(select, "SELECT * FROM cte");
    }

    #[test]
    fn test_split_with_query_not_a_with() {
        let sql = "SELECT * FROM x";
        let source_tree = SourceTree::new(sql).unwrap();
        assert!(split_with_query(&source_tree).is_none());
    }

    #[test]
    fn test_split_with_query_no_trailing_select() {
        let sql = "WITH cte AS (SELECT 1) VISUALISE DRAW point";
        let source_tree = SourceTree::new(sql).unwrap();
        assert!(split_with_query(&source_tree).is_none());
    }

    #[test]
    fn test_split_with_query_stat_transform_output() {
        // Realistic stat transform output (histogram pattern)
        let sql = "WITH __stat_src__ AS (SELECT x FROM data), \
                   __binned__ AS (SELECT x, COUNT(*) AS count FROM __stat_src__ GROUP BY x) \
                   SELECT *, count * 1.0 / SUM(count) OVER () AS density FROM __binned__";
        let source_tree = SourceTree::new(sql).unwrap();
        let (prefix, select) = split_with_query(&source_tree).unwrap();

        assert!(prefix.starts_with("WITH __stat_src__"));
        assert!(prefix.contains("__binned__"));
        assert!(prefix.ends_with(")"));
        assert!(select.starts_with("SELECT *"));
        assert!(select.contains("density"));
    }
}
