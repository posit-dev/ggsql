//! Database schema introspection for the Positron Connections pane.
//!
//! Adapts the Reader's catalog/schema/table hierarchy to the Positron
//! connections protocol. Hierarchy levels that the driver doesn't
//! support (returning zero results) are skipped so that tables are
//! shown directly at the root.

use ggsql::reader::Reader;
use serde::Serialize;
use serde_json::Value;

/// An object in the schema hierarchy (catalog, schema, table, or view).
#[derive(Debug, Serialize)]
pub struct ObjectSchema {
    pub name: String,
    pub kind: String,
}

/// A field (column) in a table.
#[derive(Debug, Serialize)]
pub struct FieldSchema {
    pub name: String,
    pub dtype: String,
}

/// How many leading hierarchy levels to skip because the driver
/// returns no results for them.
fn depth_offset(reader: &dyn Reader) -> usize {
    let catalogs = reader.list_catalogs().unwrap_or_default();
    if catalogs.is_empty() {
        let schemas = reader.list_schemas("").unwrap_or_default();
        if schemas.is_empty() { 2 } else { 1 }
    } else {
        0
    }
}

fn full_path<'a>(offset: usize, path: &'a [String]) -> Vec<&'a str> {
    std::iter::repeat("")
        .take(offset)
        .chain(path.iter().map(|s| s.as_str()))
        .collect()
}

/// Resolve a UI path to a full `[catalog, schema, ...]` path, padding
/// skipped hierarchy levels with empty strings.
pub fn resolve_path<'a>(reader: &dyn Reader, path: &'a [String]) -> Vec<&'a str> {
    full_path(depth_offset(reader), path)
}

/// List objects at the given path depth, skipping empty hierarchy levels.
pub fn list_objects(reader: &dyn Reader, path: &[String]) -> Result<Vec<ObjectSchema>, String> {
    let full = full_path(depth_offset(reader), path);
    match full.len() {
        0 => list_catalogs(reader),
        1 => list_schemas(reader, full[0]),
        2 => list_tables(reader, full[0], full[1]),
        _ => Ok(vec![]),
    }
}

/// List fields (columns) for the object at the given path.
pub fn list_fields(reader: &dyn Reader, path: &[String]) -> Result<Vec<FieldSchema>, String> {
    let full = full_path(depth_offset(reader), path);
    if full.len() != 3 {
        return Ok(vec![]);
    }
    list_columns(reader, full[0], full[1], full[2])
}

/// Whether the path points to an object that contains data (table or view).
pub fn contains_data(path: &[Value]) -> bool {
    path.last()
        .and_then(|v| v.get("kind"))
        .and_then(|k| k.as_str())
        .map(|k| k == "table" || k == "view")
        .unwrap_or(false)
}

fn list_catalogs(reader: &dyn Reader) -> Result<Vec<ObjectSchema>, String> {
    let catalogs = reader
        .list_catalogs()
        .map_err(|e| format!("Failed to list catalogs: {}", e))?;

    Ok(catalogs
        .into_iter()
        .map(|name| ObjectSchema {
            name,
            kind: "catalog".to_string(),
        })
        .collect())
}

fn list_schemas(reader: &dyn Reader, catalog: &str) -> Result<Vec<ObjectSchema>, String> {
    let schemas = reader
        .list_schemas(catalog)
        .map_err(|e| format!("Failed to list schemas: {}", e))?;

    Ok(schemas
        .into_iter()
        .map(|name| ObjectSchema {
            name,
            kind: "schema".to_string(),
        })
        .collect())
}

fn list_tables(
    reader: &dyn Reader,
    catalog: &str,
    schema: &str,
) -> Result<Vec<ObjectSchema>, String> {
    let tables = reader
        .list_tables(catalog, schema)
        .map_err(|e| format!("Failed to list tables: {}", e))?;

    Ok(tables
        .into_iter()
        .filter_map(|t| {
            let upper = t.table_type.to_uppercase();
            let kind = if upper.contains("VIEW") {
                "view"
            } else if upper == "TABLE" || upper == "BASE TABLE" || upper.contains("TABLE") {
                "table"
            } else {
                return None;
            };
            Some(ObjectSchema {
                name: t.name,
                kind: kind.to_string(),
            })
        })
        .collect())
}

fn list_columns(
    reader: &dyn Reader,
    catalog: &str,
    schema: &str,
    table: &str,
) -> Result<Vec<FieldSchema>, String> {
    let columns = reader
        .list_columns(catalog, schema, table)
        .map_err(|e| format!("Failed to list columns: {}", e))?;

    Ok(columns
        .into_iter()
        .map(|c| FieldSchema {
            name: c.name,
            dtype: c.data_type,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_data_table() {
        let path = vec![
            serde_json::json!({"name": "memory", "kind": "catalog"}),
            serde_json::json!({"name": "main", "kind": "schema"}),
            serde_json::json!({"name": "users", "kind": "table"}),
        ];
        assert!(contains_data(&path));
    }

    #[test]
    fn test_contains_data_schema() {
        let path = vec![
            serde_json::json!({"name": "memory", "kind": "catalog"}),
            serde_json::json!({"name": "main", "kind": "schema"}),
        ];
        assert!(!contains_data(&path));
    }

    #[test]
    fn test_contains_data_catalog() {
        let path = vec![serde_json::json!({"name": "memory", "kind": "catalog"})];
        assert!(!contains_data(&path));
    }

    #[test]
    fn test_contains_data_view() {
        let path = vec![
            serde_json::json!({"name": "memory", "kind": "catalog"}),
            serde_json::json!({"name": "main", "kind": "schema"}),
            serde_json::json!({"name": "my_view", "kind": "view"}),
        ];
        assert!(contains_data(&path));
    }
}
