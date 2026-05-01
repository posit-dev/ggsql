//! Database schema introspection for the Positron Connections pane.
//!
//! Adapts the Reader's catalog/schema/table hierarchy to the Positron
//! connections protocol. Hierarchy levels that the driver doesn't
//! support (returning zero results) are skipped so that e.g. SQLite
//! (which has no real catalogs or schemas) shows tables directly at
//! the root.

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
/// returns no results for them (e.g. SQLite ODBC has no catalogs
/// or schemas, so both are skipped and tables show at the root).
fn depth_offset(reader: &dyn Reader) -> (usize, String, String) {
    let catalogs = reader.list_catalogs().unwrap_or_default();
    if catalogs.is_empty() {
        let schemas = reader.list_schemas("").unwrap_or_default();
        if schemas.is_empty() {
            (2, String::new(), String::new())
        } else {
            (1, String::new(), String::new())
        }
    } else {
        (0, String::new(), String::new())
    }
}

/// List objects at the given path depth, skipping empty hierarchy levels.
pub fn list_objects(reader: &dyn Reader, path: &[String]) -> Result<Vec<ObjectSchema>, String> {
    let (offset, default_catalog, default_schema) = depth_offset(reader);
    let effective = path.len() + offset;

    match effective {
        0 => list_catalogs(reader),
        1 => {
            let catalog = if offset >= 1 { &default_catalog } else { &path[0] };
            list_schemas(reader, catalog)
        }
        2 => {
            let (catalog, schema) = match offset {
                2 => (&default_catalog, &default_schema),
                1 => (&default_catalog, &path[0]),
                _ => (&path[0], &path[1]),
            };
            list_tables(reader, catalog, schema)
        }
        _ => Ok(vec![]),
    }
}

/// List fields (columns) for the object at the given path.
pub fn list_fields(reader: &dyn Reader, path: &[String]) -> Result<Vec<FieldSchema>, String> {
    let (offset, default_catalog, default_schema) = depth_offset(reader);
    let effective = path.len() + offset;

    if effective != 3 {
        return Ok(vec![]);
    }

    let (catalog, schema, table) = match offset {
        2 => (default_catalog.as_str(), default_schema.as_str(), path[0].as_str()),
        1 => (default_catalog.as_str(), path[0].as_str(), path[1].as_str()),
        _ => (path[0].as_str(), path[1].as_str(), path[2].as_str()),
    };

    list_columns(reader, catalog, schema, table)
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
