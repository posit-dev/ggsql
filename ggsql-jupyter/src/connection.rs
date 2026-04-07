//! Database schema introspection for the Positron Connections pane.
//!
//! Delegates introspection SQL to the reader's `SqlDialect`, which provides
//! backend-specific queries (e.g. `information_schema` for DuckDB/PostgreSQL,
//! `sqlite_master` / `PRAGMA` for SQLite).

use crate::util::find_column;
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

/// List objects at the given path depth.
///
/// Path semantics (catalog → schema → table):
/// - `[]` → list catalogs
/// - `[catalog]` → list schemas in that catalog
/// - `[catalog, schema]` → list tables and views
pub fn list_objects(reader: &dyn Reader, path: &[String]) -> Result<Vec<ObjectSchema>, String> {
    match path.len() {
        0 => list_catalogs(reader),
        1 => list_schemas(reader, &path[0]),
        2 => list_tables(reader, &path[0], &path[1]),
        _ => Ok(vec![]),
    }
}

/// List fields (columns) for the object at the given path.
///
/// - `[catalog, schema, table]` → list columns
pub fn list_fields(reader: &dyn Reader, path: &[String]) -> Result<Vec<FieldSchema>, String> {
    if path.len() == 3 {
        list_columns(reader, &path[0], &path[1], &path[2])
    } else {
        Ok(vec![])
    }
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
    let sql = reader.dialect().sql_list_catalogs();
    let df = reader
        .execute_sql(&sql)
        .map_err(|e| format!("Failed to list catalogs: {}", e))?;

    let col = find_column(&df, &["catalog_name", "name"])
        .map_err(|e| format!("Missing catalog_name/name column: {}", e))?;

    let mut catalogs = Vec::new();
    for i in 0..df.height() {
        if let Ok(val) = col.get(i) {
            let name = val.to_string().trim_matches('"').to_string();
            catalogs.push(ObjectSchema {
                name,
                kind: "catalog".to_string(),
            });
        }
    }
    Ok(catalogs)
}

fn list_schemas(reader: &dyn Reader, catalog: &str) -> Result<Vec<ObjectSchema>, String> {
    let sql = reader.dialect().sql_list_schemas(catalog);
    let df = reader
        .execute_sql(&sql)
        .map_err(|e| format!("Failed to list schemas: {}", e))?;

    let col = find_column(&df, &["schema_name", "name"])
        .map_err(|e| format!("Missing schema_name/name column: {}", e))?;

    let mut schemas = Vec::new();
    for i in 0..df.height() {
        if let Ok(val) = col.get(i) {
            let name = val.to_string().trim_matches('"').to_string();
            schemas.push(ObjectSchema {
                name,
                kind: "schema".to_string(),
            });
        }
    }
    Ok(schemas)
}

fn list_tables(
    reader: &dyn Reader,
    catalog: &str,
    schema: &str,
) -> Result<Vec<ObjectSchema>, String> {
    let sql = reader.dialect().sql_list_tables(catalog, schema);
    let df = reader
        .execute_sql(&sql)
        .map_err(|e| format!("Failed to list tables: {}", e))?;

    let name_col = find_column(&df, &["table_name", "name"])
        .map_err(|e| format!("Missing table_name/name column: {}", e))?;
    let type_col = find_column(&df, &["table_type", "kind"])
        .map_err(|e| format!("Missing table_type/kind column: {}", e))?;

    let mut objects = Vec::new();
    for i in 0..df.height() {
        if let (Ok(name_val), Ok(type_val)) = (name_col.get(i), type_col.get(i)) {
            let name = name_val.to_string().trim_matches('"').to_string();
            let table_type = type_val.to_string().trim_matches('"').to_uppercase();
            let kind = if table_type.contains("VIEW") {
                "view"
            } else if table_type == "TABLE"
                || table_type == "BASE TABLE"
                || table_type.contains("TABLE")
            {
                "table"
            } else {
                continue; // Skip non-table/view objects (stages, procedures, etc.)
            };
            objects.push(ObjectSchema {
                name,
                kind: kind.to_string(),
            });
        }
    }
    Ok(objects)
}

fn list_columns(
    reader: &dyn Reader,
    catalog: &str,
    schema: &str,
    table: &str,
) -> Result<Vec<FieldSchema>, String> {
    let sql = reader.dialect().sql_list_columns(catalog, schema, table);
    let df = reader
        .execute_sql(&sql)
        .map_err(|e| format!("Failed to list columns: {}", e))?;

    let name_col = find_column(&df, &["column_name"])
        .map_err(|e| format!("Missing column_name column: {}", e))?;
    let type_col =
        find_column(&df, &["data_type"]).map_err(|e| format!("Missing data_type column: {}", e))?;

    let mut fields = Vec::new();
    for i in 0..df.height() {
        if let (Ok(name_val), Ok(type_val)) = (name_col.get(i), type_col.get(i)) {
            let name = name_val.to_string().trim_matches('"').to_string();
            let dtype = type_val.to_string().trim_matches('"').to_string();
            fields.push(FieldSchema { name, dtype });
        }
    }
    Ok(fields)
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
