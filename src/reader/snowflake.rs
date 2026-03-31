//! Snowflake-specific SQL dialect.
//!
//! Overrides schema introspection to use Snowflake's SHOW commands
//! instead of information_schema queries.

pub struct SnowflakeDialect;

impl super::SqlDialect for SnowflakeDialect {
    fn sql_list_catalogs(&self) -> String {
        "SHOW DATABASES".into()
    }

    fn sql_list_schemas(&self, catalog: &str) -> String {
        let catalog_ident = catalog.replace('"', "\"\"");
        format!("SHOW SCHEMAS IN DATABASE \"{catalog_ident}\"")
    }

    fn sql_list_tables(&self, catalog: &str, schema: &str) -> String {
        let catalog_ident = catalog.replace('"', "\"\"");
        let schema_ident = schema.replace('"', "\"\"");
        format!("SHOW OBJECTS IN SCHEMA \"{catalog_ident}\".\"{schema_ident}\"")
    }

    fn sql_list_columns(&self, catalog: &str, schema: &str, table: &str) -> String {
        let catalog_ident = catalog.replace('"', "\"\"");
        let schema_ident = schema.replace('"', "\"\"");
        let table_ident = table.replace('"', "\"\"");
        format!("SHOW COLUMNS IN TABLE \"{catalog_ident}\".\"{schema_ident}\".\"{table_ident}\"")
    }
}
