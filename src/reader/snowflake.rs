//! Snowflake-specific SQL dialect.
//!
//! Overrides schema introspection to use Snowflake's SHOW commands
//! instead of information_schema queries.

use crate::naming;

pub struct SnowflakeDialect;

impl super::SqlDialect for SnowflakeDialect {
    fn sql_list_catalogs(&self) -> String {
        "SHOW DATABASES".into()
    }

    fn sql_list_schemas(&self, catalog: &str) -> String {
        format!("SHOW SCHEMAS IN DATABASE {}", naming::quote_ident(catalog))
    }

    fn sql_list_tables(&self, catalog: &str, schema: &str) -> String {
        format!("SHOW OBJECTS IN SCHEMA {}.{}", naming::quote_ident(catalog), naming::quote_ident(schema))
    }

    fn sql_list_columns(&self, catalog: &str, schema: &str, table: &str) -> String {
        format!("SHOW COLUMNS IN TABLE {}.{}.{}", naming::quote_ident(catalog), naming::quote_ident(schema), naming::quote_ident(table))
    }
}
