//! Polars SQL context data source implementation
//!
//! Provides a reader that uses Polars' built-in SQL context for querying DataFrames.

use crate::reader::Reader;
use crate::{DataFrame, GgsqlError, Result};
use polars::prelude::*;
use polars::sql::SQLContext;
use std::cell::RefCell;
use std::collections::HashSet;

/// Polars SQL context reader
///
/// Executes SQL queries against registered Polars DataFrames using Polars' built-in
/// SQL context. This is a pure in-memory reader with no external database connection.
///
/// # Examples
///
/// ```rust,ignore
/// use ggsql::reader::{Reader, PolarsReader};
/// use polars::prelude::*;
///
/// // Create an in-memory reader
/// let mut reader = PolarsReader::from_connection_string("polars://memory")?;
///
/// // Register a DataFrame
/// let df = df! {
///     "x" => [1, 2, 3],
///     "y" => [10, 20, 30],
/// }?;
/// reader.register("data", df, false)?;
///
/// // Query it with SQL
/// let result = reader.execute_sql("SELECT * FROM data WHERE x > 1")?;
/// ```
pub struct PolarsReader {
    ctx: RefCell<SQLContext>,
    registered_tables: RefCell<HashSet<String>>,
}

impl PolarsReader {
    /// Create a new Polars reader from a connection string
    ///
    /// # Arguments
    ///
    /// * `uri` - Connection string (e.g., "polars://memory" or "polars://")
    ///
    /// # Returns
    ///
    /// A configured Polars reader with an empty SQL context
    ///
    /// # Errors
    ///
    /// Returns an error if the connection string format is invalid
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_info = super::connection::parse_connection_string(uri)?;

        match conn_info {
            super::connection::ConnectionInfo::PolarsMemory => Ok(Self {
                ctx: RefCell::new(SQLContext::new()),
                registered_tables: RefCell::new(HashSet::new()),
            }),
            _ => Err(GgsqlError::ReaderError(format!(
                "Connection string '{}' is not supported by PolarsReader",
                uri
            ))),
        }
    }

    /// Create a new Polars reader with default settings
    ///
    /// Equivalent to `from_connection_string("polars://memory")`
    pub fn new() -> Self {
        Self {
            ctx: RefCell::new(SQLContext::new()),
            registered_tables: RefCell::new(HashSet::new()),
        }
    }

    /// Check if a table is registered
    fn table_exists(&self, name: &str) -> bool {
        self.registered_tables.borrow().contains(name)
    }

    /// List registered table names
    ///
    /// When `internal` is false, filters out internal tables (prefixed with `__ggsql_`).
    pub fn list_tables(&self, internal: bool) -> Vec<String> {
        self.registered_tables
            .borrow()
            .iter()
            .filter(|name| internal || !name.starts_with("__ggsql_"))
            .cloned()
            .collect()
    }
}

impl Default for PolarsReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a table name
fn validate_table_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(GgsqlError::ReaderError("Table name cannot be empty".into()));
    }

    // Reject characters that could break identifiers or cause issues
    let forbidden = ['"', '\0', '\n', '\r'];
    for ch in forbidden {
        if name.contains(ch) {
            return Err(GgsqlError::ReaderError(format!(
                "Table name '{}' contains invalid character '{}'",
                name,
                ch.escape_default()
            )));
        }
    }

    // Reasonable length limit
    if name.len() > 128 {
        return Err(GgsqlError::ReaderError(format!(
            "Table name '{}' exceeds maximum length of 128 characters",
            name
        )));
    }

    Ok(())
}

impl Reader for PolarsReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Check if this is a DDL statement - Polars SQL context doesn't support DDL
        let trimmed = sql.trim().to_uppercase();
        let is_ddl = trimmed.starts_with("CREATE ")
            || trimmed.starts_with("DROP ")
            || trimmed.starts_with("INSERT ")
            || trimmed.starts_with("UPDATE ")
            || trimmed.starts_with("DELETE ")
            || trimmed.starts_with("ALTER ");

        if is_ddl {
            return Err(GgsqlError::ReaderError(
                format!("Polars SQL context does not support DDL statements. Use register() to add tables. {}", sql)
            ));
        }

        // Handle ggsql:name namespaced identifiers (builtin datasets)
        #[cfg(feature = "builtin-data")]
        {
            let dataset_names = super::data::extract_builtin_dataset_names(sql)?;
            for name in &dataset_names {
                let table_name = crate::naming::builtin_data_table(name);
                if !self.table_exists(&table_name) {
                    let df = super::data::load_builtin_dataframe(name)?;
                    self.register(&table_name, df, true)?;
                }
            }
        }

        // Rewrite ggsql:name → __ggsql_data_name__ in SQL
        let sql = super::data::rewrite_namespaced_sql(sql)?;

        // Execute the query - this returns a LazyFrame
        let lazy_frame = self.ctx.borrow_mut().execute(&sql).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to execute SQL `{}`: {}", sql, e))
        })?;

        // Collect the LazyFrame into a DataFrame
        let df = lazy_frame.collect().map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to collect query result: {}", e))
        })?;

        Ok(df)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        // Validate table name
        validate_table_name(name)?;

        // Handle existing table
        if self.table_exists(name) {
            if replace {
                // Unregister existing table first
                self.ctx.borrow_mut().unregister(name);
                self.registered_tables.borrow_mut().remove(name);
            } else {
                return Err(GgsqlError::ReaderError(format!(
                    "Table '{}' already exists",
                    name
                )));
            }
        }

        // Register the DataFrame with the SQL context
        // Polars SQLContext takes a LazyFrame
        self.ctx.borrow_mut().register(name, df.lazy());

        // Track the table so we can unregister it later
        self.registered_tables.borrow_mut().insert(name.to_string());

        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        // Only allow unregistering tables we created via register()
        if !self.registered_tables.borrow().contains(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        // Unregister from the SQL context
        self.ctx.borrow_mut().unregister(name);

        // Remove from tracking
        self.registered_tables.borrow_mut().remove(name);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_reader() {
        let reader = PolarsReader::from_connection_string("polars://memory");
        assert!(reader.is_ok());
    }

    #[test]
    fn test_create_reader_default() {
        let _reader = PolarsReader::new();
    }

    #[test]
    fn test_register_and_query() {
        let reader = PolarsReader::new();

        let df = df! {
            "x" => [1i32, 2, 3],
            "y" => [10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_table", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT * FROM my_table ORDER BY x")
            .unwrap();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.get_column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_register_and_filter() {
        let reader = PolarsReader::new();

        let df = df! {
            "x" => [1i32, 2, 3, 4, 5],
            "y" => [10i32, 20, 30, 40, 50],
        }
        .unwrap();

        reader.register("data", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT * FROM data WHERE x > 2")
            .unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_register_duplicate_name_errors() {
        let reader = PolarsReader::new();

        let df1 = df! { "a" => [1i32] }.unwrap();
        let df2 = df! { "b" => [2i32] }.unwrap();

        // First registration should succeed
        reader.register("dup_table", df1, false).unwrap();

        // Second registration with same name should fail (when replace=false)
        let result = reader.register("dup_table", df2, false);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn test_register_invalid_table_names() {
        let reader = PolarsReader::new();
        let df = df! { "a" => [1i32] }.unwrap();

        // Empty name
        let result = reader.register("", df.clone(), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Name with double quote
        let result = reader.register("bad\"name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        // Name with null byte
        let result = reader.register("bad\0name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));

        // Name too long
        let long_name = "a".repeat(200);
        let result = reader.register(&long_name, df, false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum length"));
    }

    #[test]
    fn test_unregister() {
        let reader = PolarsReader::new();
        let df = df! { "x" => [1i32, 2, 3] }.unwrap();

        reader.register("test_data", df, false).unwrap();

        // Should be queryable
        let result = reader.execute_sql("SELECT * FROM test_data").unwrap();
        assert_eq!(result.height(), 3);

        // Unregister
        reader.unregister("test_data").unwrap();

        // Should no longer exist
        let result = reader.execute_sql("SELECT * FROM test_data");
        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_not_registered() {
        let reader = PolarsReader::new();

        // Should fail - we didn't register anything
        let result = reader.unregister("nonexistent");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("was not registered via this reader"));
    }

    #[test]
    fn test_reregister_after_unregister() {
        let reader = PolarsReader::new();
        let df = df! { "x" => [1i32, 2, 3] }.unwrap();

        reader.register("data", df.clone(), false).unwrap();
        reader.unregister("data").unwrap();

        // Should be able to register again
        reader.register("data", df, false).unwrap();
        let result = reader.execute_sql("SELECT * FROM data").unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_invalid_sql() {
        let reader = PolarsReader::new();
        let result = reader.execute_sql("INVALID SQL SYNTAX");
        assert!(result.is_err());
    }

    #[test]
    fn test_ddl_not_supported() {
        let reader = PolarsReader::new();

        // CREATE should fail
        let result = reader.execute_sql("CREATE TABLE test (x INT)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("DDL"));

        // DROP should fail
        let result = reader.execute_sql("DROP TABLE test");
        assert!(result.is_err());

        // INSERT should fail
        let result = reader.execute_sql("INSERT INTO test VALUES (1)");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_with_aggregation() {
        let reader = PolarsReader::new();

        let df = df! {
            "region" => ["US", "US", "EU"],
            "revenue" => [100.0f64, 200.0, 150.0],
        }
        .unwrap();

        reader.register("sales", df, false).unwrap();

        let result = reader
            .execute_sql("SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
            .unwrap();

        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result.get_column_names(), vec!["region", "total"]);
    }

    #[test]
    fn test_multiple_tables() {
        let reader = PolarsReader::new();

        let sales = df! {
            "id" => [1i32, 2, 3],
            "amount" => [100i32, 200, 300],
            "product_id" => [1i32, 1, 2],
        }
        .unwrap();

        let products = df! {
            "id" => [1i32, 2],
            "name" => ["Widget", "Gadget"],
        }
        .unwrap();

        reader.register("sales", sales, false).unwrap();
        reader.register("products", products, false).unwrap();

        let result = reader
            .execute_sql(
                "SELECT s.id, s.amount, p.name
                 FROM sales s
                 JOIN products p ON s.product_id = p.id",
            )
            .unwrap();

        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_namespaced_sql_with_preregistered_data() {
        use crate::naming;

        let reader = PolarsReader::new();

        let df = df! {
            "x" => [1i32, 2, 3],
            "y" => [10i32, 20, 30],
        }
        .unwrap();

        // Register under the internal table name that ggsql:penguins rewrites to
        let table_name = naming::builtin_data_table("penguins");
        reader.register(&table_name, df, false).unwrap();

        // ggsql:penguins should be rewritten to __ggsql_data_penguins__ and resolve
        let result = reader.execute_sql("SELECT * FROM ggsql:penguins").unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_namespaced_sql_without_registration_errors() {
        let reader = PolarsReader::new();

        // Without builtin-data feature and without pre-registration, should error
        // (when builtin-data is enabled, this test still passes because
        // the dataset gets auto-loaded)
        let result = reader.execute_sql("SELECT * FROM ggsql:unknown_dataset");
        // Either errors from "not pre-loaded" or from SQL execution failing
        assert!(result.is_err());
    }
}

#[cfg(feature = "builtin-data")]
#[cfg(test)]
mod builtin_data_tests {
    use super::*;

    #[test]
    fn test_builtin_penguins_auto_loads() {
        let reader = PolarsReader::new();

        // ggsql:penguins should auto-load from embedded parquet
        let result = reader
            .execute_sql("SELECT * FROM ggsql:penguins LIMIT 5")
            .unwrap();
        assert_eq!(result.height(), 5);
        assert!(result.width() > 0);
    }

    #[test]
    fn test_builtin_airquality_auto_loads() {
        let reader = PolarsReader::new();

        let result = reader
            .execute_sql("SELECT * FROM ggsql:airquality LIMIT 5")
            .unwrap();
        assert_eq!(result.height(), 5);
        assert!(result.width() > 0);
    }
}
