//! Generic ODBC data source implementation
//!
//! Provides a reader for any ODBC-compatible database (Snowflake, PostgreSQL,
//! SQL Server, etc.) using the `odbc-api` crate.

use crate::reader::Reader;
use crate::{naming, DataFrame, GgsqlError, Result};
use odbc_api::{buffers::TextRowSet, ConnectionOptions, Cursor, Environment};
use polars::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Global ODBC environment (must be a singleton per process).
fn odbc_env() -> &'static Environment {
    static ENV: OnceLock<Environment> = OnceLock::new();
    ENV.get_or_init(|| Environment::new().expect("Failed to create ODBC environment"))
}

/// Detect the backend SQL dialect from an ODBC connection string.
///
/// Returns a dialect matching the detected backend (e.g. Snowflake, SQLite,
/// DuckDB, or ANSI for generic/unknown backends).
fn detect_dialect(conn_str: &str) -> Box<dyn super::SqlDialect> {
    let lower = conn_str.to_lowercase();
    if lower.contains("driver=snowflake") {
        Box::new(super::snowflake::SnowflakeDialect)
    } else if lower.contains("driver=sqlite") || lower.contains("driver={sqlite") {
        #[cfg(feature = "sqlite")]
        {
            Box::new(super::sqlite::SqliteDialect)
        }
        #[cfg(not(feature = "sqlite"))]
        {
            Box::new(super::AnsiDialect)
        }
    } else if lower.contains("driver=duckdb") || lower.contains("driver={duckdb") {
        #[cfg(feature = "duckdb")]
        {
            Box::new(super::duckdb::DuckDbDialect)
        }
        #[cfg(not(feature = "duckdb"))]
        {
            Box::new(super::AnsiDialect)
        }
    } else {
        Box::new(super::AnsiDialect)
    }
}

/// Generic ODBC reader implementing the `Reader` trait.
pub struct OdbcReader {
    connection: odbc_api::Connection<'static>,
    dialect: Box<dyn super::SqlDialect>,
    registered_tables: RefCell<HashSet<String>>,
}

// Safety: odbc_api::Connection is Send when we ensure single-threaded access.
// The Reader trait requires &self (immutable) for execute_sql, and ODBC
// connections are safe to use from one thread at a time.
unsafe impl Send for OdbcReader {}

impl OdbcReader {
    /// Create a new ODBC reader from a `odbc://` connection URI.
    ///
    /// The URI format is `odbc://` followed by the raw ODBC connection string.
    /// For Snowflake with Posit Workbench credentials, the reader will
    /// automatically detect and inject OAuth tokens.
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let conn_str = uri
            .strip_prefix("odbc://")
            .ok_or_else(|| GgsqlError::ReaderError("ODBC URI must start with odbc://".into()))?;

        let mut conn_str = conn_str.to_string();

        // Snowflake ConnectionName resolution from connections.toml
        if is_snowflake(&conn_str) {
            if let Some(resolved) = resolve_connection_name(&conn_str) {
                conn_str = resolved;
            }
        }

        // Snowflake Workbench credential detection
        if is_snowflake(&conn_str) && !has_token(&conn_str) {
            if let Some(token) = detect_workbench_token() {
                conn_str = inject_snowflake_token(&conn_str, &token);
            }
        }

        // Detect backend dialect from connection string
        let dialect = detect_dialect(&conn_str);

        let env = odbc_env();
        let connection = env
            .connect_with_connection_string(&conn_str, ConnectionOptions::default())
            .map_err(|e| GgsqlError::ReaderError(format!("ODBC connection failed: {}", e)))?;

        Ok(Self {
            connection,
            dialect,
            registered_tables: RefCell::new(HashSet::new()),
        })
    }
}

impl Reader for OdbcReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        // Execute the query (3rd arg = query timeout, None = no timeout)
        let cursor = self
            .connection
            .execute(sql, (), None)
            .map_err(|e| GgsqlError::ReaderError(format!("ODBC execute failed: {}", e)))?;

        let Some(cursor) = cursor else {
            // DDL or non-query statement — return empty DataFrame
            return DataFrame::new(Vec::<Column>::new())
                .map_err(|e| GgsqlError::ReaderError(format!("Empty DataFrame error: {}", e)));
        };

        cursor_to_dataframe(cursor)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        super::validate_table_name(name)?;

        if replace {
            let drop_sql = format!("DROP TABLE IF EXISTS {}", naming::quote_ident(name));
            // Ignore errors from DROP — table may not exist
            let _ = self.connection.execute(&drop_sql, (), None);
        }

        // Build CREATE TEMP TABLE with typed columns
        let schema = df.schema();
        let col_defs: Vec<String> = schema
            .iter()
            .map(|(col_name, dtype)| format!("{} {}", naming::quote_ident(col_name), polars_dtype_to_sql(dtype)))
            .collect();
        let create_sql = format!(
            "CREATE TEMPORARY TABLE {} ({})",
            naming::quote_ident(name),
            col_defs.join(", ")
        );
        self.connection
            .execute(&create_sql, (), None)
            .map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create temp table '{}': {}", name, e))
            })?;

        // Insert data using ODBC bulk text inserter
        let num_rows = df.height();
        if num_rows > 0 {
            let num_cols = df.width();
            let placeholders: Vec<&str> = vec!["?"; num_cols];
            let insert_sql = format!(
                "INSERT INTO \"{}\" VALUES ({})",
                name,
                placeholders.join(", ")
            );

            // Convert all columns to string representation for text insertion
            let string_columns: Vec<Vec<Option<String>>> = df
                .get_columns()
                .iter()
                .map(|col| {
                    (0..num_rows)
                        .map(|row| {
                            let val = col.get(row).ok()?;
                            if val == AnyValue::Null {
                                None
                            } else {
                                Some(format!("{}", val))
                            }
                        })
                        .collect()
                })
                .collect();

            // Determine max string length per column for buffer allocation
            let max_str_lens: Vec<usize> = string_columns
                .iter()
                .map(|col| {
                    col.iter()
                        .filter_map(|v| v.as_ref().map(|s| s.len()))
                        .max()
                        .unwrap_or(1)
                        .max(1) // minimum buffer size of 1
                })
                .collect();

            const BATCH_SIZE: usize = 1024;
            let prepared = self.connection.prepare(&insert_sql).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to prepare INSERT for '{}': {}", name, e))
            })?;

            let batch_capacity = num_rows.min(BATCH_SIZE);
            let mut inserter = prepared
                .into_text_inserter(batch_capacity, max_str_lens)
                .map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to create bulk inserter for '{}': {}",
                        name, e
                    ))
                })?;

            let mut rows_in_batch = 0;
            for row_idx in 0..num_rows {
                let row_values: Vec<Option<&[u8]>> = string_columns
                    .iter()
                    .map(|col| col[row_idx].as_ref().map(|s| s.as_bytes()))
                    .collect();

                inserter.append(row_values.into_iter()).map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to append row {} to '{}': {}",
                        row_idx, name, e
                    ))
                })?;
                rows_in_batch += 1;

                if rows_in_batch >= BATCH_SIZE {
                    inserter.execute().map_err(|e| {
                        GgsqlError::ReaderError(format!(
                            "Failed to execute batch insert into '{}': {}",
                            name, e
                        ))
                    })?;
                    inserter.clear();
                    rows_in_batch = 0;
                }
            }

            // Execute final partial batch
            if rows_in_batch > 0 {
                inserter.execute().map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to execute final batch insert into '{}': {}",
                        name, e
                    ))
                })?;
            }
        }

        self.registered_tables.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        if !self.registered_tables.borrow().contains(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        let sql = format!("DROP TABLE IF EXISTS {}", naming::quote_ident(name));
        self.connection.execute(&sql, (), None).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        self.registered_tables.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<super::Spec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &*self.dialect
    }
}

/// Map a Polars data type to a SQL column type string.
fn polars_dtype_to_sql(dtype: &DataType) -> &'static str {
    match dtype {
        DataType::Boolean => "BOOLEAN",
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => "BIGINT",
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => "BIGINT",
        DataType::Float32 | DataType::Float64 => "DOUBLE PRECISION",
        DataType::Date => "DATE",
        DataType::Datetime(_, _) => "TIMESTAMP",
        DataType::Time => "TIME",
        _ => "TEXT",
    }
}

/// Convert an ODBC cursor to a Polars DataFrame by fetching all rows as text.
fn cursor_to_dataframe(mut cursor: impl Cursor) -> Result<DataFrame> {
    let col_count = cursor
        .num_result_cols()
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to get column count: {}", e)))?
        as usize;

    if col_count == 0 {
        return DataFrame::new(Vec::<Column>::new())
            .map_err(|e| GgsqlError::ReaderError(e.to_string()));
    }

    // Collect column names
    let mut col_names = Vec::with_capacity(col_count);
    for i in 1..=col_count as u16 {
        let name = cursor.col_name(i).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to get column {} name: {}", i, e))
        })?;
        col_names.push(name);
    }

    // Fetch all rows as text into column-oriented vectors
    let batch_size = 1000;
    let max_str_len = 4096;
    let mut columns: Vec<Vec<Option<String>>> = vec![Vec::new(); col_count];

    let mut row_set = TextRowSet::for_cursor(batch_size, &mut cursor, Some(max_str_len))
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to create row set: {}", e)))?;

    let mut block_cursor = cursor
        .bind_buffer(&mut row_set)
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to bind buffer: {}", e)))?;

    while let Some(batch) = block_cursor
        .fetch()
        .map_err(|e| GgsqlError::ReaderError(format!("Failed to fetch batch: {}", e)))?
    {
        let num_rows = batch.num_rows();
        for (col_idx, column) in columns.iter_mut().enumerate() {
            for row_idx in 0..num_rows {
                let value = batch
                    .at_as_str(col_idx, row_idx)
                    .ok()
                    .flatten()
                    .map(|s| s.to_string());
                column.push(value);
            }
        }
    }

    // Build Polars Series from the text data, attempting type inference
    let series: Vec<Column> = col_names
        .iter()
        .zip(columns.iter())
        .map(|(name, values)| {
            // Try to parse as numeric first, then fall back to string
            let series = if let Some(int_series) = try_parse_integers(name, values) {
                int_series
            } else if let Some(float_series) = try_parse_floats(name, values) {
                float_series
            } else {
                // Fall back to string
                Series::new(
                    name.into(),
                    values
                        .iter()
                        .map(|v| v.as_deref())
                        .collect::<Vec<Option<&str>>>(),
                )
            };
            Column::from(series)
        })
        .collect();

    DataFrame::new(series).map_err(|e| GgsqlError::ReaderError(e.to_string()))
}

/// Try to parse all non-null values as i64.
fn try_parse_integers(name: &str, values: &[Option<String>]) -> Option<Series> {
    let parsed: Vec<Option<i64>> = values
        .iter()
        .map(|v| match v {
            None => Some(None),
            Some(s) => s.parse::<i64>().ok().map(Some),
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Series::new(name.into(), parsed))
}

/// Try to parse all non-null values as f64.
fn try_parse_floats(name: &str, values: &[Option<String>]) -> Option<Series> {
    let parsed: Vec<Option<f64>> = values
        .iter()
        .map(|v| match v {
            None => Some(None),
            Some(s) => s.parse::<f64>().ok().map(Some),
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Series::new(name.into(), parsed))
}

// ============================================================================
// Snowflake Workbench credential detection
// ============================================================================

fn is_snowflake(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("driver=snowflake")
}

fn has_token(conn_str: &str) -> bool {
    conn_str.to_lowercase().contains("token=")
}

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(std::path::PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

/// Find the Snowflake connections.toml file, checking standard locations.
fn find_snowflake_connections_toml() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // 1. $SNOWFLAKE_HOME/connections.toml
    if let Ok(snowflake_home) = std::env::var("SNOWFLAKE_HOME") {
        let p = PathBuf::from(&snowflake_home).join("connections.toml");
        if p.exists() {
            return Some(p);
        }
    }

    // 2. ~/.snowflake/connections.toml
    if let Some(home) = home_dir() {
        let p = home.join(".snowflake").join("connections.toml");
        if p.exists() {
            return Some(p);
        }
    }

    // 3. Platform-specific paths
    if let Some(home) = home_dir() {
        #[cfg(target_os = "macos")]
        {
            let p = home.join("Library/Application Support/snowflake/connections.toml");
            if p.exists() {
                return Some(p);
            }
        }

        #[cfg(target_os = "linux")]
        {
            let xdg = std::env::var("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| home.join(".config"));
            let p = xdg.join("snowflake").join("connections.toml");
            if p.exists() {
                return Some(p);
            }
        }

        #[cfg(target_os = "windows")]
        {
            let p = home.join("AppData/Local/snowflake/connections.toml");
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Resolve a `ConnectionName=<name>` parameter in a Snowflake ODBC connection
/// string by reading the named entry from `~/.snowflake/connections.toml` and
/// building a full ODBC connection string from it.
fn resolve_connection_name(conn_str: &str) -> Option<String> {
    // Extract ConnectionName value (case-insensitive)
    let lower = conn_str.to_lowercase();
    let cn_key = "connectionname=";
    let cn_start = lower.find(cn_key)?;
    let value_start = cn_start + cn_key.len();

    let rest = &conn_str[value_start..];
    let value_end = rest.find(';').unwrap_or(rest.len());
    let connection_name = rest[..value_end].trim();

    if connection_name.is_empty() {
        return None;
    }

    // Read and parse connections.toml
    let toml_path = find_snowflake_connections_toml()?;
    let content = std::fs::read_to_string(&toml_path).ok()?;
    let doc = content.parse::<toml_edit::DocumentMut>().ok()?;

    let entry = doc.get(connection_name)?;
    if !entry.is_table() && !entry.is_inline_table() {
        return None;
    }

    // Build ODBC connection string from TOML entry fields
    let get_str = |key: &str| -> Option<String> { entry.get(key)?.as_str().map(|s| s.to_string()) };

    let account = get_str("account")?;
    let mut parts = vec![
        "Driver=Snowflake".to_string(),
        format!("Server={}.snowflakecomputing.com", account),
    ];

    if let Some(user) = get_str("user") {
        parts.push(format!("UID={}", user));
    }
    if let Some(password) = get_str("password") {
        parts.push(format!("PWD={}", password));
    }
    if let Some(authenticator) = get_str("authenticator") {
        parts.push(format!("Authenticator={}", authenticator));
    }
    if let Some(token) = get_str("token") {
        parts.push(format!("Token={}", token));
    }
    if let Some(warehouse) = get_str("warehouse") {
        parts.push(format!("Warehouse={}", warehouse));
    }
    if let Some(database) = get_str("database") {
        parts.push(format!("Database={}", database));
    }
    if let Some(schema) = get_str("schema") {
        parts.push(format!("Schema={}", schema));
    }
    if let Some(role) = get_str("role") {
        parts.push(format!("Role={}", role));
    }

    Some(parts.join(";"))
}

/// Detect Posit Workbench Snowflake OAuth token.
///
/// Checks `SNOWFLAKE_HOME` for a Workbench-managed `connections.toml` file
/// containing OAuth credentials.
fn detect_workbench_token() -> Option<String> {
    let snowflake_home = std::env::var("SNOWFLAKE_HOME").ok()?;

    // Only use Workbench credentials if the path indicates Workbench management
    if !snowflake_home.contains("posit-workbench") {
        return None;
    }

    let toml_path = std::path::Path::new(&snowflake_home).join("connections.toml");
    let content = std::fs::read_to_string(&toml_path).ok()?;

    let doc = content.parse::<toml_edit::DocumentMut>().ok()?;
    let token = doc.get("workbench")?.get("token")?.as_str()?.to_string();

    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// Inject OAuth token into a Snowflake ODBC connection string.
fn inject_snowflake_token(conn_str: &str, token: &str) -> String {
    // Append authenticator and token parameters
    let mut result = conn_str.trim_end_matches(';').to_string();
    result.push_str(";Authenticator=oauth;Token=");
    result.push_str(token);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_snowflake() {
        assert!(is_snowflake(
            "Driver=Snowflake;Server=foo.snowflakecomputing.com"
        ));
        assert!(!is_snowflake("Driver={PostgreSQL};Server=localhost"));
    }

    #[test]
    fn test_has_token() {
        assert!(has_token("Driver=Snowflake;Token=abc123"));
        assert!(!has_token("Driver=Snowflake;Server=foo"));
    }

    #[test]
    fn test_detect_dialect() {
        // Snowflake uses SHOW commands
        let dialect = detect_dialect("Driver=Snowflake;Server=foo");
        assert!(dialect.sql_list_catalogs().contains("SHOW"));

        // PostgreSQL uses information_schema (ANSI default)
        let dialect = detect_dialect("Driver={PostgreSQL};Server=localhost");
        assert!(dialect.sql_list_catalogs().contains("information_schema"));

        // Generic uses information_schema (ANSI default)
        let dialect = detect_dialect("Driver=SomeOther;Server=localhost");
        assert!(dialect.sql_list_catalogs().contains("information_schema"));
    }

    #[test]
    fn test_inject_snowflake_token() {
        let result = inject_snowflake_token(
            "Driver=Snowflake;Server=foo.snowflakecomputing.com",
            "mytoken",
        );
        assert!(result.contains("Authenticator=oauth"));
        assert!(result.contains("Token=mytoken"));
    }

    #[test]
    fn test_resolve_connection_name_with_toml() {
        use std::io::Write;

        // Create a temp dir with a connections.toml
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("connections.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
default_connection_name = "myconn"

[myconn]
account = "myaccount"
user = "myuser"
password = "mypass"
warehouse = "mywh"
database = "mydb"
schema = "public"
role = "myrole"

[other]
account = "otheraccount"
"#
        )
        .unwrap();

        // Point SNOWFLAKE_HOME at our temp dir
        std::env::set_var("SNOWFLAKE_HOME", dir.path());

        let result = resolve_connection_name("Driver=Snowflake;ConnectionName=myconn");
        assert!(result.is_some());
        let conn = result.unwrap();
        assert!(conn.contains("Driver=Snowflake"));
        assert!(conn.contains("Server=myaccount.snowflakecomputing.com"));
        assert!(conn.contains("UID=myuser"));
        assert!(conn.contains("PWD=mypass"));
        assert!(conn.contains("Warehouse=mywh"));
        assert!(conn.contains("Database=mydb"));
        assert!(conn.contains("Schema=public"));
        assert!(conn.contains("Role=myrole"));

        // Test with a connection that has fewer fields
        let result2 = resolve_connection_name("Driver=Snowflake;ConnectionName=other");
        assert!(result2.is_some());
        let conn2 = result2.unwrap();
        assert!(conn2.contains("Server=otheraccount.snowflakecomputing.com"));
        assert!(!conn2.contains("UID="));

        // Test with non-existent connection name
        let result3 = resolve_connection_name("Driver=Snowflake;ConnectionName=nonexistent");
        assert!(result3.is_none());

        // No ConnectionName param → None
        let result4 = resolve_connection_name("Driver=Snowflake;Server=foo");
        assert!(result4.is_none());

        // Clean up env
        std::env::remove_var("SNOWFLAKE_HOME");
    }

    #[test]
    fn test_polars_dtype_to_sql() {
        assert_eq!(polars_dtype_to_sql(&DataType::Int64), "BIGINT");
        assert_eq!(polars_dtype_to_sql(&DataType::Float64), "DOUBLE PRECISION");
        assert_eq!(polars_dtype_to_sql(&DataType::Boolean), "BOOLEAN");
        assert_eq!(polars_dtype_to_sql(&DataType::Date), "DATE");
        assert_eq!(polars_dtype_to_sql(&DataType::String), "TEXT");
    }
}
