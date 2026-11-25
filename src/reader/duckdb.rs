//! DuckDB data source implementation
//!
//! Provides a reader for DuckDB databases with Arrow DataFrame integration.

use crate::reader::Reader;
use crate::{naming, DataFrame, GgsqlError, Result};
use arrow::compute::concat_batches;
use duckdb::vtab::arrow::{arrow_recordbatch_to_query_params, ArrowVTab};
use duckdb::{params, Connection};

// =============================================================================
// DuckDB builtin data registration
// =============================================================================

/// Register any builtin datasets referenced in the SQL with a DuckDB connection.
///
/// Finds `ggsql:X` patterns in the SQL, writes the embedded parquet data to
/// a temp file, and creates a table named `__ggsql_data_X__` in DuckDB.
#[cfg(feature = "builtin-data")]
fn register_builtin_datasets_duckdb(sql: &str, conn: &Connection) -> Result<()> {
    use std::{env, fs};

    let dataset_names = crate::parser::extract_builtin_dataset_names(sql)?;

    // Load spatial extension before registering datasets that contain
    // geometry columns, so that spatial features are available.
    if dataset_names.iter().any(|n| n == "world") {
        let _ = conn.execute("LOAD spatial", params![]);
    }

    for name in dataset_names {
        let Some(parquet_bytes) = super::data::builtin_parquet_bytes(&name) else {
            continue;
        };

        let table_name = naming::builtin_data_table(&name);

        // Write parquet to temp file for DuckDB's read_parquet
        let mut tmp_path = env::temp_dir();
        tmp_path.push(format!("{}.parquet", name));
        if !tmp_path.exists() {
            fs::write(&tmp_path, parquet_bytes).map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "Failed to write builtin dataset '{}' to {}: {}",
                    name,
                    tmp_path.display(),
                    e
                ))
            })?;
        }

        // WORKAROUND(duckdb-rs#714): Arrow export aborts on GEOMETRY columns.
        // Store geometry as WKB so Arrow transport doesn't crash.
        // https://github.com/duckdb/duckdb-rs/issues/714
        let select_expr = if name == "world" {
            "* REPLACE (ST_AsWKB(geom) AS geom)"
        } else {
            "*"
        };
        let create_sql = format!(
            "CREATE TABLE IF NOT EXISTS {} AS SELECT {} FROM read_parquet('{}')",
            naming::quote_ident(&table_name),
            select_expr,
            tmp_path.display()
        );

        conn.execute(&create_sql, params![]).map_err(|e| {
            GgsqlError::ReaderError(format!(
                "Failed to register builtin dataset '{}': {}",
                name, e
            ))
        })?;
    }
    Ok(())
}

// The DuckDB dialect lives in `super::dialects::duckdb` alongside every other
// backend dialect; re-exported here so `reader::duckdb::DuckDbDialect` keeps
// working for existing callers.
pub use super::dialects::DuckDbDialect;

/// Build the DuckDB config for a connection URI's query string.
///
/// Each `key=value` pair is a DuckDB configuration option, set when the
/// database is opened (`access_mode=read_only`, `threads=4`, ...). Some
/// options, `access_mode` among them, cannot be changed once the database is
/// running, so they have to come in through the URI. DuckDB rejects unknown
/// keys and invalid values.
fn duckdb_config(query: &str) -> Result<duckdb::Config> {
    let mut config = duckdb::Config::default();
    for segment in query.split('&').filter(|s| !s.is_empty()) {
        let (key, value) = segment.split_once('=').ok_or_else(|| {
            GgsqlError::ReaderError(format!(
                "DuckDB option '{}' in connection URI needs a value (key=value)",
                segment
            ))
        })?;
        config = config.with(key, value).map_err(|e| {
            GgsqlError::ReaderError(format!(
                "Invalid DuckDB option '{}={}' in connection URI: {}",
                key, value, e
            ))
        })?;
    }
    Ok(config)
}

/// DuckDB database reader
///
/// Executes SQL queries against DuckDB databases (in-memory or file-based)
/// and returns results as Polars DataFrames.
///
/// # Examples
///
/// ```rust,ignore
/// use ggsql::reader::{Reader, DuckDBReader};
///
/// // In-memory database
/// let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
/// let df = reader.execute_sql("SELECT 1 as x, 2 as y")?;
///
/// // File-based database
/// let reader = DuckDBReader::from_connection_string("duckdb://data.db")?;
/// let df = reader.execute_sql("SELECT * FROM sales")?;
///
/// // Read-only, so other read-only processes can open the file at the same time
/// let reader = DuckDBReader::from_connection_string("duckdb://data.db?access_mode=read_only")?;
/// ```
pub struct DuckDBReader {
    conn: Connection,
    registered_tables: crate::reader::RegisteredTables,
}

impl DuckDBReader {
    /// Create a new DuckDB reader from a connection string
    ///
    /// # Arguments
    ///
    /// * `uri` - Connection string (e.g., "duckdb://memory" or "duckdb://file.db"),
    ///   optionally followed by DuckDB configuration options
    ///   (e.g., "duckdb://file.db?access_mode=read_only")
    ///
    /// # Returns
    ///
    /// A configured DuckDB reader
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The connection string format is invalid
    /// - A configuration option is unknown or invalid
    /// - The database file cannot be opened
    /// - DuckDB initialization fails
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let rest = uri.strip_prefix("duckdb://").ok_or_else(|| {
            GgsqlError::ReaderError(format!(
                "DuckDB URI must start with duckdb://, got '{}'",
                uri
            ))
        })?;
        let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
        let config = duckdb_config(query)?;

        let conn = if path == "memory" {
            Connection::open_in_memory_with_flags(config).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to open in-memory DuckDB: {}", e))
            })?
        } else if path.is_empty() {
            return Err(GgsqlError::ReaderError(
                "DuckDB file path cannot be empty".to_string(),
            ));
        } else {
            Connection::open_with_flags(path, config).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to open DuckDB file '{}': {}", path, e))
            })?
        };

        // https://github.com/duckdb/duckdb/issues/22133
        #[cfg(debug_assertions)]
        conn.execute("SET disabled_optimizers TO 'common_subplan'", params![])
            .map_err(|e| {
                GgsqlError::ReaderError(format!(
                    "Failed to disable common_subplan optimizer: {}",
                    e
                ))
            })?;

        conn.register_table_function::<ArrowVTab>("arrow")
            .map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register arrow function: {}", e))
            })?;

        Ok(Self {
            conn,
            registered_tables: crate::reader::RegisteredTables::new(),
        })
    }

    /// Append `df`'s rows to the temporary table `name` with DuckDB's Arrow
    /// appender.
    fn append_rows(&self, name: &str, df: &DataFrame) -> Result<()> {
        let fail = |e: duckdb::Error| {
            GgsqlError::ReaderError(format!("Failed to append to table '{}': {}", name, e))
        };
        // `register` creates temporary tables, which DuckDB keeps in its
        // `temp` catalog.
        let mut appender = self
            .conn
            .appender_to_catalog_and_db(name, "temp", "main")
            .map_err(fail)?;
        appender
            .append_record_batch(df.inner().clone())
            .map_err(fail)?;
        appender.flush().map_err(fail)
    }

    /// Get a reference to the underlying DuckDB connection
    ///
    /// Useful for executing setup queries (CREATE TABLE, INSERT, etc.)
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Check if a table exists in the database
    fn table_exists(&self, name: &str) -> Result<bool> {
        let sql = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?";
        let count: i64 = self
            .conn
            .query_row(sql, [name], |row| row.get(0))
            .unwrap_or(0);
        Ok(count > 0)
    }
}

use super::validate_table_name;

/// Convert a DataFrame to DuckDB Arrow query parameters.
///
/// Since our DataFrame is already an Arrow RecordBatch, this is a simple passthrough.
fn dataframe_to_arrow_params(df: &DataFrame) -> Result<[usize; 2]> {
    Ok(arrow_recordbatch_to_query_params(df.inner().clone()))
}

impl Reader for DuckDBReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        #[cfg(feature = "builtin-data")]
        register_builtin_datasets_duckdb(sql, &self.conn)?;

        let sql = crate::parser::rewrite_namespaced_sql(sql, self.dialect())?;

        if !super::returns_rows(&sql) {
            self.conn
                .execute(&sql, params![])
                .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute SQL: {}", e)))?;

            return Ok(DataFrame::empty());
        }

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to prepare SQL: {}", e)))?;

        let arrow_result = stmt
            .query_arrow(params![])
            .map_err(|e| GgsqlError::ReaderError(format!("Failed to execute SQL: {}", e)))?;

        let schema = arrow_result.get_schema();
        let batches: Vec<_> = arrow_result.collect();

        if batches.is_empty() {
            return Ok(DataFrame::from_record_batch(
                arrow::record_batch::RecordBatch::new_empty(schema),
            ));
        }

        let combined = concat_batches(&schema, &batches).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to combine result batches: {}", e))
        })?;

        let normalized = super::normalize_result_batch(combined)?;
        Ok(DataFrame::from_record_batch(normalized))
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        validate_table_name(name)?;

        if !replace && self.table_exists(name)? {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' already exists",
                name
            )));
        }

        // Workaround for a duckdb-rs limitation (not a DuckDB limitation).
        //
        // duckdb-rs's `ArrowVTab` writes each RecordBatch into a single DuckDB
        // `DataChunk`, which has a fixed capacity of `STANDARD_VECTOR_SIZE`.
        // That constant is defined in DuckDB's C++ source at
        // `src/include/duckdb/common/constants.hpp` and is currently 2048.
        // When a RecordBatch exceeds this, `FlatVector::copy` panics with
        // `assertion failed: data.len() <= self.capacity()`.
        //
        // So the first chunk of a large DataFrame creates the table (letting
        // DuckDB infer the schema from Arrow), and the rest goes through
        // DuckDB's Arrow appender, which is not subject to the limit. A frame
        // the appender cannot take is inserted chunk by chunk instead.
        const MAX_ARROW_BATCH_ROWS: usize = 2048;
        let total_rows = df.height();
        let create_or_replace = if replace {
            "CREATE OR REPLACE"
        } else {
            "CREATE"
        };

        if total_rows <= MAX_ARROW_BATCH_ROWS {
            // Small DataFrame: register in a single batch
            let params = dataframe_to_arrow_params(&df)?;
            let sql = format!(
                "{} TEMP TABLE {} AS SELECT * FROM arrow(?, ?)",
                create_or_replace,
                naming::quote_ident(name)
            );
            self.conn.execute(&sql, params).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register table '{}': {}", name, e))
            })?;
        } else {
            // Large DataFrame: create table from first chunk, then insert remaining chunks
            let first_chunk = df.slice(0, MAX_ARROW_BATCH_ROWS);
            let params = dataframe_to_arrow_params(&first_chunk)?;
            let create_sql = format!(
                "{} TEMP TABLE {} AS SELECT * FROM arrow(?, ?)",
                create_or_replace,
                naming::quote_ident(name)
            );
            self.conn.execute(&create_sql, params).map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to register table '{}': {}", name, e))
            })?;

            // The rest goes through DuckDB's Arrow appender, which is several
            // times faster than one `INSERT` per chunk on large frames (it
            // chunks to the vector size itself). A type the appender cannot
            // convert falls back to the `INSERT` loop, starting over so a
            // partial append never leaves a half-filled table.
            let appended = self.append_rows(
                name,
                &df.slice(MAX_ARROW_BATCH_ROWS, total_rows - MAX_ARROW_BATCH_ROWS),
            );
            let Err(append_error) = appended else {
                self.registered_tables.note_registered(name);
                return Ok(());
            };
            // Should the fallback fail too, its error says why the appender
            // was not used.
            let fail = |e: duckdb::Error| {
                GgsqlError::ReaderError(format!(
                    "Failed to register table '{}': {} (after {})",
                    name, e, append_error
                ))
            };
            let params = dataframe_to_arrow_params(&first_chunk)?;
            let recreate_sql = format!(
                "CREATE OR REPLACE TEMP TABLE {} AS SELECT * FROM arrow(?, ?)",
                naming::quote_ident(name)
            );
            self.conn.execute(&recreate_sql, params).map_err(fail)?;

            let mut offset = MAX_ARROW_BATCH_ROWS;
            while offset < total_rows {
                let chunk_size = std::cmp::min(MAX_ARROW_BATCH_ROWS, total_rows - offset);
                let chunk = df.slice(offset, chunk_size);
                let params = dataframe_to_arrow_params(&chunk)?;
                let insert_sql = format!(
                    "INSERT INTO {} SELECT * FROM arrow(?, ?)",
                    naming::quote_ident(name)
                );
                self.conn.execute(&insert_sql, params).map_err(fail)?;
                offset += chunk_size;
            }
        }

        // Track the table so we can unregister it later
        self.registered_tables.note_registered(name);
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        // Only allow unregistering tables we created via register()
        if !self.registered_tables.is_registered(name) {
            return Err(GgsqlError::ReaderError(format!(
                "Table '{}' was not registered via this reader",
                name
            )));
        }

        // Drop the temp table
        let sql = format!("DROP TABLE IF EXISTS {}", naming::quote_ident(name));
        self.conn.execute(&sql, []).map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to unregister table '{}': {}", name, e))
        })?;

        // Remove from tracking
        self.registered_tables.note_unregistered(name);

        Ok(())
    }

    fn execute(&self, query: &str) -> Result<super::ResolvedSpec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &DuckDbDialect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_util::{as_i32, as_i64, as_str};
    use crate::df;

    #[test]
    fn test_create_in_memory() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory");
        assert!(reader.is_ok());
    }

    /// A DuckDB file holding one table, `t`, with one row.
    fn create_test_db(dir: &std::path::Path) -> String {
        let path = dir.join("test.duckdb").to_str().unwrap().to_string();
        let conn = Connection::open(&path).unwrap();
        conn.execute("CREATE TABLE t AS SELECT 42 AS x", params![])
            .unwrap();
        path
    }

    #[test]
    fn test_access_mode_read_only_opens_file_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_db(dir.path());

        let reader =
            DuckDBReader::from_connection_string(&format!("duckdb://{path}?access_mode=read_only"))
                .unwrap();

        // Reads work, and the query string is not part of the file path.
        let df = reader.execute_sql("SELECT x FROM t").unwrap();
        assert_eq!(df.height(), 1);
        assert!(!std::path::Path::new(&format!("{path}?access_mode=read_only")).exists());

        // Writes are refused.
        let err = reader
            .connection()
            .execute("CREATE TABLE u (y INTEGER)", params![])
            .unwrap_err()
            .to_string();
        assert!(err.contains("read-only"), "got: {err}");
    }

    #[test]
    fn test_read_only_readers_share_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_db(dir.path());
        let uri = format!("duckdb://{path}?access_mode=read_only");

        let first = DuckDBReader::from_connection_string(&uri).unwrap();
        let second = DuckDBReader::from_connection_string(&uri).unwrap();

        assert_eq!(first.execute_sql("SELECT x FROM t").unwrap().height(), 1);
        assert_eq!(second.execute_sql("SELECT x FROM t").unwrap().height(), 1);
    }

    #[test]
    fn test_unknown_duckdb_option_is_an_error() {
        let err = DuckDBReader::from_connection_string("duckdb://memory?no_such_option=1")
            .err()
            .unwrap()
            .to_string();
        // DuckDB rejects unknown keys when the database opens.
        assert!(err.contains("no_such_option"), "got: {err}");
    }

    #[test]
    fn test_duckdb_option_without_value_is_an_error() {
        let err = DuckDBReader::from_connection_string("duckdb://memory?access_mode")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("needs a value"), "got: {err}");
    }

    #[test]
    fn test_simple_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader.execute_sql("SELECT 1 as x, 2 as y").unwrap();

        assert_eq!(df.shape(), (1, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_table_creation_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create table
        reader
            .connection()
            .execute("CREATE TABLE test(x INT, y INT)", params![])
            .unwrap();

        // Insert data
        reader
            .connection()
            .execute("INSERT INTO test VALUES (1, 2), (3, 4)", params![])
            .unwrap();

        // Query data
        let df = reader.execute_sql("SELECT * FROM test").unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "DuckDB crashes on Windows with invalid SQL"
    )]
    fn test_invalid_sql() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let result = reader.execute_sql("INVALID SQL SYNTAX");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_with_aggregation() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        reader
            .connection()
            .execute("CREATE TABLE sales(region TEXT, revenue REAL)", params![])
            .unwrap();

        reader
            .connection()
            .execute(
                "INSERT INTO sales VALUES ('US', 100), ('US', 200), ('EU', 150)",
                params![],
            )
            .unwrap();

        let df = reader
            .execute_sql("SELECT region, SUM(revenue) as total FROM sales GROUP BY region")
            .unwrap();

        assert_eq!(df.shape(), (2, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["region".to_string(), "total".to_string()]
        );
    }

    #[test]
    fn test_register_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create a DataFrame using the df! macro
        let df = df! {
            "x" => vec![1i32, 2, 3],
            "y" => vec![10i32, 20, 30],
        }
        .unwrap();

        // Register the DataFrame
        reader.register("my_table", df, false).unwrap();

        // Query the registered table
        let result = reader
            .execute_sql("SELECT * FROM my_table ORDER BY x")
            .unwrap();
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(
            result.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_register_large_frame_keeps_every_row_in_order() {
        // Past one Arrow chunk, the rest goes through the appender.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let n = 5000;
        let df = df! {
            "i" => (0..n as i64).collect::<Vec<_>>(),
            "s" => (0..n).map(|i| format!("row {i}")).collect::<Vec<_>>(),
            "f" => (0..n).map(|i| (i % 7 == 0).then_some(i as f64 / 2.0)).collect::<Vec<_>>(),
        }
        .unwrap();
        reader.register("big", df, false).unwrap();

        let out = reader
            .execute_sql("SELECT i, s, f FROM big ORDER BY rowid")
            .unwrap();
        assert_eq!(out.height(), n);
        let i = as_i64(out.column("i").unwrap()).unwrap();
        let s = as_str(out.column("s").unwrap()).unwrap();
        assert!((0..n).all(|r| i.value(r) == r as i64));
        assert_eq!(s.value(4999), "row 4999");
        assert_eq!(out.column("f").unwrap().null_count(), n - n.div_ceil(7));
    }

    #[test]
    fn test_register_duplicate_name_errors() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df1 = df! { "a" => vec![1i32] }.unwrap();
        let df2 = df! { "b" => vec![2i32] }.unwrap();

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
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = df! { "a" => vec![1i32] }.unwrap();

        // Empty name
        let result = reader.register("", df.clone(), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Name with double quote should succeed (quote_ident escapes it)
        let result = reader.register("bad\"name", df.clone(), false);
        assert!(result.is_ok());
        reader.unregister("bad\"name").unwrap();

        // Name with null byte
        let result = reader.register("bad\0name", df.clone(), false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid character"));
    }

    #[test]
    fn test_register_empty_dataframe() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create an empty DataFrame with schema
        let df = df! {
            "x" => Vec::<i32>::new(),
            "y" => Vec::<&str>::new(),
        }
        .unwrap();

        reader.register("empty_table", df, false).unwrap();

        // Query should return empty result with correct schema
        let result = reader.execute_sql("SELECT * FROM empty_table").unwrap();
        assert_eq!(result.shape(), (0, 2));
        assert_eq!(
            result.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_unregister() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = df! { "x" => vec![1i32, 2, 3] }.unwrap();

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
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create a table directly (not via register)
        reader
            .connection()
            .execute("CREATE TABLE user_table (x INT)", params![])
            .unwrap();

        // Should fail - we didn't register this via register()
        let result = reader.unregister("user_table");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("was not registered via this reader"));
    }

    #[test]
    fn test_reregister_after_unregister() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = df! { "x" => vec![1i32, 2, 3] }.unwrap();

        reader.register("data", df.clone(), false).unwrap();
        reader.unregister("data").unwrap();

        // Should be able to register again
        reader.register("data", df, false).unwrap();
        let result = reader.execute_sql("SELECT * FROM data").unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_register_large_dataframe() {
        // duckdb-rs Arrow vtab has a vector capacity of 2048 rows. DataFrames
        // larger than this must be chunked to avoid a panic.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let n = 3000;
        let ids: Vec<i32> = (0..n).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let names: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();

        let df = df! {
            "id" => ids,
            "value" => values,
            "name" => names,
        }
        .unwrap();

        reader.register("large_table", df, false).unwrap();

        // Verify row count
        let result = reader
            .execute_sql("SELECT COUNT(*) as cnt FROM large_table")
            .unwrap();
        let count = as_i64(result.column("cnt").unwrap()).unwrap().value(0);
        assert_eq!(count, n as i64);

        // Verify first and last rows survived chunking intact
        let result = reader
            .execute_sql("SELECT id, name FROM large_table ORDER BY id LIMIT 1")
            .unwrap();
        assert_eq!(as_i32(result.column("id").unwrap()).unwrap().value(0), 0);
        assert_eq!(
            as_str(result.column("name").unwrap()).unwrap().value(0),
            "item_0"
        );

        let result = reader
            .execute_sql("SELECT id, name FROM large_table ORDER BY id DESC LIMIT 1")
            .unwrap();
        assert_eq!(
            as_i32(result.column("id").unwrap()).unwrap().value(0),
            (n - 1)
        );
        assert_eq!(
            as_str(result.column("name").unwrap()).unwrap().value(0),
            format!("item_{}", n - 1)
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_date_vegalite_temporal() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE date_data AS SELECT * FROM (VALUES
                    ('2024-01-01'::DATE, 10),
                    ('2024-01-02'::DATE, 20),
                    ('2024-01-03'::DATE, 30)
                ) AS t(date, value)",
            )
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM date_data VISUALISE DRAW line MAPPING date AS x, value AS y")
            .unwrap();

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"temporal\""),
            "Expected temporal type in Vega-Lite output: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_bar_count_stat() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE bar_data AS SELECT * FROM (VALUES
                    ('A'), ('B'), ('A'), ('C'), ('A'), ('B')
                ) AS t(category)",
            )
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM bar_data VISUALISE DRAW bar MAPPING category AS x")
            .unwrap();

        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.plot().layers.len(), 1);
        assert!(plot.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"bar\""),
            "Expected bar mark in output: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_histogram() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE hist_data AS SELECT generate_series * 2.0 AS value FROM GENERATE_SERIES(0, 49)",
            )
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM hist_data VISUALISE DRAW histogram MAPPING value AS x")
            .unwrap();

        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.plot().layers.len(), 1);
        let layer_df = plot.layer_data(0).unwrap();
        assert!(
            layer_df.height() < 50,
            "Histogram should bin data: got {} rows",
            layer_df.height()
        );

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"bar\""),
            "Histogram should render as bar mark: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_density() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE density_data AS SELECT generate_series * 0.5 AS value FROM GENERATE_SERIES(0, 49)",
            )
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM density_data VISUALISE DRAW density MAPPING value AS x")
            .unwrap();

        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.plot().layers.len(), 1);
        assert!(plot.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(
            json.contains("\"area\""),
            "Density should render as area mark: {}",
            json
        );
    }

    #[cfg(feature = "vegalite")]
    #[test]
    fn test_geom_boxplot() {
        use crate::writer::{VegaLiteWriter, Writer};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql(
                "CREATE TABLE box_data AS
                SELECT 'A' AS grp, generate_series * 1.0 AS value FROM GENERATE_SERIES(1, 10)
                UNION ALL
                SELECT 'B' AS grp, generate_series * 1.0 + 4.0 AS value FROM GENERATE_SERIES(1, 10)",
            )
            .unwrap();

        let spec = reader
            .execute("SELECT * FROM box_data VISUALISE DRAW boxplot MAPPING grp AS x, value AS y")
            .unwrap();

        assert!(spec.as_plot().unwrap().layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let json = writer.render(&spec).unwrap();
        assert!(!json.is_empty(), "Boxplot should render successfully");
    }

    #[cfg(feature = "spatial")]
    #[test]
    fn test_select_wkb_parquet_column() {
        use std::{env, fs};
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader.execute_sql("INSTALL spatial").unwrap();
        reader.execute_sql("LOAD spatial").unwrap();

        let mut path = env::temp_dir();
        path.push("ggsql_test_wkb.parquet");
        reader
            .execute_sql(&format!(
                "COPY (SELECT ST_AsWKB(ST_GeomFromText('POINT(1 2)')) AS geom, 'a' AS name) \
                 TO '{}' (FORMAT PARQUET)",
                path.display()
            ))
            .unwrap();

        let df = reader
            .execute_sql(&format!("SELECT * FROM read_parquet('{}')", path.display()))
            .unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 2);
        fs::remove_file(&path).ok();
    }

    #[cfg(all(feature = "spatial", feature = "builtin-data"))]
    #[test]
    fn test_select_geometry_from_builtin_world() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let df = reader
            .execute_sql("SELECT geom FROM ggsql:world LIMIT 5")
            .unwrap();
        assert_eq!(df.height(), 5);
        assert_eq!(df.width(), 1);
    }
}
