//! Google BigQuery support.
//!
//! - `BigQueryDialect` — SQL dialect for BigQuery-specific syntax and type names,
//!   used by `BigQueryReader`.
//! - `BigQueryReader` — native reader via Application Default Credentials,
//!   behind the `bigquery` feature flag.

use crate::{DataFrame, GgsqlError, Result};
use arrow::array::*;
use chrono::{Datelike, Timelike};
use gcloud_bigquery::client::{Client, ClientConfig};
use gcloud_bigquery::http::dataset::DatasetReference;
use gcloud_bigquery::http::job::get_query_results::GetQueryResultsRequest;
use gcloud_bigquery::http::job::query::{QueryRequest, QueryResponse};
use gcloud_bigquery::http::table::{TableFieldSchema, TableFieldType, TableSchema};
use gcloud_bigquery::http::tabledata::list::{Tuple, Value};
use gcloud_bigquery::http::types::ConnectionProperty;
use std::cell::RefCell;
use std::sync::Arc;
use tokio::runtime::Runtime;

const DEFAULT_LOCATION: &str = "US";
const PAGE_SIZE: i64 = 10_000;

// ============================================================================
// Dialect
// ============================================================================

/// BigQuery SQL dialect.
pub struct BigQueryDialect;

impl super::SqlDialect for BigQueryDialect {
    fn number_type_name(&self) -> Option<&str> {
        Some("FLOAT64")
    }

    fn integer_type_name(&self) -> Option<&str> {
        Some("INT64")
    }

    fn datetime_type_name(&self) -> Option<&str> {
        Some("DATETIME")
    }

    fn string_type_name(&self) -> Option<&str> {
        Some("STRING")
    }

    fn sql_greatest(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("GREATEST({})", exprs.join(", "))
    }

    fn sql_least(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("LEAST({})", exprs.join(", "))
    }

    fn sql_generate_series(&self, n: usize) -> String {
        format!(
            "`__ggsql_seq__`(n) AS (SELECT CAST(n AS FLOAT64) AS n FROM UNNEST(GENERATE_ARRAY(0, {})) AS n)",
            n - 1
        )
    }

    fn sql_quantile_inline(&self, column: &str, fraction: f64) -> Option<String> {
        // APPROX_QUANTILES splits the column into 100 buckets; OFFSET(N) picks
        // the Nth percentile boundary. Unlike PERCENTILE_CONT (a window-only
        // function), it is a genuine aggregate, so it composes directly inside a
        // GROUP BY query — no correlated subquery, which BigQuery rejects.
        let offset = (fraction * 100.0).round() as i64;
        Some(format!(
            "APPROX_QUANTILES({}, 100)[OFFSET({})]",
            quote_ident(column),
            offset
        ))
    }

    fn sql_percentile(
        &self,
        column: &str,
        fraction: f64,
        _from: &str,
        _groups: &[String],
    ) -> String {
        // BigQuery has no exact percentile *aggregate* and rejects the correlated
        // scalar-subquery form of the portable default. Every caller (boxplot,
        // density) embeds the result in the SELECT list of a GROUP BY query,
        // where the APPROX_QUANTILES aggregate computes per group — so `from`
        // and `groups` are not needed here.
        self.sql_quantile_inline(column, fraction)
            .expect("BigQuery sql_quantile_inline always returns Some")
    }

    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        format!("DATE_ADD(DATE '1970-01-01', INTERVAL {} DAY)", days_since_epoch)
    }

    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        format!("DATETIME(TIMESTAMP_MICROS({}))", microseconds_since_epoch)
    }

    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let microseconds = nanoseconds_since_midnight / 1_000;
        format!("TIME(TIMESTAMP_MICROS({}))", microseconds)
    }

    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let body = if column_aliases.is_empty() {
            body_sql.to_string()
        } else {
            let cols = column_aliases
                .iter()
                .map(|c| quote_ident(c))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "WITH __ggsql_aliased__({}) AS ({}) SELECT * FROM __ggsql_aliased__",
                cols, body_sql
            )
        };
        vec![format!(
            "CREATE OR REPLACE TEMP TABLE {} AS {}",
            quote_ident(name),
            body
        )]
    }
}

impl BigQueryDialect {
    pub(crate) fn sql_list_catalogs() -> String {
        "SELECT @@project_id AS catalog_name".to_string()
    }

    pub(crate) fn sql_list_schemas(catalog: &str) -> String {
        format!(
            "SELECT schema_name FROM {}.INFORMATION_SCHEMA.SCHEMATA ORDER BY schema_name",
            quote_path(&[catalog])
        )
    }

    pub(crate) fn sql_list_tables(catalog: &str, schema: &str) -> String {
        format!(
            "SELECT table_name, table_type FROM {}.INFORMATION_SCHEMA.TABLES ORDER BY table_name",
            quote_path(&[catalog, schema])
        )
    }

    pub(crate) fn sql_list_columns(catalog: &str, schema: &str, table: &str) -> String {
        format!(
            "SELECT column_name, data_type FROM {}.INFORMATION_SCHEMA.COLUMNS \
             WHERE table_name = '{}' ORDER BY ordinal_position",
            quote_path(&[catalog, schema]),
            table.replace('\'', "''")
        )
    }
}

fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn quote_path(parts: &[&str]) -> String {
    quote_ident(&parts.join("."))
}

// ============================================================================
// Reader
// ============================================================================

/// Native BigQuery reader using Application Default Credentials.
pub struct BigQueryReader {
    runtime: Runtime,
    client: Client,
    project_id: String,
    default_dataset: Option<String>,
    location: String,
    session_id: RefCell<Option<String>>,
}

// The reader owns a single BigQuery session id and mutates it through
// interior mutability; use it from one thread at a time like the ODBC reader.
unsafe impl Send for BigQueryReader {}

impl BigQueryReader {
    pub fn from_connection_string(uri: &str) -> Result<Self> {
        let info = match super::connection::parse_connection_string(uri)? {
            super::connection::ConnectionInfo::BigQuery(info) => info,
            _ => {
                return Err(GgsqlError::ReaderError(format!(
                    "Connection string '{}' is not supported by BigQueryReader",
                    uri
                )))
            }
        };

        let runtime = Runtime::new().map_err(|e| {
            GgsqlError::ReaderError(format!("Failed to create BigQuery runtime: {}", e))
        })?;

        let (client, detected_project) = runtime.block_on(async {
            let (config, detected_project) =
                ClientConfig::new_with_auth().await.map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to initialize BigQuery authentication: {}",
                        e
                    ))
                })?;
            let client = Client::new(config).await.map_err(|e| {
                GgsqlError::ReaderError(format!("Failed to create BigQuery client: {}", e))
            })?;
            Ok::<_, GgsqlError>((client, detected_project))
        })?;

        let project_id = match info.project_id {
            Some(id) => id,
            None => detected_project.ok_or_else(|| {
                GgsqlError::ReaderError(
                    "No BigQuery project specified in the URI and ADC did not provide one. \
                     Set GOOGLE_CLOUD_PROJECT or run `gcloud config set project <id>`."
                        .to_string(),
                )
            })?,
        };

        Ok(Self {
            runtime,
            client,
            project_id,
            default_dataset: info.default_dataset,
            location: info.location.unwrap_or_else(|| DEFAULT_LOCATION.to_string()),
            session_id: RefCell::new(None),
        })
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    async fn run_query(&self, sql: &str) -> Result<QueryResponse> {
        let response = self
            .client
            .job()
            .query(&self.project_id, &self.query_request(sql))
            .await
            .map_err(|e| GgsqlError::ReaderError(format!("BigQuery query failed: {}", e)))?;

        if let Some(errors) = &response.errors {
            if !errors.is_empty() {
                return Err(GgsqlError::ReaderError(format!(
                    "BigQuery query failed: {:?}",
                    errors
                )));
            }
        }

        if self.session_id.borrow().is_none() {
            if let Some(session_id) = response
                .session_info
                .as_ref()
                .and_then(|info| info.session_id.clone())
            {
                *self.session_id.borrow_mut() = Some(session_id);
            }
        }

        Ok(response)
    }

    async fn wait_for_results(
        &self,
        response: QueryResponse,
    ) -> Result<(Option<TableSchema>, Vec<Tuple>)> {
        let mut schema = response.schema;
        let mut rows = response.rows.unwrap_or_default();
        let mut page_token = response.page_token;
        let job_id = response.job_reference.job_id;
        let location = response
            .job_reference
            .location
            .or_else(|| Some(self.location.clone()));

        let mut job_complete = response.job_complete;
        loop {
            if job_complete && page_token.is_none() {
                break;
            }

            let next = self
                .client
                .job()
                .get_query_results(
                    &self.project_id,
                    &job_id,
                    &GetQueryResultsRequest {
                        page_token,
                        max_results: Some(PAGE_SIZE),
                        timeout_ms: Some(200_000),
                        location: location.clone(),
                        ..GetQueryResultsRequest::default()
                    },
                )
                .await
                .map_err(|e| {
                    GgsqlError::ReaderError(format!("BigQuery result fetch failed: {}", e))
                })?;

            if let Some(errors) = &next.errors {
                if !errors.is_empty() {
                    return Err(GgsqlError::ReaderError(format!(
                        "BigQuery result fetch failed: {:?}",
                        errors
                    )));
                }
            }

            if schema.is_none() {
                schema = next.schema;
            }
            rows.extend(next.rows.unwrap_or_default());
            page_token = next.page_token;
            job_complete = next.job_complete;
        }

        Ok((schema, rows))
    }

    fn query_request(&self, sql: &str) -> QueryRequest {
        let session_id = self.session_id.borrow().clone();
        let connection_properties = session_id
            .map(|value| {
                vec![ConnectionProperty {
                    key: "session_id".to_string(),
                    value,
                }]
            })
            .unwrap_or_default();

        QueryRequest {
            query: sql.to_string(),
            max_results: Some(PAGE_SIZE),
            timeout_ms: Some(200_000),
            use_legacy_sql: false,
            location: self.location.clone(),
            create_session: Some(self.session_id.borrow().is_none()),
            default_dataset: self.default_dataset.as_ref().map(|dataset_id| {
                DatasetReference {
                    project_id: self.project_id.clone(),
                    dataset_id: dataset_id.clone(),
                }
            }),
            connection_properties,
            ..QueryRequest::default()
        }
    }
}

impl super::Reader for BigQueryReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        let sql = super::data::rewrite_namespaced_sql(sql)?;
        let sql = sql.replace('"', "`");

        let response = self.runtime.block_on(self.run_query(&sql))?;
        let (schema, rows) = self.runtime.block_on(self.wait_for_results(response))?;

        let Some(schema) = schema else {
            return Ok(DataFrame::empty());
        };

        rows_to_dataframe(&schema, &rows)
    }

    fn execute(&self, query: &str) -> Result<super::Spec> {
        super::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn super::SqlDialect {
        &BigQueryDialect
    }

    fn list_catalogs(&self) -> Result<Vec<String>> {
        let df = self.execute_sql(&BigQueryDialect::sql_list_catalogs())?;
        let col = df.column("catalog_name")?;
        Ok((0..df.height())
            .filter(|&i| !col.is_null(i))
            .map(|i| crate::array_util::value_to_string(col, i))
            .collect())
    }

    fn list_schemas(&self, catalog: &str) -> Result<Vec<String>> {
        let df = self.execute_sql(&BigQueryDialect::sql_list_schemas(catalog))?;
        let col = df.column("schema_name")?;
        Ok((0..df.height())
            .filter(|&i| !col.is_null(i))
            .map(|i| crate::array_util::value_to_string(col, i))
            .collect())
    }

    fn list_tables(&self, catalog: &str, schema: &str) -> Result<Vec<super::TableInfo>> {
        let df = self.execute_sql(&BigQueryDialect::sql_list_tables(catalog, schema))?;
        let name_col = df.column("table_name")?;
        let type_col = df.column("table_type")?;
        Ok((0..df.height())
            .filter(|&i| !name_col.is_null(i))
            .map(|i| super::TableInfo {
                name: crate::array_util::value_to_string(name_col, i),
                table_type: crate::array_util::value_to_string(type_col, i),
            })
            .collect())
    }

    fn list_columns(&self, catalog: &str, schema: &str, table: &str) -> Result<Vec<super::ColumnInfo>> {
        let df = self.execute_sql(&BigQueryDialect::sql_list_columns(catalog, schema, table))?;
        let name_col = df.column("column_name")?;
        let type_col = df.column("data_type")?;
        Ok((0..df.height())
            .filter(|&i| !name_col.is_null(i))
            .map(|i| super::ColumnInfo {
                name: crate::array_util::value_to_string(name_col, i),
                data_type: crate::array_util::value_to_string(type_col, i),
            })
            .collect())
    }

    fn register(&self, name: &str, _df: DataFrame, _replace: bool) -> Result<()> {
        // BigQuery is a remote service; there is no in-process table store to
        // register Arrow data into. This is not called by the execution pipeline:
        // CTE materialization uses create_or_replace_temp_table_sql + execute_sql
        // (BigQuery-native DDL) instead of register().
        Err(GgsqlError::ReaderError(format!(
            "BigQueryReader does not support in-process DataFrame registration ('{}')",
            name
        )))
    }
}

// ============================================================================
// Row conversion helpers
// ============================================================================

fn rows_to_dataframe(schema: &TableSchema, rows: &[Tuple]) -> Result<DataFrame> {
    let arrays = schema
        .fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let values = rows
                .iter()
                .map(|row| row.f.get(idx).map(|cell| &cell.v).unwrap_or(&Value::Null))
                .collect::<Vec<_>>();
            Ok((field.name.clone(), values_to_array(field, &values)?))
        })
        .collect::<Result<Vec<_>>>()?;

    DataFrame::new(arrays)
}

fn values_to_array(field: &TableFieldSchema, values: &[&Value]) -> Result<ArrayRef> {
    if matches!(
        field.mode,
        Some(gcloud_bigquery::http::table::TableFieldMode::Repeated)
    ) {
        return Ok(string_array(values));
    }

    match field.data_type {
        TableFieldType::Integer | TableFieldType::Int64 => Ok(Arc::new(Int64Array::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(|s| s.parse::<i64>().ok()))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Float | TableFieldType::Float64 => Ok(Arc::new(Float64Array::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(|s| s.parse::<f64>().ok()))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Numeric
        | TableFieldType::Decimal
        | TableFieldType::Bignumeric
        | TableFieldType::Bigdecimal => Ok(Arc::new(Float64Array::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(|s| s.parse::<f64>().ok()))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Boolean | TableFieldType::Bool => Ok(Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(|s| s.parse::<bool>().ok()))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Date => Ok(Arc::new(Date32Array::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(parse_date_days))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Timestamp => Ok(Arc::new(TimestampMicrosecondArray::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(parse_timestamp_micros))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Datetime => Ok(Arc::new(TimestampMicrosecondArray::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(parse_datetime_micros))
                .collect::<Vec<_>>(),
        ))),
        TableFieldType::Time => Ok(Arc::new(Time64NanosecondArray::from(
            values
                .iter()
                .map(|v| value_as_str(v).and_then(parse_time_nanos))
                .collect::<Vec<_>>(),
        ))),
        _ => Ok(string_array(values)),
    }
}

fn value_as_str(value: &Value) -> Option<&str> {
    match value {
        Value::String(s) => Some(s),
        _ => None,
    }
}

fn string_array(values: &[&Value]) -> ArrayRef {
    let strings = values
        .iter()
        .map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            other => Some(format!("{:?}", other)),
        })
        .collect::<Vec<_>>();
    Arc::new(StringArray::from(strings)) as ArrayRef
}

fn parse_date_days(value: &str) -> Option<i32> {
    const EPOCH_DAYS_FROM_CE: i32 = 719_163;
    chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .ok()
        .map(|date| date.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
}

fn parse_timestamp_micros(value: &str) -> Option<i64> {
    value
        .parse::<f64>()
        .ok()
        .map(|seconds| (seconds * 1_000_000.0).round() as i64)
}

fn parse_datetime_micros(value: &str) -> Option<i64> {
    chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f"))
        .ok()
        .map(|dt| dt.and_utc().timestamp_micros())
}

fn parse_time_nanos(value: &str) -> Option<i64> {
    chrono::NaiveTime::parse_from_str(value, "%H:%M:%S%.f")
        .ok()
        .map(|time| {
            let seconds = time.num_seconds_from_midnight() as i64;
            seconds * 1_000_000_000 + time.nanosecond() as i64
        })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::{Reader, SqlDialect};
    use arrow::datatypes::DataType;

    #[test]
    fn test_bigquery_temp_table_sql_uses_backticks() {
        let statements = BigQueryDialect.create_or_replace_temp_table_sql(
            "__ggsql_data",
            &["x".to_string()],
            "SELECT 1",
        );
        assert_eq!(
            statements,
            vec![
                "CREATE OR REPLACE TEMP TABLE `__ggsql_data` AS WITH __ggsql_aliased__(`x`) AS (SELECT 1) SELECT * FROM __ggsql_aliased__".to_string()
            ]
        );
    }

    #[test]
    fn test_bigquery_generate_series() {
        assert_eq!(
            BigQueryDialect.sql_generate_series(3),
            "`__ggsql_seq__`(n) AS (SELECT CAST(n AS FLOAT64) AS n FROM UNNEST(GENERATE_ARRAY(0, 2)) AS n)"
        );
    }

    #[test]
    fn test_bigquery_quantile_inline_uses_approx_quantiles() {
        assert_eq!(
            BigQueryDialect.sql_quantile_inline("value", 0.25),
            Some("APPROX_QUANTILES(`value`, 100)[OFFSET(25)]".to_string())
        );
        assert_eq!(
            BigQueryDialect.sql_quantile_inline("value", 0.5),
            Some("APPROX_QUANTILES(`value`, 100)[OFFSET(50)]".to_string())
        );
    }

    #[test]
    fn test_bigquery_percentile_is_not_correlated() {
        // sql_percentile must not emit a correlated subquery — BigQuery rejects
        // those. It delegates to the APPROX_QUANTILES aggregate, valid inside the
        // GROUP BY queries that boxplot and density build.
        let sql =
            BigQueryDialect.sql_percentile("value", 0.75, "SELECT * FROM t", &["g".to_string()]);
        assert_eq!(sql, "APPROX_QUANTILES(`value`, 100)[OFFSET(75)]");
        assert!(!sql.contains("SELECT"), "must not be a subquery");
        assert!(
            !sql.contains("__ggsql_qt__"),
            "must not correlate to an outer table"
        );
    }

    #[test]
    fn test_rows_to_dataframe_basic_types() {
        use gcloud_bigquery::http::tabledata::list::Cell;

        let schema = TableSchema {
            fields: vec![
                TableFieldSchema {
                    name: "i".to_string(),
                    data_type: TableFieldType::Int64,
                    ..TableFieldSchema::default()
                },
                TableFieldSchema {
                    name: "s".to_string(),
                    data_type: TableFieldType::String,
                    ..TableFieldSchema::default()
                },
                TableFieldSchema {
                    name: "b".to_string(),
                    data_type: TableFieldType::Bool,
                    ..TableFieldSchema::default()
                },
            ],
        };
        let rows = vec![Tuple {
            f: vec![
                Cell { v: Value::String("42".to_string()) },
                Cell { v: Value::String("hello".to_string()) },
                Cell { v: Value::String("true".to_string()) },
            ],
        }];

        let df = rows_to_dataframe(&schema, &rows).unwrap();
        assert_eq!(df.shape(), (1, 3));
        assert_eq!(
            df.get_column_names(),
            vec!["i".to_string(), "s".to_string(), "b".to_string()]
        );
        assert_eq!(df.column_dtype("i").unwrap(), DataType::Int64);
        assert_eq!(df.column_dtype("s").unwrap(), DataType::Utf8);
        assert_eq!(df.column_dtype("b").unwrap(), DataType::Boolean);
    }

    // -------------------------------------------------------------------------
    // Integration tests — create real infra, run assertions, clean up.
    //
    // These tests are #[ignore] by default because they touch live BigQuery.
    // They share a single dataset/table (created once, cleaned up at process
    // exit) so individual tests can run in parallel without redundant setup.
    //
    // Run all of them with:
    //
    //   GGSQL_BIGQUERY_TEST_URI=bigquery://your-project \
    //     cargo test --features bigquery -- --ignored bq_integration
    // -------------------------------------------------------------------------

    use std::sync::OnceLock;

    /// RAII fixture: creates a uniquely-named dataset and a few tables, drops
    /// the whole dataset on drop. Stored in a static so all integration tests
    /// share the same infra.
    ///
    /// Tables:
    /// - `readings` — 3 rows of id/label/value/flag, for the listing and DRAW tests.
    /// - `types`    — DATE/TIMESTAMP/DATETIME/TIME/NUMERIC columns, for conversion.
    /// - `wide`     — 25_000 rows, for result pagination (PAGE_SIZE is 10_000).
    struct BqTestFixture {
        project_id: String,
        dataset_id: String,
        table_id: String,
        types_table: String,
        wide_table: String,
    }

    impl BqTestFixture {
        fn setup(base_uri: &str) -> crate::Result<Self> {
            // Both ring and aws-lc-rs are in the dep tree; install ring explicitly
            // so rustls doesn't panic trying to auto-select between them.
            let _ = rustls::crypto::ring::default_provider().install_default();

            let dataset_id = format!("ggsql_test_{}", uuid::Uuid::new_v4().simple());
            let table_id = "readings".to_string();
            let types_table = "types".to_string();
            let wide_table = "wide".to_string();

            let bootstrap = BigQueryReader::from_connection_string(base_uri)?;
            let project_id = bootstrap.project_id().to_string();

            bootstrap.execute_sql(&format!(
                "CREATE SCHEMA `{}.{}`",
                project_id, dataset_id
            ))?;
            bootstrap.execute_sql(&format!(
                "CREATE TABLE `{}.{}.{}` AS
                    SELECT 1 AS id, 'alpha' AS label, 1.5 AS value, TRUE  AS flag UNION ALL
                    SELECT 2,       'beta',            2.5,          FALSE          UNION ALL
                    SELECT 3,       'gamma',           3.5,          TRUE",
                project_id, dataset_id, table_id
            ))?;
            // One populated row and one all-NULL row, so the conversion exercises
            // both the value and the null path for every type.
            bootstrap.execute_sql(&format!(
                "CREATE TABLE `{}.{}.{}` AS
                    SELECT DATE '2026-05-21' AS d, \
                           TIMESTAMP '2026-05-21 12:00:00 UTC' AS ts, \
                           DATETIME '2026-05-21 12:00:00' AS dt, \
                           TIME '12:00:00' AS t, \
                           NUMERIC '123.45' AS n UNION ALL
                    SELECT CAST(NULL AS DATE), CAST(NULL AS TIMESTAMP), \
                           CAST(NULL AS DATETIME), CAST(NULL AS TIME), \
                           CAST(NULL AS NUMERIC)",
                project_id, dataset_id, types_table
            ))?;
            bootstrap.execute_sql(&format!(
                "CREATE TABLE `{}.{}.{}` AS \
                 SELECT n FROM UNNEST(GENERATE_ARRAY(1, 25000)) AS n",
                project_id, dataset_id, wide_table
            ))?;

            Ok(Self {
                project_id,
                dataset_id,
                table_id,
                types_table,
                wide_table,
            })
        }

        /// Build a reader whose default dataset points at this fixture's dataset.
        fn reader(&self) -> crate::Result<BigQueryReader> {
            BigQueryReader::from_connection_string(&format!(
                "bigquery://{}/{}",
                self.project_id, self.dataset_id
            ))
        }
    }

    impl Drop for BqTestFixture {
        fn drop(&mut self) {
            if let Ok(reader) = BigQueryReader::from_connection_string(&format!(
                "bigquery://{}",
                self.project_id
            )) {
                let _ = reader.execute_sql(&format!(
                    "DROP SCHEMA IF EXISTS `{}.{}` CASCADE",
                    self.project_id, self.dataset_id
                ));
            }
        }
    }

    static BQ_FIXTURE: OnceLock<BqTestFixture> = OnceLock::new();

    fn integration_fixture() -> &'static BqTestFixture {
        BQ_FIXTURE.get_or_init(|| {
            let uri = std::env::var("GGSQL_BIGQUERY_TEST_URI")
                .expect("set GGSQL_BIGQUERY_TEST_URI=bigquery://your-project");
            BqTestFixture::setup(&uri).expect("BQ integration fixture setup failed")
        })
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_list_catalogs() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let catalogs = reader.list_catalogs().unwrap();
        assert!(
            catalogs.contains(&f.project_id),
            "expected project '{}' in catalogs, got {:?}",
            f.project_id, catalogs
        );
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_list_schemas() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let schemas = reader.list_schemas(&f.project_id).unwrap();
        assert!(
            schemas.contains(&f.dataset_id),
            "expected dataset '{}' in schemas, got {:?}",
            f.dataset_id, schemas
        );
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_list_tables() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let tables = reader.list_tables(&f.project_id, &f.dataset_id).unwrap();
        let names: Vec<_> = tables.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&f.table_id.as_str()),
            "expected table '{}', got {:?}",
            f.table_id, names
        );
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_list_columns() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let columns = reader
            .list_columns(&f.project_id, &f.dataset_id, &f.table_id)
            .unwrap();
        let names: Vec<_> = columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "label", "value", "flag"]);
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_execute_sql() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let df = reader
            .execute_sql(&format!("SELECT * FROM `{}` ORDER BY id", f.table_id))
            .unwrap();
        assert_eq!(df.shape(), (3, 4));
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_draw_point() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let spec = reader
            .execute(&format!(
                "SELECT id, value FROM `{}` ORDER BY id \
                 VISUALISE id AS x, value AS y DRAW point",
                f.table_id
            ))
            .unwrap();
        assert_eq!(spec.metadata().rows, 3);
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_draw_boxplot() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let spec = reader
            .execute(&format!(
                "SELECT value FROM `{}` VISUALISE value AS y DRAW boxplot",
                f.table_id
            ))
            .unwrap();
        assert!(spec.metadata().rows > 0, "boxplot returned no rows");
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_draw_boxplot_grouped() {
        // Grouped boxplot drives sql_percentile with a real group column,
        // exercising the APPROX_QUANTILES aggregate path per quartile.
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let spec = reader
            .execute(&format!(
                "SELECT flag, value FROM `{}` \
                 VISUALISE value AS y, flag AS fill DRAW boxplot",
                f.table_id
            ))
            .unwrap();
        assert!(spec.metadata().rows > 0, "grouped boxplot returned no rows");
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_draw_histogram() {
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let spec = reader
            .execute(&format!(
                "SELECT value FROM `{}` VISUALISE value AS x DRAW histogram",
                f.table_id
            ))
            .unwrap();
        assert!(spec.metadata().rows > 0, "histogram returned no rows");
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_type_conversion() {
        use arrow::datatypes::TimeUnit;

        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let df = reader
            .execute_sql(&format!("SELECT d, ts, dt, t, n FROM `{}`", f.types_table))
            .unwrap();
        assert_eq!(df.shape(), (2, 5));
        assert_eq!(df.column_dtype("d").unwrap(), DataType::Date32);
        assert_eq!(
            df.column_dtype("ts").unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            df.column_dtype("dt").unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            df.column_dtype("t").unwrap(),
            DataType::Time64(TimeUnit::Nanosecond)
        );
        assert_eq!(df.column_dtype("n").unwrap(), DataType::Float64);
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_pagination() {
        // PAGE_SIZE is 10_000, so a 25_000-row scan must stitch three result
        // pages. A wrong row count means a page was dropped or duplicated.
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let df = reader
            .execute_sql(&format!("SELECT n FROM `{}`", f.wide_table))
            .unwrap();
        assert_eq!(df.shape(), (25_000, 1));
    }

    #[test]
    #[ignore = "touches live BigQuery infra; set GGSQL_BIGQUERY_TEST_URI=bigquery://project and run with --ignored"]
    fn bq_integration_query_error() {
        // A failing query must surface as an Err, not a panic.
        let f = integration_fixture();
        let reader = f.reader().unwrap();
        let result = reader.execute_sql("SELECT * FROM `__ggsql_no_such_table__`");
        assert!(result.is_err(), "expected an error for a missing table");
    }
}
