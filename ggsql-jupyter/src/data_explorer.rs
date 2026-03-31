//! Data explorer backend for the Positron data viewer.
//!
//! Implements the `positron.dataExplorer` comm protocol, providing SQL-backed
//! paginated data access. No full table load — each `get_data_values` request
//! issues a `SELECT ... LIMIT/OFFSET` query.

use ggsql::reader::Reader;
use serde_json::{json, Value};

/// Result of handling an RPC call.
pub struct RpcResponse {
    /// The JSON-RPC result to send as the reply.
    pub result: Value,
    /// An optional event to send on iopub (e.g. `return_column_profiles`).
    pub event: Option<RpcEvent>,
}

/// An asynchronous event to send back on the comm after the RPC reply.
pub struct RpcEvent {
    pub method: String,
    pub params: Value,
}

impl RpcResponse {
    /// Create a simple reply with no async event.
    pub fn reply(result: Value) -> Self {
        Self {
            result,
            event: None,
        }
    }
}

/// Cached column metadata for a table.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    /// Backend-specific type name (e.g. "INTEGER", "VARCHAR").
    pub type_name: String,
    /// Positron display type (e.g. "integer", "string").
    pub type_display: String,
}

/// State for one open data explorer comm.
pub struct DataExplorerState {
    /// Fully qualified and quoted table path, e.g. `"memory"."main"."users"`.
    table_path: String,
    /// Display title shown in the data viewer tab.
    title: String,
    /// Cached column schemas.
    columns: Vec<ColumnInfo>,
    /// Cached total row count.
    num_rows: usize,
}

impl DataExplorerState {
    /// Open a data explorer for a table at the given connection path.
    ///
    /// Runs `SELECT COUNT(*)` and a column metadata query to cache schema
    /// information. Does **not** load the full table into memory.
    pub fn open(reader: &dyn Reader, path: &[String]) -> Result<Self, String> {
        if path.len() < 3 {
            return Err(format!(
                "Expected [catalog, schema, table] path, got {} elements",
                path.len()
            ));
        }

        let catalog = &path[0];
        let schema = &path[1];
        let table = &path[2];

        let table_path = format!(
            "\"{}\".\"{}\".\"{}\""  ,
            catalog.replace('"', "\"\""),
            schema.replace('"', "\"\""),
            table.replace('"', "\"\""),
        );

        // Get row count
        let count_sql = format!("SELECT COUNT(*) AS n FROM {}", table_path);
        let count_df = reader
            .execute_sql(&count_sql)
            .map_err(|e| format!("Failed to count rows: {}", e))?;
        let num_rows = count_df
            .column("n")
            .ok()
            .and_then(|col| col.get(0).ok())
            .and_then(|val| {
                // Polars AnyValue — try common integer representations
                let s = format!("{}", val);
                s.parse::<usize>().ok()
            })
            .unwrap_or(0);

        // Get column metadata from information_schema
        let columns_sql = reader.dialect().sql_list_columns(catalog, schema, table);
        let columns_df = reader
            .execute_sql(&columns_sql)
            .map_err(|e| format!("Failed to list columns: {}", e))?;

        let name_col = columns_df
            .column("column_name")
            .map_err(|e| format!("Missing column_name: {}", e))?;
        let type_col = columns_df
            .column("data_type")
            .map_err(|e| format!("Missing data_type: {}", e))?;

        let mut columns = Vec::new();
        for i in 0..columns_df.height() {
            if let (Ok(name_val), Ok(type_val)) = (name_col.get(i), type_col.get(i)) {
                let name = name_val.to_string().trim_matches('"').to_string();
                let type_name = type_val.to_string().trim_matches('"').to_string();
                let type_display = sql_type_to_display(&type_name).to_string();
                columns.push(ColumnInfo {
                    name,
                    type_name,
                    type_display,
                });
            }
        }

        Ok(Self {
            table_path,
            title: table.clone(),
            columns,
            num_rows,
        })
    }

    /// Dispatch a JSON-RPC method call.
    ///
    /// Returns the RPC result and an optional async event to send on iopub
    /// (used by `get_column_profiles` to deliver results asynchronously).
    pub fn handle_rpc(&self, method: &str, params: &Value, reader: &dyn Reader) -> RpcResponse {
        match method {
            "get_state" => RpcResponse::reply(self.get_state()),
            "get_schema" => RpcResponse::reply(self.get_schema(params)),
            "get_data_values" => RpcResponse::reply(self.get_data_values(params, reader)),
            "get_column_profiles" => self.get_column_profiles(params, reader),
            "set_row_filters" => {
                // Stub: accept but ignore filters, return current shape
                RpcResponse::reply(json!({
                    "selected_num_rows": self.num_rows,
                    "had_errors": false
                }))
            }
            "set_sort_columns" | "set_column_filters" | "search_schema" => {
                RpcResponse::reply(json!(null))
            }
            _ => {
                tracing::warn!("Unhandled data explorer method: {}", method);
                RpcResponse::reply(json!(null))
            }
        }
    }

    fn get_state(&self) -> Value {
        let num_columns = self.columns.len();
        json!({
            "display_name": self.title,
            "table_shape": {
                "num_rows": self.num_rows,
                "num_columns": num_columns
            },
            "table_unfiltered_shape": {
                "num_rows": self.num_rows,
                "num_columns": num_columns
            },
            "has_row_labels": false,
            "column_filters": [],
            "row_filters": [],
            "sort_keys": [],
            "supported_features": {
                "search_schema": {
                    "support_status": "unsupported",
                    "supported_types": []
                },
                "set_column_filters": {
                    "support_status": "unsupported",
                    "supported_types": []
                },
                "set_row_filters": {
                    "support_status": "unsupported",
                    "supports_conditions": "unsupported",
                    "supported_types": []
                },
                "get_column_profiles": {
                    "support_status": "supported",
                    "supported_types": [
                        {"profile_type": "null_count", "support_status": "supported"},
                        {"profile_type": "summary_stats", "support_status": "supported"},
                        {"profile_type": "small_histogram", "support_status": "supported"},
                        {"profile_type": "small_frequency_table", "support_status": "supported"}
                    ]
                },
                "set_sort_columns": {
                    "support_status": "unsupported"
                },
                "export_data_selection": {
                    "support_status": "unsupported",
                    "supported_formats": []
                },
                "convert_to_code": {
                    "support_status": "unsupported"
                }
            }
        })
    }

    fn get_schema(&self, params: &Value) -> Value {
        let indices: Vec<usize> = params
            .get("column_indices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let columns: Vec<Value> = indices
            .iter()
            .filter_map(|&idx| {
                self.columns.get(idx).map(|col| {
                    json!({
                        "column_name": col.name,
                        "column_index": idx,
                        "type_name": col.type_name,
                        "type_display": col.type_display
                    })
                })
            })
            .collect();

        json!({ "columns": columns })
    }

    fn get_data_values(&self, params: &Value, reader: &dyn Reader) -> Value {
        let selections = match params.get("columns").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return json!({ "columns": [] }),
        };

        // Determine the row range from the first selection's spec
        let (first_index, last_index) = selections
            .first()
            .and_then(|sel| sel.get("spec"))
            .map(|spec| {
                let first = spec
                    .get("first_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let last = spec
                    .get("last_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                (first, last)
            })
            .unwrap_or((0, 0));

        let limit = last_index.saturating_sub(first_index) + 1;

        // Collect requested column indices
        let col_indices: Vec<usize> = selections
            .iter()
            .filter_map(|sel| sel.get("column_index").and_then(|v| v.as_u64()).map(|n| n as usize))
            .collect();

        // Build column list for SELECT
        let col_names: Vec<String> = col_indices
            .iter()
            .filter_map(|&idx| {
                self.columns.get(idx).map(|col| {
                    format!("\"{}\"", col.name.replace('"', "\"\""))
                })
            })
            .collect();

        if col_names.is_empty() {
            return json!({ "columns": [] });
        }

        let sql = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            col_names.join(", "),
            self.table_path,
            limit,
            first_index,
        );

        let df = match reader.execute_sql(&sql) {
            Ok(df) => df,
            Err(e) => {
                tracing::error!("get_data_values query failed: {}", e);
                let empty: Vec<Vec<String>> = col_indices.iter().map(|_| vec![]).collect();
                return json!({ "columns": empty });
            }
        };

        // Format each column's values as strings.
        // Positron's ColumnValue is `number | string`: numbers are special
        // value codes (0 = NULL, 1 = NA, 2 = NaN), strings are formatted data.
        const SPECIAL_VALUE_NULL: i64 = 0;

        let columns: Vec<Vec<Value>> = (0..df.width())
            .map(|col_idx| {
                let col = df.get_columns()[col_idx].clone();
                (0..df.height())
                    .map(|row_idx| {
                        match col.get(row_idx) {
                            Ok(val) => {
                                if val.is_null() {
                                    json!(SPECIAL_VALUE_NULL)
                                } else {
                                    let s = format!("{}", val);
                                    // Strip surrounding quotes from string values
                                    let s = s.trim_matches('"');
                                    Value::String(s.to_string())
                                }
                            }
                            Err(_) => json!(SPECIAL_VALUE_NULL),
                        }
                    })
                    .collect()
            })
            .collect();

        json!({ "columns": columns })
    }

    /// Handle `get_column_profiles` — returns `{}` as the RPC result and sends
    /// profile data back as an async `return_column_profiles` event.
    fn get_column_profiles(&self, params: &Value, reader: &dyn Reader) -> RpcResponse {
        let callback_id = params
            .get("callback_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let requests = match params.get("profiles").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => {
                return RpcResponse {
                    result: json!({}),
                    event: Some(RpcEvent {
                        method: "return_column_profiles".into(),
                        params: json!({
                            "callback_id": callback_id,
                            "profiles": []
                        }),
                    }),
                };
            }
        };

        let mut profiles = Vec::new();
        for req in requests {
            let col_idx = req
                .get("column_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let specs = req
                .get("profiles")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let profile = self.compute_column_profile(col_idx, &specs, reader);
            profiles.push(profile);
        }

        RpcResponse {
            result: json!({}),
            event: Some(RpcEvent {
                method: "return_column_profiles".into(),
                params: json!({
                    "callback_id": callback_id,
                    "profiles": profiles
                }),
            }),
        }
    }

    /// Compute profile results for a single column.
    fn compute_column_profile(
        &self,
        col_idx: usize,
        specs: &[Value],
        reader: &dyn Reader,
    ) -> Value {
        let col = match self.columns.get(col_idx) {
            Some(c) => c,
            None => return json!({}),
        };

        let mut wants_null_count = false;
        let mut wants_summary = false;
        let mut histogram_params: Option<&Value> = None;
        let mut freq_table_params: Option<&Value> = None;
        for spec in specs {
            match spec
                .get("profile_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
            {
                "null_count" => wants_null_count = true,
                "summary_stats" => wants_summary = true,
                "small_histogram" => histogram_params = spec.get("params"),
                "small_frequency_table" => freq_table_params = spec.get("params"),
                _ => {}
            }
        }

        let dialect = reader.dialect();
        let quoted_col = format!("\"{}\"", col.name.replace('"', "\"\""));
        let display = col.type_display.as_str();

        // Build a single SQL query that computes all needed aggregates.
        // All expressions use ANSI SQL or existing dialect methods.
        let mut select_parts = Vec::new();
        if wants_null_count {
            select_parts.push(format!(
                "SUM(CASE WHEN {} IS NULL THEN 1 ELSE 0 END) AS \"null_count\"",
                quoted_col
            ));
        }
        if wants_summary {
            match display {
                "integer" | "floating" => {
                    let float_type = dialect.number_type_name().unwrap_or("DOUBLE PRECISION");
                    select_parts.push(format!("MIN({}) AS \"min_val\"", quoted_col));
                    select_parts.push(format!("MAX({}) AS \"max_val\"", quoted_col));
                    select_parts.push(format!(
                        "AVG(CAST({} AS {})) AS \"mean_val\"",
                        quoted_col, float_type
                    ));
                    // Stddev: fetch raw aggregates, compute in Rust
                    select_parts.push(format!(
                        "SUM(CAST({c} AS {t}) * CAST({c} AS {t})) AS \"sum_sq\"",
                        c = quoted_col,
                        t = float_type
                    ));
                    select_parts.push(format!(
                        "SUM(CAST({} AS {})) AS \"sum_val\"",
                        quoted_col, float_type
                    ));
                    select_parts.push(format!("COUNT({}) AS \"cnt\"", quoted_col));
                }
                "boolean" => {
                    let true_lit = dialect.sql_boolean_literal(true);
                    let false_lit = dialect.sql_boolean_literal(false);
                    select_parts.push(format!(
                        "SUM(CASE WHEN {} = {} THEN 1 ELSE 0 END) AS \"true_count\"",
                        quoted_col, true_lit
                    ));
                    select_parts.push(format!(
                        "SUM(CASE WHEN {} = {} THEN 1 ELSE 0 END) AS \"false_count\"",
                        quoted_col, false_lit
                    ));
                }
                "string" => {
                    select_parts.push(format!("COUNT(DISTINCT {}) AS \"num_unique\"", quoted_col));
                    select_parts.push(format!(
                        "SUM(CASE WHEN {} = '' THEN 1 ELSE 0 END) AS \"num_empty\"",
                        quoted_col
                    ));
                }
                "date" | "datetime" => {
                    select_parts.push(format!("MIN({}) AS \"min_val\"", quoted_col));
                    select_parts.push(format!("MAX({}) AS \"max_val\"", quoted_col));
                    select_parts.push(format!("COUNT(DISTINCT {}) AS \"num_unique\"", quoted_col));
                }
                _ => {}
            }
        }

        if select_parts.is_empty() {
            return json!({});
        }

        let sql = format!(
            "SELECT {} FROM {}",
            select_parts.join(", "),
            self.table_path
        );

        let df = match reader.execute_sql(&sql) {
            Ok(df) => df,
            Err(e) => {
                tracing::error!("Column profile query failed: {}", e);
                return json!({});
            }
        };

        let get_str = |name: &str| -> Option<String> {
            df.column(name)
                .ok()
                .and_then(|c| c.get(0).ok())
                .and_then(|v| {
                    if v.is_null() {
                        None
                    } else {
                        Some(format!("{}", v).trim_matches('"').to_string())
                    }
                })
        };

        let get_i64 = |name: &str| -> Option<i64> {
            get_str(name).and_then(|s| s.parse::<i64>().ok())
        };

        let get_f64 = |name: &str| -> Option<f64> {
            get_str(name).and_then(|s| s.parse::<f64>().ok())
        };

        let mut result = json!({});

        if wants_null_count {
            if let Some(n) = get_i64("null_count") {
                result["null_count"] = json!(n);
            }
        }

        if wants_summary {
            let stats = match display {
                "integer" | "floating" => {
                    let mut number_stats = json!({});
                    if let Some(v) = get_str("min_val") {
                        number_stats["min_value"] = json!(v);
                    }
                    if let Some(v) = get_str("max_val") {
                        number_stats["max_value"] = json!(v);
                    }
                    if let Some(v) = get_str("mean_val") {
                        number_stats["mean"] = json!(v);
                    }
                    // Compute sample stddev from raw aggregates
                    if let (Some(sum_sq), Some(sum_val), Some(cnt)) =
                        (get_f64("sum_sq"), get_f64("sum_val"), get_i64("cnt"))
                    {
                        if cnt > 1 {
                            let variance =
                                (sum_sq - sum_val * sum_val / cnt as f64) / (cnt - 1) as f64;
                            let stdev = variance.max(0.0).sqrt();
                            number_stats["stdev"] = json!(format!("{}", stdev));
                        }
                    }
                    // Median via dialect's sql_percentile (uses QUANTILE_CONT on
                    // DuckDB, NTILE fallback on other backends)
                    let col_name = col.name.replace('"', "\"\"");
                    let from_query = format!("SELECT * FROM {}", self.table_path);
                    let median_expr =
                        dialect.sql_percentile(&col_name, 0.5, &from_query, &[]);
                    let median_sql = format!("SELECT {} AS \"median_val\"", median_expr);
                    if let Ok(median_df) = reader.execute_sql(&median_sql) {
                        if let Some(v) = median_df
                            .column("median_val")
                            .ok()
                            .and_then(|c| c.get(0).ok())
                            .and_then(|v| {
                                if v.is_null() {
                                    None
                                } else {
                                    Some(format!("{}", v).trim_matches('"').to_string())
                                }
                            })
                        {
                            number_stats["median"] = json!(v);
                        }
                    }
                    json!({
                        "type_display": display,
                        "number_stats": number_stats
                    })
                }
                "boolean" => {
                    json!({
                        "type_display": display,
                        "boolean_stats": {
                            "true_count": get_i64("true_count").unwrap_or(0),
                            "false_count": get_i64("false_count").unwrap_or(0)
                        }
                    })
                }
                "string" => {
                    json!({
                        "type_display": display,
                        "string_stats": {
                            "num_unique": get_i64("num_unique").unwrap_or(0),
                            "num_empty": get_i64("num_empty").unwrap_or(0)
                        }
                    })
                }
                "date" => {
                    let mut date_stats = json!({});
                    if let Some(v) = get_str("min_val") {
                        date_stats["min_date"] = json!(v);
                    }
                    if let Some(v) = get_str("max_val") {
                        date_stats["max_date"] = json!(v);
                    }
                    if let Some(n) = get_i64("num_unique") {
                        date_stats["num_unique"] = json!(n);
                    }
                    json!({
                        "type_display": display,
                        "date_stats": date_stats
                    })
                }
                "datetime" => {
                    let mut datetime_stats = json!({});
                    if let Some(v) = get_str("min_val") {
                        datetime_stats["min_date"] = json!(v);
                    }
                    if let Some(v) = get_str("max_val") {
                        datetime_stats["max_date"] = json!(v);
                    }
                    if let Some(n) = get_i64("num_unique") {
                        datetime_stats["num_unique"] = json!(n);
                    }
                    json!({
                        "type_display": display,
                        "datetime_stats": datetime_stats
                    })
                }
                _ => json!({"type_display": display}),
            };
            result["summary_stats"] = stats;
        }

        // Compute histogram if requested (only for numeric types)
        if let Some(params) = histogram_params {
            if matches!(display, "integer" | "floating") {
                if let Some(hist) = self.compute_histogram(col, params, reader) {
                    result["small_histogram"] = hist;
                }
            }
        }

        // Compute frequency table if requested (for string/boolean types)
        if let Some(params) = freq_table_params {
            if matches!(display, "string" | "boolean") {
                if let Some(ft) = self.compute_frequency_table(col, params, reader) {
                    result["small_frequency_table"] = ft;
                }
            }
        }

        result
    }

    /// Compute a histogram for a numeric column.
    fn compute_histogram(
        &self,
        col: &ColumnInfo,
        params: &Value,
        reader: &dyn Reader,
    ) -> Option<Value> {
        let max_bins = params
            .get("num_bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize;

        if max_bins == 0 {
            return None;
        }

        let dialect = reader.dialect();
        let float_type = dialect.number_type_name().unwrap_or("DOUBLE PRECISION");
        let quoted_col = format!("\"{}\"", col.name.replace('"', "\"\""));
        let is_integer = col.type_display == "integer";

        // Get min, max, count in one query
        let bounds_sql = format!(
            "SELECT \
                MIN(CAST({c} AS {t})) AS \"min_val\", \
                MAX(CAST({c} AS {t})) AS \"max_val\", \
                COUNT({c}) AS \"cnt\" \
             FROM {table} WHERE {c} IS NOT NULL",
            c = quoted_col,
            t = float_type,
            table = self.table_path,
        );

        let bounds_df = reader.execute_sql(&bounds_sql).ok()?;
        let get_f64 = |name: &str| -> Option<f64> {
            bounds_df
                .column(name)
                .ok()
                .and_then(|c| c.get(0).ok())
                .and_then(|v| {
                    if v.is_null() {
                        None
                    } else {
                        format!("{}", v).trim_matches('"').parse::<f64>().ok()
                    }
                })
        };

        let min_val = get_f64("min_val")?;
        let max_val = get_f64("max_val")?;
        let count = get_f64("cnt").unwrap_or(0.0) as usize;

        // Handle edge case: all values identical
        if (max_val - min_val).abs() < f64::EPSILON {
            return Some(json!({
                "bin_edges": [format!("{}", min_val), format!("{}", max_val)],
                "bin_counts": [count as i64],
                "quantiles": []
            }));
        }

        // Determine actual bin count using Sturges' formula, capped at max_bins.
        // For integers, also cap at (max - min + 1) to avoid sub-unit bins.
        let mut num_bins = if count > 1 {
            ((count as f64).log2().ceil() as usize + 1).max(1)
        } else {
            1
        };
        if is_integer {
            let int_range = (max_val - min_val) as usize + 1;
            num_bins = num_bins.min(int_range);
        }
        num_bins = num_bins.min(max_bins).max(1);

        let bin_width = (max_val - min_val) / num_bins as f64;

        // Bin the data using FLOOR. Clamp the last bin to num_bins-1 so
        // max value doesn't create an extra bin.
        let hist_sql = format!(
            "SELECT \
                CASE \
                    WHEN \"bin\" >= {num_bins} THEN {last_bin} \
                    ELSE \"bin\" \
                END AS \"clamped_bin\", \
                COUNT(*) AS \"cnt\" \
             FROM ( \
                SELECT FLOOR((CAST({c} AS {t}) - {min}) / {width}) AS \"bin\" \
                FROM {table} \
                WHERE {c} IS NOT NULL \
             ) AS \"__bins__\" \
             GROUP BY \"clamped_bin\" \
             ORDER BY \"clamped_bin\"",
            c = quoted_col,
            t = float_type,
            table = self.table_path,
            min = min_val,
            width = bin_width,
            num_bins = num_bins,
            last_bin = num_bins - 1,
        );

        let hist_df = reader.execute_sql(&hist_sql).ok()?;

        // Build bin_edges: num_bins + 1 edges
        let bin_edges: Vec<String> = (0..=num_bins)
            .map(|i| format!("{}", min_val + i as f64 * bin_width))
            .collect();

        // Build bin_counts: fill from query results (sparse bins get 0)
        let mut bin_counts = vec![0i64; num_bins];
        let bin_col = hist_df.column("clamped_bin").ok()?;
        let cnt_col = hist_df.column("cnt").ok()?;
        for i in 0..hist_df.height() {
            if let (Ok(bin_val), Ok(cnt_val)) = (bin_col.get(i), cnt_col.get(i)) {
                let bin_str = format!("{}", bin_val);
                // Parse bin index — may be float (e.g., "3.0") on some backends
                if let Ok(bin_idx) = bin_str.parse::<f64>() {
                    let idx = bin_idx as usize;
                    if idx < num_bins {
                        let count_str = format!("{}", cnt_val);
                        bin_counts[idx] = count_str.parse::<i64>().unwrap_or(0);
                    }
                }
            }
        }

        // Compute requested quantiles
        let quantiles_param = params
            .get("quantiles")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut quantile_results = Vec::new();
        let from_query = format!("SELECT * FROM {}", self.table_path);
        let col_name = col.name.replace('"', "\"\"");
        for q in &quantiles_param {
            if let Some(q_val) = q.as_f64() {
                let expr = dialect.sql_percentile(&col_name, q_val, &from_query, &[]);
                let q_sql = format!("SELECT {} AS \"q_val\"", expr);
                if let Ok(q_df) = reader.execute_sql(&q_sql) {
                    if let Some(v) = q_df
                        .column("q_val")
                        .ok()
                        .and_then(|c| c.get(0).ok())
                        .and_then(|v| {
                            if v.is_null() {
                                None
                            } else {
                                Some(format!("{}", v).trim_matches('"').to_string())
                            }
                        })
                    {
                        quantile_results.push(json!({"q": q_val, "value": v}));
                    }
                }
            }
        }

        Some(json!({
            "bin_edges": bin_edges,
            "bin_counts": bin_counts,
            "quantiles": quantile_results
        }))
    }

    /// Compute a frequency table for a string or boolean column.
    fn compute_frequency_table(
        &self,
        col: &ColumnInfo,
        params: &Value,
        reader: &dyn Reader,
    ) -> Option<Value> {
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;

        let quoted_col = format!("\"{}\"", col.name.replace('"', "\"\""));

        let sql = format!(
            "SELECT {c} AS \"value\", COUNT(*) AS \"count\" \
             FROM {table} \
             WHERE {c} IS NOT NULL \
             GROUP BY {c} \
             ORDER BY COUNT(*) DESC \
             LIMIT {limit}",
            c = quoted_col,
            table = self.table_path,
            limit = limit,
        );

        let df = reader.execute_sql(&sql).ok()?;

        let val_col = df.column("value").ok()?;
        let cnt_col = df.column("count").ok()?;

        let mut values = Vec::new();
        let mut counts = Vec::new();
        let mut top_total: i64 = 0;

        for i in 0..df.height() {
            if let (Ok(v), Ok(c)) = (val_col.get(i), cnt_col.get(i)) {
                let val_str = format!("{}", v).trim_matches('"').to_string();
                let count: i64 = format!("{}", c).parse().unwrap_or(0);
                values.push(Value::String(val_str));
                counts.push(count);
                top_total += count;
            }
        }

        // Compute other_count: total non-null rows minus the top-K sum
        let count_sql = format!(
            "SELECT COUNT({c}) AS \"total\" FROM {table}",
            c = quoted_col,
            table = self.table_path,
        );
        let other_count = reader
            .execute_sql(&count_sql)
            .ok()
            .and_then(|df| {
                df.column("total")
                    .ok()
                    .and_then(|c| c.get(0).ok())
                    .and_then(|v| format!("{}", v).parse::<i64>().ok())
            })
            .map(|total| total - top_total)
            .unwrap_or(0);

        Some(json!({
            "values": values,
            "counts": counts,
            "other_count": other_count
        }))
    }
}

/// Map a SQL type name (from information_schema) to a Positron display type.
fn sql_type_to_display(type_name: &str) -> &'static str {
    let upper = type_name.to_uppercase();
    let upper = upper.as_str();

    if upper.contains("INT") {
        return "integer";
    }
    if upper.contains("FLOAT")
        || upper.contains("DOUBLE")
        || upper.contains("REAL")
        || upper.contains("NUMERIC")
        || upper.contains("DECIMAL")
    {
        return "floating";
    }
    if upper.contains("BOOL") {
        return "boolean";
    }
    if upper.contains("TIMESTAMP") || upper.contains("DATETIME") {
        return "datetime";
    }
    if upper.contains("DATE") {
        return "date";
    }
    if upper.contains("TIME") {
        return "time";
    }
    if upper.contains("CHAR")
        || upper.contains("TEXT")
        || upper.contains("STRING")
        || upper.contains("VARCHAR")
        || upper.contains("CLOB")
    {
        return "string";
    }
    if upper.contains("BLOB") || upper.contains("BINARY") || upper.contains("BYTE") {
        return "string";
    }

    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_type_to_display() {
        assert_eq!(sql_type_to_display("INTEGER"), "integer");
        assert_eq!(sql_type_to_display("BIGINT"), "integer");
        assert_eq!(sql_type_to_display("SMALLINT"), "integer");
        assert_eq!(sql_type_to_display("TINYINT"), "integer");
        assert_eq!(sql_type_to_display("INT"), "integer");
        assert_eq!(sql_type_to_display("DOUBLE"), "floating");
        assert_eq!(sql_type_to_display("FLOAT"), "floating");
        assert_eq!(sql_type_to_display("REAL"), "floating");
        assert_eq!(sql_type_to_display("NUMERIC(10,2)"), "floating");
        assert_eq!(sql_type_to_display("DECIMAL(10,2)"), "floating");
        assert_eq!(sql_type_to_display("BOOLEAN"), "boolean");
        assert_eq!(sql_type_to_display("BOOL"), "boolean");
        assert_eq!(sql_type_to_display("VARCHAR"), "string");
        assert_eq!(sql_type_to_display("TEXT"), "string");
        assert_eq!(sql_type_to_display("DATE"), "date");
        assert_eq!(sql_type_to_display("TIMESTAMP"), "datetime");
        assert_eq!(sql_type_to_display("TIMESTAMP WITH TIME ZONE"), "datetime");
        assert_eq!(sql_type_to_display("TIME"), "time");
        assert_eq!(sql_type_to_display("BLOB"), "string");
        assert_eq!(sql_type_to_display("UNKNOWN_TYPE"), "unknown");
    }
}
