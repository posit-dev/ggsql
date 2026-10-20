//! Data explorer backend for the Positron data viewer.
//!
//! Implements the `positron.dataExplorer` comm protocol over SQL: every page,
//! filter, sort and profile is a query against the explorer's table, which is
//! never loaded into memory.

mod export;
mod filter;

pub use filter::Capabilities;

use crate::results::{self, StoredResult};
use ggsql::reader::Reader;
use ggsql::sql::FromItem;
use ggsql::DataFrame;
use serde_json::{json, Value};

/// Result of handling an RPC call.
pub struct RpcResponse {
    /// The JSON-RPC result to send as the reply, or the message of an error.
    pub result: Result<Value, String>,
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
        Self::answer(Ok(result))
    }

    /// Reply with a result or an error, and no async event.
    pub fn answer(result: Result<Value, String>) -> Self {
        Self {
            result,
            event: None,
        }
    }
}

/// The session's database, which Convert to Code writes SQL for. For a
/// retained result that is not the database the explorer queries.
pub struct Session<'a> {
    pub reader: &'a dyn Reader,
    pub caps: Capabilities,
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

/// Which database an explorer's table lives in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A table in the session's own database, opened from the Connections pane.
    Session,
    /// A retained query result in the private results database, by key.
    Result(String),
}

/// Which rows of the filtered, sorted view to read.
enum Rows {
    All,
    /// `first..=last`.
    Range(usize, usize),
}

/// State for one open data explorer comm.
pub struct DataExplorerState {
    source: Source,
    /// Fully qualified and quoted table path, e.g. `"memory"."main"."users"`.
    table_path: String,
    /// Display title shown in the data viewer tab.
    title: String,
    /// The query a retained result came from, which Convert to Code wraps.
    query: Option<String>,
    /// What the database holding the table can do.
    caps: Capabilities,
    /// Cached column schemas.
    columns: Vec<ColumnInfo>,
    /// Cached total row count.
    num_rows: usize,
    /// The row filters as last set, each marked valid or not, for `get_state`.
    row_filters: Vec<Value>,
    sort_keys: Vec<Value>,
    /// The valid row filters ANDed together; `None` when unfiltered.
    predicate: Option<String>,
    /// Rows passing the filters.
    selected_num_rows: usize,
}

impl DataExplorerState {
    /// Open a data explorer for a table at the given connection path.
    pub fn open(reader: &dyn Reader, path: &[String], caps: Capabilities) -> Result<Self, String> {
        let full = crate::connection::resolve_path(reader, path);
        if full.len() < 3 {
            return Err(format!(
                "Expected [catalog, schema, table] path, got {} elements",
                full.len()
            ));
        }
        let (catalog, schema, table) = (&full[0], &full[1], &full[2]);
        let mut state = Self::open_table(reader, catalog, schema, table, table)?;
        state.caps = caps;
        Ok(state)
    }

    /// Open a data explorer for a retained query result. `reader` is the
    /// private results database.
    pub fn open_result(reader: &dyn Reader, result: &StoredResult) -> Result<Self, String> {
        let mut state = Self::open_table(
            reader,
            results::CATALOG,
            results::SCHEMA,
            &result.table,
            &result.title,
        )?;
        state.source = Source::Result(result.key.clone());
        state.query = Some(result.query.clone());
        // The results database is DuckDB.
        state.caps = Capabilities { regex: true };
        Ok(state)
    }

    /// Which database this explorer's table lives in.
    pub fn source(&self) -> &Source {
        &self.source
    }

    /// Runs `SELECT COUNT(*)` and a column metadata query to cache schema
    /// information. Does **not** load the full table into memory.
    fn open_table(
        reader: &dyn Reader,
        catalog: &str,
        schema: &str,
        table: &str,
        title: &str,
    ) -> Result<Self, String> {
        let table_path = [catalog, schema, table]
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| ggsql::naming::quote_ident(s))
            .collect::<Vec<_>>()
            .join(".");

        let num_rows = count_rows(reader, &table_path, None)
            .map_err(|e| format!("Failed to count rows: {}", e))?;

        let column_infos = reader
            .list_columns(catalog, schema, table)
            .map_err(|e| format!("Failed to list columns: {}", e))?;

        let mut columns = Vec::new();
        for col in &column_infos {
            let type_display = sql_type_to_display(&col.data_type).to_string();
            let type_name = clean_type_name(&col.data_type);
            columns.push(ColumnInfo {
                name: col.name.clone(),
                type_name,
                type_display,
            });
        }

        Ok(Self {
            source: Source::Session,
            table_path,
            title: title.to_string(),
            query: None,
            caps: Capabilities { regex: false },
            columns,
            num_rows,
            row_filters: Vec::new(),
            sort_keys: Vec::new(),
            predicate: None,
            selected_num_rows: num_rows,
        })
    }

    /// Dispatch a JSON-RPC method call.
    ///
    /// `reader` holds the explorer's table. Returns the RPC result and an
    /// optional async event to send on iopub (used by `get_column_profiles` to
    /// deliver results asynchronously).
    pub fn handle_rpc(
        &mut self,
        method: &str,
        params: &Value,
        reader: &dyn Reader,
        session: Session,
    ) -> RpcResponse {
        match method {
            "get_state" => RpcResponse::reply(self.get_state()),
            "get_schema" => RpcResponse::reply(self.get_schema(params)),
            "get_data_values" => RpcResponse::answer(self.get_data_values(params, reader)),
            "get_column_profiles" => self.get_column_profiles(params, reader),
            "set_row_filters" => RpcResponse::answer(self.set_row_filters(params, reader)),
            "set_sort_columns" => RpcResponse::answer(self.set_sort_columns(params)),
            "search_schema" => RpcResponse::reply(json!({
                "matches": filter::search_schema(params, &self.columns)
            })),
            "export_data_selection" => {
                RpcResponse::answer(self.export_data_selection(params, reader))
            }
            "suggest_code_syntax" => RpcResponse::reply(json!({"code_syntax_name": "SQL"})),
            "convert_to_code" => RpcResponse::reply(self.convert_to_code(params, &session)),
            // Unsupported, and advertised as such in `get_state`.
            "set_column_filters" => RpcResponse::reply(json!(null)),
            _ => {
                tracing::warn!("Unhandled data explorer method: {}", method);
                RpcResponse::answer(Err(format!("Unknown method {method:?}")))
            }
        }
    }

    fn get_state(&self) -> Value {
        let num_columns = self.columns.len();
        let row_filter_types: Vec<Value> = [
            "between",
            "compare",
            "is_empty",
            "is_false",
            "is_null",
            "is_true",
            "not_between",
            "not_empty",
            "not_null",
            "search",
            "set_membership",
        ]
        .iter()
        .map(|t| json!({"row_filter_type": t, "support_status": "supported"}))
        .collect();

        json!({
            "display_name": self.title,
            "table_shape": {
                "num_rows": self.selected_num_rows,
                "num_columns": num_columns
            },
            "table_unfiltered_shape": {
                "num_rows": self.num_rows,
                "num_columns": num_columns
            },
            "has_row_labels": false,
            "column_filters": [],
            "row_filters": self.row_filters,
            "sort_keys": self.sort_keys,
            "supported_features": {
                "search_schema": {
                    "support_status": "supported",
                    "supported_types": [
                        {"column_filter_type": "text_search", "support_status": "supported"},
                        {"column_filter_type": "match_data_types", "support_status": "supported"}
                    ]
                },
                "set_column_filters": {
                    "support_status": "unsupported",
                    "supported_types": []
                },
                "set_row_filters": {
                    "support_status": "supported",
                    // Filters are always ANDed.
                    "supports_conditions": "unsupported",
                    "supported_types": row_filter_types
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
                    "support_status": "supported"
                },
                "export_data_selection": {
                    "support_status": "supported",
                    "supported_formats": ["csv", "tsv", "html"]
                },
                "convert_to_code": {
                    "support_status": "supported",
                    "code_syntaxes": [{"code_syntax_name": "SQL"}]
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

    /// The filtered table as a `FROM` item, for the queries that aggregate it.
    fn relation(&self) -> String {
        match &self.predicate {
            None => self.table_path.clone(),
            Some(p) => format!(
                "(SELECT * FROM {} WHERE {}) AS \"__ggsql_filtered__\"",
                self.table_path, p
            ),
        }
    }

    /// Read `rows` of the filtered, sorted view, for the columns `cols`.
    fn fetch(&self, reader: &dyn Reader, cols: &[usize], rows: Rows) -> Result<DataFrame, String> {
        let names = cols
            .iter()
            .map(|&i| {
                self.columns
                    .get(i)
                    .map(|c| ggsql::naming::quote_ident(&c.name))
                    .ok_or_else(|| format!("There is no column {i}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if names.is_empty() {
            return Ok(DataFrame::empty());
        }

        let filter = self
            .predicate
            .as_ref()
            .map(|p| format!(" WHERE {p}"))
            .unwrap_or_default();
        // A retained result is a DuckDB table, whose `rowid` is its original
        // row order; that keeps paging through equal sort keys stable.
        let tiebreak = matches!(self.source, Source::Result(_)).then_some("rowid");
        let order = filter::order_by(&self.sort_keys, &self.columns, tiebreak)?;
        let limit = match rows {
            Rows::All => String::new(),
            Rows::Range(first, last) => {
                format!(" LIMIT {} OFFSET {}", last.saturating_sub(first) + 1, first)
            }
        };

        let sql = format!(
            "SELECT {} FROM {}{filter}{order}{limit}",
            names.join(", "),
            self.table_path,
        );
        reader.execute_sql(&sql).map_err(|e| e.to_string())
    }

    fn get_data_values(&self, params: &Value, reader: &dyn Reader) -> Result<Value, String> {
        let selections = params["columns"].as_array().cloned().unwrap_or_default();
        if selections.is_empty() {
            return Ok(json!({ "columns": [] }));
        }

        // Each column asks for a range or a list of rows. One query covers the
        // span of them all, and each column then takes the rows it asked for.
        let wanted: Vec<Vec<usize>> = selections
            .iter()
            .map(|sel| {
                let spec = &sel["spec"];
                match spec["indices"].as_array() {
                    Some(indices) => indices
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect(),
                    None => {
                        let first = spec["first_index"].as_u64().unwrap_or(0) as usize;
                        let last = spec["last_index"].as_u64().unwrap_or(0) as usize;
                        (first..=last).collect()
                    }
                }
            })
            .collect();
        let first = wanted.iter().flatten().min().copied().unwrap_or(0);
        let last = wanted.iter().flatten().max().copied().unwrap_or(0);

        let cols: Vec<usize> = selections
            .iter()
            .map(|sel| sel["column_index"].as_u64().unwrap_or(0) as usize)
            .collect();
        let df = self.fetch(reader, &cols, Rows::Range(first, last))?;

        let columns: Vec<Vec<Value>> = wanted
            .iter()
            .enumerate()
            .map(|(i, rows)| {
                let col = &df.get_columns()[i];
                rows.iter()
                    .map(|&r| r - first)
                    .filter(|&r| r < df.height())
                    .map(|r| cell_value(col, r))
                    .collect()
            })
            .collect();

        Ok(json!({ "columns": columns }))
    }

    /// Apply row filters, returning how many rows pass.
    ///
    /// A filter that cannot be built, or that the database rejects, is marked
    /// invalid with the reason and left out rather than failing the rest.
    fn set_row_filters(&mut self, params: &Value, reader: &dyn Reader) -> Result<Value, String> {
        let mut filters = params["filters"].as_array().cloned().unwrap_or_default();
        let mut had_errors = false;
        let mut invalidate = |filter: &mut Value, message: String| {
            filter["is_valid"] = json!(false);
            filter["error_message"] = json!(message);
            had_errors = true;
        };

        let mut built = Vec::new();
        for (i, f) in filters.iter_mut().enumerate() {
            match filter::predicate(f, &self.columns, reader.dialect(), self.caps) {
                Ok(p) => built.push((i, p)),
                Err(e) => invalidate(f, e),
            }
        }

        let combine = |built: &[(usize, String)]| {
            (!built.is_empty()).then(|| {
                built
                    .iter()
                    .map(|(_, p)| format!("({p})"))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            })
        };

        let mut count = count_rows(reader, &self.table_path, combine(&built).as_deref());
        if count.is_err() {
            // Find the filters the database rejects, and count without them.
            built.retain(
                |(i, p)| match count_rows(reader, &self.table_path, Some(p)) {
                    Ok(_) => true,
                    Err(e) => {
                        invalidate(&mut filters[*i], e);
                        false
                    }
                },
            );
            count = count_rows(reader, &self.table_path, combine(&built).as_deref());
        }
        let selected = count?;

        for (i, _) in &built {
            filters[*i]["is_valid"] = json!(true);
        }
        self.row_filters = filters;
        self.predicate = combine(&built);
        self.selected_num_rows = selected;

        Ok(json!({
            "selected_num_rows": selected,
            "had_errors": had_errors
        }))
    }

    fn set_sort_columns(&mut self, params: &Value) -> Result<Value, String> {
        let keys = params["sort_keys"].as_array().cloned().unwrap_or_default();
        // Checked here so a bad key fails the request, not every page after it.
        filter::order_by(&keys, &self.columns, None)?;
        self.sort_keys = keys;
        Ok(json!({}))
    }

    /// The selected cells as CSV, TSV or HTML.
    fn export_data_selection(&self, params: &Value, reader: &dyn Reader) -> Result<Value, String> {
        let format = params["format"].as_str().unwrap_or("csv");
        let kind = params["selection"]["kind"].as_str().unwrap_or("");
        let sel = &params["selection"]["selection"];

        let index = |v: &Value| v.as_u64().map(|n| n as usize).ok_or("Bad selection index");
        let span = |first: &Value, last: &Value| -> Result<Vec<usize>, String> {
            Ok((index(first)?..=index(last)?).collect())
        };
        let list = |v: &Value| -> Result<Vec<usize>, String> {
            v.as_array()
                .ok_or("Bad selection indices")?
                .iter()
                .map(|i| index(i).map_err(str::to_string))
                .collect()
        };
        let all_columns: Vec<usize> = (0..self.columns.len()).collect();

        // `None` rows means every row of the view.
        let (rows, cols): (Option<Vec<usize>>, Vec<usize>) = match kind {
            "single_cell" => (
                Some(vec![index(&sel["row_index"])?]),
                vec![index(&sel["column_index"])?],
            ),
            "cell_range" => (
                Some(span(&sel["first_row_index"], &sel["last_row_index"])?),
                span(&sel["first_column_index"], &sel["last_column_index"])?,
            ),
            "column_range" => (None, span(&sel["first_index"], &sel["last_index"])?),
            "row_range" => (
                Some(span(&sel["first_index"], &sel["last_index"])?),
                all_columns,
            ),
            "column_indices" => (None, list(&sel["indices"])?),
            "row_indices" => (Some(list(&sel["indices"])?), all_columns),
            "cell_indices" => (
                Some(list(&sel["row_indices"])?),
                list(&sel["column_indices"])?,
            ),
            other => return Err(format!("Unknown selection kind {other:?}")),
        };

        let (df, offset) = match &rows {
            None => (self.fetch(reader, &cols, Rows::All)?, 0),
            Some(rows) => {
                let first = rows.iter().min().copied().unwrap_or(0);
                let last = rows.iter().max().copied().unwrap_or(0);
                (self.fetch(reader, &cols, Rows::Range(first, last))?, first)
            }
        };
        let picked: Vec<usize> = match &rows {
            None => (0..df.height()).collect(),
            Some(rows) => rows
                .iter()
                .map(|r| r - offset)
                .filter(|&r| r < df.height())
                .collect(),
        };
        let cells: Vec<Vec<String>> = picked
            .iter()
            .map(|&r| df.get_columns().iter().map(|c| cell_string(c, r)).collect())
            .collect();

        // A single cell is copied as its bare value.
        let data = if kind == "single_cell" {
            cells
                .first()
                .and_then(|r| r.first())
                .cloned()
                .unwrap_or_default()
        } else {
            let header: Vec<String> = cols.iter().map(|&i| self.columns[i].name.clone()).collect();
            export::format(Some(&header), &cells, format)?
        };
        Ok(json!({ "data": data, "format": format }))
    }

    /// SQL for the session's database reproducing the explorer's view.
    fn convert_to_code(&self, params: &Value, session: &Session) -> Value {
        let from = match &self.query {
            Some(query) => filter::subquery(query),
            None => self.table_path.clone(),
        };
        let lines = filter::convert_to_code(
            params,
            &from,
            &self.columns,
            session.reader.dialect(),
            session.caps,
        );
        json!({ "converted_code": lines })
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
                    result: Ok(json!({})),
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
            result: Ok(json!({})),
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
        let quoted_col = ggsql::naming::quote_ident(&col.name);
        let display = col.type_display.as_str();

        // Build a single SQL query that computes all needed aggregates.
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
                    let float_type = dialect.type_names().number.unwrap_or("DOUBLE PRECISION");
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
            self.relation()
        );

        let df = match reader.execute_sql(&sql) {
            Ok(df) => df,
            Err(e) => {
                tracing::error!("Column profile query failed: {}", e);
                return json!({});
            }
        };

        let get_str = |name: &str| -> Option<String> {
            use arrow::array::Array;
            df.column(name).ok().and_then(|c| {
                if c.is_empty() || c.is_null(0) {
                    None
                } else {
                    Some(
                        ggsql::array_util::value_to_string(c, 0)
                            .trim_matches('"')
                            .to_string(),
                    )
                }
            })
        };

        let get_i64 =
            |name: &str| -> Option<i64> { get_str(name).and_then(|s| s.parse::<i64>().ok()) };

        let get_f64 =
            |name: &str| -> Option<f64> { get_str(name).and_then(|s| s.parse::<f64>().ok()) };

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
                    // Median via the dialect's quantile hook. The expression
                    // must sit in a query over `from` (native dialects return
                    // a bare aggregate, which without a FROM would compute
                    // over an empty input and yield NULL).
                    let from_query = format!("SELECT * FROM {}", self.relation());
                    let median_expr =
                        dialect.sql_quantile(&col.name, 0.5, FromItem::Query(&from_query), &[]);
                    let median_sql = format!(
                        "SELECT {} AS \"median_val\" FROM ({}) AS \"__ggsql_qt__\"",
                        median_expr, from_query
                    );
                    if let Ok(median_df) = reader.execute_sql(&median_sql) {
                        use arrow::array::Array;
                        if let Some(v) = median_df.column("median_val").ok().and_then(|c| {
                            if c.is_empty() || c.is_null(0) {
                                None
                            } else {
                                Some(
                                    ggsql::array_util::value_to_string(c, 0)
                                        .trim_matches('"')
                                        .to_string(),
                                )
                            }
                        }) {
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
        let float_type = dialect.type_names().number.unwrap_or("DOUBLE PRECISION");
        let quoted_col = ggsql::naming::quote_ident(&col.name);
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
            table = self.relation(),
        );

        let bounds_df = reader.execute_sql(&bounds_sql).ok()?;
        let get_f64 = |name: &str| -> Option<f64> {
            use arrow::array::Array;
            bounds_df.column(name).ok().and_then(|c| {
                if c.is_empty() || c.is_null(0) {
                    None
                } else {
                    ggsql::array_util::value_to_string(c, 0)
                        .trim_matches('"')
                        .parse::<f64>()
                        .ok()
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
            table = self.relation(),
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
            let bin_str = ggsql::array_util::value_to_string(bin_col, i);
            // Parse bin index — may be float (e.g., "3.0") on some backends
            if let Ok(bin_idx) = bin_str.parse::<f64>() {
                let idx = bin_idx as usize;
                if idx < num_bins {
                    let count_str = ggsql::array_util::value_to_string(cnt_col, i);
                    bin_counts[idx] = count_str.parse::<i64>().unwrap_or(0);
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
        let from_query = format!("SELECT * FROM {}", self.relation());
        for q in &quantiles_param {
            if let Some(q_val) = q.as_f64() {
                let expr =
                    dialect.sql_quantile(&col.name, q_val, FromItem::Query(&from_query), &[]);
                // See the median path above: the expression needs the FROM.
                let q_sql = format!(
                    "SELECT {} AS \"q_val\" FROM ({}) AS \"__ggsql_qt__\"",
                    expr, from_query
                );
                if let Ok(q_df) = reader.execute_sql(&q_sql) {
                    use arrow::array::Array;
                    if let Some(v) = q_df.column("q_val").ok().and_then(|c| {
                        if c.is_empty() || c.is_null(0) {
                            None
                        } else {
                            Some(
                                ggsql::array_util::value_to_string(c, 0)
                                    .trim_matches('"')
                                    .to_string(),
                            )
                        }
                    }) {
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
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(8) as usize;

        let quoted_col = ggsql::naming::quote_ident(&col.name);

        let sql = format!(
            "SELECT {c} AS \"value\", COUNT(*) AS \"count\" \
             FROM {table} \
             WHERE {c} IS NOT NULL \
             GROUP BY {c} \
             ORDER BY COUNT(*) DESC \
             LIMIT {limit}",
            c = quoted_col,
            table = self.relation(),
            limit = limit,
        );

        let df = reader.execute_sql(&sql).ok()?;

        let val_col = df.column("value").ok()?;
        let cnt_col = df.column("count").ok()?;

        let mut values = Vec::new();
        let mut counts = Vec::new();
        let mut top_total: i64 = 0;

        for i in 0..df.height() {
            let val_str = ggsql::array_util::value_to_string(val_col, i)
                .trim_matches('"')
                .to_string();
            let count: i64 = ggsql::array_util::value_to_string(cnt_col, i)
                .parse()
                .unwrap_or(0);
            values.push(Value::String(val_str));
            counts.push(count);
            top_total += count;
        }

        // Compute other_count: total non-null rows minus the top-K sum
        let count_sql = format!(
            "SELECT COUNT({c}) AS \"total\" FROM {table}",
            c = quoted_col,
            table = self.relation(),
        );
        let other_count = reader
            .execute_sql(&count_sql)
            .ok()
            .and_then(|df| {
                use arrow::array::Array;
                df.column("total").ok().and_then(|c| {
                    if c.is_empty() || c.is_null(0) {
                        None
                    } else {
                        ggsql::array_util::value_to_string(c, 0).parse::<i64>().ok()
                    }
                })
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

/// `SELECT COUNT(*)` over `table`, filtered by `predicate` when given.
fn count_rows(reader: &dyn Reader, table: &str, predicate: Option<&str>) -> Result<usize, String> {
    let filter = predicate.map(|p| format!(" WHERE {p}")).unwrap_or_default();
    let sql = format!("SELECT COUNT(*) AS \"n\" FROM {table}{filter}");
    let df = reader.execute_sql(&sql).map_err(|e| e.to_string())?;
    let col = df.column("n").map_err(|e| e.to_string())?;
    if col.is_empty() {
        return Ok(0);
    }
    ggsql::array_util::value_to_string(col, 0)
        .parse::<usize>()
        .map_err(|e| e.to_string())
}

/// A cell as text, with string values unquoted and a null as empty.
fn cell_string(col: &arrow::array::ArrayRef, row: usize) -> String {
    use arrow::array::Array;
    if col.is_null(row) {
        return String::new();
    }
    ggsql::array_util::value_to_string(col, row)
        .trim_matches('"')
        .to_string()
}

/// A cell as Positron's `ColumnValue`: formatted text, or a number coding a
/// special value — 0 null, 2 NaN, 10 infinity, 11 negative infinity.
fn cell_value(col: &arrow::array::ArrayRef, row: usize) -> Value {
    use arrow::array::Array;
    if col.is_null(row) {
        return json!(0);
    }
    let s = cell_string(col, row);
    if col.data_type().is_floating() {
        match s.to_ascii_lowercase().as_str() {
            "nan" => return json!(2),
            "inf" | "infinity" => return json!(10),
            "-inf" | "-infinity" => return json!(11),
            _ => {}
        }
    }
    Value::String(s)
}

/// Map a SQL type name (from information_schema or SHOW COLUMNS) to a Positron display type.
///
/// Handles both simple type names (e.g. "INTEGER", "VARCHAR") and Snowflake's
/// JSON format (e.g. `{"type":"FIXED","precision":38,"scale":0,...}`).
fn sql_type_to_display(type_name: &str) -> &'static str {
    // Handle Snowflake JSON type format
    if type_name.starts_with('{') {
        if let Ok(obj) = serde_json::from_str::<Value>(type_name) {
            if let Some(t) = obj.get("type").and_then(|v| v.as_str()) {
                return match t {
                    "FIXED" => {
                        let scale = obj.get("scale").and_then(|v| v.as_i64()).unwrap_or(0);
                        if scale > 0 {
                            "floating"
                        } else {
                            "integer"
                        }
                    }
                    "REAL" | "FLOAT" => "floating",
                    "TEXT" => "string",
                    "BOOLEAN" => "boolean",
                    "DATE" => "date",
                    "TIMESTAMP_NTZ" | "TIMESTAMP_LTZ" | "TIMESTAMP_TZ" => "datetime",
                    "TIME" => "time",
                    "BINARY" => "string",
                    "VARIANT" | "OBJECT" | "ARRAY" => "string",
                    _ => "unknown",
                };
            }
        }
    }

    // Simple type names (DuckDB, PostgreSQL, SQLite, etc.)
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

/// Clean up a raw type name for display in the schema response.
///
/// For Snowflake JSON types, extracts the `type` field (e.g. "NUMBER", "TEXT").
/// For simple type names, returns as-is.
fn clean_type_name(type_name: &str) -> String {
    if type_name.starts_with('{') {
        if let Ok(obj) = serde_json::from_str::<Value>(type_name) {
            if let Some(t) = obj.get("type").and_then(|v| v.as_str()) {
                return match t {
                    "FIXED" => {
                        let scale = obj.get("scale").and_then(|v| v.as_i64()).unwrap_or(0);
                        if scale > 0 {
                            format!(
                                "NUMBER({},{})",
                                obj.get("precision").and_then(|v| v.as_i64()).unwrap_or(38),
                                scale
                            )
                        } else {
                            "NUMBER".to_string()
                        }
                    }
                    other => other.to_string(),
                };
            }
        }
    }
    type_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUERY: &str = "SELECT i AS n, 'r' || i AS label FROM range(10) t(i)";

    /// An explorer on QUERY's result, with the store that keeps its table.
    fn explorer() -> (crate::results::ResultStore, DataExplorerState) {
        let df = ggsql::reader::connection::reader_from_uri("duckdb://memory")
            .unwrap()
            .execute_sql(QUERY)
            .unwrap();
        let mut store = crate::results::ResultStore::new(4).unwrap();
        let key = store
            .insert(df, "Query [3]".into(), QUERY.to_string(), |_| false)
            .unwrap();
        let state =
            DataExplorerState::open_result(store.reader(), store.get(&key).unwrap()).unwrap();
        assert_eq!(*state.source(), Source::Result(key));
        (store, state)
    }

    fn rpc(
        state: &mut DataExplorerState,
        store: &crate::results::ResultStore,
        method: &str,
        params: Value,
    ) -> Result<Value, String> {
        let session = Session {
            reader: store.reader(),
            caps: Capabilities { regex: true },
        };
        state
            .handle_rpc(method, &params, store.reader(), session)
            .result
    }

    /// The first `n` values of a column, as the grid would ask for them.
    fn page(
        state: &mut DataExplorerState,
        store: &crate::results::ResultStore,
        column: usize,
        n: usize,
    ) -> Value {
        let params = json!({"columns": [
            {"column_index": column, "spec": {"first_index": 0, "last_index": n - 1}}
        ]});
        rpc(state, store, "get_data_values", params).unwrap()["columns"][0].clone()
    }

    fn compare(column: usize, op: &str, value: &str) -> Value {
        json!({
            "filter_id": format!("{column}{op}{value}"),
            "filter_type": "compare",
            "column_schema": {"column_index": column},
            "condition": "and",
            "params": {"op": op, "value": value}
        })
    }

    #[test]
    fn test_a_retained_result_opens_and_pages() {
        let (store, mut state) = explorer();

        let st = rpc(&mut state, &store, "get_state", json!({})).unwrap();
        assert_eq!(st["display_name"], "Query [3]");
        assert_eq!(st["table_shape"]["num_rows"], 10);
        assert_eq!(st["table_shape"]["num_columns"], 2);

        let values = rpc(
            &mut state,
            &store,
            "get_data_values",
            json!({"columns": [
                {"column_index": 0, "spec": {"first_index": 1, "last_index": 2}},
                {"column_index": 1, "spec": {"indices": [4, 2]}}
            ]}),
        )
        .unwrap();
        assert_eq!(values["columns"], json!([["1", "2"], ["r4", "r2"]]));
    }

    #[test]
    fn test_filters_narrow_the_view() {
        let (store, mut state) = explorer();
        let reply = rpc(
            &mut state,
            &store,
            "set_row_filters",
            json!({"filters": [compare(0, ">=", "5"), compare(0, "!=", "7")]}),
        )
        .unwrap();
        assert_eq!(reply, json!({"selected_num_rows": 4, "had_errors": false}));

        let st = rpc(&mut state, &store, "get_state", json!({})).unwrap();
        assert_eq!(st["table_shape"]["num_rows"], 4);
        assert_eq!(st["table_unfiltered_shape"]["num_rows"], 10);
        assert_eq!(st["row_filters"][0]["is_valid"], true);
        assert_eq!(page(&mut state, &store, 0, 4), json!(["5", "6", "8", "9"]));
    }

    #[test]
    fn test_a_bad_filter_is_left_out_and_says_why() {
        let (store, mut state) = explorer();
        // One the kernel rejects, and one only the database can: `(` is not a
        // valid regular expression.
        let bad_regex = json!({
            "filter_id": "re",
            "filter_type": "search",
            "column_schema": {"column_index": 1},
            "condition": "and",
            "params": {"search_type": "regex_match", "term": "(", "case_sensitive": true}
        });
        let reply = rpc(
            &mut state,
            &store,
            "set_row_filters",
            json!({"filters": [compare(0, "=", "abc"), bad_regex, compare(0, "<", "3")]}),
        )
        .unwrap();
        assert_eq!(reply, json!({"selected_num_rows": 3, "had_errors": true}));

        let st = rpc(&mut state, &store, "get_state", json!({})).unwrap();
        let filters = st["row_filters"].as_array().unwrap();
        assert_eq!(filters[0]["is_valid"], false);
        assert!(filters[0]["error_message"]
            .as_str()
            .unwrap()
            .contains("not a number"));
        assert_eq!(filters[1]["is_valid"], false);
        assert_eq!(filters[2]["is_valid"], true);
    }

    #[test]
    fn test_sorting_applies_on_top_of_filters() {
        let (store, mut state) = explorer();
        rpc(
            &mut state,
            &store,
            "set_row_filters",
            json!({"filters": [compare(0, "<", "4")]}),
        )
        .unwrap();
        rpc(
            &mut state,
            &store,
            "set_sort_columns",
            json!({"sort_keys": [{"column_index": 0, "ascending": false}]}),
        )
        .unwrap();
        assert_eq!(
            page(&mut state, &store, 1, 4),
            json!(["r3", "r2", "r1", "r0"])
        );

        // Clearing the sort restores the query's own order.
        rpc(
            &mut state,
            &store,
            "set_sort_columns",
            json!({"sort_keys": []}),
        )
        .unwrap();
        assert_eq!(page(&mut state, &store, 1, 2), json!(["r0", "r1"]));
    }

    #[test]
    fn test_profiles_see_only_the_filtered_rows() {
        let (store, mut state) = explorer();
        rpc(
            &mut state,
            &store,
            "set_row_filters",
            json!({"filters": [compare(0, ">=", "6")]}),
        )
        .unwrap();
        let response = state.handle_rpc(
            "get_column_profiles",
            &json!({"callback_id": "cb", "profiles": [{
                "column_index": 0,
                "profiles": [{"profile_type": "summary_stats"}]
            }]}),
            store.reader(),
            Session {
                reader: store.reader(),
                caps: Capabilities { regex: true },
            },
        );
        let event = response.event.unwrap();
        let stats = &event.params["profiles"][0]["summary_stats"]["number_stats"];
        assert_eq!(stats["min_value"], "6");
        assert_eq!(stats["max_value"], "9");
    }

    #[test]
    fn test_export_follows_the_view() {
        let (store, mut state) = explorer();
        rpc(
            &mut state,
            &store,
            "set_sort_columns",
            json!({"sort_keys": [{"column_index": 0, "ascending": false}]}),
        )
        .unwrap();

        let csv = rpc(
            &mut state,
            &store,
            "export_data_selection",
            json!({"format": "csv", "selection": {"kind": "cell_range", "selection": {
                "first_row_index": 0, "last_row_index": 1,
                "first_column_index": 0, "last_column_index": 1
            }}}),
        )
        .unwrap();
        assert_eq!(csv["data"], "n,label\n9,r9\n8,r8");

        let cell = rpc(
            &mut state,
            &store,
            "export_data_selection",
            json!({"format": "tsv", "selection": {"kind": "single_cell", "selection": {
                "row_index": 2, "column_index": 1
            }}}),
        )
        .unwrap();
        assert_eq!(cell["data"], "r7");

        let rows = rpc(
            &mut state,
            &store,
            "export_data_selection",
            json!({"format": "tsv", "selection": {"kind": "row_indices", "selection": {
                "indices": [3, 0]
            }}}),
        )
        .unwrap();
        assert_eq!(rows["data"], "n\tlabel\n6\tr6\n9\tr9");
    }

    #[test]
    fn test_convert_to_code_wraps_the_original_query() {
        let (store, mut state) = explorer();
        let code = rpc(
            &mut state,
            &store,
            "convert_to_code",
            json!({
                "column_filters": [],
                "row_filters": [compare(0, ">", "2")],
                "sort_keys": [{"column_index": 1, "ascending": true}],
                "code_syntax_name": {"code_syntax_name": "SQL"}
            }),
        )
        .unwrap();
        assert_eq!(
            code["converted_code"],
            json!([
                "SELECT *",
                "FROM (",
                format!("    {QUERY}"),
                ") AS \"result\"",
                "WHERE \"n\" > 2",
                "ORDER BY \"label\" ASC"
            ])
        );
    }

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
