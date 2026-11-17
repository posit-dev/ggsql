//! One view of a result: its filters and sort, and the rows they select.
//!
//! The result is a table in the results cache, with a [`ROW`] column holding
//! each row's position in the query's own output. Once a view is filtered or
//! sorted, it keeps an **order table** of `(pos, ROW)` pairs: `pos` is a row's
//! position in the view. A page then joins that small table back to the
//! result, so scrolling never sorts again. An unfiltered, unsorted view needs
//! no order table: its positions are the `ROW` values themselves.

use super::filter::{self, Capabilities, Filter, SortKey};
use super::{export, Column, ROW};
use ggsql::naming::quote_ident;
use ggsql::reader::Reader;
use ggsql::sql::FromItem;
use ggsql::DataFrame;
use serde::Deserialize;
use serde_json::{json, Value};

/// The rows of a view to read.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Rows {
    /// `first..=last`.
    Range { first: usize, last: usize },
    /// These positions, in this order.
    Indices { indices: Vec<usize> },
}

impl Rows {
    fn positions(&self) -> Vec<usize> {
        match self {
            Rows::Range { first, last } => (*first..=*last).collect(),
            Rows::Indices { indices } => indices.clone(),
        }
    }
}

/// Which profiles to compute for a column.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProfileRequest {
    #[serde(default)]
    pub null_count: bool,
    #[serde(default)]
    pub summary: bool,
    pub histogram: Option<HistogramRequest>,
    pub frequency: Option<FrequencyRequest>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HistogramRequest {
    #[serde(default = "default_bins")]
    pub bins: usize,
    #[serde(default)]
    pub quantiles: Vec<f64>,
}

fn default_bins() -> usize {
    20
}

#[derive(Debug, Clone, Deserialize)]
pub struct FrequencyRequest {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    8
}

/// A view of a result table.
pub struct ResultView {
    /// The result this views.
    pub key: String,
    /// The result table, quoted.
    table: String,
    /// The order table's name, used once the view is filtered or sorted.
    order_name: String,
    /// Whether the order table exists.
    ordered: bool,
    pub columns: Vec<Column>,
    /// Rows in the result.
    pub num_rows: usize,
    /// Rows passing the filters.
    pub selected_rows: usize,
    /// The filters as last set, valid or not.
    pub filters: Vec<Filter>,
    /// Why each of `filters` was left out, or `None` for one that applies.
    rejected: Vec<Option<String>>,
    /// The valid filters ANDed together, or `None` when unfiltered.
    predicate: Option<String>,
    pub sort: Vec<SortKey>,
}

impl ResultView {
    pub fn new(key: &str, id: &str, table: &str, columns: Vec<Column>, num_rows: usize) -> Self {
        Self {
            key: key.to_string(),
            table: quote_ident(table),
            order_name: format!("__ggsql_view_{id}__"),
            ordered: false,
            columns,
            num_rows,
            selected_rows: num_rows,
            filters: Vec::new(),
            rejected: Vec::new(),
            predicate: None,
            sort: Vec::new(),
        }
    }

    /// The result table filtered, as a `FROM` item, for the queries that
    /// aggregate it.
    fn relation(&self) -> String {
        match &self.predicate {
            None => self.table.clone(),
            Some(p) => format!(
                "(SELECT * FROM {} WHERE {}) AS \"__ggsql_filtered__\"",
                self.table, p
            ),
        }
    }

    /// Set the row filters, returning an error message per filter (`None` for
    /// a valid one). A filter that cannot be built, or that the database
    /// rejects, is left out rather than failing the rest.
    pub fn set_filters(
        &mut self,
        filters: Vec<Filter>,
        reader: &dyn Reader,
        caps: Capabilities,
    ) -> Result<Vec<Option<String>>, String> {
        let mut errors: Vec<Option<String>> = vec![None; filters.len()];
        let mut built = Vec::new();
        for (i, f) in filters.iter().enumerate() {
            match filter::predicate(f, &self.columns, reader.dialect(), caps) {
                Ok(p) => built.push((i, p)),
                Err(e) => errors[i] = Some(e),
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
        if count_rows(reader, &self.table, combine(&built).as_deref()).is_err() {
            // Find the filters the database rejects.
            built.retain(|(i, p)| match count_rows(reader, &self.table, Some(p)) {
                Ok(_) => true,
                Err(e) => {
                    errors[*i] = Some(e);
                    false
                }
            });
        }

        self.filters = filters;
        self.rejected = errors.clone();
        self.predicate = combine(&built);
        self.rebuild(reader)?;
        Ok(errors)
    }

    /// Set the sort keys.
    pub fn set_sort(&mut self, keys: Vec<SortKey>, reader: &dyn Reader) -> Result<(), String> {
        filter::order_terms(&keys, &self.columns)?;
        self.sort = keys;
        self.rebuild(reader)
    }

    /// Rebuild the order table for the current filters and sort, and count
    /// the rows they select.
    fn rebuild(&mut self, reader: &dyn Reader) -> Result<(), String> {
        self.drop_order(reader);
        if self.predicate.is_none() && self.sort.is_empty() {
            self.selected_rows = self.num_rows;
            return Ok(());
        }

        // Ties break on the query's own order, so paging is stable and clearing
        // the sort restores it.
        let mut terms = filter::order_terms(&self.sort, &self.columns)?;
        terms.push(quote_ident(ROW));
        let filter = self
            .predicate
            .as_ref()
            .map(|p| format!(" WHERE {p}"))
            .unwrap_or_default();
        let body = format!(
            "SELECT ROW_NUMBER() OVER (ORDER BY {}) - 1 AS \"pos\", {row} FROM {}{filter}",
            terms.join(", "),
            self.table,
            row = quote_ident(ROW),
        );
        let dialect = reader.dialect();
        for stmt in dialect.create_or_replace_temp_table_sql(&self.order_name, &[], &body) {
            reader.execute_sql(&stmt).map_err(|e| e.to_string())?;
        }
        self.ordered = true;
        self.selected_rows = count_rows(reader, &quote_ident(&self.order_name), None)?;
        Ok(())
    }

    fn drop_order(&mut self, reader: &dyn Reader) {
        if self.ordered {
            let sql = reader.dialect().drop_table_sql(&self.order_name);
            if let Err(e) = reader.execute_sql(&sql) {
                tracing::warn!("Failed to drop {}: {}", self.order_name, e);
            }
            self.ordered = false;
        }
    }

    /// Drop what the view keeps in the cache.
    pub fn close(&mut self, reader: &dyn Reader) {
        self.drop_order(reader);
    }

    /// Read the view's rows `first..=last`, or every row, for `cols`.
    fn fetch(
        &self,
        reader: &dyn Reader,
        cols: &[usize],
        span: Option<(usize, usize)>,
    ) -> Result<DataFrame, String> {
        let names = cols
            .iter()
            .map(|&i| {
                self.columns
                    .get(i)
                    .map(|c| format!("t.{}", quote_ident(&c.name)))
                    .ok_or_else(|| format!("There is no column {i}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if names.is_empty() {
            return Ok(DataFrame::empty());
        }
        let row = quote_ident(ROW);
        let sql = if self.ordered {
            let range = span
                .map(|(a, b)| format!(" WHERE o.\"pos\" BETWEEN {a} AND {b}"))
                .unwrap_or_default();
            format!(
                "SELECT {} FROM {} AS o JOIN {} AS t ON t.{row} = o.{row}{range} ORDER BY o.\"pos\"",
                names.join(", "),
                quote_ident(&self.order_name),
                self.table,
            )
        } else {
            let range = span
                .map(|(a, b)| format!(" WHERE t.{row} BETWEEN {a} AND {b}"))
                .unwrap_or_default();
            format!(
                "SELECT {} FROM {} AS t{range} ORDER BY t.{row}",
                names.join(", "),
                self.table,
            )
        };
        reader.execute_sql(&sql).map_err(|e| e.to_string())
    }

    /// The cells at `rows` of each of `cols`, as `columns[col][row]`: a string,
    /// or `null` for a null.
    pub fn page(
        &self,
        reader: &dyn Reader,
        rows: &Rows,
        cols: &[usize],
    ) -> Result<Vec<Vec<Value>>, String> {
        let positions = rows.positions();
        let (Some(&first), Some(&last)) = (positions.iter().min(), positions.iter().max()) else {
            return Ok(vec![Vec::new(); cols.len()]);
        };
        let df = self.fetch(reader, cols, Some((first, last)))?;
        Ok(df
            .get_columns()
            .iter()
            .map(|col| {
                positions
                    .iter()
                    .map(|&r| r - first)
                    .filter(|&r| r < df.height())
                    .map(|r| cell_value(col, r))
                    .collect()
            })
            .collect())
    }

    /// The cells at `rows` (every row when `None`) of `cols` as CSV, TSV or
    /// HTML, with a header row of column names.
    pub fn export(
        &self,
        reader: &dyn Reader,
        rows: Option<&Rows>,
        cols: &[usize],
        format: &str,
    ) -> Result<String, String> {
        let (df, picked): (DataFrame, Vec<usize>) = match rows {
            None => {
                let df = self.fetch(reader, cols, None)?;
                let n = df.height();
                (df, (0..n).collect())
            }
            Some(rows) => {
                let positions = rows.positions();
                let first = positions.iter().min().copied().unwrap_or(0);
                let last = positions.iter().max().copied().unwrap_or(0);
                let df = self.fetch(reader, cols, Some((first, last)))?;
                let n = df.height();
                let picked = positions
                    .iter()
                    .map(|r| r - first)
                    .filter(|&r| r < n)
                    .collect();
                (df, picked)
            }
        };
        let cells: Vec<Vec<String>> = picked
            .iter()
            .map(|&r| df.get_columns().iter().map(|c| cell_string(c, r)).collect())
            .collect();
        let header: Vec<String> = cols.iter().map(|&i| self.columns[i].name.clone()).collect();
        export::format(Some(&header), &cells, format)
    }

    /// SQL for `dialect` reproducing this view over the result's `query`.
    pub fn to_sql(
        &self,
        query: &str,
        dialect: &dyn ggsql::reader::SqlDialect,
        caps: Capabilities,
    ) -> Vec<String> {
        // Only the filters the view applies: one the cache rejected would make
        // the SQL fail, or select rows the view does not show.
        let mut applied = Vec::new();
        let mut lines = Vec::new();
        for (i, f) in self.filters.iter().enumerate() {
            match self.rejected.get(i).cloned().flatten() {
                None => applied.push(f.clone()),
                Some(e) => lines.push(format!("-- Filter left out: {e}")),
            }
        }
        lines.extend(filter::to_sql(
            &applied,
            &self.sort,
            &filter::subquery(query),
            &self.columns,
            dialect,
            caps,
        ));
        lines
    }

    /// Profiles of one column over the view's filtered rows.
    pub fn profile(
        &self,
        reader: &dyn Reader,
        index: usize,
        request: &ProfileRequest,
    ) -> Result<Value, String> {
        let col = self
            .columns
            .get(index)
            .ok_or_else(|| format!("There is no column {index}"))?;
        let mut out = json!({});
        let kind = col.kind.as_str();

        if request.null_count {
            let sql = format!(
                "SELECT SUM(CASE WHEN {} IS NULL THEN 1 ELSE 0 END) AS \"n\" FROM {}",
                quote_ident(&col.name),
                self.relation()
            );
            out["null_count"] = json!(scalar_i64(reader, &sql).unwrap_or(0));
        }
        if request.summary {
            // A column the summary cannot handle still gets the rest of its
            // profile, and an empty summary rather than none, which a
            // frontend would wait on.
            out["summary"] = self.summary(reader, col).unwrap_or_else(|e| {
                tracing::warn!("Summarising column {} failed: {}", col.name, e);
                json!({})
            });
        }
        if let Some(h) = &request.histogram {
            if matches!(kind, "integer" | "floating") {
                if let Some(hist) = self.histogram(reader, col, h) {
                    out["histogram"] = hist;
                }
            }
        }
        if let Some(f) = &request.frequency {
            if matches!(kind, "string" | "boolean") {
                if let Some(freq) = self.frequency(reader, col, f.limit) {
                    out["frequency"] = freq;
                }
            }
        }
        Ok(out)
    }

    /// Summary statistics, with fields that depend on the column's kind.
    fn summary(&self, reader: &dyn Reader, col: &Column) -> Result<Value, String> {
        let dialect = reader.dialect();
        let c = quote_ident(&col.name);
        let float = dialect.type_names().number.unwrap_or("DOUBLE PRECISION");
        let parts: Vec<String> = match col.kind.as_str() {
            "integer" | "floating" => vec![
                format!("MIN({c}) AS \"min\""),
                format!("MAX({c}) AS \"max\""),
                format!("AVG(CAST({c} AS {float})) AS \"mean\""),
            ],
            "boolean" => vec![
                format!(
                    "SUM(CASE WHEN {c} = {} THEN 1 ELSE 0 END) AS \"true_count\"",
                    dialect.sql_boolean_literal(true)
                ),
                format!(
                    "SUM(CASE WHEN {c} = {} THEN 1 ELSE 0 END) AS \"false_count\"",
                    dialect.sql_boolean_literal(false)
                ),
            ],
            "string" => vec![
                format!("COUNT(DISTINCT {c}) AS \"num_unique\""),
                format!("SUM(CASE WHEN {c} = '' THEN 1 ELSE 0 END) AS \"num_empty\""),
            ],
            "date" | "datetime" => vec![
                format!("MIN({c}) AS \"min\""),
                format!("MAX({c}) AS \"max\""),
                format!("COUNT(DISTINCT {c}) AS \"num_unique\""),
            ],
            _ => return Ok(json!({})),
        };
        let sql = format!("SELECT {} FROM {}", parts.join(", "), self.relation());
        let df = reader.execute_sql(&sql).map_err(|e| e.to_string())?;
        let get = |name: &str| first_string(&df, name);
        let get_i64 = |name: &str| get(name).and_then(|s| s.parse::<i64>().ok());

        let mut out = json!({});
        match col.kind.as_str() {
            "integer" | "floating" => {
                for name in ["min", "max", "mean"] {
                    if let Some(v) = get(name) {
                        out[name] = json!(v);
                    }
                }
                if let Some(sd) = self.stdev(reader, col) {
                    out["stdev"] = json!(sd);
                }
                if let Some(m) = self.quantile(reader, col, 0.5) {
                    out["median"] = json!(m);
                }
            }
            "boolean" => {
                out["true_count"] = json!(get_i64("true_count").unwrap_or(0));
                out["false_count"] = json!(get_i64("false_count").unwrap_or(0));
            }
            "string" => {
                out["num_unique"] = json!(get_i64("num_unique").unwrap_or(0));
                out["num_empty"] = json!(get_i64("num_empty").unwrap_or(0));
            }
            _ => {
                for name in ["min", "max"] {
                    if let Some(v) = get(name) {
                        out[name] = json!(v);
                    }
                }
                if let Some(n) = get_i64("num_unique") {
                    out["num_unique"] = json!(n);
                }
            }
        }
        Ok(out)
    }

    /// The sample standard deviation of a numeric column, computed by the
    /// database rather than from sums, which lose every digit of a small spread
    /// around large values. Left out when there is none to report: fewer than
    /// two values, or a NaN or infinity among them.
    fn stdev(&self, reader: &dyn Reader, col: &Column) -> Option<String> {
        let float = reader
            .dialect()
            .type_names()
            .number
            .unwrap_or("DOUBLE PRECISION");
        let sql = format!(
            "SELECT STDDEV_SAMP(CAST({} AS {float})) AS \"sd\" FROM {}",
            quote_ident(&col.name),
            self.relation()
        );
        let df = reader.execute_sql(&sql).ok()?;
        let sd = first_string(&df, "sd")?;
        sd.parse::<f64>().ok().filter(|v| v.is_finite())?;
        Some(sd)
    }

    /// The `q` quantile of a column, through the dialect's quantile hook.
    fn quantile(&self, reader: &dyn Reader, col: &Column, q: f64) -> Option<String> {
        // The expression must sit in a query over `from`: native dialects
        // return a bare aggregate, which without a FROM computes over nothing.
        let from = format!("SELECT * FROM {}", self.relation());
        let expr = reader
            .dialect()
            .sql_quantile(&col.name, q, FromItem::Query(&from), &[]);
        let sql = format!("SELECT {expr} AS \"q\" FROM ({from}) AS \"__ggsql_qt__\"");
        let df = reader.execute_sql(&sql).ok()?;
        first_string(&df, "q")
    }

    /// A histogram of a numeric column: Sturges' rule for the bin count,
    /// capped at the request and, for integers, at one bin per value.
    fn histogram(
        &self,
        reader: &dyn Reader,
        col: &Column,
        req: &HistogramRequest,
    ) -> Option<Value> {
        if req.bins == 0 {
            return None;
        }
        let float = reader
            .dialect()
            .type_names()
            .number
            .unwrap_or("DOUBLE PRECISION");
        let c = quote_ident(&col.name);
        let bounds = reader
            .execute_sql(&format!(
                "SELECT MIN(CAST({c} AS {float})) AS \"min\", MAX(CAST({c} AS {float})) AS \"max\", \
                 COUNT({c}) AS \"cnt\" FROM {} WHERE {c} IS NOT NULL",
                self.relation()
            ))
            .ok()?;
        let min: f64 = first_string(&bounds, "min")?.parse().ok()?;
        let max: f64 = first_string(&bounds, "max")?.parse().ok()?;
        let count: usize = first_string(&bounds, "cnt")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let quantiles: Vec<Value> = req
            .quantiles
            .iter()
            .filter_map(|&q| {
                self.quantile(reader, col, q)
                    .map(|v| json!({"q": q, "value": v}))
            })
            .collect();

        if (max - min).abs() < f64::EPSILON {
            return Some(json!({
                "bin_edges": [min.to_string(), max.to_string()],
                "bin_counts": [count],
                "quantiles": quantiles
            }));
        }

        let mut bins = if count > 1 {
            (count as f64).log2().ceil() as usize + 1
        } else {
            1
        };
        if col.kind == "integer" {
            bins = bins.min((max - min) as usize + 1);
        }
        let bins = bins.min(req.bins).max(1);
        let width = (max - min) / bins as f64;

        // The maximum falls in the last bin rather than one past it.
        let sql = format!(
            "SELECT CASE WHEN \"bin\" >= {bins} THEN {last} ELSE \"bin\" END AS \"b\", COUNT(*) AS \"n\" \
             FROM (SELECT FLOOR((CAST({c} AS {float}) - {min}) / {width}) AS \"bin\" \
                   FROM {} WHERE {c} IS NOT NULL) AS \"__bins__\" \
             GROUP BY \"b\" ORDER BY \"b\"",
            self.relation(),
            last = bins - 1,
        );
        let df = reader.execute_sql(&sql).ok()?;
        let mut counts = vec![0i64; bins];
        let (b, n) = (df.column("b").ok()?, df.column("n").ok()?);
        for i in 0..df.height() {
            let bin = ggsql::array_util::value_to_string(b, i)
                .parse::<f64>()
                .ok()
                .map(|v| v as usize);
            if let Some(bin) = bin.filter(|&bin| bin < bins) {
                counts[bin] = ggsql::array_util::value_to_string(n, i)
                    .parse()
                    .unwrap_or(0);
            }
        }
        let edges: Vec<String> = (0..=bins)
            .map(|i| (min + i as f64 * width).to_string())
            .collect();
        Some(json!({ "bin_edges": edges, "bin_counts": counts, "quantiles": quantiles }))
    }

    /// The `limit` most frequent values of a column, and the count of the rest.
    fn frequency(&self, reader: &dyn Reader, col: &Column, limit: usize) -> Option<Value> {
        let c = quote_ident(&col.name);
        let df = reader
            .execute_sql(&format!(
                "SELECT {c} AS \"value\", COUNT(*) AS \"count\" FROM {} WHERE {c} IS NOT NULL \
                 GROUP BY {c} ORDER BY COUNT(*) DESC, {c} LIMIT {limit}",
                self.relation()
            ))
            .ok()?;
        let (v, n) = (df.column("value").ok()?, df.column("count").ok()?);
        let values: Vec<String> = (0..df.height()).map(|i| cell_string(v, i)).collect();
        let counts: Vec<i64> = (0..df.height())
            .map(|i| {
                ggsql::array_util::value_to_string(n, i)
                    .parse()
                    .unwrap_or(0)
            })
            .collect();
        let total = scalar_i64(
            reader,
            &format!("SELECT COUNT({c}) AS \"n\" FROM {}", self.relation()),
        )
        .unwrap_or(0);
        let other = total - counts.iter().sum::<i64>();
        Some(json!({ "values": values, "counts": counts, "other_count": other }))
    }
}

/// `SELECT COUNT(*)` over `table`, filtered by `predicate` when given.
pub fn count_rows(
    reader: &dyn Reader,
    table: &str,
    predicate: Option<&str>,
) -> Result<usize, String> {
    let filter = predicate.map(|p| format!(" WHERE {p}")).unwrap_or_default();
    let n = scalar_i64(
        reader,
        &format!("SELECT COUNT(*) AS \"n\" FROM {table}{filter}"),
    )?;
    Ok(n.max(0) as usize)
}

/// The first column of a one-row query as an integer.
fn scalar_i64(reader: &dyn Reader, sql: &str) -> Result<i64, String> {
    let df = reader.execute_sql(sql).map_err(|e| e.to_string())?;
    let name = df
        .get_column_names()
        .into_iter()
        .next()
        .ok_or("The query returned no columns")?;
    Ok(first_string(&df, &name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0))
}

/// Row 0 of `name` as text, or `None` when absent or null.
fn first_string(df: &DataFrame, name: &str) -> Option<String> {
    use arrow::array::Array;
    let c = df.column(name).ok()?;
    if c.is_empty() || c.is_null(0) {
        None
    } else {
        Some(cell_string(c, 0))
    }
}

/// A cell as text, with a null as empty.
fn cell_string(col: &arrow::array::ArrayRef, row: usize) -> String {
    use arrow::array::Array;
    if col.is_null(row) {
        return String::new();
    }
    crate::display::cell_text(col, row)
}

/// A cell for a page: its text, or `null`.
fn cell_value(col: &arrow::array::ArrayRef, row: usize) -> Value {
    use arrow::array::Array;
    if col.is_null(row) {
        Value::Null
    } else {
        Value::String(cell_string(col, row))
    }
}
