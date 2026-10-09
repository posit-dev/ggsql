//! Test doubles and assertion helpers shared across reader tests.
//!
//! Extracted from `reader/mod.rs` in the Phase 5 cleanup; module path is
//! unchanged (`crate::reader::test_support`).

use super::{execute_with_reader, returns_rows, ColumnInfo, Reader, Spec, SqlDialect, TableInfo};
use crate::{DataFrame, GgsqlError, Result};

/// A reader that can serve as an in-memory, writable caching backend.
///
/// Only exercised by cache tests today — production cache backends are
/// built from URIs via [`crate::reader::connection`].
pub(crate) trait CacheBackend: Reader {
    fn new_in_memory() -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "duckdb")]
impl CacheBackend for super::DuckDBReader {
    fn new_in_memory() -> Result<Self> {
        Self::from_connection_string("duckdb://memory")
    }
}

#[cfg(feature = "sqlite")]
impl CacheBackend for super::SqliteReader {
    fn new_in_memory() -> Result<Self> {
        Self::new()
    }
}
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A `Reader` that records every `execute_sql` it receives, delegating
/// everything to an inner reader.
pub(crate) struct SpyReader {
    inner: Box<dyn Reader + Send>,
    log: Arc<Mutex<Vec<String>>>,
}

impl SpyReader {
    /// Wrap `inner`, returning the boxed spy and a handle to its call log.
    pub(crate) fn wrap(
        inner: Box<dyn Reader + Send>,
    ) -> (Box<dyn Reader + Send>, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            Box::new(SpyReader {
                inner,
                log: log.clone(),
            }),
            log,
        )
    }
}

impl Reader for SpyReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        self.log.lock().unwrap().push(sql.to_string());
        self.inner.execute_sql(sql)
    }
    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.inner.register(name, df, replace)
    }
    fn unregister(&self, name: &str) -> Result<()> {
        self.inner.unregister(name)
    }
    fn execute(&self, query: &str) -> Result<Spec> {
        execute_with_reader(self, query)
    }
    fn dialect(&self) -> &dyn SqlDialect {
        self.inner.dialect()
    }
    fn list_catalogs(&self) -> Result<Vec<String>> {
        self.inner.list_catalogs()
    }
    fn list_schemas(&self, c: &str) -> Result<Vec<String>> {
        self.inner.list_schemas(c)
    }
    fn list_tables(&self, c: &str, s: &str) -> Result<Vec<TableInfo>> {
        self.inner.list_tables(c, s)
    }
    fn list_columns(&self, c: &str, s: &str, t: &str) -> Result<Vec<ColumnInfo>> {
        self.inner.list_columns(c, s, t)
    }
}

/// A `Reader` that wraps an inner reader and **refuses every write**: any
/// `register`/`unregister` and any non-row-returning `execute_sql`
/// (CREATE/INSERT/DROP/…) returns an error.
pub(crate) struct ReadOnlyReader {
    inner: Box<dyn Reader + Send>,
}

impl ReadOnlyReader {
    pub(crate) fn new(inner: Box<dyn Reader + Send>) -> Self {
        Self { inner }
    }

    fn refuse(op: &str) -> GgsqlError {
        GgsqlError::ReaderError(format!("read-only primary: refused {op}"))
    }
}

impl Reader for ReadOnlyReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        if !returns_rows(sql) {
            return Err(Self::refuse(&format!("write statement: {sql}")));
        }
        self.inner.execute_sql(sql)
    }
    fn register(&self, _name: &str, _df: DataFrame, _replace: bool) -> Result<()> {
        Err(Self::refuse("register"))
    }
    fn unregister(&self, _name: &str) -> Result<()> {
        Err(Self::refuse("unregister"))
    }
    fn execute(&self, query: &str) -> Result<Spec> {
        execute_with_reader(self, query)
    }
    fn dialect(&self) -> &dyn SqlDialect {
        self.inner.dialect()
    }
    fn list_catalogs(&self) -> Result<Vec<String>> {
        self.inner.list_catalogs()
    }
    fn list_schemas(&self, c: &str) -> Result<Vec<String>> {
        self.inner.list_schemas(c)
    }
    fn list_tables(&self, c: &str, s: &str) -> Result<Vec<TableInfo>> {
        self.inner.list_tables(c, s)
    }
    fn list_columns(&self, c: &str, s: &str, t: &str) -> Result<Vec<ColumnInfo>> {
        self.inner.list_columns(c, s, t)
    }
}

/// A `Reader` that never connects anywhere: it records every SQL
/// statement it is asked to run and fabricates an empty result whose
/// schema is inferred from the statement's top-level SELECT list.
///
/// Combined with a per-test dialect this lets the golden SQL tests
/// capture exactly what the plot pipeline emits for each backend
/// without needing a live server. Column types are recovered from
/// registered tables by name; anything unrecognized defaults to
/// `Float64`, which is the right shape for the derived columns
/// (bins, densities, quantiles) the pipeline produces.
pub(crate) struct StubReader {
    dialect: Box<dyn SqlDialect + Send>,
    tables: Mutex<HashMap<String, Arc<Schema>>>,
    log: Arc<Mutex<Vec<String>>>,
}

impl StubReader {
    /// Create a stub for `dialect`, returning it together with a
    /// handle to the shared SQL log.
    pub(crate) fn new(dialect: Box<dyn SqlDialect + Send>) -> (Self, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                dialect,
                tables: Mutex::new(HashMap::new()),
                log: log.clone(),
            },
            log,
        )
    }

    /// Fabricate a three-row result for a row-returning statement.
    ///
    /// Three rows (rather than zero) so that data-dependent stages —
    /// scale training, histogram statistics — have values to work
    /// with, and min/max ranges are non-degenerate.
    fn fake_result(&self, sql: &str) -> DataFrame {
        let cols = self.projected_columns(sql);
        if cols.is_empty() {
            return DataFrame::empty();
        }
        let fields: Vec<Field> = cols
            .iter()
            .map(|(name, ty)| Field::new(name, ty.clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<arrow::array::ArrayRef> = schema
            .fields()
            .iter()
            .map(|f| {
                let arr = sample_array(f.data_type());
                // The single-row minmax query reads each column's MAX from
                // a paired `__ggsql_max_*` column. Rotating the sample
                // gives it the larger values the old two-row shape supplied
                // via its second row — an equal min/max would collapse the
                // trained range and silently empty the binned cases' SQL.
                if f.name().starts_with("__ggsql_max_") {
                    let idx = arrow::array::UInt32Array::from(vec![1u32, 2, 0]);
                    arrow::compute::take(arr.as_ref(), &idx, None).expect("sample rotate")
                } else {
                    arr
                }
            })
            .collect();
        let batch =
            RecordBatch::try_new(schema, arrays).expect("stub arrays must match fabricated schema");
        DataFrame::from_record_batch(batch)
    }

    /// Names and types of the columns a row-returning statement would
    /// produce, inferred well enough for the pipeline's generated SQL:
    ///
    /// - `AS` aliases and plain identifiers in the outermost SELECT
    ///   list (after stripping `DISTINCT` / `TOP n`),
    /// - bare `*`, expanded from every registered table the statement
    ///   references (the pipeline's subqueries carry base columns
    ///   through, so this matches even when the FROM is a subquery),
    /// - internal `"__ggsql_*"` identifiers mentioned anywhere in the
    ///   statement (aesthetic/stat channel columns), typed via the
    ///   expression that defines them.
    ///
    /// Column types come from the registered table whose column appears
    /// in the defining expression; anything unrecognized defaults to
    /// `Float64`, the right shape for derived numeric columns (bins,
    /// densities, counts, quantiles).
    fn projected_columns(&self, sql: &str) -> Vec<(String, DataType)> {
        let mut cols: Vec<(String, DataType)> = Vec::new();
        let push = |name: String, ty: DataType, cols: &mut Vec<(String, DataType)>| {
            if !cols.iter().any(|(n, _)| n == &name) {
                cols.push((name, ty));
            }
        };

        let mut has_star = false;
        if let Some((list, _)) = select_list_span(sql) {
            for item in split_top_level_commas(&list) {
                let item = item.trim();
                if item == "*" {
                    has_star = true;
                } else if let Some(name) = output_name(item) {
                    let ty = self
                        .registered_type(&name)
                        .or_else(|| self.type_from_expr(item))
                        .unwrap_or(DataType::Float64);
                    push(name, ty, &mut cols);
                }
            }
        }

        if has_star {
            let tables = self.tables.lock().unwrap();
            for (table, schema) in tables.iter() {
                if contains_word(sql, table) {
                    for field in schema.fields() {
                        push(field.name().clone(), field.data_type().clone(), &mut cols);
                    }
                }
            }
        }

        for name in scan_internal_idents(sql) {
            // Temp-table names are internal identifiers too, but they
            // name tables, not columns: without this filter a probe
            // against a temp table would invent a column named after
            // the table itself.
            if self.tables.lock().unwrap().contains_key(&name) {
                continue;
            }
            let ty = self
                .type_from_alias_definition(sql, &name)
                .unwrap_or(DataType::Float64);
            push(name, ty, &mut cols);
        }

        cols
    }

    /// Type of a registered table column with this exact
    /// (case-insensitive) name.
    fn registered_type(&self, name: &str) -> Option<DataType> {
        let tables = self.tables.lock().unwrap();
        let find = |schema: &Arc<Schema>| {
            schema.fields().iter().find_map(|f| {
                if f.name().eq_ignore_ascii_case(name) {
                    Some(f.data_type().clone())
                } else {
                    None
                }
            })
        };
        // Prefer real source tables over internal temp aliases: an alias
        // registered under a "__ggsql_" name may carry a *stale* schema
        // (e.g. from an earlier fixture), and HashMap iteration order
        // would otherwise decide which schema wins — the same
        // determinism fix as track_ddl's source preference.
        tables
            .iter()
            .filter(|(n, _)| !n.starts_with("__ggsql_"))
            .find_map(|(_, s)| find(s))
            .or_else(|| tables.values().find_map(find))
    }

    /// Type of the longest registered column name appearing as a word
    /// in `expr` (e.g. `CAST("day" AS DATE)` → the type of `day`).
    fn type_from_expr(&self, expr: &str) -> Option<DataType> {
        let tables = self.tables.lock().unwrap();
        let mut best: Option<DataType> = None;
        let mut best_len = 0;
        let mut best_is_source = false;
        for (table, schema) in tables.iter() {
            // See registered_type: real source tables beat internal
            // aliases on ties (equal-length column names), which is what
            // keeps the result independent of HashMap iteration order.
            let is_source = !table.starts_with("__ggsql_");
            for field in schema.fields() {
                let better = field.name().len() > best_len
                    || (field.name().len() == best_len && is_source && !best_is_source);
                if better && contains_word(expr, field.name()) {
                    best = Some(field.data_type().clone());
                    best_len = field.name().len();
                    best_is_source = is_source;
                }
            }
        }
        best
    }

    /// Track temp-table lineage for non-row-returning statements:
    /// a statement that mentions an internal `"__ggsql_*"` name and a
    /// registered source table (`CREATE ... AS SELECT * FROM t`,
    /// `SELECT * INTO ... FROM t`) aliases the temp name to the
    /// source schema; `DROP` removes it. Later probes against the
    /// temp copy then see the source columns.
    fn track_ddl(&self, sql: &str) {
        let idents = scan_internal_idents(sql);
        if idents.is_empty() {
            return;
        }
        let mut tables = self.tables.lock().unwrap();
        if sql
            .split_whitespace()
            .next()
            .unwrap_or("")
            .eq_ignore_ascii_case("drop")
        {
            for name in idents {
                tables.remove(&name);
            }
            return;
        }
        // Prefer real source tables over stale internal aliases when
        // resolving lineage (the statement names the temp table it
        // creates too, and HashMap iteration order is arbitrary).
        let source = tables
            .iter()
            .filter(|(name, _)| !name.starts_with("__ggsql_"))
            .find(|(name, _)| contains_word(sql, name))
            .or_else(|| tables.iter().find(|(name, _)| contains_word(sql, name)))
            .map(|(_, schema)| schema.clone());
        if let Some(schema) = source {
            for name in idents {
                tables.insert(name, schema.clone());
            }
        }
    }

    /// Find `<expr> AS <name>` in the statement and infer the type
    /// from `expr`. Used for internal channel columns whose defining
    /// expression wraps a registered column (`"category" AS
    /// "__ggsql_aes_pos1__"`, `toDate32("day") AS ...`).
    fn type_from_alias_definition(&self, sql: &str, name: &str) -> Option<DataType> {
        let quoted = format!("\"{name}\"");
        let mut search_from = 0;
        while let Some(rel) = sql[search_from..].find(&quoted) {
            let pos = search_from + rel;
            let before = sql[..pos].trim_end();
            if before.len() >= 2 && before[before.len() - 2..].eq_ignore_ascii_case("as") {
                let expr_start = before[..before.len() - 2]
                    .rfind(['(', ','])
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let expr = &before[expr_start..before.len() - 2];
                if let Some(ty) = self.type_from_expr(expr) {
                    return Some(ty);
                }
            }
            search_from = pos + quoted.len();
        }
        None
    }
}

impl Reader for StubReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        self.log.lock().unwrap().push(sql.to_string());
        // MSSQL's `SELECT ... INTO <temp> FROM ...` starts with SELECT
        // but is DDL — treat it as a write so lineage is tracked.
        let mut is_select_into = false;
        for_each_top_level_keyword(sql, "into", |_| is_select_into = true);
        if returns_rows(sql) && !is_select_into {
            Ok(self.fake_result(sql))
        } else {
            self.track_ddl(sql);
            Ok(DataFrame::empty())
        }
    }
    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        let mut tables = self.tables.lock().unwrap();
        if replace || !tables.contains_key(name) {
            tables.insert(name.to_string(), df.schema());
        }
        Ok(())
    }
    fn unregister(&self, name: &str) -> Result<()> {
        self.tables.lock().unwrap().remove(name);
        Ok(())
    }
    fn execute(&self, query: &str) -> Result<Spec> {
        execute_with_reader(self, query)
    }
    fn dialect(&self) -> &dyn SqlDialect {
        &*self.dialect
    }
}

// ------------------------------------------------------------------------
// Minimal top-level SQL scanning helpers for `StubReader`
// ------------------------------------------------------------------------

/// Strip one layer of identifier quoting: `"x"`, `` `x` ``, or `[x]`.
fn unquote_ident(s: &str) -> String {
    let t = s.trim();
    for (open, close) in [('"', '"'), ('`', '`'), ('[', ']')] {
        if t.len() >= 2 && t.starts_with(open) && t.ends_with(close) {
            return t[1..t.len() - 1].to_string();
        }
    }
    t.to_string()
}

/// Scan `sql`, calling `f` for each top-level keyword occurrence.
///
/// Tracks parenthesis depth and single/double/backtick quoting, so
/// keywords inside subqueries, string literals, and quoted identifiers
/// are skipped. `f` receives the keyword's byte start.
fn for_each_top_level_keyword(sql: &str, keyword: &str, mut f: impl FnMut(usize)) {
    let bytes = sql.as_bytes();
    let kw = keyword.as_bytes();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            if c == q {
                // SQL escapes a quote by doubling it; skip the pair.
                if i + 1 < bytes.len() && bytes[i + 1] == q {
                    i += 2;
                    continue;
                }
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' | b'`' => quote = Some(c),
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {
                if depth == 0
                    && i + kw.len() <= bytes.len()
                    && bytes[i..i + kw.len()].eq_ignore_ascii_case(kw)
                    && (i == 0 || !is_ident_byte(bytes[i - 1]))
                    && (i + kw.len() >= bytes.len() || !is_ident_byte(bytes[i + kw.len()]))
                {
                    f(i);
                }
            }
        }
        i += 1;
    }
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Locate the outermost SELECT list, returning the list text and the
/// bare table name following the top-level FROM, if any.
fn select_list_span(sql: &str) -> Option<(String, Option<String>)> {
    let mut sel = None;
    for_each_top_level_keyword(sql, "select", |pos| {
        if sel.is_none() {
            sel = Some(pos);
        }
    });
    let sel = sel?;
    let mut from = None;
    for_each_top_level_keyword(sql, "from", |pos| {
        if pos > sel && from.is_none() {
            from = Some(pos);
        }
    });
    let list_start = sel + "select".len();
    let list = match from {
        Some(f) => sql[list_start..f].trim().to_string(),
        None => sql[list_start..].trim().to_string(),
    };
    let list = strip_select_modifiers(list);
    let table = from.and_then(|f| {
        let rest = sql[f + "from".len()..].trim_start();
        let end = rest
            .find(|c: char| c.is_whitespace() || c == '(' || c == ',')
            .unwrap_or(rest.len());
        let name = rest[..end].trim();
        if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        }
    });
    Some((list, table))
}

/// Does `needle` appear in `haystack` bounded by non-identifier bytes
/// (so `cat` doesn't match `category`)? Case-insensitive.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let hay = haystack.as_bytes();
    let nee = needle.as_bytes();
    if nee.is_empty() || hay.len() < nee.len() {
        return false;
    }
    (0..=hay.len() - nee.len()).any(|i| {
        hay[i..i + nee.len()].eq_ignore_ascii_case(nee)
            && (i == 0 || !is_ident_byte(hay[i - 1]))
            && (i + nee.len() >= hay.len() || !is_ident_byte(hay[i + nee.len()]))
    })
}

/// All internal `"__ggsql_*"` identifiers mentioned in the statement,
/// in either double-quote or backtick quoting. These name channel
/// columns (`__ggsql_aes_pos1__`, `__ggsql_stat_count`, …) that the
/// pipeline carries through subqueries and later looks up by name.
fn scan_internal_idents(sql: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if (bytes[i] == b'"' || bytes[i] == b'`') && sql[i + 1..].starts_with("__ggsql_") {
            let quote = bytes[i];
            let start = i + 1;
            if let Some(end) = sql[start..].find(quote as char) {
                let name = &sql[start..start + end];
                if name.bytes().all(is_ident_byte) && !out.iter().any(|n| n == name) {
                    out.push(name.to_string());
                }
                i = start + end + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// A deterministic three-row array for a fabricated column. Types the
/// pipeline doesn't map to a registered table column fall back to
/// all-null via `new_null_array`.
fn sample_array(dtype: &DataType) -> arrow::array::ArrayRef {
    use arrow::array::{
        BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };
    match dtype {
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        DataType::Int32 => Arc::new(Int32Array::from(vec![1, 2, 3])),
        DataType::Int64 => Arc::new(Int64Array::from(vec![1i64, 2, 3])),
        // ISO-date strings: fabricated values flow through scale training,
        // which coerces text to the scale's target type. A real backend
        // holding a text-typed date column returns parseable strings (that
        // is exactly the text_date_cast battery case's premise), so the
        // stub does too — an unparseable value would pin a coercion error
        // instead of the SQL the case exists to pin. Equally valid as
        // plain strings for non-temporal columns.
        DataType::Utf8 => Arc::new(StringArray::from(vec![
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
        ])),
        DataType::Boolean => Arc::new(BooleanArray::from(vec![true, false, true])),
        DataType::Date32 => Arc::new(Date32Array::from(vec![19000, 19001, 19002])),
        other => arrow::array::new_null_array(other, 3),
    }
}

/// Drop leading SELECT-list modifiers the schema probe may add:
/// `DISTINCT` and MSSQL's `TOP n`.
fn strip_select_modifiers(list: String) -> String {
    let mut rest = list.trim_start().to_string();
    loop {
        let lower = rest.to_ascii_lowercase();
        if let Some(after) = lower.strip_prefix("distinct") {
            if after.starts_with(|c: char| c.is_whitespace()) {
                rest = rest["distinct".len()..].trim_start().to_string();
                continue;
            }
        }
        if let Some(after) = lower.strip_prefix("top") {
            let after = after.trim_start();
            let digits: usize = after.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits > 0 {
                let ws: usize = rest.len() - rest.trim_start().len();
                rest = rest[ws + 3..]
                    .trim_start()
                    .chars()
                    .skip(digits)
                    .collect::<String>()
                    .trim_start()
                    .to_string();
                continue;
            }
        }
        return rest;
    }
}

/// Split a SELECT list on top-level commas.
fn split_top_level_commas(list: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut start = 0;
    let bytes = list.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            if c == q {
                if i + 1 < bytes.len() && bytes[i + 1] == q {
                    i += 2;
                    continue;
                }
                quote = None;
            }
        } else {
            match c {
                b'\'' | b'"' | b'`' => quote = Some(c),
                b'(' => depth += 1,
                b')' => depth -= 1,
                b',' if depth == 0 => {
                    parts.push(list[start..i].to_string());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    parts.push(list[start..].to_string());
    parts
}

/// Output column name for one SELECT-list item: the `AS` alias if
/// present, else the final segment of a plain (possibly qualified)
/// identifier, else `None` for unaliased expressions.
fn output_name(item: &str) -> Option<String> {
    let mut alias = None;
    for_each_top_level_keyword(item, "as", |pos| alias = Some(pos));
    if let Some(pos) = alias {
        let name = item[pos + 2..].trim();
        if !name.is_empty() {
            return Some(unquote_ident(name));
        }
    }
    let t = item.trim();
    if !t.is_empty()
        && t.bytes()
            .all(|b| is_ident_byte(b) || b == b'.' || b == b'"' || b == b'`')
    {
        let last = t.rsplit('.').next().unwrap_or(t);
        return Some(unquote_ident(last));
    }
    None
}

/// Compare two DataFrames by schema (field names + types) and by
/// per-column Arrow array contents. We don't use a blanket
/// `assert_eq!(df, df)` because `DataFrame` doesn't implement `PartialEq`;
/// going through schema + per-column equality is also more diagnostic
/// when one of them diverges.
#[cfg(feature = "adbc")]
pub(crate) fn assert_dataframes_equal(a: &DataFrame, b: &DataFrame, ctx: &str) {
    let a_schema = a.schema();
    let b_schema = b.schema();
    assert_eq!(
        a_schema.fields().len(),
        b_schema.fields().len(),
        "{ctx}: column count mismatch (a={}, b={})",
        a_schema.fields().len(),
        b_schema.fields().len(),
    );
    for (i, (af, bf)) in a_schema
        .fields()
        .iter()
        .zip(b_schema.fields().iter())
        .enumerate()
    {
        assert_eq!(
            af.name(),
            bf.name(),
            "{ctx}: column {i} name mismatch (a='{}', b='{}')",
            af.name(),
            bf.name(),
        );
        assert_eq!(
            af.data_type(),
            bf.data_type(),
            "{ctx}: column '{}' type mismatch (a={:?}, b={:?})",
            af.name(),
            af.data_type(),
            bf.data_type(),
        );
    }
    assert_eq!(
        a.height(),
        b.height(),
        "{ctx}: row count mismatch (a={}, b={})",
        a.height(),
        b.height(),
    );
    for field in a_schema.fields() {
        let ac = a.column(field.name()).unwrap();
        let bc = b.column(field.name()).unwrap();
        assert_eq!(
            ac.as_ref(),
            bc.as_ref(),
            "{ctx}: column '{}' data mismatch",
            field.name(),
        );
    }
}
