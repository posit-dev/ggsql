//! Data source abstraction layer for ggsql
//!
//! The reader module provides a pluggable interface for executing SQL queries
//! against various data sources and returning Polars DataFrames for visualization.
//!
//! # Architecture
//!
//! All readers implement the `Reader` trait, which provides:
//! - SQL query execution → DataFrame conversion
//! - Visualization query execution → Spec
//! - Optional DataFrame registration for queryable tables
//! - Connection management and error handling
//!
//! # Example
//!
//! ```rust,ignore
//! use ggsql::reader::{Reader, DuckDBReader};
//! use ggsql::writer::{Writer, VegaLiteWriter};
//!
//! // Execute a ggsql query
//! let reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
//!
//! // Render to Vega-Lite JSON
//! let writer = VegaLiteWriter::new();
//! let json = writer.render(&spec)?;
//!
//! // With DataFrame registration
//! let mut reader = DuckDBReader::from_connection_string("duckdb://memory")?;
//! reader.register("my_table", some_dataframe, false)?;
//! let spec = reader.execute("SELECT * FROM my_table VISUALISE x, y DRAW point")?;
//! ```

use std::collections::HashMap;

use crate::execute::prepare_data_with_reader;
use crate::plot::{CastTargetType, Plot};
use crate::validate::{validate, ValidationWarning};
use crate::{naming, DataFrame, GgsqlError, Result};

// =============================================================================
// SQL Dialect
// =============================================================================

/// SQL type names and functionality in the syntax supported by that backend.
///
/// Default implementations produce portable ANSI SQL.
pub trait SqlDialect {
    /// SQL type name for numeric columns (e.g., "DOUBLE PRECISION")
    fn number_type_name(&self) -> Option<&str> {
        Some("DOUBLE PRECISION")
    }

    /// SQL type name for integer columns (e.g., "BIGINT")
    fn integer_type_name(&self) -> Option<&str> {
        Some("BIGINT")
    }

    /// SQL type name for DATE columns (e.g., "DATE")
    fn date_type_name(&self) -> Option<&str> {
        Some("DATE")
    }

    /// SQL type name for DATETIME/TIMESTAMP columns
    fn datetime_type_name(&self) -> Option<&str> {
        Some("TIMESTAMP")
    }

    /// SQL type name for TIME columns
    fn time_type_name(&self) -> Option<&str> {
        Some("TIME")
    }

    /// SQL type name for STRING/VARCHAR columns
    fn string_type_name(&self) -> Option<&str> {
        Some("VARCHAR")
    }

    /// SQL type name for BOOLEAN columns
    fn boolean_type_name(&self) -> Option<&str> {
        Some("BOOLEAN")
    }

    /// Get the SQL type name for a cast target type.
    fn type_name_for(&self, target: CastTargetType) -> Option<&str> {
        match target {
            CastTargetType::Number => self.number_type_name(),
            CastTargetType::Integer => self.integer_type_name(),
            CastTargetType::Date => self.date_type_name(),
            CastTargetType::DateTime => self.datetime_type_name(),
            CastTargetType::Time => self.time_type_name(),
            CastTargetType::String => self.string_type_name(),
            CastTargetType::Boolean => self.boolean_type_name(),
        }
    }

    /// Quote an identifier using this backend's convention.
    ///
    /// Default is SQL-standard double quotes. Override for backends with a
    /// different quoting convention (e.g. backticks for MySQL/ClickHouse).
    fn quote_ident(&self, name: &str) -> String {
        naming::quote_ident(name)
    }

    /// Statements to execute once on a freshly opened connection, before any
    /// other SQL. Default: none.
    ///
    /// Use for session settings the generated SQL silently relies on — e.g.
    /// MySQL's `ANSI_QUOTES`, because ggsql quotes many internal identifiers
    /// (`__ggsql_*` CTEs and columns) with ANSI double quotes, which MySQL
    /// otherwise parses as string literals.
    fn session_init_sql(&self) -> Vec<String> {
        Vec::new()
    }

    /// Append a row limit to a query.
    ///
    /// Default uses `LIMIT n`. Override for backends with different limit
    /// syntax (e.g. SQL Server's `TOP`, Oracle's `FETCH FIRST`).
    fn sql_limit(&self, query: &str, n: usize) -> String {
        format!("{} LIMIT {}", query, n)
    }

    /// Cast an expression to a SQL type name.
    ///
    /// Default uses `CAST(expr AS type)`. Override for backends that prefer a
    /// non-throwing cast (e.g. BigQuery's `SAFE_CAST`, Snowflake/Trino's
    /// `TRY_CAST`). `type_name` should come from [`type_name_for`] so it is
    /// already backend-appropriate.
    ///
    /// [`type_name_for`]: SqlDialect::type_name_for
    fn sql_cast(&self, expr: &str, type_name: &str) -> String {
        format!("CAST({} AS {})", expr, type_name)
    }

    /// Whether this backend supports spatial (geometry) operations.
    ///
    /// When `false`, the executor should fail fast with a clear
    /// "spatial not supported on this backend" error rather than emitting
    /// spatial SQL that the backend cannot run. Default is `true` because the
    /// ANSI defaults target PostGIS-compatible backends.
    fn supports_spatial(&self) -> bool {
        true
    }

    /// Scalar MAX across any number of SQL expressions.
    fn sql_greatest(&self, exprs: &[&str]) -> String {
        let mut result = exprs[0].to_string();
        for expr in &exprs[1..] {
            result =
                format!("(CASE WHEN ({result}) >= ({expr}) THEN ({result}) ELSE ({expr}) END)");
        }
        result
    }

    /// Scalar MIN across any number of SQL expressions.
    fn sql_least(&self, exprs: &[&str]) -> String {
        let mut result = exprs[0].to_string();
        for expr in &exprs[1..] {
            result =
                format!("(CASE WHEN ({result}) <= ({expr}) THEN ({result}) ELSE ({expr}) END)");
        }
        result
    }

    /// SQL expression to convert a geometry column to WKB.
    ///
    /// Default uses `ST_AsBinary` (OGC standard). Override for backends
    /// with different function names (e.g. DuckDB uses `ST_AsWKB`).
    fn sql_geometry_to_wkb(&self, column: &str) -> String {
        format!("ST_AsBinary({column})")
    }

    /// WORKAROUND(duckdb-rs#714): Ensures a column is native GEOMETRY type.
    ///
    /// Geometry columns may arrive as WKB BLOB (because Arrow export crashes on
    /// native GEOMETRY, forcing pre-conversion). This normalizes both GEOMETRY
    /// and BLOB to GEOMETRY so spatial functions work uniformly.
    ///
    /// Default is identity (column is already geometry). Override for backends
    /// where geometry arrives as a different type.
    fn sql_ensure_geometry(&self, column: &str) -> String {
        column.to_string()
    }

    /// Produce a SELECT that replaces a single column with a new expression.
    ///
    /// When `all_columns` is provided, enumerates them explicitly (substituting
    /// `expr` for `col`), avoiding duplicate-column issues on PostgreSQL.
    /// When `all_columns` is empty, falls back to `SELECT expr AS col, *` which
    /// works for direct-to-DataFrame execution (first occurrence wins) but NOT
    /// for `CREATE TABLE AS` on PostgreSQL.
    ///
    /// Override for backends with native REPLACE syntax (DuckDB).
    fn sql_select_replace(
        &self,
        expr: &str,
        col: &str,
        from: &str,
        all_columns: &[String],
    ) -> String {
        let __ggsql_sr__ = self.quote_ident("__ggsql_sr__");
        if expr == col {
            // Explicit alias: MySQL/MariaDB reject unaliased derived tables.
            return format!("SELECT * FROM ({from}) {__ggsql_sr__}");
        }
        if all_columns.is_empty() {
            return format!("SELECT {expr} AS {col}, * FROM ({from}) {__ggsql_sr__}");
        }
        let select_list: Vec<String> = all_columns
            .iter()
            .map(|c| {
                let qc = self.quote_ident(c);
                if qc == col {
                    format!("{expr} AS {col}")
                } else {
                    qc
                }
            })
            .collect();
        format!(
            "SELECT {} FROM ({from}) {__ggsql_sr__}",
            select_list.join(", ")
        )
    }

    /// SQL expression to transform a geometry from one CRS to another.
    ///
    /// Default uses the PostGIS-compatible pattern: set the source SRID on the
    /// geometry, then transform to either a numeric SRID or a PROJ string.
    fn sql_st_transform(&self, column: &str, source_crs: &str, target_crs: &str) -> String {
        let source_srid = extract_epsg_srid(source_crs).unwrap_or(4326);
        let target = match extract_epsg_srid(target_crs) {
            Some(srid) => format!("{}", srid),
            None => format!("'{}'", target_crs.replace('\'', "''")),
        };
        format!(
            "ST_Transform(ST_SetSRID({}, {}), {})",
            column, source_srid, target
        )
    }

    /// SQL query that computes the bounding box of a geometry column.
    ///
    /// Must return a single row with columns `xmin`, `ymin`, `xmax`, `ymax` (DOUBLE).
    /// `from` is the table or subquery to aggregate over.
    fn sql_geometry_bbox(&self, column: &str, from: &str) -> String {
        // The derived table must be aliased: Postgres (< 16) and
        // MySQL/MariaDB reject unaliased derived tables.
        format!(
            "SELECT ST_XMin(ext) AS xmin, ST_YMin(ext) AS ymin, \
                    ST_XMax(ext) AS xmax, ST_YMax(ext) AS ymax \
             FROM (SELECT ST_Extent({column}) AS ext FROM {from}) AS __ggsql_ext__"
        )
    }

    /// SQL expression building a rectangular polygon from corner coordinates.
    ///
    /// Default uses the PostGIS-style `ST_MakeEnvelope`. Override for backends
    /// with different function names (e.g. SpatiaLite uses `BuildMbr`).
    fn sql_make_envelope(&self, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> String {
        format!("ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})")
    }

    /// SQL statements to run before spatial operations.
    ///
    /// Override for backends that need an extension loaded (e.g. DuckDB spatial).
    fn sql_spatial_setup(&self) -> Vec<String> {
        vec![]
    }

    /// Generate a series of integers 0..n-1 as a CTE fragment.
    ///
    /// Target type name for casting an expression to a floating-point type.
    ///
    /// `REAL` is widely supported (Postgres, SQLite, T-SQL, DuckDB, ...).
    /// MariaDB's `CAST` has no `REAL` target; MySQL/MariaDB override this
    /// with `DOUBLE` (both support it).
    fn sql_real_cast_type(&self) -> &'static str {
        "REAL"
    }

    /// Returns CTE fragment(s) producing table `__ggsql_seq__` with column `n`.
    fn sql_generate_series(&self, n: usize) -> String {
        // Uses a cube-root decomposition to avoid deep recursion: only recurses
        // ~cbrt(n) times, then cross-joins three copies to cover the full range.
        let base_size = (n as f64).cbrt().ceil() as usize;
        let base_sq = base_size * base_size;
        let base_max = base_size - 1;
        let __ggsql_base__ = self.quote_ident("__ggsql_base__");
        let __ggsql_seq__ = self.quote_ident("__ggsql_seq__");
        let real = self.sql_real_cast_type();
        format!(
            "{__ggsql_base__}(n) AS (\
               SELECT 0 UNION ALL SELECT n + 1 FROM {__ggsql_base__} WHERE n < {base_max}\
             ),\
             {__ggsql_seq__}(n) AS (\
               SELECT CAST(a.n * {base_sq} + b.n * {base_size} + c.n AS {real}) AS n \
               FROM {__ggsql_base__} a, {__ggsql_base__} b, {__ggsql_base__} c \
               WHERE a.n * {base_sq} + b.n * {base_size} + c.n < {n}\
             )"
        )
    }

    /// Keyword introducing a CTE block that contains recursive CTEs.
    ///
    /// ANSI/Postgres/MySQL accept `WITH RECURSIVE`; T-SQL and Oracle use
    /// plain `WITH`, where recursion is implied by self-reference.
    fn sql_with_recursive(&self) -> &'static str {
        "WITH RECURSIVE"
    }

    /// An `ORDER BY` clause for a query nested in a derived table or CTE.
    ///
    /// Default emits `ORDER BY <ordering>`. SQL Server forbids ORDER BY in
    /// views, derived tables, subqueries, and CTEs unless TOP, OFFSET, or
    /// FOR XML is present (error 1033), so it appends `OFFSET 0 ROWS`,
    /// which legitimizes the clause without changing the ordering.
    fn sql_derived_order_by(&self, ordering: &str) -> String {
        format!("ORDER BY {ordering}")
    }

    /// Null-safe equality comparison between two expressions.
    ///
    /// The ANSI form is `IS NOT DISTINCT FROM`; MySQL/MariaDB use the
    /// `<=>` operator instead. ClickHouse only accepts the ANSI form in a
    /// `JOIN ON` section, so callers targeting it should place the
    /// comparison in a join condition. The comparison is parenthesized:
    /// DataFusion's parser otherwise binds a following `AND` into the
    /// right-hand operand ("logical boolean operation Utf8View AND
    /// Boolean").
    fn sql_null_safe_eq(&self, left: &str, right: &str) -> String {
        format!("({left} IS NOT DISTINCT FROM {right})")
    }

    /// Ceiling of a numeric expression.
    ///
    /// ANSI `CEIL`; SQL Server only has `CEILING`.
    fn sql_ceil(&self, expr: &str) -> String {
        format!("CEIL({expr})")
    }

    /// Wrap a query as a derived table: `SELECT * FROM (query) AS alias`.
    ///
    /// Dialects that forbid CTEs inside derived tables (SQL Server)
    /// override this to hoist any leading `WITH` clause out of the
    /// parentheses; see [`crate::reader::dialects::split_cte_prefix`].
    fn wrap_as_subquery(&self, query: &str, alias: &str) -> String {
        format!("SELECT * FROM ({query}) AS {alias}")
    }

    /// Wrap a query in an outer SELECT with a custom select list:
    /// `SELECT {select_list} FROM ({query}) AS {alias}`.
    ///
    /// Dialects that forbid CTEs inside derived tables (SQL Server)
    /// override this to hoist any leading `WITH` clause out of the
    /// parentheses; see [`crate::reader::dialects::split_cte_prefix`].
    fn select_from_subquery(&self, select_list: &str, query: &str, alias: &str) -> String {
        format!("SELECT {select_list} FROM ({query}) AS {alias}")
    }

    /// Compute a percentile of a column
    ///
    /// Returns a scalar subquery expression that computes the specified percentile
    /// of a column within an optional grouping context.
    ///
    /// The default implements `percentile_cont` semantics with window
    /// functions only (no native quantile aggregate required): with
    /// `x = fraction * (cnt - 1)`, the result interpolates linearly between
    /// the rows ranked `floor(x) + 1` and `ceil(x) + 1`. This is exact for
    /// every fraction — the earlier NTILE(4) construction was only exact at
    /// the quartiles and returned boundary averages (or NULL past p75)
    /// elsewhere.
    fn sql_percentile(&self, column: &str, fraction: f64, from: &str, groups: &[String]) -> String {
        let __ggsql_pct__ = self.quote_ident("__ggsql_pct__");
        let __ggsql_qt__ = self.quote_ident("__ggsql_qt__");
        // The derived table needs an explicit alias: MySQL/MariaDB reject
        // unaliased derived tables ("Every derived table must have its own
        // alias"), and other engines accept the alias harmlessly.
        let __ggsql_tile__ = self.quote_ident("__ggsql_tile__");
        let quoted_column = self.quote_ident(column);

        // x = fraction * (cnt - 1) is the zero-based fractional rank of the
        // percentile; interpolate between the rows bracketing it.
        let x = format!("{fraction} * (cnt - 1)");
        let lo = format!("1 + FLOOR({x})");
        let hi = format!("1 + {}", self.sql_ceil(&x));
        let frac = format!("{x} - FLOOR({x})");

        // Group correlation belongs in the scalar subquery's own WHERE, not
        // the windowed derived table's: MariaDB cannot resolve outer-query
        // aliases from inside a derived table (Error 1054, "Unknown column
        // ... in 'WHERE'"). The windows are instead partitioned by the
        // group columns and the correlation filters one level up, which
        // keeps the kept group's ranks identical.
        let (partition_by, group_cols, group_filter) = if groups.is_empty() {
            (String::new(), String::new(), String::new())
        } else {
            let quoted: Vec<String> = groups.iter().map(|g| self.quote_ident(g)).collect();
            let filter = quoted
                .iter()
                .map(|q| {
                    self.sql_null_safe_eq(
                        &format!("{__ggsql_tile__}.{q}"),
                        &format!("{__ggsql_qt__}.{q}"),
                    )
                })
                .collect::<Vec<_>>()
                .join(" AND ");
            (
                format!("PARTITION BY {} ", quoted.join(", ")),
                format!(", {}", quoted.join(", ")),
                format!(" WHERE {filter}"),
            )
        };

        format!(
            "(SELECT \
               MAX(CASE WHEN rn = {lo} THEN __val END) + \
               (MAX(CASE WHEN rn = {hi} THEN __val END) - \
                MAX(CASE WHEN rn = {lo} THEN __val END)) * MAX({frac}) \
             FROM (\
               SELECT {quoted_column} AS __val, \
                      ROW_NUMBER() OVER ({partition_by}ORDER BY {quoted_column}) AS rn, \
                      COUNT(*) OVER ({partition_by}) AS cnt{group_cols} \
               FROM ({from}) AS {__ggsql_pct__} \
               WHERE {quoted_column} IS NOT NULL\
             ) AS {__ggsql_tile__}{group_filter})"
        )
    }

    /// Inline-form quantile aggregate, usable directly in a `SELECT` list.
    ///
    /// Returns `Some(sql_fragment)` when the dialect supports a native quantile
    /// aggregate that can be combined with other aggregates in the same `GROUP BY`
    /// query (e.g. DuckDB's `QUANTILE_CONT`). Returns `None` when no native
    /// inline form exists; callers should then fall back to [`sql_percentile`],
    /// which produces a correlated scalar subquery.
    fn sql_quantile_inline(&self, _column: &str, _fraction: f64) -> Option<String> {
        None
    }

    /// SQL fragment for a simple aggregate function applied to an
    /// already-quoted column expression.
    ///
    /// Returns `Some(expr)` when the dialect can express this aggregate inline
    /// in a `GROUP BY` query. Returns `None` when the aggregate is not
    /// supported by this backend; the stat layer surfaces a clear error.
    ///
    /// Names handled here are the entries of `stat_aggregate::AGG_NAMES` other
    /// than the percentile/iqr family, which goes through [`sql_quantile_inline`]
    /// / [`sql_percentile`] instead.
    fn sql_aggregate(&self, name: &str, qcol: &str) -> Option<String> {
        default_sql_aggregate(&|c: &str| self.quote_ident(c), name, qcol)
    }

    /// SQL literal for a date value (days since Unix epoch).
    ///
    /// The default renders an ISO `DATE 'YYYY-MM-DD'` literal, valid on
    /// every backend that accepts ANSI literals; interval arithmetic
    /// (`INTERVAL n DAY`) is not portable.
    fn sql_date_literal(&self, days_since_epoch: i32) -> String {
        // 719163 is the proleptic Gregorian day number of the Unix epoch.
        let date = chrono::NaiveDate::from_num_days_from_ce_opt(719163 + days_since_epoch)
            .expect("date literal out of range");
        format!("DATE '{}'", date.format("%Y-%m-%d"))
    }

    /// SQL literal for a datetime value (microseconds since Unix epoch).
    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        let dt = chrono::DateTime::from_timestamp_micros(microseconds_since_epoch)
            .expect("datetime literal out of range")
            .naive_utc();
        let base = dt.format("%Y-%m-%d %H:%M:%S");
        let micros = microseconds_since_epoch.rem_euclid(1_000_000);
        if micros == 0 {
            format!("TIMESTAMP '{base}'")
        } else {
            format!("TIMESTAMP '{base}.{micros:06}'")
        }
    }

    /// SQL literal for a time value (nanoseconds since midnight).
    fn sql_time_literal(&self, nanoseconds_since_midnight: i64) -> String {
        let seconds = nanoseconds_since_midnight.div_euclid(1_000_000_000);
        let nanos = nanoseconds_since_midnight.rem_euclid(1_000_000_000);
        let time =
            chrono::NaiveTime::from_num_seconds_from_midnight_opt(seconds as u32, nanos as u32)
                .expect("time literal out of range");
        let base = time.format("%H:%M:%S");
        if nanos == 0 {
            format!("TIME '{base}'")
        } else {
            format!("TIME '{base}.{nanos:09}'")
        }
    }

    /// Expression converting a temporal value to its epoch number, for
    /// numeric comparison: days since epoch for dates, microseconds for
    /// datetimes, nanoseconds for times (matching the units ggsql uses for
    /// temporal scale values). Used when a temporal column must be compared
    /// numerically (binned scales with a non-temporal transform).
    ///
    /// Default: a plain cast to the number type, valid where temporal →
    /// numeric casts are allowed (T-SQL, ClickHouse). Dialects with
    /// stricter casts (DuckDB, Postgres, MySQL, …) override.
    fn sql_temporal_as_number(&self, expr: &str, _kind: CastTargetType) -> String {
        let ty = self.number_type_name().unwrap_or("DOUBLE PRECISION");
        self.sql_cast(expr, ty)
    }

    /// SQL literal for a boolean value.
    fn sql_boolean_literal(&self, value: bool) -> String {
        if value {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        }
    }

    /// Whether this backend fundamentally lacks the temporary-table support
    /// ggsql needs to stage internal tables (CTEs, stat transforms), so a
    /// connection through it must always be wrapped in a caching reader.
    ///
    /// Set this for query engines with no DDL at all (Druid, Drill,
    /// DataFusion) or where `CREATE TEMP TABLE` is broadly unsupported
    /// (Trino). Backends whose support is merely *uncertain* — e.g. the
    /// account may be read-only — should keep the default: connections are
    /// probed once on connect and wrapped only when the probe fails.
    fn requires_cache(&self) -> bool {
        false
    }

    /// SQL listing catalogs, with a single `catalog_name` output column.
    ///
    /// Returns `None` to use the `Reader` default (`information_schema`).
    /// Override for backends without `information_schema` (e.g. Exasol's
    /// `SYS.EXA_*` tables).
    fn sql_list_catalogs(&self) -> Option<String> {
        None
    }

    /// SQL listing schemas in `catalog`, with a single `schema_name` column.
    fn sql_list_schemas(&self, _catalog: &str) -> Option<String> {
        None
    }

    /// SQL listing tables in `catalog`/`schema`, with `table_name` and
    /// `table_type` output columns.
    fn sql_list_tables(&self, _catalog: &str, _schema: &str) -> Option<String> {
        None
    }

    /// SQL listing columns of `catalog`/`schema`/`table`, with `column_name`
    /// and `data_type` output columns.
    fn sql_list_columns(&self, _catalog: &str, _schema: &str, _table: &str) -> Option<String> {
        None
    }

    /// DDL statement(s) creating an empty temporary table with explicit
    /// column definitions (`"name TYPE"` pairs), used by `Reader::register`
    /// to stage a data frame before bulk-inserting into it.
    fn sql_create_empty_temp_table(&self, name: &str, column_defs: &[String]) -> Vec<String> {
        vec![format!(
            "CREATE TEMPORARY TABLE {} ({})",
            self.quote_ident(name),
            column_defs.join(", ")
        )]
    }

    /// Build the DDL statement(s) needed to (re)create a temporary table
    /// that holds the result of `body_sql`.
    ///
    /// Column aliases from `WITH t(a, b) AS (...)` are preserved portably by
    /// wrapping the body in a named CTE with a column alias list, so the
    /// backend never needs to support `CREATE TABLE t(a, b) AS ...` syntax.
    ///
    /// Returned statements must be executed in order via `Reader::execute_sql`.
    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = self.quote_ident(name);
        let body =
            wrap_with_column_aliases(&|c: &str| self.quote_ident(c), body_sql, column_aliases);
        vec![
            format!("DROP TABLE IF EXISTS {}", qname),
            format!("CREATE TEMP TABLE {} AS {}", qname, body),
        ]
    }
}

/// Wrap a body SQL in a CTE with a column alias list when aliases are present.
/// This is a portable way to rename the body's output columns without relying
/// on `CREATE TABLE t(a, b) AS ...` (which SQLite does not support).
pub(crate) fn wrap_with_column_aliases(
    quote: &dyn Fn(&str) -> String,
    body_sql: &str,
    column_aliases: &[String],
) -> String {
    if column_aliases.is_empty() {
        return body_sql.to_string();
    }
    let cols = column_aliases
        .iter()
        .map(|c| quote(c))
        .collect::<Vec<_>>()
        .join(", ");
    let __ggsql_aliased__ = quote("__ggsql_aliased__");
    format!(
        "WITH {__ggsql_aliased__}({}) AS ({}) SELECT * FROM {__ggsql_aliased__}",
        cols, body_sql
    )
}

/// Default aggregate SQL emission, shared so dialects can opt into the standard
/// portable forms while overriding selected functions.
///
/// `first` / `last` are expressed as `MAX(CASE WHEN __ggsql_rn__ = … THEN col END)`,
/// which depends on the row-number columns the stat layer injects when any
/// aggregate references them. Backends with a cheaper native equivalent
/// (e.g. DuckDB's `FIRST`/`LAST`) override [`SqlDialect::sql_aggregate`].
pub fn default_sql_aggregate(
    quote: &dyn Fn(&str) -> String,
    name: &str,
    qcol: &str,
) -> Option<String> {
    let __ggsql_rn__ = quote("__ggsql_rn__");
    let __ggsql_max_rn__ = quote("__ggsql_max_rn__");
    let s = match name {
        "count" => format!("COUNT({})", qcol),
        "sum" => format!("SUM({})", qcol),
        "prod" => format!("EXP(SUM(LN({})))", qcol),
        "min" => format!("MIN({})", qcol),
        "max" => format!("MAX({})", qcol),
        "range" => format!("(MAX({c}) - MIN({c}))", c = qcol),
        "mid" => format!("((MIN({c}) + MAX({c})) / 2.0)", c = qcol),
        "mean" => format!("AVG({})", qcol),
        "geomean" => format!("EXP(AVG(LN({})))", qcol),
        "harmean" => format!("(COUNT({c}) * 1.0 / SUM(1.0 / {c}))", c = qcol),
        "rms" => format!("SQRT(AVG({c} * {c}))", c = qcol),
        "sdev" => format!("STDDEV_POP({})", qcol),
        "se" => format!("(STDDEV_POP({c}) / SQRT(COUNT({c})))", c = qcol),
        "var" => format!("VAR_POP({})", qcol),
        "first" => format!("MAX(CASE WHEN {__ggsql_rn__} = 1 THEN {} END)", qcol),
        "last" => format!(
            "MAX(CASE WHEN {__ggsql_rn__} = {__ggsql_max_rn__} THEN {} END)",
            qcol
        ),
        "diff" => format!(
            "(MAX(CASE WHEN {__ggsql_rn__} = {__ggsql_max_rn__} THEN {c} END) \
             - MAX(CASE WHEN {__ggsql_rn__} = 1 THEN {c} END))",
            c = qcol
        ),
        _ => return None,
    };
    Some(s)
}

pub struct AnsiDialect;
impl SqlDialect for AnsiDialect {}

/// Fail fast when a spatial feature is used on a backend whose dialect
/// reports `supports_spatial() == false`, rather than emitting spatial SQL
/// the backend cannot run.
pub(crate) fn ensure_spatial_supported(dialect: &dyn SqlDialect) -> Result<()> {
    if dialect.supports_spatial() {
        Ok(())
    } else {
        Err(GgsqlError::ValidationError(
            "Spatial operations are not supported by this database backend".into(),
        ))
    }
}

#[cfg(feature = "duckdb")]
pub mod duckdb;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "odbc")]
pub mod odbc;

#[cfg(feature = "adbc")]
pub mod adbc;

#[cfg(any(feature = "duckdb", feature = "sqlite"))]
pub mod cache;

#[cfg(all(test, feature = "duckdb", feature = "sqlite"))]
mod cache_equivalence;

pub mod connection;
pub mod data;
pub mod dialects;
mod spec;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteReader;

#[cfg(feature = "odbc")]
pub use odbc::OdbcReader;

#[cfg(feature = "adbc")]
pub use adbc::AdbcReader;

#[cfg(any(feature = "duckdb", feature = "sqlite"))]
pub use cache::CachingReader;

// ============================================================================
// Shared utilities
// ============================================================================

/// Extract the numeric SRID from an EPSG string (e.g. "EPSG:4326" → 4326).
pub(crate) fn extract_epsg_srid(crs: &str) -> Option<u32> {
    crs.strip_prefix("EPSG:").and_then(|s| s.parse().ok())
}

/// Validate a table name for use in SQL statements.
///
/// Rejects empty names and names containing null bytes or newlines.
pub(crate) fn validate_table_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(GgsqlError::ReaderError("Table name cannot be empty".into()));
    }

    let forbidden = ['\0', '\n', '\r'];
    for ch in forbidden {
        if name.contains(ch) {
            return Err(GgsqlError::ReaderError(format!(
                "Table name '{}' contains invalid character '{}'",
                name,
                ch.escape_default()
            )));
        }
    }

    Ok(())
}

/// Does the SQL statement return rows?
///
/// Looks at the first keyword to decide: `SELECT`, `WITH`, `FROM`,
/// `DESCRIBE`, `SHOW` and `EXPLAIN` produce result sets; everything else
/// (DDL, DML) does not.
pub(crate) fn returns_rows(sql: &str) -> bool {
    let first_word = sql.split_whitespace().next().unwrap_or("");
    matches!(
        first_word.to_ascii_uppercase().as_str(),
        "SELECT" | "WITH" | "DESCRIBE" | "SHOW" | "EXPLAIN" | "FROM"
    )
}

/// Shared test helpers for reader equivalence suites.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{
        execute_with_reader, returns_rows, ColumnInfo, Reader, Spec, SqlDialect, TableInfo,
    };
    use crate::{DataFrame, GgsqlError, Result};
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
        dialect: Box<dyn SqlDialect>,
        tables: Mutex<HashMap<String, Arc<Schema>>>,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl StubReader {
        /// Create a stub for `dialect`, returning it together with a
        /// handle to the shared SQL log.
        pub(crate) fn new(dialect: Box<dyn SqlDialect>) -> (Self, Arc<Mutex<Vec<String>>>) {
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
                .map(|f| sample_array(f.data_type()))
                .collect();
            let batch = RecordBatch::try_new(schema, arrays)
                .expect("stub arrays must match fabricated schema");
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
            tables.values().find_map(|schema| {
                schema.fields().iter().find_map(|f| {
                    if f.name().eq_ignore_ascii_case(name) {
                        Some(f.data_type().clone())
                    } else {
                        None
                    }
                })
            })
        }

        /// Type of the longest registered column name appearing as a word
        /// in `expr` (e.g. `CAST("day" AS DATE)` → the type of `day`).
        fn type_from_expr(&self, expr: &str) -> Option<DataType> {
            let tables = self.tables.lock().unwrap();
            let mut best: Option<DataType> = None;
            let mut best_len = 0;
            for schema in tables.values() {
                for field in schema.fields() {
                    if field.name().len() > best_len && contains_word(expr, field.name()) {
                        best = Some(field.data_type().clone());
                        best_len = field.name().len();
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
            DataType::Utf8 => Arc::new(StringArray::from(vec!["a", "b", "c"])),
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
}

// ============================================================================
// Spec - Result of reader.execute()
// ============================================================================

/// Result of executing a ggsql query, ready for rendering.
pub struct Spec {
    /// Single resolved plot specification
    pub(crate) plot: Plot,
    /// Internal data map (global + layer-specific DataFrames)
    pub(crate) data: HashMap<String, DataFrame>,
    /// Cached metadata about the prepared visualization
    pub(crate) metadata: Metadata,
    /// The main SQL query that was executed
    pub(crate) sql: String,
    /// The raw VISUALISE portion text
    pub(crate) visual: String,
    /// Per-layer filter/source queries (None = uses global data directly)
    pub(crate) layer_sql: Vec<Option<String>>,
    /// Per-layer stat transform queries (None = no stat transform)
    pub(crate) stat_sql: Vec<Option<String>>,
    /// Validation warnings from preparation
    pub(crate) warnings: Vec<ValidationWarning>,
}

/// Metadata about the prepared visualization.
#[derive(Debug, Clone)]
pub struct Metadata {
    pub rows: usize,
    pub columns: Vec<String>,
    pub layer_count: usize,
}

// ============================================================================
// Reader Trait
// ============================================================================

/// Trait for data source readers
///
/// Readers execute SQL queries and return Polars DataFrames.
/// They provide a uniform interface for different database backends.
///
/// # DataFrame Registration
///
/// Readers support registering DataFrames as queryable tables using
/// the [`register`](Reader::register) method. This allows you to query
/// in-memory DataFrames with SQL, join them with other tables, etc.
///
/// ```rust,ignore
/// // Register a DataFrame (takes ownership)
/// reader.register("sales", sales_df, false)?;
///
/// // Now you can query it
/// let result = reader.execute_sql("SELECT * FROM sales WHERE amount > 100")?;
/// ```
pub trait Reader {
    /// Execute a SQL query and return the result as a DataFrame.
    ///
    /// This is the **source surface**: base reads of the user's data plus user
    /// setup/DML. A plain reader runs everything on its one connection; a caching
    /// reader reads the primary here (with result memoization).
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::ReaderError` if:
    /// - The SQL is invalid
    /// - The connection fails
    /// - The table or columns don't exist
    fn execute_sql(&self, sql: &str) -> Result<DataFrame>;

    /// Execute SQL against the *compute surface* — dialect-generated/derived SQL
    /// over internal `__ggsql_*` tables.
    ///
    /// Defaults to the source surface ([`Reader::execute_sql`]), so a plain reader
    /// runs everything on one connection. A caching reader overrides this to run
    /// on the in-memory cache, where all derived `__ggsql_*` tables live.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::ReaderError` if:
    /// - The SQL is invalid
    /// - The connection fails
    /// - The table or columns don't exist
    fn execute_sql_cached(&self, sql: &str) -> Result<DataFrame> {
        self.execute_sql(sql)
    }

    /// Register a DataFrame as a queryable table (takes ownership)
    ///
    /// After registration, the DataFrame can be queried by name in SQL:
    /// ```sql
    /// SELECT * FROM <name> WHERE ...
    /// ```
    ///
    /// # Arguments
    ///
    /// * `name` - The table name to register under
    /// * `df` - The DataFrame to register (ownership is transferred)
    /// * `replace` - If true, replace any existing table with the same name.
    ///   If false, return an error if the table already exists.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if registration fails.
    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()>;

    /// Unregister a previously registered table
    ///
    /// # Arguments
    ///
    /// * `name` - The table name to unregister
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Default Implementation
    ///
    /// Returns an error by default. Override for readers that support unregistration.
    fn unregister(&self, name: &str) -> Result<()> {
        Err(GgsqlError::ReaderError(format!(
            "This reader does not support unregistering table '{}'",
            name
        )))
    }

    /// Execute a ggsql query and return the visualization specification.
    ///
    /// This is the main entry point for creating visualizations. It parses the query,
    /// executes the SQL portion, and returns a `Spec` ready for rendering.
    ///
    /// # Arguments
    ///
    /// * `query` - The ggsql query (SQL + VISUALISE clause)
    ///
    /// # Returns
    ///
    /// A `Spec` containing the resolved visualization specification and data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query syntax is invalid
    /// - The query has no VISUALISE clause
    /// - The SQL execution fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ggsql::reader::{Reader, DuckDBReader};
    /// use ggsql::writer::{Writer, VegaLiteWriter};
    ///
    /// let mut reader = DuckDBReader::from_connection_string("duckdb://memory")?;
    /// let spec = reader.execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")?;
    ///
    /// let writer = VegaLiteWriter::new();
    /// let json = writer.render(&spec)?;
    /// ```
    fn execute(&self, query: &str) -> Result<Spec>;

    /// Get the SQL dialect for this reader.
    ///
    /// Database-specific SQL type names and SQL generation methods
    fn dialect(&self) -> &dyn SqlDialect {
        &AnsiDialect
    }

    /// Materialize the result of `body_sql` as a temporary table named `name`.
    fn materialize_table(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Result<()> {
        for stmt in self
            .dialect()
            .create_or_replace_temp_table_sql(name, column_aliases, body_sql)
        {
            self.execute_sql(&stmt)?;
        }
        Ok(())
    }

    /// Whether this reader stages external data sources into a separate cache.
    fn caches_sources(&self) -> bool {
        false
    }

    /// Clear any cached query results held by this reader.
    fn clear_cache(&self) -> Result<()> {
        Ok(())
    }

    // =========================================================================
    // Schema introspection
    // =========================================================================

    fn list_catalogs(&self) -> Result<Vec<String>> {
        let sql = self.dialect().sql_list_catalogs().unwrap_or_else(|| {
            "SELECT DISTINCT catalog_name FROM information_schema.schemata ORDER BY catalog_name"
                .to_string()
        });
        let df = self.execute_sql(&sql)?;
        let col = df.column("catalog_name")?;
        let mut results = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            if !col.is_null(i) {
                results.push(crate::array_util::value_to_string(col, i));
            }
        }
        Ok(results)
    }

    fn list_schemas(&self, catalog: &str) -> Result<Vec<String>> {
        let sql = self.dialect().sql_list_schemas(catalog).unwrap_or_else(|| {
            format!(
                "SELECT DISTINCT schema_name FROM information_schema.schemata \
                 WHERE catalog_name = {} ORDER BY schema_name",
                naming::quote_literal(catalog)
            )
        });
        let df = self.execute_sql(&sql)?;
        let col = df.column("schema_name")?;
        let mut results = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            if !col.is_null(i) {
                results.push(crate::array_util::value_to_string(col, i));
            }
        }
        Ok(results)
    }

    fn list_tables(&self, catalog: &str, schema: &str) -> Result<Vec<TableInfo>> {
        let sql = self
            .dialect()
            .sql_list_tables(catalog, schema)
            .unwrap_or_else(|| {
                format!(
                    "SELECT DISTINCT table_name, table_type FROM information_schema.tables \
                 WHERE table_catalog = {} AND table_schema = {} ORDER BY table_name",
                    naming::quote_literal(catalog),
                    naming::quote_literal(schema)
                )
            });
        let df = self.execute_sql(&sql)?;
        let name_col = df.column("table_name")?;
        let type_col = df.column("table_type")?;
        let mut results = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            if !name_col.is_null(i) {
                results.push(TableInfo {
                    name: crate::array_util::value_to_string(name_col, i),
                    table_type: crate::array_util::value_to_string(type_col, i),
                });
            }
        }
        Ok(results)
    }

    fn list_columns(&self, catalog: &str, schema: &str, table: &str) -> Result<Vec<ColumnInfo>> {
        let sql = self
            .dialect()
            .sql_list_columns(catalog, schema, table)
            .unwrap_or_else(|| {
                format!(
                    "SELECT column_name, data_type FROM information_schema.columns \
                     WHERE table_catalog = {} AND table_schema = {} AND table_name = {} \
                     ORDER BY ordinal_position",
                    naming::quote_literal(catalog),
                    naming::quote_literal(schema),
                    naming::quote_literal(table)
                )
            });
        let df = self.execute_sql(&sql)?;
        let name_col = df.column("column_name")?;
        let type_col = df.column("data_type")?;
        let mut results = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            if !name_col.is_null(i) {
                results.push(ColumnInfo {
                    name: crate::array_util::value_to_string(name_col, i),
                    data_type: crate::array_util::value_to_string(type_col, i),
                });
            }
        }
        Ok(results)
    }
}

/// A reader that can serve as an in-memory, writable caching backend.
///
/// Cache backends take no options: they are always a fresh in-memory, writable
/// database scoped to the process; consumed by [`CachingReader`].
pub trait CacheBackend: Reader {
    fn new_in_memory() -> Result<Self>
    where
        Self: Sized;
}

/// A table or view in the schema.
pub struct TableInfo {
    pub name: String,
    pub table_type: String,
}

/// A column in a table.
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
}

/// Execute a ggsql query using any reader
///
/// This is the shared implementation behind `Reader::execute()`. Concrete
/// readers delegate to this so the trait stays object-safe (no `Self: Sized`
/// bound on `execute`).
pub fn execute_with_reader(reader: &dyn Reader, query: &str) -> Result<Spec> {
    let validated = validate(query)?;
    let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

    let prepared_data = prepare_data_with_reader(query, reader)?;

    let plot =
        prepared_data.specs.into_iter().next().ok_or_else(|| {
            GgsqlError::ValidationError("No visualization spec found".to_string())
        })?;

    let layer_sql = vec![None; plot.layers.len()];
    let stat_sql = vec![None; plot.layers.len()];

    Ok(Spec::new(
        plot,
        prepared_data.data,
        prepared_data.sql,
        prepared_data.visual,
        layer_sql,
        stat_sql,
        warnings,
    ))
}

#[cfg(test)]
#[cfg(all(feature = "duckdb", feature = "vegalite"))]
mod tests {
    use super::*;
    use crate::df;
    use crate::writer::{VegaLiteWriter, Writer};

    fn data_layer(json: &serde_json::Value, index: usize) -> &serde_json::Value {
        json["layer"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|l| {
                !matches!(
                    l.get("description").and_then(|d| d.as_str()),
                    Some("background" | "foreground")
                )
            })
            .nth(index)
            .expect("data layer not found at index")
    }

    #[test]
    fn test_execute_and_render() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")
            .unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert_eq!(spec.metadata().layer_count, 1);
        assert!(spec.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    #[test]
    fn test_execute_metadata() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute(
                "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y) VISUALISE x, y DRAW point",
            )
            .unwrap();

        let metadata = spec.metadata();
        assert_eq!(metadata.rows, 3);
        // Columns now includes both user mappings (pos1, pos2) and resolved defaults (size, stroke, fill, opacity, shape, linewidth)
        // Aesthetics are transformed to internal names (x -> pos1, y -> pos2)
        assert!(metadata.columns.contains(&"pos1".to_string()));
        assert!(metadata.columns.contains(&"pos2".to_string()));
        assert_eq!(metadata.layer_count, 1);
    }

    #[test]
    fn test_execute_with_cte() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            WITH data AS (
                SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
            )
            SELECT * FROM data
            VISUALISE x, y DRAW point
        "#;

        let spec = reader.execute(query).unwrap();

        assert_eq!(spec.plot().layers.len(), 1);
        assert!(spec.layer_data(0).is_some());
        let df = spec.layer_data(0).unwrap();
        assert_eq!(df.height(), 2);
    }

    #[test]
    fn test_render_multi_layer() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y
            DRAW line MAPPING x AS x, y AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        assert!(result.contains("layer"));
    }

    #[test]
    fn test_polar_project_with_start() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING start => 90
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Parse the JSON to verify the theta scale range is set correctly
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();

        // The encoding should have a theta channel with a scale range offset by 90 degrees
        // 90 degrees = π/2 radians
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        assert!(theta.is_object(), "theta encoding should exist");

        // Check that the scale has a range with the start offset
        let scale = &theta["scale"];
        let range = scale["range"].as_array().unwrap();
        assert_eq!(range.len(), 2);

        // π/2 ≈ 1.5707963
        let start = range[0].as_f64().unwrap();
        assert!(
            (start - std::f64::consts::FRAC_PI_2).abs() < 0.001,
            "start should be π/2 (90 degrees), got {}",
            start
        );

        // π/2 + 2π ≈ 7.8539816
        let end = range[1].as_f64().unwrap();
        let expected_end = std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI;
        assert!(
            (end - expected_end).abs() < 0.001,
            "end should be π/2 + 2π, got {}",
            end
        );
    }

    #[test]
    fn test_polar_project_default_start() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Parse the JSON
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();

        // The theta encoding should NOT have a scale with range when start is 0 (default)
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        assert!(theta.is_object(), "theta encoding should exist");

        // Either no scale, or no range in scale (since default is 0)
        if let Some(scale) = theta.get("scale") {
            assert!(
                scale.get("range").is_none(),
                "theta scale should not have range when start is 0"
            );
        }
    }

    #[test]
    fn test_polar_project_with_end() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING start => -90, end => 90
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        let range = theta["scale"]["range"].as_array().unwrap();

        // -90° = -π/2 ≈ -1.5708, 90° = π/2 ≈ 1.5708
        let start = range[0].as_f64().unwrap();
        let end = range[1].as_f64().unwrap();
        assert!(
            (start - (-std::f64::consts::FRAC_PI_2)).abs() < 0.001,
            "start should be -π/2 (-90 degrees), got {}",
            start
        );
        assert!(
            (end - std::f64::consts::FRAC_PI_2).abs() < 0.001,
            "end should be π/2 (90 degrees), got {}",
            end
        );
    }

    #[test]
    fn test_polar_project_with_end_only() {
        // Test using end without explicit start (start defaults to 0)
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING end => 180
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let theta = &layer["encoding"]["theta"];
        let range = theta["scale"]["range"].as_array().unwrap();

        // start=0 (default), end=180° = π
        let start = range[0].as_f64().unwrap();
        let end = range[1].as_f64().unwrap();
        assert!(
            start.abs() < 0.001,
            "start should be 0 (default), got {}",
            start
        );
        assert!(
            (end - std::f64::consts::PI).abs() < 0.001,
            "end should be π (180 degrees), got {}",
            end
        );
    }

    #[test]
    fn test_polar_encoding_keys_independent_of_user_names() {
        // This test verifies that polar projections always produce theta/radius encoding keys
        // in Vega-Lite output, regardless of what position names the user specified in PROJECT.
        // This is critical because Vega-Lite expects specific channel names for polar marks.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Helper to check encoding keys
        fn check_encoding_keys(json: &serde_json::Value, test_name: &str) {
            let layer = data_layer(json, 0);
            assert!(
                layer["encoding"].get("theta").is_some(),
                "{} should produce theta encoding, got keys: {:?}",
                test_name,
                layer["encoding"]
                    .as_object()
                    .map(|o| o.keys().collect::<Vec<_>>())
            );
            // Also verify no x or y keys exist (they should be mapped to theta/radius)
            assert!(
                layer["encoding"].get("x").is_none(),
                "{} should NOT have x encoding in polar mode",
                test_name
            );
            assert!(
                layer["encoding"].get("y").is_none(),
                "{} should NOT have y encoding in polar mode",
                test_name
            );
        }

        // Test case 1: PROJECT y, x TO polar (y as pos1→radius, x as pos2→theta)
        let query1 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar
        "#;
        let spec1 = reader.execute(query1).unwrap();
        let writer = VegaLiteWriter::new();
        let result1 = writer.render(&spec1).unwrap();
        let json1: serde_json::Value = serde_json::from_str(&result1).unwrap();
        check_encoding_keys(&json1, "PROJECT y, x TO polar");

        // Test case 2: PROJECT x, y TO polar (x as pos1→radius, y as pos2→theta)
        let query2 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS x, category AS fill
            DRAW bar
            PROJECT x, y TO polar
        "#;
        let spec2 = reader.execute(query2).unwrap();
        let result2 = writer.render(&spec2).unwrap();
        let json2: serde_json::Value = serde_json::from_str(&result2).unwrap();
        check_encoding_keys(&json2, "PROJECT x, y TO polar");

        // Test case 3: PROJECT TO polar (default radius/angle names)
        let query3 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS angle, category AS fill
            DRAW bar
            PROJECT TO polar
        "#;
        let spec3 = reader.execute(query3).unwrap();
        let result3 = writer.render(&spec3).unwrap();
        let json3: serde_json::Value = serde_json::from_str(&result3).unwrap();
        check_encoding_keys(&json3, "PROJECT TO polar");

        // Test case 4: PROJECT a, b TO polar (custom aesthetic names)
        let query4 = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS a, category AS fill
            DRAW bar
            PROJECT a, b TO polar
        "#;
        let spec4 = reader.execute(query4).unwrap();
        let result4 = writer.render(&spec4).unwrap();
        let json4: serde_json::Value = serde_json::from_str(&result4).unwrap();
        check_encoding_keys(&json4, "PROJECT a, b TO polar (custom names)");
    }

    #[test]
    fn test_cartesian_encoding_keys_with_custom_names() {
        // This test verifies that cartesian projections produce x/y encoding keys
        // even when custom position names are used in PROJECT.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        fn check_cartesian_keys(json: &serde_json::Value, test_name: &str) {
            let layer = data_layer(json, 0);
            assert!(
                layer["encoding"].get("x").is_some(),
                "{} should produce x encoding, got keys: {:?}",
                test_name,
                layer["encoding"]
                    .as_object()
                    .map(|o| o.keys().collect::<Vec<_>>())
            );
            // Verify no theta/radius keys exist
            assert!(
                layer["encoding"].get("theta").is_none(),
                "{} should NOT have theta encoding in cartesian mode",
                test_name
            );
        }

        // Test case: PROJECT a, b TO cartesian (custom aesthetic names)
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE category AS a, value AS b
            DRAW bar
            PROJECT a, b TO cartesian
        "#;
        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        check_cartesian_keys(&json, "PROJECT a, b TO cartesian (custom names)");
    }

    #[test]
    fn test_register_and_query() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let df = df! {
            "x" => vec![1i32, 2, 3],
            "y" => vec![10i32, 20, 30],
        }
        .unwrap();

        reader.register("my_data", df, false).unwrap();

        let query = "SELECT * FROM my_data VISUALISE x, y DRAW point";
        let spec = reader.execute(query).unwrap();

        assert_eq!(spec.metadata().rows, 3);
        // Aesthetics are transformed to internal names (x -> pos1)
        assert!(spec.metadata().columns.contains(&"pos1".to_string()));

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    #[test]
    fn test_register_and_join() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let sales = df! {
            "id" => vec![1i32, 2, 3],
            "amount" => vec![100i32, 200, 300],
            "product_id" => vec![1i32, 1, 2],
        }
        .unwrap();

        let products = df! {
            "id" => vec![1i32, 2],
            "name" => vec!["Widget", "Gadget"],
        }
        .unwrap();

        reader.register("sales", sales, false).unwrap();
        reader.register("products", products, false).unwrap();

        let query = r#"
            SELECT s.id, s.amount, p.name
            FROM sales s
            JOIN products p ON s.product_id = p.id
            VISUALISE id AS x, amount AS y
            DRAW bar
        "#;

        let spec = reader.execute(query).unwrap();
        assert_eq!(spec.metadata().rows, 3);
    }

    #[test]
    fn test_execute_no_viz_fails() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = "SELECT 1 as x, 2 as y";

        let result = reader.execute(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_binned_fill_legend_renders_threshold_scale() {
        // End-to-end test for binned fill scale rendering to Vega-Lite
        // Verifies that binned material aesthetics use threshold scale type
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        // Create data with values that span the binned range
        // Binned scales use FROM [min, max] for range and SETTING breaks => [...] for explicit breaks
        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 15.0),
                (2, 20, 35.0),
                (3, 30, 55.0),
                (4, 40, 85.0)
            ) AS t(x, y, value)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, value AS fill
            SCALE BINNED fill FROM [0, 100] TO viridis SETTING breaks => [0, 25, 50, 75, 100]
        "#;

        let spec = reader.execute(query).unwrap();

        // Verify spec structure
        assert_eq!(spec.plot().layers.len(), 1);
        // Note: scales may include auto-generated x/y scales plus the explicit fill scale
        assert!(
            spec.plot().find_scale("fill").is_some(),
            "Should have a fill scale"
        );

        // Render to Vega-Lite
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

        // Verify threshold scale type for fill
        let fill_scale = &vl["layer"][0]["encoding"]["fill"]["scale"];
        assert_eq!(
            fill_scale["type"],
            "threshold",
            "Binned fill should use threshold scale type. Got: {}",
            serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]).unwrap()
        );

        // Verify internal breaks as domain (excludes first and last terminals)
        // breaks = [0, 25, 50, 75, 100] → domain = [25, 50, 75]
        let domain = fill_scale["domain"].as_array().unwrap();
        assert_eq!(
            domain.len(),
            3,
            "Threshold domain should have internal breaks only. Got: {:?}",
            domain
        );
        assert_eq!(domain[0], 25.0);
        assert_eq!(domain[1], 50.0);
        assert_eq!(domain[2], 75.0);

        // Verify color output - viridis palette gets expanded to an explicit range array
        // for threshold scales (Vega-Lite needs explicit colors for threshold domain)
        assert!(
            fill_scale["range"].is_array() || fill_scale["scheme"] == "viridis",
            "Should have color range or scheme. Got scale: {}",
            serde_json::to_string_pretty(fill_scale).unwrap()
        );

        // Verify legend values
        // For `fill` alone (single binned legend scale), uses gradient legend with all 5 break values
        // For symbol legends (multiple binned scales or non-gradient aesthetics), would have N-1 values
        let legend_values = &vl["layer"][0]["encoding"]["fill"]["legend"]["values"];
        assert!(
            legend_values.is_array(),
            "Legend should have values array. Got: {}",
            serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]["legend"]).unwrap()
        );
        let values = legend_values.as_array().unwrap();
        assert_eq!(
            values.len(),
            5,
            "Gradient legend should have all 5 break values. Got: {:?}",
            values
        );
    }

    #[test]
    fn test_binned_color_legend_with_label_mapping() {
        // Test binned color scale with custom labels renders correctly
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

        let query = r#"
            SELECT * FROM (VALUES
                (1, 10, 20.0),
                (2, 20, 60.0),
                (3, 30, 90.0)
            ) AS t(x, y, score)
            VISUALISE
            DRAW point MAPPING x AS x, y AS y, score AS color
            SCALE BINNED color FROM [0, 100] TO ['blue', 'yellow', 'red'] SETTING breaks => [0, 50, 100]
                RENAMING 0 => 'Low', 50 => 'High'
        "#;

        let spec = reader.execute(query).unwrap();

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

        // Verify threshold scale
        // Note: "color" aesthetic is mapped to "stroke" for point geom (not fill)
        let encoding = if vl["layer"].is_array() {
            &vl["layer"][0]["encoding"]
        } else {
            &vl["encoding"]
        };
        // Find the stroke or fill encoding (color maps to one of these)
        let color_encoding = if encoding["stroke"].is_object() {
            &encoding["stroke"]
        } else {
            &encoding["fill"]
        };
        assert_eq!(
            color_encoding["scale"]["type"],
            "threshold",
            "Binned color should use threshold scale. Got encoding: {}",
            serde_json::to_string_pretty(color_encoding).unwrap()
        );

        // Verify labelExpr exists for custom labels
        let legend = &color_encoding["legend"];
        assert!(
            legend["labelExpr"].is_string(),
            "Legend should have labelExpr for custom labels. Got legend: {}",
            serde_json::to_string_pretty(legend).unwrap()
        );

        let label_expr = legend["labelExpr"].as_str().unwrap_or("");
        // For symbol legends, VL generates range-style labels like "0 – 50"
        // Our labelExpr should map these to custom range formats
        assert!(
            label_expr.contains("Low") || label_expr.contains("High"),
            "labelExpr should contain custom labels, got: {}",
            label_expr
        );
    }

    #[test]
    fn test_polar_project_with_inner() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS y, category AS fill
            DRAW bar
            PROJECT y, x TO polar SETTING inner => 0.5
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Check radius scale has range with expressions
        let radius = &layer["encoding"]["radius"];
        assert!(radius["scale"]["range"].is_array());
        let range = radius["scale"]["range"].as_array().unwrap();

        // First element should be inner proportion expression
        assert!(
            range[0]["expr"].as_str().unwrap().contains("0.5"),
            "Inner radius expression should contain 0.5, got: {:?}",
            range[0]
        );

        // Second element should be the outer radius expression
        assert!(
            range[1]["expr"]
                .as_str()
                .unwrap()
                .contains("min(width, height) / 2"),
            "Outer radius expression should contain min(width, height) / 2, got: {:?}",
            range[1]
        );
    }

    #[test]
    fn test_stacked_bar_chart() {
        // Test stacked bar chart via position => 'stack'
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 'X', 10),
                ('A', 'Y', 20),
                ('B', 'X', 15),
                ('B', 'Y', 25)
            ) AS t(cat, grp, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y, grp AS fill
            SETTING position => 'stack'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
        let encoding = &layer["encoding"];
        assert!(encoding["y"].is_object(), "Should have y encoding");
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars"
        );

        // Verify Vega-Lite stacking is disabled (we handle it ourselves)
        assert!(
            encoding["y"]["stack"].is_null(),
            "y encoding should have stack: null to disable VL stacking. Got: {}",
            serde_json::to_string_pretty(&encoding["y"]).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_stacked_bar_chart_dummy_x() {
        // Test stacked bar chart with no x mapping (dummy x column)
        // This is the case where only fill is mapped: all bars at same x position should stack
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW bar MAPPING species AS fill
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
        let encoding = &layer["encoding"];
        assert!(encoding["y"].is_object(), "Should have y encoding");
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars with dummy x. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );

        // Verify Vega-Lite stacking is disabled (we handle it ourselves)
        assert!(
            encoding["y"]["stack"].is_null(),
            "y encoding should have stack: null to disable VL stacking. Got: {}",
            serde_json::to_string_pretty(&encoding["y"]).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_boxplot_dummy_x() {
        // Boxplot with only y mapped: should render a single boxplot of the
        // whole distribution and suppress the categorical x axis.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW boxplot MAPPING bill_len AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Boxplot is a composite renderer (multiple sub-layers). Check that
        // the first layer's x encoding suppresses its axis.
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Boxplot dummy x should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_violin_dummy_x() {
        // Violin with only y mapped: single violin spanning the whole dataset.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW violin MAPPING bill_len AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Violin dummy x should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_point_dummy_x() {
        // Point with only y mapped: strip plot at a single dummy x position.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW point MAPPING bill_len AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Point dummy x should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[test]
    fn test_range_dummy_x() {
        // Range with only ymin/ymax mapped: a single vertical interval.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT 10.0 AS lo, 20.0 AS hi
            VISUALISE
            DRAW range MAPPING lo AS ymin, hi AS ymax
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Range dummy x should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_point_dummy_y() {
        // Symmetric to test_point_dummy_x: only x mapped means dummy y.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW point MAPPING bill_len AS x
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["y"]["axis"].is_null(),
            "Point dummy y should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_point_dummy_both_with_aggregate() {
        // Both axes omitted, but aggregate gives the single point meaning:
        // a count of all rows at the dummy x/y intersection.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW point MAPPING bill_len AS size
            SETTING aggregate => 'size:count'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Both-dummy point should hide x axis. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
        assert!(
            encoding["y"]["axis"].is_null(),
            "Both-dummy point should hide y axis. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_point_dummy_x_with_aggregate() {
        // Point with aggregate SETTING and no x mapping: should aggregate the
        // whole dataset to a single point and suppress the dummy x axis.
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW point MAPPING bill_len AS y
            SETTING aggregate => 'mean'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];
        assert!(
            encoding["x"]["axis"].is_null(),
            "Aggregated point with dummy x should have axis: null. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[cfg(feature = "builtin-data")]
    #[test]
    fn test_bar_chart_with_expand_setting() {
        // Test bar chart with SCALE y SETTING expand - should work even when y is stat-derived
        // This tests that:
        // 1. Scale type inference works for stat-generated count columns
        // 2. Stacking still works (y2 encoding exists) when SCALE y is specified
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            VISUALISE FROM ggsql:penguins
            DRAW bar MAPPING species AS fill
            SCALE y SETTING expand => [0.05, 0.05]
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        // Should succeed without "discrete scale does not support SETTING 'expand'" error
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify stacking works (y2 encoding exists for stacked bars)
        let encoding = &layer["encoding"];
        assert!(
            encoding["y2"].is_object(),
            "Should have y2 encoding for stacked bars. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[test]
    fn test_dodged_bar_chart() {
        // Test dodged bar chart via position => 'dodge'
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 'X', 10),
                ('A', 'Y', 20),
                ('B', 'X', 15),
                ('B', 'Y', 25)
            ) AS t(cat, grp, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y, grp AS fill
            SETTING position => 'dodge'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify xOffset encoding exists (dodged bars use xOffset for displacement)
        let encoding = &layer["encoding"];
        assert!(
            encoding["xOffset"].is_object(),
            "Should have xOffset encoding for dodged bars. Encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );

        // Verify bar width uses bandwidth expression with adjusted_width for dodged bars
        // For 2 groups with default width 0.9: adjusted_width = 0.9 / 2 = 0.45
        let mark = &layer["mark"];
        let width_expr = mark["width"]["expr"].as_str();
        assert!(
            width_expr.is_some(),
            "Dodged bars should have expression-based width. Mark: {}",
            serde_json::to_string_pretty(mark).unwrap()
        );
        let expr = width_expr.unwrap();
        assert!(
            expr.contains("bandwidth('x')") && expr.contains("0.45"),
            "Width expression should use bandwidth('x') * adjusted_width, got: {}",
            expr
        );
    }

    #[test]
    fn test_position_identity_default() {
        // Test that identity position (default) doesn't modify data
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES
                ('A', 10),
                ('B', 20)
            ) AS t(cat, val)
            VISUALISE
            DRAW bar MAPPING cat AS x, val AS y
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);

        // Verify no xOffset encoding (identity position)
        let encoding = &layer["encoding"];
        assert!(
            encoding.get("xOffset").is_none(),
            "Identity position should not have xOffset encoding"
        );
    }

    #[test]
    fn test_label_with_flipped_project() {
        // End-to-end test: LABEL x/y with PROJECT y, x TO cartesian
        // Labels should be correctly applied to the flipped axes
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
            VISUALISE
            DRAW bar MAPPING x AS y, y AS x
            PROJECT y, x TO cartesian
            LABEL x => 'Value', y => 'Category'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];

        // With PROJECT y, x TO cartesian:
        // - y is pos1 (first position), renders to VL x-axis in cartesian
        // - x is pos2 (second position), renders to VL y-axis in cartesian
        // So LABEL y => 'Category' should appear on VL x-axis, LABEL x => 'Value' on VL y-axis
        let x_title = encoding["x"]["title"].as_str();
        let y_title = encoding["y"]["title"].as_str();

        assert_eq!(
            x_title,
            Some("Category"),
            "x-axis should have 'Category' title (from LABEL y). Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
        assert_eq!(
            y_title,
            Some("Value"),
            "y-axis should have 'Value' title (from LABEL x). Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }

    #[test]
    fn test_label_with_polar_project() {
        // End-to-end test: LABEL angle/radius with PROJECT TO polar
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let query = r#"
            SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
            VISUALISE value AS angle, category AS fill
            DRAW bar
            PROJECT TO polar
            LABEL angle => 'Angle', radius => 'Distance'
        "#;

        let spec = reader.execute(query).unwrap();
        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();

        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let layer = data_layer(&json, 0);
        let encoding = &layer["encoding"];

        // Verify theta encoding has the label
        let theta_title = encoding["theta"]["title"].as_str();
        assert_eq!(
            theta_title,
            Some("Angle"),
            "theta encoding should have 'Angle' title. Got encoding: {}",
            serde_json::to_string_pretty(encoding).unwrap()
        );
    }
}
