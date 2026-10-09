//! Data source abstraction layer for ggsql
//!
//! The reader module provides a pluggable interface for executing SQL queries
//! against various data sources and returning Polars DataFrames for visualization.
//!
//! # Architecture
//!
//! All readers implement the `Reader` trait, which provides:
//! - SQL query execution → DataFrame conversion
//! - Visualization query execution → ResolvedPlot
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

use crate::execute::{prepare_data_with_reader, resolve_table_with_reader};
use crate::parser::{self, SourceTree};
use crate::plot::{CastTargetType, Plot};
use crate::table::TableLayout;
use crate::validate::{validate, ValidationWarning};
use crate::{naming, DataFrame, GgsqlError, Result, Spec};

// =============================================================================
// SQL Dialect
// =============================================================================

/// SQL type names and functionality in the syntax supported by that backend.
///
/// Default implementations produce portable ANSI SQL.
///
/// # Quoting convention
///
/// Identifier parameters are **raw** (unquoted) unless the parameter name
/// says otherwise: parameters called `qcol`/`qname` or documented as
/// "already-quoted" arrive quoted by the caller. Methods receiving a raw
/// identifier must quote it with [`SqlDialect::quote_ident`] before
/// interpolating. SQL *fragments* (expressions, `from` arguments) arrive
/// fully composed; `from` arguments follow the
/// [`crate::sql::FromItem::Fragment`] contract — a bare (possibly quoted)
/// table/CTE name or an already-parenthesized relation, never a raw query.
pub trait SqlDialect {
    // =====================================================================
    // Capabilities and self-knowledge
    // =====================================================================

    /// Whether this backend supports spatial (geometry) operations.
    ///
    /// Default is `false`: most backends are not PostGIS-compatible, and an
    /// opt-in means an unsupported backend fails fast with a clear "spatial
    /// not supported on this backend" error rather than emitting spatial SQL
    /// it cannot run. Dialects opt in by returning `true`.
    ///
    /// Opting in commits the dialect to the whole `sql_st_*` surface below —
    /// predicates, validity repair, collection extraction, SRID handling —
    /// plus [`sql_spatial_setup`] when an extension must be loaded first.
    /// The demanding requirement is reprojection ([`sql_st_transform`]):
    /// several backends ship geometry types and predicates but no CRS
    /// transforms, and rendering a map in the wrong CRS is worse than
    /// refusing to render one, so those backends keep returning `false`
    /// even though basic spatial SQL would run.
    ///
    /// [`sql_spatial_setup`]: SqlDialect::sql_spatial_setup
    /// [`sql_st_transform`]: SqlDialect::sql_st_transform
    fn supports_spatial(&self) -> bool {
        false
    }

    /// Whether [`sql_limit`](SqlDialect::sql_limit) wraps the query in a
    /// derived table (T-SQL's `SELECT TOP n * FROM (…)`, Oracle's ROWNUM
    /// wrap) rather than appending a clause (`LIMIT n`). Callers use this
    /// to keep ORDER BY out of the derived table, where T-SQL rejects it
    /// (error 1033). Default `false`; wrapper-style dialects must override.
    fn sql_limit_wraps_query(&self) -> bool {
        false
    }

    /// Whether a `WITH` clause may appear inside a parenthesized derived
    /// table. T-SQL forbids it ("Incorrect syntax near the keyword
    /// 'WITH'"), so [`crate::sql`] hoists leading CTEs out of the derived
    /// table for dialects returning `false`.
    fn allows_cte_in_derived_table(&self) -> bool {
        true
    }

    /// Whether this backend fundamentally lacks the temporary-table support
    /// ggsql needs to stage internal tables (CTEs, stat transforms), so a
    /// connection through it must always be wrapped in a caching reader.
    ///
    /// Set this for query engines with no DDL at all (Druid, Drill)
    /// or where `CREATE TEMP TABLE` is broadly unsupported
    /// (Trino). Backends whose support is merely *uncertain* — e.g. the
    /// account may be read-only — should keep the default: connections are
    /// probed once on connect and wrapped only when the probe fails.
    fn requires_cache(&self) -> bool {
        false
    }

    /// Whether result batches should have ISO-8601 date/datetime strings
    /// sniffed into temporal Arrow types at the reader boundary.
    ///
    /// Default `false`: a backend's VARCHAR is a real string type and must
    /// not be reinterpreted. Opt in only for drivers that lose temporal type
    /// information entirely — e.g. the prerelease Druid Foundry driver
    /// surfaces Druid's LONG-based timestamps as plain strings/epoch
    /// integers — where an ISO-looking string is almost certainly a date
    /// the driver failed to type. Mirrors the sqlite reader's value
    /// sniffing, which exists for the same reason (sqlite has no temporal
    /// storage types).
    fn sniff_temporal_strings(&self) -> bool {
        false
    }

    /// How this backend (re)creates a temporary table holding a query
    /// result. Drives the default [`create_or_replace_temp_table_sql`];
    /// dialects pick a variant instead of overriding that method.
    ///
    /// [`create_or_replace_temp_table_sql`]: SqlDialect::create_or_replace_temp_table_sql
    fn temp_table_style(&self) -> TempTableStyle {
        TempTableStyle::DropThenCreateTemp
    }

    // =====================================================================
    // Types and identifiers
    // =====================================================================

    /// SQL type names for table creation and casts. Dialects override this
    /// one method with a [`TypeNames`] literal; call sites read individual
    /// fields off it (`dialect.type_names().number`, …).
    fn type_names(&self) -> TypeNames {
        TypeNames::ANSI
    }

    /// Get the SQL type name for a cast target type.
    fn type_name_for(&self, target: CastTargetType) -> Option<&str> {
        let names = self.type_names();
        match target {
            CastTargetType::Number => names.number,
            CastTargetType::Integer => names.integer,
            CastTargetType::Date => names.date,
            CastTargetType::DateTime => names.datetime,
            CastTargetType::Time => names.time,
            CastTargetType::String => names.string,
            CastTargetType::Boolean => names.boolean,
        }
    }

    /// Quote an identifier using this backend's convention.
    ///
    /// Default is SQL-standard double quotes. Override for backends with a
    /// different quoting convention (e.g. backticks for MySQL/ClickHouse).
    fn quote_ident(&self, name: &str) -> String {
        naming::quote_ident(name)
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

    /// Table-alias clause for a FROM item.
    ///
    /// Default is the ANSI `AS "alias"`. Oracle rejects `AS` before table
    /// aliases and emits just the quoted alias.
    fn sql_table_alias(&self, alias: &str) -> String {
        format!("AS {}", self.quote_ident(alias))
    }

    /// How `*` is emitted in a select list that also contains other items,
    /// when selecting from a derived table with this alias.
    ///
    /// Default is a bare `*`. Oracle rejects an unqualified `*` alongside
    /// other items and requires `alias.*`. (A select list consisting only
    /// of `*` is valid everywhere and does not go through this hook.)
    fn sql_select_star(&self, table_alias: &str) -> String {
        let _ = table_alias;
        "*".to_string()
    }

    // =====================================================================
    // Scalar SQL translations
    // =====================================================================

    /// Scalar MAX across any number of SQL expressions.
    ///
    /// Default uses the `GREATEST` function, supported by most backends.
    /// Backends without it (SQL Server, SQLite, Druid, Drill, MonetDB)
    /// override with the `dialects::case_greatest` helper.
    fn sql_greatest(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("GREATEST({})", exprs.join(", "))
    }

    /// Scalar MIN across any number of SQL expressions.
    ///
    /// Default uses the `LEAST` function; see [`sql_greatest`].
    ///
    /// [`sql_greatest`]: SqlDialect::sql_greatest
    fn sql_least(&self, exprs: &[&str]) -> String {
        if exprs.len() == 1 {
            return exprs[0].to_string();
        }
        format!("LEAST({})", exprs.join(", "))
    }

    /// Ceiling of a numeric expression.
    ///
    /// ANSI `CEIL`; SQL Server only has `CEILING`.
    fn sql_ceil(&self, expr: &str) -> String {
        format!("CEIL({expr})")
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

    /// Compute a quantile of a column, as a SELECT-list item.
    ///
    /// `column` is the raw (unquoted) column name; the dialect quotes it.
    /// The result must be valid as an item in the SELECT list of a query
    /// `SELECT ... FROM (<from>) AS "__ggsql_qt__" [WHERE ...] GROUP BY
    /// <groups>` — every caller (boxplot, density, aggregate stats) embeds it
    /// in exactly that shape. Dialects with a native quantile aggregate
    /// return it directly (it computes per group, so `from`/`groups` go
    /// unused); the default instead builds a correlated scalar subquery.
    ///
    /// The default implements `percentile_cont` semantics with window
    /// functions only (no native quantile aggregate required): with
    /// `x = fraction * (cnt - 1)`, the result interpolates linearly between
    /// the rows ranked `floor(x) + 1` and `ceil(x) + 1`, which is exact for
    /// every fraction.
    fn sql_quantile(
        &self,
        column: &str,
        fraction: f64,
        from: crate::sql::FromItem<'_>,
        groups: &[String],
    ) -> String {
        // The correlation predicate references the enclosing query's alias
        // and this scalar subquery's own alias, so both are needed quoted.
        let __ggsql_qt__ = self.quote_ident("__ggsql_qt__");
        let __ggsql_tile__ = self.quote_ident("__ggsql_tile__");
        let quoted_column = self.quote_ident(column);

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
            (String::new(), String::new(), None)
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
                Some(filter),
            )
        };

        let inner = crate::sql::Select::new(self)
            .select(format!(
                "{quoted_column} AS __val, \
                 ROW_NUMBER() OVER ({partition_by}ORDER BY {quoted_column}) AS rn, \
                 COUNT(*) OVER ({partition_by}) AS cnt{group_cols}"
            ))
            .from_aliased(from, "__ggsql_pct__")
            .and_where(format!("{quoted_column} IS NOT NULL"))
            .build();

        let mut outer = crate::sql::Select::new(self)
            .select(format!(
                "MAX(CASE WHEN rn = {lo} THEN __val END) + \
                 (MAX(CASE WHEN rn = {hi} THEN __val END) - \
                  MAX(CASE WHEN rn = {lo} THEN __val END)) * MAX({frac})"
            ))
            .from_aliased(crate::sql::FromItem::Query(&inner), "__ggsql_tile__");
        if let Some(filter) = group_filter {
            outer = outer.and_where(filter);
        }
        format!("({})", outer.build())
    }

    /// SQL fragment for a simple aggregate function applied to an
    /// already-quoted column expression.
    ///
    /// Returns `Some(expr)` when the dialect can express this aggregate inline
    /// in a `GROUP BY` query. Returns `None` when the aggregate is not
    /// supported by this backend; the stat layer surfaces a clear error.
    ///
    /// Names handled here are the entries of `stat_aggregate::AGG_NAMES` other
    /// than the percentile/iqr family, which goes through [`sql_quantile`]
    /// instead.
    ///
    /// [`sql_quantile`]: SqlDialect::sql_quantile
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
        // Clamp out-of-range input rather than panicking in library code.
        let day = (days_since_epoch as i64)
            .saturating_add(719163)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let date = chrono::NaiveDate::from_num_days_from_ce_opt(day).unwrap_or({
            if day < 0 {
                chrono::NaiveDate::MIN
            } else {
                chrono::NaiveDate::MAX
            }
        });
        format!("DATE '{}'", date.format("%Y-%m-%d"))
    }

    /// SQL literal for a datetime value (microseconds since Unix epoch).
    fn sql_datetime_literal(&self, microseconds_since_epoch: i64) -> String {
        // Clamp out-of-range input rather than panicking in library code.
        let dt = chrono::DateTime::from_timestamp_micros(microseconds_since_epoch)
            .map(|d| d.naive_utc())
            .unwrap_or(if microseconds_since_epoch < 0 {
                chrono::NaiveDateTime::MIN
            } else {
                chrono::NaiveDateTime::MAX
            });
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
        // Clamp into a representable time-of-day rather than letting
        // negative or ≥ 24h input wrap the `as u32` cast and panic.
        let clamped = nanoseconds_since_midnight.clamp(0, 86_399_999_999_999);
        let seconds = clamped.div_euclid(1_000_000_000);
        let nanos = clamped.rem_euclid(1_000_000_000);
        let time =
            chrono::NaiveTime::from_num_seconds_from_midnight_opt(seconds as u32, nanos as u32)
                .expect("time literal is in range after clamping");
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
        let ty = self.type_names().number.unwrap_or("DOUBLE PRECISION");
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

    // =====================================================================
    // Statement generators and DDL
    // =====================================================================

    /// Append a row limit to a query.
    ///
    /// Default uses `LIMIT n`. Override for backends with different limit
    /// syntax (e.g. SQL Server's `TOP`, Oracle's `FETCH FIRST`).
    fn sql_limit(&self, query: &str, n: usize) -> String {
        format!("{} LIMIT {}", query, n)
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
        if expr == col {
            return crate::sql::wrap_all(self, from, "__ggsql_sr__");
        }
        if all_columns.is_empty() {
            // The replacement expression comes first so its occurrence of a
            // duplicated column wins over the star's.
            return crate::sql::Select::new(self)
                .select_plus_star(&[format!("{expr} AS {col}")], "__ggsql_sr__")
                .from_aliased(crate::sql::FromItem::Query(from), "__ggsql_sr__")
                .build();
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
        crate::sql::select_from(
            self,
            &select_list.join(", "),
            crate::sql::FromItem::Query(from),
            "__ggsql_sr__",
        )
    }

    /// Keyword introducing a CTE block that contains recursive CTEs.
    ///
    /// ANSI/Postgres/MySQL accept `WITH RECURSIVE`; T-SQL and Oracle use
    /// plain `WITH`, where recursion is implied by self-reference.
    fn sql_with_recursive(&self) -> &'static str {
        "WITH RECURSIVE"
    }

    /// Returns CTE fragment(s) producing table `__ggsql_seq__` with column `n`,
    /// holding the integers 0..n-1.
    fn sql_generate_series(&self, n: usize) -> String {
        recursive_series_cte(self, n, "REAL")
    }

    /// SQL listing catalogs, with a single `catalog_name` output column.
    ///
    /// Default queries `information_schema`; override for backends without it
    /// (e.g. Exasol's `SYS.EXA_*` tables).
    fn sql_list_catalogs(&self) -> String {
        "SELECT DISTINCT catalog_name FROM information_schema.schemata ORDER BY catalog_name"
            .to_string()
    }

    /// SQL listing schemas in `catalog`, with a single `schema_name` column.
    fn sql_list_schemas(&self, catalog: &str) -> String {
        format!(
            "SELECT DISTINCT schema_name FROM information_schema.schemata \
             WHERE catalog_name = {} ORDER BY schema_name",
            naming::quote_literal(catalog)
        )
    }

    /// SQL listing tables in `catalog`/`schema`, with `table_name` and
    /// `table_type` output columns.
    fn sql_list_tables(&self, catalog: &str, schema: &str) -> String {
        format!(
            "SELECT DISTINCT table_name, table_type FROM information_schema.tables \
             WHERE table_catalog = {} AND table_schema = {} ORDER BY table_name",
            naming::quote_literal(catalog),
            naming::quote_literal(schema)
        )
    }

    /// SQL listing columns of `catalog`/`schema`/`table`, with `column_name`
    /// and `data_type` output columns.
    fn sql_list_columns(&self, catalog: &str, schema: &str, table: &str) -> String {
        format!(
            "SELECT column_name, data_type FROM information_schema.columns \
             WHERE table_catalog = {} AND table_schema = {} AND table_name = {} \
             ORDER BY ordinal_position",
            naming::quote_literal(catalog),
            naming::quote_literal(schema),
            naming::quote_literal(table)
        )
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
    /// Drop a table ggsql created (materialized internal table or registered
    /// data), best effort: the statement must not fail when the table is
    /// absent. Default is `DROP TABLE IF EXISTS`; Oracle overrides with a
    /// PL/SQL-guarded drop (it has no `IF EXISTS`).
    fn drop_table_sql(&self, name: &str) -> String {
        format!("DROP TABLE IF EXISTS {}", self.quote_ident(name))
    }

    /// Column aliases from `WITH t(a, b) AS (...)` are preserved portably by
    /// wrapping the body in a named CTE with a column alias list, so the
    /// backend never needs to support `CREATE TABLE t(a, b) AS ...` syntax.
    ///
    /// The statement shapes are data-driven via [`temp_table_style`]; the
    /// default implementation covers every style, so dialects should not
    /// need to override this method.
    ///
    /// Returned statements must be executed in order via `Reader::execute_sql`.
    ///
    /// [`temp_table_style`]: SqlDialect::temp_table_style
    fn create_or_replace_temp_table_sql(
        &self,
        name: &str,
        column_aliases: &[String],
        body_sql: &str,
    ) -> Vec<String> {
        let qname = self.quote_ident(name);
        let body =
            wrap_with_column_aliases(&|c: &str| self.quote_ident(c), body_sql, column_aliases);
        match self.temp_table_style() {
            TempTableStyle::DropThenCreateTemp => vec![
                format!("DROP TABLE IF EXISTS {}", qname),
                format!("CREATE TEMP TABLE {} AS {}", qname, body),
            ],
            TempTableStyle::CreateOrReplaceTemp => {
                vec![format!(
                    "CREATE OR REPLACE TEMP TABLE {} AS {}",
                    qname, body
                )]
            }
            TempTableStyle::DropTemporaryThenCreateTemp => vec![
                format!("DROP TEMPORARY TABLE IF EXISTS {}", qname),
                format!("CREATE TEMPORARY TABLE {} AS {}", qname, body),
            ],
            TempTableStyle::DropThenCreateTempPreserveRows => vec![
                format!("DROP TABLE IF EXISTS {}", qname),
                format!(
                    "CREATE TEMP TABLE {} AS {} ON COMMIT PRESERVE ROWS",
                    qname, body
                ),
            ],
            TempTableStyle::CreateOrReplaceTempView => {
                vec![format!("CREATE OR REPLACE TEMP VIEW {} AS {}", qname, body)]
            }
            TempTableStyle::DropThenCreate => vec![
                format!("DROP TABLE IF EXISTS {}", qname),
                format!("CREATE TABLE {} AS {}", qname, body),
            ],
            TempTableStyle::GuardedDropThenCreate => vec![
                self.drop_table_sql(name),
                format!("CREATE TABLE {} AS {}", qname, body),
            ],
            TempTableStyle::SelectInto => {
                // Session-local `#temp` tables would be ideal, but ggsql
                // references materialized tables by their plain (quoted) name
                // in later statements, which `#name` would break — so regular
                // tables + DROP cleanup.
                let __ggsql_src__ = self.quote_ident("__ggsql_src__");
                vec![
                    format!("DROP TABLE IF EXISTS {}", qname),
                    format!("SELECT * INTO {} FROM ({}) AS {__ggsql_src__}", qname, body),
                ]
            }
        }
    }

    // =====================================================================
    // Spatial SQL
    // =====================================================================

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

    /// SQL expression constructing a point geometry from x/y expressions.
    ///
    /// Default is the PostGIS-style `ST_Point`. Override for backends with
    /// different constructors (e.g. SpatiaLite's `MakePoint`).
    fn sql_st_point(&self, x: &str, y: &str) -> String {
        format!("ST_Point({x}, {y})")
    }

    /// SQL predicate: whether geometry `a` contains geometry `b`.
    fn sql_st_contains(&self, a: &str, b: &str) -> String {
        format!("ST_Contains({a}, {b})")
    }

    /// SQL predicate: whether geometries `a` and `b` intersect.
    fn sql_st_intersects(&self, a: &str, b: &str) -> String {
        format!("ST_Intersects({a}, {b})")
    }

    /// SQL expression constructing a geometry from WKT. `wkt` is the raw
    /// WKT content; the hook adds the string-literal quoting.
    fn sql_geom_from_text(&self, wkt: &str) -> String {
        format!("ST_GeomFromText('{wkt}')")
    }

    /// SQL expression returning the SRID of a geometry.
    fn sql_st_srid(&self, geom: &str) -> String {
        format!("ST_SRID({geom})")
    }

    /// SQL expression converting a geometry to WKT text.
    fn sql_st_as_text(&self, geom: &str) -> String {
        format!("ST_AsText({geom})")
    }

    /// SQL expression for the difference of two geometries (`a` minus `b`).
    fn sql_st_difference(&self, a: &str, b: &str) -> String {
        format!("ST_Difference({a}, {b})")
    }

    /// SQL expression for the intersection of two geometries.
    fn sql_st_intersection(&self, a: &str, b: &str) -> String {
        format!("ST_Intersection({a}, {b})")
    }

    /// SQL expression repairing an invalid geometry.
    fn sql_st_make_valid(&self, geom: &str) -> String {
        format!("ST_MakeValid({geom})")
    }

    /// SQL expression extracting geometries of one type from a collection
    /// (`type_index`: 1 = points, 2 = linestrings, 3 = polygons).
    fn sql_st_collection_extract(&self, geom: &str, type_index: u32) -> String {
        format!("ST_CollectionExtract({geom}, {type_index})")
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
        let extent = crate::sql::Select::new(self)
            .select(format!("ST_Extent({column}) AS ext"))
            .from(crate::sql::FromItem::Fragment(from))
            .build();
        crate::sql::select_from(
            self,
            "ST_XMin(ext) AS xmin, ST_YMin(ext) AS ymin, \
             ST_XMax(ext) AS xmax, ST_YMax(ext) AS ymax",
            crate::sql::FromItem::Query(&extent),
            "__ggsql_ext__",
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
}

/// SQL type names a dialect uses for table creation and casts. `None` means
/// the backend has no native type for that kind (e.g. ClickHouse TIME);
/// callers surface a clear error. Override per field against
/// [`TypeNames::ANSI`]:
///
/// ```ignore
/// fn type_names(&self) -> TypeNames {
///     TypeNames { number: Some("DOUBLE"), ..TypeNames::ANSI }
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TypeNames {
    pub number: Option<&'static str>,
    pub integer: Option<&'static str>,
    pub date: Option<&'static str>,
    pub datetime: Option<&'static str>,
    pub time: Option<&'static str>,
    pub string: Option<&'static str>,
    pub boolean: Option<&'static str>,
}

impl TypeNames {
    /// ANSI defaults, close to PostgreSQL.
    pub const ANSI: Self = Self {
        number: Some("DOUBLE PRECISION"),
        integer: Some("BIGINT"),
        date: Some("DATE"),
        datetime: Some("TIMESTAMP"),
        time: Some("TIME"),
        string: Some("VARCHAR"),
        boolean: Some("BOOLEAN"),
    };
}

/// How a backend (re)creates a temporary table holding a query result; see
/// [`SqlDialect::temp_table_style`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempTableStyle {
    /// `DROP TABLE IF EXISTS` then `CREATE TEMP TABLE AS`.
    DropThenCreateTemp,
    /// Single `CREATE OR REPLACE TEMP TABLE AS` (BigQuery, DuckDB).
    CreateOrReplaceTemp,
    /// `DROP TEMPORARY TABLE IF EXISTS` then `CREATE TEMPORARY TABLE AS`
    /// (MySQL/MariaDB, ClickHouse).
    DropTemporaryThenCreateTemp,
    /// `DROP TABLE IF EXISTS` then `CREATE TEMP TABLE AS … ON COMMIT
    /// PRESERVE ROWS` (MonetDB temp tables default to ON COMMIT DELETE
    /// ROWS: with ODBC autocommit the CTAS statement's own commit would
    /// wipe the rows it just staged).
    DropThenCreateTempPreserveRows,
    /// Single `CREATE OR REPLACE TEMP VIEW AS` (Databricks/Spark).
    CreateOrReplaceTempView,
    /// `DROP TABLE IF EXISTS` then plain `CREATE TABLE AS` — no temp-table
    /// support (DataFusion).
    DropThenCreate,
    /// PL/SQL-guarded `DROP TABLE` then plain `CREATE TABLE AS` (Oracle has
    /// no `DROP TABLE IF EXISTS`).
    GuardedDropThenCreate,
    /// `DROP TABLE IF EXISTS` then `SELECT * INTO` (SQL Server has no
    /// `CREATE TABLE AS`).
    SelectInto,
}

/// Emulate `GENERATE_SERIES(0, n - 1)` for backends that lack it.
///
/// Cube-root-decomposed recursive series CTE: recurses only ~cbrt(n) times,
/// then cross-joins three copies to cover the full range. Shared by the
/// default [`SqlDialect::sql_generate_series`] and dialects that differ only
/// in the float cast target (MySQL/MariaDB: `DOUBLE` — their `CAST` has no
/// `REAL` target).
pub(crate) fn recursive_series_cte<D: SqlDialect + ?Sized>(
    dialect: &D,
    n: usize,
    real_cast: &str,
) -> String {
    let base_size = (n as f64).cbrt().ceil() as usize;
    let base_sq = base_size * base_size;
    let base_max = base_size - 1;
    let __ggsql_base__ = dialect.quote_ident("__ggsql_base__");
    let __ggsql_seq__ = dialect.quote_ident("__ggsql_seq__");
    format!(
        "{__ggsql_base__}(n) AS (\
           SELECT 0 UNION ALL SELECT n + 1 FROM {__ggsql_base__} WHERE n < {base_max}\
         ),\
         {__ggsql_seq__}(n) AS (\
           SELECT CAST(a.n * {base_sq} + b.n * {base_size} + c.n AS {real_cast}) AS n \
           FROM {__ggsql_base__} a, {__ggsql_base__} b, {__ggsql_base__} c \
           WHERE a.n * {base_sq} + b.n * {base_size} + c.n < {n}\
         )"
    )
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

pub use dialects::AnsiDialect;

/// A shared dialect instance. Dialects are stateless unit structs, so
/// registry lookups hand out `'static` references rather than boxing a
/// fresh trait object per call.
pub type DialectRef = &'static (dyn SqlDialect + Send + Sync);

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

pub mod cache;

#[cfg(all(test, feature = "duckdb", feature = "sqlite"))]
mod cache_equivalence;

pub mod connection;
pub mod data;
pub mod dialects;
pub mod registry;
mod spec;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBReader;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteReader;

#[cfg(feature = "odbc")]
pub use odbc::OdbcReader;

#[cfg(feature = "adbc")]
pub use adbc::AdbcReader;

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

// ============================================================================
// Shared reader helpers
// ============================================================================

/// Column type for `register` DDL: the dialect's own type names for each
/// Arrow type (so e.g. Oracle gets `VARCHAR2` rather than the nonexistent
/// `TEXT`), with portable fallbacks when a dialect doesn't name one.
///
/// Errors on Arrow types with no sensible column type rather than silently
/// mapping them to TEXT. SQLite does not use this — its storage classes and
/// parameter binding need their own mapping (`arrow_type_to_sqlite`).
#[cfg(any(feature = "adbc", feature = "odbc"))]
pub(crate) fn register_column_type(
    dialect: &dyn SqlDialect,
    dtype: &arrow::datatypes::DataType,
) -> Result<String> {
    use arrow::datatypes::DataType;

    let name = match dtype {
        DataType::Boolean => dialect.type_names().boolean.unwrap_or("BOOLEAN"),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => dialect.type_names().integer.unwrap_or("BIGINT"),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            dialect.type_names().number.unwrap_or("DOUBLE PRECISION")
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            dialect.type_names().string.unwrap_or("VARCHAR")
        }
        DataType::Date32 | DataType::Date64 => dialect.type_names().date.unwrap_or("DATE"),
        DataType::Timestamp(_, _) => dialect.type_names().datetime.unwrap_or("TIMESTAMP"),
        DataType::Time32(_) | DataType::Time64(_) => dialect.type_names().time.unwrap_or("TIME"),
        other => {
            return Err(GgsqlError::ReaderError(format!(
                "register: unsupported Arrow type for table DDL: {other:?}"
            )))
        }
    };
    Ok(name.to_string())
}

/// Build a `CREATE TABLE <name> (col TYPE, …)` statement from an Arrow
/// schema, with column types from [`register_column_type`].
#[cfg(feature = "adbc")]
pub(crate) fn create_table_sql(
    name: &str,
    schema: &arrow::datatypes::Schema,
    dialect: &dyn SqlDialect,
) -> Result<String> {
    let mut cols: Vec<String> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let ty = register_column_type(dialect, field.data_type())
            .map_err(|e| GgsqlError::ReaderError(format!("column '{}': {}", field.name(), e)))?;
        cols.push(format!("{} {}", dialect.quote_ident(field.name()), ty));
    }
    Ok(format!(
        "CREATE TABLE {} ({})",
        dialect.quote_ident(name),
        cols.join(", ")
    ))
}

/// Normalize a query-result batch to the types downstream ggsql code
/// standardizes on: Decimal128 columns become Float64 (typed accessors are
/// numeric) and the string family (LargeUtf8, Utf8View) becomes plain Utf8
/// (typed accessors standardize on `StringArray`). One conversion point at
/// the reader boundary beats teaching every consumer about these types.
///
/// Zero-cost when no column needs conversion.
#[cfg(any(feature = "adbc", feature = "duckdb"))]
pub(crate) fn normalize_result_batch(
    batch: arrow::record_batch::RecordBatch,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = batch.schema();
    let target_for = |dtype: &DataType| match dtype {
        DataType::Decimal128(_, _) => Some(DataType::Float64),
        DataType::LargeUtf8 | DataType::Utf8View => Some(DataType::Utf8),
        // JDBC-style drivers (e.g. the Druid Foundry driver) surface DATE
        // as Date64; Date32 is the date type the whole pipeline keys on.
        DataType::Date64 => Some(DataType::Date32),
        _ => None,
    };
    if !schema
        .fields()
        .iter()
        .any(|f| target_for(f.data_type()).is_some())
    {
        return Ok(batch);
    }

    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        match target_for(field.data_type()) {
            Some(target) => {
                let casted = arrow::compute::cast(batch.column(i), &target).map_err(|e| {
                    GgsqlError::ReaderError(format!(
                        "Failed to normalize column '{}' from {:?} to {:?}: {}",
                        field.name(),
                        field.data_type(),
                        target,
                        e
                    ))
                })?;
                fields.push(Arc::new(Field::new(
                    field.name(),
                    target,
                    field.is_nullable(),
                )));
                columns.push(casted);
            }
            None => {
                fields.push(field.clone());
                columns.push(batch.column(i).clone());
            }
        }
    }
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        columns,
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to normalize result batch: {e}")))
}

/// Recover temporal columns from ISO-8601 strings for drivers that lose
/// temporal type information (see [`SqlDialect::sniff_temporal_strings`]).
///
/// Every Utf8 column is probed: if all non-null values parse as ISO dates
/// (`YYYY-MM-DD`) it becomes Date32; otherwise if all parse as ISO
/// datetimes (`YYYY-MM-DD` + `T`/space + time) it becomes Timestamp(µs).
/// Anything else is left untouched. Columns with no non-null values are
/// left alone — there is nothing to infer from.
#[cfg(feature = "adbc")]
pub(crate) fn sniff_temporal_strings_in_batch(
    batch: arrow::record_batch::RecordBatch,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::array::{Array, ArrayRef, Date32Array, StringArray, TimestampMicrosecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use chrono::Datelike;
    use std::sync::Arc;

    const EPOCH_DAYS_FROM_CE: i32 = 719_163;

    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Utf8))
    {
        return Ok(batch);
    }

    let parse_column = |array: &StringArray| -> Option<(ArrayRef, DataType)> {
        let mut saw_value = false;
        let mut dates: Vec<Option<i32>> = Vec::with_capacity(array.len());
        let mut datetimes: Vec<Option<i64>> = Vec::with_capacity(array.len());
        let mut date_ok = true;
        let mut datetime_ok = true;
        for value in array.iter() {
            let Some(s) = value else {
                dates.push(None);
                datetimes.push(None);
                continue;
            };
            saw_value = true;
            if date_ok {
                match chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                    Ok(d) => dates.push(Some(d.num_days_from_ce() - EPOCH_DAYS_FROM_CE)),
                    Err(_) => date_ok = false,
                }
            }
            if datetime_ok {
                // Accept a 'T' or space separator, fractional seconds, and a
                // trailing 'Z' — the shapes ISO-8601 serializations take.
                let stripped = s.strip_suffix('Z').unwrap_or(s);
                let parsed = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"]
                    .iter()
                    .find_map(|fmt| chrono::NaiveDateTime::parse_from_str(stripped, fmt).ok());
                match parsed {
                    Some(dt) => datetimes.push(Some(dt.and_utc().timestamp_micros())),
                    None => datetime_ok = false,
                }
            }
            if !date_ok && !datetime_ok {
                return None;
            }
        }
        if !saw_value {
            return None;
        }
        if date_ok {
            return Some((
                Arc::new(Date32Array::from(dates)) as ArrayRef,
                DataType::Date32,
            ));
        }
        if datetime_ok {
            return Some((
                Arc::new(TimestampMicrosecondArray::from(datetimes)) as ArrayRef,
                DataType::Timestamp(TimeUnit::Microsecond, None),
            ));
        }
        None
    };

    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let sniffed = match field.data_type() {
            DataType::Utf8 => batch
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .and_then(parse_column),
            _ => None,
        };
        match sniffed {
            Some((array, dtype)) => {
                fields.push(Arc::new(Field::new(
                    field.name(),
                    dtype,
                    field.is_nullable(),
                )));
                columns.push(array);
            }
            None => {
                fields.push(field.clone());
                columns.push(batch.column(i).clone());
            }
        }
    }
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        columns,
    )
    .map_err(|e| GgsqlError::ReaderError(format!("Failed to sniff temporal strings: {e}")))
}

/// Registered-table bookkeeping shared by the concrete readers: a set of
/// names registered through this reader, so `unregister` can reject tables
/// it doesn't own and readers can answer `is_registered`.
pub(crate) struct RegisteredTables(std::cell::RefCell<std::collections::HashSet<String>>);

impl RegisteredTables {
    pub(crate) fn new() -> Self {
        Self(std::cell::RefCell::new(std::collections::HashSet::new()))
    }

    pub(crate) fn note_registered(&self, name: &str) {
        self.0.borrow_mut().insert(name.to_string());
    }

    pub(crate) fn note_unregistered(&self, name: &str) {
        self.0.borrow_mut().remove(name);
    }

    pub(crate) fn is_registered(&self, name: &str) -> bool {
        self.0.borrow().contains(name)
    }

    /// Snapshot of all registered names.
    pub(crate) fn names(&self) -> Vec<String> {
        self.0.borrow().iter().cloned().collect()
    }
}

impl Default for RegisteredTables {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared test helpers for reader equivalence suites.
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(all(test, feature = "duckdb"))]
mod helper_tests {
    use super::*;
    use arrow::array::{Array, Decimal128Array, StringViewArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn normalize_result_batch_casts_decimal_and_stringview() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Decimal128(10, 2), true),
            Field::new("s", DataType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    Decimal128Array::from(vec![Some(12345)])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                Arc::new(StringViewArray::from(vec!["x"])),
            ],
        )
        .unwrap();

        let out = normalize_result_batch(batch).unwrap();
        assert_eq!(out.column(0).data_type(), &DataType::Float64);
        assert_eq!(out.column(1).data_type(), &DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert!((col.value(0) - 123.45).abs() < 1e-9);
    }

    #[test]
    fn normalize_result_batch_casts_date64_to_date32() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Date64Array::from(vec![
                Some(19000 * 86_400_000),
                None,
            ]))],
        )
        .unwrap();

        let out = normalize_result_batch(batch).unwrap();
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Date32Array>()
            .expect("Date64 should normalize to Date32");
        assert_eq!(col.value(0), 19000);
        assert!(col.is_null(1));
    }

    #[test]
    fn normalize_result_batch_passthrough_when_nothing_to_do() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int64Array::from(vec![1, 2]))],
        )
        .unwrap();
        let out = normalize_result_batch(batch).unwrap();
        assert_eq!(out.column(0).data_type(), &DataType::Int64);
    }

    #[test]
    fn sniff_temporal_strings_recovers_dates_and_datetimes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Utf8, true),
            Field::new("ts", DataType::Utf8, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec![
                    Some("2021-01-01"),
                    None,
                    Some("2021-01-03"),
                ])),
                Arc::new(arrow::array::StringArray::from(vec![
                    Some("2021-01-01T00:00:00.000Z"),
                    Some("2021-01-02 12:30:00"),
                    None,
                ])),
                Arc::new(arrow::array::StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("c"),
                ])),
            ],
        )
        .unwrap();

        let out = sniff_temporal_strings_in_batch(batch).unwrap();
        let dates = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Date32Array>()
            .expect("ISO date strings should sniff to Date32");
        assert_eq!(dates.value(0), 18628); // 2021-01-01
        assert!(dates.is_null(1));
        assert_eq!(dates.value(2), 18630);
        let micros = out
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
            .expect("ISO datetime strings should sniff to Timestamp(µs)");
        assert_eq!(micros.value(0), 1_609_459_200_000_000);
        assert_eq!(micros.value(1), 1_609_590_600_000_000);
        assert!(micros.is_null(2));
        assert_eq!(out.column(2).data_type(), &DataType::Utf8);
    }

    #[test]
    fn sniff_temporal_strings_leaves_partial_and_empty_columns_alone() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("mixed", DataType::Utf8, true),
            Field::new("nulls", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec![
                    Some("2021-01-01"),
                    Some("not a date"),
                ])),
                Arc::new(arrow::array::StringArray::from(vec![
                    None::<&str>,
                    None::<&str>,
                ])),
            ],
        )
        .unwrap();

        let out = sniff_temporal_strings_in_batch(batch).unwrap();
        assert_eq!(out.column(0).data_type(), &DataType::Utf8);
        assert_eq!(out.column(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn registered_tables_tracks_names() {
        let t = RegisteredTables::new();
        assert!(!t.is_registered("a"));
        t.note_registered("a");
        assert!(t.is_registered("a"));
        assert_eq!(t.names(), vec!["a".to_string()]);
        t.note_unregistered("a");
        assert!(!t.is_registered("a"));
    }
}

// ============================================================================
// ResolvedPlot - Result of reader.execute()
// ============================================================================

/// Result of executing a ggsql query, ready for rendering.
pub struct ResolvedPlot {
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
// ResolvedTable - Result of reader.execute() for a TABULATE statement
// ============================================================================

/// Result of executing a ggsql TABULATE query, ready for rendering.
pub struct ResolvedTable {
    /// The resolved layout (cells, columns, rows), built from
    /// `table.source` (or the main SQL if there was no TABULATE FROM) and
    /// validated before handoff. See `TableLayout` for the shape invariants
    /// it maintains.
    pub(crate) layout: TableLayout,
    /// The SQL query that was executed to produce the layout's cells
    pub(crate) sql: String,
    /// Validation warnings from preparation
    pub(crate) warnings: Vec<ValidationWarning>,
}

// ============================================================================
// ResolvedSpec - Result of reader.execute()
// ============================================================================

/// Result of executing a ggsql query: either a resolved plot or a resolved
/// table, mirroring the parse-time `Spec` (`Plot` or `Table`).
pub enum ResolvedSpec {
    // Both boxed, mirroring `Spec`'s own two variants: whichever of
    // `ResolvedPlot`/`ResolvedTable` is larger, an unboxed size gap between
    // them trips clippy's `large_enum_variant` lint.
    Plot(Box<ResolvedPlot>),
    Table(Box<ResolvedTable>),
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

    /// Execute a ggsql query and return the resolved specification.
    ///
    /// This is the main entry point for creating visualizations or tables.
    /// It parses the query, executes the SQL portion, and returns a
    /// `ResolvedSpec` ready for rendering — either a `ResolvedPlot` (from a
    /// `VISUALISE`) or a `ResolvedTable` (from a `TABULATE`).
    ///
    /// No default body: implementations delegate to `execute_with_reader`
    /// (each with a concrete, `Sized` `self`) so the trait stays object-safe
    /// — a default method here would need to unsize `&Self` into
    /// `&dyn Reader` to call that same free function, which requires
    /// `Self: Sized` and would remove `execute` from `dyn Reader`'s vtable.
    ///
    /// # Arguments
    ///
    /// * `query` - The ggsql query (SQL + VISUALISE/TABULATE clause)
    ///
    /// # Returns
    ///
    /// A `ResolvedSpec` containing the resolved plot or table.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query syntax is invalid
    /// - The query has no VISUALISE/TABULATE clause
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
    fn execute(&self, query: &str) -> Result<ResolvedSpec>;

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
        let sql = self.dialect().sql_list_catalogs();
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
        let sql = self.dialect().sql_list_schemas(catalog);
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
        let sql = self.dialect().sql_list_tables(catalog, schema);
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
        let sql = self.dialect().sql_list_columns(catalog, schema, table);
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

/// Resolve a VISUALISE query into a `ResolvedPlot`.
///
/// This is the Plot-side counterpart to `execute::resolve_table_with_reader`
/// — see that function's doc comment for how the two relate.
pub fn resolve_plot_with_reader(reader: &dyn Reader, query: &str) -> Result<ResolvedPlot> {
    let validated = validate(query)?;
    let warnings: Vec<ValidationWarning> = validated.warnings().to_vec();

    let prepared_data = prepare_data_with_reader(query, reader)?;

    let plot =
        prepared_data.specs.into_iter().next().ok_or_else(|| {
            GgsqlError::ValidationError("No visualization spec found".to_string())
        })?;

    Ok(ResolvedPlot::new(
        plot,
        prepared_data.data,
        prepared_data.sql,
        prepared_data.visual,
        warnings,
    ))
}

/// Execute a ggsql query using any reader.
///
/// This is the shared implementation behind `Reader::execute()`. Concrete
/// readers delegate to this so the trait stays object-safe (no `Self: Sized`
/// bound on `execute`). Dispatches to the Plot or Table pipeline depending on
/// which kind of `Spec` the query's first statement is.
pub fn execute_with_reader(reader: &dyn Reader, query: &str) -> Result<ResolvedSpec> {
    let source_tree = SourceTree::new(query)?;
    source_tree.validate()?;
    let specs = parser::build_ast(&source_tree)?;

    match specs.into_iter().next() {
        Some(Spec::Table(_)) => {
            let resolved = resolve_table_with_reader(query, reader)?;
            Ok(ResolvedSpec::Table(Box::new(resolved)))
        }
        _ => {
            let resolved = resolve_plot_with_reader(reader, query)?;
            Ok(ResolvedSpec::Plot(Box::new(resolved)))
        }
    }
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

        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.plot().layers.len(), 1);
        assert_eq!(plot.metadata().layer_count, 1);
        assert!(plot.layer_data(0).is_some());

        let writer = VegaLiteWriter::new();
        let result = writer.render(&spec).unwrap();
        assert!(result.contains("point"));
    }

    /// Confirms the actual dispatch in `execute_with_reader` — not just its
    /// two pieces (`resolve_plot_with_reader`, `resolve_table_with_reader`)
    /// tested elsewhere — routes each Spec kind correctly through
    /// `Reader::execute()`, and that `Writer::render()` rejects a Table
    /// cleanly rather than panicking.
    #[test]
    fn test_execute_dispatches_plot_and_table() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        reader
            .execute_sql("CREATE TABLE sales AS SELECT 1 AS id")
            .unwrap();

        let plot_spec = reader
            .execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")
            .unwrap();
        assert!(plot_spec.as_plot().is_some());
        assert!(plot_spec.as_table().is_none());

        let table_spec = reader.execute("TABULATE * FROM sales").unwrap();
        assert!(table_spec.as_table().is_some());
        assert!(table_spec.as_plot().is_none());

        let writer = VegaLiteWriter::new();
        assert!(writer.render(&table_spec).is_err());
    }

    #[test]
    fn test_execute_metadata() {
        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let spec = reader
            .execute(
                "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y) VISUALISE x, y DRAW point",
            )
            .unwrap();

        let metadata = spec.as_plot().unwrap().metadata();
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
        let plot = spec.as_plot().unwrap();

        assert_eq!(plot.plot().layers.len(), 1);
        assert!(plot.layer_data(0).is_some());
        assert_eq!(plot.layer_data(0).unwrap().height(), 2);
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

        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.metadata().rows, 3);
        // Aesthetics are transformed to internal names (x -> pos1)
        assert!(plot.metadata().columns.contains(&"pos1".to_string()));

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
        assert_eq!(spec.as_plot().unwrap().metadata().rows, 3);
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
        let plot = spec.as_plot().unwrap();
        assert_eq!(plot.plot().layers.len(), 1);
        // Note: scales may include auto-generated x/y scales plus the explicit fill scale
        assert!(
            plot.plot().find_scale("fill").is_some(),
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
