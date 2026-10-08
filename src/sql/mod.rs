//! Central SQL composition for generated queries.
//!
//! Every derived-table wrap, CTE hoist, table alias, and qualified star in
//! generated SQL goes through this module, so that dialect portability rules
//! live in exactly one place:
//!
//! - derived tables are always aliased and the alias always quoted
//!   (MySQL/MariaDB reject unaliased derived tables);
//! - the alias clause comes from [`SqlDialect::sql_table_alias`] (Oracle
//!   rejects `AS` before table aliases);
//! - a leading `WITH` is hoisted out of the parenthesized derived table when
//!   the dialect forbids CTEs in that position
//!   ([`SqlDialect::allows_cte_in_derived_table`], e.g. T-SQL);
//! - `*` combined with other select items goes through
//!   [`SqlDialect::sql_select_star`] (Oracle requires `alias.*`).
//!
//! Dialects supply only those primitives; the composition logic is shared.

use crate::reader::SqlDialect;

/// Split a query into its leading `WITH` clause and the remaining main
/// query. Returns `None` when the query does not start with `WITH`.
///
/// Used by dialects that forbid CTEs inside derived tables (SQL Server) to
/// hoist the CTE definitions out of a subquery wrap. The scanner is aware
/// of string literals and quoted identifiers, and handles optional CTE
/// column lists (`cte(c1, c2) AS (...)`) and `WITH RECURSIVE`.
pub fn split_cte_prefix(query: &str) -> Option<(&str, &str)> {
    let s = query.trim_start();
    let bytes = s.as_bytes();
    if bytes.len() < 5 || !s[..4].eq_ignore_ascii_case("with") || !bytes[4].is_ascii_whitespace() {
        return None;
    }
    let mut i = 4;
    skip_ws(bytes, &mut i);
    if s.len() - i >= 9 && s[i..i + 9].eq_ignore_ascii_case("recursive") {
        i += 9;
    }
    loop {
        skip_ws(bytes, &mut i);
        if i >= bytes.len() {
            return None;
        }
        let name_start = i;
        match bytes[i] {
            q @ (b'"' | b'`') => skip_quoted(bytes, &mut i, q),
            _ => {
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'$'))
                {
                    i += 1;
                }
            }
        }
        if i == name_start {
            return None;
        }
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b'(' {
            skip_balanced_parens(bytes, &mut i)?;
            skip_ws(bytes, &mut i);
        }
        if s.len() - i < 2 || !s[i..i + 2].eq_ignore_ascii_case("as") {
            return None;
        }
        i += 2;
        skip_ws(bytes, &mut i);
        if i >= bytes.len() || bytes[i] != b'(' {
            return None;
        }
        skip_balanced_parens(bytes, &mut i)?;
        let cte_end = i;
        let mut j = i;
        skip_ws(bytes, &mut j);
        if j < bytes.len() && bytes[j] == b',' {
            i = j + 1;
            continue;
        }
        return Some((&s[..cte_end], s[j..].trim_start()));
    }
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

/// Skip past a quoted region; `*i` is at the opening quote. A doubled quote
/// is treated as an escape (SQL string/identifier convention).
fn skip_quoted(bytes: &[u8], i: &mut usize, quote: u8) {
    *i += 1;
    while *i < bytes.len() {
        if bytes[*i] == quote {
            if *i + 1 < bytes.len() && bytes[*i + 1] == quote {
                *i += 2;
                continue;
            }
            *i += 1;
            return;
        }
        *i += 1;
    }
}

/// Skip a balanced parenthesised region; `*i` is at the opening `(`.
/// Returns `None` when the parens never balance.
fn skip_balanced_parens(bytes: &[u8], i: &mut usize) -> Option<()> {
    let mut depth = 0usize;
    while *i < bytes.len() {
        match bytes[*i] {
            b'(' => {
                depth += 1;
                *i += 1;
            }
            b')' => {
                depth -= 1;
                *i += 1;
                if depth == 0 {
                    return Some(());
                }
            }
            q @ (b'\'' | b'"' | b'`') => skip_quoted(bytes, i, q),
            _ => *i += 1,
        }
    }
    None
}

/// What a FROM item is, decided by the caller — never guessed from text.
///
/// The one judged case is [`FromItem::Fragment`], the contract for `from`
/// arguments passed across the [`SqlDialect`] trait: generated SQL only ever
/// passes a bare (possibly quoted) table/CTE name or an already-parenthesized
/// relation, so a leading `(` unambiguously means "already a relation".
#[derive(Debug, Clone, Copy)]
pub enum FromItem<'a> {
    /// A bare table or CTE name; emitted verbatim, never parenthesized
    /// (MySQL/MariaDB/T-SQL reject `FROM (name)`).
    Table(&'a str),
    /// A full query; parenthesized as a derived table, with a leading `WITH`
    /// hoisted out when the dialect forbids CTEs in that position.
    Query(&'a str),
    /// An already-composed relation (e.g. [`Select::build_derived`] output);
    /// emitted verbatim.
    Raw(&'a str),
    /// A trait-level FROM fragment: [`FromItem::Raw`] when it starts with
    /// `(`, otherwise [`FromItem::Table`].
    Fragment(&'a str),
}

/// `SELECT * FROM (<query>) AS "alias"` — the canonical derived-table wrap.
pub fn wrap_all<D: SqlDialect + ?Sized>(dialect: &D, query: &str, alias: &str) -> String {
    Select::new(dialect)
        .select_star()
        .from_aliased(FromItem::Query(query), alias)
        .build()
}

/// `SELECT <list> FROM <from> AS "alias"`.
pub fn select_from<D: SqlDialect + ?Sized>(
    dialect: &D,
    list: &str,
    from: FromItem<'_>,
    alias: &str,
) -> String {
    Select::new(dialect)
        .select(list)
        .from_aliased(from, alias)
        .build()
}

/// A SELECT statement under construction.
///
/// Obtained from [`Select::new`], configured through the `select_*`/`from_*`
/// methods, and rendered with [`Select::build`].
pub struct Select<'d, D: SqlDialect + ?Sized> {
    dialect: &'d D,
    /// Hoisted CTE definitions (without their `WITH`/`WITH RECURSIVE`
    /// keyword), rendered as one leading block.
    ctes: Vec<String>,
    cte_recursive: bool,
    list: Option<String>,
    from: String,
    joins: Vec<String>,
    where_clauses: Vec<String>,
    group_by: Option<String>,
    window: Option<String>,
    ordering: Option<String>,
}

impl<'d, D: SqlDialect + ?Sized> Select<'d, D> {
    pub fn new(dialect: &'d D) -> Self {
        Select {
            dialect,
            ctes: Vec::new(),
            cte_recursive: false,
            list: None,
            from: String::new(),
            joins: Vec::new(),
            where_clauses: Vec::new(),
            group_by: None,
            window: None,
            ordering: None,
        }
    }

    /// Add a CTE definition (`name_and_cols` is e.g. `"\"__ggsql_x__\"(a, b)"`).
    ///
    /// When the body itself starts with `WITH` and the dialect forbids CTEs
    /// inside derived tables, the body's own CTEs are hoisted into the
    /// statement's leading WITH block first (a CTE body has the same
    /// restriction as a derived table there).
    pub fn with_cte(mut self, name_and_cols: &str, body: &str) -> Self {
        let mut body_text = body.trim();
        if !self.dialect.allows_cte_in_derived_table() {
            if let Some((cte, rest)) = split_cte_prefix(body_text) {
                self.push_hoisted_cte(cte);
                body_text = rest;
            }
        }
        self.ctes.push(format!("{name_and_cols} AS ({body_text})"));
        self
    }

    /// Raw select list, e.g. `"a, b AS c"`.
    pub fn select(mut self, list: impl Into<String>) -> Self {
        self.list = Some(list.into());
        self
    }

    /// Select items, joined with `", "`. Prefer over [`Select::select`] when
    /// the list is assembled from parts: each item stays on its own line at
    /// the call site instead of embedding newlines in one big string.
    pub fn select_items(mut self, items: &[String]) -> Self {
        self.list = Some(items.join(", "));
        self
    }

    /// `*` as the whole select list (valid unqualified on every backend).
    pub fn select_star(mut self) -> Self {
        self.list = Some("*".to_string());
        self
    }

    /// Extra items followed by `*` (star last, e.g. `expr AS col, *` where
    /// the first occurrence of a duplicated column wins). The star goes
    /// through the dialect like in [`Select::select_star_plus`].
    pub fn select_plus_star(mut self, items: &[String], alias: &str) -> Self {
        let mut list: Vec<String> = items.to_vec();
        list.push(self.dialect.sql_select_star(alias));
        self.list = Some(list.join(", "));
        self
    }

    /// `*` plus extra items. The star goes through the dialect so backends
    /// that reject an unqualified `*` alongside other items (Oracle) get
    /// `alias.*`; `alias` must match the alias passed to
    /// [`Select::from_aliased`].
    pub fn select_star_plus(mut self, extras: &[String], alias: &str) -> Self {
        let star = self.dialect.sql_select_star(alias);
        let mut items = vec![star];
        items.extend(extras.iter().cloned());
        self.list = Some(items.join(", "));
        self
    }

    /// FROM a table, query, or composed relation (see [`FromItem`]), without
    /// an alias.
    pub fn from(mut self, from: FromItem<'_>) -> Self {
        self.from = self.prepare_from(from);
        self
    }

    /// FROM with a table alias. The alias clause is emitted by the dialect
    /// (`AS "alias"` on most backends, bare `"alias"` on Oracle).
    pub fn from_aliased(mut self, from: FromItem<'_>, alias: &str) -> Self {
        let fragment = self.prepare_from(from);
        self.from = format!("{} {}", fragment, self.dialect.sql_table_alias(alias));
        self
    }

    /// Append a raw JOIN fragment (`CROSS JOIN ...`, `INNER JOIN ... ON ...`).
    pub fn join_raw(mut self, fragment: &str) -> Self {
        self.joins.push(fragment.to_string());
        self
    }

    /// Add a WHERE predicate; multiple predicates are combined with AND.
    pub fn and_where(mut self, predicate: impl Into<String>) -> Self {
        self.where_clauses.push(predicate.into());
        self
    }

    /// GROUP BY clause (raw column/expression list).
    pub fn group_by(mut self, cols: impl Into<String>) -> Self {
        self.group_by = Some(cols.into());
        self
    }

    /// WINDOW clause (e.g. `"w AS (PARTITION BY ...)"`).
    pub fn window(mut self, clause: impl Into<String>) -> Self {
        self.window = Some(clause.into());
        self
    }

    /// ORDER BY on the final statement.
    ///
    /// There is deliberately no nested variant: ordering inside a derived
    /// table is not guaranteed to propagate to the outer query, and T-SQL
    /// rejects it outright without TOP/OFFSET (error 1033). Order only the
    /// outermost query.
    pub fn order_by(mut self, ordering: impl Into<String>) -> Self {
        self.ordering = Some(ordering.into());
        self
    }

    /// Build the SELECT statement.
    ///
    /// When no `from`/`from_aliased` was configured, renders a from-less
    /// SELECT (a single-row expression relation like `SELECT 1 AS a`) —
    /// valid only where the grammar allows it, e.g. as a derived table via
    /// [`Select::build_derived`].
    pub fn build(self) -> String {
        let mut out = String::new();
        if !self.ctes.is_empty() {
            let keyword = if self.cte_recursive {
                self.dialect.sql_with_recursive()
            } else {
                "WITH"
            };
            out.push_str(keyword);
            out.push(' ');
            out.push_str(&self.ctes.join(", "));
            out.push(' ');
        }
        out.push_str("SELECT ");
        out.push_str(self.list.as_deref().unwrap_or("*"));
        if !self.from.is_empty() {
            out.push_str(" FROM ");
            out.push_str(&self.from);
        }
        for join in &self.joins {
            out.push(' ');
            out.push_str(join);
        }
        if !self.where_clauses.is_empty() {
            out.push_str(" WHERE ");
            out.push_str(&self.where_clauses.join(" AND "));
        }
        if let Some(cols) = &self.group_by {
            out.push_str(" GROUP BY ");
            out.push_str(cols);
        }
        if let Some(clause) = &self.window {
            out.push_str(" WINDOW ");
            out.push_str(clause);
        }
        if let Some(ordering) = &self.ordering {
            out.push_str(" ORDER BY ");
            out.push_str(ordering);
        }
        out
    }

    /// Build and apply the dialect's row-limit wrapper (`LIMIT n`,
    /// `SELECT TOP n * FROM (...)`, `WHERE ROWNUM <= n`, ...).
    ///
    /// When the dialect wraps the query in a derived table
    /// ([`SqlDialect::sql_limit_wraps_query`], e.g. T-SQL's TOP), an ORDER BY
    /// is moved outside the wrap — ordering inside a derived table is T-SQL
    /// error 1033. Clause-style limits (`LIMIT n`) keep the ordering inside,
    /// where `ORDER BY … LIMIT n` is valid.
    pub fn build_limited(self, n: usize) -> String {
        if self.ordering.is_some() && self.dialect.sql_limit_wraps_query() {
            let ordering = self.ordering.clone().expect("checked above");
            let body = Select {
                ordering: None,
                ..self
            }
            .build();
            format!("{} ORDER BY {}", self.dialect.sql_limit(&body, n), ordering)
        } else {
            self.dialect.sql_limit(&self.build(), n)
        }
    }

    /// Build as a parenthesized, aliased derived-table fragment:
    /// `(<SELECT ...>) <alias clause>`. Useful when a hand-composed query
    /// needs to embed a relation (e.g. a one-row expression relation) as a
    /// FROM item; the alias clause goes through
    /// [`SqlDialect::sql_table_alias`] like any other.
    pub fn build_derived(self, alias: &str) -> String {
        let alias_clause = self.dialect.sql_table_alias(alias);
        format!("({}) {}", self.build(), alias_clause)
    }

    /// The FROM fragment for `from` (see [`FromItem`]): queries are
    /// parenthesized, with a leading `WITH` hoisted into `self.ctes` when
    /// the dialect forbids CTEs inside derived tables.
    fn prepare_from(&mut self, from: FromItem<'_>) -> String {
        let trimmed = match from {
            FromItem::Table(t) => return t.trim().to_string(),
            FromItem::Raw(r) => return r.trim().to_string(),
            FromItem::Fragment(f) => return f.trim().to_string(),
            FromItem::Query(q) => q.trim(),
        };
        if !self.dialect.allows_cte_in_derived_table() {
            if let Some((cte, body)) = split_cte_prefix(trimmed) {
                self.push_hoisted_cte(cte);
                return format!("({body})");
            }
        }
        format!("({trimmed})")
    }

    fn push_hoisted_cte(&mut self, prefix: &str) {
        let s = prefix.trim_start();
        debug_assert!(s.len() >= 4 && s[..4].eq_ignore_ascii_case("with"));
        let mut rest = s[4..].trim_start();
        if rest.len() >= 9 && rest[..9].eq_ignore_ascii_case("recursive") {
            self.cte_recursive = true;
            rest = rest[9..].trim_start();
        }
        self.ctes.push(rest.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::dialects::mssql::MssqlDialect;
    use crate::reader::dialects::oracle::OracleDialect;
    use crate::reader::AnsiDialect;

    #[test]
    fn splits_cte_prefix() {
        let (cte, body) = split_cte_prefix(
            "WITH a AS (SELECT 1 AS x), b(n) AS (SELECT 2) SELECT * FROM a JOIN b ON a.x = b.n",
        )
        .unwrap();
        assert_eq!(cte, "WITH a AS (SELECT 1 AS x), b(n) AS (SELECT 2)");
        assert_eq!(body, "SELECT * FROM a JOIN b ON a.x = b.n");
    }

    #[test]
    fn splits_cte_with_parens_and_strings() {
        let (cte, body) = split_cte_prefix(
            "WITH RECURSIVE \"__ggsql_t__\" AS (SELECT '(' AS s, f(1, (2)) AS v) SELECT v FROM \"__ggsql_t__\"",
        )
        .unwrap();
        assert_eq!(
            cte,
            "WITH RECURSIVE \"__ggsql_t__\" AS (SELECT '(' AS s, f(1, (2)) AS v)"
        );
        assert_eq!(body, "SELECT v FROM \"__ggsql_t__\"");
    }

    #[test]
    fn no_cte_returns_none() {
        assert!(split_cte_prefix("SELECT 1").is_none());
        assert!(split_cte_prefix("WITHHELD AS x").is_none());
        assert!(split_cte_prefix("WITH a AS (SELECT 1").is_none());
    }

    #[test]
    fn basic_wrap() {
        assert_eq!(
            wrap_all(&AnsiDialect, "SELECT a FROM t", "__ggsql_x__"),
            "SELECT * FROM (SELECT a FROM t) AS \"__ggsql_x__\""
        );
    }

    #[test]
    fn bare_table_is_not_parenthesized() {
        assert_eq!(
            select_from(&AnsiDialect, "a, b", FromItem::Table("mytable"), "s"),
            "SELECT a, b FROM mytable AS \"s\""
        );
    }

    #[test]
    fn raw_relation_is_emitted_verbatim() {
        // A pre-composed relation keeps its own alias; the FROM alias is not
        // stacked on top (that would be invalid SQL).
        let rel = Select::new(&AnsiDialect)
            .select("1 AS a")
            .build_derived("u");
        assert_eq!(
            select_from(&AnsiDialect, "a", FromItem::Raw(&rel), "s"),
            "SELECT a FROM (SELECT 1 AS a) AS \"u\" AS \"s\""
        );
    }

    #[test]
    fn fragment_distinguishes_relation_from_table() {
        assert_eq!(
            select_from(&AnsiDialect, "a", FromItem::Fragment("\"my table\""), "s"),
            "SELECT a FROM \"my table\" AS \"s\""
        );
        assert_eq!(
            select_from(
                &AnsiDialect,
                "a",
                FromItem::Fragment("(SELECT 1 AS a) AS \"u\""),
                "s"
            ),
            "SELECT a FROM (SELECT 1 AS a) AS \"u\" AS \"s\""
        );
    }

    #[test]
    fn nested_cte_stays_put_on_ansi() {
        let q = "WITH c AS (SELECT 1 AS a) SELECT a FROM c";
        assert_eq!(
            wrap_all(&AnsiDialect, q, "s"),
            "SELECT * FROM (WITH c AS (SELECT 1 AS a) SELECT a FROM c) AS \"s\""
        );
    }

    #[test]
    fn mssql_hoists_cte_out_of_derived_table() {
        let q = "WITH c AS (SELECT 1 AS a) SELECT a FROM c";
        assert_eq!(
            wrap_all(&MssqlDialect, q, "s"),
            "WITH c AS (SELECT 1 AS a) SELECT * FROM (SELECT a FROM c) AS \"s\""
        );
    }

    #[test]
    fn mssql_hoisted_recursive_cte_loses_recursive_keyword() {
        // T-SQL has no RECURSIVE keyword; the hoist re-emits via the dialect.
        let q = "WITH RECURSIVE c(n) AS (SELECT 0 AS n) SELECT n FROM c";
        assert_eq!(
            wrap_all(&MssqlDialect, q, "s"),
            "WITH c(n) AS (SELECT 0 AS n) SELECT * FROM (SELECT n FROM c) AS \"s\""
        );
    }

    #[test]
    fn oracle_table_alias_has_no_as() {
        assert_eq!(
            wrap_all(&OracleDialect, "SELECT a FROM t", "s"),
            "SELECT * FROM (SELECT a FROM t) \"s\""
        );
    }

    #[test]
    fn oracle_star_plus_is_qualified() {
        assert_eq!(
            Select::new(&OracleDialect)
                .select_star_plus(&["1 AS one".to_string()], "s")
                .from_aliased(FromItem::Query("SELECT a FROM t"), "s")
                .build(),
            "SELECT \"s\".*, 1 AS one FROM (SELECT a FROM t) \"s\""
        );
        assert_eq!(
            Select::new(&AnsiDialect)
                .select_star_plus(&["1 AS one".to_string()], "s")
                .from_aliased(FromItem::Query("SELECT a FROM t"), "s")
                .build(),
            "SELECT *, 1 AS one FROM (SELECT a FROM t) AS \"s\""
        );
    }

    #[test]
    fn where_and_order_by() {
        assert_eq!(
            Select::new(&AnsiDialect)
                .select("a")
                .from(FromItem::Table("t"))
                .and_where("a > 1")
                .and_where("b < 2")
                .order_by("a")
                .build(),
            "SELECT a FROM t WHERE a > 1 AND b < 2 ORDER BY a"
        );
    }

    #[test]
    fn from_less_select_and_derived_fragment() {
        let rel = Select::new(&AnsiDialect)
            .select("ST_GeomFromText('POINT(0 0)') AS g")
            .build_derived("__ggsql_bbox__");
        assert_eq!(
            rel,
            "(SELECT ST_GeomFromText('POINT(0 0)') AS g) AS \"__ggsql_bbox__\""
        );
        // Oracle: table alias without AS.
        let rel = Select::new(&OracleDialect)
            .select("1 AS one")
            .build_derived("x");
        assert_eq!(rel, "(SELECT 1 AS one) \"x\"");
    }

    #[test]
    fn limit_goes_through_dialect() {
        assert_eq!(
            Select::new(&AnsiDialect)
                .select("a")
                .from(FromItem::Table("t"))
                .build_limited(3),
            "SELECT a FROM t LIMIT 3"
        );
        assert_eq!(
            Select::new(&MssqlDialect)
                .select("a")
                .from(FromItem::Table("t"))
                .build_limited(3),
            "SELECT TOP 3 * FROM (SELECT a FROM t) AS \"__ggsql_lim__\""
        );
    }
}
