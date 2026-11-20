//! The results API as named JSON calls.
//!
//! | Method | Params | Result |
//! | --- | --- | --- |
//! | `result_open` | `{key}` | `{view, key, title, query, columns: [{name, type_name, kind}], num_rows}` |
//! | `result_set_filters` | `{view, filters: [Filter]}` | `{num_rows, errors: [string \| null]}` |
//! | `result_set_sort` | `{view, keys: [{column, ascending}]}` | `{}` |
//! | `result_page` | `{view, rows: {first, last} \| {indices}, columns: [index]}` | `{columns: [[string \| null]]}` |
//! | `result_profile` | `{view, column, profiles: {null_count?, summary?, histogram?: {bins, quantiles}, frequency?: {limit}}}` | `{null_count?, summary?, histogram?, frequency?}` |
//! | `result_export` | `{view, rows?, columns, format: csv \| tsv \| html}` | `{data}` |
//! | `result_sql` | `{view}` | `{sql: [line]}`, in the session's dialect |
//! | `result_close` | `{view}` | `{}` |
//!
//! `num_rows` after filtering counts the rows that pass. A filter that cannot
//! be applied is left out and its message returned at its index in `errors`.
//! A sort or set of filters that cannot be applied at all is an error, and
//! leaves the view as it was.
//!
//! A page holds at most [`MAX_PAGE_ROWS`] rows. A range is cut short at the end
//! of the view; an index outside it is an error.
//!
//! `result_sql` is here rather than in a frontend because only the kernel
//! knows the session's database: it writes the view as SQL for that dialect,
//! which a frontend could not do from the neutral calls alone.
//!
//! [`MAX_PAGE_ROWS`]: super::view::MAX_PAGE_ROWS

use super::filter::{Filter, SortKey};
use super::view::{ProfileRequest, Rows};
use super::ResultStore;
use ggsql::reader::Reader;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

/// The method names this module answers.
pub fn handles(method: &str) -> bool {
    method.starts_with("result_")
}

/// The session's database, which `result_sql` writes SQL for.
pub struct Session<'a> {
    pub reader: &'a dyn Reader,
}

#[derive(Deserialize)]
struct Open {
    key: String,
}

#[derive(Deserialize)]
struct ViewOnly {
    view: String,
}

#[derive(Deserialize)]
struct SetFilters {
    view: String,
    filters: Vec<Filter>,
}

#[derive(Deserialize)]
struct SetSort {
    view: String,
    keys: Vec<SortKey>,
}

#[derive(Deserialize)]
struct Page {
    view: String,
    rows: Rows,
    columns: Vec<usize>,
}

#[derive(Deserialize)]
struct Profile {
    view: String,
    column: usize,
    profiles: ProfileRequest,
}

#[derive(Deserialize)]
struct Export {
    view: String,
    rows: Option<Rows>,
    columns: Vec<usize>,
    format: String,
}

/// Params arrive as the arguments of a call: an array whose first element is
/// the object, or the object itself.
fn parse<T: DeserializeOwned>(params: &Value) -> Result<T, String> {
    let object = match params {
        Value::Array(args) => args.first().cloned().unwrap_or(Value::Null),
        other => other.clone(),
    };
    serde_json::from_value(object).map_err(|e| format!("Invalid parameters: {e}"))
}

/// Answer one call.
pub fn call(
    store: &mut ResultStore,
    session: &Session,
    method: &str,
    params: &Value,
) -> Result<Value, String> {
    let err = |e: anyhow::Error| e.to_string();
    match method {
        "result_open" => {
            let p: Open = parse(params)?;
            let opened = store.open(&p.key).map_err(err)?;
            serde_json::to_value(opened).map_err(|e| e.to_string())
        }
        "result_set_filters" => {
            let p: SetFilters = parse(params)?;
            let errors = store.set_filters(&p.view, p.filters)?;
            let (view, _) = store.view_mut(&p.view).map_err(err)?;
            Ok(json!({ "num_rows": view.selected_rows, "errors": errors }))
        }
        "result_set_sort" => {
            let p: SetSort = parse(params)?;
            store.set_sort(&p.view, p.keys)?;
            Ok(json!({}))
        }
        "result_page" => {
            let p: Page = parse(params)?;
            let (view, cache) = store.view_mut(&p.view).map_err(err)?;
            let columns = view.page(cache.backend(), &p.rows, &p.columns)?;
            Ok(json!({ "columns": columns }))
        }
        "result_profile" => {
            let p: Profile = parse(params)?;
            let (view, cache) = store.view_mut(&p.view).map_err(err)?;
            view.profile(cache.backend(), p.column, &p.profiles)
        }
        "result_export" => {
            let p: Export = parse(params)?;
            let (view, cache) = store.view_mut(&p.view).map_err(err)?;
            let data = view.export(cache.backend(), p.rows.as_ref(), &p.columns, &p.format)?;
            Ok(json!({ "data": data }))
        }
        "result_sql" => {
            let p: ViewOnly = parse(params)?;
            let query = store.query_of(&p.view).map_err(err)?.to_string();
            let (view, _) = store.view_mut(&p.view).map_err(err)?;
            let sql = view.to_sql(&query, session.reader.dialect());
            Ok(json!({ "sql": sql }))
        }
        "result_close" => {
            let p: ViewOnly = parse(params)?;
            store.close(&p.view).map_err(err)?;
            Ok(json!({}))
        }
        other => Err(format!("Unknown method {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggsql::reader::connection::reader_from_uri;

    fn setup() -> (ResultStore, Box<dyn Reader + Send>, String) {
        let session = reader_from_uri("duckdb://memory").unwrap();
        let df = session
            .execute_sql("SELECT i AS n, 'x' || i AS s FROM range(5) t(i)")
            .unwrap();
        let mut store = ResultStore::new(super::super::DEFAULT_MAX_BYTES).unwrap();
        let key = store
            .insert(df, "Query [1]".into(), "SELECT * FROM t;".into())
            .unwrap()
            .unwrap();
        (store, session, key)
    }

    #[test]
    fn test_a_round_trip_through_the_calls() {
        let (mut store, reader, key) = setup();
        let session = Session { reader: &*reader };
        let mut call = |m: &str, p: Value| call(&mut store, &session, m, &p);

        // Positional arguments, as `callMethod` sends them.
        let opened = call("result_open", json!([{ "key": key }])).unwrap();
        assert_eq!(opened["num_rows"], 5);
        assert_eq!(opened["title"], "Query [1]");
        let view = opened["view"].as_str().unwrap().to_string();

        let r = call(
            "result_set_filters",
            json!({"view": view, "filters": [{"type": "compare", "column": 0, "op": ">", "value": "1"}]}),
        )
        .unwrap();
        assert_eq!(r, json!({"num_rows": 3, "errors": [null]}));

        call(
            "result_set_sort",
            json!({"view": view, "keys": [{"column": 0, "ascending": false}]}),
        )
        .unwrap();
        let page = call(
            "result_page",
            json!({"view": view, "rows": {"first": 0, "last": 1}, "columns": [1]}),
        )
        .unwrap();
        assert_eq!(page, json!({"columns": [["x4", "x3"]]}));

        let sql = call("result_sql", json!({ "view": view })).unwrap();
        let sql: Vec<&str> = sql["sql"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_str().unwrap())
            .collect();
        assert_eq!(
            sql,
            [
                "SELECT *",
                "FROM (",
                "    SELECT * FROM t",
                ") AS \"result\"",
                "WHERE \"n\" > 1",
                "ORDER BY \"n\" DESC NULLS LAST"
            ]
        );

        let profile = call(
            "result_profile",
            json!({"view": view, "column": 0, "profiles": {"null_count": true, "summary": true}}),
        )
        .unwrap();
        assert_eq!(profile["null_count"], 0);
        assert_eq!(profile["summary"]["min"], "2");
        assert_eq!(profile["summary"]["max"], "4");

        let export = call(
            "result_export",
            json!({"view": view, "rows": {"indices": [1, 0]}, "columns": [0, 1], "format": "tsv"}),
        )
        .unwrap();
        assert_eq!(export["data"], "n\ts\n3\tx3\n4\tx4");

        assert_eq!(
            call("result_close", json!({ "view": view })).unwrap(),
            json!({})
        );
        assert!(call(
            "result_page",
            json!({"view": view, "rows": {"first": 0, "last": 0}, "columns": [0]})
        )
        .unwrap_err()
        .contains("no view"));
    }

    #[test]
    fn test_the_session_database_is_untouched() {
        let (mut store, reader, key) = setup();
        let session = Session { reader: &*reader };
        let tables = || {
            reader
                .execute_sql("SELECT table_name FROM duckdb_tables()")
                .unwrap()
                .height()
        };
        let before = tables();
        let opened = call(&mut store, &session, "result_open", &json!({ "key": key })).unwrap();
        let view = opened["view"].as_str().unwrap();
        call(
            &mut store,
            &session,
            "result_set_sort",
            &json!({"view": view, "keys": [{"column": 0, "ascending": false}]}),
        )
        .unwrap();
        assert_eq!(tables(), before);
    }

    #[test]
    fn test_bad_params_and_unknown_methods_are_errors() {
        let (mut store, reader, _) = setup();
        let session = Session { reader: &*reader };
        assert!(call(&mut store, &session, "result_open", &json!({}))
            .unwrap_err()
            .contains("Invalid parameters"));
        assert!(call(&mut store, &session, "result_nope", &json!({})).is_err());
        assert!(handles("result_page"));
        assert!(!handles("open_editor"));
    }
}
