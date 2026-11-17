//! Benchmark for the results API: the DuckDB results cache it uses, against
//! sorting and filtering the Arrow data directly with row indices.
//!
//! ```sh
//! cargo run --release -p ggsql-jupyter --example results_bench -- store 10000000
//! cargo run --release -p ggsql-jupyter --example results_bench -- arrow 10000000
//! ```
//!
//! Each mode runs in its own process so the resident-memory figures don't mix.
//! `store` also opens and closes a result repeatedly, to check that memory
//! levels off rather than growing with every cycle.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray, UInt32Array,
};
use arrow::compute::kernels::cmp::{gt_eq, lt_eq};
use arrow::compute::kernels::comparison::contains;
use arrow::compute::{and, filter, lexsort_to_indices, take, SortColumn, SortOptions};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ggsql::reader::connection::reader_from_uri;
use ggsql::DataFrame;
use ggsql_jupyter::results::filter::Capabilities;
use ggsql_jupyter::results::methods::{self, Session};
use ggsql_jupyter::results::ResultStore;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

const CATEGORIES: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn make_data(n: usize) -> RecordBatch {
    let mut rng = Lcg(42);
    let id: Int64Array = (0..n as i64).collect();
    let x: Float64Array = (0..n)
        .map(|_| {
            let v = rng.next();
            (v % 50 != 0).then(|| (v % 1_000_000) as f64 / 7.0)
        })
        .collect();
    let category: StringArray = (0..n)
        .map(|_| Some(CATEGORIES[(rng.next() % CATEGORIES.len() as u64) as usize]))
        .collect();
    let name: StringArray = (0..n)
        .map(|_| {
            let len = 12 + (rng.next() % 20) as usize;
            let s: String = (0..len)
                .map(|_| (b'a' + (rng.next() % 26) as u8) as char)
                .collect();
            Some(s)
        })
        .collect();
    let day: Date32Array = (0..n).map(|_| Some((rng.next() % 20_000) as i32)).collect();
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
        Field::new("category", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id),
            Arc::new(x),
            Arc::new(category),
            Arc::new(name),
            Arc::new(day),
        ],
    )
    .unwrap()
}

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
        / 1024.0
}

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    println!(
        "  {label:<40} {:>9.1} ms   rss {:>7.0} MB",
        t.elapsed().as_secs_f64() * 1000.0,
        rss_mb()
    );
    r
}

/// Sorting and filtering the Arrow data directly, for comparison.
fn run_arrow(batch: RecordBatch) {
    let n = batch.num_rows();
    let rows: ArrayRef = Arc::new(UInt32Array::from_iter_values(0..n as u32));
    let col = |name: &str| batch.column(batch.schema().index_of(name).unwrap()).clone();
    let asc = Some(SortOptions {
        descending: false,
        nulls_first: false,
    });
    // Stable sorts: the row index is the last key.
    let sort = |keys: &[&str]| {
        let mut columns: Vec<SortColumn> = keys
            .iter()
            .map(|k| SortColumn {
                values: col(k),
                options: asc,
            })
            .collect();
        columns.push(SortColumn {
            values: rows.clone(),
            options: asc,
        });
        lexsort_to_indices(&columns, None).unwrap()
    };

    let sort1 = time("sort x", || sort(&["x"]));
    time("sort category, x", || sort(&["category", "x"]));
    time("sort x (sort_to_indices, unstable)", || {
        arrow::compute::sort_to_indices(&col("x"), asc, None).unwrap()
    });
    time("filter name contains 'abc'", || {
        let name = col("name");
        let name = name.as_any().downcast_ref::<StringArray>().unwrap();
        let mask = contains(name, &StringArray::new_scalar("abc")).unwrap();
        filter(&rows, &mask).unwrap()
    });
    time("filter 1000 <= x <= 50000", || {
        let x = col("x");
        let mask: BooleanArray = and(
            &gt_eq(&x, &Float64Array::new_scalar(1000.0)).unwrap(),
            &lt_eq(&x, &Float64Array::new_scalar(50000.0)).unwrap(),
        )
        .unwrap();
        filter(&rows, &mask).unwrap()
    });
    time("page 100 rows mid sorted view", || {
        let page = sort1.slice(n / 2, 100);
        batch
            .columns()
            .iter()
            .map(|c| take(c, &page, None).unwrap())
            .collect::<Vec<_>>()
    });
}

/// The results API itself, through its JSON calls.
fn run_store(batch: RecordBatch, cycles: usize) {
    let n = batch.num_rows();
    let session_reader = reader_from_uri("duckdb://memory").unwrap();
    let session = Session {
        reader: &*session_reader,
        caps: Capabilities { regex: true },
    };
    let mut store = ResultStore::new(4).unwrap();
    let key = store.insert(
        DataFrame::from_record_batch(batch),
        "bench".into(),
        "SELECT 1".into(),
    );
    let mut call = |method: &str, params: Value| {
        methods::call(&mut store, &session, method, &params)
            .unwrap_or_else(|e| panic!("{method}: {e}"))
    };

    let view = time("open (copy into the cache)", || {
        call("result_open", json!({ "key": key }))
    })["view"]
        .as_str()
        .unwrap()
        .to_string();
    let sort = |keys: Value| json!({ "view": view, "keys": keys });
    let filters = |f: Value| json!({ "view": view, "filters": f });
    let mid = json!({ "view": view, "rows": {"first": n / 2, "last": n / 2 + 99}, "columns": [0, 1, 2, 3, 4] });

    time("page 100 rows mid unsorted", || {
        call("result_page", mid.clone())
    });
    time("sort x", || {
        call("result_set_sort", sort(json!([{"column": 1}])))
    });
    time("page 100 rows mid sorted view", || {
        call("result_page", mid.clone())
    });
    time("sort category, x", || {
        call(
            "result_set_sort",
            sort(json!([{"column": 2}, {"column": 1}])),
        )
    });
    call("result_set_sort", sort(json!([])));
    time("filter name contains 'abc'", || {
        call(
            "result_set_filters",
            filters(
                json!([{"type": "text", "column": 3, "match": "contains", "term": "abc", "case_sensitive": true}]),
            ),
        )
    });
    time("filter 1000 <= x <= 50000", || {
        call(
            "result_set_filters",
            filters(json!([{"type": "between", "column": 1, "low": "1000", "high": "50000"}])),
        )
    });
    time("filter range then sort x", || {
        call("result_set_sort", sort(json!([{"column": 1}])))
    });
    time("profile x (summary + histogram)", || {
        call(
            "result_profile",
            json!({"view": view, "column": 1, "profiles": {"null_count": true, "summary": true, "histogram": {"bins": 20, "quantiles": []}}}),
        )
    });
    call("result_close", json!({ "view": view }));
    println!(
        "  after close                                           rss {:>7.0} MB",
        rss_mb()
    );

    // Reopen, sort, page and close repeatedly: memory must level off.
    for cycle in 1..=cycles {
        let view = call("result_open", json!({ "key": key }))["view"]
            .as_str()
            .unwrap()
            .to_string();
        call(
            "result_set_sort",
            json!({ "view": view, "keys": [{"column": 2}, {"column": 1}] }),
        );
        call(
            "result_set_filters",
            json!({ "view": view, "filters": [{"type": "between", "column": 1, "low": "1000", "high": "50000"}] }),
        );
        call(
            "result_page",
            json!({ "view": view, "rows": {"first": 0, "last": 99}, "columns": [0, 1] }),
        );
        call("result_close", json!({ "view": view }));
        println!(
            "  cycle {cycle:>2} open/sort/filter/page/close          rss {:>7.0} MB",
            rss_mb()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("store");
    let n: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let cycles: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("mode={mode} rows={n}  baseline rss {:.0} MB", rss_mb());
    let batch = time("generate data", || make_data(n));
    let bytes: usize = batch
        .columns()
        .iter()
        .map(|c| c.get_array_memory_size())
        .sum();
    println!("  arrow data size {:.0} MB", bytes as f64 / 1e6);
    match mode {
        "arrow" => run_arrow(batch),
        "store" => run_store(batch, cycles),
        _ => panic!("mode must be store or arrow"),
    }
}
