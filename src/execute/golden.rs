//! Golden SQL tests (Tier 1 dialect conformance).
//!
//! Runs the canonical battery ([`battery`] — shared with the live-backend
//! tests in src/tests/dialect_live.rs) once per dialect against a
//! [`StubReader`] — a reader that records every statement and fabricates
//! empty results — and compares the emitted SQL against checked-in golden
//! files under `src/golden/<dialect>.sql`, one `#[test]` per dialect.
//!
//! These tests catch *composition* errors that per-function dialect unit
//! tests cannot: how a dialect's overrides interact when a generated
//! subquery is embedded in a larger statement (e.g. the two-stage histogram
//! GROUP BY, or qualified-projection aliasing in boxplot/density).
//!
//! After an intentional change to generated SQL, regenerate the goldens:
//!
//! ```sh
//! GGSQL_BLESS=1 cargo test -p ggsql --lib golden
//! ```
//!
//! ...and review the diff like any other code change. Bless individual
//! dialects by naming them: `GGSQL_BLESS=1 cargo test -p ggsql --lib golden::sqlite`.

#![cfg(test)]

#[path = "../tests/battery/mod.rs"]
mod battery;

use crate::reader::registry;
use crate::reader::test_support::StubReader;
use crate::reader::{AnsiDialect, Reader, SqlDialect};
use crate::DataFrame;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The dialect a golden file is named for: `ansi` is the checked-in
/// baseline (AnsiDialect is also flightsql's dialect, but the baseline
/// predates and outlives any single scheme); everything else resolves
/// through the registry, so a dialect constructor can only live in one
/// place. The `goldens_cover_registry` test keeps the file set in step.
fn dialect_for(name: &str) -> Box<dyn SqlDialect + Send> {
    if name == "ansi" {
        return Box::new(AnsiDialect);
    }
    registry::by_scheme(name)
        .unwrap_or_else(|| panic!("no registry entry for golden dialect '{name}'"))
        .dialect()
}

/// The shared fixture as a DataFrame, matching the live table's schema:
/// id/val/grp/day(a real date)/"mixed Case". `TextDay` swaps the date for
/// ISO text to force the text→temporal cast path.
fn fixture_df(fixture: &battery::Fixture) -> DataFrame {
    use arrow::array::{ArrayRef, Date32Array, Float64Array, Int32Array, StringArray};
    let day: ArrayRef = match fixture {
        battery::Fixture::Shared => Arc::new(Date32Array::from(
            battery::ROWS
                .iter()
                .map(|r| r.day_epoch)
                .collect::<Vec<_>>(),
        )),
        battery::Fixture::TextDay => Arc::new(StringArray::from(
            battery::ROWS
                .iter()
                .map(|r| battery::day_iso(r.day_epoch))
                .collect::<Vec<_>>(),
        )),
    };
    DataFrame::new(vec![
        (
            "id",
            Arc::new(Int32Array::from(
                battery::ROWS.iter().map(|r| r.id).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "val",
            Arc::new(Float64Array::from(
                battery::ROWS.iter().map(|r| r.val).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "grp",
            Arc::new(StringArray::from(
                battery::ROWS.iter().map(|r| r.grp).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("day", day),
        (
            "mixed Case",
            Arc::new(Float64Array::from(
                battery::ROWS.iter().map(|r| r.mixed).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap()
}

/// Collapse all whitespace runs so goldens don't churn on formatting, and
/// replace the random per-session hex in generated temp-table names
/// (`__ggsql_global_<32 hex>__`) with a stable placeholder.
fn normalize(sql: &str) -> String {
    let collapsed = sql.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(collapsed.len());
    let chars: Vec<char> = collapsed.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_hexdigit() {
            let run: String = chars[i..]
                .iter()
                .take_while(|c| c.is_ascii_hexdigit())
                .collect();
            if run.len() == 32 {
                out.push_str("<session>");
                i += 32;
                continue;
            }
            out.push_str(&run);
            i += run.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden")
}

/// Render every battery case's SQL for one dialect. Re-registers the
/// fixture only when its variant changes.
fn render_dialect(dialect: Box<dyn SqlDialect + Send>) -> String {
    let (reader, log) = StubReader::new(dialect);
    let mut out = String::new();
    let mut registered: Option<u8> = None;

    for case in battery::cases() {
        let variant = match case.fixture {
            battery::Fixture::Shared => 0u8,
            battery::Fixture::TextDay => 1u8,
        };
        if registered != Some(variant) {
            if registered.is_some() {
                reader.unregister(battery::TABLE).unwrap();
            }
            reader
                .register(battery::TABLE, fixture_df(&case.fixture), true)
                .unwrap();
            registered = Some(variant);
        }
        out.push_str(&format!("-- case: {}\n", case.name));
        let query = case.query.replace("{table}", battery::TABLE);
        if let Err(e) = reader.execute(&query) {
            out.push_str(&format!("-- ERROR: {e}\n"));
        }
        drain_log(&log, &mut out);
        out.push('\n');
    }
    out
}

fn drain_log(log: &Arc<Mutex<Vec<String>>>, out: &mut String) {
    for stmt in log.lock().unwrap().drain(..) {
        out.push_str(&normalize(&stmt));
        out.push_str(";\n");
    }
}

/// One dialect's golden check (or bless, under GGSQL_BLESS=1). On mismatch
/// the failure message carries a unified diff, so the change is reviewable
/// from the test output alone.
fn check_dialect(name: &str) {
    let path = golden_dir().join(format!("{name}.sql"));
    let out = render_dialect(dialect_for(name));

    if std::env::var("GGSQL_BLESS").is_ok() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, &out).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing golden file {} — bless with \
             `GGSQL_BLESS=1 cargo test -p ggsql --lib golden::{name}`",
            path.display()
        )
    });
    if expected != out {
        panic!(
            "golden SQL mismatch for {name}:\n{}\n\
             Bless intentional changes with \
             `GGSQL_BLESS=1 cargo test -p ggsql --lib golden::{name}`.",
            unified_diff(&expected, &out)
        );
    }
}

/// Minimal unified diff (3 lines of context) between two texts. Goldens
/// are a few hundred lines, so an O(n·m) LCS table is fine and keeps the
/// test binary dependency-free.
fn unified_diff(old: &str, new: &str) -> String {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    // LCS lengths over suffixes.
    let mut lcs = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    // Walk the table, coalescing changes into hunks with context.
    let mut ops: Vec<(char, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            ops.push((' ', a[i]));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            ops.push(('-', a[i]));
            i += 1;
        } else {
            ops.push(('+', b[j]));
            j += 1;
        }
    }
    while i < a.len() {
        ops.push(('-', a[i]));
        i += 1;
    }
    while j < b.len() {
        ops.push(('+', b[j]));
        j += 1;
    }
    // Emit with 3 lines of context around changes, hunks merged when close.
    const CTX: usize = 3;
    let changed: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, (t, _))| *t != ' ')
        .map(|(k, _)| k)
        .collect();
    if changed.is_empty() {
        return String::from("(files differ only in trailing whitespace/newline)");
    }
    let mut out = String::from("--- expected\n+++ actual\n");
    let mut shown_up_to = 0usize;
    let mut k = 0;
    while k < changed.len() {
        let start = changed[k].saturating_sub(CTX);
        let mut end = changed[k] + CTX + 1;
        while k + 1 < changed.len() && changed[k + 1] <= end + CTX {
            k += 1;
            end = changed[k] + CTX + 1;
        }
        let end = end.min(ops.len());
        if start > shown_up_to {
            out.push_str("@@\n");
        }
        for (t, line) in &ops[start.max(shown_up_to)..end] {
            out.push(*t);
            out.push_str(line);
            out.push('\n');
        }
        shown_up_to = end;
        k += 1;
    }
    out
}

macro_rules! per_dialect {
    ($($name:ident),* $(,)?) => {
        $( #[test] fn $name() { check_dialect(stringify!($name)); } )*
    };
}

// One test per golden file. `goldens_cover_registry` fails if this list
// and the registry drift apart (a new dialect needs its golden blessed).
per_dialect!(
    ansi, bigquery, clickhouse, databricks, datafusion, drill, druid, duckdb, exasol, monetdb,
    mssql, mysql, oracle, postgres, redshift, snowflake, sqlite, trino,
);

/// The golden file set must match the registry exactly: every canonical
/// scheme has a golden (flightsql excepted — its dialect *is*
/// AnsiDialect, covered by ansi.sql), and every golden file except the
/// ansi baseline names a canonical scheme.
#[test]
fn goldens_cover_registry() {
    let mut files: Vec<String> = std::fs::read_dir(golden_dir())
        .unwrap()
        .map(|e| {
            e.unwrap()
                .file_name()
                .into_string()
                .unwrap()
                .trim_end_matches(".sql")
                .to_string()
        })
        .collect();
    files.sort();

    let mut expected: Vec<&str> = registry::REGISTRY
        .iter()
        .map(|e| e.scheme)
        .filter(|s| *s != "flightsql")
        .chain(["ansi"])
        .collect();
    expected.sort_unstable();

    assert_eq!(
        files, expected,
        "golden files and registry schemes diverged — bless the new dialect's golden \
         (GGSQL_BLESS=1 cargo test -p ggsql --lib golden::<scheme>) or remove the stale file"
    );
}

/// Byte-identical goldens mean a dialect's overrides are never exercised
/// by the battery — exactly what this suite exists to catch. Pairs on
/// this list are tolerated; extend the battery rather than the list.
const IDENTICAL_ALLOWLIST: &[(&str, &str)] = &[
    // Exasol's overrides (schema handling, quoting) don't surface in any
    // current battery case; its SQL is ANSI-identical.
    ("ansi", "exasol"),
];

#[test]
fn no_identical_goldens_unless_allowlisted() {
    let mut contents: Vec<(String, Vec<u8>)> = std::fs::read_dir(golden_dir())
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            (
                p.file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    contents.sort();

    let mut identical = Vec::new();
    for i in 0..contents.len() {
        for j in i + 1..contents.len() {
            if contents[i].1 == contents[j].1 {
                let pair = (contents[i].0.as_str(), contents[j].0.as_str());
                if !IDENTICAL_ALLOWLIST.contains(&pair) {
                    identical.push(format!("{} == {}", pair.0, pair.1));
                }
            }
        }
    }
    assert!(
        identical.is_empty(),
        "byte-identical goldens outside the allow-list (a dialect's overrides \
         are unexercised — extend the battery, or allow-list with justification):\n{}",
        identical.join("\n")
    );
}

/// `.github/scripts/live/ggsql_live_test.csv` seeds the Druid datasource
/// (and any future non-Rust seed); it must be byte-identical to the CSV
/// derived from the battery's ROWS, or the live table and the golden
/// fixture silently diverge.
#[test]
fn fixture_csv_is_current() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../.github/scripts/live/ggsql_live_test.csv");
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — write battery::fixture_csv() to it",
            path.display()
        )
    });
    assert_eq!(
        on_disk,
        battery::fixture_csv(),
        "fixture CSV drifted from the battery ROWS — regenerate it (the CSV is the \
         seed source for the Druid live leg)"
    );
}
