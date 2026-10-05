//! Golden SQL tests (Tier 1 dialect conformance).
//!
//! Runs a canonical battery of VISUALISE pipelines once per dialect against
//! a [`StubReader`] — a reader that records every statement and fabricates
//! empty results — and compares the emitted SQL against checked-in golden
//! files under `src/golden/<dialect>.sql`.
//!
//! These tests catch *composition* errors that per-function dialect unit
//! tests cannot: how a dialect's overrides interact when a generated
//! subquery is embedded in a larger statement (e.g. the two-stage histogram
//! GROUP BY, or qualified-projection aliasing in boxplot/density).
//!
//! After an intentional change to generated SQL, regenerate the goldens:
//!
//! ```sh
//! GGSQL_SKIP_GENERATE=1 GGSQL_BLESS=1 cargo test -p ggsql --lib golden
//! ```
//!
//! ...and review the diff like any other code change.

#![cfg(test)]

use crate::df;
use crate::reader::dialects::*;
use crate::reader::test_support::StubReader;
use crate::reader::{AnsiDialect, Reader, SqlDialect};
use crate::DataFrame;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Every dialect ggsql ships, labeled by the golden file name.
fn all_dialects() -> Vec<(&'static str, Box<dyn SqlDialect>)> {
    vec![
        ("ansi", Box::new(AnsiDialect)),
        ("bigquery", Box::new(BigQueryDialect)),
        ("clickhouse", Box::new(ClickHouseDialect)),
        ("databricks", Box::new(DatabricksDialect)),
        ("datafusion", Box::new(DataFusionDialect)),
        ("drill", Box::new(DrillDialect)),
        ("druid", Box::new(DruidDialect)),
        ("duckdb", Box::new(DuckDbDialect)),
        ("exasol", Box::new(ExasolDialect)),
        ("monetdb", Box::new(MonetDbDialect)),
        ("mssql", Box::new(MssqlDialect)),
        ("mysql", Box::new(MySqlDialect)),
        ("oracle", Box::new(OracleDialect)),
        ("postgres", Box::new(PostgresDialect)),
        ("redshift", Box::new(RedshiftDialect)),
        ("snowflake", Box::new(SnowflakeDialect)),
        ("sqlite", Box::new(SqliteDialect)),
        ("trino", Box::new(TrinoDialect)),
    ]
}

struct Case {
    name: &'static str,
    table: &'static str,
    df: DataFrame,
    query: &'static str,
}

/// The canonical pipeline battery. Each case is chosen to exercise a
/// distinct slice of dialect surface: quoting and limits, the two-stage
/// histogram, qualified projections, quantiles, temporal literals, and
/// aggregation.
fn battery() -> Vec<Case> {
    vec![
        Case {
            name: "scatter_color",
            table: "pts",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0],
                "b" => vec![10.0f64, 20.0, 15.0, 25.0],
                "g" => vec!["A", "B", "A", "B"],
            }
            .unwrap(),
            query: "VISUALISE DRAW point MAPPING a AS x, b AS y, g AS color FROM pts",
        },
        Case {
            name: "histogram",
            table: "hist_data",
            df: df! {
                "value" => vec![1.0f64, 2.0, 2.5, 3.0, 3.5, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW histogram MAPPING value AS x FROM hist_data",
        },
        Case {
            name: "boxplot_grouped",
            table: "box_data",
            df: df! {
                "grp" => vec!["A", "A", "B", "B"],
                "value" => vec![1.0f64, 2.0, 3.0, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW boxplot MAPPING grp AS x, value AS y FROM box_data",
        },
        Case {
            name: "line_aggregate",
            table: "line_data",
            df: df! {
                "grp" => vec!["A", "A", "B", "B"],
                "value" => vec![1.0f64, 2.0, 3.0, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW line MAPPING grp AS x, value AS y FROM line_data SETTING aggregate => 'y:mean'",
        },
        Case {
            name: "density_grouped",
            table: "dens_data",
            df: df! {
                "value" => vec![1.0f64, 2.0, 3.0, 4.0],
                "grp" => vec!["A", "A", "B", "B"],
            }
            .unwrap(),
            query: "VISUALISE DRAW density MAPPING value AS x, grp AS color FROM dens_data",
        },
        Case {
            name: "line_temporal",
            table: "ts_data",
            // Real Date32 column (df! has no date impl) so temporal
            // literal/cast dialect overrides are exercised.
            df: DataFrame::new(vec![
                (
                    "day",
                    Arc::new(arrow::array::Date32Array::from(vec![19000, 19001, 19002]))
                        as arrow::array::ArrayRef,
                ),
                (
                    "value",
                    Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 3.0])),
                ),
            ])
            .unwrap(),
            query: "SELECT * FROM ts_data VISUALISE day AS x, value AS y DRAW line",
        },
        Case {
            name: "bar_count",
            table: "bar_data",
            df: df! {
                "category" => vec!["a", "b", "a"],
            }
            .unwrap(),
            query: "VISUALISE DRAW bar MAPPING category AS x FROM bar_data",
        },
        Case {
            name: "smooth_lm",
            table: "smooth_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0],
                "b" => vec![2.0f64, 4.0, 5.0, 8.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW smooth MAPPING a AS x, b AS y FROM smooth_data SETTING method => 'ols'",
        },
        Case {
            name: "ribbon",
            table: "rib_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0],
                "lo" => vec![0.0f64, 1.0, 1.5, 2.0],
                "hi" => vec![2.0f64, 3.0, 3.5, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW ribbon MAPPING a AS x, lo AS ymin, hi AS ymax FROM rib_data",
        },
        Case {
            name: "segment",
            table: "seg_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0],
                "b" => vec![1.0f64, 2.0, 3.0],
                "c" => vec![2.0f64, 3.0, 4.0],
                "d" => vec![2.0f64, 1.0, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW segment MAPPING a AS x, b AS y, c AS xend, d AS yend FROM seg_data",
        },
        Case {
            name: "tile",
            table: "tile_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b" => vec![1.0f64, 2.0, 1.5, 2.5, 3.0, 3.5],
            }
            .unwrap(),
            query: "VISUALISE DRAW tile MAPPING a AS x, b AS y FROM tile_data",
        },
        Case {
            name: "area",
            table: "area_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0],
                "b" => vec![2.0f64, 4.0, 3.0, 5.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW area MAPPING a AS x, b AS y FROM area_data",
        },
        Case {
            name: "violin",
            table: "vio_data",
            df: df! {
                "g" => vec!["A", "A", "B", "B"],
                "v" => vec![1.0f64, 2.0, 3.0, 4.0],
            }
            .unwrap(),
            query: "VISUALISE DRAW violin MAPPING g AS x, v AS y FROM vio_data",
        },
        Case {
            name: "filter_layer",
            table: "pts",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0],
                "b" => vec![10.0f64, 20.0, 15.0],
                "g" => vec!["A", "B", "A"],
            }
            .unwrap(),
            query: "SELECT * FROM pts VISUALISE DRAW point MAPPING a AS x, b AS y FILTER g = 'A'",
        },
        // Regression cases from the 2026-10 dialect review. Each mirrors a
        // live-battery case in src/tests/dialect_live.rs and pins the SQL
        // shape per dialect.
        Case {
            name: "binned_temporal",
            table: "ts_data",
            df: DataFrame::new(vec![
                (
                    "day",
                    Arc::new(arrow::array::Date32Array::from(vec![19000, 19001, 19002]))
                        as arrow::array::ArrayRef,
                ),
                (
                    "value",
                    Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 3.0])),
                ),
            ])
            .unwrap(),
            // A non-temporal transform on a temporal column forces the
            // numeric-CASE fallback: the column must be cast to the
            // dialect's number type and quoted per dialect.
            query: "VISUALISE DRAW point MAPPING day AS x, value AS y FROM ts_data SCALE BINNED x VIA identity",
        },
        Case {
            name: "percentile_aggregate",
            table: "box_data",
            df: df! {
                "grp" => vec!["A", "A", "B", "B"],
                "value" => vec![1.0f64, 2.0, 3.0, 4.0],
            }
            .unwrap(),
            // p10 is off the quartiles: pins the ROW_NUMBER-based
            // percentile fallback (and native inline forms) per dialect.
            query: "VISUALISE DRAW point MAPPING grp AS x, value AS y FROM box_data SETTING aggregate => 'y:p10'",
        },
        Case {
            name: "quoted_column",
            table: "box_data",
            df: df! {
                "grp" => vec!["A", "A", "B", "B"],
                "mixed Case" => vec![1.0f64, 2.0, 3.0, 4.0],
            }
            .unwrap(),
            // A column name needing quoting through the quantile path:
            // dialects receive the raw name and must quote it themselves.
            query: "VISUALISE DRAW boxplot MAPPING grp AS x, \"mixed Case\" AS y FROM box_data",
        },
        Case {
            name: "partition_by_quoted",
            table: "pts",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0],
                "mixed Case" => vec![10.0f64, 20.0, 15.0],
            }
            .unwrap(),
            // PARTITION BY identifiers follow the MAPPING convention:
            // stored unquoted, re-quoted via the dialect at emission.
            query: "VISUALISE DRAW point MAPPING a AS x, \"mixed Case\" AS y FROM pts SETTING aggregate => 'y:mean' PARTITION BY \"mixed Case\"",
        },
        Case {
            name: "grouped_bar",
            table: "bar_data",
            df: df! {
                "category" => vec!["a", "b", "a", "b"],
                "grp" => vec!["A", "A", "B", "B"],
            }
            .unwrap(),
            query: "VISUALISE DRAW bar MAPPING category AS x, grp AS fill FROM bar_data",
        },
        Case {
            name: "grouped_histogram",
            table: "hist_data",
            df: df! {
                "value" => vec![1.0f64, 2.0, 2.5, 3.0, 3.5, 4.0],
                "grp" => vec!["A", "A", "A", "B", "B", "B"],
            }
            .unwrap(),
            query: "VISUALISE DRAW histogram MAPPING value AS x, grp AS fill FROM hist_data",
        },
        Case {
            name: "grouped_smooth",
            table: "smooth_data",
            df: df! {
                "a" => vec![1.0f64, 2.0, 3.0, 4.0],
                "b" => vec![2.0f64, 4.0, 5.0, 8.0],
                "grp" => vec!["A", "A", "B", "B"],
            }
            .unwrap(),
            query: "VISUALISE DRAW smooth MAPPING a AS x, b AS y, grp AS color FROM smooth_data SETTING method => 'ols'",
        },
    ]
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

#[test]
fn golden_sql_per_dialect() {
    let bless = std::env::var("GGSQL_BLESS").is_ok();
    let dir = golden_dir();
    let mut failures = Vec::new();

    for (name, dialect) in all_dialects() {
        let (reader, log) = StubReader::new(dialect);
        let mut out = String::new();

        for case in battery() {
            reader.register(case.table, case.df.clone(), true).unwrap();
            out.push_str(&format!("-- case: {}\n", case.name));
            match reader.execute(case.query) {
                Ok(_) => {}
                Err(e) => out.push_str(&format!("-- ERROR: {e}\n")),
            }
            drain_log(&log, &mut out);
            out.push('\n');
            reader.unregister(case.table).unwrap();
        }

        let path = dir.join(format!("{name}.sql"));
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &out).unwrap();
            continue;
        }
        let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!(
                "missing golden file {} — create it with `GGSQL_BLESS=1 cargo test -p ggsql --lib golden`",
                path.display()
            )
        });
        if expected != out {
            failures.push(name.to_string());
        }
    }

    assert!(
        failures.is_empty(),
        "golden SQL mismatch for: {}. Inspect with `cargo test -p ggsql --lib golden -- --nocapture`, \
         then bless intentional changes with GGSQL_BLESS=1.",
        failures.join(", ")
    );
}

fn drain_log(log: &Arc<Mutex<Vec<String>>>, out: &mut String) {
    for stmt in log.lock().unwrap().drain(..) {
        out.push_str(&normalize(&stmt));
        out.push_str(";\n");
    }
}
