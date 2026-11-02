//! Canonical dialect-conformance battery, shared by the Tier 1 golden tests
//! (`src/execute/golden.rs`) and the Tier 2 live-backend tests
//! (`src/tests/dialect_live.rs`). Both include this file via `#[path]` so
//! the case list and dataset exist exactly once; it lives in a
//! subdirectory so cargo does not treat it as its own test target.
//!
//! Data plus helpers over `arrow`/`crate::DataFrame` (`fixture_df`); both
//! including contexts are targets of this crate, so those deps resolve in
//! either one.

// Each including tier reads only its own subset: the golden tests ignore
// the live assertions (Expect, live_skip, live, runs_live_on), and the
// live harness ignores fixture_csv. Without this allow each inclusion
// warns the other tier's half dead.
#![allow(dead_code)]

/// The one table every battery case queries. The golden tests register a
/// fabricated DataFrame under this name; the live harness CREATEs and
/// populates it (or, on BigQuery, register()s it; on Druid, the start
/// script seeds it from fixture.csv).
pub const TABLE: &str = "ggsql_live_test";

/// One dataset row. `day_epoch` is days since the unix epoch
/// (2022-01-01 = 18993). Eight rows, four per group: density/violin
/// compute a Silverman bandwidth from NTILE(4) tiles, which is degenerate
/// (NULL) with fewer than four values per group — their grids would come
/// back empty and the battery would pass vacuously.
pub struct Row {
    pub id: i32,
    pub val: f64,
    pub grp: &'static str,
    pub day_epoch: i32,
    pub mixed: f64,
}

pub const ROWS: &[Row] = &[
    Row {
        id: 1,
        val: 1.5,
        grp: "a",
        day_epoch: 18993,
        mixed: 10.5,
    },
    Row {
        id: 2,
        val: 2.5,
        grp: "b",
        day_epoch: 18994,
        mixed: 11.5,
    },
    Row {
        id: 3,
        val: 3.5,
        grp: "a",
        day_epoch: 18995,
        mixed: 12.5,
    },
    Row {
        id: 4,
        val: 4.5,
        grp: "b",
        day_epoch: 18996,
        mixed: 13.5,
    },
    Row {
        id: 5,
        val: 5.5,
        grp: "a",
        day_epoch: 18997,
        mixed: 14.5,
    },
    Row {
        id: 6,
        val: 6.5,
        grp: "b",
        day_epoch: 18998,
        mixed: 15.5,
    },
    Row {
        id: 7,
        val: 7.5,
        grp: "a",
        day_epoch: 18999,
        mixed: 16.5,
    },
    Row {
        id: 8,
        val: 8.5,
        grp: "b",
        day_epoch: 19000,
        mixed: 17.5,
    },
];

/// The `day` column spelled as an ISO date (`2022-01-0N`).
pub fn day_iso(day_epoch: i32) -> String {
    // All fixture days are in January 2022; keep the helper honest anyway.
    assert!((18993..=19000).contains(&day_epoch));
    format!("2022-01-{:02}", day_epoch - 18992)
}

/// The dataset as CSV with header — the single source for seeds that can't
/// run Rust (the Druid MSQ ingest in .github/scripts/live/druid.sh).
/// `.github/scripts/live/ggsql_live_test.csv` must match this exactly; the
/// `fixture_csv_is_current` test fails on drift.
pub fn fixture_csv() -> String {
    let mut out = String::from("id,val,grp,day,mixed Case\n");
    for r in ROWS {
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            r.id,
            r.val,
            r.grp,
            day_iso(r.day_epoch),
            r.mixed
        ));
    }
    out
}

/// What the live harness asserts about layer 0 after running a case.
/// Golden tests ignore this — they pin the SQL itself.
pub enum Expect {
    /// The pipeline runs without error.
    Runs,
    /// Layer 0 has at least this many rows (guards against vacuous passes:
    /// a broken stat can return zero rows and otherwise succeed).
    MinRows(usize),
    /// Layer 0 has exactly this many rows.
    ExactRows(usize),
    /// The named column's row-0 value falls within [lo, hi] (percentile
    /// assertions; bounds wide enough for approximate-quantile engines).
    F64In(&'static str, f64, f64),
}

/// Fixture variant for cases that can't run against the shared table.
/// Non-Shared fixtures are golden-only: the live harness skips them.
pub enum Fixture {
    /// The shared 8-row table.
    Shared,
    /// The shared table with `day` as ISO text instead of a date — forces
    /// the text→temporal CAST path (SAFE_CAST/TRY_CAST per dialect).
    TextDay,
}

/// The fixture as a DataFrame, matching the live table's schema:
/// id/val/grp/day(a real date)/"mixed Case". `TextDay` swaps the date for
/// ISO text. Used by the golden tier's stub registrations and by live
/// backends whose setup goes through `register()` rather than DDL
/// (DataFusion). BigQuery is the exception: its ADBC register path needs
/// Int64 ids, so `dialect_live.rs` builds that batch by hand.
pub fn fixture_df(fixture: &Fixture) -> crate::DataFrame {
    use arrow::array::{ArrayRef, Date32Array, Float64Array, Int32Array, StringArray};
    use std::sync::Arc;
    let day: ArrayRef = match fixture {
        Fixture::Shared => Arc::new(Date32Array::from(
            ROWS.iter().map(|r| r.day_epoch).collect::<Vec<_>>(),
        )),
        Fixture::TextDay => Arc::new(StringArray::from(
            ROWS.iter()
                .map(|r| day_iso(r.day_epoch))
                .collect::<Vec<_>>(),
        )),
    };
    crate::DataFrame::new(vec![
        (
            "id",
            Arc::new(Int32Array::from(
                ROWS.iter().map(|r| r.id).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "val",
            Arc::new(Float64Array::from(
                ROWS.iter().map(|r| r.val).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "grp",
            Arc::new(StringArray::from(
                ROWS.iter().map(|r| r.grp).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("day", day),
        (
            "mixed Case",
            Arc::new(Float64Array::from(
                ROWS.iter().map(|r| r.mixed).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap()
}

pub struct Case {
    pub name: &'static str,
    /// Query template; `{table}` is substituted by the harness.
    pub query: &'static str,
    pub expect: Expect,
    /// Schemes the case is skipped on live, with the reason. For golden
    /// tests only `fixture` matters.
    pub live_skip: &'static [(&'static str, &'static str)],
    pub fixture: Fixture,
    /// False for golden-only cases (e.g. spatial: only some backends
    /// support it, so there is nothing to assert across the live matrix).
    pub live: bool,
}

const NO_SKIP: &[(&str, &str)] = &[];

const fn shared(name: &'static str, query: &'static str, expect: Expect) -> Case {
    Case {
        name,
        query,
        expect,
        live_skip: NO_SKIP,
        fixture: Fixture::Shared,
        live: true,
    }
}

/// The canonical battery. Every case runs in both tiers unless its
/// `live_skip` or `fixture` says otherwise; assertions (`expect`) apply
/// only live, SQL-shape pinning only in goldens.
pub fn cases() -> Vec<Case> {
    vec![
        shared(
            "scatter_color",
            "VISUALISE DRAW point MAPPING id AS x, val AS y, grp AS color FROM {table}",
            Expect::ExactRows(8),
        ),
        shared(
            "histogram",
            "VISUALISE DRAW histogram MAPPING val AS x FROM {table}",
            Expect::MinRows(1),
        ),
        // Grouped boxplot: dialect quantile forms and qualified
        // projections. Two groups always yield whisker/box/median rows.
        shared(
            "boxplot_grouped",
            "VISUALISE DRAW boxplot MAPPING grp AS x, val AS y FROM {table}",
            Expect::MinRows(1),
        ),
        // Line with an aggregating stat: the geom's required ordering is
        // carried as data on the stat result and applied on the final,
        // outermost query only, so no ORDER BY is nested inside a derived
        // table (which T-SQL rejects with error 1033).
        shared(
            "line_aggregate",
            "VISUALISE DRAW line MAPPING grp AS x, val AS y FROM {table} \
             SETTING aggregate => 'y:mean'",
            Expect::ExactRows(2),
        ),
        // Grouped density: cross-group grid join with qualified
        // projections. A degenerate (NULL) bandwidth would silently
        // produce an empty result.
        shared(
            "density_grouped",
            "VISUALISE DRAW density MAPPING val AS x, grp AS color FROM {table}",
            Expect::MinRows(1),
        ),
        shared(
            "line_temporal",
            "SELECT * FROM {table} VISUALISE day AS x, val AS y DRAW line",
            Expect::ExactRows(8),
        ),
        // Global-SQL source + multi-reference stat: the source is
        // materialized as an internal table and the boxplot stats query
        // references it several times in one statement. MySQL/MariaDB refuse
        // to open a *temporary* table twice in one statement (error 1137,
        // "Can't reopen table"), so their dialect materializes internal
        // tables as regular tables (TempTableStyle::DropThenCreate), which
        // have no such restriction.
        shared(
            "boxplot_global_source",
            "SELECT * FROM {table} VISUALISE DRAW boxplot MAPPING grp AS x, val AS y",
            Expect::MinRows(1),
        ),
        shared(
            "bar_count",
            "VISUALISE DRAW bar MAPPING grp AS x FROM {table}",
            Expect::ExactRows(2),
        ),
        shared(
            "smooth_lm",
            "VISUALISE DRAW smooth MAPPING id AS x, val AS y FROM {table} SETTING method => 'ols'",
            Expect::Runs,
        ),
        shared(
            "ribbon",
            "VISUALISE DRAW ribbon MAPPING id AS x, id AS ymin, val AS ymax FROM {table}",
            Expect::Runs,
        ),
        shared(
            "segment",
            "VISUALISE DRAW segment MAPPING id AS x, val AS y, id AS xend, val AS yend FROM {table}",
            Expect::Runs,
        ),
        shared(
            "tile",
            "VISUALISE DRAW tile MAPPING val AS x, id AS y FROM {table}",
            Expect::MinRows(1),
        ),
        shared(
            "area",
            "VISUALISE DRAW area MAPPING id AS x, val AS y FROM {table}",
            Expect::Runs,
        ),
        shared(
            "violin",
            "VISUALISE DRAW violin MAPPING grp AS x, val AS y FROM {table}",
            Expect::MinRows(1),
        ),
        shared(
            "filter_layer",
            "SELECT * FROM {table} VISUALISE DRAW point MAPPING id AS x, val AS y FILTER grp = 'a'",
            Expect::ExactRows(4),
        ),
        // Binned scale over a temporal column with a non-temporal
        // transform: the numeric-CASE fallback must quote the column with
        // the active dialect and convert temporal→number per dialect
        // (sql_temporal_as_number). Live, a zero-row result re-runs the
        // pipeline's extent and CASE statements for diagnosis.
        shared(
            "binned_temporal",
            "VISUALISE DRAW point MAPPING day AS x, val AS y FROM {table} SCALE BINNED x VIA identity",
            Expect::MinRows(1),
        ),
        shared(
            "binned_temporal_date",
            "VISUALISE DRAW point MAPPING day AS x, val AS y FROM {table} SCALE BINNED x VIA date",
            Expect::MinRows(1),
        ),
        shared(
            "sum_aggregate",
            "VISUALISE DRAW bar MAPPING grp AS x, val AS y FROM {table} SETTING aggregate => 'y:sum'",
            Expect::ExactRows(2),
        ),
        // Percentile aggregates off the quartiles, with real value
        // assertions: group a is {1.5, 3.5, 5.5, 7.5}, so the correct p10
        // is ~2.1 and the correct p90 is ~6.9.
        shared(
            "percentile_p10",
            "VISUALISE DRAW point MAPPING grp AS x, val AS y FROM {table} \
             SETTING aggregate => 'y:p10' FILTER grp = 'a'",
            Expect::F64In("__ggsql_aes_pos2__", 1.5, 2.35),
        ),
        shared(
            "percentile_p90",
            "VISUALISE DRAW point MAPPING grp AS x, val AS y FROM {table} \
             SETTING aggregate => 'y:p90' FILTER grp = 'a'",
            Expect::F64In("__ggsql_aes_pos2__", 6.5, 7.5),
        ),
        // A column whose name needs quoting, through the quantile path:
        // stat_aggregate passes the raw name to sql_quantile, so dialects
        // must quote it themselves.
        shared(
            "quoted_column",
            "VISUALISE DRAW boxplot MAPPING grp AS x, \"mixed Case\" AS y FROM {table}",
            Expect::MinRows(1),
        ),
        // PARTITION BY follows the same identifier rules as MAPPING: the
        // name is stored unquoted and re-quoted via the dialect. With a
        // float column every row is its own group, so the mean is the
        // value itself.
        shared(
            "partition_by_quoted",
            "VISUALISE DRAW point MAPPING id AS x, \"mixed Case\" AS y FROM {table} \
             SETTING aggregate => 'y:mean' PARTITION BY \"mixed Case\"",
            Expect::ExactRows(8),
        ),
        // Grouped variants of the derived-table geoms: the group columns
        // join the partition columns into stat queries, exercising grouped
        // window and derived-table paths the ungrouped cases miss
        // (unaliased derived tables break MySQL/MariaDB).
        shared(
            "grouped_bar",
            "VISUALISE DRAW bar MAPPING grp AS x, grp AS fill FROM {table}",
            Expect::Runs,
        ),
        shared(
            "grouped_histogram",
            "VISUALISE DRAW histogram MAPPING val AS x, grp AS fill FROM {table}",
            Expect::MinRows(1),
        ),
        shared(
            "grouped_smooth",
            "VISUALISE DRAW smooth MAPPING id AS x, val AS y, grp AS color FROM {table} \
             SETTING method => 'ols'",
            Expect::Runs,
        ),
        // Spatial: a map projection routes position columns through the
        // dialect's spatial hooks (ST_Point, ST_Contains, …). Golden-only:
        // only some backends support spatial, so there is nothing to
        // assert live across the matrix.
        Case {
            name: "spatial_mercator",
            query: "VISUALISE DRAW point MAPPING id AS lon, val AS lat FROM {table} \
                    PROJECT TO mercator",
            live: false,
            ..shared("", "", Expect::Runs)
        },
        // Text-typed temporal column under a temporal transform: the
        // pipeline must CAST the column per dialect (SAFE_CAST/TRY_CAST
        // semantics). Golden-only via its fixture variant.
        Case {
            name: "text_date_cast",
            query: "VISUALISE DRAW point MAPPING day AS x, val AS y FROM {table} SCALE x VIA date",
            fixture: Fixture::TextDay,
            live: false,
            ..shared("", "", Expect::Runs)
        },
    ]
}

impl Case {
    /// Whether the live harness runs this case on `scheme`.
    pub fn runs_live_on(&self, scheme: &str) -> bool {
        self.live
            && matches!(self.fixture, Fixture::Shared)
            && !self.live_skip.iter().any(|(s, _)| *s == scheme)
    }
}
