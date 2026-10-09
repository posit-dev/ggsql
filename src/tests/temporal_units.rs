//! Temporal columns typed Timestamp (not Date32) through the battery's
//! temporal cases. Oracle's ODBC driver surfaces DATE as SQL_TYPE_TIMESTAMP,
//! so this is the shape the Oracle live leg trains scales on. The pinned
//! failure mode: a binned scale with `VIA date` feeding microseconds to
//! day-based break math panics in chrono::Duration::days.
#![cfg(feature = "duckdb")]

use ggsql::reader::connection::reader_from_uri;
use ggsql::DataFrame;
use std::sync::Arc;

fn ts_reader() -> Box<dyn ggsql::reader::Reader + Send> {
    use arrow::array::{Float64Array, Int32Array, TimestampMicrosecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;

    let reader = reader_from_uri("duckdb://memory").unwrap();
    let days = [18993i32, 18994, 18995, 18996, 18997, 18998, 18999, 19000];
    let micros: Vec<i64> = days.iter().map(|d| *d as i64 * 86_400_000_000).collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Float64, false),
            Field::new(
                "day",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(Float64Array::from(vec![
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            ])),
            Arc::new(TimestampMicrosecondArray::from(micros)),
        ],
    )
    .unwrap();
    reader
        .register("ts_test", DataFrame::from_record_batch(batch), true)
        .unwrap();
    reader
}

#[test]
fn line_temporal_timestamp() {
    let reader = ts_reader();
    let spec = reader
        .execute("SELECT * FROM ts_test VISUALISE day AS x, val AS y DRAW line")
        .unwrap();
    assert_eq!(spec.layer_data(0).unwrap().height(), 8);
}

#[test]
fn binned_temporal_date_timestamp() {
    let reader = ts_reader();
    let spec = reader
        .execute(
            "VISUALISE DRAW point MAPPING day AS x, val AS y FROM ts_test SCALE BINNED x VIA date",
        )
        .unwrap();
    assert!(spec.layer_data(0).unwrap().height() > 0);
}

#[test]
fn binned_temporal_identity_timestamp() {
    let reader = ts_reader();
    let spec = reader
        .execute(
            "VISUALISE DRAW point MAPPING day AS x, val AS y FROM ts_test SCALE BINNED x VIA identity",
        )
        .unwrap();
    assert!(spec.layer_data(0).unwrap().height() > 0);
}

/// A numeric epoch-milliseconds column under `VIA date` must error, not
/// panic: the date transform's break math overflows chrono::Duration::days
/// on values that large. This is the shape a driver produces when it
/// surfaces a LONG-based date with no temporal type (the Druid leg's
/// original failure).
#[test]
fn binned_temporal_date_millis_column_errors() {
    use arrow::array::Int64Array;

    let reader = reader_from_uri("duckdb://memory").unwrap();
    let days = [18993i64, 18994, 18995, 18996, 18997, 18998, 18999, 19000];
    let millis: Vec<i64> = days.iter().map(|d| *d * 86_400_000).collect();
    let batch = arrow::record_batch::RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("day", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("val", arrow::datatypes::DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(millis)),
            Arc::new(arrow::array::Float64Array::from(vec![
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            ])),
        ],
    )
    .unwrap();
    reader
        .register("ms_test", DataFrame::from_record_batch(batch), true)
        .unwrap();

    let result = reader.execute(
        "VISUALISE DRAW point MAPPING day AS x, val AS y FROM ms_test SCALE BINNED x VIA date",
    );
    let err = match result {
        Ok(_) => panic!("millis-as-days must be a proper error, not a panic"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("outside the representable range"),
        "unexpected error: {msg}"
    );
}
