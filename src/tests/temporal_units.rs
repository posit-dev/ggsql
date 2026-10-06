//! Temporal columns typed Timestamp (not Date32) through the battery's
//! temporal cases. Oracle's ODBC driver surfaces DATE as SQL_TYPE_TIMESTAMP,
//! so this is the shape the Oracle live leg trains scales on; a binned scale
//! with `VIA date` used to feed microseconds to day-based break math and
//! panic in chrono::Duration::days.
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
