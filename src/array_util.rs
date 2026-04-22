//! Typed array access helpers for Arrow arrays.
//!
//! Replaces the polars pattern of `series.f64()`, `series.str()`, etc.
//! with arrow downcasting via `as_f64(array)`, `as_str(array)`, etc.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, StringArray, Time64NanosecondArray,
    TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::compute;
use arrow::datatypes::DataType;
use std::sync::Arc;

use crate::{GgsqlError, Result};

// ============================================================================
// Downcast helpers
// ============================================================================

macro_rules! downcast_fn {
    ($fn_name:ident, $arrow_type:ty, $type_name:expr) => {
        pub fn $fn_name(array: &ArrayRef) -> Result<&$arrow_type> {
            array.as_any().downcast_ref::<$arrow_type>().ok_or_else(|| {
                GgsqlError::InternalError(format!(
                    "Expected {} array, got {:?}",
                    $type_name,
                    array.data_type()
                ))
            })
        }
    };
}

downcast_fn!(as_f64, Float64Array, "Float64");
downcast_fn!(as_f32, Float32Array, "Float32");
downcast_fn!(as_i64, Int64Array, "Int64");
downcast_fn!(as_i32, Int32Array, "Int32");
downcast_fn!(as_i16, Int16Array, "Int16");
downcast_fn!(as_i8, Int8Array, "Int8");
downcast_fn!(as_u64, UInt64Array, "UInt64");
downcast_fn!(as_u32, UInt32Array, "UInt32");
downcast_fn!(as_u16, UInt16Array, "UInt16");
downcast_fn!(as_u8, UInt8Array, "UInt8");
downcast_fn!(as_str, StringArray, "String");
downcast_fn!(as_bool, BooleanArray, "Boolean");
downcast_fn!(as_date32, Date32Array, "Date32");
downcast_fn!(
    as_timestamp_us,
    TimestampMicrosecondArray,
    "Timestamp(Microsecond)"
);
downcast_fn!(as_time64_ns, Time64NanosecondArray, "Time64(Nanosecond)");

// ============================================================================
// Cast helper
// ============================================================================

/// Cast an array to a different data type.
pub fn cast_array(array: &ArrayRef, to: &DataType) -> Result<ArrayRef> {
    compute::cast(array, to)
        .map_err(|e| GgsqlError::InternalError(format!("Failed to cast to {:?}: {}", to, e)))
}

// ============================================================================
// Array construction helpers
// ============================================================================

/// Create a Float64 array from optional values.
pub fn new_f64_array(values: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

/// Create a Float64 array from non-null values.
pub fn new_f64_array_non_null(values: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

/// Create an Int32 array from optional values.
pub fn new_i32_array(values: Vec<Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

/// Create an Int64 array from optional values.
pub fn new_i64_array(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

/// Create a String array from optional values.
pub fn new_str_array(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

/// Create a String array from owned strings.
pub fn new_string_array(values: Vec<String>) -> ArrayRef {
    let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    Arc::new(StringArray::from(refs))
}

/// Create a Boolean array from optional values.
pub fn new_bool_array(values: Vec<Option<bool>>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
}

/// Create a constant Float64 array (all same value).
pub fn new_constant_f64(value: f64, len: usize) -> ArrayRef {
    Arc::new(Float64Array::from(vec![value; len]))
}

/// Create a constant String array (all same value).
pub fn new_constant_str(value: &str, len: usize) -> ArrayRef {
    Arc::new(StringArray::from(vec![value; len]))
}

/// Create a constant Boolean array (all same value).
pub fn new_constant_bool(value: bool, len: usize) -> ArrayRef {
    Arc::new(BooleanArray::from(vec![value; len]))
}

// ============================================================================
// Null handling
// ============================================================================

/// Replace null values in a Float64 array with a fill value.
pub fn fill_null_f64(array: &Float64Array, fill: f64) -> Float64Array {
    let mut builder = arrow::array::Float64Builder::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_value(fill);
        } else {
            builder.append_value(array.value(i));
        }
    }
    builder.finish()
}

// ============================================================================
// Value extraction helpers
// ============================================================================

/// Get a string representation of a value at an index, for any array type.
/// Used for building composite group keys (e.g., in dodge).
pub fn value_to_string(array: &ArrayRef, idx: usize) -> String {
    if array.is_null(idx) {
        return "null".to_string();
    }
    match array.data_type() {
        DataType::Int8 => as_i8(array).unwrap().value(idx).to_string(),
        DataType::Int16 => as_i16(array).unwrap().value(idx).to_string(),
        DataType::Int32 => as_i32(array).unwrap().value(idx).to_string(),
        DataType::Int64 => as_i64(array).unwrap().value(idx).to_string(),
        DataType::UInt8 => as_u8(array).unwrap().value(idx).to_string(),
        DataType::UInt16 => as_u16(array).unwrap().value(idx).to_string(),
        DataType::UInt32 => as_u32(array).unwrap().value(idx).to_string(),
        DataType::UInt64 => as_u64(array).unwrap().value(idx).to_string(),
        DataType::Float32 => as_f32(array).unwrap().value(idx).to_string(),
        DataType::Float64 => as_f64(array).unwrap().value(idx).to_string(),
        DataType::Utf8 => as_str(array).unwrap().value(idx).to_string(),
        DataType::Boolean => as_bool(array).unwrap().value(idx).to_string(),
        _ => format!("{:?}", array.data_type()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downcast_f64() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let f64_arr = as_f64(&arr).unwrap();
        assert_eq!(f64_arr.value(0), 1.0);
        assert_eq!(f64_arr.value(2), 3.0);
    }

    #[test]
    fn test_downcast_wrong_type() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        assert!(as_f64(&arr).is_err());
    }

    #[test]
    fn test_cast_array() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let casted = cast_array(&arr, &DataType::Float64).unwrap();
        let f64_arr = as_f64(&casted).unwrap();
        assert_eq!(f64_arr.value(0), 1.0);
    }

    #[test]
    fn test_fill_null_f64() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let filled = fill_null_f64(&arr, 0.0);
        assert_eq!(filled.value(0), 1.0);
        assert_eq!(filled.value(1), 0.0);
        assert_eq!(filled.value(2), 3.0);
        assert!(!filled.is_null(1));
    }

    #[test]
    fn test_new_constant_f64() {
        let arr = new_constant_f64(42.0, 3);
        let f64_arr = as_f64(&arr).unwrap();
        assert_eq!(f64_arr.len(), 3);
        assert_eq!(f64_arr.value(0), 42.0);
        assert_eq!(f64_arr.value(2), 42.0);
    }

    #[test]
    fn test_value_to_string() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["hello", "world"]));
        assert_eq!(value_to_string(&arr, 0), "hello");

        let arr: ArrayRef = Arc::new(Float64Array::from(vec![3.14]));
        assert_eq!(value_to_string(&arr, 0), "3.14");
    }
}
