//! Thin DataFrame wrapper around Arrow RecordBatch.
//!
//! Provides ergonomic column-by-name access and mutation methods
//! (with_column, rename, drop) that RecordBatch lacks natively.
//! Each mutation returns a new DataFrame (RecordBatch is immutable).

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use std::sync::Arc;

use crate::{GgsqlError, Result};

/// A thin wrapper around Arrow's `RecordBatch` providing named-column access
/// and ergonomic mutation methods.
///
/// Clone is cheap — the underlying arrays are reference-counted.
#[derive(Debug, Clone)]
pub struct DataFrame {
    inner: RecordBatch,
}

impl DataFrame {
    // ========================================================================
    // Construction
    // ========================================================================

    /// Create a DataFrame from named columns.
    ///
    /// All arrays must have the same length. Names accept anything convertible
    /// to `String` (`&str`, `String`, `&String`), so callers can pass either
    /// borrowed or owned names without wrestling with lifetimes.
    pub fn new<N: Into<String>>(columns: Vec<(N, ArrayRef)>) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Self::empty());
        }
        let (names, arrays): (Vec<String>, Vec<ArrayRef>) = columns
            .into_iter()
            .map(|(name, arr)| (name.into(), arr))
            .unzip();
        let fields: Vec<Field> = names
            .into_iter()
            .zip(arrays.iter())
            .map(|(name, arr)| Field::new(name, arr.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let rb = RecordBatch::try_new(schema, arrays)
            .map_err(|e| GgsqlError::InternalError(format!("Failed to create DataFrame: {}", e)))?;
        Ok(Self { inner: rb })
    }

    /// Create an empty DataFrame (0 columns, 0 rows).
    pub fn empty() -> Self {
        Self {
            inner: RecordBatch::new_empty(Arc::new(Schema::empty())),
        }
    }

    /// Wrap an existing RecordBatch.
    pub fn from_record_batch(rb: RecordBatch) -> Self {
        Self { inner: rb }
    }

    // ========================================================================
    // Read access
    // ========================================================================

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.inner.num_rows()
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.inner.num_columns()
    }

    /// (rows, columns) tuple.
    pub fn shape(&self) -> (usize, usize) {
        (self.height(), self.width())
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Result<&ArrayRef> {
        let idx = self.column_index(name)?;
        Ok(self.inner.column(idx))
    }

    /// Get all columns as a slice.
    pub fn get_columns(&self) -> &[ArrayRef] {
        self.inner.columns()
    }

    /// Get column names.
    pub fn get_column_names(&self) -> Vec<String> {
        self.inner
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Get the Arrow schema (reference-counted).
    pub fn schema(&self) -> Arc<Schema> {
        self.inner.schema().clone()
    }

    /// Access the underlying RecordBatch directly.
    pub fn inner(&self) -> &RecordBatch {
        &self.inner
    }

    /// Consume the wrapper and return the RecordBatch.
    pub fn into_inner(self) -> RecordBatch {
        self.inner
    }

    /// Get the data type of a column by name.
    pub fn column_dtype(&self, name: &str) -> Result<DataType> {
        let idx = self.column_index(name)?;
        Ok(self.inner.schema().field(idx).data_type().clone())
    }

    // ========================================================================
    // Mutation (returns new DataFrame)
    // ========================================================================

    /// Add or replace a column. If a column with `name` already exists, it is replaced.
    pub fn with_column(&self, name: &str, array: ArrayRef) -> Result<Self> {
        if array.len() != self.height() && self.width() > 0 {
            return Err(GgsqlError::InternalError(format!(
                "Cannot add column '{}' with {} rows to DataFrame with {} rows",
                name,
                array.len(),
                self.height()
            )));
        }

        let mut fields: Vec<Field> = Vec::with_capacity(self.width() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.width() + 1);
        let mut replaced = false;

        for (i, field) in self.inner.schema().fields().iter().enumerate() {
            if field.name() == name {
                fields.push(Field::new(name, array.data_type().clone(), true));
                arrays.push(array.clone());
                replaced = true;
            } else {
                fields.push(field.as_ref().clone());
                arrays.push(self.inner.column(i).clone());
            }
        }

        if !replaced {
            fields.push(Field::new(name, array.data_type().clone(), true));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let rb = RecordBatch::try_new(schema, arrays).map_err(|e| {
            GgsqlError::InternalError(format!("Failed to add column '{}': {}", name, e))
        })?;
        Ok(Self { inner: rb })
    }

    /// Rename a column.
    pub fn rename(&self, old: &str, new: &str) -> Result<Self> {
        let idx = self.column_index(old)?;

        let fields: Vec<Field> = self
            .inner
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == idx {
                    Field::new(new, f.data_type().clone(), f.is_nullable())
                } else {
                    f.as_ref().clone()
                }
            })
            .collect();

        let schema = Arc::new(Schema::new(fields));
        let rb = RecordBatch::try_new(schema, self.inner.columns().to_vec()).map_err(|e| {
            GgsqlError::InternalError(format!(
                "Failed to rename column '{}' to '{}': {}",
                old, new, e
            ))
        })?;
        Ok(Self { inner: rb })
    }

    /// Drop a column by name. Returns error if column doesn't exist.
    pub fn drop(&self, name: &str) -> Result<Self> {
        let idx = self.column_index(name)?;
        self.drop_by_index(idx)
    }

    /// Drop multiple columns by name. Silently ignores names that don't exist.
    ///
    /// If every column is dropped, the returned DataFrame preserves the original
    /// row count (0 columns × N rows), which annotation layers rely on to know
    /// how many marks to draw.
    pub fn drop_many<S: AsRef<str>>(&self, names: &[S]) -> Result<Self> {
        let drop_set: std::collections::HashSet<&str> = names.iter().map(|s| s.as_ref()).collect();

        let mut fields = Vec::new();
        let mut arrays = Vec::new();

        for (i, field) in self.inner.schema().fields().iter().enumerate() {
            if !drop_set.contains(field.name().as_str()) {
                fields.push(field.as_ref().clone());
                arrays.push(self.inner.column(i).clone());
            }
        }

        build_record_batch(fields, arrays, self.height())
            .map(|inner| Self { inner })
            .map_err(|e| GgsqlError::InternalError(format!("Failed to drop columns: {}", e)))
    }

    /// Replace a column's array (keeping the same name).
    pub fn replace(&self, name: &str, array: ArrayRef) -> Result<Self> {
        let idx = self.column_index(name)?;

        if array.len() != self.height() {
            return Err(GgsqlError::InternalError(format!(
                "Replacement column '{}' has {} rows, expected {}",
                name,
                array.len(),
                self.height()
            )));
        }

        let fields: Vec<Field> = self
            .inner
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == idx {
                    Field::new(name, array.data_type().clone(), f.is_nullable())
                } else {
                    f.as_ref().clone()
                }
            })
            .collect();

        let mut arrays: Vec<ArrayRef> = self.inner.columns().to_vec();
        arrays[idx] = array;

        let schema = Arc::new(Schema::new(fields));
        let rb = RecordBatch::try_new(schema, arrays).map_err(|e| {
            GgsqlError::InternalError(format!("Failed to replace column '{}': {}", name, e))
        })?;
        Ok(Self { inner: rb })
    }

    /// Slice the DataFrame (offset and length).
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        Self {
            inner: self.inner.slice(offset, length),
        }
    }

    /// Return a new DataFrame containing the rows at `indices` (in order).
    ///
    /// Wraps `arrow::compute::take` across every column.
    pub fn take(&self, indices: &arrow::array::UInt32Array) -> Result<Self> {
        let names = self.get_column_names();
        let mut new_cols: Vec<(&str, ArrayRef)> = Vec::with_capacity(self.width());
        for (i, name) in names.iter().enumerate() {
            let taken = arrow::compute::take(self.inner.column(i).as_ref(), indices, None)
                .map_err(|e| {
                    GgsqlError::InternalError(format!("Failed to take column '{}': {}", name, e))
                })?;
            new_cols.push((name, taken));
        }
        Self::new(new_cols)
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    fn column_index(&self, name: &str) -> Result<usize> {
        self.inner
            .schema()
            .index_of(name)
            .map_err(|_| GgsqlError::InternalError(format!("Column '{}' not found", name)))
    }

    fn drop_by_index(&self, idx: usize) -> Result<Self> {
        let mut fields = Vec::with_capacity(self.width() - 1);
        let mut arrays = Vec::with_capacity(self.width() - 1);

        for (i, field) in self.inner.schema().fields().iter().enumerate() {
            if i != idx {
                fields.push(field.as_ref().clone());
                arrays.push(self.inner.column(i).clone());
            }
        }

        build_record_batch(fields, arrays, self.height())
            .map(|inner| Self { inner })
            .map_err(|e| {
                GgsqlError::InternalError(format!("Failed to drop column at index {}: {}", idx, e))
            })
    }
}

/// Build a `RecordBatch`, preserving `row_count` even when the schema has no fields.
///
/// Arrow's default constructor discards the row count for zero-column batches,
/// which would silently lose "N rows × 0 columns" — a state annotation layers
/// depend on (they draw one mark per row regardless of data columns).
fn build_record_batch(
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
    row_count: usize,
) -> std::result::Result<RecordBatch, arrow::error::ArrowError> {
    let schema = Arc::new(Schema::new(fields));
    if arrays.is_empty() {
        let options = RecordBatchOptions::new().with_row_count(Some(row_count));
        RecordBatch::try_new_with_options(schema, arrays, &options)
    } else {
        RecordBatch::try_new(schema, arrays)
    }
}

/// Convenience macro for creating test DataFrames, similar to polars' `df!`.
///
/// # Examples
///
/// ```ignore
/// let df = df! {
///     "name" => vec!["Alice", "Bob"],
///     "age" => vec![30i32, 25],
/// }.unwrap();
/// ```
#[macro_export]
macro_rules! df {
    ($($col_name:expr => $values:expr),+ $(,)?) => {{
        {
            let columns: Vec<(&str, arrow::array::ArrayRef)> = vec![
                $(
                    ($col_name, $crate::dataframe::into_array_ref($values)),
                )+
            ];
            $crate::dataframe::DataFrame::new(columns)
        }
    }};
}

// ============================================================================
// Conversion helpers for the df! macro
// ============================================================================

/// Convert typed Vecs into ArrayRef. Used by the `df!` macro.
pub fn into_array_ref<T: IntoArrayRef>(values: T) -> ArrayRef {
    values.into_array_ref()
}

/// Trait for converting typed collections into Arrow arrays.
pub trait IntoArrayRef {
    fn into_array_ref(self) -> ArrayRef;
}

// --- Vec<f64> ---
impl IntoArrayRef for Vec<f64> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Float64Array::from(self))
    }
}

// --- Vec<Option<f64>> ---
impl IntoArrayRef for Vec<Option<f64>> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Float64Array::from(self))
    }
}

// --- Vec<i32> ---
impl IntoArrayRef for Vec<i32> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Int32Array::from(self))
    }
}

// --- Vec<Option<i32>> ---
impl IntoArrayRef for Vec<Option<i32>> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Int32Array::from(self))
    }
}

// --- Vec<i64> ---
impl IntoArrayRef for Vec<i64> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Int64Array::from(self))
    }
}

// --- Vec<Option<i64>> ---
impl IntoArrayRef for Vec<Option<i64>> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::Int64Array::from(self))
    }
}

// --- Vec<bool> ---
impl IntoArrayRef for Vec<bool> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::BooleanArray::from(self))
    }
}

// --- Vec<Option<bool>> ---
impl IntoArrayRef for Vec<Option<bool>> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::BooleanArray::from(self))
    }
}

// --- Vec<&str> ---
impl IntoArrayRef for Vec<&str> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::StringArray::from(self))
    }
}

// --- Vec<Option<&str>> ---
impl IntoArrayRef for Vec<Option<&str>> {
    fn into_array_ref(self) -> ArrayRef {
        Arc::new(arrow::array::StringArray::from(self))
    }
}

// --- Vec<String> ---
impl IntoArrayRef for Vec<String> {
    fn into_array_ref(self) -> ArrayRef {
        let refs: Vec<&str> = self.iter().map(|s| s.as_str()).collect();
        Arc::new(arrow::array::StringArray::from(refs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int32Array, StringArray};

    #[test]
    fn test_new_and_accessors() {
        let df = DataFrame::new(vec![
            ("x", Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef),
            (
                "y",
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
            ),
        ])
        .unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(
            df.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
        assert!(df.column("x").is_ok());
        assert!(df.column("z").is_err());
    }

    #[test]
    fn test_empty() {
        let df = DataFrame::empty();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 0);
    }

    #[test]
    fn test_with_column_add() {
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df
            .with_column(
                "y",
                Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
            )
            .unwrap();

        assert_eq!(df2.width(), 2);
        assert_eq!(
            df2.get_column_names(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_with_column_replace() {
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df
            .with_column("x", Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef)
            .unwrap();

        assert_eq!(df2.width(), 1);
        let col = df2.column("x").unwrap();
        let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(arr.value(0), 10);
    }

    #[test]
    fn test_rename() {
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df.rename("x", "renamed").unwrap();
        assert!(df2.column("renamed").is_ok());
        assert!(df2.column("x").is_err());
    }

    #[test]
    fn test_drop() {
        let df = DataFrame::new(vec![
            ("x", Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef),
            (
                "y",
                Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            ),
        ])
        .unwrap();

        let df2 = df.drop("x").unwrap();
        assert_eq!(df2.width(), 1);
        assert_eq!(df2.get_column_names(), vec!["y"]);
    }

    #[test]
    fn test_drop_many() {
        let df = DataFrame::new(vec![
            ("a", Arc::new(Int32Array::from(vec![1])) as ArrayRef),
            ("b", Arc::new(Int32Array::from(vec![2])) as ArrayRef),
            ("c", Arc::new(Int32Array::from(vec![3])) as ArrayRef),
        ])
        .unwrap();

        let df2 = df.drop_many(&["a", "c"]).unwrap();
        assert_eq!(df2.get_column_names(), vec!["b".to_string()]);
    }

    #[test]
    fn test_drop_last_column_preserves_row_count() {
        // Annotation layers rely on "N rows × 0 columns" to know how many marks to draw.
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df.drop("x").unwrap();
        assert_eq!(df2.width(), 0);
        assert_eq!(df2.height(), 3);

        let df3 = df.drop_many(&["x"]).unwrap();
        assert_eq!(df3.width(), 0);
        assert_eq!(df3.height(), 3);
    }

    #[test]
    fn test_replace() {
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df
            .replace("x", Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef)
            .unwrap();
        let arr = df2
            .column("x")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(arr.value(0), 10);
        assert_eq!(arr.value(1), 20);
    }

    #[test]
    fn test_slice() {
        let df = DataFrame::new(vec![(
            "x",
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
        )])
        .unwrap();

        let df2 = df.slice(1, 3);
        assert_eq!(df2.height(), 3);
        let arr = df2
            .column("x")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(arr.value(0), 2);
        assert_eq!(arr.value(2), 4);
    }

    #[test]
    fn test_df_macro() {
        let df = df! {
            "name" => vec!["Alice", "Bob"],
            "age" => vec![30i32, 25],
            "score" => vec![95.5, 87.3],
        }
        .unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 3);
        assert_eq!(
            df.get_column_names(),
            vec!["name".to_string(), "age".to_string(), "score".to_string()]
        );

        let names = df
            .column("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "Alice");
    }

    #[test]
    fn test_df_macro_with_optionals() {
        let df = df! {
            "x" => vec![Some(1.0), None, Some(3.0)],
        }
        .unwrap();

        assert_eq!(df.height(), 3);
        let col = df
            .column("x")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(!col.is_null(0));
        assert!(col.is_null(1));
        assert!(!col.is_null(2));
    }
}
