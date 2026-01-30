// Allow useless_conversion due to false positive from pyo3 macro expansion
// See: https://github.com/PyO3/pyo3/issues/4327
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::io::Cursor;

use ggsql::api::{prepare as rust_prepare, validate as rust_validate, Prepared, ValidationWarning};
use ggsql::reader::{DuckDBReader as RustDuckDBReader, Reader};
use ggsql::writer::{VegaLiteWriter as RustVegaLiteWriter, Writer};

use polars::prelude::{DataFrame, IpcReader, IpcWriter, SerReader, SerWriter};

// ============================================================================
// Custom Exception Types
// ============================================================================

// Base exception for all ggsql errors
pyo3::create_exception!(ggsql, PyGgsqlError, pyo3::exceptions::PyException);

// Specific exception types
pyo3::create_exception!(ggsql, PyParseError, PyGgsqlError);
pyo3::create_exception!(ggsql, PyValidationError, PyGgsqlError);
pyo3::create_exception!(ggsql, PyReaderError, PyGgsqlError);
pyo3::create_exception!(ggsql, PyWriterError, PyGgsqlError);
pyo3::create_exception!(ggsql, NoVisualiseError, PyGgsqlError);

/// Convert a GgsqlError to the appropriate Python exception
fn ggsql_error_to_pyerr(e: ggsql::GgsqlError) -> PyErr {
    use ggsql::GgsqlError;
    match e {
        GgsqlError::ParseError(msg) => PyParseError::new_err(msg),
        GgsqlError::ValidationError(msg) => PyValidationError::new_err(msg),
        GgsqlError::ReaderError(msg) => PyReaderError::new_err(msg),
        GgsqlError::WriterError(msg) => PyWriterError::new_err(msg),
        GgsqlError::NoVisualise => {
            NoVisualiseError::new_err("Query has no VISUALISE clause".to_string())
        }
        GgsqlError::InternalError(msg) => PyGgsqlError::new_err(format!("Internal error: {}", msg)),
    }
}

// ============================================================================
// Helper Functions for DataFrame Conversion
// ============================================================================

/// Convert a Polars DataFrame to a Python polars DataFrame via IPC serialization
fn polars_to_py(py: Python<'_>, df: &DataFrame) -> PyResult<Py<PyAny>> {
    let mut buffer = Vec::new();
    IpcWriter::new(&mut buffer)
        .finish(&mut df.clone())
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to serialize DataFrame: {}",
                e
            ))
        })?;

    let io = py.import("io")?;
    let bytes_io = io.call_method1("BytesIO", (PyBytes::new(py, &buffer),))?;

    let polars = py.import("polars")?;
    polars
        .call_method1("read_ipc", (bytes_io,))
        .map(|obj| obj.into())
}

/// Convert a Python polars DataFrame to a Rust Polars DataFrame via IPC serialization
fn py_to_polars(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let io = py.import("io")?;
    let bytes_io = io.call_method0("BytesIO")?;
    df.call_method1("write_ipc", (&bytes_io,))?;
    bytes_io.call_method1("seek", (0i64,))?;

    let ipc_bytes: Vec<u8> = bytes_io.call_method0("read")?.extract()?;
    let cursor = Cursor::new(ipc_bytes);

    IpcReader::new(cursor).finish().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to read DataFrame: {}", e))
    })
}

/// Convert validation errors/warnings to a Python list of dicts
fn errors_to_pylist(
    py: Python<'_>,
    items: &[(String, Option<(usize, usize)>)],
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for (message, location) in items {
        let dict = PyDict::new(py);
        dict.set_item("message", message)?;
        if let Some((line, column)) = location {
            let loc_dict = PyDict::new(py);
            loc_dict.set_item("line", line)?;
            loc_dict.set_item("column", column)?;
            dict.set_item("location", loc_dict)?;
        } else {
            dict.set_item("location", py.None())?;
        }
        list.append(dict)?;
    }
    Ok(list.into())
}

/// Convert ValidationWarning slice to Python list format
fn warnings_to_pylist(py: Python<'_>, warnings: &[ValidationWarning]) -> PyResult<Py<PyList>> {
    let items: Vec<_> = warnings
        .iter()
        .map(|w| {
            (
                w.message.clone(),
                w.location.as_ref().map(|l| (l.line, l.column)),
            )
        })
        .collect();
    errors_to_pylist(py, &items)
}

// ============================================================================
// PyDuckDBReader
// ============================================================================

/// DuckDB database reader for executing SQL queries and ggsql visualizations.
///
/// Creates an in-memory or file-based DuckDB connection that can execute
/// SQL queries and register DataFrames as queryable tables.
#[pyclass(name = "DuckDBReader", unsendable)]
struct PyDuckDBReader {
    inner: RustDuckDBReader,
    connection: String,
}

#[pymethods]
impl PyDuckDBReader {
    /// Create a new DuckDB reader from a connection string.
    #[new]
    fn new(connection: &str) -> PyResult<Self> {
        let inner =
            RustDuckDBReader::from_connection_string(connection).map_err(ggsql_error_to_pyerr)?;
        Ok(Self {
            inner,
            connection: connection.to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!("<DuckDBReader connection={:?}>", self.connection)
    }

    /// Execute a ggsql query with optional DataFrame registration.
    ///
    /// DataFrames are registered before query execution and automatically
    /// unregistered afterward (even on error) to avoid polluting the namespace.
    #[pyo3(signature = (query, data=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        query: &str,
        data: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyPrepared> {
        // First, validate that the query has a VISUALISE clause
        let validated = rust_validate(query).map_err(ggsql_error_to_pyerr)?;

        if !validated.has_visual() {
            return Err(NoVisualiseError::new_err(
                "Query has no VISUALISE clause. Use execute_sql() for plain SQL queries.",
            ));
        }

        // Collect table names to register
        let mut table_names: Vec<String> = Vec::new();

        // Register DataFrames
        if let Some(data_dict) = data {
            for (key, value) in data_dict.iter() {
                let name: String = key.extract()?;
                let rust_df = py_to_polars(py, &value)?;
                self.inner
                    .register(&name, rust_df)
                    .map_err(ggsql_error_to_pyerr)?;
                table_names.push(name);
            }
        }

        // Execute the query, ensuring cleanup happens even on error
        let result = rust_prepare(query, &self.inner);

        // Always unregister tables (cleanup in finally-style)
        for name in &table_names {
            self.inner.unregister(name);
        }

        // Return the result (or propagate the error)
        result
            .map(|p| PyPrepared { inner: p })
            .map_err(ggsql_error_to_pyerr)
    }

    /// Execute a SQL query and return the result as a DataFrame.
    ///
    /// This is for plain SQL queries without visualization. For ggsql queries
    /// with VISUALISE clauses, use execute() instead.
    #[pyo3(name = "execute_sql")]
    fn execute_sql(&self, py: Python<'_>, sql: &str) -> PyResult<Py<PyAny>> {
        let df = self.inner.execute_sql(sql).map_err(ggsql_error_to_pyerr)?;
        polars_to_py(py, &df)
    }

    /// Register a DataFrame as a queryable table.
    ///
    /// After registration, the DataFrame can be queried by name in SQL.
    /// Note: When using execute(), DataFrames are automatically registered
    /// and unregistered, so manual registration is usually unnecessary.
    fn register(&mut self, py: Python<'_>, name: &str, df: &Bound<'_, PyAny>) -> PyResult<()> {
        let rust_df = py_to_polars(py, df)?;
        self.inner
            .register(name, rust_df)
            .map_err(ggsql_error_to_pyerr)
    }

    /// Unregister a table by name.
    ///
    /// Fails silently if the table doesn't exist.
    fn unregister(&mut self, name: &str) {
        self.inner.unregister(name);
    }
}

// ============================================================================
// PyVegaLiteWriter
// ============================================================================

/// Vega-Lite JSON output writer (internal).
///
/// Converts prepared visualization specifications to Vega-Lite v6 JSON.
/// Use the Python VegaLiteWriter class which wraps this and adds render_chart().
#[pyclass(name = "_VegaLiteWriter")]
struct PyVegaLiteWriter {
    inner: RustVegaLiteWriter,
}

#[pymethods]
impl PyVegaLiteWriter {
    /// Create a new Vega-Lite writer.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustVegaLiteWriter::new(),
        }
    }

    fn __repr__(&self) -> &'static str {
        "<VegaLiteWriter>"
    }

    /// Render a prepared visualization to Vega-Lite JSON.
    fn render(&self, spec: &PyPrepared) -> PyResult<String> {
        self.inner.render(&spec.inner).map_err(ggsql_error_to_pyerr)
    }
}

// ============================================================================
// PyValidated
// ============================================================================

/// Result of validate() - query inspection and validation without SQL execution.
///
/// Contains information about query structure and any validation errors/warnings.
/// The tree() method from Rust is not exposed as it's not useful in Python.
#[pyclass(name = "Validated")]
struct PyValidated {
    sql: String,
    visual: String,
    has_visual: bool,
    valid: bool,
    errors: Vec<(String, Option<(usize, usize)>)>,
    warnings: Vec<(String, Option<(usize, usize)>)>,
}

#[pymethods]
impl PyValidated {
    fn __repr__(&self) -> String {
        format!(
            "<Validated valid={} has_visual={} errors={}>",
            self.valid,
            self.has_visual,
            self.errors.len()
        )
    }

    /// Whether the query contains a VISUALISE clause.
    fn has_visual(&self) -> bool {
        self.has_visual
    }

    /// The SQL portion (before VISUALISE).
    fn sql(&self) -> &str {
        &self.sql
    }

    /// The VISUALISE portion (raw text).
    fn visual(&self) -> &str {
        &self.visual
    }

    /// Whether the query is valid (no errors).
    fn valid(&self) -> bool {
        self.valid
    }

    /// Validation errors (fatal issues).
    fn errors(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        errors_to_pylist(py, &self.errors)
    }

    /// Validation warnings (non-fatal issues).
    fn warnings(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        errors_to_pylist(py, &self.warnings)
    }
}

// ============================================================================
// PyPrepared
// ============================================================================

/// Result of reader.execute(), ready for rendering.
///
/// Contains the resolved plot specification, data, and metadata.
/// Use writer.render(spec) or writer.render_chart(spec) to generate output.
#[pyclass(name = "Prepared")]
struct PyPrepared {
    inner: Prepared,
}

#[pymethods]
impl PyPrepared {
    fn __repr__(&self) -> String {
        let m = self.inner.metadata();
        format!(
            "<Prepared rows={} columns={} layers={}>",
            m.rows,
            m.columns.len(),
            m.layer_count
        )
    }

    /// Get visualization metadata.
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let m = self.inner.metadata();
        let dict = PyDict::new(py);
        dict.set_item("rows", m.rows)?;
        dict.set_item("columns", m.columns.clone())?;
        dict.set_item("layer_count", m.layer_count)?;
        Ok(dict.into())
    }

    /// The main SQL query that was executed.
    fn sql(&self) -> &str {
        self.inner.sql()
    }

    /// The VISUALISE portion (raw text).
    fn visual(&self) -> &str {
        self.inner.visual()
    }

    /// Number of layers.
    fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    /// Get global data (main query result).
    fn data(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner.data().map(|df| polars_to_py(py, df)).transpose()
    }

    /// Get layer-specific data (from FILTER or FROM clause).
    fn layer_data(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .layer_data(index)
            .map(|df| polars_to_py(py, df))
            .transpose()
    }

    /// Get stat transform data (e.g., histogram bins, density estimates).
    fn stat_data(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .stat_data(index)
            .map(|df| polars_to_py(py, df))
            .transpose()
    }

    /// Layer filter/source query, or None if using global data.
    fn layer_sql(&self, index: usize) -> Option<String> {
        self.inner.layer_sql(index).map(|s| s.to_string())
    }

    /// Stat transform query, or None if no stat transform.
    fn stat_sql(&self, index: usize) -> Option<String> {
        self.inner.stat_sql(index).map(|s| s.to_string())
    }

    /// Validation warnings from preparation.
    fn warnings(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        warnings_to_pylist(py, self.inner.warnings())
    }
}

// ============================================================================
// Module Functions
// ============================================================================

/// Validate query syntax and semantics without executing SQL.
#[pyfunction]
fn validate(query: &str) -> PyResult<PyValidated> {
    let v = rust_validate(query).map_err(ggsql_error_to_pyerr)?;

    Ok(PyValidated {
        sql: v.sql().to_string(),
        visual: v.visual().to_string(),
        has_visual: v.has_visual(),
        valid: v.valid(),
        errors: v
            .errors()
            .iter()
            .map(|e| {
                (
                    e.message.clone(),
                    e.location.as_ref().map(|l| (l.line, l.column)),
                )
            })
            .collect(),
        warnings: v
            .warnings()
            .iter()
            .map(|w| {
                (
                    w.message.clone(),
                    w.location.as_ref().map(|l| (l.line, l.column)),
                )
            })
            .collect(),
    })
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _ggsql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Exception classes (exported without the Py prefix)
    m.add("GgsqlError", m.py().get_type::<PyGgsqlError>())?;
    m.add("ParseError", m.py().get_type::<PyParseError>())?;
    m.add("ValidationError", m.py().get_type::<PyValidationError>())?;
    m.add("ReaderError", m.py().get_type::<PyReaderError>())?;
    m.add("WriterError", m.py().get_type::<PyWriterError>())?;
    m.add("NoVisualiseError", m.py().get_type::<NoVisualiseError>())?;

    // Classes
    m.add_class::<PyDuckDBReader>()?;
    m.add_class::<PyVegaLiteWriter>()?;
    m.add_class::<PyValidated>()?;
    m.add_class::<PyPrepared>()?;

    // Functions
    m.add_function(wrap_pyfunction!(validate, m)?)?;

    Ok(())
}
