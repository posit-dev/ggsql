// Allow useless_conversion due to false positive from pyo3 macro expansion
// See: https://github.com/PyO3/pyo3/issues/4327
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::io::Cursor;

use ggsql::reader::Spec;
use ggsql::reader::{DuckDBReader as RustDuckDBReader, Reader};
use ggsql::validate::{validate as rust_validate, ValidationWarning};
use ggsql::writer::{VegaLiteWriter as RustVegaLiteWriter, Writer as RustWriter};
use ggsql::GgsqlError;

use polars::prelude::{DataFrame, IpcReader, IpcWriter, SerReader, SerWriter};

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

/// Convert a Python polars DataFrame to Rust DataFrame - for use inside Python::attach
/// This variant is used by PyReaderBridge where we already hold the GIL.
fn py_to_polars_inner(df: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let py = df.py();
    let io = py.import("io")?;
    let bytes_io = io.call_method0("BytesIO")?;

    df.call_method1("write_ipc", (&bytes_io,)).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Reader.execute_sql() must return a polars.DataFrame",
        )
    })?;

    bytes_io.call_method1("seek", (0i64,))?;
    let ipc_bytes: Vec<u8> = bytes_io.call_method0("read")?.extract()?;
    let cursor = Cursor::new(ipc_bytes);

    IpcReader::new(cursor).finish().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to deserialize DataFrame: {}",
            e
        ))
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
// PyReaderBridge - Bridges Python reader objects to Rust Reader trait
// ============================================================================

/// Bridges a Python reader object to the Rust Reader trait.
///
/// This allows any Python object with an `execute_sql(sql: str) -> polars.DataFrame`
/// method to be used as a ggsql reader.
struct PyReaderBridge {
    obj: Py<PyAny>,
}

impl Reader for PyReaderBridge {
    fn execute_sql(&self, sql: &str) -> ggsql::Result<DataFrame> {
        Python::attach(|py| {
            let bound = self.obj.bind(py);
            let result = bound.call_method1("execute_sql", (sql,)).map_err(|e| {
                GgsqlError::ReaderError(format!("Reader.execute_sql() failed: {}", e))
            })?;
            py_to_polars_inner(&result).map_err(|e| GgsqlError::ReaderError(e.to_string()))
        })
    }

    fn supports_register(&self) -> bool {
        Python::attach(|py| {
            self.obj
                .bind(py)
                .call_method0("supports_register")
                .and_then(|r| r.extract::<bool>())
                .unwrap_or(false)
        })
    }

    fn register(&mut self, name: &str, df: DataFrame) -> ggsql::Result<()> {
        Python::attach(|py| {
            let py_df =
                polars_to_py(py, &df).map_err(|e| GgsqlError::ReaderError(e.to_string()))?;
            self.obj
                .bind(py)
                .call_method1("register", (name, py_df))
                .map_err(|e| GgsqlError::ReaderError(format!("Reader.register() failed: {}", e)))?;
            Ok(())
        })
    }

    fn unregister(&mut self, name: &str) -> ggsql::Result<()> {
        Python::attach(|py| {
            self.obj
                .bind(py)
                .call_method1("unregister", (name,))
                .map_err(|e| {
                    GgsqlError::ReaderError(format!("Reader.unregister() failed: {}", e))
                })?;
            Ok(())
        })
    }
}

// ============================================================================
// Native Reader Detection Macro
// ============================================================================

/// Macro to try native readers and fall back to bridge.
/// Adding new native readers = add to the macro invocation list.
macro_rules! try_native_readers {
    ($query:expr, $reader:expr, $($native_type:ty),*) => {{
        $(
            if let Ok(native) = $reader.downcast::<$native_type>() {
                return native.borrow().inner.execute($query)
                    .map(|s| PySpec { inner: s })
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
            }
        )*
    }};
}

// ============================================================================
// PyDuckDBReader
// ============================================================================

/// DuckDB database reader for executing SQL queries.
///
/// Creates an in-memory or file-based DuckDB connection that can execute
/// SQL queries and register DataFrames as queryable tables.
///
/// Examples
/// --------
/// >>> reader = DuckDBReader("duckdb://memory")
/// >>> df = reader.execute_sql("SELECT 1 as x, 2 as y")
///
/// >>> reader = DuckDBReader("duckdb://memory")
/// >>> reader.register("data", pl.DataFrame({"x": [1, 2, 3]}))
/// >>> df = reader.execute_sql("SELECT * FROM data WHERE x > 1")
#[pyclass(name = "DuckDBReader", unsendable)]
struct PyDuckDBReader {
    inner: RustDuckDBReader,
}

#[pymethods]
impl PyDuckDBReader {
    /// Create a new DuckDB reader from a connection string.
    ///
    /// Parameters
    /// ----------
    /// connection : str
    ///     Connection string. Use "duckdb://memory" for in-memory database
    ///     or "duckdb://path/to/file.db" for file-based database.
    ///
    /// Returns
    /// -------
    /// DuckDBReader
    ///     A configured DuckDB reader instance.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the connection string is invalid or the database cannot be opened.
    #[new]
    fn new(connection: &str) -> PyResult<Self> {
        let inner = RustDuckDBReader::from_connection_string(connection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Register a DataFrame as a queryable table.
    ///
    /// After registration, the DataFrame can be queried by name in SQL.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The table name to register under.
    /// df : polars.DataFrame
    ///     The DataFrame to register. Must be a polars DataFrame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If registration fails or the table name is invalid.
    fn register(&mut self, py: Python<'_>, name: &str, df: &Bound<'_, PyAny>) -> PyResult<()> {
        let rust_df = py_to_polars(py, df)?;
        self.inner
            .register(name, rust_df)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Unregister a previously registered table.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The table name to unregister.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the table wasn't registered via this reader or unregistration fails.
    fn unregister(&mut self, name: &str) -> PyResult<()> {
        self.inner
            .unregister(name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Execute a SQL query and return the result as a DataFrame.
    ///
    /// Parameters
    /// ----------
    /// sql : str
    ///     The SQL query to execute.
    ///
    /// Returns
    /// -------
    /// polars.DataFrame
    ///     The query result as a polars DataFrame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the SQL is invalid or execution fails.
    fn execute_sql(&self, py: Python<'_>, sql: &str) -> PyResult<Py<PyAny>> {
        let df = self
            .inner
            .execute_sql(sql)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        polars_to_py(py, &df)
    }

    /// Check if this reader supports DataFrame registration.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if register() is supported, False otherwise.
    fn supports_register(&self) -> bool {
        self.inner.supports_register()
    }

    /// Execute a ggsql query and return the visualization specification.
    ///
    /// This is the main entry point for creating visualizations. It parses
    /// the query, executes the SQL portion, and returns a PySpec ready
    /// for rendering.
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     The ggsql query (SQL + VISUALISE clause).
    ///
    /// Returns
    /// -------
    /// Spec
    ///     The resolved visualization specification ready for rendering.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the query syntax is invalid, has no VISUALISE clause, or SQL execution fails.
    ///
    /// Examples
    /// --------
    /// >>> reader = DuckDBReader("duckdb://memory")
    /// >>> spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
    /// >>> writer = VegaLiteWriter()
    /// >>> json_output = writer.render(spec)
    fn execute(&self, query: &str) -> PyResult<PySpec> {
        self.inner
            .execute(query)
            .map(|s| PySpec { inner: s })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

// ============================================================================
// PyVegaLiteWriter
// ============================================================================

/// Vega-Lite JSON output writer.
///
/// Converts visualization specifications to Vega-Lite v6 JSON.
///
/// Examples
/// --------
/// >>> writer = VegaLiteWriter()
/// >>> spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
/// >>> json_output = writer.render(spec)
#[pyclass(name = "VegaLiteWriter")]
struct PyVegaLiteWriter {
    inner: RustVegaLiteWriter,
}

#[pymethods]
impl PyVegaLiteWriter {
    /// Create a new Vega-Lite writer.
    ///
    /// Returns
    /// -------
    /// VegaLiteWriter
    ///     A configured Vega-Lite writer instance.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustVegaLiteWriter::new(),
        }
    }

    /// Render a Spec to Vega-Lite JSON output
    ///
    /// Parameters
    /// ----------
    /// spec : Spec
    ///     The visualization specification from reader.execute().
    ///
    /// Returns
    /// -------
    /// str
    ///     The output (i.e., Vega-Lite JSON string).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If rendering fails.
    ///
    /// Examples
    /// --------
    /// >>> reader = DuckDBReader("duckdb://memory")
    /// >>> spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
    /// >>> writer = VegaLiteWriter()
    /// >>> json_output = writer.render(spec)
    fn render(&self, spec: &PySpec) -> PyResult<String> {
        self.inner
            .render(&spec.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
    /// Whether the query contains a VISUALISE clause.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the query has a VISUALISE clause.
    fn has_visual(&self) -> bool {
        self.has_visual
    }

    /// The SQL portion (before VISUALISE).
    ///
    /// Returns
    /// -------
    /// str
    ///     The SQL part of the query.
    fn sql(&self) -> &str {
        &self.sql
    }

    /// The VISUALISE portion (raw text).
    ///
    /// Returns
    /// -------
    /// str
    ///     The VISUALISE part of the query.
    fn visual(&self) -> &str {
        &self.visual
    }

    /// Whether the query is valid (no errors).
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the query is syntactically and semantically valid.
    fn valid(&self) -> bool {
        self.valid
    }

    /// Validation errors (fatal issues).
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     List of error dictionaries with 'message' and optional 'location' keys.
    fn errors(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        errors_to_pylist(py, &self.errors)
    }

    /// Validation warnings (non-fatal issues).
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     List of warning dictionaries with 'message' and optional 'location' keys.
    fn warnings(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        errors_to_pylist(py, &self.warnings)
    }
}

// ============================================================================
// PySpec
// ============================================================================

/// Result of reader.execute(), ready for rendering.
///
/// Contains the resolved plot specification, data, and metadata.
/// Use writer.render(spec) to generate output.
///
/// Examples
/// --------
/// >>> spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
/// >>> print(f"Rows: {spec.metadata()['rows']}")
/// >>> writer = VegaLiteWriter()
/// >>> json_output = writer.render(spec)
#[pyclass(name = "Spec")]
struct PySpec {
    inner: Spec,
}

#[pymethods]
impl PySpec {
    /// Get visualization metadata.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary with 'rows', 'columns', and 'layer_count' keys.
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let m = self.inner.metadata();
        let dict = PyDict::new(py);
        dict.set_item("rows", m.rows)?;
        dict.set_item("columns", m.columns.clone())?;
        dict.set_item("layer_count", m.layer_count)?;
        Ok(dict.into())
    }

    /// The main SQL query that was executed.
    ///
    /// Returns
    /// -------
    /// str
    ///     The SQL query string.
    fn sql(&self) -> &str {
        self.inner.sql()
    }

    /// The VISUALISE portion (raw text).
    ///
    /// Returns
    /// -------
    /// str
    ///     The VISUALISE clause text.
    fn visual(&self) -> &str {
        self.inner.visual()
    }

    /// Number of layers.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of DRAW clauses in the visualization.
    fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    /// Get global data (main query result).
    ///
    /// Returns
    /// -------
    /// polars.DataFrame | None
    ///     The main query result DataFrame, or None if not available.
    fn data(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .layer_data(0)
            .map(|df| polars_to_py(py, df))
            .transpose()
    }

    /// Get layer-specific data (from FILTER or FROM clause).
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     The layer index (0-based).
    ///
    /// Returns
    /// -------
    /// polars.DataFrame | None
    ///     The layer-specific DataFrame, or None if the layer uses global data.
    fn layer_data(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .layer_data(index)
            .map(|df| polars_to_py(py, df))
            .transpose()
    }

    /// Get stat transform data (e.g., histogram bins, density estimates).
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     The layer index (0-based).
    ///
    /// Returns
    /// -------
    /// polars.DataFrame | None
    ///     The stat transform DataFrame, or None if no stat transform.
    fn stat_data(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .stat_data(index)
            .map(|df| polars_to_py(py, df))
            .transpose()
    }

    /// Layer filter/source query, or None if using global data.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     The layer index (0-based).
    ///
    /// Returns
    /// -------
    /// str | None
    ///     The filter SQL query, or None if the layer uses global data directly.
    fn layer_sql(&self, index: usize) -> Option<String> {
        self.inner.layer_sql(index).map(|s| s.to_string())
    }

    /// Stat transform query, or None if no stat transform.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     The layer index (0-based).
    ///
    /// Returns
    /// -------
    /// str | None
    ///     The stat transform SQL query, or None if no stat transform.
    fn stat_sql(&self, index: usize) -> Option<String> {
        self.inner.stat_sql(index).map(|s| s.to_string())
    }

    /// Validation warnings from preparation.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     List of warning dictionaries with 'message' and optional 'location' keys.
    fn warnings(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        warnings_to_pylist(py, self.inner.warnings())
    }
}

// ============================================================================
// Module Functions
// ============================================================================

/// Validate query syntax and semantics without executing SQL.
///
/// Parameters
/// ----------
/// query : str
///     The ggsql query to validate.
///
/// Returns
/// -------
/// Validated
///     Validation result with query inspection methods.
///
/// Raises
/// ------
/// ValueError
///     If validation fails unexpectedly (not for syntax errors, which are captured).
#[pyfunction]
fn validate(query: &str) -> PyResult<PyValidated> {
    let v = rust_validate(query)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

/// Execute a ggsql query using a custom Python reader.
///
/// This is a convenience function for custom readers. For native readers,
/// prefer using `reader.execute()` directly.
///
/// Parameters
/// ----------
/// query : str
///     The ggsql query to execute.
/// reader : Reader | object
///     The database reader to execute SQL against. Can be a native Reader
///     for optimal performance, or any Python object with an
///     `execute_sql(sql: str) -> polars.DataFrame` method.
///
/// Returns
/// -------
/// Spec
///     The resolved visualization specification ready for rendering.
///
/// Raises
/// ------
/// ValueError
///     If parsing, validation, or SQL execution fails.
///
/// Examples
/// --------
/// >>> # Using native reader (prefer reader.execute() instead)
/// >>> reader = DuckDBReader("duckdb://memory")
/// >>> spec = execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point", reader)
/// >>> writer = VegaLiteWriter()
/// >>> json_output = writer.render(spec)
///
/// >>> # Using custom Python reader
/// >>> class MyReader:
/// ...     def execute_sql(self, sql: str) -> pl.DataFrame:
/// ...         return pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
/// >>> reader = MyReader()
/// >>> spec = execute("SELECT * FROM data VISUALISE x, y DRAW point", reader)
#[pyfunction]
fn execute(query: &str, reader: &Bound<'_, PyAny>) -> PyResult<PySpec> {
    // Fast path: try all known native reader types
    // Add new native readers to this list as they're implemented
    try_native_readers!(query, reader, PyDuckDBReader);

    // Bridge path: wrap Python object as Reader
    let bridge = PyReaderBridge {
        obj: reader.clone().unbind(),
    };
    bridge
        .execute(query)
        .map(|s| PySpec { inner: s })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _ggsql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyDuckDBReader>()?;
    m.add_class::<PyVegaLiteWriter>()?;
    m.add_class::<PyValidated>()?;
    m.add_class::<PySpec>()?;

    // Functions
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;

    Ok(())
}
