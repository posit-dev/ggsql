"""Type stubs for the ggsql native extension module."""

from typing import Any

import polars as pl

# ============================================================================
# Exception Types
# ============================================================================


class GgsqlError(Exception):
    """Base exception for all ggsql errors."""

    ...


class ParseError(GgsqlError):
    """Raised when query parsing fails."""

    ...


class ValidationError(GgsqlError):
    """Raised when query validation fails (semantic errors)."""

    ...


class ReaderError(GgsqlError):
    """Raised when database/data source operations fail."""

    ...


class WriterError(GgsqlError):
    """Raised when output generation fails."""

    ...


class NoVisualiseError(GgsqlError):
    """Raised when execute() is called on a query without VISUALISE clause."""

    ...


# ============================================================================
# Classes
# ============================================================================


class DuckDBReader:
    """DuckDB database reader for executing SQL queries and ggsql visualizations."""

    def __init__(self, connection: str) -> None:
        """Create a new DuckDB reader from a connection string.

        Parameters
        ----------
        connection
            Connection string. Use "duckdb://memory" for in-memory database
            or "duckdb://path/to/file.db" for file-based database.
        """
        ...

    def __repr__(self) -> str: ...

    def execute(
        self, query: str, data: dict[str, pl.DataFrame] | None = None
    ) -> Prepared:
        """Execute a ggsql query with optional DataFrame registration.

        DataFrames are registered before query execution and automatically
        unregistered afterward (even on error) to avoid polluting the namespace.

        Parameters
        ----------
        query
            The ggsql query to execute. Must contain a VISUALISE clause.
        data
            DataFrames to register as queryable tables. Keys are table names.

        Returns
        -------
        Prepared
            A prepared visualization ready for rendering.

        Raises
        ------
        NoVisualiseError
            If the query has no VISUALISE clause.
        ParseError
            If query parsing fails.
        ValidationError
            If query validation fails.
        ReaderError
            If SQL execution fails.
        """
        ...

    def execute_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query and return the result as a DataFrame.

        This is for plain SQL queries without visualization. For ggsql queries
        with VISUALISE clauses, use execute() instead.

        Parameters
        ----------
        sql
            The SQL query to execute.

        Returns
        -------
        polars.DataFrame
            The query result as a polars DataFrame.
        """
        ...

    def register(self, name: str, df: pl.DataFrame) -> None:
        """Register a DataFrame as a queryable table.

        After registration, the DataFrame can be queried by name in SQL.
        Note: When using execute(), DataFrames are automatically registered
        and unregistered, so manual registration is usually unnecessary.

        Parameters
        ----------
        name
            The table name to register under.
        df
            The DataFrame to register.
        """
        ...

    def unregister(self, name: str) -> None:
        """Unregister a table by name.

        Fails silently if the table doesn't exist.

        Parameters
        ----------
        name
            The table name to unregister.
        """
        ...


class _VegaLiteWriter:
    """Vega-Lite JSON output writer (internal).

    Use the Python VegaLiteWriter class which wraps this and adds render_chart().
    """

    def __init__(self) -> None:
        """Create a new Vega-Lite writer."""
        ...

    def __repr__(self) -> str: ...

    def render(self, spec: Prepared) -> str:
        """Render a prepared visualization to Vega-Lite JSON.

        Parameters
        ----------
        spec
            The prepared visualization (from reader.execute()).

        Returns
        -------
        str
            The Vega-Lite JSON specification as a string.
        """
        ...


class Validated:
    """Result of validate() - query inspection and validation without SQL execution."""

    def __repr__(self) -> str: ...

    def has_visual(self) -> bool:
        """Whether the query contains a VISUALISE clause."""
        ...

    def sql(self) -> str:
        """The SQL portion (before VISUALISE)."""
        ...

    def visual(self) -> str:
        """The VISUALISE portion (raw text)."""
        ...

    def valid(self) -> bool:
        """Whether the query is valid (no errors)."""
        ...

    def errors(self) -> list[dict[str, Any]]:
        """Validation errors (fatal issues).

        Returns
        -------
        list[dict]
            List of error dictionaries with 'message' and optional 'location' keys.
        """
        ...

    def warnings(self) -> list[dict[str, Any]]:
        """Validation warnings (non-fatal issues).

        Returns
        -------
        list[dict]
            List of warning dictionaries with 'message' and optional 'location' keys.
        """
        ...


class Prepared:
    """Result of reader.execute(), ready for rendering."""

    def __repr__(self) -> str: ...

    def metadata(self) -> dict[str, Any]:
        """Get visualization metadata.

        Returns
        -------
        dict
            Dictionary with 'rows', 'columns', and 'layer_count' keys.
        """
        ...

    def sql(self) -> str:
        """The main SQL query that was executed."""
        ...

    def visual(self) -> str:
        """The VISUALISE portion (raw text)."""
        ...

    def layer_count(self) -> int:
        """Number of layers."""
        ...

    def data(self) -> pl.DataFrame | None:
        """Get global data (main query result)."""
        ...

    def layer_data(self, index: int) -> pl.DataFrame | None:
        """Get layer-specific data (from FILTER or FROM clause).

        Parameters
        ----------
        index
            The layer index (0-based).
        """
        ...

    def stat_data(self, index: int) -> pl.DataFrame | None:
        """Get stat transform data (e.g., histogram bins, density estimates).

        Parameters
        ----------
        index
            The layer index (0-based).
        """
        ...

    def layer_sql(self, index: int) -> str | None:
        """Layer filter/source query, or None if using global data.

        Parameters
        ----------
        index
            The layer index (0-based).
        """
        ...

    def stat_sql(self, index: int) -> str | None:
        """Stat transform query, or None if no stat transform.

        Parameters
        ----------
        index
            The layer index (0-based).
        """
        ...

    def warnings(self) -> list[dict[str, Any]]:
        """Validation warnings from preparation."""
        ...


# ============================================================================
# Functions
# ============================================================================


def validate(query: str) -> Validated:
    """Validate query syntax and semantics without executing SQL.

    Parameters
    ----------
    query
        The ggsql query to validate.

    Returns
    -------
    Validated
        Validation result with query inspection methods.
    """
    ...
