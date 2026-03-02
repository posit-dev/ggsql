"""Type stubs for the ggsql native module (_ggsql)."""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Exceptions (all subclass ValueError for backwards compatibility)
# ---------------------------------------------------------------------------

class ParseError(ValueError): ...
class ValidationError(ValueError): ...
class ReaderError(ValueError): ...
class WriterError(ValueError): ...

# ---------------------------------------------------------------------------
# DuckDBReader
# ---------------------------------------------------------------------------

class DuckDBReader:
    """DuckDB database reader for executing SQL queries.

    Creates an in-memory or file-based DuckDB connection that can execute
    SQL queries and register DataFrames as queryable tables.

    Parameters
    ----------
    connection
        DuckDB connection string. Use ``"duckdb://memory"`` for in-memory
        database or ``"duckdb://path/to/file.db"`` for file-based database.

    Raises
    ------
    ValueError
        If the connection string is invalid or the database cannot be opened.
    """

    def __init__(self, connection: str) -> None: ...
    def execute_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query and return results as a polars DataFrame.

        Parameters
        ----------
        sql
            The SQL query to execute.

        Returns
        -------
        polars.DataFrame
            The query result as a polars DataFrame.

        Raises
        ------
        ValueError
            If the SQL is invalid or execution fails.
        """
        ...

    def register(
        self, name: str, df: pl.DataFrame, replace: bool = False
    ) -> None:
        """Register a polars DataFrame as a named table.

        After registration the DataFrame can be queried by name in SQL.

        Parameters
        ----------
        name
            The table name to register under.
        df
            The DataFrame to register. Must be a polars DataFrame.
        replace
            Whether to replace an existing table with the same name.

        Raises
        ------
        ValueError
            If registration fails or the table name is invalid.
        """
        ...

    def unregister(self, name: str) -> None:
        """Unregister a previously registered table.

        Parameters
        ----------
        name
            The table name to unregister.

        Raises
        ------
        ValueError
            If the table was not registered or unregistration fails.
        """
        ...

    def execute(
        self,
        query: str,
        *,
        data: dict[str, pl.DataFrame] | None = None,
    ) -> Spec:
        """Execute a ggsql query and return the visualization specification.

        This is the main entry point for creating visualizations. It parses
        the query, executes the SQL portion, and returns a ``Spec`` ready
        for rendering.

        Parameters
        ----------
        query
            The ggsql query (SQL + VISUALISE clause).
        data
            Optional dictionary mapping table names to DataFrames. Tables are
            registered before execution and unregistered afterward (even on
            error).

        Returns
        -------
        Spec
            The resolved visualization specification ready for rendering.

        Raises
        ------
        ValueError
            If the query syntax is invalid, has no VISUALISE clause, or
            SQL execution fails.
        """
        ...

# ---------------------------------------------------------------------------
# VegaLiteWriter
# ---------------------------------------------------------------------------

class VegaLiteWriter:
    """Vega-Lite v6 JSON output writer.

    Converts visualization specifications to Vega-Lite v6 JSON.
    """

    def __init__(self) -> None: ...
    def render(self, spec: Spec) -> str:
        """Render a Spec to a Vega-Lite JSON string.

        Parameters
        ----------
        spec
            The visualization specification from ``reader.execute()``.

        Returns
        -------
        str
            The Vega-Lite JSON string.

        Raises
        ------
        ValueError
            If rendering fails.
        """
        ...

# ---------------------------------------------------------------------------
# Validated
# ---------------------------------------------------------------------------

class Validated:
    """Result of ``validate()`` — query inspection without SQL execution.

    Contains information about query structure and any validation
    errors/warnings.
    """

    def has_visual(self) -> bool:
        """Whether the query contains a VISUALISE clause.

        Returns
        -------
        bool
            ``True`` if the query has a VISUALISE clause.
        """
        ...

    def sql(self) -> str:
        """The SQL portion (before VISUALISE).

        Returns
        -------
        str
            The SQL part of the query.
        """
        ...

    def visual(self) -> str:
        """The VISUALISE portion (raw text).

        Returns
        -------
        str
            The VISUALISE part of the query.
        """
        ...

    def valid(self) -> bool:
        """Whether the query is valid (no errors).

        Returns
        -------
        bool
            ``True`` if the query is syntactically and semantically valid.
        """
        ...

    def errors(self) -> list[dict[str, object]]:
        """Validation errors (fatal issues).

        Returns
        -------
        list[dict]
            List of error dictionaries with ``"message"`` (str) and
            ``"location"`` (``{"line": int, "column": int}`` or ``None``)
            keys.
        """
        ...

    def warnings(self) -> list[dict[str, object]]:
        """Validation warnings (non-fatal issues).

        Returns
        -------
        list[dict]
            List of warning dictionaries with ``"message"`` (str) and
            ``"location"`` (``{"line": int, "column": int}`` or ``None``)
            keys.
        """
        ...

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

class Spec:
    """Result of ``reader.execute()`` — resolved visualization spec.

    Contains the resolved plot specification, data, and metadata.
    Use ``writer.render(spec)`` to generate output.
    """

    def metadata(self) -> dict[str, object]:
        """Get visualization metadata.

        Returns
        -------
        dict
            Dictionary with ``"rows"`` (int), ``"columns"`` (list[str]),
            and ``"layer_count"`` (int) keys.
        """
        ...

    def sql(self) -> str:
        """The main SQL query that was executed.

        Returns
        -------
        str
            The SQL query string.
        """
        ...

    def visual(self) -> str:
        """The VISUALISE portion (raw text).

        Returns
        -------
        str
            The VISUALISE clause text.
        """
        ...

    def layer_count(self) -> int:
        """Number of DRAW layers.

        Returns
        -------
        int
            The number of DRAW clauses in the visualization.
        """
        ...

    def data(self) -> pl.DataFrame | None:
        """Main query result DataFrame.

        Returns
        -------
        polars.DataFrame or None
            The main query result DataFrame, or ``None`` if not available.
        """
        ...

    def layer_data(self, index: int) -> pl.DataFrame | None:
        """Layer-specific DataFrame (from FILTER or FROM clause).

        Parameters
        ----------
        index
            The layer index (0-based).

        Returns
        -------
        polars.DataFrame or None
            The layer-specific DataFrame, or ``None`` if the layer uses
            global data.
        """
        ...

    def stat_data(self, index: int) -> pl.DataFrame | None:
        """Statistical transform DataFrame.

        Parameters
        ----------
        index
            The layer index (0-based).

        Returns
        -------
        polars.DataFrame or None
            The stat transform DataFrame, or ``None`` if no stat transform.
        """
        ...

    def layer_sql(self, index: int) -> str | None:
        """Layer filter/source query.

        Parameters
        ----------
        index
            The layer index (0-based).

        Returns
        -------
        str or None
            The filter SQL query, or ``None`` if the layer uses global data.
        """
        ...

    def stat_sql(self, index: int) -> str | None:
        """Stat transform query.

        Parameters
        ----------
        index
            The layer index (0-based).

        Returns
        -------
        str or None
            The stat transform SQL query, or ``None`` if no stat transform.
        """
        ...

    def warnings(self) -> list[dict[str, object]]:
        """Validation warnings from preparation.

        Returns
        -------
        list[dict]
            List of warning dictionaries with ``"message"`` (str) and
            ``"location"`` (``{"line": int, "column": int}`` or ``None``)
            keys.
        """
        ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

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

    Raises
    ------
    ValueError
        If validation fails unexpectedly (syntax errors are captured in
        the returned ``Validated`` object, not raised).
    """
    ...

def execute(
    query: str,
    reader: object,
    *,
    data: dict[str, pl.DataFrame] | None = None,
) -> Spec:
    """Execute a ggsql query with a reader (native or custom Python object).

    This is a convenience function for custom readers. For native readers,
    prefer using ``reader.execute()`` directly.

    Parameters
    ----------
    query
        The ggsql query to execute.
    reader
        The database reader to execute SQL against. Can be a native
        ``DuckDBReader`` for optimal performance, or any Python object with
        an ``execute_sql(sql: str) -> polars.DataFrame`` method.
    data
        Optional dictionary mapping table names to DataFrames. Tables are
        registered before execution and unregistered afterward (even on
        error).

    Returns
    -------
    Spec
        The resolved visualization specification ready for rendering.

    Raises
    ------
    ValueError
        If parsing, validation, or SQL execution fails.
    """
    ...
