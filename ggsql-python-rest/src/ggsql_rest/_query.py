"""Query execution with hybrid local/remote support."""

import json
import uuid
from typing import Any

import polars as pl
from sqlalchemy import Engine, text

from ggsql import validate, VegaLiteWriter

from ._sessions import Session


def execute_ggsql(
    query: str,
    session: Session,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """
    Execute a ggsql query with hybrid approach.

    If engine is provided, SQL portion runs on remote database,
    result is registered in session's DuckDB, and VISUALISE
    portion runs locally.
    """
    validated = validate(query)

    if not validated.has_visual():
        raise ValueError("Query must contain VISUALISE clause")

    sql_portion = validated.sql()

    if engine is not None and sql_portion.strip():
        # Execute SQL on remote database
        df = execute_remote(engine, sql_portion)

        # Register result in session's DuckDB
        table_name = f"__remote_result_{uuid.uuid4().hex[:8]}__"
        session.duckdb.register(table_name, df)

        # Rewrite query to use local table
        local_query = f"SELECT * FROM {table_name} {validated.visual()}"
    else:
        # All local
        local_query = query

    # Execute full ggsql in session's DuckDB
    spec = session.duckdb.execute(local_query)

    writer = VegaLiteWriter()
    vegalite_json = writer.render(spec)

    return {
        "spec": json.loads(vegalite_json),
        "metadata": {
            "rows": spec.metadata()["rows"],
            "columns": spec.metadata()["columns"],
            "layers": spec.metadata()["layer_count"],
        },
    }


def execute_remote(engine: Engine, sql: str) -> pl.DataFrame:
    """Execute SQL on remote database, return as Polars DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = result.fetchall()

        # Convert to dict of lists for Polars
        data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        return pl.DataFrame(data)


def execute_sql(
    query: str,
    session: Session,
    engine: Engine | None = None,
    max_rows: int = 10000,
) -> dict[str, Any]:
    """Execute pure SQL query and return results as JSON."""
    if engine is not None:
        df = execute_remote(engine, query)
    else:
        df = session.duckdb.execute_sql(query)

    row_count = len(df)
    truncated = row_count > max_rows

    if truncated:
        df = df.head(max_rows)

    return {
        "rows": df.to_dicts(),
        "columns": df.columns,
        "row_count": row_count,
        "truncated": truncated,
    }
