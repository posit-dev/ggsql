"""Tests for query execution."""

import pytest
from ggsql_rest._sessions import Session
from ggsql_rest._query import execute_ggsql


def test_execute_ggsql_local():
    """Test executing a ggsql query against local DuckDB."""
    session = Session("test", timeout_mins=30)

    # Create test data in session's DuckDB
    session.duckdb.execute_sql(
        "CREATE TABLE test AS SELECT 1 as x, 2 as y UNION SELECT 3, 4"
    )

    result = execute_ggsql(
        "SELECT * FROM test VISUALISE x, y DRAW point",
        session,
        engine=None,
    )

    assert "spec" in result
    assert "metadata" in result
    assert result["metadata"]["rows"] == 2
    assert "x" in result["metadata"]["columns"]
    assert "y" in result["metadata"]["columns"]


def test_execute_ggsql_no_visualise():
    """Test that queries without VISUALISE raise an error."""
    session = Session("test", timeout_mins=30)

    with pytest.raises(ValueError, match="VISUALISE"):
        execute_ggsql("SELECT 1 as x", session, engine=None)
