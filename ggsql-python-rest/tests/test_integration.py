"""Integration tests with SQLAlchemy backend."""

import io
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from ggsql_rest import create_app, ConnectionRegistry


@pytest.mark.anyio
async def test_remote_query_with_sqlite():
    """Test hybrid execution with SQLite as remote database."""
    # Set up SQLite as "remote" database with StaticPool to persist in-memory data
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sales (x INTEGER, y INTEGER)"))
        conn.execute(text("INSERT INTO sales VALUES (1, 10), (2, 20), (3, 30)"))

    # Set up app with connection registry
    registry = ConnectionRegistry()
    registry.register("test_db", lambda req: engine)

    app = create_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        response = await client.post("/sessions")
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        # Query remote database
        response = await client.post(
            f"/sessions/{session_id}/query",
            json={
                "query": "SELECT * FROM sales VISUALISE x, y DRAW point",
                "connection": "test_db",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "spec" in data
        assert data["metadata"]["rows"] == 3


@pytest.mark.anyio
async def test_sql_endpoint_with_remote():
    """Test pure SQL execution against remote database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')"))

    registry = ConnectionRegistry()
    registry.register("test_db", lambda req: engine)

    app = create_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/sessions")
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        response = await client.post(
            f"/sessions/{session_id}/sql",
            json={
                "query": "SELECT * FROM users",
                "connection": "test_db",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["rows"]) == 2
        assert data["columns"] == ["id", "name"]


@pytest.mark.anyio
async def test_remote_query_with_filter():
    """Test remote query with WHERE clause."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE products (id INTEGER, name TEXT, price REAL, category TEXT)"
            )
        )
        conn.execute(
            text(
                """
            INSERT INTO products VALUES
                (1, 'Laptop', 999.99, 'Electronics'),
                (2, 'Mouse', 29.99, 'Electronics'),
                (3, 'Desk', 199.99, 'Furniture'),
                (4, 'Chair', 149.99, 'Furniture')
            """
            )
        )

    registry = ConnectionRegistry()
    registry.register("test_db", lambda req: engine)

    app = create_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        response = await client.post("/sessions")
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        # Query remote database with filter
        response = await client.post(
            f"/sessions/{session_id}/query",
            json={
                "query": """
                    SELECT * FROM products
                    WHERE category = 'Electronics'
                    VISUALISE price AS x, name AS y DRAW bar
                """,
                "connection": "test_db",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should only have 2 electronics items
        assert data["metadata"]["rows"] == 2
