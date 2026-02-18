"""Tests for ggsql Python bindings.

These tests focus on Python-specific logic:
- DataFrame conversion via narwhals
- Return type handling
- Two-stage API (reader.execute() -> render)

Rust logic (parsing, Vega-Lite generation) is tested in the Rust test suite.
"""

import json

import duckdb
import pytest
import polars as pl
import altair

import ggsql

# Optional dependency for ibis test
try:
    import ibis

    HAS_IBIS = hasattr(ibis, "duckdb")
except ImportError:
    HAS_IBIS = False


class TestValidate:
    """Tests for validate() function."""

    def test_valid_query_with_visualise(self):
        validated = ggsql.validate(
            "SELECT 1 AS x, 2 AS y VISUALISE DRAW point MAPPING x AS x, y AS y"
        )
        assert validated.has_visual()
        assert validated.valid()
        assert "SELECT" in validated.sql()
        assert "VISUALISE" in validated.visual()
        assert len(validated.errors()) == 0

    def test_valid_query_without_visualise(self):
        validated = ggsql.validate("SELECT 1 AS x, 2 AS y")
        assert not validated.has_visual()
        assert validated.valid()
        assert validated.sql() == "SELECT 1 AS x, 2 AS y"
        assert validated.visual() == ""

    def test_invalid_query_has_errors(self):
        validated = ggsql.validate("SELECT 1 VISUALISE DRAW invalid_geom")
        assert not validated.valid()
        assert len(validated.errors()) > 0

    def test_missing_required_aesthetic(self):
        # Point requires x and y, only providing x
        validated = ggsql.validate(
            "SELECT 1 AS x, 2 AS y VISUALISE DRAW point MAPPING x AS x"
        )
        assert not validated.valid()
        errors = validated.errors()
        assert len(errors) > 0
        assert any("y" in e["message"] for e in errors)


class TestDuckDBReader:
    """Tests for DuckDBReader class."""

    def test_create_in_memory(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        assert reader is not None

    def test_execute_simple_query(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        df = reader.execute_sql("SELECT 1 AS x, 2 AS y")
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (1, 2)
        assert list(df.columns) == ["x", "y"]

    def test_register_and_query(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        reader.register("my_data", df)

        result = reader.execute_sql("SELECT * FROM my_data WHERE x > 1")
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_supports_register(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        assert reader.supports_register() is True

    def test_invalid_connection_string(self):
        with pytest.raises(ValueError):
            ggsql.DuckDBReader("invalid://connection")


class TestVegaLiteWriter:
    """Tests for VegaLiteWriter class."""

    def test_create_writer(self):
        writer = ggsql.VegaLiteWriter()
        assert writer is not None


class TestExecute:
    """Tests for reader.execute() method."""

    def test_execute_simple_query(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        assert spec is not None
        assert spec.layer_count() == 1

    def test_execute_with_registered_data(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        reader.register("data", df)

        spec = reader.execute("SELECT * FROM data VISUALISE x, y DRAW point")
        assert spec.metadata()["rows"] == 3

    def test_execute_metadata(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute(
            "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y) "
            "VISUALISE x, y DRAW point",
        )

        metadata = spec.metadata()
        assert metadata["rows"] == 3
        assert "x" in metadata["columns"]
        assert "y" in metadata["columns"]
        assert metadata["layer_count"] == 1

    def test_execute_sql_accessor(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        assert "SELECT" in spec.sql()

    def test_execute_visual_accessor(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        assert "VISUALISE" in spec.visual()

    def test_execute_data_accessor(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        data = spec.data()
        assert isinstance(data, pl.DataFrame)
        assert data.shape == (1, 2)

    def test_execute_without_visualise_fails(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        with pytest.raises(ValueError):
            reader.execute("SELECT 1 AS x, 2 AS y")


class TestWriterRender:
    """Tests for VegaLiteWriter.render() method."""

    def test_render_to_vegalite(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        writer = ggsql.VegaLiteWriter()

        result = writer.render(spec)
        assert isinstance(result, str)

        spec_dict = json.loads(result)
        assert "$schema" in spec_dict
        assert "vega-lite" in spec_dict["$schema"]

    def test_render_contains_data(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        reader.register("data", df)

        spec = reader.execute("SELECT * FROM data VISUALISE x, y DRAW point")
        writer = ggsql.VegaLiteWriter()

        result = writer.render(spec)
        spec_dict = json.loads(result)
        # Data should be in the spec (either inline or in datasets)
        assert "data" in spec_dict or "datasets" in spec_dict

    def test_render_multi_layer(self):
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute(
            "SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y) "
            "VISUALISE "
            "DRAW point MAPPING x AS x, y AS y "
            "DRAW line MAPPING x AS x, y AS y",
        )
        writer = ggsql.VegaLiteWriter()

        result = writer.render(spec)
        spec_dict = json.loads(result)
        assert "layer" in spec_dict


class TestRenderAltairDataFrameConversion:
    """Tests for DataFrame handling in render_altair()."""

    def test_accepts_polars_dataframe(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
        assert isinstance(chart, altair.TopLevelMixin)

    def test_accepts_polars_lazyframe(self):
        lf = pl.LazyFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(lf, "VISUALISE x, y DRAW point")
        assert isinstance(chart, altair.TopLevelMixin)

    def test_accepts_narwhals_dataframe(self):
        import narwhals as nw

        pl_df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        nw_df = nw.from_native(pl_df)

        chart = ggsql.render_altair(nw_df, "VISUALISE x, y DRAW point")
        assert isinstance(chart, altair.TopLevelMixin)

    def test_accepts_pandas_dataframe(self):
        pd = pytest.importorskip("pandas")

        pd_df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(pd_df, "VISUALISE x, y DRAW point")
        assert isinstance(chart, altair.TopLevelMixin)

    def test_rejects_invalid_dataframe_type(self):
        with pytest.raises(TypeError, match="must be a narwhals DataFrame"):
            ggsql.render_altair({"x": [1, 2, 3]}, "VISUALISE x, y DRAW point")


class TestRenderAltairReturnType:
    """Tests for render_altair() return type."""

    def test_returns_altair_chart(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
        assert isinstance(chart, altair.TopLevelMixin)

    def test_chart_has_data(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
        spec = chart.to_dict()
        # Data should be embedded in datasets
        assert "datasets" in spec

    def test_chart_can_be_serialized(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
        # Should not raise
        json_str = chart.to_json()
        assert len(json_str) > 0


class TestRenderAltairChartTypeDetection:
    """Tests for correct Altair chart type detection based on spec structure."""

    def test_simple_chart_returns_layer_chart(self):
        """Simple DRAW specs produce LayerChart (ggsql always wraps in layer)."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")
        # ggsql wraps all charts in a layer
        assert isinstance(chart, altair.LayerChart)

    def test_layered_chart_can_round_trip(self):
        """LayerChart can be converted to dict and back."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        chart = ggsql.render_altair(df, "VISUALISE x, y DRAW point")

        # Convert to dict and back
        spec = chart.to_dict()
        assert "layer" in spec

        # Should be able to recreate from dict
        recreated = altair.LayerChart.from_dict(spec)
        assert isinstance(recreated, altair.LayerChart)

    def test_faceted_chart_returns_facet_chart(self):
        """FACET WRAP specs produce FacetChart."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "y": [10, 20, 30, 40, 50, 60],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        # Need validate=False because ggsql produces v6 specs
        chart = ggsql.render_altair(
            df, "VISUALISE x, y FACET WRAP group DRAW point", validate=False
        )
        assert isinstance(chart, altair.FacetChart)

    def test_faceted_chart_can_round_trip(self):
        """FacetChart can be converted to dict and back."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "y": [10, 20, 30, 40, 50, 60],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        chart = ggsql.render_altair(
            df, "VISUALISE x, y FACET WRAP group DRAW point", validate=False
        )

        # Convert to dict (skip validation for ggsql specs)
        spec = chart.to_dict(validate=False)
        assert "facet" in spec or "spec" in spec

        # Should be able to recreate from dict (with validation disabled)
        recreated = altair.FacetChart.from_dict(spec, validate=False)
        assert isinstance(recreated, altair.FacetChart)

    def test_chart_with_color_encoding(self):
        """Charts with color encoding still return correct type."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
                "category": ["A", "B", "A", "B"],
            }
        )
        chart = ggsql.render_altair(df, "VISUALISE x, y, category AS color DRAW point")
        # Should still be a LayerChart (ggsql wraps in layer)
        assert isinstance(chart, altair.LayerChart)


class TestRenderAltairErrorHandling:
    """Tests for error handling in render_altair()."""

    def test_invalid_viz_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        with pytest.raises(ValueError):
            ggsql.render_altair(df, "NOT VALID SYNTAX")


class TestTwoStageAPIIntegration:
    """Integration tests for the two-stage reader.execute() -> render API."""

    def test_end_to_end_workflow(self):
        """Complete workflow: create reader, register data, execute, render."""
        # Create reader
        reader = ggsql.DuckDBReader("duckdb://memory")

        # Register data
        df = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "value": [10, 20, 30],
                "region": ["North", "South", "North"],
            }
        )
        reader.register("sales", df)

        # Execute visualization
        spec = reader.execute(
            "SELECT * FROM sales VISUALISE date AS x, value AS y, region AS color DRAW line",
        )

        # Verify metadata
        assert spec.metadata()["rows"] == 3
        assert spec.layer_count() == 1

        # Render to Vega-Lite
        writer = ggsql.VegaLiteWriter()
        result = writer.render(spec)

        # Verify output
        spec_dict = json.loads(result)
        assert "$schema" in spec_dict
        assert "line" in json.dumps(spec_dict)

    def test_can_introspect_spec(self):
        """Test all introspection methods on Spec."""
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")

        # All these should work without error
        assert spec.sql() is not None
        assert spec.visual() is not None
        assert spec.layer_count() >= 1
        assert spec.metadata() is not None
        assert spec.data() is not None
        assert spec.warnings() is not None

        # Layer-specific accessors (may return None)
        _ = spec.layer_data(0)
        _ = spec.stat_data(0)
        _ = spec.layer_sql(0)
        _ = spec.stat_sql(0)


class TestCustomReader:
    """Tests for custom Python reader support."""

    def test_simple_custom_reader(self):
        """Custom reader with execute_sql() method works."""

        class SimpleReader:
            def __init__(self):
                self.conn = duckdb.connect()
                self.conn.execute(
                    "CREATE TABLE data AS SELECT * FROM ("
                    "VALUES (1, 10), (2, 20), (3, 30)"
                    ") AS t(x, y)"
                )

            def execute_sql(self, sql: str) -> pl.DataFrame:
                return self.conn.execute(sql).pl()

        reader = SimpleReader()
        spec = ggsql.execute("SELECT * FROM data VISUALISE x, y DRAW point", reader)
        assert spec.metadata()["rows"] == 3

    def test_custom_reader_with_register(self):
        """Custom reader with register() support."""

        class RegisterReader:
            def __init__(self):
                self.conn = duckdb.connect()

            def execute_sql(self, sql: str) -> pl.DataFrame:
                return self.conn.execute(sql).pl()

            def supports_register(self) -> bool:
                return True

            def register(self, name: str, df: pl.DataFrame) -> None:
                self.conn.register(name, df)

        reader = RegisterReader()
        spec = ggsql.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point", reader)
        assert spec is not None

    def test_custom_reader_error_handling(self):
        """Custom reader errors are propagated."""

        class ErrorReader:
            def execute_sql(self, sql: str) -> pl.DataFrame:
                raise ValueError("Custom reader error")

        reader = ErrorReader()
        with pytest.raises(ValueError, match="Custom reader error"):
            ggsql.execute("SELECT 1 VISUALISE x, y DRAW point", reader)

    def test_custom_reader_wrong_return_type(self):
        """Custom reader returning wrong type raises TypeError."""

        class WrongTypeReader:
            def execute_sql(self, sql: str):
                return {"x": [1, 2, 3]}  # dict, not DataFrame

        reader = WrongTypeReader()
        with pytest.raises((ValueError, TypeError)):
            ggsql.execute("SELECT 1 VISUALISE x, y DRAW point", reader)

    def test_native_reader_fast_path(self):
        """Native DuckDBReader still works (fast path)."""
        reader = ggsql.DuckDBReader("duckdb://memory")
        spec = reader.execute("SELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point")
        assert spec.metadata()["rows"] == 1

    def test_custom_reader_can_render(self):
        """Custom reader result can be rendered to Vega-Lite."""

        class DuckDBBackedReader:
            def __init__(self):
                self.conn = duckdb.connect()
                self.conn.execute(
                    "CREATE TABLE data AS SELECT * FROM ("
                    "VALUES (1, 10, 'A'), (2, 40, 'B'), (3, 20, 'A'), "
                    "(4, 50, 'B'), (5, 30, 'A')"
                    ") AS t(x, y, category)"
                )

            def execute_sql(self, sql: str) -> pl.DataFrame:
                return self.conn.execute(sql).pl()

        reader = DuckDBBackedReader()
        spec = ggsql.execute(
            "SELECT * FROM data VISUALISE x, y, category AS color DRAW point",
            reader,
        )

        writer = ggsql.VegaLiteWriter()
        result = writer.render(spec)

        spec_dict = json.loads(result)
        assert "$schema" in spec_dict
        assert "vega-lite" in spec_dict["$schema"]

    def test_custom_reader_execute_sql_called(self):
        """Verify execute_sql() is called on the custom reader."""

        class RecordingReader:
            def __init__(self):
                self.conn = duckdb.connect()
                self.conn.execute(
                    "CREATE TABLE data AS SELECT * FROM ("
                    "VALUES (1, 2)) AS t(x, y)"
                )
                self.execute_calls = []

            def execute_sql(self, sql: str) -> pl.DataFrame:
                self.execute_calls.append(sql)
                return self.conn.execute(sql).pl()

        reader = RecordingReader()
        ggsql.execute(
            "SELECT * FROM data VISUALISE x, y DRAW point",
            reader,
        )

        # execute_sql() should have been called at least once
        assert len(reader.execute_calls) > 0
        # All calls should be valid SQL strings
        assert all(isinstance(sql, str) for sql in reader.execute_calls)

    @pytest.mark.skipif(not HAS_IBIS, reason="ibis not installed")
    def test_custom_reader_ibis(self):
        """Test custom reader using ibis as backend."""

        class IbisReader:
            def __init__(self):
                self.con = ibis.duckdb.connect()

            def execute_sql(self, sql: str) -> pl.DataFrame:
                return self.con.con.execute(sql).pl()

            def supports_register(self) -> bool:
                return True

            def register(self, name: str, df: pl.DataFrame) -> None:
                self.con.create_table(name, df.to_arrow(), overwrite=True)

            def unregister(self, name: str) -> None:
                self.con.drop_table(name)

        reader = IbisReader()
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        reader.register("mydata", df)

        spec = ggsql.execute(
            "SELECT * FROM mydata VISUALISE x, y DRAW point",
            reader,
        )

        assert spec.metadata()["rows"] == 3
        writer = ggsql.VegaLiteWriter()
        json_output = writer.render(spec)
        assert "point" in json_output
