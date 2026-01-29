"""Tests for ggsql Python bindings.

These tests focus on Python-specific logic:
- DataFrame conversion via narwhals
- Return type handling

Rust logic (parsing, Vega-Lite generation) is tested in the Rust test suite.
"""

import pytest
import polars as pl
import altair

import ggsql


class TestSplitQuery:
    """Tests for split_query() function."""

    def test_splits_sql_and_visualise(self):
        sql, viz = ggsql.split_query(
            "SELECT x, y FROM data VISUALISE x, y DRAW point"
        )
        assert "SELECT" in sql
        assert "VISUALISE" in viz

    def test_no_visualise_returns_empty_viz(self):
        sql, viz = ggsql.split_query("SELECT * FROM data")
        assert sql == "SELECT * FROM data"
        assert viz == ""


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
        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        # Need validate=False because ggsql produces v6 specs
        chart = ggsql.render_altair(df, "VISUALISE x, y FACET WRAP group DRAW point", validate=False)
        assert isinstance(chart, altair.FacetChart)

    def test_faceted_chart_can_round_trip(self):
        """FacetChart can be converted to dict and back."""
        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        chart = ggsql.render_altair(df, "VISUALISE x, y FACET WRAP group DRAW point", validate=False)

        # Convert to dict (skip validation for ggsql specs)
        spec = chart.to_dict(validate=False)
        assert "facet" in spec or "spec" in spec

        # Should be able to recreate from dict (with validation disabled)
        recreated = altair.FacetChart.from_dict(spec, validate=False)
        assert isinstance(recreated, altair.FacetChart)

    def test_chart_with_color_encoding(self):
        """Charts with color encoding still return correct type."""
        df = pl.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "category": ["A", "B", "A", "B"],
        })
        chart = ggsql.render_altair(df, "VISUALISE x, y, category AS color DRAW point")
        # Should still be a LayerChart (ggsql wraps in layer)
        assert isinstance(chart, altair.LayerChart)


class TestRenderAltairErrorHandling:
    """Tests for error handling in render_altair()."""

    def test_invalid_viz_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        with pytest.raises(ValueError):
            ggsql.render_altair(df, "NOT VALID SYNTAX")
