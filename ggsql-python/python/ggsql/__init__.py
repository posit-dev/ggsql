from __future__ import annotations

import json
from typing import Any, Union

import altair
import narwhals as nw
from narwhals.typing import IntoFrame

from ggsql._ggsql import (
    DuckDBReader,
    VegaLiteWriter,
    Validated,
    Spec,
    validate,
    execute,
)

__all__ = [
    # Classes
    "DuckDBReader",
    "VegaLiteWriter",
    "Validated",
    "Spec",
    # Functions
    "validate",
    "execute",
    "render_altair",
]
__version__ = "0.1.3"

# Type alias for any Altair chart type
AltairChart = Union[
    altair.Chart,
    altair.LayerChart,
    altair.FacetChart,
    altair.ConcatChart,
    altair.HConcatChart,
    altair.VConcatChart,
    altair.RepeatChart,
]


def render_altair(
    df: IntoFrame,
    viz: str,
    **kwargs: Any,
) -> AltairChart:
    """Render a DataFrame with a VISUALISE spec to an Altair chart.

    Parameters
    ----------
    df
        Data to visualize. Accepts polars, pandas, or any narwhals-compatible
        DataFrame. LazyFrames are collected automatically.
    viz
        VISUALISE spec string (e.g., "VISUALISE x, y DRAW point")
    **kwargs
        Additional keyword arguments passed to `from_json()`.
        Common options include `validate=False` to skip schema validation.

    Returns
    -------
    AltairChart
        An Altair chart object (Chart, LayerChart, FacetChart, etc.).
    """
    df = nw.from_native(df, pass_through=True)

    if isinstance(df, nw.LazyFrame):
        df = df.collect()

    if not isinstance(df, nw.DataFrame):
        raise TypeError("df must be a narwhals DataFrame or compatible type")

    pl_df = df.to_polars()

    # Create temporary reader and register data
    reader = DuckDBReader("duckdb://memory")
    reader.register("__data__", pl_df)

    # Build full query: SELECT * FROM __data__ + VISUALISE clause
    query = f"SELECT * FROM __data__ {viz}"

    # Execute and render
    spec = reader.execute(query)
    writer = VegaLiteWriter()
    vegalite_json = writer.render(spec)

    # Parse to determine the correct Altair class
    spec = json.loads(vegalite_json)

    # Determine the correct Altair class based on spec structure
    if "layer" in spec:
        return altair.LayerChart.from_json(vegalite_json, **kwargs)
    elif "facet" in spec or "spec" in spec:
        return altair.FacetChart.from_json(vegalite_json, **kwargs)
    elif "concat" in spec:
        return altair.ConcatChart.from_json(vegalite_json, **kwargs)
    elif "hconcat" in spec:
        return altair.HConcatChart.from_json(vegalite_json, **kwargs)
    elif "vconcat" in spec:
        return altair.VConcatChart.from_json(vegalite_json, **kwargs)
    elif "repeat" in spec:
        return altair.RepeatChart.from_json(vegalite_json, **kwargs)
    else:
        return altair.Chart.from_json(vegalite_json, **kwargs)
