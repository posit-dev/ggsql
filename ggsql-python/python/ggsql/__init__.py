from __future__ import annotations
from typing import Literal, Any

import polars as pl

from ggsql._ggsql import split_query, render as _render

__all__ = ["split_query", "render"]
__version__ = "0.1.0"


def _to_polars(df: Any) -> pl.DataFrame:
    """Convert various DataFrame types to polars DataFrame."""
    # Already a polars DataFrame
    if isinstance(df, pl.DataFrame):
        return df

    # Polars LazyFrame
    if isinstance(df, pl.LazyFrame):
        return df.collect()

    # Narwhals DataFrame - convert to native then to polars
    if hasattr(df, "to_native"):
        native = df.to_native()
        return _to_polars(native)

    # Pandas DataFrame
    if hasattr(df, "to_pandas"):
        # It's likely a pandas-like object, try polars conversion
        pass

    # Try polars.from_pandas as last resort for pandas DataFrames
    if type(df).__name__ == "DataFrame" and type(df).__module__.startswith("pandas"):
        return pl.from_pandas(df)

    raise TypeError(
        f"Expected polars DataFrame, LazyFrame, or narwhals DataFrame, got {type(df)}"
    )


def render(
    df: "pl.DataFrame | pl.LazyFrame | Any",
    viz: str,
    *,
    writer: Literal["vegalite"] = "vegalite",
) -> str:
    """Render a DataFrame with a VISUALISE spec.

    Parameters
    ----------
    df : polars.DataFrame | polars.LazyFrame | narwhals.DataFrame
        Data to visualize. LazyFrames are collected automatically.
        Narwhals DataFrames are converted to polars.
    viz : str
        VISUALISE spec string (e.g., "VISUALISE x, y DRAW point")
    writer : Literal["vegalite"]
        Output format. Currently only "vegalite" supported.

    Returns
    -------
    str
        Vega-Lite JSON specification.
    """
    df = _to_polars(df)
    return _render(df, viz, writer=writer)
