"""Writer classes for ggsql."""

from __future__ import annotations

import json
from typing import Any, Union

import altair

from ggsql._ggsql import _VegaLiteWriter, Prepared

__all__ = ["VegaLite", "AltairChart"]

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


class VegaLite:
    """Vega-Lite JSON output writer.

    Converts prepared visualization specifications to Vega-Lite v6 JSON.

    Examples
    --------
    >>> writer = ggsql.writers.VegaLite()
    >>> json_str = writer.render_json(spec)
    >>> chart = writer.render_chart(spec)
    """

    def __init__(self) -> None:
        """Create a new Vega-Lite writer."""
        self._inner = _VegaLiteWriter()

    def __repr__(self) -> str:
        return "<VegaLite>"

    def render_json(self, spec: Prepared) -> str:
        """Render a prepared visualization to Vega-Lite JSON.

        Parameters
        ----------
        spec : Prepared
            The prepared visualization (from reader.execute()).

        Returns
        -------
        str
            The Vega-Lite JSON specification as a string.

        Raises
        ------
        WriterError
            If rendering fails.
        """
        return self._inner.render(spec)

    def render_chart(self, spec: Prepared, **kwargs: Any) -> AltairChart:
        """Render a prepared visualization to an Altair chart object.

        Parameters
        ----------
        spec : Prepared
            The prepared visualization (from reader.execute()).
        **kwargs
            Additional keyword arguments passed to Altair's `from_json()`.
            Common options include `validate=False` to skip schema validation.
            Note: `validate=False` is used by default since ggsql produces
            Vega-Lite v6 specs.

        Returns
        -------
        AltairChart
            An Altair chart object (Chart, LayerChart, FacetChart, etc.)
            appropriate for the visualization structure.

        Raises
        ------
        WriterError
            If rendering fails.
        """
        json_str = self._inner.render(spec)

        # Default to validate=False since ggsql produces v6 specs
        if "validate" not in kwargs:
            kwargs["validate"] = False

        # Parse the JSON to determine the chart type
        spec_dict = json.loads(json_str)

        # Determine the correct Altair class based on spec structure
        if "layer" in spec_dict:
            return altair.LayerChart.from_json(json_str, **kwargs)
        elif "facet" in spec_dict or "spec" in spec_dict:
            return altair.FacetChart.from_json(json_str, **kwargs)
        elif "concat" in spec_dict:
            return altair.ConcatChart.from_json(json_str, **kwargs)
        elif "hconcat" in spec_dict:
            return altair.HConcatChart.from_json(json_str, **kwargs)
        elif "vconcat" in spec_dict:
            return altair.VConcatChart.from_json(json_str, **kwargs)
        elif "repeat" in spec_dict:
            return altair.RepeatChart.from_json(json_str, **kwargs)
        else:
            return altair.Chart.from_json(json_str, **kwargs)
