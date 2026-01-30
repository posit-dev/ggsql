"""ggsql - SQL extension for declarative data visualization."""

from __future__ import annotations

from ggsql import readers, types, writers
from ggsql._ggsql import validate

__all__ = [
    # Submodules
    "readers",
    "writers",
    "types",
    # Functions
    "validate",
]
__version__ = "0.1.0"
version_info = (0, 1, 0)
