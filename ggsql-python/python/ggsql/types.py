"""Type classes and exceptions for ggsql."""

from ggsql._ggsql import (
    GgsqlError,
    NoVisualiseError,
    ParseError,
    Prepared,
    ReaderError,
    Validated,
    ValidationError,
    WriterError,
)

__all__ = [
    # Base exception
    "GgsqlError",
    # Specific exceptions
    "ParseError",
    "ValidationError",
    "ReaderError",
    "WriterError",
    "NoVisualiseError",
    # Type classes
    "Prepared",
    "Validated",
]
