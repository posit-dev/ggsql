"""
Tree-sitter bindings for ggsql

This package provides Python bindings for the tree-sitter-ggsql grammar.
ggsql is a SQL extension for declarative data visualization based on the Grammar of Graphics.
"""

from tree_sitter import Language

try:
    from .binding import language
except ImportError:
    import os
    import tree_sitter

    # Try to load the language from the compiled shared library
    LIB_PATH = os.path.join(os.path.dirname(__file__), "binding.so")
    if os.path.exists(LIB_PATH):
        language = Language(LIB_PATH, "ggsql")
    else:
        # Fallback: try to compile from source
        import subprocess
        import tempfile

        # Get the path to the grammar source
        grammar_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")

        try:
            # Use tree-sitter to compile the language
            language = tree_sitter.Language.build_library(
                LIB_PATH,
                [grammar_path]
            )
        except Exception as e:
            raise ImportError(f"Could not load tree-sitter-ggsql language: {e}")

__version__ = "0.1.0"
__all__ = ["language"]
