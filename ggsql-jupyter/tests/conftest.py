"""Shared fixtures for the ggsql-jupyter protocol test suite.

`test_integration.py` (pytest) and `test_compliance.py` (unittest-style
setup_module) both need the kernel binary, so the build lives here once
rather than duplicated in each file. CI's own build step already generates
the grammar and compiles the binary before pytest runs; calling this again
from a fixture is a cheap up-to-date check, not a rebuild.
"""

import subprocess
from pathlib import Path

import pytest


def build_kernel_binary() -> str:
    """Build ggsql-jupyter and return the path to its binary."""
    repo_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        ["cargo", "build", "--bin", "ggsql-jupyter"],
        cwd=repo_root / "ggsql-jupyter",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build kernel: {result.stderr}")

    binary_path = repo_root / "target" / "debug" / "ggsql-jupyter"
    if not binary_path.exists():
        raise RuntimeError(f"Kernel binary not found at {binary_path}")

    return str(binary_path)


@pytest.fixture(scope="session")
def kernel_binary() -> str:
    """Build and return path to ggsql-jupyter binary, once per session."""
    return build_kernel_binary()
