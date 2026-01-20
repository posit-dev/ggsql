# ggsql-jupyter Test Suite

Comprehensive testing infrastructure for the ggsql Jupyter kernel, validating Jupyter messaging protocol compliance and functional correctness.

## Overview

### Test Layers

1. **Integration Tests** (`test_integration.py`)

   - Real kernel execution with `jupyter_client`
   - Message flow validation
   - Multi-cell execution with persistent state
   - Error handling verification

2. **Compliance Tests** (`test_compliance.py`)

   - Official `jupyter_kernel_test` suite
   - Validates protocol compliance against spec
   - Industry-standard kernel testing

## Setup

### 1. Install Python Dependencies

```bash
# From ggsql-jupyter/tests/ directory
pip install -r requirements.txt
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the Kernel

```bash
# From repository root
cargo build --bin ggsql-jupyter
```

## Running Tests

### Integration Tests (jupyter_client)

Run integration tests using real Jupyter client:

```bash
# From ggsql-jupyter/tests/ directory
pytest test_integration.py -v

# Run specific test class
pytest test_integration.py::TestExecution -v

# Run specific test
pytest test_integration.py::TestExecution::test_simple_sql_execution -v

# Run with detailed output
pytest test_integration.py -v -s
```

### Compliance Tests (jupyter_kernel_test)

Run official Jupyter kernel compliance tests:

```bash
# From ggsql-jupyter/tests/ directory
pytest test_compliance.py -v

# Note: This will install the kernel spec temporarily
```
