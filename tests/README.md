# Testing Guide for Cultivation Project

## Setup

The project uses pytest for testing. To run tests, make sure you have pytest installed:

```bash
pip install pytest
```

## Project Structure

The project follows this structure for Python modules:

```
cultivation/           # Main package
├── __init__.py        # Makes cultivation a proper package
├── scripts/           # Scripts directory
│   ├── __init__.py    # Makes scripts a proper package
│   ├── running/       # Running-related scripts
│   │   ├── __init__.py
│   │   └── ...
│   ├── biology/
│   ├── software/
│   └── synergy/
└── ...
tests/                 # Test directory
├── conftest.py        # Pytest configuration
├── test_*.py          # Test files
└── ...
```

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_walk_utils.py
```

To run with verbose output:

```bash
pytest -v tests/test_walk_utils.py
```

## Import Structure

Tests should import modules from the `cultivation` package like this:

```python
from cultivation.scripts.running import walk_utils
```

The `conftest.py` file in the tests directory adds the project root to the Python path, so imports work correctly.

## Test Writing Guidelines

1. Create test files with names starting with `test_`.
2. Create test functions with names starting with `test_`.
3. Use descriptive names for test functions.
4. Include tests for edge cases and error conditions.
5. Use pytest fixtures for common setup.
6. Keep tests independent of each other.

## Common Issues

### Import Errors

If you encounter import errors like `ModuleNotFoundError: No module named 'cultivation'`, check:

1. That all necessary `__init__.py` files exist in the package structure
2. That the `conftest.py` file is correctly adding the project root to the Python path
3. That you're running pytest from the project root directory

### Function Signature Mismatches

When testing functions, make sure the test calls match the actual function signatures. If a function's parameters change, update the tests accordingly.
