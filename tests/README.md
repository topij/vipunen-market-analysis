# Vipunen Project Tests

This directory contains test files for the Vipunen project.

## Test Structure

- `conftest.py`: Contains pytest fixtures and setup/teardown functions
- `test_data_processor.py`: Tests for the data_processor module
- `test_market_share_analyzer.py`: Tests for the market_share_analyzer module
- `test_qualification_analyzer.py`: Tests for the qualification_analyzer module
- `test_volume_plots.py`: Tests for the volume_plots module
- `test_analyze_education_market.py`: Tests for the main analysis script

## Running Tests

To run the tests, execute the following command from the project root:

```bash
python -m pytest
```

To run specific tests:

```bash
python -m pytest tests/test_data_processor.py
```

To run tests with a specific marker:

```bash
python -m pytest -m unit
```

## Creating Tests

When creating new tests:

1. Create a file named `test_*.py` in the `tests` directory
2. Import the module under test
3. Create test functions with names starting with `test_`
4. Use assertions to verify expected behavior

Example:

```python
import pytest
from src.vipunen.module import function_to_test

def test_function_behavior():
    # Arrange
    input_data = ...
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_result
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`. These include:

- `sample_education_data`: Sample education data for testing
- `sample_market_data`: Sample market data for testing
- `sample_volume_data`: Sample volume data for visualization testing
- `sample_volumes_by_qualification`: Sample qualification volume data for testing
- `mock_file_utils`: Mock for FileUtils

You can use these fixtures in your tests by adding them as parameters to your test functions. 